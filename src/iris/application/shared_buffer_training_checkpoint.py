"""Atomic arm-C teacher checkpoint v3 with exact shared-buffer identity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from iris.application.arm_profiles import arm_execution_profile_set_sha256_v4
from iris.application.errors import IrisTeacherCheckpointError
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointRestoreResult,
    IrisTeacherCheckpointWriteResult,
    _cleanup_temporary_directory,
    _dtype_name,
    _optimizer_config_record,
    _owned_cpu_tensor,
    _reject_duplicate_keys,
    _snapshots_equal,
    _software_record,
    _write_exact_file,
)
from iris.artifacts.checkpoint_v1 import (
    CheckpointFloatDTypeV1,
    TeacherAdamWParameterStateV1,
    TeacherBackboneRuntimeV1,
    TeacherCheckpointOptimizerV1,
    TeacherCheckpointStateFileV1,
    TeacherCheckpointTensorV1,
)
from iris.artifacts.checkpoint_v3 import (
    IrisSharedBufferTeacherCheckpointEnvelopeV3,
    SharedBufferTeacherCheckpointModelIdentityV3,
    SharedBufferTeacherCheckpointPayloadV3,
    SharedBufferTeacherCheckpointSchemaV3,
)
from iris.schemas import BackboneConfig, CognitiveConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.training import (
    IrisAdamWConfig,
    IrisAdamWParameterSnapshot,
    IrisSharedBufferTeacherCheckpointProvenanceV1,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerSnapshot,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)

_PAYLOAD_DOMAIN = "iris/teacher-checkpoint/v3/payload"
_SAFETENSORS_METADATA = {"format": "iris.teacher-checkpoint.v3"}
_MANIFEST_FILENAME = "manifest.json"
_STATE_FILENAME: Literal["state.safetensors"] = "state.safetensors"
_PROTOCOL_PAYLOAD_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_PROFILE_SET_V4_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_POLICY_SHA256: Literal["56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"] = (
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
)
_TEACHER_GRAPH: Literal["append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"] = (
    "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
)


@dataclass(frozen=True, slots=True)
class IrisSharedBufferTeacherCheckpointBundleV3:
    """One verified arm-C manifest paired with immutable safetensors bytes."""

    manifest: IrisSharedBufferTeacherCheckpointEnvelopeV3
    state_bytes: bytes


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    manifest: IrisSharedBufferTeacherCheckpointEnvelopeV3
    tensors: dict[str, torch.Tensor]
    manifest_bytes: bytes
    state_bytes: bytes


def _tensor_domain(descriptor: TeacherCheckpointTensorV1) -> str:
    if descriptor.owner == "sidecar":
        return f"iris/teacher-checkpoint/v3/sidecar/{descriptor.state_name}"
    parameter_name = descriptor.parameter_name
    if parameter_name is None:
        message = "AdamW checkpoint descriptor is missing parameter_name"
        raise ValueError(message)
    return f"iris/teacher-checkpoint/v3/adamw/{parameter_name}/{descriptor.state_name}"


def _tensor_descriptor(
    *,
    key: str,
    owner: Literal["sidecar", "adamw"],
    state_name: str,
    parameter_name: str | None,
    tensor: torch.Tensor,
) -> TeacherCheckpointTensorV1:
    descriptor = TeacherCheckpointTensorV1(
        key=key,
        owner=owner,
        parameter_name=parameter_name,
        state_name=state_name,
        shape=tuple(tensor.shape),
        dtype=_dtype_name(tensor.dtype),
        byte_length=tensor.numel() * tensor.element_size(),
        sha256="0" * 64,
    )
    return descriptor.model_copy(
        update={"sha256": tensor_sha256(tensor, domain=_tensor_domain(descriptor))}
    )


def _parse_model_provenance(
    provenance: IrisSharedBufferTeacherCheckpointProvenanceV1,
) -> tuple[BackboneConfig, CognitiveConfig]:
    if type(provenance) is not IrisSharedBufferTeacherCheckpointProvenanceV1:
        message = "Arm-C teacher model returned invalid checkpoint provenance"
        raise TypeError(message)
    try:
        backbone = BackboneConfig.model_validate_json(provenance.backbone_configuration_json)
        cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    except Exception as error:
        message = "Arm-C checkpoint configuration provenance is invalid"
        raise ValueError(message) from error
    modules = cognitive.num_modules
    if (
        provenance.arm_id != "C"
        or provenance.teacher_graph != _TEACHER_GRAPH
        or shared_buffer_policy_sha256_v1() != _POLICY_SHA256
        or provenance.shared_buffer_policy_sha256 != _POLICY_SHA256
        or provenance.cognitive_cycles != 3
        or provenance.pre_update_read_count != 3
        or provenance.final_output_read_count != 1
        or provenance.entries_per_cycle != modules
        or provenance.final_entry_count != modules * 3
        or provenance.latent_token_count != modules
        or provenance.component_names
        != IrisSharedBufferTeacherForcingModel._CANONICAL_COMPONENT_NAMES
    ):
        message = "Arm-C checkpoint graph provenance is inconsistent"
        raise ValueError(message)
    return backbone, cognitive


def _clone_model_state(
    model: IrisSharedBufferTeacherForcingModel,
    *,
    provenance: IrisSharedBufferTeacherCheckpointProvenanceV1,
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    try:
        state = model.state_dict()
    except Exception as error:
        message = "Failed to enumerate Arm-C sidecar checkpoint state"
        raise ValueError(message) from error
    names = tuple(state)
    if not names or len(set(names)) != len(names):
        message = "Arm-C sidecar checkpoint state names are empty or duplicated"
        raise ValueError(message)
    expected_device = torch.device(provenance.device)
    expected_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(provenance.dtype)
    if expected_dtype is None:
        message = "Arm-C checkpoint provenance has an unsupported dtype"
        raise ValueError(message)
    owned: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if type(name) is not str or not name or "\0" in name:
            message = "Arm-C sidecar state names must be nonempty without NUL"
            raise ValueError(message)
        if type(tensor) is not torch.Tensor:
            message = f"Arm-C sidecar state {name!r} is not an exact tensor"
            raise TypeError(message)
        if tensor.device != expected_device or tensor.dtype != expected_dtype:
            message = f"Arm-C sidecar state {name!r} disagrees with placement"
            raise ValueError(message)
        owned[name] = _owned_cpu_tensor(
            tensor,
            label=f"arm-C sidecar state {name!r}",
        )
    return names, owned


def build_shared_buffer_teacher_checkpoint_v3(
    *,
    model: IrisSharedBufferTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisSharedBufferTeacherCheckpointBundleV3:
    """Capture one exact arm-C graph-and-AdamW checkpoint in memory."""
    try:
        if type(model) is not IrisSharedBufferTeacherForcingModel:
            message = "checkpoint v3 model must be exactly arm-C teacher model"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own the arm-C teacher model"
            raise TypeError(message)
        profile_sha256 = arm_execution_profile_set_sha256_v4()
        if profile_sha256 != _PROFILE_SET_V4_SHA256:
            message = "Canonical execution profile set v4 identity drifted"
            raise ValueError(message)
        optimizer_snapshot = optimizer.checkpoint_snapshot()
        provenance = model.checkpoint_provenance
        backbone, cognitive = _parse_model_provenance(provenance)
        state_names, sidecar_tensors = _clone_model_state(
            model,
            provenance=provenance,
        )

        tensors: dict[str, torch.Tensor] = {}
        descriptors: dict[str, TeacherCheckpointTensorV1] = {}
        for index, state_name in enumerate(state_names):
            key = f"sidecar.{index:08d}"
            tensor = sidecar_tensors[state_name]
            tensors[key] = tensor
            descriptors[key] = _tensor_descriptor(
                key=key,
                owner="sidecar",
                state_name=state_name,
                parameter_name=None,
                tensor=tensor,
            )
        parameter_indices = {
            name: index for index, name in enumerate(optimizer_snapshot.parameter_names)
        }
        for state in optimizer_snapshot.parameter_states:
            parameter_index = parameter_indices[state.parameter_name]
            for state_name, source in (
                ("exp_avg", state.exp_avg),
                ("exp_avg_sq", state.exp_avg_sq),
            ):
                key = f"adamw.{parameter_index:08d}.{state_name}"
                tensor = _owned_cpu_tensor(
                    source,
                    label=f"arm-C AdamW {state_name} for {state.parameter_name!r}",
                )
                tensors[key] = tensor
                descriptors[key] = _tensor_descriptor(
                    key=key,
                    owner="adamw",
                    state_name=state_name,
                    parameter_name=state.parameter_name,
                    tensor=tensor,
                )
        ordered_tensors = dict(sorted(tensors.items()))
        state_bytes = save_safetensors(
            ordered_tensors,
            metadata=_SAFETENSORS_METADATA,
        )
        runtime = TeacherBackboneRuntimeV1(
            hidden_size=provenance.hidden_size,
            num_hidden_layers=provenance.num_hidden_layers,
            vocabulary_size=provenance.vocabulary_size,
            max_position_embeddings=provenance.max_position_embeddings,
            max_prompt_tokens=provenance.max_prompt_tokens,
            max_target_tokens=provenance.max_target_tokens,
            device=provenance.device,
            dtype=cast(CheckpointFloatDTypeV1, provenance.dtype),
        )
        payload = SharedBufferTeacherCheckpointPayloadV3(
            model=SharedBufferTeacherCheckpointModelIdentityV3(
                arm_id="C",
                arm_kind="llm_modules_unlimited_shared_buffer",
                teacher_graph=_TEACHER_GRAPH,
                execution_profile_set_sha256=_PROFILE_SET_V4_SHA256,
                shared_buffer_policy_sha256=_POLICY_SHA256,
                protocol_declaration_payload_sha256=_PROTOCOL_PAYLOAD_SHA256,
                backbone=backbone,
                cognitive=cognitive,
                cognitive_cycles=3,
                pre_update_read_count=3,
                final_output_read_count=1,
                entries_per_cycle=provenance.entries_per_cycle,
                final_entry_count=provenance.final_entry_count,
                latent_token_count=provenance.latent_token_count,
                component_names=provenance.component_names,
                runtime=runtime,
                state_names=state_names,
                sha256=provenance.sidecar_state_sha256,
            ),
            optimizer=TeacherCheckpointOptimizerV1(
                optimizer_type="torch.optim.AdamW",
                completed_steps=optimizer_snapshot.completed_steps,
                config=_optimizer_config_record(optimizer_snapshot.config),
                parameter_names=optimizer_snapshot.parameter_names,
                parameter_states=tuple(
                    TeacherAdamWParameterStateV1(
                        parameter_name=state.parameter_name,
                        step=state.step,
                    )
                    for state in optimizer_snapshot.parameter_states
                ),
            ),
            state_file=TeacherCheckpointStateFileV1(
                filename=_STATE_FILENAME,
                size_bytes=len(state_bytes),
                tensor_count=len(ordered_tensors),
                sha256=bytes_sha256(state_bytes),
            ),
            tensors=tuple(descriptors[key] for key in sorted(descriptors)),
            software=_software_record(),
        )
        manifest = IrisSharedBufferTeacherCheckpointEnvelopeV3(
            schema_identity=SharedBufferTeacherCheckpointSchemaV3(
                artifact_type="iris.teacher_checkpoint",
                version=3,
                canonicalization="iris-json-v1",
                tensor_encoding="safetensors-v1",
                hash_algorithm="sha256",
            ),
            payload=payload,
            sha256=domain_sha256(
                _PAYLOAD_DOMAIN,
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
        )
        if model.checkpoint_provenance != provenance:
            message = "Arm-C sidecar provenance changed during checkpoint capture"
            raise ValueError(message)
        if not _snapshots_equal(optimizer.checkpoint_snapshot(), optimizer_snapshot):
            message = "Arm-C AdamW state changed during checkpoint capture"
            raise ValueError(message)
        verified = _validate_bundle_integrity(
            IrisSharedBufferTeacherCheckpointBundleV3(
                manifest=manifest,
                state_bytes=state_bytes,
            )
        )
        return IrisSharedBufferTeacherCheckpointBundleV3(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-C teacher checkpoint v3 could not be built"
        raise IrisTeacherCheckpointError(message) from error


def _validate_bundle_integrity(
    bundle: IrisSharedBufferTeacherCheckpointBundleV3,
) -> _VerifiedBundle:
    if type(bundle) is not IrisSharedBufferTeacherCheckpointBundleV3:
        message = "checkpoint bundle must be exactly arm-C checkpoint v3"
        raise TypeError(message)
    if type(bundle.manifest) is not IrisSharedBufferTeacherCheckpointEnvelopeV3:
        message = "arm-C checkpoint bundle manifest has an invalid type"
        raise TypeError(message)
    if type(bundle.state_bytes) is not bytes or not bundle.state_bytes:
        message = "arm-C checkpoint state_bytes must be nonempty exact bytes"
        raise TypeError(message)
    manifest = IrisSharedBufferTeacherCheckpointEnvelopeV3.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    profile_sha256 = arm_execution_profile_set_sha256_v4()
    if (
        profile_sha256 != _PROFILE_SET_V4_SHA256
        or manifest.payload.model.execution_profile_set_sha256 != _PROFILE_SET_V4_SHA256
    ):
        message = "arm-C checkpoint is not bound to execution profile set v4"
        raise ValueError(message)
    payload_bytes = canonical_json_bytes(manifest.payload.model_dump(mode="json"))
    if manifest.sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "arm-C checkpoint payload SHA-256 is inconsistent"
        raise ValueError(message)
    state_bytes = bundle.state_bytes
    if manifest.payload.state_file.size_bytes != len(
        state_bytes
    ) or manifest.payload.state_file.sha256 != bytes_sha256(state_bytes):
        message = "arm-C checkpoint safetensors identity is inconsistent"
        raise ValueError(message)
    try:
        tensors = load_safetensors(state_bytes)
    except Exception as error:
        message = "arm-C checkpoint safetensors bytes cannot be decoded"
        raise ValueError(message) from error
    if type(tensors) is not dict:
        message = "arm-C checkpoint decoder returned an invalid mapping"
        raise TypeError(message)
    canonical_state = save_safetensors(
        dict(sorted(tensors.items())),
        metadata=_SAFETENSORS_METADATA,
    )
    if canonical_state != state_bytes:
        message = "checkpoint state file is not canonical Iris safetensors v3"
        raise ValueError(message)
    descriptors = manifest.payload.tensors
    descriptor_keys = tuple(descriptor.key for descriptor in descriptors)
    if tuple(sorted(tensors)) != descriptor_keys:
        message = "arm-C checkpoint tensor keys disagree with manifest"
        raise ValueError(message)
    for descriptor in descriptors:
        tensor = tensors[descriptor.key]
        if (
            type(tensor) is not torch.Tensor
            or tensor.device != torch.device("cpu")
            or tensor.layout != torch.strided
            or tensor.is_inference()
            or tuple(tensor.shape) != descriptor.shape
            or _dtype_name(tensor.dtype) != descriptor.dtype
            or tensor.numel() * tensor.element_size() != descriptor.byte_length
            or not bool(torch.isfinite(tensor).all())
            or tensor_sha256(tensor, domain=_tensor_domain(descriptor)) != descriptor.sha256
        ):
            message = f"arm-C checkpoint tensor {descriptor.key!r} is invalid"
            raise ValueError(message)
    return _VerifiedBundle(
        manifest=manifest,
        tensors=tensors,
        manifest_bytes=canonical_json_file_bytes(manifest.model_dump(mode="json")),
        state_bytes=state_bytes,
    )


def validate_shared_buffer_teacher_checkpoint_bundle_v3(
    bundle: IrisSharedBufferTeacherCheckpointBundleV3,
) -> IrisSharedBufferTeacherCheckpointBundleV3:
    """Return an owned arm-C bundle after manifest and tensor validation."""
    try:
        verified = _validate_bundle_integrity(bundle)
        return IrisSharedBufferTeacherCheckpointBundleV3(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-C teacher checkpoint v3 could not be verified"
        raise IrisTeacherCheckpointError(message) from error


def read_shared_buffer_teacher_checkpoint_v3(
    path: Path,
) -> IrisSharedBufferTeacherCheckpointBundleV3:
    """Read one canonical arm-C checkpoint v3 directory."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "arm-C checkpoint reader requires an explicit directory Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_dir():
            message = "arm-C checkpoint path must be a real directory"
            raise ValueError(message)
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in entries) != (
            _MANIFEST_FILENAME,
            _STATE_FILENAME,
        ):
            message = "arm-C checkpoint directory must contain exactly two files"
            raise ValueError(message)
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            message = "arm-C checkpoint files must be real regular files"
            raise ValueError(message)
        manifest_bytes = (path / _MANIFEST_FILENAME).read_bytes()
        if (
            not manifest_bytes.endswith(b"\n")
            or manifest_bytes.endswith(b"\n\n")
            or manifest_bytes.startswith(b"\xef\xbb\xbf")
        ):
            message = "arm-C checkpoint manifest must be canonical UTF-8 JSON"
            raise ValueError(message)
        decoded = manifest_bytes[:-1].decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number is forbidden: {value}")
            ),
        )
        if canonical_json_file_bytes(raw) != manifest_bytes:
            message = "arm-C checkpoint manifest is not canonical iris-json-v1"
            raise ValueError(message)
        manifest = IrisSharedBufferTeacherCheckpointEnvelopeV3.model_validate_json(decoded)
        if path.name != manifest.sha256:
            message = "arm-C checkpoint directory name does not match payload hash"
            raise ValueError(message)
        verified = _validate_bundle_integrity(
            IrisSharedBufferTeacherCheckpointBundleV3(
                manifest=manifest,
                state_bytes=(path / _STATE_FILENAME).read_bytes(),
            )
        )
        if verified.manifest_bytes != manifest_bytes:
            message = "arm-C checkpoint manifest round trip changed bytes"
            raise ValueError(message)
        return IrisSharedBufferTeacherCheckpointBundleV3(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-C teacher checkpoint v3 could not be read"
        raise IrisTeacherCheckpointError(message) from error


def _existing_bundle_matches(path: Path, expected: _VerifiedBundle) -> bool:
    try:
        actual = read_shared_buffer_teacher_checkpoint_v3(path)
    except IrisTeacherCheckpointError:
        return False
    verified = _validate_bundle_integrity(actual)
    return (
        verified.manifest_bytes == expected.manifest_bytes
        and verified.state_bytes == expected.state_bytes
    )


def write_shared_buffer_teacher_checkpoint_v3(
    bundle: IrisSharedBufferTeacherCheckpointBundleV3,
    *,
    root: Path,
) -> IrisTeacherCheckpointWriteResult:
    """Install one arm-C checkpoint under ``teacher/v3/<payload hash>``."""
    temporary_path: Path | None = None
    try:
        if not isinstance(root, Path) or not root.name:
            message = "arm-C checkpoint writer requires an explicit root Path"
            raise TypeError(message)
        verified = _validate_bundle_integrity(bundle)
        parent = root / "teacher" / "v3"
        target = parent / verified.manifest.sha256
        if target.exists():
            if not _existing_bundle_matches(target, verified):
                message = "Existing arm-C checkpoint conflicts with expected bytes"
                raise IrisTeacherCheckpointError(message)
            reused = True
        else:
            parent.mkdir(parents=True, exist_ok=True)
            staging_candidate = parent / f".iris-{uuid4().hex}.tmp"
            os.mkdir(staging_candidate, mode=0o700)
            temporary_path = staging_candidate
            _write_exact_file(temporary_path / _STATE_FILENAME, verified.state_bytes)
            _write_exact_file(
                temporary_path / _MANIFEST_FILENAME,
                verified.manifest_bytes,
            )
            if (temporary_path / _STATE_FILENAME).read_bytes() != verified.state_bytes or (
                temporary_path / _MANIFEST_FILENAME
            ).read_bytes() != verified.manifest_bytes:
                message = "Arm-C checkpoint temporary files failed byte verification"
                raise OSError(message)
            try:
                rename_directory_with_windows_retry(temporary_path, target)
            except OSError as error:
                if not target.exists() or not _existing_bundle_matches(target, verified):
                    message = "Concurrent arm-C checkpoint installation conflicted"
                    raise IrisTeacherCheckpointError(message) from error
                _cleanup_temporary_directory(temporary_path)
                temporary_path = None
                reused = True
            else:
                temporary_path = None
                reused = False
        return IrisTeacherCheckpointWriteResult(
            path=target,
            payload_sha256=verified.manifest.sha256,
            state_file_sha256=verified.manifest.payload.state_file.sha256,
            state_size_bytes=len(verified.state_bytes),
            reused_existing=reused,
        )
    except BaseException as primary_error:
        if temporary_path is not None and temporary_path.exists():
            try:
                _cleanup_temporary_directory(temporary_path)
            except BaseException as cleanup_error:
                failures = BaseExceptionGroup(
                    "Arm-C checkpoint persistence and cleanup both failed",
                    [primary_error, cleanup_error],
                )
                raise IrisTeacherCheckpointError(
                    "Arm-C checkpoint failed and cleanup was incomplete"
                ) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Arm-C teacher checkpoint v3 could not be installed"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


def _identity_matches_current_model(
    model: IrisSharedBufferTeacherForcingModel,
    manifest: IrisSharedBufferTeacherCheckpointEnvelopeV3,
) -> dict[str, torch.Tensor]:
    provenance = model.checkpoint_provenance
    backbone, cognitive = _parse_model_provenance(provenance)
    identity = manifest.payload.model
    if (
        identity.arm_id != provenance.arm_id
        or identity.teacher_graph != provenance.teacher_graph
        or identity.execution_profile_set_sha256 != _PROFILE_SET_V4_SHA256
        or identity.shared_buffer_policy_sha256 != provenance.shared_buffer_policy_sha256
        or identity.protocol_declaration_payload_sha256 != _PROTOCOL_PAYLOAD_SHA256
        or identity.cognitive_cycles != provenance.cognitive_cycles
        or identity.pre_update_read_count != provenance.pre_update_read_count
        or identity.final_output_read_count != provenance.final_output_read_count
        or identity.entries_per_cycle != provenance.entries_per_cycle
        or identity.final_entry_count != provenance.final_entry_count
        or identity.latent_token_count != provenance.latent_token_count
        or identity.component_names != provenance.component_names
        or backbone.model_dump(mode="json") != identity.backbone.model_dump(mode="json")
        or cognitive.model_dump(mode="json") != identity.cognitive.model_dump(mode="json")
    ):
        message = "Arm-C checkpoint identity does not match target model"
        raise ValueError(message)
    runtime_values = (
        provenance.hidden_size,
        provenance.num_hidden_layers,
        provenance.vocabulary_size,
        provenance.max_position_embeddings,
        provenance.max_prompt_tokens,
        provenance.max_target_tokens,
        provenance.device,
        provenance.dtype,
    )
    manifest_values = (
        identity.runtime.hidden_size,
        identity.runtime.num_hidden_layers,
        identity.runtime.vocabulary_size,
        identity.runtime.max_position_embeddings,
        identity.runtime.max_prompt_tokens,
        identity.runtime.max_target_tokens,
        identity.runtime.device,
        identity.runtime.dtype,
    )
    if runtime_values != manifest_values:
        message = "Arm-C checkpoint runtime does not match target model"
        raise ValueError(message)
    current_state = model.state_dict(keep_vars=True)
    if tuple(current_state) != identity.state_names:
        message = "Arm-C checkpoint state topology does not match target model"
        raise ValueError(message)
    return dict(current_state)


def _optimizer_snapshot_from_checkpoint(
    verified: _VerifiedBundle,
) -> IrisTeacherOptimizerSnapshot:
    optimizer = verified.manifest.payload.optimizer
    config = optimizer.config
    descriptor_by_identity = {
        (descriptor.parameter_name, descriptor.state_name): descriptor
        for descriptor in verified.manifest.payload.tensors
        if descriptor.owner == "adamw"
    }
    states: list[IrisAdamWParameterSnapshot] = []
    for state in optimizer.parameter_states:
        exp_avg_descriptor = descriptor_by_identity[(state.parameter_name, "exp_avg")]
        exp_avg_sq_descriptor = descriptor_by_identity[(state.parameter_name, "exp_avg_sq")]
        states.append(
            IrisAdamWParameterSnapshot(
                parameter_name=state.parameter_name,
                step=state.step,
                exp_avg=verified.tensors[exp_avg_descriptor.key].detach().clone(),
                exp_avg_sq=verified.tensors[exp_avg_sq_descriptor.key].detach().clone(),
            )
        )
    return IrisTeacherOptimizerSnapshot(
        completed_steps=optimizer.completed_steps,
        config=IrisAdamWConfig(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
            max_gradient_norm=config.max_gradient_norm,
        ),
        parameter_names=optimizer.parameter_names,
        parameter_states=tuple(states),
    )


def restore_shared_buffer_teacher_checkpoint_v3(
    bundle: IrisSharedBufferTeacherCheckpointBundleV3,
    *,
    model: IrisSharedBufferTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisTeacherCheckpointRestoreResult:
    """Restore arm-C weights and AdamW state into a fresh exact arm-C owner."""
    backups: dict[str, torch.Tensor] | None = None
    targets: dict[str, torch.Tensor] | None = None
    model_mutated = False
    try:
        if type(model) is not IrisSharedBufferTeacherForcingModel:
            message = "checkpoint v3 target must be exactly arm-C teacher model"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own target arm-C model"
            raise TypeError(message)
        verified = _validate_bundle_integrity(bundle)
        targets = _identity_matches_current_model(model, verified.manifest)
        sidecar_descriptors = {
            descriptor.state_name: descriptor
            for descriptor in verified.manifest.payload.tensors
            if descriptor.owner == "sidecar"
        }
        candidates: dict[str, torch.Tensor] = {}
        backups = {}
        with torch.inference_mode(False), torch.no_grad():
            for name, target in targets.items():
                descriptor = sidecar_descriptors[name]
                source = verified.tensors[descriptor.key]
                if (
                    tuple(target.shape) != descriptor.shape
                    or _dtype_name(target.dtype) != descriptor.dtype
                    or str(target.device) != verified.manifest.payload.model.runtime.device
                ):
                    message = f"Arm-C checkpoint tensor {name!r} does not match target"
                    raise ValueError(message)
                candidates[name] = (
                    source.detach()
                    .to(device=target.device, dtype=target.dtype)
                    .contiguous()
                    .clone()
                )
                backups[name] = target.detach().clone()
        optimizer_snapshot = _optimizer_snapshot_from_checkpoint(verified)
        optimizer.validate_checkpoint_snapshot(optimizer_snapshot)

        model_mutated = True
        with torch.no_grad():
            for name in verified.manifest.payload.model.state_names:
                targets[name].copy_(candidates[name])
        restored = model.checkpoint_provenance
        if restored.sidecar_state_sha256 != verified.manifest.payload.model.sha256:
            message = "Restored arm-C sidecar hash does not match checkpoint"
            raise ValueError(message)
        optimizer.restore_checkpoint_snapshot(optimizer_snapshot)
        return IrisTeacherCheckpointRestoreResult(
            payload_sha256=verified.manifest.sha256,
            sidecar_state_sha256=restored.sidecar_state_sha256,
            completed_steps=optimizer.completed_steps,
        )
    except BaseException as primary_error:
        if model_mutated and backups is not None and targets is not None:
            try:
                with torch.no_grad():
                    for name, backup in backups.items():
                        targets[name].copy_(backup)
            except BaseException as rollback_error:
                failures = BaseExceptionGroup(
                    "Arm-C checkpoint restore and rollback both failed",
                    [primary_error, rollback_error],
                )
                raise IrisTeacherCheckpointError(
                    "Arm-C checkpoint restore failed and rollback was incomplete"
                ) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Arm-C teacher checkpoint v3 could not be restored"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


__all__ = [
    "IrisSharedBufferTeacherCheckpointBundleV3",
    "build_shared_buffer_teacher_checkpoint_v3",
    "read_shared_buffer_teacher_checkpoint_v3",
    "restore_shared_buffer_teacher_checkpoint_v3",
    "validate_shared_buffer_teacher_checkpoint_bundle_v3",
    "write_shared_buffer_teacher_checkpoint_v3",
]
