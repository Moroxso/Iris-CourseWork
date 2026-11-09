"""Atomic arm-B teacher checkpoint v2 with exact no-workspace restore identity."""

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
from iris.artifacts.checkpoint_v2 import (
    IrisNoWorkspaceTeacherCheckpointEnvelopeV2,
    NoWorkspaceTeacherCheckpointModelIdentityV2,
    NoWorkspaceTeacherCheckpointPayloadV2,
    NoWorkspaceTeacherCheckpointSchemaV2,
)
from iris.schemas import BackboneConfig, CognitiveConfig
from iris.training import (
    IrisAdamWConfig,
    IrisAdamWParameterSnapshot,
    IrisNoWorkspaceTeacherCheckpointProvenanceV1,
    IrisNoWorkspaceTeacherForcingModel,
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

_PAYLOAD_DOMAIN = "iris/teacher-checkpoint/v2/payload"
_SAFETENSORS_METADATA = {"format": "iris.teacher-checkpoint.v2"}
_MANIFEST_FILENAME = "manifest.json"
_STATE_FILENAME = "state.safetensors"
_PROFILE_SET_V3_SHA256: Literal[
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
] = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"


@dataclass(frozen=True, slots=True)
class IrisNoWorkspaceTeacherCheckpointBundleV2:
    """One verified arm-B manifest paired with immutable safetensors bytes."""

    manifest: IrisNoWorkspaceTeacherCheckpointEnvelopeV2
    state_bytes: bytes


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    manifest: IrisNoWorkspaceTeacherCheckpointEnvelopeV2
    tensors: dict[str, torch.Tensor]
    manifest_bytes: bytes
    state_bytes: bytes


def _tensor_domain(descriptor: TeacherCheckpointTensorV1) -> str:
    if descriptor.owner == "sidecar":
        return f"iris/teacher-checkpoint/v2/sidecar/{descriptor.state_name}"
    parameter_name = descriptor.parameter_name
    if parameter_name is None:
        message = "AdamW checkpoint descriptor is missing parameter_name"
        raise ValueError(message)
    return f"iris/teacher-checkpoint/v2/adamw/{parameter_name}/{descriptor.state_name}"


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
    provenance: IrisNoWorkspaceTeacherCheckpointProvenanceV1,
) -> tuple[BackboneConfig, CognitiveConfig]:
    if type(provenance) is not IrisNoWorkspaceTeacherCheckpointProvenanceV1:
        message = "Arm-B teacher model returned invalid checkpoint provenance"
        raise TypeError(message)
    try:
        backbone = BackboneConfig.model_validate_json(provenance.backbone_configuration_json)
        cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    except Exception as error:
        message = "Arm-B checkpoint configuration provenance is invalid"
        raise ValueError(message) from error
    if (
        provenance.arm_id != "B"
        or provenance.teacher_graph != "independent_recurrent_modules_zero_memory_read_teacher_v1"
        or provenance.cognitive_cycles != 3
        or provenance.latent_token_count != cognitive.num_modules
    ):
        message = "Arm-B checkpoint graph provenance is inconsistent"
        raise ValueError(message)
    return backbone, cognitive


def _clone_model_state(
    model: IrisNoWorkspaceTeacherForcingModel,
    *,
    provenance: IrisNoWorkspaceTeacherCheckpointProvenanceV1,
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    state = model.state_dict()
    names = tuple(state)
    if not names or len(set(names)) != len(names):
        message = "Arm-B sidecar checkpoint state names are empty or duplicated"
        raise ValueError(message)
    expected_device = torch.device(provenance.device)
    expected_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(provenance.dtype)
    if expected_dtype is None:
        message = "Arm-B checkpoint provenance has an unsupported dtype"
        raise ValueError(message)
    owned: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if type(name) is not str or not name or "\0" in name:
            message = "Arm-B sidecar state names must be nonempty without NUL"
            raise ValueError(message)
        if type(tensor) is not torch.Tensor:
            message = f"Arm-B sidecar state {name!r} is not an exact tensor"
            raise TypeError(message)
        if tensor.device != expected_device or tensor.dtype != expected_dtype:
            message = f"Arm-B sidecar state {name!r} disagrees with placement"
            raise ValueError(message)
        owned[name] = _owned_cpu_tensor(tensor, label=f"arm-B sidecar state {name!r}")
    return names, owned


def build_no_workspace_teacher_checkpoint_v2(
    *,
    model: IrisNoWorkspaceTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisNoWorkspaceTeacherCheckpointBundleV2:
    """Capture one arm-B graph-and-AdamW checkpoint entirely in memory."""
    try:
        if type(model) is not IrisNoWorkspaceTeacherForcingModel:
            message = "checkpoint v2 model must be exactly arm-B teacher model"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own the arm-B teacher model"
            raise TypeError(message)
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
                    label=f"arm-B AdamW {state_name} for {state.parameter_name!r}",
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
        state_bytes = save_safetensors(ordered_tensors, metadata=_SAFETENSORS_METADATA)
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
        payload = NoWorkspaceTeacherCheckpointPayloadV2(
            model=NoWorkspaceTeacherCheckpointModelIdentityV2(
                arm_id="B",
                arm_kind="llm_modules_no_workspace",
                teacher_graph=("independent_recurrent_modules_zero_memory_read_teacher_v1"),
                execution_profile_set_sha256=_PROFILE_SET_V3_SHA256,
                backbone=backbone,
                cognitive=cognitive,
                cognitive_cycles=3,
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
                filename="state.safetensors",
                size_bytes=len(state_bytes),
                tensor_count=len(ordered_tensors),
                sha256=bytes_sha256(state_bytes),
            ),
            tensors=tuple(descriptors[key] for key in sorted(descriptors)),
            software=_software_record(),
        )
        manifest = IrisNoWorkspaceTeacherCheckpointEnvelopeV2(
            schema_identity=NoWorkspaceTeacherCheckpointSchemaV2(
                artifact_type="iris.teacher_checkpoint",
                version=2,
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
            message = "Arm-B sidecar provenance changed during checkpoint capture"
            raise ValueError(message)
        if not _snapshots_equal(optimizer.checkpoint_snapshot(), optimizer_snapshot):
            message = "Arm-B AdamW state changed during checkpoint capture"
            raise ValueError(message)
        verified = _validate_bundle_integrity(
            IrisNoWorkspaceTeacherCheckpointBundleV2(
                manifest=manifest,
                state_bytes=state_bytes,
            )
        )
        return IrisNoWorkspaceTeacherCheckpointBundleV2(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-B teacher checkpoint v2 could not be built"
        raise IrisTeacherCheckpointError(message) from error


def _validate_bundle_integrity(
    bundle: IrisNoWorkspaceTeacherCheckpointBundleV2,
) -> _VerifiedBundle:
    if type(bundle) is not IrisNoWorkspaceTeacherCheckpointBundleV2:
        message = "checkpoint bundle must be exactly arm-B checkpoint v2"
        raise TypeError(message)
    if type(bundle.manifest) is not IrisNoWorkspaceTeacherCheckpointEnvelopeV2:
        message = "arm-B checkpoint bundle manifest has an invalid type"
        raise TypeError(message)
    if type(bundle.state_bytes) is not bytes or not bundle.state_bytes:
        message = "arm-B checkpoint state_bytes must be nonempty exact bytes"
        raise TypeError(message)
    manifest = IrisNoWorkspaceTeacherCheckpointEnvelopeV2.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    payload_bytes = canonical_json_bytes(manifest.payload.model_dump(mode="json"))
    if manifest.sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "arm-B checkpoint payload SHA-256 is inconsistent"
        raise ValueError(message)
    state_bytes = bundle.state_bytes
    if manifest.payload.state_file.size_bytes != len(
        state_bytes
    ) or manifest.payload.state_file.sha256 != bytes_sha256(state_bytes):
        message = "arm-B checkpoint safetensors identity is inconsistent"
        raise ValueError(message)
    try:
        tensors = load_safetensors(state_bytes)
    except Exception as error:
        message = "arm-B checkpoint safetensors bytes cannot be decoded"
        raise ValueError(message) from error
    if type(tensors) is not dict:
        message = "arm-B checkpoint decoder returned an invalid mapping"
        raise TypeError(message)
    canonical_state = save_safetensors(
        dict(sorted(tensors.items())),
        metadata=_SAFETENSORS_METADATA,
    )
    if canonical_state != state_bytes:
        message = "checkpoint state file is not canonical Iris safetensors v2"
        raise ValueError(message)
    descriptors = manifest.payload.tensors
    descriptor_keys = tuple(descriptor.key for descriptor in descriptors)
    if tuple(sorted(tensors)) != descriptor_keys:
        message = "arm-B checkpoint tensor keys disagree with manifest"
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
            message = f"arm-B checkpoint tensor {descriptor.key!r} is invalid"
            raise ValueError(message)
    return _VerifiedBundle(
        manifest=manifest,
        tensors=tensors,
        manifest_bytes=canonical_json_file_bytes(manifest.model_dump(mode="json")),
        state_bytes=state_bytes,
    )


def validate_no_workspace_teacher_checkpoint_bundle_v2(
    bundle: IrisNoWorkspaceTeacherCheckpointBundleV2,
) -> IrisNoWorkspaceTeacherCheckpointBundleV2:
    """Return an owned arm-B bundle after full manifest and tensor validation."""
    try:
        verified = _validate_bundle_integrity(bundle)
        return IrisNoWorkspaceTeacherCheckpointBundleV2(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-B teacher checkpoint v2 could not be verified"
        raise IrisTeacherCheckpointError(message) from error


def read_no_workspace_teacher_checkpoint_v2(
    path: Path,
) -> IrisNoWorkspaceTeacherCheckpointBundleV2:
    """Read one canonical arm-B checkpoint v2 directory."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "arm-B checkpoint reader requires an explicit directory Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_dir():
            message = "arm-B checkpoint path must be a real directory"
            raise ValueError(message)
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in entries) != (
            _MANIFEST_FILENAME,
            _STATE_FILENAME,
        ):
            message = "arm-B checkpoint directory must contain exactly two files"
            raise ValueError(message)
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            message = "arm-B checkpoint files must be real regular files"
            raise ValueError(message)
        manifest_bytes = (path / _MANIFEST_FILENAME).read_bytes()
        if (
            not manifest_bytes.endswith(b"\n")
            or manifest_bytes.endswith(b"\n\n")
            or manifest_bytes.startswith(b"\xef\xbb\xbf")
        ):
            message = "arm-B checkpoint manifest must be canonical UTF-8 JSON"
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
            message = "arm-B checkpoint manifest is not canonical iris-json-v1"
            raise ValueError(message)
        manifest = IrisNoWorkspaceTeacherCheckpointEnvelopeV2.model_validate_json(decoded)
        if path.name != manifest.sha256:
            message = "arm-B checkpoint directory name does not match payload hash"
            raise ValueError(message)
        verified = _validate_bundle_integrity(
            IrisNoWorkspaceTeacherCheckpointBundleV2(
                manifest=manifest,
                state_bytes=(path / _STATE_FILENAME).read_bytes(),
            )
        )
        if verified.manifest_bytes != manifest_bytes:
            message = "arm-B checkpoint manifest round trip changed bytes"
            raise ValueError(message)
        return IrisNoWorkspaceTeacherCheckpointBundleV2(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Arm-B teacher checkpoint v2 could not be read"
        raise IrisTeacherCheckpointError(message) from error


def _existing_bundle_matches(path: Path, expected: _VerifiedBundle) -> bool:
    try:
        actual = read_no_workspace_teacher_checkpoint_v2(path)
    except IrisTeacherCheckpointError:
        return False
    verified = _validate_bundle_integrity(actual)
    return (
        verified.manifest_bytes == expected.manifest_bytes
        and verified.state_bytes == expected.state_bytes
    )


def write_no_workspace_teacher_checkpoint_v2(
    bundle: IrisNoWorkspaceTeacherCheckpointBundleV2,
    *,
    root: Path,
) -> IrisTeacherCheckpointWriteResult:
    """Install one arm-B checkpoint under ``teacher/v2/<payload hash>``."""
    temporary_path: Path | None = None
    try:
        if not isinstance(root, Path) or not root.name:
            message = "arm-B checkpoint writer requires an explicit root Path"
            raise TypeError(message)
        verified = _validate_bundle_integrity(bundle)
        parent = root / "teacher" / "v2"
        target = parent / verified.manifest.sha256
        if target.exists():
            if not _existing_bundle_matches(target, verified):
                message = "Existing arm-B checkpoint conflicts with expected bytes"
                raise IrisTeacherCheckpointError(message)
            reused = True
        else:
            parent.mkdir(parents=True, exist_ok=True)
            staging_candidate = parent / f".iris-{uuid4().hex}.tmp"
            os.mkdir(staging_candidate, mode=0o700)
            temporary_path = staging_candidate
            _write_exact_file(temporary_path / _STATE_FILENAME, verified.state_bytes)
            _write_exact_file(temporary_path / _MANIFEST_FILENAME, verified.manifest_bytes)
            if (temporary_path / _STATE_FILENAME).read_bytes() != verified.state_bytes or (
                temporary_path / _MANIFEST_FILENAME
            ).read_bytes() != verified.manifest_bytes:
                message = "Arm-B checkpoint temporary files failed byte verification"
                raise OSError(message)
            try:
                rename_directory_with_windows_retry(temporary_path, target)
            except OSError as error:
                if not target.exists() or not _existing_bundle_matches(target, verified):
                    message = "Concurrent arm-B checkpoint installation conflicted"
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
                    "Arm-B checkpoint persistence and cleanup both failed",
                    [primary_error, cleanup_error],
                )
                raise IrisTeacherCheckpointError(
                    "Arm-B checkpoint failed and cleanup was incomplete"
                ) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Arm-B teacher checkpoint v2 could not be installed"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


def _identity_matches_current_model(
    model: IrisNoWorkspaceTeacherForcingModel,
    manifest: IrisNoWorkspaceTeacherCheckpointEnvelopeV2,
) -> dict[str, torch.Tensor]:
    provenance = model.checkpoint_provenance
    backbone, cognitive = _parse_model_provenance(provenance)
    identity = manifest.payload.model
    if (
        identity.arm_id != provenance.arm_id
        or identity.teacher_graph != provenance.teacher_graph
        or identity.execution_profile_set_sha256 != _PROFILE_SET_V3_SHA256
        or identity.cognitive_cycles != provenance.cognitive_cycles
        or identity.latent_token_count != provenance.latent_token_count
        or identity.component_names != provenance.component_names
        or backbone.model_dump(mode="json") != identity.backbone.model_dump(mode="json")
        or cognitive.model_dump(mode="json") != identity.cognitive.model_dump(mode="json")
    ):
        message = "Arm-B checkpoint identity does not match target model"
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
        message = "Arm-B checkpoint runtime does not match target model"
        raise ValueError(message)
    current_state = model.state_dict(keep_vars=True)
    if tuple(current_state) != identity.state_names:
        message = "Arm-B checkpoint state topology does not match target model"
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


def restore_no_workspace_teacher_checkpoint_v2(
    bundle: IrisNoWorkspaceTeacherCheckpointBundleV2,
    *,
    model: IrisNoWorkspaceTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisTeacherCheckpointRestoreResult:
    """Restore arm-B weights and AdamW state into one fresh exact arm-B owner."""
    backups: dict[str, torch.Tensor] | None = None
    targets: dict[str, torch.Tensor] | None = None
    try:
        if type(model) is not IrisNoWorkspaceTeacherForcingModel:
            message = "checkpoint v2 target must be exactly arm-B teacher model"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own target arm-B model"
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
                    message = f"Arm-B checkpoint tensor {name!r} does not match target"
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
        with torch.no_grad():
            for name in verified.manifest.payload.model.state_names:
                targets[name].copy_(candidates[name])
        restored = model.checkpoint_provenance
        if restored.sidecar_state_sha256 != verified.manifest.payload.model.sha256:
            message = "Restored arm-B sidecar hash does not match checkpoint"
            raise ValueError(message)
        optimizer.restore_checkpoint_snapshot(optimizer_snapshot)
        return IrisTeacherCheckpointRestoreResult(
            payload_sha256=verified.manifest.sha256,
            sidecar_state_sha256=restored.sidecar_state_sha256,
            completed_steps=optimizer.completed_steps,
        )
    except BaseException as primary_error:
        if backups is not None and targets is not None:
            try:
                with torch.no_grad():
                    for name, backup in backups.items():
                        targets[name].copy_(backup)
            except BaseException as rollback_error:
                failures = BaseExceptionGroup(
                    "Arm-B checkpoint restore and rollback both failed",
                    [primary_error, rollback_error],
                )
                raise IrisTeacherCheckpointError(
                    "Arm-B checkpoint restore failed and rollback was incomplete"
                ) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Arm-B teacher checkpoint v2 could not be restored"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


__all__ = [
    "IrisNoWorkspaceTeacherCheckpointBundleV2",
    "build_no_workspace_teacher_checkpoint_v2",
    "read_no_workspace_teacher_checkpoint_v2",
    "restore_no_workspace_teacher_checkpoint_v2",
    "validate_no_workspace_teacher_checkpoint_bundle_v2",
    "write_no_workspace_teacher_checkpoint_v2",
]
