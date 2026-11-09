"""Versioned atomic checkpoint and exact resume for the Iris teacher sidecar."""

from __future__ import annotations

import json
import os
import platform
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from iris import __version__
from iris.application.errors import IrisTeacherCheckpointError
from iris.artifacts.checkpoint_v1 import (
    CheckpointFloatDTypeV1,
    IrisTeacherCheckpointEnvelopeV1,
    TeacherAdamWConfigV1,
    TeacherAdamWParameterStateV1,
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
    TeacherCheckpointOptimizerV1,
    TeacherCheckpointPayloadV1,
    TeacherCheckpointSchemaV1,
    TeacherCheckpointSoftwareV1,
    TeacherCheckpointStateFileV1,
    TeacherCheckpointTensorV1,
)
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training import (
    IrisAdamWConfig,
    IrisAdamWParameterSnapshot,
    IrisTeacherForcingModel,
    IrisTeacherModelCheckpointProvenance,
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

_PAYLOAD_DOMAIN = "iris/teacher-checkpoint/v1/payload"
_SAFETENSORS_METADATA = {"format": "iris.teacher-checkpoint.v1"}
_MANIFEST_FILENAME = "manifest.json"
_STATE_FILENAME = "state.safetensors"


@dataclass(frozen=True, slots=True)
class IrisTeacherCheckpointBundleV1:
    """One verified manifest paired with its immutable safetensors bytes."""

    manifest: IrisTeacherCheckpointEnvelopeV1
    state_bytes: bytes


@dataclass(frozen=True, slots=True)
class IrisTeacherCheckpointWriteResult:
    """Filesystem identity of one installed content-addressed checkpoint."""

    path: Path
    payload_sha256: str
    state_file_sha256: str
    state_size_bytes: int
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class IrisTeacherCheckpointRestoreResult:
    """Identity of sidecar and AdamW state restored into a fresh owner."""

    payload_sha256: str
    sidecar_state_sha256: str
    completed_steps: int


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    manifest: IrisTeacherCheckpointEnvelopeV1
    tensors: dict[str, torch.Tensor]
    manifest_bytes: bytes
    state_bytes: bytes


def _dtype_name(dtype: torch.dtype) -> CheckpointFloatDTypeV1:
    names: dict[torch.dtype, CheckpointFloatDTypeV1] = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as error:
        message = f"Unsupported checkpoint tensor dtype: {dtype}"
        raise ValueError(message) from error


def _tensor_domain(descriptor: TeacherCheckpointTensorV1) -> str:
    if descriptor.owner == "sidecar":
        return f"iris/teacher-checkpoint/v1/sidecar/{descriptor.state_name}"
    parameter_name = descriptor.parameter_name
    if parameter_name is None:
        message = "AdamW checkpoint descriptor is missing parameter_name"
        raise ValueError(message)
    return f"iris/teacher-checkpoint/v1/adamw/{parameter_name}/{descriptor.state_name}"


def _owned_cpu_tensor(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    if (
        type(tensor) is not torch.Tensor
        or tensor.layout != torch.strided
        or tensor.is_inference()
        or not torch.is_floating_point(tensor)
        or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}
        or not bool(torch.isfinite(tensor).all())
    ):
        message = f"{label} must be a finite ordinary dense checkpoint float tensor"
        raise ValueError(message)
    with torch.inference_mode(False), torch.no_grad():
        return tensor.detach().to(device="cpu").contiguous().clone()


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
    provenance: IrisTeacherModelCheckpointProvenance,
) -> tuple[BackboneConfig, CognitiveConfig, WorkspaceConfig]:
    if type(provenance) is not IrisTeacherModelCheckpointProvenance:
        message = "Teacher model returned invalid checkpoint provenance"
        raise TypeError(message)
    try:
        backbone = BackboneConfig.model_validate_json(provenance.backbone_configuration_json)
        cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
        workspace = WorkspaceConfig.model_validate_json(provenance.workspace_configuration_json)
    except Exception as error:
        message = "Teacher checkpoint configuration provenance is invalid"
        raise ValueError(message) from error
    return backbone, cognitive, workspace


def _clone_model_state(
    model: IrisTeacherForcingModel,
    *,
    provenance: IrisTeacherModelCheckpointProvenance,
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    try:
        state = model.state_dict()
    except Exception as error:
        message = "Failed to enumerate teacher sidecar checkpoint state"
        raise ValueError(message) from error
    names = tuple(state)
    if not names or len(set(names)) != len(names):
        message = "Teacher sidecar checkpoint state names are empty or duplicated"
        raise ValueError(message)
    expected_device = torch.device(provenance.device)
    expected_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(provenance.dtype)
    if expected_dtype is None:
        message = "Teacher checkpoint provenance has an unsupported dtype"
        raise ValueError(message)
    owned: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if type(name) is not str or not name or "\0" in name:
            message = "Teacher sidecar state names must be nonempty strings without NUL"
            raise ValueError(message)
        if type(tensor) is not torch.Tensor:
            message = f"Teacher sidecar state {name!r} is not an exact tensor"
            raise TypeError(message)
        if tensor.device != expected_device or tensor.dtype != expected_dtype:
            message = f"Teacher sidecar state {name!r} disagrees with checkpoint placement"
            raise ValueError(message)
        owned[name] = _owned_cpu_tensor(tensor, label=f"sidecar state {name!r}")
    return names, owned


def _optimizer_config_record(config: IrisAdamWConfig) -> TeacherAdamWConfigV1:
    return TeacherAdamWConfigV1(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay,
        max_gradient_norm=config.max_gradient_norm,
    )


def _software_record() -> TeacherCheckpointSoftwareV1:
    return TeacherCheckpointSoftwareV1(
        iris=__version__,
        python=platform.python_version(),
        torch=str(torch.__version__),
        safetensors=version("safetensors"),
    )


def _snapshots_equal(
    left: IrisTeacherOptimizerSnapshot,
    right: IrisTeacherOptimizerSnapshot,
) -> bool:
    if (
        left.completed_steps != right.completed_steps
        or left.config != right.config
        or left.parameter_names != right.parameter_names
        or len(left.parameter_states) != len(right.parameter_states)
    ):
        return False
    return all(
        left_state.parameter_name == right_state.parameter_name
        and left_state.step == right_state.step
        and torch.equal(left_state.exp_avg, right_state.exp_avg)
        and torch.equal(left_state.exp_avg_sq, right_state.exp_avg_sq)
        for left_state, right_state in zip(
            left.parameter_states,
            right.parameter_states,
            strict=True,
        )
    )


def build_teacher_checkpoint_v1(
    *,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisTeacherCheckpointBundleV1:
    """Capture one graph-and-optimizer checkpoint entirely in memory."""
    try:
        if not isinstance(model, IrisTeacherForcingModel):
            message = "checkpoint model must be an IrisTeacherForcingModel"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own the supplied teacher model"
            raise TypeError(message)
        optimizer_snapshot = optimizer.checkpoint_snapshot()
        provenance = model.checkpoint_provenance
        backbone, cognitive, workspace = _parse_model_provenance(provenance)
        state_names, sidecar_tensors = _clone_model_state(model, provenance=provenance)

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
                    label=f"AdamW {state_name} for {state.parameter_name!r}",
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
        payload = TeacherCheckpointPayloadV1(
            model=TeacherCheckpointModelIdentityV1(
                backbone=backbone,
                cognitive=cognitive,
                workspace=workspace,
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
        manifest = IrisTeacherCheckpointEnvelopeV1(
            schema_identity=TeacherCheckpointSchemaV1(
                artifact_type="iris.teacher_checkpoint",
                version=1,
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
            message = "Teacher sidecar provenance changed during checkpoint capture"
            raise ValueError(message)
        if not _snapshots_equal(optimizer.checkpoint_snapshot(), optimizer_snapshot):
            message = "AdamW state changed during checkpoint capture"
            raise ValueError(message)
        verified = _validate_bundle_integrity(
            IrisTeacherCheckpointBundleV1(manifest=manifest, state_bytes=state_bytes)
        )
        return IrisTeacherCheckpointBundleV1(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Iris teacher checkpoint could not be built"
        raise IrisTeacherCheckpointError(message) from error


def _validate_bundle_integrity(bundle: IrisTeacherCheckpointBundleV1) -> _VerifiedBundle:
    if type(bundle) is not IrisTeacherCheckpointBundleV1:
        message = "checkpoint bundle must be exactly IrisTeacherCheckpointBundleV1"
        raise TypeError(message)
    if type(bundle.manifest) is not IrisTeacherCheckpointEnvelopeV1:
        message = "checkpoint bundle manifest has an invalid type"
        raise TypeError(message)
    if type(bundle.state_bytes) is not bytes or not bundle.state_bytes:
        message = "checkpoint bundle state_bytes must be nonempty exact bytes"
        raise TypeError(message)
    manifest = IrisTeacherCheckpointEnvelopeV1.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    payload_bytes = canonical_json_bytes(manifest.payload.model_dump(mode="json"))
    if manifest.sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "checkpoint manifest payload SHA-256 is inconsistent"
        raise ValueError(message)
    state_bytes = bundle.state_bytes
    if manifest.payload.state_file.size_bytes != len(
        state_bytes
    ) or manifest.payload.state_file.sha256 != bytes_sha256(state_bytes):
        message = "checkpoint safetensors file identity is inconsistent"
        raise ValueError(message)
    try:
        tensors = load_safetensors(state_bytes)
    except Exception as error:
        message = "checkpoint safetensors bytes cannot be decoded"
        raise ValueError(message) from error
    if type(tensors) is not dict:
        message = "checkpoint safetensors decoder returned an invalid mapping"
        raise TypeError(message)
    canonical_state = save_safetensors(
        dict(sorted(tensors.items())),
        metadata=_SAFETENSORS_METADATA,
    )
    if canonical_state != state_bytes:
        message = "checkpoint state file is not canonical Iris safetensors v1"
        raise ValueError(message)
    descriptors = manifest.payload.tensors
    descriptor_keys = tuple(descriptor.key for descriptor in descriptors)
    if tuple(sorted(tensors)) != descriptor_keys:
        message = "checkpoint state-file tensor keys disagree with the manifest"
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
            message = f"checkpoint tensor {descriptor.key!r} failed manifest validation"
            raise ValueError(message)
    manifest_bytes = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    return _VerifiedBundle(
        manifest=manifest,
        tensors=tensors,
        manifest_bytes=manifest_bytes,
        state_bytes=state_bytes,
    )


def validate_teacher_checkpoint_bundle_v1(
    bundle: IrisTeacherCheckpointBundleV1,
) -> IrisTeacherCheckpointBundleV1:
    """Return an owned checkpoint bundle after complete manifest and tensor validation."""
    try:
        verified = _validate_bundle_integrity(bundle)
        return IrisTeacherCheckpointBundleV1(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Iris teacher checkpoint bundle could not be verified"
        raise IrisTeacherCheckpointError(message) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        value[key] = item
    return value


def read_teacher_checkpoint_v1(path: Path) -> IrisTeacherCheckpointBundleV1:
    """Read and verify one exact two-file teacher checkpoint directory."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "checkpoint reader requires an explicit directory Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_dir():
            message = "checkpoint path must be a real directory, not a symlink"
            raise ValueError(message)
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in entries) != (_MANIFEST_FILENAME, _STATE_FILENAME):
            message = (
                "checkpoint directory must contain exactly manifest.json and state.safetensors"
            )
            raise ValueError(message)
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            message = "checkpoint files must be real regular files"
            raise ValueError(message)
        manifest_bytes = (path / _MANIFEST_FILENAME).read_bytes()
        if (
            not manifest_bytes.endswith(b"\n")
            or manifest_bytes.endswith(b"\n\n")
            or manifest_bytes.startswith(b"\xef\xbb\xbf")
        ):
            message = "checkpoint manifest must be canonical UTF-8 JSON with one final LF"
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
            message = "checkpoint manifest is not canonical iris-json-v1"
            raise ValueError(message)
        manifest = IrisTeacherCheckpointEnvelopeV1.model_validate_json(decoded)
        if path.name != manifest.sha256:
            message = "checkpoint directory name does not match its payload SHA-256"
            raise ValueError(message)
        state_bytes = (path / _STATE_FILENAME).read_bytes()
        verified = _validate_bundle_integrity(
            IrisTeacherCheckpointBundleV1(manifest=manifest, state_bytes=state_bytes)
        )
        if verified.manifest_bytes != manifest_bytes:
            message = "checkpoint manifest canonical round trip changed its bytes"
            raise ValueError(message)
        return IrisTeacherCheckpointBundleV1(
            manifest=verified.manifest,
            state_bytes=verified.state_bytes,
        )
    except IrisTeacherCheckpointError:
        raise
    except Exception as error:
        message = "Iris teacher checkpoint could not be read or verified"
        raise IrisTeacherCheckpointError(message) from error


def _write_exact_file(path: Path, content: bytes) -> None:
    file_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "wb", buffering=0) as stream:
            remaining = memoryview(content)
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0:
                    message = "Checkpoint write made no forward progress"
                    raise OSError(message)
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(file_descriptor)
        raise


def _cleanup_temporary_directory(path: Path) -> None:
    # A completed rename can lose its acknowledgement; the owned stage is gone.
    if not path.exists():
        return
    for filename in (_MANIFEST_FILENAME, _STATE_FILENAME):
        (path / filename).unlink(missing_ok=True)
    path.rmdir()


def _existing_bundle_matches(path: Path, expected: _VerifiedBundle) -> bool:
    try:
        actual = read_teacher_checkpoint_v1(path)
    except IrisTeacherCheckpointError:
        return False
    actual_verified = _validate_bundle_integrity(actual)
    return (
        actual_verified.manifest_bytes == expected.manifest_bytes
        and actual_verified.state_bytes == expected.state_bytes
    )


def write_teacher_checkpoint_v1(
    bundle: IrisTeacherCheckpointBundleV1,
    *,
    root: Path,
) -> IrisTeacherCheckpointWriteResult:
    """Install one complete checkpoint directory atomically without overwrite."""
    temporary_path: Path | None = None
    try:
        if not isinstance(root, Path) or not root.name:
            message = "checkpoint writer requires an explicit root Path"
            raise TypeError(message)
        verified = _validate_bundle_integrity(bundle)
        parent = root / "teacher" / "v1"
        target = parent / verified.manifest.sha256
        if target.exists():
            if not _existing_bundle_matches(target, verified):
                message = "Existing content-addressed checkpoint conflicts with expected bytes"
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
                message = "Checkpoint temporary files failed byte verification"
                raise OSError(message)
            try:
                rename_directory_with_windows_retry(temporary_path, target)
            except OSError as error:
                if not target.exists() or not _existing_bundle_matches(target, verified):
                    message = "Concurrent checkpoint installation conflicted with expected bytes"
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
                    "Checkpoint persistence and temporary cleanup both failed",
                    [primary_error, cleanup_error],
                )
                message = "Iris teacher checkpoint failed and cleanup was incomplete"
                raise IrisTeacherCheckpointError(message) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris teacher checkpoint could not be installed atomically"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


def _identity_matches_current_model(
    model: IrisTeacherForcingModel,
    manifest: IrisTeacherCheckpointEnvelopeV1,
) -> dict[str, torch.Tensor]:
    provenance = model.checkpoint_provenance
    backbone, cognitive, workspace = _parse_model_provenance(provenance)
    identity = manifest.payload.model
    if (
        backbone.model_dump(mode="json") != identity.backbone.model_dump(mode="json")
        or cognitive.model_dump(mode="json") != identity.cognitive.model_dump(mode="json")
        or workspace.model_dump(mode="json") != identity.workspace.model_dump(mode="json")
    ):
        message = "Checkpoint configuration does not match the target teacher model"
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
        message = "Checkpoint backbone runtime identity does not match the target model"
        raise ValueError(message)
    current_state = model.state_dict(keep_vars=True)
    if tuple(current_state) != identity.state_names:
        message = "Checkpoint state topology does not match the target teacher model"
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


def restore_teacher_checkpoint_v1(
    bundle: IrisTeacherCheckpointBundleV1,
    *,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisTeacherCheckpointRestoreResult:
    """Restore verified sidecar weights and AdamW moments into one fresh owner."""
    backups: dict[str, torch.Tensor] | None = None
    targets: dict[str, torch.Tensor] | None = None
    try:
        if not isinstance(model, IrisTeacherForcingModel):
            message = "checkpoint target must be an IrisTeacherForcingModel"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "checkpoint optimizer must exactly own the target teacher model"
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
                    message = f"Checkpoint tensor {name!r} does not match target state"
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
        restored_provenance = model.checkpoint_provenance
        if restored_provenance.sidecar_state_sha256 != verified.manifest.payload.model.sha256:
            message = "Restored sidecar state SHA-256 does not match the checkpoint"
            raise ValueError(message)
        optimizer.restore_checkpoint_snapshot(optimizer_snapshot)
        return IrisTeacherCheckpointRestoreResult(
            payload_sha256=verified.manifest.sha256,
            sidecar_state_sha256=restored_provenance.sidecar_state_sha256,
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
                    "Checkpoint restore and sidecar rollback both failed",
                    [primary_error, rollback_error],
                )
                message = "Iris teacher checkpoint restore failed and rollback was incomplete"
                raise IrisTeacherCheckpointError(message) from failures
        if isinstance(primary_error, IrisTeacherCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris teacher checkpoint could not be restored"
            raise IrisTeacherCheckpointError(message) from primary_error
        raise


__all__ = [
    "IrisTeacherCheckpointBundleV1",
    "IrisTeacherCheckpointRestoreResult",
    "IrisTeacherCheckpointWriteResult",
    "build_teacher_checkpoint_v1",
    "read_teacher_checkpoint_v1",
    "restore_teacher_checkpoint_v1",
    "validate_teacher_checkpoint_bundle_v1",
    "write_teacher_checkpoint_v1",
]
