"""Bounded, immutable scientific D exports and transfer to an existing request graph.

Export the current clean run. For a historical/best checkpoint, first use the
existing exact restore into a fresh compatible run, then export that run. No
dataset, probes, optimizer, RNG state or frozen Qwen tensors enter the export.
"""

import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application.request import IrisRequestOrchestrator
from iris.application.request_sidecar_state import IrisRequestSidecarInstallResultV1
from iris.application.scientific_checkpoint import (
    _bounded_read,
    _json,
    _regular,
    _verify,
    build_scientific_checkpoint_v1,
)
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.checkpoint_v1 import (
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
)
from iris.artifacts.scientific_checkpoint_v1 import ScientificTensorV1
from iris.artifacts.scientific_inference_v1 import (
    REQUEST_COMPONENTS_V1,
    IrisScientificInferenceManifestV1,
    ScientificInferencePayloadV1,
)
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    named_module_state_sha256,
)

_META = {"format": "iris.scientific-inference.v1"}
_DOMAIN = "iris/scientific-inference/v1/payload"
_MAX_MANIFEST = 4 * 1024**2
_MAX_ARTIFACT = 200_000_000 - 1
_WEIGHTS = "iris_model.safetensors"
_TEACHER_COMPONENTS = (*REQUEST_COMPONENTS_V1[:-1], "sequence_assembler")


class IrisScientificInferenceError(RuntimeError):
    """Inference export could not be built, verified, published or transferred."""


@dataclass(frozen=True, slots=True)
class IrisScientificInferenceBundleV1:
    manifest: IrisScientificInferenceManifestV1
    state_bytes: bytes


@dataclass(frozen=True, slots=True)
class IrisScientificInferenceTransferV1:
    manifest_sha256: str
    source_checkpoint_sha256: str
    completed_steps: int
    installation: IrisRequestSidecarInstallResultV1


def _checked(
    bundle: IrisScientificInferenceBundleV1,
) -> tuple[IrisScientificInferenceManifestV1, dict[str, torch.Tensor]]:
    if (
        type(bundle) is not IrisScientificInferenceBundleV1
        or type(bundle.manifest) is not IrisScientificInferenceManifestV1
        or type(bundle.state_bytes) is not bytes
    ):
        raise ValueError("Exact inference bundle required")
    manifest = IrisScientificInferenceManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python", warnings=False)
    )
    payload = manifest.payload
    content = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    if (
        len(content) > _MAX_MANIFEST
        or len(content) + len(bundle.state_bytes) > _MAX_ARTIFACT
        or manifest.sha256
        != domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json")))
        or payload.state_sha256 != bytes_sha256(bundle.state_bytes)
        or payload.state_size_bytes != len(bundle.state_bytes)
    ):
        raise ValueError("Inference hashes or artifact byte budget differ")
    tensors = load_tensors(bundle.state_bytes)
    if set(tensors) != set(payload.model.state_names):
        raise ValueError("Inference tensor inventory differs")
    for entry in payload.tensors:
        value = tensors[entry.key]
        if (
            value.dtype != torch.float32
            or value.is_inference()
            or tuple(value.shape) != entry.shape
            or value.numel() * value.element_size() != entry.byte_length
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Inference tensor shape, dtype or values differ")
    state = {name: tensors[name] for name in payload.model.state_names}
    if (
        save_tensors(dict(sorted(state.items())), metadata=_META) != bundle.state_bytes
        or named_module_state_sha256(component_names=REQUEST_COMPONENTS_V1, state=state)
        != payload.model.sha256
        or named_module_state_sha256(component_names=_TEACHER_COMPONENTS, state=state)
        != payload.source_teacher_sidecar_sha256
    ):
        raise ValueError("Inference canonical state/topology identity differs")
    return manifest, state


def build_scientific_inference_v1(run: IrisScientificRunV1) -> IrisScientificInferenceBundleV1:
    """Project one clean current run, preserving its mode, weights, optimizer and RNG."""
    locked = False
    try:
        # This also supplies an exact, independently restorable source identity.
        checkpoint = build_scientific_checkpoint_v1(run)
        source_manifest, tensors = _verify(checkpoint)
        run._validate_boundary()
        if run._busy:
            raise ValueError("Idle scientific export source required")
        run._busy = locked = True
        model = run._optimizer._model
        teacher = model.teacher
        provenance = teacher.checkpoint_provenance
        names = tuple(teacher.state_dict())
        if (
            tuple(teacher._modules) != _TEACHER_COMPONENTS
            or names != tuple(name for name, _ in teacher.named_parameters())
            or source_manifest.payload.completed_steps != run.completed_steps
        ):
            raise ValueError("Export requires the canonical parameter-only D teacher")
        state = {name: tensors[f"model/teacher.{name}"] for name in names}
        if any(
            not torch.equal(value, teacher.state_dict()[name].detach().cpu())
            for name, value in state.items()
        ):
            raise ValueError("Export source changed after checkpoint capture")
        state_bytes = save_tensors(dict(sorted(state.items())), metadata=_META)
        runtime = TeacherBackboneRuntimeV1(
            hidden_size=provenance.hidden_size,
            num_hidden_layers=provenance.num_hidden_layers,
            vocabulary_size=provenance.vocabulary_size,
            max_position_embeddings=provenance.max_position_embeddings,
            max_prompt_tokens=provenance.max_prompt_tokens,
            max_target_tokens=provenance.max_target_tokens,
            device=provenance.device,
            dtype="float32",
        )
        payload = ScientificInferencePayloadV1(
            purpose=run._optimizer._policy.purpose,
            source_checkpoint_sha256=source_manifest.sha256,
            source_run_configuration_sha256=run.configuration_sha256,
            source_teacher_configuration_sha256=model.configuration_sha256,
            source_teacher_sidecar_sha256=provenance.sidecar_state_sha256,
            backbone_runtime_sha256=run._backbone_identity,
            completed_steps=run.completed_steps,
            model=TeacherCheckpointModelIdentityV1(
                sha256=named_module_state_sha256(
                    component_names=REQUEST_COMPONENTS_V1, state=state
                ),
                backbone=BackboneConfig.model_validate_json(provenance.backbone_configuration_json),
                cognitive=CognitiveConfig.model_validate_json(
                    provenance.cognitive_configuration_json
                ),
                workspace=WorkspaceConfig.model_validate_json(
                    provenance.workspace_configuration_json
                ),
                runtime=runtime,
                state_names=names,
            ),
            tensors=tuple(
                ScientificTensorV1(
                    key=name,
                    shape=tuple(value.shape),
                    dtype="float32",
                    byte_length=value.numel() * value.element_size(),
                )
                for name, value in sorted(state.items())
            ),
            state_sha256=bytes_sha256(state_bytes),
            state_size_bytes=len(state_bytes),
        )
        bundle = IrisScientificInferenceBundleV1(
            IrisScientificInferenceManifestV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            ),
            state_bytes,
        )
        _checked(bundle)
        run._validate_boundary()
        return bundle
    except Exception:
        raise IrisScientificInferenceError(
            "Scientific inference export could not be built"
        ) from None
    finally:
        if locked:
            run._busy = False


def transfer_scientific_inference_to_request_v1(
    bundle: IrisScientificInferenceBundleV1, *, request: IrisRequestOrchestrator
) -> IrisScientificInferenceTransferV1:
    """Validate before copying into an already-composed, independently owned D graph.

    Never loads a backbone or constructs modules from untrusted export dimensions.
    Uses the existing transactional installer and its explicit uncertain-state error.
    """
    try:
        if type(request) is not IrisRequestOrchestrator:
            raise ValueError("An exact precomposed arm-D request is required")
        manifest, state = _checked(bundle)
        payload, backbone = manifest.payload, request.backbone
        model = payload.model
        _ = request.sidecar_state_sha256
        if (
            model.backbone != BackboneConfig.model_validate_json(backbone.configuration_json)
            or model.cognitive
            != CognitiveConfig.model_validate_json(request._cognitive_config_snapshot)
            or model.workspace
            != WorkspaceConfig.model_validate_json(request._workspace_config_snapshot)
            or model.runtime.device != str(backbone.device)
            or model.runtime.dtype != str(backbone.dtype).removeprefix("torch.")
            or (
                model.runtime.hidden_size,
                model.runtime.num_hidden_layers,
                model.runtime.vocabulary_size,
                model.runtime.max_position_embeddings,
            )
            != (
                backbone.hidden_size,
                backbone.num_hidden_layers,
                backbone.vocabulary_size,
                backbone.max_position_embeddings,
            )
            or payload.backbone_runtime_sha256 != backbone.checkpoint_runtime_sha256
        ):
            raise ValueError("Inference model/backbone/tokenizer configuration differs")
        installation = request.restore_teacher_sidecar_state_v1(
            state, expected_installed_sha256=model.sha256
        )
        return IrisScientificInferenceTransferV1(
            manifest.sha256, payload.source_checkpoint_sha256, payload.completed_steps, installation
        )
    except Exception as error:
        # Preserve installer failure/rollback evidence in the cause chain.
        raise IrisScientificInferenceError(
            "Scientific inference state could not be transferred"
        ) from error


def read_scientific_inference_v1(path: Path) -> IrisScientificInferenceBundleV1:
    try:
        if not isinstance(path, Path) or not path.name:
            raise ValueError("Explicit inference directory required")
        path = path.absolute()
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {"manifest.json", _WEIGHTS}:
            raise ValueError("Inference directory inventory differs")
        content = _bounded_read(path / "manifest.json", _MAX_MANIFEST)
        if not content.endswith(b"\n"):
            raise ValueError("Inference manifest requires a final newline")
        _json(content[:-1])
        manifest = IrisScientificInferenceManifestV1.model_validate_json(content[:-1])
        if (
            path.name != manifest.sha256
            or canonical_json_file_bytes(manifest.model_dump(mode="json")) != content
        ):
            raise ValueError("Inference directory identity differs")
        bundle = IrisScientificInferenceBundleV1(
            manifest, _bounded_read(path / _WEIGHTS, _MAX_ARTIFACT - len(content))
        )
        _checked(bundle)
        return bundle
    except Exception:
        raise IrisScientificInferenceError(
            "Scientific inference export could not be read"
        ) from None


def write_scientific_inference_v1(
    bundle: IrisScientificInferenceBundleV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 512 * 1024**2,
) -> Path:
    """Publish once/reuse exact bytes; bounded dedicated store, no retention/deletion."""
    temporary: Path | None = None
    try:
        manifest, _ = _checked(bundle)
        if (
            not isinstance(root, Path)
            or not root.name
            or type(maximum_storage_bytes) is not int
            or not 1 <= maximum_storage_bytes <= 16 * 1024**3
        ):
            raise ValueError("Explicit inference root and bounded storage budget required")
        root = root.absolute() / "scientific-inference-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / manifest.sha256
        if target.exists():
            if read_scientific_inference_v1(target) != bundle:
                raise ValueError("Existing inference export conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        files = {
            "manifest.json": canonical_json_file_bytes(manifest.model_dump(mode="json")),
            _WEIGHTS: bundle.state_bytes,
        }
        required = sum(map(len, files.values()))
        if (
            occupied + required > maximum_storage_bytes
            or shutil.disk_usage(root).free < required + 1024**3
        ):
            raise ValueError("Inference storage budget/free-space reserve exceeded")
        staging_candidate = root / f".iris-inference-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary = staging_candidate  # Cleanup ownership begins only after exclusive creation.
        for name, content in files.items():
            _write_exact_file(temporary / name, content)
            if _bounded_read(temporary / name, len(content)) != content:
                raise ValueError("Staged inference bytes differ")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_inference_v1(target) != bundle:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificInferenceError(
            "Scientific inference export could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            for name in ("manifest.json", _WEIGHTS):
                (temporary / name).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "IrisScientificInferenceBundleV1",
    "IrisScientificInferenceError",
    "IrisScientificInferenceTransferV1",
    "build_scientific_inference_v1",
    "read_scientific_inference_v1",
    "transfer_scientific_inference_to_request_v1",
    "write_scientific_inference_v1",
]
