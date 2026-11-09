"""Source-section22 nine-file scientific run delivery and exact fresh-owner resume."""

import json
import os
import shutil
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _bounded_read,
    _json,
    _regular,
    _verify,
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizerPolicyV1,
    scientific_learning_rate_v1,
)
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_torch_state import (
    decode_scientific_torch_state_v1,
    encode_scientific_torch_state_v1,
)
from iris.application.scientific_validation import IrisScientificValidationV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_delivery_v1 import (
    ScientificDeliveryConfigurationV1,
    ScientificDeliveryMetricsV1,
    ScientificDeliveryValidationV1,
)
from iris.integrations.tokenizer_semantics import tokenizer_semantics_sha256
from iris.schemas import BackboneConfig
from iris.training.probes import probe_vocabulary_sha256
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)

_LIMIT = 200_000_000 - 1
_VERSION = b"iris-scientific-delivery-v1\n"
_MODEL_META = {"format": "iris.scientific-delivery.model.v1"}
_CHECKPOINT_META = {"format": "iris.scientific-checkpoint.v1"}
_FILE_LIMITS = {
    "README.md": 65536,
    "backbone.json": 65536,
    "config.yaml": 16 * 1024**2,
    "environment.json": 1024**2,
    "iris_model.safetensors": _LIMIT,
    "metrics.json": 1024**2,
    "schema_version.txt": 256,
    "tokenizer.json": 32 * 1024**2,
    "training_state.pt": _LIMIT,
}
_README = b"""# Iris scientific training checkpoint\n
Format: iris-scientific-delivery-v1; arm D; explicit FP32 run state.
iris_model.safetensors holds all registered trainable teacher AND probe weights.
training_state.pt is a canonical torch.save container with control_json and
optimizer/RNG/identity-buffer tensors; it contains no trainable model weights.
PyTorch may inspect it with weights_only=True. Iris never unpickles incoming
metadata: bounded raw storages are checked by exact canonical reserialization.
config.yaml is the canonical JSON subset of YAML 1.2, without tags or resolvers.
It includes the full run manifest, teacher/probe policies and ordered label domains.
backbone.json references external frozen weights; Qwen tensors are not copied here.
tokenizer.json is the canonical fast-tokenizer backend with transient padding and
truncation disabled; tokenizer special-token/runtime semantics are also bound.
metrics.json records an optional supplied same-step validation report; null means
unavailable, not zero. environment.json records the source run software settings.
Use read_scientific_delivery_v1 then restore_scientific_delivery_v1 into a fresh
compatible IrisScientificRunV1 with the original corpus and already-loaded backbone.
No automatic installation, backbone download, model construction or code execution.
Export the restored run with ADR0065 for deployment without the training probes.
The nine-file directory's name is the content identity of every member.
This is a run checkpoint, not a lifecycle best/current envelope or an experiment
metric history. No CUDA, cross-environment bitwise or scientific verdict is implied.
"""


class IrisScientificDeliveryError(RuntimeError):
    """Source-facing training delivery is incompatible or could not be persisted."""


@dataclass(frozen=True, slots=True)
class IrisScientificDeliveryBundleV1:
    sha256: str
    files: tuple[tuple[str, bytes], ...]


def _identity(files: dict[str, bytes]) -> str:
    return domain_sha256(
        "iris/scientific-delivery/v1",
        canonical_json_bytes(
            [
                {"name": name, "size_bytes": len(data), "sha256": bytes_sha256(data)}
                for name, data in sorted(files.items())
            ]
        ),
    )


def _json_file(content: bytes) -> object:
    if not content.endswith(b"\n"):
        raise ValueError("Canonical delivery JSON requires a final newline")
    return _json(content[:-1])


def _tokenizer_file(run: IrisScientificRunV1) -> bytes:
    backend = json.loads(run._backbone._tokenizer.backend_tokenizer.to_str())
    backend["padding"] = backend["truncation"] = None
    return canonical_json_file_bytes(backend)


def _control(config: ScientificDeliveryConfigurationV1) -> str:
    payload = config.checkpoint.payload
    run_config = _json(payload.configuration_json.encode())
    policy = IrisScientificOptimizerPolicyV1.model_validate(run_config["optimizer_policy"])
    return canonical_json_bytes(
        {
            "kind": "iris-scientific-training-control-v1",
            "source_checkpoint_sha256": config.checkpoint.sha256,
            "global_step": payload.completed_steps,
            "examples_seen": payload.examples_seen,
            "target_tokens_seen": payload.target_tokens_seen,
            "history_sha256": payload.history_sha256,
            "parameter_names": payload.parameter_names,
            "parameter_steps": payload.parameter_steps,
            "rng_and_backend": payload.runtime.model_dump(mode="json"),
            "scheduler": {
                "policy": policy.model_dump(mode="json"),
                "current_learning_rate": scientific_learning_rate_v1(
                    policy, step=max(1, payload.completed_steps)
                ),
            },
            "precision": "fp32",
            "grad_scaler": None,
        }
    ).decode("utf-8")


def _side_files(config: ScientificDeliveryConfigurationV1) -> dict[str, bytes]:
    teacher = _json(config.teacher_configuration_json.encode())
    backbone_config = BackboneConfig.model_validate_json(teacher["backbone_configuration_json"])
    if backbone_config.model_dump_json() != teacher["backbone_configuration_json"]:
        raise ValueError("Backbone configuration snapshot is not its normalized schema encoding")
    backbone = backbone_config.model_dump(mode="json")
    run_config = _json(config.checkpoint.payload.configuration_json.encode())
    return {
        "schema_version.txt": _VERSION,
        "README.md": _README,
        "backbone.json": canonical_json_file_bytes(
            {
                "repository": backbone["repository"],
                "revision": backbone["revision"],
                "tokenizer_revision": backbone["tokenizer_revision"],
                "frozen": True,
            }
        ),
        "environment.json": canonical_json_file_bytes(run_config["software"]),
        "config.yaml": canonical_json_file_bytes(config.model_dump(mode="json")),
    }


def _checked(
    bundle: IrisScientificDeliveryBundleV1,
) -> tuple[IrisScientificCheckpointBundleV1, ScientificDeliveryConfigurationV1]:
    if type(bundle) is not IrisScientificDeliveryBundleV1 or type(bundle.files) is not tuple:
        raise ValueError("Exact delivery bundle required")
    if (
        len(bundle.files) != len(_FILE_LIMITS)
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not bytes
            for pair in bundle.files
        )
        or tuple(name for name, _ in bundle.files) != tuple(sorted(_FILE_LIMITS))
    ):
        raise ValueError("Exact nine-file delivery inventory required")
    files = dict(bundle.files)
    if (
        any(not 1 <= len(data) <= _FILE_LIMITS[name] for name, data in files.items())
        or sum(map(len, files.values())) > _LIMIT
        or _identity(files) != bundle.sha256
    ):
        raise ValueError("Delivery content identity or byte budget differs")
    _json_file(files["config.yaml"])
    config = ScientificDeliveryConfigurationV1.model_validate_json(files["config.yaml"])
    for name, expected in _side_files(config).items():
        if files[name] != expected:
            raise ValueError("Delivery configuration/environment/backbone metadata differs")
    _json_file(files["tokenizer.json"])
    if bytes_sha256(files["tokenizer.json"]) != config.tokenizer_json_sha256:
        raise ValueError("Delivery tokenizer file identity differs")
    _json_file(files["metrics.json"])
    metrics = ScientificDeliveryMetricsV1.model_validate_json(files["metrics.json"])
    if canonical_json_file_bytes(metrics.model_dump(mode="json")) != files["metrics.json"]:
        raise ValueError("Noncanonical delivery metrics")
    payload = config.checkpoint.payload
    if metrics.validation is not None and (
        metrics.validation.at_optimizer_step != payload.completed_steps
        or metrics.validation.run_configuration_sha256 != payload.configuration_sha256
    ):
        raise ValueError("Delivery validation metric belongs to a different run/step")
    parameters = load_tensors(files["iris_model.safetensors"])
    if (
        set(payload.model_state_names) - set(payload.parameter_names)
        != {"configuration_identity", "probes.configuration_identity"}
        or set(parameters) != set(payload.parameter_names)
        or any(value.dtype != torch.float32 for value in parameters.values())
        or save_tensors(dict(sorted(parameters.items())), metadata=_MODEL_META)
        != files["iris_model.safetensors"]
    ):
        raise ValueError("Delivery trainable safetensors inventory differs")
    parameter_keys = {f"model/{name}" for name in payload.parameter_names}
    auxiliary = decode_scientific_torch_state_v1(
        files["training_state.pt"],
        control_json=_control(config),
        descriptors=tuple(t for t in payload.tensors if t.key not in parameter_keys),
    )
    state = {f"model/{name}": value for name, value in parameters.items()} | auxiliary
    checkpoint = IrisScientificCheckpointBundleV1(
        config.checkpoint, save_tensors(dict(sorted(state.items())), metadata=_CHECKPOINT_META)
    )
    _verify(checkpoint)
    # Bind the human-facing policies back to the actual immutable graph identities.
    teacher_hash = domain_sha256(
        "iris/scientific-teacher-configuration/v1", config.teacher_configuration_json.encode()
    )
    probe_hash = domain_sha256(
        "iris/workspace-probe-heads/v1", config.probe_configuration_json.encode()
    )
    teacher = _json(config.teacher_configuration_json.encode())
    probes = _json(config.probe_configuration_json.encode())
    run_config = _json(payload.configuration_json.encode())
    if (
        bytes(state["model/configuration_identity"].tolist()).hex() != teacher_hash
        or bytes(state["model/probes.configuration_identity"].tolist()).hex() != probe_hash
        or teacher["probe_configuration_sha256"] != probe_hash
        or probes["vocabulary_sha256"] != probe_vocabulary_sha256(config.probe_vocabulary)
        or teacher["policy"] != run_config["teacher_policy"]
        or domain_sha256(
            "iris/scientific-optimizer-configuration/v1",
            canonical_json_bytes(
                {
                    "policy": run_config["optimizer_policy"],
                    "model_configuration_sha256": teacher_hash,
                    "parameter_names": payload.parameter_names,
                }
            ),
        )
        != run_config["optimizer_sha256"]
    ):
        raise ValueError("Delivery graph/label/training configuration binding differs")
    return checkpoint, config


def build_scientific_delivery_v1(
    run: IrisScientificRunV1,
    *,
    validation: IrisScientificValidationV1 | None = None,
) -> IrisScientificDeliveryBundleV1:
    """Snapshot all nine source files; no invented metric, model download or disk write."""
    try:
        checkpoint = build_scientific_checkpoint_v1(run)
    except Exception:
        raise IrisScientificDeliveryError("Scientific delivery could not be built") from None
    bundle = build_scientific_checkpoint_delivery_v1(checkpoint, context=run, validation=validation)
    if checkpoint.manifest.payload.completed_steps != run.completed_steps:
        raise IrisScientificDeliveryError("Live delivery source changed during construction")
    return bundle


def build_scientific_checkpoint_delivery_v1(
    checkpoint: IrisScientificCheckpointBundleV1,
    *,
    context: IrisScientificRunV1,
    validation: IrisScientificValidationV1 | None = None,
) -> IrisScientificDeliveryBundleV1:
    """Deliver a saved run state using compatible live graph/tokenizer metadata only.

    Never copy saved tensors into the context or substitute its current weights.
    The caller must independently establish which checkpoint is selected best.
    """
    run = context
    locked = False
    try:
        if type(run) is not IrisScientificRunV1:
            raise ValueError("Exact scientific metadata context required")
        manifest, state = _verify(checkpoint)
        run._validate_boundary()
        if (
            run._busy
            or manifest.payload.configuration_sha256 != run.configuration_sha256
            or manifest.payload.configuration_json != run._configuration_json
        ):
            raise ValueError("Idle matching scientific delivery context required")
        run._busy = locked = True
        model = run._optimizer._model
        tokenizer = _tokenizer_file(run)
        config = ScientificDeliveryConfigurationV1(
            checkpoint=manifest,
            teacher_configuration_json=model.configuration_json,
            probe_configuration_json=model.probes.configuration_json,
            probe_vocabulary=model.probes.vocabulary,
            tokenizer_json_sha256=bytes_sha256(tokenizer),
            tokenizer_semantics_sha256=tokenizer_semantics_sha256(run._backbone._tokenizer),
        )
        report = None
        if validation is not None:
            if type(validation) is not IrisScientificValidationV1:
                raise ValueError("Exact validation report required")
            report = ScientificDeliveryValidationV1.model_validate(asdict(validation))
        parameters = {name: state[f"model/{name}"] for name in manifest.payload.parameter_names}
        parameter_keys = {f"model/{name}" for name in parameters}
        files = _side_files(config) | {
            "tokenizer.json": tokenizer,
            "metrics.json": canonical_json_file_bytes(
                ScientificDeliveryMetricsV1(validation=report).model_dump(mode="json")
            ),
            "iris_model.safetensors": save_tensors(
                dict(sorted(parameters.items())), metadata=_MODEL_META
            ),
            "training_state.pt": encode_scientific_torch_state_v1(
                _control(config),
                {name: value for name, value in state.items() if name not in parameter_keys},
            ),
        }
        bundle = IrisScientificDeliveryBundleV1(_identity(files), tuple(sorted(files.items())))
        reconstructed, _ = _checked(bundle)
        if reconstructed != checkpoint:
            raise ValueError("Delivery snapshot changed during construction")
        run._validate_boundary()
        return bundle
    except Exception:
        raise IrisScientificDeliveryError("Scientific delivery could not be built") from None
    finally:
        if locked:
            run._busy = False


def restore_scientific_delivery_v1(
    bundle: IrisScientificDeliveryBundleV1, *, run: IrisScientificRunV1
) -> None:
    try:
        checkpoint, config = _checked(bundle)
        if type(run) is not IrisScientificRunV1:
            raise ValueError("Fresh scientific run required")
        model = run._optimizer._model
        if (
            config.teacher_configuration_json != model.configuration_json
            or config.probe_configuration_json != model.probes.configuration_json
            or config.probe_vocabulary != model.probes.vocabulary
            or dict(bundle.files)["tokenizer.json"] != _tokenizer_file(run)
            or config.tokenizer_semantics_sha256
            != tokenizer_semantics_sha256(run._backbone._tokenizer)
        ):
            raise ValueError("Delivery model/probe/tokenizer configuration differs")
        restore_scientific_checkpoint_v1(checkpoint, run=run)
    except Exception as error:
        # Preserve the existing restore's post-copy poison and failure evidence.
        raise IrisScientificDeliveryError("Scientific delivery could not be restored") from error


def _read_delivery_directory(
    path: Path, *, require_content_address: bool
) -> IrisScientificDeliveryBundleV1:
    """Same bounded canonical content checks for stores and source-named CLI copies."""
    try:
        if not isinstance(path, Path) or not path.name:
            raise ValueError("Explicit delivery directory required")
        path = path.absolute()
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {file.name for file in path.iterdir()} != set(_FILE_LIMITS):
            raise ValueError("Delivery directory inventory differs")
        for name in _FILE_LIMITS:
            _regular(path / name, directory=False)
        if sum((path / name).stat().st_size for name in _FILE_LIMITS) > _LIMIT:
            raise ValueError("Delivery directory exceeds its aggregate byte budget")
        files = {name: _bounded_read(path / name, limit) for name, limit in _FILE_LIMITS.items()}
        identity = path.name if require_content_address else _identity(files)
        bundle = IrisScientificDeliveryBundleV1(identity, tuple(sorted(files.items())))
        _checked(bundle)
        return bundle
    except Exception:
        raise IrisScientificDeliveryError("Scientific delivery could not be read") from None


def read_scientific_delivery_v1(path: Path) -> IrisScientificDeliveryBundleV1:
    """Read an immutable content-addressed delivery; friendly names remain invalid here."""
    return _read_delivery_directory(path, require_content_address=True)


def write_scientific_delivery_v1(
    bundle: IrisScientificDeliveryBundleV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 1024**3,
) -> Path:
    """Bounded exclusive nine-file publication; matching retry never trains or overwrites."""
    temporary: Path | None = None
    try:
        _checked(bundle)
        if (
            not isinstance(root, Path)
            or not root.name
            or type(maximum_storage_bytes) is not int
            or not 1 <= maximum_storage_bytes <= 16 * 1024**3
        ):
            raise ValueError("Explicit root and bounded delivery store budget required")
        root = root.absolute() / "scientific-delivery-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / bundle.sha256
        if target.exists():
            if read_scientific_delivery_v1(target) != bundle:
                raise ValueError("Existing delivery conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        required = sum(len(data) for _, data in bundle.files)
        if (
            occupied + required > maximum_storage_bytes
            or shutil.disk_usage(root).free < required + 1024**3
        ):
            raise ValueError("Delivery storage budget/free-space reserve exceeded")
        staging_candidate = root / f".iris-delivery-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary = staging_candidate  # Cleanup ownership begins only after exclusive creation.
        for name, data in bundle.files:
            _write_exact_file(temporary / name, data)
            if _bounded_read(temporary / name, len(data)) != data:
                raise ValueError("Staged delivery file differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_delivery_v1(target) != bundle:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificDeliveryError("Scientific delivery could not be published") from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            for name in _FILE_LIMITS:
                (temporary / name).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "IrisScientificDeliveryBundleV1",
    "IrisScientificDeliveryError",
    "build_scientific_checkpoint_delivery_v1",
    "build_scientific_delivery_v1",
    "read_scientific_delivery_v1",
    "restore_scientific_delivery_v1",
    "write_scientific_delivery_v1",
]
