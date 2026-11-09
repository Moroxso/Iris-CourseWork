"""Immutable, bounded safetensors/JSON checkpoints and fresh-owner scientific resume."""

import json
import os
import random
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application.scientific_optimization import scientific_learning_rate_v1
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_checkpoint_v1 import (
    IrisScientificCheckpointManifestV1,
    ScientificCheckpointPayloadV1,
    ScientificRuntimeV1,
    ScientificTensorV1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.utils.seeds import RuntimeStateSnapshot

_META = {"format": "iris.scientific-checkpoint.v1"}
_DOMAIN = "iris/scientific-checkpoint/v1/payload"
_MAX_MANIFEST = 16 * 1024**2
_MAX_STATE = 512 * 1024**2


class IrisScientificCheckpointError(RuntimeError):
    """Scientific checkpoint could not be verified, published or restored."""


@dataclass(frozen=True, slots=True)
class IrisScientificCheckpointBundleV1:
    manifest: IrisScientificCheckpointManifestV1
    state_bytes: bytes


def _clone(value: torch.Tensor) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.is_inference()
        or value.dtype not in (torch.float32, torch.uint8)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("Unsupported scientific state tensor")
    return value.detach().to(device="cpu").contiguous().clone()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(content: bytes) -> Any:
    value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique)
    if canonical_json_bytes(value) != content:
        raise ValueError("Noncanonical scientific JSON")
    return value


def _runtime_record(value: RuntimeStateSnapshot, device: torch.device) -> ScientificRuntimeV1:
    return ScientificRuntimeV1(
        python_version=value.python_rng_state[0],
        python_state=value.python_rng_state[1],
        python_gaussian=value.python_rng_state[2],
        device=str(device),
        deterministic_algorithms=value.deterministic_algorithms,
        deterministic_warn_only=value.deterministic_warn_only,
        cudnn_deterministic=value.cudnn_deterministic,
        cudnn_benchmark=value.cudnn_benchmark,
    )


def _runtime(
    payload: ScientificCheckpointPayloadV1, tensors: dict[str, torch.Tensor]
) -> RuntimeStateSnapshot:
    record = payload.runtime
    python_state = (record.python_version, tuple(record.python_state), record.python_gaussian)
    random.Random().setstate(python_state)  # Validate without mutating the caller's RNG.
    cpu = tensors["runtime/cpu"]
    if cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Invalid Torch CPU RNG encoding")
    torch.Generator(device="cpu").set_state(cpu)
    cuda = tensors.get("runtime/cuda")
    device = None if record.device == "cpu" else torch.device(record.device)
    if cuda is not None and (cuda.dtype != torch.uint8 or cuda.ndim != 1):
        raise ValueError("Invalid CUDA RNG encoding")
    return RuntimeStateSnapshot(
        python_state,
        cpu.clone(),
        device,
        None if cuda is None else cuda.clone(),
        record.deterministic_algorithms,
        record.deterministic_warn_only,
        record.cudnn_deterministic,
        record.cudnn_benchmark,
    )


def _verify(
    bundle: IrisScientificCheckpointBundleV1,
) -> tuple[IrisScientificCheckpointManifestV1, dict[str, torch.Tensor]]:
    if (
        type(bundle) is not IrisScientificCheckpointBundleV1
        or type(bundle.state_bytes) is not bytes
    ):
        raise ValueError("Exact scientific checkpoint bundle required")
    manifest = IrisScientificCheckpointManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    payload = manifest.payload
    manifest_bytes = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    if len(manifest_bytes) > _MAX_MANIFEST or not 1 <= len(bundle.state_bytes) <= _MAX_STATE:
        raise ValueError("Scientific checkpoint byte limit exceeded")
    if (
        manifest.sha256
        != domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json")))
        or payload.state_size_bytes != len(bundle.state_bytes)
        or payload.state_sha256 != bytes_sha256(bundle.state_bytes)
        or payload.configuration_sha256
        != domain_sha256(
            "iris/scientific-run-configuration/v1", payload.configuration_json.encode("utf-8")
        )
    ):
        raise ValueError("Scientific checkpoint hashes differ")
    configuration = _json(payload.configuration_json.encode("utf-8"))
    if type(configuration) is not dict:
        raise ValueError("Invalid scientific configuration")
    tensors = load_tensors(bundle.state_bytes)
    if set(tensors) != {entry.key for entry in payload.tensors}:
        raise ValueError("Scientific tensor inventory differs")
    for entry in payload.tensors:
        value = tensors[entry.key]
        _clone(value)
        if (
            tuple(value.shape) != entry.shape
            or str(value.dtype).removeprefix("torch.") != entry.dtype
            or value.numel() * value.element_size() != entry.byte_length
            or (entry.key.endswith("/exp_avg_sq") and bool((value < 0).any()))
        ):
            raise ValueError("Scientific tensor descriptor/moment differs")
    if save_tensors(dict(sorted(tensors.items())), metadata=_META) != bundle.state_bytes:
        raise ValueError("Noncanonical scientific safetensors")
    _runtime(payload, tensors)
    return manifest, tensors


def build_scientific_checkpoint_v1(run: IrisScientificRunV1) -> IrisScientificCheckpointBundleV1:
    try:
        if type(run) is not IrisScientificRunV1 or run._busy:
            raise ValueError("Checkpoint requires an exact idle scientific run")
        run._validate_boundary()
        run._busy = True
        try:
            if run._backbone.checkpoint_runtime_sha256 != run._backbone_identity:
                raise ValueError("Frozen backbone identity differs")
            optimizer = run._optimizer
            tensors = {
                f"model/{name}": _clone(value)
                for name, value in optimizer._model.state_dict().items()
            }
            for index, (parameter, count) in enumerate(
                zip(optimizer._topology.parameters, optimizer._parameter_steps, strict=True)
            ):
                if count:
                    for name in ("exp_avg", "exp_avg_sq"):
                        tensors[f"adamw/{index}/{name}"] = _clone(
                            optimizer._optimizer.state[parameter][name]
                        )
            tensors["runtime/cpu"] = _clone(run._runtime.cpu_rng_state)
            if run._runtime.cuda_rng_state is not None:
                tensors["runtime/cuda"] = _clone(run._runtime.cuda_rng_state)
            if sum(value.numel() * value.element_size() for value in tensors.values()) > _MAX_STATE:
                raise ValueError("Scientific state exceeds its byte budget")
            state = save_tensors(dict(sorted(tensors.items())), metadata=_META)
            payload = ScientificCheckpointPayloadV1(
                configuration_json=run._configuration_json,
                configuration_sha256=run.configuration_sha256,
                completed_steps=run.completed_steps,
                examples_seen=run.examples_seen,
                target_tokens_seen=run.target_tokens_seen,
                history_sha256=run._history_sha256,
                parameter_names=optimizer._topology.names,
                parameter_steps=optimizer._parameter_steps,
                model_state_names=tuple(optimizer._model.state_dict()),
                runtime=_runtime_record(run._runtime, optimizer._device),
                tensors=tuple(
                    ScientificTensorV1(
                        key=key,
                        shape=tuple(value.shape),
                        dtype="float32" if value.dtype == torch.float32 else "uint8",
                        byte_length=value.numel() * value.element_size(),
                    )
                    for key, value in sorted(tensors.items())
                ),
                state_sha256=bytes_sha256(state),
                state_size_bytes=len(state),
            )
            bundle = IrisScientificCheckpointBundleV1(
                IrisScientificCheckpointManifestV1(
                    sha256=domain_sha256(
                        _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                    payload=payload,
                ),
                state,
            )
            _verify(bundle)
            run._validate_boundary()
            return bundle
        finally:
            run._busy = False
    except Exception:
        raise IrisScientificCheckpointError("Scientific checkpoint could not be built") from None


def _regular(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
    ):
        raise ValueError("Real non-reparse scientific checkpoint path required")


def _bounded_read(path: Path, limit: int) -> bytes:
    _regular(path, directory=False)
    if path.stat().st_size > limit:
        raise ValueError("Scientific checkpoint file budget exceeded")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Scientific checkpoint file grew past its budget")
    return value


def read_scientific_checkpoint_v1(path: Path) -> IrisScientificCheckpointBundleV1:
    try:
        _regular(path, directory=True)
        if {entry.name for entry in path.iterdir()} != {"manifest.json", "state.safetensors"}:
            raise ValueError("Scientific checkpoint directory inventory differs")
        content = _bounded_read(path / "manifest.json", _MAX_MANIFEST)
        if not content.endswith(b"\n"):
            raise ValueError("Scientific manifest requires one final newline")
        _json(content[:-1])
        manifest = IrisScientificCheckpointManifestV1.model_validate_json(content[:-1])
        if (
            canonical_json_file_bytes(manifest.model_dump(mode="json")) != content
            or path.name != manifest.sha256
        ):
            raise ValueError("Scientific manifest canonical identity differs")
        bundle = IrisScientificCheckpointBundleV1(
            manifest, _bounded_read(path / "state.safetensors", _MAX_STATE)
        )
        _verify(bundle)
        return bundle
    except Exception:
        raise IrisScientificCheckpointError("Scientific checkpoint could not be read") from None


def write_scientific_checkpoint_v1(bundle: IrisScientificCheckpointBundleV1, *, root: Path) -> Path:
    temporary: Path | None = None
    try:
        manifest, _ = _verify(bundle)
        if not isinstance(root, Path) or not root.name:
            raise ValueError("Explicit scientific checkpoint root required")
        root = root.absolute()
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / manifest.sha256
        if target.exists():
            if read_scientific_checkpoint_v1(target) != bundle:
                raise ValueError("Existing scientific checkpoint differs")
            return target
        staging_candidate = root / f".iris-scientific-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary = staging_candidate  # Cleanup ownership begins only after exclusive creation.
        _write_exact_file(temporary / "state.safetensors", bundle.state_bytes)
        _write_exact_file(
            temporary / "manifest.json", canonical_json_file_bytes(manifest.model_dump(mode="json"))
        )
        if _bounded_read(
            temporary / "state.safetensors", _MAX_STATE
        ) != bundle.state_bytes or _bounded_read(
            temporary / "manifest.json", _MAX_MANIFEST
        ) != canonical_json_file_bytes(manifest.model_dump(mode="json")):
            raise ValueError("Staged scientific checkpoint bytes differ")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_checkpoint_v1(target) != bundle:
                raise
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificCheckpointError(
            "Scientific checkpoint could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            # Only the two explicitly owned staging files; never recursive cleanup.
            try:
                _regular(temporary, directory=True)
                for name in ("manifest.json", "state.safetensors"):
                    (temporary / name).unlink(missing_ok=True)
                temporary.rmdir()
            except BaseException:
                raise IrisScientificCheckpointError("Scientific staging cleanup failed") from None


def restore_scientific_checkpoint_v1(
    bundle: IrisScientificCheckpointBundleV1, *, run: IrisScientificRunV1
) -> None:
    mutation_started = False
    locked = False
    try:
        if type(run) is not IrisScientificRunV1 or run._busy:
            raise ValueError("Fresh idle scientific run required")
        run._validate_boundary()
        if run.completed_steps or run.examples_seen or run.target_tokens_seen:
            raise ValueError("Cannot restore over a nonfresh run")
        run._busy = locked = True
        manifest, tensors = _verify(bundle)
        payload = manifest.payload
        optimizer = run._optimizer
        if (
            payload.configuration_sha256 != run.configuration_sha256
            or payload.configuration_json != run._configuration_json
            or payload.parameter_names != optimizer._topology.names
            or payload.completed_steps > optimizer._policy.max_optimizer_steps
            or payload.runtime.device != str(optimizer._device)
            or run._backbone.checkpoint_runtime_sha256 != run._backbone_identity
        ):
            raise ValueError("Scientific checkpoint run configuration differs")
        steps = payload.completed_steps
        policy = optimizer._policy
        if (
            not steps <= payload.examples_seen <= steps * policy.maximum_examples
            or not steps * policy.minimum_target_tokens
            <= payload.target_tokens_seen
            <= steps * policy.maximum_target_tokens
            or (not steps and payload.history_sha256 != run._history_sha256)
        ):
            raise ValueError("Scientific cursor does not match completed windows")
        targets = optimizer._model.state_dict()
        if tuple(targets) != payload.model_state_names:
            raise ValueError("Scientific model state inventory differs")
        candidates: dict[str, torch.Tensor] = {}
        for name, target in targets.items():
            source = tensors[f"model/{name}"]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError("Scientific model state shape/dtype differs")
            if name not in optimizer._topology.names and not torch.equal(
                source, target.detach().cpu()
            ):
                raise ValueError("Scientific model immutable buffer identity differs")
            candidates[name] = source.to(device=target.device).clone()
        moments: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for index, (parameter, count) in enumerate(
            zip(optimizer._topology.parameters, payload.parameter_steps, strict=True)
        ):
            if count:
                state = {"step": torch.tensor(float(count), dtype=torch.float32, device="cpu")}
                for name in ("exp_avg", "exp_avg_sq"):
                    source = tensors[f"adamw/{index}/{name}"]
                    if source.shape != parameter.shape or source.dtype != parameter.dtype:
                        raise ValueError("Scientific moment shape/dtype differs")
                    state[name] = source.to(device=parameter.device).clone()
                moments[parameter] = state
        runtime = _runtime(payload, tensors)
        if runtime.cuda_device is not None:
            assert runtime.cuda_rng_state is not None
            torch.Generator(device=runtime.cuda_device).set_state(runtime.cuda_rng_state)
        run._validate_boundary()
        mutation_started = True
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(candidates[name])
        optimizer._optimizer.state.update(moments)
        optimizer._parameter_steps = payload.parameter_steps
        optimizer._completed_steps = steps
        optimizer._current_lr = scientific_learning_rate_v1(policy, step=max(1, steps))
        optimizer._optimizer.param_groups[0]["lr"] = optimizer._current_lr
        optimizer._versions = tuple(p._version for p in optimizer._topology.parameters)
        run._completed_steps = steps
        run._examples_seen = payload.examples_seen
        run._target_tokens_seen = payload.target_tokens_seen
        run._history_sha256 = payload.history_sha256
        run._runtime = runtime
        run._validate_boundary()
        for name, target in targets.items():
            if not torch.equal(target.detach().cpu(), tensors[f"model/{name}"]):
                raise ValueError("Scientific restored tensor bytes differ")
    except BaseException as error:
        if mutation_started:
            run._poisoned = True
            run._optimizer._poisoned = True
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise IrisScientificCheckpointError("Scientific checkpoint could not be restored") from None
    finally:
        if locked:
            run._busy = False


__all__ = [
    "IrisScientificCheckpointBundleV1",
    "IrisScientificCheckpointError",
    "build_scientific_checkpoint_v1",
    "read_scientific_checkpoint_v1",
    "restore_scientific_checkpoint_v1",
    "write_scientific_checkpoint_v1",
]
