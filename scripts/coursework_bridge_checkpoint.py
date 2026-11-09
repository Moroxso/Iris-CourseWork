"""CPU academic bridge+AdamW checkpoint. Separate from scientific artifacts."""

from __future__ import annotations

import copy
import json
import os
import random
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

KIND = "iris-coursework-bridge-adamw-checkpoint-v1"
MAX_STATE_BYTES = 16 * 1024**2


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")


def _owner(module: torch.nn.Module, optimizer: torch.optim.AdamW) -> dict[str, torch.nn.Parameter]:
    params = dict(module.named_parameters())
    if not params or list(module.named_buffers()):
        raise ValueError("Nonempty parameter-only bridge required")
    if any(
        p.device.type != "cpu"
        or p.dtype != torch.float32
        or not p.requires_grad
        or p.grad is not None
        for p in params.values()
    ):
        raise ValueError("CPU fp32 bridge with cleared gradients required")
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("One exact AdamW parameter group required")
    owned = optimizer.param_groups[0]["params"]
    if len(owned) != len(params) or any(
        a is not b for a, b in zip(owned, params.values(), strict=True)
    ):
        raise ValueError("Optimizer must own every bridge parameter in named order")
    return params


def _group(optimizer: torch.optim.AdamW) -> dict[str, Any]:
    group = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
    if any(group.get(k, False) for k in ("amsgrad", "capturable", "differentiable", "maximize")):
        raise ValueError("Only ordinary non-AMSGrad CPU AdamW supported")
    return dict(json.loads(canonical(group)))


def _steps(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError("Completed steps must be an exact integer in 1..1000")
    return value


def _exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_tensors(
    tensors: dict[str, torch.Tensor],
    params: dict[str, torch.nn.Parameter],
    steps: int,
) -> None:
    expected = {"rng.torch"}
    for name, param in params.items():
        expected.add("model." + name)
        expected.update("adam." + name + "." + k for k in ("step", "exp_avg", "exp_avg_sq"))
        for key in ("model." + name, "adam." + name + ".exp_avg", "adam." + name + ".exp_avg_sq"):
            tensor = tensors.get(key)
            if tensor is None or tensor.shape != param.shape or tensor.dtype != torch.float32:
                raise ValueError("Checkpoint tensor geometry/dtype mismatch")
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Finite CPU checkpoint tensors required")
        if bool((tensors["adam." + name + ".exp_avg_sq"] < 0).any()):
            raise ValueError("Adam second moment must be nonnegative")
        step = tensors.get("adam." + name + ".step")
        if (
            step is None
            or step.shape != ()
            or step.dtype != torch.float32
            or step.device.type != "cpu"
            or float(step) != steps
        ):
            raise ValueError("Optimizer step does not match completed cursor")
    if set(tensors) != expected:
        raise ValueError("Checkpoint tensor names mismatch")
    rng = tensors["rng.torch"]
    if rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() > 65536:
        raise ValueError("Bounded CPU RNG tensor required")
    torch.Generator(device="cpu").set_state(rng)


def write_checkpoint(
    path: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    *,
    completed_steps: int,
    provenance: dict[str, Any],
) -> None:
    """Publish manifest last; retain partial outputs on failure, never overwrite."""
    params = _owner(module, optimizer)
    steps = _steps(completed_steps)
    if not path.is_absolute() or path.exists() or ".." in path.parts:
        raise ValueError("Fresh absolute checkpoint directory required")
    if not provenance:
        raise ValueError("Explicit provenance required")
    tensors = {"model." + n: p.detach().clone().contiguous() for n, p in params.items()}
    for name, param in params.items():
        state = optimizer.state.get(param)
        if state is None or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Complete initialized ordinary AdamW state required")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            tensors["adam." + name + "." + key] = state[key].detach().clone().contiguous()
    tensors["rng.torch"] = torch.get_rng_state().clone()
    _validate_tensors(tensors, params, steps)
    raw = save_tensors(tensors)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("Academic checkpoint size limit exceeded")
    manifest = dict(
        kind=KIND,
        version=1,
        prefix_layout="chat-boundary",
        scientific_acceptance=False,
        completed_steps=steps,
        parameter_names=list(params),
        provenance=provenance,
        optimizer_group=_group(optimizer),
        python_rng=random.getstate(),
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        module_training=module.training,
        state_file=dict(name="state.safetensors", bytes=len(raw), sha256=sha256(raw).hexdigest()),
    )
    metadata = canonical(manifest)
    if len(metadata) > 512 * 1024:
        raise ValueError("Bounded checkpoint metadata required")
    path.mkdir(parents=True)
    _exclusive(path / "state.safetensors", raw)
    _exclusive(path / "manifest.json", metadata + b"\n")


def restore_checkpoint(
    path: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    *,
    expected_provenance: dict[str, Any],
) -> int:
    """Validate before mutation; restore model, AdamW and Python/CPU RNG together."""
    params = _owner(module, optimizer)
    manifest_path = path / "manifest.json"
    if manifest_path.stat().st_size > 512 * 1024:
        raise ValueError("Bounded checkpoint metadata required")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    canonical(manifest)
    if (
        manifest.get("kind") != KIND
        or type(manifest.get("version")) is not int
        or manifest.get("version") != 1
        or manifest.get("prefix_layout") != "chat-boundary"
        or manifest.get("scientific_acceptance") is not False
    ):
        raise ValueError("Exact academic checkpoint kind required")
    steps = _steps(manifest.get("completed_steps"))
    if (
        manifest.get("parameter_names") != list(params)
        or canonical(manifest.get("provenance")) != canonical(expected_provenance)
        or canonical(manifest.get("optimizer_group")) != canonical(_group(optimizer))
    ):
        raise ValueError("Checkpoint provenance/optimizer ownership mismatch")
    if (
        manifest.get("torch_version") != torch.__version__
        or manifest.get("cpu_threads") != torch.get_num_threads()
        or manifest.get("deterministic_algorithms")
        is not torch.are_deterministic_algorithms_enabled()
        or manifest.get("deterministic_warn_only")
        is not torch.is_deterministic_algorithms_warn_only_enabled()
        or manifest.get("module_training") is not module.training
    ):
        raise ValueError("Checkpoint runtime policy mismatch")
    descriptor = manifest.get("state_file", {})
    size = descriptor.get("bytes")
    if (
        descriptor.get("name") != "state.safetensors"
        or type(size) is not int
        or not 1 <= size <= MAX_STATE_BYTES
    ):
        raise ValueError("Bounded fixed checkpoint state file required")
    state_path = path / "state.safetensors"
    if state_path.stat().st_size != size:
        raise ValueError("Checkpoint state byte length mismatch")
    raw = state_path.read_bytes()
    if sha256(raw).hexdigest() != descriptor.get("sha256"):
        raise ValueError("Checkpoint state SHA256 mismatch")
    tensors = load_tensors(raw)
    _validate_tensors(tensors, params, steps)
    python_rng = manifest.get("python_rng")
    if not isinstance(python_rng, list) or len(python_rng) != 3:
        raise ValueError("Python RNG state required")
    rng_state = (python_rng[0], tuple(python_rng[1]), python_rng[2])
    random.Random().setstate(rng_state)
    model_state = {name: tensors["model." + name] for name in params}
    current = optimizer.state_dict()
    state = {
        index: {
            key: tensors["adam." + name + "." + key] for key in ("step", "exp_avg", "exp_avg_sq")
        }
        for index, name in enumerate(params)
    }
    restored = dict(state=state, param_groups=current["param_groups"])
    backup_model = {n: p.detach().clone() for n, p in params.items()}
    backup_optimizer = copy.deepcopy(current)
    backup_cpu_rng, backup_python_rng = torch.get_rng_state(), random.getstate()
    try:
        module.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(restored)
        torch.set_rng_state(tensors["rng.torch"])
        random.setstate(rng_state)
    except BaseException:
        module.load_state_dict(backup_model, strict=True)
        optimizer.load_state_dict(backup_optimizer)
        torch.set_rng_state(backup_cpu_rng)
        random.setstate(backup_python_rng)
        raise
    return steps
