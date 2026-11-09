"""Transactional state installation for already-composed request sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from re import fullmatch

import torch
from torch import nn

from iris.application.errors import IrisRequestCheckpointError
from iris.utils.hashing import module_state_sha256


@dataclass(frozen=True, slots=True)
class IrisRequestSidecarInstallResultV1:
    """Committed request-sidecar state transition without checkpoint metadata."""

    previous_state_sha256: str
    installed_state_sha256: str
    state_names: tuple[str, ...]
    tensor_count: int

    def __post_init__(self) -> None:
        for field, value in (
            ("previous_state_sha256", self.previous_state_sha256),
            ("installed_state_sha256", self.installed_state_sha256),
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                message = f"{field} must be a canonical SHA-256 digest"
                raise ValueError(message)
        if (
            type(self.state_names) is not tuple
            or not self.state_names
            or any(type(name) is not str or not name or "\0" in name for name in self.state_names)
            or len(set(self.state_names)) != len(self.state_names)
        ):
            message = "request sidecar state_names must be a nonempty unique tuple"
            raise ValueError(message)
        if type(self.tensor_count) is not int or self.tensor_count != len(self.state_names):
            message = "request sidecar tensor_count must equal the state-name count"
            raise ValueError(message)


def _require_modules(
    modules: object,
) -> tuple[tuple[str, nn.Module], ...]:
    if type(modules) is not tuple or not modules:
        message = "request sidecar installation requires a nonempty exact module tuple"
        raise TypeError(message)
    checked: list[tuple[str, nn.Module]] = []
    component_names: list[str] = []
    module_ids: list[int] = []
    for item in modules:
        if type(item) is not tuple or len(item) != 2:
            message = "request sidecar module entries must be exact name/module pairs"
            raise TypeError(message)
        name, module = item
        if type(name) is not str or not name or "\0" in name:
            message = "request sidecar component names must be nonempty exact strings"
            raise TypeError(message)
        if not isinstance(module, nn.Module):
            message = f"request sidecar component {name!r} must be a torch module"
            raise TypeError(message)
        component_names.append(name)
        module_ids.append(id(module))
        checked.append((name, module))
    if len(set(component_names)) != len(component_names):
        message = "request sidecar component names must be unique"
        raise ValueError(message)
    if len(set(module_ids)) != len(module_ids):
        message = "request sidecar components must have distinct identities"
        raise ValueError(message)
    return tuple(checked)


def _target_state(
    modules: tuple[tuple[str, nn.Module], ...],
) -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    for component_name, module in modules:
        if any(child.training for child in module.modules()):
            message = f"request sidecar component {component_name!r} must be in eval mode"
            raise ValueError(message)
        for state_name, tensor in module.state_dict(keep_vars=True).items():
            qualified_name = f"{component_name}.{state_name}"
            if qualified_name in targets:
                message = f"duplicate request sidecar state name {qualified_name!r}"
                raise ValueError(message)
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                message = f"request sidecar target {qualified_name!r} must be strided"
                raise TypeError(message)
            if tensor.is_inference():
                message = f"request sidecar target {qualified_name!r} cannot be inference-owned"
                raise ValueError(message)
            if tensor.grad is not None:
                message = f"request sidecar target {qualified_name!r} must be gradient-free"
                raise ValueError(message)
            if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
                message = f"request sidecar target {qualified_name!r} must be finite"
                raise ValueError(message)
            targets[qualified_name] = tensor
    if not targets:
        message = "request sidecar target contains no persisted tensor state"
        raise ValueError(message)
    return targets


def _require_source_state(
    state: object,
    *,
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if type(state) is not dict:
        message = "request sidecar source state must be an exact dict"
        raise TypeError(message)
    if tuple(state) != tuple(targets):
        message = "teacher checkpoint state topology does not match request sidecar"
        raise ValueError(message)
    checked: dict[str, torch.Tensor] = {}
    for name, target in targets.items():
        source = state[name]
        if type(source) is not torch.Tensor:
            message = f"request sidecar source {name!r} must be an exact tensor"
            raise TypeError(message)
        if (
            source.layout != torch.strided
            or source.device.type != "cpu"
            or source.requires_grad
            or source.grad_fn is not None
            or source.is_inference()
        ):
            message = f"request sidecar source {name!r} is not an owned CPU tensor"
            raise ValueError(message)
        if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
            message = f"request sidecar source {name!r} differs in shape or dtype"
            raise ValueError(message)
        if torch.is_floating_point(source) and not bool(torch.isfinite(source).all()):
            message = f"request sidecar source {name!r} must be finite"
            raise ValueError(message)
        checked[name] = source
    return checked


def _copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    """Single patchable mutation boundary used by rollback tests."""
    target.copy_(source)


def install_request_sidecar_state_v1(
    *,
    modules: tuple[tuple[str, nn.Module], ...],
    state: dict[str, torch.Tensor],
    expected_current_sha256: str,
    expected_installed_sha256: str | None = None,
) -> IrisRequestSidecarInstallResultV1:
    """Install exact teacher tensors into an eval request graph with rollback."""
    backups: dict[str, torch.Tensor] | None = None
    targets: dict[str, torch.Tensor] | None = None
    mutation_started = False
    try:
        if (
            type(expected_current_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", expected_current_sha256) is None
        ):
            message = "expected request sidecar state must be a canonical SHA-256"
            raise ValueError(message)
        if expected_installed_sha256 is not None and (
            type(expected_installed_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", expected_installed_sha256) is None
        ):
            message = "expected installed request sidecar state must be a canonical SHA-256"
            raise ValueError(message)
        checked_modules = _require_modules(modules)
        targets = _target_state(checked_modules)
        sources = _require_source_state(state, targets=targets)
        previous_sha256 = module_state_sha256(checked_modules)
        if previous_sha256 != expected_current_sha256:
            message = "request sidecar changed before checkpoint installation"
            raise ValueError(message)

        candidates: dict[str, torch.Tensor] = {}
        backups = {}
        with torch.inference_mode(False), torch.no_grad():
            for name, target in targets.items():
                candidates[name] = (
                    sources[name]
                    .detach()
                    .to(device=target.device, dtype=target.dtype)
                    .contiguous()
                    .clone()
                )
                backups[name] = target.detach().clone()

        with torch.no_grad():
            for name, target in targets.items():
                mutation_started = True
                _copy_tensor(target, candidates[name])
        for name, target in targets.items():
            if not torch.equal(target.detach(), candidates[name]):
                message = f"installed request sidecar tensor {name!r} changed unexpectedly"
                raise ValueError(message)
        installed_sha256 = module_state_sha256(checked_modules)
        if (
            expected_installed_sha256 is not None
            and installed_sha256 != expected_installed_sha256
        ):
            message = "installed request sidecar digest differs from checkpoint projection"
            raise ValueError(message)
        return IrisRequestSidecarInstallResultV1(
            previous_state_sha256=previous_sha256,
            installed_state_sha256=installed_sha256,
            state_names=tuple(targets),
            tensor_count=len(targets),
        )
    except BaseException as primary_error:
        if mutation_started and backups is not None and targets is not None:
            try:
                with torch.no_grad():
                    for name, backup in backups.items():
                        _copy_tensor(targets[name], backup)
                restored_modules = _require_modules(modules)
                if module_state_sha256(restored_modules) != expected_current_sha256:
                    message = "request sidecar rollback did not restore the prior state"
                    raise ValueError(message)
            except BaseException as rollback_error:
                failures = BaseExceptionGroup(
                    "Request sidecar installation and rollback both failed",
                    [primary_error, rollback_error],
                )
                raise IrisRequestCheckpointError(
                    "Request sidecar state is uncertain after failed checkpoint installation"
                ) from failures
        if isinstance(primary_error, IrisRequestCheckpointError):
            raise
        if isinstance(primary_error, Exception):
            message = "Teacher checkpoint state could not be installed into the request sidecar"
            raise IrisRequestCheckpointError(message) from primary_error
        raise


__all__ = ["IrisRequestSidecarInstallResultV1", "install_request_sidecar_state_v1"]
