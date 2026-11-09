"""Transactional request-sidecar state installation tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

import iris.application.request_sidecar_state as sidecar_state
from iris.application.errors import IrisRequestCheckpointError
from iris.application.request_sidecar_state import (
    IrisRequestSidecarInstallResultV1,
    install_request_sidecar_state_v1,
)
from iris.utils.hashing import module_state_sha256, named_module_state_sha256


def _modules() -> tuple[tuple[str, nn.Module], ...]:
    first = nn.Linear(3, 2)
    second = nn.LayerNorm(2)
    first.eval()
    second.eval()
    return (("first", first), ("second", second))


def _flat_state(
    modules: tuple[tuple[str, nn.Module], ...],
    *,
    offset: float,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for component_name, module in modules:
        for state_name, tensor in module.state_dict().items():
            result[f"{component_name}.{state_name}"] = tensor.detach().clone() + offset
    return result


def _snapshot(modules: tuple[tuple[str, nn.Module], ...]) -> dict[str, torch.Tensor]:
    return _flat_state(modules, offset=0.0)


def test_install_commits_exact_ordered_state_and_returns_both_hashes() -> None:
    torch.manual_seed(1907)
    modules = _modules()
    before = module_state_sha256(modules)
    source = _flat_state(modules, offset=0.25)

    result = install_request_sidecar_state_v1(
        modules=modules,
        state=source,
        expected_current_sha256=before,
        expected_installed_sha256=named_module_state_sha256(
            component_names=tuple(name for name, _ in modules),
            state=source,
        ),
    )

    assert type(result) is IrisRequestSidecarInstallResultV1
    assert result.previous_state_sha256 == before
    assert result.installed_state_sha256 == module_state_sha256(modules)
    assert result.installed_state_sha256 != before
    assert result.state_names == tuple(source)
    assert result.tensor_count == len(source)
    assert _snapshot(modules).keys() == source.keys()
    assert all(torch.equal(_snapshot(modules)[name], value) for name, value in source.items())


def test_named_state_projection_matches_live_module_framing_with_stateless_tail() -> None:
    modules = (*_modules(), ("stateless_tail", nn.Identity()))
    state = _snapshot(modules)

    projected = named_module_state_sha256(
        component_names=tuple(name for name, _ in modules),
        state=state,
    )

    assert projected == module_state_sha256(modules)


def test_wrong_expected_projection_rolls_back_every_tensor() -> None:
    modules = _modules()
    before_hash = module_state_sha256(modules)
    before = _snapshot(modules)

    with pytest.raises(IrisRequestCheckpointError, match="could not be installed") as raised:
        install_request_sidecar_state_v1(
            modules=modules,
            state=_flat_state(modules, offset=0.25),
            expected_current_sha256=before_hash,
            expected_installed_sha256="0" * 64,
        )

    assert "differs from checkpoint projection" in str(raised.value.__cause__)
    assert module_state_sha256(modules) == before_hash
    assert all(torch.equal(_snapshot(modules)[name], value) for name, value in before.items())


def test_topology_mismatch_fails_before_mutation() -> None:
    modules = _modules()
    before = _snapshot(modules)
    source = _flat_state(modules, offset=0.25)
    source["unexpected.weight"] = torch.ones(1)

    with pytest.raises(IrisRequestCheckpointError, match="could not be installed") as raised:
        install_request_sidecar_state_v1(
            modules=modules,
            state=source,
            expected_current_sha256=module_state_sha256(modules),
        )

    assert "topology" in str(raised.value.__cause__)
    assert all(torch.equal(_snapshot(modules)[name], value) for name, value in before.items())


def test_stale_current_hash_fails_before_mutation() -> None:
    modules = _modules()
    before = _snapshot(modules)

    with pytest.raises(IrisRequestCheckpointError, match="could not be installed") as raised:
        install_request_sidecar_state_v1(
            modules=modules,
            state=_flat_state(modules, offset=0.25),
            expected_current_sha256="0" * 64,
        )

    assert "changed before" in str(raised.value.__cause__)
    assert all(torch.equal(_snapshot(modules)[name], value) for name, value in before.items())


def test_mid_copy_failure_rolls_back_every_previously_written_tensor() -> None:
    modules = _modules()
    before_hash = module_state_sha256(modules)
    before = _snapshot(modules)
    original_copy = sidecar_state._copy_tensor
    calls = 0

    def fail_second_copy(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated second-tensor copy failure")
        original_copy(target, source)

    with (
        patch.object(sidecar_state, "_copy_tensor", side_effect=fail_second_copy),
        pytest.raises(IrisRequestCheckpointError, match="could not be installed") as raised,
    ):
        install_request_sidecar_state_v1(
            modules=modules,
            state=_flat_state(modules, offset=0.25),
            expected_current_sha256=before_hash,
        )

    assert "second-tensor" in str(raised.value.__cause__)
    assert module_state_sha256(modules) == before_hash
    assert all(torch.equal(_snapshot(modules)[name], value) for name, value in before.items())
    assert calls == 2 + len(before)
