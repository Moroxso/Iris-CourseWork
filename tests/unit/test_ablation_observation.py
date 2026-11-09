"""Exact native observation ownership/budgets/phases; no neural claims."""

from dataclasses import fields
from typing import Any

import pytest
import torch

from iris.training.ablation_observation import (
    _AblationWorkspaceObservationSessionV1,
    ablation_workspace_observation_scalars_v1,
)
from iris.workspace.state import WorkspaceState

CONTEXT = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
GEOMETRY = (3, 2, 3, 4)


def state() -> WorkspaceState:
    result = WorkspaceState.empty(
        batch_size=1,
        num_slots=2,
        state_dim=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    result.slots.fill_(0.25)
    result.age.fill_(2)
    result.occupancy.fill_(True)
    result.source_module.fill_(1)
    result.confidence.fill_(0.5)
    result.ttl.fill_(3)
    result.write_count.fill_(4)
    return result


def session() -> _AblationWorkspaceObservationSessionV1:
    result = _AblationWorkspaceObservationSessionV1(
        cycles=3,
        slots=2,
        width=3,
        modules=4,
        maximum=108,
    )
    result.claim(context=CONTEXT, geometry=GEOMETRY)
    return result


def test_budget_counts_all_seven_fields_in_both_phases() -> None:
    assert ablation_workspace_observation_scalars_v1(cycles=3, slots=2, width=3) == 108
    assert ablation_workspace_observation_scalars_v1(cycles=3, slots=8, width=256) == 12576


@pytest.mark.parametrize("maximum", (True, 0, 107, 2_000_001, 108.0, "108"))
def test_exact_complete_budget_preflight(maximum: Any) -> None:
    with pytest.raises(ValueError):
        _AblationWorkspaceObservationSessionV1(
            cycles=3,
            slots=2,
            width=3,
            modules=4,
            maximum=maximum,
        )


@pytest.mark.parametrize("value", (True, 0, -1, 2.0, "2"))
def test_exact_positive_geometry(value: Any) -> None:
    with pytest.raises(ValueError):
        ablation_workspace_observation_scalars_v1(cycles=3, slots=value, width=3)


def test_both_snapshots_are_real_independent_seven_field_copies() -> None:
    before, written = state(), state()
    written.slots.add_(1)
    result = session()
    result.before_write(before, 1)
    result.native_written(before, written, 1)
    observed = result._observations[0]
    for actual, original in ((observed.before_write, before), (observed.native_written, written)):
        for f in fields(WorkspaceState):
            a, b = getattr(actual, f.name), getattr(original, f.name)
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
            assert not a.requires_grad and not a.is_inference()
    before.slots.add_(4)
    written.slots.add_(8)
    assert float(observed.before_write.slots[0, 0, 0]) == 0.25
    assert float(observed.native_written.slots[0, 0, 0]) == 1.25


@pytest.mark.parametrize("field", tuple(f.name for f in fields(WorkspaceState)))
@pytest.mark.parametrize("restore", (False, True))
def test_inplace_writer_drift_in_any_field_poisoned_even_if_values_restored(
    field: str,
    restore: bool,
) -> None:
    before, written = state(), state()
    result = session()
    result.before_write(before, 1)
    tensor = getattr(before, field)
    original = tensor.clone()
    if tensor.dtype == torch.bool:
        tensor.logical_not_()
    else:
        tensor.add_(0.1 if field == "confidence" else 1)
    if restore:
        tensor.copy_(original)
    with pytest.raises(ValueError, match="mutated"):
        result.native_written(before, written, 1)
    assert result._failed and not result._observations
    with pytest.raises(ValueError):
        result.before_write(state(), 1)


@pytest.mark.parametrize("kind", ("replay", "injected", "geometry", "subclass"))
def test_foreign_or_reused_collector_is_poisoned(kind: str) -> None:
    result = _AblationWorkspaceObservationSessionV1(
        cycles=3,
        slots=2,
        width=3,
        modules=4,
        maximum=108,
    )
    if kind == "replay":
        result.claim(context=CONTEXT, geometry=GEOMETRY)
    elif kind == "injected":
        object.__setattr__(result, "callback", lambda: None)
    elif kind == "subclass":

        class Derived(_AblationWorkspaceObservationSessionV1):
            pass

        result = Derived(cycles=3, slots=2, width=3, modules=4, maximum=108)
    with pytest.raises(ValueError):
        result.claim(context=CONTEXT, geometry=(1, 2, 3, 4) if kind == "geometry" else GEOMETRY)
    assert result._failed


@pytest.mark.parametrize("kind", ("unclaimed", "written-first", "wrong-cycle", "twice-before"))
def test_phase_order_and_one_shot_ownership(kind: str) -> None:
    result = session()
    before = state()
    if kind == "unclaimed":
        result = _AblationWorkspaceObservationSessionV1(
            cycles=3,
            slots=2,
            width=3,
            modules=4,
            maximum=108,
        )
    if kind == "twice-before":
        result.before_write(before, 1)
    with pytest.raises(ValueError):
        if kind == "written-first":
            result.native_written(before, state(), 1)
        else:
            result.before_write(before, 2 if kind == "wrong-cycle" else 1)
    assert result._failed
