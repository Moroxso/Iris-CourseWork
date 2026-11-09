"""Capacity changes actual target selection, not merely post-hoc slot erasure."""

from dataclasses import fields, replace
from typing import Any

import pytest
import torch

from iris.interventions import ReduceWorkspaceCapacity, WorkspaceInterventionError
from iris.interventions.capacity import restrict_allocation_capacity
from iris.workspace.outputs import SlotAllocatorOutput
from tests.unit.test_workspace_intervention_phases import components, proposal_fixture
from tests.unit.test_workspace_intervention_state import DTYPES, assert_same, state_fixture


def assert_allocation_same(left: SlotAllocatorOutput, right: SlotAllocatorOutput) -> None:
    for field in fields(left):
        assert torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("capacity", [1, 2, 3, 4])
@pytest.mark.parametrize("winners", [1, 2, 3, 4])
def test_capacity_composes_with_real_allocator_writer_and_preserves_their_contracts(
    dtype: torch.dtype, capacity: int, winners: int
) -> None:
    intervention = ReduceWorkspaceCapacity(capacity)
    state = intervention.apply(state_fixture(dtype), 1)
    proposal = proposal_fixture(dtype)
    _, arbiter, allocator, writer = components(dtype, winners_per_cycle=winners)
    goal = torch.ones(2, 3, dtype=dtype)
    with torch.no_grad():
        arbitration = arbiter(proposal, state, goal)
        baseline = allocator(proposal, state, arbitration)
        changed = intervention.apply_allocation(state, baseline, 1)
        allocator._validate_output(
            changed,
            arbitration=arbitration,
            batch_size=2,
            dtype=changed.allocation_scores.dtype,
            device=state.slots.device,
        )
        assert torch.equal(changed.module_indices, arbitration.winner_indices)
        assert bool((changed.slot_indices[:, : min(capacity, winners)] < capacity).all())
        if capacity < winners:
            assert bool((changed.slot_indices[:, capacity:] >= capacity).all())
        written = writer(proposal, state, changed, goal)
        final = intervention.apply(written, 1)
        assert final.slots.shape == state.slots.shape
        assert not bool(final.occupancy[:, capacity:].any())
        assert bool((final.source_module[:, capacity:] == -1).all())
        assert bool((final.write_count[:, capacity:] == 0).all())
        for row in range(2):
            for rank in range(min(capacity, winners)):
                slot = int(changed.slot_indices[row, rank])
                assert final.source_module[row, slot] == arbitration.winner_indices[row, rank]
                assert final.write_count[row, slot] == state.write_count[row, slot] + 1
        if capacity == 4:
            assert_allocation_same(changed, baseline)
        for field in fields(changed):
            assert (
                getattr(changed, field.name).data_ptr() != getattr(baseline, field.name).data_ptr()
            )


def test_first_rank_cannot_waste_usable_capacity_even_when_disabled_scores_are_higher() -> None:
    intervention = ReduceWorkspaceCapacity(1)
    state, proposal = intervention.apply(state_fixture(), 1), proposal_fixture()
    _, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        arbitration = arbiter(proposal, state, goal)
        baseline = allocator(proposal, state, arbitration)
        scores = baseline.allocation_scores.clone()
        scores[:, :, 0] = -1000
        scores[:, :, 1:] = 1000
        baseline = replace(baseline, allocation_scores=scores)
        changed = intervention.apply_allocation(state, baseline, 1)
        assert changed.slot_indices.tolist() == [[0, 1], [0, 1]]
        final = intervention.apply(writer(proposal, state, changed, goal), 1)
        assert final.source_module[:, 0].tolist() == [0, 3]
        assert final.occupancy.sum(dim=-1).tolist() == [1, 1]
        # Counterfactual scoring diagnostics are retained; only eligibility changed.
        assert torch.equal(changed.allocation_scores, scores)


def test_target_selection_prefers_empty_then_minimum_retention_inside_active_prefix() -> None:
    state = state_fixture()
    # With capacity3: row0 has empty slot1, row1 has empty slot2. Prefer occupied
    # slot0 next; deliberately give disabled slot3 the lowest retention of all.
    _, arbiter, allocator, _ = components()
    proposal, goal = proposal_fixture(), torch.ones(2, 3)
    with torch.no_grad():
        baseline = allocator(proposal, state, arbiter(proposal, state, goal))
        baseline = replace(
            baseline,
            retention_scores=torch.tensor([[1.0, 9.0, 5.0, -100.0], [1.0, 5.0, 9.0, -100.0]]),
        )
        changed = ReduceWorkspaceCapacity(3).apply_allocation(state, baseline, 1)
        assert changed.target_mask.tolist() == [
            [True, True, False, False],
            [True, False, True, False],
        ]


def test_equal_compatibility_uses_lowest_available_slot_index_per_rank() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, _ = components()
    with torch.no_grad():
        baseline = allocator(proposal, state, arbiter(proposal, state, torch.ones(2, 3)))
        baseline = replace(baseline, allocation_scores=torch.zeros_like(baseline.allocation_scores))
        changed = ReduceWorkspaceCapacity(2).apply_allocation(state, baseline, 1)
        assert changed.slot_indices.tolist() == [[0, 1], [0, 1]]
        assert changed.assignment_probabilities[0, changed.module_indices[0, 0]].tolist() == [
            0.5,
            0.5,
            0.0,
            0.0,
        ]


def test_inactive_and_full_capacity_are_exact_owned_allocation_copies() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, _ = components()
    with torch.no_grad():
        baseline = allocator(proposal, state, arbiter(proposal, state, torch.ones(2, 3)))
        for op in (ReduceWorkspaceCapacity(1, cycles=(2,)), ReduceWorkspaceCapacity(4)):
            changed = op.apply_allocation(state, baseline, 1)
            assert_allocation_same(changed, baseline)
            changed.assignment_mask.zero_()
            assert bool(baseline.assignment_mask.any())
            assert_same(op.apply(state, 1), state)


@pytest.mark.parametrize("capacity", [0, -1, 5, True, 1.0, None])
def test_internal_capacity_boundary_rejects_invalid_exact_dimensions(capacity: Any) -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, _ = components()
    with torch.no_grad():
        baseline = allocator(proposal, state, arbiter(proposal, state, torch.ones(2, 3)))
        with pytest.raises(WorkspaceInterventionError):
            restrict_allocation_capacity(state, baseline, capacity=capacity)


def test_reduced_target_rule_rejects_training_surrogate_and_malformed_allocation() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, _ = components()
    baseline = allocator(proposal, state, arbiter(proposal, state, torch.ones(2, 3)))
    assert baseline.allocation_scores.requires_grad
    with pytest.raises(WorkspaceInterventionError, match="no-grad"):
        ReduceWorkspaceCapacity(1).apply_allocation(state, baseline, 1)
    with torch.no_grad(), pytest.raises(WorkspaceInterventionError):
        ReduceWorkspaceCapacity(1).apply_allocation(
            state, replace(baseline, assignment_weights=baseline.assignment_weights * 0), 1
        )
