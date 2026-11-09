"""Ordered per-request sessions, exact graph binding and fail-closed lifecycle."""

from dataclasses import dataclass, fields, replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.cognitive.outputs import ModuleProposal
from iris.interventions import (
    WorkspaceInterventionError,
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    ZeroWorkspace,
    intervention_plan_sha256_v1,
)
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.workspace import WorkspaceState
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from tests.unit.test_workspace_intervention_phases import components, proposal_fixture
from tests.unit.test_workspace_intervention_state import DTYPES, assert_same, clone, state_fixture


@dataclass(frozen=True)
class Cycle:
    state: WorkspaceState
    read: WorkspaceReaderOutput
    proposal: ModuleProposal
    arbitration: WorkspaceArbiterOutput
    allocation: SlotAllocatorOutput
    written: WorkspaceState


def cycle_fixture(dtype: torch.dtype = torch.float32, *, winners: int = 2) -> Cycle:
    reader, arbiter, allocator, writer = components(dtype, winners_per_cycle=winners)
    state, proposal = state_fixture(dtype), proposal_fixture(dtype)
    goal = torch.ones(2, 3, dtype=dtype)
    with torch.no_grad():
        read = reader(state)
        arbitration = arbiter(proposal, state, goal)
        allocation = allocator(proposal, state, arbitration)
        written = writer(proposal, state, allocation, goal)
    return Cycle(state, read, proposal, arbitration, allocation, written)


def session(
    *specifications: dict[str, Any], cycles: int = 1, winners: int = 2
) -> WorkspaceInterventionSessionV1:
    return WorkspaceInterventionSessionV1(
        WorkspaceInterventionPlanV1(
            interventions=tuple(
                WorkspaceInterventionSpecV1.model_validate(spec) for spec in specifications
            )
        ),
        num_slots=4,
        num_modules=4,
        state_dim=3,
        cycles=cycles,
        winners_per_cycle=winners,
    )


def invoke(owner: WorkspaceInterventionSessionV1, data: Cycle, phase: int, cycle: int = 1) -> Any:
    if phase == 0:
        return owner.before_read(data.state, cycle)
    if phase == 1:
        return owner.after_read(data.read, cycle)
    if phase == 2:
        return owner.after_proposal(data.proposal, cycle)
    if phase == 3:
        return owner.after_arbitration(data.arbitration, cycle)
    if phase == 4:
        return owner.after_allocation(data.state, data.allocation, cycle)
    if phase == 5:
        return owner.after_write(data.state, data.written, data.allocation, cycle)
    raise AssertionError(phase)


def advance(owner: WorkspaceInterventionSessionV1, data: Cycle, phases: int) -> None:
    for phase in range(phases):
        invoke(owner, data, phase)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_plan_preserves_all_real_phase_outputs_and_finishes_once(dtype: torch.dtype) -> None:
    owner, data = session(cycles=3), cycle_fixture(dtype)
    expected: tuple[
        WorkspaceState
        | WorkspaceReaderOutput
        | ModuleProposal
        | WorkspaceArbiterOutput
        | SlotAllocatorOutput,
        ...,
    ] = (
        data.state,
        data.read,
        data.proposal,
        data.arbitration,
        data.allocation,
        data.written,
    )
    for cycle in range(1, 4):
        for phase, original in enumerate(expected):
            output = invoke(owner, data, phase, cycle)
            for field in fields(original):
                assert torch.equal(getattr(output, field.name), getattr(original, field.name))
    owner.finish()
    with pytest.raises(WorkspaceInterventionError, match="exactly once"):
        owner.finish()
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        owner.before_read(data.state, 1)


@pytest.mark.parametrize("phase", range(1, 6))
def test_skipping_a_phase_poisons_session_instead_of_silently_omitting_an_effect(
    phase: int,
) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phase - 1)
    with pytest.raises(WorkspaceInterventionError, match="phase/cycle"):
        invoke(owner, data, phase)
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        invoke(owner, data, phase - 1)


@pytest.mark.parametrize("phase", range(6))
def test_repeating_a_phase_poisons_session(phase: int) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phase + 1)
    with pytest.raises(WorkspaceInterventionError, match="phase/cycle"):
        invoke(owner, data, phase)
    with pytest.raises(WorkspaceInterventionError):
        owner.finish()


@pytest.mark.parametrize("cycle", [0, -1, 2, True, 1.0, None])
def test_cycle_must_be_exact_expected_one_based_integer(cycle: Any) -> None:
    owner, data = session(cycles=2), cycle_fixture()
    with pytest.raises(WorkspaceInterventionError, match="phase/cycle"):
        owner.before_read(data.state, cycle)
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        owner.before_read(data.state, 1)


@pytest.mark.parametrize("phases", range(6))
def test_incomplete_finish_closes_session(phases: int) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phases)
    with pytest.raises(WorkspaceInterventionError, match="exactly once"):
        owner.finish()
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        invoke(owner, data, phases)


@pytest.mark.parametrize("phase", range(6))
def test_invalid_phase_data_poisons_session_even_for_an_empty_plan(phase: int) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phase)
    if phase == 0:
        data.state.source_module[0, 0] = 4
    elif phase == 1:
        data.read.workspace_reads[0, 0, 0] = torch.nan
    elif phase == 2:
        data.proposal.ttl[0, 0] = 0
    elif phase == 3:
        data.arbitration.scores[0, 0] = torch.inf
    elif phase == 4:
        data.allocation.assignment_weights.zero_()
    else:
        data.written.source_module[0, 0] = 4
    with pytest.raises(WorkspaceInterventionError):
        invoke(owner, data, phase)
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        invoke(owner, cycle_fixture(), phase)


@pytest.mark.parametrize("phase", range(6))
def test_otherwise_valid_foreign_batch_is_rejected_at_every_phase(phase: int) -> None:
    owner, data = session(cycles=2), cycle_fixture()
    advance(owner, data, 6)
    # Cycle2 cannot change the established batch, even if every local tensor is valid.
    smaller = Cycle(
        **{
            field.name: type(value)(
                **{part.name: getattr(value, part.name)[:1].clone() for part in fields(value)}
            )
            for field in fields(data)
            for value in (getattr(data, field.name),)
        }
    )
    for previous in range(phase):
        invoke(owner, data, previous, 2)
    with pytest.raises(WorkspaceInterventionError):
        invoke(owner, smaller, phase, 2)


@pytest.mark.parametrize("phase", [1, 2, 3, 4, 5])
def test_otherwise_valid_foreign_dtype_is_rejected(phase: int) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phase)
    with pytest.raises(WorkspaceInterventionError):
        invoke(owner, cycle_fixture(torch.float64), phase)


@pytest.mark.parametrize("phase", [3, 4])
def test_otherwise_valid_wrong_winner_count_is_rejected(phase: int) -> None:
    owner, data = session(), cycle_fixture()
    advance(owner, data, phase)
    with pytest.raises(WorkspaceInterventionError, match="bound"):
        invoke(owner, cycle_fixture(winners=3), phase)


def test_keyboard_interrupt_poisoning_does_not_swallow_the_interrupt() -> None:
    owner, data = session({"kind": "zero_workspace"}), cycle_fixture()
    advance(owner, data, 5)
    with (
        patch.object(ZeroWorkspace, "apply", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        invoke(owner, data, 5)
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        invoke(owner, data, 5)
    fresh = session({"kind": "zero_workspace"})
    advance(fresh, data, 5)
    assert not bool(fresh.after_write(data.state, data.written, data.allocation, 1).occupancy.any())
    fresh.finish()


def test_caller_plan_mutation_cannot_change_bound_intent_or_identity() -> None:
    spec = WorkspaceInterventionSpecV1(kind="zero_slot", slot=0)
    plan = WorkspaceInterventionPlanV1(interventions=(spec,))
    expected_hash = intervention_plan_sha256_v1(plan)
    owner = WorkspaceInterventionSessionV1(
        plan, num_slots=4, num_modules=4, state_dim=3, cycles=1, winners_per_cycle=2
    )
    object.__setattr__(plan.interventions[0], "slot", 3)
    object.__setattr__(plan, "interventions", ())
    data = cycle_fixture()
    advance(owner, data, 5)
    changed = owner.after_write(data.state, data.written, data.allocation, 1)
    assert owner.plan_sha256 == expected_hash
    assert not bool(changed.occupancy[:, 0].any())
    assert torch.equal(changed.slots[:, 3], data.written.slots[:, 3])
    owner.finish()


def test_minimum_active_capacity_wins_over_later_replacement_and_looser_capacity() -> None:
    owner = session(
        {"kind": "reduce_workspace_capacity", "capacity": 1, "cycles": (1,)},
        {"kind": "reduce_workspace_capacity", "capacity": 3},
        {
            "kind": "replace_slot",
            "slot": 2,
            "values": (9.0, 8.0, 7.0),
            "source_module": 2,
            "confidence": 0.75,
            "ttl": 4.0,
        },
        cycles=2,
    )
    data = cycle_fixture()
    reader, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        for cycle, capacity in ((1, 1), (2, 3)):
            state = owner.before_read(data.state, cycle)
            assert not bool(state.occupancy[:, capacity:].any())
            owner.after_read(reader(state), cycle)
            proposal = owner.after_proposal(data.proposal, cycle)
            arbitration = owner.after_arbitration(arbiter(proposal, state, goal), cycle)
            allocation = owner.after_allocation(
                state, allocator(proposal, state, arbitration), cycle
            )
            assert bool((allocation.slot_indices[:, : min(capacity, 2)] < capacity).all())
            changed = owner.after_write(
                state, writer(proposal, state, allocation, goal), allocation, cycle
            )
            assert not bool(changed.occupancy[:, capacity:].any())
            if cycle == 2:
                assert bool(changed.occupancy[:, 2].all())
                assert torch.equal(
                    changed.slots[:, 2], torch.tensor([[9.0, 8.0, 7.0]]).expand(2, -1)
                )
    owner.finish()


def test_freeze_restores_private_snapshot_before_read_after_decay_and_after_write() -> None:
    owner, data = session({"kind": "freeze_slot", "slot": 0}, cycles=2), cycle_fixture()
    advance(owner, data, 5)
    first = owner.after_write(data.state, data.written, data.allocation, 1)
    expected = clone(first)
    first.slots.zero_()
    decayed = replace(data.state, slots=data.state.slots * 0.1, ttl=data.state.ttl * 0.1)
    restored = owner.before_read(decayed, 2)
    for field in fields(restored):
        assert torch.equal(getattr(restored, field.name)[:, 0], getattr(expected, field.name)[:, 0])
    for phase in range(1, 5):
        invoke(owner, data, phase, 2)
    second = owner.after_write(data.state, data.written, data.allocation, 2)
    for field in fields(second):
        assert torch.equal(getattr(second, field.name)[:, 0], getattr(expected, field.name)[:, 0])
    assert_same(data.state, state_fixture())
    owner.finish()
