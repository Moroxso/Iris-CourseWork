"""Real component effects of module controls, without claiming full request wiring."""

from dataclasses import fields, replace
from typing import Any

import pytest
import torch

from iris.cognitive.outputs import ModuleProposal
from iris.interventions import (
    BlockModuleRead,
    BlockModuleWrite,
    ForceModuleWinner,
    ScaleModuleBid,
    WorkspaceIntervention,
    WorkspaceInterventionError,
)
from iris.schemas import SlotAllocatorConfig, WorkspaceArbiterConfig
from iris.utils.hashing import module_state_sha256
from iris.workspace import SlotAllocator, WorkspaceArbiter, WorkspaceReader, WorkspaceWriter
from iris.workspace.outputs import WorkspaceArbiterOutput
from tests.unit.test_workspace_intervention_state import DTYPES, assert_same, clone, state_fixture


def proposal_fixture(dtype: torch.dtype = torch.float32) -> ModuleProposal:
    generator = torch.Generator(device="cpu").manual_seed(37)
    return ModuleProposal(
        content=torch.randn(2, 4, 3, dtype=dtype, generator=generator),
        key=torch.randn(2, 4, 3, dtype=dtype, generator=generator),
        bid=torch.tensor([[4, 3, 2, 1], [1, 2, 3, 4]], dtype=dtype),
        confidence=torch.full((2, 4), 0.5, dtype=dtype),
        ttl=torch.full((2, 4), 2.5, dtype=dtype),
    )


def components(
    dtype: torch.dtype = torch.float32,
    *,
    winners_per_cycle: int = 2,
) -> tuple[WorkspaceReader, WorkspaceArbiter, SlotAllocator, WorkspaceWriter]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        reader = WorkspaceReader(state_dim=3, num_modules=4, num_slots=4).to(dtype).eval()
        arbiter = (
            WorkspaceArbiter(
                WorkspaceArbiterConfig(
                    confidence_weight=0.0, novelty_weight=0.0, relevance_weight=0.0
                ),
                state_dim=3,
                num_modules=4,
                num_slots=4,
                winners_per_cycle=winners_per_cycle,
            )
            .to(dtype)
            .eval()
        )
        allocator = (
            SlotAllocator(
                SlotAllocatorConfig(
                    empty_bonus=1.0, allocation_age_penalty=0.1, retention_age_penalty=0.1
                ),
                state_dim=3,
                num_modules=4,
                num_slots=4,
                winners_per_cycle=winners_per_cycle,
            )
            .to(dtype)
            .eval()
        )
        writer = (
            WorkspaceWriter(
                state_dim=3, num_modules=4, num_slots=4, winners_per_cycle=winners_per_cycle
            )
            .to(dtype)
            .eval()
        )
    return reader, arbiter, allocator, writer


def arbitration_fixture(scores: torch.Tensor) -> WorkspaceArbiterOutput:
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :2]
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
    return WorkspaceArbiterOutput(
        scores=scores,
        novelty=torch.zeros_like(scores),
        relevance=torch.zeros_like(scores),
        soft_probabilities=torch.softmax(scores, dim=-1),
        selection_weights=mask.to(scores.dtype),
        winner_mask=mask,
        winner_indices=indices,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_read_zeroes_only_selected_module_values_and_attention(dtype: torch.dtype) -> None:
    state = state_fixture(dtype)
    reader, _, _, _ = components(dtype)
    with torch.no_grad():
        read = reader(state)
        before_values = read.workspace_reads.clone()
        before_attention = read.attention_weights.clone()
        changed = BlockModuleRead(1).apply_read(read, 1)
        assert not bool(changed.workspace_reads[:, 1].any())
        assert not bool(changed.attention_weights[:, 1].any())
        for module in (0, 2, 3):
            assert torch.equal(changed.workspace_reads[:, module], read.workspace_reads[:, module])
            assert torch.equal(
                changed.attention_weights[:, module], read.attention_weights[:, module]
            )
        twice = BlockModuleRead(3).apply_read(changed, 1)
        assert not bool(twice.workspace_reads[:, 1].any())
        assert not bool(twice.workspace_reads[:, 3].any())
        changed.workspace_reads.zero_()
        changed.attention_weights.zero_()
        assert torch.equal(read.workspace_reads, before_values)
        assert torch.equal(read.attention_weights, before_attention)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_bid_changes_actual_arbitration_without_touching_other_proposal_fields(
    dtype: torch.dtype,
) -> None:
    state, proposal = state_fixture(dtype), proposal_fixture(dtype)
    before = ModuleProposal(
        **{field.name: getattr(proposal, field.name).clone() for field in fields(proposal)}
    )
    _, arbiter, allocator, writer = components(dtype)
    goal = torch.ones(2, 3, dtype=dtype)
    with torch.no_grad():
        baseline = arbiter(proposal, state, goal)
        changed = ScaleModuleBid(0, 10.0).apply_proposal(proposal, 1)
        assert torch.equal(changed.bid[:, 0], before.bid[:, 0] * 10)
        assert torch.equal(changed.bid[:, 1:], before.bid[:, 1:])
        for field in fields(proposal):
            assert torch.equal(getattr(proposal, field.name), getattr(before, field.name))
            if field.name != "bid":
                assert torch.equal(getattr(changed, field.name), getattr(proposal, field.name))
            assert (
                getattr(changed, field.name).data_ptr() != getattr(proposal, field.name).data_ptr()
            )
        arbitration = arbiter(changed, state, goal)
        assert baseline.winner_indices[1].tolist() == [3, 2]
        assert arbitration.winner_indices[1].tolist() == [0, 3]
        allocation = allocator(changed, state, arbitration)
        output = writer(changed, state, allocation, goal)
        assert bool((output.source_module[1] == 0).any())


@pytest.mark.parametrize("dtype", DTYPES)
def test_forced_winners_pass_unchanged_allocator_and_writer_contracts(dtype: torch.dtype) -> None:
    state, proposal = state_fixture(dtype), proposal_fixture(dtype)
    reader, arbiter, allocator, writer = components(dtype)
    modules = (
        ("reader", reader),
        ("arbiter", arbiter),
        ("allocator", allocator),
        ("writer", writer),
    )
    digest = module_state_sha256(modules)
    goal = torch.ones(2, 3, dtype=dtype)
    with torch.no_grad():
        baseline = arbiter(proposal, state, goal)
        scores_before = baseline.scores.clone()
        first = ForceModuleWinner(0).apply_arbitration(baseline, 1)
        both = ForceModuleWinner(2).apply_arbitration(first, 1)
        assert both.winner_indices.tolist() == [[2, 0], [2, 0]]
        assert torch.equal(
            both.winner_indices,
            torch.argsort(both.scores, dim=-1, descending=True, stable=True)[:, :2],
        )
        assert torch.equal(both.soft_probabilities, torch.softmax(both.scores, dim=-1))
        assert torch.equal(both.novelty, baseline.novelty)
        assert torch.equal(both.relevance, baseline.relevance)
        allocation = allocator(proposal, state, both)
        output = writer(proposal, state, allocation, goal)
        for row in range(2):
            for rank in range(2):
                assert (
                    output.source_module[row, allocation.slot_indices[row, rank]]
                    == allocation.module_indices[row, rank]
                )
        first.scores.zero_()
        assert torch.equal(baseline.scores, scores_before)
    assert module_state_sha256(modules) == digest


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_write_restores_complete_prewrite_record_without_promoting_losers(
    dtype: torch.dtype,
) -> None:
    state, proposal = state_fixture(dtype), proposal_fixture(dtype)
    before = clone(state)
    _, arbiter, allocator, writer = components(dtype)
    goal = torch.ones(2, 3, dtype=dtype)
    with torch.no_grad():
        arbitration = arbiter(proposal, state, goal)
        allocation = allocator(proposal, state, arbitration)
        written = writer(proposal, state, allocation, goal)
        snapshot = clone(written)
        changed = BlockModuleWrite(0).apply_write(state, written, allocation, 1)
        selected = allocation.assignment_mask[:, 0]
        assert selected.sum(dim=-1).tolist() == [1, 0]
        for field in fields(state):
            mask = selected.unsqueeze(-1) if field.name == "slots" else selected
            expected = torch.where(mask, getattr(state, field.name), getattr(written, field.name))
            assert torch.equal(getattr(changed, field.name), expected)
        # Block every module sequentially against the same pre-write state.
        restored = changed
        for module in (1, 2, 3):
            restored = BlockModuleWrite(module).apply_write(state, restored, allocation, 1)
        assert_same(restored, state)
        changed.slots.zero_()
        assert_same(written, snapshot)
        assert_same(state, before)


def test_module_controls_expose_required_state_protocol_without_claiming_state_only_effect() -> (
    None
):
    state = state_fixture()
    for op in (
        BlockModuleRead(1),
        BlockModuleWrite(1),
        ScaleModuleBid(1, 2.0),
        ForceModuleWinner(1),
    ):
        assert isinstance(op, WorkspaceIntervention)
        output = op.apply(state, 1)
        assert_same(output, state)
        output.slots.zero_()
        assert bool(state.slots.any())


def test_phase_schedules_are_inactive_copies_and_active_effects() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    reader, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        read = reader(state)
        unchanged_read = BlockModuleRead(0, cycles=(2,)).apply_read(read, 1)
        assert torch.equal(unchanged_read.workspace_reads, read.workspace_reads)
        assert unchanged_read.workspace_reads.data_ptr() != read.workspace_reads.data_ptr()
        unchanged_proposal = ScaleModuleBid(0, 2.0, cycles=(2,)).apply_proposal(proposal, 1)
        assert torch.equal(unchanged_proposal.bid, proposal.bid)
        arbitration = arbiter(proposal, state, goal)
        unchanged_arbitration = ForceModuleWinner(0, cycles=(2,)).apply_arbitration(arbitration, 1)
        for field in fields(arbitration):
            assert torch.equal(
                getattr(unchanged_arbitration, field.name), getattr(arbitration, field.name)
            )
            assert (
                getattr(unchanged_arbitration, field.name).data_ptr()
                != getattr(arbitration, field.name).data_ptr()
            )
        allocation = allocator(proposal, state, arbitration)
        written = writer(proposal, state, allocation, goal)
        assert_same(
            BlockModuleWrite(0, cycles=(2,)).apply_write(state, written, allocation, 1), written
        )


@pytest.mark.parametrize("cycle", [0, -1, True, 1.0])
def test_phase_controls_reject_invalid_cycles(cycle: Any) -> None:
    state, proposal = state_fixture(), proposal_fixture()
    reader, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        arbitration = arbiter(proposal, state, goal)
        allocation = allocator(proposal, state, arbitration)
        written = writer(proposal, state, allocation, goal)
        with pytest.raises(ValueError):
            BlockModuleRead(0).apply_read(reader(state), cycle)
        with pytest.raises(ValueError):
            ScaleModuleBid(0, 2.0).apply_proposal(proposal, cycle)
        with pytest.raises(ValueError):
            ForceModuleWinner(0).apply_arbitration(arbitration, cycle)
        with pytest.raises(ValueError):
            BlockModuleWrite(0).apply_write(state, written, allocation, cycle)


def test_inactive_controls_still_reject_out_of_graph_modules() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    reader, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        arbitration = arbiter(proposal, state, goal)
        allocation = allocator(proposal, state, arbitration)
        written = writer(proposal, state, allocation, goal)
        with pytest.raises(WorkspaceInterventionError, match="module"):
            BlockModuleRead(4, cycles=(2,)).apply_read(reader(state), 1)
        with pytest.raises(WorkspaceInterventionError, match="module"):
            ScaleModuleBid(4, 2.0, cycles=(2,)).apply_proposal(proposal, 1)
        with pytest.raises(WorkspaceInterventionError, match="module"):
            ForceModuleWinner(4, cycles=(2,)).apply_arbitration(arbitration, 1)
        with pytest.raises(WorkspaceInterventionError, match="module"):
            BlockModuleWrite(4, cycles=(2,)).apply_write(state, written, allocation, 1)


@pytest.mark.parametrize("field", ["content", "key", "bid", "confidence", "ttl"])
def test_bid_control_rejects_nonfinite_proposals(field: str) -> None:
    proposal = proposal_fixture()
    getattr(proposal, field)[0, 0] = torch.nan
    with pytest.raises(WorkspaceInterventionError, match="finite"):
        ScaleModuleBid(0, 2.0).apply_proposal(proposal, 1)


def test_bid_scaling_overflow_fails_without_mutating_input() -> None:
    proposal = proposal_fixture(torch.float16)
    before = proposal.bid.clone()
    with pytest.raises(WorkspaceInterventionError, match="finite"):
        ScaleModuleBid(0, 1e100).apply_proposal(proposal, 1)
    assert torch.equal(proposal.bid, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forced_score_overflow_does_not_silently_relax_order_or_emit_infinity(
    dtype: torch.dtype,
) -> None:
    scores = torch.tensor([[torch.finfo(dtype).max, 0, 0, 0]], dtype=dtype)
    arbitration = arbitration_fixture(scores)
    with pytest.raises(WorkspaceInterventionError, match="finitely"):
        ForceModuleWinner(3).apply_arbitration(arbitration, 1)
    assert torch.equal(arbitration.scores, scores)


def test_force_rejects_training_gradients_and_preserves_tie_order() -> None:
    scores = torch.zeros(2, 4, requires_grad=True)
    arbitration = arbitration_fixture(scores)
    with pytest.raises(WorkspaceInterventionError, match="inference"):
        ForceModuleWinner(3).apply_arbitration(arbitration, 1)
    with torch.no_grad():
        forced = ForceModuleWinner(3).apply_arbitration(arbitration, 1)
    assert forced.winner_indices.tolist() == [[3, 0], [3, 0]]
    assert torch.equal(arbitration.scores, torch.zeros_like(scores))


@pytest.mark.parametrize(
    "field", ["scores", "novelty", "relevance", "soft_probabilities", "selection_weights"]
)
def test_force_rejects_nonfinite_arbitration(field: str) -> None:
    arbitration = arbitration_fixture(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    getattr(arbitration, field)[0, 0] = torch.nan
    with pytest.raises(WorkspaceInterventionError, match="finite"):
        ForceModuleWinner(3).apply_arbitration(arbitration, 1)


def test_force_rejects_inconsistent_hard_and_soft_selection() -> None:
    arbitration = arbitration_fixture(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    for changed in (
        replace(arbitration, winner_indices=torch.tensor([[1, 0]])),
        replace(arbitration, winner_mask=torch.ones_like(arbitration.winner_mask)),
        replace(arbitration, selection_weights=torch.zeros_like(arbitration.selection_weights)),
        replace(
            arbitration, soft_probabilities=torch.ones_like(arbitration.soft_probabilities) / 4
        ),
        replace(arbitration, scores=arbitration.scores.half()),
    ):
        with pytest.raises(WorkspaceInterventionError):
            ForceModuleWinner(3).apply_arbitration(changed, 1)


def test_read_control_rejects_inconsistent_attention_and_geometry() -> None:
    reader, _, _, _ = components()
    with torch.no_grad():
        read = reader(state_fixture())
    for changed in (
        replace(read, attention_weights=torch.zeros_like(read.attention_weights)),
        replace(read, attention_weights=-read.attention_weights),
        replace(read, attention_weights=read.attention_weights[:, :1]),
        replace(read, workspace_reads=read.workspace_reads.to("meta")),
    ):
        with pytest.raises(WorkspaceInterventionError):
            BlockModuleRead(0).apply_read(changed, 1)


@pytest.mark.parametrize(
    "field",
    ["allocation_scores", "retention_scores", "assignment_probabilities", "assignment_weights"],
)
def test_write_control_rejects_nonfinite_allocation(field: str) -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        allocation = allocator(proposal, state, arbiter(proposal, state, goal))
        written = writer(proposal, state, allocation, goal)
    getattr(allocation, field)[0, 0] = torch.nan
    with pytest.raises(WorkspaceInterventionError, match="finite"):
        BlockModuleWrite(0).apply_write(state, written, allocation, 1)


def test_write_control_rejects_inconsistent_assignment_and_state_geometry() -> None:
    state, proposal = state_fixture(), proposal_fixture()
    _, arbiter, allocator, writer = components()
    goal = torch.ones(2, 3)
    with torch.no_grad():
        allocation = allocator(proposal, state, arbiter(proposal, state, goal))
        written = writer(proposal, state, allocation, goal)
    for changed in (
        replace(allocation, assignment_mask=torch.zeros_like(allocation.assignment_mask)),
        replace(allocation, target_mask=~allocation.target_mask),
        replace(allocation, slot_indices=torch.full_like(allocation.slot_indices, 4)),
        replace(allocation, module_indices=allocation.module_indices[:, :1]),
        replace(allocation, assignment_weights=torch.zeros_like(allocation.assignment_weights)),
        replace(
            allocation,
            assignment_probabilities=torch.zeros_like(allocation.assignment_probabilities),
        ),
        replace(allocation, retention_scores=allocation.retention_scores[:, :1]),
    ):
        with pytest.raises(WorkspaceInterventionError):
            BlockModuleWrite(0).apply_write(state, written, changed, 1)
    with pytest.raises(WorkspaceInterventionError, match="geometry"):
        BlockModuleWrite(0).apply_write(state_fixture(torch.float64), written, allocation, 1)
    invalid = clone(state)
    invalid.source_module[0, 0] = 4
    with pytest.raises(WorkspaceInterventionError, match="domain"):
        BlockModuleWrite(0).apply_write(invalid, written, allocation, 1)
