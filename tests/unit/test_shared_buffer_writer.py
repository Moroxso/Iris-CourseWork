"""Tests for the gated append-only writer used by arm C."""

from dataclasses import replace

import pytest
import torch

from iris.cognitive import ModuleProposal
from iris.shared_buffer import SharedBufferState, SharedBufferWriter, SharedBufferWriterError
from iris.workspace import SlotAllocatorOutput, WorkspaceState, WorkspaceWriter


def _proposal(
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    offset: float = 0.0,
) -> ModuleProposal:
    torch.manual_seed(90210)
    content = (torch.randn(2, 3, 4, dtype=dtype) + offset).requires_grad_(requires_grad)
    key = torch.randn(2, 3, 4, dtype=dtype).requires_grad_(requires_grad)
    bid = torch.randn(2, 3, dtype=dtype).requires_grad_(requires_grad)
    confidence = torch.full((2, 3), 0.75, dtype=dtype).requires_grad_(requires_grad)
    ttl = torch.full((2, 3), 2.0, dtype=dtype).requires_grad_(requires_grad)
    return ModuleProposal(
        content=content,
        key=key,
        bid=bid,
        confidence=confidence,
        ttl=ttl,
    )


def _empty(*, dtype: torch.dtype = torch.float32) -> SharedBufferState:
    return SharedBufferState.empty(
        batch_size=2,
        state_dim=4,
        num_modules=3,
        cycles=3,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def _goal(
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=dtype,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"state_dim": 0, "num_modules": 3, "cycles": 3},
        {"state_dim": 4, "num_modules": True, "cycles": 3},
        {"state_dim": 4, "num_modules": 3, "cycles": -1},
    ],
)
def test_writer_rejects_invalid_dimensions(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SharedBufferWriter(**arguments)  # type: ignore[arg-type]


def test_first_append_adds_every_module_in_canonical_order() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    output = writer(_proposal(), _empty(), _goal(), cycle_number=1)

    assert output.entry_count == 3
    assert output.current_cycle == 1
    assert output.entries.shape == (2, 3, 4)
    assert torch.equal(output.source_module, torch.tensor([[0, 1, 2], [0, 1, 2]]))
    assert torch.equal(output.write_cycle, torch.ones(2, 3, dtype=torch.int64))
    assert bool(torch.isfinite(output.entries).all())


def test_append_is_equation_compatible_with_d_empty_slot_writer() -> None:
    proposal = _proposal()
    goal = _goal()
    shared_writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    workspace_writer = WorkspaceWriter(
        state_dim=4,
        num_modules=3,
        num_slots=3,
        winners_per_cycle=3,
    )
    workspace_writer.load_state_dict(shared_writer.state_dict(), strict=True)
    hard = torch.eye(3, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1).clone()
    hard_float = hard.to(dtype=torch.float32)
    allocation = SlotAllocatorOutput(
        allocation_scores=torch.zeros(2, 3, 3),
        retention_scores=torch.zeros(2, 3),
        target_mask=torch.ones(2, 3, dtype=torch.bool),
        assignment_mask=hard,
        assignment_probabilities=hard_float,
        assignment_weights=hard_float,
        module_indices=torch.arange(3).expand(2, -1).clone(),
        slot_indices=torch.arange(3).expand(2, -1).clone(),
    )
    workspace = WorkspaceState.empty(
        batch_size=2,
        num_slots=3,
        state_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    shared_output = shared_writer(proposal, _empty(), goal, cycle_number=1)
    workspace_output = workspace_writer(proposal, workspace, allocation, goal)

    assert torch.allclose(shared_output.entries, workspace_output.slots, atol=1e-6, rtol=1e-5)


def test_second_append_preserves_old_prefix_bitwise() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    first = writer(_proposal(), _empty(), _goal(), cycle_number=1)
    old_entries = first.entries.detach().clone()
    old_sources = first.source_module.clone()
    old_cycles = first.write_cycle.clone()
    second = writer(_proposal(offset=1.0), first, _goal(), cycle_number=2)

    assert second.entry_count == 6
    assert torch.equal(second.entries[:, :3], old_entries)
    assert torch.equal(second.source_module[:, :3], old_sources)
    assert torch.equal(second.write_cycle[:, :3], old_cycles)
    assert torch.equal(
        second.source_module,
        torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]]),
    )
    assert torch.equal(
        second.write_cycle,
        torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]]),
    )


def test_non_content_proposal_fields_do_not_change_buffer_values() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    first = _proposal()
    second = ModuleProposal(
        content=first.content.detach().clone(),
        key=first.key.detach().clone().mul(100),
        bid=first.bid.detach().clone().mul(-50),
        confidence=torch.full_like(first.confidence, 0.1),
        ttl=torch.full_like(first.ttl, 3.0),
    )
    first_output = writer(first, _empty(), _goal(), cycle_number=1)
    second_output = writer(second, _empty(), _goal(), cycle_number=1)
    assert torch.equal(first_output.entries, second_output.entries)


def test_goal_changes_gated_appended_values() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    proposal = _proposal()
    first = writer(proposal, _empty(), _goal(), cycle_number=1)
    second = writer(proposal, _empty(), _goal().neg(), cycle_number=1)
    assert not torch.equal(first.entries, second.entries)


def test_writer_preserves_graph_across_multiple_functional_appends() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    first_proposal = _proposal(requires_grad=True)
    second_proposal = _proposal(requires_grad=True, offset=0.5)
    goal = _goal(requires_grad=True)
    first = writer(first_proposal, _empty(), goal, cycle_number=1)
    second = writer(second_proposal, first, goal, cycle_number=2)
    weights = torch.linspace(0.1, 1.0, second.entries.numel()).reshape_as(second.entries)
    (second.entries * weights).sum().backward()

    assert first_proposal.content.grad is not None
    assert second_proposal.content.grad is not None
    assert goal.grad is not None
    assert bool(torch.isfinite(first_proposal.content.grad).all())
    assert bool(torch.isfinite(second_proposal.content.grad).all())
    assert bool(torch.isfinite(goal.grad).all())
    assert first_proposal.key.grad is None
    assert first_proposal.bid.grad is None
    assert first_proposal.confidence.grad is None
    assert first_proposal.ttl.grad is None
    assert all(parameter.grad is not None for parameter in writer.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_writer_accumulates_stably_in_fp32(dtype: torch.dtype) -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3).to(dtype=dtype)
    output = writer(
        _proposal(dtype=dtype, offset=100.0),
        _empty(dtype=dtype),
        _goal(dtype=dtype),
        cycle_number=1,
    )
    assert output.entries.dtype == dtype
    assert bool(torch.isfinite(output.entries).all())


def test_writer_materializes_inference_inputs_before_training_use() -> None:
    with torch.inference_mode():
        proposal = _proposal()
        state = _empty()
        goal = _goal()
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    output = writer(proposal, state, goal, cycle_number=1)
    assert proposal.content.is_inference()
    assert not output.entries.is_inference()
    assert output.entries.requires_grad


def test_writer_rejects_skipped_repeated_and_post_horizon_cycles() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    with pytest.raises(SharedBufferWriterError, match="next one-based"):
        writer(_proposal(), _empty(), _goal(), cycle_number=2)
    first = writer(_proposal(), _empty(), _goal(), cycle_number=1)
    with pytest.raises(SharedBufferWriterError, match="next one-based"):
        writer(_proposal(), first, _goal(), cycle_number=1)
    second = writer(_proposal(), first, _goal(), cycle_number=2)
    third = writer(_proposal(), second, _goal(), cycle_number=3)
    with pytest.raises(SharedBufferWriterError, match="every configured"):
        writer(_proposal(), third, _goal(), cycle_number=4)


def test_writer_rejects_bad_goal_proposal_state_and_parameter_boundaries() -> None:
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    with pytest.raises(SharedBufferWriterError, match="Goal state"):
        writer(_proposal(), _empty(), torch.zeros(2, 5), cycle_number=1)
    bad_proposal = replace(_proposal(), bid=torch.full((2, 3), torch.nan))
    with pytest.raises(SharedBufferWriterError, match="NaN or Inf"):
        writer(bad_proposal, _empty(), _goal(), cycle_number=1)
    malformed_state = replace(_empty(), write_cycle=torch.zeros(2, 1, dtype=torch.int64))
    with pytest.raises(SharedBufferWriterError, match="write_cycle"):
        writer(_proposal(), malformed_state, _goal(), cycle_number=1)
    writer = writer.to(dtype=torch.float64)
    with pytest.raises(SharedBufferWriterError, match="parameters"):
        writer(_proposal(), _empty(), _goal(), cycle_number=1)
