"""Contract tests for deterministic multi-winner workspace slot allocation."""

from dataclasses import FrozenInstanceError, fields, replace
from math import sqrt
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional

from iris.cognitive import ModuleProposal
from iris.schemas import SlotAllocatorConfig
from iris.workspace import (
    SlotAllocator,
    SlotAllocatorError,
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceState,
)


def allocator_config(
    *,
    empty_bonus: float = 1.0,
    allocation_age_penalty: float = 0.1,
    retention_age_penalty: float = 0.1,
) -> SlotAllocatorConfig:
    """Build an explicit allocator coefficient contract for a test."""
    return SlotAllocatorConfig(
        empty_bonus=empty_bonus,
        allocation_age_penalty=allocation_age_penalty,
        retention_age_penalty=retention_age_penalty,
    )


def make_allocator(
    *,
    state_dim: int = 4,
    num_modules: int = 3,
    num_slots: int = 4,
    winners_per_cycle: int = 2,
    empty_bonus: float = 1.0,
    allocation_age_penalty: float = 0.1,
    retention_age_penalty: float = 0.1,
) -> SlotAllocator:
    """Create a small deterministic allocator."""
    return SlotAllocator(
        allocator_config(
            empty_bonus=empty_bonus,
            allocation_age_penalty=allocation_age_penalty,
            retention_age_penalty=retention_age_penalty,
        ),
        state_dim=state_dim,
        num_modules=num_modules,
        num_slots=num_slots,
        winners_per_cycle=winners_per_cycle,
    )


def make_proposal(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    key: torch.Tensor | None = None,
) -> ModuleProposal:
    """Create a valid proposal with deterministic keys."""
    if key is None:
        generator = torch.Generator().manual_seed(99991)
        key = torch.randn(
            batch_size,
            num_modules,
            state_dim,
            generator=generator,
            dtype=dtype,
        )
    scalar_shape = (batch_size, num_modules)
    return ModuleProposal(
        content=torch.zeros_like(key),
        key=key,
        bid=torch.zeros(scalar_shape, dtype=dtype, device=key.device),
        confidence=torch.full(scalar_shape, 0.5, dtype=dtype, device=key.device),
        ttl=torch.full(scalar_shape, 2.0, dtype=dtype, device=key.device),
    )


def make_workspace(
    *,
    batch_size: int = 2,
    num_slots: int = 4,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    slots: torch.Tensor | None = None,
    occupancy: torch.Tensor | None = None,
    age: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> WorkspaceState:
    """Create a valid workspace and canonical metadata for empty slots."""
    if slots is None:
        slots = torch.zeros(batch_size, num_slots, state_dim, dtype=dtype)
    device = slots.device
    if occupancy is None:
        occupancy = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=device)
    if age is None:
        age = torch.zeros(batch_size, num_slots, dtype=torch.int64, device=device)
    if confidence is None:
        confidence = occupancy.to(dtype=dtype) * 0.5
    state = WorkspaceState.empty(
        batch_size=batch_size,
        num_slots=num_slots,
        state_dim=state_dim,
        dtype=dtype,
        device=device,
    )
    return replace(
        state,
        slots=slots,
        occupancy=occupancy,
        age=age,
        source_module=torch.where(
            occupancy,
            torch.zeros_like(state.source_module),
            state.source_module,
        ),
        confidence=confidence,
        ttl=occupancy.to(dtype=dtype) * 2.0,
        write_count=occupancy.to(dtype=torch.int64),
    )


def make_arbitration(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    winners_per_cycle: int = 2,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    scores: torch.Tensor | None = None,
    straight_through: bool = False,
) -> WorkspaceArbiterOutput:
    """Create a canonical arbiter output, optionally retaining its score graph."""
    if device is None:
        device = torch.device("cpu")
    if scores is None:
        base = torch.arange(num_modules, 0, -1, dtype=dtype, device=device)
        scores = base.expand(batch_size, -1).clone()
    winner_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[
        :, :winners_per_cycle
    ]
    winner_mask = torch.zeros(
        batch_size,
        num_modules,
        dtype=torch.bool,
        device=device,
    ).scatter(
        dim=1,
        index=winner_indices,
        src=torch.ones_like(winner_indices, dtype=torch.bool),
    )
    probabilities = torch.softmax(scores, dim=-1)
    hard = winner_mask.to(dtype=dtype)
    selection_weights = hard
    if straight_through:
        selection_weights = hard + (probabilities - probabilities.detach())
    zeros = torch.zeros_like(scores)
    return WorkspaceArbiterOutput(
        scores=scores,
        novelty=zeros,
        relevance=zeros,
        soft_probabilities=probabilities,
        selection_weights=selection_weights,
        winner_mask=winner_mask,
        winner_indices=winner_indices,
    )


def test_has_one_learned_bias_free_slot_projection_and_explicit_coefficients() -> None:
    config = allocator_config(
        empty_bonus=1.25,
        allocation_age_penalty=0.2,
        retention_age_penalty=0.3,
    )
    model = SlotAllocator(
        config,
        state_dim=5,
        num_modules=4,
        num_slots=3,
        winners_per_cycle=2,
    )

    assert set(dict(model.named_children())) == {"slot_projection"}
    assert isinstance(model.slot_projection, nn.Linear)
    assert model.slot_projection.in_features == 5
    assert model.slot_projection.out_features == 5
    assert model.slot_projection.bias is None
    assert model.empty_bonus == config.empty_bonus
    assert model.allocation_age_penalty == config.allocation_age_penalty
    assert model.retention_age_penalty == config.retention_age_penalty


def test_matches_complete_dense_score_and_retention_equations() -> None:
    model = make_allocator(
        state_dim=2,
        num_modules=3,
        num_slots=3,
        empty_bonus=1.5,
        allocation_age_penalty=0.25,
        retention_age_penalty=0.4,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.slot_projection.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))
    keys = torch.tensor(
        [[[1.0, 2.0], [-2.0, 0.5], [0.25, -1.5]]],
        dtype=torch.float64,
    )
    slots = torch.tensor(
        [[[2.0, -1.0], [1e100, -1e100], [-0.5, 3.0]]],
        dtype=torch.float64,
    )
    occupancy = torch.tensor([[True, False, True]])
    age = torch.tensor([[2, 0, 3]])
    confidence = torch.tensor([[0.9, 0.0, 0.2]], dtype=torch.float64)
    proposal = make_proposal(
        batch_size=1,
        num_modules=3,
        state_dim=2,
        dtype=torch.float64,
        key=keys,
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        dtype=torch.float64,
        slots=slots,
        occupancy=occupancy,
        age=age,
        confidence=confidence,
    )

    output = model(
        proposal,
        workspace,
        make_arbitration(
            batch_size=1,
            num_modules=3,
            winners_per_cycle=2,
            dtype=torch.float64,
        ),
    )

    canonical_slots = slots * occupancy.unsqueeze(-1)
    projected_slots = functional.linear(canonical_slots, model.slot_projection.weight)
    compatibility = torch.einsum("bmd,bkd->bmk", keys / sqrt(2), projected_slots)
    expected_scores = (
        compatibility + 1.5 * (~occupancy).unsqueeze(1).to(torch.float64) - 0.25 * age.unsqueeze(1)
    )
    expected_retention = confidence - 0.4 * age.to(torch.float64)
    torch.testing.assert_close(output.allocation_scores, expected_scores, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output.retention_scores, expected_retention, rtol=0, atol=0)
    assert output.allocation_scores.shape == (1, 3, 3)


def test_unoccupied_stale_content_is_ignored_exactly() -> None:
    model = make_allocator(state_dim=3, num_slots=3)
    occupancy = torch.tensor([[True, False, True], [False, True, False]])
    clean_slots = torch.tensor(
        [
            [[1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [-2.0, 0.5, 1.0]],
            [[0.0, 0.0, 0.0], [1.5, -1.0, 2.0], [0.0, 0.0, 0.0]],
        ]
    )
    stale_slots = clean_slots.clone()
    stale_slots[~occupancy] = torch.tensor([30_000.0, -40_000.0, 50_000.0])
    proposal = make_proposal(state_dim=3)
    arbitration = make_arbitration()

    clean = model(
        proposal,
        make_workspace(num_slots=3, state_dim=3, slots=clean_slots, occupancy=occupancy),
        arbitration,
    )
    stale = model(
        proposal,
        make_workspace(num_slots=3, state_dim=3, slots=stale_slots, occupancy=occupancy),
        arbitration,
    )

    torch.testing.assert_close(stale.allocation_scores, clean.allocation_scores, rtol=0, atol=0)
    torch.testing.assert_close(stale.slot_indices, clean.slot_indices, rtol=0, atol=0)


def test_unoccupied_stale_age_and_confidence_are_canonicalized_to_zero() -> None:
    model = make_allocator(state_dim=2, num_slots=3)
    occupancy = torch.tensor([[True, False, True]])
    proposal = make_proposal(batch_size=1, state_dim=2)
    arbitration = make_arbitration(batch_size=1)
    clean = make_workspace(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        occupancy=occupancy,
        age=torch.tensor([[2, 0, 4]]),
        confidence=torch.tensor([[0.7, 0.0, 0.2]]),
    )
    stale = replace(
        clean,
        age=torch.tensor([[2, 10_000, 4]]),
        confidence=torch.tensor([[0.7, 1.0, 0.2]]),
    )

    clean_output = model(proposal, clean, arbitration)
    stale_output = model(proposal, stale, arbitration)

    torch.testing.assert_close(stale_output.allocation_scores, clean_output.allocation_scores)
    torch.testing.assert_close(stale_output.retention_scores, clean_output.retention_scores)
    torch.testing.assert_close(stale_output.target_mask, clean_output.target_mask)


def test_all_empty_target_set_uses_lowest_stable_slot_ids() -> None:
    model = make_allocator(num_slots=5, winners_per_cycle=3)

    output = model(
        make_proposal(),
        make_workspace(num_slots=5),
        make_arbitration(winners_per_cycle=3),
    )

    torch.testing.assert_close(
        output.target_mask,
        torch.tensor([[True, True, True, False, False]]).expand(2, -1),
        rtol=0,
        atol=0,
    )


def test_partial_target_set_takes_all_empty_then_lowest_retention_slots() -> None:
    model = make_allocator(
        num_modules=4,
        num_slots=5,
        winners_per_cycle=3,
        retention_age_penalty=0.2,
    )
    occupancy = torch.tensor([[True, False, True, True, False]])
    age = torch.tensor([[0, 0, 1, 4, 0]])
    confidence = torch.tensor([[0.1, 0.0, 0.8, 0.9, 0.0]])

    output = model(
        make_proposal(batch_size=1, num_modules=4),
        make_workspace(
            batch_size=1,
            num_slots=5,
            occupancy=occupancy,
            age=age,
            confidence=confidence,
        ),
        make_arbitration(
            batch_size=1,
            num_modules=4,
            winners_per_cycle=3,
        ),
    )

    # Empty slots 1 and 4 are reserved first; slot 3 has minimum occupied retention.
    torch.testing.assert_close(
        output.target_mask,
        torch.tensor([[False, True, False, True, True]]),
        rtol=0,
        atol=0,
    )


def test_excess_empty_slots_are_reserved_by_stable_id_not_compatibility() -> None:
    model = make_allocator(state_dim=1, num_modules=2, num_slots=4, winners_per_cycle=2)
    with torch.no_grad():
        model.slot_projection.weight.fill_(1.0)
    slots = torch.tensor([[[0.0], [10_000.0], [20_000.0], [30_000.0]]])
    occupancy = torch.tensor([[False, True, False, False]])

    output = model(
        make_proposal(
            batch_size=1,
            num_modules=2,
            state_dim=1,
            key=torch.ones(1, 2, 1),
        ),
        make_workspace(
            batch_size=1,
            num_slots=4,
            state_dim=1,
            slots=slots,
            occupancy=occupancy,
        ),
        make_arbitration(batch_size=1, num_modules=2),
    )

    torch.testing.assert_close(
        output.target_mask,
        torch.tensor([[True, False, True, False]]),
        rtol=0,
        atol=0,
    )


def test_full_workspace_selects_lowest_retention_with_stable_ties() -> None:
    model = make_allocator(
        num_modules=3,
        num_slots=5,
        winners_per_cycle=3,
        retention_age_penalty=0.25,
    )
    occupancy = torch.ones(1, 5, dtype=torch.bool)
    age = torch.tensor([[0, 2, 2, 1, 0]])
    confidence = torch.tensor([[0.1, 0.8, 0.8, 0.0, 0.9]])

    output = model(
        make_proposal(batch_size=1),
        make_workspace(
            batch_size=1,
            num_slots=5,
            occupancy=occupancy,
            age=age,
            confidence=confidence,
        ),
        make_arbitration(batch_size=1, winners_per_cycle=3),
    )

    # Retentions are [0.1, 0.3, 0.3, -0.25, 0.9]; tie 1/2 prefers slot 1.
    torch.testing.assert_close(
        output.target_mask,
        torch.tensor([[True, True, False, True, False]]),
        rtol=0,
        atol=0,
    )


def test_mixed_batch_builds_each_target_set_independently() -> None:
    model = make_allocator(num_slots=4)
    occupancy = torch.tensor(
        [[False, False, False, False], [True, False, True, True], [True, True, True, True]]
    )
    age = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 5], [0, 3, 1, 2]])
    confidence = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.4, 0.8], [0.1, 0.9, 0.2, 0.7]])

    output = model(
        make_proposal(batch_size=3),
        make_workspace(
            batch_size=3,
            occupancy=occupancy,
            age=age,
            confidence=confidence,
        ),
        make_arbitration(batch_size=3),
    )

    torch.testing.assert_close(
        output.target_mask,
        torch.tensor(
            [[True, True, False, False], [True, True, False, False], [True, False, True, False]]
        ),
        rtol=0,
        atol=0,
    )


def test_greedy_assignment_resolves_collisions_in_arbiter_winner_order() -> None:
    model = make_allocator(
        state_dim=1,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
        retention_age_penalty=0.0,
    )
    with torch.no_grad():
        model.slot_projection.weight.fill_(1.0)
    proposal = make_proposal(
        batch_size=1,
        num_modules=3,
        state_dim=1,
        key=torch.tensor([[[2.0], [0.0], [3.0]]]),
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=2,
        state_dim=1,
        slots=torch.tensor([[[1.0], [3.0]]]),
        occupancy=torch.ones(1, 2, dtype=torch.bool),
        confidence=torch.zeros(1, 2),
    )
    arbitration = make_arbitration(
        batch_size=1,
        num_modules=3,
        scores=torch.tensor([[2.0, 0.0, 3.0]]),
    )

    output = model(proposal, workspace, arbitration)

    torch.testing.assert_close(output.module_indices, torch.tensor([[2, 0]]), rtol=0, atol=0)
    torch.testing.assert_close(output.slot_indices, torch.tensor([[1, 0]]), rtol=0, atol=0)
    expected = torch.tensor([[[True, False], [False, False], [False, True]]])
    torch.testing.assert_close(output.assignment_mask, expected, rtol=0, atol=0)
    assert torch.equal(output.assignment_weights, expected.to(torch.float32))


def test_exact_allocation_ties_choose_lowest_remaining_slot_id() -> None:
    model = make_allocator(
        state_dim=2,
        num_modules=2,
        num_slots=3,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
    )
    with torch.no_grad():
        model.slot_projection.weight.zero_()
    occupancy = torch.tensor([[True, False, True]])

    output = model(
        make_proposal(batch_size=1, num_modules=2, state_dim=2),
        make_workspace(
            batch_size=1,
            num_slots=3,
            state_dim=2,
            occupancy=occupancy,
            confidence=torch.tensor([[0.0, 0.0, 0.0]]),
        ),
        make_arbitration(batch_size=1, num_modules=2),
    )

    # Slot 1 is empty and therefore reserved; slot 0 wins the retention tie.
    torch.testing.assert_close(output.target_mask, torch.tensor([[True, True, False]]))
    # Empty bonus makes slot 1 the first winner's choice; the second gets slot 0.
    torch.testing.assert_close(output.slot_indices, torch.tensor([[1, 0]]), rtol=0, atol=0)


def test_dense_probabilities_follow_greedy_remaining_target_sets() -> None:
    model = make_allocator(
        state_dim=1,
        num_modules=3,
        num_slots=3,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
        retention_age_penalty=0.0,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.slot_projection.weight.fill_(1.0)
    proposal = make_proposal(
        batch_size=1,
        num_modules=3,
        state_dim=1,
        dtype=torch.float64,
        key=torch.tensor([[[-1.0], [0.0], [2.0]]], dtype=torch.float64),
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=3,
        state_dim=1,
        dtype=torch.float64,
        slots=torch.tensor([[[1.0], [7.0], [3.0]]], dtype=torch.float64),
        occupancy=torch.ones(1, 3, dtype=torch.bool),
        confidence=torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float64),
    )
    arbitration = make_arbitration(
        batch_size=1,
        num_modules=3,
        dtype=torch.float64,
        scores=torch.tensor([[2.0, 0.0, 3.0]], dtype=torch.float64),
    )

    output = model(proposal, workspace, arbitration)

    expected = torch.zeros(1, 3, 3, dtype=torch.float64)
    expected[0, 2, [0, 2]] = torch.softmax(torch.tensor([2.0, 6.0], dtype=torch.float64), dim=0)
    expected[0, 0, 0] = 1.0
    torch.testing.assert_close(output.assignment_probabilities, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(output.assignment_probabilities[0, 1]) == 0
    torch.testing.assert_close(output.slot_indices, torch.tensor([[2, 0]]), rtol=0, atol=0)


@pytest.mark.parametrize("winners_per_cycle", [1, 2, 4])
def test_output_is_dense_typed_frozen_and_has_exact_assignment_counts(
    winners_per_cycle: int,
) -> None:
    model = make_allocator(
        num_modules=4,
        num_slots=4,
        winners_per_cycle=winners_per_cycle,
    )

    output = model(
        make_proposal(num_modules=4),
        make_workspace(),
        make_arbitration(num_modules=4, winners_per_cycle=winners_per_cycle),
    )

    assert isinstance(output, SlotAllocatorOutput)
    assert {field.name for field in fields(output)} == {
        "allocation_scores",
        "retention_scores",
        "target_mask",
        "assignment_mask",
        "assignment_probabilities",
        "assignment_weights",
        "module_indices",
        "slot_indices",
    }
    assert output.allocation_scores.shape == (2, 4, 4)
    assert output.retention_scores.shape == (2, 4)
    assert output.target_mask.shape == (2, 4)
    for tensor in (
        output.assignment_mask,
        output.assignment_probabilities,
        output.assignment_weights,
    ):
        assert tensor.shape == (2, 4, 4)
    assert output.module_indices.shape == (2, winners_per_cycle)
    assert output.slot_indices.shape == (2, winners_per_cycle)
    assert output.target_mask.dtype == torch.bool
    assert output.assignment_mask.dtype == torch.bool
    torch.testing.assert_close(
        output.assignment_mask.sum(dim=(1, 2)),
        torch.full((2,), winners_per_cycle, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.assignment_mask.sum(dim=1).amax(dim=-1),
        torch.ones(2, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    with pytest.raises(FrozenInstanceError):
        output.slot_indices = torch.zeros_like(output.slot_indices)  # type: ignore[misc]


def test_training_and_evaluation_have_identical_hard_forward_assignments() -> None:
    model = make_allocator(state_dim=2, num_slots=3)
    keys = torch.tensor(
        [[[1.0, 0.5], [-0.5, 1.0], [0.25, -1.0]]],
        requires_grad=True,
    )
    proposal = make_proposal(batch_size=1, state_dim=2, key=keys)
    workspace = make_workspace(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        slots=torch.tensor([[[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]]),
        occupancy=torch.ones(1, 3, dtype=torch.bool),
        confidence=torch.tensor([[0.1, 0.2, 0.9]]),
    )
    arbitration = make_arbitration(batch_size=1, straight_through=True)

    model.train()
    training = model(proposal, workspace, arbitration)
    model.eval()
    evaluation = model(proposal, workspace, arbitration)

    torch.testing.assert_close(training.assignment_mask, evaluation.assignment_mask)
    torch.testing.assert_close(training.slot_indices, evaluation.slot_indices)
    assert torch.equal(training.assignment_weights, training.assignment_mask.to(torch.float32))
    assert torch.equal(evaluation.assignment_weights, evaluation.assignment_mask.to(torch.float32))
    assert training.assignment_weights.requires_grad
    assert not evaluation.assignment_weights.requires_grad
    assert evaluation.assignment_weights.grad_fn is None


def test_slot_straight_through_backward_is_softmax_jacobian_on_remaining_targets() -> None:
    model = make_allocator(
        state_dim=1,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
        retention_age_penalty=0.0,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.slot_projection.weight.fill_(1.0)
    output = model(
        make_proposal(
            batch_size=1,
            num_modules=3,
            state_dim=1,
            dtype=torch.float64,
            key=torch.tensor([[[0.4], [-0.2], [0.1]]], dtype=torch.float64),
        ),
        make_workspace(
            batch_size=1,
            num_slots=2,
            state_dim=1,
            dtype=torch.float64,
            slots=torch.tensor([[[1.0], [2.0]]], dtype=torch.float64),
            occupancy=torch.ones(1, 2, dtype=torch.bool),
            confidence=torch.zeros(1, 2, dtype=torch.float64),
        ),
        make_arbitration(
            batch_size=1,
            num_modules=3,
            dtype=torch.float64,
            straight_through=True,
        ),
    )
    coefficients = torch.zeros(1, 3, 2, dtype=torch.float64)
    coefficients[0, 0] = torch.tensor([0.7, -0.2], dtype=torch.float64)

    gradient = torch.autograd.grad(
        (output.assignment_weights * coefficients).sum(),
        output.allocation_scores,
        retain_graph=True,
    )[0]

    probabilities = output.assignment_probabilities[0, 0]
    expected_row = probabilities * (coefficients[0, 0] - (probabilities * coefficients[0, 0]).sum())
    expected = torch.zeros_like(output.allocation_scores)
    expected[0, 0] = expected_row
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


def test_assignment_weights_propagate_exact_arbiter_softmax_gradient() -> None:
    model = make_allocator(
        state_dim=1,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
        retention_age_penalty=0.0,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.slot_projection.weight.fill_(1.0)
    scores = torch.tensor([[2.0, 1.0, -0.5]], dtype=torch.float64, requires_grad=True)
    arbitration = make_arbitration(
        batch_size=1,
        num_modules=3,
        dtype=torch.float64,
        scores=scores,
        straight_through=True,
    )
    output = model(
        make_proposal(
            batch_size=1,
            num_modules=3,
            state_dim=1,
            dtype=torch.float64,
            key=torch.tensor([[[0.2], [0.3], [-0.1]]], dtype=torch.float64),
        ),
        make_workspace(
            batch_size=1,
            num_slots=2,
            state_dim=1,
            dtype=torch.float64,
            slots=torch.tensor([[[1.0], [2.0]]], dtype=torch.float64),
            occupancy=torch.ones(1, 2, dtype=torch.bool),
            confidence=torch.zeros(1, 2, dtype=torch.float64),
        ),
        arbitration,
    )
    coefficients = torch.tensor(
        [[[0.7, -0.2], [1.3, -0.4], [-0.6, 0.9]]],
        dtype=torch.float64,
    )

    gradient = torch.autograd.grad((output.assignment_weights * coefficients).sum(), scores)[0]

    hard_module_coefficients = (output.assignment_mask.to(torch.float64) * coefficients).sum(-1)
    probabilities = torch.softmax(scores.detach(), dim=-1)
    expected = probabilities * (
        hard_module_coefficients
        - (hard_module_coefficients * probabilities).sum(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(gradient) == 3


def test_slot_surrogate_reaches_projection_winner_keys_and_occupied_slots() -> None:
    model = make_allocator(
        state_dim=2,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
        retention_age_penalty=0.0,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        model.slot_projection.weight.copy_(torch.tensor([[1.0, 0.2], [-0.3, 0.7]]))
    keys = torch.tensor(
        [[[0.4, -0.2], [0.3, 0.5], [-0.6, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    slots = torch.tensor(
        [[[0.5, 1.0], [-0.7, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = model(
        make_proposal(
            batch_size=1,
            num_modules=3,
            state_dim=2,
            dtype=torch.float64,
            key=keys,
        ),
        make_workspace(
            batch_size=1,
            num_slots=2,
            state_dim=2,
            dtype=torch.float64,
            slots=slots,
            occupancy=torch.ones(1, 2, dtype=torch.bool),
            confidence=torch.zeros(1, 2, dtype=torch.float64),
        ),
        make_arbitration(
            batch_size=1,
            num_modules=3,
            dtype=torch.float64,
            straight_through=True,
        ),
    )
    coefficients = torch.zeros(1, 3, 2, dtype=torch.float64)
    coefficients[0, 0] = torch.tensor([1.1, -0.7], dtype=torch.float64)

    (output.assignment_weights * coefficients).sum().backward()

    assert model.slot_projection.weight.grad is not None
    assert torch.isfinite(model.slot_projection.weight.grad).all()
    assert torch.count_nonzero(model.slot_projection.weight.grad) > 0
    assert keys.grad is not None
    assert torch.isfinite(keys.grad).all()
    assert torch.count_nonzero(keys.grad[0, 0]) > 0
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert torch.count_nonzero(slots.grad[0, 0]) > 0
    assert torch.count_nonzero(slots.grad[0, 1]) > 0


def test_empty_slot_content_and_keys_have_exact_zero_assignment_gradient() -> None:
    model = make_allocator(
        state_dim=2,
        num_modules=2,
        num_slots=2,
        winners_per_cycle=2,
        allocation_age_penalty=0.0,
    )
    keys = torch.tensor([[[0.5, -1.0], [0.25, 0.75]]], requires_grad=True)
    slots = torch.tensor([[[100.0, -200.0], [-300.0, 400.0]]], requires_grad=True)
    output = model(
        make_proposal(batch_size=1, num_modules=2, state_dim=2, key=keys),
        make_workspace(
            batch_size=1,
            num_slots=2,
            state_dim=2,
            slots=slots,
        ),
        make_arbitration(batch_size=1, num_modules=2, straight_through=True),
    )
    coefficients = torch.tensor([[[0.8, -0.4], [0.0, 0.0]]])

    (output.assignment_weights * coefficients).sum().backward()

    assert keys.grad is not None
    assert slots.grad is not None
    assert torch.count_nonzero(keys.grad) == 0
    assert torch.count_nonzero(slots.grad) == 0
    assert model.slot_projection.weight.grad is not None
    assert torch.count_nonzero(model.slot_projection.weight.grad) == 0


def test_forward_does_not_mutate_any_input_tensor() -> None:
    model = make_allocator()
    proposal = make_proposal()
    workspace = make_workspace(
        slots=torch.randn(2, 4, 4),
        occupancy=torch.tensor([[True, False, True, False], [False, True, False, True]]),
    )
    arbitration = make_arbitration(straight_through=True)
    proposal_before = tuple(
        cast(torch.Tensor, getattr(proposal, field.name)).detach().clone()
        for field in fields(proposal)
    )
    workspace_before = tuple(
        cast(torch.Tensor, getattr(workspace, field.name)).detach().clone()
        for field in fields(workspace)
    )
    arbitration_before = tuple(
        cast(torch.Tensor, getattr(arbitration, field.name)).detach().clone()
        for field in fields(arbitration)
    )

    model(proposal, workspace, arbitration)

    for field, before in zip(fields(proposal), proposal_before, strict=True):
        assert torch.equal(cast(torch.Tensor, getattr(proposal, field.name)), before)
    for field, before in zip(fields(workspace), workspace_before, strict=True):
        assert torch.equal(cast(torch.Tensor, getattr(workspace, field.name)), before)
    for field, before in zip(fields(arbitration), arbitration_before, strict=True):
        assert torch.equal(cast(torch.Tensor, getattr(arbitration, field.name)), before)


@pytest.mark.parametrize(
    ("dtype", "large_value"),
    [(torch.float16, 60_000.0), (torch.bfloat16, 1e18)],
    ids=["float16", "bfloat16"],
)
def test_reduced_precision_extreme_inputs_accumulate_in_fp32(
    dtype: torch.dtype,
    large_value: float,
) -> None:
    model = make_allocator(state_dim=8, num_slots=3).to(dtype=dtype)
    with torch.no_grad():
        model.slot_projection.weight.copy_(torch.eye(8, dtype=dtype))
    keys = torch.full((1, 3, 8), large_value, dtype=dtype)
    keys[:, 1] *= -1
    slots = torch.full((1, 3, 8), large_value, dtype=dtype)
    slots[:, 1] *= -1
    occupancy = torch.tensor([[True, True, False]])

    output = model(
        make_proposal(
            batch_size=1,
            state_dim=8,
            dtype=dtype,
            key=keys,
        ),
        make_workspace(
            batch_size=1,
            num_slots=3,
            state_dim=8,
            dtype=dtype,
            slots=slots,
            occupancy=occupancy,
        ),
        make_arbitration(batch_size=1, dtype=torch.float32),
    )

    for tensor in (
        output.allocation_scores,
        output.retention_scores,
        output.assignment_probabilities,
        output.assignment_weights,
    ):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_float64_preserves_precision_for_all_floating_outputs() -> None:
    model = make_allocator().to(dtype=torch.float64)
    output = model(
        make_proposal(dtype=torch.float64),
        make_workspace(dtype=torch.float64),
        make_arbitration(dtype=torch.float64),
    )

    for tensor in (
        output.allocation_scores,
        output.retention_scores,
        output.assignment_probabilities,
        output.assignment_weights,
    ):
        assert tensor.dtype == torch.float64


def test_external_autocast_cannot_lower_allocator_precision() -> None:
    model = make_allocator(state_dim=3, num_slots=3)
    proposal = make_proposal(state_dim=3)
    workspace = make_workspace(
        num_slots=3,
        state_dim=3,
        slots=torch.tensor(
            [
                [[1.0, 2.0, -1.0], [-0.5, 0.25, 2.0], [0.1, -0.2, 0.3]],
                [[-1.0, 0.5, 2.0], [0.7, -1.2, 0.4], [1.0, 1.0, -1.0]],
            ]
        ),
        occupancy=torch.tensor([[True, True, False], [True, False, True]]),
    )
    arbitration = make_arbitration()
    reference = model(proposal, workspace, arbitration)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model(proposal, workspace, arbitration)

    for actual_tensor, reference_tensor in (
        (actual.allocation_scores, reference.allocation_scores),
        (actual.retention_scores, reference.retention_scores),
        (actual.assignment_probabilities, reference.assignment_probabilities),
        (actual.assignment_weights, reference.assignment_weights),
    ):
        assert actual_tensor.dtype == torch.float32
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


def test_materializes_inference_inputs_outside_global_inference_mode() -> None:
    model = make_allocator(state_dim=3, num_slots=3)
    with torch.inference_mode():
        proposal = make_proposal(state_dim=3)
        workspace = make_workspace(
            num_slots=3,
            state_dim=3,
            slots=torch.randn(2, 3, 3),
            occupancy=torch.tensor([[True, True, False], [False, True, True]]),
        )
        arbitration = make_arbitration()

    output = model(proposal, workspace, arbitration)

    for field in fields(output):
        tensor = cast(torch.Tensor, getattr(output, field.name))
        assert not tensor.is_inference()
    coefficients = torch.randn_like(output.assignment_weights)
    (output.assignment_weights * coefficients).sum().backward()
    assert model.slot_projection.weight.grad is not None
    assert torch.isfinite(model.slot_projection.weight.grad).all()


def test_supports_global_inference_mode() -> None:
    model = make_allocator()

    with torch.inference_mode():
        output = model(make_proposal(), make_workspace(), make_arbitration())

    for field in fields(output):
        tensor = cast(torch.Tensor, getattr(output, field.name))
        assert tensor.is_inference()
        assert not tensor.requires_grad


def test_deterministic_forward_does_not_consume_rng() -> None:
    model = make_allocator()
    proposal = make_proposal()
    workspace = make_workspace()
    arbitration = make_arbitration()
    torch.manual_seed(1234567)
    rng_before = torch.random.get_rng_state().clone()

    first = model(proposal, workspace, arbitration)
    rng_after = torch.random.get_rng_state()
    torch.manual_seed(9)
    second = model(proposal, workspace, arbitration)

    assert torch.equal(rng_after, rng_before)
    for first_field, second_field in zip(fields(first), fields(second), strict=True):
        torch.testing.assert_close(
            cast(torch.Tensor, getattr(first, first_field.name)),
            cast(torch.Tensor, getattr(second, second_field.name)),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("state_dim", 0),
        ("state_dim", -1),
        ("state_dim", True),
        ("num_modules", 0),
        ("num_slots", 0),
        ("winners_per_cycle", 0),
    ],
)
def test_rejects_invalid_positive_integer_constructor_arguments(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "state_dim": 4,
        "num_modules": 3,
        "num_slots": 4,
        "winners_per_cycle": 2,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        SlotAllocator(allocator_config(), **cast(dict[str, int], arguments))


@pytest.mark.parametrize(
    ("num_modules", "num_slots", "winners_per_cycle", "message"),
    [(2, 4, 3, "cannot exceed num_modules"), (4, 2, 3, "cannot exceed num_slots")],
)
def test_rejects_winner_count_larger_than_module_or_slot_capacity(
    num_modules: int,
    num_slots: int,
    winners_per_cycle: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_allocator(
            num_modules=num_modules,
            num_slots=num_slots,
            winners_per_cycle=winners_per_cycle,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("empty_bonus", -0.1),
        ("empty_bonus", float("nan")),
        ("empty_bonus", float("inf")),
        ("empty_bonus", True),
        ("allocation_age_penalty", -0.1),
        ("allocation_age_penalty", float("nan")),
        ("allocation_age_penalty", "0.1"),
        ("retention_age_penalty", -0.1),
        ("retention_age_penalty", float("inf")),
        ("retention_age_penalty", False),
    ],
)
def test_revalidates_bypassed_invalid_config_at_module_boundary(
    field: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "empty_bonus": 1.0,
        "allocation_age_penalty": 0.1,
        "retention_age_penalty": 0.1,
    }
    values[field] = value
    config = SlotAllocatorConfig.model_construct(**values)

    with pytest.raises(ValueError, match=field):
        SlotAllocator(
            config,
            state_dim=4,
            num_modules=3,
            num_slots=4,
            winners_per_cycle=2,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("content", "not-a-tensor"),
        ("content", torch.ones(2, 3)),
        ("content", torch.ones(0, 3, 4)),
        ("content", torch.ones(2, 2, 4)),
        ("content", torch.ones(2, 3, 5)),
        ("content", torch.ones(2, 3, 4, dtype=torch.int64)),
        ("content", torch.full((2, 3, 4), torch.nan)),
        ("key", torch.ones(2, 2, 4)),
        ("key", torch.ones(2, 3, 4, dtype=torch.float64)),
        ("key", torch.full((2, 3, 4), torch.inf)),
        ("bid", torch.ones(2, 3, 1)),
        ("bid", torch.ones(2, 3, dtype=torch.int64)),
        ("confidence", torch.full((2, 3), -0.1)),
        ("confidence", torch.full((2, 3), 1.1)),
        ("ttl", torch.full((2, 3), torch.nan)),
        ("ttl", torch.full((2, 3), 0.9)),
    ],
)
def test_rejects_malformed_proposal_field(field: str, replacement: object) -> None:
    proposal = make_proposal()

    with pytest.raises(SlotAllocatorError):
        make_allocator()(
            replace(proposal, **{field: cast(Any, replacement)}),
            make_workspace(),
            make_arbitration(),
        )


def test_rejects_non_proposal_input() -> None:
    with pytest.raises(SlotAllocatorError, match="ModuleProposal"):
        make_allocator()(
            cast(Any, torch.ones(2, 3, 4)),
            make_workspace(),
            make_arbitration(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("slots", "not-a-tensor"),
        ("slots", torch.ones(2, 4)),
        ("slots", torch.ones(2, 3, 4)),
        ("slots", torch.ones(2, 4, 5)),
        ("slots", torch.ones(2, 4, 4, dtype=torch.int64)),
        ("slots", torch.ones(2, 4, 4, dtype=torch.float64)),
        ("slots", torch.full((2, 4, 4), torch.nan)),
        ("occupancy", torch.zeros(2, 3, dtype=torch.bool)),
        ("occupancy", torch.zeros(2, 4, dtype=torch.int64)),
        ("age", torch.zeros(2, 3, dtype=torch.int64)),
        ("age", torch.zeros(2, 4, dtype=torch.int32)),
        ("age", torch.full((2, 4), -1, dtype=torch.int64)),
        ("source_module", torch.zeros(2, 4, dtype=torch.int32)),
        ("source_module", torch.full((2, 4), -2, dtype=torch.int64)),
        ("source_module", torch.full((2, 4), 3, dtype=torch.int64)),
        ("confidence", torch.zeros(2, 3)),
        ("confidence", torch.zeros(2, 4, dtype=torch.float64)),
        ("confidence", torch.full((2, 4), torch.nan)),
        ("confidence", torch.full((2, 4), -0.1)),
        ("confidence", torch.full((2, 4), 1.1)),
        ("ttl", torch.zeros(2, 3)),
        ("ttl", torch.zeros(2, 4, dtype=torch.float64)),
        ("ttl", torch.full((2, 4), torch.inf)),
        ("ttl", torch.full((2, 4), -0.1)),
        ("ttl", torch.ones(2, 4)),
        ("write_count", torch.zeros(2, 4, dtype=torch.int32)),
        ("write_count", torch.full((2, 4), -1, dtype=torch.int64)),
    ],
)
def test_rejects_malformed_workspace_field(field: str, replacement: object) -> None:
    workspace = make_workspace()

    with pytest.raises(SlotAllocatorError):
        make_allocator()(
            make_proposal(),
            replace(workspace, **{field: cast(Any, replacement)}),
            make_arbitration(),
        )


def test_rejects_non_workspace_input() -> None:
    with pytest.raises(SlotAllocatorError, match="WorkspaceState"):
        make_allocator()(
            make_proposal(),
            cast(Any, torch.ones(2, 4, 4)),
            make_arbitration(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("scores", torch.ones(2, 2)),
        ("scores", torch.ones(2, 3, dtype=torch.float64)),
        ("scores", torch.full((2, 3), torch.nan)),
        ("novelty", torch.ones(2, 3, 1)),
        ("novelty", torch.full((2, 3), torch.inf)),
        ("relevance", torch.ones(2, 2)),
        ("soft_probabilities", torch.tensor([[0.5, 0.5, -0.0], [0.5, 0.5, -0.1]])),
        ("soft_probabilities", torch.full((2, 3), 0.5)),
        ("selection_weights", torch.ones(2, 3)),
        ("winner_mask", torch.ones(2, 3, dtype=torch.int64)),
        ("winner_mask", torch.tensor([[True, False, False], [True, False, False]])),
        ("winner_indices", torch.zeros(2, 1, dtype=torch.int64)),
        ("winner_indices", torch.zeros(2, 2, dtype=torch.int32)),
        ("winner_indices", torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)),
    ],
)
def test_rejects_malformed_arbitration_field(field: str, replacement: object) -> None:
    arbitration = make_arbitration()

    with pytest.raises(SlotAllocatorError):
        make_allocator()(
            make_proposal(),
            make_workspace(),
            replace(arbitration, **{field: cast(Any, replacement)}),
        )


def test_rejects_non_arbitration_input() -> None:
    with pytest.raises(SlotAllocatorError, match="WorkspaceArbiterOutput"):
        make_allocator()(
            make_proposal(),
            make_workspace(),
            cast(Any, torch.ones(2, 3)),
        )


def test_rejects_parameter_dtype_mismatch_and_non_finite_weight() -> None:
    model = make_allocator()
    proposal = make_proposal(dtype=torch.float64)
    workspace = make_workspace(dtype=torch.float64)
    arbitration = make_arbitration(dtype=torch.float64)

    with pytest.raises(SlotAllocatorError, match="parameters"):
        model(proposal, workspace, arbitration)

    model = make_allocator()
    with torch.no_grad():
        model.slot_projection.weight[0, 0] = torch.nan
    with pytest.raises(SlotAllocatorError, match="parameters"):
        model(make_proposal(), make_workspace(), make_arbitration())
