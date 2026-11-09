"""Contract tests for deterministic global-workspace proposal arbitration."""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
import torch

from iris.cognitive import ModuleProposal
from iris.schemas import WorkspaceArbiterConfig
from iris.workspace import (
    WorkspaceArbiter,
    WorkspaceArbiterError,
    WorkspaceArbiterOutput,
    WorkspaceState,
)


def arbiter_config(
    *,
    confidence_weight: float = 0.25,
    novelty_weight: float = 0.10,
    relevance_weight: float = 0.25,
) -> WorkspaceArbiterConfig:
    """Build an explicit score-weight contract for a test."""
    return WorkspaceArbiterConfig(
        confidence_weight=confidence_weight,
        novelty_weight=novelty_weight,
        relevance_weight=relevance_weight,
    )


def make_arbiter(
    *,
    state_dim: int = 4,
    num_modules: int = 3,
    num_slots: int = 4,
    winners_per_cycle: int = 2,
    confidence_weight: float = 0.25,
    novelty_weight: float = 0.10,
    relevance_weight: float = 0.25,
) -> WorkspaceArbiter:
    """Create a small deterministic arbiter."""
    return WorkspaceArbiter(
        arbiter_config(
            confidence_weight=confidence_weight,
            novelty_weight=novelty_weight,
            relevance_weight=relevance_weight,
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
    content: torch.Tensor | None = None,
    bid: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> ModuleProposal:
    """Create a valid proposal whose unused fields still satisfy the full contract."""
    generator = torch.Generator().manual_seed(65537)
    if content is None:
        content = torch.randn(
            batch_size,
            num_modules,
            state_dim,
            generator=generator,
            dtype=dtype,
        )
    if bid is None:
        bid = torch.linspace(-0.4, 0.6, num_modules, dtype=dtype).expand(batch_size, -1).clone()
    if confidence is None:
        confidence = (
            torch.linspace(0.2, 0.8, num_modules, dtype=dtype).expand(batch_size, -1).clone()
        )
    return ModuleProposal(
        content=content,
        key=torch.zeros_like(content),
        bid=bid,
        confidence=confidence,
        ttl=torch.full_like(bid, 2.0),
    )


def make_workspace(
    *,
    batch_size: int = 2,
    num_slots: int = 4,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    slots: torch.Tensor | None = None,
    occupancy: torch.Tensor | None = None,
) -> WorkspaceState:
    """Create a canonical state and populate only explicitly occupied slots."""
    if slots is None:
        slots = torch.zeros(batch_size, num_slots, state_dim, dtype=dtype)
    if occupancy is None:
        occupancy = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=slots.device)
    state = WorkspaceState.empty(
        batch_size=batch_size,
        num_slots=num_slots,
        state_dim=state_dim,
        dtype=dtype,
        device=slots.device,
    )
    return replace(
        state,
        slots=slots,
        occupancy=occupancy,
        source_module=torch.where(
            occupancy,
            torch.zeros_like(state.source_module),
            state.source_module,
        ),
        confidence=occupancy.to(dtype=dtype) * 0.5,
        ttl=occupancy.to(dtype=dtype) * 2.0,
    )


def test_matches_complete_hand_computed_score_formula_and_config_lambdas() -> None:
    config = arbiter_config(
        confidence_weight=0.5,
        novelty_weight=0.7,
        relevance_weight=1.1,
    )
    model = WorkspaceArbiter(
        config,
        state_dim=2,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=2,
    )
    content = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float64,
    )
    bid = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    confidence = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float64)
    proposal = make_proposal(
        batch_size=1,
        num_modules=3,
        state_dim=2,
        dtype=torch.float64,
        content=content,
        bid=bid,
        confidence=confidence,
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=2,
        state_dim=2,
        dtype=torch.float64,
        slots=torch.tensor([[[1.0, 0.0], [1e100, -1e100]]], dtype=torch.float64),
        occupancy=torch.tensor([[True, False]]),
    )
    goal = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    output = model(proposal, workspace, goal)

    expected_novelty = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    expected_relevance = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    expected_scores = bid + 0.5 * confidence + 0.7 * expected_novelty + 1.1 * expected_relevance
    torch.testing.assert_close(output.novelty, expected_novelty, rtol=0, atol=0)
    torch.testing.assert_close(output.relevance, expected_relevance, rtol=0, atol=0)
    torch.testing.assert_close(output.scores, expected_scores, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        output.soft_probabilities,
        torch.softmax(expected_scores, dim=-1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert model.confidence_weight == config.confidence_weight
    assert model.novelty_weight == config.novelty_weight
    assert model.relevance_weight == config.relevance_weight


def test_empty_workspace_assigns_exact_unit_novelty() -> None:
    model = make_arbiter()
    proposal = make_proposal()
    workspace = make_workspace()

    output = model(proposal, workspace, torch.ones(2, 4))

    torch.testing.assert_close(output.novelty, torch.ones(2, 3), rtol=0, atol=0)
    assert torch.isfinite(output.scores).all()


def test_partial_workspace_ignores_stale_unoccupied_content() -> None:
    model = make_arbiter(state_dim=3, num_slots=3, novelty_weight=1.0)
    proposal = make_proposal(state_dim=3)
    occupancy = torch.tensor([[True, False, True], [False, True, False]])
    base_slots = torch.tensor(
        [
            [[1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [-2.0, 0.5, 1.0]],
            [[0.0, 0.0, 0.0], [1.5, -1.0, 2.0], [0.0, 0.0, 0.0]],
        ]
    )
    stale_slots = base_slots.clone()
    stale_slots[~occupancy] = torch.tensor([30_000.0, -40_000.0, 50_000.0])
    goal = torch.tensor([[0.5, -0.25, 0.75], [-0.5, 1.0, 0.25]])

    base = model(
        proposal,
        make_workspace(num_slots=3, state_dim=3, slots=base_slots, occupancy=occupancy),
        goal,
    )
    stale = model(
        proposal,
        make_workspace(num_slots=3, state_dim=3, slots=stale_slots, occupancy=occupancy),
        goal,
    )

    torch.testing.assert_close(stale.novelty, base.novelty, rtol=0, atol=0)
    torch.testing.assert_close(stale.scores, base.scores, rtol=0, atol=0)
    torch.testing.assert_close(stale.winner_indices, base.winner_indices, rtol=0, atol=0)


def test_mixed_empty_and_occupied_batch_uses_per_example_novelty_rule() -> None:
    model = make_arbiter(state_dim=2, num_modules=2, num_slots=2, winners_per_cycle=1)
    proposal = make_proposal(batch_size=2, num_modules=2, state_dim=2)
    slots = torch.tensor([[[99.0, 99.0], [88.0, 88.0]], [[1.0, 0.0], [0.0, 1.0]]])
    occupancy = torch.tensor([[False, False], [True, False]])
    proposal = replace(
        proposal,
        content=torch.tensor([[[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]]),
    )

    output = model(
        proposal,
        make_workspace(
            batch_size=2,
            num_slots=2,
            state_dim=2,
            slots=slots,
            occupancy=occupancy,
        ),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )

    torch.testing.assert_close(output.novelty[0], torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(output.novelty[1], torch.tensor([0.0, 2.0]), rtol=0, atol=0)


def test_unoccupied_slots_are_ignored_and_receive_exact_zero_gradient() -> None:
    model = make_arbiter(
        state_dim=3,
        num_modules=3,
        num_slots=3,
        novelty_weight=1.0,
        confidence_weight=0.0,
        relevance_weight=0.0,
    )
    slots = torch.tensor(
        [[[1.0, 2.0, -1.0], [8.0, 9.0, 10.0], [-2.0, 1.0, 0.5]]],
        requires_grad=True,
    )
    occupancy = torch.tensor([[True, False, True]])
    proposal = make_proposal(batch_size=1, state_dim=3)

    output = model(
        proposal,
        make_workspace(
            batch_size=1,
            num_slots=3,
            state_dim=3,
            slots=slots,
            occupancy=occupancy,
        ),
        torch.tensor([[0.5, -0.25, 1.0]]),
    )
    output.scores.sum().backward()

    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert torch.count_nonzero(slots.grad[:, 1]) == 0
    assert torch.count_nonzero(slots.grad[:, [0, 2]]) > 0


def test_zero_vectors_and_negative_cosine_have_safe_exact_semantics() -> None:
    model = make_arbiter(
        state_dim=2,
        num_modules=3,
        num_slots=1,
        winners_per_cycle=1,
    )
    proposal = make_proposal(
        batch_size=1,
        num_modules=3,
        state_dim=2,
        content=torch.tensor([[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]]),
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=1,
        state_dim=2,
        slots=torch.tensor([[[1.0, 0.0]]]),
        occupancy=torch.tensor([[True]]),
    )

    output = model(proposal, workspace, torch.zeros(1, 2))

    torch.testing.assert_close(output.relevance, torch.zeros(1, 3), rtol=0, atol=0)
    torch.testing.assert_close(output.novelty, torch.tensor([[1.0, 2.0, 0.0]]), rtol=0, atol=0)
    assert torch.isfinite(output.scores).all()


def test_zero_content_has_an_exact_zero_cosine_subgradient() -> None:
    model = make_arbiter(
        state_dim=2,
        num_modules=1,
        num_slots=1,
        winners_per_cycle=1,
        confidence_weight=0.0,
        novelty_weight=1.0,
        relevance_weight=1.0,
    )
    content = torch.zeros(1, 1, 2, requires_grad=True)
    proposal = make_proposal(
        batch_size=1,
        num_modules=1,
        state_dim=2,
        content=content,
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=1,
        state_dim=2,
        slots=torch.tensor([[[1.0, 0.0]]]),
        occupancy=torch.tensor([[True]]),
    )

    model(proposal, workspace, torch.tensor([[0.0, 1.0]])).scores.sum().backward()

    assert content.grad is not None
    assert torch.count_nonzero(content.grad) == 0


@pytest.mark.parametrize("winners_per_cycle", [1, 2, 4])
def test_output_shapes_and_exact_winner_count_follow_r(winners_per_cycle: int) -> None:
    model = make_arbiter(
        num_modules=4,
        num_slots=4,
        winners_per_cycle=winners_per_cycle,
    )

    output = model(
        make_proposal(num_modules=4),
        make_workspace(),
        torch.ones(2, 4),
    )

    assert isinstance(output, WorkspaceArbiterOutput)
    for tensor in (
        output.scores,
        output.novelty,
        output.relevance,
        output.soft_probabilities,
        output.selection_weights,
        output.winner_mask,
    ):
        assert tensor.shape == (2, 4)
    assert output.winner_indices.shape == (2, winners_per_cycle)
    torch.testing.assert_close(
        output.winner_mask.sum(dim=-1),
        torch.full((2,), winners_per_cycle, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    with pytest.raises(FrozenInstanceError):
        output.scores = torch.zeros_like(output.scores)  # type: ignore[misc]


def test_equal_scores_have_stable_low_index_ties_and_do_not_consume_rng() -> None:
    model = make_arbiter(
        num_modules=5,
        num_slots=3,
        winners_per_cycle=3,
        confidence_weight=0.0,
        novelty_weight=0.0,
        relevance_weight=0.0,
    )
    proposal = make_proposal(
        batch_size=1,
        num_modules=5,
        bid=torch.zeros(1, 5),
        confidence=torch.zeros(1, 5),
    )
    workspace = make_workspace(batch_size=1, num_slots=3)
    goal = torch.zeros(1, 4)
    torch.manual_seed(1234567)
    rng_before = torch.random.get_rng_state().clone()

    first = model(proposal, workspace, goal)
    rng_after = torch.random.get_rng_state()
    torch.manual_seed(9)
    second = model(proposal, workspace, goal)

    assert torch.equal(rng_after, rng_before)
    torch.testing.assert_close(first.winner_indices, torch.tensor([[0, 1, 2]]), rtol=0, atol=0)
    torch.testing.assert_close(
        first.winner_mask,
        torch.tensor([[True, True, True, False, False]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(second.winner_indices, first.winner_indices, rtol=0, atol=0)
    torch.testing.assert_close(second.winner_mask, first.winner_mask, rtol=0, atol=0)


def test_training_and_evaluation_have_the_same_hard_forward_selection() -> None:
    model = make_arbiter(
        confidence_weight=0.0,
        novelty_weight=0.0,
        relevance_weight=0.0,
    )
    bid = torch.tensor([[2.0, 0.5, -1.0], [-1.0, 3.0, 0.0]], requires_grad=True)
    proposal = make_proposal(bid=bid)
    workspace = make_workspace()
    goal = torch.zeros(2, 4)

    model.train()
    training = model(proposal, workspace, goal)
    model.eval()
    evaluation = model(proposal, workspace, goal)

    torch.testing.assert_close(training.selection_weights, training.winner_mask.to(torch.float32))
    torch.testing.assert_close(
        evaluation.selection_weights, evaluation.winner_mask.to(torch.float32)
    )
    torch.testing.assert_close(training.selection_weights, evaluation.selection_weights)
    torch.testing.assert_close(training.winner_indices, evaluation.winner_indices)
    assert training.selection_weights.requires_grad
    assert not evaluation.selection_weights.requires_grad
    assert evaluation.selection_weights.grad_fn is None


def test_straight_through_backward_is_the_unscaled_ordinary_softmax_jacobian() -> None:
    model = make_arbiter(
        state_dim=2,
        num_modules=4,
        num_slots=2,
        winners_per_cycle=2,
        confidence_weight=0.0,
        novelty_weight=0.0,
        relevance_weight=0.0,
    ).train()
    bid = torch.tensor([[1.25, -0.75, 0.5, 2.0]], dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([[0.7, -0.2, 1.3, -0.9]], dtype=torch.float64)
    proposal = make_proposal(
        batch_size=1,
        num_modules=4,
        state_dim=2,
        dtype=torch.float64,
        bid=bid,
        confidence=torch.zeros(1, 4, dtype=torch.float64),
    )

    output = model(
        proposal,
        make_workspace(batch_size=1, num_slots=2, state_dim=2, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64),
    )
    gradient = torch.autograd.grad((output.selection_weights * coefficients).sum(), bid)[0]

    probabilities = torch.softmax(bid.detach(), dim=-1)
    expected = probabilities * (
        coefficients - (coefficients * probabilities).sum(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(output.soft_probabilities, probabilities, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


def test_losing_proposals_receive_straight_through_score_gradients() -> None:
    model = make_arbiter(
        state_dim=3,
        num_modules=3,
        num_slots=2,
        winners_per_cycle=1,
        confidence_weight=0.7,
        novelty_weight=0.0,
        relevance_weight=0.8,
    ).train()
    content = torch.tensor(
        [[[0.5, 1.0, -0.25], [-1.0, 0.4, 0.3], [0.2, -0.7, 1.1]]],
        requires_grad=True,
    )
    bid = torch.tensor([[4.0, 0.0, -2.0]], requires_grad=True)
    confidence = torch.tensor([[0.8, 0.4, 0.2]], requires_grad=True)
    proposal = make_proposal(
        batch_size=1,
        state_dim=3,
        content=content,
        bid=bid,
        confidence=confidence,
    )
    output = model(
        proposal,
        make_workspace(batch_size=1, num_slots=2, state_dim=3),
        torch.tensor([[0.3, -0.7, 0.5]]),
    )
    coefficients = torch.tensor([[1.2, -0.4, 0.7]])
    (output.selection_weights * coefficients).sum().backward()

    assert output.winner_indices.item() == 0
    for gradient in (bid.grad, confidence.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[:, 1:]) == 2
    assert content.grad is not None
    assert torch.isfinite(content.grad).all()
    assert torch.count_nonzero(content.grad[:, 1]) > 0
    assert torch.count_nonzero(content.grad[:, 2]) > 0


def test_evaluation_hard_selection_has_no_surrogate_gradient() -> None:
    model = make_arbiter(
        confidence_weight=0.0,
        novelty_weight=0.0,
        relevance_weight=0.0,
    ).eval()
    bid = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]], requires_grad=True)

    output = model(make_proposal(bid=bid), make_workspace(), torch.zeros(2, 4))

    assert output.scores.requires_grad
    assert output.soft_probabilities.requires_grad
    assert not output.selection_weights.requires_grad
    assert output.selection_weights.grad_fn is None


@pytest.mark.parametrize(
    ("dtype", "large_value"),
    [(torch.float16, 60_000.0), (torch.bfloat16, 2e38)],
    ids=["float16", "bfloat16"],
)
def test_reduced_precision_safe_normalization_avoids_overflow(
    dtype: torch.dtype,
    large_value: float,
) -> None:
    model = make_arbiter(state_dim=8, num_slots=3)
    content = torch.full((1, 3, 8), large_value, dtype=dtype)
    content[:, 1] *= -1
    content[:, 2, ::2] *= -1
    proposal = make_proposal(
        batch_size=1,
        state_dim=8,
        dtype=dtype,
        content=content,
    )
    slots = torch.full((1, 3, 8), large_value, dtype=dtype)
    slots[:, 1] *= -1
    occupancy = torch.tensor([[True, True, False]])
    goal = torch.full((1, 8), large_value, dtype=dtype)

    output = model(
        proposal,
        make_workspace(
            batch_size=1,
            num_slots=3,
            state_dim=8,
            dtype=dtype,
            slots=slots,
            occupancy=occupancy,
        ),
        goal,
    )

    for tensor in (
        output.scores,
        output.novelty,
        output.relevance,
        output.soft_probabilities,
        output.selection_weights,
    ):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(output.relevance, torch.tensor([[1.0, -1.0, 0.0]]))
    torch.testing.assert_close(output.novelty, torch.tensor([[0.0, 0.0, 1.0]]))


def test_float64_preserves_precision_for_all_floating_outputs() -> None:
    model = make_arbiter(state_dim=3, num_slots=2)
    proposal = make_proposal(state_dim=3, dtype=torch.float64)
    workspace = make_workspace(num_slots=2, state_dim=3, dtype=torch.float64)

    output = model(proposal, workspace, torch.ones(2, 3, dtype=torch.float64))

    for tensor in (
        output.scores,
        output.novelty,
        output.relevance,
        output.soft_probabilities,
        output.selection_weights,
    ):
        assert tensor.dtype == torch.float64


def test_external_autocast_cannot_lower_score_or_normalization_precision() -> None:
    model = make_arbiter()
    proposal = make_proposal()
    workspace = make_workspace(
        slots=torch.tensor(
            [
                [[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]] * 2,
                [[-1.0, 0.5, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]] * 2,
            ]
        ).reshape(2, 4, 4),
        occupancy=torch.tensor([[True, False, True, False], [True, False, True, False]]),
    )
    goal = torch.tensor([[0.5, -0.25, 1.0, 0.75], [-1.0, 0.25, 0.5, 0.1]])
    reference = model(proposal, workspace, goal)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model(proposal, workspace, goal)

    for actual_tensor, reference_tensor in (
        (actual.scores, reference.scores),
        (actual.novelty, reference.novelty),
        (actual.relevance, reference.relevance),
        (actual.soft_probabilities, reference.soft_probabilities),
        (actual.selection_weights, reference.selection_weights),
    ):
        assert actual_tensor.dtype == torch.float32
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


def test_materializes_inference_inputs_outside_global_inference_mode() -> None:
    model = make_arbiter()
    with torch.inference_mode():
        proposal = make_proposal()
        workspace = make_workspace()
        goal = torch.ones(2, 4)

    output = model(proposal, workspace, goal)

    for tensor in (
        output.scores,
        output.novelty,
        output.relevance,
        output.soft_probabilities,
        output.selection_weights,
        output.winner_mask,
        output.winner_indices,
    ):
        assert not tensor.is_inference()
        assert torch.isfinite(tensor).all()


def test_materialized_inference_proposal_and_slots_allow_goal_backward() -> None:
    model = make_arbiter(
        confidence_weight=0.0,
        novelty_weight=1.0,
        relevance_weight=1.0,
    )
    with torch.inference_mode():
        proposal = make_proposal()
        workspace = make_workspace(
            slots=torch.randn(2, 4, 4),
            occupancy=torch.tensor([[True, False, True, False], [False, True, False, True]]),
        )
    goal = torch.randn(2, 4, requires_grad=True)

    output = model(proposal, workspace, goal)
    output.scores.sum().backward()

    assert goal.grad is not None
    assert torch.isfinite(goal.grad).all()
    assert torch.count_nonzero(goal.grad) > 0


def test_materialized_inference_workspace_and_goal_allow_content_backward() -> None:
    model = make_arbiter(
        confidence_weight=0.0,
        novelty_weight=1.0,
        relevance_weight=1.0,
    )
    content = torch.randn(2, 3, 4, requires_grad=True)
    proposal = make_proposal(content=content)
    with torch.inference_mode():
        workspace = make_workspace(
            slots=torch.randn(2, 4, 4),
            occupancy=torch.tensor([[True, False, True, False], [False, True, False, True]]),
        )
        goal = torch.randn(2, 4)

    output = model(proposal, workspace, goal)
    output.scores.sum().backward()

    assert content.grad is not None
    assert torch.isfinite(content.grad).all()
    assert torch.count_nonzero(content.grad) > 0


def test_supports_global_inference_mode() -> None:
    model = make_arbiter()

    with torch.inference_mode():
        output = model(make_proposal(), make_workspace(), torch.ones(2, 4))

    for tensor in (
        output.scores,
        output.novelty,
        output.relevance,
        output.soft_probabilities,
        output.selection_weights,
        output.winner_mask,
        output.winner_indices,
    ):
        assert tensor.is_inference()
        assert not tensor.requires_grad


@pytest.mark.parametrize(
    ("num_modules", "num_slots", "winners_per_cycle", "message"),
    [(2, 4, 3, "cannot exceed num_modules"), (4, 2, 3, "cannot exceed num_slots")],
)
def test_rejects_r_larger_than_module_or_slot_capacity(
    num_modules: int,
    num_slots: int,
    winners_per_cycle: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_arbiter(
            num_modules=num_modules,
            num_slots=num_slots,
            winners_per_cycle=winners_per_cycle,
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
        ("content", torch.full((2, 3, 4), torch.inf)),
        ("key", "not-a-tensor"),
        ("key", torch.ones(2, 2, 4)),
        ("key", torch.ones(2, 3, 4, dtype=torch.int64)),
        ("key", torch.ones(2, 3, 4, dtype=torch.float64)),
        ("key", torch.full((2, 3, 4), torch.inf)),
        ("bid", torch.ones(2, 3, 1)),
        ("bid", torch.ones(2, 3, dtype=torch.int64)),
        ("bid", torch.ones(2, 3, dtype=torch.float64)),
        ("bid", torch.full((2, 3), torch.nan)),
        ("confidence", torch.ones(2, 3, 1)),
        ("confidence", torch.ones(2, 3, dtype=torch.int64)),
        ("confidence", torch.ones(2, 3, dtype=torch.float64)),
        ("confidence", torch.full((2, 3), torch.nan)),
        ("confidence", torch.full((2, 3), -0.1)),
        ("confidence", torch.full((2, 3), 1.1)),
        ("ttl", torch.ones(2, 3, 1)),
        ("ttl", torch.ones(2, 3, dtype=torch.int64)),
        ("ttl", torch.ones(2, 3, dtype=torch.float64)),
        ("ttl", torch.full((2, 3), torch.inf)),
        ("ttl", torch.full((2, 3), 0.9)),
    ],
)
def test_rejects_malformed_proposal_field(field: str, replacement: object) -> None:
    proposal = make_proposal()
    model = make_arbiter()

    with pytest.raises(WorkspaceArbiterError):
        model(
            replace(proposal, **{field: cast(Any, replacement)}),
            make_workspace(),
            torch.ones(2, 4),
        )


def test_rejects_non_proposal_input() -> None:
    with pytest.raises(WorkspaceArbiterError, match="ModuleProposal"):
        make_arbiter()(cast(Any, torch.ones(2, 3, 4)), make_workspace(), torch.ones(2, 4))


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
        ("source_module", torch.zeros(2, 3, dtype=torch.int64)),
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
        ("write_count", torch.zeros(2, 3, dtype=torch.int64)),
        ("write_count", torch.zeros(2, 4, dtype=torch.int32)),
        ("write_count", torch.full((2, 4), -1, dtype=torch.int64)),
    ],
)
def test_rejects_malformed_workspace_metadata(field: str, replacement: object) -> None:
    state = make_workspace()

    with pytest.raises(WorkspaceArbiterError):
        make_arbiter()(
            make_proposal(),
            replace(state, **{field: cast(Any, replacement)}),
            torch.ones(2, 4),
        )


def test_rejects_non_workspace_input() -> None:
    with pytest.raises(WorkspaceArbiterError, match="WorkspaceState"):
        make_arbiter()(make_proposal(), cast(Any, torch.ones(2, 4, 4)), torch.ones(2, 4))


@pytest.mark.parametrize(
    "goal",
    [
        "not-a-tensor",
        torch.ones(2, 4, 1),
        torch.ones(1, 4),
        torch.ones(2, 3),
        torch.ones(2, 4, dtype=torch.int64),
        torch.ones(2, 4, dtype=torch.float64),
        torch.full((2, 4), torch.nan),
        torch.full((2, 4), torch.inf),
    ],
)
def test_rejects_malformed_goal_state(goal: object) -> None:
    with pytest.raises(WorkspaceArbiterError):
        make_arbiter()(make_proposal(), make_workspace(), cast(Any, goal))
