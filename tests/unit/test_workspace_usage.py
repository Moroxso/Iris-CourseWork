"""Analytical soft-surrogate routing, hard diagnostics and gradient boundaries."""

from dataclasses import replace
from math import log
from typing import Any

import pytest
import torch

from iris.training.losses import IrisLossError, uniform_utilization_loss
from iris.training.outputs import IrisTeacherCycleTraceV1
from iris.training.usage import aggregate_workspace_usage_v1


def _cycle() -> IrisTeacherCycleTraceV1:
    return IrisTeacherCycleTraceV1(
        module_probabilities=torch.tensor([[0.8, 0.2]], dtype=torch.float64),
        allocation_scores=torch.tensor([[[log(3), 0.0], [0.0, log(3)]]], dtype=torch.float64),
        winner_mask=torch.tensor([[True, False]]),
        assignment_mask=torch.tensor([[[True, False], [False, False]]]),
    )


def test_full_slot_mixture_is_not_masked_or_a_hard_selection_histogram() -> None:
    result = aggregate_workspace_usage_v1((_cycle(), _cycle()))
    assert torch.allclose(
        result.module_probabilities, torch.tensor([0.8, 0.2], dtype=torch.float64)
    )
    assert torch.allclose(
        result.slot_probabilities, torch.tensor([0.65, 0.35], dtype=torch.float64)
    )
    assert result.hard_module_frequencies.tolist() == [1.0, 0.0]
    assert result.hard_slot_frequencies.tolist() == [1.0, 0.0]
    changed = replace(
        _cycle(),
        winner_mask=torch.tensor([[False, True]]),
        assignment_mask=torch.tensor([[[False, False], [False, True]]]),
    )
    alternate = aggregate_workspace_usage_v1((changed,))
    assert torch.equal(alternate.slot_probabilities, result.slot_probabilities)
    assert alternate.hard_slot_frequencies.tolist() == [0.0, 1.0]


def test_cycle_average_precedes_kl_and_batches_are_equal_weighted() -> None:
    first = _cycle()
    second = replace(
        first,
        module_probabilities=first.module_probabilities.flip(-1),
        allocation_scores=first.allocation_scores.flip(-1),
    )
    result = aggregate_workspace_usage_v1((first, second))
    assert result.module_probabilities.tolist() == [0.5, 0.5]
    assert result.slot_probabilities.tolist() == pytest.approx([0.65, 0.35])
    batched = IrisTeacherCycleTraceV1(
        **{
            name: torch.cat([getattr(first, name), getattr(second, name)], dim=0)
            for name in (
                "module_probabilities",
                "allocation_scores",
                "winner_mask",
                "assignment_mask",
            )
        }
    )
    together = aggregate_workspace_usage_v1((batched,))
    assert torch.equal(result.slot_probabilities, together.slot_probabilities)
    assert torch.equal(result.module_probabilities, together.module_probabilities)


def test_surrogate_gradients_reach_nonwinner_modules_and_hard_excluded_slots() -> None:
    module_logits = torch.tensor([[log(4), 0.0]], dtype=torch.float64, requires_grad=True)
    scores = _cycle().allocation_scores.clone().requires_grad_(True)
    before = scores.detach().clone()
    cycle = replace(
        _cycle(), module_probabilities=module_logits.softmax(-1), allocation_scores=scores
    )
    usage = aggregate_workspace_usage_v1((cycle,))
    torch.autograd.backward(
        uniform_utilization_loss(usage.module_probabilities, usage.slot_probabilities)
    )
    assert scores.grad is not None and bool((scores.grad.abs() > 1e-10).all())
    assert module_logits.grad is not None and bool((module_logits.grad.abs() > 1e-10).all())
    assert torch.equal(scores.detach(), before)
    assert not usage.hard_slot_frequencies.requires_grad


def test_surrogate_numerical_gradcheck_and_single_slot_zero() -> None:
    module_logits = torch.tensor([[0.5, -0.4]], dtype=torch.float64, requires_grad=True)
    scores = _cycle().allocation_scores.clone().requires_grad_(True)

    def objective(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        usage = aggregate_workspace_usage_v1(
            (
                replace(
                    _cycle(),
                    module_probabilities=logits.softmax(-1),
                    allocation_scores=values,
                ),
            )
        )
        return uniform_utilization_loss(usage.module_probabilities, usage.slot_probabilities)

    assert torch.autograd.gradcheck(objective, (module_logits, scores))
    one = IrisTeacherCycleTraceV1(
        torch.ones(1, 1, dtype=torch.float64),
        torch.zeros(1, 1, 1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    usage = aggregate_workspace_usage_v1((one,))
    assert (
        uniform_utilization_loss(usage.module_probabilities, usage.slot_probabilities).item() == 0
    )


@pytest.mark.parametrize(
    "change",
    [
        {"module_probabilities": torch.tensor([[0.7, 0.1]], dtype=torch.float64)},
        {"module_probabilities": torch.tensor([[-0.1, 1.1]], dtype=torch.float64)},
        {"module_probabilities": torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)},
        {"module_probabilities": torch.ones(1, 2, dtype=torch.int64)},
        {"module_probabilities": torch.ones(2, dtype=torch.float64)},
        {"allocation_scores": torch.ones(1, 2, 2)},
        {"allocation_scores": torch.full((1, 2, 2), float("inf"), dtype=torch.float64)},
        {"allocation_scores": torch.empty(1, 2, 0, dtype=torch.float64)},
        {"winner_mask": torch.tensor([[1, 0]])},
        {"winner_mask": torch.tensor([[False, False]])},
        {"assignment_mask": torch.zeros(1, 2, 2, dtype=torch.bool)},
        {"assignment_mask": torch.tensor([[[True, True], [False, False]]])},
        {
            "winner_mask": torch.tensor([[True, True]]),
            "assignment_mask": torch.tensor([[[True, False], [True, False]]]),
        },
    ],
)
def test_invalid_cycle_never_produces_a_distribution(change: dict[str, Any]) -> None:
    with pytest.raises(IrisLossError):
        aggregate_workspace_usage_v1((replace(_cycle(), **change),))


def test_bad_late_cycle_and_empty_trace_are_rejected() -> None:
    with pytest.raises(IrisLossError):
        aggregate_workspace_usage_v1(())
    changed = replace(
        _cycle(),
        module_probabilities=torch.ones(1, 3, dtype=torch.float64) / 3,
        allocation_scores=torch.zeros(1, 3, 2, dtype=torch.float64),
    )
    with pytest.raises(IrisLossError):
        aggregate_workspace_usage_v1((_cycle(), changed))
