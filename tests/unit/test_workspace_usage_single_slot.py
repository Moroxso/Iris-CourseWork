"""Actual singleton simplex identity, numerical roundoff and zero allocation gradient."""

from itertools import product

import pytest
import torch

from iris.training.losses import IrisLossError, uniform_utilization_loss
from iris.training.outputs import IrisTeacherCycleTraceV1
from iris.training.usage import aggregate_workspace_usage_v1


def cycle(probabilities: torch.Tensor, scores: torch.Tensor) -> IrisTeacherCycleTraceV1:
    batch, modules = probabilities.shape
    winners = torch.zeros(batch, modules, dtype=torch.bool)
    winners[:, 0] = True
    return IrisTeacherCycleTraceV1(probabilities, scores, winners, winners.unsqueeze(-1))


def test_native_fp32_softmax_roundoff_cannot_make_the_only_slot_greater_than_one() -> None:
    # Deterministic grid of REAL softmax inputs, not fabricated non-unit half data.
    generator = torch.Generator(device="cpu").manual_seed(42)
    logits = torch.randn(256, 4, generator=generator, dtype=torch.float32)
    probabilities = logits.softmax(-1)
    assert bool((probabilities.sum(-1) > 1.0).any())
    scores = torch.zeros(256, 4, 1, dtype=torch.float32)
    result = aggregate_workspace_usage_v1((cycle(probabilities, scores),))
    assert torch.equal(result.slot_probabilities, torch.ones(1, dtype=torch.float32))
    assert uniform_utilization_loss(
        result.module_probabilities, result.slot_probabilities
    ).isfinite()


@pytest.mark.parametrize(
    "dtype,batch,cycles,modules",
    tuple(
        product(
            (torch.float32, torch.float64),
            (1, 3),
            (1, 3),
            (2, 4, 64),
        )
    ),
)
def test_all_single_slot_soft_usage_is_exact_one_and_preserves_module_and_zero_slot_gradients(
    dtype: torch.dtype,
    batch: int,
    cycles: int,
    modules: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(2026)
    logits = torch.randn(batch, modules, generator=generator, dtype=dtype, requires_grad=True)
    scores = torch.randn(batch, modules, 1, generator=generator, dtype=dtype, requires_grad=True)
    p = logits.softmax(-1)
    frames = tuple(cycle(p, scores + index) for index in range(cycles))
    result = aggregate_workspace_usage_v1(frames)
    # Preserve the ORIGINAL two-stage batch/cycle arithmetic, including its
    # ordinary float rounding; do not replace it with a single batch mean.
    expected_modules = torch.stack([p.mean(0) for _ in range(cycles)]).mean(0)
    assert torch.equal(result.module_probabilities, expected_modules)
    assert torch.equal(result.slot_probabilities, torch.ones(1, dtype=dtype))
    assert result.slot_probabilities.requires_grad
    reference = uniform_utilization_loss(result.module_probabilities, torch.ones(1, dtype=dtype))
    loss = uniform_utilization_loss(result.module_probabilities, result.slot_probabilities)
    expected_module_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    actual_module_gradient, allocation_gradient = torch.autograd.grad(loss, (logits, scores))
    assert torch.equal(actual_module_gradient, expected_module_gradient)
    assert torch.equal(allocation_gradient, torch.zeros_like(scores))
    assert not result.hard_slot_frequencies.requires_grad


def test_single_slot_logits_gradcheck_and_invalid_module_mass_still_denied() -> None:
    logits = torch.tensor([[0.5, -0.4, 0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    scores = torch.zeros(1, 4, 1, dtype=torch.float64, requires_grad=True)

    def objective(values: torch.Tensor, allocation: torch.Tensor) -> torch.Tensor:
        result = aggregate_workspace_usage_v1((cycle(values.softmax(-1), allocation),))
        return uniform_utilization_loss(result.module_probabilities, result.slot_probabilities)

    assert torch.autograd.gradcheck(objective, (logits, scores))
    with pytest.raises(IrisLossError):
        aggregate_workspace_usage_v1((cycle(torch.full((1, 4), 0.2, dtype=torch.float64), scores),))
    with pytest.raises(IrisLossError):
        uniform_utilization_loss(torch.full((4,), 0.25), torch.tensor([1.0000001192092896]))
