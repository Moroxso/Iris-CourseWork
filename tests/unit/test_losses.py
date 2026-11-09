"""Analytical values, gradient flow and strict loss boundaries; no model loading."""

from math import log
from typing import Any, Literal

import pytest
import torch

from iris.training.losses import (
    PROBE_FACT_FIELDS,
    IrisLossError,
    IrisLossWeights,
    combine_iris_losses,
    uniform_utilization_loss,
    workspace_diversity_loss,
    workspace_norm_loss,
    workspace_probe_loss,
)


def _tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, requires_grad=True)


def _probes(
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = {name: _tensor([[0.0, 0.0], [0.0, 0.0]], dtype=dtype) for name in PROBE_FACT_FIELDS}
    targets = {name: torch.tensor([0, 1], dtype=torch.int64) for name in PROBE_FACT_FIELDS}
    return logits, targets


def test_source_weights_and_exact_five_term_total_with_gradient_coefficients() -> None:
    weights = IrisLossWeights()
    assert weights.model_dump() == {
        "probe": 0.05,
        "diversity": 0.01,
        "utilization": 0.01,
        "norm": 0.001,
    }
    terms = tuple(_tensor(float(index)) for index in range(1, 6))
    result = combine_iris_losses(
        language_model_loss=terms[0],
        probe_loss=terms[1],
        diversity_loss=terms[2],
        utilization_loss=terms[3],
        norm_loss=terms[4],
        weights=weights,
    )
    assert result.total.item() == pytest.approx(1 + 0.05 * 2 + 0.01 * 3 + 0.01 * 4 + 0.001 * 5)
    torch.autograd.backward(result.total)
    for term, expected in zip(terms, (1.0, 0.05, 0.01, 0.01, 0.001), strict=True):
        assert term.grad is not None and term.grad.item() == pytest.approx(expected)
    assert result.weights is not weights
    assert result.weights == weights


def test_diversity_uses_only_ordered_distinct_pairs_and_all_slots() -> None:
    slots = _tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    # Two ordered identical-vector pairs, four orthogonal pairs; denominator 3*2.
    assert workspace_diversity_loss(slots).item() == pytest.approx(2 * 0.7 / 6)
    opposite = _tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    assert workspace_diversity_loss(opposite).item() == 0
    assert workspace_diversity_loss(slots, margin=1.0).item() == 0
    assert workspace_diversity_loss(slots, margin=0.0).item() == pytest.approx(1 / 3)


def test_diversity_batch_mean_is_not_cross_example_pairing() -> None:
    slots = _tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]]])
    assert workspace_diversity_loss(slots).item() == pytest.approx(0.35)


def test_single_slot_ablation_has_graph_attached_zero_diversity() -> None:
    slots = _tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    loss = workspace_diversity_loss(slots)
    assert loss.item() == 0 and loss.requires_grad
    torch.autograd.backward(loss)
    assert slots.grad is not None and torch.equal(slots.grad, torch.zeros_like(slots))


def test_zero_vectors_have_explicit_finite_cosine_extension() -> None:
    slots = _tensor([[[0.0, 0.0], [0.0, 0.0]]])
    loss = workspace_diversity_loss(slots)
    assert loss.item() == 0
    torch.autograd.backward(loss)
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())


def test_norm_formula_includes_empty_slots_and_targets_sqrt_width() -> None:
    slots = _tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    assert workspace_norm_loss(slots).item() == pytest.approx((0 + 4) / 2)
    both_empty = _tensor([[[0.0] * 4] * 2])
    loss = workspace_norm_loss(both_empty)
    assert loss.item() == 4
    torch.autograd.backward(loss)
    assert both_empty.grad is not None and bool(torch.isfinite(both_empty.grad).all())


@pytest.mark.parametrize(
    "modules,slots,expected",
    [
        ([0.5, 0.5], [0.25] * 4, 0.0),
        ([1.0, 0.0], [1.0, 0.0, 0.0, 0.0], log(2) + log(4)),
        ([0.75, 0.25], [0.5, 0.5], 0.75 * log(1.5) + 0.25 * log(0.5)),
        ([1.0], [1.0], 0.0),
    ],
)
def test_utilization_is_forward_kl_to_uniform_in_nats(
    modules: list[float], slots: list[float], expected: float
) -> None:
    module_p, slot_p = _tensor(modules), _tensor(slots)
    result = uniform_utilization_loss(module_p, slot_p)
    assert result.item() == pytest.approx(expected)
    torch.autograd.backward(result)
    for tensor in (module_p, slot_p):
        assert tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
        if bool((tensor == 0).any()):
            assert torch.equal(tensor.grad[tensor == 0], torch.zeros_like(tensor.grad[tensor == 0]))


def test_utilization_distribution_estimation_is_owned_by_caller() -> None:
    # Each task exclusively selects one module, but aggregate use is balanced.
    # Mean KL per task would be log(2); KL of the supplied aggregate is zero.
    usage = _tensor([[1.0, 0.0], [0.0, 1.0]])
    assert uniform_utilization_loss(usage.mean(dim=0), _tensor([0.5, 0.5])).item() == 0
    with pytest.raises(IrisLossError):
        uniform_utilization_loss(usage, _tensor([0.5, 0.5]))


@pytest.mark.parametrize("reduction,factor", [("sum", 4), ("mean", 1)])
def test_four_probe_cross_entropies_have_explicit_head_reduction(
    reduction: Literal["sum", "mean"], factor: int
) -> None:
    logits, labels = _probes()
    result = workspace_probe_loss(logits, labels, head_reduction=reduction)
    assert result.item() == pytest.approx(factor * log(2))
    torch.autograd.backward(result)
    for name in PROBE_FACT_FIELDS:
        gradient = logits[name].grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        assert gradient[0, 0] < 0 and gradient[0, 1] > 0
        assert gradient[1, 0] > 0 and gradient[1, 1] < 0


def test_probe_vocabularies_can_differ_and_reduction_averages_examples_within_head() -> None:
    logits, labels = _probes()
    logits["operation"] = _tensor([[0.0] * 3] * 2)
    labels["operation"] = torch.tensor([0, 2])
    expected = (3 * log(2) + log(3)) / 4
    assert workspace_probe_loss(logits, labels, head_reduction="mean").item() == pytest.approx(
        expected
    )
    # Mapping insertion order must not pair a head with a different head's labels.
    reversed_logits = dict(reversed(list(logits.items())))
    assert torch.equal(
        workspace_probe_loss(reversed_logits, labels, head_reduction="mean"),
        workspace_probe_loss(logits, labels, head_reduction="mean"),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_workspace_and_probe_losses_accumulate_safely_without_mutating_inputs(
    dtype: torch.dtype,
) -> None:
    slots = _tensor([[[1.0, 0.5], [0.25, 2.0]]], dtype=dtype)
    original = slots.detach().clone()
    logits, labels = _probes(dtype=dtype)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = (
            workspace_diversity_loss(slots),
            workspace_norm_loss(slots),
            workspace_probe_loss(logits, labels, head_reduction="mean"),
        )
    assert all(loss.dtype == expected_dtype for loss in losses)
    torch.autograd.backward(torch.stack(losses).sum())
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())
    assert torch.equal(slots.detach(), original)


def test_numerical_gradients_for_nonboundary_workspace_and_soft_usage() -> None:
    slots = _tensor([[[1.2, 0.3, 0.4], [0.8, 0.7, 0.2], [-0.2, 0.9, 0.4]]])
    assert torch.autograd.gradcheck(workspace_diversity_loss, (slots,))
    assert torch.autograd.gradcheck(workspace_norm_loss, (slots,))
    modules = _tensor([0.2, -0.5, 1.2])
    slot_logits = _tensor([0.7, -0.3])

    def usage(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return uniform_utilization_loss(left.softmax(dim=0), right.softmax(dim=0))

    assert torch.autograd.gradcheck(usage, (modules, slot_logits))


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(2, 3),
        torch.zeros(0, 2, 3),
        torch.zeros(1, 0, 3),
        torch.zeros(1, 2, 0),
        torch.zeros(1, 2, 3, dtype=torch.int64),
        torch.zeros(1, 2, 3, dtype=torch.complex64),
        torch.full((1, 2, 3), float("nan")),
        torch.full((1, 2, 3), float("inf")),
    ],
)
def test_bad_workspace_tensors_are_rejected(value: torch.Tensor) -> None:
    with pytest.raises(IrisLossError):
        workspace_diversity_loss(value)
    with pytest.raises(IrisLossError):
        workspace_norm_loss(value)


@pytest.mark.parametrize("margin", [True, "0.3", -1.01, 1.01, float("nan"), float("inf")])
def test_invalid_diversity_margin_is_rejected(margin: Any) -> None:
    with pytest.raises(IrisLossError):
        workspace_diversity_loss(_tensor([[[1.0], [2.0]]]), margin=margin)


@pytest.mark.parametrize(
    "probabilities",
    [
        _tensor([0.0, 0.0]),
        _tensor([-0.1, 1.1]),
        _tensor([0.4, 0.4]),
        _tensor([1.0, 1.0]),
        _tensor([float("nan"), 0.0]),
        _tensor([float("inf"), 0.0]),
        _tensor([0.5, 0.5], dtype=torch.float16),
    ],
)
def test_invalid_usage_is_not_silently_normalized(probabilities: torch.Tensor) -> None:
    with pytest.raises(IrisLossError):
        uniform_utilization_loss(probabilities, _tensor([0.5, 0.5]))


@pytest.mark.parametrize("kind", ["missing", "extra", "batch", "labels", "label_dtype", "nan"])
def test_invalid_probe_contract_is_rejected(kind: str) -> None:
    logits, labels = _probes()
    if kind == "missing":
        labels.pop("operation")
    elif kind == "extra":
        logits["task_type"] = _tensor([[0.0, 0.0]] * 2)
    elif kind == "batch":
        logits["operation"] = _tensor([[0.0, 0.0]])
    elif kind == "labels":
        labels["operation"] = torch.tensor([0, 2])
    elif kind == "label_dtype":
        labels["operation"] = torch.tensor([False, True])
    else:
        logits["operation"] = _tensor([[float("nan"), 0.0]] * 2)
    with pytest.raises(IrisLossError):
        workspace_probe_loss(logits, labels, head_reduction="mean")


def test_loss_combiner_owns_revalidated_weights_and_rejects_bad_scalars() -> None:
    arguments = {
        "language_model_loss": _tensor(1.0),
        "probe_loss": _tensor(2.0),
        "diversity_loss": _tensor(3.0),
        "utilization_loss": _tensor(4.0),
        "norm_loss": _tensor(5.0),
    }
    weights = IrisLossWeights().model_copy(update={"norm": float("nan")})
    with pytest.raises(IrisLossError):
        combine_iris_losses(**arguments, weights=weights)
    for bad in (_tensor(-1.0), _tensor([1.0]), _tensor(float("inf"))):
        with pytest.raises(IrisLossError):
            combine_iris_losses(**(arguments | {"probe_loss": bad}), weights=IrisLossWeights())


def test_finite_inputs_with_overflowing_arithmetic_fail_instead_of_false_zero() -> None:
    enormous = _tensor([[[1e30, 1e30], [1e30, 1e30]]], dtype=torch.float32)
    try:
        diversity = workspace_diversity_loss(enormous)
    except IrisLossError:
        pass  # A backend whose vector norm overflows must fail, not report zero diversity.
    else:
        assert diversity.item() == pytest.approx(0.7)
    with pytest.raises(IrisLossError):
        workspace_norm_loss(enormous)
    assert workspace_diversity_loss(enormous.to(dtype=torch.float64)).item() == pytest.approx(0.7)


def test_combiner_does_not_reinterpret_or_renormalize_the_supplied_lm_loss() -> None:
    arguments = {
        "probe_loss": _tensor(0.0),
        "diversity_loss": _tensor(0.0),
        "utilization_loss": _tensor(0.0),
        "norm_loss": _tensor(0.0),
    }
    for supplied in (0.5, 500.0):
        result = combine_iris_losses(
            language_model_loss=_tensor(supplied), **arguments, weights=IrisLossWeights()
        )
        assert result.total.item() == supplied
