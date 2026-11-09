"""Pure contracts for token-weighted teacher-forced evaluation aggregation."""

from __future__ import annotations

import gc
import math
import random
import weakref
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from itertools import permutations
from typing import cast
from unittest.mock import patch

import pytest
import torch

import iris.training as training
from iris.training import (
    IrisEvaluationAggregationError,
    IrisTeacherForcedOutput,
    IrisTokenWeightedEvaluationAccumulator,
    IrisTokenWeightedEvaluationSummary,
)
from iris.workspace import WorkspaceState


class _OutputSubclass(IrisTeacherForcedOutput):
    pass


class _IntSubclass(int):
    pass


def _output(
    loss: object,
    *,
    token_count: object = 1,
    trace: object = (),
) -> IrisTeacherForcedOutput:
    return IrisTeacherForcedOutput(
        loss=cast(torch.Tensor, loss),
        supervised_token_count=cast(int, token_count),
        workspace_trace=cast(tuple[WorkspaceState, ...], trace),
    )


def _summary_after(
    *values: tuple[float, int],
) -> IrisTokenWeightedEvaluationSummary:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    for loss, token_count in values:
        accumulator.update(
            _output(torch.tensor(loss, dtype=torch.float64), token_count=token_count)
        )
    return accumulator.summary()


def test_public_contract_is_frozen_slotted_and_canonically_exported() -> None:
    summary = IrisTokenWeightedEvaluationSummary(
        example_count=2,
        supervised_token_count=5,
        mean_loss=1.25,
        perplexity=math.exp(1.25),
    )

    assert tuple(field.name for field in fields(summary)) == (
        "example_count",
        "supervised_token_count",
        "mean_loss",
        "perplexity",
    )
    with pytest.raises(FrozenInstanceError):
        summary.mean_loss = 2.0  # type: ignore[misc]
    assert not hasattr(summary, "__dict__")
    assert not hasattr(IrisTokenWeightedEvaluationAccumulator(), "__dict__")
    assert training.IrisEvaluationAggregationError is IrisEvaluationAggregationError
    assert training.IrisTokenWeightedEvaluationAccumulator is IrisTokenWeightedEvaluationAccumulator
    assert training.IrisTokenWeightedEvaluationSummary is IrisTokenWeightedEvaluationSummary
    assert {
        "IrisEvaluationAggregationError",
        "IrisTokenWeightedEvaluationAccumulator",
        "IrisTokenWeightedEvaluationSummary",
    }.issubset(training.__all__)
    assert issubclass(IrisEvaluationAggregationError, RuntimeError)


def test_empty_stream_has_no_synthetic_summary() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()

    with pytest.raises(IrisEvaluationAggregationError, match="empty evaluation stream"):
        accumulator.summary()


def test_single_output_preserves_mean_and_uses_natural_exponential_perplexity() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    loss = math.log(4.0)

    accumulator.update(_output(torch.tensor(loss, dtype=torch.float64), token_count=3))
    summary = accumulator.summary()

    assert summary.example_count == 1
    assert summary.supervised_token_count == 3
    assert summary.mean_loss == loss
    assert summary.perplexity == pytest.approx(4.0)


def test_unequal_examples_are_weighted_by_supervised_tokens() -> None:
    summary = _summary_after((1.0, 1), (3.0, 3))

    assert summary.example_count == 2
    assert summary.supervised_token_count == 4
    assert summary.mean_loss == 2.5
    assert summary.mean_loss != 2.0
    assert summary.perplexity == math.exp(2.5)


def test_exact_fraction_state_makes_all_update_orders_bit_identical() -> None:
    values = ((1.0e20, 1), (1.0, 3), (math.pi, 7), (0.25, 11))
    observed = {_summary_after(*ordering).mean_loss.hex() for ordering in permutations(values)}

    assert len(observed) == 1


def test_summary_is_idempotent_non_destructive_and_allows_later_updates() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(1.0), token_count=2))

    first = accumulator.summary()
    second = accumulator.summary()
    accumulator.update(_output(torch.tensor(3.0), token_count=2))
    third = accumulator.summary()

    assert first == second
    assert first.example_count == 1
    assert third.example_count == 2
    assert third.supervised_token_count == 4
    assert third.mean_loss == 2.0


def test_duplicate_updates_are_independent_stream_elements() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    output = _output(torch.tensor(1.5), token_count=2)

    accumulator.update(output)
    accumulator.update(output)

    assert accumulator.summary() == IrisTokenWeightedEvaluationSummary(
        example_count=2,
        supervised_token_count=4,
        mean_loss=1.5,
        perplexity=math.exp(1.5),
    )


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        object(),
        "not an output",
        _OutputSubclass(torch.tensor(1.0), 1, ()),
    ],
)
def test_rejects_every_non_exact_output_type_without_state_change(candidate: object) -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()

    with pytest.raises(IrisEvaluationAggregationError, match="exactly"):
        accumulator.update(cast(IrisTeacherForcedOutput, candidate))

    assert accumulator.summary() == before


def _inference_tensor() -> torch.Tensor:
    with torch.inference_mode():
        return torch.tensor(1.0)


@pytest.mark.parametrize(
    "loss_factory",
    [
        lambda: 1.0,
        lambda: torch.tensor([1.0]),
        lambda: torch.tensor(1, dtype=torch.int64),
        lambda: torch.tensor(1.0 + 0.0j),
        lambda: torch.tensor(1.0, requires_grad=True),
        _inference_tensor,
    ],
    ids=["not-tensor", "not-scalar", "integer", "complex", "attached", "inference"],
)
def test_rejects_malformed_loss_structure_transactionally(
    loss_factory: Callable[[], object],
) -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()

    with pytest.raises(IrisEvaluationAggregationError):
        accumulator.update(_output(loss_factory()))

    assert accumulator.summary() == before


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.25])
def test_rejects_nonfinite_or_negative_loss_transactionally(value: float) -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()

    with pytest.raises(IrisEvaluationAggregationError, match="finite and nonnegative"):
        accumulator.update(_output(torch.tensor(value, dtype=torch.float64)))

    assert accumulator.summary() == before


@pytest.mark.parametrize("token_count", [True, False, 0, -1, 1.0, _IntSubclass(1)])
def test_rejects_non_exact_or_nonpositive_token_counts_transactionally(
    token_count: object,
) -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()

    with pytest.raises(IrisEvaluationAggregationError, match="exact positive integer"):
        accumulator.update(_output(torch.tensor(1.0), token_count=token_count))

    assert accumulator.summary() == before


def test_loss_inspection_failure_preserves_exact_cause_and_state() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()
    failure = RuntimeError("inspection exploded")

    with (
        patch(
            "iris.training.evaluation.torch.is_floating_point",
            side_effect=failure,
        ),
        pytest.raises(IrisEvaluationAggregationError, match="inspect") as captured,
    ):
        accumulator.update(_output(torch.tensor(1.0)))

    assert captured.value.__cause__ is failure
    assert accumulator.summary() == before


def test_loss_materialization_failure_preserves_exact_cause_and_state() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    accumulator.update(_output(torch.tensor(2.0), token_count=2))
    before = accumulator.summary()

    with pytest.raises(IrisEvaluationAggregationError, match="materialize") as captured:
        accumulator.update(_output(torch.empty((), device="meta")))

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert accumulator.summary() == before


def test_perplexity_overflow_is_none_without_losing_finite_mean_loss() -> None:
    summary = _summary_after((1000.0, 7))

    assert summary.mean_loss == 1000.0
    assert math.isfinite(summary.mean_loss)
    assert summary.perplexity is None


def test_zero_loss_has_unit_perplexity() -> None:
    summary = _summary_after((0.0, 1))

    assert summary.mean_loss == 0.0
    assert summary.perplexity == 1.0


def test_arbitrarily_large_python_token_count_does_not_overflow_weighting() -> None:
    token_count = 10**400
    summary = _summary_after((1.25, token_count))

    assert summary.example_count == 1
    assert summary.supervised_token_count == token_count
    assert summary.mean_loss == 1.25
    assert summary.perplexity == math.exp(1.25)


def test_workspace_trace_is_neither_inspected_nor_required_for_metrics() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()

    accumulator.update(_output(torch.tensor(1.0), trace=object()))

    assert accumulator.summary().example_count == 1


def _add_transient_output(
    accumulator: IrisTokenWeightedEvaluationAccumulator,
) -> tuple[weakref.ReferenceType[torch.Tensor], weakref.ReferenceType[torch.Tensor]]:
    loss = torch.tensor(1.0)
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=1,
        state_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    loss_reference = weakref.ref(loss)
    slot_reference = weakref.ref(state.slots)
    accumulator.update(_output(loss, trace=(state,)))
    return loss_reference, slot_reference


def test_accumulator_does_not_retain_output_loss_or_workspace_trace() -> None:
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    loss_reference, slot_reference = _add_transient_output(accumulator)

    gc.collect()

    assert loss_reference() is None
    assert slot_reference() is None
    assert accumulator.summary().mean_loss == 1.0


def test_update_and_summary_do_not_mutate_inputs_or_random_state() -> None:
    loss = torch.tensor(1.75)
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=2,
        state_dim=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    state_tensors = tuple(getattr(state, field.name) for field in fields(state))
    state_snapshots = tuple(tensor.clone() for tensor in state_tensors)
    loss_snapshot = loss.clone()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    accumulator = IrisTokenWeightedEvaluationAccumulator()

    accumulator.update(_output(loss, token_count=2, trace=(state,)))
    accumulator.summary()

    assert torch.equal(loss, loss_snapshot)
    for tensor, snapshot in zip(state_tensors, state_snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
