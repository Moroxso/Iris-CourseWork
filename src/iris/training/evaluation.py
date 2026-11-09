"""Exact token-weighted streaming aggregation for teacher-forced evaluation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

import torch

from iris.training.outputs import IrisTeacherForcedOutput

_MAX_FLOAT64_LOG = math.log(sys.float_info.max)


class IrisEvaluationAggregationError(RuntimeError):
    """Raised when teacher-forced evaluation metrics cannot be aggregated safely."""


@dataclass(frozen=True, slots=True)
class IrisTokenWeightedEvaluationSummary:
    """One immutable snapshot of token-weighted teacher-forced metrics."""

    example_count: int
    supervised_token_count: int
    mean_loss: float
    perplexity: float | None


class IrisTokenWeightedEvaluationAccumulator:
    """Accumulate detached per-example mean losses without retaining model outputs."""

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: tuple[int, int, Fraction] = (0, 0, Fraction(0))

    def update(self, output: IrisTeacherForcedOutput) -> None:
        """Add one validated evaluation output as one transactional state change."""
        if type(output) is not IrisTeacherForcedOutput:
            message = "output must be exactly IrisTeacherForcedOutput"
            raise IrisEvaluationAggregationError(message)

        loss = output.loss
        token_count = output.supervised_token_count
        if type(token_count) is not int or token_count <= 0:
            message = "supervised_token_count must be an exact positive integer"
            raise IrisEvaluationAggregationError(message)
        if type(loss) is not torch.Tensor or loss.ndim != 0:
            message = "Evaluation loss must be an exact scalar torch.Tensor"
            raise IrisEvaluationAggregationError(message)
        try:
            valid_tensor = (
                torch.is_floating_point(loss)
                and not loss.requires_grad
                and loss.grad_fn is None
                and not loss.is_inference()
            )
        except Exception as error:
            message = "Failed to inspect evaluation loss"
            raise IrisEvaluationAggregationError(message) from error
        if not valid_tensor:
            message = "Evaluation loss must be a floating ordinary detached tensor"
            raise IrisEvaluationAggregationError(message)

        try:
            loss_value = float(loss.item())
        except Exception as error:
            message = "Failed to materialize evaluation loss"
            raise IrisEvaluationAggregationError(message) from error
        if not math.isfinite(loss_value) or loss_value < 0.0:
            message = "Evaluation loss must be finite and nonnegative"
            raise IrisEvaluationAggregationError(message)

        example_count, accumulated_tokens, weighted_loss = self._state
        try:
            candidate_state = (
                example_count + 1,
                accumulated_tokens + token_count,
                weighted_loss + Fraction.from_float(loss_value) * token_count,
            )
        except Exception as error:
            message = "Failed to accumulate token-weighted evaluation loss"
            raise IrisEvaluationAggregationError(message) from error
        self._state = candidate_state

    def summary(self) -> IrisTokenWeightedEvaluationSummary:
        """Return a non-destructive snapshot, or reject an empty stream."""
        example_count, token_count, weighted_loss = self._state
        if example_count == 0:
            message = "Cannot summarize an empty evaluation stream"
            raise IrisEvaluationAggregationError(message)
        try:
            mean_loss = float(weighted_loss / token_count)
        except Exception as error:
            message = "Failed to compute token-weighted mean loss"
            raise IrisEvaluationAggregationError(message) from error
        if not math.isfinite(mean_loss) or mean_loss < 0.0:
            message = "Token-weighted mean loss is invalid"
            raise IrisEvaluationAggregationError(message)

        perplexity: float | None
        if mean_loss > _MAX_FLOAT64_LOG:
            perplexity = None
        else:
            try:
                candidate_perplexity = math.exp(mean_loss)
            except OverflowError:
                perplexity = None
            else:
                perplexity = candidate_perplexity if math.isfinite(candidate_perplexity) else None
        return IrisTokenWeightedEvaluationSummary(
            example_count=example_count,
            supervised_token_count=token_count,
            mean_loss=mean_loss,
            perplexity=perplexity,
        )
