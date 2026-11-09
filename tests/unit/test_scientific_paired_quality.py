"""Paired measurement arithmetic, strict no-NLL contract and baseline compatibility."""

import hashlib
import json
from itertools import product
from typing import Any

import pytest

from iris.artifacts.scientific_evaluation_v2 import (
    IrisScientificEvaluationV2,
    ScientificAnswerMeasurementsV2,
    ScientificScoredRowV2,
)
from iris.artifacts.scientific_intervention_evaluation_v1 import (
    ScientificPairedQualityV1,
    ScientificPairedScoredRowV1,
    scientific_paired_quality_v1,
)
from iris.integrations.python_comparison import PythonOutcome


def measurement(outcome: PythonOutcome, *, index: int = 1) -> ScientificAnswerMeasurementsV2:
    return ScientificAnswerMeasurementsV2(
        example_id=f"test-{index}",
        generation_example_sha256="a" * 64,
        scoring_example_sha256="b" * 64,
        task_type="delayed_goal",
        output_sha256="c" * 64,
        output_byte_length=12,
        outcome=outcome,
        reason="semantic_match"
        if outcome == "correct"
        else "content_mismatch"
        if outcome == "incorrect"
        else "invalid_envelope",
        envelope_valid=outcome != "invalid_output",
        structured_exact_match=outcome == "correct",
        tool_action_match=None,
        conflict_flag_match=None,
        case_count=0,
        passed_case_count=0,
        execution=None,
    )


def test_all_nine_outcome_transitions_have_exact_denominators_and_no_nll() -> None:
    outcomes: tuple[PythonOutcome, ...] = ("correct", "incorrect", "invalid_output")
    rows = tuple(
        ScientificPairedScoredRowV1(
            difficulty=1,
            bound_plan_sha256="d" * 64,
            control=measurement(left, index=index),
            intervened=measurement(right, index=index),
        )
        for index, (left, right) in enumerate(product(outcomes, repeat=2), start=1)
    )
    result = scientific_paired_quality_v1(rows)
    assert (
        result.both_correct,
        result.control_only_correct,
        result.intervened_only_correct,
        result.neither_correct,
    ) == (1, 2, 2, 4)
    assert result.control.examples == result.intervened.examples == 9
    assert result.control.correct == result.intervened.correct == 3
    assert result.rate_deltas["task_success_rate"] == result.relative_task_success_drop == 0
    assert "nll" not in result.model_dump_json()
    assert result.rate_deltas["test_pass_rate"] is None
    assert result.control.rates["task_success_rate"] == 1 / 3


@pytest.mark.parametrize("empty", [True, False])
def test_zero_control_success_never_divides_by_zero(empty: bool) -> None:
    rows = (
        ()
        if empty
        else (
            ScientificPairedScoredRowV1(
                difficulty=2,
                bound_plan_sha256="d" * 64,
                control=measurement("incorrect"),
                intervened=measurement("correct"),
            ),
        )
    )
    result = scientific_paired_quality_v1(rows)
    assert result.relative_task_success_drop is None
    assert result.rate_deltas["task_success_rate"] == (None if empty else 1.0)


def test_relative_improvement_is_negative_drop_and_deltas_use_rates_not_percent_points() -> None:
    rows = tuple(
        ScientificPairedScoredRowV1(
            difficulty=3,
            bound_plan_sha256="d" * 64,
            control=measurement("correct" if index == 1 else "incorrect", index=index),
            intervened=measurement("correct", index=index),
        )
        for index in (1, 2)
    )
    result = scientific_paired_quality_v1(rows)
    assert result.relative_task_success_drop == -1.0
    assert result.rate_deltas["task_success_rate"] == 0.5


@pytest.mark.parametrize(
    "update",
    [
        {"both_correct": 1},
        {"neither_correct": 1},
        {"relative_task_success_drop": 0.0},
        {"rate_deltas": {}},
        {"rate_deltas": {"task_success_rate": float("nan")}},
    ],
)
def test_malformed_aggregate_cannot_be_stored(update: dict[str, Any]) -> None:
    empty = scientific_paired_quality_v1(())
    with pytest.raises(ValueError):
        ScientificPairedQualityV1.model_validate({**empty.model_dump(mode="python"), **update})


@pytest.mark.parametrize(
    "update",
    [
        {"example_id": "test-2"},
        {"generation_example_sha256": "e" * 64},
        {"scoring_example_sha256": "e" * 64},
        {"task_type": "next_tool_prediction", "tool_action_match": False},
    ],
)
def test_cross_example_or_task_pairing_is_rejected(update: dict[str, Any]) -> None:
    control = measurement("incorrect")
    with pytest.raises(ValueError):
        ScientificPairedScoredRowV1(
            difficulty=1,
            bound_plan_sha256="d" * 64,
            control=control,
            intervened=control.model_copy(update=update),
        )


def test_shared_measurements_do_not_widen_baseline_or_invent_counterfactual_nll() -> None:
    answer = measurement("correct")
    with pytest.raises(ValueError):
        ScientificScoredRowV2.model_validate(answer.model_dump(mode="python"))
    with pytest.raises(ValueError):
        ScientificAnswerMeasurementsV2.model_validate(
            {**answer.model_dump(mode="python"), "target_nll_sum": 0.0}
        )
    # Verified against the pre-change installed wheel, not guessed from current types.
    digest = hashlib.sha256(
        json.dumps(IrisScientificEvaluationV2.model_json_schema(), sort_keys=True).encode()
    ).hexdigest()
    assert digest == "411ae3927d1bdabc842af0130b4dfff70adbb9cfc89d8765f4ed26b131653450"
