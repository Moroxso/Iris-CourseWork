"""Hash-only paired answer quality: exact task/difficulty groups, no invented NLL."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_evaluation_v2 import (
    ScientificAnswerMeasurementsV2,
    ScientificPythonLimitsV2,
)
from iris.data.schemas import SyntheticTask


class ScientificAnswerQualityCountsV1(_Strict):
    """Answer-only denominators; absence of a family is undefined, not zero quality."""

    examples: int = Field(ge=0)
    correct: int = Field(ge=0)
    incorrect: int = Field(ge=0)
    invalid_output: int = Field(ge=0)
    envelope_valid: int = Field(ge=0)
    structured_exact_match: int = Field(ge=0)
    code_examples: int = Field(ge=0)
    code_executed: int = Field(ge=0)
    code_correct: int = Field(ge=0)
    code_cases: int = Field(ge=0)
    code_cases_passed: int = Field(ge=0)
    tool_examples: int = Field(ge=0)
    tool_action_correct: int = Field(ge=0)
    conflict_examples: int = Field(ge=0)
    conflict_flag_correct: int = Field(ge=0)

    @model_validator(mode="after")
    def check_denominators(self) -> Self:
        if (
            self.correct + self.incorrect + self.invalid_output != self.examples
            or not self.correct <= self.envelope_valid <= self.examples
            or self.structured_exact_match > self.envelope_valid
            or not self.code_correct <= self.code_executed <= self.code_examples
            or self.code_correct > self.correct
            or self.code_cases_passed > self.code_cases
            or self.code_cases < 4 * self.code_examples
            or (not self.code_examples and self.code_cases)
            or self.tool_action_correct > self.tool_examples
            or self.conflict_flag_correct > self.conflict_examples
            or self.code_examples + self.tool_examples + self.conflict_examples > self.examples
        ):
            raise ValueError("Answer quality counts contradict their denominators")
        return self

    @property
    def rates(self) -> dict[str, float | None]:
        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        return {
            "task_success_rate": ratio(self.correct, self.examples),
            "invalid_output_rate": ratio(self.invalid_output, self.examples),
            "json_schema_validity": ratio(self.envelope_valid, self.examples),
            "structured_exact_match": ratio(self.structured_exact_match, self.examples),
            "code_execution_rate": ratio(self.code_executed, self.code_examples),
            "code_pass_rate": ratio(self.code_correct, self.code_examples),
            "test_pass_rate": ratio(self.code_cases_passed, self.code_cases),
            "tool_choice_accuracy": ratio(self.tool_action_correct, self.tool_examples),
            "conflict_flag_accuracy": ratio(self.conflict_flag_correct, self.conflict_examples),
        }


def scientific_answer_quality_counts_v1(
    rows: tuple[ScientificAnswerMeasurementsV2, ...],
) -> ScientificAnswerQualityCountsV1:
    return ScientificAnswerQualityCountsV1(
        examples=len(rows),
        correct=sum(row.outcome == "correct" for row in rows),
        incorrect=sum(row.outcome == "incorrect" for row in rows),
        invalid_output=sum(row.outcome == "invalid_output" for row in rows),
        envelope_valid=sum(row.envelope_valid for row in rows),
        structured_exact_match=sum(row.structured_exact_match for row in rows),
        code_examples=sum(row.task_type == "split_spec_repair" for row in rows),
        code_executed=sum(row.execution is not None for row in rows),
        code_correct=sum(
            row.task_type == "split_spec_repair" and row.outcome == "correct" for row in rows
        ),
        code_cases=sum(row.case_count for row in rows),
        code_cases_passed=sum(row.passed_case_count for row in rows),
        tool_examples=sum(row.task_type == "next_tool_prediction" for row in rows),
        tool_action_correct=sum(row.tool_action_match is True for row in rows),
        conflict_examples=sum(row.task_type == "conflict_resolution" for row in rows),
        conflict_flag_correct=sum(row.conflict_flag_match is True for row in rows),
    )


class ScientificPairedScoredRowV1(_Strict):
    difficulty: int = Field(ge=1, le=3)
    bound_plan_sha256: Digest
    control: ScientificAnswerMeasurementsV2
    intervened: ScientificAnswerMeasurementsV2

    @model_validator(mode="after")
    def same_example(self) -> Self:
        left, right = self.control, self.intervened
        if (
            left.example_id != right.example_id
            or left.generation_example_sha256 != right.generation_example_sha256
            or left.scoring_example_sha256 != right.scoring_example_sha256
            or left.task_type != right.task_type
            or left.case_count != right.case_count
        ):
            raise ValueError("Paired scores must refer to the same held-out example and cases")
        return self


def _rate_deltas(
    control: ScientificAnswerQualityCountsV1, intervened: ScientificAnswerQualityCountsV1
) -> dict[str, float | None]:
    changed = intervened.rates
    result: dict[str, float | None] = {}
    for key, value in control.rates.items():
        other = changed[key]
        result[key] = None if value is None or other is None else other - value
    return result


class ScientificPairedQualityV1(_Strict):
    control: ScientificAnswerQualityCountsV1
    intervened: ScientificAnswerQualityCountsV1
    both_correct: int = Field(ge=0)
    control_only_correct: int = Field(ge=0)
    intervened_only_correct: int = Field(ge=0)
    neither_correct: int = Field(ge=0)
    delta_convention: Literal["intervened_minus_control_rate"] = "intervened_minus_control_rate"
    rate_deltas: dict[str, float | None]
    relative_task_success_drop: float | None

    @model_validator(mode="after")
    def check_pairs(self) -> Self:
        left, right = self.control, self.intervened
        denominators = (
            "examples",
            "code_examples",
            "code_cases",
            "tool_examples",
            "conflict_examples",
        )
        if (
            any(getattr(left, name) != getattr(right, name) for name in denominators)
            or self.both_correct + self.control_only_correct != left.correct
            or self.both_correct + self.intervened_only_correct != right.correct
            or self.both_correct
            + self.control_only_correct
            + self.intervened_only_correct
            + self.neither_correct
            != left.examples
            or self.rate_deltas != _rate_deltas(left, right)
            or self.relative_task_success_drop
            != ((left.correct - right.correct) / left.correct if left.correct else None)
        ):
            raise ValueError("Paired counts/deltas contradict the same-example denominators")
        return self


def scientific_paired_quality_v1(
    rows: tuple[ScientificPairedScoredRowV1, ...],
) -> ScientificPairedQualityV1:
    control = scientific_answer_quality_counts_v1(tuple(row.control for row in rows))
    intervened = scientific_answer_quality_counts_v1(tuple(row.intervened for row in rows))
    both = sum(row.control.outcome == row.intervened.outcome == "correct" for row in rows)
    return ScientificPairedQualityV1(
        control=control,
        intervened=intervened,
        both_correct=both,
        control_only_correct=control.correct - both,
        intervened_only_correct=intervened.correct - both,
        neither_correct=len(rows) - control.correct - intervened.correct + both,
        rate_deltas=_rate_deltas(control, intervened),
        relative_task_success_drop=(
            (control.correct - intervened.correct) / control.correct if control.correct else None
        ),
    )


class ScientificPairedFamilyV1(_Strict):
    task_type: SyntheticTask
    quality: ScientificPairedQualityV1


class ScientificPairedDifficultyV1(_Strict):
    difficulty: int = Field(ge=1, le=3)
    quality: ScientificPairedQualityV1


class ScientificPairedFamilyDifficultyV1(ScientificPairedDifficultyV1):
    task_type: SyntheticTask


class ScientificInterventionEvaluationPayloadV1(_Strict):
    kind: Literal["iris-scientific-intervention-evaluation-v1"] = (
        "iris-scientific-intervention-evaluation-v1"
    )
    scoring_profile: Literal["iris-synthetic-01-scoring-v2"] = "iris-synthetic-01-scoring-v2"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"
    nll_scope: Literal["not_included_intervened_teacher_nll_not_measured"] = (
        "not_included_intervened_teacher_nll_not_measured"
    )
    difficulty_scope: Literal["source_row_difficulty_not_simple_complex_acceptance"] = (
        "source_row_difficulty_not_simple_complex_acceptance"
    )
    intervention_sha256: Digest
    baseline_generation_sha256: Digest
    plan_sha256: Digest
    request_graph_sha256: Digest
    request_sidecar_sha256: Digest
    execution_sha256: Digest
    terminal_event_sha256: Digest
    best_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1)
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    python_resource_scope: Literal["source_section15_v2", "not_applicable_no_code"]
    python_limits: ScientificPythonLimitsV2 | None
    python_policy: PythonSandboxPolicyV2 | None
    rows: tuple[ScientificPairedScoredRowV1, ...] = Field(min_length=1, max_length=5000)
    overall: ScientificPairedQualityV1
    families: tuple[ScientificPairedFamilyV1, ...] = Field(min_length=4, max_length=4)
    difficulties: tuple[ScientificPairedDifficultyV1, ...] = Field(min_length=3, max_length=3)
    family_difficulties: tuple[ScientificPairedFamilyDifficultyV1, ...] = Field(
        min_length=12, max_length=12
    )

    @model_validator(mode="after")
    def exact_aggregation(self) -> Self:
        if len({row.control.example_id for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate paired scored rows")
        repair = any(row.control.task_type == "split_spec_repair" for row in self.rows)
        if (
            self.python_resource_scope
            != ("source_section15_v2" if repair else "not_applicable_no_code")
            or self.python_limits != (ScientificPythonLimitsV2() if repair else None)
            or repair != (self.python_policy is not None)
            or (self.python_policy is not None and self.python_policy.wall_time_ms != 5000)
            or any(
                score.execution is not None and score.execution.policy != self.python_policy
                for row in self.rows
                for score in (row.control, row.intervened)
            )
        ):
            raise ValueError("Explicit source repair policy/limits must match both branches")
        if (
            self.overall != scientific_paired_quality_v1(self.rows)
            or self.families != paired_families_v1(self.rows)
            or self.difficulties != paired_difficulties_v1(self.rows)
            or self.family_difficulties != paired_family_difficulties_v1(self.rows)
        ):
            raise ValueError("Stored paired aggregates differ from the exact scored rows")
        if self.purpose == "scientific" and tuple(
            f.quality.control.examples for f in self.families
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific paired report requires all 5000 held-out rows")
        return self


def paired_families_v1(
    rows: tuple[ScientificPairedScoredRowV1, ...],
) -> tuple[ScientificPairedFamilyV1, ...]:
    return tuple(
        ScientificPairedFamilyV1(
            task_type=task,
            quality=scientific_paired_quality_v1(
                tuple(row for row in rows if row.control.task_type == task)
            ),
        )
        for task in TASKS
    )


def paired_difficulties_v1(
    rows: tuple[ScientificPairedScoredRowV1, ...],
) -> tuple[ScientificPairedDifficultyV1, ...]:
    return tuple(
        ScientificPairedDifficultyV1(
            difficulty=level,
            quality=scientific_paired_quality_v1(
                tuple(row for row in rows if row.difficulty == level)
            ),
        )
        for level in (1, 2, 3)
    )


def paired_family_difficulties_v1(
    rows: tuple[ScientificPairedScoredRowV1, ...],
) -> tuple[ScientificPairedFamilyDifficultyV1, ...]:
    return tuple(
        ScientificPairedFamilyDifficultyV1(
            task_type=task,
            difficulty=level,
            quality=scientific_paired_quality_v1(
                tuple(
                    row for row in rows if row.control.task_type == task and row.difficulty == level
                )
            ),
        )
        for task in TASKS
        for level in (1, 2, 3)
    )


class IrisScientificInterventionEvaluationV1(_Strict):
    sha256: Digest
    payload: ScientificInterventionEvaluationPayloadV1
