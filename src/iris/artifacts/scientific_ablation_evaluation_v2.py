"""Distinct mode-bound hash-only answer measurements; not scientific launch approval."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_capture_v1 import (
    IrisScientificAblationCaptureCompleteV1,
    IrisScientificAblationCapturePlanV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    ScientificFamilyQualityV1,
    ScientificQualityCountsV1,
    scientific_quality_counts_v1,
)
from iris.artifacts.scientific_evaluation_v2 import (
    ScientificAnswerMeasurementsV2,
    ScientificPythonLimitsV2,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class ScientificAblationScoredRowV2(ScientificAnswerMeasurementsV2):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    capture_ack_sha256: Digest
    teacher_invocation_sha256: Digest
    teacher_target_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    target_nll_sum: float = Field(ge=0.0)


class ScientificAblationEvaluationPayloadV2(_Strict):
    kind: Literal["iris-scientific-ablation-evaluation-v2"] = (
        "iris-scientific-ablation-evaluation-v2"
    )
    scoring_profile: Literal["iris-synthetic-01-scoring-v2"] = "iris-synthetic-01-scoring-v2"
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"
    nll_scope: Literal["copied_from_bound_capture_not_recomputed"] = (
        "copied_from_bound_capture_not_recomputed"
    )
    plan: IrisScientificAblationCapturePlanV1
    capture: IrisScientificAblationCaptureCompleteV1
    python_resource_scope: Literal["source_section15_v2", "not_applicable_no_code"]
    python_limits: ScientificPythonLimitsV2 | None
    python_policy: PythonSandboxPolicyV2 | None
    rows: tuple[ScientificAblationScoredRowV2, ...] = Field(min_length=1, max_length=5000)
    overall: ScientificQualityCountsV1
    families: tuple[ScientificFamilyQualityV1, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def bound_complete_aggregation(self) -> Self:
        for manifest, domain in (
            (self.plan, "iris/scientific-ablation-capture-plan/v1"),
            (self.capture, "iris/scientific-ablation-capture-complete/v1"),
        ):
            if manifest.sha256 != domain_sha256(
                domain, canonical_json_bytes(manifest.payload.model_dump(mode="json"))
            ):
                raise ValueError("Evaluation plan/capture checksum differs")
        p, c = self.plan.payload, self.capture.payload
        if (
            c.plan_sha256 != self.plan.sha256
            or c.input_sha256 != p.input.sha256
            or c.example_count != len(p.rows)
            or len(self.rows) != len(p.rows)
            or c.acknowledgement_head_sha256 != self.rows[-1].capture_ack_sha256
        ):
            raise ValueError("Complete evaluation coverage/ack head differs")
        for row, expected in zip(self.rows, p.rows, strict=True):
            if (
                row.row_index != expected.row_index
                or row.example_id != expected.example_id
                or row.generation_example_sha256 != expected.example_sha256
                or row.task_type != expected.task_type
                or row.target_token_count != expected.target_token_count
                or row.teacher_target_sha256 != expected.teacher_target_sha256
                or row.teacher_invocation_sha256 != expected.teacher_invocation_sha256
            ):
                raise ValueError("Scored row differs from owned mode-bound inventory")
        repair = any(row.task_type == "split_spec_repair" for row in self.rows)
        if (
            self.python_resource_scope
            != ("source_section15_v2" if repair else "not_applicable_no_code")
            or self.python_limits != (ScientificPythonLimitsV2() if repair else None)
            or repair != (self.python_policy is not None)
            or (self.python_policy is not None and self.python_policy.wall_time_ms != 5000)
            or any(
                row.execution is not None and row.execution.policy != self.python_policy
                for row in self.rows
            )
        ):
            raise ValueError("Exact source-v2 repair policy/limits required")
        actual = scientific_quality_counts_v1(self.rows)
        if (
            self.overall != actual
            or actual.target_tokens != c.target_token_count
            or actual.target_tokens != p.input.payload.target_token_count
            or actual.target_nll_sum != c.target_nll_sum
            or self.families
            != tuple(
                ScientificFamilyQualityV1(
                    task_type=task,
                    counts=scientific_quality_counts_v1(
                        tuple(row for row in self.rows if row.task_type == task)
                    ),
                )
                for task in TASKS
            )
        ):
            raise ValueError("Evaluation counts/target NLL differ from exact capture")
        if p.input.payload.purpose == "scientific" and tuple(
            family.counts.examples for family in self.families
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific evaluation requires all 5000 source rows")
        return self


class IrisScientificAblationEvaluationV2(_Strict):
    sha256: Digest
    payload: ScientificAblationEvaluationPayloadV2


__all__ = [
    "IrisScientificAblationEvaluationV2",
    "ScientificAblationEvaluationPayloadV2",
    "ScientificAblationScoredRowV2",
]
