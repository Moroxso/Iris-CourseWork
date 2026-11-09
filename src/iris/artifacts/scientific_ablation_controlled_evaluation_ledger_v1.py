"""Distinct hash-only score-plan/row/ack/complete protocol, not execution attestation."""

from math import fsum
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_controlled_capture_v1 import (
    IrisScientificAblationControlledCaptureCompleteV1,
    IrisScientificAblationControlledCapturePlanV1,
)
from iris.artifacts.scientific_ablation_controlled_evaluation_v1 import (
    ScientificAblationControlledScoredRowV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class ScientificAblationControlledEvaluationExpectationV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    capture_ack_sha256: Digest
    scoring_example_sha256: Digest
    output_sha256: Digest
    output_byte_length: int = Field(ge=0, le=4 * 1024**2)
    case_count: int = Field(ge=0, le=128)
    target_nll_sum: float = Field(ge=0.0)


class ScientificAblationControlledEvaluationPlanPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-evaluation-plan-v1"] = (
        "iris-scientific-ablation-controlled-evaluation-plan-v1"
    )
    scope: Literal["owned_score_plan_not_measurements_or_launch_approval"] = (
        "owned_score_plan_not_measurements_or_launch_approval"
    )
    scoring_profile: Literal["iris-synthetic-01-scoring-v2"] = "iris-synthetic-01-scoring-v2"
    generation_plan: IrisScientificAblationControlledCapturePlanV1
    capture: IrisScientificAblationControlledCaptureCompleteV1
    parquet_spec_sha256: tuple[Digest, Digest, Digest]
    python_policy: PythonSandboxPolicyV2 | None
    worker_sha256: Digest | None
    maximum_report_file_bytes: int = Field(ge=1, le=32 * 1024**2)
    maximum_score_row_file_bytes: int = Field(ge=1, le=65536)
    rows: tuple[ScientificAblationControlledEvaluationExpectationV1, ...] = Field(
        min_length=1, max_length=5000
    )

    @model_validator(mode="after")
    def exact_owned_inventory(self) -> Self:
        for manifest, domain in (
            (self.generation_plan, "iris/scientific-ablation-controlled-capture-plan/v1"),
            (self.capture, "iris/scientific-ablation-controlled-capture-complete/v1"),
        ):
            if manifest.sha256 != domain_sha256(
                domain, canonical_json_bytes(manifest.payload.model_dump(mode="json"))
            ):
                raise ValueError("Score plan generation/capture hash differs")
        p, c = self.generation_plan.payload, self.capture.payload
        repair = any(row.task_type == "split_spec_repair" for row in p.rows)
        if (
            c.plan_sha256 != self.generation_plan.sha256
            or c.input_sha256 != p.input.sha256
            or c.generation_control_plan_sha256 != p.input.payload.generation_control_plan_sha256
            or c.target_nll_semantics != p.input.payload.target_nll_semantics
            or c.example_count != len(p.rows)
            or len(self.rows) != len(p.rows)
            or tuple(row.row_index for row in self.rows) != tuple(range(len(p.rows)))
            or self.rows[-1].capture_ack_sha256 != c.acknowledgement_head_sha256
            or fsum(row.target_nll_sum for row in self.rows) != c.target_nll_sum
            or c.target_token_count != p.input.payload.target_token_count
            or repair != (self.python_policy is not None)
            or repair != (self.worker_sha256 is not None)
            or any(
                (
                    row.case_count < 4
                    if expected.task_type == "split_spec_repair"
                    else row.case_count != 0
                )
                for row, expected in zip(self.rows, p.rows, strict=True)
            )
        ):
            raise ValueError("Complete ordered score plan/policy/NLL inventory differs")
        return self


class IrisScientificAblationControlledEvaluationPlanV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledEvaluationPlanPayloadV1


class ScientificAblationControlledEvaluationRowPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-evaluation-row-v1"] = (
        "iris-scientific-ablation-controlled-evaluation-row-v1"
    )
    plan_sha256: Digest
    measurements: ScientificAblationControlledScoredRowV1


class IrisScientificAblationControlledEvaluationRowV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledEvaluationRowPayloadV1


class ScientificAblationControlledEvaluationAckPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-evaluation-ack-v1"] = (
        "iris-scientific-ablation-controlled-evaluation-ack-v1"
    )
    status: Literal["acknowledged_score"] = "acknowledged_score"
    plan_sha256: Digest
    row_index: int = Field(ge=0, le=4999)
    previous_ack_sha256: Digest
    row_sha256: Digest
    row_file_sha256: Digest
    row_file_bytes: int = Field(ge=1, le=65536)


class IrisScientificAblationControlledEvaluationAckV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledEvaluationAckPayloadV1


class ScientificAblationControlledEvaluationCompletePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-evaluation-complete-v1"] = (
        "iris-scientific-ablation-controlled-evaluation-complete-v1"
    )
    status: Literal["complete_not_scientific_launch_approval"] = (
        "complete_not_scientific_launch_approval"
    )
    plan_sha256: Digest
    capture_sha256: Digest
    acknowledgement_head_sha256: Digest
    evaluation_sha256: Digest
    example_count: int = Field(ge=1, le=5000)


class IrisScientificAblationControlledEvaluationCompleteV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledEvaluationCompletePayloadV1


__all__ = [
    "IrisScientificAblationControlledEvaluationAckV1",
    "IrisScientificAblationControlledEvaluationCompleteV1",
    "IrisScientificAblationControlledEvaluationPlanV1",
    "IrisScientificAblationControlledEvaluationRowV1",
    "ScientificAblationControlledEvaluationAckPayloadV1",
    "ScientificAblationControlledEvaluationCompletePayloadV1",
    "ScientificAblationControlledEvaluationExpectationV1",
    "ScientificAblationControlledEvaluationPlanPayloadV1",
    "ScientificAblationControlledEvaluationRowPayloadV1",
]
