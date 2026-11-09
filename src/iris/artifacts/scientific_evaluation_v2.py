"""Hash-only four-family quality report; no generated source or private labels."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v1 import PythonOutcome, PythonReason
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    ScientificFamilyQualityV1,
    ScientificQualityCountsV1,
    scientific_quality_counts_v1,
)
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256

ScientificScoreReasonV2 = (
    PythonReason
    | Literal[
        "semantic_match",
        "invalid_envelope",
        "invalid_content",
        "decision_mismatch",
        "next_action_mismatch",
        "content_mismatch",
    ]
)
_EARLY = {"invalid_envelope", "invalid_content", "decision_mismatch", "next_action_mismatch"}
_INVALID = {
    "invalid_envelope",
    "invalid_content",
    "invalid_source",
    "syntax_error",
    "missing_entrypoint",
}


class ScientificCodeEvidenceV2(_Strict):
    policy: PythonSandboxPolicyV2
    policy_sha256: Digest
    worker_sha256: Digest
    image_reference: Literal[
        "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
    ]
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_version: Literal["release-20260831.0"]
    docker_version: Literal["29.8.0"]
    kernel_release: str = Field(min_length=1, max_length=256)
    submitted_input_sha256: Digest
    stdout_sha256: Digest
    stdout_byte_length: int = Field(ge=0)
    stderr_sha256: Digest
    stderr_byte_length: int = Field(ge=0)
    transport_termination: Literal["exited", "timeout", "output_limit"]
    container_exit_code: int = Field(ge=0, le=255)
    oom_killed: bool

    @model_validator(mode="after")
    def check_policy(self) -> Self:
        if (
            self.policy.wall_time_ms != 5000
            or self.container_exit_code == 80
            or self.stdout_byte_length + self.stderr_byte_length > self.policy.max_output_bytes
            or self.policy_sha256
            != domain_sha256(
                "iris/python-sandbox-policy/v2",
                canonical_json_bytes(self.policy.model_dump(mode="json")),
            )
        ):
            raise ValueError("Scientific code evidence policy/bounds differ")
        return self


class ScientificAnswerMeasurementsV2(_Strict):
    """Actual answer measurements shared by control and intervened branches; no NLL."""

    example_id: str = Field(pattern=r"^test-[0-9]+$")
    generation_example_sha256: Digest
    scoring_example_sha256: Digest
    task_type: SyntheticTask
    output_sha256: Digest
    output_byte_length: int = Field(ge=0, le=4 * 1024**2)
    outcome: PythonOutcome
    reason: ScientificScoreReasonV2
    envelope_valid: bool
    structured_exact_match: bool
    tool_action_match: bool | None
    conflict_flag_match: bool | None
    case_count: int = Field(ge=0, le=128)
    passed_case_count: int = Field(ge=0, le=128)
    execution: ScientificCodeEvidenceV2 | None

    @model_validator(mode="after")
    def check_score(self) -> Self:
        if (
            self.envelope_valid != (self.reason != "invalid_envelope")
            or (self.structured_exact_match and not self.envelope_valid)
            or (self.tool_action_match is not None) != (self.task_type == "next_tool_prediction")
            or (self.conflict_flag_match is not None) != (self.task_type == "conflict_resolution")
            or (not self.envelope_valid and (self.tool_action_match or self.conflict_flag_match))
            or self.passed_case_count > self.case_count
            or (self.outcome == "invalid_output") != (self.reason in _INVALID)
        ):
            raise ValueError("Scored row shape/outcome contradicts its measurements")
        repair = self.task_type == "split_spec_repair"
        if repair:
            if (
                self.case_count < 4
                or (self.execution is None) != (self.reason in _EARLY)
                or (self.outcome == "correct") != (self.reason == "all_cases_pass")
                or (self.outcome == "correct") != (self.passed_case_count == self.case_count)
                or (
                    self.reason not in {"all_cases_pass", "case_mismatch"}
                    and self.passed_case_count
                )
                or self.reason in {"semantic_match", "content_mismatch"}
            ):
                raise ValueError("Repair scores require actual bound code evidence")
        elif (
            self.execution is not None
            or self.case_count
            or self.passed_case_count
            or self.reason not in _EARLY | {"semantic_match", "content_mismatch"}
            or (self.outcome == "correct") != (self.reason == "semantic_match")
        ):
            raise ValueError("Non-code score carries code evidence or inconsistent outcome")
        if self.outcome == "correct" and (
            self.tool_action_match is False or self.conflict_flag_match is False
        ):
            raise ValueError("Successful semantic score contradicts its family metric")
        if self.structured_exact_match and (
            self.reason in _EARLY or (not repair and self.outcome != "correct")
        ):
            raise ValueError("Exact structured answer contradicts its score")
        if self.execution is not None:
            evidence = self.execution
            resource_reason = (
                "memory_limit"
                if evidence.oom_killed
                else "output_limit"
                if evidence.transport_termination == "output_limit"
                else "wall_time_limit"
                if evidence.transport_termination == "timeout"
                else "execution_terminated"
                if evidence.container_exit_code != 0
                else None
            )
            if (resource_reason is not None and self.reason != resource_reason) or (
                resource_reason is None
                and self.reason
                in {"memory_limit", "output_limit", "wall_time_limit", "execution_terminated"}
            ):
                raise ValueError("Execution outcome contradicts resource-failure precedence")
        return self


class ScientificScoredRowV2(ScientificAnswerMeasurementsV2):
    target_token_count: int = Field(ge=1, le=65536)
    target_nll_sum: float = Field(ge=0.0)


class ScientificPythonLimitsV2(_Strict):
    """Literal source limits, not tunable values or proof of historical execution."""

    cpu_cores: int = Field(default=1, ge=1, le=1)
    memory_bytes: int = Field(default=512000000, ge=512000000, le=512000000)
    swap_bytes: int = Field(default=0, ge=0, le=0)
    process_limit: int = Field(default=32, ge=32, le=32)
    candidate_timeout_ms: int = Field(default=5000, ge=5000, le=5000)
    environment: Literal["empty"] = "empty"
    network: Literal["none"] = "none"
    root_filesystem: Literal["read_only"] = "read_only"
    host_mount_count: int = Field(default=0, ge=0, le=0)
    clock_scope: Literal["after_live_input_gate"] = "after_live_input_gate"


class ScientificEvaluationPayloadV2(_Strict):
    kind: Literal["iris-scientific-evaluation-v2"] = "iris-scientific-evaluation-v2"
    scoring_profile: Literal["iris-synthetic-01-scoring-v2"] = "iris-synthetic-01-scoring-v2"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"
    python_resource_scope: Literal["source_section15_v2", "not_applicable_no_code"]
    python_limits: ScientificPythonLimitsV2 | None
    nll_scope: Literal["copied_from_bound_generation_not_recomputed"] = (
        "copied_from_bound_generation_not_recomputed"
    )
    generation_sha256: Digest
    execution_sha256: Digest
    terminal_event_sha256: Digest
    best_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1)
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    python_policy: PythonSandboxPolicyV2 | None
    rows: tuple[ScientificScoredRowV2, ...] = Field(min_length=1, max_length=5000)
    overall: ScientificQualityCountsV1
    families: tuple[ScientificFamilyQualityV1, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def exact_aggregation(self) -> Self:
        if len({row.example_id for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate scored rows")
        repair = any(row.task_type == "split_spec_repair" for row in self.rows)
        if self.python_resource_scope != (
            "source_section15_v2" if repair else "not_applicable_no_code"
        ) or self.python_limits != (ScientificPythonLimitsV2() if repair else None):
            raise ValueError("Effective resource declaration differs from repair policy")
        if repair != (self.python_policy is not None) or (
            self.python_policy is not None and self.python_policy.wall_time_ms != 5000
        ):
            raise ValueError("Explicit five-second repair policy required, only for repair")
        if any(
            row.execution is not None and row.execution.policy != self.python_policy
            for row in self.rows
        ):
            raise ValueError("Code evidence uses another policy")
        if self.overall != scientific_quality_counts_v1(self.rows) or self.families != tuple(
            ScientificFamilyQualityV1(
                task_type=task,
                counts=scientific_quality_counts_v1(
                    tuple(row for row in self.rows if row.task_type == task)
                ),
            )
            for task in TASKS
        ):
            raise ValueError("Stored counts differ from exact row aggregation")
        if self.purpose == "scientific" and tuple(f.counts.examples for f in self.families) != (
            2000,
            1000,
            1000,
            1000,
        ):
            raise ValueError("Scientific report does not cover all 5000 held-out rows")
        return self


class IrisScientificEvaluationV2(_Strict):
    sha256: Digest
    payload: ScientificEvaluationPayloadV2


__all__ = [
    "IrisScientificEvaluationV2",
    "ScientificAnswerMeasurementsV2",
    "ScientificCodeEvidenceV2",
    "ScientificEvaluationPayloadV2",
    "ScientificPythonLimitsV2",
    "ScientificScoredRowV2",
]
