"""Hash-only four-family quality report; no generated source or private labels."""

from math import fsum, isfinite
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v1 import PythonOutcome, PythonReason, PythonSandboxPolicyV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256

ScientificScoreReasonV1 = (
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
TASKS: tuple[SyntheticTask, ...] = (
    "split_spec_repair",
    "delayed_goal",
    "conflict_resolution",
    "next_tool_prediction",
)


class ScientificCodeEvidenceV1(_Strict):
    policy: PythonSandboxPolicyV1
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
                "iris/python-sandbox-policy/v1",
                canonical_json_bytes(self.policy.model_dump(mode="json")),
            )
        ):
            raise ValueError("Scientific code evidence policy/bounds differ")
        return self


class ScientificScoredRowV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    generation_example_sha256: Digest
    scoring_example_sha256: Digest
    task_type: SyntheticTask
    output_sha256: Digest
    output_byte_length: int = Field(ge=0, le=4 * 1024**2)
    outcome: PythonOutcome
    reason: ScientificScoreReasonV1
    envelope_valid: bool
    structured_exact_match: bool
    tool_action_match: bool | None
    conflict_flag_match: bool | None
    case_count: int = Field(ge=0, le=128)
    passed_case_count: int = Field(ge=0, le=128)
    execution: ScientificCodeEvidenceV1 | None
    target_token_count: int = Field(ge=1, le=65536)
    target_nll_sum: float = Field(ge=0.0)

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


class ScientificQualityCountsV1(_Strict):
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
    target_tokens: int = Field(ge=0)
    target_nll_sum: float = Field(ge=0.0)

    @property
    def rates(self) -> dict[str, float | None]:
        def ratio(numerator: int | float, denominator: int) -> float | None:
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
            "target_nll_per_token": ratio(self.target_nll_sum, self.target_tokens),
        }


class _QualityRow(Protocol):
    """Version-neutral read-only measurements, not a widened v1 artifact schema."""

    @property
    def target_nll_sum(self) -> float: ...

    @property
    def outcome(self) -> PythonOutcome: ...

    @property
    def envelope_valid(self) -> bool: ...

    @property
    def structured_exact_match(self) -> bool: ...

    @property
    def task_type(self) -> SyntheticTask: ...

    @property
    def execution(self) -> object | None: ...

    @property
    def case_count(self) -> int: ...

    @property
    def passed_case_count(self) -> int: ...

    @property
    def tool_action_match(self) -> bool | None: ...

    @property
    def conflict_flag_match(self) -> bool | None: ...

    @property
    def target_token_count(self) -> int: ...


def scientific_quality_counts_v1(rows: tuple[_QualityRow, ...]) -> ScientificQualityCountsV1:
    try:
        total = fsum(row.target_nll_sum for row in rows)
    except OverflowError:
        raise ValueError("NLL aggregation overflowed") from None
    if not isfinite(total):
        raise ValueError("Nonfinite NLL aggregation")
    return ScientificQualityCountsV1(
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
        target_tokens=sum(row.target_token_count for row in rows),
        target_nll_sum=total,
    )


class ScientificFamilyQualityV1(_Strict):
    task_type: SyntheticTask
    counts: ScientificQualityCountsV1


class ScientificEvaluationPayloadV1(_Strict):
    kind: Literal["iris-scientific-evaluation-v1"] = "iris-scientific-evaluation-v1"
    scoring_profile: Literal["iris-synthetic-01-scoring-v1"] = "iris-synthetic-01-scoring-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"
    python_resource_scope: Literal["engineering_v1_not_final_source_limits"] = (
        "engineering_v1_not_final_source_limits"
    )
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
    python_policy: PythonSandboxPolicyV1 | None
    rows: tuple[ScientificScoredRowV1, ...] = Field(min_length=1, max_length=5000)
    overall: ScientificQualityCountsV1
    families: tuple[ScientificFamilyQualityV1, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def exact_aggregation(self) -> Self:
        if len({row.example_id for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate scored rows")
        repair = any(row.task_type == "split_spec_repair" for row in self.rows)
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


class IrisScientificEvaluationV1(_Strict):
    sha256: Digest
    payload: ScientificEvaluationPayloadV1


__all__ = [
    "IrisScientificEvaluationV1",
    "ScientificCodeEvidenceV1",
    "ScientificEvaluationPayloadV1",
    "ScientificFamilyQualityV1",
    "ScientificQualityCountsV1",
    "ScientificScoredRowV1",
    "scientific_quality_counts_v1",
]
