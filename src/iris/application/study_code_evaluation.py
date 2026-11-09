"""External functional comparisons after explicitly isolated Python execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyCodeEvaluationError
from iris.application.study_benchmark import IrisStudyAnswerScoreV1, validate_study_benchmark_v1
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1, StudyPythonIntegerListRuleV1
from iris.integrations.python_comparison import (
    PythonOutcome as _Outcome,
)
from iris.integrations.python_comparison import (
    PythonReason as _Reason,
)
from iris.integrations.python_comparison import (
    assess_integer_list_observation_v1,
    compare_integer_list_response_v1,
)
from iris.integrations.python_comparison import (
    check_python_observation_v1 as _check_observation,
)
from iris.integrations.python_sandbox import (
    IMAGE,
    PythonSandboxError,
    PythonSandboxObservationV1,
    execute_python_job_v1,
)
from iris.utils.hashing import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class IrisStudyCodeAnswerScoreV1:
    """Hash-only standalone evidence, NOT a request receipt or execution attestation.

    Runtime settings belong to the immutable named policy profile. Case outputs
    and diagnostics are not retained here. Request-bound receipts separately
    bind the real request/checkpoint and complete-input preflight.
    """

    answer: IrisStudyAnswerScoreV1
    reason: _Reason
    case_count: int
    passed_case_count: int
    source_submitted: bool
    policy: PythonSandboxPolicyV1
    policy_sha256: str
    worker_sha256: str
    image_reference: str
    image_id: str
    runtime_version: str
    docker_version: str
    kernel_release: str
    submitted_input_sha256: str
    stdout_sha256: str
    stdout_byte_length: int
    stderr_sha256: str
    stderr_byte_length: int
    transport_termination: Literal["exited", "timeout", "output_limit"]
    container_exit_code: int
    oom_killed: bool


def _compare_response(
    raw: bytes, rule: StudyPythonIntegerListRuleV1, *, source_submitted: bool
) -> tuple[_Outcome, _Reason, int]:
    return compare_integer_list_response_v1(raw, rule.cases, source_submitted=source_submitted)


def _assessment(
    observation: PythonSandboxObservationV1, rule: StudyPythonIntegerListRuleV1, submitted: bool
) -> tuple[_Outcome, _Reason, int]:
    return assess_integer_list_observation_v1(observation, rule.cases, source_submitted=submitted)


def score_study_python_answer_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    example_id: str,
    output_text: str,
    policy: PythonSandboxPolicyV1,
) -> IrisStudyCodeAnswerScoreV1:
    """Execute raw Python only inside the verified pinned gVisor worker.

    Correctness and unchanged inputs are compared here, outside the container.
    This is observable behavior, not tamperproof function-return attestation
    against a malicious program sharing the worker interpreter. Explicit policy
    and a free verified runtime are required, even for invalid raw source.
    """
    score, _observation = _score_study_python_answer_v1(
        benchmark, example_id=example_id, output_text=output_text, policy=policy, observed=None
    )
    return score


def _score_study_python_answer_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    example_id: str,
    output_text: str,
    policy: PythonSandboxPolicyV1,
    observed: PythonSandboxObservationV1 | None,
) -> tuple[IrisStudyCodeAnswerScoreV1, PythonSandboxObservationV1]:
    """Capture once, or recompute from supplied private observations without execution.

    The replay branch checks consistency, never historical execution authenticity.
    It shares the exact preparation, external comparisons and failure precedence.
    """
    try:
        if type(example_id) is not str or type(output_text) is not str:
            raise TypeError("Code scoring requires exact strings")
        if len(output_text) > 64 * 1024 * 1024:
            raise ValueError("Code scoring API input exceeds the safety limit")
        encoded = output_text.encode("utf-8", errors="strict")
        if len(encoded) > 64 * 1024 * 1024:
            raise ValueError("Code scoring API input exceeds the safety limit")
        if type(policy) is not PythonSandboxPolicyV1:
            raise TypeError("Code scoring requires an explicit exact policy")
        owned_policy = PythonSandboxPolicyV1.model_validate(policy.model_dump(mode="python"))
        owned = validate_study_benchmark_v1(benchmark)
        dataset = owned.payload.dataset.payload
        index = next(
            i
            for i, item in enumerate(dataset.manifest.payload.examples)
            if item.example_id == example_id
        )
        example = dataset.manifest.payload.examples[index]
        rule = owned.payload.scoring[index].rule
        if type(rule) is not StudyPythonIntegerListRuleV1:
            raise ValueError("The selected task is not a functional Python task")
        submitted = (
            bool(output_text.strip(" \t\r\n"))
            and "\0" not in output_text
            and len(encoded) <= 1048576
        )
        # This is the only candidate data passed to the worker: no targets,
        # expected answers, benchmark file paths, credentials or host mounts.
        request = (
            {
                "kind": "python_integer_list_v1",
                "source": output_text,
                "inputs": [list(case.inputs) for case in rule.cases],
            }
            if submitted
            else {"kind": "invalid_source_v1"}
        )
        request_bytes = canonical_json_bytes(request)
        if observed is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise ValueError("The synchronous scoring API requires no running event loop")
            observation = asyncio.run(
                execute_python_job_v1(request_bytes=request_bytes, policy=owned_policy)
            )
        else:
            observation = observed
        _check_observation(observation, request_bytes, owned_policy)
        outcome, reason, passed = _assessment(observation, rule, submitted)
        score = IrisStudyCodeAnswerScoreV1(
            answer=IrisStudyAnswerScoreV1(
                benchmark_payload_sha256=owned.payload_sha256,
                example_id=example_id,
                task_family=dataset.assignments[index].task_family,
                split=example.split,
                output_sha256=sha256(encoded).hexdigest(),
                output_byte_length=len(encoded),
                outcome=outcome,
            ),
            reason=reason,
            case_count=len(rule.cases),
            passed_case_count=passed,
            source_submitted=submitted,
            policy=owned_policy,
            policy_sha256=observation.policy_sha256,
            worker_sha256=observation.worker_sha256,
            image_reference=IMAGE,
            image_id=observation.image_id,
            runtime_version=observation.runtime_version,
            docker_version=observation.docker_version,
            kernel_release=observation.kernel_release,
            submitted_input_sha256=observation.submitted_input_sha256,
            stdout_sha256=sha256(observation.process.stdout).hexdigest(),
            stdout_byte_length=len(observation.process.stdout),
            stderr_sha256=sha256(observation.process.stderr).hexdigest(),
            stderr_byte_length=len(observation.process.stderr),
            transport_termination=observation.process.termination,
            container_exit_code=observation.container_exit_code,
            oom_killed=observation.oom_killed,
        )
        return score, observation
    except PythonSandboxError:
        raise IrisCodeEvaluationUnavailableError(
            "The explicit Python sandbox is unavailable, unverified, busy, or not clean"
        ) from None
    except Exception:
        raise IrisStudyCodeEvaluationError(
            "Iris functional code answer could not be scored"
        ) from None


__all__ = ["IrisStudyCodeAnswerScoreV1", "score_study_python_answer_v1"]
