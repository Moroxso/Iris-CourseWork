"""Candidate four-family scoring; no legacy family aliases or launch approval.

Text answers trim only outer ASCII space/tab/CR/LF. Conflict/tool content is
strict typed JSON compared without key-order or whitespace sensitivity. Valid
confidence values are not correctness targets. Python content is assessed only
by the explicitly selected pinned sandbox, never by host eval/exec/test_code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisSyntheticScoringError
from iris.application.synthetic_scoring import SyntheticReason, _content
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.data.schemas import (
    IrisExample,
    SyntheticSplit,
    SyntheticTask,
    parse_structured_response,
)
from iris.integrations.python_comparison import PythonOutcome
from iris.integrations.python_comparison_v2 import (
    assess_integer_list_observation_v2,
    check_python_observation_v2,
)
from iris.integrations.python_sandbox import IMAGE, PythonSandboxError
from iris.integrations.python_sandbox_v2 import PythonSandboxObservationV2, execute_python_job_v2
from iris.utils.hashing import canonical_json_bytes, domain_sha256


@dataclass(frozen=True, slots=True)
class IrisSyntheticCodeEvidenceV2:
    """Hash-only observations, not execution attestation or a scientific receipt."""

    policy: PythonSandboxPolicyV2
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


@dataclass(frozen=True, slots=True)
class IrisSyntheticAnswerScoreV2:
    """Binds the complete private row and outer answer, not just a selected ID."""

    example_sha256: str
    example_id: str
    task_type: SyntheticTask
    split: SyntheticSplit
    output_sha256: str
    output_byte_length: int
    outcome: PythonOutcome
    reason: SyntheticReason
    case_count: int
    passed_case_count: int
    execution: IrisSyntheticCodeEvidenceV2 | None
    scoring_profile: Literal["iris-synthetic-01-scoring-v2"] = "iris-synthetic-01-scoring-v2"


def _evidence(observation: PythonSandboxObservationV2) -> IrisSyntheticCodeEvidenceV2:
    return IrisSyntheticCodeEvidenceV2(
        policy=observation.policy,
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


def score_synthetic_answer_v2(
    example: IrisExample,
    *,
    output_text: str,
    python_policy: PythonSandboxPolicyV2 | None,
) -> IrisSyntheticAnswerScoreV2:
    """Score one validated row under an explicit candidate evaluation profile.

    Repair requires an exact policy with the row's five-second wall budget. Other
    families require None. Invalid envelopes/decisions/actions do not dispatch code;
    valid repair content always needs a verified backend, even if it equals the
    reference source. Infrastructure failures raise, never count as model errors.
    The source-v2 profile enforces section15 limits. No worker observations are
    accepted from callers. The in-interpreter worker protocol is not tamperproof against a hostile
    candidate. Non-UTF8 or over-64-MiB API strings are request errors, not scores.
    """
    try:
        if type(example) is not IrisExample or type(output_text) is not str:
            raise TypeError("Exact row and string required")
        if len(output_text) > 64 * 1024 * 1024:
            raise ValueError("Answer exceeds API safety limit")
        encoded = output_text.encode("utf-8", errors="strict")
        if len(encoded) > 64 * 1024 * 1024:
            raise ValueError("Answer exceeds API safety limit")
        owned = IrisExample.model_validate(example.model_dump(mode="python"))
        policy: PythonSandboxPolicyV2 | None = None
        if owned.executable:
            if type(python_policy) is not PythonSandboxPolicyV2:
                raise TypeError("Repair requires an explicit exact sandbox policy")
            policy = PythonSandboxPolicyV2.model_validate(python_policy.model_dump(mode="python"))
            if policy.wall_time_ms != owned.timeout_seconds * 1000:
                raise ValueError("Sandbox budget must match the row timeout")
        elif python_policy is not None:
            raise ValueError("Non-executable tasks require no sandbox policy")
        expected = parse_structured_response(owned.expected_response)
        expected_content = _content(owned.task_type, expected)
        if owned.task_type != "next_tool_prediction" and expected.next_action != (
            "run_tests" if owned.executable else "finish"
        ):
            raise ValueError("Reference next action disagrees with the task")
        example_hash = domain_sha256(
            "iris/synthetic-example/v1", canonical_json_bytes(owned.model_dump(mode="json"))
        )

        def result(
            outcome: PythonOutcome,
            reason: SyntheticReason,
            passed: int = 0,
            execution: IrisSyntheticCodeEvidenceV2 | None = None,
        ) -> IrisSyntheticAnswerScoreV2:
            return IrisSyntheticAnswerScoreV2(
                example_sha256=example_hash,
                example_id=owned.id,
                task_type=owned.task_type,
                split=owned.split,
                output_sha256=sha256(encoded).hexdigest(),
                output_byte_length=len(encoded),
                outcome=outcome,
                reason=reason,
                case_count=len(owned.code_cases),
                passed_case_count=passed,
                execution=execution,
            )

        try:
            answer = parse_structured_response(output_text)
        except ValueError:
            return result("invalid_output", "invalid_envelope")
        if answer.decision != expected.decision:
            return result("incorrect", "decision_mismatch")
        try:
            actual_content = _content(owned.task_type, answer)
        except (ValueError, TypeError, RecursionError, OverflowError):
            return result("invalid_output", "invalid_content")
        if answer.next_action != expected.next_action:
            return result("incorrect", "next_action_mismatch")
        if not owned.executable:
            return (
                result("correct", "semantic_match")
                if actual_content == expected_content
                else result("incorrect", "content_mismatch")
            )
        assert policy is not None
        # Only source and concrete inputs cross isolation. No prompt, facts,
        # reference source, expected results or stored test_code go to the worker.
        request = canonical_json_bytes(
            {
                "kind": "python_integer_list_v1",
                "source": answer.content,
                "inputs": [list(case.inputs) for case in owned.code_cases],
            }
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ValueError("Synchronous code scoring requires no running event loop")
        observation = asyncio.run(execute_python_job_v2(request_bytes=request, policy=policy))
        check_python_observation_v2(observation, request, policy)
        outcome, reason, passed = assess_integer_list_observation_v2(
            observation, owned.code_cases, source_submitted=True
        )
        return result(outcome, reason, passed, _evidence(observation))
    except PythonSandboxError:
        raise IrisCodeEvaluationUnavailableError(
            "The explicit Python sandbox is unavailable, unverified, busy, or not clean"
        ) from None
    except Exception:
        raise IrisSyntheticScoringError("Iris structured answer could not be scored") from None


__all__ = [
    "IrisSyntheticAnswerScoreV2",
    "IrisSyntheticCodeEvidenceV2",
    "score_synthetic_answer_v2",
]
