"""External-oracle contract; mocked observations never execute candidate source."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace
from hashlib import sha256
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from iris.application import study_code_evaluation as service
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyCodeEvaluationError
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import StudyPythonIntegerListRuleV1
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_comparison import check_python_observation_v1
from iris.integrations.python_sandbox import PythonSandboxError, PythonSandboxObservationV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_study_benchmark import _benchmark

_POLICY = PythonSandboxPolicyV1(
    profile="iris_wsl_gvisor_python312_restricted_v1", wall_time_ms=2000, max_output_bytes=1048576
)
_SECRET = "PRIVATE_CODE_EVALUATION_MARKER"


def _rule() -> StudyPythonIntegerListRuleV1:
    return cast(StudyPythonIntegerListRuleV1, _benchmark().payload.scoring[2].rule)


def _response(rule: StudyPythonIntegerListRuleV1) -> dict[str, Any]:
    return {
        "kind": "observations",
        "cases": [
            {"kind": "value", "result": case.expected, "input_after": list(case.inputs)}
            for case in rule.cases
        ],
    }


def _observation(raw: bytes, request: bytes = b"request") -> PythonSandboxObservationV1:
    return PythonSandboxObservationV1(
        policy=_POLICY,
        policy_sha256=domain_sha256(
            "iris/python-sandbox-policy/v1", canonical_json_bytes(_POLICY.model_dump(mode="json"))
        ),
        worker_sha256="a" * 64,
        image_id="sha256:" + "b" * 64,
        runtime_version="release-20260831.0",
        docker_version="29.8.0",
        kernel_release="test-kernel",
        submitted_input_sha256=sha256(request).hexdigest(),
        process=ProcessResult(0, raw, b"", "exited"),
        container_exit_code=0,
        oom_killed=False,
    )


def test_every_expected_answer_remains_outside_worker_request() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    rule = cast(StudyPythonIntegerListRuleV1, benchmark.payload.scoring[2].rule)

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        request = json.loads(request_bytes)
        assert set(request) == {"kind", "source", "inputs"}
        assert request["source"] == example.target
        assert request["inputs"] == [list(case.inputs) for case in rule.cases]
        assert policy == _POLICY and policy is not _POLICY
        return _observation(canonical_json_bytes(_response(rule)), request_bytes)

    with patch.object(service, "execute_python_job_v1", execute):
        result = service.score_study_python_answer_v1(
            benchmark, example_id=example.example_id, output_text=example.target, policy=_POLICY
        )
    assert result.answer.outcome == "correct" and result.reason == "all_cases_pass"
    assert result.case_count == result.passed_case_count == len(rule.cases)
    assert result.answer.output_sha256 == sha256(example.target.encode()).hexdigest()
    assert result.source_submitted is True
    assert not hasattr(result, "stdout") and not hasattr(result, "stderr")


@pytest.mark.parametrize("change", ["result", "bool", "mutated", "non_integer", "exception"])
def test_external_comparison_rejects_wrong_values_types_and_input_mutation(change: str) -> None:
    rule = _rule()
    response = _response(rule)
    if change == "result":
        response["cases"][0]["result"] += 1
    elif change == "bool":
        response["cases"][0]["result"] = True
    elif change == "mutated":
        response["cases"][0]["input_after"] = [*rule.cases[0].inputs, 999]
    else:
        response["cases"][0] = {"kind": change}
    outcome, reason, passed = service._compare_response(
        canonical_json_bytes(response), rule, source_submitted=True
    )
    assert outcome == "incorrect"
    assert reason == ("protocol_error" if change == "bool" else "case_mismatch")
    assert passed == (0 if change == "bool" else len(rule.cases) - 1)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"null",
        b"[]",
        b"true",
        b"\xff",
        b'{"passed":true}',
        b'{"kind":"observations","cases":[]}',
        b'{"kind":"syntax_error","kind":"syntax_error"}',
        b'{"kind":"syntax_error","extra":0}',
        b'{"kind":"syntax_error"} trailing',
        b'{"kind":"value","result":NaN}',
        b'{"kind":"value","result":1.5}',
        b'{"kind":"value","result":123456789012345678901}',
        b"[" * 2000 + b"]" * 2000,
    ],
    ids=[
        "empty",
        "null",
        "array",
        "bool",
        "utf8",
        "claimed-pass",
        "missing-cases",
        "duplicate",
        "extra",
        "trailing",
        "nan",
        "float",
        "huge-int",
        "depth",
    ],
)
def test_malformed_or_self_declared_protocol_is_not_a_correct_score(raw: bytes) -> None:
    assert service._compare_response(raw, _rule(), source_submitted=True) == (
        "incorrect",
        "protocol_error",
        0,
    )


@pytest.mark.parametrize(
    ("kind", "outcome"),
    [
        ("syntax_error", "invalid_output"),
        ("missing_entrypoint", "invalid_output"),
        ("load_exception", "incorrect"),
    ],
)
def test_structured_source_failures_are_distinguished(kind: str, outcome: str) -> None:
    assert service._compare_response(
        canonical_json_bytes({"kind": kind}), _rule(), source_submitted=True
    ) == (outcome, kind, 0)


@pytest.mark.parametrize(
    "output", ["", " \t\r\n", "a\0b", "x" * 1048577], ids=["empty", "space", "nul", "over-limit"]
)
def test_invalid_source_still_requires_runtime_but_never_sends_raw_source(output: str) -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        assert json.loads(request_bytes) == {"kind": "invalid_source_v1"}
        return _observation(b'{"kind":"invalid_source"}', request_bytes)

    with patch.object(service, "execute_python_job_v1", execute):
        result = service.score_study_python_answer_v1(
            benchmark, example_id=example.example_id, output_text=output, policy=_POLICY
        )
    assert result.answer.outcome == "invalid_output" and not result.source_submitted
    assert result.answer.output_sha256 == sha256(output.encode()).hexdigest()


@pytest.mark.parametrize(
    ("termination", "code", "oom", "reason"),
    [
        ("timeout", 137, False, "wall_time_limit"),
        ("output_limit", 137, False, "output_limit"),
        ("exited", 137, True, "memory_limit"),
        ("timeout", 137, True, "memory_limit"),
        ("exited", 7, False, "execution_terminated"),
    ],
)
def test_resource_or_exit_failure_cannot_accept_forged_success(
    termination: Any, code: int, oom: bool, reason: str
) -> None:
    rule = _rule()
    base = _observation(canonical_json_bytes(_response(rule)))
    observation = replace(
        base,
        process=replace(base.process, termination=termination),
        container_exit_code=code,
        oom_killed=oom,
    )
    assert service._assessment(observation, rule, True) == ("incorrect", reason, 0)


@pytest.mark.parametrize("output", ["", "def solve(values): return 0"])
def test_runtime_failure_never_becomes_a_model_zero(output: str) -> None:
    benchmark = _benchmark()
    with (
        patch.object(
            service, "execute_python_job_v1", new=AsyncMock(side_effect=PythonSandboxError(_SECRET))
        ),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        service.score_study_python_answer_v1(
            benchmark,
            example_id=benchmark.payload.scoring[2].example_id,
            output_text=output,
            policy=_POLICY,
        )
    assert _SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wall_time_ms", True),
        ("wall_time_ms", 0),
        ("wall_time_ms", 600001),
        ("max_output_bytes", 0),
        ("max_output_bytes", 8388609),
        ("profile", "runc"),
    ],
)
def test_policy_is_exact_bounded_and_has_no_fallback(field: str, value: Any) -> None:
    args = _POLICY.model_dump(mode="python")
    args[field] = value
    with pytest.raises(ValidationError):
        PythonSandboxPolicyV1.model_validate(args)


@pytest.mark.parametrize(
    "invalid", ["text-family", "unknown", "corrupt-benchmark", "utf8", "policy"]
)
def test_invalid_api_inputs_fail_before_any_job(invalid: str) -> None:
    benchmark = _benchmark()
    identifier = benchmark.payload.scoring[2].example_id
    output = _SECRET
    policy = _POLICY
    if invalid == "text-family":
        identifier = benchmark.payload.scoring[0].example_id
    elif invalid == "unknown":
        identifier = "unknown"
    elif invalid == "corrupt-benchmark":
        object.__setattr__(benchmark, "payload_sha256", "0" * 64)
    elif invalid == "utf8":
        output += "\ud800"
    else:
        policy = _POLICY.model_copy(update={"wall_time_ms": True})
    with patch.object(service, "execute_python_job_v1") as execute:
        with pytest.raises(IrisStudyCodeEvaluationError) as caught:
            service.score_study_python_answer_v1(
                benchmark, example_id=identifier, output_text=output, policy=policy
            )
        execute.assert_not_called()
    assert _SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_sha256", "0" * 64),
        ("submitted_input_sha256", "0" * 64),
        ("worker_sha256", "bad"),
        ("image_id", "image:latest"),
        ("runtime_version", "runc"),
        ("docker_version", "unknown"),
        ("container_exit_code", 80),
        ("container_exit_code", True),
        ("oom_killed", 0),
    ],
)
def test_inconsistent_backend_evidence_is_rejected_before_scoring(field: str, value: Any) -> None:
    observation = replace(_observation(b"{}"), **{field: value})
    with pytest.raises(PythonSandboxError):
        check_python_observation_v1(observation, b"request", _POLICY)


def test_public_api_exports_the_explicit_scorer_and_error() -> None:
    from iris import application

    assert application.score_study_python_answer_v1 is service.score_study_python_answer_v1
    assert application.IrisStudyCodeAnswerScoreV1 is service.IrisStudyCodeAnswerScoreV1
    assert application.IrisStudyCodeEvaluationError is IrisStudyCodeEvaluationError
