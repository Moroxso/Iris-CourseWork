"""Four-family scoring contracts; candidate source is never host-executed."""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iris.application import synthetic_scoring as service
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisSyntheticScoringError
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan, parse_structured_response
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_sandbox import PythonSandboxError, PythonSandboxObservationV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

_POLICY = PythonSandboxPolicyV1(
    profile="iris_wsl_gvisor_python312_restricted_v1", wall_time_ms=5000, max_output_bytes=1048576
)
_SECRET = "PRIVATE_SYNTHETIC_SCORING_MARKER"


def _rows() -> tuple[IrisExample, ...]:
    return tuple(
        generate_synthetic_split(SyntheticSplitPlan(split="validation", first_seed=1000, count=15))
    )


def _answer(row: IrisExample, **changes: Any) -> str:
    return json.dumps({**json.loads(row.expected_response), **changes})


def _observations(row: IrisExample) -> bytes:
    return canonical_json_bytes(
        {
            "kind": "observations",
            "cases": [
                {"kind": "value", "result": case.expected, "input_after": list(case.inputs)}
                for case in row.code_cases
            ],
        }
    )


def _observation(raw: bytes, request: bytes) -> PythonSandboxObservationV1:
    return PythonSandboxObservationV1(
        policy=_POLICY,
        policy_sha256=domain_sha256(
            "iris/python-sandbox-policy/v1", canonical_json_bytes(_POLICY.model_dump(mode="json"))
        ),
        worker_sha256="a" * 64,
        image_id="sha256:" + "b" * 64,
        runtime_version="release-20260831.0",
        docker_version="29.8.0",
        kernel_release="mock-kernel",
        submitted_input_sha256=sha256(request).hexdigest(),
        process=ProcessResult(0, raw, b"", "exited"),
        container_exit_code=0,
        oom_killed=False,
    )


@pytest.mark.parametrize("index", [2, 3, 4, 7, 8, 9, 12, 13, 14])
@pytest.mark.parametrize("confidence", [0, 0.37, 1])
def test_all_noncode_references_pass_without_exact_confidence(
    index: int, confidence: float
) -> None:
    row = _rows()[index]
    output = _answer(row, confidence=confidence)
    with patch.object(service, "execute_python_job_v1") as execute:
        score = service.score_synthetic_answer_v1(row, output_text=output, python_policy=None)
        execute.assert_not_called()
    assert score.outcome == "correct" and score.reason == "semantic_match"
    assert score.case_count == score.passed_case_count == 0 and score.execution is None
    assert score.task_type == row.task_type and score.split == row.split
    assert score.output_sha256 == sha256(output.encode()).hexdigest()
    assert score.output_byte_length == len(output.encode())
    assert score.example_sha256 == domain_sha256(
        "iris/synthetic-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
    )


@pytest.mark.parametrize("index", [3, 4])
def test_nested_json_key_order_and_whitespace_do_not_change_success(index: int) -> None:
    row = _rows()[index]
    content = json.loads(parse_structured_response(row.expected_response).content)
    output = _answer(row, content=json.dumps(dict(reversed(list(content.items()))), indent=3))
    score = service.score_synthetic_answer_v1(row, output_text="\n" + output, python_policy=None)
    assert score.outcome == "correct"


@pytest.mark.parametrize("extra,correct", [(" \t\r\n", True), ("\u00a0", False), ("x", False)])
def test_delayed_goal_only_trims_outer_ascii_whitespace(extra: str, correct: bool) -> None:
    row = _rows()[2]
    expected = parse_structured_response(row.expected_response).content
    score = service.score_synthetic_answer_v1(
        row, output_text=_answer(row, content=extra + expected + extra), python_policy=None
    )
    assert (score.outcome == "correct") is correct


@pytest.mark.parametrize("index", [0, 2, 3, 4])
@pytest.mark.parametrize(
    "invalid", ["fence", "duplicate", "nan", "extra", "bool", "limit", "depth"]
)
def test_invalid_envelopes_never_dispatch_code(index: int, invalid: str) -> None:
    row = _rows()[index]
    output = {
        "fence": "```json\n" + row.expected_response + "\n```",
        "duplicate": row.expected_response[:-1] + ',"confidence":0}',
        "nan": _answer(row, confidence=float("nan")),
        "extra": _answer(row, extra=True),
        "bool": _answer(row, confidence=True),
        "limit": "x" * 1048577,
        "depth": "[" * 2000 + "]" * 2000,
    }[invalid]
    with patch.object(service, "execute_python_job_v1") as execute:
        score = service.score_synthetic_answer_v1(
            row, output_text=output, python_policy=_POLICY if row.executable else None
        )
        execute.assert_not_called()
    assert (score.outcome, score.reason) == ("invalid_output", "invalid_envelope")
    assert score.execution is None


@pytest.mark.parametrize("index", [3, 4])
@pytest.mark.parametrize("invalid", ["duplicate", "extra", "type", "list", "depth", "trailing"])
def test_invalid_nested_json_is_not_a_semantic_match(index: int, invalid: str) -> None:
    row = _rows()[index]
    raw = parse_structured_response(row.expected_response).content
    content = json.loads(raw)
    first = next(iter(content))
    bad = {
        "duplicate": raw[:-1] + f',"{first}":' + json.dumps(content[first]) + "}",
        "extra": json.dumps({**content, "extra": True}),
        "type": json.dumps({**content, **({"conflict": 1} if index == 3 else {"target": 1})}),
        "list": "[]",
        "depth": "[" * 2000 + "]" * 2000,
        "trailing": raw + " trailing",
    }[invalid]
    score = service.score_synthetic_answer_v1(
        row, output_text=_answer(row, content=bad), python_policy=None
    )
    assert (score.outcome, score.reason) == ("invalid_output", "invalid_content")


@pytest.mark.parametrize(
    "field,value",
    [("conflict", False), ("source", "other"), ("status", "other"), ("reason", "majority")],
)
def test_wrong_conflict_values_are_incorrect(field: str, value: Any) -> None:
    row = _rows()[3]
    content = json.loads(parse_structured_response(row.expected_response).content)
    content[field] = value
    score = service.score_synthetic_answer_v1(
        row, output_text=_answer(row, content=json.dumps(content)), python_policy=None
    )
    assert (score.outcome, score.reason) == ("incorrect", "content_mismatch")


def test_tool_prediction_checks_action_consistency_without_running_tools() -> None:
    row = _rows()[4]
    reference = parse_structured_response(row.expected_response)
    wrong = "finish" if reference.next_action != "finish" else "read_file"
    content = json.loads(reference.content)
    with patch.object(service, "execute_python_job_v1") as execute:
        inconsistent = service.score_synthetic_answer_v1(
            row, output_text=_answer(row, next_action=wrong), python_policy=None
        )
        incorrect = service.score_synthetic_answer_v1(
            row,
            output_text=_answer(
                row, content=json.dumps({**content, "action": wrong}), next_action=wrong
            ),
            python_policy=None,
        )
        execute.assert_not_called()
    assert (inconsistent.outcome, inconsistent.reason) == ("invalid_output", "invalid_content")
    assert (incorrect.outcome, incorrect.reason) == ("incorrect", "next_action_mismatch")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"decision": "answer"}, "decision_mismatch"),
        ({"next_action": "finish"}, "next_action_mismatch"),
    ],
)
def test_repair_decision_and_action_must_match_before_dispatch(change: Any, reason: str) -> None:
    row = _rows()[0]
    with patch.object(service, "execute_python_job_v1") as execute:
        score = service.score_synthetic_answer_v1(
            row, output_text=_answer(row, **change), python_policy=_POLICY
        )
        execute.assert_not_called()
    assert score.outcome == "incorrect" and score.reason == reason and score.execution is None


@pytest.mark.parametrize("index", [0, 1, 5, 6, 10, 11])
def test_only_content_and_inputs_cross_isolation_even_for_reference_source(index: int) -> None:
    row = _rows()[index]
    source = parse_structured_response(row.expected_response).content

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        assert json.loads(request_bytes) == {
            "kind": "python_integer_list_v1",
            "source": source,
            "inputs": [list(case.inputs) for case in row.code_cases],
        }
        assert policy == _POLICY and policy is not _POLICY
        return _observation(_observations(row), request_bytes)

    with patch.object(service, "execute_python_job_v1", new=AsyncMock(side_effect=execute)) as run:
        score = service.score_synthetic_answer_v1(
            row, output_text=row.expected_response, python_policy=_POLICY
        )
        run.assert_awaited_once()
    assert score.outcome == "correct" and score.reason == "all_cases_pass"
    assert score.case_count == score.passed_case_count == len(row.code_cases)
    assert score.execution is not None
    assert score.execution.stdout_sha256 == sha256(_observations(row)).hexdigest()
    report = json.dumps(asdict(score), default=lambda value: value.model_dump(mode="json"))
    assert row.test_code is not None
    assert row.prompt not in report and source not in report and row.test_code not in report
    assert "stdout" not in asdict(score.execution) and "stderr" not in asdict(score.execution)


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("value", "case_mismatch"),
        ("mutation", "case_mismatch"),
        ("bool", "protocol_error"),
        ("forge", "protocol_error"),
    ],
)
def test_external_oracle_rejects_wrong_results_and_mutation(variant: str, reason: str) -> None:
    row = _rows()[0]
    response = json.loads(_observations(row))
    if variant == "value":
        response["cases"][0]["result"] += 1
    elif variant == "mutation":
        response["cases"][0]["input_after"].append(999)
    elif variant == "bool":
        response["cases"][0]["result"] = True
    else:
        response = {"passed": True}

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        return _observation(canonical_json_bytes(response), request_bytes)

    with patch.object(service, "execute_python_job_v1", execute):
        score = service.score_synthetic_answer_v1(
            row, output_text=row.expected_response, python_policy=_POLICY
        )
    assert score.outcome == "incorrect" and score.reason == reason


@pytest.mark.parametrize("failure", ["unavailable", "request", "runtime", "worker", "reserved"])
def test_infrastructure_failure_never_becomes_model_incorrect(failure: str) -> None:
    row = _rows()[0]

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        if failure == "unavailable":
            raise PythonSandboxError(_SECRET)
        changes: dict[str, dict[str, Any]] = {
            "request": {"submitted_input_sha256": "0" * 64},
            "runtime": {"runtime_version": "runc"},
            "worker": {"worker_sha256": "invalid"},
            "reserved": {"container_exit_code": 80},
        }
        return replace(_observation(_observations(row), request_bytes), **changes[failure])

    with (
        patch.object(service, "execute_python_job_v1", execute),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        service.score_synthetic_answer_v1(
            row, output_text=row.expected_response, python_policy=_POLICY
        )
    assert _SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "bad",
    [
        "no-policy",
        "budget",
        "corrupt-policy",
        "text-policy",
        "row",
        "reference",
        "reference-action",
        "utf8",
        "wrong-row-type",
    ],
)
def test_invalid_requests_fail_before_dispatch(bad: str) -> None:
    row: Any = _rows()[0]
    policy: PythonSandboxPolicyV1 | None = _POLICY
    output = _SECRET
    if bad == "no-policy":
        policy = None
    elif bad in ("budget", "corrupt-policy"):
        policy = _POLICY.model_copy(update={"wall_time_ms": 2000 if bad == "budget" else True})
    elif bad == "text-policy":
        row = _rows()[2]
    elif bad == "row":
        row.hidden_facts["operation"] = "tampered"
    elif bad == "reference":
        row = _rows()[3]
        row = row.model_copy(update={"expected_response": _answer(row, content="[]")})
        policy = None
    elif bad == "reference-action":
        row = row.model_copy(update={"expected_response": _answer(row, next_action="finish")})
    elif bad == "utf8":
        output += "\ud800"
    else:
        row = row.model_dump()
    with patch.object(service, "execute_python_job_v1") as execute:
        with pytest.raises(IrisSyntheticScoringError) as caught:
            service.score_synthetic_answer_v1(row, output_text=output, python_policy=policy)
        execute.assert_not_called()
    assert _SECRET not in "".join(traceback.format_exception(caught.value))


def test_active_event_loop_is_rejected_without_creating_worker_coroutine() -> None:
    row = _rows()[0]

    async def call() -> None:
        with patch.object(service, "execute_python_job_v1") as execute:
            with pytest.raises(IrisSyntheticScoringError):
                service.score_synthetic_answer_v1(
                    row, output_text=row.expected_response, python_policy=_POLICY
                )
            execute.assert_not_called()

    asyncio.run(call())


def test_public_api_exports() -> None:
    from iris import application

    assert application.score_synthetic_answer_v1 is service.score_synthetic_answer_v1
    assert application.IrisSyntheticAnswerScoreV1 is service.IrisSyntheticAnswerScoreV1
    assert application.IrisSyntheticCodeEvidenceV1 is service.IrisSyntheticCodeEvidenceV1
    assert application.IrisSyntheticScoringError is IrisSyntheticScoringError
