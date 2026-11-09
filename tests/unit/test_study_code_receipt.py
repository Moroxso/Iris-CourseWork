"""Request/code consistency tests with synthetic observations, never host candidate execution."""

import asyncio
import base64
import traceback
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from iris.application import study_code_evaluation as scorer
from iris.application import study_code_receipt as service
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.request_artifact_v4 import read_request_artifact_v4, write_request_artifact_v4
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1, StudyPythonIntegerListRuleV1
from iris.artifacts.study_code_evaluation_v1 import IrisStudyCodeEvaluationV1, StudyCodeStreamV1
from iris.integrations.python_sandbox import PythonSandboxError, PythonSandboxObservationV1, _worker
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_request_artifact_v4 import build_artifact_v4
from tests.unit.test_study_code_evaluation import _POLICY, _observation, _response
from tests.unit.test_study_evaluation import _request
from tests.unit.test_study_input_preflight import _assets, _tokenizer

type _Fixture = tuple[IrisStudyBenchmarkV1, Path, str, str]


def _fixture(tmp_path: Path, *, output: str | None = None, **kwargs: Any) -> _Fixture:
    if output is None:
        benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
        output = benchmark.payload.dataset.payload.manifest.payload.examples[2].target
    return _request(tmp_path, output=output, example_index=2, **kwargs)


def _evaluate(
    fixture: _Fixture, *, raw: bytes | None = None, **changes: Any
) -> IrisStudyCodeEvaluationV1:
    benchmark, path, identity, output = fixture
    rule = cast(StudyPythonIntegerListRuleV1, benchmark.payload.scoring[2].rule)
    response = canonical_json_bytes(_response(rule)) if raw is None else raw

    async def capture(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        assert policy == _POLICY
        return replace(
            _observation(response, request_bytes),
            worker_sha256=sha256(_worker()).hexdigest(),
            **changes,
        )

    with _assets(_tokenizer()), patch.object(scorer, "execute_python_job_v1", capture):
        return service.evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=output,
            terminal_token_id=1,
            policy=_POLICY,
        )


def _verify(evaluation: IrisStudyCodeEvaluationV1, fixture: _Fixture) -> IrisStudyCodeEvaluationV1:
    with _assets(_tokenizer()), patch.object(scorer, "execute_python_job_v1") as forbidden:
        result = service.verify_study_code_evaluation_v1(
            evaluation, benchmark=fixture[0], request_path=fixture[1], output_text=fixture[3]
        )
    forbidden.assert_not_called()
    return result


def _rehash(raw: dict[str, Any]) -> IrisStudyCodeEvaluationV1:
    """Simulate a consistent attacker-controlled file; hashing is not authentication."""
    raw["payload"]["receipt"]["observation_sha256"] = domain_sha256(
        service._OBSERVATION_DOMAIN, canonical_json_bytes(raw["payload"]["observation"])
    )
    raw["payload_sha256"] = domain_sha256(
        service._PAYLOAD_DOMAIN, canonical_json_bytes(raw["payload"])
    )
    return IrisStudyCodeEvaluationV1.model_validate(raw)


def test_hash_only_receipt_and_private_capture_bind_exact_request_and_full_corpus(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evaluation = _evaluate(fixture)
    receipt = evaluation.payload.receipt
    assert receipt.outcome == "correct" and receipt.passed_case_count == receipt.case_count
    assert receipt.arm_id == "A" and receipt.checkpoint_transfer is None
    assert receipt.request_payload_sha256 == fixture[1].stem
    assert receipt.benchmark_payload_sha256 == fixture[0].payload_sha256
    assert receipt.output_sha256 == sha256(fixture[3].encode()).hexdigest()
    assert receipt.output_byte_length == len(fixture[3].encode())
    assert receipt.input_binding.corpus_examples == 15
    assert receipt.scientific_launch_status == "not_assessed"
    assert receipt.verification_scope.endswith("not_execution_attestation")
    assert fixture[3].encode() not in canonical_json_bytes(receipt.model_dump(mode="json"))
    assert _verify(evaluation, fixture) == evaluation
    assert _verify(evaluation, fixture) is not evaluation
    assert evaluation.payload.observation.stdout.byte_length > 0


@pytest.mark.parametrize(
    "raw,outcome,reason",
    [
        (b'{"kind":"syntax_error"}', "invalid_output", "syntax_error"),
        (b'{"kind":"missing_entrypoint"}', "invalid_output", "missing_entrypoint"),
        (b'{"kind":"load_exception"}', "incorrect", "load_exception"),
        (b"PRIVATE_BROKEN_PROTOCOL\xff", "incorrect", "protocol_error"),
    ],
)
def test_recorded_failures_recompute_without_runtime(
    tmp_path: Path, raw: bytes, outcome: str, reason: str
) -> None:
    fixture = _fixture(tmp_path)
    result = _evaluate(fixture, raw=raw)
    assert result.payload.receipt.outcome == outcome
    assert result.payload.receipt.reason == reason
    assert _verify(result, fixture) == result
    assert base64.b64decode(result.payload.observation.stdout.data_base64) == raw


def test_invalid_source_still_captures_nonexecuting_worker_control(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, output=" \r\n")
    result = _evaluate(fixture, raw=b'{"kind":"invalid_source"}')
    assert not result.payload.receipt.source_submitted
    assert result.payload.receipt.reason == "invalid_source"
    assert _verify(result, fixture) == result


@pytest.mark.parametrize(
    "termination,oom,reason",
    [
        ("timeout", False, "wall_time_limit"),
        ("output_limit", False, "output_limit"),
        ("timeout", True, "memory_limit"),
        ("exited", False, "execution_terminated"),
    ],
)
def test_resource_and_exit_failures_preserve_external_precedence(
    tmp_path: Path, termination: str, oom: bool, reason: str
) -> None:
    fixture = _fixture(tmp_path)
    process = replace(_observation(b"").process, termination=termination)  # type: ignore[arg-type]
    result = _evaluate(fixture, process=process, container_exit_code=137, oom_killed=oom)
    assert result.payload.receipt.reason == reason and result.payload.receipt.passed_case_count == 0
    assert _verify(result, fixture) == result


@pytest.mark.parametrize("change", ["same_length", "truncated", "masked"])
def test_prompt_tokens_and_mask_must_match_before_capture(tmp_path: Path, change: str) -> None:
    fixture = _fixture(tmp_path, token_change=change)
    with (
        patch.object(service, "_score_study_python_answer_v1") as capture,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)
    capture.assert_not_called()


def test_full_corpus_budget_failure_prevents_capture(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, max_new_tokens=2)
    with (
        patch.object(service, "_score_study_python_answer_v1") as capture,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)
    capture.assert_not_called()


@pytest.mark.parametrize("change", ["output", "example", "text_family", "policy"])
def test_private_binding_and_api_failure_precede_tokenizer_and_capture(
    tmp_path: Path, change: str
) -> None:
    benchmark, path, identity, output = _fixture(tmp_path)
    policy = _POLICY
    if change == "output":
        output += "PRIVATE_SOURCE_MISMATCH"
    elif change == "example":
        identity = benchmark.payload.dataset.payload.manifest.payload.examples[7].example_id
    elif change == "text_family":
        identity = benchmark.payload.dataset.payload.manifest.payload.examples[0].example_id
    else:
        policy = _POLICY.model_copy(update={"wall_time_ms": True})
    with (
        patch.object(service, "preflight_study_inputs_v1") as preflight,
        patch.object(scorer, "execute_python_job_v1") as executor,
        pytest.raises(IrisStudyEvaluationError) as caught,
    ):
        service.evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=output,
            terminal_token_id=1,
            policy=policy,
        )
    assert "PRIVATE_SOURCE_MISMATCH" not in "".join(traceback.format_exception(caught.value))
    preflight.assert_not_called()
    executor.assert_not_called()


@pytest.mark.parametrize("arm", ["B", "C", "D"])
def test_missing_checkpoint_fails_before_runtime(tmp_path: Path, arm: str) -> None:
    benchmark, _, identity, output = _fixture(tmp_path / "source")
    path = write_request_artifact_v4(build_artifact_v4(tmp_path / "other", arm)).path
    with (
        patch.object(scorer, "execute_python_job_v1") as executor,
        pytest.raises(IrisStudyEvaluationError),
    ):
        service.evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=output,
            terminal_token_id=1,
            policy=_POLICY,
        )
    executor.assert_not_called()


def test_request_is_reverified_after_actual_capture_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = scorer._score_study_python_answer_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        fixture[1].write_bytes(b"CHANGED_PRIVATE_REQUEST")
        return result

    with (
        patch.object(service, "_score_study_python_answer_v1", changed),
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)


def test_runtime_failure_remains_unavailable_not_model_zero(tmp_path: Path) -> None:
    benchmark, path, identity, output = _fixture(tmp_path)
    with (
        _assets(_tokenizer()),
        patch.object(
            scorer,
            "execute_python_job_v1",
            AsyncMock(side_effect=PythonSandboxError("PRIVATE_RUNTIME")),
        ),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        service.evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=output,
            terminal_token_id=1,
            policy=_POLICY,
        )
    assert "PRIVATE_RUNTIME" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "field",
    [
        "benchmark_payload_sha256",
        "dataset_payload_sha256",
        "manifest_payload_sha256",
        "protocol_declaration_payload_sha256",
        "request_payload_sha256",
        "configuration_sha256",
        "execution_profile_set_sha256",
        "scoring_rule_sha256",
        "output_sha256",
        "outcome",
        "split",
        "case_count",
        "source_submitted",
        "preflight_sha256",
        "tokenizer_semantics_sha256",
    ],
)
def test_recomputed_hashes_do_not_hide_forged_receipt(tmp_path: Path, field: str) -> None:
    fixture = _fixture(tmp_path)
    raw = _evaluate(fixture).model_dump(mode="python")
    receipt = raw["payload"]["receipt"]
    if field in ("preflight_sha256", "tokenizer_semantics_sha256"):
        receipt["input_binding"][field] = "0" * 64
    elif field == "outcome":
        receipt.update(outcome="incorrect", reason="case_mismatch", passed_case_count=0)
    elif field == "case_count":
        receipt["case_count"] += 1
        receipt["passed_case_count"] += 1
    elif field == "split":
        receipt[field] = "test"
    elif field == "source_submitted":
        receipt[field] = False
    else:
        receipt[field] = "0" * 64
    forged = _rehash(raw)
    assert service.validate_study_code_evaluation_v1(forged) == forged
    with pytest.raises(IrisStudyEvaluationError):
        _verify(forged, fixture)


@pytest.mark.parametrize(
    "field", ["submitted_input_sha256", "worker_sha256", "transport_returncode", "stdout"]
)
def test_altered_observations_are_checked_against_source_worker_and_score(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    raw = _evaluate(fixture).model_dump(mode="python")
    observation = raw["payload"]["observation"]
    if field == "stdout":
        observation[field] = service._stream(b'{"kind":"syntax_error"}').model_dump(mode="python")
    elif field == "transport_returncode":
        observation[field] = 1
    else:
        observation[field] = "0" * 64
    forged = _rehash(raw)
    with pytest.raises(IrisStudyEvaluationError):
        _verify(forged, fixture)


def test_worker_data_change_during_capture_prevents_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with (
        patch.object(service, "_worker", side_effect=[_worker(), b"changed worker data"]),
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)


def test_replay_works_in_active_async_loop_without_spawning_processes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evaluation = _evaluate(fixture)

    async def replay() -> IrisStudyCodeEvaluationV1:
        return _verify(evaluation, fixture)

    assert asyncio.run(replay()) == evaluation


def test_fabricated_consistent_observations_are_not_historical_execution_attestation(
    tmp_path: Path,
) -> None:
    # This wrong source is deliberately NOT executed. Synthetic matching observations
    # can be internally consistent: the scope must never promise historical truth.
    fixture = _fixture(tmp_path, output="def solve(values): return 0\n")
    evaluation = _evaluate(fixture)
    assert _verify(evaluation, fixture).payload.receipt.outcome == "correct"
    assert "not_execution_attestation" in evaluation.payload.receipt.verification_scope


@pytest.mark.parametrize(
    "change", ["space", "padding", "alphabet", "noncanonical_bits", "length", "hash", "bool"]
)
def test_private_streams_require_canonical_bounded_base64_and_exact_lengths(change: str) -> None:
    raw = service._stream(b"a").model_dump(mode="python")
    if change == "space":
        raw["data_base64"] = "Y Q=="
    elif change == "padding":
        raw["data_base64"] = "YQ"
    elif change == "alphabet":
        raw["data_base64"] = "!!!!"
    elif change == "noncanonical_bits":
        raw["data_base64"] = "YR=="
    elif change == "length":
        raw["byte_length"] = 2
    elif change == "hash":
        raw["sha256"] = "0" * 64
    else:
        raw["byte_length"] = True
    with pytest.raises(ValidationError):
        StudyCodeStreamV1.model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("container_exit_code", 80),
        ("container_exit_code", True),
        ("oom_killed", 0),
        ("transport_returncode", False),
        ("transport_termination", "unknown"),
        ("runtime_version", "unverified"),
        ("docker_version", "unverified"),
        ("image_reference", "python:latest"),
    ],
)
def test_observation_schema_rejects_unverified_runtime_and_coercion(
    tmp_path: Path, field: str, value: Any
) -> None:
    raw = _evaluate(_fixture(tmp_path)).model_dump(mode="python")
    raw["payload"]["observation"][field] = value
    with pytest.raises(ValidationError):
        IrisStudyCodeEvaluationV1.model_validate(raw)


def test_combined_stream_cap_and_policy_hash_are_not_optional(tmp_path: Path) -> None:
    raw = _evaluate(_fixture(tmp_path)).model_dump(mode="python")
    raw["payload"]["observation"]["policy"]["max_output_bytes"] = 1024
    raw["payload"]["observation"]["stderr"] = service._stream(b"x" * 1024).model_dump(mode="python")
    with pytest.raises(ValidationError):
        _rehash(raw)
    raw["payload"]["observation"]["stderr"] = service._stream(b"").model_dump(mode="python")
    raw["payload"]["observation"]["stdout"] = service._stream(b"").model_dump(mode="python")
    with pytest.raises(IrisStudyEvaluationError):
        service.validate_study_code_evaluation_v1(_rehash(raw))


def test_current_request_configuration_is_used_for_entire_preflight(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = preflight_study_inputs_v1
    with patch.object(service, "preflight_study_inputs_v1", wraps=original) as preflight:
        _evaluate(fixture)
    assert (
        preflight.call_args.kwargs["config"]
        == read_request_artifact_v4(fixture[1]).payload.configuration.value
    )


@pytest.mark.parametrize(
    "field",
    [
        "configuration_sha256",
        "execution_profile_set_sha256",
        "transformers_version",
        "tokenizers_version",
        "other_example_budget",
    ],
)
def test_preflight_mismatches_anywhere_in_corpus_prevent_execution(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    with _assets(_tokenizer()):
        report = preflight_study_inputs_v1(
            fixture[0],
            config=read_request_artifact_v4(fixture[1]).payload.configuration.value,
            terminal_token_id=1,
        )
    assert not report.examples[2].violations
    if field == "other_example_budget":
        examples = list(report.examples)
        examples[-1] = replace(examples[-1], violations=("target_over_limit",))
        changed = replace(report, examples=tuple(examples))
    else:
        changes: dict[str, Any] = {field: "0" * 64}
        changed = replace(report, **changes)
    with (
        patch.object(service, "preflight_study_inputs_v1", return_value=changed),
        patch.object(service, "_score_study_python_answer_v1") as capture,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)
    capture.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_family", "split_information"),
        ("arm_id", "B"),
        ("arm_id", "C"),
        ("arm_id", "D"),
        ("case_count", True),
        ("passed_case_count", True),
        ("source_submitted", 1),
        ("outcome", "incorrect"),
        ("passed_case_count", 0),
        ("reason", "syntax_error"),
        ("scientific_launch_status", "approved"),
        ("verification_scope", "execution_attestation"),
    ],
)
def test_receipt_schema_rejects_impossible_score_or_scope(
    tmp_path: Path, field: str, value: Any
) -> None:
    raw = _evaluate(_fixture(tmp_path)).model_dump(mode="python")
    raw["payload"]["receipt"][field] = value
    with pytest.raises(ValidationError):
        IrisStudyCodeEvaluationV1.model_validate(raw)


def test_exact_total_capture_budget_can_roundtrip_without_truncating_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    policy = _POLICY.model_copy(update={"max_output_bytes": 8 * 1024 * 1024})
    raw = _evaluate(fixture).model_dump(mode="python")
    observed = raw["payload"]["observation"]
    observed["policy"] = policy.model_dump(mode="python")
    observed["policy_sha256"] = domain_sha256(
        "iris/python-sandbox-policy/v1", canonical_json_bytes(policy.model_dump(mode="json"))
    )
    observed["stderr"] = service._stream(
        b"x" * (policy.max_output_bytes - observed["stdout"]["byte_length"])
    ).model_dump(mode="python")
    evaluation = _rehash(raw)
    assert service.validate_study_code_evaluation_v1(evaluation) == evaluation
    assert _verify(evaluation, fixture) == evaluation


@pytest.mark.parametrize("field", ["payload_sha256", "observation_sha256", "size", "type"])
def test_static_validation_does_not_trust_constructed_instances(tmp_path: Path, field: str) -> None:
    evaluation = _evaluate(_fixture(tmp_path))
    if field == "payload_sha256":
        evaluation = evaluation.model_copy(update={field: "0" * 64})
    elif field == "observation_sha256":
        receipt = evaluation.payload.receipt.model_copy(update={field: "0" * 64})
        payload = evaluation.payload.model_copy(update={"receipt": receipt})
        evaluation = evaluation.model_copy(update={"payload": payload})
    elif field == "type":
        evaluation = cast(IrisStudyCodeEvaluationV1, None)
    else:
        with (
            patch.object(service, "_MAX_CODE_EVALUATION_BYTES", 1),
            pytest.raises(IrisStudyEvaluationError),
        ):
            service.validate_study_code_evaluation_v1(evaluation)
        return
    with pytest.raises(IrisStudyEvaluationError):
        service.validate_study_code_evaluation_v1(evaluation)
