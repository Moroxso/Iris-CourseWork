"""Request-bound text scores, real A request verification, and explicit non-attestation."""

import traceback
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from iris.application import study_evaluation as service
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    read_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.study_evaluation import (
    evaluate_study_request_answer_v1,
    validate_study_evaluation_v1,
    verify_study_evaluation_v1,
)
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.verification_v4 import verify_request_artifact_v4
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_evaluation_v1 import IrisStudyEvaluationV1
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256
from tests.unit.test_request_artifact_v3 import artifact_inputs
from tests.unit.test_request_artifact_v4 import build_artifact_v4
from tests.unit.test_study_input_preflight import _assets, _tokenizer


def _request(
    tmp_path: Path,
    *,
    output: str | None = None,
    example_index: int = 0,
    token_change: str | None = None,
    max_new_tokens: int = 128,
) -> tuple[IrisStudyBenchmarkV1, Path, str, str]:
    """Build a labelled synthetic execution record; this fixture is not model inference."""
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    example = benchmark.payload.dataset.payload.manifest.payload.examples[example_index]
    answer = "\t" + example.target + " \r\n" if output is None else output
    runtime, execution, kwargs = artifact_inputs(tmp_path, "A")
    config = IrisBootstrapConfig.model_validate_json(runtime.configuration_json)
    config.backbone.limits.max_new_tokens = max_new_tokens
    configuration = canonical_json_bytes(config.model_dump(mode="json"))
    runtime = replace(
        runtime,
        configuration_json=configuration.decode(),
        configuration_sha256=domain_sha256("iris/configuration/v1", configuration),
    )
    tokenizer = _tokenizer()
    ids = torch.tensor([tokenizer(example.prompt)["input_ids"]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    if token_change == "same_length":
        ids[0, -1] = 2
    elif token_change == "truncated":
        ids, mask = ids[:, -2:].clone(), mask[:, -2:].clone()
    elif token_change == "masked":
        mask[0, -1] = 0
    assert execution.receipt is not None
    execution = replace(
        execution,
        generation=replace(execution.generation, text=answer),
        prompt_input_ids=ids,
        prompt_attention_mask=mask,
        receipt=replace(
            execution.receipt,
            prompt_sha256=domain_sha256(
                "iris/request-execution/v1/prompt", example.prompt.encode()
            ),
        ),
    )
    kwargs.update(prompt=example.prompt, execution=execution, runtime=runtime)
    artifact = build_request_artifact_v4(**kwargs, checkpoint_transfer=None)  # type: ignore[arg-type]
    return benchmark, write_request_artifact_v4(artifact).path, example.example_id, answer


def _evaluate(fixture: tuple[IrisStudyBenchmarkV1, Path, str, str]) -> IrisStudyEvaluationV1:
    benchmark, path, identity, answer = fixture
    with _assets(_tokenizer()):
        return evaluate_study_request_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=answer,
            terminal_token_id=1,
        )


@pytest.mark.parametrize("index", [0, 1, 3, 4, 5, 10])
def test_exact_text_receipt_binds_dataset_request_tokens_and_raw_output(
    tmp_path: Path, index: int
) -> None:
    fixture = _request(tmp_path, example_index=index)
    benchmark, path, identity, output = fixture
    evaluation = _evaluate(fixture)
    payload = evaluation.payload
    assert payload.outcome == "correct"
    assert payload.arm_id == "A" and payload.checkpoint_transfer is None
    assert payload.example_id == identity
    assert payload.benchmark_payload_sha256 == benchmark.payload_sha256
    assert payload.request_payload_sha256 == path.stem
    assert payload.input_binding.corpus_examples == 15
    assert payload.output_sha256 == bytes_sha256(output.encode())
    assert payload.output_byte_length == len(output.encode())
    assert payload.scientific_launch_status == "not_assessed"
    with _assets(_tokenizer()):
        verified = verify_study_evaluation_v1(
            evaluation, benchmark=benchmark, request_path=path, output_text=output
        )
    assert verified == evaluation and verified is not evaluation
    serialized = canonical_json_bytes(evaluation.model_dump(mode="json"))
    assert (
        benchmark.payload.dataset.payload.manifest.payload.examples[index].prompt.encode()
        not in serialized
    )
    assert output.encode() not in serialized


@pytest.mark.parametrize(
    "output,outcome", [("wrong", "incorrect"), ("", "invalid_output"), ("a\0b", "invalid_output")]
)
def test_bound_incorrect_or_invalid_output_remains_a_scored_result(
    tmp_path: Path, output: str, outcome: str
) -> None:
    assert _evaluate(_request(tmp_path, output=output)).payload.outcome == outcome


@pytest.mark.parametrize("change", ["same_length", "truncated", "masked"])
def test_complete_request_token_and_mask_identity_is_required_before_scoring(
    tmp_path: Path, change: str
) -> None:
    fixture = _request(tmp_path, token_change=change)
    with (
        patch.object(service, "score_study_answer_v1") as scorer,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)
    scorer.assert_not_called()


def test_all_corpus_budgets_must_fit_even_if_selected_text_example_fits(tmp_path: Path) -> None:
    fixture = _request(tmp_path, max_new_tokens=2)
    with _assets(_tokenizer()):
        report = preflight_study_inputs_v1(
            fixture[0],
            config=read_request_artifact_v4(fixture[1]).payload.configuration.value,
            terminal_token_id=1,
        )
    assert not report.examples[0].violations
    assert not report.inputs_fit
    assert any("target_over_limit" in item.violations for item in report.examples[1:])
    with pytest.raises(IrisStudyEvaluationError):
        _evaluate(fixture)


@pytest.mark.parametrize("change", ["output", "example"])
def test_wrong_private_input_is_rejected_before_tokenizer_loading(
    tmp_path: Path, change: str
) -> None:
    benchmark, path, identity, answer = _request(tmp_path)
    if change == "output":
        answer += "CHANGED_PRIVATE"
    else:
        identity = benchmark.payload.dataset.payload.manifest.payload.examples[1].example_id
    with (
        patch.object(service, "preflight_study_inputs_v1") as preflight,
        pytest.raises(IrisStudyEvaluationError) as caught,
    ):
        evaluate_study_request_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=answer,
            terminal_token_id=1,
        )
    preflight.assert_not_called()
    assert "CHANGED_PRIVATE" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("arm", ["B", "C", "D"])
def test_missing_teacher_checkpoint_cannot_produce_an_evaluation(tmp_path: Path, arm: str) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    path = write_request_artifact_v4(build_artifact_v4(tmp_path, arm)).path
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    with (
        patch.object(service, "preflight_study_inputs_v1") as preflight,
        pytest.raises(IrisStudyEvaluationError),
    ):
        evaluate_study_request_answer_v1(
            benchmark,
            request_path=path,
            example_id=example.example_id,
            output_text=example.target,
            terminal_token_id=1,
        )
    preflight.assert_not_called()


def test_functional_code_never_acquires_a_text_receipt(tmp_path: Path) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    with (
        patch.object(service, "verify_request_artifact_v4") as verifier,
        pytest.raises(IrisCodeEvaluationUnavailableError),
    ):
        evaluate_study_request_answer_v1(
            benchmark,
            request_path=tmp_path / "absent.json",
            example_id=example.example_id,
            output_text=example.target,
            terminal_token_id=1,
        )
    verifier.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "outcome",
        "split",
        "benchmark_payload_sha256",
        "request_payload_sha256",
        "configuration_sha256",
        "execution_profile_set_sha256",
        "output_sha256",
        "preflight_sha256",
        "tokenizer_semantics_sha256",
        "request_prompt_input_ids_sha256",
    ],
)
def test_outer_digest_is_not_a_substitute_for_recomputing_the_score(
    tmp_path: Path, field: str
) -> None:
    fixture = _request(tmp_path)
    benchmark, path, _, output = fixture
    raw = _evaluate(fixture).model_dump(mode="python")
    if field == "outcome":
        raw["payload"][field] = "incorrect"
    elif field == "split":
        raw["payload"][field] = "test"
    elif field in {
        "preflight_sha256",
        "tokenizer_semantics_sha256",
        "request_prompt_input_ids_sha256",
    }:
        raw["payload"]["input_binding"][field] = "0" * 64
    else:
        raw["payload"][field] = "0" * 64
    raw["payload_sha256"] = domain_sha256(
        "iris/study-evaluation/v1/payload", canonical_json_bytes(raw["payload"])
    )
    forged = IrisStudyEvaluationV1.model_validate(raw)
    assert validate_study_evaluation_v1(forged) == forged
    with _assets(_tokenizer()), pytest.raises(IrisStudyEvaluationError):
        verify_study_evaluation_v1(
            forged, benchmark=benchmark, request_path=path, output_text=output
        )


@pytest.mark.parametrize("mutation", ["version", "code", "arm"])
def test_schema_rejects_coerced_version_or_inapplicable_scoring_scope(
    tmp_path: Path, mutation: str
) -> None:
    raw = _evaluate(_request(tmp_path)).model_dump(mode="python")
    if mutation == "version":
        raw["schema_identity"]["version"] = True
    elif mutation == "code":
        raw["payload"]["task_family"] = "code_generation_and_verification"
    else:
        raw["payload"]["arm_id"] = "B"
    with pytest.raises(ValidationError):
        IrisStudyEvaluationV1.model_validate(raw)


def test_changed_request_at_final_verification_returns_no_receipt(tmp_path: Path) -> None:
    fixture = _request(tmp_path)
    initial = verify_request_artifact_v4(fixture[1])
    changed = replace(initial, payload_sha256="0" * 64)
    with (
        patch.object(service, "verify_request_artifact_v4", side_effect=[initial, changed]),
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)


def test_invalid_envelope_and_size_are_rejected(tmp_path: Path) -> None:
    evaluation = _evaluate(_request(tmp_path))
    with pytest.raises(IrisStudyEvaluationError):
        validate_study_evaluation_v1(evaluation.model_copy(update={"payload_sha256": "0" * 64}))
    with patch.object(service, "_MAX_EVALUATION_BYTES", 1), pytest.raises(IrisStudyEvaluationError):
        validate_study_evaluation_v1(evaluation)
    with pytest.raises(IrisStudyEvaluationError):
        validate_study_evaluation_v1(cast(IrisStudyEvaluationV1, None))


@pytest.mark.parametrize("field", ["transformers_version", "tokenizers_version"])
def test_changed_scoring_software_version_is_rejected_before_scoring(
    tmp_path: Path, field: str
) -> None:
    fixture = _request(tmp_path)
    benchmark, path, _, _ = fixture
    with _assets(_tokenizer()):
        report = preflight_study_inputs_v1(
            benchmark,
            config=read_request_artifact_v4(path).payload.configuration.value,
            terminal_token_id=1,
        )
    changed = (
        replace(report, transformers_version="different-version")
        if field == "transformers_version"
        else replace(report, tokenizers_version="different-version")
    )
    with (
        patch.object(service, "preflight_study_inputs_v1", return_value=changed),
        patch.object(service, "score_study_answer_v1") as scorer,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture)
    scorer.assert_not_called()


@pytest.mark.parametrize("change", ["missing_id", "unicode", "type"])
def test_invalid_private_inputs_are_redacted_before_request_verification(
    tmp_path: Path, change: str
) -> None:
    benchmark, path, identity, answer = _request(tmp_path)
    private_value = "UNENCODABLE_PRIVATE" + chr(0xD800)
    if change == "missing_id":
        identity = "no-such-example"
    elif change == "unicode":
        answer = private_value
    else:
        answer = cast(str, None)
    with (
        patch.object(service, "verify_request_artifact_v4") as verifier,
        pytest.raises(IrisStudyEvaluationError) as caught,
    ):
        evaluate_study_request_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=answer,
            terminal_token_id=1,
        )
    verifier.assert_not_called()
    assert "UNENCODABLE_PRIVATE" not in "".join(traceback.format_exception(caught.value))
