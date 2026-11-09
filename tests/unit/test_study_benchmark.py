"""Benchmark scoring, strict private-code cases, and nested integrity contracts."""

import traceback
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from iris.application import study_benchmark as service
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyBenchmarkError
from iris.application.study_benchmark import (
    build_study_benchmark_v1,
    score_study_answer_v1,
    validate_study_benchmark_v1,
)
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.artifacts.study_benchmark_v1 import (
    IrisStudyBenchmarkV1,
    StudyExampleScoringV1,
    StudyPythonIntegerListCaseV1,
    StudyPythonIntegerListRuleV1,
)
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256


def _benchmark() -> IrisStudyBenchmarkV1:
    return build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)


def test_builder_and_validator_own_nested_dataset_and_scoring() -> None:
    original = _benchmark()
    copied = build_study_benchmark_v1(
        dataset=original.payload.dataset, scoring=original.payload.scoring
    )
    validated = validate_study_benchmark_v1(copied)
    assert copied == original == validated
    assert copied is not original
    assert copied.payload.dataset is not original.payload.dataset
    assert copied.payload.scoring[2].rule is not original.payload.scoring[2].rule
    object.__setattr__(
        original.payload.dataset.payload.manifest.payload.examples[0], "target", "changed"
    )
    assert validate_study_benchmark_v1(copied) == validated


@pytest.mark.parametrize("suffix", ["", "\r\n", " \t"])
def test_correct_text_answers_use_only_declared_boundary_whitespace_normalization(
    suffix: str,
) -> None:
    benchmark = _benchmark()
    for example, scoring in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples,
        benchmark.payload.scoring,
        strict=True,
    ):
        if scoring.rule.kind != "exact_text_v1":
            continue
        output = "\t " + example.target + suffix
        result = score_study_answer_v1(benchmark, example_id=example.example_id, output_text=output)
        assert result.outcome == "correct"
        assert result.split == example.split
        assert result.benchmark_payload_sha256 == benchmark.payload_sha256
        assert result.output_sha256 == bytes_sha256(output.encode())
        assert result.output_byte_length == len(output.encode())


@pytest.mark.parametrize("output", ["wrong answer", "`D123`", "<think>reasoning</think>D123"])
def test_incorrect_or_wrapped_answers_are_not_silently_repaired(output: str) -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    assert (
        score_study_answer_v1(benchmark, example_id=example.example_id, output_text=output).outcome
        == "incorrect"
    )


@pytest.mark.parametrize("output", ["", " \t\r\n", "a\0b"])
def test_invalid_model_outputs_are_scored_failures_not_missing_results(output: str) -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    assert (
        score_study_answer_v1(benchmark, example_id=example.example_id, output_text=output).outcome
        == "invalid_output"
    )


def test_non_ascii_space_is_not_normalized_away() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    result = score_study_answer_v1(
        benchmark, example_id=example.example_id, output_text="\u00a0" + example.target
    )
    assert result.outcome == "incorrect"


def test_unencodable_output_for_an_existing_example_has_a_redacted_error() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    # Tracebacks legitimately include the caller's source line, not its local values.
    private_output = "PRIVATE\ud800"
    with pytest.raises(IrisStudyBenchmarkError) as caught:
        score_study_answer_v1(benchmark, example_id=example.example_id, output_text=private_output)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))


def test_output_limit_counts_utf8_bytes_and_preserves_the_raw_output_digest() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[0]
    output = "\u00e9" * (512 * 1024 + 1)
    result = score_study_answer_v1(benchmark, example_id=example.example_id, output_text=output)
    assert result.outcome == "invalid_output"
    assert result.output_byte_length == 1024 * 1024 + 2
    assert result.output_sha256 == bytes_sha256(output.encode("utf-8"))


def test_code_scoring_is_infrastructure_unavailable_even_for_reference_source() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    with patch("subprocess.Popen") as subprocess, pytest.raises(IrisCodeEvaluationUnavailableError):
        score_study_answer_v1(benchmark, example_id=example.example_id, output_text=example.target)
    subprocess.assert_not_called()


@pytest.mark.parametrize("level", ["outer", "dataset", "manifest"])
def test_rehashed_outer_envelope_cannot_hide_stale_nested_digest(level: str) -> None:
    raw = _benchmark().model_dump(mode="python")
    if level == "outer":
        raw["payload_sha256"] = "0" * 64
    else:
        target = raw["payload"]["dataset"]
        if level == "manifest":
            target = target["payload"]["manifest"]
        target["payload_sha256"] = "0" * 64
        payload = IrisStudyBenchmarkV1.model_validate(raw).payload
        raw["payload_sha256"] = domain_sha256(
            "iris/study-benchmark/v1/payload", canonical_json_bytes(payload.model_dump(mode="json"))
        )
    forged = IrisStudyBenchmarkV1.model_validate(raw)
    with pytest.raises(IrisStudyBenchmarkError):
        validate_study_benchmark_v1(forged)
    with pytest.raises(IrisStudyBenchmarkError):
        score_study_answer_v1(forged, example_id="absent", output_text="anything")


@pytest.mark.parametrize("mutation", ["order", "code_as_text", "text_as_code", "duplicate_case"])
def test_schema_rejects_scoring_membership_or_family_mismatches(mutation: str) -> None:
    raw = _benchmark().model_dump(mode="python")
    scoring = list(raw["payload"]["scoring"])
    if mutation == "order":
        scoring.reverse()
    elif mutation == "code_as_text":
        scoring[2]["rule"] = scoring[0]["rule"]
    elif mutation == "text_as_code":
        scoring[0]["rule"] = scoring[2]["rule"]
    else:
        cases = scoring[2]["rule"]["cases"]
        scoring[2]["rule"]["cases"] = (*cases, cases[0])
    raw["payload"]["scoring"] = tuple(scoring)
    with pytest.raises(ValidationError):
        IrisStudyBenchmarkV1.model_validate(raw)


@pytest.mark.parametrize("value", [True, 1.0, "1", 2**31])
def test_functional_case_inputs_are_bounded_exact_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        StudyPythonIntegerListCaseV1.model_validate({"inputs": (value,), "expected": 0})


def test_code_case_collection_requires_isolation_and_input_preservation() -> None:
    rule = _benchmark().payload.scoring[2].rule
    assert isinstance(rule, StudyPythonIntegerListRuleV1)
    assert rule.execution_requirement == "isolated_python_backend"
    assert rule.input_policy == "input_list_unchanged"
    assert rule.comparison == "exact_integer_excluding_bool"
    assert rule.aggregation == "all_cases_pass"


def test_size_limit_is_enforced_by_every_public_consumer() -> None:
    benchmark = _benchmark()
    with patch.object(service, "_MAX_BENCHMARK_BYTES", 10):
        with pytest.raises(IrisStudyBenchmarkError):
            validate_study_benchmark_v1(benchmark)
        with pytest.raises(IrisStudyBenchmarkError):
            build_study_benchmark_v1(
                dataset=benchmark.payload.dataset, scoring=benchmark.payload.scoring
            )


def test_builder_rejects_list_and_private_errors_are_redacted() -> None:
    benchmark = _benchmark()
    with pytest.raises(IrisStudyBenchmarkError):
        build_study_benchmark_v1(
            dataset=benchmark.payload.dataset, scoring=cast(tuple[StudyExampleScoringV1, ...], [])
        )
    private_id, private_output = "PRIVATE_ABSENT_ID", "PRIVATE\ud800"
    with pytest.raises(IrisStudyBenchmarkError) as caught:
        score_study_answer_v1(benchmark, example_id=private_id, output_text=private_output)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
