"""Scientific CLI: actual decoding, saved-capture scoring, bounded/redacted delivery."""

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from iris.application import scientific_answer_command as command
from iris.application import scientific_evaluation as evaluation
from iris.application import scientific_generation as generation
from iris.application import synthetic_scoring as scoring
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.cli import app
from iris.integrations.python_sandbox import PythonSandboxError
from tests.integration.test_scientific_command import _files
from tests.integration.test_scientific_command import loaders as loaders
from tests.integration.test_scientific_evaluation import CaptureCase, _fixture_answers
from tests.integration.test_scientific_evaluation import captured as captured
from tests.integration.test_scientific_evaluation import case as case
from tests.unit.test_synthetic_scoring import _POLICY

pytestmark = pytest.mark.integration
runner = CliRunner()


def _descriptor(case: CaptureCase, root: Path) -> Path:
    root.mkdir(parents=True)
    parquet = root / "held-out.parquet"
    parquet.write_bytes((case.recipe.parent / case.test.path).read_bytes())
    descriptor = root / "input.json"
    descriptor.write_text(
        json.dumps({**case.test.model_dump(mode="json"), "path": parquet.name}), encoding="utf-8"
    )
    return descriptor


def _arguments(case: CaptureCase, descriptor: Path, output: Path, operation: str) -> list[str]:
    args = [
        "scientific",
        operation,
        "--config",
        str(case.recipe),
        "--execution",
        str(case.execution),
        "--test-input",
        str(descriptor),
        "--output-root",
        str(output),
    ]
    if operation == "generate":
        return [*args, "--strategy", "greedy", "--seed", "1337"]
    return [*args, "--generation", str(case.generation_path), "--code-profile", _POLICY.profile]


def _redacted(output: str, case: CaptureCase) -> None:
    assert "expected_response" not in output and "hidden_facts" not in output
    for source, generated in zip(case.rows, case.capture.payload.rows, strict=True):
        assert source.prompt not in output
        assert source.expected_response not in output
        if len(generated.output_text) > 16:
            assert generated.output_text not in output


def test_real_generate_command_reproduces_best_capture(
    case: CaptureCase, tmp_path: Path, loaders: Any
) -> None:
    descriptor = _descriptor(case, tmp_path / "separate-data")
    output = tmp_path / "new-captures"
    before = _files(case.execution)
    with patch.object(scoring, "execute_python_job_v1") as sandbox:
        result = runner.invoke(app, _arguments(case, descriptor, output, "generate"))
    assert result.exit_code == 0, result.output
    path = output / "scientific-generation-v1" / case.capture.sha256
    assert generation.read_scientific_generation_v1(path) == case.capture
    sandbox.assert_not_called()
    assert _files(case.execution) == before
    _redacted(result.output, case)
    assert "source-bound generation" in result.output
    assert "Selected step" in result.output and "Captured target NLL/token" in result.output
    assert len(list(path.parent.iterdir())) == 1


def test_evaluate_and_inspect_are_model_free_and_never_print_answers(
    case: CaptureCase, tmp_path: Path
) -> None:
    descriptor = _descriptor(case, tmp_path / "separate-data")
    output = tmp_path / "reports"
    args = _arguments(case, descriptor, output, "evaluate")
    source_before, capture_before = _files(case.execution), _files(case.generation_path)
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch.object(command, "generate_scientific_test_v1") as generate,
        patch.object(scoring, "execute_python_job_v1") as sandbox,
    ):
        first = runner.invoke(app, args)
        repeated = runner.invoke(app, args)
        assert first.exit_code == repeated.exit_code == 0, first.output
        (path,) = (output / "scientific-evaluation-v1").iterdir()
        report = evaluation.read_scientific_evaluation_v1(path)
        assert report.payload.overall.examples == report.payload.overall.invalid_output == 4
        inspect = runner.invoke(
            app, ["scientific", "inspect-evaluation", "--evaluation", str(path)]
        )
        capture = runner.invoke(
            app, ["scientific", "inspect-generation", "--generation", str(case.generation_path)]
        )
        assert inspect.exit_code == capture.exit_code == 0
    tokenizer.assert_not_called()
    model.assert_not_called()
    generate.assert_not_called()
    sandbox.assert_not_called()
    for answer in (first, repeated, inspect, capture):
        _redacted(answer.output, case)
    assert "source-bound scoring" in first.output
    assert "source not re-audited" in inspect.output and "source not re-audited" in capture.output
    assert "Quality metrics" in first.output
    assert "not_assessed" in first.output
    assert _files(case.execution) == source_before
    assert _files(case.generation_path) == capture_before


@pytest.mark.parametrize(
    "kind", ["extra", "duplicate", "oversize", "wrong_type", "too_many", "network", "digest"]
)
def test_bad_descriptor_never_starts_generation(
    case: CaptureCase, tmp_path: Path, kind: str
) -> None:
    path = _descriptor(case, tmp_path / "data")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if kind == "extra":
        raw["PRIVATE_MARKER"] = "not allowed"
    elif kind == "wrong_type":
        raw["maximum_rows"] = "4"
    elif kind == "too_many":
        raw["maximum_rows"] = 5001
    elif kind == "network":
        raw["path"] = "https://PRIVATE_MARKER/rows.parquet"
    elif kind == "digest":
        raw["sha256"] = "PRIVATE_MARKER"
    text = json.dumps(raw)
    if kind == "duplicate":
        text = text[:-1] + ',"path":"PRIVATE_MARKER"}'
    elif kind == "oversize":
        text = "PRIVATE_MARKER" * 2000
    path.write_text(text, encoding="utf-8")
    with patch.object(command, "generate_scientific_test_v1") as generate:
        result = runner.invoke(app, _arguments(case, path, tmp_path / "out", "generate"))
    assert result.exit_code == 1
    assert "descriptor is invalid" in result.output and "PRIVATE_MARKER" not in result.output
    generate.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ["generate", "evaluate"])
@pytest.mark.parametrize("overlap", ["execution", "descriptor"])
def test_output_store_cannot_modify_immutable_inputs(
    case: CaptureCase, tmp_path: Path, operation: str, overlap: str
) -> None:
    kind = "scientific-generation-v1" if operation == "generate" else "scientific-evaluation-v1"
    root = case.execution if overlap == "execution" else tmp_path / "out"
    descriptor = _descriptor(case, tmp_path / "data" if overlap == "execution" else root / kind)
    before = _files(case.execution)
    with (
        patch.object(command, "generate_scientific_test_v1") as generate,
        patch.object(command, "evaluate_scientific_generation_v1") as evaluate,
    ):
        result = runner.invoke(app, _arguments(case, descriptor, root, operation))
    assert result.exit_code == 1
    generate.assert_not_called()
    evaluate.assert_not_called()
    assert _files(case.execution) == before
    assert descriptor.exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--seed", "-1"],
        ["--top-p", "1.5"],
        ["--strategy", "beam"],
        ["--maximum-storage-bytes", "0"],
        ["--temperature", "0"],
        ["--temperature", "nan"],
        ["--top-p", "0"],
        ["--top-p", "0.8"],
    ],
)
def test_invalid_decoding_never_loads_or_publishes(
    case: CaptureCase, tmp_path: Path, flags: list[str]
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    with patch.object(command, "generate_scientific_test_v1") as generate:
        result = runner.invoke(
            app, _arguments(case, descriptor, tmp_path / "out", "generate") + flags
        )
    assert result.exit_code in {1, 2}
    generate.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_sampling_flags_reach_real_generation(
    case: CaptureCase, tmp_path: Path, loaders: Any
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    args = _arguments(case, descriptor, tmp_path / "out", "generate")
    args += ["--strategy", "sampling", "--temperature", "0.7", "--top-p", "0.8", "--seed", "2026"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    (path,) = (tmp_path / "out" / "scientific-generation-v1").iterdir()
    capture = generation.read_scientific_generation_v1(path)
    assert capture.payload.decoding == ScientificDecodingPolicyV1(
        strategy="sampling", seed=2026, temperature=0.7, top_p=0.8
    )
    assert all(row.sampling_seed is not None for row in capture.payload.rows)
    assert capture.payload.selected_step == 1


def test_backend_unavailable_exits_three_without_partial_report(
    case: CaptureCase, tmp_path: Path
) -> None:
    case = _fixture_answers(case)  # Controlled scoring fixture, NOT learned-model quality.
    descriptor = _descriptor(case, tmp_path / "data")
    with patch.object(
        scoring,
        "execute_python_job_v1",
        AsyncMock(side_effect=PythonSandboxError("PRIVATE_MARKER")),
    ):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", "evaluate"))
    assert result.exit_code == 3
    assert "infrastructure is unavailable" in result.output
    assert "PRIVATE_MARKER" not in result.output and "Traceback" not in result.output
    assert not (tmp_path / "out").exists()
    _redacted(result.output, case)


def test_missing_code_profile_is_not_implicitly_enabled(case: CaptureCase, tmp_path: Path) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    args = _arguments(case, descriptor, tmp_path / "out", "evaluate")[:-2]
    with patch.object(scoring, "execute_python_job_v1") as sandbox:
        missing = runner.invoke(app, args)
        unused = runner.invoke(app, [*args, "--maximum-code-output-bytes", "4096"])
    assert missing.exit_code == unused.exit_code == 1
    assert "explicit code profile" in unused.output
    sandbox.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_descriptor_change_after_decode_does_not_publish(case: CaptureCase, tmp_path: Path) -> None:
    descriptor = _descriptor(case, tmp_path / "data")

    def changed(**kwargs: Any) -> Any:
        data = json.loads(descriptor.read_text(encoding="utf-8"))
        data["sha256"] = "0" * 64
        descriptor.write_text(json.dumps(data), encoding="utf-8")
        return case.capture

    with patch.object(command, "generate_scientific_test_v1", side_effect=changed):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", "generate"))
    assert result.exit_code == 1 and not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "operation,option",
    [("inspect-generation", "--generation"), ("inspect-evaluation", "--evaluation")],
)
def test_inspection_missing_artifact_is_redacted(
    operation: str, option: str, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["scientific", operation, option, str(tmp_path / "PRIVATE_MARKER")])
    assert result.exit_code == 1
    assert "PRIVATE_MARKER" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize(
    "operation", ["generate", "evaluate", "inspect-generation", "inspect-evaluation"]
)
def test_help_is_available_without_model_or_sandbox(operation: str) -> None:
    with patch.object(command, "generate_scientific_test_v1") as generate:
        result = runner.invoke(app, ["scientific", operation, "--help"])
    assert result.exit_code == 0
    generate.assert_not_called()


@pytest.mark.parametrize("operation", ["generate", "evaluate"])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_output_store_cannot_contain_original_recipe_data(
    case: CaptureCase, tmp_path: Path, operation: str, split: str
) -> None:
    kind = "scientific-generation-v1" if operation == "generate" else "scientific-evaluation-v1"
    descriptor = _descriptor(case, tmp_path / "data")
    store = tmp_path / "out" / kind
    store.mkdir(parents=True)
    recipe = json.loads(case.recipe.read_text(encoding="utf-8"))
    target = store / f"{split}.parquet"
    content = (case.recipe.parent / recipe[split]["path"]).read_bytes()
    target.write_bytes(content)
    recipe[split]["path"] = str(target)
    case.recipe.write_text(json.dumps(recipe), encoding="utf-8")
    with (
        patch.object(command, "generate_scientific_test_v1") as generate,
        patch.object(command, "evaluate_scientific_generation_v1") as evaluate,
    ):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", operation))
    assert result.exit_code == 1
    generate.assert_not_called()
    evaluate.assert_not_called()
    assert target.read_bytes() == content
    assert list(store.iterdir()) == [target]


def test_inspection_is_artifact_only_when_original_recipe_is_unavailable(
    case: CaptureCase, tmp_path: Path
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    created = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", "evaluate"))
    assert created.exit_code == 0, created.output
    (path,) = (tmp_path / "out" / "scientific-evaluation-v1").iterdir()
    case.recipe.unlink()
    with patch.object(command, "evaluate_scientific_generation_v1") as evaluate:
        read = runner.invoke(app, ["scientific", "inspect-evaluation", "--evaluation", str(path)])
    assert read.exit_code == 0
    assert "artifact integrity only" in read.output and "source not re-audited" in read.output
    evaluate.assert_not_called()


def test_small_report_quota_leaves_capture_retryable(case: CaptureCase, tmp_path: Path) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    before = _files(case.generation_path)
    args = _arguments(case, descriptor, tmp_path / "out", "evaluate")
    failed = runner.invoke(app, [*args, "--maximum-storage-bytes", "1"])
    assert failed.exit_code == 1
    assert list((tmp_path / "out" / "scientific-evaluation-v1").iterdir()) == []
    assert _files(case.generation_path) == before
    retry = runner.invoke(app, args)
    assert retry.exit_code == 0
    assert len(list((tmp_path / "out" / "scientific-evaluation-v1").iterdir())) == 1


def test_changed_descriptor_after_scoring_does_not_publish(
    case: CaptureCase, tmp_path: Path
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    original = evaluation.evaluate_scientific_generation_v1

    def changed(**kwargs: Any) -> Any:
        result = original(**kwargs)
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
        raw["sha256"] = "0" * 64
        descriptor.write_text(json.dumps(raw), encoding="utf-8")
        return result

    with patch.object(command, "evaluate_scientific_generation_v1", side_effect=changed):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", "evaluate"))
    assert result.exit_code == 1 and not (tmp_path / "out").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit real IrisCodeEval gate",
)
def test_actual_gvisor_cli_evaluation_and_inspection(case: CaptureCase, tmp_path: Path) -> None:
    case = _fixture_answers(case)
    descriptor = _descriptor(case, tmp_path / "data")
    result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out", "evaluate"))
    assert result.exit_code == 0, result.output
    (path,) = (tmp_path / "out" / "scientific-evaluation-v1").iterdir()
    report = evaluation.read_scientific_evaluation_v1(path)
    assert report.payload.overall.correct == 4
    assert report.payload.overall.code_executed == report.payload.overall.code_correct == 1
    with patch.object(scoring, "execute_python_job_v1") as sandbox:
        read = runner.invoke(app, ["scientific", "inspect-evaluation", "--evaluation", str(path)])
    assert read.exit_code == 0
    sandbox.assert_not_called()
    _redacted(result.output, case)
    _redacted(read.output, case)
