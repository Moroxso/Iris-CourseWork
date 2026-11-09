"""Source-v2 declaration, version separation, saved-report inspection and CLI delivery."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from iris.application import scientific_answer_command_v2 as command
from iris.application import scientific_evaluation as legacy
from iris.application import scientific_evaluation_v2 as evaluation
from iris.application import scientific_generation as generation
from iris.application import synthetic_scoring_v2 as scoring
from iris.artifacts.scientific_evaluation_v2 import (
    IrisScientificEvaluationV2,
    ScientificEvaluationPayloadV2,
    ScientificPythonLimitsV2,
)
from iris.cli import app
from iris.integrations.python_sandbox import PythonSandboxError
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_answer_commands import _descriptor, _redacted
from tests.integration.test_scientific_command import _files
from tests.integration.test_scientific_evaluation import CaptureCase, _fixture_answers, _recapture
from tests.integration.test_scientific_evaluation import _evaluate as evaluate_legacy
from tests.integration.test_scientific_evaluation import captured as captured
from tests.integration.test_scientific_evaluation import case as case
from tests.integration.test_scientific_evaluation_v2 import _evaluate
from tests.integration.test_scientific_generation import _test_spec
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = pytest.mark.integration
runner = CliRunner()


@pytest.fixture(scope="module")
def report(captured: CaptureCase) -> IrisScientificEvaluationV2:
    return _evaluate(captured)


def _arguments(case: CaptureCase, descriptor: Path, output: Path) -> list[str]:
    return [
        "scientific",
        "evaluate-v2",
        "--config",
        str(case.recipe),
        "--execution",
        str(case.execution),
        "--test-input",
        str(descriptor),
        "--generation",
        str(case.generation_path),
        "--output-root",
        str(output),
        "--code-profile",
        _POLICY.profile,
    ]


def test_versions_have_distinct_stores_and_no_fallback(case: CaptureCase, tmp_path: Path) -> None:
    old, new = evaluate_legacy(case), _evaluate(case)
    old_path = legacy.write_scientific_evaluation_v1(old, root=tmp_path / "reports")
    new_path = evaluation.write_scientific_evaluation_v2(new, root=tmp_path / "reports")
    assert old.sha256 != new.sha256 and old_path.parent != new_path.parent
    assert old.payload.overall == new.payload.overall
    assert old.payload.families == new.payload.families
    assert command.read_scientific_evaluation_report(old_path) == old
    assert command.read_scientific_evaluation_report(new_path) == new
    for reader, path in (
        (legacy.read_scientific_evaluation_v1, new_path),
        (evaluation.read_scientific_evaluation_v2, old_path),
    ):
        with pytest.raises(evaluation.IrisScientificEvaluationError):
            reader(path)
    raw = new.model_dump(mode="python")
    with pytest.raises(ValueError):
        type(old).model_validate(raw)
    with pytest.raises(ValueError):
        IrisScientificEvaluationV2.model_validate(old.model_dump(mode="python"))


@pytest.mark.parametrize("kind", ["missing_lf", "extra_lf", "unknown", "duplicate", "space"])
def test_dispatch_requires_canonical_known_version_without_downgrade(
    report: IrisScientificEvaluationV2, tmp_path: Path, kind: str
) -> None:
    path = evaluation.write_scientific_evaluation_v2(report, root=tmp_path)
    file = path / "evaluation.json"
    content = file.read_bytes()
    if kind == "missing_lf":
        content = content[:-1]
    elif kind == "extra_lf":
        content += b"\n"
    elif kind == "space":
        content = b" " + content
    elif kind == "duplicate":
        content = content[:-2] + b',"sha256":"' + report.sha256.encode() + b'"}\n'
    else:
        raw = json.loads(content)
        raw["payload"]["kind"] = "iris-scientific-evaluation-v99"
        content = canonical_json_file_bytes(raw)
    file.write_bytes(content)
    with (
        patch.object(command, "read_scientific_evaluation_v1") as old,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        command.read_scientific_evaluation_report(path)
    old.assert_not_called()


@pytest.mark.parametrize(
    "field,bad",
    [
        ("cpu_cores", True),
        ("cpu_cores", 2),
        ("memory_bytes", 512 * 1024**2),
        ("memory_bytes", "512000000"),
        ("process_limit", 128),
        ("candidate_timeout_ms", 5001),
        ("candidate_timeout_ms", 5000.0),
        ("swap_bytes", 1),
        ("host_mount_count", True),
        ("environment", "inherited"),
        ("network", "bridge"),
        ("root_filesystem", "writable"),
        ("clock_scope", "including_startup"),
    ],
)
def test_literal_source_limits_cannot_be_relabelled(field: str, bad: Any) -> None:
    raw = ScientificPythonLimitsV2().model_dump(mode="python")
    raw[field] = bad
    with pytest.raises(ValueError):
        ScientificPythonLimitsV2.model_validate(raw)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("python_limits", None),
        ("python_policy", None),
        ("python_resource_scope", "not_applicable_no_code"),
        ("kind", "iris-scientific-evaluation-v1"),
        ("scoring_profile", "iris-synthetic-01-scoring-v1"),
    ],
)
def test_report_cross_checks_declared_profile_even_without_execution(
    report: IrisScientificEvaluationV2, field: str, bad: Any
) -> None:
    assert report.payload.overall.code_executed == 0
    raw = report.payload.model_dump(mode="python")
    raw[field] = bad
    with pytest.raises(ValueError):
        ScientificEvaluationPayloadV2.model_validate(raw)


def test_cli_roundtrip_shows_limits_is_model_free_and_inspection_needs_no_sources(
    case: CaptureCase, tmp_path: Path
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    before = (_files(case.execution), _files(case.generation_path))
    args = _arguments(case, descriptor, tmp_path / "reports")
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch.object(generation, "generate_scientific_test_v1") as decode,
        patch.object(scoring, "execute_python_job_v2") as sandbox,
    ):
        first = runner.invoke(app, args)
        retry = runner.invoke(app, args)
        assert first.exit_code == retry.exit_code == 0, first.output
        (path,) = (tmp_path / "reports" / "scientific-evaluation-v2").iterdir()
        report = evaluation.read_scientific_evaluation_v2(path)
        assert report.payload.python_limits == ScientificPythonLimitsV2()
        case.recipe.unlink()
        with patch.object(evaluation, "_audit_inputs") as audit:
            inspected = runner.invoke(
                app, ["scientific", "inspect-evaluation", "--evaluation", str(path)]
            )
        audit.assert_not_called()
    assert inspected.exit_code == 0, inspected.output
    for mock in (tokenizer, model, decode, sandbox):
        mock.assert_not_called()
    for result in (first, retry, inspected):
        _redacted(result.output, case)
        assert "Declared Python source limits" in result.output
        assert "512000000" in result.output
        assert "not_assessed" in result.output
    assert "source not re-audited" in inspected.output
    assert before == (_files(case.execution), _files(case.generation_path))


@pytest.mark.parametrize("kind", ["legacy_profile", "missing", "implicit_budget", "small_budget"])
def test_cli_invalid_or_implicit_policy_never_dispatches(
    case: CaptureCase, tmp_path: Path, kind: str
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    args = _arguments(case, descriptor, tmp_path / "out")
    if kind == "legacy_profile":
        args[-1] = "iris_wsl_gvisor_python312_restricted_v1"
    elif kind in {"missing", "implicit_budget"}:
        args = args[:-2]
        if kind == "implicit_budget":
            args += ["--maximum-code-output-bytes", "4096"]
    else:
        args += ["--maximum-code-output-bytes", "1023"]
    with patch.object(scoring, "execute_python_job_v2") as sandbox:
        result = runner.invoke(app, args)
    assert result.exit_code == (2 if kind in {"legacy_profile", "small_budget"} else 1)
    sandbox.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_v1_cli_rejects_source_v2_profile(case: CaptureCase, tmp_path: Path) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    args = _arguments(case, descriptor, tmp_path / "out")
    args[1] = "evaluate"
    result = runner.invoke(app, args)
    assert result.exit_code == 2 and not (tmp_path / "out").exists()


def test_cli_backend_failure_is_exit_three_and_never_a_partial_report(
    case: CaptureCase, tmp_path: Path
) -> None:
    case = _fixture_answers(case)
    descriptor = _descriptor(case, tmp_path / "data")
    with patch.object(
        scoring,
        "execute_python_job_v2",
        AsyncMock(side_effect=PythonSandboxError("PRIVATE_MARKER")),
    ):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out"))
    assert result.exit_code == 3
    assert "PRIVATE_MARKER" not in result.output and "Traceback" not in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("overlap", ["execution", "descriptor", "train", "validation"])
def test_cli_store_cannot_modify_bound_inputs(
    case: CaptureCase, tmp_path: Path, overlap: str
) -> None:
    root = case.execution if overlap == "execution" else tmp_path / "out"
    store = root / "scientific-evaluation-v2"
    descriptor = _descriptor(case, store if overlap == "descriptor" else tmp_path / "data")
    if overlap in {"train", "validation"}:
        store.mkdir(parents=True)
        recipe = json.loads(case.recipe.read_text(encoding="utf-8"))
        target = store / f"{overlap}.parquet"
        target.write_bytes((case.recipe.parent / recipe[overlap]["path"]).read_bytes())
        recipe[overlap]["path"] = str(target)
        case.recipe.write_text(json.dumps(recipe), encoding="utf-8")
    before = _files(case.execution)
    with patch.object(command, "evaluate_scientific_generation_v2") as evaluate:
        result = runner.invoke(app, _arguments(case, descriptor, root))
    assert result.exit_code == 1
    evaluate.assert_not_called()
    assert _files(case.execution) == before and descriptor.exists()


def test_descriptor_change_after_scoring_prevents_publication(
    case: CaptureCase, tmp_path: Path
) -> None:
    descriptor = _descriptor(case, tmp_path / "data")
    original = evaluation.evaluate_scientific_generation_v2

    def changed(**kwargs: Any) -> Any:
        result = original(**kwargs)
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
        raw["sha256"] = "0" * 64
        descriptor.write_text(json.dumps(raw), encoding="utf-8")
        return result

    with patch.object(command, "evaluate_scientific_generation_v2", side_effect=changed):
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out"))
    assert result.exit_code == 1 and not (tmp_path / "out").exists()


def test_noncode_cli_uses_v2_without_claiming_python_execution(
    case: CaptureCase, tmp_path: Path
) -> None:
    rows = tuple(row for row in case.rows if not row.executable)
    generated = tuple(
        row for row in case.capture.payload.rows if row.task_type != "split_spec_repair"
    )
    spec = _test_spec(case.recipe.parent / "noncode.parquet", rows)
    _, digest = generation._own_test_rows(rows, excluded_seeds=set(), scientific=False)
    case = replace(
        _recapture(
            case, rows=generated, test_parquet_sha256=spec.sha256, test_corpus_sha256=digest
        ),
        rows=rows,
        test=spec,
    )
    descriptor = _descriptor(case, tmp_path / "data")
    with patch.object(scoring, "execute_python_job_v2") as sandbox:
        result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out")[:-2])
    assert result.exit_code == 0, result.output
    sandbox.assert_not_called()
    (path,) = (tmp_path / "out" / "scientific-evaluation-v2").iterdir()
    report = evaluation.read_scientific_evaluation_v2(path)
    assert report.payload.python_limits is None
    assert report.payload.python_policy is None
    assert report.payload.python_resource_scope == "not_applicable_no_code"
    assert "Declared Python source limits" not in result.output


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit IrisCodeEval gate",
)
def test_actual_source_profile_cli_scores_controlled_references(
    case: CaptureCase, tmp_path: Path
) -> None:
    case = _fixture_answers(case)
    descriptor = _descriptor(case, tmp_path / "data")
    result = runner.invoke(app, _arguments(case, descriptor, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    (path,) = (tmp_path / "out" / "scientific-evaluation-v2").iterdir()
    report = evaluation.read_scientific_evaluation_v2(path)
    assert report.payload.overall.correct == 4
    assert report.payload.overall.code_executed == report.payload.overall.code_correct == 1
    assert report.payload.python_limits == ScientificPythonLimitsV2()
    _redacted(result.output, case)


def test_help_does_not_load_or_dispatch() -> None:
    with patch.object(command, "evaluate_scientific_answers_command_v2") as evaluate:
        result = runner.invoke(app, ["scientific", "evaluate-v2", "--help"])
    assert result.exit_code == 0
    evaluate.assert_not_called()
