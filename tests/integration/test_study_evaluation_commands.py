"""Packaged parsing, real private file flows and redacted complete-evaluation commands."""

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

import iris.application as application_api
from iris.application import study_code_evaluation as scorer
from iris.application import study_evaluation_command as service
from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_benchmark_io import read_study_benchmark_v1, write_study_benchmark_v1
from iris.application.study_evaluation_run_io import read_study_evaluation_summary_v1
from iris.application.study_generation_io import read_study_generation_v1, write_study_generation_v1
from iris.cli import app
from iris.integrations.python_sandbox import PythonSandboxError
from tests.integration.test_study_code_evaluation_smoke import _empty_inventory
from tests.unit.test_study_code_evaluation import _POLICY
from tests.unit.test_study_evaluation_run import _BUDGET, _runtime
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_input_preflight import _assets, _tokenizer

pytestmark = pytest.mark.integration
runner = CliRunner()


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "generation_path": tmp_path / "generation.json",
        "benchmark_path": tmp_path / "benchmark.json",
        "request_root": tmp_path / "artifacts",
        "directory": tmp_path / "evaluation",
    }


def _persist(tmp_path: Path) -> dict[str, Path]:
    generation, benchmark, _ = _fixture(tmp_path)
    paths = _paths(tmp_path)
    write_study_generation_v1(generation, path=paths["generation_path"])
    write_study_benchmark_v1(benchmark, path=paths["benchmark_path"])
    return paths


def _options(tmp_path: Path) -> dict[str, str]:
    paths = _paths(tmp_path)
    return {
        "--generation": str(paths["generation_path"]),
        "--benchmark": str(paths["benchmark_path"]),
        "--request-root": str(paths["request_root"]),
        "--directory": str(paths["directory"]),
        "--mode": "new",
        "--code-profile": _POLICY.profile,
        "--code-wall-time-ms": str(_POLICY.wall_time_ms),
        "--code-max-output-bytes": str(_POLICY.max_output_bytes),
        "--max-storage-bytes": str(_BUDGET),
    }


def _args(
    tmp_path: Path, *, verify: bool = False, changes: dict[str, str | None] | None = None
) -> list[str]:
    options = _options(tmp_path)
    if verify:
        options = {
            key: value
            for key, value in options.items()
            if key in ("--generation", "--benchmark", "--request-root", "--directory")
        }
    for key, value in (changes or {}).items():
        if value is None:
            options.pop(key)
        else:
            options[key] = value
    return [
        *(["verify", "study-evaluation"] if verify else ["study", "evaluate"]),
        *(item for pair in options.items() for item in pair),
    ]


def _policy() -> dict[str, Any]:
    return dict(
        code_profile=_POLICY.profile,
        code_wall_time_ms=_POLICY.wall_time_ms,
        code_max_output_bytes=_POLICY.max_output_bytes,
        max_storage_bytes=_BUDGET,
        resume=False,
    )


def _evaluate_files(
    paths: dict[str, Path], **kwargs: Any
) -> service.IrisStudyEvaluationFileReportV1:
    return service.evaluate_study_generation_file_v1(
        generation_path=paths["generation_path"],
        benchmark_path=paths["benchmark_path"],
        request_root=paths["request_root"],
        directory=paths["directory"],
        **kwargs,
    )


def _snapshot(tmp_path: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_real_cli_new_resume_verify_redaction_and_no_execution_reuse(tmp_path: Path) -> None:
    paths = _persist(tmp_path)
    generation = read_study_generation_v1(paths["generation_path"])
    benchmark = read_study_benchmark_v1(paths["benchmark_path"])
    with (
        _runtime() as runtime,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
    ):
        first = runner.invoke(app, _args(tmp_path))
        assert first.exit_code == 0, first.output
        before = _snapshot(tmp_path)
        resumed = runner.invoke(app, _args(tmp_path, changes={"--mode": "resume"}))
        assert resumed.exit_code == 0, resumed.output
        verified = runner.invoke(app, _args(tmp_path, verify=True))
        assert verified.exit_code == 0, verified.output
        assert runtime.await_count == 1
        loader.assert_not_called()
    assert before == _snapshot(tmp_path)
    summary = read_study_evaluation_summary_v1(paths["directory"])
    assert summary.payload.overall.model_dump() == dict(
        total=5, correct=4, incorrect=0, invalid_output=1
    )
    for result in (first, resumed, verified):
        assert "content-redacted" in result.output and "not_assessed" in result.output
        assert "Invalid outputs" in result.output and "Total examples" in result.output
        for answer in generation.payload.answers:
            assert answer.example_id not in result.output
        for example in benchmark.payload.dataset.payload.manifest.payload.examples:
            assert example.prompt not in result.output
        assert "def solve" not in result.output
        assert "stdout_base64" not in result.output
    assert "Evaluated now" in first.output and "Reused evaluations" in resumed.output
    assert "Evaluated now" not in verified.output


def test_file_api_reports_complete_counts_without_private_example_references(
    tmp_path: Path,
) -> None:
    paths = _persist(tmp_path)
    with _runtime():
        evaluated = _evaluate_files(paths, **_policy())
        verified = service.verify_study_evaluation_files_v1(**paths)
    assert evaluated.evaluated_now == 5 and evaluated.reused_evaluations == 0
    assert evaluated.retained_file_bytes is not None and evaluated.retained_file_bytes > 0
    assert verified.evaluated_now is None and verified.reused_evaluations is None
    assert verified.retained_file_bytes is None
    assert evaluated.summary_payload_sha256 == verified.summary_payload_sha256
    assert evaluated.overall == verified.overall and evaluated.by_family == verified.by_family
    assert verified.arm_id == "A" and verified.split == "train"
    rendered = repr(asdict(verified))
    for answer in read_study_generation_v1(paths["generation_path"]).payload.answers:
        assert answer.example_id not in rendered
    assert "def solve" not in rendered


@pytest.mark.parametrize("option", list(_options(Path("C:/unused")).keys()))
def test_every_evaluation_option_is_required_before_application_call(
    tmp_path: Path, option: str
) -> None:
    with patch.object(service, "evaluate_study_generation_file_v1") as evaluate:
        result = runner.invoke(app, _args(tmp_path, changes={option: None}))
        assert result.exit_code == 2
        evaluate.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option", ["--generation", "--benchmark", "--request-root", "--directory"])
def test_every_verifier_path_is_required(tmp_path: Path, option: str) -> None:
    with patch.object(service, "verify_study_evaluation_files_v1") as verify:
        result = runner.invoke(app, _args(tmp_path, verify=True, changes={option: None}))
        assert result.exit_code == 2
        verify.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--mode", "automatic"),
        ("--code-profile", "host-python"),
        ("--code-wall-time-ms", "0"),
        ("--code-wall-time-ms", "600001"),
        ("--code-max-output-bytes", "1023"),
        ("--code-max-output-bytes", str(8 * 1024**2 + 1)),
        ("--max-storage-bytes", "4095"),
        ("--max-storage-bytes", str(20 * 1024**3 + 1)),
        ("--max-storage-bytes", "auto"),
    ],
)
def test_invalid_cli_policy_never_calls_evaluator(tmp_path: Path, option: str, value: str) -> None:
    with patch.object(service, "evaluate_study_generation_file_v1") as evaluate:
        result = runner.invoke(app, _args(tmp_path, changes={option: value}))
        assert result.exit_code == 2
        evaluate.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("code_profile", "host-python"),
        ("code_wall_time_ms", True),
        ("code_max_output_bytes", "4096"),
        ("max_storage_bytes", True),
        ("max_storage_bytes", 4095),
        ("max_storage_bytes", 20 * 1024**3 + 1),
        ("resume", 1),
    ],
)
def test_direct_api_rejects_implicit_or_invalid_policy_before_reading_files(
    tmp_path: Path, name: str, value: Any
) -> None:
    kwargs = _policy()
    kwargs[name] = value
    with patch.object(service, "read_study_generation_v1") as read:
        with pytest.raises(IrisStudyEvaluationError):
            _evaluate_files(_paths(tmp_path), **kwargs)
        read.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_invalid_private_files_are_redacted_even_in_debug(
    tmp_path: Path, verify: bool, debug: bool
) -> None:
    paths = _paths(tmp_path)
    paths["generation_path"].write_bytes(b'{"answer":"PRIVATE_CANDIDATE_SECRET"}')
    before = _snapshot(tmp_path)
    args = [*(["--debug"] if debug else []), *_args(tmp_path, verify=verify)]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "PRIVATE_CANDIDATE_SECRET" not in result.output
    assert "Evaluation complete" not in result.output
    assert _snapshot(tmp_path) == before
    if not debug:
        assert "Traceback" not in result.output


@pytest.mark.parametrize("debug", [False, True])
def test_unavailable_runtime_has_distinct_exit_and_preserves_prefix(
    tmp_path: Path, debug: bool
) -> None:
    paths = _persist(tmp_path)
    with (
        _assets(_tokenizer()),
        patch.object(
            scorer,
            "execute_python_job_v1",
            AsyncMock(side_effect=PythonSandboxError("PRIVATE_RUNTIME_SECRET")),
        ),
    ):
        result = runner.invoke(app, [*(["--debug"] if debug else []), *_args(tmp_path)])
    assert result.exit_code == 3
    assert "PRIVATE_RUNTIME_SECRET" not in result.output
    assert not (paths["directory"] / "summary.json").exists()
    assert len(list((paths["directory"] / "receipts").iterdir())) == 2
    with _runtime() as runtime:
        resumed = runner.invoke(app, _args(tmp_path, changes={"--mode": "resume"}))
        assert resumed.exit_code == 0, resumed.output
        assert runtime.await_count == 1


@pytest.mark.parametrize("change", ["new_again", "budget", "receipt"])
def test_conflicting_resume_never_overwrites_or_reexecutes(tmp_path: Path, change: str) -> None:
    paths = _persist(tmp_path)
    with _runtime() as runtime:
        assert runner.invoke(app, _args(tmp_path)).exit_code == 0
        changes: dict[str, str | None] = {"--mode": "resume"}
        if change == "new_again":
            changes["--mode"] = "new"
        elif change == "budget":
            changes["--max-storage-bytes"] = str(_BUDGET + 1)
        else:
            (paths["directory"] / "receipts/000000.json").write_bytes(b"PRIVATE_BAD_RECEIPT")
        before = _snapshot(tmp_path)
        result = runner.invoke(app, _args(tmp_path, changes=changes))
        assert result.exit_code == 1
        assert "PRIVATE_BAD_RECEIPT" not in result.output
        assert runtime.await_count == 1
        assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(
    "path_key", ["generation_path", "benchmark_path", "request_root", "directory"]
)
def test_relative_paths_rejected_without_resolution_or_writes(
    tmp_path: Path, verify: bool, path_key: str
) -> None:
    paths = _paths(tmp_path)
    paths[path_key] = Path("relative.json")
    with pytest.raises(IrisStudyEvaluationError):
        if verify:
            service.verify_study_evaluation_files_v1(**paths)
        else:
            _evaluate_files(paths, **_policy())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("path_key", ["generation_path", "benchmark_path"])
def test_changed_input_file_at_return_cannot_report_success(
    tmp_path: Path, verify: bool, path_key: str
) -> None:
    paths = _persist(tmp_path)
    with _runtime():
        if verify:
            _evaluate_files(paths, **_policy())
        name = "verify_study_evaluation_run_v1" if verify else "evaluate_study_split_v1"
        original = getattr(service, name)

        def mutate(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            paths[path_key].write_bytes(b"PRIVATE_INPUT_REPLACEMENT")
            return result

        with patch.object(service, name, side_effect=mutate):
            result = runner.invoke(app, _args(tmp_path, verify=verify))
        assert result.exit_code == 1
        assert "PRIVATE_INPUT_REPLACEMENT" not in result.output
        assert (paths["directory"] / "summary.json").exists()


def test_read_only_verifier_rejects_missing_run_without_creating_it(tmp_path: Path) -> None:
    _persist(tmp_path)
    before = _snapshot(tmp_path)
    with patch("iris.application.study_code_evaluation.execute_python_job_v1") as execute:
        result = runner.invoke(app, _args(tmp_path, verify=True))
        assert result.exit_code == 1
        execute.assert_not_called()
    assert before == _snapshot(tmp_path)
    assert not _paths(tmp_path)["directory"].exists()


@pytest.mark.parametrize(
    "args", [["study", "evaluate", "--help"], ["verify", "study-evaluation", "--help"]]
)
def test_new_command_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "--generation" in result.output and "--request-root" in result.output


def test_public_file_apis_are_exported() -> None:
    for name in service.__all__:
        assert name in application_api.__all__
        assert getattr(application_api, name) is getattr(service, name)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires the installed opt-in IrisCodeEval gVisor runtime",
)
def test_actual_gvisor_cli_run_and_read_only_verification(tmp_path: Path) -> None:
    paths = _persist(tmp_path)
    with (
        _assets(_tokenizer()),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
    ):
        result = runner.invoke(app, _args(tmp_path))
        assert result.exit_code == 0, result.output
        assert read_study_evaluation_summary_v1(paths["directory"]).payload.overall.total == 5
        before = _snapshot(tmp_path)
        with patch.object(scorer, "execute_python_job_v1") as execute:
            resumed = runner.invoke(app, _args(tmp_path, changes={"--mode": "resume"}))
            verified = runner.invoke(app, _args(tmp_path, verify=True))
            assert resumed.exit_code == verified.exit_code == 0
            execute.assert_not_called()
        loader.assert_not_called()
    assert _snapshot(tmp_path) == before
    _empty_inventory()
