"""Packaged generation parsing, real private file bindings and redacted recovery reports."""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import iris.application as application_api
from iris.application import study_generation_command as service
from iris.application import study_generation_run as generation_service
from iris.application import study_generation_run_io as storage
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_benchmark_io import read_study_benchmark_v1, write_study_benchmark_v1
from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_study_evaluation_commands import _args as evaluation_args
from tests.unit.test_study_evaluation_run import _runtime
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_generation_run import _Fixture, _inference, _saved

pytestmark = pytest.mark.integration
runner = CliRunner()


def _paths(root: Path) -> dict[str, Path]:
    return {
        "benchmark_path": root / "benchmark.json",
        "configuration_path": root / "configuration.json",
        "decoding_path": root / "decoding.json",
        "directory": root / "run",
    }


def _persist(root: Path) -> tuple[_Fixture, dict[str, Path]]:
    fixture = _fixture(root)
    generation, benchmark, request_root = fixture
    paths = _paths(root)
    write_study_benchmark_v1(benchmark, path=paths["benchmark_path"])
    request = read_request_artifact_v4(
        request_root
        / "requests"
        / "v4"
        / f"{generation.payload.answers[0].request_payload_sha256}.json"
    )
    paths["configuration_path"].write_bytes(
        canonical_json_file_bytes(request.payload.configuration.value.model_dump(mode="json"))
    )
    paths["decoding_path"].write_bytes(
        canonical_json_file_bytes(
            [
                {
                    "example_id": answer.example_id,
                    "task_family": answer.task_family,
                    "decoding": answer.decoding.model_dump(mode="json"),
                }
                for answer in generation.payload.answers
            ]
        )
    )
    return fixture, paths


def _options(root: Path) -> dict[str, str]:
    paths = _paths(root)
    return {
        "--benchmark": str(paths["benchmark_path"]),
        "--configuration": str(paths["configuration_path"]),
        "--decoding": str(paths["decoding_path"]),
        "--directory": str(paths["directory"]),
        "--arm": "A",
        "--split": "train",
        "--terminal-token-id": "1",
        "--mode": "new",
        "--max-artifact-bytes": str(1024 * 1024),
        "--max-record-bytes": str(1024 * 1024),
        "--max-storage-bytes": str(16 * 1024 * 1024),
    }


def _args(
    root: Path, *, verify: bool = False, changes: Mapping[str, str | None] | None = None
) -> list[str]:
    options = _options(root)
    if verify:
        options = {
            key: value for key, value in options.items() if key in ("--benchmark", "--directory")
        }
    for key, value in (changes or {}).items():
        if value is None:
            options.pop(key)
        else:
            options[key] = value
    return [
        *(["verify", "study-generation"] if verify else ["study", "generate"]),
        *(item for pair in options.items() for item in pair),
    ]


def _controls() -> dict[str, Any]:
    return dict(
        arm_id="A",
        checkpoint_identity=None,
        split="train",
        terminal_token_id=1,
        max_artifact_bytes=1024 * 1024,
        max_record_bytes=1024 * 1024,
        max_storage_bytes=16 * 1024 * 1024,
        resume=False,
    )


def _generate(paths: dict[str, Path], **controls: Any) -> service.IrisStudyGenerationFileReportV1:
    return service.generate_study_split_files_v1(
        benchmark_path=paths["benchmark_path"],
        configuration_path=paths["configuration_path"],
        decoding_path=paths["decoding_path"],
        directory=paths["directory"],
        **controls,
    )


def test_cli_new_resume_verify_redacted_complete_counts_and_unchanged_storage(
    tmp_path: Path,
) -> None:
    fixture, paths = _persist(tmp_path)
    with _inference(fixture) as seen:
        first = runner.invoke(app, _args(tmp_path))
        assert first.exit_code == 0, first.output
        before = _saved(tmp_path)
        resumed = runner.invoke(app, _args(tmp_path, changes={"--mode": "resume"}))
        assert resumed.exit_code == 0, resumed.output
        verified = runner.invoke(app, _args(tmp_path, verify=True))
        assert verified.exit_code == 0, verified.output
        assert seen == list(range(5))
        assert _saved(tmp_path) == before
        report = service.verify_study_generation_files_v1(
            benchmark_path=paths["benchmark_path"], directory=paths["directory"]
        )
    assert report.total_examples == 5 and len(report.by_family) == 5
    assert all(count == 1 for _, count in report.by_family)
    assert report.generated_now is report.reused_answers is report.retained_file_bytes is None
    assert report.arm_id == "A" and report.split == "train"
    for result in (first, resumed, verified):
        assert "content-redacted" in result.output and "not_assessed" in result.output
        assert "Total examples" in result.output
        assert "def solve" not in result.output and "sampling_seed" not in result.output
        for answer in fixture[0].payload.answers:
            assert answer.example_id not in result.output
        for example in fixture[1].payload.dataset.payload.manifest.payload.examples:
            assert example.prompt not in result.output
    assert "Generated now" in first.output and "Reused answers" in resumed.output
    assert "Generated now" not in verified.output
    assert "def solve" not in repr(asdict(report))


def test_cli_resumes_complete_pending_record_without_reinference(tmp_path: Path) -> None:
    fixture, _ = _persist(tmp_path)
    original = storage._stage

    def interrupted(path: Path, content: bytes, **kwargs: Any) -> None:
        original(path, content, **kwargs)
        if path.name == ".answer-pending.json" and json.loads(content)["payload"]["index"] == 2:
            raise KeyboardInterrupt

    with _inference(fixture) as seen:
        with patch.object(generation_service, "_stage", side_effect=interrupted):
            stopped = runner.invoke(app, _args(tmp_path))
            assert stopped.exit_code != 0
        assert seen == [0, 1, 2]
        result = runner.invoke(app, _args(tmp_path, changes={"--mode": "resume"}))
        assert result.exit_code == 0, result.output
        assert seen == [0, 1, 2, 3, 4]
        assert runner.invoke(app, _args(tmp_path, verify=True)).exit_code == 0


@pytest.mark.parametrize("option", tuple(_options(Path("C:/unused"))))
def test_every_generation_option_required_before_service(tmp_path: Path, option: str) -> None:
    with patch.object(service, "generate_study_split_files_v1") as generate:
        result = runner.invoke(app, _args(tmp_path, changes={option: None}))
        assert result.exit_code == 2
        generate.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option", ("--benchmark", "--directory"))
def test_verifier_options_required(tmp_path: Path, option: str) -> None:
    with patch.object(service, "verify_study_generation_files_v1") as verify:
        assert (
            runner.invoke(app, _args(tmp_path, verify=True, changes={option: None})).exit_code == 2
        )
        verify.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--arm", "E"),
        ("--split", "all"),
        ("--mode", "auto"),
        ("--terminal-token-id", "-1"),
        ("--max-artifact-bytes", "1023"),
        ("--max-artifact-bytes", str(64 * 1024**2 + 1)),
        ("--max-record-bytes", "4095"),
        ("--max-record-bytes", str(64 * 1024**2 + 1)),
        ("--max-storage-bytes", "4095"),
        ("--max-storage-bytes", str(20 * 1024**3 + 1)),
    ],
)
def test_invalid_cli_controls_never_call_service(tmp_path: Path, option: str, value: str) -> None:
    with patch.object(service, "generate_study_split_files_v1") as generate:
        assert runner.invoke(app, _args(tmp_path, changes={option: value})).exit_code == 2
        generate.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resume", 1),
        ("split", "all"),
        ("terminal_token_id", True),
        ("max_artifact_bytes", True),
        ("max_record_bytes", "4096"),
        ("max_storage_bytes", 4095),
        ("arm_id", "E"),
        ("arm_id", "B"),
    ],
)
def test_direct_invalid_controls_rejected_before_file_read(
    tmp_path: Path, field: str, value: Any
) -> None:
    controls = _controls()
    controls[field] = value
    with patch.object(service, "read_study_benchmark_v1") as read:
        with pytest.raises(IrisStudyGenerationError):
            _generate(_paths(tmp_path), **controls)
        read.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field", ("benchmark_path", "configuration_path", "decoding_path", "directory")
)
@pytest.mark.parametrize("kind", ("relative", "traversal", "string"))
def test_input_paths_cannot_be_implicitly_resolved(tmp_path: Path, field: str, kind: str) -> None:
    paths: dict[str, Any] = _paths(tmp_path)
    paths[field] = {
        "relative": Path("input.json"),
        "traversal": tmp_path / ".." / "input.json",
        "string": str(tmp_path / "input.json"),
    }[kind]
    with patch.object(service, "read_study_benchmark_v1") as read:
        with pytest.raises(IrisStudyGenerationError):
            _generate(paths, **_controls())
        read.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        "missing_config_field",
        "config_download",
        "config_extra",
        "decode_download",
        "decode_seed_missing",
        "decode_duplicate",
        "decode_reorder",
        "decode_family",
        "decode_unknown_field",
        "decode_boolean_seed",
        "noncanonical",
        "duplicate_key",
        "nonfinite",
        "oversized_config",
    ],
)
def test_invalid_private_policy_files_fail_before_runner(tmp_path: Path, kind: str) -> None:
    _, paths = _persist(tmp_path)
    config_path, decode_path = paths["configuration_path"], paths["decoding_path"]
    config, decoded = json.loads(config_path.read_bytes()), json.loads(decode_path.read_bytes())
    if kind == "missing_config_field":
        del config["backbone"]["local_files_only"]
    elif kind == "config_download":
        config["backbone"]["local_files_only"] = False
    elif kind == "config_extra":
        config["PRIVATE_CONFIGURATION_SECRET"] = "PRIVATE_VALUE_SECRET"
    elif kind == "decode_download":
        decoded[0]["decoding"]["allow_download"] = True
    elif kind == "decode_seed_missing":
        del decoded[0]["decoding"]["sampling_seed"]
    elif kind == "decode_duplicate":
        decoded[1] = decoded[0]
    elif kind == "decode_reorder":
        decoded.reverse()
    elif kind == "decode_family":
        decoded[0]["task_family"] = decoded[1]["task_family"]
    elif kind == "decode_unknown_field":
        decoded[0]["PRIVATE_POLICY_SECRET"] = "PRIVATE_VALUE_SECRET"
    elif kind == "decode_boolean_seed":
        decoded[0]["decoding"].update(strategy="sampling", sampling_seed=True)
    config_path.write_bytes(canonical_json_file_bytes(config))
    decode_path.write_bytes(canonical_json_file_bytes(decoded))
    if kind == "noncanonical":
        decode_path.write_bytes(json.dumps(decoded, indent=2).encode())
    elif kind == "duplicate_key":
        config_path.write_bytes(b'{"seed":0,"seed":1}\n')
    elif kind == "nonfinite":
        decode_path.write_bytes(b"[NaN]\n")
    elif kind == "oversized_config":
        config_path.write_bytes(b" " * (1024 * 1024 + 1))
    before = _saved(tmp_path)
    with patch.object(service, "generate_study_split_run_v1") as generate:
        with pytest.raises(IrisStudyGenerationError, match="did not complete"):
            _generate(paths, **_controls())
        generate.assert_not_called()
    assert _saved(tmp_path) == before


@pytest.mark.parametrize("verify", (False, True))
@pytest.mark.parametrize("debug", (False, True))
def test_private_parse_failures_are_redacted_including_debug(
    tmp_path: Path, verify: bool, debug: bool
) -> None:
    _paths(tmp_path)["benchmark_path"].write_bytes(b'{"prompt":"PRIVATE_BENCHMARK_SECRET"}')
    before = _saved(tmp_path)
    result = runner.invoke(app, [*(["--debug"] if debug else []), *_args(tmp_path, verify=verify)])
    assert result.exit_code == 1
    assert "PRIVATE_BENCHMARK_SECRET" not in result.output
    assert "Complete study generation" not in result.output
    assert _saved(tmp_path) == before


@pytest.mark.parametrize(
    "field", ("benchmark_path", "configuration_path", "decoding_path", "run", "generation")
)
def test_late_file_mutation_never_reports_complete(tmp_path: Path, field: str) -> None:
    fixture, paths = _persist(tmp_path)
    original = generation_service.generate_study_split_run_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = paths[field] if field in paths else paths["directory"] / f"{field}.json"
        previous = path.stat()
        content = path.read_bytes()
        path.write_bytes(b" " + content[1:])  # same size, restored mtime, invalid canonical input
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        return result

    with (
        _inference(fixture),
        patch.object(service, "generate_study_split_run_v1", side_effect=changed),
    ):
        result = runner.invoke(app, _args(tmp_path))
        assert result.exit_code == 1
        assert "Complete study generation" not in result.output
    assert (paths["directory"] / "generation.json").exists()  # preserve evidence on late failure


@pytest.mark.parametrize("arm", ("A", "B", "C", "D"))
def test_checkpoint_cli_constraints_before_input_reads(tmp_path: Path, arm: str) -> None:
    changes = {"--arm": arm}
    if arm == "A":
        changes["--checkpoint-payload-sha256"] = "a" * 64
    with patch.object(service, "generate_study_split_files_v1") as generate:
        assert runner.invoke(app, _args(tmp_path, changes=changes)).exit_code == 1
        generate.assert_not_called()


def test_public_exports_and_command_help() -> None:
    for name in service.__all__:
        assert getattr(application_api, name) is getattr(service, name)
        assert name in application_api.__all__
    for command in (("study", "generate"), ("verify", "study-generation")):
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert "--allow-download" not in result.output


@pytest.mark.parametrize("change", ("new_again", "budget", "configuration", "decoding", "record"))
def test_cli_conflicting_resume_preserves_all_evidence_without_inference(
    tmp_path: Path,
    change: str,
) -> None:
    fixture, paths = _persist(tmp_path)
    with _inference(fixture) as seen:
        assert runner.invoke(app, _args(tmp_path)).exit_code == 0
        changes = {"--mode": "resume"}
        if change == "new_again":
            changes["--mode"] = "new"
        elif change == "budget":
            changes["--max-storage-bytes"] = str(16 * 1024 * 1024 + 1)
        elif change == "configuration":
            raw = json.loads(paths["configuration_path"].read_bytes())
            raw["runtime"]["seed"] += 1
            paths["configuration_path"].write_bytes(canonical_json_file_bytes(raw))
        elif change == "decoding":
            raw = json.loads(paths["decoding_path"].read_bytes())
            for entry in raw:
                entry["decoding"]["temperature"] = 0.5
            paths["decoding_path"].write_bytes(canonical_json_file_bytes(raw))
        else:
            (paths["directory"] / "answers/000000.json").write_bytes(b"PRIVATE_DAMAGED_ANSWER")
        before = _saved(tmp_path)
        result = runner.invoke(app, _args(tmp_path, changes=changes))
        assert result.exit_code == 1 and "PRIVATE_DAMAGED_ANSWER" not in result.output
        assert seen == list(range(5))
        assert _saved(tmp_path) == before


@pytest.mark.parametrize("field", ("benchmark", "run", "generation"))
def test_read_only_command_rechecks_files_after_core_verification(
    tmp_path: Path, field: str
) -> None:
    fixture, paths = _persist(tmp_path)
    original = generation_service.verify_study_generation_run_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = (
            paths["benchmark_path"]
            if field == "benchmark"
            else paths["directory"] / f"{field}.json"
        )
        path.write_bytes(b"PRIVATE_LATE_REPLACEMENT")
        return result

    with _inference(fixture) as seen:
        _generate(paths, **_controls())
        with patch.object(service, "verify_study_generation_run_v1", side_effect=changed):
            result = runner.invoke(app, _args(tmp_path, verify=True))
        assert result.exit_code == 1 and "PRIVATE_LATE_REPLACEMENT" not in result.output
        assert seen == list(range(5))


@pytest.mark.parametrize("verify", (False, True))
def test_final_output_reread_after_original_benchmark_check(tmp_path: Path, verify: bool) -> None:
    fixture, paths = _persist(tmp_path)
    read = read_study_benchmark_v1
    calls = 0

    def changed(path: Path) -> Any:
        nonlocal calls
        result = read(path)
        calls += 1
        if calls == 2:
            (paths["directory"] / "generation.json").write_bytes(b"PRIVATE_FINAL_REPLACEMENT")
        return result

    with _inference(fixture):
        if verify:
            _generate(paths, **_controls())
        with patch.object(service, "read_study_benchmark_v1", side_effect=changed):
            result = runner.invoke(app, _args(tmp_path, verify=verify))
        assert result.exit_code == 1 and "PRIVATE_FINAL_REPLACEMENT" not in result.output
        assert calls == 2


def test_verifier_does_not_create_missing_run(tmp_path: Path) -> None:
    _persist(tmp_path)
    before = _saved(tmp_path)
    with patch.object(generation_service, "_iter_study_generation_answers") as inference:
        result = runner.invoke(app, _args(tmp_path, verify=True))
        assert result.exit_code == 1
        inference.assert_not_called()
    assert _saved(tmp_path) == before


@pytest.mark.parametrize("field", ("benchmark_path", "configuration_path", "decoding_path"))
def test_original_inputs_cannot_be_inside_managed_run(tmp_path: Path, field: str) -> None:
    paths = _paths(tmp_path)
    paths[field] = paths["directory"] / "original.json"
    with patch.object(service, "read_study_benchmark_v1") as read:
        with pytest.raises(IrisStudyGenerationError):
            _generate(paths, **_controls())
        read.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_generated_file_flows_into_existing_evaluation_and_verification_commands(
    tmp_path: Path,
) -> None:
    fixture, paths = _persist(tmp_path)
    with _inference(fixture):
        assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    # Only the existing isolated observation is substituted; full bindings/regrading are real.
    changes: dict[str, str | None] = {"--generation": str(paths["directory"] / "generation.json")}
    with _runtime() as runtime:
        evaluated = runner.invoke(app, evaluation_args(tmp_path, changes=changes))
        assert evaluated.exit_code == 0, evaluated.output
        calls = runtime.await_count
        verified = runner.invoke(app, evaluation_args(tmp_path, verify=True, changes=changes))
        assert verified.exit_code == 0, verified.output
        assert runtime.await_count == calls == 1
