"""CLI grammar/persistence/error routing only; not native or installed acceptance."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_scientific_ablation_operator_v1 import _request

runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    [
        "",
        "new",
        "inspect",
        "best",
        "phase-request",
        "train",
        "validate",
        "checkpoint",
        "resume",
        "export-request",
        "export",
        "inspect-export",
    ],
)
def test_ablation_command_help_never_invokes_native_factories(command: str) -> None:
    arguments = ["scientific-ablation", *([command] if command else []), "--help"]
    with (
        patch(
            "iris.application.scientific_ablation_operator.create_scientific_ablation_command_lifecycle_v1",
            side_effect=AssertionError("No model"),
        ) as phase_factory,
        patch(
            "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("No model"),
        ) as export_factory,
    ):
        result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert phase_factory.call_count == export_factory.call_count == 0


def test_baseline_scientific_help_does_not_become_ablation_dispatch() -> None:
    result = runner.invoke(app, ["scientific", "--help"])
    assert result.exit_code == 0 and "create" in result.stdout
    assert "phase-request" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--operator", "operator", "--base", "."],
        ["resume", "--operator", "operator", "--base", "."],
        ["phase-request", "--operator", "operator", "--phase", "train", "--out", "intent.json"],
        [
            "phase-request",
            "--operator",
            "operator",
            "--phase",
            "latest",
            "--expected-previous-sha256",
            "a" * 64,
            "--out",
            "intent.json",
        ],
        [
            "export-request",
            "--operator",
            "operator",
            "--checkpoint-event-sha256",
            "a" * 64,
            "--out",
            "export.json",
        ],
        ["export", "--operator", "operator", "--base", ".", "--root", "exports"],
        [
            "new",
            "--recipe",
            "recipe.json",
            "--base",
            ".",
            "--root",
            "operators",
            "--maximum-parameter-bytes",
            "0",
        ],
        [
            "train",
            "--operator",
            "operator",
            "--request",
            "intent.json",
            "--base",
            ".",
            "--maximum-replay-phases",
            "-1",
        ],
    ],
)
def test_no_implicit_head_or_native_work_when_required_intent_controls_are_absent(
    arguments: list[str],
) -> None:
    with (
        patch(
            "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1"
        ) as advance,
        patch(
            "iris.application.scientific_ablation_operator_export.export_scientific_ablation_operator_v1"
        ) as project,
    ):
        result = runner.invoke(app, ["scientific-ablation", *arguments])
    assert result.exit_code == 2 and advance.call_count == project.call_count == 0


def test_phase_request_is_saved_exclusively_before_native_action(tmp_path: Path) -> None:
    request = _request()
    output = tmp_path / "intents" / "saved.json"
    arguments = [
        "scientific-ablation",
        "phase-request",
        "--operator",
        str(tmp_path / "operator"),
        "--expected-previous-sha256",
        request.payload.expected_previous_sha256,
        "--phase",
        "train",
        "--out",
        str(output),
    ]
    with (
        patch(
            "iris.application.scientific_ablation_operator.make_scientific_ablation_phase_request_v1",
            return_value=request,
        ) as make,
        patch(
            "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1",
            side_effect=AssertionError("Intent creation never trains"),
        ) as advance,
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        before = output.lstat()
        repeat = runner.invoke(app, arguments)
    assert repeat.exit_code == 0 and make.call_count == 2 and advance.call_count == 0
    assert json.loads(result.stdout) == request.model_dump(mode="json")
    assert output.read_bytes() == canonical_json_file_bytes(request.model_dump(mode="json"))
    after = output.lstat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
    )


@pytest.mark.parametrize("existing", [b"old private intent", b"{"])
def test_existing_or_partial_intent_is_not_overwritten_or_executed(
    tmp_path: Path, existing: bytes
) -> None:
    request = _request()
    output = tmp_path / "saved.json"
    output.write_bytes(existing)
    with (
        patch(
            "iris.application.scientific_ablation_operator.make_scientific_ablation_phase_request_v1",
            return_value=request,
        ),
        patch(
            "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1"
        ) as advance,
    ):
        result = runner.invoke(
            app,
            [
                "scientific-ablation",
                "phase-request",
                "--operator",
                str(tmp_path / "operator"),
                "--expected-previous-sha256",
                request.payload.expected_previous_sha256,
                "--phase",
                "train",
                "--out",
                str(output),
            ],
        )
    assert result.exit_code == 2 and output.read_bytes() == existing and advance.call_count == 0
    assert "old private intent" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("command", ["validate", "checkpoint"])
def test_phase_command_cannot_silently_execute_another_valid_intent(
    tmp_path: Path, command: str
) -> None:
    request = _request()
    saved = tmp_path / "train.json"
    saved.write_bytes(canonical_json_file_bytes(request.model_dump(mode="json")))
    with patch(
        "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1"
    ) as advance:
        result = runner.invoke(
            app,
            [
                "scientific-ablation",
                command,
                "--operator",
                str(tmp_path / "operator"),
                "--request",
                str(saved),
                "--base",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 2 and advance.call_count == 0


def test_resume_forwards_same_saved_request_and_explicit_replay_limit(tmp_path: Path) -> None:
    request = _request()
    saved = tmp_path / "train.json"
    saved.write_bytes(canonical_json_file_bytes(request.model_dump(mode="json")))
    with patch(
        "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1",
        side_effect=ValueError("PRIVATE_CORPUS_CONTENT"),
    ) as advance:
        result = runner.invoke(
            app,
            [
                "scientific-ablation",
                "resume",
                "--operator",
                str(tmp_path / "operator"),
                "--request",
                str(saved),
                "--base",
                str(tmp_path),
                "--maximum-replay-phases",
                "7",
            ],
        )
    assert result.exit_code == 2 and advance.call_count == 1
    assert advance.call_args.kwargs == dict(request=request, base=tmp_path, maximum_replay_phases=7)
    assert "PRIVATE_CORPUS_CONTENT" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("fault,exit_code", [(SystemExit(9), 9), (KeyboardInterrupt(), 130)])
def test_primary_signal_is_not_redacted_into_success_or_a_different_phase(
    fault: BaseException, exit_code: int
) -> None:
    with patch(
        "iris.application.scientific_ablation_operator.inspect_scientific_ablation_operator_v1",
        side_effect=fault,
    ):
        result = runner.invoke(app, ["scientific-ablation", "inspect", "--operator", "operator"])
    assert result.exit_code == exit_code and "Ablation command failed" not in result.output


def test_intent_cannot_be_written_into_operator_source(tmp_path: Path) -> None:
    source = tmp_path / "operator"
    source.mkdir()
    request = _request()
    with patch(
        "iris.application.scientific_ablation_operator.make_scientific_ablation_phase_request_v1",
        return_value=request,
    ):
        result = runner.invoke(
            app,
            [
                "scientific-ablation",
                "phase-request",
                "--operator",
                str(source),
                "--expected-previous-sha256",
                request.payload.expected_previous_sha256,
                "--phase",
                "train",
                "--out",
                str(source / "injected.json"),
            ],
        )
    assert result.exit_code == 2 and list(source.iterdir()) == []


def test_partial_intent_write_signal_is_preserved_and_does_not_launch_native_work(
    tmp_path: Path,
) -> None:
    request = _request()
    output = tmp_path / "saved.json"

    def interrupted(path: Path, raw: bytes) -> None:
        path.write_bytes(raw[:1])
        raise SystemExit(9)

    with (
        patch(
            "iris.application.scientific_ablation_operator.make_scientific_ablation_phase_request_v1",
            return_value=request,
        ),
        patch("iris.scientific_ablation_cli._write_exact_file", side_effect=interrupted),
        patch(
            "iris.application.scientific_ablation_operator.advance_scientific_ablation_operator_v1"
        ) as advance,
    ):
        result = runner.invoke(
            app,
            [
                "scientific-ablation",
                "phase-request",
                "--operator",
                str(tmp_path / "operator"),
                "--expected-previous-sha256",
                request.payload.expected_previous_sha256,
                "--phase",
                "train",
                "--out",
                str(output),
            ],
        )
    assert result.exit_code == 9 and output.read_bytes() == b"{" and advance.call_count == 0
