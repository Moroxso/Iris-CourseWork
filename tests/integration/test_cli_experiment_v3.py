"""CLI contract for additive profile-v4 B/C/D experiment execution."""

from __future__ import annotations

from pathlib import Path
from re import search
from typing import Any, cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iris.application.experiment_runner_v3 import IrisExperimentExecutionResultV3
from iris.application.verification_v3 import IrisExperimentJournalVerificationReportV3
from iris.cli import app
from tests.integration.test_cli import _arm_experiment_run_arguments

pytestmark = pytest.mark.integration

runner = CliRunner()


def _arguments(tmp_path: Path, *, arm_id: str) -> list[str]:
    arguments = _arm_experiment_run_arguments(tmp_path, arm_id=arm_id)
    if arguments[:2] != ["experiment", "run-arm"]:
        raise AssertionError("legacy CLI fixture changed unexpectedly")
    arguments[1] = "run-arm-v3"
    return arguments


def _result(root: Path, *, arm_id: str) -> IrisExperimentExecutionResultV3:
    return IrisExperimentExecutionResultV3(
        arm_id=cast(Any, arm_id),
        path=root / "experiments" / "v3" / "cli-profile-v4",
        status="complete",
        run_payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        event_count=5,
        optimizer_completed_steps=2,
        completed_epochs=1,
        next_epoch_index=None,
        next_training_ordinal=None,
        latest_checkpoint_schema_version=cast(
            Any,
            {"B": 2, "C": 3, "D": 1}[arm_id],
        ),
        latest_checkpoint_payload_sha256="3" * 64,
        replayed_training_steps=1,
        new_training_steps=1,
    )


def test_run_arm_v3_help_exposes_additive_profile_v4_command() -> None:
    result = runner.invoke(app, ["experiment", "run-arm-v3", "--help"])

    assert result.exit_code == 0
    assert "A, B, C, or D" in result.stdout
    assert "canonical v3 run" in result.stdout
    assert "run-arm-v3" not in result.stderr


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_run_arm_v3_forwards_exact_trainable_arm_and_renders_redacted_result(
    tmp_path: Path,
    arm_id: str,
) -> None:
    root = tmp_path / "runs"
    report = _result(root, arm_id=arm_id)
    with patch(
        "iris.cli.arm_experiment_command_v3_service.run_packaged_arm_experiment_v3",
        return_value=report,
    ) as run:
        result = runner.invoke(app, _arguments(tmp_path, arm_id=arm_id), terminal_width=220)

    assert result.exit_code == 0
    kwargs = run.call_args.kwargs
    assert kwargs["arm_id"] == arm_id
    assert kwargs["root"] == tmp_path / "runs"
    assert kwargs["allow_download"] is False
    assert "Profile-v4 B/C/D experiment execution" in result.stdout
    assert search(rf"Arm\s+│\s+{arm_id}\s+│", result.stdout) is not None
    assert "Latest checkpoint schema" in result.stdout
    assert "private metadata" in result.stdout


def test_run_arm_v3_rejects_arm_a_before_execution_service(tmp_path: Path) -> None:
    with patch(
        "iris.cli.arm_experiment_command_v3_service.run_packaged_arm_experiment_v3"
    ) as run:
        result = runner.invoke(app, _arguments(tmp_path, arm_id="A"))

    assert result.exit_code == 1
    assert "no trainable teacher graph" in result.stderr
    run.assert_not_called()


def test_run_arm_v3_rejects_lowercase_label_as_usage_error(tmp_path: Path) -> None:
    with patch(
        "iris.cli.arm_experiment_command_v3_service.run_packaged_arm_experiment_v3"
    ) as run:
        result = runner.invoke(app, _arguments(tmp_path, arm_id="c"))

    assert result.exit_code == 2
    run.assert_not_called()


def test_verify_journal_dispatches_profile_v4_report_without_v2_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments" / "v3" / "cli-profile-v4"
    manifest_path = tmp_path / "manifest.json"
    report = IrisExperimentJournalVerificationReportV3(
        path=path,
        artifact_type="iris.experiment_run",
        version=3,
        run_id="cli-profile-v4",
        arm_id="C",
        arm_kind="llm_modules_unlimited_shared_buffer",
        request_graph="append_only_all_proposals_prior_cycle_shared_buffer_v1",
        teacher_graph="append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
        execution_profile_set_version=4,
        execution_profile_set_sha256=(
            "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
        ),
        shared_buffer_policy_sha256=(
            "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
        ),
        run_payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        manifest_payload_sha256="3" * 64,
        execution_mode="deterministic_epochs",
        event_count=5,
        training_step_event_count=2,
        evaluation_event_count=2,
        checkpoint_event_count=1,
        optimizer_completed_steps=2,
        sidecar_state_sha256="4" * 64,
        chain_head_sha256="5" * 64,
        checkpoint_reference_count=2,
        unique_checkpoint_count=2,
        latest_checkpoint_schema_version=3,
        latest_checkpoint_payload_sha256="6" * 64,
        managed_phase="complete",
        completed_epochs=1,
        active_epoch=None,
        next_training_ordinal=None,
    )
    with (
        patch(
            "iris.cli.verification_v3_service.verify_experiment_journal_v3",
            return_value=report,
        ) as verify_v3,
        patch("iris.cli.verification_v2_service.verify_experiment_journal_v2") as verify_v2,
    ):
        result = runner.invoke(
            app,
            [
                "verify",
                "journal",
                "--path",
                str(path),
                "--manifest",
                str(manifest_path),
            ],
            terminal_width=240,
        )

    assert result.exit_code == 0
    verify_v3.assert_called_once_with(path=path, manifest_path=manifest_path)
    verify_v2.assert_not_called()
    assert "Profile-v4 experiment journal verification" in result.stdout
    assert "llm_modules_unlimited_shared_buffer" in result.stdout
    assert "Shared-buffer policy SHA-256" in result.stdout
    assert "Latest checkpoint schema version" in result.stdout
