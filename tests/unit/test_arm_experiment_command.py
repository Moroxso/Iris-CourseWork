"""Application-boundary tests for the packaged B/D experiment command."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

import iris.application.arm_experiment_command as command
import iris.application.bootstrap as bootstrap_service
import iris.application.experiment_command as legacy_command
from iris.application.arm_profiles import get_arm_execution_profile_v3
from iris.application.errors import IrisArmExecutionProfileError, IrisExperimentCommandError
from iris.application.experiment_runner_v2 import IrisExperimentExecutionResultV2
from iris.backbone import BackboneAdapter
from iris.training import IrisTeacherOptimizer
from iris.utils.seeds import RuntimeStateSnapshot
from tests.unit.test_experiment_command import (
    CPU_INFO,
    _arguments,
    _manifest,
    _owned_config,
)

BACKBONE = cast(BackboneAdapter, object())
MODEL = cast(Any, object())
OPTIMIZER = cast(IrisTeacherOptimizer, object())
RUNTIME_SNAPSHOT = cast(RuntimeStateSnapshot, object())


def _arm_arguments(
    tmp_path: Path,
    *,
    arm_id: object,
    **overrides: object,
) -> dict[str, Any]:
    values = _arguments(tmp_path)
    values.update(overrides)
    values["arm_id"] = arm_id
    return values


def _result(
    root: Path,
    *,
    arm_id: str,
    status: str = "complete",
) -> IrisExperimentExecutionResultV2:
    bounded = status == "step_limit_reached"
    return IrisExperimentExecutionResultV2(
        arm_id=cast(Any, arm_id),
        path=root / "experiments" / "v2" / "managed-command-run",
        status=cast(Any, status),
        run_payload_sha256="a" * 64,
        configuration_sha256="b" * 64,
        event_count=7,
        optimizer_completed_steps=3,
        completed_epochs=1,
        next_epoch_index=2 if bounded else None,
        next_training_ordinal=1 if bounded else None,
        latest_checkpoint_schema_version=2 if arm_id == "B" else 1,
        latest_checkpoint_payload_sha256="c" * 64,
        replayed_training_steps=1,
        new_training_steps=1 if bounded else 2,
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_command_composes_exact_arm_and_forwards_v2_runner(
    tmp_path: Path,
    arm_id: str,
) -> None:
    manifest = _manifest()
    config = _owned_config(tmp_path)
    root = tmp_path / "runs"
    result = _result(root, arm_id=arm_id)
    with (
        patch.object(command, "read_experiment_manifest_v1", return_value=manifest),
        patch.object(command, "_compose_experiment_config", return_value=object()),
        patch.object(command, "_own_config", return_value=config),
        patch.object(command, "_preflight_existing_run_v2") as preflight,
        patch.object(bootstrap_service, "_validate_cuda_determinism"),
        patch.object(
            bootstrap_service,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(command, "_validate_ambient_training_context"),
        patch.object(command, "capture_runtime_state", return_value=RUNTIME_SNAPSHOT),
        patch.object(command, "apply_runtime_policy"),
        patch.object(command, "_create_backbone", return_value=BACKBONE),
        patch.object(command, "_validate_runtime_backbone"),
        patch.object(command, "_validate_teacher_target_policy"),
        patch.object(
            command,
            "build_arm_teacher_forcing_model_v3",
            return_value=MODEL,
        ) as build_model,
        patch.object(command, "IrisTeacherOptimizer", return_value=OPTIMIZER),
        patch.object(command, "_validate_pristine_training_owner_v2"),
        patch.object(command, "_require_manifest_stable"),
        patch.object(command, "execute_experiment_v2", return_value=result) as execute,
        patch.object(command, "_validate_live_training_owner_v2"),
        patch.object(command, "_restore_after_success") as restore,
    ):
        observed = command.run_packaged_arm_experiment_v2(
            **_arm_arguments(tmp_path, arm_id=arm_id, root=root)
        )

    assert observed is result
    model_kwargs = build_model.call_args.kwargs
    assert model_kwargs["arm_id"] == arm_id
    assert model_kwargs["backbone"] is BACKBONE
    assert model_kwargs["cognitive_config"] == config.cognitive
    assert model_kwargs["workspace_config"] == (config.workspace if arm_id == "D" else None)
    runner_kwargs = execute.call_args.kwargs
    assert runner_kwargs["arm_id"] == arm_id
    assert runner_kwargs["model"] is MODEL
    assert runner_kwargs["optimizer"] is OPTIMIZER
    assert runner_kwargs["root"] == root
    preflight.assert_called_once()
    restore.assert_called_once_with(RUNTIME_SNAPSHOT)


@pytest.mark.parametrize(
    ("arm_id", "message"),
    [
        ("A", "no trainable teacher graph"),
        ("C", "teacher graph is blocked"),
    ],
)
def test_nonteacher_arms_fail_before_command_input_or_manifest_inspection(
    tmp_path: Path,
    arm_id: str,
    message: str,
) -> None:
    with (
        patch.object(command, "_own_command") as own,
        patch.object(command, "read_experiment_manifest_v1") as read_manifest,
        pytest.raises(IrisArmExecutionProfileError, match=message),
    ):
        command.run_packaged_arm_experiment_v2(**_arm_arguments(tmp_path, arm_id=arm_id))

    own.assert_not_called()
    read_manifest.assert_not_called()


def test_existing_run_cross_arm_conflict_is_detected_from_redacted_identity(
    tmp_path: Path,
) -> None:
    values = _arguments(tmp_path)
    owned = legacy_command._own_command(**values)
    path = owned.root / "experiments" / "v2" / owned.run_id
    path.mkdir(parents=True)
    conflicting = SimpleNamespace(
        run=SimpleNamespace(
            payload=SimpleNamespace(
                run_id=owned.run_id,
                arm_id="D",
            )
        )
    )
    manifest = _manifest()
    config = _owned_config(tmp_path)
    policy = legacy_command._execution_policy(owned)

    with (
        patch.object(command, "read_experiment_log_v2", return_value=conflicting),
        pytest.raises(ValueError, match="identity conflicts"),
    ):
        command._preflight_existing_run_v2(
            command=owned,
            profile=get_arm_execution_profile_v3("B"),
            arm_id="B",
            manifest=manifest,
            config=config,
            policy=policy,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arm_id": "D"}, "wrong arm or run path"),
        ({"latest_checkpoint_schema_version": 1}, "outside requested bounds"),
        ({"status": "step_limit_reached"}, "unbounded arm-labelled execution"),
    ],
)
def test_result_validation_rejects_cross_arm_checkpoint_and_status_mismatch(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    owned = legacy_command._own_command(**_arguments(tmp_path))
    result = replace(_result(owned.root, arm_id="B"), **cast(Any, change))

    with pytest.raises(ValueError, match=message):
        command._validate_execution_result_v2(
            result,
            command=owned,
            arm_id="B",
            manifest=_manifest(),
        )


def test_execution_failure_restores_captured_runtime_once(tmp_path: Path) -> None:
    manifest = _manifest()
    config = _owned_config(tmp_path)
    failure = RuntimeError("managed v2 failure")
    with (
        patch.object(command, "read_experiment_manifest_v1", return_value=manifest),
        patch.object(command, "_compose_experiment_config", return_value=object()),
        patch.object(command, "_own_config", return_value=config),
        patch.object(command, "_preflight_existing_run_v2"),
        patch.object(bootstrap_service, "_validate_cuda_determinism"),
        patch.object(
            bootstrap_service,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(command, "_validate_ambient_training_context"),
        patch.object(command, "capture_runtime_state", return_value=RUNTIME_SNAPSHOT),
        patch.object(command, "apply_runtime_policy"),
        patch.object(command, "_create_backbone", side_effect=failure),
        patch.object(command, "_restore_after_failure") as restore,
        pytest.raises(IrisExperimentCommandError, match="backbone loading"),
    ):
        command.run_packaged_arm_experiment_v2(**_arm_arguments(tmp_path, arm_id="B"))

    restore.assert_called_once()
    assert restore.call_args.args[0] is RUNTIME_SNAPSHOT
    restored_error = restore.call_args.args[1]
    assert isinstance(restored_error, IrisExperimentCommandError)
    assert restored_error.__cause__ is failure


def test_base_exception_restores_runtime_and_is_reraised_unchanged(tmp_path: Path) -> None:
    manifest = _manifest()
    config = _owned_config(tmp_path)
    interruption = KeyboardInterrupt("operator interruption")
    with (
        patch.object(command, "read_experiment_manifest_v1", return_value=manifest),
        patch.object(command, "_compose_experiment_config", return_value=object()),
        patch.object(command, "_own_config", return_value=config),
        patch.object(command, "_preflight_existing_run_v2"),
        patch.object(bootstrap_service, "_validate_cuda_determinism"),
        patch.object(
            bootstrap_service,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(command, "_validate_ambient_training_context"),
        patch.object(command, "capture_runtime_state", return_value=RUNTIME_SNAPSHOT),
        patch.object(command, "apply_runtime_policy", side_effect=interruption),
        patch.object(command, "_restore_after_failure") as restore,
        pytest.raises(KeyboardInterrupt) as captured,
    ):
        command.run_packaged_arm_experiment_v2(**_arm_arguments(tmp_path, arm_id="D"))

    assert captured.value is interruption
    restore.assert_called_once_with(RUNTIME_SNAPSHOT, interruption)


def test_public_profile_preflight_returns_exact_v3_profile() -> None:
    assert command.require_arm_experiment_command_profile_v2("B") == (
        get_arm_execution_profile_v3("B")
    )
