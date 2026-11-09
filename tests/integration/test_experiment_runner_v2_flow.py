"""Offline B/D managed execution, interruption, and exact replay tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
)
from iris.application.errors import IrisExperimentRunnerError
from iris.application.experiment_log_v2 import read_experiment_log_v2
from iris.application.experiment_runner_v2 import (
    IrisExperimentExecutionResultV2,
    execute_experiment_v2,
)
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.artifacts.experiment_log_v1 import ExperimentEpochExecutionPolicyV1
from iris.artifacts.experiment_log_v2 import ExperimentCheckpointEventV2
from iris.schemas import CognitiveConfig
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingInput,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
)
from tests.integration.test_experiment_iteration import (
    _build as _build_arm_d,
)
from tests.integration.test_experiment_iteration import (
    _manifest,
)
from tests.integration.test_teacher_forcing_sidecar import _cognitive_config

pytestmark = pytest.mark.integration

_SEED = 1907
_TERMINAL_TOKEN_ID = 15


@contextmanager
def _deterministic_runtime(seed: int = _SEED) -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(
                previous_enabled,
                warn_only=previous_warn_only,
            )


def _policy(
    *,
    epoch_count: int = 1,
    evaluate_test: bool = True,
) -> ExperimentEpochExecutionPolicyV1:
    return ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=epoch_count,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=evaluate_test,
        resume_policy="latest_checkpoint_plus_verified_replay",
    )


def _optimizer_config() -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        max_gradient_norm=1.0,
    )


def _arm_b_cognitive_config() -> CognitiveConfig:
    values = _cognitive_config().model_dump(mode="python")
    values["num_modules"] = 2
    return CognitiveConfig.model_validate(values)


def _fresh_owner(
    arm_id: IrisTeacherArmId,
) -> tuple[IrisArmTeacherModel, IrisTeacherOptimizer]:
    torch.manual_seed(_SEED)
    arm_d_model, _ = _build_arm_d()
    model: IrisArmTeacherModel
    if arm_id == "B":
        model = build_no_workspace_teacher_forcing_model(
            backbone=arm_d_model.backbone,
            cognitive_config=_arm_b_cognitive_config(),
        )
    else:
        model = arm_d_model
    return model, IrisTeacherOptimizer(model=model, config=_optimizer_config())


def _execute(
    *,
    root: Path,
    run_id: str,
    arm_id: IrisTeacherArmId,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    policy: ExperimentEpochExecutionPolicyV1 | None = None,
    maximum_new_training_steps: int | None = None,
) -> IrisExperimentExecutionResultV2:
    return execute_experiment_v2(
        run_id=run_id,
        arm_id=arm_id,
        manifest=_manifest(),
        terminal_token_id=_TERMINAL_TOKEN_ID,
        initialization_seed=_SEED,
        deterministic=True,
        execution_policy=policy or _policy(),
        model=model,
        optimizer=optimizer,
        root=root,
        maximum_new_training_steps=maximum_new_training_steps,
    )


@pytest.mark.parametrize(("arm_id", "checkpoint_version"), [("B", 2), ("D", 1)])
def test_fresh_runner_executes_complete_arm_labelled_epoch(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner(arm_id)
        result = _execute(
            root=tmp_path,
            run_id=f"managed-full-{arm_id.lower()}",
            arm_id=arm_id,
            model=model,
            optimizer=optimizer,
        )
        snapshot = read_experiment_log_v2(result.path)

        fresh_model, fresh_optimizer = _fresh_owner(arm_id)
        repeated = _execute(
            root=tmp_path,
            run_id=f"managed-full-{arm_id.lower()}",
            arm_id=arm_id,
            model=fresh_model,
            optimizer=fresh_optimizer,
        )

    assert result.arm_id == arm_id
    assert result.status == "complete"
    assert result.event_count == 5
    assert result.optimizer_completed_steps == 2
    assert result.completed_epochs == 1
    assert result.next_epoch_index is None
    assert result.next_training_ordinal is None
    assert result.latest_checkpoint_schema_version == checkpoint_version
    assert result.new_training_steps == 2
    assert result.replayed_training_steps == 0
    assert [event.payload.record.event_type for event in snapshot.events] == [
        "training_step",
        "training_step",
        "checkpoint",
        "evaluation",
        "evaluation",
    ]
    checkpoint_record = snapshot.events[2].payload.record
    assert type(checkpoint_record) is ExperimentCheckpointEventV2
    assert checkpoint_record.checkpoint.arm_id == arm_id
    assert checkpoint_record.checkpoint.checkpoint_schema_version == checkpoint_version
    assert (
        tmp_path
        / "teacher"
        / f"v{checkpoint_version}"
        / checkpoint_record.checkpoint.payload_sha256
    ).is_dir()
    assert model.training
    assert repeated.status == "complete"
    assert repeated.event_count == result.event_count
    assert repeated.new_training_steps == 0
    assert repeated.replayed_training_steps == 0
    assert read_experiment_log_v2(result.path).events == snapshot.events


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_mid_epoch_pause_resumes_with_exact_replay_and_matches_reference(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
) -> None:
    resumed_root = tmp_path / "resumed"
    reference_root = tmp_path / "reference"
    with _deterministic_runtime():
        interrupted_model, interrupted_optimizer = _fresh_owner(arm_id)
        interrupted = _execute(
            root=resumed_root,
            run_id=f"managed-resume-{arm_id.lower()}",
            arm_id=arm_id,
            model=interrupted_model,
            optimizer=interrupted_optimizer,
            maximum_new_training_steps=1,
        )

        resumed_model, resumed_optimizer = _fresh_owner(arm_id)
        resumed = _execute(
            root=resumed_root,
            run_id=f"managed-resume-{arm_id.lower()}",
            arm_id=arm_id,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )
        resumed_checkpoint = build_arm_teacher_checkpoint(
            arm_id=arm_id,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )

        reference_model, reference_optimizer = _fresh_owner(arm_id)
        reference = _execute(
            root=reference_root,
            run_id=f"managed-reference-{arm_id.lower()}",
            arm_id=arm_id,
            model=reference_model,
            optimizer=reference_optimizer,
        )
        reference_checkpoint = build_arm_teacher_checkpoint(
            arm_id=arm_id,
            model=reference_model,
            optimizer=reference_optimizer,
        )

    assert interrupted.status == "step_limit_reached"
    assert interrupted.event_count == 1
    assert interrupted.next_epoch_index == 1
    assert interrupted.next_training_ordinal == 2
    assert resumed.status == reference.status == "complete"
    assert resumed.replayed_training_steps == 1
    assert resumed.new_training_steps == 1
    assert resumed.event_count == reference.event_count == 5
    assert resumed_checkpoint == reference_checkpoint


@pytest.mark.parametrize(("requested_arm", "model_arm"), [("B", "D"), ("D", "B")])
def test_cross_arm_owner_is_rejected_before_filesystem_mutation(
    tmp_path: Path,
    requested_arm: IrisTeacherArmId,
    model_arm: IrisTeacherArmId,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner(model_arm)
        with pytest.raises(IrisExperimentRunnerError, match="input validation"):
            _execute(
                root=tmp_path,
                run_id="cross-arm-owner",
                arm_id=requested_arm,
                model=model,
                optimizer=optimizer,
            )

    assert not (tmp_path / "experiments").exists()
    assert not (tmp_path / "teacher").exists()


def test_existing_run_rejects_stepped_owner_without_appending(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner("B")
        partial = _execute(
            root=tmp_path,
            run_id="managed-stepped-owner-v2",
            arm_id="B",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        with pytest.raises(IrisExperimentRunnerError, match="pristine resume preflight"):
            _execute(
                root=tmp_path,
                run_id="managed-stepped-owner-v2",
                arm_id="B",
                model=model,
                optimizer=optimizer,
            )

    assert read_experiment_log_v2(partial.path).event_count == 1


def test_rng_consuming_training_fails_closed_without_event(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner("D")
        original_step = optimizer.step

        def consume_rng(
            example: IrisTeacherForcingInput,
        ) -> IrisTeacherOptimizationOutput:
            output = original_step(example)
            _ = torch.rand(1)
            return output

        with (
            patch.object(IrisTeacherOptimizer, "step", side_effect=consume_rng),
            pytest.raises(
                IrisExperimentRunnerError,
                match="next manifest-ordered training step",
            ),
        ):
            _execute(
                root=tmp_path,
                run_id="managed-rng-change-v2",
                arm_id="D",
                model=model,
                optimizer=optimizer,
            )

    path = tmp_path / "experiments" / "v2" / "managed-rng-change-v2"
    assert read_experiment_log_v2(path).event_count == 0
