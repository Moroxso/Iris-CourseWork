"""Deterministic profile-v4 B/C/D epoch execution with exact replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
    read_arm_teacher_checkpoint,
    restore_arm_teacher_checkpoint,
)
from iris.application.errors import IrisExperimentRunnerError
from iris.application.experiment_execution_v3 import (
    IrisExperimentEpochCursorV3,
    analyze_experiment_epoch_progress_v3,
    extend_experiment_epoch_progress_v3,
)
from iris.application.experiment_log_v3 import (
    IrisExperimentLogSnapshotV3,
    checkpoint_identity_from_reference_v3,
    create_experiment_log_v3,
    inspect_experiment_training_snapshot_v3,
    read_experiment_log_v3,
    record_experiment_checkpoint_v3,
    record_experiment_evaluation_v3,
    record_experiment_training_step_v3,
)
from iris.application.experiment_manifest import (
    IrisExperimentExampleIterator,
    IrisPreparedExperimentExample,
    iterate_experiment_split_v1,
    validate_experiment_manifest_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
    validate_experiment_run_id_v1,
)
from iris.artifacts.experiment_log_v3 import (
    ExperimentArmIdV3,
    ExperimentCheckpointReferenceV3,
)
from iris.artifacts.experiment_v1 import ExperimentSplitV1, IrisExperimentManifestV1
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationAccumulator,
    IrisTokenWeightedEvaluationSummary,
)


@dataclass(frozen=True, slots=True)
class IrisExperimentExecutionResultV3:
    """Observed terminal or bounded progress of one arm-labelled invocation."""

    arm_id: ExperimentArmIdV3
    path: Path
    status: Literal["complete", "step_limit_reached"]
    run_payload_sha256: str
    configuration_sha256: str
    event_count: int
    optimizer_completed_steps: int
    completed_epochs: int
    next_epoch_index: int | None
    next_training_ordinal: int | None
    latest_checkpoint_schema_version: Literal[1, 2, 3]
    latest_checkpoint_payload_sha256: str
    replayed_training_steps: int
    new_training_steps: int


@dataclass(frozen=True, slots=True)
class _TorchRngSnapshot:
    cpu_state: bytes
    device_type: Literal["cpu", "cuda"]
    device_state: bytes | None


@dataclass(frozen=True, slots=True)
class _ReplayState:
    replayed_training_steps: int
    train_iterator: IrisExperimentExampleIterator | None
    active_epoch: int | None
    iterator_next_ordinal: int


def _state_bytes(state: torch.Tensor) -> bytes:
    if type(state) is not torch.Tensor or state.dtype != torch.uint8 or state.ndim != 1:
        message = "torch RNG returned an invalid state tensor"
        raise TypeError(message)
    return bytes(state.detach().to(device="cpu").tolist())


def _torch_rng_snapshot(model: IrisArmTeacherModel) -> _TorchRngSnapshot:
    device = model.backbone.device
    if device.type == "cpu":
        device_state = None
    elif device.type == "cuda":
        device_state = _state_bytes(torch.cuda.get_rng_state(device))
    else:
        message = f"deterministic arm-labelled RNG observation does not support {device.type!r}"
        raise ValueError(message)
    return _TorchRngSnapshot(
        cpu_state=_state_bytes(torch.random.get_rng_state()),
        device_type=cast(Literal["cpu", "cuda"], device.type),
        device_state=device_state,
    )


def _require_rng_unchanged(
    before: _TorchRngSnapshot,
    model: IrisArmTeacherModel,
    *,
    operation: str,
) -> None:
    if _torch_rng_snapshot(model) != before:
        message = f"{operation} consumed global torch RNG state"
        raise ValueError(message)


def _owned_epoch_policy(
    policy: ExperimentEpochExecutionPolicyV1,
) -> ExperimentEpochExecutionPolicyV1:
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "execution_policy must be exactly ExperimentEpochExecutionPolicyV1"
        raise TypeError(message)
    return ExperimentEpochExecutionPolicyV1.model_validate(policy.model_dump(mode="python"))


def _require_owner(
    arm_id: IrisTeacherArmId,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> None:
    if type(arm_id) is not str or arm_id not in {"B", "C", "D"}:
        message = "experiment runner arm_id must be exactly B, C, or D"
        raise ValueError(message)
    expected_models: dict[IrisTeacherArmId, type[torch.nn.Module]] = {
        "B": IrisNoWorkspaceTeacherForcingModel,
        "C": IrisSharedBufferTeacherForcingModel,
        "D": IrisTeacherForcingModel,
    }
    expected_model = expected_models[arm_id]
    if type(model) is not expected_model:
        message = f"arm {arm_id} runner requires exact model type {expected_model.__name__}"
        raise TypeError(message)
    if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
        message = "experiment runner requires the exact optimizer owner for its model"
        raise TypeError(message)


def _run_path(*, root: Path, run_id: str) -> Path:
    if not isinstance(root, Path) or not root.name:
        message = "experiment runner requires an explicit root Path"
        raise TypeError(message)
    owned_run_id = validate_experiment_run_id_v1(run_id)
    return root / "experiments" / "v3" / owned_run_id


def _validate_request_identity(
    *,
    snapshot: IrisExperimentLogSnapshotV3,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentEpochExecutionPolicyV1,
) -> None:
    run_payload = snapshot.run.payload
    reference = run_payload.manifest
    configuration = run_payload.configuration
    if (
        run_payload.arm_id != arm_id
        or configuration.model.arm_id != arm_id
        or run_payload.initial_checkpoint.arm_id != arm_id
        or manifest.payload.dataset_id != reference.dataset_id
        or manifest.payload_sha256 != reference.payload_sha256
        or manifest.payload.split_counts != reference.split_counts
        or configuration.target_preparation.terminal_token_id != terminal_token_id
        or configuration.runtime_policy.initialization_seed != initialization_seed
        or configuration.runtime_policy.deterministic is not deterministic
        or configuration.execution_policy != execution_policy
    ):
        message = "requested arm-labelled experiment identity conflicts with the installed run"
        raise ValueError(message)


def _validate_checkpoint_reference(
    checkpoint: IrisArmTeacherCheckpointBundle,
    reference: ExperimentCheckpointReferenceV3,
) -> None:
    if checkpoint.identity != checkpoint_identity_from_reference_v3(reference):
        message = "referenced arm checkpoint does not match the experiment journal"
        raise ValueError(message)


def _training_record(
    prepared: IrisPreparedExperimentExample,
    output: IrisTeacherOptimizationOutput,
) -> ExperimentTrainingStepMetricV1:
    return ExperimentTrainingStepMetricV1(
        event_type="training_step",
        split="train",
        example_id=prepared.example_id,
        split_ordinal=prepared.split_ordinal,
        split_example_count=prepared.split_example_count,
        step_index=output.step_index,
        loss=output.loss,
        supervised_token_count=output.supervised_token_count,
        gradient_l2_norm=output.gradient_l2_norm,
        gradients_clipped=output.gradients_clipped,
    )


def _require_live_cursor(
    *,
    snapshot: IrisExperimentLogSnapshotV3,
    cursor: IrisExperimentEpochCursorV3,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> None:
    current = read_experiment_log_v3(snapshot.path)
    if (
        current.event_count != snapshot.event_count
        or current.chain_head_sha256 != snapshot.chain_head_sha256
    ):
        message = "experiment journal changed while validating the live owner"
        raise ValueError(message)
    live = inspect_experiment_training_snapshot_v3(
        snapshot=current,
        model=model,
        optimizer=optimizer,
    )
    if (
        live.arm_id != current.arm_id
        or live.configuration_sha256 != current.run.payload.configuration.sha256
    ):
        message = "experiment journal identity changed while validating the live owner"
        raise ValueError(message)
    if (
        live.optimizer_completed_steps != cursor.optimizer_completed_steps
        or live.sidecar_state_sha256 != cursor.sidecar_state_sha256
    ):
        message = "live teacher state does not match the experiment chain head"
        raise ValueError(message)


def _restore_and_replay(
    *,
    root: Path,
    snapshot: IrisExperimentLogSnapshotV3,
    cursor: IrisExperimentEpochCursorV3,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> _ReplayState:
    reference = cursor.latest_checkpoint
    identity = checkpoint_identity_from_reference_v3(reference)
    if identity.arm_id != snapshot.arm_id:
        message = "latest checkpoint arm disagrees with the experiment run"
        raise ValueError(message)
    checkpoint = read_arm_teacher_checkpoint(identity, root=root)
    _validate_checkpoint_reference(checkpoint, reference)
    restored = restore_arm_teacher_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
    )
    if (
        restored.payload_sha256 != reference.payload_sha256
        or restored.sidecar_state_sha256 != reference.sidecar_state_sha256
        or restored.completed_steps != reference.completed_steps
    ):
        message = "restored arm checkpoint state disagrees with its journal reference"
        raise ValueError(message)

    iterator = None
    replayed = 0
    for event in snapshot.events[cursor.latest_checkpoint_event_count :]:
        record = event.payload.record
        if type(record) is not ExperimentTrainingStepMetricV1:
            continue
        before_rng = _torch_rng_snapshot(model)
        if iterator is None:
            iterator = iterate_experiment_split_v1(
                manifest=manifest,
                split="train",
                backbone=model.backbone,
                terminal_token_id=terminal_token_id,
            )
        prepared = next(iterator)
        output = optimizer.step(prepared.teacher_input)
        provenance = model.checkpoint_provenance
        _require_rng_unchanged(before_rng, model, operation="training replay")
        if (
            _training_record(prepared, output) != record
            or optimizer.completed_steps != event.payload.optimizer_completed_steps
            or provenance.sidecar_state_sha256 != event.payload.sidecar_state_sha256
        ):
            message = "recorded arm training tail failed exact deterministic replay"
            raise ValueError(message)
        replayed += 1

    _require_live_cursor(
        snapshot=snapshot,
        cursor=cursor,
        model=model,
        optimizer=optimizer,
    )
    positioned_iterator = iterator if cursor.phase == "training" and replayed else None
    positioned_epoch = cursor.epoch_index if positioned_iterator is not None else None
    positioned_ordinal = cursor.next_training_ordinal if positioned_iterator is not None else 1
    if type(positioned_ordinal) is not int:
        message = "replayed training cursor is missing its next ordinal"
        raise TypeError(message)
    return _ReplayState(
        replayed_training_steps=replayed,
        train_iterator=positioned_iterator,
        active_epoch=positioned_epoch,
        iterator_next_ordinal=positioned_ordinal,
    )


def _evaluate_split(
    *,
    manifest: IrisExperimentManifestV1,
    split: ExperimentSplitV1,
    terminal_token_id: int,
    model: IrisArmTeacherModel,
) -> IrisTokenWeightedEvaluationSummary:
    before_rng = _torch_rng_snapshot(model)
    model.eval()
    try:
        accumulator = IrisTokenWeightedEvaluationAccumulator()
        for prepared in iterate_experiment_split_v1(
            manifest=manifest,
            split=split,
            backbone=model.backbone,
            terminal_token_id=terminal_token_id,
        ):
            accumulator.update(model.evaluation_forward(prepared.teacher_input))
        summary = accumulator.summary()
    finally:
        model.train()
    _require_rng_unchanged(before_rng, model, operation=f"{split} evaluation")
    return summary


def _execution_result(
    *,
    snapshot: IrisExperimentLogSnapshotV3,
    cursor: IrisExperimentEpochCursorV3,
    status: Literal["complete", "step_limit_reached"],
    replayed_training_steps: int,
    new_training_steps: int,
) -> IrisExperimentExecutionResultV3:
    reference = cursor.latest_checkpoint
    return IrisExperimentExecutionResultV3(
        arm_id=snapshot.arm_id,
        path=snapshot.path,
        status=status,
        run_payload_sha256=snapshot.run.payload_sha256,
        configuration_sha256=snapshot.run.payload.configuration.sha256,
        event_count=snapshot.event_count,
        optimizer_completed_steps=cursor.optimizer_completed_steps,
        completed_epochs=cursor.completed_epochs,
        next_epoch_index=cursor.epoch_index,
        next_training_ordinal=cursor.next_training_ordinal,
        latest_checkpoint_schema_version=reference.checkpoint_schema_version,
        latest_checkpoint_payload_sha256=reference.payload_sha256,
        replayed_training_steps=replayed_training_steps,
        new_training_steps=new_training_steps,
    )


def execute_experiment_v3(
    *,
    run_id: str,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentEpochExecutionPolicyV1,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    root: Path,
    maximum_new_training_steps: int | None = None,
) -> IrisExperimentExecutionResultV3:
    """Create or exactly resume one B/C/D fixed-order epoch experiment."""
    stage = "input validation"
    try:
        _require_owner(arm_id, model, optimizer)
        path = _run_path(root=root, run_id=run_id)
        if type(terminal_token_id) is not int or type(initialization_seed) is not int:
            message = "terminal_token_id and initialization_seed must be exact integers"
            raise TypeError(message)
        if type(deterministic) is not bool or not deterministic:
            message = "deterministic epoch execution requires deterministic=True"
            raise ValueError(message)
        if maximum_new_training_steps is not None and (
            type(maximum_new_training_steps) is not int or maximum_new_training_steps < 1
        ):
            message = "maximum_new_training_steps must be an exact positive integer"
            raise ValueError(message)
        policy = _owned_epoch_policy(execution_policy)
        owned_manifest = validate_experiment_manifest_v1(manifest)
        if not model.training:
            message = "experiment runner requires the teacher model in training mode"
            raise ValueError(message)
        if torch.is_deterministic_algorithms_warn_only_enabled():
            message = "deterministic epoch execution forbids warn-only algorithms"
            raise ValueError(message)

        stage = "run creation or loading"
        run_preexisted = path.exists()
        if run_preexisted:
            snapshot = read_experiment_log_v3(path)
            reused_existing = True
        else:
            created = create_experiment_log_v3(
                run_id=run_id,
                arm_id=arm_id,
                manifest=owned_manifest,
                terminal_token_id=terminal_token_id,
                initialization_seed=initialization_seed,
                deterministic=deterministic,
                execution_policy=policy,
                model=model,
                optimizer=optimizer,
                root=root,
            )
            snapshot = read_experiment_log_v3(created.path)
            reused_existing = created.reused_existing or snapshot.event_count > 0
        _validate_request_identity(
            snapshot=snapshot,
            arm_id=arm_id,
            manifest=owned_manifest,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=policy,
        )
        progress = analyze_experiment_epoch_progress_v3(
            run=snapshot.run,
            events=snapshot.events,
            manifest=owned_manifest,
        )
        cursor = progress.cursor

        replay_state = _ReplayState(
            replayed_training_steps=0,
            train_iterator=None,
            active_epoch=None,
            iterator_next_ordinal=1,
        )
        if reused_existing:
            stage = "pristine resume preflight"
            live = inspect_experiment_training_snapshot_v3(
                snapshot=snapshot,
                model=model,
                optimizer=optimizer,
            )
            if live.optimizer_completed_steps != 0:
                message = "resuming an existing experiment requires a pristine optimizer owner"
                raise ValueError(message)
            stage = "checkpoint restore and training-tail replay"
            replay_state = _restore_and_replay(
                root=root,
                snapshot=snapshot,
                cursor=cursor,
                manifest=owned_manifest,
                terminal_token_id=terminal_token_id,
                model=model,
                optimizer=optimizer,
            )
        else:
            _require_live_cursor(
                snapshot=snapshot,
                cursor=cursor,
                model=model,
                optimizer=optimizer,
            )

        new_training_steps = 0
        active_epoch = replay_state.active_epoch
        train_iterator = replay_state.train_iterator
        iterator_next_ordinal = replay_state.iterator_next_ordinal
        while True:
            stage = "managed journal validation"
            snapshot = read_experiment_log_v3(path)
            progress = extend_experiment_epoch_progress_v3(
                progress=progress,
                run=snapshot.run,
                events=snapshot.events,
            )
            cursor = progress.cursor
            _require_live_cursor(
                snapshot=snapshot,
                cursor=cursor,
                model=model,
                optimizer=optimizer,
            )
            if cursor.phase == "complete":
                return _execution_result(
                    snapshot=snapshot,
                    cursor=cursor,
                    status="complete",
                    replayed_training_steps=replay_state.replayed_training_steps,
                    new_training_steps=new_training_steps,
                )
            if cursor.phase == "training":
                requested_epoch = cursor.epoch_index
                requested_ordinal = cursor.next_training_ordinal
                if type(requested_epoch) is not int or type(requested_ordinal) is not int:
                    message = "training cursor is missing its epoch or ordinal"
                    raise TypeError(message)
                if (
                    maximum_new_training_steps is not None
                    and new_training_steps >= maximum_new_training_steps
                ):
                    return _execution_result(
                        snapshot=snapshot,
                        cursor=cursor,
                        status="step_limit_reached",
                        replayed_training_steps=replay_state.replayed_training_steps,
                        new_training_steps=new_training_steps,
                    )
                stage = "next manifest-ordered training step"
                before_rng = _torch_rng_snapshot(model)
                if active_epoch != requested_epoch or iterator_next_ordinal != requested_ordinal:
                    train_iterator = iterate_experiment_split_v1(
                        manifest=owned_manifest,
                        split="train",
                        backbone=model.backbone,
                        terminal_token_id=terminal_token_id,
                    )
                    iterator_next_ordinal = 1
                    while iterator_next_ordinal < requested_ordinal:
                        next(train_iterator)
                        iterator_next_ordinal += 1
                    active_epoch = requested_epoch
                if train_iterator is None:
                    message = "training iterator was not initialized"
                    raise RuntimeError(message)
                prepared = next(train_iterator)
                iterator_next_ordinal += 1
                output = optimizer.step(prepared.teacher_input)
                _require_rng_unchanged(before_rng, model, operation="training step")
                record_experiment_training_step_v3(
                    path=snapshot.path,
                    manifest=owned_manifest,
                    prepared=prepared,
                    output=output,
                    model=model,
                    optimizer=optimizer,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    epoch_progress=progress,
                )
                new_training_steps += 1
            elif cursor.phase == "checkpoint":
                stage = "epoch checkpoint"
                before_rng = _torch_rng_snapshot(model)
                checkpoint = build_arm_teacher_checkpoint(
                    arm_id=arm_id,
                    model=model,
                    optimizer=optimizer,
                )
                _require_rng_unchanged(before_rng, model, operation="checkpoint capture")
                record_experiment_checkpoint_v3(
                    path=snapshot.path,
                    checkpoint=checkpoint,
                    manifest=owned_manifest,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    epoch_progress=progress,
                )
            elif cursor.phase in {"validation", "test"}:
                stage = f"{cursor.phase} evaluation"
                summary = _evaluate_split(
                    manifest=owned_manifest,
                    split=cursor.phase,
                    terminal_token_id=terminal_token_id,
                    model=model,
                )
                record_experiment_evaluation_v3(
                    path=snapshot.path,
                    manifest=owned_manifest,
                    split=cursor.phase,
                    summary=summary,
                    model=model,
                    optimizer=optimizer,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    epoch_progress=progress,
                )
            else:
                message = f"unsupported managed experiment phase: {cursor.phase}"
                raise ValueError(message)
    except IrisExperimentRunnerError:
        raise
    except Exception as error:
        message = f"arm-labelled experiment runner failed during {stage}"
        raise IrisExperimentRunnerError(message) from error


__all__ = [
    "IrisExperimentExecutionResultV3",
    "execute_experiment_v3",
]

