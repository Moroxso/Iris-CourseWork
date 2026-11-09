"""Deterministic epoch execution with checkpoint-boundary verified replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

from iris.application.errors import IrisExperimentRunnerError
from iris.application.experiment_execution import (
    IrisExperimentEpochCursorV1,
    analyze_experiment_epoch_progress_v1,
    extend_experiment_epoch_progress_v1,
)
from iris.application.experiment_log import (
    IrisExperimentLogSnapshotV1,
    IrisExperimentLogWriterV1,
    create_experiment_log_v1,
    inspect_experiment_training_snapshot_v1,
    read_experiment_log_v1,
    record_experiment_checkpoint_v1,
    record_experiment_evaluation_v1,
    record_experiment_training_step_v1,
)
from iris.application.experiment_manifest import (
    IrisExperimentExampleIterator,
    IrisPreparedExperimentExample,
    iterate_experiment_split_v1,
    validate_experiment_manifest_v1,
)
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    build_teacher_checkpoint_v1,
    read_teacher_checkpoint_v1,
    restore_teacher_checkpoint_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointReferenceV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
    validate_experiment_run_id_v1,
)
from iris.artifacts.experiment_v1 import ExperimentSplitV1, IrisExperimentManifestV1
from iris.training import (
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationAccumulator,
    IrisTokenWeightedEvaluationSummary,
)


@dataclass(frozen=True, slots=True)
class IrisExperimentExecutionResult:
    """Observed terminal or bounded-progress state of one managed invocation."""

    path: Path
    status: Literal["complete", "step_limit_reached"]
    run_payload_sha256: str
    configuration_sha256: str
    event_count: int
    optimizer_completed_steps: int
    completed_epochs: int
    next_epoch_index: int | None
    next_training_ordinal: int | None
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


def _torch_rng_snapshot(model: IrisTeacherForcingModel) -> _TorchRngSnapshot:
    device = model.backbone.device
    if device.type == "cpu":
        device_state = None
    elif device.type == "cuda":
        device_state = _state_bytes(torch.cuda.get_rng_state(device))
    else:
        message = f"deterministic epoch RNG observation does not support {device.type!r}"
        raise ValueError(message)
    return _TorchRngSnapshot(
        cpu_state=_state_bytes(torch.random.get_rng_state()),
        device_type=cast(Literal["cpu", "cuda"], device.type),
        device_state=device_state,
    )


def _require_rng_unchanged(
    before: _TorchRngSnapshot,
    model: IrisTeacherForcingModel,
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


def _run_path(*, root: Path, run_id: str) -> Path:
    if not isinstance(root, Path) or not root.name:
        message = "experiment runner requires an explicit root Path"
        raise TypeError(message)
    owned_run_id = validate_experiment_run_id_v1(run_id)
    return root / "experiments" / "v1" / owned_run_id


def _validate_request_identity(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
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
        manifest.payload.dataset_id != reference.dataset_id
        or manifest.payload_sha256 != reference.payload_sha256
        or manifest.payload.split_counts != reference.split_counts
        or configuration.target_preparation.terminal_token_id != terminal_token_id
        or configuration.runtime_policy.initialization_seed != initialization_seed
        or configuration.runtime_policy.deterministic is not deterministic
        or configuration.execution_policy != execution_policy
    ):
        message = "requested experiment identity conflicts with the installed run"
        raise ValueError(message)


def _validate_checkpoint_reference(
    checkpoint: IrisTeacherCheckpointBundleV1,
    reference: ExperimentCheckpointReferenceV1,
) -> None:
    manifest = getattr(checkpoint, "manifest", None)
    if (
        manifest is None
        or manifest.sha256 != reference.payload_sha256
        or manifest.payload.state_file.sha256 != reference.state_file_sha256
        or manifest.payload.model.sha256 != reference.sidecar_state_sha256
        or manifest.payload.optimizer.completed_steps != reference.completed_steps
    ):
        message = "referenced checkpoint does not match the experiment journal"
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


def _restore_and_replay(
    *,
    root: Path,
    snapshot: IrisExperimentLogSnapshotV1,
    cursor: IrisExperimentEpochCursorV1,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    writer: IrisExperimentLogWriterV1,
) -> _ReplayState:
    reference = cursor.latest_checkpoint
    checkpoint_path = root / "teacher" / "v1" / reference.payload_sha256
    checkpoint = read_teacher_checkpoint_v1(checkpoint_path)
    _validate_checkpoint_reference(checkpoint, reference)
    restored = restore_teacher_checkpoint_v1(
        checkpoint,
        model=model,
        optimizer=optimizer,
    )
    if (
        restored.payload_sha256 != reference.payload_sha256
        or restored.sidecar_state_sha256 != reference.sidecar_state_sha256
        or restored.completed_steps != reference.completed_steps
    ):
        message = "restored checkpoint state disagrees with its journal reference"
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
            message = "recorded training tail failed exact deterministic replay"
            raise ValueError(message)
        replayed += 1

    _require_live_cursor(
        snapshot=snapshot,
        cursor=cursor,
        model=model,
        optimizer=optimizer,
        writer=writer,
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


def _require_live_cursor(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    cursor: IrisExperimentEpochCursorV1,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    writer: IrisExperimentLogWriterV1,
) -> None:
    current = writer.refresh()
    if (
        current.event_count != snapshot.event_count
        or current.chain_head_sha256 != snapshot.chain_head_sha256
    ):
        message = "experiment journal changed while validating the live owner"
        raise ValueError(message)
    live = inspect_experiment_training_snapshot_v1(
        snapshot=current,
        model=model,
        optimizer=optimizer,
    )
    if live.configuration_sha256 != current.run.payload.configuration.sha256:
        message = "experiment journal configuration changed while validating the live owner"
        raise ValueError(message)
    if (
        live.optimizer_completed_steps != cursor.optimizer_completed_steps
        or live.sidecar_state_sha256 != cursor.sidecar_state_sha256
    ):
        message = "live teacher state does not match the experiment chain head"
        raise ValueError(message)


def _evaluate_split(
    *,
    manifest: IrisExperimentManifestV1,
    split: ExperimentSplitV1,
    terminal_token_id: int,
    model: IrisTeacherForcingModel,
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
    snapshot: IrisExperimentLogSnapshotV1,
    cursor: IrisExperimentEpochCursorV1,
    status: Literal["complete", "step_limit_reached"],
    replayed_training_steps: int,
    new_training_steps: int,
) -> IrisExperimentExecutionResult:
    return IrisExperimentExecutionResult(
        path=snapshot.path,
        status=status,
        run_payload_sha256=snapshot.run.payload_sha256,
        configuration_sha256=snapshot.run.payload.configuration.sha256,
        event_count=snapshot.event_count,
        optimizer_completed_steps=cursor.optimizer_completed_steps,
        completed_epochs=cursor.completed_epochs,
        next_epoch_index=cursor.epoch_index,
        next_training_ordinal=cursor.next_training_ordinal,
        latest_checkpoint_payload_sha256=cursor.latest_checkpoint.payload_sha256,
        replayed_training_steps=replayed_training_steps,
        new_training_steps=new_training_steps,
    )


def execute_experiment_v1(
    *,
    run_id: str,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentEpochExecutionPolicyV1,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    root: Path,
    maximum_new_training_steps: int | None = None,
) -> IrisExperimentExecutionResult:
    """Create or exactly resume one fixed-order epoch experiment to a safe boundary."""
    stage = "input validation"
    try:
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
        if not isinstance(model, IrisTeacherForcingModel):
            message = "experiment runner requires an IrisTeacherForcingModel"
            raise TypeError(message)
        if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
            message = "experiment runner requires the exact optimizer owner for its model"
            raise TypeError(message)
        if not model.training:
            message = "experiment runner requires the teacher model in training mode"
            raise ValueError(message)
        if torch.is_deterministic_algorithms_warn_only_enabled():
            message = "deterministic epoch execution forbids warn-only algorithms"
            raise ValueError(message)
        stage = "run creation or loading"
        run_preexisted = path.exists()
        if run_preexisted:
            snapshot = read_experiment_log_v1(path)
            reused_existing = True
        else:
            created = create_experiment_log_v1(
                run_id=run_id,
                manifest=owned_manifest,
                terminal_token_id=terminal_token_id,
                initialization_seed=initialization_seed,
                deterministic=deterministic,
                execution_policy=policy,
                model=model,
                optimizer=optimizer,
                root=root,
            )
            snapshot = read_experiment_log_v1(created.path)
            reused_existing = created.reused_existing or snapshot.event_count > 0
        _validate_request_identity(
            snapshot=snapshot,
            manifest=owned_manifest,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=policy,
        )
        progress = analyze_experiment_epoch_progress_v1(
            run=snapshot.run,
            events=snapshot.events,
            manifest=owned_manifest,
        )
        cursor = progress.cursor
        writer = IrisExperimentLogWriterV1(snapshot)

        replay_state = _ReplayState(
            replayed_training_steps=0,
            train_iterator=None,
            active_epoch=None,
            iterator_next_ordinal=1,
        )
        if reused_existing:
            stage = "pristine resume preflight"
            live = inspect_experiment_training_snapshot_v1(
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
                writer=writer,
            )
        else:
            _require_live_cursor(
                snapshot=snapshot,
                cursor=cursor,
                model=model,
                optimizer=optimizer,
                writer=writer,
            )

        new_training_steps = 0
        active_epoch = replay_state.active_epoch
        train_iterator = replay_state.train_iterator
        iterator_next_ordinal = replay_state.iterator_next_ordinal
        while True:
            stage = "managed journal validation"
            snapshot = writer.refresh()
            progress = extend_experiment_epoch_progress_v1(
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
                writer=writer,
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
                record_experiment_training_step_v1(
                    path=snapshot.path,
                    manifest=owned_manifest,
                    prepared=prepared,
                    output=output,
                    model=model,
                    optimizer=optimizer,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    writer=writer,
                    epoch_progress=progress,
                )
                new_training_steps += 1
            elif cursor.phase == "checkpoint":
                stage = "epoch checkpoint"
                before_rng = _torch_rng_snapshot(model)
                checkpoint = build_teacher_checkpoint_v1(model=model, optimizer=optimizer)
                _require_rng_unchanged(before_rng, model, operation="checkpoint capture")
                record_experiment_checkpoint_v1(
                    path=snapshot.path,
                    checkpoint=checkpoint,
                    manifest=owned_manifest,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    writer=writer,
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
                record_experiment_evaluation_v1(
                    path=snapshot.path,
                    manifest=owned_manifest,
                    split=cursor.phase,
                    summary=summary,
                    model=model,
                    optimizer=optimizer,
                    expected_predecessor_sha256=snapshot.chain_head_sha256,
                    writer=writer,
                    epoch_progress=progress,
                )
            else:
                message = f"unsupported managed experiment phase: {cursor.phase}"
                raise ValueError(message)
    except IrisExperimentRunnerError:
        raise
    except Exception as error:
        message = f"Iris experiment runner failed during {stage}"
        raise IrisExperimentRunnerError(message) from error


__all__ = [
    "IrisExperimentExecutionResult",
    "execute_experiment_v1",
]
