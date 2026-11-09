"""Pure state-machine validation for runner-managed experiment journals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from iris.application.experiment_manifest import validate_experiment_manifest_v1
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentCheckpointReferenceV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentEventRecordV1,
    ExperimentTrainingStepMetricV1,
    IrisExperimentEventV1,
    IrisExperimentRunV1,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1

ExperimentEpochPhaseV1 = Literal[
    "training",
    "checkpoint",
    "validation",
    "test",
    "complete",
]


@dataclass(frozen=True, slots=True)
class IrisExperimentEpochCursorV1:
    """Next legal transition and latest replay boundary for one valid prefix."""

    phase: ExperimentEpochPhaseV1
    completed_epochs: int
    epoch_index: int | None
    next_training_ordinal: int | None
    next_training_example_id: str | None
    optimizer_completed_steps: int
    sidecar_state_sha256: str
    latest_checkpoint: ExperimentCheckpointReferenceV1
    latest_checkpoint_event_count: int


@dataclass(frozen=True, slots=True)
class IrisExperimentEpochProgressV1:
    """Hash-bound managed cursor for incrementally verified event prefixes."""

    run_payload_sha256: str
    configuration_sha256: str
    manifest_payload_sha256: str
    verified_event_count: int
    chain_head_sha256: str
    cursor: IrisExperimentEpochCursorV1
    _train_example_ids: tuple[str, ...] = field(repr=False)


def _manifest_matches_run(
    manifest: IrisExperimentManifestV1,
    run: IrisExperimentRunV1,
) -> IrisExperimentManifestV1:
    owned = validate_experiment_manifest_v1(manifest)
    reference = run.payload.manifest
    if (
        owned.payload.dataset_id != reference.dataset_id
        or owned.payload_sha256 != reference.payload_sha256
        or owned.payload.split_counts != reference.split_counts
    ):
        message = "epoch state machine manifest does not match the experiment run"
        raise ValueError(message)
    return owned


def _validated_epoch_inputs(
    *,
    run: IrisExperimentRunV1,
    events: tuple[IrisExperimentEventV1, ...],
    manifest: IrisExperimentManifestV1,
) -> tuple[ExperimentEpochExecutionPolicyV1, tuple[str, ...]]:
    if type(run) is not IrisExperimentRunV1:
        message = "epoch state machine requires exactly IrisExperimentRunV1"
        raise TypeError(message)
    if type(events) is not tuple or any(
        type(event) is not IrisExperimentEventV1 for event in events
    ):
        message = "epoch state machine requires an exact event tuple"
        raise TypeError(message)
    policy = run.payload.configuration.execution_policy
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "epoch state machine requires deterministic_epochs execution policy"
        raise TypeError(message)
    owned_manifest = _manifest_matches_run(manifest, run)
    train_ids = tuple(
        example.example_id
        for example in owned_manifest.payload.examples
        if example.split == "train"
    )
    return policy, train_ids


def _initial_epoch_cursor(
    *,
    run: IrisExperimentRunV1,
    train_ids: tuple[str, ...],
) -> IrisExperimentEpochCursorV1:
    initial = run.payload.initial_checkpoint
    return IrisExperimentEpochCursorV1(
        phase="training",
        completed_epochs=0,
        epoch_index=1,
        next_training_ordinal=1,
        next_training_example_id=train_ids[0],
        optimizer_completed_steps=initial.completed_steps,
        sidecar_state_sha256=initial.sidecar_state_sha256,
        latest_checkpoint=initial,
        latest_checkpoint_event_count=0,
    )


def _advance_epoch_cursor(
    *,
    run: IrisExperimentRunV1,
    policy: ExperimentEpochExecutionPolicyV1,
    train_ids: tuple[str, ...],
    cursor: IrisExperimentEpochCursorV1,
    event: IrisExperimentEventV1,
    event_count: int,
) -> IrisExperimentEpochCursorV1:
    payload = event.payload
    record = payload.record
    train_count = len(train_ids)
    phase = cursor.phase
    completed_epochs = cursor.completed_epochs
    active_epoch = cursor.epoch_index
    next_ordinal = cursor.next_training_ordinal
    next_example_id = cursor.next_training_example_id
    latest_checkpoint = cursor.latest_checkpoint
    latest_checkpoint_event_count = cursor.latest_checkpoint_event_count
    initial_steps = run.payload.initial_checkpoint.completed_steps

    if phase == "training":
        if type(active_epoch) is not int or type(next_ordinal) is not int:
            message = "epoch training cursor is missing its epoch or ordinal"
            raise TypeError(message)
        if type(record) is not ExperimentTrainingStepMetricV1:
            message = "epoch journal expected the next manifest-ordered training step"
            raise ValueError(message)
        expected_step = initial_steps + (active_epoch - 1) * train_count + next_ordinal
        if (
            record.example_id != next_example_id
            or record.split_ordinal != next_ordinal
            or record.split_example_count != train_count
            or record.step_index != expected_step
            or payload.optimizer_completed_steps != expected_step
        ):
            message = "epoch training event violates manifest order or step identity"
            raise ValueError(message)
        if next_ordinal == train_count:
            next_phase: ExperimentEpochPhaseV1 = "checkpoint"
            next_training_ordinal = None
            next_training_example_id = None
        else:
            next_phase = "training"
            next_training_ordinal = next_ordinal + 1
            next_training_example_id = train_ids[next_training_ordinal - 1]
    elif phase == "checkpoint":
        if type(active_epoch) is not int:
            message = "epoch checkpoint cursor is missing its epoch"
            raise TypeError(message)
        if type(record) is not ExperimentCheckpointEventV1:
            message = "epoch journal expected a checkpoint after the complete train split"
            raise ValueError(message)
        expected_boundary_step = initial_steps + active_epoch * train_count
        if (
            record.checkpoint.completed_steps != expected_boundary_step
            or payload.optimizer_completed_steps != expected_boundary_step
        ):
            message = "epoch checkpoint is not attached to the completed epoch boundary"
            raise ValueError(message)
        next_phase = "validation"
        next_training_ordinal = None
        next_training_example_id = None
        latest_checkpoint = record.checkpoint
        latest_checkpoint_event_count = event_count
    elif phase == "validation":
        if type(active_epoch) is not int:
            message = "epoch validation cursor is missing its epoch"
            raise TypeError(message)
        expected_boundary_step = initial_steps + active_epoch * train_count
        if (
            type(record) is not ExperimentEvaluationMetricV1
            or record.split != "validation"
            or payload.optimizer_completed_steps != expected_boundary_step
        ):
            message = "epoch journal expected validation after the epoch checkpoint"
            raise ValueError(message)
        completed_epochs = active_epoch
        if active_epoch < policy.epoch_count:
            active_epoch += 1
            next_phase = "training"
            next_training_ordinal = 1
            next_training_example_id = train_ids[0]
        elif policy.evaluate_test_after_training:
            active_epoch = None
            next_phase = "test"
            next_training_ordinal = None
            next_training_example_id = None
        else:
            active_epoch = None
            next_phase = "complete"
            next_training_ordinal = None
            next_training_example_id = None
    elif phase == "test":
        expected_boundary_step = initial_steps + policy.epoch_count * train_count
        if (
            type(record) is not ExperimentEvaluationMetricV1
            or record.split != "test"
            or payload.optimizer_completed_steps != expected_boundary_step
        ):
            message = "epoch journal expected one final test evaluation"
            raise ValueError(message)
        active_epoch = None
        next_phase = "complete"
        next_training_ordinal = None
        next_training_example_id = None
    else:
        message = "epoch journal contains an event after managed execution completed"
        raise ValueError(message)

    return IrisExperimentEpochCursorV1(
        phase=next_phase,
        completed_epochs=completed_epochs,
        epoch_index=active_epoch,
        next_training_ordinal=next_training_ordinal,
        next_training_example_id=next_training_example_id,
        optimizer_completed_steps=payload.optimizer_completed_steps,
        sidecar_state_sha256=payload.sidecar_state_sha256,
        latest_checkpoint=latest_checkpoint,
        latest_checkpoint_event_count=latest_checkpoint_event_count,
    )


def analyze_experiment_epoch_progress_v1(
    *,
    run: IrisExperimentRunV1,
    events: tuple[IrisExperimentEventV1, ...],
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentEpochProgressV1:
    """Validate a complete managed prefix and bind an incremental cursor to its head."""
    policy, train_ids = _validated_epoch_inputs(
        run=run,
        events=events,
        manifest=manifest,
    )
    cursor = _initial_epoch_cursor(run=run, train_ids=train_ids)
    for event_count, event in enumerate(events, start=1):
        cursor = _advance_epoch_cursor(
            run=run,
            policy=policy,
            train_ids=train_ids,
            cursor=cursor,
            event=event,
            event_count=event_count,
        )
    chain_head = events[-1].payload_sha256 if events else run.payload_sha256
    return IrisExperimentEpochProgressV1(
        run_payload_sha256=run.payload_sha256,
        configuration_sha256=run.payload.configuration.sha256,
        manifest_payload_sha256=manifest.payload_sha256,
        verified_event_count=len(events),
        chain_head_sha256=chain_head,
        cursor=cursor,
        _train_example_ids=train_ids,
    )


def extend_experiment_epoch_progress_v1(
    *,
    progress: IrisExperimentEpochProgressV1,
    run: IrisExperimentRunV1,
    events: tuple[IrisExperimentEventV1, ...],
) -> IrisExperimentEpochProgressV1:
    """Validate only events after an already hash-bound managed prefix."""
    if type(progress) is not IrisExperimentEpochProgressV1:
        message = "epoch progress extension requires exactly IrisExperimentEpochProgressV1"
        raise TypeError(message)
    if type(run) is not IrisExperimentRunV1:
        message = "epoch progress extension requires exactly IrisExperimentRunV1"
        raise TypeError(message)
    if type(events) is not tuple:
        message = "epoch progress extension requires an exact event tuple"
        raise TypeError(message)
    policy = run.payload.configuration.execution_policy
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "epoch progress extension requires deterministic_epochs execution policy"
        raise TypeError(message)
    if (
        progress.run_payload_sha256 != run.payload_sha256
        or progress.configuration_sha256 != run.payload.configuration.sha256
        or progress.manifest_payload_sha256 != run.payload.manifest.payload_sha256
    ):
        message = "epoch progress identity does not match the experiment run"
        raise ValueError(message)
    if len(events) < progress.verified_event_count:
        message = "experiment event prefix shrank below its verified progress"
        raise ValueError(message)
    if progress.verified_event_count:
        verified_head = events[progress.verified_event_count - 1]
        if type(verified_head) is not IrisExperimentEventV1:
            message = "epoch progress head must be an exact experiment event"
            raise TypeError(message)
        observed_head = verified_head.payload_sha256
    else:
        observed_head = run.payload_sha256
    if observed_head != progress.chain_head_sha256:
        message = "experiment event prefix changed before its verified progress head"
        raise ValueError(message)

    cursor = progress.cursor
    chain_head = progress.chain_head_sha256
    for index in range(progress.verified_event_count, len(events)):
        event = events[index]
        if type(event) is not IrisExperimentEventV1:
            message = "epoch progress suffix must contain exact experiment events"
            raise TypeError(message)
        payload = event.payload
        event_count = index + 1
        if (
            payload.sequence != event_count
            or payload.run_id != run.payload.run_id
            or payload.run_payload_sha256 != run.payload_sha256
            or payload.manifest_payload_sha256 != progress.manifest_payload_sha256
            or payload.configuration_sha256 != progress.configuration_sha256
            or payload.predecessor_sha256 != chain_head
        ):
            message = "experiment event suffix identity or predecessor is inconsistent"
            raise ValueError(message)
        cursor = _advance_epoch_cursor(
            run=run,
            policy=policy,
            train_ids=progress._train_example_ids,
            cursor=cursor,
            event=event,
            event_count=event_count,
        )
        chain_head = event.payload_sha256

    if len(events) == progress.verified_event_count:
        return progress
    return IrisExperimentEpochProgressV1(
        run_payload_sha256=progress.run_payload_sha256,
        configuration_sha256=progress.configuration_sha256,
        manifest_payload_sha256=progress.manifest_payload_sha256,
        verified_event_count=len(events),
        chain_head_sha256=chain_head,
        cursor=cursor,
        _train_example_ids=progress._train_example_ids,
    )


def analyze_experiment_epoch_prefix_v1(
    *,
    run: IrisExperimentRunV1,
    events: tuple[IrisExperimentEventV1, ...],
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentEpochCursorV1:
    """Validate a complete managed prefix and return its next legal transition."""
    return analyze_experiment_epoch_progress_v1(
        run=run,
        events=events,
        manifest=manifest,
    ).cursor


def validate_experiment_epoch_candidate_v1(
    *,
    cursor: IrisExperimentEpochCursorV1,
    record: ExperimentEventRecordV1,
) -> None:
    """Reject an append that is not the exact next managed transition."""
    if type(cursor) is not IrisExperimentEpochCursorV1:
        message = "epoch candidate validation requires an exact cursor"
        raise TypeError(message)
    if cursor.phase == "training":
        valid = (
            type(record) is ExperimentTrainingStepMetricV1
            and record.example_id == cursor.next_training_example_id
            and record.split_ordinal == cursor.next_training_ordinal
            and record.step_index == cursor.optimizer_completed_steps + 1
        )
    elif cursor.phase == "checkpoint":
        valid = type(record) is ExperimentCheckpointEventV1
    elif cursor.phase in {"validation", "test"}:
        valid = type(record) is ExperimentEvaluationMetricV1 and record.split == cursor.phase
    else:
        valid = False
    if not valid:
        message = f"experiment event is not the expected managed {cursor.phase} transition"
        raise ValueError(message)


__all__ = [
    "ExperimentEpochPhaseV1",
    "IrisExperimentEpochCursorV1",
    "IrisExperimentEpochProgressV1",
    "analyze_experiment_epoch_prefix_v1",
    "analyze_experiment_epoch_progress_v1",
    "extend_experiment_epoch_progress_v1",
    "validate_experiment_epoch_candidate_v1",
]
