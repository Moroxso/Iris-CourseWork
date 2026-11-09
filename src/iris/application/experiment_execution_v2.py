"""Pure state-machine validation for arm-labelled experiment journals v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from iris.application.experiment_manifest import validate_experiment_manifest_v1
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmIdV2,
    ExperimentCheckpointEventV2,
    ExperimentCheckpointReferenceV2,
    ExperimentEventRecordV2,
    IrisExperimentEventV2,
    IrisExperimentRunV2,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1

ExperimentEpochPhaseV2 = Literal[
    "training",
    "checkpoint",
    "validation",
    "test",
    "complete",
]


@dataclass(frozen=True, slots=True)
class IrisExperimentEpochCursorV2:
    """Next legal transition and latest replay boundary for one valid v2 prefix."""

    phase: ExperimentEpochPhaseV2
    completed_epochs: int
    epoch_index: int | None
    next_training_ordinal: int | None
    next_training_example_id: str | None
    optimizer_completed_steps: int
    sidecar_state_sha256: str
    latest_checkpoint: ExperimentCheckpointReferenceV2
    latest_checkpoint_event_count: int


@dataclass(frozen=True, slots=True)
class IrisExperimentEpochProgressV2:
    """Arm- and hash-bound cursor for incrementally verified v2 event prefixes."""

    arm_id: ExperimentArmIdV2
    run_payload_sha256: str
    configuration_sha256: str
    manifest_payload_sha256: str
    verified_event_count: int
    chain_head_sha256: str
    cursor: IrisExperimentEpochCursorV2
    _train_example_ids: tuple[str, ...] = field(repr=False)


def _manifest_matches_run(
    manifest: IrisExperimentManifestV1,
    run: IrisExperimentRunV2,
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
    run: IrisExperimentRunV2,
    events: tuple[IrisExperimentEventV2, ...],
    manifest: IrisExperimentManifestV1,
) -> tuple[ExperimentEpochExecutionPolicyV1, tuple[str, ...]]:
    if type(run) is not IrisExperimentRunV2:
        message = "epoch state machine requires exactly IrisExperimentRunV2"
        raise TypeError(message)
    if type(events) is not tuple or any(
        type(event) is not IrisExperimentEventV2 for event in events
    ):
        message = "epoch state machine requires an exact v2 event tuple"
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
    run: IrisExperimentRunV2,
    train_ids: tuple[str, ...],
) -> IrisExperimentEpochCursorV2:
    initial = run.payload.initial_checkpoint
    return IrisExperimentEpochCursorV2(
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


def _validate_event_identity(
    *,
    run: IrisExperimentRunV2,
    event: IrisExperimentEventV2,
    expected_sequence: int,
    predecessor_sha256: str,
) -> None:
    payload = event.payload
    if (
        payload.sequence != expected_sequence
        or payload.run_id != run.payload.run_id
        or payload.arm_id != run.payload.arm_id
        or payload.run_payload_sha256 != run.payload_sha256
        or payload.manifest_payload_sha256 != run.payload.manifest.payload_sha256
        or payload.configuration_sha256 != run.payload.configuration.sha256
        or payload.predecessor_sha256 != predecessor_sha256
    ):
        message = "experiment event identity, arm, or predecessor is inconsistent"
        raise ValueError(message)


def _advance_epoch_cursor(
    *,
    run: IrisExperimentRunV2,
    policy: ExperimentEpochExecutionPolicyV1,
    train_ids: tuple[str, ...],
    cursor: IrisExperimentEpochCursorV2,
    event: IrisExperimentEventV2,
    event_count: int,
) -> IrisExperimentEpochCursorV2:
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
            next_phase: ExperimentEpochPhaseV2 = "checkpoint"
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
        if type(record) is not ExperimentCheckpointEventV2:
            message = "epoch journal expected a checkpoint after the complete train split"
            raise ValueError(message)
        expected_boundary_step = initial_steps + active_epoch * train_count
        if (
            record.checkpoint.arm_id != run.payload.arm_id
            or record.checkpoint.completed_steps != expected_boundary_step
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

    return IrisExperimentEpochCursorV2(
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


def analyze_experiment_epoch_progress_v2(
    *,
    run: IrisExperimentRunV2,
    events: tuple[IrisExperimentEventV2, ...],
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentEpochProgressV2:
    """Validate a complete v2 managed prefix and bind a cursor to its head."""
    policy, train_ids = _validated_epoch_inputs(
        run=run,
        events=events,
        manifest=manifest,
    )
    cursor = _initial_epoch_cursor(run=run, train_ids=train_ids)
    chain_head = run.payload_sha256
    for event_count, event in enumerate(events, start=1):
        _validate_event_identity(
            run=run,
            event=event,
            expected_sequence=event_count,
            predecessor_sha256=chain_head,
        )
        cursor = _advance_epoch_cursor(
            run=run,
            policy=policy,
            train_ids=train_ids,
            cursor=cursor,
            event=event,
            event_count=event_count,
        )
        chain_head = event.payload_sha256
    return IrisExperimentEpochProgressV2(
        arm_id=run.payload.arm_id,
        run_payload_sha256=run.payload_sha256,
        configuration_sha256=run.payload.configuration.sha256,
        manifest_payload_sha256=manifest.payload_sha256,
        verified_event_count=len(events),
        chain_head_sha256=chain_head,
        cursor=cursor,
        _train_example_ids=train_ids,
    )


def extend_experiment_epoch_progress_v2(
    *,
    progress: IrisExperimentEpochProgressV2,
    run: IrisExperimentRunV2,
    events: tuple[IrisExperimentEventV2, ...],
) -> IrisExperimentEpochProgressV2:
    """Validate only v2 events after an already arm- and hash-bound prefix."""
    if type(progress) is not IrisExperimentEpochProgressV2:
        message = "epoch progress extension requires exactly IrisExperimentEpochProgressV2"
        raise TypeError(message)
    if type(run) is not IrisExperimentRunV2:
        message = "epoch progress extension requires exactly IrisExperimentRunV2"
        raise TypeError(message)
    if type(events) is not tuple:
        message = "epoch progress extension requires an exact event tuple"
        raise TypeError(message)
    policy = run.payload.configuration.execution_policy
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "epoch progress extension requires deterministic_epochs execution policy"
        raise TypeError(message)
    if (
        progress.arm_id != run.payload.arm_id
        or progress.run_payload_sha256 != run.payload_sha256
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
        if type(verified_head) is not IrisExperimentEventV2:
            message = "epoch progress head must be an exact v2 experiment event"
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
        if type(event) is not IrisExperimentEventV2:
            message = "epoch progress suffix must contain exact v2 experiment events"
            raise TypeError(message)
        event_count = index + 1
        _validate_event_identity(
            run=run,
            event=event,
            expected_sequence=event_count,
            predecessor_sha256=chain_head,
        )
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
    return IrisExperimentEpochProgressV2(
        arm_id=progress.arm_id,
        run_payload_sha256=progress.run_payload_sha256,
        configuration_sha256=progress.configuration_sha256,
        manifest_payload_sha256=progress.manifest_payload_sha256,
        verified_event_count=len(events),
        chain_head_sha256=chain_head,
        cursor=cursor,
        _train_example_ids=progress._train_example_ids,
    )


def analyze_experiment_epoch_prefix_v2(
    *,
    run: IrisExperimentRunV2,
    events: tuple[IrisExperimentEventV2, ...],
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentEpochCursorV2:
    """Validate a complete v2 managed prefix and return its next transition."""
    return analyze_experiment_epoch_progress_v2(
        run=run,
        events=events,
        manifest=manifest,
    ).cursor


def validate_experiment_epoch_candidate_v2(
    *,
    cursor: IrisExperimentEpochCursorV2,
    record: ExperimentEventRecordV2,
) -> None:
    """Reject an append that is not the exact next managed v2 transition."""
    if type(cursor) is not IrisExperimentEpochCursorV2:
        message = "epoch candidate validation requires an exact v2 cursor"
        raise TypeError(message)
    if cursor.phase == "training":
        valid = (
            type(record) is ExperimentTrainingStepMetricV1
            and record.example_id == cursor.next_training_example_id
            and record.split_ordinal == cursor.next_training_ordinal
            and record.step_index == cursor.optimizer_completed_steps + 1
        )
    elif cursor.phase == "checkpoint":
        valid = type(record) is ExperimentCheckpointEventV2
    elif cursor.phase in {"validation", "test"}:
        valid = type(record) is ExperimentEvaluationMetricV1 and record.split == cursor.phase
    else:
        valid = False
    if not valid:
        message = f"experiment event is not the expected managed {cursor.phase} transition"
        raise ValueError(message)


__all__ = [
    "ExperimentEpochPhaseV2",
    "IrisExperimentEpochCursorV2",
    "IrisExperimentEpochProgressV2",
    "analyze_experiment_epoch_prefix_v2",
    "analyze_experiment_epoch_progress_v2",
    "extend_experiment_epoch_progress_v2",
    "validate_experiment_epoch_candidate_v2",
]
