from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from iris.application.experiment_execution_v3 import (
    IrisExperimentEpochCursorV3,
    IrisExperimentEpochProgressV3,
    analyze_experiment_epoch_prefix_v3,
    analyze_experiment_epoch_progress_v3,
    extend_experiment_epoch_progress_v3,
    validate_experiment_epoch_candidate_v3,
)
from iris.application.experiment_manifest import build_experiment_manifest_v1
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManifestReferenceV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v3 import (
    ExperimentArmBCheckpointReferenceV3,
    ExperimentArmCCheckpointReferenceV3,
    ExperimentArmDCheckpointReferenceV3,
    ExperimentArmIdV3,
    ExperimentCheckpointEventV3,
    ExperimentCheckpointReferenceV3,
    ExperimentEventPayloadV3,
    ExperimentEventRecordV3,
    ExperimentEventSchemaV3,
    ExperimentRunPayloadV3,
    ExperimentRunSchemaV3,
    IrisExperimentEventV3,
    IrisExperimentRunV3,
)
from iris.artifacts.experiment_v1 import ExperimentExampleV1, IrisExperimentManifestV1
from tests.unit.test_experiment_log_schema_v3 import _run_payload


def _sha(value: int) -> str:
    return f"{value:064x}"


def _manifest() -> IrisExperimentManifestV1:
    examples = (
        ExperimentExampleV1(
            example_id="train-2",
            split="train",
            prompt="second prompt",
            target="second target",
        ),
        ExperimentExampleV1(
            example_id="validation-1",
            split="validation",
            prompt="validation prompt",
            target="validation target",
        ),
        ExperimentExampleV1(
            example_id="train-1",
            split="train",
            prompt="first prompt",
            target="first target",
        ),
        ExperimentExampleV1(
            example_id="test-1",
            split="test",
            prompt="test prompt",
            target="test target",
        ),
    )
    return build_experiment_manifest_v1(dataset_id="managed-v3", examples=examples)


def _managed_run(
    arm_id: ExperimentArmIdV3,
    *,
    epoch_count: int = 1,
) -> tuple[IrisExperimentRunV3, IrisExperimentManifestV1]:
    manifest = _manifest()
    values = _run_payload(arm_id).model_dump(mode="python")
    values["manifest"] = ExperimentManifestReferenceV1(
        dataset_id=manifest.payload.dataset_id,
        payload_sha256=manifest.payload_sha256,
        split_counts=manifest.payload.split_counts,
    ).model_dump(mode="python")
    values["configuration"]["execution_policy"] = ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=epoch_count,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=True,
        resume_policy="latest_checkpoint_plus_verified_replay",
    ).model_dump(mode="python")
    payload = ExperimentRunPayloadV3.model_validate(values)
    return (
        IrisExperimentRunV3(
            schema_identity=ExperimentRunSchemaV3(
                artifact_type="iris.experiment_run",
                version=3,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                journal_layout="immutable_run_and_chained_events",
            ),
            payload_sha256=_sha(100),
            payload=payload,
        ),
        manifest,
    )


def _checkpoint(
    arm_id: ExperimentArmIdV3,
    *,
    completed_steps: int,
    sidecar_state_sha256: str,
) -> ExperimentCheckpointReferenceV3:
    if arm_id == "B":
        return ExperimentArmBCheckpointReferenceV3(
            checkpoint_schema_version=2,
            arm_id="B",
            payload_sha256=_sha(201),
            state_file_sha256=_sha(202),
            sidecar_state_sha256=sidecar_state_sha256,
            completed_steps=completed_steps,
        )
    if arm_id == "C":
        return ExperimentArmCCheckpointReferenceV3(
            checkpoint_schema_version=3,
            arm_id="C",
            payload_sha256=_sha(201),
            state_file_sha256=_sha(202),
            sidecar_state_sha256=sidecar_state_sha256,
            completed_steps=completed_steps,
        )
    return ExperimentArmDCheckpointReferenceV3(
        checkpoint_schema_version=1,
        arm_id="D",
        payload_sha256=_sha(201),
        state_file_sha256=_sha(202),
        sidecar_state_sha256=sidecar_state_sha256,
        completed_steps=completed_steps,
    )


def _event(
    *,
    run: IrisExperimentRunV3,
    sequence: int,
    predecessor_sha256: str,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    record: ExperimentEventRecordV3,
    arm_id: ExperimentArmIdV3 | None = None,
) -> IrisExperimentEventV3:
    return IrisExperimentEventV3(
        schema_identity=ExperimentEventSchemaV3(
            artifact_type="iris.experiment_event",
            version=3,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=_sha(300 + sequence),
        payload=ExperimentEventPayloadV3(
            sequence=sequence,
            run_id=run.payload.run_id,
            arm_id=arm_id or run.payload.arm_id,
            run_payload_sha256=run.payload_sha256,
            manifest_payload_sha256=run.payload.manifest.payload_sha256,
            configuration_sha256=run.payload.configuration.sha256,
            predecessor_sha256=predecessor_sha256,
            optimizer_completed_steps=optimizer_completed_steps,
            sidecar_state_sha256=sidecar_state_sha256,
            record=record,
        ),
    )


def _complete_events(run: IrisExperimentRunV3) -> tuple[IrisExperimentEventV3, ...]:
    first_state = _sha(401)
    second_state = _sha(402)
    records: tuple[tuple[ExperimentEventRecordV3, int, str], ...] = (
        (
            ExperimentTrainingStepMetricV1(
                event_type="training_step",
                split="train",
                example_id="train-2",
                split_ordinal=1,
                split_example_count=2,
                step_index=1,
                loss=1.0,
                supervised_token_count=2,
                gradient_l2_norm=0.5,
                gradients_clipped=False,
            ),
            1,
            first_state,
        ),
        (
            ExperimentTrainingStepMetricV1(
                event_type="training_step",
                split="train",
                example_id="train-1",
                split_ordinal=2,
                split_example_count=2,
                step_index=2,
                loss=0.75,
                supervised_token_count=3,
                gradient_l2_norm=0.25,
                gradients_clipped=True,
            ),
            2,
            second_state,
        ),
        (
            ExperimentCheckpointEventV3(
                event_type="checkpoint",
                checkpoint=_checkpoint(
                    run.payload.arm_id,
                    completed_steps=2,
                    sidecar_state_sha256=second_state,
                ),
            ),
            2,
            second_state,
        ),
        (
            ExperimentEvaluationMetricV1(
                event_type="evaluation",
                split="validation",
                example_count=1,
                supervised_token_count=2,
                mean_loss=0.0,
                perplexity=1.0,
            ),
            2,
            second_state,
        ),
        (
            ExperimentEvaluationMetricV1(
                event_type="evaluation",
                split="test",
                example_count=1,
                supervised_token_count=2,
                mean_loss=0.0,
                perplexity=1.0,
            ),
            2,
            second_state,
        ),
    )
    predecessor = run.payload_sha256
    events: list[IrisExperimentEventV3] = []
    for sequence, (record, completed_steps, state_hash) in enumerate(records, start=1):
        event = _event(
            run=run,
            sequence=sequence,
            predecessor_sha256=predecessor,
            optimizer_completed_steps=completed_steps,
            sidecar_state_sha256=state_hash,
            record=record,
        )
        events.append(event)
        predecessor = event.payload_sha256
    return tuple(events)


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_complete_epoch_prefix_routes_all_profile_v4_checkpoint_types(
    arm_id: ExperimentArmIdV3,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)

    progress = analyze_experiment_epoch_progress_v3(
        run=run,
        events=events,
        manifest=manifest,
    )

    expected_checkpoint_type = {
        "B": ExperimentArmBCheckpointReferenceV3,
        "C": ExperimentArmCCheckpointReferenceV3,
        "D": ExperimentArmDCheckpointReferenceV3,
    }[arm_id]
    assert progress.arm_id == arm_id
    assert progress.verified_event_count == 5
    assert progress.chain_head_sha256 == events[-1].payload_sha256
    assert progress.cursor.phase == "complete"
    assert progress.cursor.completed_epochs == 1
    assert progress.cursor.optimizer_completed_steps == 2
    assert type(progress.cursor.latest_checkpoint) is expected_checkpoint_type
    assert (
        progress.cursor.latest_checkpoint
        == cast(ExperimentCheckpointEventV3, events[2].payload.record).checkpoint
    )
    assert progress.cursor.latest_checkpoint_event_count == 3
    assert (
        analyze_experiment_epoch_prefix_v3(
            run=run,
            events=events,
            manifest=manifest,
        )
        == progress.cursor
    )


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_incremental_extension_matches_complete_replay(
    arm_id: ExperimentArmIdV3,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    prefix = analyze_experiment_epoch_progress_v3(
        run=run,
        events=events[:2],
        manifest=manifest,
    )

    extended = extend_experiment_epoch_progress_v3(
        progress=prefix,
        run=run,
        events=events,
    )
    complete = analyze_experiment_epoch_progress_v3(
        run=run,
        events=events,
        manifest=manifest,
    )

    assert prefix.cursor.phase == "checkpoint"
    assert extended == complete
    assert (
        extend_experiment_epoch_progress_v3(
            progress=complete,
            run=run,
            events=events,
        )
        is complete
    )


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_candidate_validation_tracks_every_managed_phase(
    arm_id: ExperimentArmIdV3,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    initial = analyze_experiment_epoch_prefix_v3(run=run, events=(), manifest=manifest)
    validate_experiment_epoch_candidate_v3(
        cursor=initial,
        record=events[0].payload.record,
    )
    with pytest.raises(ValueError, match="managed training"):
        validate_experiment_epoch_candidate_v3(
            cursor=initial,
            record=events[2].payload.record,
        )

    checkpoint = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=events[:2],
        manifest=manifest,
    )
    validate_experiment_epoch_candidate_v3(
        cursor=checkpoint,
        record=events[2].payload.record,
    )
    with pytest.raises(ValueError, match="managed checkpoint"):
        validate_experiment_epoch_candidate_v3(
            cursor=checkpoint,
            record=events[3].payload.record,
        )

    validation = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=events[:3],
        manifest=manifest,
    )
    validate_experiment_epoch_candidate_v3(
        cursor=validation,
        record=events[3].payload.record,
    )
    test = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=events[:4],
        manifest=manifest,
    )
    validate_experiment_epoch_candidate_v3(cursor=test, record=events[4].payload.record)

    complete = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=events,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="managed complete"):
        validate_experiment_epoch_candidate_v3(
            cursor=complete,
            record=events[-1].payload.record,
        )


def test_validation_boundary_starts_the_next_epoch_in_manifest_order() -> None:
    run, manifest = _managed_run("C", epoch_count=2)
    first_epoch = _complete_events(run)[:4]

    cursor = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=first_epoch,
        manifest=manifest,
    )
    next_step = ExperimentTrainingStepMetricV1(
        event_type="training_step",
        split="train",
        example_id="train-2",
        split_ordinal=1,
        split_example_count=2,
        step_index=3,
        loss=0.5,
        supervised_token_count=2,
        gradient_l2_norm=0.2,
        gradients_clipped=False,
    )

    assert cursor.phase == "training"
    assert cursor.completed_epochs == 1
    assert cursor.epoch_index == 2
    assert cursor.next_training_ordinal == 1
    assert cursor.next_training_example_id == "train-2"
    validate_experiment_epoch_candidate_v3(cursor=cursor, record=next_step)


@pytest.mark.parametrize(
    ("arm_id", "other_arm"),
    [("B", "C"), ("C", "D"), ("D", "B")],
)
def test_event_arm_hash_predecessor_and_verified_head_are_fail_closed(
    arm_id: ExperimentArmIdV3,
    other_arm: ExperimentArmIdV3,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    cross_arm = _event(
        run=run,
        sequence=1,
        predecessor_sha256=run.payload_sha256,
        optimizer_completed_steps=1,
        sidecar_state_sha256=events[0].payload.sidecar_state_sha256,
        record=events[0].payload.record,
        arm_id=other_arm,
    )
    with pytest.raises(ValueError, match="identity, arm, hash, or predecessor"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(cross_arm,),
            manifest=manifest,
        )

    wrong_hash = events[0].model_copy(
        update={
            "payload": events[0].payload.model_copy(
                update={"configuration_sha256": _sha(997)}
            )
        }
    )
    with pytest.raises(ValueError, match="identity, arm, hash, or predecessor"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(wrong_hash,),
            manifest=manifest,
        )

    wrong_predecessor = events[0].model_copy(
        update={
            "payload": events[0].payload.model_copy(
                update={"predecessor_sha256": _sha(999)}
            )
        }
    )
    with pytest.raises(ValueError, match="identity, arm, hash, or predecessor"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(wrong_predecessor,),
            manifest=manifest,
        )

    prefix = analyze_experiment_epoch_progress_v3(
        run=run,
        events=events[:2],
        manifest=manifest,
    )
    changed_head = events[1].model_copy(update={"payload_sha256": _sha(998)})
    with pytest.raises(ValueError, match="changed before"):
        extend_experiment_epoch_progress_v3(
            progress=prefix,
            run=run,
            events=(events[0], changed_head),
        )
    with pytest.raises(ValueError, match="shrank"):
        extend_experiment_epoch_progress_v3(
            progress=prefix,
            run=run,
            events=events[:1],
        )
    changed_binding = replace(prefix, configuration_sha256=_sha(996))
    with pytest.raises(ValueError, match="identity or hashes"):
        extend_experiment_epoch_progress_v3(
            progress=changed_binding,
            run=run,
            events=events,
        )


@pytest.mark.parametrize(
    ("arm_id", "other_arm"),
    [("B", "C"), ("C", "D"), ("D", "B")],
)
def test_cross_arm_checkpoint_reference_is_rejected_before_transition(
    arm_id: ExperimentArmIdV3,
    other_arm: ExperimentArmIdV3,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    checkpoint_record = cast(ExperimentCheckpointEventV3, events[2].payload.record)
    cross_arm_checkpoint = _checkpoint(
        other_arm,
        completed_steps=2,
        sidecar_state_sha256=events[2].payload.sidecar_state_sha256,
    )
    tampered_record = checkpoint_record.model_copy(
        update={"checkpoint": cross_arm_checkpoint}
    )
    tampered_event = events[2].model_copy(
        update={"payload": events[2].payload.model_copy(update={"record": tampered_record})}
    )

    with pytest.raises(TypeError, match=f"exact arm-{arm_id} v3 checkpoint"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(*events[:2], tampered_event),
            manifest=manifest,
        )
    checkpoint_cursor = analyze_experiment_epoch_prefix_v3(
        run=run,
        events=events[:2],
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="managed checkpoint"):
        validate_experiment_epoch_candidate_v3(
            cursor=checkpoint_cursor,
            record=tampered_record,
        )


def test_exact_v3_types_and_matching_manifest_are_required() -> None:
    run, manifest = _managed_run("C")
    events = _complete_events(run)

    class RunSubclass(IrisExperimentRunV3):
        pass

    class EventSubclass(IrisExperimentEventV3):
        pass

    run_subclass = RunSubclass.model_validate(run.model_dump(mode="python"))
    event_subclass = EventSubclass.model_validate(events[0].model_dump(mode="python"))
    with pytest.raises(TypeError, match="exactly IrisExperimentRunV3"):
        analyze_experiment_epoch_progress_v3(
            run=run_subclass,
            events=(),
            manifest=manifest,
        )
    with pytest.raises(TypeError, match="exact v3 event tuple"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(event_subclass,),
            manifest=manifest,
        )
    with pytest.raises(TypeError, match="exact v3 event tuple"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=cast(tuple[IrisExperimentEventV3, ...], list(events)),
            manifest=manifest,
        )

    other_manifest = build_experiment_manifest_v1(
        dataset_id="other-v3",
        examples=manifest.payload.examples,
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=events,
            manifest=other_manifest,
        )


def test_progress_and_cursor_types_are_exact() -> None:
    run, manifest = _managed_run("C")
    events = _complete_events(run)
    progress = analyze_experiment_epoch_progress_v3(
        run=run,
        events=events[:1],
        manifest=manifest,
    )

    class ProgressSubclass(IrisExperimentEpochProgressV3):
        pass

    class CursorSubclass(IrisExperimentEpochCursorV3):
        pass

    progress_subclass = ProgressSubclass(
        arm_id=progress.arm_id,
        run_payload_sha256=progress.run_payload_sha256,
        configuration_sha256=progress.configuration_sha256,
        manifest_payload_sha256=progress.manifest_payload_sha256,
        verified_event_count=progress.verified_event_count,
        chain_head_sha256=progress.chain_head_sha256,
        cursor=progress.cursor,
        _train_example_ids=("train-2", "train-1"),
    )
    with pytest.raises(TypeError, match="exactly IrisExperimentEpochProgressV3"):
        extend_experiment_epoch_progress_v3(
            progress=progress_subclass,
            run=run,
            events=events,
        )

    cursor_subclass = CursorSubclass(
        phase=progress.cursor.phase,
        completed_epochs=progress.cursor.completed_epochs,
        epoch_index=progress.cursor.epoch_index,
        next_training_ordinal=progress.cursor.next_training_ordinal,
        next_training_example_id=progress.cursor.next_training_example_id,
        optimizer_completed_steps=progress.cursor.optimizer_completed_steps,
        sidecar_state_sha256=progress.cursor.sidecar_state_sha256,
        latest_checkpoint=progress.cursor.latest_checkpoint,
        latest_checkpoint_event_count=progress.cursor.latest_checkpoint_event_count,
    )
    with pytest.raises(TypeError, match="exact v3 cursor"):
        validate_experiment_epoch_candidate_v3(
            cursor=cursor_subclass,
            record=events[1].payload.record,
        )


def test_checkpoint_reference_subclass_is_rejected() -> None:
    run, manifest = _managed_run("C")
    events = _complete_events(run)
    checkpoint_record = cast(ExperimentCheckpointEventV3, events[2].payload.record)

    class DerivedCheckpoint(ExperimentArmCCheckpointReferenceV3):
        pass

    derived = DerivedCheckpoint.model_validate(
        checkpoint_record.checkpoint.model_dump(mode="python")
    )
    tampered_record = checkpoint_record.model_copy(update={"checkpoint": derived})
    tampered_event = events[2].model_copy(
        update={"payload": events[2].payload.model_copy(update={"record": tampered_record})}
    )
    with pytest.raises(TypeError, match="exact arm-C v3 checkpoint"):
        analyze_experiment_epoch_progress_v3(
            run=run,
            events=(*events[:2], tampered_event),
            manifest=manifest,
        )

    tampered_payload = run.payload.model_copy(update={"initial_checkpoint": derived})
    tampered_run = run.model_copy(update={"payload": tampered_payload})
    with pytest.raises(TypeError, match=r"initial checkpoint.*exact arm-C"):
        analyze_experiment_epoch_progress_v3(
            run=tampered_run,
            events=(),
            manifest=manifest,
        )
