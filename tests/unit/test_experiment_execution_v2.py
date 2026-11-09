from __future__ import annotations

from typing import cast

import pytest

from iris.application.experiment_execution_v2 import (
    IrisExperimentEpochProgressV2,
    analyze_experiment_epoch_prefix_v2,
    analyze_experiment_epoch_progress_v2,
    extend_experiment_epoch_progress_v2,
    validate_experiment_epoch_candidate_v2,
)
from iris.application.experiment_manifest import build_experiment_manifest_v1
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManifestReferenceV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmBCheckpointReferenceV2,
    ExperimentArmDCheckpointReferenceV2,
    ExperimentArmIdV2,
    ExperimentCheckpointEventV2,
    ExperimentCheckpointReferenceV2,
    ExperimentEventPayloadV2,
    ExperimentEventRecordV2,
    ExperimentEventSchemaV2,
    ExperimentRunPayloadV2,
    ExperimentRunSchemaV2,
    IrisExperimentEventV2,
    IrisExperimentRunV2,
)
from iris.artifacts.experiment_v1 import ExperimentExampleV1, IrisExperimentManifestV1
from tests.unit.test_experiment_log_schema_v2 import _run_payload


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
    return build_experiment_manifest_v1(dataset_id="managed-v2", examples=examples)


def _managed_run(
    arm_id: ExperimentArmIdV2,
) -> tuple[IrisExperimentRunV2, IrisExperimentManifestV1]:
    manifest = _manifest()
    values = _run_payload(arm_id).model_dump(mode="python")
    values["manifest"] = ExperimentManifestReferenceV1(
        dataset_id=manifest.payload.dataset_id,
        payload_sha256=manifest.payload_sha256,
        split_counts=manifest.payload.split_counts,
    ).model_dump(mode="python")
    values["configuration"]["execution_policy"] = ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=1,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=True,
        resume_policy="latest_checkpoint_plus_verified_replay",
    ).model_dump(mode="python")
    payload = ExperimentRunPayloadV2.model_validate(values)
    return (
        IrisExperimentRunV2(
            schema_identity=ExperimentRunSchemaV2(
                artifact_type="iris.experiment_run",
                version=2,
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
    arm_id: ExperimentArmIdV2,
    *,
    completed_steps: int,
    sidecar_state_sha256: str,
) -> ExperimentCheckpointReferenceV2:
    if arm_id == "B":
        return ExperimentArmBCheckpointReferenceV2(
            checkpoint_schema_version=2,
            arm_id="B",
            payload_sha256=_sha(201),
            state_file_sha256=_sha(202),
            sidecar_state_sha256=sidecar_state_sha256,
            completed_steps=completed_steps,
        )
    return ExperimentArmDCheckpointReferenceV2(
        checkpoint_schema_version=1,
        arm_id="D",
        payload_sha256=_sha(201),
        state_file_sha256=_sha(202),
        sidecar_state_sha256=sidecar_state_sha256,
        completed_steps=completed_steps,
    )


def _event(
    *,
    run: IrisExperimentRunV2,
    sequence: int,
    predecessor_sha256: str,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    record: ExperimentEventRecordV2,
    arm_id: ExperimentArmIdV2 | None = None,
) -> IrisExperimentEventV2:
    return IrisExperimentEventV2(
        schema_identity=ExperimentEventSchemaV2(
            artifact_type="iris.experiment_event",
            version=2,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=_sha(300 + sequence),
        payload=ExperimentEventPayloadV2(
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


def _complete_events(run: IrisExperimentRunV2) -> tuple[IrisExperimentEventV2, ...]:
    first_state = _sha(401)
    second_state = _sha(402)
    records: tuple[tuple[ExperimentEventRecordV2, int, str], ...] = (
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
            ExperimentCheckpointEventV2(
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
    events: list[IrisExperimentEventV2] = []
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


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_complete_epoch_prefix_is_arm_bound_and_reaches_terminal_state(
    arm_id: ExperimentArmIdV2,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)

    progress = analyze_experiment_epoch_progress_v2(
        run=run,
        events=events,
        manifest=manifest,
    )

    assert progress.arm_id == arm_id
    assert progress.verified_event_count == 5
    assert progress.chain_head_sha256 == events[-1].payload_sha256
    assert progress.cursor.phase == "complete"
    assert progress.cursor.completed_epochs == 1
    assert progress.cursor.optimizer_completed_steps == 2
    assert (
        progress.cursor.latest_checkpoint
        == cast(ExperimentCheckpointEventV2, events[2].payload.record).checkpoint
    )
    assert progress.cursor.latest_checkpoint_event_count == 3
    assert (
        analyze_experiment_epoch_prefix_v2(
            run=run,
            events=events,
            manifest=manifest,
        )
        == progress.cursor
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_incremental_extension_matches_complete_analysis(
    arm_id: ExperimentArmIdV2,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    prefix = analyze_experiment_epoch_progress_v2(
        run=run,
        events=events[:2],
        manifest=manifest,
    )

    extended = extend_experiment_epoch_progress_v2(
        progress=prefix,
        run=run,
        events=events,
    )
    complete = analyze_experiment_epoch_progress_v2(
        run=run,
        events=events,
        manifest=manifest,
    )

    assert prefix.cursor.phase == "checkpoint"
    assert extended == complete
    assert (
        extend_experiment_epoch_progress_v2(
            progress=complete,
            run=run,
            events=events,
        )
        is complete
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_event_arm_predecessor_and_verified_head_are_fail_closed(
    arm_id: ExperimentArmIdV2,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    other_arm: ExperimentArmIdV2 = "D" if arm_id == "B" else "B"
    cross_arm = _event(
        run=run,
        sequence=1,
        predecessor_sha256=run.payload_sha256,
        optimizer_completed_steps=1,
        sidecar_state_sha256=events[0].payload.sidecar_state_sha256,
        record=events[0].payload.record,
        arm_id=other_arm,
    )
    with pytest.raises(ValueError, match="identity, arm, or predecessor"):
        analyze_experiment_epoch_progress_v2(
            run=run,
            events=(cross_arm,),
            manifest=manifest,
        )

    wrong_predecessor = events[0].model_copy(
        update={"payload": events[0].payload.model_copy(update={"predecessor_sha256": _sha(999)})}
    )
    with pytest.raises(ValueError, match="identity, arm, or predecessor"):
        analyze_experiment_epoch_progress_v2(
            run=run,
            events=(wrong_predecessor,),
            manifest=manifest,
        )

    prefix = analyze_experiment_epoch_progress_v2(
        run=run,
        events=events[:2],
        manifest=manifest,
    )
    changed_head = events[1].model_copy(update={"payload_sha256": _sha(998)})
    with pytest.raises(ValueError, match="changed before"):
        extend_experiment_epoch_progress_v2(
            progress=prefix,
            run=run,
            events=(events[0], changed_head),
        )
    with pytest.raises(ValueError, match="shrank"):
        extend_experiment_epoch_progress_v2(
            progress=prefix,
            run=run,
            events=events[:1],
        )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_candidate_validation_tracks_each_managed_phase(
    arm_id: ExperimentArmIdV2,
) -> None:
    run, manifest = _managed_run(arm_id)
    events = _complete_events(run)
    initial = analyze_experiment_epoch_prefix_v2(run=run, events=(), manifest=manifest)
    validate_experiment_epoch_candidate_v2(
        cursor=initial,
        record=events[0].payload.record,
    )
    with pytest.raises(ValueError, match="managed training"):
        validate_experiment_epoch_candidate_v2(
            cursor=initial,
            record=events[2].payload.record,
        )

    checkpoint = analyze_experiment_epoch_prefix_v2(
        run=run,
        events=events[:2],
        manifest=manifest,
    )
    validate_experiment_epoch_candidate_v2(
        cursor=checkpoint,
        record=events[2].payload.record,
    )
    with pytest.raises(ValueError, match="managed checkpoint"):
        validate_experiment_epoch_candidate_v2(
            cursor=checkpoint,
            record=events[3].payload.record,
        )

    complete = analyze_experiment_epoch_prefix_v2(
        run=run,
        events=events,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="managed complete"):
        validate_experiment_epoch_candidate_v2(
            cursor=complete,
            record=events[-1].payload.record,
        )


def test_exact_v2_types_and_matching_manifest_are_required() -> None:
    run, manifest = _managed_run("B")
    events = _complete_events(run)

    class RunSubclass(IrisExperimentRunV2):
        pass

    subclass = RunSubclass.model_validate(run.model_dump(mode="python"))
    with pytest.raises(TypeError, match="exactly IrisExperimentRunV2"):
        analyze_experiment_epoch_progress_v2(
            run=subclass,
            events=(),
            manifest=manifest,
        )

    other_manifest = build_experiment_manifest_v1(
        dataset_id="other-v2",
        examples=manifest.payload.examples,
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        analyze_experiment_epoch_progress_v2(
            run=run,
            events=events,
            manifest=other_manifest,
        )

    with pytest.raises(TypeError, match="exact v2 event tuple"):
        analyze_experiment_epoch_progress_v2(
            run=run,
            events=cast(tuple[IrisExperimentEventV2, ...], list(events)),
            manifest=manifest,
        )


def test_progress_type_is_exact_for_extension() -> None:
    run, manifest = _managed_run("D")
    events = _complete_events(run)
    progress = analyze_experiment_epoch_progress_v2(
        run=run,
        events=events[:1],
        manifest=manifest,
    )

    class ProgressSubclass(IrisExperimentEpochProgressV2):
        pass

    subclass = ProgressSubclass(
        arm_id=progress.arm_id,
        run_payload_sha256=progress.run_payload_sha256,
        configuration_sha256=progress.configuration_sha256,
        manifest_payload_sha256=progress.manifest_payload_sha256,
        verified_event_count=progress.verified_event_count,
        chain_head_sha256=progress.chain_head_sha256,
        cursor=progress.cursor,
        _train_example_ids=("train-2", "train-1"),
    )
    with pytest.raises(TypeError, match="exactly IrisExperimentEpochProgressV2"):
        extend_experiment_epoch_progress_v2(
            progress=subclass,
            run=run,
            events=events,
        )
