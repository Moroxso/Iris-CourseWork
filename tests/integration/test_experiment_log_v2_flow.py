"""Offline B/D end-to-end integration tests for experiment journals v2."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import cast

import pytest
import torch

from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
)
from iris.application.errors import IrisExperimentLogError
from iris.application.experiment_log_v2 import (
    IrisExperimentLogCreateResultV2,
    create_experiment_log_v2,
    read_experiment_log_v2,
    record_experiment_checkpoint_v2,
    record_experiment_evaluation_v2,
    record_experiment_training_step_v2,
    validate_experiment_checkpoint_for_run_v2,
)
from iris.application.experiment_manifest import (
    IrisPreparedExperimentExample,
    build_experiment_manifest_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmBCheckpointReferenceV2,
    ExperimentArmDCheckpointReferenceV2,
    ExperimentCheckpointEventV2,
)
from iris.artifacts.experiment_v1 import ExperimentExampleV1, IrisExperimentManifestV1
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationSummary,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_teacher_forcing_sidecar import _build_offline_model
from tests.unit.test_no_workspace_teacher_forcing import make_harness as make_b_harness

pytestmark = pytest.mark.integration

_SEED = 1907
_TERMINAL_TOKEN_ID = 15
_PROMPT = "offline teacher prompt"
_MANUAL_POLICY = ExperimentManualExecutionPolicyV1(mode="manual_event_chain")
_MANAGED_POLICY = ExperimentEpochExecutionPolicyV1(
    mode="deterministic_epochs",
    epoch_count=1,
    training_order="manifest_order_each_epoch",
    checkpoint_policy="after_each_complete_epoch",
    validation_policy="after_each_complete_epoch",
    evaluate_test_after_training=False,
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


def _owner(
    arm_id: IrisTeacherArmId,
) -> tuple[IrisArmTeacherModel, IrisTeacherOptimizer]:
    model: IrisArmTeacherModel
    if arm_id == "B":
        model = make_b_harness().model
    else:
        untyped_model, _, _ = _build_offline_model()
        model = cast(IrisTeacherForcingModel, untyped_model)
    return model, IrisTeacherOptimizer(model=model, config=_optimizer_config())


def _manifest() -> IrisExperimentManifestV1:
    return build_experiment_manifest_v1(
        dataset_id="arm-labelled-offline-v2",
        examples=(
            ExperimentExampleV1(
                example_id="train-1",
                split="train",
                prompt=_PROMPT,
                target="short target",
            ),
            ExperimentExampleV1(
                example_id="validation-1",
                split="validation",
                prompt="offline validation prompt",
                target="short target",
            ),
        ),
    )


def _prepared(manifest: IrisExperimentManifestV1) -> IrisPreparedExperimentExample:
    return IrisPreparedExperimentExample(
        dataset_id=manifest.payload.dataset_id,
        manifest_payload_sha256=manifest.payload_sha256,
        split="train",
        split_ordinal=1,
        split_example_count=1,
        example_id="train-1",
        teacher_input=IrisTeacherForcingInput(
            prompt=_PROMPT,
            target_input_ids=torch.tensor([[5, _TERMINAL_TOKEN_ID]], dtype=torch.int64),
            target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        ),
    )


def _summary() -> IrisTokenWeightedEvaluationSummary:
    mean_loss = 0.5
    return IrisTokenWeightedEvaluationSummary(
        example_count=1,
        supervised_token_count=2,
        mean_loss=mean_loss,
        perplexity=math.exp(mean_loss),
    )


def _create(
    *,
    root: Path,
    arm_id: IrisTeacherArmId,
    run_id: str,
    execution_policy: ExperimentManualExecutionPolicyV1
    | ExperimentEpochExecutionPolicyV1 = _MANUAL_POLICY,
    deterministic: bool = False,
) -> tuple[
    IrisArmTeacherModel,
    IrisTeacherOptimizer,
    IrisExperimentManifestV1,
    IrisExperimentLogCreateResultV2,
]:
    torch.manual_seed(_SEED)
    model, optimizer = _owner(arm_id)
    manifest = _manifest()
    created = create_experiment_log_v2(
        run_id=run_id,
        arm_id=arm_id,
        manifest=manifest,
        terminal_token_id=_TERMINAL_TOKEN_ID,
        initialization_seed=_SEED,
        deterministic=deterministic,
        execution_policy=execution_policy,
        model=model,
        optimizer=optimizer,
        root=root,
    )
    return model, optimizer, manifest, created


def _tree_snapshot(root: Path) -> tuple[tuple[str, bytes | None], ...]:
    if not root.exists():
        return ()
    entries: list[tuple[str, bytes | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        entries.append((relative, None if path.is_dir() else path.read_bytes()))
    return tuple(entries)


def test_event_staging_is_outside_authoritative_chain_and_cleanup_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer, manifest, created = _create(
        root=tmp_path,
        arm_id="D",
        run_id="external-event-staging",
    )
    prepared = _prepared(manifest)
    output = optimizer.step(prepared.teacher_input)
    observed_source: Path | None = None

    def fail_link(source: Path, target: Path) -> None:
        nonlocal observed_source
        observed_source = Path(source)
        assert Path(target).parent == created.path / "events"
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(IrisExperimentLogError, match="could not be installed atomically"):
        record_experiment_training_step_v2(
            path=created.path,
            manifest=manifest,
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
        )

    assert observed_source is not None
    assert observed_source.parent == created.path.parent
    assert not observed_source.exists()
    assert read_experiment_log_v2(created.path).event_count == 0

    simulated_crash_orphan = created.path.parent / ".iris-event-simulated-crash.tmp"
    simulated_crash_orphan.write_bytes(b"uncommitted")
    assert read_experiment_log_v2(created.path).event_count == 0


def test_transient_windows_directory_rename_denials_are_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rename = os.rename
    remaining_failures = {"checkpoint": 2, "run": 2}
    attempts = {"checkpoint": 0, "run": 0}

    def flaky_rename(source: Path, target: Path) -> None:
        target_path = Path(target)
        kind: str | None = None
        if target_path.parent == tmp_path / "teacher" / "v1":
            kind = "checkpoint"
        elif target_path.parent == tmp_path / "experiments" / "v2":
            kind = "run"
        if kind is not None:
            attempts[kind] += 1
            if remaining_failures[kind] > 0:
                remaining_failures[kind] -= 1
                raise PermissionError(5, "simulated transient access denial", str(target))
        original_rename(source, target)

    monkeypatch.setattr(os, "rename", flaky_rename)
    _, _, _, created = _create(
        root=tmp_path,
        arm_id="D",
        run_id="permission-retry",
    )

    assert created.path.is_dir()
    assert attempts == {"checkpoint": 3, "run": 3}
    assert not tuple(tmp_path.rglob(".iris-*.tmp"))


@pytest.mark.parametrize(
    ("arm_id", "checkpoint_version", "reference_type"),
    [
        ("B", 2, ExperimentArmBCheckpointReferenceV2),
        ("D", 1, ExperimentArmDCheckpointReferenceV2),
    ],
)
def test_create_read_and_reuse_preserve_arm_checkpoint_identity(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
    reference_type: type[ExperimentArmBCheckpointReferenceV2 | ExperimentArmDCheckpointReferenceV2],
) -> None:
    with torch.random.fork_rng(devices=[]):
        model, optimizer, manifest, created = _create(
            root=tmp_path,
            arm_id=arm_id,
            run_id=f"arm-{arm_id.lower()}-create",
        )
        repeated = create_experiment_log_v2(
            run_id=f"arm-{arm_id.lower()}-create",
            arm_id=arm_id,
            manifest=manifest,
            terminal_token_id=_TERMINAL_TOKEN_ID,
            initialization_seed=_SEED,
            deterministic=False,
            execution_policy=_MANUAL_POLICY,
            model=model,
            optimizer=optimizer,
            root=tmp_path,
        )
        snapshot = read_experiment_log_v2(created.path)

    assert type(created) is IrisExperimentLogCreateResultV2
    assert created.arm_id == snapshot.arm_id == arm_id
    assert created.path == tmp_path / "experiments" / "v2" / f"arm-{arm_id.lower()}-create"
    assert created.initial_checkpoint_path == (
        tmp_path / "teacher" / f"v{checkpoint_version}" / created.initial_checkpoint_payload_sha256
    )
    assert created.initial_checkpoint_path.is_dir()
    assert type(snapshot.run.payload.initial_checkpoint) is reference_type
    assert snapshot.run.payload.initial_checkpoint.checkpoint_schema_version == checkpoint_version
    assert snapshot.run.payload.configuration.model.arm_id == arm_id
    assert snapshot.event_count == 0
    assert not created.reused_existing
    assert repeated.reused_existing
    assert repeated == IrisExperimentLogCreateResultV2(
        arm_id=arm_id,
        path=created.path,
        run_payload_sha256=created.run_payload_sha256,
        configuration_sha256=created.configuration_sha256,
        initial_checkpoint_path=created.initial_checkpoint_path,
        initial_checkpoint_payload_sha256=created.initial_checkpoint_payload_sha256,
        reused_existing=True,
    )


@pytest.mark.parametrize(("arm_id", "checkpoint_version"), [("B", 2), ("D", 1)])
def test_training_checkpoint_evaluation_chain_and_exact_retries(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
) -> None:
    with torch.random.fork_rng(devices=[]):
        model, optimizer, manifest, created = _create(
            root=tmp_path,
            arm_id=arm_id,
            run_id=f"arm-{arm_id.lower()}-flow",
        )
        prepared = _prepared(manifest)
        output = optimizer.step(prepared.teacher_input)
        training = record_experiment_training_step_v2(
            path=created.path,
            manifest=manifest,
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
        )
        checkpoint = build_arm_teacher_checkpoint(
            arm_id=arm_id,
            model=model,
            optimizer=optimizer,
        )
        checkpoint_event = record_experiment_checkpoint_v2(
            path=created.path,
            checkpoint=checkpoint,
        )
        evaluation = record_experiment_evaluation_v2(
            path=created.path,
            manifest=manifest,
            split="validation",
            summary=_summary(),
            model=model,
            optimizer=optimizer,
        )

        repeated_training = record_experiment_training_step_v2(
            path=created.path,
            manifest=manifest,
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
        )
        repeated_checkpoint = record_experiment_checkpoint_v2(
            path=created.path,
            checkpoint=checkpoint,
        )
        repeated_evaluation = record_experiment_evaluation_v2(
            path=created.path,
            manifest=manifest,
            split="validation",
            summary=_summary(),
            model=model,
            optimizer=optimizer,
        )
        snapshot = read_experiment_log_v2(created.path)

    assert [event.payload.sequence for event in snapshot.events] == [1, 2, 3]
    assert [event.payload.record.event_type for event in snapshot.events] == [
        "training_step",
        "checkpoint",
        "evaluation",
    ]
    assert type(snapshot.events[0].payload.record) is ExperimentTrainingStepMetricV1
    assert type(snapshot.events[1].payload.record) is ExperimentCheckpointEventV2
    assert type(snapshot.events[2].payload.record) is ExperimentEvaluationMetricV1
    checkpoint_record = snapshot.events[1].payload.record
    assert checkpoint_record.checkpoint.arm_id == arm_id
    assert checkpoint_record.checkpoint.checkpoint_schema_version == checkpoint_version
    assert checkpoint_event.referenced_checkpoint_path == (
        tmp_path / "teacher" / f"v{checkpoint_version}" / checkpoint.identity.payload_sha256
    )
    assert checkpoint_event.referenced_checkpoint_path is not None
    assert checkpoint_event.referenced_checkpoint_path.is_dir()
    assert snapshot.optimizer_completed_steps == 1
    assert snapshot.sidecar_state_sha256 == checkpoint.identity.sidecar_state_sha256
    assert not training.reused_existing
    assert not checkpoint_event.reused_existing
    assert not evaluation.reused_existing
    assert repeated_training.sequence == training.sequence == 1
    assert repeated_checkpoint.sequence == checkpoint_event.sequence == 2
    assert repeated_evaluation.sequence == evaluation.sequence == 3
    assert repeated_training.reused_existing
    assert repeated_checkpoint.reused_existing
    assert repeated_evaluation.reused_existing
    assert tuple(path.name for path in sorted((created.path / "events").iterdir())) == (
        "00000001.json",
        "00000002.json",
        "00000003.json",
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_create_reuses_same_immutable_run_after_events_exist(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
) -> None:
    with torch.random.fork_rng(devices=[]):
        model, optimizer, manifest, created = _create(
            root=tmp_path,
            arm_id=arm_id,
            run_id=f"arm-{arm_id.lower()}-progressed-reuse",
        )
        prepared = _prepared(manifest)
        output = optimizer.step(prepared.teacher_input)
        record_experiment_training_step_v2(
            path=created.path,
            manifest=manifest,
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
        )

        torch.manual_seed(_SEED)
        initial_model, initial_optimizer = _owner(arm_id)
        repeated = create_experiment_log_v2(
            run_id=f"arm-{arm_id.lower()}-progressed-reuse",
            arm_id=arm_id,
            manifest=manifest,
            terminal_token_id=_TERMINAL_TOKEN_ID,
            initialization_seed=_SEED,
            deterministic=False,
            execution_policy=_MANUAL_POLICY,
            model=initial_model,
            optimizer=initial_optimizer,
            root=tmp_path,
        )
        snapshot = read_experiment_log_v2(created.path)

    assert repeated.reused_existing
    assert repeated.path == created.path
    assert repeated.run_payload_sha256 == created.run_payload_sha256
    assert snapshot.event_count == 1


@pytest.mark.parametrize("requested_arm", ["B", "D"])
def test_cross_arm_create_rejects_before_filesystem_mutation(
    tmp_path: Path,
    requested_arm: IrisTeacherArmId,
) -> None:
    other_arm: IrisTeacherArmId = "D" if requested_arm == "B" else "B"
    root = tmp_path / "cross-arm-create"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SEED)
        wrong_model, wrong_optimizer = _owner(other_arm)
        with pytest.raises(IrisExperimentLogError, match="could not be created"):
            create_experiment_log_v2(
                run_id="cross-arm-create",
                arm_id=requested_arm,
                manifest=_manifest(),
                terminal_token_id=_TERMINAL_TOKEN_ID,
                initialization_seed=_SEED,
                deterministic=False,
                execution_policy=_MANUAL_POLICY,
                model=wrong_model,
                optimizer=wrong_optimizer,
                root=root,
            )

    assert not root.exists()


@pytest.mark.parametrize("run_arm", ["B", "D"])
def test_cross_arm_checkpoint_rejects_before_any_persistence(
    tmp_path: Path,
    run_arm: IrisTeacherArmId,
) -> None:
    other_arm: IrisTeacherArmId = "D" if run_arm == "B" else "B"
    with torch.random.fork_rng(devices=[]):
        _, _, _, created = _create(
            root=tmp_path,
            arm_id=run_arm,
            run_id=f"arm-{run_arm.lower()}-cross-checkpoint",
        )
        torch.manual_seed(_SEED + 1)
        other_model, other_optimizer = _owner(other_arm)
        wrong_checkpoint = build_arm_teacher_checkpoint(
            arm_id=other_arm,
            model=other_model,
            optimizer=other_optimizer,
        )
        before = _tree_snapshot(tmp_path)
        with pytest.raises(IrisExperimentLogError, match="could not be recorded"):
            record_experiment_checkpoint_v2(
                path=created.path,
                checkpoint=wrong_checkpoint,
            )

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_checkpoint_run_validator_rejects_optimizer_configuration_mismatch(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
) -> None:
    _, _, _, created = _create(
        root=tmp_path,
        arm_id=arm_id,
        run_id=f"configuration-mismatch-{arm_id.lower()}",
    )
    snapshot = read_experiment_log_v2(created.path)
    other_model, _ = _owner(arm_id)
    other_optimizer = IrisTeacherOptimizer(
        model=other_model,
        config=IrisAdamWConfig(
            learning_rate=2e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        ),
    )
    checkpoint = build_arm_teacher_checkpoint(
        arm_id=arm_id,
        model=other_model,
        optimizer=other_optimizer,
    )
    before = _tree_snapshot(tmp_path)

    with pytest.raises(IrisExperimentLogError, match="checkpoint/run validation failed"):
        validate_experiment_checkpoint_for_run_v2(
            checkpoint=checkpoint,
            run=snapshot.run,
        )

    assert _tree_snapshot(tmp_path) == before
    assert read_experiment_log_v2(created.path).event_count == 0


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_reader_rejects_tampered_run_and_noncanonical_event(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
) -> None:
    with torch.random.fork_rng(devices=[]):
        model, optimizer, manifest, created = _create(
            root=tmp_path,
            arm_id=arm_id,
            run_id=f"arm-{arm_id.lower()}-tamper",
        )
        prepared = _prepared(manifest)
        output = optimizer.step(prepared.teacher_input)
        recorded = record_experiment_training_step_v2(
            path=created.path,
            manifest=manifest,
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
        )

    run_path = created.path / "run.json"
    original_run = run_path.read_bytes()
    raw_run = json.loads(original_run)
    cast(dict[str, object], raw_run)["payload_sha256"] = "0" * 64
    run_path.write_bytes(canonical_json_file_bytes(raw_run))
    with pytest.raises(IrisExperimentLogError, match="could not be read or verified"):
        read_experiment_log_v2(created.path)

    run_path.write_bytes(original_run)
    event_path = recorded.path
    original_event = event_path.read_bytes()
    event_path.write_bytes(original_event[:-1] + b" \n")
    with pytest.raises(IrisExperimentLogError, match="could not be read or verified"):
        read_experiment_log_v2(created.path)


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_managed_exact_head_retry_is_idempotent_not_a_new_transition(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
) -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            model, optimizer, manifest, created = _create(
                root=tmp_path,
                arm_id=arm_id,
                run_id=f"arm-{arm_id.lower()}-managed-retry",
                execution_policy=_MANAGED_POLICY,
                deterministic=True,
            )
            initial_head = read_experiment_log_v2(created.path).chain_head_sha256
            prepared = _prepared(manifest)
            output = optimizer.step(prepared.teacher_input)
            first = record_experiment_training_step_v2(
                path=created.path,
                manifest=manifest,
                prepared=prepared,
                output=output,
                model=model,
                optimizer=optimizer,
                expected_predecessor_sha256=initial_head,
            )
            repeated = record_experiment_training_step_v2(
                path=created.path,
                manifest=manifest,
                prepared=prepared,
                output=output,
                model=model,
                optimizer=optimizer,
                expected_predecessor_sha256=initial_head,
            )
            snapshot = read_experiment_log_v2(created.path)
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )

    assert not first.reused_existing
    assert repeated.reused_existing
    assert repeated.sequence == first.sequence == 1
    assert repeated.payload_sha256 == first.payload_sha256
    assert snapshot.event_count == 1
