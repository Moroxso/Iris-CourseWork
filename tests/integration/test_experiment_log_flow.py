"""Offline end-to-end experiment journal integration tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch

import iris.application.experiment_log as experiment_log_module
from iris.application.errors import IrisExperimentLogError
from iris.application.experiment_log import (
    IrisExperimentEventWriteResult,
    IrisExperimentLogCreateResult,
    build_experiment_run_v1,
    create_experiment_log_v1,
    open_experiment_log_writer_v1,
    read_experiment_log_v1,
    record_experiment_checkpoint_v1,
    record_experiment_evaluation_v1,
    record_experiment_training_step_v1,
)
from iris.application.experiment_manifest import (
    IrisPreparedExperimentExample,
    iterate_experiment_split_v1,
)
from iris.application.training_checkpoint import build_teacher_checkpoint_v1
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationAccumulator,
    IrisTokenWeightedEvaluationSummary,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from tests.integration.test_experiment_iteration import _build, _manifest

pytestmark = pytest.mark.integration

_MANUAL_EXECUTION = ExperimentManualExecutionPolicyV1(mode="manual_event_chain")


def test_native_windows_denial_retries_publication_without_recreating_checkpoint_or_run(
    tmp_path: Path,
) -> None:
    from iris.application import training_checkpoint as checkpoint_module
    from iris.utils import directory_publication as publication

    real_rename = os.rename
    attempts: dict[Path, list[Path]] = {}

    def deny_once(source: Path, target: Path) -> None:
        seen = attempts.setdefault(target, [])
        seen.append(source)
        if len(seen) == 1:
            error = PermissionError(13, "synthetic native access denial")
            error.winerror = 5
            raise error
        real_rename(source, target)

    with (
        torch.random.fork_rng(devices=[]),
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(os, "rename", side_effect=deny_once),
        patch.object(
            checkpoint_module, "_write_exact_file", wraps=checkpoint_module._write_exact_file
        ) as checkpoint_writes,
        patch.object(
            experiment_log_module,
            "_write_exact_file",
            wraps=experiment_log_module._write_exact_file,
        ) as run_writes,
    ):
        torch.manual_seed(1907)
        _, optimizer, created = _create(tmp_path)
    assert checkpoint_writes.call_count == 2
    assert run_writes.call_count == 1
    assert len(attempts) == 2
    assert all(len(seen) >= 2 and len(set(seen)) == 1 for seen in attempts.values())
    assert optimizer.completed_steps == 0
    assert read_experiment_log_v1(created.path).event_count == 0


def _optimizer(model: IrisTeacherForcingModel) -> IrisTeacherOptimizer:
    return IrisTeacherOptimizer(
        model=model,
        config=IrisAdamWConfig(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        ),
    )


def _create(
    tmp_path: Path,
) -> tuple[IrisTeacherForcingModel, IrisTeacherOptimizer, IrisExperimentLogCreateResult]:
    teacher, _ = _build()
    optimizer = _optimizer(teacher)
    created = create_experiment_log_v1(
        run_id="offline-run-1",
        manifest=_manifest(),
        terminal_token_id=15,
        initialization_seed=1907,
        deterministic=False,
        execution_policy=_MANUAL_EXECUTION,
        model=teacher,
        optimizer=optimizer,
        root=tmp_path,
    )
    return teacher, optimizer, created


def _next_training_example(
    teacher: IrisTeacherForcingModel,
) -> IrisPreparedExperimentExample:
    return next(
        iterate_experiment_split_v1(
            manifest=_manifest(),
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )
    )


def _validation_summary(teacher: IrisTeacherForcingModel) -> IrisTokenWeightedEvaluationSummary:
    teacher.eval()
    accumulator = IrisTokenWeightedEvaluationAccumulator()
    for prepared in iterate_experiment_split_v1(
        manifest=_manifest(),
        split="validation",
        backbone=teacher._backbone,
        terminal_token_id=15,
    ):
        accumulator.update(teacher.evaluation_forward(prepared.teacher_input))
    summary = accumulator.summary()
    teacher.train()
    return summary


def _tamper_json(path: Path, mutate: object) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cast(dict[str, object], raw)["payload_sha256"] = mutate
    path.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",
    )


def test_create_log_persists_exact_initial_checkpoint_and_private_run_identity(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        repeated = create_experiment_log_v1(
            run_id="offline-run-1",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=False,
            execution_policy=_MANUAL_EXECUTION,
            model=teacher,
            optimizer=optimizer,
            root=tmp_path,
        )
        snapshot = read_experiment_log_v1(created.path)

    assert type(created) is IrisExperimentLogCreateResult
    assert created.path == tmp_path / "experiments" / "v1" / "offline-run-1"
    assert created.initial_checkpoint_path.is_dir()
    assert (
        created.initial_checkpoint_payload_sha256
        == snapshot.run.payload.initial_checkpoint.payload_sha256
    )
    assert created.run_payload_sha256 == snapshot.run.payload_sha256
    assert created.configuration_sha256 == snapshot.run.payload.configuration.sha256
    assert not created.reused_existing
    assert repeated.reused_existing
    assert repeated == IrisExperimentLogCreateResult(
        path=created.path,
        run_payload_sha256=created.run_payload_sha256,
        configuration_sha256=created.configuration_sha256,
        initial_checkpoint_path=created.initial_checkpoint_path,
        initial_checkpoint_payload_sha256=created.initial_checkpoint_payload_sha256,
        reused_existing=True,
    )
    assert snapshot.event_count == 0
    assert snapshot.optimizer_completed_steps == 0
    assert (
        snapshot.sidecar_state_sha256
        == snapshot.run.payload.initial_checkpoint.sidecar_state_sha256
    )
    run_text = (created.path / "run.json").read_text(encoding="utf-8")
    assert "offline teacher prompt" not in run_text
    assert "short target" not in run_text
    assert "longer target" not in run_text
    assert "target" in run_text  # target-preparation policy only, never target text


def test_build_run_is_deterministic_and_terminal_policy_changes_identity(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, _ = _create(tmp_path)
        checkpoint = build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)
        first = build_experiment_run_v1(
            run_id="deterministic-run",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=_MANUAL_EXECUTION,
            initial_checkpoint=checkpoint,
        )
        second = build_experiment_run_v1(
            run_id="deterministic-run",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=_MANUAL_EXECUTION,
            initial_checkpoint=checkpoint,
        )
        different_terminal = build_experiment_run_v1(
            run_id="deterministic-run",
            manifest=_manifest(),
            terminal_token_id=14,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=_MANUAL_EXECUTION,
            initial_checkpoint=checkpoint,
        )
        different_seed = build_experiment_run_v1(
            run_id="deterministic-run",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1908,
            deterministic=True,
            execution_policy=_MANUAL_EXECUTION,
            initial_checkpoint=checkpoint,
        )
        epoch_execution = build_experiment_run_v1(
            run_id="deterministic-run",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=ExperimentEpochExecutionPolicyV1(
                mode="deterministic_epochs",
                epoch_count=1,
                training_order="manifest_order_each_epoch",
                checkpoint_policy="after_each_complete_epoch",
                validation_policy="after_each_complete_epoch",
                evaluate_test_after_training=True,
                resume_policy="latest_checkpoint_plus_verified_replay",
            ),
            initial_checkpoint=checkpoint,
        )

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert different_terminal.payload.configuration.sha256 != first.payload.configuration.sha256
    assert different_terminal.payload_sha256 != first.payload_sha256
    assert different_seed.payload.configuration.sha256 != first.payload.configuration.sha256
    assert different_seed.payload_sha256 != first.payload_sha256
    assert epoch_execution.payload.configuration.sha256 != first.payload.configuration.sha256
    assert epoch_execution.payload_sha256 != first.payload_sha256


@pytest.mark.parametrize(
    ("initialization_seed", "deterministic"),
    [(1908, False), (1907, True)],
)
def test_creation_rejects_ambient_runtime_policy_mismatch_before_filesystem_change(
    tmp_path: Path,
    initialization_seed: int,
    deterministic: bool,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, _ = _build()
        optimizer = _optimizer(teacher)

        with pytest.raises(IrisExperimentLogError, match="created"):
            create_experiment_log_v1(
                run_id="policy-mismatch",
                manifest=_manifest(),
                terminal_token_id=15,
                initialization_seed=initialization_seed,
                deterministic=deterministic,
                execution_policy=_MANUAL_EXECUTION,
                model=teacher,
                optimizer=optimizer,
                root=tmp_path,
            )

    assert not (tmp_path / "teacher").exists()
    assert not (tmp_path / "experiments").exists()


def test_managed_creation_rejects_warn_only_determinism_before_filesystem_change(
    tmp_path: Path,
) -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1907)
            torch.use_deterministic_algorithms(True, warn_only=True)
            teacher, _ = _build()
            optimizer = _optimizer(teacher)

            with pytest.raises(IrisExperimentLogError, match="created"):
                create_experiment_log_v1(
                    run_id="warn-only-policy",
                    manifest=_manifest(),
                    terminal_token_id=15,
                    initialization_seed=1907,
                    deterministic=True,
                    execution_policy=ExperimentEpochExecutionPolicyV1(
                        mode="deterministic_epochs",
                        epoch_count=1,
                        training_order="manifest_order_each_epoch",
                        checkpoint_policy="after_each_complete_epoch",
                        validation_policy="after_each_complete_epoch",
                        evaluate_test_after_training=True,
                        resume_policy="latest_checkpoint_plus_verified_replay",
                    ),
                    model=teacher,
                    optimizer=optimizer,
                    root=tmp_path,
                )
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )

    assert not (tmp_path / "teacher").exists()
    assert not (tmp_path / "experiments").exists()


def test_training_evaluation_checkpoint_chain_is_exact_and_idempotent(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        iterator = iterate_experiment_split_v1(
            manifest=_manifest(),
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )
        first_prepared = next(iterator)
        first_output = optimizer.step(first_prepared.teacher_input)
        train_event = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=first_prepared,
            output=first_output,
            model=teacher,
            optimizer=optimizer,
        )
        summary = _validation_summary(teacher)
        evaluation_event = record_experiment_evaluation_v1(
            path=created.path,
            manifest=_manifest(),
            split="validation",
            summary=summary,
            model=teacher,
            optimizer=optimizer,
        )
        checkpoint = build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)
        checkpoint_event = record_experiment_checkpoint_v1(
            path=created.path,
            checkpoint=checkpoint,
        )
        repeated_train = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=first_prepared,
            output=first_output,
            model=teacher,
            optimizer=optimizer,
        )
        repeated_evaluation = record_experiment_evaluation_v1(
            path=created.path,
            manifest=_manifest(),
            split="validation",
            summary=summary,
            model=teacher,
            optimizer=optimizer,
        )
        repeated_checkpoint = record_experiment_checkpoint_v1(
            path=created.path,
            checkpoint=checkpoint,
        )
        snapshot = read_experiment_log_v1(created.path)

    assert [event.payload.sequence for event in snapshot.events] == [1, 2, 3]
    assert [event.payload.record.event_type for event in snapshot.events] == [
        "training_step",
        "evaluation",
        "checkpoint",
    ]
    assert snapshot.optimizer_completed_steps == 1
    assert snapshot.sidecar_state_sha256 == checkpoint.manifest.payload.model.sha256
    assert train_event.sequence == repeated_train.sequence == 1
    assert evaluation_event.sequence == repeated_evaluation.sequence == 2
    assert checkpoint_event.sequence == repeated_checkpoint.sequence == 3
    assert not train_event.reused_existing
    assert repeated_train.reused_existing
    assert repeated_evaluation.reused_existing
    assert repeated_checkpoint.reused_existing
    assert checkpoint_event.referenced_checkpoint_path is not None
    assert checkpoint_event.referenced_checkpoint_path == (
        tmp_path / "teacher" / "v1" / checkpoint.manifest.sha256
    )
    assert checkpoint_event.referenced_checkpoint_path.is_dir()
    assert tuple(path.name for path in sorted((created.path / "events").iterdir())) == (
        "00000001.json",
        "00000002.json",
        "00000003.json",
    )
    assert isinstance(snapshot.events[0].payload.record, ExperimentTrainingStepMetricV1)
    assert isinstance(snapshot.events[1].payload.record, ExperimentEvaluationMetricV1)
    assert isinstance(snapshot.events[2].payload.record, ExperimentCheckpointEventV1)
    for event_path in (created.path / "events").iterdir():
        text = event_path.read_text(encoding="utf-8")
        assert "offline teacher prompt" not in text
        assert "short target" not in text
        assert "longer target" not in text


def test_two_training_steps_must_form_exact_counter_and_hash_chain(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        iterator = iterate_experiment_split_v1(
            manifest=_manifest(),
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )
        results: list[IrisExperimentEventWriteResult] = []
        for prepared in iterator:
            output = optimizer.step(prepared.teacher_input)
            results.append(
                record_experiment_training_step_v1(
                    path=created.path,
                    manifest=_manifest(),
                    prepared=prepared,
                    output=output,
                    model=teacher,
                    optimizer=optimizer,
                )
            )
        snapshot = read_experiment_log_v1(created.path)

    assert [result.sequence for result in results] == [1, 2]
    assert [event.payload.optimizer_completed_steps for event in snapshot.events] == [1, 2]
    assert snapshot.events[0].payload.predecessor_sha256 == snapshot.run.payload_sha256
    assert snapshot.events[1].payload.predecessor_sha256 == snapshot.events[0].payload_sha256
    assert snapshot.optimizer_completed_steps == optimizer.completed_steps == 2


def test_training_record_rejects_wrong_example_or_stale_output(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        wrong_prepared = IrisPreparedExperimentExample(
            dataset_id=prepared.dataset_id,
            manifest_payload_sha256=prepared.manifest_payload_sha256,
            split=prepared.split,
            split_ordinal=2,
            split_example_count=prepared.split_example_count,
            example_id=prepared.example_id,
            teacher_input=prepared.teacher_input,
        )
        with pytest.raises(IrisExperimentLogError, match="training-step"):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=wrong_prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
            )
        stale = IrisTeacherOptimizationOutput(
            step_index=2,
            loss=output.loss,
            supervised_token_count=output.supervised_token_count,
            gradient_l2_norm=output.gradient_l2_norm,
            gradients_clipped=output.gradients_clipped,
        )
        with pytest.raises(IrisExperimentLogError, match="training-step"):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=stale,
                model=teacher,
                optimizer=optimizer,
            )
        wrong_token_count = IrisTeacherOptimizationOutput(
            step_index=output.step_index,
            loss=output.loss,
            supervised_token_count=output.supervised_token_count - 1,
            gradient_l2_norm=output.gradient_l2_norm,
            gradients_clipped=output.gradients_clipped,
        )
        with pytest.raises(IrisExperimentLogError, match="training-step"):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=wrong_token_count,
                model=teacher,
                optimizer=optimizer,
            )

    assert read_experiment_log_v1(created.path).event_count == 0


def test_evaluation_requires_complete_nonempty_split_and_unchanged_state(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        incomplete = IrisTokenWeightedEvaluationSummary(
            example_count=2,
            supervised_token_count=2,
            mean_loss=1.0,
            perplexity=math.exp(1.0),
        )
        with pytest.raises(IrisExperimentLogError, match="evaluation metric"):
            record_experiment_evaluation_v1(
                path=created.path,
                manifest=_manifest(),
                split="validation",
                summary=incomplete,
                model=teacher,
                optimizer=optimizer,
            )
        with pytest.raises(IrisExperimentLogError, match="evaluation metric"):
            record_experiment_evaluation_v1(
                path=created.path,
                manifest=_manifest(include_test=False),
                split="test",
                summary=cast(IrisTokenWeightedEvaluationSummary, object()),
                model=teacher,
                optimizer=optimizer,
            )

    assert read_experiment_log_v1(created.path).event_count == 0


def test_checkpoint_event_rejects_stale_checkpoint_after_next_step(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        initial_checkpoint = build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=teacher,
            optimizer=optimizer,
        )

        with pytest.raises(IrisExperimentLogError, match="checkpoint event"):
            record_experiment_checkpoint_v1(
                path=created.path,
                checkpoint=initial_checkpoint,
            )

    snapshot = read_experiment_log_v1(created.path)
    assert snapshot.event_count == 1
    assert snapshot.optimizer_completed_steps == 1


@pytest.mark.parametrize("target", ["run", "event"])
def test_reader_rejects_hash_tampering(tmp_path: Path, target: str) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        if target == "event":
            prepared = _next_training_example(teacher)
            output = optimizer.step(prepared.teacher_input)
            recorded = record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
            )
            tampered = recorded.path
        else:
            tampered = created.path / "run.json"
        _tamper_json(tampered, "f" * 64)

        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)


def test_reader_rejects_event_gaps_extras_and_noncanonical_bytes(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        result = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=teacher,
            optimizer=optimizer,
        )
        result.path.rename(result.path.with_name("00000002.json"))

        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)

        result.path.with_name("00000002.json").rename(result.path)
        (created.path / "extra.txt").write_text("extra", encoding="utf-8")
        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)

        (created.path / "extra.txt").unlink()
        result.path.write_bytes(result.path.read_bytes() + b"\n")
        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        _, _, created = _create(tmp_path)
        run_path = created.path / "run.json"
        content = run_path.read_bytes()
        needle = b'"artifact_type":"iris.experiment_run"'
        duplicate = needle + b',"artifact_type":"iris.experiment_run"'
        assert content.count(needle) == 1
        run_path.write_bytes(content.replace(needle, duplicate, 1))

        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)


@pytest.mark.parametrize("tamper", ["predecessor", "counter"])
def test_reader_rejects_rehashed_but_semantically_broken_event_chain(
    tmp_path: Path,
    tamper: str,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        recorded = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=teacher,
            optimizer=optimizer,
        )
        raw = cast(dict[str, object], json.loads(recorded.path.read_text(encoding="utf-8")))
        payload = cast(dict[str, object], raw["payload"])
        if tamper == "predecessor":
            payload["predecessor_sha256"] = "f" * 64
        else:
            payload["optimizer_completed_steps"] = 2
            record = cast(dict[str, object], payload["record"])
            record["step_index"] = 2
        raw["payload_sha256"] = domain_sha256(
            "iris/experiment-log/v1/event-payload",
            canonical_json_bytes(payload),
        )
        recorded.path.write_bytes(canonical_json_file_bytes(raw))

        with pytest.raises(IrisExperimentLogError):
            read_experiment_log_v1(created.path)


def test_event_install_handles_concurrent_identical_winner_and_cleans_temp(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)

        def install_winner(source: Path, target: Path) -> None:
            target.write_bytes(source.read_bytes())
            raise FileExistsError

        with patch.object(os, "link", side_effect=install_winner):
            result = record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
            )

    assert result.reused_existing
    assert read_experiment_log_v1(created.path).event_count == 1
    assert not tuple((created.path / "events").glob(".iris-*.tmp"))


def test_event_install_failure_leaves_no_partial_event_or_temporary_file(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)

        with (
            patch.object(os, "link", side_effect=OSError("synthetic link failure")),
            pytest.raises(IrisExperimentLogError, match="atomically"),
        ):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
            )

    assert read_experiment_log_v1(created.path).event_count == 0
    assert not tuple((created.path / "events").glob(".iris-*.tmp"))


def test_run_id_conflict_never_overwrites_existing_log(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        original = (created.path / "run.json").read_bytes()

        with pytest.raises(IrisExperimentLogError, match="conflict"):
            create_experiment_log_v1(
                run_id="offline-run-1",
                manifest=_manifest(),
                terminal_token_id=14,
                initialization_seed=1907,
                deterministic=False,
                execution_policy=_MANUAL_EXECUTION,
                model=teacher,
                optimizer=optimizer,
                root=tmp_path,
            )

    assert (created.path / "run.json").read_bytes() == original
    assert read_experiment_log_v1(created.path).event_count == 0


def test_open_writer_appends_and_reuses_without_full_journal_rereads(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        writer = open_experiment_log_writer_v1(created.path)

        with patch.object(
            experiment_log_module,
            "read_experiment_log_v1",
            wraps=read_experiment_log_v1,
        ) as full_reader:
            recorded = record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
                writer=writer,
            )
            repeated = record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
                writer=writer,
            )

    assert full_reader.call_count == 0
    assert not recorded.reused_existing
    assert repeated.reused_existing
    assert repeated.sequence == recorded.sequence == 1
    assert writer.snapshot.event_count == 1
    assert read_experiment_log_v1(created.path) == writer.snapshot


def test_writer_refresh_discovers_and_verifies_a_concurrent_suffix(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        writer = open_experiment_log_writer_v1(created.path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        recorded = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=teacher,
            optimizer=optimizer,
        )

        assert writer.snapshot.event_count == 0
        refreshed = writer.refresh()

    assert refreshed.event_count == 1
    assert refreshed.chain_head_sha256 == recorded.payload_sha256
    assert writer.snapshot == read_experiment_log_v1(created.path)


def test_writer_poisoned_by_head_tamper_must_be_reopened(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        writer = open_experiment_log_writer_v1(created.path)
        recorded = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=teacher,
            optimizer=optimizer,
            writer=writer,
        )
        _tamper_json(recorded.path, "f" * 64)

        with pytest.raises(IrisExperimentLogError, match="refreshed"):
            writer.refresh()

    assert writer.poisoned
    with pytest.raises(IrisExperimentLogError, match="poisoned"):
        _ = writer.snapshot


def test_uncertain_writer_install_failure_poisoned_and_leaves_no_event(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path)
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        writer = open_experiment_log_writer_v1(created.path)

        with (
            patch.object(
                experiment_log_module,
                "_write_exact_file",
                side_effect=OSError("synthetic write failure"),
            ),
            pytest.raises(IrisExperimentLogError, match="atomically"),
        ):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
                writer=writer,
            )

    assert writer.poisoned
    assert read_experiment_log_v1(created.path).event_count == 0
    assert not tuple((created.path / "events").glob(".iris-*.tmp"))


def test_writer_cannot_be_used_for_a_different_journal(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, optimizer, created = _create(tmp_path / "first")
        _, _, other = _create(tmp_path / "second")
        prepared = _next_training_example(teacher)
        output = optimizer.step(prepared.teacher_input)
        writer = open_experiment_log_writer_v1(created.path)

        with pytest.raises(IrisExperimentLogError, match="training-step"):
            record_experiment_training_step_v1(
                path=other.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=teacher,
                optimizer=optimizer,
                writer=writer,
            )

    assert not writer.poisoned
    assert read_experiment_log_v1(created.path).event_count == 0
    assert read_experiment_log_v1(other.path).event_count == 0
