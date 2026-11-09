"""Offline managed-epoch execution, interruption, and exact replay tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import iris.application.experiment_execution as execution_module
import iris.application.experiment_log as experiment_log_module
import iris.application.experiment_runner as runner_module
from iris.application.errors import IrisExperimentLogError, IrisExperimentRunnerError
from iris.application.experiment_execution import (
    IrisExperimentEpochProgressV1,
    analyze_experiment_epoch_prefix_v1,
    analyze_experiment_epoch_progress_v1,
    extend_experiment_epoch_progress_v1,
)
from iris.application.experiment_log import (
    create_experiment_log_v1,
    read_experiment_log_v1,
    record_experiment_checkpoint_v1,
    record_experiment_training_step_v1,
)
from iris.application.experiment_manifest import (
    iterate_experiment_split_v1,
)
from iris.application.experiment_runner import (
    IrisExperimentExecutionResult,
    execute_experiment_v1,
)
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    build_teacher_checkpoint_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentTrainingStepMetricV1,
    IrisExperimentEventV1,
    IrisExperimentRunV1,
)
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from tests.integration.test_experiment_iteration import _build, _manifest

pytestmark = pytest.mark.integration

_EVENT_PAYLOAD_DOMAIN = "iris/experiment-log/v1/event-payload"


@contextmanager
def _deterministic_runtime(seed: int = 1907) -> Iterator[None]:
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


def _fresh_owner() -> tuple[IrisTeacherForcingModel, IrisTeacherOptimizer]:
    torch.manual_seed(1907)
    model, _ = _build()
    return model, _optimizer(model)


def _execute(
    *,
    root: Path,
    run_id: str,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    policy: ExperimentEpochExecutionPolicyV1 | None = None,
    maximum_new_training_steps: int | None = None,
) -> IrisExperimentExecutionResult:
    return execute_experiment_v1(
        run_id=run_id,
        manifest=_manifest(),
        terminal_token_id=15,
        initialization_seed=1907,
        deterministic=True,
        execution_policy=policy or _policy(),
        model=model,
        optimizer=optimizer,
        root=root,
        maximum_new_training_steps=maximum_new_training_steps,
    )


def test_fresh_runner_executes_exact_epoch_checkpoint_validation_and_test(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        result = _execute(
            root=tmp_path,
            run_id="managed-full",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=2,
        )
        snapshot = read_experiment_log_v1(result.path)
        checkpoint = build_teacher_checkpoint_v1(model=model, optimizer=optimizer)
        with pytest.raises(IrisExperimentLogError, match="recorded"):
            record_experiment_checkpoint_v1(
                path=result.path,
                checkpoint=checkpoint,
                manifest=_manifest(),
                expected_predecessor_sha256=snapshot.chain_head_sha256,
            )

    checkpoint_record = snapshot.events[2].payload.record
    assert isinstance(checkpoint_record, ExperimentCheckpointEventV1)
    assert result == IrisExperimentExecutionResult(
        path=tmp_path / "experiments" / "v1" / "managed-full",
        status="complete",
        run_payload_sha256=snapshot.run.payload_sha256,
        configuration_sha256=snapshot.run.payload.configuration.sha256,
        event_count=5,
        optimizer_completed_steps=2,
        completed_epochs=1,
        next_epoch_index=None,
        next_training_ordinal=None,
        latest_checkpoint_payload_sha256=(checkpoint_record.checkpoint.payload_sha256),
        replayed_training_steps=0,
        new_training_steps=2,
    )
    assert [event.payload.record.event_type for event in snapshot.events] == [
        "training_step",
        "training_step",
        "checkpoint",
        "evaluation",
        "evaluation",
    ]
    assert [
        event.payload.record.split
        for event in snapshot.events
        if isinstance(event.payload.record, ExperimentEvaluationMetricV1)
    ] == ["validation", "test"]
    assert isinstance(snapshot.events[0].payload.record, ExperimentTrainingStepMetricV1)
    assert isinstance(snapshot.events[2].payload.record, ExperimentCheckpointEventV1)
    assert model.training
    assert optimizer.completed_steps == 2


def test_mid_epoch_pause_resumes_from_checkpoint_with_exact_replay(
    tmp_path: Path,
) -> None:
    resumed_root = tmp_path / "resumed"
    reference_root = tmp_path / "reference"
    with _deterministic_runtime():
        interrupted_model, interrupted_optimizer = _fresh_owner()
        interrupted = _execute(
            root=resumed_root,
            run_id="managed-resume",
            model=interrupted_model,
            optimizer=interrupted_optimizer,
            maximum_new_training_steps=1,
        )
        torch.manual_seed(1907)
        resumed_model, resumed_tokenizer = _build()
        resumed_optimizer = _optimizer(resumed_model)
        resumed = _execute(
            root=resumed_root,
            run_id="managed-resume",
            model=resumed_model,
            optimizer=resumed_optimizer,
        )
        resumed_checkpoint = build_teacher_checkpoint_v1(
            model=resumed_model,
            optimizer=resumed_optimizer,
        )

        reference_model, reference_optimizer = _fresh_owner()
        reference = _execute(
            root=reference_root,
            run_id="managed-reference",
            model=reference_model,
            optimizer=reference_optimizer,
        )
        reference_checkpoint = build_teacher_checkpoint_v1(
            model=reference_model,
            optimizer=reference_optimizer,
        )

    assert interrupted.status == "step_limit_reached"
    assert interrupted.event_count == 1
    assert interrupted.completed_epochs == 0
    assert interrupted.next_epoch_index == 1
    assert interrupted.next_training_ordinal == 2
    assert resumed.status == reference.status == "complete"
    assert resumed.replayed_training_steps == 1
    assert resumed.new_training_steps == 1
    assert resumed.event_count == reference.event_count == 5
    assert resumed_checkpoint == reference_checkpoint
    assert sum(text == "longer target" for text, _ in resumed_tokenizer.calls) == 2


def test_second_epoch_resume_uses_latest_boundary_and_replays_only_its_tail(
    tmp_path: Path,
) -> None:
    policy = _policy(epoch_count=2)
    with _deterministic_runtime():
        interrupted_model, interrupted_optimizer = _fresh_owner()
        interrupted = _execute(
            root=tmp_path,
            run_id="managed-second-epoch",
            model=interrupted_model,
            optimizer=interrupted_optimizer,
            policy=policy,
            maximum_new_training_steps=3,
        )
        resumed_model, resumed_optimizer = _fresh_owner()
        resumed = _execute(
            root=tmp_path,
            run_id="managed-second-epoch",
            model=resumed_model,
            optimizer=resumed_optimizer,
            policy=policy,
        )
        snapshot = read_experiment_log_v1(resumed.path)

    assert interrupted.status == "step_limit_reached"
    assert interrupted.event_count == 5
    assert interrupted.completed_epochs == 1
    assert interrupted.next_epoch_index == 2
    assert interrupted.next_training_ordinal == 2
    assert resumed.status == "complete"
    assert resumed.replayed_training_steps == 1
    assert resumed.new_training_steps == 1
    assert resumed.completed_epochs == 2
    assert resumed.optimizer_completed_steps == 4
    assert resumed.event_count == 9
    assert [event.payload.record.event_type for event in snapshot.events] == [
        "training_step",
        "training_step",
        "checkpoint",
        "evaluation",
        "training_step",
        "training_step",
        "checkpoint",
        "evaluation",
        "evaluation",
    ]


def test_existing_run_rejects_reusing_a_stepped_optimizer_without_appending(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-stepped-owner",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )

        with pytest.raises(IrisExperimentRunnerError, match="pristine resume preflight"):
            _execute(
                root=tmp_path,
                run_id="managed-stepped-owner",
                model=model,
                optimizer=optimizer,
            )

    assert read_experiment_log_v1(partial.path).event_count == 1


def test_existing_run_rejects_a_different_epoch_policy_without_appending(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-policy-conflict",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        fresh_model, fresh_optimizer = _fresh_owner()

        with pytest.raises(IrisExperimentRunnerError, match="run creation or loading"):
            _execute(
                root=tmp_path,
                run_id="managed-policy-conflict",
                model=fresh_model,
                optimizer=fresh_optimizer,
                policy=_policy(epoch_count=2),
            )

    assert read_experiment_log_v1(partial.path).event_count == 1


def test_managed_training_append_exact_retry_reuses_installed_head(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        created = create_experiment_log_v1(
            run_id="managed-exact-retry",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=_policy(),
            model=model,
            optimizer=optimizer,
            root=tmp_path,
        )
        predecessor = read_experiment_log_v1(created.path).chain_head_sha256
        prepared = next(
            iterate_experiment_split_v1(
                manifest=_manifest(),
                split="train",
                backbone=model.backbone,
                terminal_token_id=15,
            )
        )
        output = optimizer.step(prepared.teacher_input)
        first = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
            expected_predecessor_sha256=predecessor,
        )
        repeated = record_experiment_training_step_v1(
            path=created.path,
            manifest=_manifest(),
            prepared=prepared,
            output=output,
            model=model,
            optimizer=optimizer,
            expected_predecessor_sha256=predecessor,
        )

    assert not first.reused_existing
    assert repeated.reused_existing
    assert repeated.sequence == first.sequence == 1
    assert repeated.payload_sha256 == first.payload_sha256
    assert read_experiment_log_v1(created.path).event_count == 1


def test_managed_low_level_training_rejects_warn_only_runtime_without_append(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        created = create_experiment_log_v1(
            run_id="managed-warn-only-append",
            manifest=_manifest(),
            terminal_token_id=15,
            initialization_seed=1907,
            deterministic=True,
            execution_policy=_policy(),
            model=model,
            optimizer=optimizer,
            root=tmp_path,
        )
        prepared = next(
            iterate_experiment_split_v1(
                manifest=_manifest(),
                split="train",
                backbone=model.backbone,
                terminal_token_id=15,
            )
        )
        output = optimizer.step(prepared.teacher_input)
        torch.use_deterministic_algorithms(True, warn_only=True)

        with pytest.raises(IrisExperimentLogError, match="recorded"):
            record_experiment_training_step_v1(
                path=created.path,
                manifest=_manifest(),
                prepared=prepared,
                output=output,
                model=model,
                optimizer=optimizer,
            )

    assert read_experiment_log_v1(created.path).event_count == 0


def test_runner_rejects_interposed_chain_advance_before_using_stale_live_owner(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        other_model, other_optimizer = _fresh_owner()
        owned_manifest = _manifest()
        call_count = 0

        def interpose(
            *,
            progress: IrisExperimentEpochProgressV1,
            run: IrisExperimentRunV1,
            events: tuple[IrisExperimentEventV1, ...],
        ) -> IrisExperimentEpochProgressV1:
            nonlocal call_count
            current = extend_experiment_epoch_progress_v1(
                progress=progress,
                run=run,
                events=events,
            )
            call_count += 1
            if call_count == 1:
                prepared = next(
                    iterate_experiment_split_v1(
                        manifest=owned_manifest,
                        split="train",
                        backbone=other_model.backbone,
                        terminal_token_id=15,
                    )
                )
                output = other_optimizer.step(prepared.teacher_input)
                record_experiment_training_step_v1(
                    path=tmp_path / "experiments" / "v1" / "managed-interposed",
                    manifest=owned_manifest,
                    prepared=prepared,
                    output=output,
                    model=other_model,
                    optimizer=other_optimizer,
                    expected_predecessor_sha256=run.payload_sha256,
                )
            return current

        with (
            patch.object(
                runner_module,
                "extend_experiment_epoch_progress_v1",
                side_effect=interpose,
            ),
            pytest.raises(IrisExperimentRunnerError, match="managed journal validation"),
        ):
            _execute(
                root=tmp_path,
                run_id="managed-interposed",
                model=model,
                optimizer=optimizer,
            )

    snapshot = read_experiment_log_v1(tmp_path / "experiments" / "v1" / "managed-interposed")
    assert snapshot.event_count == 1
    assert optimizer.completed_steps == 0
    assert other_optimizer.completed_steps == 1


def test_rehashed_training_metric_tamper_fails_replay_before_any_append(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-replay-tamper",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        event_path = partial.path / "events" / "00000001.json"
        raw = json.loads(event_path.read_text(encoding="utf-8"))
        raw["payload"]["record"]["loss"] += 0.25
        raw["payload_sha256"] = domain_sha256(
            _EVENT_PAYLOAD_DOMAIN,
            canonical_json_bytes(raw["payload"]),
        )
        event_path.write_bytes(canonical_json_file_bytes(raw))

        resumed_model, resumed_optimizer = _fresh_owner()
        with pytest.raises(
            IrisExperimentRunnerError,
            match="checkpoint restore and training-tail replay",
        ):
            _execute(
                root=tmp_path,
                run_id="managed-replay-tamper",
                model=resumed_model,
                optimizer=resumed_optimizer,
            )

    assert read_experiment_log_v1(partial.path).event_count == 1


def test_rehashed_manifest_order_tamper_fails_state_machine_before_restore(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-order-tamper",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        event_path = partial.path / "events" / "00000001.json"
        raw = json.loads(event_path.read_text(encoding="utf-8"))
        raw["payload"]["record"]["example_id"] = "train-short"
        raw["payload"]["record"]["split_ordinal"] = 2
        raw["payload_sha256"] = domain_sha256(
            _EVENT_PAYLOAD_DOMAIN,
            canonical_json_bytes(raw["payload"]),
        )
        event_path.write_bytes(canonical_json_file_bytes(raw))
        assert read_experiment_log_v1(partial.path).event_count == 1

        resumed_model, resumed_optimizer = _fresh_owner()
        with pytest.raises(IrisExperimentRunnerError, match="run creation or loading"):
            _execute(
                root=tmp_path,
                run_id="managed-order-tamper",
                model=resumed_model,
                optimizer=resumed_optimizer,
            )

    assert read_experiment_log_v1(partial.path).event_count == 1


def test_managed_low_level_checkpoint_append_rejects_wrong_phase(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-wrong-phase",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        checkpoint = build_teacher_checkpoint_v1(model=model, optimizer=optimizer)

        with pytest.raises(IrisExperimentLogError, match="recorded"):
            record_experiment_checkpoint_v1(
                path=partial.path,
                checkpoint=checkpoint,
                expected_predecessor_sha256=(
                    read_experiment_log_v1(partial.path).chain_head_sha256
                ),
            )
        with pytest.raises(IrisExperimentLogError, match="recorded"):
            record_experiment_checkpoint_v1(
                path=partial.path,
                checkpoint=checkpoint,
                manifest=_manifest(),
                expected_predecessor_sha256=(
                    read_experiment_log_v1(partial.path).chain_head_sha256
                ),
            )

    assert read_experiment_log_v1(partial.path).event_count == 1


def test_checkpoint_capture_rng_consumption_fails_before_checkpoint_event(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        original_capture = build_teacher_checkpoint_v1

        def consume_rng_after_capture(
            *,
            model: IrisTeacherForcingModel,
            optimizer: IrisTeacherOptimizer,
        ) -> IrisTeacherCheckpointBundleV1:
            checkpoint = original_capture(model=model, optimizer=optimizer)
            _ = torch.rand(1)
            return checkpoint

        with (
            patch.object(
                runner_module,
                "build_teacher_checkpoint_v1",
                side_effect=consume_rng_after_capture,
            ),
            pytest.raises(IrisExperimentRunnerError, match="epoch checkpoint"),
        ):
            _execute(
                root=tmp_path,
                run_id="managed-checkpoint-rng",
                model=model,
                optimizer=optimizer,
            )

    snapshot = read_experiment_log_v1(tmp_path / "experiments" / "v1" / "managed-checkpoint-rng")
    assert snapshot.event_count == 2
    assert all(
        not isinstance(event.payload.record, ExperimentCheckpointEventV1)
        for event in snapshot.events
    )


def test_evaluation_rng_consumption_fails_before_evaluation_event(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        original_evaluation = model.evaluation_forward

        def consume_rng_after_evaluation(
            example: IrisTeacherForcingInput,
        ) -> IrisTeacherForcedOutput:
            output = original_evaluation(example)
            _ = torch.rand(1)
            return output

        with (
            patch.object(
                IrisTeacherForcingModel,
                "evaluation_forward",
                side_effect=consume_rng_after_evaluation,
            ),
            pytest.raises(IrisExperimentRunnerError, match="validation evaluation"),
        ):
            _execute(
                root=tmp_path,
                run_id="managed-evaluation-rng",
                model=model,
                optimizer=optimizer,
            )

    snapshot = read_experiment_log_v1(tmp_path / "experiments" / "v1" / "managed-evaluation-rng")
    assert snapshot.event_count == 3
    assert isinstance(snapshot.events[-1].payload.record, ExperimentCheckpointEventV1)


def test_runner_rejects_eval_mode_without_mutating_or_creating_files(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        model.eval()

        with pytest.raises(IrisExperimentRunnerError, match="input validation"):
            _execute(
                root=tmp_path,
                run_id="managed-eval-owner",
                model=model,
                optimizer=optimizer,
            )

    assert not model.training
    assert not (tmp_path / "experiments").exists()


def test_rng_consuming_training_fails_closed_without_recording_the_step(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
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
                run_id="managed-rng-change",
                model=model,
                optimizer=optimizer,
            )

    path = tmp_path / "experiments" / "v1" / "managed-rng-change"
    assert read_experiment_log_v1(path).event_count == 0


def test_runner_rejects_nondeterminism_and_invalid_step_limit_before_filesystem_change(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        model, optimizer = _fresh_owner()
        with pytest.raises(IrisExperimentRunnerError, match="input validation"):
            execute_experiment_v1(
                run_id="managed-nondeterministic",
                manifest=_manifest(),
                terminal_token_id=15,
                initialization_seed=1907,
                deterministic=False,
                execution_policy=_policy(),
                model=model,
                optimizer=optimizer,
                root=tmp_path,
            )
        with pytest.raises(IrisExperimentRunnerError, match="input validation"):
            execute_experiment_v1(
                run_id="managed-zero-limit",
                manifest=_manifest(),
                terminal_token_id=15,
                initialization_seed=1907,
                deterministic=True,
                execution_policy=_policy(),
                model=model,
                optimizer=optimizer,
                root=tmp_path,
                maximum_new_training_steps=0,
            )

    assert not (tmp_path / "experiments").exists()


def test_epoch_progress_matches_full_analysis_while_advancing_one_event_at_a_time(
    tmp_path: Path,
) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        result = _execute(
            root=tmp_path,
            run_id="managed-incremental-progress",
            model=model,
            optimizer=optimizer,
            policy=_policy(epoch_count=2),
        )
        snapshot = read_experiment_log_v1(result.path)
        manifest = _manifest()
        progress = analyze_experiment_epoch_progress_v1(
            run=snapshot.run,
            events=(),
            manifest=manifest,
        )

        for event_count in range(1, snapshot.event_count + 1):
            with patch.object(
                execution_module,
                "_advance_epoch_cursor",
                wraps=execution_module._advance_epoch_cursor,
            ) as advance:
                progress = extend_experiment_epoch_progress_v1(
                    progress=progress,
                    run=snapshot.run,
                    events=snapshot.events[:event_count],
                )
            expected = analyze_experiment_epoch_prefix_v1(
                run=snapshot.run,
                events=snapshot.events[:event_count],
                manifest=manifest,
            )
            assert advance.call_count == 1
            assert progress.verified_event_count == event_count
            assert progress.cursor == expected

        unchanged = extend_experiment_epoch_progress_v1(
            progress=progress,
            run=snapshot.run,
            events=snapshot.events,
        )

    assert snapshot.event_count == 9
    assert unchanged is progress
    assert progress.cursor.phase == "complete"


def test_epoch_progress_rejects_shrink_changed_head_and_run_identity(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        partial = _execute(
            root=tmp_path,
            run_id="managed-progress-binding",
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=1,
        )
        snapshot = read_experiment_log_v1(partial.path)
        progress = analyze_experiment_epoch_progress_v1(
            run=snapshot.run,
            events=snapshot.events,
            manifest=_manifest(),
        )
        changed_head = snapshot.events[0].model_copy(
            update={"payload_sha256": "f" * 64},
        )

        with pytest.raises(ValueError, match="shrank"):
            extend_experiment_epoch_progress_v1(
                progress=progress,
                run=snapshot.run,
                events=(),
            )
        with pytest.raises(ValueError, match="changed"):
            extend_experiment_epoch_progress_v1(
                progress=progress,
                run=snapshot.run,
                events=(changed_head,),
            )
        with pytest.raises(ValueError, match="identity"):
            extend_experiment_epoch_progress_v1(
                progress=replace(progress, run_payload_sha256="f" * 64),
                run=snapshot.run,
                events=snapshot.events,
            )


def test_two_epoch_runner_uses_constant_full_journal_reads(tmp_path: Path) -> None:
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        real_reader = read_experiment_log_v1
        with (
            patch.object(
                experiment_log_module,
                "read_experiment_log_v1",
                wraps=real_reader,
            ) as log_reader,
            patch.object(
                runner_module,
                "read_experiment_log_v1",
                wraps=real_reader,
            ) as runner_reader,
        ):
            result = _execute(
                root=tmp_path,
                run_id="managed-streaming-read-bound",
                model=model,
                optimizer=optimizer,
                policy=_policy(epoch_count=2),
            )

    assert result.status == "complete"
    assert result.event_count == 9
    assert log_reader.call_count == 1
    assert runner_reader.call_count == 1
