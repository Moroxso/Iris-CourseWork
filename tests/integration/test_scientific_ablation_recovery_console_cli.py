"""Actual installed console recovery/inspection, not CliRunner or exception death.

Requires the explicit owned ConsoleClient from the native integration harness.
Fresh original random fixture and actual CLI training/selected export ONCE;
two disjoint original workflow roots, ALL4 kernels, original stop3/4 and16MiB.
Four non-executable rows; never execute generated Python on the Windows host.
Diagnostics/publish/defaults/scientific acceptance remain separate obligations.
"""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from inspect import unwrap
from pathlib import Path
from typing import Any, Literal

import pytest

from iris.application import scientific_ablation_operator_export as export
from iris.application import scientific_ablation_operator_workflow as binding
from iris.artifacts.scientific_ablation_command_workflow_v1 import COMMAND_WORKFLOW_NAMESPACE
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPhaseDataV1,
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._console_json_wire import windows_json_object
from tests.integration._full_console_context import client, fixture_scope
from tests.integration.test_scientific_ablation_execution import _config, _tree
from tests.integration.test_scientific_ablation_execution import (
    cache_ready as original_cache_fixture,
)
from tests.integration.test_scientific_ablation_full_console_cli import _append, _save
from tests.integration.test_scientific_ablation_workflow_planning_cli import _common
from tests.integration.test_scientific_generation import _test_spec
from tests.unit._workflow_data_setup import intent as data_intent

pytestmark = pytest.mark.integration
ACTUAL_CLI_OBSERVATIONS: list[dict[str, Any]] = []
TRAINING_OBSERVATIONS: list[dict[str, Any]] = []


def _inspection_anchors(
    before: tuple[str, ...], completed: IrisScientificAblationWorkflowPhaseDataV1 | None
) -> tuple[str, ...]:
    """A completed phase is inspected only with its explicitly read DATA anchor."""
    if completed is None:
        return before
    assert completed.payload.phase.index == len(before)
    assert completed.payload.predecessor_sha256 == (before[-1] if before else None)
    assert completed.payload.origin == "kernel_data"
    return (*before, completed.sha256)


class ConsoleProgram:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.transcript = root / "actual-cli-transcript.jsonl"
        with self.transcript.open("xb"):
            pass

    def call(
        self,
        argv: list[str],
        *,
        effect: Literal["data", "loads", "forward"] = "data",
        expected: int = 0,
    ) -> dict[str, Any]:
        real = client()
        full_argv = ["scientific-ablation", *argv]
        _append(
            self.transcript,
            dict(
                event="started",
                index=len(real.jobs),
                argv=full_argv,
                utc=datetime.now(UTC).isoformat(),
            ),
        )
        fixture_sha = sha256(
            (real.base / "tiny-hf-cache-fixture-manifest.json").read_bytes()
        ).hexdigest()
        result = real.run(
            full_argv, effect=effect, expected=expected, fixture_manifest_sha=fixture_sha
        )
        assert result["outer"]["actual_exit_code"] == expected
        assert not result["child_exit"]["child_operator_method_errors"]
        response = windows_json_object(result["stdout"].encode()) if expected == 0 else None
        self.calls.append(dict(result=result, response=response))
        _append(self.transcript, dict(event="returned", result=result, response=response))
        if expected != 0:
            if expected == 2:
                assert result["stderr"]
            else:
                assert expected == 83
                assert result["child_exit"]["termination"] == "post-ACK-hard-exit-requested"
                assert result["child_exit"]["normal_atexit_finish_entered"] is False
            return {}
        assert type(response) is dict
        assert result["stdout"].encode() == canonical_json_file_bytes(response)[:-1] + b"\r\n"
        return response

    def last_trace(self) -> dict[str, Any]:
        value: dict[str, Any] = self.calls[-1]["result"]["child_exit"]["kernel_observation"]
        return value

    def no_kernel_work(self) -> None:
        trace = self.last_trace()
        assert not any(trace["calls"].values())
        assert trace["actual_row_calls"] == trace["actual_row_returns"] == []
        assert trace["actual_published_rows"] == trace["actual_kernel_aggregate_publications"] == []
        assert trace["row_pins_before"] == trace["row_pins_after"]


@pytest.fixture(scope="module")
def console_export(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    real = client()
    assert not real.jobs
    build: Callable[[], Path] = unwrap(original_cache_fixture)
    with fixture_scope():
        cache = build()
    assert cache == real.base / "tiny-hf-cache"
    base = tmp_path_factory.mktemp("recovery-console-trained-export")
    program = ConsoleProgram(base)
    config = _config(base)
    recipe = base / "recipe.json"
    _save(recipe, config.model_dump(mode="json"))
    new_args = [
        "new",
        "--recipe",
        str(recipe),
        "--base",
        str(base),
        "--root",
        str(base / "operators"),
    ]
    state = program.call(new_args, effect="loads")
    path = Path(state["operator_path"])
    assert state["step"] == state["acknowledged_phases"] == 0
    training_requests: list[dict[str, Any]] = []
    training_results: list[dict[str, Any]] = []
    for index, phase in enumerate(
        (
            "train",
            "validation",
            "train",
            "validation",
            "checkpoint",
            "train",
            "validation",
            "checkpoint",
        )
    ):
        assert state["phase"] == phase and state["acknowledged_phases"] == index
        request_path = base / f"training-{index:02d}.json"
        request = program.call(
            [
                "phase-request",
                "--operator",
                str(path),
                "--expected-previous-sha256",
                state["head_sha256"],
                "--phase",
                phase,
                "--out",
                str(request_path),
            ]
        )
        verb = "resume" if index == 2 else ("validate" if phase == "validation" else phase)
        args = [
            verb,
            "--operator",
            str(path),
            "--request",
            str(request_path),
            "--base",
            str(base),
            "--maximum-replay-phases",
            "8",
        ]
        result = program.call(args, effect="forward")
        assert result["payload"]["request"] == request
        event = result["payload"]["event"]
        assert event["payload"]["sequence"] == index + 1
        assert event["payload"]["previous_sha256"] == state["head_sha256"]
        assert event["payload"]["record"]["kind"] == phase
        if phase == "train":
            metric = event["payload"]["record"]["metric"]
            assert metric["supervised_token_count"] > 0 and metric["gradient_l2_norm"] > 0
            assert metric["nonzero_gradient_parameter_count_before_clip"] > 0
            assert (
                metric["scientific_state_before_sha256"] != metric["scientific_state_after_sha256"]
            )
        before = _tree(path)
        assert program.call(["resume", *args[1:]]) == result
        program.no_kernel_work()
        assert _tree(path) == before
        training_requests.append(request)
        training_results.append(result)
        state = program.call(["inspect", "--operator", str(path)])
        assert state["head_sha256"] == event["sha256"]
    best = program.call(["best", "--operator", str(path)])
    assert state["phase"] == "stopped" and state["step"] == 3
    assert state["acknowledged_phases"] == 8 and state["best_step"] == 1
    assert best["payload"]["validated_best_is_persisted"] is True
    assert best["payload"]["persisted_best_step"] == 1
    request_path = base / "export-request.json"
    request = program.call(
        [
            "export-request",
            "--operator",
            str(path),
            "--checkpoint-event-sha256",
            best["payload"]["persisted_checkpoint_event_sha256"],
            "--lifecycle-sha256",
            best["payload"]["persisted_lifecycle_sha256"],
            "--selection",
            "selected_best",
            "--maximum-storage-bytes",
            str(4 * 1024**2),
            "--minimum-free-bytes",
            "0",
            "--out",
            str(request_path),
        ]
    )
    exports = base / "exports"
    published = program.call(
        [
            "export",
            "--operator",
            str(path),
            "--request",
            str(request_path),
            "--base",
            str(base),
            "--root",
            str(exports),
        ],
        effect="loads",
    )
    assert published["result"]["payload"]["request"] == request
    assert published["result"]["payload"]["selected_completed_steps"] == 1
    source = binding.read_scientific_ablation_operator_generation_source_v1(
        path,
        request=export.read_scientific_ablation_export_request_v1(request_path),
        export_root=exports,
    )
    rows = tuple(
        row
        for row in generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
        if row.task_type == "next_tool_prediction" and row.difficulty == 1
    )[:4]
    assert len(rows) == 4 and not any(row.executable for row in rows)
    test = _test_spec(base / "test.parquet", rows)
    control = WorkspaceInterventionPlanV1(
        interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
    )
    payload = ScientificAblationWorkflowIntentV1.model_validate(
        data_intent(base).model_dump(mode="python")
        | dict(
            source_sha256=source.bundle.manifest.sha256,
            lifecycle_sha256=request["payload"]["lifecycle_sha256"],
            inference_sha256=published["result"]["payload"]["inference_sha256"],
            selected_checkpoint_sha256=request["payload"]["source_checkpoint_sha256"],
            lifecycle_path=str(source.lifecycle_path),
            inference_path=str(source.inference_path),
            inputs_root=str(base),
            training=config.train.model_dump(mode="python"),
            validation=config.validation.model_dump(mode="python"),
            test=test.model_dump(mode="python"),
            controls=(control,),
        )
    )
    assert payload.resources.maximum_storage_bytes == 16 * 1024**2 and payload.python_policy is None
    assert config.backbone.limits.max_new_tokens == 256
    payload_path = base / "workflow-payload.json"
    _save(payload_path, payload.model_dump(mode="json"))
    result = dict(
        base=base,
        operator=path,
        request=request_path,
        exports=exports,
        payload=payload_path,
        trained_tree=_tree(path),
        export_tree=_tree(exports),
        calls=program.calls,
    )
    TRAINING_OBSERVATIONS.append(dict(base=str(base), actual_calls=program.calls, completed=True))
    _save(
        base / "actual-training-export-witness.json",
        dict(
            kind="iris-recovery-console-actual-training-export-witness-v1",
            base=str(base),
            operator_path=str(path),
            export_root=str(exports),
            export_request=request,
            published_export=published,
            terminal_operator=state,
            best=best,
            actual_training_requests=training_requests,
            actual_training_results=training_results,
            actual_calls=program.calls,
            cache_fixture=str(cache),
            scientific_acceptance=False,
        ),
    )
    return result


@pytest.mark.parametrize("scenario", ("partial", "aggregate"))
def test_actual_installed_console_recover_and_inspect_all_four_original_kernels(
    tmp_path: Path, console_export: dict[str, Any], scenario: str
) -> None:
    shared = console_export
    base, path, request, exports = (shared[k] for k in ("base", "operator", "request", "exports"))
    program = ConsoleProgram(tmp_path)
    plan_file, root = tmp_path / "workflow-plan.json", tmp_path / "commands"
    anchors: tuple[str, ...] = ()

    def common(visible_anchors: tuple[str, ...] | None = None) -> list[str]:
        prefix = anchors if visible_anchors is None else visible_anchors
        return ["workflow", *_common(path, request, exports, plan_file, base, root, prefix)]

    program.call([*common(), "plan-request", "--payload", str(shared["payload"])])
    plan = IrisScientificAblationWorkflowPlanV1.model_validate_json(plan_file.read_bytes())
    status = program.call([*common(), "status"])
    store = root / COMMAND_WORKFLOW_NAMESPACE / status["definition"]["sha256"]
    observed: dict[str, Any] = dict(
        scenario=scenario, workflow_path=str(store), actual_calls=program.calls, completed=False
    )
    ACTUAL_CLI_OBSERVATIONS.append(observed)
    cases: list[dict[str, Any]] = []
    stop, crash_count = (3, 1) if scenario == "partial" else (4, 4)
    phases = scientific_ablation_workflow_phases_v1(plan)
    assert len(phases) == 8 and len(plan.payload.controls) == 1
    for phase in phases[:4]:
        assert phase.index == len(anchors) and phase.kind in ("capture", "score")
        effect: Literal["data", "loads", "forward"] = (
            "forward" if phase.kind == "capture" else "data"
        )
        operation = "generate" if phase.kind == "capture" else "score"
        initial, chunk_file, command_file = (
            tmp_path / (phase.name + suffix)
            for suffix in (".initial.json", ".chunk.json", ".command.json")
        )
        program.call(
            [
                *common(),
                "kernel-request",
                "--phase",
                phase.name,
                "--initial",
                "--rows-per-call",
                str(stop),
                "--out",
                str(initial),
            ]
        )
        chunk = program.call(
            [*common(), "kernel-plan", "--request", str(initial), "--out", str(chunk_file)],
            effect="loads" if phase.kind == "capture" else "data",
        )
        assert (
            chunk["acknowledged_rows"] == 0
            and chunk["total_rows"] == 4
            and chunk["stop_row_index"] == stop
        )
        command = program.call(
            [
                *common(),
                "command-request",
                "--phase",
                phase.name,
                "--expected-chunk",
                str(chunk_file),
                "--rows-per-call",
                str(stop),
                "--out",
                str(command_file),
            ]
        )
        dispatch = [operation, "--command", str(command_file)]
        program.call([*common(), *dispatch], effect=effect, expected=83)
        crash = program.calls[-1]["result"]
        trace = program.last_trace()
        assert (
            trace["actual_row_calls"] == trace["actual_published_rows"] == list(range(crash_count))
        )
        kernel = Path(trace["kernel_path"])
        original_pins = trace["row_pins_after"]
        row_prefix = "row" if phase.kind == "capture" else "score"
        assert not (kernel / "complete").exists()
        before = _tree(store)
        inspection = [
            "inspect-kernel",
            "--phase",
            phase.name,
            "--kernel-plan-sha256",
            chunk["kernel_plan_sha256"],
        ]
        inspected = program.call([*common(), *inspection])
        program.no_kernel_work()
        assert (
            inspected["read_only"] is True
            and inspected["kernel_progress"]["acknowledged_rows"] == crash_count
        )
        assert _tree(store) == before
        program.call([*common(), *dispatch], expected=2)
        program.no_kernel_work()
        assert _tree(store) == before
        recover_args = ["recover", "--operation", operation, "--command", str(command_file)]
        recovered = program.call(
            [*common(), *recover_args], effect=effect if scenario == "partial" else "data"
        )
        recovery_trace = program.last_trace()
        snapshot = recovered["kernel_progress"]
        assert recovered["original_return"] == "unavailable_not_reconstructed"
        assert snapshot["acknowledged_rows"] == stop and snapshot["total_rows"] == 4
        assert recovery_trace["row_pins_before"] == original_pins
        for index, pin in original_pins.items():
            assert recovery_trace["row_pins_after"][index] == pin
        handoff = store / "handoffs" / command["sha256"][:2] / command["sha256"]
        assert {p.name for p in handoff.iterdir()} == {"intent.json"}
        completed_phase: IrisScientificAblationWorkflowPhaseDataV1 | None = None
        if scenario == "partial":
            assert (
                recovery_trace["actual_row_calls"]
                == recovery_trace["actual_published_rows"]
                == [1, 2]
            )
            assert recovery_trace["actual_kernel_aggregate_publications"] == []
            assert not (kernel / f"{row_prefix}-00003").exists()
        else:
            assert (
                recovery_trace["actual_row_calls"] == recovery_trace["actual_published_rows"] == []
            )
            assert recovery_trace["calls"] == dict(
                factory=0, producer=0, finalizer=1, finalizer_returned=1
            )
            assert recovery_trace["actual_kernel_aggregate_publications"] == [
                str(kernel / "complete")
            ]
            completed_phase = IrisScientificAblationWorkflowPhaseDataV1.model_validate_json(
                (store / f"phase-{phase.index:03d}" / "complete.json").read_bytes()
            )
            before = _tree(store)
            # The old BEFORE prefix cannot silently adopt the now-ready phase.
            program.call([*common(), *inspection], expected=2)
            program.no_kernel_work()
            assert _tree(store) == before
        inspection_anchors = _inspection_anchors(anchors, completed_phase)
        before = _tree(store)
        inspected = program.call(
            [
                *common(inspection_anchors),
                *inspection,
                "--expected-acknowledgement-head-sha256",
                snapshot["acknowledgement_head_sha256"],
            ]
        )
        program.no_kernel_work()
        assert inspected["kernel_progress"]["acknowledged_rows"] == stop
        program.call(
            [
                *common(inspection_anchors),
                *inspection,
                "--expected-acknowledgement-head-sha256",
                "f" * 64,
            ],
            expected=2,
        )
        program.no_kernel_work()
        assert _tree(store) == before
        program.call(
            [*common(), *recover_args, "--expected-operator-ack-sha256", "f" * 64], expected=2
        )
        program.no_kernel_work()
        assert _tree(store) == before
        saved_replay = [
            *recover_args,
            "--expected-operator-ack-sha256",
            recovered["operator_ack"]["sha256"],
        ]
        assert program.call([*common(), *saved_replay]) == recovered
        program.no_kernel_work()
        assert _tree(store) == before
        if scenario == "partial":
            retained, final_chunk, final_command = (
                tmp_path / (phase.name + suffix)
                for suffix in (".retained.json", ".final-chunk.json", ".final-command.json")
            )
            retained_args = [
                "kernel-request",
                "--phase",
                phase.name,
                "--rows-per-call",
                "1",
                "--expected-kernel-plan-sha256",
                chunk["kernel_plan_sha256"],
                "--expected-acknowledgement-head-sha256",
                snapshot["acknowledgement_head_sha256"],
                "--expected-acknowledged-rows",
                "3",
            ]
            program.call([*common(), *retained_args, "--out", str(retained)])
            final_interval = program.call(
                [*common(), "kernel-plan", "--request", str(retained), "--out", str(final_chunk)]
            )
            program.no_kernel_work()
            assert (
                final_interval["acknowledged_rows"] == 3 and final_interval["stop_row_index"] == 4
            )
            assert _tree(store) == before
            wrong = retained_args.copy()
            wrong[wrong.index("--expected-acknowledgement-head-sha256") + 1] = "f" * 64
            wrong_file, missing = (
                tmp_path / (phase.name + ".wrong.json"),
                tmp_path / (phase.name + ".must-not-exist.json"),
            )
            program.call([*common(), *wrong, "--out", str(wrong_file)])
            program.call(
                [*common(), "kernel-plan", "--request", str(wrong_file), "--out", str(missing)],
                expected=2,
            )
            program.no_kernel_work()
            assert not missing.exists() and _tree(store) == before
            final_intent = program.call(
                [
                    *common(),
                    "command-request",
                    "--phase",
                    phase.name,
                    "--expected-chunk",
                    str(final_chunk),
                    "--rows-per-call",
                    "1",
                    "--out",
                    str(final_command),
                ]
            )
            assert final_intent["sha256"] != command["sha256"]
            finished = program.call(
                [*common(), operation, "--command", str(final_command)], effect=effect
            )
            final_trace = program.last_trace()
            assert finished["kernel_progress"]["acknowledged_rows"] == 4
            assert final_trace["actual_row_calls"] == final_trace["actual_published_rows"] == [3]
            assert final_trace["actual_kernel_aggregate_publications"] == [str(kernel / "complete")]
            for index, pin in original_pins.items():
                assert final_trace["row_pins_after"][index] == pin
        record = IrisScientificAblationWorkflowPhaseDataV1.model_validate_json(
            (store / f"phase-{phase.index:03d}" / "complete.json").read_bytes()
        )
        assert record.payload.phase == phase and record.payload.origin == "kernel_data"
        anchors = (*anchors, record.sha256)
        before = _tree(store)
        replay_after_head4 = program.call([*common(), *saved_replay])
        program.no_kernel_work()
        assert replay_after_head4 == recovered and _tree(store) == before
        inspected = program.call([*common(), *inspection])
        program.no_kernel_work()
        assert inspected["kernel_progress"]["acknowledged_rows"] == 4
        assert inspected["kernel_progress"]["kernel_complete_sha256"] is not None
        status = program.call([*common(), "status"])
        assert tuple(status["progress"]["completed_phase_sha256"]) == anchors
        cases.append(
            dict(
                phase=phase.model_dump(mode="json"),
                original_command=command,
                actual_crash=crash,
                recovered=recovered,
                recovery_observation=recovery_trace,
                inspection_completed_phase_sha256=inspection_anchors,
                final_record=record.model_dump(mode="json"),
                saved_replay_after_head4=replay_after_head4,
            )
        )
    assert len(cases) == len(anchors) == 4
    assert (
        status["progress"]["complete"] is False
        and status["progress"]["next_phase"] == "diagnostics"
    )
    assert _tree(path) == shared["trained_tree"] and _tree(exports) == shared["export_tree"]
    observed.update(completed=True, phase_anchors=anchors, cases=cases)
    _save(
        tmp_path / "actual-recovery-console-witness.json",
        dict(
            kind="iris-recovery-console-actual-witness-v2",
            scenario=scenario,
            operator_path=str(path),
            base=str(base),
            export_root=str(exports),
            workflow_path=str(store),
            plan=plan.model_dump(mode="json"),
            phase_anchors=anchors,
            cases=cases,
            actual_CLI_calls=program.calls,
            actual_console_caller_pid=os.getpid(),
            full_workflow_or_scientific_acceptance=False,
        ),
    )
