"""Actual generated iris.exe training + ALL8 workflow phases, honest CHILD evidence.

Same complete original full-source CLI program/phase/control/writer assertions.
NEW fixture only; no parent native forwards or trained fixture substitution.
This is tiny H16/M4/D8 engineering, NOT default/scientific/isolated-positive proof.
"""

import json
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
    ScientificAblationWorkflowResourcesV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
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
from tests.integration.test_scientific_ablation_workflow_planning_cli import _common
from tests.integration.test_scientific_generation import _test_spec
from tests.unit._workflow_data_setup import intent as data_intent
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = pytest.mark.integration
ACTUAL_CLI_OBSERVATIONS: list[dict[str, Any]] = []
CHILD_METHOD_ERRORS: list[dict[str, Any]] = []


@pytest.fixture(scope="module")
def cache_ready() -> Path:
    # functools.update_wrapper on the pinned pytest9.1 fixture preserves the
    # ORIGINAL function; only its NEW random-cache construction is permitted.
    build: Callable[[], Path] = unwrap(original_cache_fixture)
    with fixture_scope():
        return build()


def _save(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_file_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, value: Any) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_json_file_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _invoke(
    argv: list[str],
    logs: list[dict[str, Any]],
    counts: dict[str, int],
    transcript: Path,
    *,
    effect: Literal["data", "loads", "forward"] = "data",
    expected: int = 0,
) -> dict[str, Any]:
    real = client()
    assert counts is real.child_totals
    index = len(logs)
    _append(
        transcript,
        dict(
            event="started",
            index=index,
            utc=datetime.now(UTC).isoformat(),
            argv=["scientific-ablation", *argv],
            counter_scope="actual-child-only",
        ),
    )
    fixture_sha = sha256(
        (real.base / "tiny-hf-cache-fixture-manifest.json").read_bytes()
    ).hexdigest()
    result = real.run(
        ["scientific-ablation", *argv],
        effect=effect,
        expected=expected,
        fixture_manifest_sha=fixture_sha,
    )
    actual_code = result["outer"]["actual_exit_code"]
    response = windows_json_object(result["stdout"].encode("utf-8")) if actual_code == 0 else None
    CHILD_METHOD_ERRORS.extend(result["child_exit"]["child_operator_method_errors"])
    record: dict[str, Any] = dict(
        event="returned",
        index=index,
        utc=datetime.now(UTC).isoformat(),
        argv=["scientific-ablation", *argv],
        exit_code=actual_code,
        stdout=result["stdout"],
        stderr=result["stderr"],
        response=response,
        child_counts=result["child_counts"],
        counter_scope="actual-child-only",
        job_directory=result["job_directory"],
        job_sha256=result["job_sha256"],
        actual_child_pid=result["startup"]["pid"],
        actual_launch_pid=result["outer"]["launch_pid"],
        child_exit=result["child_exit"],
        actual_outer=result["outer"],
    )
    logs.append(record)
    _append(transcript, record)
    assert actual_code == expected, record
    delta = record["child_counts"]
    if effect == "data":
        assert not any(delta.values()), record
    else:
        assert all(delta[key] > 0 for key in ("graph", "model", "tokenizer")), record
        if effect == "forward":
            assert delta["qwen"] > 0, record
    if expected != 0:
        assert response is None and result["stderr"], record
        return {}
    assert type(response) is dict
    assert result["stdout"].encode() == canonical_json_file_bytes(response)[:-1] + b"\r\n", record
    return response


def test_actual_full_console_cli_training_selected_export_all4_diagnostics_and_both_writers(
    tmp_path: Path, cache_ready: Path
) -> None:
    assert cache_ready == client().base / "tiny-hf-cache"
    config = _config(tmp_path)
    recipe_file = tmp_path / "recipe.json"
    _save(recipe_file, config.model_dump(mode="json"))
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row
        for task in TASKS
        for row in tuple(r for r in pool if r.task_type == task and r.difficulty == 1)[:2]
    )
    assert len(rows) == 8 and {row.task_type for row in rows} == set(TASKS)
    assert sum(row.executable for row in rows) == 2
    test = _test_spec(tmp_path / "test.parquet", rows)
    logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    global CHILD_METHOD_ERRORS
    CHILD_METHOD_ERRORS = errors
    transcript = tmp_path / "actual-cli-transcript.jsonl"
    with transcript.open("xb"):
        pass
    observation: dict[str, Any] = dict(
        test_root=str(tmp_path), calls=logs, actual_child_method_errors=errors, completed=False
    )
    ACTUAL_CLI_OBSERVATIONS.append(observation)
    real = client()
    counts = real.child_totals
    assert not real.jobs
    new_args = [
        "new",
        "--recipe",
        str(recipe_file),
        "--base",
        str(tmp_path),
        "--root",
        str(tmp_path / "operators"),
    ]
    created = _invoke(new_args, logs, counts, transcript, effect="loads")
    path = Path(created["operator_path"])
    assert created["acknowledged_phases"] == created["step"] == 0
    assert created["best_step"] is None and created["scientific_acceptance"] is False
    initial_tree = _tree(path)
    assert _invoke(new_args, logs, counts, transcript) == created
    assert _tree(path) == initial_tree
    best = _invoke(["best", "--operator", str(path)], logs, counts, transcript)
    assert best["payload"]["validated_best_step"] is None
    phase_results: list[dict[str, Any]] = []
    phase_requests: list[dict[str, Any]] = []
    request_files: list[Path] = []
    expected_phases = (
        "train",
        "validation",
        "train",
        "validation",
        "checkpoint",
        "train",
        "validation",
        "checkpoint",
    )
    state = created
    for index, training_phase in enumerate(expected_phases):
        assert state["phase"] == training_phase and state["acknowledged_phases"] == index
        request_file = tmp_path / f"phase-{index:02d}.request.json"
        request_args = [
            "phase-request",
            "--operator",
            str(path),
            "--expected-previous-sha256",
            state["head_sha256"],
            "--phase",
            training_phase,
            "--out",
            str(request_file),
        ]
        request = _invoke(request_args, logs, counts, transcript)
        assert request_file.read_bytes() == canonical_json_file_bytes(request)
        assert _invoke(request_args, logs, counts, transcript) == request
        request_files.append(request_file)
        phase_requests.append(request)
        native_command = (
            "resume"
            if index == 2
            else ("validate" if training_phase == "validation" else training_phase)
        )
        command_args = [
            native_command,
            "--operator",
            str(path),
            "--request",
            str(request_file),
            "--base",
            str(tmp_path),
            "--maximum-replay-phases",
            "8",
        ]
        if index == 0:
            before = _tree(path)
            _invoke(["validate", *command_args[1:]], logs, counts, transcript, expected=2)
            assert _tree(path) == before
        result = _invoke(command_args, logs, counts, transcript, effect="forward")
        assert result["payload"]["request"] == request
        event = result["payload"]["event"]
        assert event["payload"]["record"]["kind"] == training_phase
        assert event["payload"]["sequence"] == index + 1
        assert event["payload"]["previous_sha256"] == state["head_sha256"]
        if training_phase == "train":
            metric = event["payload"]["record"]["metric"]
            assert metric["supervised_token_count"] > 0
            assert metric["gradient_l2_norm"] > 0
            assert metric["nonzero_gradient_parameter_count_before_clip"] > 0
            assert (
                metric["scientific_state_before_sha256"] != metric["scientific_state_after_sha256"]
            )
        phase_results.append(result)
        before = _tree(path)
        replay_args = ["resume", *command_args[1:]]
        assert _invoke(replay_args, logs, counts, transcript) == result
        assert _tree(path) == before
        state = _invoke(["inspect", "--operator", str(path)], logs, counts, transcript)
        assert state["head_sha256"] == event["sha256"]
        assert state["step"] == event["payload"]["after"]["step"]
        best = _invoke(["best", "--operator", str(path)], logs, counts, transcript)
        if index == 1:
            assert best["payload"]["validated_best_step"] == 1
            assert best["payload"]["validated_best_is_persisted"] is False
            refused = tmp_path / "must-not-create-early-export.json"
            _invoke(
                [
                    "export-request",
                    "--operator",
                    str(path),
                    "--checkpoint-event-sha256",
                    "f" * 64,
                    "--lifecycle-sha256",
                    "f" * 64,
                    "--out",
                    str(refused),
                ],
                logs,
                counts,
                transcript,
                expected=2,
            )
            assert not refused.exists()
    assert state["phase"] == "stopped" and state["step"] == 3
    assert state["acknowledged_phases"] == 8 and state["best_step"] == 1
    assert best["payload"]["validated_best_is_persisted"] is True
    assert best["payload"]["persisted_best_step"] == 1
    trained_tree = _tree(path)
    old_training_result = _invoke(
        [
            "resume",
            "--operator",
            str(path),
            "--request",
            str(request_files[0]),
            "--base",
            str(tmp_path),
            "--maximum-replay-phases",
            "0",
        ],
        logs,
        counts,
        transcript,
    )
    assert old_training_result == phase_results[0] and _tree(path) == trained_tree
    assert _invoke(new_args, logs, counts, transcript) == state
    request_file = tmp_path / "export-request.json"
    request = _invoke(
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
            str(request_file),
        ],
        logs,
        counts,
        transcript,
    )
    exports = tmp_path / "exports"
    export_args = [
        "export",
        "--operator",
        str(path),
        "--request",
        str(request_file),
        "--base",
        str(tmp_path),
        "--root",
        str(exports),
    ]
    published_export = _invoke(export_args, logs, counts, transcript, effect="loads")
    assert published_export["result"]["payload"]["request"] == request
    assert published_export["result"]["payload"]["selected_completed_steps"] == 1
    export_tree = _tree(exports)
    assert _invoke(export_args, logs, counts, transcript) == published_export
    export_inspection = _invoke(
        [
            "inspect-export",
            "--operator",
            str(path),
            "--request",
            str(request_file),
            "--root",
            str(exports),
        ],
        logs,
        counts,
        transcript,
    )
    assert export_inspection["result"] == published_export["result"]
    assert export_inspection["read_only"] is True
    before_counts = counts.copy()
    source = binding.read_scientific_ablation_operator_generation_source_v1(
        path,
        request=export.read_scientific_ablation_export_request_v1(request_file),
        export_root=exports,
    )
    assert counts == before_counts
    assert _tree(path) == trained_tree and _tree(exports) == export_tree
    control = WorkspaceInterventionPlanV1(
        interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
    )
    payload = ScientificAblationWorkflowIntentV1.model_validate(
        data_intent(tmp_path).model_dump(mode="python")
        | dict(
            source_sha256=source.bundle.manifest.sha256,
            lifecycle_sha256=request["payload"]["lifecycle_sha256"],
            inference_sha256=published_export["result"]["payload"]["inference_sha256"],
            selected_checkpoint_sha256=request["payload"]["source_checkpoint_sha256"],
            lifecycle_path=str(source.lifecycle_path),
            inference_path=str(source.inference_path),
            inputs_root=str(tmp_path),
            training=config.train.model_dump(mode="python"),
            validation=config.validation.model_dump(mode="python"),
            test=test.model_dump(mode="python"),
            controls=(control,),
            resources=ScientificAblationWorkflowResourcesV1(
                maximum_storage_bytes=2 * 1024**3,
                minimum_free_bytes=1024**3,
            ),
            python_policy=_POLICY,
        )
    )
    assert payload.resources.maximum_writer_report_bytes == 512 * 1024**2
    payload_file, plan_file = (
        tmp_path / "workflow-payload.json",
        tmp_path / "workflow-plan.json",
    )
    _save(payload_file, payload.model_dump(mode="json"))
    root = tmp_path / "commands"
    anchors: tuple[str, ...] = ()
    common = _common(path, request_file, exports, plan_file, tmp_path, root, anchors)
    plan_data = _invoke(
        ["workflow", *common, "plan-request", "--payload", str(payload_file)],
        logs,
        counts,
        transcript,
    )
    plan = IrisScientificAblationWorkflowPlanV1.model_validate_json(plan_file.read_bytes())
    assert plan_data == plan.model_dump(mode="json")
    status = _invoke(["workflow", *common, "status"], logs, counts, transcript)
    store = root / COMMAND_WORKFLOW_NAMESPACE / status["definition"]["sha256"]
    phases = scientific_ablation_workflow_phases_v1(plan)
    assert len(phases) == 8 and plan.payload.controls == (control,)
    phase_calls: list[dict[str, Any]] = []
    for phase in phases:
        common = _common(path, request_file, exports, plan_file, tmp_path, root, anchors)
        command_file = tmp_path / f"{phase.name}.command.json"
        command_args = [
            "workflow",
            *common,
            "command-request",
            "--phase",
            phase.name,
            "--out",
            str(command_file),
        ]
        if phase.kind in ("capture", "score"):
            planning_file = tmp_path / f"{phase.name}.planning.json"
            chunk_file = tmp_path / f"{phase.name}.chunk.json"
            _invoke(
                [
                    "workflow",
                    *common,
                    "kernel-request",
                    "--phase",
                    phase.name,
                    "--initial",
                    "--rows-per-call",
                    "8",
                    "--out",
                    str(planning_file),
                ],
                logs,
                counts,
                transcript,
            )
            chunk = _invoke(
                [
                    "workflow",
                    *common,
                    "kernel-plan",
                    "--request",
                    str(planning_file),
                    "--out",
                    str(chunk_file),
                ],
                logs,
                counts,
                transcript,
                effect="loads" if phase.kind == "capture" else "data",
            )
            assert chunk["total_rows"] == chunk["stop_row_index"] == 8
            assert chunk["acknowledged_rows"] == 0
            command_args.extend(("--expected-chunk", str(chunk_file), "--rows-per-call", "8"))
        command = _invoke(command_args, logs, counts, transcript)
        assert command["payload"]["phase"] == phase.model_dump(mode="json")
        assert tuple(command["payload"]["completed_phase_sha256"]) == anchors
        operation = {
            "capture": "generate",
            "score": "score",
            "diagnostics": "diagnose",
            "workspace": "diagnose",
            "message": "diagnose",
            "publish": "publish",
        }[phase.kind]
        dispatch = ["workflow", *common, operation, "--command", str(command_file)]
        if phase.kind == "score":
            before = _tree(store)
            _invoke(dispatch, logs, counts, transcript, expected=2)
            assert _tree(store) == before
            dispatch.append("--allow-isolated-python")
        reply = _invoke(
            dispatch,
            logs,
            counts,
            transcript,
            effect="forward"
            if phase.kind in ("capture", "diagnostics", "workspace", "message")
            else "data",
        )
        assert reply["delivery"] == "original_reference"
        next_anchors = tuple(reply["progress"]["completed_phase_sha256"])
        assert next_anchors[:-1] == anchors and len(next_anchors) == phase.index + 1
        anchors = next_anchors
        data_files = _tree(store / f"phase-{phase.index:03d}")
        record = IrisScientificAblationWorkflowPhaseDataV1.model_validate_json(
            data_files["complete.json"]
        )
        assert record.sha256 == anchors[-1] and record.payload.phase == phase
        assert record.payload.origin != "not_applicable_no_next_cycle"
        assert set(data_files) == {"complete.json"} | {file.name for file in record.payload.files}
        for member in record.payload.files:
            assert len(data_files[member.name]) == member.bytes
            assert sha256(data_files[member.name]).hexdigest() == member.sha256
        phase_calls.append(
            dict(
                phase=phase.model_dump(mode="json"),
                command=command,
                reply=reply,
                record=record.model_dump(mode="json"),
            )
        )
        current = _common(path, request_file, exports, plan_file, tmp_path, root, anchors)
        before = _tree(store)
        replay = _invoke(
            [
                "workflow",
                *current,
                operation,
                "--command",
                str(command_file),
                "--expected-operator-ack-sha256",
                reply["operator_ack"]["sha256"],
            ],
            logs,
            counts,
            transcript,
        )
        assert replay["delivery"] == "saved_original_data"
        for field in ("operator_ack", "progress", "kernel_progress"):
            assert replay[field] == reply[field]
        assert _tree(store) == before
        status = _invoke(["workflow", *current, "status"], logs, counts, transcript)
        assert tuple(status["progress"]["completed_phase_sha256"]) == anchors
    assert status["progress"]["complete"] is True
    assert status["progress"]["next_phase"] is None
    assert len(anchors) == len(phase_calls) == 8
    assert _tree(path) == trained_tree and _tree(exports) == export_tree
    assert all((store / "published" / name).is_dir() for name in ("workspace", "message"))
    summary = json.loads((store / "phase-007" / "summary.json").read_bytes())
    assert summary["scientific_acceptance"] == "not_assessed"
    assert set(summary["published"]) == {"workspace", "message"}
    assert len(real.jobs) == len(logs) == 103 and not errors
    observation.update(completed=True, child_counts=counts.copy(), phase_anchors=anchors)
    _save(
        tmp_path / "actual-full-console-cli-witness.json",
        dict(
            kind="iris-full-console-cli-actual-witness-v1",
            operator_path=str(path),
            base=str(tmp_path),
            export_root=str(exports),
            workflow_path=str(store),
            recipe=config.model_dump(mode="json"),
            actual_training_requests=phase_requests,
            actual_training_results=phase_results,
            terminal_operator=state,
            best=best,
            export_request=request,
            published_export=published_export,
            plan=plan.model_dump(mode="json"),
            phase_anchors=anchors,
            actual_workflow_phases=phase_calls,
            publication_summary=summary,
            actual_CLI_calls=logs,
            actual_child_method_errors=errors,
            cache_fixture=str(cache_ready),
            scientific_default_or_isolated_Python_acceptance=False,
            actual_console_caller_pid=os.getpid(),
            actual_console_job_directories=[row["job_directory"] for row in real.jobs],
        ),
    )
