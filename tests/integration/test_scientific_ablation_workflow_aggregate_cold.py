"""Real OS exit83 after LAST ACK + FRESH DATA-only finalization for all FOUR kernels.

Four non-Python rows, original16MiB, immutable0/budget4/stop4. No serialized owner,
stage adoption, row/NN replay, original-return reconstruction or model download.
This gate is sequential hard-exit/cold recovery, NOT an OS competition/race proof.
"""

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_recovery_v1 import (
    IrisScientificAblationWorkflowRecoveredResultV1,
    IrisScientificAblationWorkflowRecoveryAckV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_operator_export import _intent
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    _row_pins,
)
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    cache_ready as cache_ready,
)
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    completed as completed,
)
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    multirow_bound as multirow_bound,
)
from tests.integration.test_scientific_ablation_workflow_public_chunk import _binding

pytestmark = pytest.mark.integration


def _launch_worker(job: Path, mode: Literal["crash", "resume"], output: Path) -> dict[str, Any]:
    """The ONLY parent subprocess callsite admitted by the pinned process guard."""
    worker = Path(os.environ["IRIS_COLD_WORKER_PATH"])
    before = Path(os.environ["IRIS_COLD_BEFORE_PATH"])
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-u",
        str(worker),
        str(before),
        os.environ["IRIS_COLD_BEFORE_SHA256"],
        str(job),
        sha256(job.read_bytes()).hexdigest(),
        mode,
        str(output),
    ]
    assert not output.exists()
    child = subprocess.Popen(
        argv,
        cwd=str(job.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = child.communicate(timeout=1860)
    assert (
        type(stdout) is type(stderr) is bytes and len(stdout) <= 1024**2 and len(stderr) <= 1024**2
    )
    for suffix, raw in (("stdout.log", stdout), ("stderr.log", stderr)):
        with output.with_suffix("." + suffix).open("xb") as stream:
            stream.write(raw)
    expected = 83 if mode == "crash" else 0
    assert child.returncode == expected, stderr.decode(errors="replace")
    raw = output.read_bytes()
    value = json.loads(raw)
    assert canonical_json_file_bytes(value) == raw
    assert value["mode"] == mode and value["job_sha256"] == argv[8]
    assert (value["pid"] == child.pid and value["parent_pid"] == os.getpid()) or (
        value["pid"] != child.pid and value["parent_pid"] == child.pid
    )
    receipt = dict(
        actual_launch_pid=child.pid,
        actual_observer_pid=os.getpid(),
        actual_exit_code=child.returncode,
        argv=argv,
        stdout_sha256=sha256(stdout).hexdigest(),
        stderr_sha256=sha256(stderr).hexdigest(),
        worker_receipt_sha256=sha256(raw).hexdigest(),
        native=value,
    )
    with output.with_suffix(".process.json").open("xb") as stream:
        stream.write(canonical_json_file_bytes(receipt))
    return receipt


def test_genuine_all4_last_ack_hard_exit_then_fresh_process_data_only_aggregate_recovery(
    multirow_bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1], tmp_path: Path
) -> None:
    path, base, exports, plan = multirow_bound
    request = _intent(path)
    anchors: tuple[str, ...] = ()
    cases: list[dict[str, Any]] = []
    actual_pids: set[int] = set()
    for phase in scientific_ablation_workflow_phases_v1(plan)[:4]:
        facade = public.create_scientific_ablation_operator_command_workflow_v1(
            path,
            request=request,
            export_root=exports,
            plan=plan,
            base=base,
            root=tmp_path / "commands",
            completed_phase_sha256=anchors,
        )
        original = facade._seal.job
        module, builder, _ = _binding(phase)
        if phase.kind == "capture":
            heldout = steps._owner(plan, phase, original._seal.factory)
            kernel_plan = getattr(module, builder)(
                heldout, expected_input_sha256=heldout.manifest.sha256
            )
        else:
            prefix = tuple(
                original._seal.store.read(p, expected_sha256=d)
                for p, d in zip(scientific_ablation_workflow_phases_v1(plan), anchors, strict=False)
            )
            kernel_plan = getattr(module, builder)(
                **steps._score_args(original._seal.store, prefix[-1])
            )
        assert len(kernel_plan.payload.rows) == 4
        interval = Chunk(
            kernel_plan_sha256=kernel_plan.sha256,
            acknowledgement_head_sha256=module._genesis(kernel_plan.sha256),
            acknowledged_rows=0,
            total_rows=4,
            rows_per_call=4,
            stop_row_index=4,
        )
        command = build(
            Intent(
                namespace=facade.definition,
                phase=phase,
                completed_phase_sha256=anchors,
                expected_chunk=interval,
                rows_per_call=4,
            )
        )
        job = dict(
            kind="iris-trusted-original-all-ack-aggregate-cold-job-v1",
            operator_path=str(path),
            request=request.model_dump(mode="json"),
            export_root=str(exports),
            plan=plan.model_dump(mode="json"),
            base=str(base),
            root=str(tmp_path / "commands"),
            command=command.model_dump(mode="json"),
            expected_definition_sha256=facade.definition.sha256,
        )
        job_path = tmp_path / f"phase-{phase.index:02d}.input.json"
        with job_path.open("xb") as stream:
            stream.write(canonical_json_file_bytes(job))
        job_sha = sha256(job_path.read_bytes()).hexdigest()
        crashed = _launch_worker(
            job_path, "crash", tmp_path / f"phase-{phase.index:02d}.crash.json"
        )
        c = crashed["native"]
        assert (
            c["status"] == "hard_exit_after_last_ack_before_aggregate"
            and c["actual_exit_requested"] == 83
        )
        assert c["actual_row_calls"] == c["actual_published_rows"] == [0, 1, 2, 3]
        assert c["no_aggregate_before_fresh_recovery"] is True
        assert c["actual_kernel_aggregate_publications"] == []
        assert c["fresh_process_registry_was_empty"] is c["no_serialized_owner_or_stage"] is True
        observed = facade.inspect_kernel(
            expected_phase=phase, expected_kernel_plan_sha256=kernel_plan.sha256
        )
        assert observed.acknowledged_rows == 4 and observed.kernel_complete_sha256 is None
        kernel = Path(c["kernel_path"])
        row_prefix = "row" if phase.kind == "capture" else "score"
        assert not (kernel / "complete").exists()
        pins = {str(i): _row_pins(kernel / f"{row_prefix}-{i:05d}") for i in range(4)}
        assert json.loads(canonical_json_file_bytes(pins)) == c["all_row_pins"]
        resumed = _launch_worker(
            job_path, "resume", tmp_path / f"phase-{phase.index:02d}.resume.json"
        )
        r = resumed["native"]
        assert r["status"] == "cold_all_ack_aggregate_finalized"
        assert r["fresh_process_registry_was_empty"] is r["no_serialized_owner_or_stage"] is True
        assert c["pid"] != r["pid"] and c["parent_pid"] != r["pid"]
        assert r["actual_row_calls"] == r["actual_published_rows"] == []
        assert not any(r["calls"].values())
        assert r["actual_kernel_aggregate_publications"] == [str(kernel / "complete")]
        assert r["aggregate_was_absent_on_fresh_entry"] is True
        assert r["all_acknowledged_rows_unchanged"] is True
        assert r["no_factory_or_row_or_NN_on_aggregate_recovery"] is True
        assert r["no_original_return_reconstructed"] is True
        assert "completed_original_return" not in r and "completed_command" not in r
        assert r["no_native_on_DATA_retry"] is True
        assert r["original_command"] == c["original_command"] == job["command"]
        assert r["recovery_result"]["payload"]["resumed_from_row"] is None
        assert r["recovery_result"]["payload"]["route"] == "verified_kernel_data"
        assert r["recovery_result"]["payload"]["kernel_progress"]["acknowledged_rows"] == 4
        assert {str(i): _row_pins(kernel / f"{row_prefix}-{i:05d}") for i in range(4)} == pins
        assert json.loads(canonical_json_file_bytes(pins)) == r["all_row_pins"]
        assert sha256(job_path.read_bytes()).hexdigest() == job_sha
        actual_pids.update((c["pid"], r["pid"]))
        result = IrisScientificAblationWorkflowRecoveredResultV1.model_validate_json(
            canonical_json_file_bytes(r["recovery_result"])
        )
        ack = IrisScientificAblationWorkflowRecoveryAckV1.model_validate_json(
            canonical_json_file_bytes(r["recovery_operator_ack"])
        )
        assert ack.payload.original_request_sha256 == command.sha256
        assert ack.payload.recovered_result_sha256 == result.sha256
        assert result.payload.original_return == "unavailable_not_reconstructed"
        assert result.payload.phase_record is not None
        assert result.payload.kernel_progress.kernel_complete_sha256 is not None
        before = anchors
        anchors = tuple(r["phase_anchors"])
        assert anchors == (*before, result.payload.phase_record.sha256)
        assert anchors == ack.payload.completed_phase_sha256
        assert len(anchors) == phase.index + 1
        reopened = public.create_scientific_ablation_operator_command_workflow_v1(
            path,
            request=request,
            export_root=exports,
            plan=plan,
            base=base,
            root=tmp_path / "commands",
            completed_phase_sha256=anchors,
        )
        assert reopened.progress.completed_phase_sha256 == anchors
        actual = reopened._seal.store.read(phase, expected_sha256=anchors[-1])
        assert actual.record == result.payload.phase_record
        handoff = reopened._seal.store.path / "handoffs" / command.sha256[:2] / command.sha256
        assert {file.name for file in handoff.iterdir()} == {"intent.json"}
        cases.append(
            dict(
                phase=phase.model_dump(mode="json"),
                original_command=job["command"],
                last_ack_observation=observed.model_dump(mode="json"),
                all_rows_before=pins,
                crash_process=crashed,
                resume_process=resumed,
                immutable_job_sha256=job_sha,
            )
        )
    assert len(cases) == len(anchors) == 4 and len(actual_pids) == 8
    (tmp_path / "actual-all4-last-ack-cold-recovery-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                base=str(base),
                export_root=str(exports),
                workflow_path=str(reopened._seal.store.path),
                plan=plan.model_dump(mode="json"),
                phase_anchors=anchors,
                actual_aggregate_cold_cases=cases,
                actual_worker_pids=sorted(actual_pids),
                genuine_hard_exit_and_fresh_process=True,
                all_acknowledged_rows_unchanged=True,
                no_factory_or_row_or_NN_on_aggregate_recovery=True,
                no_original_return_reconstructed=True,
                generated_candidate_or_scientific_acceptance=False,
                concurrent_process_race_acceptance=False,
            )
        )
    )
