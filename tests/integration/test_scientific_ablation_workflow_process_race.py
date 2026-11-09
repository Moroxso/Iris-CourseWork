"""Genuine overlapping public recovery processes, BOTH original locks, ALL4 kernels.

Two complete cases: partial original0/budget3/stop3 and all-ACK aggregate-only.
The separate inner-lock attempt is an OS probe, NOT a direct native API race.
No generated Python, copied runtime outputs, quota increase or JSON ownership.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import subprocess
import sys
from datetime import UTC, datetime
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
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._recovery_race_packets import binding, publish, read, signature, wait
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
Mode = Literal["crash", "holder", "contender", "replay"]


def _argv(job: Path, mode: Mode, output: Path) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-B",
        "-u",
        os.environ["IRIS_RACE_WORKER_PATH"],
        os.environ["IRIS_RACE_BEFORE_PATH"],
        os.environ["IRIS_RACE_BEFORE_SHA256"],
        str(job),
        sha256(job.read_bytes()).hexdigest(),
        mode,
        str(output),
    ]


def _start_worker(job: Path, mode: Mode, output: Path) -> subprocess.Popen[bytes]:
    """The ONLY original collected callsite admitted by the pinned process guard."""
    assert not any(
        output.with_suffix(s).exists()
        for s in (".json", ".started.json", ".process.json", ".stdout.log", ".stderr.log")
    )
    return subprocess.Popen(
        _argv(job, mode, output),
        cwd=str(job.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _finish_worker(
    child: subprocess.Popen[bytes],
    job: Path,
    mode: Mode,
    output: Path,
) -> dict[str, Any]:
    stdout, stderr = child.communicate(timeout=1860)
    assert (
        type(stdout) is type(stderr) is bytes and len(stdout) <= 1024**2 and len(stderr) <= 1024**2
    )
    for suffix, raw in ((".stdout.log", stdout), (".stderr.log", stderr)):
        with output.with_suffix(suffix).open("xb") as stream:
            stream.write(raw)
    native_raw = output.read_bytes() if output.exists() else None
    native = None if native_raw is None else json.loads(native_raw)
    receipt: dict[str, Any] = dict(
        actual_launch_pid=child.pid,
        actual_observer_pid=os.getpid(),
        actual_exit_code=child.returncode,
        argv=_argv(job, mode, output),
        stdout_sha256=sha256(stdout).hexdigest(),
        stderr_sha256=sha256(stderr).hexdigest(),
        worker_receipt_sha256=None if native_raw is None else sha256(native_raw).hexdigest(),
        native=native,
    )
    with output.with_suffix(".process.json").open("xb") as stream:
        stream.write(canonical_json_file_bytes(receipt))
    assert child.returncode == (83 if mode == "crash" else 0), stderr.decode(errors="replace")
    assert type(native) is dict and canonical_json_file_bytes(native) == native_raw
    assert native["mode"] == mode and native["job_sha256"] == receipt["argv"][8]
    assert (native["pid"] == child.pid and native["parent_pid"] == os.getpid()) or (
        native["pid"] != child.pid and native["parent_pid"] == child.pid
    )
    return receipt


def _process_identity(native: dict[str, Any]) -> tuple[int, datetime]:
    """Identify an observed worker lifetime, not a recyclable OS PID alone.

    This is observation DATA, never native ownership. The caller separately
    binds each original Popen handle, argv, startup bytes and worker receipt.
    """
    pid, stamp = native["pid"], native["started_utc"]
    if type(pid) is not int or pid <= 0 or type(stamp) is not str:
        raise ValueError("Exact positive PID and original UTC start string required")
    started = datetime.fromisoformat(stamp)
    if started.tzinfo is None or started.utcoffset() != UTC.utcoffset(None):
        raise ValueError("Original timezone-aware UTC worker start required")
    return pid, started.astimezone(UTC)


def _tree(root: Path, locked: set[Path]) -> dict[str, Any]:
    """Read everything; only the TWO live ownership bytes use identity-only views.

    This avoids requiring cross-process byte reads through OS byte-range locks.
    The holder verifies their bytes via its OWN original descriptors, and full
    post-release DATA auditing must read their complete bytes as well.
    """
    files, directories = {}, {}
    for path in (root, *sorted(root.rglob("*"))):
        info = path.lstat()
        assert not getattr(info, "st_file_attributes", 0) & 0x400 and not path.is_symlink()
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            directories[name] = dict(
                signature=signature(info), members=sorted(p.name for p in path.iterdir())
            )
        else:
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            content = None if path in locked else path.read_bytes()
            assert signature(path.lstat()) == signature(info)
            files[name] = dict(
                signature=signature(info),
                sha256=None if content is None else sha256(content).hexdigest(),
            )
    assert len([p for p in locked if p.is_relative_to(root)]) == 2
    return dict(
        files=files,
        directories=directories,
        identity_only_live_lock_paths=sorted(str(p) for p in locked),
    )


@pytest.mark.parametrize("scenario", ("partial", "aggregate"))
def test_genuine_all4_overlapping_recovery_excludes_contender_before_native_or_writes(
    multirow_bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
    scenario: str,
) -> None:
    path, base, exports, plan = multirow_bound
    request = _intent(path)
    anchors: tuple[str, ...] = ()
    cases: list[dict[str, Any]] = []
    pids: set[int] = set()
    identities: set[tuple[int, datetime]] = set()
    before_file = Path(os.environ["IRIS_RACE_BEFORE_PATH"])
    before = json.loads(before_file.read_bytes())
    expected_rows = 1 if scenario == "partial" else 4
    stop = 3 if scenario == "partial" else 4
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
        assert (
            len(kernel_plan.payload.rows) == 4
            and plan.payload.resources.maximum_storage_bytes == 16 * 1024**2
        )
        command = build(
            Intent(
                namespace=facade.definition,
                phase=phase,
                completed_phase_sha256=anchors,
                expected_chunk=Chunk(
                    kernel_plan_sha256=kernel_plan.sha256,
                    acknowledgement_head_sha256=module._genesis(kernel_plan.sha256),
                    acknowledged_rows=0,
                    total_rows=4,
                    rows_per_call=stop,
                    stop_row_index=stop,
                ),
                rows_per_call=stop,
            )
        )
        stem = f"phase-{phase.index:02d}"
        ipc = tmp_path / (stem + ".ipc")
        ipc.mkdir()
        job = dict(
            kind="iris-trusted-original-recovery-process-race-job-v1",
            operator_path=str(path),
            request=request.model_dump(mode="json"),
            export_root=str(exports),
            plan=plan.model_dump(mode="json"),
            base=str(base),
            root=str(tmp_path / "commands"),
            command=command.model_dump(mode="json"),
            expected_definition_sha256=facade.definition.sha256,
            scenario=scenario,
            nonce=secrets.token_hex(16),
            ipc_root=str(ipc),
        )
        job_file = tmp_path / (stem + ".input.json")
        with job_file.open("xb") as stream:
            stream.write(canonical_json_file_bytes(job))
        job_sha = sha256(job_file.read_bytes()).hexdigest()
        bound = binding(
            dict(
                job_sha256=job_sha,
                command_sha256=command.sha256,
                nonce=job["nonce"],
                phase=phase.index,
                scenario=scenario,
            )
        )
        crash_output = tmp_path / (stem + ".crash.json")
        crashed = _finish_worker(
            _start_worker(job_file, "crash", crash_output), job_file, "crash", crash_output
        )
        crash = crashed["native"]
        assert (
            crash["actual_row_calls"]
            == crash["actual_published_rows"]
            == list(range(expected_rows))
        )
        assert not crash["actual_kernel_aggregate_publications"]
        observed = facade.inspect_kernel(
            expected_phase=phase, expected_kernel_plan_sha256=kernel_plan.sha256
        )
        assert (
            observed.acknowledged_rows == expected_rows and observed.kernel_complete_sha256 is None
        )
        kernel = Path(crash["kernel_path"])
        row_prefix = "row" if phase.kind == "capture" else "score"
        original_pins = {
            str(i): _row_pins(kernel / f"{row_prefix}-{i:05d}") for i in range(expected_rows)
        }
        holder_output = tmp_path / (stem + ".holder.json")
        holder = _start_worker(job_file, "holder", holder_output)

        def holder_is_alive(child: subprocess.Popen[bytes] = holder) -> bool:
            return child.poll() is None

        try:
            ready, ready_sha = wait(ipc, "ready", bound, alive=holder_is_alive)
        except BaseException as primary:
            if holder.poll() is not None:
                try:
                    _finish_worker(holder, job_file, "holder", holder_output)
                except AssertionError:
                    # A failed child is expected on this error path; its actual
                    # stdout/stderr/exit report was written BEFORE the assertion.
                    pass
                except BaseException as retention_error:
                    raise primary from retention_error
            raise
        assert ready["source_sha256"] == before["source_sha256"]
        assert (
            ready["worker_sha256"]
            == before["helper_pins"][Path(os.environ["IRIS_RACE_WORKER_PATH"]).name]
        )
        assert (
            ready["started_sha256"]
            == sha256(holder_output.with_suffix(".started.json").read_bytes()).hexdigest()
        )
        assert ready["pid"] == holder.pid or ready["parent_pid"] == holder.pid
        locks = {Path(item["path"]) for item in ready["locks"].values()}
        assert locks == {
            facade._seal.store.path / ".iris-owner.lock",
            kernel.parent / ".iris-owner.lock",
        }
        paused_tree = _tree(facade._seal.store.path, locks)
        assert (
            sum(item["signature"][3] for item in paused_tree["files"].values())
            <= plan.payload.resources.maximum_storage_bytes
        )
        contender_output = tmp_path / (stem + ".contender.json")
        contender = _finish_worker(
            _start_worker(job_file, "contender", contender_output),
            job_file,
            "contender",
            contender_output,
        )
        excluded = contender["native"]
        assert excluded["actual_outer_lock_excluded_public_recover"] is True
        assert excluded["separate_inner_OS_lock_probe_excluded"] is True
        assert not excluded["actual_row_calls"] and not excluded["actual_published_rows"]
        assert not excluded["actual_kernel_aggregate_publications"] and not any(
            excluded["calls"].values()
        )
        assert holder.poll() is None and _tree(facade._seal.store.path, locks) == paused_tree
        assert read(ipc, "ready", bound) == (ready, ready_sha)
        contender_sha = sha256(
            contender_output.with_suffix(".process.json").read_bytes()
        ).hexdigest()
        release = dict(
            kind="iris-recovery-race-release-v1",
            binding=bound,
            ready_sha256=ready_sha,
            contender_process_sha256=contender_sha,
        )
        release_sha = publish(ipc, "release", release, bound)
        held = _finish_worker(holder, job_file, "holder", holder_output)
        result = held["native"]
        assert result["both_original_locks_checked_before_and_after_actual_contender"] is True
        assert result["ready_sha256"] == ready_sha and result["release_sha256"] == release_sha
        assert result["contender_process_sha256"] == contender_sha
        assert (
            result["recovery_result"]["payload"]["original_return"]
            == "unavailable_not_reconstructed"
        )
        assert result["recovery_result"]["payload"]["kernel_progress"]["acknowledged_rows"] == stop
        if scenario == "partial":
            assert result["recovery_row_calls"] == [1, 2]
            assert result["actual_row_calls"] == result["actual_published_rows"] == [1, 2, 3]
            assert result["no_row3_before_new_command"] is True
            assert result["completed_command"] != job["command"]
        else:
            assert result["actual_row_calls"] == result["actual_published_rows"] == []
            assert not any(result["calls"].values())
        assert result["actual_kernel_aggregate_publications"] == [str(kernel / "complete")]
        assert {
            str(i): _row_pins(kernel / f"{row_prefix}-{i:05d}") for i in range(expected_rows)
        } == original_pins
        replay_output = tmp_path / (stem + ".replay.json")
        replayed = _finish_worker(
            _start_worker(job_file, "replay", replay_output), job_file, "replay", replay_output
        )
        retained = replayed["native"]
        for field in ("recovery_intent", "recovery_result", "recovery_operator_ack"):
            assert retained[field] == result[field]
        assert not any(retained["calls"].values())
        assert retained["actual_row_calls"] == retained["actual_published_rows"] == []
        assert retained["actual_kernel_aggregate_publications"] == []
        assert retained["all_row_pins"] == result["all_row_pins"]
        old_anchors = anchors
        anchors = tuple(result["phase_anchors"])
        assert anchors[: phase.index] == old_anchors and len(anchors) == phase.index + 1
        assert sha256(job_file.read_bytes()).hexdigest() == job_sha
        for packet in (crashed, held, contender, replayed):
            identity = _process_identity(packet["native"])
            assert identity not in identities
            identities.add(identity)
            pids.add(identity[0])
        cases.append(
            dict(
                phase=phase.model_dump(mode="json"),
                original_command=job["command"],
                immutable_job_sha256=job_sha,
                original_acknowledged_rows=expected_rows,
                original_row_pins=original_pins,
                ready=ready,
                ready_sha256=ready_sha,
                paused_workflow_tree_before_and_after_contender=paused_tree,
                actual_holder_live_before_and_after_contender=True,
                release=release,
                release_sha256=release_sha,
                crash_process=crashed,
                holder_process=held,
                contender_process=contender,
                replay_process=replayed,
            )
        )
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
        reopened._seal.store.read(phase, expected_sha256=anchors[-1])
        handoff = reopened._seal.store.path / "handoffs" / command.sha256[:2] / command.sha256
        assert {item.name for item in handoff.iterdir()} == {"intent.json"}
    assert len(cases) == len(anchors) == 4 and len(identities) == 16
    (tmp_path / "actual-all4-process-race-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                base=str(base),
                export_root=str(exports),
                workflow_path=str(reopened._seal.store.path),
                plan=plan.model_dump(mode="json"),
                phase_anchors=anchors,
                scenario=scenario,
                actual_process_race_cases=cases,
                actual_worker_pids=sorted(pids),
                actual_worker_process_identities=[
                    dict(pid=pid, started_utc=started.isoformat())
                    for pid, started in sorted(identities)
                ],
                genuine_overlapping_public_recover_processes=True,
                BOTH_original_OS_locks_held_across_actual_contender=True,
                inner_attempt_is_only_OS_lock_probe_not_direct_native_API_race=True,
                no_original_return_reconstructed=True,
                concurrent_process_race_native_observation=True,
                independent_DATA_installed_or_scientific_acceptance=False,
            )
        )
    )
