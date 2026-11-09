"""Fresh CPU job, complete two-file storage and cold read; not answer quality."""

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_writer_io as module
from iris.application import scientific_ablation_writer_reports as service
from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
)
from iris.application.scientific_ablation_inference_io import _read_file, _regular, _signature
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    WriterSourceProtocol,
    scientific_ablation_writer_source_bytes_v1,
    scientific_ablation_writer_source_preflight_v1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready
from tests.integration.test_scientific_ablation_diagnostics import _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make

pytestmark = pytest.mark.integration
WORKER = Path(__file__).with_name("_scientific_ablation_writer_io_worker.py")


@dataclass(frozen=True, slots=True)
class Job:
    completed: service.WriterCompleted
    protocol: WriterSourceProtocol
    report_sha256: str
    source_sha256: str
    runtime: ScientificAblationWriterCPURuntimeV1
    source_content: bytes
    report_content: bytes


@pytest.fixture(scope="module", params=(False, True), ids=("workspace", "message"))
def job(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Job:
    message = bool(request.param)
    root = tmp_path_factory.mktemp("actual-writer-io-job")
    ex = _experiment(_make(root / "case", rows=_rows()), root / "scores")
    owner = _heldout(ex)
    model = owner._guard.context._run._optimizer._model
    measure = (
        service.measure_scientific_ablation_message_writer_completed_v1
        if message
        else service.measure_scientific_ablation_workspace_writer_completed_v1
    )
    runtime = ScientificAblationWriterCPURuntimeV1(
        torch_version=str(torch.__version__), torch_git_version=torch.version.git_version
    )
    with patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute:
        completed = measure(
            owner,
            input=_input(ex),
            policy=_policy(message),
            limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
        )
        ready(owner)
        assert execute.call_count == completed.report.payload.actual_teacher_executions
        assert execute.call_count == (20 if message else 24)
    assert completed.runtime == runtime
    pf = scientific_ablation_writer_source_preflight_v1(
        completed.source, expected_sha256=completed.source.sha256
    )
    source_content = scientific_ablation_writer_source_bytes_v1(
        completed.source, expected_sha256=completed.source.sha256
    )
    report_content = scientific_ablation_writer_report_bytes_v1(
        completed.report, preflight=pf, runtime=runtime
    )
    (root / "producer-source.json").write_bytes(source_content)
    (root / "producer-report.json").write_bytes(report_content)
    # Anchors are retained from the authenticated job BEFORE any storage candidate.
    protocol: WriterSourceProtocol = "message" if message else "workspace"
    return Job(
        completed,
        protocol,
        completed.report.sha256,
        completed.source.sha256,
        runtime,
        source_content,
        report_content,
    )


@pytest.fixture(autouse=True)
def no_execution_during_io(job: Job) -> Iterator[None]:
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("No new IO model")),
        patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No IO model call")),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No generated Python execution on Windows"),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No IO weight loading"),
        ),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No IO tokenizer loading"),
        ),
    ):
        yield


def arguments(job: Job) -> dict[str, Any]:
    return dict(
        expected_protocol=job.protocol,
        expected_report_sha256=job.report_sha256,
        expected_source_sha256=job.source_sha256,
        runtime=job.runtime,
        maximum_source_bytes=len(job.source_content),
        maximum_report_bytes=len(job.report_content),
    )


def write(job: Job, root: Path, **changes: Any) -> Path:
    return module.write_scientific_ablation_writer_bundle_v1(
        job.completed.report,
        **(
            dict(source=job.completed.source, root=root, minimum_free_bytes=0, **arguments(job))
            | changes
        ),
    )


def read(job: Job, path: Path, **changes: Any) -> module.ScientificAblationWriterBundleV1:
    return module.read_scientific_ablation_writer_bundle_v1(path, **(arguments(job) | changes))


def tree(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def test_two_whole_files_exact_retry_and_read_without_measurement(job: Job, tmp_path: Path) -> None:
    path = write(job, tmp_path)
    assert path.name == job.report_sha256
    assert path.parent.name == f"scientific-ablation-{job.protocol}-writer-v1"
    assert {entry.name for entry in path.iterdir()} == {"source.json", "report.json"}
    assert (path / "source.json").read_bytes() == job.source_content
    assert (path / "report.json").read_bytes() == job.report_content
    before = tree(tmp_path)
    with patch.object(module, "_write", side_effect=AssertionError("No retry writing")):
        assert write(job, tmp_path) == path
    assert tree(tmp_path) == before
    loaded = read(job, path)
    assert loaded.report == job.completed.report and loaded.source == job.completed.source
    assert tree(tmp_path) == before


@pytest.mark.parametrize(
    "change",
    (
        "source-digest",
        "report-digest",
        "protocol",
        "runtime",
        "source-cap",
        "report-cap",
        "storage-cap",
        "free-reserve",
    ),
)
def test_wrong_independent_anchor_or_cap_denies_before_store_write(
    job: Job, tmp_path: Path, change: str
) -> None:
    changes: dict[str, Any] = {}
    if change == "source-digest":
        changes["expected_source_sha256"] = "f" * 64
    elif change == "report-digest":
        changes["expected_report_sha256"] = "f" * 64
    elif change == "protocol":
        changes["expected_protocol"] = "workspace" if job.protocol == "message" else "message"
    elif change == "runtime":
        changes["runtime"] = ScientificAblationWriterCPURuntimeV1("untrusted", "f" * 40)
    elif change == "source-cap":
        changes["maximum_source_bytes"] = len(job.source_content) - 1
    elif change == "report-cap":
        changes["maximum_report_bytes"] = len(job.report_content) - 1
    elif change == "storage-cap":
        changes["maximum_storage_bytes"] = len(job.source_content) + len(job.report_content)
    else:
        changes["minimum_free_bytes"] = False
    with pytest.raises(module.IrisScientificAblationWriterStorageError):
        write(job, tmp_path, **changes)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ("source.json", "report.json"))
def test_existing_corrupt_content_is_never_overwritten(job: Job, tmp_path: Path, name: str) -> None:
    path = write(job, tmp_path)
    (path / name).write_bytes((path / name).read_bytes() + b" ")
    before = tree(tmp_path)
    with pytest.raises(module.IrisScientificAblationWriterStorageError):
        write(job, tmp_path)
    with pytest.raises(module.IrisScientificAblationWriterStorageError):
        read(job, path)
    assert tree(tmp_path) == before


def test_unusable_source_denies_before_report_file_read(job: Job, tmp_path: Path) -> None:
    path = write(job, tmp_path)
    (path / "source.json").write_bytes(b"{}\n")
    original = _read_file
    accessed: list[str] = []

    def reader(path: Path, maximum: int) -> bytes:
        accessed.append(path.name)
        assert path.name == "source.json", "No unbound report allocation"
        return original(path, maximum)

    with (
        patch.object(module, "_read_file", new=reader),
        pytest.raises(module.IrisScientificAblationWriterStorageError),
    ):
        read(job, path)
    assert accessed == ["source.json"]


@pytest.mark.parametrize("name", ("source.json", "report.json"))
def test_actual_hardlinked_file_denies_read_and_retry(job: Job, tmp_path: Path, name: str) -> None:
    path = write(job, tmp_path)
    os.link(path / name, tmp_path / "caller-link")
    before = tree(tmp_path)
    with pytest.raises(module.IrisScientificAblationWriterStorageError):
        read(job, path)
    with pytest.raises(module.IrisScientificAblationWriterStorageError):
        write(job, tmp_path)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("members", (0, 1, 2))
def test_old_partial_or_complete_stage_is_charged_and_never_adopted(
    job: Job, tmp_path: Path, members: int
) -> None:
    store = tmp_path / f"scientific-ablation-{job.protocol}-writer-v1"
    stage = store / (".iris-writer-" + "a" * 32 + ".tmp")
    stage.mkdir(parents=True)
    if members >= 1:
        (stage / "source.json").write_bytes(job.source_content)
    if members == 2:
        (stage / "report.json").write_bytes(job.report_content)
    before = tree(stage)
    size = len(job.source_content) + len(job.report_content)
    if members:
        with pytest.raises(module.IrisScientificAblationWriterStorageError):
            write(job, tmp_path, maximum_storage_bytes=size + 1)
        assert tree(stage) == before
    path = write(job, tmp_path, maximum_storage_bytes=2 * size + 1)
    assert path.is_dir() and stage.is_dir() and tree(stage) == before
    assert {entry.name for entry in stage.iterdir()} == set(
        () if members == 0 else ("source.json",) if members == 1 else ("source.json", "report.json")
    )


@pytest.mark.parametrize("limit", ("files", "directories"))
def test_prospective_limits_count_both_files_before_stage_creation(
    job: Job, tmp_path: Path, limit: str
) -> None:
    with (
        patch.object(
            module,
            "_MAX_FILES" if limit == "files" else "_MAX_DIRECTORIES",
            2 if limit == "files" else 0,
        ),
        pytest.raises(module.IrisScientificAblationWriterStorageError),
    ):
        write(job, tmp_path)
    store = tmp_path / f"scientific-ablation-{job.protocol}-writer-v1"
    assert {entry.name for entry in store.iterdir()} == {".iris-owner.lock"}


@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
def test_partial_write_preserves_first_signal_and_owned_stage(
    job: Job, tmp_path: Path, signal: type[BaseException]
) -> None:
    failure = signal("writer report write injection")
    original = module._write

    def writer(stage: Any, name: str, content: bytes, claim: Any) -> None:
        if name == "report.json":
            raise failure
        original(stage, name, content, claim)

    with (
        patch.object(module, "_write", new=writer),
        pytest.raises(
            module.IrisScientificAblationWriterStorageError if signal is ValueError else signal
        ) as caught,
    ):
        write(job, tmp_path)
    assert (
        (caught.value.__cause__ is failure) if signal is ValueError else (caught.value is failure)
    )
    store = tmp_path / f"scientific-ablation-{job.protocol}-writer-v1"
    stages = list(store.glob(".iris-writer-*.tmp"))
    assert len(stages) == 1 and not (store / job.report_sha256).exists()
    assert {p.name for p in stages[0].iterdir()} == {"source.json"}
    assert (stages[0] / "source.json").read_bytes() == job.source_content


def test_actual_rename_with_lost_acknowledgement_reads_entire_committed_pair(
    job: Job, tmp_path: Path
) -> None:
    def rename(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        raise OSError("Actual rename completed; acknowledgement lost")

    with patch.object(module, "rename_directory_with_windows_retry", new=rename):
        path = write(job, tmp_path)
    assert not list(path.parent.glob(".iris-writer-*.tmp"))
    assert read(job, path).report == job.completed.report


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_control_signal_after_actual_commit_never_becomes_success(
    job: Job, tmp_path: Path, signal: type[BaseException]
) -> None:
    failure = signal("Lost commit acknowledgement with control signal")

    @contextmanager
    def lock(path: Path) -> Iterator[ProcessFileLock]:
        with process_file_lock(path) as acquired:
            yield acquired
        raise failure

    with patch.object(module, "process_file_lock", new=lock), pytest.raises(signal) as caught:
        write(job, tmp_path)
    assert caught.value is failure
    path = tmp_path / f"scientific-ablation-{job.protocol}-writer-v1" / job.report_sha256
    assert read(job, path).source == job.completed.source


def worker_job(job: Job, tmp_path: Path, phase: str) -> tuple[list[str], Path, Path]:
    inputs = tmp_path / "independently-retained-job"
    inputs.mkdir()
    (inputs / "source.json").write_bytes(job.source_content)
    (inputs / "report.json").write_bytes(job.report_content)
    options = arguments(job)
    options["runtime"] = dict(
        torch_version=job.runtime.torch_version, torch_git_version=job.runtime.torch_git_version
    )
    marker, release = tmp_path / "lock-ready.json", tmp_path / "lock-release"
    packet = dict(
        phase=phase,
        input_root=str(inputs),
        store_root=str(tmp_path / "published"),
        marker=str(marker),
        release=str(release),
        bindings=options,
    )
    data = (json.dumps(packet, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert len(data) <= 64 * 1024
    path = tmp_path / "worker-job.json"
    with path.open("xb") as stream:
        stream.write(data)
    return [sys.executable, "-I", "-B", str(WORKER), str(path)], marker, release


def record_worker(
    tmp_path: Path,
    command: list[str],
    launch_pid: int,
    result: subprocess.CompletedProcess[bytes],
) -> None:
    (tmp_path / "worker.stdout.jsonl").write_bytes(result.stdout)
    (tmp_path / "worker.stderr.log").write_bytes(result.stderr)
    receipt = dict(
        kind="iris-writer-io-actual-child-outer-v1",
        observer_pid=os.getpid(),
        launch_pid=launch_pid,
        actual_returncode=result.returncode,
        command_argv=command,
        worker_sha256=sha256(WORKER.read_bytes()).hexdigest(),
        job_sha256=sha256(Path(command[-1]).read_bytes()).hexdigest(),
        stdout_bytes=len(result.stdout),
        stdout_sha256=sha256(result.stdout).hexdigest(),
        stderr_bytes=len(result.stderr),
        stderr_sha256=sha256(result.stderr).hexdigest(),
    )
    (tmp_path / "worker-outer.json").write_bytes(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )


def worker_events(job: Job, tmp_path: Path, launch_pid: int) -> list[dict[str, Any]]:
    data = (tmp_path / "worker.stdout.jsonl").read_bytes()
    assert 1 <= len(data) <= 256 * 1024
    events = [json.loads(line) for line in data.splitlines()]
    for event in events:
        assert event["kind"] == "iris-writer-io-child-event-v1"
        assert event["pid"] != os.getpid()
        assert (event["pid"] == launch_pid and event["parent_pid"] == os.getpid()) or (
            event["pid"] != launch_pid and event["parent_pid"] == launch_pid
        )
        assert event["report_sha256"] == job.report_sha256
        assert event["source_sha256"] == job.source_sha256
        assert event["all_project_bindings_staged"] and event["loaded_project_bindings"] > 0
        assert event["model_module_calls_constructors_and_loaders_denied"]
        assert event["candidate_execution_denied"] and not event["test_package_modules_imported"]
    return events


def invoke_worker(
    job: Job, tmp_path: Path, phase: str
) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, Any]]]:
    command, _, _ = worker_job(job, tmp_path, phase)
    stage_root = Path(__file__).resolve().parents[2]
    with subprocess.Popen(
        command, cwd=stage_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        stdout, stderr = process.communicate(timeout=120)
        launch_pid = process.pid
        assert process.returncode is not None
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    record_worker(tmp_path, command, launch_pid, result)
    expected = {"read": 0, "source": 73, "report": 74, "commit": 75}[phase]
    assert result.returncode == expected, result.stdout + result.stderr
    return result, worker_events(job, tmp_path, launch_pid)


def test_separate_cold_process_reads_pair_without_model_or_candidate_execution(
    job: Job, tmp_path: Path
) -> None:
    path = write(job, tmp_path / "published")
    result, events = invoke_worker(job, tmp_path, "read")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(events) == 1 and events[0]["event"] == "read-complete"
    assert Path(events[0]["directory"]) == path
    assert set(events[0]["files"]) == {"source.json", "report.json"}
    for name, content in (("source.json", job.source_content), ("report.json", job.report_content)):
        assert events[0]["files"][name]["sha256"] == sha256(content).hexdigest()
        assert events[0]["files"][name]["bytes"] == len(content)


@pytest.mark.parametrize("phase", ("source", "report", "commit"))
def test_actual_abrupt_child_exit_recovers_without_adopting_or_remeasuring(
    job: Job, tmp_path: Path, phase: str
) -> None:
    result, events = invoke_worker(job, tmp_path, phase)
    assert result.returncode == {"source": 73, "report": 74, "commit": 75}[phase], (
        result.stdout + result.stderr
    )
    assert len(events) == 1 and events[0]["phase"] == phase
    store = tmp_path / "published" / f"scientific-ablation-{job.protocol}-writer-v1"
    target = store / job.report_sha256
    stages = list(store.glob(".iris-writer-*.tmp"))
    if phase == "commit":
        assert target.is_dir() and not stages
        assert events[0]["event"] == "abrupt-after-commit"
        before = tree(target)
        with patch.object(module, "_write", side_effect=AssertionError("No committed retry write")):
            assert write(job, tmp_path / "published") == target
        assert tree(target) == before
    else:
        assert not target.exists() and len(stages) == 1
        assert events[0]["event"] == "abrupt-after-file"
        assert Path(events[0]["directory"]) == stages[0]
        expected = {"source.json"} if phase == "source" else {"source.json", "report.json"}
        assert set(events[0]["files"]) == {p.name for p in stages[0].iterdir()} == expected
        before = tree(stages[0])
        with patch.object(module, "_write", wraps=module._write) as writing:
            assert write(job, tmp_path / "published") == target
        assert writing.call_count == 2  # New own stage, NEVER adoption of the stopped child's.
        assert tree(stages[0]) == before and stages[0].is_dir()
    assert read(job, target).report == job.completed.report


def test_actual_other_process_lock_excludes_writer_and_releases_for_retry(
    job: Job, tmp_path: Path
) -> None:
    command, marker, release = worker_job(job, tmp_path, "lock")
    stage_root = Path(__file__).resolve().parents[2]
    with subprocess.Popen(
        command, cwd=stage_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        try:
            deadline = time.monotonic() + 60
            while not marker.exists() and process.poll() is None:
                assert time.monotonic() < deadline, "Bounded worker lock-readiness deadline"
                time.sleep(0.05)
            assert marker.exists(), "Worker exited before holding its actual OS lock"
            assert process.poll() is None
            store = tmp_path / "published" / f"scientific-ablation-{job.protocol}-writer-v1"
            assert {p.name for p in store.iterdir()} == {".iris-owner.lock"}
            lock_path = store / ".iris-owner.lock"
            before = _signature(_regular(lock_path, directory=False))
            if os.name == "nt":
                with pytest.raises(PermissionError):
                    lock_path.read_bytes()
            with pytest.raises(module.IrisScientificAblationWriterStorageError):
                write(job, tmp_path / "published")
            assert _signature(_regular(lock_path, directory=False)) == before
            assert {p.name for p in store.iterdir()} == {".iris-owner.lock"}
        finally:
            # The child has its OWN90s deadline and exits normally on this handshake.
            # No launcher-only kill, PID guessing, stale-lock reaping or forced deletion.
            release.write_bytes(b"release\n")
            stdout, stderr = process.communicate(timeout=120)
            launch_pid = process.pid
            assert process.returncode is not None
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            record_worker(tmp_path, command, launch_pid, result)
    assert result.returncode == 0, result.stdout + result.stderr
    events = worker_events(job, tmp_path, launch_pid)
    assert [event["event"] for event in events] == ["lock-acquired", "lock-released"]
    assert events[0]["pid"] == events[1]["pid"]
    assert json.loads(marker.read_bytes()) == events[0]
    assert events[0]["files"][".iris-owner.lock"]["content_not_read_while_locked"]
    assert events[1]["files"][".iris-owner.lock"]["sha256"] == sha256(b"1").hexdigest()
    assert lock_path.read_bytes() == b"1"
    assert _signature(_regular(lock_path, directory=False)) == before
    path = write(job, tmp_path / "published")
    assert read(job, path).source == job.completed.source
