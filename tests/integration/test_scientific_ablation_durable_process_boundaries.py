"""Trusted spawn targets prove competing publication and abrupt owned-process death.

No generated answers or dynamically generated child source are executed here.
"""

import json
import multiprocessing
import os
import random
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from multiprocessing.connection import _ConnectionBase
from multiprocessing.process import BaseProcess
from pathlib import Path
from time import monotonic
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch

import iris
import iris.utils.process_file_lock as locks
from iris.application import scientific_ablation_lifecycle as lifecycle
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.training_checkpoint import _write_exact_file
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import bytes_sha256, canonical_json_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt
from tests.integration.test_scientific_ablation_durable_process import (
    _assert_disk_boundary,
    _observation,
)
from tests.integration.test_scientific_ablation_run import _stochastic

pytestmark = pytest.mark.integration

type _Boundary = Literal["partial_stage", "complete_stage", "renamed_before_ack"]
type _Role = Literal["holder", "contender", "recovery"]

_MAX_MESSAGE_BYTES = 32 * 1024
_CASE_TIMEOUT_SECONDS = 450.0
_USER_BYTES = b"preserve unrelated durable-boundary user file\n"


@dataclass(frozen=True, slots=True)
class _Child:
    process: BaseProcess
    # CPython/typeshed common base of Windows PipeConnection and POSIX Connection.
    channel: _ConnectionBase


def _iris_root() -> str:
    assert iris.__file__ is not None
    return str(Path(iris.__file__).resolve().parent)


def _remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    assert remaining > 0, "Durable boundary case exceeded its bounded deadline"
    return remaining


def _send(channel: _ConnectionBase, event: str, fields: dict[str, object]) -> None:
    assert not {"event", "pid", "module_root"}.intersection(fields)
    encoded = canonical_json_bytes(
        {"event": event, "pid": os.getpid(), "module_root": _iris_root(), **fields}
    )
    assert len(encoded) <= _MAX_MESSAGE_BYTES
    channel.send_bytes(encoded)


def _decode(encoded: bytes) -> dict[str, object]:
    decoded: object = json.loads(encoded)
    assert isinstance(decoded, dict)
    assert all(isinstance(key, str) for key in decoded)
    return cast(dict[str, object], decoded)


def _string(message: dict[str, object], key: str) -> str:
    value = message[key]
    assert isinstance(value, str)
    return value


def _integer(message: dict[str, object], key: str) -> int:
    value = message[key]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _receive(child: _Child, event: str, deadline: float) -> dict[str, object]:
    while not child.channel.poll(min(_remaining(deadline), 1.0)):
        assert child.process.is_alive(), (
            f"Owned {child.process.name} exited before {event}: {child.process.exitcode}"
        )
    try:
        message = _decode(child.channel.recv_bytes(maxlength=_MAX_MESSAGE_BYTES))
    except (EOFError, OSError) as error:
        pytest.fail(f"Owned {child.process.name} closed its bounded pipe: {error}")
    assert _integer(message, "pid") == child.process.pid
    assert _string(message, "module_root") == _iris_root()
    if message.get("event") == "error":
        pytest.fail(f"Owned {child.process.name} failed:\n{_string(message, 'error')}")
    assert message.get("event") == event, f"Unexpected {child.process.name} event: {message}"
    return message


def _command(channel: _ConnectionBase, deadline: float) -> None:
    while not channel.poll(min(_remaining(deadline), 1.0)):
        pass
    assert _decode(channel.recv_bytes(maxlength=_MAX_MESSAGE_BYTES)) == {"command": "retry"}


def _wait_for_native_kill(channel: _ConnectionBase, deadline: float) -> None:
    # Parent must kill the exact process. No command can release this barrier.
    while not channel.poll(min(_remaining(deadline), 1.0)):
        pass
    raise AssertionError("Holder barrier was released instead of abruptly killed")


def _new_controller(root: Path) -> IrisScientificAblationLifecycleV1:
    return _durable(root, maximum=8, checkpoint=3, validate=1, cycles=3, patience=8)


def _bundle_fingerprint(bundle: io.IrisScientificAblationLifecycleBundleV1) -> str:
    # Hash actual encodings and state bytes rather than trusting claimed identities.
    files = io._files(bundle)
    return canonical_json_bytes(
        {
            "manifest_id": bundle.manifest.sha256,
            "files": {
                name: {"size": len(content), "sha256": bytes_sha256(content)}
                for name, content in sorted(files.items())
            },
        }
    ).decode()


def _directory_snapshot(directory: Path) -> str:
    metadata = directory.lstat()
    entries: dict[str, object] = {}
    for path in sorted(directory.iterdir()):
        assert path.is_file() and not path.is_symlink()
        child = path.lstat()
        entries[path.name] = {
            "device": child.st_dev,
            "inode": child.st_ino,
            "size": child.st_size,
            "sha256": bytes_sha256(path.read_bytes()),
        }
    return canonical_json_bytes(
        {"device": metadata.st_dev, "inode": metadata.st_ino, "files": entries}
    ).decode()


def _store_snapshot(store: Path) -> str:
    # Do not open the locked byte on Windows while another process holds its claim.
    return canonical_json_bytes(
        {
            entry.name: _directory_snapshot(entry)
            for entry in sorted(store.iterdir())
            if entry.name != io._LOCK
        }
    ).decode()


def _store_bytes(store: Path) -> int:
    return sum(
        child.stat().st_size
        for entry in store.iterdir()
        if entry.name != io._LOCK
        for child in entry.iterdir()
    )


def _advance_to_checkpoint(controller: IrisScientificAblationLifecycleV1, step: int) -> None:
    for _ in range(30):
        state = controller.state
        if state.step == step and state.phase == "checkpoint":
            return
        assert state.step <= step and state.phase != "stopped"
        with _stochastic(controller._run):
            result = controller.advance()
        assert type(result) is not io.IrisScientificAblationLifecycleReceiptV1
    raise AssertionError("Actual controller did not reach the requested checkpoint phase")


def _next_update(controller: IrisScientificAblationLifecycleV1, step: int) -> tuple[str, str]:
    best = controller.best_checkpoint
    with _stochastic(controller._run):
        result = controller.advance()
    assert type(result) is IrisScientificAblationOptimizationOutputV1
    assert result.step_index == step and controller.state.step == step
    assert controller.state.phase == "validation" and controller.best_checkpoint == best
    return canonical_json_bytes(asdict(result)).decode(), _observation(controller)


def _reference(
    root: Path,
) -> tuple[
    io.IrisScientificAblationLifecycleBundleV1,
    io.IrisScientificAblationLifecycleBundleV1,
    dict[str, object],
]:
    # This is a real uninterrupted twin, not a fabricated expected result.
    controller = _new_controller(root)
    receipt_a = _receipt(controller)
    bundle_a = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt_a.path)
    _assert_disk_boundary(controller, bundle_a, receipt_a.path)
    observation_a = _observation(controller)
    output_four, following_four = _next_update(controller, 4)
    receipt_b = _receipt(controller)
    bundle_b = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt_b.path)
    assert controller.state.step == 6 and controller.state.phase == "train"
    assert controller.state.checkpoint_count == 2 and controller.state.best_step == 1
    observation_b = _observation(controller)
    output_seven, following_seven = _next_update(controller, 7)
    return (
        bundle_a,
        bundle_b,
        {
            "observation_a": observation_a,
            "output_four": output_four,
            "following_four": following_four,
            "observation_b": observation_b,
            "output_seven": output_seven,
            "following_seven": following_seven,
        },
    )


def _holder(channel: _ConnectionBase, root: Path, boundary: _Boundary, deadline: float) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    reference_a, reference_b, reference = _reference(root.parent / "uninterrupted")
    root.mkdir(parents=True, exist_ok=True)
    (root / "user-owned.txt").write_bytes(_USER_BYTES)
    controller = _new_controller(root)
    receipt_a = _receipt(controller)
    bundle_a = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt_a.path)
    assert bundle_a == reference_a and _observation(controller) == reference["observation_a"]
    _assert_disk_boundary(controller, bundle_a, receipt_a.path)
    acknowledged_snapshot = _directory_snapshot(receipt_a.path)
    _advance_to_checkpoint(controller, 6)
    before = controller.state
    assert before.checkpoint_count == 1 and before.last_checkpoint_step == 3
    assert before.best_step == 1 and before.validation_count == 6
    original_write = _write_exact_file
    original_rename = rename_directory_with_windows_retry
    write_count = 0

    def pause(stage: Path) -> None:
        pending = controller._pending
        assert pending is not None and pending == reference_b
        assert controller._state == before and controller._run._completed_steps == 6
        assert controller._busy and controller._run._busy and not controller.poisoned
        assert pending.manifest.payload.state.phase == "train"
        assert pending.manifest.payload.state.checkpoint_count == 2
        assert pending.manifest.payload.best is not None
        assert pending.manifest.payload.best.payload.completed_steps == 1
        assert pending.best_state_bytes is not None
        assert _directory_snapshot(receipt_a.path) == acknowledged_snapshot
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        _send(
            channel,
            "ready",
            {
                "boundary": boundary,
                "path_a": str(receipt_a.path),
                "path_b": str(receipt_a.path.parent / pending.manifest.sha256),
                "stage": str(stage),
                "pending_fingerprint": _bundle_fingerprint(pending),
                "acknowledged_snapshot": acknowledged_snapshot,
                "store_snapshot": _store_snapshot(receipt_a.path.parent),
                "pending_state": canonical_json_bytes(before.model_dump(mode="json")).decode(),
                "caller_preserved": True,
                **reference,
            },
        )
        _wait_for_native_kill(channel, deadline)

    def staged_write(path: Path, content: bytes) -> None:
        nonlocal write_count
        original_write(path, content)
        write_count += 1
        if boundary == "partial_stage" and write_count == 1:
            assert path.name == "manifest.json"
            assert path.read_bytes() == content
            pause(path.parent)

    def staged_rename(source: Path, target: Path) -> None:
        assert target.name == reference_b.manifest.sha256
        assert set(path.name for path in source.iterdir()) == set(io._files(reference_b))
        if boundary == "complete_stage":
            pause(source)
        original_rename(source, target)
        if boundary == "renamed_before_ack":
            assert not source.exists()
            assert io.read_scientific_ablation_lifecycle_checkpoint_v1(target) == reference_b
            pause(source)

    with (
        patch.object(io, "_write_exact_file", side_effect=staged_write),
        patch.object(io, "rename_directory_with_windows_retry", side_effect=staged_rename),
    ):
        controller.advance()
    raise AssertionError("Required abrupt-death boundary was not reached")


@contextmanager
def _forbid_phase_work(controller: IrisScientificAblationLifecycleV1) -> Iterator[None]:
    with (
        patch.object(
            controller._run, "_step_authorized", side_effect=AssertionError("Retry trained")
        ) as train,
        patch.object(
            lifecycle,
            "_evaluate_scientific_ablation_validation_v1",
            side_effect=AssertionError("Retry evaluated"),
        ) as evaluate,
        patch.object(
            lifecycle,
            "build_scientific_ablation_checkpoint_v1",
            side_effect=AssertionError("Retry rebuilt current"),
        ) as current,
        patch.object(
            lifecycle,
            "build_scientific_ablation_lifecycle_checkpoint_v1",
            side_effect=AssertionError("Retry rebuilt pending"),
        ) as pending,
    ):
        yield
    train.assert_not_called()
    evaluate.assert_not_called()
    current.assert_not_called()
    pending.assert_not_called()


def _contender(
    channel: _ConnectionBase, root: Path, path_a: Path, path_b: Path, deadline: float
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    controller = _new_controller(root)
    bundle_a = io.read_scientific_ablation_lifecycle_checkpoint_v1(path_a)
    io.restore_scientific_ablation_lifecycle_v1(bundle_a, controller=controller)
    _assert_disk_boundary(controller, bundle_a, path_a)
    _advance_to_checkpoint(controller, 6)
    before_state = controller.state
    before_current = build_scientific_ablation_checkpoint_v1(controller._run)
    before_best = controller.best_checkpoint
    store = path_a.parent
    original_lock = locks._lock
    original_publish = io.write_scientific_ablation_lifecycle_checkpoint_v1
    original_write = _write_exact_file
    original_rename = rename_directory_with_windows_retry
    original_occupied = io._occupied
    acquisitions = 0
    refusals = 0
    writes: list[Path] = []
    renames: list[tuple[Path, Path]] = []
    occupied: list[int] = []
    publisher_arguments: list[io.IrisScientificAblationLifecycleBundleV1] = []

    def actual_publish(bundle: io.IrisScientificAblationLifecycleBundleV1, *, root: Path) -> Path:
        publisher_arguments.append(bundle)
        return original_publish(bundle, root=root)

    def actual_lock(descriptor: int, *, acquire: bool) -> None:
        nonlocal acquisitions, refusals
        if acquire:
            acquisitions += 1
        try:
            original_lock(descriptor, acquire=acquire)
        except OSError:
            if acquire:
                refusals += 1
            raise

    def actual_write(path: Path, content: bytes) -> None:
        original_write(path, content)
        writes.append(path)

    def actual_rename(source: Path, target: Path) -> None:
        original_rename(source, target)
        renames.append((source, target))

    def actual_occupied(directory: Path, claim: locks.ProcessFileLock) -> int:
        measured = original_occupied(directory, claim)
        assert measured == _store_bytes(directory)
        occupied.append(measured)
        return measured

    def unchanged() -> None:
        assert controller.state == before_state
        assert controller.best_checkpoint == before_best
        assert build_scientific_ablation_checkpoint_v1(controller._run) == before_current
        assert controller.publication_pending and not controller.poisoned

    with (
        patch.object(
            lifecycle,
            "write_scientific_ablation_lifecycle_checkpoint_v1",
            side_effect=actual_publish,
        ),
        patch.object(locks, "_lock", side_effect=actual_lock),
        patch.object(io, "_write_exact_file", side_effect=actual_write),
        patch.object(io, "rename_directory_with_windows_retry", side_effect=actual_rename),
        patch.object(io, "_occupied", side_effect=actual_occupied),
    ):
        with pytest.raises(lifecycle.IrisScientificAblationLifecycleError):
            controller.advance()
        pending = controller._pending
        assert pending is not None and pending.manifest.sha256 == path_b.name
        assert len(publisher_arguments) == 1 and publisher_arguments[0] is pending
        fingerprint = _bundle_fingerprint(pending)
        unchanged()
        with (
            _forbid_phase_work(controller),
            pytest.raises(lifecycle.IrisScientificAblationLifecycleError),
        ):
            controller.advance()
        unchanged()
        assert controller._pending is pending and _bundle_fingerprint(pending) == fingerprint
        assert len(publisher_arguments) == 2
        assert all(bundle is pending for bundle in publisher_arguments)
        assert acquisitions == refusals == 2
        assert not writes and not renames and not occupied
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        _send(
            channel,
            "blocked",
            {
                "pending_fingerprint": fingerprint,
                "pending_state": canonical_json_bytes(
                    before_state.model_dump(mode="json")
                ).decode(),
                "acquisitions": acquisitions,
                "native_refusals": refusals,
                "caller_preserved": True,
            },
        )
        _command(channel, deadline)
        store_before = _store_snapshot(store)
        occupied_before = _store_bytes(store)
        target_existed = path_b.exists()
        existing_snapshot = _directory_snapshot(path_b) if target_existed else ""
        with _forbid_phase_work(controller):
            receipt = controller.advance()
        assert type(receipt) is io.IrisScientificAblationLifecycleReceiptV1
        assert receipt.path == path_b and receipt.manifest == pending.manifest
        assert len(publisher_arguments) == 3
        assert all(bundle is pending for bundle in publisher_arguments)
        assert controller._pending is None and not controller.publication_pending
        assert controller.state == pending.manifest.payload.state
        assert controller.state.step == 6 and controller.state.checkpoint_count == 2
        assert acquisitions == 3 and refusals == 2
        assert io.read_scientific_ablation_lifecycle_checkpoint_v1(path_b) == pending
        if target_existed:
            assert not writes and not renames and not occupied
            assert _directory_snapshot(path_b) == existing_snapshot
            assert _store_snapshot(store) == store_before
        else:
            assert occupied == [occupied_before]
            assert len(writes) == 3 and len(renames) == 1
            assert {path.name for path in writes} == set(io._files(pending))
            assert len({path.parent for path in writes}) == 1
            assert renames == [(writes[0].parent, path_b)]
        observation_b = _observation(controller)
    output_seven, following_seven = _next_update(controller, 7)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    _send(
        channel,
        "complete",
        {
            "pending_fingerprint": fingerprint,
            "observation_b": observation_b,
            "output_seven": output_seven,
            "following_seven": following_seven,
            "acquisitions": acquisitions,
            "native_refusals": refusals,
            "writes": len(writes),
            "renames": len(renames),
            "publisher_attempts": len(publisher_arguments),
            "same_pending_object": True,
            "caller_preserved": True,
        },
    )


def _recovery(channel: _ConnectionBase, root: Path, path_a: Path, path_b: Path) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    # Only explicit disk paths cross the process boundary, never owner/model/bundle objects.
    restored_a = _new_controller(root / "from-a")
    bundle_a = io.read_scientific_ablation_lifecycle_checkpoint_v1(path_a)
    io.restore_scientific_ablation_lifecycle_v1(bundle_a, controller=restored_a)
    _assert_disk_boundary(restored_a, bundle_a, path_a)
    observation_a = _observation(restored_a)
    output_four, following_four = _next_update(restored_a, 4)
    restored_b = _new_controller(root / "from-b")
    bundle_b = io.read_scientific_ablation_lifecycle_checkpoint_v1(path_b)
    io.restore_scientific_ablation_lifecycle_v1(bundle_b, controller=restored_b)
    assert restored_b.state == bundle_b.manifest.payload.state
    assert restored_b.state.step == 6 and restored_b.state.phase == "train"
    assert restored_b.state.checkpoint_count == 2 and restored_b.state.best_step == 1
    observation_b = _observation(restored_b)
    output_seven, following_seven = _next_update(restored_b, 7)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    _send(
        channel,
        "complete",
        {
            "observation_a": observation_a,
            "output_four": output_four,
            "following_four": following_four,
            "observation_b": observation_b,
            "output_seven": output_seven,
            "following_seven": following_seven,
            "caller_preserved": True,
        },
    )


def _entry(
    channel: _ConnectionBase,
    role: _Role,
    root: str,
    boundary: _Boundary,
    path_a: str,
    path_b: str,
    deadline: float,
) -> None:
    try:
        torch.set_num_threads(1)
        seed = {"holder": 6101, "contender": 7103, "recovery": 8107}[role]
        random.seed(seed)
        random.gauss(0, 1)
        torch.manual_seed(seed + 2)
        if role == "holder":
            _holder(channel, Path(root), boundary, deadline)
        elif role == "contender":
            _contender(channel, Path(root), Path(path_a), Path(path_b), deadline)
        else:
            _recovery(channel, Path(root), Path(path_a), Path(path_b))
    except BaseException:
        with suppress(EOFError, OSError):
            _send(channel, "error", {"error": traceback.format_exc()[-4096:]})
        raise
    finally:
        # Receipt of this marker means Python cleanup ran; it must be absent for the holder.
        with suppress(EOFError, OSError):
            _send(channel, "finally", {})
        channel.close()


def _cleanup_owned(child: _Child) -> None:
    try:
        if child.process.pid is not None:
            if child.process.is_alive():
                with suppress(OSError):
                    child.process.kill()
                child.process.join(timeout=5.0)
            if child.process.is_alive():
                with suppress(OSError):
                    child.process.terminate()
                child.process.join(timeout=5.0)
            assert not child.process.is_alive(), (
                f"Could not stop exact owned {child.process.name} process"
            )
    finally:
        child.channel.close()
        if not child.process.is_alive():
            child.process.close()


def _cleanup_all(owned: list[_Child]) -> None:
    failures: list[BaseException] = []
    for child in reversed(owned):
        try:
            _cleanup_owned(child)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("Exact owned spawn-process cleanup failed", failures)


def _spawn(
    role: _Role,
    root: Path,
    boundary: _Boundary,
    path_a: str,
    path_b: str,
    deadline: float,
) -> _Child:
    # Spawn inherits the caller's import environment, including an installed-package gate.
    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe(duplex=True)
    process = context.Process(
        target=_entry,
        args=(child_channel, role, str(root), boundary, path_a, path_b, deadline),
        name=f"iris-durable-boundary-{role}",
    )
    child = _Child(process, parent_channel)
    try:
        process.start()
    except BaseException:
        _cleanup_owned(child)
        raise
    finally:
        child_channel.close()
    return child


def _join(child: _Child, deadline: float, *, killed: bool) -> None:
    while child.process.is_alive():
        child.process.join(timeout=min(_remaining(deadline), 1.0))
    if killed:
        assert child.process.exitcode is not None and child.process.exitcode != 0
        # EOF, not a normal-finally marker, proves no Python unwinding acknowledged B.
        try:
            readable = child.channel.poll(min(_remaining(deadline), 1.0))
        except BrokenPipeError as error:
            # Windows PeekNamedPipe exposes peer closure here, before recv maps it to EOF.
            assert os.name == "nt" and error.winerror == 109
        else:
            assert readable
        with pytest.raises(EOFError):
            child.channel.recv_bytes(maxlength=_MAX_MESSAGE_BYTES)
    else:
        assert child.process.exitcode == 0
        message = _receive(child, "finally", deadline)
        assert set(message) == {"event", "pid", "module_root"}


@pytest.mark.parametrize("boundary", ("partial_stage", "complete_stage", "renamed_before_ack"))
def test_actual_competing_pending_publisher_recovers_after_native_holder_kill(
    tmp_path: Path,
    boundary: _Boundary,
    record_property: Callable[[str, object], None],
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    deadline = monotonic() + _CASE_TIMEOUT_SECONDS
    owned: list[_Child] = []
    try:
        holder = _spawn("holder", tmp_path / "owner", boundary, "", "", deadline)
        owned.append(holder)
        ready = _receive(holder, "ready", deadline)
        assert ready["boundary"] == boundary and ready["caller_preserved"] is True
        path_a = Path(_string(ready, "path_a"))
        path_b = Path(_string(ready, "path_b"))
        stage = Path(_string(ready, "stage"))
        store = tmp_path / "owner" / io._STORE
        assert path_a.parent == path_b.parent == stage.parent == store
        assert stage.name.startswith(".iris-ablation-lifecycle-") and stage.suffix == ".tmp"
        acknowledged_snapshot = _string(ready, "acknowledged_snapshot")
        assert _directory_snapshot(path_a) == acknowledged_snapshot
        assert _store_snapshot(store) == ready["store_snapshot"]
        lock_path = store / io._LOCK
        lock_before = lock_path.lstat()
        lock_identity = lock_before.st_dev, lock_before.st_ino
        assert lock_before.st_size == 1
        orphan_snapshot = ""
        published_snapshot = ""
        if boundary == "renamed_before_ack":
            assert not stage.exists() and path_b.is_dir()
            published_snapshot = _directory_snapshot(path_b)
            assert (
                _bundle_fingerprint(io.read_scientific_ablation_lifecycle_checkpoint_v1(path_b))
                == ready["pending_fingerprint"]
            )
        else:
            assert stage.is_dir() and not path_b.exists()
            expected = {"manifest.json"}
            if boundary == "complete_stage":
                expected |= {"current.safetensors", "best.safetensors"}
            assert {path.name for path in stage.iterdir()} == expected
            orphan_snapshot = _directory_snapshot(stage)
            with pytest.raises(io.IrisScientificAblationLifecycleCheckpointError):
                io.read_scientific_ablation_lifecycle_checkpoint_v1(stage)
        contender = _spawn(
            "contender", tmp_path / "owner", boundary, str(path_a), str(path_b), deadline
        )
        owned.append(contender)
        blocked = _receive(contender, "blocked", deadline)
        assert blocked["pending_fingerprint"] == ready["pending_fingerprint"]
        assert blocked["pending_state"] == ready["pending_state"]
        assert blocked["acquisitions"] == blocked["native_refusals"] == 2
        assert blocked["caller_preserved"] is True
        assert holder.process.is_alive() and contender.process.is_alive()
        assert _store_snapshot(store) == ready["store_snapshot"]

        # Identity was checked against the real child PID before killing this exact handle.
        holder.process.kill()
        _join(holder, deadline, killed=True)
        assert _directory_snapshot(path_a) == acknowledged_snapshot
        assert _store_snapshot(store) == ready["store_snapshot"]
        lock_after = lock_path.lstat()
        assert (lock_after.st_dev, lock_after.st_ino) == lock_identity
        assert lock_path.read_bytes() == b"1"
        # Actual postdeath claim succeeds without unlinking or PID-based stale reaping.
        with locks.process_file_lock(lock_path) as claim:
            claim.check()
        assert (lock_path.lstat().st_dev, lock_path.lstat().st_ino) == lock_identity
        contender.channel.send_bytes(canonical_json_bytes({"command": "retry"}))
        complete = _receive(contender, "complete", deadline)
        _join(contender, deadline, killed=False)
        assert complete["pending_fingerprint"] == ready["pending_fingerprint"]
        assert complete["acquisitions"] == 3 and complete["native_refusals"] == 2
        assert complete["publisher_attempts"] == 3 and complete["same_pending_object"] is True
        assert complete["caller_preserved"] is True
        for key in ("observation_b", "output_seven", "following_seven"):
            assert complete[key] == ready[key]
        assert _directory_snapshot(path_a) == acknowledged_snapshot
        assert (
            _bundle_fingerprint(io.read_scientific_ablation_lifecycle_checkpoint_v1(path_b))
            == ready["pending_fingerprint"]
        )
        if boundary == "renamed_before_ack":
            assert complete["writes"] == complete["renames"] == 0
            assert _directory_snapshot(path_b) == published_snapshot
            assert {path.name for path in store.iterdir()} == {
                io._LOCK,
                path_a.name,
                path_b.name,
            }
        else:
            assert complete["writes"] == 3 and complete["renames"] == 1
            assert _directory_snapshot(stage) == orphan_snapshot
            assert {path.name for path in store.iterdir()} == {
                io._LOCK,
                path_a.name,
                path_b.name,
                stage.name,
            }
            with pytest.raises(io.IrisScientificAblationLifecycleCheckpointError):
                io.read_scientific_ablation_lifecycle_checkpoint_v1(stage)
        store_after_retry = _store_snapshot(store)
        recovery = _spawn(
            "recovery", tmp_path / "recovery", boundary, str(path_a), str(path_b), deadline
        )
        owned.append(recovery)
        recovered = _receive(recovery, "complete", deadline)
        _join(recovery, deadline, killed=False)
        assert recovered["caller_preserved"] is True
        for key in (
            "observation_a",
            "output_four",
            "following_four",
            "observation_b",
            "output_seven",
            "following_seven",
        ):
            assert recovered[key] == ready[key]
        assert _store_snapshot(store) == store_after_retry
        assert _directory_snapshot(path_a) == acknowledged_snapshot
        assert (tmp_path / "owner" / "user-owned.txt").read_bytes() == _USER_BYTES
        final_lock = lock_path.lstat()
        assert (final_lock.st_dev, final_lock.st_ino) == lock_identity
        assert lock_path.read_bytes() == b"1"
        assert (
            len({os.getpid(), holder.process.pid, contender.process.pid, recovery.process.pid}) == 4
        )
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        record_property("durable_boundary", boundary)
        record_property("durable_boundary_parent_pid", os.getpid())
        record_property("durable_boundary_parent_iris_root", _iris_root())
        for role, message in (("holder", ready), ("contender", complete), ("recovery", recovered)):
            record_property(f"durable_boundary_{role}_pid", _integer(message, "pid"))
            record_property(f"durable_boundary_{role}_iris_root", _string(message, "module_root"))
        record_property("durable_boundary_holder_exitcode", holder.process.exitcode)
        record_property("durable_boundary_holder_termination", "parent-owned-Process.kill")
        record_property("durable_boundary_publisher_attempts", complete["publisher_attempts"])
        record_property("durable_boundary_same_pending_object", complete["same_pending_object"])
        record_property("durable_boundary_pending_fingerprint", ready["pending_fingerprint"])
        record_property("durable_boundary_lock_device", lock_identity[0])
        record_property("durable_boundary_lock_inode", lock_identity[1])
    finally:
        _cleanup_all(owned)
