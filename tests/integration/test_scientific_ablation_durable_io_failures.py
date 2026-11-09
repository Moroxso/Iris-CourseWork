"""Actual durable ablation IO preserves primary signals and unowned disk entries."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pytest

import iris.utils.process_file_lock as locks
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.training_checkpoint import _write_exact_file
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt

pytestmark = pytest.mark.integration

type SignalKind = Literal["keyboard_interrupt", "system_exit", "base_exception_group"]
type FailureKind = SignalKind | Literal["oserror"]

_SIGNALS: tuple[SignalKind, ...] = (
    "keyboard_interrupt",
    "system_exit",
    "base_exception_group",
)
_FAILURES: tuple[FailureKind, ...] = (*_SIGNALS, "oserror")
_STORAGE = 8 * 1024**2
_USER_BYTES = b"preserve external user file\n"
_FOREIGN_BYTES = b"preserve foreign staging content\n"


def _failure(kind: FailureKind, message: str) -> BaseException:
    if kind == "keyboard_interrupt":
        return KeyboardInterrupt(message)
    if kind == "system_exit":
        return SystemExit(83)
    if kind == "base_exception_group":
        return BaseExceptionGroup(message, [ValueError("ordinary child"), SystemExit(84)])
    return OSError(message)


def _destination(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    (root / "user-owned.txt").write_bytes(_USER_BYTES)
    return root


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {entry.name: entry.read_bytes() for entry in directory.iterdir()}


@pytest.fixture(scope="module")
def disk_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> io.IrisScientificAblationLifecycleBundleV1:
    root = tmp_path_factory.mktemp("trusted-ablation-durable-io")
    controller = _durable(root, storage=_STORAGE)
    receipt = _receipt(controller)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    assert bundle.manifest == receipt.manifest and bundle.best_state_bytes is not None
    assert sum(map(len, io._files(bundle).values())) < _STORAGE
    return bundle


@pytest.mark.parametrize("primary_kind", _SIGNALS)
@pytest.mark.parametrize("secondary_kind", _FAILURES)
def test_primary_write_signal_survives_temporary_lexists_failure(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    primary_kind: SignalKind,
    secondary_kind: FailureKind,
) -> None:
    root = _destination(tmp_path)
    primary = _failure(primary_kind, "primary staged write")
    secondary = _failure(secondary_kind, "secondary staging existence check")
    original_write = _write_exact_file
    original_lexists = os.path.lexists
    written: list[tuple[Path, bytes]] = []
    secondary_checks: list[Path] = []

    def interrupted_write(path: Path, content: bytes) -> None:
        original_write(path, content)
        written.append((path, content))
        raise primary

    def interrupted_lexists(path: Path) -> bool:
        if written and path == written[0][0].parent:
            secondary_checks.append(path)
            raise secondary
        return original_lexists(path)

    with (
        patch.object(io, "_write_exact_file", side_effect=interrupted_write),
        patch.object(os.path, "lexists", side_effect=interrupted_lexists),
        pytest.raises(type(primary)) as caught,
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)

    assert caught.value is primary
    assert len(written) == 1
    path, content = written[0]
    assert secondary_checks == [path.parent]
    assert path.read_bytes() == content
    assert {entry.name for entry in path.parent.iterdir()} == {path.name}
    store = root / io._STORE
    assert {entry.name for entry in store.iterdir()} == {io._LOCK, path.parent.name}
    assert not (store / disk_bundle.manifest.sha256).exists()
    assert (root / "user-owned.txt").read_bytes() == _USER_BYTES


@pytest.mark.parametrize("kind", ("quota", "reserve"))
def test_actual_store_quota_or_reserve_refusal_never_stages_or_publishes(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: Literal["quota", "reserve"],
) -> None:
    root = _destination(tmp_path)
    store = root / io._STORE
    existing = store / "user-owned-checkpoint"
    existing.mkdir(parents=True)
    required = sum(map(len, io._files(disk_bundle).values()))
    occupied_bytes = b"x" * (_STORAGE - required + 1) if kind == "quota" else _FOREIGN_BYTES
    existing_file = existing / "user-owned.bin"
    existing_file.write_bytes(occupied_bytes)
    free = required + 1024**3 - (1 if kind == "reserve" else 0)

    with (
        patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=free)),
        patch.object(io, "_write_exact_file") as write,
        patch.object(io, "rename_directory_with_windows_retry") as publish,
        pytest.raises(io.IrisScientificAblationLifecycleCheckpointError),
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)

    write.assert_not_called()
    publish.assert_not_called()
    assert {entry.name for entry in store.iterdir()} == {io._LOCK, existing.name}
    assert {entry.name for entry in existing.iterdir()} == {existing_file.name}
    assert existing_file.read_bytes() == occupied_bytes
    assert (store / io._LOCK).read_bytes() == b"1"
    assert (root / "user-owned.txt").read_bytes() == _USER_BYTES


@pytest.mark.parametrize("kind", ("foreign_entry", "replaced_directory"))
@pytest.mark.parametrize("failure_kind", _FAILURES)
def test_failed_write_preserves_foreign_or_replaced_staging(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: Literal["foreign_entry", "replaced_directory"],
    failure_kind: FailureKind,
) -> None:
    root = _destination(tmp_path)
    primary = _failure(failure_kind, "primary staged write")
    original_write = _write_exact_file
    written: list[tuple[Path, bytes]] = []
    parked = root / "detached-owned-stage"

    def replaced_or_foreign_write(path: Path, content: bytes) -> None:
        original_write(path, content)
        written.append((path, content))
        if kind == "foreign_entry":
            (path.parent / "foreign.txt").write_bytes(_FOREIGN_BYTES)
        else:
            # Keep the original inode alive so a new directory cannot reuse its identity.
            path.parent.rename(parked)
            path.parent.mkdir()
            (path.parent / "manifest.json").write_bytes(_FOREIGN_BYTES)
        raise primary

    expected = (
        io.IrisScientificAblationLifecycleCheckpointError
        if isinstance(primary, Exception)
        else type(primary)
    )
    with (
        patch.object(io, "_write_exact_file", side_effect=replaced_or_foreign_write),
        patch.object(io, "rename_directory_with_windows_retry") as publish,
        pytest.raises(expected) as caught,
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)

    if failure_kind != "oserror":
        assert caught.value is primary
    publish.assert_not_called()
    assert len(written) == 1
    path, content = written[0]
    if kind == "foreign_entry":
        assert _snapshot(path.parent) == {path.name: content, "foreign.txt": _FOREIGN_BYTES}
    else:
        assert _snapshot(path.parent) == {"manifest.json": _FOREIGN_BYTES}
        assert _snapshot(parked) == {path.name: content}
        original, replacement = parked.lstat(), path.parent.lstat()
        assert (original.st_dev, original.st_ino) != (replacement.st_dev, replacement.st_ino)
    store = root / io._STORE
    assert {entry.name for entry in store.iterdir()} == {io._LOCK, path.parent.name}
    assert not (store / disk_bundle.manifest.sha256).exists()
    assert (root / "user-owned.txt").read_bytes() == _USER_BYTES


def test_conflicting_existing_target_is_preserved_without_staging(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1, tmp_path: Path
) -> None:
    root = _destination(tmp_path)
    target = io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)
    assert io.read_scientific_ablation_lifecycle_checkpoint_v1(target) == disk_bundle
    (target / "manifest.json").write_bytes(_FOREIGN_BYTES)
    before = _snapshot(target)

    with (
        patch.object(io, "_write_exact_file") as write,
        patch.object(io, "rename_directory_with_windows_retry") as publish,
        pytest.raises(io.IrisScientificAblationLifecycleCheckpointError),
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)

    write.assert_not_called()
    publish.assert_not_called()
    assert _snapshot(target) == before
    assert {entry.name for entry in target.parent.iterdir()} == {io._LOCK, target.name}
    assert (root / "user-owned.txt").read_bytes() == _USER_BYTES


@pytest.mark.parametrize("primary_kind", _SIGNALS)
@pytest.mark.parametrize("secondary_kind", _FAILURES)
def test_primary_write_signal_survives_native_lock_release_failure(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    primary_kind: SignalKind,
    secondary_kind: FailureKind,
) -> None:
    root = _destination(tmp_path)
    primary = _failure(primary_kind, "primary staged write")
    secondary = _failure(secondary_kind, "secondary lock release")
    original_write = _write_exact_file
    original_lock = locks._lock
    written: list[Path] = []
    releases: list[int] = []

    def interrupted_write(path: Path, content: bytes) -> None:
        original_write(path, content)
        written.append(path)
        raise primary

    def failed_release(descriptor: int, *, acquire: bool) -> None:
        original_lock(descriptor, acquire=acquire)
        if not acquire:
            releases.append(descriptor)
            raise secondary

    with (
        patch.object(io, "_write_exact_file", side_effect=interrupted_write),
        patch.object(locks, "_lock", side_effect=failed_release),
        pytest.raises(type(primary)) as caught,
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(disk_bundle, root=root)

    assert caught.value is primary
    assert len(written) == len(releases) == 1
    assert not written[0].parent.exists()
    store = root / io._STORE
    assert {entry.name for entry in store.iterdir()} == {io._LOCK}
    lock_path = store / io._LOCK
    assert lock_path.read_bytes() == b"1"
    with locks.process_file_lock(lock_path) as claim:
        claim.check()
    assert (root / "user-owned.txt").read_bytes() == _USER_BYTES
