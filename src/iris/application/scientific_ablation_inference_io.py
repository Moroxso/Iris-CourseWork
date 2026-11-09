"""Bounded distinct inference store, trusted readback and owned publication.

No graph construction, training, model download, stale-stage reaping or recursive
deletion. Lock/staging checks address cooperating processes and ordinary local
replacement mistakes; they are not a hostile-filesystem or power-loss guarantee.
"""

import os
import re
import shutil
import stat
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

import torch

from iris.application.scientific_ablation_inference import (
    _MAX_MANIFEST,
    IrisScientificAblationInferenceBundleV1,
    IrisScientificAblationInferenceError,
    _checked,
    _checked_manifest,
    _digest,
)
from iris.application.scientific_checkpoint import _json
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_STORE = "scientific-ablation-inference-v1"
_LOCK = ".iris-owner.lock"
_WEIGHTS = "state.safetensors"
_FILES = {"manifest.json", _WEIGHTS}
_STAGE_PREFIX = ".iris-scientific-ablation-inference-"
_MAX_STORE_ENTRIES = 4096
_MAX_STORE_FILES = 8192
_MAX_STORAGE = 16 * 1024**3


def _path(path: Path) -> Path:
    if not isinstance(path, Path) or not path.name or ".." in path.parts:
        raise ValueError("Explicit non-traversing inference path required")
    return path.absolute()


def _regular(path: Path, *, directory: bool) -> os.stat_result:
    value = path.lstat()
    if (
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0) & 0x400
        or not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
        or (not directory and value.st_nlink != 1)
    ):
        raise ValueError("Real non-reparse, non-hardlinked inference path required")
    return value


def _chain(path: Path) -> None:
    for parent in (path, *path.parents):
        _regular(parent, directory=True)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _signature(value: os.stat_result) -> tuple[int, ...]:
    stamp = value.st_ctime_ns
    if os.name == "nt":
        # Observed Python3.12 Windows lstat/fstat ctime semantics differ.
        # Compare exact creation FILETIME via birthtime, NOT unlike ctime fields
        # or rounded/tolerance-based timestamps. POSIX retains exact change time.
        birthtime = getattr(value, "st_birthtime_ns", None)
        if type(birthtime) is not int:
            raise OSError("Exact Windows birthtime metadata is unavailable")
        stamp = birthtime
    return (*_identity(value), value.st_size, value.st_mtime_ns, stamp, value.st_nlink)


def _finish_descriptor(primary: BaseException | None, close: Callable[[], None]) -> None:
    try:
        close()
    except BaseException as cleanup:
        if primary is None:
            raise
        if not isinstance(primary, Exception):
            raise primary from cleanup
        if not isinstance(cleanup, Exception):
            raise cleanup from primary
        raise BaseExceptionGroup(
            "Inference file operation and close failed", [primary, cleanup]
        ) from None
    if primary is not None:
        raise primary


def _read_file(path: Path, maximum: int) -> bytes:
    before = _regular(path, directory=False)
    if not maximum > 0 or before.st_size > maximum:
        raise ValueError("Inference file byte budget exceeded")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    primary: BaseException | None = None
    stream = None
    content = b""
    try:
        stream = os.fdopen(descriptor, "rb")
        opened = os.fstat(stream.fileno())
        if _signature(opened) != _signature(before):
            raise ValueError(
                f"Inference file changed during open: named={_signature(before)!r}, "
                f"opened={_signature(opened)!r}"
            )
        content = stream.read(maximum + 1)
        if (
            len(content) != before.st_size
            or len(content) > maximum
            or _signature(os.fstat(stream.fileno())) != _signature(before)
            or _signature(_regular(path, directory=False)) != _signature(before)
        ):
            raise ValueError("Inference file changed during bounded read")
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, (lambda: os.close(descriptor)) if stream is None else stream.close)
    return content


def _inventory(path: Path) -> dict[str, tuple[int, ...]]:
    entries: dict[str, tuple[int, ...]] = {}
    for entry in path.iterdir():
        if entry.name not in _FILES or entry.name in entries:
            raise ValueError("Inference directory inventory differs")
        entries[entry.name] = _signature(_regular(entry, directory=False))
    return entries


def _read_directory(
    path: Path, *, artifact: str, source: str, lifecycle: str
) -> IrisScientificAblationInferenceBundleV1:
    _chain(path)
    directory = _signature(_regular(path, directory=True))
    inventory = _inventory(path)
    if set(inventory) != _FILES:
        raise ValueError("Inference directory must contain exactly manifest and state")
    content = _read_file(path / "manifest.json", _MAX_MANIFEST)
    if not content.endswith(b"\n"):
        raise ValueError("Inference manifest requires canonical JSON plus one LF")
    _json(content[:-1])
    candidate = IrisScientificAblationInferenceManifestV1.model_validate_json(content[:-1])
    manifest = _checked_manifest(
        candidate,
        expected_sha256=artifact,
        expected_source_checkpoint_sha256=source,
        expected_lifecycle_sha256=lifecycle,
    )  # ALL source/config/policy metadata checked before weights are even read.
    if canonical_json_file_bytes(manifest.model_dump(mode="json")) != content:
        raise ValueError("Canonical inference manifest differs")
    state = _read_file(path / _WEIGHTS, manifest.payload.state_size_bytes)
    bundle = IrisScientificAblationInferenceBundleV1(manifest, state)
    with torch.inference_mode(False), torch.device("cpu"):
        _checked(
            bundle,
            expected_sha256=artifact,
            expected_source_checkpoint_sha256=source,
            expected_lifecycle_sha256=lifecycle,
        )
    _chain(path)
    if _signature(_regular(path, directory=True)) != directory or _inventory(path) != inventory:
        raise ValueError("Inference directory or members changed during readback")
    return bundle


def read_scientific_ablation_inference_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
) -> IrisScientificAblationInferenceBundleV1:
    """Read/authenticate without a training owner or constructing an untrusted graph."""
    try:
        for digest in (
            expected_sha256,
            expected_source_checkpoint_sha256,
            expected_lifecycle_sha256,
        ):
            _digest(digest)
        path = _path(path)
        if path.name != expected_sha256:
            raise ValueError("Inference directory name differs from trusted artifact SHA")
        return _read_directory(
            path,
            artifact=expected_sha256,
            source=expected_source_checkpoint_sha256,
            lifecycle=expected_lifecycle_sha256,
        )
    except Exception as error:
        raise IrisScientificAblationInferenceError(
            "Mode-bound inference readback failed"
        ) from error


@dataclass(frozen=True, slots=True)
class _StoreClaim:
    path: Path
    identity: tuple[int, int]
    lock: ProcessFileLock

    def check(self) -> None:
        self.lock.check()
        _chain(self.path)
        if _identity(_regular(self.path, directory=True)) != self.identity:
            raise ValueError("Inference store was replaced")


@dataclass(slots=True)
class _Stage:
    path: Path
    identity: tuple[int, int]
    files: dict[str, tuple[int, int]] = field(default_factory=dict)

    def check(self, claim: _StoreClaim) -> None:
        claim.check()
        _chain(self.path)
        if (
            self.path.parent != claim.path
            or _identity(_regular(self.path, directory=True)) != self.identity
        ):
            raise ValueError("Owned inference staging directory was replaced")


def _write_bytes(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Inference write made no forward progress")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _write_owned(stage: _Stage, name: str, content: bytes, claim: _StoreClaim) -> None:
    stage.check(claim)
    path = stage.path / name
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    primary: BaseException | None = None
    try:
        # Record actual opened identity BEFORE the first write; failures must not
        # turn an existing/unacknowledged/replaced file into owned cleanup data.
        opened = os.fstat(descriptor)
        named = _regular(path, directory=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(opened) != _identity(named)
        ):
            raise ValueError("New inference file identity differs")
        stage.files[name] = _identity(opened)
        _write_bytes(descriptor, content)
        stage.check(claim)
        if _identity(_regular(path, directory=False)) != stage.files[name]:
            raise ValueError("Staged inference file was replaced while writing")
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, lambda: os.close(descriptor))
    if _read_file(path, len(content)) != content:
        raise ValueError("Staged inference byte readback differs")


def _cleanup(stage: _Stage, claim: _StoreClaim) -> None:
    try:
        stage.path.lstat()
    except FileNotFoundError:
        return  # Successful but unacknowledged rename: never delete the target.
    stage.check(claim)
    inventory = _inventory(stage.path)
    if set(inventory) != set(stage.files):
        raise ValueError("Unknown staging member; refusing ALL cleanup")
    for name, value in inventory.items():
        if value[:2] != stage.files[name]:
            raise ValueError("Replaced staging file; refusing ALL cleanup")
    for name in sorted(stage.files):
        stage.check(claim)
        path = stage.path / name
        if _identity(_regular(path, directory=False)) != stage.files[name]:
            raise ValueError("Staging file changed immediately before cleanup")
        path.unlink()
    stage.check(claim)
    if any(stage.path.iterdir()):
        raise ValueError("Staging gained foreign entries; refusing directory removal")
    stage.path.rmdir()


def _occupied(claim: _StoreClaim) -> int:
    occupied, directories, files = 0, 0, 0
    for entry in claim.path.iterdir():
        claim.check()
        if entry.name == _LOCK:
            continue  # Persistent lock identity/contents checked by claim, never unlinked.
        if (
            re.fullmatch(r"[0-9a-f]{64}", entry.name) is None
            and re.fullmatch(re.escape(_STAGE_PREFIX) + r"[0-9a-f]{32}\.tmp", entry.name) is None
        ):
            raise ValueError("Foreign entry in dedicated inference store")
        _chain(entry)
        directories += 1
        if directories > _MAX_STORE_ENTRIES:
            raise ValueError("Inference store directory count exceeded")
        for child in entry.iterdir():
            if child.name not in _FILES:
                raise ValueError("Unknown store member; no automatic orphan deletion")
            files += 1
            if files > _MAX_STORE_FILES:
                raise ValueError("Inference store file count exceeded")
            occupied += _regular(child, directory=False).st_size
    return occupied


def _publish(
    files: dict[str, bytes],
    target: Path,
    stage: _Stage,
    claim: _StoreClaim,
    *,
    artifact: str,
    source: str,
    lifecycle: str,
) -> None:
    for name, content in files.items():
        _write_owned(stage, name, content, claim)
    stage.check(claim)
    staged = _read_directory(stage.path, artifact=artifact, source=source, lifecycle=lifecycle)
    if staged.state_bytes != files[_WEIGHTS]:
        raise ValueError("Staged verified inference differs")
    stage.check(claim)
    try:
        rename_directory_with_windows_retry(stage.path, target)
    except OSError:
        # A successful but unacknowledged rename or concurrent winner is accepted
        # ONLY by independently authenticated exact bytes, not target existence.
        if not os.path.lexists(target):
            raise
        winner = _read_directory(target, artifact=artifact, source=source, lifecycle=lifecycle)
        if (
            winner.state_bytes != files[_WEIGHTS]
            or canonical_json_file_bytes(winner.manifest.model_dump(mode="json"))
            != files["manifest.json"]
        ):
            raise ValueError("Concurrent/uncertain publication conflicts") from None
    published = _read_directory(target, artifact=artifact, source=source, lifecycle=lifecycle)
    if (
        published.state_bytes != files[_WEIGHTS]
        or canonical_json_file_bytes(published.manifest.model_dump(mode="json"))
        != files["manifest.json"]
    ):
        raise ValueError("Published inference differs")
    claim.check()


def _fail(primary: BaseException, secondary: list[BaseException]) -> NoReturn:
    if not isinstance(primary, Exception):
        if secondary:
            raise primary from BaseExceptionGroup(
                "Secondary inference publication failures", secondary
            )
        raise primary
    for error in secondary:
        if not isinstance(error, Exception):
            raise error from BaseExceptionGroup(
                "Inference publication/restoration failures", [primary, *secondary]
            )
    cause = (
        primary
        if not secondary
        else BaseExceptionGroup("Inference publication/cleanup failures", [primary, *secondary])
    )
    raise IrisScientificAblationInferenceError("Mode-bound inference publication failed") from cause


def write_scientific_ablation_inference_v1(
    bundle: IrisScientificAblationInferenceBundleV1,
    *,
    root: Path,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> Path:
    """Authenticate before ANY mkdir/write; exact retry, bounded explicit store.

    Only the same invocation's created identity-matching stage/files may be cleaned.
    Old stages/artifacts/root user files are never deleted. If ownership is uncertain,
    preserve entries and report failure even when a verified target already exists.
    """
    try:
        if type(bundle) is not IrisScientificAblationInferenceBundleV1:
            raise ValueError("Exact mode-bound inference bundle required")
        snapshot = IrisScientificAblationInferenceBundleV1(bundle.manifest, bundle.state_bytes)
        with torch.inference_mode(False), torch.device("cpu"):
            manifest, _ = _checked(
                snapshot,
                expected_sha256=expected_sha256,
                expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
                expected_lifecycle_sha256=expected_lifecycle_sha256,
            )
        files = {
            "manifest.json": canonical_json_file_bytes(manifest.model_dump(mode="json")),
            _WEIGHTS: snapshot.state_bytes,
        }
        if (
            type(maximum_storage_bytes) is not int
            or not 1 <= maximum_storage_bytes <= _MAX_STORAGE
            or type(minimum_free_bytes) is not int
            or not 0 <= minimum_free_bytes <= _MAX_STORAGE
        ):
            raise ValueError("Exact bounded storage quota/free-space reserve required")
        store = _path(root) / _STORE
        for parent in (store, *store.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _chain(store)
        store_identity = _identity(_regular(store, directory=True))
    except BaseException as error:
        _fail(error, [])
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    target = store / expected_sha256
    try:
        with process_file_lock(store / _LOCK) as lock:
            claim = _StoreClaim(store, store_identity, lock)
            stage: _Stage | None = None
            try:
                claim.check()
                occupied = _occupied(claim)
                if occupied > maximum_storage_bytes:
                    raise ValueError("Inference store already exceeds quota")
                if os.path.lexists(target):
                    existing = _read_directory(
                        target,
                        artifact=expected_sha256,
                        source=expected_source_checkpoint_sha256,
                        lifecycle=expected_lifecycle_sha256,
                    )
                    if (
                        existing.state_bytes != files[_WEIGHTS]
                        or canonical_json_file_bytes(existing.manifest.model_dump(mode="json"))
                        != files["manifest.json"]
                    ):
                        raise ValueError("Existing inference target conflicts")
                else:
                    required = sum(map(len, files.values()))
                    if (
                        occupied + required > maximum_storage_bytes
                        or shutil.disk_usage(store).free < required + minimum_free_bytes
                    ):
                        raise ValueError("Inference quota/free-space reserve exceeded")
                    claim.check()
                    candidate = store / f"{_STAGE_PREFIX}{uuid4().hex}.tmp"
                    os.mkdir(candidate, mode=0o700)
                    # No cleanup ownership before mkdir AND identity capture return.
                    stage = _Stage(candidate, _identity(_regular(candidate, directory=True)))
                    _publish(
                        files,
                        target,
                        stage,
                        claim,
                        artifact=expected_sha256,
                        source=expected_source_checkpoint_sha256,
                        lifecycle=expected_lifecycle_sha256,
                    )
                claim.check()
            except BaseException as error:
                primary = error  # Preserve independently before lock __exit__ can fail.
            finally:
                if stage is not None:
                    try:
                        _cleanup(stage, claim)
                    except BaseException as error:
                        if primary is None:
                            primary = error
                        else:
                            secondary.append(error)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            secondary.append(error)
    if primary is not None:
        _fail(primary, secondary)
    return target


__all__ = ["read_scientific_ablation_inference_v1", "write_scientific_ablation_inference_v1"]
