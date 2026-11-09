"""Private call-owned checkpoint projections, revalidated by full file bytes on every use."""

import asyncio
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import get_ident

from iris.application.arm_request_checkpoint import IrisArmRequestCheckpointProjectionV1
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import (
    ApplicationError,
    IrisCodeEvaluationUnavailableError,
    IrisStudyEvaluationError,
    IrisStudyGenerationError,
)
from iris.application.study_dataset_io import _check_parents, _file_identity, _is_link_or_reparse

_CHUNK_BYTES = 1024 * 1024
_FILES = ("manifest.json", "state.safetensors")
type _Owner = tuple[int, int, int | None]


def _owner() -> _Owner:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), None if task is None else id(task)


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    identity: tuple[int, int, int, int]
    ctime_ns: int
    sha256: str


def _regular(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise ValueError("checkpoint dependency must be a nonempty real regular file")
    return metadata


def _fingerprint(path: Path) -> _FileFingerprint:
    """Hash the observed full file with bounded memory; never trust size/mtime as content."""
    _check_parents(path, create=False)
    before = _regular(path)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        opened = os.fstat(stream.fileno())
        if _is_link_or_reparse(opened) or _file_identity(opened) != _file_identity(before):
            raise ValueError("checkpoint dependency changed before hashing")
        digest = sha256()
        remaining = before.st_size
        while remaining:
            chunk = stream.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("checkpoint dependency was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("checkpoint dependency grew while hashing")
        after_read = os.fstat(stream.fileno())
    _check_parents(path, create=False)
    after = _regular(path)
    if (
        _file_identity(before) != _file_identity(after_read)
        or _file_identity(before) != _file_identity(after)
        # Windows ctime is compared within, not across, lstat/fstat sources.
        or opened.st_ctime_ns != after_read.st_ctime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError("checkpoint dependency changed while hashing")
    return _FileFingerprint(_file_identity(before), before.st_ctime_ns, digest.hexdigest())


@dataclass(frozen=True, slots=True)
class _CheckpointFingerprint:
    directories: tuple[tuple[int, int], ...]
    manifest: _FileFingerprint
    state: _FileFingerprint


def _directory_chain(directory: Path) -> tuple[tuple[int, int], ...]:
    _check_parents(directory / _FILES[0], create=False)
    identities = []
    for path in (*reversed(directory.parents), directory):
        metadata = path.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("checkpoint parents must remain real directories")
        # Unrelated sibling creation can legitimately change directory timestamps.
        identities.append((metadata.st_dev, metadata.st_ino))
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > len(_FILES):
                raise ValueError("checkpoint contains extra directory entries")
    if tuple(sorted(names)) != _FILES:
        raise ValueError("checkpoint must retain its exact two-file layout")
    return tuple(identities)


def _checkpoint_fingerprint(directory: Path) -> _CheckpointFingerprint:
    directories = _directory_chain(directory)
    manifest, state = (_fingerprint(directory / name) for name in _FILES)
    if _directory_chain(directory) != directories:
        raise ValueError("checkpoint directory identity changed")
    for name, fingerprint in zip(_FILES, (manifest, state), strict=True):
        final = _regular(directory / name)
        if (
            _file_identity(final) != fingerprint.identity
            or final.st_ctime_ns != fingerprint.ctime_ns
        ):
            raise ValueError("checkpoint file changed during dependency collection")
    return _CheckpointFingerprint(directories, manifest, state)


@dataclass(frozen=True, slots=True)
class _VerifiedProjection:
    directory: Path
    fingerprint: _CheckpointFingerprint
    projection: IrisArmRequestCheckpointProjectionV1


class _CheckpointScope:
    def __init__(self) -> None:
        self.owner = _owner()
        self.closed = False
        self.invalid = False
        self.entries: dict[tuple[Path, IrisArmTeacherCheckpointIdentity], _VerifiedProjection] = {}

    def matches_owner(self) -> bool:
        return not self.closed and self.owner == _owner()

    def _require_valid(self) -> None:
        if not self.matches_owner() or self.invalid:
            raise ValueError("checkpoint context is closed, foreign or invalidated")

    def projection(
        self,
        identity: IrisArmTeacherCheckpointIdentity,
        root: Path,
        load: Callable[[], IrisArmRequestCheckpointProjectionV1],
    ) -> IrisArmRequestCheckpointProjectionV1:
        self._require_valid()
        try:
            if not root.is_absolute() or ".." in root.parts:
                raise ValueError("checkpoint context requires an absolute artifact root")
            directory = identity.expected_path(root=root)
            key = (root, identity)
            before = _checkpoint_fingerprint(directory)
            if before.state.sha256 != identity.state_file_sha256:
                raise ValueError("checkpoint state bytes disagree with their exact identity")
            if key in self.entries:
                saved = self.entries[key]
                if before != saved.fingerprint:
                    raise ValueError("previously verified checkpoint files changed")
                return deepcopy(saved.projection)
            # Only this trusted in-process loader can establish semantic validity.
            # It is never a public caller-supplied assertion of prior verification.
            projection = load()
            if projection.checkpoint_identity != identity:
                raise ValueError("verified checkpoint projection identity disagrees")
            if _checkpoint_fingerprint(directory) != before:
                raise ValueError("checkpoint changed during full semantic verification")
            self.entries[key] = _VerifiedProjection(directory, before, deepcopy(projection))
            return deepcopy(projection)
        except BaseException:
            self.invalid = True
            self.entries.clear()
            raise

    def check_all(self) -> None:
        self._require_valid()
        try:
            for saved in self.entries.values():
                if _checkpoint_fingerprint(saved.directory) != saved.fingerprint:
                    raise ValueError("checkpoint changed before batch completion")
        except BaseException:
            self.invalid = True
            self.entries.clear()
            raise

    def close(self) -> None:
        self.closed = True
        self.entries.clear()


_ACTIVE: ContextVar[_CheckpointScope | None] = ContextVar(
    "iris_study_checkpoint_scope", default=None
)


def _current_checkpoint_scope() -> _CheckpointScope | None:
    current = _ACTIVE.get()
    return current if current is not None and current.matches_owner() else None


@contextmanager
def _checkpoint_verification_context() -> Iterator[None]:
    existing = _current_checkpoint_scope()
    if existing is not None:
        existing._require_valid()
        try:
            yield
            existing.check_all()
        except BaseException:
            existing.invalid = True
            existing.entries.clear()
            raise
        return
    current = _CheckpointScope()
    token = _ACTIVE.set(current)
    try:
        yield
        current.check_all()
    finally:
        current.close()
        _ACTIVE.reset(token)


def _scoped[**P, R](
    action: Callable[P, R], error_type: type[ApplicationError], message: str
) -> Callable[P, R]:
    @wraps(action)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            with _checkpoint_verification_context():
                return action(*args, **kwargs)
        except (
            IrisCodeEvaluationUnavailableError,
            IrisStudyEvaluationError,
            IrisStudyGenerationError,
        ):
            raise
        except Exception:
            raise error_type(message) from None

    return run


def _scoped_evaluation[**P, R](action: Callable[P, R]) -> Callable[P, R]:
    return _scoped(action, IrisStudyEvaluationError, "Iris study checkpoint verification failed")


def _scoped_generation[**P, R](action: Callable[P, R]) -> Callable[P, R]:
    return _scoped(
        action, IrisStudyGenerationError, "Iris study generation checkpoint verification failed"
    )
