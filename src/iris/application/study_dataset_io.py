"""Canonical, no-clobber local storage for plaintext study datasets."""

from __future__ import annotations

import json
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

from iris.application.errors import IrisStudyDatasetError
from iris.application.study_dataset import validate_study_dataset_v1
from iris.artifacts.study_dataset_v1 import IrisStudyDatasetV1
from iris.utils.hashing import canonical_json_file_bytes

_MAX_DATASET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IrisStudyDatasetWriteResultV1:
    """Identity of one explicit local plaintext study dataset file."""

    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def _explicit_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("study dataset requires an explicit .json str or Path")
    result = Path(path)
    if result.suffix != ".json" or not result.name or ".." in result.parts:
        raise ValueError("study dataset requires a .json path without parent traversal")
    return result


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _check_parents(path: Path, *, create: bool) -> None:
    parent = path.absolute().parent
    for directory in (*reversed(parent.parents), parent):
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            if not create:
                raise
            with suppress(FileExistsError):
                directory.mkdir()
            metadata = directory.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("study dataset parents must be real directories")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    # Windows lstat and fstat can report different ctime for the same hard link.
    # Compare stable inode/device/size/mtime across these two metadata sources.
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_regular_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    size_limit = _MAX_DATASET_BYTES if max_bytes is None else max_bytes
    if type(size_limit) is not int or not 0 < size_limit <= _MAX_DATASET_BYTES:
        raise ValueError("invalid study file safety limit")
    _check_parents(path, create=False)
    before = path.lstat()
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("study dataset path must be a real regular file")
    if before.st_size <= 0 or before.st_size > size_limit:
        raise ValueError("study file size is outside its safety limit")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        opened = os.fstat(stream.fileno())
        if _is_link_or_reparse(opened) or _file_identity(opened) != _file_identity(before):
            raise ValueError("study dataset file changed before reading")
        content = stream.read(size_limit + 1)
        after_read = os.fstat(stream.fileno())
    _check_parents(path, create=False)
    after = path.lstat()
    if (
        _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or _file_identity(before) != _file_identity(after_read)
        or _file_identity(before) != _file_identity(after)
        # ctime is comparable within each metadata source, not across lstat/fstat on Windows.
        or opened.st_ctime_ns != after_read.st_ctime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(content) != before.st_size
    ):
        raise ValueError("study dataset file changed while reading")
    return content


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate study dataset JSON key is forbidden")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> NoReturn:
    raise ValueError("nonfinite study dataset JSON number is forbidden")


def read_study_dataset_v1(path: str | Path) -> IrisStudyDatasetV1:
    """Read canonical UTF-8 JSON and validate both dataset and embedded manifest."""
    try:
        content = _read_regular_bytes(_explicit_path(path))
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            raise ValueError("study dataset requires canonical UTF-8 JSON with one final LF")
        decoded = content[:-1].decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        if canonical_json_file_bytes(raw) != content:
            raise ValueError("study dataset is not canonical iris-json-v1")
        dataset = IrisStudyDatasetV1.model_validate_json(decoded)
        owned = validate_study_dataset_v1(dataset)
        if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
            raise ValueError("study dataset canonical round trip changed its bytes")
        return owned
    except Exception:
        # Pydantic/JSON errors may contain plaintext examples; keep them out of tracebacks.
        raise IrisStudyDatasetError("Iris study dataset could not be read or verified") from None


def _existing_matches(path: Path, expected: bytes) -> bool:
    try:
        return _read_regular_bytes(path) == expected
    except (OSError, ValueError):
        return False


def _install_no_clobber(path: Path, content: bytes) -> bool:
    _check_parents(path, create=True)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not _existing_matches(path, content):
            raise IrisStudyDatasetError("Existing study dataset conflicts with expected bytes")
        return True
    temporary_path = path.parent / f".iris-study-{uuid4().hex}.tmp"
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        temporary_created = True
        try:
            stream = os.fdopen(descriptor, "wb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            remaining = memoryview(content)
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0:
                    raise OSError("Study dataset write made no forward progress")
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())
        if _read_regular_bytes(temporary_path) != content:
            raise OSError("Study dataset temporary bytes failed verification")
        _check_parents(path, create=False)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError:
            if not _existing_matches(path, content):
                raise IrisStudyDatasetError(
                    "Concurrent study dataset conflicts with expected bytes"
                ) from None
            reused = True
        if not _existing_matches(path, content):
            raise IrisStudyDatasetError("Installed study dataset bytes failed verification")
        temporary_path.unlink()
        return reused
    except BaseException as primary_error:
        try:
            if temporary_created:
                temporary_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "Study dataset persistence and cleanup both failed",
                [primary_error, cleanup_error],
            )
            raise IrisStudyDatasetError(
                "Iris study dataset failed and cleanup was incomplete"
            ) from failures
        raise


def write_study_dataset_v1(
    dataset: IrisStudyDatasetV1,
    *,
    path: str | Path,
) -> IrisStudyDatasetWriteResultV1:
    """Validate before filesystem changes and install atomically without overwrite."""
    try:
        explicit_path = _explicit_path(path)
        owned = validate_study_dataset_v1(dataset)
        content = canonical_json_file_bytes(owned.model_dump(mode="json"))
        if len(content) > _MAX_DATASET_BYTES:
            raise ValueError("study dataset exceeds the 64 MiB safety limit")
        reused = _install_no_clobber(explicit_path, content)
        return IrisStudyDatasetWriteResultV1(
            path=explicit_path,
            payload_sha256=owned.payload_sha256,
            size_bytes=len(content),
            reused_existing=reused,
        )
    except IrisStudyDatasetError:
        raise
    except Exception:
        raise IrisStudyDatasetError("Iris study dataset could not be persisted") from None


__all__ = [
    "IrisStudyDatasetWriteResultV1",
    "read_study_dataset_v1",
    "write_study_dataset_v1",
]
