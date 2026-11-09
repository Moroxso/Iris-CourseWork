"""Bounded local run storage, fixed pending slots and recoverable interrupted publication."""

import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_dataset_io import (
    _check_parents,
    _is_link_or_reparse,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.artifacts.study_evaluation_run_v1 import (
    IrisStudyEvaluationRunV1,
    IrisStudyEvaluationSummaryV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock

_RUN_DOMAIN = "iris/study-evaluation-run/v1/payload"
_SUMMARY_DOMAIN = "iris/study-evaluation-summary/v1/payload"
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_PENDING_NAMES = (".run-pending.json", ".receipt-pending.json", ".summary-pending.json")


def _directory(value: Path) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or not value.name
        or ".." in value.parts
    ):
        raise ValueError("evaluation directory must be an explicit absolute non-root Path")
    return value


def _regular_size(path: Path, *, empty: bool = False) -> int:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("evaluation storage requires real regular files")
    if metadata.st_size < (0 if empty else 1):
        raise ValueError("unexpected empty evaluation file")
    return metadata.st_size


def _inventory(directory: Path, *, answers: int) -> int:
    """Bound names/counts and account all file lengths, including quarantined pending bytes."""
    _check_parents(directory / "run.json", create=False)
    total = 0
    for index, path in enumerate(directory.iterdir()):
        if index > 8:
            raise ValueError("unexpected evaluation directory entries")
        if path.name in (".lock", "run.json", "summary.json", *_PENDING_NAMES):
            size = _regular_size(path, empty=path.name in _PENDING_NAMES)
            if size > (1 if path.name == ".lock" else _MAX_METADATA_BYTES):
                raise ValueError("evaluation root file exceeds its safety limit")
            total += size
        elif path.name in ("receipts", "interrupted"):
            _check_parents(path / "entry.json", create=False)
            for number, item in enumerate(path.iterdir()):
                if number >= (answers if path.name == "receipts" else 100000):
                    raise ValueError("evaluation directory exceeds its entry limit")
                pattern = r"[0-9]{6}\.json" if path.name == "receipts" else r"[0-9a-f]{64}\.bin"
                if re.fullmatch(pattern, item.name) is None:
                    raise ValueError("unexpected evaluation file")
                size = _regular_size(item, empty=path.name == "interrupted")
                maximum = 12 * 1024 * 1024 if path.name == "receipts" else _MAX_METADATA_BYTES
                if size > maximum:
                    raise ValueError("evaluation data file exceeds its safety limit")
                total += size
        else:
            raise ValueError("unrecognized evaluation directory entry")
    return total


def _bounded_bytes(path: Path, *, maximum: int, empty: bool = False) -> bytes:
    if empty and _regular_size(path, empty=True) == 0:
        # The owner holds the OS lock; only this known uncommitted slot may be empty.
        _check_parents(path, create=False)
        return b""
    return _read_regular_bytes(path, max_bytes=maximum)


def _canonical(path: Path, *, maximum: int = _MAX_METADATA_BYTES) -> tuple[dict[str, Any], bytes]:
    content = _read_regular_bytes(path, max_bytes=maximum)
    raw = json.loads(
        content.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if type(raw) is not dict or canonical_json_file_bytes(raw) != content:
        raise ValueError("evaluation metadata is not canonical JSON")
    return raw, content


def _validated_run(run: IrisStudyEvaluationRunV1) -> IrisStudyEvaluationRunV1:
    if type(run) is not IrisStudyEvaluationRunV1:
        raise TypeError("run must be exactly IrisStudyEvaluationRunV1")
    owned = IrisStudyEvaluationRunV1.model_validate(run.model_dump(mode="python"))
    if owned.payload_sha256 != domain_sha256(
        _RUN_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("run payload hash disagrees")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > _MAX_METADATA_BYTES:
        raise ValueError("run metadata exceeds safety limit")
    return owned


def _validated_summary(summary: IrisStudyEvaluationSummaryV1) -> IrisStudyEvaluationSummaryV1:
    if type(summary) is not IrisStudyEvaluationSummaryV1:
        raise TypeError("summary must be exactly IrisStudyEvaluationSummaryV1")
    owned = IrisStudyEvaluationSummaryV1.model_validate(summary.model_dump(mode="python"))
    if owned.payload_sha256 != domain_sha256(
        _SUMMARY_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("summary payload hash disagrees")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > _MAX_METADATA_BYTES:
        raise ValueError("summary exceeds safety limit")
    return owned


def read_study_evaluation_run_v1(directory: Path) -> IrisStudyEvaluationRunV1:
    """Read canonical plan metadata only, not generation/dependency/score verification."""
    try:
        raw, content = _canonical(_directory(directory) / "run.json")
        owned = _validated_run(IrisStudyEvaluationRunV1.model_validate_json(content))
        if owned.model_dump(mode="json") != raw:
            raise ValueError("run metadata changed during canonical roundtrip")
        return owned
    except Exception:
        raise IrisStudyEvaluationError("Iris evaluation run could not be read") from None


def read_study_evaluation_summary_v1(directory: Path) -> IrisStudyEvaluationSummaryV1:
    """Read canonical summary/count integrity only, not referenced receipt or dependency truth."""
    try:
        raw, content = _canonical(_directory(directory) / "summary.json")
        owned = _validated_summary(IrisStudyEvaluationSummaryV1.model_validate_json(content))
        if owned.model_dump(mode="json") != raw:
            raise ValueError("summary changed during canonical roundtrip")
        return owned
    except Exception:
        raise IrisStudyEvaluationError("Iris evaluation summary could not be read") from None


def _archive_pending(path: Path, *, claim: ProcessFileLock, maximum: int) -> None:
    """Preserve, never discard, an incomplete private pending file in this claimed run."""
    claim.check()
    if path.parent != claim.path.parent or path.name not in _PENDING_NAMES:
        raise ValueError("unowned pending path")
    content = _bounded_bytes(path, maximum=maximum, empty=True)
    destination = path.parent / "interrupted" / f"{sha256(content).hexdigest()}.bin"
    _check_parents(destination, create=True)
    try:
        os.link(path, destination)
    except FileExistsError:
        if _bounded_bytes(destination, maximum=maximum, empty=True) != content:
            raise ValueError("interrupted bytes conflict") from None
    claim.check()
    if _bounded_bytes(destination, maximum=maximum, empty=True) != content:
        raise ValueError("interrupted byte preservation failed")
    path.unlink()


def _publish(
    destination: Path,
    content: bytes,
    *,
    pending: Path,
    claim: ProcessFileLock,
) -> None:
    """Under the persistent lock, fsync a fixed slot and atomically link without overwrite.

    A process death can leave the slot intact. Resume revalidates it before adoption;
    invalid/incomplete bytes are preserved in the run's bounded interrupted area.
    """
    claim.check()
    if pending.parent != claim.path.parent or pending.name not in _PENDING_NAMES:
        raise ValueError("unowned publication slot")
    if destination.parent not in (pending.parent, pending.parent / "receipts"):
        raise ValueError("publication escaped the claimed run")
    _check_parents(destination, create=True)
    try:
        pending.lstat()
    except FileNotFoundError:
        descriptor = os.open(
            pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600
        )
        try:
            remaining = memoryview(content)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("pending publication made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if _read_regular_bytes(pending) != content:
        raise ValueError("pending bytes disagree with intended publication")
    claim.check()
    try:
        os.link(pending, destination)
    except FileExistsError:
        if _read_regular_bytes(destination) != content:
            raise ValueError("existing publication conflicts") from None
    if _read_regular_bytes(destination) != content:
        raise ValueError("installed publication bytes disagree")
    claim.check()
    pending.unlink()


__all__ = ["read_study_evaluation_run_v1", "read_study_evaluation_summary_v1"]
