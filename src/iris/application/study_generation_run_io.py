"""Owned generation storage: fixed pending slots and preserved interrupted bytes."""

import json
import os
import re
import stat
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from iris.application.errors import IrisStudyGenerationError
from iris.application.study_dataset_io import (
    _check_parents,
    _is_link_or_reparse,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.artifacts.study_generation_run_v1 import (
    IrisStudyGenerationAnswerV1,
    IrisStudyGenerationRunV1,
)
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.utils.process_file_lock import ProcessFileLock

_RUN_DOMAIN = "iris/study-generation-run/v1/payload"
_ANSWER_DOMAIN = "iris/study-generation-answer/v1/payload"
_MAX_BYTES = 64 * 1024**2
_PENDING_NAMES = (".run-pending.json", ".answer-pending.json", ".generation-pending.json")


def _directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or not path.name or ".." in path.parts:
        raise ValueError("generation directory must be an explicit absolute non-root Path")
    return path


def _names(directory: Path, maximum: int) -> tuple[str, ...]:
    _check_parents(directory / "entry.json", create=False)
    result: list[str] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if len(result) >= maximum:
                raise ValueError("generation directory exceeds its entry bound")
            result.append(entry.name)
    return tuple(result)


def _size(path: Path, *, maximum: int, empty: bool = False) -> int:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("generation storage requires real regular files")
    if not (0 if empty else 1) <= metadata.st_size <= maximum:
        raise ValueError("generation file violates its byte bound")
    return metadata.st_size


def _pending_bytes(path: Path, maximum: int) -> bytes:
    _check_parents(path, create=False)
    if _size(path, maximum=maximum, empty=True) == 0:
        return b""
    return _read_regular_bytes(path, max_bytes=maximum)


def _inventory(directory: Path, run: IrisStudyGenerationRunV1) -> int:
    total = 0
    for name in _names(directory, 8):
        path = directory / name
        if name in (".lock", "run.json", "generation.json", *_PENDING_NAMES):
            maximum = (
                1
                if name == ".lock"
                else run.payload.max_record_bytes
                if name == ".answer-pending.json"
                else run.payload.max_artifact_bytes
                if name in ("generation.json", ".generation-pending.json")
                else _MAX_BYTES
            )
            total += _size(path, maximum=maximum, empty=name in _PENDING_NAMES)
            # Windows excludes reads through another descriptor while the owned
            # byte is locked. process_file_lock validates that byte on acquisition;
            # inventories inspect only its regular-file metadata and exact size.
        elif name in ("answers", "interrupted"):
            maximum_entries = len(run.payload.answers) if name == "answers" else 100000
            for child in _names(path, maximum_entries):
                pattern = r"[0-9]{6}\.json" if name == "answers" else r"[0-9a-f]{64}\.bin"
                if re.fullmatch(pattern, child) is None:
                    raise ValueError("unexpected generation data entry")
                total += _size(
                    path / child,
                    maximum=run.payload.max_record_bytes if name == "answers" else _MAX_BYTES,
                    empty=name == "interrupted",
                )
        else:
            raise ValueError("unrecognized generation directory entry")
    return total


def _validated[T: IrisStudyGenerationRunV1 | IrisStudyGenerationAnswerV1](
    artifact: T, model: type[T], domain: str, maximum: int = _MAX_BYTES
) -> T:
    if type(artifact) is not model:
        raise TypeError("generation run artifact has an incorrect exact type")
    owned = cast(T, model.model_validate(artifact.model_dump(mode="python")))
    if owned.payload_sha256 != domain_sha256(
        domain, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("generation run artifact hash disagrees")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > maximum:
        raise ValueError("generation run artifact exceeds its byte bound")
    return owned


def _decode[T: BaseModel](content: bytes, model: type[T]) -> T:
    raw = json.loads(
        content.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if type(raw) is not dict or canonical_json_file_bytes(raw) != content:
        raise ValueError("generation storage requires canonical JSON objects")
    owned = model.model_validate_json(content)
    if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
        raise ValueError("generation storage canonical round trip changed bytes")
    return owned


def read_study_generation_run_v1(directory: Path) -> IrisStudyGenerationRunV1:
    """Read plan schema/hash only; does not verify answers or restore any state."""
    try:
        content = _read_regular_bytes(_directory(directory) / "run.json")
        return _validated(
            _decode(content, IrisStudyGenerationRunV1), IrisStudyGenerationRunV1, _RUN_DOMAIN
        )
    except Exception:
        raise IrisStudyGenerationError("Iris generation run could not be read") from None


def _read_answer(path: Path, maximum: int) -> IrisStudyGenerationAnswerV1:
    content = _read_regular_bytes(path, max_bytes=maximum)
    return _validated(
        _decode(content, IrisStudyGenerationAnswerV1),
        IrisStudyGenerationAnswerV1,
        _ANSWER_DOMAIN,
        maximum,
    )


def _stage(pending: Path, content: bytes, *, claim: ProcessFileLock) -> None:
    """Fsync the full raw-answer/request envelope before touching external requests."""
    claim.check()
    if pending.parent != claim.path.parent or pending.name not in _PENDING_NAMES:
        raise ValueError("unowned generation pending slot")
    if not 0 < len(content) <= _MAX_BYTES:
        raise ValueError("invalid generation publication length")
    _check_parents(pending, create=False)
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
                    raise OSError("generation pending write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if _read_regular_bytes(pending) != content:
        raise ValueError("generation pending bytes conflict")
    claim.check()


def _publish(destination: Path, content: bytes, *, pending: Path, claim: ProcessFileLock) -> None:
    if destination.parent not in (claim.path.parent, claim.path.parent / "answers"):
        raise ValueError("generation publication escaped the owned run")
    if destination.parent == claim.path.parent:
        expected_pending = {
            "run.json": ".run-pending.json",
            "generation.json": ".generation-pending.json",
        }.get(destination.name)
        if pending.name != expected_pending:
            raise ValueError("generation publication has an incorrect pending slot")
    elif (
        re.fullmatch(r"[0-9]{6}\.json", destination.name) is None
        or pending.name != ".answer-pending.json"
    ):
        raise ValueError("generation answer publication has an incorrect pending slot")
    _stage(pending, content, claim=claim)
    _check_parents(destination, create=True)
    claim.check()
    try:
        os.link(pending, destination)
    except FileExistsError:
        if _read_regular_bytes(destination) != content:
            raise ValueError("generation publication conflicts with existing bytes") from None
    if _read_regular_bytes(destination) != content:
        raise ValueError("published generation bytes changed")
    claim.check()
    if _read_regular_bytes(pending) != content:
        raise ValueError("pending generation changed before unlink")
    pending.unlink()


def _archive_pending(path: Path, *, claim: ProcessFileLock, maximum: int) -> None:
    claim.check()
    if path.parent != claim.path.parent or path.name not in _PENDING_NAMES:
        raise ValueError("unowned interrupted generation slot")
    content = _pending_bytes(path, maximum)
    destination = path.parent / "interrupted" / f"{bytes_sha256(content)}.bin"
    _check_parents(destination, create=True)
    try:
        os.link(path, destination)
    except FileExistsError:
        if _pending_bytes(destination, maximum) != content:
            raise ValueError("interrupted generation bytes conflict") from None
    claim.check()
    if _pending_bytes(destination, maximum) != content or _pending_bytes(path, maximum) != content:
        raise ValueError("interrupted generation byte preservation failed")
    path.unlink()


__all__ = ["read_study_generation_run_v1"]
