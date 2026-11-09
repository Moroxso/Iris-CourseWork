"""Immutable mode-bound score reports; no scoring, model loading or stage reaping.

Directory rename plus exact readback addresses cooperating local processes and
lost write acknowledgements. It is not hostile-filesystem execution attestation
or a power-loss durability guarantee. Failed/old stages are preserved and charged.
"""

import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from iris.application.scientific_ablation_evaluation_v2 import (
    _MAX_REPORT,
    IrisScientificAblationEvaluationError,
    _checked,
)
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _finish_descriptor,
    _identity,
    _path,
    _read_file,
    _regular,
    _signature,
    _write_bytes,
)
from iris.application.scientific_checkpoint import _json
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_STORE = "scientific-ablation-evaluation-v2"
_LOCK = ".iris-owner.lock"
_FILE = "evaluation.json"
_STAGE_PATTERN = re.compile(r"\.iris-evaluation-[0-9a-f]{32}\.tmp")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_STORAGE = 16 * 1024**3
_MAX_DIRECTORIES = 4096
_MAX_FILES = 4097  # One report per directory, plus the persistent one-byte lock.


def _fail(primary: BaseException, secondary: list[BaseException]) -> NoReturn:
    if not isinstance(primary, Exception):
        if secondary:
            raise primary from BaseExceptionGroup("Evaluation lock cleanup failed", secondary)
        raise primary
    for error in secondary:
        if not isinstance(error, Exception):
            raise error from BaseExceptionGroup(
                "Evaluation write/cleanup failed", [primary, *secondary]
            )
    raise IrisScientificAblationEvaluationError("Mode-bound evaluation storage failed") from None


def _anchors(artifact: str, plan: str, capture: str) -> None:
    for value in (artifact, plan, capture):
        _digest(value)


def _bound(
    result: IrisScientificAblationEvaluationV2, artifact: str, plan: str, capture: str
) -> bytes:
    content = _checked(result)  # Exact tree/schema/domain BEFORE serialization can erase fields.
    if (
        result.sha256 != artifact
        or result.payload.plan.sha256 != plan
        or result.payload.capture.sha256 != capture
    ):
        raise ValueError("Report differs from independently trusted evaluation/plan/capture")
    return content


def _inventory(path: Path) -> dict[str, tuple[int, ...]]:
    members: dict[str, tuple[int, ...]] = {}
    for entry in path.iterdir():
        if entry.name != _FILE or entry.name in members:
            raise ValueError("Foreign evaluation directory member")
        members[entry.name] = _signature(_regular(entry, directory=False))
    return members


def _read_directory(
    path: Path, artifact: str, plan: str, capture: str, maximum: int
) -> tuple[IrisScientificAblationEvaluationV2, bytes]:
    _chain(path)
    before = _signature(_regular(path, directory=True))
    members = _inventory(path)
    if set(members) != {_FILE}:
        raise ValueError("Exactly one immutable evaluation report required")
    content = _read_file(path / _FILE, maximum)
    if not content.endswith(b"\n"):
        raise ValueError("Canonical evaluation JSON plus LF required")
    _json(content[:-1])  # Reject duplicate fields/nonfinite values before Pydantic decoding.
    result = IrisScientificAblationEvaluationV2.model_validate_json(content[:-1])
    if _bound(result, artifact, plan, capture) != content:
        raise ValueError("Canonical evaluation report bytes differ")
    _chain(path)
    if _signature(_regular(path, directory=True)) != before or _inventory(path) != members:
        raise ValueError("Evaluation directory or report changed during readback")
    return result, content


def read_scientific_ablation_evaluation_v2(
    path: Path,
    *,
    expected_sha256: str,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    maximum_report_bytes: int = _MAX_REPORT,
) -> IrisScientificAblationEvaluationV2:
    """Read a hash-only report using three independent anchors; never execute code.

    Integrity/readback does not attest that an untrusted producer actually scored
    the data. Keep the expected hashes independently of this store.
    """
    try:
        _anchors(expected_sha256, expected_plan_sha256, expected_capture_sha256)
        if type(maximum_report_bytes) is not int or not 1 <= maximum_report_bytes <= _MAX_REPORT:
            raise ValueError("Exact bounded report-read quota required")
        path = _path(path)
        if path.name != expected_sha256 or path.parent.name != _STORE:
            raise ValueError("Distinct evaluation store and trusted artifact directory required")
        result, _ = _read_directory(
            path,
            expected_sha256,
            expected_plan_sha256,
            expected_capture_sha256,
            maximum_report_bytes,
        )
        return result
    except Exception:
        raise IrisScientificAblationEvaluationError(
            "Mode-bound evaluation readback failed"
        ) from None


@dataclass(frozen=True, slots=True)
class _Claim:
    store: Path
    identity: tuple[int, int]
    lock: ProcessFileLock

    def check(self) -> None:
        self.lock.check()
        _chain(self.store)
        if _identity(_regular(self.store, directory=True)) != self.identity:
            raise ValueError("Evaluation store was replaced")


@dataclass(slots=True)
class _Stage:
    path: Path
    identity: tuple[int, int]
    file_identity: tuple[int, int] | None = None

    def check(self, claim: _Claim) -> None:
        claim.check()
        _chain(self.path)
        if (
            self.path.parent != claim.store
            or _identity(_regular(self.path, directory=True)) != self.identity
        ):
            raise ValueError("Evaluation stage or parent was replaced")
        if (
            self.file_identity is not None
            and _identity(_regular(self.path / _FILE, directory=False)) != self.file_identity
        ):
            raise ValueError("Owned evaluation report was replaced")


def _occupied(claim: _Claim) -> tuple[int, int, int]:
    """Exact shallow store inventory; all partial/stale stages consume quota."""
    claim.check()
    before = _signature(_regular(claim.store, directory=True))
    total, directories, files = 0, 0, 0
    for entry in claim.store.iterdir():
        if entry.name == _LOCK:
            files += 1
            total += _regular(entry, directory=False).st_size
            if files > _MAX_FILES or total > _MAX_STORAGE:
                raise ValueError("Evaluation file-count/global byte budget exceeded")
            continue
        stage = _STAGE_PATTERN.fullmatch(entry.name) is not None
        if not stage and _DIGEST_PATTERN.fullmatch(entry.name) is None:
            raise ValueError("Foreign evaluation store entry")
        directory = _signature(_regular(entry, directory=True))
        directories += 1
        if directories > _MAX_DIRECTORIES:
            raise ValueError("Evaluation directory-count budget exceeded")
        members = _inventory(entry)
        if not stage and set(members) != {_FILE}:
            raise ValueError("Published evaluation must contain its sole report")
        for signature in members.values():
            files += 1
            total += signature[2]
            if files > _MAX_FILES or total > _MAX_STORAGE:
                raise ValueError("Evaluation file-count/global byte budget exceeded")
        if _signature(_regular(entry, directory=True)) != directory or _inventory(entry) != members:
            raise ValueError("Evaluation store member changed during inventory")
    claim.check()
    if _signature(_regular(claim.store, directory=True)) != before:
        raise ValueError("Evaluation store changed during inventory")
    return total, directories, files


def _write(stage: _Stage, content: bytes, claim: _Claim) -> None:
    stage.check(claim)
    path = stage.path / _FILE
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    primary = None
    try:
        opened = _identity(os.fstat(descriptor))
        if opened != _identity(_regular(path, directory=False)):
            raise ValueError("New evaluation file identity differs")
        stage.file_identity = opened
        _write_bytes(descriptor, content)
        stage.check(claim)
        if opened != _identity(_regular(path, directory=False)):
            raise ValueError("Evaluation file replaced during write")
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, lambda: os.close(descriptor))
    stage.check(claim)
    if _read_file(path, len(content)) != content:
        raise ValueError("Exact evaluation staged bytes differ")
    stage.check(claim)


def _publish(
    stage: _Stage,
    target: Path,
    content: bytes,
    claim: _Claim,
    artifact: str,
    plan: str,
    capture: str,
) -> None:
    stage.check(claim)
    if _read_directory(stage.path, artifact, plan, capture, len(content))[1] != content:
        raise ValueError("Staged evaluation bytes changed")
    stage.check(claim)
    try:
        rename_directory_with_windows_retry(stage.path, target)
    except OSError:
        if not os.path.lexists(target):
            raise
    # Only an ordinary uncertain rename is reconciled. Signals never become success.
    if _read_directory(target, artifact, plan, capture, len(content))[1] != content:
        raise ValueError("Existing/uncertain evaluation publication conflicts")
    if os.path.lexists(stage.path):
        stage.check(claim)
        if _read_directory(stage.path, artifact, plan, capture, len(content))[1] != content:
            raise ValueError("Unpublished stage changed; preserve without adoption")
    claim.check()


def write_scientific_ablation_evaluation_v2(
    result: IrisScientificAblationEvaluationV2,
    *,
    root: Path,
    expected_sha256: str,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> Path:
    """Validate BEFORE writes; publish atomically, exact retry, no overwrite/reaping.

    A persistent nonblocking OS lock spans quota checks and immutable publication.
    Even this invocation's failed stage is preserved for explicit later retention.
    No scoring, generation, source regeneration or model construction occurs.
    """
    try:
        _anchors(expected_sha256, expected_plan_sha256, expected_capture_sha256)
        content = _bound(result, expected_sha256, expected_plan_sha256, expected_capture_sha256)
        if (
            type(maximum_storage_bytes) is not int
            or not 1 <= maximum_storage_bytes <= _MAX_STORAGE
            or type(minimum_free_bytes) is not int
            or not 0 <= minimum_free_bytes <= _MAX_STORAGE
            or len(content) + 1 > maximum_storage_bytes
        ):
            raise ValueError("Exact bounded evaluation quota/free-space reserve required")
        store = _path(root) / _STORE
        for parent in (store, *store.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _chain(store)
        identity = _identity(_regular(store, directory=True))
    except BaseException as error:
        _fail(error, [])
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    target = store / expected_sha256
    try:
        with process_file_lock(store / _LOCK) as lock:
            claim = _Claim(store, identity, lock)
            try:
                occupied, directories, files = _occupied(claim)
                if occupied > maximum_storage_bytes:
                    raise ValueError("Existing evaluation store exceeds quota")
                if os.path.lexists(target):
                    if (
                        _read_directory(
                            target,
                            expected_sha256,
                            expected_plan_sha256,
                            expected_capture_sha256,
                            len(content),
                        )[1]
                        != content
                    ):
                        raise ValueError("Immutable evaluation target conflicts")
                else:
                    if (
                        occupied + len(content) > maximum_storage_bytes
                        or directories >= _MAX_DIRECTORIES
                        or files >= _MAX_FILES
                        or shutil.disk_usage(store).free < len(content) + minimum_free_bytes
                    ):
                        raise ValueError("Quota/free-space/count exceeded before report write")
                    claim.check()
                    path = store / f".iris-evaluation-{uuid4().hex}.tmp"
                    os.mkdir(path, 0o700)
                    stage = _Stage(path, _identity(_regular(path, directory=True)))
                    _write(stage, content, claim)
                    if (
                        _occupied(claim)[0] > maximum_storage_bytes
                        or shutil.disk_usage(store).free < minimum_free_bytes
                    ):
                        raise ValueError("Evaluation quota/reserve changed before publication")
                    _publish(
                        stage,
                        target,
                        content,
                        claim,
                        expected_sha256,
                        expected_plan_sha256,
                        expected_capture_sha256,
                    )
                claim.check()
            except BaseException as error:
                primary = error  # Keep first signal independently of lock __exit__.
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            secondary.append(error)
    if primary is not None:
        _fail(primary, secondary)
    return target


__all__ = ["read_scientific_ablation_evaluation_v2", "write_scientific_ablation_evaluation_v2"]
