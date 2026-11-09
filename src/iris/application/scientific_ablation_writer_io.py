"""Bounded immutable source/report delivery, not native execution attestation.

Caller-trusted source/report digests and CPU expectation are mandatory. No owner,
model/scorer/control execution, stage adoption/reaping or recursive deletion.
Local cooperating-process checks do not guarantee hostile-FS or power-loss safety.
"""

import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, TypedDict
from uuid import uuid4

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
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    WriterReport,
    _runtime,
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    IrisScientificAblationWriterSourceV1,
    WriterSourceProtocol,
    _digest,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
    scientific_ablation_writer_source_preflight_v1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_LOCK = ".iris-owner.lock"
_SOURCE = "source.json"
_REPORT = "report.json"
_FILES = {_SOURCE, _REPORT}
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_STAGE_PATTERN = re.compile(r"\.iris-writer-[0-9a-f]{32}\.tmp")
_MAX_STORAGE = 16 * 1024**3  # Ceiling only; default caller quota is512MiB, no allocation.
_MAX_DIRECTORIES = 4096
_MAX_FILES = 2 * _MAX_DIRECTORIES + 1  # BOTH files per directory and persistent lock.


class IrisScientificAblationWriterStorageError(RuntimeError):
    """Unbound, incomplete, conflicting or uncertain writer storage operation."""


@dataclass(frozen=True, slots=True)
class ScientificAblationWriterBundleV1:
    """Validated detached data, NOT a caller-proof native completion token."""

    report: WriterReport
    source: IrisScientificAblationWriterSourceV1


class _ReadOptions(TypedDict):
    protocol: WriterSourceProtocol
    report: str
    source: str
    runtime: ScientificAblationWriterCPURuntimeV1
    maximum_source_bytes: int
    maximum_report_bytes: int


def _store(protocol: WriterSourceProtocol) -> str:
    return f"scientific-ablation-{protocol}-writer-v1"


def _anchors(
    protocol: WriterSourceProtocol,
    report: str,
    source: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int,
) -> None:
    if type(protocol) is not str or protocol not in ("workspace", "message"):
        raise ValueError("Exact independently selected writer protocol required")
    _digest(report)
    _digest(source)
    _runtime(runtime)
    for maximum in (maximum_source_bytes, maximum_report_bytes):
        if type(maximum) is not int or not 1 <= maximum <= MAX_REPORT_BYTES:
            raise ValueError("Exact positive bounded writer source/report read caps required")


def _fail(primary: BaseException, secondary: list[BaseException]) -> NoReturn:
    if not isinstance(primary, Exception):
        raise primary from (
            BaseExceptionGroup("Writer storage cleanup failed", secondary) if secondary else None
        )
    signal = next((error for error in secondary if not isinstance(error, Exception)), None)
    if signal is not None:
        raise signal from BaseExceptionGroup(
            "Writer storage operation failed", [primary, *secondary]
        )
    raise IrisScientificAblationWriterStorageError("Writer bundle storage did not complete") from (
        primary
        if not secondary
        else BaseExceptionGroup("Writer storage operation/cleanup failed", [primary, *secondary])
    )


def _inventory(path: Path) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for entry in path.iterdir():
        if entry.name not in _FILES or entry.name in result:
            raise ValueError("Foreign writer bundle member")
        result[entry.name] = _signature(_regular(entry, directory=False))
    return result


def _read_directory(
    path: Path,
    *,
    protocol: WriterSourceProtocol,
    report: str,
    source: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int,
) -> tuple[ScientificAblationWriterBundleV1, dict[str, bytes]]:
    _chain(path)
    before = _signature(_regular(path, directory=True))
    members = _inventory(path)
    if set(members) != _FILES:
        raise ValueError("Exactly original source AND complete report required")
    source_content = _read_file(path / _SOURCE, maximum_source_bytes)
    original = read_scientific_ablation_writer_source_v1(
        source_content, expected_sha256=source, maximum_source_bytes=maximum_source_bytes
    )
    if original.payload.protocol != protocol:
        raise ValueError("Writer source differs from independently selected protocol")
    pf = scientific_ablation_writer_source_preflight_v1(original, expected_sha256=source)
    report_content = _read_file(
        path / _REPORT,
        min(maximum_report_bytes, pf.maximum_report_bytes, pf.report_byte_preflight_bound),
    )
    result = read_scientific_ablation_writer_report_v1(
        report_content, preflight=pf, runtime=runtime
    )
    if result.sha256 != report:
        raise ValueError("Report differs from independently trusted writer report")
    _chain(path)
    if _signature(_regular(path, directory=True)) != before or _inventory(path) != members:
        raise ValueError("Writer directory or either file changed during full readback")
    return ScientificAblationWriterBundleV1(result, original), {
        _SOURCE: source_content,
        _REPORT: report_content,
    }


def read_scientific_ablation_writer_bundle_v1(
    path: Path,
    *,
    expected_protocol: WriterSourceProtocol,
    expected_report_sha256: str,
    expected_source_sha256: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int = MAX_REPORT_BYTES,
) -> ScientificAblationWriterBundleV1:
    """Cold full read using independent anchors; no model/scorer/control execution."""
    try:
        _anchors(
            expected_protocol,
            expected_report_sha256,
            expected_source_sha256,
            runtime,
            maximum_source_bytes,
            maximum_report_bytes,
        )
        path = _path(path)
        if path.name != expected_report_sha256 or path.parent.name != _store(expected_protocol):
            raise ValueError("Distinct protocol store and trusted report directory required")
        result, _ = _read_directory(
            path,
            protocol=expected_protocol,
            report=expected_report_sha256,
            source=expected_source_sha256,
            runtime=runtime,
            maximum_source_bytes=maximum_source_bytes,
            maximum_report_bytes=maximum_report_bytes,
        )
        return result
    except BaseException as error:
        _fail(error, [])


@dataclass(frozen=True, slots=True)
class _Claim:
    store: Path
    identity: tuple[int, int]
    lock: ProcessFileLock

    def check(self) -> None:
        self.lock.check()
        _chain(self.store)
        if _identity(_regular(self.store, directory=True)) != self.identity:
            raise ValueError("Writer store identity changed")


@dataclass(slots=True)
class _Stage:
    path: Path
    identity: tuple[int, int]
    files: dict[str, tuple[int, int]] = field(default_factory=dict)

    def check(self, claim: _Claim) -> None:
        claim.check()
        _chain(self.path)
        if (
            self.path.parent != claim.store
            or _identity(_regular(self.path, directory=True)) != self.identity
        ):
            raise ValueError("Owned writer stage identity changed")
        members = _inventory(self.path)
        if set(members) != set(self.files) or any(
            signature[:2] != self.files[name] for name, signature in members.items()
        ):
            raise ValueError("Owned writer stage membership or file identity changed")


def _occupied(claim: _Claim) -> tuple[int, int, int]:
    claim.check()
    before = _signature(_regular(claim.store, directory=True))
    total, directories, files = 0, 0, 0
    for entry in claim.store.iterdir():
        if entry.name == _LOCK:
            files += 1
            total += _regular(entry, directory=False).st_size
        else:
            stage = _STAGE_PATTERN.fullmatch(entry.name) is not None
            if not stage and _DIGEST_PATTERN.fullmatch(entry.name) is None:
                raise ValueError("Foreign writer store entry")
            directory = _signature(_regular(entry, directory=True))
            directories += 1
            members = _inventory(entry)
            if not stage and set(members) != _FILES:
                raise ValueError("Published writer bundle is incomplete")
            files += len(members)
            total += sum(signature[2] for signature in members.values())
            if (
                _signature(_regular(entry, directory=True)) != directory
                or _inventory(entry) != members
            ):
                raise ValueError("Writer store member changed during inventory")
        if total > _MAX_STORAGE or directories > _MAX_DIRECTORIES or files > _MAX_FILES:
            raise ValueError("Writer occupied byte/directory/file budget exceeded")
    claim.check()
    if _signature(_regular(claim.store, directory=True)) != before:
        raise ValueError("Writer store changed during occupied inventory")
    return total, directories, files


def _write(stage: _Stage, name: str, content: bytes, claim: _Claim) -> None:
    stage.check(claim)
    if name not in _FILES or name in stage.files:
        raise ValueError("Exact new writer bundle member required")
    path = stage.path / name
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
            raise ValueError("New writer file identity differs")
        stage.files[name] = opened
        _write_bytes(descriptor, content)
        stage.check(claim)
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, lambda: os.close(descriptor))
    stage.check(claim)
    if _read_file(path, len(content)) != content:
        raise ValueError("Exact staged writer bytes differ")
    stage.check(claim)


def _bound(
    report: WriterReport,
    source: IrisScientificAblationWriterSourceV1,
    *,
    protocol: WriterSourceProtocol,
    report_sha256: str,
    source_sha256: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int,
) -> dict[str, bytes]:
    source_content = scientific_ablation_writer_source_bytes_v1(
        source, expected_sha256=source_sha256
    )
    original = read_scientific_ablation_writer_source_v1(
        source_content, expected_sha256=source_sha256, maximum_source_bytes=maximum_source_bytes
    )
    if original.payload.protocol != protocol:
        raise ValueError("Original source differs from independently selected writer protocol")
    pf = scientific_ablation_writer_source_preflight_v1(original, expected_sha256=source_sha256)
    report_content = scientific_ablation_writer_report_bytes_v1(
        report, preflight=pf, runtime=runtime
    )
    if len(report_content) > maximum_report_bytes:
        raise ValueError("Caller writer report byte cap exceeded")
    owned = read_scientific_ablation_writer_report_v1(report_content, preflight=pf, runtime=runtime)
    if owned.sha256 != report_sha256:
        raise ValueError("Report differs from independently trusted writer report")
    return {_SOURCE: source_content, _REPORT: report_content}


def write_scientific_ablation_writer_bundle_v1(
    report: WriterReport,
    *,
    source: IrisScientificAblationWriterSourceV1,
    root: Path,
    expected_protocol: WriterSourceProtocol,
    expected_report_sha256: str,
    expected_source_sha256: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int = MAX_REPORT_BYTES,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> Path:
    """Validate before writes; exact retry without remeasurement or stage reaping.

    No completion/provenance claim follows from constructible report/source data.
    Expected digests and CPU expectation must come from the independently trusted job.
    """
    try:
        _anchors(
            expected_protocol,
            expected_report_sha256,
            expected_source_sha256,
            runtime,
            maximum_source_bytes,
            maximum_report_bytes,
        )
        content = _bound(
            report,
            source,
            protocol=expected_protocol,
            report_sha256=expected_report_sha256,
            source_sha256=expected_source_sha256,
            runtime=runtime,
            maximum_source_bytes=maximum_source_bytes,
            maximum_report_bytes=maximum_report_bytes,
        )
        required = sum(len(value) for value in content.values())
        if (
            type(maximum_storage_bytes) is not int
            or not 1 <= maximum_storage_bytes <= _MAX_STORAGE
            or type(minimum_free_bytes) is not int
            or not 0 <= minimum_free_bytes <= _MAX_STORAGE
            or required + 1 > maximum_storage_bytes
        ):
            raise ValueError("Exact bounded writer quota/free-space reserve required")
        store = _path(root) / _store(expected_protocol)
        for parent in (store, *store.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _chain(store)
        identity = _identity(_regular(store, directory=True))
    except BaseException as error:
        _fail(error, [])
    target = store / expected_report_sha256
    read_options: _ReadOptions = dict(
        protocol=expected_protocol,
        report=expected_report_sha256,
        source=expected_source_sha256,
        runtime=runtime,
        maximum_source_bytes=maximum_source_bytes,
        maximum_report_bytes=maximum_report_bytes,
    )
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    try:
        with process_file_lock(store / _LOCK) as lock:
            claim = _Claim(store, identity, lock)
            try:
                occupied, directories, files = _occupied(claim)
                if occupied > maximum_storage_bytes:
                    raise ValueError("Occupied writer store exceeds caller quota")
                if os.path.lexists(target):
                    if _read_directory(target, **read_options)[1] != content:
                        raise ValueError("Existing immutable writer target conflicts")
                else:
                    if (
                        occupied + required > maximum_storage_bytes
                        or directories + 1 > _MAX_DIRECTORIES
                        or files + len(_FILES) > _MAX_FILES
                        or shutil.disk_usage(store).free < required + minimum_free_bytes
                    ):
                        raise ValueError("Writer quota/free-space/count exceeded before writes")
                    claim.check()
                    path = store / f".iris-writer-{uuid4().hex}.tmp"
                    os.mkdir(path, 0o700)
                    stage = _Stage(path, _identity(_regular(path, directory=True)))
                    for name in (_SOURCE, _REPORT):
                        _write(stage, name, content[name], claim)
                    if (
                        _occupied(claim)[0] > maximum_storage_bytes
                        or shutil.disk_usage(store).free < minimum_free_bytes
                    ):
                        raise ValueError("Writer quota/reserve changed before publication")
                    stage.check(claim)
                    if _read_directory(stage.path, **read_options)[1] != content:
                        raise ValueError("Staged complete writer bytes changed")
                    stage.check(claim)
                    try:
                        rename_directory_with_windows_retry(stage.path, target)
                    except OSError:
                        if not os.path.lexists(target):
                            raise
                    if _read_directory(target, **read_options)[1] != content:
                        raise ValueError("Existing/uncertain writer publication conflicts")
                    if os.path.lexists(stage.path):
                        stage.check(claim)
                        if _read_directory(stage.path, **read_options)[1] != content:
                            raise ValueError("Unpublished writer stage changed; preserve")
                claim.check()
            except BaseException as error:
                primary = error
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            secondary.append(error)
    if primary is not None:
        _fail(primary, secondary)
    return target


__all__ = [
    "IrisScientificAblationWriterStorageError",
    "ScientificAblationWriterBundleV1",
    "read_scientific_ablation_writer_bundle_v1",
    "write_scientific_ablation_writer_bundle_v1",
]
