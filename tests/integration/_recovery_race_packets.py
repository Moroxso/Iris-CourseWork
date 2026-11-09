"""Bounded, no-overwrite observation IPC. These packets grant NO Iris authority.

Only cooperating pinned test processes use this module. A complete nonempty
directory is published, so a reader never consumes a partially written packet.
Unknown stages are retained on failure; no deletion or ownership reconstruction.
"""

import json
import math
import os
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

Role = Literal["ready", "release"]
MAX_PACKET = 64 * 1024


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
    )


def binding(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "job_sha256",
        "command_sha256",
        "nonce",
        "phase",
        "scenario",
    }:
        raise ValueError("Exact observation binding required")
    for name, width in (("job_sha256", 64), ("command_sha256", 64), ("nonce", 32)):
        if (
            type(value[name]) is not str
            or re.fullmatch("[0-9a-f]{" + str(width) + "}", value[name]) is None
        ):
            raise ValueError("Exact lowercase observation digest/nonce required")
    if type(value["phase"]) is not int or not 0 <= value["phase"] < 4:
        raise ValueError("Original four-kernel phase index required")
    if type(value["scenario"]) is not str or value["scenario"] not in {"partial", "aggregate"}:
        raise ValueError("Exact race scenario required")
    return value.copy()


def checked(value: object, expected: dict[str, Any], role: Role) -> dict[str, Any]:
    expected = binding(expected)
    if role not in {"ready", "release"} or type(value) is not dict:
        raise ValueError("Exact observation packet required")
    fields = {"kind", "binding"} | (
        {"pid", "parent_pid", "started_sha256", "source_sha256", "worker_sha256", "locks", "rows"}
        if role == "ready"
        else {"ready_sha256", "contender_process_sha256"}
    )
    if set(value) != fields or value["kind"] != "iris-recovery-race-" + role + "-v1":
        raise ValueError("Closed observation packet fields required")
    if binding(value["binding"]) != expected:
        raise ValueError("Observation belongs to another immutable job/command/phase")
    digests = (
        ("started_sha256", "source_sha256", "worker_sha256")
        if role == "ready"
        else ("ready_sha256", "contender_process_sha256")
    )
    for name in digests:
        if type(value[name]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None:
            raise ValueError("Exact observed packet digest required")
    if role == "ready":
        if any(type(value[name]) is not int or value[name] <= 0 for name in ("pid", "parent_pid")):
            raise ValueError("Actual process IDs required")
        if value["pid"] == value["parent_pid"]:
            raise ValueError("Distinct actual process/parent required")
        if type(value["rows"]) is not int or value["rows"] != (
            1 if expected["scenario"] == "partial" else 4
        ):
            raise ValueError("Exact pre-race acknowledged row count required")
        if type(value["locks"]) is not dict or set(value["locks"]) != {"workflow", "kernel"}:
            raise ValueError("Both actual original lock observations required")
        paths = []
        for lock in value["locks"].values():
            if type(lock) is not dict or set(lock) != {"path", "signature", "content_sha256"}:
                raise ValueError("Exact lock observation required")
            if (
                type(lock["path"]) is not str
                or not Path(lock["path"]).is_absolute()
                or ".." in Path(lock["path"]).parts
            ):
                raise ValueError("Absolute original lock path required")
            signature = lock["signature"]
            if (
                type(signature) is not list
                or len(signature) != 6
                or any(type(n) is not int or n < 0 for n in signature)
            ):
                raise ValueError("Exact actual lock signature required")
            if (
                signature[2:4] != [1, 1]
                or lock["content_sha256"]
                != "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"
            ):
                raise ValueError("Original single-link one-byte lock required")
            paths.append(lock["path"])
        if Path(paths[0]) == Path(paths[1]):
            raise ValueError("Distinct original workflow/kernel locks required")
    if len(canonical(value)) > MAX_PACKET:
        raise ValueError("Observation packet exceeds its fixed bound")
    return value.copy()


def signature(info: os.stat_result) -> list[int]:
    stamp = info.st_ctime_ns
    if os.name == "nt":
        # Preserved run01 RED: unlike lstat/fstat ctime on Python3.12 Windows.
        # Compare the exact SAME creation FILETIME, never rounded timestamps.
        birthtime = getattr(info, "st_birthtime_ns", None)
        if type(birthtime) is not int:
            raise OSError("Exact Windows creation FILETIME is unavailable")
        stamp = birthtime
    return [
        info.st_dev,
        info.st_ino,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        stamp,
    ]


def _named_signature(info: os.stat_result) -> list[int]:
    # Preserve exact same-API ctime in addition to cross-API creation time.
    return [*signature(info), info.st_ctime_ns]


def _safe(info: os.stat_result, *, directory: bool) -> bool:
    return bool(
        (
            stat.S_ISDIR(info.st_mode)
            if directory
            else stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        )
        and not getattr(info, "st_file_attributes", 0) & 0x400
    )


def _chain(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Absolute bounded observation root required")
    for ancestor in (path, *path.parents):
        if not _safe(ancestor.lstat(), directory=True):
            raise ValueError("No reparse/symlink observation ancestors")


def publish(root: Path, role: Role, value: object, expected: dict[str, Any]) -> str:
    from hashlib import sha256

    packet = checked(value, expected, role)
    _chain(root)
    target = root / role
    if os.path.lexists(target):
        raise FileExistsError("Observation output must be new")
    stage = root / (".packet-" + role + "-" + expected["nonce"] + ".tmp")
    stage.mkdir(mode=0o700)
    raw = canonical(packet)
    with (stage / "packet.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # A completed target is nonempty, so another publication cannot overwrite it.
    os.rename(stage, target)
    actual, digest = read(root, role, expected)
    if actual != packet or digest != sha256(raw).hexdigest():
        raise ValueError("Published observation changed")
    return digest


def read(root: Path, role: Role, expected: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from hashlib import sha256

    binding(expected)
    if role not in {"ready", "release"}:
        raise ValueError("Closed observation role required")
    _chain(root)
    directory = root / role
    before_directory = directory.lstat()
    if not _safe(before_directory, directory=True) or {p.name for p in directory.iterdir()} != {
        "packet.json"
    }:
        raise ValueError("Exact complete observation directory required")
    path = directory / "packet.json"
    before = path.lstat()
    if not _safe(before, directory=False) or not 1 <= before.st_size <= MAX_PACKET:
        raise ValueError("Bounded single-link regular packet required")
    with path.open("rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("Observation changed while opening")
        raw = stream.read(MAX_PACKET + 1)
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("Observation changed during read")
    if _named_signature(path.lstat()) != _named_signature(before) or _named_signature(
        directory.lstat()
    ) != _named_signature(before_directory):
        raise ValueError("Observation identity changed")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate observation JSON key")
            result[key] = item
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if canonical(value) != raw:
        raise ValueError("Canonical finite observation JSON required")
    return checked(value, expected, role), sha256(raw).hexdigest()


def wait(
    root: Path,
    role: Role,
    expected: dict[str, Any],
    *,
    alive: Callable[[], bool],
    timeout: float = 600.0,
    interval: float = 0.1,
) -> tuple[dict[str, Any], str]:
    if any(
        type(n) not in (int, float) or not math.isfinite(n) for n in (timeout, interval)
    ) or not (0 < timeout <= 600 and 0.01 <= interval <= 0.5):
        raise ValueError("Bounded observation wait required")
    binding(expected)
    _chain(root)
    deadline = time.monotonic() + timeout
    while True:
        if alive() is not True:
            raise RuntimeError("Actual observed process is not live; never restart from a marker")
        try:
            return read(root, role, expected)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Same observation wait expired; not a restart authorization"
                ) from None
            time.sleep(interval)
