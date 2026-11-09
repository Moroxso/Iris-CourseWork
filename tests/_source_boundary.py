"""Portable, stdlib-only source import boundary for authored test subprocesses.

A snapshot describes every current package file; it grants no model ownership,
native result or installed-package acceptance. No date-stamped operational tools.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

MAX_BYTES = 64 * 1024**2
MAX_FILES = 5000
type Pins = dict[str, tuple[int, str]]


def regular(path: Path, maximum: int = MAX_BYTES) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or getattr(before, "st_file_attributes", 0) & 0x400
        or not 0 <= before.st_size <= maximum
    ):
        raise ValueError("Bounded single-link ordinary test input required")
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(raw) != before.st_size:
        raise ValueError("Test input changed during read")
    return raw


def ordinary_directory(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Absolute ordinary directory required")
    for parent in (path, *path.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Linked/reparse test directory refused")


def _members(source: Path) -> set[str]:
    ordinary_directory(source)
    package = source / "iris"
    ordinary_directory(package)
    names: set[str] = set()

    def refused(error: OSError) -> None:
        raise error

    for current, folders, files in os.walk(package, onerror=refused, followlinks=False):
        directory = Path(current)
        for node in (directory, *(directory / name for name in folders)):
            info = node.lstat()
            if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Ordinary complete package directories required")
        folders[:] = [name for name in folders if name != "__pycache__"]
        names.update(
            (directory / name).relative_to(source).as_posix()
            for name in files
            if Path(name).suffix not in {".pyc", ".pyo"}
        )
        if len(names) > MAX_FILES:
            raise ValueError("Source package exceeds fixed test file-count bound")
    if "iris/__init__.py" not in names:
        raise ValueError("Concrete Iris package required, not a namespace fallback")
    return names


def _pins(value: object) -> Pins:
    if type(value) is not dict or not 1 <= len(value) <= MAX_FILES:
        raise ValueError("Nonempty bounded package pin mapping required")
    result: Pins = {}
    total = 0
    for name, pin in value.items():
        if (
            type(name) is not str
            or "\\" in name
            or ":" in name
            or not name.startswith("iris/")
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            or type(pin) not in (list, tuple)
            or len(pin) != 2
            or type(pin[0]) is not int
            or not 0 <= pin[0] <= MAX_BYTES
            or type(pin[1]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pin[1]) is None
        ):
            raise ValueError("Exact relative package name/size/SHA256 required")
        total += pin[0]
        result[name] = (pin[0], pin[1])
    if total > MAX_BYTES or "iris/__init__.py" not in result:
        raise ValueError("Complete package exceeds fixed test byte bound")
    return result


def snapshot_package(source: Path) -> Pins:
    """Two complete membership reads; no source code is executed."""
    names = _members(source)
    result: Pins = {}
    for name in sorted(names):
        raw = regular(source / name)
        result[name] = (len(raw), sha256(raw).hexdigest())
    checked = _pins(result)
    if _members(source) != names:
        raise ValueError("Package membership changed during snapshot")
    return checked


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    )


def write_snapshot(path: Path, pins: Pins) -> str:
    ordinary_directory(path.parent)
    raw = _canonical(dict(kind="iris-authored-test-source-snapshot-v1", files=_pins(pins)))
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def read_snapshot(path: Path, expected_sha256: str) -> Pins:
    raw = regular(path, 1024**2)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Original parent-pinned snapshot bytes required")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate snapshot JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if (
        type(value) is not dict
        or set(value) != {"kind", "files"}
        or value["kind"] != "iris-authored-test-source-snapshot-v1"
        or _canonical(value) != raw
    ):
        raise ValueError("Exact canonical source snapshot required")
    return _pins(value["files"])


class PackageBoundary(importlib.abc.MetaPathFinder):
    def __init__(self, source: Path, expected: Pins) -> None:
        self.source = source
        self.expected = _pins(expected)
        self.origins: dict[str, str] = {}
        self.verify()

    def verify(self) -> None:
        if snapshot_package(self.source) != self.expected:
            raise ValueError("Complete pinned package bytes/membership changed")

    def check(self, fullname: str, origin: str | None) -> Path:
        if type(origin) is not str or not (fullname == "iris" or fullname.startswith("iris.")):
            raise ImportError("Concrete Iris module origin required")
        assert isinstance(origin, str)
        path = Path(origin).resolve(strict=True)
        if not path.is_relative_to(self.source):
            raise ImportError("Foreign Iris source fallback refused: " + fullname)
        name = path.relative_to(self.source).as_posix()
        raw = regular(path)
        if self.expected.get(name) != (len(raw), sha256(raw).hexdigest()):
            raise ImportError("Iris module is not in parent-pinned package: " + fullname)
        self.origins[fullname] = name
        return path

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> Any:
        if fullname != "iris" and not fullname.startswith("iris."):
            return None
        result = importlib.machinery.PathFinder.find_spec(fullname, path)
        if result is None:
            raise ImportError("No fallback for missing pinned Iris module: " + fullname)
        self.check(fullname, result.origin)
        if result.submodule_search_locations is not None and not all(
            Path(name).resolve().is_relative_to(self.source)
            for name in result.submodule_search_locations
        ):
            raise ImportError("Foreign Iris namespace location refused")
        return result
