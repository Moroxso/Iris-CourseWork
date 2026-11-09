"""Nonblocking local-process ownership; the persistent lock inode is never unlinked."""

import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def _regular(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        and metadata.st_nlink == 1
    )


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _lock(descriptor: int, *, acquire: bool) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    elif sys.platform in ("linux", "darwin"):
        import fcntl

        fcntl.flock(descriptor, (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)
    else:
        raise OSError("No reviewed local file-lock backend for this platform")


@dataclass(frozen=True, slots=True)
class ProcessFileLock:
    """Call check before each mutation; not a defense against a hostile filesystem owner."""

    path: Path
    descriptor: int
    identity: tuple[int, int]

    def check(self) -> None:
        named, opened = self.path.lstat(), os.fstat(self.descriptor)
        if (
            not _regular(named)
            or not _regular(opened)
            or _identity(named) != self.identity
            or _identity(opened) != self.identity
            or named.st_size != 1
            or opened.st_size != 1
        ):
            raise OSError("Local ownership file changed")


@contextmanager
def process_file_lock(path: Path) -> Iterator[ProcessFileLock]:
    """Claim a regular local file without waiting; the OS releases ownership on death.

    Caller must validate the containing directory. The one-byte file is kept to
    avoid unlink/recreate races between holders. No PID-based stale lock reaping.
    The byte-range lock may extend past EOF while the first holder initializes it.
    """
    try:
        before = path.lstat()
    except FileNotFoundError:
        before = None
    if before is not None and (not _regular(before) or before.st_size not in (0, 1)):
        raise OSError("Invalid local ownership file")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        os.set_inheritable(descriptor, False)
        opened, named = os.fstat(descriptor), path.lstat()
        if (
            not _regular(opened)
            or not _regular(named)
            or _identity(opened) != _identity(named)
            or (before is not None and _identity(before) != _identity(opened))
        ):
            raise OSError("Ownership file changed while opening")
        _lock(descriptor, acquire=True)
        acquired = True
        os.lseek(descriptor, 0, os.SEEK_SET)
        value = os.read(descriptor, 2)
        if value == b"":
            if os.write(descriptor, b"1") != 1:
                raise OSError("Ownership initialization failed")
            os.fsync(descriptor)
        elif value != b"1":
            raise OSError("Unexpected local ownership bytes")
        claim = ProcessFileLock(path, descriptor, _identity(opened))
        claim.check()
        yield claim
        claim.check()
    finally:
        try:
            if acquired:
                _lock(descriptor, acquire=False)
        finally:
            os.close(descriptor)


__all__ = ["ProcessFileLock", "process_file_lock"]
