"""Bounded publication retries for owned sibling directories on Windows."""

import os
import stat
import sys
from pathlib import Path
from time import sleep

_IS_WINDOWS = sys.platform == "win32"
_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16)
_RETRYABLE_WINDOWS_ERRORS = {5, 32, 33}


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & 0x400
    ):
        raise ValueError("publication source must remain a real directory")
    return metadata.st_dev, metadata.st_ino


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def rename_directory_with_windows_retry(source: Path, target: Path) -> None:
    """Move the same owned staging directory, never retry training or file creation.

    Only native Windows access/sharing/lock denials are retried, at most five
    times (310 ms total delay). A target appearing or a replaced/missing source
    aborts retry. Callers still verify an existing target before treating it as
    a successful concurrent publication. This is not a hostile-filesystem lock.
    """
    if source.parent.resolve(strict=True) != target.parent.resolve(strict=True):
        raise ValueError("publication directories must have the same existing parent")
    original_identity = _directory_identity(source)
    for attempt in range(len(_RETRY_DELAYS) + 1):
        if _directory_identity(source) != original_identity:
            raise ValueError("publication source identity changed")
        if _entry_exists(target):
            raise FileExistsError("publication target already exists")
        try:
            os.rename(source, target)
            return
        except OSError as error:
            if (
                not _IS_WINDOWS
                or getattr(error, "winerror", None) not in _RETRYABLE_WINDOWS_ERRORS
                or attempt == len(_RETRY_DELAYS)
            ):
                raise
            # Do not delay a conflict; the caller must audit the winner's exact bytes.
            if _entry_exists(target) or _directory_identity(source) != original_identity:
                raise
            sleep(_RETRY_DELAYS[attempt])


__all__ = ["rename_directory_with_windows_retry"]
