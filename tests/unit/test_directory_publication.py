"""Bounded native-Windows retries preserve the owned directory and concurrent winner."""

import os
from pathlib import Path
from time import sleep as real_sleep
from unittest.mock import patch

import pytest

from iris.utils import directory_publication as publication


class _WindowsDenial(PermissionError):
    def __init__(self, winerror: int) -> None:
        super().__init__(13, "synthetic native Windows denial")
        self.winerror = winerror


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "owned-stage"
    source.mkdir()
    (source / "private.json").write_bytes(b"owned bytes")
    return source, tmp_path / "published"


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_native_denial_retries_the_same_directory_only(tmp_path: Path, winerror: int) -> None:
    source, target = _paths(tmp_path)
    with (
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(publication, "sleep") as sleep,
        patch.object(os, "rename", side_effect=[_WindowsDenial(winerror), None]) as move,
    ):
        publication.rename_directory_with_windows_retry(source, target)
    assert move.call_count == 2
    assert all(call.args == (source, target) for call in move.call_args_list)
    sleep.assert_called_once_with(0.01)
    assert (source / "private.json").read_bytes() == b"owned bytes"


def test_permanent_denial_stops_after_bounded_budget(tmp_path: Path) -> None:
    source, target = _paths(tmp_path)
    denied = _WindowsDenial(5)
    with (
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(publication, "sleep") as sleep,
        patch.object(os, "rename", side_effect=denied) as move,
        pytest.raises(PermissionError) as caught,
    ):
        publication.rename_directory_with_windows_retry(source, target)
    assert caught.value is denied
    assert move.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert source.is_dir()
    assert not target.exists()


@pytest.mark.parametrize(
    ("windows", "error"),
    [
        (False, _WindowsDenial(5)),
        (True, _WindowsDenial(87)),
        (True, PermissionError(13, "no native code")),
        (True, OSError(28, "disk full")),
    ],
)
def test_unrelated_or_non_windows_errors_are_not_retried(
    tmp_path: Path, windows: bool, error: OSError
) -> None:
    source, target = _paths(tmp_path)
    with (
        patch.object(publication, "_IS_WINDOWS", windows),
        patch.object(publication, "sleep") as sleep,
        patch.object(os, "rename", side_effect=error) as move,
        pytest.raises(OSError) as caught,
    ):
        publication.rename_directory_with_windows_retry(source, target)
    assert caught.value is error
    move.assert_called_once_with(source, target)
    sleep.assert_not_called()


def test_existing_target_is_never_overwritten_even_when_empty(tmp_path: Path) -> None:
    source, target = _paths(tmp_path)
    target.mkdir()
    with patch.object(os, "rename") as move, pytest.raises(FileExistsError):
        publication.rename_directory_with_windows_retry(source, target)
    move.assert_not_called()
    assert source.is_dir()
    assert target.is_dir()


def test_concurrent_winner_stops_retry_for_caller_verification(tmp_path: Path) -> None:
    source, target = _paths(tmp_path)

    def winner_appears(_source: Path, _target: Path) -> None:
        target.mkdir()
        (target / "winner.json").write_bytes(b"concurrent private bytes")
        raise _WindowsDenial(5)

    with (
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(publication, "sleep") as sleep,
        patch.object(os, "rename", side_effect=winner_appears) as move,
        pytest.raises(PermissionError),
    ):
        publication.rename_directory_with_windows_retry(source, target)
    move.assert_called_once_with(source, target)
    sleep.assert_not_called()
    assert source.is_dir()
    assert (target / "winner.json").read_bytes() == b"concurrent private bytes"


@pytest.mark.parametrize("initial_native_denial", [False, True])
def test_replaced_source_aborts_retry(tmp_path: Path, initial_native_denial: bool) -> None:
    source, target = _paths(tmp_path)
    real_rename = os.rename
    replacement_done = False
    native_denials = 0
    replacement_error = _WindowsDenial(5)

    def replace_source(_source: Path, _target: Path) -> None:
        nonlocal native_denials, replacement_done
        assert not replacement_done, "must never attempt to publish a replacement directory"
        try:
            # A real Windows sharing denial can precede the injected replacement.
            # Force that ordering once as well as exercising the native operation.
            if initial_native_denial and native_denials == 0:
                raise _WindowsDenial(32)
            real_rename(source, tmp_path / "preserved-original")
        except OSError:
            native_denials += 1
            raise
        source.mkdir()
        replacement_done = True
        raise replacement_error

    def delay_before_replacement_only(seconds: float) -> None:
        assert not replacement_done, "must never delay retry after source replacement"
        real_sleep(seconds)

    with (
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(publication, "sleep", side_effect=delay_before_replacement_only) as sleep,
        patch.object(os, "rename", side_effect=replace_source) as move,
        pytest.raises(PermissionError) as caught,
    ):
        publication.rename_directory_with_windows_retry(source, target)
    assert caught.value is replacement_error
    assert replacement_done
    assert move.call_count == native_denials + 1
    assert sleep.call_count == native_denials
    if initial_native_denial:
        assert native_denials >= 1
    assert not target.exists()
    assert (tmp_path / "preserved-original" / "private.json").read_bytes() == b"owned bytes"


def test_changed_identity_aborts_before_any_retry_delay(tmp_path: Path) -> None:
    source, target = _paths(tmp_path)
    denied = _WindowsDenial(5)
    with (
        patch.object(publication, "_IS_WINDOWS", True),
        patch.object(publication, "_directory_identity", side_effect=[(1, 2), (1, 2), (1, 3)]),
        patch.object(publication, "sleep") as sleep,
        patch.object(os, "rename", side_effect=denied) as move,
        pytest.raises(PermissionError) as caught,
    ):
        publication.rename_directory_with_windows_retry(source, target)
    assert caught.value is denied
    move.assert_called_once_with(source, target)
    sleep.assert_not_called()
    assert (source / "private.json").read_bytes() == b"owned bytes"
    assert not target.exists()


def test_non_sibling_move_is_rejected_before_attempt(tmp_path: Path) -> None:
    source, _ = _paths(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with patch.object(os, "rename") as move, pytest.raises(ValueError):
        publication.rename_directory_with_windows_retry(source, other / "published")
    move.assert_not_called()
