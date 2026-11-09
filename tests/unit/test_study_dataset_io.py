"""Canonical storage, no-clobber installation, and path safety for study inputs."""

from __future__ import annotations

import os
import stat
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from iris.application import study_dataset_io
from iris.application.errors import IrisStudyDatasetError
from iris.application.study_dataset_io import (
    IrisStudyDatasetWriteResultV1,
    read_study_dataset_v1,
    write_study_dataset_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_study_dataset import _dataset


def test_canonical_round_trip_and_idempotent_reuse(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "nested" / "inputs" / "study.json"

    result = write_study_dataset_v1(dataset, path=str(path))
    loaded = read_study_dataset_v1(str(path))
    reused = write_study_dataset_v1(dataset, path=path)

    assert type(result) is IrisStudyDatasetWriteResultV1
    assert result.path == path
    assert result.payload_sha256 == dataset.payload_sha256
    assert result.size_bytes == path.stat().st_size
    assert result.reused_existing is False
    assert reused.reused_existing is True
    assert loaded == dataset
    assert loaded is not dataset
    assert loaded.payload.manifest is not dataset.payload.manifest
    assert path.read_bytes() == canonical_json_file_bytes(dataset.model_dump(mode="json"))
    assert not tuple(path.parent.glob(".iris-study-*.tmp"))


@pytest.mark.parametrize(
    "tamper",
    [
        "bom",
        "missing_lf",
        "double_lf",
        "crlf",
        "whitespace",
        "duplicate",
        "hash",
        "nested_hash",
        "unknown",
        "nan",
        "infinity",
        "invalid_utf8",
    ],
)
def test_reader_rejects_noncanonical_and_tampered_bytes(tmp_path: Path, tamper: str) -> None:
    dataset = _dataset()
    path = tmp_path / "study.json"
    content = canonical_json_file_bytes(dataset.model_dump(mode="json"))
    if tamper == "bom":
        content = b"\xef\xbb\xbf" + content
    elif tamper == "missing_lf":
        content = content[:-1]
    elif tamper == "double_lf":
        content += b"\n"
    elif tamper == "crlf":
        content = content[:-1] + b"\r\n"
    elif tamper == "whitespace":
        content = content[:-1] + b" \n"
    elif tamper == "duplicate":
        content = b'{"payload_sha256":"' + dataset.payload_sha256.encode() + b'",' + content[1:]
    elif tamper == "hash":
        content = content.replace(dataset.payload_sha256.encode(), b"0" * 64, 1)
    elif tamper == "nested_hash":
        content = content.replace(dataset.payload.manifest.payload_sha256.encode(), b"0" * 64, 1)
    elif tamper == "unknown":
        content = content[:-2] + b',"unknown":true}\n'
    elif tamper == "nan":
        content = content[:-2] + b',"unknown":NaN}\n'
    elif tamper == "infinity":
        content = content[:-2] + b',"unknown":Infinity}\n'
    else:
        content = b'"\xff"\n'
    path.write_bytes(content)

    with pytest.raises(IrisStudyDatasetError, match="read or verified"):
        read_study_dataset_v1(path)


def test_reader_does_not_expose_invalid_plaintext_in_traceback(tmp_path: Path) -> None:
    secret = "PRIVATE_INPUT_MUST_NOT_APPEAR_IN_ERROR"
    path = tmp_path / "study.json"
    path.write_bytes(canonical_json_file_bytes({"payload": secret}))

    with pytest.raises(IrisStudyDatasetError) as caught:
        read_study_dataset_v1(path)

    assert secret not in str(caught.value)
    assert secret not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("name", ["study.txt", "study", "parent/../study.json"])
def test_reader_writer_require_explicit_json_without_parent_traversal(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    with pytest.raises(IrisStudyDatasetError):
        read_study_dataset_v1(path)
    with pytest.raises(IrisStudyDatasetError):
        write_study_dataset_v1(_dataset(), path=path)
    assert not (tmp_path / "parent").exists()


def test_reader_writer_reject_nonpath_arguments() -> None:
    bad_path = cast(Path, 123)
    with pytest.raises(IrisStudyDatasetError):
        read_study_dataset_v1(bad_path)
    with pytest.raises(IrisStudyDatasetError):
        write_study_dataset_v1(_dataset(), path=bad_path)


def test_reader_writer_reject_directory_leaf(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.mkdir()
    with pytest.raises(IrisStudyDatasetError):
        read_study_dataset_v1(path)
    with pytest.raises(IrisStudyDatasetError):
        write_study_dataset_v1(_dataset(), path=path)
    assert path.is_dir()


def test_writer_revalidates_before_creating_directories(tmp_path: Path) -> None:
    dataset = _dataset().model_copy(update={"payload_sha256": "0" * 64})
    path = tmp_path / "uncreated" / "study.json"
    with pytest.raises(IrisStudyDatasetError):
        write_study_dataset_v1(dataset, path=path)
    assert not path.parent.exists()


def test_writer_preserves_conflicting_file(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    original = b"unrelated private user data"
    path.write_bytes(original)
    with pytest.raises(IrisStudyDatasetError, match="conflicts"):
        write_study_dataset_v1(_dataset(), path=path)
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob(".iris-study-*.tmp"))


@pytest.mark.parametrize("same_content", [True, False])
def test_concurrent_install_preserves_winner_and_cleans_temp(
    tmp_path: Path, same_content: bool
) -> None:
    dataset = _dataset()
    expected = canonical_json_file_bytes(dataset.model_dump(mode="json"))
    winner = expected if same_content else b"concurrent unrelated content"
    path = tmp_path / "study.json"

    def install_winner(source: Path, target: Path) -> None:
        del source
        target.write_bytes(winner)
        raise FileExistsError

    with patch.object(os, "link", side_effect=install_winner):
        if same_content:
            assert write_study_dataset_v1(dataset, path=path).reused_existing
        else:
            with pytest.raises(IrisStudyDatasetError, match="conflicts"):
                write_study_dataset_v1(dataset, path=path)

    assert path.read_bytes() == winner
    assert not tuple(tmp_path.glob(".iris-study-*.tmp"))


def test_link_failure_removes_only_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    unrelated = tmp_path / "keep.txt"
    unrelated.write_bytes(b"keep")
    with (
        patch.object(os, "link", side_effect=OSError("synthetic link failure")),
        pytest.raises(IrisStudyDatasetError),
    ):
        write_study_dataset_v1(_dataset(), path=path)
    assert not path.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not tuple(tmp_path.glob(".iris-study-*.tmp"))


def test_temporary_filename_collision_never_deletes_existing_file(tmp_path: Path) -> None:
    temporary_path = tmp_path / ".iris-study-collision.tmp"
    temporary_path.write_bytes(b"existing temporary user data")
    with (
        patch.object(study_dataset_io, "uuid4", return_value=SimpleNamespace(hex="collision")),
        pytest.raises(IrisStudyDatasetError),
    ):
        write_study_dataset_v1(_dataset(), path=tmp_path / "study.json")
    assert temporary_path.read_bytes() == b"existing temporary user data"


@pytest.mark.parametrize("kind", ["leaf", "parent"])
def test_reader_writer_reject_windows_reparse_entry(tmp_path: Path, kind: str) -> None:
    dataset = _dataset()
    parent = tmp_path / "nested"
    parent.mkdir()
    path = parent / "study.json"
    expected = canonical_json_file_bytes(dataset.model_dump(mode="json"))
    path.write_bytes(expected)
    rejected = path if kind == "leaf" else parent
    real_lstat = Path.lstat

    def reparse_lstat(candidate: Path) -> os.stat_result:
        if candidate == rejected:
            return cast(
                os.stat_result,
                SimpleNamespace(
                    st_mode=stat.S_IFREG if kind == "leaf" else stat.S_IFDIR,
                    st_file_attributes=0x400,
                ),
            )
        return real_lstat(candidate)

    with patch.object(Path, "lstat", reparse_lstat):
        with pytest.raises(IrisStudyDatasetError):
            read_study_dataset_v1(path)
        with pytest.raises(IrisStudyDatasetError):
            write_study_dataset_v1(dataset, path=path)
    assert path.read_bytes() == expected
    assert not tuple(parent.glob(".iris-study-*.tmp"))


@pytest.mark.parametrize("kind", ["leaf", "parent"])
def test_reader_writer_reject_symlink_entry(tmp_path: Path, kind: str) -> None:
    dataset = _dataset()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    target = real_parent / "study.json"
    expected = canonical_json_file_bytes(dataset.model_dump(mode="json"))
    target.write_bytes(expected)
    link = tmp_path / ("link.json" if kind == "leaf" else "link")
    try:
        link.symlink_to(
            target if kind == "leaf" else real_parent,
            target_is_directory=kind == "parent",
        )
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {type(error).__name__}")
    path = link if kind == "leaf" else link / "study.json"
    with pytest.raises(IrisStudyDatasetError):
        read_study_dataset_v1(path)
    with pytest.raises(IrisStudyDatasetError):
        write_study_dataset_v1(dataset, path=path)
    assert target.read_bytes() == expected


def test_reader_enforces_size_limit_before_opening_file(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.write_bytes(b"x" * 11)
    with (
        patch.object(study_dataset_io, "_MAX_DATASET_BYTES", 10),
        patch.object(os, "open") as open_file,
        pytest.raises(IrisStudyDatasetError),
    ):
        read_study_dataset_v1(path)
    open_file.assert_not_called()


def test_writer_enforces_size_limit_before_creating_directories(tmp_path: Path) -> None:
    path = tmp_path / "uncreated" / "study.json"
    with (
        patch.object(study_dataset_io, "_MAX_DATASET_BYTES", 10),
        pytest.raises(IrisStudyDatasetError),
    ):
        write_study_dataset_v1(_dataset(), path=path)
    assert not path.parent.exists()


def test_reader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.write_bytes(b"")
    with pytest.raises(IrisStudyDatasetError):
        read_study_dataset_v1(path)


@pytest.mark.parametrize("changes_during_read", [False, True])
def test_reader_compares_ctime_only_within_the_same_metadata_source(
    tmp_path: Path, changes_during_read: bool
) -> None:
    dataset = _dataset()
    path = tmp_path / "study.json"
    path.write_bytes(canonical_json_file_bytes(dataset.model_dump(mode="json")))
    real_fstat = os.fstat
    calls = 0

    def fstat_with_different_ctime(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_size=observed.st_size,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns + 100 + int(changes_during_read and calls > 1),
                st_mode=observed.st_mode,
                st_file_attributes=getattr(observed, "st_file_attributes", 0),
            ),
        )

    with patch.object(os, "fstat", side_effect=fstat_with_different_ctime):
        if changes_during_read:
            with pytest.raises(IrisStudyDatasetError):
                read_study_dataset_v1(path)
        else:
            assert read_study_dataset_v1(path) == dataset


def test_reader_does_not_double_close_owned_stream_on_validation_failure(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.write_bytes(canonical_json_file_bytes(_dataset().model_dump(mode="json")))
    with (
        patch.object(os, "fstat", side_effect=OSError("synthetic read metadata failure")),
        patch.object(os, "close", wraps=os.close) as close,
        pytest.raises(IrisStudyDatasetError),
    ):
        read_study_dataset_v1(path)
    # fdopen transferred ownership to the stream; only that stream should close its descriptor.
    close.assert_not_called()
