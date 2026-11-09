"""Authored package bytes/import metadata tests; no Iris/native execution proof."""

import json
import os
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from tests._source_boundary import (
    PackageBoundary,
    read_snapshot,
    regular,
    snapshot_package,
    write_snapshot,
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    package = root / "iris"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b'"""Labelled source specimen."""\n')
    (package / "specimen.py").write_bytes(b"LABELLED = True\n")
    (package / "asset.txt").write_bytes(b"asset\n")
    return root


def test_full_snapshot_roundtrip_and_no_overwrite(source: Path, tmp_path: Path) -> None:
    before = snapshot_package(source)
    path = tmp_path / "snapshot.json"
    digest = write_snapshot(path, before)
    assert read_snapshot(path, digest) == before and len(before) == 3
    raw = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_snapshot(path, before)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ("bytes", "extra", "missing", "asset"))
def test_every_source_member_is_bound(source: Path, fault: str) -> None:
    boundary = PackageBoundary(source, snapshot_package(source))
    path = source / "iris/specimen.py"
    if fault == "bytes":
        path.write_bytes(b"CHANGED = True\n")
    elif fault == "extra":
        (source / "iris/foreign.py").write_bytes(b"FOREIGN = True\n")
    elif fault == "missing":
        path.unlink()
    else:
        (source / "iris/asset.txt").write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        boundary.verify()


def test_regenerable_bytecode_is_not_a_source_delivery(source: Path) -> None:
    expected = snapshot_package(source)
    cache = source / "iris/__pycache__"
    cache.mkdir()
    (cache / "specimen.pyc").write_bytes(b"not executed")
    assert snapshot_package(source) == expected


@pytest.mark.parametrize(
    "name",
    ("outside.py", "iris/../outside.py", "iris//x.py", "iris/./x.py", "iris\\x.py", "iris/C:x.py"),
)
def test_noncanonical_or_escaping_names_are_rejected(source: Path, name: str) -> None:
    pins = snapshot_package(source)
    pins[name] = (0, "0" * 64)
    with pytest.raises(ValueError):
        PackageBoundary(source, pins)


@pytest.mark.parametrize(
    "pin",
    (
        (True, "0" * 64),
        (-1, "0" * 64),
        (0, "A" * 64),
        (0, "0" * 63),
        (0, None),
        (0, "0" * 64, "extra"),
    ),
)
def test_snapshot_scalars_are_exact(source: Path, pin: object) -> None:
    candidate = dict(snapshot_package(source))
    candidate["iris/specimen.py"] = pin  # type: ignore[assignment]
    with pytest.raises(ValueError):
        PackageBoundary(source, candidate)


def test_pinned_origin_and_real_finder_without_executing_source(source: Path) -> None:
    boundary = PackageBoundary(source, snapshot_package(source))
    path = source / "iris/specimen.py"
    assert boundary.check("iris.specimen", str(path)) == path
    spec = boundary.find_spec("iris.specimen", [str(source / "iris")])
    assert spec is not None and spec.origin == str(path)
    assert boundary.origins == {"iris.specimen": "iris/specimen.py"}
    assert boundary.find_spec("unrelated.module") is None


@pytest.mark.parametrize("fault", ("none", "foreign", "unlisted", "missing"))
def test_foreign_missing_and_unlisted_origins_fail(
    source: Path, tmp_path: Path, fault: str
) -> None:
    boundary = PackageBoundary(source, snapshot_package(source))
    path = tmp_path / "foreign.py" if fault == "foreign" else source / "iris/unlisted.py"
    if fault != "missing":
        path.write_bytes(b"AUTHORED = True\n")
    with pytest.raises((ImportError, FileNotFoundError)):
        boundary.check("iris.specimen", None if fault == "none" else str(path))


def test_missing_module_has_no_import_fallback(source: Path) -> None:
    boundary = PackageBoundary(source, snapshot_package(source))
    with (
        patch("importlib.machinery.PathFinder.find_spec", return_value=None),
        pytest.raises(ImportError),
    ):
        boundary.find_spec("iris.missing", [str(source / "iris")])


def test_manifest_hash_duplicates_and_shape_are_bound(source: Path, tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    digest = write_snapshot(path, snapshot_package(source))
    with pytest.raises(ValueError):
        read_snapshot(path, "0" * 64)
    assert read_snapshot(path, digest)
    raw = b'{"kind":"iris-authored-test-source-snapshot-v1","kind":"duplicate","files":{}}\n'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Duplicate"):
        read_snapshot(path, sha256(raw).hexdigest())
    raw = json.dumps(dict(kind="other", files={})).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_snapshot(path, sha256(raw).hexdigest())


def test_source_change_between_membership_reads_is_refused(source: Path) -> None:
    from tests import _source_boundary as module

    original = module._members
    first = original(source)
    with (
        patch.object(module, "_members", side_effect=[first, first | {"iris/late.py"}]),
        pytest.raises(ValueError),
    ):
        snapshot_package(source)


def test_hardlinked_inputs_are_refused(source: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked.py"
    os.link(source / "iris/specimen.py", linked)
    with pytest.raises(ValueError):
        regular(linked)
    with pytest.raises(ValueError):
        snapshot_package(source)
