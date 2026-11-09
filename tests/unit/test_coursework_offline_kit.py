"""Local kit inventory/admission; no model load, installation or network."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def kit() -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_kit_test", ROOT / "scripts/coursework_offline_kit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path", ["../escape", "./file", "a//b", "C:/escape", "a\\b", "", "/absolute"]
)
def test_refuses_unsafe_manifest_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="relative POSIX"):
        kit().relative_file(tmp_path, path)


def test_inventory_hashes_and_detects_same_size_modification(tmp_path: Path) -> None:
    module = kit()
    project = tmp_path / "project"
    snapshot = tmp_path / "snapshot"
    project.mkdir()
    snapshot.mkdir()
    (project / "code.py").write_bytes(b"original")
    (snapshot / "config.json").write_bytes(b"{}")
    roots = dict(project=project, snapshot=snapshot)
    layout = dict(project=["code.py"], snapshot=["config.json"])
    entries = module.collect(roots, layout)
    assert module.verify_files(entries, roots, layout) == 2
    (project / "code.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match=r"project:code\.py"):
        module.verify_files(entries, roots, layout)


@pytest.mark.parametrize("failure", ["omit", "duplicate", "foreign", "traversal"])
def test_manifest_cannot_shrink_or_expand_inventory(tmp_path: Path, failure: str) -> None:
    module = kit()
    path = tmp_path / "file"
    path.write_bytes(b"retained")
    layout, roots = dict(project=["file"]), dict(project=tmp_path)
    entries = module.collect(roots, layout)
    if failure == "omit":
        entries = []
    elif failure == "duplicate":
        entries.append(entries[0].copy())
    elif failure == "foreign":
        entries[0]["scope"] = "foreign"
    else:
        entries[0]["path"] = "../escape"
    with pytest.raises(ValueError, match="membership"):
        module.verify_files(entries, roots, layout)
    assert path.read_bytes() == b"retained"


def test_missing_reference_is_a_failure(tmp_path: Path) -> None:
    module = kit()
    entries = [dict(scope="project", path="missing", bytes=0, sha256="0" * 64)]
    with pytest.raises(ValueError, match="content mismatch"):
        module.verify_files(entries, dict(project=tmp_path), dict(project=["missing"]))


def test_exclusive_output_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "retained"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        kit().write_new(path, b"replacement")
    assert path.read_bytes() == b"original"


def test_checkpoint_must_be_inside_project(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside this project"):
        kit().layout_for(tmp_path / "checkpoint")


def test_relative_snapshot_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kit",
            "build",
            "--snapshot",
            "relative",
            "--checkpoint",
            str(tmp_path),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="pinned absolute"):
        kit().main()
    assert not output.exists()


def test_wrapper_uses_separate_arguments_and_no_policy_bypass() -> None:
    text = (ROOT / "scripts/coursework_defence.ps1").read_text("utf-8")
    assert "& $taskPython -I -B $taskKitScript verify" in text
    assert "--prompt $Prompt" in text
    assert "Invoke-Expression" not in text and "ExecutionPolicy" not in text
