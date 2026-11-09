"""Supervisor artifact/terminal safeguards without a model or background process."""

import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_scientific_ablation_writer_geometry_gate as runner


def test_fixed_whole_wide_pilot_keeps_exact_case_and_full_source_file_pin() -> None:
    selected, count = runner.scope_config("wide-pilot")
    assert selected == [runner.WIDE] and count == 1
    assert runner.scope_files(selected) == [runner.GEOMETRY]
    assert runner.scope_files([runner.GEOMETRY, runner.WIDE]) == [runner.GEOMETRY]


def test_actual_pass_ids_keep_exact_order_and_parameter_ids(tmp_path: Path) -> None:
    path = tmp_path / "passed.xml"
    path.write_text(
        '<testsuites><testsuite tests="2" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.integration.fixture" name="test_first[a]"/>'
        '<testcase classname="tests.integration.fixture" name="test_second[b]"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert runner.passing_ids(path) == [
        "tests/integration/fixture.py::test_first[a]",
        "tests/integration/fixture.py::test_second[b]",
    ]


@pytest.mark.parametrize("tag", ("failure", "error", "skipped"))
def test_nonpassing_junit_cannot_be_promoted_by_zero_exit_or_zero_suite_counters(
    tmp_path: Path,
    tag: str,
) -> None:
    path = tmp_path / "not-passed.xml"
    path.write_text(
        '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.x" name="test_x"><' + tag + "/></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        runner.passing_ids(path)


@pytest.mark.parametrize("counter", ("errors", "failures", "skipped"))
def test_nonzero_suite_counter_denied_even_without_child_failure_node(
    tmp_path: Path,
    counter: str,
) -> None:
    path = tmp_path / "counter.xml"
    path.write_text(
        "<testsuites><testsuite " + counter + '="1">'
        '<testcase classname="tests.x" name="test_x"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        runner.passing_ids(path)


def test_empty_junit_is_not_passing_zero_cases(tmp_path: Path) -> None:
    path = tmp_path / "empty.xml"
    path.write_text('<testsuites><testsuite tests="0"/></testsuites>', encoding="utf-8")
    with pytest.raises(ValueError):
        runner.passing_ids(path)


def test_exclusive_canonical_receipt_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    runner.write_once(path, {"z": 1, "a": "\u03c3", "actual": False})
    body = path.read_bytes()
    assert json.loads(body) == {"a": "\u03c3", "actual": False, "z": 1}
    assert body == '{"a":"\u03c3","actual":false,"z":1}\n'.encode()
    with pytest.raises(FileExistsError):
        runner.write_once(path, {"actual": True})
    assert path.read_bytes() == body


def test_exact_new_direct_child_output_and_occupied_refusal(tmp_path: Path) -> None:
    path = tmp_path / "r015-writer-owned"
    assert runner.output_path(str(path), tmp_path) == path.resolve()
    path.mkdir()
    with pytest.raises(ValueError):
        runner.output_path(str(path), tmp_path)


@pytest.mark.parametrize(
    "child", ("other-prefix", "nested/r015-writer-owned", "../r015-writer-owned")
)
def test_foreign_prefix_nesting_or_ancestor_output_denied(tmp_path: Path, child: str) -> None:
    with pytest.raises(ValueError):
        runner.output_path(str(tmp_path / child), tmp_path)


def test_source_pin_cannot_hide_length_or_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "pinned.py"
    path.write_bytes(b"original\n")
    pin = {
        "path": str(path),
        "bytes": len(path.read_bytes()),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    assert runner.pins_match([pin])
    path.write_bytes(b"changed!\n")
    assert not runner.pins_match([pin])
