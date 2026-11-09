"""Isolated byte-component freeze/readback, not a new full1808 or science verdict.

Before collection is a trusted pytest subprocess. Readback itself forbids model,
project and test imports and never reruns collection, tests or a model. ORIGINAL
physical-source authentication remains in the immutable accepted raw receipt;
this component independently binds every saved preflight/report to those pins.
"""

import argparse
import importlib.abc
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader


class DenyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}:
            raise AssertionError("Forbidden independent reader import: " + fullname)
        return None


sys.meta_path.insert(0, DenyImports())
raw.require(
    not any(
        name.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}
        for name in sys.modules
    ),
    "No forbidden imports during initial component load",
)

ROOT = Path(__file__).resolve().parents[1]
OLD_BEFORE = "artifacts/operational/r015-writer-report-before-20261014-run01.json"
OLD_BEFORE_SHA = "880409b2dc28cb9b239ac503351f01c3c7bda7fec7a3124df090c6f9bff70144"
OLD_RAW = "artifacts/operational/r015-writer-report-readback-20261014-run01.json"
OLD_RAW_SHA = "f3062411b304e72479221777c210b7279f2aac2bffec61af1e2b4de004f994db"
TEST = "tests/unit/test_scientific_ablation_writer_independent_reader.py"
NEW = (
    "scripts/scientific_ablation_writer_report_reader.py",
    TEST,
    "scripts/verify_scientific_ablation_writer_report_reader.py",
)


def binding(path: str) -> dict[str, Any]:
    body = (ROOT / path).read_bytes()
    return dict(path=path, bytes=len(body), sha256=sha256(body).hexdigest())


def load(path: str, checksum: str | None = None) -> dict[str, Any]:
    body = (ROOT / path).read_bytes()
    if checksum is not None:
        raw.require(sha256(body).hexdigest() == checksum, "Immutable trusted receipt changed")
    value = json.loads(body)
    raw.require(type(value) is dict, "Exact receipt root")
    assert isinstance(value, dict)
    raw.require(raw.canonical(value) + b"\n" == body, "Exact canonical receipt bytes")
    return value


def original_scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before, accepted = load(OLD_BEFORE, OLD_BEFORE_SHA), load(OLD_RAW, OLD_RAW_SHA)
    protected = [
        *before["inherited_bindings"],
        *before["inherited_auditors"],
        *before["new_bindings"],
        before["collector"],
        *before["original_native_files"],
        *accepted["reader_pins"],
    ]
    raw.require(len(protected) == 617, "Entire accepted frozen code/native inventory")
    raw.require(accepted["passed"] is True, "Actual accepted inherited raw receipt")
    inputs = [
        pin
        for entry in accepted["public_reports"]
        for pin in (entry["report"], entry["separately_retained_preflight"])
    ]
    raw.require(
        len(inputs) == 48 and len({p["path"] for p in inputs}) == 48,
        "All24 public reports and DISTINCT separately original preflights",
    )
    for pin in [*protected, *inputs]:
        raw.require(
            binding(pin["path"]) == pin, "Accepted source/input pin changed: " + pin["path"]
        )
    return protected, inputs


def snapshot() -> dict[str, Any]:
    protected, inputs = original_scope()
    own = [binding(path) for path in NEW]
    collected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            TEST,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    raw.require(
        collected.returncode == 0,
        "Actual collection failed:\n" + collected.stdout + collected.stderr,
    )
    ids = [line.strip() for line in collected.stdout.splitlines() if "::test_" in line]
    raw.require(len(ids) == len(set(ids)) and len(ids) > 100, "Exact new component collection")
    raw.require(
        original_scope() == (protected, inputs), "Inherited inputs changed during collection"
    )
    raw.require([binding(path) for path in NEW] == own, "New code changed during collection")
    return dict(
        kind="iris-r015-independent-byte-reader-before-v1",
        scope="byte-component-only",
        modules=[TEST],
        nodeids=ids,
        expected_cases=len(ids),
        new_bindings=own,
        old_before=binding(OLD_BEFORE),
        old_raw=binding(OLD_RAW),
        inherited_protected_count=len(protected),
        inherited_protected_sha256=sha256(raw.canonical(protected)).hexdigest(),
        separately_trusted_inputs=inputs,
        actual_collection_exit_code=collected.returncode,
        full1808_not_rerun=True,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def passing_ids(path: Path) -> list[str]:
    root = ET.fromstring(path.read_bytes())
    cases = list(root.iter("testcase"))
    raw.require(bool(cases), "Actual component JUnit required")
    raw.require(
        not list(root.iter("failure"))
        and not list(root.iter("error"))
        and not list(root.iter("skipped")),
        "No failure/error/skip",
    )
    for suite in root.iter("testsuite"):
        raw.require(
            all(int(suite.attrib.get(key, "0")) == 0 for key in ("errors", "failures", "skipped")),
            "Actual zero failure counters",
        )
    return [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]


def audit(before_path: Path, junit: Path, runtime_exit: int) -> dict[str, Any]:
    raw.require(runtime_exit == 0, "Actual producer terminal exit0 required")
    protected, inputs = original_scope()
    before = load(str(before_path.relative_to(ROOT)).replace("\\", "/"))
    raw.require(
        before["kind"] == "iris-r015-independent-byte-reader-before-v1",
        "Distinct byte-component before",
    )
    raw.require(before["new_bindings"] == [binding(path) for path in NEW], "Frozen new code")
    raw.require(before["separately_trusted_inputs"] == inputs, "Frozen all original input pairs")
    raw.require(
        before["inherited_protected_count"] == len(protected)
        and before["inherited_protected_sha256"] == sha256(raw.canonical(protected)).hexdigest(),
        "Frozen entire inherited inventory commitment",
    )
    raw.require(
        before["old_before"] == binding(OLD_BEFORE) and before["old_raw"] == binding(OLD_RAW),
        "Frozen inherited receipts",
    )
    ids = passing_ids(junit)
    raw.require(
        ids == before["nodeids"] and len(ids) == len(set(ids)) == before["expected_cases"],
        "Exact ordered component JUnit, not a sum of development pilots",
    )
    accepted = load(OLD_RAW, OLD_RAW_SHA)
    positives = []
    for entry in accepted["public_reports"]:
        content = (ROOT / entry["report"]["path"]).read_bytes()
        expected = load(entry["separately_retained_preflight"]["path"])
        value = reader.read_report(content, expected=expected, runtime=raw.CPU_STREAM)
        raw.require(
            reader.format_report(value, expected=expected, runtime=raw.CPU_STREAM) == content,
            "Entire byte-identical read/format roundtrip",
        )
        positives.append(
            dict(
                protocol=entry["protocol"],
                report=entry["report"],
                separately_retained_preflight=entry["separately_retained_preflight"],
            )
        )
    expected = load(positives[0]["separately_retained_preflight"]["path"])
    denied = []
    for name, content in (
        ("empty", b""),
        ("duplicate-root", b'{"x":0,"x":1}\n'),
        ("duplicate-nested", b'{"x":{"y":0,"y":1}}\n'),
        ("nonfinite", b'{"x":NaN}\n'),
        ("overflowing-float", b'{"x":1e1000}\n'),
        ("non-utf8", b'{"x":"\xff"}\n'),
        ("nonroot", b"[]\n"),
        ("depth", b"[" * 41 + b"0" + b"]" * 41),
        ("original-bound", b" " * (expected["report_byte_preflight_bound"] + 1)),
    ):
        try:
            reader.read_report(content, expected=expected, runtime=raw.CPU_STREAM)
        except ValueError:
            denied.append(name)
        else:
            raise ValueError("Independent byte corruption accepted: " + name)
    raw.require(
        original_scope() == (protected, inputs), "Entire inherited scope unchanged after audit"
    )
    raw.require(
        before["new_bindings"] == [binding(path) for path in NEW], "New code stable after audit"
    )
    forbidden = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}
    )
    raw.require(not forbidden, "Readback made forbidden imports")
    return dict(
        kind="iris-r015-independent-byte-reader-readback-v1",
        scope="byte-component-only",
        passed=True,
        actual_runtime_exit_code=runtime_exit,
        actual_runtime_cases=len(ids),
        junit=binding(str(junit.relative_to(ROOT)).replace("\\", "/")),
        before=binding(str(before_path.relative_to(ROOT)).replace("\\", "/")),
        current_bindings=before["new_bindings"],
        stable_inherited_protected_files=len(protected),
        stable_original_inputs=len(inputs),
        all24_public_roundtrips=positives,
        byte_corruption_denials=denied,
        forbidden_imports=forbidden,
        additional_teacher_forwards=0,
        original_physical_source_authentication="inherited-immutable-full-raw-receipt-not-replayed",
        whole_wide_public_occupied_delivery_acceptance=False,
        full1808_not_rerun=True,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--runtime-exit", type=int)
    args = parser.parse_args()
    if args.before:
        value = snapshot()
    else:
        raw.require(
            args.snapshot is not None and args.junit is not None and args.runtime_exit is not None,
            "Explicit terminal evidence required",
        )
        value = audit(args.snapshot.resolve(), args.junit.resolve(), args.runtime_exit)
    print(raw.canonical(value).decode())


if __name__ == "__main__":
    main()
