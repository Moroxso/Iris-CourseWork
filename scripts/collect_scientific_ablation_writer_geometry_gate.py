"""Actual exact new scope collection and immutable source/input bindings, no model."""

import argparse
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_usage_epoch as epoch
import verify_scientific_ablation_writer_geometry_reports as geometry
import verify_scientific_ablation_writer_report_reader as boundary
from run_scientific_ablation_writer_geometry_gate import (
    SCOPES,
    output_path,
    scope_config,
    scope_files,
    write_once,
)
from scientific_ablation_writer_affected_scope import modules as affected_modules

ROOT = Path(__file__).resolve().parents[1]
EXTRA = (
    "tests/integration/test_scientific_ablation_writer_nonwinning_report.py",
    "scripts/run_scientific_ablation_writer_geometry_gate.py",
    "scripts/collect_scientific_ablation_writer_geometry_gate.py",
    "scripts/verify_scientific_ablation_writer_geometry_gate.py",
    "scripts/scientific_ablation_writer_usage_epoch.py",
    "tests/unit/test_workspace_usage_single_slot.py",
    "scripts/scientific_ablation_writer_affected_scope.py",
    "scripts/scientific_ablation_writer_affected_raw.py",
)
V2 = "scripts/verify_scientific_ablation_writer_geometry_reports_v2.py"


def snapshot(mode: str) -> dict[str, Any]:
    protected, originals, delta = epoch.scope()
    current = [boundary.binding(p) for p in geometry.NEW]
    extra = [boundary.binding(p) for p in EXTRA]
    v2 = boundary.binding(V2)
    modules, count = scope_config(mode)
    module_pins = [boundary.binding(p) for p in scope_files(modules)]
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
            *modules,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    raw.require(
        collected.returncode == 0,
        "Actual collection failed:\n" + collected.stdout + collected.stderr,
    )
    ids = [line.strip() for line in collected.stdout.splitlines() if "::test_" in line]
    raw.require(len(ids) == len(set(ids)) == count, "Entire exact declared current runtime scope")
    if mode == "affected-full":
        declared, inherited_ids = affected_modules()
        raw.require(
            modules == declared
            and [node for node in ids if node in set(inherited_ids)] == inherited_ids,
            "Every inherited1808ID retained in complete affected union",
        )
    raw.require(
        epoch.scope() == (protected, originals, delta),
        "Inherited source/input stability during collection",
    )
    raw.require(
        current == [boundary.binding(p) for p in geometry.NEW]
        and extra == [boundary.binding(p) for p in EXTRA]
        and v2 == boundary.binding(V2),
        "New frozen source stability during collection",
    )
    raw.require(
        module_pins == [boundary.binding(p) for p in scope_files(modules)],
        "ALL affected test modules stable",
    )
    return dict(
        kind="iris-r015-original-geometry-writer-before-v1",
        gate_scope=mode,
        scope=mode,
        pilot=mode in {"nonwinning-pilot", "wide-pilot"},
        modules=modules,
        affected_module_bindings=module_pins,
        expected_cases=count,
        nodeids=ids,
        new_bindings=current,
        extra_producer_bindings=extra,
        geometry_v2_reader_pin=v2,
        intentional_source_delta=delta,
        inherited_byte_receipt=boundary.binding(geometry.BYTE_RECEIPT),
        protected_count=len(protected),
        protected_sha256=sha256(raw.canonical(protected)).hexdigest(),
        original_inputs=originals,
        actual_collection_exit_code=collected.returncode,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=[*SCOPES, "affected-full"], required=True)
    parser.add_argument("--output", help="Exclusive NEW operational before artifact")
    args = parser.parse_args()
    destination = output_path(args.output, ROOT / "artifacts/operational") if args.output else None
    value = snapshot(args.scope)
    if destination is None:
        print(raw.canonical(value).decode())
    else:
        write_once(destination, value)
        raw.require(
            boundary.load(destination.relative_to(ROOT).as_posix()) == value,
            "Whole exclusive saved before artifact readback",
        )
        print(
            raw.canonical(
                dict(
                    before=boundary.binding(destination.relative_to(ROOT).as_posix()),
                    modules=len(value["modules"]),
                    actual_cases=len(value["nodeids"]),
                    actual_collection_exit_code=value["actual_collection_exit_code"],
                    runtime_not_run=True,
                    full_iris_gw_01_complete=False,
                )
            ).decode()
        )


if __name__ == "__main__":
    main()
