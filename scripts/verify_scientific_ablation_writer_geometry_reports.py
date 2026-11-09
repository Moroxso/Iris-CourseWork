"""Fresh original-geometry public report freeze and independent raw readback.

No model/project/test imports or execution in readback. Before mode uses ONLY
trusted pytest collection. Both protocols retain full original physical source,
separate before-BUSY expectations and actual complete public artifacts. A pilot
is never promoted to the entire37case geometry matrix or full Iris-GW-0.1.
"""

import argparse
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_writer_report_reader as boundary
import verify_scientific_ablation_writer_reports as full

ROOT = Path(__file__).resolve().parents[1]
BYTE_RECEIPT = "artifacts/operational/r015-independent-byte-readback-20261014-run01.json"
BYTE_RECEIPT_SHA = "1245dbb23d5ada5ba12d4e1ff21b56e312c5605c668b317fe117647c5f390843"
TEST = "tests/integration/test_scientific_ablation_writer_geometry_reports.py"
NEW = (
    "tests/integration/scientific_ablation_writer_geometry_fixture.py",
    TEST,
    "scripts/verify_scientific_ablation_writer_geometry_reports.py",
)


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protected, originals = boundary.original_scope()
    accepted = boundary.load(BYTE_RECEIPT, BYTE_RECEIPT_SHA)
    protected = [*protected, *accepted["current_bindings"]]
    raw.require(
        accepted["passed"] is True and len(protected) == 620, "Actual inherited byte boundary"
    )
    for pin in protected:
        raw.require(boundary.binding(pin["path"]) == pin, "Accepted source changed: " + pin["path"])
    return protected, originals


def snapshot(pilot: bool) -> dict[str, Any]:
    protected, originals = scope()
    own = [boundary.binding(path) for path in NEW]
    targets = (
        [
            TEST
            + "::test_all12_both_public_original_occupied_expiry_and_nonsurvival_geometry["
            + scenario
            + "-enabled-enabled-learned]"
            for scenario in ("occupied", "expiry", "nonsurvival")
        ]
        if pilot
        else [TEST]
    )
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
            *targets,
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
    raw.require(len(ids) == len(set(ids)) == (3 if pilot else 37), "Exact whole current scope")
    raw.require(scope() == (protected, originals), "Inherited source unchanged during collection")
    raw.require(
        own == [boundary.binding(path) for path in NEW], "New source unchanged during collection"
    )
    return dict(
        kind="iris-r015-original-geometry-writer-before-v1",
        scope="three-original-geometry-pilots" if pilot else "all37-original-geometry-cases",
        pilot=pilot,
        expected_cases=len(ids),
        nodeids=ids,
        new_bindings=own,
        inherited_byte_receipt=boundary.binding(BYTE_RECEIPT),
        protected_count=len(protected),
        protected_sha256=sha256(raw.canonical(protected)).hexdigest(),
        original_inputs=originals,
        actual_collection_exit_code=collected.returncode,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def metrics(data: dict[str, Any]) -> dict[str, int]:
    p = data["payload"]
    design = p["source_metadata"]["observed_budget"]["payload"]["design"]["payload"]
    g = design["geometry"]
    frames = p["controlled_passes"]
    empty = {f["row_index"]: f for f in frames["empty"]}
    occupied = nonwinning = expired = 0
    for frame in frames["trials"].get("block", []):
        cycle, slot = frame["write_cycle"] - 1, frame["slot"]
        o = frame["output"]
        before = o["observations"][cycle]["before_write"]
        if before["occupancy"]["values"][0][slot]:
            occupied += 1
            raw.require(
                raw.equal(before, o["controlled"]["states"][cycle]),
                "ALL7 actual occupied BLOCK equals REAL prewrite record",
            )
            raw.require(
                not raw.equal(before, o["observations"][cycle]["native_written"]),
                "Occupied BLOCK is not a no-op writer",
            )
    for frame in frames["trials"].get("donor", []):
        donor = empty[frame["donor_row_index"]]["output"]["controlled"]
        nonwinning += (
            frame["writer"]
            not in donor["trace"]["cycles"][frame["write_cycle"] - 1]["selected_modules"]
        )
    for frame in frames["empty"]:
        output = frame["output"]
        for cycle in range(1, g["cycle_count"]):
            previous = output["controlled"]["states"][cycle - 1]
            before = output["observations"][cycle]["before_write"]
            for slot in range(g["num_slots"]):
                expired += (
                    previous["occupancy"]["values"][0][slot]
                    and previous["ttl"]["values"][0][slot] <= 1.0
                    and not before["occupancy"]["values"][0][slot]
                )
    return dict(
        actual_teacher_executions=p["actual_teacher_executions"],
        effects=len(p["effects"]),
        pairs=len(p["statistics"]["pairs"]),
        occupied_block_frames=occupied,
        nonwinning_donor_frames=nonwinning,
        actual_expired_records=expired,
        overwrite_effects=sum(e["overwrote_existing_record"] for e in p["effects"]),
        nonsurviving_effects=sum(not e["survives_to_next_read"] for e in p["effects"]),
    )


def inventory(base: Path) -> list[dict[str, Any]]:
    return [
        boundary.binding(path.relative_to(ROOT).as_posix())
        for path in sorted(base.rglob("*"))
        if path.is_file()
    ]


def audit(before_path: Path, junit: Path, base: Path, runtime_exit: int) -> dict[str, Any]:
    raw.require(runtime_exit == 0, "Actually observed producer terminal exit0 required")
    protected, originals = scope()
    before = boundary.load(before_path.relative_to(ROOT).as_posix())
    raw.require(
        before["kind"] == "iris-r015-original-geometry-writer-before-v1",
        "Distinct original geometry before",
    )
    raw.require(before["new_bindings"] == [boundary.binding(p) for p in NEW], "Frozen new code")
    raw.require(
        before["original_inputs"] == originals
        and before["protected_count"] == len(protected)
        and before["protected_sha256"] == sha256(raw.canonical(protected)).hexdigest(),
        "Entire inherited frozen source and original input inventory",
    )
    raw.require(
        before["inherited_byte_receipt"] == boundary.binding(BYTE_RECEIPT),
        "Immutable inherited accepted byte receipt",
    )
    ids = boundary.passing_ids(junit)
    raw.require(
        ids == before["nodeids"] and len(ids) == before["expected_cases"],
        "Exact ordered actual current runtime JUnit, no sum of pilots",
    )
    files = inventory(base)
    reports = sorted(base.glob("test_*/public-writer-*.json"))
    raw.require(len(reports) == 2 * len(ids), "Entire both-protocol positive public inventory")
    results, chains = [], []
    for path in reports:
        protocol = "message" if path.name.endswith("message.json") else "workspace"
        expected = boundary.load(
            (path.parent / ("preflight-" + protocol + ".json")).relative_to(ROOT).as_posix()
        )
        body = path.read_bytes()
        data = reader.read_report(body, expected=expected, runtime=raw.CPU_STREAM)
        raw.require(
            reader.format_report(data, expected=expected, runtime=raw.CPU_STREAM) == body,
            "Whole independent original-geometry byte roundtrip",
        )
        full.physical_metadata(path, data["payload"])
        derived = metrics(data)
        witness = boundary.load(
            (path.parent / "geometry-witness.json").relative_to(ROOT).as_posix()
        )
        raw.require(
            derived == witness["metrics"][protocol], "Native-derived metrics, not authored flags"
        )
        results.append(
            dict(
                report=boundary.binding(path.relative_to(ROOT).as_posix()),
                preflight=boundary.binding(
                    (path.parent / ("preflight-" + protocol + ".json")).relative_to(ROOT).as_posix()
                ),
                protocol=protocol,
                geometry=witness["geometry"],
                metrics=derived,
            )
        )
        if protocol == "workspace":
            chains.append(
                sources.physical_chain(
                    path.parent, expected["source_metadata"]["evaluation_plan"]["sha256"]
                )
            )
    if not before["pilot"]:
        wide = [r for r in results if r["geometry"]["modules"] == 64]
        raw.require(
            len(wide) == 2
            and all(
                r["metrics"]["effects"] == 12096 and r["metrics"]["pairs"] == 4032 for r in wide
            ),
            "Both real WHOLE wide reports",
        )
        raw.require(
            sum(r["metrics"]["nonwinning_donor_frames"] for r in results) > 0,
            "Actual nonwinning donor content through PUBLIC source-owned service",
        )
    raw.require(scope() == (protected, originals), "Inherited source/input stability after audit")
    raw.require(before["new_bindings"] == [boundary.binding(p) for p in NEW], "New code stability")
    raw.require(inventory(base) == files, "ALL positive raw/source files stable before/end")
    forbidden = sorted(
        n
        for n in sys.modules
        if n.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}
    )
    raw.require(not forbidden, "No forbidden independent reader imports")
    return dict(
        kind="iris-r015-original-geometry-writer-readback-v1",
        passed=True,
        pilot=before["pilot"],
        scope=before["scope"],
        actual_runtime_cases=len(ids),
        actual_runtime_exit_code=runtime_exit,
        before=boundary.binding(before_path.relative_to(ROOT).as_posix()),
        junit=boundary.binding(junit.relative_to(ROOT).as_posix()),
        new_bindings=before["new_bindings"],
        stable_inherited_protected_files=len(protected),
        stable_original_input_files=len(originals),
        stable_positive_files=len(files),
        positive_inventory_sha256=sha256(raw.canonical(files)).hexdigest(),
        physical_chains=len(chains),
        physical_chains_sha256=sha256(raw.canonical(chains)).hexdigest(),
        whole_public_reports=results,
        forbidden_imports=forbidden,
        additional_teacher_forwards=0,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--runtime-exit", type=int)
    args = parser.parse_args()
    if args.before:
        value = snapshot(args.pilot)
    else:
        raw.require(
            args.snapshot is not None
            and args.junit is not None
            and args.base is not None
            and args.runtime_exit is not None,
            "Exact terminal inputs",
        )
        value = audit(
            args.snapshot.resolve(), args.junit.resolve(), args.base.resolve(), args.runtime_exit
        )
    print(raw.canonical(value).decode())


if __name__ == "__main__":
    main()
