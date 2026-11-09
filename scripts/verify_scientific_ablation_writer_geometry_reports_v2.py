"""Independent declared-geometry physical score/capture chain and whole readback.

The historical C3-only ledger reader and first geometry reader stay unchanged.
This NEW reader checks the ENTIRE physical chain against separately retained
original source/geometry, not a candidate-supplied C5 exception or skipped guard.
Its own bytes are sealed before/end; the producer's original freeze is retained.
"""

import argparse
import math
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader
import verify_scientific_ablation_writer_geometry_reports as original
import verify_scientific_ablation_writer_report_reader as boundary
import verify_scientific_ablation_writer_reports as full
from verify_scientific_ablation_evaluation_io import binding, canonical, manifest
from verify_scientific_ablation_evaluation_ledger import genesis, load, valid_capture
from verify_scientific_ablation_observation import score_chain_directory

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/verify_scientific_ablation_writer_geometry_reports_v2.py"


def physical_chain(fixture: Path, expected: dict[str, Any]) -> dict[str, Any]:
    source = expected["source_metadata"]
    g = source["observed_budget"]["payload"]["design"]["payload"]["geometry"]
    path = score_chain_directory(fixture, source["evaluation_plan"]["sha256"])
    plan = manifest(path / "plan.json", "iris/scientific-ablation-evaluation-plan/v2")
    raw.require(raw.equal(plan, source["evaluation_plan"]), "Physical original score plan")
    payload = plan["payload"]
    generation, captured = payload["generation_plan"], payload["capture"]
    capture_roots = {
        valid_capture(directory): directory
        for store in fixture.rglob("scientific-ablation-capture-v1")
        for directory in store.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    }
    key = (generation["sha256"], captured["sha256"])
    raw.require(key in capture_roots, "ONE original complete capture source required")
    capture = capture_roots[key]
    raw.require(
        manifest(capture / "plan.json", "iris/scientific-ablation-capture-plan/v1") == generation
        and manifest(
            capture / "complete/complete.json", "iris/scientific-ablation-capture-complete/v1"
        )
        == captured,
        "Whole physical generation/complete anchors",
    )
    report = manifest(path / "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2")
    complete = manifest(
        path / "complete/complete.json", "iris/scientific-ablation-evaluation-complete/v2"
    )
    raw.require(
        report == source["evaluation"] and complete == source["evaluation_complete"],
        "BOTH original physical COMPLETE score anchors",
    )
    raw.require(
        report["payload"]["plan"] == generation
        and report["payload"]["capture"] == captured
        and path.name == plan["sha256"],
        "Original physical plan/path/report binding",
    )
    original_source = generation["payload"]["source"]["payload"]
    raw.require(
        original_source["selected_step"] == 1
        and original_source["terminal_step"] == 3
        and generation["payload"]["cycles"] == g["cycle_count"],
        "Actual best1/terminal3/SEPARATELY ORIGINAL declared cycles",
    )
    raw_pins = [
        binding(path / "plan.json"),
        binding(path / "complete/evaluation.json"),
        binding(path / "complete/complete.json"),
        binding(capture / "plan.json"),
        binding(capture / "complete/complete.json"),
    ]
    head = genesis(plan["sha256"])
    scores = []
    for index, expectation in enumerate(payload["rows"]):
        directory = path / f"score-{index:05d}"
        raw.require(
            {child.name for child in directory.iterdir()}
            == {"attempt.json", "row.json", "ack.json"},
            "Exact committed score members",
        )
        attempt = load(directory / "attempt.json")
        raw.require(
            attempt
            == dict(
                kind="iris-scientific-ablation-score-attempt-v2",
                plan_sha256=plan["sha256"],
                row_index=index,
                previous_ack_sha256=head,
                status="owned_before_scoring",
            )
            and canonical(attempt) + b"\n" == (directory / "attempt.json").read_bytes(),
            "Physical original owned scoring attempt",
        )
        row = manifest(directory / "row.json", "iris/scientific-ablation-evaluation-row/v2")
        ack = manifest(directory / "ack.json", "iris/scientific-ablation-evaluation-ack/v2")
        pin = binding(directory / "row.json")
        raw.require(
            ack["payload"]
            == dict(
                kind="iris-scientific-ablation-evaluation-ack-v2",
                status="acknowledged_score",
                plan_sha256=plan["sha256"],
                row_index=index,
                previous_ack_sha256=head,
                row_sha256=row["sha256"],
                row_file_sha256=pin["sha256"],
                row_file_bytes=pin["bytes"],
            ),
            "Every physical score SHA/file-length/previous-head acknowledgement",
        )
        actual_row = manifest(
            capture / f"row-{index:05d}/row.json", "iris/scientific-ablation-heldout-row/v1"
        )
        actual_ack = manifest(
            capture / f"row-{index:05d}/ack.json", "iris/scientific-ablation-capture-row-ack/v1"
        )
        score = row["payload"]["measurements"]
        raw.require(
            row["payload"]["plan_sha256"] == plan["sha256"] and score["row_index"] == index,
            "Original complete score job/index order",
        )
        raw.require(
            score["capture_row_sha256"] == actual_row["sha256"] == expectation["capture_row_sha256"]
            and score["capture_ack_sha256"]
            == actual_ack["sha256"]
            == expectation["capture_ack_sha256"],
            "Every original physical capture row/ack binding",
        )
        raw.require(
            score["target_nll_sum"]
            == actual_row["payload"]["target_nll_sum"]
            == expectation["target_nll_sum"]
            and score["target_token_count"] == actual_row["payload"]["target_token_count"],
            "Original EOS/count/NLL copied exactly",
        )
        output = actual_row["payload"]["output_text"].encode()
        raw.require(
            score["output_sha256"] == sha256(output).hexdigest() == expectation["output_sha256"]
            and score["output_byte_length"] == len(output),
            "Original captured output bytes",
        )
        if score["execution"] is not None:
            raw.require(
                score["execution"]["policy"] == payload["python_policy"]
                and score["execution"]["worker_sha256"] == payload["worker_sha256"],
                "Original isolated source-v2 policy/worker",
            )
        scores.append(score)
        raw_pins.extend(
            [
                pin,
                binding(directory / "attempt.json"),
                binding(directory / "ack.json"),
                binding(capture / f"row-{index:05d}/row.json"),
                binding(capture / f"row-{index:05d}/ack.json"),
            ]
        )
        head = ack["sha256"]
    raw.require(
        report["payload"]["rows"] == scores
        and report["payload"]["overall"]["target_nll_sum"]
        == math.fsum(row["target_nll_sum"] for row in scores),
        "Whole score prefix and exact NLL sum",
    )
    raw.require(
        complete["payload"]
        == dict(
            kind="iris-scientific-ablation-evaluation-complete-v2",
            status="complete_not_scientific_launch_approval",
            plan_sha256=plan["sha256"],
            capture_sha256=captured["sha256"],
            acknowledgement_head_sha256=head,
            evaluation_sha256=report["sha256"],
            example_count=len(scores),
        ),
        "Whole physical COMPLETE prefix, no omitted rows",
    )
    raw.require(
        not any(
            b'"' + name + b'"' in canonical(report) or b'"' + name + b'"' in canonical(plan)
            for name in (b"output_text", b"expected_response", b"hidden_facts", b"prompt")
        ),
        "No private data in public score artifacts",
    )
    return dict(
        path=str(path),
        geometry=g,
        rows=len(scores),
        report_sha256=report["sha256"],
        raw_bindings=raw_pins,
    )


def audit(before_path: Path, junit: Path, base: Path, runtime_exit: int) -> dict[str, Any]:
    raw.require(runtime_exit == 0, "Actually observed producer terminal exit0 required")
    own = boundary.binding(SELF)
    protected, originals = original.scope()
    before = boundary.load(before_path.relative_to(ROOT).as_posix())
    raw.require(
        before["kind"] == "iris-r015-original-geometry-writer-before-v1", "Original producer freeze"
    )
    raw.require(
        before["new_bindings"] == [boundary.binding(p) for p in original.NEW],
        "ENTIRE original producer/helper/v1-reader bytes unchanged",
    )
    raw.require(
        before["original_inputs"] == originals
        and before["protected_count"] == len(protected)
        and before["protected_sha256"] == sha256(raw.canonical(protected)).hexdigest(),
        "Entire inherited source/input commitment unchanged",
    )
    raw.require(
        before["inherited_byte_receipt"] == boundary.binding(original.BYTE_RECEIPT),
        "Inherited byte receipt",
    )
    ids = boundary.passing_ids(junit)
    raw.require(
        ids == before["nodeids"] and len(ids) == before["expected_cases"],
        "Exact original current JUnit",
    )
    files = original.inventory(base)
    reports = sorted(base.glob("test_*/public-writer-*.json"))
    raw.require(len(reports) == 2 * len(ids), "Both protocols ENTIRE original positive inventory")
    results, chains = [], []
    for path in reports:
        protocol = "message" if path.name.endswith("message.json") else "workspace"
        pfpath = path.parent / ("preflight-" + protocol + ".json")
        expected = boundary.load(pfpath.relative_to(ROOT).as_posix())
        body = path.read_bytes()
        data = reader.read_report(body, expected=expected, runtime=raw.CPU_STREAM)
        raw.require(
            reader.format_report(data, expected=expected, runtime=raw.CPU_STREAM) == body,
            "Original whole independent byte roundtrip",
        )
        full.physical_metadata(path, data["payload"])
        derived = original.metrics(data)
        witness = boundary.load(
            (path.parent / "geometry-witness.json").relative_to(ROOT).as_posix()
        )
        raw.require(
            derived == witness["metrics"][protocol], "Native-derived metrics, not authored flags"
        )
        results.append(
            dict(
                report=boundary.binding(path.relative_to(ROOT).as_posix()),
                preflight=boundary.binding(pfpath.relative_to(ROOT).as_posix()),
                protocol=protocol,
                geometry=witness["geometry"],
                metrics=derived,
            )
        )
        if protocol == "workspace":
            chains.append(physical_chain(path.parent, expected))
    if not before["pilot"]:
        wide = [r for r in results if r["geometry"]["modules"] == 64]
        raw.require(
            len(wide) == 2
            and all(
                r["metrics"]["effects"] == 12096 and r["metrics"]["pairs"] == 4032 for r in wide
            ),
            "Both legitimate real WHOLE wide reports",
        )
        raw.require(
            sum(r["metrics"]["nonwinning_donor_frames"] for r in results) > 0,
            "Actual nonwinning donor through PUBLIC source-owned service",
        )
    raw.require(original.scope() == (protected, originals), "Inherited scope stable before/end")
    raw.require(
        before["new_bindings"] == [boundary.binding(p) for p in original.NEW],
        "Original new scope stable",
    )
    raw.require(
        boundary.binding(SELF) == own and original.inventory(base) == files,
        "Reader and ALL witnesses stable",
    )
    forbidden = sorted(
        n
        for n in sys.modules
        if n.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}
    )
    raw.require(not forbidden, "No forbidden independent reader imports")
    return dict(
        kind="iris-r015-original-geometry-writer-readback-v2",
        passed=True,
        pilot=before["pilot"],
        scope=before["scope"],
        actual_runtime_cases=len(ids),
        actual_runtime_exit_code=runtime_exit,
        before=boundary.binding(before_path.relative_to(ROOT).as_posix()),
        junit=boundary.binding(junit.relative_to(ROOT).as_posix()),
        frozen_producer_bindings=before["new_bindings"],
        additional_reader_pin=own,
        stable_inherited_protected_files=len(protected),
        stable_original_input_files=len(originals),
        stable_positive_files=len(files),
        positive_inventory_sha256=sha256(raw.canonical(files)).hexdigest(),
        physical_chains=len(chains),
        physical_chains_sha256=sha256(raw.canonical(chains)).hexdigest(),
        whole_public_reports=results,
        forbidden_imports=forbidden,
        additional_teacher_forwards=0,
        original_c3_ledger_and_v1_readers_unchanged=True,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime-exit", type=int, required=True)
    args = parser.parse_args()
    value = audit(
        args.snapshot.resolve(), args.junit.resolve(), args.base.resolve(), args.runtime_exit
    )
    print(raw.canonical(value).decode())


if __name__ == "__main__":
    main()
