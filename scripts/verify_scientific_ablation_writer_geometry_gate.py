"""Fresh explicit usage epoch/owned terminal/whole physical and semantic readback.

Historical v1/v2 observers remain immutable. This reader owns the current
single-source delta, not a retrofit of old source/runtime verdicts.
"""

import argparse
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_affected_raw as affected
import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader
import scientific_ablation_writer_usage_epoch as epoch
import verify_scientific_ablation_writer_geometry_reports_v2 as geometry
import verify_scientific_ablation_writer_report_reader as boundary
import verify_scientific_ablation_writer_reports as full
from collect_scientific_ablation_writer_geometry_gate import EXTRA, V2
from run_scientific_ablation_writer_geometry_gate import scope_config, scope_files
from scientific_ablation_writer_affected_scope import modules as affected_modules
from verify_scientific_ablation_writer_geometry_reports import NEW, inventory, metrics

ROOT = Path(__file__).resolve().parents[1]


def audit(before_path: Path, junit: Path, base: Path, terminal: Path) -> dict[str, Any]:
    before = boundary.load(before_path.relative_to(ROOT).as_posix())
    modules, count = scope_config(before["gate_scope"])
    module_pins = [boundary.binding(p) for p in scope_files(modules)]
    raw.require(
        before["modules"] == modules
        and before["expected_cases"] == count
        and before["affected_module_bindings"] == module_pins,
        "Exact fixed scope and ALL current module bindings",
    )
    if before["gate_scope"] == "affected-full":
        inherited_ids = affected_modules()[1]
        raw.require(
            [node for node in before["nodeids"] if node in set(inherited_ids)] == inherited_ids,
            "ALL inherited1808 ordered collection IDs retained, no clipping",
        )
    protected, originals, delta = epoch.scope()
    extra = [boundary.binding(p) for p in EXTRA]
    raw.require(
        before["extra_producer_bindings"] == extra
        and before["geometry_v2_reader_pin"] == boundary.binding(V2),
        "Entire additional producer/runner/collector/reader and immutable v2 pins",
    )
    raw.require(
        before["new_bindings"] == [boundary.binding(p) for p in NEW]
        and before["original_inputs"] == originals
        and before["intentional_source_delta"] == delta
        and before["protected_count"] == len(protected)
        and before["protected_sha256"] == sha256(raw.canonical(protected)).hexdigest(),
        "Current ONE-source corrective epoch and entire other inherited inventory",
    )
    end = boundary.load(terminal.relative_to(ROOT).as_posix())
    raw.require(
        end["kind"] == "iris-r015-owned-writer-gate-exit-v1"
        and end["actual_child_exit_code"] == 0
        and end["actual_runtime_passed"] is True
        and end["frozen_bindings_stable"] is True
        and end["exact_ordered_junit_ids"] is True,
        "Actual successful owned terminal, not a launch/partial file",
    )
    raw.require(
        end["before_sha256_after"]
        == boundary.binding(before_path.relative_to(ROOT).as_posix())["sha256"]
        and end["junit_sha256"] == boundary.binding(junit.relative_to(ROOT).as_posix())["sha256"],
        "Actual terminal binds exact original freeze and JUnit bytes",
    )
    prefix = str(terminal)[: -len("exit.json")]
    launch_path = Path(prefix + "launch.json")
    launch = boundary.load(launch_path.relative_to(ROOT).as_posix())
    raw.require(
        boundary.binding(launch_path.relative_to(ROOT).as_posix())["sha256"]
        == end["launch_sha256"],
        "Actual owned launch/terminal binding",
    )
    for name in ("stdout", "stderr"):
        path = Path(prefix + name + ".log")
        raw.require(
            boundary.binding(path.relative_to(ROOT).as_posix())["sha256"] == end[name + "_sha256"],
            "Actual child complete log binding",
        )
    runner = "scripts/run_scientific_ablation_writer_geometry_gate.py"
    raw.require(
        launch["runner_sha256"] == end["runner_sha256_after"] == boundary.binding(runner)["sha256"],
        "Actual frozen owned supervisor",
    )
    ids = boundary.passing_ids(junit)
    raw.require(
        ids == before["nodeids"] and len(ids) == len(set(ids)) == before["expected_cases"],
        "Exact ordered ENTIRE current runtime, no pilot sum",
    )
    raw.require(
        launch["gate_scope"] == before["gate_scope"]
        and launch["expected_cases"] == count
        and launch["before_path"] == str(before_path)
        and launch["before_sha256"] == end["before_sha256_after"]
        and launch["base"] == str(base)
        and launch["junit"] == str(junit)
        and end["actual_junit_cases"] == end["validated_passing_id_count"] == count,
        "Actual launch/terminal outputs and scope match entire authenticated runtime",
    )
    files = inventory(base)
    inherited = affected.audit(base) if before["gate_scope"] == "affected-full" else None
    reports = sorted(
        path
        for path in base.glob("test_*/public-writer-*.json")
        if (path.parent / "geometry-witness.json").is_file()
    )
    geometry_cases = {
        "nonwinning-pilot": 1,
        "wide-pilot": 1,
        "original-geometry-full": 37,
        "combined-full": 38,
        "affected-full": 38,
    }[before["gate_scope"]]
    raw.require(
        len(reports) == 2 * geometry_cases
        and len({p.parent for p in reports}) == geometry_cases
        and all(
            {p.name for p in reports if p.parent == parent}
            == {"public-writer-message.json", "public-writer-workspace.json"}
            for parent in {p.parent for p in reports}
        ),
        "Entire declared both-protocol geometry positive inventory",
    )
    inherited_paths = (
        {ROOT / r["report"]["path"] for r in inherited["public_reports"]}
        if inherited is not None
        else set()
    )
    raw.require(
        set(base.glob("test_*/public-writer-*.json")) == set(reports) | inherited_paths,
        "Every whole PUBLIC report assigned to an authenticated family",
    )
    results, chains = [], []
    for path in reports:
        protocol = "message" if path.name.endswith("message.json") else "workspace"
        pfpath = path.parent / ("preflight-" + protocol + ".json")
        expected = boundary.load(pfpath.relative_to(ROOT).as_posix())
        body = path.read_bytes()
        data = reader.read_report(body, expected=expected, runtime=raw.CPU_STREAM)
        raw.require(
            reader.format_report(data, expected=expected, runtime=raw.CPU_STREAM) == body,
            "Whole source-bound byte read/format roundtrip",
        )
        full.physical_metadata(path, data["payload"])
        derived = metrics(data)
        witness = boundary.load(
            (path.parent / "geometry-witness.json").relative_to(ROOT).as_posix()
        )
        actual_geometry = expected["source_metadata"]["observed_budget"]["payload"]["design"][
            "payload"
        ]["geometry"]
        raw.require(
            witness["geometry"] == affected.original_geometry(path, expected)
            and all(
                witness["geometry"][name] == actual_geometry[field]
                for name, field in (
                    ("modules", "num_modules"),
                    ("slots", "num_slots"),
                    ("width", "state_dim"),
                    ("cycles", "cycle_count"),
                    ("writers", "writers_per_cycle"),
                )
            ),
            "Witness geometry equals separately original complete source geometry",
        )
        raw.require(derived == witness["metrics"][protocol], "Native-derived corner metrics")
        results.append(
            dict(
                report=boundary.binding(path.relative_to(ROOT).as_posix()),
                preflight=boundary.binding(pfpath.relative_to(ROOT).as_posix()),
                protocol=protocol,
                geometry=witness["geometry"],
                original_mode=affected.mode(expected),
                metrics=derived,
            )
        )
        if protocol == "workspace":
            chains.append(geometry.physical_chain(path.parent, expected))
    if before["gate_scope"] != "wide-pilot":
        raw.require(
            sum(r["metrics"]["nonwinning_donor_frames"] for r in results) > 0,
            "Actual nonwinning original paired proposal donor required",
        )
    if not before["pilot"]:
        affected.geometry_matrix(
            results, includes_nonwinning=before["gate_scope"] in {"combined-full", "affected-full"}
        )
    if not before["pilot"] or before["gate_scope"] == "wide-pilot":
        wide = [r for r in results if r["geometry"]["modules"] == 64]
        raw.require(
            len(wide) == 2
            and all(
                r["metrics"]["effects"] == 12096
                and r["metrics"]["pairs"] == 4032
                and r["metrics"]["actual_teacher_executions"] == 768
                for r in wide
            ),
            "Both legitimate actual WHOLE wide reports",
        )
    raw.require(
        before["extra_producer_bindings"] == [boundary.binding(p) for p in EXTRA]
        and before["geometry_v2_reader_pin"] == boundary.binding(V2),
        "Additional source stable before/end",
    )
    raw.require(
        epoch.scope() == (protected, originals, delta) and inventory(base) == files,
        "Entire current epoch/raw positive inventory stable before/end",
    )
    raw.require(
        before["new_bindings"] == [boundary.binding(p) for p in NEW],
        "Original geometry source stable",
    )
    raw.require(
        module_pins == [boundary.binding(p) for p in scope_files(modules)], "ALL module pins stable"
    )
    forbidden = sorted(
        n
        for n in sys.modules
        if n.split(".")[0] in {"iris", "torch", "pydantic", "tests", "pytest"}
    )
    raw.require(not forbidden, "No forbidden readback imports")
    return dict(
        kind="iris-r015-owned-original-geometry-writer-readback-v2",
        passed=True,
        gate_scope=before["gate_scope"],
        pilot=before["pilot"],
        actual_runtime_cases=len(ids),
        intentional_source_delta=delta,
        current_protected_source_files=len(protected),
        unchanged_other_inherited_files=619,
        stable_original_inputs=len(originals),
        before=boundary.binding(before_path.relative_to(ROOT).as_posix()),
        junit=boundary.binding(junit.relative_to(ROOT).as_posix()),
        owned_terminal=boundary.binding(terminal.relative_to(ROOT).as_posix()),
        new_bindings=before["new_bindings"],
        extra_frozen_bindings=extra,
        immutable_v2_reader_pin=before["geometry_v2_reader_pin"],
        stable_positive_files=len(files),
        positive_inventory_sha256=sha256(raw.canonical(files)).hexdigest(),
        physical_chains=len(chains),
        physical_chains_sha256=sha256(raw.canonical(chains)).hexdigest(),
        whole_public_reports=results,
        inherited_family_readback=inherited,
        total_whole_public_reports=len(results) + len(inherited_paths),
        additional_teacher_forwards=0,
        forbidden_imports=forbidden,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("snapshot", "junit", "base", "terminal"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.snapshot.resolve(), args.junit.resolve(), args.base.resolve(), args.terminal.resolve()
    )
    print(raw.canonical(value).decode())


if __name__ == "__main__":
    main()
