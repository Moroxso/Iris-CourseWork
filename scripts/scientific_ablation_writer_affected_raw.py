"""Whole inherited positive families in a NEW affected-run output tree.

Pure readers only: no old main/audit, mutable global BASE override, project/model
imports, additional forwards or promotion of historical runtime verdicts. The
caller authenticates the current source epoch, actual terminal and exact IDs.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_design as designs
import verify_scientific_ablation_causal_observed_budget as budgets
import verify_scientific_ablation_causal_report_layout as layouts
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_controls as kernel
import verify_scientific_ablation_native_writer_effects as legacy
import verify_scientific_ablation_observation as native
import verify_scientific_ablation_observed_controls as stages
import verify_scientific_ablation_writer_reports as full
import verify_scientific_ablation_writer_statistics as statistics
from verify_scientific_ablation_evaluation_io import manifest

COUNTS = {
    "preflights": 24,
    "old_stages": 24,
    "new_stages": 24,
    "observations": 12,
    "old_controls": 24,
    "kernels": 168,
    "overwrites": 1,
    "budgets": 28,
    "encodings": 2,
    "statistics": 25,
    "private": 24,
    "public": 24,
}


def families(base: Path) -> dict[str, list[Path]]:
    """Original whole families; added geometry reports are a distinct namespace."""
    result = {
        "preflights": sorted(base.glob("test_all12_complete_original*/preflight-*.json")),
        "old_stages": sorted(base.glob("test_all12_original_best1_both*/stage-*.json")),
        "new_stages": sorted(base.glob("test_all12_complete_original*/stage-*.json")),
        "observations": sorted(base.glob("test_all12_actual_original_bes*/observations.json")),
        "old_controls": sorted(base.glob("test_all12_actual_best1_*/controls-*.json")),
        "kernels": sorted(base.rglob("controls.json")),
        "overwrites": sorted(base.glob("test_real_preexisting_overwrit*/overwrite.json")),
        "budgets": sorted(base.rglob("observed-budget.json")),
        "encodings": sorted(base.glob("test_whole_encoding_specimen*/encoding-specimen.json")),
        "statistics": sorted(base.rglob("statistics-specimen.json")),
        "private": sorted(base.glob("test_all12_both_full_original_*/native-writer-*.json")),
        "public": sorted(base.glob("test_all12_both_public_source_*/public-writer-*.json")),
    }
    raw.require(
        {name: len(paths) for name, paths in result.items()} == COUNTS,
        "Exact ALL inherited positive families; no missing/extra-count waiver",
    )
    return result


def mode(data: dict[str, Any]) -> tuple[str, str, str]:
    policy = data["source_metadata"]["observed_budget"]["payload"]["design"]["payload"]["ablation"]
    return policy["routing"], policy["goal"], policy["broadcast"]


def original_geometry(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    source = expected["source_metadata"]["evaluation"]["payload"]["plan"]["payload"]["source"][
        "payload"
    ]
    candidates = [
        candidate
        for candidate in (path.parent / "case/source").rglob("manifest.json")
        if controls.load(candidate).get("sha256") == source["inference_sha256"]
    ]
    raw.require(len(candidates) == 1, "ONE original physical selected inference configuration")
    inference = manifest(candidates[0], "iris/scientific-ablation-inference/v1/payload")["payload"]
    base = json.loads(inference["source_model_configuration_json"])["base_configuration"]
    cognitive = json.loads(base["cognitive_configuration_json"])
    workspace = json.loads(base["workspace_configuration_json"])
    return dict(
        modules=cognitive["num_modules"],
        slots=workspace["num_slots"],
        width=cognitive["state_dim"],
        cycles=workspace["cycles"],
        writers=workspace["winners_per_cycle"],
        empty_threshold=workspace["empty_threshold"],
    )


def geometry_matrix(results: list[dict[str, Any]], *, includes_nonwinning: bool) -> None:
    """Every protocol/mode/corner, not count-only or a wide statistics surrogate."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(Path(result["report"]["path"]).parent), []).append(result)
    scenarios: Counter[str] = Counter()
    observed: dict[str, set[tuple[str, str, str]]] = {}
    for pair in groups.values():
        raw.require(
            len(pair) == 2
            and {r["protocol"] for r in pair} == {"message", "workspace"}
            and pair[0]["geometry"] == pair[1]["geometry"]
            and pair[0]["original_mode"] == pair[1]["original_mode"],
            "Both protocols share the same original geometry/mode fixture",
        )
        g = pair[0]["geometry"]
        shape = tuple(
            g[n] for n in ("modules", "slots", "width", "cycles", "writers", "empty_threshold")
        )
        names = {
            (4, 1, 8, 3, 1, 1e-4): "occupied",
            (4, 4, 8, 5, 1, 1.0): "expiry",
            (4, 1, 8, 3, 1, 4.0): "nonsurvival",
            (64, 1, 1, 2, 1, 1e-4): "wide",
        }
        raw.require(shape in names, "Exact original configured full geometry matrix")
        scenario = names[shape]
        scenarios[scenario] += 1
        observed.setdefault(scenario, set()).add(tuple(pair[0]["original_mode"]))
        if scenario == "occupied":
            raw.require(
                all(r["metrics"]["overwrite_effects"] > 0 for r in pair)
                and any(r["metrics"]["occupied_block_frames"] > 0 for r in pair),
                "Every occupied fixture has real overwrite/BLOCK evidence",
            )
        elif scenario == "expiry":
            raw.require(
                any(r["metrics"]["actual_expired_records"] > 0 for r in pair),
                "Every expiry fixture has actual old-record TTL expiration",
            )
        elif scenario == "nonsurvival":
            raw.require(
                any(r["metrics"]["nonsurviving_effects"] > 0 for r in pair),
                "Every nonsurvival fixture has a real missing next-read writer",
            )
        else:
            raw.require(
                all(
                    r["metrics"]["effects"] == 12096
                    and r["metrics"]["pairs"] == 4032
                    and r["metrics"]["actual_teacher_executions"] == 768
                    for r in pair
                ),
                "Both complete wide native reports, not repeated effects",
            )
    raw.require(
        scenarios
        == {"occupied": 12 + int(includes_nonwinning), "expiry": 12, "nonsurvival": 12, "wide": 1}
        and all(observed[n] == designs.MODES for n in ("occupied", "expiry", "nonsurvival")),
        "ALL12 modes for EACH original corner plus ONE BOTH wide fixture",
    )


def audit(base: Path) -> dict[str, Any]:
    inventory = families(base)
    fixtures = sorted(
        {
            path.parent
            for name in (
                "preflights",
                "old_stages",
                "observations",
                "old_controls",
                "private",
                "public",
            )
            for path in inventory[name]
        }
        | {path for path in base.glob("report-layout-original-source*") if path.is_dir()}
    )
    raw.require(len(fixtures) == 73, "ALL73 inherited COMPLETE original physical fixtures")
    chains = []
    for fixture in fixtures:
        scores = fixture / "scores/scientific-ablation-evaluation-ledger-v2"
        plans = [path for path in scores.iterdir() if path.is_dir()]
        raw.require(len(plans) == 1, "ONE complete original score source per fixture")
        chains.append(sources.physical_chain(fixture, plans[0].name))
    expectations: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in inventory["preflights"]:
        expected = full.load(path)
        reservation = sources.preflight(expected)
        key = reservation["layout_sha256"]
        raw.require(key not in expectations, "Unique separately original full preflight")
        expectations[key] = (path, expected)
    stage_results = [
        stages.stage(path) for name in ("old_stages", "new_stages") for path in inventory[name]
    ]
    raw.require(
        {(protocol, value) for protocol, value, _ in stage_results}
        == {
            (protocol, value)
            for protocol in ("workspace-record", "paired-message")
            for value in designs.MODES
        }
        and sum(n for _, _, n in stage_results) == 120,
        "Both complete ALL12 old/new EMPTY stages",
    )
    raw.require(
        {native.native(path) for path in inventory["observations"]} == designs.MODES,
        "ALL12 original native observations",
    )
    for path in inventory["old_controls"]:
        controls.controls(
            path,
            native.score_chain_directory(
                path.parent, full.load(path)["payload"]["evaluation_plan"]["sha256"]
            ),
        )
    for path in inventory["kernels"]:
        kernel.witness(path)
    for path in inventory["overwrites"]:
        value = full.load(path)
        native.isolation(value, value["configuration"])
    for path in inventory["budgets"]:
        budgets.witness(path)
    raw.require(
        {layouts.specimen(path)["protocol"] for path in inventory["encodings"]}
        == {"workspace", "message"},
        "Both whole encoding specimens",
    )
    synthetic = [statistics.specimen(path) for path in inventory["statistics"]]
    raw.require(
        any(r["effects"] == 12096 and r["directed_pairs"] == 4032 for r in synthetic),
        "Inherited synthetic wide statistics, not actual whole wide report proof",
    )
    private = [legacy.inspect(path) for path in inventory["private"]]
    all_modes = {
        (protocol, value) for protocol in ("workspace", "message") for value in designs.MODES
    }
    raw.require(
        {(r["protocol"], r["mode"]) for r in private} == all_modes
        and sum(r["effects"] for r in private) == 720,
        "ALL24 private original modes/protocols/720 effects",
    )
    original_denials = {}
    for result in private:
        if result["protocol"] not in original_denials:
            original_denials[result["protocol"]] = legacy.negative_probes(
                full.ROOT / result["path"]["path"]
            )
    raw.require(
        sum(len(names) for names in original_denials.values()) == 67,
        "Complete inherited independent67 corruptions",
    )
    public, corruptions = [], {}
    public_modes = set()
    for path in inventory["public"]:
        report = full.load(path)
        key = report["payload"]["byte_preflight"]["layout_sha256"]
        raw.require(key in expectations, "Candidate cannot invent the original preflight")
        expected_path, expected = expectations[key]
        body = path.read_bytes()
        data = reader.read_report(body, expected=expected, runtime=raw.CPU_STREAM)
        raw.require(
            reader.format_report(data, expected=expected, runtime=raw.CPU_STREAM) == body,
            "Whole inherited PUBLIC byte read/format/source-bound roundtrip",
        )
        full.physical_metadata(path, data["payload"])
        result = raw.validate(data, expected, runtime=raw.CPU_STREAM)
        public_modes.add((result["protocol"], mode(expected)))
        result.update(
            report=full.binding(path), separately_retained_preflight=full.binding(expected_path)
        )
        public.append(result)
        if result["protocol"] not in corruptions:
            corruptions[result["protocol"]] = full.negative_probes(data, expected)
    raw.require(
        public_modes == all_modes
        and sum(r["effects"] for r in public) == 720
        and sum(r["observed_passes"] for r in public) == 468
        and sum(len(names) for names in corruptions.values()) == 36,
        "ALL24 PUBLIC original modes/protocols/720 effects/468 passes/36 rehashed denials",
    )
    raw.require(families(base) == inventory, "Whole inherited family membership stable")
    return dict(
        positive_family_counts=COUNTS,
        original_physical_chains=len(chains),
        physical_chains_sha256=raw.domain("iris/operational/physical-chains/v1", chains),
        public_reports=public,
        private_reports=[
            {
                key: result[key]
                for key in (
                    "path",
                    "protocol",
                    "mode",
                    "teacher_calls",
                    "observed_calls",
                    "effects",
                )
            }
            for result in private
        ],
        original_private_corruption_denials=original_denials,
        public_rehashed_corruption_denials=corruptions,
        additional_teacher_forwards=0,
        scientific_acceptance="not_assessed",
    )
