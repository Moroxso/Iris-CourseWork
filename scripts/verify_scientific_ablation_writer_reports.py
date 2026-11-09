"""Independent full1808 runtime/raw readback: original families AND public roots.

No Iris/Torch/tests/pytest imports, models, source helpers' main, file writes or
test-domain casting of public reports. Original physical fixtures remain tiny;
generic semantic codec support is not large-model/AMP/scientific acceptance.
"""

import copy
import json
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scientific_ablation_writer_report_raw as reader
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_design as designs
import verify_scientific_ablation_causal_observed_budget as budgets
import verify_scientific_ablation_causal_report_layout as layouts
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_controls as kernel
import verify_scientific_ablation_native_writer_effects as legacy
import verify_scientific_ablation_observation as native
import verify_scientific_ablation_observed_controls as stages
import verify_scientific_ablation_writer_statistics as statistics
from verify_scientific_ablation_collapse import original_membership
from verify_scientific_ablation_evaluation_io import manifest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "artifacts/operational/r015-writer-report-before-20261014-run01.json"
JUNIT = "artifacts/operational/r015-writer-report-final-20261014-run01.xml"
TERMINAL = "artifacts/operational/r015-writer-report-terminal-20261014-run01.json"
BASE = ROOT / "tmp/r015-writer-report-final-20261014-run01"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return dict(
        path=str(path.relative_to(ROOT)).replace("\\", "/"),
        bytes=len(raw),
        sha256=sha256(raw).hexdigest(),
    )


def matches(pin: dict[str, Any]) -> bool:
    observed = binding(ROOT / pin["path"])
    result = observed["bytes"] == pin["bytes"] and observed["sha256"] == pin["sha256"]
    assert isinstance(result, bool)
    return result


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(
        type(value) is dict and reader.canonical(value) + b"\n" == raw,
        "Exact original canonical JSON",
    )
    assert isinstance(value, dict)
    return value


def physical_metadata(path: Path, data: dict[str, Any]) -> None:
    """Bind source to original physical ledger/Parquet/selected configuration, not self SHA."""
    source = data["source_metadata"]
    source_info = source["evaluation"]["payload"]["plan"]["payload"]["source"]["payload"]
    design = source["observed_budget"]["payload"]["design"]["payload"]
    ledger = native.score_chain_directory(path.parent, source["evaluation_plan"]["sha256"])
    for name, filename, namespace in (
        ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
        ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
        (
            "evaluation_complete",
            "complete/complete.json",
            "iris/scientific-ablation-evaluation-complete/v2",
        ),
    ):
        require(
            reader.equal(source[name], manifest(ledger / filename, namespace)),
            "ALL original physical COMPLETE anchors",
        )
    original_membership(
        path.parent / "case/test.parquet",
        source["evaluation"]["payload"]["plan"]["payload"]["input"]["payload"][
            "test_parquet_sha256"
        ],
        design["members"],
    )
    candidates = [
        candidate
        for candidate in (path.parent / "case/source").rglob("manifest.json")
        if controls.load(candidate).get("sha256") == source_info["inference_sha256"]
    ]
    require(len(candidates) == 1, "ONE original selected physical inference")
    inference = manifest(candidates[0], "iris/scientific-ablation-inference/v1/payload")["payload"]
    configuration = json.loads(inference["source_model_configuration_json"])["base_configuration"]
    cfg = {
        name: configuration[name + "_configuration_json"]
        for name in ("backbone", "cognitive", "workspace")
    }
    require(
        source["run_seed"] == json.loads(inference["source_run_configuration_json"])["seed"],
        "Original physical run seed",
    )
    require(
        source["sidecar_sha256"]
        == source_info["teacher_sidecar_sha256"]
        == inference["source_teacher_sidecar_sha256"],
        "Original physical trained sidecar",
    )
    require(
        source["ordinary_teacher_graph_sha256"]
        == reader.domain(
            "iris/cognitive-ablation-teacher-graph/v1", cfg | {"policy": design["ablation"]}
        )
        and source["controlled_teacher_graph_sha256"]
        == reader.domain(
            "iris/cognitive-ablation-control-teacher-graph/v1",
            cfg | {"ablation_policy_sha256": design["ablation_policy_sha256"]},
        ),
        "BOTH physical configuration-derived graph identities",
    )


def negative_probes(report: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Candidate rehashes are kept coherent; independent original expectation is fixed."""
    protocol = (
        "message"
        if expected["source_metadata"]["observed_budget"]["payload"]["design"]["payload"][
            "donor_pairs"
        ]
        else "workspace"
    )
    roles = ("block", "donor") if protocol == "message" else ("shuffle", "zero")
    failures = []
    for name in (
        "source-seed",
        "layout",
        "ordinary-ack",
        "ordinary-order",
        "missing-pass",
        "duplicate-pass",
        "role-order",
        "effect-distance",
        "effect-flag",
        "effect-anchor",
        "effect-order",
        "statistics-mean",
        "quota",
        "teacher-count",
        "post-count",
        "prompt",
        "trace-seed",
        "competition",
    ):
        broken = copy.deepcopy(report)
        p = broken["payload"]
        trials = p["controlled_passes"]["trials"][roles[0]]
        if name == "source-seed":
            p["source_metadata"]["run_seed"] += 1
        elif name == "layout":
            p["byte_preflight"]["layout_sha256"] = "e" * 64
        elif name == "ordinary-ack":
            p["ordinary_rows"][0]["evaluation_ack_sha256"] = "e" * 64
        elif name == "ordinary-order":
            p["ordinary_rows"].reverse()
        elif name == "missing-pass":
            trials.pop()
        elif name == "duplicate-pass":
            trials[1] = copy.deepcopy(trials[0])
        elif name == "role-order":
            trials.reverse()
        elif name == "effect-distance":
            p["effects"][0]["genuine_distance"] = 0.123456
        elif name == "effect-flag":
            p["effects"][0]["survives_to_next_read"] = not p["effects"][0]["survives_to_next_read"]
        elif name == "effect-anchor":
            p["effects"][0]["intervention_invocation_sha256"] = "e" * 64
        elif name == "effect-order":
            p["effects"].reverse()
        elif name == "statistics-mean":
            pair = next(
                pair
                for pair in p["statistics"]["pairs"]
                if pair["mean_genuine_distance"] is not None
            )
            pair["mean_genuine_distance"] = 0.123456
        elif name == "quota":
            trials[0]["output"]["maximum_observation_scalars"] += 1
        elif name == "teacher-count":
            p["actual_teacher_executions"] -= 1
        elif name == "post-count":
            p["actual_post_intervention_scalar_bound"] -= 1
        elif name == "prompt":
            trials[0]["output"]["controlled"]["prompt_sha256"] = "e" * 64
        elif name == "trace-seed":
            trials[0]["output"]["controlled"]["trace"]["options"]["seed"] += 1
        else:
            trials[0]["output"]["controlled"]["trace"]["cycles"][trials[0]["write_cycle"] - 1][
                "module_bids"
            ][0] += 0.125
        broken["sha256"] = reader.domain(
            "iris/scientific-ablation-" + protocol + "-writer-report/v1", p
        )
        try:
            reader.validate(broken, expected, runtime=reader.CPU_STREAM)
        except ValueError:
            failures.append(name)
        else:
            raise ValueError("Independent corruption unexpectedly accepted: " + name)
    return failures


def audit() -> dict[str, Any]:
    before, terminal = load(ROOT / BEFORE), load(ROOT / TERMINAL)
    require(
        binding(ROOT / BEFORE)["sha256"]
        == "880409b2dc28cb9b239ac503351f01c3c7bda7fec7a3124df090c6f9bff70144",
        "Exact frozen current before",
    )
    require(
        binding(ROOT / TERMINAL)["sha256"]
        == "5fbbc45998c6b7700d0fc76808ac76f9e2d9849f8ee3ea20239ac18b7f8f0f90"
        and terminal["actual_terminal_tool_exit_code"] == 0,
        "Actual whole runtime terminal",
    )
    junit_pin = binding(ROOT / JUNIT)
    require(
        junit_pin["sha256"] == "493e882f32313e7a92d849e8fb1ee51dc0a12e6a2495b145124f5399c6724c72",
        "Exact full current JUnit",
    )
    cases = list(ET.parse(ROOT / JUNIT).getroot().iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(cases) == len(set(ids)) == 1808
        and all(not list(case) for case in cases)
        and ids == before["nodeids"],
        "Exact ordered full1808IDs/no fail/error/skip",
    )
    frozen = (
        before["inherited_bindings"]
        + before["inherited_auditors"]
        + before["new_bindings"]
        + [before["collector"]]
    )
    require(
        len(frozen) == 591
        and all(matches(pin) for pin in frozen + before["original_native_files"]),
        "572source/test+18auditors+collector/24retained witnesses",
    )
    own_pins = [
        binding(Path(__file__)),
        binding(ROOT / "scripts/scientific_ablation_writer_report_raw.py"),
    ]
    inventory = {
        "preflights": sorted(BASE.glob("test_all12_complete_original*/preflight-*.json")),
        "old_stages": sorted(BASE.glob("test_all12_original_best1_both*/stage-*.json")),
        "new_stages": sorted(BASE.glob("test_all12_complete_original*/stage-*.json")),
        "observations": sorted(BASE.glob("test_all12_actual_original_bes*/observations.json")),
        "old_controls": sorted(BASE.glob("test_all12_actual_best1_*/controls-*.json")),
        "kernels": sorted(BASE.rglob("controls.json")),
        "overwrites": sorted(BASE.glob("test_real_preexisting_overwrit*/overwrite.json")),
        "budgets": sorted(BASE.rglob("observed-budget.json")),
        "encodings": sorted(BASE.glob("test_whole_encoding_specimen*/encoding-specimen.json")),
        "statistics": sorted(BASE.rglob("statistics-specimen.json")),
        "private": sorted(BASE.glob("test_all12_both_full_original_*/native-writer-*.json")),
        "public": sorted(BASE.glob("test_all12_both_public_source_*/public-writer-*.json")),
    }
    require(
        [len(paths) for paths in inventory.values()]
        == [24, 24, 24, 12, 24, 168, 1, 28, 2, 25, 24, 24],
        "ALL inherited/new positive families",
    )
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
        | {path for path in BASE.glob("report-layout-original-source*") if path.is_dir()}
    )
    require(len(fixtures) == 73, "ALL61inherited+12public physical original source fixtures")
    roots = set(fixtures) | {path.parent for paths in inventory.values() for path in paths}

    def positive_files() -> list[Path]:
        return sorted({path for parent in roots for path in parent.rglob("*") if path.is_file()})

    paths = positive_files()
    pins = [binding(path) for path in paths]
    chains = []
    for fixture in fixtures:
        scores = fixture / "scores/scientific-ablation-evaluation-ledger-v2"
        plans = [path for path in scores.iterdir() if path.is_dir()]
        require(len(plans) == 1, "One whole physical COMPLETE per original fixture")
        chains.append(sources.physical_chain(fixture, plans[0].name))
    expectations = {}
    for path in inventory["preflights"]:
        expected = load(path)
        reservation = sources.preflight(expected)
        require(
            reservation["layout_sha256"] not in expectations, "Unique separately retained preflight"
        )
        expectations[reservation["layout_sha256"]] = (path, expected)
    require(len(expectations) == 24, "All original complete mode/protocol preflights")
    stage_results = [
        stages.stage(path) for name in ("old_stages", "new_stages") for path in inventory[name]
    ]
    require(
        {(protocol, mode) for protocol, mode, _ in stage_results}
        == {
            (protocol, mode)
            for protocol in ("workspace-record", "paired-message")
            for mode in designs.MODES
        }
        and sum(n for _, _, n in stage_results) == 120,
        "Complete both ALL12old/new EMPTY stages",
    )
    require(
        {native.native(path) for path in inventory["observations"]} == designs.MODES,
        "All12original native observations",
    )
    for path in inventory["old_controls"]:
        controls.controls(
            path,
            native.score_chain_directory(
                path.parent, load(path)["payload"]["evaluation_plan"]["sha256"]
            ),
        )
    for path in inventory["kernels"]:
        kernel.witness(path)
    for path in inventory["overwrites"]:
        value = load(path)
        native.isolation(value, value["configuration"])
    for path in inventory["budgets"]:
        budgets.witness(path)
    require(
        {layouts.specimen(path)["protocol"] for path in inventory["encodings"]}
        == {"workspace", "message"},
        "Both full encoding specimens",
    )
    synthetic = [statistics.specimen(path) for path in inventory["statistics"]]
    require(
        any(
            result["effects"] == 12096 and result["directed_pairs"] == 4032 for result in synthetic
        ),
        "Legitimate WIDE statistics fixture (not full report acceptance)",
    )
    private = [legacy.inspect(path) for path in inventory["private"]]
    require(
        sum(result["effects"] for result in private) == 720, "ALL private current native effects"
    )
    original_denials = {
        result["protocol"]: legacy.negative_probes(Path(result["path"]["path"]))
        for result in private[:2]
    }
    require(
        sum(len(names) for names in original_denials.values()) == 67,
        "Full inherited independent67corruptions",
    )
    public = []
    corruptions = {}
    for path in inventory["public"]:
        report = load(path)
        key = report["payload"]["byte_preflight"]["layout_sha256"]
        require(key in expectations, "Candidate cannot invent its expected preflight")
        expected_path, expected = expectations[key]
        physical_metadata(path, report["payload"])
        result = reader.validate(report, expected, runtime=reader.CPU_STREAM)
        result.update(report=binding(path), separately_retained_preflight=binding(expected_path))
        public.append(result)
        if result["protocol"] not in corruptions:
            corruptions[result["protocol"]] = negative_probes(report, expected)
    require(
        len(public) == 24
        and sum(result["effects"] for result in public) == 720
        and sum(result["observed_passes"] for result in public) == 468,
        "ALL24PUBLIC/protocols/modes/full original720effects/468passes",
    )
    forbidden = [
        name
        for name in sys.modules
        if any(
            name == banned or name.startswith(banned + ".")
            for banned in ("iris", "torch", "tests", "pytest")
        )
    ]
    require(not forbidden, "No model/project/test imports in independent reader")
    require(
        all(matches(pin) for pin in frozen + before["original_native_files"] + own_pins + pins)
        and paths == positive_files()
        and binding(ROOT / JUNIT) == junit_pin,
        "ALL source/auditor/reader/old+newpositive files stable before/end",
    )
    return dict(
        kind="iris-r015-complete-writer-report-independent-readback-v1",
        passed=True,
        before=binding(ROOT / BEFORE),
        junit=junit_pin,
        actual_runtime_cases=1808,
        inherited_cases=1617,
        new_cases=191,
        frozen_source_test_pins=572,
        frozen_auditor_collector_pins=19,
        reader_pins=own_pins,
        positive_family_counts={name: len(values) for name, values in inventory.items()},
        original_physical_chains=len(chains),
        physical_chains_sha256=reader.domain("iris/operational/physical-chains/v1", chains),
        stable_positive_files=len(pins),
        positive_inventory_sha256=reader.domain("iris/operational/positive-inventory/v1", pins),
        public_reports=public,
        original_private_corruption_denials=original_denials,
        public_rehashed_corruption_denials=corruptions,
        forbidden_imports=forbidden,
        additional_teacher_forwards=0,
        generic_schema_api_not_yet_accepted=True,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


if __name__ == "__main__":
    sys.stdout.buffer.write(reader.canonical(audit()) + b"\n")
