"""Independent source/JSON reservation readback; no model/project imports in audit."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_observed_budget as budgets
import verify_scientific_ablation_causal_report_layout as layout
import verify_scientific_ablation_controls as kernel
import verify_scientific_ablation_observation as native
import verify_scientific_ablation_observed_controls as stages
from scientific_ablation_observed_budget_guard import checked_budget
from verify_scientific_ablation_causal_design import MODES
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture

BASELINE = "artifacts/operational/r015-report-layout-before-20261013-run03.json"
BASELINE_SHA = "eb5af9c8c22789e074bd6fe563998a916354062007e7f19f83a487371cb83240"
OBSERVED = "artifacts/operational/r015-observed-stage-before-20261013-run02.json"
OBSERVED_SHA = "479c41c2c07c7bac5f3448898042f03698781bd6349951fce02bcb4806b3fe29"
CHANGED = {
    "src/iris/application/scientific_ablation_causal_controls.py",
    "src/iris/artifacts/scientific_ablation_causal_report_layout_v1.py",
}
NEW = (
    "tests/unit/test_scientific_ablation_causal_report_source_types.py",
    "tests/unit/test_scientific_ablation_complete_source_limits.py",
    "tests/integration/test_scientific_ablation_complete_source_preflight.py",
)
SELF = "scripts/verify_scientific_ablation_complete_source_preflight.py"


def normalized(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v.replace("\\", "/") if k in ("root", "path") else v for k, v in record.items()}


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for path, checksum in ((BASELINE, BASELINE_SHA), (OBSERVED, OBSERVED_SHA)):
        require(binding(Path(path))["sha256"] == checksum, "Immutable accepted predecessor drift")
    old = controls.load(Path(BASELINE))
    bindings, changed = [], []
    for prior in old["bindings"]:
        current = pin(Path.cwd(), prior["path"])
        if prior["path"].replace("\\", "/") in CHANGED:
            require(
                normalized(current) != normalized(prior), "Both intentional source edits required"
            )
            changed.append(dict(previous=prior, current=current))
        else:
            require(
                normalized(current) == normalized(prior), "Unrelated accepted source/test drift"
            )
        bindings.append(current)
    require(len(changed) == 2, "Only the two explicit existing production edits")
    auditors = []
    for prior in old["auditors"]:
        current = pin(Path.cwd(), prior["path"])
        require(normalized(current) == normalized(prior), "Historical observer drift")
        auditors.append(current)
    return (
        [*bindings, *(pin(Path.cwd(), p) for p in NEW)],
        [*auditors, pin(Path.cwd(), SELF)],
        changed,
    )


def snapshot() -> dict[str, Any]:
    frozen = scope()
    prior, observed = controls.load(Path(BASELINE)), controls.load(Path(OBSERVED))
    modules = list(dict.fromkeys([*observed["modules"], *prior["modules"], *NEW]))
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
        capture_output=True,
        check=True,
        timeout=180,
    )
    ids = [s.strip() for s in collected.stdout.decode().splitlines() if "::test_" in s]
    inherited = set(prior["nodeids"]) | set(observed["nodeids"])
    new = {s for s in ids if any(s.startswith(p + "::") for p in NEW)}
    require(
        len(ids) == len(set(ids)) and set(ids) == inherited | new,
        "Exact inherited union plus new cases; no coverage shrink",
    )
    require(len(inherited) == 1023 and scope() == frozen, "Stable accepted complete affected scope")
    return dict(
        kind="iris-r015-complete-source-before-v1",
        bindings=frozen[0],
        auditors=frozen[1],
        changed=frozen[2],
        modules=modules,
        nodeids=ids,
        inherited_cases=len(inherited),
        new_cases=len(new),
        expected_cases=len(ids),
        actual_collection_exit_code=collected.returncode,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def preflight(raw: dict[str, Any]) -> dict[str, Any]:
    names = {
        "kind",
        "scope",
        "actual_preflight_teacher_executions",
        "before_observations",
        "complete_layout",
        "source_metadata",
        "maximum_report_bytes",
        "report_byte_preflight_bound",
        "source_metadata_bytes",
        "ordinary_rows_byte_bound",
        "controlled_passes_byte_bound",
        "effects_byte_bound",
        "statistics_byte_bound",
        "framing_byte_bound",
        "post_intervention_scalar_bound",
        "layout_sha256",
        "scientific_acceptance",
        "full_r015_complete",
    }
    require(
        set(raw) == names
        and raw["kind"] == "iris-tests-original-complete-source-before-busy-v1"
        and raw["scope"] == "trusted-ready-numeric-and-json-reservation-not-full-effects"
        and type(raw["actual_preflight_teacher_executions"]) is int
        and raw["actual_preflight_teacher_executions"] == 0
        and type(raw["before_observations"]) is list
        and len(raw["before_observations"]) == 2
        and raw["scientific_acceptance"] == "not_assessed"
        and raw["full_r015_complete"] is False,
        "Honest original READY/no-forward reservation witness",
    )
    for event, stage in zip(raw["before_observations"], ("numeric", "bytes"), strict=True):
        require(
            type(event) is dict
            and set(event)
            == {
                "stage",
                "states",
                "run_busy",
                "original_modes_restored",
                "caller_rng_unchanged",
                "private_rng_unchanged",
            }
            and event["stage"] == stage
            and event["states"] == ["ready"] * 3
            and event["run_busy"] is False
            and all(
                event[k] is True
                for k in (
                    "original_modes_restored",
                    "caller_rng_unchanged",
                    "private_rng_unchanged",
                )
            ),
            "Actual captured READY/run/modes/RNG before each original preflight",
        )
    metadata, descriptor = raw["source_metadata"], raw["complete_layout"]
    checked_budget(metadata["observed_budget"])
    derived = layout.inventory(descriptor, metadata)
    total = layout.node_bound(descriptor) + 1
    size = len(canonical(metadata))
    framing = (
        total - size - sum(v for k, v in derived.items() if k != "post_intervention_scalar_bound")
    )
    derived.update(source_metadata_bytes=size, framing_byte_bound=framing)
    require(
        all(type(raw[n]) is int and raw[n] == v for n, v in derived.items())
        and type(raw["report_byte_preflight_bound"]) is int
        and raw["report_byte_preflight_bound"] == total
        and type(raw["maximum_report_bytes"]) is int
        and total <= raw["maximum_report_bytes"] <= 512 * 1024**2,
        "Independent whole JSON components and exact quota",
    )
    protocol = (
        "message"
        if metadata["observed_budget"]["payload"]["design"]["payload"]["donor_pairs"]
        else "workspace"
    )
    checksum = domain(
        "iris/scientific-ablation-causal-report-layout/v1",
        dict(
            protocol=protocol,
            source_metadata=metadata,
            maximum_report_bytes=raw["maximum_report_bytes"],
            **{n: v for n, v in derived.items() if n != "source_metadata_bytes"},
            report_byte_preflight_bound=total,
            complete_layout=descriptor,
        ),
    )
    require(checksum == raw["layout_sha256"], "Distinct complete layout identity")
    return dict(
        protocol=protocol,
        bound=total,
        layout_sha256=checksum,
        evaluation_plan_sha256=metadata["evaluation_plan"]["sha256"],
        **derived,
    )


def physical_chain(fixture: Path, expected: str) -> dict[str, Any]:
    captures = {
        valid_capture(d): d
        for s in fixture.rglob("scientific-ablation-capture-v1")
        for d in s.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    }
    return chain(native.score_chain_directory(fixture, expected), captures)


def audit(before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    frozen, before_pin, junit_pin = scope(), binding(before_path), binding(junit)
    before = controls.load(before_path)
    require(
        frozen == (before["bindings"], before["auditors"], before["changed"]),
        "Exact frozen runtime scope",
    )
    ids = stages._passing_ids(junit)
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and before["actual_collection_exit_code"] == 0,
        "Actual complete affected gate, not a union of stale production runs",
    )
    new = sorted(base.glob("test_all12_complete_original*/preflight-*.json"))
    old_stages = sorted(base.glob("test_all12_original_best1_both*/stage-*.json"))
    new_stages = sorted(base.glob("test_all12_complete_original*/stage-*.json"))
    natives = sorted(base.glob("test_all12_actual_original_bes*/observations.json"))
    old_controls = sorted(base.glob("test_all12_actual_best1_*/controls-*.json"))
    kernels = sorted(base.rglob("controls.json"))
    overwrites = sorted(base.glob("test_real_preexisting_overwrit*/overwrite.json"))
    budget_paths = sorted(base.rglob("observed-budget.json"))
    specimens = sorted(base.glob("test_whole_encoding_specimen*/encoding-specimen.json"))
    require(
        tuple(
            map(
                len,
                (
                    new,
                    old_stages,
                    new_stages,
                    natives,
                    old_controls,
                    kernels,
                    overwrites,
                    budget_paths,
                    specimens,
                ),
            )
        )
        == (24, 24, 24, 12, 24, 168, 1, 28, 2),
        "Complete inherited/new positive inventory",
    )
    fixtures = sorted(
        {p.parent for p in (*new, *old_stages, *natives, *old_controls)}
        | {p for p in base.glob("report-layout-original-source*") if not p.is_symlink()}
    )
    positive = (
        *new,
        *old_stages,
        *new_stages,
        *natives,
        *old_controls,
        *kernels,
        *overwrites,
        *specimens,
    )
    paths = sorted(
        {
            p
            for w in (*positive, *fixtures)
            for p in (w if w.is_dir() else w.parent).rglob("*")
            if p.is_file()
        }
        | {p for b in budget_paths for p in (b, b.with_name("design.json"))}
    )
    raw_pins = [binding(p) for p in paths]
    chains = []
    for fixture in fixtures:
        store = fixture / "scores/scientific-ablation-evaluation-ledger-v2"
        plans = [p for p in store.iterdir() if p.is_dir()]
        require(len(plans) == 1, "One actual original COMPLETE source per native fixture")
        chains.append(physical_chain(fixture, plans[0].name))
    results = [preflight(controls.load(p)) for p in new]
    stage_results = [stages.stage(p) for p in (*old_stages, *new_stages)]
    require(
        {(p, m) for p, m, _ in stage_results}
        == {(p, m) for p in ("workspace-record", "paired-message") for m in MODES}
        and sum(n for _, _, n in stage_results) == 120,
        "Both all12 current stage families",
    )
    for path, result in zip(new, results, strict=True):
        raw = controls.load(path)
        protocol = result["protocol"]
        stage = controls.load(path.with_name(f"stage-{protocol}.json"))
        metadata = raw["source_metadata"]
        require(
            stage["observed_budget"] == metadata["observed_budget"],
            "Actual stage uses reserved whole budget",
        )
        ledger = native.score_chain_directory(path.parent, result["evaluation_plan_sha256"])
        for name, file, checksum_domain in (
            ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
            ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
            (
                "evaluation_complete",
                "complete/complete.json",
                "iris/scientific-ablation-evaluation-complete/v2",
            ),
        ):
            require(
                metadata[name] == stage[name] == manifest(ledger / file, checksum_domain),
                "Actual original COMPLETE physical score binding",
            )
    require({native.native(p) for p in natives} == MODES, "Inherited all12 native observations")
    for p in old_controls:
        plan = controls.load(p)["payload"]["evaluation_plan"]["sha256"]
        controls.controls(p, native.score_chain_directory(p.parent, plan))
    for p in kernels:
        kernel.witness(p)
    overwritten = controls.load(overwrites[0])
    native.isolation(overwritten, overwritten["configuration"])
    for p in budget_paths:
        budgets.witness(p)
    layouts = [layout.specimen(p) for p in specimens]
    require(
        {r["protocol"] for r in layouts} == {"workspace", "message"},
        "Both current full encoding specimens",
    )
    require(
        scope() == frozen
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin
        and [binding(p) for p in paths] == raw_pins,
        "Stable source/observer/ALL positive physical bytes before/end",
    )
    return dict(
        kind="iris-r015-complete-source-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        new_cases=before["new_cases"],
        inherited_cases=before["inherited_cases"],
        changed=frozen[2],
        source_test_pins=len(frozen[0]),
        auditor_pins=len(frozen[1]),
        actual_complete_preflight_witnesses=24,
        actual_complete_preflight_modes=12,
        actual_new_stage_ordinary_executions=60,
        actual_new_stage_observed_executions=60,
        actual_inherited_stage_executions=120,
        original_physical_chains=chains,
        stable_positive_file_pins=raw_pins,
        reservations=results,
        layouts=layouts,
        additional_teacher_executions_by_reader=0,
        reader_model_imports=0,
        trusted_before_busy_integration_complete=True,
        actual_measured_writer_effects=0,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        result = snapshot()
    else:
        require(
            args.before is not None and args.junit is not None and args.base is not None,
            "Explicit actual runtime inputs required",
        )
        result = audit(args.before, args.junit, args.base)
    require(
        not any(
            n == p or n.startswith(p + ".")
            for n in sys.modules
            for p in ("iris", "torch", "pytest", "tests")
        ),
        "No project/model imports in observer",
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                receipt=binding(args.output),
                cases=result.get("expected_cases", result.get("exact_cases")),
                passed=result.get("passed"),
                additional_forwards=0,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
