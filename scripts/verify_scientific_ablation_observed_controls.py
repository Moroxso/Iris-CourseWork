"""Independent retained raw readback for the internal two-pass observed stage.

Snapshot alone collects child pytest. Audit imports no Iris/Torch/tests, performs
zero forwards and does not claim complete effects, byte preflight or science.
"""

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_observed_budget as budgets
import verify_scientific_ablation_controls as kernel
import verify_scientific_ablation_observation as native
from scientific_ablation_observed_budget_guard import checked_budget
from verify_scientific_ablation_causal_design import MODES, read_design
from verify_scientific_ablation_collapse import original_membership
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture

PREVIOUS = "artifacts/operational/r015-observed-budget-before-20261013-run01.json"
PREVIOUS_SHA = "a6fb19b9f64eb039da90fc76a83a08bdc171003dd2642e3a96fbc1c79c17291b"
VERDICT = "artifacts/operational/r015-observed-budget-readback-20261013-run01.json"
VERDICT_SHA = "7cb95009c1dc6e4f57614c91638a969030b946b997512d3a3fcaf71717a36161"
NATIVE = budgets.PREVIOUS
NATIVE_SHA = budgets.PREVIOUS_SHA
CHANGED = "src/iris/application/scientific_ablation_causal_controls.py"
NEW = "tests/integration/test_scientific_ablation_observed_controls.py"
READER_TEST = "tests/unit/test_scientific_ablation_observed_controls_reader.py"
GUARD = "scripts/scientific_ablation_observed_budget_guard.py"
SELF = "scripts/verify_scientific_ablation_observed_controls.py"
PRODUCER_BEFORE = "artifacts/operational/r015-observed-stage-before-20261013-run01.json"
PRODUCER_BEFORE_SHA = "048f68593cdcbb2a75254ec85422965b8c3b776fe8c75ce6c9f4c86b1ac34367"
PRODUCER_AUDITOR_SHA = "0b87dd18a9bc4dedd44ec2de6768cc58bfd0a48e8503b4f0921539d71afbd159"
MODULES = tuple(dict.fromkeys((*native.MODULES, *budgets.MODULES, NEW, READER_TEST)))


def scope(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    for path, digest in (
        (PREVIOUS, PREVIOUS_SHA),
        (VERDICT, VERDICT_SHA),
        (NATIVE, NATIVE_SHA),
        (budgets.VERDICT, budgets.VERDICT_SHA),
    ):
        require(pin(root, path)["sha256"] == digest, "Immutable predecessor changed")
    previous, verdict = controls.load(root / PREVIOUS), controls.load(root / VERDICT)
    require(
        len(previous["bindings"]) == 548
        and len(previous["auditors"]) == 10
        and verdict["passed"] is True
        and verdict["exact_cases"] == 248,
        "Accepted numeric predecessor scope required",
    )
    bindings, changes = [], []
    for prior in previous["bindings"]:
        current = pin(root, prior["path"])
        if current != prior:
            require(prior["path"] == CHANGED, "Unrelated accepted source/test changed")
            changes.append(dict(previous=prior, current=current))
        bindings.append(current)
    require(len(changes) == 1, "Exactly one intentional original-source refactor required")
    for auditor in previous["auditors"]:
        require(pin(root, auditor["path"]) == auditor, "Accepted observer changed")
    return (
        [*bindings, pin(root, NEW), pin(root, READER_TEST)],
        [*previous["auditors"], pin(root, SELF), pin(root, GUARD)],
        changes[0],
    )


def snapshot(root: Path) -> dict[str, Any]:
    bindings, auditors, changed = scope(root)
    inherited = set(controls.load(root / NATIVE)["nodeids"]) | set(
        controls.load(root / PREVIOUS)["nodeids"]
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
            *MODULES,
        ],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=180,
    )
    ids = [s.strip() for s in collected.stdout.decode().splitlines() if "::test_" in s]
    new = {i for i in ids if any(i.startswith(p + "::") for p in (NEW, READER_TEST))}
    require(
        len(ids) == len(set(ids)) == 944
        and len(inherited) == 891
        and len(new) == 53
        and sum(i.startswith(READER_TEST + "::") for i in ids) == 17
        and set(ids) == inherited | new,
        "Exact891 inherited plus36 native-stage plus17 reader cases required",
    )
    require(scope(root) == (bindings, auditors, changed), "Scope drift during collection")
    return dict(
        kind="iris-r015-original-source-observed-stage-before-v1",
        previous=pin(root, PREVIOUS),
        numeric_verdict=pin(root, VERDICT),
        native_before=pin(root, NATIVE),
        native_verdict=pin(root, budgets.VERDICT),
        bindings=bindings,
        auditors=auditors,
        changed=changed,
        modules=list(MODULES),
        nodeids=ids,
        expected_cases=len(ids),
        new_cases=len(new),
        inherited_cases=len(inherited),
        actual_collection_exit_code=collected.returncode,
        scope="combined-affected-native-source-budget-and-internal-stage-not-repository-suite",
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
    )


def stage(path: Path) -> tuple[str, tuple[str, str, str], int]:
    raw = controls.load(path)
    require(
        set(raw)
        == {
            "kind",
            "scope",
            "design",
            "observed_budget",
            "evaluation_plan",
            "evaluation",
            "evaluation_complete",
            "run_seed",
            "actual_ordinary_executions",
            "actual_observed_executions",
            "actual_teacher_executions",
            "rows",
            "scientific_acceptance",
            "full_r015_complete",
        }
        and raw["kind"] == "iris-tests-original-source-observed-empty-stage-v1"
        and raw["scope"] == "two-pass-internal-stage-not-full-effects-or-report-byte-preflight"
        and raw["scientific_acceptance"] == "not_assessed"
        and raw["full_r015_complete"] is False,
        "Distinct honest internal test witness, not public effect artifact",
    )
    suffix = "message" if path.name == "stage-message.json" else "workspace"
    design_path, budget_path = (path.with_name(f"{n}-{suffix}.json") for n in ("design", "budget"))
    require(
        controls.load(design_path) == raw["design"]
        and controls.load(budget_path) == raw["observed_budget"],
        "Exact saved design and complete observed-budget bytes",
    )
    checked_budget(raw["observed_budget"])
    protocol, mode, purpose, _ = read_design(design_path)
    # The unchanged numeric observer independently rederives every planned count.
    # It expects a sibling design.json; adapt its witness naming via an explicit
    # reader below, not by creating/copying/relabeling public report bytes.
    b, d = raw["observed_budget"]["payload"], raw["design"]["payload"]
    require(
        raw["observed_budget"]["sha256"] == domain(budgets.DOMAIN, b)
        and b["design"] == raw["design"]
        and b["maximum_observation_scalars"] == 336
        and b["observation_scalar_bound_per_pass"] == 336
        and b["total_observation_scalar_bound"] == d["budget"]["controlled_executions"] * 336
        and b["original_trace_proposal_scalar_bound"] == d["budget"]["total_numeric_scalars"]
        and b["total_trace_proposal_observation_scalar_bound"]
        == b["original_trace_proposal_scalar_bound"] + b["total_observation_scalar_bound"]
        <= d["policy"]["maximum_total_numeric_scalars"],
        "Owned whole-experiment observation reservation, not actual full execution",
    )
    for field, design_field in (
        ("planned_ordinary_teacher_executions", "ordinary_executions"),
        ("planned_controlled_teacher_executions", "controlled_executions"),
        ("planned_total_teacher_executions", "total_executions"),
        ("planned_observed_teacher_executions", "controlled_executions"),
    ):
        require(b[field] == d["budget"][design_field], "Planned counts differ")
    ledger = native.score_chain_directory(path.parent, raw["evaluation_plan"]["sha256"])
    for field, filename, artifact_domain in (
        ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
        ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
        (
            "evaluation_complete",
            "complete/complete.json",
            "iris/scientific-ablation-evaluation-complete/v2",
        ),
    ):
        require(
            raw[field] == manifest(ledger / filename, artifact_domain),
            "Physical COMPLETE score binding",
        )
    generation = raw["evaluation"]["payload"]["plan"]["payload"]
    source = generation["source"]["payload"]
    require(
        (source["selected_step"], source["terminal_step"]) == (1, 3)
        and d["ablation"] == source["ablation"]
        and purpose == source["purpose"] == "smoke"
        and protocol == ("paired-message" if suffix == "message" else "workspace-record"),
        "Genuine original selected best/mode/protocol",
    )
    original_membership(
        path.parent / "case/test.parquet",
        generation["input"]["payload"]["test_parquet_sha256"],
        d["members"],
    )
    for member, expected in zip(d["members"], generation["rows"], strict=True):
        require(
            all(
                member[k] == expected[k]
                for k in ("example_id", "example_sha256", "prompt_sha256", "task_type")
            ),
            "ALL original row membership, including unselected",
        )
    candidates = [
        p
        for p in (path.parent / "case/source").rglob("manifest.json")
        if controls.load(p).get("sha256") == source["inference_sha256"]
    ]
    require(len(candidates) == 1, "One actual selected inference source")
    inference = manifest(candidates[0], "iris/scientific-ablation-inference/v1/payload")["payload"]
    saved = binding(candidates[0].parent / "state.safetensors")
    require(
        saved["sha256"] == inference["state_sha256"]
        and saved["bytes"] == inference["state_size_bytes"]
        and raw["run_seed"] == json.loads(inference["source_run_configuration_json"])["seed"],
        "Physical selected-state/run seed",
    )
    base = json.loads(inference["source_model_configuration_json"])["base_configuration"]
    config = {n: base[n + "_configuration_json"] for n in ("backbone", "cognitive", "workspace")}
    rows = raw["rows"]
    require(
        [r["row"]["row_index"] for r in rows] == d["selected_row_indices"]
        and len(rows) == (2 if suffix == "message" else 3)
        and raw["actual_ordinary_executions"] == raw["actual_observed_executions"] == len(rows)
        and raw["actual_teacher_executions"]
        == 2 * len(rows)
        < b["planned_total_teacher_executions"],
        "Complete selected stage coverage without hidden baseline",
    )
    for measurement in rows:
        require(set(measurement) == {"row", "observation"}, "Closed measurement witness")
        row, observation = measurement["row"], measurement["observation"]
        c = native.observed(observation, config)
        index = row["row_index"]
        member, score = d["members"][index], raw["evaluation"]["payload"]["rows"][index]
        score_row = manifest(
            ledger / f"score-{index:05d}/row.json", "iris/scientific-ablation-evaluation-row/v2"
        )
        score_ack = manifest(
            ledger / f"score-{index:05d}/ack.json", "iris/scientific-ablation-evaluation-ack/v2"
        )
        require(
            row["evaluation_row_sha256"] == score_row["sha256"]
            and row["evaluation_ack_sha256"] == score_ack["sha256"]
            and row["capture_row_sha256"] == score["capture_row_sha256"]
            and row["ordinary_teacher_invocation_sha256"] == score["teacher_invocation_sha256"]
            and row["target_sha256"] == c["target_sha256"] == score["teacher_target_sha256"]
            and row["target_token_count"]
            == c["supervised_token_count"]
            == score["target_token_count"]
            and row["target_nll_sum"] == score["target_nll_sum"]
            and row["ordinary_mean_target_nll"] == row["controlled_mean_target_nll"] == c["loss"]
            and math.isclose(
                c["loss"] * row["target_token_count"],
                row["target_nll_sum"],
                rel_tol=2e-7,
                abs_tol=1e-7,
            )
            and row["controlled_invocation_sha256"] == c["invocation_sha256"]
            and row["controlled_trace"] == c["trace"]
            and row["messages"] == c["messages"]
            and c["prompt_sha256"] == member["cognitive_prompt_sha256"]
            and c["sidecar_state_sha256"] == source["teacher_sidecar_sha256"],
            "Saved EOS NLL/target/original COMPLETE row/ack/actual observed receipt",
        )
        expected_control = (
            dict(kind="iris-workspace-message-plan-v1", donor=None)
            if suffix == "message"
            else dict(kind="iris-workspace-intervention-plan-v1", interventions=[])
        )
        require(
            c["plan"]["ablation"] == d["ablation"]
            and c["plan"]["control"] == expected_control
            and c["plan"]["maximum_message_scalars"]
            == (d["policy"]["maximum_message_scalars"] if suffix == "message" else None)
            and len(c["messages"]) == (12 if suffix == "message" else 0),
            "Exactly EMPTY recipient control; no intervention or donor",
        )
        for cycle, state in zip(observation["observations"], c["states"], strict=True):
            require(
                all(
                    canonical(cycle["native_written"][f]["values"]) == canonical(state[f])
                    for f in native.FIELDS
                ),
                "ALL7 EMPTY native-written/postwrite identity",
            )
        ordinary, controlled = row["ordinary_trace"], row["controlled_trace"]
        for trace, mode_name, plan_sha in (
            (ordinary, d["policy"]["ordinary_trace_mode"], d["ablation_policy_sha256"]),
            (controlled, "full", c["plan_sha256"]),
        ):
            require(
                trace["intervention_plan_sha256"] == plan_sha
                and trace["options"]
                == dict(
                    mode=mode_name,
                    example_id=member["example_id"],
                    checkpoint_id=source["best_checkpoint_sha256"],
                    seed=raw["run_seed"],
                    maximum_scalars=d["policy"]["maximum_trace_scalars"],
                ),
                "Actual original mode-bound ordinary/EMPTY trace",
            )
        require(
            [{k: v for k, v in t.items() if k != "full"} for t in ordinary["cycles"]]
            == [{k: v for k, v in t.items() if k != "full"} for t in controlled["cycles"]],
            "Ordinary/observed EMPTY cycle metadata parity",
        )
    return protocol, mode, len(rows)


def _passing_ids(path: Path) -> list[str]:
    xml = ET.fromstring(path.read_bytes())
    cases = list(xml.iter("testcase"))
    ids = [c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases]
    require(
        len(ids) == len(set(ids))
        and not any(list(c) for c in cases)
        and all(
            s.attrib.get(key, "0") == "0"
            for s in xml.iter("testsuite")
            for key in ("failures", "errors", "skipped")
        ),
        "Exact unique passing JUnit IDs without failure/error/skip required",
    )
    return ids


def audit(
    root: Path,
    before_path: Path,
    junit: Path,
    base: Path,
    *,
    runtime_before: Path | None = None,
    reader_junit: Path | None = None,
) -> dict[str, Any]:
    before_pin, junit_pin = binding(before_path), binding(junit)
    before = controls.load(before_path)
    frozen = scope(root)
    require(frozen == (before["bindings"], before["auditors"], before["changed"]), "Frozen scope")
    require(
        (runtime_before is None) == (reader_junit is None), "Both explicit split inputs required"
    )
    source_ids = _passing_ids(junit)
    reader_ids: list[str] = []
    runtime_pin = reader_pin = repair = None
    if runtime_before is not None and reader_junit is not None:
        runtime_pin, reader_pin = binding(runtime_before), binding(reader_junit)
        require(
            pin(root, PRODUCER_BEFORE)["sha256"] == PRODUCER_BEFORE_SHA
            and runtime_pin["sha256"] == PRODUCER_BEFORE_SHA,
            "Exact immutable original producer before01 required",
        )
        original = controls.load(runtime_before)
        require(
            original["bindings"] == frozen[0][:-1]
            and frozen[0][-1]["path"] == READER_TEST
            and original["auditors"][:-1] == frozen[1][:-2]
            and original["auditors"][-1]["path"] == SELF
            and original["auditors"][-1]["sha256"] == PRODUCER_AUDITOR_SHA
            and frozen[1][-2]["path"] == SELF
            and frozen[1][-1]["path"] == GUARD
            and original["changed"] == before["changed"]
            and original["expected_cases"] == 927
            and original["modules"] == [p for p in before["modules"] if p != READER_TEST]
            and set(original["nodeids"])
            == {i for i in before["nodeids"] if not i.startswith(READER_TEST + "::")}
            and sorted(source_ids) == sorted(original["nodeids"]),
            "Original549 pins and927cases unchanged; ONLYSELF repair plus new guard/test",
        )
        reader_ids = _passing_ids(reader_junit)
        require(
            len(source_ids) == 927
            and len(reader_ids) == 17
            and all(i.startswith(READER_TEST + "::") for i in reader_ids)
            and set(reader_ids)
            == {i for i in before["nodeids"] if i.startswith(READER_TEST + "::")},
            "Exact separate17 reader guard cases required; not another model run",
        )
        repair = dict(previous=original["auditors"][-1], current=frozen[1][-2])
    ids = [*source_ids, *reader_ids]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"] == 944
        and sorted(ids) == sorted(before["nodeids"])
        and before["actual_collection_exit_code"] == 0,
        "Exact complete current944case union; retain distinct actual producer provenance",
    )
    stages = sorted(base.glob("test_all12_original_best1_both*/stage-*.json"))
    natives = sorted(base.glob("test_all12_actual_original_bes*/observations.json"))
    old_controls = sorted(base.glob("test_all12_actual_best1_*/controls-*.json"))
    kernels = sorted(base.rglob("controls.json"))
    overwrites = sorted(base.glob("test_real_preexisting_overwrit*/overwrite.json"))
    budget_paths = sorted(base.rglob("observed-budget.json"))
    require(
        tuple(map(len, (stages, natives, old_controls, kernels, overwrites, budget_paths)))
        == (24, 12, 24, 168, 1, 28),
        "All actual inherited and new positive witness inventories",
    )
    positive = (*stages, *natives, *old_controls, *kernels, *overwrites)
    paths = sorted(
        {p for witness in positive for p in witness.parent.rglob("*") if p.is_file()}
        | {p for budget in budget_paths for p in (budget, budget.with_name("design.json"))}
    )
    raw_pins = [binding(p) for p in paths]
    chains = []
    for fixture in sorted({p.parent for p in (*stages, *natives, *old_controls)}):
        captures = {}
        for store in fixture.rglob("scientific-ablation-capture-v1"):
            for directory in store.iterdir():
                if directory.is_dir() and not directory.name.startswith("."):
                    captures[valid_capture(directory)] = directory
        store = fixture / "scores/scientific-ablation-evaluation-ledger-v2"
        plans = [p for p in store.iterdir() if p.is_dir()]
        require(len(plans) == 1, "One original COMPLETE evaluation plan per actual fixture")
        chains.append(chain(native.score_chain_directory(fixture, plans[0].name), captures))
    results = [stage(p) for p in stages]
    require(
        {(protocol, mode) for protocol, mode, _ in results}
        == {(p, m) for p in ("workspace-record", "paired-message") for m in MODES}
        and sum(n for _, _, n in results) == 60,
        "All12 modes BOTH original-source stage protocols and60observed rows",
    )
    require({native.native(p) for p in natives} == MODES, "Inherited all12 native sources")
    for p in old_controls:
        plan = controls.load(p)["payload"]["evaluation_plan"]["sha256"]
        controls.controls(p, native.score_chain_directory(p.parent, plan))
    for p in kernels:
        kernel.witness(p)
    overwritten = controls.load(overwrites[0])
    native.isolation(overwritten, overwritten["configuration"])
    prior, slot = overwritten["blocked"]["observations"][1]["before_write"], overwritten["slot"]
    require(
        prior["occupancy"]["values"][0][slot]
        and all(
            prior[f]["values"][0][slot] > 0 for f in ("age", "confidence", "ttl", "write_count")
        ),
        "Inherited actual preexisting ALL7 overwrite, not a fabricated empty record",
    )
    budget_results = [budgets.witness(p) for p in budget_paths]
    require(
        {
            (p, m)
            for p, m, purpose, ordinary in budget_results
            if purpose == "smoke" and ordinary == "metadata"
        }
        == {(p, m) for p in ("workspace-record", "paired-message") for m in MODES}
        and {p for p, _, purpose, _ in budget_results if purpose == "scientific"}
        == {"workspace-record", "paired-message"}
        and {p for p, _, _, ordinary in budget_results if ordinary == "full"}
        == {"workspace-record", "paired-message"},
        "Inherited all12/full5000/full-ordinary numeric witnesses",
    )
    require(
        scope(root) == frozen
        and [binding(p) for p in paths] == raw_pins
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin
        and (runtime_before is None or binding(runtime_before) == runtime_pin)
        and (reader_junit is None or binding(reader_junit) == reader_pin),
        "Source and all raw positive inputs stable before/end",
    )
    return dict(
        kind="iris-r015-original-source-observed-stage-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        runtime_before=runtime_pin,
        reader_unit_junit=reader_pin,
        observer_only_repair=repair,
        actual_source_runtime_cases=len(source_ids),
        actual_separate_reader_cases=len(reader_ids),
        exact_cases=len(ids),
        new_cases=53,
        inherited_cases=891,
        source_test_pins=len(frozen[0]),
        auditor_pins=len(frozen[1]),
        changed=frozen[2],
        actual_stage_witnesses=24,
        actual_stage_modes=12,
        actual_stage_observed_rows=60,
        actual_stage_ordinary_executions=60,
        actual_stage_observed_executions=60,
        actual_stage_teacher_executions=120,
        actual_physical_complete_chains=len(chains),
        inherited_native_witnesses=12,
        inherited_controls=24,
        inherited_kernel_witnesses=168,
        inherited_numeric_budget_witnesses=28,
        original_complete_chains=chains,
        raw_positive_files=raw_pins,
        additional_teacher_executions_by_reader=0,
        scientific_acceptance="not_assessed",
        scope="source-owned-two-pass-observed-empty-stage-not-full-effects-statistics-or-byte-preflight",
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--runtime-before", type=Path)
    parser.add_argument("--reader-junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        result = snapshot(args.workspace.resolve())
    else:
        require(
            args.before is not None and args.junit is not None and args.base is not None,
            "Explicit retained inputs",
        )
        result = audit(
            args.workspace.resolve(),
            args.before,
            args.junit,
            args.base,
            runtime_before=args.runtime_before,
            reader_junit=args.reader_junit,
        )
    require(
        not any(
            n == prefix or n.startswith(prefix + ".")
            for n in sys.modules
            for prefix in ("iris", "torch", "pytest", "tests")
        ),
        "Observer imported project/model/tests",
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
            )
        )
    )


if __name__ == "__main__":
    main()
