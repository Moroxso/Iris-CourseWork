"""Independent source/control readback, PyArrow via pinned original-row reader.

No Iris, Torch or pytest imports/model execution. Snapshot alone runs collection
in a child. Audit hashes physical sources/chains and actual saved control traces;
this is not execution attestation, scientific acceptance or installed delivery.
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
from verify_scientific_ablation_causal_design import MODES, read_design
from verify_scientific_ablation_collapse import original_membership
from verify_scientific_ablation_evaluation_io import (
    binding,
    canonical,
    domain,
    manifest,
    nonfinite,
    pairs,
    require,
)
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture

PREVIOUS = "artifacts/operational/r015-causal-design-before-20261012-run02.json"
PREVIOUS_SHA = "41c4e285dbdc824d37978c64276fef4dc07ddd23f48872262f9b4b8a23766508"
NEW = (
    "src/iris/artifacts/scientific_ablation_causal_controls_v1.py",
    "src/iris/application/scientific_ablation_causal_controls.py",
    "tests/integration/test_scientific_ablation_causal_controls.py",
)
SELF = "scripts/verify_scientific_ablation_causal_controls.py"
MODULES = (
    NEW[2],
    "tests/unit/test_scientific_ablation_causal_design.py",
    "tests/unit/test_cognitive_ablation_controls.py",
    "tests/integration/test_scientific_ablation_diagnostics.py",
)
DOMAIN = "iris/scientific-ablation-causal-controls/v1"


def load(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    require(len(content) <= 64 * 1024**2, "Bounded raw controls/snapshot required")
    raw: dict[str, Any] = json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(canonical(raw) + b"\n" == content, "Canonical raw controls/snapshot required")
    return raw


def scope(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(pin(root, PREVIOUS)["sha256"] == PREVIOUS_SHA, "Accepted design snapshot changed")
    previous = load(root / PREVIOUS)
    require(
        len(previous["bindings"]) == 538 and len(previous["auditors"]) == 7,
        "Exact accepted design scope required",
    )
    for item in (*previous["bindings"], *previous["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted source/test/auditor changed")
    return [*previous["bindings"], *(pin(root, p) for p in NEW)], [
        *previous["auditors"],
        pin(root, SELF),
    ]


def snapshot(root: Path) -> dict[str, Any]:
    bindings, auditors = scope(root)
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
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    new = [i for i in ids if i.startswith(NEW[2] + "::")]
    require(
        len(ids) == len(set(ids))
        and len(new) >= 46
        and sum("::test_all12_actual_best1_terminal3_" in i for i in new) == 12,
        "Exact new collection/all12 source cases required",
    )
    require(scope(root) == (bindings, auditors), "Source drift during collection")
    return dict(
        kind="iris-r015-source-causal-controls-before-v1",
        previous=pin(root, PREVIOUS),
        bindings=bindings,
        auditors=auditors,
        modules=MODULES,
        nodeids=ids,
        expected_cases=len(ids),
        new_cases=len(new),
        actual_collection_exit_code=collected.returncode,
        full_r015_complete=False,
    )


def controls(path: Path, ledger: Path) -> tuple[str, tuple[str, str, str], dict[str, Any]]:
    raw = load(path)
    require(raw["sha256"] == domain(DOMAIN, raw["payload"]), "Control domain differs")
    p = raw["payload"]
    require(
        p["kind"] == "iris-scientific-ablation-causal-controls-v1"
        and p["scope"] == "source-owned-empty-controls-not-causal-effects-or-answer-quality"
        and p["scientific_acceptance"] == "not_assessed",
        "Honest distinct control scope required",
    )
    message = path.name == "controls-message.json"
    design_path = path.with_name("design-message.json" if message else "design-workspace.json")
    protocol, mode, purpose, budget = read_design(design_path)
    require(
        load(design_path) == p["design"]
        and purpose == "smoke"
        and protocol == ("paired-message" if message else "workspace-record"),
        "Physical actual source design/report differs",
    )
    d, g = p["design"]["payload"], p["design"]["payload"]["geometry"]
    require(
        manifest(ledger / "plan.json", "iris/scientific-ablation-evaluation-plan/v2")
        == p["evaluation_plan"]
        and manifest(ledger / "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2")
        == p["evaluation"]
        and manifest(
            ledger / "complete/complete.json", "iris/scientific-ablation-evaluation-complete/v2"
        )
        == p["evaluation_complete"],
        "Actual physical COMPLETE original scores differ",
    )
    generation = p["evaluation"]["payload"]["plan"]["payload"]
    source = generation["source"]["payload"]
    require(
        d["ablation"] == source["ablation"] and d["purpose"] == source["purpose"],
        "Original best/mode/design source differs",
    )
    parquet = path.parent / "case/test.parquet"
    original = original_membership(
        parquet, generation["input"]["payload"]["test_parquet_sha256"], d["members"]
    )
    for member, expectation in zip(d["members"], generation["rows"], strict=True):
        require(
            all(
                member[k] == expectation[k]
                for k in ("example_id", "example_sha256", "prompt_sha256", "task_type")
            ),
            "ALL original generation/design members differ",
        )
    # These are fixture-scoped ORIGINAL inference manifests, not fabricated graph receipts.
    source_files = sorted((path.parent / "case/source").rglob("manifest.json"))
    projections = []
    for candidate in source_files:
        payload = load(candidate)
        if payload.get("sha256") == source["inference_sha256"]:
            projections.append(manifest(candidate, "iris/scientific-ablation-inference/v1/payload"))
            state = binding(candidate.parent / "state.safetensors")
            require(
                state["sha256"] == payload["payload"]["state_sha256"]
                and state["bytes"] == payload["payload"]["state_size_bytes"],
                "Actual selected inference state bytes differ",
            )
    require(len(projections) == 1, "Exact physical selected-best inference source missing")
    inference = projections[0]["payload"]
    config = json.loads(
        inference["source_model_configuration_json"],
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
    )["base_configuration"]
    graph_config = {
        name: config[name + "_configuration_json"]
        for name in ("backbone", "cognitive", "workspace")
    }
    require(
        p["ordinary_teacher_graph_sha256"]
        == domain(
            "iris/cognitive-ablation-teacher-graph/v1", graph_config | {"policy": d["ablation"]}
        )
        and p["controlled_teacher_graph_sha256"]
        == domain(
            "iris/cognitive-ablation-control-teacher-graph/v1",
            graph_config | {"ablation_policy_sha256": d["ablation_policy_sha256"]},
        )
        and p["sidecar_sha256"] == source["teacher_sidecar_sha256"]
        and p["run_seed"] == json.loads(inference["source_run_configuration_json"])["seed"],
        "Original actual configuration graph/sidecar/run seed differs",
    )
    empty = dict(
        kind="iris-cognitive-ablation-control-plan-v1",
        ablation=d["ablation"],
        control=(
            dict(kind="iris-workspace-message-plan-v1", donor=None)
            if message
            else dict(kind="iris-workspace-intervention-plan-v1", interventions=[])
        ),
        maximum_message_scalars=d["policy"]["maximum_message_scalars"] if message else None,
    )
    require(
        p["empty_plan_sha256"] == domain("iris/cognitive-ablation-control-plan/v1", empty),
        "Exact combined empty-control identity differs",
    )
    rows = p["rows"]
    require(
        [r["row_index"] for r in rows] == d["selected_row_indices"],
        "Input-selected actual control coverage differs",
    )
    scalars = (
        budget["ordinary_trace_scalars"]
        + budget["full_trace_scalars"]
        + budget["message_scalars_per_collection"]
    )
    base_bytes = len(
        canonical(
            {k: p[k] for k in ("design", "evaluation_plan", "evaluation", "evaluation_complete")}
        )
    )
    byte_bound = (
        base_bytes
        + 65536
        + len(rows)
        * (32768 + 32 * scalars + 1024 * (g["num_modules"] * g["cycle_count"] if message else 0))
    )
    require(
        p["actual_teacher_executions"] == 2 * len(rows) < budget["total_executions"]
        and p["actual_trace_proposal_scalar_bound"] == len(rows) * scalars
        and p["report_byte_preflight_bound"] == byte_bound <= p["maximum_report_bytes"]
        and len(canonical(raw)) + 1 <= byte_bound,
        "Actual controls/future budget/report byte bound differs",
    )
    geometry = tuple(g[k] for k in ("num_modules", "num_slots", "state_dim", "cycle_count"))
    for row in rows:
        index = row["row_index"]
        member, score = d["members"][index], p["evaluation"]["payload"]["rows"][index]
        physical_row = manifest(
            ledger / f"score-{index:05d}/row.json", "iris/scientific-ablation-evaluation-row/v2"
        )
        physical_ack = manifest(
            ledger / f"score-{index:05d}/ack.json", "iris/scientific-ablation-evaluation-ack/v2"
        )
        require(
            row["evaluation_row_sha256"] == physical_row["sha256"]
            and row["evaluation_ack_sha256"] == physical_ack["sha256"]
            and row["capture_row_sha256"] == score["capture_row_sha256"]
            and row["ordinary_teacher_invocation_sha256"] == score["teacher_invocation_sha256"]
            and row["target_sha256"] == score["teacher_target_sha256"]
            and row["target_token_count"] == score["target_token_count"]
            and row["target_nll_sum"] == score["target_nll_sum"]
            and math.isfinite(row["ordinary_mean_target_nll"])
            and row["ordinary_mean_target_nll"] == row["controlled_mean_target_nll"]
            and math.isclose(
                row["target_nll_sum"],
                row["ordinary_mean_target_nll"] * row["target_token_count"],
                rel_tol=2e-7,
                abs_tol=1e-7,
            ),
            "Actual selected saved EOS NLL/physical score/ack differs",
        )
        require(
            row["controlled_invocation_sha256"]
            == domain(
                "iris/cognitive-ablation-controlled-teacher-invocation/v1",
                dict(
                    plan_sha256=p["empty_plan_sha256"],
                    prompt_sha256=member["cognitive_prompt_sha256"],
                    target_sha256=row["target_sha256"],
                    teacher_graph_sha256=p["controlled_teacher_graph_sha256"],
                    sidecar_state_sha256=p["sidecar_sha256"],
                    supervised_token_count=row["target_token_count"],
                ),
            ),
            "Actual controlled target/count/graph invocation differs",
        )
        ordinary, controlled = row["ordinary_trace"], row["controlled_trace"]
        for trace, mode_name, plan_sha in (
            (ordinary, d["policy"]["ordinary_trace_mode"], d["ablation_policy_sha256"]),
            (controlled, "full", p["empty_plan_sha256"]),
        ):
            require(
                tuple(trace[k] for k in ("num_modules", "num_slots", "state_dim", "cycle_count"))
                == geometry
                and trace["intervention_plan_sha256"] == plan_sha
                and trace["options"]
                == dict(
                    mode=mode_name,
                    example_id=member["example_id"],
                    checkpoint_id=source["best_checkpoint_sha256"],
                    seed=p["run_seed"],
                    maximum_scalars=d["policy"]["maximum_trace_scalars"],
                ),
                "Actual original best/mode/geometry control trace differs",
            )
        require(
            [{k: v for k, v in c.items() if k != "full"} for c in ordinary["cycles"]]
            == [{k: v for k, v in c.items() if k != "full"} for c in controlled["cycles"]]
            and len(controlled["cycles"]) == g["cycle_count"]
            and all(
                c["full"] is not None and len(c["write_modules"]) == g["writers_per_cycle"]
                for c in controlled["cycles"]
            ),
            "Actual empty-control metadata cycle parity differs",
        )
        if d["policy"]["ordinary_trace_mode"] == "full":
            require(
                ordinary["cycles"] == controlled["cycles"],
                "Actual empty-control full vectors differ",
            )
        require(
            len(row["messages"]) == (g["num_modules"] * g["cycle_count"] if message else 0),
            "All actual proposal receipts required",
        )
        for offset, proposal in enumerate(row["messages"]):
            require(
                (proposal["cycle"], proposal["module"])
                == (offset // g["num_modules"] + 1, offset % g["num_modules"])
                and proposal["example_id"] == member["example_id"]
                and proposal["prompt_sha256"] == member["cognitive_prompt_sha256"]
                and proposal["teacher_graph_sha256"] == p["controlled_teacher_graph_sha256"]
                and proposal["sidecar_state_sha256"] == p["sidecar_sha256"]
                and len(proposal["values"]) == g["state_dim"]
                and proposal["content_sha256"]
                == domain("iris/module-message-content/v1", {"values": proposal["values"]}),
                "Actual ordered same-mode proposal content differs",
            )
    require(
        not any(
            b'"' + k + b'"' in canonical(raw)
            for k in (
                b"prompt",
                b"expected_response",
                b"output_text",
                b"hidden_facts",
                b"python_state",
            )
        ),
        "Private original data leaked into public controls",
    )
    return (
        protocol,
        mode,
        dict(
            report=binding(path),
            design=binding(design_path),
            original_membership=original,
            rows=len(rows),
            actual_teacher_executions=p["actual_teacher_executions"],
        ),
    )


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_pin, junit_pin = binding(before_path), binding(junit)
    before = load(before_path)
    bindings, auditors = scope(root)
    require(
        before["bindings"] == bindings
        and before["auditors"] == auditors
        and before["modules"] == list(MODULES)
        and before["actual_collection_exit_code"] == 0,
        "Exact fresh source/control scope differs",
    )
    xml = ET.fromstring(junit.read_bytes())
    cases = list(xml.iter("testcase"))
    ids = [c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(c) for c in cases)
        and all(
            s.attrib.get(k, "0") == "0"
            for s in xml.iter("testsuite")
            for k in ("failures", "errors", "skipped")
        ),
        "Exact passing JUnit cases required",
    )
    paths = sorted(base.glob("test_all12_actual_best1_*/controls-*.json"))
    require(len(paths) == 24, "Actual all12x2 raw controls required")
    # Only exact positive fixture trees. Mutated negatives are not successful evidence.
    raw_paths = sorted({p for report in paths for p in report.parent.rglob("*") if p.is_file()})
    raw_pins = [binding(p) for p in raw_paths]
    captures: dict[tuple[str, str], Path] = {}
    for report in paths:
        for store in report.parent.rglob("scientific-ablation-capture-v1"):
            for candidate in store.iterdir():
                if candidate.is_dir() and not candidate.name.startswith("."):
                    captures[valid_capture(candidate)] = candidate
    verified: dict[str, dict[str, Any]] = {}
    observed = []
    for path in paths:
        p = load(path)["payload"]
        plan_sha = p["evaluation_plan"]["sha256"]
        ledger = path.parent / "scores/scientific-ablation-evaluation-ledger-v2" / plan_sha
        if plan_sha not in verified:
            verified[plan_sha] = chain(ledger, captures)
        observed.append(controls(path, ledger))
    require(
        {(protocol, mode) for protocol, mode, _ in observed}
        == {
            (protocol, mode)
            for protocol in ("workspace-record", "paired-message")
            for mode in MODES
        },
        "Actual all12 modes BOTH source protocols missing",
    )
    require(
        scope(root) == (bindings, auditors)
        and [binding(p) for p in raw_paths] == raw_pins
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin,
        "Source/control/original physical inputs changed during audit",
    )
    return dict(
        kind="iris-r015-source-causal-controls-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        new_cases=before["new_cases"],
        source_test_pins=len(bindings),
        auditor_pins=len(auditors),
        raw_positive_tree_pins=len(raw_pins),
        controls=[r for _, _, r in observed],
        physical_chains=list(verified.values()),
        all12_modes_both_source_protocols=True,
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        limits=[
            "Original Parquet/physical source/score/capture readback, not execution attestation",
            "Actual tiny source/control phase, not writer effects/statistics/5000 forwards",
            "No project/model/pytest imports; pinned PyArrow original Parquet reader",
            "No new forwards, sandbox jobs, download, install, extraction or deletion",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        result = snapshot(args.workspace.resolve())
    else:
        require(
            args.before is not None and args.junit is not None and args.base is not None,
            "Explicit audit inputs required",
        )
        result = audit(args.workspace.resolve(), args.before, args.junit, args.base)
    require(
        not any(
            n == prefix or n.startswith(prefix + ".")
            for n in sys.modules
            for prefix in ("iris", "torch", "pytest", "tests")
        ),
        "Observer imported project/model/test",
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
