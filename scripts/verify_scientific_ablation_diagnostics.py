"""Independent stdlib source/JUnit/raw ordinary diagnostic audit, not model attestation.

Snapshot alone invokes explicit pytest collection in a child; audit does not.
Inputs are read-only. An optional exclusive new output is an observer receipt.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

_PREVIOUS = "artifacts/operational/r015-evaluation-ledger-before-20261012-run03.json"
_PREVIOUS_SHA = "55066ad2f32d521402b100e15f05a8b1aac67a00a8f01534c479aeda34384fd6"
_NEW = (
    "src/iris/application/scientific_ablation_diagnostics.py",
    "src/iris/artifacts/scientific_ablation_diagnostics_v1.py",
    "tests/integration/test_scientific_ablation_diagnostics.py",
)
MODULES = (
    "tests/integration/test_scientific_ablation_diagnostics.py",
    "tests/integration/test_cognitive_ablation_teacher.py",
    "tests/unit/test_workspace_trace.py",
    "tests/unit/test_workspace_probes.py",
)
PROBES = ("active_constraint", "operation", "conflicting_conditions", "result_type")


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite))


def snapshot(root: Path) -> dict[str, Any]:
    previous_pin = pin(root, _PREVIOUS)
    require(previous_pin["sha256"] == _PREVIOUS_SHA, "Accepted ledger snapshot changed")
    previous = load(root / _PREVIOUS)
    require(len(previous["bindings"]) == 524, "Accepted524 source inventory missing")
    for item in (*previous["bindings"], *previous["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted source/auditor changed")
    command = [
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
    ]
    collected = subprocess.run(command, cwd=root, capture_output=True, timeout=180, check=True)
    text = collected.stdout.decode("utf-8")
    ids = [line.strip() for line in text.splitlines() if "::test_" in line]
    require(len(ids) == len(set(ids)) and len(ids) > 100, "Exact current collection missing")
    return {
        "kind": "iris-r015-ordinary-diagnostics-before-v1",
        "previous": previous_pin,
        "bindings": [*previous["bindings"], *(pin(root, name) for name in _NEW)],
        "auditors": [
            *previous["auditors"],
            pin(root, "scripts/verify_scientific_ablation_diagnostics.py"),
        ],
        "modules": MODULES,
        "expected_cases": len(ids),
        "nodeids": ids,
        "actual_collection_exit_code": collected.returncode,
        "actual_collection_stdout_sha256": hashlib.sha256(collected.stdout).hexdigest(),
        "actual_collection_stdout_bytes": len(collected.stdout),
        "full_r015_complete": False,
    }


def pooled(traces: list[dict[str, Any]]) -> dict[str, Any]:
    first = traces[0]
    m, k = first["num_modules"], first["num_slots"]
    modules, slots = [0] * m, [0] * k
    ttl: list[float] = []
    writes: list[int] = []
    norms: list[float] = []
    cosines: list[float] = []
    cross_mass: list[float] = []
    active_reads = cross_reads = overwritten = 0
    for trace in traces:
        for cycle in trace["cycles"]:
            for module in cycle["selected_modules"]:
                modules[module] += 1
            for slot in cycle["selected_slots"]:
                slots[slot] += 1
            overwritten += len(cycle["overwritten_slots"])
            for slot, occupied in enumerate(cycle["slot_occupancy"]):
                if occupied:
                    ttl.append(cycle["slot_ttl"][slot])
                    writes.append(cycle["slot_write_counts"][slot])
                if cycle["slot_norms"][slot] > 0:
                    for other in range(slot + 1, k):
                        if cycle["slot_norms"][other] > 0:
                            cosines.append(cycle["workspace_cosine_matrix"][slot][other])
            norms.append(math.hypot(*cycle["slot_norms"]))
            for module, weights in enumerate(cycle["read_attention"]):
                if math.fsum(weights) > 0:
                    active_reads += 1
                    mass = math.fsum(
                        weight
                        for slot, weight in enumerate(weights)
                        if cycle["read_slot_sources"][slot] != module
                    )
                    cross_reads += int(mass > 0)
                    cross_mass.append(mass)
    cycles = sum(trace["cycle_count"] for trace in traces)

    def entropy(counts: list[int]) -> tuple[float, float]:
        total = sum(counts)
        if not total:
            return 0.0, 0.0
        value = -math.fsum((count / total) * math.log(count / total) for count in counts if count)
        return value, min(float(len(counts)), math.exp(value))

    def mean(values: list[float] | list[int]) -> float:
        # Same stable scaled summation convention, independently implemented.
        scale = max(abs(value) for value in values)
        return (
            0.0
            if scale == 0
            else scale * (math.fsum(value / scale for value in values) / len(values))
        )

    module_entropy, active_modules = entropy(modules)
    slot_entropy, active_slots = entropy(slots)
    count, mass_sum = sum(slots), math.fsum(cross_mass)
    return {
        "kind": "iris-workspace-diagnostics-v1",
        "write_scope": "allocated_before_post_write_interventions",
        "causal_influence": "not_measured",
        "scientific_acceptance": "not_assessed",
        "examples": len(traces),
        "cycles": cycles,
        "used_slots": sum(value > 0 for value in slots),
        "module_win_counts": modules,
        "slot_allocation_counts": slots,
        "module_win_rates": [value / cycles for value in modules],
        "module_entropy_nats": module_entropy,
        "slot_entropy_nats": slot_entropy,
        "effective_active_modules": active_modules,
        "effective_active_slots": active_slots,
        "allocated_writes": count,
        "overwrites": overwritten,
        "overwrite_rate": overwritten / count if count else None,
        "occupied_slot_observations": len(ttl),
        "mean_ttl": mean(ttl) if ttl else None,
        "mean_write_count": mean(writes) if writes else None,
        "mean_workspace_norm": mean(norms),
        "cosine_pairs": len(cosines),
        "mean_slot_cosine_similarity": mean(cosines) if cosines else None,
        "read_opportunities": cycles * m,
        "active_reads": active_reads,
        "cross_module_reads": cross_reads,
        "cross_module_read_rate": cross_reads / active_reads if active_reads else None,
        "cross_module_attention_mass": mass_sum,
        "mean_cross_module_attention_mass": mass_sum / active_reads if active_reads else None,
    }


def diagnostic(path: Path, ledger: Path) -> dict[str, Any]:
    result = manifest(path, "iris/scientific-ablation-diagnostics/v1")
    p = result["payload"]
    plan, evaluation, complete = p["evaluation_plan"], p["evaluation"], p["evaluation_complete"]
    require(
        plan == manifest(ledger / "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
        "Actual score plan differs",
    )
    require(
        evaluation
        == manifest(ledger / "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
        "Actual score report differs",
    )
    require(
        complete
        == manifest(
            ledger / "complete/complete.json", "iris/scientific-ablation-evaluation-complete/v2"
        ),
        "Actual complete score differs",
    )
    require(
        p["policy_sha256"] == domain("iris/scientific-ablation-diagnostic-policy/v1", p["policy"]),
        "Diagnostic policy differs",
    )
    require(
        p["scientific_acceptance"] == "not_assessed" and p["source_20_4"] == "not_assessed_smoke",
        "Tiny controls claimed science",
    )
    source = evaluation["payload"]["plan"]["payload"]["source"]["payload"]
    require(
        (source["selected_step"], source["terminal_step"]) == (1, 3),
        "Actual earlier best selection differs",
    )
    rows = p["rows"]
    require(
        len(rows) == p["teacher_executions"] == len(evaluation["payload"]["rows"]),
        "Incomplete diagnostic coverage",
    )
    traces = []
    for index, (row, scored) in enumerate(zip(rows, evaluation["payload"]["rows"], strict=True)):
        record = manifest(
            ledger / f"score-{index:05d}/row.json", "iris/scientific-ablation-evaluation-row/v2"
        )
        ack = manifest(
            ledger / f"score-{index:05d}/ack.json", "iris/scientific-ablation-evaluation-ack/v2"
        )
        require(
            row["evaluation_row_sha256"] == record["sha256"]
            and row["evaluation_ack_sha256"] == ack["sha256"]
            and row["row_index"] == index,
            "Physical diagnostic score row/ack differs",
        )
        for field in (
            "capture_row_sha256",
            "teacher_invocation_sha256",
            "teacher_target_sha256",
            "target_token_count",
            "target_nll_sum",
        ):
            require(row[field] == scored[field], "Original complete EOS NLL/target binding differs")
        trace, options = row["trace"], row["trace"]["options"]
        require(
            options
            == {
                "mode": p["policy"]["trace_mode"],
                "example_id": scored["example_id"],
                "checkpoint_id": source["best_checkpoint_sha256"],
                "seed": p["run_seed"],
                "maximum_scalars": p["policy"]["maximum_trace_scalars"],
            },
            "Actual diagnostic options differ",
        )
        require(
            trace["cycle_count"] == 3
            and trace["intervention_plan_sha256"] == source["ablation_policy_sha256"]
            and [cycle["cycle"] for cycle in trace["cycles"]] == [1, 2, 3],
            "Actual complete C3/mode trace differs",
        )
        require(
            [probe["field"] for probe in row["probes"]] == list(PROBES),
            "All frozen probe heads required",
        )
        for probe, classes in zip(row["probes"], p["probe_class_counts"], strict=True):
            targets = [
                target
                for target in range(classes)
                if probe["target_sha256"]
                == domain(
                    "iris/scientific-ablation-probe-target/v1",
                    {
                        "example_sha256": scored["generation_example_sha256"],
                        "vocabulary_sha256": p["probe_vocabulary_sha256"],
                        "field": probe["field"],
                        "class_index": target,
                    },
                )
            ]
            require(
                len(targets) == 1
                and 0 <= probe["predicted_class"] < classes
                and probe["correct"] == (probe["predicted_class"] == targets[0]),
                "Probe domain/correctness differs",
            )
        if source["ablation"]["broadcast"] == "disabled":
            require(
                all(
                    weight == 0
                    for cycle in trace["cycles"]
                    for weights in cycle["read_attention"]
                    for weight in weights
                ),
                "Disabled broadcast has active reads",
            )
        require(
            all(
                (cycle["full"] is not None) == (p["policy"]["trace_mode"] == "full")
                for cycle in trace["cycles"]
            ),
            "Explicit full-vector policy differs",
        )
        traces.append(trace)
    actual = pooled(traces)
    require(actual.keys() == p["workspace"].keys(), "Workspace inventory differs")
    for field, value in actual.items():
        saved = p["workspace"][field]
        require(
            math.isclose(value, saved, rel_tol=1e-14, abs_tol=1e-14)
            if isinstance(value, float)
            else value == saved,
            "Actual pooled raw workspace counts/means differ",
        )
    require(
        p["probe_correct_counts"]
        == [sum(row["probes"][i]["correct"] for row in rows) for i in range(4)],
        "Pooled probe counts differ",
    )
    first = traces[0]
    m, k, d, c = (first[key] for key in ("num_modules", "num_slots", "state_dim", "cycle_count"))
    bound = c * (
        1
        + 10 * m
        + 20 * k
        + k * k
        + m * k
        + (2 * (m + k) * d if p["policy"]["trace_mode"] == "full" else 0)
    )
    require(
        p["trace_scalar_bound"] == len(rows) * bound
        and bound <= p["policy"]["maximum_trace_scalars"]
        and len(rows) * bound <= p["policy"]["maximum_total_trace_scalars"]
        and len(rows) <= p["policy"]["maximum_teacher_executions"],
        "Complete diagnostic budget differs",
    )
    return {
        "report": binding(path),
        "rows": len(rows),
        "mode": source["ablation"],
        "trace_mode": p["policy"]["trace_mode"],
        "earlier_best": True,
        "native_eos_nll_equal_to_actual_score_chain": True,
    }


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before = load(before_path)
    require(
        pin(root, before["previous"]["path"]) == before["previous"],
        "Accepted before snapshot changed",
    )
    for item in (*before["bindings"], *before["auditors"]):
        require(pin(root, item["path"]) == item, "Source/test/auditor changed during gate")
    xml = ET.fromstring(junit.read_bytes())
    cases = list(xml.iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"]),
        "Exact selected JUnit IDs differ",
    )
    require(
        not any(list(case) for case in cases)
        and all(
            suite.attrib.get(key, "0") == "0"
            for suite in xml.iter("testsuite")
            for key in ("failures", "errors", "skipped")
        ),
        "Gate has failures/errors/skips",
    )
    captures: dict[tuple[str, str], Path] = {}
    for store in base.rglob("scientific-ablation-capture-v1"):
        for directory in store.iterdir():
            if directory.is_dir() and not directory.name.startswith("."):
                captures.setdefault(valid_capture(directory), directory)
    stores = list(base.rglob("scientific-ablation-evaluation-ledger-v2"))
    reports = []
    checked_chains = {}
    for path in sorted(base.glob("*/diagnostics.json")):
        raw = manifest(path, "iris/scientific-ablation-diagnostics/v1")
        plan_sha = raw["payload"]["evaluation_plan"]["sha256"]
        if plan_sha not in checked_chains:
            failures = []
            for store in stores:
                candidate = store / plan_sha
                if not candidate.is_dir():
                    continue
                try:
                    evidence = chain(candidate, captures)
                except Exception as error:
                    failures.append(type(error).__name__)
                    continue
                checked_chains[plan_sha] = (candidate, evidence)
                break
            require(
                plan_sha in checked_chains, "No independently valid physical score/capture chain"
            )
        reports.append(diagnostic(path, checked_chains[plan_sha][0]))
    modes = {
        (report["mode"]["routing"], report["mode"]["goal"], report["mode"]["broadcast"])
        for report in reports
    }
    require(
        modes
        == {
            (routing, goal, broadcast)
            for routing in ("learned", "random", "uniform")
            for goal in ("enabled", "disabled")
            for broadcast in ("enabled", "disabled")
        },
        "Actual saved all12 diagnostic mode coverage missing",
    )
    require(
        any(report["rows"] == 4 for report in reports)
        and {report["trace_mode"] for report in reports} == {"metadata", "full"},
        "Actual four-family/full-vector witnesses missing",
    )
    return {
        "kind": "iris-r015-ordinary-diagnostics-readback-v1",
        "passed": True,
        "source_pins": len(before["bindings"]),
        "auditor_pins": len(before["auditors"]),
        "exact_cases": len(ids),
        "junit": binding(junit),
        "before": binding(before_path),
        "reports": reports,
        "physical_chains": [item[1] for item in checked_chains.values()],
        "all12_c3_earlier_best": True,
        "full_r015_complete": False,
        "full_iris_gw_01_complete": False,
        "limits": [
            "No model execution attestation or learned-quality claim",
            "Read-only raw inputs; optional new exclusive observer receipt only",
            "No Iris/Torch imports in observer; snapshot alone uses pytest child collection",
            "No CUDA, installs, downloads, extraction or deletion",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.mode == "snapshot":
        result = snapshot(root)
    else:
        require(
            args.before is not None and args.junit is not None and args.base is not None,
            "Explicit retained inputs required",
        )
        result = audit(root, args.before, args.junit, args.base)
    require(
        not any(
            name == "iris" or name.startswith("iris.") or name == "torch" for name in sys.modules
        ),
        "Observer imported project/model code",
    )
    with args.output.open("xb") as output:
        output.write(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(args.output),
                "receipt": binding(args.output),
                "expected_cases": result.get("expected_cases"),
                "exact_cases": result.get("exact_cases"),
                "reports": len(result.get("reports", [])),
                "passed": result.get("passed"),
            }
        )
    )


if __name__ == "__main__":
    main()
