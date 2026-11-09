"""Independent raw collapse/Parquet/chain audit; no Iris/Torch/pytest imports.

Snapshot explicitly collects selected pytest modules in a child. Audit reads
original Parquet using PyArrow plus stdlib/pinned older raw observers, never
models. Optional NEW exclusive output is an observer receipt, not publication.
"""

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_scientific_ablation_diagnostics import diagnostic, load
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture

_PREVIOUS = "artifacts/operational/r015-diagnostics-before-20261012-run04.json"
_PREVIOUS_SHA = "b2514da74af0eccf28ba4170be6dd0a8b278e11671d9aa8c2cf2774da0798521"
_NEW = (
    "src/iris/artifacts/scientific_ablation_collapse_v1.py",
    "src/iris/application/scientific_ablation_collapse.py",
    "tests/unit/test_scientific_ablation_collapse.py",
    "tests/integration/test_scientific_ablation_collapse.py",
)
MODULES = (
    "tests/unit/test_scientific_ablation_collapse.py",
    "tests/integration/test_scientific_ablation_collapse.py",
    "tests/unit/test_scientific_collapse.py",
    "tests/integration/test_scientific_ablation_diagnostics.py",
    "tests/integration/test_cognitive_ablation_teacher.py",
    "tests/unit/test_workspace_trace.py",
    "tests/unit/test_workspace_probes.py",
)
TASKS = ("split_spec_repair", "delayed_goal", "conflict_resolution", "next_tool_prediction")


def snapshot(root: Path) -> dict[str, Any]:
    previous_pin = pin(root, _PREVIOUS)
    require(previous_pin["sha256"] == _PREVIOUS_SHA, "Accepted current-format snapshot changed")
    previous = load(root / _PREVIOUS)
    require(len(previous["bindings"]) == 527, "Accepted527 scope missing")
    for item in (*previous["bindings"], *previous["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted predecessor source/auditor changed")
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
        timeout=180,
        check=True,
    )
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    require(len(ids) == len(set(ids)) and len(ids) > 200, "Exact selected collection missing")
    return {
        "kind": "iris-r015-ablation-collapse-before-v1",
        "previous": previous_pin,
        "bindings": [*previous["bindings"], *(pin(root, name) for name in _NEW)],
        "auditors": [
            *previous["auditors"],
            pin(root, "scripts/verify_scientific_ablation_collapse.py"),
        ],
        "modules": MODULES,
        "expected_cases": len(ids),
        "nodeids": ids,
        "actual_collection_exit_code": collected.returncode,
        "actual_collection_stdout_sha256": hashlib.sha256(collected.stdout).hexdigest(),
        "actual_collection_stdout_bytes": len(collected.stdout),
        "full_r015_complete": False,
    }


def summary(rows: list[dict[str, Any]], modules: int, slots: int) -> dict[str, Any] | None:
    if not rows:
        return None
    wins, allocations = [0] * modules, [0] * slots
    for row in rows:
        for number, cycle in enumerate(row["cycles"], 1):
            require(cycle["cycle"] == number, "Compact cycle order differs")
            require(
                len(cycle["write_modules"]) == len(cycle["selected_slots"]),
                "Writer correspondence differs",
            )
            require(
                cycle["write_modules"] == sorted(cycle["write_modules"]), "Module order differs"
            )
            for indices, counts in (
                (cycle["write_modules"], wins),
                (cycle["selected_slots"], allocations),
            ):
                require(
                    len(indices) == len(set(indices))
                    and all(type(i) is int and 0 <= i < len(counts) for i in indices),
                    "Hard indices differ",
                )
                for index in indices:
                    counts[index] += 1

    def entropy(counts: list[int]) -> tuple[float, float]:
        total = sum(counts)
        require(total > 0, "Nonempty actual allocations required")
        value = -math.fsum((count / total) * math.log(count / total) for count in counts if count)
        return value, min(float(len(counts)), math.exp(value))

    mh, me = entropy(wins)
    sh, se = entropy(allocations)
    return {
        "examples": len(rows),
        "cycles": sum(len(row["cycles"]) for row in rows),
        "allocated_writes": sum(wins),
        "module_win_counts": wins,
        "slot_allocation_counts": allocations,
        "module_entropy_nats": mh,
        "slot_entropy_nats": sh,
        "effective_active_modules": me,
        "effective_active_slots": se,
    }


def original_membership(
    path: Path, expected_sha: str, members: list[dict[str, Any]]
) -> dict[str, Any]:
    before = binding(path)
    require(
        before["sha256"] == expected_sha and before["bytes"] <= 1024**2,
        "Original fixture Parquet hash/budget differs",
    )
    physical = pq.ParquetFile(io.BytesIO(path.read_bytes()))
    require(physical.metadata.num_rows == len(members) <= 5000, "Original full membership differs")
    logical = physical.read().to_pylist()
    for index, (row, member) in enumerate(zip(logical, members, strict=True)):
        facts = {}
        for key, tagged in row["hidden_facts"]:
            kind = tagged["kind"]
            require(
                kind in ("string", "integer", "boolean") and key not in facts,
                "Original fact tags differ",
            )
            facts[key] = tagged[f"{kind}_value"]
        row["hidden_facts"] = facts
        require(
            member["row_index"] == index
            and member["example_id"] == row["id"]
            and member["task_type"] == row["task_type"]
            and member["difficulty"] == row["difficulty"]
            and type(row["difficulty"]) is int
            and row["difficulty"] in (1, 2, 3)
            and member["example_sha256"] == domain("iris/scientific-ablation-example/v1", row),
            "Original full logical row/task/difficulty fingerprint differs",
        )
    require(binding(path) == before, "Original Parquet changed during readback")
    return {
        "parquet": before,
        "original_rows": len(logical),
        "full_example_task_difficulty_hashes": True,
    }


def compact(path: Path, ledger: Path, parquet: Path) -> dict[str, Any]:
    result = manifest(path, "iris/scientific-ablation-collapse/v1")
    p = result["payload"]
    ordinary_path = path.with_name("ordinary.json")
    ordinary_audit = diagnostic(ordinary_path, ledger)
    ordinary = manifest(ordinary_path, "iris/scientific-ablation-diagnostics/v1")
    d = ordinary["payload"]
    require(
        p["ordinary_diagnostics_sha256"] == ordinary["sha256"],
        "Actual metadata producer reference differs",
    )
    require(
        all(
            p[name] == d[name]
            for name in (
                "evaluation_plan",
                "evaluation",
                "evaluation_complete",
                "run_seed",
                "teacher_graph_sha256",
                "teacher_executions",
                "trace_scalar_bound",
            )
        ),
        "Original measurement/ledger inventory differs",
    )
    source = p["evaluation"]["payload"]["plan"]["payload"]["source"]["payload"]
    require(
        p["scientific_acceptance"] == "not_assessed"
        and p["policy"]["trace_mode"] == "metadata"
        and p["policy"]["coverage"] == "complete_original_test_order",
        "Compact scope differs",
    )
    require(
        p["policy_sha256"] == domain("iris/scientific-ablation-collapse-policy/v1", p["policy"]),
        "Compact policy hash differs",
    )
    require(
        p["membership_sha256"]
        == domain("iris/scientific-ablation-collapse-members/v1", p["members"]),
        "Original membership digest differs",
    )
    input_payload = p["evaluation"]["payload"]["plan"]["payload"]["input"]["payload"]
    members = original_membership(parquet, input_payload["test_parquet_sha256"], p["members"])
    rows = p["rows"]
    inventory = p["evaluation"]["payload"]["plan"]["payload"]["rows"]
    require(len(rows) == len(p["members"]) == len(d["rows"]), "All original rows required")
    for index, (member, row, full, expected) in enumerate(
        zip(p["members"], rows, d["rows"], inventory, strict=True)
    ):
        trace = full["trace"]
        require(
            all(
                row[name] == full[name]
                for name in (
                    "row_index",
                    "capture_row_sha256",
                    "evaluation_row_sha256",
                    "evaluation_ack_sha256",
                    "teacher_invocation_sha256",
                    "teacher_target_sha256",
                    "target_token_count",
                    "target_nll_sum",
                )
            ),
            "Original score/ack/EOS differs",
        )
        require(
            member["row_index"] == row["row_index"] == index
            and all(
                member[key] == expected[key]
                for key in (
                    "example_id",
                    "example_sha256",
                    "prompt_sha256",
                    "cognitive_prompt_sha256",
                    "task_type",
                )
            ),
            "Capture expectation differs",
        )
        require(
            row["metadata_trace_sha256"]
            == domain("iris/scientific-ablation-collapse-metadata-trace/v1", trace),
            "Full actual metadata digest differs",
        )
        require(
            row["context"]
            == {
                "example_id": member["example_id"],
                "checkpoint_sha256": source["best_checkpoint_sha256"],
                "ablation_policy_sha256": source["ablation_policy_sha256"],
                "seed": p["run_seed"],
                "num_modules": trace["num_modules"],
                "num_slots": trace["num_slots"],
                "state_dim": trace["state_dim"],
                "cycle_count": trace["cycle_count"],
                "maximum_trace_scalars": trace["options"]["maximum_scalars"],
            },
            "Actual non-null mode/best/context differs",
        )
        require(
            (p["num_modules"], p["num_slots"], p["state_dim"], p["cycle_count"])
            == (trace["num_modules"], trace["num_slots"], trace["state_dim"], trace["cycle_count"]),
            "Actual geometry differs",
        )
        require(
            row["cycles"]
            == [
                {name: cycle[name] for name in ("cycle", "write_modules", "selected_slots")}
                for cycle in trace["cycles"]
            ]
            and all(
                len(cycle["write_modules"]) == p["writers_per_cycle"] for cycle in row["cycles"]
            ),
            "Compact cycles/writers differ",
        )
    selectors = [
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    ]
    require(len(p["groups"]) == 20, "Exactly twenty groups required")
    for group, (task, level) in zip(p["groups"], selectors, strict=True):
        selected = [
            row
            for member, row in zip(p["members"], rows, strict=True)
            if (task is None or member["task_type"] == task)
            and (level is None or member["difficulty"] == level)
        ]
        actual = {
            "task_type": task,
            "difficulty": level,
            "examples": len(selected),
            "summary": summary(selected, p["num_modules"], p["num_slots"]),
        }
        require(group == actual, "Twenty independently pooled groups differ")
    criterion = {
        "source_section": "20.4",
        "required_modules": 8,
        "required_slots": 8,
        "minimum_effective_modules": 4,
        "minimum_effective_slots": 4,
        "scope": "one-seed-complete-test-corpus",
        "outcome": "not_assessed_smoke",
    }
    require(p["source_20_4"] == criterion, "Tiny fixture cannot claim science")
    bound = p["cycle_count"] * (
        1
        + 10 * p["num_modules"]
        + 20 * p["num_slots"]
        + p["num_slots"] ** 2
        + p["num_modules"] * p["num_slots"]
    )
    require(
        len(rows) <= min(p["policy"]["maximum_examples"], p["policy"]["maximum_teacher_executions"])
        and bound <= p["policy"]["maximum_trace_scalars"]
        and p["trace_scalar_bound"]
        == len(rows) * bound
        <= p["policy"]["maximum_total_trace_scalars"],
        "Full transient metadata/execution budgets differ",
    )
    return {
        "report": binding(path),
        "ordinary": ordinary_audit,
        "original_membership": members,
        "rows": len(rows),
        "groups": 20,
        "mode": source["ablation"],
        "compact_full_trace_bindings": True,
    }


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before = load(before_path)
    require(
        pin(root, before["previous"]["path"]) == before["previous"], "Predecessor snapshot changed"
    )
    for item in (*before["bindings"], *before["auditors"]):
        require(pin(root, item["path"]) == item, "Gate source/auditor changed")
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(case) for case in cases),
        "Exact passing JUnit IDs differ",
    )
    captures: dict[tuple[str, str], Path] = {}
    for store in base.rglob("scientific-ablation-capture-v1"):
        for candidate in store.iterdir():
            if candidate.is_dir() and not candidate.name.startswith("."):
                try:
                    captures.setdefault(valid_capture(candidate), candidate)
                except ValueError:
                    continue
    parquets = {binding(path)["sha256"]: path for path in base.rglob("test.parquet")}
    stores = list(base.rglob("scientific-ablation-evaluation-ledger-v2"))
    verified: dict[str, tuple[Path, dict[str, Any]]] = {}
    reports = []
    for path in sorted(base.glob("*/collapse.json")):
        p = manifest(path, "iris/scientific-ablation-collapse/v1")["payload"]
        plan_sha = p["evaluation_plan"]["sha256"]
        if plan_sha not in verified:
            for store in stores:
                candidate = store / plan_sha
                if candidate.is_dir():
                    try:
                        verified[plan_sha] = candidate, chain(candidate, captures)
                        break
                    except ValueError:
                        continue
        require(plan_sha in verified, "Valid actual physical capture/score chain missing")
        parquet_sha = p["evaluation"]["payload"]["plan"]["payload"]["input"]["payload"][
            "test_parquet_sha256"
        ]
        require(parquet_sha in parquets, "Original physical test Parquet missing")
        reports.append(compact(path, verified[plan_sha][0], parquets[parquet_sha]))
    require(
        len(reports) == 13 and any(report["rows"] == 4 for report in reports),
        "Actual all12 plus four-family reports missing",
    )
    require(
        {(r["mode"]["routing"], r["mode"]["goal"], r["mode"]["broadcast"]) for r in reports}
        == {
            (r, g, b)
            for r in ("learned", "random", "uniform")
            for g in ("enabled", "disabled")
            for b in ("enabled", "disabled")
        },
        "Actual all12 modes missing",
    )
    for item in (*before["bindings"], *before["auditors"]):
        require(pin(root, item["path"]) == item, "Source/auditor drift during readback")
    return {
        "kind": "iris-r015-ablation-collapse-readback-v1",
        "passed": True,
        "before": binding(before_path),
        "junit": binding(junit),
        "exact_cases": len(ids),
        "source_pins": len(before["bindings"]),
        "auditor_pins": len(before["auditors"]),
        "reports": reports,
        "physical_chains": [item[1] for item in verified.values()],
        "all12_c3_earlier_best": True,
        "all20_original_input_bound_groups": True,
        "full_r015_complete": False,
        "full_iris_gw_01_complete": False,
        "limits": [
            "No Iris/Torch/pytest imports or model execution in audit",
            "PyArrow reads original Parquet; only hashes/counts leave observer",
            "Integrity readback is not execution attestation/scientific acceptance/CUDA",
            "No deletion/download/install/publication; exclusive new observer receipt only",
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
            name == "iris" or name.startswith("iris.") or name in ("torch", "pytest")
            for name in sys.modules
        ),
        "Observer imported project/model/test code",
    )
    with args.output.open("xb") as output:
        output.write(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(args.output),
                "receipt": binding(args.output),
                "cases": result.get("expected_cases", result.get("exact_cases")),
                "passed": result.get("passed"),
            }
        )
    )


if __name__ == "__main__":
    main()
