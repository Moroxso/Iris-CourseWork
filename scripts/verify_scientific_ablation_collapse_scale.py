"""Independent stdlib full5000 FORMAT readback, NOT teacher/sandbox attestation.

Only snapshot invokes an explicit pytest collection child. Audit reads raw
JSON/XML and separately pinned older stdlib helpers; no Iris/Torch/PyArrow/pytest.
Optional NEW exclusive output is an observer receipt, never report publication.
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
from verify_scientific_ablation_evaluation_ledger import framed, pin

_PREVIOUS = "artifacts/operational/r015-collapse-before-20261012-run01.json"
_PREVIOUS_SHA = "888cacafab415290bc707b05451d7d87bca0a294d94742f52c4b143e6ab9a6a9"
_TEST = "tests/integration/test_scientific_ablation_collapse_scale.py"
_AUDITOR = "scripts/verify_scientific_ablation_collapse_scale.py"
MODULES = (
    _TEST,
    "tests/unit/test_scientific_ablation_collapse.py",
    "tests/unit/test_scientific_collapse.py",
)
TASKS = ("split_spec_repair", "delayed_goal", "conflict_resolution", "next_tool_prediction")
_ACK = "iris/scientific-ablation-evaluation-ack/v2"
_CAP = 64 * 1024**2


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite))


def unwrap(value: dict[str, Any], name: str) -> dict[str, Any]:
    require(
        set(value) == {"sha256", "payload"} and value["sha256"] == domain(name, value["payload"]),
        "Synthetic format manifest domain differs",
    )
    return dict(value["payload"])


def snapshot(root: Path) -> dict[str, Any]:
    previous_pin = pin(root, _PREVIOUS)
    require(previous_pin["sha256"] == _PREVIOUS_SHA, "Accepted count snapshot changed")
    previous = load(root / _PREVIOUS)
    require(
        len(previous["bindings"]) == 531 and len(previous["auditors"]) == 4,
        "Accepted count scope missing",
    )
    for item in (*previous["bindings"], *previous["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted source/auditor changed")
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
    require(len(ids) == len(set(ids)) == 80, "Exact80 scale/count compatibility IDs required")
    return {
        "kind": "iris-r015-collapse-scale-before-v1",
        "previous": previous_pin,
        "bindings": [*previous["bindings"], pin(root, _TEST)],
        "auditors": [*previous["auditors"], pin(root, _AUDITOR)],
        "modules": MODULES,
        "expected_cases": len(ids),
        "nodeids": ids,
        "actual_collection_exit_code": collected.returncode,
        "actual_collection_stdout_bytes": len(collected.stdout),
        "actual_collection_stdout_sha256": hashlib.sha256(collected.stdout).hexdigest(),
        "production_changed": False,
        "scientific_acceptance": "not_assessed",
    }


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    wins, slots = [0] * 8, [0] * 8
    for row in rows:
        for number, cycle in enumerate(row["cycles"], 1):
            require(
                cycle["cycle"] == number
                and len(cycle["write_modules"]) == len(cycle["selected_slots"]) == 2
                and cycle["write_modules"] == sorted(cycle["write_modules"]),
                "Synthetic cycle/writer order differs",
            )
            for indices, counts in (
                (cycle["write_modules"], wins),
                (cycle["selected_slots"], slots),
            ):
                require(
                    len(set(indices)) == 2
                    and all(type(index) is int and 0 <= index < 8 for index in indices),
                    "Synthetic hard allocation indices differ",
                )
                for index in indices:
                    counts[index] += 1

    def entropy(counts: list[int]) -> tuple[float, float]:
        total = sum(counts)
        value = -math.fsum((count / total) * math.log(count / total) for count in counts if count)
        return value, min(8.0, math.exp(value))

    mh, me = entropy(wins)
    sh, se = entropy(slots)
    return {
        "examples": len(rows),
        "cycles": 3 * len(rows),
        "allocated_writes": 6 * len(rows),
        "module_win_counts": wins,
        "slot_allocation_counts": slots,
        "module_entropy_nats": mh,
        "slot_entropy_nats": sh,
        "effective_active_modules": me,
        "effective_active_slots": se,
    }


def scale(path: Path) -> dict[str, Any]:
    compact = manifest(path, "iris/scientific-ablation-collapse/v1")
    ordinary_path = path.with_name("synthetic-ordinary.json")
    ordinary = manifest(ordinary_path, "iris/scientific-ablation-diagnostics/v1")
    metrics_path = path.with_name("envelope.json")
    input_bindings = [binding(item) for item in (path, ordinary_path, metrics_path)]
    metrics = load(metrics_path)
    p, d = compact["payload"], ordinary["payload"]
    require(
        metrics["kind"] == "iris-synthetic-5000-collapse-envelope-v1"
        and metrics["teacher_forwards_for_expanded_rows"] == 0
        and metrics["full_capture_physical_execution_proof"] is False
        and metrics["scientific_acceptance"] == "not_assessed",
        "Synthetic witnesses cannot claim actual experiment execution",
    )
    require(
        p["scientific_acceptance"] == d["scientific_acceptance"] == "not_assessed",
        "Fixture claims launch approval",
    )
    require(
        metrics["compact_bytes"] == binding(path)["bytes"] <= _CAP
        and metrics["ordinary_metadata_bytes"] == binding(ordinary_path)["bytes"] <= _CAP
        and metrics["ordinary_fits"] is True
        and metrics["cap_bytes"] == _CAP,
        "Actual canonical format byte envelope differs",
    )
    require(
        p["ordinary_diagnostics_sha256"] == ordinary["sha256"]
        and all(
            p[name] == d[name]
            for name in (
                "evaluation_plan",
                "evaluation",
                "evaluation_complete",
                "teacher_graph_sha256",
                "run_seed",
                "teacher_executions",
                "trace_scalar_bound",
            )
        ),
        "Full synthetic ordinary/compact inventory differs",
    )
    plan = unwrap(p["evaluation_plan"], "iris/scientific-ablation-evaluation-plan/v2")
    report = unwrap(p["evaluation"], "iris/scientific-ablation-evaluation/v2")
    complete = unwrap(p["evaluation_complete"], "iris/scientific-ablation-evaluation-complete/v2")
    generation = unwrap(report["plan"], "iris/scientific-ablation-capture-plan/v1")
    capture = unwrap(report["capture"], "iris/scientific-ablation-capture-complete/v1")
    source = unwrap(generation["source"], "iris/scientific-ablation-generation-source/v1/payload")
    reconstruction = unwrap(
        generation["reconstruction"],
        "iris/scientific-ablation-generation-reconstruction/v1/payload",
    )
    heldout = unwrap(generation["input"], "iris/scientific-ablation-heldout-input/v1")
    require(
        plan["generation_plan"] == report["plan"]
        and plan["capture"] == report["capture"]
        and source["purpose"] == heldout["purpose"] == "scientific"
        and source["training_examples"] == 50000
        and source["validation_examples"] == 5000
        and reconstruction["source_sha256"]
        == heldout["source_sha256"]
        == generation["source"]["sha256"]
        and heldout["reconstruction_sha256"] == generation["reconstruction"]["sha256"],
        "Synthetic full-source-shaped headers differ",
    )
    require(
        (source["selected_step"], source["terminal_step"]) == (1, 3)
        and p["teacher_graph_sha256"] == generation["teacher_graph_sha256"]
        and p["policy_sha256"] == domain("iris/scientific-ablation-collapse-policy/v1", p["policy"])
        and p["membership_sha256"]
        == domain("iris/scientific-ablation-collapse-members/v1", p["members"]),
        "Full synthetic identities differ",
    )
    require(
        len(p["members"])
        == len(p["rows"])
        == len(d["rows"])
        == len(report["rows"])
        == len(plan["rows"])
        == len(generation["rows"])
        == p["teacher_executions"]
        == heldout["example_count"]
        == 5000
        and p["trace_scalar_bound"] == 5_535_000,
        "Complete5000 format coverage/scalars differs",
    )
    require(
        (p["num_modules"], p["num_slots"], p["state_dim"], p["cycle_count"], p["writers_per_cycle"])
        == (8, 8, 256, 3, 2),
        "Standard source geometry differs",
    )
    for name in ("example_id", "example_sha256", "prompt_sha256", "cognitive_prompt_sha256"):
        require(
            len({member[name] for member in p["members"]}) == 5000,
            "Synthetic full5000 unique membership differs",
        )
    head = framed(_ACK + "/genesis", p["evaluation_plan"]["sha256"].encode())
    for index, (member, row, traced, score, expected, score_expected) in enumerate(
        zip(
            p["members"],
            p["rows"],
            d["rows"],
            report["rows"],
            generation["rows"],
            plan["rows"],
            strict=True,
        )
    ):
        require(
            member["row_index"]
            == row["row_index"]
            == score["row_index"]
            == expected["row_index"]
            == score_expected["row_index"]
            == index
            and member["example_id"]
            == expected["example_id"]
            == score["example_id"]
            == f"test-{index}"
            and member["difficulty"] in (1, 2, 3),
            "Full5000 ordered membership differs",
        )
        require(
            all(
                member[name] == expected[name]
                for name in (
                    "example_sha256",
                    "prompt_sha256",
                    "cognitive_prompt_sha256",
                    "task_type",
                )
            )
            and score["generation_example_sha256"] == member["example_sha256"]
            and score["task_type"] == member["task_type"],
            "Synthetic capture/member hashes differ",
        )
        record_payload = {
            "kind": "iris-scientific-ablation-evaluation-row-v2",
            "plan_sha256": p["evaluation_plan"]["sha256"],
            "measurements": score,
        }
        record_sha = domain("iris/scientific-ablation-evaluation-row/v2", record_payload)
        content = canonical({"sha256": record_sha, "payload": record_payload}) + b"\n"
        ack = {
            "kind": "iris-scientific-ablation-evaluation-ack-v2",
            "status": "acknowledged_score",
            "plan_sha256": p["evaluation_plan"]["sha256"],
            "row_index": index,
            "previous_ack_sha256": head,
            "row_sha256": record_sha,
            "row_file_sha256": hashlib.sha256(content).hexdigest(),
            "row_file_bytes": len(content),
        }
        head = domain(_ACK, ack)
        require(
            row["evaluation_row_sha256"] == record_sha
            and row["evaluation_ack_sha256"] == head
            and all(
                row[name] == score[name] == traced[name]
                for name in (
                    "capture_row_sha256",
                    "teacher_invocation_sha256",
                    "teacher_target_sha256",
                    "target_token_count",
                    "target_nll_sum",
                )
            ),
            "Full5000 canonical score-file/ack/EOS format chain differs",
        )
        require(
            score_expected["capture_row_sha256"] == score["capture_row_sha256"]
            and score_expected["capture_ack_sha256"] == score["capture_ack_sha256"]
            and score_expected["target_nll_sum"] == score["target_nll_sum"],
            "Synthetic score-plan expectation differs",
        )
        trace = traced["trace"]
        require(
            row["metadata_trace_sha256"]
            == domain("iris/scientific-ablation-collapse-metadata-trace/v1", trace)
            and row["cycles"]
            == [
                {name: cycle[name] for name in ("cycle", "write_modules", "selected_slots")}
                for cycle in trace["cycles"]
            ],
            "Full synthetic trace/compact pair differs",
        )
        require(
            row["context"]
            == {
                "example_id": member["example_id"],
                "checkpoint_sha256": source["best_checkpoint_sha256"],
                "ablation_policy_sha256": source["ablation_policy_sha256"],
                "seed": p["run_seed"],
                "num_modules": 8,
                "num_slots": 8,
                "state_dim": 256,
                "cycle_count": 3,
                "maximum_trace_scalars": p["policy"]["maximum_trace_scalars"],
            }
            and trace["intervention_plan_sha256"] == source["ablation_policy_sha256"],
            "Full synthetic mode/best context differs",
        )
        require(
            [sum(cycle["read_slot_occupancy"]) for cycle in trace["cycles"]] == [0, 2, 4]
            and [sum(cycle["slot_occupancy"]) for cycle in trace["cycles"]] == [2, 4, 6],
            "Envelope is not the stated empty-initialized occupancy shape",
        )
    require(
        complete["acknowledgement_head_sha256"] == head
        and complete["evaluation_sha256"] == p["evaluation"]["sha256"]
        and complete["plan_sha256"] == p["evaluation_plan"]["sha256"]
        and complete["example_count"] == capture["example_count"] == 5000
        and heldout["target_token_count"]
        == capture["target_token_count"]
        == sum(score["target_token_count"] for score in report["rows"])
        and capture["target_nll_sum"]
        == math.fsum(score["target_nll_sum"] for score in report["rows"]),
        "Full5000 complete head/NLL/token totals differ",
    )
    selectors = [
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    ]
    require(
        len(p["groups"]) == 20
        and [sum(member["task_type"] == task for member in p["members"]) for task in TASKS]
        == [2000, 1000, 1000, 1000],
        "Full scientific-shaped task coverage differs",
    )
    for group, (task, level) in zip(p["groups"], selectors, strict=True):
        selected = [
            row
            for member, row in zip(p["members"], p["rows"], strict=True)
            if (task is None or member["task_type"] == task)
            and (level is None or member["difficulty"] == level)
        ]
        require(
            group
            == {
                "task_type": task,
                "difficulty": level,
                "examples": len(selected),
                "summary": pooled(selected),
            },
            "Full5000 twenty independently pooled groups differ",
        )
    require(
        p["source_20_4"]
        == {
            "source_section": "20.4",
            "required_modules": 8,
            "required_slots": 8,
            "minimum_effective_modules": 4,
            "minimum_effective_slots": 4,
            "scope": "one-seed-complete-test-corpus",
            "outcome": "met",
        },
        "Fixed format gate differs",
    )
    overall = pooled(p["rows"])
    require(
        overall is not None
        and overall["effective_active_modules"] >= 4
        and overall["effective_active_slots"] >= 4,
        "Synthetic fixed outcome differs",
    )
    require(
        [binding(item) for item in (path, ordinary_path, metrics_path)] == input_bindings,
        "Scale inputs changed during complete raw readback",
    )
    return {
        "compact": binding(path),
        "ordinary": binding(ordinary_path),
        "metrics": binding(metrics_path),
        "synthetic_examples": 5000,
        "expanded_teacher_forwards": 0,
        "canonical_score_ack_records_checked": 5000,
        "all20_groups": True,
        "format_byte_envelope_passed": True,
        "execution_attestation": False,
        "scientific_acceptance": "not_assessed",
    }


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_binding, junit_binding = binding(before_path), binding(junit)
    before = load(before_path)
    previous = load(root / _PREVIOUS)
    require(
        before["kind"] == "iris-r015-collapse-scale-before-v1"
        and before["previous"]["path"] == _PREVIOUS
        and before["previous"]["sha256"] == _PREVIOUS_SHA
        and before["bindings"] == [*previous["bindings"], pin(root, _TEST)]
        and before["auditors"] == [*previous["auditors"], pin(root, _AUDITOR)]
        and len(before["bindings"]) == 532
        and len(before["auditors"]) == 5
        and before["modules"] == list(MODULES)
        and before["expected_cases"] == 80
        and before["actual_collection_exit_code"] == 0
        and before["production_changed"] is False,
        "Exact accepted source plus new FORMAT observer scope required",
    )
    require(
        pin(root, before["previous"]["path"]) == before["previous"], "Previous snapshot changed"
    )
    for item in (*before["bindings"], *before["auditors"]):
        require(pin(root, item["path"]) == item, "Scale source/auditor changed")
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(case) for case in cases),
        "Exact passing selected JUnit IDs differ",
    )
    reports = list(base.glob("*/synthetic-collapse.json"))
    require(len(reports) == 1, "Exactly one full5000 synthetic report pair required")
    result = scale(reports[0])
    for item in (*before["bindings"], *before["auditors"]):
        require(pin(root, item["path"]) == item, "Source/auditor drift during readback")
    require(
        binding(before_path) == before_binding and binding(junit) == junit_binding,
        "Raw before/JUnit changed during scale observation",
    )
    return {
        "kind": "iris-r015-collapse-scale-readback-v1",
        "passed": True,
        "before": binding(before_path),
        "junit": binding(junit),
        "exact_cases": len(ids),
        "source_pins": len(before["bindings"]),
        "auditor_pins": len(before["auditors"]),
        "scale": result,
        "full_r015_complete": False,
        "full_iris_gw_01_complete": False,
        "limits": [
            "Synthetic full5000 FORMAT/byte/ack-chain feasibility only; "
            "not actual scientific capture or neural execution",
            "Audit imports no Iris/Torch/PyArrow/pytest; snapshot collects pytest in a child",
            "No install/download/CUDA/WSL/candidate/deletion; new observer receipt only",
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
            "Explicit scale inputs required",
        )
        result = audit(args.workspace.resolve(), args.before, args.junit, args.base)
    require(
        not any(
            name == "iris"
            or name.startswith("iris.")
            or name in ("torch", "pytest", "pyarrow")
            or name.startswith("pyarrow.")
            for name in sys.modules
        ),
        "Scale observer imported project/model/codec/test code",
    )
    with args.output.open("xb") as output:
        output.write(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "receipt": binding(args.output),
                "cases": result.get("expected_cases", result.get("exact_cases")),
                "passed": result.get("passed"),
            }
        )
    )


if __name__ == "__main__":
    main()
