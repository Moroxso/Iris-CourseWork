"""Read-only independent pins/JUnit/raw score-chain audit; no Iris/Torch imports."""

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_scientific_ablation_evaluation_io import (
    binding,
    canonical,
    manifest,
    pairs,
    require,
)

_NEW = (
    "src/iris/application/scientific_ablation_evaluation_ledger.py",
    "src/iris/artifacts/scientific_ablation_evaluation_ledger_v2.py",
    "tests/integration/test_scientific_ablation_evaluation_ledger.py",
    "tests/integration/test_scientific_ablation_evaluation_ledger_integrity.py",
    "tests/integration/test_scientific_ablation_evaluation_ledger_process.py",
    "tests/integration/_scientific_ablation_evaluation_worker.py",
)
_CHANGED = "src/iris/application/scientific_ablation_evaluation_v2.py"
_IO = "artifacts/operational/r015-evaluation-io-before-20261012-run03.json"
_ACK = "iris/scientific-ablation-evaluation-ack/v2"


def load(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    return result


def pin(root: Path, relative: str) -> dict[str, Any]:
    result = binding(root / relative)
    result["path"] = relative
    return result


def snapshot(root: Path, collection: Path) -> dict[str, Any]:
    prior = load(root / _IO)
    old = prior["prior_snapshot"]
    require(pin(root, old["path"]) == old, "Original516 snapshot binding changed")
    original = [*load(root / old["path"])["bindings"], *prior["new_bindings"]]
    require(len(original) == 518, "Exact old accepted518 scope missing")
    current = []
    changes = []
    for previous in original:
        actual = pin(root, previous["path"])
        if actual != previous:
            require(previous["path"] == _CHANGED, "Unrelated accepted source drift")
            changes.append({"previous": previous, "current": actual})
        current.append(actual)
    require(len(changes) == 1, "Exactly the shared scorer is expected to change")
    current.extend(pin(root, relative) for relative in _NEW)
    ids = load(collection)["nodeids"]
    require(len(ids) == len(set(ids)) and len(ids) > 60, "Actual distinct collection missing")
    return {
        "kind": "iris-r015-score-ledger-before-v2",
        "prior518_snapshot": pin(root, _IO),
        "allowed_changed_original_scorer": changes,
        "bindings": current,
        "binding_count": len(current),
        "auditors": [
            pin(root, "scripts/verify_scientific_ablation_evaluation_io.py"),
            pin(root, "scripts/verify_scientific_ablation_evaluation_ledger.py"),
        ],
        "collected_case_ids": ids,
        "collection": binding(collection),
        "expected_cases": len(ids),
        "full_r015_complete": False,
    }


def framed(name: str, content: bytes) -> str:
    value = hashlib.sha256()
    for part in (name.encode(), content):
        value.update(len(part).to_bytes(8, "big"))
        value.update(part)
    return value.hexdigest()


def genesis(plan: str) -> str:
    return framed(_ACK + "/genesis", plan.encode())


def valid_capture(path: Path) -> tuple[str, str]:
    generation = manifest(path / "plan.json", "iris/scientific-ablation-capture-plan/v1")
    complete = manifest(
        path / "complete" / "complete.json", "iris/scientific-ablation-capture-complete/v1"
    )
    head = framed(
        "iris/scientific-ablation-capture-row-ack/v1/genesis", generation["sha256"].encode()
    )
    for index in range(len(generation["payload"]["rows"])):
        directory = path / f"row-{index:05d}"
        row = manifest(directory / "row.json", "iris/scientific-ablation-heldout-row/v1")
        ack = manifest(directory / "ack.json", "iris/scientific-ablation-capture-row-ack/v1")
        data = binding(directory / "row.json")
        require(
            ack["payload"]["previous_ack_sha256"] == head
            and ack["payload"]["row_sha256"] == row["sha256"]
            and ack["payload"]["row_file_sha256"] == data["sha256"]
            and ack["payload"]["row_file_bytes"] == data["bytes"],
            "Original private capture physical chain differs",
        )
        head = ack["sha256"]
    require(
        complete["payload"]["acknowledgement_head_sha256"] == head
        and complete["payload"]["plan_sha256"] == generation["sha256"],
        "Actual original capture head differs",
    )
    return generation["sha256"], complete["sha256"]


def process_proof(path: Path, *, isolated: bool) -> dict[str, Any]:
    first = load(path / "hard-exit.stdout.txt")
    cold = load(path / "cold-resume.stdout.txt")
    retry = load(path / "complete-retry.stdout.txt")
    require(
        first["kind"] == "iris-score-worker-hard-exit-v2"
        and first["scored_indices"] == [0]
        and first["committed_row"] == "score-00000",
        "Actual hard-exit row0 observation differs",
    )
    require(
        cold["kind"] == "iris-score-worker-complete-v2"
        and cold["scored_indices"] == [1, 2, 3]
        and cold["sandbox_indices"] == cold["linux_phases"] == [],
        "Cold process reexecuted acknowledged sandbox work",
    )
    require(
        first["pid"] != cold["pid"]
        and cold["caller_restored"]
        and not cold["model_constructed"]
        and not cold["host_candidate_execution"],
        "Cold model-free caller/process scope differs",
    )
    require(
        retry["scored_indices"] == retry["sandbox_indices"] == retry["linux_phases"] == []
        and retry["evaluation_sha256"] == cold["evaluation_sha256"],
        "Complete retry repeats work",
    )
    require(
        first["sandbox_indices"] == ([0] if isolated else []),
        "Actual original isolated invocation count differs",
    )
    if isolated:
        phases = [event["phase"] for event in first["linux_phases"]]
        require(
            "create" in phases
            and "rm" in phases
            and all(event["passed"] for event in first["linux_phases"]),
            "Actual original sandbox creation/owned cleanup missing",
        )
    else:
        require(first["linux_phases"] == [], "Non-code worker submitted Linux work")
    return {
        "isolated": isolated,
        "hard_exit": binding(path / "hard-exit.stdout.txt"),
        "cold_resume": binding(path / "cold-resume.stdout.txt"),
        "complete_retry": binding(path / "complete-retry.stdout.txt"),
        "original_linux_calls": len(first["linux_phases"]),
        "sandbox_jobs_before_death": len(first["sandbox_indices"]),
        "sandbox_jobs_after_death": 0,
    }


def chain(path: Path, capture_roots: dict[tuple[str, str], Path]) -> dict[str, Any]:
    plan = manifest(path / "plan.json", "iris/scientific-ablation-evaluation-plan/v2")
    payload = plan["payload"]
    generation, captured = payload["generation_plan"], payload["capture"]
    key = (generation["sha256"], captured["sha256"])
    require(key in capture_roots, "Actual original capture not present")
    capture = capture_roots[key]
    require(
        manifest(capture / "plan.json", "iris/scientific-ablation-capture-plan/v1") == generation,
        "Original generation plan differs",
    )
    require(
        manifest(
            capture / "complete" / "complete.json", "iris/scientific-ablation-capture-complete/v1"
        )
        == captured,
        "Original complete capture differs",
    )
    report = manifest(
        path / "complete" / "evaluation.json", "iris/scientific-ablation-evaluation/v2"
    )
    complete = manifest(
        path / "complete" / "complete.json", "iris/scientific-ablation-evaluation-complete/v2"
    )
    require(
        report["payload"]["plan"] == generation and report["payload"]["capture"] == captured,
        "Report differs from original capture",
    )
    require(path.name == plan["sha256"], "Score plan path differs")
    head = genesis(plan["sha256"])
    scores = []
    raw_pins = [
        binding(path / "plan.json"),
        binding(path / "complete" / "evaluation.json"),
        binding(path / "complete" / "complete.json"),
    ]
    for index, expectation in enumerate(payload["rows"]):
        directory = path / f"score-{index:05d}"
        require(
            {item.name for item in directory.iterdir()} == {"attempt.json", "row.json", "ack.json"},
            "Committed score members differ",
        )
        attempt = load(directory / "attempt.json")
        require(
            attempt
            == {
                "kind": "iris-scientific-ablation-score-attempt-v2",
                "plan_sha256": plan["sha256"],
                "row_index": index,
                "previous_ack_sha256": head,
                "status": "owned_before_scoring",
            },
            "Actual owned attempt differs",
        )
        require(
            canonical(attempt) + b"\n" == (directory / "attempt.json").read_bytes(),
            "Attempt noncanonical",
        )
        row = manifest(directory / "row.json", "iris/scientific-ablation-evaluation-row/v2")
        ack = manifest(directory / "ack.json", _ACK)
        raw = binding(directory / "row.json")
        require(
            ack["payload"]
            == {
                "kind": "iris-scientific-ablation-evaluation-ack-v2",
                "status": "acknowledged_score",
                "plan_sha256": plan["sha256"],
                "row_index": index,
                "previous_ack_sha256": head,
                "row_sha256": row["sha256"],
                "row_file_sha256": raw["sha256"],
                "row_file_bytes": raw["bytes"],
            },
            "Physical acknowledged score chain differs",
        )
        actual_row = manifest(
            capture / f"row-{index:05d}" / "row.json", "iris/scientific-ablation-heldout-row/v1"
        )
        actual_ack = manifest(
            capture / f"row-{index:05d}" / "ack.json", "iris/scientific-ablation-capture-row-ack/v1"
        )
        score = row["payload"]["measurements"]
        require(
            row["payload"]["plan_sha256"] == plan["sha256"] and score["row_index"] == index,
            "Score job/index differs",
        )
        require(
            score["capture_row_sha256"] == actual_row["sha256"] == expectation["capture_row_sha256"]
            and score["capture_ack_sha256"]
            == actual_ack["sha256"]
            == expectation["capture_ack_sha256"],
            "Raw original row/ack differs",
        )
        require(
            score["target_nll_sum"]
            == actual_row["payload"]["target_nll_sum"]
            == expectation["target_nll_sum"]
            and score["target_token_count"] == actual_row["payload"]["target_token_count"],
            "Original actual NLL was not copied",
        )
        output = actual_row["payload"]["output_text"].encode()
        require(
            score["output_sha256"]
            == hashlib.sha256(output).hexdigest()
            == expectation["output_sha256"]
            and score["output_byte_length"] == len(output),
            "Actual captured answer digest differs",
        )
        if score["execution"] is not None:
            require(
                score["execution"]["policy"] == payload["python_policy"]
                and score["execution"]["worker_sha256"] == payload["worker_sha256"],
                "Original source-v2 policy/worker differs",
            )
        scores.append(score)
        raw_pins.extend(
            [
                raw,
                binding(directory / "attempt.json"),
                binding(directory / "ack.json"),
                binding(capture / f"row-{index:05d}" / "row.json"),
                binding(capture / f"row-{index:05d}" / "ack.json"),
            ]
        )
        head = ack["sha256"]
    require(report["payload"]["rows"] == scores, "Report differs from actual score prefix")
    require(
        report["payload"]["overall"]["target_nll_sum"]
        == math.fsum(row["target_nll_sum"] for row in scores),
        "Exact NLL aggregation differs",
    )
    require(
        complete["payload"]
        == {
            "kind": "iris-scientific-ablation-evaluation-complete-v2",
            "status": "complete_not_scientific_launch_approval",
            "plan_sha256": plan["sha256"],
            "capture_sha256": captured["sha256"],
            "acknowledgement_head_sha256": head,
            "evaluation_sha256": report["sha256"],
            "example_count": len(scores),
        },
        "Complete score/report chain differs",
    )
    require(
        not any(
            b'"' + key + b'"' in canonical(report) or b'"' + key + b'"' in canonical(plan)
            for key in (b"output_text", b"expected_response", b"hidden_facts", b"prompt")
        ),
        "Public score private data disclosure",
    )
    source = generation["payload"]["source"]["payload"]
    require(
        source["selected_step"] == 1
        and source["terminal_step"] == 3
        and generation["payload"]["cycles"] == 3,
        "Actual best1/terminal3/C3 differs",
    )
    return {
        "path": str(path),
        "mode": source["ablation"],
        "rows": len(scores),
        "report_sha256": report["sha256"],
        "raw_bindings": raw_pins,
    }


def audit(root: Path, before_path: Path, junit: Path, bases: list[Path]) -> dict[str, Any]:
    before = load(before_path)
    require(before["binding_count"] == 524, "Expected exact current524 source pins")
    for item in [*before["bindings"], *before["auditors"]]:
        require(pin(root, item["path"]) == item, "Actual frozen source/auditor changed")
    require(pin(root, _IO) == before["prior518_snapshot"], "Prior frozen history differs")
    tests = list(ET.parse(junit).getroot().iter("testcase"))
    require(not any(list(case) for case in tests), "Runtime has failure/error/skip")
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in tests
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["collected_case_ids"]),
        "Actual exact JUnit scope differs from collection",
    )
    captures: dict[tuple[str, str], Path] = {}
    for base in bases:
        for store in base.rglob("scientific-ablation-capture-v1"):
            for path in store.glob("*/plan.json"):
                end = path.parent / "complete" / "complete.json"
                if end.is_file():
                    try:
                        key = valid_capture(path.parent)
                    except (ValueError, OSError, KeyError):
                        # Adversarial tests intentionally retain corrupted copies.
                        # They cannot be selected as independent source evidence.
                        continue
                    captures[key] = path.parent
    results = []
    processes = []
    modes = set()
    for base in bases:
        for fixture in base.iterdir():
            if not fixture.is_dir() or not fixture.name.startswith(
                (
                    "test_actual_all12_c3_earlie",
                    "test_partial_resume_exact_retr",
                    "test_real_hard_exit_os_rele",
                    "test_real_gvisor_ack_surviv",
                )
            ):
                continue
            paths = list(
                fixture.rglob("scientific-ablation-evaluation-ledger-v2/*/complete/evaluation.json")
            )
            if not paths and fixture.name.startswith("test_actual_all12"):
                continue  # Same truncated pytest prefix in old, separate report-IO tests.
            require(len(paths) == 1, "Actual successful fixture ledger ambiguous/missing")
            result = chain(paths[0].parent.parent, captures)
            results.append(result)
            if fixture.name.startswith(
                ("test_real_hard_exit_os_rele", "test_real_gvisor_ack_surviv")
            ):
                processes.append(
                    process_proof(fixture, isolated=fixture.name.startswith("test_real_gvisor"))
                )
            if fixture.name.startswith("test_actual_all12"):
                policy = result["mode"]
                mode = (policy["routing"], policy["goal"], policy["broadcast"])
                require(mode not in modes, "Duplicate actual persisted mode")
                modes.add(mode)
    expected = {
        (r, g, b)
        for r in ("learned", "random", "uniform")
        for g in ("enabled", "disabled")
        for b in ("enabled", "disabled")
    }
    require(
        modes == expected and len(results) >= 17, "Actual all12/partial/process evidence missing"
    )
    require(
        len(processes) == 4 and sum(item["isolated"] for item in processes) == 3,
        "Actual non-code plus all3 gVisor process proof missing",
    )
    return {
        "kind": "iris-r015-score-ledger-independent-readback-v2",
        "passed": True,
        "source_pins_unchanged": 524,
        "exact_runtime_ids": len(ids),
        "before": binding(before_path),
        "junit": binding(junit),
        "actual_raw_chains": results,
        "actual_cold_process_proofs": processes,
        "limits": [
            "Local tiny CPU engineering, not installed/scientific acceptance",
            "Read-only stdlib audit; no Iris/Torch/source imports or execution",
            "gVisor phase assertions require separately executed controlled process cases",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "snapshot":
        require(args.collection is not None, "Collection required")
        result = snapshot(args.workspace, args.collection)
    else:
        require(args.before is not None and args.junit is not None, "Before/JUnit required")
        result = audit(args.workspace, args.before, args.junit, args.base)
    if args.output is None:
        print(
            json.dumps(
                result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            )
        )
    else:
        # Produce a fresh evidence receipt; never mutate any audited input/history.
        with args.output.open("xb") as file:
            file.write(canonical(result) + b"\n")
        print(
            json.dumps(
                {
                    "kind": result["kind"],
                    "output": binding(args.output),
                    "passed": result.get("passed"),
                    "binding_count": result.get("binding_count"),
                    "expected_cases": result.get("expected_cases"),
                    "exact_runtime_ids": result.get("exact_runtime_ids"),
                }
            )
        )


if __name__ == "__main__":
    main()
