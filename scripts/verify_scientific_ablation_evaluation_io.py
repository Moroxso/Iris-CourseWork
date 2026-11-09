"""Independent read-only JUnit/pin/raw-report audit; no Iris/Torch/pytest imports."""

import argparse
import hashlib
import json
import math
import re
import stat
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and not getattr(metadata, "st_file_attributes", 0) & 0x400,
        "Regular nonlinked audit input required",
    )
    data = path.read_bytes()
    require(len(data) == metadata.st_size, "Audit input size changed")
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def domain(name: str, value: Any) -> str:
    digest = hashlib.sha256()
    for part in (name.encode(), canonical(value)):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "Duplicate audit JSON field")
        result[key] = value
    return result


def nonfinite(value: str) -> Any:
    raise ValueError(f"Nonfinite audit JSON: {value}")


def manifest(path: Path, name: str) -> dict[str, Any]:
    binding(path)
    content = path.read_bytes()
    result: dict[str, Any] = json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(canonical(result) + b"\n" == content, "Raw canonical artifact differs")
    require(set(result) == {"payload", "sha256"}, "Artifact wrapper differs")
    require(
        result["sha256"] == domain(name, result["payload"]), "Independent artifact domain differs"
    )
    return result


def audit(workspace: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before: dict[str, Any] = json.loads(before_path.read_bytes(), object_pairs_hook=pairs)
    require(before["prior516_unchanged"] is True, "Prior scorer bindings were not checked")
    prior_pin = before["prior_snapshot"]
    prior_path = workspace / prior_pin["path"]
    require(
        prior_path.resolve().is_relative_to(workspace.resolve()), "Prior snapshot escapes workspace"
    )
    prior_actual = binding(prior_path)
    require(
        prior_actual["bytes"] == prior_pin["bytes"]
        and prior_actual["sha256"] == prior_pin["sha256"],
        "Frozen predecessor snapshot differs",
    )
    prior: dict[str, Any] = json.loads(prior_path.read_bytes(), object_pairs_hook=pairs)
    require(len(prior["bindings"]) == 516, "Exact predecessor binding scope differs")
    pins = [*prior["bindings"], *before["new_bindings"]]
    require(
        len(pins) == before["binding_count"] == 518,
        "Exact source/test/dependency pin scope differs",
    )
    require(len({pin["path"] for pin in pins}) == 518, "Duplicate source pin")
    for pin in (*pins, before["auditor"]):
        path = workspace / pin["path"]
        require(path.resolve().is_relative_to(workspace.resolve()), "Audit input escapes workspace")
        actual = binding(path)
        require(
            actual["bytes"] == pin["bytes"] and actual["sha256"] == pin["sha256"],
            "Source/auditor drift",
        )
    cases = list(ET.parse(junit).iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"] == 221,
        "Exact runtime case count differs",
    )
    require(
        set(ids) == set(before["collected_case_ids"]), "Precollected and actual runtime IDs differ"
    )
    require(
        all(not list(case) for case in cases), "Runtime includes failure/error/skip/output nodes"
    )
    expected_modules = {
        "tests/integration/test_scientific_ablation_evaluation_io.py": 82,
        "tests/integration/test_scientific_ablation_evaluation_v2.py": 48,
        "tests/integration/test_scientific_ablation_inference_io.py": 69,
        "tests/unit/test_directory_publication.py": 14,
        "tests/unit/test_process_file_lock.py": 8,
    }
    require(set(before["modules"]) == set(expected_modules), "Exact five modules differ")
    require(
        all(
            sum(value.startswith(module + "::") for value in ids) == count
            for module, count in expected_modules.items()
        ),
        "Actual per-module scope differs",
    )
    expected_modes = {
        (routing, goal, broadcast)
        for routing in ("learned", "random", "uniform")
        for goal in ("enabled", "disabled")
        for broadcast in ("enabled", "disabled")
    }
    group = "test_actual_all12_c3_earlier_best_capture_score_disk_binding"
    mode_ids = [
        case.attrib["name"] for case in cases if case.attrib["name"].startswith(group + "[")
    ]
    require(len(mode_ids) == 12, "Actual new all12 runtime coverage missing")
    roots = [
        path
        for path in base.iterdir()
        if path.is_dir() and path.name.startswith("test_actual_all12_c3_earlie")
    ]
    require(len(roots) == 12, "Exact all12 raw fixture roots missing")
    modes: set[tuple[str, str, str]] = set()
    reports: list[dict[str, Any]] = []
    for root in sorted(roots):
        files = list(
            (root / "reports" / "scientific-ablation-evaluation-v2").glob("*/evaluation.json")
        )
        require(len(files) == 1, "Actual published report missing/ambiguous")
        path = files[0]
        result = manifest(path, "iris/scientific-ablation-evaluation/v2")
        p = result["payload"]
        require(
            path.parent.name == result["sha256"]
            and set(p)
            == {
                "kind",
                "scoring_profile",
                "scientific_launch_status",
                "nll_scope",
                "plan",
                "capture",
                "python_resource_scope",
                "python_limits",
                "python_policy",
                "rows",
                "overall",
                "families",
            },
            "Hash-only report shape differs",
        )
        require(
            p["scientific_launch_status"] == "not_assessed"
            and p["nll_scope"] == "copied_from_bound_capture_not_recomputed",
            "Report scope laundering",
        )
        capture_roots = list(
            (root / "source" / "captures" / "scientific-ablation-capture-v1").glob("*/plan.json")
        )
        require(len(capture_roots) == 1, "Actual capture plan missing/ambiguous")
        capture_root = capture_roots[0].parent
        plan = manifest(capture_root / "plan.json", "iris/scientific-ablation-capture-plan/v1")
        complete = manifest(
            capture_root / "complete" / "complete.json",
            "iris/scientific-ablation-capture-complete/v1",
        )
        row = manifest(
            capture_root / "row-00000" / "row.json", "iris/scientific-ablation-heldout-row/v1"
        )
        ack = manifest(
            capture_root / "row-00000" / "ack.json", "iris/scientific-ablation-capture-row-ack/v1"
        )
        require(
            p["plan"] == plan and p["capture"] == complete,
            "Report differs from actual independently read captured plan/complete",
        )
        source = plan["payload"]["source"]["payload"]
        policy = source["ablation"]
        mode = (policy["routing"], policy["goal"], policy["broadcast"])
        require(mode not in modes and mode in expected_modes, "Duplicated/foreign persisted mode")
        modes.add(mode)
        require(
            source["selected_step"] == 1
            and source["terminal_step"] == 3
            and source["current_checkpoint_sha256"] != source["best_checkpoint_sha256"],
            "Actual earlier-best fixture binding differs",
        )
        require(
            plan["payload"]["cycles"] == 3 and len(p["rows"]) == 1,
            "Actual C3/one-row fixture coverage differs",
        )
        scored = p["rows"][0]
        require(
            scored["capture_row_sha256"] == row["sha256"]
            and scored["capture_ack_sha256"] == ack["sha256"],
            "Score/capture row/ack binding differs",
        )
        require(
            scored["target_token_count"] == row["payload"]["target_token_count"]
            and scored["target_nll_sum"] == row["payload"]["target_nll_sum"],
            "Capture NLL was not copied exactly",
        )
        require(
            p["overall"]["target_tokens"] == sum(item["target_token_count"] for item in p["rows"]),
            "Overall token sum differs",
        )
        require(
            p["overall"]["target_nll_sum"]
            == math.fsum(item["target_nll_sum"] for item in p["rows"]),
            "Overall NLL sum differs",
        )
        require(
            re.search(
                r'"(?:output_text|raw_candidate|prompt_text|reference_answer)"\s*:',
                canonical(result).decode(),
            )
            is None,
            "Raw answer/prompt/label disclosure",
        )
        reports.append(
            {
                "mode": list(mode),
                "report": binding(path),
                "artifact_sha256": result["sha256"],
                "plan_sha256": plan["sha256"],
                "complete_capture_sha256": complete["sha256"],
                "actual_row_sha256": row["sha256"],
                "actual_ack_sha256": ack["sha256"],
            }
        )
    require(modes == expected_modes, "Full persisted all12 mode coverage differs")
    return {
        "kind": "iris-r015-evaluation-io-independent-readback-v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "source_bindings_unchanged": 518,
        "exact_runtime_ids": len(ids),
        "new_storage_ids": sum(
            "test_scientific_ablation_evaluation_io.py::" in value for value in ids
        ),
        "all12_earlier_best_c3_actual_capture_report_bindings": reports,
        "junit": binding(junit),
        "before": binding(before_path),
        "limits": [
            "Local tiny CPU engineering, not learned-model quality or launch approval.",
            "No source imports, candidate execution, model loads or audit filesystem writes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.workspace, args.before, args.junit, args.basetemp),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
