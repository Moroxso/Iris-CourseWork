"""Independent stdlib input-design readback; zero native-source/model attestation."""

import argparse
import json
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
    nonfinite,
    pairs,
    require,
)
from verify_scientific_ablation_evaluation_ledger import pin

PREVIOUS = "artifacts/operational/r015-controls-before-20261012-run01.json"
PREVIOUS_SHA = "5061484e12a800088d32abb05599d83ce66b6c6accaf230b49a6ca107d2b45f8"
NEW = (
    "src/iris/artifacts/scientific_ablation_causal_design_v1.py",
    "tests/unit/test_scientific_ablation_causal_design.py",
)
SELF = "scripts/verify_scientific_ablation_causal_design.py"
MODULES = (
    NEW[1],
    "tests/unit/test_scientific_workspace.py",
    "tests/unit/test_scientific_message_causal.py",
    "tests/unit/test_workspace_trace.py",
    "tests/unit/test_message_causal.py",
    "tests/unit/test_cognitive_ablation_controls.py",
)
TASKS = ("split_spec_repair", "delayed_goal", "conflict_resolution", "next_tool_prediction")
MODES = {
    (r, g, b)
    for r in ("learned", "random", "uniform")
    for g in ("enabled", "disabled")
    for b in ("enabled", "disabled")
}


def load(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    require(len(content) <= 4 * 1024**2, "Design/snapshot byte bound exceeded")
    value: dict[str, Any] = json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(canonical(value) + b"\n" == content, "Canonical design metadata required")
    return value


def scope(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(pin(root, PREVIOUS)["sha256"] == PREVIOUS_SHA, "Accepted kernel snapshot changed")
    old = load(root / PREVIOUS)
    require(
        len(old["bindings"]) == 536 and len(old["auditors"]) == 6,
        "Exact accepted kernel scope required",
    )
    for item in (*old["bindings"], *old["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted kernel source/test/auditor changed")
    return [*old["bindings"], *(pin(root, p) for p in NEW)], [*old["auditors"], pin(root, SELF)]


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
        timeout=180,
        check=True,
    )
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    require(
        len(ids) == len(set(ids)) and sum(i.startswith(NEW[1] + "::") for i in ids) == 81,
        "Exact81 new cases and full compatibility collection required",
    )
    require(scope(root) == (bindings, auditors), "Source drift during collection")
    return dict(
        kind="iris-r015-ablation-causal-design-before-v1",
        previous=pin(root, PREVIOUS),
        bindings=bindings,
        auditors=auditors,
        modules=MODULES,
        nodeids=ids,
        expected_cases=len(ids),
        new_cases=81,
        actual_collection_exit_code=collected.returncode,
    )


def input_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in ("example_id", "prompt_sha256", "task_type", "difficulty")}


def read_design(path: Path) -> tuple[str, tuple[str, str, str], str, dict[str, int]]:
    raw = load(path)
    p = raw["payload"]
    require(
        raw["sha256"] == domain("iris/scientific-ablation-causal-design/v1", p)
        and p["kind"] == "iris-scientific-ablation-causal-design-v1"
        and p["scope"] == "input-design-not-source-authentication-or-measurements"
        and p["scientific_acceptance"] == "not_assessed",
        "Distinct honest design identity required",
    )
    policy, ablation, members, graph = p["policy"], p["ablation"], p["members"], p["geometry"]
    mode = (ablation["routing"], ablation["goal"], ablation["broadcast"])
    require(
        mode in MODES
        and p["ablation_policy_sha256"] == domain("iris/cognitive-ablation-policy/v1", ablation)
        and p["policy_sha256"]
        == domain("iris/scientific-ablation-causal-design-policy/v1", policy),
        "Mode/policy domain differs",
    )
    message = policy["kind"] == "iris-ablation-paired-message-design-policy-v1"
    require(
        message or policy["kind"] == "iris-ablation-workspace-record-design-policy-v1",
        "Baseline or foreign design cannot be relabelled",
    )
    protocol = "paired-message" if message else "workspace-record"
    require(
        tuple(m["row_index"] for m in members) == tuple(range(len(members)))
        and 1 <= len(members) <= 5000,
        "Complete dense original membership missing",
    )
    for field in ("example_id", "prompt_sha256", "cognitive_prompt_sha256"):
        require(
            len({m[field] for m in members}) == len(members), "Duplicate original input identity"
        )
    selected: list[int] = []
    exclusions: list[dict[str, Any]] = []
    donor_pairs: list[dict[str, Any]] = []
    for task in TASKS:
        for difficulty in (1, 2, 3):
            rows = [m for m in members if (m["task_type"], m["difficulty"]) == (task, difficulty)]
            ranked = sorted(
                rows,
                key=lambda m: (
                    domain(
                        f"iris/scientific-ablation-{protocol}-selection/v1",
                        {"seed": policy["selection_seed"], "input": input_identity(m)},
                    ),
                    m["example_id"],
                ),
            )
            capped = min(len(rows), policy["examples_per_stratum"])
            count = capped - capped % 2 if message else capped
            chosen = ranked[:count]
            selected.extend(m["row_index"] for m in chosen)
            for rank, row in enumerate(ranked[count:], count):
                reason = (
                    "singleton_stratum"
                    if message and len(rows) == 1
                    else "odd_rank_tail"
                    if message and rank < capped
                    else "stratum_cap"
                )
                exclusions.append(dict(row_index=row["row_index"], reason=reason))
            if message:
                shuffled = sorted(
                    chosen,
                    key=lambda m: (
                        domain(
                            "iris/scientific-ablation-message-donor-rank/v1",
                            {"seed": policy["shuffle_seed"], "input": input_identity(m)},
                        ),
                        m["example_id"],
                    ),
                )
                for left, right in zip(shuffled[::2], shuffled[1::2], strict=True):
                    donor_pairs.append(
                        dict(
                            sha256=domain(
                                "iris/scientific-ablation-message-donor-pair/v1",
                                dict(
                                    seed=policy["shuffle_seed"],
                                    first=input_identity(left),
                                    second=input_identity(right),
                                ),
                            ),
                            task_type=task,
                            difficulty=difficulty,
                            first_row_index=left["row_index"],
                            second_row_index=right["row_index"],
                        )
                    )
    require(
        p["selected_row_indices"] == selected
        and p["exclusions"] == exclusions
        and p["donor_pairs"] == donor_pairs,
        "Actual input-only selection/exclusions/pairs differ",
    )
    require(
        set(selected).isdisjoint(e["row_index"] for e in exclusions)
        and set(selected) | {e["row_index"] for e in exclusions} == set(range(len(members))),
        "Selected/excluded inventory is not a full partition",
    )
    selectors = (
        (None, None),
        *((t, None) for t in TASKS),
        *((None, d) for d in (1, 2, 3)),
        *((t, d) for t in TASKS for d in (1, 2, 3)),
    )
    groups = []
    for group_task, group_difficulty in selectors:
        available = {
            m["row_index"]
            for m in members
            if (group_task is None or m["task_type"] == group_task)
            and (group_difficulty is None or m["difficulty"] == group_difficulty)
        }
        observed = len(available & set(selected))
        groups.append(
            dict(
                task_type=group_task,
                difficulty=group_difficulty,
                available_examples=len(available),
                selected_examples=observed,
                excluded_examples=len(available) - observed,
            )
        )
    require(p["groups"] == groups and len(groups) == 20, "All20 descriptive groups differ")
    m, k, d, c, r = (
        graph[n]
        for n in ("num_modules", "num_slots", "state_dim", "cycle_count", "writers_per_cycle")
    )
    require(
        2 <= m <= 64 and 1 <= k <= 64 and 1 <= d <= 4096 and 2 <= c <= 64 and 1 <= r <= min(m, k),
        "Actual declared next-cycle geometry invalid",
    )
    n = len(selected)
    require(1 <= n <= policy["maximum_examples"], "Selected inventory exceeded; no shrinking")
    base = c * (1 + 10 * m + 20 * k + k * k + m * k)
    full = base + c * 2 * (m + k) * d
    ordinary_full = policy["ordinary_trace_mode"] == "full"
    ordinary = full if ordinary_full else base
    controlled = 1 + 2 * (c - 1) * r if message else 1 + (c - 1) * (r + 1)
    collections = n * (1 + (c - 1) * r) if message else 0
    proposal = c * m * d if message else 0
    trace_total = n * (ordinary + controlled * full)
    message_total = collections * proposal
    budget = dict(
        examples=n,
        ordinary_executions=n,
        controlled_executions=n * controlled,
        total_executions=n * (1 + controlled),
        full_trace_executions=n * (controlled + ordinary_full),
        metadata_trace_executions=0 if ordinary_full else n,
        message_collect_executions=collections,
        ordinary_trace_scalars=ordinary,
        full_trace_scalars=full,
        message_scalars_per_collection=proposal,
        total_trace_scalars=trace_total,
        total_message_scalars=message_total,
        total_numeric_scalars=trace_total + message_total,
        maximum_effect_rows=n * (c - 1) * r * (m - 1),
    )
    require(
        p["budget"] == budget
        and full <= policy["maximum_trace_scalars"]
        and 1 + controlled <= policy["maximum_executions_per_example"]
        and budget["total_executions"] <= policy["maximum_executions"]
        and budget["total_numeric_scalars"] <= policy["maximum_total_numeric_scalars"],
        "All transient ordinary/controlled traces and proposals must fit",
    )
    if message:
        require(
            n % 2 == 0
            and policy["examples_per_stratum"] % 2 == 0
            and proposal <= policy["maximum_message_scalars"]
            and len(donor_pairs) * 2 == n,
            "Even disjoint donor/message budget differs",
        )
    purpose = p["purpose"]
    require(purpose in ("smoke", "scientific"), "Declared fixture purpose invalid")
    if purpose == "scientific":
        require(
            tuple(sum(row["task_type"] == task for row in members) for task in TASKS)
            == (2000, 1000, 1000, 1000),
            "Full5000 synthetic membership missing",
        )
    return protocol, mode, purpose, budget


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_pin, junit_pin = binding(before_path), binding(junit)
    before = load(before_path)
    bindings, auditors = scope(root)
    require(
        before["kind"] == "iris-r015-ablation-causal-design-before-v1"
        and before["bindings"] == bindings
        and before["auditors"] == auditors
        and before["modules"] == list(MODULES)
        and before["actual_collection_exit_code"] == 0,
        "Exact fresh source/test/auditor scope differs",
    )
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(case) for case in cases),
        "Exact passing selected JUnit scope differs",
    )
    paths = sorted(base.glob("*/design.json"))
    require(len(paths) == 26, "All24 mode/protocol and two full5000 raw designs required")
    raw_pins = [binding(p) for p in paths]
    observed = [read_design(p) for p in paths]
    require(
        {(protocol, mode) for protocol, mode, purpose, budget in observed if purpose == "smoke"}
        == {
            (protocol, mode)
            for protocol in ("workspace-record", "paired-message")
            for mode in MODES
        }
        and {protocol for protocol, mode, purpose, budget in observed if purpose == "scientific"}
        == {"workspace-record", "paired-message"},
        "Exact all12x2/full5000 fixture coverage missing",
    )
    require(
        scope(root) == (bindings, auditors)
        and [binding(p) for p in paths] == raw_pins
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin,
        "Source or raw inputs changed during readback",
    )
    return dict(
        kind="iris-r015-ablation-causal-design-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        new_cases=81,
        source_test_pins=len(bindings),
        auditor_pins=len(auditors),
        raw_witnesses=raw_pins,
        all12_modes_both_protocols=True,
        both_full5000_synthetic_designs=True,
        independent_selection_pairs_exclusions_all20_groups_budgets=True,
        new_teacher_executions=0,
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
        limits=[
            "Synthetic input-only design, "
            "not authenticated original Parquet/native-best measurement",
            "Numeric trace/proposal volume, "
            "not publication bytes/peak memory or execution attestation",
            "Hashes do not attest external preregistration or task independence",
            "Snapshot collection invokes pytest; audit imports no project/model/pytest/codec",
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
            "Explicit pinned audit inputs required",
        )
        result = audit(args.workspace.resolve(), args.before, args.junit, args.base)
    require(
        not any(
            name == "iris" or name.startswith("iris.") or name in ("torch", "pytest", "pyarrow")
            for name in sys.modules
        ),
        "Observer imported project/test/model/codec",
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
