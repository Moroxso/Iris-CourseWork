"""Independent stdlib mode/control pin/JUnit/raw-witness audit; not science.

Snapshot alone explicitly collects pytest in a child. Audit imports no project,
Torch, pytest or codec, executes no model/candidate and writes only a NEW receipt.
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
    nonfinite,
    pairs,
    require,
)
from verify_scientific_ablation_evaluation_ledger import pin

PREVIOUS = "artifacts/operational/r015-collapse-scale-before-20261012-run01.json"
PREVIOUS_SHA = "fe4b5caefa5ac312f906ce12654d97e1995a89d11e4b76a1c3711c5b00c22206"
CHANGED = "src/iris/training/model.py"
NEW = (
    "src/iris/ablations/controls.py",
    "src/iris/training/ablation_controls.py",
    "tests/unit/test_cognitive_ablation_controls.py",
    "tests/integration/test_cognitive_ablation_controls.py",
)
SELF = "scripts/verify_scientific_ablation_controls.py"
MODULES = (
    NEW[2],
    NEW[3],
    "tests/unit/test_teacher_forcing.py",
    "tests/unit/test_cognitive_ablation.py",
    "tests/unit/test_cognitive_ablation_default_device.py",
    "tests/unit/test_message_session.py",
    "tests/integration/test_cognitive_ablation_teacher.py",
    "tests/integration/test_intervened_teacher.py",
    "tests/integration/test_message_causal.py",
    "tests/integration/test_scientific_ablation_diagnostics.py",
)
KINDS = (
    "zero_workspace",
    "zero_slot",
    "freeze_slot",
    "swap_slots",
    "shuffle_slots",
    "gaussian_noise",
    "replace_slot",
    "block_module_write",
    "block_module_read",
    "force_module_winner",
    "scale_module_bid",
    "reduce_workspace_capacity",
)
MODES = {
    (r, g, b)
    for r in ("learned", "random", "uniform")
    for g in ("enabled", "disabled")
    for b in ("enabled", "disabled")
}


def load(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    value: dict[str, Any] = json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(canonical(value) + b"\n" == content, "Canonical raw witness/receipt differs")
    return value


def scope(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require(pin(root, PREVIOUS)["sha256"] == PREVIOUS_SHA, "Accepted scale snapshot changed")
    previous = load(root / PREVIOUS)
    require(
        len(previous["bindings"]) == 532 and len(previous["auditors"]) == 5,
        "Exact accepted scale scope missing",
    )
    current, changes = [], []
    for old in previous["bindings"]:
        now = pin(root, old["path"])
        if now != old:
            require(old["path"] == CHANGED, "Unrelated accepted source/test changed")
            changes.append({"previous": old, "current": now})
        current.append(now)
    require(len(changes) == 1, "Exactly the shared teacher kernel source must change")
    for item in previous["auditors"]:
        require(pin(root, item["path"]) == item, "Accepted auditor changed")
    current.extend(pin(root, name) for name in NEW)
    auditors = [*previous["auditors"], pin(root, SELF)]
    return previous, current, auditors, changes[0]


def snapshot(root: Path) -> dict[str, Any]:
    _, bindings, auditors, change = scope(root)
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
    require(len(ids) == len(set(ids)) and len(ids) > 400, "Full selected fresh collection missing")
    require(
        sum(node.startswith(NEW[2] + "::") or node.startswith(NEW[3] + "::") for node in ids)
        == 228,
        "Exact228new cases required",
    )
    return {
        "kind": "iris-r015-ablation-controls-before-v1",
        "previous": pin(root, PREVIOUS),
        "bindings": bindings,
        "auditors": auditors,
        "changed": change,
        "modules": MODULES,
        "nodeids": ids,
        "expected_cases": len(ids),
        "new_cases": 228,
        "actual_collection_exit_code": collected.returncode,
        "actual_collection_stdout_sha256": hashlib.sha256(collected.stdout).hexdigest(),
    }


def mode(receipt: dict[str, Any]) -> tuple[str, str, str]:
    ablation = receipt["plan"]["ablation"]
    return ablation["routing"], ablation["goal"], ablation["broadcast"]


def receipt(raw: dict[str, Any], source: dict[str, Any]) -> None:
    plan = raw["plan"]
    require(
        raw["kind"] == "iris-ablation-controlled-teacher-output-v1"
        and plan["kind"] == "iris-cognitive-ablation-control-plan-v1",
        "Distinct receipt/plan kind required",
    )
    require(
        raw["plan_sha256"] == domain("iris/cognitive-ablation-control-plan/v1", plan),
        "Combined plan hash differs",
    )
    policy_sha = domain("iris/cognitive-ablation-policy/v1", plan["ablation"])
    require(
        raw["ablation_policy_sha256"] == policy_sha and mode(raw) in MODES,
        "Actual cognitive mode differs",
    )
    control = plan["control"]
    message = control["kind"] == "iris-workspace-message-plan-v1"
    require(
        message or control["kind"] == "iris-workspace-intervention-plan-v1",
        "Closed control protocol required",
    )
    control_domain = (
        "iris/workspace-message-plan/v1" if message else "iris/workspace-intervention-plan/v1"
    )
    require(
        raw["control_plan_sha256"] == domain(control_domain, control), "Control identity differs"
    )
    graph = domain(
        "iris/cognitive-ablation-control-teacher-graph/v1",
        {name: source[name] for name in ("backbone", "cognitive", "workspace")}
        | {"ablation_policy_sha256": policy_sha},
    )
    require(raw["teacher_graph_sha256"] == graph, "Actual mode-bound configuration graph differs")
    invocation = {
        name: raw[name]
        for name in (
            "plan_sha256",
            "prompt_sha256",
            "target_sha256",
            "teacher_graph_sha256",
            "sidecar_state_sha256",
            "supervised_token_count",
        )
    }
    require(
        raw["invocation_sha256"]
        == domain("iris/cognitive-ablation-controlled-teacher-invocation/v1", invocation),
        "Target/count/graph invocation differs",
    )
    require(
        type(raw["loss"]) is float
        and math.isfinite(raw["loss"])
        and raw["loss"] >= 0
        and type(raw["supervised_token_count"]) is int
        and raw["supervised_token_count"] > 0,
        "Actual detached NLL/count differs",
    )
    trace = raw["trace"]
    require(
        trace["intervention_plan_sha256"] == raw["plan_sha256"]
        and trace["options"]["mode"] == "full"
        and (trace["num_modules"], trace["num_slots"], trace["state_dim"], trace["cycle_count"])
        == (4, 4, 8, 3)
        and len(trace["cycles"]) == len(raw["states"]) == 3,
        "Complete new-domain actual trace differs",
    )
    for index, (cycle, state) in enumerate(zip(trace["cycles"], raw["states"], strict=True), 1):
        require(
            cycle["cycle"] == index
            and cycle["full"] is not None
            and state["slots"] == [cycle["full"]["workspace_slots"]],
            "Actual full-state/vector trace differs",
        )
        for field, key in (
            ("age", "slot_ages"),
            ("ttl", "slot_ttl"),
            ("occupancy", "slot_occupancy"),
            ("source_module", "slot_sources"),
            ("write_count", "slot_write_counts"),
        ):
            require(state[field] == [cycle[key]], "Actual complete workspace metadata differs")
    messages = raw["messages"]
    require(
        (message and len(messages) == 12) or (not message and not messages),
        "Proposal collection coverage differs",
    )
    for index, snapshot in enumerate(messages):
        require(
            (snapshot["cycle"], snapshot["module"]) == (index // 4 + 1, index % 4)
            and snapshot["example_id"] == trace["options"]["example_id"]
            and len(snapshot["values"]) == 8
            and all(
                snapshot[name] == raw[name]
                for name in ("prompt_sha256", "teacher_graph_sha256", "sidecar_state_sha256")
            )
            and snapshot["content_sha256"]
            == domain("iris/module-message-content/v1", {"values": snapshot["values"]}),
            "Actual ordered mode-bound proposal snapshot differs",
        )


def witness(path: Path) -> tuple[str, tuple[str, str, str], str | None]:
    item = load(path)
    require(
        item["kind"] == "iris-tests-ablation-controls-witness-v1"
        and item["scientific_acceptance"] == "not_assessed"
        and item["fixture"] == "offline-random-tiny-qwen",
        "Witness cannot claim learned/scientific acceptance",
    )
    rows = item["receipts"]
    names = {
        "empty": ("workspace", "messages"),
        "operation": ("changed",),
        "donor": ("control", "donor", "changed", "blocked"),
    }[item["scenario"]]
    for name in names:
        receipt(rows[name], item)
    main = rows[names[0]]
    if item["scenario"] == "empty":
        require(
            rows["workspace"]["loss"] == rows["messages"]["loss"] == rows["ordinary_loss"]
            and rows["workspace"]["trace"]["cycles"]
            == rows["messages"]["trace"]["cycles"]
            == rows["ordinary_trace"]["cycles"],
            "Bitwise ordinary/controlled/message parity differs",
        )
    elif item["scenario"] == "operation":
        changed = rows["changed"]
        operations = changed["plan"]["control"]["interventions"]
        require(
            len(operations) == 1
            and operations[0]["kind"] in KINDS
            and operations[0]["cycles"] == [2]
            and changed["trace"]["cycles"][0] == rows["ordinary_trace"]["cycles"][0],
            "Real scheduled operator/prior-cycle isolation differs",
        )
        return "operation", mode(changed), operations[0]["kind"]
    else:
        control, donor, changed, blocked = (rows[name] for name in names)
        require(
            mode(control) == mode(donor) == mode(changed) == mode(blocked)
            and all(
                row["teacher_graph_sha256"] == control["teacher_graph_sha256"]
                and row["sidecar_state_sha256"] == control["sidecar_state_sha256"]
                for row in (donor, changed, blocked)
            ),
            "Recipient/donor mode/weights differ",
        )
        snapshot = changed["plan"]["control"]["donor"]
        require(
            snapshot in donor["messages"]
            and snapshot["cycle"] == 1
            and donor["prompt_sha256"] != control["prompt_sha256"],
            "Transfer is not the paired actual donor proposal",
        )
        normal = control["trace"]["cycles"][0]
        module = normal["write_modules"][0]
        slot = normal["selected_slots"][0]
        require(
            snapshot["module"] == module
            and blocked["plan"]["control"]["interventions"][0]["module"] == module
            and not blocked["states"][0]["occupancy"][0][slot],
            "Actual allocated single-write blocker differs",
        )
        for altered in (changed, blocked):
            row = altered["trace"]["cycles"][0]
            require(
                all(
                    row[name] == normal[name]
                    for name in (
                        "module_bids",
                        "module_confidence",
                        "selected_modules",
                        "write_modules",
                        "selected_slots",
                        "read_attention",
                        "read_slot_sources",
                        "read_slot_occupancy",
                    )
                )
                and all(
                    row["full"][name] == normal["full"][name]
                    for name in ("read_workspace_slots", "module_states", "workspace_reads")
                ),
                "Recipient pre-write routing/content-independent allocation differs",
            )
            for field, values in control["states"][0].items():
                require(
                    all(
                        values[0][index] == altered["states"][0][field][0][index]
                        for index in range(4)
                        if index != slot
                    ),
                    "Another recipient record changed",
                )
                if altered is changed and field != "slots":
                    require(
                        values == altered["states"][0][field],
                        "Transplant changed recipient metadata",
                    )
        if mode(control)[2] == "disabled":
            require(
                control["trace"]["cycles"][1]["full"]["module_states"]
                == changed["trace"]["cycles"][1]["full"]["module_states"]
                == blocked["trace"]["cycles"][1]["full"]["module_states"],
                "Disabled-broadcast next-module physical null differs",
            )
    return item["scenario"], mode(main), None


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_binding, junit_binding = binding(before_path), binding(junit)
    before = load(before_path)
    _, bindings, auditors, change = scope(root)
    require(
        before["kind"] == "iris-r015-ablation-controls-before-v1"
        and before["previous"] == pin(root, PREVIOUS)
        and before["bindings"] == bindings
        and before["auditors"] == auditors
        and before["changed"] == change
        and before["modules"] == list(MODULES)
        and before["actual_collection_exit_code"] == 0,
        "Exact fresh source/collection scope differs",
    )
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(case) for case in cases),
        "Exact selected passing JUnit scope differs",
    )
    paths = sorted(base.glob("*/controls.json"))
    require(len(paths) == 168, "All12empty+144operator+12donor raw witnesses required")
    raw_before = [binding(path) for path in paths]
    observed = [witness(path) for path in paths]
    require(
        {key for scenario, key, operation in observed if scenario == "empty"} == MODES
        and {key for scenario, key, operation in observed if scenario == "donor"} == MODES
        and {(key, operation) for scenario, key, operation in observed if scenario == "operation"}
        == {(key, kind) for key in MODES for kind in KINDS},
        "Complete all12xall12 actual control coverage differs",
    )
    require(
        [binding(path) for path in paths] == raw_before
        and scope(root)[1:] == (bindings, auditors, change)
        and binding(before_path) == before_binding
        and binding(junit) == junit_binding,
        "Inputs/source changed during full readback",
    )
    return {
        "kind": "iris-r015-ablation-controls-readback-v1",
        "passed": True,
        "before": before_binding,
        "junit": junit_binding,
        "source_test_pins": len(bindings),
        "auditor_pins": len(auditors),
        "exact_cases": len(ids),
        "new_cases": 228,
        "raw_witnesses": raw_before,
        "all12_modes_empty_and_donor": True,
        "all144_workspace_operator_modes": True,
        "scientific_acceptance": "not_assessed",
        "full_r015_complete": False,
        "limits": [
            "Actual offline random tiny-Qwen kernel math; "
            "not selected-best/full-corpus/learned/scientific acceptance",
            "Audit imports/executes no project/model/pytest/codec; "
            "collection alone invokes pytest child",
            "Sequential immutable input rechecks; "
            "not hostile filesystem/concurrent-writer/external execution attestation",
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
            "Explicit immutable readback inputs required",
        )
        result = audit(args.workspace.resolve(), args.before, args.junit, args.base)
    require(
        not any(
            name == "iris" or name.startswith("iris.") or name in ("torch", "pytest", "pyarrow")
            for name in sys.modules
        ),
        "Observer imported project/test/model/codec",
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
