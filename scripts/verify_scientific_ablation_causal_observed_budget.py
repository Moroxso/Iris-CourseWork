"""Independent stdlib observed-volume audit; no source/model attestation.

Snapshot collects a child pytest process; audit only reads immutable witnesses.
The accepted native observation boundary remains unchanged. This does not run
the future full source executor or preflight its complete publication bytes.
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_design as original
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, require
from verify_scientific_ablation_evaluation_ledger import pin

PREVIOUS = "artifacts/operational/r015-observation-before-20261012-run02.json"
PREVIOUS_SHA = "b1b1d53dc32544279f355d4fe0f12fb6617174b833c71e2757adaed8f87f01d5"
VERDICT = "artifacts/operational/r015-observation-readback-20261012-run02.json"
VERDICT_SHA = "7b39c1f44383e4e3f61697096b8b21e98a695a9c64f25bb8538f2ea777519b70"
DESIGN = "artifacts/operational/r015-causal-design-before-20261012-run02.json"
DESIGN_SHA = "41c4e285dbdc824d37978c64276fef4dc07ddd23f48872262f9b4b8a23766508"
NEW = (
    "src/iris/artifacts/scientific_ablation_causal_observed_budget_v1.py",
    "tests/unit/test_scientific_ablation_causal_observed_budget.py",
)
SELF = "scripts/verify_scientific_ablation_causal_observed_budget.py"
MODULES = (*original.MODULES, NEW[1])
DOMAIN = "iris/scientific-ablation-causal-observed-budget/v1"
COUNTS = (
    "planned_ordinary_teacher_executions",
    "planned_controlled_teacher_executions",
    "planned_total_teacher_executions",
    "planned_observed_teacher_executions",
    "observation_scalar_bound_per_pass",
    "total_observation_scalar_bound",
    "original_trace_proposal_scalar_bound",
    "total_trace_proposal_observation_scalar_bound",
)


def scope(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(pin(root, PREVIOUS)["sha256"] == PREVIOUS_SHA, "Native before02 changed")
    require(pin(root, VERDICT)["sha256"] == VERDICT_SHA, "Accepted native readback changed")
    require(pin(root, DESIGN)["sha256"] == DESIGN_SHA, "Original design node inventory changed")
    previous, verdict = original.load(root / PREVIOUS), original.load(root / VERDICT)
    require(
        len(previous["bindings"]) == 546
        and len(previous["auditors"]) == 9
        and verdict["passed"] is True
        and verdict["exact_cases"] == 748,
        "Exact accepted native scope required",
    )
    for item in (*previous["bindings"], *previous["auditors"]):
        require(pin(root, item["path"]) == item, "Accepted source/test/observer changed")
    return (
        [*previous["bindings"], *(pin(root, p) for p in NEW)],
        [*previous["auditors"], pin(root, SELF)],
    )


def snapshot(root: Path) -> dict[str, Any]:
    bindings, auditors = scope(root)
    inherited = set(original.load(root / DESIGN)["nodeids"])
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
    new = {i for i in ids if i.startswith(NEW[1] + "::")}
    require(
        len(ids) == len(set(ids)) == 248
        and len(inherited) == 186
        and len(new) == 62
        and set(ids) == inherited | new,
        "Exact186 inherited plus62 new unit cases required",
    )
    require(scope(root) == (bindings, auditors), "Source drift during collection")
    return dict(
        kind="iris-r015-observed-causal-budget-before-v1",
        previous=pin(root, PREVIOUS),
        native_verdict=pin(root, VERDICT),
        design_node_inventory=pin(root, DESIGN),
        bindings=bindings,
        auditors=auditors,
        modules=list(MODULES),
        nodeids=ids,
        expected_cases=len(ids),
        new_cases=len(new),
        inherited_cases=len(inherited),
        actual_collection_exit_code=collected.returncode,
        scope="observed-numeric-budget-unit-scope-not-full-native-or-repository-suite",
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
    )


def witness(path: Path) -> tuple[str, tuple[str, str, str], str, str]:
    raw = original.load(path)
    require(set(raw) == {"payload", "sha256"}, "Closed observed-budget root required")
    p = raw["payload"]
    require(
        set(p)
        == {
            "kind",
            "scope",
            "scientific_acceptance",
            "design",
            "maximum_observation_scalars",
            *COUNTS,
        }
        and p["kind"] == "iris-scientific-ablation-causal-observed-budget-v1"
        and p["scope"]
        == "all-controlled-passes-observed-numeric-preflight-not-measurements-or-byte-guarantee"
        and p["scientific_acceptance"] == "not_assessed"
        and raw["sha256"] == domain(DOMAIN, p),
        "Distinct honest observed-budget schema/domain required",
    )
    design_path = path.with_name("design.json")
    require(original.load(design_path) == p["design"], "Actual original design bytes differ")
    protocol, mode, purpose, b = original.read_design(design_path)
    d = p["design"]["payload"]
    g, policy = d["geometry"], d["policy"]
    observation = 2 * g["cycle_count"] * g["num_slots"] * (g["state_dim"] + 6)
    total_observation = observation * b["controlled_executions"]
    expected = dict(
        planned_ordinary_teacher_executions=b["ordinary_executions"],
        planned_controlled_teacher_executions=b["controlled_executions"],
        planned_total_teacher_executions=b["total_executions"],
        planned_observed_teacher_executions=b["controlled_executions"],
        observation_scalar_bound_per_pass=observation,
        total_observation_scalar_bound=total_observation,
        original_trace_proposal_scalar_bound=b["total_numeric_scalars"],
        total_trace_proposal_observation_scalar_bound=b["total_numeric_scalars"]
        + total_observation,
    )
    require(
        all(type(p[n]) is int and p[n] == v for n, v in expected.items())
        and type(p["maximum_observation_scalars"]) is int
        and observation <= p["maximum_observation_scalars"] <= 2_000_000
        and expected["total_trace_proposal_observation_scalar_bound"]
        <= policy["maximum_total_numeric_scalars"],
        "Exact whole trace/proposal/ALL7 volume or declared caps differ",
    )
    if purpose == "scientific":
        require(
            len(d["members"]) == 5000
            and len(d["selected_row_indices"]) == 48
            and (
                g["num_modules"],
                g["num_slots"],
                g["state_dim"],
                g["cycle_count"],
                g["writers_per_cycle"],
            )
            == (8, 8, 256, 3, 2)
            and observation == 12_576,
            "Full5000/M8/K8/D256/C3/R2 synthetic budget profile differs",
        )
    return protocol, mode, purpose, policy["ordinary_trace_mode"]


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_pin, junit_pin = binding(before_path), binding(junit)
    before = original.load(before_path)
    bindings, auditors = scope(root)
    require(
        (bindings, auditors) == (before["bindings"], before["auditors"]),
        "Frozen current source/test/auditor scope differs",
    )
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"] == 248
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(c) for c in cases),
        "Exact passing248JUnit IDs required",
    )
    paths = sorted(base.rglob("observed-budget.json"))
    require(len(paths) == 28, "All28actual saved observed-budget witnesses required")
    files = sorted(
        {p for witness_path in paths for p in (witness_path, witness_path.with_name("design.json"))}
    )
    raw_pins = [binding(p) for p in files]
    observed = [witness(p) for p in paths]
    require(
        {
            (protocol, mode)
            for protocol, mode, purpose, ordinary in observed
            if purpose == "smoke" and ordinary == "metadata"
        }
        == {
            (protocol, mode)
            for protocol in ("workspace-record", "paired-message")
            for mode in original.MODES
        }
        and {protocol for protocol, mode, purpose, ordinary in observed if purpose == "scientific"}
        == {"workspace-record", "paired-message"}
        and {protocol for protocol, mode, purpose, ordinary in observed if ordinary == "full"}
        == {"workspace-record", "paired-message"},
        "All12/both protocols/full5000/full-ordinary coverage required",
    )
    require(
        scope(root) == (bindings, auditors)
        and [binding(p) for p in files] == raw_pins
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin,
        "Stable source and raw inputs before/end required",
    )
    return dict(
        kind="iris-r015-observed-causal-budget-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        new_cases=before["new_cases"],
        source_test_pins=len(bindings),
        auditor_pins=len(auditors),
        actual_saved_budget_witnesses=len(paths),
        raw_witnesses=raw_pins,
        all12_modes_both_protocols=True,
        both_full5000_synthetic_m8_k8_d256_profiles=True,
        full_ordinary_trace_both_protocols=True,
        numeric_counts_include_every_controlled_observation=True,
        additional_teacher_executions_by_reader=0,
        scope="numeric-preflight-not-authenticated-source-execution-or-complete-report-byte-preflight",
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
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
            "Explicit retained inputs",
        )
        result = audit(args.workspace.resolve(), args.before, args.junit, args.base)
    require(
        not any(
            n == prefix or n.startswith(prefix + ".")
            for n in sys.modules
            for prefix in ("iris", "torch", "pytest", "tests")
        ),
        "Observer imported model/project/tests",
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
