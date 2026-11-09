"""Independent raw native observation audit; no Iris/Torch/pytest imports.

Snapshot explicitly collects a child pytest process. Audit only reads retained
bytes, original Parquet and pinned old readers; it performs zero model forwards.
"""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from re import fullmatch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as original
import verify_scientific_ablation_controls as kernel
from verify_scientific_ablation_causal_design import read_design
from verify_scientific_ablation_collapse import original_membership
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture

PREVIOUS = "artifacts/operational/r015-source-controls-before-20261012-run03.json"
PREVIOUS_SHA = "033cf3d70b06829ac21e9a0014b87ec3a9cfc0d52bda0d0714c47f70d49fa471"
KERNEL = "artifacts/operational/r015-controls-before-20261012-run01.json"
KERNEL_SHA = "5061484e12a800088d32abb05599d83ce66b6c6accaf230b49a6ca107d2b45f8"
CHANGED = "src/iris/training/model.py"
NEW = (
    "src/iris/training/ablation_observation.py",
    "tests/unit/test_ablation_observation.py",
    "tests/integration/test_ablation_observation.py",
    "tests/integration/test_ablation_observation_identity.py",
    "tests/unit/test_ablation_observation_reader.py",
)
SELF = "scripts/verify_scientific_ablation_observation.py"
MODULES = tuple(dict.fromkeys((*kernel.MODULES, *original.MODULES, *NEW[1:])))
FIELDS = ("slots", "age", "occupancy", "source_module", "confidence", "ttl", "write_count")


def scope(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require(pin(root, PREVIOUS)["sha256"] == PREVIOUS_SHA, "Accepted source snapshot changed")
    before = original.load(root / PREVIOUS)
    require(len(before["bindings"]) == 541 and len(before["auditors"]) == 8, "Predecessor scope")
    bindings, changes = [], []
    for previous in before["bindings"]:
        current = pin(root, previous["path"])
        if current != previous:
            require(previous["path"] == CHANGED, "Unrelated predecessor source/test changed")
            changes.append(dict(previous=previous, current=current))
        bindings.append(current)
    require(len(changes) == 1, "Exactly one intentional teacher kernel change required")
    for auditor in before["auditors"]:
        require(pin(root, auditor["path"]) == auditor, "Accepted observer changed")
    return (
        [*bindings, *(pin(root, p) for p in NEW)],
        [*before["auditors"], pin(root, SELF)],
        changes[0],
    )


def snapshot(root: Path) -> dict[str, Any]:
    bindings, auditors, changed = scope(root)
    require(pin(root, KERNEL)["sha256"] == KERNEL_SHA, "Accepted kernel collection changed")
    inherited = set(original.load(root / KERNEL)["nodeids"]) | set(
        original.load(root / PREVIOUS)["nodeids"]
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
        timeout=180,
        check=True,
    )
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    new = {i for i in ids if any(i.startswith(p + "::") for p in NEW[1:])}
    require(
        len(ids) == len(set(ids)) and set(ids) == inherited | new, "Exact combined union required"
    )
    require(
        len(inherited) == 644 and len(new) == 104, "Exact inherited/new case membership differs"
    )
    return dict(
        kind="iris-r015-ablation-observation-before-v1",
        previous=pin(root, PREVIOUS),
        kernel=pin(root, KERNEL),
        bindings=bindings,
        auditors=auditors,
        changed=changed,
        modules=list(MODULES),
        nodeids=ids,
        expected_cases=len(ids),
        new_cases=len(new),
        inherited_cases=len(inherited),
        actual_collection_exit_code=collected.returncode,
        actual_collection_stdout_sha256=hashlib.sha256(collected.stdout).hexdigest(),
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
    )


def tensor(raw: dict[str, Any], shape: list[int], dtype: str) -> str:
    require(set(raw) == {"values", "dtype", "shape", "tensor_sha256"}, "Closed tensor witness")
    require(raw["dtype"] == dtype and raw["shape"] == shape, "Actual tensor dtype/shape differs")

    def flatten(value: Any, dims: list[int]) -> list[Any]:
        if not dims:
            require(
                type(value) is (bool if dtype == "bool" else int if dtype == "int64" else float),
                "Exact tensor scalar types required",
            )
            require(not isinstance(value, float) or math.isfinite(value), "Finite tensor scalar")
            return [value]
        require(type(value) is list and len(value) == dims[0], "Exact dense tensor shape required")
        return [v for row in value for v in flatten(row, dims[1:])]

    values = flatten(raw["values"], shape)
    fmt = {"float32": "f", "float64": "d", "int64": "q", "bool": "?"}[dtype]
    data = b"".join(struct.pack("<" + fmt, v) for v in values)
    h = hashlib.sha256()
    for content in (b"iris/tensor/v1", canonical(dict(dtype=dtype, shape=shape)), data):
        h.update(len(content).to_bytes(8, "big"))
        h.update(content)
    require(h.hexdigest() == raw["tensor_sha256"], "Actual raw tensor bytes/hash differ")
    return h.hexdigest()


def state(raw: dict[str, Any], *, slots: int, width: int, modules: int) -> str:
    require(set(raw) == set(FIELDS), "All seven exact state fields required")
    floating = raw["slots"]["dtype"]
    require(floating in ("float32", "float64"), "Supported ordinary native precision required")
    hashes = {}
    for name in FIELDS:
        dtype = (
            "bool"
            if name == "occupancy"
            else "int64"
            if name in ("age", "source_module", "write_count")
            else floating
        )
        hashes[name] = tensor(
            raw[name], [1, slots, width] if name == "slots" else [1, slots], dtype
        )
    for j in range(slots):
        occupied = raw["occupancy"]["values"][0][j]
        require(
            all(raw[f]["values"][0][j] >= 0 for f in ("age", "ttl", "write_count"))
            and 0 <= raw["confidence"]["values"][0][j] <= 1,
            "Canonical metadata domains",
        )
        if occupied:
            require(0 <= raw["source_module"]["values"][0][j] < modules, "Actual source module")
        else:
            for name in FIELDS:
                expected: Any = (
                    [0.0] * width if name == "slots" else -1 if name == "source_module" else 0
                )
                require(raw[name]["values"][0][j] == expected, "Canonical actual empty metadata")
    return domain("iris/ablation-workspace-observed-state/v1", hashes)


def observed(raw: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    require(raw["kind"] == "iris-ablation-observed-teacher-output-v1", "Distinct native receipt")
    c: dict[str, Any] = raw["controlled"]
    trace = c["trace"]
    k, d, m, cycles = (trace[n] for n in ("num_slots", "state_dim", "num_modules", "cycle_count"))
    require((m, d, cycles) == (4, 8, 3) and k in (2, 4), "Actual native fixture geometry")
    if k == 4:
        kernel.receipt(c, config)
    else:
        require(c["kind"] == "iris-ablation-controlled-teacher-output-v1", "Actual controlled V1")
        require(
            c["plan_sha256"] == domain("iris/cognitive-ablation-control-plan/v1", c["plan"]), "Plan"
        )
        require(
            c["teacher_graph_sha256"]
            == domain(
                "iris/cognitive-ablation-control-teacher-graph/v1",
                config | {"ablation_policy_sha256": c["ablation_policy_sha256"]},
            ),
            "Native graph",
        )
        require(
            c["invocation_sha256"]
            == domain(
                "iris/cognitive-ablation-controlled-teacher-invocation/v1",
                {
                    n: c[n]
                    for n in (
                        "plan_sha256",
                        "prompt_sha256",
                        "target_sha256",
                        "teacher_graph_sha256",
                        "sidecar_state_sha256",
                        "supervised_token_count",
                    )
                },
            ),
            "Invocation",
        )
    actual = 2 * cycles * k * (d + 6)
    require(
        type(raw["maximum_observation_scalars"]) is int
        and actual
        == raw["actual_observation_scalars"]
        <= raw["maximum_observation_scalars"]
        <= 2_000_000
        and len(raw["observations"]) == cycles,
        "Actual complete observation scalar bound",
    )
    bindings = []
    for index, observation in enumerate(raw["observations"], 1):
        require(
            type(observation["cycle"]) is int and observation["cycle"] == index, "Ordered phases"
        )
        for phase in ("before_write", "native_written"):
            require(
                state(observation[phase], slots=k, width=d, modules=m)
                == observation[phase + "_sha256"],
                "Actual complete phase fingerprint",
            )
        bindings.append(
            {n: observation[n] for n in ("cycle", "before_write_sha256", "native_written_sha256")}
        )
    require(
        raw["invocation_sha256"]
        == domain(
            "iris/ablation-workspace-observed-teacher-invocation/v1",
            dict(
                controlled_invocation_sha256=c["invocation_sha256"],
                maximum_observation_scalars=raw["maximum_observation_scalars"],
                actual_observation_scalars=actual,
                observations=bindings,
            ),
        ),
        "Distinct observation invocation",
    )
    return c


def isolation(raw: dict[str, Any], config: dict[str, Any]) -> None:
    baseline, blocked = observed(raw["baseline"], config), observed(raw["blocked"], config)
    slot, writer = raw["slot"], raw["writer"]
    cycle = baseline["trace"]["cycles"][1]
    require(
        (writer, slot) in zip(cycle["selected_modules"], cycle["selected_slots"], strict=True),
        "Allocated writer",
    )
    require(
        blocked["plan"]["control"]["interventions"]
        == [
            dict(
                kind="block_module_write",
                cycles=[2],
                module=writer,
                slot=None,
                other_slot=None,
                factor=None,
                capacity=None,
                seed=None,
                sigma=None,
                values=None,
                source_module=None,
                confidence=None,
                ttl=None,
            )
        ],
        "Exactly isolated write block",
    )
    for phase in ("before_write", "native_written"):
        require(
            raw["baseline"]["observations"][1][phase + "_sha256"]
            == raw["blocked"]["observations"][1][phase + "_sha256"],
            "Original target-cycle background",
        )
    for index, observation in enumerate(raw["baseline"]["observations"]):
        require(
            all(
                canonical(observation["native_written"][f]["values"])
                == canonical(baseline["states"][index][f])
                for f in FIELDS
            ),
            "Real baseline native write",
        )
    target = raw["blocked"]["observations"][1]
    for f in FIELDS:
        for j in range(baseline["trace"]["num_slots"]):
            phase = "before_write" if j == slot else "native_written"
            require(
                canonical(blocked["states"][1][f][0][j])
                == canonical(target[phase][f]["values"][0][j]),
                "ALL7 selected block/other-slot isolation",
            )


def native(path: Path) -> tuple[str, str, str]:
    raw = original.load(path)
    source = raw["original_source"]
    require(
        (source["selected_step"], source["terminal_step"]) == (1, 3)
        and raw["actual_observed_executions"] == 3
        and not raw["full_r015_complete"]
        and raw["scientific_acceptance"] == "not_assessed",
        "Genuine scoped earlier-best fixture",
    )
    ledger = (
        path.parent
        / "scores/scientific-ablation-evaluation-ledger-v2"
        / raw["evaluation_plan_sha256"]
    )
    plan = manifest(ledger / "plan.json", "iris/scientific-ablation-evaluation-plan/v2")["payload"]
    generation = plan["generation_plan"]
    require(
        generation["sha256"] == raw["generation_plan_sha256"]
        and generation["payload"]["source"]["payload"] == source,
        "Actual original COMPLETE score/generation source differs",
    )
    require(
        original.load(path.with_name("design.json")) == raw["design"],
        "Actual original design bytes",
    )
    read_design(path.with_name("design.json"))
    original_membership(
        path.parent / "case/test.parquet",
        generation["payload"]["input"]["payload"]["test_parquet_sha256"],
        raw["design"]["payload"]["members"],
    )
    manifests = [
        p
        for p in (path.parent / "case/source").rglob("manifest.json")
        if original.load(p).get("sha256") == source["inference_sha256"]
    ]
    require(len(manifests) == 1, "Actual selected inference manifest missing")
    inference = manifest(manifests[0], "iris/scientific-ablation-inference/v1/payload")["payload"]
    saved_state = binding(manifests[0].parent / "state.safetensors")
    require(
        saved_state["sha256"] == inference["state_sha256"]
        and saved_state["bytes"] == inference["state_size_bytes"],
        "Actual inference state bytes",
    )
    base = json.loads(inference["source_model_configuration_json"])["base_configuration"]
    config = {n: base[n + "_configuration_json"] for n in ("backbone", "cognitive", "workspace")}
    isolation(raw, config)
    messages = observed(raw["messages"], config)
    baseline = raw["baseline"]["controlled"]
    require(
        messages["loss"] == baseline["loss"] and len(messages["messages"]) == 12,
        "Actual message collection parity",
    )
    require(
        all(
            raw["messages"]["observations"][i][phase + "_sha256"]
            == raw["baseline"]["observations"][i][phase + "_sha256"]
            for i in range(3)
            for phase in ("before_write", "native_written")
        ),
        "Message/record phase parity",
    )
    scores = manifest(
        ledger / "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"
    )["payload"]
    row = scores["rows"][raw["row_index"]]
    require(
        baseline["sidecar_state_sha256"] == source["teacher_sidecar_sha256"]
        and baseline["target_sha256"] == row["teacher_target_sha256"]
        and baseline["supervised_token_count"] == row["target_token_count"]
        and math.isclose(
            baseline["loss"] * row["target_token_count"],
            row["target_nll_sum"],
            rel_tol=2e-7,
            abs_tol=1e-7,
        ),
        "Original saved EOS score binding",
    )
    return kernel.mode(baseline)


def score_chain_directory(fixture: Path, expected_sha256: str) -> Path:
    require(
        type(expected_sha256) is str and fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
        "Exact expected score-plan SHA256 required",
    )
    store = fixture / "scores/scientific-ablation-evaluation-ledger-v2"
    require(store.is_dir() and not store.is_symlink(), "Actual ordinary score-store required")
    entries = tuple(store.iterdir())
    require(
        len(entries) == 2 and {p.name for p in entries} == {expected_sha256, ".iris-owner.lock"},
        "Only the expected plan and owner lock may occupy the score-store",
    )
    ledger, lock = store / expected_sha256, store / ".iris-owner.lock"
    require(
        ledger.is_dir() and not ledger.is_symlink() and lock.is_file() and not lock.is_symlink(),
        "Ordinary plan directory and regular owner lock required",
    )
    return ledger


def audit(root: Path, before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    before_pin, junit_pin = binding(before_path), binding(junit)
    before = original.load(before_path)
    bindings, auditors, changed = scope(root)
    require(
        (bindings, auditors, changed)
        == (before["bindings"], before["auditors"], before["changed"]),
        "Frozen current source scope",
    )
    cases = list(ET.fromstring(junit.read_bytes()).iter("testcase"))
    ids = [c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases]
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and not any(list(c) for c in cases),
        "Exact passing combined JUnit IDs required",
    )
    native_paths = sorted(base.glob("test_all12_actual_original_bes*/observations.json"))
    controls = sorted(base.glob("test_all12_actual_best1_*/controls-*.json"))
    kernel_paths = sorted(base.rglob("controls.json"))
    overwrite = list(base.glob("test_real_preexisting_overwrit*/overwrite.json"))
    require(
        len(native_paths) == 12
        and len(controls) == 24
        and len(kernel_paths) == 168
        and len(overwrite) == 1,
        "Full actual inherited/new positive witnesses required",
    )
    paths = sorted(
        {
            p
            for witness in (*native_paths, *controls, *kernel_paths, *overwrite)
            for p in witness.parent.rglob("*")
            if p.is_file()
        }
    )
    raw_pins = [binding(p) for p in paths]
    chains = []
    for fixture in sorted({p.parent for p in (*native_paths, *controls)}):
        captures = {}
        for store in fixture.rglob("scientific-ablation-capture-v1"):
            for directory in store.iterdir():
                if directory.is_dir() and not directory.name.startswith("."):
                    captures[valid_capture(directory)] = directory
        witnesses = [p for p in (*native_paths, *controls) if p.parent == fixture]
        expected_plans = {
            original.load(p)["evaluation_plan_sha256"]
            if p.name == "observations.json"
            else original.load(p)["payload"]["evaluation_plan"]["sha256"]
            for p in witnesses
        }
        require(len(expected_plans) == 1, "One expected score plan per actual source fixture")
        chains.append(chain(score_chain_directory(fixture, expected_plans.pop()), captures))
    require(
        {native(p) for p in native_paths} == kernel.MODES,
        "All12 actual native source modes required",
    )
    for p in controls:
        plan_sha = original.load(p)["payload"]["evaluation_plan"]["sha256"]
        original.controls(
            p, p.parent / "scores/scientific-ablation-evaluation-ledger-v2" / plan_sha
        )
    for p in kernel_paths:
        kernel.witness(p)
    overwritten = original.load(overwrite[0])
    isolation(overwritten, overwritten["configuration"])
    prior = overwritten["blocked"]["observations"][1]["before_write"]
    slot = overwritten["slot"]
    require(
        prior["occupancy"]["values"][0][slot]
        and all(
            prior[f]["values"][0][slot] > 0 for f in ("age", "confidence", "ttl", "write_count")
        ),
        "Actual preexisting overwrite record",
    )
    require(
        scope(root) == (bindings, auditors, changed)
        and [binding(p) for p in paths] == raw_pins
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin,
        "Stable physical inputs before/end",
    )
    return dict(
        kind="iris-r015-ablation-observation-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        new_cases=before["new_cases"],
        source_test_pins=len(bindings),
        auditor_pins=len(auditors),
        changed=changed,
        actual_native_source_witnesses=len(native_paths),
        actual_native_source_modes=12,
        inherited_controls=len(controls),
        inherited_kernel_witnesses=len(kernel_paths),
        complete_physical_chains=chains,
        stable_positive_tree_file_pins=len(raw_pins),
        genuine_preexisting_overwrite=True,
        additional_teacher_executions=0,
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
            "Explicit inputs",
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
