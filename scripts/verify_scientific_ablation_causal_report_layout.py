"""Independent planned-layout arithmetic/raw readback; NEVER measured effects.

Only snapshot starts a child pytest collector. Audit imports no Iris/Torch/tests
and performs zero forwards. Synthetic encoding specimens remain explicitly false
as native measurements; their real metadata's original COMPLETE source is checked.
"""

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from re import fullmatch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_observed_budget as budgets
from scientific_ablation_observed_budget_guard import checked_budget
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import chain, pin, valid_capture
from verify_scientific_ablation_observation import score_chain_directory

BEFORE = "artifacts/operational/r015-observed-stage-before-20261013-run02.json"
BEFORE_SHA = "479c41c2c07c7bac5f3448898042f03698781bd6349951fce02bcb4806b3fe29"
VERDICT = "artifacts/operational/r015-observed-stage-readback-20261013-run01.json"
VERDICT_SHA = "91afd71bb4741e35e9e9a3d6119408c9a81b7db951da8ce45ec4704c02aadbae"
NEW = (
    "src/iris/artifacts/scientific_ablation_causal_report_layout_v1.py",
    "tests/unit/test_scientific_ablation_causal_report_encoding_bound.py",
    "tests/integration/test_scientific_ablation_causal_report_layout.py",
    "tests/unit/test_scientific_ablation_causal_report_layout_reader.py",
)
SELF = "scripts/verify_scientific_ablation_causal_report_layout.py"
RUNTIME_BEFORE_SHA = "0db645c89273a0a42dd25accff368d38cf1921058064eb4834f6b911c32c98d8"
READER = NEW[-1]
MODULES = (*budgets.MODULES, *NEW[1:])
STATE_FIELDS = {"slots", "age", "confidence", "ttl", "occupancy", "source_module", "write_count"}


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for path, sha in ((BEFORE, BEFORE_SHA), (VERDICT, VERDICT_SHA)):
        require(binding(Path(path))["sha256"] == sha, "Immutable accepted stage evidence changed")
    old = controls.load(Path(BEFORE))
    for frozen in (*old["bindings"], *old["auditors"]):
        require(pin(Path.cwd(), frozen["path"]) == frozen, "Accepted source/test/observer changed")
    return (
        [*old["bindings"], *(pin(Path.cwd(), p) for p in NEW)],
        [*old["auditors"], pin(Path.cwd(), SELF)],
    )


def snapshot() -> dict[str, Any]:
    pins = scope()
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
        capture_output=True,
        check=True,
        timeout=180,
    )
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    new = {i for i in ids if any(i.startswith(p + "::") for p in NEW[1:])}
    require(
        len(ids) == len(set(ids)) == 327
        and len(new) == 79
        and sum(i.startswith(READER + "::") for i in ids) == 13,
        "Exact248 original plus66new layout plus13separate reader cases",
    )
    require(scope() == pins, "Scope drift during collection")
    return dict(
        kind="iris-r015-complete-report-layout-before-v1",
        bindings=pins[0],
        auditors=pins[1],
        modules=list(MODULES),
        nodeids=ids,
        expected_cases=327,
        new_cases=79,
        inherited_cases=248,
        actual_collection_exit_code=collected.returncode,
        scientific_acceptance="not_assessed",
        full_r015_complete=False,
    )


def members(node: dict[str, Any], names: set[str] | None = None) -> dict[str, Any]:
    require(node["kind"] == "object", "Required complete object layout")
    result = dict(node["members"])
    require(
        len(result) == len(node["members"]) and (names is None or set(result) == names),
        "Required complete unique layout fields",
    )
    return result


def fixed(node: dict[str, Any]) -> Any:
    require(node["kind"] == "fixed", "Required exact frozen metadata")
    return json.loads(node["encoded"])


def array(node: dict[str, Any], count: int) -> dict[str, Any]:
    require(
        node["kind"] == "array" and node["count"] == count and node["minimum"] is None,
        "Every planned pass/state/vector is reserved exactly",
    )
    item = node["item"]
    assert type(item) is dict
    return item


def state(node: dict[str, Any], k: int, d: int) -> None:
    for field, tensor in members(node, STATE_FIELDS).items():
        fields = members(tensor, {"dtype", "shape", "tensor_sha256", "values"})
        require(
            fixed(fields["shape"]) == ([1, k, d] if field == "slots" else [1, k]),
            "Complete actual tensor metadata geometry",
        )
        values = array(array(fields["values"], 1), k)
        if field == "slots":
            values = array(values, d)
        require(
            values["kind"] == "scalar"
            and values["scalar"]
            == (
                "bool"
                if field == "occupancy"
                else "int"
                if field in ("age", "source_module", "write_count")
                else "float"
            ),
            "Every ALL7 value has its exact encoding family",
        )


def inventory(descriptor: dict[str, Any], metadata: dict[str, Any]) -> dict[str, int]:
    budget = metadata["observed_budget"]["payload"]
    design, g = budget["design"]["payload"], budget["design"]["payload"]["geometry"]
    n, m, k, d, c, r = (
        design["budget"]["examples"],
        g["num_modules"],
        g["num_slots"],
        g["state_dim"],
        g["cycle_count"],
        g["writers_per_cycle"],
    )
    message = bool(design["donor_pairs"])
    payload = members(
        members(descriptor, {"sha256", "payload"})["payload"],
        {
            "kind",
            "scope",
            "scientific_acceptance",
            "source_metadata",
            "byte_preflight",
            "actual_teacher_executions",
            "actual_trace_proposal_observation_scalar_bound",
            "actual_post_intervention_scalar_bound",
            "ordinary_rows",
            "controlled_passes",
            "effects",
            "statistics",
        },
    )
    require(fixed(payload["source_metadata"]) == metadata, "All original source bytes included")
    array(payload["ordinary_rows"], n)
    array(payload["effects"], design["budget"]["maximum_effect_rows"])
    statistics = members(payload["statistics"])
    array(statistics["pairs"], m * (m - 1))
    controls = members(payload["controlled_passes"], {"empty", "trials"})
    trials = members(controls["trials"], {"block", "donor"} if message else {"shuffle", "zero"})
    passes = [("empty", array(controls["empty"], n))]
    passes += [
        (role, array(node, n * (c - 1) * (1 if role == "shuffle" else r)))
        for role, node in trials.items()
    ]
    for role, node in passes:
        wrapper = members(
            node,
            {
                "row_index",
                "role",
                "write_cycle",
                "writer",
                "slot",
                "donor_row_index",
                "donor_pair_sha256",
                "output",
            },
        )
        require(fixed(wrapper["role"]) == role, "Separate scheduled control roles")
        observed = members(
            wrapper["output"],
            {
                "kind",
                "controlled",
                "maximum_observation_scalars",
                "actual_observation_scalars",
                "invocation_sha256",
                "observations",
            },
        )
        controlled = members(observed["controlled"])
        state(array(controlled["states"], c), k, d)
        observation = members(
            array(observed["observations"], c),
            {
                "cycle",
                "before_write",
                "native_written",
                "before_write_sha256",
                "native_written_sha256",
            },
        )
        state(observation["before_write"], k, d)
        state(observation["native_written"], k, d)
        array(controlled["messages"], c * m if message and role in ("empty", "donor") else 0)
        trace = members(controlled["trace"])
        cycle = members(array(trace["cycles"], c))
        full = members(
            cycle["full"],
            {"read_workspace_slots", "workspace_slots", "module_states", "workspace_reads"},
        )
        for field, rows in (
            ("read_workspace_slots", k),
            ("workspace_slots", k),
            ("module_states", m),
            ("workspace_reads", m),
        ):
            array(array(full[field], rows), d)
        if role == "donor":
            donor = members(members(controlled["plan"])["control"])["donor"]
            array(members(donor)["values"], d)
    post = budget["planned_controlled_teacher_executions"] * c * k * (d + 6)
    require(
        fixed(payload["actual_post_intervention_scalar_bound"]) == post,
        "Post-intervention ALL7 is separate from both native observations",
    )
    return dict(
        post_intervention_scalar_bound=post,
        ordinary_rows_byte_bound=node_bound(payload["ordinary_rows"]),
        controlled_passes_byte_bound=node_bound(payload["controlled_passes"]),
        effects_byte_bound=node_bound(payload["effects"]),
        statistics_byte_bound=node_bound(payload["statistics"]),
    )


def node_bound(
    node: dict[str, Any], value: Any = None, *, check: bool = False, depth: int = 0
) -> int:
    require(type(node) is dict and depth <= 48, "Bounded closed descriptor")
    kind = node["kind"]
    if kind == "fixed":
        require(set(node) == {"kind", "bytes", "encoded"}, "Closed fixed descriptor")
        encoded = node["encoded"].encode()
        require(
            type(node["bytes"]) is int
            and canonical(json.loads(encoded)) == encoded
            and len(encoded) == node["bytes"],
            "Exact canonical fixed metadata bytes",
        )
        if check:
            require(canonical(value) == encoded, "Actual frozen report value differs")
        return len(encoded)
    if kind == "scalar":
        require(set(node) == {"kind", "scalar", "limit"}, "Closed scalar descriptor")
        scalar, limit = node["scalar"], node["limit"]
        require(type(limit) is int and 0 <= limit <= 4096, "Exact bounded scalar text quota")
        assert type(limit) is int
        result = {"float": 32, "int": 32, "bool": 5, "digest": 66, "text": 2 + 6 * limit}[scalar]
        if check:
            require(
                (scalar == "float" and type(value) is float and math.isfinite(value))
                or (scalar == "int" and type(value) is int and -(2**63) <= value < 2**63)
                or (scalar == "bool" and type(value) is bool)
                or (
                    scalar == "digest"
                    and type(value) is str
                    and fullmatch(r"[0-9a-f]{64}", value) is not None
                )
                or (scalar == "text" and type(value) is str and len(value) <= limit),
                "Exact scalar family",
            )
            require(
                len(canonical(value)) <= result, "Actual scalar encoding exceeds its reservation"
            )
        return result
    if kind == "array":
        require(set(node) == {"kind", "count", "minimum", "item"}, "Closed array descriptor")
        count, minimum = node["count"], node["minimum"]
        require(
            type(count) is int
            and count >= 0
            and (minimum is None or (type(minimum) is int and 0 <= minimum <= count)),
            "Array quota",
        )
        assert type(count) is int
        if check:
            require(
                type(value) is list
                and (count if minimum is None else minimum) <= len(value) <= count,
                "Actual complete array inventory",
            )
            for child in value:
                node_bound(node["item"], child, check=True, depth=depth + 1)
        return 2 + count * node_bound(node["item"], depth=depth + 1) + max(0, count - 1)
    if kind == "object":
        require(set(node) == {"kind", "members"}, "Closed object descriptor")
        fields = members(node)
        if check:
            require(
                type(value) is dict and set(value) == set(fields), "Actual closed object fields"
            )
        return (
            2
            + max(0, len(fields) - 1)
            + sum(
                len(canonical(name))
                + 1
                + node_bound(child, value[name] if check else None, check=check, depth=depth + 1)
                for name, child in fields.items()
            )
        )
    require(
        kind == "choice" and set(node) == {"kind", "choices"} and node["choices"],
        "Closed alternatives",
    )
    sizes = [node_bound(child, depth=depth + 1) for child in node["choices"]]
    if check:
        for child in node["choices"]:
            try:
                node_bound(child, value, check=True, depth=depth + 1)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Actual report value has no valid layout alternative")
    return max(sizes)


def specimen(path: Path) -> dict[str, Any]:
    raw = controls.load(path)
    require(
        raw["kind"] == "iris-tests-complete-report-layout-encoding-only-v1"
        and raw["native_measurements"] is False
        and raw["actual_preflight_teacher_executions"] == 0
        and raw["scientific_acceptance"] == "not_assessed",
        "Honest synthetic encoding specimen",
    )
    descriptor, report = raw["complete_layout"], raw["example"]
    metadata = report["payload"]["source_metadata"]
    checked_budget(metadata["observed_budget"])
    derived = inventory(descriptor, metadata)
    total = node_bound(descriptor, report, check=True) + 1
    actual = len(canonical(report)) + 1
    require(
        total == raw["complete_bound"] and actual == raw["actual_canonical_bytes"] <= total,
        "Exact independent canonical bound/framing/actual encoding",
    )
    base = len(canonical(metadata))
    framing = (
        total - base - sum(derived[n] for n in derived if n != "post_intervention_scalar_bound")
    )
    derived.update(source_metadata_bytes=base, framing_byte_bound=framing)
    require(
        all(raw[n] == value for n, value in derived.items()),
        "All byte/scalar components reconciled",
    )
    byte = report["payload"]["byte_preflight"]
    protocol = (
        "message"
        if metadata["observed_budget"]["payload"]["design"]["payload"]["donor_pairs"]
        else "workspace"
    )
    checksum = domain(
        "iris/scientific-ablation-causal-report-layout/v1",
        dict(
            protocol=protocol,
            source_metadata=metadata,
            maximum_report_bytes=byte["maximum_report_bytes"],
            **{n: v for n, v in derived.items() if n != "source_metadata_bytes"},
            report_byte_preflight_bound=total,
            complete_layout=descriptor,
        ),
    )
    require(
        checksum == raw["layout_sha256"] == byte["layout_sha256"]
        and byte["report_byte_preflight_bound"]
        == total
        <= byte["maximum_report_bytes"]
        <= 512 * 1024**2,
        "Whole distinct layout identity/quota, not result checksum/measurement acceptance",
    )
    return dict(
        protocol=protocol,
        complete_bound=total,
        actual_canonical_bytes=actual,
        evaluation_plan_sha256=metadata["evaluation_plan"]["sha256"],
        **derived,
    )


def _passing_ids(path: Path) -> list[str]:
    cases = list(ET.fromstring(path.read_bytes()).iter("testcase"))
    ids = [c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases]
    require(
        len(ids) == len(set(ids)) and all(len(c) == 0 for c in cases),
        "Actual unique passing cases; no failure/error/skip",
    )
    return ids


def audit(
    before: Path, junit: Path, base: Path, *, runtime_before: Path, reader_junit: Path
) -> dict[str, Any]:
    frozen, before_pin, junit_pin = scope(), binding(before), binding(junit)
    prior = controls.load(before)
    require(frozen == (prior["bindings"], prior["auditors"]), "Frozen before/runtime scope")
    runtime_pin, reader_pin = binding(runtime_before), binding(reader_junit)
    require(runtime_pin["sha256"] == RUNTIME_BEFORE_SHA, "Immutable actual314producer before01")
    original = controls.load(runtime_before)
    source_ids, reader_ids = _passing_ids(junit), _passing_ids(reader_junit)
    require(
        original["bindings"] == frozen[0][:-1]
        and frozen[0][-1]["path"] == READER
        and original["auditors"][:-1] == frozen[1][:-1]
        and original["auditors"][-1]["path"] == frozen[1][-1]["path"] == SELF
        and original["modules"] == [p for p in prior["modules"] if p != READER]
        and sorted(source_ids) == sorted(original["nodeids"])
        and set(original["nodeids"])
        == {i for i in prior["nodeids"] if not i.startswith(READER + "::")},
        "Original553producer pins/314IDs unchanged; ONLYSELF repair and separate reader test",
    )
    require(
        len(source_ids) == 314
        and len(reader_ids) == 13
        and all(i.startswith(READER + "::") for i in reader_ids),
        "Actual314plus separate13reader cases",
    )
    ids = [*source_ids, *reader_ids]
    require(
        len(ids) == len(set(ids)) == 327
        and sorted(ids) == sorted(prior["nodeids"])
        and prior["actual_collection_exit_code"] == 0,
        "Exact327unique case union; NOT a single327model run",
    )
    paths = sorted(base.glob("test_whole_encoding_specimen*/encoding-specimen.json"))
    require(len(paths) == 2, "Both complete protocol encoding specimens")
    raw_pins = [binding(p) for p in paths]
    results = [specimen(p) for p in paths]
    require({r["protocol"] for r in results} == {"workspace", "message"}, "Both declared layouts")
    fixtures = sorted(base.glob("report-layout-original-source*"))
    require(len(fixtures) == 1, "One genuine original metadata fixture, not all12source proof")
    fixture = fixtures[0]
    physical = sorted(p for p in fixture.rglob("*") if p.is_file())
    physical_pins = [binding(p) for p in physical]
    captures = {
        valid_capture(directory): directory
        for store in fixture.rglob("scientific-ablation-capture-v1")
        for directory in store.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    }
    ledger = score_chain_directory(fixture, results[0]["evaluation_plan_sha256"])
    original_chain = chain(ledger, captures)
    for path in paths:
        metadata = controls.load(path)["example"]["payload"]["source_metadata"]
        for name, filename, checksum_domain in (
            ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
            ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
            (
                "evaluation_complete",
                "complete/complete.json",
                "iris/scientific-ablation-evaluation-complete/v2",
            ),
        ):
            require(
                metadata[name] == manifest(ledger / filename, checksum_domain),
                "Actual original COMPLETE metadata",
            )
    require(
        scope() == frozen
        and binding(before) == before_pin
        and binding(junit) == junit_pin
        and binding(runtime_before) == runtime_pin
        and binding(reader_junit) == reader_pin
        and [binding(p) for p in paths] == raw_pins
        and [binding(p) for p in physical] == physical_pins,
        "All original/new bytes stable before/end",
    )
    return dict(
        kind="iris-r015-complete-report-layout-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        runtime_before=runtime_pin,
        reader_junit=reader_pin,
        observer_only_repair=dict(previous=original["auditors"][-1], current=frozen[1][-1]),
        exact_cases=327,
        actual_source_runtime_cases=314,
        actual_separate_reader_cases=13,
        new_cases=79,
        inherited_cases=248,
        bindings=len(frozen[0]),
        auditors=len(frozen[1]),
        specimens=results,
        raw_specimen_files=raw_pins,
        original_complete_chain=original_chain,
        original_physical_files=physical_pins,
        additional_teacher_executions_by_reader=0,
        actual_measured_writer_effects=0,
        scope="planned-complete-json-layout-not-full-effect-service",
        trusted_before_busy_integration_complete=False,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--runtime-before", type=Path)
    parser.add_argument("--reader-junit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        result = snapshot()
    else:
        require(
            args.before is not None
            and args.junit is not None
            and args.base is not None
            and args.runtime_before is not None
            and args.reader_junit is not None,
            "Explicit retained audit inputs",
        )
        result = audit(
            args.before,
            args.junit,
            args.base,
            runtime_before=args.runtime_before,
            reader_junit=args.reader_junit,
        )
    require(
        not any(
            n == p or n.startswith(p + ".")
            for n in sys.modules
            for p in ("iris", "torch", "pytest", "tests")
        ),
        "Reader imported project/model/tests",
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
