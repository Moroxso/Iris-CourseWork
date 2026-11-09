"""Independent raw native writer inspection; no Iris/Torch/tests/model imports.

This reads genuine tiny-source fixtures, not a general public artifact format or
scientific acceptance. ALL7 tensor bytes, original source/score chains, complete
scheduled plans/isolation/receiver effects and clustered statistics are checked.
Whole-record permutations are checked as multisets plus input-only seed bindings;
this reader does not emulate Torch's CPU permutation RNG or reprove the complete
pre-forward JSON-layout theorem. Those have distinct kernel/preflight evidence.
"""

import argparse
import copy
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from re import fullmatch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_design as designs
import verify_scientific_ablation_causal_observed_budget as budgets
import verify_scientific_ablation_causal_report_layout as layouts
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_controls as kernel
import verify_scientific_ablation_observation as native
import verify_scientific_ablation_observed_controls as stages
import verify_scientific_ablation_writer_statistics as statistics
from scientific_ablation_observed_budget_guard import checked_budget
from verify_scientific_ablation_collapse import original_membership
from verify_scientific_ablation_evaluation_io import binding, canonical, domain, manifest, require
from verify_scientific_ablation_evaluation_ledger import pin

STAT_BEFORE = "artifacts/operational/r015-writer-statistics-before-20261013-run01.json"
STAT_READBACK = "artifacts/operational/r015-writer-statistics-readback-20261013-run01.json"
PREDECESSORS = {
    STAT_BEFORE: "69e87de8e1cf530aa52480765f0b06aa43495614be67a3543e32402adf827e6d",
    STAT_READBACK: "e20a68b800dafc49e8ad31d287f86334881cc5fd69f075fafaf4a29c38bd166f",
    statistics.BASELINE: statistics.BASELINE_SHA,
    statistics.VERDICT: statistics.VERDICT_SHA,
}
NEW_TESTS = (
    "tests/integration/test_scientific_ablation_writer_effects.py",
    "tests/unit/test_scientific_ablation_native_pass_frames.py",
    "tests/unit/test_scientific_ablation_native_writer_reader.py",
)
NEW_FILES = (
    "src/iris/application/scientific_ablation_causal_receipts.py",
    "src/iris/application/scientific_ablation_writer_effects.py",
    *NEW_TESTS,
)
SELF = "scripts/verify_scientific_ablation_native_writer_effects.py"
CHANGED = "src/iris/application/scientific_ablation_causal_controls.py"


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for name, checksum in PREDECESSORS.items():
        require(binding(Path(name))["sha256"] == checksum, "Immutable accepted predecessor drift")
    before = controls.load(Path(STAT_BEFORE))
    bindings, changes = [], []
    for previous in before["bindings"]:
        current = pin(Path.cwd(), previous["path"])
        if previous["path"].replace("\\", "/") == CHANGED:
            require(current != previous, "Explicit common-stage retention edit required")
            changes.append(dict(previous=previous, current=current))
        else:
            require(current == previous, "Unrelated accepted source/test drift")
        bindings.append(current)
    require(len(bindings) == 559 and len(changes) == 1, "Exact accepted559 + ONE existing edit")
    auditors = []
    for previous in before["auditors"]:
        current = pin(Path.cwd(), previous["path"])
        require(current == previous, "Accepted independent auditor drift")
        auditors.append(current)
    require(len(auditors) == 15, "All15 predecessor auditors retained")
    return (
        [*bindings, *(pin(Path.cwd(), n) for n in NEW_FILES)],
        [*auditors, pin(Path.cwd(), SELF)],
        changes,
    )


def snapshot() -> dict[str, Any]:
    """Fresh exact union; collection is NOT a runtime verdict."""
    frozen = scope()
    stats_before = controls.load(Path(STAT_BEFORE))
    source_before = controls.load(Path(statistics.BASELINE))
    modules = list(dict.fromkeys([*source_before["modules"], *stats_before["modules"], *NEW_TESTS]))
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
            *modules,
        ],
        capture_output=True,
        check=True,
        timeout=180,
    )
    ids = [line.strip() for line in collected.stdout.decode().splitlines() if "::test_" in line]
    inherited = set(source_before["nodeids"]) | set(stats_before["nodeids"])
    added = {i for i in ids if any(i.startswith(n + "::") for n in NEW_TESTS)}
    require(
        len(inherited) == 1230 and len(ids) == len(set(ids)) and set(ids) == inherited | added,
        "Exact inherited union + ALL new cases; no shrink",
    )
    require(scope() == frozen, "Current source/test/reader stable throughout collection")
    return dict(
        kind="iris-r015-native-writer-before-v1",
        bindings=frozen[0],
        auditors=frozen[1],
        changed=frozen[2],
        predecessors=[binding(Path(n)) for n in PREDECESSORS],
        modules=modules,
        nodeids=ids,
        expected_cases=len(ids),
        inherited_cases=len(inherited),
        new_cases=len(added),
        actual_collection_exit_code=collected.returncode,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


class _EmbeddedDesign(Path):
    """Canonical embedded JSON adapter, NEVER a separate physical file claim."""

    def __init__(self, value: dict[str, Any]) -> None:
        super().__init__("embedded-native-writer-design.json")
        self._encoded = canonical(value) + b"\n"

    def read_bytes(self) -> bytes:
        return self._encoded


def _closed(value: Any, fields: str, label: str) -> None:
    require(type(value) is dict and set(value) == set(fields.split()), f"Closed {label}")


def _integer(value: Any, minimum: int, maximum: int, label: str) -> None:
    require(type(value) is int and minimum <= value <= maximum, f"Exact {label} integer domain")


def _digest(value: Any) -> None:
    require(type(value) is str and fullmatch(r"[0-9a-f]{64}", value) is not None, "Exact SHA256")


def _vector(value: Any, count: int, family: type, label: str) -> None:
    require(
        type(value) is list
        and len(value) == count
        and all(
            type(v) is family and (not isinstance(v, float) or math.isfinite(v)) for v in value
        ),
        f"Exact {label} vector families/geometry",
    )


def _trace(trace: dict[str, Any]) -> None:
    _closed(
        trace,
        "kind semantics options num_modules num_slots state_dim cycle_count "
        "intervention_plan_sha256 cycles",
        "trace",
    )
    require(
        trace["kind"] == "iris-workspace-diagnostic-trace-v1"
        and trace["semantics"] == "post-intervention-read-and-post-write-v1",
        "Trace domain",
    )
    for name, value in (("num_modules", 4), ("num_slots", 4), ("state_dim", 8), ("cycle_count", 3)):
        _integer(trace[name], value, value, name)
    _digest(trace["intervention_plan_sha256"])
    options = trace["options"]
    _closed(options, "mode example_id checkpoint_id seed maximum_scalars", "trace options")
    require(
        options["mode"] in ("metadata", "full")
        and all(type(options[n]) is str and options[n] for n in ("example_id", "checkpoint_id")),
        "Exact original trace labels",
    )
    _integer(options["seed"], 0, 2**63 - 1, "trace seed")
    _integer(options["maximum_scalars"], 1, 2_000_000, "trace quota")
    _vector(trace["cycles"], 3, dict, "cycle inventory")
    for number, cycle in enumerate(trace["cycles"], 1):
        _closed(
            cycle,
            "cycle module_bids module_confidence selected_modules selected_slots "
            "write_modules overwritten_slots slot_ages slot_ttl slot_norms slot_sources "
            "slot_occupancy slot_write_counts module_state_norms workspace_cosine_matrix "
            "read_slot_sources read_slot_occupancy read_attention full",
            "trace cycle",
        )
        _integer(cycle["cycle"], number, number, "cycle")
        for name in (
            "module_bids",
            "module_confidence",
            "module_state_norms",
            "slot_ttl",
            "slot_norms",
        ):
            _vector(cycle[name], 4, float, name)
            require(name == "module_bids" or all(v >= 0 for v in cycle[name]), "Nonnegative trace")
        require(all(v <= 1 for v in cycle["module_confidence"]), "Actual confidence range")
        for name in ("slot_ages", "slot_write_counts", "slot_sources", "read_slot_sources"):
            _vector(cycle[name], 4, int, name)
            require(
                all(-1 <= v < 4 for v in cycle[name])
                if "sources" in name
                else all(v >= 0 for v in cycle[name]),
                "Actual trace integer domain",
            )
        for name in ("slot_occupancy", "read_slot_occupancy"):
            _vector(cycle[name], 4, bool, name)
        for name in ("selected_modules", "selected_slots", "write_modules", "overwritten_slots"):
            indices = cycle[name]
            require(
                type(indices) is list
                and all(type(v) is int and 0 <= v < 4 for v in indices)
                and len(indices) == len(set(indices)),
                "Unique actual allocation indices",
            )
        require(
            len(cycle["write_modules"]) == len(cycle["selected_slots"])
            and set(cycle["write_modules"]) == set(cycle["selected_modules"])
            and set(cycle["overwritten_slots"]) <= set(cycle["selected_slots"]),
            "Original allocation correspondence",
        )
        for name in ("workspace_cosine_matrix", "read_attention"):
            _vector(cycle[name], 4, list, name)
            for row in cycle[name]:
                _vector(row, 4, float, name)
                require(
                    all((-1 if name == "workspace_cosine_matrix" else 0) <= v <= 1 for v in row),
                    "Trace matrix domain",
                )
        full = cycle["full"]
        require((full is not None) == (options["mode"] == "full"), "Exact full-vector presence")
        if full is not None:
            _closed(
                full,
                "read_workspace_slots workspace_slots module_states workspace_reads",
                "full trace",
            )
            for matrix in full.values():
                _vector(matrix, 4, list, "full matrix")
                for row in matrix:
                    _vector(row, 8, float, "full row")


def _frame(frame: dict[str, Any]) -> None:
    _closed(
        frame,
        "row_index role write_cycle writer slot donor_row_index donor_pair_sha256 output",
        "native pass frame",
    )
    _integer(frame["row_index"], 0, 3, "frame row")
    role = frame["role"]
    require(
        type(role) is str and role in ("empty", "shuffle", "zero", "block", "donor"), "Pass role"
    )
    for name in ("write_cycle", "writer", "slot", "donor_row_index"):
        required = (
            (name == "write_cycle" and role != "empty")
            or (name in ("writer", "slot") and role in ("zero", "block", "donor"))
            or (name == "donor_row_index" and role == "donor")
        )
        if required:
            _integer(
                frame[name],
                1 if name == "write_cycle" else 0,
                2 if name == "write_cycle" else 3,
                name,
            )
        else:
            require(frame[name] is None, "Inapplicable native frame field is null")
    if role == "donor":
        _digest(frame["donor_pair_sha256"])
        require(frame["row_index"] != frame["donor_row_index"], "Different original donor row")
    else:
        require(frame["donor_pair_sha256"] is None, "Non-donor frame has no pair")


def _effect(effect: dict[str, Any], message: bool) -> None:
    _closed(
        effect,
        "kind row_index write_cycle source_module target_module slot "
        "survives_to_next_read overwrote_existing_record "
        "genuine_distance negative_control_distance "
        "empty_invocation_sha256 intervention_invocation_sha256 negative_control_invocation_sha256 "
        "recipient_message_sha256 donor_message_sha256 donor_pair_sha256 donor_row_index",
        "writer effect",
    )
    short = "message" if message else "workspace"
    require(
        effect["kind"] == f"iris-scientific-ablation-{short}-writer-effect-v1",
        "Distinct effect domain",
    )
    for name in ("row_index", "source_module", "target_module", "slot", "write_cycle"):
        _integer(
            effect[name], 1 if name == "write_cycle" else 0, 2 if name == "write_cycle" else 3, name
        )
    require(effect["source_module"] != effect["target_module"], "Every OTHER receiver")
    for name in ("survives_to_next_read", "overwrote_existing_record"):
        require(type(effect[name]) is bool, "Exact effect flag family")
    for name in ("genuine_distance", "negative_control_distance"):
        require(
            type(effect[name]) is float and math.isfinite(effect[name]) and 0 <= effect[name] <= 1,
            "Exact finite unit effect distance",
        )
    for name in (
        "empty_invocation_sha256",
        "intervention_invocation_sha256",
        "negative_control_invocation_sha256",
    ):
        _digest(effect[name])
    for name in (
        "recipient_message_sha256",
        "donor_message_sha256",
        "donor_pair_sha256",
        "donor_row_index",
    ):
        if not message:
            require(effect[name] is None, "Workspace effects have no donor metadata")
        elif name == "donor_row_index":
            _integer(effect[name], 0, 3, "effect donor row")
            require(effect[name] != effect["row_index"], "Different effect donor row")
        else:
            _digest(effect[name])


def metadata(raw: dict[str, Any]) -> None:
    """Closed primitive families only; NOT native/source/semantic acceptance."""
    _closed(
        raw,
        "kind native_measurements source_design mode actual_execute_calls projected_report "
        "scientific_acceptance full_r015_complete",
        "native witness",
    )
    _vector(raw["mode"], 3, str, "mode")
    _integer(raw["actual_execute_calls"], 1, 100000, "execution count")
    _closed(raw["projected_report"], "payload sha256", "projected report")
    _digest(raw["projected_report"]["sha256"])
    data = raw["projected_report"]["payload"]
    _closed(
        data,
        "kind scope scientific_acceptance source_metadata byte_preflight actual_teacher_executions "
        "actual_trace_proposal_observation_scalar_bound actual_post_intervention_scalar_bound "
        "ordinary_rows controlled_passes effects statistics",
        "projected payload",
    )
    meta = data["source_metadata"]
    _closed(
        meta,
        "controlled_teacher_graph_sha256 ordinary_teacher_graph_sha256 sidecar_sha256 run_seed "
        "evaluation evaluation_complete evaluation_plan observed_budget",
        "source metadata",
    )
    for name in (
        "controlled_teacher_graph_sha256",
        "ordinary_teacher_graph_sha256",
        "sidecar_sha256",
    ):
        _digest(meta[name])
    _integer(meta["run_seed"], 0, 2**63 - 1, "run seed")
    quota = data["byte_preflight"]
    _closed(
        quota, "layout_sha256 maximum_report_bytes report_byte_preflight_bound", "byte preflight"
    )
    _digest(quota["layout_sha256"])
    for name in ("maximum_report_bytes", "report_byte_preflight_bound"):
        _integer(quota[name], 1, 512 * 1024**2, name)
    for name in (
        "actual_teacher_executions",
        "actual_trace_proposal_observation_scalar_bound",
        "actual_post_intervention_scalar_bound",
    ):
        _integer(data[name], 1, 10**12, name)
    for name in ("ordinary_rows", "effects"):
        require(type(data[name]) is list and data[name], "Nonempty original row/effect lists")
    for row in data["ordinary_rows"]:
        _closed(
            row,
            "row_index capture_row_sha256 evaluation_row_sha256 evaluation_ack_sha256 "
            "ordinary_teacher_invocation_sha256 target_sha256 target_token_count target_nll_sum "
            "ordinary_mean_target_nll ordinary_trace",
            "ordinary row",
        )
        _integer(row["row_index"], 0, 3, "ordinary row")
        _integer(row["target_token_count"], 1, 10**9, "target count")
        for name in ("target_nll_sum", "ordinary_mean_target_nll"):
            require(
                type(row[name]) is float and math.isfinite(row[name]) and row[name] >= 0,
                "Exact ordinary finite NLL",
            )
        for name in row:
            if name.endswith("sha256"):
                _digest(row[name])
        _trace(row["ordinary_trace"])
    _closed(data["controlled_passes"], "empty trials", "pass groups")
    groups = data["controlled_passes"]
    message = data["kind"] == "iris-scientific-ablation-message-writer-report-v1"
    require(
        type(groups["trials"]) is dict
        and set(groups["trials"]) == ({"block", "donor"} if message else {"shuffle", "zero"}),
        "Distinct trial families",
    )
    for role, frames in (("empty", groups["empty"]), *groups["trials"].items()):
        require(type(frames) is list and bool(frames), "Nonempty whole pass group")
        for frame in frames:
            _frame(frame)
            require(frame["role"] == role, "Pass cannot move into another family")
    for effect in data["effects"]:
        _effect(effect, message)


def _values(state: dict[str, Any]) -> dict[str, Any]:
    return {name: state[name]["values"] for name in native.FIELDS}


def _state_record(state: dict[str, Any], slot: int) -> bytes:
    return canonical({name: state[name]["values"][0][slot] for name in native.FIELDS})


def _output(raw: dict[str, Any], config: dict[str, Any]) -> None:
    require(
        set(raw)
        == {
            "kind",
            "controlled",
            "maximum_observation_scalars",
            "actual_observation_scalars",
            "invocation_sha256",
            "observations",
        },
        "Closed observed native wire fields",
    )
    c = raw["controlled"]
    _closed(
        c,
        "kind loss supervised_token_count plan plan_sha256 ablation_policy_sha256 "
        "control_plan_sha256 prompt_sha256 target_sha256 teacher_graph_sha256 sidecar_state_sha256 "
        "invocation_sha256 trace messages states",
        "controlled receipt",
    )
    _trace(c["trace"])
    _integer(raw["maximum_observation_scalars"], 1, 2_000_000, "observation cap")
    _integer(raw["actual_observation_scalars"], 1, 2_000_000, "observation count")
    _vector(c["states"], 3, dict, "native poststate inventory")
    _vector(raw["observations"], 3, dict, "native observation inventory")
    for observation in raw["observations"]:
        _closed(
            observation,
            "cycle before_write native_written before_write_sha256 native_written_sha256",
            "native observation",
        )
    require(type(c["messages"]) is list, "Exact proposal inventory list")
    for proposal in c["messages"]:
        _closed(
            proposal,
            "kind cycle module example_id prompt_sha256 teacher_graph_sha256 "
            "sidecar_state_sha256 values content_sha256",
            "native proposal",
        )
        require(proposal["kind"] == "iris-module-message-snapshot-v1", "Proposal domain")
        _integer(proposal["cycle"], 1, 3, "proposal cycle")
        _integer(proposal["module"], 0, 3, "proposal module")
        _vector(proposal["values"], 8, float, "proposal content")
    for state in c["states"]:
        native.state(state, slots=4, width=8, modules=4)
    # The old controlled reader uses bare values; independently checked ALL7
    # tensor metadata remains in the original wire object, not invented defaults.
    native.observed(dict(raw, controlled=dict(c, states=[_values(s) for s in c["states"]])), config)


def _distance(a: list[float], b: list[float]) -> float:
    require(
        len(a) == len(b) == 8 and all(type(x) is float and math.isfinite(x) for x in (*a, *b)),
        "Actual exact finite receiver vectors",
    )
    scale = max(abs(x) for x in (*a, *b))
    if not scale:
        return 0.0
    a, b = [x / scale for x in a], [x / scale for x in b]
    return min(1.0, math.dist(a, b) / (math.hypot(*a) + math.hypot(*b)))


def _plan(
    design: dict[str, Any],
    role: str,
    cycle: int | None,
    writer: int | None,
    slot: int | None,
    seed: int | None = None,
    donor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = bool(design["donor_pairs"])
    collect = message and role in ("empty", "donor")
    control: dict[str, Any]
    if collect:
        control = dict(kind="iris-workspace-message-plan-v1", donor=donor)
    else:
        ops: list[dict[str, Any]] = []
        if role != "empty":
            operation: dict[str, Any] = {
                n: None
                for n in (
                    "slot",
                    "other_slot",
                    "module",
                    "factor",
                    "capacity",
                    "seed",
                    "sigma",
                    "values",
                    "source_module",
                    "confidence",
                    "ttl",
                )
            }
            operation.update(
                kind={
                    "zero": "zero_slot",
                    "shuffle": "shuffle_slots",
                    "block": "block_module_write",
                }[role],
                cycles=[cycle],
            )
            operation["slot" if role == "zero" else "module" if role == "block" else "seed"] = (
                slot if role == "zero" else writer if role == "block" else seed
            )
            ops.append(operation)
        control = dict(kind="iris-workspace-intervention-plan-v1", interventions=ops)
    return dict(
        kind="iris-cognitive-ablation-control-plan-v1",
        ablation=design["ablation"],
        control=control,
        maximum_message_scalars=design["policy"]["maximum_message_scalars"] if collect else None,
    )


def _isolation(empty: dict[str, Any], changed: dict[str, Any]) -> None:
    cycle, role, slot = changed["write_cycle"], changed["role"], changed["slot"]
    original, trial = empty["output"], changed["output"]
    a, b = original["controlled"], trial["controlled"]
    require(
        a["trace"]["cycles"][: cycle - 1] == b["trace"]["cycles"][: cycle - 1],
        "Unchanged prefix cycles",
    )
    for i in range(cycle):
        require(
            original["observations"][i]["before_write"] == trial["observations"][i]["before_write"],
            "Real ALL7 original recipient prewrite",
        )
        if i < cycle - 1:
            require(
                original["observations"][i] == trial["observations"][i]
                and a["states"][i] == b["states"][i],
                "Complete native prefix isolation",
            )
    left, right = a["trace"]["cycles"][cycle - 1], b["trace"]["cycles"][cycle - 1]
    for name in (
        "module_bids",
        "module_confidence",
        "selected_modules",
        "selected_slots",
        "write_modules",
        "overwritten_slots",
        "read_slot_sources",
        "read_slot_occupancy",
        "read_attention",
        "module_state_norms",
    ):
        require(left[name] == right[name], "Original recipient competition/allocation")
    require(
        all(
            left["full"][n] == right["full"][n]
            for n in ("module_states", "workspace_reads", "read_workspace_slots")
        ),
        "Original recipient read/module inputs",
    )
    before = original["observations"][cycle - 1]["before_write"]
    written = original["observations"][cycle - 1]["native_written"]
    actual = trial["observations"][cycle - 1]["native_written"]
    post = b["states"][cycle - 1]
    if role == "donor":
        require(post == actual, "DONOR uses actual native writer, no hidden post operation")
        for name in native.FIELDS:
            for i in range(4):
                if name != "slots" or i != slot:
                    require(
                        written[name]["values"][0][i] == actual[name]["values"][0][i],
                        "Donor retains all metadata and other records",
                    )
        require(
            [m for m in a["messages"] if m["cycle"] <= cycle]
            == [m for m in b["messages"] if m["cycle"] <= cycle],
            "Original proposals before donor content substitution",
        )
        return  # Gated/normalized workspace slot is NOT the donor content vector.
    require(written == actual, "Record operation retains actual original native writer")
    if role == "shuffle":
        require(
            Counter(_state_record(written, i) for i in range(4))
            == Counter(_state_record(post, i) for i in range(4)),
            "One ALL7 whole-record permutation, identity allowed",
        )
        return
    for name in native.FIELDS:
        for i in range(4):
            expected = written[name]["values"][0][i]
            if i == slot:
                expected = (
                    before[name]["values"][0][i]
                    if role == "block"
                    else (
                        [0.0] * 8
                        if name == "slots"
                        else -1
                        if name == "source_module"
                        else False
                        if name == "occupancy"
                        else 0.0
                        if name in ("ttl", "confidence")
                        else 0
                    )
                )
            require(
                post[name]["values"][0][i] == expected,
                "Actual ALL7 selected ZERO/BLOCK and other records",
            )


def _statistics(design: dict[str, Any], effects: list[dict[str, Any]]) -> dict[str, Any]:
    message = bool(design["donor_pairs"])
    policy, m = design["policy"], design["geometry"]["num_modules"]
    units = (
        [[p["first_row_index"], p["second_row_index"]] for p in design["donor_pairs"]]
        if message
        else ([[i] for i in design["selected_row_indices"]])
    )
    indexed: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for e in effects:
        indexed[e["source_module"], e["target_module"], e["row_index"]].append(e)
    records = []
    for source in range(m):
        for target in range(m):
            if source == target:
                continue
            a, b = [], []
            recipients = trials = overwritten = expired = 0
            for unit in units:
                ua, ub = [], []
                for i in unit:
                    rows = indexed.get((source, target, i), [])
                    if not rows:
                        continue
                    recipients += 1
                    trials += len(rows)
                    overwritten += sum(e["overwrote_existing_record"] for e in rows)
                    expired += sum(not e["survives_to_next_read"] for e in rows)
                    ua.append(math.fsum(e["genuine_distance"] for e in rows) / len(rows))
                    ub.append(math.fsum(e["negative_control_distance"] for e in rows) / len(rows))
                if ua:
                    a.append(math.fsum(ua) / len(ua))
                    b.append(math.fsum(ub) / len(ub))
            delta = [x - y for x, y in zip(a, b, strict=True)]
            pos = sum(x > policy["numerical_tolerance"] for x in delta)
            neg = sum(x < -policy["numerical_tolerance"] for x in delta)
            records.append(
                dict(
                    source_module=source,
                    target_module=target,
                    independent_units=len(a),
                    recipient_examples=recipients,
                    write_trials=trials,
                    overwritten_trials=overwritten,
                    expired_trials=expired,
                    mean_genuine_distance=math.fsum(a) / len(a) if a else None,
                    mean_negative_control_distance=math.fsum(b) / len(b) if b else None,
                    mean_excess=math.fsum(delta) / len(delta) if delta else None,
                    positive_units=pos,
                    negative_units=neg,
                    tied_units=len(a) - pos - neg,
                    sign_test_p=statistics.coin_tail(pos, neg),
                )
            )
    order = sorted(range(len(records)), key=lambda i: records[i]["sign_test_p"])
    scaled = [records[i]["sign_test_p"] * (len(records) - rank) for rank, i in enumerate(order)]
    for rank, i in enumerate(order):
        adjusted = min(1.0, max(scaled[: rank + 1]))
        records[i].update(
            holm_adjusted_p=adjusted, significant_positive_effect=adjusted <= policy["alpha"]
        )
    protocol = "message" if message else "workspace"
    return dict(
        kind=f"iris-scientific-ablation-{protocol}-writer-statistics-v1",
        inferential_unit="disjoint-donor-pair" if message else "example-cycle-mean",
        alpha=policy["alpha"],
        numerical_tolerance=policy["numerical_tolerance"],
        independence="input-uniqueness-does-not-prove-task-independence",
        pairs=records,
    )


def inspect(path: Path) -> dict[str, Any]:
    initial = binding(path)
    raw = controls.load(path)
    return _inspect_parsed(path, raw, initial)


def _inspect_parsed(path: Path, raw: dict[str, Any], initial: dict[str, Any]) -> dict[str, Any]:
    """Also used for in-memory corruptions; only inspect() reads a claimed witness."""
    metadata(raw)
    require(
        set(raw)
        == {
            "kind",
            "native_measurements",
            "source_design",
            "mode",
            "actual_execute_calls",
            "projected_report",
            "scientific_acceptance",
            "full_r015_complete",
        }
        and raw["kind"] == "iris-tests-actual-native-writer-schedule-v1"
        and raw["native_measurements"] is True
        and raw["scientific_acceptance"] == "not_assessed"
        and raw["full_r015_complete"] is False,
        "Honest actual private native fixture, not public/scientific acceptance",
    )
    protocol, mode, purpose, _ = designs.read_design(_EmbeddedDesign(raw["source_design"]))
    require(
        mode == tuple(raw["mode"]) and purpose == "smoke", "Actual full native tiny fixture only"
    )
    short = "message" if protocol == "paired-message" else "workspace"
    message = short == "message"
    projected = raw["projected_report"]
    data = projected["payload"]
    require(
        set(projected) == {"payload", "sha256"}
        and data["kind"] == f"iris-scientific-ablation-{short}-writer-report-v1"
        and data["scope"] == "complete-teacher-writer-effects-not-generated-answer-quality"
        and data["scientific_acceptance"] == "not_assessed",
        "Distinct honest projected report domain",
    )
    require(
        projected["sha256"] == domain(f"iris/tests/native-writer-projection/{short}/v1", data),
        "Projection checksum, NOT public report attestation",
    )
    meta, design = data["source_metadata"], raw["source_design"]["payload"]
    checked_budget(meta["observed_budget"])
    require(
        meta["observed_budget"]["payload"]["design"] == raw["source_design"],
        "Complete original input design",
    )
    source = meta["evaluation"]["payload"]["plan"]["payload"]["source"]["payload"]
    require(
        (source["selected_step"], source["terminal_step"]) == (1, 3)
        and source["ablation"] == design["ablation"],
        "Genuine earlier best/mode/terminal",
    )
    original_membership(
        path.parent / "case/test.parquet",
        meta["evaluation"]["payload"]["plan"]["payload"]["input"]["payload"]["test_parquet_sha256"],
        design["members"],
    )
    chain = sources.physical_chain(path.parent, meta["evaluation_plan"]["sha256"])
    ledger = native.score_chain_directory(path.parent, meta["evaluation_plan"]["sha256"])
    for name, filename, namespace in (
        ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
        ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
        (
            "evaluation_complete",
            "complete/complete.json",
            "iris/scientific-ablation-evaluation-complete/v2",
        ),
    ):
        require(
            meta[name] == manifest(ledger / filename, namespace),
            "Physical original COMPLETE score anchor",
        )
    candidates = [
        p
        for p in (path.parent / "case/source").rglob("manifest.json")
        if controls.load(p).get("sha256") == source["inference_sha256"]
    ]
    require(len(candidates) == 1, "ONE original physical selected source")
    inference = manifest(candidates[0], "iris/scientific-ablation-inference/v1/payload")["payload"]
    cfg = json.loads(inference["source_model_configuration_json"])["base_configuration"]
    config = {n: cfg[n + "_configuration_json"] for n in ("backbone", "cognitive", "workspace")}
    require(
        meta["run_seed"] == json.loads(inference["source_run_configuration_json"])["seed"],
        "Original run seed",
    )
    require(
        meta["sidecar_sha256"]
        == source["teacher_sidecar_sha256"]
        == inference["source_teacher_sidecar_sha256"],
        "Physical original trained sidecar anchor",
    )
    require(
        meta["ordinary_teacher_graph_sha256"]
        == domain(
            "iris/cognitive-ablation-teacher-graph/v1", config | {"policy": design["ablation"]}
        )
        and meta["controlled_teacher_graph_sha256"]
        == domain(
            "iris/cognitive-ablation-control-teacher-graph/v1",
            config | {"ablation_policy_sha256": design["ablation_policy_sha256"]},
        ),
        "BOTH physical configuration-derived teacher graphs",
    )
    controls_raw = data["controlled_passes"]
    require(
        set(controls_raw) == {"empty", "trials"}
        and set(controls_raw["trials"]) == ({"block", "donor"} if message else {"shuffle", "zero"}),
        "Distinct complete trial families",
    )
    passes = [
        *controls_raw["empty"],
        *(p for rows in controls_raw["trials"].values() for p in rows),
    ]
    indexed = {}
    for p in passes:
        require(
            set(p)
            == {
                "row_index",
                "role",
                "write_cycle",
                "writer",
                "slot",
                "donor_row_index",
                "donor_pair_sha256",
                "output",
            },
            "Closed native pass frame",
        )
        require(
            type(p["row_index"]) is int
            and type(p["role"]) is str
            and all(
                p[n] is None or type(p[n]) is int
                for n in ("write_cycle", "writer", "slot", "donor_row_index")
            ),
            "Exact pass primitive families",
        )
        _output(p["output"], config)
        c = p["output"]["controlled"]
        score = meta["evaluation"]["payload"]["rows"][p["row_index"]]
        require(
            statistics.exact(
                c["trace"]["options"],
                dict(
                    mode="full",
                    example_id=design["members"][p["row_index"]]["example_id"],
                    checkpoint_id=source["best_checkpoint_sha256"],
                    seed=meta["run_seed"],
                    maximum_scalars=design["policy"]["maximum_trace_scalars"],
                ),
            )
            and p["output"]["maximum_observation_scalars"]
            == meta["observed_budget"]["payload"]["maximum_observation_scalars"],
            "Actual original selected checkpoint/run/row/quota labels",
        )
        require(
            c["plan"]["ablation"] == design["ablation"]
            and c["teacher_graph_sha256"] == meta["controlled_teacher_graph_sha256"]
            and c["sidecar_state_sha256"] == meta["sidecar_sha256"]
            and c["prompt_sha256"] == design["members"][p["row_index"]]["cognitive_prompt_sha256"]
            and (c["target_sha256"], c["supervised_token_count"])
            == (score["teacher_target_sha256"], score["target_token_count"]),
            "Every actual trial binds ORIGINAL mode/prompt/EOS/source graph",
        )
        key = p["row_index"], p["role"], p["write_cycle"], p["writer"]
        require(key not in indexed, "No repeated/duplicate native pass")
        indexed[key] = p
    require(
        len(passes) == design["budget"]["controlled_executions"]
        and data["actual_teacher_executions"]
        == raw["actual_execute_calls"]
        == len(passes) + len(data["ordinary_rows"])
        == design["budget"]["total_executions"],
        "Exact whole8/10 schedule without hidden baselines",
    )
    require(
        all(
            type(data[n]) is int
            for n in (
                "actual_teacher_executions",
                "actual_trace_proposal_observation_scalar_bound",
                "actual_post_intervention_scalar_bound",
            )
        )
        and type(raw["actual_execute_calls"]) is int,
        "Exact execution/scalar primitive families",
    )
    proposal_scalars = sum(
        len(m["values"]) for p in passes for m in p["output"]["controlled"]["messages"]
    )
    observation_scalars = sum(p["output"]["actual_observation_scalars"] for p in passes)
    require(
        proposal_scalars == design["budget"]["total_message_scalars"]
        and observation_scalars
        == meta["observed_budget"]["payload"]["total_observation_scalar_bound"]
        and data["actual_trace_proposal_observation_scalar_bound"]
        == design["budget"]["total_numeric_scalars"] + observation_scalars
        and data["actual_post_intervention_scalar_bound"] == len(passes) * 3 * 4 * (8 + 6),
        "Complete all-pass proposal/trace/observation/poststate scalar reconciliation",
    )
    expected_keys: set[tuple[int, str, int | None, int | None]] = set()
    grouped_keys: dict[str, list[tuple[Any, ...]]] = {role: [] for role in controls_raw["trials"]}
    expected_effect_keys = []
    effects = data["effects"]
    effect_map = {
        (e["row_index"], e["write_cycle"], e["source_module"], e["target_module"]): e
        for e in effects
    }
    require(
        len(effect_map) == len(effects) == design["budget"]["maximum_effect_rows"],
        "ALL original writer/other receiver effects, no duplicates",
    )
    donors = {
        i: (j, p["sha256"])
        for p in design["donor_pairs"]
        for i, j in (
            (p["first_row_index"], p["second_row_index"]),
            (p["second_row_index"], p["first_row_index"]),
        )
    }
    require(
        [r["row_index"] for r in data["ordinary_rows"]] == design["selected_row_indices"],
        "Every original saved ordinary row",
    )
    for ordinary, row in zip(data["ordinary_rows"], design["selected_row_indices"], strict=True):
        empty = indexed[row, "empty", None, None]
        expected_keys.add((row, "empty", None, None))
        original = empty["output"]["controlled"]
        score = meta["evaluation"]["payload"]["rows"][row]
        require(
            statistics.exact(original["plan"], _plan(design, "empty", None, None, None))
            and ordinary["capture_row_sha256"] == score["capture_row_sha256"]
            and ordinary["ordinary_teacher_invocation_sha256"] == score["teacher_invocation_sha256"]
            and ordinary["target_nll_sum"] == score["target_nll_sum"]
            and ordinary["target_sha256"] == score["teacher_target_sha256"]
            and ordinary["target_token_count"] == score["target_token_count"]
            and ordinary["ordinary_mean_target_nll"] == original["loss"],
            "ONE original ordinary/EMPTY and saved EOS/NLL reproduction",
        )
        ordinary_trace = ordinary["ordinary_trace"]
        require(
            statistics.exact(
                ordinary_trace["options"],
                dict(
                    mode=design["policy"]["ordinary_trace_mode"],
                    example_id=design["members"][row]["example_id"],
                    checkpoint_id=source["best_checkpoint_sha256"],
                    seed=meta["run_seed"],
                    maximum_scalars=design["policy"]["maximum_trace_scalars"],
                ),
            )
            and ordinary_trace["intervention_plan_sha256"] == design["ablation_policy_sha256"]
            and statistics.exact(
                [{n: v for n, v in c.items() if n != "full"} for c in ordinary_trace["cycles"]],
                [{n: v for n, v in c.items() if n != "full"} for c in original["trace"]["cycles"]],
            )
            and (
                ordinary_trace["options"]["mode"] != "full"
                or statistics.exact(ordinary_trace["cycles"], original["trace"]["cycles"])
            )
            and math.isclose(
                ordinary["target_nll_sum"],
                ordinary["ordinary_mean_target_nll"] * ordinary["target_token_count"],
                rel_tol=2e-7,
                abs_tol=1e-7,
            ),
            "Actual complete ordinary/EMPTY trace and saved target NLL parity",
        )
        for binding_key, suffix, namespace in (
            ("evaluation_row_sha256", "row.json", "iris/scientific-ablation-evaluation-row/v2"),
            ("evaluation_ack_sha256", "ack.json", "iris/scientific-ablation-evaluation-ack/v2"),
        ):
            require(
                ordinary[binding_key]
                == manifest(ledger / f"score-{row:05d}" / suffix, namespace)["sha256"],
                "Actual original score/ack bindings",
            )
        for o, state in zip(empty["output"]["observations"], original["states"], strict=True):
            require(o["native_written"] == state, "Actual EMPTY keeps ALL7 native writer")
        for cycle in (1, 2):
            written, next_read = original["trace"]["cycles"][cycle - 1 : cycle + 1]
            require(len(written["write_modules"]) == 2, "Every original actual writer")
            if not message:
                shuffle = indexed[row, "shuffle", cycle, None]
                expected_keys.add((row, "shuffle", cycle, None))
                grouped_keys["shuffle"].append((row, cycle, None))
                seed = (
                    int(
                        domain(
                            "iris/scientific-ablation-whole-record-shuffle/v1",
                            dict(
                                seed=design["policy"]["shuffle_seed"],
                                example_id=design["members"][row]["example_id"],
                                prompt_sha256=design["members"][row]["cognitive_prompt_sha256"],
                                cycle=cycle,
                            ),
                        )[:16],
                        16,
                    )
                    % 2**63
                )
                require(
                    shuffle["output"]["controlled"]["plan"]["control"]["interventions"][0]["seed"]
                    == seed,
                    "Input-only SHUFFLE seed, no label/outcome",
                )
                require(
                    statistics.exact(
                        shuffle["output"]["controlled"]["plan"],
                        _plan(design, "shuffle", cycle, None, None, seed=seed),
                    ),
                    "Exactly one whole-record SHUFFLE/cycle operation",
                )
            for writer, slot in zip(
                written["write_modules"], written["selected_slots"], strict=True
            ):
                genuine = indexed[row, "block" if message else "zero", cycle, writer]
                negative = indexed[row, "donor", cycle, writer] if message else shuffle
                expected_keys.add((row, genuine["role"], cycle, writer))
                grouped_keys[genuine["role"]].append((row, cycle, writer))
                if message:
                    expected_keys.add((row, "donor", cycle, writer))
                    grouped_keys["donor"].append((row, cycle, writer))
                    donor_row, pair = donors[row]
                    donor = indexed[donor_row, "empty", None, None]["output"]["controlled"][
                        "messages"
                    ][(cycle - 1) * 4 + writer]
                    require(
                        (negative["donor_row_index"], negative["donor_pair_sha256"])
                        == (donor_row, pair)
                        and statistics.exact(
                            negative["output"]["controlled"]["plan"]["control"]["donor"], donor
                        ),
                        "Original paired same-module/cycle content, donor need not win",
                    )
                for trial in (genuine, negative):
                    _isolation(empty, trial)
                require(
                    genuine["slot"] == slot
                    and statistics.exact(
                        genuine["output"]["controlled"]["plan"],
                        _plan(design, "block" if message else "zero", cycle, writer, slot),
                    ),
                    "Exact isolated ORIGINAL writer/slot/cycle operation",
                )
                if message:
                    require(
                        negative["slot"] == slot
                        and statistics.exact(
                            negative["output"]["controlled"]["plan"],
                            _plan(design, "donor", cycle, writer, slot, donor=donor),
                        ),
                        "ONLY paired donor CONTENT, no extra operation",
                    )
                for receiver in range(4):
                    if receiver == writer:
                        continue
                    key = row, cycle, writer, receiver
                    expected_effect_keys.append(key)
                    effect = effect_map[key]
                    if message:
                        recipient = original["messages"][(cycle - 1) * 4 + writer]
                        require(
                            effect["donor_row_index"] == donor_row
                            and effect["donor_pair_sha256"] == pair
                            and effect["recipient_message_sha256"]
                            == domain("iris/module-message-snapshot/v1", recipient)
                            and effect["donor_message_sha256"]
                            == domain("iris/module-message-snapshot/v1", donor),
                            "Effect binds ORIGINAL recipient/donor proposals and disjoint pair",
                        )
                    a = next_read["full"]["module_states"][receiver]
                    b = genuine["output"]["controlled"]["trace"]["cycles"][cycle]["full"][
                        "module_states"
                    ][receiver]
                    n = negative["output"]["controlled"]["trace"]["cycles"][cycle]["full"][
                        "module_states"
                    ][receiver]
                    require(
                        all(
                            type(effect[f]) is float
                            and math.isclose(effect[f], v, rel_tol=1e-13, abs_tol=1e-15)
                            for f, v in (
                                ("genuine_distance", _distance(a, b)),
                                (
                                    "negative_control_distance",
                                    _distance(b, n) if message else _distance(a, n),
                                ),
                            )
                        ),
                        "Independent real receiver endpoints",
                    )
                    require(
                        effect["slot"] == slot
                        and all(
                            type(effect[f]) is bool
                            for f in ("survives_to_next_read", "overwrote_existing_record")
                        )
                        and effect["survives_to_next_read"]
                        is (
                            next_read["read_slot_occupancy"][slot]
                            and next_read["read_slot_sources"][slot] == writer
                        )
                        and effect["overwrote_existing_record"]
                        is empty["output"]["observations"][cycle - 1]["before_write"]["occupancy"][
                            "values"
                        ][0][slot],
                        "Actual expiry/overwrite flags",
                    )
                    require(
                        (
                            effect["empty_invocation_sha256"],
                            effect["intervention_invocation_sha256"],
                            effect["negative_control_invocation_sha256"],
                        )
                        == tuple(
                            p["output"]["invocation_sha256"] for p in (empty, genuine, negative)
                        ),
                        "SAME background/native invocation anchors",
                    )
    require(
        set(indexed) == expected_keys and list(effect_map) == expected_effect_keys,
        "Complete canonical effect and whole pass inventories",
    )
    require(
        [p["row_index"] for p in controls_raw["empty"]] == design["selected_row_indices"]
        and all(
            [(p["row_index"], p["write_cycle"], p["writer"]) for p in controls_raw["trials"][role]]
            == keys
            for role, keys in grouped_keys.items()
        ),
        "Original canonical ordering within EVERY full native pass group",
    )
    require(
        statistics.exact(data["statistics"], _statistics(design, effects)),
        "Independent available pair/example cycle means/exact sign/Holm ALL directed pairs",
    )
    byte_meta = data["byte_preflight"]
    require(
        all(
            type(byte_meta[n]) is int
            for n in ("maximum_report_bytes", "report_byte_preflight_bound")
        )
        and len(canonical(projected)) + 1
        <= byte_meta["report_byte_preflight_bound"]
        <= byte_meta["maximum_report_bytes"]
        <= 512 * 1024**2,
        "Actual complete projected bytes fit recorded quota (not independent layout theorem)",
    )
    require(binding(path) == initial, "Raw native witness stable before/end")
    return dict(
        path=initial,
        protocol=short,
        mode=mode,
        teacher_calls=data["actual_teacher_executions"],
        observed_calls=len(passes),
        effects=len(effects),
        source_chain=chain,
        full_r015_complete=False,
        scientific_acceptance="not_assessed",
        additional_forwards=0,
    )


def _rehash_projection(raw: dict[str, Any]) -> None:
    data = raw["projected_report"]["payload"]
    short = "message" if bool(raw["source_design"]["payload"]["donor_pairs"]) else "workspace"
    raw["projected_report"]["sha256"] = domain(
        f"iris/tests/native-writer-projection/{short}/v1", data
    )


def _rehash_pass(raw: dict[str, Any], frame: dict[str, Any]) -> None:
    """Keep local plan/receipt hashes coherent; ORIGINAL source/schedule still required."""
    output, c = frame["output"], frame["output"]["controlled"]
    old = output["invocation_sha256"]
    c["plan_sha256"] = domain("iris/cognitive-ablation-control-plan/v1", c["plan"])
    c["control_plan_sha256"] = domain(
        "iris/workspace-message-plan/v1"
        if "donor" in c["plan"]["control"]
        else "iris/workspace-intervention-plan/v1",
        c["plan"]["control"],
    )
    c["trace"]["intervention_plan_sha256"] = c["plan_sha256"]
    c["invocation_sha256"] = domain(
        "iris/cognitive-ablation-controlled-teacher-invocation/v1",
        {
            name: c[name]
            for name in (
                "plan_sha256",
                "prompt_sha256",
                "target_sha256",
                "teacher_graph_sha256",
                "sidecar_state_sha256",
                "supervised_token_count",
            )
        },
    )
    output["invocation_sha256"] = domain(
        "iris/ablation-workspace-observed-teacher-invocation/v1",
        dict(
            controlled_invocation_sha256=c["invocation_sha256"],
            maximum_observation_scalars=output["maximum_observation_scalars"],
            actual_observation_scalars=output["actual_observation_scalars"],
            observations=[
                {n: o[n] for n in ("cycle", "before_write_sha256", "native_written_sha256")}
                for o in output["observations"]
            ],
        ),
    )
    for effect in raw["projected_report"]["payload"]["effects"]:
        for name in (
            "empty_invocation_sha256",
            "intervention_invocation_sha256",
            "negative_control_invocation_sha256",
        ):
            if effect[name] == old:
                effect[name] = output["invocation_sha256"]


def negative_probes(path: Path) -> list[dict[str, Any]]:
    """Actual raw fixtures corrupted ONLY in memory; no writes/forwards/claimed new passes."""
    initial = binding(path)
    original = controls.load(path)
    # FIRST accept the real unmodified physical witness, not an always-denying checker.
    _inspect_parsed(path, original, initial)
    message = bool(original["source_design"]["payload"]["donor_pairs"])
    role, negative = ("block", "donor") if message else ("zero", "shuffle")
    payload = ("projected_report", "payload")
    trial = (*payload, "controlled_passes", "trials", role, 0)
    counter = (*payload, "controlled_passes", "trials", negative, 0)
    changes: list[tuple[str, tuple[Any, ...], Any]] = [
        ("hidden-payload", (*payload, "foreign"), "hidden"),
        ("hidden-source", (*payload, "source_metadata", "foreign"), "hidden"),
        ("hidden-byte-metadata", (*payload, "byte_preflight", "foreign"), "hidden"),
        ("hidden-ordinary-row", (*payload, "ordinary_rows", 0, "foreign"), "hidden"),
        ("hidden-frame", (*trial, "foreign"), "hidden"),
        ("hidden-controlled", (*trial, "output", "controlled", "foreign"), "hidden"),
        ("hidden-observation", (*trial, "output", "observations", 0, "foreign"), "hidden"),
        ("hidden-effect", (*payload, "effects", 0, "foreign"), "hidden"),
        (
            "foreign-ordinary-graph",
            (*payload, "source_metadata", "ordinary_teacher_graph_sha256"),
            "0" * 64,
        ),
        ("boolean-run-seed", (*payload, "source_metadata", "run_seed"), True),
        ("foreign-capture", (*payload, "ordinary_rows", 0, "capture_row_sha256"), "0" * 64),
        (
            "foreign-ordinary-seed",
            (*payload, "ordinary_rows", 0, "ordinary_trace", "options", "seed"),
            0,
        ),
        ("foreign-trial-seed", (*trial, "output", "controlled", "trace", "options", "seed"), 0),
        ("boolean-row", (*trial, "row_index"), False),
        (
            "float-writer",
            (*trial, "writer"),
            float(
                original["projected_report"]["payload"]["controlled_passes"]["trials"][role][0][
                    "writer"
                ]
            ),
        ),
        ("boolean-write-cycle", (*trial, "write_cycle"), True),
        ("boolean-effect-cycle", (*payload, "effects", 0, "write_cycle"), True),
        ("integer-effect-flag", (*payload, "effects", 0, "survives_to_next_read"), 1),
        ("integer-distance", (*payload, "effects", 0, "genuine_distance"), 0),
        ("wrong-effect-domain", (*payload, "effects", 0, "kind"), "foreign-effect"),
        ("wrong-effect-slot", (*payload, "effects", 0, "slot"), 3),
        ("wrong-native-invocation", (*payload, "effects", 0, "empty_invocation_sha256"), "0" * 64),
        ("wrong-statistics", (*payload, "statistics", "pairs", 0, "positive_units"), 100),
        (
            "missing-effect",
            (*payload, "effects"),
            original["projected_report"]["payload"]["effects"][:-1],
        ),
        (
            "reverse-trial-order",
            (*payload, "controlled_passes", "trials", role),
            list(
                reversed(
                    original["projected_report"]["payload"]["controlled_passes"]["trials"][role]
                )
            ),
        ),
        (
            "missing-ordinary-row",
            (*payload, "ordinary_rows"),
            original["projected_report"]["payload"]["ordinary_rows"][:-1],
        ),
        (
            "tiny-recorded-byte-bound",
            (*payload, "byte_preflight", "report_byte_preflight_bound"),
            1,
        ),
        ("false-completion", ("full_r015_complete",), True),
        ("false-science", ("scientific_acceptance",), "accepted"),
    ]
    if message:
        changes += [
            ("foreign-effect-donor", (*payload, "effects", 0, "donor_message_sha256"), "0" * 64),
            (
                "foreign-effect-recipient",
                (*payload, "effects", 0, "recipient_message_sha256"),
                "0" * 64,
            ),
            ("foreign-effect-pair", (*payload, "effects", 0, "donor_pair_sha256"), "0" * 64),
            ("boolean-donor-row", (*counter, "donor_row_index"), True),
        ]
    else:
        changes += [("foreign-workspace-donor", (*payload, "effects", 0, "donor_row_index"), 1)]
    results = []
    for label, keys, new in changes:
        altered = copy.deepcopy(original)
        destination = altered
        for key in keys[:-1]:
            destination = destination[key]
        destination[keys[-1]] = new
        _rehash_projection(altered)
        try:
            _inspect_parsed(path, altered, initial)
        except ValueError as error:
            results.append(dict(probe=label, rejected=True, reason=str(error)))
        else:
            raise ValueError(f"Corrupted rehashed witness accepted: {label}")
    for label in ("self-rehashed-wrong-operation", "self-rehashed-negative-plan"):
        altered = copy.deepcopy(original)
        group = role if label.endswith("operation") else negative
        frame = altered["projected_report"]["payload"]["controlled_passes"]["trials"][group][0]
        control = frame["output"]["controlled"]["plan"]["control"]
        if group == "donor":
            control["donor"]["values"][0] += 0.25
            control["donor"]["content_sha256"] = domain(
                "iris/module-message-content/v1", {"values": control["donor"]["values"]}
            )
        else:
            operation = control["interventions"][0]
            name = "module" if group == "block" else "slot" if group == "zero" else "seed"
            operation[name] = (operation[name] + 1) % (4 if name != "seed" else 2**63)
        _rehash_pass(altered, frame)
        _rehash_projection(altered)
        try:
            _inspect_parsed(path, altered, initial)
        except ValueError as error:
            results.append(dict(probe=label, rejected=True, reason=str(error)))
        else:
            raise ValueError(f"Self-rehashed locally coherent plan accepted: {label}")
    require(
        binding(path) == initial,
        "Original positive fixture unchanged after ALL in-memory negatives",
    )
    return results


def audit(before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    """Exact fresh inherited+native runtime and ALL positive raw families, no forwards."""
    frozen, before_pin, junit_pin = scope(), binding(before_path), binding(junit)
    before = controls.load(before_path)
    require(
        frozen == (before["bindings"], before["auditors"], before["changed"]),
        "Exact current frozen producer/reader inputs",
    )
    ids = stages._passing_ids(junit)
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"])
        and before["actual_collection_exit_code"] == 0,
        "ONE exact fresh whole runtime gate, not stale pilot unions",
    )
    inventory = {
        "preflights": sorted(base.glob("test_all12_complete_original*/preflight-*.json")),
        "old_stages": sorted(base.glob("test_all12_original_best1_both*/stage-*.json")),
        "new_stages": sorted(base.glob("test_all12_complete_original*/stage-*.json")),
        "native_observations": sorted(
            base.glob("test_all12_actual_original_bes*/observations.json")
        ),
        "old_controls": sorted(base.glob("test_all12_actual_best1_*/controls-*.json")),
        "kernels": sorted(base.rglob("controls.json")),
        "overwrites": sorted(base.glob("test_real_preexisting_overwrit*/overwrite.json")),
        "budgets": sorted(base.rglob("observed-budget.json")),
        "encodings": sorted(base.glob("test_whole_encoding_specimen*/encoding-specimen.json")),
        "statistics": sorted(base.rglob("statistics-specimen.json")),
        "writers": sorted(base.glob("test_all12_both_full_original_*/native-writer-*.json")),
    }
    require(
        [len(p) for p in inventory.values()] == [24, 24, 24, 12, 24, 168, 1, 28, 2, 25, 24],
        "Exact ALL inherited+new positive raw family inventory",
    )
    fixtures = sorted(
        {
            p.parent
            for name in (
                "preflights",
                "old_stages",
                "native_observations",
                "old_controls",
                "writers",
            )
            for p in inventory[name]
        }
        | {p for p in base.glob("report-layout-original-source*") if not p.is_symlink()}
    )
    require(len(fixtures) == 61, "ALL61 original COMPLETE physical source chains")
    roots = {*fixtures, *(p.parent for group in inventory.values() for p in group)}

    def positive_paths() -> list[Path]:
        return sorted({p for root in roots for p in root.rglob("*") if p.is_file()})

    paths = positive_paths()
    pins = [binding(p) for p in paths]
    chains = []
    for fixture in fixtures:
        plans = [
            p
            for p in (fixture / "scores/scientific-ablation-evaluation-ledger-v2").iterdir()
            if p.is_dir()
        ]
        require(len(plans) == 1, "ONE actual original COMPLETE source per native fixture")
        chains.append(sources.physical_chain(fixture, plans[0].name))
    preflights = [sources.preflight(controls.load(p)) for p in inventory["preflights"]]
    stage_results = [
        stages.stage(p) for name in ("old_stages", "new_stages") for p in inventory[name]
    ]
    require(
        {(p, m) for p, m, _ in stage_results}
        == {(p, m) for p in ("workspace-record", "paired-message") for m in designs.MODES}
        and sum(n for _, _, n in stage_results) == 120,
        "Both inherited ALL12 original EMPTY stages",
    )
    for p, result in zip(inventory["preflights"], preflights, strict=True):
        metadata = controls.load(p)["source_metadata"]
        stage = controls.load(p.with_name(f"stage-{result['protocol']}.json"))
        require(
            metadata["observed_budget"] == stage["observed_budget"],
            "Actual reserved whole observed budget",
        )
        ledger = native.score_chain_directory(p.parent, result["evaluation_plan_sha256"])
        for name, file, namespace in (
            ("evaluation_plan", "plan.json", "iris/scientific-ablation-evaluation-plan/v2"),
            ("evaluation", "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"),
            (
                "evaluation_complete",
                "complete/complete.json",
                "iris/scientific-ablation-evaluation-complete/v2",
            ),
        ):
            require(
                metadata[name] == stage[name] == manifest(ledger / file, namespace),
                "ALL original physical COMPLETE score anchors",
            )
    require(
        {native.native(p) for p in inventory["native_observations"]} == designs.MODES,
        "Inherited ALL12 native before/write observations",
    )
    for p in inventory["old_controls"]:
        controls.controls(
            p,
            native.score_chain_directory(
                p.parent, controls.load(p)["payload"]["evaluation_plan"]["sha256"]
            ),
        )
    for p in inventory["kernels"]:
        kernel.witness(p)
    overwritten = controls.load(inventory["overwrites"][0])
    native.isolation(overwritten, overwritten["configuration"])
    for p in inventory["budgets"]:
        budgets.witness(p)
    require(
        {layouts.specimen(p)["protocol"] for p in inventory["encodings"]}
        == {"workspace", "message"},
        "BOTH complete encoding specimens",
    )
    stats = [statistics.specimen(p) for p in inventory["statistics"]]
    require(
        {(s["protocol"], tuple(s["mode"])) for s in stats}
        == {(p, m) for p in ("workspace-record", "paired-message") for m in designs.MODES}
        and any(s["effects"] == 12096 and s["directed_pairs"] == 4032 for s in stats),
        "ALL synthetic statistic modes and full wide-directed-pair case",
    )
    writers = [inspect(p) for p in inventory["writers"]]
    require(
        {(w["protocol"], tuple(w["mode"])) for w in writers}
        == {(p, m) for p in ("workspace", "message") for m in designs.MODES}
        and tuple(
            sum(w[n] for w in writers) for n in ("teacher_calls", "observed_calls", "effects")
        )
        == (528, 468, 720),
        "Actual FULL8/10 both ALL12 native writer measurements",
    )
    probes = {w["protocol"]: negative_probes(Path(w["path"]["path"])) for w in writers[:2]}
    require(
        set(probes) == {"workspace", "message"} and sum(map(len, probes.values())) == 67,
        "BOTH positive fixtures plus ALL67 denied rehashed in-memory corruptions",
    )
    require(
        scope() == frozen
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin
        and positive_paths() == paths
        and [binding(p) for p in paths] == pins,
        "ALL producer/reader/positive inventories and bytes stable before/end",
    )
    return dict(
        kind="iris-r015-native-writer-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        inherited_cases=before["inherited_cases"],
        new_cases=before["new_cases"],
        source_test_pins=len(frozen[0]),
        auditor_pins=len(frozen[1]),
        positive_family_counts={name: len(values) for name, values in inventory.items()},
        stable_positive_files=len(pins),
        positive_inventory_sha256=domain("iris/operational/positive-inventory/v1", pins),
        original_physical_chains=len(chains),
        physical_chains_sha256=domain("iris/operational/physical-chains/v1", chains),
        native_writer_teacher_calls=528,
        native_writer_observed_calls=468,
        native_writer_effects=720,
        native_writer_source_files=[w["path"] for w in writers],
        in_memory_negative_probes=probes,
        reader_additional_forwards=0,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--snapshot", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--negative-probes", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        require(not args.negative_probes, "Negative probes require an actual witness")
        result = snapshot()
    elif args.audit:
        require(
            not args.negative_probes
            and all(p is not None for p in (args.before, args.junit, args.base)),
            "Full audit requires original before, actual JUnit and fresh base",
        )
        result = audit(args.before, args.junit, args.base)
    else:
        result = inspect(args.input)
        if args.negative_probes:
            result["in_memory_negative_probes"] = negative_probes(args.input)
        result.pop("source_chain")
    require(
        not any(
            n == p or n.startswith(p + ".")
            for n in sys.modules
            for p in ("iris", "torch", "tests", "pytest")
        ),
        "ZERO project/model/test imports",
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
