"""Independent complete writer semantics against a separately trusted preflight.

No Iris/Torch/Pydantic/tests imports or model calls. This validates ENTIRE reserved
wire data, not execution attestation. The caller must authenticate the original
physical source/preflight and CPU stream proof independently of the candidate.
The existing independent layout/statistic mathematics is reused, not production
validators/formatters or test-domain checksum relabelling. Geometry is not M4-only.
"""

import json
import math
import struct
from hashlib import sha256
from typing import Any

import verify_scientific_ablation_causal_report_layout as layouts
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_native_writer_effects as reference

FLOATS = ("float16", "bfloat16", "float32", "float64")
FIELDS = ("slots", "age", "confidence", "ttl", "occupancy", "source_module", "write_count")
CPU_STREAM = ("2.13.0+cpu", "cf30153c4c131c8164ee7798e5022d810682e2cb")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def fingerprint(*parts: bytes) -> str:
    h = sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def domain(name: str, value: object) -> str:
    return fingerprint(name.encode(), canonical(value))


def equal(left: object, right: object) -> bool:
    return canonical(left) == canonical(right)  # Preserve signed zero/primitive families.


def own_json(value: object, depth: int = 0) -> object:
    """Plain complete JSON tree BEFORE class/key/equality/hash/serializer callbacks.

    No arbitrary8192-sequence cap: original layout closes actual array counts,
    including the legitimate wide effects field. API callers cannot supply a
    reflected model/object/foreign leaf under a raw dictionary.
    """
    require(depth <= 40, "Closed report metadata depth")
    if type(value) is dict:
        assert isinstance(value, dict)
        require(all(type(key) is str for key in value), "Plain JSON field keys before hashing")
        return {key: own_json(child, depth + 1) for key, child in value.items()}
    if type(value) is list:
        assert isinstance(value, list)
        return [own_json(child, depth + 1) for child in value]
    if type(value) is float:
        assert isinstance(value, float)
        require(math.isfinite(value), "Finite JSON floats")
        return value
    require(
        any(type(value) is kind for kind in (str, int, bool, type(None))),
        "Plain JSON primitives before coercion/callbacks",
    )
    return value


def runtime_guard(runtime: tuple[str, str]) -> None:
    require(
        type(runtime) is tuple and len(runtime) == 2 and all(type(part) is str for part in runtime),
        "Plain original CPU stream context",
    )
    require(runtime == CPU_STREAM, "Separately trusted original supported CPU stream")


def scalar_bytes(value: object, dtype: str) -> bytes:
    if dtype == "bool":
        require(type(value) is bool, "Exact boolean tensor scalar")
        return bytes([1 if value else 0])
    if dtype == "int64":
        require(type(value) is int and -(2**63) <= value < 2**63, "Exact bounded int64 scalar")
        return struct.pack("<q", value)
    require(type(value) is float and math.isfinite(value), "Exact finite floating scalar")
    assert isinstance(value, float)
    fmt = {"float16": "e", "bfloat16": "f", "float32": "f", "float64": "d"}[dtype]
    try:
        body = struct.pack("<" + fmt, value)
    except (OverflowError, struct.error) as error:
        raise ValueError("Declared floating representation overflow") from error
    decoded = struct.unpack("<" + fmt, body)[0]
    require(
        decoded == value and math.copysign(1.0, decoded) == math.copysign(1.0, value),
        "No rounding to declared tensor dtype, including signed zero",
    )
    if dtype == "bfloat16":
        require(int.from_bytes(body, "little") & 65535 == 0, "Exact bfloat16 low-bit fidelity")
        return body[2:]
    return body


def dense(value: object, shape: list[int], dtype: str) -> bytes:
    if not shape:
        return scalar_bytes(value, dtype)
    require(type(value) is list and len(value) == shape[0], "Exact dense tensor shape")
    assert isinstance(value, list)
    return b"".join(dense(child, shape[1:], dtype) for child in value)


def state(value: dict[str, Any], *, m: int, k: int, d: int) -> str:
    require(type(value) is dict and set(value) == set(FIELDS), "Entire ALL7 state inventory")
    dtype = value["slots"]["dtype"]
    require(dtype in FLOATS, "All FOUR native floating dtypes supported")
    hashes = {}
    for name in FIELDS:
        tensor = value[name]
        shape = [1, k, d] if name == "slots" else [1, k]
        kind = (
            dtype
            if name in ("slots", "confidence", "ttl")
            else "bool"
            if name == "occupancy"
            else "int64"
        )
        require(
            type(tensor) is dict and set(tensor) == {"dtype", "shape", "values", "tensor_sha256"},
            "Closed tensor inventory",
        )
        require(
            tensor["dtype"] == kind and equal(tensor["shape"], shape),
            "Complete geometry and dtype agreement",
        )
        body = dense(tensor["values"], shape, kind)
        digest = fingerprint(b"iris/tensor/v1", canonical(dict(dtype=kind, shape=shape)), body)
        require(tensor["tensor_sha256"] == digest, "Exact declared native binary tensor SHA")
        hashes[name] = digest
    for slot in range(k):
        record = {name: value[name]["values"][0][slot] for name in FIELDS}
        require(
            record["age"] >= 0
            and record["ttl"] >= 0
            and record["write_count"] >= 0
            and 0 <= record["confidence"] <= 1,
            "State metadata ranges",
        )
        if record["occupancy"]:
            require(0 <= record["source_module"] < m, "Occupied original source range")
        else:
            require(
                record["source_module"] == -1
                and all(x == 0 for x in record["slots"])
                and all(record[n] == 0 for n in ("age", "ttl", "write_count", "confidence")),
                "Canonical empty ALL7 record",
            )
    return domain("iris/ablation-workspace-observed-state/v1", hashes)


def cpu_order(seed: int, slots: int, runtime: tuple[str, str]) -> list[int]:
    runtime_guard(runtime)
    require(
        type(seed) is int and 0 <= seed < 2**63 and type(slots) is int and 1 <= slots <= 64,
        "Closed CPU seed/geometry",
    )
    result = list(range(slots))
    if slots == 1:
        return result
    words = [seed & 0xFFFFFFFF]
    for j in range(1, 624):
        previous = words[j - 1]
        words.append((1812433253 * (previous ^ (previous >> 30)) + j) % 2**32)
    for j in range(624):
        bits = (words[j] & 0x80000000) + (words[(j + 1) % 624] & 0x7FFFFFFF)
        words[j] = words[(j + 397) % 624] ^ (bits >> 1) ^ (0x9908B0DF * (bits & 1))
    for j in range(slots - 1):
        number = words[j]
        number ^= number >> 11
        number ^= (number << 7) & 0x9D2C5680
        number ^= (number << 15) & 0xEFC60000
        number ^= number >> 18
        chosen = j + number % (slots - j)
        result[chosen], result[j] = result[j], result[chosen]
    return result


def output(value: dict[str, Any], *, m: int, k: int, d: int, c: int) -> None:
    ctl = value["controlled"]
    plan = ctl["plan"]
    control = plan["control"]
    message = control["kind"] == "iris-workspace-message-plan-v1"
    require(
        ctl["plan_sha256"] == domain("iris/cognitive-ablation-control-plan/v1", plan), "Plan SHA"
    )
    require(
        ctl["ablation_policy_sha256"]
        == domain("iris/cognitive-ablation-policy/v1", plan["ablation"]),
        "Policy SHA",
    )
    require(
        ctl["control_plan_sha256"]
        == domain(
            "iris/workspace-message-plan/v1" if message else "iris/workspace-intervention-plan/v1",
            control,
        ),
        "Control domain",
    )
    require(
        ctl["invocation_sha256"]
        == domain(
            "iris/cognitive-ablation-controlled-teacher-invocation/v1",
            {
                n: ctl[n]
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
        "Controlled invocation SHA",
    )
    require(ctl["trace"]["intervention_plan_sha256"] == ctl["plan_sha256"], "Actual trace plan")
    require(
        value["actual_observation_scalars"]
        == 2 * c * k * (d + 6)
        <= value["maximum_observation_scalars"],
        "All observed scalar counts",
    )
    bindings = []
    original_dtype = ctl["states"][0]["slots"]["dtype"]
    for index, (native, post, traced) in enumerate(
        zip(value["observations"], ctl["states"], ctl["trace"]["cycles"], strict=True), 1
    ):
        require(native["cycle"] == traced["cycle"] == index, "Ordered complete native cycles")
        state(post, m=m, k=k, d=d)
        require(
            post["slots"]["dtype"] == original_dtype
            and all(
                native[phase]["slots"]["dtype"] == original_dtype
                for phase in ("before_write", "native_written")
            ),
            "Original dtype retained across ALL cycles/phases",
        )
        for phase in ("before_write", "native_written"):
            require(
                native[phase + "_sha256"] == state(native[phase], m=m, k=k, d=d), "Native phase SHA"
            )
        bindings.append(
            {n: native[n] for n in ("cycle", "before_write_sha256", "native_written_sha256")}
        )
        require(
            equal(post["slots"]["values"][0], traced["full"]["workspace_slots"]),
            "All post slot values/trace",
        )
        for name, field in (
            ("age", "slot_ages"),
            ("ttl", "slot_ttl"),
            ("occupancy", "slot_occupancy"),
            ("source_module", "slot_sources"),
            ("write_count", "slot_write_counts"),
        ):
            require(equal(post[name]["values"][0], traced[field]), "ALL post metadata/trace")
        before = native["before_write"]
        require(
            equal(before["slots"]["values"][0], traced["full"]["read_workspace_slots"])
            and equal(before["occupancy"]["values"][0], traced["read_slot_occupancy"])
            and equal(before["source_module"]["values"][0], traced["read_slot_sources"]),
            "Actual native prewrite/read bindings",
        )
        require(
            sorted(traced["overwritten_slots"])
            == sorted(
                slot for slot in traced["selected_slots"] if before["occupancy"]["values"][0][slot]
            ),
            "Real occupied-target overwrite",
        )
        if (
            message
            or not control["interventions"]
            or index not in control["interventions"][0]["cycles"]
        ):
            require(equal(post, native["native_written"]), "No unscheduled post operation")
    require(
        value["invocation_sha256"]
        == domain(
            "iris/ablation-workspace-observed-teacher-invocation/v1",
            dict(
                controlled_invocation_sha256=ctl["invocation_sha256"],
                maximum_observation_scalars=value["maximum_observation_scalars"],
                actual_observation_scalars=value["actual_observation_scalars"],
                observations=bindings,
            ),
        ),
        "Observed invocation SHA",
    )
    messages = ctl["messages"]
    require(
        [(p["cycle"], p["module"]) for p in messages]
        == ([(i, j) for i in range(1, c + 1) for j in range(m)] if message else []),
        "Whole ordered proposal inventory",
    )
    if message:
        require(c * m * d <= plan["maximum_message_scalars"], "Original proposal quota")
    for proposal in messages:
        require(
            proposal["content_sha256"]
            == domain("iris/module-message-content/v1", {"values": proposal["values"]}),
            "Proposal content SHA",
        )
        require(
            (
                proposal["example_id"],
                proposal["prompt_sha256"],
                proposal["teacher_graph_sha256"],
                proposal["sidecar_state_sha256"],
            )
            == (
                ctl["trace"]["options"]["example_id"],
                ctl["prompt_sha256"],
                ctl["teacher_graph_sha256"],
                ctl["sidecar_state_sha256"],
            ),
            "Original collected proposal context",
        )


def trace_ranges(trace: dict[str, Any], *, m: int, k: int, c: int, r: int) -> None:
    require(
        (trace["num_modules"], trace["num_slots"], trace["cycle_count"]) == (m, k, c),
        "Original complete trace geometry",
    )
    for index, row in enumerate(trace["cycles"], 1):
        require(
            row["cycle"] == index and len(row["write_modules"]) == r,
            "Ordered whole trace cycles/writers",
        )
        for names, limit in (
            ("selected_modules", m),
            ("write_modules", m),
            ("selected_slots", k),
            ("overwritten_slots", k),
        ):
            values = row[names]
            require(
                len(set(values)) == len(values) and all(0 <= i < limit for i in values),
                "Unique in-domain trace allocation",
            )
        require(
            set(row["write_modules"]) == set(row["selected_modules"])
            and set(row["overwritten_slots"]).issubset(row["selected_slots"]),
            "Original winner/write/overwrite correspondence",
        )
        require(all(0 <= x <= 1 for x in row["module_confidence"]), "Trace confidence range")
        require(
            all(
                x >= 0
                for name in (
                    "slot_ages",
                    "slot_ttl",
                    "slot_norms",
                    "slot_write_counts",
                    "module_state_norms",
                )
                for x in row[name]
            ),
            "Trace nonnegative counters/norms/TTL",
        )
        for fields in (
            ("slot_sources", "slot_occupancy"),
            ("read_slot_sources", "read_slot_occupancy"),
        ):
            require(
                all(
                    (0 <= source < m) if active else (source == -1)
                    for source, active in zip(row[fields[0]], row[fields[1]], strict=True)
                ),
                "All original trace source/occupancy pairs",
            )
        for i in range(k):
            if not row["slot_occupancy"][i]:
                require(
                    all(
                        row[name][i] == 0
                        for name in ("slot_ages", "slot_ttl", "slot_norms", "slot_write_counts")
                    ),
                    "Canonical empty trace metadata",
                )
            for j, cosine in enumerate(row["workspace_cosine_matrix"][i]):
                require(
                    -1 <= cosine <= 1
                    and abs(cosine - row["workspace_cosine_matrix"][j][i]) <= 1e-12,
                    "Finite symmetric cosine range",
                )
                if row["slot_norms"][i] == 0 or row["slot_norms"][j] == 0:
                    require(cosine == 0, "Zero-norm cosine convention")
                elif i == j:
                    require(abs(cosine - 1.0) <= 1e-12, "Nonzero self cosine")
        for weights in row["read_attention"]:
            total = math.fsum(weights)
            require(
                all(
                    0 <= weight <= 1 and (weight == 0 or row["read_slot_occupancy"][i])
                    for i, weight in enumerate(weights)
                )
                and (total == 0 or abs(total - 1.0) <= 1e-5),
                "Occupied unit-mass or blocked-zero read attention",
            )


def isolation(
    empty: dict[str, Any], trial: dict[str, Any], *, k: int, d: int, runtime: tuple[str, str]
) -> None:
    cycle, role, slot = trial["write_cycle"], trial["role"], trial["slot"]
    a, b = empty["output"], trial["output"]
    original, changed = a["controlled"], b["controlled"]
    require(
        equal(original["trace"]["cycles"][: cycle - 1], changed["trace"]["cycles"][: cycle - 1]),
        "Original trace prefix",
    )
    for index in range(cycle):
        x, y = a["observations"][index], b["observations"][index]
        require(equal(x["before_write"], y["before_write"]), "Original ALL7 prewrite prefix")
        if index < cycle - 1:
            require(
                equal(x, y) and equal(original["states"][index], changed["states"][index]),
                "Whole native prefix",
            )
    x, y = original["trace"]["cycles"][cycle - 1], changed["trace"]["cycles"][cycle - 1]
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
        require(equal(x[name], y[name]), "Original competition/allocation/read metadata")
    for name in ("module_states", "workspace_reads", "read_workspace_slots"):
        require(
            equal(x["full"][name], y["full"][name]), "Original recipient read/goal/module inputs"
        )
    before = a["observations"][cycle - 1]["before_write"]
    written = a["observations"][cycle - 1]["native_written"]
    actual = b["observations"][cycle - 1]["native_written"]
    post = changed["states"][cycle - 1]
    if role == "donor":
        require(equal(post, actual), "No donor post-record intervention")
        for name in FIELDS:
            for index in range(k):
                if name != "slots" or index != slot:
                    require(
                        equal(written[name]["values"][0][index], actual[name]["values"][0][index]),
                        "Only original allocated donor content changes",
                    )
        require(
            equal(
                [p for p in original["messages"] if p["cycle"] <= cycle],
                [p for p in changed["messages"] if p["cycle"] <= cycle],
            ),
            "Recipient proposals unchanged before transplant",
        )
        return
    require(equal(written, actual), "Record control retains original native writer")
    if role == "shuffle":
        seed = changed["plan"]["control"]["interventions"][0]["seed"]
        seed = (
            int(
                domain(
                    "iris/workspace-intervention-rng/v1",
                    dict(kind="shuffle_slots", seed=seed, cycle=cycle),
                )[:16],
                16,
            )
            % 2**63
        )
        permutation = cpu_order(seed, k, runtime)
        for name in FIELDS:
            require(
                equal(
                    post[name]["values"][0], [written[name]["values"][0][i] for i in permutation]
                ),
                "EXACT input-only whole CPU SHUFFLE, identity valid",
            )
        return
    for name in FIELDS:
        for index in range(k):
            expected = written[name]["values"][0][index]
            if index == slot:
                expected = (
                    before[name]["values"][0][index]
                    if role == "block"
                    else [0.0] * d
                    if name == "slots"
                    else -1
                    if name == "source_module"
                    else False
                    if name == "occupancy"
                    else 0.0
                    if name in ("confidence", "ttl")
                    else 0
                )
            require(
                equal(post[name]["values"][0][index], expected),
                "Whole selected ZERO/BLOCK, all other records exact",
            )


def distance(left: list[float], right: list[float]) -> float:
    require(len(left) == len(right) > 0, "Complete receiver endpoint width")
    scale = max(abs(x) for x in left + right)
    if scale == 0:
        return 0.0
    a, b = [x / scale for x in left], [x / scale for x in right]
    return min(1.0, math.dist(a, b) / (math.hypot(*a) + math.hypot(*b)))


def score_bindings(source: dict[str, Any]) -> list[tuple[str, str]]:
    """Recompute EVERY original acknowledgement, including unselected rows."""
    plan = source["evaluation_plan"]["sha256"]
    head = fingerprint(b"iris/scientific-ablation-evaluation-ack/v2/genesis", plan.encode())
    result = []
    for index, row in enumerate(source["evaluation"]["payload"]["rows"]):
        payload = dict(
            kind="iris-scientific-ablation-evaluation-row-v2", plan_sha256=plan, measurements=row
        )
        digest = domain("iris/scientific-ablation-evaluation-row/v2", payload)
        body = canonical(dict(sha256=digest, payload=payload)) + b"\n"
        ack = dict(
            kind="iris-scientific-ablation-evaluation-ack-v2",
            status="acknowledged_score",
            plan_sha256=plan,
            row_index=index,
            previous_ack_sha256=head,
            row_sha256=digest,
            row_file_sha256=sha256(body).hexdigest(),
            row_file_bytes=len(body),
        )
        head = domain("iris/scientific-ablation-evaluation-ack/v2", ack)
        result.append((digest, head))
    complete = source["evaluation_complete"]["payload"]
    require(
        (
            complete["plan_sha256"],
            complete["capture_sha256"],
            complete["evaluation_sha256"],
            complete["acknowledgement_head_sha256"],
            complete["example_count"],
        )
        == (
            plan,
            source["evaluation_plan"]["payload"]["capture"]["sha256"],
            source["evaluation"]["sha256"],
            head,
            len(result),
        ),
        "WHOLE original COMPLETE acknowledgement chain",
    )
    return result


def trace_metadata(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in cycle.items() if key != "full"} for cycle in trace["cycles"]
    ]


def validate(
    report: dict[str, Any], expected: dict[str, Any], *, runtime: tuple[str, str]
) -> dict[str, Any]:
    """ENTIRE candidate against separately trusted original complete preflight.

    expected must not be derived from the candidate; physical/source authentication
    is the caller's explicit responsibility. No hashes attest native execution.
    """
    runtime_guard(runtime)
    candidate = own_json(report)
    original = own_json(expected)
    require(
        type(candidate) is dict and type(original) is dict,
        "Exact report/original preflight dictionaries",
    )
    assert isinstance(candidate, dict) and isinstance(original, dict)
    report, expected = candidate, original
    reservation = sources.preflight(expected)
    descriptor, source = expected["complete_layout"], expected["source_metadata"]
    # Independent original layout closes ALL metadata/primitive families/array counts,
    # including wide effects, BEFORE any candidate comparison/validation/serialization.
    layouts.node_bound(descriptor, report, check=True)
    p = report["payload"]
    require(equal(p["source_metadata"], source), "SEPARATE original source expectation")
    design = source["observed_budget"]["payload"]["design"]["payload"]
    g = design["geometry"]
    m, k, d, c, r = (
        g[n] for n in ("num_modules", "num_slots", "state_dim", "cycle_count", "writers_per_cycle")
    )
    require(
        2 <= m <= 64 and 1 <= k <= 64 and 1 <= d <= 4096 and 2 <= c <= 64 and 1 <= r <= min(m, k),
        "Whole declared original geometry",
    )
    message = bool(design["donor_pairs"])
    protocol = "message" if message else "workspace"
    require(
        report["sha256"] == domain("iris/scientific-ablation-" + protocol + "-writer-report/v1", p),
        "Distinct PRODUCTION report checksum",
    )
    require(
        p["byte_preflight"]
        == dict(
            layout_sha256=reservation["layout_sha256"],
            report_byte_preflight_bound=reservation["bound"],
            maximum_report_bytes=expected["maximum_report_bytes"],
        ),
        "Original whole byte reservation",
    )
    require(
        len(canonical(report)) + 1 <= reservation["bound"] <= expected["maximum_report_bytes"],
        "Actual entire canonical bytes",
    )
    budget = source["observed_budget"]["payload"]
    require(
        p["actual_teacher_executions"] == budget["planned_total_teacher_executions"]
        and p["actual_trace_proposal_observation_scalar_bound"]
        == budget["total_trace_proposal_observation_scalar_bound"]
        and p["actual_post_intervention_scalar_bound"]
        == reservation["post_intervention_scalar_bound"],
        "Original all-execution/scalar counts",
    )
    groups = {"empty": p["controlled_passes"]["empty"], **p["controlled_passes"]["trials"]}
    indexed = {}
    original_checkpoint = source["evaluation"]["payload"]["plan"]["payload"]["source"]["payload"][
        "best_checkpoint_sha256"
    ]
    for role, entries in groups.items():
        for frame in entries:
            key = (frame["row_index"], role, frame["write_cycle"], frame["writer"])
            require(
                frame["role"] == role
                and key not in indexed
                and frame["row_index"] in design["selected_row_indices"],
                "Whole unique original frame roles/rows",
            )
            indexed[key] = frame
            ctl = frame["output"]["controlled"]
            score = source["evaluation"]["payload"]["rows"][frame["row_index"]]
            member = design["members"][frame["row_index"]]
            require(
                (
                    ctl["prompt_sha256"],
                    ctl["target_sha256"],
                    ctl["supervised_token_count"],
                    ctl["teacher_graph_sha256"],
                    ctl["sidecar_state_sha256"],
                    ctl["ablation_policy_sha256"],
                )
                == (
                    member["cognitive_prompt_sha256"],
                    score["teacher_target_sha256"],
                    score["target_token_count"],
                    source["controlled_teacher_graph_sha256"],
                    source["sidecar_sha256"],
                    design["ablation_policy_sha256"],
                ),
                "Every original input/EOS/graph/policy",
            )
            require(
                ctl["trace"]["options"]
                == dict(
                    mode="full",
                    example_id=member["example_id"],
                    checkpoint_id=original_checkpoint,
                    seed=source["run_seed"],
                    maximum_scalars=design["policy"]["maximum_trace_scalars"],
                ),
                "Original trace source/context/options",
            )
            require(
                frame["output"]["maximum_observation_scalars"]
                == budget["maximum_observation_scalars"],
                "Original observation quota",
            )
            output(frame["output"], m=m, k=k, d=d, c=c)
            trace_ranges(ctl["trace"], m=m, k=k, c=c, r=r)
    require(
        len(indexed) == budget["planned_controlled_teacher_executions"],
        "Complete original native pass count",
    )
    pairs = {
        row: (other, pair["sha256"])
        for pair in design["donor_pairs"]
        for row, other in (
            (pair["first_row_index"], pair["second_row_index"]),
            (pair["second_row_index"], pair["first_row_index"]),
        )
    }
    acknowledgements = score_bindings(source)
    keys: dict[str, list[tuple[int, str, int | None, int | None]]] = {role: [] for role in groups}
    derived = []
    require(
        [row["row_index"] for row in p["ordinary_rows"]] == design["selected_row_indices"],
        "Whole ordinary selection order",
    )
    for row in p["ordinary_rows"]:
        index = row["row_index"]
        empty_key = (index, "empty", None, None)
        require(empty_key in indexed, "ONE original EMPTY per selected row")
        empty = indexed[empty_key]
        keys["empty"].append(empty_key)
        ctl = empty["output"]["controlled"]
        score = source["evaluation"]["payload"]["rows"][index]
        require(
            (row["evaluation_row_sha256"], row["evaluation_ack_sha256"]) == acknowledgements[index],
            "Original COMPLETE row/ack bindings",
        )
        require(
            (
                row["capture_row_sha256"],
                row["ordinary_teacher_invocation_sha256"],
                row["target_sha256"],
                row["target_token_count"],
                row["target_nll_sum"],
            )
            == (
                score["capture_row_sha256"],
                score["teacher_invocation_sha256"],
                score["teacher_target_sha256"],
                score["target_token_count"],
                score["target_nll_sum"],
            ),
            "Original ordinary capture/score/EOS/NLL",
        )
        require(
            equal(row["target_nll_sum"], score["target_nll_sum"]), "Bit-faithful original saved NLL"
        )
        require(
            equal(row["ordinary_mean_target_nll"], ctl["loss"])
            and math.isclose(
                row["target_nll_sum"],
                row["ordinary_mean_target_nll"] * row["target_token_count"],
                rel_tol=2e-7,
                abs_tol=1e-7,
            ),
            "Ordinary/EMPTY NLL, separate saved-fp32 rounding bound",
        )
        require(
            equal(ctl["plan"], reference._plan(design, "empty", None, None, None)),
            "Original EMPTY plan",
        )
        options = ctl["trace"]["options"] | {"mode": design["policy"]["ordinary_trace_mode"]}
        trace_ranges(row["ordinary_trace"], m=m, k=k, c=c, r=r)
        require(
            row["ordinary_trace"]["options"] == options
            and row["ordinary_trace"]["intervention_plan_sha256"]
            == design["ablation_policy_sha256"],
            "Original ordinary options/policy",
        )
        require(
            equal(trace_metadata(row["ordinary_trace"]), trace_metadata(ctl["trace"]))
            and (
                options["mode"] != "full"
                or equal(row["ordinary_trace"]["cycles"], ctl["trace"]["cycles"])
            ),
            "Exact ordinary/EMPTY trace parity",
        )
        for cycle in range(1, c):
            written, next_read = ctl["trace"]["cycles"][cycle - 1 : cycle + 1]
            require(
                len(written["write_modules"]) == r
                and len(set(written["selected_slots"])) == r
                and len(set(written["write_modules"])) == r,
                "All actual original distinct allocations",
            )
            if not message:
                shuffle_key = (index, "shuffle", cycle, None)
                require(shuffle_key in indexed, "Original whole SHUFFLE missing")
                keys["shuffle"].append(shuffle_key)
            for writer, slot in zip(
                written["write_modules"], written["selected_slots"], strict=True
            ):
                genuine_key = (index, "block" if message else "zero", cycle, writer)
                negative_key = (index, "donor", cycle, writer) if message else shuffle_key
                require(
                    genuine_key in indexed and negative_key in indexed, "Every actual writer trial"
                )
                genuine, negative = indexed[genuine_key], indexed[negative_key]
                keys[genuine["role"]].append(genuine_key)
                if message:
                    keys["donor"].append(negative_key)
                donor = None
                for trial in (genuine, negative):
                    role = trial["role"]
                    require(
                        role == "shuffle" or trial["slot"] == slot,
                        "Original allocated recipient slot",
                    )
                    seed = None
                    if role == "shuffle":
                        member = design["members"][index]
                        seed = (
                            int(
                                domain(
                                    "iris/scientific-ablation-whole-record-shuffle/v1",
                                    dict(
                                        seed=design["policy"]["shuffle_seed"],
                                        example_id=member["example_id"],
                                        prompt_sha256=member["cognitive_prompt_sha256"],
                                        cycle=cycle,
                                    ),
                                )[:16],
                                16,
                            )
                            % 2**63
                        )
                    if role == "donor":
                        donor_row, pair = pairs[index]
                        require(
                            (trial["donor_row_index"], trial["donor_pair_sha256"])
                            == (donor_row, pair),
                            "Original disjoint donor pair",
                        )
                        donor = indexed[donor_row, "empty", None, None]["output"]["controlled"][
                            "messages"
                        ][(cycle - 1) * m + writer]
                    require(
                        equal(
                            trial["output"]["controlled"]["plan"],
                            reference._plan(
                                design,
                                role,
                                cycle,
                                trial["writer"],
                                trial["slot"],
                                seed=seed,
                                donor=donor,
                            ),
                        ),
                        "Original input-derived plan/paired proposal",
                    )
                    isolation(empty, trial, k=k, d=d, runtime=runtime)
                before = empty["output"]["observations"][cycle - 1]["before_write"]
                changed = genuine["output"]["controlled"]["trace"]["cycles"][cycle]["full"][
                    "module_states"
                ]
                alternative = negative["output"]["controlled"]["trace"]["cycles"][cycle]["full"][
                    "module_states"
                ]
                require(
                    written["slot_occupancy"][slot] and written["slot_sources"][slot] == writer,
                    "Actual original writer record",
                )
                for target in range(m):
                    if target == writer:
                        continue
                    baseline = next_read["full"]["module_states"][target]
                    background = changed[target] if message else baseline
                    actual = baseline if message else changed[target]
                    effect = dict(
                        kind="iris-scientific-ablation-" + protocol + "-writer-effect-v1",
                        row_index=index,
                        write_cycle=cycle,
                        source_module=writer,
                        target_module=target,
                        slot=slot,
                        survives_to_next_read=next_read["read_slot_occupancy"][slot]
                        and next_read["read_slot_sources"][slot] == writer,
                        overwrote_existing_record=before["occupancy"]["values"][0][slot],
                        genuine_distance=distance(background, actual),
                        negative_control_distance=distance(background, alternative[target]),
                        empty_invocation_sha256=empty["output"]["invocation_sha256"],
                        intervention_invocation_sha256=genuine["output"]["invocation_sha256"],
                        negative_control_invocation_sha256=negative["output"]["invocation_sha256"],
                        recipient_message_sha256=None,
                        donor_message_sha256=None,
                        donor_row_index=None,
                        donor_pair_sha256=None,
                    )
                    if message:
                        recipient = ctl["messages"][(cycle - 1) * m + writer]
                        effect.update(
                            recipient_message_sha256=domain(
                                "iris/module-message-snapshot/v1", recipient
                            ),
                            donor_message_sha256=domain("iris/module-message-snapshot/v1", donor),
                            donor_row_index=pairs[index][0],
                            donor_pair_sha256=pairs[index][1],
                        )
                    derived.append(effect)
    require(
        all(
            [
                (frame["row_index"], frame["role"], frame["write_cycle"], frame["writer"])
                for frame in groups[role]
            ]
            == expected_keys
            for role, expected_keys in keys.items()
        ),
        "Exact complete per-role order, no missing/extra passes",
    )
    require(len(derived) == len(p["effects"]), "All original writer to OTHER receiver effects")
    for actual, computed in zip(p["effects"], derived, strict=True):
        for name in actual:
            if name in ("genuine_distance", "negative_control_distance"):
                require(
                    math.isclose(actual[name], computed[name], rel_tol=1e-13, abs_tol=1e-15),
                    "Independent symmetric relative-L2 endpoints",
                )
            else:
                require(
                    equal(actual[name], computed[name]),
                    "All original effect flags/indices/native anchors",
                )
    # Distances have independently passed the tight arithmetic identity bound above;
    # recompute statistics from retained floats to keep declared tolerance/tie decisions exact.
    require(
        equal(p["statistics"], reference._statistics(design, p["effects"])),
        "Complete grouped sign/Holm family",
    )
    return dict(
        protocol=protocol,
        geometry=g,
        observed_passes=len(indexed),
        effects=len(derived),
        all7_states=3 * c * len(indexed),
        maximum_report_bytes=expected["maximum_report_bytes"],
        report_bytes=len(canonical(report)) + 1,
        scope="source-expectation-bound-wire-semantics-not-native-execution-attestation",
    )
