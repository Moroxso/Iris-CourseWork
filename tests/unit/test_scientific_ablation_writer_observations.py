"""Complete wire consistency on retained genuine witnesses; no new teacher calls."""

import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.artifacts.scientific_ablation_writer_observations_v1 import (
    ScientificAblationObservedWriterOutputV1,
    ScientificAblationWriterPassFrameV1,
    writer_observed_wire_v1,
    writer_pass_wire_v1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256

ROOT = Path(__file__).resolve().parents[2]
type Frames = dict[str, list[list[dict[str, Any]]]]


@pytest.fixture(scope="module")
def native_frames() -> Frames:
    receipt_bytes = (
        ROOT / "artifacts/operational/r015-native-writer-readback-20261013-run01.json"
    ).read_bytes()
    assert (
        sha256(receipt_bytes).hexdigest()
        == "db2470b637990d439341f10ba7014b2dfd2527ff18df9c39ec8aac46df4bfe52"
    )
    receipt = json.loads(receipt_bytes)
    assert receipt["passed"] is True and receipt["exact_cases"] == 1371
    result: Frames = {"message": [[] for _ in range(12)], "workspace": [[] for _ in range(12)]}
    for pin in receipt["native_writer_source_files"]:
        path = ROOT / pin["path"]
        raw = path.read_bytes()
        assert len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["sha256"]
        match = re.fullmatch(r"test_all12_both_full_original_([0-9]+)", path.parent.name)
        assert match is not None
        mode = int(match.group(1))
        protocol = "message" if path.name.endswith("message.json") else "workspace"
        groups = json.loads(raw)["projected_report"]["payload"]["controlled_passes"]
        result[protocol][mode] = groups["empty"] + [
            frame for group in groups["trials"].values() for frame in group
        ]
    assert all(
        len(frames) == (18 if protocol == "message" else 21)
        for protocol, modes in result.items()
        for frames in modes
    )
    return result


def frame_for(samples: Frames, role: str) -> dict[str, Any]:
    protocol = "message" if role in ("donor", "message-empty", "block") else "workspace"
    wanted = "empty" if role == "message-empty" else role
    return deepcopy(next(frame for frame in samples[protocol][0] if frame["role"] == wanted))


def _rehash(output: dict[str, Any]) -> None:
    controlled = output["controlled"]
    plan = controlled["plan"]
    controlled["plan_sha256"] = domain_sha256(
        "iris/cognitive-ablation-control-plan/v1", canonical_json_bytes(plan)
    )
    controlled["ablation_policy_sha256"] = domain_sha256(
        "iris/cognitive-ablation-policy/v1", canonical_json_bytes(plan["ablation"])
    )
    control = plan["control"]
    controlled["control_plan_sha256"] = domain_sha256(
        "iris/workspace-message-plan/v1"
        if control["kind"] == "iris-workspace-message-plan-v1"
        else "iris/workspace-intervention-plan/v1",
        canonical_json_bytes(control),
    )
    controlled["trace"]["intervention_plan_sha256"] = controlled["plan_sha256"]
    controlled["invocation_sha256"] = domain_sha256(
        "iris/cognitive-ablation-controlled-teacher-invocation/v1",
        canonical_json_bytes(
            {
                name: controlled[name]
                for name in (
                    "plan_sha256",
                    "prompt_sha256",
                    "target_sha256",
                    "teacher_graph_sha256",
                    "sidecar_state_sha256",
                    "supervised_token_count",
                )
            }
        ),
    )
    for observation in output["observations"]:
        for name in ("before_write", "native_written"):
            observation[name + "_sha256"] = domain_sha256(
                "iris/ablation-workspace-observed-state/v1",
                canonical_json_bytes(
                    {field: tensor["tensor_sha256"] for field, tensor in observation[name].items()}
                ),
            )
    output["invocation_sha256"] = domain_sha256(
        "iris/ablation-workspace-observed-teacher-invocation/v1",
        canonical_json_bytes(
            dict(
                controlled_invocation_sha256=controlled["invocation_sha256"],
                maximum_observation_scalars=output["maximum_observation_scalars"],
                actual_observation_scalars=output["actual_observation_scalars"],
                observations=[
                    {
                        name: row[name]
                        for name in (
                            "cycle",
                            "before_write_sha256",
                            "native_written_sha256",
                        )
                    }
                    for row in output["observations"]
                ],
            )
        ),
    )


def _tensor_change(state: dict[str, Any], name: str, index: int, value: Any) -> None:
    wire = state[name]
    tensor = torch.tensor(wire["values"], dtype=getattr(torch, wire["dtype"]))
    tensor[0, index] = value
    wire["values"], wire["tensor_sha256"] = tensor.tolist(), tensor_sha256(tensor)


@pytest.mark.parametrize("protocol", ("message", "workspace"))
@pytest.mark.parametrize("mode", range(12))
def test_every_retained_native_pass_all_modes_both_protocols_roundtrips_whole_layout(
    native_frames: Frames,
    protocol: str,
    mode: int,
) -> None:
    for raw in native_frames[protocol][mode]:
        value = ScientificAblationWriterPassFrameV1.model_validate(raw)
        assert canonical_json_bytes(writer_pass_wire_v1(value)) == canonical_json_bytes(raw)
        assert canonical_json_bytes(writer_observed_wire_v1(value.output)) == canonical_json_bytes(
            raw["output"]
        )


@pytest.mark.parametrize("role", ("empty", "message-empty", "shuffle", "zero", "block", "donor"))
def test_each_actual_role_keeps_all_phases_and_proposals(native_frames: Frames, role: str) -> None:
    raw = frame_for(native_frames, role)
    value = ScientificAblationWriterPassFrameV1.model_validate(raw)
    assert len(value.output.observations) == len(value.output.controlled.states) == 3
    expected = 12 if role in ("message-empty", "donor") else 0
    assert len(value.output.controlled.messages) == expected
    assert ScientificAblationWriterPassFrameV1.model_validate_json(json.dumps(raw)) == value
    assert (
        ScientificAblationObservedWriterOutputV1.model_validate_json(json.dumps(raw["output"]))
        == value.output
    )


@pytest.mark.parametrize(
    "path,bad",
    (
        (("row_index",), True),
        (("write_cycle",), 1.0),
        (("writer",), False),
        (("slot",), "1"),
        (("donor_row_index",), False),
        (("output", "actual_observation_scalars"), 336.0),
        (("output", "maximum_observation_scalars"), True),
        (("output", "controlled", "loss"), 1),
        (("output", "controlled", "supervised_token_count"), True),
        (("output", "controlled", "trace", "options", "seed"), True),
        (("output", "controlled", "trace", "cycles", 0, "module_bids", 0), 0),
        (("output", "controlled", "trace", "cycles", 0, "slot_ages", 0), 0.0),
        (("output", "controlled", "trace", "cycles", 0, "slot_occupancy", 0), 0),
        (("output", "controlled", "trace", "cycles", 0, "full", "workspace_slots", 0, 0), True),
        (("output", "controlled", "messages", 0, "values", 0), False),
        (("output", "observations", 0, "before_write", "slots", "values", 0, 0, 0), 0),
    ),
)
def test_primitive_families_guarded_before_old_model_coercion(
    native_frames: Frames,
    path: tuple[Any, ...],
    bad: Any,
) -> None:
    raw = frame_for(native_frames, "donor")
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = bad
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize(
    "root",
    (
        "frame",
        "output",
        "controlled",
        "plan",
        "trace",
        "observation",
        "state",
        "tensor",
        "proposal",
    ),
)
def test_every_nested_dictionary_is_closed(native_frames: Frames, root: str) -> None:
    raw = frame_for(native_frames, "donor")
    output = raw["output"]
    locations = dict(
        frame=raw,
        output=output,
        controlled=output["controlled"],
        plan=output["controlled"]["plan"],
        trace=output["controlled"]["trace"],
        observation=output["observations"][0],
        state=output["observations"][0]["before_write"],
        tensor=output["observations"][0]["before_write"]["slots"],
        proposal=output["controlled"]["messages"][0],
    )
    locations[root]["hidden"] = True
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize(
    "name",
    (
        "plan_sha256",
        "ablation_policy_sha256",
        "control_plan_sha256",
        "invocation_sha256",
    ),
)
def test_original_controlled_domains_cannot_be_interchanged(
    native_frames: Frames, name: str
) -> None:
    raw = frame_for(native_frames, "donor")
    raw["output"]["controlled"][name] = "a" * 64
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize(
    "change", ("missing", "reordered", "cycle", "count", "quota", "invocation")
)
def test_rehashed_observation_inventory_and_count_are_rederived(
    native_frames: Frames, change: str
) -> None:
    raw = frame_for(native_frames, "empty")
    output = raw["output"]
    if change == "missing":
        output["observations"].pop()
    elif change == "reordered":
        output["observations"].reverse()
    elif change == "cycle":
        output["observations"][0]["cycle"] = 2
    elif change == "count":
        output["actual_observation_scalars"] -= 1
    elif change == "quota":
        output["maximum_observation_scalars"] = output["actual_observation_scalars"] - 1
    _rehash(output)
    if change == "invocation":
        output["invocation_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "reordered",
        "cycle",
        "module",
        "prompt",
        "graph",
        "sidecar",
        "width",
        "content",
        "quota",
    ),
)
def test_proposals_keep_original_complete_geometry_order_and_context(
    native_frames: Frames, change: str
) -> None:
    raw = frame_for(native_frames, "message-empty")
    output, controlled = raw["output"], raw["output"]["controlled"]
    proposals = controlled["messages"]
    if change == "missing":
        proposals.pop()
    elif change == "reordered":
        proposals.reverse()
    elif change in ("cycle", "module"):
        proposals[0][change] += 1
    elif change == "width":
        proposals[0]["values"].pop()
    elif change == "content":
        proposals[0]["values"][0] = 0.125
    elif change == "quota":
        controlled["plan"]["maximum_message_scalars"] = 95
    else:
        proposals[0][
            {
                "prompt": "prompt_sha256",
                "graph": "teacher_graph_sha256",
                "sidecar": "sidecar_state_sha256",
            }[change]
        ] = "a" * 64
    if change in ("width", "content"):
        proposals[0]["content_sha256"] = domain_sha256(
            "iris/module-message-content/v1",
            canonical_json_bytes(dict(values=proposals[0]["values"])),
        )
    _rehash(output)
    # A different genuine proposal vector alone is internally consistent: its
    # ORIGINAL value must additionally be bound by the full native/source seal.
    if change == "content":
        assert ScientificAblationWriterPassFrameV1.model_validate(raw)
    else:
        with pytest.raises(ValueError):
            ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize(
    "name", ("row_index", "write_cycle", "writer", "slot", "donor_row_index", "donor_pair_sha256")
)
def test_original_frame_applicability_and_plan_allocation_binding(
    native_frames: Frames, name: str
) -> None:
    raw = frame_for(native_frames, "donor")
    raw[name] = raw["row_index"] if name == "donor_row_index" else None
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("role", ("shuffle", "zero", "block", "donor"))
@pytest.mark.parametrize("field", ("write_cycle", "writer", "slot"))
def test_frame_wrong_cycle_writer_or_slot_cannot_relabel_actual_plan(
    native_frames: Frames,
    role: str,
    field: str,
) -> None:
    raw = frame_for(native_frames, role)
    if raw[field] is None:
        raw[field] = 0
    else:
        raw[field] += 1
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("role", ("empty", "message-empty", "shuffle", "zero", "block", "donor"))
def test_missing_confidence_is_not_replaced_with_an_invented_default(
    native_frames: Frames, role: str
) -> None:
    raw = frame_for(native_frames, role)
    del raw["output"]["controlled"]["states"][0]["confidence"]
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("field", ("age", "ttl", "source_module", "write_count", "slots"))
def test_locally_rehashed_poststate_still_must_match_original_full_trace(
    native_frames: Frames, field: str
) -> None:
    raw = frame_for(native_frames, "empty")
    output = raw["output"]
    state = output["controlled"]["states"][0]
    index = state["occupancy"]["values"][0].index(True)
    _tensor_change(state, field, index, 0.5 if field in ("ttl", "slots") else 3)
    _rehash(output)
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("phase", ("before_write", "native_written"))
def test_rehashed_phase_change_cannot_hide_read_or_unplanned_post_drift(
    native_frames: Frames, phase: str
) -> None:
    raw = frame_for(native_frames, "empty")
    output = raw["output"]
    state = output["observations"][1][phase]
    index = state["occupancy"]["values"][0].index(True)
    _tensor_change(state, "slots", index, 0.125)
    _rehash(output)
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


def test_signed_zero_trace_vs_state_comparison_is_byte_faithful(native_frames: Frames) -> None:
    raw = frame_for(native_frames, "empty")
    state = raw["output"]["controlled"]["states"][0]
    index = state["occupancy"]["values"][0].index(False)
    _tensor_change(state, "slots", index, -0.0)
    with pytest.raises(ValueError, match="full trace"):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("role", ("zero", "block", "shuffle"))
def test_self_rehashed_wrong_whole_record_operation_is_denied(
    native_frames: Frames, role: str
) -> None:
    raw = frame_for(native_frames, role)
    output, controlled = raw["output"], raw["output"]["controlled"]
    cycle = raw["write_cycle"] - 1
    post = controlled["states"][cycle]
    # Confidence has no post trace field; only ALL7 phase validation can catch it.
    occupied = post["occupancy"]["values"][0]
    if not any(occupied):
        pytest.fail("This genuine retained phase must include an occupied record")
    index = occupied.index(True)
    original = post["confidence"]["values"][0][index]
    _tensor_change(post, "confidence", index, 0.25 if original == 0.125 else 0.125)
    _rehash(output)
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)


@pytest.mark.parametrize("field", ("messages", "states", "observations", "plan", "loss"))
def test_formatter_rechecks_reflected_nested_metadata(native_frames: Frames, field: str) -> None:
    value = ScientificAblationWriterPassFrameV1.model_validate(frame_for(native_frames, "donor"))
    target = value.output if field == "observations" else value.output.controlled
    bad: object = [] if field in ("messages", "states", "observations") else 1
    object.__setattr__(target, field, bad)
    with pytest.raises(ValueError):
        writer_pass_wire_v1(value)


@pytest.mark.parametrize("field", ("output", "controlled", "plan", "trace"))
def test_formatter_does_not_coerce_reflected_raw_dictionary_into_owned_model(
    native_frames: Frames, field: str
) -> None:
    raw = frame_for(native_frames, "donor")
    value = ScientificAblationWriterPassFrameV1.model_validate(raw)
    target = (
        value
        if field == "output"
        else value.output
        if field == "controlled"
        else value.output.controlled
    )
    replacement = (
        raw["output"]
        if field == "output"
        else raw["output"]["controlled"]
        if field == "controlled"
        else raw["output"]["controlled"][field]
    )
    object.__setattr__(target, field, replacement)
    with pytest.raises(ValueError, match="immutable model"):
        writer_pass_wire_v1(value)


def test_candidate_plan_subclass_serializer_never_runs(native_frames: Frames) -> None:
    calls: list[str] = []

    class ForeignPlan(CognitiveAblationControlPlanV1):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append("serializer")
            raise AssertionError("Candidate serializer called")

    value = ScientificAblationWriterPassFrameV1.model_validate(frame_for(native_frames, "donor"))
    object.__setattr__(value.output.controlled, "plan", ForeignPlan.model_construct())
    with pytest.raises(ValueError):
        writer_pass_wire_v1(value)
    assert calls == []


def test_foreign_root_metaclass_dictionary_key_and_float_callbacks_denied(
    native_frames: Frames,
) -> None:
    calls: list[str] = []

    class ForeignMeta(type):
        def __hash__(cls) -> int:
            calls.append("class hash")
            raise AssertionError("Foreign metaclass hash called")

        def __eq__(cls, other: object) -> bool:
            calls.append("class equality")
            raise AssertionError("Foreign metaclass equality called")

    class Foreign(metaclass=ForeignMeta):
        def __getattribute__(self, name: str) -> Any:
            calls.append("attribute")
            raise AssertionError("Foreign attribute called")

    class ForeignKey(str):
        def __hash__(self) -> int:
            calls.append("key hash")
            return str.__hash__(self)

    class ForeignFloat(float):
        def __float__(self) -> float:
            calls.append("float conversion")
            raise AssertionError("Foreign float conversion called")

    with pytest.raises(ValueError):
        writer_pass_wire_v1(Foreign())
    with pytest.raises(ValueError):
        writer_observed_wire_v1(Foreign())
    raw = frame_for(native_frames, "donor")
    raw[ForeignKey("hidden")] = True
    calls.clear()
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)
    raw = frame_for(native_frames, "donor")
    raw["output"]["controlled"]["loss"] = ForeignFloat(1.0)
    with pytest.raises(ValueError):
        ScientificAblationWriterPassFrameV1.model_validate(raw)
    assert calls == []


def test_formatter_requires_exact_roots_even_model_construct_subclasses(
    native_frames: Frames,
) -> None:
    class ForeignFrame(ScientificAblationWriterPassFrameV1):
        pass

    raw = frame_for(native_frames, "donor")
    with pytest.raises(ValueError):
        writer_pass_wire_v1(ForeignFrame.model_construct(**raw))
    with pytest.raises(ValueError):
        writer_observed_wire_v1(raw["output"])


def test_owned_projection_is_detached_and_not_execution_or_source_attestation(
    native_frames: Frames,
) -> None:
    raw = frame_for(native_frames, "message-empty")
    # Consistently replacing every local prompt identity remains a locally valid
    # wire object. The full report validator MUST bind the ORIGINAL trusted source.
    controlled = raw["output"]["controlled"]
    controlled["prompt_sha256"] = "a" * 64
    for proposal in controlled["messages"]:
        proposal["prompt_sha256"] = "a" * 64
    _rehash(raw["output"])
    value = ScientificAblationWriterPassFrameV1.model_validate(raw)
    wire = writer_pass_wire_v1(value)
    assert canonical_json_bytes(wire) == canonical_json_bytes(raw)
    raw["output"]["controlled"]["loss"] += 1.0
    assert canonical_json_bytes(writer_pass_wire_v1(value)) == canonical_json_bytes(wire)
