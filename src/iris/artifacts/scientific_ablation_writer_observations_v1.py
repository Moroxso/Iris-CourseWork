"""Complete typed native writer outputs and pass frames for report construction.

All reserved values/phases are retained. Validators establish local consistency,
NOT original source authentication, pass isolation, effects or execution. The full
report service must still bind original source/preflight/native retention seals.
No source transaction, teacher forward, publication or report return occurs here.
"""

from collections import Counter
from math import isfinite
from types import UnionType
from typing import Annotated, Literal, Self, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.artifacts.scientific_ablation_writer_tensors_v1 import (
    ScientificAblationWriterStateV1,
    writer_state_sha256_v1,
    writer_state_wire_v1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest
from iris.interventions.messages import ModuleMessageSnapshotV1, WorkspaceMessagePlanV1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.trace import WorkspaceTraceV1


def _raw_model(value: object, expected: type[BaseModel], depth: int = 0) -> dict[str, object]:
    """Own declared primitive families BEFORE candidate validators/serializers."""
    if depth > 32:
        raise ValueError("Writer metadata depth exceeded")
    immutable = type(value) is expected
    if type(value) is dict:
        data = value
    else:
        # No candidate metaclass hashing/equality or attributes before identity.
        if not immutable:
            raise ValueError("Exact declared writer metadata model required")
        assert isinstance(value, BaseModel)
        data = vars(value)
        fields = value.__pydantic_fields_set__
        if (
            value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
            or type(fields) is not set
            or any(type(name) is not str for name in fields)
            or not fields.issubset(expected.model_fields)
        ):
            raise ValueError("Hidden writer model metadata forbidden")
    if type(data) is not dict:
        raise ValueError("Exact writer metadata dictionary required")
    assert isinstance(data, dict)
    if any(type(name) is not str for name in data) or set(data) != set(expected.model_fields):
        raise ValueError("Complete exact writer field inventory required")
    return {
        name: _annotation(field.annotation, data[name], immutable=immutable, depth=depth + 1)
        for name, field in expected.model_fields.items()
    }


def _annotation(annotation: object, value: object, *, immutable: bool, depth: int) -> object:
    if depth > 32:
        raise ValueError("Writer metadata depth exceeded")
    # Primitive leaves dominate full-width reports. Their exact family/finite
    # checks do not require reinterpreting a trusted typing annotation per value.
    if (
        annotation is float
        or annotation is int
        or annotation is bool
        or annotation is str
        or annotation is type(None)
    ):
        if type(value) is not annotation:
            raise ValueError("Exact declared writer primitive family required")
        if type(value) is float:
            assert isinstance(value, float)
            if not isfinite(value):
                raise ValueError("Finite writer floating metadata required")
        return value
    if isinstance(annotation, TypeAliasType):
        return _annotation(annotation.__value__, value, immutable=immutable, depth=depth)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _annotation(args[0], value, immutable=immutable, depth=depth)
    if origin is Literal:
        if not any(type(value) is type(literal) and value == literal for literal in args):
            raise ValueError("Exact declared writer literal required")
        return value
    if origin is UnionType or origin is Union:
        for option in args:
            try:
                return _annotation(option, value, immutable=immutable, depth=depth)
            except ValueError:
                continue
        raise ValueError("Writer field differs from every declared alternative")
    if origin is tuple:
        if type(value) is not tuple and (immutable or type(value) is not list):
            raise ValueError("Exact immutable writer sequence or raw JSON array required")
        assert isinstance(value, (tuple, list))
        if len(value) > 8192:
            raise ValueError("Writer metadata sequence exceeds the declared component bound")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _annotation(args[0], child, immutable=immutable, depth=depth + 1) for child in value
            )
        if len(value) != len(args):
            raise ValueError("Writer tuple length differs from its declared shape")
        return tuple(
            _annotation(item, child, immutable=immutable, depth=depth + 1)
            for item, child in zip(args, value, strict=True)
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if immutable and type(value) is not annotation:
            raise ValueError("Owned writer fields must retain their exact immutable model family")
        return _raw_model(value, annotation, depth)
    raise ValueError("Unsupported declared writer metadata annotation")


class _WriterWire(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    @model_validator(mode="before")
    @classmethod
    def exact_input(cls, value: object) -> object:
        if not any(
            cls is expected
            for expected in (
                ScientificAblationWriterObservationV1,
                ScientificAblationControlledWriterOutputV1,
                ScientificAblationObservedWriterOutputV1,
                ScientificAblationWriterPassFrameV1,
            )
        ):
            raise ValueError("Exact declared writer component class required")
        return _raw_model(value, cls)


class ScientificAblationWriterObservationV1(_WriterWire):
    cycle: int = Field(ge=1, le=64)
    before_write: ScientificAblationWriterStateV1
    native_written: ScientificAblationWriterStateV1
    before_write_sha256: Digest
    native_written_sha256: Digest

    @model_validator(mode="after")
    def state_bindings(self) -> Self:
        if (
            writer_state_sha256_v1(self.before_write, num_modules=64) != self.before_write_sha256
            or writer_state_sha256_v1(self.native_written, num_modules=64)
            != self.native_written_sha256
        ):
            raise ValueError(
                "Observation state digest differs from its complete exact tensor bytes"
            )
        return self


class ScientificAblationControlledWriterOutputV1(_WriterWire):
    kind: Literal["iris-ablation-controlled-teacher-output-v1"]
    loss: float = Field(ge=0)
    supervised_token_count: int = Field(ge=1, le=2**63 - 1)
    plan: CognitiveAblationControlPlanV1
    plan_sha256: Digest
    ablation_policy_sha256: Digest
    control_plan_sha256: Digest
    prompt_sha256: Digest
    target_sha256: Digest
    teacher_graph_sha256: Digest
    sidecar_state_sha256: Digest
    invocation_sha256: Digest
    trace: WorkspaceTraceV1
    messages: tuple[ModuleMessageSnapshotV1, ...] = Field(max_length=4096)
    states: tuple[ScientificAblationWriterStateV1, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def local_consistency(self) -> Self:
        trace = self.trace
        m, k, d, c = trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count
        if (
            trace.options.mode != "full"
            or not 2 <= m <= 64
            or not 1 <= k <= 64
            or not 1 <= d <= 4096
            or not 1 <= c <= 64
            or len(self.states) != c
            or trace.intervention_plan_sha256 != self.plan_sha256
        ):
            raise ValueError("Complete full-trace writer output geometry or plan binding differs")
        writers = len(trace.cycles[0].write_modules)
        if not 1 <= writers <= min(m, k) or any(
            len(row.write_modules) != writers for row in trace.cycles
        ):
            raise ValueError("Writer output must retain each cycle's original allocated writers")
        control = self.plan.control
        message = type(control) is WorkspaceMessagePlanV1
        if type(control) is WorkspaceInterventionPlanV1:
            control.validate_dimensions(
                num_slots=k, num_modules=m, state_dim=d, cycles=c, winners_per_cycle=writers
            )
        control_domain = (
            "iris/workspace-message-plan/v1" if message else "iris/workspace-intervention-plan/v1"
        )
        if (
            domain_sha256(
                "iris/cognitive-ablation-control-plan/v1",
                canonical_json_bytes(self.plan.model_dump(mode="json")),
            )
            != self.plan_sha256
            or domain_sha256(
                "iris/cognitive-ablation-policy/v1",
                canonical_json_bytes(self.plan.ablation.model_dump(mode="json")),
            )
            != self.ablation_policy_sha256
            or domain_sha256(control_domain, canonical_json_bytes(control.model_dump(mode="json")))
            != self.control_plan_sha256
            or domain_sha256(
                "iris/cognitive-ablation-controlled-teacher-invocation/v1",
                canonical_json_bytes(
                    {
                        name: getattr(self, name)
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
            != self.invocation_sha256
        ):
            raise ValueError("Original controlled plan/policy/control/invocation domains differ")
        dtype = self.states[0].slots.dtype
        for state, row in zip(self.states, trace.cycles, strict=True):
            writer_state_wire_v1(state, num_modules=m)
            if state.slots.shape != (1, k, d) or state.slots.dtype != dtype:
                raise ValueError(
                    "Poststate complete geometry or native dtype changes within output"
                )
            if row.full is None or canonical_json_bytes(
                state.slots.values[0]
            ) != canonical_json_bytes(row.full.workspace_slots):
                raise ValueError("Full poststate slots differ from original full trace values")
            for field, traced in (
                (state.age, row.slot_ages),
                (state.ttl, row.slot_ttl),
                (state.occupancy, row.slot_occupancy),
                (state.source_module, row.slot_sources),
                (state.write_count, row.slot_write_counts),
            ):
                if canonical_json_bytes(field.values[0]) != canonical_json_bytes(traced):
                    raise ValueError("Poststate metadata differs from original full trace")
        expected_messages = (
            tuple((cycle, module) for cycle in range(1, c + 1) for module in range(m))
            if message
            else ()
        )
        if (
            tuple((proposal.cycle, proposal.module) for proposal in self.messages)
            != expected_messages
        ):
            raise ValueError("Complete ordered original proposal inventory required")
        if message:
            limit = self.plan.maximum_message_scalars
            if limit is None or c * m * d > limit:
                raise ValueError("Complete proposal inventory exceeds its declared original quota")
        for proposal in self.messages:
            if (
                proposal.prompt_sha256 != self.prompt_sha256
                or proposal.teacher_graph_sha256 != self.teacher_graph_sha256
                or proposal.sidecar_state_sha256 != self.sidecar_state_sha256
                or proposal.example_id != trace.options.example_id
                or len(proposal.values) != d
            ):
                raise ValueError(
                    "Proposal context or complete width differs from its original output"
                )
        if type(control) is WorkspaceMessagePlanV1 and control.donor is not None:
            donor = control.donor
            if (
                donor.teacher_graph_sha256 != self.teacher_graph_sha256
                or donor.sidecar_state_sha256 != self.sidecar_state_sha256
                or donor.cycle > c
                or donor.module >= m
                or len(donor.values) != d
            ):
                raise ValueError("Donor model/sidecar/geometry differs from recipient output")
        return self


class ScientificAblationObservedWriterOutputV1(_WriterWire):
    kind: Literal["iris-ablation-observed-teacher-output-v1"]
    controlled: ScientificAblationControlledWriterOutputV1
    maximum_observation_scalars: int = Field(ge=1, le=2_000_000)
    actual_observation_scalars: int = Field(ge=1, le=2_000_000)
    invocation_sha256: Digest
    observations: tuple[ScientificAblationWriterObservationV1, ...] = Field(
        min_length=1, max_length=64
    )

    @model_validator(mode="after")
    def complete_observations(self) -> Self:
        controlled, trace = self.controlled, self.controlled.trace
        m, k, d, c = trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count
        expected = 2 * c * k * (d + 6)
        if (
            self.actual_observation_scalars != expected
            or expected > self.maximum_observation_scalars
            or tuple(row.cycle for row in self.observations) != tuple(range(1, c + 1))
        ):
            raise ValueError(
                "Complete ordered observation inventory or original scalar quota differs"
            )
        for observation, post, row in zip(
            self.observations, controlled.states, trace.cycles, strict=True
        ):
            for state in (observation.before_write, observation.native_written):
                writer_state_wire_v1(state, num_modules=m)
                if state.slots.shape != (1, k, d) or state.slots.dtype != post.slots.dtype:
                    raise ValueError("All three native phases must retain the same geometry/dtype")
            before = observation.before_write
            assert row.full is not None
            if (
                canonical_json_bytes(before.slots.values[0])
                != canonical_json_bytes(row.full.read_workspace_slots)
                or before.occupancy.values[0] != row.read_slot_occupancy
                or before.source_module.values[0] != row.read_slot_sources
                or tuple(slot for slot in row.selected_slots if before.occupancy.values[0][slot])
                != row.overwritten_slots
            ):
                raise ValueError("Actual prewrite/read/occupied-overwrite trace bindings differ")
            _post_write_phase(controlled, observation, post)
        invocation = dict(
            controlled_invocation_sha256=controlled.invocation_sha256,
            maximum_observation_scalars=self.maximum_observation_scalars,
            actual_observation_scalars=expected,
            observations=[
                dict(
                    cycle=row.cycle,
                    before_write_sha256=row.before_write_sha256,
                    native_written_sha256=row.native_written_sha256,
                )
                for row in self.observations
            ],
        )
        if (
            domain_sha256(
                "iris/ablation-workspace-observed-teacher-invocation/v1",
                canonical_json_bytes(invocation),
            )
            != self.invocation_sha256
        ):
            raise ValueError("Observed invocation differs from its original phase bindings")
        return self


def _phase_values(state: ScientificAblationWriterStateV1) -> dict[str, tuple[object, ...]]:
    # Called only after the complete exact model/primitive/state ownership pass.
    return {
        "slots": state.slots.values[0],
        "age": state.age.values[0],
        "confidence": state.confidence.values[0],
        "ttl": state.ttl.values[0],
        "occupancy": state.occupancy.values[0],
        "source_module": state.source_module.values[0],
        "write_count": state.write_count.values[0],
    }


def _post_write_phase(
    controlled: ScientificAblationControlledWriterOutputV1,
    observation: ScientificAblationWriterObservationV1,
    post: ScientificAblationWriterStateV1,
) -> None:
    """Own-output phase consistency, NOT original EMPTY/trial isolation.

    SHUFFLE here checks the whole-record multiset only. Exact version-bound CPU
    order and input-derived seed remain mandatory in the full report validator.
    """
    control = controlled.plan.control
    native_values, post_values = _phase_values(observation.native_written), _phase_values(post)
    before_values = _phase_values(observation.before_write)
    operation = None
    if type(control) is WorkspaceInterventionPlanV1:
        if len(control.interventions) > 1 or any(
            op.kind not in ("zero_slot", "shuffle_slots", "block_module_write")
            or op.cycles is None
            or len(op.cycles) != 1
            for op in control.interventions
        ):
            raise ValueError("Complete writer output requires its reserved single-operation plan")
        if control.interventions and control.interventions[0].cycles == (observation.cycle,):
            operation = control.interventions[0]
    if operation is None:
        if canonical_json_bytes(native_values) != canonical_json_bytes(post_values):
            raise ValueError("Unplanned post-writer state change, including EMPTY/DONOR")
        return
    if operation.kind == "shuffle_slots":
        slots = post.slots.shape[1]

        def records(values: dict[str, tuple[object, ...]]) -> Counter[bytes]:
            return Counter(
                canonical_json_bytes({name: items[index] for name, items in values.items()})
                for index in range(slots)
            )

        if records(native_values) != records(post_values):
            raise ValueError(
                "SHUFFLE must preserve every complete native record including duplicates"
            )
        return
    if operation.kind == "zero_slot":
        slot = operation.slot
    else:
        row = controlled.trace.cycles[observation.cycle - 1]
        if operation.module not in row.write_modules:
            raise ValueError("BLOCK must target an actual allocated recipient writer")
        slot = row.selected_slots[row.write_modules.index(operation.module)]
    assert slot is not None
    expected: dict[str, list[object]] = {
        name: list(values) for name, values in native_values.items()
    }
    for name, values in expected.items():
        if operation.kind == "block_module_write":
            values[slot] = before_values[name][slot]
        else:
            values[slot] = (
                (0.0,) * post.slots.shape[2]
                if name == "slots"
                else -1
                if name == "source_module"
                else False
                if name == "occupancy"
                else 0.0
                if name in ("confidence", "ttl")
                else 0
            )
    if canonical_json_bytes(expected) != canonical_json_bytes(post_values):
        raise ValueError("Post-writer ZERO/BLOCK differs from the real complete record operation")


class ScientificAblationWriterPassFrameV1(_WriterWire):
    row_index: int = Field(ge=0, le=4999)
    role: Literal["empty", "shuffle", "zero", "block", "donor"]
    write_cycle: int | None = Field(ge=1, le=63)
    writer: int | None = Field(ge=0, le=63)
    slot: int | None = Field(ge=0, le=63)
    donor_row_index: int | None = Field(ge=0, le=4999)
    donor_pair_sha256: Digest | None
    output: ScientificAblationObservedWriterOutputV1

    @model_validator(mode="after")
    def role_binding(self) -> Self:
        empty, allocated, donor = (
            self.role == "empty",
            self.role in ("zero", "block", "donor"),
            self.role == "donor",
        )
        if (
            (self.write_cycle is None) != empty
            or (self.writer is not None) != allocated
            or (self.slot is not None) != allocated
            or (self.donor_row_index is not None) != donor
            or (self.donor_pair_sha256 is not None) != donor
            or (donor and self.donor_row_index == self.row_index)
        ):
            raise ValueError("Pass role has missing or inapplicable original frame metadata")
        controlled, trace = self.output.controlled, self.output.controlled.trace
        control = controlled.plan.control
        if empty:
            if (type(control) is WorkspaceInterventionPlanV1 and control.interventions) or (
                type(control) is WorkspaceMessagePlanV1 and control.donor is not None
            ):
                raise ValueError("EMPTY frame must retain its actual empty native plan")
            return self
        cycle = self.write_cycle
        assert cycle is not None
        if cycle >= trace.cycle_count:
            raise ValueError("Writer trial requires its original next-cycle receiver endpoint")
        row = trace.cycles[cycle - 1]
        if allocated and (self.writer, self.slot) not in tuple(
            zip(row.write_modules, row.selected_slots, strict=True)
        ):
            raise ValueError("Frame writer/slot differs from actual original recipient allocation")
        if donor:
            if type(control) is not WorkspaceMessagePlanV1 or control.donor is None:
                raise ValueError("DONOR frame must carry its actual complete donor proposal")
            if control.donor.cycle != cycle or control.donor.module != self.writer:
                raise ValueError("DONOR frame cycle/module differs from actual paired proposal")
            return self
        if type(control) is not WorkspaceInterventionPlanV1 or len(control.interventions) != 1:
            raise ValueError("Writer record frame requires exactly its original operation")
        operation = control.interventions[0]
        expected_kind = {
            "shuffle": "shuffle_slots",
            "zero": "zero_slot",
            "block": "block_module_write",
        }[self.role]
        if (
            operation.kind != expected_kind
            or operation.cycles != (cycle,)
            or (self.role == "zero" and operation.slot != self.slot)
            or (self.role == "block" and operation.module != self.writer)
        ):
            raise ValueError("Frame role/cycle/writer/slot differs from its actual native plan")
        return self


def writer_observed_wire_v1(value: object) -> dict[str, object]:
    """Own and revalidate complete local metadata; NOT original source acceptance."""
    if type(value) is not ScientificAblationObservedWriterOutputV1:
        raise ValueError("Exact complete observed writer wire root required")
    owned = ScientificAblationObservedWriterOutputV1.model_validate(
        _raw_model(value, ScientificAblationObservedWriterOutputV1)
    )
    return owned.model_dump(mode="json", warnings=False)


def writer_pass_wire_v1(value: object) -> dict[str, object]:
    """Retain the entire reserved frame/output; no callback/partial report return."""
    if type(value) is not ScientificAblationWriterPassFrameV1:
        raise ValueError("Exact complete writer frame wire root required")
    owned = ScientificAblationWriterPassFrameV1.model_validate(
        _raw_model(value, ScientificAblationWriterPassFrameV1)
    )
    return owned.model_dump(mode="json", warnings=False)
