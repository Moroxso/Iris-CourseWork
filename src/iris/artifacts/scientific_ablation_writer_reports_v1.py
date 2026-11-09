"""Complete source-expectation-bound writer reports, not execution attestation.

All reserved fields survive. Acceptance requires a SEPARATE original preflight
and CPU stream expectation. Reconstructing those from a candidate proves only
consistency; physical source/native retention belongs to the measurement service.
No teacher, generation, scoring, publication or source transaction is performed.
"""

import json
from dataclasses import dataclass, fields
from math import isclose
from typing import Literal, Self, TypedDict

from pydantic import BaseModel, Field, model_validator

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.artifacts.scientific_ablation_causal_controls_v1 import _empty_plan, _score_bindings
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    IrisScientificAblationCausalDesignV1,
    ScientificAblationMessageCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    IrisScientificAblationCausalObservedBudgetV1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    ScientificAblationCausalReportBytePreflightV1,
    causal_report_layout_bytes_v1,
    validate_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.artifacts.scientific_ablation_writer_observations_v1 import (
    ScientificAblationWriterPassFrameV1,
    _annotation,
    _raw_model,
)
from iris.artifacts.scientific_ablation_writer_statistics_v1 import (
    ScientificAblationMessageWriterEffectV1,
    ScientificAblationMessageWriterStatisticsV1,
    ScientificAblationWorkspaceWriterEffectV1,
    ScientificAblationWorkspaceWriterStatisticsV1,
    WriterEffect,
    build_scientific_ablation_writer_statistics_v1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.interventions.messages import (
    ModuleMessageSnapshotV1,
    WorkspaceMessagePlanV1,
    message_snapshot_sha256_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import relative_state_distance_v1
from iris.workspace.trace import WorkspaceTraceV1

CPU_VERSION = "2.13.0+cpu"
CPU_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
_MAX_EFFECTS = 8192 * 63  # Execution bound times ALL other receivers, NOT 8192 effects.


@dataclass(frozen=True, slots=True)
class ScientificAblationWriterCPURuntimeV1:
    """Separately trusted operational expectation; NEVER a report/self-attestation field."""

    torch_version: str
    torch_git_version: str


def _runtime(value: ScientificAblationWriterCPURuntimeV1) -> None:
    if type(value) is not ScientificAblationWriterCPURuntimeV1:
        raise ValueError("Exact separately trusted CPU stream expectation required")
    if (
        type(value.torch_version) is not str
        or type(value.torch_git_version) is not str
        or value.torch_version != CPU_VERSION
        or value.torch_git_version != CPU_REVISION
    ):
        raise ValueError("CPU permutation stream revision needs a new verified parity proof")


def writer_cpu_permutation_v1(
    seed: int, slots: int, *, runtime: ScientificAblationWriterCPURuntimeV1
) -> tuple[int, ...]:
    """Bounded CPU forward-swap/MT19937 mathematics, no Torch RNG/model calls.

    Source revision: PyTorch CPU TensorFactories.cpp and core/MT19937RNGEngine.h
    at CPU_REVISION. This independently written bounded algorithm uses at most
    63 draws, one in-place twist, low32 seed initialization and identity/no reroll.
    Not Python's random.shuffle; never silently apply to a different runtime.
    """
    _runtime(runtime)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Exact bounded CPU seed required")
    if type(slots) is not int or not 1 <= slots <= 64:
        raise ValueError("Exact complete slot bound required")
    order = list(range(slots))
    if slots == 1:
        return (0,)
    state = [seed & 0xFFFFFFFF]
    for index in range(1, 624):
        previous = state[-1]
        state.append((1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF)
    for index in range(624):
        joined = (state[index] & 0x80000000) | (state[(index + 1) % 624] & 0x7FFFFFFF)
        state[index] = (
            state[(index + 397) % 624] ^ (joined >> 1) ^ (0x9908B0DF if joined & 1 else 0)
        )
    for index in range(slots - 1):
        draw = state[index]
        draw ^= draw >> 11
        draw ^= (draw << 7) & 0x9D2C5680
        draw ^= (draw << 15) & 0xEFC60000
        draw ^= draw >> 18
        other = index + draw % (slots - index)
        order[index], order[other] = order[other], order[index]
    return tuple(order)


def _report_raw(value: object, expected: type[BaseModel]) -> dict[str, object]:
    """Own each wide effect individually BEFORE any candidate validator/serializer."""
    immutable = type(value) is expected
    if type(value) is dict:
        data = value
    else:
        if not immutable:
            raise ValueError("Exact declared complete writer report model required")
        assert isinstance(value, BaseModel)
        data = vars(value)
        assigned = value.__pydantic_fields_set__
        if (
            value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
            or type(assigned) is not set
            or any(type(key) is not str for key in assigned)
            or not assigned.issubset(expected.model_fields)
        ):
            raise ValueError("Hidden complete writer report metadata forbidden")
    if type(data) is not dict:
        raise ValueError("Plain complete report dictionary required")
    assert isinstance(data, dict)
    if any(type(key) is not str for key in data) or set(data) != set(expected.model_fields):
        raise ValueError("Exact complete report field inventory required")
    result: dict[str, object] = {}
    for name, field in expected.model_fields.items():
        child = data[name]
        if name == "payload":
            payload_type = (
                ScientificAblationWorkspaceWriterReportPayloadV1
                if expected is IrisScientificAblationWorkspaceWriterReportV1
                else ScientificAblationMessageWriterReportPayloadV1
            )
            if immutable and type(child) is not payload_type:
                raise ValueError("Owned report payload must retain its declared model family")
            result[name] = _report_raw(child, payload_type)
        elif name == "effects":
            if type(child) is not tuple and (immutable or type(child) is not list):
                raise ValueError("Exact wide effect sequence required")
            assert isinstance(child, (tuple, list))
            if len(child) > _MAX_EFFECTS:
                raise ValueError("Effects exceed the complete execution/receiver bound")
            effect_type = (
                ScientificAblationWorkspaceWriterEffectV1
                if expected is ScientificAblationWorkspaceWriterReportPayloadV1
                else ScientificAblationMessageWriterEffectV1
            )
            if immutable and any(type(effect) is not effect_type for effect in child):
                raise ValueError("Owned effects must retain their exact distinct family")
            result[name] = tuple(_raw_model(effect, effect_type) for effect in child)
        else:
            result[name] = _annotation(field.annotation, child, immutable=immutable, depth=1)
    return result


class _ReportWire(_Strict):
    @model_validator(mode="before")
    @classmethod
    def owned(cls, value: object) -> object:
        if not any(cls is declared for declared in _REPORT_TYPES):
            raise ValueError("Exact declared complete report class required")
        return _report_raw(value, cls)


class ScientificAblationWriterSourceMetadataV1(_ReportWire):
    observed_budget: IrisScientificAblationCausalObservedBudgetV1
    evaluation_plan: IrisScientificAblationEvaluationPlanV2
    evaluation: IrisScientificAblationEvaluationV2
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2
    run_seed: int = Field(ge=0, le=2**63 - 1)
    ordinary_teacher_graph_sha256: Digest
    controlled_teacher_graph_sha256: Digest
    sidecar_sha256: Digest


class ScientificAblationWriterBytePreflightV1(_ReportWire):
    layout_sha256: Digest
    report_byte_preflight_bound: int = Field(ge=1)
    maximum_report_bytes: int = Field(ge=1, le=512 * 1024**2)


class ScientificAblationWriterOrdinaryRowV1(_ReportWire):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    evaluation_row_sha256: Digest
    evaluation_ack_sha256: Digest
    ordinary_teacher_invocation_sha256: Digest
    target_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    target_nll_sum: float = Field(ge=0)
    ordinary_mean_target_nll: float = Field(ge=0)
    ordinary_trace: WorkspaceTraceV1


class ScientificAblationWorkspaceWriterTrialsV1(_ReportWire):
    shuffle: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(max_length=8192)
    zero: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(max_length=8192)


class ScientificAblationMessageWriterTrialsV1(_ReportWire):
    block: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(max_length=8192)
    donor: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(max_length=8192)


class ScientificAblationWorkspaceWriterPassesV1(_ReportWire):
    empty: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(min_length=1, max_length=768)
    trials: ScientificAblationWorkspaceWriterTrialsV1


class ScientificAblationMessageWriterPassesV1(_ReportWire):
    empty: tuple[ScientificAblationWriterPassFrameV1, ...] = Field(min_length=1, max_length=768)
    trials: ScientificAblationMessageWriterTrialsV1


class _PayloadFields(_ReportWire):
    scope: Literal["complete-teacher-writer-effects-not-generated-answer-quality"]
    scientific_acceptance: Literal["not_assessed"]
    source_metadata: ScientificAblationWriterSourceMetadataV1
    byte_preflight: ScientificAblationWriterBytePreflightV1
    actual_teacher_executions: int = Field(ge=1, le=8192)
    actual_trace_proposal_observation_scalar_bound: int = Field(ge=1, le=64_000_000)
    actual_post_intervention_scalar_bound: int = Field(ge=1)
    ordinary_rows: tuple[ScientificAblationWriterOrdinaryRowV1, ...] = Field(
        min_length=1, max_length=768
    )


class ScientificAblationWorkspaceWriterReportPayloadV1(_PayloadFields):
    kind: Literal["iris-scientific-ablation-workspace-writer-report-v1"]
    controlled_passes: ScientificAblationWorkspaceWriterPassesV1
    effects: tuple[ScientificAblationWorkspaceWriterEffectV1, ...] = Field(max_length=_MAX_EFFECTS)
    statistics: ScientificAblationWorkspaceWriterStatisticsV1


class ScientificAblationMessageWriterReportPayloadV1(_PayloadFields):
    kind: Literal["iris-scientific-ablation-message-writer-report-v1"]
    controlled_passes: ScientificAblationMessageWriterPassesV1
    effects: tuple[ScientificAblationMessageWriterEffectV1, ...] = Field(max_length=_MAX_EFFECTS)
    statistics: ScientificAblationMessageWriterStatisticsV1


def _domain(message: bool) -> str:
    return f"iris/scientific-ablation-{'message' if message else 'workspace'}-writer-report/v1"


class IrisScientificAblationWorkspaceWriterReportV1(_ReportWire):
    sha256: Digest
    payload: ScientificAblationWorkspaceWriterReportPayloadV1

    @model_validator(mode="after")
    def checksum(self) -> Self:
        if self.sha256 != domain_sha256(
            _domain(False), canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Distinct production workspace writer report checksum differs")
        return self


class IrisScientificAblationMessageWriterReportV1(_ReportWire):
    sha256: Digest
    payload: ScientificAblationMessageWriterReportPayloadV1

    @model_validator(mode="after")
    def checksum(self) -> Self:
        if self.sha256 != domain_sha256(
            _domain(True), canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Distinct production message writer report checksum differs")
        return self


_REPORT_TYPES = (
    ScientificAblationWriterSourceMetadataV1,
    ScientificAblationWriterBytePreflightV1,
    ScientificAblationWriterOrdinaryRowV1,
    ScientificAblationWorkspaceWriterTrialsV1,
    ScientificAblationMessageWriterTrialsV1,
    ScientificAblationWorkspaceWriterPassesV1,
    ScientificAblationMessageWriterPassesV1,
    ScientificAblationWorkspaceWriterReportPayloadV1,
    ScientificAblationMessageWriterReportPayloadV1,
    IrisScientificAblationWorkspaceWriterReportV1,
    IrisScientificAblationMessageWriterReportV1,
)
type WriterReport = (
    IrisScientificAblationWorkspaceWriterReportV1 | IrisScientificAblationMessageWriterReportV1
)
type WriterPayload = (
    ScientificAblationWorkspaceWriterReportPayloadV1
    | ScientificAblationMessageWriterReportPayloadV1
)
type PassKey = tuple[int, str, int | None, int | None]


class _EffectFields(TypedDict):
    row_index: int
    write_cycle: int
    source_module: int
    target_module: int
    slot: int
    survives_to_next_read: bool
    overwrote_existing_record: bool
    genuine_distance: float
    negative_control_distance: float
    empty_invocation_sha256: str
    intervention_invocation_sha256: str
    negative_control_invocation_sha256: str


def _wire(value: object) -> object:
    # Only called AFTER the exact declared tree is owned/revalidated.
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if type(value) is tuple:
        return tuple(_wire(child) for child in value)
    return value


def _same(left: object, right: object) -> bool:
    """Compare owned wire bytes; float equality alone erases signed zero."""
    return canonical_json_bytes(_wire(left)) == canonical_json_bytes(_wire(right))


def _key(frame: ScientificAblationWriterPassFrameV1) -> PassKey:
    return frame.row_index, frame.role, frame.write_cycle, frame.writer


def _source_metadata(pf: ScientificAblationCausalReportBytePreflightV1) -> dict[str, object]:
    return {
        name: getattr(pf, name) for name in ScientificAblationWriterSourceMetadataV1.model_fields
    }


def _preflight(
    value: ScientificAblationCausalReportBytePreflightV1,
) -> ScientificAblationCausalReportBytePreflightV1:
    if type(value) is not ScientificAblationCausalReportBytePreflightV1:
        raise ValueError("Separately trusted original preflight required")
    # Exact slotted dataclass: no __dict__/subclass/hidden fields are possible.
    # Own source primitive families before the existing preflight rebuild/validator.
    _raw_model(_source_metadata(value), ScientificAblationWriterSourceMetadataV1)
    for field in fields(value):
        if (
            field.name in ScientificAblationWriterSourceMetadataV1.model_fields
            or field.name == "_shape"
        ):
            continue
        expected = str if field.name == "layout_sha256" else int
        if type(getattr(value, field.name)) is not expected:
            raise ValueError("Exact original preflight primitive families required")
    return validate_scientific_ablation_causal_report_byte_preflight_v1(value)


def _trace_context(
    trace: WorkspaceTraceV1,
    row: int,
    pf: ScientificAblationCausalReportBytePreflightV1,
    *,
    ordinary: bool,
) -> None:
    design = pf.observed_budget.payload.design.payload
    g = design.geometry
    source = pf.evaluation.payload.plan.payload.source.payload
    if (
        trace.options.example_id != design.members[row].example_id
        or trace.options.checkpoint_id != source.best_checkpoint_sha256
        or trace.options.seed != pf.run_seed
        or trace.options.maximum_scalars != design.policy.maximum_trace_scalars
        or trace.options.mode != (design.policy.ordinary_trace_mode if ordinary else "full")
        or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
        != (g.num_modules, g.num_slots, g.state_dim, g.cycle_count)
        or any(len(cycle.write_modules) != g.writers_per_cycle for cycle in trace.cycles)
        or (ordinary and trace.intervention_plan_sha256 != design.ablation_policy_sha256)
    ):
        raise ValueError("Trace differs from original source/run/checkpoint/geometry/policy")


def _shuffle_seed(design: IrisScientificAblationCausalDesignV1, row: int, cycle: int) -> int:
    member = design.payload.members[row]
    return (
        int(
            domain_sha256(
                "iris/scientific-ablation-whole-record-shuffle/v1",
                canonical_json_bytes(
                    dict(
                        seed=design.payload.policy.shuffle_seed,
                        example_id=member.example_id,
                        prompt_sha256=member.cognitive_prompt_sha256,
                        cycle=cycle,
                    )
                ),
            )[:16],
            16,
        )
        % 2**63
    )


def _plan(
    design: IrisScientificAblationCausalDesignV1,
    frame: ScientificAblationWriterPassFrameV1,
    donor: ModuleMessageSnapshotV1 | None,
) -> CognitiveAblationControlPlanV1:
    d = design.payload
    if frame.role == "empty":
        return _empty_plan(design)
    if frame.role == "donor":
        if type(d.policy) is not ScientificAblationMessageCausalPolicyV1 or donor is None:
            raise ValueError("Original paired donor required")
        return CognitiveAblationControlPlanV1(
            ablation=d.ablation,
            control=WorkspaceMessagePlanV1(donor=donor),
            maximum_message_scalars=d.policy.maximum_message_scalars,
        )
    assert frame.write_cycle is not None
    if frame.role == "shuffle":
        operation = WorkspaceInterventionSpecV1(
            kind="shuffle_slots",
            cycles=(frame.write_cycle,),
            seed=_shuffle_seed(design, frame.row_index, frame.write_cycle),
        )
    elif frame.role == "zero":
        operation = WorkspaceInterventionSpecV1(
            kind="zero_slot", cycles=(frame.write_cycle,), slot=frame.slot
        )
    elif frame.role == "block":
        operation = WorkspaceInterventionSpecV1(
            kind="block_module_write", cycles=(frame.write_cycle,), module=frame.writer
        )
    else:
        raise ValueError("Only whole original scheduled roles allowed")
    return CognitiveAblationControlPlanV1(
        ablation=d.ablation, control=WorkspaceInterventionPlanV1(interventions=(operation,))
    )


def _isolation(
    empty: ScientificAblationWriterPassFrameV1,
    trial: ScientificAblationWriterPassFrameV1,
    runtime: ScientificAblationWriterCPURuntimeV1,
) -> None:
    cycle = trial.write_cycle
    assert cycle is not None
    original, changed = empty.output.controlled, trial.output.controlled
    if not _same(original.trace.cycles[: cycle - 1], changed.trace.cycles[: cycle - 1]):
        raise ValueError("Trial changed original pre-intervention trace prefix")
    for index in range(cycle):
        before, after = empty.output.observations[index], trial.output.observations[index]
        if not _same(before.before_write, after.before_write) or (
            index < cycle - 1
            and (
                not _same(before.native_written, after.native_written)
                or not _same(original.states[index], changed.states[index])
            )
        ):
            raise ValueError("Trial changed original ALL7 native prefix/prewrite")
    left, right = original.trace.cycles[cycle - 1], changed.trace.cycles[cycle - 1]
    assert left.full is not None and right.full is not None
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
        if not _same(getattr(left, name), getattr(right, name)):
            raise ValueError("Trial changed original competition/allocation/read phase")
    for name in ("module_states", "workspace_reads", "read_workspace_slots"):
        if not _same(getattr(left.full, name), getattr(right.full, name)):
            raise ValueError("Trial changed original module/goal/read inputs")
    native = empty.output.observations[cycle - 1].native_written
    trial_native = trial.output.observations[cycle - 1].native_written
    if trial.role == "donor":
        assert trial.slot is not None
        for name in type(native).model_fields:
            a, b = getattr(native, name), getattr(trial_native, name)
            if name != "slots":
                if not _same(a, b):
                    raise ValueError("Donor changed original native record metadata")
            elif not _same(
                tuple(v for i, v in enumerate(a.values[0]) if i != trial.slot),
                tuple(v for i, v in enumerate(b.values[0]) if i != trial.slot),
            ):
                raise ValueError("Donor changed another original native slot")
        a = tuple(m.model_dump(mode="json") for m in original.messages if m.cycle <= cycle)
        b = tuple(m.model_dump(mode="json") for m in changed.messages if m.cycle <= cycle)
        if not _same(a, b):
            raise ValueError("Donor changed original recipient proposals before substitution")
    elif not _same(native, trial_native):
        raise ValueError("Record control changed original native writer")
    if trial.role == "shuffle":
        control = changed.plan.control
        assert type(control) is WorkspaceInterventionPlanV1
        seed = control.interventions[0].seed
        assert seed is not None
        digest = domain_sha256(
            "iris/workspace-intervention-rng/v1",
            canonical_json_bytes(dict(kind="shuffle_slots", seed=seed, cycle=cycle)),
        )
        order = writer_cpu_permutation_v1(
            int(digest[:16], 16) & (2**63 - 1), native.slots.shape[1], runtime=runtime
        )
        post = changed.states[cycle - 1]
        for name in type(native).model_fields:
            expected = tuple(getattr(native, name).values[0][index] for index in order)
            if not _same(expected, getattr(post, name).values[0]):
                raise ValueError("SHUFFLE is not the exact original input-only CPU permutation")
    # Local observed-output validation already enforces whole ZERO/BLOCK/DONOR post phases.


def _semantics(
    payload: WriterPayload,
    pf: ScientificAblationCausalReportBytePreflightV1,
    runtime: ScientificAblationWriterCPURuntimeV1,
) -> None:
    design = pf.observed_budget.payload.design
    d, g, budget = design.payload, design.payload.geometry, pf.observed_budget.payload
    message = type(d.policy) is ScientificAblationMessageCausalPolicyV1
    if type(payload) is not (
        ScientificAblationMessageWriterReportPayloadV1
        if message
        else ScientificAblationWorkspaceWriterReportPayloadV1
    ):
        raise ValueError("Report protocol differs from original source policy")
    expected_source = ScientificAblationWriterSourceMetadataV1.model_validate(_source_metadata(pf))
    if not _same(payload.source_metadata, expected_source) or (
        payload.byte_preflight.layout_sha256,
        payload.byte_preflight.report_byte_preflight_bound,
        payload.byte_preflight.maximum_report_bytes,
    ) != (pf.layout_sha256, pf.report_byte_preflight_bound, pf.maximum_report_bytes):
        raise ValueError("Report differs from separate original source/preflight expectation")
    if (
        payload.actual_teacher_executions != budget.planned_total_teacher_executions
        or payload.actual_trace_proposal_observation_scalar_bound
        != budget.total_trace_proposal_observation_scalar_bound
        or payload.actual_post_intervention_scalar_bound != pf.post_intervention_scalar_bound
        or len(payload.effects) != d.budget.maximum_effect_rows
        or tuple(row.row_index for row in payload.ordinary_rows) != d.selected_row_indices
    ):
        raise ValueError("Whole original report counters/ordinary/effect inventory differs")
    groups = {"empty": payload.controlled_passes.empty}
    for name in type(payload.controlled_passes.trials).model_fields:
        groups[name] = getattr(payload.controlled_passes.trials, name)
    indexed: dict[PassKey, ScientificAblationWriterPassFrameV1] = {}
    for role, frames in groups.items():
        for frame in frames:
            key = _key(frame)
            if (
                frame.role != role
                or key in indexed
                or frame.row_index not in d.selected_row_indices
            ):
                raise ValueError("Duplicate/foreign scheduled role or source row")
            indexed[key] = frame
            control = frame.output.controlled
            score = pf.evaluation.payload.rows[frame.row_index]
            if (
                control.prompt_sha256 != d.members[frame.row_index].cognitive_prompt_sha256
                or control.target_sha256 != score.teacher_target_sha256
                or control.supervised_token_count != score.target_token_count
                or control.teacher_graph_sha256 != pf.controlled_teacher_graph_sha256
                or control.sidecar_state_sha256 != pf.sidecar_sha256
                or control.ablation_policy_sha256 != d.ablation_policy_sha256
                or frame.output.maximum_observation_scalars != budget.maximum_observation_scalars
                or frame.output.actual_observation_scalars
                != budget.observation_scalar_bound_per_pass
            ):
                raise ValueError("Controlled pass differs from original input/EOS/graph/quota")
            _trace_context(control.trace, frame.row_index, pf, ordinary=False)
    if len(indexed) != budget.planned_controlled_teacher_executions:
        raise ValueError("Every original controlled pass required, no repeats/subsets")
    donor_pairs = {
        row: (other, pair.sha256)
        for pair in d.donor_pairs
        for row, other in (
            (pair.first_row_index, pair.second_row_index),
            (pair.second_row_index, pair.first_row_index),
        )
    }
    expected_keys: dict[str, list[PassKey]] = {role: [] for role in groups}
    expected_effects: list[WriterEffect] = []
    bindings = _score_bindings(pf.evaluation_plan, pf.evaluation, pf.evaluation_complete)
    for ordinary in payload.ordinary_rows:
        row = ordinary.row_index
        empty_key: PassKey = (row, "empty", None, None)
        if empty_key not in indexed:
            raise ValueError("Original EMPTY missing")
        empty = indexed[empty_key]
        expected_keys["empty"].append(empty_key)
        score = pf.evaluation.payload.rows[row]
        if (
            (ordinary.evaluation_row_sha256, ordinary.evaluation_ack_sha256) != bindings[row]
            or ordinary.capture_row_sha256 != score.capture_row_sha256
            or ordinary.ordinary_teacher_invocation_sha256 != score.teacher_invocation_sha256
            or ordinary.target_sha256 != score.teacher_target_sha256
            or ordinary.target_token_count != score.target_token_count
            or not _same(ordinary.target_nll_sum, score.target_nll_sum)
            or not _same(ordinary.ordinary_mean_target_nll, empty.output.controlled.loss)
            or not isclose(
                ordinary.target_nll_sum,
                ordinary.ordinary_mean_target_nll * ordinary.target_token_count,
                rel_tol=2e-7,
                abs_tol=1e-7,
            )
            or not _same(empty.output.controlled.plan, _empty_plan(design))
        ):
            raise ValueError("Original ordinary/EMPTY capture/COMPLETE/NLL/plan differs")
        _trace_context(ordinary.ordinary_trace, row, pf, ordinary=True)
        a = tuple(
            c.model_dump(mode="json", exclude={"full"}) for c in ordinary.ordinary_trace.cycles
        )
        b = tuple(
            c.model_dump(mode="json", exclude={"full"})
            for c in empty.output.controlled.trace.cycles
        )
        if not _same(a, b) or (
            d.policy.ordinary_trace_mode == "full"
            and not _same(ordinary.ordinary_trace.cycles, empty.output.controlled.trace.cycles)
        ):
            raise ValueError("EMPTY is not original ordinary trace parity")
        for cycle in range(1, g.cycle_count):
            written, next_read = empty.output.controlled.trace.cycles[cycle - 1 : cycle + 1]
            if not message:
                key = (row, "shuffle", cycle, None)
                if key not in indexed:
                    raise ValueError("Original SHUFFLE missing")
                expected_keys["shuffle"].append(key)
            for writer, slot in zip(written.write_modules, written.selected_slots, strict=True):
                genuine_key: PassKey = (row, "block" if message else "zero", cycle, writer)
                negative_key: PassKey = (
                    (row, "donor", cycle, writer) if message else (row, "shuffle", cycle, None)
                )
                if genuine_key not in indexed or negative_key not in indexed:
                    raise ValueError("Original allocated writer trial missing")
                genuine, negative = indexed[genuine_key], indexed[negative_key]
                expected_keys[genuine.role].append(genuine_key)
                if message:
                    expected_keys["donor"].append(negative_key)
                donor: ModuleMessageSnapshotV1 | None = None
                for trial in (genuine, negative):
                    if trial.role != "shuffle" and trial.slot != slot:
                        raise ValueError("Trial targets a different original recipient allocation")
                    if trial.role == "donor":
                        donor_row, pair = donor_pairs[row]
                        donor_key: PassKey = (donor_row, "empty", None, None)
                        if donor_key not in indexed or (
                            trial.donor_row_index,
                            trial.donor_pair_sha256,
                        ) != (donor_row, pair):
                            raise ValueError("Original paired donor EMPTY/context missing")
                        donor = indexed[donor_key].output.controlled.messages[
                            (cycle - 1) * g.num_modules + writer
                        ]
                    if not _same(trial.output.controlled.plan, _plan(design, trial, donor)):
                        raise ValueError("Plan differs from original input-only schedule/donor")
                    _isolation(empty, trial, runtime)
                assert next_read.full is not None
                next_g = genuine.output.controlled.trace.cycles[cycle].full
                next_n = negative.output.controlled.trace.cycles[cycle].full
                assert next_g is not None and next_n is not None
                before = empty.output.observations[cycle - 1].before_write
                overwrote = before.occupancy.values[0][slot]
                if (
                    overwrote is not (slot in written.overwritten_slots)
                    or not written.slot_occupancy[slot]
                    or written.slot_sources[slot] != writer
                ):
                    raise ValueError("Original actual writer/overwrite record differs")
                survives = (
                    next_read.read_slot_occupancy[slot]
                    and next_read.read_slot_sources[slot] == writer
                )
                for receiver in range(g.num_modules):
                    if receiver == writer:
                        continue
                    baseline = next_read.full.module_states[receiver]
                    background = next_g.module_states[receiver] if message else baseline
                    genuine_state = baseline if message else next_g.module_states[receiver]
                    common: _EffectFields = dict(
                        row_index=row,
                        write_cycle=cycle,
                        source_module=writer,
                        target_module=receiver,
                        slot=slot,
                        survives_to_next_read=survives,
                        overwrote_existing_record=overwrote,
                        genuine_distance=relative_state_distance_v1(background, genuine_state),
                        negative_control_distance=relative_state_distance_v1(
                            background, next_n.module_states[receiver]
                        ),
                        empty_invocation_sha256=empty.output.invocation_sha256,
                        intervention_invocation_sha256=genuine.output.invocation_sha256,
                        negative_control_invocation_sha256=negative.output.invocation_sha256,
                    )
                    if message:
                        assert donor is not None
                        donor_row, pair = donor_pairs[row]
                        recipient = empty.output.controlled.messages[
                            (cycle - 1) * g.num_modules + writer
                        ]
                        expected_effects.append(
                            ScientificAblationMessageWriterEffectV1(
                                **common,
                                recipient_message_sha256=message_snapshot_sha256_v1(recipient),
                                donor_message_sha256=message_snapshot_sha256_v1(donor),
                                donor_row_index=donor_row,
                                donor_pair_sha256=pair,
                            )
                        )
                    else:
                        expected_effects.append(ScientificAblationWorkspaceWriterEffectV1(**common))
    if any(
        tuple(_key(frame) for frame in groups[role]) != tuple(keys)
        for role, keys in expected_keys.items()
    ):
        raise ValueError(
            "Whole canonical per-role schedule differs, no missing/extra/reordered passes"
        )
    if len(expected_effects) != len(payload.effects) or any(
        not _same(a, b) for a, b in zip(payload.effects, expected_effects, strict=True)
    ):
        raise ValueError("Effects differ from actual original next-cycle endpoints/flags/anchors")
    statistics = build_scientific_ablation_writer_statistics_v1(design, tuple(expected_effects))
    if not _same(payload.statistics, statistics):
        raise ValueError("Complete original grouped sign/Holm statistics differ")


def scientific_ablation_writer_report_bytes_v1(
    value: WriterReport,
    *,
    preflight: ScientificAblationCausalReportBytePreflightV1,
    runtime: ScientificAblationWriterCPURuntimeV1,
) -> bytes:
    """Mandatory semantic acceptance THEN complete layout/canonical byte acceptance."""
    pf = _preflight(preflight)
    _runtime(runtime)
    if type(value) is IrisScientificAblationWorkspaceWriterReportV1:
        owned: WriterReport = IrisScientificAblationWorkspaceWriterReportV1.model_validate(
            _report_raw(value, IrisScientificAblationWorkspaceWriterReportV1)
        )
    elif type(value) is IrisScientificAblationMessageWriterReportV1:
        owned = IrisScientificAblationMessageWriterReportV1.model_validate(
            _report_raw(value, IrisScientificAblationMessageWriterReportV1)
        )
    else:
        raise ValueError("Exact distinct complete writer report root required")
    _semantics(owned.payload, pf, runtime)
    return causal_report_layout_bytes_v1(owned.model_dump(mode="json"), pf)


def build_scientific_ablation_writer_report_v1(
    payload: WriterPayload,
    *,
    preflight: ScientificAblationCausalReportBytePreflightV1,
    runtime: ScientificAblationWriterCPURuntimeV1,
) -> WriterReport:
    """Metadata builder, NOT native measurement evidence; source service must own inputs."""
    if type(payload) is ScientificAblationWorkspaceWriterReportPayloadV1:
        owned: WriterPayload = ScientificAblationWorkspaceWriterReportPayloadV1.model_validate(
            _report_raw(payload, ScientificAblationWorkspaceWriterReportPayloadV1)
        )
        result: WriterReport = IrisScientificAblationWorkspaceWriterReportV1.model_validate(
            dict(
                sha256=domain_sha256(
                    _domain(False), canonical_json_bytes(owned.model_dump(mode="json"))
                ),
                payload=owned,
            )
        )
    elif type(payload) is ScientificAblationMessageWriterReportPayloadV1:
        owned = ScientificAblationMessageWriterReportPayloadV1.model_validate(
            _report_raw(payload, ScientificAblationMessageWriterReportPayloadV1)
        )
        result = IrisScientificAblationMessageWriterReportV1.model_validate(
            dict(
                sha256=domain_sha256(
                    _domain(True), canonical_json_bytes(owned.model_dump(mode="json"))
                ),
                payload=owned,
            )
        )
    else:
        raise ValueError("Exact distinct whole writer payload required")
    scientific_ablation_writer_report_bytes_v1(result, preflight=preflight, runtime=runtime)
    return result


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON field forbidden")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("Nonfinite JSON constant forbidden")


def read_scientific_ablation_writer_report_v1(
    content: bytes,
    *,
    preflight: ScientificAblationCausalReportBytePreflightV1,
    runtime: ScientificAblationWriterCPURuntimeV1,
) -> WriterReport:
    """Bounded exact canonical reader, not a model-import-free independent observer."""
    pf = _preflight(preflight)
    _runtime(runtime)
    if type(content) is not bytes or not 1 <= len(content) <= min(
        pf.maximum_report_bytes, pf.report_byte_preflight_bound
    ):
        raise ValueError("Exact bounded report bytes required")
    # Bound nesting BEFORE JSON allocations/model validators, accounting for escaped strings.
    depth = 0
    quoted = escaped = False
    for byte in content:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 40:
                raise ValueError("Complete report JSON nesting bound exceeded")
        elif byte in (93, 125):
            depth -= 1
    try:
        raw = json.loads(
            content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Valid exact UTF8 report JSON required") from error
    message = (
        type(pf.observed_budget.payload.design.payload.policy)
        is ScientificAblationMessageCausalPolicyV1
    )
    if message:
        result: WriterReport = IrisScientificAblationMessageWriterReportV1.model_validate(raw)
    else:
        result = IrisScientificAblationWorkspaceWriterReportV1.model_validate(raw)
    if scientific_ablation_writer_report_bytes_v1(result, preflight=pf, runtime=runtime) != content:
        raise ValueError("Exact canonical report framing required")
    return result
