"""Full private native writer schedule within ONE authenticated source transaction.

This stage measures both protocols, including every original writer/receiver and
all native ALL7 phases. It is not a public report or generated-answer/scientific
acceptance. Publication must still bind its output to successful transaction exit
and a distinct complete semantic report; callers cannot supply native receipts.
"""

from dataclasses import dataclass, fields
from re import fullmatch
from typing import Literal, TypedDict

import torch

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.application.scientific_ablation_causal_controls import (
    _ControlMeasurement,
    _observed_controls,
    _Source,
)
from iris.application.scientific_ablation_causal_receipts import (
    _ObservedPassSeal,
    _seal_observed,
)
from iris.artifacts.scientific_ablation_causal_controls_v1 import (
    ScientificAblationCausalControlRowV1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import _own
from iris.artifacts.scientific_ablation_writer_statistics_v1 import (
    ScientificAblationMessageWriterEffectV1,
    ScientificAblationWorkspaceWriterEffectV1,
    WriterEffect,
    WriterStatistics,
    build_scientific_ablation_writer_statistics_v1,
    scientific_ablation_writer_statistics_bytes_v1,
)
from iris.interventions.messages import (
    ModuleMessageSnapshotV1,
    WorkspaceMessagePlanV1,
    message_snapshot_sha256_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.training.ablation_observation import IrisAblationObservedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.outputs import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import restore_runtime_state
from iris.workspace.causal import relative_state_distance_v1
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import _scalar_bound

type _Role = Literal["empty", "shuffle", "zero", "block", "donor"]
type _PassKey = tuple[int, _Role, int | None, int | None]


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


@dataclass(frozen=True, slots=True)
class _WriterPass:
    row_index: int
    role: _Role
    write_cycle: int | None
    writer: int | None
    slot: int | None
    donor_row_index: int | None
    donor_pair_sha256: str | None
    seal: _ObservedPassSeal

    def __post_init__(self) -> None:
        if (
            type(self) is not _WriterPass
            or type(self.row_index) is not int
            or not 0 <= self.row_index <= 4999
            or type(self.role) is not str
            or self.role not in ("empty", "shuffle", "zero", "block", "donor")
            or type(self.seal) is not _ObservedPassSeal
        ):
            raise ValueError("Exact original private pass-frame metadata required")
        empty, slot_trial, donor = (
            self.role == "empty",
            self.role not in ("empty", "shuffle"),
            self.role == "donor",
        )
        for value, applicable, maximum in (
            (self.write_cycle, not empty, 63),
            (self.writer, slot_trial, 63),
            (self.slot, slot_trial, 63),
            (self.donor_row_index, donor, 4999),
        ):
            if (not applicable and value is not None) or (
                applicable and (type(value) is not int or not 0 <= value <= maximum)
            ):
                raise ValueError("Exact applicable private pass indices required, no bool-as-int")
        if not empty and self.write_cycle == 0:
            raise ValueError("Native write cycle is one-based")
        if donor:
            if (
                type(self.donor_pair_sha256) is not str
                or fullmatch(r"[0-9a-f]{64}", self.donor_pair_sha256) is None
            ):
                raise ValueError("Original exact donor pair digest required")
        elif self.donor_pair_sha256 is not None:
            raise ValueError("Only content transfers carry donor pair metadata")

    @property
    def output(self) -> IrisAblationObservedTeacherOutputV1:
        return self.seal.output


@dataclass(frozen=True, slots=True)
class _WriterMeasurements:
    design_sha256: str
    layout_sha256: str
    rows: tuple[_ControlMeasurement, ...]
    passes: tuple[_WriterPass, ...]
    effects: tuple[WriterEffect, ...]
    statistics: WriterStatistics
    actual_teacher_executions: int
    actual_trace_proposal_observation_scalar_bound: int
    actual_post_intervention_scalar_bound: int


def _same_state(left: WorkspaceState, right: WorkspaceState) -> bool:
    return all(torch.equal(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))


def _post_trace(output: IrisAblationObservedTeacherOutputV1) -> None:
    """Read actual post-intervention tensors, never fabricate missing confidence."""
    c = output.controlled
    for state, row in zip(c.workspace_trace, c.diagnostic_trace.cycles, strict=True):
        if row.full is None or tuple(tuple(v) for v in state.slots[0].tolist()) != (
            row.full.workspace_slots
        ):
            raise ValueError("Native post-intervention slot vectors differ from full trace")
        for field, name in (
            ("age", "slot_ages"),
            ("occupancy", "slot_occupancy"),
            ("source_module", "slot_sources"),
            ("ttl", "slot_ttl"),
            ("write_count", "slot_write_counts"),
        ):
            if tuple(getattr(state, field)[0].tolist()) != getattr(row, name):
                raise ValueError("Actual post-intervention metadata differs from full trace")


def _before(
    empty: IrisAblationObservedTeacherOutputV1,
    changed: IrisAblationObservedTeacherOutputV1,
    cycle: int,
) -> None:
    a, b = empty.controlled, changed.controlled
    if a.diagnostic_trace.cycles[: cycle - 1] != b.diagnostic_trace.cycles[: cycle - 1]:
        raise ValueError("Writer trial changed a cycle before its scheduled intervention")
    for i in range(cycle):
        original, trial = empty.observations[i], changed.observations[i]
        if not _same_state(original.before_write, trial.before_write) or (
            i < cycle - 1
            and (
                not _same_state(original.native_written, trial.native_written)
                or not _same_state(a.workspace_trace[i], b.workspace_trace[i])
            )
        ):
            raise ValueError("Trial changed actual ALL7 native prewrite/prefix states")
    left, right = a.diagnostic_trace.cycles[cycle - 1], b.diagnostic_trace.cycles[cycle - 1]
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
        if getattr(left, name) != getattr(right, name):
            raise ValueError("Trial changed original recipient competition/allocation/read phase")
    if any(
        getattr(left.full, name) != getattr(right.full, name)
        for name in ("module_states", "workspace_reads", "read_workspace_slots")
    ):
        raise ValueError("Trial changed original recipient goal/read/module inputs")


def _isolation(empty: _WriterPass, changed: _WriterPass) -> None:
    cycle = changed.write_cycle
    assert cycle is not None
    _before(empty.output, changed.output, cycle)
    native = empty.output.observations[cycle - 1].native_written
    before = empty.output.observations[cycle - 1].before_write
    trial_native = changed.output.observations[cycle - 1].native_written
    post = changed.output.controlled.workspace_trace[cycle - 1]
    if changed.role == "donor":
        assert changed.slot is not None
        for f in fields(WorkspaceState):
            left, right = getattr(native, f.name), getattr(trial_native, f.name)
            others = [i for i in range(native.slots.shape[1]) if i != changed.slot]
            if not torch.equal(left[:, others], right[:, others]) or (
                f.name != "slots" and not torch.equal(left, right)
            ):
                raise ValueError("Donor content changed recipient metadata/another native record")
        if not _same_state(trial_native, post):
            raise ValueError("Content trial applied an unplanned post-writer record operation")
        if tuple(m for m in changed.output.controlled.messages if m.cycle <= cycle) != tuple(
            m for m in empty.output.controlled.messages if m.cycle <= cycle
        ):
            raise ValueError("Donor changed recipient proposals BEFORE content substitution")
        # The writer blends/normalizes donor content with recipient goal/old slot.
        # Never demand that the resulting slot equal the donor proposal vector.
        return
    if not _same_state(native, trial_native):
        raise ValueError("Record intervention changed the actual original native writer")
    if changed.role == "shuffle":
        operation = changed.output.controlled.plan.control
        assert type(operation) is WorkspaceInterventionPlanV1
        spec = operation.interventions[0]
        digest = domain_sha256(
            "iris/workspace-intervention-rng/v1",
            canonical_json_bytes(dict(kind="shuffle_slots", seed=spec.seed, cycle=cycle)),
        )
        rng = torch.Generator(device="cpu").manual_seed(int(digest[:16], 16) & (2**63 - 1))
        indices = torch.randperm(native.slots.shape[1], generator=rng, device="cpu").to(
            device=native.slots.device
        )
        if any(
            not torch.equal(getattr(post, f.name), getattr(native, f.name)[:, indices])
            for f in fields(WorkspaceState)
        ):
            raise ValueError("SHUFFLE must permute every field of the SAME whole slot record")
        return  # A deterministic identity permutation is valid, with no reroll.
    assert changed.slot is not None
    for f in fields(WorkspaceState):
        expected = getattr(native, f.name).clone()
        if changed.role == "zero":
            expected[:, changed.slot] = -1 if f.name == "source_module" else 0
        elif changed.role == "block":
            expected[:, changed.slot] = getattr(before, f.name)[:, changed.slot]
        else:
            raise ValueError("Only exact ZERO/BLOCK/SHUFFLE/DONOR scheduled trials allowed")
        if not torch.equal(expected, getattr(post, f.name)):
            raise ValueError("ZERO/BLOCK failed full selected-record/other-record isolation")


def _shuffle_seed(source: _Source, row: int, cycle: int) -> int:
    p = source.design.payload
    member = p.members[row]
    return (
        int(
            domain_sha256(
                "iris/scientific-ablation-whole-record-shuffle/v1",
                canonical_json_bytes(
                    dict(
                        seed=p.policy.shuffle_seed,
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
    source: _Source,
    role: _Role,
    row: int,
    cycle: int,
    writer: int | None,
    slot: int | None,
    donor: ModuleMessageSnapshotV1 | None,
) -> CognitiveAblationControlPlanV1:
    p = source.design.payload
    if role == "donor":
        if type(p.policy) is not ScientificAblationMessageCausalPolicyV1 or donor is None:
            raise ValueError("Only original paired-message content donor allowed")
        return CognitiveAblationControlPlanV1(
            ablation=p.ablation,
            control=WorkspaceMessagePlanV1(donor=donor),
            maximum_message_scalars=p.policy.maximum_message_scalars,
        )
    if role == "shuffle":
        spec = WorkspaceInterventionSpecV1(
            kind="shuffle_slots", cycles=(cycle,), seed=_shuffle_seed(source, row, cycle)
        )
    elif role == "zero":
        spec = WorkspaceInterventionSpecV1(kind="zero_slot", cycles=(cycle,), slot=slot)
    elif role == "block":
        spec = WorkspaceInterventionSpecV1(
            kind="block_module_write", cycles=(cycle,), module=writer
        )
    else:
        raise ValueError("EMPTY is retained once, never executed again as a trial")
    return CognitiveAblationControlPlanV1(
        ablation=p.ablation,
        control=WorkspaceInterventionPlanV1(interventions=(spec,)),
    )


def _prepared(source: _Source, measurement: _ControlMeasurement) -> IrisTeacherForcingInput:
    prepared, seal = measurement.prepared, measurement.retention_seal
    if (
        type(prepared) is not IrisTeacherForcingInput
        or type(seal) is not _ObservedPassSeal
        or seal.prepared is not prepared
        or seal.output is not measurement.observation
    ):
        raise ValueError("Original immediately sealed EMPTY prepared input required")
    seal.audit()
    row = _own(measurement.row, ScientificAblationCausalControlRowV1)
    if (
        prepared.prompt != source.seal.rows[row.row_index].prompt
        or teacher_target_sha256_v1(prepared.target_input_ids, prepared.target_attention_mask)
        != row.target_sha256
        or int(prepared.target_attention_mask.sum()) != row.target_token_count
    ):
        raise ValueError("Prepared trial input differs from original source/EOS score")
    return prepared


def _trial(
    source: _Source,
    measurement: _ControlMeasurement,
    role: _Role,
    cycle: int,
    writer: int | None = None,
    slot: int | None = None,
    donor_row: int | None = None,
    pair: str | None = None,
    donor: ModuleMessageSnapshotV1 | None = None,
) -> _WriterPass:
    source.identity()
    if source.complete_preflight is None:
        raise ValueError("Full original BEFORE-BUSY reservation required")
    prepared = _prepared(source, measurement)
    empty = measurement.observation
    assert empty is not None
    plan = _plan(source, role, measurement.row.row_index, cycle, writer, slot, donor)
    restore_runtime_state(source.run._runtime)
    output = source.model.teacher.evaluation_forward_ablation_observed(
        prepared,
        plan=plan,
        diagnostics=empty.controlled.diagnostic_trace.options,
        maximum_observation_scalars=source.complete_preflight.observed_budget.payload.maximum_observation_scalars,
    )
    sealed = _seal_observed(output, prepared)  # BEFORE a later pass or row executes.
    original, actual = empty.controlled, output.controlled
    if (
        actual.plan != plan
        or any(
            getattr(actual, name) != getattr(original, name)
            for name in (
                "ablation_policy_sha256",
                "prompt_sha256",
                "target_sha256",
                "teacher_graph_sha256",
                "sidecar_state_sha256",
                "supervised_token_count",
            )
        )
        or actual.diagnostic_trace.options != original.diagnostic_trace.options
        or (
            actual.diagnostic_trace.num_modules,
            actual.diagnostic_trace.num_slots,
            actual.diagnostic_trace.state_dim,
            actual.diagnostic_trace.cycle_count,
        )
        != (
            original.diagnostic_trace.num_modules,
            original.diagnostic_trace.num_slots,
            original.diagnostic_trace.state_dim,
            original.diagnostic_trace.cycle_count,
        )
        or output.maximum_observation_scalars != empty.maximum_observation_scalars
        or output.actual_observation_scalars != empty.actual_observation_scalars
    ):
        raise ValueError("Actual native trial differs from original selected-best source/plan")
    result = _WriterPass(
        measurement.row.row_index,
        role,
        cycle,
        writer,
        slot,
        donor_row,
        pair,
        sealed,
    )
    _post_trace(output)
    return result


def _derive_effects(
    source: _Source, rows: tuple[_ControlMeasurement, ...], passes: tuple[_WriterPass, ...]
) -> tuple[WriterEffect, ...]:
    p, g = source.design.payload, source.design.payload.geometry
    message = type(p.policy) is ScientificAblationMessageCausalPolicyV1
    if tuple(r.row.row_index for r in rows) != p.selected_row_indices:
        raise ValueError("Every selected original ordinary/EMPTY row required in order")
    indexed: dict[_PassKey, _WriterPass] = {}
    for entry in passes:
        if type(entry) is not _WriterPass or type(entry.seal) is not _ObservedPassSeal:
            raise TypeError("Exact trusted native scheduled pass required")
        entry.__post_init__()
        entry.seal.audit()
        _post_trace(entry.output)
        key = (entry.row_index, entry.role, entry.write_cycle, entry.writer)
        if key in indexed:
            raise ValueError("Duplicate scheduled native pass")
        indexed[key] = entry
    donors = {
        row: (other, pair.sha256)
        for pair in p.donor_pairs
        for row, other in (
            (pair.first_row_index, pair.second_row_index),
            (pair.second_row_index, pair.first_row_index),
        )
    }
    expected_keys: list[_PassKey] = []
    effects: list[WriterEffect] = []
    for measurement in rows:
        prepared = _prepared(source, measurement)
        index = measurement.row.row_index
        empty = indexed[index, "empty", None, None]
        expected_keys.append((index, "empty", None, None))
        if (
            empty.seal is not measurement.retention_seal
            or empty.seal.prepared is not prepared
            or empty.slot is not None
            or empty.donor_row_index is not None
            or empty.donor_pair_sha256 is not None
        ):
            raise ValueError("ONE original EMPTY stage per row, no replacement/repeated baseline")
        original = empty.output.controlled
        if any(
            not _same_state(o.native_written, state)
            for o, state in zip(empty.output.observations, original.workspace_trace, strict=True)
        ):
            raise ValueError("EMPTY must retain the actual native writer unchanged in ALL7 fields")
        for cycle in range(1, g.cycle_count):
            written, next_read = original.diagnostic_trace.cycles[cycle - 1 : cycle + 1]
            if len(written.write_modules) != g.writers_per_cycle:
                raise ValueError("Every actual allocated original nonfinal writer required")
            if not message:
                negative = indexed[index, "shuffle", cycle, None]
                expected_keys.append((index, "shuffle", cycle, None))
                _isolation(empty, negative)
                if (
                    negative.slot is not None
                    or negative.donor_row_index is not None
                    or negative.donor_pair_sha256 is not None
                ):
                    raise ValueError("Whole-record SHUFFLE cannot target a writer/slot/donor")
            for writer, slot in zip(written.write_modules, written.selected_slots, strict=True):
                before = empty.output.observations[cycle - 1].before_write
                if not written.slot_occupancy[slot] or written.slot_sources[slot] != writer:
                    raise ValueError("Actual original selected writer record is missing")
                overwrote = bool(before.occupancy[0, slot])
                if overwrote is not (slot in written.overwritten_slots):
                    raise ValueError("Overwrite flag must come from actual ALL7 prewrite occupancy")
                genuine = indexed[index, "block" if message else "zero", cycle, writer]
                expected_keys.append((index, genuine.role, cycle, writer))
                negative = indexed[index, "donor", cycle, writer] if message else negative
                if message:
                    expected_keys.append((index, "donor", cycle, writer))
                for trial in (genuine, negative):
                    if trial.seal.prepared is not prepared:
                        raise ValueError(
                            "Every trial must use the SAME original prepared recipient"
                        )
                    _isolation(empty, trial)
                    donor = None
                    if trial.role == "donor":
                        donor_row, pair = donors[index]
                        donor_empty = indexed[donor_row, "empty", None, None]
                        donor = donor_empty.output.controlled.messages[
                            (cycle - 1) * g.num_modules + writer
                        ]
                        if (trial.donor_row_index, trial.donor_pair_sha256) != (donor_row, pair):
                            raise ValueError(
                                "Donor must come from original input-only disjoint pair"
                            )
                    elif trial.donor_row_index is not None or trial.donor_pair_sha256 is not None:
                        raise ValueError("Record trials cannot carry donor metadata")
                    if trial.output.controlled.plan != _plan(
                        source, trial.role, index, cycle, trial.writer, trial.slot, donor
                    ) or (trial.role != "shuffle" and trial.slot != slot):
                        raise ValueError(
                            "Native trial plan must bind actual original writer/slot/donor"
                        )
                assert next_read.full is not None
                changed_next = genuine.output.controlled.diagnostic_trace.cycles[cycle].full
                negative_next = negative.output.controlled.diagnostic_trace.cycles[cycle].full
                assert changed_next is not None and negative_next is not None
                survives = (
                    next_read.read_slot_occupancy[slot]
                    and next_read.read_slot_sources[slot] == writer
                )
                for receiver in range(g.num_modules):
                    if receiver == writer:
                        continue
                    baseline = next_read.full.module_states[receiver]
                    background = changed_next.module_states[receiver] if message else baseline
                    genuine_state = baseline if message else changed_next.module_states[receiver]
                    common: _EffectFields = dict(
                        row_index=index,
                        write_cycle=cycle,
                        source_module=writer,
                        target_module=receiver,
                        slot=slot,
                        survives_to_next_read=survives,
                        overwrote_existing_record=overwrote,
                        genuine_distance=relative_state_distance_v1(background, genuine_state),
                        negative_control_distance=relative_state_distance_v1(
                            background, negative_next.module_states[receiver]
                        ),
                        empty_invocation_sha256=empty.output.invocation_sha256,
                        intervention_invocation_sha256=genuine.output.invocation_sha256,
                        negative_control_invocation_sha256=negative.output.invocation_sha256,
                    )
                    if message:
                        donor_row, pair = donors[index]
                        recipient = original.messages[(cycle - 1) * g.num_modules + writer]
                        donor = indexed[donor_row, "empty", None, None].output.controlled.messages[
                            (cycle - 1) * g.num_modules + writer
                        ]
                        effects.append(
                            ScientificAblationMessageWriterEffectV1(
                                **common,
                                donor_row_index=donor_row,
                                donor_pair_sha256=pair,
                                recipient_message_sha256=message_snapshot_sha256_v1(recipient),
                                donor_message_sha256=message_snapshot_sha256_v1(donor),
                            )
                        )
                    else:
                        effects.append(ScientificAblationWorkspaceWriterEffectV1(**common))
    ordered = tuple(k for k in expected_keys if k[1] == "empty") + tuple(
        k for k in expected_keys if k[1] != "empty"
    )
    if tuple(indexed) != ordered or len(passes) != p.budget.controlled_executions:
        raise ValueError("Complete scheduled inventory required, no extra/missing native passes")
    return tuple(effects)


def _audit_writer_measurements(source: _Source, result: _WriterMeasurements) -> None:
    source.audit(measuring=True)
    pf = source.complete_preflight
    if pf is None or type(result) is not _WriterMeasurements:
        raise ValueError("Original complete source reservation/native measurements required")
    if (
        type(result.rows) is not tuple
        or any(type(row) is not _ControlMeasurement for row in result.rows)
        or type(result.passes) is not tuple
        or type(result.effects) is not tuple
        or any(
            type(getattr(result, name)) is not int
            for name in (
                "actual_teacher_executions",
                "actual_trace_proposal_observation_scalar_bound",
                "actual_post_intervention_scalar_bound",
            )
        )
        or any(
            type(getattr(result, name)) is not str for name in ("design_sha256", "layout_sha256")
        )
    ):
        raise ValueError("Exact private complete-result metadata families required")
    for effect in result.effects:
        if type(effect) is ScientificAblationWorkspaceWriterEffectV1:
            _own(effect, ScientificAblationWorkspaceWriterEffectV1)
        elif type(effect) is ScientificAblationMessageWriterEffectV1:
            _own(effect, ScientificAblationMessageWriterEffectV1)
        else:
            raise ValueError("Exact individually owned native effects required before comparison")
    effects = _derive_effects(source, result.rows, result.passes)
    statistics = build_scientific_ablation_writer_statistics_v1(source.design, effects)
    traces = [r.row.ordinary_trace for r in result.rows] + [
        p.output.controlled.diagnostic_trace for p in result.passes
    ]
    trace_volume = sum(
        _scalar_bound(
            t.num_modules, t.num_slots, t.state_dim, t.cycle_count, t.options.mode == "full"
        )
        for t in traces
    )
    proposal_volume = sum(
        len(m.values) for p in result.passes for m in p.output.controlled.messages
    )
    observation_volume = sum(p.output.actual_observation_scalars for p in result.passes)
    if (
        result.design_sha256 != source.design.sha256
        or result.layout_sha256 != pf.layout_sha256
        or result.effects != effects
        or scientific_ablation_writer_statistics_bytes_v1(result.statistics)
        != scientific_ablation_writer_statistics_bytes_v1(statistics)
        or result.actual_teacher_executions != len(result.rows) + len(result.passes)
        or result.actual_teacher_executions
        != pf.observed_budget.payload.planned_total_teacher_executions
        or observation_volume != pf.observed_budget.payload.total_observation_scalar_bound
        or proposal_volume != source.design.payload.budget.total_message_scalars
        or result.actual_trace_proposal_observation_scalar_bound
        != pf.observed_budget.payload.total_trace_proposal_observation_scalar_bound
        or result.actual_trace_proposal_observation_scalar_bound
        != trace_volume + proposal_volume + observation_volume
        or result.actual_post_intervention_scalar_bound
        != sum(
            getattr(s, f.name).numel()
            for p in result.passes
            for s in p.output.controlled.workspace_trace
            for f in fields(WorkspaceState)
        )
        or result.actual_post_intervention_scalar_bound != pf.post_intervention_scalar_bound
    ):
        raise ValueError("Native execution/effect/statistic/scalar/source reconciliation differs")


def _measure_writer_effects(source: _Source) -> _WriterMeasurements:
    """No public result escapes this BUSY stage; no second source/model/baseline."""
    source.audit(measuring=True)
    pf = source.complete_preflight
    if pf is None or source.complete_limits is None:
        raise ValueError("Full numeric AND canonical JSON preflight required before all passes")
    rows = _observed_controls(source, budget=pf.observed_budget)
    passes = []
    for measurement in rows:
        seal = measurement.retention_seal
        assert seal is not None
        passes.append(
            _WriterPass(measurement.row.row_index, "empty", None, None, None, None, None, seal)
        )
    empty_by_row = {p.row_index: p for p in passes}
    design = source.design.payload
    message = type(design.policy) is ScientificAblationMessageCausalPolicyV1
    donors = {
        row: (other, pair.sha256)
        for pair in design.donor_pairs
        for row, other in (
            (pair.first_row_index, pair.second_row_index),
            (pair.second_row_index, pair.first_row_index),
        )
    }
    for measurement in rows:
        index = measurement.row.row_index
        empty = empty_by_row[index]
        for cycle in range(1, design.geometry.cycle_count):
            written = empty.output.controlled.diagnostic_trace.cycles[cycle - 1]
            if not message:
                passes.append(_trial(source, measurement, "shuffle", cycle))
            for writer, slot in zip(written.write_modules, written.selected_slots, strict=True):
                passes.append(
                    _trial(source, measurement, "block" if message else "zero", cycle, writer, slot)
                )
                if message:
                    donor_row, pair = donors[index]
                    donor = empty_by_row[donor_row].output.controlled.messages[
                        (cycle - 1) * design.geometry.num_modules + writer
                    ]
                    passes.append(
                        _trial(
                            source,
                            measurement,
                            "donor",
                            cycle,
                            writer,
                            slot,
                            donor_row,
                            pair,
                            donor,
                        )
                    )
    native_passes = tuple(passes)
    effects = _derive_effects(source, rows, native_passes)
    result = _WriterMeasurements(
        source.design.sha256,
        pf.layout_sha256,
        rows,
        native_passes,
        effects,
        build_scientific_ablation_writer_statistics_v1(source.design, effects),
        len(rows) + len(native_passes),
        pf.observed_budget.payload.total_trace_proposal_observation_scalar_bound,
        sum(
            getattr(s, f.name).numel()
            for p in passes
            for s in p.output.controlled.workspace_trace
            for f in fields(WorkspaceState)
        ),
    )
    _audit_writer_measurements(source, result)
    return result
