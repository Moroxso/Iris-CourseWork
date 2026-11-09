"""Actual isolated write/paired donor-content experiments on a loaded eval teacher."""

from dataclasses import dataclass, fields

import torch

from iris.application.workspace_causal import _same_pre_write
from iris.interventions.messages import (
    WorkspaceMessagePlanV1,
    message_plan_sha256_v1,
    message_snapshot_sha256_v1,
)
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.training.messages import IrisMessageTeacherOutputV1
from iris.training.model import IrisTeacherForcingModel
from iris.training.outputs import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import relative_state_distance_v1
from iris.workspace.message_causal import (
    MessageCausalEffectV1,
    MessageCausalSubjectV1,
    WorkspaceMessageCausalCaptureV1,
    WorkspaceMessageCausalPolicyV1,
    message_causal_budget_v1,
    message_causal_statistics_v1,
    pair_message_subjects_v1,
)
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceV1


@dataclass(frozen=True, slots=True)
class PreparedMessageExampleV1:
    example_id: str
    stratum: str
    example: IrisTeacherForcingInput


def _check_write_isolation(
    control: IrisMessageTeacherOutputV1,
    changed: IrisMessageTeacherOutputV1 | IrisIntervenedTeacherOutputV1,
    *,
    cycle: int,
    slot: int,
    transferred: bool,
) -> None:
    trace = changed.diagnostic_trace
    assert trace is not None
    _same_pre_write(control.diagnostic_trace, trace, cycle)
    ordinary, counterfactual = (
        control.workspace_trace[cycle - 1],
        changed.workspace_trace[cycle - 1],
    )
    for field in fields(WorkspaceState):
        left, right = getattr(ordinary, field.name), getattr(counterfactual, field.name)
        indices = [i for i in range(ordinary.slots.shape[1]) if i != slot]
        if not torch.equal(left[:, indices], right[:, indices]):
            raise ValueError("Message counterfactual changed another slot record")
        if transferred and field.name != "slots" and not torch.equal(left, right):
            raise ValueError("Donor content changed recipient slot metadata")
    written = control.diagnostic_trace.cycles[cycle - 1]
    assert written.full is not None
    if not transferred and (
        tuple(float(v) for v in counterfactual.slots[0, slot].detach().double().cpu().tolist())
        != written.full.read_workspace_slots[slot]
        or bool(counterfactual.occupancy[0, slot]) != written.read_slot_occupancy[slot]
        or int(counterfactual.source_module[0, slot]) != written.read_slot_sources[slot]
    ):
        raise ValueError("Write blocker did not restore the actual pre-write record")
    if transferred:
        assert type(changed) is IrisMessageTeacherOutputV1
        if tuple(m for m in changed.messages if m.cycle <= cycle) != tuple(
            m for m in control.messages if m.cycle <= cycle
        ):
            raise ValueError("Message transplant changed computation before its scheduled writer")


def measure_workspace_message_causal_v1(
    model: IrisTeacherForcingModel,
    examples: tuple[PreparedMessageExampleV1, ...],
    *,
    seed: int,
    policy: WorkspaceMessageCausalPolicyV1,
) -> WorkspaceMessageCausalCaptureV1:
    """Measure every allocated non-final write and all other next-cycle modules.

    For each recipient, compare its genuine write and its paired donor's content
    with the SAME single-write-blocked background. Donors use the same module and
    cycle, even if that donor module did not win. No label/outcome selects donors.
    """
    if (
        type(model) is not IrisTeacherForcingModel
        or type(examples) is not tuple
        or any(
            type(e) is not PreparedMessageExampleV1
            or type(e.example) is not IrisTeacherForcingInput
            for e in examples
        )
    ):
        raise TypeError("Exact loaded teacher and prepared message examples required")
    if type(policy) is not WorkspaceMessageCausalPolicyV1:
        raise TypeError("Exact message causal policy required")
    policy = WorkspaceMessageCausalPolicyV1.model_validate(policy)
    if any(module.training for module in model.modules()):
        raise ValueError("Message causal measurement requires explicit eval mode")
    provenance = model.evaluation_provenance
    cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    workspace = WorkspaceConfig.model_validate_json(provenance.workspace_configuration_json)
    graph = domain_sha256(
        "iris/intervened-teacher-graph/v1",
        canonical_json_bytes(
            {
                "backbone": provenance.backbone_configuration_json,
                "cognitive": provenance.cognitive_configuration_json,
                "workspace": provenance.workspace_configuration_json,
            }
        ),
    )
    subjects = tuple(
        sorted(
            (
                MessageCausalSubjectV1(
                    example_id=e.example_id,
                    stratum=e.stratum,
                    prompt_sha256=domain_sha256(
                        "iris/intervened-teacher-prompt/v1", e.example.prompt.encode("utf-8")
                    ),
                )
                for e in examples
            ),
            key=lambda s: s.example_id,
        )
    )
    pairs = pair_message_subjects_v1(subjects, policy)
    executions, volume = message_causal_budget_v1(
        policy,
        examples=len(subjects),
        modules=cognitive.num_modules,
        slots=workspace.num_slots,
        width=cognitive.state_dim,
        cycles=workspace.cycles,
        writers=workspace.winners_per_cycle,
    )
    owned = {e.example_id: e.example for e in examples}
    targets = {}
    options = {}
    for subject in subjects:
        example = owned[subject.example_id]
        model._validate_example(example)
        targets[subject.example_id] = (
            teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask),
            int(example.target_attention_mask.sum()),
        )
        options[subject.example_id] = WorkspaceTraceOptionsV1(
            mode="full",
            example_id=subject.example_id,
            checkpoint_id=provenance.sidecar_state_sha256,
            seed=seed,
            maximum_scalars=policy.maximum_trace_scalars,
        )
    calls = 0

    def verify(
        subject: MessageCausalSubjectV1,
        output: IrisMessageTeacherOutputV1 | IrisIntervenedTeacherOutputV1,
        digest: str,
    ) -> WorkspaceTraceV1:
        trace = output.diagnostic_trace
        if (
            output.prompt_sha256 != subject.prompt_sha256
            or (output.target_sha256, output.supervised_token_count) != targets[subject.example_id]
            or output.teacher_graph_sha256 != graph
            or output.sidecar_state_sha256 != provenance.sidecar_state_sha256
            or type(trace) is not WorkspaceTraceV1
        ):
            raise ValueError("Message experiment returned a different prompt/target/model receipt")
        trace = WorkspaceTraceV1.model_validate(trace)
        if (
            trace.options != options[subject.example_id]
            or trace.intervention_plan_sha256 != digest
            or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
            != (cognitive.num_modules, workspace.num_slots, cognitive.state_dim, workspace.cycles)
        ):
            raise ValueError("Message experiment trace differs from executed context")
        return trace

    def messages(
        subject: MessageCausalSubjectV1, plan: WorkspaceMessagePlanV1
    ) -> IrisMessageTeacherOutputV1:
        nonlocal calls
        output = model.evaluation_forward_messages(
            owned[subject.example_id],
            plan=plan,
            diagnostics=options[subject.example_id],
            maximum_message_scalars=policy.maximum_message_scalars,
        )
        calls += 1
        digest = message_plan_sha256_v1(plan)
        if type(output) is not IrisMessageTeacherOutputV1 or output.message_plan_sha256 != digest:
            raise ValueError("Exact message execution receipt required")
        verify(subject, output, digest)
        # Revalidate receipt/snapshots even if a DTO was modified after construction.
        output.__post_init__()
        return output

    controls = {
        subject.example_id: messages(subject, WorkspaceMessagePlanV1()) for subject in subjects
    }
    snapshots = {
        (id, m.cycle, m.module): m for id, control in controls.items() for m in control.messages
    }
    donors = {}
    for pair in pairs:
        donors[pair.first_example_id] = (pair.second_example_id, pair.sha256)
        donors[pair.second_example_id] = (pair.first_example_id, pair.sha256)
    effects = []
    for subject in subjects:
        control = controls[subject.example_id]
        donor_id, pair_sha = donors[subject.example_id]
        trace = control.diagnostic_trace
        for cycle in range(1, workspace.cycles):
            written, next_read = trace.cycles[cycle - 1], trace.cycles[cycle]
            assert next_read.full is not None
            if len(written.write_modules) != workspace.winners_per_cycle:
                raise ValueError("Control is missing expected allocated writers")
            for source, slot in zip(written.write_modules, written.selected_slots, strict=True):
                if not written.slot_occupancy[slot] or written.slot_sources[slot] != source:
                    raise ValueError("Recipient control did not commit its declared write")
                block = WorkspaceInterventionPlanV1(
                    interventions=(
                        WorkspaceInterventionSpecV1(
                            kind="block_module_write",
                            module=source,
                            cycles=(cycle,),
                        ),
                    )
                )
                blocked = model.evaluation_forward_intervened(
                    owned[subject.example_id],
                    plan=block,
                    diagnostics=options[subject.example_id],
                )
                calls += 1
                block_sha = intervention_plan_sha256_v1(block)
                if (
                    type(blocked) is not IrisIntervenedTeacherOutputV1
                    or blocked.intervention_plan_sha256 != block_sha
                ):
                    raise ValueError("Exact isolated-write blocking receipt required")
                blocked_trace = verify(subject, blocked, block_sha)
                donor = snapshots[donor_id, cycle, source]
                transfer_plan = WorkspaceMessagePlanV1(donor=donor)
                transferred = messages(subject, transfer_plan)
                _check_write_isolation(control, blocked, cycle=cycle, slot=slot, transferred=False)
                _check_write_isolation(
                    control, transferred, cycle=cycle, slot=slot, transferred=True
                )
                blocked_next = blocked_trace.cycles[cycle].full
                transferred_next = transferred.diagnostic_trace.cycles[cycle].full
                assert blocked_next is not None and transferred_next is not None
                for receiver in range(cognitive.num_modules):
                    if receiver == source:
                        continue
                    background = blocked_next.module_states[receiver]
                    effects.append(
                        MessageCausalEffectV1(
                            example_id=subject.example_id,
                            donor_example_id=donor_id,
                            donor_pair_sha256=pair_sha,
                            write_cycle=cycle,
                            source_module=source,
                            slot=slot,
                            target_module=receiver,
                            overwrote_existing_record=slot in written.overwritten_slots,
                            survives_to_next_read=next_read.read_slot_occupancy[slot]
                            and next_read.read_slot_sources[slot] == source,
                            genuine_write_distance=relative_state_distance_v1(
                                background, next_read.full.module_states[receiver]
                            ),
                            donor_write_distance=relative_state_distance_v1(
                                background, transferred_next.module_states[receiver]
                            ),
                            recipient_message_sha256=message_snapshot_sha256_v1(
                                snapshots[subject.example_id, cycle, source]
                            ),
                            donor_message_sha256=message_snapshot_sha256_v1(donor),
                            block_plan_sha256=block_sha,
                            transfer_plan_sha256=message_plan_sha256_v1(transfer_plan),
                        )
                    )
                del blocked, transferred
    if calls != executions or model.evaluation_provenance != provenance:
        raise ValueError("Message experiment execution count or model identity changed")
    for subject in subjects:
        example = owned[subject.example_id]
        if (
            domain_sha256("iris/intervened-teacher-prompt/v1", example.prompt.encode("utf-8"))
            != subject.prompt_sha256
            or teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
            != targets[subject.example_id][0]
        ):
            raise ValueError("Owned inputs changed during the message experiment")
    records = tuple(effects)
    return WorkspaceMessageCausalCaptureV1(
        policy=policy,
        teacher_graph_sha256=graph,
        sidecar_state_sha256=provenance.sidecar_state_sha256,
        seed=seed,
        num_modules=cognitive.num_modules,
        num_slots=workspace.num_slots,
        state_dim=cognitive.state_dim,
        cycle_count=workspace.cycles,
        writers_per_cycle=workspace.winners_per_cycle,
        executions=executions,
        trace_scalar_bound=volume,
        subjects=subjects,
        donor_pairs=pairs,
        effects=records,
        statistics=message_causal_statistics_v1(
            pairs, records, modules=cognitive.num_modules, policy=policy
        ),
    )
