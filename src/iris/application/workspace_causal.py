"""Execute bounded, paired next-cycle interventions on an already loaded teacher."""

from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.training.model import IrisTeacherForcingModel
from iris.training.outputs import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import (
    WorkspaceCausalCaptureV1,
    WorkspaceCausalEffectV1,
    WorkspaceCausalPolicyV1,
    relative_state_distance_v1,
)
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceRecorderV1, WorkspaceTraceV1


def _same_pre_write(control: WorkspaceTraceV1, changed: WorkspaceTraceV1, cycle: int) -> None:
    if control.cycles[: cycle - 1] != changed.cycles[: cycle - 1]:
        raise ValueError("Counterfactual changed a cycle before its scheduled intervention")
    a, b = control.cycles[cycle - 1], changed.cycles[cycle - 1]
    assert a.full is not None and b.full is not None
    fields = (
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
    )
    if (
        any(getattr(a, field) != getattr(b, field) for field in fields)
        or a.full.module_states != b.full.module_states
        or a.full.workspace_reads != b.full.workspace_reads
        or a.full.read_workspace_slots != b.full.read_workspace_slots
    ):
        raise ValueError("Post-write intervention changed pre-write computation")


def measure_workspace_causal_v1(
    model: IrisTeacherForcingModel,
    example: IrisTeacherForcingInput,
    *,
    example_id: str,
    seed: int,
    policy: WorkspaceCausalPolicyV1,
) -> WorkspaceCausalCaptureV1:
    """Ablate EVERY actual non-final write; observe ALL other modules next cycle.

    An empty-plan control, one slot-record permutation per cycle and one ZeroSlot
    run per writer all use the same frozen sidecar and prompt. Labels never select
    a writer or perturbation seed. This is not checkpoint selection, model loading,
    training, code execution, generation scoring or a scientific launch command.
    """
    if type(model) is not IrisTeacherForcingModel or type(example) is not IrisTeacherForcingInput:
        raise TypeError("An exact loaded teacher and prepared input are required")
    if type(policy) is not WorkspaceCausalPolicyV1:
        raise TypeError("An exact bounded causal policy is required")
    policy = WorkspaceCausalPolicyV1.model_validate(policy)
    if any(module.training for module in model.modules()):
        raise ValueError("Causal measurement requires caller-selected eval mode")
    provenance = model.evaluation_provenance
    cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    workspace = WorkspaceConfig.model_validate_json(provenance.workspace_configuration_json)
    calls = 1 + (workspace.cycles - 1) * (workspace.winners_per_cycle + 1)
    if (
        workspace.cycles < 2
        or not 2 <= cognitive.num_modules <= 64
        or calls > policy.maximum_executions
    ):
        raise ValueError("Next-cycle design exceeds execution budget or has no other module/cycle")
    options = WorkspaceTraceOptionsV1(
        mode="full",
        example_id=example_id,
        checkpoint_id=provenance.sidecar_state_sha256,
        seed=seed,
        maximum_scalars=policy.maximum_trace_scalars,
    )
    # Validate memory budget before ANY expensive model pass.
    WorkspaceTraceRecorderV1(
        options,
        num_modules=cognitive.num_modules,
        num_slots=workspace.num_slots,
        state_dim=cognitive.state_dim,
        cycles=workspace.cycles,
    )
    prompt = domain_sha256("iris/intervened-teacher-prompt/v1", example.prompt.encode("utf-8"))
    target = teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
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

    def execute(plan: WorkspaceInterventionPlanV1) -> WorkspaceTraceV1:
        output = model.evaluation_forward_intervened(example, plan=plan, diagnostics=options)
        digest = intervention_plan_sha256_v1(plan)
        if (
            type(output) is not IrisIntervenedTeacherOutputV1
            or output.prompt_sha256 != prompt
            or output.target_sha256 != target
            or output.teacher_graph_sha256 != graph
            or output.sidecar_state_sha256 != provenance.sidecar_state_sha256
            or output.intervention_plan_sha256 != digest
            or type(output.diagnostic_trace) is not WorkspaceTraceV1
        ):
            raise ValueError("Causal execution receipt differs from the owned experiment")
        trace = WorkspaceTraceV1.model_validate(output.diagnostic_trace)
        if (
            trace.options != options
            or trace.intervention_plan_sha256 != digest
            or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
            != (cognitive.num_modules, workspace.num_slots, cognitive.state_dim, workspace.cycles)
        ):
            raise ValueError("Causal trace differs from actual execution context")
        return trace

    control = execute(WorkspaceInterventionPlanV1(interventions=()))
    effects: list[WorkspaceCausalEffectV1] = []
    for cycle in range(1, workspace.cycles):
        written, next_read = control.cycles[cycle - 1], control.cycles[cycle]
        assert written.full is not None and next_read.full is not None
        if len(written.write_modules) != workspace.winners_per_cycle:
            raise ValueError("Control did not produce the complete expected write set")
        shuffle_seed = (
            int(
                domain_sha256(
                    "iris/causal-slot-shuffle/v1",
                    canonical_json_bytes(
                        {
                            "seed": policy.shuffle_seed,
                            "example_id": example_id,
                            "prompt_sha256": prompt,
                            "cycle": cycle,
                        }
                    ),
                )[:16],
                16,
            )
            % 2**63
        )
        shuffle_plan = WorkspaceInterventionPlanV1(
            interventions=(
                WorkspaceInterventionSpecV1(
                    kind="shuffle_slots",
                    seed=shuffle_seed,
                    cycles=(cycle,),
                ),
            )
        )
        shuffled = execute(shuffle_plan)
        _same_pre_write(control, shuffled, cycle)
        shuffled_next = shuffled.cycles[cycle].full
        assert shuffled_next is not None
        for source, slot in zip(written.write_modules, written.selected_slots, strict=True):
            if not written.slot_occupancy[slot] or written.slot_sources[slot] != source:
                raise ValueError("Observed source write was not present after the control writer")
            ablation_plan = WorkspaceInterventionPlanV1(
                interventions=(
                    WorkspaceInterventionSpecV1(
                        kind="zero_slot",
                        slot=slot,
                        cycles=(cycle,),
                    ),
                )
            )
            ablated = execute(ablation_plan)
            _same_pre_write(control, ablated, cycle)
            ablated_next = ablated.cycles[cycle].full
            assert ablated_next is not None
            survives = (
                next_read.read_slot_occupancy[slot] and next_read.read_slot_sources[slot] == source
            )
            for receiver in range(cognitive.num_modules):
                if receiver == source:
                    continue
                baseline_state = next_read.full.module_states[receiver]
                effects.append(
                    WorkspaceCausalEffectV1(
                        write_cycle=cycle,
                        source_module=source,
                        slot=slot,
                        target_module=receiver,
                        survives_to_next_read=survives,
                        overwrote_existing_record=slot in written.overwritten_slots,
                        ablated_distance=relative_state_distance_v1(
                            baseline_state, ablated_next.module_states[receiver]
                        ),
                        shuffled_distance=relative_state_distance_v1(
                            baseline_state, shuffled_next.module_states[receiver]
                        ),
                        ablation_plan_sha256=intervention_plan_sha256_v1(ablation_plan),
                        shuffle_plan_sha256=intervention_plan_sha256_v1(shuffle_plan),
                    )
                )
    if model.evaluation_provenance != provenance:
        raise ValueError("Causal experiment model provenance changed")
    return WorkspaceCausalCaptureV1(
        policy=policy,
        example_id=example_id,
        prompt_sha256=prompt,
        teacher_graph_sha256=graph,
        sidecar_state_sha256=provenance.sidecar_state_sha256,
        seed=seed,
        num_modules=cognitive.num_modules,
        num_slots=workspace.num_slots,
        cycle_count=workspace.cycles,
        writers_per_cycle=workspace.winners_per_cycle,
        executions=calls,
        effects=tuple(effects),
    )
