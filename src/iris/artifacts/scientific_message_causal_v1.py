"""Selected-best, source-bound paired message diagnostics; never launch acceptance."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_workspace_v1 import (
    ScientificWorkspaceGroupV1,
    ScientificWorkspaceMemberV1,
    ScientificWorkspaceRowV1,
    scientific_workspace_groups_v1,
)
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.message_causal import (
    MessageCausalSubjectV1,
    WorkspaceMessageCausalCaptureV1,
    WorkspaceMessageCausalPolicyV1,
    message_causal_budget_v1,
)
from iris.workspace.trace import _scalar_bound


class ScientificMessageCausalPolicyV1(_Strict):
    selection: Literal["hash-ranked-even-task-difficulty-subset-v1"] = (
        "hash-ranked-even-task-difficulty-subset-v1"
    )
    selection_seed: int = Field(ge=0, le=2**63 - 1)
    examples_per_stratum: int = Field(default=4, ge=2, le=64)
    trace_mode: Literal["metadata", "full"] = "metadata"
    causal: WorkspaceMessageCausalPolicyV1

    @model_validator(mode="after")
    def even_cap(self) -> Self:
        if self.examples_per_stratum % 2:
            raise ValueError("An even donor subset cap is required")
        return self


def message_stratum_v1(member: ScientificWorkspaceMemberV1) -> str:
    return f"{member.task_type}/difficulty-{member.difficulty}"


def select_message_members_v1(
    members: tuple[ScientificWorkspaceMemberV1, ...], policy: ScientificMessageCausalPolicyV1
) -> tuple[ScientificWorkspaceMemberV1, ...]:
    """Input-only even selection; omit a last odd row by a fixed declared rule."""
    policy = ScientificMessageCausalPolicyV1.model_validate(policy)
    members = tuple(ScientificWorkspaceMemberV1.model_validate(m) for m in members)
    if not 1 <= len(members) <= 5000 or any(
        len({getattr(m, field) for m in members}) != len(members)
        for field in ("example_id", "prompt_sha256", "teacher_prompt_sha256")
    ):
        raise ValueError("Unique complete held-out membership required")

    def rank(member: ScientificWorkspaceMemberV1) -> tuple[str, str]:
        return (
            domain_sha256(
                "iris/scientific-message-selection/v1",
                canonical_json_bytes(
                    {
                        "seed": policy.selection_seed,
                        "example_id": member.example_id,
                        "prompt_sha256": member.prompt_sha256,
                    }
                ),
            ),
            member.example_id,
        )

    selected = []
    for task in TASKS:
        for difficulty in (1, 2, 3):
            rows = sorted(
                (m for m in members if (m.task_type, m.difficulty) == (task, difficulty)),
                key=rank,
            )
            count = min(len(rows), policy.examples_per_stratum)
            selected.extend(rows[: count - count % 2])
    if not 2 <= len(selected) <= policy.causal.maximum_examples:
        raise ValueError("No bounded paired subset fits the declared input strata")
    return tuple(selected)


def scientific_message_budget_v1(
    policy: ScientificMessageCausalPolicyV1,
    *,
    examples: int,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    writers: int,
) -> tuple[int, int]:
    """Include the ordinary selected-best NLL/trace pass, not just causal reruns."""
    calls, volume = message_causal_budget_v1(
        policy.causal,
        examples=examples,
        modules=modules,
        slots=slots,
        width=width,
        cycles=cycles,
        writers=writers,
    )
    calls += examples
    volume += examples * _scalar_bound(modules, slots, width, cycles, policy.trace_mode == "full")
    if (
        calls > policy.causal.maximum_executions
        or volume > policy.causal.maximum_total_trace_scalars
    ):
        raise ValueError("Whole source-bound message experiment exceeds budget")
    return calls, volume


class ScientificMessageCausalPayloadV1(_Strict):
    kind: Literal["iris-scientific-message-causal-v1"] = "iris-scientific-message-causal-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    aggregation_scope: Literal["even_stratified_subset_not_full_corpus_estimate"] = (
        "even_stratified_subset_not_full_corpus_estimate"
    )
    policy: ScientificMessageCausalPolicyV1
    baseline_generation_sha256: Digest
    execution_sha256: Digest
    terminal_event_sha256: Digest
    best_validation_event_sha256: Digest
    best_checkpoint_sha256: Digest
    inference_sha256: Digest
    selected_step: int = Field(ge=1)
    terminal_step: int = Field(ge=1)
    run_seed: int = Field(ge=0, le=2**63 - 1)
    teacher_configuration_sha256: Digest
    teacher_graph_sha256: Digest
    teacher_sidecar_sha256: Digest
    backbone_runtime_sha256: Digest
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    members: tuple[ScientificWorkspaceMemberV1, ...] = Field(min_length=1, max_length=5000)
    rows: tuple[ScientificWorkspaceRowV1, ...] = Field(min_length=2, max_length=512)
    causal: WorkspaceMessageCausalCaptureV1
    executions: int = Field(ge=1)
    trace_scalar_bound: int = Field(ge=1)
    groups: tuple[ScientificWorkspaceGroupV1, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def bound_design(self) -> Self:
        selected = select_message_members_v1(self.members, self.policy)
        causal = self.causal
        if (
            self.selected_step > self.terminal_step
            or tuple(m.example_id for m in selected) != tuple(r.example_id for r in self.rows)
            or causal.subjects
            != tuple(
                sorted(
                    (
                        MessageCausalSubjectV1(
                            example_id=m.example_id,
                            prompt_sha256=m.teacher_prompt_sha256,
                            stratum=message_stratum_v1(m),
                        )
                        for m in selected
                    ),
                    key=lambda s: s.example_id,
                )
            )
            or causal.policy != self.policy.causal
            or causal.seed != self.run_seed
            or causal.teacher_graph_sha256 != self.teacher_graph_sha256
            or causal.sidecar_state_sha256 != self.teacher_sidecar_sha256
        ):
            raise ValueError("Message subset/model/seed differs from selected source")
        if self.purpose == "scientific" and tuple(
            sum(m.task_type == task for m in self.members) for task in TASKS
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific membership requires all5000 held-out rows")
        expected = scientific_message_budget_v1(
            self.policy,
            examples=len(self.rows),
            modules=causal.num_modules,
            slots=causal.num_slots,
            width=causal.state_dim,
            cycles=causal.cycle_count,
            writers=causal.writers_per_cycle,
        )
        if (self.executions, self.trace_scalar_bound) != expected:
            raise ValueError("Source-bound message execution/scalar volume differs")
        for row in self.rows:
            trace = row.trace
            if (
                row.causal is not None
                or trace.options.example_id != row.example_id
                or trace.options.checkpoint_id != self.best_checkpoint_sha256
                or trace.options.seed != self.run_seed
                or trace.options.mode != self.policy.trace_mode
                or trace.options.maximum_scalars != self.policy.causal.maximum_trace_scalars
                or trace.intervention_plan_sha256 is not None
                or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
                != (causal.num_modules, causal.num_slots, causal.state_dim, causal.cycle_count)
                or any(len(c.write_modules) != causal.writers_per_cycle for c in trace.cycles)
            ):
                raise ValueError("Ordinary control trace does not match selected best/design")
        controls = {row.example_id: row.trace for row in self.rows}
        for effect in causal.effects:
            trace = controls[effect.example_id]
            written, next_read = trace.cycles[effect.write_cycle - 1 : effect.write_cycle + 1]
            block = WorkspaceInterventionPlanV1(
                interventions=(
                    WorkspaceInterventionSpecV1(
                        kind="block_module_write",
                        module=effect.source_module,
                        cycles=(effect.write_cycle,),
                    ),
                )
            )
            if (
                (effect.source_module, effect.slot)
                not in set(zip(written.write_modules, written.selected_slots, strict=True))
                or effect.overwrote_existing_record != (effect.slot in written.overwritten_slots)
                or effect.survives_to_next_read
                != (
                    next_read.read_slot_occupancy[effect.slot]
                    and next_read.read_slot_sources[effect.slot] == effect.source_module
                )
                or effect.block_plan_sha256 != intervention_plan_sha256_v1(block)
            ):
                raise ValueError("Message trial does not match the actual ordinary writer")
        if self.groups != scientific_workspace_groups_v1(self.members, self.rows):
            raise ValueError("Message control descriptive groups differ from raw traces")
        return self


class IrisScientificMessageCausalV1(_Strict):
    sha256: Digest
    payload: ScientificMessageCausalPayloadV1 = Field(repr=False)
