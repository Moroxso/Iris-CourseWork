"""Compact complete-test ablation allocations, not baseline or launch evidence.

Only count/entropy primitives are shared with the baseline collapse contract.
Actual mode-bound traces and original score acknowledgements remain distinct.
Trace/member digests are integrity references, not execution attestation.
"""

from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationAckPayloadV2,
    ScientificAblationEvaluationRowPayloadV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_collapse_v1 import (
    ScientificCollapseCriterionV1,
    ScientificCollapseCycleV1,
    ScientificCollapseGroupV1,
    ScientificCollapsePolicyV1,
    ScientificCollapseSummaryV1,
    scientific_collapse_budget_v1,
    scientific_collapse_criterion_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.trace import WorkspaceTraceV1, _entropy


class ScientificAblationCollapsePolicyV1(_Strict):
    coverage: Literal["complete_original_test_order"] = "complete_original_test_order"
    trace_mode: Literal["metadata"] = "metadata"
    maximum_examples: int = Field(default=5000, ge=1, le=5000)
    maximum_teacher_executions: int = Field(default=5000, ge=1, le=5000)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_trace_scalars: int = Field(default=16_000_000, ge=1, le=64_000_000)


def scientific_ablation_collapse_budget_v1(
    policy: ScientificAblationCollapsePolicyV1,
    *,
    examples: int,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    writers: int,
) -> tuple[int, int]:
    """Budget ALL transient metadata, not merely the persisted compact witness."""
    policy = ScientificAblationCollapsePolicyV1.model_validate(policy)
    return scientific_collapse_budget_v1(
        ScientificCollapsePolicyV1(
            maximum_examples=policy.maximum_examples,
            maximum_executions=policy.maximum_teacher_executions,
            maximum_trace_scalars=policy.maximum_trace_scalars,
            maximum_total_trace_scalars=policy.maximum_total_trace_scalars,
        ),
        examples=examples,
        modules=modules,
        slots=slots,
        width=width,
        cycles=cycles,
        writers=writers,
    )


class ScientificAblationCollapseMemberV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    cognitive_prompt_sha256: Digest
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)


class ScientificAblationCollapseTraceContextV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    checkpoint_sha256: Digest
    ablation_policy_sha256: Digest
    seed: int = Field(ge=0, le=2**63 - 1)
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    maximum_trace_scalars: int = Field(ge=1, le=2_000_000)


class ScientificAblationCollapseRowV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    evaluation_row_sha256: Digest
    evaluation_ack_sha256: Digest
    teacher_invocation_sha256: Digest
    teacher_target_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    target_nll_sum: float = Field(ge=0.0)
    metadata_trace_sha256: Digest
    context: ScientificAblationCollapseTraceContextV1
    cycles: tuple[ScientificCollapseCycleV1, ...] = Field(min_length=1)


def scientific_ablation_collapse_witness_v1(
    trace: WorkspaceTraceV1, *, expected_ablation_policy_sha256: str
) -> tuple[Digest, ScientificAblationCollapseTraceContextV1, tuple[ScientificCollapseCycleV1, ...]]:
    """Detach actual allocations without erasing their non-null ablation context."""
    if type(trace) is not WorkspaceTraceV1:
        raise ValueError("Exact actual ablation metadata trace required")
    trace = WorkspaceTraceV1.model_validate(trace)
    if (
        trace.options.mode != "metadata"
        or type(expected_ablation_policy_sha256) is not str
        or trace.intervention_plan_sha256 is None
        or trace.intervention_plan_sha256 != expected_ablation_policy_sha256
    ):
        raise ValueError("Full/baseline/other-mode traces cannot witness this ablation")
    return (
        domain_sha256(
            "iris/scientific-ablation-collapse-metadata-trace/v1",
            canonical_json_bytes(trace.model_dump(mode="json")),
        ),
        ScientificAblationCollapseTraceContextV1(
            example_id=trace.options.example_id,
            checkpoint_sha256=trace.options.checkpoint_id,
            ablation_policy_sha256=trace.intervention_plan_sha256,
            seed=trace.options.seed,
            num_modules=trace.num_modules,
            num_slots=trace.num_slots,
            state_dim=trace.state_dim,
            cycle_count=trace.cycle_count,
            maximum_trace_scalars=trace.options.maximum_scalars,
        ),
        tuple(
            ScientificCollapseCycleV1(
                cycle=cycle.cycle,
                write_modules=cycle.write_modules,
                selected_slots=cycle.selected_slots,
            )
            for cycle in trace.cycles
        ),
    )


def summarize_scientific_ablation_collapse_v1(
    rows: tuple[ScientificAblationCollapseRowV1, ...], *, modules: int, slots: int
) -> ScientificCollapseSummaryV1:
    """Pool raw allocations, never individual entropies or different mode/seed contexts."""
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not ScientificAblationCollapseRowV1 for row in rows)
        or type(modules) is not int
        or type(slots) is not int
        or min(modules, slots) < 1
    ):
        raise ValueError("Exact nonempty ablation rows and positive geometry required")
    checked = tuple(ScientificAblationCollapseRowV1.model_validate(row) for row in rows)
    first = checked[0].context
    if (
        len({row.row_index for row in checked}) != len(checked)
        or len({row.context.example_id for row in checked}) != len(checked)
        or (first.num_modules, first.num_slots) != (modules, slots)
        or any(
            row.context.model_dump(exclude={"example_id"})
            != first.model_dump(exclude={"example_id"})
            or len(row.cycles) != first.cycle_count
            for row in checked
        )
    ):
        raise ValueError("Duplicate or mixed mode/checkpoint/seed/geometry witnesses")
    module_counts, slot_counts = [0] * modules, [0] * slots
    for row in checked:
        for number, cycle in enumerate(row.cycles, 1):
            if (
                cycle.cycle != number
                or len(cycle.write_modules) != len(cycle.selected_slots)
                or tuple(sorted(cycle.write_modules)) != cycle.write_modules
            ):
                raise ValueError("Actual ordered module-to-slot correspondence differs")
            for indices, counts in (
                (cycle.write_modules, module_counts),
                (cycle.selected_slots, slot_counts),
            ):
                if len(set(indices)) != len(indices) or any(
                    type(index) is not int or not 0 <= index < len(counts) for index in indices
                ):
                    raise ValueError("Unique in-domain actual allocation indices required")
                for index in indices:
                    counts[index] += 1
    mh, me = _entropy(module_counts)
    sh, se = _entropy(slot_counts)
    return ScientificCollapseSummaryV1(
        examples=len(checked),
        cycles=sum(len(row.cycles) for row in checked),
        allocated_writes=sum(module_counts),
        module_win_counts=tuple(module_counts),
        slot_allocation_counts=tuple(slot_counts),
        module_entropy_nats=mh,
        slot_entropy_nats=sh,
        effective_active_modules=me,
        effective_active_slots=se,
    )


def scientific_ablation_collapse_groups_v1(
    members: tuple[ScientificAblationCollapseMemberV1, ...],
    rows: tuple[ScientificAblationCollapseRowV1, ...],
    *,
    modules: int,
    slots: int,
) -> tuple[ScientificCollapseGroupV1, ...]:
    """All twenty source-order descriptive groups, including explicit empty cells."""
    if (
        type(members) is not tuple
        or any(type(member) is not ScientificAblationCollapseMemberV1 for member in members)
        or len(members) != len(rows)
        or any(
            member.row_index != row.row_index or member.example_id != row.context.example_id
            for member, row in zip(members, rows, strict=True)
        )
    ):
        raise ValueError("Complete ordered membership/measurement pairing required")
    members = tuple(ScientificAblationCollapseMemberV1.model_validate(member) for member in members)
    summarize_scientific_ablation_collapse_v1(rows, modules=modules, slots=slots)
    selectors: tuple[tuple[SyntheticTask | None, int | None], ...] = (
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    )
    groups = []
    for task, level in selectors:
        selected = tuple(
            row
            for member, row in zip(members, rows, strict=True)
            if (task is None or member.task_type == task)
            and (level is None or member.difficulty == level)
        )
        groups.append(
            ScientificCollapseGroupV1(
                task_type=task,
                difficulty=level,
                examples=len(selected),
                summary=summarize_scientific_ablation_collapse_v1(
                    selected, modules=modules, slots=slots
                )
                if selected
                else None,
            )
        )
    return tuple(groups)


class ScientificAblationCollapsePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-collapse-v1"] = "iris-scientific-ablation-collapse-v1"
    scope: Literal["complete_test_allocations_not_causal_message_or_launch_approval"] = (
        "complete_test_allocations_not_causal_message_or_launch_approval"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    entropy_definition: Literal["exp-shannon-natural-log-of-pooled-counts-v1"] = (
        "exp-shannon-natural-log-of-pooled-counts-v1"
    )
    evaluation_plan: IrisScientificAblationEvaluationPlanV2
    evaluation: IrisScientificAblationEvaluationV2
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2
    policy: ScientificAblationCollapsePolicyV1
    policy_sha256: Digest
    membership_sha256: Digest
    ordinary_diagnostics_sha256: Digest
    run_seed: int = Field(ge=0, le=2**63 - 1)
    teacher_graph_sha256: Digest
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    writers_per_cycle: int = Field(ge=1)
    members: tuple[ScientificAblationCollapseMemberV1, ...] = Field(min_length=1, max_length=5000)
    rows: tuple[ScientificAblationCollapseRowV1, ...] = Field(min_length=1, max_length=5000)
    teacher_executions: int = Field(ge=1, le=5000)
    trace_scalar_bound: int = Field(ge=1)
    groups: tuple[ScientificCollapseGroupV1, ...] = Field(min_length=20, max_length=20)
    source_20_4: ScientificCollapseCriterionV1

    @model_validator(mode="after")
    def complete_bound_counts(self) -> Self:
        plan, report = self.evaluation_plan.payload, self.evaluation.payload
        for manifest, domain in (
            (self.evaluation_plan, "iris/scientific-ablation-evaluation-plan/v2"),
            (self.evaluation, "iris/scientific-ablation-evaluation/v2"),
            (self.evaluation_complete, "iris/scientific-ablation-evaluation-complete/v2"),
        ):
            if manifest.sha256 != domain_sha256(
                domain, canonical_json_bytes(manifest.payload.model_dump(mode="json"))
            ):
                raise ValueError("Original complete score ledger manifest checksum differs")
        if (
            report.plan != plan.generation_plan
            or report.capture != plan.capture
            or len(self.rows) != len(report.rows)
            or len(self.members) != len(self.rows)
            or self.teacher_graph_sha256 != report.plan.payload.teacher_graph_sha256
            or self.cycle_count != report.plan.payload.cycles
            or self.policy_sha256
            != domain_sha256(
                "iris/scientific-ablation-collapse-policy/v1",
                canonical_json_bytes(self.policy.model_dump(mode="json")),
            )
            or self.membership_sha256
            != domain_sha256(
                "iris/scientific-ablation-collapse-members/v1",
                canonical_json_bytes(
                    tuple(member.model_dump(mode="json") for member in self.members)
                ),
            )
            or any(
                len({getattr(member, field) for member in self.members}) != len(self.members)
                for field in (
                    "example_id",
                    "example_sha256",
                    "prompt_sha256",
                    "cognitive_prompt_sha256",
                )
            )
        ):
            raise ValueError("Complete unique source/test/mode/policy inventory differs")
        source = report.plan.payload.source.payload
        budget = scientific_ablation_collapse_budget_v1(
            self.policy,
            examples=len(self.rows),
            modules=self.num_modules,
            slots=self.num_slots,
            width=self.state_dim,
            cycles=self.cycle_count,
            writers=self.writers_per_cycle,
        )
        if (self.teacher_executions, self.trace_scalar_bound) != budget:
            raise ValueError("ALL metadata/execution budgets must be accounted before compacting")
        head = domain_sha256(
            "iris/scientific-ablation-evaluation-ack/v2/genesis",
            self.evaluation_plan.sha256.encode(),
        )
        for index, (member, row, score, expected) in enumerate(
            zip(self.members, self.rows, report.rows, report.plan.payload.rows, strict=True)
        ):
            record_payload = ScientificAblationEvaluationRowPayloadV2(
                plan_sha256=self.evaluation_plan.sha256, measurements=score
            )
            record_sha = domain_sha256(
                "iris/scientific-ablation-evaluation-row/v2",
                canonical_json_bytes(record_payload.model_dump(mode="json")),
            )
            content = canonical_json_file_bytes(
                IrisScientificAblationEvaluationRowV2(
                    sha256=record_sha, payload=record_payload
                ).model_dump(mode="json")
            )
            ack = ScientificAblationEvaluationAckPayloadV2(
                plan_sha256=self.evaluation_plan.sha256,
                row_index=index,
                previous_ack_sha256=head,
                row_sha256=record_sha,
                row_file_sha256=sha256(content).hexdigest(),
                row_file_bytes=len(content),
            )
            head = domain_sha256(
                "iris/scientific-ablation-evaluation-ack/v2",
                canonical_json_bytes(ack.model_dump(mode="json")),
            )
            if (
                member.row_index != index
                or row.row_index != index
                or member.example_id != expected.example_id
                or member.example_sha256 != expected.example_sha256
                or member.prompt_sha256 != expected.prompt_sha256
                or member.cognitive_prompt_sha256 != expected.cognitive_prompt_sha256
                or member.task_type != expected.task_type
                or row.capture_row_sha256 != score.capture_row_sha256
                or row.evaluation_row_sha256 != record_sha
                or row.evaluation_ack_sha256 != head
                or row.teacher_invocation_sha256 != score.teacher_invocation_sha256
                or row.teacher_target_sha256 != score.teacher_target_sha256
                or row.target_token_count != score.target_token_count
                or row.target_nll_sum != score.target_nll_sum
                or row.context
                != ScientificAblationCollapseTraceContextV1(
                    example_id=member.example_id,
                    checkpoint_sha256=source.best_checkpoint_sha256,
                    ablation_policy_sha256=source.ablation_policy_sha256,
                    seed=self.run_seed,
                    num_modules=self.num_modules,
                    num_slots=self.num_slots,
                    state_dim=self.state_dim,
                    cycle_count=self.cycle_count,
                    maximum_trace_scalars=self.policy.maximum_trace_scalars,
                )
                or len(row.cycles) != self.cycle_count
                or any(len(cycle.write_modules) != self.writers_per_cycle for cycle in row.cycles)
            ):
                raise ValueError("Actual ordered best/mode/allocation/EOS/ack witness differs")
        complete = self.evaluation_complete.payload
        if (
            complete.plan_sha256 != self.evaluation_plan.sha256
            or complete.capture_sha256 != plan.capture.sha256
            or complete.evaluation_sha256 != self.evaluation.sha256
            or complete.acknowledgement_head_sha256 != head
            or complete.example_count != len(self.rows)
        ):
            raise ValueError("Original complete score acknowledgement chain differs")
        groups = scientific_ablation_collapse_groups_v1(
            self.members, self.rows, modules=self.num_modules, slots=self.num_slots
        )
        summary = groups[0].summary
        assert summary is not None
        if self.groups != groups or self.source_20_4 != scientific_collapse_criterion_v1(
            purpose=source.purpose, summary=summary
        ):
            raise ValueError("All twenty pooled groups or fixed source20.4 outcome differs")
        return self


class IrisScientificAblationCollapseV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCollapsePayloadV1


__all__ = [
    "IrisScientificAblationCollapseV1",
    "ScientificAblationCollapseMemberV1",
    "ScientificAblationCollapsePayloadV1",
    "ScientificAblationCollapsePolicyV1",
    "ScientificAblationCollapseRowV1",
    "ScientificAblationCollapseTraceContextV1",
    "scientific_ablation_collapse_budget_v1",
    "scientific_ablation_collapse_groups_v1",
    "scientific_ablation_collapse_witness_v1",
    "summarize_scientific_ablation_collapse_v1",
]
