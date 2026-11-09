"""Complete held-out hard-allocation witnesses and the isolated source20.4 gate.

No ranked subset, per-example entropy averaging, rescaled thresholds or claim
that this single diagnostic establishes the full scientific acceptance decision.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_workspace_v1 import ScientificWorkspaceMemberV1
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.trace import WorkspaceTraceV1, _entropy, _scalar_bound


class ScientificCollapsePolicyV1(_Strict):
    coverage: Literal["complete-held-out-test-v1"] = "complete-held-out-test-v1"
    trace_mode: Literal["metadata"] = "metadata"
    maximum_examples: int = Field(default=5000, ge=1, le=5000)
    maximum_executions: int = Field(default=5000, ge=1, le=5000)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_trace_scalars: int = Field(default=16_000_000, ge=1, le=128_000_000)


def scientific_collapse_budget_v1(
    policy: ScientificCollapsePolicyV1,
    *,
    examples: int,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    writers: int,
) -> tuple[int, int]:
    """Budget every actual ordinary trace before loading, with no subsampling."""
    policy = ScientificCollapsePolicyV1.model_validate(policy)
    if any(type(v) is not int or v < 1 for v in (examples, modules, slots, width, cycles, writers)):
        raise ValueError("Exact positive full-corpus geometry required")
    scalars = _scalar_bound(modules, slots, width, cycles, False)
    if (
        writers > min(modules, slots)
        or examples > policy.maximum_examples
        or examples > policy.maximum_executions
        or scalars > policy.maximum_trace_scalars
        or examples * scalars > policy.maximum_total_trace_scalars
    ):
        raise ValueError("Full-corpus execution/scalar budget exceeded; never reduce coverage")
    return examples, examples * scalars


class ScientificCollapseCycleV1(_Strict):
    cycle: int = Field(ge=1)
    write_modules: tuple[int, ...] = Field(min_length=1)
    selected_slots: tuple[int, ...] = Field(min_length=1)


class ScientificCollapseTraceContextV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    checkpoint_sha256: Digest
    seed: int = Field(ge=0, le=2**63 - 1)
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    maximum_trace_scalars: int = Field(ge=1, le=2_000_000)


class ScientificCollapseRowV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    target_sha256: Digest
    target_token_count: int = Field(ge=1)
    control_nll_sum: float = Field(ge=0)
    ordinary_trace_sha256: Digest
    ordinary_context: ScientificCollapseTraceContextV1
    cycles: tuple[ScientificCollapseCycleV1, ...] = Field(min_length=1)


def scientific_collapse_witness_v1(
    trace: WorkspaceTraceV1,
) -> tuple[Digest, ScientificCollapseTraceContextV1, tuple[ScientificCollapseCycleV1, ...]]:
    """Detach only actual module-to-slot allocations from a validated normal trace."""
    if type(trace) is not WorkspaceTraceV1:
        raise ValueError("Exact ordinary diagnostic trace required")
    trace = WorkspaceTraceV1.model_validate(trace)
    if trace.options.mode != "metadata" or trace.intervention_plan_sha256 is not None:
        raise ValueError("Only unintervened metadata traces can witness full-test collapse")
    return (
        domain_sha256(
            "iris/scientific-collapse-ordinary-trace/v1",
            canonical_json_bytes(trace.model_dump(mode="json")),
        ),
        ScientificCollapseTraceContextV1(
            example_id=trace.options.example_id,
            checkpoint_sha256=trace.options.checkpoint_id,
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


class ScientificCollapseSummaryV1(_Strict):
    examples: int = Field(ge=1)
    cycles: int = Field(ge=1)
    allocated_writes: int = Field(ge=1)
    module_win_counts: tuple[int, ...]
    slot_allocation_counts: tuple[int, ...]
    module_entropy_nats: float = Field(ge=0)
    slot_entropy_nats: float = Field(ge=0)
    effective_active_modules: float = Field(ge=1)
    effective_active_slots: float = Field(ge=1)


def summarize_scientific_collapse_v1(
    rows: tuple[ScientificCollapseRowV1, ...], *, modules: int, slots: int
) -> ScientificCollapseSummaryV1:
    """Pool counts over every example/cycle before entropy; no seed pooling."""
    if (
        type(rows) is not tuple
        or not rows
        or type(modules) is not int
        or type(slots) is not int
        or modules < 1
        or slots < 1
    ):
        raise ValueError("Nonempty exact rows and positive geometry required")
    checked = tuple(ScientificCollapseRowV1.model_validate(row) for row in rows)
    if len({r.example_id for r in checked}) != len(checked):
        raise ValueError("Duplicate example in full-test usage")
    first = checked[0].ordinary_context
    if (first.num_modules, first.num_slots) != (modules, slots) or any(
        row.ordinary_context.example_id != row.example_id
        or len(row.cycles) != row.ordinary_context.cycle_count
        or row.ordinary_context.model_dump(exclude={"example_id"})
        != first.model_dump(exclude={"example_id"})
        for row in checked
    ):
        raise ValueError("Cannot pool different graph/checkpoint/seed/trace contexts")
    module_counts, slot_counts = [0] * modules, [0] * slots
    for row in checked:
        for number, cycle in enumerate(row.cycles, 1):
            if cycle.cycle != number or len(cycle.write_modules) != len(cycle.selected_slots):
                raise ValueError("Allocation cycles/order/correspondence differ")
            for indices, counts in (
                (cycle.write_modules, module_counts),
                (cycle.selected_slots, slot_counts),
            ):
                if len(set(indices)) != len(indices) or any(
                    not 0 <= i < len(counts) for i in indices
                ):
                    raise ValueError("Hard allocation indices must be unique and in-domain")
                for index in indices:
                    counts[index] += 1
            if tuple(sorted(cycle.write_modules)) != cycle.write_modules:
                raise ValueError("Actual allocator witnesses use ascending module order")
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


class ScientificCollapseGroupV1(_Strict):
    task_type: SyntheticTask | None
    difficulty: int | None = Field(ge=1, le=3)
    examples: int = Field(ge=0)
    summary: ScientificCollapseSummaryV1 | None


def scientific_collapse_groups_v1(
    members: tuple[ScientificWorkspaceMemberV1, ...],
    rows: tuple[ScientificCollapseRowV1, ...],
    *,
    modules: int,
    slots: int,
) -> tuple[ScientificCollapseGroupV1, ...]:
    selectors: tuple[tuple[SyntheticTask | None, int | None], ...] = (
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    )
    groups = []
    for task, level in selectors:
        ids = {
            m.example_id
            for m in members
            if (task is None or m.task_type == task) and (level is None or m.difficulty == level)
        }
        selected = tuple(row for row in rows if row.example_id in ids)
        groups.append(
            ScientificCollapseGroupV1(
                task_type=task,
                difficulty=level,
                examples=len(selected),
                summary=summarize_scientific_collapse_v1(selected, modules=modules, slots=slots)
                if selected
                else None,
            )
        )
    return tuple(groups)


class ScientificCollapseCriterionV1(_Strict):
    source_section: Literal["20.4"] = "20.4"
    required_modules: Literal[8] = 8
    required_slots: Literal[8] = 8
    minimum_effective_modules: Literal[4] = 4
    minimum_effective_slots: Literal[4] = 4
    scope: Literal["one-seed-complete-test-corpus"] = "one-seed-complete-test-corpus"
    outcome: Literal["met", "not_met", "not_assessed_smoke", "not_applicable_geometry"]


def scientific_collapse_criterion_v1(
    *, purpose: Literal["smoke", "scientific"], summary: ScientificCollapseSummaryV1
) -> ScientificCollapseCriterionV1:
    """Fixed source thresholds; caller must validate complete scientific membership."""
    summary = ScientificCollapseSummaryV1.model_validate(summary)
    if purpose == "smoke":
        outcome: Literal["met", "not_met", "not_assessed_smoke", "not_applicable_geometry"] = (
            "not_assessed_smoke"
        )
    elif purpose != "scientific":
        raise ValueError("Explicit source purpose required")
    elif summary.examples != 5000:
        raise ValueError("Source20.4 requires the complete 5000-row scientific test corpus")
    elif (len(summary.module_win_counts), len(summary.slot_allocation_counts)) != (8, 8):
        outcome = "not_applicable_geometry"
    else:
        outcome = (
            "met"
            if summary.effective_active_modules >= 4 and summary.effective_active_slots >= 4
            else "not_met"
        )
    return ScientificCollapseCriterionV1(outcome=outcome)


class ScientificCollapsePayloadV1(_Strict):
    kind: Literal["iris-scientific-collapse-v1"] = "iris-scientific-collapse-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    aggregation_scope: Literal["complete_held_out_test_pooled_hard_allocations"] = (
        "complete_held_out_test_pooled_hard_allocations"
    )
    entropy_definition: Literal["exp-shannon-natural-log-of-pooled-counts-v1"] = (
        "exp-shannon-natural-log-of-pooled-counts-v1"
    )
    policy: ScientificCollapsePolicyV1
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
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    writers_per_cycle: int = Field(ge=1)
    members: tuple[ScientificWorkspaceMemberV1, ...] = Field(min_length=1, max_length=5000)
    rows: tuple[ScientificCollapseRowV1, ...] = Field(min_length=1, max_length=5000)
    executions: int = Field(ge=1)
    trace_scalar_bound: int = Field(ge=1)
    groups: tuple[ScientificCollapseGroupV1, ...] = Field(min_length=20, max_length=20)
    source_20_4: ScientificCollapseCriterionV1

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        if self.selected_step > self.terminal_step or tuple(
            m.example_id for m in self.members
        ) != tuple(row.example_id for row in self.rows):
            raise ValueError("Every audited test member needs one ordered measurement at best")
        if any(
            len({getattr(member, field) for member in self.members}) != len(self.members)
            for field in ("example_id", "prompt_sha256", "teacher_prompt_sha256")
        ):
            raise ValueError("Unique full-test examples and prompts required")
        if self.purpose == "scientific" and tuple(
            sum(m.task_type == task for m in self.members) for task in TASKS
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific collapse needs the complete 5000-row source test corpus")
        budget = scientific_collapse_budget_v1(
            self.policy,
            examples=len(self.rows),
            modules=self.num_modules,
            slots=self.num_slots,
            width=self.state_dim,
            cycles=self.cycle_count,
            writers=self.writers_per_cycle,
        )
        if (self.executions, self.trace_scalar_bound) != budget or any(
            len(row.cycles) != self.cycle_count
            or any(len(c.write_modules) != self.writers_per_cycle for c in row.cycles)
            or row.ordinary_context
            != ScientificCollapseTraceContextV1(
                example_id=row.example_id,
                checkpoint_sha256=self.best_checkpoint_sha256,
                seed=self.run_seed,
                num_modules=self.num_modules,
                num_slots=self.num_slots,
                state_dim=self.state_dim,
                cycle_count=self.cycle_count,
                maximum_trace_scalars=self.policy.maximum_trace_scalars,
            )
            for row in self.rows
        ):
            raise ValueError("Full-test execution/cycle/writer coverage differs")
        groups = scientific_collapse_groups_v1(
            self.members, self.rows, modules=self.num_modules, slots=self.num_slots
        )
        summary = groups[0].summary
        assert summary is not None
        if self.groups != groups or self.source_20_4 != scientific_collapse_criterion_v1(
            purpose=self.purpose, summary=summary
        ):
            raise ValueError("Pooled counts/entropy or fixed source20.4 outcome differs")
        return self


class IrisScientificCollapseV1(_Strict):
    sha256: Digest
    payload: ScientificCollapsePayloadV1 = Field(repr=False)
