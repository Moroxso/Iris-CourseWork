"""Source-bound stratified workspace diagnostics; not scientific acceptance."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import (
    WorkspaceCausalCaptureV1,
    WorkspaceCausalPolicyV1,
    WorkspaceCausalSummaryV1,
    summarize_workspace_causal_v1,
)
from iris.workspace.trace import (
    WorkspaceDiagnosticsV1,
    WorkspaceTraceV1,
    _scalar_bound,
    summarize_workspace_traces_v1,
)


class ScientificWorkspacePolicyV1(_Strict):
    """Fix before inspecting results; hashes cannot attest external preregistration."""

    selection: Literal["hash-ranked-task-difficulty-strata-v1"] = (
        "hash-ranked-task-difficulty-strata-v1"
    )
    selection_seed: int = Field(ge=0, le=2**63 - 1)
    examples_per_stratum: int = Field(default=4, ge=1, le=64)
    trace_mode: Literal["metadata", "full"] = "metadata"
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_trace_scalars: int = Field(default=2_000_000, ge=1, le=16_000_000)
    maximum_executions: int = Field(default=1024, ge=1, le=8192)
    causal: WorkspaceCausalPolicyV1 | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)
    numerical_tolerance: float = Field(default=1e-6, ge=0, lt=1)


class ScientificWorkspaceMemberV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    teacher_prompt_sha256: Digest
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)


def select_workspace_members_v1(
    members: tuple[ScientificWorkspaceMemberV1, ...], policy: ScientificWorkspacePolicyV1
) -> tuple[ScientificWorkspaceMemberV1, ...]:
    """Rank only input identity inside fixed task/difficulty strata, never outcomes."""
    policy = ScientificWorkspacePolicyV1.model_validate(policy)
    members = tuple(ScientificWorkspaceMemberV1.model_validate(m) for m in members)
    if not 1 <= len(members) <= 5000 or any(
        len({getattr(m, field) for m in members}) != len(members)
        for field in ("example_id", "prompt_sha256", "teacher_prompt_sha256")
    ):
        raise ValueError("Unique held-out examples/prompts required")

    def rank(member: ScientificWorkspaceMemberV1) -> tuple[str, str]:
        return (
            domain_sha256(
                "iris/scientific-workspace-selection/v1",
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

    return tuple(
        member
        for task in TASKS
        for difficulty in (1, 2, 3)
        for member in sorted(
            (m for m in members if (m.task_type, m.difficulty) == (task, difficulty)),
            key=rank,
        )[: policy.examples_per_stratum]
    )


def workspace_execution_budget_v1(
    policy: ScientificWorkspacePolicyV1,
    *,
    examples: int,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    writers: int,
) -> tuple[int, int]:
    """Conservative scalar-volume/execution preflight, including transient reruns."""
    for value in (examples, modules, slots, width, cycles, writers):
        if type(value) is not int or value < 1:
            raise ValueError("Exact positive diagnostic geometry required")
    if writers > min(modules, slots):
        raise ValueError("Writer count exceeds graph")
    scalars = _scalar_bound(modules, slots, width, cycles, policy.trace_mode == "full")
    calls = 1
    if scalars > policy.maximum_trace_scalars:
        raise ValueError("Per-trace scalar budget exceeded")
    if policy.causal is not None:
        extra = 1 + (cycles - 1) * (writers + 1)
        full = _scalar_bound(modules, slots, width, cycles, True)
        if (
            cycles < 2
            or not 2 <= modules <= 64
            or extra > policy.causal.maximum_executions
            or full > policy.causal.maximum_trace_scalars
        ):
            raise ValueError("Causal next-cycle geometry/budget is incompatible")
        calls += extra
        scalars += extra * full
    calls, scalars = examples * calls, examples * scalars
    if calls > policy.maximum_executions or scalars > policy.maximum_total_trace_scalars:
        raise ValueError("Total diagnostic execution/scalar budget exceeded")
    return calls, scalars


class ScientificWorkspaceRowV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    target_sha256: Digest
    target_token_count: int = Field(ge=1)
    control_nll_sum: float = Field(ge=0)
    trace: WorkspaceTraceV1
    causal: WorkspaceCausalCaptureV1 | None = None


class ScientificWorkspaceGroupV1(_Strict):
    task_type: SyntheticTask | None
    difficulty: int | None = Field(ge=1, le=3)
    available_examples: int = Field(ge=0)
    selected_examples: int = Field(ge=0)
    diagnostics: WorkspaceDiagnosticsV1 | None


def scientific_workspace_groups_v1(
    members: tuple[ScientificWorkspaceMemberV1, ...],
    rows: tuple[ScientificWorkspaceRowV1, ...],
) -> tuple[ScientificWorkspaceGroupV1, ...]:
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
        traces = tuple(row.trace for row in rows if row.example_id in ids)
        groups.append(
            ScientificWorkspaceGroupV1(
                task_type=task,
                difficulty=level,
                available_examples=len(ids),
                selected_examples=len(traces),
                diagnostics=summarize_workspace_traces_v1(traces) if traces else None,
            )
        )
    return tuple(groups)


class ScientificWorkspacePayloadV1(_Strict):
    kind: Literal["iris-scientific-workspace-v1"] = "iris-scientific-workspace-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    aggregation_scope: Literal["stratified_subset_not_full_corpus_estimate"] = (
        "stratified_subset_not_full_corpus_estimate"
    )
    policy: ScientificWorkspacePolicyV1
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
    rows: tuple[ScientificWorkspaceRowV1, ...] = Field(min_length=1, max_length=768)
    executions: int = Field(ge=1)
    trace_scalar_bound: int = Field(ge=1)
    groups: tuple[ScientificWorkspaceGroupV1, ...] = Field(min_length=20, max_length=20)
    causal_summary: WorkspaceCausalSummaryV1 | None = None

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        selected = select_workspace_members_v1(self.members, self.policy)
        if self.selected_step > self.terminal_step or tuple(
            m.example_id for m in selected
        ) != tuple(row.example_id for row in self.rows):
            raise ValueError("Selected diagnostic subset/step differs from declared design")
        if self.purpose == "scientific" and tuple(
            sum(m.task_type == task for m in self.members) for task in TASKS
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific membership must cover the complete 5000-row test corpus")
        first = self.rows[0].trace
        writers = len(first.cycles[0].write_modules)
        calls, scalars = workspace_execution_budget_v1(
            self.policy,
            examples=len(self.rows),
            modules=first.num_modules,
            slots=first.num_slots,
            width=first.state_dim,
            cycles=first.cycle_count,
            writers=writers,
        )
        if (self.executions, self.trace_scalar_bound) != (calls, scalars):
            raise ValueError("Diagnostic execution/scalar volume differs")
        captures = []
        for member, row in zip(selected, self.rows, strict=True):
            trace, causal = row.trace, row.causal
            if (
                trace.options.example_id != row.example_id
                or trace.options.checkpoint_id != self.best_checkpoint_sha256
                or trace.options.seed != self.run_seed
                or trace.options.mode != self.policy.trace_mode
                or trace.options.maximum_scalars != self.policy.maximum_trace_scalars
                or trace.intervention_plan_sha256 is not None
                or any(len(c.write_modules) != writers for c in trace.cycles)
                or (causal is None) != (self.policy.causal is None)
            ):
                raise ValueError("Diagnostic trace context differs from source/design")
            if causal is not None:
                if (
                    causal.example_id != row.example_id
                    or causal.prompt_sha256 != member.teacher_prompt_sha256
                    or causal.teacher_graph_sha256 != self.teacher_graph_sha256
                    or causal.sidecar_state_sha256 != self.teacher_sidecar_sha256
                    or causal.seed != self.run_seed
                    or causal.policy != self.policy.causal
                    or (causal.num_modules, causal.num_slots, causal.cycle_count)
                    != (trace.num_modules, trace.num_slots, trace.cycle_count)
                    or causal.writers_per_cycle != writers
                ):
                    raise ValueError("Causal capture context differs from bound trace/source")
                captures.append(causal)
        expected_causal = (
            summarize_workspace_causal_v1(
                tuple(captures),
                alpha=self.policy.alpha,
                numerical_tolerance=self.policy.numerical_tolerance,
            )
            if captures
            else None
        )
        if (
            self.groups != scientific_workspace_groups_v1(self.members, self.rows)
            or self.causal_summary != expected_causal
        ):
            raise ValueError("Diagnostic aggregates differ from bound raw measurements")
        return self


class IrisScientificWorkspaceV1(_Strict):
    sha256: Digest
    payload: ScientificWorkspacePayloadV1 = Field(repr=False)
