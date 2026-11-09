"""Bounded detached section-21 diagnostics, never auxiliary-loss graphs or causal proof."""

from collections.abc import Sequence
from math import exp, fsum, hypot, log
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.cognitive.outputs import ModuleProposal
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.state import WorkspaceState

Nonnegative = Annotated[float, Field(ge=0)]
Counter = Annotated[int, Field(ge=0)]
Matrix = tuple[tuple[float, ...], ...]


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class WorkspaceTraceOptionsV1(_Strict):
    """Caller labels are context, not authenticated checkpoint/data identities."""

    mode: Literal["none", "metadata", "full"] = "none"
    example_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")
    checkpoint_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")
    seed: int = Field(ge=0, le=2**63 - 1)
    maximum_scalars: int = Field(default=65536, ge=1, le=2_000_000)


class WorkspaceFullCycleV1(_Strict):
    """Diagnostic vectors only; no prompt, targets, logits or autograd tensors."""

    read_workspace_slots: Matrix
    workspace_slots: Matrix
    module_states: Matrix
    workspace_reads: Matrix


class WorkspaceCycleMetadataV1(_Strict):
    cycle: int = Field(ge=1)
    module_bids: tuple[float, ...]
    module_confidence: tuple[Annotated[float, Field(ge=0, le=1)], ...]
    selected_modules: tuple[Counter, ...]
    selected_slots: tuple[Counter, ...]
    write_modules: tuple[Counter, ...]
    overwritten_slots: tuple[Counter, ...]
    slot_ages: tuple[Counter, ...]
    slot_ttl: tuple[Nonnegative, ...]
    slot_norms: tuple[Nonnegative, ...]
    slot_sources: tuple[Annotated[int, Field(ge=-1)], ...]
    slot_occupancy: tuple[bool, ...]
    slot_write_counts: tuple[Counter, ...]
    module_state_norms: tuple[Nonnegative, ...]
    workspace_cosine_matrix: tuple[tuple[Annotated[float, Field(ge=-1, le=1)], ...], ...]
    read_slot_sources: tuple[Annotated[int, Field(ge=-1)], ...]
    read_slot_occupancy: tuple[bool, ...]
    read_attention: tuple[tuple[Annotated[float, Field(ge=0, le=1)], ...], ...]
    full: WorkspaceFullCycleV1 | None = None


def _scalar_bound(modules: int, slots: int, width: int, cycles: int, full: bool) -> int:
    # Conservative preflight bound for all numeric/bool cycle fields, not file bytes.
    return cycles * (
        1
        + 10 * modules
        + 20 * slots
        + slots * slots
        + modules * slots
        + (2 * (modules + slots) * width if full else 0)
    )


class WorkspaceTraceV1(_Strict):
    kind: Literal["iris-workspace-diagnostic-trace-v1"] = "iris-workspace-diagnostic-trace-v1"
    semantics: Literal["post-intervention-read-and-post-write-v1"] = (
        "post-intervention-read-and-post-write-v1"
    )
    options: WorkspaceTraceOptionsV1
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    intervention_plan_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    cycles: tuple[WorkspaceCycleMetadataV1, ...]

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        m, k, d = self.num_modules, self.num_slots, self.state_dim
        if self.options.mode == "none" or len(self.cycles) != self.cycle_count:
            raise ValueError("A diagnostic trace must contain every requested cycle")
        full = self.options.mode == "full"
        if _scalar_bound(m, k, d, self.cycle_count, full) > self.options.maximum_scalars:
            raise ValueError("Diagnostic trace exceeds its scalar budget")
        for number, row in enumerate(self.cycles, 1):
            if row.cycle != number:
                raise ValueError("Trace cycles must be ordered and consecutive")
            if any(
                len(vector) != m
                for vector in (
                    row.module_bids,
                    row.module_confidence,
                    row.module_state_norms,
                )
            ) or any(
                len(vector) != k
                for vector in (
                    row.slot_ages,
                    row.slot_ttl,
                    row.slot_norms,
                    row.slot_sources,
                    row.slot_occupancy,
                    row.slot_write_counts,
                    row.read_slot_sources,
                    row.read_slot_occupancy,
                )
            ):
                raise ValueError("Trace vector dimensions differ from graph")
            if (
                len(row.workspace_cosine_matrix) != k
                or any(len(vector) != k for vector in row.workspace_cosine_matrix)
                or len(row.read_attention) != m
                or any(len(vector) != k for vector in row.read_attention)
            ):
                raise ValueError("Trace matrix dimensions differ from graph")
            for indices, limit in (
                (row.selected_modules, m),
                (row.write_modules, m),
                (row.selected_slots, k),
                (row.overwritten_slots, k),
            ):
                if len(set(indices)) != len(indices) or any(i >= limit for i in indices):
                    raise ValueError("Trace selections must be unique in-domain indices")
            if (
                len(row.write_modules) != len(row.selected_slots)
                or set(row.write_modules) != set(row.selected_modules)
                or not set(row.overwritten_slots).issubset(row.selected_slots)
            ):
                raise ValueError("Trace winner/write correspondence differs")
            for sources, occupied in (
                (row.slot_sources, row.slot_occupancy),
                (row.read_slot_sources, row.read_slot_occupancy),
            ):
                if any(
                    (not active and source != -1) or (active and not 0 <= source < m)
                    for source, active in zip(sources, occupied, strict=True)
                ):
                    raise ValueError("Trace source ownership differs from occupancy")
            for j, active in enumerate(row.slot_occupancy):
                if not active and any(
                    (row.slot_ages[j], row.slot_ttl[j], row.slot_norms[j], row.slot_write_counts[j])
                ):
                    raise ValueError("Empty traced slots must have canonical metadata")
                for h, cosine in enumerate(row.workspace_cosine_matrix[j]):
                    if (row.slot_norms[j] == 0 or row.slot_norms[h] == 0) and cosine != 0:
                        raise ValueError("Zero-norm cosine pairs must use the zero convention")
                    if abs(cosine - row.workspace_cosine_matrix[h][j]) > 1e-12:
                        raise ValueError("Workspace cosine matrix must be symmetric")
                    if j == h and row.slot_norms[j] > 0 and abs(cosine - 1) > 1e-12:
                        raise ValueError("Nonzero slot self-cosine must be one")
            for weights in row.read_attention:
                total = fsum(weights)
                if (total != 0 and abs(total - 1) > 1e-5) or any(
                    value != 0 and not row.read_slot_occupancy[j] for j, value in enumerate(weights)
                ):
                    raise ValueError("Read attention must be occupied unit-mass or blocked zero")
            if full != (row.full is not None):
                raise ValueError("Full vectors must appear only in full mode")
            if row.full is not None:
                for values, height in (
                    (row.full.read_workspace_slots, k),
                    (row.full.workspace_slots, k),
                    (row.full.module_states, m),
                    (row.full.workspace_reads, m),
                ):
                    if len(values) != height or any(len(v) != d for v in values):
                        raise ValueError("Full trace vector geometry differs from graph")
        return self


def _cpu(value: torch.Tensor, shape: tuple[int, ...], *, floating: bool = True) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or tuple(value.shape) != shape
        or value.device.type not in {"cpu", "cuda"}
        or (floating and not value.is_floating_point())
    ):
        raise ValueError("Invalid diagnostic tensor boundary")
    result = value.detach().to(device="cpu", dtype=torch.float64 if floating else value.dtype)
    if floating and not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite diagnostic tensor")
    return result


def _vector(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(v) for v in value.tolist())


def _matrix(value: torch.Tensor) -> Matrix:
    return tuple(_vector(row) for row in value)


def _norm_and_unit_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale finite CPU float64 rows without clamping away subnormal magnitudes."""
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / torch.where(scale > 0, scale, torch.ones_like(scale))
    lengths = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    unit = scaled / torch.where(lengths > 0, lengths, torch.ones_like(lengths))
    return (scale * lengths).squeeze(-1), unit


def _mean(values: Sequence[float]) -> float:
    """Avoid an overflowing intermediate sum or underflow from dividing tiny terms."""
    scale = max(abs(value) for value in values)
    return (fsum(value / scale for value in values) / len(values)) * scale if scale else 0.0


class WorkspaceTraceRecorderV1:
    """One call-owned recorder; it never retains model tensors or mutates the graph."""

    def __init__(
        self,
        options: WorkspaceTraceOptionsV1 | None,
        *,
        num_modules: int,
        num_slots: int,
        state_dim: int,
        cycles: int,
        intervention_plan_sha256: str | None = None,
    ) -> None:
        if options is not None and type(options) is not WorkspaceTraceOptionsV1:
            raise TypeError("Trace options require their exact validated type")
        self.options = None if options is None else WorkspaceTraceOptionsV1.model_validate(options)
        for value in (num_modules, num_slots, state_dim, cycles):
            if type(value) is not int or value < 1:
                raise ValueError("Trace dimensions must be exact positive integers")
        self.m, self.k, self.d, self.count = num_modules, num_slots, state_dim, cycles
        self.plan = intervention_plan_sha256
        self.rows: list[WorkspaceCycleMetadataV1] = []
        self.finished = False
        if (
            self.enabled
            and self.options is not None
            and _scalar_bound(self.m, self.k, self.d, self.count, self.options.mode == "full")
            > self.options.maximum_scalars
        ):
            raise ValueError("Diagnostic trace exceeds scalar budget before model execution")

    @property
    def enabled(self) -> bool:
        return self.options is not None and self.options.mode != "none"

    @torch.no_grad()
    def record(
        self,
        *,
        cycle: int,
        read_state: WorkspaceState,
        before_write: WorkspaceState,
        workspace: WorkspaceState,
        reads: WorkspaceReaderOutput,
        module_states: torch.Tensor,
        proposal: ModuleProposal,
        arbitration: WorkspaceArbiterOutput,
        allocation: SlotAllocatorOutput,
    ) -> None:
        if (
            self.finished
            or type(cycle) is not int
            or cycle != len(self.rows) + 1
            or cycle > self.count
        ):
            raise ValueError("Diagnostic recorder requires exactly ordered cycles")
        if not self.enabled:
            raise ValueError("Disabled diagnostic recorder must not be called")
        m, k, d = self.m, self.k, self.d
        for state in (read_state, before_write, workspace):
            if type(state) is not WorkspaceState:
                raise TypeError("Trace requires exact workspace states")
            _cpu(state.slots, (1, k, d))
            for name, dtype in (
                ("age", torch.int64),
                ("occupancy", torch.bool),
                ("source_module", torch.int64),
                ("write_count", torch.int64),
            ):
                if _cpu(getattr(state, name), (1, k), floating=False).dtype != dtype:
                    raise ValueError("Trace workspace metadata dtype differs")
            _cpu(state.ttl, (1, k))
        slots = _cpu(workspace.slots, (1, k, d))[0]
        states = _cpu(module_states, (1, m, d))[0]
        mask = _cpu(allocation.assignment_mask, (1, m, k), floating=False)[0]
        winners = _cpu(arbitration.winner_mask, (1, m), floating=False)[0]
        if (
            mask.dtype != torch.bool
            or winners.dtype != torch.bool
            or bool((mask.sum(0) > 1).any())
            or bool((mask.sum(1) > 1).any())
            or not torch.equal(mask.any(1), winners)
        ):
            raise ValueError("Diagnostic allocation must map winners to unique slots")
        module_indices, slot_indices = mask.nonzero(as_tuple=True)
        # Stable float64 cosine; empty/zero-norm slots have explicit zero pairs.
        norms, normalized = _norm_and_unit_rows(slots)
        module_norms, _ = _norm_and_unit_rows(states)
        cosine = (normalized @ normalized.T).clamp(-1, 1)
        full = None
        assert self.options is not None
        if self.options.mode == "full":
            full = WorkspaceFullCycleV1(
                read_workspace_slots=_matrix(_cpu(read_state.slots, (1, k, d))[0]),
                workspace_slots=_matrix(slots),
                module_states=_matrix(states),
                workspace_reads=_matrix(_cpu(reads.workspace_reads, (1, m, d))[0]),
            )
        row = WorkspaceCycleMetadataV1(
            cycle=cycle,
            module_bids=_vector(_cpu(proposal.bid, (1, m))[0]),
            module_confidence=_vector(_cpu(proposal.confidence, (1, m))[0]),
            selected_modules=tuple(int(v) for v in winners.nonzero().flatten().tolist()),
            selected_slots=tuple(int(v) for v in slot_indices.tolist()),
            write_modules=tuple(int(v) for v in module_indices.tolist()),
            overwritten_slots=tuple(
                int(v) for v in slot_indices.tolist() if bool(before_write.occupancy[0, v])
            ),
            slot_ages=tuple(int(v) for v in workspace.age[0].tolist()),
            slot_ttl=_vector(_cpu(workspace.ttl, (1, k))[0]),
            slot_norms=_vector(norms),
            slot_sources=tuple(int(v) for v in workspace.source_module[0].tolist()),
            slot_occupancy=tuple(bool(v) for v in workspace.occupancy[0].tolist()),
            slot_write_counts=tuple(int(v) for v in workspace.write_count[0].tolist()),
            module_state_norms=_vector(module_norms),
            workspace_cosine_matrix=_matrix(cosine),
            read_slot_sources=tuple(int(v) for v in read_state.source_module[0].tolist()),
            read_slot_occupancy=tuple(bool(v) for v in read_state.occupancy[0].tolist()),
            read_attention=_matrix(_cpu(reads.attention_weights, (1, m, k))[0]),
            full=full,
        )
        self.rows.append(row)

    def finish(self) -> WorkspaceTraceV1 | None:
        if self.finished:
            raise ValueError("Diagnostic recorder already completed")
        self.finished = True
        if not self.enabled:
            return None
        assert self.options is not None
        return WorkspaceTraceV1(
            options=self.options,
            num_modules=self.m,
            num_slots=self.k,
            state_dim=self.d,
            cycle_count=self.count,
            intervention_plan_sha256=self.plan,
            cycles=tuple(self.rows),
        )


class WorkspaceDiagnosticsV1(_Strict):
    """Hard allocation usage and descriptive attention; not causal acceptance."""

    kind: Literal["iris-workspace-diagnostics-v1"] = "iris-workspace-diagnostics-v1"
    write_scope: Literal["allocated_before_post_write_interventions"] = (
        "allocated_before_post_write_interventions"
    )
    causal_influence: Literal["not_measured"] = "not_measured"
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    examples: int
    cycles: int
    used_slots: int
    module_win_counts: tuple[int, ...]
    slot_allocation_counts: tuple[int, ...]
    module_win_rates: tuple[float, ...]
    module_entropy_nats: float
    slot_entropy_nats: float
    effective_active_modules: float
    effective_active_slots: float
    allocated_writes: int
    overwrites: int
    overwrite_rate: float | None
    occupied_slot_observations: int
    mean_ttl: float | None
    mean_write_count: float | None
    mean_workspace_norm: float
    cosine_pairs: int
    mean_slot_cosine_similarity: float | None
    read_opportunities: int
    active_reads: int
    cross_module_reads: int
    cross_module_read_rate: float | None
    cross_module_attention_mass: float
    mean_cross_module_attention_mass: float | None


def _entropy(counts: list[int]) -> tuple[float, float]:
    total = sum(counts)
    if not total:
        return 0.0, 0.0
    entropy = -fsum((count / total) * log(count / total) for count in counts if count)
    return entropy, min(float(len(counts)), exp(entropy))


def summarize_workspace_traces_v1(traces: tuple[WorkspaceTraceV1, ...]) -> WorkspaceDiagnosticsV1:
    """Pool raw counts, not per-request entropies. No simple/complex or p-value claim."""
    if type(traces) is not tuple or not traces:
        raise ValueError("Diagnostics need nonempty exact traces")
    checked = tuple(WorkspaceTraceV1.model_validate(trace) for trace in traces)
    first = checked[0]
    if len({trace.options.example_id for trace in checked}) != len(checked):
        raise ValueError("Duplicate example in diagnostics")
    if any(
        (
            t.num_modules,
            t.num_slots,
            t.state_dim,
            t.cycle_count,
            t.options.checkpoint_id,
            t.options.seed,
            t.intervention_plan_sha256,
        )
        != (
            first.num_modules,
            first.num_slots,
            first.state_dim,
            first.cycle_count,
            first.options.checkpoint_id,
            first.options.seed,
            first.intervention_plan_sha256,
        )
        for t in checked
    ):
        raise ValueError("Cannot pool different graph/checkpoint/seed/intervention contexts")
    modules, slots = [0] * first.num_modules, [0] * first.num_slots
    ttl: list[float] = []
    writes: list[int] = []
    norms: list[float] = []
    cosines: list[float] = []
    cross_mass: list[float] = []
    active_reads = cross_reads = overwritten = 0
    for trace in checked:
        for row in trace.cycles:
            for module in row.selected_modules:
                modules[module] += 1
            for slot in row.selected_slots:
                slots[slot] += 1
            overwritten += len(row.overwritten_slots)
            for j, occupied in enumerate(row.slot_occupancy):
                if occupied:
                    ttl.append(row.slot_ttl[j])
                    writes.append(row.slot_write_counts[j])
                if row.slot_norms[j] > 0:
                    for h in range(j + 1, first.num_slots):
                        if row.slot_norms[h] > 0:
                            cosines.append(row.workspace_cosine_matrix[j][h])
            norms.append(hypot(*row.slot_norms))
            for module, weights in enumerate(row.read_attention):
                if fsum(weights) > 0:
                    active_reads += 1
                    mass = fsum(
                        weight
                        for j, weight in enumerate(weights)
                        if row.read_slot_sources[j] != module
                    )
                    cross_reads += int(mass > 0)
                    cross_mass.append(mass)
    cycle_count = sum(t.cycle_count for t in checked)
    module_entropy, active_modules = _entropy(modules)
    slot_entropy, active_slots = _entropy(slots)
    count = sum(slots)
    mass_sum = fsum(cross_mass)
    return WorkspaceDiagnosticsV1(
        examples=len(checked),
        cycles=cycle_count,
        used_slots=sum(value > 0 for value in slots),
        module_win_counts=tuple(modules),
        slot_allocation_counts=tuple(slots),
        module_win_rates=tuple(value / cycle_count for value in modules),
        module_entropy_nats=module_entropy,
        slot_entropy_nats=slot_entropy,
        effective_active_modules=active_modules,
        effective_active_slots=active_slots,
        allocated_writes=count,
        overwrites=overwritten,
        overwrite_rate=overwritten / count if count else None,
        occupied_slot_observations=len(ttl),
        mean_ttl=_mean(ttl) if ttl else None,
        mean_write_count=_mean(writes) if writes else None,
        mean_workspace_norm=_mean(norms),
        cosine_pairs=len(cosines),
        mean_slot_cosine_similarity=_mean(cosines) if cosines else None,
        read_opportunities=cycle_count * first.num_modules,
        active_reads=active_reads,
        cross_module_reads=cross_reads,
        cross_module_read_rate=cross_reads / active_reads if active_reads else None,
        cross_module_attention_mass=mass_sum,
        mean_cross_module_attention_mass=mass_sum / active_reads if active_reads else None,
    )
