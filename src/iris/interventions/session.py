"""A fresh owned intervention session for one inference or teacher evaluation."""

from collections.abc import Callable
from typing import TypeVar

import torch

from iris.cognitive.outputs import ModuleProposal
from iris.interventions.phases import (
    BlockModuleRead,
    BlockModuleWrite,
    ForceModuleWinner,
    ScaleModuleBid,
    _validate_allocation,
    _validate_arbitration,
    _validate_proposal,
    _validate_read,
)
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
    validate_intervention_plan_v1,
)
from iris.interventions.state import (
    FreezeSlot,
    GaussianNoise,
    ReduceWorkspaceCapacity,
    ReplaceSlot,
    ShuffleSlots,
    SwapSlots,
    WorkspaceInterventionError,
    ZeroSlot,
    ZeroWorkspace,
    _StateOperation,
    validate_intervention_state,
)
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.state import WorkspaceState

WORKSPACE_INTERVENTION_SEMANTICS_V1 = "iris-workspace-intervention-semantics-v1"
_Result = TypeVar("_Result")


def _build_operation(spec: WorkspaceInterventionSpecV1) -> _StateOperation:
    cycles = spec.cycles
    match spec.kind:
        case "zero_workspace":
            return ZeroWorkspace(cycles=cycles)
        case "zero_slot" | "freeze_slot":
            assert spec.slot is not None
            cls = ZeroSlot if spec.kind == "zero_slot" else FreezeSlot
            return cls(spec.slot, cycles=cycles)
        case "swap_slots":
            assert spec.slot is not None and spec.other_slot is not None
            return SwapSlots(spec.slot, spec.other_slot, cycles=cycles)
        case "shuffle_slots":
            assert spec.seed is not None
            return ShuffleSlots(spec.seed, cycles=cycles)
        case "gaussian_noise":
            assert spec.seed is not None and spec.sigma is not None
            return GaussianNoise(spec.sigma, spec.seed, cycles=cycles)
        case "replace_slot":
            assert (
                spec.slot is not None and spec.values is not None and spec.source_module is not None
            )
            assert spec.confidence is not None and spec.ttl is not None
            return ReplaceSlot(
                spec.slot,
                spec.values,
                source_module=spec.source_module,
                confidence=spec.confidence,
                ttl=spec.ttl,
                cycles=cycles,
            )
        case "block_module_write" | "block_module_read" | "force_module_winner":
            assert spec.module is not None
            if spec.kind == "block_module_write":
                return BlockModuleWrite(spec.module, cycles=cycles)
            if spec.kind == "block_module_read":
                return BlockModuleRead(spec.module, cycles=cycles)
            return ForceModuleWinner(spec.module, cycles=cycles)
        case "scale_module_bid":
            assert spec.module is not None and spec.factor is not None
            return ScaleModuleBid(spec.module, spec.factor, cycles=cycles)
        case "reduce_workspace_capacity":
            assert spec.capacity is not None
            return ReduceWorkspaceCapacity(spec.capacity, cycles=cycles)
    raise WorkspaceInterventionError("Unknown intervention kind")


class WorkspaceInterventionSessionV1:
    """Fixed trusted operations; no user callbacks or temporary module hooks.

    Phase order: pre-read freeze restore/capacity clamp, blocked reads, scaled
    proposals, forced winners, capacity allocation, blocked writes then ordered
    state operations. Capacity is a constraint: the minimum active capacity is
    re-applied last, so later swaps/replacements/freezes cannot reopen disabled
    slots. A failure poisons this session; the next request builds a new one.
    """

    def __init__(
        self,
        plan: WorkspaceInterventionPlanV1,
        *,
        num_slots: int,
        num_modules: int,
        state_dim: int,
        cycles: int,
        winners_per_cycle: int,
    ) -> None:
        self._plan = validate_intervention_plan_v1(plan)
        self._plan.validate_dimensions(
            num_slots=num_slots,
            num_modules=num_modules,
            state_dim=state_dim,
            cycles=cycles,
            winners_per_cycle=winners_per_cycle,
        )
        self.plan_sha256 = intervention_plan_sha256_v1(self._plan)
        self._num_slots, self._num_modules, self._state_dim = num_slots, num_modules, state_dim
        self._cycles = cycles
        self._winners_per_cycle = winners_per_cycle
        self._operations = tuple(_build_operation(spec) for spec in self._plan.interventions)
        self._cycle = 1
        self._phase = 0
        self._failed = False
        self._finished = False
        self._geometry: tuple[int, torch.dtype, torch.device] | None = None

    def _run(self, cycle: int, phase: int, operation: Callable[[], _Result]) -> _Result:
        try:
            if (
                self._failed
                or self._finished
                or type(cycle) is not int
                or cycle != self._cycle
                or phase != self._phase
                or cycle > self._cycles
            ):
                raise WorkspaceInterventionError(
                    "Intervention session phase/cycle is invalid or closed"
                )
            result = operation()
            self._phase += 1
            if self._phase == 6:
                self._phase = 0
                self._cycle += 1
            return result
        except BaseException:
            self._failed = True
            raise

    def _state(self, state: WorkspaceState) -> None:
        validate_intervention_state(state)
        if tuple(state.slots.shape[1:]) != (self._num_slots, self._state_dim):
            raise WorkspaceInterventionError(
                "Workspace dimensions differ from the intervention graph"
            )
        if not bool((state.source_module.masked_select(state.occupancy) < self._num_modules).all()):
            raise WorkspaceInterventionError("Workspace source is outside the intervention graph")
        geometry = (state.slots.shape[0], state.slots.dtype, state.slots.device)
        if self._geometry is None:
            self._geometry = geometry
        elif geometry != self._geometry:
            raise WorkspaceInterventionError(
                "Workspace batch/dtype/device changed during intervention"
            )

    def _capacity(self, cycle: int) -> int:
        capacities = [
            spec.capacity
            for spec in self._plan.interventions
            if spec.kind == "reduce_workspace_capacity"
            and spec.active(cycle)
            and spec.capacity is not None
        ]
        return min(capacities, default=self._num_slots)

    def _phase_tensor(
        self, tensor: torch.Tensor, shape: tuple[int, ...], *, accumulation: bool = False
    ) -> None:
        if self._geometry is None:
            raise WorkspaceInterventionError(
                "Intervention workspace geometry has not been established"
            )
        batch, dtype, device = self._geometry
        if accumulation and dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        if (
            tuple(tensor.shape) != (batch, *shape)
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            raise WorkspaceInterventionError(
                "Phase tensor differs from the bound intervention graph"
            )

    def before_read(self, state: WorkspaceState, cycle: int) -> WorkspaceState:
        def apply() -> WorkspaceState:
            self._state(state)
            result = state
            for op in self._operations:
                if isinstance(op, FreezeSlot):
                    result = op.restore_before_read(result, cycle)
            capacity = self._capacity(cycle)
            if capacity < self._num_slots:
                result = ReduceWorkspaceCapacity(capacity).apply(result, cycle)
            self._state(result)
            return result

        return self._run(cycle, 0, apply)

    def after_read(self, read: WorkspaceReaderOutput, cycle: int) -> WorkspaceReaderOutput:
        def apply() -> WorkspaceReaderOutput:
            _validate_read(read)
            self._phase_tensor(read.workspace_reads, (self._num_modules, self._state_dim))
            self._phase_tensor(
                read.attention_weights, (self._num_modules, self._num_slots), accumulation=True
            )
            result = read
            for op in self._operations:
                if isinstance(op, BlockModuleRead):
                    result = op.apply_read(result, cycle)
            return result

        return self._run(cycle, 1, apply)

    def after_proposal(self, proposal: ModuleProposal, cycle: int) -> ModuleProposal:
        def apply() -> ModuleProposal:
            _validate_proposal(proposal)
            self._phase_tensor(proposal.content, (self._num_modules, self._state_dim))
            result = proposal
            for op in self._operations:
                if isinstance(op, ScaleModuleBid):
                    result = op.apply_proposal(result, cycle)
            return result

        return self._run(cycle, 2, apply)

    def after_arbitration(
        self, arbitration: WorkspaceArbiterOutput, cycle: int
    ) -> WorkspaceArbiterOutput:
        def apply() -> WorkspaceArbiterOutput:
            _, _, winners = _validate_arbitration(arbitration)
            self._phase_tensor(arbitration.scores, (self._num_modules,), accumulation=True)
            if winners != self._winners_per_cycle:
                raise WorkspaceInterventionError(
                    "Arbitration winner count differs from bound graph"
                )
            result = arbitration
            for op in self._operations:
                if isinstance(op, ForceModuleWinner):
                    result = op.apply_arbitration(result, cycle)
            return result

        return self._run(cycle, 3, apply)

    def after_allocation(
        self, state: WorkspaceState, allocation: SlotAllocatorOutput, cycle: int
    ) -> SlotAllocatorOutput:
        def apply() -> SlotAllocatorOutput:
            self._state(state)
            modules = _validate_allocation(allocation, state)
            if (
                modules != self._num_modules
                or allocation.module_indices.shape[1] != self._winners_per_cycle
            ):
                raise WorkspaceInterventionError("Allocation differs from bound intervention graph")
            capacity = self._capacity(cycle)
            if capacity == self._num_slots:
                return allocation
            return ReduceWorkspaceCapacity(capacity).apply_allocation(state, allocation, cycle)

        return self._run(cycle, 4, apply)

    def after_write(
        self,
        before: WorkspaceState,
        written: WorkspaceState,
        allocation: SlotAllocatorOutput,
        cycle: int,
    ) -> WorkspaceState:
        def apply() -> WorkspaceState:
            self._state(before)
            self._state(written)
            result = written
            for op in self._operations:
                if isinstance(op, BlockModuleWrite):
                    result = op.apply_write(before, result, allocation, cycle)
            for op in self._operations:
                if not isinstance(
                    op, (BlockModuleWrite, BlockModuleRead, ScaleModuleBid, ForceModuleWinner)
                ):
                    result = op.apply(result, cycle)
            capacity = self._capacity(cycle)
            if capacity < self._num_slots:
                result = ReduceWorkspaceCapacity(capacity).apply(result, cycle)
            self._state(result)
            return result

        return self._run(cycle, 5, apply)

    def finish(self) -> None:
        if self._failed or self._finished or self._phase != 0 or self._cycle != self._cycles + 1:
            self._failed = True
            raise WorkspaceInterventionError("Intervention session did not complete exactly once")
        self._finished = True
