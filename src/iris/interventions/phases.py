"""Owned module-phase transformations for the trusted intervention session.

The required apply(state,cycle) interface preserves the state for these controls.
Their actual causal effects occur only through the explicit read/proposal/
arbitration/write methods. The graph owner must call the appropriate method;
calling apply alone does NOT implement a module intervention.
"""

from dataclasses import fields

import torch

from iris.cognitive.outputs import ModuleProposal
from iris.interventions.policy import WorkspaceInterventionSpecV1
from iris.interventions.state import (
    WorkspaceInterventionError,
    _clone,
    _StateOperation,
    validate_intervention_state,
)
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.state import WorkspaceState


def _tensor(
    value: object,
    name: str,
    *,
    ndim: int,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(dimension < 1 for dimension in value.shape)
        or (shape is not None and tuple(value.shape) != shape)
        or (dtype is not None and value.dtype != dtype)
        or (device is not None and value.device != device)
    ):
        raise WorkspaceInterventionError(f"Invalid {name} tensor shape/dtype/device")
    if dtype is None and value.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise WorkspaceInterventionError(f"{name} must be a supported floating tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise WorkspaceInterventionError(f"{name} must be finite")
    return value


def _validate_proposal(proposal: ModuleProposal) -> tuple[int, int, int]:
    if type(proposal) is not ModuleProposal:
        raise WorkspaceInterventionError("An exact ModuleProposal is required")
    content = _tensor(proposal.content, "proposal content", ndim=3)
    batch, modules, width = content.shape
    for name in ("key", "bid", "confidence", "ttl"):
        shape = (batch, modules, width) if name == "key" else (batch, modules)
        _tensor(
            getattr(proposal, name),
            f"proposal {name}",
            ndim=len(shape),
            shape=shape,
            dtype=content.dtype,
            device=content.device,
        )
    if not bool(((proposal.confidence >= 0) & (proposal.confidence <= 1)).all()):
        raise WorkspaceInterventionError("Proposal confidence must lie in [0,1]")
    if not bool((proposal.ttl >= 1).all()):
        raise WorkspaceInterventionError("Proposal TTL must be at least one")
    return batch, modules, width


def _validate_read(read: WorkspaceReaderOutput) -> tuple[int, int]:
    if type(read) is not WorkspaceReaderOutput:
        raise WorkspaceInterventionError("An exact WorkspaceReaderOutput is required")
    values = _tensor(read.workspace_reads, "workspace reads", ndim=3)
    batch, modules, _ = values.shape
    accumulation = (
        torch.float32 if values.dtype in {torch.float16, torch.bfloat16} else values.dtype
    )
    attention = _tensor(
        read.attention_weights, "read attention", ndim=3, dtype=accumulation, device=values.device
    )
    if tuple(attention.shape[:2]) != (batch, modules):
        raise WorkspaceInterventionError("Read attention batch/module dimensions do not match")
    sums = attention.sum(dim=-1)
    no_read = sums == 0
    if (
        not bool(((attention >= 0) & (attention <= 1)).all())
        or not torch.allclose(sums, (~no_read).to(sums.dtype), rtol=1e-4, atol=1e-6)
        or bool((values.masked_select(no_read.unsqueeze(-1)) != 0).any())
    ):
        raise WorkspaceInterventionError("Read attention must be normalized or an exact zero read")
    return batch, modules


def _validate_arbitration(arbitration: WorkspaceArbiterOutput) -> tuple[int, int, int]:
    if type(arbitration) is not WorkspaceArbiterOutput:
        raise WorkspaceInterventionError("An exact WorkspaceArbiterOutput is required")
    scores = _tensor(arbitration.scores, "arbitration scores", ndim=2)
    if scores.dtype not in {torch.float32, torch.float64}:
        raise WorkspaceInterventionError(
            "Arbitration must use its float32/float64 accumulation dtype"
        )
    batch, modules = scores.shape
    for name in ("novelty", "relevance", "soft_probabilities", "selection_weights"):
        _tensor(
            getattr(arbitration, name),
            f"arbitration {name}",
            ndim=2,
            shape=(batch, modules),
            dtype=scores.dtype,
            device=scores.device,
        )
    mask = _tensor(
        arbitration.winner_mask,
        "winner mask",
        ndim=2,
        shape=(batch, modules),
        dtype=torch.bool,
        device=scores.device,
    )
    indices = _tensor(
        arbitration.winner_indices,
        "winner indices",
        ndim=2,
        dtype=torch.int64,
        device=scores.device,
    )
    if indices.shape[0] != batch or indices.shape[1] > modules:
        raise WorkspaceInterventionError("Invalid arbitration winner dimensions")
    winners = indices.shape[1]
    expected_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :winners]
    if not torch.equal(indices, expected_indices):
        raise WorkspaceInterventionError("Winner indices must follow deterministic score order")
    expected_mask = torch.zeros_like(mask).scatter(1, indices, True)
    if not torch.equal(mask, expected_mask) or not torch.equal(
        arbitration.selection_weights, mask.to(scores.dtype)
    ):
        raise WorkspaceInterventionError("Winner mask/weights must match hard ordered selection")
    if not torch.allclose(
        arbitration.soft_probabilities, torch.softmax(scores, dim=-1), rtol=1e-4, atol=1e-6
    ) or not bool(
        ((arbitration.soft_probabilities >= 0) & (arbitration.soft_probabilities <= 1)).all()
    ):
        raise WorkspaceInterventionError("Arbitration probabilities must match its scores")
    return batch, modules, winners


def _validate_allocation(allocation: SlotAllocatorOutput, state: WorkspaceState) -> int:
    if type(allocation) is not SlotAllocatorOutput:
        raise WorkspaceInterventionError("An exact SlotAllocatorOutput is required")
    batch, slots, _ = state.slots.shape
    dtype = (
        torch.float32 if state.slots.dtype in {torch.float16, torch.bfloat16} else state.slots.dtype
    )
    scores = _tensor(
        allocation.allocation_scores,
        "allocation scores",
        ndim=3,
        dtype=dtype,
        device=state.slots.device,
    )
    modules = scores.shape[1]
    if scores.shape[0] != batch or scores.shape[2] != slots:
        raise WorkspaceInterventionError("Allocation dimensions do not match workspace")
    _tensor(
        allocation.retention_scores,
        "retention scores",
        ndim=2,
        shape=(batch, slots),
        dtype=dtype,
        device=scores.device,
    )
    for name in ("assignment_probabilities", "assignment_weights"):
        _tensor(
            getattr(allocation, name),
            name,
            ndim=3,
            shape=(batch, modules, slots),
            dtype=dtype,
            device=scores.device,
        )
    mask = _tensor(
        allocation.assignment_mask,
        "assignment mask",
        ndim=3,
        shape=(batch, modules, slots),
        dtype=torch.bool,
        device=scores.device,
    )
    target = _tensor(
        allocation.target_mask,
        "target mask",
        ndim=2,
        shape=(batch, slots),
        dtype=torch.bool,
        device=scores.device,
    )
    indices = _tensor(
        allocation.module_indices,
        "allocated module indices",
        ndim=2,
        dtype=torch.int64,
        device=scores.device,
    )
    winners = indices.shape[1]
    if indices.shape[0] != batch or winners > min(slots, modules):
        raise WorkspaceInterventionError("Invalid allocation winner dimensions")
    slot_indices = _tensor(
        allocation.slot_indices,
        "allocated slot indices",
        ndim=2,
        shape=(batch, winners),
        dtype=torch.int64,
        device=scores.device,
    )
    if not bool(((indices >= 0) & (indices < modules)).all()) or not bool(
        ((slot_indices >= 0) & (slot_indices < slots)).all()
    ):
        raise WorkspaceInterventionError("Allocation indices are outside their domains")
    expected = torch.zeros_like(mask)
    expected[torch.arange(batch, device=scores.device).unsqueeze(1), indices, slot_indices] = True
    if (
        not torch.equal(mask, expected)
        or not bool((mask.sum(dim=(1, 2)) == winners).all())
        or bool((mask.sum(dim=1) > 1).any())
        or bool((mask.sum(dim=-1) > 1).any())
        or not torch.equal(target, mask.any(dim=1))
        or not torch.equal(allocation.assignment_weights, mask.to(dtype))
    ):
        raise WorkspaceInterventionError("Allocation must describe one-to-one hard assignments")
    probabilities = allocation.assignment_probabilities
    if (
        not bool(((probabilities >= 0) & (probabilities <= 1)).all())
        or not torch.allclose(
            probabilities.sum(dim=-1), mask.any(dim=-1).to(dtype), rtol=1e-4, atol=1e-6
        )
        or bool((probabilities.masked_select(~target.unsqueeze(1)) != 0).any())
    ):
        raise WorkspaceInterventionError(
            "Allocation probabilities must stay inside assigned targets"
        )
    return modules


class _ModuleOperation(_StateOperation):
    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        # Bare state cannot express read/bid/winner/write-route controls.
        return _clone(state)

    def _module(self, count: int, cycle: int) -> tuple[int, bool]:
        spec = self.spec
        active = spec.active(cycle)
        assert spec.module is not None
        if spec.module >= count:
            raise WorkspaceInterventionError("Intervention module is outside the actual phase")
        return spec.module, active


class BlockModuleRead(_ModuleOperation):
    """Zero one module's read vector AND its attention diagnostic after reading."""

    _kind = "block_module_read"

    def __init__(self, module: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, module=module, cycles=cycles))

    def apply_read(self, read: WorkspaceReaderOutput, cycle: int) -> WorkspaceReaderOutput:
        _, count = _validate_read(read)
        module, active = self._module(count, cycle)
        output = WorkspaceReaderOutput(read.workspace_reads.clone(), read.attention_weights.clone())
        if active:
            output.workspace_reads[:, module] = 0
            output.attention_weights[:, module] = 0
        _validate_read(output)
        return output


class ScaleModuleBid(_ModuleOperation):
    """Scale only the declared module's bid before ordinary arbitration."""

    _kind = "scale_module_bid"

    def __init__(
        self, module: int, factor: float, *, cycles: tuple[int, ...] | None = None
    ) -> None:
        super().__init__(
            WorkspaceInterventionSpecV1(
                kind=self._kind, module=module, factor=factor, cycles=cycles
            )
        )

    def apply_proposal(self, proposal: ModuleProposal, cycle: int) -> ModuleProposal:
        _, count, _ = _validate_proposal(proposal)
        module, active = self._module(count, cycle)
        output = ModuleProposal(
            **{field.name: getattr(proposal, field.name).clone() for field in fields(proposal)}
        )
        if active:
            factor = self.spec.factor
            assert factor is not None
            output.bid[:, module] = (proposal.bid[:, module].double() * factor).to(
                proposal.bid.dtype
            )
        _validate_proposal(output)
        return output


class ForceModuleWinner(_ModuleOperation):
    """Promote a module above the row maximum with one finite representable step.

    Recompute hard winners and soft probabilities from the counterfactual scores
    so downstream allocation keeps its original score-order contract. This method
    is inference-only when inputs carry gradients: nextafter has no defined
    training surrogate. All forced modules are promoted, even existing winners,
    so ordered distinct forces (at most R, checked by the plan) remain winners.
    """

    _kind = "force_module_winner"

    def __init__(self, module: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, module=module, cycles=cycles))

    def apply_arbitration(
        self, arbitration: WorkspaceArbiterOutput, cycle: int
    ) -> WorkspaceArbiterOutput:
        _, count, winners = _validate_arbitration(arbitration)
        module, active = self._module(count, cycle)
        values = {
            field.name: getattr(arbitration, field.name).clone() for field in fields(arbitration)
        }
        if active:
            if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in values.values()):
                raise WorkspaceInterventionError(
                    "Forced winner scoring requires inference/no-grad execution"
                )
            scores = values["scores"]
            maximum = scores.amax(dim=-1)
            promoted = torch.nextafter(maximum, torch.full_like(maximum, torch.inf))
            if not bool(torch.isfinite(promoted).all()):
                raise WorkspaceInterventionError("Forced score cannot be represented finitely")
            scores[:, module] = promoted
            indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :winners]
            mask = torch.zeros_like(arbitration.winner_mask).scatter(1, indices, True)
            values.update(
                scores=scores,
                winner_indices=indices,
                winner_mask=mask,
                soft_probabilities=torch.softmax(scores, dim=-1),
                selection_weights=mask.to(scores.dtype),
            )
        output = WorkspaceArbiterOutput(**values)
        _validate_arbitration(output)
        return output


class BlockModuleWrite(_ModuleOperation):
    """Restore a blocked winner's complete target record from pre-write state.

    Competition and allocation stay unchanged; a blocked write is not reassigned
    to a losing module. Apply after the ordinary writer and before state-phase
    interventions. Multiple blockers may compose against the same pre-write state.
    """

    _kind = "block_module_write"

    def __init__(self, module: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, module=module, cycles=cycles))

    def apply_write(
        self,
        before: WorkspaceState,
        written: WorkspaceState,
        allocation: SlotAllocatorOutput,
        cycle: int,
    ) -> WorkspaceState:
        validate_intervention_state(before)
        validate_intervention_state(written)
        if (
            written.slots.shape != before.slots.shape
            or written.slots.dtype != before.slots.dtype
            or written.slots.device != before.slots.device
        ):
            raise WorkspaceInterventionError("Pre-write and written workspace geometry must match")
        count = _validate_allocation(allocation, before)
        module, active = self._module(count, cycle)
        for state in (before, written):
            if not bool((state.source_module.masked_select(state.occupancy) < count).all()):
                raise WorkspaceInterventionError(
                    "Workspace source module is outside allocation domain"
                )
        if not active:
            return _clone(written)
        targets = allocation.assignment_mask[:, module]
        output = WorkspaceState(
            **{
                field.name: torch.where(
                    targets.unsqueeze(-1) if field.name == "slots" else targets,
                    getattr(before, field.name),
                    getattr(written, field.name),
                )
                for field in fields(before)
            }
        )
        validate_intervention_state(output)
        return output


__all__ = ["BlockModuleRead", "BlockModuleWrite", "ForceModuleWinner", "ScaleModuleBid"]
