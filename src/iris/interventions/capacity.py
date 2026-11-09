"""Inference-only constrained allocation with unchanged legacy tensor geometry."""

from dataclasses import fields

import torch

from iris.interventions.phases import _validate_allocation
from iris.interventions.state import WorkspaceInterventionError, validate_intervention_state
from iris.workspace.outputs import SlotAllocatorOutput
from iris.workspace.state import WorkspaceState


def restrict_allocation_capacity(
    state: WorkspaceState, allocation: SlotAllocatorOutput, *, capacity: int
) -> SlotAllocatorOutput:
    """Reserve targets in the usable prefix, with deterministic discarded overflow.

    The ordinary allocator supplies validated learned scores/retention and ranked
    winners. Select empties first then minimum-retention occupied slots, but only
    among indices < capacity. Assign the first min(R,capacity) winners to these
    targets by the same stable greedy compatibility rule. If capacity<R, remaining
    winners use distinct disabled suffix slots which MUST be cleared after writing.
    Competition, R, K and prefix length remain unchanged. The graph must also clear
    the disabled suffix before reading/scoring. No straight-through training rule
    is inferred for the altered target constraint.
    """
    validate_intervention_state(state)
    modules = _validate_allocation(allocation, state)
    batch, slots, _ = state.slots.shape
    if type(capacity) is not int or not 1 <= capacity <= slots:
        raise WorkspaceInterventionError("Capacity must be an exact integer within the workspace")
    if not bool((state.source_module.masked_select(state.occupancy) < modules).all()):
        raise WorkspaceInterventionError("Workspace source module is outside allocation domain")
    if capacity == slots:
        return SlotAllocatorOutput(
            **{field.name: getattr(allocation, field.name).clone() for field in fields(allocation)}
        )
    if torch.is_grad_enabled() and (
        any(getattr(state, field.name).requires_grad for field in fields(state))
        or any(getattr(allocation, field.name).requires_grad for field in fields(allocation))
    ):
        raise WorkspaceInterventionError(
            "Constrained allocation requires inference/no-grad execution"
        )

    winners = allocation.module_indices.shape[1]
    usable_writes = min(capacity, winners)
    scores = allocation.allocation_scores
    active = torch.arange(slots, device=scores.device).expand(batch, -1) < capacity
    empty = ~state.occupancy & active
    selected_empty = empty & (empty.to(torch.int64).cumsum(dim=-1) <= usable_writes)
    remaining = usable_writes - selected_empty.sum(dim=-1)
    occupied_order = torch.argsort(
        allocation.retention_scores.masked_fill(~(state.occupancy & active), torch.inf),
        dim=-1,
        stable=True,
    )
    selected_order = (
        torch.arange(slots, device=scores.device).expand(batch, -1) < remaining[:, None]
    )
    selected_occupied = torch.zeros_like(active).scatter(1, occupied_order, selected_order)
    usable_targets = selected_empty | (selected_occupied & state.occupancy & active)
    overflow_targets = torch.zeros_like(active)
    overflow_targets[:, capacity : capacity + winners - usable_writes] = True
    target_mask = usable_targets | overflow_targets

    assignment_mask = torch.zeros_like(allocation.assignment_mask)
    probabilities = torch.zeros_like(allocation.assignment_probabilities)
    remaining_targets = target_mask.clone()
    assigned_indices: list[torch.Tensor] = []
    rows = torch.arange(batch, device=scores.device)
    for rank in range(winners):
        module_indices = allocation.module_indices[:, rank]
        allowed = remaining_targets & (active if rank < usable_writes else ~active)
        masked = scores[rows, module_indices].masked_fill(~allowed, -torch.inf)
        if not bool(allowed.any(dim=-1).all()):
            raise WorkspaceInterventionError("Constrained allocation has no eligible target")
        slot_indices = torch.argmax(masked, dim=-1)
        probabilities[rows, module_indices] = torch.softmax(masked, dim=-1)
        assignment_mask[rows, module_indices, slot_indices] = True
        remaining_targets[rows, slot_indices] = False
        assigned_indices.append(slot_indices)
    output = SlotAllocatorOutput(
        allocation_scores=scores.clone(),
        retention_scores=allocation.retention_scores.clone(),
        target_mask=target_mask,
        assignment_mask=assignment_mask,
        assignment_probabilities=probabilities,
        assignment_weights=assignment_mask.to(scores.dtype),
        module_indices=allocation.module_indices.clone(),
        slot_indices=torch.stack(assigned_indices, dim=1),
    )
    _validate_allocation(output, state)
    return output
