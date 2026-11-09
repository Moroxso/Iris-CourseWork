"""Explicit differentiable usage surrogate, separate from measured hard routing."""

from dataclasses import dataclass

import torch

from iris.training.losses import IrisLossError, uniform_utilization_loss
from iris.training.outputs import IrisTeacherCycleTraceV1


@dataclass(frozen=True, slots=True)
class IrisWorkspaceUsageV1:
    """Cycle/batch averages; soft surrogate is not an actual selection histogram."""

    module_probabilities: torch.Tensor
    slot_probabilities: torch.Tensor
    hard_module_frequencies: torch.Tensor
    hard_slot_frequencies: torch.Tensor


def aggregate_workspace_usage_v1(
    cycles: tuple[IrisTeacherCycleTraceV1, ...],
) -> IrisWorkspaceUsageV1:
    """Average soft module probabilities and their all-slot compatibility mixture.

    p(slot|module) = softmax(full pre-mask allocation scores), NOT the conditional
    probability among the allocator's hard-selected R targets. Average their
    module-probability-weighted mixture over cycles and batch. Hard winner/write
    masks are counted independently and never substituted into the loss graph.
    """
    if type(cycles) is not tuple or not cycles:
        raise IrisLossError("Usage requires a nonempty explicit cycle trace")
    modules: list[torch.Tensor] = []
    slots: list[torch.Tensor] = []
    hard_modules: list[torch.Tensor] = []
    hard_slots: list[torch.Tensor] = []
    boundary = None
    for cycle in cycles:
        if type(cycle) is not IrisTeacherCycleTraceV1:
            raise IrisLossError("Usage requires exact cycle trace records")
        probability, scores = cycle.module_probabilities, cycle.allocation_scores
        if (
            type(probability) is not torch.Tensor
            or type(scores) is not torch.Tensor
            or probability.ndim != 2
            or scores.ndim != 3
            or probability.layout != torch.strided
            or scores.layout != torch.strided
            or any(size < 1 for size in scores.shape)
            or tuple(probability.shape) != tuple(scores.shape[:2])
            or probability.dtype not in (torch.float32, torch.float64)
            or scores.dtype != probability.dtype
            or probability.device.type not in ("cpu", "cuda")
            or scores.device != probability.device
            or not bool(torch.isfinite(probability).all())
            or not bool(torch.isfinite(scores).all())
            or bool((probability < 0).any())
        ):
            raise IrisLossError("Invalid floating usage trace boundary")
        current = (tuple(scores.shape), scores.dtype, scores.device)
        if boundary is not None and current != boundary:
            raise IrisLossError("Usage cycle shape, dtype or device changed")
        boundary = current
        tolerance = 1e-12 if probability.dtype == torch.float64 else 1e-6
        if bool((probability.detach().sum(dim=-1) - 1.0).abs().gt(tolerance).any()):
            raise IrisLossError("Module probabilities must be unit-sum per example")
        winners, assignments = cycle.winner_mask, cycle.assignment_mask
        for value, shape in ((winners, probability.shape), (assignments, scores.shape)):
            if (
                type(value) is not torch.Tensor
                or value.layout != torch.strided
                or value.dtype != torch.bool
                or value.device != scores.device
                or value.shape != shape
            ):
                raise IrisLossError("Invalid hard routing trace")
        counts = winners.sum(dim=-1)
        if (
            bool((counts < 1).any())
            or bool((counts != counts[0]).any())
            or not torch.equal(assignments.sum(dim=-1), winners.to(dtype=torch.int64))
            or bool((assignments.sum(dim=1) > 1).any())
        ):
            raise IrisLossError("Hard routing must write each winner once to unique slots")
        if hard_modules and int(counts[0]) != int(cycles[0].winner_mask[0].sum()):
            raise IrisLossError("Winner count changed across usage cycles")
        with torch.autocast(device_type=scores.device.type, enabled=False):
            conditional = torch.softmax(scores, dim=-1)
            modules.append(probability.mean(dim=0))
            # On the singleton simplex every conditional is EXACTLY one, and
            # the only slot's KL/score gradient is exactly zero. Re-summing
            # rounded module softmax values can instead produce 1+ulp and fail
            # the strict probability boundary. Retain the actual softmax graph;
            # no clamping, renormalization or hard-routing histogram is used.
            mixture = (
                conditional[:, 0, :]
                if scores.shape[-1] == 1
                else (probability.unsqueeze(-1) * conditional).sum(dim=1)
            )
            slots.append(mixture.mean(dim=0))
            denominator = counts.to(dtype=scores.dtype).unsqueeze(-1)
            hard_modules.append((winners.to(dtype=scores.dtype) / denominator).mean(dim=0))
            hard_slots.append(
                (assignments.sum(dim=1).to(dtype=scores.dtype) / denominator).mean(dim=0)
            )
    result = IrisWorkspaceUsageV1(
        module_probabilities=torch.stack(modules).mean(dim=0),
        slot_probabilities=torch.stack(slots).mean(dim=0),
        hard_module_frequencies=torch.stack(hard_modules).mean(dim=0),
        hard_slot_frequencies=torch.stack(hard_slots).mean(dim=0),
    )
    # Reuse the strict simplex/numerical checks without replacing the probabilities.
    uniform_utilization_loss(result.module_probabilities, result.slot_probabilities)
    return result


__all__ = ["IrisWorkspaceUsageV1", "aggregate_workspace_usage_v1"]
