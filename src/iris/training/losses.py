"""Differentiable loss primitives from source specification section 12.

These functions do not select a training graph, estimate usage distributions,
construct probe heads/vocabularies or change the historical LM-only trainer.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional

PROBE_FACT_FIELDS = (
    "active_constraint",
    "operation",
    "conflicting_conditions",
    "result_type",
)


class IrisLossError(ValueError):
    """Loss inputs or numerical results do not satisfy the declared contract."""


class IrisLossWeights(BaseModel):
    """Initial section-12.6 coefficients; LM's coefficient remains exactly one."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )
    probe: float = Field(default=0.05, ge=0.0)
    diversity: float = Field(default=0.01, ge=0.0)
    utilization: float = Field(default=0.01, ge=0.0)
    norm: float = Field(default=0.001, ge=0.0)


@dataclass(frozen=True, slots=True)
class IrisLossBreakdown:
    """Graph-attached scalar terms, not a serialized metric or training receipt."""

    language_model: torch.Tensor
    probe: torch.Tensor
    diversity: torch.Tensor
    utilization: torch.Tensor
    norm: torch.Tensor
    total: torch.Tensor
    weights: IrisLossWeights


def _floating_tensor(value: torch.Tensor, *, ndim: int, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or value.layout != torch.strided
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or value.device.type not in ("cpu", "cuda")
        or any(size <= 0 for size in value.shape)
    ):
        raise IrisLossError(f"{name} has an invalid floating tensor contract")
    if not bool(torch.isfinite(value).all()):
        raise IrisLossError(f"{name} must contain only finite values")
    return value


def _finite_result(value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise IrisLossError("Loss arithmetic produced a non-finite result")
    return value


def _accumulation_dtype(values: tuple[torch.Tensor, ...]) -> torch.dtype:
    return torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32


def workspace_diversity_loss(slots: torch.Tensor, *, margin: float = 0.30) -> torch.Tensor:
    """Mean over batches of the ordered off-diagonal K(K-1) cosine hinge.

    Includes every slot. K=1 has no pairs and yields graph-attached zero.
    Zero-vector cosine uses normalization epsilon 1e-8. No occupancy masking.
    """
    slots = _floating_tensor(slots, ndim=3, name="Workspace [batch, slots, width]")
    if type(margin) not in (int, float) or not isfinite(margin) or not -1.0 <= margin <= 1.0:
        raise IrisLossError("Diversity margin must be a finite number in [-1, 1]")
    count = slots.shape[1]
    with torch.autocast(device_type=slots.device.type, enabled=False):
        values = slots.to(dtype=_accumulation_dtype((slots,)))
        if count == 1:
            # Multiplication before reduction avoids overflowing an irrelevant sum.
            return (values * 0.0).sum()
        norms = _finite_result(torch.linalg.vector_norm(values, dim=-1, keepdim=True))
        normalized = values / norms.clamp_min(1e-8)
        similarities = normalized @ normalized.transpose(-1, -2)
        hinge = functional.relu(similarities - float(margin))
        diagonal = torch.eye(count, dtype=torch.bool, device=slots.device)
        hinge = hinge.masked_fill(diagonal.unsqueeze(0), 0.0)
        result = (hinge.sum(dim=(-2, -1)) / (count * (count - 1))).mean()
    return _finite_result(result)


def workspace_norm_loss(slots: torch.Tensor) -> torch.Tensor:
    """Mean over batch and all K slots of (L2 norm - sqrt(D)) squared.

    Empty slots contribute D, as the source formula specifies. Torch's norm
    convention gives a zero subgradient at an exactly zero vector.
    """
    slots = _floating_tensor(slots, ndim=3, name="Workspace [batch, slots, width]")
    with torch.autocast(device_type=slots.device.type, enabled=False):
        values = slots.to(dtype=_accumulation_dtype((slots,)))
        result = (torch.linalg.vector_norm(values, dim=-1) - sqrt(slots.shape[-1])).square().mean()
    return _finite_result(result)


def _probabilities(value: torch.Tensor, *, name: str) -> torch.Tensor:
    value = _floating_tensor(value, ndim=1, name=name)
    # Arbiter/allocation probabilities already use an accumulation dtype. Requiring
    # it here avoids accepting visibly non-unit half-precision vectors as a simplex.
    if value.dtype not in (torch.float32, torch.float64):
        raise IrisLossError("Utilization probabilities must be float32 or float64")
    tolerance = 1e-6 if value.dtype == torch.float32 else 1e-12
    if (
        bool((value < 0).any())
        or bool((value > 1).any())
        or abs(float(value.detach().sum()) - 1.0) > tolerance
    ):
        raise IrisLossError("Utilization requires nonnegative unit-sum probability vectors")
    return value


def uniform_utilization_loss(
    module_probabilities: torch.Tensor, slot_probabilities: torch.Tensor
) -> torch.Tensor:
    """KL(p_modules || U_M) + KL(p_slots || U_K), using natural logarithms.

    Caller supplies already aggregated distributions: no implicit softmax,
    renormalization, hard-winner histogram, batch or cycle averaging. Zero mass
    contributes exactly zero, with a finite zero derivative chosen at that
    boundary; it is not the limiting derivative of p*log(p). Roundoff-negative
    KL terms are clamped to zero. Probability gradients are otherwise preserved.
    """
    modules = _probabilities(module_probabilities, name="Module probabilities [M]")
    slots = _probabilities(slot_probabilities, name="Slot probabilities [K]")
    if modules.device != slots.device:
        raise IrisLossError("Utilization probabilities must share one device")
    dtype = _accumulation_dtype((modules, slots))
    with torch.autocast(device_type=modules.device.type, enabled=False):
        terms = []
        for distribution in (modules, slots):
            probabilities = distribution.to(dtype=dtype)
            count = probabilities.numel()
            log_input = torch.where(
                probabilities > 0, probabilities * count, torch.ones_like(probabilities)
            )
            term = (probabilities * log_input.log()).sum()
            terms.append(term.clamp_min(0.0))
        result = terms[0] + terms[1]
    return _finite_result(result)


def workspace_probe_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    head_reduction: Literal["sum", "mean"],
) -> torch.Tensor:
    """Four categorical cross-entropies, batch-mean per head; explicit head reduction.

    Class vocabularies, labels and logits must be supplied by the training caller.
    No labels are read from model inputs, inferred from task family, or fitted to
    validation/test data here. The reduction has no silent scientific default.
    """
    if (
        type(logits) is not dict
        or type(targets) is not dict
        or set(logits) != set(PROBE_FACT_FIELDS)
        or set(targets) != set(PROBE_FACT_FIELDS)
        or type(head_reduction) is not str
        or head_reduction not in ("sum", "mean")
    ):
        raise IrisLossError("Probe loss requires exactly four fact heads and an explicit reduction")
    predictions = tuple(
        _floating_tensor(logits[name], ndim=2, name=f"Probe {name} logits [batch, classes]")
        for name in PROBE_FACT_FIELDS
    )
    batch, device = predictions[0].shape[0], predictions[0].device
    for name, prediction in zip(PROBE_FACT_FIELDS, predictions, strict=True):
        label = targets[name]
        if prediction.shape[0] != batch or prediction.device != device:
            raise IrisLossError("Probe heads must share one batch and device")
        if (
            not isinstance(label, torch.Tensor)
            or label.layout != torch.strided
            or tuple(label.shape) != (batch,)
            or label.dtype != torch.int64
            or label.device != device
            or bool((label < 0).any())
            or bool((label >= prediction.shape[1]).any())
        ):
            raise IrisLossError("Probe labels must be valid int64 class indices for each example")
    dtype = _accumulation_dtype(predictions)
    with torch.autocast(device_type=device.type, enabled=False):
        losses = torch.stack(
            [
                functional.cross_entropy(
                    prediction.to(dtype=dtype), targets[name], reduction="mean"
                )
                for name, prediction in zip(PROBE_FACT_FIELDS, predictions, strict=True)
            ]
        )
        result = losses.sum() if head_reduction == "sum" else losses.mean()
    return _finite_result(result)


def combine_iris_losses(
    *,
    language_model_loss: torch.Tensor,
    probe_loss: torch.Tensor,
    diversity_loss: torch.Tensor,
    utilization_loss: torch.Tensor,
    norm_loss: torch.Tensor,
    weights: IrisLossWeights,
) -> IrisLossBreakdown:
    """Combine five supplied scalar terms without detaching or taking a training step.

    LM normalization is not changed: the caller must explicitly resolve summed
    source LM versus historical token-mean training when integrating this API.
    Nondefault weights must later be bound to a new run by the training launcher;
    this arithmetic function does not create or approve an MLflow run.
    """
    if type(weights) is not IrisLossWeights:
        raise IrisLossError("Loss weights must be exactly IrisLossWeights")
    try:
        owned = IrisLossWeights.model_validate(weights.model_dump(mode="python"))
    except ValueError:
        raise IrisLossError("Loss weights must be finite and nonnegative") from None
    values = tuple(
        _floating_tensor(value, ndim=0, name=name)
        for name, value in (
            ("LM loss", language_model_loss),
            ("Probe loss", probe_loss),
            ("Diversity loss", diversity_loss),
            ("Utilization loss", utilization_loss),
            ("Norm loss", norm_loss),
        )
    )
    device = values[0].device
    if any(value.device != device or bool(value < 0) for value in values):
        raise IrisLossError("Loss terms must be nonnegative scalars on one device")
    dtype = _accumulation_dtype(values)
    with torch.autocast(device_type=device.type, enabled=False):
        lm, probe, diversity, utilization, norm = (value.to(dtype=dtype) for value in values)
        total = _finite_result(
            lm
            + owned.probe * probe
            + owned.diversity * diversity
            + owned.utilization * utilization
            + owned.norm * norm
        )
    return IrisLossBreakdown(lm, probe, diversity, utilization, norm, total, owned)


__all__ = [
    "PROBE_FACT_FIELDS",
    "IrisLossBreakdown",
    "IrisLossError",
    "IrisLossWeights",
    "combine_iris_losses",
    "uniform_utilization_loss",
    "workspace_diversity_loss",
    "workspace_norm_loss",
    "workspace_probe_loss",
]
