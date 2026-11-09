"""Typed outputs for Iris second-pass sequence construction and generation."""

from dataclasses import dataclass
from typing import Literal

import torch

from iris.workspace.state import WorkspaceState


@dataclass(frozen=True, slots=True)
class SecondPassSequenceOutput:
    """Teacher-forced embeddings and causal metadata for the frozen backbone."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True, slots=True)
class SecondPassGenerationPrefixOutput:
    """Validated `[prompt][workspace]` prefix for one cached generation loop."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


type GenerationStopReason = Literal["stop_token", "max_new_tokens"]
type DecodingStrategy = Literal["greedy", "sampling"]


@dataclass(frozen=True, slots=True)
class IrisGenerationOutput:
    """Generated token IDs, termination provenance, and cognitive trace."""

    token_ids: torch.Tensor
    stop_reason: GenerationStopReason
    matched_stop_token_id: int | None
    workspace_trace: tuple[WorkspaceState, ...]


@dataclass(frozen=True, slots=True)
class IrisGeneratorOutput:
    """Decoded user-facing text with exact token and cognitive provenance."""

    text: str
    token_ids: torch.Tensor
    stop_reason: GenerationStopReason
    matched_stop_token_id: int | None
    workspace_trace: tuple[WorkspaceState, ...]
