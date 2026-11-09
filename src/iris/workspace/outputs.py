"""Typed outputs produced by global-workspace components."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WorkspaceReaderOutput:
    """Independent per-module reads and slot-attention weights."""

    workspace_reads: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class WorkspaceArbiterOutput:
    """Competition scores, diagnostics, and hard/soft winner selections."""

    scores: torch.Tensor
    novelty: torch.Tensor
    relevance: torch.Tensor
    soft_probabilities: torch.Tensor
    selection_weights: torch.Tensor
    winner_mask: torch.Tensor
    winner_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class SlotAllocatorOutput:
    """Unique module-to-slot assignments and their training surrogates."""

    allocation_scores: torch.Tensor
    retention_scores: torch.Tensor
    target_mask: torch.Tensor
    assignment_mask: torch.Tensor
    assignment_probabilities: torch.Tensor
    assignment_weights: torch.Tensor
    module_indices: torch.Tensor
    slot_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class LatentBridgeOutput:
    """Workspace latent tokens and their second-pass sequence metadata."""

    workspace_embeddings: torch.Tensor
    workspace_attention_mask: torch.Tensor
    workspace_position_ids: torch.Tensor
