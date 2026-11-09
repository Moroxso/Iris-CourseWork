"""Typed outputs produced by Iris cognitive components."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LayerSamplerOutput:
    """Context features and provenance produced by one layer-sampling pass."""

    context_features: torch.Tensor
    attention_mask: torch.Tensor
    mixing_weights: torch.Tensor
    selected_layer_indices: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class GoalEncoderOutput:
    """Request-level goal and its token-level attention distribution."""

    goal_state: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModuleInputRouterOutput:
    """Per-module prompt context and token-level routing attention."""

    module_inputs: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class CognitiveModuleOutput:
    """Updated local state for every independent cognitive module."""

    module_states: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModuleLatentBridgeOutput:
    """Per-module latent tokens and their frozen-backbone sequence metadata."""

    module_embeddings: torch.Tensor
    module_attention_mask: torch.Tensor
    module_position_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModuleProposal:
    """Latent workspace proposals for all cognitive modules."""

    content: torch.Tensor
    key: torch.Tensor
    bid: torch.Tensor
    confidence: torch.Tensor
    ttl: torch.Tensor
