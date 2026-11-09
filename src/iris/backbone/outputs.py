"""Typed outputs produced by backbone forward passes."""

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache


@dataclass(frozen=True, slots=True)
class BackboneTokenizedTarget:
    """Owned target-token tensors prepared by the private tokenizer boundary."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class BackboneForwardOutput:
    """Validated tensors returned by one prompt forward pass."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class BackboneFirstPassOutput(BackboneForwardOutput):
    """First-pass tensors plus the exact prompt token provenance."""

    input_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class BackboneCausalOutput:
    """Validated output of a causal forward pass driven by input embeddings."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    past_key_values: Cache | None
