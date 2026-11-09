"""Attention pooling for request-level goal representations."""

from math import sqrt

import torch
from torch import nn

from iris.cognitive.outputs import GoalEncoderOutput


class GoalEncoderError(RuntimeError):
    """Raised when context features cannot satisfy the goal encoder contract."""


class GoalEncoder(nn.Module):
    """Pool prompt context into one request-level goal state."""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim <= 0:
            message = "state_dim must be a positive integer"
            raise ValueError(message)
        self.state_dim = state_dim
        self.pooling_query = nn.Parameter(torch.zeros(state_dim))

    def forward(
        self, context_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> GoalEncoderOutput:
        """Apply learned masked attention pooling over the prompt sequence."""
        batch_size, sequence_length, dtype, device = self._validate_context(context_features)
        mask = self._validate_attention_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        if context_features.is_inference() and not torch.is_inference_mode_enabled():
            context_features = context_features.detach().clone()

        score_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        with torch.autocast(device_type=device.type, enabled=False):
            safe_context = context_features.masked_fill(~mask.unsqueeze(-1), 0).to(
                dtype=score_dtype
            )
            scores = torch.einsum(
                "btd,d->bt",
                safe_context,
                self.pooling_query.to(dtype=score_dtype) / sqrt(self.state_dim),
            )
            if not bool(torch.isfinite(scores).all()):
                message = "Goal attention scores contain NaN or Inf"
                raise GoalEncoderError(message)
            scores = scores.masked_fill(~mask, -torch.inf)
            attention_weights = torch.softmax(scores, dim=-1)
            goal_state = torch.einsum("bt,btd->bd", attention_weights, safe_context).to(dtype=dtype)
        self._validate_output(
            goal_state,
            attention_weights,
            mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
        )
        return GoalEncoderOutput(
            goal_state=goal_state,
            attention_weights=attention_weights,
        )

    def _validate_context(
        self, context_features: torch.Tensor
    ) -> tuple[int, int, torch.dtype, torch.device]:
        if not isinstance(context_features, torch.Tensor):
            message = "Context features must be a tensor"
            raise GoalEncoderError(message)
        if context_features.ndim != 3:
            message = "Context features must have shape [batch, sequence, state_dim]"
            raise GoalEncoderError(message)
        batch_size, sequence_length, state_dim = context_features.shape
        if batch_size <= 0 or sequence_length <= 0:
            message = "Context features must have non-empty batch and sequence axes"
            raise GoalEncoderError(message)
        if state_dim != self.state_dim:
            message = f"Context feature width must equal state_dim={self.state_dim}"
            raise GoalEncoderError(message)
        if not torch.is_floating_point(context_features):
            message = "Context features must use a floating dtype"
            raise GoalEncoderError(message)
        if not bool(torch.isfinite(context_features).all()):
            message = "Context features contain NaN or Inf"
            raise GoalEncoderError(message)
        return batch_size, sequence_length, context_features.dtype, context_features.device

    @staticmethod
    def _validate_attention_mask(
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, torch.Tensor):
            message = "Attention mask must be a tensor"
            raise GoalEncoderError(message)
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if attention_mask.dtype not in integer_dtypes:
            message = "Attention mask must use a boolean or integer dtype"
            raise GoalEncoderError(message)
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            message = "Attention mask must match [batch, sequence]"
            raise GoalEncoderError(message)
        if attention_mask.device != device:
            message = "Attention mask must share the context-feature device"
            raise GoalEncoderError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Attention mask must contain only zeros and ones"
            raise GoalEncoderError(message)
        if not bool(attention_mask.any(dim=1).all()):
            message = "Every sample must contain at least one attended token"
            raise GoalEncoderError(message)
        return attention_mask.detach().clone().to(dtype=torch.bool)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if self.pooling_query.dtype != dtype or self.pooling_query.device != device:
            message = (
                "GoalEncoder parameters must explicitly match the context-feature dtype and device"
            )
            raise GoalEncoderError(message)
        if not bool(torch.isfinite(self.pooling_query).all()):
            message = "GoalEncoder pooling query contains NaN or Inf"
            raise GoalEncoderError(message)

    def _validate_output(
        self,
        goal_state: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(goal_state.shape) != (batch_size, self.state_dim):
            message = "Goal state must have shape [batch, state_dim]"
            raise GoalEncoderError(message)
        if goal_state.dtype != dtype or goal_state.device != device:
            message = "Goal state must preserve the context-feature dtype and device"
            raise GoalEncoderError(message)
        if tuple(attention_weights.shape) != (batch_size, sequence_length):
            message = "Goal attention weights must have shape [batch, sequence]"
            raise GoalEncoderError(message)
        if attention_weights.device != device or not bool(torch.isfinite(attention_weights).all()):
            message = "Goal attention weights must be finite and remain on the input device"
            raise GoalEncoderError(message)
        if not bool((attention_weights >= 0).all()):
            message = "Goal attention weights must be non-negative"
            raise GoalEncoderError(message)
        if not bool((attention_weights.masked_select(~attention_mask) == 0).all()):
            message = "Masked prompt tokens must receive zero goal attention"
            raise GoalEncoderError(message)
        expected_sums = torch.ones(batch_size, dtype=attention_weights.dtype, device=device)
        if not torch.allclose(attention_weights.sum(dim=1), expected_sums, rtol=1e-4, atol=1e-6):
            message = "Goal attention weights must sum to one"
            raise GoalEncoderError(message)
        if not bool(torch.isfinite(goal_state).all()):
            message = "Goal state contains NaN or Inf"
            raise GoalEncoderError(message)
