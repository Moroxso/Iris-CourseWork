"""Relative-depth sampling for frozen backbone hidden states."""

from math import ceil

import torch
from torch import nn

from iris.backbone.outputs import BackboneForwardOutput
from iris.cognitive.outputs import LayerSamplerOutput
from iris.schemas import LayerSamplerConfig


class LayerSamplerError(RuntimeError):
    """Raised when hidden states cannot satisfy the sampler contract."""


class LayerSampler(nn.Module):
    """Project four relative backbone depths and mix them with learned weights."""

    def __init__(
        self,
        config: LayerSamplerConfig,
        *,
        backbone_hidden_size: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        if (
            not isinstance(backbone_hidden_size, int)
            or isinstance(backbone_hidden_size, bool)
            or backbone_hidden_size <= 0
        ):
            message = "backbone_hidden_size must be a positive integer"
            raise ValueError(message)
        if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim <= 0:
            message = "state_dim must be a positive integer"
            raise ValueError(message)

        self.backbone_hidden_size = backbone_hidden_size
        self.state_dim = state_dim
        self.layer_fractions = config.layer_fractions
        self.projections = nn.ModuleList(
            nn.Linear(backbone_hidden_size, state_dim, bias=False) for _ in self.layer_fractions
        )
        self.mixing_logits = nn.Parameter(torch.zeros(len(self.layer_fractions)))

    def normalized_mixing_weights(self) -> torch.Tensor:
        """Return the current convex mixing weights."""
        return torch.softmax(self.mixing_logits, dim=0)

    def forward(self, backbone_output: BackboneForwardOutput) -> LayerSamplerOutput:
        """Build trainable context features from a validated frozen forward pass."""
        hidden_states = backbone_output.hidden_states
        if not isinstance(hidden_states, tuple) or len(hidden_states) < 2:
            message = "Backbone output must contain an embedding state and block states"
            raise LayerSamplerError(message)

        num_hidden_layers = len(hidden_states) - 1
        selected_indices = self.resolve_layer_indices(
            self.layer_fractions,
            num_hidden_layers,
        )
        selected_states = tuple(hidden_states[index] for index in selected_indices)
        batch_size, sequence_length, input_dtype, input_device = self._validate_hidden_states(
            selected_states
        )
        attention_mask = self._validate_attention_mask(
            backbone_output.attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=input_device,
        )
        self._validate_parameter_boundary(dtype=input_dtype, device=input_device)

        materialized_states = tuple(state.detach().clone() for state in selected_states)
        projected_states = torch.stack(
            [
                projection(state)
                for projection, state in zip(self.projections, materialized_states, strict=True)
            ],
            dim=2,
        )
        mixing_weights = self.normalized_mixing_weights()
        context_features = torch.einsum("btld,l->btd", projected_states, mixing_weights)
        self._validate_output(
            context_features,
            mixing_weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=input_dtype,
            device=input_device,
        )
        return LayerSamplerOutput(
            context_features=context_features,
            attention_mask=attention_mask,
            mixing_weights=mixing_weights,
            selected_layer_indices=selected_indices,
        )

    @staticmethod
    def resolve_layer_indices(
        layer_fractions: tuple[float, float, float, float],
        num_hidden_layers: int,
    ) -> tuple[int, int, int, int]:
        """Resolve one-based block outputs without constructing trainable parameters."""
        if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
            message = "num_hidden_layers must be a positive integer"
            raise LayerSamplerError(message)
        indices = tuple(ceil(fraction * num_hidden_layers) for fraction in layer_fractions)
        if len(set(indices)) != len(indices):
            message = (
                "Layer fractions do not resolve to four unique block depths for "
                f"{num_hidden_layers} transformer layers"
            )
            raise LayerSamplerError(message)
        if any(index < 1 or index > num_hidden_layers for index in indices):
            message = "Resolved layer depth is outside the transformer block range"
            raise LayerSamplerError(message)
        return indices  # type: ignore[return-value]

    def _validate_hidden_states(
        self, selected_states: tuple[torch.Tensor, ...]
    ) -> tuple[int, int, torch.dtype, torch.device]:
        first = selected_states[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 3:
            message = "Selected hidden states must be rank-3 tensors"
            raise LayerSamplerError(message)
        batch_size, sequence_length, hidden_size = first.shape
        if batch_size <= 0 or sequence_length <= 0:
            message = "Selected hidden states must have non-empty batch and sequence axes"
            raise LayerSamplerError(message)
        if hidden_size != self.backbone_hidden_size:
            message = (
                "Selected hidden-state width must equal "
                f"backbone_hidden_size={self.backbone_hidden_size}"
            )
            raise LayerSamplerError(message)
        if not torch.is_floating_point(first):
            message = "Selected hidden states must use a floating dtype"
            raise LayerSamplerError(message)

        expected_shape = tuple(first.shape)
        for state in selected_states:
            if not isinstance(state, torch.Tensor) or state.ndim != 3:
                message = "Selected hidden states must be rank-3 tensors"
                raise LayerSamplerError(message)
            if tuple(state.shape) != expected_shape:
                message = "Selected hidden states must have identical shapes"
                raise LayerSamplerError(message)
            if state.dtype != first.dtype or state.device != first.device:
                message = "Selected hidden states must share one dtype and device"
                raise LayerSamplerError(message)
            if not torch.is_floating_point(state):
                message = "Selected hidden states must use a floating dtype"
                raise LayerSamplerError(message)
            if state.requires_grad or state.grad_fn is not None:
                message = "Frozen backbone hidden states must not track gradients"
                raise LayerSamplerError(message)
            if not bool(torch.isfinite(state).all()):
                message = "Selected hidden states contain NaN or Inf"
                raise LayerSamplerError(message)
        return batch_size, sequence_length, first.dtype, first.device

    @staticmethod
    def _validate_attention_mask(
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, torch.Tensor):
            message = "Backbone attention mask must be a tensor"
            raise LayerSamplerError(message)
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if attention_mask.dtype not in integer_dtypes:
            message = "Backbone attention mask must use a boolean or integer dtype"
            raise LayerSamplerError(message)
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            message = "Backbone attention mask must match [batch, sequence]"
            raise LayerSamplerError(message)
        if attention_mask.device != device:
            message = "Backbone attention mask must share the hidden-state device"
            raise LayerSamplerError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Backbone attention mask must contain only zeros and ones"
            raise LayerSamplerError(message)
        if not bool(attention_mask.any(dim=1).all()):
            message = "Every sample must contain at least one attended token"
            raise LayerSamplerError(message)
        return attention_mask.detach().clone().to(dtype=torch.bool)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = (
                    "LayerSampler parameters must explicitly match the hidden-state "
                    "dtype and device"
                )
                raise LayerSamplerError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"LayerSampler parameter {name!r} contains NaN or Inf"
                raise LayerSamplerError(message)

    def _validate_output(
        self,
        context_features: torch.Tensor,
        mixing_weights: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        expected_shape = (batch_size, sequence_length, self.state_dim)
        if tuple(context_features.shape) != expected_shape:
            message = f"Context features must have shape {expected_shape}"
            raise LayerSamplerError(message)
        if context_features.dtype != dtype or context_features.device != device:
            message = "Context features must preserve the hidden-state dtype and device"
            raise LayerSamplerError(message)
        if not bool(torch.isfinite(context_features).all()):
            message = "Context features contain NaN or Inf"
            raise LayerSamplerError(message)
        if not bool(torch.isfinite(mixing_weights).all()) or not bool((mixing_weights >= 0).all()):
            message = "Layer mixing weights must be finite and non-negative"
            raise LayerSamplerError(message)
        if not torch.allclose(
            mixing_weights.sum(),
            torch.ones((), dtype=dtype, device=device),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Layer mixing weights must sum to one"
            raise LayerSamplerError(message)
