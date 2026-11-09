"""Soft prompt routing for independent cognitive modules."""

from math import sqrt

import torch
from torch import nn

from iris.cognitive.outputs import ModuleInputRouterOutput


class ModuleInputRouterError(RuntimeError):
    """Raised when prompt context cannot satisfy the module-router contract."""


class ModuleInputRouter(nn.Module):
    """Route prompt context to modules using one learned query per module."""

    def __init__(self, *, state_dim: int, num_modules: int) -> None:
        super().__init__()
        if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim <= 0:
            message = "state_dim must be a positive integer"
            raise ValueError(message)
        if not isinstance(num_modules, int) or isinstance(num_modules, bool) or num_modules <= 0:
            message = "num_modules must be a positive integer"
            raise ValueError(message)
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.module_queries = nn.Parameter(torch.zeros(num_modules, state_dim))

    def forward(
        self, context_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> ModuleInputRouterOutput:
        """Apply masked soft attention for every module query."""
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

        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        with torch.autocast(device_type=device.type, enabled=False):
            safe_context = context_features.masked_fill(~mask.unsqueeze(-1), 0).to(
                dtype=accumulation_dtype
            )
            scores = torch.einsum(
                "btd,md->bmt",
                safe_context,
                self.module_queries.to(dtype=accumulation_dtype) / sqrt(self.state_dim),
            )
            if not bool(torch.isfinite(scores).all()):
                message = "Module routing scores contain NaN or Inf"
                raise ModuleInputRouterError(message)
            scores = scores.masked_fill(~mask.unsqueeze(1), -torch.inf)
            attention_weights = torch.softmax(scores, dim=-1)
            module_inputs = torch.einsum("bmt,btd->bmd", attention_weights, safe_context).to(
                dtype=dtype
            )

        self._validate_output(
            module_inputs,
            attention_weights,
            mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
        )
        return ModuleInputRouterOutput(
            module_inputs=module_inputs,
            attention_weights=attention_weights,
        )

    def _validate_context(
        self, context_features: torch.Tensor
    ) -> tuple[int, int, torch.dtype, torch.device]:
        if not isinstance(context_features, torch.Tensor):
            message = "Context features must be a tensor"
            raise ModuleInputRouterError(message)
        if context_features.ndim != 3:
            message = "Context features must have shape [batch, sequence, state_dim]"
            raise ModuleInputRouterError(message)
        batch_size, sequence_length, state_dim = context_features.shape
        if batch_size <= 0 or sequence_length <= 0:
            message = "Context features must have non-empty batch and sequence axes"
            raise ModuleInputRouterError(message)
        if state_dim != self.state_dim:
            message = f"Context feature width must equal state_dim={self.state_dim}"
            raise ModuleInputRouterError(message)
        if not torch.is_floating_point(context_features):
            message = "Context features must use a floating dtype"
            raise ModuleInputRouterError(message)
        if not bool(torch.isfinite(context_features).all()):
            message = "Context features contain NaN or Inf"
            raise ModuleInputRouterError(message)
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
            raise ModuleInputRouterError(message)
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
            raise ModuleInputRouterError(message)
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            message = "Attention mask must match [batch, sequence]"
            raise ModuleInputRouterError(message)
        if attention_mask.device != device:
            message = "Attention mask must share the context-feature device"
            raise ModuleInputRouterError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Attention mask must contain only zeros and ones"
            raise ModuleInputRouterError(message)
        if not bool(attention_mask.any(dim=1).all()):
            message = "Every sample must contain at least one attended token"
            raise ModuleInputRouterError(message)
        return attention_mask.detach().clone().to(dtype=torch.bool)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if self.module_queries.dtype != dtype or self.module_queries.device != device:
            message = (
                "ModuleInputRouter parameters must explicitly match the context-feature "
                "dtype and device"
            )
            raise ModuleInputRouterError(message)
        if not bool(torch.isfinite(self.module_queries).all()):
            message = "ModuleInputRouter queries contain NaN or Inf"
            raise ModuleInputRouterError(message)

    def _validate_output(
        self,
        module_inputs: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(module_inputs.shape) != (batch_size, self.num_modules, self.state_dim):
            message = "Module inputs must have shape [batch, modules, state_dim]"
            raise ModuleInputRouterError(message)
        if module_inputs.dtype != dtype or module_inputs.device != device:
            message = "Module inputs must preserve the context-feature dtype and device"
            raise ModuleInputRouterError(message)
        expected_attention_shape = (batch_size, self.num_modules, sequence_length)
        if tuple(attention_weights.shape) != expected_attention_shape:
            message = "Routing attention must have shape [batch, modules, sequence]"
            raise ModuleInputRouterError(message)
        if attention_weights.device != device or not bool(torch.isfinite(attention_weights).all()):
            message = "Routing attention must be finite and remain on the input device"
            raise ModuleInputRouterError(message)
        if not bool((attention_weights >= 0).all()):
            message = "Routing attention must be non-negative"
            raise ModuleInputRouterError(message)
        expanded_mask = attention_mask.unsqueeze(1).expand_as(attention_weights)
        if not bool((attention_weights.masked_select(~expanded_mask) == 0).all()):
            message = "Masked prompt tokens must receive zero routing attention"
            raise ModuleInputRouterError(message)
        expected_sums = torch.ones(
            (batch_size, self.num_modules),
            dtype=attention_weights.dtype,
            device=device,
        )
        if not torch.allclose(attention_weights.sum(dim=-1), expected_sums, rtol=1e-4, atol=1e-6):
            message = "Routing attention must sum to one for every module"
            raise ModuleInputRouterError(message)
        if not bool(torch.isfinite(module_inputs).all()):
            message = "Module inputs contain NaN or Inf"
            raise ModuleInputRouterError(message)
