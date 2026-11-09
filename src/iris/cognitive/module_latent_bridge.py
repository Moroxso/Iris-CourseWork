"""Projection of final no-workspace module states into frozen-LLM tokens."""

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import nn

from iris.cognitive.outputs import ModuleLatentBridgeOutput


class _StableLayerNormalization(torch.autograd.Function):
    """Scaled LayerNorm core with finite gradients at extreme magnitudes."""

    @staticmethod
    def forward(context: Any, tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
        maximum = tensor.abs().amax(dim=-1, keepdim=True)
        epsilon_scale = torch.full_like(maximum, epsilon**0.5)
        scale = torch.maximum(maximum, epsilon_scale)
        scaled = tensor / scale
        centered = scaled - scaled.mean(dim=-1, keepdim=True)
        scaled_standard_deviation = torch.linalg.vector_norm(
            centered,
            dim=-1,
            keepdim=True,
        ) / (tensor.shape[-1] ** 0.5)
        denominator = torch.hypot(
            scaled_standard_deviation,
            epsilon_scale / scale,
        )
        normalized = centered / denominator
        inverse_standard_deviation = torch.reciprocal(scale) / denominator
        context.save_for_backward(normalized, inverse_standard_deviation)
        return normalized

    @staticmethod
    def backward(
        context: Any,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        normalized, inverse_standard_deviation = context.saved_tensors
        mean_gradient = output_gradient.mean(dim=-1, keepdim=True)
        mean_gradient_projection = (output_gradient * normalized).mean(
            dim=-1,
            keepdim=True,
        )
        input_gradient = inverse_standard_deviation * (
            output_gradient - mean_gradient - normalized * mean_gradient_projection
        )
        return input_gradient, None


class ModuleLatentBridgeError(RuntimeError):
    """Raised when no-workspace module states cannot form latent tokens."""


class ModuleLatentBridge(nn.Module):
    """Map each final independent module state to one frozen-backbone token."""

    def __init__(
        self,
        *,
        state_dim: int,
        backbone_hidden_size: int,
        num_modules: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "backbone_hidden_size": backbone_hidden_size,
            "num_modules": num_modules,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.num_modules = num_modules
        self.bridge_projection = nn.Linear(state_dim, backbone_hidden_size, bias=False)
        self.normalization = nn.LayerNorm(backbone_hidden_size)
        self.module_type_embedding = nn.Parameter(torch.empty(backbone_hidden_size))
        nn.init.normal_(
            self.module_type_embedding,
            mean=0.0,
            std=backbone_hidden_size**-0.5,
        )

    def forward(
        self,
        module_states: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> ModuleLatentBridgeOutput:
        """Return one attended latent token per final independent module state."""
        batch_size, dtype, device = self._validate_module_states(module_states)
        prompt_mask = self._validate_prompt_attention_mask(
            prompt_attention_mask,
            batch_size=batch_size,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)
        values = self._materialize(module_states)
        accumulation_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        with torch.autocast(device_type=device.type, enabled=False):
            projected = functional.linear(
                values.to(dtype=accumulation_dtype),
                self.bridge_projection.weight.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(projected).all()):
                message = "Projected module embeddings contain NaN or Inf"
                raise ModuleLatentBridgeError(message)
            normalized = self._stable_layer_norm(
                projected,
                weight=self.normalization.weight.to(dtype=accumulation_dtype),
                bias=self.normalization.bias.to(dtype=accumulation_dtype),
            )
            latent = normalized + self.module_type_embedding.to(dtype=accumulation_dtype)
            if not bool(torch.isfinite(latent).all()):
                message = "Latent module embeddings contain NaN or Inf"
                raise ModuleLatentBridgeError(message)
            module_embeddings = latent.to(dtype=dtype)
        if not bool(torch.isfinite(module_embeddings).all()):
            message = "Latent module embeddings contain NaN or Inf after storage cast"
            raise ModuleLatentBridgeError(message)

        module_attention_mask = torch.ones(
            (batch_size, self.num_modules),
            dtype=torch.bool,
            device=device,
        )
        prompt_lengths = prompt_mask.to(dtype=torch.int64).sum(dim=-1, keepdim=True)
        module_position_ids = prompt_lengths + torch.arange(
            self.num_modules,
            dtype=torch.int64,
            device=device,
        )
        output = ModuleLatentBridgeOutput(
            module_embeddings=module_embeddings,
            module_attention_mask=module_attention_mask,
            module_position_ids=module_position_ids,
        )
        self._validate_output(
            output,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            expected_position_ids=module_position_ids,
        )
        return output

    def _validate_module_states(
        self,
        module_states: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if type(module_states) is not torch.Tensor or module_states.ndim != 3:
            message = "Module states must have shape [batch, modules, state_dim]"
            raise ModuleLatentBridgeError(message)
        batch_size = module_states.shape[0]
        if batch_size <= 0 or tuple(module_states.shape) != (
            batch_size,
            self.num_modules,
            self.state_dim,
        ):
            message = "Module state shape does not match the latent bridge contract"
            raise ModuleLatentBridgeError(message)
        if not torch.is_floating_point(module_states):
            message = "Module states must use a floating dtype"
            raise ModuleLatentBridgeError(message)
        try:
            finite = bool(torch.isfinite(module_states).all())
        except Exception as error:
            message = f"Module states cannot be inspected on {module_states.device}"
            raise ModuleLatentBridgeError(message) from error
        if not finite:
            message = "Module states contain NaN or Inf"
            raise ModuleLatentBridgeError(message)
        return batch_size, module_states.dtype, module_states.device

    @staticmethod
    def _validate_prompt_attention_mask(
        prompt_attention_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if (
            type(prompt_attention_mask) is not torch.Tensor
            or prompt_attention_mask.ndim != 2
            or prompt_attention_mask.shape[0] != batch_size
            or prompt_attention_mask.shape[1] <= 0
        ):
            message = "Prompt attention mask must have shape [batch, prompt_sequence]"
            raise ModuleLatentBridgeError(message)
        if prompt_attention_mask.dtype not in integer_dtypes:
            message = "Prompt attention mask must use a boolean or integer dtype"
            raise ModuleLatentBridgeError(message)
        if prompt_attention_mask.device != device:
            message = "Prompt attention mask must share the module-state device"
            raise ModuleLatentBridgeError(message)
        if not bool(torch.logical_or(prompt_attention_mask == 0, prompt_attention_mask == 1).all()):
            message = "Prompt attention mask must contain only zeros and ones"
            raise ModuleLatentBridgeError(message)
        if not bool(prompt_attention_mask.any(dim=1).all()):
            message = "Every prompt must contain at least one attended token"
            raise ModuleLatentBridgeError(message)
        canonical = prompt_attention_mask.to(dtype=torch.bool)
        transitions = canonical[:, 1:].to(dtype=torch.int8) - canonical[:, :-1].to(dtype=torch.int8)
        contiguous_left = (transitions >= 0).all(dim=1)
        contiguous_right = (transitions <= 0).all(dim=1)
        if not bool(torch.logical_or(contiguous_left, contiguous_right).all()):
            message = "Prompt attention mask must use contiguous left or right padding"
            raise ModuleLatentBridgeError(message)
        return canonical.detach().clone()

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = "ModuleLatentBridge parameters must share input dtype and device"
                raise ModuleLatentBridgeError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"ModuleLatentBridge parameter {name!r} contains NaN or Inf"
                raise ModuleLatentBridgeError(message)

    def _validate_output(
        self,
        output: ModuleLatentBridgeOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        expected_position_ids: torch.Tensor,
    ) -> None:
        if tuple(output.module_embeddings.shape) != (
            batch_size,
            self.num_modules,
            self.backbone_hidden_size,
        ):
            message = "Module embeddings have an invalid shape"
            raise ModuleLatentBridgeError(message)
        if (
            output.module_embeddings.dtype != dtype
            or output.module_embeddings.device != device
            or not bool(torch.isfinite(output.module_embeddings).all())
        ):
            message = "Module embeddings have an invalid dtype, device, or value"
            raise ModuleLatentBridgeError(message)
        expected_matrix = (batch_size, self.num_modules)
        if (
            tuple(output.module_attention_mask.shape) != expected_matrix
            or output.module_attention_mask.dtype != torch.bool
            or output.module_attention_mask.device != device
            or not bool(output.module_attention_mask.all())
        ):
            message = "Every fixed module latent token must be attended"
            raise ModuleLatentBridgeError(message)
        if (
            tuple(output.module_position_ids.shape) != expected_matrix
            or output.module_position_ids.dtype != torch.int64
            or output.module_position_ids.device != device
            or not torch.equal(output.module_position_ids, expected_position_ids)
        ):
            message = "Module position IDs do not follow the logical prompt"
            raise ModuleLatentBridgeError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor

    def _stable_layer_norm(
        self,
        tensor: torch.Tensor,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        normalize = cast(
            Callable[[torch.Tensor, float], torch.Tensor],
            _StableLayerNormalization.apply,
        )
        normalized = normalize(tensor, self.normalization.eps)
        return normalized * weight + bias


__all__ = ["ModuleLatentBridge", "ModuleLatentBridgeError"]
