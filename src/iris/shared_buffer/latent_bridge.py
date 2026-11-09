"""Projection of the final all-buffer read into token-matched arm-C latents."""

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import nn

from iris.workspace.outputs import LatentBridgeOutput, WorkspaceReaderOutput


class _StableLayerNormalization(torch.autograd.Function):
    """Layer normalization with scaled forward and explicit stable backward."""

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
        denominator = torch.hypot(scaled_standard_deviation, epsilon_scale / scale)
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


class SharedBufferLatentBridgeError(RuntimeError):
    """Raised when the final complete-buffer read cannot become latent tokens."""


class SharedBufferLatentBridge(nn.Module):
    """Project M final reader summaries into M token-matched backbone latents."""

    def __init__(
        self,
        *,
        state_dim: int,
        backbone_hidden_size: int,
        num_modules: int,
        cycles: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "backbone_hidden_size": backbone_hidden_size,
            "num_modules": num_modules,
            "cycles": cycles,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.num_modules = num_modules
        self.cycles = cycles
        self.final_entry_count = num_modules * cycles
        self.bridge_projection = nn.Linear(state_dim, backbone_hidden_size, bias=False)
        self.normalization = nn.LayerNorm(backbone_hidden_size)
        self.shared_buffer_type_embedding = nn.Parameter(torch.empty(backbone_hidden_size))
        nn.init.normal_(
            self.shared_buffer_type_embedding,
            mean=0.0,
            std=backbone_hidden_size**-0.5,
        )

    def forward(
        self,
        final_read: WorkspaceReaderOutput,
        prompt_attention_mask: torch.Tensor,
    ) -> LatentBridgeOutput:
        """Require an all-entry final read and return exactly M latent tokens."""
        batch_size, dtype, device = self._validate_final_read(final_read)
        prompt_mask = self._validate_prompt_attention_mask(
            prompt_attention_mask,
            batch_size=batch_size,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)
        reads = self._materialize(final_read.workspace_reads)
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

        with torch.autocast(device_type=device.type, enabled=False):
            projected = functional.linear(
                reads.to(dtype=accumulation_dtype),
                self.bridge_projection.weight.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(projected).all()):
                message = "Projected shared-buffer read embeddings contain NaN or Inf"
                raise SharedBufferLatentBridgeError(message)
            normalized = self._stable_layer_norm(
                projected,
                weight=self.normalization.weight.to(dtype=accumulation_dtype),
                bias=self.normalization.bias.to(dtype=accumulation_dtype),
            )
            latent_embeddings = normalized + self.shared_buffer_type_embedding.to(
                dtype=accumulation_dtype
            )
            if not bool(torch.isfinite(latent_embeddings).all()):
                message = "Latent shared-buffer read embeddings contain NaN or Inf"
                raise SharedBufferLatentBridgeError(message)
            workspace_embeddings = latent_embeddings.to(dtype=dtype)

        if not bool(torch.isfinite(workspace_embeddings).all()):
            message = "Latent shared-buffer embeddings contain NaN or Inf after storage cast"
            raise SharedBufferLatentBridgeError(message)
        workspace_attention_mask = torch.ones(
            (batch_size, self.num_modules),
            dtype=torch.bool,
            device=device,
        )
        prompt_lengths = prompt_mask.to(dtype=torch.int64).sum(dim=-1, keepdim=True)
        workspace_position_ids = prompt_lengths + torch.arange(
            self.num_modules,
            dtype=torch.int64,
            device=device,
        )
        output = LatentBridgeOutput(
            workspace_embeddings=workspace_embeddings,
            workspace_attention_mask=workspace_attention_mask,
            workspace_position_ids=workspace_position_ids,
        )
        self._validate_output(
            output,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            expected_position_ids=workspace_position_ids,
        )
        return output

    def _validate_final_read(
        self,
        final_read: WorkspaceReaderOutput,
    ) -> tuple[int, torch.dtype, torch.device]:
        if type(final_read) is not WorkspaceReaderOutput:
            message = "Shared-buffer latent bridge requires exact WorkspaceReaderOutput"
            raise SharedBufferLatentBridgeError(message)
        reads = final_read.workspace_reads
        if type(reads) is not torch.Tensor or reads.ndim != 3:
            message = "Final shared-buffer reads must have shape [batch, modules, state_dim]"
            raise SharedBufferLatentBridgeError(message)
        batch_size = reads.shape[0]
        if batch_size <= 0 or tuple(reads.shape) != (
            batch_size,
            self.num_modules,
            self.state_dim,
        ):
            message = "Final shared-buffer read shape does not match the bridge contract"
            raise SharedBufferLatentBridgeError(message)
        if not torch.is_floating_point(reads) or not bool(torch.isfinite(reads).all()):
            message = "Final shared-buffer reads must be finite floating tensors"
            raise SharedBufferLatentBridgeError(message)
        attention = final_read.attention_weights
        expected_attention_shape = (
            batch_size,
            self.num_modules,
            self.final_entry_count,
        )
        accumulation_dtype = (
            torch.float32 if reads.dtype in {torch.float16, torch.bfloat16} else reads.dtype
        )
        if (
            type(attention) is not torch.Tensor
            or tuple(attention.shape) != expected_attention_shape
            or attention.dtype != accumulation_dtype
            or attention.device != reads.device
            or not bool(torch.isfinite(attention).all())
            or not bool((attention >= 0).all())
        ):
            message = "Final read attention must cover every configured buffer entry"
            raise SharedBufferLatentBridgeError(message)
        if not torch.allclose(
            attention.sum(dim=-1),
            torch.ones(
                (batch_size, self.num_modules),
                dtype=attention.dtype,
                device=attention.device,
            ),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Final all-buffer attention must sum to one"
            raise SharedBufferLatentBridgeError(message)
        return batch_size, reads.dtype, reads.device

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
        if type(prompt_attention_mask) is not torch.Tensor:
            message = "Prompt attention mask must be a tensor"
            raise SharedBufferLatentBridgeError(message)
        if prompt_attention_mask.dtype not in integer_dtypes:
            message = "Prompt attention mask must use a boolean or integer dtype"
            raise SharedBufferLatentBridgeError(message)
        if prompt_attention_mask.ndim != 2 or prompt_attention_mask.shape[0] != batch_size:
            message = "Prompt attention mask must have shape [batch, prompt_sequence]"
            raise SharedBufferLatentBridgeError(message)
        if prompt_attention_mask.shape[1] <= 0 or prompt_attention_mask.device != device:
            message = "Prompt attention mask has an invalid length or device"
            raise SharedBufferLatentBridgeError(message)
        if not bool(torch.logical_or(prompt_attention_mask == 0, prompt_attention_mask == 1).all()):
            message = "Prompt attention mask must contain only zeros and ones"
            raise SharedBufferLatentBridgeError(message)
        if not bool(prompt_attention_mask.any(dim=1).all()):
            message = "Every prompt must contain at least one attended token"
            raise SharedBufferLatentBridgeError(message)
        canonical_mask = prompt_attention_mask.to(dtype=torch.bool)
        transitions = canonical_mask[:, 1:].to(dtype=torch.int8) - canonical_mask[:, :-1].to(
            dtype=torch.int8
        )
        contiguous_left = (transitions >= 0).all(dim=1)
        contiguous_right = (transitions <= 0).all(dim=1)
        if not bool(torch.logical_or(contiguous_left, contiguous_right).all()):
            message = "Prompt attention mask must use contiguous left or right padding"
            raise SharedBufferLatentBridgeError(message)
        return canonical_mask.detach().clone()

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = "SharedBufferLatentBridge parameters must match read dtype and device"
                raise SharedBufferLatentBridgeError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"SharedBufferLatentBridge parameter {name!r} contains NaN or Inf"
                raise SharedBufferLatentBridgeError(message)

    def _validate_output(
        self,
        output: LatentBridgeOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        expected_position_ids: torch.Tensor,
    ) -> None:
        if tuple(output.workspace_embeddings.shape) != (
            batch_size,
            self.num_modules,
            self.backbone_hidden_size,
        ):
            message = "Shared-buffer latent embeddings have an invalid shape"
            raise SharedBufferLatentBridgeError(message)
        if (
            output.workspace_embeddings.dtype != dtype
            or output.workspace_embeddings.device != device
            or not bool(torch.isfinite(output.workspace_embeddings).all())
        ):
            message = "Shared-buffer embeddings changed dtype, device, or value"
            raise SharedBufferLatentBridgeError(message)
        expected_shape = (batch_size, self.num_modules)
        if (
            tuple(output.workspace_attention_mask.shape) != expected_shape
            or output.workspace_attention_mask.dtype != torch.bool
            or output.workspace_attention_mask.device != device
            or not bool(output.workspace_attention_mask.all())
        ):
            message = "Every token-matched shared-buffer latent must be attended"
            raise SharedBufferLatentBridgeError(message)
        if (
            tuple(output.workspace_position_ids.shape) != expected_shape
            or output.workspace_position_ids.dtype != torch.int64
            or output.workspace_position_ids.device != device
            or not torch.equal(output.workspace_position_ids, expected_position_ids)
        ):
            message = "Shared-buffer position IDs must follow the logical prompt"
            raise SharedBufferLatentBridgeError(message)

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
