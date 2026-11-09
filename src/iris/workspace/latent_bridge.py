"""Projection of functional workspace state into frozen-backbone latent tokens."""

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import nn

from iris.workspace.outputs import LatentBridgeOutput
from iris.workspace.state import WorkspaceState


class _StableLayerNormalization(torch.autograd.Function):
    """Layer normalization with a scaled forward and explicit stable backward."""

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


class LatentBridgeError(RuntimeError):
    """Raised when workspace state cannot satisfy the latent-prefix contract."""


class LatentBridge(nn.Module):
    """Map every canonical workspace slot from cognitive width to LLM width."""

    def __init__(
        self,
        *,
        state_dim: int,
        backbone_hidden_size: int,
        num_slots: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "backbone_hidden_size": backbone_hidden_size,
            "num_slots": num_slots,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)

        self.state_dim = state_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.num_slots = num_slots
        self.bridge_projection = nn.Linear(state_dim, backbone_hidden_size, bias=False)
        self.normalization = nn.LayerNorm(backbone_hidden_size)
        self.workspace_type_embedding = nn.Parameter(torch.empty(backbone_hidden_size))
        nn.init.normal_(
            self.workspace_type_embedding,
            mean=0.0,
            std=backbone_hidden_size**-0.5,
        )

    def forward(
        self,
        workspace: WorkspaceState,
        prompt_attention_mask: torch.Tensor,
    ) -> LatentBridgeOutput:
        """Return a fixed-width latent prefix positioned after each logical prompt."""
        batch_size, dtype, device = self._validate_workspace(workspace)
        prompt_mask = self._validate_prompt_attention_mask(
            prompt_attention_mask,
            batch_size=batch_size,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        slots = self._materialize(workspace.slots)
        occupancy = self._materialize(workspace.occupancy)
        canonical_slots = torch.where(
            occupancy.unsqueeze(-1),
            slots,
            torch.zeros_like(slots),
        )
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

        with torch.autocast(device_type=device.type, enabled=False):
            projected = functional.linear(
                canonical_slots.to(dtype=accumulation_dtype),
                self.bridge_projection.weight.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(projected).all()):
                message = "Projected workspace embeddings contain NaN or Inf"
                raise LatentBridgeError(message)
            normalized = self._stable_layer_norm(
                projected,
                weight=self.normalization.weight.to(dtype=accumulation_dtype),
                bias=self.normalization.bias.to(dtype=accumulation_dtype),
            )
            latent_embeddings = normalized + self.workspace_type_embedding.to(
                dtype=accumulation_dtype
            )
            if not bool(torch.isfinite(latent_embeddings).all()):
                message = "Latent workspace embeddings contain NaN or Inf"
                raise LatentBridgeError(message)
            workspace_embeddings = latent_embeddings.to(dtype=dtype)

        if not bool(torch.isfinite(workspace_embeddings).all()):
            message = "Latent workspace embeddings contain NaN or Inf after storage cast"
            raise LatentBridgeError(message)

        workspace_attention_mask = torch.ones(
            (batch_size, self.num_slots),
            dtype=torch.bool,
            device=device,
        )
        prompt_lengths = prompt_mask.to(dtype=torch.int64).sum(dim=-1, keepdim=True)
        workspace_position_ids = prompt_lengths + torch.arange(
            self.num_slots,
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

    def _validate_workspace(
        self,
        workspace: WorkspaceState,
    ) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(workspace, WorkspaceState):
            message = "Latent bridge input must be WorkspaceState"
            raise LatentBridgeError(message)
        slots = workspace.slots
        if not isinstance(slots, torch.Tensor) or slots.ndim != 3:
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise LatentBridgeError(message)
        batch_size = slots.shape[0]
        if batch_size <= 0 or tuple(slots.shape) != (
            batch_size,
            self.num_slots,
            self.state_dim,
        ):
            message = "Workspace slot shape does not match the latent bridge contract"
            raise LatentBridgeError(message)
        if not torch.is_floating_point(slots):
            message = "Workspace slots must use a floating dtype"
            raise LatentBridgeError(message)
        if not bool(torch.isfinite(slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise LatentBridgeError(message)

        matrix_shape = (batch_size, self.num_slots)
        expected_metadata: tuple[tuple[str, object, torch.dtype], ...] = (
            ("age", workspace.age, torch.int64),
            ("occupancy", workspace.occupancy, torch.bool),
            ("source_module", workspace.source_module, torch.int64),
            ("confidence", workspace.confidence, slots.dtype),
            ("ttl", workspace.ttl, slots.dtype),
            ("write_count", workspace.write_count, torch.int64),
        )
        for name, value, expected_dtype in expected_metadata:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != matrix_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise LatentBridgeError(message)
            if value.dtype != expected_dtype:
                message = f"Workspace {name} has an invalid dtype"
                raise LatentBridgeError(message)
            if value.device != slots.device:
                message = "Workspace state tensors must share one device"
                raise LatentBridgeError(message)

        if not bool((workspace.age >= 0).all()) or not bool((workspace.write_count >= 0).all()):
            message = "Workspace counters must be non-negative"
            raise LatentBridgeError(message)
        if not bool((workspace.source_module >= -1).all()):
            message = "Workspace source_module must be -1 or non-negative"
            raise LatentBridgeError(message)
        if not bool(torch.isfinite(workspace.confidence).all()) or not bool(
            torch.logical_and(workspace.confidence >= 0, workspace.confidence <= 1).all()
        ):
            message = "Workspace confidence must be finite and lie in [0, 1]"
            raise LatentBridgeError(message)
        if not bool(torch.isfinite(workspace.ttl).all()) or not bool((workspace.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise LatentBridgeError(message)
        if not bool((workspace.ttl.masked_select(~workspace.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise LatentBridgeError(message)
        occupied = workspace.occupancy
        if not bool((workspace.source_module.masked_select(occupied) >= 0).all()):
            message = "Occupied workspace slots must identify a source module"
            raise LatentBridgeError(message)
        empty = ~occupied
        for name, tensor, expected in (
            ("slots", workspace.slots, 0),
            ("age", workspace.age, 0),
            ("source_module", workspace.source_module, -1),
            ("confidence", workspace.confidence, 0),
            ("ttl", workspace.ttl, 0),
            ("write_count", workspace.write_count, 0),
        ):
            mask = empty.unsqueeze(-1) if name == "slots" else empty
            if not bool((tensor.masked_select(mask) == expected).all()):
                message = f"Unoccupied workspace slots must have canonical {name}"
                raise LatentBridgeError(message)
        return batch_size, slots.dtype, slots.device

    @staticmethod
    def _validate_prompt_attention_mask(
        prompt_attention_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(prompt_attention_mask, torch.Tensor):
            message = "Prompt attention mask must be a tensor"
            raise LatentBridgeError(message)
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if prompt_attention_mask.dtype not in integer_dtypes:
            message = "Prompt attention mask must use a boolean or integer dtype"
            raise LatentBridgeError(message)
        if prompt_attention_mask.ndim != 2 or prompt_attention_mask.shape[0] != batch_size:
            message = "Prompt attention mask must have shape [batch, prompt_sequence]"
            raise LatentBridgeError(message)
        if prompt_attention_mask.shape[1] <= 0:
            message = "Prompt attention mask must have a non-empty sequence axis"
            raise LatentBridgeError(message)
        if prompt_attention_mask.device != device:
            message = "Prompt attention mask must share the workspace device"
            raise LatentBridgeError(message)
        if not bool(torch.logical_or(prompt_attention_mask == 0, prompt_attention_mask == 1).all()):
            message = "Prompt attention mask must contain only zeros and ones"
            raise LatentBridgeError(message)
        if not bool(prompt_attention_mask.any(dim=1).all()):
            message = "Every prompt must contain at least one attended token"
            raise LatentBridgeError(message)
        canonical_mask = prompt_attention_mask.to(dtype=torch.bool)
        transitions = canonical_mask[:, 1:].to(dtype=torch.int8) - canonical_mask[:, :-1].to(
            dtype=torch.int8
        )
        contiguous_left = (transitions >= 0).all(dim=1)
        contiguous_right = (transitions <= 0).all(dim=1)
        if not bool(torch.logical_or(contiguous_left, contiguous_right).all()):
            message = "Prompt attention mask must use contiguous left or right padding"
            raise LatentBridgeError(message)
        return canonical_mask.detach().clone()

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = "LatentBridge parameters must share the workspace dtype and device"
                raise LatentBridgeError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"LatentBridge parameter {name} contains NaN or Inf"
                raise LatentBridgeError(message)

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
            self.num_slots,
            self.backbone_hidden_size,
        ):
            message = "Workspace embeddings have an invalid shape"
            raise LatentBridgeError(message)
        if (
            output.workspace_embeddings.dtype != dtype
            or output.workspace_embeddings.device != device
        ):
            message = "Workspace embeddings must preserve the workspace dtype and device"
            raise LatentBridgeError(message)
        if not bool(torch.isfinite(output.workspace_embeddings).all()):
            message = "Workspace embeddings contain NaN or Inf"
            raise LatentBridgeError(message)
        expected_matrix_shape = (batch_size, self.num_slots)
        if (
            tuple(output.workspace_attention_mask.shape) != expected_matrix_shape
            or output.workspace_attention_mask.dtype != torch.bool
            or output.workspace_attention_mask.device != device
            or not bool(output.workspace_attention_mask.all())
        ):
            message = "Every fixed workspace latent token must be attended"
            raise LatentBridgeError(message)
        if (
            tuple(output.workspace_position_ids.shape) != expected_matrix_shape
            or output.workspace_position_ids.dtype != torch.int64
            or output.workspace_position_ids.device != device
            or not torch.equal(output.workspace_position_ids, expected_position_ids)
        ):
            message = "Workspace position IDs do not follow the logical prompt"
            raise LatentBridgeError(message)

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
        """Apply algebraic LayerNorm with finite gradients at extreme magnitudes."""
        normalize = cast(
            Callable[[torch.Tensor, float], torch.Tensor],
            _StableLayerNormalization.apply,
        )
        normalized = normalize(tensor, self.normalization.eps)
        return normalized * weight + bias
