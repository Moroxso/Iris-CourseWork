"""Inference-only `[prompt][workspace]` prefix assembly for Iris generation."""

import torch
from torch import nn

from iris.generation.outputs import SecondPassGenerationPrefixOutput
from iris.workspace.outputs import LatentBridgeOutput


class SecondPassGenerationPrefixError(RuntimeError):
    """Raised when a frozen prompt and latent workspace cannot form a prefix."""


class SecondPassGenerationPrefixAssembler(nn.Module):
    """Assemble one cached generation prefix without hiding gradient detaches."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_slots: int,
        max_prompt_tokens: int,
        max_new_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_slots": num_slots,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "max_context_tokens": max_context_tokens,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if max_prompt_tokens + num_slots + max_new_tokens > max_context_tokens:
            message = "Configured prompt, workspace, and generation maxima exceed context"
            raise ValueError(message)

        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        workspace: LatentBridgeOutput,
    ) -> SecondPassGenerationPrefixOutput:
        """Return a detached single-request prefix with explicit causal metadata."""
        prompt_length, dtype, device = self._validate_prompt_embeddings(prompt_embeddings)
        prompt_mask = self._validate_prompt_mask(
            prompt_attention_mask,
            prompt_length=prompt_length,
            device=device,
        )
        prompt_logical_length = prompt_mask.to(dtype=torch.int64).sum(dim=-1)
        self._validate_workspace(
            workspace,
            dtype=dtype,
            device=device,
            prompt_logical_length=prompt_logical_length,
        )
        total_length = prompt_length + self.num_slots
        if total_length + self.max_new_tokens > self.max_context_tokens:
            message = "Generation prefix plus requested output exceeds the context window"
            raise SecondPassGenerationPrefixError(message)

        prompt_values = self._materialize(prompt_embeddings)
        workspace_values = self._materialize(workspace.workspace_embeddings)
        inputs_embeds = torch.cat((prompt_values, workspace_values), dim=1)
        attention_mask = torch.cat(
            (prompt_mask, workspace.workspace_attention_mask.detach().clone()),
            dim=1,
        )
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        if not torch.equal(position_ids[:, prompt_length:], workspace.workspace_position_ids):
            message = "Workspace positions disagree with the generation prefix"
            raise SecondPassGenerationPrefixError(message)
        if not bool((position_ids + self.max_new_tokens <= self.max_context_tokens).all()):
            message = "Generation positions exceed the context window"
            raise SecondPassGenerationPrefixError(message)

        output = SecondPassGenerationPrefixOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        self._validate_output(
            output,
            total_length=total_length,
            dtype=dtype,
            device=device,
        )
        return output

    def _validate_prompt_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            message = "Prompt embeddings must have shape [1, prompt, hidden_size]"
            raise SecondPassGenerationPrefixError(message)
        batch_size, prompt_length, hidden_size = embeddings.shape
        if (
            batch_size != 1
            or prompt_length <= 0
            or prompt_length > self.max_prompt_tokens
            or hidden_size != self.hidden_size
        ):
            message = "Prompt embedding shape exceeds the generation-prefix contract"
            raise SecondPassGenerationPrefixError(message)
        self._validate_inference_embeddings(embeddings, name="Prompt")
        return prompt_length, embeddings.dtype, embeddings.device

    def _validate_prompt_mask(
        self,
        mask: torch.Tensor,
        *,
        prompt_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompt_mask = self._validate_binary_mask(
            mask,
            expected_shape=(1, prompt_length),
            device=device,
            name="Prompt",
        )
        transitions = prompt_mask[:, 1:].to(dtype=torch.int8) - prompt_mask[:, :-1].to(
            dtype=torch.int8
        )
        contiguous_left = (transitions >= 0).all(dim=1)
        contiguous_right = (transitions <= 0).all(dim=1)
        if not bool(torch.logical_or(contiguous_left, contiguous_right).all()):
            message = "Prompt attention must use contiguous left or right padding"
            raise SecondPassGenerationPrefixError(message)
        return prompt_mask

    @staticmethod
    def _validate_binary_mask(
        mask: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected_shape:
            message = f"{name} attention mask has an invalid shape"
            raise SecondPassGenerationPrefixError(message)
        if mask.dtype not in integer_dtypes or mask.device != device:
            message = f"{name} attention mask has an invalid dtype or device"
            raise SecondPassGenerationPrefixError(message)
        if not bool(torch.logical_or(mask == 0, mask == 1).all()):
            message = f"{name} attention mask must contain only zeros and ones"
            raise SecondPassGenerationPrefixError(message)
        if not bool(mask.any(dim=1).all()):
            message = f"Every {name.lower()} must contain an attended token"
            raise SecondPassGenerationPrefixError(message)
        return mask.detach().clone().to(dtype=torch.bool)

    def _validate_workspace(
        self,
        workspace: LatentBridgeOutput,
        *,
        dtype: torch.dtype,
        device: torch.device,
        prompt_logical_length: torch.Tensor,
    ) -> None:
        if not isinstance(workspace, LatentBridgeOutput):
            message = "Workspace prefix must be LatentBridgeOutput"
            raise SecondPassGenerationPrefixError(message)
        embeddings = workspace.workspace_embeddings
        expected_embeddings = (1, self.num_slots, self.hidden_size)
        if (
            not isinstance(embeddings, torch.Tensor)
            or tuple(embeddings.shape) != expected_embeddings
        ):
            message = "Workspace embeddings have an invalid shape"
            raise SecondPassGenerationPrefixError(message)
        if embeddings.dtype != dtype or embeddings.device != device:
            message = "Prompt and workspace embeddings must share one dtype and device"
            raise SecondPassGenerationPrefixError(message)
        self._validate_inference_embeddings(embeddings, name="Workspace")

        expected_matrix = (1, self.num_slots)
        workspace_mask = workspace.workspace_attention_mask
        if (
            not isinstance(workspace_mask, torch.Tensor)
            or tuple(workspace_mask.shape) != expected_matrix
            or workspace_mask.dtype != torch.bool
            or workspace_mask.device != device
            or not bool(workspace_mask.all())
        ):
            message = "Every fixed workspace token must be attended"
            raise SecondPassGenerationPrefixError(message)
        positions = workspace.workspace_position_ids
        expected_positions = prompt_logical_length.unsqueeze(1) + torch.arange(
            self.num_slots,
            dtype=torch.int64,
            device=device,
        )
        if (
            not isinstance(positions, torch.Tensor)
            or tuple(positions.shape) != expected_matrix
            or positions.dtype != torch.int64
            or positions.device != device
            or not torch.equal(positions, expected_positions)
        ):
            message = "Workspace position IDs must follow the logical prompt"
            raise SecondPassGenerationPrefixError(message)

    @staticmethod
    def _validate_inference_embeddings(embeddings: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(embeddings):
            message = f"{name} embeddings must use a floating dtype"
            raise SecondPassGenerationPrefixError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = f"{name} embeddings cannot be inspected on {embeddings.device}"
            raise SecondPassGenerationPrefixError(message) from error
        if not finite:
            message = f"{name} embeddings contain NaN or Inf"
            raise SecondPassGenerationPrefixError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = f"{name} embeddings must not track gradients during generation"
            raise SecondPassGenerationPrefixError(message)

    def _validate_output(
        self,
        output: SecondPassGenerationPrefixOutput,
        *,
        total_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        expected_embeddings = (1, total_length, self.hidden_size)
        if tuple(output.inputs_embeds.shape) != expected_embeddings:
            message = "Generation-prefix embeddings have an invalid shape"
            raise SecondPassGenerationPrefixError(message)
        if output.inputs_embeds.dtype != dtype or output.inputs_embeds.device != device:
            message = "Generation-prefix embeddings changed dtype or device"
            raise SecondPassGenerationPrefixError(message)
        if output.inputs_embeds.requires_grad or output.inputs_embeds.grad_fn is not None:
            message = "Generation-prefix embeddings unexpectedly track gradients"
            raise SecondPassGenerationPrefixError(message)
        if not bool(torch.isfinite(output.inputs_embeds).all()):
            message = "Generation-prefix embeddings contain NaN or Inf"
            raise SecondPassGenerationPrefixError(message)
        expected_matrix = (1, total_length)
        if (
            tuple(output.attention_mask.shape) != expected_matrix
            or output.attention_mask.dtype != torch.bool
            or output.attention_mask.device != device
        ):
            message = "Generation-prefix attention mask is invalid"
            raise SecondPassGenerationPrefixError(message)
        if (
            tuple(output.position_ids.shape) != expected_matrix
            or output.position_ids.dtype != torch.int64
            or output.position_ids.device != device
        ):
            message = "Generation-prefix position IDs are invalid"
            raise SecondPassGenerationPrefixError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor
