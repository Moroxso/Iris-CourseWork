"""Inference `[prompt][module states]` prefix for no-workspace arm B."""

import torch
from torch import nn

from iris.cognitive import ModuleLatentBridgeOutput
from iris.generation.outputs import SecondPassGenerationPrefixOutput


class ModuleGenerationPrefixError(RuntimeError):
    """Raised when prompt and module latent tokens cannot form a cached prefix."""


class ModuleGenerationPrefixAssembler(nn.Module):
    """Assemble one arm-B prefix without constructing any workspace state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_modules: int,
        max_prompt_tokens: int,
        max_new_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_modules": num_modules,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "max_context_tokens": max_context_tokens,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if max_prompt_tokens + num_modules + max_new_tokens > max_context_tokens:
            message = "Configured prompt, module, and generation maxima exceed context"
            raise ValueError(message)
        self.hidden_size = hidden_size
        self.num_modules = num_modules
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        modules: ModuleLatentBridgeOutput,
    ) -> SecondPassGenerationPrefixOutput:
        """Return a detached single-request `[prompt][modules]` prefix."""
        prompt_length, dtype, device = self._validate_prompt_embeddings(prompt_embeddings)
        prompt_mask = self._validate_prompt_mask(
            prompt_attention_mask,
            prompt_length=prompt_length,
            device=device,
        )
        prompt_logical_length = prompt_mask.to(dtype=torch.int64).sum(dim=-1)
        self._validate_modules(
            modules,
            dtype=dtype,
            device=device,
            prompt_logical_length=prompt_logical_length,
        )
        total_length = prompt_length + self.num_modules
        if total_length + self.max_new_tokens > self.max_context_tokens:
            message = "Module generation prefix plus output exceeds the context window"
            raise ModuleGenerationPrefixError(message)

        inputs_embeds = torch.cat(
            (
                self._materialize(prompt_embeddings),
                self._materialize(modules.module_embeddings),
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (prompt_mask, modules.module_attention_mask.detach().clone()),
            dim=1,
        )
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        if not torch.equal(position_ids[:, prompt_length:], modules.module_position_ids):
            message = "Module positions disagree with the generation prefix"
            raise ModuleGenerationPrefixError(message)
        if not bool((position_ids + self.max_new_tokens <= self.max_context_tokens).all()):
            message = "Module generation positions exceed the context window"
            raise ModuleGenerationPrefixError(message)
        output = SecondPassGenerationPrefixOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        self._validate_output(output, total_length=total_length, dtype=dtype, device=device)
        return output

    def _validate_prompt_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if type(embeddings) is not torch.Tensor or embeddings.ndim != 3:
            message = "Prompt embeddings must have shape [1, prompt, hidden_size]"
            raise ModuleGenerationPrefixError(message)
        batch_size, prompt_length, hidden_size = embeddings.shape
        if (
            batch_size != 1
            or prompt_length <= 0
            or prompt_length > self.max_prompt_tokens
            or hidden_size != self.hidden_size
        ):
            message = "Prompt embedding shape exceeds the module-prefix contract"
            raise ModuleGenerationPrefixError(message)
        self._validate_embeddings(embeddings, name="Prompt")
        return prompt_length, embeddings.dtype, embeddings.device

    @staticmethod
    def _validate_prompt_mask(
        mask: torch.Tensor,
        *,
        prompt_length: int,
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
        if type(mask) is not torch.Tensor or tuple(mask.shape) != (1, prompt_length):
            message = "Prompt attention mask has an invalid shape"
            raise ModuleGenerationPrefixError(message)
        if mask.dtype not in integer_dtypes or mask.device != device:
            message = "Prompt attention mask has an invalid dtype or device"
            raise ModuleGenerationPrefixError(message)
        if not bool(torch.logical_or(mask == 0, mask == 1).all()) or not bool(mask.any()):
            message = "Prompt attention mask must be nonempty and binary"
            raise ModuleGenerationPrefixError(message)
        canonical = mask.detach().clone().to(dtype=torch.bool)
        transitions = canonical[:, 1:].to(dtype=torch.int8) - canonical[:, :-1].to(dtype=torch.int8)
        contiguous_left = bool((transitions >= 0).all())
        contiguous_right = bool((transitions <= 0).all())
        if not (contiguous_left or contiguous_right):
            message = "Prompt attention must use contiguous left or right padding"
            raise ModuleGenerationPrefixError(message)
        return canonical

    def _validate_modules(
        self,
        modules: ModuleLatentBridgeOutput,
        *,
        dtype: torch.dtype,
        device: torch.device,
        prompt_logical_length: torch.Tensor,
    ) -> None:
        if type(modules) is not ModuleLatentBridgeOutput:
            message = "Module prefix must be exactly ModuleLatentBridgeOutput"
            raise ModuleGenerationPrefixError(message)
        embeddings = modules.module_embeddings
        if (
            type(embeddings) is not torch.Tensor
            or tuple(embeddings.shape) != (1, self.num_modules, self.hidden_size)
            or embeddings.dtype != dtype
            or embeddings.device != device
        ):
            message = "Module embeddings have an invalid shape, dtype, or device"
            raise ModuleGenerationPrefixError(message)
        self._validate_embeddings(embeddings, name="Module")
        expected_matrix = (1, self.num_modules)
        mask = modules.module_attention_mask
        if (
            type(mask) is not torch.Tensor
            or tuple(mask.shape) != expected_matrix
            or mask.dtype != torch.bool
            or mask.device != device
            or not bool(mask.all())
        ):
            message = "Every module latent token must be attended"
            raise ModuleGenerationPrefixError(message)
        expected_positions = prompt_logical_length.unsqueeze(1) + torch.arange(
            self.num_modules,
            dtype=torch.int64,
            device=device,
        )
        positions = modules.module_position_ids
        if (
            type(positions) is not torch.Tensor
            or tuple(positions.shape) != expected_matrix
            or positions.dtype != torch.int64
            or positions.device != device
            or not torch.equal(positions, expected_positions)
        ):
            message = "Module position IDs must follow the logical prompt"
            raise ModuleGenerationPrefixError(message)

    @staticmethod
    def _validate_embeddings(embeddings: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(embeddings):
            message = f"{name} embeddings must use a floating dtype"
            raise ModuleGenerationPrefixError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = f"{name} embeddings must not track gradients during generation"
            raise ModuleGenerationPrefixError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = f"{name} embeddings cannot be inspected on {embeddings.device}"
            raise ModuleGenerationPrefixError(message) from error
        if not finite:
            message = f"{name} embeddings contain NaN or Inf"
            raise ModuleGenerationPrefixError(message)

    def _validate_output(
        self,
        output: SecondPassGenerationPrefixOutput,
        *,
        total_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            tuple(output.inputs_embeds.shape) != (1, total_length, self.hidden_size)
            or output.inputs_embeds.dtype != dtype
            or output.inputs_embeds.device != device
            or output.inputs_embeds.requires_grad
            or output.inputs_embeds.grad_fn is not None
            or not bool(torch.isfinite(output.inputs_embeds).all())
        ):
            message = "Module generation prefix embeddings are invalid"
            raise ModuleGenerationPrefixError(message)
        expected_matrix = (1, total_length)
        if (
            tuple(output.attention_mask.shape) != expected_matrix
            or output.attention_mask.dtype != torch.bool
            or output.attention_mask.device != device
            or tuple(output.position_ids.shape) != expected_matrix
            or output.position_ids.dtype != torch.int64
            or output.position_ids.device != device
        ):
            message = "Module generation prefix causal metadata is invalid"
            raise ModuleGenerationPrefixError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor


__all__ = ["ModuleGenerationPrefixAssembler", "ModuleGenerationPrefixError"]
