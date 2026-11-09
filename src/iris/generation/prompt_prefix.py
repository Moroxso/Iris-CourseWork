"""Prompt-only cached-generation prefix for the frozen arm-A baseline."""

import torch
from torch import nn

from iris.generation.outputs import SecondPassGenerationPrefixOutput


class PromptOnlyGenerationPrefixError(RuntimeError):
    """Raised when a frozen prompt cannot form the arm-A generation prefix."""


class PromptOnlyGenerationPrefixAssembler(nn.Module):
    """Build one validated `[prompt]` prefix without workspace latent tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        max_prompt_tokens: int,
        max_new_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "max_context_tokens": max_context_tokens,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if max_prompt_tokens + max_new_tokens > max_context_tokens:
            message = "Configured prompt and generation maxima exceed context"
            raise ValueError(message)

        self.hidden_size = hidden_size
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> SecondPassGenerationPrefixOutput:
        """Return an owned detached prefix for one prompt-only cached decode."""
        prompt_length, dtype, device = self._validate_embeddings(prompt_embeddings)
        attention_mask = self._validate_attention_mask(
            prompt_attention_mask,
            prompt_length=prompt_length,
            device=device,
        )
        if prompt_length + self.max_new_tokens > self.max_context_tokens:
            message = "Prompt prefix plus requested output exceeds the context window"
            raise PromptOnlyGenerationPrefixError(message)

        inputs_embeds = self._materialize(prompt_embeddings)
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        if not bool((position_ids + self.max_new_tokens <= self.max_context_tokens).all()):
            message = "Prompt generation positions exceed the context window"
            raise PromptOnlyGenerationPrefixError(message)

        output = SecondPassGenerationPrefixOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        self._validate_output(
            output,
            prompt_length=prompt_length,
            dtype=dtype,
            device=device,
        )
        return output

    def _validate_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if type(embeddings) is not torch.Tensor or embeddings.ndim != 3:
            message = "Prompt embeddings must have shape [1, prompt, hidden_size]"
            raise PromptOnlyGenerationPrefixError(message)
        batch_size, prompt_length, hidden_size = embeddings.shape
        if (
            batch_size != 1
            or prompt_length <= 0
            or prompt_length > self.max_prompt_tokens
            or hidden_size != self.hidden_size
        ):
            message = "Prompt embedding shape exceeds the prompt-only prefix contract"
            raise PromptOnlyGenerationPrefixError(message)
        if not torch.is_floating_point(embeddings):
            message = "Prompt embeddings must use a floating dtype"
            raise PromptOnlyGenerationPrefixError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = "Prompt embeddings must not track gradients during generation"
            raise PromptOnlyGenerationPrefixError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = f"Prompt embeddings cannot be inspected on {embeddings.device}"
            raise PromptOnlyGenerationPrefixError(message) from error
        if not finite:
            message = "Prompt embeddings contain NaN or Inf"
            raise PromptOnlyGenerationPrefixError(message)
        return prompt_length, embeddings.dtype, embeddings.device

    @staticmethod
    def _validate_attention_mask(
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
            raise PromptOnlyGenerationPrefixError(message)
        if mask.dtype not in integer_dtypes or mask.device != device:
            message = "Prompt attention mask has an invalid dtype or device"
            raise PromptOnlyGenerationPrefixError(message)
        if not bool(torch.logical_or(mask == 0, mask == 1).all()):
            message = "Prompt attention mask must contain only zeros and ones"
            raise PromptOnlyGenerationPrefixError(message)
        canonical = mask.detach().clone().to(dtype=torch.bool)
        if not bool(canonical.any()) or not bool(canonical[:, -1].all()):
            message = "Prompt-only generation requires an attended final prefix token"
            raise PromptOnlyGenerationPrefixError(message)
        transitions = canonical[:, 1:].to(dtype=torch.int8) - canonical[:, :-1].to(dtype=torch.int8)
        if not bool((transitions >= 0).all()):
            message = "Prompt-only attention must use contiguous left padding"
            raise PromptOnlyGenerationPrefixError(message)
        return canonical

    def _validate_output(
        self,
        output: SecondPassGenerationPrefixOutput,
        *,
        prompt_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            tuple(output.inputs_embeds.shape) != (1, prompt_length, self.hidden_size)
            or output.inputs_embeds.dtype != dtype
            or output.inputs_embeds.device != device
            or output.inputs_embeds.requires_grad
            or output.inputs_embeds.grad_fn is not None
            or not bool(torch.isfinite(output.inputs_embeds).all())
        ):
            message = "Prompt-only prefix embeddings are invalid"
            raise PromptOnlyGenerationPrefixError(message)
        expected_matrix = (1, prompt_length)
        if (
            tuple(output.attention_mask.shape) != expected_matrix
            or output.attention_mask.dtype != torch.bool
            or output.attention_mask.device != device
        ):
            message = "Prompt-only prefix attention mask is invalid"
            raise PromptOnlyGenerationPrefixError(message)
        if (
            tuple(output.position_ids.shape) != expected_matrix
            or output.position_ids.dtype != torch.int64
            or output.position_ids.device != device
        ):
            message = "Prompt-only prefix position IDs are invalid"
            raise PromptOnlyGenerationPrefixError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor.detach().clone()


__all__ = [
    "PromptOnlyGenerationPrefixAssembler",
    "PromptOnlyGenerationPrefixError",
]
