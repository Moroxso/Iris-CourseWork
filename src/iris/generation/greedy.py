"""Deterministic cached greedy decoding for the Iris latent second pass."""

from collections.abc import Callable

import torch
from transformers.cache_utils import Cache

from iris.backbone.adapter import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation.outputs import (
    IrisGenerationOutput,
    SecondPassGenerationPrefixOutput,
)
from iris.workspace.state import WorkspaceState


class IrisGenerationError(RuntimeError):
    """Raised when a validated second-pass prefix cannot be decoded safely."""


class _CachedIrisDecoder:
    """Shared single-request loop with a private, linearly owned KV cache."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        *,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...],
    ) -> None:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            message = "max_new_tokens must be a positive integer"
            raise ValueError(message)
        if max_new_tokens <= 0:
            message = "max_new_tokens must be a positive integer"
            raise ValueError(message)
        if not isinstance(stop_token_ids, tuple):
            message = "stop_token_ids must be a tuple of unique token IDs"
            raise ValueError(message)
        try:
            vocabulary_size = backbone.vocabulary_size
        except Exception as error:
            message = "Failed to read the backbone vocabulary size"
            raise ValueError(message) from error
        if not isinstance(vocabulary_size, int) or isinstance(vocabulary_size, bool):
            message = "Backbone vocabulary size must be a positive integer"
            raise ValueError(message)
        if vocabulary_size <= 0:
            message = "Backbone vocabulary size must be a positive integer"
            raise ValueError(message)
        if any(type(token_id) is not int for token_id in stop_token_ids):
            message = "Stop token IDs must be integers"
            raise ValueError(message)
        if len(set(stop_token_ids)) != len(stop_token_ids):
            message = "Stop token IDs must be unique"
            raise ValueError(message)
        if any(token_id < 0 or token_id >= vocabulary_size for token_id in stop_token_ids):
            message = "Stop token ID is outside the backbone vocabulary"
            raise ValueError(message)

        self._backbone = backbone
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids
        self._stop_token_set = frozenset(stop_token_ids)
        self._vocabulary_size = vocabulary_size

    def _generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        workspace_trace: tuple[WorkspaceState, ...],
        select_next_token: Callable[[torch.Tensor], torch.Tensor],
    ) -> IrisGenerationOutput:
        """Decode through one cache using the caller's validated token selector."""
        prefix_length, hidden_size, dtype, device = self._validate_prefix(prefix)
        self._validate_workspace_trace(workspace_trace)
        try:
            max_position_embeddings = self._backbone.max_position_embeddings
        except Exception as error:
            message = "Failed to read the backbone context limit"
            raise IrisGenerationError(message) from error
        if (
            not isinstance(max_position_embeddings, int)
            or isinstance(max_position_embeddings, bool)
            or max_position_embeddings <= 0
        ):
            message = "Backbone context limit must be a positive integer"
            raise IrisGenerationError(message)
        if prefix_length + self.max_new_tokens > max_position_embeddings:
            message = "Generation prefix plus requested output exceeds the backbone context"
            raise IrisGenerationError(message)
        if not bool((prefix.position_ids + self.max_new_tokens <= max_position_embeddings).all()):
            message = "Requested generated positions exceed the backbone context"
            raise IrisGenerationError(message)

        working_embeddings = prefix.inputs_embeds.detach().clone()
        working_attention = prefix.attention_mask.detach().clone()
        working_positions = prefix.position_ids.detach().clone()
        causal_output = self._prefill(
            working_embeddings,
            working_attention,
            working_positions,
            expected_cache_length=prefix_length,
            dtype=dtype,
            device=device,
        )
        generated_tokens: list[torch.Tensor] = []
        last_prefix_position = working_positions[:, -1:]

        for generated_count in range(1, self.max_new_tokens + 1):
            next_token = select_next_token(causal_output.logits.detach().clone())
            if (
                not isinstance(next_token, torch.Tensor)
                or tuple(next_token.shape) != (1, 1)
                or next_token.dtype != torch.int64
                or next_token.device != device
                or next_token.requires_grad
                or next_token.grad_fn is not None
            ):
                message = "Generation token selection produced an invalid token tensor"
                raise IrisGenerationError(message)
            selected_token_id = int(next_token.item())
            if selected_token_id < 0 or selected_token_id >= self._vocabulary_size:
                message = "Generation token selection produced an out-of-vocabulary ID"
                raise IrisGenerationError(message)
            generated_tokens.append(next_token.detach().clone())

            if selected_token_id in self._stop_token_set:
                return self._build_output(
                    generated_tokens,
                    stop_reason="stop_token",
                    matched_stop_token_id=selected_token_id,
                    workspace_trace=workspace_trace,
                    device=device,
                )
            if generated_count == self.max_new_tokens:
                return self._build_output(
                    generated_tokens,
                    stop_reason="max_new_tokens",
                    matched_stop_token_id=None,
                    workspace_trace=workspace_trace,
                    device=device,
                )

            cache = causal_output.past_key_values
            if cache is None:
                message = "Cached generation pass did not return past_key_values"
                raise IrisGenerationError(message)
            token_embeddings = self._embed_generated_token(
                next_token,
                hidden_size=hidden_size,
                dtype=dtype,
                device=device,
            )
            working_attention = torch.cat(
                (
                    working_attention,
                    torch.ones((1, 1), dtype=torch.bool, device=device),
                ),
                dim=1,
            )
            current_position = last_prefix_position + generated_count
            causal_output = self._cached_step(
                token_embeddings,
                working_attention,
                current_position,
                cache=cache,
                expected_cache_length=prefix_length + generated_count,
                dtype=dtype,
                device=device,
            )

        message = "Cached generation loop terminated without a stop reason"
        raise IrisGenerationError(message)

    def _validate_prefix(
        self,
        prefix: SecondPassGenerationPrefixOutput,
    ) -> tuple[int, int, torch.dtype, torch.device]:
        if not isinstance(prefix, SecondPassGenerationPrefixOutput):
            message = "Cached decoder input must be SecondPassGenerationPrefixOutput"
            raise IrisGenerationError(message)
        embeddings = prefix.inputs_embeds
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            message = "Generation-prefix embeddings must have shape [1, sequence, hidden]"
            raise IrisGenerationError(message)
        batch_size, prefix_length, hidden_size = embeddings.shape
        try:
            expected_hidden_size = self._backbone.hidden_size
        except Exception as error:
            message = "Failed to read the backbone hidden size"
            raise IrisGenerationError(message) from error
        if batch_size != 1 or prefix_length <= 0 or hidden_size != expected_hidden_size:
            message = "Generation-prefix embedding shape does not match the backbone"
            raise IrisGenerationError(message)
        if not torch.is_floating_point(embeddings):
            message = "Generation-prefix embeddings must use a floating dtype"
            raise IrisGenerationError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = "Greedy generation requires detached inference embeddings"
            raise IrisGenerationError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = "Generation-prefix embeddings cannot be inspected"
            raise IrisGenerationError(message) from error
        if not finite:
            message = "Generation-prefix embeddings contain NaN or Inf"
            raise IrisGenerationError(message)

        expected_matrix = (1, prefix_length)
        attention_mask = prefix.attention_mask
        position_ids = prefix.position_ids
        if (
            not isinstance(attention_mask, torch.Tensor)
            or tuple(attention_mask.shape) != expected_matrix
            or attention_mask.dtype != torch.bool
            or attention_mask.device != embeddings.device
            or not bool(attention_mask.any())
            or not bool(attention_mask[:, -1].all())
        ):
            message = "Generation-prefix attention mask is invalid"
            raise IrisGenerationError(message)
        if (
            not isinstance(position_ids, torch.Tensor)
            or tuple(position_ids.shape) != expected_matrix
            or position_ids.dtype != torch.int64
            or position_ids.device != embeddings.device
        ):
            message = "Generation-prefix position IDs are invalid"
            raise IrisGenerationError(message)
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        expected_positions = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        if not torch.equal(position_ids, expected_positions):
            message = "Generation-prefix position IDs do not match its attention mask"
            raise IrisGenerationError(message)
        return prefix_length, hidden_size, embeddings.dtype, embeddings.device

    @staticmethod
    def _validate_workspace_trace(workspace_trace: tuple[WorkspaceState, ...]) -> None:
        if not isinstance(workspace_trace, tuple) or not all(
            isinstance(state, WorkspaceState) for state in workspace_trace
        ):
            message = "workspace_trace must be a tuple of WorkspaceState snapshots"
            raise IrisGenerationError(message)

    def _prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        expected_cache_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> BackboneCausalOutput:
        try:
            output = self._backbone.forward_embeddings(
                inputs_embeds.detach().clone(),
                attention_mask.detach().clone(),
                position_ids.detach().clone(),
                use_cache=True,
                logits_to_keep=1,
            )
        except Exception as error:
            message = "Backbone generation prefill failed"
            raise IrisGenerationError(message) from error
        self._validate_causal_output(
            output,
            expected_cache_length=expected_cache_length,
            dtype=dtype,
            device=device,
        )
        return output

    def _embed_generated_token(
        self,
        token_ids: torch.Tensor,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            embeddings = self._backbone.embed_tokens(token_ids.detach().clone())
        except Exception as error:
            message = "Failed to embed a generated token"
            raise IrisGenerationError(message) from error
        expected_shape = (1, 1, hidden_size)
        if (
            not isinstance(embeddings, torch.Tensor)
            or tuple(embeddings.shape) != expected_shape
            or embeddings.dtype != dtype
            or embeddings.device != device
            or embeddings.requires_grad
            or embeddings.grad_fn is not None
        ):
            message = "Generated-token embeddings violate the frozen backbone contract"
            raise IrisGenerationError(message)
        if not bool(torch.isfinite(embeddings).all()):
            message = "Generated-token embeddings contain NaN or Inf"
            raise IrisGenerationError(message)
        return embeddings

    def _cached_step(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        cache: Cache,
        expected_cache_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> BackboneCausalOutput:
        try:
            output = self._backbone.forward_embeddings(
                inputs_embeds.detach().clone(),
                attention_mask.detach().clone(),
                position_ids.detach().clone(),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        except Exception as error:
            message = "Backbone cached generation step failed"
            raise IrisGenerationError(message) from error
        self._validate_causal_output(
            output,
            expected_cache_length=expected_cache_length,
            dtype=dtype,
            device=device,
        )
        return output

    def _validate_causal_output(
        self,
        output: BackboneCausalOutput,
        *,
        expected_cache_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(output, BackboneCausalOutput):
            message = "Backbone returned an invalid causal generation output"
            raise IrisGenerationError(message)
        logits = output.logits
        expected_shape = (1, 1, self._vocabulary_size)
        if (
            not isinstance(logits, torch.Tensor)
            or tuple(logits.shape) != expected_shape
            or not torch.is_floating_point(logits)
            or logits.dtype != dtype
            or logits.device != device
            or logits.requires_grad
            or logits.grad_fn is not None
        ):
            message = "Backbone generation logits violate the decoder contract"
            raise IrisGenerationError(message)
        if not bool(torch.isfinite(logits).all()):
            message = "Backbone generation logits contain NaN or Inf"
            raise IrisGenerationError(message)
        if output.loss is not None:
            message = "Cached generation must not return a teacher-forcing loss"
            raise IrisGenerationError(message)
        cache = output.past_key_values
        if not isinstance(cache, Cache):
            message = "Cached generation output must contain a Transformers Cache"
            raise IrisGenerationError(message)
        try:
            cache_length = cache.get_seq_length()
        except Exception as error:
            message = "Failed to read the generated cache length"
            raise IrisGenerationError(message) from error
        if (
            not isinstance(cache_length, int)
            or isinstance(cache_length, bool)
            or cache_length != expected_cache_length
        ):
            message = "Generated cache has an invalid sequence length"
            raise IrisGenerationError(message)

    @staticmethod
    def _build_output(
        generated_tokens: list[torch.Tensor],
        *,
        stop_reason: str,
        matched_stop_token_id: int | None,
        workspace_trace: tuple[WorkspaceState, ...],
        device: torch.device,
    ) -> IrisGenerationOutput:
        token_ids = torch.cat(generated_tokens, dim=1)
        if (
            token_ids.ndim != 2
            or token_ids.shape[0] != 1
            or token_ids.shape[1] <= 0
            or token_ids.dtype != torch.int64
            or token_ids.device != device
            or token_ids.requires_grad
        ):
            message = "Generated token output is invalid"
            raise IrisGenerationError(message)
        if stop_reason == "stop_token":
            return IrisGenerationOutput(
                token_ids=token_ids,
                stop_reason="stop_token",
                matched_stop_token_id=matched_stop_token_id,
                workspace_trace=workspace_trace,
            )
        if stop_reason == "max_new_tokens" and matched_stop_token_id is None:
            return IrisGenerationOutput(
                token_ids=token_ids,
                stop_reason="max_new_tokens",
                matched_stop_token_id=None,
                workspace_trace=workspace_trace,
            )
        message = "Generated output has inconsistent termination metadata"
        raise IrisGenerationError(message)


class GreedyIrisDecoder(_CachedIrisDecoder):
    """Generate deterministically by selecting the maximum-logit token."""

    @torch.inference_mode()
    def generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        workspace_trace: tuple[WorkspaceState, ...],
    ) -> IrisGenerationOutput:
        """Decode greedily; stop tokens are included and caches never escape."""
        return self._generate(
            prefix,
            workspace_trace=workspace_trace,
            select_next_token=self._select_next_token,
        )

    @staticmethod
    def _select_next_token(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)
