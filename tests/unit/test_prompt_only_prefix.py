"""Contract tests for the arm-A prompt-only generation prefix."""

from typing import Any

import pytest
import torch

from iris.generation import (
    PromptOnlyGenerationPrefixAssembler,
    PromptOnlyGenerationPrefixError,
)


def assembler() -> PromptOnlyGenerationPrefixAssembler:
    return PromptOnlyGenerationPrefixAssembler(
        hidden_size=4,
        max_prompt_tokens=5,
        max_new_tokens=2,
        max_context_tokens=7,
    ).eval()


def test_prompt_only_prefix_preserves_left_padding_and_owns_embeddings() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    mask = torch.tensor([[0, 1, 1]], dtype=torch.int64)

    output = assembler()(prompt_embeddings=source, prompt_attention_mask=mask)

    assert torch.equal(output.inputs_embeds, source)
    assert output.inputs_embeds.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert output.attention_mask.dtype == torch.bool
    assert torch.equal(output.attention_mask, torch.tensor([[False, True, True]]))
    assert torch.equal(output.position_ids, torch.tensor([[0, 0, 1]], dtype=torch.int64))
    assert not output.inputs_embeds.requires_grad


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 0),
        ("max_prompt_tokens", True),
        ("max_new_tokens", -1),
        ("max_context_tokens", 0),
    ],
)
def test_prompt_only_prefix_rejects_invalid_constructor_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "hidden_size": 4,
        "max_prompt_tokens": 5,
        "max_new_tokens": 2,
        "max_context_tokens": 7,
    }
    values[field] = value

    with pytest.raises(ValueError, match="positive integer"):
        PromptOnlyGenerationPrefixAssembler(**values)


def test_prompt_only_prefix_rejects_impossible_configured_context() -> None:
    with pytest.raises(ValueError, match="exceed context"):
        PromptOnlyGenerationPrefixAssembler(
            hidden_size=4,
            max_prompt_tokens=5,
            max_new_tokens=3,
            max_context_tokens=7,
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.zeros((3, 4), dtype=torch.float32),
        torch.zeros((2, 3, 4), dtype=torch.float32),
        torch.zeros((1, 6, 4), dtype=torch.float32),
        torch.zeros((1, 3, 5), dtype=torch.float32),
        torch.zeros((1, 3, 4), dtype=torch.int64),
        torch.full((1, 3, 4), float("nan")),
    ],
)
def test_prompt_only_prefix_rejects_invalid_embeddings(embeddings: torch.Tensor) -> None:
    with pytest.raises(PromptOnlyGenerationPrefixError):
        assembler()(
            prompt_embeddings=embeddings,
            prompt_attention_mask=torch.ones((1, 3), dtype=torch.int64),
        )


def test_prompt_only_prefix_rejects_grad_tracking_embeddings() -> None:
    embeddings = torch.zeros((1, 3, 4), dtype=torch.float32, requires_grad=True)
    with pytest.raises(PromptOnlyGenerationPrefixError, match="track gradients"):
        assembler()(
            prompt_embeddings=embeddings,
            prompt_attention_mask=torch.ones((1, 3), dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones((3,), dtype=torch.int64),
        torch.ones((1, 3), dtype=torch.float32),
        torch.tensor([[0, 2, 1]], dtype=torch.int64),
        torch.zeros((1, 3), dtype=torch.int64),
        torch.tensor([[1, 1, 0]], dtype=torch.int64),
        torch.tensor([[0, 1, 0]], dtype=torch.int64),
    ],
)
def test_prompt_only_prefix_rejects_decoder_incompatible_masks(mask: torch.Tensor) -> None:
    with pytest.raises(PromptOnlyGenerationPrefixError):
        assembler()(
            prompt_embeddings=torch.zeros((1, 3, 4), dtype=torch.float32),
            prompt_attention_mask=mask,
        )


def test_prompt_only_prefix_materializes_inference_tensor_for_normal_owner() -> None:
    with torch.inference_mode():
        inference_embeddings = torch.ones((1, 2, 4), dtype=torch.float32)
    assert inference_embeddings.is_inference()

    output = assembler()(
        prompt_embeddings=inference_embeddings,
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
    )

    assert not output.inputs_embeds.is_inference()
    assert torch.equal(output.inputs_embeds, inference_embeddings)
