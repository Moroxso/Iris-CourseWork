"""Contract tests for the no-workspace arm-B generation prefix."""

from dataclasses import replace
from typing import Any

import pytest
import torch

import iris.generation as generation
from iris.cognitive import ModuleLatentBridgeOutput
from iris.generation import ModuleGenerationPrefixAssembler, ModuleGenerationPrefixError


def assembler() -> ModuleGenerationPrefixAssembler:
    return ModuleGenerationPrefixAssembler(
        hidden_size=4,
        num_modules=3,
        max_prompt_tokens=5,
        max_new_tokens=2,
        max_context_tokens=10,
    ).eval()


def module_output(*, prompt_length: int = 2) -> ModuleLatentBridgeOutput:
    return ModuleLatentBridgeOutput(
        module_embeddings=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        module_attention_mask=torch.ones((1, 3), dtype=torch.bool),
        module_position_ids=torch.arange(
            prompt_length,
            prompt_length + 3,
            dtype=torch.int64,
        ).unsqueeze(0),
    )


def test_module_prefix_preserves_prompt_then_all_module_tokens() -> None:
    prompt = torch.full((1, 2, 4), -1.0, dtype=torch.float32)
    modules = module_output()

    output = assembler()(
        prompt_embeddings=prompt,
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
        modules=modules,
    )

    assert output.inputs_embeds.shape == (1, 5, 4)
    assert torch.equal(output.inputs_embeds[:, :2], prompt)
    assert torch.equal(output.inputs_embeds[:, 2:], modules.module_embeddings)
    assert torch.equal(output.attention_mask, torch.ones((1, 5), dtype=torch.bool))
    assert torch.equal(output.position_ids, torch.arange(5, dtype=torch.int64).unsqueeze(0))


def test_module_prefix_positions_follow_left_padded_logical_prompt() -> None:
    modules = module_output(prompt_length=2)
    output = assembler()(
        prompt_embeddings=torch.zeros((1, 3, 4), dtype=torch.float32),
        prompt_attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.int64),
        modules=modules,
    )

    assert torch.equal(output.position_ids, torch.tensor([[0, 0, 1, 2, 3, 4]]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 0),
        ("num_modules", True),
        ("max_prompt_tokens", -1),
        ("max_new_tokens", 0),
        ("max_context_tokens", 0),
    ],
)
def test_module_prefix_rejects_invalid_constructor_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "hidden_size": 4,
        "num_modules": 3,
        "max_prompt_tokens": 5,
        "max_new_tokens": 2,
        "max_context_tokens": 10,
    }
    values[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        ModuleGenerationPrefixAssembler(**values)


def test_module_prefix_rejects_impossible_context() -> None:
    with pytest.raises(ValueError, match="exceed context"):
        ModuleGenerationPrefixAssembler(
            hidden_size=4,
            num_modules=3,
            max_prompt_tokens=5,
            max_new_tokens=3,
            max_context_tokens=10,
        )


@pytest.mark.parametrize(
    "prompt",
    [
        torch.zeros((2, 4), dtype=torch.float32),
        torch.zeros((1, 6, 4), dtype=torch.float32),
        torch.zeros((1, 2, 5), dtype=torch.float32),
        torch.zeros((1, 2, 4), dtype=torch.int64),
        torch.full((1, 2, 4), float("nan")),
    ],
)
def test_module_prefix_rejects_invalid_prompt_embeddings(prompt: torch.Tensor) -> None:
    with pytest.raises(ModuleGenerationPrefixError):
        assembler()(
            prompt_embeddings=prompt,
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
            modules=module_output(),
        )


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones((2,), dtype=torch.int64),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.zeros((1, 2), dtype=torch.int64),
        torch.tensor([[1, 0, 1]], dtype=torch.int64),
    ],
)
def test_module_prefix_rejects_invalid_prompt_mask(mask: torch.Tensor) -> None:
    with pytest.raises(ModuleGenerationPrefixError):
        assembler()(
            prompt_embeddings=torch.zeros((1, mask.shape[-1], 4), dtype=torch.float32),
            prompt_attention_mask=mask,
            modules=module_output(),
        )


@pytest.mark.parametrize(
    "modules",
    [
        replace(
            module_output(),
            module_embeddings=torch.zeros((1, 2, 4), dtype=torch.float32),
        ),
        replace(
            module_output(),
            module_attention_mask=torch.tensor([[True, False, True]]),
        ),
        replace(
            module_output(),
            module_position_ids=torch.tensor([[3, 4, 5]], dtype=torch.int64),
        ),
    ],
)
def test_module_prefix_rejects_invalid_module_metadata(
    modules: ModuleLatentBridgeOutput,
) -> None:
    with pytest.raises(ModuleGenerationPrefixError):
        assembler()(
            prompt_embeddings=torch.zeros((1, 2, 4), dtype=torch.float32),
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
            modules=modules,
        )


def test_module_prefix_rejects_grad_tracking_latents() -> None:
    modules = replace(
        module_output(),
        module_embeddings=torch.zeros((1, 3, 4), dtype=torch.float32, requires_grad=True),
    )
    with pytest.raises(ModuleGenerationPrefixError, match="track gradients"):
        assembler()(
            prompt_embeddings=torch.zeros((1, 2, 4), dtype=torch.float32),
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
            modules=modules,
        )


def test_module_prefix_api_is_public() -> None:
    assert generation.ModuleGenerationPrefixAssembler is ModuleGenerationPrefixAssembler
