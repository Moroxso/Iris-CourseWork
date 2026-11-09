from typing import Any, cast

import pytest
import torch

import iris.generation as generation
from iris.cognitive import ModuleLatentBridgeOutput
from iris.generation import (
    ModuleSecondPassSequenceAssembler,
    ModuleSecondPassSequenceAssemblyError,
    SecondPassSequenceOutput,
)


def _assembler() -> ModuleSecondPassSequenceAssembler:
    return ModuleSecondPassSequenceAssembler(
        hidden_size=5,
        num_modules=3,
        vocabulary_size=17,
        max_prompt_tokens=4,
        max_target_tokens=3,
        max_context_tokens=10,
    )


def _modules(
    *,
    batch_size: int = 1,
    prompt_lengths: tuple[int, ...] = (2,),
    requires_grad: bool = True,
) -> ModuleLatentBridgeOutput:
    embeddings = torch.randn(batch_size, 3, 5, requires_grad=requires_grad)
    positions = torch.tensor(prompt_lengths, dtype=torch.int64).unsqueeze(1) + torch.arange(3)
    return ModuleLatentBridgeOutput(
        module_embeddings=embeddings,
        module_attention_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        module_position_ids=positions,
    )


def _execute(
    *,
    assembler: ModuleSecondPassSequenceAssembler | None = None,
    prompt_embeddings: torch.Tensor | None = None,
    prompt_attention_mask: torch.Tensor | None = None,
    modules: ModuleLatentBridgeOutput | None = None,
    target_input_ids: torch.Tensor | None = None,
    target_embeddings: torch.Tensor | None = None,
    target_attention_mask: torch.Tensor | None = None,
) -> SecondPassSequenceOutput:
    return cast(
        SecondPassSequenceOutput,
        (assembler or _assembler())(
            prompt_embeddings=(
                torch.randn(1, 2, 5) if prompt_embeddings is None else prompt_embeddings
            ),
            prompt_attention_mask=(
                torch.ones(1, 2, dtype=torch.int64)
                if prompt_attention_mask is None
                else prompt_attention_mask
            ),
            modules=_modules() if modules is None else modules,
            target_input_ids=(
                torch.tensor([[4, 6]], dtype=torch.int64)
                if target_input_ids is None
                else target_input_ids
            ),
            target_embeddings=(
                torch.randn(1, 2, 5) if target_embeddings is None else target_embeddings
            ),
            target_attention_mask=(
                torch.ones(1, 2, dtype=torch.int64)
                if target_attention_mask is None
                else target_attention_mask
            ),
        ),
    )


def test_public_exports_are_stable() -> None:
    assert generation.ModuleSecondPassSequenceAssembler is ModuleSecondPassSequenceAssembler
    assert generation.ModuleSecondPassSequenceAssemblyError is ModuleSecondPassSequenceAssemblyError


def test_assembles_prompt_modules_target_and_labels_only_target() -> None:
    prompt = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    modules = _modules()
    target = torch.arange(10, 20, dtype=torch.float32).reshape(1, 2, 5)
    output = _execute(
        prompt_embeddings=prompt,
        modules=modules,
        target_embeddings=target,
    )
    assert type(output) is SecondPassSequenceOutput
    assert tuple(output.inputs_embeds.shape) == (1, 7, 5)
    assert torch.equal(output.inputs_embeds[:, :2], prompt)
    assert torch.equal(output.inputs_embeds[:, 2:5], modules.module_embeddings)
    assert torch.equal(output.inputs_embeds[:, 5:], target)
    assert output.attention_mask.tolist() == [[True] * 7]
    assert output.position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert output.labels.tolist() == [[-100, -100, -100, -100, -100, 4, 6]]


def test_preserves_module_autograd_and_frozen_token_boundaries() -> None:
    modules = _modules()
    output = _execute(modules=modules)
    torch.autograd.backward((output.inputs_embeds.sum(),))
    assert modules.module_embeddings.grad is not None
    assert torch.equal(
        modules.module_embeddings.grad,
        torch.ones_like(modules.module_embeddings),
    )


def test_supports_left_padded_prompt_and_right_padded_target() -> None:
    prompt = torch.randn(2, 4, 5)
    prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    target_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    target_embeddings = torch.randn(2, 3, 5)
    target_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    output = _execute(
        prompt_embeddings=prompt,
        prompt_attention_mask=prompt_mask,
        modules=_modules(batch_size=2, prompt_lengths=(2, 3)),
        target_input_ids=target_ids,
        target_embeddings=target_embeddings,
        target_attention_mask=target_mask,
    )
    assert output.position_ids.tolist() == [
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 0],
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    ]
    assert output.labels.tolist() == [
        [-100] * 7 + [1, 2, -100],
        [-100] * 7 + [3, 4, 5],
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        {"hidden_size": True},
        {"num_modules": 0},
        {"vocabulary_size": -1},
        {"max_prompt_tokens": 0},
        {"max_target_tokens": 0},
        {"max_context_tokens": 0},
    ],
)
def test_rejects_invalid_constructor_dimensions(arguments: dict[str, Any]) -> None:
    values: dict[str, object] = {
        "hidden_size": 5,
        "num_modules": 3,
        "vocabulary_size": 17,
        "max_prompt_tokens": 4,
        "max_target_tokens": 3,
        "max_context_tokens": 10,
    }
    values.update(arguments)
    with pytest.raises(ValueError):
        ModuleSecondPassSequenceAssembler(**cast(Any, values))


def test_rejects_configured_context_overflow() -> None:
    with pytest.raises(ValueError, match="exceed context"):
        ModuleSecondPassSequenceAssembler(
            hidden_size=5,
            num_modules=3,
            vocabulary_size=17,
            max_prompt_tokens=4,
            max_target_tokens=3,
            max_context_tokens=9,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_embeddings", torch.randn(1, 2, 4)),
        ("prompt_embeddings", torch.full((1, 2, 5), float("nan"))),
        ("prompt_attention_mask", torch.tensor([[1, 0, 1]])),
        ("target_input_ids", torch.tensor([[4, 17]], dtype=torch.int64)),
        ("target_embeddings", torch.randn(1, 2, 4)),
        ("target_attention_mask", torch.tensor([[0, 1]], dtype=torch.int64)),
    ],
)
def test_rejects_invalid_prompt_or_target_tensor(field: str, value: torch.Tensor) -> None:
    with pytest.raises(ModuleSecondPassSequenceAssemblyError):
        _execute(**cast(Any, {field: value}))


def test_rejects_frozen_token_embedding_with_grad_graph() -> None:
    with pytest.raises(ModuleSecondPassSequenceAssemblyError, match="frozen backbone"):
        _execute(prompt_embeddings=torch.randn(1, 2, 5, requires_grad=True))


@pytest.mark.parametrize("mutation", ["type", "shape", "mask", "position", "finite"])
def test_rejects_invalid_module_prefix(mutation: str) -> None:
    if mutation == "type":
        modules = cast(ModuleLatentBridgeOutput, object())
    else:
        nominal = _modules()
        modules = ModuleLatentBridgeOutput(
            module_embeddings=(
                torch.randn(1, 2, 5)
                if mutation == "shape"
                else torch.full((1, 3, 5), float("inf"))
                if mutation == "finite"
                else nominal.module_embeddings
            ),
            module_attention_mask=(
                torch.tensor([[1, 0, 1]], dtype=torch.bool)
                if mutation == "mask"
                else nominal.module_attention_mask
            ),
            module_position_ids=(
                torch.tensor([[3, 4, 5]], dtype=torch.int64)
                if mutation == "position"
                else nominal.module_position_ids
            ),
        )
    with pytest.raises(ModuleSecondPassSequenceAssemblyError):
        _execute(modules=modules)
