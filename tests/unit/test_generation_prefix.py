"""Contract tests for the inference-only second-pass generation prefix."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, cast

import pytest
import torch

from iris.generation import (
    SecondPassGenerationPrefixAssembler,
    SecondPassGenerationPrefixError,
    SecondPassGenerationPrefixOutput,
)
from iris.workspace import LatentBridgeOutput

HIDDEN_SIZE = 3
NUM_SLOTS = 2
PROMPT_LENGTH = 4
MAX_NEW_TOKENS = 3
MAX_CONTEXT_TOKENS = 9


@dataclass(frozen=True, slots=True)
class PrefixInputs:
    prompt_embeddings: torch.Tensor
    prompt_attention_mask: torch.Tensor
    workspace: LatentBridgeOutput


def make_assembler(
    *,
    hidden_size: int = HIDDEN_SIZE,
    num_slots: int = NUM_SLOTS,
    max_prompt_tokens: int = PROMPT_LENGTH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> SecondPassGenerationPrefixAssembler:
    return SecondPassGenerationPrefixAssembler(
        hidden_size=hidden_size,
        num_slots=num_slots,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
    )


def make_inputs(
    *,
    prompt_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> PrefixInputs:
    if prompt_mask is None:
        prompt_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
    prompt_logical_length = int(prompt_mask.to(dtype=torch.int64).sum())
    prompt = torch.arange(
        PROMPT_LENGTH * HIDDEN_SIZE,
        dtype=dtype,
    ).reshape(1, PROMPT_LENGTH, HIDDEN_SIZE)
    workspace_values = torch.arange(
        100,
        100 + NUM_SLOTS * HIDDEN_SIZE,
        dtype=dtype,
    ).reshape(1, NUM_SLOTS, HIDDEN_SIZE)
    workspace = LatentBridgeOutput(
        workspace_embeddings=workspace_values,
        workspace_attention_mask=torch.ones(1, NUM_SLOTS, dtype=torch.bool),
        workspace_position_ids=(
            torch.arange(NUM_SLOTS, dtype=torch.int64) + prompt_logical_length
        ).unsqueeze(0),
    )
    return PrefixInputs(
        prompt_embeddings=prompt,
        prompt_attention_mask=prompt_mask,
        workspace=workspace,
    )


def assemble(
    inputs: PrefixInputs,
    assembler: SecondPassGenerationPrefixAssembler | None = None,
) -> SecondPassGenerationPrefixOutput:
    if assembler is None:
        assembler = make_assembler()
    return cast(
        SecondPassGenerationPrefixOutput,
        assembler(
            prompt_embeddings=inputs.prompt_embeddings,
            prompt_attention_mask=inputs.prompt_attention_mask,
            workspace=inputs.workspace,
        ),
    )


@pytest.mark.parametrize(
    ("prompt_mask", "expected_positions"),
    [
        (
            torch.tensor([[False, False, True, True]], dtype=torch.bool),
            [[0, 0, 0, 1, 2, 3]],
        ),
        (
            torch.tensor([[True, True, False, False]], dtype=torch.bool),
            [[0, 1, 0, 0, 2, 3]],
        ),
        (
            torch.ones(1, PROMPT_LENGTH, dtype=torch.bool),
            [[0, 1, 2, 3, 4, 5]],
        ),
    ],
)
def test_exact_prefix_for_left_right_and_unpadded_prompts(
    prompt_mask: torch.Tensor,
    expected_positions: list[list[int]],
) -> None:
    inputs = make_inputs(prompt_mask=prompt_mask)

    output = assemble(inputs)

    expected_embeddings = torch.cat(
        (inputs.prompt_embeddings, inputs.workspace.workspace_embeddings),
        dim=1,
    )
    expected_mask = torch.cat(
        (prompt_mask, torch.ones(1, NUM_SLOTS, dtype=torch.bool)),
        dim=1,
    )
    torch.testing.assert_close(output.inputs_embeds, expected_embeddings)
    assert torch.equal(output.attention_mask, expected_mask)
    assert output.position_ids.tolist() == expected_positions
    assert output.inputs_embeds.shape == (1, PROMPT_LENGTH + NUM_SLOTS, HIDDEN_SIZE)
    assert output.inputs_embeds.dtype == inputs.prompt_embeddings.dtype
    assert output.attention_mask.dtype == torch.bool
    assert output.position_ids.dtype == torch.int64
    assert output.inputs_embeds.requires_grad is False
    assert output.inputs_embeds.grad_fn is None


def test_prefix_output_is_frozen_slotted_data() -> None:
    output = assemble(make_inputs())

    assert not hasattr(output, "__dict__")
    with pytest.raises(FrozenInstanceError):
        output.position_ids = torch.zeros_like(output.position_ids)  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    [
        "hidden_size",
        "num_slots",
        "max_prompt_tokens",
        "max_new_tokens",
        "max_context_tokens",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None])
def test_constructor_requires_exact_positive_integers(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "hidden_size": HIDDEN_SIZE,
        "num_slots": NUM_SLOTS,
        "max_prompt_tokens": PROMPT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        SecondPassGenerationPrefixAssembler(**cast(Any, arguments))


def test_constructor_accepts_exact_context_boundary() -> None:
    assembler = make_assembler(max_context_tokens=MAX_CONTEXT_TOKENS)

    output = assemble(make_inputs(), assembler)

    assert output.inputs_embeds.shape[1] + MAX_NEW_TOKENS == MAX_CONTEXT_TOKENS


def test_constructor_rejects_configured_context_overflow() -> None:
    with pytest.raises(ValueError, match="exceed context"):
        make_assembler(max_context_tokens=MAX_CONTEXT_TOKENS - 1)


PromptMutator = Callable[[PrefixInputs], PrefixInputs]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, prompt_embeddings=cast(Any, "not a tensor")),
        lambda value: replace(value, prompt_embeddings=value.prompt_embeddings[0]),
        lambda value: replace(
            value,
            prompt_embeddings=torch.ones(2, PROMPT_LENGTH, HIDDEN_SIZE),
        ),
        lambda value: replace(value, prompt_embeddings=torch.ones(1, 0, HIDDEN_SIZE)),
        lambda value: replace(
            value,
            prompt_embeddings=torch.ones(1, PROMPT_LENGTH + 1, HIDDEN_SIZE),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=torch.ones(1, PROMPT_LENGTH, HIDDEN_SIZE + 1),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=value.prompt_embeddings.to(dtype=torch.int64),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=torch.full_like(value.prompt_embeddings, float("nan")),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=torch.full_like(value.prompt_embeddings, float("inf")),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=value.prompt_embeddings.detach().requires_grad_(),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=(torch.ones_like(value.prompt_embeddings, requires_grad=True) * 2),
        ),
        lambda value: replace(
            value,
            prompt_embeddings=torch.ones(
                1,
                PROMPT_LENGTH,
                HIDDEN_SIZE,
                device="meta",
            ),
        ),
    ],
)
def test_rejects_malformed_or_gradient_tracking_prompt_embeddings(
    mutator: PromptMutator,
) -> None:
    with pytest.raises(SecondPassGenerationPrefixError):
        assemble(mutator(make_inputs()))


@pytest.mark.parametrize(
    "mask",
    [
        cast(Any, "not a tensor"),
        torch.ones(PROMPT_LENGTH, dtype=torch.bool),
        torch.ones(1, PROMPT_LENGTH, 1, dtype=torch.bool),
        torch.ones(2, PROMPT_LENGTH, dtype=torch.bool),
        torch.ones(1, PROMPT_LENGTH - 1, dtype=torch.bool),
        torch.ones(1, PROMPT_LENGTH),
        torch.tensor([[1, 2, 1, 1]], dtype=torch.int64),
        torch.zeros(1, PROMPT_LENGTH, dtype=torch.bool),
        torch.tensor([[True, False, True, False]], dtype=torch.bool),
        torch.ones(1, PROMPT_LENGTH, dtype=torch.bool, device="meta"),
    ],
)
def test_rejects_malformed_nonbinary_empty_or_noncontiguous_prompt_masks(
    mask: object,
) -> None:
    inputs = replace(make_inputs(), prompt_attention_mask=cast(Any, mask))

    with pytest.raises(SecondPassGenerationPrefixError):
        assemble(inputs)


WorkspaceMutator = Callable[[LatentBridgeOutput], object]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda workspace: cast(Any, "not a LatentBridgeOutput"),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=workspace.workspace_embeddings[0],
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.ones(2, NUM_SLOTS, HIDDEN_SIZE),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.ones(1, NUM_SLOTS + 1, HIDDEN_SIZE),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.ones(1, NUM_SLOTS, HIDDEN_SIZE + 1),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=workspace.workspace_embeddings.to(torch.int64),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=workspace.workspace_embeddings.to(torch.float64),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.full_like(
                workspace.workspace_embeddings,
                float("nan"),
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=workspace.workspace_embeddings.detach().requires_grad_(),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=(
                torch.ones_like(workspace.workspace_embeddings, requires_grad=True) * 2
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.ones(
                1,
                NUM_SLOTS,
                HIDDEN_SIZE,
                device="meta",
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.ones(1, NUM_SLOTS - 1, dtype=torch.bool),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.ones(1, NUM_SLOTS, dtype=torch.int64),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.tensor([[True, False]], dtype=torch.bool),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.ones(
                1,
                NUM_SLOTS,
                dtype=torch.bool,
                device="meta",
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=torch.ones(1, NUM_SLOTS - 1, dtype=torch.int64),
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=workspace.workspace_position_ids.to(torch.int32),
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=workspace.workspace_position_ids + 1,
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=torch.ones(
                1,
                NUM_SLOTS,
                dtype=torch.int64,
                device="meta",
            ),
        ),
    ],
)
def test_rejects_malformed_or_noncanonical_workspace(mutator: WorkspaceMutator) -> None:
    inputs = make_inputs()

    with pytest.raises(SecondPassGenerationPrefixError):
        assemble(replace(inputs, workspace=cast(Any, mutator(inputs.workspace))))


def test_workspace_positions_are_bound_to_logical_not_physical_prompt_length() -> None:
    inputs = make_inputs()
    physical_positions = torch.arange(
        PROMPT_LENGTH,
        PROMPT_LENGTH + NUM_SLOTS,
        dtype=torch.int64,
    ).unsqueeze(0)
    malformed = replace(inputs.workspace, workspace_position_ids=physical_positions)

    with pytest.raises(SecondPassGenerationPrefixError, match="logical prompt"):
        assemble(replace(inputs, workspace=malformed))


def test_forward_has_no_parameters_buffers_state_rng_or_mode_dependence() -> None:
    assembler = make_assembler()
    inputs = make_inputs()
    assembler.train()
    rng_before = torch.random.get_rng_state().clone()

    training = assemble(inputs, assembler)
    rng_after = torch.random.get_rng_state()
    assembler.eval()
    evaluation = assemble(inputs, assembler)

    assert dict(assembler.named_parameters()) == {}
    assert dict(assembler.named_buffers()) == {}
    assert assembler.state_dict() == {}
    assert torch.equal(rng_before, rng_after)
    for train_value, eval_value in zip(
        (training.inputs_embeds, training.attention_mask, training.position_ids),
        (evaluation.inputs_embeds, evaluation.attention_mask, evaluation.position_ids),
        strict=True,
    ):
        torch.testing.assert_close(train_value, eval_value)


def test_forward_does_not_mutate_or_alias_any_input_storage() -> None:
    inputs = make_inputs()
    prompt_before = inputs.prompt_embeddings.clone()
    prompt_mask_before = inputs.prompt_attention_mask.clone()
    workspace_before = inputs.workspace.workspace_embeddings.clone()
    workspace_mask_before = inputs.workspace.workspace_attention_mask.clone()
    workspace_positions_before = inputs.workspace.workspace_position_ids.clone()

    output = assemble(inputs)

    torch.testing.assert_close(inputs.prompt_embeddings, prompt_before)
    torch.testing.assert_close(inputs.prompt_attention_mask, prompt_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_embeddings, workspace_before)
    torch.testing.assert_close(inputs.workspace.workspace_attention_mask, workspace_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_position_ids, workspace_positions_before)
    input_storage_pointers = {
        inputs.prompt_embeddings.untyped_storage().data_ptr(),
        inputs.prompt_attention_mask.untyped_storage().data_ptr(),
        inputs.workspace.workspace_embeddings.untyped_storage().data_ptr(),
        inputs.workspace.workspace_attention_mask.untyped_storage().data_ptr(),
        inputs.workspace.workspace_position_ids.untyped_storage().data_ptr(),
    }
    for tensor in (output.inputs_embeds, output.attention_mask, output.position_ids):
        assert tensor.untyped_storage().data_ptr() not in input_storage_pointers

    output.inputs_embeds.zero_()
    output.attention_mask.zero_()
    output.position_ids.zero_()
    torch.testing.assert_close(inputs.prompt_embeddings, prompt_before)
    torch.testing.assert_close(inputs.prompt_attention_mask, prompt_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_embeddings, workspace_before)
    torch.testing.assert_close(inputs.workspace.workspace_attention_mask, workspace_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_position_ids, workspace_positions_before)


def test_inference_tensors_are_materialized_when_consumed_outside_inference_mode() -> None:
    with torch.inference_mode():
        inputs = make_inputs()
    assert inputs.prompt_embeddings.is_inference()
    assert inputs.workspace.workspace_embeddings.is_inference()

    output = assemble(inputs)

    assert not output.inputs_embeds.is_inference()
    assert not output.attention_mask.is_inference()
    assert not output.position_ids.is_inference()


def test_forward_is_inference_safe_and_never_builds_a_graph() -> None:
    inputs = make_inputs(dtype=torch.float64)

    with torch.inference_mode():
        output = assemble(inputs)

    assert output.inputs_embeds.is_inference()
    assert output.attention_mask.is_inference()
    assert output.position_ids.is_inference()
    assert output.inputs_embeds.dtype == torch.float64
    assert output.inputs_embeds.requires_grad is False
    assert output.inputs_embeds.grad_fn is None
