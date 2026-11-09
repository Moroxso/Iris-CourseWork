"""Contract tests for teacher-forced second-pass sequence assembly."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from iris.backbone import BackboneAdapter
from iris.generation import (
    SecondPassSequenceAssembler,
    SecondPassSequenceAssemblyError,
    SecondPassSequenceOutput,
)
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo
from iris.workspace import LatentBridge, LatentBridgeOutput, WorkspaceState

CPU = torch.device("cpu")
HIDDEN_SIZE = 3
NUM_SLOTS = 2
VOCABULARY_SIZE = 16
PROMPT_LENGTH = 4
TARGET_LENGTH = 3
TOTAL_LENGTH = PROMPT_LENGTH + NUM_SLOTS + TARGET_LENGTH


@dataclass(frozen=True, slots=True)
class AssemblyInputs:
    """One complete, independently replaceable assembler input bundle."""

    prompt_embeddings: torch.Tensor
    prompt_attention_mask: torch.Tensor
    workspace: LatentBridgeOutput
    target_input_ids: torch.Tensor
    target_embeddings: torch.Tensor
    target_attention_mask: torch.Tensor


class FakeTokenizer:
    """The adapter integration only needs mutable padding attributes."""

    padding_side = "right"
    truncation_side = "right"


def make_assembler(
    *,
    hidden_size: int = HIDDEN_SIZE,
    num_slots: int = NUM_SLOTS,
    vocabulary_size: int = VOCABULARY_SIZE,
    max_prompt_tokens: int = PROMPT_LENGTH,
    max_target_tokens: int = TARGET_LENGTH,
    max_context_tokens: int = TOTAL_LENGTH,
) -> SecondPassSequenceAssembler:
    return SecondPassSequenceAssembler(
        hidden_size=hidden_size,
        num_slots=num_slots,
        vocabulary_size=vocabulary_size,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        max_context_tokens=max_context_tokens,
    )


def make_inputs(*, workspace_requires_grad: bool = False) -> AssemblyInputs:
    prompt_embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
    prompt_attention_mask = torch.tensor(
        [[False, False, True, True], [True, True, True, False]],
        dtype=torch.bool,
    )
    workspace_embeddings = 10 + torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 10
    workspace_embeddings.requires_grad_(workspace_requires_grad)
    workspace = LatentBridgeOutput(
        workspace_embeddings=workspace_embeddings,
        workspace_attention_mask=torch.ones(2, 2, dtype=torch.bool),
        workspace_position_ids=torch.tensor([[2, 3], [3, 4]], dtype=torch.int64),
    )
    target_input_ids = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int64)
    target_embeddings = 20 + torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10
    target_attention_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    return AssemblyInputs(
        prompt_embeddings=prompt_embeddings,
        prompt_attention_mask=prompt_attention_mask,
        workspace=workspace,
        target_input_ids=target_input_ids,
        target_embeddings=target_embeddings,
        target_attention_mask=target_attention_mask,
    )


def assemble(
    inputs: AssemblyInputs,
    assembler: SecondPassSequenceAssembler | None = None,
) -> SecondPassSequenceOutput:
    selected = assembler or make_assembler()
    return cast(
        SecondPassSequenceOutput,
        selected(
            prompt_embeddings=inputs.prompt_embeddings,
            prompt_attention_mask=inputs.prompt_attention_mask,
            workspace=inputs.workspace,
            target_input_ids=inputs.target_input_ids,
            target_embeddings=inputs.target_embeddings,
            target_attention_mask=inputs.target_attention_mask,
        ),
    )


def test_assembles_exact_segments_masks_positions_and_unshifted_labels() -> None:
    inputs = make_inputs()

    output = assemble(inputs)

    assert isinstance(output, SecondPassSequenceOutput)
    torch.testing.assert_close(
        output.inputs_embeds,
        torch.cat(
            (
                inputs.prompt_embeddings,
                inputs.workspace.workspace_embeddings,
                inputs.target_embeddings,
            ),
            dim=1,
        ),
    )
    assert output.attention_mask.tolist() == [
        [False, False, True, True, True, True, True, True, False],
        [True, True, True, False, True, True, True, False, False],
    ]
    assert output.position_ids.tolist() == [
        [0, 0, 0, 1, 2, 3, 4, 5, 0],
        [0, 1, 2, 0, 3, 4, 5, 0, 0],
    ]
    assert output.labels.tolist() == [
        [-100, -100, -100, -100, -100, -100, 5, 6, -100],
        [-100, -100, -100, -100, -100, -100, 8, -100, -100],
    ]
    assert output.inputs_embeds.shape == (2, TOTAL_LENGTH, HIDDEN_SIZE)
    assert output.inputs_embeds.dtype == torch.float32
    assert output.attention_mask.dtype == torch.bool
    assert output.position_ids.dtype == torch.int64
    assert output.labels.dtype == torch.int64
    assert all(
        tensor.device == CPU
        for tensor in (
            output.inputs_embeds,
            output.attention_mask,
            output.position_ids,
            output.labels,
        )
    )


def test_labels_follow_hugging_face_unshifted_causal_convention() -> None:
    inputs = make_inputs()
    output = assemble(inputs)
    target_start = PROMPT_LENGTH + NUM_SLOTS

    assert bool((output.labels[:, :target_start] == -100).all())
    torch.testing.assert_close(
        output.labels[:, target_start:],
        torch.where(
            inputs.target_attention_mask,
            inputs.target_input_ids,
            torch.full_like(inputs.target_input_ids, -100),
        ),
    )
    # Hugging Face shifts labels inside the causal LM. Thus the first target
    # label supervises the logit emitted at the final workspace position.
    shifted_labels = output.labels[:, 1:]
    torch.testing.assert_close(
        shifted_labels[:, target_start - 1],
        inputs.target_input_ids[:, 0],
    )
    assert bool((output.labels[:, target_start - 1] == -100).all())


def test_equal_logical_prompt_lengths_make_workspace_and_target_positions_padding_invariant() -> (
    None
):
    inputs = make_inputs()
    masks = torch.tensor(
        [[False, False, True, True], [True, True, False, False]],
        dtype=torch.bool,
    )
    workspace = replace(
        inputs.workspace,
        workspace_position_ids=torch.tensor([[2, 3], [2, 3]], dtype=torch.int64),
    )
    equal_lengths = replace(inputs, prompt_attention_mask=masks, workspace=workspace)

    output = assemble(equal_lengths)

    assert output.position_ids[:, PROMPT_LENGTH:].tolist() == [
        [2, 3, 4, 5, 0],
        [2, 3, 4, 0, 0],
    ]


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_accepts_every_binary_integer_mask_dtype_and_canonicalizes_to_bool(
    dtype: torch.dtype,
) -> None:
    inputs = make_inputs()
    converted = replace(
        inputs,
        prompt_attention_mask=inputs.prompt_attention_mask.to(dtype=dtype),
        target_attention_mask=inputs.target_attention_mask.to(dtype=dtype),
    )

    output = assemble(converted)

    assert output.attention_mask.dtype == torch.bool
    assert torch.equal(output.attention_mask, assemble(inputs).attention_mask)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("hidden_size", 0),
        ("hidden_size", -1),
        ("hidden_size", True),
        ("hidden_size", 3.0),
        ("num_slots", 0),
        ("num_slots", False),
        ("num_slots", 2.5),
        ("vocabulary_size", 0),
        ("vocabulary_size", True),
        ("max_prompt_tokens", 0),
        ("max_prompt_tokens", False),
        ("max_target_tokens", -1),
        ("max_target_tokens", True),
        ("max_context_tokens", 0),
        ("max_context_tokens", 9.5),
    ],
)
def test_rejects_non_positive_or_non_integer_constructor_values(
    name: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "hidden_size": HIDDEN_SIZE,
        "num_slots": NUM_SLOTS,
        "vocabulary_size": VOCABULARY_SIZE,
        "max_prompt_tokens": PROMPT_LENGTH,
        "max_target_tokens": TARGET_LENGTH,
        "max_context_tokens": TOTAL_LENGTH,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        SecondPassSequenceAssembler(**cast(Any, arguments))


def test_constructor_rejects_maxima_that_exceed_context_and_accepts_exact_boundary() -> None:
    with pytest.raises(ValueError, match="maxima exceed the context window"):
        make_assembler(max_context_tokens=TOTAL_LENGTH - 1)

    assembler = make_assembler(max_context_tokens=TOTAL_LENGTH)
    output = assemble(make_inputs(), assembler)

    assert output.inputs_embeds.shape[1] == TOTAL_LENGTH
    assert assembler.max_context_tokens == TOTAL_LENGTH


def test_shorter_actual_sequences_remain_valid_under_larger_configured_maxima() -> None:
    assembler = make_assembler(
        max_prompt_tokens=6,
        max_target_tokens=4,
        max_context_tokens=12,
    )

    output = assemble(make_inputs(), assembler)

    assert output.inputs_embeds.shape == (2, TOTAL_LENGTH, HIDDEN_SIZE)


InputMutator = Callable[[AssemblyInputs], AssemblyInputs]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, prompt_embeddings=cast(Any, "not a tensor")),
        lambda value: replace(value, prompt_embeddings=value.prompt_embeddings[0]),
        lambda value: replace(value, prompt_embeddings=torch.empty(0, 4, 3)),
        lambda value: replace(value, prompt_embeddings=torch.empty(2, 0, 3)),
        lambda value: replace(value, prompt_embeddings=torch.empty(2, 5, 3)),
        lambda value: replace(value, prompt_embeddings=torch.empty(2, 4, 2)),
        lambda value: replace(value, prompt_embeddings=value.prompt_embeddings.to(torch.int64)),
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
            prompt_embeddings=torch.ones_like(
                value.prompt_embeddings,
                requires_grad=True,
            )
            * 2,
        ),
    ],
)
def test_rejects_malformed_non_frozen_prompt_embeddings(mutator: InputMutator) -> None:
    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(mutator(make_inputs()))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, target_embeddings=cast(Any, "not a tensor")),
        lambda value: replace(value, target_embeddings=value.target_embeddings[0]),
        lambda value: replace(value, target_embeddings=torch.empty(0, 3, 3)),
        lambda value: replace(value, target_embeddings=torch.empty(2, 0, 3)),
        lambda value: replace(value, target_embeddings=torch.empty(2, 4, 3)),
        lambda value: replace(value, target_embeddings=torch.empty(2, 3, 2)),
        lambda value: replace(value, target_embeddings=value.target_embeddings.to(torch.int64)),
        lambda value: replace(value, target_embeddings=value.target_embeddings.to(torch.float64)),
        lambda value: replace(
            value,
            target_embeddings=torch.full_like(value.target_embeddings, float("nan")),
        ),
        lambda value: replace(
            value,
            target_embeddings=torch.full_like(value.target_embeddings, float("-inf")),
        ),
        lambda value: replace(
            value,
            target_embeddings=value.target_embeddings.detach().requires_grad_(),
        ),
        lambda value: replace(
            value,
            target_embeddings=torch.ones_like(
                value.target_embeddings,
                requires_grad=True,
            )
            * 2,
        ),
        lambda value: replace(
            value,
            target_embeddings=torch.ones(2, 3, 3, dtype=torch.float32, device="meta"),
        ),
    ],
)
def test_rejects_malformed_non_frozen_target_embeddings(mutator: InputMutator) -> None:
    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(mutator(make_inputs()))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, prompt_attention_mask=cast(Any, "not a tensor")),
        lambda value: replace(value, prompt_attention_mask=torch.ones(2, 4, 1, dtype=torch.bool)),
        lambda value: replace(value, prompt_attention_mask=torch.ones(1, 4, dtype=torch.bool)),
        lambda value: replace(value, prompt_attention_mask=torch.ones(2, 3, dtype=torch.bool)),
        lambda value: replace(value, prompt_attention_mask=torch.ones(2, 4)),
        lambda value: replace(
            value,
            prompt_attention_mask=torch.tensor([[1, 2, 1, 1], [1, 1, 1, 0]]),
        ),
        lambda value: replace(value, prompt_attention_mask=torch.zeros(2, 4, dtype=torch.bool)),
        lambda value: replace(
            value,
            prompt_attention_mask=torch.tensor(
                [[True, False, True, False], [True, True, True, False]],
            ),
        ),
        lambda value: replace(
            value,
            prompt_attention_mask=torch.ones(2, 4, dtype=torch.bool, device="meta"),
        ),
    ],
)
def test_rejects_malformed_prompt_attention_masks(mutator: InputMutator) -> None:
    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(mutator(make_inputs()))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, target_attention_mask=cast(Any, "not a tensor")),
        lambda value: replace(value, target_attention_mask=torch.ones(2, 3, 1, dtype=torch.bool)),
        lambda value: replace(value, target_attention_mask=torch.ones(1, 3, dtype=torch.bool)),
        lambda value: replace(value, target_attention_mask=torch.ones(2, 2, dtype=torch.bool)),
        lambda value: replace(value, target_attention_mask=torch.ones(2, 3)),
        lambda value: replace(
            value,
            target_attention_mask=torch.tensor([[1, 2, 0], [1, 0, 0]]),
        ),
        lambda value: replace(value, target_attention_mask=torch.zeros(2, 3, dtype=torch.bool)),
        lambda value: replace(
            value,
            target_attention_mask=torch.tensor(
                [[False, True, True], [True, False, False]],
            ),
        ),
        lambda value: replace(
            value,
            target_attention_mask=torch.tensor(
                [[True, False, True], [True, False, False]],
            ),
        ),
        lambda value: replace(
            value,
            target_attention_mask=torch.ones(2, 3, dtype=torch.bool, device="meta"),
        ),
    ],
)
def test_rejects_malformed_or_non_right_padded_target_masks(mutator: InputMutator) -> None:
    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(mutator(make_inputs()))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: replace(value, target_input_ids=cast(Any, "not a tensor")),
        lambda value: replace(value, target_input_ids=value.target_input_ids[0]),
        lambda value: replace(value, target_input_ids=torch.ones(1, 3, dtype=torch.int64)),
        lambda value: replace(value, target_input_ids=torch.ones(2, 2, dtype=torch.int64)),
        lambda value: replace(value, target_input_ids=value.target_input_ids.to(torch.int32)),
        lambda value: replace(
            value,
            target_input_ids=torch.tensor([[-1, 6, 7], [8, 9, 10]], dtype=torch.int64),
        ),
        lambda value: replace(
            value,
            target_input_ids=torch.tensor([[5, 6, 16], [8, 9, 10]], dtype=torch.int64),
        ),
        # Padding IDs remain physical embedding indices and must also be in vocabulary.
        lambda value: replace(
            value,
            target_input_ids=torch.tensor([[5, 6, 99], [8, 9, 10]], dtype=torch.int64),
        ),
        lambda value: replace(
            value,
            target_input_ids=torch.ones(2, 3, dtype=torch.int64, device="meta"),
        ),
    ],
)
def test_rejects_malformed_or_out_of_vocabulary_target_ids(mutator: InputMutator) -> None:
    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(mutator(make_inputs()))


WorkspaceMutator = Callable[[LatentBridgeOutput], object]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda workspace: cast(Any, "not a LatentBridgeOutput"),
        lambda workspace: replace(
            workspace, workspace_embeddings=workspace.workspace_embeddings[0]
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.empty(1, 2, 3),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.empty(2, 1, 3),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.empty(2, 2, 2),
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
            workspace_embeddings=torch.full_like(
                workspace.workspace_embeddings,
                float("inf"),
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_embeddings=torch.ones(
                2,
                2,
                3,
                dtype=torch.float32,
                device="meta",
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.ones(2, 2, dtype=torch.int64),
        ),
        lambda workspace: replace(
            workspace,
            workspace_attention_mask=torch.tensor(
                [[True, False], [True, True]],
                dtype=torch.bool,
            ),
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=torch.ones(2, 1, dtype=torch.int64),
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
            workspace_attention_mask=torch.ones(2, 2, dtype=torch.bool, device="meta"),
        ),
        lambda workspace: replace(
            workspace,
            workspace_position_ids=torch.ones(2, 2, dtype=torch.int64, device="meta"),
        ),
    ],
)
def test_rejects_noncanonical_or_malformed_workspace_dto(mutator: WorkspaceMutator) -> None:
    inputs = make_inputs()
    malformed = mutator(inputs.workspace)

    with pytest.raises(SecondPassSequenceAssemblyError):
        assemble(replace(inputs, workspace=cast(Any, malformed)))


def test_rejects_workspace_positions_from_a_different_prompt_padding_contract() -> None:
    inputs = make_inputs()
    changed_prompt = torch.tensor(
        [[False, True, True, True], [True, True, True, False]],
        dtype=torch.bool,
    )

    with pytest.raises(SecondPassSequenceAssemblyError, match="follow the logical prompt"):
        assemble(replace(inputs, prompt_attention_mask=changed_prompt))


def test_workspace_embeddings_preserve_the_only_trainable_input_graph() -> None:
    leaf = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).requires_grad_()
    transform = torch.nn.Parameter(torch.tensor(1.5))
    derived = leaf * transform
    inputs = make_inputs()
    graph_workspace = replace(inputs.workspace, workspace_embeddings=derived)

    output = assemble(replace(inputs, workspace=graph_workspace))
    loss = output.inputs_embeds[:, PROMPT_LENGTH : PROMPT_LENGTH + NUM_SLOTS].square().sum()
    torch.autograd.backward(loss)

    assert leaf.grad is not None
    assert bool((leaf.grad != 0).any())
    assert transform.grad is not None
    assert bool(transform.grad != 0)
    assert output.inputs_embeds.requires_grad
    assert output.inputs_embeds.grad_fn is not None


def test_forward_has_no_parameters_buffers_state_rng_or_mode_dependent_behavior() -> None:
    assembler = make_assembler()
    inputs = make_inputs()
    assembler.train()
    cpu_rng_before = torch.random.get_rng_state().clone()

    training_output = assemble(inputs, assembler)
    cpu_rng_after = torch.random.get_rng_state()
    assembler.eval()
    evaluation_output = assemble(inputs, assembler)

    assert dict(assembler.named_parameters()) == {}
    assert dict(assembler.named_buffers()) == {}
    assert assembler.state_dict() == {}
    assert torch.equal(cpu_rng_before, cpu_rng_after)
    for training_tensor, evaluation_tensor in zip(
        (
            training_output.inputs_embeds,
            training_output.attention_mask,
            training_output.position_ids,
            training_output.labels,
        ),
        (
            evaluation_output.inputs_embeds,
            evaluation_output.attention_mask,
            evaluation_output.position_ids,
            evaluation_output.labels,
        ),
        strict=True,
    ):
        torch.testing.assert_close(training_tensor, evaluation_tensor)


def test_forward_does_not_mutate_inputs_and_outputs_do_not_alias_input_storage() -> None:
    inputs = make_inputs()
    prompt_before = inputs.prompt_embeddings.clone()
    prompt_mask_before = inputs.prompt_attention_mask.clone()
    workspace_embeddings_before = inputs.workspace.workspace_embeddings.clone()
    workspace_mask_before = inputs.workspace.workspace_attention_mask.clone()
    workspace_positions_before = inputs.workspace.workspace_position_ids.clone()
    target_ids_before = inputs.target_input_ids.clone()
    target_embeddings_before = inputs.target_embeddings.clone()
    target_mask_before = inputs.target_attention_mask.clone()

    output = assemble(inputs)

    torch.testing.assert_close(inputs.prompt_embeddings, prompt_before)
    torch.testing.assert_close(inputs.prompt_attention_mask, prompt_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_embeddings, workspace_embeddings_before)
    torch.testing.assert_close(inputs.workspace.workspace_attention_mask, workspace_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_position_ids, workspace_positions_before)
    torch.testing.assert_close(inputs.target_input_ids, target_ids_before)
    torch.testing.assert_close(inputs.target_embeddings, target_embeddings_before)
    torch.testing.assert_close(inputs.target_attention_mask, target_mask_before)
    input_storages = {
        inputs.prompt_embeddings.untyped_storage().data_ptr(),
        inputs.workspace.workspace_embeddings.untyped_storage().data_ptr(),
        inputs.target_embeddings.untyped_storage().data_ptr(),
    }
    assert output.inputs_embeds.untyped_storage().data_ptr() not in input_storages
    assert (
        output.attention_mask.untyped_storage().data_ptr()
        != inputs.prompt_attention_mask.untyped_storage().data_ptr()
    )
    assert (
        output.position_ids.untyped_storage().data_ptr()
        != inputs.workspace.workspace_position_ids.untyped_storage().data_ptr()
    )
    assert (
        output.labels.untyped_storage().data_ptr()
        != inputs.target_input_ids.untyped_storage().data_ptr()
    )

    output.inputs_embeds.zero_()
    output.attention_mask.zero_()
    output.position_ids.zero_()
    output.labels.zero_()
    torch.testing.assert_close(inputs.prompt_embeddings, prompt_before)
    torch.testing.assert_close(inputs.workspace.workspace_embeddings, workspace_embeddings_before)
    torch.testing.assert_close(inputs.target_embeddings, target_embeddings_before)
    torch.testing.assert_close(inputs.prompt_attention_mask, prompt_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_attention_mask, workspace_mask_before)
    torch.testing.assert_close(inputs.workspace.workspace_position_ids, workspace_positions_before)
    torch.testing.assert_close(inputs.target_input_ids, target_ids_before)
    torch.testing.assert_close(inputs.target_attention_mask, target_mask_before)


def test_inference_created_inputs_are_materialized_when_consumed_outside_inference_mode() -> None:
    with torch.inference_mode():
        inputs = make_inputs()
    assert inputs.prompt_embeddings.is_inference()
    assert inputs.workspace.workspace_embeddings.is_inference()
    assert inputs.target_embeddings.is_inference()

    output = assemble(inputs)

    assert not output.inputs_embeds.is_inference()
    assert not output.attention_mask.is_inference()
    assert not output.position_ids.is_inference()
    assert not output.labels.is_inference()


def test_forward_is_valid_inside_global_inference_mode() -> None:
    inputs = make_inputs()

    with torch.inference_mode():
        output = assemble(inputs)

    assert output.inputs_embeds.is_inference()
    assert output.attention_mask.is_inference()
    assert output.position_ids.is_inference()
    assert output.labels.is_inference()
    assert bool(torch.isfinite(output.inputs_embeds).all())


def tiny_backbone_config() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "qwen3_0_6b",
            "repository": "Qwen/Qwen3-0.6B",
            "revision": "test-model-revision",
            "tokenizer_revision": "test-tokenizer-revision",
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": "fp32",
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {"max_prompt_tokens": 3, "max_new_tokens": 2},
        }
    )


def cpu_info() -> DeviceInfo:
    return DeviceInfo(
        device=CPU,
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )


def tiny_qwen_adapter() -> tuple[BackboneAdapter, Qwen3ForCausalLM]:
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
        )
    )
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        adapter = BackboneAdapter(tiny_backbone_config(), cpu_info())
    return adapter, model


def occupied_workspace(slots: torch.Tensor) -> WorkspaceState:
    batch_size, num_slots, _ = slots.shape
    matrix_shape = (batch_size, num_slots)
    return WorkspaceState(
        slots=slots,
        age=torch.ones(matrix_shape, dtype=torch.int64),
        occupancy=torch.ones(matrix_shape, dtype=torch.bool),
        source_module=torch.arange(num_slots, dtype=torch.int64).expand(batch_size, -1),
        confidence=torch.full(matrix_shape, 0.75, dtype=slots.dtype),
        ttl=torch.full(matrix_shape, 2.0, dtype=slots.dtype),
        write_count=torch.ones(matrix_shape, dtype=torch.int64),
    )


def test_tiny_qwen_loss_reaches_last_workspace_and_bridge_but_not_frozen_backbone() -> None:
    torch.manual_seed(3251)
    adapter, model = tiny_qwen_adapter()
    bridge = LatentBridge(
        state_dim=4,
        backbone_hidden_size=adapter.hidden_size,
        num_slots=2,
    )
    workspace_slots = torch.randn(1, 2, 4, requires_grad=True)
    prompt_ids = torch.tensor([[0, 2, 3]], dtype=torch.int64)
    prompt_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    target_ids = torch.tensor([[5, 6]], dtype=torch.int64)
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    prompt_embeddings = adapter.embed_tokens(prompt_ids)
    target_embeddings = adapter.embed_tokens(target_ids)
    latent = bridge(occupied_workspace(workspace_slots), prompt_mask)
    latent.workspace_embeddings.retain_grad()
    assembler = SecondPassSequenceAssembler(
        hidden_size=adapter.hidden_size,
        num_slots=2,
        vocabulary_size=adapter.vocabulary_size,
        max_prompt_tokens=3,
        max_target_tokens=2,
        max_context_tokens=adapter.max_position_embeddings,
    )

    sequence = assembler(
        prompt_embeddings=prompt_embeddings,
        prompt_attention_mask=prompt_mask,
        workspace=latent,
        target_input_ids=target_ids,
        target_embeddings=target_embeddings,
        target_attention_mask=target_mask,
    )
    causal_output = adapter.forward_embeddings(
        sequence.inputs_embeds,
        sequence.attention_mask,
        sequence.position_ids,
        labels=sequence.labels,
    )

    assert sequence.labels.tolist() == [[-100, -100, -100, -100, -100, 5, 6]]
    assert sequence.position_ids.tolist() == [[0, 0, 1, 2, 3, 4, 5]]
    assert causal_output.loss is not None
    assert causal_output.loss.requires_grad
    torch.autograd.backward(causal_output.loss)

    assert latent.workspace_embeddings.grad is not None
    assert bool((latent.workspace_embeddings.grad[:, -1] != 0).any())
    assert workspace_slots.grad is not None
    assert bool((workspace_slots.grad[:, -1] != 0).any())
    assert bridge.workspace_type_embedding.grad is not None
    assert bool((bridge.workspace_type_embedding.grad != 0).any())
    assert bridge.bridge_projection.weight.grad is not None
    assert bool((bridge.bridge_projection.weight.grad != 0).any())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert adapter.has_parameter_gradients is False
