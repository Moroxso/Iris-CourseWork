"""Contract tests for the frozen prompt-only arm-A request graph."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import iris.application as application
from iris.application import (
    BaseLlmRequestOrchestrator,
    IrisBaseLlmRequestError,
    IrisCompositionError,
    build_base_llm_request_orchestrator,
)
from iris.backbone import BackboneAdapter
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    PromptOnlyGenerationPrefixAssembler,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.workspace import WorkspaceState

CPU = torch.device("cpu")
HIDDEN_SIZE = 4
VOCABULARY_SIZE = 17
MAX_PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 2
MAX_CONTEXT_TOKENS = 7


class RecordingBackbone(BackboneAdapter):
    """Already-loaded frozen backbone double with prompt-only public operations."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.last_tokenized: dict[str, torch.Tensor] | None = None
        self.mutate_embedded_ids = False
        self.trainable_count = 0
        self.gradient_flag = False
        self.reported_context = MAX_CONTEXT_TOKENS

    @property
    def hidden_size(self) -> int:
        return HIDDEN_SIZE

    @property
    def vocabulary_size(self) -> int:
        return VOCABULARY_SIZE

    @property
    def max_position_embeddings(self) -> int:
        return self.reported_context

    @property
    def max_prompt_tokens(self) -> int:
        return MAX_PROMPT_TOKENS

    @property
    def max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS

    @property
    def device(self) -> torch.device:
        return CPU

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def trainable_parameter_count(self) -> int:
        return self.trainable_count

    @property
    def has_parameter_gradients(self) -> bool:
        return self.gradient_flag

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        self.events.append(f"tokenize:{text}")
        output = {
            "input_ids": torch.tensor([[3, 5, 7]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        }
        self.last_tokenized = output
        return output

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.events.append("embed")
        if self.mutate_embedded_ids:
            input_ids.add_(1)
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, HIDDEN_SIZE)

    def forward_prompt(self, text: str) -> Any:
        raise AssertionError(f"arm A must not run an unused first pass for {text!r}")


class RecordingGenerator(IrisGenerator):
    """Exact generator subtype exposing the facade contract without an LM callback."""

    def __init__(self, backbone: BackboneAdapter) -> None:
        self._backbone = backbone
        self.max_new_tokens = MAX_NEW_TOKENS
        self.stop_token_ids = (2,)
        self.sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
        self.calls: list[
            tuple[
                SecondPassGenerationPrefixOutput,
                DecodingStrategy,
                tuple[WorkspaceState, ...],
                torch.Generator | None,
            ]
        ] = []

    @property
    def backbone(self) -> BackboneAdapter:
        return self._backbone

    def generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        strategy: DecodingStrategy,
        workspace_trace: tuple[WorkspaceState, ...],
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        self.calls.append((prefix, strategy, workspace_trace, generator))
        return IrisGeneratorOutput(
            text="baseline",
            token_ids=torch.tensor([[11, 13]], dtype=torch.int64),
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=workspace_trace,
        )


def prefix_assembler() -> PromptOnlyGenerationPrefixAssembler:
    return PromptOnlyGenerationPrefixAssembler(
        hidden_size=HIDDEN_SIZE,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    ).eval()


def orchestrator(
    backbone: RecordingBackbone | None = None,
) -> tuple[BaseLlmRequestOrchestrator, RecordingBackbone, RecordingGenerator]:
    owned_backbone = backbone if backbone is not None else RecordingBackbone()
    generator = RecordingGenerator(owned_backbone)
    return (
        BaseLlmRequestOrchestrator(
            backbone=owned_backbone,
            prefix_assembler=prefix_assembler(),
            iris_generator=generator,
        ),
        owned_backbone,
        generator,
    )


def test_arm_a_executes_one_prompt_only_decode_with_empty_trace() -> None:
    graph, backbone, generator = orchestrator()

    output = graph.execute("hello", strategy="greedy")

    assert graph.arm_id == "A"
    assert graph.arm_kind == "base_llm"
    assert graph.backbone is backbone
    assert backbone.events == ["tokenize:hello", "embed"]
    assert output.generation.text == "baseline"
    assert output.generation.workspace_trace == ()
    assert output.receipt is not None
    assert output.receipt.strategy == "greedy"
    assert output.receipt.stop_token_ids == (2,)
    assert output.receipt.sampling == generator.sampling
    assert output.receipt.sampling_seed is None
    assert len(generator.calls) == 1
    prefix, strategy, trace, random_generator = generator.calls[0]
    assert strategy == "greedy"
    assert trace == ()
    assert random_generator is None
    assert tuple(prefix.inputs_embeds.shape) == (1, 3, HIDDEN_SIZE)
    assert torch.equal(prefix.attention_mask, torch.ones((1, 3), dtype=torch.bool))
    assert torch.equal(prefix.position_ids, torch.tensor([[0, 1, 2]], dtype=torch.int64))


def test_arm_a_returns_owned_exact_prompt_provenance() -> None:
    graph, backbone, _ = orchestrator()

    output = graph.execute("hello", strategy="greedy")

    assert backbone.last_tokenized is not None
    source_ids = backbone.last_tokenized["input_ids"]
    source_mask = backbone.last_tokenized["attention_mask"]
    assert torch.equal(output.prompt_input_ids, source_ids)
    assert torch.equal(output.prompt_attention_mask, source_mask)
    assert output.prompt_input_ids.untyped_storage().data_ptr() != (
        source_ids.untyped_storage().data_ptr()
    )
    assert output.prompt_attention_mask.untyped_storage().data_ptr() != (
        source_mask.untyped_storage().data_ptr()
    )
    assert not output.prompt_input_ids.is_inference()
    assert not output.prompt_attention_mask.is_inference()


def test_arm_a_sampling_preserves_explicit_generator_identity() -> None:
    graph, _, facade = orchestrator()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)

    output = graph.generate("sample", strategy="sampling", generator=generator)

    assert output.text == "baseline"
    assert facade.calls[0][3] is generator


def test_arm_a_sampling_receipt_captures_the_actual_generator_seed() -> None:
    graph, _, _ = orchestrator()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7001)

    execution = graph.execute("sample", strategy="sampling", generator=generator)

    assert execution.receipt is not None
    assert execution.receipt.strategy == "sampling"
    assert execution.receipt.sampling_seed == 7001


@pytest.mark.parametrize("prompt", ["", "  ", cast(str, 1)])
def test_arm_a_rejects_invalid_prompt_before_tokenization(prompt: str) -> None:
    graph, backbone, _ = orchestrator()

    with pytest.raises(IrisBaseLlmRequestError, match="prompt"):
        graph.execute(prompt, strategy="greedy")

    assert backbone.events == []


def test_arm_a_rejects_invalid_generation_ownership() -> None:
    graph, _, _ = orchestrator()

    with pytest.raises(IrisBaseLlmRequestError, match="does not accept"):
        graph.execute(
            "hello",
            strategy="greedy",
            generator=torch.Generator(device="cpu"),
        )
    with pytest.raises(IrisBaseLlmRequestError, match="requires an explicit"):
        graph.execute("hello", strategy="sampling")


def test_arm_a_detects_prompt_mutation_by_embedding_boundary() -> None:
    backbone = RecordingBackbone()
    backbone.mutate_embedded_ids = True
    graph, _, generator = orchestrator(backbone)

    with pytest.raises(IrisBaseLlmRequestError, match="mutated exact prompt"):
        graph.execute("hello", strategy="greedy")

    assert generator.calls == []


def test_arm_a_rejects_noncanonical_tokenizer_attention_dtype() -> None:
    graph, backbone, _ = orchestrator()

    def bad_tokenize(text: str) -> dict[str, torch.Tensor]:
        del text
        return {
            "input_ids": torch.tensor([[3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[True]], dtype=torch.bool),
        }

    backbone.tokenize = bad_tokenize  # type: ignore[method-assign]
    with pytest.raises(IrisBaseLlmRequestError, match="tokenization violates"):
        graph.execute("hello", strategy="greedy")


def test_arm_a_detects_runtime_context_metadata_change() -> None:
    graph, backbone, _ = orchestrator()
    backbone.reported_context += 1

    with pytest.raises(IrisBaseLlmRequestError, match="runtime boundary"):
        graph.execute("hello", strategy="greedy")


def test_arm_a_builder_constructs_only_prompt_prefix_and_shared_generator() -> None:
    backbone = RecordingBackbone()

    graph = build_base_llm_request_orchestrator(
        backbone=backbone,
        stop_token_ids=(2,),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )

    assert type(graph) is BaseLlmRequestOrchestrator
    assert graph.backbone is backbone
    assert not cast(Any, graph)._prefix_assembler.training


@pytest.mark.parametrize(
    "mutation",
    ["trainable", "gradient", "context"],
)
def test_arm_a_builder_rejects_nonbaseline_backbone(mutation: str) -> None:
    backbone = RecordingBackbone()
    if mutation == "trainable":
        backbone.trainable_count = 1
    elif mutation == "gradient":
        backbone.gradient_flag = True
    else:
        backbone.reported_context = MAX_CONTEXT_TOKENS - 1

    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_base_llm_request_orchestrator(
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def test_arm_a_builder_rejects_invalid_stop_ids() -> None:
    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_base_llm_request_orchestrator(
            backbone=RecordingBackbone(),
            stop_token_ids=(VOCABULARY_SIZE,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def test_arm_a_api_is_public() -> None:
    assert application.BaseLlmRequestOrchestrator is BaseLlmRequestOrchestrator
    assert application.build_base_llm_request_orchestrator is build_base_llm_request_orchestrator
