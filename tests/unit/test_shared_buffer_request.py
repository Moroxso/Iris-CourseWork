"""Contract tests for the append-only shared-buffer arm-C request graph."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from iris.application.errors import IrisCompositionError, IrisSharedBufferRequestError
from iris.application.shared_buffer_request import (
    CognitiveSharedBufferRequestOrchestrator,
    build_shared_buffer_request_orchestrator,
)
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
    CognitiveModuleOutput,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    SecondPassGenerationPrefixAssembler,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig
from iris.shared_buffer import (
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferState,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.workspace import WorkspaceReaderOutput, WorkspaceState

CPU = torch.device("cpu")
HIDDEN_SIZE = 6
STATE_DIM = 4
NUM_MODULES = 3
NUM_HIDDEN_LAYERS = 4
VOCABULARY_SIZE = 17
PROMPT_LENGTH = 3
MAX_PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 2
MAX_CONTEXT_TOKENS = 10


def cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "arm-c-test-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": NUM_MODULES,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


class RecordingBackbone(BackboneAdapter):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.last_first_pass: BackboneFirstPassOutput | None = None
        self.reported_context = MAX_CONTEXT_TOKENS
        self.trainable_count = 0
        self.gradient_flag = False

    @property
    def hidden_size(self) -> int:
        return HIDDEN_SIZE

    @property
    def num_hidden_layers(self) -> int:
        return NUM_HIDDEN_LAYERS

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

    def forward_prompt(self, text: str) -> BackboneFirstPassOutput:
        self.events.append(f"forward:{text}")
        input_ids = torch.tensor([[3, 5, 7]], dtype=torch.int64)
        attention_mask = torch.ones((1, PROMPT_LENGTH), dtype=torch.int64)
        hidden_states = tuple(
            torch.full(
                (1, PROMPT_LENGTH, HIDDEN_SIZE),
                float(index),
                dtype=torch.float32,
            )
            for index in range(NUM_HIDDEN_LAYERS + 1)
        )
        output = BackboneFirstPassOutput(
            logits=torch.zeros((1, PROMPT_LENGTH, VOCABULARY_SIZE)),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        self.last_first_pass = output
        return output

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.events.append("embed")
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, HIDDEN_SIZE)


class RecordingCognitiveModule(CognitiveModule):
    def __init__(self) -> None:
        super().__init__(state_dim=STATE_DIM, num_modules=NUM_MODULES)
        self.observed_reads: list[torch.Tensor] = []

    def forward(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        self.observed_reads.append(workspace_reads.detach().clone())
        return super().forward(
            module_inputs,
            workspace_reads,
            goal_state,
            previous_states=previous_states,
        )


class RecordingSharedBufferReader(SharedBufferReader):
    def __init__(self) -> None:
        super().__init__(state_dim=STATE_DIM, num_modules=NUM_MODULES, cycles=3)
        self.entry_counts: list[int] = []

    def forward(self, state: SharedBufferState) -> WorkspaceReaderOutput:
        self.entry_counts.append(state.entry_count)
        return super().forward(state)


class RecordingGenerator(IrisGenerator):
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
            text="shared-buffer",
            token_ids=torch.tensor([[11, 13]], dtype=torch.int64),
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=workspace_trace,
        )


def build_graph() -> tuple[
    CognitiveSharedBufferRequestOrchestrator,
    RecordingBackbone,
    RecordingCognitiveModule,
    RecordingSharedBufferReader,
    RecordingGenerator,
]:
    backbone = RecordingBackbone()
    config = cognitive_config()
    modules = RecordingCognitiveModule()
    reader = RecordingSharedBufferReader()
    generator = RecordingGenerator(backbone)
    components = (
        LayerSampler(
            config.layer_sampler,
            backbone_hidden_size=HIDDEN_SIZE,
            state_dim=STATE_DIM,
        ),
        GoalEncoder(state_dim=STATE_DIM),
        ModuleInputRouter(state_dim=STATE_DIM, num_modules=NUM_MODULES),
        reader,
        modules,
        ProposalHead(state_dim=STATE_DIM, num_modules=NUM_MODULES, cycles=3),
        SharedBufferWriter(state_dim=STATE_DIM, num_modules=NUM_MODULES, cycles=3),
        SharedBufferLatentBridge(
            state_dim=STATE_DIM,
            backbone_hidden_size=HIDDEN_SIZE,
            num_modules=NUM_MODULES,
            cycles=3,
        ),
        SecondPassGenerationPrefixAssembler(
            hidden_size=HIDDEN_SIZE,
            num_slots=NUM_MODULES,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            max_context_tokens=MAX_CONTEXT_TOKENS,
        ),
    )
    for component in components:
        component.eval()
    graph = CognitiveSharedBufferRequestOrchestrator(
        backbone=backbone,
        cognitive_config=config,
        policy=build_shared_buffer_policy_v1(),
        layer_sampler=components[0],
        goal_encoder=components[1],
        module_input_router=components[2],
        shared_buffer_reader=components[3],
        cognitive_modules=components[4],
        proposal_head=components[5],
        shared_buffer_writer=components[6],
        shared_buffer_latent_bridge=components[7],
        prefix_assembler=components[8],
        iris_generator=generator,
    )
    return graph, backbone, modules, reader, generator


def test_arm_c_runs_three_cycles_and_one_final_read_with_token_matched_prefix() -> None:
    graph, backbone, modules, reader, generator = build_graph()
    output = graph.execute("hello", strategy="greedy")

    assert graph.arm_id == "C"
    assert graph.arm_kind == "llm_modules_unlimited_shared_buffer"
    assert graph.cycles == 3
    assert graph.backbone is backbone
    assert graph.shared_buffer_policy_sha256 == shared_buffer_policy_sha256_v1()
    assert backbone.events == ["forward:hello", "embed"]
    assert reader.entry_counts == [0, 3, 6, 9]
    assert len(modules.observed_reads) == 3
    assert torch.equal(modules.observed_reads[0], torch.zeros_like(modules.observed_reads[0]))
    assert [state.entry_count for state in output.shared_buffer_trace] == [3, 6, 9]
    assert [read.attention_weights.shape[-1] for read in output.pre_update_reads] == [0, 3, 6]
    assert output.final_output_read.attention_weights.shape == (1, 3, 9)
    assert output.generation.text == "shared-buffer"
    assert [state.slots.shape[1] for state in output.generation.workspace_trace] == [3, 6, 9]
    assert all(bool(state.occupancy.all()) for state in output.generation.workspace_trace)
    assert output.receipt.strategy == "greedy"
    assert output.receipt.stop_token_ids == (2,)
    assert output.receipt.sampling_seed is None
    assert len(generator.calls) == 1
    prefix, strategy, trace, random_generator = generator.calls[0]
    assert strategy == "greedy"
    assert trace is output.generation.workspace_trace
    assert random_generator is None
    assert prefix.inputs_embeds.shape == (1, PROMPT_LENGTH + NUM_MODULES, HIDDEN_SIZE)
    assert torch.equal(prefix.position_ids[:, -NUM_MODULES:], torch.tensor([[3, 4, 5]]))


def test_arm_c_returns_independent_exact_prompt_and_native_trace_provenance() -> None:
    graph, backbone, _, _, _ = build_graph()
    output = graph.execute("hello", strategy="greedy")

    assert backbone.last_first_pass is not None
    source = backbone.last_first_pass
    assert torch.equal(output.prompt_input_ids, source.input_ids)
    assert torch.equal(output.prompt_attention_mask, source.attention_mask)
    assert output.prompt_input_ids.untyped_storage().data_ptr() != (
        source.input_ids.untyped_storage().data_ptr()
    )
    assert not output.prompt_input_ids.is_inference()
    for native, projected in zip(
        output.shared_buffer_trace,
        output.generation.workspace_trace,
        strict=True,
    ):
        assert torch.equal(native.entries, projected.slots)
        assert native.entries.untyped_storage().data_ptr() != (
            projected.slots.untyped_storage().data_ptr()
        )
        assert bool((projected.confidence == 1).all())
        assert bool((projected.ttl == 1).all())


def test_arm_c_sampling_preserves_explicit_generator_identity() -> None:
    graph, _, _, _, facade = build_graph()
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(9)
    output = graph.generate("sample", strategy="sampling", generator=random_generator)
    assert output.text == "shared-buffer"
    assert facade.calls[0][3] is random_generator


def test_arm_c_sidecar_hash_detects_parameter_mutation() -> None:
    graph, _, modules, _, _ = build_graph()
    original = graph.sidecar_state_sha256
    with torch.no_grad():
        next(modules.parameters()).add_(1)
    with pytest.raises(IrisSharedBufferRequestError, match="sidecar state changed"):
        _ = graph.sidecar_state_sha256
    assert len(original) == 64


def test_arm_c_detects_runtime_backbone_metadata_change() -> None:
    graph, backbone, _, _, _ = build_graph()
    backbone.reported_context += 1
    with pytest.raises(IrisSharedBufferRequestError, match="runtime boundary"):
        graph.execute("hello", strategy="greedy")


@pytest.mark.parametrize("prompt", ["", "  ", cast(str, 1)])
def test_arm_c_rejects_invalid_prompt_before_backbone(prompt: str) -> None:
    graph, backbone, _, _, _ = build_graph()
    with pytest.raises(IrisSharedBufferRequestError, match="prompt"):
        graph.execute(prompt, strategy="greedy")
    assert backbone.events == []


def test_arm_c_rejects_invalid_generation_ownership() -> None:
    graph, _, _, _, _ = build_graph()
    with pytest.raises(IrisSharedBufferRequestError, match="does not accept"):
        graph.execute(
            "hello",
            strategy="greedy",
            generator=torch.Generator(device="cpu"),
        )
    with pytest.raises(IrisSharedBufferRequestError, match="requires an explicit"):
        graph.execute("hello", strategy="sampling")


def test_arm_c_builder_constructs_exact_shared_buffer_topology() -> None:
    backbone = RecordingBackbone()
    graph = build_shared_buffer_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config(),
        stop_token_ids=(2,),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )
    assert type(graph) is CognitiveSharedBufferRequestOrchestrator
    assert graph.backbone is backbone
    names = tuple(name for name, _ in cast(Any, graph)._sidecar_modules)
    assert names == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "shared_buffer_reader",
        "cognitive_modules",
        "proposal_head",
        "shared_buffer_writer",
        "shared_buffer_latent_bridge",
        "prefix_assembler",
    )
    assert all(forbidden not in names for forbidden in ("workspace_arbiter", "slot_allocator"))


@pytest.mark.parametrize("mutation", ["trainable", "gradient", "context"])
def test_arm_c_builder_rejects_noncomparable_backbone(mutation: str) -> None:
    backbone = RecordingBackbone()
    if mutation == "trainable":
        backbone.trainable_count = 1
    elif mutation == "gradient":
        backbone.gradient_flag = True
    else:
        backbone.reported_context -= 1
    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_shared_buffer_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config(),
            stop_token_ids=(2,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def test_arm_c_builder_rejects_invalid_stop_ids() -> None:
    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_shared_buffer_request_orchestrator(
            backbone=RecordingBackbone(),
            cognitive_config=cognitive_config(),
            stop_token_ids=(VOCABULARY_SIZE,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
