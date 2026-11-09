"""Contract tests for the recurrent no-workspace arm-B request graph."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import iris.application as application
from iris.application import (
    CognitiveNoWorkspaceRequestOrchestrator,
    IrisCompositionError,
    IrisNoWorkspaceRequestError,
    build_no_workspace_request_orchestrator,
)
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
    CognitiveModuleOutput,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ModuleLatentBridge,
)
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    ModuleGenerationPrefixAssembler,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig
from iris.workspace import WorkspaceState

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
            "name": "arm-b-test-cognitive",
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
            logits=torch.zeros((1, PROMPT_LENGTH, VOCABULARY_SIZE), dtype=torch.float32),
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
    def __init__(self, *, mutate_zero_reads: bool = False) -> None:
        super().__init__(state_dim=STATE_DIM, num_modules=NUM_MODULES)
        self.observed_reads: list[torch.Tensor] = []
        self.mutate_zero_reads = mutate_zero_reads

    def forward(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        self.observed_reads.append(workspace_reads.detach().clone())
        output = super().forward(
            module_inputs,
            workspace_reads,
            goal_state,
            previous_states=previous_states,
        )
        if self.mutate_zero_reads:
            workspace_reads.add_(1)
        return output


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
            text="no-workspace",
            token_ids=torch.tensor([[11, 13]], dtype=torch.int64),
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=workspace_trace,
        )


def build_graph(
    *,
    backbone: RecordingBackbone | None = None,
    cognitive_modules: RecordingCognitiveModule | None = None,
) -> tuple[
    CognitiveNoWorkspaceRequestOrchestrator,
    RecordingBackbone,
    RecordingCognitiveModule,
    RecordingGenerator,
]:
    owned_backbone = backbone if backbone is not None else RecordingBackbone()
    config = cognitive_config()
    module_bank = cognitive_modules if cognitive_modules is not None else RecordingCognitiveModule()
    generator = RecordingGenerator(owned_backbone)
    components = (
        LayerSampler(
            config.layer_sampler,
            backbone_hidden_size=HIDDEN_SIZE,
            state_dim=STATE_DIM,
        ),
        GoalEncoder(state_dim=STATE_DIM),
        ModuleInputRouter(state_dim=STATE_DIM, num_modules=NUM_MODULES),
        module_bank,
        ModuleLatentBridge(
            state_dim=STATE_DIM,
            backbone_hidden_size=HIDDEN_SIZE,
            num_modules=NUM_MODULES,
        ),
        ModuleGenerationPrefixAssembler(
            hidden_size=HIDDEN_SIZE,
            num_modules=NUM_MODULES,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            max_context_tokens=MAX_CONTEXT_TOKENS,
        ),
    )
    for component in components:
        component.eval()
    graph = CognitiveNoWorkspaceRequestOrchestrator(
        backbone=owned_backbone,
        cognitive_config=config,
        layer_sampler=components[0],
        goal_encoder=components[1],
        module_input_router=components[2],
        cognitive_modules=components[3],
        module_latent_bridge=components[4],
        prefix_assembler=components[5],
        iris_generator=generator,
    )
    return graph, owned_backbone, module_bank, generator


def test_arm_b_runs_three_cycles_with_exact_zero_reads_and_no_workspace_trace() -> None:
    graph, backbone, modules, generator = build_graph()

    output = graph.execute("hello", strategy="greedy")

    assert graph.arm_id == "B"
    assert graph.arm_kind == "llm_modules_no_workspace"
    assert graph.cycles == 3
    assert graph.backbone is backbone
    assert backbone.events == ["forward:hello", "embed"]
    assert len(modules.observed_reads) == 3
    assert all(tuple(read.shape) == (1, NUM_MODULES, STATE_DIM) for read in modules.observed_reads)
    assert all(bool((read == 0).all()) for read in modules.observed_reads)
    assert output.generation.text == "no-workspace"
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
    assert prefix.inputs_embeds.shape == (1, PROMPT_LENGTH + NUM_MODULES, HIDDEN_SIZE)


def test_arm_b_returns_independent_exact_first_pass_provenance() -> None:
    graph, backbone, _, _ = build_graph()

    output = graph.execute("hello", strategy="greedy")

    assert backbone.last_first_pass is not None
    source = backbone.last_first_pass
    assert torch.equal(output.prompt_input_ids, source.input_ids)
    assert torch.equal(output.prompt_attention_mask, source.attention_mask)
    assert output.prompt_input_ids.untyped_storage().data_ptr() != (
        source.input_ids.untyped_storage().data_ptr()
    )
    assert output.prompt_attention_mask.untyped_storage().data_ptr() != (
        source.attention_mask.untyped_storage().data_ptr()
    )
    assert not output.prompt_input_ids.is_inference()
    assert not output.prompt_attention_mask.is_inference()


def test_arm_b_sampling_preserves_explicit_generator_identity() -> None:
    graph, _, _, facade = build_graph()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)

    output = graph.generate("sample", strategy="sampling", generator=generator)

    assert output.text == "no-workspace"
    assert facade.calls[0][3] is generator


def test_arm_b_detects_zero_read_mutation_before_generation() -> None:
    modules = RecordingCognitiveModule(mutate_zero_reads=True)
    graph, _, _, generator = build_graph(cognitive_modules=modules)

    with pytest.raises(IrisNoWorkspaceRequestError, match=r"mutated.*zero"):
        graph.execute("hello", strategy="greedy")

    assert generator.calls == []


def test_arm_b_sidecar_hash_detects_parameter_mutation() -> None:
    graph, _, modules, _ = build_graph()
    original = graph.sidecar_state_sha256
    with torch.no_grad():
        next(modules.parameters()).add_(1)

    with pytest.raises(IrisNoWorkspaceRequestError, match="sidecar state changed"):
        _ = graph.sidecar_state_sha256
    assert len(original) == 64


def test_arm_b_detects_runtime_backbone_metadata_change() -> None:
    graph, backbone, _, _ = build_graph()
    backbone.reported_context += 1

    with pytest.raises(IrisNoWorkspaceRequestError, match="runtime boundary"):
        graph.execute("hello", strategy="greedy")


@pytest.mark.parametrize("prompt", ["", "  ", cast(str, 1)])
def test_arm_b_rejects_invalid_prompt_before_backbone(prompt: str) -> None:
    graph, backbone, _, _ = build_graph()

    with pytest.raises(IrisNoWorkspaceRequestError, match="prompt"):
        graph.execute(prompt, strategy="greedy")

    assert backbone.events == []


def test_arm_b_rejects_invalid_generation_ownership() -> None:
    graph, _, _, _ = build_graph()
    with pytest.raises(IrisNoWorkspaceRequestError, match="does not accept"):
        graph.execute(
            "hello",
            strategy="greedy",
            generator=torch.Generator(device="cpu"),
        )
    with pytest.raises(IrisNoWorkspaceRequestError, match="requires an explicit"):
        graph.execute("hello", strategy="sampling")


def test_arm_b_builder_constructs_only_no_workspace_components() -> None:
    backbone = RecordingBackbone()

    graph = build_no_workspace_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config(),
        stop_token_ids=(2,),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )

    assert type(graph) is CognitiveNoWorkspaceRequestOrchestrator
    assert graph.backbone is backbone
    names = tuple(name for name, _ in cast(Any, graph)._sidecar_modules)
    assert names == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "cognitive_modules",
        "module_latent_bridge",
        "prefix_assembler",
    )
    assert all("workspace" not in name for name in names)


@pytest.mark.parametrize("mutation", ["trainable", "gradient", "context"])
def test_arm_b_builder_rejects_noncomparable_backbone(mutation: str) -> None:
    backbone = RecordingBackbone()
    if mutation == "trainable":
        backbone.trainable_count = 1
    elif mutation == "gradient":
        backbone.gradient_flag = True
    else:
        backbone.reported_context -= 1

    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_no_workspace_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config(),
            stop_token_ids=(2,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def test_arm_b_builder_rejects_invalid_stop_ids() -> None:
    with pytest.raises(IrisCompositionError, match="did not complete"):
        build_no_workspace_request_orchestrator(
            backbone=RecordingBackbone(),
            cognitive_config=cognitive_config(),
            stop_token_ids=(VOCABULARY_SIZE,),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def test_arm_b_api_is_public() -> None:
    assert (
        application.CognitiveNoWorkspaceRequestOrchestrator
        is CognitiveNoWorkspaceRequestOrchestrator
    )
    assert (
        application.build_no_workspace_request_orchestrator
        is build_no_workspace_request_orchestrator
    )
