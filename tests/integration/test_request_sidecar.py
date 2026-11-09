"""Offline CPU integration for the complete Iris request sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn

from iris.application import IrisRequestOrchestrator, build_request_orchestrator
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.workspace import LatentBridge, WorkspaceState

pytestmark = pytest.mark.integration

CPU = torch.device("cpu")
BATCH_SIZE = 1
PROMPT_LENGTH = 3
BACKBONE_HIDDEN_SIZE = 6
STATE_DIM = 4
NUM_MODULES = 3
NUM_SLOTS = 2
CYCLES = 3
WINNERS_PER_CYCLE = 2
MAX_PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 2
MAX_CONTEXT_TOKENS = 9
VOCABULARY_SIZE = 16
LAYER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


class _ForbiddenTokenizer:
    """Fail loudly if the orchestrator tries to tokenize a second time."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> Never:
        self.calls += 1
        raise AssertionError("offline request integration must not invoke a tokenizer")


class _TinyFrozenModel(nn.Module):
    """Local frozen tensor source with a trap for ``model.generate``."""

    embedding: nn.Embedding
    lm_head: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY_SIZE, BACKBONE_HIDDEN_SIZE)
        self.lm_head = nn.Linear(BACKBONE_HIDDEN_SIZE, VOCABULARY_SIZE, bias=False)
        self.generate_calls = 0
        self.requires_grad_(False)
        self.eval()

    def generate(self, *args: object, **kwargs: object) -> Never:
        self.generate_calls += 1
        raise AssertionError("IrisRequestOrchestrator must not call model.generate")


class _OfflineFrozenBackbone(BackboneAdapter):
    """Nominal adapter double that performs only tiny local frozen tensor work."""

    def __init__(self) -> None:
        self._offline_model = _TinyFrozenModel()
        self._offline_tokenizer = _ForbiddenTokenizer()
        self._model = self._offline_model
        self._tokenizer = self._offline_tokenizer
        self.forward_prompts: list[str] = []
        self.embed_inputs: list[torch.Tensor] = []
        self.embed_outputs: list[torch.Tensor] = []
        self.inference_flags: list[bool] = []

    @property
    def hidden_size(self) -> int:
        return BACKBONE_HIDDEN_SIZE

    @property
    def num_hidden_layers(self) -> int:
        return 4

    @property
    def vocabulary_size(self) -> int:
        return VOCABULARY_SIZE

    @property
    def max_position_embeddings(self) -> int:
        return MAX_CONTEXT_TOKENS

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
    def tokenizer_calls(self) -> int:
        return self._offline_tokenizer.calls

    @property
    def model_generate_calls(self) -> int:
        return self._offline_model.generate_calls

    def forward_prompt(self, text: str) -> BackboneFirstPassOutput:
        self.inference_flags.append(torch.is_inference_mode_enabled())
        self.forward_prompts.append(text)
        input_ids = torch.tensor([[0, 2, 7]], dtype=torch.int64, device=CPU)
        attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.int64, device=CPU)
        embedding_state = self._offline_model.embedding(input_ids)
        hidden_states = (
            embedding_state,
            *(
                torch.tanh(embedding_state + 0.125 * layer_index)
                for layer_index in range(1, self.num_hidden_layers + 1)
            ),
        )
        logits = self._offline_model.lm_head(hidden_states[-1])
        return BackboneFirstPassOutput(
            logits=logits,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.inference_flags.append(torch.is_inference_mode_enabled())
        self.embed_inputs.append(input_ids.detach().clone())
        embeddings = cast(torch.Tensor, self._offline_model.embedding(input_ids))
        self.embed_outputs.append(embeddings)
        return embeddings


class _OfflineIrisGenerator(IrisGenerator):
    """Terminal facade double that validates and preserves orchestration inputs."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        *,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...],
        sampling: TemperatureTopPSamplingConfig,
    ) -> None:
        self._backbone = backbone
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids
        self.sampling = sampling
        self.prefixes: list[SecondPassGenerationPrefixOutput] = []
        self.workspace_traces: list[tuple[WorkspaceState, ...]] = []
        self.inference_flags: list[bool] = []

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
        self.inference_flags.append(torch.is_inference_mode_enabled())
        if strategy != "greedy" or generator is not None:
            raise AssertionError("offline integration expects deterministic greedy decoding")
        self.prefixes.append(prefix)
        self.workspace_traces.append(workspace_trace)
        token_ids = torch.tensor(
            [[11, 12]],
            dtype=torch.int64,
            device=prefix.inputs_embeds.device,
        )
        return IrisGeneratorOutput(
            text="offline sidecar response",
            token_ids=token_ids,
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=workspace_trace,
        )


@dataclass(frozen=True, slots=True)
class _Harness:
    orchestrator: IrisRequestOrchestrator
    backbone: _OfflineFrozenBackbone
    iris_generator: _OfflineIrisGenerator
    latent_bridge: LatentBridge


def _cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-offline-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": NUM_MODULES,
            "layer_sampler": {"layer_fractions": LAYER_FRACTIONS},
        }
    )


def _workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-offline-workspace",
            "num_slots": NUM_SLOTS,
            "cycles": CYCLES,
            "winners_per_cycle": WINNERS_PER_CYCLE,
            "decay_rate": 0.5,
            "empty_threshold": 1e-4,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 1.0,
                "relevance_weight": 1.0,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def _build_harness() -> _Harness:
    cognitive_config = _cognitive_config()
    workspace_config = _workspace_config()
    backbone = _OfflineFrozenBackbone()
    sampling = TemperatureTopPSamplingConfig(temperature=1.0, top_p=1.0)
    with patch("iris.application.composition.IrisGenerator", _OfflineIrisGenerator):
        orchestrator = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
            stop_token_ids=(),
            sampling=sampling,
        )
    implementation = cast(Any, orchestrator)
    iris_generator = cast(_OfflineIrisGenerator, implementation._iris_generator)
    latent_bridge = cast(LatentBridge, implementation._latent_bridge)
    return _Harness(
        orchestrator=orchestrator,
        backbone=backbone,
        iris_generator=iris_generator,
        latent_bridge=latent_bridge,
    )


def _assert_canonical_trace(trace: tuple[WorkspaceState, ...]) -> None:
    assert len(trace) == CYCLES
    for state in trace:
        assert state.slots.shape == (BATCH_SIZE, NUM_SLOTS, STATE_DIM)
        assert state.slots.dtype == torch.float32
        assert state.slots.device == CPU
        assert state.age.shape == (BATCH_SIZE, NUM_SLOTS)
        assert state.occupancy.shape == (BATCH_SIZE, NUM_SLOTS)
        assert state.source_module.shape == (BATCH_SIZE, NUM_SLOTS)
        assert state.confidence.shape == (BATCH_SIZE, NUM_SLOTS)
        assert state.ttl.shape == (BATCH_SIZE, NUM_SLOTS)
        assert state.write_count.shape == (BATCH_SIZE, NUM_SLOTS)
        assert bool(torch.isfinite(state.slots).all())
        assert bool(torch.isfinite(state.confidence).all())
        assert bool(torch.isfinite(state.ttl).all())
        assert bool(state.occupancy.all())
        assert bool((state.source_module >= 0).all())
        assert bool((state.source_module < NUM_MODULES).all())
        assert bool((state.write_count >= 1).all())


def test_complete_greedy_request_runs_real_sidecar_entirely_offline_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        with (
            patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer_loader,
            patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model_loader,
        ):
            harness = _build_harness()
            global_rng_state = torch.get_rng_state().clone()

            output = harness.orchestrator.generate(
                "offline Iris request",
                strategy="greedy",
            )

            assert torch.equal(torch.get_rng_state(), global_rng_state)
            tokenizer_loader.assert_not_called()
            model_loader.assert_not_called()

    assert output.text == "offline sidecar response"
    assert output.token_ids.shape == (BATCH_SIZE, MAX_NEW_TOKENS)
    assert output.token_ids.dtype == torch.int64
    assert output.token_ids.device == CPU
    assert output.stop_reason == "max_new_tokens"
    assert output.matched_stop_token_id is None
    assert harness.backbone.forward_prompts == ["offline Iris request"]
    assert harness.backbone.inference_flags == [True, True]
    assert harness.backbone.trainable_parameter_count == 0
    assert harness.backbone.has_parameter_gradients is False
    assert harness.backbone.tokenizer_calls == 0
    assert harness.backbone.model_generate_calls == 0
    assert len(harness.backbone.embed_inputs) == 1
    assert torch.equal(
        harness.backbone.embed_inputs[0],
        torch.tensor([[0, 2, 7]], dtype=torch.int64),
    )

    assert harness.iris_generator.inference_flags == [True]
    assert harness.iris_generator.workspace_traces == [output.workspace_trace]
    assert harness.iris_generator.workspace_traces[0] is output.workspace_trace
    _assert_canonical_trace(output.workspace_trace)

    assert len(harness.iris_generator.prefixes) == 1
    prefix = harness.iris_generator.prefixes[0]
    assert prefix.inputs_embeds.shape == (
        BATCH_SIZE,
        PROMPT_LENGTH + NUM_SLOTS,
        BACKBONE_HIDDEN_SIZE,
    )
    assert prefix.inputs_embeds.dtype == torch.float32
    assert prefix.inputs_embeds.device == CPU
    assert bool(torch.isfinite(prefix.inputs_embeds).all())
    assert torch.equal(
        prefix.attention_mask,
        torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool),
    )
    assert torch.equal(
        prefix.position_ids,
        torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.int64),
    )
    assert torch.equal(
        prefix.inputs_embeds[:, :PROMPT_LENGTH],
        harness.backbone.embed_outputs[0],
    )

    with torch.inference_mode():
        expected_latent = harness.latent_bridge(
            output.workspace_trace[-1],
            torch.tensor([[0, 1, 1]], dtype=torch.int64),
        )
    assert torch.equal(
        prefix.inputs_embeds[:, PROMPT_LENGTH:],
        expected_latent.workspace_embeddings,
    )
