"""Contract tests for one-request Iris cognitive orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch import nn

import iris.application as application
from iris.application.errors import IrisRequestError
from iris.application.request import (
    IrisRequestExecutionOutput,
    IrisRequestExecutionReceiptV1,
    IrisRequestOrchestrator,
)
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput, BackboneForwardOutput
from iris.cognitive import (
    CognitiveModule,
    CognitiveModuleOutput,
    GoalEncoder,
    GoalEncoderOutput,
    LayerSampler,
    LayerSamplerOutput,
    ModuleInputRouter,
    ModuleInputRouterOutput,
    ModuleProposal,
    ProposalHead,
)
from iris.generation import (
    IrisGenerator,
    IrisGeneratorOutput,
    SecondPassGenerationPrefixAssembler,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.workspace import (
    LatentBridge,
    LatentBridgeOutput,
    SlotAllocator,
    SlotAllocatorOutput,
    WorkspaceArbiter,
    WorkspaceArbiterOutput,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceReaderOutput,
    WorkspaceState,
    WorkspaceWriter,
)

CPU = torch.device("cpu")
BATCH_SIZE = 1
PROMPT_LENGTH = 3
BACKBONE_HIDDEN_SIZE = 6
STATE_DIM = 4
NUM_MODULES = 3
NUM_SLOTS = 2
CYCLES = 3
WINNERS = 2
MAX_PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 2
MAX_CONTEXT_TOKENS = 10
LAYER_FRACTIONS = (0.2, 0.4, 0.7, 1.0)

CYCLIC_STAGES = frozenset(
    {"decay", "reader", "cognitive", "proposal", "arbiter", "allocator", "writer"}
)


@dataclass(slots=True)
class EventControl:
    """Shared deterministic event log and adversarial failure controller."""

    events: list[str] = field(default_factory=list)
    inference_modes: list[bool] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    failure_stage: str | None = None
    failure: RuntimeError = field(default_factory=lambda: RuntimeError("stage exploded"))
    mutation_stage: str | None = None
    mutation_action: Callable[[], None] | None = None

    def record(self, stage: str) -> int:
        index = self.counts.get(stage, 0)
        self.counts[stage] = index + 1
        label = f"{stage}:{index}" if stage in CYCLIC_STAGES else stage
        self.events.append(label)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        if self.mutation_stage == stage and self.mutation_action is not None:
            self.mutation_action()
        if self.failure_stage == stage:
            raise self.failure
        return index


@dataclass(frozen=True, slots=True)
class TensorMutationObservation:
    before_version: int
    after_version: int
    before_storage: int
    after_storage: int


def mutate_prompt_tensor(
    tensor: torch.Tensor,
    *,
    kind: str,
) -> TensorMutationObservation:
    """Apply one malicious callback mutation and expose its version/storage evidence."""
    before_version = tensor._version
    before_storage = tensor.untyped_storage().data_ptr()
    if kind == "persistent":
        tensor.add_(1)
    elif kind == "restore":
        original = tensor.detach().clone()
        tensor.add_(1)
        tensor.copy_(original)
    else:
        assert kind == "rebind"
        with torch.inference_mode(False):
            replacement = tensor.detach().clone()
        tensor.data = replacement
    return TensorMutationObservation(
        before_version=before_version,
        after_version=tensor._version,
        before_storage=before_storage,
        after_storage=tensor.untyped_storage().data_ptr(),
    )


def cognitive_config(*, num_modules: int = NUM_MODULES) -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "test-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": num_modules,
            "layer_sampler": {"layer_fractions": LAYER_FRACTIONS},
        }
    )


def workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "test-workspace",
            "num_slots": NUM_SLOTS,
            "cycles": CYCLES,
            "winners_per_cycle": WINNERS,
            "decay_rate": 0.5,
            "empty_threshold": 0.01,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 2.0,
                "relevance_weight": 3.0,
            },
            "allocator": {
                "empty_bonus": 4.0,
                "allocation_age_penalty": 5.0,
                "retention_age_penalty": 6.0,
            },
        }
    )


def marked_workspace(marker: int) -> WorkspaceState:
    """Create one unmistakable functional workspace without shared storage."""
    return WorkspaceState(
        slots=torch.full(
            (BATCH_SIZE, NUM_SLOTS, STATE_DIM),
            float(marker),
            dtype=torch.float32,
        ),
        age=torch.full((BATCH_SIZE, NUM_SLOTS), marker, dtype=torch.int64),
        occupancy=torch.ones((BATCH_SIZE, NUM_SLOTS), dtype=torch.bool),
        source_module=torch.tensor([[0, 1]], dtype=torch.int64),
        confidence=torch.full((BATCH_SIZE, NUM_SLOTS), 0.75, dtype=torch.float32),
        ttl=torch.full((BATCH_SIZE, NUM_SLOTS), 2.0, dtype=torch.float32),
        write_count=torch.full((BATCH_SIZE, NUM_SLOTS), marker + 1, dtype=torch.int64),
    )


class SpyBackbone(BackboneAdapter):
    """Nominal frozen adapter double with one authoritative tokenization."""

    def __init__(self, control: EventControl) -> None:
        self.control = control
        self.forward_prompts: list[str] = []
        self.first_pass_outputs: list[BackboneFirstPassOutput] = []
        self.embed_inputs: list[torch.Tensor] = []
        self.embed_outputs: list[torch.Tensor] = []
        self.direct_tokenize_calls = 0
        self.return_legacy_output = False

    @property
    def hidden_size(self) -> int:
        return BACKBONE_HIDDEN_SIZE

    @property
    def max_position_embeddings(self) -> int:
        return MAX_CONTEXT_TOKENS

    @property
    def max_prompt_tokens(self) -> int:
        return MAX_PROMPT_TOKENS

    @property
    def max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        self.direct_tokenize_calls += 1
        raise AssertionError(f"orchestrator retokenized prompt {text!r}")

    def forward_prompt(self, text: str) -> BackboneFirstPassOutput:
        self.control.record("first_pass")
        self.forward_prompts.append(text)
        input_ids = torch.tensor([[0, 2, 7]], dtype=torch.int64)
        attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.int64)
        hidden_states = tuple(
            torch.full(
                (BATCH_SIZE, PROMPT_LENGTH, BACKBONE_HIDDEN_SIZE),
                float(index),
            )
            for index in range(5)
        )
        if self.return_legacy_output:
            return cast(
                BackboneFirstPassOutput,
                BackboneForwardOutput(
                    logits=torch.ones(BATCH_SIZE, PROMPT_LENGTH, 8),
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                ),
            )
        output = BackboneFirstPassOutput(
            logits=torch.ones(BATCH_SIZE, PROMPT_LENGTH, 8),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        self.first_pass_outputs.append(output)
        return output

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.control.record("embed")
        self.embed_inputs.append(input_ids)
        output = torch.arange(
            BATCH_SIZE * PROMPT_LENGTH * BACKBONE_HIDDEN_SIZE,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, PROMPT_LENGTH, BACKBONE_HIDDEN_SIZE)
        self.embed_outputs.append(output)
        return output


class SpyLayerSampler(LayerSampler):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.backbone_hidden_size = BACKBONE_HIDDEN_SIZE
        self.layer_fractions = LAYER_FRACTIONS
        self.calls: list[BackboneForwardOutput] = []
        self.outputs: list[LayerSamplerOutput] = []
        self.mutation_target: str | None = None
        self.mutation_kind: str | None = None
        self.mutation_observations: list[TensorMutationObservation] = []
        self.eval()

    def forward(self, backbone_output: BackboneForwardOutput) -> LayerSamplerOutput:
        self.control.record("sampler")
        self.calls.append(backbone_output)
        if self.mutation_target is not None and self.mutation_kind is not None:
            first_pass = cast(BackboneFirstPassOutput, backbone_output)
            target = cast(torch.Tensor, getattr(first_pass, self.mutation_target))
            self.mutation_observations.append(mutate_prompt_tensor(target, kind=self.mutation_kind))
        output = LayerSamplerOutput(
            context_features=torch.arange(
                BATCH_SIZE * PROMPT_LENGTH * STATE_DIM,
                dtype=torch.float32,
            ).reshape(BATCH_SIZE, PROMPT_LENGTH, STATE_DIM),
            attention_mask=backbone_output.attention_mask.to(dtype=torch.bool),
            mixing_weights=torch.full((4,), 0.25, dtype=torch.float32),
            selected_layer_indices=(1, 2, 3, 4),
        )
        self.outputs.append(output)
        return output


class SpyGoalEncoder(GoalEncoder):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.outputs: list[GoalEncoderOutput] = []
        self.eval()

    def forward(
        self,
        context_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> GoalEncoderOutput:
        self.control.record("goal")
        self.calls.append((context_features, attention_mask))
        output = GoalEncoderOutput(
            goal_state=torch.full((BATCH_SIZE, STATE_DIM), 0.5),
            attention_weights=torch.full(
                (BATCH_SIZE, PROMPT_LENGTH),
                1.0 / PROMPT_LENGTH,
            ),
        )
        self.outputs.append(output)
        return output


class SpyModuleInputRouter(ModuleInputRouter):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.outputs: list[ModuleInputRouterOutput] = []
        self.eval()

    def forward(
        self,
        context_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ModuleInputRouterOutput:
        self.control.record("router")
        self.calls.append((context_features, attention_mask))
        output = ModuleInputRouterOutput(
            module_inputs=torch.full((BATCH_SIZE, NUM_MODULES, STATE_DIM), 0.25),
            attention_weights=torch.full(
                (BATCH_SIZE, NUM_MODULES, PROMPT_LENGTH),
                1.0 / PROMPT_LENGTH,
            ),
        )
        self.outputs.append(output)
        return output


class SpyWorkspaceReader(WorkspaceReader):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.inputs: list[WorkspaceState] = []
        self.outputs: list[WorkspaceReaderOutput] = []
        self.eval()

    def forward(self, state: WorkspaceState) -> WorkspaceReaderOutput:
        index = self.control.record("reader")
        self.inputs.append(state)
        output = WorkspaceReaderOutput(
            workspace_reads=torch.full(
                (BATCH_SIZE, NUM_MODULES, STATE_DIM),
                float(index + 1),
            ),
            attention_weights=torch.full(
                (BATCH_SIZE, NUM_MODULES, NUM_SLOTS),
                1.0 / NUM_SLOTS,
            ),
        )
        self.outputs.append(output)
        return output


class SpyCognitiveModules(CognitiveModule):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.initial_calls: list[dict[str, object]] = []
        self.initial_outputs: list[torch.Tensor] = []
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self.outputs: list[CognitiveModuleOutput] = []
        self.eval()

    def initial_state(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        self.control.record("initial_state")
        self.initial_calls.append({"batch_size": batch_size, "dtype": dtype, "device": device})
        output = torch.zeros(
            (batch_size, self.num_modules, self.state_dim),
            dtype=dtype,
            device=device,
        )
        self.initial_outputs.append(output)
        return output

    def forward(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        index = self.control.record("cognitive")
        self.calls.append((module_inputs, workspace_reads, goal_state, previous_states))
        output = CognitiveModuleOutput(
            module_states=torch.full(
                (BATCH_SIZE, NUM_MODULES, STATE_DIM),
                float(index + 1),
            )
        )
        self.outputs.append(output)
        return output


class SpyProposalHead(ProposalHead):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.cycles = CYCLES
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[ModuleProposal] = []
        self.eval()

    def forward(self, module_states: torch.Tensor) -> ModuleProposal:
        index = self.control.record("proposal")
        self.inputs.append(module_states)
        output = ModuleProposal(
            content=torch.full(
                (BATCH_SIZE, NUM_MODULES, STATE_DIM),
                float(index + 1),
            ),
            key=torch.full((BATCH_SIZE, NUM_MODULES, STATE_DIM), 0.5),
            bid=torch.full((BATCH_SIZE, NUM_MODULES), 0.5),
            confidence=torch.full((BATCH_SIZE, NUM_MODULES), 0.5),
            ttl=torch.full((BATCH_SIZE, NUM_MODULES), 2.0),
        )
        self.outputs.append(output)
        return output


class SpyWorkspaceArbiter(WorkspaceArbiter):
    def __init__(self, control: EventControl, config: WorkspaceConfig) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.winners_per_cycle = WINNERS
        self.confidence_weight = config.arbiter.confidence_weight
        self.novelty_weight = config.arbiter.novelty_weight
        self.relevance_weight = config.arbiter.relevance_weight
        self.calls: list[tuple[ModuleProposal, WorkspaceState, torch.Tensor]] = []
        self.outputs: list[WorkspaceArbiterOutput] = []
        self.eval()

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        goal_state: torch.Tensor,
    ) -> WorkspaceArbiterOutput:
        self.control.record("arbiter")
        self.calls.append((proposal, workspace, goal_state))
        scores = torch.arange(NUM_MODULES, dtype=torch.float32).unsqueeze(0)
        output = WorkspaceArbiterOutput(
            scores=scores,
            novelty=torch.ones_like(scores),
            relevance=torch.ones_like(scores),
            soft_probabilities=torch.softmax(scores, dim=-1),
            selection_weights=torch.ones_like(scores),
            winner_mask=torch.tensor([[True, True, False]]),
            winner_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        )
        self.outputs.append(output)
        return output


class SpySlotAllocator(SlotAllocator):
    def __init__(self, control: EventControl, config: WorkspaceConfig) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.winners_per_cycle = WINNERS
        self.empty_bonus = config.allocator.empty_bonus
        self.allocation_age_penalty = config.allocator.allocation_age_penalty
        self.retention_age_penalty = config.allocator.retention_age_penalty
        self.calls: list[tuple[ModuleProposal, WorkspaceState, WorkspaceArbiterOutput]] = []
        self.outputs: list[SlotAllocatorOutput] = []
        self.eval()

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        arbitration: WorkspaceArbiterOutput,
    ) -> SlotAllocatorOutput:
        self.control.record("allocator")
        self.calls.append((proposal, workspace, arbitration))
        matrix = torch.zeros(BATCH_SIZE, NUM_MODULES, NUM_SLOTS)
        assignment = torch.tensor([[[True, False], [False, True], [False, False]]])
        output = SlotAllocatorOutput(
            allocation_scores=matrix,
            retention_scores=torch.zeros(BATCH_SIZE, NUM_SLOTS),
            target_mask=torch.ones(BATCH_SIZE, NUM_SLOTS, dtype=torch.bool),
            assignment_mask=assignment,
            assignment_probabilities=assignment.to(dtype=torch.float32),
            assignment_weights=assignment.to(dtype=torch.float32),
            module_indices=torch.tensor([[0, 1]], dtype=torch.int64),
            slot_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        )
        self.outputs.append(output)
        return output


class SpyWorkspaceWriter(WorkspaceWriter):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.winners_per_cycle = WINNERS
        self.calls: list[
            tuple[ModuleProposal, WorkspaceState, SlotAllocatorOutput, torch.Tensor]
        ] = []
        self.outputs: list[WorkspaceState] = []
        self.eval()

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        allocation: SlotAllocatorOutput,
        goal_state: torch.Tensor,
    ) -> WorkspaceState:
        index = self.control.record("writer")
        self.calls.append((proposal, workspace, allocation, goal_state))
        output = marked_workspace(10 + index)
        self.outputs.append(output)
        return output


class SpyWorkspaceDecay(WorkspaceDecay):
    def __init__(self, control: EventControl, config: WorkspaceConfig) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.decay_rate = config.decay_rate
        self.empty_threshold = config.empty_threshold
        self.inputs: list[WorkspaceState] = []
        self.outputs: list[WorkspaceState] = []
        self.eval()

    def forward(self, workspace: WorkspaceState) -> WorkspaceState:
        index = self.control.record("decay")
        self.inputs.append(workspace)
        output = marked_workspace(20 + index)
        self.outputs.append(output)
        return output


class SpyLatentBridge(LatentBridge):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.backbone_hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_slots = NUM_SLOTS
        self.calls: list[tuple[WorkspaceState, torch.Tensor]] = []
        self.outputs: list[LatentBridgeOutput] = []
        self.mutation_kind: str | None = None
        self.mutation_observations: list[TensorMutationObservation] = []
        self.eval()

    def forward(
        self,
        workspace: WorkspaceState,
        prompt_attention_mask: torch.Tensor,
    ) -> LatentBridgeOutput:
        self.control.record("bridge")
        self.calls.append((workspace, prompt_attention_mask))
        if self.mutation_kind is not None:
            self.mutation_observations.append(
                mutate_prompt_tensor(prompt_attention_mask, kind=self.mutation_kind)
            )
        logical_prompt_length = prompt_attention_mask.to(dtype=torch.int64).sum(dim=-1)
        output = LatentBridgeOutput(
            workspace_embeddings=torch.full(
                (BATCH_SIZE, NUM_SLOTS, BACKBONE_HIDDEN_SIZE),
                0.125,
            ),
            workspace_attention_mask=torch.ones(
                BATCH_SIZE,
                NUM_SLOTS,
                dtype=torch.bool,
            ),
            workspace_position_ids=logical_prompt_length.unsqueeze(1)
            + torch.arange(NUM_SLOTS, dtype=torch.int64),
        )
        self.outputs.append(output)
        return output


@dataclass(frozen=True, slots=True)
class PrefixCall:
    prompt_embeddings: torch.Tensor
    prompt_attention_mask: torch.Tensor
    workspace: LatentBridgeOutput


class SpyPrefixAssembler(SecondPassGenerationPrefixAssembler):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_slots = NUM_SLOTS
        self.max_prompt_tokens = MAX_PROMPT_TOKENS
        self.max_new_tokens = MAX_NEW_TOKENS
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.child = nn.Identity()
        self.calls: list[PrefixCall] = []
        self.outputs: list[SecondPassGenerationPrefixOutput] = []
        self.mutation_kind: str | None = None
        self.mutation_observations: list[TensorMutationObservation] = []
        self.eval()

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        workspace: LatentBridgeOutput,
    ) -> SecondPassGenerationPrefixOutput:
        self.control.record("prefix")
        self.calls.append(
            PrefixCall(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=prompt_attention_mask,
                workspace=workspace,
            )
        )
        if self.mutation_kind is not None:
            self.mutation_observations.append(
                mutate_prompt_tensor(prompt_attention_mask, kind=self.mutation_kind)
            )
        mask = torch.cat(
            (
                prompt_attention_mask.to(dtype=torch.bool),
                workspace.workspace_attention_mask,
            ),
            dim=1,
        )
        raw_positions = mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        output = SecondPassGenerationPrefixOutput(
            inputs_embeds=torch.cat(
                (prompt_embeddings, workspace.workspace_embeddings),
                dim=1,
            ),
            attention_mask=mask,
            position_ids=torch.where(mask, raw_positions, torch.zeros_like(raw_positions)),
        )
        self.outputs.append(output)
        return output


@dataclass(frozen=True, slots=True)
class FacadeCall:
    prefix: SecondPassGenerationPrefixOutput
    strategy: str
    workspace_trace: tuple[WorkspaceState, ...]
    generator: torch.Generator | None


class SpyIrisGenerator(IrisGenerator):
    def __init__(self, control: EventControl, backbone: BackboneAdapter) -> None:
        self.control = control
        self._backbone = backbone
        self.max_new_tokens = MAX_NEW_TOKENS
        self.stop_token_ids = (2,)
        self.sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
        self.calls: list[FacadeCall] = []
        self.outputs: list[IrisGeneratorOutput] = []
        self.draws_before_return = 0
        self.error_after_draw: RuntimeError | None = None

    @property
    def backbone(self) -> BackboneAdapter:
        return self._backbone

    def generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        strategy: Any,
        workspace_trace: tuple[WorkspaceState, ...],
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        self.control.record("facade")
        self.calls.append(
            FacadeCall(
                prefix=prefix,
                strategy=cast(str, strategy),
                workspace_trace=workspace_trace,
                generator=generator,
            )
        )
        original_generator_state: torch.Tensor | None = None
        if self.draws_before_return:
            assert isinstance(generator, torch.Generator)
            original_generator_state = generator.get_state().clone()
            for _ in range(self.draws_before_return):
                torch.rand((), generator=generator)
        if self.error_after_draw is not None:
            if original_generator_state is not None:
                assert isinstance(generator, torch.Generator)
                generator.set_state(original_generator_state)
            raise self.error_after_draw
        output = IrisGeneratorOutput(
            text="orchestrated response",
            token_ids=torch.tensor([[1, 3]], dtype=torch.int64),
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=workspace_trace,
        )
        self.outputs.append(output)
        return output


@dataclass(slots=True)
class RequestParts:
    control: EventControl
    cognitive_config: CognitiveConfig
    workspace_config: WorkspaceConfig
    backbone: SpyBackbone
    layer_sampler: SpyLayerSampler
    goal_encoder: SpyGoalEncoder
    module_input_router: SpyModuleInputRouter
    workspace_reader: SpyWorkspaceReader
    cognitive_modules: SpyCognitiveModules
    proposal_head: SpyProposalHead
    workspace_arbiter: SpyWorkspaceArbiter
    slot_allocator: SpySlotAllocator
    workspace_writer: SpyWorkspaceWriter
    workspace_decay: SpyWorkspaceDecay
    latent_bridge: SpyLatentBridge
    prefix_assembler: SpyPrefixAssembler
    iris_generator: SpyIrisGenerator
    initial_workspace: WorkspaceState
    empty_calls: list[dict[str, object]] = field(default_factory=list)

    def make_empty_workspace(self, **kwargs: object) -> WorkspaceState:
        self.control.record("empty")
        self.empty_calls.append(kwargs)
        return self.initial_workspace


def make_parts() -> RequestParts:
    control = EventControl()
    cognitive = cognitive_config()
    workspace = workspace_config()
    backbone = SpyBackbone(control)
    return RequestParts(
        control=control,
        cognitive_config=cognitive,
        workspace_config=workspace,
        backbone=backbone,
        layer_sampler=SpyLayerSampler(control),
        goal_encoder=SpyGoalEncoder(control),
        module_input_router=SpyModuleInputRouter(control),
        workspace_reader=SpyWorkspaceReader(control),
        cognitive_modules=SpyCognitiveModules(control),
        proposal_head=SpyProposalHead(control),
        workspace_arbiter=SpyWorkspaceArbiter(control, workspace),
        slot_allocator=SpySlotAllocator(control, workspace),
        workspace_writer=SpyWorkspaceWriter(control),
        workspace_decay=SpyWorkspaceDecay(control, workspace),
        latent_bridge=SpyLatentBridge(control),
        prefix_assembler=SpyPrefixAssembler(control),
        iris_generator=SpyIrisGenerator(control, backbone),
        initial_workspace=WorkspaceState.empty(
            batch_size=BATCH_SIZE,
            num_slots=NUM_SLOTS,
            state_dim=STATE_DIM,
            dtype=torch.float32,
            device=CPU,
        ),
    )


def build_orchestrator(parts: RequestParts) -> IrisRequestOrchestrator:
    return IrisRequestOrchestrator(
        backbone=parts.backbone,
        cognitive_config=parts.cognitive_config,
        workspace_config=parts.workspace_config,
        layer_sampler=parts.layer_sampler,
        goal_encoder=parts.goal_encoder,
        module_input_router=parts.module_input_router,
        workspace_reader=parts.workspace_reader,
        cognitive_modules=parts.cognitive_modules,
        proposal_head=parts.proposal_head,
        workspace_arbiter=parts.workspace_arbiter,
        slot_allocator=parts.slot_allocator,
        workspace_writer=parts.workspace_writer,
        workspace_decay=parts.workspace_decay,
        latent_bridge=parts.latent_bridge,
        prefix_assembler=parts.prefix_assembler,
        iris_generator=parts.iris_generator,
    )


def run_request(
    parts: RequestParts,
    orchestrator: IrisRequestOrchestrator,
    *,
    prompt: Any = "  preserve this exact prompt  ",
    strategy: Any = "greedy",
    generator: Any = None,
) -> IrisGeneratorOutput:
    with patch(
        "iris.application.request.WorkspaceState.empty",
        side_effect=parts.make_empty_workspace,
    ):
        return orchestrator.generate(
            prompt,
            strategy=strategy,
            generator=generator,
        )


HAPPY_EVENTS = [
    "first_pass",
    "sampler",
    "goal",
    "router",
    "initial_state",
    "empty",
    "reader:0",
    "cognitive:0",
    "proposal:0",
    "arbiter:0",
    "allocator:0",
    "writer:0",
    "decay:0",
    "reader:1",
    "cognitive:1",
    "proposal:1",
    "arbiter:1",
    "allocator:1",
    "writer:1",
    "decay:1",
    "reader:2",
    "cognitive:2",
    "proposal:2",
    "arbiter:2",
    "allocator:2",
    "writer:2",
    "embed",
    "bridge",
    "prefix",
    "facade",
]


def test_public_application_import_surface_is_canonical() -> None:
    assert application.IrisRequestExecutionOutput is IrisRequestExecutionOutput
    assert application.IrisRequestExecutionReceiptV1 is IrisRequestExecutionReceiptV1
    assert application.IrisRequestOrchestrator is IrisRequestOrchestrator
    assert application.IrisRequestError is IrisRequestError


def test_execute_returns_exact_first_pass_token_provenance_without_retokenizing() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)

    with patch(
        "iris.application.request.WorkspaceState.empty",
        side_effect=parts.make_empty_workspace,
    ):
        execution = orchestrator.execute(
            "  preserve this exact prompt  ",
            strategy="greedy",
        )

    assert type(execution) is IrisRequestExecutionOutput
    assert execution.generation is parts.iris_generator.outputs[0]
    first_pass = parts.backbone.first_pass_outputs[0]
    assert torch.equal(execution.prompt_input_ids, first_pass.input_ids)
    assert torch.equal(execution.prompt_attention_mask, first_pass.attention_mask)
    assert execution.receipt is not None
    assert execution.receipt.strategy == "greedy"
    assert execution.receipt.stop_token_ids == (2,)
    assert execution.receipt.sampling == parts.iris_generator.sampling
    assert execution.receipt.sampling_seed is None
    assert (
        execution.prompt_input_ids
        is cast(BackboneFirstPassOutput, parts.layer_sampler.calls[0]).input_ids
    )
    assert execution.prompt_input_ids.untyped_storage().data_ptr() != (
        first_pass.input_ids.untyped_storage().data_ptr()
    )
    assert execution.prompt_attention_mask.untyped_storage().data_ptr() != (
        first_pass.attention_mask.untyped_storage().data_ptr()
    )
    assert parts.backbone.direct_tokenize_calls == 0
    assert parts.control.events == HAPPY_EVENTS


def test_complete_request_preserves_exact_order_recurrence_and_trace_identity() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)

    output = run_request(parts, orchestrator)

    assert parts.control.events == HAPPY_EVENTS
    assert all(parts.control.inference_modes)
    assert not torch.is_inference_mode_enabled()
    assert parts.backbone.forward_prompts == ["  preserve this exact prompt  "]
    assert parts.backbone.direct_tokenize_calls == 0
    assert len(parts.backbone.first_pass_outputs) == 1
    first_pass = parts.backbone.first_pass_outputs[0]
    sampler_input = cast(BackboneFirstPassOutput, parts.layer_sampler.calls[0])
    assert torch.equal(sampler_input.input_ids, first_pass.input_ids)
    assert torch.equal(sampler_input.attention_mask, first_pass.attention_mask)
    assert first_pass.input_ids.is_inference()
    assert first_pass.attention_mask.is_inference()
    for owned, frozen in (
        (sampler_input.input_ids, first_pass.input_ids),
        (sampler_input.attention_mask, first_pass.attention_mask),
    ):
        assert not owned.is_inference()
        assert owned.requires_grad is False
        assert owned.grad_fn is None
        assert owned.dtype == frozen.dtype
        assert owned.device == frozen.device
        assert owned.untyped_storage().data_ptr() != frozen.untyped_storage().data_ptr()

    sampled = parts.layer_sampler.outputs[0]
    assert parts.goal_encoder.calls == [(sampled.context_features, sampled.attention_mask)]
    assert parts.module_input_router.calls == [(sampled.context_features, sampled.attention_mask)]
    assert parts.cognitive_modules.initial_calls == [
        {"batch_size": 1, "dtype": torch.float32, "device": CPU}
    ]
    assert parts.empty_calls == [
        {
            "batch_size": 1,
            "num_slots": NUM_SLOTS,
            "state_dim": STATE_DIM,
            "dtype": torch.float32,
            "device": CPU,
        }
    ]
    empty = parts.initial_workspace
    assert not bool(empty.occupancy.any())
    assert not bool(empty.slots.any())
    assert parts.workspace_reader.inputs[0] is empty

    routed = parts.module_input_router.outputs[0]
    goal = parts.goal_encoder.outputs[0]
    assert len(parts.cognitive_modules.calls) == CYCLES
    for index, call in enumerate(parts.cognitive_modules.calls):
        module_inputs, workspace_reads, goal_state, previous_states = call
        assert module_inputs is routed.module_inputs
        assert workspace_reads is parts.workspace_reader.outputs[index].workspace_reads
        assert goal_state is goal.goal_state
        if index == 0:
            assert previous_states is parts.cognitive_modules.initial_outputs[0]
        else:
            assert previous_states is parts.cognitive_modules.outputs[index - 1].module_states
        assert (
            parts.proposal_head.inputs[index]
            is parts.cognitive_modules.outputs[index].module_states
        )

    assert len(parts.workspace_decay.inputs) == CYCLES - 1
    assert len(parts.workspace_writer.outputs) == CYCLES
    for index in range(CYCLES - 1):
        assert parts.workspace_decay.inputs[index] is parts.workspace_writer.outputs[index]
        assert parts.workspace_reader.inputs[index + 1] is parts.workspace_decay.outputs[index]
        assert parts.workspace_reader.inputs[index + 1] is not parts.workspace_writer.outputs[index]
        torch.testing.assert_close(
            parts.workspace_writer.outputs[index].slots,
            torch.full_like(parts.workspace_writer.outputs[index].slots, float(10 + index)),
        )

    for index, reader_workspace in enumerate(parts.workspace_reader.inputs):
        proposal = parts.proposal_head.outputs[index]
        arbitration_call = parts.workspace_arbiter.calls[index]
        allocation_call = parts.slot_allocator.calls[index]
        writer_call = parts.workspace_writer.calls[index]
        assert arbitration_call == (proposal, reader_workspace, goal.goal_state)
        assert allocation_call == (
            proposal,
            reader_workspace,
            parts.workspace_arbiter.outputs[index],
        )
        assert writer_call == (
            proposal,
            reader_workspace,
            parts.slot_allocator.outputs[index],
            goal.goal_state,
        )

    assert len(parts.backbone.embed_inputs) == 1
    assert torch.equal(parts.backbone.embed_inputs[0], first_pass.input_ids)
    assert not parts.backbone.embed_inputs[0].is_inference()
    assert (
        parts.backbone.embed_inputs[0].untyped_storage().data_ptr()
        != first_pass.input_ids.untyped_storage().data_ptr()
    )
    assert parts.latent_bridge.calls[0][0] is parts.workspace_writer.outputs[-1]
    assert torch.equal(parts.latent_bridge.calls[0][1], first_pass.attention_mask)
    assert not parts.latent_bridge.calls[0][1].is_inference()
    assert (
        parts.latent_bridge.calls[0][1].untyped_storage().data_ptr()
        != first_pass.attention_mask.untyped_storage().data_ptr()
    )
    prefix_call = parts.prefix_assembler.calls[0]
    assert prefix_call.prompt_embeddings is parts.backbone.embed_outputs[0]
    assert torch.equal(prefix_call.prompt_attention_mask, first_pass.attention_mask)
    assert not prefix_call.prompt_attention_mask.is_inference()
    assert (
        prefix_call.prompt_attention_mask.untyped_storage().data_ptr()
        != first_pass.attention_mask.untyped_storage().data_ptr()
    )
    assert prefix_call.workspace is parts.latent_bridge.outputs[0]

    assert len(parts.iris_generator.calls) == 1
    facade_call = parts.iris_generator.calls[0]
    assert facade_call.prefix is parts.prefix_assembler.outputs[0]
    assert facade_call.strategy == "greedy"
    assert facade_call.generator is None
    assert len(facade_call.workspace_trace) == CYCLES
    assert all(
        trace_state is writer_state
        for trace_state, writer_state in zip(
            facade_call.workspace_trace,
            parts.workspace_writer.outputs,
            strict=True,
        )
    )
    assert output is parts.iris_generator.outputs[0]
    assert output.workspace_trace is facade_call.workspace_trace
    assert parts.latent_bridge.calls[0][0] is output.workspace_trace[-1]


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("prompt", "strategy", "generator", "message"),
    [
        (None, "greedy", None, "prompt"),
        ("", "greedy", None, "prompt"),
        (" \t\n", "greedy", None, "prompt"),
        (StringSubclass("hello"), "greedy", None, "prompt"),
        ("hello", None, None, "strategy"),
        ("hello", "beam", None, "strategy"),
        ("hello", StringSubclass("greedy"), None, "strategy"),
        ("hello", "greedy", torch.Generator(), "do not accept"),
        ("hello", "sampling", None, "explicit torch.Generator"),
        ("hello", "sampling", object(), "explicit torch.Generator"),
        ("hello", "sampling", torch.default_generator, "caller-owned"),
    ],
)
def test_invalid_request_is_rejected_before_the_backbone(
    prompt: Any,
    strategy: Any,
    generator: Any,
    message: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    global_state = torch.get_rng_state().clone()

    with pytest.raises(IrisRequestError, match=message):
        run_request(
            parts,
            orchestrator,
            prompt=prompt,
            strategy=strategy,
            generator=generator,
        )

    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []
    assert torch.equal(torch.get_rng_state(), global_state)


SIDECAR_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "workspace_reader",
    "cognitive_modules",
    "proposal_head",
    "workspace_arbiter",
    "slot_allocator",
    "workspace_writer",
    "workspace_decay",
    "latent_bridge",
    "prefix_assembler",
)


@pytest.mark.parametrize("component_name", SIDECAR_NAMES)
def test_training_sidecar_is_rejected_before_the_backbone(component_name: str) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    component = cast(nn.Module, getattr(parts, component_name))
    component.train()

    with pytest.raises(IrisRequestError, match="evaluation mode"):
        run_request(parts, orchestrator)

    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []


def test_training_child_is_rejected_before_the_backbone() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.prefix_assembler.child.train()

    with pytest.raises(IrisRequestError, match="all of its children"):
        run_request(parts, orchestrator)

    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []


def test_callback_cannot_switch_a_later_child_to_training_mode() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    generator = torch.Generator(device=CPU).manual_seed(2026)
    original_state = generator.get_state().clone()

    def switch_child_to_training() -> None:
        parts.prefix_assembler.child.train()

    parts.control.mutation_stage = "goal"
    parts.control.mutation_action = switch_child_to_training

    with pytest.raises(IrisRequestError, match="request goal encoding") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, IrisRequestError)
    assert "all of its children" in str(captured.value.__cause__)
    assert parts.control.events == ["first_pass", "sampler", "goal"]
    assert parts.module_input_router.calls == []
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


@pytest.mark.parametrize(
    ("component_name", "mutation_target", "stage_name"),
    [
        ("layer_sampler", "input_ids", "sampler"),
        ("layer_sampler", "attention_mask", "sampler"),
        ("latent_bridge", "attention_mask", "bridge"),
        ("prefix_assembler", "attention_mask", "prefix"),
    ],
)
def test_prompt_metadata_persistent_mutation_is_caught_at_the_callback_boundary(
    component_name: str,
    mutation_target: str,
    stage_name: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    component = getattr(parts, component_name)
    if isinstance(component, SpyLayerSampler):
        component.mutation_target = mutation_target
    component.mutation_kind = "persistent"
    generator = torch.Generator(device=CPU).manual_seed(2027)
    original_state = generator.get_state().clone()
    expected_index = next(
        index
        for index, event in enumerate(HAPPY_EVENTS)
        if event.split(":", maxsplit=1)[0] == stage_name
    )

    with pytest.raises(IrisRequestError) as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, IrisRequestError)
    assert f"mutated exact first-pass {mutation_target}" in str(captured.value.__cause__)
    assert parts.control.events == HAPPY_EVENTS[: expected_index + 1]
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)
    first_pass = parts.backbone.first_pass_outputs[0]
    assert torch.equal(first_pass.input_ids, torch.tensor([[0, 2, 7]], dtype=torch.int64))
    assert torch.equal(
        first_pass.attention_mask,
        torch.tensor([[0, 1, 1]], dtype=torch.int64),
    )


@pytest.mark.parametrize("mutation_target", ["input_ids", "attention_mask"])
def test_prompt_metadata_mutate_then_restore_is_caught_by_version_counter(
    mutation_target: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.layer_sampler.mutation_target = mutation_target
    parts.layer_sampler.mutation_kind = "restore"
    generator = torch.Generator(device=CPU).manual_seed(2028)
    original_state = generator.get_state().clone()

    with pytest.raises(IrisRequestError, match="backbone layer sampling") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    observation = parts.layer_sampler.mutation_observations[0]
    assert observation.after_version > observation.before_version
    sampler_input = cast(BackboneFirstPassOutput, parts.layer_sampler.calls[0])
    expected = (
        torch.tensor([[0, 2, 7]], dtype=torch.int64)
        if mutation_target == "input_ids"
        else torch.tensor([[0, 1, 1]], dtype=torch.int64)
    )
    assert torch.equal(cast(torch.Tensor, getattr(sampler_input, mutation_target)), expected)
    assert isinstance(captured.value.__cause__, IrisRequestError)
    assert f"mutated exact first-pass {mutation_target}" in str(captured.value.__cause__)
    assert parts.control.events == ["first_pass", "sampler"]
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


@pytest.mark.parametrize("mutation_target", ["input_ids", "attention_mask"])
def test_prompt_metadata_equal_value_data_rebinding_is_caught_by_storage_identity(
    mutation_target: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.layer_sampler.mutation_target = mutation_target
    parts.layer_sampler.mutation_kind = "rebind"
    generator = torch.Generator(device=CPU).manual_seed(2029)
    original_state = generator.get_state().clone()

    with pytest.raises(IrisRequestError, match="backbone layer sampling") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    observation = parts.layer_sampler.mutation_observations[0]
    assert observation.after_version == observation.before_version
    assert observation.after_storage != observation.before_storage
    assert isinstance(captured.value.__cause__, IrisRequestError)
    assert f"mutated exact first-pass {mutation_target}" in str(captured.value.__cause__)
    assert parts.control.events == ["first_pass", "sampler"]
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


@pytest.mark.parametrize("config_name", ["cognitive_config", "workspace_config"])
def test_postconstruction_config_mutation_is_rejected_before_the_backbone(
    config_name: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    if config_name == "cognitive_config":
        parts.cognitive_config.state_dim = STATE_DIM + 1
    else:
        parts.workspace_config.cycles = CYCLES + 1
    generator = torch.Generator(device=CPU).manual_seed(2030)
    original_state = generator.get_state().clone()

    with pytest.raises(IrisRequestError, match="runtime boundary") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert "configuration changed after construction" in str(captured.value.__cause__)
    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []
    assert torch.equal(generator.get_state(), original_state)


def test_postconstruction_component_mutation_is_rejected_before_the_backbone() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.goal_encoder.state_dim = STATE_DIM + 1
    generator = torch.Generator(device=CPU).manual_seed(2031)
    original_state = generator.get_state().clone()

    with pytest.raises(IrisRequestError, match="runtime boundary") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert "goal_encoder.state_dim" in str(captured.value.__cause__)
    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []
    assert torch.equal(generator.get_state(), original_state)


def test_callback_component_wiring_mutation_is_rejected_before_the_next_callback() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    generator = torch.Generator(device=CPU).manual_seed(2032)
    original_state = generator.get_state().clone()

    def mutate_later_component() -> None:
        parts.goal_encoder.state_dim = STATE_DIM + 1

    parts.control.mutation_stage = "sampler"
    parts.control.mutation_action = mutate_later_component

    with pytest.raises(IrisRequestError, match="backbone layer sampling") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert "goal_encoder.state_dim" in str(captured.value.__cause__)
    assert parts.control.events == ["first_pass", "sampler"]
    assert parts.goal_encoder.calls == []
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


def test_callback_config_mutation_is_rejected_before_the_next_callback() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    generator = torch.Generator(device=CPU).manual_seed(2033)
    original_state = generator.get_state().clone()

    def mutate_config() -> None:
        parts.workspace_config.cycles = CYCLES + 1

    parts.control.mutation_stage = "goal"
    parts.control.mutation_action = mutate_config

    with pytest.raises(IrisRequestError, match="request goal encoding") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert "configuration changed after construction" in str(captured.value.__cause__)
    assert parts.control.events == ["first_pass", "sampler", "goal"]
    assert parts.module_input_router.calls == []
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


STAGE_MESSAGES = {
    "first_pass": "frozen first pass",
    "sampler": "backbone layer sampling",
    "goal": "request goal encoding",
    "router": "module input routing",
    "initial_state": "cognitive state initialization",
    "empty": "workspace initialization",
    "reader": "workspace read for cycle 1",
    "cognitive": "cognitive modules for cycle 1",
    "proposal": "proposal head for cycle 1",
    "arbiter": "workspace arbitration for cycle 1",
    "allocator": "slot allocation for cycle 1",
    "writer": "workspace write for cycle 1",
    "decay": "workspace decay before cycle 2",
    "embed": "exact prompt token embedding",
    "bridge": "workspace latent bridge",
    "prefix": "second-pass generation prefix",
}


@pytest.mark.parametrize("failure_stage", tuple(STAGE_MESSAGES))
def test_prefacade_stage_failure_chains_short_circuits_and_preserves_sampling_rng(
    failure_stage: str,
) -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.control.failure_stage = failure_stage
    generator = torch.Generator(device=CPU).manual_seed(1097)
    original_state = generator.get_state().clone()
    expected_index = next(
        index
        for index, event in enumerate(HAPPY_EVENTS)
        if event.split(":", maxsplit=1)[0] == failure_stage
    )

    with pytest.raises(
        IrisRequestError,
        match=STAGE_MESSAGES[failure_stage],
    ) as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert captured.value.__cause__ is parts.control.failure
    assert parts.control.events == HAPPY_EVENTS[: expected_index + 1]
    assert parts.iris_generator.calls == []
    assert torch.equal(generator.get_state(), original_state)


def test_facade_failure_is_typed_and_is_the_final_pipeline_event() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.control.failure_stage = "facade"

    with pytest.raises(IrisRequestError, match="Iris text generation failed") as captured:
        run_request(parts, orchestrator)

    assert captured.value.__cause__ is parts.control.failure
    assert parts.control.events == HAPPY_EVENTS


def test_sampling_generator_is_passed_through_and_committed_only_by_successful_facade() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.iris_generator.draws_before_return = 2
    generator = torch.Generator(device=CPU).manual_seed(2034)
    expected_generator = torch.Generator(device=CPU).manual_seed(2034)
    global_state = torch.get_rng_state().clone()
    for _ in range(2):
        torch.rand((), generator=expected_generator)

    output = run_request(
        parts,
        orchestrator,
        strategy="sampling",
        generator=generator,
    )

    assert output is parts.iris_generator.outputs[0]
    assert parts.iris_generator.calls[0].generator is generator
    assert torch.equal(generator.get_state(), expected_generator.get_state())
    assert torch.equal(torch.get_rng_state(), global_state)


def test_sampling_facade_rollback_is_preserved_when_orchestrator_wraps_failure() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.iris_generator.draws_before_return = 2
    facade_error = RuntimeError("transaction rolled back")
    parts.iris_generator.error_after_draw = facade_error
    generator = torch.Generator(device=CPU).manual_seed(2035)
    original_state = generator.get_state().clone()
    global_state = torch.get_rng_state().clone()

    with pytest.raises(IrisRequestError, match="Iris text generation failed") as captured:
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert captured.value.__cause__ is facade_error
    assert parts.iris_generator.calls[0].generator is generator
    assert torch.equal(generator.get_state(), original_state)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_recursive_modules_failure_is_typed_before_backbone_and_preserves_rng() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    modules_error = RuntimeError("recursive module walk failed")
    generator = torch.Generator(device=CPU).manual_seed(2036)
    original_state = generator.get_state().clone()

    with (
        patch.object(parts.goal_encoder, "modules", side_effect=modules_error),
        pytest.raises(IrisRequestError, match="runtime boundary") as captured,
    ):
        run_request(
            parts,
            orchestrator,
            strategy="sampling",
            generator=generator,
        )

    assert captured.value.__cause__ is modules_error
    assert parts.control.events == []
    assert parts.backbone.forward_prompts == []
    assert torch.equal(generator.get_state(), original_state)


def test_legacy_first_pass_output_short_circuits_before_sampling() -> None:
    parts = make_parts()
    orchestrator = build_orchestrator(parts)
    parts.backbone.return_legacy_output = True

    with pytest.raises(IrisRequestError, match="first-pass token provenance"):
        run_request(parts, orchestrator)

    assert parts.control.events == ["first_pass"]
    assert parts.layer_sampler.calls == []


@pytest.mark.parametrize(
    ("component_name", "attribute", "wrong_value"),
    [
        ("layer_sampler", "backbone_hidden_size", BACKBONE_HIDDEN_SIZE + 1),
        ("layer_sampler", "state_dim", STATE_DIM + 1),
        ("layer_sampler", "state_dim", float(STATE_DIM)),
        ("layer_sampler", "layer_fractions", (0.1, 0.3, 0.6, 1.0)),
        ("goal_encoder", "state_dim", STATE_DIM + 1),
        ("module_input_router", "num_modules", NUM_MODULES + 1),
        ("workspace_reader", "num_slots", NUM_SLOTS + 1),
        ("cognitive_modules", "state_dim", STATE_DIM + 1),
        ("proposal_head", "cycles", CYCLES + 1),
        ("workspace_arbiter", "winners_per_cycle", 1),
        ("workspace_arbiter", "confidence_weight", 9.0),
        ("workspace_arbiter", "confidence_weight", 1),
        ("slot_allocator", "empty_bonus", 9.0),
        ("workspace_writer", "num_slots", NUM_SLOTS + 1),
        ("workspace_decay", "decay_rate", 0.25),
        ("workspace_decay", "empty_threshold", 0.25),
        ("latent_bridge", "backbone_hidden_size", BACKBONE_HIDDEN_SIZE + 1),
        ("prefix_assembler", "hidden_size", BACKBONE_HIDDEN_SIZE + 1),
        ("prefix_assembler", "max_prompt_tokens", MAX_PROMPT_TOKENS + 1),
        ("prefix_assembler", "max_new_tokens", MAX_NEW_TOKENS + 1),
        ("prefix_assembler", "max_context_tokens", MAX_CONTEXT_TOKENS + 1),
        ("iris_generator", "max_new_tokens", MAX_NEW_TOKENS + 1),
        ("iris_generator", "max_new_tokens", float(MAX_NEW_TOKENS)),
    ],
)
def test_constructor_rejects_dimensional_and_config_wiring_mismatch(
    component_name: str,
    attribute: str,
    wrong_value: object,
) -> None:
    parts = make_parts()
    component = getattr(parts, component_name)
    setattr(component, attribute, wrong_value)

    with pytest.raises(ValueError, match=rf"{component_name}\.{attribute}"):
        build_orchestrator(parts)


def test_constructor_rejects_more_winners_than_cognitive_modules() -> None:
    parts = make_parts()
    parts.cognitive_config = cognitive_config(num_modules=1)

    with pytest.raises(ValueError, match=r"winners_per_cycle.*num_modules"):
        build_orchestrator(parts)


def test_constructor_requires_facade_to_own_the_exact_backbone() -> None:
    parts = make_parts()
    parts.iris_generator._backbone = SpyBackbone(parts.control)

    with pytest.raises(ValueError, match="exact request backbone adapter"):
        build_orchestrator(parts)


def test_constructor_wraps_throwing_component_attribute_getter() -> None:
    parts = make_parts()
    getter_error = RuntimeError("state_dim getter exploded")

    with (
        patch.object(
            SpyGoalEncoder,
            "state_dim",
            new_callable=PropertyMock,
            create=True,
        ) as getter,
        pytest.raises(ValueError, match=r"goal_encoder\.state_dim") as captured,
    ):
        getter.side_effect = getter_error
        build_orchestrator(parts)

    assert captured.value.__cause__ is getter_error


def test_constructor_wraps_throwing_generator_backbone_getter() -> None:
    parts = make_parts()
    getter_error = RuntimeError("backbone getter exploded")

    with (
        patch.object(
            SpyIrisGenerator,
            "backbone",
            new_callable=PropertyMock,
        ) as getter,
        pytest.raises(ValueError, match=r"iris_generator\.backbone") as captured,
    ):
        getter.side_effect = getter_error
        build_orchestrator(parts)

    assert captured.value.__cause__ is getter_error
