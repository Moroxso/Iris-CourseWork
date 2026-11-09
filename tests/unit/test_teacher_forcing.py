"""Core contracts for differentiable Iris teacher forcing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers.cache_utils import Cache

import iris.training as training
from iris.backbone import (
    BackboneAdapter,
    BackboneCausalOutput,
    BackboneFirstPassOutput,
    BackboneForwardOutput,
)
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
from iris.generation import SecondPassSequenceAssembler, SecondPassSequenceOutput
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training import (
    IrisTeacherForcedOutput,
    IrisTeacherForcingError,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
)
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
VOCABULARY_SIZE = 16
STATE_DIM = 4
NUM_MODULES = 3
NUM_SLOTS = 2
CYCLES = 3
WINNERS = 2
MAX_PROMPT_TOKENS = 5
MAX_TARGET_TOKENS = 3
MAX_CONTEXT_TOKENS = 10
LAYER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

CYCLIC_STAGES = frozenset(
    {"reader", "cognitive", "proposal", "arbiter", "allocator", "writer", "decay"}
)


@dataclass(slots=True)
class EventControl:
    """One deterministic event log and injected-failure controller."""

    events: list[str]
    counts: dict[str, int]
    failure_stage: str | None = None
    failure: RuntimeError | None = None

    @classmethod
    def empty(cls) -> EventControl:
        return cls(events=[], counts={})

    def record(self, stage: str) -> int:
        index = self.counts.get(stage, 0)
        self.counts[stage] = index + 1
        label = f"{stage}:{index}" if stage in CYCLIC_STAGES else stage
        if stage == "embed":
            label = f"embed:{index}"
        self.events.append(label)
        if self.failure_stage == stage:
            if self.failure is None:
                self.failure = RuntimeError(f"{stage} exploded")
            raise self.failure
        return index


def cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "teacher-test-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": NUM_MODULES,
            "layer_sampler": {"layer_fractions": LAYER_FRACTIONS},
        }
    )


def workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "teacher-test-workspace",
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


def workspace_state(
    marker: int,
    *,
    slots: torch.Tensor | None = None,
    occupied: bool = True,
) -> WorkspaceState:
    matrix_shape = (BATCH_SIZE, NUM_SLOTS)
    if slots is None:
        slots = torch.full(
            (*matrix_shape, STATE_DIM),
            float(marker),
            dtype=torch.float32,
        )
    return WorkspaceState(
        slots=slots,
        age=torch.full(matrix_shape, marker if occupied else 0, dtype=torch.int64),
        occupancy=torch.full(matrix_shape, occupied, dtype=torch.bool),
        source_module=torch.tensor([[0, 1]], dtype=torch.int64)
        if occupied
        else torch.full(matrix_shape, -1, dtype=torch.int64),
        confidence=torch.full(
            matrix_shape,
            0.75 if occupied else 0.0,
            dtype=torch.float32,
        ),
        ttl=torch.full(matrix_shape, 2.0 if occupied else 0.0, dtype=torch.float32),
        write_count=torch.full(
            matrix_shape,
            marker + 1 if occupied else 0,
            dtype=torch.int64,
        ),
    )


class SpyBackbone(BackboneAdapter):
    """Frozen boundary double with exact prompt and causal call provenance."""

    def __init__(self, control: EventControl) -> None:
        self.control = control
        self.forward_prompts: list[str] = []
        self.first_pass_outputs: list[BackboneFirstPassOutput] = []
        self.embed_inputs: list[torch.Tensor] = []
        self.embed_grad_modes: list[bool] = []
        self.causal_calls: list[dict[str, object]] = []
        self.causal_outputs: list[BackboneCausalOutput] = []
        self.mutate_embed_index: int | None = None

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
        return MAX_TARGET_TOKENS

    @property
    def device(self) -> torch.device:
        return CPU

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def trainable_parameter_count(self) -> int:
        return 0

    @property
    def has_parameter_gradients(self) -> bool:
        return False

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
        output = BackboneFirstPassOutput(
            logits=torch.ones(BATCH_SIZE, PROMPT_LENGTH, VOCABULARY_SIZE),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        self.first_pass_outputs.append(output)
        return output

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        call_index = self.control.record("embed")
        self.embed_grad_modes.append(torch.is_grad_enabled())
        self.embed_inputs.append(input_ids.detach().clone())
        if self.mutate_embed_index == call_index:
            input_ids.add_(1)
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return values.expand(-1, -1, BACKBONE_HIDDEN_SIZE).clone() / 10.0

    def forward_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        logits_to_keep: Literal[0, 1] = 0,
    ) -> BackboneCausalOutput:
        self.control.record("causal")
        self.causal_calls.append(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "labels": labels,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_enabled": torch.is_inference_mode_enabled(),
            }
        )
        logits = inputs_embeds.mean(dim=-1, keepdim=True).expand(
            -1,
            -1,
            VOCABULARY_SIZE,
        )
        if labels is None:
            loss = None
        else:
            shifted_labels = labels[:, 1:]
            supervised = shifted_labels != -100
            supervised_values = inputs_embeds[:, :-1].mean(dim=-1)[supervised]
            target_values = shifted_labels[supervised].to(dtype=inputs_embeds.dtype)
            loss = (supervised_values - target_values / 10.0).square().mean()
        output = BackboneCausalOutput(logits=logits, loss=loss, past_key_values=None)
        self.causal_outputs.append(output)
        return output


class SpyLayerSampler(LayerSampler):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.backbone_hidden_size = BACKBONE_HIDDEN_SIZE
        self.layer_fractions = LAYER_FRACTIONS
        self.inputs: list[BackboneForwardOutput] = []
        self.mutate_prompt_field: str | None = None
        self.eval()

    def forward(self, backbone_output: BackboneForwardOutput) -> LayerSamplerOutput:
        self.control.record("sampler")
        self.inputs.append(backbone_output)
        if self.mutate_prompt_field is not None:
            tensor = cast(torch.Tensor, getattr(backbone_output, self.mutate_prompt_field))
            tensor.add_(1)
        return LayerSamplerOutput(
            context_features=torch.arange(
                BATCH_SIZE * PROMPT_LENGTH * STATE_DIM,
                dtype=torch.float32,
            ).reshape(BATCH_SIZE, PROMPT_LENGTH, STATE_DIM),
            attention_mask=backbone_output.attention_mask.to(dtype=torch.bool),
            mixing_weights=torch.full((4,), 0.25),
            selected_layer_indices=(1, 2, 3, 4),
        )


class SpyGoalEncoder(GoalEncoder):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.eval()

    def forward(
        self,
        context_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> GoalEncoderOutput:
        self.control.record("goal")
        return GoalEncoderOutput(
            goal_state=torch.full((BATCH_SIZE, STATE_DIM), 0.5),
            attention_weights=torch.full((BATCH_SIZE, PROMPT_LENGTH), 1 / PROMPT_LENGTH),
        )


class SpyModuleInputRouter(ModuleInputRouter):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.eval()

    def forward(
        self,
        context_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ModuleInputRouterOutput:
        self.control.record("router")
        return ModuleInputRouterOutput(
            module_inputs=torch.full((BATCH_SIZE, NUM_MODULES, STATE_DIM), 0.25),
            attention_weights=torch.full(
                (BATCH_SIZE, NUM_MODULES, PROMPT_LENGTH),
                1 / PROMPT_LENGTH,
            ),
        )


class SpyWorkspaceReader(WorkspaceReader):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.inputs: list[WorkspaceState] = []
        self.eval()

    def forward(self, state: WorkspaceState) -> WorkspaceReaderOutput:
        index = self.control.record("reader")
        self.inputs.append(state)
        return WorkspaceReaderOutput(
            workspace_reads=torch.full(
                (BATCH_SIZE, NUM_MODULES, STATE_DIM),
                float(index % CYCLES + 1),
            ),
            attention_weights=torch.full(
                (BATCH_SIZE, NUM_MODULES, NUM_SLOTS),
                1 / NUM_SLOTS,
            ),
        )


class SpyCognitiveModules(CognitiveModule):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.initial_outputs: list[torch.Tensor] = []
        self.previous_states: list[torch.Tensor | None] = []
        self.eval()

    def initial_state(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        self.control.record("initial_state")
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
        self.previous_states.append(previous_states)
        return CognitiveModuleOutput(
            module_states=torch.full(
                (BATCH_SIZE, NUM_MODULES, STATE_DIM),
                float(index % CYCLES + 1),
            )
        )


class SpyProposalHead(ProposalHead):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.cycles = CYCLES
        self.eval()

    def forward(self, module_states: torch.Tensor) -> ModuleProposal:
        index = self.control.record("proposal")
        value = float(index % CYCLES + 1)
        return ModuleProposal(
            content=torch.full((BATCH_SIZE, NUM_MODULES, STATE_DIM), value),
            key=torch.full((BATCH_SIZE, NUM_MODULES, STATE_DIM), 0.5),
            bid=torch.full((BATCH_SIZE, NUM_MODULES), 0.5),
            confidence=torch.full((BATCH_SIZE, NUM_MODULES), 0.5),
            ttl=torch.full((BATCH_SIZE, NUM_MODULES), 2.0),
        )


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
        self.eval()

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        goal_state: torch.Tensor,
    ) -> WorkspaceArbiterOutput:
        self.control.record("arbiter")
        scores = torch.tensor([[3.0, 2.0, 1.0]])
        return WorkspaceArbiterOutput(
            scores=scores,
            novelty=torch.ones_like(scores),
            relevance=torch.ones_like(scores),
            soft_probabilities=torch.softmax(scores, dim=-1),
            selection_weights=torch.tensor([[1.0, 1.0, 0.0]]),
            winner_mask=torch.tensor([[True, True, False]]),
            winner_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        )


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
        self.eval()

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        arbitration: WorkspaceArbiterOutput,
    ) -> SlotAllocatorOutput:
        self.control.record("allocator")
        assignment = torch.tensor([[[True, False], [False, True], [False, False]]])
        matrix = torch.zeros(BATCH_SIZE, NUM_MODULES, NUM_SLOTS)
        return SlotAllocatorOutput(
            allocation_scores=matrix,
            retention_scores=torch.zeros(BATCH_SIZE, NUM_SLOTS),
            target_mask=torch.ones(BATCH_SIZE, NUM_SLOTS, dtype=torch.bool),
            assignment_mask=assignment,
            assignment_probabilities=assignment.to(dtype=torch.float32),
            assignment_weights=assignment.to(dtype=torch.float32),
            module_indices=torch.tensor([[0, 1]], dtype=torch.int64),
            slot_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        )


class SpyWorkspaceWriter(WorkspaceWriter):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.num_modules = NUM_MODULES
        self.num_slots = NUM_SLOTS
        self.winners_per_cycle = WINNERS
        self.scale = nn.Parameter(torch.tensor(1.0))
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
        marker = index % CYCLES + 1
        slots = self.scale * torch.full(
            (BATCH_SIZE, NUM_SLOTS, STATE_DIM),
            float(marker),
        )
        output = workspace_state(marker, slots=slots)
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
        self.eval()

    def forward(self, workspace: WorkspaceState) -> WorkspaceState:
        self.control.record("decay")
        self.inputs.append(workspace)
        return workspace


class SpyLatentBridge(LatentBridge):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.backbone_hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_slots = NUM_SLOTS
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.inputs: list[WorkspaceState] = []
        self.eval()

    def forward(
        self,
        workspace: WorkspaceState,
        prompt_attention_mask: torch.Tensor,
    ) -> LatentBridgeOutput:
        self.control.record("bridge")
        self.inputs.append(workspace)
        values = workspace.slots.mean(dim=-1, keepdim=True) * self.scale
        embeddings = values.expand(-1, -1, BACKBONE_HIDDEN_SIZE)
        logical_prompt_length = prompt_attention_mask.to(dtype=torch.int64).sum(dim=-1)
        return LatentBridgeOutput(
            workspace_embeddings=embeddings,
            workspace_attention_mask=torch.ones(BATCH_SIZE, NUM_SLOTS, dtype=torch.bool),
            workspace_position_ids=logical_prompt_length.unsqueeze(1)
            + torch.arange(NUM_SLOTS, dtype=torch.int64),
        )


@dataclass(frozen=True, slots=True)
class SequenceCall:
    prompt_embeddings: torch.Tensor
    prompt_attention_mask: torch.Tensor
    workspace: LatentBridgeOutput
    target_input_ids: torch.Tensor
    target_embeddings: torch.Tensor
    target_attention_mask: torch.Tensor


class SpySequenceAssembler(SecondPassSequenceAssembler):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_slots = NUM_SLOTS
        self.vocabulary_size = VOCABULARY_SIZE
        self.max_prompt_tokens = MAX_PROMPT_TOKENS
        self.max_target_tokens = MAX_TARGET_TOKENS
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.child = nn.Identity()
        self.calls: list[SequenceCall] = []
        self.outputs: list[SecondPassSequenceOutput] = []
        self.mutate_target_attention_mask = False
        self.eval()

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        workspace: LatentBridgeOutput,
        target_input_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> SecondPassSequenceOutput:
        self.control.record("sequence")
        self.calls.append(
            SequenceCall(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=prompt_attention_mask,
                workspace=workspace,
                target_input_ids=target_input_ids,
                target_embeddings=target_embeddings,
                target_attention_mask=target_attention_mask,
            )
        )
        if self.mutate_target_attention_mask:
            target_attention_mask.zero_()
        prompt_mask = prompt_attention_mask.to(dtype=torch.bool)
        target_mask = target_attention_mask.to(dtype=torch.bool)
        attention_mask = torch.cat(
            (prompt_mask, workspace.workspace_attention_mask, target_mask),
            dim=1,
        )
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        labels = torch.cat(
            (
                torch.full(
                    (BATCH_SIZE, PROMPT_LENGTH + NUM_SLOTS),
                    -100,
                    dtype=torch.int64,
                ),
                torch.where(
                    target_mask,
                    target_input_ids,
                    torch.full_like(target_input_ids, -100),
                ),
            ),
            dim=1,
        )
        output = SecondPassSequenceOutput(
            inputs_embeds=torch.cat(
                (prompt_embeddings, workspace.workspace_embeddings, target_embeddings),
                dim=1,
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        self.outputs.append(output)
        return output


@dataclass(slots=True)
class Harness:
    control: EventControl
    model: IrisTeacherForcingModel
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
    sequence_assembler: SpySequenceAssembler
    empty_outputs: list[WorkspaceState]

    def empty_workspace(self, **kwargs: object) -> WorkspaceState:
        self.control.record("empty")
        assert kwargs == {
            "batch_size": BATCH_SIZE,
            "num_slots": NUM_SLOTS,
            "state_dim": STATE_DIM,
            "dtype": torch.float32,
            "device": CPU,
        }
        output = workspace_state(0, occupied=False)
        self.empty_outputs.append(output)
        return output


def make_harness() -> Harness:
    control = EventControl.empty()
    cognitive = cognitive_config()
    workspace = workspace_config()
    backbone = SpyBackbone(control)
    layer_sampler = SpyLayerSampler(control)
    goal_encoder = SpyGoalEncoder(control)
    module_input_router = SpyModuleInputRouter(control)
    workspace_reader = SpyWorkspaceReader(control)
    cognitive_modules = SpyCognitiveModules(control)
    proposal_head = SpyProposalHead(control)
    workspace_arbiter = SpyWorkspaceArbiter(control, workspace)
    slot_allocator = SpySlotAllocator(control, workspace)
    workspace_writer = SpyWorkspaceWriter(control)
    workspace_decay = SpyWorkspaceDecay(control, workspace)
    latent_bridge = SpyLatentBridge(control)
    sequence_assembler = SpySequenceAssembler(control)
    model = IrisTeacherForcingModel(
        backbone=backbone,
        cognitive_config=cognitive,
        workspace_config=workspace,
        layer_sampler=layer_sampler,
        goal_encoder=goal_encoder,
        module_input_router=module_input_router,
        workspace_reader=workspace_reader,
        cognitive_modules=cognitive_modules,
        proposal_head=proposal_head,
        workspace_arbiter=workspace_arbiter,
        slot_allocator=slot_allocator,
        workspace_writer=workspace_writer,
        workspace_decay=workspace_decay,
        latent_bridge=latent_bridge,
        sequence_assembler=sequence_assembler,
    )
    return Harness(
        control=control,
        model=model,
        backbone=backbone,
        layer_sampler=layer_sampler,
        goal_encoder=goal_encoder,
        module_input_router=module_input_router,
        workspace_reader=workspace_reader,
        cognitive_modules=cognitive_modules,
        proposal_head=proposal_head,
        workspace_arbiter=workspace_arbiter,
        slot_allocator=slot_allocator,
        workspace_writer=workspace_writer,
        workspace_decay=workspace_decay,
        latent_bridge=latent_bridge,
        sequence_assembler=sequence_assembler,
        empty_outputs=[],
    )


def nominal_input() -> IrisTeacherForcingInput:
    return IrisTeacherForcingInput(
        prompt="  exact teacher prompt  ",
        target_input_ids=torch.tensor([[5, 6]], dtype=torch.int64),
        target_attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
    )


def run_training(
    harness: Harness,
    example: IrisTeacherForcingInput | None = None,
) -> IrisTeacherForcedOutput:
    with patch.object(
        WorkspaceState,
        "empty",
        side_effect=harness.empty_workspace,
    ):
        return harness.model.training_forward(example or nominal_input())


def run_evaluation(
    harness: Harness,
    example: IrisTeacherForcingInput | None = None,
) -> IrisTeacherForcedOutput:
    with patch.object(
        WorkspaceState,
        "empty",
        side_effect=harness.empty_workspace,
    ):
        return harness.model.evaluation_forward(example or nominal_input())


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
    "embed:0",
    "embed:1",
    "bridge",
    "sequence",
    "causal",
]


def test_public_dtos_are_frozen_exact_and_canonical_exports() -> None:
    example = nominal_input()
    trace = (workspace_state(1),)
    output = IrisTeacherForcedOutput(
        loss=torch.tensor(1.25),
        supervised_token_count=2,
        workspace_trace=trace,
    )

    assert tuple(field.name for field in fields(example)) == (
        "prompt",
        "target_input_ids",
        "target_attention_mask",
    )
    assert tuple(field.name for field in fields(output)) == (
        "loss",
        "supervised_token_count",
        "workspace_trace",
        "diagnostic_trace",
    )
    # ADR0079 adds opt-in observational metadata; legacy construction stays untraced.
    assert output.diagnostic_trace is None
    positional = IrisTeacherForcedOutput(output.loss, output.supervised_token_count, trace)
    assert positional.diagnostic_trace is None
    assert positional.workspace_trace is trace
    with pytest.raises(FrozenInstanceError):
        example.prompt = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        output.supervised_token_count = 3  # type: ignore[misc]

    assert training.IrisTeacherForcingInput is IrisTeacherForcingInput
    assert training.IrisTeacherForcedOutput is IrisTeacherForcedOutput
    assert training.IrisTeacherForcingError is IrisTeacherForcingError
    assert training.IrisTeacherForcingModel is IrisTeacherForcingModel


def test_teacher_exposes_exact_frozen_backbone_for_preparation() -> None:
    harness = make_harness()

    assert harness.model.backbone is harness.backbone
    assert issubclass(IrisTeacherForcingError, RuntimeError)


def test_forward_delegates_exactly_once_to_training_forward() -> None:
    harness = make_harness()
    example = nominal_input()
    expected = IrisTeacherForcedOutput(
        loss=torch.tensor(1.0, requires_grad=True),
        supervised_token_count=2,
        workspace_trace=(workspace_state(1),),
    )

    with patch.object(
        IrisTeacherForcingModel,
        "training_forward",
        autospec=True,
        return_value=expected,
    ) as training_forward:
        actual = harness.model(example)

    assert actual is expected
    training_forward.assert_called_once_with(harness.model, example)


def test_happy_path_has_exact_order_counts_values_and_output_boundary() -> None:
    harness = make_harness()
    example = nominal_input()

    output = run_training(harness, example)

    assert type(output) is IrisTeacherForcedOutput
    assert harness.control.events == HAPPY_EVENTS
    assert harness.backbone.forward_prompts == [example.prompt]
    assert len(harness.backbone.embed_inputs) == 2
    assert torch.equal(harness.backbone.embed_inputs[0], torch.tensor([[0, 2, 7]]))
    assert torch.equal(harness.backbone.embed_inputs[1], example.target_input_ids)
    assert len(harness.sequence_assembler.calls) == 1
    sequence_call = harness.sequence_assembler.calls[0]
    assert torch.equal(sequence_call.target_input_ids, example.target_input_ids)
    assert torch.equal(sequence_call.target_attention_mask, example.target_attention_mask)
    assert len(harness.backbone.causal_calls) == 1
    causal_call = harness.backbone.causal_calls[0]
    sequence = harness.sequence_assembler.outputs[0]
    assert causal_call["inputs_embeds"] is sequence.inputs_embeds
    assert causal_call["attention_mask"] is sequence.attention_mask
    assert causal_call["position_ids"] is sequence.position_ids
    assert causal_call["labels"] is sequence.labels
    assert causal_call["past_key_values"] is None
    assert causal_call["use_cache"] is False
    assert causal_call["logits_to_keep"] == 0
    assert sequence.labels.tolist() == [[-100, -100, -100, -100, -100, 5, 6]]
    assert sequence.position_ids.tolist() == [[0, 0, 1, 2, 3, 4, 5]]
    assert output.loss is harness.backbone.causal_outputs[0].loss
    assert output.loss.ndim == 0
    assert output.loss.dtype == torch.float32
    assert output.loss.device == CPU
    assert bool(torch.isfinite(output.loss))
    assert output.loss.requires_grad
    assert output.loss.grad_fn is not None
    assert type(output.supervised_token_count) is int
    assert output.supervised_token_count == 2
    assert len(output.workspace_trace) == CYCLES
    for output_state, writer_state in zip(
        output.workspace_trace,
        harness.workspace_writer.outputs,
        strict=True,
    ):
        assert output_state is writer_state
    assert harness.workspace_decay.inputs == harness.workspace_writer.outputs[: CYCLES - 1]
    assert harness.latent_bridge.inputs == [harness.workspace_writer.outputs[-1]]
    assert (
        harness.cognitive_modules.previous_states[0]
        is (harness.cognitive_modules.initial_outputs[0])
    )

    torch.autograd.backward(output.loss)
    assert harness.workspace_writer.scale.grad is not None
    assert bool(harness.workspace_writer.scale.grad != 0)
    assert harness.latent_bridge.scale.grad is not None
    assert bool(harness.latent_bridge.scale.grad != 0)


class StringSubclass(str):
    """Adversarial prompt subclass for exact-type validation."""


@pytest.mark.parametrize(
    "example",
    [
        cast(Any, object()),
        cast(Any, "not a teacher input"),
    ],
)
def test_rejects_wrong_input_dto_before_any_stage(example: Any) -> None:
    harness = make_harness()

    with pytest.raises(IrisTeacherForcingError):
        harness.model.training_forward(example)

    assert harness.control.events == []


@pytest.mark.parametrize(
    "prompt",
    ["", "   ", cast(Any, 7), StringSubclass("subclass"), "bad-surrogate-\ud800"],
)
def test_rejects_invalid_or_non_utf8_prompt_before_any_stage(prompt: Any) -> None:
    harness = make_harness()
    example = replace(nominal_input(), prompt=prompt)

    with pytest.raises(IrisTeacherForcingError):
        harness.model.training_forward(example)

    assert harness.control.events == []


def malformed_targets() -> list[tuple[torch.Tensor, torch.Tensor]]:
    nominal_ids = torch.tensor([[5, 6]], dtype=torch.int64)
    nominal_mask = torch.tensor([[1, 1]], dtype=torch.int64)
    return [
        (torch.tensor([5, 6], dtype=torch.int64), nominal_mask),
        (torch.tensor([[5, 6], [7, 8]], dtype=torch.int64), torch.ones(2, 2, dtype=torch.int64)),
        (torch.tensor([[5, 6]], dtype=torch.int32), nominal_mask),
        (torch.empty((1, 2), dtype=torch.int64, device="meta"), nominal_mask),
        (torch.tensor([[-1, 6]], dtype=torch.int64), nominal_mask),
        (torch.tensor([[VOCABULARY_SIZE, 6]], dtype=torch.int64), nominal_mask),
        (torch.empty((1, 0), dtype=torch.int64), torch.empty((1, 0), dtype=torch.int64)),
        (torch.tensor([[1, 2, 3, 4]], dtype=torch.int64), torch.ones(1, 4, dtype=torch.int64)),
        (nominal_ids, torch.tensor([1, 1], dtype=torch.int64)),
        (nominal_ids, torch.ones(1, 2, dtype=torch.float32)),
        (nominal_ids, torch.empty((1, 2), dtype=torch.int64, device="meta")),
        (nominal_ids, torch.tensor([[1, 2]], dtype=torch.int64)),
        (nominal_ids, torch.tensor([[0, 0]], dtype=torch.int64)),
        (nominal_ids, torch.tensor([[0, 1]], dtype=torch.int64)),
        (
            torch.tensor([[5, 6, 7]], dtype=torch.int64),
            torch.tensor([[1, 0, 1]], dtype=torch.int64),
        ),
    ]


@pytest.mark.parametrize(("target_input_ids", "target_attention_mask"), malformed_targets())
def test_rejects_malformed_target_before_any_stage(
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
) -> None:
    harness = make_harness()
    example = replace(
        nominal_input(),
        target_input_ids=target_input_ids,
        target_attention_mask=target_attention_mask,
    )

    with pytest.raises(IrisTeacherForcingError):
        harness.model.training_forward(example)

    assert harness.control.events == []


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_training_rejects_disabled_autograd_before_any_stage(
    context: Callable[[], Any],
) -> None:
    harness = make_harness()

    with context(), pytest.raises(IrisTeacherForcingError):
        harness.model.training_forward(nominal_input())

    assert harness.control.events == []


@pytest.mark.parametrize("mixed_child", ["goal_encoder", "nested_sequence_child"])
def test_training_rejects_any_eval_child_before_any_stage(mixed_child: str) -> None:
    harness = make_harness()
    if mixed_child == "goal_encoder":
        harness.goal_encoder.eval()
    else:
        harness.sequence_assembler.child.eval()

    with pytest.raises(IrisTeacherForcingError):
        harness.model.training_forward(nominal_input())

    assert harness.model.training
    assert harness.control.events == []


def test_evaluation_rejects_training_mode_before_any_stage() -> None:
    harness = make_harness()

    with pytest.raises(IrisTeacherForcingError):
        harness.model.evaluation_forward(nominal_input())

    assert harness.control.events == []


def test_evaluation_rejects_inference_mode_before_any_stage() -> None:
    harness = make_harness()
    harness.model.eval()

    with (
        torch.inference_mode(),
        pytest.raises(IrisTeacherForcingError, match=r"cannot run under torch.inference_mode"),
    ):
        harness.model.evaluation_forward(nominal_input())

    assert harness.control.events == []


def test_evaluation_is_no_grad_and_preserves_mode_rng_parameters_and_existing_grads() -> None:
    harness = make_harness()
    harness.model.eval()
    parameters = tuple(harness.model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, float(index + 7))
    gradient_objects = tuple(parameter.grad for parameter in parameters)
    gradient_snapshots = tuple(
        cast(torch.Tensor, parameter.grad).clone() for parameter in parameters
    )
    state_snapshot = {
        name: tensor.detach().clone() for name, tensor in harness.model.state_dict().items()
    }
    mode_snapshot = tuple(module.training for module in harness.model.modules())
    rng_snapshot = torch.get_rng_state().clone()

    output = run_evaluation(harness)

    assert harness.control.events == HAPPY_EVENTS
    assert output.loss.requires_grad is False
    assert output.loss.grad_fn is None
    assert not output.loss.is_inference()
    assert harness.backbone.causal_calls[0]["grad_enabled"] is False
    assert harness.backbone.causal_calls[0]["inference_enabled"] is False
    assert tuple(module.training for module in harness.model.modules()) == mode_snapshot
    assert torch.equal(torch.get_rng_state(), rng_snapshot)
    for parameter, original_object, original_value in zip(
        parameters,
        gradient_objects,
        gradient_snapshots,
        strict=True,
    ):
        assert parameter.grad is original_object
        torch.testing.assert_close(cast(torch.Tensor, parameter.grad), original_value)
    assert state_snapshot.keys() == harness.model.state_dict().keys()
    for name, tensor in harness.model.state_dict().items():
        torch.testing.assert_close(tensor, state_snapshot[name])


@pytest.mark.parametrize("failure_stage", ["goal", "writer", "causal"])
def test_stage_failures_are_wrapped_with_exact_cause(failure_stage: str) -> None:
    harness = make_harness()
    failure = RuntimeError(f"{failure_stage} exploded")
    harness.control.failure_stage = failure_stage
    harness.control.failure = failure

    with pytest.raises(IrisTeacherForcingError) as captured:
        run_training(harness)

    assert captured.value.__cause__ is failure


def test_repeated_calls_start_with_fresh_module_and_workspace_state() -> None:
    harness = make_harness()

    first = run_training(harness)
    second = run_training(harness)

    assert len(harness.cognitive_modules.initial_outputs) == 2
    assert len(harness.empty_outputs) == 2
    first_module_state, second_module_state = harness.cognitive_modules.initial_outputs
    first_workspace, second_workspace = harness.empty_outputs
    assert first_module_state.untyped_storage().data_ptr() != (
        second_module_state.untyped_storage().data_ptr()
    )
    assert first_workspace.slots.untyped_storage().data_ptr() != (
        second_workspace.slots.untyped_storage().data_ptr()
    )
    assert all(
        first_state is not second_state
        for first_state, second_state in zip(
            first.workspace_trace,
            second.workspace_trace,
            strict=True,
        )
    )
    assert harness.control.counts["first_pass"] == 2
    assert harness.control.counts["writer"] == 2 * CYCLES
    assert harness.control.counts["decay"] == 2 * (CYCLES - 1)


def test_different_targets_cannot_change_prompt_cognition_or_workspace_trace() -> None:
    harness = make_harness()
    first_example = nominal_input()
    second_example = replace(
        first_example,
        target_input_ids=torch.tensor([[8, 9]], dtype=torch.int64),
    )

    first = run_training(harness, first_example)
    second = run_training(harness, second_example)

    assert not torch.equal(first.loss, second.loss)
    assert first.supervised_token_count == second.supervised_token_count == 2
    for first_state, second_state in zip(
        first.workspace_trace,
        second.workspace_trace,
        strict=True,
    ):
        for field in fields(WorkspaceState):
            torch.testing.assert_close(
                cast(torch.Tensor, getattr(first_state, field.name)),
                cast(torch.Tensor, getattr(second_state, field.name)),
            )
    assert torch.equal(
        cast(BackboneFirstPassOutput, harness.layer_sampler.inputs[0]).input_ids,
        cast(BackboneFirstPassOutput, harness.layer_sampler.inputs[1]).input_ids,
    )


def cause_messages(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    return "\n".join(messages)


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (
            lambda harness: setattr(harness.layer_sampler, "mutate_prompt_field", "input_ids"),
            "mutated exact prompt input_ids",
        ),
        (
            lambda harness: setattr(
                harness.layer_sampler,
                "mutate_prompt_field",
                "attention_mask",
            ),
            "mutated exact prompt attention_mask",
        ),
        (
            lambda harness: setattr(harness.backbone, "mutate_embed_index", 1),
            "mutated exact target input_ids",
        ),
        (
            lambda harness: setattr(
                harness.sequence_assembler,
                "mutate_target_attention_mask",
                True,
            ),
            "mutated exact target attention_mask",
        ),
    ],
)
def test_callback_mutation_of_owned_provenance_is_detected_without_touching_caller(
    configure: Callable[[Harness], None],
    expected: str,
) -> None:
    harness = make_harness()
    example = nominal_input()
    target_ids_before = example.target_input_ids.clone()
    target_mask_before = example.target_attention_mask.clone()
    configure(harness)

    with pytest.raises(IrisTeacherForcingError) as captured:
        run_training(harness, example)

    assert expected in cause_messages(captured.value)
    torch.testing.assert_close(example.target_input_ids, target_ids_before)
    torch.testing.assert_close(example.target_attention_mask, target_mask_before)
    first_pass = harness.backbone.first_pass_outputs[0]
    torch.testing.assert_close(first_pass.input_ids, torch.tensor([[0, 2, 7]]))
    torch.testing.assert_close(first_pass.attention_mask, torch.tensor([[0, 1, 1]]))
