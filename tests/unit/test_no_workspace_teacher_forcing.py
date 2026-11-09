from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from torch import nn

import iris.training as training
from iris.backbone import BackboneCausalOutput
from iris.cognitive import (
    NO_WORKSPACE_COGNITIVE_CYCLES_V1,
    CognitiveModuleOutput,
    ModuleLatentBridge,
    ModuleLatentBridgeOutput,
)
from iris.generation import ModuleSecondPassSequenceAssembler, SecondPassSequenceOutput
from iris.schemas import BackboneConfig
from iris.training import (
    IrisNoWorkspaceTeacherCheckpointProvenanceV1,
    IrisNoWorkspaceTeacherForcingError,
    IrisNoWorkspaceTeacherForcingModel,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
)
from tests.unit.test_teacher_forcing import (
    BACKBONE_HIDDEN_SIZE,
    BATCH_SIZE,
    MAX_CONTEXT_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TARGET_TOKENS,
    NUM_MODULES,
    PROMPT_LENGTH,
    STATE_DIM,
    VOCABULARY_SIZE,
    EventControl,
    SpyBackbone,
    SpyCognitiveModules,
    SpyGoalEncoder,
    SpyLayerSampler,
    SpyModuleInputRouter,
    cognitive_config,
    nominal_input,
)

B_MAX_CONTEXT_TOKENS = MAX_CONTEXT_TOKENS + 1


class ProvenanceBackbone(SpyBackbone):
    @property
    def max_position_embeddings(self) -> int:
        return B_MAX_CONTEXT_TOKENS

    @property
    def configuration_json(self) -> str:
        return BackboneConfig.model_validate(
            {
                "name": "arm-b-teacher-test",
                "repository": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
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
                "limits": {
                    "max_prompt_tokens": MAX_PROMPT_TOKENS,
                    "max_new_tokens": MAX_TARGET_TOKENS,
                },
            }
        ).model_dump_json()


class NoWorkspaceSpyCognitiveModules(SpyCognitiveModules):
    def __init__(self, control: EventControl) -> None:
        super().__init__(control)
        self.memory_reads: list[torch.Tensor] = []

    def forward(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        self.memory_reads.append(workspace_reads)
        return super().forward(
            module_inputs,
            workspace_reads,
            goal_state,
            previous_states=previous_states,
        )


class SpyModuleLatentBridge(ModuleLatentBridge):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.state_dim = STATE_DIM
        self.backbone_hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_modules = NUM_MODULES
        self.scale = nn.Parameter(torch.tensor([0.5]))
        self.inputs: list[torch.Tensor] = []
        self.train()

    def forward(
        self,
        module_states: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> ModuleLatentBridgeOutput:
        self.control.record("bridge")
        self.inputs.append(module_states)
        values = module_states.mean(dim=-1, keepdim=True) * self.scale
        embeddings = values.expand(-1, -1, BACKBONE_HIDDEN_SIZE)
        prompt_lengths = prompt_attention_mask.to(dtype=torch.int64).sum(dim=-1)
        return ModuleLatentBridgeOutput(
            module_embeddings=embeddings,
            module_attention_mask=torch.ones(
                BATCH_SIZE,
                NUM_MODULES,
                dtype=torch.bool,
            ),
            module_position_ids=prompt_lengths.unsqueeze(1)
            + torch.arange(NUM_MODULES, dtype=torch.int64),
        )


@dataclass(frozen=True, slots=True)
class ModuleSequenceCall:
    modules: ModuleLatentBridgeOutput
    target_input_ids: torch.Tensor
    target_attention_mask: torch.Tensor


class SpyModuleSequenceAssembler(ModuleSecondPassSequenceAssembler):
    def __init__(self, control: EventControl) -> None:
        nn.Module.__init__(self)
        self.control = control
        self.hidden_size = BACKBONE_HIDDEN_SIZE
        self.num_modules = NUM_MODULES
        self.vocabulary_size = VOCABULARY_SIZE
        self.max_prompt_tokens = MAX_PROMPT_TOKENS
        self.max_target_tokens = MAX_TARGET_TOKENS
        self.max_context_tokens = B_MAX_CONTEXT_TOKENS
        self.calls: list[ModuleSequenceCall] = []
        self.train()

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        modules: ModuleLatentBridgeOutput,
        target_input_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> SecondPassSequenceOutput:
        self.control.record("sequence")
        self.calls.append(
            ModuleSequenceCall(
                modules=modules,
                target_input_ids=target_input_ids,
                target_attention_mask=target_attention_mask,
            )
        )
        prompt_mask = prompt_attention_mask.to(dtype=torch.bool)
        target_mask = target_attention_mask.to(dtype=torch.bool)
        attention_mask = torch.cat(
            (prompt_mask, modules.module_attention_mask, target_mask),
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
                    (BATCH_SIZE, PROMPT_LENGTH + NUM_MODULES),
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
        return SecondPassSequenceOutput(
            inputs_embeds=torch.cat(
                (prompt_embeddings, modules.module_embeddings, target_embeddings),
                dim=1,
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )


@dataclass(slots=True)
class Harness:
    control: EventControl
    model: IrisNoWorkspaceTeacherForcingModel
    backbone: ProvenanceBackbone
    cognitive_modules: NoWorkspaceSpyCognitiveModules
    module_latent_bridge: SpyModuleLatentBridge
    sequence_assembler: SpyModuleSequenceAssembler


def make_harness() -> Harness:
    control = EventControl.empty()
    backbone = ProvenanceBackbone(control)
    cognitive_modules = NoWorkspaceSpyCognitiveModules(control)
    bridge = SpyModuleLatentBridge(control)
    sequence = SpyModuleSequenceAssembler(control)
    model = IrisNoWorkspaceTeacherForcingModel(
        backbone=backbone,
        cognitive_config=cognitive_config(),
        layer_sampler=SpyLayerSampler(control),
        goal_encoder=SpyGoalEncoder(control),
        module_input_router=SpyModuleInputRouter(control),
        cognitive_modules=cognitive_modules,
        module_latent_bridge=bridge,
        sequence_assembler=sequence,
    )
    return Harness(
        control=control,
        model=model,
        backbone=backbone,
        cognitive_modules=cognitive_modules,
        module_latent_bridge=bridge,
        sequence_assembler=sequence,
    )


def test_public_surface_and_exact_component_ownership() -> None:
    harness = make_harness()
    assert training.IrisNoWorkspaceTeacherForcingModel is IrisNoWorkspaceTeacherForcingModel
    assert training.IrisNoWorkspaceTeacherForcingError is IrisNoWorkspaceTeacherForcingError
    assert (
        training.IrisNoWorkspaceTeacherCheckpointProvenanceV1
        is IrisNoWorkspaceTeacherCheckpointProvenanceV1
    )
    assert tuple(harness.model._modules) == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "cognitive_modules",
        "module_latent_bridge",
        "sequence_assembler",
    )
    forbidden = {
        "workspace_reader",
        "proposal_head",
        "workspace_arbiter",
        "slot_allocator",
        "workspace_writer",
        "workspace_decay",
        "latent_bridge",
    }
    assert forbidden.isdisjoint(harness.model._modules)
    assert "_backbone" not in harness.model._modules
    assert harness.model.cycles == NO_WORKSPACE_COGNITIVE_CYCLES_V1 == 3


def test_training_forward_runs_three_zero_read_cycles_and_empty_trace() -> None:
    harness = make_harness()
    output = harness.model.training_forward(nominal_input())
    assert type(output) is IrisTeacherForcedOutput
    assert output.supervised_token_count == 2
    assert output.workspace_trace == ()
    assert output.loss.requires_grad
    assert output.loss.grad_fn is not None
    assert harness.control.events == [
        "first_pass",
        "sampler",
        "goal",
        "router",
        "initial_state",
        "cognitive:0",
        "cognitive:1",
        "cognitive:2",
        "embed:0",
        "embed:1",
        "bridge",
        "sequence",
        "causal",
    ]
    assert len(harness.cognitive_modules.memory_reads) == 3
    first_read = harness.cognitive_modules.memory_reads[0]
    assert tuple(first_read.shape) == (1, NUM_MODULES, STATE_DIM)
    assert torch.count_nonzero(first_read).item() == 0
    assert all(read is first_read for read in harness.cognitive_modules.memory_reads)
    assert len(harness.cognitive_modules.previous_states) == 3
    assert torch.equal(
        cast(torch.Tensor, harness.cognitive_modules.previous_states[0]),
        torch.zeros(1, NUM_MODULES, STATE_DIM),
    )
    assert torch.equal(
        cast(torch.Tensor, harness.cognitive_modules.previous_states[1]),
        torch.ones(1, NUM_MODULES, STATE_DIM),
    )
    assert torch.equal(
        cast(torch.Tensor, harness.cognitive_modules.previous_states[2]),
        torch.full((1, NUM_MODULES, STATE_DIM), 2.0),
    )
    assert harness.sequence_assembler.calls[0].modules.module_embeddings.shape[1] == NUM_MODULES
    assert harness.backbone.causal_calls[0]["past_key_values"] is None
    assert harness.backbone.causal_calls[0]["use_cache"] is False
    assert harness.backbone.causal_calls[0]["logits_to_keep"] == 0


def test_training_loss_backpropagates_only_into_sidecar() -> None:
    harness = make_harness()
    output = harness.model.training_forward(nominal_input())
    torch.autograd.backward((output.loss,))
    assert harness.module_latent_bridge.scale.grad is not None
    assert bool(torch.isfinite(harness.module_latent_bridge.scale.grad))
    assert harness.backbone.trainable_parameter_count == 0
    assert not harness.backbone.has_parameter_gradients


def test_evaluation_requires_eval_mode_and_returns_detached_empty_trace() -> None:
    harness = make_harness()
    with pytest.raises(IrisNoWorkspaceTeacherForcingError, match="evaluation"):
        harness.model.evaluation_forward(nominal_input())
    harness.model.eval()
    output = harness.model.evaluation_forward(nominal_input())
    assert output.workspace_trace == ()
    assert not output.loss.requires_grad
    assert output.loss.grad_fn is None
    assert harness.backbone.causal_calls[-1]["grad_enabled"] is False


def test_checkpoint_provenance_is_arm_labelled_and_tracks_current_state() -> None:
    harness = make_harness()
    first = harness.model.checkpoint_provenance
    assert type(first) is IrisNoWorkspaceTeacherCheckpointProvenanceV1
    assert first.arm_id == "B"
    assert first.teacher_graph == "independent_recurrent_modules_zero_memory_read_teacher_v1"
    assert first.cognitive_cycles == 3
    assert first.latent_token_count == NUM_MODULES
    assert first.component_names == tuple(harness.model._modules)
    assert first.max_prompt_tokens == MAX_PROMPT_TOKENS
    assert first.max_target_tokens == MAX_TARGET_TOKENS
    with torch.no_grad():
        harness.module_latent_bridge.scale.add_(1.0)
    second = harness.model.checkpoint_provenance
    assert second.sidecar_state_sha256 != first.sidecar_state_sha256
    assert second.component_names == first.component_names


@pytest.mark.parametrize(
    "context",
    [
        torch.no_grad,
        torch.inference_mode,
    ],
)
def test_training_rejects_disabled_autograd_or_inference(
    context: Any,
) -> None:
    harness = make_harness()
    with context(), pytest.raises(IrisNoWorkspaceTeacherForcingError):
        harness.model.training_forward(nominal_input())


def test_rejects_invalid_example_without_running_backbone() -> None:
    harness = make_harness()
    invalid = IrisTeacherForcingInput(
        prompt=" ",
        target_input_ids=torch.tensor([[1]], dtype=torch.int64),
        target_attention_mask=torch.ones(1, 1, dtype=torch.int64),
    )
    with pytest.raises(IrisNoWorkspaceTeacherForcingError):
        harness.model.training_forward(invalid)
    assert harness.control.events == []


def test_zero_read_mutation_is_detected() -> None:
    harness = make_harness()
    original = harness.cognitive_modules.forward

    def mutate(
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        output = original(
            module_inputs,
            workspace_reads,
            goal_state,
            previous_states=previous_states,
        )
        workspace_reads.fill_(1.0)
        return output

    harness.cognitive_modules.forward = mutate  # type: ignore[method-assign]
    with pytest.raises(IrisNoWorkspaceTeacherForcingError, match="zero memory-read"):
        harness.model.training_forward(nominal_input())


def test_rejects_invalid_causal_output() -> None:
    harness = make_harness()

    def invalid(*args: object, **kwargs: object) -> BackboneCausalOutput:
        del args, kwargs
        return BackboneCausalOutput(
            logits=torch.zeros(1, 1, VOCABULARY_SIZE),
            loss=torch.tensor(1.0),
            past_key_values=None,
        )

    harness.backbone.forward_embeddings = invalid  # type: ignore[method-assign]
    with pytest.raises(IrisNoWorkspaceTeacherForcingError, match="causal logits"):
        harness.model.training_forward(nominal_input())


def test_exact_type_checks_reject_workspace_or_model_substitution() -> None:
    harness = make_harness()

    class InputSubclass(IrisTeacherForcingInput):
        pass

    with pytest.raises(IrisNoWorkspaceTeacherForcingError, match="exactly"):
        harness.model.training_forward(
            InputSubclass(
                prompt="x",
                target_input_ids=torch.tensor([[1]], dtype=torch.int64),
                target_attention_mask=torch.ones(1, 1, dtype=torch.int64),
            )
        )
    with pytest.raises(TypeError, match="exactly CognitiveConfig"):
        IrisNoWorkspaceTeacherForcingModel(
            backbone=harness.backbone,
            cognitive_config=cast(Any, object()),
            layer_sampler=harness.model.layer_sampler,
            goal_encoder=harness.model.goal_encoder,
            module_input_router=harness.model.module_input_router,
            cognitive_modules=harness.model.cognitive_modules,
            module_latent_bridge=harness.model.module_latent_bridge,
            sequence_assembler=harness.model.sequence_assembler,
        )
