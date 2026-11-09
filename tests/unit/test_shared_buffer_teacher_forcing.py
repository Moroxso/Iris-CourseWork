"""Exact teacher-forcing contract for the append-only arm-C graph."""

from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from torch import nn

import iris.training as training
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ModuleProposal,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler
from iris.schemas import BackboneConfig
from iris.shared_buffer import (
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferState,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.training import (
    IrisSharedBufferTeacherCheckpointProvenanceV1,
    IrisSharedBufferTeacherForcingError,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcedOutput,
)
from iris.workspace import WorkspaceReaderOutput
from tests.unit.test_teacher_forcing import (
    BACKBONE_HIDDEN_SIZE,
    MAX_PROMPT_TOKENS,
    MAX_TARGET_TOKENS,
    NUM_MODULES,
    PROMPT_LENGTH,
    STATE_DIM,
    VOCABULARY_SIZE,
    EventControl,
    SpyBackbone,
    cognitive_config,
    nominal_input,
)

C_MAX_CONTEXT_TOKENS = MAX_PROMPT_TOKENS + NUM_MODULES + MAX_TARGET_TOKENS


class ArmCBackbone(SpyBackbone):
    """The shared-buffer prefix has one latent token per module."""

    @property
    def max_position_embeddings(self) -> int:
        return C_MAX_CONTEXT_TOKENS

    @property
    def configuration_json(self) -> str:
        return BackboneConfig.model_validate(
            {
                "name": "arm-c-teacher-test",
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


class RecordingSharedBufferReader(SharedBufferReader):
    def __init__(self) -> None:
        super().__init__(state_dim=STATE_DIM, num_modules=NUM_MODULES, cycles=3)
        self.entry_counts: list[int] = []
        self.outputs: list[WorkspaceReaderOutput] = []

    def forward(self, state: SharedBufferState) -> WorkspaceReaderOutput:
        self.entry_counts.append(state.entries.shape[1])
        output = super().forward(state)
        self.outputs.append(output)
        return output


class RecordingSharedBufferWriter(SharedBufferWriter):
    def __init__(self) -> None:
        super().__init__(state_dim=STATE_DIM, num_modules=NUM_MODULES, cycles=3)
        self.input_counts: list[int] = []
        self.proposals: list[ModuleProposal] = []
        self.outputs: list[SharedBufferState] = []

    def forward(
        self,
        proposal: ModuleProposal,
        state: SharedBufferState,
        goal_state: torch.Tensor,
        *,
        cycle_number: int,
    ) -> SharedBufferState:
        self.input_counts.append(state.entries.shape[1])
        if torch.is_grad_enabled() and proposal.content.requires_grad:
            proposal.content.retain_grad()
        self.proposals.append(proposal)
        output = super().forward(
            proposal,
            state,
            goal_state,
            cycle_number=cycle_number,
        )
        self.outputs.append(output)
        return output


@dataclass(slots=True)
class Harness:
    model: IrisSharedBufferTeacherForcingModel
    backbone: ArmCBackbone
    reader: RecordingSharedBufferReader
    writer: RecordingSharedBufferWriter


def make_harness() -> Harness:
    torch.manual_seed(7401)
    control = EventControl.empty()
    backbone = ArmCBackbone(control)
    cognitive = cognitive_config()
    reader = RecordingSharedBufferReader()
    writer = RecordingSharedBufferWriter()
    model = IrisSharedBufferTeacherForcingModel(
        backbone=backbone,
        cognitive_config=cognitive,
        policy=build_shared_buffer_policy_v1(),
        layer_sampler=LayerSampler(
            cognitive.layer_sampler,
            backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
            state_dim=STATE_DIM,
        ),
        goal_encoder=GoalEncoder(state_dim=STATE_DIM),
        module_input_router=ModuleInputRouter(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
        ),
        shared_buffer_reader=reader,
        cognitive_modules=CognitiveModule(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
        ),
        proposal_head=ProposalHead(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
            cycles=3,
        ),
        shared_buffer_writer=writer,
        shared_buffer_latent_bridge=SharedBufferLatentBridge(
            state_dim=STATE_DIM,
            backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
            num_modules=NUM_MODULES,
            cycles=3,
        ),
        sequence_assembler=SecondPassSequenceAssembler(
            hidden_size=BACKBONE_HIDDEN_SIZE,
            num_slots=NUM_MODULES,
            vocabulary_size=VOCABULARY_SIZE,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            max_target_tokens=MAX_TARGET_TOKENS,
            max_context_tokens=C_MAX_CONTEXT_TOKENS,
        ),
    )
    return Harness(model=model, backbone=backbone, reader=reader, writer=writer)


def test_public_surface_and_exact_nine_component_ownership() -> None:
    harness = make_harness()
    assert training.IrisSharedBufferTeacherForcingModel is IrisSharedBufferTeacherForcingModel
    assert training.IrisSharedBufferTeacherForcingError is IrisSharedBufferTeacherForcingError
    assert (
        training.IrisSharedBufferTeacherCheckpointProvenanceV1
        is IrisSharedBufferTeacherCheckpointProvenanceV1
    )
    assert tuple(harness.model._modules) == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "shared_buffer_reader",
        "cognitive_modules",
        "proposal_head",
        "shared_buffer_writer",
        "shared_buffer_latent_bridge",
        "sequence_assembler",
    )
    assert "_backbone" not in harness.model._modules
    assert harness.model.cycles == 3
    assert harness.model.shared_buffer_policy_sha256 == shared_buffer_policy_sha256_v1()


def test_training_runs_three_prior_cycle_reads_plus_one_final_read() -> None:
    harness = make_harness()
    output = harness.model.training_forward(nominal_input())

    assert type(output) is IrisTeacherForcedOutput
    assert output.loss.requires_grad
    assert output.loss.grad_fn is not None
    assert output.supervised_token_count == 2
    assert harness.reader.entry_counts == [0, NUM_MODULES, 2 * NUM_MODULES, 3 * NUM_MODULES]
    assert harness.writer.input_counts == [0, NUM_MODULES, 2 * NUM_MODULES]
    assert [state.entries.shape[1] for state in harness.writer.outputs] == [
        NUM_MODULES,
        2 * NUM_MODULES,
        3 * NUM_MODULES,
    ]
    assert [state.slots.shape[1] for state in output.workspace_trace] == [
        NUM_MODULES,
        2 * NUM_MODULES,
        3 * NUM_MODULES,
    ]
    assert all(not state.slots.requires_grad for state in output.workspace_trace)

    causal = harness.backbone.causal_calls[0]
    inputs_embeds = causal["inputs_embeds"]
    position_ids = causal["position_ids"]
    labels = causal["labels"]
    assert type(inputs_embeds) is torch.Tensor
    assert type(position_ids) is torch.Tensor
    assert type(labels) is torch.Tensor
    assert inputs_embeds.shape[1] == PROMPT_LENGTH + NUM_MODULES + 2
    assert labels[:, : PROMPT_LENGTH + NUM_MODULES].eq(-100).all()
    assert torch.equal(
        position_ids[:, PROMPT_LENGTH : PROMPT_LENGTH + NUM_MODULES],
        torch.tensor([[2, 3, 4]], dtype=torch.int64),
    )
    assert torch.equal(
        position_ids[:, PROMPT_LENGTH + NUM_MODULES :],
        torch.tensor([[5, 6]], dtype=torch.int64),
    )


def test_backward_reaches_all_cycles_but_not_dead_competitive_heads() -> None:
    harness = make_harness()
    output = harness.model.training_forward(nominal_input())
    torch.autograd.backward((output.loss,))

    assert len(harness.writer.proposals) == 3
    for proposal in harness.writer.proposals:
        assert proposal.content.grad is not None
        assert bool(torch.isfinite(proposal.content.grad).all())
    final_content_gradient = harness.writer.proposals[-1].content.grad
    assert final_content_gradient is not None
    assert bool(torch.count_nonzero(final_content_gradient))

    model = harness.model
    output_projection = cast(
        nn.Linear,
        model.cognitive_modules.module_cells[0].output_projection,
    )
    live_parameters: tuple[torch.Tensor, ...] = (
        model.shared_buffer_reader.read_queries,
        model.shared_buffer_writer.gate_projection.weight,
        model.shared_buffer_writer.candidate_projection.weight,
        model.shared_buffer_latent_bridge.bridge_projection.weight,
        output_projection.weight,
        model.proposal_head.content_head.weight,
    )
    for parameter in live_parameters:
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    for dead_head in (
        model.proposal_head.key_head,
        model.proposal_head.bid_head,
        model.proposal_head.confidence_head,
        model.proposal_head.ttl_head,
    ):
        assert all(parameter.grad is None for parameter in dead_head.parameters())
    assert harness.backbone.trainable_parameter_count == 0
    assert not harness.backbone.has_parameter_gradients


def test_evaluation_is_detached_and_preserves_eval_mode() -> None:
    harness = make_harness()
    with pytest.raises(IrisSharedBufferTeacherForcingError, match="evaluation"):
        harness.model.evaluation_forward(nominal_input())

    harness.model.eval()
    output = harness.model.evaluation_forward(nominal_input())
    assert not output.loss.requires_grad
    assert output.loss.grad_fn is None
    assert all(not child.training for child in harness.model.modules())
    assert all(
        not state.slots.requires_grad and state.slots.grad_fn is None
        for state in output.workspace_trace
    )
    assert harness.backbone.causal_calls[-1]["grad_enabled"] is False


def test_checkpoint_provenance_binds_policy_lifecycle_and_current_state() -> None:
    harness = make_harness()
    first = harness.model.checkpoint_provenance
    assert type(first) is IrisSharedBufferTeacherCheckpointProvenanceV1
    assert first.arm_id == "C"
    assert first.teacher_graph == (
        "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
    )
    assert first.shared_buffer_policy_sha256 == shared_buffer_policy_sha256_v1()
    assert first.cognitive_cycles == 3
    assert first.pre_update_read_count == 3
    assert first.final_output_read_count == 1
    assert first.entries_per_cycle == NUM_MODULES
    assert first.final_entry_count == 3 * NUM_MODULES
    assert first.latent_token_count == NUM_MODULES
    assert first.component_names == tuple(harness.model._modules)

    with torch.no_grad():
        harness.model.shared_buffer_latent_bridge.shared_buffer_type_embedding.add_(1.0)
    second = harness.model.checkpoint_provenance
    assert second.sidecar_state_sha256 != first.sidecar_state_sha256


def test_valid_shaped_component_replacement_breaks_exact_ownership() -> None:
    harness = make_harness()
    replacement = RecordingSharedBufferReader()
    replacement.train()
    harness.model.shared_buffer_reader = replacement

    with pytest.raises(IrisSharedBufferTeacherForcingError, match="wiring is not stable"):
        harness.model.training_forward(nominal_input())
    with pytest.raises(IrisSharedBufferTeacherForcingError, match="wiring is not stable"):
        _ = harness.model.checkpoint_provenance


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_training_rejects_disabled_autograd_or_inference(context: Any) -> None:
    harness = make_harness()
    with context(), pytest.raises(IrisSharedBufferTeacherForcingError):
        harness.model.training_forward(nominal_input())
