"""Optimizer ownership contracts for the exact arm-C teacher sidecar."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler
from iris.shared_buffer import (
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
)
from iris.training.optimization import (
    IrisAdamWConfig,
    IrisTeacherOptimizer,
)
from iris.training.shared_buffer_model import IrisSharedBufferTeacherForcingModel
from tests.unit.test_teacher_forcing import (
    BACKBONE_HIDDEN_SIZE,
    CYCLES,
    MAX_CONTEXT_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TARGET_TOKENS,
    NUM_MODULES,
    STATE_DIM,
    VOCABULARY_SIZE,
    EventControl,
    SpyBackbone,
    cognitive_config,
    nominal_input,
)

ARM_C_MAX_CONTEXT_TOKENS = MAX_CONTEXT_TOKENS + 1
DEAD_PROPOSAL_PARAMETERS = {
    "proposal_head.key_head.weight",
    "proposal_head.bid_head.weight",
    "proposal_head.confidence_head.weight",
    "proposal_head.ttl_head.weight",
}


class _ArmCBackbone(SpyBackbone):
    @property
    def max_position_embeddings(self) -> int:
        return ARM_C_MAX_CONTEXT_TOKENS


class _DuckTeacher(nn.Module):
    """Expose the apparent optimizer surface without being an Iris owner."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.float32))


def _config() -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )


def _make_model() -> IrisSharedBufferTeacherForcingModel:
    torch.manual_seed(194)
    config = cognitive_config()
    backbone = _ArmCBackbone(EventControl.empty())
    return IrisSharedBufferTeacherForcingModel(
        backbone=backbone,
        cognitive_config=config,
        policy=build_shared_buffer_policy_v1(),
        layer_sampler=LayerSampler(
            config.layer_sampler,
            backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
            state_dim=STATE_DIM,
        ),
        goal_encoder=GoalEncoder(state_dim=STATE_DIM),
        module_input_router=ModuleInputRouter(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
        ),
        shared_buffer_reader=SharedBufferReader(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
            cycles=CYCLES,
        ),
        cognitive_modules=CognitiveModule(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
        ),
        proposal_head=ProposalHead(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
            cycles=CYCLES,
        ),
        shared_buffer_writer=SharedBufferWriter(
            state_dim=STATE_DIM,
            num_modules=NUM_MODULES,
            cycles=CYCLES,
        ),
        shared_buffer_latent_bridge=SharedBufferLatentBridge(
            state_dim=STATE_DIM,
            backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
            num_modules=NUM_MODULES,
            cycles=CYCLES,
        ),
        sequence_assembler=SecondPassSequenceAssembler(
            hidden_size=BACKBONE_HIDDEN_SIZE,
            num_slots=NUM_MODULES,
            vocabulary_size=VOCABULARY_SIZE,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            max_target_tokens=MAX_TARGET_TOKENS,
            max_context_tokens=ARM_C_MAX_CONTEXT_TOKENS,
        ),
    )


def test_optimizer_owns_and_updates_exact_arm_c_teacher_topology() -> None:
    model = _make_model()
    assert type(model) is IrisSharedBufferTeacherForcingModel
    parameter_names = tuple(name for name, _ in model.named_parameters())
    bridge_before = model.shared_buffer_latent_bridge.shared_buffer_type_embedding.detach().clone()
    owner = IrisTeacherOptimizer(model=model, config=_config())

    assert owner.model is model
    assert owner.completed_steps == 0
    pristine = owner.checkpoint_snapshot()
    assert pristine.parameter_names == parameter_names
    assert pristine.parameter_states == ()

    output = owner.step(nominal_input())

    assert output.step_index == 1
    assert output.supervised_token_count == 2
    assert output.loss >= 0.0
    assert output.gradient_l2_norm >= 0.0
    assert owner.completed_steps == 1
    assert not torch.equal(
        model.shared_buffer_latent_bridge.shared_buffer_type_embedding.detach(),
        bridge_before,
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    snapshot = owner.checkpoint_snapshot()
    active_names = {state.parameter_name for state in snapshot.parameter_states}
    assert snapshot.completed_steps == 1
    assert active_names
    assert DEAD_PROPOSAL_PARAMETERS.isdisjoint(active_names)
    assert "proposal_head.content_head.weight" in active_names
    assert "shared_buffer_latent_bridge.shared_buffer_type_embedding" in active_names


def test_arm_c_optimizer_snapshot_restores_into_fresh_exact_owner() -> None:
    source = _make_model()
    source_owner = IrisTeacherOptimizer(model=source, config=_config())
    source_owner.step(nominal_input())
    snapshot = source_owner.checkpoint_snapshot()

    target = _make_model()
    target_owner = IrisTeacherOptimizer(model=target, config=_config())
    target_owner.restore_checkpoint_snapshot(snapshot)
    restored = target_owner.checkpoint_snapshot()

    assert restored.completed_steps == snapshot.completed_steps == 1
    assert restored.parameter_names == snapshot.parameter_names
    assert len(restored.parameter_states) == len(snapshot.parameter_states)
    for actual, expected in zip(
        restored.parameter_states,
        snapshot.parameter_states,
        strict=True,
    ):
        assert actual.parameter_name == expected.parameter_name
        assert actual.step == expected.step
        assert torch.equal(actual.exp_avg, expected.exp_avg)
        assert torch.equal(actual.exp_avg_sq, expected.exp_avg_sq)


def test_optimizer_rejects_duck_typed_non_owner_fail_closed() -> None:
    with pytest.raises(TypeError, match="IrisSharedBufferTeacherForcingModel"):
        IrisTeacherOptimizer(
            model=cast(IrisSharedBufferTeacherForcingModel, _DuckTeacher()),
            config=_config(),
        )
