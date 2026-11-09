import torch

from iris.training import (
    IrisAdamWConfig,
    IrisNoWorkspaceTeacherForcingModel,
    IrisTeacherOptimizer,
)
from tests.unit.test_no_workspace_teacher_forcing import make_harness
from tests.unit.test_teacher_forcing import nominal_input


def _config() -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )


def test_optimizer_owns_and_updates_exact_arm_b_teacher_topology() -> None:
    harness = make_harness()
    model = harness.model
    assert type(model) is IrisNoWorkspaceTeacherForcingModel
    before = harness.module_latent_bridge.scale.detach().clone()
    owner = IrisTeacherOptimizer(model=model, config=_config())
    assert owner.model is model
    assert owner.completed_steps == 0
    pristine = owner.checkpoint_snapshot()
    assert pristine.parameter_names == ("module_latent_bridge.scale",)
    assert pristine.parameter_states == ()

    output = owner.step(nominal_input())
    assert output.step_index == 1
    assert output.supervised_token_count == 2
    assert output.loss >= 0.0
    assert output.gradient_l2_norm >= 0.0
    assert owner.completed_steps == 1
    assert not torch.equal(harness.module_latent_bridge.scale.detach(), before)
    assert harness.module_latent_bridge.scale.grad is None

    snapshot = owner.checkpoint_snapshot()
    assert snapshot.completed_steps == 1
    assert snapshot.parameter_names == ("module_latent_bridge.scale",)
    assert len(snapshot.parameter_states) == 1
    assert snapshot.parameter_states[0].parameter_name == "module_latent_bridge.scale"


def test_arm_b_optimizer_snapshot_restores_into_fresh_exact_owner() -> None:
    source = make_harness()
    source_owner = IrisTeacherOptimizer(model=source.model, config=_config())
    source_owner.step(nominal_input())
    snapshot = source_owner.checkpoint_snapshot()

    target = make_harness()
    target_owner = IrisTeacherOptimizer(model=target.model, config=_config())
    target_owner.restore_checkpoint_snapshot(snapshot)
    restored = target_owner.checkpoint_snapshot()
    assert restored.completed_steps == snapshot.completed_steps
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
