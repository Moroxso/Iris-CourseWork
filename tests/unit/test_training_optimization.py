"""Adversarial tests for strict teacher-sidecar optimizer ownership."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn

from iris.training import (
    IrisAdamWConfig,
    IrisAdamWParameterSnapshot,
    IrisOptimizationError,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerProvenance,
    IrisTeacherOptimizerSnapshot,
)

CPU = torch.device("cpu")


def _config(*, max_gradient_norm: float = 1.0) -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        max_gradient_norm=max_gradient_norm,
    )


def _example() -> IrisTeacherForcingInput:
    return IrisTeacherForcingInput(
        prompt="optimizer test",
        target_input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
        target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
    )


class _ToyTeacher(IrisTeacherForcingModel):
    """Minimal registered module that satisfies the optimizer-facing surface."""

    def __init__(self, *, unused_parameter: bool = False) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
        if unused_parameter:
            self.unused = nn.Parameter(torch.tensor([7.0], dtype=torch.float32))
        self.mode = "normal"
        self.calls = 0
        self.child = nn.Identity()

    def training_forward(self, example: IrisTeacherForcingInput) -> IrisTeacherForcedOutput:
        del example
        self.calls += 1
        mode = self.mode
        if mode == "raise":
            raise RuntimeError("synthetic forward failure")
        if mode == "raise_once":
            self.mode = "normal"
            raise RuntimeError("synthetic one-shot failure")
        if mode == "mutate_forward":
            with torch.no_grad():
                self.weight.add_(1.0)

        loss = 0.5 * torch.sum(self.weight * self.weight)
        if mode == "detached":
            loss = loss.detach()
        elif mode == "negative":
            loss = -loss
        elif mode == "nonfinite":
            loss = loss * torch.tensor(float("inf"))
        elif mode == "wrong_dtype":
            loss = loss.to(dtype=torch.float64)
        elif mode == "no_parameter_gradient":
            external = torch.ones((), dtype=torch.float32, requires_grad=True)
            loss = external * 1.0
        elif mode == "mutate_backward":

            def mutate_parameter(gradient: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    self.weight.add_(1.0)
                return gradient

            loss.register_hook(mutate_parameter)  # type: ignore[no-untyped-call]

        token_count: int = 2
        if mode == "zero_tokens":
            token_count = 0
        elif mode == "bool_tokens":
            token_count = cast(int, True)
        output = IrisTeacherForcedOutput(
            loss=loss,
            supervised_token_count=token_count,
            workspace_trace=(),
        )
        if mode == "output_subclass":
            return cast(
                IrisTeacherForcedOutput,
                _OutputSubclass(
                    loss=output.loss,
                    supervised_token_count=output.supervised_token_count,
                    workspace_trace=output.workspace_trace,
                ),
            )
        return output


@dataclass(frozen=True, slots=True)
class _OutputSubclass(IrisTeacherForcedOutput):
    pass


class _EmptyTeacher(IrisTeacherForcingModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)


def _build(
    model: _ToyTeacher | None = None,
    *,
    max_gradient_norm: float = 1.0,
) -> tuple[_ToyTeacher, IrisTeacherOptimizer]:
    teacher = model or _ToyTeacher()
    owner = IrisTeacherOptimizer(
        model=teacher,
        config=_config(max_gradient_norm=max_gradient_norm),
    )
    return teacher, owner


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("learning_rate", math.inf),
        ("learning_rate", 1),
        ("beta1", -0.1),
        ("beta1", 1.0),
        ("beta1", True),
        ("beta2", math.nan),
        ("beta2", 1.0),
        ("epsilon", 0.0),
        ("epsilon", math.inf),
        ("weight_decay", -0.1),
        ("weight_decay", math.nan),
        ("max_gradient_norm", 0.0),
        ("max_gradient_norm", math.inf),
    ],
)
def test_config_rejects_noncanonical_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "learning_rate": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        IrisAdamWConfig(**cast(dict[str, float], values))


def test_config_accepts_zero_betas_and_weight_decay() -> None:
    config = IrisAdamWConfig(
        learning_rate=1e-4,
        beta1=0.0,
        beta2=0.0,
        epsilon=1e-12,
        weight_decay=0.0,
        max_gradient_norm=1e-6,
    )

    assert config.beta1 == 0.0
    assert config.beta2 == 0.0


def test_constructor_builds_one_explicit_group_over_every_parameter() -> None:
    teacher = _ToyTeacher(unused_parameter=True)
    _, owner = _build(teacher)
    optimizer = owner._optimizer

    assert type(optimizer) is torch.optim.AdamW
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["params"] == [teacher.weight, teacher.unused]
    assert group["lr"] == 0.1
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0
    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is False
    assert group["capturable"] is False
    assert group["differentiable"] is False
    assert group["fused"] is False
    assert owner.config is owner._config
    assert owner.completed_steps == 0


def test_successful_step_returns_detached_metrics_and_clears_gradients() -> None:
    teacher, owner = _build()
    initial = teacher.weight.detach().clone()

    output = owner.step(_example())

    assert type(output) is IrisTeacherOptimizationOutput
    assert output == IrisTeacherOptimizationOutput(
        step_index=1,
        loss=12.5,
        supervised_token_count=2,
        gradient_l2_norm=5.0,
        gradients_clipped=True,
    )
    assert owner.completed_steps == 1
    assert not torch.equal(teacher.weight, initial)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        not isinstance(value, torch.Tensor)
        for value in (
            output.step_index,
            output.loss,
            output.supervised_token_count,
            output.gradient_l2_norm,
            output.gradients_clipped,
        )
    )


def test_step_index_and_optimizer_moments_persist_across_updates() -> None:
    teacher, owner = _build(max_gradient_norm=10.0)

    first = owner.step(_example())
    state_after_first = {
        id(parameter): {name: value.detach().clone() for name, value in state.items()}
        for parameter, state in owner._optimizer.state.items()
    }
    second = owner.step(_example())

    assert first.step_index == 1
    assert first.gradients_clipped is False
    assert second.step_index == 2
    assert owner.completed_steps == 2
    assert state_after_first
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for parameter, state in owner._optimizer.state.items():
        assert id(parameter) in state_after_first
        assert float(state["step"]) == 2.0


def test_unused_trainable_parameter_is_owned_but_need_not_receive_a_gradient() -> None:
    teacher, owner = _build(_ToyTeacher(unused_parameter=True))
    unused_before = teacher.unused.detach().clone()

    output = owner.step(_example())

    assert output.step_index == 1
    assert torch.equal(teacher.unused, unused_before)
    assert teacher.unused.grad is None
    assert teacher.unused not in owner._optimizer.state


@pytest.mark.parametrize("invalid_model", [nn.Linear(1, 1), object()])
def test_constructor_rejects_non_teacher_models(invalid_model: object) -> None:
    with pytest.raises(TypeError, match="IrisTeacherForcingModel"):
        IrisTeacherOptimizer(
            model=cast(IrisTeacherForcingModel, invalid_model),
            config=_config(),
        )


def test_constructor_rejects_config_subclasses() -> None:
    class ConfigSubclass(IrisAdamWConfig):
        pass

    with pytest.raises(TypeError, match="exactly IrisAdamWConfig"):
        IrisTeacherOptimizer(
            model=_ToyTeacher(),
            config=ConfigSubclass(
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.0,
                max_gradient_norm=1.0,
            ),
        )


def test_constructor_rejects_empty_or_aliased_parameter_topology() -> None:
    with pytest.raises(IrisOptimizationError, match="at least one"):
        IrisTeacherOptimizer(model=_EmptyTeacher(), config=_config())

    aliased = _ToyTeacher()
    aliased.alias = aliased.weight
    with pytest.raises(IrisOptimizationError, match="aliased"):
        IrisTeacherOptimizer(model=aliased, config=_config())


@pytest.mark.parametrize("mutation", ["frozen", "nonfinite", "mixed_dtype", "stale_grad"])
def test_constructor_rejects_invalid_parameter_state(mutation: str) -> None:
    teacher = _ToyTeacher()
    if mutation == "frozen":
        teacher.weight.requires_grad_(False)
    elif mutation == "nonfinite":
        with torch.no_grad():
            teacher.weight[0] = float("nan")
    elif mutation == "mixed_dtype":
        teacher.other = nn.Parameter(torch.ones((), dtype=torch.float64))
    else:
        teacher.weight.grad = torch.ones_like(teacher.weight)

    with pytest.raises(IrisOptimizationError):
        IrisTeacherOptimizer(model=teacher, config=_config())


def test_constructor_rejects_evaluation_mode_including_one_child() -> None:
    teacher = _ToyTeacher()
    teacher.eval()
    with pytest.raises(IrisOptimizationError, match="training mode"):
        IrisTeacherOptimizer(model=teacher, config=_config())

    teacher.train()
    teacher.child.eval()
    with pytest.raises(IrisOptimizationError, match="all registered children"):
        IrisTeacherOptimizer(model=teacher, config=_config())


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (torch.no_grad, "autograd"),
        (torch.inference_mode, "inference_mode"),
    ],
)
def test_step_rejects_disabled_autograd_boundaries(
    context: type[torch.no_grad] | type[torch.inference_mode],
    message: str,
) -> None:
    teacher, owner = _build()

    with context(), pytest.raises(IrisOptimizationError, match=message):
        owner.step(_example())

    assert teacher.calls == 0
    assert owner.completed_steps == 0


def test_step_rejects_ambient_autocast() -> None:
    teacher, owner = _build()

    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(IrisOptimizationError, match="mixed precision"),
    ):
        owner.step(_example())

    assert teacher.calls == 0


def test_step_rejects_input_subclasses_before_forward() -> None:
    class InputSubclass(IrisTeacherForcingInput):
        pass

    teacher, owner = _build()
    example = InputSubclass(
        prompt="x",
        target_input_ids=torch.tensor([[1]]),
        target_attention_mask=torch.tensor([[True]]),
    )

    with pytest.raises(IrisOptimizationError, match="exactly"):
        owner.step(example)

    assert teacher.calls == 0


def test_step_rejects_external_gradients_and_cleans_them() -> None:
    teacher, owner = _build()
    teacher.weight.grad = torch.ones_like(teacher.weight)

    with pytest.raises(IrisOptimizationError, match="Gradient accumulation"):
        owner.step(_example())

    assert teacher.calls == 0
    assert teacher.weight.grad is None
    assert owner.completed_steps == 0


def test_step_rejects_model_or_child_switched_to_evaluation() -> None:
    teacher, owner = _build()
    teacher.eval()
    with pytest.raises(IrisOptimizationError, match="training mode"):
        owner.step(_example())

    teacher.train()
    teacher.child.eval()
    with pytest.raises(IrisOptimizationError, match="all registered children"):
        owner.step(_example())

    assert teacher.calls == 0


def test_step_detects_config_parameter_and_optimizer_drift() -> None:
    teacher, owner = _build()
    object.__setattr__(owner.config, "learning_rate", 0.2)
    with pytest.raises(IrisOptimizationError, match="configuration changed"):
        owner.step(_example())

    teacher, owner = _build()
    teacher.extra = nn.Parameter(torch.ones((), dtype=torch.float32))
    with pytest.raises(IrisOptimizationError, match="topology changed"):
        owner.step(_example())

    teacher, owner = _build()
    owner._optimizer.param_groups[0]["lr"] = 0.2
    with pytest.raises(IrisOptimizationError, match="field 'lr' changed"):
        owner.step(_example())

    assert teacher.calls == 0


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("detached", "graph-attached"),
        ("negative", "graph-attached"),
        ("nonfinite", "graph-attached"),
        ("wrong_dtype", "graph-attached"),
        ("zero_tokens", "token count"),
        ("bool_tokens", "token count"),
        ("output_subclass", "exactly IrisTeacherForcedOutput"),
        ("no_parameter_gradient", "no sidecar gradients"),
        ("mutate_forward", "mutated a sidecar parameter"),
        ("mutate_backward", "loss backward"),
    ],
)
def test_failed_forward_or_backward_never_commits_a_step(mode: str, message: str) -> None:
    teacher, owner = _build()
    teacher.mode = mode

    with pytest.raises(IrisOptimizationError, match=message):
        owner.step(_example())

    assert owner.completed_steps == 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_nonfinite_gradient_is_rejected_before_optimizer_update() -> None:
    teacher = _ToyTeacher()
    teacher.weight.register_hook(  # type: ignore[no-untyped-call]
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    _, owner = _build(teacher)
    before = teacher.weight.detach().clone()

    with pytest.raises(IrisOptimizationError, match="Gradient for sidecar parameter"):
        owner.step(_example())

    assert torch.equal(teacher.weight, before)
    assert teacher.weight.grad is None
    assert owner.completed_steps == 0
    assert not owner._optimizer.state


def test_forward_failure_is_retryable_after_owned_cleanup() -> None:
    teacher, owner = _build()
    teacher.mode = "raise_once"

    with pytest.raises(IrisOptimizationError, match="teacher-forced forward"):
        owner.step(_example())
    output = owner.step(_example())

    assert output.step_index == 1
    assert teacher.calls == 2


def test_optimizer_step_failure_poisons_owner_and_prevents_retry() -> None:
    teacher, owner = _build()

    with (
        patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("partial update")),
        pytest.raises(IrisOptimizationError, match="AdamW update"),
    ):
        owner.step(_example())

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert owner.completed_steps == 0
    calls_after_failure = teacher.calls
    with pytest.raises(IrisOptimizationError, match="poisoned"):
        owner.step(_example())
    assert teacher.calls == calls_after_failure


def test_post_update_corruption_poisons_owner() -> None:
    teacher, owner = _build()
    original_step = owner._optimizer.step

    def corrupting_step() -> None:
        original_step()
        with torch.no_grad():
            teacher.weight[0] = float("nan")

    with (
        patch.object(owner._optimizer, "step", side_effect=corrupting_step),
        pytest.raises(IrisOptimizationError, match="post-update validation"),
    ):
        owner.step(_example())

    with pytest.raises(IrisOptimizationError, match="poisoned"):
        owner.step(_example())


def test_gradient_cleanup_failure_reports_committed_but_poisoned_owner() -> None:
    teacher, owner = _build()
    original_zero_grad = owner._optimizer.zero_grad
    calls = 0

    def fail_second_clear(*, set_to_none: bool = True) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cleanup failure")
        original_zero_grad(set_to_none=set_to_none)

    with (
        patch.object(owner._optimizer, "zero_grad", side_effect=fail_second_clear),
        pytest.raises(IrisOptimizationError, match="committed but gradient cleanup failed"),
    ):
        owner.step(_example())

    assert owner.completed_steps == 0
    assert teacher.weight.grad is not None
    assert owner._poisoned is True


def test_pristine_checkpoint_snapshot_has_topology_but_no_moments() -> None:
    teacher, owner = _build(_ToyTeacher(unused_parameter=True))

    snapshot = owner.checkpoint_snapshot()

    assert type(snapshot) is IrisTeacherOptimizerSnapshot
    assert snapshot.completed_steps == 0
    assert snapshot.config is owner.config
    assert snapshot.parameter_names == tuple(name for name, _ in teacher.named_parameters())
    assert snapshot.parameter_states == ()


def test_lightweight_provenance_validates_policy_topology_and_counter() -> None:
    teacher, owner = _build()

    pristine = owner.provenance_snapshot()
    output = owner.step(_example())
    stepped = owner.provenance_snapshot()

    assert type(pristine) is IrisTeacherOptimizerProvenance
    assert pristine.completed_steps == 0
    assert pristine.config == _config()
    assert pristine.parameter_names == ("weight",)
    assert stepped.completed_steps == output.step_index == 1
    assert stepped.config == pristine.config
    assert stepped.parameter_names == pristine.parameter_names
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_lightweight_provenance_rejects_gradients_poison_and_malformed_state() -> None:
    teacher, owner = _build()
    teacher.weight.grad = torch.ones_like(teacher.weight)
    with pytest.raises(IrisOptimizationError, match="gradients"):
        owner.provenance_snapshot()
    teacher.weight.grad = None

    owner.step(_example())
    state = owner._optimizer.state[teacher.weight]
    state["unknown"] = torch.zeros((), dtype=torch.float32)
    with pytest.raises(IrisOptimizationError, match="canonical"):
        owner.provenance_snapshot()

    _, poisoned = _build()
    poisoned._poisoned = True
    with pytest.raises(IrisOptimizationError, match="poisoned"):
        poisoned.provenance_snapshot()


def test_checkpoint_snapshot_owns_canonical_cpu_moment_copies() -> None:
    teacher, owner = _build(_ToyTeacher(unused_parameter=True))
    owner.step(_example())

    snapshot = owner.checkpoint_snapshot()

    assert snapshot.completed_steps == 1
    assert [state.parameter_name for state in snapshot.parameter_states] == ["weight"]
    state = snapshot.parameter_states[0]
    assert state.step == 1
    assert state.exp_avg.device == CPU
    assert state.exp_avg_sq.device == CPU
    assert state.exp_avg.dtype == teacher.weight.dtype
    assert not state.exp_avg.requires_grad
    assert not state.exp_avg.is_inference()
    optimizer_average = owner._optimizer.state[teacher.weight]["exp_avg"]
    assert state.exp_avg.data_ptr() != optimizer_average.data_ptr()
    expected = optimizer_average.detach().clone()
    state.exp_avg.zero_()
    assert torch.equal(optimizer_average, expected)


def test_checkpoint_snapshot_requires_clean_unpoisoned_owner() -> None:
    teacher, owner = _build()
    teacher.weight.grad = torch.ones_like(teacher.weight)
    with pytest.raises(IrisOptimizationError, match="gradients are present"):
        owner.checkpoint_snapshot()

    teacher.weight.grad = None
    owner._poisoned = True
    with pytest.raises(IrisOptimizationError, match="poisoned"):
        owner.checkpoint_snapshot()


def test_optimizer_checkpoint_round_trip_reproduces_the_next_update() -> None:
    trained, original = _build()
    original.step(_example())
    snapshot = original.checkpoint_snapshot()
    trained_state = {name: tensor.detach().clone() for name, tensor in trained.state_dict().items()}

    fresh_model = _ToyTeacher()
    fresh_model.load_state_dict(trained_state, strict=True)
    _, restored = _build(fresh_model)
    restored.restore_checkpoint_snapshot(snapshot)

    original_output = original.step(_example())
    restored_output = restored.step(_example())

    assert restored_output == original_output
    assert restored.completed_steps == original.completed_steps == 2
    for (left_name, left), (right_name, right) in zip(
        trained.state_dict().items(),
        fresh_model.state_dict().items(),
        strict=True,
    ):
        assert left_name == right_name
        assert torch.equal(left, right)
    original_snapshot = original.checkpoint_snapshot()
    restored_snapshot = restored.checkpoint_snapshot()
    for left, right in zip(
        original_snapshot.parameter_states,
        restored_snapshot.parameter_states,
        strict=True,
    ):
        assert left.parameter_name == right.parameter_name
        assert left.step == right.step
        assert torch.equal(left.exp_avg, right.exp_avg)
        assert torch.equal(left.exp_avg_sq, right.exp_avg_sq)


def test_restore_copies_snapshot_tensors_before_installation() -> None:
    _, source = _build()
    source.step(_example())
    snapshot = source.checkpoint_snapshot()
    fresh, restored = _build()

    restored.restore_checkpoint_snapshot(snapshot)
    installed = restored._optimizer.state[fresh.weight]["exp_avg"].detach().clone()
    snapshot.parameter_states[0].exp_avg.zero_()

    assert torch.equal(restored._optimizer.state[fresh.weight]["exp_avg"], installed)


def test_restore_rejects_nonpristine_or_incompatible_owner() -> None:
    _, source = _build()
    source.step(_example())
    snapshot = source.checkpoint_snapshot()

    _, already_stepped = _build()
    already_stepped.step(_example())
    with pytest.raises(IrisOptimizationError, match="pristine"):
        already_stepped.restore_checkpoint_snapshot(snapshot)

    different_model = _ToyTeacher(unused_parameter=True)
    _, different_topology = _build(different_model)
    with pytest.raises(IrisOptimizationError, match="topology"):
        different_topology.restore_checkpoint_snapshot(snapshot)

    _, different_config = _build(max_gradient_norm=2.0)
    with pytest.raises(IrisOptimizationError, match="configuration"):
        different_config.restore_checkpoint_snapshot(snapshot)


@pytest.mark.parametrize("mutation", ["duplicate", "step", "shape", "nonfinite", "inference"])
def test_restore_rejects_malformed_parameter_snapshots_transactionally(mutation: str) -> None:
    _, source = _build()
    source.step(_example())
    snapshot = source.checkpoint_snapshot()
    state = snapshot.parameter_states[0]
    states: tuple[IrisAdamWParameterSnapshot, ...]
    if mutation == "duplicate":
        states = (state, state)
    elif mutation == "step":
        states = (replace(state, step=2),)
    elif mutation == "shape":
        states = (replace(state, exp_avg=torch.zeros((1,), dtype=torch.float32)),)
    elif mutation == "nonfinite":
        states = (replace(state, exp_avg=torch.full_like(state.exp_avg, float("nan"))),)
    else:
        with torch.inference_mode():
            inference_average = state.exp_avg.clone()
        states = (replace(state, exp_avg=inference_average),)
    malformed = replace(snapshot, parameter_states=states)
    fresh, restored = _build()
    before = fresh.weight.detach().clone()

    with pytest.raises(IrisOptimizationError):
        restored.restore_checkpoint_snapshot(malformed)

    assert restored.completed_steps == 0
    assert not restored._optimizer.state
    assert torch.equal(fresh.weight, before)
    assert restored._poisoned is False


def test_restore_rejects_wrong_snapshot_type_and_zero_step_moments() -> None:
    _, owner = _build()
    with pytest.raises(IrisOptimizationError, match="exactly"):
        owner.restore_checkpoint_snapshot(cast(IrisTeacherOptimizerSnapshot, object()))

    malformed = IrisTeacherOptimizerSnapshot(
        completed_steps=0,
        config=owner.config,
        parameter_names=("weight",),
        parameter_states=(
            IrisAdamWParameterSnapshot(
                parameter_name="weight",
                step=1,
                exp_avg=torch.zeros((2,), dtype=torch.float32),
                exp_avg_sq=torch.zeros((2,), dtype=torch.float32),
            ),
        ),
    )
    with pytest.raises(IrisOptimizationError, match="zero-step"):
        owner.restore_checkpoint_snapshot(malformed)
