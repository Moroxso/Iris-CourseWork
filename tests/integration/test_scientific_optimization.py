"""Token-window optimizer equivalence on genuine tiny random Qwen; no downloads."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from iris.application.scientific_optimization import (
    IrisScientificOptimizationError,
    IrisScientificOptimizerPolicyV1,
    IrisScientificOptimizerV1,
    scientific_learning_rate_v1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.training.losses import uniform_utilization_loss
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _policy(**updates: Any) -> IrisScientificOptimizerPolicyV1:
    return IrisScientificOptimizerPolicyV1.model_validate(
        {
            "purpose": "smoke",
            "minimum_target_tokens": 1,
            "maximum_target_tokens": 65536,
            "max_optimizer_steps": 20,
            **updates,
        }
    )


def _close(value: torch.Tensor) -> torch.Tensor:
    value = value.double()
    return value / value.sum()


@pytest.mark.parametrize("clip", [1.0, 1e-3])
def test_two_pass_gradients_and_adamw_match_retained_full_window_reference(clip: float) -> None:
    model, _, frozen = _build()
    reference, _, _ = _build()
    rows = _rows()[:3]
    policy = _policy(max_gradient_norm=clip)
    owner = IrisScientificOptimizerV1(model=model, policy=policy)
    optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=scientific_learning_rate_v1(policy, step=1),
        betas=(policy.beta1, policy.beta2),
        eps=policy.epsilon,
        weight_decay=policy.weight_decay,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    outputs = [reference.training_forward(row, terminal_token_id=1) for row in rows]
    weights = reference.policy.weights
    other = torch.stack(
        [
            out.losses.language_model.double()
            + weights.probe * out.losses.probe.double()
            + weights.diversity * out.losses.diversity.double()
            + weights.norm * out.losses.norm.double()
            for out in outputs
        ]
    ).mean()
    module_mean = torch.stack([_close(out.usage.module_probabilities) for out in outputs]).mean(0)
    slot_mean = torch.stack([_close(out.usage.slot_probabilities) for out in outputs]).mean(0)
    utilization = uniform_utilization_loss(module_mean, slot_mean)
    exact_loss = other + weights.utilization * utilization
    torch.autograd.backward(exact_loss)
    expected_gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in reference.named_parameters()
    }
    expected_norm = float(
        torch.nn.utils.clip_grad_norm_(reference.parameters(), clip, error_if_nonfinite=True)
    )
    optimizer.step()
    captured: dict[str, torch.Tensor | None] = {}
    actual_step = owner._optimizer.step

    def observe_update() -> None:
        captured.update(
            {
                name: None if p.grad is None else p.grad.clone()
                for name, p in model.named_parameters()
            }
        )
        actual_step()

    with patch.object(owner._optimizer, "step", side_effect=observe_update):
        result = owner.step(rows, terminal_token_id=1)
    assert result.example_count == 3
    assert result.supervised_token_count == sum(out.supervised_token_count for out in outputs)
    assert result.loss == pytest.approx(float(exact_loss.detach()), rel=1e-7)
    assert result.utilization_loss == pytest.approx(float(utilization.detach()), rel=1e-8)
    assert result.gradient_l2_norm == pytest.approx(expected_norm, rel=2e-5)
    scale = min(1.0, clip / result.gradient_l2_norm)
    for name, expected in expected_gradients.items():
        actual = captured[name]
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert torch.allclose(actual, expected * scale, rtol=3e-4, atol=2e-8), name
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(left, right, rtol=3e-5, atol=2e-6)
    assert owner.completed_steps == 1 and not owner.poisoned
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None and not p.requires_grad for p in frozen.parameters())


def test_window_kl_is_not_average_of_per_example_kl() -> None:
    model, _, _ = _build()
    rows = _rows()[:3]
    observed = [model.training_forward(row, terminal_token_id=1) for row in rows]
    average = sum(float(out.losses.utilization.detach()) for out in observed) / len(rows)
    del observed
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    result = owner.step(rows, terminal_token_id=1)
    assert result.utilization_loss < average - 1e-8


def test_underfilled_and_overfull_windows_never_update_parameters_or_rng() -> None:
    for policy in (_policy(minimum_target_tokens=10000), _policy(maximum_target_tokens=1)):
        model, _, _ = _build()
        owner = IrisScientificOptimizerV1(model=model, policy=policy)
        before = deepcopy(model.state_dict())
        rng = torch.random.get_rng_state().clone()
        with pytest.raises(IrisScientificOptimizationError):
            owner.step(_rows()[:2], terminal_token_id=1)
        assert owner.completed_steps == 0 and not owner.poisoned
        assert not owner._optimizer.state
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("kind", ["duplicate", "held_out", "logical_budget", "too_many", "empty"])
def test_all_rows_are_prevalidated_before_any_model_call(kind: str) -> None:
    model, _, _ = _build()
    policy = _policy(maximum_logical_bytes=1) if kind == "logical_budget" else _policy()
    if kind == "too_many":
        policy = _policy(maximum_examples=1)
    owner = IrisScientificOptimizerV1(model=model, policy=policy)
    rows = _rows()[:2]
    if kind == "duplicate":
        rows = (rows[0], rows[0])
    elif kind == "held_out":
        row = rows[1]
        rows = (rows[0], row.model_copy(update={"split": "test", "id": f"test-{row.seed}"}))
    elif kind == "empty":
        rows = ()
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificOptimizationError),
    ):
        owner.step(rows, terminal_token_id=1)
    forward.assert_not_called()


def test_replay_drift_is_rejected_before_update_and_cleans_gradients() -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    before = deepcopy(model.state_dict())
    original = model.training_forward
    calls = 0

    def drift(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        output = original(*args, **kwargs)
        return (
            replace(output, supervised_token_count=output.supervised_token_count + 1)
            if calls == 4
            else output
        )

    with (
        patch.object(model, "training_forward", side_effect=drift),
        pytest.raises(IrisScientificOptimizationError),
    ):
        owner.step(_rows()[:2], terminal_token_id=1)
    assert calls == 4 and owner.completed_steps == 0 and not owner.poisoned
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    owner.step(_rows()[:2], terminal_token_id=1)  # Safe retry after pre-update failure.
    assert owner.completed_steps == 1


def test_rng_draws_are_replayed_but_advance_the_caller_only_once() -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    original = model.training_forward
    draws = []

    def random_forward(*args: Any, **kwargs: Any) -> Any:
        draws.append(float(torch.rand(())))
        return original(*args, **kwargs)

    initial = torch.random.get_rng_state().clone()
    torch.rand(2)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial)
    with patch.object(model, "training_forward", side_effect=random_forward):
        owner.step(_rows()[:2], terminal_token_id=1)
    assert draws[:2] == draws[2:]
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_optimizer_update_failure_poisoning_and_external_parameter_drift() -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    with (
        patch.object(owner._optimizer, "step", side_effect=RuntimeError("PRIVATE update failure")),
        pytest.raises(IrisScientificOptimizationError) as caught,
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value) and owner.poisoned
    assert owner.completed_steps == 0 and all(p.grad is None for p in model.parameters())
    with pytest.raises(IrisScientificOptimizationError):
        owner.step(_rows()[:1], terminal_token_id=1)
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(IrisScientificOptimizationError):
        owner.step(_rows()[:1], terminal_token_id=1)
    assert owner.poisoned


def test_schedule_source_boundaries_and_step_budget() -> None:
    policy = IrisScientificOptimizerPolicyV1()
    assert scientific_learning_rate_v1(policy, step=1) == pytest.approx(3e-4 / 1500)
    assert scientific_learning_rate_v1(policy, step=1500) == pytest.approx(3e-4)
    assert scientific_learning_rate_v1(policy, step=30000) == pytest.approx(3e-5)
    assert scientific_learning_rate_v1(policy, step=1501) < 3e-4
    assert scientific_learning_rate_v1(_policy(warmup_ratio=0.0), step=1) == 3e-4
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy(max_optimizer_steps=2))
    first = owner.step(_rows()[:1], terminal_token_id=1)
    second = owner.step(_rows()[1:2], terminal_token_id=1)
    assert first.step_index == 1 and second.step_index == 2
    assert second.learning_rate == pytest.approx(3e-5)
    with pytest.raises(IrisScientificOptimizationError):
        owner.step(_rows()[:1], terminal_token_id=1)


@pytest.mark.parametrize("kind", ["no_update", "nan_moment", "cleanup_failure"])
def test_uncertain_post_update_never_commits_and_poisoning_is_permanent(kind: str) -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    actual_step = owner._optimizer.step

    def corrupt_update() -> None:
        if kind == "no_update":
            return
        actual_step()
        if kind == "nan_moment":
            next(iter(owner._optimizer.state.values()))["exp_avg"].fill_(float("nan"))

    def broken_cleanup(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("PRIVATE cleanup failure")

    cleanup = broken_cleanup if kind == "cleanup_failure" else owner._optimizer.zero_grad
    with (
        patch.object(owner._optimizer, "step", side_effect=corrupt_update),
        patch.object(owner._optimizer, "zero_grad", side_effect=cleanup),
        pytest.raises(IrisScientificOptimizationError) as caught,
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value)
    assert owner.poisoned and owner.completed_steps == 0
    with pytest.raises(IrisScientificOptimizationError):
        owner.step(_rows()[:1], terminal_token_id=1)


@pytest.mark.parametrize("kind", ["missing_state", "wrong_count", "wrong_group"])
def test_optimizer_state_drift_is_rejected_before_another_forward(kind: str) -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    owner.step(_rows()[:1], terminal_token_id=1)
    if kind == "missing_state":
        owner._optimizer.state.pop(next(iter(owner._optimizer.state)))
    elif kind == "wrong_count":
        next(iter(owner._optimizer.state.values()))["step"].add_(1)
    else:
        owner._optimizer.param_groups[0]["weight_decay"] = 0.0
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificOptimizationError),
    ):
        owner.step(_rows()[1:2], terminal_token_id=1)
    forward.assert_not_called()
    assert owner.completed_steps == 1


def test_backwards_rng_side_effect_is_rejected_and_restored() -> None:
    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    initial = torch.random.get_rng_state().clone()

    def mutate_rng(gradient: torch.Tensor) -> torch.Tensor:
        torch.rand(())
        return gradient

    # Installed torch exposes this hook without a typed callable signature.
    handle = next(model.parameters()).register_hook(mutate_rng)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(IrisScientificOptimizationError):
            owner.step(_rows()[:1], terminal_token_id=1)
    finally:
        handle.remove()
    assert torch.equal(torch.random.get_rng_state(), initial)
    assert not owner.poisoned and owner.completed_steps == 0
    assert not owner._optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())


def test_one_forward_output_is_released_before_next_example_or_replay() -> None:
    import weakref

    model, _, _ = _build()
    owner = IrisScientificOptimizerV1(model=model, policy=_policy())
    original = model.training_forward
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def check_lifetime(*args: Any, **kwargs: Any) -> Any:
        assert all(reference() is None for reference in references)
        output = original(*args, **kwargs)
        references.append(weakref.ref(output.losses.total))
        references.append(weakref.ref(output.trace.teacher.workspace_trace[-1].slots))
        return output

    with patch.object(model, "training_forward", side_effect=check_lifetime):
        owner.step(_rows()[:3], terminal_token_id=1)
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "updates",
    [
        {"minimum_target_tokens": 16383},
        {"minimum_target_tokens": True},
        {"maximum_target_tokens": 100},
        {"learning_rate": float("inf")},
        {"warmup_ratio": 1.0},
        {"precision": "bf16"},
        {"objective_reduction": "token_mean"},
    ],
)
def test_scientific_policy_cannot_silently_become_an_underfilled_smoke(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        IrisScientificOptimizerPolicyV1.model_validate(updates)


def test_source_sized_window_really_accumulates_at_least_16384_target_tokens(
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    model, _, frozen = _build(cycles=1)
    # Real unique development rows and actual tokenizer target counts, never mocked counts.
    candidates = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=600),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows: list[IrisExample] = []
    token_count = 0
    for row in candidates:
        tokenized = model.teacher.backbone.tokenize_teacher_target(
            row.expected_response, terminal_token_id=1
        )
        rows.append(row)
        token_count += tokenized.input_ids.shape[1]
        if token_count >= 16384:
            break
    assert token_count >= 16384
    owner = IrisScientificOptimizerV1(model=model, policy=IrisScientificOptimizerPolicyV1())
    result = owner.step(tuple(rows), terminal_token_id=1)
    assert result.supervised_token_count == token_count and result.example_count == len(rows)
    assert result.purpose == "scientific" and owner.completed_steps == 1
    record_testsuite_property("scientific_window_examples", result.example_count)
    record_testsuite_property("scientific_window_target_tokens", result.supervised_token_count)
    record_testsuite_property("scientific_window_cognitive_cycles", 1)
    record_testsuite_property("scientific_window_loss", result.loss)
    record_testsuite_property("scientific_window_gradient_norm", result.gradient_l2_norm)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in frozen.parameters()
    )
