"""Actual tiny-Qwen ablation windows, numerical reference and ownership guards."""

import json
import math
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

import iris.application as application
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationError,
    IrisScientificAblationOptimizationOutputV1,
    IrisScientificAblationOptimizerV1,
)
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import (
    IrisScientificOptimizationError,
    IrisScientificOptimizerPolicyV1,
    IrisScientificOptimizerV1,
    _same_runtime,
    scientific_learning_rate_v1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.training.losses import uniform_utilization_loss
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_cognitive_ablation_teacher import _policy as _ablation
from tests.integration.test_scientific_optimization import _close, _policy
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _model(
    routing: Literal["learned", "random", "uniform"] = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    cycles: int = 3,
) -> IrisScientificAblationTeacherV1:
    base, _, _ = _build(cycles=cycles)
    return IrisScientificAblationTeacherV1(
        base=base, ablation=_ablation(routing, goal=goal, broadcast=broadcast)
    )


def _state(model: IrisScientificAblationTeacherV1) -> str:
    return module_state_sha256((("scientific_ablation_teacher", model),))


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
@pytest.mark.parametrize("clip", [1.0, 1e-3])
def test_all_modes_match_full_window_autograd_and_reference_adamw(
    routing: Literal["learned", "random", "uniform"], goal: str, broadcast: str, clip: float
) -> None:
    model = _model(routing, goal=goal, broadcast=broadcast)
    reference = _model(routing, goal=goal, broadcast=broadcast)
    rows = _rows()[:3]
    policy = _policy(max_gradient_norm=clip)
    owner = IrisScientificAblationOptimizerV1(model=model, policy=policy)
    before = _state(model)
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
    utilization = uniform_utilization_loss(
        torch.stack([_close(out.usage.module_probabilities) for out in outputs]).mean(0),
        torch.stack([_close(out.usage.slot_probabilities) for out in outputs]).mean(0),
    )
    loss = other + weights.utilization * utilization
    torch.autograd.backward(loss)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in reference.named_parameters()
    }
    norm = math.sqrt(
        math.fsum(float(g.double().square().sum()) for g in gradients.values() if g is not None)
    )
    if norm > clip:
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip / norm)
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
    assert type(result) is IrisScientificAblationOptimizationOutputV1
    assert result.loss == pytest.approx(float(loss.detach()), rel=1e-7)
    assert result.utilization_loss == pytest.approx(float(utilization.detach()), rel=1e-8)
    assert result.gradient_l2_norm == pytest.approx(norm, rel=2e-5)
    assert result.gradients_clipped == (norm > clip)
    assert result.supervised_token_count == sum(out.supervised_token_count for out in outputs)
    assert result.scientific_state_before_sha256 == before
    assert result.scientific_state_after_sha256 == _state(model) != before
    assert result.model_configuration_sha256 == model.configuration_sha256
    assert result.ablation_policy_sha256 == model.ablation_policy_sha256
    assert result.registered_parameter_count == sum(p.numel() for p in model.parameters())
    assert result.gradient_present_parameter_count == sum(
        p.numel() for p in reference.parameters() if p.grad is not None
    )
    assert result.nonzero_gradient_parameter_count_before_clip == sum(
        int(g.count_nonzero()) for g in captured.values() if g is not None
    )
    assert (
        0
        < result.nonzero_gradient_parameter_count_before_clip
        <= (result.gradient_present_parameter_count)
    )
    scale = min(1.0, clip / norm)
    for name, expected in gradients.items():
        actual = captured[name]
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert torch.allclose(actual, expected * scale, rtol=3e-4, atol=2e-8), name
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(left, right, rtol=3e-5, atol=2e-6)
        assert (left in owner._optimizer.state) == (right in optimizer.state)
        if right in optimizer.state:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.allclose(
                    owner._optimizer.state[left][key],
                    optimizer.state[right][key],
                    rtol=6e-4,
                    atol=2e-8,
                )
    assert owner.completed_steps == 1 and not owner.poisoned
    assert all(p.grad is None for p in model.parameters())
    assert not model.teacher.backbone.has_parameter_gradients


def test_normal_successor_matches_baseline_bitwise_with_distinct_domains() -> None:
    model = _model()
    base, _, _ = _build()
    policy = _policy()
    successor = IrisScientificAblationOptimizerV1(model=model, policy=policy)
    ordinary = IrisScientificOptimizerV1(model=base, policy=policy)
    for rows in (_rows()[:2], _rows()[2:4]):
        a = successor.step(rows, terminal_token_id=1)
        b = ordinary.step(rows, terminal_token_id=1)
        for name in (
            "step_index",
            "example_count",
            "supervised_token_count",
            "learning_rate",
            "gradient_l2_norm",
            "gradients_clipped",
            "loss",
            "language_model_loss",
            "probe_loss",
            "diversity_loss",
            "utilization_loss",
            "norm_loss",
            "purpose",
        ):
            assert getattr(a, name) == getattr(b, name), name
        assert a.configuration_sha256 != b.configuration_sha256
        assert a.window_sha256 != b.window_sha256
        for left, right in zip(model.parameters(), base.parameters(), strict=True):
            assert torch.equal(left, right)
            assert (left in successor._optimizer.state) == (right in ordinary._optimizer.state)
            if left in successor._optimizer.state:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    assert torch.equal(
                        successor._optimizer.state[left][key], ordinary._optimizer.state[right][key]
                    )
    assert successor.completed_steps == ordinary.completed_steps == 2
    assert application.IrisScientificAblationOptimizerV1 is IrisScientificAblationOptimizerV1
    assert application.IrisScientificAblationOptimizationOutputV1 is (
        IrisScientificAblationOptimizationOutputV1
    )
    assert (
        domain_sha256(
            "iris/scientific-ablation-optimizer-configuration/v1",
            successor.configuration_json.encode(),
        )
        == successor.configuration_sha256
    )
    assert (
        json.loads(successor.configuration_json)["kind"] == "iris-scientific-ablation-optimizer-v1"
    )
    assert ordinary.configuration_sha256 == domain_sha256(
        "iris/scientific-optimizer-configuration/v1",
        canonical_json_bytes(
            {
                "policy": policy.model_dump(mode="json"),
                "model_configuration_sha256": base.configuration_sha256,
                "parameter_names": ordinary._topology.names,
            }
        ),
    )


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
def test_unused_paths_keep_parameters_but_no_decay_moments_or_counters(
    routing: Literal["learned", "random", "uniform"],
) -> None:
    model = _model(routing, goal="disabled", broadcast="disabled")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    before = {name: p.clone().detach() for name, p in model.named_parameters()}
    first = owner.step(_rows()[:2], terminal_token_id=1)
    second = owner.step(_rows()[2:4], terminal_token_id=1)
    assert first.scientific_state_after_sha256 == second.scientific_state_before_sha256
    assert second.gradient_present_parameter_count < second.registered_parameter_count
    for name, p, count in zip(
        owner._topology.names, owner._topology.parameters, owner._parameter_steps, strict=True
    ):
        unused = (
            ".goal_encoder." in name
            or ".workspace_reader." in name
            or (".module_input_router." in name and routing != "learned")
            or ".ttl_head." in name
        )
        if unused:
            assert count == 0 and p not in owner._optimizer.state, name
            assert torch.equal(p, before[name]) and p.requires_grad, name
        elif count:
            assert count == 2 and float(owner._optimizer.state[p]["step"]) == 2
    assert owner.completed_steps == 2 and not owner.poisoned


def test_kl_uses_window_mean_not_mean_of_row_kls() -> None:
    model = _model("random")
    rows = _rows()[:3]
    outputs = [model.training_forward(row, terminal_token_id=1) for row in rows]
    mean = sum(float(out.losses.utilization.detach()) for out in outputs) / len(rows)
    del outputs
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    result = owner.step(rows, terminal_token_id=1)
    assert result.utilization_loss < mean - 1e-8


@pytest.mark.parametrize("kind", ["duplicate", "held_out", "bytes", "count", "empty", "list"])
def test_complete_window_prevalidation_precedes_model_calls(kind: str) -> None:
    model = _model()
    policy = _policy(maximum_logical_bytes=1) if kind == "bytes" else _policy()
    if kind == "count":
        policy = _policy(maximum_examples=1)
    owner = IrisScientificAblationOptimizerV1(model=model, policy=policy)
    rows: Any = _rows()[:2]
    if kind == "duplicate":
        rows = (rows[0], rows[0])
    elif kind == "held_out":
        row = rows[1]
        rows = (rows[0], row.model_copy(update={"split": "test", "id": f"test-{row.seed}"}))
    elif kind == "empty":
        rows = ()
    elif kind == "list":
        rows = list(rows)
    before = _state(model)
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(rows, terminal_token_id=1)
    forward.assert_not_called()
    assert before == _state(model) and not owner.poisoned
    assert owner.completed_steps == 0 and not owner._optimizer.state


@pytest.mark.parametrize("terminal", [True, -1, 0, 2, "1"])
def test_only_exact_declared_terminal_is_accepted(terminal: Any) -> None:
    model = _model()
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[:1], terminal_token_id=terminal)
    forward.assert_not_called()
    assert not owner.poisoned and owner.completed_steps == 0


@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_token_budgets_fail_without_weight_state_or_rng_updates(bound: str) -> None:
    model = _model()
    policy = (
        _policy(minimum_target_tokens=10000)
        if bound == "minimum"
        else _policy(maximum_target_tokens=1)
    )
    owner = IrisScientificAblationOptimizerV1(model=model, policy=policy)
    before = _state(model)
    runtime = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(IrisScientificAblationOptimizationError):
        owner.step(_rows()[:2], terminal_token_id=1)
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    assert _state(model) == before
    assert owner.completed_steps == 0 and not owner.poisoned and not owner._optimizer.state
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "example",
        "configuration",
        "scientific_state",
        "invocation",
        "count",
        "teacher_mode",
        "teacher_prompt",
        "teacher_target",
        "teacher_graph",
        "teacher_state",
        "teacher_invocation",
        "teacher_policy",
    ],
)
def test_mislabelled_or_stale_output_is_rejected_before_update(field: str) -> None:
    model = _model()
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    original = model.training_forward
    before = _state(model)

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        updates: dict[str, Any] = {
            "mode": {"mode": "evaluation"},
            "example": {"example_sha256": "a" * 64},
            "configuration": {"configuration_sha256": "a" * 64},
            "scientific_state": {"scientific_state_sha256": "a" * 64},
            "invocation": {"invocation_sha256": "a" * 64},
            "count": {"supervised_token_count": out.supervised_token_count + 1},
        }
        if field in updates:
            return replace(out, **updates[field])
        teacher_updates: dict[str, Any] = {
            "teacher_mode": {"mode": "evaluation"},
            "teacher_prompt": {"prompt_sha256": "a" * 64},
            "teacher_target": {"target_sha256": "a" * 64},
            "teacher_graph": {"teacher_graph_sha256": "a" * 64},
            "teacher_state": {"sidecar_state_sha256": "a" * 64},
            "teacher_invocation": {"invocation_sha256": "a" * 64},
            "teacher_policy": {"policy": _ablation("uniform")},
        }
        return replace(out, teacher=replace(out.teacher, **teacher_updates[field]))

    with (
        patch.object(model, "training_forward", side_effect=corrupt),
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    assert owner.completed_steps == 0 and not owner.poisoned and not owner._optimizer.state
    assert _state(model) == before and all(p.grad is None for p in model.parameters())


def test_replay_scalar_drift_rolls_back_runtime_and_permits_safe_retry() -> None:
    model = _model("random")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    original = model.training_forward
    calls = 0
    before = _state(model)
    runtime = capture_runtime_state(torch.device("cpu"))

    def drift(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        out = original(*args, **kwargs)
        if calls == 4:
            return replace(out, losses=replace(out.losses, norm=out.losses.norm + 1))
        return out

    with (
        patch.object(model, "training_forward", side_effect=drift),
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[:2], terminal_token_id=1)
    assert calls == 4 and _state(model) == before and not owner.poisoned
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    assert all(p.grad is None for p in model.parameters())
    owner.step(_rows()[:2], terminal_token_id=1)
    assert owner.completed_steps == 1


@pytest.mark.parametrize("kind", ["parameter", "data", "buffer", "frozen", "tokenizer"])
def test_external_state_drift_is_rejected_and_permanently_poisoned(kind: str) -> None:
    model = _model()
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    if kind == "parameter":
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
    elif kind == "data":
        next(model.parameters()).data.add_(0.01)
    elif kind == "buffer":
        model.configuration_identity[0] ^= 1
    elif kind == "frozen":
        with torch.no_grad():
            next(model.teacher.backbone._model.parameters()).add_(0.01)
    else:
        model.teacher.backbone._tokenizer.add_tokens(["new-optimizer-token"])
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    forward.assert_not_called()
    assert owner.completed_steps == 0 and owner.poisoned


@pytest.mark.parametrize("kind", ["throw", "no_update", "nan_moment", "cleanup", "cleanup_data"])
def test_uncertain_update_or_cleanup_never_commits_and_poisoning_is_permanent(kind: str) -> None:
    model = _model()
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    step = owner._optimizer.step
    cleanup = owner._optimizer.zero_grad

    def update() -> None:
        if kind == "throw":
            raise RuntimeError("PRIVATE update")
        if kind == "no_update":
            return
        step()
        if kind == "nan_moment":
            next(iter(owner._optimizer.state.values()))["exp_avg"].fill_(float("nan"))

    def clear(*args: Any, **kwargs: Any) -> None:
        if kind == "cleanup":
            raise RuntimeError("PRIVATE cleanup")
        cleanup(*args, **kwargs)
        if kind == "cleanup_data":
            next(model.parameters()).data.add_(0.01)

    with (
        patch.object(owner._optimizer, "step", side_effect=update),
        patch.object(owner._optimizer, "zero_grad", side_effect=clear),
        pytest.raises(IrisScientificAblationOptimizationError) as caught,
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value)
    assert owner.completed_steps == 0 and owner.poisoned
    with pytest.raises(IrisScientificAblationOptimizationError):
        owner.step(_rows()[:1], terminal_token_id=1)


def test_one_live_graph_and_rng_replay_and_reentry_do_not_change_outer_provenance() -> None:
    model = _model("random")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    original = model.training_forward
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    draws: list[float] = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        assert all(reference() is None for reference in references)
        with pytest.raises(IrisScientificAblationOptimizationError):
            owner.step(_rows()[:1], terminal_token_id=1)
        draws.append(float(torch.rand(())))
        out = original(*args, **kwargs)
        references.append(weakref.ref(out.losses.total))
        references.append(weakref.ref(out.trace.teacher.workspace_trace[-1].slots))
        return out

    initial = torch.random.get_rng_state().clone()
    torch.rand(2)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial)
    with patch.object(model, "training_forward", side_effect=forward):
        owner.step(_rows()[:2], terminal_token_id=1)
    assert draws[:2] == draws[2:]
    assert all(reference() is None for reference in references)
    assert torch.equal(torch.random.get_rng_state(), expected)
    assert owner.completed_steps == 1 and not owner.poisoned


def test_legacy_owner_rejects_new_type_and_new_owner_rejects_baseline() -> None:
    model = _model()
    with pytest.raises(IrisScientificOptimizationError):
        IrisScientificOptimizerV1(model=model, policy=_policy())  # type: ignore[arg-type]
    with pytest.raises(IrisScientificAblationOptimizationError):
        IrisScientificAblationOptimizerV1(model=model.base, policy=_policy())  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["state", "counter", "group", "policy", "configuration"])
def test_owned_optimizer_metadata_and_adamw_drift_is_rejected_before_forward(kind: str) -> None:
    model = _model("uniform", goal="disabled")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    owner.step(_rows()[:1], terminal_token_id=1)
    if kind == "state":
        owner._optimizer.state.pop(next(iter(owner._optimizer.state)))
    elif kind == "counter":
        next(iter(owner._optimizer.state.values()))["step"].add_(1)
    elif kind == "group":
        owner._optimizer.param_groups[0]["weight_decay"] = 0.0
    elif kind == "policy":
        owner._policy = _policy(weight_decay=0.0)
    else:
        owner._configuration_json = "{}"
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[1:2], terminal_token_id=1)
    forward.assert_not_called()
    assert owner.completed_steps == 1


@pytest.mark.parametrize("kind", ["evaluation", "no_grad", "inference", "autocast"])
def test_explicit_training_precision_and_runtime_boundary(kind: str) -> None:
    model = _model()
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    if kind == "evaluation":
        model.eval()
    context = (
        torch.no_grad()
        if kind == "no_grad"
        else torch.inference_mode()
        if kind == "inference"
        else torch.autocast("cpu", dtype=torch.bfloat16)
        if kind == "autocast"
        else nullcontext()
    )
    with (
        context,
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[:1], terminal_token_id=1)
    forward.assert_not_called()
    assert owner.completed_steps == 0 and not owner._optimizer.state


@pytest.mark.parametrize("kind", ["rng", "state"])
def test_backward_side_effect_restores_runtime_or_poison_state_without_repair(kind: str) -> None:
    model = _model("random")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy())
    initial = capture_runtime_state(torch.device("cpu"))
    before = _state(model)

    def mutate(gradient: torch.Tensor) -> torch.Tensor:
        if kind == "rng":
            torch.rand(())
        else:
            next(model.parameters()).data.add_(0.01)
        return gradient

    handle = next(model.parameters()).register_hook(mutate)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(IrisScientificAblationOptimizationError):
            owner.step(_rows()[:1], terminal_token_id=1)
    finally:
        handle.remove()
    assert owner.completed_steps == 0 and not owner._optimizer.state
    assert all(p.grad is None for p in model.parameters())
    assert _same_runtime(initial, capture_runtime_state(torch.device("cpu")))
    assert owner.poisoned == (kind == "state")
    assert (_state(model) == before) == (kind == "rng")


def test_schedule_budget_remains_numeric_policy_and_no_forward_after_exhaustion() -> None:
    model = _model("uniform")
    owner = IrisScientificAblationOptimizerV1(model=model, policy=_policy(max_optimizer_steps=2))
    first = owner.step(_rows()[:1], terminal_token_id=1)
    second = owner.step(_rows()[1:2], terminal_token_id=1)
    assert first.learning_rate == 3e-4
    assert second.learning_rate == pytest.approx(3e-5)
    with (
        patch.object(model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationOptimizationError),
    ):
        owner.step(_rows()[2:3], terminal_token_id=1)
    forward.assert_not_called()
    assert owner.completed_steps == 2 and not owner.poisoned


def test_default_source_sized_ablation_window_has_real_16384_plus_target_tokens(
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    model = _model("random", goal="disabled", broadcast="disabled", cycles=1)
    candidates = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=600),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows: list[IrisExample] = []
    token_count = 0
    for row in candidates:
        rows.append(row)
        token_count += model.teacher.backbone.tokenize_teacher_target(
            row.expected_response, terminal_token_id=1
        ).input_ids.shape[1]
        if token_count >= 16384:
            break
    owner = IrisScientificAblationOptimizerV1(model=model, policy=IrisScientificOptimizerPolicyV1())
    result = owner.step(tuple(rows), terminal_token_id=1)
    assert result.supervised_token_count == token_count >= 16384
    assert result.example_count == len(rows) and result.purpose == "scientific"
    assert result.gradient_present_parameter_count < result.registered_parameter_count
    assert owner.completed_steps == 1 and not owner.poisoned
    for name, value in (
        ("examples", result.example_count),
        ("tokens", result.supervised_token_count),
        ("cycles", 1),
        ("mode", "random-goal-disabled-broadcast-disabled"),
        ("window_sha256", result.window_sha256),
        ("loss", result.loss),
    ):
        record_testsuite_property("scientific_ablation_window_" + name, value)
    assert all(p.grad is None for p in model.parameters())
    assert not model.teacher.backbone.has_parameter_gradients
