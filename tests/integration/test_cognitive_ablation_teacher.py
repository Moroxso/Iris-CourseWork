"""Actual tiny-Qwen training/eval cognitive modes and unmodified baseline contracts."""

from dataclasses import replace
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.application.scientific_optimization import _same_runtime
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.model import IrisTeacherForcingError
from iris.utils.hashing import module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_intervened_teacher import prepared
from tests.integration.test_scientific_teacher import _build
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration


def _policy(
    routing: Literal["learned", "random", "uniform"] = "learned", **extra: Any
) -> CognitiveAblationPolicyV1:
    return CognitiveAblationPolicyV1(
        routing=routing, routing_seed=42 if routing == "random" else None, **extra
    )


def _options(mode: Literal["metadata", "full"] = "metadata") -> WorkspaceTraceOptionsV1:
    return WorkspaceTraceOptionsV1(
        mode=mode, example_id="ablation-example", checkpoint_id="selected", seed=42
    )


def test_normal_policy_has_bitwise_baseline_loss_states_cycle_graph_and_gradients() -> None:
    model, _, lm = _build()
    teacher, example = model.teacher, prepared(model)
    normal = teacher.training_forward_with_trace(example, diagnostics=_options("full"))
    torch.autograd.backward(normal.teacher.loss)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in teacher.named_parameters()
    }
    teacher.zero_grad(set_to_none=True)
    before = module_state_sha256((("teacher", teacher), ("qwen", lm)))
    caller = capture_runtime_state(torch.device("cpu"))
    ablated = teacher.training_forward_ablated(
        example, policy=_policy(), diagnostics=_options("full")
    )
    assert type(ablated) is IrisAblatedTeacherOutputV1 and ablated.mode == "training"
    assert torch.equal(normal.teacher.loss, ablated.traced.teacher.loss)
    assert normal.teacher.supervised_token_count == ablated.traced.teacher.supervised_token_count
    for left, right in zip(
        normal.teacher.workspace_trace, ablated.traced.teacher.workspace_trace, strict=True
    ):
        assert_same(left, right)
    assert (
        normal.teacher.diagnostic_trace is not None
        and ablated.traced.teacher.diagnostic_trace is not None
    )
    assert normal.teacher.diagnostic_trace.cycles == ablated.traced.teacher.diagnostic_trace.cycles
    assert normal.teacher.diagnostic_trace.intervention_plan_sha256 is None
    assert ablated.traced.teacher.diagnostic_trace.intervention_plan_sha256 == ablated.policy_sha256
    assert all(c.module_probabilities.requires_grad for c in ablated.traced.cycles)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert before == module_state_sha256((("teacher", teacher), ("qwen", lm)))
    torch.autograd.backward(ablated.traced.teacher.loss)
    for name, p in teacher.named_parameters():
        expected = gradients[name]
        assert (expected is None and p.grad is None) or (
            expected is not None and p.grad is not None and torch.equal(expected, p.grad)
        )


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_all_twelve_modes_execute_real_training_and_cut_only_selected_parameter_paths(
    routing: Literal["learned", "random", "uniform"], goal: str, broadcast: str
) -> None:
    model, _, lm = _build()
    teacher, example = model.teacher, prepared(model)
    before = module_state_sha256((("model", model), ("qwen", lm)))
    policy = _policy(routing, goal=goal, broadcast=broadcast)
    result = teacher.training_forward_ablated(example, policy=policy, diagnostics=_options())
    assert result.traced.teacher.loss.requires_grad and len(result.traced.cycles) == 3
    torch.autograd.backward(result.traced.teacher.loss)
    assert (teacher.module_input_router.module_queries.grad is None) == (routing != "learned")
    assert (next(teacher.goal_encoder.parameters()).grad is None) == (goal == "disabled")
    assert (teacher.workspace_reader.read_queries.grad is None) == (broadcast == "disabled")
    assert any(
        p.grad is not None and bool((p.grad != 0).any())
        for p in teacher.cognitive_modules.parameters()
    )
    assert any(p.grad is not None for p in teacher.latent_bridge.parameters())
    assert all(p.grad is None for p in lm.parameters())
    assert before == module_state_sha256((("model", model), ("qwen", lm)))
    diagnostic = result.traced.teacher.diagnostic_trace
    assert diagnostic is not None and diagnostic.intervention_plan_sha256 == result.policy_sha256
    if broadcast == "disabled":
        assert all(row.read_attention == ((0.0,) * 4,) * 4 for row in diagnostic.cycles)
    assert diagnostic.cycles[-1].selected_slots and any(diagnostic.cycles[-1].slot_occupancy)
    model.eval()
    evaluated = teacher.evaluation_forward_ablated(example, policy=policy, diagnostics=_options())
    assert not evaluated.traced.teacher.loss.requires_grad
    assert torch.equal(evaluated.traced.teacher.loss, result.traced.teacher.loss)
    assert evaluated.teacher_graph_sha256 == result.teacher_graph_sha256
    assert evaluated.policy_sha256 == result.policy_sha256
    assert evaluated.invocation_sha256 != result.invocation_sha256
    assert evaluated.traced.teacher.diagnostic_trace == diagnostic


@pytest.mark.parametrize("cycles", [1, 5])
def test_source_cycle_extremes_have_complete_real_train_eval_sessions(cycles: int) -> None:
    model, _, _ = _build(cycles=cycles)
    teacher, example = model.teacher, prepared(model)
    policy = _policy("random", goal="disabled", broadcast="disabled")
    trained = teacher.training_forward_ablated(example, policy=policy, diagnostics=_options())
    assert len(trained.traced.cycles) == len(trained.traced.teacher.workspace_trace) == cycles
    torch.autograd.backward(trained.traced.teacher.loss)
    assert teacher.module_input_router.module_queries.grad is None
    assert teacher.workspace_reader.read_queries.grad is None
    model.eval()
    evaluated = teacher.evaluation_forward_ablated(example, policy=policy, diagnostics=_options())
    assert torch.equal(trained.traced.teacher.loss, evaluated.traced.teacher.loss)
    assert trained.traced.teacher.diagnostic_trace == evaluated.traced.teacher.diagnostic_trace


def test_disabled_goal_reaches_modules_arbiter_and_writer_as_zero() -> None:
    model, _, _ = _build()
    model.eval()
    teacher, example = model.teacher, prepared(model)
    goals: list[torch.Tensor] = []
    actual_modules, actual_arbiter, actual_writer = (
        teacher.cognitive_modules.forward,
        teacher.workspace_arbiter.forward,
        teacher.workspace_writer.forward,
    )

    def modules(*args: Any, **kwargs: Any) -> Any:
        goals.append(args[2].clone())
        return actual_modules(*args, **kwargs)

    def arbiter(*args: Any, **kwargs: Any) -> Any:
        goals.append(args[2].clone())
        return actual_arbiter(*args, **kwargs)

    def writer(*args: Any, **kwargs: Any) -> Any:
        goals.append(args[3].clone())
        return actual_writer(*args, **kwargs)

    with (
        patch.object(teacher.cognitive_modules, "forward", side_effect=modules),
        patch.object(teacher.workspace_arbiter, "forward", side_effect=arbiter),
        patch.object(teacher.workspace_writer, "forward", side_effect=writer),
    ):
        teacher.evaluation_forward_ablated(example, policy=_policy(goal="disabled"))
    assert len(goals) == 9 and all(torch.count_nonzero(g) == 0 for g in goals)


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
def test_eval_preserves_existing_gradients_weights_modes_and_caller_rng_and_is_repeatable(
    routing: Literal["learned", "random", "uniform"],
) -> None:
    model, _, lm = _build()
    teacher, example = model.teacher, prepared(model)
    torch.autograd.backward(teacher.training_forward(example).loss)
    grads = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    model.eval()
    before = module_state_sha256((("model", model), ("qwen", lm)))
    caller = capture_runtime_state(torch.device("cpu"))
    result = teacher.evaluation_forward_ablated(
        example, policy=_policy(routing), diagnostics=_options("full")
    )
    second = teacher.evaluation_forward_ablated(
        example, policy=_policy(routing), diagnostics=_options("full")
    )
    assert result.mode == "evaluation" and not result.traced.teacher.loss.requires_grad
    assert torch.equal(result.traced.teacher.loss, second.traced.teacher.loss)
    assert result.invocation_sha256 == second.invocation_sha256
    assert result.traced.teacher.diagnostic_trace == second.traced.teacher.diagnostic_trace
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert before == module_state_sha256((("model", model), ("qwen", lm)))
    assert not model.training and not lm.training
    for name, p in model.named_parameters():
        expected = grads[name]
        assert (expected is None and p.grad is None) or (
            expected is not None and p.grad is not None and torch.equal(expected, p.grad)
        )


def test_random_routing_is_target_independent_while_invocation_binds_the_target() -> None:
    model, _, _ = _build()
    model.eval()
    teacher, example = model.teacher, prepared(model)
    other = replace(
        example, target_input_ids=torch.tensor([[1]]), target_attention_mask=torch.tensor([[1]])
    )
    first = teacher.evaluation_forward_ablated(
        example, policy=_policy("random"), diagnostics=_options("full")
    )
    second = teacher.evaluation_forward_ablated(
        other, policy=_policy("random"), diagnostics=_options("full")
    )
    assert (
        first.target_sha256 != second.target_sha256
        and first.invocation_sha256 != second.invocation_sha256
    )
    assert (
        first.policy_sha256 == second.policy_sha256 and first.prompt_sha256 == second.prompt_sha256
    )
    assert first.traced.teacher.diagnostic_trace == second.traced.teacher.diagnostic_trace


def test_failure_restores_rng_and_redacts_private_prompt() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    caller = capture_runtime_state(torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(13)
        raise RuntimeError(example.prompt)

    with (
        patch.object(CognitiveAblationSessionV1, "route", side_effect=fail),
        pytest.raises(IrisTeacherForcingError) as error,
    ):
        model.teacher.evaluation_forward_ablated(example, policy=_policy())
    assert example.prompt not in str(error.value) and error.value.__suppress_context__
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "invalid",
    [
        "train-eval-mode",
        "eval-train-mode",
        "train-no-grad",
        "inference",
        "autocast",
        "copied-policy",
    ],
)
def test_explicit_runtime_and_policy_boundaries_reject_before_backbone(invalid: str) -> None:
    model, _, _ = _build()
    teacher, example = model.teacher, prepared(model)
    with (
        patch.object(teacher.backbone, "forward_prompt") as forward,
        pytest.raises(IrisTeacherForcingError),
    ):
        if invalid == "train-eval-mode":
            model.eval()
            teacher.training_forward_ablated(example, policy=_policy())
        elif invalid == "eval-train-mode":
            teacher.evaluation_forward_ablated(example, policy=_policy())
        elif invalid == "train-no-grad":
            with torch.no_grad():
                teacher.training_forward_ablated(example, policy=_policy())
        elif invalid == "inference":
            model.eval()
            with torch.inference_mode():
                teacher.evaluation_forward_ablated(example, policy=_policy())
        elif invalid == "autocast":
            model.eval()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                teacher.evaluation_forward_ablated(example, policy=_policy())
        else:
            model.eval()
            teacher.evaluation_forward_ablated(
                example, policy=_policy().model_copy(update={"routing": "random"})
            )
    forward.assert_not_called()


def test_training_dto_is_a_distinct_canonical_public_type() -> None:
    import iris.training as training

    assert training.IrisAblatedTeacherOutputV1 is IrisAblatedTeacherOutputV1


@pytest.mark.parametrize("training", [True, False])
def test_finite_weight_mutation_during_either_forward_is_detected(training: bool) -> None:
    model, _, _ = _build()
    if not training:
        model.eval()
    teacher, example = model.teacher, prepared(model)
    actual = teacher.workspace_reader.forward

    def mutated(*args: Any, **kwargs: Any) -> Any:
        result = actual(*args, **kwargs)
        with torch.no_grad():
            teacher.workspace_reader.read_queries.add_(0.01)
        return result

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(teacher.workspace_reader, "forward", side_effect=mutated),
        pytest.raises(IrisTeacherForcingError),
    ):
        if training:
            teacher.training_forward_ablated(example, policy=_policy())
        else:
            teacher.evaluation_forward_ablated(example, policy=_policy())
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_low_routing_budget_fails_before_router_and_keeps_weights_rng() -> None:
    model, _, lm = _build()
    model.eval()
    teacher, example = model.teacher, prepared(model)
    before = module_state_sha256((("model", model), ("qwen", lm)))
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(teacher.module_input_router, "forward") as forward,
        pytest.raises(IrisTeacherForcingError),
    ):
        teacher.evaluation_forward_ablated(example, policy=_policy(maximum_routing_scalars=1))
    forward.assert_not_called()
    assert before == module_state_sha256((("model", model), ("qwen", lm)))
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_raw_policy_dictionary_cannot_bypass_exact_type_before_backbone() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    raw: Any = _policy().model_dump()
    with (
        patch.object(model.teacher.backbone, "forward_prompt") as forward,
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_ablated(example, policy=raw)
    forward.assert_not_called()


def test_incomplete_session_cannot_return_a_teacher_result() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    with (
        patch.object(
            CognitiveAblationSessionV1, "after_read", side_effect=lambda reads, cycle: reads
        ),
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_ablated(example, policy=_policy())


def test_even_baseline_ablation_control_trace_cannot_be_relabelled_ordinary_collapse() -> None:
    from iris.artifacts.scientific_collapse_v1 import scientific_collapse_witness_v1

    model, _, _ = _build()
    model.eval()
    result = model.teacher.evaluation_forward_ablated(
        prepared(model), policy=_policy(), diagnostics=_options()
    )
    trace = result.traced.teacher.diagnostic_trace
    assert trace is not None
    with pytest.raises(ValueError):
        scientific_collapse_witness_v1(trace)
