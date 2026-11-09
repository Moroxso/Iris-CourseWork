"""Masked, differentiable and call-owned source17 cognitive mode math."""

from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.ablations.policy import cognitive_ablation_policy_sha256_v1
from iris.cognitive import ModuleInputRouter, ModuleInputRouterError
from iris.cognitive.outputs import GoalEncoderOutput
from iris.workspace.outputs import WorkspaceReaderOutput


def _policy(
    routing: Literal["learned", "random", "uniform"] = "learned", **extra: Any
) -> CognitiveAblationPolicyV1:
    return CognitiveAblationPolicyV1(
        routing=routing, routing_seed=42 if routing == "random" else None, **extra
    )


def _session(
    policy: CognitiveAblationPolicyV1, *, digest: str = "1" * 64
) -> CognitiveAblationSessionV1:
    return CognitiveAblationSessionV1(
        policy, prompt_sha256=digest, num_modules=3, num_slots=2, state_dim=2, cycles=2
    )


def _goal(length: int, dtype: torch.dtype = torch.float32) -> GoalEncoderOutput:
    return GoalEncoderOutput(
        goal_state=torch.ones(1, 2, dtype=dtype, requires_grad=True),
        attention_weights=torch.ones(1, length, dtype=dtype) / length,
    )


def _reads() -> WorkspaceReaderOutput:
    return WorkspaceReaderOutput(
        workspace_reads=torch.ones(1, 3, 2, requires_grad=True),
        attention_weights=torch.ones(1, 3, 2, requires_grad=True) / 2,
    )


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_routes_are_masked_normalized_finite_differentiable_and_keep_input_state(
    routing: Literal["learned", "random", "uniform"], dtype: torch.dtype
) -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=3).to(dtype=dtype)
    context = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [99.0, -99.0]]], dtype=dtype, requires_grad=True
    )
    mask = torch.tensor([[1, 1, 0]])
    before, rng = context.detach().clone(), torch.get_rng_state().clone()
    session = _session(_policy(routing))
    session.after_goal(_goal(3, dtype))
    result = session.route(router, context, mask)
    assert result.module_inputs.dtype == dtype
    assert torch.count_nonzero(result.attention_weights[..., 2]) == 0
    assert torch.allclose(
        result.attention_weights.sum(-1), torch.ones(1, 3, dtype=result.attention_weights.dtype)
    )
    assert torch.isfinite(result.module_inputs).all()
    assert torch.equal(context, before) and torch.equal(rng, torch.get_rng_state())
    torch.autograd.backward(result.module_inputs.sum())
    assert context.grad is not None and torch.count_nonzero(context.grad[:, 2]) == 0
    assert bool((context.grad[:, :2] != 0).all())
    assert (router.module_queries.grad is None) == (routing != "learned")
    session.after_read(_reads(), 1)
    session.after_read(_reads(), 2)
    session.finish()


def test_learned_route_is_the_identical_native_operation_and_uniform_is_masked_mean() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=3)
    context = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1e5, 1e5]]])
    mask = torch.tensor([[1, 1, 0]])
    with torch.no_grad():
        router.module_queries.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
    ordinary = router(context, mask)
    session = _session(_policy())
    session.after_goal(_goal(3))
    with patch.object(router, "forward", wraps=router.forward) as call:
        learned = session.route(router, context, mask)
    assert call.call_count == 1
    assert torch.equal(ordinary.attention_weights, learned.attention_weights)
    assert torch.equal(ordinary.module_inputs, learned.module_inputs)
    uniform = _session(_policy("uniform"))
    uniform.after_goal(_goal(3))
    result = uniform.route(router, context, mask)
    assert torch.equal(result.module_inputs, torch.tensor([[[2.0, 3.0]]] * 3).transpose(0, 1))
    assert torch.equal(result.attention_weights, torch.tensor([[[0.5, 0.5, 0.0]] * 3]))


def _random(
    context: torch.Tensor,
    mask: torch.Tensor,
    policy: CognitiveAblationPolicyV1 | None = None,
    digest: str = "1" * 64,
) -> Any:
    session = _session(policy or _policy("random"), digest=digest)
    session.after_goal(_goal(context.shape[1]))
    return session.route(ModuleInputRouter(state_dim=2, num_modules=3), context, mask)


def test_random_attention_is_repeatable_and_independent_of_padding() -> None:
    context = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = _random(context, torch.ones(1, 2, dtype=torch.int64))
    padded = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [9e9, -9e9]]])
    shifted = _random(padded, torch.tensor([[0, 1, 0, 1, 0]]))
    assert torch.equal(result.module_inputs, shifted.module_inputs)
    assert torch.equal(result.attention_weights, shifted.attention_weights[..., (1, 3)])
    assert not torch.equal(result.attention_weights[:, 0], result.attention_weights[:, 1])
    assert not torch.equal(result.attention_weights, torch.full((1, 3, 2), 0.5))
    assert torch.equal(
        result.attention_weights,
        _random(context * 2, torch.ones(1, 2, dtype=torch.int64)).attention_weights,
    )
    assert not torch.equal(
        result.attention_weights,
        _random(context, torch.ones(1, 2, dtype=torch.int64), digest="2" * 64).attention_weights,
    )
    other = _policy("random").model_copy(update={"routing_seed": 1337})
    assert not torch.equal(
        result.attention_weights,
        _random(context, torch.ones(1, 2, dtype=torch.int64), other).attention_weights,
    )


def test_goal_and_broadcast_off_are_zero_graph_cuts_without_mutating_original_outputs() -> None:
    original_goal, original_reads = _goal(2), _reads()
    session = _session(_policy(goal="disabled", broadcast="disabled"))
    goal = session.after_goal(original_goal)
    assert not goal.goal_state.requires_grad and not goal.attention_weights.requires_grad
    assert torch.count_nonzero(goal.goal_state) == torch.count_nonzero(goal.attention_weights) == 0
    session.route(
        ModuleInputRouter(state_dim=2, num_modules=3),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.int64),
    )
    reads = session.after_read(original_reads, 1)
    assert not reads.workspace_reads.requires_grad and not reads.attention_weights.requires_grad
    assert (
        torch.count_nonzero(reads.workspace_reads)
        == torch.count_nonzero(reads.attention_weights)
        == 0
    )
    assert bool((original_goal.goal_state == 1).all()) and bool(
        (original_reads.workspace_reads == 1).all()
    )
    assert original_goal.goal_state.requires_grad and original_reads.workspace_reads.requires_grad


def test_enabled_goal_and_broadcast_return_exact_original_objects() -> None:
    goal, reads = _goal(2), _reads()
    session = _session(_policy())
    assert session.after_goal(goal) is goal
    session.route(
        ModuleInputRouter(state_dim=2, num_modules=3),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.int64),
    )
    assert session.after_read(reads, 1) is reads


@pytest.mark.parametrize(
    "bad",
    [
        "premature-finish",
        "premature-route",
        "duplicate-goal",
        "wrong-read",
        "bool-read",
        "bad-goal",
        "bad-route",
        "reuse",
    ],
)
def test_any_failed_or_reused_phase_permanently_closes_session(bad: str) -> None:
    session = _session(_policy())
    router, context, mask = (
        ModuleInputRouter(state_dim=2, num_modules=3),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.int64),
    )
    with pytest.raises((ValueError, TypeError, AttributeError, ModuleInputRouterError)):
        if bad == "premature-finish":
            session.finish()
        elif bad == "premature-route":
            session.route(router, context, mask)
        elif bad == "bad-goal":
            session.after_goal(
                GoalEncoderOutput(goal_state=torch.ones(2, 2), attention_weights=torch.ones(2, 2))
            )
        else:
            session.after_goal(_goal(2))
            if bad == "duplicate-goal":
                session.after_goal(_goal(2))
            elif bad == "bad-route":
                session.route(router, context, torch.zeros_like(mask))
            else:
                session.route(router, context, mask)
                if bad == "wrong-read":
                    session.after_read(_reads(), 2)
                elif bad == "bool-read":
                    session.after_read(_reads(), True)
                else:
                    session.after_read(_reads(), 1)
                    session.after_read(_reads(), 2)
                    session.finish()
                    session.finish()
    with pytest.raises(ValueError):
        session.after_goal(_goal(2))
    with pytest.raises(ValueError):
        session.finish()


def test_budget_precedes_any_router_forward_or_random_allocation() -> None:
    session = _session(_policy(maximum_routing_scalars=1))
    session.after_goal(_goal(2))
    router = ModuleInputRouter(state_dim=2, num_modules=3)
    with patch.object(router, "forward") as forward, pytest.raises(ValueError):
        session.route(router, torch.ones(1, 2, 2), torch.ones(1, 2, dtype=torch.int64))
    forward.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        {"routing": "random"},
        {"routing_seed": 42},
        {"routing": "random", "routing_seed": True},
        {"goal": "off"},
        {"broadcast": False},
        {"unknown": 1},
        {"maximum_routing_scalars": True},
        {"routing": "hard"},
    ],
)
def test_policy_is_closed_and_requires_only_relevant_seed(raw: Any) -> None:
    with pytest.raises(ValueError):
        CognitiveAblationPolicyV1.model_validate(raw)


def test_policy_identity_covers_mode_seed_and_budget_and_revalidates_bypass() -> None:
    normal = _policy()
    assert len(cognitive_ablation_policy_sha256_v1(normal)) == 64
    for change in (
        {"goal": "disabled"},
        {"broadcast": "disabled"},
        {"maximum_routing_scalars": 65535},
    ):
        assert cognitive_ablation_policy_sha256_v1(
            normal.model_copy(update=change)
        ) != cognitive_ablation_policy_sha256_v1(normal)
    with pytest.raises(ValueError):
        cognitive_ablation_policy_sha256_v1(normal.model_copy(update={"routing": "random"}))
