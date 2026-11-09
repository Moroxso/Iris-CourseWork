"""Prompt-only phase composition, NOT native generated answers or task quality."""

from dataclasses import fields, replace
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations.controls import ablation_control_plan_sha256_v1
from iris.ablations.generation_controls import (
    _GenerationControlBindingV1,
    _GenerationWorkspaceControlSessionV1,
)
from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application.errors import IrisRequestError
from iris.application.request import IrisRequestOrchestrator, _ScientificAblationOwnedRequest
from iris.application.scientific_optimization import _same_runtime
from iris.cognitive.module_input_router import ModuleInputRouter
from iris.cognitive.outputs import GoalEncoderOutput
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.seeds import capture_runtime_state
from iris.workspace.state import WorkspaceState
from tests.unit.test_workspace_intervention_phases import components, proposal_fixture

type Routing = Literal["learned", "random", "uniform"]
type Switch = Literal["enabled", "disabled"]


def binding(cycles: int = 3) -> _GenerationControlBindingV1:
    return _GenerationControlBindingV1(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, (4, 4, 3, cycles, 2)
    )


def plan(kind: str = "empty") -> WorkspaceInterventionPlanV1:
    if kind == "empty":
        operations: tuple[WorkspaceInterventionSpecV1, ...] = ()
    elif kind == "zero":
        operations = (WorkspaceInterventionSpecV1(kind="zero_workspace"),)
    else:
        operations = (WorkspaceInterventionSpecV1(kind="shuffle_slots", seed=19),)
    return WorkspaceInterventionPlanV1(interventions=operations)


def session() -> _GenerationWorkspaceControlSessionV1:
    return _GenerationWorkspaceControlSessionV1(
        CognitiveAblationPolicyV1(), plan(), binding=binding()
    )


def test_distinct_generation_domain_owns_intent_and_caller_binding() -> None:
    expected, control, policy = binding(), plan("zero"), CognitiveAblationPolicyV1()
    current = _GenerationWorkspaceControlSessionV1(policy, control, binding=expected)
    assert current.plan_sha256 != ablation_control_plan_sha256_v1(current._kernel.plan)
    assert current._binding == expected and current._binding is not expected
    assert current.ablation._policy is not policy
    assert current.workspace._plan is not control
    old = current.plan_sha256
    object.__setattr__(expected, "prompt_sha256", "f" * 64)
    object.__setattr__(control.interventions[0], "kind", "zero_slot")
    object.__setattr__(policy, "goal", "disabled")
    assert current.plan_sha256 == old
    current.claim(binding=binding())


@pytest.mark.parametrize(
    "field",
    (
        "artifact_sha256",
        "source_checkpoint_sha256",
        "lifecycle_sha256",
        "sidecar_state_sha256",
        "prompt_sha256",
        "geometry",
    ),
)
def test_changed_invocation_binding_poisoning_never_reopens(field: str) -> None:
    current = session()
    if field == "geometry":
        changed = replace(binding(), geometry=(4, 4, 3, 1, 2))
    elif field == "artifact_sha256":
        changed = replace(binding(), artifact_sha256="f" * 64)
    elif field == "source_checkpoint_sha256":
        changed = replace(binding(), source_checkpoint_sha256="f" * 64)
    elif field == "lifecycle_sha256":
        changed = replace(binding(), lifecycle_sha256="f" * 64)
    elif field == "sidecar_state_sha256":
        changed = replace(binding(), sidecar_state_sha256="f" * 64)
    else:
        assert field == "prompt_sha256"
        changed = replace(binding(), prompt_sha256="f" * 64)
    with pytest.raises(ValueError):
        current.claim(binding=changed)
    with pytest.raises(ValueError):
        current.claim(binding=binding())


@pytest.mark.parametrize("where", ("policy", "control", "operation"))
def test_injected_callback_rejected_before_any_dump_or_execution(where: str) -> None:
    policy, control = CognitiveAblationPolicyV1(), plan("zero")
    selected: Any = {"policy": policy, "control": control, "operation": control.interventions[0]}[
        where
    ]
    with patch("builtins.print", side_effect=AssertionError("No callback")) as callback:
        object.__setattr__(selected, "model_dump", callback)
        with pytest.raises(TypeError):
            _GenerationWorkspaceControlSessionV1(policy, control, binding=binding())
        callback.assert_not_called()


@pytest.mark.parametrize("which", ("policy", "control", "kernel", "root-plan"))
def test_actual_private_child_intent_and_identity_are_rechecked(which: str) -> None:
    current = session()
    if which == "policy":
        current.ablation._policy = CognitiveAblationPolicyV1(goal="disabled")
    elif which == "control":
        current.workspace._plan = plan("zero")
    elif which == "kernel":
        current._kernel = session()._kernel
    else:
        object.__setattr__(current._kernel.plan, "callback", lambda: None)
    with pytest.raises((TypeError, ValueError)):
        current.claim(binding=binding())
    with pytest.raises(ValueError):
        current.claim(binding=binding())


def test_one_use_claim_and_incomplete_completion_are_fail_closed() -> None:
    current = session()
    current.claim(binding=binding())
    with pytest.raises(ValueError):
        current.claim(binding=binding())
    with pytest.raises(ValueError):
        current.complete()
    other = session()
    other.claim(binding=binding())
    with pytest.raises(ValueError):
        other.complete()
    assert other._state == "failed"


@pytest.mark.parametrize("owned", (False, True))
def test_existing_request_boundaries_reject_new_session_before_graph_effects(owned: bool) -> None:
    request = object.__new__(_ScientificAblationOwnedRequest if owned else IrisRequestOrchestrator)
    wrong: Any = session()
    with (
        patch.object(
            IrisRequestOrchestrator,
            "_validate_runtime_boundary",
            side_effect=AssertionError("No graph entry"),
        ) as boundary,
        pytest.raises(IrisRequestError),
    ):
        IrisRequestOrchestrator._execute_core(request, "prompt", strategy="greedy", ablation=wrong)
    boundary.assert_not_called()


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
@pytest.mark.parametrize("cycles", (1, 3, 5))
@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
def test_all12_modes_real_ordered_phase_completion_c1_c3_c5(
    routing: Routing, goal: Switch, broadcast: Switch, cycles: int, kind: str
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    expected = binding(cycles)
    current = _GenerationWorkspaceControlSessionV1(
        CognitiveAblationPolicyV1(
            routing=routing,
            routing_seed=7 if routing == "random" else None,
            goal=goal,
            broadcast=broadcast,
        ),
        plan(kind),
        binding=expected,
    )
    current.claim(binding=expected)
    reader, arbiter, allocator, writer = components()
    proposal = proposal_fixture()
    proposal = replace(
        proposal,
        **{field.name: getattr(proposal, field.name)[:1].clone() for field in fields(proposal)},
    )
    state = WorkspaceState.empty(
        batch_size=1, num_slots=4, state_dim=3, dtype=torch.float32, device=torch.device("cpu")
    )
    with torch.no_grad():
        actual_goal = current.ablation.after_goal(
            GoalEncoderOutput(torch.ones(1, 3), torch.ones(1, 2) / 2)
        )
        current.ablation.route(
            ModuleInputRouter(state_dim=3, num_modules=4),
            torch.ones(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
        )
        for cycle in range(1, cycles + 1):
            state = current.workspace.before_read(state, cycle)
            reads = current.ablation.after_read(reader(state), cycle)
            current.workspace.after_read(reads, cycle)
            actual_proposal = current.workspace.after_proposal(proposal, cycle)
            arbitration = current.workspace.after_arbitration(
                arbiter(actual_proposal, state, actual_goal.goal_state), cycle
            )
            allocation = current.workspace.after_allocation(
                state, allocator(actual_proposal, state, arbitration), cycle
            )
            written = writer(actual_proposal, state, allocation, actual_goal.goal_state)
            state = current.workspace.after_write(state, written, allocation, cycle)
            if kind == "zero":
                assert not bool(state.occupancy.any()) and not bool(state.slots.any())
        current.workspace.finish()
        current.ablation.finish()
    current.complete()
    assert current._state == "completed"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    with pytest.raises(ValueError):
        current.complete()
