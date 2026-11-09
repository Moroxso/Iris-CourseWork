"""Closed combined identities and fresh private ownership; no neural claims."""

from typing import Any

import pytest

from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    _AblationControlSessionV1,
    ablation_control_plan_sha256_v1,
    validate_ablation_control_plan_v1,
)
from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.interventions.messages import WorkspaceMessagePlanV1
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1


def plan() -> CognitiveAblationControlPlanV1:
    return CognitiveAblationControlPlanV1(
        ablation=CognitiveAblationPolicyV1(),
        control=WorkspaceInterventionPlanV1(interventions=()),
    )


def test_snapshot_is_owned_and_combined_domain_binds_both_intents() -> None:
    original = plan()
    copied = validate_ablation_control_plan_v1(original)
    assert copied == original and copied is not original
    assert copied.ablation is not original.ablation and copied.control is not original.control
    changed_mode = CognitiveAblationControlPlanV1(
        ablation=CognitiveAblationPolicyV1(broadcast="disabled"),
        control=original.control,
    )
    changed_control = CognitiveAblationControlPlanV1(
        ablation=original.ablation,
        control=WorkspaceInterventionPlanV1(
            interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
        ),
    )
    assert (
        len({ablation_control_plan_sha256_v1(p) for p in (original, changed_mode, changed_control)})
        == 3
    )


@pytest.mark.parametrize("where", ("root", "ablation", "control", "operation"))
def test_injected_attributes_cannot_be_laundered_by_model_dump(where: str) -> None:
    value = CognitiveAblationControlPlanV1(
        ablation=CognitiveAblationPolicyV1(),
        control=WorkspaceInterventionPlanV1(
            interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
        ),
    )
    assert type(value.control) is WorkspaceInterventionPlanV1
    selected: Any = {
        "root": value,
        "ablation": value.ablation,
        "control": value.control,
        "operation": value.control.interventions[0],
    }[where]
    object.__setattr__(selected, "callback", lambda: None)
    with pytest.raises(TypeError):
        validate_ablation_control_plan_v1(value)


@pytest.mark.parametrize("problem", ("dictionary", "subclass", "list", "invalid-policy"))
def test_exact_tree_and_fresh_revalidation(problem: str) -> None:
    value: Any = plan()
    if problem == "dictionary":
        value = value.model_dump()
    elif problem == "subclass":

        class Derived(CognitiveAblationControlPlanV1):
            pass

        value = Derived(ablation=value.ablation, control=value.control)
    elif problem == "list":
        value = value.model_copy(
            update={"control": value.control.model_copy(update={"interventions": []})}
        )
    else:
        value = value.model_copy(
            update={"ablation": value.ablation.model_copy(update={"routing": "random"})}
        )
    with pytest.raises((TypeError, ValueError)):
        validate_ablation_control_plan_v1(value)


@pytest.mark.parametrize("problem", ("missing", "workspace-extra", "zero", "overbound", "bool"))
def test_explicit_exact_message_budget_contract(problem: str) -> None:
    raw: dict[str, Any] = dict(
        ablation=CognitiveAblationPolicyV1(), control=WorkspaceMessagePlanV1()
    )
    if problem == "workspace-extra":
        raw.update(control=WorkspaceInterventionPlanV1(interventions=()), maximum_message_scalars=1)
    elif problem != "missing":
        raw["maximum_message_scalars"] = {"zero": 0, "overbound": 2_000_001, "bool": True}[problem]
    with pytest.raises(ValueError):
        CognitiveAblationControlPlanV1.model_validate(raw)


def context() -> dict[str, Any]:
    return dict(
        example_id="row",
        prompt_sha256="a" * 64,
        target_sha256="b" * 64,
        teacher_graph_sha256="c" * 64,
        sidecar_state_sha256="d" * 64,
        num_modules=4,
        num_slots=4,
        state_dim=8,
        cycles=3,
        winners_per_cycle=2,
    )


def test_private_composed_session_is_one_use() -> None:
    session = _AblationControlSessionV1(plan(), **context())
    session.claim(**context())
    with pytest.raises(ValueError):
        session.claim(**context())


@pytest.mark.parametrize("problem", ("prompt", "target", "graph", "sidecar", "geometry", "phase"))
def test_private_context_and_freshness_are_checked(problem: str) -> None:
    session = _AblationControlSessionV1(plan(), **context())
    changed = context()
    if problem == "geometry":
        changed["state_dim"] = 9
    elif problem == "phase":
        session.ablation._phase = 1
    else:
        changed[
            {
                "prompt": "prompt_sha256",
                "target": "target_sha256",
                "graph": "teacher_graph_sha256",
                "sidecar": "sidecar_state_sha256",
            }[problem]
        ] = "f" * 64
    with pytest.raises(ValueError):
        session.claim(**changed)
    with pytest.raises(ValueError):
        session.claim(**context())


@pytest.mark.parametrize("which", ("ablation", "workspace", "message"))
def test_actual_child_intent_cannot_differ_from_the_captured_parent_hash(which: str) -> None:
    selected = plan()
    if which == "message":
        selected = CognitiveAblationControlPlanV1(
            ablation=selected.ablation,
            control=WorkspaceMessagePlanV1(),
            maximum_message_scalars=65536,
        )
    session = _AblationControlSessionV1(selected, **context())
    if which == "ablation":
        session.ablation._policy = CognitiveAblationPolicyV1(goal="disabled")
    elif which == "workspace":
        from iris.interventions.session import WorkspaceInterventionSessionV1

        assert type(session.control) is WorkspaceInterventionSessionV1
        session.control._plan = WorkspaceInterventionPlanV1(
            interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
        )
    else:
        from iris.interventions.messages import WorkspaceMessageSessionV1

        assert type(session.control) is WorkspaceMessageSessionV1
        session.control.plan = WorkspaceMessagePlanV1().model_copy(update={"kind": "wrong"})
    with pytest.raises(ValueError):
        session.claim(**context())
