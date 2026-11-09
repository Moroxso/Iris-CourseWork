"""Strict source-section18 intent, graph binding and provenance identities."""

from typing import Any

import pytest
from pydantic import ValidationError

from iris.interventions import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
    validate_intervention_plan_v1,
)

SPECS: tuple[dict[str, Any], ...] = (
    {"kind": "zero_workspace"},
    {"kind": "zero_slot", "slot": 1},
    {"kind": "freeze_slot", "slot": 1},
    {"kind": "swap_slots", "slot": 0, "other_slot": 2},
    {"kind": "shuffle_slots", "seed": 42},
    {"kind": "gaussian_noise", "seed": 42, "sigma": 0.25},
    {
        "kind": "replace_slot",
        "slot": 0,
        "values": (1.0, 2.0, 3.0),
        "source_module": 3,
        "confidence": 0.5,
        "ttl": 1.5,
    },
    {"kind": "block_module_write", "module": 1},
    {"kind": "block_module_read", "module": 1},
    {"kind": "force_module_winner", "module": 1},
    {"kind": "scale_module_bid", "module": 1, "factor": 0.5},
    {"kind": "reduce_workspace_capacity", "capacity": 1},
)


def bind(plan: WorkspaceInterventionPlanV1, **changes: Any) -> None:
    dimensions = dict(num_slots=4, num_modules=4, state_dim=3, cycles=5, winners_per_cycle=2)
    dimensions.update(changes)
    plan.validate_dimensions(**dimensions)


@pytest.mark.parametrize("parameters", SPECS)
def test_all_twelve_exact_kinds_round_trip_and_bind(parameters: dict[str, Any]) -> None:
    spec = WorkspaceInterventionSpecV1.model_validate(parameters)
    plan = WorkspaceInterventionPlanV1(interventions=(spec,))
    bind(plan)
    assert spec.active(1) and spec.active(5)
    snapshot = validate_intervention_plan_v1(plan)
    assert snapshot == plan and snapshot is not plan
    assert snapshot.interventions[0] is not plan.interventions[0]
    assert WorkspaceInterventionPlanV1.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValidationError):
        WorkspaceInterventionSpecV1.model_validate({**parameters, "irrelevant": True})
    with pytest.raises(ValidationError):
        WorkspaceInterventionSpecV1.model_validate({**parameters, "capacity": 2, "slot": 0})
    for key in set(parameters) - {"kind"}:
        with pytest.raises(ValidationError):
            WorkspaceInterventionSpecV1.model_validate(
                {k: v for k, v in parameters.items() if k != key}
            )


@pytest.mark.parametrize("cycles", [(), (0,), (-1,), (True,), (1.0,), (2, 1), (1, 1), [1, 2]])
def test_reject_ambiguous_cycles(cycles: Any) -> None:
    with pytest.raises(ValidationError):
        WorkspaceInterventionSpecV1(kind="zero_workspace", cycles=cycles)


@pytest.mark.parametrize("cycle", [0, -1, True, 1.0, "1", None])
def test_active_requires_exact_one_based_cycle(cycle: Any) -> None:
    with pytest.raises(ValueError):
        WorkspaceInterventionSpecV1(kind="zero_workspace").active(cycle)


@pytest.mark.parametrize(
    "parameters",
    [
        {"kind": "zero_slot", "slot": True},
        {"kind": "zero_slot", "slot": -1},
        {"kind": "zero_slot", "slot": 1.0},
        {"kind": "swap_slots", "slot": 1, "other_slot": 1},
        {"kind": "shuffle_slots", "seed": -1},
        {"kind": "shuffle_slots", "seed": 2**63},
        {"kind": "shuffle_slots", "seed": False},
        {"kind": "gaussian_noise", "seed": 1, "sigma": float("nan")},
        {"kind": "gaussian_noise", "seed": 1, "sigma": float("inf")},
        {"kind": "gaussian_noise", "seed": 1, "sigma": -0.1},
        {"kind": "scale_module_bid", "module": 0, "factor": True},
        {"kind": "scale_module_bid", "module": 0, "factor": -1.0},
        {"kind": "reduce_workspace_capacity", "capacity": 0},
        {
            "kind": "replace_slot",
            "slot": 0,
            "values": (float("inf"),),
            "source_module": 0,
            "confidence": 0.5,
            "ttl": 1.0,
        },
        {
            "kind": "replace_slot",
            "slot": 0,
            "values": (1.0,),
            "source_module": 0,
            "confidence": 1.1,
            "ttl": 1.0,
        },
        {
            "kind": "replace_slot",
            "slot": 0,
            "values": (1.0,),
            "source_module": 0,
            "confidence": 0.5,
            "ttl": 0.9,
        },
    ],
)
def test_reject_invalid_and_nonfinite_parameters(parameters: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WorkspaceInterventionSpecV1.model_validate(parameters)


@pytest.mark.parametrize(
    "index,changes",
    [
        (1, {"slot": 4}),
        (3, {"other_slot": 4}),
        (6, {"source_module": 4}),
        (6, {"values": (1.0, 2.0)}),
        (7, {"module": 4}),
        (11, {"capacity": 5}),
        (0, {"cycles": (6,)}),
    ],
)
def test_dimension_binding_rejects_intent_outside_actual_graph(
    index: int, changes: dict[str, Any]
) -> None:
    plan = WorkspaceInterventionPlanV1(
        interventions=(WorkspaceInterventionSpecV1.model_validate({**SPECS[index], **changes}),)
    )
    with pytest.raises(ValueError):
        bind(plan)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_slots": False},
        {"num_modules": 1.0},
        {"state_dim": -1},
        {"cycles": 0},
        {"winners_per_cycle": 0},
        {"winners_per_cycle": 5},
    ],
)
def test_dimension_binding_rejects_invalid_graph(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        bind(WorkspaceInterventionPlanV1(interventions=()), **changes)


def test_forced_winner_conflicts_use_schedules_and_distinct_modules() -> None:
    always = WorkspaceInterventionSpecV1(kind="force_module_winner", module=0)
    second = WorkspaceInterventionSpecV1(kind="force_module_winner", module=1, cycles=(3,))
    third = WorkspaceInterventionSpecV1(kind="force_module_winner", module=2, cycles=(4,))
    bind(WorkspaceInterventionPlanV1(interventions=(always, always, second, third)))
    with pytest.raises(ValueError, match="forced winners"):
        bind(
            WorkspaceInterventionPlanV1(
                interventions=(always, second, third.model_copy(update={"cycles": (3,)}))
            )
        )
    bind(WorkspaceInterventionPlanV1(interventions=(always,)), cycles=10**12)


def test_forged_instances_are_revalidated_and_subclasses_rejected() -> None:
    spec = WorkspaceInterventionSpecV1(kind="zero_slot", slot=1)
    valid = WorkspaceInterventionPlanV1(interventions=(spec,))
    forged = valid.model_copy(update={"interventions": (spec.model_copy(update={"slot": -1}),)})
    for operation in (bind, validate_intervention_plan_v1, intervention_plan_sha256_v1):
        with pytest.raises(ValidationError):
            operation(forged)

    class DerivedSpec(WorkspaceInterventionSpecV1):
        pass

    class DerivedPlan(WorkspaceInterventionPlanV1):
        pass

    with pytest.raises(TypeError):
        validate_intervention_plan_v1(
            valid.model_copy(update={"interventions": (DerivedSpec(kind="zero_workspace"),)})
        )
    with pytest.raises(TypeError):
        validate_intervention_plan_v1(DerivedPlan(interventions=()))
    with pytest.raises(TypeError):
        validate_intervention_plan_v1(valid.model_dump())
    with pytest.raises(ValidationError):
        bind(valid.model_copy(update={"kind": "baseline"}))
    with pytest.raises(ValidationError):
        WorkspaceInterventionPlanV1(interventions=(spec,) * 17)


def test_plan_identity_binds_order_seed_schedule_and_values() -> None:
    first = WorkspaceInterventionSpecV1(kind="zero_slot", slot=1)
    second = WorkspaceInterventionSpecV1(kind="gaussian_noise", seed=42, sigma=0.25)
    plans = [
        WorkspaceInterventionPlanV1(interventions=()),
        WorkspaceInterventionPlanV1(interventions=(first, second)),
        WorkspaceInterventionPlanV1(interventions=(second, first)),
        WorkspaceInterventionPlanV1(interventions=(first, second.model_copy(update={"seed": 43}))),
        WorkspaceInterventionPlanV1(
            interventions=(first, second.model_copy(update={"sigma": 0.5}))
        ),
        WorkspaceInterventionPlanV1(
            interventions=(first.model_copy(update={"cycles": (1,)}), second)
        ),
    ]
    fingerprints = [intervention_plan_sha256_v1(plan) for plan in plans]
    assert len(set(fingerprints)) == len(plans)
    assert all(len(value) == 64 and value == value.lower() for value in fingerprints)
    assert (
        intervention_plan_sha256_v1(
            WorkspaceInterventionPlanV1.model_validate_json(plans[1].model_dump_json())
        )
        == fingerprints[1]
    )
    replacement = WorkspaceInterventionSpecV1.model_validate(SPECS[6])
    assert intervention_plan_sha256_v1(
        WorkspaceInterventionPlanV1(interventions=(replacement,))
    ) != intervention_plan_sha256_v1(
        WorkspaceInterventionPlanV1(
            interventions=(replacement.model_copy(update={"values": (3.0, 2.0, 1.0)}),)
        )
    )
