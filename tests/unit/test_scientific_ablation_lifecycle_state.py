"""Pure declared-cadence/schema tests, not actual500/1000step training evidence."""

import pytest
from pydantic import ValidationError

from iris.application.scientific_ablation_lifecycle import _validate_cadence
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1 as Policy,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    ScientificAblationLifecycleStateV1 as State,
)
from iris.artifacts.scientific_lifecycle_v1 import ScientificLifecycleStateV1


@pytest.mark.parametrize(
    "values",
    (
        {},
        {"step": 499},
        {"step": 500, "phase": "validation"},
        {
            "step": 500,
            "validation_count": 1,
            "last_validation_step": 500,
            "best_step": 500,
            "best_nll_per_token": 5.0,
        },
        {
            "step": 1000,
            "phase": "validation",
            "validation_count": 1,
            "last_validation_step": 500,
            "best_step": 500,
            "best_nll_per_token": 5.0,
        },
        {
            "step": 1000,
            "phase": "checkpoint",
            "validation_count": 2,
            "last_validation_step": 1000,
            "best_step": 500,
            "best_nll_per_token": 5.0,
            "nonimproving_validations": 1,
        },
        {
            "step": 1000,
            "validation_count": 2,
            "last_validation_step": 1000,
            "best_step": 500,
            "best_nll_per_token": 5.0,
            "nonimproving_validations": 1,
            "checkpoint_count": 1,
            "last_checkpoint_step": 1000,
        },
        {
            "step": 1051,
            "phase": "validation",
            "validation_count": 2,
            "last_validation_step": 1000,
            "best_step": 500,
            "best_nll_per_token": 5.0,
            "nonimproving_validations": 1,
            "checkpoint_count": 1,
            "last_checkpoint_step": 1000,
        },
        {
            "step": 1051,
            "phase": "checkpoint",
            "validation_count": 3,
            "last_validation_step": 1051,
            "best_step": 500,
            "best_nll_per_token": 5.0,
            "nonimproving_validations": 2,
            "checkpoint_count": 1,
            "last_checkpoint_step": 1000,
            "stop_reason": "step_budget",
        },
        {
            "step": 1051,
            "phase": "stopped",
            "validation_count": 3,
            "last_validation_step": 1051,
            "best_step": 500,
            "best_nll_per_token": 5.0,
            "nonimproving_validations": 2,
            "checkpoint_count": 2,
            "last_checkpoint_step": 1051,
            "stop_reason": "step_budget",
        },
    ),
)
def test_source_declared_cadences_and_terminal_pending_phases(values: dict[str, object]) -> None:
    _validate_cadence(State.model_validate(values), Policy(), 1051)


@pytest.mark.parametrize(
    "changes",
    (
        {"validation_count": 1, "nonimproving_validations": 0},
        {"checkpoint_count": 2},
        {"last_checkpoint_step": 999},
        {"best_step": 501, "nonimproving_validations": 0},
        {"phase": "validation"},
        {"stop_reason": "early_stopping", "phase": "checkpoint"},
    ),
)
def test_in_range_but_semantically_forged_counters_reject(changes: dict[str, object]) -> None:
    state = State(
        step=1000,
        validation_count=2,
        last_validation_step=1000,
        best_step=500,
        best_nll_per_token=5.0,
        nonimproving_validations=1,
        checkpoint_count=1,
        last_checkpoint_step=1000,
    ).model_copy(update=changes)
    with pytest.raises((ValueError, ValidationError)):
        _validate_cadence(state, Policy(), 1051)


@pytest.mark.parametrize(
    "kind", ("state_unknown", "policy_unknown", "bool_step", "historical_state")
)
def test_exact_complete_types_are_checked_before_dump(kind: str) -> None:
    state, policy = State(), Policy()
    if kind == "state_unknown":
        state = state.model_copy(update={"not_a_field": True})
    elif kind == "policy_unknown":
        policy = policy.model_copy(update={"not_a_field": True})
    elif kind == "bool_step":
        state = state.model_copy(update={"step": True})
    else:
        with pytest.raises(ValueError):
            _validate_cadence(ScientificLifecycleStateV1(), policy, 10)  # type: ignore[arg-type]
        return
    with pytest.raises(ValueError):
        _validate_cadence(state, policy, 10)
