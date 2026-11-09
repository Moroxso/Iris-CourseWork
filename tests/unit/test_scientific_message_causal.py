"""Fixed even input-only subset and whole source-bound message budgets."""

from typing import Any

import pytest

from iris.artifacts.scientific_message_causal_v1 import (
    ScientificMessageCausalPolicyV1,
    message_stratum_v1,
    scientific_message_budget_v1,
    select_message_members_v1,
)
from iris.workspace.message_causal import WorkspaceMessageCausalPolicyV1
from tests.unit.test_scientific_workspace import _members


def policy(**changes: Any) -> ScientificMessageCausalPolicyV1:
    return ScientificMessageCausalPolicyV1.model_validate(
        {
            "selection_seed": 42,
            "causal": WorkspaceMessageCausalPolicyV1(shuffle_seed=1337),
            **changes,
        }
    )


def test_even_selection_is_input_only_order_independent_and_never_reuses_a_row() -> None:
    members = _members()
    selected = select_message_members_v1(members, policy())
    assert len(selected) == 24
    assert select_message_members_v1(tuple(reversed(members)), policy()) == selected
    assert len({m.example_id for m in selected}) == len(selected)
    assert {message_stratum_v1(m) for m in selected} == {message_stratum_v1(m) for m in members}
    assert all(
        sum(message_stratum_v1(m) == group for m in selected) == 2
        for group in {message_stratum_v1(m) for m in selected}
    )
    altered = tuple(m.model_copy(update={"example_sha256": "f" * 64}) for m in members)
    assert tuple(m.example_id for m in select_message_members_v1(altered, policy())) == tuple(
        m.example_id for m in selected
    )
    assert select_message_members_v1(members, policy(selection_seed=43)) != selected


def test_odd_available_row_is_excluded_by_rank_and_singletons_remain_unobserved() -> None:
    members = _members()[:4]
    selected = select_message_members_v1(members, policy(examples_per_stratum=64))
    assert len(selected) == 2
    assert all(m.difficulty == 1 for m in selected)
    assert set(selected) < set(members)
    with pytest.raises(ValueError):
        select_message_members_v1(_members()[::3], policy())


@pytest.mark.parametrize("cap", [3, 63])
def test_declared_cap_must_be_even(cap: int) -> None:
    with pytest.raises(ValueError):
        policy(examples_per_stratum=cap)


@pytest.mark.parametrize("field", ["example_id", "prompt_sha256", "teacher_prompt_sha256"])
def test_full_membership_duplicates_reject_even_if_row_would_not_be_selected(field: str) -> None:
    members = _members()
    bad = (*members[:-1], members[-1].model_copy(update={field: getattr(members[0], field)}))
    with pytest.raises(ValueError):
        select_message_members_v1(bad, policy())


def budget(config: ScientificMessageCausalPolicyV1, **changes: Any) -> tuple[int, int]:
    geometry = dict(examples=4, modules=4, slots=2, width=8, cycles=2, writers=2)
    geometry.update(changes)
    return scientific_message_budget_v1(config, **geometry)


def test_whole_budget_includes_ordinary_controls_and_all_actual_message_reruns() -> None:
    assert budget(policy()) == (24, 9072)
    assert budget(policy(trace_mode="full")) == (24, 9840)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_executions": 23},
        {"maximum_total_trace_scalars": 9071},
        {"maximum_trace_scalars": 377},
        {"maximum_message_scalars": 63},
        {"maximum_examples": 2},
    ],
)
def test_combined_limits_fail_before_execution(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        budget(
            policy(
                causal=WorkspaceMessageCausalPolicyV1.model_validate(
                    {"shuffle_seed": 1337, **changes}
                )
            )
        )


@pytest.mark.parametrize(
    "changes", [{"cycles": 1}, {"modules": 1}, {"writers": 3}, {"width": True}]
)
def test_source_geometry_does_not_weaken_core_message_constraints(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        budget(policy(), **changes)
