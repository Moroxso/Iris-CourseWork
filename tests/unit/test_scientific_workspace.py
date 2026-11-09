"""Input-only stratification and bounded diagnostic execution design."""

from typing import Any

import pytest

from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_workspace_v1 import (
    ScientificWorkspaceMemberV1,
    ScientificWorkspacePolicyV1,
    select_workspace_members_v1,
    workspace_execution_budget_v1,
)
from iris.workspace.causal import WorkspaceCausalPolicyV1


def _members() -> tuple[ScientificWorkspaceMemberV1, ...]:
    return tuple(
        ScientificWorkspaceMemberV1(
            example_id=f"test-{n}",
            example_sha256=f"{n:064x}",
            prompt_sha256=f"{n + 100:064x}",
            teacher_prompt_sha256=f"{n + 200:064x}",
            task_type=TASKS[n // 9],
            difficulty=(n // 3) % 3 + 1,
        )
        for n in range(36)
    )


def test_hash_strata_cover_each_task_level_without_order_or_target_dependence() -> None:
    members = _members()
    policy = ScientificWorkspacePolicyV1(selection_seed=42, examples_per_stratum=2)
    selected = select_workspace_members_v1(members, policy)
    assert len(selected) == 24
    assert select_workspace_members_v1(tuple(reversed(members)), policy) == selected
    # Changing the full example hash (includes targets) must not change ranking.
    altered = tuple(m.model_copy(update={"example_sha256": "f" * 64}) for m in members)
    assert tuple(m.example_id for m in select_workspace_members_v1(altered, policy)) == tuple(
        m.example_id for m in selected
    )
    for task in TASKS:
        for difficulty in (1, 2, 3):
            assert sum((m.task_type, m.difficulty) == (task, difficulty) for m in selected) == 2
    assert (
        select_workspace_members_v1(members, policy.model_copy(update={"selection_seed": 43}))
        != selected
    )


def test_small_or_missing_strata_are_never_duplicated_or_silently_replaced() -> None:
    members = _members()[:4]
    policy = ScientificWorkspacePolicyV1(selection_seed=42, examples_per_stratum=64)
    assert set(select_workspace_members_v1(members, policy)) == set(members)


@pytest.mark.parametrize("field", ["example_id", "prompt_sha256", "teacher_prompt_sha256"])
def test_duplicate_experimental_units_reject(field: str) -> None:
    members = _members()
    bad = (members[0], members[1].model_copy(update={field: getattr(members[0], field)}))
    with pytest.raises(ValueError):
        select_workspace_members_v1(bad, ScientificWorkspacePolicyV1(selection_seed=42))


def _budget(policy: ScientificWorkspacePolicyV1, **geometry: Any) -> tuple[int, int]:
    args = dict(examples=4, modules=4, slots=2, width=8, cycles=2, writers=2)
    args.update(geometry)
    return workspace_execution_budget_v1(policy, **args)


def test_budget_counts_full_transient_causal_reruns_not_just_saved_metadata() -> None:
    policy = ScientificWorkspacePolicyV1(selection_seed=42)
    assert _budget(policy) == (4, 744)
    policy = policy.model_copy(update={"causal": WorkspaceCausalPolicyV1(shuffle_seed=1337)})
    assert _budget(policy) == (20, 6792)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_executions": 19},
        {"maximum_total_trace_scalars": 6791},
        {"maximum_trace_scalars": 185},
        {"causal": WorkspaceCausalPolicyV1(shuffle_seed=42, maximum_trace_scalars=377)},
        {"causal": WorkspaceCausalPolicyV1(shuffle_seed=42, maximum_executions=3)},
    ],
)
def test_limits_fail_without_running_a_teacher(changes: dict[str, Any]) -> None:
    raw = ScientificWorkspacePolicyV1(
        selection_seed=42, causal=WorkspaceCausalPolicyV1(shuffle_seed=42)
    ).model_dump()
    raw.update(changes)
    with pytest.raises(ValueError):
        _budget(ScientificWorkspacePolicyV1.model_validate(raw))


@pytest.mark.parametrize(
    "geometry", [{"cycles": 1}, {"modules": 1}, {"writers": 3}, {"width": True}]
)
def test_causal_geometry_rejects(geometry: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _budget(
            ScientificWorkspacePolicyV1(
                selection_seed=42, causal=WorkspaceCausalPolicyV1(shuffle_seed=42)
            ),
            **geometry,
        )
