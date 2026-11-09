"""Mode-bound count semantics; synthetic 5000-row fixtures are NOT model forwards."""

from math import exp, log
from typing import Any

import pytest

from iris.artifacts.scientific_ablation_collapse_v1 import (
    ScientificAblationCollapseMemberV1,
    ScientificAblationCollapsePolicyV1,
    ScientificAblationCollapseRowV1,
    ScientificAblationCollapseTraceContextV1,
    scientific_ablation_collapse_budget_v1,
    scientific_ablation_collapse_groups_v1,
    summarize_scientific_ablation_collapse_v1,
)
from iris.artifacts.scientific_collapse_v1 import (
    ScientificCollapseCycleV1,
    scientific_collapse_criterion_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS


def _row(index: int, *, cycle_count: int = 3, active: int = 0) -> ScientificAblationCollapseRowV1:
    return ScientificAblationCollapseRowV1(
        row_index=index,
        capture_row_sha256="1" * 64,
        evaluation_row_sha256="2" * 64,
        evaluation_ack_sha256="3" * 64,
        teacher_invocation_sha256="4" * 64,
        teacher_target_sha256="5" * 64,
        target_token_count=3,
        target_nll_sum=10.0,
        metadata_trace_sha256=f"{index:064x}",
        context=ScientificAblationCollapseTraceContextV1(
            example_id=f"test-{index}",
            checkpoint_sha256="6" * 64,
            ablation_policy_sha256="7" * 64,
            seed=42,
            num_modules=8,
            num_slots=8,
            state_dim=256,
            cycle_count=cycle_count,
            maximum_trace_scalars=65536,
        ),
        cycles=tuple(
            ScientificCollapseCycleV1(
                cycle=cycle, write_modules=(active,), selected_slots=(active,)
            )
            for cycle in range(1, cycle_count + 1)
        ),
    )


def _member(index: int) -> ScientificAblationCollapseMemberV1:
    return ScientificAblationCollapseMemberV1(
        row_index=index,
        example_id=f"test-{index}",
        example_sha256=f"{index:064x}",
        prompt_sha256=f"{index + 5000:064x}",
        cognitive_prompt_sha256=f"{index + 10000:064x}",
        task_type=TASKS[0 if index < 2000 else 1 + (index - 2000) // 1000],
        difficulty=index % 3 + 1,
    )


def _budget(**changes: Any) -> tuple[int, int]:
    values = dict(examples=5000, modules=8, slots=8, width=256, cycles=3, writers=2)
    policy = changes.pop("policy", ScientificAblationCollapsePolicyV1())
    return scientific_ablation_collapse_budget_v1(policy, **{**values, **changes})


def test_all_5000_transient_traces_are_budgeted_not_only_compact_counts() -> None:
    assert _budget() == (5000, 5_535_000)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_examples", 4999),
        ("maximum_teacher_executions", 4999),
        ("maximum_trace_scalars", 1106),
        ("maximum_total_trace_scalars", 5_534_999),
    ],
)
def test_insufficient_budget_never_reduces_coverage(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        _budget(policy=ScientificAblationCollapsePolicyV1.model_validate({field: value}))


@pytest.mark.parametrize(
    "change", [{"writers": 9}, {"modules": True}, {"examples": 0}, {"cycles": 0}]
)
def test_invalid_geometry_fails_before_counting(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _budget(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"trace_mode": "full"},
        {"coverage": "subset"},
        {"difficulty": 4},
        {"minimum_effective_modules": 1},
        {"maximum_teacher_executions": True},
    ],
)
def test_policy_cannot_select_subsets_latents_or_relax_source_gate(change: Any) -> None:
    with pytest.raises(ValueError):
        ScientificAblationCollapsePolicyV1.model_validate(change)


def test_pooled_entropy_is_not_mean_individual_entropy_or_ever_used_count() -> None:
    rows = tuple(_row(index, active=int(index == 3)) for index in range(4))
    summary = summarize_scientific_ablation_collapse_v1(rows, modules=8, slots=8)
    expected = exp(-0.75 * log(0.75) - 0.25 * log(0.25))
    assert summary.module_win_counts == summary.slot_allocation_counts == (9, 3, 0, 0, 0, 0, 0, 0)
    assert summary.effective_active_modules == pytest.approx(expected)
    assert 1 < expected < 2
    assert all(
        summarize_scientific_ablation_collapse_v1(
            (row,), modules=8, slots=8
        ).effective_active_modules
        == 1
        for row in rows
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 1337),
        ("ablation_policy_sha256", "8" * 64),
        ("checkpoint_sha256", "9" * 64),
        ("state_dim", 8),
    ],
)
def test_cannot_pool_different_native_mode_seed_checkpoint_or_geometry(
    field: str, value: Any
) -> None:
    raw = _row(1).model_dump()
    raw["context"][field] = value
    with pytest.raises(ValueError):
        summarize_scientific_ablation_collapse_v1(
            (_row(0), ScientificAblationCollapseRowV1.model_validate(raw)), modules=8, slots=8
        )


@pytest.mark.parametrize("problem", ["duplicate", "domain", "negative", "order", "cycle", "arity"])
def test_invalid_actual_allocation_witnesses_are_rejected(problem: str) -> None:
    raw = _row(0).model_dump()
    cycle = raw["cycles"][0]
    if problem == "duplicate":
        cycle.update(write_modules=(0, 0), selected_slots=(0, 1))
    elif problem == "domain":
        cycle["selected_slots"] = (8,)
    elif problem == "negative":
        cycle["write_modules"] = (-1,)
    elif problem == "order":
        cycle.update(write_modules=(1, 0), selected_slots=(0, 1))
    elif problem == "cycle":
        cycle["cycle"] = 2
    else:
        cycle["selected_slots"] = (0, 1)
    with pytest.raises(ValueError):
        summarize_scientific_ablation_collapse_v1(
            (ScientificAblationCollapseRowV1.model_validate(raw),), modules=8, slots=8
        )


def test_all_twenty_groups_with_empty_cells_and_ordered_membership() -> None:
    members, rows = tuple(_member(i) for i in range(4)), tuple(_row(i, active=i) for i in range(4))
    groups = scientific_ablation_collapse_groups_v1(members, rows, modules=8, slots=8)
    selectors = (
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    )
    assert tuple((group.task_type, group.difficulty) for group in groups) == selectors
    assert len(groups) == 20 and groups[0].examples == 4
    assert any(group.examples == 0 and group.summary is None for group in groups)
    assert tuple(group.examples for group in groups[5:8]) == (2, 1, 1)
    with pytest.raises(ValueError):
        scientific_ablation_collapse_groups_v1(members[::-1], rows, modules=8, slots=8)


def test_synthetic_full_5000_count_groups_and_fixed_criterion_not_learned_science() -> None:
    members = tuple(_member(i) for i in range(5000))
    rows = tuple(_row(i, active=i % 4) for i in range(5000))
    groups = scientific_ablation_collapse_groups_v1(members, rows, modules=8, slots=8)
    summary = groups[0].summary
    assert summary is not None
    assert summary.module_win_counts == summary.slot_allocation_counts == (3750,) * 4 + (0,) * 4
    assert tuple(group.examples for group in groups[1:5]) == (2000, 1000, 1000, 1000)
    assert scientific_collapse_criterion_v1(purpose="scientific", summary=summary).outcome == "met"
    collapsed = summarize_scientific_ablation_collapse_v1(
        tuple(_row(i) for i in range(5000)), modules=8, slots=8
    )
    assert (
        scientific_collapse_criterion_v1(purpose="scientific", summary=collapsed).outcome
        == "not_met"
    )
    assert (
        scientific_collapse_criterion_v1(purpose="smoke", summary=summary).outcome
        == "not_assessed_smoke"
    )
    with pytest.raises(ValueError):
        scientific_collapse_criterion_v1(
            purpose="scientific",
            summary=summarize_scientific_ablation_collapse_v1(rows[:4999], modules=8, slots=8),
        )


def test_scientific_other_geometry_is_not_rescaled_to_pass() -> None:
    rows = []
    for index in range(5000):
        raw = _row(index, active=index % 4).model_dump()
        raw["context"].update(num_modules=4, num_slots=4)
        rows.append(ScientificAblationCollapseRowV1.model_validate(raw))
    summary = summarize_scientific_ablation_collapse_v1(tuple(rows), modules=4, slots=4)
    assert (
        scientific_collapse_criterion_v1(purpose="scientific", summary=summary).outcome
        == "not_applicable_geometry"
    )


def test_unchecked_out_of_domain_difficulty_cannot_drop_members_from_cells() -> None:
    member = _member(0).model_copy(update={"difficulty": 4})
    with pytest.raises(ValueError):
        scientific_ablation_collapse_groups_v1((member,), (_row(0),), modules=8, slots=8)


def test_duplicate_and_empty_rows_cannot_be_counted() -> None:
    row = _row(0)
    for rows in ((), (row, row)):
        with pytest.raises(ValueError):
            summarize_scientific_ablation_collapse_v1(rows, modules=8, slots=8)
