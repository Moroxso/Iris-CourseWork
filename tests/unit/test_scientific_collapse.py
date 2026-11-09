"""Full-corpus budget, pooled entropy, fixed thresholds and compact witness validation."""

from math import exp, log
from typing import Any

import pytest

from iris.artifacts.scientific_collapse_v1 import (
    ScientificCollapseCycleV1,
    ScientificCollapsePayloadV1,
    ScientificCollapsePolicyV1,
    ScientificCollapseRowV1,
    ScientificCollapseTraceContextV1,
    scientific_collapse_budget_v1,
    scientific_collapse_criterion_v1,
    scientific_collapse_groups_v1,
    summarize_scientific_collapse_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_workspace_v1 import ScientificWorkspaceMemberV1


def _row(
    index: int, modules: tuple[int, ...], slots: tuple[int, ...], *, cycles: int = 1
) -> ScientificCollapseRowV1:
    return ScientificCollapseRowV1(
        example_id=f"test-{index}",
        target_sha256="1" * 64,
        target_token_count=3,
        control_nll_sum=10.0,
        ordinary_trace_sha256=f"{index:064x}",
        ordinary_context=ScientificCollapseTraceContextV1(
            example_id=f"test-{index}",
            checkpoint_sha256="2" * 64,
            seed=42,
            num_modules=8,
            num_slots=8,
            state_dim=256,
            cycle_count=cycles,
            maximum_trace_scalars=65536,
        ),
        cycles=tuple(
            ScientificCollapseCycleV1(cycle=n, write_modules=modules, selected_slots=slots)
            for n in range(1, cycles + 1)
        ),
    )


def _budget(**changes: Any) -> tuple[int, int]:
    arguments = dict(examples=5000, modules=8, slots=8, width=256, cycles=3, writers=2)
    policy = changes.pop("policy", ScientificCollapsePolicyV1())
    arguments.update(changes)
    return scientific_collapse_budget_v1(policy, **arguments)


def test_standard_full_test_fits_metadata_budget_with_exact_all_forward_count() -> None:
    assert _budget() == (5000, 5_535_000)
    assert _budget(examples=1, modules=4, slots=2, width=8, cycles=2) == (1, 186)


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_examples": 4999},
        {"maximum_executions": 4999},
        {"maximum_trace_scalars": 1106},
        {"maximum_total_trace_scalars": 5_534_999},
    ],
)
def test_budget_fails_instead_of_reducing_full_test_coverage(change: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _budget(policy=ScientificCollapsePolicyV1.model_validate(change))


@pytest.mark.parametrize(
    "change",
    [{"examples": 0}, {"examples": 5001}, {"writers": 9}, {"modules": True}, {"cycles": 0}],
)
def test_invalid_geometry_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _budget(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"trace_mode": "full"},
        {"coverage": "ranked-subset"},
        {"selection_seed": 42},
        {"examples_per_stratum": 4},
        {"minimum_effective_modules": 1},
        {"maximum_executions": True},
    ],
)
def test_policy_cannot_enable_subsets_latents_or_relax_source_thresholds(change: Any) -> None:
    with pytest.raises(ValueError):
        ScientificCollapsePolicyV1.model_validate(change)


def test_entropy_pools_raw_counts_not_per_example_effective_counts() -> None:
    rows = (_row(0, (0,), (0,)), _row(1, (1,), (1,)))
    individual = tuple(summarize_scientific_collapse_v1((r,), modules=8, slots=8) for r in rows)
    combined = summarize_scientific_collapse_v1(rows, modules=8, slots=8)
    assert all(s.effective_active_modules == s.effective_active_slots == 1 for s in individual)
    assert combined.module_win_counts == combined.slot_allocation_counts == (1, 1, 0, 0, 0, 0, 0, 0)
    assert combined.effective_active_modules == combined.effective_active_slots == 2
    assert (combined.examples, combined.cycles, combined.allocated_writes) == (2, 2, 2)


def test_nonuniform_effective_count_uses_natural_log_not_number_of_ever_used_modules() -> None:
    rows = tuple(_row(i, (int(i == 3),), (int(i == 3),)) for i in range(4))
    summary = summarize_scientific_collapse_v1(rows, modules=8, slots=8)
    expected = exp(-0.75 * log(0.75) - 0.25 * log(0.25))
    assert summary.effective_active_modules == pytest.approx(expected)
    assert summary.effective_active_slots == pytest.approx(expected)
    assert 1 < expected < 2


@pytest.mark.parametrize("problem", ["duplicate", "negative", "domain", "order", "cycle", "arity"])
def test_invalid_witnesses_cannot_be_aggregated(problem: str) -> None:
    row = _row(0, (0, 1), (0, 1))
    raw = row.model_dump()
    cycle = raw["cycles"][0]
    if problem == "duplicate":
        cycle["selected_slots"] = (0, 0)
    elif problem == "negative":
        cycle["write_modules"] = (-1, 1)
    elif problem == "domain":
        cycle["selected_slots"] = (0, 8)
    elif problem == "order":
        cycle["write_modules"] = (1, 0)
    elif problem == "cycle":
        cycle["cycle"] = 2
    else:
        cycle["write_modules"] = (0,)
    with pytest.raises(ValueError):
        summarize_scientific_collapse_v1(
            (ScientificCollapseRowV1.model_validate(raw),), modules=8, slots=8
        )


def test_duplicate_examples_and_empty_corpus_cannot_be_aggregated() -> None:
    row = _row(0, (0,), (0,))
    for rows in ((), (row, row)):
        with pytest.raises(ValueError):
            summarize_scientific_collapse_v1(rows, modules=8, slots=8)


@pytest.fixture(scope="module")
def scientific_payload() -> ScientificCollapsePayloadV1:
    # Schema-scale fixtures, NOT 5000 model forwards or learned acceptance data.
    members = tuple(
        ScientificWorkspaceMemberV1(
            example_id=f"test-{i}",
            example_sha256=f"{i:064x}",
            prompt_sha256=f"{i + 5000:064x}",
            teacher_prompt_sha256=f"{i + 10000:064x}",
            task_type=TASKS[0 if i < 2000 else 1 + (i - 2000) // 1000],
            difficulty=i % 3 + 1,
        )
        for i in range(5000)
    )
    rows = tuple(
        _row(i, tuple(sorted((i % 4, (i + 1) % 4))), (i % 4, (i + 1) % 4), cycles=3)
        for i in range(5000)
    )
    groups = scientific_collapse_groups_v1(members, rows, modules=8, slots=8)
    summary = groups[0].summary
    assert summary is not None
    return ScientificCollapsePayloadV1(
        purpose="scientific",
        policy=ScientificCollapsePolicyV1(),
        baseline_generation_sha256="1" * 64,
        execution_sha256="1" * 64,
        terminal_event_sha256="1" * 64,
        best_validation_event_sha256="1" * 64,
        best_checkpoint_sha256="2" * 64,
        inference_sha256="1" * 64,
        selected_step=1,
        terminal_step=3,
        run_seed=42,
        teacher_configuration_sha256="1" * 64,
        teacher_graph_sha256="1" * 64,
        teacher_sidecar_sha256="1" * 64,
        backbone_runtime_sha256="1" * 64,
        test_parquet_sha256="1" * 64,
        test_corpus_sha256="1" * 64,
        num_modules=8,
        num_slots=8,
        state_dim=256,
        cycle_count=3,
        writers_per_cycle=2,
        members=members,
        rows=rows,
        executions=5000,
        trace_scalar_bound=5_535_000,
        groups=groups,
        source_20_4=scientific_collapse_criterion_v1(purpose="scientific", summary=summary),
    )


def test_full_5000_row_schema_counts_groups_and_fixed_four_of_eight_gate(
    scientific_payload: ScientificCollapsePayloadV1,
) -> None:
    p = scientific_payload
    assert len(p.members) == len(p.rows) == p.executions == 5000
    summary = p.groups[0].summary
    assert summary is not None
    assert summary.module_win_counts == summary.slot_allocation_counts == (7500,) * 4 + (0,) * 4
    assert summary.effective_active_modules == summary.effective_active_slots == 4
    assert (summary.cycles, summary.allocated_writes) == (15000, 30000)
    assert tuple(g.examples for g in p.groups[1:5]) == (2000, 1000, 1000, 1000)
    assert p.source_20_4.outcome == "met" and p.scientific_acceptance == "not_assessed"
    assert ScientificCollapsePayloadV1.model_validate_json(p.model_dump_json()) == p


def test_smoke_and_other_geometry_never_rescale_scientific_threshold() -> None:
    rows = tuple(_row(i, (0, 1), (0, 1)) for i in range(5000))
    summary = summarize_scientific_collapse_v1(rows, modules=8, slots=8)
    assert (
        scientific_collapse_criterion_v1(purpose="scientific", summary=summary).outcome == "not_met"
    )
    assert (
        scientific_collapse_criterion_v1(purpose="smoke", summary=summary).outcome
        == "not_assessed_smoke"
    )
    smaller_rows = tuple(
        row.model_copy(
            update={
                "ordinary_context": row.ordinary_context.model_copy(
                    update={"num_modules": 4, "num_slots": 2}
                )
            }
        )
        for row in rows
    )
    smaller = summarize_scientific_collapse_v1(smaller_rows, modules=4, slots=2)
    result = scientific_collapse_criterion_v1(purpose="scientific", summary=smaller)
    assert result.outcome == "not_applicable_geometry"
    assert result.minimum_effective_modules == result.minimum_effective_slots == 4


def test_partial_scientific_summary_cannot_yield_source_gate() -> None:
    summary = summarize_scientific_collapse_v1(
        (_row(0, (0, 1, 2, 3), (0, 1, 2, 3)),), modules=8, slots=8
    )
    with pytest.raises(ValueError):
        scientific_collapse_criterion_v1(purpose="scientific", summary=summary)


@pytest.mark.parametrize(
    "field", ["seed", "checkpoint_sha256", "state_dim", "cycle_count", "example_id"]
)
def test_raw_aggregation_rejects_mixed_or_mislabelled_context(field: str) -> None:
    first, second = _row(0, (0,), (0,)), _row(1, (1,), (1,))
    value: Any = (
        "f" * 64 if field == "checkpoint_sha256" else "test-99" if field == "example_id" else 99
    )
    second = second.model_copy(
        update={"ordinary_context": second.ordinary_context.model_copy(update={field: value})}
    )
    with pytest.raises(ValueError):
        summarize_scientific_collapse_v1((first, second), modules=8, slots=8)


def test_full_scientific_not_met_remains_a_valid_report_not_a_measurement_error(
    scientific_payload: ScientificCollapsePayloadV1,
) -> None:
    rows = tuple(_row(i, (0, 1), (0, 1), cycles=3) for i in range(5000))
    groups = scientific_collapse_groups_v1(scientific_payload.members, rows, modules=8, slots=8)
    summary = groups[0].summary
    assert summary is not None
    raw = scientific_payload.model_dump()
    raw.update(
        rows=rows,
        groups=groups,
        source_20_4=scientific_collapse_criterion_v1(purpose="scientific", summary=summary),
    )
    result = ScientificCollapsePayloadV1.model_validate(raw)
    assert result.source_20_4.outcome == "not_met"
    assert result.scientific_acceptance == "not_assessed"


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "reordered",
        "duplicate-prompt",
        "wrong-task",
        "wrong-gate",
        "forged-entropy",
        "wrong-seed",
        "wrong-checkpoint",
        "wrong-budget",
    ],
)
def test_full_test_schema_rejects_partial_or_forged_reports(
    scientific_payload: ScientificCollapsePayloadV1, bad: str
) -> None:
    raw = scientific_payload.model_dump()
    if bad == "missing":
        raw["rows"] = raw["rows"][:-1]
    elif bad == "reordered":
        raw["rows"] = tuple(reversed(raw["rows"]))
    elif bad == "duplicate-prompt":
        raw["members"][1]["prompt_sha256"] = raw["members"][0]["prompt_sha256"]
    elif bad == "wrong-task":
        raw["members"][0]["task_type"] = TASKS[1]
    elif bad == "wrong-gate":
        raw["source_20_4"]["outcome"] = "not_met"
    elif bad == "forged-entropy":
        raw["groups"][0]["summary"]["effective_active_modules"] = 8.0
    elif bad == "wrong-seed":
        raw["run_seed"] = 1337
    elif bad == "wrong-checkpoint":
        raw["best_checkpoint_sha256"] = "f" * 64
    else:
        raw["trace_scalar_bound"] -= 1
    with pytest.raises(ValueError):
        ScientificCollapsePayloadV1.model_validate(raw)
