"""Token-weighted sums, exact group partitions, and strict NLL companion schemas."""

from typing import Any

import pytest
from pydantic import ValidationError

from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_intervention_nll_v1 import (
    ScientificInterventionNllPayloadV1,
    ScientificInterventionNllRowV1,
    ScientificPairedNllV1,
    scientific_nll_groups_v1,
    scientific_paired_nll_v1,
)


def row(**changes: Any) -> ScientificInterventionNllRowV1:
    return ScientificInterventionNllRowV1.model_validate(
        {
            "example_id": "test-1",
            "example_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "target_sha256": "c" * 64,
            "bound_plan_sha256": "d" * 64,
            "task_type": "delayed_goal",
            "difficulty": 2,
            "target_token_count": 2,
            "control_nll_sum": 2.0,
            "intervened_nll_sum": 4.0,
            **changes,
        }
    )


def test_token_weighted_not_unweighted_row_average_and_exact_twenty_groups() -> None:
    rows = (
        row(),
        row(
            example_id="test-2",
            target_token_count=8,
            control_nll_sum=32.0,
            intervened_nll_sum=16.0,
            task_type="conflict_resolution",
            difficulty=3,
        ),
    )
    totals = scientific_paired_nll_v1(rows)
    assert totals.examples == 2 and totals.target_token_count == 10
    assert totals.control_mean_nll == 3.4 and totals.intervened_mean_nll == 2.0
    assert totals.delta_mean_nll == 2.0 - 3.4
    groups = scientific_nll_groups_v1(rows)
    assert len(groups) == 20 and groups[0].nll == totals
    assert len({(g.task_type, g.difficulty) for g in groups}) == 20
    assert sum(g.nll.target_token_count for g in groups[1:5]) == 10
    assert sum(g.nll.target_token_count for g in groups[5:8]) == 10
    assert sum(g.nll.target_token_count for g in groups[8:]) == 10
    empty = scientific_paired_nll_v1(())
    assert empty.examples == empty.target_token_count == 0
    assert empty.control_mean_nll is empty.intervened_mean_nll is empty.delta_mean_nll is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_token_count", True),
        ("target_token_count", 0),
        ("control_nll_sum", -1.0),
        ("control_nll_sum", float("nan")),
        ("intervened_nll_sum", float("inf")),
        ("target_sha256", "A" * 64),
        ("difficulty", 4),
        ("example_id", "validation-1"),
    ],
)
def test_invalid_row_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        row(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("examples", 0),
        ("target_token_count", 0),
        ("control_mean_nll", 17.0),
        ("intervened_mean_nll", None),
        ("delta_mean_nll", -1.0),
    ],
)
def test_aggregate_cannot_relabel_count_rates_or_delta(field: str, value: Any) -> None:
    raw = scientific_paired_nll_v1((row(),)).model_dump(mode="python")
    with pytest.raises(ValidationError):
        ScientificPairedNllV1.model_validate({**raw, field: value})


def payload() -> ScientificInterventionNllPayloadV1:
    rows = (row(),)
    identities = {
        name: "a" * 64
        for name in (
            "intervention_sha256",
            "baseline_generation_sha256",
            "plan_sha256",
            "execution_sha256",
            "terminal_event_sha256",
            "best_checkpoint_sha256",
            "teacher_configuration_sha256",
            "teacher_graph_sha256",
            "teacher_sidecar_sha256",
            "backbone_runtime_sha256",
            "test_parquet_sha256",
            "test_corpus_sha256",
        )
    }
    return ScientificInterventionNllPayloadV1.model_validate(
        {
            **identities,
            "purpose": "smoke",
            "selected_step": 1,
            "rows": rows,
            "groups": scientific_nll_groups_v1(rows),
        }
    )


@pytest.mark.parametrize("mutation", ["duplicate", "groups", "scientific", "unknown"])
def test_report_requires_unique_rows_exact_partitions_and_complete_scientific_split(
    mutation: str,
) -> None:
    raw = payload().model_dump(mode="python")
    if mutation == "duplicate":
        raw["rows"] *= 2
    elif mutation == "groups":
        raw["groups"] = tuple(reversed(raw["groups"]))
    elif mutation == "scientific":
        raw["purpose"] = "scientific"
    else:
        raw["generated_answer_quality"] = 1.0
    with pytest.raises(ValidationError):
        ScientificInterventionNllPayloadV1.model_validate(raw)


def test_full_scientific_shape_requires_and_accepts_exact_source_family_counts() -> None:
    rows = tuple(
        row(
            example_id=f"test-{index}",
            task_type=TASKS[0] if index < 2000 else TASKS[1 + (index - 2000) // 1000],
        )
        for index in range(5000)
    )
    raw = payload().model_dump(mode="python")
    raw.update(purpose="scientific", rows=rows, groups=scientific_nll_groups_v1(rows))
    full = ScientificInterventionNllPayloadV1.model_validate(raw)
    assert full.groups[0].nll.examples == 5000
    assert full.groups[0].nll.target_token_count == 10000
    # A rehashed/complete-looking report with one missing row is still incomplete.
    incomplete = rows[:-1]
    raw.update(rows=incomplete, groups=scientific_nll_groups_v1(incomplete))
    with pytest.raises(ValidationError):
        ScientificInterventionNllPayloadV1.model_validate(raw)
