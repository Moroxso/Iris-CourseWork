"""Controlled v2 construction and explicit-profile compatibility contracts."""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from typing import Any
from unittest.mock import patch

import pytest

from iris.application.synthetic_scoring import score_synthetic_answer_v1
from iris.data.audit import audit_synthetic_plan
from iris.data.generator import generate_synthetic_dataset, generate_synthetic_split
from iris.data.generator_v2 import V2_SEED_LIMIT, generate_synthetic_split_v2
from iris.data.parquet import decode_synthetic_parquet, encode_synthetic_parquet
from iris.data.schemas import (
    IrisExample,
    SyntheticDatasetPlan,
    SyntheticSplitPlan,
    SyntheticTemplate,
    parse_structured_response,
)

_V2: SyntheticTemplate = "iris-synthetic-01-v2"


def _plan(count: int = 90) -> SyntheticDatasetPlan:
    return SyntheticDatasetPlan(
        splits=(
            SyntheticSplitPlan(split="train", first_seed=0, count=count),
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=count),
            SyntheticSplitPlan(split="test", first_seed=2000000, count=count),
        )
    )


@pytest.fixture(scope="module")
def rows() -> tuple[IrisExample, ...]:
    return tuple(generate_synthetic_dataset(_plan(), template_id=_V2))


def test_v2_is_explicit_and_deterministic_without_global_rng_mutation(
    rows: tuple[IrisExample, ...],
) -> None:
    before = random.getstate()
    assert tuple(generate_synthetic_dataset(_plan(), template_id=_V2)) == rows
    assert random.getstate() == before
    assert all(row.template_id == _V2 for row in rows)
    legacy = tuple(generate_synthetic_dataset(_plan()))
    explicit = tuple(generate_synthetic_dataset(_plan(), template_id="iris-synthetic-01-v1"))
    assert legacy == explicit and legacy != rows
    assert all(row.template_id == "iris-synthetic-01-v1" for row in legacy)


def test_family_and_action_balance_at_every_difficulty(rows: tuple[IrisExample, ...]) -> None:
    actions = {"read_file", "search_symbol", "edit_file", "run_tests", "inspect_error", "finish"}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row.split == split]
        assert Counter(row.task_type for row in selected) == {
            "split_spec_repair": 36,
            "delayed_goal": 18,
            "conflict_resolution": 18,
            "next_tool_prediction": 18,
        }
        for difficulty in (1, 2, 3):
            tool_rows = [
                row
                for row in selected
                if row.task_type == "next_tool_prediction" and row.difficulty == difficulty
            ]
            assert Counter(
                parse_structured_response(row.expected_response).next_action for row in tool_rows
            ) == dict.fromkeys(actions, 1)


def test_tool_states_are_reachable_and_both_edit_states_exist(
    rows: tuple[IrisExample, ...],
) -> None:
    edit_states = set()
    for row in rows:
        if row.task_type != "next_tool_prediction":
            continue
        facts = row.hidden_facts
        state = tuple(
            facts[key]
            for key in (
                "symbol_located",
                "target_read",
                "patch_applied",
                "tests_after_patch",
                "tests_passed",
                "failure_inspected",
            )
        )
        known, read, edited, tested, passed, inspected = state
        assert not read or known
        assert not edited or read
        assert not tested or edited
        assert not passed or tested
        assert not inspected or (tested and not passed)
        if facts["operation"] == "edit_file":
            edit_states.add(state)
    assert edit_states == {
        (True, True, False, False, False, False),
        (True, True, True, True, False, True),
    }


def test_delayed_goal_needs_both_color_and_priority_direction(
    rows: tuple[IrisExample, ...],
) -> None:
    for row in rows:
        if row.task_type != "delayed_goal":
            continue
        facts = row.hidden_facts
        records = json.loads(str(facts["records"]))
        selected = [item for item in records if item[1] == facts["desired_color"]]
        assert len(selected) >= 2
        minimum, maximum = (
            min(selected, key=lambda item: item[2]),
            max(selected, key=lambda item: item[2]),
        )
        assert minimum[0] != maximum[0]
        high = facts["operation"] == "maximum"
        expected = maximum if high else minimum
        color_agnostic = (max if high else min)(records, key=lambda item: item[2])
        assert expected[0] != color_agnostic[0]
        assert parse_structured_response(row.expected_response).content == expected[0]


def test_conflict_has_no_minority_and_each_source_needs_recency(
    rows: tuple[IrisExample, ...],
) -> None:
    for row in rows:
        if row.task_type != "conflict_resolution":
            continue
        reports = json.loads(str(row.hidden_facts["reports"]))
        assert sorted(Counter(item[2] for item in reports).values()) == [3, 3]
        for source in {item[0] for item in reports}:
            assert len({item[2] for item in reports if item[0] == source}) == 2


def test_repair_progression_recovers_the_seed_and_has_nonzero_edge_witnesses(
    rows: tuple[IrisExample, ...],
) -> None:
    recovered = set()
    for row in rows:
        if not row.executable:
            continue
        facts = row.hidden_facts
        lower, upper, divisor = (facts[key] for key in ("lower", "upper", "divisor"))
        assert type(lower) is int and type(upper) is int and type(divisor) is int
        first = lower if facts["include_lower"] else lower + divisor
        last = upper if facts["include_upper"] else upper - divisor
        point_count = (last - first) // divisor + 1
        assert 33 <= first <= 8224 and 2 <= point_count <= 8193 and lower > 0
        rank = (first - 33) + (point_count - 2) * 8192
        seed = ((rank - 1729) * pow(48271, -1, V2_SEED_LIMIT)) % V2_SEED_LIMIT
        assert seed == row.seed and seed not in recovered
        recovered.add(seed)
        cases = {case.inputs: case.expected for case in row.code_cases}
        assert cases[()] == 0
        for endpoint, included in (
            (lower, facts["include_lower"]),
            (upper, facts["include_upper"]),
        ):
            assert (cases[(endpoint,)] != 0) is included
        assert cases[(first,)] > 0
        ast.parse(parse_structured_response(row.expected_response).content)


def test_all_noncode_v2_references_use_existing_semantic_scoring(
    rows: tuple[IrisExample, ...],
) -> None:
    for row in rows:
        if not row.executable:
            score = score_synthetic_answer_v1(
                row, output_text=row.expected_response, python_policy=None
            )
            assert score.outcome == "correct" and score.execution is None


def test_v2_round_trips_in_existing_typed_parquet(rows: tuple[IrisExample, ...]) -> None:
    budgets = {"max_logical_bytes": 4 * 1024**2, "max_file_bytes": 4 * 1024**2}
    encoded = encode_synthetic_parquet(rows, **budgets)
    assert decode_synthetic_parquet(encoded, max_rows=len(rows), **budgets) == rows


def test_small_v2_audit_rejects_all_measured_v1_shortcuts() -> None:
    report = audit_synthetic_plan(
        _plan(), template_id=_V2, max_rows=270, max_logical_bytes=4 * 1024**2
    )
    assert report["template_id"] == _V2 and not report["scientific_release"]
    for split in report["splits"].values():
        for key in (
            "reference_mismatches",
            "repair_case_label_mismatches",
            "problem_repeated_within_split",
            "problem_repeated_across_splits",
            "repair_function_repeated_within_split",
            "repair_function_repeated_across_splits",
            "conflict_unique_minority_source_and_status_success",
            "conflict_unique_minority_status_success",
            "delayed_priority_direction_irrelevant",
            "delayed_single_qualifying_job",
            "delayed_color_agnostic_success",
            "tool_incoherent_workflow_states",
        ):
            assert split["counts"].get(key, 0) == 0, key
        assert all(
            set(actions.values()) == {1} for actions in split["tool_actions_by_difficulty"].values()
        )


@pytest.mark.parametrize("extra", [1, 15, 100000])
def test_v2_out_of_domain_split_rejected(extra: int) -> None:
    plan = SyntheticSplitPlan(split="train", first_seed=V2_SEED_LIMIT - 15 + extra, count=15)
    with pytest.raises(ValueError, match="v2 seeds"):
        next(generate_synthetic_split(plan, template_id=_V2))


def test_last_valid_seed_interval_is_supported() -> None:
    plan = SyntheticSplitPlan(split="test", first_seed=V2_SEED_LIMIT - 15, count=15)
    result = tuple(generate_synthetic_split_v2(plan))
    assert len(result) == 15 and result[-1].seed == V2_SEED_LIMIT - 1


def test_bad_later_split_fails_before_yielding_partial_dataset() -> None:
    plan = _plan(15)
    bad = SyntheticDatasetPlan(
        splits=(
            plan.splits[0],
            plan.splits[1],
            SyntheticSplitPlan(split="test", first_seed=V2_SEED_LIMIT, count=15),
        )
    )
    with patch("iris.data.generator.generate_synthetic_split") as generate:
        with pytest.raises(ValueError, match="v2 seeds"):
            next(generate_synthetic_dataset(bad, template_id=_V2))
        generate.assert_not_called()


@pytest.mark.parametrize("bad", [None, True, "v3"])
def test_unknown_template_is_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        next(generate_synthetic_split(_plan().splits[0], template_id=bad))
    with pytest.raises(ValueError, match="Unknown"):
        next(generate_synthetic_dataset(_plan(), template_id=bad))
    with pytest.raises(ValueError):
        audit_synthetic_plan(_plan(), template_id=bad, max_rows=270, max_logical_bytes=1048576)
