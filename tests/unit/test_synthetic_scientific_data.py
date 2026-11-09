"""Four-family logical data, independent oracles and bounded Parquet round trips."""

import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from iris.data.generator import generate_synthetic_dataset, generate_synthetic_split
from iris.data.parquet import decode_synthetic_parquet, encode_synthetic_parquet
from iris.data.schemas import (
    IrisExample,
    SyntheticDatasetPlan,
    SyntheticSplitPlan,
    parse_structured_response,
)


def _plan(count: int = 30) -> SyntheticDatasetPlan:
    return SyntheticDatasetPlan(
        splits=(
            SyntheticSplitPlan(split="train", first_seed=1000, count=count),
            SyntheticSplitPlan(split="validation", first_seed=100000, count=count),
            SyntheticSplitPlan(split="test", first_seed=200000, count=count),
        )
    )


@pytest.fixture(scope="module")
def rows() -> tuple[IrisExample, ...]:
    return tuple(generate_synthetic_dataset(_plan()))


def test_four_families_exact_distribution_and_disjoint_seeds(rows: tuple[IrisExample, ...]) -> None:
    assert len(rows) == 90
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row.split == split]
        assert Counter(row.task_type for row in selected) == {
            "split_spec_repair": 12,
            "delayed_goal": 6,
            "conflict_resolution": 6,
            "next_tool_prediction": 6,
        }
        assert {row.difficulty for row in selected} == {1, 2, 3}
    assert len({row.seed for row in rows}) == len(rows)
    assert len({row.problem_sha256 for row in rows}) == len(rows)
    assert all(not row.required_modules for row in rows)


def test_same_plan_exactly_reproduces_without_global_rng_mutation(
    rows: tuple[IrisExample, ...],
) -> None:
    before = random.getstate()
    assert tuple(generate_synthetic_dataset(_plan())) == rows
    assert random.getstate() == before
    assert tuple(generate_synthetic_split(_plan().splits[2])) == tuple(
        row for row in rows if row.split == "test"
    )


def test_full_spec_sizes_are_valid_plans_without_materializing() -> None:
    plan = SyntheticDatasetPlan(
        splits=(
            SyntheticSplitPlan(split="train", first_seed=0, count=50000),
            SyntheticSplitPlan(split="validation", first_seed=50000, count=5000),
            SyntheticSplitPlan(split="test", first_seed=55000, count=5000),
        )
    )
    assert sum(item.count for item in plan.splits) == 60000


@pytest.mark.parametrize(
    "changes",
    [
        {"count": True},
        {"count": 14},
        {"count": 16},
        {"count": 100005},
        {"first_seed": -1},
        {"first_seed": 2**63 - 1},
        {"first_seed": "1"},
        {"split": "dev"},
    ],
)
def test_invalid_split_plan(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SyntheticSplitPlan.model_validate(
            {"split": "train", "count": 30, "first_seed": 1000, **changes}
        )


def test_overlap_and_reordered_splits_fail() -> None:
    plan = _plan()
    with pytest.raises(ValueError):
        SyntheticDatasetPlan(splits=(plan.splits[1], plan.splits[0], plan.splits[2]))
    with pytest.raises(ValueError):
        SyntheticDatasetPlan(
            splits=(
                plan.splits[0],
                SyntheticSplitPlan(split="validation", first_seed=1001, count=30),
                plan.splits[2],
            )
        )


def test_duplicate_problem_fails_without_silent_relabel(rows: tuple[IrisExample, ...]) -> None:
    with (
        patch(
            "iris.data.generator.generate_synthetic_split", return_value=iter((rows[0], rows[0]))
        ),
        pytest.raises(ValueError, match="Duplicate"),
    ):
        tuple(generate_synthetic_dataset(_plan()))


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 0.42, 1.0])
def test_answer_json_number_confidence(confidence: float) -> None:
    raw = json.dumps(
        {
            "decision": "answer",
            "content": "job-1",
            "confidence": confidence,
            "next_action": "finish",
        }
    )
    assert parse_structured_response(raw).confidence == confidence


@pytest.mark.parametrize("bad", [True, False, -0.1, 1.1, "1", None, float("nan"), float("inf")])
def test_invalid_confidence(bad: Any) -> None:
    with pytest.raises(ValueError, match="Invalid Iris"):
        parse_structured_response(
            json.dumps(
                {
                    "decision": "answer",
                    "content": "private",
                    "confidence": bad,
                    "next_action": "finish",
                }
            )
        )


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        "null",
        "```json\n{}\n```",
        "prefix {}",
        "{}{}",
        '{"decision":"answer","decision":"patch","content":"x","confidence":1,"next_action":"finish"}',
        '{"decision":"answer","content":"x","confidence":1,"next_action":"finish","extra":1}',
        '{"decision":"answer","content":"\\u0000","confidence":1,"next_action":"finish"}',
    ],
)
def test_bad_answer_envelope_redacted(text: str) -> None:
    with pytest.raises(ValueError, match=r"^Invalid Iris structured response$"):
        parse_structured_response(text)


def test_independent_reference_oracles_and_code_ast_only(rows: tuple[IrisExample, ...]) -> None:
    for row in rows:
        response = parse_structured_response(row.expected_response)
        facts = row.hidden_facts
        assert row.distractor_count == (0, 4, 12)[row.difficulty - 1]
        if row.task_type == "split_spec_repair":
            for tag in ("GOAL", "CODE", "CONSTRAINTS", "TESTS", "DISTRACTORS"):
                assert f"<{tag}>" in row.prompt and f"</{tag}>" in row.prompt
            # Parse only. Neither reference Python nor test_code is executed on the host.
            syntax = ast.parse(response.content)
            assert isinstance(syntax.body[0], ast.FunctionDef)
            assert syntax.body[0].name == "solve"
            assert not any(
                isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(syntax)
            )
            test_syntax = ast.parse(cast(str, row.test_code))
            assert len(test_syntax.body) == len(row.code_cases)
            lower, upper = cast(int, facts["lower"]), cast(int, facts["upper"])
            divisor, residue = cast(int, facts["divisor"]), cast(int, facts["residue"])
            assert lower % divisor == upper % divisor == residue
            for case in row.code_cases:
                selected = []
                for value in case.inputs:
                    if value < lower or value > upper:
                        continue
                    if (value == lower and not facts["include_lower"]) or (
                        value == upper and not facts["include_upper"]
                    ):
                        continue
                    if value % divisor == residue:
                        selected.append(value)
                expected = (
                    len(selected)
                    if facts["operation"] == "count"
                    else sum(
                        value**2 if facts["operation"] == "squared_sum" else value
                        for value in selected
                    )
                )
                assert case.expected == expected
            assert response.next_action == "run_tests"
        elif row.task_type == "delayed_goal":
            records = json.loads(cast(str, facts["records"]))
            matching_jobs = [item for item in records if item[1] == facts["desired_color"]]
            ordering = sorted(
                matching_jobs,
                key=lambda item: (
                    -item[2] if facts["operation"] == "maximum" else item[2],
                    item[0],
                ),
            )
            assert response.content == ordering[0][0]
        elif row.task_type == "conflict_resolution":
            priority = json.loads(cast(str, facts["priorities"]))
            reports = json.loads(cast(str, facts["reports"]))
            best = max(reports, key=lambda item: (priority[item[0]], item[1]))
            actual = json.loads(response.content)
            assert actual["source"] == best[0] and actual["status"] == best[2]
            assert len({item[2] for item in reports}) > 1 and actual["conflict"] is True
        else:
            conditions = (
                (not facts["symbol_located"], "search_symbol"),
                (not facts["target_read"], "read_file"),
                (not facts["patch_applied"], "edit_file"),
                (not facts["tests_after_patch"], "run_tests"),
                (facts["tests_passed"], "finish"),
                (not facts["failure_inspected"], "inspect_error"),
                (True, "edit_file"),
            )
            expected_action = next(action for condition, action in conditions if condition)
            assert response.next_action == expected_action
            assert json.loads(response.content) == {
                "action": expected_action,
                "target": facts["target"],
            }


_BUDGET = {"max_logical_bytes": 8 * 1024**2, "max_file_bytes": 8 * 1024**2}


def test_actual_parquet_round_trip(tmp_path: Path, rows: tuple[IrisExample, ...]) -> None:
    content = encode_synthetic_parquet(rows, **_BUDGET)
    path = tmp_path / "scientific-candidate.parquet"
    path.write_bytes(content)
    assert content[:4] == content[-4:] == b"PAR1"
    decoded = decode_synthetic_parquet(path.read_bytes(), max_rows=90, **_BUDGET)
    assert decoded == rows
    assert encode_synthetic_parquet(rows, **_BUDGET) == content
    for original, restored in zip(rows, decoded, strict=True):
        for key, value in original.hidden_facts.items():
            assert type(restored.hidden_facts[key]) is type(value)
    table = pq.ParquetFile(path).read()
    assert table.num_rows == 90 and "hidden_facts" in table.column_names


@pytest.mark.parametrize("budget", ["max_logical_bytes", "max_file_bytes"])
def test_encoder_budget(rows: tuple[IrisExample, ...], budget: str) -> None:
    with pytest.raises(ValueError):
        encode_synthetic_parquet(rows, **{**_BUDGET, budget: 1})


@pytest.mark.parametrize("change", ["max_rows", "max_logical_bytes", "max_file_bytes"])
def test_decoder_budget(rows: tuple[IrisExample, ...], change: str) -> None:
    content = encode_synthetic_parquet(rows, **_BUDGET)
    with pytest.raises(ValueError):
        decode_synthetic_parquet(content, **{"max_rows": 90, **_BUDGET, change: 1})


def test_logical_checksum_and_schema_reject_modified_parquet(rows: tuple[IrisExample, ...]) -> None:
    content = encode_synthetic_parquet(rows, **_BUDGET)
    table = pq.ParquetFile(pa.BufferReader(content)).read()
    metadata = dict(table.schema.metadata)
    metadata[b"iris.logical_sha256"] = b"0" * 64
    altered = table.replace_schema_metadata(metadata)
    for candidate in (altered, table.drop(["seed"])):
        sink = pa.BufferOutputStream()
        pq.write_table(candidate, sink)
        with pytest.raises(ValueError):
            decode_synthetic_parquet(sink.getvalue().to_pybytes(), max_rows=90, **_BUDGET)


def test_corrupt_row_and_duplicate_encoding_rejected(rows: tuple[IrisExample, ...]) -> None:
    with pytest.raises(ValueError):
        encode_synthetic_parquet((rows[0], rows[0]), **_BUDGET)
    forged = rows[0].model_copy(
        update={"hidden_facts": {**rows[0].hidden_facts, "operation": "invented"}}
    )
    with pytest.raises(ValueError):
        encode_synthetic_parquet((forged,), **_BUDGET)
