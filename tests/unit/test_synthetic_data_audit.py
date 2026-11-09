"""Development audit boundaries and deliberate quality-defect witnesses."""

from __future__ import annotations

import json
import random
import traceback
from hashlib import sha256
from typing import Any
from unittest.mock import patch

import pytest

from iris.data import audit
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticDatasetPlan, SyntheticSplitPlan
from iris.utils.hashing import canonical_json_bytes


def _plan() -> SyntheticDatasetPlan:
    return SyntheticDatasetPlan(
        splits=(
            SyntheticSplitPlan(split="train", first_seed=0, count=15),
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            SyntheticSplitPlan(split="test", first_seed=2000000, count=15),
        )
    )


def _run(plan: SyntheticDatasetPlan | None = None) -> dict[str, Any]:
    return audit.audit_synthetic_plan(plan or _plan(), max_rows=45, max_logical_bytes=1048576)


def _replace_facts(row: IrisExample, facts: dict[str, Any]) -> IrisExample:
    return row.model_copy(
        update={
            "hidden_facts": facts,
            "problem_sha256": sha256(
                canonical_json_bytes({"task": row.task_type, "facts": facts})
            ).hexdigest(),
        }
    )


def test_streaming_audit_is_deterministic_nonexecuting_and_content_redacted() -> None:
    rng_state = random.getstate()
    report = _run()
    assert report == _run() and random.getstate() == rng_state
    assert report["scientific_release"] is False
    assert report["python_sources_executed"] is False
    assert report["tokenizer_context_fit_verified"] is False
    assert 0 < report["logical_bytes"] < 1048576
    for split in report["splits"].values():
        assert split["families"] == {
            "split_spec_repair": 6,
            "delayed_goal": 3,
            "conflict_resolution": 3,
            "next_tool_prediction": 3,
        }
        assert split["counts"]["rows"] == 15
        assert split["counts"].get("reference_mismatches", 0) == 0
        assert split["counts"].get("repair_case_label_mismatches", 0) == 0
        assert split["counts"]["conflict_unique_minority_source_and_status_success"] == 3
        assert sum(split["tool_actions"].values()) == 3
    encoded = json.dumps(report)
    for row in generate_synthetic_split(_plan().splits[0]):
        assert row.prompt not in encoded and row.expected_response not in encoded


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_rows", True),
        ("max_rows", 0),
        ("max_rows", 44),
        ("max_rows", 100001),
        ("max_logical_bytes", True),
        ("max_logical_bytes", 0),
        ("max_logical_bytes", 512 * 1024**2 + 1),
    ],
)
def test_invalid_budgets_fail_before_generation(key: str, value: Any) -> None:
    args = {"max_rows": 45, "max_logical_bytes": 1048576, key: value}
    with patch.object(audit, "generate_synthetic_split") as generate:
        with pytest.raises(ValueError, match="could not be completed"):
            audit.audit_synthetic_plan(_plan(), **args)
        generate.assert_not_called()


def test_logical_byte_budget_is_enforced_during_iteration() -> None:
    with pytest.raises(ValueError, match="could not be completed"):
        audit.audit_synthetic_plan(_plan(), max_rows=45, max_logical_bytes=1)


@pytest.mark.parametrize("variant", ["short", "long", "order", "corrupt"])
def test_malformed_generator_output_cannot_produce_a_partial_success(variant: str) -> None:
    rows = list(generate_synthetic_split(_plan().splits[0]))
    if variant == "short":
        rows.pop()
    elif variant == "long":
        rows.append(rows[-1])
    elif variant == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0].hidden_facts["operation"] = "PRIVATE_AUDIT_MARKER"
    with (
        patch.object(audit, "generate_synthetic_split", return_value=iter(rows)),
        pytest.raises(ValueError) as caught,
    ):
        _run()
    assert "PRIVATE_AUDIT_MARKER" not in "".join(traceback.format_exception(caught.value))


def test_exact_and_semantically_equivalent_repair_repetitions_are_distinguished() -> None:
    plan = _plan()
    rows = {split.split: list(generate_synthetic_split(split)) for split in plan.splits}
    original = rows["train"][0]
    equivalent = dict(original.hidden_facts)
    lower, divisor = equivalent["lower"], equivalent["divisor"]
    assert type(lower) is int and type(divisor) is int
    equivalent["lower"] = lower + (-divisor if equivalent["include_lower"] else divisor)
    equivalent["include_lower"] = not equivalent["include_lower"]
    # Shift an endpoint by one lattice step and reverse its inclusion: the full
    # integer-list function stays identical although the fact identity changes.
    sibling = _replace_facts(original, equivalent).model_copy(update={"id": "train-1", "seed": 1})
    rows["train"][1] = sibling
    rows["validation"][0] = original.model_copy(
        update={
            "id": "validation-1000000",
            "seed": 1000000,
            "split": "validation",
        }
    )
    with patch.object(
        audit, "generate_synthetic_split", side_effect=lambda split: iter(rows[split.split])
    ):
        report = _run(plan)
    train = report["splits"]["train"]["counts"]
    validation = report["splits"]["validation"]["counts"]
    assert train["problem_repeated_within_split"] == 0
    assert train["repair_function_repeated_within_split"] == 1
    assert validation["problem_repeated_across_splits"] == 1
    assert validation["repair_function_repeated_across_splits"] == 1


@pytest.mark.parametrize(
    "family", ["split_spec_repair", "delayed_goal", "conflict_resolution", "next_tool_prediction"]
)
def test_reference_or_case_corruption_is_detected_independently_of_self_scoring(
    family: str,
) -> None:
    def generate(split: SyntheticSplitPlan) -> Any:
        for row in generate_synthetic_split(split):
            if row.task_type == family:
                if row.executable:
                    bad_case = row.code_cases[0].model_copy(
                        update={"expected": row.code_cases[0].expected + 1}
                    )
                    row = row.model_copy(update={"code_cases": (bad_case, *row.code_cases[1:])})
                else:
                    response = json.loads(row.expected_response)
                    if family == "delayed_goal":
                        response["content"] = "wrong-job"
                    else:
                        content = json.loads(response["content"])
                        content["source" if family == "conflict_resolution" else "target"] = "wrong"
                        response["content"] = json.dumps(content)
                    row = row.model_copy(update={"expected_response": json.dumps(response)})
            yield row

    with patch.object(audit, "generate_synthetic_split", generate):
        report = _run()
    metric = (
        "repair_case_label_mismatches" if family == "split_spec_repair" else "reference_mismatches"
    )
    for split in report["splits"].values():
        assert split["counts"][metric] == (6 if family == "split_spec_repair" else 3)


def test_coherence_is_separate_from_following_first_matching_tool_rule() -> None:
    def generate(split: SyntheticSplitPlan) -> Any:
        for row in generate_synthetic_split(split):
            if row.task_type == "next_tool_prediction":
                facts = dict(row.hidden_facts)
                for name in (
                    "symbol_located",
                    "target_read",
                    "patch_applied",
                    "tests_after_patch",
                    "tests_passed",
                    "failure_inspected",
                ):
                    facts[name] = name == "target_read"
                facts["operation"] = "search_symbol"
                response = json.loads(row.expected_response)
                response.update(
                    next_action="search_symbol",
                    content=json.dumps(
                        {
                            "action": "search_symbol",
                            "target": facts["target"],
                        }
                    ),
                )
                row = _replace_facts(row, facts).model_copy(
                    update={"expected_response": json.dumps(response)}
                )
            yield row

    with patch.object(audit, "generate_synthetic_split", generate):
        report = _run()
    for split in report["splits"].values():
        assert split["counts"]["tool_incoherent_workflow_states"] == 3
        assert split["counts"].get("reference_mismatches", 0) == 0


def test_bad_plan_is_revalidated_before_generation() -> None:
    plan = _plan()
    bad = plan.model_copy(update={"splits": (plan.splits[0], plan.splits[0], plan.splits[2])})
    with patch.object(audit, "generate_synthetic_split") as generate:
        with pytest.raises(ValueError):
            _run(bad)
        generate.assert_not_called()
