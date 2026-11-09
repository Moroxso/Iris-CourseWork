"""Opt-in engineering checks of structured repair answers in real gVisor."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from iris.application.synthetic_scoring import score_synthetic_answer_v1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import (
    IrisExample,
    SyntheticSplitPlan,
    SyntheticTemplate,
    parse_structured_response,
)
from iris.integrations import python_sandbox as sandbox
from tests.unit.test_synthetic_scoring import _POLICY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="set IRIS_RUN_CODE_ISOLATION_SMOKE=1 with the verified IrisCodeEval runtime",
    ),
]


def _row(
    operation: str, difficulty: int, template_id: SyntheticTemplate = "iris-synthetic-01-v1"
) -> IrisExample:
    # Fixed test-fixture seeds, not scientific train/validation/test seed selection.
    return next(
        row
        for row in generate_synthetic_split(
            SyntheticSplitPlan(split="validation", first_seed=1000, count=150),
            template_id=template_id,
        )
        if row.executable
        and row.hidden_facts["operation"] == operation
        and row.difficulty == difficulty
    )


def _empty_inventory() -> None:
    assert (
        asyncio.run(
            sandbox._docker(
                "ps", "--all", "--filter", "name=^/" + sandbox._NAME + "$", "--format", "{{.ID}}"
            )
        )
        == ""
    )


@pytest.mark.parametrize("operation", ["sum", "count", "squared_sum"])
@pytest.mark.parametrize("difficulty", [1, 2, 3])
@pytest.mark.parametrize("template_id", ["iris-synthetic-01-v1", "iris-synthetic-01-v2"])
def test_repair_references_really_pass(
    operation: str, difficulty: int, template_id: SyntheticTemplate
) -> None:
    row = _row(operation, difficulty, template_id)
    score = score_synthetic_answer_v1(row, output_text=row.expected_response, python_policy=_POLICY)
    assert score.outcome == "correct" and score.reason == "all_cases_pass"
    assert score.case_count == score.passed_case_count == len(row.code_cases)
    assert score.execution is not None and score.execution.container_exit_code == 0
    assert not score.execution.oom_killed
    assert score.execution.runtime_version == "release-20260831.0"
    _empty_inventory()


@pytest.mark.parametrize("template_id", ["iris-synthetic-01-v1", "iris-synthetic-01-v2"])
def test_functionally_equivalent_loop_does_not_need_reference_source_equality(
    template_id: SyntheticTemplate,
) -> None:
    row = _row("squared_sum", 3, template_id)
    facts = row.hidden_facts
    lower_op = ">=" if facts["include_lower"] else ">"
    upper_op = "<=" if facts["include_upper"] else "<"
    source = (
        "def solve(values):\n    total = 0\n    for item in values:\n"
        f"        if item {lower_op} {facts['lower']} and item {upper_op} {facts['upper']}:\n"
        f"            if item % {facts['divisor']} == {facts['residue']}:\n"
        "                total += item ** 2\n    return total\n"
    )
    assert source != parse_structured_response(row.expected_response).content
    response = json.loads(row.expected_response)
    response.update(content=source, confidence=0.23)
    score = score_synthetic_answer_v1(row, output_text=json.dumps(response), python_policy=_POLICY)
    assert score.outcome == "correct" and score.case_count == score.passed_case_count
    _empty_inventory()


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("wrong", "case_mismatch"),
        ("mutation", "case_mismatch"),
        ("bool", "case_mismatch"),
        ("syntax", "syntax_error"),
        ("timeout", "wall_time_limit"),
        ("forged", "protocol_error"),
    ],
)
@pytest.mark.parametrize("template_id", ["iris-synthetic-01-v1", "iris-synthetic-01-v2"])
def test_real_incorrect_repairs_are_not_accepted(
    variant: str, reason: str, template_id: SyntheticTemplate
) -> None:
    row = _row("sum", 1, template_id)
    reference = parse_structured_response(row.expected_response).content
    source = {
        "wrong": reference
        + "\n_original = solve\ndef solve(values): return _original(values) + 1\n",
        "mutation": reference
        + (
            "\n_original = solve\ndef solve(values):\n"
            "    result = _original(values)\n    values.append(999)\n    return result\n"
        ),
        "bool": "def solve(values): return True\n",
        "syntax": "def solve(values) broken\n",
        "timeout": "def solve(values):\n    while True: pass\n",
        "forged": (
            "import os\ndef solve(values):\n"
            "    os.write(3, b'{\"passed\":true}')\n    os._exit(0)\n"
        ),
    }[variant]
    response = json.loads(row.expected_response)
    response["content"] = source
    score = score_synthetic_answer_v1(row, output_text=json.dumps(response), python_policy=_POLICY)
    assert score.outcome != "correct" and score.reason == reason
    assert score.passed_case_count == 0
    assert score.execution is not None
    _empty_inventory()


@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_v2_endpoint_inclusion_changes_are_caught_by_real_cases(boundary: str) -> None:
    row = _row("sum", 3, "iris-synthetic-01-v2")
    facts = row.hidden_facts
    source = parse_structured_response(row.expected_response).content
    old = "<=" if facts[f"include_{boundary}"] else "<"
    new = "<" if old == "<=" else "<="
    old_part = f"{facts['lower']} {old} x" if boundary == "lower" else f"x {old} {facts['upper']}"
    new_part = f"{facts['lower']} {new} x" if boundary == "lower" else f"x {new} {facts['upper']}"
    changed = source.replace(old_part, new_part, 1)
    assert changed != source
    response = json.loads(row.expected_response)
    response["content"] = changed
    score = score_synthetic_answer_v1(row, output_text=json.dumps(response), python_policy=_POLICY)
    assert score.outcome == "incorrect" and score.reason == "case_mismatch"
    assert score.passed_case_count < score.case_count
    _empty_inventory()
