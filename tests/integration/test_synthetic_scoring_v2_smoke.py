"""Real source-profile scorer controls, not learned-model quality or acceptance."""

import os
import sys

import pytest

from iris.application.synthetic_scoring_v2 import score_synthetic_answer_v2
from iris.data.schemas import parse_structured_response
from tests.unit.test_synthetic_scoring import _answer, _rows
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="requires explicit real IrisCodeEval source-profile gate",
    ),
]


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("empty_environment", "all_cases_pass"),
        ("timeout", "wall_time_limit"),
        ("output", "output_limit"),
    ],
)
def test_real_source_profile_scorer_resource_controls(kind: str, reason: str) -> None:
    row = _rows()[0]
    source = parse_structured_response(row.expected_response).content
    source += {
        "empty_environment": (
            "\nimport os\nassert dict(os.environ) == {}\n"
            "with open('/proc/self/environ', 'rb') as f: assert f.read() == b''\n"
        ),
        "timeout": "\nwhile True: pass\n",
        # A single huge unbuffered write can legally be short (observed64KiB).
        # Keep producing actual diagnostic bytes until the controller caps them.
        "output": "\nimport os\nwhile True: os.write(2, b'x' * 8192)\n",
    }[kind]
    result = score_synthetic_answer_v2(
        row, output_text=_answer(row, content=source), python_policy=_POLICY
    )
    assert result.reason == reason
    assert result.outcome == ("correct" if kind == "empty_environment" else "incorrect")
    assert result.execution is not None
    assert result.execution.policy == _POLICY
    assert result.passed_case_count == (result.case_count if kind == "empty_environment" else 0)
