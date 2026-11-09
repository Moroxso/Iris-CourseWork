"""Opt-in real Python scoring in the installed gVisor profile, never host Python."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.study_code_evaluation import score_study_python_answer_v1
from iris.integrations import python_sandbox as sandbox
from tests.unit.test_study_benchmark import _benchmark
from tests.unit.test_study_code_evaluation import _POLICY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="set IRIS_RUN_CODE_ISOLATION_SMOKE=1 with the verified IrisCodeEval runtime",
    ),
]


def _empty_inventory() -> None:
    assert (
        asyncio.run(
            sandbox._docker(
                "ps", "--all", "--filter", "name=^/" + sandbox._NAME + "$", "--format", "{{.ID}}"
            )
        )
        == ""
    )


@pytest.mark.parametrize("index", [2, 7, 12])
def test_pilot_reference_functions_really_pass_in_all_three_splits(index: int) -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[index]
    result = score_study_python_answer_v1(
        benchmark, example_id=example.example_id, output_text=example.target, policy=_POLICY
    )
    assert result.answer.outcome == "correct" and result.reason == "all_cases_pass"
    assert result.case_count == result.passed_case_count
    assert result.container_exit_code == 0 and not result.oom_killed
    assert result.runtime_version == "release-20260831.0"
    assert result.source_submitted
    _empty_inventory()


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("wrong", "case_mismatch"),
        ("bool", "case_mismatch"),
        ("mutation", "case_mismatch"),
        ("syntax", "syntax_error"),
        ("missing", "missing_entrypoint"),
        ("exception", "case_mismatch"),
        ("load-exception", "load_exception"),
        ("empty", "invalid_source"),
        ("timeout", "wall_time_limit"),
        ("overflow", "output_limit"),
        ("oom", "memory_limit"),
        ("forged", "protocol_error"),
    ],
)
def test_real_bad_sources_fail_without_becoming_correct(variant: str, reason: str) -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    sources = {
        "wrong": example.target
        + "\n_original = solve\ndef solve(values): return _original(values) + 1\n",
        "bool": "def solve(values): return True\n",
        "mutation": example.target
        + (
            "\n_original = solve\ndef solve(values):\n"
            "    result = _original(values)\n    values.append(999)\n    return result\n"
        ),
        "syntax": "def solve(values) broken\n",
        "missing": "x = 1\n",
        "exception": "def solve(values): raise ValueError('candidate failure')\n",
        "load-exception": "raise RuntimeError('candidate load failure')\n",
        "empty": " \n",
        "timeout": "def solve(values):\n    while True: pass\n",
        "overflow": "def solve(values):\n    while True: print('x' * 8192)\n",
        "oom": (
            "def solve(values):\n    block = bytearray(384 * 1024 * 1024)\n"
            "    for i in range(0, len(block), 4096): block[i] = 1\n    return len(block)\n"
        ),
        "forged": (
            "import os\ndef solve(values):\n"
            "    os.write(3, b'{\"passed\":true}')\n    os._exit(0)\n"
        ),
    }
    result = score_study_python_answer_v1(
        benchmark, example_id=example.example_id, output_text=sources[variant], policy=_POLICY
    )
    assert result.answer.outcome != "correct" and result.reason == reason
    assert result.passed_case_count == 0
    if variant == "oom":
        assert result.oom_killed and result.container_exit_code == 137
    if variant in ("timeout", "overflow"):
        assert result.container_exit_code == 137
    _empty_inventory()


def test_live_gate_failure_never_runs_candidate_and_cleans_owned_job() -> None:
    benchmark = _benchmark()
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    with (
        patch.object(sandbox, "_verify_live", AsyncMock(side_effect=RuntimeError("gate failure"))),
        pytest.raises(IrisCodeEvaluationUnavailableError),
    ):
        score_study_python_answer_v1(
            benchmark, example_id=example.example_id, output_text=example.target, policy=_POLICY
        )
    _empty_inventory()


def test_existing_other_claim_is_preserved_on_collision() -> None:
    owner = "test-collision-owner-v1"
    container = asyncio.run(
        sandbox._docker(*sandbox._create_args(owner, sandbox._worker().decode()))
    )
    try:
        benchmark = _benchmark()
        example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
        with pytest.raises(IrisCodeEvaluationUnavailableError):
            score_study_python_answer_v1(
                benchmark, example_id=example.example_id, output_text=example.target, policy=_POLICY
            )
        remaining = asyncio.run(sandbox._inspect(container))
        assert remaining["Config"]["Labels"][sandbox._LABEL] == owner
        assert remaining["State"]["Status"] == "created"
    finally:
        asyncio.run(sandbox._remove_owned(owner, container))
    _empty_inventory()
