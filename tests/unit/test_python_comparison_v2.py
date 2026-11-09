"""Exact version/worker binding and resource precedence for source-profile scoring."""

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iris.application import synthetic_scoring_v2 as scoring
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisSyntheticScoringError
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_comparison_v2 import (
    assess_integer_list_observation_v2,
    check_python_observation_v2,
)
from iris.integrations.python_sandbox import PythonSandboxError
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_synthetic_scoring import _POLICY as LEGACY_POLICY
from tests.unit.test_synthetic_scoring import _observation as legacy_observation
from tests.unit.test_synthetic_scoring import _observations, _rows
from tests.unit.test_synthetic_scoring_v2 import _POLICY, _observation


@pytest.mark.parametrize(
    "kind",
    ["legacy", "legacy_policy", "legacy_domain", "worker", "forged_policy", "process", "type"],
)
def test_wrong_backend_identity_is_infrastructure_failure(kind: str) -> None:
    row = _rows()[0]

    async def execute(*, request_bytes: bytes, policy: PythonSandboxPolicyV2) -> Any:
        observation: Any = _observation(_observations(row), request_bytes)
        if kind == "legacy":
            return legacy_observation(_observations(row), request_bytes)
        if kind == "type":
            return {"success": True}
        updates: dict[str, dict[str, Any]] = {
            "legacy_policy": {"policy": LEGACY_POLICY},
            "legacy_domain": {
                "policy_sha256": domain_sha256(
                    "iris/python-sandbox-policy/v1",
                    canonical_json_bytes(policy.model_dump(mode="json")),
                )
            },
            "worker": {"worker_sha256": "a" * 64},
            "forged_policy": {"policy": policy.model_copy(update={"max_output_bytes": True})},
            "process": {"process": None},
        }
        return replace(observation, **updates[kind])

    with (
        patch.object(scoring, "execute_python_job_v2", AsyncMock(side_effect=execute)),
        pytest.raises(IrisCodeEvaluationUnavailableError),
    ):
        scoring.score_synthetic_answer_v2(
            row, output_text=row.expected_response, python_policy=_POLICY
        )


def test_standalone_checker_revalidates_both_forged_policies() -> None:
    policy = _POLICY.model_copy(update={"wall_time_ms": 4999})
    observation = replace(
        _observation(b"{}", b"{}"),
        policy=policy,
        policy_sha256=domain_sha256(
            "iris/python-sandbox-policy/v2", canonical_json_bytes(policy.model_dump(mode="json"))
        ),
    )
    with pytest.raises(PythonSandboxError):
        check_python_observation_v2(observation, b"{}", policy)


@pytest.mark.parametrize("policy", [LEGACY_POLICY, _POLICY.model_copy(update={"wall_time_ms": 1})])
def test_wrong_caller_policy_is_request_failure_before_backend(policy: Any) -> None:
    row = _rows()[0]
    with (
        patch.object(scoring, "execute_python_job_v2") as execute,
        pytest.raises(IrisSyntheticScoringError),
    ):
        scoring.score_synthetic_answer_v2(
            row, output_text=row.expected_response, python_policy=policy
        )
    execute.assert_not_called()


@pytest.mark.parametrize(
    "termination,exit_code,oom,reason",
    [
        ("timeout", 137, True, "memory_limit"),
        ("output_limit", 137, False, "output_limit"),
        ("timeout", 137, False, "wall_time_limit"),
        ("exited", 137, False, "execution_terminated"),
    ],
)
def test_resource_failure_overrides_even_forged_success(
    termination: Any, exit_code: int, oom: bool, reason: str
) -> None:
    row = _rows()[0]
    observation = replace(
        _observation(_observations(row), b"{}"),
        process=ProcessResult(0, _observations(row), b"", termination),
        container_exit_code=exit_code,
        oom_killed=oom,
    )
    check_python_observation_v2(observation, b"{}", _POLICY)
    assert assess_integer_list_observation_v2(
        observation, row.code_cases, source_submitted=True
    ) == ("incorrect", reason, 0)
