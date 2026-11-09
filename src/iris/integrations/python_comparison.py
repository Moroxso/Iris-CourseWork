"""Shared external comparisons of the pinned integer-list worker protocol.

This module parses observations only. It neither runs candidates nor chooses
task families, benchmark targets, scientific budgets or an execution backend.
Callers must supply already validated bounded cases and check the observation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Protocol

from iris.artifacts.code_execution_v1 import PythonOutcome as PythonOutcome
from iris.artifacts.code_execution_v1 import PythonReason as PythonReason
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_sandbox import PythonSandboxError, PythonSandboxObservationV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class IntegerListCase(Protocol):
    @property
    def inputs(self) -> tuple[int, ...]: ...

    @property
    def expected(self) -> int: ...


class _ExecutionObservation(Protocol):
    """Read-only transport fields shared by explicitly checked runtime versions."""

    @property
    def process(self) -> ProcessResult: ...

    @property
    def container_exit_code(self) -> int: ...

    @property
    def oom_killed(self) -> bool: ...


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate worker response member")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 20:
        raise ValueError("Worker response integer is too large")
    parsed = int(value)
    if not -(2**63) <= parsed < 2**63:
        raise ValueError("Worker response integer is out of range")
    return parsed


def _not_number(_value: str) -> Any:
    raise ValueError("Non-integer worker response number")


def compare_integer_list_response_v1(
    raw: bytes, expected_cases: Sequence[IntegerListCase], *, source_submitted: bool
) -> tuple[PythonOutcome, PythonReason, int]:
    """Only data is parsed; no eval, pickle, candidate pass flag or oracle transfer."""
    try:
        response = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicates,
            parse_int=_integer,
            parse_float=_not_number,
            parse_constant=_not_number,
        )
        if type(response) is not dict or type(response.get("kind")) is not str:
            raise ValueError("Invalid worker response")
        kind = response["kind"]
        if not source_submitted:
            if response != {"kind": "invalid_source"}:
                raise ValueError("Invalid non-executing worker response")
            return "invalid_output", "invalid_source", 0
        if kind in ("syntax_error", "missing_entrypoint", "load_exception"):
            if set(response) != {"kind"}:
                raise ValueError("Invalid worker failure response")
            if kind == "syntax_error":
                return "invalid_output", "syntax_error", 0
            if kind == "missing_entrypoint":
                return "invalid_output", "missing_entrypoint", 0
            return "incorrect", "load_exception", 0
        if set(response) != {"kind", "cases"} or kind != "observations":
            raise ValueError("Unknown worker protocol")
        cases = response["cases"]
        if type(cases) is not list or len(cases) != len(expected_cases):
            raise ValueError("Incomplete worker case observations")
        passed = 0
        for observed, case in zip(cases, expected_cases, strict=True):
            if type(observed) is not dict:
                raise ValueError("Invalid worker case")
            if observed in ({"kind": "exception"}, {"kind": "non_integer"}):
                continue
            if (
                set(observed) != {"kind", "result", "input_after"}
                or observed["kind"] != "value"
                or type(observed["result"]) is not int
            ):
                raise ValueError("Invalid worker value")
            after = observed["input_after"]
            if after is None:
                continue
            if (
                type(after) is not list
                or len(after) > 1024
                or any(type(x) is not int or not -(2**31) <= x < 2**31 for x in after)
            ):
                raise ValueError("Invalid worker input observation")
            if observed["result"] == case.expected and tuple(after) == case.inputs:
                passed += 1
        if passed == len(cases):
            return "correct", "all_cases_pass", passed
        return "incorrect", "case_mismatch", passed
    except Exception:
        # Disrupted candidate channels never establish success. Reserved worker
        # and runtime failures are rejected before entering this parser.
        return "incorrect", "protocol_error", 0


def assess_integer_list_observation_v1(
    observation: _ExecutionObservation,
    cases: Sequence[IntegerListCase],
    *,
    source_submitted: bool,
) -> tuple[PythonOutcome, PythonReason, int]:
    """Apply resource-failure precedence, then compare outside the container."""
    if observation.oom_killed:
        return "incorrect", "memory_limit", 0
    if observation.process.termination == "output_limit":
        return "incorrect", "output_limit", 0
    if observation.process.termination == "timeout":
        return "incorrect", "wall_time_limit", 0
    if observation.container_exit_code != 0:
        return "incorrect", "execution_terminated", 0
    if observation.process.returncode != 0:
        raise PythonSandboxError("Container/client exit observations disagree")
    return compare_integer_list_response_v1(
        observation.process.stdout, cases, source_submitted=source_submitted
    )


def check_python_observation_v1(
    observation: PythonSandboxObservationV1, request: bytes, policy: PythonSandboxPolicyV1
) -> None:
    """Validate binding and bounds, not historical execution authenticity."""
    if (
        type(observation) is not PythonSandboxObservationV1
        or type(observation.process) is not ProcessResult
        or observation.policy != policy
        or observation.policy_sha256
        != domain_sha256(
            "iris/python-sandbox-policy/v1", canonical_json_bytes(policy.model_dump(mode="json"))
        )
        or observation.submitted_input_sha256 != sha256(request).hexdigest()
        or type(observation.worker_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", observation.worker_sha256) is None
        or type(observation.image_id) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", observation.image_id) is None
        or observation.runtime_version != "release-20260831.0"
        or observation.docker_version != "29.8.0"
        or type(observation.kernel_release) is not str
        or not 1 <= len(observation.kernel_release) <= 256
        or type(observation.process.stdout) is not bytes
        or type(observation.process.stderr) is not bytes
        or len(observation.process.stdout) + len(observation.process.stderr)
        > policy.max_output_bytes
        or observation.process.termination not in ("exited", "timeout", "output_limit")
        or type(observation.process.returncode) is not int
        or type(observation.container_exit_code) is not int
        or not 0 <= observation.container_exit_code <= 255
        or observation.container_exit_code == 80
        or type(observation.oom_killed) is not bool
    ):
        raise PythonSandboxError("Sandbox observation identity or bounds disagree")
