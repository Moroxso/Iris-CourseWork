"""Exact source-v2 observation identity; external integer-list-v1 comparison."""

import re
from collections.abc import Sequence
from hashlib import sha256

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_comparison import (
    IntegerListCase,
    PythonOutcome,
    PythonReason,
    assess_integer_list_observation_v1,
)
from iris.integrations.python_sandbox import PythonSandboxError, _worker
from iris.integrations.python_sandbox_v2 import PythonSandboxObservationV2
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def check_python_observation_v2(
    observation: PythonSandboxObservationV2, request: bytes, policy: PythonSandboxPolicyV2
) -> None:
    """Validate binding and bounds, not historical execution authenticity."""
    try:
        if (
            type(observation) is not PythonSandboxObservationV2
            or type(policy) is not PythonSandboxPolicyV2
            or type(observation.policy) is not PythonSandboxPolicyV2
            or type(request) is not bytes
        ):
            raise ValueError("Exact v2 transport and policy required")
        # model_copy/model_construct may bypass frozen strict model validators.
        policy = PythonSandboxPolicyV2.model_validate(policy.model_dump(mode="python"))
        PythonSandboxPolicyV2.model_validate(observation.policy.model_dump(mode="python"))
        _check_binding(observation, request, policy)
    except Exception:
        raise PythonSandboxError("Sandbox observation identity or bounds disagree") from None


def _check_binding(
    observation: PythonSandboxObservationV2, request: bytes, policy: PythonSandboxPolicyV2
) -> None:
    if (
        type(observation) is not PythonSandboxObservationV2
        or type(observation.process) is not ProcessResult
        or type(policy) is not PythonSandboxPolicyV2
        or type(observation.policy) is not PythonSandboxPolicyV2
        or observation.policy != policy
        or observation.policy_sha256
        != domain_sha256(
            "iris/python-sandbox-policy/v2", canonical_json_bytes(policy.model_dump(mode="json"))
        )
        or observation.submitted_input_sha256 != sha256(request).hexdigest()
        or type(observation.worker_sha256) is not str
        or observation.worker_sha256 != sha256(_worker("python_worker_v2.txt")).hexdigest()
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


def assess_integer_list_observation_v2(
    observation: PythonSandboxObservationV2,
    cases: Sequence[IntegerListCase],
    *,
    source_submitted: bool,
) -> tuple[PythonOutcome, PythonReason, int]:
    """Compare the same integer-list protocol only after exact v2 identity checks."""
    return assess_integer_list_observation_v1(observation, cases, source_submitted=source_submitted)
