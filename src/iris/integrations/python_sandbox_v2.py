"""Source-section15 WSL/gVisor resource profile; no host candidate execution."""

import re
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.integrations import python_sandbox as runtime
from iris.integrations.bounded_process import ProcessLimits, ProcessResult, run_bounded_process
from iris.utils.hashing import canonical_json_bytes, domain_sha256


@dataclass(frozen=True, slots=True)
class PythonSandboxObservationV2:
    policy: PythonSandboxPolicyV2
    policy_sha256: str
    worker_sha256: str
    image_id: str
    runtime_version: str
    docker_version: str
    kernel_release: str
    submitted_input_sha256: str
    process: ProcessResult
    container_exit_code: int
    oom_killed: bool


async def execute_python_job_v2(
    *, request_bytes: bytes, policy: PythonSandboxPolicyV2
) -> PythonSandboxObservationV2:
    """Use the trusted worker with source resource bounds and an empty environment.

    The same exclusive Docker name/owner protocol protects v1 and v2 jobs from
    overlapping. Preflight and cleanup failure are infrastructure errors; no
    automatic downloads, profile fallback, host execution or stale-job reaping.
    """
    try:
        runtime._require(type(policy) is PythonSandboxPolicyV2)
        owned = PythonSandboxPolicyV2.model_validate(policy.model_dump(mode="python"))
        runtime._require(type(request_bytes) is bytes and 0 < len(request_bytes) <= 8 * 1024**2)
        worker_bytes = runtime._worker("python_worker_v2.txt")
        source = worker_bytes.decode("utf-8")
        image_id, kernel, image_environment = await runtime._preflight()
        owner = uuid.uuid4().hex
        container: str | None = None
        final_state: list[dict[str, Any]] = []
        try:
            created = await runtime._docker(
                *runtime._create_args(owner, source, runtime=runtime._SOURCE_V2)
            )
            runtime._require(re.fullmatch(r"[0-9a-f]{64}", created) is not None)
            container = created

            def verify(state: dict[str, Any]) -> None:
                runtime._verify_configuration(
                    state,
                    owner=owner,
                    image_id=image_id,
                    source=source,
                    image_environment=image_environment,
                    runtime=runtime._SOURCE_V2,
                )

            verify(await runtime._inspect(created))

            async def gate() -> None:
                await runtime._verify_live(created, runtime=runtime._SOURCE_V2)

            async def stop() -> None:
                state = await runtime._inspect(created)
                verify(state)
                if state["State"]["Running"]:
                    runtime._require(await runtime._docker("kill", created) == created)
                stopped = (await runtime._inspect(created))["State"]
                runtime._require(stopped["Running"] is False and stopped["Error"] == "")
                runtime._require(
                    type(stopped["ExitCode"]) is int and 0 <= stopped["ExitCode"] <= 255
                )
                runtime._require(type(stopped["OOMKilled"]) is bool)
                final_state.append(stopped)

            observation = await run_bounded_process(
                (*runtime._PREFIX, *runtime._DOCKER, "start", "--attach", "--interactive", created),
                input_bytes=request_bytes,
                limits=ProcessLimits(
                    8 * 1024**2, owned.max_output_bytes, 30000, owned.wall_time_ms, 30000
                ),
                before_input=gate,
                before_stop=stop,
            )
            runtime._require(len(final_state) == 1 and final_state[0]["ExitCode"] != 80)
            runtime._require(runtime._worker("python_worker_v2.txt") == worker_bytes)
        finally:
            await runtime._final_cleanup(owner, container)
        return PythonSandboxObservationV2(
            policy=owned,
            policy_sha256=domain_sha256(
                "iris/python-sandbox-policy/v2", canonical_json_bytes(owned.model_dump(mode="json"))
            ),
            worker_sha256=sha256(worker_bytes).hexdigest(),
            image_id=image_id,
            runtime_version="release-20260831.0",
            docker_version="29.8.0",
            kernel_release=kernel,
            submitted_input_sha256=sha256(request_bytes).hexdigest(),
            process=observation,
            container_exit_code=final_state[0]["ExitCode"],
            oom_killed=final_state[0]["OOMKilled"],
        )
    except Exception:
        raise runtime.PythonSandboxError(
            "Source-profile Python execution could not complete safely"
        ) from None


__all__ = ["PythonSandboxObservationV2", "execute_python_job_v2"]
