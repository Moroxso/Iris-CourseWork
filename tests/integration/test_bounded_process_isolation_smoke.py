"""Opt-in fixed transport fixtures through the already installed gVisor runtime.

This is NOT a generated-code scorer. All programs are fixed trusted test strings;
no model, benchmark, reference solution, image pull, or installation is involved.
The existing operational probe owns its exact labelled container configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from hashlib import sha256
from typing import Any, cast

import pytest
from scripts import verify_iris_code_isolation as isolation

from iris.integrations.bounded_process import ProcessLimits, run_bounded_process

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="set IRIS_RUN_CODE_ISOLATION_SMOKE=1 with the verified Windows IrisCodeEval runtime",
    ),
]

_PREFIX = (isolation.WSL, "--distribution", "IrisCodeEval", "--user", "root", "--cd", "/", "--exec")
_CONTROL_LIMITS = ProcessLimits(
    max_input_bytes=1,
    max_output_bytes=256 * 1024,
    startup_timeout_ms=5000,
    execution_timeout_ms=5000,
    cleanup_timeout_ms=3000,
)
_EXCHANGE_LIMITS = ProcessLimits(
    max_input_bytes=256 * 1024,
    max_output_bytes=256 * 1024,
    startup_timeout_ms=30000,
    execution_timeout_ms=2000,
    cleanup_timeout_ms=15000,
)
_FIXTURES = {
    # -u exposes raw FileIO: a single write may legitimately be partial.
    "binary_echo": "\nwhile payload: payload = payload[sys.stdout.buffer.write(payload):]",
    "stdout_overflow": "\nwhile True: sys.stdout.buffer.write(b'x' * 8192)",
    "stderr_overflow": "\nwhile True: sys.stderr.buffer.write(b'x' * 8192)",
    "timeout": "import time; time.sleep(60)",
}


async def _linux(*args: str) -> str:
    observation = await run_bounded_process(
        (*_PREFIX, *args), input_bytes=b"", limits=_CONTROL_LIMITS
    )
    assert observation.termination == "exited" and observation.returncode == 0
    return observation.stdout.decode("utf-8", errors="strict").strip()


async def _inspect(container: str) -> dict[str, Any]:
    parsed = json.loads(await _linux("/usr/bin/docker", "inspect", container))
    assert type(parsed) is list and len(parsed) == 1 and type(parsed[0]) is dict
    return cast(dict[str, Any], parsed[0])


@pytest.mark.parametrize("fixture", tuple(_FIXTURES))
def test_fixed_gvisor_job_uses_bounded_io_and_real_container_cleanup(fixture: str) -> None:
    assert __debug__, "Isolation verification requires enabled Python assertions"
    # Trusted fixed, bounded setup diagnostics; never untrusted candidate output.
    image = json.loads(isolation.linux("docker", "image", "inspect", isolation.IMAGE))[0]
    assert image["Os"] == "linux" and image["Architecture"] == "amd64"
    assert not image["Config"].get("Volumes")
    assert isolation.linux("dpkg", "--verify", "runsc") == ""
    source = "import sys\npayload = sys.stdin.buffer.read()\n" + _FIXTURES[fixture]
    container = isolation.create(source)
    stages: list[str] = []
    stopped_state: list[dict[str, Any]] = []
    payload = bytes(range(256)) * 512

    async def before_input() -> None:
        while not (await _inspect(container))["State"]["Running"]:
            await asyncio.sleep(0.05)
        state = await _inspect(container)
        pid = state["State"]["Pid"]
        assert type(pid) is int and pid > 0
        assert await _linux("/usr/bin/readlink", f"/proc/{pid}/exe") == (
            "/usr/bin/gvisor-bin/gvisor_sentry"
        )
        assert await _linux("/usr/bin/cat", f"/proc/{pid}/cgroup") == (
            f"0::/system.slice/docker-{container}.scope"
        )
        root = f"/sys/fs/cgroup/system.slice/docker-{container}.scope"
        expected = {
            "memory.max": "268435456",
            "memory.swap.max": "0",
            "cpu.max": "50000 100000",
            "pids.max": "128",
        }
        for field, value in expected.items():
            assert await _linux("/usr/bin/cat", root + "/" + field) == value
        stages.append("live runtime verified")

    async def before_stop() -> None:
        owned = await _inspect(container)
        assert owned["Config"]["Labels"].get("iris.operational-isolation-probe") == "true"
        if owned["State"]["Running"]:
            # Terminate the actual Linux job BEFORE closing its observing WSL client.
            assert await _linux("/usr/bin/docker", "kill", container) == container
        state = (await _inspect(container))["State"]
        assert state["Running"] is False
        stopped_state.append(state)
        stages.append("actual container stopped")

    try:
        isolation.verify_configuration(container, image["Id"])
        result = asyncio.run(
            run_bounded_process(
                (*_PREFIX, "/usr/bin/docker", "start", "--attach", "--interactive", container),
                input_bytes=payload,
                limits=_EXCHANGE_LIMITS,
                before_input=before_input,
                before_stop=before_stop,
            ),
            debug=True,
        )
        assert stages == ["live runtime verified", "actual container stopped"]
        assert stopped_state[0]["OOMKilled"] is False
        if fixture == "binary_echo":
            assert result.termination == "exited" and result.returncode == 0
            assert (len(result.stdout), sha256(result.stdout).hexdigest()) == (
                len(payload),
                sha256(payload).hexdigest(),
            )
            assert result.stderr == b"", repr(result.stderr[:500])
            assert stopped_state[0]["ExitCode"] == 0
        elif fixture == "timeout":
            assert result.termination == "timeout"
            assert result.stdout == result.stderr == b""
            assert stopped_state[0]["ExitCode"] == 137
        else:
            assert result.termination == "output_limit"
            assert len(result.stdout) + len(result.stderr) == _EXCHANGE_LIMITS.max_output_bytes
            assert stopped_state[0]["ExitCode"] == 137
    finally:
        owned = isolation.inspect(container)
        assert owned["Config"]["Labels"].get("iris.operational-isolation-probe") == "true"
        assert isolation.linux("docker", "rm", "--force", container) == container
        assert (
            isolation.linux(
                "docker", "ps", "--all", "--filter", "id=" + container, "--format", "{{.ID}}"
            )
            == ""
        )
