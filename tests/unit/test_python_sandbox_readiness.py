"""Bounded metadata-only Docker cold readiness; never a candidate/job retry."""

import asyncio
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

import iris
from iris.integrations import python_sandbox as sandbox
from iris.integrations.bounded_process import BoundedProcessError, ProcessLimits, ProcessResult


@pytest.mark.parametrize("unavailable", (0, 1, 2))
def test_only_info_is_retried_before_container_creation(unavailable: int) -> None:
    errors = [sandbox.PythonSandboxError("not ready") for _ in range(unavailable)]
    docker = AsyncMock(side_effect=[*errors, json.dumps({"ServerVersion": "29.8.0"})])
    sleep = AsyncMock()
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(asyncio, "sleep", sleep),
    ):
        observed = asyncio.run(sandbox._docker_ready_info())
    assert observed == {"ServerVersion": "29.8.0"}
    assert docker.await_args_list == [call("info", "--format", "{{json .}}")] * (unavailable + 1)
    assert docker.await_count == unavailable + 1
    assert sleep.await_args_list == [call(0.25)] * unavailable


def test_persistent_unavailability_uses_exactly_three_existing_budgets() -> None:
    failure = sandbox.PythonSandboxError("not ready")
    docker, sleep = AsyncMock(side_effect=failure), AsyncMock()
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(asyncio, "sleep", sleep),
        pytest.raises(sandbox.PythonSandboxError) as caught,
    ):
        asyncio.run(sandbox._docker_ready_info())
    assert caught.value is failure and docker.await_count == 3
    assert sleep.await_args_list == [call(0.25), call(0.25)]
    assert all(c.args == ("info", "--format", "{{json .}}") for c in docker.await_args_list)


@pytest.mark.parametrize("raw", ("not-json", "[]", "null", '"scalar"'))
def test_successful_malformed_metadata_is_not_retried(raw: str) -> None:
    docker, sleep = AsyncMock(return_value=raw), AsyncMock()
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(asyncio, "sleep", sleep),
        pytest.raises((ValueError, sandbox.PythonSandboxError)),
    ):
        asyncio.run(sandbox._docker_ready_info())
    assert docker.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    (
        BoundedProcessError("transport ownership failed"),
        ValueError("invalid"),
        asyncio.CancelledError(),
    ),
)
def test_process_errors_and_cancellation_are_not_retried(failure: BaseException) -> None:
    docker, sleep = AsyncMock(side_effect=failure), AsyncMock()
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(asyncio, "sleep", sleep),
        pytest.raises(type(failure)) as caught,
    ):
        asyncio.run(sandbox._docker_ready_info())
    assert caught.value is failure and docker.await_count == 1
    sleep.assert_not_awaited()


def test_successful_wrong_version_fails_preflight_without_retry_or_create() -> None:
    docker, sleep = AsyncMock(return_value='{"ServerVersion":"wrong"}'), AsyncMock()
    linux = AsyncMock(return_value="")
    with (
        patch.object(sys, "platform", "win32"),
        patch.object(sandbox, "_docker", docker),
        patch.object(sandbox, "_linux", linux),
        patch.object(asyncio, "sleep", sleep),
        pytest.raises(sandbox.PythonSandboxError),
    ):
        asyncio.run(sandbox._preflight())
    assert docker.await_args_list == [call("info", "--format", "{{json .}}")]
    linux.assert_awaited_once_with("/usr/bin/test", "!", "-e", sandbox._CLIENT_CONFIG)
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "module",
    (
        "iris.integrations.python_sandbox",
        "iris.integrations.python_sandbox_v2",
        "iris.integrations.python_comparison",
        "iris.artifacts.scientific_evaluation_v1",
        "iris.artifacts.scientific_evaluation_v2",
    ),
)
def test_actual_cold_import_without_application_or_comparison_prewarm(module: str) -> None:
    assert iris.__file__ is not None
    package_root = Path(iris.__file__).resolve().parent.parent
    worker = Path(__file__).with_name("_python_sandbox_import_worker.py")
    actual = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(worker),
            str(package_root),
            module,
            sha256(worker.read_bytes()).hexdigest(),
        ],
        capture_output=True,
        check=False,
        timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert actual.returncode == 0, actual.stderr.decode("utf-8", errors="replace")
    assert actual.stdout.strip() == b"COLD_IMPORT_OK" and actual.stderr == b""


def test_exact_first_readonly_wsl_command_has_separate_bounded_bootstrap_clock() -> None:
    exchange = AsyncMock(return_value=ProcessResult(0, b"", b"", "exited"))
    with patch.object(sandbox, "run_bounded_process", exchange):
        assert asyncio.run(sandbox._linux(*sandbox._WSL_BOOTSTRAP_TEST)) == ""
    assert ProcessLimits(1, 256 * 1024, 5000, 10000, 3000) == sandbox._CONTROL
    assert ProcessLimits(1, 256 * 1024, 5000, 30000, 3000) == sandbox._WSL_BOOTSTRAP_CONTROL
    exchange.assert_awaited_once_with(
        (*sandbox._PREFIX, "/usr/bin/test", "!", "-e", sandbox._CLIENT_CONFIG),
        input_bytes=b"",
        limits=sandbox._WSL_BOOTSTRAP_CONTROL,
    )


@pytest.mark.parametrize(
    "commands",
    (
        (*sandbox._DOCKER, "info", "--format", "{{json .}}"),
        (*sandbox._DOCKER, "create", "original-arguments"),
        ("/usr/bin/dpkg", "--verify", "runsc"),
        ("/usr/bin/runsc", "--version"),
        ("/usr/bin/uname", "-r"),
        ("/usr/bin/test", "!", "-e", "/different-client-config"),
        ("/usr/bin/test", "-e", sandbox._CLIENT_CONFIG),
        ("/bin/test", "!", "-e", sandbox._CLIENT_CONFIG),
    ),
)
def test_no_other_control_or_similar_command_inherits_bootstrap_budget(
    commands: tuple[str, ...],
) -> None:
    exchange = AsyncMock(return_value=ProcessResult(0, b"", b"", "exited"))
    with patch.object(sandbox, "run_bounded_process", exchange):
        assert asyncio.run(sandbox._linux(*commands)) == ""
    exchange.assert_awaited_once_with(
        (*sandbox._PREFIX, *commands), input_bytes=b"", limits=sandbox._CONTROL
    )


@pytest.mark.parametrize(
    "observation",
    (
        ProcessResult(1, b"", b"", "exited"),
        ProcessResult(0, b"", b"", "timeout"),
        ProcessResult(0, b"", b"", "output_limit"),
    ),
)
def test_failed_bootstrap_still_denies_before_docker_without_retry(
    observation: ProcessResult,
) -> None:
    exchange = AsyncMock(return_value=observation)
    with (
        patch.object(sys, "platform", "win32"),
        patch.object(sandbox, "run_bounded_process", exchange),
        patch.object(sandbox, "_docker", AsyncMock()) as docker,
        patch.object(asyncio, "sleep", AsyncMock()) as sleep,
        pytest.raises(sandbox.PythonSandboxError),
    ):
        asyncio.run(sandbox._preflight())
    assert exchange.await_count == 1
    docker.assert_not_awaited()
    sleep.assert_not_awaited()


@pytest.mark.parametrize("failure", (BoundedProcessError("unavailable"), asyncio.CancelledError()))
def test_bootstrap_transport_error_and_cancellation_propagate_without_retry(
    failure: BaseException,
) -> None:
    exchange = AsyncMock(side_effect=failure)
    with (
        patch.object(sys, "platform", "win32"),
        patch.object(sandbox, "run_bounded_process", exchange),
        patch.object(sandbox, "_docker", AsyncMock()) as docker,
        patch.object(asyncio, "sleep", AsyncMock()) as sleep,
        pytest.raises(type(failure)) as caught,
    ):
        asyncio.run(sandbox._preflight())
    assert caught.value is failure and exchange.await_count == 1
    docker.assert_not_awaited()
    sleep.assert_not_awaited()
