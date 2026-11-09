"""Source resource bounds, legacy isolation identity and fail-closed lifecycle."""

import ast
import asyncio
import copy
from hashlib import sha256
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.integrations import python_sandbox as runtime
from iris.integrations import python_sandbox_v2 as sandbox
from iris.integrations.bounded_process import ProcessResult
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_python_sandbox import _ID, _OWNER, _SOURCE, _configuration

POLICY = PythonSandboxPolicyV2(
    profile="iris_wsl_gvisor_python312_source_v2", wall_time_ms=5000, max_output_bytes=1024**2
)


def configuration(source: str = _SOURCE) -> dict[str, Any]:
    state = _configuration()
    state["Config"].update(
        Entrypoint=["/usr/bin/env"],
        Cmd=["-i", "/usr/local/bin/python3", "-I", "-S", "-B", "-u", "-c", source],
    )
    host = state["HostConfig"]
    host.update(Memory=512000000, MemorySwap=512000000, NanoCpus=1000000000, PidsLimit=32)
    next(item for item in host["Ulimits"] if item["Name"] == "nproc").update(Soft=32, Hard=32)
    state["State"] = {"Running": False, "Error": "", "ExitCode": 0, "OOMKilled": False}
    return state


def verify(state: dict[str, Any]) -> None:
    runtime._verify_configuration(
        state,
        owner=_OWNER,
        image_id="sha256:" + _ID,
        source=_SOURCE,
        image_environment=("PATH=/usr/bin",),
        runtime=runtime._SOURCE_V2,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "iris_wsl_gvisor_python312_restricted_v1"},
        {"wall_time_ms": 4999},
        {"wall_time_ms": 5001},
        {"wall_time_ms": "5000"},
        {"wall_time_ms": True},
        {"max_output_bytes": 1023},
        {"max_output_bytes": 8 * 1024**2 + 1},
        {"memory": 1024},
    ],
)
def test_policy_is_explicit_and_fixed(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PythonSandboxPolicyV2.model_validate({**POLICY.model_dump(mode="python"), **change})


def test_v1_schema_does_not_accept_or_relabel_v2_policy() -> None:
    with pytest.raises(ValidationError):
        PythonSandboxPolicyV1.model_validate(POLICY.model_dump(mode="python"))


def test_source_configuration_and_command() -> None:
    verify(configuration())
    args = runtime._create_args(_OWNER, _SOURCE, runtime=runtime._SOURCE_V2)
    for flag in (
        "--memory=512000000",
        "--memory-swap=512000000",
        "--cpus=1.0",
        "--pids-limit=32",
        "nproc=32:32",
        "--entrypoint=/usr/bin/env",
        "--network=none",
        "--read-only",
        "--pull=never",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    ):
        assert flag in args
    assert args[-8:] == ("-i", "/usr/local/bin/python3", "-I", "-S", "-B", "-u", "-c", _SOURCE)
    assert not any(arg.startswith("--volume") for arg in args)
    old = runtime._create_args(_OWNER, _SOURCE)
    assert "--cpus=0.5" in old and "--memory=256m" in old and "--pids-limit=128" in old
    assert "nproc=8:8" in old and "--entrypoint=python3" in old and "-i" not in old


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("HostConfig", "Memory", 268435456),
        ("HostConfig", "MemorySwap", 1073741824),
        ("HostConfig", "NanoCpus", 500000000),
        ("HostConfig", "PidsLimit", 128),
        ("HostConfig", "Ulimits", []),
        ("HostConfig", "NetworkMode", "host"),
        ("HostConfig", "Binds", ["/mnt/c:/host"]),
        ("HostConfig", "ReadonlyRootfs", False),
        ("Config", "Entrypoint", ["python3"]),
        ("Config", "Env", ["PRIVATE_TOKEN=not-permitted"]),
        ("Config", "Cmd", ["-i", "python3", "-c", _SOURCE]),
    ],
)
def test_source_profile_rejects_relaxed_or_old_settings(
    section: str, field: str, value: Any
) -> None:
    state = configuration()
    state[section][field] = value
    with pytest.raises(runtime.PythonSandboxError):
        verify(state)


def test_v1_verifier_does_not_accept_source_profile_as_old_evidence() -> None:
    with pytest.raises(runtime.PythonSandboxError):
        runtime._verify_configuration(
            configuration(),
            owner=_OWNER,
            image_id="sha256:" + _ID,
            source=_SOURCE,
            image_environment=("PATH=/usr/bin",),
        )


def test_worker_clears_environment_before_compiling_or_executing_candidate() -> None:
    source = runtime._worker("python_worker_v2.txt").decode("utf-8")
    tree = ast.parse(source)  # Candidate/worker code is NOT executed on the host.
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    first = main.body[0]
    assert isinstance(first, ast.Expr) and ast.unparse(first) == "os.environ.clear()"
    assert "(resource.RLIMIT_NPROC, 32)" in source
    old = runtime._worker().decode("utf-8")
    assert "os.environ.clear()" not in old and "(resource.RLIMIT_NPROC, 8)" in old


@pytest.mark.parametrize("bad_key", [None, "memory.max", "memory.swap.max", "cpu.max", "pids.max"])
def test_live_cgroups_are_verified_before_input(bad_key: str | None) -> None:
    values = {
        "memory.max": "512000000",
        "memory.swap.max": "0",
        "cpu.max": "100000 100000",
        "pids.max": "32",
    }

    async def linux(*args: str) -> str:
        if args[0] == "/usr/bin/readlink":
            return "/usr/bin/gvisor-bin/gvisor_sentry"
        if args[-1] == "/proc/123/cgroup":
            return f"0::/system.slice/docker-{_ID}.scope"
        key = args[-1].rsplit("/", 1)[-1]
        return "invalid" if key == bad_key else values[key]

    with (
        patch.object(
            runtime, "_inspect", AsyncMock(return_value={"State": {"Running": True, "Pid": 123}})
        ),
        patch.object(runtime, "_linux", side_effect=linux),
    ):
        if bad_key is None:
            asyncio.run(runtime._verify_live(_ID, runtime=runtime._SOURCE_V2))
        else:
            with pytest.raises(runtime.PythonSandboxError):
                asyncio.run(runtime._verify_live(_ID, runtime=runtime._SOURCE_V2))


@pytest.mark.parametrize("kind", ["empty", "wrong_type", "oversize"])
def test_invalid_input_never_preflights_or_creates(kind: str) -> None:
    bad: Any = (
        b"" if kind == "empty" else "{}" if kind == "wrong_type" else b"x" * (8 * 1024**2 + 1)
    )
    with (
        patch.object(runtime, "_preflight", AsyncMock()) as preflight,
        pytest.raises(runtime.PythonSandboxError),
    ):
        asyncio.run(sandbox.execute_python_job_v2(request_bytes=bad, policy=POLICY))
    preflight.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "preflight",
        "create",
        "configuration",
        "gate",
        "post_configuration",
        "worker",
        "exit80",
        "cleanup",
        "cancel",
    ],
)
def test_owned_lifecycle_and_cleanup(failure: str | None) -> None:
    source = runtime._worker("python_worker_v2.txt").decode()
    state = configuration(source)
    worker_bytes = source.encode()
    changed = copy.deepcopy(state)
    changed["HostConfig"]["PidsLimit"] = 128
    if failure == "exit80":
        state["State"]["ExitCode"] = 80
    inspect = AsyncMock(
        side_effect=[
            changed if failure == "configuration" else state,
            changed if failure == "post_configuration" else state,
            state,
        ]
    )
    cleanup = AsyncMock(side_effect=RuntimeError("private") if failure == "cleanup" else None)
    gate = AsyncMock(side_effect=RuntimeError("private") if failure == "gate" else None)

    async def exchange(*args: Any, **kwargs: Any) -> ProcessResult:
        assert kwargs["input_bytes"] == b"{}"
        assert kwargs["limits"].execution_timeout_ms == 5000
        await kwargs["before_input"]()
        if failure == "cancel":
            raise asyncio.CancelledError
        await kwargs["before_stop"]()
        return ProcessResult(0, b"observation", b"", "exited")

    with (
        patch.object(
            runtime,
            "_preflight",
            AsyncMock(
                return_value=("sha256:" + _ID, "kernel", ("PATH=/usr/bin",)),
                side_effect=RuntimeError("private") if failure == "preflight" else None,
            ),
        ),
        patch.object(
            runtime,
            "_docker",
            AsyncMock(
                return_value=_ID,
                side_effect=RuntimeError("private") if failure == "create" else None,
            ),
        ),
        patch.object(runtime, "_inspect", inspect),
        patch.object(runtime, "_verify_live", gate),
        patch.object(runtime, "_final_cleanup", cleanup),
        patch.object(
            runtime,
            "_worker",
            side_effect=[worker_bytes, b"changed" if failure == "worker" else worker_bytes],
        ),
        patch("iris.integrations.python_sandbox_v2.uuid.uuid4") as owner,
        patch.object(sandbox, "run_bounded_process", side_effect=exchange),
    ):
        owner.return_value.hex = _OWNER
        if failure is None:
            observed = asyncio.run(
                sandbox.execute_python_job_v2(request_bytes=b"{}", policy=POLICY)
            )
            assert (
                observed.policy == POLICY
                and observed.worker_sha256 == sha256(worker_bytes).hexdigest()
            )
            assert observed.policy_sha256 == domain_sha256(
                "iris/python-sandbox-policy/v2",
                canonical_json_bytes(POLICY.model_dump(mode="json")),
            )
        else:
            with pytest.raises(
                asyncio.CancelledError if failure == "cancel" else runtime.PythonSandboxError
            ):
                asyncio.run(sandbox.execute_python_job_v2(request_bytes=b"{}", policy=POLICY))
    if failure == "preflight":
        cleanup.assert_not_called()
    else:
        cleanup.assert_awaited_once_with(_OWNER, None if failure == "create" else _ID)
    if failure in {None, "post_configuration", "worker", "exit80", "cleanup", "cancel", "gate"}:
        gate.assert_awaited_once_with(_ID, runtime=runtime._SOURCE_V2)
