"""Fail-closed settings and ownership, without spawning candidate code."""

from __future__ import annotations

import ast
import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iris.integrations import python_sandbox as sandbox
from tests.unit.test_study_code_evaluation import _POLICY

_ID = "a" * 64
_OWNER = "b" * 32
_SOURCE = "fixed-worker"


def _configuration() -> dict[str, Any]:
    return {
        "Name": "/" + sandbox._NAME,
        "Image": "sha256:" + _ID,
        "Config": {
            "Labels": {sandbox._LABEL: _OWNER},
            "User": "65534:65534",
            "Volumes": None,
            "Env": ["PATH=/usr/bin"],
            "Entrypoint": ["python3"],
            "Cmd": ["-I", "-S", "-B", "-u", "-c", _SOURCE],
            "OpenStdin": True,
            "Tty": False,
        },
        "HostConfig": {
            "Runtime": "runsc",
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "Memory": 268435456,
            "MemorySwap": 268435456,
            "NanoCpus": 500000000,
            "PidsLimit": 128,
            "CgroupnsMode": "private",
            "PidMode": "",
            "IpcMode": "private",
            "UTSMode": "",
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"},
            "ShmSize": 8388608,
            "LogConfig": {"Type": "none", "Config": {}},
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Ulimits": [
                {"Name": key, "Soft": value, "Hard": value}
                for key, value in {"fsize": 1048576, "nofile": 64, "nproc": 8, "core": 0}.items()
            ],
        },
    }


def _verify(state: dict[str, Any]) -> None:
    sandbox._verify_configuration(
        state,
        owner=_OWNER,
        image_id="sha256:" + _ID,
        source=_SOURCE,
        image_environment=("PATH=/usr/bin",),
    )


def test_complete_hardened_configuration_is_accepted() -> None:
    _verify(_configuration())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Runtime", "runc"),
        ("NetworkMode", "host"),
        ("ReadonlyRootfs", False),
        ("Privileged", True),
        ("CapDrop", []),
        ("SecurityOpt", ["seccomp=unconfined"]),
        ("Memory", 0),
        ("MemorySwap", -1),
        ("NanoCpus", 0),
        ("PidsLimit", -1),
        ("CgroupnsMode", "host"),
        ("PidMode", "host"),
        ("IpcMode", "host"),
        ("UTSMode", "host"),
        ("Tmpfs", {}),
        ("ShmSize", 0),
        ("Ulimits", []),
        ("LogConfig", {"Type": "json-file", "Config": {}}),
        ("RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}),
        ("Binds", ["/mnt/c:/host"]),
        ("Mounts", [{"Type": "bind"}]),
        ("Devices", ["device"]),
        ("DeviceRequests", ["gpu"]),
        ("PortBindings", {"1/tcp": []}),
        ("CapAdd", ["SYS_ADMIN"]),
        ("VolumesFrom", ["other"]),
        ("ExtraHosts", ["host:1.1.1.1"]),
        ("GroupAdd", ["0"]),
        ("Sysctls", {"x": "y"}),
        ("Links", ["other"]),
    ],
)
def test_one_unsafe_host_setting_fails_closed(field: str, value: Any) -> None:
    state = _configuration()
    state["HostConfig"][field] = value
    with pytest.raises(sandbox.PythonSandboxError):
        _verify(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("User", "root"),
        ("Volumes", {"/host": {}}),
        ("Labels", {sandbox._LABEL: "someone-else"}),
        ("Entrypoint", ["sh"]),
        ("Cmd", ["unsafe"]),
        ("OpenStdin", False),
        ("Tty", True),
        ("Env", ["HTTP_PROXY=private-host-secret"]),
        ("ExposedPorts", {"80/tcp": {}}),
    ],
)
def test_wrong_worker_identity_or_mode_fails_closed(field: str, value: Any) -> None:
    state = _configuration()
    state["Config"][field] = value
    with pytest.raises(sandbox.PythonSandboxError):
        _verify(state)


def test_create_is_no_pull_and_uses_only_fixed_worker_as_command_source() -> None:
    args = sandbox._create_args(_OWNER, _SOURCE)
    assert "--pull=never" in args and "--runtime=runsc" in args and "--network=none" in args
    assert args[-1] == _SOURCE and args.count(_SOURCE) == 1
    assert sandbox.IMAGE in args and not any(arg.startswith("--volume") for arg in args)


def test_worker_resource_is_only_parsed_not_executed_on_host() -> None:
    source = sandbox._worker()
    parsed = ast.parse(source.decode())
    assert parsed.body
    assert b"expected" not in source.lower().split(b"import json", 1)[1]


def test_preflight_failure_prevents_create_and_input_transfer() -> None:
    with (
        patch.object(sandbox, "_preflight", AsyncMock(side_effect=RuntimeError("missing"))),
        patch.object(sandbox, "_docker", AsyncMock()) as docker,
        patch.object(sandbox, "run_bounded_process", AsyncMock()) as exchange,
        pytest.raises(sandbox.PythonSandboxError),
    ):
        asyncio.run(sandbox.execute_python_job_v1(request_bytes=b"{}", policy=_POLICY))
    docker.assert_not_called()
    exchange.assert_not_called()


def test_ambiguous_create_cleanup_removes_only_its_own_labelled_id() -> None:
    docker = AsyncMock(side_effect=[_ID, _ID, ""])
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(sandbox, "_inspect", AsyncMock(return_value=_configuration())),
    ):
        asyncio.run(sandbox._remove_owned(_OWNER, None))
    assert docker.call_args_list[1].args == ("rm", "--force", _ID)


def test_existing_other_owner_is_never_removed_on_create_collision() -> None:
    state = copy.deepcopy(_configuration())
    state["Config"]["Labels"][sandbox._LABEL] = "other-owner"
    docker = AsyncMock(return_value=_ID)
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(sandbox, "_inspect", AsyncMock(return_value=state)),
    ):
        asyncio.run(sandbox._remove_owned(_OWNER, None))
    assert docker.call_count == 1


def test_replaced_claim_fails_without_deleting_the_new_owner() -> None:
    state = _configuration()
    state["Config"]["Labels"][sandbox._LABEL] = "other-owner"
    docker = AsyncMock(return_value=_ID)
    with (
        patch.object(sandbox, "_docker", docker),
        patch.object(sandbox, "_inspect", AsyncMock(return_value=state)),
        pytest.raises(sandbox.PythonSandboxError),
    ):
        asyncio.run(sandbox._remove_owned(_OWNER, _ID))
    assert docker.call_count == 1


def test_missing_owned_claim_is_not_reported_as_successful_cleanup() -> None:
    with (
        patch.object(sandbox, "_docker", AsyncMock(return_value="")),
        pytest.raises(sandbox.PythonSandboxError),
    ):
        asyncio.run(sandbox._remove_owned(_OWNER, _ID))


def test_unexpected_effective_mount_is_rejected() -> None:
    state = _configuration()
    state["Mounts"] = [{"Type": "bind", "Source": "/", "Destination": "/tmp"}]
    with pytest.raises(sandbox.PythonSandboxError):
        _verify(state)
