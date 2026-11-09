"""Fail-closed owned WSL/gVisor job lifecycle; no host candidate execution."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from typing import Any, cast

from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.integrations.bounded_process import ProcessLimits, ProcessResult, run_bounded_process
from iris.utils.hashing import canonical_json_bytes, domain_sha256

IMAGE = "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
_WSL = r"C:\Program Files\WSL\wsl.exe"
_PREFIX = (_WSL, "--distribution", "IrisCodeEval", "--user", "root", "--cd", "/", "--exec")
_CLIENT_CONFIG = "/etc/iris-empty-docker-client-v1"
_DOCKER = ("/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config=" + _CLIENT_CONFIG)
_NAME = "iris-python-evaluation-v1"
_LABEL = "iris.python-evaluation-v1.owner"
_CONTROL = ProcessLimits(1, 256 * 1024, 5000, 10000, 3000)
# WSL can need more than ten seconds to boot its distribution after idle shutdown.
# ONLY this exact read-only first command receives a separate bootstrap clock;
# every Docker/control command and the gated candidate clock remain unchanged.
_WSL_BOOTSTRAP_TEST = ("/usr/bin/test", "!", "-e", _CLIENT_CONFIG)
_WSL_BOOTSTRAP_CONTROL = ProcessLimits(1, 256 * 1024, 5000, 30000, 3000)


@dataclass(frozen=True, slots=True)
class _RuntimeProfile:
    memory_bytes: int
    cpu_quota: int
    pids: int
    nproc: int
    empty_environment: bool

    @property
    def memory_argument(self) -> str:
        # Keep immutable v1's exact argv while expressing decimal source MB exactly.
        if self.memory_bytes % 1024**2 == 0:
            return f"{self.memory_bytes // 1024**2}m"
        return str(self.memory_bytes)


_RESTRICTED_V1 = _RuntimeProfile(268435456, 50000, 128, 8, False)
_SOURCE_V2 = _RuntimeProfile(512000000, 100000, 32, 32, True)


def _entrypoint(runtime: _RuntimeProfile) -> str:
    return "/usr/bin/env" if runtime.empty_environment else "python3"


def _command(source: str, runtime: _RuntimeProfile) -> tuple[str, ...]:
    prefix = ("-i", "/usr/local/bin/python3") if runtime.empty_environment else ()
    return (*prefix, "-I", "-S", "-B", "-u", "-c", source)


class PythonSandboxError(RuntimeError):
    """An unavailable, unverified, busy or unclean runtime; not an answer failure."""


@dataclass(frozen=True, slots=True)
class PythonSandboxObservationV1:
    policy: PythonSandboxPolicyV1
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


def _require(value: bool) -> None:
    if not value:
        raise PythonSandboxError("Sandbox verification failed")


def _object(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    _require(type(parsed) is dict)
    return cast(dict[str, Any], parsed)


def _worker(resource_name: str = "python_worker_v1.txt") -> bytes:
    value = files("iris.integrations").joinpath(resource_name).read_bytes()
    _require(0 < len(value) <= 24 * 1024)
    value.decode("utf-8", errors="strict")
    return value


async def _linux(*args: str) -> str:
    limits = _WSL_BOOTSTRAP_CONTROL if args == _WSL_BOOTSTRAP_TEST else _CONTROL
    result = await run_bounded_process((*_PREFIX, *args), input_bytes=b"", limits=limits)
    _require(result.termination == "exited" and result.returncode == 0)
    return result.stdout.decode("utf-8", errors="strict").strip()


async def _docker(*args: str) -> str:
    return await _linux(*_DOCKER, *args)


async def _inspect(container: str) -> dict[str, Any]:
    parsed = json.loads(await _docker("container", "inspect", container))
    _require(type(parsed) is list and len(parsed) == 1 and type(parsed[0]) is dict)
    return cast(dict[str, Any], parsed[0])


async def _docker_ready_info() -> dict[str, Any]:
    """Bounded read-only startup probes, before any owner/container/input exists.

    WSL may start its systemd Docker service after the first client is ready.
    Only transport-level unavailability of `info` is retried, at most three
    existing CONTROL budgets and two250ms delays. Successful malformed metadata,
    wrong versions/configuration, process errors and signals are never retried.
    This cannot restart a job, replay candidate input, change/start a service,
    download an image, or relax any subsequent runtime/resource verification.
    """
    for attempt in range(3):
        try:
            raw = await _docker("info", "--format", "{{json .}}")
        except PythonSandboxError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.25)
        else:
            return _object(raw)
    raise PythonSandboxError("Docker readiness budget exhausted")


async def _preflight() -> tuple[str, str, tuple[str, ...]]:
    _require(sys.platform == "win32")
    # Ignore user Docker contexts/proxy configuration without creating a new file.
    await _linux("/usr/bin/test", "!", "-e", _CLIENT_CONFIG)
    info = await _docker_ready_info()
    _require(info["ServerVersion"] == "29.8.0")
    _require(info["CgroupVersion"] == "2" and info["CgroupDriver"] == "systemd")
    runtime = info["Runtimes"]["runsc"]
    _require(runtime["path"] == "/usr/bin/runsc")
    _require(runtime["runtimeArgs"] == ["--platform=systrap", "--network=none"])
    _require(await _linux("/usr/bin/dpkg", "--verify", "runsc") == "")
    _require(
        await _linux("/usr/bin/runsc", "--version")
        == ("runsc version release-20260831.0\nspec: 1.2.1")
    )
    image = json.loads(await _docker("image", "inspect", IMAGE))[0]
    _require(image["Os"] == "linux" and image["Architecture"] == "amd64")
    _require(IMAGE in image["RepoDigests"] and not image["Config"].get("Volumes"))
    image_id = image["Id"]
    _require(type(image_id) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is not None)
    environment = image["Config"]["Env"]
    _require(type(environment) is list and all(type(item) is str for item in environment))
    return image_id, await _linux("/usr/bin/uname", "-r"), tuple(environment)


def _create_args(
    owner: str, source: str, *, runtime: _RuntimeProfile = _RESTRICTED_V1
) -> tuple[str, ...]:
    return (
        "create",
        "--name",
        _NAME,
        "--label",
        _LABEL + "=" + owner,
        "--pull=never",
        "--runtime=runsc",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--memory={runtime.memory_argument}",
        f"--memory-swap={runtime.memory_argument}",
        f"--cpus={runtime.cpu_quota / 100000}",
        f"--pids-limit={runtime.pids}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--shm-size=8m",
        "--ulimit",
        "fsize=1048576:1048576",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        f"nproc={runtime.nproc}:{runtime.nproc}",
        "--ulimit",
        "core=0:0",
        "--log-driver=none",
        "--restart=no",
        "--interactive",
        "--entrypoint=" + _entrypoint(runtime),
        IMAGE,
        *_command(source, runtime),
    )


def _verify_configuration(
    state: dict[str, Any],
    *,
    owner: str,
    image_id: str,
    source: str,
    image_environment: tuple[str, ...],
    runtime: _RuntimeProfile = _RESTRICTED_V1,
) -> None:
    host, config = state["HostConfig"], state["Config"]
    _require(state["Name"] == "/" + _NAME and state["Image"] == image_id)
    _require(config["Labels"].get(_LABEL) == owner)
    _require(config["User"] == "65534:65534" and not config.get("Volumes"))
    _require(config["Env"] == list(image_environment) and not config.get("ExposedPorts"))
    _require(host.get("PublishAllPorts", False) is False)
    for mount in state.get("Mounts", []):
        _require(
            mount["Type"] == "tmpfs" and mount["Destination"] == "/tmp" and not mount.get("Source")
        )
    _require(config["Entrypoint"] == [_entrypoint(runtime)])
    _require(config["Cmd"] == list(_command(source, runtime)))
    _require(config["OpenStdin"] is True and config["Tty"] is False)
    _require(host["Runtime"] == "runsc" and host["NetworkMode"] == "none")
    _require(host["ReadonlyRootfs"] is True and host["Privileged"] is False)
    for field in (
        "Binds",
        "Mounts",
        "Devices",
        "DeviceRequests",
        "PortBindings",
        "CapAdd",
        "Links",
        "VolumesFrom",
        "ExtraHosts",
        "Sysctls",
        "GroupAdd",
    ):
        _require(not host.get(field))
    _require(host["CapDrop"] == ["ALL"] and host["SecurityOpt"] == ["no-new-privileges"])
    _require(host["Memory"] == host["MemorySwap"] == runtime.memory_bytes)
    _require(host["NanoCpus"] == runtime.cpu_quota * 10000 and host["PidsLimit"] == runtime.pids)
    _require(host["CgroupnsMode"] == "private" and host["PidMode"] == "")
    _require(host["IpcMode"] == "private" and host["UTSMode"] == "")
    _require(host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"})
    _require(host["ShmSize"] == 8388608 and host["LogConfig"] == {"Type": "none", "Config": {}})
    _require(host["RestartPolicy"] == {"Name": "no", "MaximumRetryCount": 0})
    limits = {entry["Name"]: (entry["Soft"], entry["Hard"]) for entry in host["Ulimits"]}
    _require(
        limits
        == {
            "fsize": (1048576, 1048576),
            "nofile": (64, 64),
            "nproc": (runtime.nproc, runtime.nproc),
            "core": (0, 0),
        }
    )


async def _verify_live(container: str, *, runtime: _RuntimeProfile = _RESTRICTED_V1) -> None:
    while True:
        state = (await _inspect(container))["State"]
        if state["Running"] is True:
            break
        _require(state["Status"] == "created")
        await asyncio.sleep(0.05)
    pid = state["Pid"]
    _require(type(pid) is int and pid > 0)
    _require(
        await _linux("/usr/bin/readlink", f"/proc/{pid}/exe")
        == ("/usr/bin/gvisor-bin/gvisor_sentry")
    )
    _require(
        await _linux("/usr/bin/cat", f"/proc/{pid}/cgroup")
        == (f"0::/system.slice/docker-{container}.scope")
    )
    root = f"/sys/fs/cgroup/system.slice/docker-{container}.scope/"
    for key, value in {
        "memory.max": str(runtime.memory_bytes),
        "memory.swap.max": "0",
        "cpu.max": f"{runtime.cpu_quota} 100000",
        "pids.max": str(runtime.pids),
    }.items():
        _require(await _linux("/usr/bin/cat", root + key) == value)


async def _remove_owned(owner: str, container: str | None) -> None:
    # A fixed Docker name is an atomic cross-process claim. On ambiguous create,
    # look up that name but delete only if its unique ownership label is ours.
    found = await _docker(
        "ps", "--all", "--no-trunc", "--filter", "name=^/" + _NAME + "$", "--format", "{{.ID}}"
    )
    if not found:
        _require(container is None)
        return
    _require(re.fullmatch(r"[0-9a-f]{64}", found) is not None)
    state = await _inspect(found)
    if state["Config"]["Labels"].get(_LABEL) != owner:
        _require(container is None)  # Do not reap another active/stale owner's job.
        return
    _require(container is None or found == container)
    _require(await _docker("rm", "--force", found) == found)
    _require(await _docker("ps", "--all", "--filter", "id=" + found, "--format", "{{.ID}}") == "")


async def _final_cleanup(owner: str, container: str | None) -> None:
    async def cleanup() -> None:
        async with asyncio.timeout(30):
            await _remove_owned(owner, container)

    task = asyncio.create_task(cleanup())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def execute_python_job_v1(
    *, request_bytes: bytes, policy: PythonSandboxPolicyV1
) -> PythonSandboxObservationV1:
    """Run an owned data request through the pinned worker, then remove its job.

    request_bytes must be produced by the trusted scoring service, never a shell
    or candidate argv. Preflight failure, another existing job, or failed cleanup
    is infrastructure failure. No automatic pull, install, fallback or reaping.
    """
    try:
        _require(type(policy) is PythonSandboxPolicyV1)
        owned = PythonSandboxPolicyV1.model_validate(policy.model_dump(mode="python"))
        _require(type(request_bytes) is bytes and 0 < len(request_bytes) <= 8 * 1024 * 1024)
        worker_bytes = _worker()
        source = worker_bytes.decode("utf-8")
        image_id, kernel, image_environment = await _preflight()
        owner = uuid.uuid4().hex
        container: str | None = None
        final_state: list[dict[str, Any]] = []
        try:
            created = await _docker(*_create_args(owner, source))
            _require(re.fullmatch(r"[0-9a-f]{64}", created) is not None)
            container = created
            _verify_configuration(
                await _inspect(created),
                owner=owner,
                image_id=image_id,
                source=source,
                image_environment=image_environment,
            )

            async def gate() -> None:
                await _verify_live(created)

            async def stop() -> None:
                state = await _inspect(created)
                _verify_configuration(
                    state,
                    owner=owner,
                    image_id=image_id,
                    source=source,
                    image_environment=image_environment,
                )
                if state["State"]["Running"]:
                    _require(await _docker("kill", created) == created)
                stopped = (await _inspect(created))["State"]
                _require(stopped["Running"] is False and stopped["Error"] == "")
                _require(type(stopped["ExitCode"]) is int and 0 <= stopped["ExitCode"] <= 255)
                _require(type(stopped["OOMKilled"]) is bool)
                final_state.append(stopped)

            observation = await run_bounded_process(
                (*_PREFIX, *_DOCKER, "start", "--attach", "--interactive", created),
                input_bytes=request_bytes,
                limits=ProcessLimits(
                    8 * 1024 * 1024, owned.max_output_bytes, 30000, owned.wall_time_ms, 30000
                ),
                before_input=gate,
                before_stop=stop,
            )
            _require(len(final_state) == 1 and final_state[0]["ExitCode"] != 80)
            _require(_worker() == worker_bytes)  # Local worker changed during execution.
        finally:
            await _final_cleanup(owner, container)
        return PythonSandboxObservationV1(
            policy=owned,
            policy_sha256=domain_sha256(
                "iris/python-sandbox-policy/v1", canonical_json_bytes(owned.model_dump(mode="json"))
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
        raise PythonSandboxError("Isolated Python execution could not complete safely") from None


__all__ = ["PythonSandboxError", "PythonSandboxObservationV1", "execute_python_job_v1"]
