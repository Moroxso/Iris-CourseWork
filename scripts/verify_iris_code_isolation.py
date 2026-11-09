"""Fixed operational probes only, never a general candidate-code evaluator.

Run on Windows with the project Python. Python probe strings run exclusively
inside explicit gVisor containers in the dedicated IrisCodeEval WSL distro.
No user-supplied source or command line arguments are accepted. Captured output
is suitable only for these fixed bounded probes, not arbitrary generated code.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from typing import Any

WSL = r"C:\Program Files\WSL\wsl.exe"
IMAGE = "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
MEMORY = 268435456

PROBE = r"""
import errno
import json
import os
import pathlib
import resource
import signal
import socket
import subprocess
import sys

assert sys.version_info[:2] == (3, 12)
assert os.getuid() == os.geteuid() == 65534
assert not pathlib.Path('/mnt/c').exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
status_lines = pathlib.Path('/proc/self/status').read_text().splitlines()
status = dict(line.split(':', 1) for line in status_lines)
assert int(status['CapEff'].strip(), 16) == 0
assert int(status['CapBnd'].strip(), 16) == 0
assert status['NoNewPrivs'].strip() == '1'
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1048576, 1048576)
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (64, 64)
assert resource.getrlimit(resource.RLIMIT_NPROC) == (8, 8)
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
try:
    pathlib.Path('/var/tmp/iris-readonly-probe').write_text('probe')
except OSError as exc:
    assert exc.errno == errno.EROFS, exc
else:
    raise AssertionError('Root filesystem was writable')
probe = pathlib.Path('/tmp/iris-noexec-probe')
probe.write_text('#!/bin/sh\nexit 0\n')
probe.chmod(0o700)
try:
    subprocess.run([str(probe)], check=True, timeout=2)
except PermissionError as exc:
    assert exc.errno == errno.EACCES
else:
    raise AssertionError('Executable tmpfs')
try:
    socket.create_connection(('151.101.194.132', 443), timeout=1)
except OSError as exc:
    network_errno = exc.errno
else:
    raise AssertionError('External network was reachable')
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
try:
    pathlib.Path('/tmp/iris-fsize-probe').write_bytes(b'x' * 1048577)
except OSError as exc:
    assert exc.errno == errno.EFBIG, exc
else:
    raise AssertionError('File-size rlimit was not enforced')
assert sum(v * v for v in [1, 2, 3, 4] if v % 2 == 0) == 20
print(json.dumps({'probe': 'filesystem_network_privileges', 'python': sys.version.split()[0],
                  'uid': os.getuid(), 'network_errno': network_errno, 'passed': True}), flush=True)
sys.stdin.readline()
"""


FORK_PROBE = r"""
import errno
import json
import os
import signal
import sys

children = []
blocked = False
try:
    for _ in range(12):
        try:
            pid = os.fork()
        except OSError as exc:
            assert exc.errno == errno.EAGAIN, exc
            blocked = True
            break
        if pid == 0:
            signal.pause()
            os._exit(0)
        children.append(pid)
finally:
    for pid in children:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
assert blocked and len(children) <= 7, len(children)
print(json.dumps({'probe': 'bounded_process_creation', 'children': len(children),
                  'passed': True}), flush=True)
sys.stdin.readline()
"""

MEMORY_PROBE = """
block = bytearray(384 * 1024 * 1024)
for index in range(0, len(block), 4096):
    block[index] = 1
raise AssertionError('Allocation exceeded the container memory budget')
"""

TIMEOUT_PROBE = "while True: pass"


def linux(*args: str, timeout: float = 30) -> str:
    result = subprocess.run(
        [WSL, "--distribution", "IrisCodeEval", "--user", "root", "--cd", "/", "--exec", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    return result.stdout.strip()


def inspect(container: str) -> dict[str, Any]:
    return json.loads(linux("docker", "inspect", container))[0]  # type: ignore[no-any-return]


def create(source: str) -> str:
    name = "iris-isolation-probe-" + uuid.uuid4().hex
    container = linux(
        "docker",
        "create",
        "--name",
        name,
        "--label",
        "iris.operational-isolation-probe=true",
        "--pull=never",
        "--runtime=runsc",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=256m",
        "--memory-swap=256m",
        "--cpus=0.5",
        "--pids-limit=128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--shm-size=8m",
        "--ulimit",
        "fsize=1048576:1048576",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        "nproc=8:8",
        "--ulimit",
        "core=0:0",
        "--log-driver=none",
        "--restart=no",
        "--interactive",
        "--entrypoint=python3",
        IMAGE,
        "-I",
        "-S",
        "-B",
        "-u",
        "-c",
        source,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", container), container
    return container


def verify_configuration(container: str, image_id: str) -> None:
    state = inspect(container)
    host = state["HostConfig"]
    assert state["Image"] == image_id
    assert state["Config"]["User"] == "65534:65534"
    assert not state["Config"].get("Volumes")
    assert host["Runtime"] == "runsc" and host["NetworkMode"] == "none"
    assert host["ReadonlyRootfs"] is True and host["Privileged"] is False
    assert not host.get("Binds") and not host.get("Mounts") and not host.get("Devices")
    assert not host.get("DeviceRequests") and not host.get("PortBindings")
    assert host["CapDrop"] == ["ALL"] and not host.get("CapAdd")
    assert "no-new-privileges" in host["SecurityOpt"]
    assert host["Memory"] == host["MemorySwap"] == MEMORY
    assert host["NanoCpus"] == 500000000 and host["PidsLimit"] == 128
    assert host["CgroupnsMode"] == "private" and host["PidMode"] == ""
    assert host["IpcMode"] == "private"
    assert host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"}
    assert host["ShmSize"] == 8388608 and host["LogConfig"]["Type"] == "none"
    assert host["RestartPolicy"]["Name"] == "no"


def verify_live_cgroup(container: str) -> dict[str, str]:
    state = inspect(container)
    assert state["State"]["Running"] is True
    pid = state["State"]["Pid"]
    assert type(pid) is int and pid > 0
    executable = linux("readlink", f"/proc/{pid}/exe")
    assert executable == "/usr/bin/gvisor-bin/gvisor_sentry", executable
    group = linux("cat", f"/proc/{pid}/cgroup")
    expected = f"0::/system.slice/docker-{container}.scope"
    assert group == expected, group
    root = f"/sys/fs/cgroup/system.slice/docker-{container}.scope"
    values = {
        field: linux("cat", root + "/" + field)
        for field in (
            "memory.max",
            "memory.swap.max",
            "cpu.max",
            "pids.max",
        )
    }
    assert values == {
        "memory.max": str(MEMORY),
        "memory.swap.max": "0",
        "cpu.max": "50000 100000",
        "pids.max": "128",
    }, values
    return {"sentry_executable": executable, **values}


def run_live_probe(source: str, image_id: str) -> dict[str, Any]:
    container = create(source)
    process: subprocess.Popen[str] | None = None
    try:
        verify_configuration(container, image_id)
        process = subprocess.Popen(
            [
                WSL,
                "--distribution",
                "IrisCodeEval",
                "--user",
                "root",
                "--cd",
                "/",
                "--exec",
                "docker",
                "start",
                "--attach",
                "--interactive",
                container,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        deadline = time.monotonic() + 15
        while not inspect(container)["State"]["Running"]:
            if process.poll() is not None or time.monotonic() >= deadline:
                out, err = process.communicate(timeout=5)
                raise RuntimeError(f"Container failed to start: {out} {err}")
            time.sleep(0.1)
        cgroups = verify_live_cgroup(container)
        out, err = process.communicate(input="finish\n", timeout=15)
        assert process.returncode == 0, (process.returncode, inspect(container)["State"], out, err)
        result = json.loads(out)
        assert result["passed"] is True
        result.update({"image": IMAGE, "image_id": image_id, "host_cgroup": cgroups})
        return result  # type: ignore[no-any-return]
    finally:
        # Only the exact newly created, labelled disposable probe container.
        owned = inspect(container)
        assert owned["Config"]["Labels"].get("iris.operational-isolation-probe") == "true"
        linux("docker", "rm", "--force", container)
        if process is not None and process.poll() is None:
            process.communicate(timeout=10)


def run_failure_probe(source: str, image_id: str, *, expected_timeout: bool) -> dict[str, Any]:
    container = create(source)
    timed_out = False
    try:
        verify_configuration(container, image_id)
        try:
            if expected_timeout:
                linux("docker", "start", container)
                verify_live_cgroup(container)
                # Observe a detached job: killing an attached WSL client may
                # forward a signal and end the container before explicit kill.
                linux("docker", "wait", container, timeout=3)
            else:
                linux("docker", "start", "--attach", container, timeout=15)
        except subprocess.TimeoutExpired:
            if not expected_timeout:
                raise
            assert inspect(container)["State"]["Running"] is True
            linux("docker", "kill", container)
            timed_out = True
        except subprocess.CalledProcessError as exc:
            if expected_timeout or exc.returncode != 137:
                raise
        state = inspect(container)["State"]
        assert state["Running"] is False and state["ExitCode"] == 137, state
        if expected_timeout:
            assert timed_out and state["OOMKilled"] is False
        else:
            assert state["OOMKilled"] is True, state
        return {
            "probe": "supervised_timeout" if expected_timeout else "memory_limit",
            "passed": True,
            "exit_code": state["ExitCode"],
            "oom_killed": state["OOMKilled"],
        }
    finally:
        owned = inspect(container)
        assert owned["Config"]["Labels"].get("iris.operational-isolation-probe") == "true"
        linux("docker", "rm", "--force", container)


def main() -> None:
    if not __debug__:
        raise RuntimeError("Operational verification requires Python assertions enabled")
    image = json.loads(linux("docker", "image", "inspect", IMAGE))[0]
    assert image["Os"] == "linux" and image["Architecture"] == "amd64"
    assert not image["Config"].get("Volumes")
    assert linux("docker", "info", "--format", "{{.DefaultRuntime}}") == "runsc"
    assert linux("dpkg", "--verify", "runsc") == ""
    for source in (PROBE, FORK_PROBE):
        print(json.dumps(run_live_probe(source, image["Id"]), sort_keys=True), flush=True)
    print(
        json.dumps(run_failure_probe(MEMORY_PROBE, image["Id"], expected_timeout=False)), flush=True
    )
    print(
        json.dumps(run_failure_probe(TIMEOUT_PROBE, image["Id"], expected_timeout=True)), flush=True
    )


if __name__ == "__main__":
    main()
