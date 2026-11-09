"""Actual section15-profile probes; never execute these programs on the host."""

import asyncio
import json
import os
import sys
from hashlib import sha256

import pytest

from iris.integrations import python_sandbox as runtime
from iris.integrations.python_sandbox_v2 import execute_python_job_v2
from iris.utils.hashing import canonical_json_bytes
from tests.unit.test_python_sandbox_v2 import POLICY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="requires explicit IRIS_RUN_CODE_ISOLATION_SMOKE=1 and verified IrisCodeEval",
    ),
]


def empty_inventory() -> None:
    assert (
        asyncio.run(
            runtime._docker(
                "ps", "--all", "--filter", "name=^/" + runtime._NAME + "$", "--format", "{{.ID}}"
            )
        )
        == ""
    )


@pytest.mark.parametrize(
    "probe",
    [
        "environment",
        "filesystem_network",
        "memory_above_v1",
        "timeout",
        "output_limit",
        "oom",
        "syntax",
    ],
)
def test_real_source_profile(probe: str) -> None:
    sources = {
        "environment": (
            "import os, resource\n"
            "def solve(values):\n"
            "    assert dict(os.environ) == {}\n"
            "    assert open('/proc/self/environ', 'rb').read() == b''\n"
            "    assert os.getuid() == 65534\n"
            "    assert resource.getrlimit(resource.RLIMIT_NPROC) == (32, 32)\n"
            "    return sum(values)\n"
        ),
        "filesystem_network": (
            "import os, socket, tempfile, errno\n"
            "def solve(values):\n"
            "    assert not os.path.exists('/mnt/c')\n"
            "    try:\n        open('/iris-forbidden', 'w')\n"
            "    except OSError as error:\n"
            "        assert error.errno in (errno.EROFS, errno.EACCES)\n"
            "    else:\n        raise RuntimeError('root writable')\n"
            "    with tempfile.TemporaryFile(dir='/tmp') as stream:\n"
            "        stream.write(b'ok'); stream.seek(0); assert stream.read() == b'ok'\n"
            "    with socket.socket() as stream:\n"
            "        stream.settimeout(0.1)\n"
            "        try:\n            stream.connect(('192.0.2.1', 80))\n"
            "        except OSError:\n            pass\n"
            "        else:\n            raise RuntimeError('network connected')\n"
            "    return sum(values)\n"
        ),
        "memory_above_v1": (
            "data = bytearray(300 * 1024**2)\ndef solve(values): return sum(values)\n"
        ),
        "timeout": "def solve(values):\n    while True: pass\n",
        "output_limit": "import os\nwhile True: os.write(2, b'x' * 8192)\n",
        "oom": "data = bytearray(2 * 1024**3)\ndef solve(values): return sum(values)\n",
        "syntax": "def solve(values) broken\n",
    }
    inputs = [[], [1], [1, 2], [-5, 8]]
    request = canonical_json_bytes(
        {"kind": "python_integer_list_v1", "source": sources[probe], "inputs": inputs}
    )
    observation = asyncio.run(execute_python_job_v2(request_bytes=request, policy=POLICY))
    empty_inventory()
    assert observation.policy == POLICY
    assert observation.submitted_input_sha256 == sha256(request).hexdigest()
    assert observation.worker_sha256 == sha256(runtime._worker("python_worker_v2.txt")).hexdigest()
    if probe == "timeout":
        assert observation.process.termination == "timeout" and not observation.oom_killed
    elif probe == "output_limit":
        assert observation.process.termination == "output_limit"
        assert (
            len(observation.process.stdout) + len(observation.process.stderr)
            <= POLICY.max_output_bytes
        )
    elif probe == "oom":
        assert observation.oom_killed and observation.container_exit_code != 0
    elif probe == "syntax":
        assert json.loads(observation.process.stdout) == {"kind": "syntax_error"}
        assert observation.container_exit_code == 0
    else:
        assert observation.process.termination == "exited" and observation.container_exit_code == 0
        assert not observation.oom_killed
        result = json.loads(observation.process.stdout)
        assert result == {
            "kind": "observations",
            "cases": [
                {"kind": "value", "result": sum(values), "input_after": values} for values in inputs
            ],
        }
