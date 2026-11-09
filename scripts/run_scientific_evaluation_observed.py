"""Run the unchanged237 gate with redacted original Linux-call diagnostics."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iris.integrations import python_sandbox as runtime


class Observation:
    def __init__(self) -> None:
        self.nodeid = "collection"
        self.events: list[dict[str, object]] = []
        self.original = runtime._linux

    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        self.nodeid = item.nodeid

    async def linux(self, *commands: str) -> str:
        start = time.monotonic()
        event: dict[str, object] = {
            "nodeid": self.nodeid,
            "phase": commands[3] if commands[0] == "/usr/bin/docker" else Path(commands[0]).name,
        }
        self.events.append(event)
        try:
            value = await self.original(*commands)
            event["passed"] = True
            return value
        except BaseException as error:
            event["passed"] = False
            event["type"] = type(error).__name__
            event["frames"] = [
                {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(error.__traceback__)
            ]
            raise
        finally:
            event["elapsed_ms"] = round((time.monotonic() - start) * 1000, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.observation.exists():
        raise ValueError("Refuse to replace an existing observation")
    observer = Observation()
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    with patch.object(runtime, "_linux", observer.linux):
        result = pytest.main(pytest_args, plugins=[observer])
    payload = {
        "kind": "iris-r015-observed-evaluation-runtime-v1",
        "pytest_exit_code": int(result),
        "guards_limits_worker_unchanged": True,
        "host_candidate_execution": False,
        "events": observer.events,
    }
    # This is an authored runtime evidence producer, not a source/code editor.
    with args.observation.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        file.write("\n")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
