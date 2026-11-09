"""Observe unchanged source-v2 isolation calls; no candidate execution on host."""

import argparse
import asyncio
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from iris.application.synthetic_scoring_v2 import score_synthetic_answer_v2
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.data.parquet import decode_synthetic_parquet
from iris.integrations import python_sandbox as runtime


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    from hashlib import sha256

    content = args.parquet.read_bytes()
    if sha256(content).hexdigest() != args.sha256:
        raise ValueError("Independent original test file SHA differs")
    rows = decode_synthetic_parquet(
        content, max_rows=4, max_file_bytes=1024**2, max_logical_bytes=1024**2
    )
    row = next(row for row in rows if row.executable)
    policy = PythonSandboxPolicyV2(
        profile="iris_wsl_gvisor_python312_source_v2", wall_time_ms=5000, max_output_bytes=1048576
    )
    events: list[dict[str, object]] = []
    original = runtime._linux

    async def observed(*commands: str) -> str:
        start = time.monotonic()
        phase = commands[3] if commands[0] == "/usr/bin/docker" else Path(commands[0]).name
        event: dict[str, object] = {"phase": phase}
        events.append(event)
        try:
            result = await original(*commands)
            event["passed"] = True
            return result
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

    result: dict[str, object] = {
        "kind": "iris-r015-source-v2-isolation-observation-v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "parquet_sha256": args.sha256,
        "candidate_scope": "controlled_reference_not_model_quality",
        "guards_unchanged": True,
        "host_candidate_execution": False,
        "events": events,
    }
    # The synchronous scorer owns its own asyncio loop. Run only this trusted
    # caller in a thread; original isolation code/limits/worker remain unchanged.
    with patch.object(runtime, "_linux", observed):
        try:
            score = await asyncio.to_thread(
                score_synthetic_answer_v2,
                row,
                output_text=row.expected_response,
                python_policy=policy,
            )
            result.update(
                passed=score.outcome == "correct",
                reason=score.reason,
                case_count=score.case_count,
                passed_case_count=score.passed_case_count,
            )
        except Exception as error:
            result.update(passed=False, error_type=type(error).__name__)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
