"""Authored model-free cold score consumer; candidates run only through gVisor.

The job digest arrives independently in argv. A real hard exit after score rename
bypasses Python lock cleanup. Original scoring/isolated execution are observed,
never replaced with fabricated measurements or receipts.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import patch


def main() -> None:
    if len(sys.argv) != 4:
        raise ValueError("Explicit source package root/job/independent SHA required")
    package_root = Path(sys.argv[1]).resolve(strict=True)
    job_path = Path(sys.argv[2]).resolve(strict=True)
    if job_path.stat().st_size > 1024**2:
        raise ValueError("Bounded worker job required")
    content = job_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sys.argv[3]:
        raise ValueError("Independent cold-worker job SHA differs")
    sys.path.insert(0, str(package_root))

    import torch
    from pydantic import Field

    from iris.application import scientific_ablation_evaluation_ledger as ledger
    from iris.application import synthetic_scoring_v2 as scoring
    from iris.application.scientific_ablation_evaluation_v2 import _EvaluationInputs, _score_row
    from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
    from iris.application.scientific_command import ScientificParquetInputV1
    from iris.application.scientific_optimization import _same_runtime
    from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
    from iris.artifacts.scientific_ablation_evaluation_v2 import ScientificAblationScoredRowV2
    from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
    from iris.integrations import python_sandbox as runtime
    from iris.integrations.python_sandbox_v2 import (
        PythonSandboxObservationV2,
        execute_python_job_v2,
    )
    from iris.utils.directory_publication import rename_directory_with_windows_retry
    from iris.utils.seeds import capture_runtime_state

    class Job(_Strict):
        action: Literal["resume", "die_after_row_commit"]
        root: str = Field(min_length=1, max_length=4096)
        capture_path: str = Field(min_length=1, max_length=4096)
        lifecycle_path: str = Field(min_length=1, max_length=4096)
        inference_path: str = Field(min_length=1, max_length=4096)
        inputs_root: str = Field(min_length=1, max_length=4096)
        expected_evaluation_plan_sha256: Digest
        expected_plan_sha256: Digest
        expected_capture_sha256: Digest
        training: ScientificParquetInputV1
        validation: ScientificParquetInputV1
        test: ScientificParquetInputV1
        python_policy: PythonSandboxPolicyV2 | None

    job = Job.model_validate_json(content)
    torch.set_num_threads(1)
    calls: list[int] = []
    sandbox_calls: list[int] = []
    phases: list[dict[str, object]] = []
    original_score = _score_row
    original_execute = execute_python_job_v2
    original_linux = runtime._linux
    original_rename = rename_directory_with_windows_retry
    caller = capture_runtime_state(torch.device("cpu"))

    def score(inputs: _EvaluationInputs, index: int) -> ScientificAblationScoredRowV2:
        calls.append(index)
        return original_score(inputs, index)

    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV2
    ) -> PythonSandboxObservationV2:
        sandbox_calls.append(calls[-1])
        return await original_execute(request_bytes=request_bytes, policy=policy)

    async def linux(*commands: str) -> str:
        event: dict[str, object] = {
            "phase": commands[3] if commands[0] == "/usr/bin/docker" else Path(commands[0]).name,
        }
        phases.append(event)
        try:
            result = await original_linux(*commands)
        except BaseException as error:
            event["passed"] = False
            event["type"] = type(error).__name__
            raise
        event["passed"] = True
        return result

    def rename(source: Path, target: Path) -> None:
        original_rename(source, target)
        if job.action == "die_after_row_commit" and target.name == "score-00000":
            print(
                json.dumps(
                    {
                        "kind": "iris-score-worker-hard-exit-v2",
                        "pid": os.getpid(),
                        "scored_indices": calls,
                        "sandbox_indices": sandbox_calls,
                        "committed_row": target.name,
                        "evaluation_plan_sha256": job.expected_evaluation_plan_sha256,
                        "linux_phases": phases,
                        "host_candidate_execution": False,
                    }
                ),
                flush=True,
            )
            os._exit(87)

    with (
        patch.object(
            torch.nn.Module, "__init__", side_effect=AssertionError("No model construction")
        ),
        patch.object(
            IrisScientificAblationHeldoutV1,
            "execute_row",
            side_effect=AssertionError("No regeneration"),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No backbone loading"),
        ),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer loading"),
        ),
        patch.object(ledger, "_score_row", side_effect=score),
        patch.object(scoring, "execute_python_job_v2", side_effect=execute),
        patch.object(runtime, "_linux", side_effect=linux),
        patch.object(ledger, "rename_directory_with_windows_retry", side_effect=rename),
    ):
        result = ledger.evaluate_scientific_ablation_capture_resumable_v2(
            root=Path(job.root),
            capture_path=Path(job.capture_path),
            lifecycle_path=Path(job.lifecycle_path),
            inference_path=Path(job.inference_path),
            inputs_root=Path(job.inputs_root),
            expected_evaluation_plan_sha256=job.expected_evaluation_plan_sha256,
            expected_plan_sha256=job.expected_plan_sha256,
            expected_capture_sha256=job.expected_capture_sha256,
            training=job.training,
            validation=job.validation,
            test=job.test,
            python_policy=job.python_policy,
            minimum_free_bytes=0,
        )
    if (
        result.complete is None
        or result.evaluation is None
        or not _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    ):
        raise ValueError("Cold worker lacks full owned coverage/caller restoration")
    print(
        json.dumps(
            {
                "kind": "iris-score-worker-complete-v2",
                "pid": os.getpid(),
                "scored_indices": calls,
                "sandbox_indices": sandbox_calls,
                "evaluation_plan_sha256": result.plan.sha256,
                "evaluation_sha256": result.evaluation.sha256,
                "complete_sha256": result.complete.sha256,
                "row_sha256": [row.sha256 for row in result.rows],
                "ack_sha256": [ack.sha256 for ack in result.acknowledgements],
                "caller_restored": True,
                "model_constructed": False,
                "host_candidate_execution": False,
                "linux_phases": phases,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
