"""Authored source-only cold capture consumer; never trains or regenerates source.

Job SHA comes independently via argv. Only disk source/data/capture plus trusted
tiny original-factory code cross this boundary. No expected outputs are supplied.
Hard exit after row rename intentionally bypasses Python finally/lock release.
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
        raise ValueError("Explicit source package root/job/job SHA required")
    package_root = Path(sys.argv[1]).resolve(strict=True)
    job_path = Path(sys.argv[2]).resolve(strict=True)
    if job_path.stat().st_size > 1024**2:
        raise ValueError("Bounded authored worker job required")
    content = job_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sys.argv[3]:
        raise ValueError("Independent worker job SHA differs")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(package_root))

    import torch
    from pydantic import Field

    from iris.application import scientific_ablation_controlled_capture as capture_module
    from iris.application.scientific_ablation_controlled_capture import (
        build_scientific_ablation_controlled_capture_plan_v1,
        capture_scientific_ablation_controlled_heldout_v1,
    )
    from iris.application.scientific_ablation_controlled_heldout import (
        IrisScientificAblationControlledHeldoutV1,
        build_scientific_ablation_controlled_heldout_v1,
    )
    from iris.application.scientific_ablation_generation_source import (
        read_scientific_ablation_generation_source_v1,
    )
    from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
    from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
    from iris.application.scientific_command import ScientificParquetInputV1
    from iris.application.scientific_optimization import _same_runtime
    from iris.artifacts.scientific_ablation_controlled_heldout_v1 import (
        IrisScientificAblationControlledHeldoutRowV1,
    )
    from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
    from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
    from iris.interventions.policy import WorkspaceInterventionPlanV1
    from iris.utils.directory_publication import rename_directory_with_windows_retry
    from iris.utils.seeds import capture_runtime_state
    from tests.integration import test_scientific_ablation_generation_source as producer
    from tests.integration.test_scientific_ablation_controlled_heldout import _context

    class Job(_Strict):
        action: Literal["resume", "die_after_row_commit"]
        lifecycle_path: str = Field(min_length=1, max_length=4096)
        inference_path: str = Field(min_length=1, max_length=4096)
        lifecycle_sha256: Digest
        inference_sha256: Digest
        checkpoint_sha256: Digest
        source_sha256: Digest
        input_sha256: Digest
        plan_sha256: Digest
        test: ScientificParquetInputV1
        test_base: str = Field(min_length=1, max_length=4096)
        root: str = Field(min_length=1, max_length=4096)
        routing: Literal["learned", "random", "uniform"]
        goal: Literal["enabled", "disabled"]
        broadcast: Literal["enabled", "disabled"]
        decoding: ScientificDecodingPolicyV1
        control: WorkspaceInterventionPlanV1

    job = Job.model_validate_json(content)
    torch.set_num_threads(1)
    calls: list[int] = []
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(producer, "_prepared", side_effect=AssertionError("No source regeneration")),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No pretrained model load"),
        ),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer load"),
        ),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ),
        patch(
            "iris.application.scientific_ablation_request.build_scientific_ablation_request_v1",
            side_effect=AssertionError("No ordinary owner adoption"),
        ),
        patch.object(
            IrisScientificAblationRunV1, "step", side_effect=AssertionError("No training")
        ),
        patch.object(
            IrisScientificAblationOptimizerV1,
            "step",
            side_effect=AssertionError("No optimizer step"),
        ),
    ):
        source = read_scientific_ablation_generation_source_v1(
            lifecycle_path=Path(job.lifecycle_path),
            inference_path=Path(job.inference_path),
            expected_lifecycle_sha256=job.lifecycle_sha256,
            expected_inference_sha256=job.inference_sha256,
            expected_source_checkpoint_sha256=job.checkpoint_sha256,
        )
        if source.manifest.sha256 != job.source_sha256:
            raise ValueError("Independently trusted terminal source differs before factory")
        context = _context(
            source,
            routing=job.routing,
            goal=job.goal,
            broadcast=job.broadcast,
            sampling=job.decoding.strategy == "sampling",
        )
        heldout = build_scientific_ablation_controlled_heldout_v1(
            context,
            expected_reconstruction_sha256=context.reconstruction.sha256,
            test=job.test,
            base=Path(job.test_base),
            decoding=job.decoding,
            control=job.control,
        )
        if heldout.manifest.sha256 != job.input_sha256:
            raise ValueError("Independently trusted held-out input differs")
        plan = build_scientific_ablation_controlled_capture_plan_v1(
            heldout, expected_input_sha256=job.input_sha256
        )
        if plan.sha256 != job.plan_sha256:
            raise ValueError("Independently trusted full capture plan differs")
        original = IrisScientificAblationControlledHeldoutV1.execute_row

        def execute(
            owner: IrisScientificAblationControlledHeldoutV1, index: int
        ) -> IrisScientificAblationControlledHeldoutRowV1:
            calls.append(index)
            return original(owner, index)

        def rename(stage: Path, target: Path) -> None:
            rename_directory_with_windows_retry(stage, target)
            if job.action == "die_after_row_commit" and target.name.startswith("row-"):
                print(
                    json.dumps(
                        {
                            "kind": "iris-controlled-capture-worker-hard-exit-v1",
                            "pid": os.getpid(),
                            "executed_indices": calls,
                            "committed_row": target.name,
                            "plan_sha256": plan.sha256,
                        }
                    ),
                    flush=True,
                )
                os._exit(87)

        with (
            patch.object(IrisScientificAblationControlledHeldoutV1, "execute_row", new=execute),
            patch.object(capture_module, "rename_directory_with_windows_retry", side_effect=rename),
        ):
            result = capture_scientific_ablation_controlled_heldout_v1(
                heldout,
                root=Path(job.root),
                expected_input_sha256=job.input_sha256,
                expected_plan_sha256=job.plan_sha256,
                maximum_storage_bytes=32 * 1024**2,
                minimum_free_bytes=0,
            )
        if result.complete is None or not _same_runtime(
            caller, capture_runtime_state(torch.device("cpu"))
        ):
            raise ValueError("Worker lacks complete coverage/caller restoration")
        print(
            json.dumps(
                {
                    "kind": "iris-controlled-capture-worker-complete-v1",
                    "pid": os.getpid(),
                    "executed_indices": calls,
                    "plan_sha256": plan.sha256,
                    "capture_sha256": result.complete.sha256,
                    "row_sha256": [row.sha256 for row in result.rows],
                    "ack_sha256": [ack.sha256 for ack in result.acknowledgements],
                    "target_token_count": result.complete.payload.target_token_count,
                    "target_nll_sum": result.complete.payload.target_nll_sum,
                    "caller_restored": True,
                    "source_regenerated": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
