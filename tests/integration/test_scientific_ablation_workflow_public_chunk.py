"""Actual public first/resume chunks; observation never injects kernel arguments.

Tiny C1/C5 original workflows with zero, one empty or two nonempty controls.
This augments, never substitutes for, the complete prior641 native regression.
No generated Python, pretrained download, durable CLI ACK or science claim.
"""

from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_workflow as workflow
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, Geometry, prepare
from tests.integration.test_scientific_ablation_execution import _tree

pytestmark = pytest.mark.integration


def _binding(phase: ScientificAblationWorkflowPhaseV1) -> tuple[ModuleType, str, str]:
    if phase.kind == "capture":
        return (
            (
                capture,
                "build_scientific_ablation_capture_plan_v1",
                "capture_scientific_ablation_heldout_v1",
            )
            if phase.stream_index is None
            else (
                controlled_capture,
                "build_scientific_ablation_controlled_capture_plan_v1",
                "capture_scientific_ablation_controlled_heldout_v1",
            )
        )
    assert phase.kind == "score"
    return (
        (
            score,
            "build_scientific_ablation_evaluation_plan_v2",
            "evaluate_scientific_ablation_capture_resumable_v2",
        )
        if phase.stream_index is None
        else (
            controlled_score,
            "build_scientific_ablation_controlled_evaluation_plan_v1",
            "evaluate_scientific_ablation_controlled_capture_resumable_v1",
        )
    )


def _first_request(
    job: workflow.IrisScientificAblationWorkflowV1, phase: ScientificAblationWorkflowPhaseV1
) -> Chunk:
    module, builder, _ = _binding(phase)
    if phase.kind == "capture":
        # Genuine trusted source planning may reconstruct an original native
        # owner. It does NOT materialize a capture plan/row directory in advance.
        heldout = steps._owner(job._seal.store.plan, phase, job._seal.factory)
        plan = getattr(module, builder)(heldout, expected_input_sha256=heldout.manifest.sha256)
    else:
        prefix = workflow._prefix(job._seal)
        plan = getattr(module, builder)(**steps._score_args(job._seal.store, prefix[-1]))
    assert len(plan.payload.rows) == 2
    assert not (steps._kernel(job._seal.store, phase) / module._STORE / plan.sha256).exists()
    return Chunk(
        kernel_plan_sha256=plan.sha256,
        acknowledgement_head_sha256=module._genesis(plan.sha256),
        acknowledged_rows=0,
        total_rows=2,
        rows_per_call=1,
        stop_row_index=1,
    )


def _snapshot(path: Path) -> tuple[bool, dict[str, bytes]]:
    return path.exists(), _tree(path) if path.exists() else {}


def _reject_before_claim(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    request: Chunk,
    *,
    rows_per_call: int = 1,
) -> None:
    before = _snapshot(job._seal.path)
    anchors = job.progress.completed_phase_sha256
    with (
        patch.object(
            ScientificAblationWorkflowStoreV1, "claim", side_effect=AssertionError("No outer claim")
        ) as claim,
        patch.object(steps, "_owner", side_effect=AssertionError("No factory")) as factory,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(ValueError),
    ):
        job.advance_expected(
            expected_phase=phase,
            expected_completed_phase_sha256=anchors,
            rows_per_call=rows_per_call,
            expected_chunk=request,
        )
    assert claim.call_count == factory.call_count == graph.call_count == forward.call_count == 0
    assert job._state == "idle" and _snapshot(job._seal.path) == before
    assert job.progress.completed_phase_sha256 == anchors


def _reject_fresh_capture_plan_at_original_kernel(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    correct: Chunk,
) -> None:
    module, _, operation = _binding(phase)
    wrong = "0" * 64 if correct.kernel_plan_sha256 != "0" * 64 else "1" * 64
    request = Chunk.model_validate(
        correct.model_dump()
        | dict(kernel_plan_sha256=wrong, acknowledgement_head_sha256=module._genesis(wrong))
    )
    anchors = job.progress.completed_phase_sha256
    kernel_root = steps._kernel(job._seal.store, phase)
    assert not kernel_root.exists()
    with (
        patch.object(steps, "_owner", wraps=steps._owner) as owner,
        patch.object(module, operation, wraps=getattr(module, operation)) as original,
        patch.object(module, "_stage", side_effect=AssertionError("No row or plan stage")) as stage,
        pytest.raises(IrisScientificAblationWorkflowStorageError),
    ):
        job.advance_expected(
            expected_phase=phase,
            expected_completed_phase_sha256=anchors,
            rows_per_call=1,
            expected_chunk=request,
        )
    assert owner.call_count == original.call_count == 1 and stage.call_count == 0
    assert original.call_args.kwargs["expected_chunk"] == request
    assert not kernel_root.exists() and job._state == "idle"
    assert job.progress.completed_phase_sha256 == anchors
    assert (job._seal.path / f"phase-{phase.index:03d}.attempt.json").is_file()
    # The ORIGINAL capture/score attempt permits an explicit resumable request.
    # Do not mislabel this as non-resumable diagnostics/writer UNCERTAIN behavior.


@pytest.mark.parametrize("geometry", (GEOMETRIES[0], GEOMETRIES[2]), ids=("c1", "c5"))
@pytest.mark.parametrize("controls", ("zero", "single-empty", "paired-nonempty"))
def test_actual_public_first_resume_and_stale_chunks_complete_original_workflow(
    tmp_path: Path, geometry: Geometry, controls: str
) -> None:
    case = prepare(tmp_path / "case", geometry)
    control_plans = (
        ()
        if controls == "zero"
        else (WorkspaceInterventionPlanV1(interventions=()),)
        if controls == "single-empty"
        else case.plan.payload.controls
    )
    plan = build_scientific_ablation_workflow_plan_v1(
        case.plan.payload.model_copy(update={"controls": control_plans})
    )
    phases = scientific_ablation_workflow_phases_v1(plan)
    job = workflow.create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow", plan=plan, make_fresh_run=case.factory
    )
    records: list[dict[str, Any]] = []
    with patch(
        "iris.application.synthetic_scoring_v2.execute_python_job_v2",
        side_effect=AssertionError("No generated Python on host"),
    ) as candidate:
        progress = job.progress
        for phase in phases:
            if phase.kind not in ("capture", "score"):
                progress = job.advance_expected(
                    expected_phase=phase,
                    expected_completed_phase_sha256=progress.completed_phase_sha256,
                    rows_per_call=1,
                )
                continue
            module, _, operation = _binding(phase)
            request = _first_request(job, phase)
            for changes in (
                {"acknowledgement_head_sha256": "f" * 64},
                {"acknowledged_rows": 1, "stop_row_index": 2},
                {"total_rows": 3},
            ):
                _reject_before_claim(
                    job, phase, Chunk.model_validate(request.model_dump() | changes)
                )
            _reject_before_claim(job, phase, request, rows_per_call=2)
            if phase.kind == "capture":
                _reject_fresh_capture_plan_at_original_kernel(job, phase, request)
            else:
                wrong = "0" * 64
                assert request.kernel_plan_sha256 != wrong
                _reject_before_claim(
                    job,
                    phase,
                    Chunk.model_validate(
                        request.model_dump()
                        | dict(
                            kernel_plan_sha256=wrong,
                            acknowledgement_head_sha256=module._genesis(wrong),
                        )
                    ),
                )
            for row in range(2):
                # wraps observes the ORIGINAL arguments and executes the real
                # function. Unlike prior tests, it adds/rewrites NO chunk intent.
                with patch.object(module, operation, wraps=getattr(module, operation)) as original:
                    progress = job.advance_expected(
                        expected_phase=phase,
                        expected_completed_phase_sha256=progress.completed_phase_sha256,
                        rows_per_call=1,
                        expected_chunk=request,
                    )
                assert original.call_count == 1
                assert original.call_args.kwargs["expected_chunk"] == request
                observed = job.inspect_kernel(
                    expected_phase=phase, expected_kernel_plan_sha256=request.kernel_plan_sha256
                )
                assert observed.acknowledged_rows == row + 1 and observed.total_rows == 2
                assert observed.phase_is_acknowledged is (row == 1)
                assert (observed.kernel_complete_sha256 is not None) is (row == 1)
                assert observed.completed_phase_sha256 == progress.completed_phase_sha256
                records.append(
                    dict(
                        request=request.model_dump(mode="json"),
                        observed=observed.model_dump(mode="json"),
                    )
                )
                if row == 0:
                    _reject_before_claim(job, phase, request)
                    request = Chunk.model_validate(
                        request.model_dump()
                        | dict(
                            acknowledgement_head_sha256=observed.acknowledgement_head_sha256,
                            acknowledged_rows=1,
                            stop_row_index=2,
                        )
                    )
        assert candidate.call_count == 0
    assert progress.complete and len(progress.completed_phase_sha256) == 6 + 2 * len(control_plans)
    assert len(records) == 4 * (1 + len(control_plans))
    (tmp_path / "actual-public-chunk-workflow-witness.json").write_bytes(
        canonical_json_file_bytes(
            asdict(geometry)
            | dict(
                kind="iris-public-chunk-full-native-workflow-witness-v1",
                controls=controls,
                plan_sha256=plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                actual_public_kernel_requests=records,
                first_capture_plan_wrong_digest_rejected_after_original_owner_before_row_stage=True,
                first_and_resumed_chunks_forwarded_by_public_api=True,
                kernel_call_observation_only_no_argument_injection=True,
                stale_and_wrong_genesis_rows_total_budget_rejected_before_claim_factory_graph_forward=True,
                actual_os_cold_process=False,
                independently_persisted_pre_post_file_identity_snapshots=False,
                durable_cli_command_ack=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
