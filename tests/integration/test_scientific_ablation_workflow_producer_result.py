"""Real original-return/native storage retry and all four post-return kernel races.

No result injection, model substitute, native remeasurement on retry or durable
ACK claim. Reopened in-process handles are NOT a genuine cold OS-process test.
"""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as workflow
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1 as Kernel,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, Geometry, prepare
from tests.integration.test_scientific_ablation_execution import _tree
from tests.integration.test_scientific_ablation_workflow_public_chunk import (
    _binding,
    _first_request,
)

pytestmark = pytest.mark.integration
RACES = ("ordinary-capture", "ordinary-score", "controlled-capture", "controlled-score")


def _wire(result: workflow.ScientificAblationWorkflowAdvanceResultV1) -> dict[str, Any]:
    """Save actual immutable returned DATA, not authority or a synthetic ACK."""
    return dict(
        requested_phase=json.loads(result.requested_phase_json),
        completed_phase_sha256_before=result.completed_phase_sha256_before,
        expected_chunk=None
        if result.expected_chunk_json is None
        else json.loads(result.expected_chunk_json),
        rows_per_call=result.rows_per_call,
        kernel_progress=None
        if result.kernel_progress_json is None
        else json.loads(result.kernel_progress_json),
        phase_record=None
        if result.phase_record_json is None
        else json.loads(result.phase_record_json),
        phase_files=[
            dict(name=name, bytes=len(raw), sha256=sha256(raw).hexdigest())
            for name, raw in result.phase_files
        ],
        progress=asdict(result.progress),
    )


def _committed(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    result: workflow.ScientificAblationWorkflowAdvanceResultV1,
) -> None:
    assert result.phase_record_json is not None
    actual = job._seal.store.read(phase, expected_sha256=result.progress.completed_phase_sha256[-1])
    assert steps.encoded(actual.record) == result.phase_record_json
    assert actual.files == result.phase_files


def _call(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    request: Chunk | None = None,
) -> workflow.ScientificAblationWorkflowAdvanceResultV1:
    return job.advance_expected_with_result(
        expected_phase=phase,
        expected_completed_phase_sha256=job.progress.completed_phase_sha256,
        expected_chunk=request,
        rows_per_call=None if request is None else request.rows_per_call,
    )


def _reject_stale(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    request: Chunk,
) -> None:
    before = _tree(job._seal.path)
    with (
        patch.object(
            ScientificAblationWorkflowStoreV1, "claim", side_effect=AssertionError("No claim")
        ) as claim,
        patch.object(steps, "_owner", side_effect=AssertionError("No original factory")) as owner,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(ValueError),
    ):
        _call(job, phase, request)
    assert claim.call_count == owner.call_count == graph.call_count == forward.call_count == 0
    assert _tree(job._seal.path) == before and job._state == "idle"


def _next(request: Chunk, observed: Kernel) -> Chunk:
    return Chunk.model_validate(
        request.model_dump()
        | dict(
            acknowledgement_head_sha256=observed.acknowledgement_head_sha256,
            acknowledged_rows=observed.acknowledged_rows,
            stop_row_index=min(
                observed.total_rows, observed.acknowledged_rows + request.rows_per_call
            ),
        )
    )


def _native_commit_retry(
    job: workflow.IrisScientificAblationWorkflowV1,
    phase: ScientificAblationWorkflowPhaseV1,
    workflow_root: Path,
) -> workflow.ScientificAblationWorkflowAdvanceResultV1:
    assert phase.kind == "diagnostics"
    before = job.progress.completed_phase_sha256
    original_execute = steps.execute_produced
    with (
        patch.object(steps, "execute_produced", wraps=original_execute) as work,
        patch.object(
            ScientificAblationWorkflowStoreV1,
            "commit",
            side_effect=OSError("after real native result"),
        ),
        pytest.raises(IrisScientificAblationWorkflowStorageError),
    ):
        _call(job, phase)
    assert work.call_count == 1 and job._seal.ready is not None and job._state == "idle"
    ready = job._seal.ready
    assert ready.kernel_progress is None and ready.producer_result_requested
    assert job.progress.completed_phase_sha256 == before
    cold_handle = workflow.create_scientific_ablation_workflow_v1(
        root=workflow_root,
        plan=job._seal.store.plan,
        make_fresh_run=job._seal.factory,
        completed_phase_sha256=before,
    )
    tree = _tree(job._seal.path)
    with (
        patch.object(
            steps, "_owner", side_effect=AssertionError("No remeasurement after reopen")
        ) as owner,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(IrisScientificAblationWorkflowUncertainError),
    ):
        _call(cold_handle, phase)
    assert owner.call_count == forward.call_count == 0 and _tree(job._seal.path) == tree
    with (
        patch.object(
            steps, "execute_produced", side_effect=AssertionError("No second producer")
        ) as repeated,
        patch.object(
            steps, "_owner", side_effect=AssertionError("No second native owner")
        ) as owner,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No second forward")
        ) as forward,
    ):
        result = _call(job, phase)
    assert repeated.call_count == owner.call_count == forward.call_count == 0
    assert result.phase_record_json == ready.record and result.phase_files == ready.files
    assert result.completed_phase_sha256_before == before
    _committed(job, phase, result)
    return result


@pytest.mark.parametrize("geometry", (GEOMETRIES[0], GEOMETRIES[2]), ids=("c1", "c5"))
@pytest.mark.parametrize("controls", ("zero", "single-empty", "paired-nonempty"))
def test_actual_original_producer_results_complete_workflow_and_storage_retry(
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
        for phase in phases:
            if phase.kind not in ("capture", "score"):
                before = job.progress.completed_phase_sha256
                result = (
                    _native_commit_retry(job, phase, tmp_path / "workflow")
                    if phase.kind == "diagnostics"
                    else _call(job, phase)
                )
                assert result.completed_phase_sha256_before == before
                assert result.kernel_progress_json is None and result.expected_chunk_json is None
                assert result.rows_per_call is None
                _committed(job, phase, result)
                records.append(_wire(result))
                continue
            module, _, operation = _binding(phase)
            request = _first_request(job, phase)
            for row in range(2):
                before = job.progress.completed_phase_sha256
                with patch.object(module, operation, wraps=getattr(module, operation)) as original:
                    result = _call(job, phase, request)
                assert (
                    original.call_count == 1
                    and original.call_args.kwargs["expected_chunk"] == request
                )
                assert result.kernel_progress_json is not None
                produced = Kernel.model_validate_json(result.kernel_progress_json)
                assert produced.phase == phase and produced.completed_phase_sha256 == before
                assert not produced.phase_is_acknowledged and produced.acknowledged_rows == row + 1
                assert produced.kernel_plan_sha256 == request.kernel_plan_sha256
                assert result.requested_phase_json == steps.encoded(phase)
                assert result.expected_chunk_json == steps.encoded(request)
                assert result.completed_phase_sha256_before == before
                observed = job.inspect_kernel(
                    expected_phase=phase, expected_kernel_plan_sha256=request.kernel_plan_sha256
                )
                assert produced.acknowledgement_head_sha256 == observed.acknowledgement_head_sha256
                assert produced.kernel_complete_sha256 == observed.kernel_complete_sha256
                assert produced.uncommitted_stage_count == observed.uncommitted_stage_count
                if row == 0:
                    assert result.progress.completed_phase_sha256 == before
                    assert result.phase_record_json is None and result.phase_files == ()
                    _reject_stale(job, phase, request)
                else:
                    assert len(result.progress.completed_phase_sha256) == len(before) + 1
                    _committed(job, phase, result)
                records.append(_wire(result))
                request = _next(request, produced)
        assert candidate.call_count == 0
    assert job.progress.complete and len(job.progress.completed_phase_sha256) == 6 + 2 * len(
        control_plans
    )
    assert len(records) == len(phases) + 2 * (1 + len(control_plans))
    (tmp_path / "actual-original-producer-workflow-witness.json").write_bytes(
        canonical_json_file_bytes(
            asdict(geometry)
            | dict(
                kind="iris-original-producer-result-full-native-workflow-witness-v1",
                controls=controls,
                plan_sha256=plan.sha256,
                completed_phase_sha256=job.progress.completed_phase_sha256,
                actual_original_return_results=records,
                native_diagnostics_original_result_then_commit_failure=True,
                reopened_handle_without_original_ready_stays_uncertain=True,
                same_original_owner_retry_has_zero_producer_owner_forward=True,
                actual_os_cold_process=False,
                durable_independent_producer_ACK=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )


@pytest.mark.parametrize("case_name", RACES)
def test_actual_kernel_result_keeps_old_head_after_standalone_writer_advances(
    tmp_path: Path, case_name: str
) -> None:
    case = prepare(tmp_path / "case", GEOMETRIES[0])
    plan = build_scientific_ablation_workflow_plan_v1(
        case.plan.payload.model_copy(
            update={
                "controls": (WorkspaceInterventionPlanV1(interventions=()),),
            }
        )
    )
    phases = scientific_ablation_workflow_phases_v1(plan)
    target = next(
        phase
        for phase in phases
        if phase.kind == case_name.rsplit("-", 1)[1]
        and (phase.stream_index is not None) == case_name.startswith("controlled")
    )
    job = workflow.create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow", plan=plan, make_fresh_run=case.factory
    )
    for phase in phases[: target.index]:
        request = _first_request(job, phase)
        for _ in range(2):
            prior = _call(job, phase, request)
            assert prior.kernel_progress_json is not None
            request = _next(request, Kernel.model_validate_json(prior.kernel_progress_json))
    first = _first_request(job, target)
    module, _, operation = _binding(target)
    original = getattr(module, operation)
    events: list[dict[str, Any]] = []

    def execute_then_compete(*args: Any, **kwargs: Any) -> Any:
        returned = original(*args, **kwargs)
        assert len(returned.rows) == 1 and returned.complete is None
        assert kwargs["expected_chunk"] == first
        original_head = returned.acknowledgements[-1].sha256
        rival_request = Chunk.model_validate(
            first.model_dump()
            | dict(
                acknowledgement_head_sha256=original_head,
                acknowledged_rows=1,
                stop_row_index=2,
            )
        )
        rival_args = args
        if target.kind == "capture":
            rival_owner = steps._owner(plan, target, case.factory)
            assert rival_owner is not args[0]
            rival_args = (rival_owner, *args[1:])
        # This is a SECOND explicit standalone command, not modification of the
        # first public command. Its own original K claim is acquired after the
        # first K claim released, while the outer workflow W claim remains held.
        advanced = original(*rival_args, **(kwargs | {"expected_chunk": rival_request}))
        assert len(advanced.rows) == 2 and advanced.complete is not None
        assert returned.acknowledgements[-1].sha256 == original_head
        events.append(
            dict(
                first_request=first.model_dump(mode="json"),
                rival_request=rival_request.model_dump(mode="json"),
                original_head=original_head,
                rival_head=advanced.acknowledgements[-1].sha256,
                rival_complete_sha256=advanced.complete.sha256,
            )
        )
        return returned

    with patch.object(module, operation, new=execute_then_compete):
        result = _call(job, target, first)
    assert len(events) == 1 and result.kernel_progress_json is not None
    retained = result.kernel_progress_json
    produced = Kernel.model_validate_json(retained)
    observed = job.inspect_kernel(
        expected_phase=target, expected_kernel_plan_sha256=first.kernel_plan_sha256
    )
    assert produced.acknowledged_rows == 1 and produced.kernel_complete_sha256 is None
    assert produced.acknowledgement_head_sha256 == events[0]["original_head"]
    assert observed.acknowledged_rows == 2 and observed.kernel_complete_sha256 is not None
    assert observed.acknowledgement_head_sha256 == events[0]["rival_head"]
    assert produced.acknowledgement_head_sha256 != observed.acknowledgement_head_sha256
    assert result.phase_record_json is None and job.progress.next_phase == target.name
    _reject_stale(job, target, first)
    completed = _call(job, target, _next(first, observed))
    _committed(job, target, completed)
    assert result.kernel_progress_json == retained
    (tmp_path / "actual-post-return-kernel-race-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-original-producer-post-return-kernel-race-witness-v1",
                case_name=case_name,
                plan_sha256=plan.sha256,
                events=events,
                original_result=_wire(result),
                physical_after_rival=observed.model_dump(mode="json"),
                explicit_later_request_result=_wire(completed),
                two_real_original_kernel_commands=True,
                first_command_arguments_not_rewritten=True,
                true_concurrent_OS_process=False,
                durable_independent_producer_ACK=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
