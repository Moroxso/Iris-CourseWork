"""Labelled controller/serialization probes only; NOT native/row-ACK evidence."""

import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_workflow as owner
from iris.application import scientific_ablation_workflow_kernel as kernel
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowCaptureDataV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk
from tests.unit.test_scientific_ablation_workflow_expected_phase import job_and_phase

CASES = ("ordinary-capture", "controlled-capture", "ordinary-score", "controlled-score")


def labelled_return(tmp_path: Path, case: str) -> tuple[Any, ...]:
    """Explicit test double internals, never claimed as a physical/native plan."""
    p = build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    store = ScientificAblationWorkflowStoreV1(tmp_path / "out", p)
    controlled = case.startswith("controlled")
    kind = "capture" if case.endswith("capture") else "score"
    phase = scientific_ablation_workflow_phases_v1(p)[
        (2 if controlled else 0) + (1 if kind == "score" else 0)
    ]
    prefix = tuple(
        SimpleNamespace(record=SimpleNamespace(sha256=f"{index:064x}"))
        for index in range(phase.index)
    )
    if kind == "capture":
        module: Any = controlled_capture if controlled else capture
        progress_type: Any = (
            controlled_capture.IrisScientificAblationControlledCaptureProgressV1
            if controlled
            else capture.IrisScientificAblationCaptureProgressV1
        )
    else:
        module = controlled_score if controlled else score
        progress_type = (
            controlled_score.IrisScientificAblationControlledEvaluationProgressV1
            if controlled
            else score.IrisScientificAblationEvaluationProgressV2
        )
    labelled_plan = SimpleNamespace(sha256="a" * 64, payload=SimpleNamespace(rows=(0, 1)))
    acknowledgement = SimpleNamespace(sha256="b" * 64)
    args = dict(
        path=steps._kernel(store, phase) / module._STORE / labelled_plan.sha256,
        plan=labelled_plan,
        rows=(object(),),
        acknowledgements=(acknowledgement,),
        complete=None,
        uncommitted_stage_count=2,
    )
    if kind == "score":
        args["evaluation"] = None
    return store, phase, prefix, progress_type(**args), acknowledgement


def call(job: Any, phase: Any, value: Any = None, rows: Any = 1) -> Any:
    return job.advance_expected_with_result(
        expected_phase=phase,
        expected_completed_phase_sha256=(),
        expected_chunk=chunk() if value is None else value,
        rows_per_call=rows,
    )


@pytest.mark.parametrize("case", CASES)
def test_snapshot_uses_original_return_not_a_newer_inspection(tmp_path: Path, case: str) -> None:
    store, phase, prefix, original, acknowledgement = labelled_return(tmp_path, case)
    with patch.object(kernel, "read_scientific_ablation_workflow_kernel_progress_v1") as read:
        raw = steps._returned_kernel(store, phase, prefix, original, chunk())
    acknowledgement.sha256 = "f" * 64  # Labelled later state cannot mutate retained bytes.
    parsed = IrisScientificAblationWorkflowKernelProgressV1.model_validate_json(raw)
    assert parsed.acknowledged_rows == 1 and parsed.acknowledgement_head_sha256 == "b" * 64
    assert parsed.uncommitted_stage_count == 2 and parsed.phase == phase
    assert parsed.completed_phase_sha256 == tuple(data.record.sha256 for data in prefix)
    assert not parsed.new_native_execution_authorized and read.call_count == 0


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("bad", ("type", "path", "acks", "interval", "total"))
def test_snapshot_rejects_wrong_original_type_path_or_interval(
    tmp_path: Path, case: str, bad: str
) -> None:
    store, phase, prefix, original, _ = labelled_return(tmp_path, case)
    value = chunk()
    if bad == "type":
        original = SimpleNamespace(**{name: getattr(original, name) for name in original.__slots__})
    elif bad == "path":
        object.__setattr__(original, "path", tmp_path / "different")
    elif bad == "acks":
        object.__setattr__(original, "acknowledgements", ())
    elif bad == "interval":
        value = chunk(rows_per_call=2, stop_row_index=2)
    else:
        value = chunk(total_rows=3)
    with pytest.raises(ValueError):
        steps._returned_kernel(store, phase, prefix, original, value)


def test_public_result_requires_an_exact_chunk_before_claim(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(ValueError),
    ):
        job.advance_expected_with_result(
            expected_phase=phase, expected_completed_phase_sha256=(), rows_per_call=1
        )
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("index", (2, 3, 4, 5))
def test_native_result_rejects_irrelevant_row_budget_before_claim(
    tmp_path: Path, index: int
) -> None:
    job, _ = job_and_phase(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(job._seal.store.plan)[index]
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(ValueError),
    ):
        job.advance_expected_with_result(
            expected_phase=phase, expected_completed_phase_sha256=(), rows_per_call=1
        )
    assert claim.call_count == 0


def test_public_partial_return_keeps_request_and_result_immutable(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    value = chunk()
    original_claim = ScientificAblationWorkflowStoreV1.claim
    retained = b'{"labelled_controller_result":true}\n'

    @contextmanager
    def claim(store: ScientificAblationWorkflowStoreV1) -> Any:
        with original_claim(store) as held:
            value.__dict__["acknowledgement_head_sha256"] = "f" * 64
            yield held

    with (
        patch.object(kernel, "_check_chunk_before_native") as check,
        patch.object(ScientificAblationWorkflowStoreV1, "claim", new=claim),
        patch.object(
            steps, "execute_produced", return_value=steps._Produced(None, (), retained)
        ) as work,
        patch.object(steps, "execute") as legacy,
        patch.object(kernel, "read_scientific_ablation_workflow_kernel_progress_v1") as read,
    ):
        result = call(job, phase, value)
    assert work.call_count == 1 and check.call_count == 2
    assert legacy.call_count == read.call_count == 0 and job._state == "idle"
    assert result.kernel_progress_json == retained and result.phase_record_json is None
    assert result.phase_files == () and result.progress.completed_phase_sha256 == ()
    assert result.requested_phase_json == steps.encoded(phase)
    assert result.expected_chunk_json == steps.encoded(chunk())
    assert result.completed_phase_sha256_before == () and result.rows_per_call == 1
    with pytest.raises(FrozenInstanceError):
        result.rows_per_call = 2


def labelled_complete(job: Any, phase: Any) -> steps._Produced:
    data = job._seal.store.make_data(
        phase,
        None,
        origin="kernel_data",
        files={
            "capture.json": steps.encoded(
                ScientificAblationWorkflowCaptureDataV1(
                    protocol="ordinary",
                    plan_sha256="a" * 64,
                    input_sha256="b" * 64,
                    capture_sha256="c" * 64,
                    examples=2,
                )
            )
        },
    )
    return steps._Produced(
        steps.encoded(data.record), data.files, b'{"labelled_controller_result":"original"}\n'
    )


@pytest.mark.parametrize("retry", ("same", "legacy", "plain", "different", "budget"))
def test_storage_retry_retains_original_result_and_exact_request_without_work(
    tmp_path: Path, retry: str
) -> None:
    job, phase = job_and_phase(tmp_path)
    produced = labelled_complete(job, phase)
    original_commit = ScientificAblationWorkflowStoreV1.commit
    with (
        patch.object(kernel, "_check_chunk_before_native"),
        patch.object(steps, "validate_data"),
        patch.object(steps, "execute_produced", return_value=produced) as work,
        patch.object(ScientificAblationWorkflowStoreV1, "commit", side_effect=OSError("storage")),
        pytest.raises(IrisScientificAblationWorkflowStorageError),
    ):
        call(job, phase)
    assert work.call_count == 1 and job._seal.ready is not None and job._state == "idle"
    saved = job._seal.ready
    with (
        patch.object(steps, "validate_data"),
        patch.object(steps, "execute_produced") as new_work,
        patch.object(steps, "execute") as old_work,
        patch.object(kernel, "_check_chunk_before_native") as read,
        patch.object(ScientificAblationWorkflowStoreV1, "commit", new=original_commit),
    ):
        if retry == "same":
            result = call(job, phase)
            assert result.kernel_progress_json == produced.kernel_progress
            assert (
                result.phase_record_json == produced.record and result.phase_files == produced.files
            )
            assert len(result.progress.completed_phase_sha256) == 1
            assert result.completed_phase_sha256_before == () and job._seal.ready is None
            assert (
                json.loads(result.phase_record_json)["sha256"]
                == result.progress.completed_phase_sha256[0]
            )
        else:
            with pytest.raises(owner.IrisScientificAblationWorkflowError):
                if retry == "plain":
                    job.advance()
                elif retry == "legacy":
                    job.advance_expected(
                        expected_phase=phase,
                        expected_completed_phase_sha256=(),
                        expected_chunk=chunk(),
                        rows_per_call=1,
                    )
                elif retry == "different":
                    call(job, phase, chunk(acknowledgement_head_sha256="f" * 64))
                else:
                    call(job, phase, chunk(rows_per_call=2, stop_row_index=2), 2)
            assert job._seal.ready is saved and job._seal.anchors == ()
    assert new_work.call_count == old_work.call_count == read.call_count == 0


def test_original_signal_is_not_replaced_by_a_result(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    signal = KeyboardInterrupt("labelled controller signal")
    with (
        patch.object(kernel, "_check_chunk_before_native"),
        patch.object(steps, "execute_produced", side_effect=signal),
    ):
        try:
            call(job, phase)
        except BaseException as error:
            assert error is signal
        else:
            pytest.fail("Original signal was swallowed")
    assert job._state == "idle" and job._seal.ready is None
