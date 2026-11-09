"""Kernel inspection controller/schema tests; mocked DATA is not native evidence."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_workflow as owner
from iris.application import scientific_ablation_workflow_kernel as module
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowStoreV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from tests.unit._workflow_data_setup import plan

_PLAN = "a" * 64
_HEAD = "b" * 64
_COMPLETE = "c" * 64


def job_phase(
    tmp_path: Path,
) -> tuple[owner.IrisScientificAblationWorkflowV1, ScientificAblationWorkflowPhaseV1]:
    p = plan(tmp_path)
    return owner.create_scientific_ablation_workflow_v1(root=tmp_path / "out", plan=p), (
        scientific_ablation_workflow_phases_v1(p)[0]
    )


def probe(
    job: object,
    phase: object,
    kernel: object = _PLAN,
    head: object = None,
    method: str = "inspect_kernel",
) -> object:
    return getattr(job, method)(
        expected_phase=phase,
        expected_kernel_plan_sha256=kernel,
        expected_acknowledgement_head_sha256=head,
    )


def test_mocked_partial_snapshot_is_read_only_and_never_permission(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    with (
        patch.object(module, "_capture", return_value=(_PLAN, _HEAD, 1, 2, None, 3)),
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
    ):
        value = job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert value.acknowledged_rows == 1 and value.total_rows == 2
    assert value.completed_phase_sha256 == () and value.phase_is_acknowledged is False
    assert value.uncommitted_stage_count == 3 and value.kernel_complete_sha256 is None
    assert value.new_native_execution_authorized is False
    assert value.native_execution_authority == "none_data_only"
    assert claim.call_count == 0 and not (tmp_path / "out").exists() and job._state == "idle"


def test_missing_kernel_never_creates_directory_attempt_or_lock(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    with pytest.raises(RuntimeError):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert not (tmp_path / "out").exists() and job._state == "idle"


@pytest.mark.parametrize("index", range(1, 6))
def test_future_or_non_kernel_phase_denied_before_reader(tmp_path: Path, index: int) -> None:
    job, _ = job_phase(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(plan(tmp_path))[index]
    with patch.object(module, "_capture") as read, pytest.raises(ValueError):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert read.call_count == 0 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ("kernel", "head"))
@pytest.mark.parametrize("bad", (True, 1, b"a" * 64, "wrong", "A" * 64))
def test_exact_independent_digest_arguments(tmp_path: Path, field: str, bad: object) -> None:
    job, phase = job_phase(tmp_path)
    with patch.object(module, "_capture") as read, pytest.raises((TypeError, ValueError)):
        probe(job, phase, bad if field == "kernel" else _PLAN, bad if field == "head" else None)
    assert read.call_count == 0 and not (tmp_path / "out").exists()


def test_wrong_ack_head_does_not_advance_or_mutate_owner(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    seal = job._seal
    with (
        patch.object(module, "_capture", return_value=(_PLAN, _HEAD, 1, 2, None, 0)),
        pytest.raises(ValueError),
    ):
        job.inspect_kernel(
            expected_phase=phase,
            expected_kernel_plan_sha256=_PLAN,
            expected_acknowledgement_head_sha256="d" * 64,
        )
    assert job._seal is seal and job._state == "idle" and not (tmp_path / "out").exists()


def test_reader_cannot_substitute_another_kernel_identity(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    with (
        patch.object(module, "_capture", return_value=("d" * 64, _HEAD, 1, 2, None, 0)),
        pytest.raises(ValueError),
    ):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert not (tmp_path / "out").exists()


def test_unknown_ready_due_phase_is_preserved_not_adopted(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    ready = job._seal.path / "phase-000"
    ready.mkdir(parents=True)
    with (
        patch.object(module, "_capture") as read,
        pytest.raises(IrisScientificAblationWorkflowUncertainError),
    ):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert read.call_count == 0 and list(ready.iterdir()) == []


def test_hidden_phase_metadata_rejected_before_reader(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    vars(phase)["hidden"] = "not a capability"
    with patch.object(module, "_capture") as read, pytest.raises((TypeError, ValueError)):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert read.call_count == 0 and not (tmp_path / "out").exists()


def test_unsealed_and_busy_owner_rejected(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    fake = object.__new__(owner.IrisScientificAblationWorkflowV1)
    with pytest.raises(owner.IrisScientificAblationWorkflowError):
        fake.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    job._state = "busy"
    with pytest.raises(owner.IrisScientificAblationWorkflowError):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert not (tmp_path / "out").exists()


def test_owner_replacement_during_reader_is_not_accepted(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)

    def changed(*args: object) -> tuple[str, str, int, int, None, int]:
        job._seal = replace(job._seal)
        return _PLAN, _HEAD, 1, 2, None, 0

    with (
        patch.object(module, "_capture", side_effect=changed),
        pytest.raises(owner.IrisScientificAblationWorkflowError),
    ):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert not (tmp_path / "out").exists()


def test_original_signal_is_not_rewritten(tmp_path: Path) -> None:
    job, phase = job_phase(tmp_path)
    with (
        patch.object(module, "_capture", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        job.inspect_kernel(expected_phase=phase, expected_kernel_plan_sha256=_PLAN)
    assert job._state == "idle" and not (tmp_path / "out").exists()


def snapshot(tmp_path: Path) -> dict[str, Any]:
    job, phase = job_phase(tmp_path)
    return dict(
        workflow_plan_sha256=job._seal.store.plan.sha256,
        phase=phase,
        completed_phase_sha256=(),
        phase_is_acknowledged=False,
        kernel_plan_sha256=_PLAN,
        acknowledgement_head_sha256=_HEAD,
        acknowledged_rows=1,
        total_rows=2,
        kernel_complete_sha256=None,
        uncommitted_stage_count=0,
    )


@pytest.mark.parametrize(
    "change",
    (
        {"acknowledged_rows": True},
        {"acknowledged_rows": -1},
        {"total_rows": 5001},
        {"acknowledged_rows": 3},
        {"kernel_complete_sha256": _COMPLETE},
        {"phase_is_acknowledged": True},
        {"new_native_execution_authorized": True},
        {"native_execution_authority": "restored"},
        {"scientific_acceptance": "accepted"},
    ),
)
def test_snapshot_schema_cannot_be_an_incoherent_native_receipt(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        IrisScientificAblationWorkflowKernelProgressV1.model_validate(snapshot(tmp_path) | change)
