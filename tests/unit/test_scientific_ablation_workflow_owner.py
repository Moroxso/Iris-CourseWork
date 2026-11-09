"""Owner admission tests; DATA fixtures are not claimed native executions."""

from pathlib import Path
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as module
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
)
from tests.unit._workflow_data_setup import plan


def test_public_constructor_and_unsealed_object_do_not_create_native_authority() -> None:
    with pytest.raises(module.IrisScientificAblationWorkflowError):
        module.IrisScientificAblationWorkflowV1()
    fake = object.__new__(module.IrisScientificAblationWorkflowV1)
    with pytest.raises(module.IrisScientificAblationWorkflowError):
        fake.advance()


def _runtime_probe(job: object, name: str, rows: object) -> object:
    """Intentionally ill-typed runtime API probe, not a native completion callback."""
    return getattr(job, name)(rows_per_call=rows)


@pytest.mark.parametrize("rows", (True, False, 0, -1, 5001, 1.0, "1"))
def test_wrong_chunk_is_rejected_before_any_write(tmp_path: Path, rows: object) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    with pytest.raises(ValueError):
        _runtime_probe(job, "advance", rows)
    assert not (tmp_path / "output").exists()


def test_construction_and_progress_have_no_factory_or_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    with patch.object(
        steps, "_owner", side_effect=AssertionError("No original native work during admission")
    ):
        job = module.create_scientific_ablation_workflow_v1(
            root=tmp_path / "output", plan=plan(tmp_path)
        )
        assert job.progress.completed_phase_sha256 == ()
        assert job.progress.next_phase == "ordinary-capture"
    assert not (tmp_path / "output").exists()


def test_missing_factory_cannot_mark_an_original_native_attempt(tmp_path: Path) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    with pytest.raises(IrisScientificAblationWorkflowStorageError):
        job.advance()
    assert not tuple(job._seal.path.glob("*.attempt.json"))


def test_original_owner_reentry_fails_before_work_or_writes(tmp_path: Path) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    job._state = "busy"
    with pytest.raises(module.IrisScientificAblationWorkflowError):
        job.advance()
    assert not (tmp_path / "output").exists()


def test_mutated_store_plan_is_not_laundered_by_original_handle(tmp_path: Path) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    job._seal.store._plan_bytes = b"{}"
    with pytest.raises((ValueError, module.IrisScientificAblationWorkflowError)):
        job.advance()
    assert not (tmp_path / "output").exists()
