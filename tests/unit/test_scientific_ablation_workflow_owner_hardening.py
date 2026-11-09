"""Labelled owner/controller faults, NOT actual native execution witnesses."""

from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as module
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from tests.unit._workflow_data_setup import plan


def test_readonly_progress_rejects_foreign_root_member_without_a_native_call(
    tmp_path: Path,
) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    with job._seal.store.claim():
        pass
    foreign = job._seal.path / "unowned.txt"
    foreign.write_bytes(b"must not be adopted or removed\n")
    with (
        patch.object(steps, "execute", side_effect=AssertionError("Read-only must not advance")),
        pytest.raises((ValueError, module.IrisScientificAblationWorkflowError)),
    ):
        _ = job.progress
    assert foreign.read_bytes() == b"must not be adopted or removed\n"


@pytest.mark.parametrize("field", ("_seal", "_state"))
def test_deleted_owner_field_cannot_mask_first_original_signal(tmp_path: Path, field: str) -> None:
    def unused_factory() -> IrisScientificAblationRunV1:
        raise AssertionError("Manufactured controller fault must not construct native owners")

    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output",
        plan=plan(tmp_path),
        make_fresh_run=unused_factory,
    )
    signal = KeyboardInterrupt("original controller signal, not native execution")

    def changed(*args: object, **kwargs: object) -> NoReturn:
        delattr(job, field)
        raise signal

    with patch.object(steps, "execute", side_effect=changed):
        try:
            job.advance()
        except BaseException as error:
            assert error is signal
        else:
            pytest.fail("The exact first signal must be raised")


def test_readonly_progress_rejects_changed_original_plan_without_a_native_call(
    tmp_path: Path,
) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    with job._seal.store.claim():
        pass
    original = job._seal.path / "plan.json"
    original.write_bytes(b"{}")
    with pytest.raises((ValueError, module.IrisScientificAblationWorkflowError)):
        _ = job.progress
    assert original.read_bytes() == b"{}"


def test_readonly_progress_rejects_equal_byte_replacement_of_owned_root(tmp_path: Path) -> None:
    job = module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=plan(tmp_path)
    )
    with job._seal.store.claim():
        pass
    original = job._seal.path.resolve()
    retained = original.with_name(original.name + ".retained-original")
    # Move only this exact test-owned root; retain it, never delete or sweep.
    assert original.is_relative_to(tmp_path.resolve())
    assert retained.is_relative_to(tmp_path.resolve()) and not retained.exists()
    original.rename(retained)
    original.mkdir()
    (original / "plan.json").write_bytes(job._seal.plan)
    with pytest.raises(module.IrisScientificAblationWorkflowError):
        _ = job.progress
    assert retained.is_dir() and (original / "plan.json").read_bytes() == job._seal.plan
