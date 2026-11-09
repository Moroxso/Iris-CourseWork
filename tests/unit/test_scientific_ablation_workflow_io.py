"""Real filesystem DATA journal/uncertainty tests. No native calls are claimed."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_workflow_io as module
from iris.application.scientific_ablation_inference_io import _finish_descriptor
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError as Error,
)
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowUncertainError as Uncertain,
)
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from tests.unit._workflow_data_setup import plan


def test_partial_native_attempt_is_preserved_and_never_reinterpreted_as_not_started(
    tmp_path: Path,
) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[2]
    with store.claim():
        pass
    marker = store.path / f"phase-{phase.index:03d}.attempt.json"
    marker.write_bytes(b'{"interrupted":')
    before = marker.read_bytes(), marker.stat().st_ino
    with pytest.raises(Error), store.claim() as claim:
        store.begin(phase, "1" * 64, claim)
    assert (marker.read_bytes(), marker.stat().st_ino) == before


def test_complete_metadata_byte_cap_fails_before_creating_storage(tmp_path: Path) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    with pytest.raises(ValueError):
        store.make_data(
            phase, None, origin="kernel_data", files={"capture.json": b"x" * (4 * 1024**2 + 1)}
        )
    assert not store.path.exists()


def test_whole_storage_quota_rejects_before_plan_or_phase_write(tmp_path: Path) -> None:
    from iris.application.scientific_ablation_workflow_io import (
        build_scientific_ablation_workflow_plan_v1,
    )
    from tests.unit._workflow_data_setup import intent

    value = intent(tmp_path)
    value.resources.__dict__["maximum_storage_bytes"] = 1
    store = ScientificAblationWorkflowStoreV1(
        tmp_path / "data", build_scientific_ablation_workflow_plan_v1(value)
    )
    with pytest.raises(Error), store.claim():
        pytest.fail("One byte cannot contain the original complete plan")
    assert {p.name for p in store.path.iterdir()} == {".iris-owner.lock"}


def test_actual_immutable_data_commit_read_retry_and_lock_inode(tmp_path: Path) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    data = store.make_data(
        phase, None, origin="kernel_data", files={"capture.json": b'{"data":"only"}\n'}
    )
    with store.claim() as claim:
        store.begin(phase, None, claim)
        assert store.commit(data, claim) == data
    original = (store.path / "phase-000/capture.json").stat().st_ino
    lock = (store.path / ".iris-owner.lock").stat().st_ino
    assert store.read(phase, expected_sha256=data.record.sha256) == data
    assert data.record.payload.native_execution_authority == "none_data_only"
    with store.claim() as claim:
        store.begin(phase, None, claim)
        assert store.commit(data, claim) == data
    assert (store.path / "phase-000/capture.json").stat().st_ino == original
    assert (store.path / ".iris-owner.lock").stat().st_ino == lock


@pytest.mark.parametrize("kind", ("diagnostics", "workspace", "message"))
def test_existing_native_attempt_is_uncertain_never_auto_remeasured(
    tmp_path: Path, kind: str
) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = next(p for p in scientific_ablation_workflow_phases_v1(store.plan) if p.kind == kind)
    with store.claim() as claim:
        first = store.begin(phase, "1" * 64, claim)
    marker = store.path / f"phase-{phase.index:03d}.attempt.json"
    original = marker.read_bytes(), marker.stat().st_ino
    cold = ScientificAblationWorkflowStoreV1(tmp_path / "data", store.plan)
    with pytest.raises(Uncertain), cold.claim() as claim:
        cold.begin(phase, "1" * 64, claim)
    assert (marker.read_bytes(), marker.stat().st_ino) == original
    assert json.loads(original[0])["sha256"] == first.sha256


@pytest.mark.parametrize("kind", ("capture", "score", "publish"))
def test_kernel_prefix_or_data_publication_attempt_can_retry_without_native_permission(
    tmp_path: Path, kind: str
) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = next(p for p in scientific_ablation_workflow_phases_v1(store.plan) if p.kind == kind)
    previous = None if phase.index == 0 else "1" * 64
    with store.claim() as claim:
        first = store.begin(phase, previous, claim)
    with store.claim() as claim:
        assert store.begin(phase, previous, claim) == first


def test_ready_native_data_needs_independent_anchor_and_does_not_restore_permission(
    tmp_path: Path,
) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[2]
    data = store.make_data(
        phase,
        "1" * 64,
        origin="original_native_result_data",
        files={"ordinary.json": b"ordinary DATA", "collapse.json": b"compact DATA"},
    )
    with store.claim() as claim:
        store.begin(phase, "1" * 64, claim)
        store.commit(data, claim)
    cold = ScientificAblationWorkflowStoreV1(tmp_path / "data", store.plan)
    with pytest.raises(ValueError):
        cold.read(phase, expected_sha256="0" * 64)
    assert cold.read(phase, expected_sha256=data.record.sha256) == data
    with pytest.raises(Uncertain), cold.claim() as claim:
        cold.begin(phase, "1" * 64, claim)


def test_foreign_workflow_claim_cannot_write_another_store(tmp_path: Path) -> None:
    first = ScientificAblationWorkflowStoreV1(tmp_path / "one", plan(tmp_path))
    second = ScientificAblationWorkflowStoreV1(tmp_path / "two", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(first.plan)[0]
    with pytest.raises(Error), first.claim() as claim:
        second.begin(phase, None, claim)
    assert not second.path.exists()


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit))
def test_first_signal_survives_data_write_close_failure_and_preserves_partial_attempt(
    tmp_path: Path, signal: type[BaseException]
) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[2]
    with store.claim():
        pass
    first = signal("original-write")
    original_finish = _finish_descriptor

    def finish(primary: BaseException | None, close: object) -> None:
        def broken() -> None:
            assert callable(close)
            close()
            raise RuntimeError("close-after-signal")

        original_finish(primary, broken)

    with (
        patch.object(module, "_write_bytes", side_effect=first),
        patch.object(module, "_finish_descriptor", new=finish),
        pytest.raises(signal) as caught,
        store.claim() as claim,
    ):
        store.begin(phase, "1" * 64, claim)
    assert caught.value is first
    assert (store.path / f"phase-{phase.index:03d}.attempt.json").is_file()


def test_equal_byte_replacement_of_owned_staged_file_is_not_adopted(tmp_path: Path) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[2]
    data = store.make_data(
        phase,
        "1" * 64,
        origin="original_native_result_data",
        files={"ordinary.json": b"ordinary DATA", "collapse.json": b"compact DATA"},
    )
    original_write = store._write_new

    def write(path: Path, content: bytes, claim: module._Claim) -> tuple[int, int]:
        identity = original_write(path, content, claim)
        if path.name == "ordinary.json":
            replacement = path.with_name("replacement.tmp")
            replacement.write_bytes(content)
            os.replace(replacement, path)
        return identity

    with pytest.raises(Error), store.claim() as claim:
        store.begin(phase, "1" * 64, claim)
        with patch.object(store, "_write_new", new=write):
            store.commit(data, claim)
    assert not (store.path / "phase-002").exists()
    assert list(store.path.glob(".iris-workflow-*.tmp"))


def test_corrupt_complete_file_and_foreign_member_never_become_ready_data(tmp_path: Path) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    data = store.make_data(phase, None, origin="kernel_data", files={"capture.json": b"DATA"})
    with store.claim() as claim:
        store.begin(phase, None, claim)
        store.commit(data, claim)
    (store.path / "phase-000/extra.json").write_bytes(b"unrecognized")
    with pytest.raises(ValueError):
        store.read(phase, expected_sha256=data.record.sha256)


def test_process_lock_reentry_fails_without_replacing_or_removing_lock(tmp_path: Path) -> None:
    store = ScientificAblationWorkflowStoreV1(tmp_path / "data", plan(tmp_path))
    with store.claim():
        lock = (store.path / ".iris-owner.lock").stat().st_ino
        with pytest.raises(Error), store.claim():
            pytest.fail("Nested claim cannot be acquired")
    assert (store.path / ".iris-owner.lock").stat().st_ino == lock
