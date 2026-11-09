"""Real storage/controller faults with LABELLED producer doubles, NOT native proof."""

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as owner
from iris.application import scientific_ablation_workflow_io as storage
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from tests.unit._workflow_data_setup import plan
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_operator_operation_preflight import labelled_binding
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk


def labelled_job(tmp_path: Path) -> tuple[Any, Any, Any, Any, Any]:
    job, binding, descriptor, source = labelled_binding(tmp_path)
    definition = operator._command_definition(job._seal.store.plan, _request(), source)
    store = storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "output", job._seal.store.plan, definition
    )
    job._seal = replace(
        job._seal, store=store, path=store.path, storage_definition=store.namespace_definition_json
    )
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    return job, binding, descriptor, source, phase


def labelled_patches(job: Any, descriptor: Any, source: Any, stack: ExitStack) -> None:
    stack.enter_context(
        patch.object(operator, "_bind", return_value=(job._seal.store.plan, source))
    )
    stack.enter_context(
        patch.object(
            operator,
            "inspect_scientific_ablation_operator_v1",
            return_value=type("Labelled", (), {"descriptor": descriptor})(),
        )
    )
    stack.enter_context(patch.object(owner, "_check_chunk"))
    stack.enter_context(patch.object(steps, "validate_data"))


def invoke(
    job: owner.IrisScientificAblationWorkflowV1, phase: ScientificAblationWorkflowPhaseV1
) -> owner.ScientificAblationWorkflowAdvanceResultV1:
    return job.advance_expected_with_result(
        expected_phase=phase,
        expected_completed_phase_sha256=(),
        expected_chunk=chunk(),
        rows_per_call=1,
    )


def test_original_store_prepared_capability_publishes_same_blobs_without_copy(
    tmp_path: Path,
) -> None:
    store = storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    data = store.make_data(
        phase, None, origin="kernel_data", files={"capture.json": b'"DATA only"\n'}
    )
    with store.claim() as claim:
        store.begin(phase, None, claim)
        prepared = store._prepare(data, claim)
        before = storage._tree_size(store.path)[0]
        inode = (prepared.path / "capture.json").stat().st_ino
        assert not (store.path / "phase-000").exists()
        assert store.commit(data, claim) == data
        assert storage._tree_size(store.path)[0] == before
        assert (store.path / "phase-000/capture.json").stat().st_ino == inode


def test_constructible_or_cold_prepared_data_cannot_restore_live_stage_capability(
    tmp_path: Path,
) -> None:
    store = storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", plan(tmp_path))
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    data = store.make_data(phase, None, origin="kernel_data", files={"capture.json": b'"DATA"\n'})
    with store.claim() as claim:
        store.begin(phase, None, claim)
        prepared = store._prepare(data, claim)
        forged = replace(prepared)
        with pytest.raises(ValueError):
            store._publish_prepared(forged, claim)
    reopened = storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", store.plan)
    with reopened.claim() as claim, pytest.raises(ValueError):
        reopened._publish_prepared(prepared, claim)
    assert prepared.path.is_dir() and not (store.path / "phase-000").exists()


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize(
    "loss", ("received.json", "producer-ack.json", "returned.json", "operator-ack.json")
)
@pytest.mark.parametrize("signal", (False, True))
def test_exact_original_reference_survives_handoff_fault_without_reexecuting_labelled_producer(
    tmp_path: Path, partial: bool, loss: str, signal: bool
) -> None:
    job, _, descriptor, source, phase = labelled_job(tmp_path)
    store = job._seal.store
    data = (
        None
        if partial
        else store.make_data(
            phase,
            None,
            origin="kernel_data",
            files={"capture.json": b'"LABELLED controller DATA"\n'},
        )
    )
    produced = steps._Produced(
        None if data is None else steps.encoded(data.record),
        () if data is None else data.files,
        b'{"labelled_kernel_snapshot_not_native_evidence":true}\n',
    )
    original_write = handoff._Receiver._write

    def interrupted(
        self: handoff._Receiver, name: str, raw: bytes, seal: Any, claim: storage._Claim
    ) -> None:
        if name == loss:
            raise KeyboardInterrupt() if signal else RuntimeError("Labelled storage boundary")
        original_write(self, name, raw, seal, claim)

    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        native = stack.enter_context(patch.object(steps, "execute_produced", return_value=produced))
        with (
            patch.object(handoff._Receiver, "_write", interrupted),
            pytest.raises(
                KeyboardInterrupt if signal else storage.IrisScientificAblationWorkflowStorageError
            ),
        ):
            invoke(job, phase)
        assert native.call_count == 1 and job._state == "idle" and job._seal.ready is not None
        assert job._seal.ready.produced is produced
        receiver = job._seal.ready.receiver
        assert type(receiver) is handoff._Receiver
        inode = (
            (store.path / "phase-000/capture.json").stat().st_ino
            if not partial and loss in ("returned.json", "operator-ack.json")
            else None
        )
        native.side_effect = AssertionError("Storage-only retry must not call the producer")
        result = invoke(job, phase)
        assert native.call_count == 1 and result.kernel_progress_json == produced.kernel_progress
        assert job._seal.ready is None
        assert len(job.progress.completed_phase_sha256) == (0 if partial else 1)
        assert receiver._returned is result
        assert {p.name for p in receiver._path.iterdir()} == storage._HANDOFF_FILES
        received = json.loads((receiver._path / "received.json").read_bytes())
        assert produced.kernel_progress is not None
        assert received["payload"]["original_kernel_progress"] == json.loads(
            produced.kernel_progress
        )
        if inode is not None:
            assert (store.path / "phase-000/capture.json").stat().st_ino == inode
        assert not any(store.path.glob(".iris-handoff-*.tmp"))
        assert not any(store.path.glob(".iris-workflow-*.tmp"))


def test_receiver_rejects_equal_constructed_produced_object_and_keeps_original(
    tmp_path: Path,
) -> None:
    job, _, descriptor, source, phase = labelled_job(tmp_path)
    produced = steps._Produced(None, (), b'{"labelled":true}\n')
    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        stack.enter_context(patch.object(steps, "execute_produced", return_value=produced))
        stack.enter_context(
            patch.object(
                handoff._Receiver, "_finish", side_effect=RuntimeError("Hold original DATA")
            )
        )
        with pytest.raises(storage.IrisScientificAblationWorkflowStorageError):
            invoke(job, phase)
        seal = job._seal
        assert seal.ready is not None and seal.ready.receiver is not None
        job._state = "busy"
        try:
            with seal.store.claim() as claim:
                seal.ready.receiver._resume_claim(seal, claim)
                with pytest.raises(ValueError):
                    seal.ready.receiver._receive(replace(produced), None, seal, claim)
                assert seal.ready.receiver._produced is produced
        finally:
            job._state = "idle"


def test_real_metadata_rename_then_signal_keeps_exact_live_publication_capability(
    tmp_path: Path,
) -> None:
    job, _, descriptor, source, phase = labelled_job(tmp_path)
    produced = steps._Produced(None, (), b'{"labelled":true}\n')
    original_rename = os.rename

    def renamed_then_signal(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        original_rename(src, dst, *args, **kwargs)
        if Path(dst).name == "producer-ack.json":
            raise SystemExit(19)

    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        native = stack.enter_context(patch.object(steps, "execute_produced", return_value=produced))
        with patch.object(os, "rename", renamed_then_signal), pytest.raises(SystemExit):
            invoke(job, phase)
        assert job._seal.ready is not None and job._seal.ready.receiver is not None
        receiver = job._seal.ready.receiver
        assert "producer-ack.json" in receiver._publishing
        inode = (receiver._path / "producer-ack.json").stat().st_ino
        native.side_effect = AssertionError("No remeasurement after a real rename")
        result = invoke(job, phase)
        assert receiver._returned is result and native.call_count == 1
        assert (receiver._path / "producer-ack.json").stat().st_ino == inode
        assert not receiver._publishing and not tuple(job._seal.path.glob(".iris-handoff-*.tmp"))


def test_hold_is_exact_closed_metadata_worst_case_without_an_extra_native_blob_copy() -> None:
    assert len(storage._HANDOFF_FILES) == 5
    assert handoff._CAPACITY == len(storage._HANDOFF_FILES) * handoff._MAX_RECORD == 2621440
    bounded = handoff._record(
        "labelled-metadata-bound-probe-v1", {"blob": "x" * (handoff._MAX_RECORD - 1024)}
    )
    assert len(bounded) <= handoff._MAX_RECORD
    with pytest.raises(ValueError):
        handoff._record("labelled-metadata-bound-probe-v1", {"blob": "x" * handoff._MAX_RECORD})
