"""LABELLED controller/reader doubles + REAL storage/locks, NOT native/cold proof."""

import json
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_kernel_recovery as interval
from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as workflow
from iris.application import scientific_ablation_workflow_command as public
from iris.application import scientific_ablation_workflow_io as storage
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application import scientific_ablation_workflow_recovery as recovery
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowReturnedV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_recovery_v1 import (
    IrisScientificAblationWorkflowRecoveredResultV1,
    IrisScientificAblationWorkflowRecoveryAckV1,
    IrisScientificAblationWorkflowRecoveryIntentV1,
    ScientificAblationWorkflowRecoveryIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import process_file_lock
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_kernel_recovery import PROTOCOLS, heads
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_handoff import labelled_job, labelled_patches


@contextmanager
def labelled_recovery(
    tmp_path: Path, kind: str, controlled: bool, count: int = 1, total: int = 8
) -> Iterator[Any]:
    job, _, descriptor, source, _ = labelled_job(tmp_path)
    plan = storage.build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    definition = operator._command_definition(plan, _request(), source)
    store = storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "recovery-output", plan, definition
    )
    phase = next(
        v
        for v in scientific_ablation_workflow_phases_v1(plan)
        if v.kind == kind and (v.stream_index is not None) is controlled
    )
    anchors = tuple(f"{i + 1:064x}" for i in range(phase.index))
    job._seal = replace(
        job._seal,
        store=store,
        path=store.path,
        plan=steps.encoded(plan),
        anchors=anchors,
        storage_definition=store.namespace_definition_json,
    )
    namespace, domain, ack_domain = interval._PROTOCOLS[kind, controlled]
    payload = {"labelled_metadata_not_native_plan": True}
    plan_sha = domain_sha256(domain, canonical_json_bytes(payload))
    root = steps._kernel(store, phase)
    target = root / namespace / plan_sha
    chunk = Chunk(
        kernel_plan_sha256=plan_sha,
        acknowledgement_head_sha256=domain_sha256(ack_domain + "/genesis", plan_sha.encode()),
        acknowledged_rows=0,
        total_rows=total,
        rows_per_call=3,
        stop_row_index=3,
    )
    command = build(
        Intent(
            namespace=definition,
            phase=phase,
            completed_phase_sha256=anchors,
            expected_chunk=chunk,
            rows_per_call=3,
        )
    )
    expected = workflow._expected_for(job, phase, anchors, 3, chunk, producer_result_requested=True)
    state = {"count": count}
    entered: list[int] = []
    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        for module in (workflow, recovery):
            stack.enter_context(
                patch.object(
                    module, "_prefix", side_effect=lambda seal: (None,) * len(seal.anchors)
                )
            )
        job._state = "busy"
        try:
            with store.claim() as claim:
                handoff._Receiver._begin(job._seal, expected, 3, claim)
                store.begin(phase, anchors[-1] if anchors else None, claim)
                target.mkdir(parents=True)
                (target / "plan.json").write_bytes(
                    canonical_json_file_bytes(dict(sha256=plan_sha, payload=payload))
                )
        finally:
            job._state = "idle"

        def read(*args: Any, **kwargs: Any) -> Any:
            values: dict[str, Any] = dict(
                path=target,
                plan=SimpleNamespace(
                    sha256=plan_sha,
                    payload=SimpleNamespace(
                        rows=(None,) * total, input=SimpleNamespace(sha256="d" * 64)
                    ),
                ),
                rows=(None,) * state["count"],
                acknowledgements=tuple(SimpleNamespace(sha256=h) for h in heads(state["count"])),
                complete=SimpleNamespace(sha256="c" * 64) if state["count"] == total else None,
                uncommitted_stage_count=0,
            )
            progress_type: Any
            if kind == "capture":
                progress_type = (
                    controlled_capture.IrisScientificAblationControlledCaptureProgressV1
                    if controlled
                    else capture.IrisScientificAblationCaptureProgressV1
                )
            else:
                progress_type = (
                    controlled_score.IrisScientificAblationControlledEvaluationProgressV1
                    if controlled
                    else score.IrisScientificAblationEvaluationProgressV2
                )
                values["evaluation"] = SimpleNamespace(sha256="e" * 64)
            return progress_type(**values)

        stack.enter_context(
            patch.object(
                steps,
                "_score_args",
                return_value={
                    "expected_plan_sha256": "d" * 64,
                    "expected_capture_sha256": "f" * 64,
                },
            )
        )

        stack.enter_context(
            patch.object(
                kernels,
                "_capture_result" if kind == "capture" else "_score_result",
                side_effect=read,
            )
        )
        observe = recovery._observe

        def continued(*args: Any, **kwargs: Any) -> steps._Produced:
            assert kwargs["expected_chunk"] == chunk and args[5] == 3
            permit = kwargs["_recovery"]
            claim = args[4]
            with process_file_lock(target.parent / ".iris-owner.lock") as lock:
                actual = state["count"]
                stop = interval._locked_interval_stop(
                    permit,
                    root=root,
                    namespace=namespace,
                    expected_chunk=chunk,
                    plan_sha256=plan_sha,
                    acknowledgement_head_sha256=heads(actual)[-1]
                    if actual
                    else chunk.acknowledgement_head_sha256,
                    acknowledged_rows=actual,
                    total_rows=total,
                    rows_per_call=3,
                    acknowledgement_prefix=heads(actual),
                    kernel_lock=lock,
                    maximum_storage_bytes=steps.budget(store, target.parent, claim, 0),
                    minimum_free_bytes=claim.minimum_free_for(0),
                )
                entered.extend(range(actual, stop))
                state["count"] = stop
            labelled_prefix: Any = (None,) * phase.index
            snapshot = observe(job._seal, command, labelled_prefix, claim)
            return steps._Produced(None, (), steps.encoded(snapshot.kernel))

        produced = stack.enter_context(
            patch.object(steps, "execute_produced", side_effect=continued)
        )
        yield SimpleNamespace(
            job=job,
            facade=public._wrap(job),
            command=command,
            chunk=chunk,
            state=state,
            entered=entered,
            producer=produced,
            store=store,
            target=target,
            original=store.path / "handoffs" / command.sha256[:2] / command.sha256,
            operation="generate" if kind == "capture" else "score",
        )


def journal_path(item: Any) -> Path:
    roots = list((item.store.path / "recoveries").glob("*/*"))
    assert len(roots) == 1
    return cast(Path, roots[0])


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("count", (0, 1, 2, 3, 4, 8))
def test_same_original_stop_and_distinct_recovered_data(
    tmp_path: Path, kind: str, controlled: bool, count: int
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, count) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        result, ack = reply.result, reply.operator_ack
        p = result.payload
        assert p.kernel_progress.acknowledged_rows == 3
        assert p.kernel_progress.acknowledgement_head_sha256 == heads(3)[-1]
        assert p.observed_acknowledged_rows == max(count, 3)
        assert p.resumed_from_row == (count if count < 3 else None)
        assert p.original_return == "unavailable_not_reconstructed"
        assert p.original_request_sha256 == item.command.sha256
        assert ack.payload.recovered_result_sha256 == result.sha256
        assert tuple(item.entered) == tuple(range(count, 3))
        assert 3 not in item.entered and item.producer.call_count == int(count < 3)
        assert set(p.name for p in journal_path(item).iterdir()) == storage._RECOVERY_FILES
        assert {p.name for p in item.original.iterdir()} == {"intent.json"}
        with pytest.raises(ValueError):
            IrisScientificAblationWorkflowReturnedV1.model_validate_json(reply.result_json)
        with patch.object(steps, "execute_produced", side_effect=AssertionError("No new work")):
            again = item.facade.recover(
                item.command, operation=item.operation, expected_operator_ack_sha256=ack.sha256
            )
        assert again == reply and item.job._state == "idle"


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("lost", ("result.json", "operator-ack.json"))
@pytest.mark.parametrize("signal", (False, True))
def test_loss_then_storage_only_recovery_never_reexecutes_acknowledged_rows(
    tmp_path: Path, kind: str, controlled: bool, lost: str, signal: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        original_write = recovery._Journal.write

        def interrupted(journal: Any, name: str, value: Any) -> None:
            if name == lost:
                if signal:
                    raise KeyboardInterrupt("LABELLED primary boundary")
                raise OSError("LABELLED delivery failure")
            original_write(journal, name, value)

        with (
            patch.object(recovery._Journal, "write", interrupted),
            pytest.raises(
                KeyboardInterrupt if signal else public.IrisScientificAblationWorkflowCommandError
            ),
        ):
            item.facade.recover(item.command, operation=item.operation)
        assert item.entered == [1, 2] and item.job._state == "idle"
        with patch.object(
            steps, "execute_produced", side_effect=AssertionError("No remeasurement")
        ):
            reply = item.facade.recover(item.command, operation=item.operation)
        assert reply.result.payload.kernel_progress.acknowledged_rows == 3
        assert reply.result.payload.route == (
            "verified_kernel_data" if lost == "result.json" else "continued_original_interval"
        )
        assert not interval._LIVE and not interval._STARTS


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("intent", "received", "anchor", "operation", "ack"))
def test_wrong_original_data_or_explicit_operation_never_computes(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        kwargs: dict[str, Any] = {"operation": item.operation}
        command = item.command
        if fault == "intent":
            (item.original / "intent.json").write_bytes(b"{}\n")
        elif fault == "received":
            (item.original / "received.json").write_bytes(b"{}\n")
        elif fault == "anchor":
            chunk = item.chunk.model_copy(update={"acknowledgement_head_sha256": "e" * 64})
            command = build(item.command.payload.model_copy(update={"expected_chunk": chunk}))
        elif fault == "operation":
            kwargs["operation"] = "publish"
        else:
            kwargs["expected_operator_ack_sha256"] = "a" * 64
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            item.facade.recover(command, **kwargs)
        item.producer.assert_not_called()


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_whole_quota_refusal_is_before_kernel_continuation(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        (item.store.path / (".iris-recovery-" + "a" * 32 + ".tmp")).write_bytes(
            b"x" * (15 * 1024**2)
        )
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            item.facade.recover(item.command, operation=item.operation)
        item.producer.assert_not_called()
        assert item.state["count"] == 1


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("name", ("intent.json", "result.json", "operator-ack.json"))
def test_retained_record_mutation_cannot_start_new_work(
    tmp_path: Path, kind: str, controlled: bool, name: str
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        target = journal_path(item) / name
        target.write_bytes(target.read_bytes() + b" ")
        item.producer.reset_mock()
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
        item.producer.assert_not_called()


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_duplicate_workflow_object_cannot_restore_original_binding_or_permission(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        old = item.job._seal
        reopened = object.__new__(workflow.IrisScientificAblationWorkflowV1)
        reopened._state = "idle"
        reopened._seal = replace(old, job=reopened, ready=None, completed_handoff=None)
        # LABELLED same-process object reconstruction, NOT an actual cold factory/process proof.
        with (
            patch.object(steps, "execute_produced", side_effect=AssertionError("DATA only")),
            pytest.raises(storage.IrisScientificAblationWorkflowStorageError) as caught,
        ):
            recovery._recover_kernel_command_v1(
                reopened, item.command, expected_operator_ack_sha256=reply.operator_ack.sha256
            )
        context = caught.value.__context__
        assert isinstance(context, operator.IrisScientificAblationOperatorWorkflowError)
        assert isinstance(context.__context__, ValueError)
        assert "binding identity differs" in str(context.__context__)
        assert not interval._LIVE
        assert (
            item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
            == reply
        )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("loss", (None, "result.json", "operator-ack.json"))
def test_complete_kernel_data_publishes_phase_and_distinct_ack_without_native(
    tmp_path: Path, kind: str, controlled: bool, loss: str | None
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, 3, 3) as item:
        original_write = recovery._Journal.write

        def interrupted(journal: Any, name: str, value: Any) -> None:
            if name == loss:
                raise OSError("LABELLED compact DATA publication failure")
            original_write(journal, name, value)

        if loss is not None:
            with (
                patch.object(recovery._Journal, "write", interrupted),
                pytest.raises(public.IrisScientificAblationWorkflowCommandError),
            ):
                item.facade.recover(item.command, operation=item.operation)
        reply = item.facade.recover(item.command, operation=item.operation)
        record = reply.result.payload.phase_record
        assert record is not None and record.payload.origin == "kernel_data"
        assert reply.operator_ack.payload.completed_phase_sha256 == (
            *item.command.payload.completed_phase_sha256,
            record.sha256,
        )
        phase = item.command.payload.phase
        assert item.store.read(phase, expected_sha256=record.sha256).record == record
        assert item.job._seal.anchors == reply.operator_ack.payload.completed_phase_sha256
        assert (
            item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
            == reply
        )
        item.producer.assert_not_called()


@pytest.mark.parametrize("field", ("authority", "original_return", "observed_acknowledged_rows"))
def test_recovered_schema_rejects_ownership_or_bool_claims(tmp_path: Path, field: str) -> None:
    with labelled_recovery(tmp_path, "capture", False, 3) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        value = json.loads(reply.result_json)
        value["payload"][field] = True
        value["sha256"] = domain_sha256(
            "iris/" + value["payload"]["kind"], canonical_json_bytes(value["payload"])
        )
        with pytest.raises(ValueError):
            IrisScientificAblationWorkflowRecoveredResultV1.model_validate(value)


def test_closed_recovery_metadata_bound_and_types_are_distinct() -> None:
    assert recovery._CAPACITY == 3 * 512 * 1024
    assert {"intent.json", "result.json", "operator-ack.json"} == storage._RECOVERY_FILES
    assert IrisScientificAblationWorkflowRecoveryIntentV1.__name__ != (
        IrisScientificAblationWorkflowRecoveryAckV1.__name__
    )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_saved_observed_prefix_cannot_be_silently_rolled_back(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, 8) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        item.state["count"] = 3
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
        item.producer.assert_not_called()


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("name", ("result.json", "operator-ack.json"))
def test_real_metadata_rename_then_primary_signal_uses_only_retained_data_on_retry(
    tmp_path: Path, kind: str, controlled: bool, name: str
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, 3) as item:
        original = os.rename

        def renamed_then_signal(src: Any, dest: Any) -> None:
            original(src, dest)
            if Path(dest).name == name:
                raise KeyboardInterrupt("LABELLED post-rename primary signal")

        with patch.object(os, "rename", renamed_then_signal), pytest.raises(KeyboardInterrupt):
            item.facade.recover(item.command, operation=item.operation)
        assert (journal_path(item) / name).is_file()
        reply = item.facade.recover(item.command, operation=item.operation)
        assert reply.result.payload.route == "verified_kernel_data"
        item.producer.assert_not_called()


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_recovery_envelope_keeps_strict_typed_tuples_not_python_json_lists(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled) as item:
        payload = ScientificAblationWorkflowRecoveryIntentPayloadV1(original_command=item.command)
        value = payload.model_dump(mode="json")
        digest = domain_sha256("iris/" + value["kind"], canonical_json_bytes(value))
        # Reproduce the original defect directly, without a public wrapper hiding its cause.
        with pytest.raises(ValueError, match="tuple"):
            IrisScientificAblationWorkflowRecoveryIntentV1.model_validate(
                dict(sha256=digest, payload=value)
            )
        actual = recovery._envelope(payload, IrisScientificAblationWorkflowRecoveryIntentV1)
        assert actual.sha256 == digest and actual.payload == payload
        assert actual.payload.original_command == item.command
        assert (
            IrisScientificAblationWorkflowRecoveryIntentV1.model_validate_json(
                steps.encoded(actual)
            )
            == actual
        )
        item.producer.assert_not_called()
        assert not interval._LIVE
