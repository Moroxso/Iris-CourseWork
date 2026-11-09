"""LABELLED metadata/controller/loop probes with REAL OS locks, NOT row/NN proof.

Synthetic plan/ACK heads and patched source readers are intentional. Actual four
kernels stop BEFORE their first missing-row stage; no fake native acceptance.
"""

import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from iris.application import _scientific_ablation_kernel_recovery as recovery
from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as ordinary_capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as ordinary_score
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as workflow
from iris.application.scientific_ablation_capture import IrisScientificAblationCaptureError
from iris.application.scientific_ablation_controlled_capture import (
    IrisScientificAblationControlledCaptureError,
)
from iris.application.scientific_ablation_controlled_evaluation import (
    IrisScientificAblationControlledEvaluationError,
)
from iris.application.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationError
from iris.application.scientific_ablation_kernel_chunk import check_kernel_chunk_prefix_v1
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import process_file_lock
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_handoff import labelled_job, labelled_patches

PROTOCOLS = (("capture", False), ("capture", True), ("score", False), ("score", True))


@contextmanager
def labelled_permission(tmp_path: Path, kind: str, controlled: bool) -> Iterator[Any]:
    job, _, descriptor, source, _ = labelled_job(tmp_path)
    declared = intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    declared = declared.model_copy(
        update={"resources": declared.resources.model_copy(update={"minimum_free_bytes": 4096})}
    )
    p = build_scientific_ablation_workflow_plan_v1(declared)
    definition = operator._command_definition(p, _request(), source)
    store = ScientificAblationWorkflowStoreV1._for_commands(tmp_path / "output", p, definition)
    phase = next(
        v
        for v in scientific_ablation_workflow_phases_v1(p)
        if v.kind == kind and (v.stream_index is not None) is controlled
    )
    anchors = tuple(f"{i + 1:064x}" for i in range(phase.index))
    job._seal = replace(
        job._seal,
        store=store,
        path=store.path,
        plan=steps.encoded(p),
        anchors=anchors,
        storage_definition=store.namespace_definition_json,
    )
    namespace, plan_domain, ack_domain = recovery._PROTOCOLS[kind, controlled]
    plan_payload = {"labelled_metadata_not_a_native_plan": True}
    plan_sha = domain_sha256(plan_domain, canonical_json_bytes(plan_payload))
    kernel_root = steps._kernel(store, phase)
    target = kernel_root / namespace / plan_sha
    interval = Chunk(
        kernel_plan_sha256=plan_sha,
        acknowledgement_head_sha256=domain_sha256(ack_domain + "/genesis", plan_sha.encode()),
        acknowledged_rows=0,
        total_rows=8,
        rows_per_call=3,
        stop_row_index=3,
    )
    command = build_scientific_ablation_workflow_command_intent_v1(
        Intent(
            namespace=definition,
            phase=phase,
            completed_phase_sha256=anchors,
            expected_chunk=interval,
            rows_per_call=3,
        )
    )
    expected = workflow._expected_for(
        job, phase, anchors, 3, interval, producer_result_requested=True
    )
    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        stack.enter_context(patch.object(workflow, "_prefix", return_value=(None,) * phase.index))
        job._state = "busy"
        try:
            with store.claim() as claim:
                handoff._Receiver._begin(job._seal, expected, 3, claim)
                store.begin(phase, anchors[-1] if anchors else None, claim)
                target.mkdir(parents=True)
                (target / "plan.json").write_bytes(
                    canonical_json_file_bytes(dict(sha256=plan_sha, payload=plan_payload))
                )
                with recovery._original_interval_permission(
                    job._seal, command, claim
                ) as permission:
                    yield SimpleNamespace(
                        job=job,
                        command=command,
                        interval=interval,
                        permission=permission,
                        root=kernel_root,
                        namespace=namespace,
                        target=target,
                        maximum=steps.budget(store, target.parent, claim, 0),
                        floor=claim.minimum_free_for(0),
                    )
        finally:
            job._state = "idle"


def heads(count: int) -> tuple[str, ...]:
    return tuple(sha256(f"LABELLED ACK {i}".encode()).hexdigest() for i in range(count))


def arguments(item: Any, count: int = 1) -> dict[str, Any]:
    chain = heads(count)
    return dict(
        root=item.root,
        namespace=item.namespace,
        expected_chunk=item.interval,
        plan_sha256=item.interval.kernel_plan_sha256,
        acknowledgement_head_sha256=chain[-1]
        if chain
        else item.interval.acknowledgement_head_sha256,
        acknowledged_rows=count,
        total_rows=8,
        rows_per_call=3,
        acknowledgement_prefix=chain,
        maximum_storage_bytes=item.maximum,
        minimum_free_bytes=item.floor,
    )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("count", (0, 1, 2, 3))
def test_original_stop_never_becomes_current_plus_budget(
    tmp_path: Path, kind: str, controlled: bool, count: int
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        params = arguments(item, count)
        assert recovery._locked_interval_stop(item.permission, **params, kernel_lock=lock) == 3
        if count:
            with pytest.raises(ValueError):
                check_kernel_chunk_prefix_v1(
                    item.interval,
                    **{
                        key: params[key]
                        for key in (
                            "plan_sha256",
                            "acknowledgement_head_sha256",
                            "acknowledged_rows",
                            "total_rows",
                            "rows_per_call",
                        )
                    },
                )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("ahead", "behind", "bool", "head", "prefix", "total"))
def test_outside_or_wrong_locked_prefix_is_not_rebased(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        params = arguments(item, 4 if fault == "ahead" else 1)
        if fault == "behind":
            params["acknowledged_rows"] = -1
        elif fault == "bool":
            params["acknowledged_rows"] = True
        elif fault == "head":
            params["acknowledgement_head_sha256"] = "f" * 64
        elif fault == "prefix":
            params["acknowledgement_prefix"] = ()
        elif fault == "total":
            params["total_rows"] = 9
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(item.permission, **params, kernel_lock=lock)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("budget", "plan", "namespace", "root"))
def test_unchanged_original_request_and_namespace_are_mandatory(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        params = arguments(item)
        if fault == "budget":
            params.update(
                expected_chunk=Chunk.model_validate(
                    item.interval.model_dump() | dict(rows_per_call=2, stop_row_index=2)
                ),
                rows_per_call=2,
            )
        elif fault == "plan":
            params["plan_sha256"] = "f" * 64
        elif fault == "namespace":
            params["namespace"] = "other"
        else:
            params["root"] = tmp_path / "other"
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(item.permission, **params, kernel_lock=lock)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("object", "cold_object"))
def test_constructed_object_cannot_mint_live_permission(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        forged: Any = (
            object() if fault == "object" else object.__new__(recovery._KernelIntervalPermission)
        )
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(forged, **arguments(item), kernel_lock=lock)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_permission_is_revoked_after_original_context_exit(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_permission(tmp_path, kind, controlled) as item:
        permit = item.permission
    with pytest.raises(ValueError):
        recovery._checked_interval_permission(
            permit,
            root=item.root,
            namespace=item.namespace,
            expected_chunk=item.interval,
            plan_sha256=item.interval.kernel_plan_sha256,
            rows_per_call=3,
            maximum_storage_bytes=item.maximum,
            minimum_free_bytes=item.floor,
        )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_closed_kernel_lock_cannot_authorize_continuation(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_permission(tmp_path, kind, controlled) as item:
        with process_file_lock(item.target.parent / ".iris-owner.lock") as lock:
            pass
        with pytest.raises((OSError, ValueError)):
            recovery._locked_interval_stop(item.permission, **arguments(item), kernel_lock=lock)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize(
    "name", ("received.json", "producer-ack.json", "returned.json", "operator-ack.json")
)
def test_retained_producer_data_requires_storage_only_recovery(
    tmp_path: Path, kind: str, controlled: bool, name: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        root = item.job._seal.path / "handoffs" / item.command.sha256[:2] / item.command.sha256
        (root / name).write_bytes(b"{}\n")
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(item.permission, **arguments(item), kernel_lock=lock)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("intent", "plan"))
def test_changed_original_bytes_reject_before_continuation(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        path = (
            item.target / "plan.json"
            if fault == "plan"
            else item.job._seal.path
            / "handoffs"
            / item.command.sha256[:2]
            / item.command.sha256
            / "intent.json"
        )
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(item.permission, **arguments(item), kernel_lock=lock)


def module_for(kind: str, controlled: bool) -> Any:
    return (
        (controlled_capture if controlled else ordinary_capture)
        if kind == "capture"
        else (controlled_score if controlled else ordinary_score)
    )


def error_for(kind: str, controlled: bool) -> type[Exception]:
    if kind == "capture":
        return (
            IrisScientificAblationControlledCaptureError
            if controlled
            else IrisScientificAblationCaptureError
        )
    return (
        IrisScientificAblationControlledEvaluationError
        if controlled
        else IrisScientificAblationEvaluationError
    )


def invoke(
    module: Any,
    kind: str,
    controlled: bool,
    root: Path,
    chunk: Chunk,
    permit: Any,
    maximum: int = 16 * 1024**2,
    floor: int = 0,
) -> Any:
    common = dict(
        root=root,
        rows_per_call=3,
        expected_chunk=chunk,
        _recovery=permit,
        maximum_storage_bytes=maximum,
        minimum_free_bytes=floor,
    )
    if kind == "capture":
        method = (
            module.capture_scientific_ablation_controlled_heldout_v1
            if controlled
            else module.capture_scientific_ablation_heldout_v1
        )
        return method(
            Mock(),
            expected_input_sha256="a" * 64,
            expected_plan_sha256=chunk.kernel_plan_sha256,
            **common,
        )
    method = (
        module.evaluate_scientific_ablation_controlled_capture_resumable_v1
        if controlled
        else module.evaluate_scientific_ablation_capture_resumable_v2
    )
    return method(
        expected_evaluation_plan_sha256=chunk.kernel_plan_sha256,
        capture_path=root / "input",
        expected_plan_sha256="a" * 64,
        expected_capture_sha256="b" * 64,
        lifecycle_path=root / "life",
        inference_path=root / "inference",
        inputs_root=root / "inputs",
        training=Mock(),
        validation=Mock(),
        test=Mock(),
        python_policy=None,
        **common,
    )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_forged_permission_is_denied_before_any_kernel_plan_or_input_work(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    module = module_for(kind, controlled)
    builder = (
        (
            "build_scientific_ablation_controlled_capture_plan_v1"
            if controlled
            else "build_scientific_ablation_capture_plan_v1"
        )
        if kind == "capture"
        else "_prepare"
    )
    chunk = Chunk(
        kernel_plan_sha256="a" * 64,
        acknowledgement_head_sha256="b" * 64,
        acknowledged_rows=0,
        total_rows=8,
        rows_per_call=3,
        stop_row_index=3,
    )
    with (
        patch.object(
            module, "_checked_interval_permission", wraps=module._checked_interval_permission
        ) as guard,
        patch.object(module, builder) as plan,
        patch.object(module, "process_file_lock") as lock,
        pytest.raises(error_for(kind, controlled)),
    ):
        invoke(module, kind, controlled, tmp_path / "kernel", chunk, object())
    assert (
        guard.call_count == 1
        and plan.call_count == lock.call_count == 0
        and not (tmp_path / "kernel").exists()
    )


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_each_real_kernel_reaches_only_missing_row_one_with_original_stop_three(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    module = module_for(kind, controlled)
    reached: dict[str, Any] = {}

    def before_stage(*args: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        reached.update(index=frame.f_locals["index"], stop=frame.f_locals["stop"])
        raise RuntimeError("LABELLED loop boundary, not actual native execution")

    with labelled_permission(tmp_path, kind, controlled) as item, ExitStack() as stack:
        plan = SimpleNamespace(
            sha256=item.interval.kernel_plan_sha256,
            payload=SimpleNamespace(
                rows=tuple(range(8)),
                maximum_row_file_bytes=8 * 1024**2,
                maximum_score_row_file_bytes=65536,
            ),
        )
        progress = SimpleNamespace(
            plan=plan,
            rows=(object(),),
            acknowledgements=(SimpleNamespace(sha256=heads(1)[0]),),
            complete=None,
        )
        stack.enter_context(patch.object(module, "_read", return_value=progress))
        if kind == "capture":
            builder = (
                "build_scientific_ablation_controlled_capture_plan_v1"
                if controlled
                else "build_scientific_ablation_capture_plan_v1"
            )
            stack.enter_context(patch.object(module, builder, return_value=plan))
        else:
            inputs = SimpleNamespace(
                rows=tuple(range(8)),
                capture=SimpleNamespace(path=tmp_path / "capture"),
                check=Mock(),
            )
            stack.enter_context(patch.object(module, "_prepare", return_value=inputs))
            stack.enter_context(patch.object(module, "_plan", return_value=plan))
            stack.enter_context(patch.object(module, "_capture_stamp", return_value=()))
        stack.enter_context(patch.object(module, "_stage", new=before_stage))
        with pytest.raises(error_for(kind, controlled)) as caught:
            invoke(
                module,
                kind,
                controlled,
                item.root,
                item.interval,
                item.permission,
                item.maximum,
                item.floor,
            )
        assert reached == dict(index=1, stop=3), repr(caught.value.__cause__)


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_primary_signal_still_revokes_live_permission(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    permit = None
    with pytest.raises(KeyboardInterrupt), labelled_permission(tmp_path, kind, controlled) as item:
        permit = item.permission
        raise KeyboardInterrupt("Primary labelled signal")
    assert permit is not None and permit not in recovery._LIVE


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
@pytest.mark.parametrize("fault", ("whole_quota", "free_floor"))
def test_recovery_cannot_bypass_original_whole_quota_or_free_floor(
    tmp_path: Path, kind: str, controlled: bool, fault: str
) -> None:
    with (
        labelled_permission(tmp_path, kind, controlled) as item,
        process_file_lock(item.target.parent / ".iris-owner.lock") as lock,
    ):
        params = arguments(item)
        if fault == "whole_quota":
            params["maximum_storage_bytes"] += 1
        else:
            params["minimum_free_bytes"] -= 1
        with pytest.raises(ValueError):
            recovery._locked_interval_stop(item.permission, **params, kernel_lock=lock)
