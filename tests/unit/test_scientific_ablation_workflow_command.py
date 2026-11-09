"""Real storage/LABELLED producer controller gates, NOT actual native evidence."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as workflow
from iris.application import scientific_ablation_workflow_command as public
from iris.application import scientific_ablation_workflow_io as storage
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1 as Kernel,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from tests.unit._workflow_data_setup import plan
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk
from tests.unit.test_scientific_ablation_workflow_handoff import labelled_job, labelled_patches
from tests.unit.test_scientific_ablation_workflow_replay import payload


def setup(tmp_path: Path, partial: bool = True) -> tuple[Any, ...]:
    job, _, descriptor, source, phase = labelled_job(tmp_path)
    facade = public._wrap(job)
    interval = chunk() if partial else chunk(rows_per_call=2, stop_row_index=2)
    command = build(
        Intent(
            namespace=facade.definition,
            phase=phase,
            completed_phase_sha256=(),
            expected_chunk=interval,
            rows_per_call=interval.rows_per_call,
        )
    )
    data = (
        None
        if partial
        else job._seal.store.make_data(
            phase, None, origin="kernel_data", files={"capture.json": b'{"labelled_only":true}\n'}
        )
    )
    kernel = Kernel(
        workflow_plan_sha256=job._seal.store.plan.sha256,
        phase=phase,
        completed_phase_sha256=(),
        phase_is_acknowledged=False,
        kernel_plan_sha256=interval.kernel_plan_sha256,
        acknowledgement_head_sha256="b" * 64,
        acknowledged_rows=interval.stop_row_index,
        total_rows=2,
        kernel_complete_sha256=None if partial else "c" * 64,
        uncommitted_stage_count=0,
    )
    produced = steps._Produced(
        None if data is None else steps.encoded(data.record),
        () if data is None else data.files,
        steps.encoded(kernel),
    )
    return facade, job, descriptor, source, command, produced


def test_metadata_cannot_construct_command_owner(tmp_path: Path) -> None:
    with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
        public.IrisScientificAblationOperatorCommandWorkflowV1()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", (None, True, "", "Generate", "score", "diagnose", "publish"))
def test_wrong_operation_rejects_before_owner_source_or_files(
    tmp_path: Path, operation: Any
) -> None:
    command = build(payload(tmp_path))
    facade = object.__new__(public.IrisScientificAblationOperatorCommandWorkflowV1)
    with (
        patch.object(public, "_entry") as entered,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.dispatch(command, operation=operation)
    assert entered.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ("hash", "hidden", "private", "namespace"))
def test_unchecked_command_rejects_before_owner_or_source(tmp_path: Path, fault: str) -> None:
    command = build(payload(tmp_path))
    if fault == "hash":
        vars(command)["sha256"] = "f" * 64
    elif fault == "hidden":
        vars(command.payload)["completion_callback"] = object()
    elif fault == "private":
        object.__setattr__(command, "__pydantic_private__", {"native_owner": object()})
    else:
        vars(command.payload.namespace.payload)["authority"] = "native_owner"
    facade = object.__new__(public.IrisScientificAblationOperatorCommandWorkflowV1)
    with (
        patch.object(public, "_entry") as entered,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.dispatch(command, operation="generate")
    assert entered.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", (True, "F" * 64, "a" * 63, "a" * 64 + "\n"))
def test_bad_replay_anchor_rejects_before_owner_or_source(tmp_path: Path, bad: Any) -> None:
    facade = object.__new__(public.IrisScientificAblationOperatorCommandWorkflowV1)
    with (
        patch.object(public, "_entry") as entered,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.dispatch(
            build(payload(tmp_path)), operation="generate", expected_operator_ack_sha256=bad
        )
    assert entered.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ("path", "export_root", "base", "root"))
def test_relative_creation_roots_reject_before_composition(tmp_path: Path, name: str) -> None:
    arguments: dict[str, Any] = dict(
        path=tmp_path / "operator",
        export_root=tmp_path / "exports",
        base=tmp_path,
        root=tmp_path / "output",
        request=_request(),
        plan=plan(tmp_path),
    )
    arguments[name] = Path("relative")
    with (
        patch.object(public, "_create_operator_workflow") as compose,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        public.create_scientific_ablation_operator_command_workflow_v1(**arguments)
    assert compose.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("prefix", ([], ("F" * 64,), ("a" * 64,) * 135))
def test_bad_creation_prefix_rejects_before_composition(tmp_path: Path, prefix: Any) -> None:
    with (
        patch.object(public, "_create_operator_workflow") as compose,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        public.create_scientific_ablation_operator_command_workflow_v1(
            tmp_path / "operator",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=plan(tmp_path),
            base=tmp_path,
            root=tmp_path / "output",
            completed_phase_sha256=prefix,
        )
    assert compose.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("partial", (True, False))
@pytest.mark.parametrize("window", ("ack_write", "finish", "delivery", "release"))
@pytest.mark.parametrize("signal", (False, True))
def test_actual_live_completion_survives_controller_fault_without_second_producer(
    tmp_path: Path, partial: bool, window: str, signal: bool
) -> None:
    facade, job, descriptor, source, command, produced = setup(tmp_path, partial)
    store = job._seal.store
    original_write, original_finish = handoff._Receiver._write, handoff._Receiver._finish
    original_claim = storage.ScientificAblationWorkflowStoreV1.claim

    def loss(*args: Any, **kwargs: Any) -> NoReturn:
        raise (
            KeyboardInterrupt("Primary labelled signal")
            if signal
            else RuntimeError("Labelled delivery loss")
        )

    def write(self: Any, name: str, raw: bytes, seal: Any, claim: Any) -> None:
        original_write(self, name, raw, seal, claim)
        if name == "operator-ack.json":
            loss()

    def finish(self: Any, result: Any, seal: Any, claim: Any) -> None:
        original_finish(self, result, seal, claim)
        assert self._completion is not None and self._completion.result is result
        claim.lock.check()  # Completion really precedes release of the original OS claim.
        loss()

    @contextmanager
    def claim_after_yield(
        self: storage.ScientificAblationWorkflowStoreV1,
    ) -> Iterator[storage._Claim]:
        with original_claim(self) as claim:
            yield claim
            loss()

    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        producer = stack.enter_context(
            patch.object(steps, "execute_produced", return_value=produced)
        )
        with ExitStack() as failure:
            if window == "ack_write":
                failure.enter_context(patch.object(handoff._Receiver, "_write", write))
            elif window == "finish":
                failure.enter_context(patch.object(handoff._Receiver, "_finish", finish))
            elif window == "delivery":
                failure.enter_context(patch.object(public, "_reply", side_effect=loss))
            else:
                failure.enter_context(
                    patch.object(
                        storage.ScientificAblationWorkflowStoreV1, "claim", claim_after_yield
                    )
                )
            with pytest.raises(
                KeyboardInterrupt if signal else public.IrisScientificAblationWorkflowCommandError
            ):
                facade.dispatch(command, operation="generate")
        assert producer.call_count == 1 and facade._state == job._state == "idle"
        receiver = (
            job._seal.ready.receiver
            if job._seal.ready is not None
            else job._seal.completed_handoff.receiver
        )
        original_result = receiver._returned
        assert original_result is not None
        producer.side_effect = AssertionError("No repeated labelled producer")
        with ExitStack() as retry:
            if window in ("delivery", "release"):
                retry.enter_context(
                    patch.object(
                        workflow,
                        "_advance",
                        side_effect=AssertionError("No second original advance"),
                    )
                )
                retry.enter_context(
                    patch(
                        "iris.application.scientific_ablation_workflow_replay._metadata",
                        side_effect=AssertionError("No learned FS ACK"),
                    )
                )
            reply = facade.dispatch(command, operation="generate")
        assert reply.original_result is original_result and reply.delivery == "original_reference"
        assert reply.request_sha256 == command.sha256 == reply.operator_ack.payload.request_sha256
        assert receiver._completion is job._seal.completed_handoff
        assert producer.call_count == 1 and job._seal.ready is None
        assert (
            store.path / "handoffs" / command.sha256[:2] / command.sha256 / "operator-ack.json"
        ).read_bytes() == reply.operator_ack_json


@pytest.mark.parametrize("fault", ("duplicate", "token"))
def test_constructed_completion_is_not_the_receiver_minted_reference(
    tmp_path: Path, fault: str
) -> None:
    facade, job, descriptor, source, command, produced = setup(tmp_path)
    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        stack.enter_context(patch.object(steps, "execute_produced", return_value=produced))
        facade.dispatch(command, operation="generate")
        actual = job._seal.completed_handoff
        assert actual is not None
        forged = replace(actual, **({"token": object()} if fault == "token" else {}))
        with pytest.raises(ValueError):
            public._reply(facade._seal, command, forged)


def test_reentrant_dispatch_is_denied_without_changing_outer_owner(tmp_path: Path) -> None:
    facade, job, descriptor, source, command, produced = setup(tmp_path)

    def execute(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            facade.dispatch(command, operation="generate")
        assert facade._state == "busy" and job._state == "busy"
        return produced

    with ExitStack() as stack:
        labelled_patches(job, descriptor, source, stack)
        producer = stack.enter_context(patch.object(steps, "execute_produced", side_effect=execute))
        facade.dispatch(command, operation="generate")
    assert producer.call_count == 1 and facade._state == job._state == "idle"


def test_public_constructor_selects_only_concrete_original_command_namespace(
    tmp_path: Path,
) -> None:
    _, job, _, _, _, _ = setup(tmp_path)
    with patch.object(public, "_create_operator_workflow", return_value=job) as composed:
        facade = public.create_scientific_ablation_operator_command_workflow_v1(
            tmp_path / "operator",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=job._seal.store.plan,
            base=tmp_path,
            root=tmp_path / "output",
        )
    assert composed.call_count == 1 and composed.call_args.kwargs["command_namespace"] is True
    assert facade._seal.job is job and not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "index, operation",
    (
        (0, "generate"),
        (1, "score"),
        (2, "diagnose"),
        (3, "diagnose"),
        (4, "diagnose"),
        (5, "publish"),
    ),
)
def test_all_six_phase_kinds_route_only_to_explicit_operation_before_source(
    tmp_path: Path, index: int, operation: Any
) -> None:
    facade, job, _, _, _, _ = setup(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(job._seal.store.plan)[index]
    kernel = phase.kind in ("capture", "score")
    command = build(
        Intent(
            namespace=facade.definition,
            phase=phase,
            completed_phase_sha256=("f" * 64,) * index,
            expected_chunk=chunk() if kernel else None,
            rows_per_call=1 if kernel else None,
        )
    )
    with (
        patch.object(
            public,
            "_revalidate_operator_binding",
            side_effect=ValueError("Labelled stop before IO"),
        ) as source,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.dispatch(command, operation=operation)
    assert source.call_count == 1 and not (tmp_path / "output").exists()
