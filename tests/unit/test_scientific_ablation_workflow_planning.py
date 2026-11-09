"""LABELLED planner/controller/CLI DATA probes, NOT native model or source proof.

Original workflow/store/claim objects and intent files are real. Source, owner,
prefix and kernel-reader doubles below are explicitly LABELLED; they do not
authenticate physical source, mint ownership, or prove native/installed behavior.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import PropertyMock, patch

import pytest
from pydantic import ValidationError

from iris import scientific_ablation_workflow_cli as cli
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture_module
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_evaluation_ledger as score_module
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow_command as public
from iris.application import scientific_ablation_workflow_io as storage
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application import scientific_ablation_workflow_planning as planning
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    IrisScientificAblationWorkflowKernelRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    ScientificAblationWorkflowKernelRequestPayloadV1 as Payload,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_command_workflow_namespace import definition
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_cli import (
    _labelled_owner,
    _wire,
    _write,
    runner,
)
from tests.unit.test_scientific_ablation_workflow_handoff import labelled_job


def _plan(tmp_path: Path, maximum: int = 16 * 1024**2) -> Any:
    value = intent(
        tmp_path,
        controls=(
            WorkspaceInterventionPlanV1(
                interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
            ),
        ),
    )
    value = ScientificAblationWorkflowIntentV1.model_validate(
        value.model_dump(mode="python")
        | dict(resources=value.resources.model_dump() | dict(maximum_storage_bytes=maximum))
    )
    return storage.build_scientific_ablation_workflow_plan_v1(value)


def _payload(tmp_path: Path, index: int = 0, route: str = "initial") -> Payload:
    plan = _plan(tmp_path)
    return Payload.model_validate(
        dict(
            namespace=definition(plan),
            phase=scientific_ablation_workflow_phases_v1(plan)[index],
            completed_phase_sha256=("c" * 64,) * index,
            route=route,
            rows_per_call=3,
            expected_kernel_plan_sha256=None if route == "initial" else "a" * 64,
            expected_acknowledgement_head_sha256=None if route == "initial" else "b" * 64,
            expected_acknowledged_rows=0 if route == "initial" else 1,
        )
    )


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("route", ("initial", "retained"))
def test_all4_pure_request_routes_are_DATA_not_owners(
    tmp_path: Path, index: int, route: str
) -> None:
    payload = _payload(tmp_path, index, route)
    result = planning.build_scientific_ablation_workflow_kernel_request_v1(payload)
    before = result.model_dump_json()
    vars(payload)["rows_per_call"] = 5
    assert result.model_dump_json() == before
    assert (
        result.payload.authority == "planning_request_data_only" and list(tmp_path.iterdir()) == []
    )


@pytest.mark.parametrize(
    "changes",
    (
        dict(rows_per_call=True),
        dict(rows_per_call=0),
        dict(rows_per_call=5001),
        dict(expected_acknowledged_rows=False),
        dict(expected_acknowledged_rows=1),
        dict(expected_kernel_plan_sha256="a" * 64),
        dict(expected_acknowledgement_head_sha256="b" * 64),
        dict(completed_phase_sha256=("a" * 64,)),
        dict(route="latest"),
        dict(authority="native_owner"),
        dict(callback="not allowed"),
    ),
)
def test_bad_initial_request_fields_fail_without_IO(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        Payload.model_validate(_payload(tmp_path).model_dump(mode="python") | changes)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field", ("expected_kernel_plan_sha256", "expected_acknowledgement_head_sha256")
)
def test_retained_request_cannot_guess_missing_anchor(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValidationError):
        Payload.model_validate(
            _payload(tmp_path, route="retained").model_dump(mode="python") | {field: None}
        )


@pytest.mark.parametrize("index", (4, 5, 6, 7))
def test_nonkernel_phases_are_not_plannable(tmp_path: Path, index: int) -> None:
    with pytest.raises(ValidationError):
        _payload(tmp_path, index)


@pytest.mark.parametrize("fault", ("hash", "hidden", "private", "subclass"))
def test_poisoned_request_refused_before_original_owner_entry(tmp_path: Path, fault: str) -> None:
    request: Any = planning.build_scientific_ablation_workflow_kernel_request_v1(_payload(tmp_path))
    if fault == "hash":
        vars(request)["sha256"] = "f" * 64
    elif fault == "hidden":
        vars(request.payload)["native_owner"] = object()
    elif fault == "private":
        object.__setattr__(request, "__pydantic_private__", {"native_owner": object()})
    else:

        class LabelledSubclass(IrisScientificAblationWorkflowKernelRequestV1):
            pass

        request = LabelledSubclass.model_validate(request.model_dump(mode="python"))
    job: Any = object()
    with patch.object(planning, "_entry") as entered, pytest.raises((ValueError, ValidationError)):
        planning.plan_scientific_ablation_workflow_kernel_chunk_v1(job, request)
    assert entered.call_count == 0 and list(tmp_path.iterdir()) == []


def _controller(
    tmp_path: Path, index: int, route: str, maximum: int = 16 * 1024**2
) -> tuple[Any, ...]:
    job, _, _, source, _ = labelled_job(tmp_path)
    plan = _plan(tmp_path, maximum)
    namespace = operator._command_definition(plan, _request(), source)
    store = storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "planning-output", plan, namespace
    )
    job._seal = replace(
        job._seal,
        store=store,
        path=store.path,
        plan=steps.encoded(plan),
        storage_definition=store.namespace_definition_json,
        anchors=("c" * 64,) * index,
    )
    facade = public._wrap(job)
    payload = Payload.model_validate(
        _payload(tmp_path, index, route).model_dump(mode="python") | dict(namespace=namespace)
    )
    request = planning.build_scientific_ablation_workflow_kernel_request_v1(payload)
    return facade, job, store, request


@contextmanager
def _doubles(job: Any, index: int, stack: ExitStack) -> Iterator[dict[str, Any]]:
    """NOT physical source/prefix/owner proof; real original claim/entry remain."""
    prefix = tuple(object() for _ in range(index))
    source = stack.enter_context(patch.object(planning, "_check_operator_source"))
    stack.enter_context(patch.object(planning, "_prefix", return_value=prefix))
    controlled = index == 2
    owner_class = (
        IrisScientificAblationControlledHeldoutV1 if controlled else IrisScientificAblationHeldoutV1
    )
    owner = object.__new__(owner_class)
    stack.enter_context(
        patch.object(
            owner_class,
            "manifest",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(sha256="d" * 64),
        )
    )
    factory = stack.enter_context(patch.object(steps, "_owner", return_value=owner))
    stack.enter_context(patch.object(steps, "_ready"))
    planned = SimpleNamespace(sha256="a" * 64, payload=SimpleNamespace(rows=(None,) * 4))
    capture = stack.enter_context(
        patch.object(
            capture_module, "build_scientific_ablation_capture_plan_v1", return_value=planned
        )
    )
    control = stack.enter_context(
        patch.object(
            controlled_capture,
            "build_scientific_ablation_controlled_capture_plan_v1",
            return_value=planned,
        )
    )
    stack.enter_context(patch.object(steps, "_score_args", return_value={}))
    score = stack.enter_context(
        patch.object(
            score_module, "build_scientific_ablation_evaluation_plan_v2", return_value=planned
        )
    )
    score_control = stack.enter_context(
        patch.object(planning, "build_controlled_score_plan", return_value=planned)
    )
    observed = ("a" * 64, "b" * 64, 1, 4, None, 0)
    reader = stack.enter_context(patch.object(kernels, "_capture", return_value=observed))
    score_reader = stack.enter_context(patch.object(kernels, "_score", return_value=observed))
    produced = stack.enter_context(
        patch.object(steps, "execute_produced", side_effect=AssertionError("No rows"))
    )
    begin = stack.enter_context(
        patch.object(
            storage.ScientificAblationWorkflowStoreV1,
            "begin",
            side_effect=AssertionError("No attempt"),
        )
    )
    yield dict(
        source=source,
        factory=factory,
        capture=capture,
        control=control,
        score=score,
        score_control=score_control,
        reader=reader,
        score_reader=score_reader,
        produced=produced,
        begin=begin,
    )


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("route", ("initial", "retained"))
def test_real_claim_LABELLED_plans_all4_without_attempt_or_rows(
    tmp_path: Path, index: int, route: str
) -> None:
    facade, job, store, request = _controller(tmp_path, index, route)
    original_seal = job._seal
    with ExitStack() as stack, _doubles(job, index, stack) as calls:
        result = facade.plan_kernel(request)
    assert result.acknowledged_rows == (0 if route == "initial" else 1)
    assert result.stop_row_index == (3 if route == "initial" else 4)
    assert result.rows_per_call == 3 and result.total_rows == 4
    assert calls["factory"].call_count == int(route == "initial" and index % 2 == 0)
    assert calls["reader"].call_count == (2 if route == "retained" and index % 2 == 0 else 0)
    assert calls["score_reader"].call_count == (2 if route == "retained" and index % 2 else 0)
    assert calls["produced"].call_count == calls["begin"].call_count == 0
    assert not (store.path / "kernels").exists()
    assert not (store.path / f"phase-{index:03d}.attempt.json").exists()
    assert job._seal is original_seal and job._state == facade._state == "idle"


@pytest.mark.parametrize("kind", ("kernel", "attempt", "phase"))
def test_initial_cannot_adopt_or_delete_existing_paths(tmp_path: Path, kind: str) -> None:
    facade, job, store, request = _controller(tmp_path, 0, "initial")
    path = store.path / (
        "kernels/ordinary-capture/foreign"
        if kind == "kernel"
        else "phase-000.attempt.json"
        if kind == "attempt"
        else "phase-000/foreign"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserve unknown DATA")
    with (
        ExitStack() as stack,
        _doubles(job, 0, stack) as calls,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.plan_kernel(request)
    assert calls["factory"].call_count == 0 and path.read_bytes() == b"preserve unknown DATA"


@pytest.mark.parametrize("fault", ("source", "namespace", "prefix", "pending"))
def test_source_namespace_prefix_or_pending_result_refused_before_claim(
    tmp_path: Path, fault: str
) -> None:
    facade, job, store, request = _controller(tmp_path, 0, "initial")
    if fault == "namespace":
        request = planning.build_scientific_ablation_workflow_kernel_request_v1(
            Payload.model_validate(
                request.payload.model_dump(mode="python")
                | dict(namespace=definition(store.plan, operator_sha256="f" * 64))
            )
        )
    elif fault == "prefix":
        job._seal = replace(job._seal, anchors=("f" * 64,))
    elif fault == "pending":
        job._seal = replace(job._seal, ready=object())
    with ExitStack() as stack, _doubles(job, 0, stack) as calls:
        if fault == "source":
            calls["source"].side_effect = ValueError("LABELLED changed source")
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            facade.plan_kernel(request)
    assert calls["factory"].call_count == 0 and not store.path.exists()
    assert job._state == facade._state == "idle"


def test_whole_quota_rejects_before_reconstructing_any_owner(tmp_path: Path) -> None:
    facade, job, _, request = _controller(tmp_path, 0, "initial", maximum=2 * 1024**2)
    with (
        ExitStack() as stack,
        _doubles(job, 0, stack) as calls,
        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
    ):
        facade.plan_kernel(request)
    assert calls["factory"].call_count == 0 and job._state == facade._state == "idle"


@pytest.mark.parametrize(
    "fault", ("plan", "head", "rows", "stage", "stage_float", "row_bool", "changed")
)
def test_retained_refuses_stale_or_unknown_DATA_without_factory(tmp_path: Path, fault: str) -> None:
    facade, job, _, request = _controller(tmp_path, 0, "retained")
    observed: list[Any] = ["a" * 64, "b" * 64, 1, 4, None, 0]
    if fault in {"plan", "head"}:
        observed[0 if fault == "plan" else 1] = "f" * 64
    elif fault == "rows":
        observed[2] = 2
    elif fault == "stage":
        observed[5] = 1
    elif fault == "stage_float":
        observed[5] = 0.0
    elif fault == "row_bool":
        observed[2] = True
    with ExitStack() as stack, _doubles(job, 0, stack) as calls:
        if fault == "changed":
            calls["reader"].side_effect = [
                ("a" * 64, "b" * 64, 1, 4, None, 0),
                ("a" * 64, "f" * 64, 2, 4, None, 0),
            ]
        else:
            calls["reader"].return_value = tuple(observed)
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            facade.plan_kernel(request)
    assert calls["factory"].call_count == 0 and job._state == facade._state == "idle"


@pytest.mark.parametrize("grouped", (False, True))
def test_factory_primary_signal_identity_survives_original_claim(
    tmp_path: Path, grouped: bool
) -> None:
    facade, job, _, request = _controller(tmp_path, 0, "initial")
    primary = (
        BaseExceptionGroup("LABELLED", [KeyboardInterrupt(), ValueError()])
        if grouped
        else KeyboardInterrupt()
    )
    with ExitStack() as stack, _doubles(job, 0, stack) as calls:
        calls["factory"].side_effect = primary
        with pytest.raises(BaseException) as caught:
            facade.plan_kernel(request)
    assert caught.value is primary and job._state == facade._state == "idle"


@pytest.mark.parametrize(
    "kind,operation",
    (
        ("capture", "generate"),
        ("score", "score"),
        ("control-00-capture", "generate"),
        ("control-00-score", "score"),
    ),
)
@pytest.mark.parametrize("replay", (False, True))
def test_CLI_emits_ORIGINAL_partial_snapshot_without_reading_new_head(
    tmp_path: Path, kind: str, operation: str, replay: bool
) -> None:
    wire = _wire(tmp_path, kind)
    owner = _labelled_owner(wire, "saved_original_data" if replay else "original_reference")
    phase = wire.command.payload.phase
    kernel = IrisScientificAblationWorkflowKernelProgressV1(
        workflow_plan_sha256=wire.plan.sha256,
        phase=phase,
        completed_phase_sha256=wire.command.payload.completed_phase_sha256,
        phase_is_acknowledged=False,
        kernel_plan_sha256="a" * 64,
        acknowledgement_head_sha256="b" * 64,
        acknowledged_rows=1,
        total_rows=4,
        kernel_complete_sha256=None,
        uncommitted_stage_count=0,
    )
    owner.dispatch.return_value.original_result.kernel_progress_json = canonical_json_file_bytes(
        kernel.model_dump(mode="json")
    )
    args = [*wire.args, operation, "--command", str(wire.saved)]
    if replay:
        args.extend(("--expected-operator-ack-sha256", "c" * 64))
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    import json

    assert json.loads(result.stdout)["kernel_progress"] == kernel.model_dump(mode="json")
    owner.inspect_kernel.assert_not_called()
    owner.recover.assert_not_called()


@pytest.mark.parametrize("kind", ("capture", "score", "control-00-capture", "control-00-score"))
@pytest.mark.parametrize("initial", (False, True))
def test_CLI_kernel_request_saves_before_any_planning(
    tmp_path: Path, kind: str, initial: bool
) -> None:
    wire = _wire(tmp_path, kind)
    owner = _labelled_owner(wire)
    path = tmp_path / "kernel-request.json"
    args = [
        *wire.args,
        "kernel-request",
        "--phase",
        wire.command.payload.phase.name,
        "--rows-per-call",
        "3",
        "--out",
        str(path),
    ]
    args += (
        ["--initial"]
        if initial
        else [
            "--expected-kernel-plan-sha256",
            "a" * 64,
            "--expected-acknowledgement-head-sha256",
            "b" * 64,
            "--expected-acknowledged-rows",
            "1",
        ]
    )
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    wanted = cli._read_model(path, IrisScientificAblationWorkflowKernelRequestV1)
    assert wanted.payload.route == ("initial" if initial else "retained")
    assert wanted.payload.namespace == owner.definition
    owner.plan_kernel.assert_not_called()
    owner.dispatch.assert_not_called()


@pytest.mark.parametrize("bad", ("missing", "initial_head", "nonkernel", "budget"))
def test_CLI_kernel_request_refuses_implicit_or_bad_planning(tmp_path: Path, bad: str) -> None:
    wire = _wire(tmp_path)
    path = tmp_path / "request.json"
    args = [
        *wire.args,
        "kernel-request",
        "--phase",
        "publish" if bad == "nonkernel" else "ordinary-capture",
        "--rows-per-call",
        "0" if bad == "budget" else "1",
        "--out",
        str(path),
    ]
    if bad == "initial_head":
        args.extend(("--initial", "--expected-kernel-plan-sha256", "a" * 64))
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 2 and compose.call_count == 0 and not path.exists()


def test_CLI_kernel_plan_invokes_one_saved_request_and_preserves_exact_chunk(
    tmp_path: Path,
) -> None:
    wire = _wire(tmp_path)
    owner = _labelled_owner(wire)
    wanted = planning.build_scientific_ablation_workflow_kernel_request_v1(_payload(tmp_path))
    request_path = tmp_path / "planning.json"
    _write(request_path, wanted)
    chunk = Chunk(
        kernel_plan_sha256="a" * 64,
        acknowledgement_head_sha256="b" * 64,
        acknowledged_rows=0,
        total_rows=4,
        rows_per_call=3,
        stop_row_index=3,
    )
    owner.plan_kernel.return_value = chunk
    output = tmp_path / "chunk.json"
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(
            cli.app,
            [*wire.args, "kernel-plan", "--request", str(request_path), "--out", str(output)],
        )
    assert result.exit_code == 0, result.output
    owner.plan_kernel.assert_called_once_with(wanted)
    owner.dispatch.assert_not_called()
    owner.recover.assert_not_called()
    assert output.read_bytes() == canonical_json_file_bytes(chunk.model_dump(mode="json"))


def test_CLI_partial_planning_output_is_preserved_before_owner_load(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    wanted = planning.build_scientific_ablation_workflow_kernel_request_v1(_payload(tmp_path))
    request_path = tmp_path / "planning.json"
    _write(request_path, wanted)
    output = tmp_path / "chunk.json"
    output.write_bytes(b"partial")
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(
            cli.app,
            [*wire.args, "kernel-plan", "--request", str(request_path), "--out", str(output)],
        )
    assert result.exit_code == 2 and compose.call_count == 0 and output.read_bytes() == b"partial"
