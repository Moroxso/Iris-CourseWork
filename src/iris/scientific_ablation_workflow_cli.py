"""Explicit CLI bridge to the ORIGINAL source-bound command workflow.

Saved plans/intents are DATA only. No automatic head, recovery, stage adoption,
native retry, alternate factory, or scientific acceptance is inferred by CLI.
"""

import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Annotated

import typer
from pydantic import BaseModel

from iris.application import scientific_ablation_operator_export as export
from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_execution import _local
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import _chain, _read_file
from iris.application.scientific_ablation_workflow import ScientificAblationWorkflowAdvanceResultV1
from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_ablation_workflow_planning import (
    build_scientific_ablation_workflow_kernel_request_v1,
)
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    ScientificAblationWorkflowCommandIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    IrisScientificAblationWorkflowKernelRequestV1,
    ScientificAblationWorkflowKernelRequestPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.scientific_ablation_cli import _emit, _guard, _persist
from iris.utils.hashing import canonical_json_file_bytes

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
_MAX_METADATA = 512 * 1024
_OPERATIONS: dict[str, public.Operation] = {
    "capture": "generate",
    "score": "score",
    "diagnostics": "diagnose",
    "workspace": "diagnose",
    "message": "diagnose",
    "publish": "publish",
}


class RecoveryOperation(StrEnum):
    GENERATE = "generate"
    SCORE = "score"


@dataclass(frozen=True, slots=True)
class _Options:
    operator_path: Path
    export_request: Path
    export_root: Path
    plan: Path
    base: Path
    root: Path
    completed_phase_sha256: tuple[str, ...]


@app.callback()
def context(
    ctx: typer.Context,
    operator_path: Annotated[Path, typer.Option("--operator")],
    export_request: Annotated[Path, typer.Option("--export-request")],
    export_root: Annotated[Path, typer.Option("--export-root")],
    plan: Annotated[
        Path, typer.Option("--plan", help="Saved plan; OUTPUT target for plan-request.")
    ],
    base: Annotated[Path, typer.Option("--base")],
    root: Annotated[Path, typer.Option("--root")],
    completed_phase_sha256: Annotated[
        list[str] | None,
        typer.Option("--completed-phase-sha256", help="Repeat exact independent prefix in order."),
    ] = None,
) -> None:
    """Use explicit inputs; help and argument parsing perform no IO or native work."""
    ctx.obj = _Options(
        operator_path,
        export_request,
        export_root,
        plan,
        base,
        root,
        tuple(completed_phase_sha256 or ()),
    )


def _absolute(path: Path) -> Path:
    windows = PureWindowsPath(str(path))
    if (
        ".." in path.parts
        or (windows.drive and not windows.root)
        or any(":" in part for part in path.parts if part != path.anchor)
    ):
        raise ValueError("Explicit local path without aliases or alternate streams required")
    return _local(path.absolute())


def _options(ctx: typer.Context) -> _Options:
    value = ctx.obj
    if type(value) is not _Options or len(value.completed_phase_sha256) > 134:
        raise ValueError("Exact explicit workflow input context required")
    for digest in value.completed_phase_sha256:
        _digest(digest)
    return _Options(
        operator_path=_absolute(value.operator_path),
        export_request=_absolute(value.export_request),
        export_root=_absolute(value.export_root),
        plan=_absolute(value.plan),
        base=_absolute(value.base),
        root=_absolute(value.root),
        completed_phase_sha256=value.completed_phase_sha256,
    )


def _read_model[T: BaseModel](path: Path, model: type[T]) -> T:
    path = _absolute(path)
    _chain(path.parent)
    raw = _read_file(path, _MAX_METADATA)
    value = model.model_validate_json(raw)
    if canonical_json_file_bytes(value.model_dump(mode="json")) != raw:
        raise ValueError("Exact canonical bounded saved workflow DATA required")
    return value


def _output_path(options: _Options, path: Path) -> Path:
    path = _absolute(path)
    if any(
        path.is_relative_to(protected)
        for protected in (options.operator_path, options.export_root, options.root)
    ):
        raise ValueError("Saved intent must stay outside source/export/workflow namespaces")
    return path


def _persist_request(options: _Options, path: Path, value: BaseModel) -> None:
    path = _output_path(options, path)
    _persist(
        path,
        canonical_json_file_bytes(value.model_dump(mode="json")),
        operator_path=options.operator_path,
    )


def _plan(options: _Options) -> IrisScientificAblationWorkflowPlanV1:
    return _read_model(options.plan, IrisScientificAblationWorkflowPlanV1)


def _compose(
    options: _Options, plan: IrisScientificAblationWorkflowPlanV1
) -> public.IrisScientificAblationOperatorCommandWorkflowV1:
    request = export.read_scientific_ablation_export_request_v1(options.export_request)
    return public.create_scientific_ablation_operator_command_workflow_v1(
        options.operator_path,
        request=request,
        export_root=options.export_root,
        plan=plan,
        base=options.base,
        root=options.root,
        completed_phase_sha256=options.completed_phase_sha256,
    )


def _phase(
    plan: IrisScientificAblationWorkflowPlanV1, name: str
) -> ScientificAblationWorkflowPhaseV1:
    phases = [phase for phase in scientific_ablation_workflow_phases_v1(plan) if phase.name == name]
    if len(phases) != 1:
        raise ValueError("One exact explicitly named source-plan phase required")
    return phases[0]


def _operation(
    command: IrisScientificAblationWorkflowCommandIntentV1, operation: public.Operation
) -> None:
    if _OPERATIONS[command.payload.phase.kind] != operation:
        raise ValueError("Saved phase does not belong to the explicit CLI operation")


def _python_opt_in(
    plan: IrisScientificAblationWorkflowPlanV1,
    operation: public.Operation,
    allowed: bool,
    *,
    data_only: bool,
) -> None:
    # BOTH original APIs validate an independent ACK before their retained DATA
    # path. A wrong/missing ACK fails closed; it cannot fall through to native.
    if (
        operation == "score"
        and plan.payload.python_policy is not None
        and not allowed
        and not data_only
    ):
        raise ValueError(
            "Python scoring requires explicit --allow-isolated-python and saved policy"
        )


@app.command("plan-request")
def plan_request(ctx: typer.Context, payload: Annotated[Path, typer.Option("--payload")]) -> None:
    """Save pure plan DATA to --plan; never bind an owner or execute a phase."""

    def action() -> None:
        options = _options(ctx)
        intent = _read_model(payload, ScientificAblationWorkflowIntentV1)
        plan = build_scientific_ablation_workflow_plan_v1(intent)
        _persist_request(options, options.plan, plan)
        _emit(plan.model_dump(mode="json"))

    _guard(action)


@app.command("command-request")
def command_request(
    ctx: typer.Context,
    phase_name: Annotated[str, typer.Option("--phase")],
    output: Annotated[Path, typer.Option("--out")],
    expected_chunk: Annotated[Path | None, typer.Option("--expected-chunk")] = None,
    rows_per_call: Annotated[int | None, typer.Option("--rows-per-call", min=1, max=5000)] = None,
) -> None:
    """Save exact BEFORE prefix/interval DATA; never discover a newer chunk/head."""

    def action() -> None:
        options = _options(ctx)
        plan = _plan(options)
        phase = _phase(plan, phase_name)
        chunk = (
            None
            if expected_chunk is None
            else _read_model(expected_chunk, ScientificAblationKernelChunkExpectationV1)
        )
        if (
            len(options.completed_phase_sha256) != phase.index
            or ((phase.kind in {"capture", "score"}) != (chunk is not None))
            or (chunk is not None and rows_per_call != chunk.rows_per_call)
            or (chunk is None and rows_per_call is not None)
        ):
            raise ValueError("Exact independent phase prefix and original interval/budget required")
        owner = _compose(options, plan)
        command = build_scientific_ablation_workflow_command_intent_v1(
            ScientificAblationWorkflowCommandIntentPayloadV1(
                namespace=owner.definition,
                phase=phase,
                completed_phase_sha256=options.completed_phase_sha256,
                expected_chunk=chunk,
                rows_per_call=rows_per_call,
            )
        )
        _persist_request(options, output, command)
        _emit(command.model_dump(mode="json"))

    _guard(action)


def _dispatch(
    ctx: typer.Context,
    saved: Path,
    operation: public.Operation,
    ack: str | None,
    allow_python: bool = False,
) -> None:
    def action() -> None:
        command = _read_model(saved, IrisScientificAblationWorkflowCommandIntentV1)
        _operation(command, operation)
        if ack is not None:
            _digest(ack)
        options = _options(ctx)
        plan = _plan(options)
        _python_opt_in(plan, operation, allow_python, data_only=ack is not None)
        owner = _compose(options, plan)
        reply = owner.dispatch(command, operation=operation, expected_operator_ack_sha256=ack)
        _emit(
            dict(
                kind="iris-scientific-ablation-workflow-command-summary-v1",
                operation=operation,
                request_sha256=reply.request_sha256,
                delivery=reply.delivery,
                operator_ack=reply.operator_ack.model_dump(mode="json"),
                progress=asdict(reply.original_result.progress),
                kernel_progress=_kernel_progress(reply.original_result),
                scientific_acceptance=False,
            )
        )

    _guard(action)


def _kernel_progress(result: ScientificAblationWorkflowAdvanceResultV1) -> dict[str, object] | None:
    """Read ORIGINAL returned snapshot, NEVER inspect a newer moving kernel head."""
    raw = result.kernel_progress_json
    if raw is None:
        return None
    value = IrisScientificAblationWorkflowKernelProgressV1.model_validate_json(raw)
    data = value.model_dump(mode="json")
    if canonical_json_file_bytes(data) != raw:
        raise ValueError("Exact original returned kernel progress DATA required")
    return data


@app.command("kernel-request")
def kernel_request(
    ctx: typer.Context,
    phase_name: Annotated[str, typer.Option("--phase")],
    rows_per_call: Annotated[int, typer.Option("--rows-per-call", min=1, max=5000)],
    output: Annotated[Path, typer.Option("--out")],
    initial: Annotated[bool, typer.Option("--initial")] = False,
    expected_kernel_plan_sha256: Annotated[
        str | None, typer.Option("--expected-kernel-plan-sha256")
    ] = None,
    expected_head: Annotated[
        str | None, typer.Option("--expected-acknowledgement-head-sha256")
    ] = None,
    expected_rows: Annotated[
        int | None, typer.Option("--expected-acknowledged-rows", min=0, max=5000)
    ] = None,
) -> None:
    """Save initial OR independently anchored retained planning DATA before any owner load."""

    def action() -> None:
        options = _options(ctx)
        plan = _plan(options)
        phase = _phase(plan, phase_name)
        if phase.kind not in {"capture", "score"} or phase.index != len(
            options.completed_phase_sha256
        ):
            raise ValueError("Exact independently anchored due capture/score phase required")
        if initial:
            if (
                expected_kernel_plan_sha256 is not None
                or expected_head is not None
                or expected_rows not in {None, 0}
            ):
                raise ValueError("Initial request cannot contain guessed retained anchors")
            rows = 0
        else:
            if (
                expected_kernel_plan_sha256 is None
                or expected_head is None
                or expected_rows is None
            ):
                raise ValueError("Retained request needs BOTH independent anchors and exact rows")
            _digest(expected_kernel_plan_sha256)
            _digest(expected_head)
            rows = expected_rows
        request = build_scientific_ablation_workflow_kernel_request_v1(
            ScientificAblationWorkflowKernelRequestPayloadV1(
                namespace=_compose(options, plan).definition,
                phase=phase,
                completed_phase_sha256=options.completed_phase_sha256,
                route="initial" if initial else "retained",
                rows_per_call=rows_per_call,
                expected_kernel_plan_sha256=expected_kernel_plan_sha256,
                expected_acknowledgement_head_sha256=expected_head,
                expected_acknowledged_rows=rows,
            )
        )
        _persist_request(options, output, request)
        _emit(request.model_dump(mode="json"))

    _guard(action)


@app.command("kernel-plan")
def kernel_plan(
    ctx: typer.Context,
    request: Annotated[Path, typer.Option("--request")],
    output: Annotated[Path, typer.Option("--out")],
) -> None:
    """Plan one saved request; INITIAL capture may reconstruct its original model owner.

    No rows/inference/phase attempt or native return is executed or reconstructed.
    """

    def action() -> None:
        wanted = _read_model(request, IrisScientificAblationWorkflowKernelRequestV1)
        options = _options(ctx)
        target = _output_path(options, output)
        if os.path.lexists(target):
            # Reject partial/foreign/hardlinked output before any owner load.
            # Even a valid DTO is not adopted as a planning/ownership receipt.
            _read_model(target, ScientificAblationKernelChunkExpectationV1)
        chunk = _compose(options, _plan(options)).plan_kernel(wanted)
        _persist_request(options, target, chunk)
        _emit(chunk.model_dump(mode="json"))

    _guard(action)


@app.command("generate")
def generate(
    ctx: typer.Context,
    command: Annotated[Path, typer.Option("--command")],
    expected_operator_ack_sha256: Annotated[
        str | None, typer.Option("--expected-operator-ack-sha256")
    ] = None,
) -> None:
    """Dispatch exactly one saved capture command; never execute generated code."""
    _dispatch(ctx, command, "generate", expected_operator_ack_sha256)


@app.command("score")
def score(
    ctx: typer.Context,
    command: Annotated[Path, typer.Option("--command")],
    expected_operator_ack_sha256: Annotated[
        str | None, typer.Option("--expected-operator-ack-sha256")
    ] = None,
    allow_isolated_python: Annotated[bool, typer.Option("--allow-isolated-python")] = False,
) -> None:
    """Score one saved interval, using ONLY the original isolated Python policy."""
    _dispatch(ctx, command, "score", expected_operator_ack_sha256, allow_isolated_python)


@app.command("diagnose")
def diagnose(
    ctx: typer.Context,
    command: Annotated[Path, typer.Option("--command")],
    expected_operator_ack_sha256: Annotated[
        str | None, typer.Option("--expected-operator-ack-sha256")
    ] = None,
) -> None:
    """Run exactly the saved diagnostics/workspace/message phase, not a new measurement loop."""
    _dispatch(ctx, command, "diagnose", expected_operator_ack_sha256)


@app.command("publish")
def publish(
    ctx: typer.Context,
    command: Annotated[Path, typer.Option("--command")],
    expected_operator_ack_sha256: Annotated[
        str | None, typer.Option("--expected-operator-ack-sha256")
    ] = None,
) -> None:
    """Publish the exact saved report phase; acceptance is never inferred."""
    _dispatch(ctx, command, "publish", expected_operator_ack_sha256)


@app.command("recover")
def recover(
    ctx: typer.Context,
    command: Annotated[Path, typer.Option("--command")],
    operation: Annotated[RecoveryOperation, typer.Option("--operation")],
    expected_operator_ack_sha256: Annotated[
        str | None, typer.Option("--expected-operator-ack-sha256")
    ] = None,
    allow_isolated_python: Annotated[bool, typer.Option("--allow-isolated-python")] = False,
) -> None:
    """Explicit ORIGINAL-stop kernel recovery; never reconstruct a lost native return."""

    def action() -> None:
        intent = _read_model(command, IrisScientificAblationWorkflowCommandIntentV1)
        requested: public.Operation = (
            "generate" if operation is RecoveryOperation.GENERATE else "score"
        )
        _operation(intent, requested)
        if intent.payload.expected_chunk is None:
            raise ValueError("Only the original capture/score interval is resumable")
        if expected_operator_ack_sha256 is not None:
            _digest(expected_operator_ack_sha256)
        options = _options(ctx)
        plan = _plan(options)
        _python_opt_in(
            plan,
            requested,
            allow_isolated_python,
            data_only=expected_operator_ack_sha256 is not None,
        )
        reply = _compose(options, plan).recover(
            intent, operation=requested, expected_operator_ack_sha256=expected_operator_ack_sha256
        )
        result = reply.result
        _emit(
            dict(
                kind="iris-scientific-ablation-workflow-recovery-command-v1",
                operation=requested,
                original_request_sha256=intent.sha256,
                recovered_result_sha256=result.sha256,
                operator_ack=reply.operator_ack.model_dump(mode="json"),
                kernel_progress=result.payload.kernel_progress.model_dump(mode="json"),
                original_return=result.payload.original_return,
                scientific_acceptance=False,
            )
        )

    _guard(action)


@app.command("status")
def status(ctx: typer.Context) -> None:
    """Read source/namespace/exact explicit phase prefix; no allocation or native phase."""

    def action() -> None:
        options = _options(ctx)
        owner = _compose(options, _plan(options))
        _emit(
            dict(
                kind="iris-scientific-ablation-workflow-inspection-v1",
                definition=owner.definition.model_dump(mode="json"),
                progress=asdict(owner.progress),
                read_only=True,
                scientific_acceptance=False,
            )
        )

    _guard(action)


@app.command("inspect-kernel")
def inspect_kernel(
    ctx: typer.Context,
    phase_name: Annotated[str, typer.Option("--phase")],
    kernel_plan_sha256: Annotated[str, typer.Option("--kernel-plan-sha256")],
    expected_acknowledgement_head_sha256: Annotated[
        str | None, typer.Option("--expected-acknowledgement-head-sha256")
    ] = None,
) -> None:
    """Read only the explicitly selected kernel/ACK, never plan or resume a new chunk."""

    def action() -> None:
        _digest(kernel_plan_sha256)
        if expected_acknowledgement_head_sha256 is not None:
            _digest(expected_acknowledgement_head_sha256)
        options = _options(ctx)
        plan = _plan(options)
        phase = _phase(plan, phase_name)
        kernel = _compose(options, plan).inspect_kernel(
            expected_phase=phase,
            expected_kernel_plan_sha256=kernel_plan_sha256,
            expected_acknowledgement_head_sha256=expected_acknowledgement_head_sha256,
        )
        _emit(
            dict(
                kind="iris-scientific-ablation-workflow-kernel-inspection-v1",
                kernel_progress=kernel.model_dump(mode="json"),
                read_only=True,
                scientific_acceptance=False,
            )
        )

    _guard(action)


__all__ = ["app"]
