"""LABELLED CLI plumbing/real intent-file IO, NOT native or installed acceptance.

Factories/owners below are explicitly mocked. No generated Python, models,
native phase, sandbox, native replay or scientific result is executed here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import typer
from pydantic import BaseModel
from typer.main import get_command
from typer.testing import CliRunner

from iris import scientific_ablation_workflow_cli as cli
from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_workflow import ScientificAblationWorkflowProgressV1
from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1,
)
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    ScientificAblationWorkflowCommandIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_command_workflow_namespace import definition
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk

runner = CliRunner()


def _write(path: Path, value: BaseModel) -> None:
    path.write_bytes(canonical_json_file_bytes(value.model_dump(mode="json")))


@dataclass(frozen=True)
class Wire:
    plan: IrisScientificAblationWorkflowPlanV1
    command: IrisScientificAblationWorkflowCommandIntentV1
    saved: Path
    args: list[str]


def _wire(tmp_path: Path, kind: str = "capture", *, python: bool = False) -> Wire:
    """Constructible metadata only; real source owner is NEVER claimed by fixture."""
    payload = intent(
        tmp_path,
        controls=(
            WorkspaceInterventionPlanV1(
                interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
            ),
        ),
    )
    if python:
        payload = ScientificAblationWorkflowIntentV1.model_validate(
            payload.model_dump(mode="python")
            | dict(
                python_policy=PythonSandboxPolicyV2(
                    profile="iris_wsl_gvisor_python312_source_v2",
                    wall_time_ms=5000,
                    max_output_bytes=4096,
                )
            )
        )
    plan = build_scientific_ablation_workflow_plan_v1(payload)
    phase = next(
        p for p in scientific_ablation_workflow_phases_v1(plan) if p.name == kind or p.kind == kind
    )
    interval = chunk() if phase.kind in {"capture", "score"} else None
    command = build_scientific_ablation_workflow_command_intent_v1(
        ScientificAblationWorkflowCommandIntentPayloadV1(
            namespace=definition(plan),
            phase=phase,
            completed_phase_sha256=("b" * 64,) * phase.index,
            expected_chunk=interval,
            rows_per_call=None if interval is None else interval.rows_per_call,
        )
    )
    paths = {
        "--operator": tmp_path / "original-operator",
        "--export-request": tmp_path / "export-request.json",
        "--export-root": tmp_path / "original-exports",
        "--plan": tmp_path / "plan.json",
        "--base": tmp_path / "original-inputs",
        "--root": tmp_path / "original-workflows",
    }
    _write(paths["--plan"], plan)
    _write(paths["--export-request"], _request())
    saved = tmp_path / "command.json"
    _write(saved, command)
    args = [item for name, path in paths.items() for item in (name, str(path))]
    for digest in command.payload.completed_phase_sha256:
        args.extend(("--completed-phase-sha256", digest))
    return Wire(plan, command, saved, args)


def _labelled_owner(wire: Wire, delivery: str = "original_reference") -> Mock:
    """Mocked call routing ONLY; this is NOT a minted native owner/reply/ACK."""
    owner = Mock(spec=public.IrisScientificAblationOperatorCommandWorkflowV1)
    owner.definition = definition(wire.plan)
    owner.progress = ScientificAblationWorkflowProgressV1(
        wire.plan.sha256,
        wire.command.payload.completed_phase_sha256,
        wire.command.payload.phase.name,
        False,
        True,
    )
    ack = SimpleNamespace(model_dump=lambda **_: dict(LABELLED_unit_ACK_only=True))
    owner.dispatch.return_value = SimpleNamespace(
        request_sha256=wire.command.sha256,
        delivery=delivery,
        operator_ack=ack,
        original_result=SimpleNamespace(progress=owner.progress, kernel_progress_json=None),
    )
    kernel = SimpleNamespace(model_dump=lambda **_: dict(LABELLED_unit_kernel_only=True))
    owner.recover.return_value = SimpleNamespace(
        operator_ack=ack,
        result=SimpleNamespace(
            sha256="f" * 64,
            payload=SimpleNamespace(
                kernel_progress=kernel, original_return="unavailable_not_reconstructed"
            ),
        ),
    )
    owner.inspect_kernel.return_value = kernel
    return owner


@pytest.mark.parametrize(
    "command",
    (
        "",
        "plan-request",
        "command-request",
        "generate",
        "score",
        "diagnose",
        "publish",
        "recover",
        "status",
        "inspect-kernel",
    ),
)
def test_help_never_reads_files_or_composes_owner(tmp_path: Path, command: str) -> None:
    dummy = [
        item
        for key in ("--operator", "--export-request", "--export-root", "--plan", "--base", "--root")
        for item in (key, str(tmp_path / "not-created"))
    ]
    with patch.object(cli, "_compose") as compose, patch.object(cli, "_read_model") as read:
        result = runner.invoke(cli.app, ([*dummy, command] if command else []) + ["--help"])
    assert result.exit_code == 0, result.output
    assert compose.call_count == read.call_count == 0 and list(tmp_path.iterdir()) == []


def test_real_root_registers_workflow_without_changing_legacy_namespace() -> None:
    from iris.cli import app

    result = runner.invoke(app, ["scientific-ablation", "--help"])
    assert (
        result.exit_code == 0 and "workflow" in result.stdout and "phase-request" in result.stdout
    )
    baseline = runner.invoke(app, ["scientific", "--help"])
    assert baseline.exit_code == 0 and "phase-request" not in baseline.stdout


@pytest.mark.parametrize(
    "kind,operation",
    (*cli._OPERATIONS.items(), ("control-00-capture", "generate"), ("control-00-score", "score")),
)
@pytest.mark.parametrize("replay", (False, True))
def test_exact_saved_command_routes_once_without_recovery(
    tmp_path: Path, kind: str, operation: str, replay: bool
) -> None:
    wire = _wire(tmp_path, kind)
    owner = _labelled_owner(wire, "saved_original_data" if replay else "original_reference")
    ack = "a" * 64 if replay else None
    args = [*wire.args, operation, "--command", str(wire.saved)]
    if ack is not None:
        args.extend(("--expected-operator-ack-sha256", ack))
    with patch.object(
        public, "create_scientific_ablation_operator_command_workflow_v1", return_value=owner
    ) as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    owner.dispatch.assert_called_once_with(
        wire.command, operation=operation, expected_operator_ack_sha256=ack
    )
    owner.recover.assert_not_called()
    assert (
        compose.call_args.kwargs["completed_phase_sha256"]
        == wire.command.payload.completed_phase_sha256
    )
    assert compose.call_args.kwargs["plan"] == wire.plan
    assert compose.call_args.kwargs["request"] == _request()
    value = json.loads(result.stdout)
    assert (
        value["request_sha256"] == wire.command.sha256 and value["scientific_acceptance"] is False
    )
    assert value["delivery"] == ("saved_original_data" if replay else "original_reference")


@pytest.mark.parametrize(
    "kind,wrong",
    (
        ("capture", "score"),
        ("score", "generate"),
        ("diagnostics", "publish"),
        ("workspace", "generate"),
        ("message", "score"),
        ("publish", "diagnose"),
        ("control-00-capture", "score"),
        ("control-00-score", "generate"),
    ),
)
def test_wrong_closed_operation_refused_before_source_factory(
    tmp_path: Path, kind: str, wrong: str
) -> None:
    wire = _wire(tmp_path, kind)
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, [*wire.args, wrong, "--command", str(wire.saved)])
    assert result.exit_code == 2 and compose.call_count == 0


@pytest.mark.parametrize("command", ("generate", "score", "diagnose", "publish", "recover"))
def test_no_implicit_command_when_saved_intent_absent(tmp_path: Path, command: str) -> None:
    wire = _wire(tmp_path)
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, [*wire.args, command])
    assert result.exit_code == 2 and compose.call_count == 0


@pytest.mark.parametrize(
    "fault", ("duplicate", "unknown", "float", "no_lf", "whitespace", "oversize", "hash")
)
def test_bad_saved_wire_refused_without_composition(tmp_path: Path, fault: str) -> None:
    wire = _wire(tmp_path)
    raw = wire.saved.read_bytes()
    if fault == "duplicate":
        raw = b'{"sha256":"' + wire.command.sha256.encode() + b'",' + raw[1:]
    elif fault == "unknown":
        value = json.loads(raw)
        value["native_owner"] = "not an owner"
        raw = canonical_json_file_bytes(value)
    elif fault == "float":
        value = json.loads(raw)
        value["payload"]["rows_per_call"] = 1.0
        raw = canonical_json_file_bytes(value)
    elif fault == "no_lf":
        raw = raw.rstrip(b"\n")
    elif fault == "whitespace":
        raw = b" " + raw
    elif fault == "oversize":
        raw = b"x" * (cli._MAX_METADATA + 1)
    else:
        value = json.loads(raw)
        value["sha256"] = "0" * 64
        raw = canonical_json_file_bytes(value)
    wire.saved.write_bytes(raw)
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, [*wire.args, "generate", "--command", str(wire.saved)])
    assert result.exit_code == 2 and compose.call_count == 0 and wire.saved.read_bytes() == raw


@pytest.mark.parametrize("bad", ("latest", "A" * 64, "a" * 63))
def test_independent_prefix_never_guessed_or_normalized(tmp_path: Path, bad: str) -> None:
    wire = _wire(tmp_path)
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, [*wire.args, "--completed-phase-sha256", bad, "status"])
    assert result.exit_code == 2 and compose.call_count == 0


@pytest.mark.parametrize(
    "name", ("../foreign", "a/../b", "file:stream", "C:relative", "\\\\server\\share\\file")
)
def test_paths_reject_traversal_drive_relative_ADS_and_network(name: str) -> None:
    with pytest.raises(ValueError):
        cli._absolute(Path(name))


def test_dot_base_becomes_explicit_current_directory() -> None:
    assert cli._absolute(Path(".")) == Path.cwd()


def test_plan_request_is_exclusive_pure_data_and_exact_retry(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    source = tmp_path / "payload.json"
    _write(source, wire.plan.payload)
    args = [*wire.args, "plan-request", "--payload", str(source)]
    before = (tmp_path / "plan.json").stat()
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, args)
        repeated = runner.invoke(cli.app, args)
    assert result.exit_code == repeated.exit_code == 0 and compose.call_count == 0
    assert json.loads(result.stdout) == wire.plan.model_dump(mode="json")
    after = (tmp_path / "plan.json").stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_plan_request_preserves_incomplete_existing_target(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    source = tmp_path / "payload.json"
    _write(source, wire.plan.payload)
    (tmp_path / "plan.json").write_bytes(b"partial")
    result = runner.invoke(cli.app, [*wire.args, "plan-request", "--payload", str(source)])
    assert result.exit_code == 2 and (tmp_path / "plan.json").read_bytes() == b"partial"


@pytest.mark.parametrize("kind", (*cli._OPERATIONS, "control-00-capture", "control-00-score"))
def test_command_request_binds_exact_explicit_phase_and_interval(tmp_path: Path, kind: str) -> None:
    wire = _wire(tmp_path, kind)
    owner = _labelled_owner(wire)
    output = tmp_path / "new-command.json"
    args = [
        *wire.args,
        "command-request",
        "--phase",
        wire.command.payload.phase.name,
        "--out",
        str(output),
    ]
    interval = wire.command.payload.expected_chunk
    if interval is not None:
        chunk_path = tmp_path / "chunk.json"
        _write(chunk_path, interval)
        args.extend(
            ("--expected-chunk", str(chunk_path), "--rows-per-call", str(interval.rows_per_call))
        )
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == canonical_json_file_bytes(wire.command.model_dump(mode="json"))
    owner.dispatch.assert_not_called()
    owner.recover.assert_not_called()


@pytest.mark.parametrize(
    "fault", ("missing_chunk", "wrong_budget", "unknown_phase", "wrong_prefix")
)
def test_command_request_does_not_autoplan_missing_or_wrong_interval(
    tmp_path: Path, fault: str
) -> None:
    wire = _wire(tmp_path)
    output = tmp_path / "new-command.json"
    args = list(wire.args)
    if fault == "wrong_prefix":
        args.extend(("--completed-phase-sha256", "a" * 64))
    args.extend(
        (
            "command-request",
            "--phase",
            "latest" if fault == "unknown_phase" else "ordinary-capture",
            "--out",
            str(output),
        )
    )
    if fault != "missing_chunk":
        chunk_path = tmp_path / "chunk.json"
        _write(chunk_path, chunk())
        args.extend(
            (
                "--expected-chunk",
                str(chunk_path),
                "--rows-per-call",
                "2" if fault == "wrong_budget" else "1",
            )
        )
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 2 and compose.call_count == 0 and not output.exists()


@pytest.mark.parametrize(
    "root_name", ("original-operator", "original-exports", "original-workflows")
)
def test_intent_file_cannot_be_saved_inside_owned_namespaces(
    tmp_path: Path, root_name: str
) -> None:
    wire = _wire(tmp_path, "publish")
    owner = _labelled_owner(wire)
    output = tmp_path / root_name / "injected.json"
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(
            cli.app, [*wire.args, "command-request", "--phase", "publish", "--out", str(output)]
        )
    assert result.exit_code == 2 and not output.exists()
    owner.dispatch.assert_not_called()
    owner.recover.assert_not_called()


def test_dispatch_failure_does_not_retry_recover_or_print_private_error(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    owner = _labelled_owner(wire)
    owner.dispatch.side_effect = ValueError("private source detail")
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(cli.app, [*wire.args, "generate", "--command", str(wire.saved)])
    assert result.exit_code == 2 and "private source detail" not in result.output
    assert owner.dispatch.call_count == 1 and owner.recover.call_count == 0


@pytest.mark.parametrize(
    "kind,operation",
    (
        ("capture", "generate"),
        ("score", "score"),
        ("control-00-capture", "generate"),
        ("control-00-score", "score"),
    ),
)
def test_explicit_recovery_delivers_distinct_data_not_an_original_return(
    tmp_path: Path, kind: str, operation: str
) -> None:
    wire = _wire(tmp_path, kind)
    owner = _labelled_owner(wire)
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(
            cli.app,
            [
                *wire.args,
                "recover",
                "--command",
                str(wire.saved),
                "--operation",
                operation,
                "--expected-operator-ack-sha256",
                "c" * 64,
            ],
        )
    assert result.exit_code == 0, result.output
    owner.recover.assert_called_once_with(
        wire.command, operation=operation, expected_operator_ack_sha256="c" * 64
    )
    owner.dispatch.assert_not_called()
    value = json.loads(result.stdout)
    assert value["original_return"] == "unavailable_not_reconstructed"
    assert value["scientific_acceptance"] is False and "delivery" not in value


@pytest.mark.parametrize("operation", ("diagnose", "publish", "latest"))
def test_nonresumable_operation_rejected_by_recovery_grammar(
    tmp_path: Path, operation: str
) -> None:
    wire = _wire(tmp_path)
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(
            cli.app, [*wire.args, "recover", "--command", str(wire.saved), "--operation", operation]
        )
    assert result.exit_code == 2 and compose.call_count == 0


@pytest.mark.parametrize("recover", (False, True))
@pytest.mark.parametrize("opt_in", (False, True))
def test_python_score_requires_optin_and_keeps_original_isolated_policy(
    tmp_path: Path, recover: bool, opt_in: bool
) -> None:
    wire = _wire(tmp_path, "score", python=True)
    owner = _labelled_owner(wire)
    args = [*wire.args, "recover" if recover else "score", "--command", str(wire.saved)]
    if recover:
        args.extend(("--operation", "score"))
    if opt_in:
        args.append("--allow-isolated-python")
    with patch.object(cli, "_compose", return_value=owner) as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == (0 if opt_in else 2), result.output
    assert compose.call_count == int(opt_in)
    if opt_in:
        assert compose.call_args.args[1].payload.python_policy == wire.plan.payload.python_policy
    assert owner.recover.call_count == int(opt_in and recover)
    assert owner.dispatch.call_count == int(opt_in and not recover)


@pytest.mark.parametrize("operation", ("status", "inspect-kernel"))
def test_inspection_never_dispatches_or_recovers(tmp_path: Path, operation: str) -> None:
    wire = _wire(tmp_path)
    owner = _labelled_owner(wire)
    args = [*wire.args, operation]
    if operation == "inspect-kernel":
        args.extend(("--phase", "ordinary-capture", "--kernel-plan-sha256", "d" * 64))
    with patch.object(cli, "_compose", return_value=owner):
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["read_only"] is True
    owner.dispatch.assert_not_called()
    owner.recover.assert_not_called()


@pytest.mark.parametrize("recover", (False, True))
def test_independently_anchored_python_DATA_retry_needs_no_execution_optin(
    tmp_path: Path, recover: bool
) -> None:
    wire = _wire(tmp_path, "score", python=True)
    owner = _labelled_owner(wire, "saved_original_data")
    args = [
        *wire.args,
        "recover" if recover else "score",
        "--command",
        str(wire.saved),
        "--expected-operator-ack-sha256",
        "a" * 64,
    ]
    if recover:
        args.extend(("--operation", "score"))
    with patch.object(cli, "_compose", return_value=owner) as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert compose.call_args.args[1].payload.python_policy == wire.plan.payload.python_policy
    call = owner.recover if recover else owner.dispatch
    call.assert_called_once_with(
        wire.command, operation="score", expected_operator_ack_sha256="a" * 64
    )


@pytest.mark.parametrize("recover", (False, True))
@pytest.mark.parametrize("bad", ("latest", "A" * 64, "a" * 63))
def test_invalid_DATA_retry_anchor_fails_before_source_or_optin_bypass(
    tmp_path: Path, recover: bool, bad: str
) -> None:
    wire = _wire(tmp_path, "score", python=True)
    args = [
        *wire.args,
        "recover" if recover else "score",
        "--command",
        str(wire.saved),
        "--expected-operator-ack-sha256",
        bad,
    ]
    if recover:
        args.extend(("--operation", "score"))
    with patch.object(cli, "_compose") as compose:
        result = runner.invoke(cli.app, args)
    assert result.exit_code == 2 and compose.call_count == 0


@pytest.mark.parametrize("grouped", (False, True))
def test_primary_interrupt_is_preserved_without_implicit_recovery(
    tmp_path: Path, grouped: bool
) -> None:
    wire = _wire(tmp_path)
    owner = _labelled_owner(wire)
    interrupt = KeyboardInterrupt("LABELLED primary signal")
    primary = (
        BaseExceptionGroup("LABELLED secondary failure", [interrupt, ValueError("cleanup")])
        if grouped
        else interrupt
    )
    owner.dispatch.side_effect = primary
    options = cli._Options(
        tmp_path / "original-operator",
        tmp_path / "export-request.json",
        tmp_path / "original-exports",
        tmp_path / "plan.json",
        tmp_path / "original-inputs",
        tmp_path / "original-workflows",
        (),
    )
    ctx = typer.Context(get_command(cli.app), obj=options)
    with patch.object(cli, "_compose", return_value=owner), pytest.raises(BaseException) as caught:
        cli._dispatch(ctx, wire.saved, "generate", None)
    assert caught.value is primary
    assert owner.dispatch.call_count == 1 and owner.recover.call_count == 0
