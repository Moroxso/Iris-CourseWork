"""Command-line interface for Iris."""

import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from iris.application import arm_experiment_command as arm_experiment_command_service
from iris.application import arm_experiment_command_v3 as arm_experiment_command_v3_service
from iris.application import arm_request_command_v4 as arm_request_service
from iris.application import backbone as backbone_service
from iris.application import experiment_command as experiment_command_service
from iris.application import hardware as hardware_service
from iris.application import intervention_command as intervention_service
from iris.application import mlflow_projection as mlflow_projection_service
from iris.application import request_command as request_service
from iris.application import scientific_ablation_writer_command as scientific_writer_command_service
from iris.application import scientific_answer_command as scientific_answer_service
from iris.application import scientific_answer_command_v2 as scientific_answer_v2_service
from iris.application import scientific_collapse as scientific_collapse_service
from iris.application import scientific_collapse_command as scientific_collapse_command_service
from iris.application import scientific_command as scientific_command_service
from iris.application import scientific_generation as scientific_generation_service
from iris.application import scientific_message_causal as scientific_message_service
from iris.application import scientific_message_command as scientific_message_command_service
from iris.application import scientific_mlflow as scientific_mlflow_service
from iris.application import scientific_publication as scientific_publication_service
from iris.application import scientific_workspace as scientific_workspace_service
from iris.application import scientific_workspace_command as scientific_workspace_command_service
from iris.application import study_benchmark_io as study_benchmark_io_service
from iris.application import study_dataset_commands as study_dataset_service
from iris.application import study_evaluation_command as study_evaluation_service
from iris.application import study_generation_command as study_generation_service
from iris.application import study_input_command as study_input_service
from iris.application import study_pilot as study_pilot_service
from iris.application import verification as verification_service
from iris.application import verification_v2 as verification_v2_service
from iris.application import verification_v3 as verification_v3_service
from iris.application import verification_v4 as verification_v4_service
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import (
    ApplicationError,
    FeatureUnavailableError,
    IrisCodeEvaluationUnavailableError,
    IrisRequestCommandError,
)
from iris.application.experiment_runner import IrisExperimentExecutionResult
from iris.application.experiment_runner_v2 import IrisExperimentExecutionResultV2
from iris.application.experiment_runner_v3 import IrisExperimentExecutionResultV3
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import WriterSourceProtocol
from iris.artifacts.scientific_collapse_v1 import IrisScientificCollapseV1
from iris.artifacts.scientific_evaluation_v1 import IrisScientificEvaluationV1
from iris.artifacts.scientific_evaluation_v2 import IrisScientificEvaluationV2
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificDecodingPolicyV1,
)
from iris.artifacts.scientific_message_causal_v1 import IrisScientificMessageCausalV1
from iris.artifacts.scientific_workspace_v1 import IrisScientificWorkspaceV1
from iris.generation import TemperatureTopPSamplingConfig
from iris.scientific_ablation_cli import app as scientific_ablation_app
from iris.scientific_ablation_workflow_cli import app as scientific_ablation_workflow_app
from iris.training import IrisAdamWConfig

FEATURE_UNAVAILABLE_EXIT_CODE = 3

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
hardware_app = typer.Typer(no_args_is_help=True)
backbone_app = typer.Typer(no_args_is_help=True)
request_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
verify_app = typer.Typer(no_args_is_help=True)
study_app = typer.Typer(no_args_is_help=True)
scientific_app = typer.Typer(no_args_is_help=True)
app.add_typer(hardware_app, name="hardware")
app.add_typer(backbone_app, name="backbone")
app.add_typer(request_app, name="request")
app.add_typer(experiment_app, name="experiment")
app.add_typer(verify_app, name="verify")
app.add_typer(study_app, name="study")
app.add_typer(scientific_app, name="scientific")
scientific_ablation_app.add_typer(scientific_ablation_workflow_app, name="workflow")
app.add_typer(scientific_ablation_app, name="scientific-ablation")


class RequestStrategy(StrEnum):
    """Explicit decoding policies supported by the local request command."""

    GREEDY = "greedy"
    SAMPLING = "sampling"


class ScientificWriterProtocol(StrEnum):
    """Explicit distinct stored writer-report protocol, not inferred from a candidate."""

    WORKSPACE = "workspace"
    MESSAGE = "message"


def _writer_protocol(value: ScientificWriterProtocol) -> WriterSourceProtocol:
    return "workspace" if value is ScientificWriterProtocol.WORKSPACE else "message"


class StudyRequestArm(StrEnum):
    """Four-arm protocol labels with executable profile-v4 request graphs."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class StudyExperimentArm(StrEnum):
    """Four-arm labels; B/C/D own executable teacher graphs in profile set v4."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class StudyBackboneProfile(StrEnum):
    """Explicit packaged profiles available for tokenizer-only diagnostics."""

    DEBUG = "qwen3_0_6b"
    MAIN = "qwen3_1_7b"


class StudyEvaluationMode(StrEnum):
    """A new run and an explicit continuation are distinct user choices."""

    NEW = "new"
    RESUME = "resume"


class StudyCodeProfile(StrEnum):
    """Only the installed, reviewed isolated Python profile is supported."""

    RESTRICTED = "iris_wsl_gvisor_python312_restricted_v1"


class ScientificCodeProfileV2(StrEnum):
    """Explicit source-resource policy; never reinterpret a legacy v1 report."""

    SOURCE = "iris_wsl_gvisor_python312_source_v2"


class StudyGenerationMode(StrEnum):
    """Generation never guesses whether an existing run should be resumed."""

    NEW = "new"
    RESUME = "resume"


class StudyGenerationSplit(StrEnum):
    """One complete explicitly selected dataset split."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _request_checkpoint_identity(
    *,
    arm: StudyRequestArm,
    payload_sha256: str | None,
    state_file_sha256: str | None,
    sidecar_state_sha256: str | None,
    completed_steps: int | None,
) -> IrisArmTeacherCheckpointIdentity | None:
    values = (
        payload_sha256,
        state_file_sha256,
        sidecar_state_sha256,
        completed_steps,
    )
    if arm is StudyRequestArm.A:
        if any(value is not None for value in values):
            message = "Arm A forbids all teacher checkpoint identity options"
            raise IrisRequestCommandError(message)
        return None
    if any(value is None for value in values):
        message = f"Arm {arm.value} requires all four teacher checkpoint identity options"
        raise IrisRequestCommandError(message)
    if (
        type(payload_sha256) is not str
        or type(state_file_sha256) is not str
        or type(sidecar_state_sha256) is not str
        or type(completed_steps) is not int
    ):
        message = "Teacher checkpoint identity options have invalid exact types"
        raise IrisRequestCommandError(message)
    if arm is StudyRequestArm.B:
        return IrisArmTeacherCheckpointIdentity(
            arm_id="B",
            checkpoint_schema_version=2,
            payload_sha256=payload_sha256,
            state_file_sha256=state_file_sha256,
            sidecar_state_sha256=sidecar_state_sha256,
            completed_steps=completed_steps,
        )
    if arm is StudyRequestArm.C:
        return IrisArmTeacherCheckpointIdentity(
            arm_id="C",
            checkpoint_schema_version=3,
            payload_sha256=payload_sha256,
            state_file_sha256=state_file_sha256,
            sidecar_state_sha256=sidecar_state_sha256,
            completed_steps=completed_steps,
        )
    return IrisArmTeacherCheckpointIdentity(
        arm_id="D",
        checkpoint_schema_version=1,
        payload_sha256=payload_sha256,
        state_file_sha256=state_file_sha256,
        sidecar_state_sha256=sidecar_state_sha256,
        completed_steps=completed_steps,
    )


def _private_metadata_table(
    title: str,
    *,
    status_label: str,
    status_value: str,
    rows: tuple[tuple[str, object], ...],
) -> Table:
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row(status_label, status_value)
    table.add_row("Output scope", "content-redacted private metadata")
    for field, value in rows:
        rendered = "not applicable" if value is None else escape(str(value))
        table.add_row(field, rendered)
    return table


def _verification_table(title: str, rows: tuple[tuple[str, object], ...]) -> Table:
    return _private_metadata_table(
        title,
        status_label="Verification",
        status_value="passed",
        rows=rows,
    )


def _debug_enabled(context: typer.Context | None) -> bool:
    if context is None:
        return False
    root_context = context.find_root()
    return bool(root_context.params.get("debug", root_context.meta.get("debug", False)))


def run_application[R](context: typer.Context, action: Callable[[], R]) -> R:
    """Render expected failures without a traceback unless debug mode is enabled."""
    try:
        return action()
    except ApplicationError as error:
        error_console = Console(stderr=True)
        if _debug_enabled(context):
            error_console.print_exception(show_locals=False)
        else:
            error_console.print(f"[bold red]Error:[/] {escape(str(error))}")
        exit_code = (
            FEATURE_UNAVAILABLE_EXIT_CODE
            if isinstance(error, (FeatureUnavailableError, IrisCodeEvaluationUnavailableError))
            else 1
        )
        raise typer.Exit(code=exit_code) from error
    except Exception as error:
        error_console = Console(stderr=True)
        if _debug_enabled(context):
            error_console.print_exception(show_locals=False)
        else:
            error_console.print(f"[bold red]Error:[/] {escape(str(error))}")
        raise typer.Exit(code=1) from error


@app.callback()
def root(
    context: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for development."),
) -> None:
    """Iris experimental neurocognitive architecture."""
    context.meta["debug"] = debug


@app.command("intervene")
def intervene(
    context: typer.Context,
    arguments: Annotated[
        list[str] | None,
        typer.Argument(
            help="key=value: checkpoint, intervention, intervention.module_id/sigma/etc., suite; "
            "optional context. Modes: run (default), generate, evaluate (capture=...), "
            "inspect (report=...), measure-nll (capture=...), inspect-nll (report=...). "
            "Context defaults to <checkpoint-name>.intervention.json "
            "beside checkpoint; suite defaults to core. No downloads."
        ),
    ] = None,
) -> None:
    """Run paired source-section18 interventions with an explicit local JSON context."""
    result = run_application(
        context, lambda: intervention_service.run_intervention_command_v1(tuple(arguments or ()))
    )
    console = Console()
    console.print(
        _private_metadata_table(
            "Scientific paired intervention",
            status_label="Scope",
            status_value=(
                "artifact integrity only; source not re-audited"
                if result.mode in {"inspect", "inspect-nll"}
                else "source-bound paired intervention; not scientific acceptance"
            ),
            rows=(
                ("Mode", result.mode),
                ("Capture directory", result.capture_path),
                ("Report directory", result.report_path),
                ("Capture SHA256", result.capture_sha256),
                ("Plan SHA256", result.plan_sha256),
                ("Selected step", result.selected_step),
                ("Paired examples", result.examples),
                ("Evaluation", "not run" if result.report is None else result.report.sha256),
                ("NLL report", None if result.nll_report is None else result.nll_report.sha256),
            ),
        )
    )
    if result.nll_report is not None:
        measured = result.nll_report.payload
        nll = measured.groups[0].nll
        console.print(
            _private_metadata_table(
                "Paired target NLL",
                status_label="Scientific launch",
                status_value=measured.scientific_launch_status,
                rows=(
                    ("NLL scope", measured.nll_scope),
                    ("Target tokens per branch", nll.target_token_count),
                    ("Control NLL/token", nll.control_mean_nll),
                    ("Intervened NLL/token", nll.intervened_mean_nll),
                    ("Delta (intervened - control)", nll.delta_mean_nll),
                ),
            )
        )
    if result.report is None:
        return
    payload = result.report.payload
    pair = payload.overall
    console.print(
        _private_metadata_table(
            "Paired evaluation scope",
            status_label="Scientific launch",
            status_value=payload.scientific_launch_status,
            rows=(
                ("NLL scope", payload.nll_scope),
                ("Difficulty scope", payload.difficulty_scope),
                ("Python resource scope", payload.python_resource_scope),
                ("Both correct", pair.both_correct),
                ("Control only correct", pair.control_only_correct),
                ("Intervened only correct", pair.intervened_only_correct),
                ("Neither correct", pair.neither_correct),
                ("Relative task-success drop", pair.relative_task_success_drop),
            ),
        )
    )
    table = Table(title="Paired quality rates (delta = intervened - control; absent = n/a)")
    for column in ("Metric", "Control", "Intervened", "Delta"):
        table.add_column(column)
    for metric, control in pair.control.rates.items():
        values = (control, pair.intervened.rates[metric], pair.rate_deltas[metric])
        table.add_row(metric, *("n/a" if value is None else f"{value:.6g}" for value in values))
    console.print(table)


def _scientific_report(report: scientific_command_service.IrisScientificCommandReportV1) -> None:
    Console().print(
        _private_metadata_table(
            "Scientific execution journal",
            status_label="Journal integrity",
            status_value="verified; not scientific acceptance",
            rows=(
                ("Execution path", report.path),
                ("Execution SHA256", report.execution_sha256),
                ("Run configuration SHA256", report.run_configuration_sha256),
                ("Optimizer purpose", report.purpose),
                ("Committed phases", report.sequence),
                ("Next phase", report.state.phase),
                ("Optimizer steps", report.state.step),
                ("Examples seen", report.examples_seen),
                ("Target tokens seen", report.target_tokens_seen),
                ("Best step", report.state.best_step),
                ("Best validation NLL/token", report.state.best_nll_per_token),
                ("Stop reason", report.state.stop_reason),
            ),
        )
    )


def _scientific_generation_summary(
    path: Path, result: IrisScientificGenerationV1, *, inspected: bool
) -> None:
    payload = result.payload
    Console().print(
        _private_metadata_table(
            "Scientific private generation",
            status_label="Scope",
            status_value=(
                "artifact integrity only; source not re-audited"
                if inspected
                else "source-bound generation; not scientific acceptance"
            ),
            rows=(
                ("Capture directory", path),
                ("Generation SHA256", result.sha256),
                ("Arm", payload.arm),
                ("Purpose", payload.purpose),
                ("Execution SHA256", payload.execution_sha256),
                ("Selected checkpoint SHA256", payload.best_checkpoint_sha256),
                ("Selected step", payload.selected_step),
                ("Terminal step", payload.terminal_step),
                ("Test corpus SHA256", payload.test_corpus_sha256),
                ("Examples", len(payload.rows)),
                ("Decoding", payload.decoding.strategy),
                ("Decoding seed", payload.decoding.seed),
                ("New-token limit", payload.max_new_tokens),
                ("Target tokens", payload.target_token_count),
                ("Captured target NLL/token", payload.target_nll_per_token),
            ),
        )
    )


def _scientific_evaluation_summary(
    path: Path, result: IrisScientificEvaluationV1 | IrisScientificEvaluationV2, *, inspected: bool
) -> None:
    payload = result.payload
    Console().print(
        _private_metadata_table(
            "Scientific quality report",
            status_label="Scope",
            status_value=(
                "artifact integrity only; source not re-audited"
                if inspected
                else "source-bound scoring; not scientific acceptance"
            ),
            rows=(
                ("Report directory", path),
                ("Evaluation SHA256", result.sha256),
                ("Report version", payload.kind),
                ("Generation SHA256", payload.generation_sha256),
                ("Arm", payload.arm),
                ("Purpose", payload.purpose),
                ("Selected step", payload.selected_step),
                ("Examples", payload.overall.examples),
                ("Scientific launch", payload.scientific_launch_status),
                ("NLL scope", payload.nll_scope),
                ("Python resource scope", payload.python_resource_scope),
            ),
        )
    )
    if isinstance(result, IrisScientificEvaluationV2) and result.payload.python_limits is not None:
        limits = result.payload.python_limits
        Console().print(
            _private_metadata_table(
                "Declared Python source limits",
                status_label="Scope",
                status_value="profile declaration; not historical execution attestation",
                rows=(
                    ("CPU cores", limits.cpu_cores),
                    ("Memory bytes", limits.memory_bytes),
                    ("Process limit", limits.process_limit),
                    ("Candidate timeout ms", limits.candidate_timeout_ms),
                    ("Clock scope", limits.clock_scope),
                    ("Environment", limits.environment),
                    ("Network", limits.network),
                ),
            )
        )
    table = Table(title="Quality metrics (absent denominator = n/a)")
    table.add_column("Metric")
    scopes = (
        ("overall", payload.overall),
        *((family.task_type, family.counts) for family in payload.families),
    )
    labels = {
        "overall": "Overall",
        "split_spec_repair": "Repair",
        "delayed_goal": "Goal",
        "conflict_resolution": "Conflict",
        "next_tool_prediction": "Tool",
    }
    for name, _ in scopes:
        table.add_column(labels[name])
    table.add_row("examples", *(str(counts.examples) for _, counts in scopes))
    for metric in payload.overall.rates:
        values = [counts.rates[metric] for _, counts in scopes]
        table.add_row(metric, *("n/a" if value is None else f"{value:.6g}" for value in values))
    Console().print(table)


@scientific_app.command("generate")
def scientific_generate(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution_path: Annotated[
        Path, typer.Option("--execution", help="Completed scientific journal.")
    ],
    test_input: Annotated[
        Path,
        typer.Option(
            "--test-input", help="Private JSON test Parquet descriptor; file paths relative to it."
        ),
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Private generation store root.")
    ],
    strategy: Annotated[RequestStrategy, typer.Option("--strategy")],
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)],
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 1.0,
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Decode exact validation-best D weights; persist answers privately, never execute code."""

    def action() -> tuple[Path, IrisScientificGenerationV1]:
        return scientific_answer_service.generate_scientific_answers_command_v1(
            config_path=config,
            execution_path=execution_path,
            test_input_path=test_input,
            output_root=output_root,
            maximum_storage_bytes=maximum_storage_bytes,
            decoding=ScientificDecodingPolicyV1.model_validate(
                {
                    "strategy": strategy.value,
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            ),
        )

    path, result = run_application(context, action)
    _scientific_generation_summary(path, result, inspected=False)


@scientific_app.command("evaluate")
def scientific_evaluate(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution_path: Annotated[Path, typer.Option("--execution")],
    test_input: Annotated[
        Path,
        typer.Option("--test-input", help="Same private JSON test descriptor used for generation."),
    ],
    generation_path: Annotated[
        Path, typer.Option("--generation", help="Complete private capture directory.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Separate hash-only quality report root.")
    ],
    code_profile: Annotated[
        StudyCodeProfile | None,
        typer.Option(
            "--code-profile",
            help="Explicit engineering-only isolation profile; required for repair rows.",
        ),
    ] = None,
    maximum_code_output_bytes: Annotated[
        int, typer.Option("--maximum-code-output-bytes", min=1024, max=8 * 1024**2)
    ] = 1024**2,
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Score saved scientific answers; no model load or regeneration, no host code execution."""

    def action() -> tuple[Path, IrisScientificEvaluationV1]:
        if code_profile is None and maximum_code_output_bytes != 1024**2:
            raise scientific_answer_service.IrisScientificAnswerCommandError(
                "A code output budget requires an explicit code profile"
            )
        policy = (
            None
            if code_profile is None
            else PythonSandboxPolicyV1.model_validate(
                {
                    "profile": code_profile.value,
                    "wall_time_ms": 5000,
                    "max_output_bytes": maximum_code_output_bytes,
                }
            )
        )
        return scientific_answer_service.evaluate_scientific_answers_command_v1(
            config_path=config,
            execution_path=execution_path,
            test_input_path=test_input,
            generation_path=generation_path,
            output_root=output_root,
            python_policy=policy,
            maximum_storage_bytes=maximum_storage_bytes,
        )

    path, result = run_application(context, action)
    _scientific_evaluation_summary(path, result, inspected=False)


@scientific_app.command("evaluate-v2")
def scientific_evaluate_v2(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution_path: Annotated[Path, typer.Option("--execution")],
    test_input: Annotated[
        Path,
        typer.Option("--test-input", help="Same private JSON test descriptor used for generation."),
    ],
    generation_path: Annotated[
        Path, typer.Option("--generation", help="Complete private capture directory.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Separate hash-only quality report root.")
    ],
    code_profile: Annotated[
        ScientificCodeProfileV2 | None,
        typer.Option(
            "--code-profile",
            help="Explicit section15 source isolation profile; required for repair rows.",
        ),
    ] = None,
    maximum_code_output_bytes: Annotated[
        int, typer.Option("--maximum-code-output-bytes", min=1024, max=8 * 1024**2)
    ] = 1024**2,
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Score saved answers into a distinct source-resource-v2 report; no model loading."""

    def action() -> tuple[Path, IrisScientificEvaluationV2]:
        if code_profile is None and maximum_code_output_bytes != 1024**2:
            raise scientific_answer_service.IrisScientificAnswerCommandError(
                "A code output budget requires an explicit code profile"
            )
        policy = (
            None
            if code_profile is None
            else PythonSandboxPolicyV2.model_validate(
                {
                    "profile": code_profile.value,
                    "wall_time_ms": 5000,
                    "max_output_bytes": maximum_code_output_bytes,
                }
            )
        )
        return scientific_answer_v2_service.evaluate_scientific_answers_command_v2(
            config_path=config,
            execution_path=execution_path,
            test_input_path=test_input,
            generation_path=generation_path,
            output_root=output_root,
            python_policy=policy,
            maximum_storage_bytes=maximum_storage_bytes,
        )

    path, result = run_application(context, action)
    _scientific_evaluation_summary(path, result, inspected=False)


@scientific_app.command("inspect-generation")
def scientific_inspect_generation(
    context: typer.Context,
    generation_path: Annotated[Path, typer.Option("--generation")],
) -> None:
    """Read capture integrity only; no source audit, model, code execution or raw text output."""
    result = run_application(
        context,
        lambda: scientific_generation_service.read_scientific_generation_v1(generation_path),
    )
    _scientific_generation_summary(generation_path, result, inspected=True)


@scientific_app.command("inspect-evaluation")
def scientific_inspect_evaluation(
    context: typer.Context,
    evaluation_path: Annotated[Path, typer.Option("--evaluation")],
) -> None:
    """Read hash/count integrity only; no source audit, regrading or model execution."""
    result = run_application(
        context,
        lambda: scientific_answer_v2_service.read_scientific_evaluation_report(evaluation_path),
    )
    _scientific_evaluation_summary(evaluation_path, result, inspected=True)


def _scientific_workspace_summary(
    path: Path, result: IrisScientificWorkspaceV1, *, inspected: bool
) -> None:
    payload = result.payload
    Console().print(
        _private_metadata_table(
            "Scientific workspace diagnostics",
            status_label="Scope",
            status_value="artifact integrity only; source not re-audited"
            if inspected
            else "source-bound diagnostics; not scientific acceptance",
            rows=(
                ("Artifact", str(path)),
                ("SHA256", result.sha256),
                ("Best step", str(payload.selected_step)),
                ("Terminal step", str(payload.terminal_step)),
                ("Selected / available", f"{len(payload.rows)} / {len(payload.members)}"),
                ("Teacher executions", str(payload.executions)),
                ("Trace mode", payload.policy.trace_mode),
                (
                    "Record ablation",
                    "measured" if payload.causal_summary is not None else "not measured",
                ),
                ("Aggregation", payload.aggregation_scope),
                ("Scientific acceptance", payload.scientific_acceptance),
            ),
        )
    )


def _scientific_writer_summary(
    result: scientific_writer_command_service.ScientificAblationWriterCommandResultV1,
) -> None:
    table = _private_metadata_table(
        "Scientific ablation stored writer report",
        status_label="Scope",
        status_value=result.scope,
        rows=(
            ("Artifact", str(result.path)),
            ("Protocol", result.protocol),
            ("Report SHA256", result.report_sha256),
            ("Original source SHA256", result.source_sha256),
            ("Whole effects / pairs", f"{result.effects} / {result.pairs}"),
            ("Stored teacher executions", str(result.stored_teacher_executions)),
            ("Scientific acceptance", result.scientific_acceptance),
        ),
    )
    # Exact job digests and literal paths must remain readable at narrow widths.
    # Other legacy command renderers retain their existing table policy.
    table.columns[1].overflow = "fold"
    Console().print(table)


@scientific_app.command("publish-writer")
def scientific_publish_writer(
    context: typer.Context,
    source: Annotated[
        Path, typer.Option("--source", help="Complete portable original source JSON.")
    ],
    report: Annotated[Path, typer.Option("--report", help="Complete canonical report JSON.")],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Immutable writer store root.")
    ],
    protocol: Annotated[ScientificWriterProtocol, typer.Option("--protocol")],
    source_sha256: Annotated[str, typer.Option("--expected-source-sha256")],
    report_sha256: Annotated[str, typer.Option("--expected-report-sha256")],
    torch_version: Annotated[str, typer.Option("--expected-torch-version")],
    torch_git_version: Annotated[str, typer.Option("--expected-torch-git-version")],
    maximum_source_bytes: Annotated[
        int, typer.Option("--maximum-source-bytes", min=1, max=512 * 1024**2)
    ],
    maximum_report_bytes: Annotated[
        int, typer.Option("--maximum-report-bytes", min=1, max=512 * 1024**2)
    ] = 512 * 1024**2,
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 512 * 1024**2,
    minimum_free_bytes: Annotated[
        int, typer.Option("--minimum-free-bytes", min=0, max=16 * 1024**3)
    ] = 1024**3,
) -> None:
    """Publish existing writer DATA; require independent job anchors, never remeasure."""
    result = run_application(
        context,
        lambda: scientific_writer_command_service.publish_scientific_ablation_writer_command_v1(
            source_path=source,
            report_path=report,
            output_root=output_root,
            expected_protocol=_writer_protocol(protocol),
            expected_source_sha256=source_sha256,
            expected_report_sha256=report_sha256,
            runtime=ScientificAblationWriterCPURuntimeV1(torch_version, torch_git_version),
            maximum_source_bytes=maximum_source_bytes,
            maximum_report_bytes=maximum_report_bytes,
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
        ),
    )
    _scientific_writer_summary(result)


@scientific_app.command("inspect-writer")
def scientific_inspect_writer(
    context: typer.Context,
    path: Annotated[Path, typer.Argument(help="Saved full two-file writer bundle directory.")],
    protocol: Annotated[ScientificWriterProtocol, typer.Option("--protocol")],
    source_sha256: Annotated[str, typer.Option("--expected-source-sha256")],
    report_sha256: Annotated[str, typer.Option("--expected-report-sha256")],
    torch_version: Annotated[str, typer.Option("--expected-torch-version")],
    torch_git_version: Annotated[str, typer.Option("--expected-torch-git-version")],
    maximum_source_bytes: Annotated[
        int, typer.Option("--maximum-source-bytes", min=1, max=512 * 1024**2)
    ],
    maximum_report_bytes: Annotated[
        int, typer.Option("--maximum-report-bytes", min=1, max=512 * 1024**2)
    ] = 512 * 1024**2,
) -> None:
    """Validate ENTIRE stored writer DATA against caller job anchors; never load a model."""
    result = run_application(
        context,
        lambda: scientific_writer_command_service.inspect_scientific_ablation_writer_command_v1(
            path,
            expected_protocol=_writer_protocol(protocol),
            expected_source_sha256=source_sha256,
            expected_report_sha256=report_sha256,
            runtime=ScientificAblationWriterCPURuntimeV1(torch_version, torch_git_version),
            maximum_source_bytes=maximum_source_bytes,
            maximum_report_bytes=maximum_report_bytes,
        ),
    )
    _scientific_writer_summary(result)


@scientific_app.command("diagnose-workspace")
def scientific_diagnose_workspace(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution: Annotated[Path, typer.Option("--execution", help="Completed scientific journal.")],
    generation: Annotated[
        Path, typer.Option("--generation", help="Saved held-out baseline capture.")
    ],
    test_input: Annotated[
        Path, typer.Option("--test-input", help="Private test Parquet descriptor.")
    ],
    policy: Annotated[
        Path, typer.Option("--policy", help="Explicit bounded diagnostic policy JSON.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Private diagnostic store root.")
    ],
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Measure selected-best workspace traces; never train, regenerate answers or execute code."""
    path, result = run_application(
        context,
        lambda: scientific_workspace_command_service.diagnose_scientific_workspace_command_v1(
            config_path=config,
            execution_path=execution,
            generation_path=generation,
            test_input_path=test_input,
            policy_path=policy,
            output_root=output_root,
            maximum_storage_bytes=maximum_storage_bytes,
        ),
    )
    _scientific_workspace_summary(path, result, inspected=False)


@scientific_app.command("inspect-workspace")
def scientific_inspect_workspace(
    context: typer.Context,
    path: Annotated[Path, typer.Argument(help="Saved workspace diagnostic artifact directory.")],
) -> None:
    """Inspect stored integrity/selection/aggregates without loading any model or source."""
    result = run_application(
        context, lambda: scientific_workspace_service.read_scientific_workspace_v1(path)
    )
    _scientific_workspace_summary(path, result, inspected=True)


def _scientific_message_summary(
    path: Path, result: IrisScientificMessageCausalV1, *, inspected: bool
) -> None:
    payload = result.payload
    Console().print(
        _private_metadata_table(
            "Scientific paired message diagnostics",
            status_label="Scope",
            status_value="artifact integrity only; source not re-audited"
            if inspected
            else "selected-best source-bound diagnostics; not scientific acceptance",
            rows=(
                ("Artifact", str(path)),
                ("SHA256", result.sha256),
                ("Best step", str(payload.selected_step)),
                ("Terminal step", str(payload.terminal_step)),
                ("Selected / available", f"{len(payload.rows)} / {len(payload.members)}"),
                ("Disjoint donor pairs", str(len(payload.causal.donor_pairs))),
                ("Teacher executions", str(payload.executions)),
                ("Stored trace mode", payload.policy.trace_mode),
                ("Aggregation", payload.aggregation_scope),
                ("Scientific acceptance", payload.scientific_acceptance),
            ),
        )
    )


@scientific_app.command("diagnose-messages")
def scientific_diagnose_messages(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution: Annotated[Path, typer.Option("--execution", help="Completed scientific journal.")],
    generation: Annotated[
        Path, typer.Option("--generation", help="Saved held-out baseline capture.")
    ],
    test_input: Annotated[
        Path, typer.Option("--test-input", help="Private test Parquet descriptor.")
    ],
    policy: Annotated[
        Path, typer.Option("--policy", help="Explicit even-subset message policy JSON.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Private message report store.")
    ],
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Measure isolated writes/donor content at selected best; never execute generated code."""
    path, result = run_application(
        context,
        lambda: scientific_message_command_service.diagnose_scientific_messages_command_v1(
            config_path=config,
            execution_path=execution,
            generation_path=generation,
            test_input_path=test_input,
            policy_path=policy,
            output_root=output_root,
            maximum_storage_bytes=maximum_storage_bytes,
        ),
    )
    _scientific_message_summary(path, result, inspected=False)


@scientific_app.command("inspect-messages")
def scientific_inspect_messages(
    context: typer.Context,
    path: Annotated[Path, typer.Argument(help="Saved paired message report directory.")],
) -> None:
    """Inspect stored design/statistics integrity without model loading or source audit."""
    result = run_application(
        context, lambda: scientific_message_service.read_scientific_messages_v1(path)
    )
    _scientific_message_summary(path, result, inspected=True)


def _scientific_collapse_summary(
    path: Path, result: IrisScientificCollapseV1, *, inspected: bool
) -> None:
    payload = result.payload
    summary = payload.groups[0].summary
    assert summary is not None
    Console().print(
        _private_metadata_table(
            "Scientific full-test collapse diagnostics",
            status_label="Scope",
            status_value="artifact integrity only; source not re-audited"
            if inspected
            else "source-bound full-test allocations; not overall scientific acceptance",
            rows=(
                ("Artifact", str(path)),
                ("SHA256", result.sha256),
                ("Best step", str(payload.selected_step)),
                ("Terminal step", str(payload.terminal_step)),
                ("Measured / available", f"{len(payload.rows)} / {len(payload.members)}"),
                ("Teacher executions", str(payload.executions)),
                ("Effective modules", f"{summary.effective_active_modules:.6g}"),
                ("Effective slots", f"{summary.effective_active_slots:.6g}"),
                ("Source20.4 one-seed gate", payload.source_20_4.outcome),
                ("Aggregation", payload.aggregation_scope),
                ("Scientific acceptance", payload.scientific_acceptance),
            ),
        )
    )


@scientific_app.command("diagnose-collapse")
def scientific_diagnose_collapse(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Original offline training recipe.")],
    execution: Annotated[Path, typer.Option("--execution", help="Completed scientific journal.")],
    generation: Annotated[
        Path, typer.Option("--generation", help="Saved complete held-out baseline capture.")
    ],
    test_input: Annotated[
        Path, typer.Option("--test-input", help="Private complete test Parquet descriptor.")
    ],
    policy: Annotated[
        Path, typer.Option("--policy", help="Explicit full-corpus budget policy JSON; no subset.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Private full-test allocation report store.")
    ],
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 256 * 1024**2,
) -> None:
    """Measure every test allocation at selected best; no generation, training or code execution."""
    path, result = run_application(
        context,
        lambda: scientific_collapse_command_service.diagnose_scientific_collapse_command_v1(
            config_path=config,
            execution_path=execution,
            generation_path=generation,
            test_input_path=test_input,
            policy_path=policy,
            output_root=output_root,
            maximum_storage_bytes=maximum_storage_bytes,
        ),
    )
    _scientific_collapse_summary(path, result, inspected=False)


@scientific_app.command("inspect-collapse")
def scientific_inspect_collapse(
    context: typer.Context,
    path: Annotated[Path, typer.Argument(help="Saved complete-test collapse report directory.")],
) -> None:
    """Inspect stored coverage/count/entropy integrity without loading a model or source."""
    result = run_application(
        context, lambda: scientific_collapse_service.read_scientific_collapse_v1(path)
    )
    _scientific_collapse_summary(path, result, inspected=True)


@scientific_app.command("create")
def scientific_create(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Explicit offline JSON recipe.")],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Local journal/checkpoint root.")
    ],
) -> None:
    """Preflight candidate data and persist initialization; no training or downloads."""
    _scientific_report(
        run_application(
            context,
            lambda: scientific_command_service.create_scientific_command_v1(
                config_path=config, output_root=output_root
            ),
        )
    )


@scientific_app.command("resume")
def scientific_resume(
    context: typer.Context,
    config: Annotated[
        Path, typer.Option("--config", help="Original compatible offline JSON recipe.")
    ],
    execution_path: Annotated[Path, typer.Option("--execution", help="Exact execution directory.")],
    phases: Annotated[
        int,
        typer.Option(
            "--phases",
            min=1,
            max=1000,
            help="Maximum NEW phases after exact tail replay (not an optimizer-step/time limit).",
        ),
    ],
    delivery_root: Annotated[
        Path | None,
        typer.Option(
            "--delivery-root", help="Opt-in current/best nine-file exports at every checkpoint."
        ),
    ] = None,
    delivery_budget_bytes: Annotated[
        int,
        typer.Option(
            "--delivery-budget-bytes",
            min=1,
            max=16 * 1024**3,
            help="Combined nine-file delivery store cap; no automatic deletion.",
        ),
    ] = 1024**3,
) -> None:
    """Restore/replay explicitly, then advance a bounded number of phases; CPU/FP32 only."""
    _scientific_report(
        run_application(
            context,
            lambda: scientific_command_service.resume_scientific_command_v1(
                config_path=config,
                execution_path=execution_path,
                phases=phases,
                delivery_root=delivery_root,
                maximum_delivery_storage_bytes=delivery_budget_bytes,
            ),
        )
    )


@scientific_app.command("inspect")
def scientific_inspect(
    context: typer.Context,
    execution_path: Annotated[Path, typer.Option("--execution", help="Exact execution directory.")],
) -> None:
    """Audit journal/checkpoint integrity without loading models or advancing training."""
    _scientific_report(
        run_application(
            context,
            lambda: scientific_command_service.inspect_scientific_command_v1(
                execution_path=execution_path
            ),
        )
    )


@scientific_app.command("project-mlflow")
def scientific_project_mlflow(
    context: typer.Context,
    execution_path: Annotated[Path, typer.Option("--execution", help="Exact scientific journal.")],
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Separate local SQLite metadata directory.")
    ],
    experiment_name: Annotated[str, typer.Option("--experiment-name")],
    maximum_metric_points: Annotated[
        int, typer.Option("--maximum-metric-points", min=1, max=4_000_000)
    ] = 1_000_000,
    maximum_store_bytes: Annotated[
        int,
        typer.Option(
            "--maximum-store-bytes",
            min=1,
            max=1024**3,
            help="Observed bytes checked between SDK calls, not a peak-size guarantee.",
        ),
    ] = 1024**3,
) -> None:
    """Verify/project scientific metrics into local MLflow; no model, uploads or training."""
    report = run_application(
        context,
        lambda: scientific_mlflow_service.project_scientific_execution_to_mlflow_v1(
            execution_path=execution_path,
            output_root=output_root,
            experiment_name=experiment_name,
            maximum_metric_points=maximum_metric_points,
            maximum_store_bytes=maximum_store_bytes,
        ),
    )
    Console().print(
        _private_metadata_table(
            "Scientific local MLflow projection",
            status_label="Projection",
            status_value="verified; not scientific acceptance",
            rows=(
                ("Execution SHA256", report.execution_sha256),
                ("Committed source events", report.event_count),
                ("SQLite database", report.tracking_database),
                ("MLflow experiment ID", report.experiment_id),
                ("MLflow run ID", report.mlflow_run_id),
                ("Metric points", report.metric_point_count),
                ("New metric points", report.new_metric_point_count),
                ("Source status", report.status),
            ),
        )
    )


@scientific_app.command("inspect-deliveries")
def scientific_inspect_deliveries(
    context: typer.Context,
    execution_path: Annotated[Path, typer.Option("--execution", help="Exact execution directory.")],
    delivery_root: Annotated[
        Path, typer.Option("--delivery-root", help="Separate delivery store root.")
    ],
) -> None:
    """Audit current/best delivered bytes and receipts without a model or training."""
    report = run_application(
        context,
        lambda: scientific_publication_service.inspect_scientific_publications_v1(
            execution_path=execution_path,
            output_root=delivery_root,
        ),
    )
    Console().print(
        _private_metadata_table(
            "Scientific checkpoint deliveries",
            status_label="Integrity",
            status_value="verified; not scientific acceptance",
            rows=(
                ("Execution SHA256", report.execution_sha256),
                ("Source events", report.source_event_count),
                ("Checkpoint events", report.checkpoint_count),
                ("Verified receipts", len(report.receipts)),
                ("Projection complete", report.complete),
            ),
        )
    )


@hardware_app.command("inspect")
def hardware_inspect(
    context: typer.Context,
    device: str = typer.Option("cpu", "--device", help="Explicit device: cpu or cuda."),
) -> None:
    """Inspect an explicitly requested compute device."""
    report = run_application(context, lambda: hardware_service.inspect_hardware(device))
    table = Table(title="Hardware inspection", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Requested device", report.requested_device)
    table.add_row("Resolved backend", report.resolved_backend)
    table.add_row("Device name", report.device_name)
    total_memory = "not reported"
    if report.total_memory_bytes is not None:
        total_memory = str(report.total_memory_bytes)
    table.add_row("Total memory", total_memory)
    table.add_row("FP16 support", str(report.supports_fp16).lower())
    table.add_row("BF16 support", str(report.supports_bf16).lower())
    table.add_row("PyTorch version", report.torch_version)
    table.add_row("CUDA runtime", report.cuda_runtime or "not available")
    Console().print(table)


@backbone_app.command("inspect")
def backbone_inspect(context: typer.Context) -> None:
    """Inspect backbone configuration without loading model weights."""
    report = run_application(context, backbone_service.inspect_backbone)
    table = Table(title="Backbone inspection", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Backbone", report.name)
    table.add_row("Configured", str(report.configured).lower())
    table.add_row("Loaded", str(report.loaded).lower())
    Console().print(table)


@backbone_app.command("forward")
def backbone_forward(
    context: typer.Context,
    text: str = typer.Option(..., "--text", help="Prompt to run through the frozen model."),
) -> None:
    """Run a frozen backbone forward pass on CPU."""
    report = run_application(context, lambda: backbone_service.forward_backbone(text))
    table = Table(title="Backbone forward", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Backbone", report.backbone)
    table.add_row("Revision", report.revision)
    table.add_row("Device", report.device.upper())
    table.add_row("Dtype", report.dtype)
    table.add_row("Trainable parameters", str(report.trainable_parameters))
    table.add_row("Prompt tokens", str(report.prompt_tokens))
    table.add_row("Vocabulary size", str(report.vocabulary_size))
    table.add_row("Hidden layers returned", str(report.hidden_layers))
    table.add_row("Hidden size", str(report.hidden_size))
    table.add_row("Logits shape", str(list(report.logits_shape)))
    table.add_row("Last hidden shape", str(list(report.last_hidden_shape)))
    table.add_row("Finite logits", str(report.finite_logits).lower())
    table.add_row("Finite hidden states", str(report.finite_hidden_states).lower())
    table.add_row("Forward completed", str(report.forward_completed).lower())
    console = Console()
    console.print(table)
    console.print(f"[bold]Last hidden SHA-256[/] {report.last_hidden_sha256}", soft_wrap=True)
    console.print(f"[bold]Artifact[/] {escape(str(report.artifact_path))}", soft_wrap=True)


@request_app.command("generate")
def request_generate(
    context: typer.Context,
    prompt: Annotated[
        str,
        typer.Option("--prompt", help="Prompt for one complete Iris request."),
    ],
    strategy: Annotated[
        RequestStrategy,
        typer.Option(
            "--strategy",
            help="Explicit decoding strategy: greedy or sampling.",
        ),
    ],
    stop_token_id: Annotated[
        list[int],
        typer.Option(
            "--stop-token-id",
            help="Explicit stop token ID; repeat for multiple IDs.",
        ),
    ],
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="Explicit sampling temperature used to compose the request runtime.",
        ),
    ],
    top_p: Annotated[
        float,
        typer.Option(
            "--top-p",
            help="Explicit nucleus probability used to compose the request runtime.",
        ),
    ],
    sampling_seed: Annotated[
        int | None,
        typer.Option(
            "--sampling-seed",
            help="Required for sampling and forbidden for greedy generation.",
        ),
    ] = None,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned model to be downloaded when absent from the local cache.",
        ),
    ] = False,
) -> None:
    """Run one complete request and atomically persist private provenance."""

    def run_request() -> None:
        output = request_service.generate_request(
            prompt=prompt,
            strategy=strategy.value,
            stop_token_ids=tuple(stop_token_id),
            sampling=TemperatureTopPSamplingConfig(
                temperature=temperature,
                top_p=top_p,
            ),
            sampling_seed=sampling_seed,
            allow_download=allow_download,
        )
        sys.stdout.write(output.text)

    run_application(context, run_request)


@request_app.command("generate-arm")
def request_generate_arm(
    context: typer.Context,
    arm: Annotated[
        StudyRequestArm,
        typer.Option("--arm", help="Explicit four-arm protocol label: A, B, C, or D."),
    ],
    prompt: Annotated[
        str,
        typer.Option("--prompt", help="Prompt for one arm-labelled Iris request."),
    ],
    strategy: Annotated[
        RequestStrategy,
        typer.Option("--strategy", help="Explicit decoding strategy: greedy or sampling."),
    ],
    stop_token_id: Annotated[
        list[int],
        typer.Option(
            "--stop-token-id",
            help="Explicit stop token ID; repeat for multiple IDs.",
        ),
    ],
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Explicit sampling temperature."),
    ],
    top_p: Annotated[
        float,
        typer.Option("--top-p", help="Explicit nucleus probability."),
    ],
    sampling_seed: Annotated[
        int | None,
        typer.Option(
            "--sampling-seed",
            help="Required for sampling and forbidden for greedy generation.",
        ),
    ] = None,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned model to be downloaded when absent from the local cache.",
        ),
    ] = False,
    checkpoint_payload_sha256: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-payload-sha256",
            help="Required B/C/D teacher checkpoint payload identity; forbidden for A.",
        ),
    ] = None,
    checkpoint_state_file_sha256: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-state-file-sha256",
            help="Required B/C/D checkpoint safetensors identity; forbidden for A.",
        ),
    ] = None,
    checkpoint_sidecar_state_sha256: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-sidecar-state-sha256",
            help="Required B/C/D teacher sidecar identity; forbidden for A.",
        ),
    ] = None,
    checkpoint_completed_steps: Annotated[
        int | None,
        typer.Option(
            "--checkpoint-completed-steps",
            help="Required B/C/D exact optimizer step count; forbidden for A.",
        ),
    ] = None,
) -> None:
    """Run one checkpoint-bound profile-v4 request and persist artifact v4."""

    def run_request() -> None:
        arm_request_service.require_arm_request_command_profile_v4(arm.value)
        checkpoint_identity = _request_checkpoint_identity(
            arm=arm,
            payload_sha256=checkpoint_payload_sha256,
            state_file_sha256=checkpoint_state_file_sha256,
            sidecar_state_sha256=checkpoint_sidecar_state_sha256,
            completed_steps=checkpoint_completed_steps,
        )
        output = arm_request_service.generate_arm_request_v4(
            arm_id=arm.value,
            checkpoint_identity=checkpoint_identity,
            prompt=prompt,
            strategy=strategy.value,
            stop_token_ids=tuple(stop_token_id),
            sampling=TemperatureTopPSamplingConfig(
                temperature=temperature,
                top_p=top_p,
            ),
            sampling_seed=sampling_seed,
            allow_download=allow_download,
        )
        sys.stdout.write(output.text)

    run_application(context, run_request)


@verify_app.command("request")
def verify_request(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Canonical request artifact JSON path."),
    ],
) -> None:
    """Verify one installed request artifact without printing private text."""
    if path.parent.name == "v4":
        report_v4 = run_application(
            context,
            lambda: verification_v4_service.verify_request_artifact_v4(path),
        )
        Console().print(
            _verification_table(
                "Checkpoint-bound request artifact verification",
                (
                    ("Path", report_v4.path),
                    ("Artifact type", report_v4.artifact_type),
                    ("Schema version", report_v4.version),
                    ("Arm", report_v4.arm_id),
                    ("Arm kind", report_v4.arm_kind),
                    ("Request graph", report_v4.request_graph),
                    ("Execution profile version", report_v4.execution_profile_set_version),
                    ("Execution profile SHA-256", report_v4.execution_profile_set_sha256),
                    ("Shared-buffer policy SHA-256", report_v4.shared_buffer_policy_sha256),
                    ("Payload SHA-256", report_v4.payload_sha256),
                    ("Configuration SHA-256", report_v4.configuration_sha256),
                    ("Generated tokens", report_v4.generated_tokens),
                    ("Stop reason", report_v4.stop_reason),
                    ("Trace contract", report_v4.trace_contract),
                    ("Workspace cycles", report_v4.workspace_cycles),
                    ("Cognitive cycles", report_v4.cognitive_cycles),
                    ("Latent tokens", report_v4.latent_token_count),
                    ("Zero-memory-read cycles", report_v4.zero_memory_read_cycles),
                    ("Pre-update read entries", report_v4.pre_update_read_entry_counts),
                    ("Post-cycle entries", report_v4.post_cycle_entry_counts),
                    ("Final output read entries", report_v4.final_output_read_entry_count),
                    ("Sidecar present", str(report_v4.sidecar_present).lower()),
                    ("Checkpoint present", str(report_v4.checkpoint_present).lower()),
                    ("Checkpoint schema version", report_v4.checkpoint_schema_version),
                    ("Checkpoint payload SHA-256", report_v4.checkpoint_payload_sha256),
                    ("Checkpoint state file SHA-256", report_v4.checkpoint_state_file_sha256),
                    ("Teacher sidecar SHA-256", report_v4.source_teacher_sidecar_sha256),
                    ("Request sidecar SHA-256", report_v4.request_sidecar_state_sha256),
                    ("Checkpoint completed steps", report_v4.checkpoint_completed_steps),
                    ("Transferred tensors", report_v4.transferred_tensor_count),
                    ("Deterministic", str(report_v4.deterministic).lower()),
                    ("Backend", report_v4.backend),
                    ("Dtype", report_v4.dtype),
                ),
            )
        )
        return
    if path.parent.name == "v3":
        report_v3 = run_application(
            context,
            lambda: verification_v3_service.verify_request_artifact_v3(path),
        )
        Console().print(
            _verification_table(
                "Profile-v4 request artifact verification",
                (
                    ("Path", report_v3.path),
                    ("Artifact type", report_v3.artifact_type),
                    ("Schema version", report_v3.version),
                    ("Arm", report_v3.arm_id),
                    ("Arm kind", report_v3.arm_kind),
                    ("Request graph", report_v3.request_graph),
                    ("Execution profile version", report_v3.execution_profile_set_version),
                    ("Execution profile SHA-256", report_v3.execution_profile_set_sha256),
                    ("Shared-buffer policy SHA-256", report_v3.shared_buffer_policy_sha256),
                    ("Payload SHA-256", report_v3.payload_sha256),
                    ("Configuration SHA-256", report_v3.configuration_sha256),
                    ("Generated tokens", report_v3.generated_tokens),
                    ("Stop reason", report_v3.stop_reason),
                    ("Trace contract", report_v3.trace_contract),
                    ("Workspace cycles", report_v3.workspace_cycles),
                    ("Cognitive cycles", report_v3.cognitive_cycles),
                    ("Latent tokens", report_v3.latent_token_count),
                    ("Zero-memory-read cycles", report_v3.zero_memory_read_cycles),
                    ("Pre-update read entries", report_v3.pre_update_read_entry_counts),
                    ("Post-cycle entries", report_v3.post_cycle_entry_counts),
                    ("Final output read entries", report_v3.final_output_read_entry_count),
                    ("Sidecar present", str(report_v3.sidecar_present).lower()),
                    ("Deterministic", str(report_v3.deterministic).lower()),
                    ("Backend", report_v3.backend),
                    ("Dtype", report_v3.dtype),
                ),
            )
        )
        return
    if path.parent.name == "v2":
        report_v2 = run_application(
            context,
            lambda: verification_v2_service.verify_request_artifact_v2(path),
        )
        Console().print(
            _verification_table(
                "Arm-labelled request artifact verification",
                (
                    ("Path", report_v2.path),
                    ("Artifact type", report_v2.artifact_type),
                    ("Schema version", report_v2.version),
                    ("Arm", report_v2.arm_id),
                    ("Arm kind", report_v2.arm_kind),
                    ("Request graph", report_v2.request_graph),
                    ("Execution profile version", report_v2.execution_profile_set_version),
                    ("Execution profile SHA-256", report_v2.execution_profile_set_sha256),
                    ("Payload SHA-256", report_v2.payload_sha256),
                    ("Configuration SHA-256", report_v2.configuration_sha256),
                    ("Generated tokens", report_v2.generated_tokens),
                    ("Stop reason", report_v2.stop_reason),
                    ("Trace contract", report_v2.trace_contract),
                    ("Workspace cycles", report_v2.workspace_cycles),
                    ("Cognitive cycles", report_v2.cognitive_cycles),
                    ("Latent tokens", report_v2.latent_token_count),
                    ("Zero-memory-read cycles", report_v2.zero_memory_read_cycles),
                    ("Sidecar present", str(report_v2.sidecar_present).lower()),
                    ("Deterministic", str(report_v2.deterministic).lower()),
                    ("Backend", report_v2.backend),
                    ("Dtype", report_v2.dtype),
                ),
            )
        )
        return
    report = run_application(
        context,
        lambda: verification_service.verify_request_artifact_v1(path),
    )
    Console().print(
        _verification_table(
            "Request artifact verification",
            (
                ("Path", report.path),
                ("Artifact type", report.artifact_type),
                ("Schema version", report.version),
                ("Payload SHA-256", report.payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Generated tokens", report.generated_tokens),
                ("Stop reason", report.stop_reason),
                ("Workspace cycles", report.workspace_cycles),
                ("Deterministic", str(report.deterministic).lower()),
                ("Backend", report.backend),
                ("Dtype", report.dtype),
            ),
        )
    )


@verify_app.command("checkpoint")
def verify_checkpoint(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Canonical teacher checkpoint directory."),
    ],
) -> None:
    """Verify a checkpoint manifest and all safetensors values."""
    report = run_application(
        context,
        lambda: verification_service.verify_teacher_checkpoint_v1(path),
    )
    Console().print(
        _verification_table(
            "Teacher checkpoint verification",
            (
                ("Path", report.path),
                ("Artifact type", report.artifact_type),
                ("Schema version", report.version),
                ("Payload SHA-256", report.payload_sha256),
                ("State file SHA-256", report.state_file_sha256),
                ("State size bytes", report.state_size_bytes),
                ("Tensor count", report.tensor_count),
                ("Sidecar state SHA-256", report.sidecar_state_sha256),
                ("Completed steps", report.completed_steps),
                ("Dtype", report.dtype),
            ),
        )
    )


@verify_app.command("manifest")
def verify_manifest(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Experiment example manifest JSON path."),
    ],
) -> None:
    """Verify a manifest without printing example IDs or text."""
    report = run_application(
        context,
        lambda: verification_service.verify_experiment_manifest_v1(path),
    )
    Console().print(
        _verification_table(
            "Experiment manifest verification",
            (
                ("Path", report.path),
                ("Artifact type", report.artifact_type),
                ("Schema version", report.version),
                ("Payload SHA-256", report.payload_sha256),
                ("Dataset ID", report.dataset_id),
                ("Train examples", report.train_examples),
                ("Validation examples", report.validation_examples),
                ("Test examples", report.test_examples),
                ("Total examples", report.total_examples),
            ),
        )
    )


@verify_app.command("study-dataset")
def verify_study_dataset(
    context: typer.Context,
    path: Annotated[
        Path, typer.Option("--path", help="Canonical task-labelled study dataset JSON path.")
    ],
) -> None:
    """Verify private input membership and hashes, not scientific launch readiness."""
    report = run_application(context, lambda: study_dataset_service.verify_study_dataset_v1(path))
    rows: tuple[tuple[str, object], ...] = (
        ("Path", report.path),
        ("Dataset payload SHA-256", report.payload_sha256),
        ("Manifest payload SHA-256", report.manifest_payload_sha256),
        ("Protocol payload SHA-256", report.protocol_declaration_payload_sha256),
        ("Dataset ID", report.dataset_id),
        ("Purpose", report.purpose),
        ("Total examples", report.total_examples),
        ("Leakage groups", report.leakage_group_count),
        ("Validation scope", report.validation_scope),
        ("Scientific launch", report.scientific_launch_status),
    )
    rows += tuple(
        (item.task_family, f"train={item.train}, validation={item.validation}, test={item.test}")
        for item in report.task_counts
    )
    Console().print(_verification_table("Study dataset verification", rows))


@study_app.command("build-pilot")
def study_build_pilot(
    context: typer.Context,
    seed: Annotated[
        int, typer.Option("--seed", help="Explicit deterministic pilot generation seed.")
    ],
    problems_per_cell: Annotated[
        int,
        typer.Option("--problems-per-cell", help="Problems per family/split cell, from 1 to 256."),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path", help="Private output .json including targets and hidden code cases."
        ),
    ],
) -> None:
    """Build five-family operational tasks; no model or candidate code is executed."""

    def build_and_write() -> study_benchmark_io_service.IrisStudyBenchmarkWriteResultV1:
        benchmark = study_pilot_service.build_synthetic_study_pilot_v1(
            seed=seed, problems_per_cell=problems_per_cell
        )
        return study_benchmark_io_service.write_study_benchmark_v1(benchmark, path=path)

    result = run_application(context, build_and_write)
    Console().print(
        _private_metadata_table(
            "Operational five-family pilot",
            status_label="Build",
            status_value="reused" if result.reused_existing else "written",
            rows=(
                ("Path", result.path),
                ("Benchmark SHA-256", result.payload_sha256),
                ("Examples", 15 * problems_per_cell),
                ("Size bytes", result.size_bytes),
                ("Purpose", "operational_fixture"),
                ("Scientific launch", "not assessed"),
                ("Code scoring", "requires an isolated Python backend"),
            ),
        )
    )


@verify_app.command("study-benchmark")
def verify_study_benchmark(
    context: typer.Context,
    path: Annotated[
        Path, typer.Option("--path", help="Canonical private study benchmark JSON path.")
    ],
) -> None:
    """Verify dataset/scoring bindings; do not execute hidden tests or approve a launch."""
    benchmark = run_application(
        context, lambda: study_benchmark_io_service.read_study_benchmark_v1(path)
    )
    dataset = benchmark.payload.dataset
    Console().print(
        _verification_table(
            "Study benchmark verification",
            (
                ("Path", path),
                ("Benchmark SHA-256", benchmark.payload_sha256),
                ("Dataset SHA-256", dataset.payload_sha256),
                ("Purpose", dataset.payload.purpose),
                ("Examples", len(benchmark.payload.scoring)),
                ("Scientific launch", "not assessed"),
                ("Code scoring", "requires an isolated Python backend"),
            ),
        )
    )


@study_app.command("preflight-inputs")
def study_preflight_inputs(
    context: typer.Context,
    benchmark: Annotated[Path, typer.Option("--benchmark", help="Private benchmark JSON.")],
    terminal_token_id: Annotated[
        int, typer.Option("--terminal-token-id", help="Explicit teacher EOS.")
    ],
    backbone_profile: Annotated[
        StudyBackboneProfile,
        typer.Option("--backbone-profile", help="Pinned tokenizer/model config."),
    ],
    max_prompt_tokens: Annotated[int | None, typer.Option("--max-prompt-tokens")] = None,
    max_new_tokens: Annotated[int | None, typer.Option("--max-new-tokens")] = None,
) -> None:
    """Offline full-input check; no weights, generated-code execution, or silent truncation."""
    report = run_application(
        context,
        lambda: study_input_service.preflight_study_input_file_v1(
            benchmark,
            terminal_token_id=terminal_token_id,
            backbone_profile=backbone_profile.value,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
        ),
    )
    failed = sum(bool(example.violations) for example in report.examples)
    Console().print(
        _private_metadata_table(
            "Study input budget preflight",
            status_label="Input budgets",
            status_value="fit" if report.inputs_fit else "blocked",
            rows=(
                ("Benchmark SHA-256", report.benchmark_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Tokenizer SHA-256", report.tokenizer_semantics_sha256),
                ("Examples checked", len(report.examples)),
                ("Examples with violations", failed),
                ("Prompt token limit", report.max_prompt_tokens),
                ("Target/generation token limit", report.max_new_tokens),
                ("Longest prompt", max(item.prompt_tokens for item in report.examples)),
                (
                    "Longest target incl. EOS",
                    max(item.target_tokens_including_terminal for item in report.examples),
                ),
                ("A/B/C/D latent tokens", report.arm_latent_tokens),
                ("Configured context envelope fits", report.configured_context_envelope_fits),
                ("Scientific launch", report.scientific_launch_status),
            ),
        )
    )
    if not report.inputs_fit:
        raise typer.Exit(code=2)


def _study_evaluation_rows(
    report: study_evaluation_service.IrisStudyEvaluationFileReportV1,
) -> tuple[tuple[str, object], ...]:
    rows: tuple[tuple[str, object], ...] = (
        ("Directory", report.directory),
        ("Generation SHA-256", report.generation_payload_sha256),
        ("Benchmark SHA-256", report.benchmark_payload_sha256),
        ("Run SHA-256", report.run_payload_sha256),
        ("Summary SHA-256", report.summary_payload_sha256),
        ("Arm", report.arm_id),
        ("Split", report.split),
        ("Total examples", report.overall.total),
        ("Correct", report.overall.correct),
        ("Incorrect", report.overall.incorrect),
        ("Invalid outputs", report.overall.invalid_output),
        ("Verification scope", report.verification_scope),
        ("Scientific launch", report.scientific_launch_status),
    )
    if report.evaluated_now is not None:
        rows += (
            ("Evaluated now", report.evaluated_now),
            ("Reused evaluations", report.reused_evaluations),
            ("Retained file bytes", report.retained_file_bytes),
        )
    return rows + tuple(
        (
            item.task_family,
            f"total={item.total}, correct={item.correct}, "
            f"incorrect={item.incorrect}, invalid={item.invalid_output}",
        )
        for item in report.by_family
    )


@study_app.command("evaluate")
def study_evaluate(
    context: typer.Context,
    generation: Annotated[
        Path, typer.Option("--generation", help="Absolute private generation JSON.")
    ],
    benchmark: Annotated[
        Path, typer.Option("--benchmark", help="Absolute matching private benchmark JSON.")
    ],
    request_root: Annotated[
        Path, typer.Option("--request-root", help="Absolute request/checkpoint artifact root.")
    ],
    directory: Annotated[
        Path, typer.Option("--directory", help="Absolute private evaluation run directory.")
    ],
    mode: Annotated[
        StudyEvaluationMode,
        typer.Option("--mode", help="Explicit new run or identical-plan resume."),
    ],
    code_profile: Annotated[
        StudyCodeProfile,
        typer.Option("--code-profile", help="Explicit installed isolated backend."),
    ],
    code_wall_time_ms: Annotated[int, typer.Option("--code-wall-time-ms", min=1, max=600_000)],
    code_max_output_bytes: Annotated[
        int, typer.Option("--code-max-output-bytes", min=1024, max=8 * 1024 * 1024)
    ],
    max_storage_bytes: Annotated[
        int, typer.Option("--max-storage-bytes", min=4096, max=20 * 1024**3)
    ],
) -> None:
    """Score a saved complete split in isolation; no model weights or downloads."""
    report = run_application(
        context,
        lambda: study_evaluation_service.evaluate_study_generation_file_v1(
            generation_path=generation,
            benchmark_path=benchmark,
            request_root=request_root,
            directory=directory,
            code_profile=code_profile.value,
            code_wall_time_ms=code_wall_time_ms,
            code_max_output_bytes=code_max_output_bytes,
            max_storage_bytes=max_storage_bytes,
            resume=mode is StudyEvaluationMode.RESUME,
        ),
    )
    Console().print(
        _private_metadata_table(
            "Complete study evaluation",
            status_label="Evaluation",
            status_value="complete",
            rows=_study_evaluation_rows(report),
        )
    )


@verify_app.command("study-evaluation")
def verify_study_evaluation(
    context: typer.Context,
    generation: Annotated[
        Path, typer.Option("--generation", help="Absolute original private generation JSON.")
    ],
    benchmark: Annotated[
        Path, typer.Option("--benchmark", help="Absolute original private benchmark JSON.")
    ],
    request_root: Annotated[
        Path, typer.Option("--request-root", help="Absolute original request/checkpoint root.")
    ],
    directory: Annotated[
        Path, typer.Option("--directory", help="Absolute complete evaluation run directory.")
    ],
) -> None:
    """Read-only whole-run verification; no runtime start, code execution or model weights."""
    report = run_application(
        context,
        lambda: study_evaluation_service.verify_study_evaluation_files_v1(
            generation_path=generation,
            benchmark_path=benchmark,
            request_root=request_root,
            directory=directory,
        ),
    )
    Console().print(
        _verification_table("Study evaluation verification", _study_evaluation_rows(report))
    )


def _study_generation_rows(
    report: study_generation_service.IrisStudyGenerationFileReportV1,
) -> tuple[tuple[str, object], ...]:
    rows: tuple[tuple[str, object], ...] = (
        ("Directory", report.directory),
        ("Generation path", report.path),
        ("Generation SHA-256", report.generation_payload_sha256),
        ("Run SHA-256", report.run_payload_sha256),
        ("Benchmark SHA-256", report.benchmark_payload_sha256),
        ("Configuration SHA-256", report.configuration_sha256),
        ("Input preflight SHA-256", report.input_preflight_sha256),
        ("Arm", report.arm_id),
        ("Split", report.split),
        ("Total examples", report.total_examples),
        ("Verification scope", report.verification_scope),
        ("Scientific launch", report.scientific_launch_status),
    )
    if report.generated_now is not None:
        rows += (
            ("Generated now", report.generated_now),
            ("Reused answers", report.reused_answers),
            ("Retained file bytes", report.retained_file_bytes),
        )
    return rows + tuple((family, f"total={count}") for family, count in report.by_family)


@study_app.command("generate")
def study_generate(
    context: typer.Context,
    benchmark: Annotated[
        Path, typer.Option("--benchmark", help="Absolute private benchmark JSON.")
    ],
    configuration: Annotated[
        Path,
        typer.Option("--configuration", help="Absolute full canonical offline configuration JSON."),
    ],
    decoding: Annotated[
        Path,
        typer.Option(
            "--decoding", help="Absolute canonical ordered example/family/decoding array."
        ),
    ],
    arm: Annotated[StudyRequestArm, typer.Option("--arm", help="Explicit A/B/C/D execution arm.")],
    split: Annotated[
        StudyGenerationSplit, typer.Option("--split", help="Complete selected split.")
    ],
    terminal_token_id: Annotated[int, typer.Option("--terminal-token-id", min=0)],
    directory: Annotated[
        Path, typer.Option("--directory", help="Absolute private managed generation directory.")
    ],
    mode: Annotated[
        StudyGenerationMode, typer.Option("--mode", help="New or identical-plan resume.")
    ],
    max_artifact_bytes: Annotated[
        int, typer.Option("--max-artifact-bytes", min=1024, max=64 * 1024**2)
    ],
    max_record_bytes: Annotated[
        int, typer.Option("--max-record-bytes", min=4096, max=64 * 1024**2)
    ],
    max_storage_bytes: Annotated[
        int, typer.Option("--max-storage-bytes", min=4096, max=20 * 1024**3)
    ],
    checkpoint_payload_sha256: Annotated[
        str | None,
        typer.Option("--checkpoint-payload-sha256", help="Required for B/C/D, forbidden for A."),
    ] = None,
    checkpoint_state_file_sha256: Annotated[
        str | None, typer.Option("--checkpoint-state-file-sha256")
    ] = None,
    checkpoint_sidecar_state_sha256: Annotated[
        str | None, typer.Option("--checkpoint-sidecar-state-sha256")
    ] = None,
    checkpoint_completed_steps: Annotated[
        int | None, typer.Option("--checkpoint-completed-steps", min=0, max=2**24)
    ] = None,
) -> None:
    """Durably generate one split offline; no grading, code execution or inferred seeds."""
    report = run_application(
        context,
        lambda: study_generation_service.generate_study_split_files_v1(
            benchmark_path=benchmark,
            configuration_path=configuration,
            decoding_path=decoding,
            arm_id=arm.value,
            checkpoint_identity=_request_checkpoint_identity(
                arm=arm,
                payload_sha256=checkpoint_payload_sha256,
                state_file_sha256=checkpoint_state_file_sha256,
                sidecar_state_sha256=checkpoint_sidecar_state_sha256,
                completed_steps=checkpoint_completed_steps,
            ),
            split=split.value,
            terminal_token_id=terminal_token_id,
            directory=directory,
            max_artifact_bytes=max_artifact_bytes,
            max_record_bytes=max_record_bytes,
            max_storage_bytes=max_storage_bytes,
            resume=mode is StudyGenerationMode.RESUME,
        ),
    )
    Console().print(
        _private_metadata_table(
            "Complete study generation",
            status_label="Generation",
            status_value="complete",
            rows=_study_generation_rows(report),
        )
    )


@verify_app.command("study-generation")
def verify_study_generation(
    context: typer.Context,
    benchmark: Annotated[
        Path, typer.Option("--benchmark", help="Absolute original private benchmark JSON.")
    ],
    directory: Annotated[
        Path, typer.Option("--directory", help="Absolute complete managed generation directory.")
    ],
) -> None:
    """Read-only whole-run verification; no model loading, inference, repairs or scoring."""
    report = run_application(
        context,
        lambda: study_generation_service.verify_study_generation_files_v1(
            benchmark_path=benchmark,
            directory=directory,
        ),
    )
    Console().print(
        _verification_table("Study generation verification", _study_generation_rows(report))
    )


@study_app.command("export-manifest")
def study_export_manifest(
    context: typer.Context,
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Canonical task-labelled study dataset JSON path.")
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Explicit output .json; contains ALL private prompts/targets."),
    ],
) -> None:
    """Export the exact train/validation/test manifest for the existing B/C/D runner."""
    report = run_application(
        context,
        lambda: study_dataset_service.export_study_experiment_manifest_v1(
            dataset_path=dataset, manifest_path=path
        ),
    )
    Console().print(
        _private_metadata_table(
            "Study dataset manifest export",
            status_label="Export",
            status_value="reused" if report.reused_existing else "written",
            rows=(
                ("Path", report.path),
                ("Dataset payload SHA-256", report.dataset_payload_sha256),
                ("Manifest payload SHA-256", report.manifest_payload_sha256),
                ("Size bytes", report.size_bytes),
                ("File content", "plaintext private examples from ALL three splits"),
                ("Scientific launch", "not assessed"),
            ),
        )
    )


@verify_app.command("four-arm-protocol")
def verify_four_arm_protocol(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Canonical four-arm protocol declaration JSON path."),
    ],
) -> None:
    """Verify recovered four-arm facts without claiming study readiness."""
    report = run_application(
        context,
        lambda: verification_service.verify_four_arm_protocol_declaration_v1(path),
    )
    Console().print(
        _verification_table(
            "Four-arm protocol declaration verification",
            (
                ("Path", report.path),
                ("Artifact type", report.artifact_type),
                ("Schema version", report.version),
                ("Payload SHA-256", report.payload_sha256),
                ("Protocol ID", report.protocol_id),
                ("Arms", report.arm_count),
                ("Task families", report.task_family_count),
                ("Unresolved protocol items", report.unresolved_item_count),
                ("Main backbone", report.main_backbone_repository),
                ("Debug backbone", report.debug_backbone_repository),
                ("Execution status", "blocked"),
                (
                    "Execution blocker",
                    "A/B/C profiles and evaluation protocol required",
                ),
            ),
        )
    )


@verify_app.command("journal")
def verify_journal(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Canonical experiment run directory."),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Exact experiment manifest referenced by the journal.",
        ),
    ],
) -> None:
    """Verify a v1, v2, or profile-v4 v3 journal and all references."""
    if path.parent.name == "v3":
        report_v3 = run_application(
            context,
            lambda: verification_v3_service.verify_experiment_journal_v3(
                path=path,
                manifest_path=manifest,
            ),
        )
        Console().print(
            _verification_table(
                "Profile-v4 experiment journal verification",
                (
                    ("Path", report_v3.path),
                    ("Artifact type", report_v3.artifact_type),
                    ("Schema version", report_v3.version),
                    ("Run ID", report_v3.run_id),
                    ("Arm", report_v3.arm_id),
                    ("Arm kind", report_v3.arm_kind),
                    ("Request graph", report_v3.request_graph),
                    ("Teacher graph", report_v3.teacher_graph),
                    ("Execution profile version", report_v3.execution_profile_set_version),
                    ("Execution profile SHA-256", report_v3.execution_profile_set_sha256),
                    ("Shared-buffer policy SHA-256", report_v3.shared_buffer_policy_sha256),
                    ("Run payload SHA-256", report_v3.run_payload_sha256),
                    ("Configuration SHA-256", report_v3.configuration_sha256),
                    ("Manifest payload SHA-256", report_v3.manifest_payload_sha256),
                    ("Execution mode", report_v3.execution_mode),
                    ("Events", report_v3.event_count),
                    ("Training step events", report_v3.training_step_event_count),
                    ("Evaluation events", report_v3.evaluation_event_count),
                    ("Checkpoint events", report_v3.checkpoint_event_count),
                    ("Completed optimizer steps", report_v3.optimizer_completed_steps),
                    ("Sidecar state SHA-256", report_v3.sidecar_state_sha256),
                    ("Chain head SHA-256", report_v3.chain_head_sha256),
                    ("Checkpoint references", report_v3.checkpoint_reference_count),
                    ("Unique checkpoints", report_v3.unique_checkpoint_count),
                    (
                        "Latest checkpoint schema version",
                        report_v3.latest_checkpoint_schema_version,
                    ),
                    (
                        "Latest checkpoint SHA-256",
                        report_v3.latest_checkpoint_payload_sha256,
                    ),
                    ("Managed phase", report_v3.managed_phase),
                    ("Completed epochs", report_v3.completed_epochs),
                    ("Active epoch", report_v3.active_epoch),
                    ("Next training ordinal", report_v3.next_training_ordinal),
                ),
            )
        )
        return
    if path.parent.name == "v2":
        report_v2 = run_application(
            context,
            lambda: verification_v2_service.verify_experiment_journal_v2(
                path=path,
                manifest_path=manifest,
            ),
        )
        Console().print(
            _verification_table(
                "Arm-labelled experiment journal verification",
                (
                    ("Path", report_v2.path),
                    ("Artifact type", report_v2.artifact_type),
                    ("Schema version", report_v2.version),
                    ("Run ID", report_v2.run_id),
                    ("Arm", report_v2.arm_id),
                    ("Arm kind", report_v2.arm_kind),
                    ("Request graph", report_v2.request_graph),
                    ("Teacher graph", report_v2.teacher_graph),
                    ("Execution profile version", report_v2.execution_profile_set_version),
                    ("Execution profile SHA-256", report_v2.execution_profile_set_sha256),
                    ("Run payload SHA-256", report_v2.run_payload_sha256),
                    ("Configuration SHA-256", report_v2.configuration_sha256),
                    ("Manifest payload SHA-256", report_v2.manifest_payload_sha256),
                    ("Execution mode", report_v2.execution_mode),
                    ("Events", report_v2.event_count),
                    ("Training step events", report_v2.training_step_event_count),
                    ("Evaluation events", report_v2.evaluation_event_count),
                    ("Checkpoint events", report_v2.checkpoint_event_count),
                    ("Completed optimizer steps", report_v2.optimizer_completed_steps),
                    ("Sidecar state SHA-256", report_v2.sidecar_state_sha256),
                    ("Chain head SHA-256", report_v2.chain_head_sha256),
                    ("Checkpoint references", report_v2.checkpoint_reference_count),
                    ("Unique checkpoints", report_v2.unique_checkpoint_count),
                    (
                        "Latest checkpoint schema version",
                        report_v2.latest_checkpoint_schema_version,
                    ),
                    (
                        "Latest checkpoint SHA-256",
                        report_v2.latest_checkpoint_payload_sha256,
                    ),
                    ("Managed phase", report_v2.managed_phase),
                    ("Completed epochs", report_v2.completed_epochs),
                    ("Active epoch", report_v2.active_epoch),
                    ("Next training ordinal", report_v2.next_training_ordinal),
                ),
            )
        )
        return
    report = run_application(
        context,
        lambda: verification_service.verify_experiment_journal_v1(
            path=path,
            manifest_path=manifest,
        ),
    )
    Console().print(
        _verification_table(
            "Experiment journal verification",
            (
                ("Path", report.path),
                ("Artifact type", report.artifact_type),
                ("Schema version", report.version),
                ("Run ID", report.run_id),
                ("Run payload SHA-256", report.run_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Manifest payload SHA-256", report.manifest_payload_sha256),
                ("Execution mode", report.execution_mode),
                ("Events", report.event_count),
                ("Training step events", report.training_step_event_count),
                ("Evaluation events", report.evaluation_event_count),
                ("Checkpoint events", report.checkpoint_event_count),
                ("Completed optimizer steps", report.optimizer_completed_steps),
                ("Sidecar state SHA-256", report.sidecar_state_sha256),
                ("Chain head SHA-256", report.chain_head_sha256),
                ("Checkpoint references", report.checkpoint_reference_count),
                ("Unique checkpoints", report.unique_checkpoint_count),
                ("Latest checkpoint SHA-256", report.latest_checkpoint_payload_sha256),
                ("Managed phase", report.managed_phase),
                ("Completed epochs", report.completed_epochs),
                ("Active epoch", report.active_epoch),
                ("Next training ordinal", report.next_training_ordinal),
            ),
        )
    )


@experiment_app.command("project-mlflow")
def experiment_project_mlflow(
    context: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", help="Canonical experiment run directory."),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Exact experiment manifest referenced by the journal.",
        ),
    ],
    tracking_uri: Annotated[
        str,
        typer.Option(
            "--tracking-uri",
            help="Explicit MLflow tracking URI; never read from ambient process state.",
        ),
    ],
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Explicit MLflow experiment name.",
        ),
    ],
    artifact_location: Annotated[
        str,
        typer.Option(
            "--artifact-location",
            help="Explicit MLflow experiment artifact URI; no journal files are uploaded.",
        ),
    ],
) -> None:
    """Project a verified canonical journal into one reconciled MLflow run."""
    report = run_application(
        context,
        lambda: mlflow_projection_service.project_experiment_journal_to_mlflow_v1(
            path=path,
            manifest_path=manifest,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_location=artifact_location,
        ),
    )
    Console().print(
        _private_metadata_table(
            "MLflow experiment projection",
            status_label="Projection",
            status_value="reconciled",
            rows=(
                ("Path", report.path),
                ("Experiment name", report.experiment_name),
                ("Experiment ID", report.experiment_id),
                ("MLflow run ID", report.mlflow_run_id),
                ("Canonical run ID", report.canonical_run_id),
                ("Run payload SHA-256", report.run_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Manifest payload SHA-256", report.manifest_payload_sha256),
                ("Events", report.event_count),
                ("Chain head SHA-256", report.chain_head_sha256),
                ("Metric points", report.metric_point_count),
                ("New metric points", report.new_metric_point_count),
                ("Reused metric points", report.reused_metric_point_count),
                (
                    "Reused experiment",
                    str(report.reused_existing_experiment).lower(),
                ),
                ("Reused run", str(report.reused_existing_run).lower()),
                ("Source status", report.status),
            ),
        )
    )


@experiment_app.command("run")
def experiment_run(
    context: typer.Context,
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical experiment manifest JSON path."),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Explicit root for canonical run and checkpoint artifacts."),
    ],
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Portable identity for one managed experiment run."),
    ],
    terminal_token_id: Annotated[
        int,
        typer.Option(
            "--terminal-token-id",
            help="Explicit terminal token appended to every teacher target.",
        ),
    ],
    initialization_seed: Annotated[
        int,
        typer.Option(
            "--initialization-seed",
            help="Explicit sidecar initialization and deterministic runtime seed.",
        ),
    ],
    epoch_count: Annotated[
        int,
        typer.Option("--epoch-count", help="Positive fixed number of training epochs."),
    ],
    evaluate_test_after_training: Annotated[
        bool,
        typer.Option(
            "--evaluate-test/--skip-test",
            help="Explicitly enable or skip one final complete test-split evaluation.",
        ),
    ],
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", help="Explicit AdamW learning rate."),
    ],
    beta1: Annotated[
        float,
        typer.Option("--beta1", help="Explicit AdamW beta1 coefficient."),
    ],
    beta2: Annotated[
        float,
        typer.Option("--beta2", help="Explicit AdamW beta2 coefficient."),
    ],
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", help="Explicit AdamW numerical epsilon."),
    ],
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", help="Explicit AdamW decoupled weight decay."),
    ],
    max_gradient_norm: Annotated[
        float,
        typer.Option(
            "--max-gradient-norm",
            help="Explicit global sidecar gradient-clipping norm.",
        ),
    ],
    maximum_new_training_steps: Annotated[
        int | None,
        typer.Option(
            "--maximum-new-training-steps",
            help="Operational limit on newly committed training steps for this invocation.",
        ),
    ] = None,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned model to be downloaded when absent from the local cache.",
        ),
    ] = False,
) -> None:
    """Create or exactly resume one deterministic managed experiment run."""

    def run_experiment() -> IrisExperimentExecutionResult:
        optimizer_config = IrisAdamWConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            max_gradient_norm=max_gradient_norm,
        )
        return experiment_command_service.run_packaged_experiment_v1(
            manifest_path=manifest,
            root=root,
            run_id=run_id,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            epoch_count=epoch_count,
            evaluate_test_after_training=evaluate_test_after_training,
            optimizer_config=optimizer_config,
            maximum_new_training_steps=maximum_new_training_steps,
            allow_download=allow_download,
        )

    report = run_application(context, run_experiment)
    Console().print(
        _private_metadata_table(
            "Experiment execution",
            status_label="Execution",
            status_value=report.status,
            rows=(
                ("Path", report.path),
                ("Run payload SHA-256", report.run_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Events", report.event_count),
                ("Optimizer completed steps", report.optimizer_completed_steps),
                ("Completed epochs", report.completed_epochs),
                ("Next epoch index", report.next_epoch_index),
                ("Next training ordinal", report.next_training_ordinal),
                (
                    "Latest checkpoint SHA-256",
                    report.latest_checkpoint_payload_sha256,
                ),
                ("Replayed training steps", report.replayed_training_steps),
                ("New training steps", report.new_training_steps),
            ),
        )
    )


@experiment_app.command("run-arm-v3")
def experiment_run_arm_v3(
    context: typer.Context,
    arm: Annotated[
        StudyExperimentArm,
        typer.Option("--arm", case_sensitive=True, help="Exact study arm label A, B, C, or D."),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical experiment manifest JSON path."),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Explicit root for canonical v3 run and checkpoints."),
    ],
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Portable identity for one arm-labelled managed run."),
    ],
    terminal_token_id: Annotated[
        int,
        typer.Option(
            "--terminal-token-id",
            help="Explicit terminal token appended to every teacher target.",
        ),
    ],
    initialization_seed: Annotated[
        int,
        typer.Option(
            "--initialization-seed",
            help="Explicit sidecar initialization and deterministic runtime seed.",
        ),
    ],
    epoch_count: Annotated[
        int,
        typer.Option("--epoch-count", help="Positive fixed number of training epochs."),
    ],
    evaluate_test_after_training: Annotated[
        bool,
        typer.Option(
            "--evaluate-test/--skip-test",
            help="Explicitly enable or skip one final complete test-split evaluation.",
        ),
    ],
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", help="Explicit AdamW learning rate."),
    ],
    beta1: Annotated[
        float,
        typer.Option("--beta1", help="Explicit AdamW beta1 coefficient."),
    ],
    beta2: Annotated[
        float,
        typer.Option("--beta2", help="Explicit AdamW beta2 coefficient."),
    ],
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", help="Explicit AdamW numerical epsilon."),
    ],
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", help="Explicit AdamW decoupled weight decay."),
    ],
    max_gradient_norm: Annotated[
        float,
        typer.Option(
            "--max-gradient-norm",
            help="Explicit global sidecar gradient-clipping norm.",
        ),
    ],
    maximum_new_training_steps: Annotated[
        int | None,
        typer.Option(
            "--maximum-new-training-steps",
            help="Operational limit on newly committed training steps for this invocation.",
        ),
    ] = None,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned model to be downloaded when absent from the local cache.",
        ),
    ] = False,
) -> None:
    """Create or exactly resume one deterministic profile-v4 B/C/D run."""

    def run_experiment() -> IrisExperimentExecutionResultV3:
        profile = arm_experiment_command_v3_service.require_arm_experiment_command_profile_v3(
            arm.value
        )
        optimizer_config = IrisAdamWConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            max_gradient_norm=max_gradient_norm,
        )
        return arm_experiment_command_v3_service.run_packaged_arm_experiment_v3(
            arm_id=profile.arm_id,
            manifest_path=manifest,
            root=root,
            run_id=run_id,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            epoch_count=epoch_count,
            evaluate_test_after_training=evaluate_test_after_training,
            optimizer_config=optimizer_config,
            maximum_new_training_steps=maximum_new_training_steps,
            allow_download=allow_download,
        )

    report = run_application(context, run_experiment)
    Console().print(
        _private_metadata_table(
            "Profile-v4 B/C/D experiment execution",
            status_label="Execution",
            status_value=report.status,
            rows=(
                ("Arm", report.arm_id),
                ("Path", report.path),
                ("Run payload SHA-256", report.run_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Events", report.event_count),
                ("Optimizer completed steps", report.optimizer_completed_steps),
                ("Completed epochs", report.completed_epochs),
                ("Next epoch index", report.next_epoch_index),
                ("Next training ordinal", report.next_training_ordinal),
                ("Latest checkpoint schema", report.latest_checkpoint_schema_version),
                (
                    "Latest checkpoint SHA-256",
                    report.latest_checkpoint_payload_sha256,
                ),
                ("Replayed training steps", report.replayed_training_steps),
                ("New training steps", report.new_training_steps),
            ),
        )
    )


@experiment_app.command("run-arm")
def experiment_run_arm(
    context: typer.Context,
    arm: Annotated[
        StudyExperimentArm,
        typer.Option("--arm", case_sensitive=True, help="Exact study arm label A, B, C, or D."),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical experiment manifest JSON path."),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Explicit root for canonical v2 run and checkpoints."),
    ],
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Portable identity for one arm-labelled managed run."),
    ],
    terminal_token_id: Annotated[
        int,
        typer.Option(
            "--terminal-token-id",
            help="Explicit terminal token appended to every teacher target.",
        ),
    ],
    initialization_seed: Annotated[
        int,
        typer.Option(
            "--initialization-seed",
            help="Explicit sidecar initialization and deterministic runtime seed.",
        ),
    ],
    epoch_count: Annotated[
        int,
        typer.Option("--epoch-count", help="Positive fixed number of training epochs."),
    ],
    evaluate_test_after_training: Annotated[
        bool,
        typer.Option(
            "--evaluate-test/--skip-test",
            help="Explicitly enable or skip one final complete test-split evaluation.",
        ),
    ],
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", help="Explicit AdamW learning rate."),
    ],
    beta1: Annotated[
        float,
        typer.Option("--beta1", help="Explicit AdamW beta1 coefficient."),
    ],
    beta2: Annotated[
        float,
        typer.Option("--beta2", help="Explicit AdamW beta2 coefficient."),
    ],
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", help="Explicit AdamW numerical epsilon."),
    ],
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", help="Explicit AdamW decoupled weight decay."),
    ],
    max_gradient_norm: Annotated[
        float,
        typer.Option(
            "--max-gradient-norm",
            help="Explicit global sidecar gradient-clipping norm.",
        ),
    ],
    maximum_new_training_steps: Annotated[
        int | None,
        typer.Option(
            "--maximum-new-training-steps",
            help="Operational limit on newly committed training steps for this invocation.",
        ),
    ] = None,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned model to be downloaded when absent from the local cache.",
        ),
    ] = False,
) -> None:
    """Create or exactly resume one deterministic B/D arm-labelled run."""

    def run_experiment() -> IrisExperimentExecutionResultV2:
        profile = arm_experiment_command_service.require_arm_experiment_command_profile_v2(
            arm.value
        )
        optimizer_config = IrisAdamWConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            max_gradient_norm=max_gradient_norm,
        )
        return arm_experiment_command_service.run_packaged_arm_experiment_v2(
            arm_id=profile.arm_id,
            manifest_path=manifest,
            root=root,
            run_id=run_id,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            epoch_count=epoch_count,
            evaluate_test_after_training=evaluate_test_after_training,
            optimizer_config=optimizer_config,
            maximum_new_training_steps=maximum_new_training_steps,
            allow_download=allow_download,
        )

    report = run_application(context, run_experiment)
    Console().print(
        _private_metadata_table(
            "Arm-labelled experiment execution",
            status_label="Execution",
            status_value=report.status,
            rows=(
                ("Arm", report.arm_id),
                ("Path", report.path),
                ("Run payload SHA-256", report.run_payload_sha256),
                ("Configuration SHA-256", report.configuration_sha256),
                ("Events", report.event_count),
                ("Optimizer completed steps", report.optimizer_completed_steps),
                ("Completed epochs", report.completed_epochs),
                ("Next epoch index", report.next_epoch_index),
                ("Next training ordinal", report.next_training_ordinal),
                ("Latest checkpoint schema", report.latest_checkpoint_schema_version),
                (
                    "Latest checkpoint SHA-256",
                    report.latest_checkpoint_payload_sha256,
                ),
                ("Replayed training steps", report.replayed_training_steps),
                ("New training steps", report.new_training_steps),
            ),
        )
    )
