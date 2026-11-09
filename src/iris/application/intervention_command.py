"""Source-style intervention command: declared context, exact best, private resumable stages."""

import json
from dataclasses import dataclass
from pathlib import Path
from re import fullmatch
from typing import Literal, cast

from pydantic import Field, field_validator

from iris.application.errors import ApplicationError, IrisCodeEvaluationUnavailableError
from iris.application.scientific_answer_command import (
    _destination,
    _recipe_inputs,
    read_scientific_test_input_v1,
)
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _regular,
    _unique,
    read_scientific_checkpoint_v1,
)
from iris.application.scientific_command import (
    _input_bytes,
    _local_path,
    _rows,
    read_scientific_command_config_v1,
)
from iris.application.scientific_delivery import _checked as _check_delivery
from iris.application.scientific_delivery import _read_delivery_directory
from iris.application.scientific_execution import (
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_generation import generate_scientific_intervention_v1
from iris.application.scientific_intervention import (
    read_scientific_intervention_v1,
    write_scientific_intervention_v1,
)
from iris.application.scientific_intervention_evaluation import (
    evaluate_scientific_intervention_v1,
    read_scientific_intervention_evaluation_v1,
    write_scientific_intervention_evaluation_v1,
)
from iris.application.scientific_intervention_nll import (
    measure_scientific_intervention_nll_v1,
    read_scientific_intervention_nll_v1,
    write_scientific_intervention_nll_v1,
)
from iris.application.scientific_lifecycle import _bundle_parts
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_execution_v1 import ScientificExecutionCheckpointV1
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.artifacts.scientific_intervention_evaluation_v1 import (
    IrisScientificInterventionEvaluationV1,
)
from iris.artifacts.scientific_intervention_nll_v1 import IrisScientificInterventionNllV1
from iris.interventions import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_bytes


class IrisInterventionCommandError(ApplicationError):
    """Redacted intervention argument/context/source/publication failure."""


class IrisInterventionScoringUnavailableError(IrisCodeEvaluationUnavailableError):
    """A capture was preserved; no successful quality report is implied."""

    def __init__(self, capture_path: Path) -> None:
        self.capture_path = capture_path
        super().__init__(
            "Intervention scoring infrastructure is unavailable; no complete report. "
            f"Saved capture retained for mode=evaluate: {capture_path}"
        )


class ScientificInterventionCommandContextV1(_Strict):
    """Paths are relative to THIS JSON file, never environment variables or cwd guesses."""

    kind: Literal["iris-intervention-command-context-v1"] = "iris-intervention-command-context-v1"
    checkpoint_sha256: Digest
    config: str = Field(min_length=1, max_length=4096)
    execution: str = Field(min_length=1, max_length=4096)
    suites: dict[str, str] = Field(min_length=1, max_length=16)
    output_root: str = Field(min_length=1, max_length=4096)
    decoding: ScientificDecodingPolicyV1
    operator_seed: int = Field(ge=0, le=2**63 - 1)
    python_policy: PythonSandboxPolicyV2 | None
    maximum_storage_bytes: int = Field(default=256 * 1024**2, ge=1, le=16 * 1024**3)

    @field_validator("config", "execution", "output_root")
    @classmethod
    def local_paths(cls, value: str) -> str:
        _path(Path.cwd(), value)
        return value

    @field_validator("suites")
    @classmethod
    def declared_suites(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None or not 1 <= len(path) <= 4096
            for key, path in value.items()
        ):
            raise ValueError("Explicit bounded suite names and descriptor paths required")
        for path in value.values():
            _path(Path.cwd(), path)
        return value


@dataclass(frozen=True, slots=True)
class IrisInterventionCommandResultV1:
    mode: Literal["run", "generate", "evaluate", "inspect", "measure-nll", "inspect-nll"]
    capture_path: Path | None
    report_path: Path | None
    report: IrisScientificInterventionEvaluationV1 | None
    capture_sha256: str
    plan_sha256: str
    selected_step: int
    examples: int
    nll_report: IrisScientificInterventionNllV1 | None = None


_PARAMETERS = {
    "slot_id": "slot",
    "other_slot_id": "other_slot",
    "module_id": "module",
    "seed": "seed",
    "sigma": "sigma",
    "factor": "factor",
    "capacity": "capacity",
    "values": "values",
    "source_module": "source_module",
    "confidence": "confidence",
    "ttl": "ttl",
    "cycles": "cycles",
}


def _arguments(arguments: tuple[str, ...]) -> dict[str, str]:
    if type(arguments) is not tuple or not 1 <= len(arguments) <= 32:
        raise ValueError("Explicit bounded key=value arguments required")
    values: dict[str, str] = {}
    for argument in arguments:
        if type(argument) is not str or len(argument) > 65536 or "\0" in argument:
            raise ValueError("Invalid bounded argument")
        key, separator, value = argument.partition("=")
        if not separator or not value or key in values:
            raise ValueError("Every key must occur once with an explicit value")
        values[key] = value
    mode = values.get("mode", "run")
    if mode not in {"run", "generate", "evaluate", "inspect", "measure-nll", "inspect-nll"}:
        raise ValueError("Unknown intervention mode")
    if mode in {"inspect", "inspect-nll"}:
        allowed, required = {"mode", "report"}, {"report"}
    elif mode in {"evaluate", "measure-nll"}:
        allowed, required = (
            {"mode", "checkpoint", "context", "suite", "capture"},
            {"checkpoint", "capture"},
        )
    else:
        allowed = {"mode", "checkpoint", "context", "suite", "intervention"} | {
            f"intervention.{name}" for name in _PARAMETERS
        }
        required = {"checkpoint", "intervention"}
    if not required <= values.keys() or not values.keys() <= allowed:
        raise ValueError("Missing, unknown or inapplicable intervention argument")
    return values


def _read_context(path: Path) -> tuple[ScientificInterventionCommandContextV1, bytes]:
    content = _input_bytes(path, 128 * 1024)
    raw = json.loads(content.decode("utf-8"), object_pairs_hook=_unique)
    context = ScientificInterventionCommandContextV1.model_validate_json(canonical_json_bytes(raw))
    return context, content


def _path(base: Path, value: str) -> Path:
    # Validate before joining: a UNC/drive-relative value must not be disguised by its base.
    if not value or value != value.strip() or "\0" in value or "://" in value:
        raise ValueError("Explicit local path required")
    _local_path(Path(value))
    return _local_path(base / value)


def _plan(values: dict[str, str], *, seed: int) -> WorkspaceInterventionPlanV1:
    raw: dict[str, object] = {"kind": values["intervention"]}
    for name, field in _PARAMETERS.items():
        key = f"intervention.{name}"
        if key in values:
            raw[field] = json.loads(values[key])
            if raw[field] is None and field != "cycles":
                raise ValueError("Explicit intervention parameters cannot be null")
    if raw["kind"] in {"shuffle_slots", "gaussian_noise"} and "seed" not in raw:
        raw["seed"] = seed
    spec = WorkspaceInterventionSpecV1.model_validate_json(canonical_json_bytes(raw))
    return WorkspaceInterventionPlanV1(interventions=(spec,))


def _checkpoint(path: Path) -> IrisScientificCheckpointBundleV1:
    path = _local_path(path)
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    names = {file.name for file in path.iterdir()}
    if names == {"manifest.json", "state.safetensors"}:
        return read_scientific_checkpoint_v1(path)
    if "schema_version.txt" in names:
        # Source examples use checkpoints/igw-best. The CLI may read a named copy,
        # but checks all nine canonical files and exact equality with journal best.
        checkpoint, _ = _check_delivery(
            _read_delivery_directory(path, require_content_address=False)
        )
        return checkpoint
    raise ValueError("An explicit scientific checkpoint or nine-file delivery is required")


def _assert_best(
    checkpoint_path: Path, execution: Path, expected: str
) -> IrisScientificCheckpointBundleV1:
    supplied = _checkpoint(checkpoint_path)
    snapshot = read_scientific_execution_v1(execution)
    if not snapshot.events:
        raise ValueError("Terminal scientific execution required")
    terminal = snapshot.events[-1]
    if (
        terminal.payload.after.phase != "stopped"
        or type(terminal.payload.record) is not ScientificExecutionCheckpointV1
    ):
        raise ValueError("Terminal checkpoint required")
    selection = None
    for event in snapshot.events:
        selection = _selected_best(selection, event)
    _, best = _bundle_parts(
        _checkpoint_for_event(snapshot.path, snapshot.header, terminal, best_event=selection)
    )
    if (
        best is None
        or selection is None
        or supplied != best
        or supplied.manifest.sha256 != expected
    ):
        raise ValueError("Explicit checkpoint must be the exact validation-selected best")
    return supplied


def run_intervention_command_v1(arguments: tuple[str, ...]) -> IrisInterventionCommandResultV1:
    """Run all declared suite rows, or score a saved pair without loading a model.

    Defaults: mode=run, suite=core; context=<checkpoint-name>.intervention.json
    beside the checkpoint. Context supplies explicit data paths, seed, policy and
    storage limits. Never silently substitute a current/random checkpoint, a suite,
    a sandbox profile, or a smaller dataset. All twelve source operations are accepted.
    measure-nll restores the selected model for saved-pair target NLL; inspect-nll
    verifies only that companion report. Original evaluate remains model-free.
    """
    capture_path: Path | None = None
    report_path: Path | None = None
    report: IrisScientificInterventionEvaluationV1 | None = None
    nll_report: IrisScientificInterventionNllV1 | None = None
    try:
        values = _arguments(arguments)
        mode = values.get("mode", "run")
        if mode == "inspect-nll":
            report_path = _path(Path.cwd(), values["report"])
            nll_report = read_scientific_intervention_nll_v1(report_path)
            measured = nll_report.payload
            return IrisInterventionCommandResultV1(
                "inspect-nll",
                None,
                report_path,
                None,
                measured.intervention_sha256,
                measured.plan_sha256,
                measured.selected_step,
                len(measured.rows),
                nll_report,
            )
        if mode == "inspect":
            report_path = _path(Path.cwd(), values["report"])
            report = read_scientific_intervention_evaluation_v1(report_path)
            payload = report.payload
            return IrisInterventionCommandResultV1(
                "inspect",
                None,
                report_path,
                report,
                payload.intervention_sha256,
                payload.plan_sha256,
                payload.selected_step,
                payload.overall.control.examples,
            )
        checkpoint_path = _path(Path.cwd(), values["checkpoint"])
        context_path = (
            _path(Path.cwd(), values["context"])
            if "context" in values
            else checkpoint_path.with_name(checkpoint_path.name + ".intervention.json")
        )
        context, context_bytes = _read_context(context_path)
        suite = values.get("suite", "core")
        if suite not in context.suites:
            raise ValueError("Requested suite is not declared")
        base = context_path.parent
        config_path = _path(base, context.config)
        execution = _path(base, context.execution)
        test_input = _path(base, context.suites[suite])
        output_root = _path(base, context.output_root)
        test = read_scientific_test_input_v1(test_input)
        config_bytes = _input_bytes(config_path, 128 * 1024)
        config = read_scientific_command_config_v1(config_path)
        checkpoint = _assert_best(checkpoint_path, execution, context.checkpoint_sha256)
        test_rows = _rows(test, config_path.parent)
        has_code = any(row.executable for row in test_rows)
        policy = context.python_policy if has_code else None
        if (
            mode in {"run", "evaluate"}
            and has_code
            and (policy is None or policy.wall_time_ms != 5000)
        ):
            raise ValueError("Repair evaluation requires a declared source five-second policy")
        protected = (
            context_path,
            checkpoint_path,
            execution,
            test_input,
            Path(test.path),
            *_recipe_inputs(config_path),
        )
        if mode in {"evaluate", "measure-nll"}:
            capture_path = _path(Path.cwd(), values["capture"])
            protected = (*protected, capture_path)
        kinds = (
            ("scientific-intervention-nll-v1",)
            if mode == "measure-nll"
            else ("scientific-intervention-evaluation-v1",)
            if mode == "evaluate"
            else ("scientific-intervention-v1",)
            if mode == "generate"
            else ("scientific-intervention-v1", "scientific-intervention-evaluation-v1")
        )
        for kind in kinds:
            _destination(output_root, kind, context.maximum_storage_bytes, protected)
        if mode in {"run", "generate"}:
            selected = _plan(values, seed=context.operator_seed)
            selected.validate_dimensions(
                num_modules=config.cognitive.num_modules,
                num_slots=config.workspace.num_slots,
                state_dim=config.cognitive.state_dim,
                cycles=config.workspace.cycles,
                winners_per_cycle=config.workspace.winners_per_cycle,
            )
            capture = generate_scientific_intervention_v1(
                config_path=config_path,
                execution_path=execution,
                test=test,
                decoding=context.decoding,
                plan=selected,
            )
            if (
                capture.payload.baseline.payload.best_checkpoint_sha256
                != checkpoint.manifest.sha256
            ):
                raise ValueError("Generated pair selected a different checkpoint")
            if (
                _read_context(context_path) != (context, context_bytes)
                or _input_bytes(config_path, 128 * 1024) != config_bytes
                or read_scientific_test_input_v1(test_input) != test
                or _assert_best(checkpoint_path, execution, context.checkpoint_sha256) != checkpoint
            ):
                raise ValueError("Intervention inputs changed before capture publication")
            capture_path = write_scientific_intervention_v1(
                capture, root=output_root, maximum_storage_bytes=context.maximum_storage_bytes
            )
        else:
            assert capture_path is not None
            capture = read_scientific_intervention_v1(capture_path)
            if (
                capture.payload.baseline.payload.best_checkpoint_sha256
                != checkpoint.manifest.sha256
                or capture.payload.baseline.payload.decoding != context.decoding
            ):
                raise ValueError("Saved pair differs from the declared checkpoint/decoding")
        assert capture_path is not None
        report = None
        report_path = None
        if mode == "measure-nll":
            nll_report = measure_scientific_intervention_nll_v1(
                intervention_path=capture_path,
                execution_path=execution,
                config_path=config_path,
                test=test,
            )
            if nll_report.payload.intervention_sha256 != capture.sha256:
                raise ValueError("NLL companion binds another capture")
        elif mode != "generate":
            report = evaluate_scientific_intervention_v1(
                intervention_path=capture_path,
                execution_path=execution,
                config_path=config_path,
                test=test,
                python_policy=policy,
            )
        if (
            _read_context(context_path) != (context, context_bytes)
            or _input_bytes(config_path, 128 * 1024) != config_bytes
            or read_scientific_test_input_v1(test_input) != test
            or _assert_best(checkpoint_path, execution, context.checkpoint_sha256) != checkpoint
        ):
            raise ValueError("Intervention source/context changed before report publication")
        if report is not None:
            report_path = write_scientific_intervention_evaluation_v1(
                report, root=output_root, maximum_storage_bytes=context.maximum_storage_bytes
            )
        if nll_report is not None:
            report_path = write_scientific_intervention_nll_v1(
                nll_report,
                root=output_root,
                maximum_storage_bytes=context.maximum_storage_bytes,
            )
        if mode not in {"run", "generate", "evaluate", "measure-nll"}:
            raise ValueError("Unknown mode")
        return IrisInterventionCommandResultV1(
            cast(Literal["run", "generate", "evaluate", "measure-nll"], mode),
            capture_path,
            report_path,
            report,
            capture.sha256,
            capture.payload.plan_sha256,
            checkpoint.manifest.payload.completed_steps,
            len(capture.payload.rows),
            nll_report,
        )
    except IrisCodeEvaluationUnavailableError:
        if capture_path is not None:
            raise IrisInterventionScoringUnavailableError(capture_path) from None
        raise IrisCodeEvaluationUnavailableError(
            "Intervention scoring infrastructure is unavailable"
        ) from None
    except Exception:
        raise IrisInterventionCommandError(
            "Intervention command did not complete; verify key=value arguments, declared context, "
            "selected-best checkpoint, suite and explicit Python policy."
        ) from None


__all__ = [
    "IrisInterventionCommandError",
    "IrisInterventionCommandResultV1",
    "IrisInterventionScoringUnavailableError",
    "ScientificInterventionCommandContextV1",
    "run_intervention_command_v1",
]
