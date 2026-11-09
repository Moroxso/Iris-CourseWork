"""Explicit ablation operator commands; saved intents precede every native phase.

This namespace does not reinterpret baseline scientific journals. One train/
validate/checkpoint/resume command acknowledges at most the exact saved phase;
it never silently chooses a new head after a lost reply. No quality/launch claim.
"""

import os
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from iris.application import scientific_ablation_operator as operator
from iris.application import scientific_ablation_operator_export as export
from iris.application.scientific_ablation_command_config import (
    read_scientific_ablation_command_config_v1,
)
from iris.application.scientific_ablation_execution import _local, _stamp
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _bounded_read, _regular
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_operator_export_v1 import Selection
from iris.artifacts.scientific_ablation_operator_v1 import Phase
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class PhaseOption(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"


class SelectionOption(StrEnum):
    CURRENT = "current"
    SELECTED_BEST = "selected_best"


def _phase(value: PhaseOption) -> Phase:
    if value is PhaseOption.TRAIN:
        return "train"
    if value is PhaseOption.VALIDATION:
        return "validation"
    return "checkpoint"


def _selection(value: SelectionOption) -> Selection:
    return "current" if value is SelectionOption.CURRENT else "selected_best"


def _guard(action: Callable[[], None]) -> None:
    try:
        action()
    except BaseException as error:
        if _has_signal(error):
            raise
        try:
            typer.echo(
                "Ablation command failed. Preserve files; inspect or retry the same saved intent. "
                "No scientific acceptance is implied.",
                err=True,
            )
        except BaseException as cleanup:
            if _has_signal(cleanup):
                raise
        raise typer.Exit(code=2) from None


def _emit(value: dict[str, object]) -> None:
    typer.echo(canonical_json_bytes(value).decode("utf-8"))


def _summary(snapshot: operator.IrisScientificAblationOperatorSnapshotV1) -> dict[str, object]:
    head = snapshot.execution.events[-1] if snapshot.execution.events else None
    return dict(
        kind="iris-scientific-ablation-operator-command-summary-v1",
        operator_path=str(snapshot.path),
        operator_sha256=snapshot.descriptor.sha256,
        execution_sha256=snapshot.execution.header.sha256,
        head_sha256=snapshot.execution.header.sha256 if head is None else head.sha256,
        acknowledged_phases=len(snapshot.execution.events),
        step=0 if head is None else head.payload.after.step,
        phase="train" if head is None else head.payload.after.phase,
        best_step=None if head is None else head.payload.after.best_step,
        scientific_acceptance=False,
    )


def _persist(path: Path, content: bytes, *, operator_path: Path) -> None:
    """Exclusive durable intent file, exact DATA retry, never overwrite old/partial bytes."""
    path, source = _local(path), _local(operator_path)
    if path.is_relative_to(source) or len(content) > 64 * 1024:
        raise ValueError("Bounded intent must be saved outside the source operator")
    for parent in (path.parent, *path.parent.parents):
        with suppress(FileNotFoundError):
            _regular(parent, directory=True)
    if os.path.lexists(path):
        pin = _stamp(path)
        if _bounded_read(path, 64 * 1024) != content or _stamp(path) != pin:
            raise ValueError("Existing intent differs; preserve it")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pin = _stamp(path.parent, directory=True)
    _write_exact_file(path, content)
    if _stamp(path.parent, directory=True) != pin or _bounded_read(path, 64 * 1024) != content:
        raise ValueError("Intent publication changed; native execution was not requested")


@app.command("new")
def new(
    recipe: Annotated[Path, typer.Option("--recipe")],
    base: Annotated[Path, typer.Option("--base", help="Explicit recipe corpus base directory.")],
    root: Annotated[Path, typer.Option("--root")],
    maximum_parameter_bytes: Annotated[
        int, typer.Option("--maximum-parameter-bytes", min=1, max=512 * 1024**2)
    ] = 256 * 1024**2,
) -> None:
    """Create the real offline initial operator; complete retries are DATA-only."""

    def action() -> None:
        config = read_scientific_ablation_command_config_v1(recipe)
        snapshot = operator.create_scientific_ablation_operator_v1(
            config=config,
            base=base,
            root=root,
            maximum_trainable_parameter_bytes=maximum_parameter_bytes,
        )
        _emit(_summary(snapshot))

    _guard(action)


@app.command("inspect")
def inspect(operator_path: Annotated[Path, typer.Option("--operator")]) -> None:
    """Audit descriptor, whole journal and actual checkpoints; construct no model."""
    _guard(lambda: _emit(_summary(operator.inspect_scientific_ablation_operator_v1(operator_path))))


@app.command("best")
def best(operator_path: Annotated[Path, typer.Option("--operator")]) -> None:
    """Distinguish selected validation best from a physically persisted boundary."""
    _guard(
        lambda: _emit(
            export.inspect_scientific_ablation_best_v1(operator_path).model_dump(mode="json")
        )
    )


@app.command("phase-request")
def phase_request(
    operator_path: Annotated[Path, typer.Option("--operator")],
    previous: Annotated[str, typer.Option("--expected-previous-sha256")],
    phase: Annotated[PhaseOption, typer.Option("--phase")],
    output: Annotated[
        Path, typer.Option("--out", help="Exclusive saved intent; never overwritten.")
    ],
) -> None:
    """Persist explicit intent BEFORE a separate native train/validate/checkpoint command."""

    def action() -> None:
        request = operator.make_scientific_ablation_phase_request_v1(
            operator_path, expected_previous_sha256=previous, phase=_phase(phase)
        )
        _persist(
            output,
            canonical_json_file_bytes(request.model_dump(mode="json")),
            operator_path=operator_path,
        )
        _emit(request.model_dump(mode="json"))

    _guard(action)


def _advance(path: Path, saved: Path, base: Path, replay: int, phase: Phase | None) -> None:
    def action() -> None:
        request = operator.read_scientific_ablation_phase_request_v1(saved)
        if phase is not None and request.payload.phase != phase:
            raise ValueError("Saved intent does not belong to this exact phase command")
        result = operator.advance_scientific_ablation_operator_v1(
            path, request=request, base=base, maximum_replay_phases=replay
        )
        _emit(result.model_dump(mode="json"))

    _guard(action)


@app.command("train")
def train(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[Path, typer.Option("--request")],
    base: Annotated[Path, typer.Option("--base")],
    maximum_replay_phases: Annotated[
        int, typer.Option("--maximum-replay-phases", min=0, max=200000)
    ] = 4096,
) -> None:
    """Execute/retry exactly the saved train phase, not a moving training loop."""
    _advance(operator_path, request, base, maximum_replay_phases, "train")


@app.command("validate")
def validate(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[Path, typer.Option("--request")],
    base: Annotated[Path, typer.Option("--base")],
    maximum_replay_phases: Annotated[
        int, typer.Option("--maximum-replay-phases", min=0, max=200000)
    ] = 4096,
) -> None:
    """Execute/retry exactly the saved validation phase; no generated-code scoring."""
    _advance(operator_path, request, base, maximum_replay_phases, "validation")


@app.command("checkpoint")
def checkpoint(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[Path, typer.Option("--request")],
    base: Annotated[Path, typer.Option("--base")],
    maximum_replay_phases: Annotated[
        int, typer.Option("--maximum-replay-phases", min=0, max=200000)
    ] = 4096,
) -> None:
    """Publish exactly the checkpoint due at the saved prefix, never train implicitly."""
    _advance(operator_path, request, base, maximum_replay_phases, "checkpoint")


@app.command("resume")
def resume(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[
        Path, typer.Option("--request", help="The SAME original intent after a lost reply.")
    ],
    base: Annotated[Path, typer.Option("--base")],
    maximum_replay_phases: Annotated[
        int, typer.Option("--maximum-replay-phases", min=0, max=200000)
    ] = 4096,
) -> None:
    """Reconstruct bounded state for one saved phase, or return its immutable DATA result."""
    _advance(operator_path, request, base, maximum_replay_phases, None)


@app.command("export-request")
def export_request(
    operator_path: Annotated[Path, typer.Option("--operator")],
    checkpoint_event: Annotated[str, typer.Option("--checkpoint-event-sha256")],
    lifecycle: Annotated[str, typer.Option("--lifecycle-sha256")],
    output: Annotated[Path, typer.Option("--out")],
    selection: Annotated[
        SelectionOption, typer.Option("--selection")
    ] = SelectionOption.SELECTED_BEST,
    maximum_storage_bytes: Annotated[
        int, typer.Option("--maximum-storage-bytes", min=1, max=16 * 1024**3)
    ] = 512 * 1024**2,
    minimum_free_bytes: Annotated[
        int, typer.Option("--minimum-free-bytes", min=0, max=16 * 1024**3)
    ] = 1024**3,
) -> None:
    """Persist exact acknowledged checkpoint/selection intent; do not build inference yet."""

    def action() -> None:
        request = export.make_scientific_ablation_export_request_v1(
            operator_path,
            checkpoint_event_sha256=checkpoint_event,
            expected_lifecycle_sha256=lifecycle,
            selection=_selection(selection),
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        _persist(
            output,
            canonical_json_file_bytes(request.model_dump(mode="json")),
            operator_path=operator_path,
        )
        _emit(request.model_dump(mode="json"))

    _guard(action)


@app.command("export")
def export_weights(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[Path, typer.Option("--request")],
    base: Annotated[Path, typer.Option("--base")],
    root: Annotated[Path, typer.Option("--root")],
) -> None:
    """Project/export the saved persisted selection; complete retries build no model."""

    def action() -> None:
        intent = export.read_scientific_ablation_export_request_v1(request)
        receipt = export.export_scientific_ablation_operator_v1(
            operator_path, request=intent, base=base, root=root
        )
        _emit(
            dict(
                kind="iris-scientific-ablation-export-command-v1",
                result=receipt.result.model_dump(mode="json"),
                export_path=str(receipt.path),
                inference_path=str(receipt.inference_path),
                scientific_acceptance=False,
            )
        )

    _guard(action)


@app.command("inspect-export")
def inspect_export(
    operator_path: Annotated[Path, typer.Option("--operator")],
    request: Annotated[Path, typer.Option("--request")],
    root: Annotated[Path, typer.Option("--root")],
) -> None:
    """Read a published exact export; never construct a model or adopt staging."""

    def action() -> None:
        intent = export.read_scientific_ablation_export_request_v1(request)
        receipt = export.inspect_scientific_ablation_operator_export_v1(
            operator_path, request=intent, root=root
        )
        _emit(
            dict(
                kind="iris-scientific-ablation-export-inspection-v1",
                result=receipt.result.model_dump(mode="json"),
                export_path=str(receipt.path),
                inference_path=str(receipt.inference_path),
                read_only=True,
                scientific_acceptance=False,
            )
        )

    _guard(action)


__all__ = ["app"]
