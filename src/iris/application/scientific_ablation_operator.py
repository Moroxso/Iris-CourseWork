"""Actual offline ablation create/inspect and exact single-phase operator requests.

Committed retries are DATA-only, including after later progress. Fresh native
restore can replay the bounded committed tail; this is not compute exactly-once.
The private-owner filesystem is not an adversarial-host authentication boundary.
"""

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_command_composition import (
    _snapshot,
    create_scientific_ablation_command_lifecycle_v1,
)
from iris.application.scientific_ablation_command_config import (
    IrisScientificAblationCommandConfigV1,
)
from iris.application.scientific_ablation_execution import (
    IrisScientificAblationExecutionSnapshotV1,
    _claim,
    _context,
    _local,
    _member_names,
    _stamp,
    create_scientific_ablation_execution_v1,
    read_scientific_ablation_execution_v1,
    resume_scientific_ablation_execution_v1,
)
from iris.application.scientific_ablation_lifecycle import _exact
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular, _unique
from iris.application.scientific_command import _input_bytes
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_execution_v1 import IrisScientificAblationExecutionEventV1
from iris.artifacts.scientific_ablation_operator_v1 import (
    IrisScientificAblationOperatorV1,
    IrisScientificAblationPhaseRequestV1,
    IrisScientificAblationPhaseResultV1,
    Phase,
    ScientificAblationOperatorPayloadV1,
    ScientificAblationPhaseRequestPayloadV1,
    ScientificAblationPhaseResultPayloadV1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_RECIPE = "iris/scientific-ablation-operator-recipe/v1"
_ADDRESS = "iris/scientific-ablation-operator-address/v1"
_DESCRIPTOR = "iris/scientific-ablation-operator/v1"
_REQUEST = "iris/scientific-ablation-phase-request/v1"
_RESULT = "iris/scientific-ablation-phase-result/v1"
_LOCK = ".iris-command.lock"
_LIMIT = 256 * 1024


def _nonfinite(value: str) -> NoReturn:
    raise ValueError("Nonfinite operator JSON is not allowed")


class IrisScientificAblationOperatorError(ApplicationError):
    """Content-redacted operator failure, not a wrong-answer/scientific verdict."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationOperatorSnapshotV1:
    path: Path
    descriptor: IrisScientificAblationOperatorV1
    execution: IrisScientificAblationExecutionSnapshotV1


def _address(recipe_sha256: str, maximum: int) -> str:
    return domain_sha256(
        _ADDRESS, canonical_json_bytes(dict(recipe_sha256=recipe_sha256, parameter_bytes=maximum))
    )


def _recipe(descriptor: IrisScientificAblationOperatorV1) -> IrisScientificAblationCommandConfigV1:
    raw = descriptor.payload.recipe_json.encode()
    _json(raw)
    recipe = IrisScientificAblationCommandConfigV1.model_validate_json(raw, strict=True)
    if (
        canonical_json_bytes(recipe.model_dump(mode="json")) != raw
        or domain_sha256(_RECIPE, raw) != descriptor.payload.recipe_sha256
    ):
        raise ValueError("Operator recipe identity differs")
    return _snapshot(recipe)


def _read(
    path: Path, *, staged_address: str | None = None
) -> IrisScientificAblationOperatorSnapshotV1:
    path = _local(path)
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    if _member_names(path, 3) != {"operator.json", "executions", _LOCK}:
        raise ValueError("Operator directory members differ")
    _stamp(path / "executions", directory=True)
    if _member_names(path / "executions", 1) != {"scientific-ablation-executions-v1"}:
        raise ValueError("Operator execution root members differ")
    _stamp(path / _LOCK)
    if (path / _LOCK).stat().st_size != 1:
        raise ValueError("Operator lock size differs")
    pin = _stamp(path / "operator.json")
    raw = _bounded_read(path / "operator.json", _LIMIT)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
    descriptor = IrisScientificAblationOperatorV1.model_validate_json(
        canonical_json_bytes(value), strict=True
    )
    if raw != canonical_json_file_bytes(
        descriptor.model_dump(mode="json")
    ) or descriptor.sha256 != domain_sha256(
        _DESCRIPTOR, canonical_json_bytes(descriptor.payload.model_dump(mode="json"))
    ):
        raise ValueError("Operator descriptor identity differs")
    recipe = _recipe(descriptor)
    payload = descriptor.payload
    address = _address(payload.recipe_sha256, payload.maximum_trainable_parameter_bytes)
    if staged_address is None:
        if path.name != address:
            raise ValueError("Operator is not at its recipe-addressed path")
    elif staged_address != address or path.name != f".iris-operator-{address}.tmp":
        raise ValueError("Pending operator does not match this exact creation intent")
    store = path / "executions/scientific-ablation-executions-v1"
    _regular(store, directory=True)
    if _member_names(store, 2) != {".iris-create.lock", payload.execution_sha256}:
        raise ValueError("Operator execution namespace differs")
    _stamp(store / ".iris-create.lock")
    if (store / ".iris-create.lock").stat().st_size != 1:
        raise ValueError("Operator execution create lock differs")
    snapshot = read_scientific_ablation_execution_v1(store / payload.execution_sha256)
    configuration, run = _context(snapshot.header)
    if (
        snapshot.header.sha256 != payload.execution_sha256
        or snapshot.header.payload.initial.sha256 != payload.initial_checkpoint_sha256
        or snapshot.header.payload.maximum_journal_bytes != recipe.maximum_journal_bytes
        or run.ablation != recipe.ablation
        or run.optimizer_policy != recipe.optimizer
        or run.teacher_policy != recipe.objective
        or run.seed != recipe.run_seed
        or configuration.policy != recipe.lifecycle
        or _stamp(path / "operator.json") != pin
        or _bounded_read(path / "operator.json", _LIMIT) != raw
    ):
        raise ValueError("Operator recipe and actual execution disagree")
    return IrisScientificAblationOperatorSnapshotV1(path, descriptor, snapshot)


def inspect_scientific_ablation_operator_v1(path: Path) -> IrisScientificAblationOperatorSnapshotV1:
    """Full read-only descriptor/journal/current-best audit, no model construction."""
    try:
        return _read(path)
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorError("Ablation operator inspection failed") from None


def create_scientific_ablation_operator_v1(
    *,
    config: IrisScientificAblationCommandConfigV1,
    base: Path,
    root: Path,
    maximum_trainable_parameter_bytes: int = 256 * 1024**2,
) -> IrisScientificAblationOperatorSnapshotV1:
    """Atomically bind real initial execution to its recipe; complete retries are DATA-only.

    A complete unpublished descriptor is retryable without a model. Incomplete
    unknown/partial staging is preserved for inspection, never swept or overwritten.
    """
    try:
        config = _snapshot(config)
        base, root = _local(base), _local(root)
        maximum = maximum_trainable_parameter_bytes
        if type(maximum) is not int or not 1 <= maximum <= 512 * 1024**2:
            raise ValueError("Exact bounded parameter allocation ceiling required")
        recipe_raw = canonical_json_bytes(config.model_dump(mode="json"))
        recipe_sha = domain_sha256(_RECIPE, recipe_raw)
        address = _address(recipe_sha, maximum)
        parent = root / "scientific-ablation-operators-v1"
        for node in (parent, *parent.parents):
            with suppress(FileNotFoundError):
                _regular(node, directory=True)
        target, stage = parent / address, parent / f".iris-operator-{address}.tmp"
        if target.exists():
            return _read(target)
        parent.mkdir(parents=True, exist_ok=True)
        with _claim(parent, name=".iris-operator-create.lock") as claim:
            if target.exists():
                return _read(target)
            if not stage.exists():
                native = create_scientific_ablation_command_lifecycle_v1(
                    config=config, base=base, maximum_trainable_parameter_bytes=maximum
                )
                stage.mkdir(mode=0o700)
                execution = create_scientific_ablation_execution_v1(
                    native,
                    root=stage / "executions",
                    maximum_journal_bytes=config.maximum_journal_bytes,
                )
                payload = ScientificAblationOperatorPayloadV1(
                    recipe_json=recipe_raw.decode(),
                    recipe_sha256=recipe_sha,
                    maximum_trainable_parameter_bytes=maximum,
                    execution_sha256=execution.header.sha256,
                    initial_checkpoint_sha256=execution.header.payload.initial.sha256,
                )
                descriptor = IrisScientificAblationOperatorV1(
                    sha256=domain_sha256(
                        _DESCRIPTOR, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                    payload=payload,
                )
                _write_exact_file(stage / _LOCK, b"1")
                _write_exact_file(
                    stage / "operator.json",
                    canonical_json_file_bytes(descriptor.model_dump(mode="json")),
                )
            staged = _read(stage, staged_address=address)
            if staged.execution.events:
                raise ValueError("Unpublished initial operator already contains native progress")
            claim.check()
            rename_directory_with_windows_retry(stage, target)
            claim.check()
        return _read(target)
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorError(
            "Ablation operator creation failed; inspect preserved staging before retrying"
        ) from None


def _checked_request(
    request: IrisScientificAblationPhaseRequestV1,
) -> IrisScientificAblationPhaseRequestV1:
    _exact(request, IrisScientificAblationPhaseRequestV1)
    _exact(request.payload, ScientificAblationPhaseRequestPayloadV1)
    value = IrisScientificAblationPhaseRequestV1.model_validate(request.model_dump(mode="python"))
    if value.sha256 != domain_sha256(
        _REQUEST, canonical_json_bytes(value.payload.model_dump(mode="json"))
    ):
        raise ValueError("Phase request digest differs")
    return value


def make_scientific_ablation_phase_request_v1(
    path: Path,
    *,
    expected_previous_sha256: str,
    phase: Phase,
) -> IrisScientificAblationPhaseRequestV1:
    """Explicit prefix only; never silently select the current head for a retry."""
    try:
        snapshot = _read(path)
        payload = ScientificAblationPhaseRequestPayloadV1(
            operator_sha256=snapshot.descriptor.sha256,
            execution_sha256=snapshot.execution.header.sha256,
            expected_previous_sha256=expected_previous_sha256,
            phase=phase,
            purpose=_recipe(snapshot.descriptor).purpose,
        )
        return IrisScientificAblationPhaseRequestV1(
            sha256=domain_sha256(_REQUEST, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorError("Ablation phase request is invalid") from None


def read_scientific_ablation_phase_request_v1(path: Path) -> IrisScientificAblationPhaseRequestV1:
    """Bounded human-readable request JSON; reject duplicate keys before schema parsing."""
    try:
        raw = _input_bytes(path, 64 * 1024)
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite
        )
        return _checked_request(
            IrisScientificAblationPhaseRequestV1.model_validate_json(
                canonical_json_bytes(value), strict=True
            )
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorError("Ablation phase request is invalid") from None


def _position(
    snapshot: IrisScientificAblationOperatorSnapshotV1,
    request: IrisScientificAblationPhaseRequestV1,
) -> tuple[int, IrisScientificAblationExecutionEventV1 | None]:
    payload = request.payload
    if (
        payload.operator_sha256 != snapshot.descriptor.sha256
        or payload.execution_sha256 != snapshot.execution.header.sha256
        or payload.purpose != _recipe(snapshot.descriptor).purpose
    ):
        raise ValueError("Request belongs to another operator/mode/purpose")
    positions = {snapshot.execution.header.sha256: 0}
    positions.update({event.sha256: event.payload.sequence for event in snapshot.execution.events})
    if payload.expected_previous_sha256 not in positions:
        raise ValueError("Request prefix is not in this exact committed journal")
    index = positions[payload.expected_previous_sha256]
    if index < len(snapshot.execution.events):
        event = snapshot.execution.events[index]
        if event.payload.record.kind != payload.phase:
            raise ValueError("Existing next phase conflicts with this request")
        return index, event
    phase = "train" if not index else snapshot.execution.events[-1].payload.after.phase
    if phase != payload.phase:
        raise ValueError("Requested native phase is not due at this prefix")
    return index, None


def _result(
    request: IrisScientificAblationPhaseRequestV1,
    event: IrisScientificAblationExecutionEventV1,
) -> IrisScientificAblationPhaseResultV1:
    if event.payload.previous_sha256 != request.payload.expected_previous_sha256:
        raise ValueError("Returned phase does not bind the requested prefix")
    payload = ScientificAblationPhaseResultPayloadV1(request=request, event=event)
    return IrisScientificAblationPhaseResultV1(
        sha256=domain_sha256(_RESULT, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )


def advance_scientific_ablation_operator_v1(
    path: Path,
    *,
    request: IrisScientificAblationPhaseRequestV1,
    base: Path,
    maximum_replay_phases: int = 4096,
) -> IrisScientificAblationPhaseResultV1:
    """One exact native phase, or an identical DATA-only previously committed result."""
    recovery_allowed = False
    try:
        request = _checked_request(request)
        base = _local(base)
        if type(maximum_replay_phases) is not int or not 0 <= maximum_replay_phases <= 200000:
            raise ValueError("Exact bounded replay ceiling required")
        snapshot = _read(path)
        _, committed = _position(snapshot, request)
        if committed is not None:
            return _result(request, committed)
        recovery_allowed = True
        with _claim(snapshot.path, name=_LOCK) as claim:
            current = _read(snapshot.path)
            if current.descriptor != snapshot.descriptor:
                raise ValueError("Operator descriptor changed before native work")
            _, committed = _position(current, request)
            if committed is not None:
                return _result(request, committed)
            native = create_scientific_ablation_command_lifecycle_v1(
                config=_recipe(current.descriptor),
                base=base,
                maximum_trainable_parameter_bytes=current.descriptor.payload.maximum_trainable_parameter_bytes,
            )
            owner = resume_scientific_ablation_execution_v1(
                current.execution.path,
                lifecycle=native,
                maximum_replay_phases=maximum_replay_phases,
            )
            if _read(snapshot.path).descriptor != snapshot.descriptor:
                raise ValueError("Operator descriptor changed during reconstruction")
            claim.check()
            owner.advance(
                expected_previous_sha256=request.payload.expected_previous_sha256,
                expected_phase=request.payload.phase,
            )
            claim.check()
            after = _read(snapshot.path)
            if after.descriptor != snapshot.descriptor:
                raise ValueError("Operator descriptor changed after native phase")
            _, committed = _position(after, request)
            if committed is None:
                raise ValueError("Requested phase has not committed")
            return _result(request, committed)
    except BaseException as error:
        if _has_signal(error):
            raise
        # A lost ACK or a concurrent winner may already have published exactly
        # this phase. Re-audit immutable DATA, never execute a follow-on phase.
        try:
            request = _checked_request(request)
            if not recovery_allowed:
                raise ValueError("Invalid controls are not a committed retry")
            recovered = _read(path)
            _, committed = _position(recovered, request)
            if committed is not None:
                return _result(request, committed)
        except Exception:
            pass
        raise IrisScientificAblationOperatorError(
            "Ablation phase failed; inspect or retry the identical expected-prefix request"
        ) from None


__all__ = [
    "IrisScientificAblationOperatorError",
    "IrisScientificAblationOperatorSnapshotV1",
    "advance_scientific_ablation_operator_v1",
    "create_scientific_ablation_operator_v1",
    "inspect_scientific_ablation_operator_v1",
    "make_scientific_ablation_phase_request_v1",
    "read_scientific_ablation_phase_request_v1",
]
