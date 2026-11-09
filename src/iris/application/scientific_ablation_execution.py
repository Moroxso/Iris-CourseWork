"""Distinct bounded ablation phase journal, actual owner and exact checkpoint replay.

Committed tail replay reconstructs native train/validation state; it is NOT new
phase progress, generation/candidate replay, launch approval or scientific quality.
The persistent OS lock is never reaped by PID or removed. No baseline type/domain
is reinterpreted, and no native owner is recovered from serialized DATA alone.
"""

import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path, PureWindowsPath
from typing import Literal
from uuid import uuid4

from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    _RunConfiguration,
    _verify,
    build_scientific_ablation_checkpoint_v1,
)
from iris.application.scientific_ablation_lifecycle import (
    IrisScientificAblationLifecycleBoundaryV1,
    IrisScientificAblationLifecycleV1,
    _exact,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    IrisScientificAblationLifecycleReceiptV1,
    _bundle_parts,
    _Configuration,
    build_scientific_ablation_lifecycle_checkpoint_v1,
    read_scientific_ablation_lifecycle_checkpoint_v1,
    restore_scientific_ablation_lifecycle_v1,
    write_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
)
from iris.application.scientific_ablation_validation import (
    IrisScientificAblationValidationV1,
    _has_signal,
)
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular
from iris.application.scientific_optimization import scientific_learning_rate_v1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_execution_v1 import (
    IrisScientificAblationExecutionEventV1,
    IrisScientificAblationExecutionHeaderV1,
    ScientificAblationExecutionCheckpointV1,
    ScientificAblationExecutionEventPayloadV1,
    ScientificAblationExecutionHeaderPayloadV1,
    ScientificAblationExecutionRecordV1,
    ScientificAblationExecutionTrainingMetricV1,
    ScientificAblationExecutionTrainV1,
    ScientificAblationExecutionValidationMetricV1,
    ScientificAblationExecutionValidationV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import ScientificAblationLifecycleStateV1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_HEADER_DOMAIN = "iris/scientific-ablation-execution/v1"
_EVENT_DOMAIN = "iris/scientific-ablation-execution-event/v1"
_HISTORY_DOMAIN = "iris/scientific-ablation-run-history/v1"
_HEADER_LIMIT = 16 * 1024**2
_EVENT_LIMIT = 65536
_MAX_EVENTS = 200000
_HEADER_FILE = "run.json"
_INITIAL_FILE = "initial.safetensors"
_LOCK_FILE = ".iris-owner.lock"
_OPEN_TOKEN = object()


class IrisScientificAblationExecutionError(ApplicationError):
    """Content-redacted journal/native compatibility or phase publication failure."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationExecutionSnapshotV1:
    path: Path
    header: IrisScientificAblationExecutionHeaderV1
    initial_state_bytes: bytes
    events: tuple[IrisScientificAblationExecutionEventV1, ...]
    journal_size_bytes: int


@dataclass(frozen=True, slots=True)
class _Pending:
    event: IrisScientificAblationExecutionEventV1
    event_bytes: bytes
    checkpoint: IrisScientificAblationLifecycleBundleV1 | None
    checkpoint_identity: tuple[str, str, str | None] | None


def _local(path: Path) -> Path:
    if not isinstance(path, Path) or not path.name or str(path).startswith(("//", "\\\\")):
        raise ValueError("Explicit local ablation execution path required")
    windows = PureWindowsPath(str(path))
    if windows.drive and not windows.root:
        raise ValueError("Drive-relative execution paths are forbidden")
    return path.absolute()


def _header_bytes(header: IrisScientificAblationExecutionHeaderV1) -> bytes:
    _exact(header, IrisScientificAblationExecutionHeaderV1)
    _exact(header.payload, ScientificAblationExecutionHeaderPayloadV1)
    owned = IrisScientificAblationExecutionHeaderV1.model_validate(header.model_dump(mode="python"))
    if (
        domain_sha256(_HEADER_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json")))
        != owned.sha256
    ):
        raise ValueError("Ablation execution header identity differs")
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > min(_HEADER_LIMIT, owned.payload.maximum_journal_bytes):
        raise ValueError("Ablation execution header exceeds its byte ceiling")
    return content


def _context(
    header: IrisScientificAblationExecutionHeaderV1,
) -> tuple[_Configuration, _RunConfiguration]:
    _header_bytes(header)
    initial = header.payload.initial.payload
    _json(initial.configuration_json.encode())
    _json(header.payload.lifecycle_configuration_json.encode())
    run = _RunConfiguration.model_validate_json(initial.configuration_json)
    lifecycle = _Configuration.model_validate_json(header.payload.lifecycle_configuration_json)
    if (
        domain_sha256(
            "iris/scientific-ablation-run-configuration/v1", initial.configuration_json.encode()
        )
        != initial.configuration_sha256
        or domain_sha256(
            "iris/scientific-ablation-lifecycle-configuration/v1",
            header.payload.lifecycle_configuration_json.encode(),
        )
        != header.payload.lifecycle_configuration_sha256
        or lifecycle.run_sha256 != initial.configuration_sha256
        or lifecycle.ablation_policy_sha256 != run.ablation_policy_sha256
        or initial.completed_steps != 0
        or initial.examples_seen != 0
        or initial.target_tokens_seen != 0
        or initial.history_sha256
        != domain_sha256(_HISTORY_DOMAIN, initial.configuration_sha256.encode())
        or initial.scientific_state_sha256 != run.initial_scientific_state_sha256
        or lifecycle.policy.purpose != run.optimizer_policy.purpose
    ):
        raise ValueError("Initial ablation lifecycle/run identity differs")
    if lifecycle.policy.purpose == "scientific" and (
        lifecycle.validation_examples != 5000 or run.dataset_examples != 50000
    ):
        raise ValueError("Scientific execution requires complete source corpora")
    return lifecycle, run


def _event_bytes(event: IrisScientificAblationExecutionEventV1) -> bytes:
    _exact(event, IrisScientificAblationExecutionEventV1)
    _exact(event.payload, ScientificAblationExecutionEventPayloadV1)
    _exact(event.payload.before, ScientificAblationLifecycleStateV1)
    _exact(event.payload.after, ScientificAblationLifecycleStateV1)
    owned = IrisScientificAblationExecutionEventV1.model_validate(event.model_dump(mode="python"))
    if (
        domain_sha256(_EVENT_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json")))
        != owned.sha256
    ):
        raise ValueError("Ablation phase event identity differs")
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _EVENT_LIMIT:
        raise ValueError("Ablation phase event exceeds its byte ceiling")
    return content


def _transition(
    header: IrisScientificAblationExecutionHeaderV1,
    previous: IrisScientificAblationExecutionEventV1 | None,
    event: IrisScientificAblationExecutionEventV1,
) -> None:
    configuration, run = _context(header)
    policy, optimizer = configuration.policy, run.optimizer_policy
    _event_bytes(event)
    payload = event.payload
    before = ScientificAblationLifecycleStateV1() if previous is None else previous.payload.after
    examples = 0 if previous is None else previous.payload.examples_seen
    tokens = 0 if previous is None else previous.payload.target_tokens_seen
    history = (
        header.payload.initial.payload.history_sha256
        if previous is None
        else previous.payload.history_sha256
    )
    state_sha = (
        header.payload.initial.payload.scientific_state_sha256
        if previous is None
        else previous.payload.scientific_state_sha256
    )
    if (
        payload.execution_sha256 != header.sha256
        or payload.sequence != (1 if previous is None else previous.payload.sequence + 1)
        or payload.previous_sha256 != (header.sha256 if previous is None else previous.sha256)
        or payload.before != before
        or before.phase == "stopped"
        or payload.record.kind != before.phase
    ):
        raise ValueError("Ablation event does not extend its exact committed prefix")
    values = before.model_dump(mode="python")
    record = payload.record
    if type(record) is ScientificAblationExecutionTrainV1:
        metric, step = record.metric, before.step + 1
        initial = header.payload.initial.payload
        tensor_shapes = {tensor.key: tensor.shape for tensor in initial.tensors}
        # Native participation counts scalar elements, not named parameter tensors.
        registered_numel = sum(
            prod(tensor_shapes[f"model/{name}"]) for name in initial.parameter_names
        )
        weights = run.teacher_policy.weights
        loss = (
            metric.language_model_loss
            + weights.probe * metric.probe_loss
            + weights.diversity * metric.diversity_loss
            + weights.norm * metric.norm_loss
            + weights.utilization * metric.utilization_loss
        )
        if (
            metric.step_index != step
            or step > optimizer.max_optimizer_steps
            or metric.configuration_sha256 != run.optimizer_sha256
            or metric.purpose != optimizer.purpose
            or metric.ablation_policy_sha256 != run.ablation_policy_sha256
            or metric.model_configuration_sha256 != run.model_configuration_sha256
            or metric.scientific_state_before_sha256 != state_sha
            or metric.registered_parameter_count != registered_numel
            or metric.learning_rate != scientific_learning_rate_v1(optimizer, step=step)
            or not optimizer.minimum_target_tokens
            <= metric.supervised_token_count
            <= optimizer.maximum_target_tokens
            or metric.example_count > min(optimizer.maximum_examples, run.dataset_examples)
            or metric.gradients_clipped != (metric.gradient_l2_norm > optimizer.max_gradient_norm)
            or metric.loss != loss
        ):
            raise ValueError("Ablation training event contradicts its native configuration")
        examples += metric.example_count
        tokens += metric.supervised_token_count
        history = domain_sha256(
            _HISTORY_DOMAIN,
            canonical_json_bytes(
                dict(
                    previous=history,
                    result=metric.model_dump(mode="json"),
                    examples_seen=examples,
                    target_tokens_seen=tokens,
                )
            ),
        )
        state_sha = metric.scientific_state_after_sha256
        values |= dict(
            step=step,
            phase="validation"
            if step == optimizer.max_optimizer_steps or step % policy.validate_every_steps == 0
            else "checkpoint"
            if step % policy.checkpoint_every_steps == 0
            else "train",
        )
    elif type(record) is ScientificAblationExecutionValidationV1:
        report = record.metric
        if (
            report.at_optimizer_step != before.step
            or report.run_configuration_sha256
            != header.payload.initial.payload.configuration_sha256
            or report.model_configuration_sha256 != run.model_configuration_sha256
            or report.ablation_policy_sha256 != run.ablation_policy_sha256
            or report.scientific_state_sha256 != state_sha
            or report.dataset_sha256 != configuration.validation_sha256
            or report.example_count != configuration.validation_examples
            or report.target_token_count != configuration.validation_target_tokens
        ):
            raise ValueError("Ablation validation belongs to a different mode/state/corpus/step")
        better = before.best_nll_per_token is None or report.target_nll_per_token < (
            before.best_nll_per_token - policy.minimum_improvement
        )
        if better != (record.selected_checkpoint_sha256 is not None):
            raise ValueError("Validation selected-best identity contradicts its improvement")
        misses = 0 if better else before.nonimproving_validations + 1
        reason = (
            "early_stopping"
            if misses >= policy.early_stopping_patience
            else "step_budget"
            if before.step == optimizer.max_optimizer_steps
            else None
        )
        values |= dict(
            phase="checkpoint"
            if reason is not None or before.step % policy.checkpoint_every_steps == 0
            else "train",
            validation_count=before.validation_count + 1,
            last_validation_step=before.step,
            best_step=before.step if better else before.best_step,
            best_nll_per_token=report.target_nll_per_token if better else before.best_nll_per_token,
            nonimproving_validations=misses,
            stop_reason=reason,
        )
    else:
        values |= dict(
            phase="stopped" if before.stop_reason is not None else "train",
            checkpoint_count=before.checkpoint_count + 1,
            last_checkpoint_step=before.step,
        )
    if payload.after != ScientificAblationLifecycleStateV1.model_validate(values) or (
        payload.examples_seen,
        payload.target_tokens_seen,
        payload.history_sha256,
        payload.scientific_state_sha256,
    ) != (examples, tokens, history, state_sha):
        raise ValueError("Ablation state/counters/history/scientific-state chain differs")


def _selected_best(
    previous: IrisScientificAblationExecutionEventV1 | None,
    event: IrisScientificAblationExecutionEventV1,
) -> IrisScientificAblationExecutionEventV1 | None:
    record = event.payload.record
    return (
        event
        if type(record) is ScientificAblationExecutionValidationV1
        and record.selected_checkpoint_sha256 is not None
        else previous
    )


def _checkpoint_for_event(
    path: Path,
    header: IrisScientificAblationExecutionHeaderV1,
    event: IrisScientificAblationExecutionEventV1,
    *,
    best_event: IrisScientificAblationExecutionEventV1 | None,
) -> IrisScientificAblationLifecycleBundleV1:
    record = event.payload.record
    if type(record) is not ScientificAblationExecutionCheckpointV1:
        raise ValueError("Actual ablation checkpoint event required")
    folder = (
        path / "checkpoints/scientific-ablation-lifecycle-v1" / record.lifecycle_checkpoint_sha256
    )
    bundle = read_scientific_ablation_lifecycle_checkpoint_v1(folder)
    payload, current = bundle.manifest.payload, bundle.manifest.payload.current.payload
    if (
        payload.configuration_json != header.payload.lifecycle_configuration_json
        or payload.configuration_sha256 != header.payload.lifecycle_configuration_sha256
        or payload.state != event.payload.after
        or current.configuration_sha256 != header.payload.initial.payload.configuration_sha256
        or (
            current.examples_seen,
            current.target_tokens_seen,
            current.history_sha256,
            current.scientific_state_sha256,
        )
        != (
            event.payload.examples_seen,
            event.payload.target_tokens_seen,
            event.payload.history_sha256,
            event.payload.scientific_state_sha256,
        )
    ):
        raise ValueError("Physical lifecycle checkpoint contradicts its ablation event")
    if payload.best is None:
        if best_event is not None:
            raise ValueError("Previously selected native best is missing")
    elif (
        best_event is None
        or type(best_event.payload.record) is not ScientificAblationExecutionValidationV1
        or payload.best.sha256 != best_event.payload.record.selected_checkpoint_sha256
    ):
        raise ValueError("Physical best differs from the validation-selected identity")
    return bundle


def _stamp(path: Path, *, directory: bool = False) -> tuple[int, ...]:
    _regular(path, directory=directory)
    metadata = path.lstat()
    if not directory and metadata.st_nlink != 1:
        raise ValueError("Linked journal file forbidden")
    return (
        (metadata.st_dev, metadata.st_ino)
        if directory
        else (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
        )
    )


def _member_names(path: Path, maximum: int) -> set[str]:
    result = set()
    for index, entry in enumerate(path.iterdir()):
        if index >= maximum:
            raise ValueError("Journal directory member ceiling exceeded")
        result.add(entry.name)
    return result


def _event_content(folder: Path) -> bytes:
    _stamp(folder, directory=True)
    if _member_names(folder, 1) != {"event.json"}:
        raise ValueError("Unexpected phase-directory members")
    _stamp(folder / "event.json")
    return _bounded_read(folder / "event.json", _EVENT_LIMIT)


def _layout(path: Path) -> None:
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    if _member_names(path, 5) != {
        _HEADER_FILE,
        _INITIAL_FILE,
        _LOCK_FILE,
        "events",
        "checkpoints",
    }:
        raise ValueError("Ablation execution directory members differ")
    for name in ("events", "checkpoints"):
        _stamp(path / name, directory=True)
    for name in (_HEADER_FILE, _INITIAL_FILE, _LOCK_FILE):
        _stamp(path / name)
    # Windows byte-range ownership forbids another descriptor reading the locked
    # byte. The actual holder verifies b"1" through its own descriptor; a read-only
    # snapshot does not manufacture or claim native lock ownership from DATA.
    if (path / _LOCK_FILE).stat().st_size != 1:
        raise ValueError("Unexpected ablation ownership file size")


def read_scientific_ablation_execution_v1(path: Path) -> IrisScientificAblationExecutionSnapshotV1:
    """Read-only complete chain/current/best audit. No models, ownership writes or replay."""
    try:
        path = _local(path)
        _layout(path)
        initial_stamp = _stamp(path / _INITIAL_FILE)
        raw = _bounded_read(path / _HEADER_FILE, _HEADER_LIMIT)
        header = IrisScientificAblationExecutionHeaderV1.model_validate_json(raw)
        if _header_bytes(header) != raw or path.name != header.sha256:
            raise ValueError("Ablation header/path canonical identity differs")
        _context(header)
        initial = _bounded_read(path / _INITIAL_FILE, 512 * 1024**2)
        _verify(IrisScientificAblationCheckpointBundleV1(header.payload.initial, initial))
        entries: list[Path] = []
        original_names = []
        for index, entry in enumerate((path / "events").iterdir()):
            if index >= _MAX_EVENTS + 1024:
                raise ValueError("Too many committed/orphan phase entries")
            original_names.append(entry.name)
            if re.fullmatch(r"[0-9]{8}", entry.name):
                entries.append(entry)
            elif re.fullmatch(r"\.iris-event-[0-9a-f]{32}\.tmp", entry.name):
                _event_content(entry)  # An orphan is DATA, never a committed phase or native token.
            else:
                raise ValueError("Unknown ablation phase entry")
        previous = best = None
        events = []
        size = len(raw)
        signatures = []
        for index, entry in enumerate(sorted(entries), start=1):
            if index > _MAX_EVENTS or entry.name != f"{index:08d}":
                raise ValueError("Ablation phases must be contiguous and bounded")
            content = _event_content(entry)
            event = IrisScientificAblationExecutionEventV1.model_validate_json(content)
            if _event_bytes(event) != content:
                raise ValueError("Noncanonical ablation phase bytes")
            _transition(header, previous, event)
            if type(event.payload.record) is ScientificAblationExecutionCheckpointV1:
                _checkpoint_for_event(path, header, event, best_event=best)
            size += len(content)
            if size > header.payload.maximum_journal_bytes:
                raise ValueError("Ablation journal exceeds its byte ceiling")
            signatures.append((entry, _stamp(entry, directory=True), _stamp(entry / "event.json")))
            events.append(event)
            best, previous = _selected_best(best, event), event
        if (
            sorted(p.name for p in (path / "events").iterdir()) != sorted(original_names)
            or _bounded_read(path / _HEADER_FILE, _HEADER_LIMIT) != raw
            or _stamp(path / _INITIAL_FILE) != initial_stamp
            or any(
                _stamp(p, directory=True) != folder_sig or _stamp(p / "event.json") != file_sig
                for p, folder_sig, file_sig in signatures
            )
        ):
            raise ValueError("Ablation execution changed during its complete audit")
        return IrisScientificAblationExecutionSnapshotV1(path, header, initial, tuple(events), size)
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationExecutionError("Ablation execution inspection failed") from None


@contextmanager
def _claim(path: Path, *, name: str = _LOCK_FILE) -> Iterator[ProcessFileLock]:
    primary: BaseException | None = None
    try:
        with process_file_lock(path / name) as claim:
            try:
                yield claim
            except BaseException as error:
                primary = error
                raise
    except BaseException:
        if primary is not None and _has_signal(primary):
            raise primary from None
        raise


def _pending_identity(
    bundle: IrisScientificAblationLifecycleBundleV1,
) -> tuple[str, str, str | None]:
    manifest, _, _ = _bundle_parts(bundle)
    return (
        manifest.sha256,
        manifest.payload.current.payload.state_sha256,
        (None if manifest.payload.best is None else manifest.payload.best.payload.state_sha256),
    )


def _make_pending(
    header: IrisScientificAblationExecutionHeaderV1,
    previous: IrisScientificAblationExecutionEventV1 | None,
    lifecycle: IrisScientificAblationLifecycleV1,
    result: IrisScientificAblationOptimizationOutputV1
    | IrisScientificAblationValidationV1
    | IrisScientificAblationLifecycleBoundaryV1,
) -> _Pending:
    record: ScientificAblationExecutionRecordV1
    bundle = None
    if type(result) is IrisScientificAblationOptimizationOutputV1:
        record = ScientificAblationExecutionTrainV1(
            metric=ScientificAblationExecutionTrainingMetricV1.model_validate(asdict(result))
        )
    elif type(result) is IrisScientificAblationValidationV1:
        best = lifecycle.best_checkpoint
        if best is None:
            raise ValueError("Actual validation-selected best required")
        old_step = None if previous is None else previous.payload.after.best_step
        record = ScientificAblationExecutionValidationV1(
            metric=ScientificAblationExecutionValidationMetricV1.model_validate(asdict(result)),
            selected_checkpoint_sha256=best.manifest.sha256
            if lifecycle.state.best_step != old_step
            else None,
        )
    elif type(result) is IrisScientificAblationLifecycleBoundaryV1:
        bundle = build_scientific_ablation_lifecycle_checkpoint_v1(result)
        record = ScientificAblationExecutionCheckpointV1(
            lifecycle_checkpoint_sha256=bundle.manifest.sha256
        )
    else:
        raise ValueError("Exact original ablation phase output required")
    run = lifecycle._run
    payload = ScientificAblationExecutionEventPayloadV1(
        execution_sha256=header.sha256,
        sequence=1 if previous is None else previous.payload.sequence + 1,
        previous_sha256=header.sha256 if previous is None else previous.sha256,
        before=ScientificAblationLifecycleStateV1() if previous is None else previous.payload.after,
        after=lifecycle.state,
        examples_seen=run.examples_seen,
        target_tokens_seen=run.target_tokens_seen,
        history_sha256=run.history_sha256,
        scientific_state_sha256=run._committed_scientific_state_identity,
        record=record,
    )
    event = IrisScientificAblationExecutionEventV1(
        sha256=domain_sha256(_EVENT_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    _transition(header, previous, event)
    return _Pending(
        event, _event_bytes(event), bundle, None if bundle is None else _pending_identity(bundle)
    )


def _append(
    path: Path,
    header: IrisScientificAblationExecutionHeaderV1,
    previous: IrisScientificAblationExecutionEventV1 | None,
    pending: _Pending,
    best: IrisScientificAblationExecutionEventV1 | None,
    size: int,
    claim: ProcessFileLock,
) -> int:
    _transition(header, previous, pending.event)
    if (
        _event_bytes(pending.event) != pending.event_bytes
        or size + len(pending.event_bytes) > header.payload.maximum_journal_bytes
    ):
        raise ValueError("Pending phase changed or exceeds its byte ceiling")
    claim.check()
    if pending.checkpoint is not None:
        if _pending_identity(pending.checkpoint) != pending.checkpoint_identity:
            raise ValueError("Owned pending lifecycle checkpoint changed")
        write_scientific_ablation_lifecycle_checkpoint_v1(
            pending.checkpoint, root=path / "checkpoints"
        )
    if type(pending.event.payload.record) is ScientificAblationExecutionCheckpointV1:
        _checkpoint_for_event(path, header, pending.event, best_event=best)
    target = path / "events" / f"{pending.event.payload.sequence:08d}"
    if target.exists():
        if _event_content(target) != pending.event_bytes:
            raise ValueError("Committed ablation phase conflicts")
        return size + len(pending.event_bytes)
    claim.check()
    if shutil.disk_usage(path).free < len(pending.event_bytes) + 1024**3:
        raise ValueError("Ablation phase free-space reserve exceeded")
    stage = path / "events" / f".iris-event-{uuid4().hex}.tmp"
    os.mkdir(stage, mode=0o700)
    identity = _stamp(stage, directory=True)
    primary: BaseException | None = None
    try:
        _write_exact_file(stage / "event.json", pending.event_bytes)
        if _event_content(stage) != pending.event_bytes:
            raise ValueError("Staged ablation event differs")
        claim.check()
        try:
            rename_directory_with_windows_retry(stage, target)
        except OSError:
            if not target.exists() or _event_content(target) != pending.event_bytes:
                raise
        if _event_content(target) != pending.event_bytes:
            raise ValueError("Published ablation event differs")
        return size + len(pending.event_bytes)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if stage.exists():
                if _stamp(stage, directory=True) != identity or _member_names(stage, 1) - {
                    "event.json"
                }:
                    raise ValueError("Owned phase staging changed; preserve it")
                (stage / "event.json").unlink(missing_ok=True)
                stage.rmdir()
        except BaseException:
            if primary is not None and _has_signal(primary):
                raise primary from None
            raise


def _fresh(lifecycle: IrisScientificAblationLifecycleV1) -> None:
    if type(lifecycle) is not IrisScientificAblationLifecycleV1:
        raise ValueError("Exact native ablation lifecycle required")
    if lifecycle._busy or lifecycle._checkpoint_root is not None or lifecycle.publication_pending:
        raise ValueError("Idle lifecycle without an external checkpoint publisher required")
    if lifecycle.state != ScientificAblationLifecycleStateV1():
        raise ValueError("Fresh zero-phase native ablation lifecycle required")


class IrisScientificAblationExecutionV1:
    """Actual native owner plus immutable pending publication; create/resume only.

    Full open audits every phase. Active operations check immutable header/initial
    native identity/head and use an OS lock. This is not hostile-filesystem security.
    """

    def __init__(
        self,
        *,
        snapshot: IrisScientificAblationExecutionSnapshotV1,
        lifecycle: IrisScientificAblationLifecycleV1,
        _token: object | None = None,
    ) -> None:
        if _token is not _OPEN_TOKEN:
            raise IrisScientificAblationExecutionError("Use ablation execution create/resume")
        self._path = self._owned_path = snapshot.path
        self._lifecycle = self._owned_lifecycle = lifecycle
        self._header = deepcopy(snapshot.header)
        self._header_raw = _header_bytes(snapshot.header)
        self._initial_identity = _stamp(snapshot.path / _INITIAL_FILE)
        self._directories = tuple(
            _stamp(snapshot.path / name, directory=True) for name in (".", "events", "checkpoints")
        )
        self._head = deepcopy(snapshot.events[-1]) if snapshot.events else None
        self._best: IrisScientificAblationExecutionEventV1 | None = None
        for event in snapshot.events:
            self._best = _selected_best(self._best, event)
        self._journal_bytes = snapshot.journal_size_bytes
        self._pending: _Pending | None = None
        self._busy = self._poisoned = False
        self._validate_live(self._head)

    @property
    def path(self) -> Path:
        return self._owned_path

    @property
    def header(self) -> IrisScientificAblationExecutionHeaderV1:
        return deepcopy(self._header)

    @property
    def last_event(self) -> IrisScientificAblationExecutionEventV1 | None:
        return deepcopy(self._head)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def publication_pending(self) -> bool:
        return self._pending is not None

    def _validate_live(self, expected: IrisScientificAblationExecutionEventV1 | None) -> None:
        if (
            self._path != self._owned_path
            or self._lifecycle is not self._owned_lifecycle
            or self._poisoned
        ):
            raise ValueError("Execution native ownership changed")
        if _header_bytes(self._header) != self._header_raw:
            raise ValueError("Owned ablation header changed")
        lifecycle = self._owned_lifecycle
        state = ScientificAblationLifecycleStateV1() if expected is None else expected.payload.after
        initial = self._header.payload.initial.payload
        counters = (
            (0, 0, initial.history_sha256, initial.scientific_state_sha256)
            if expected is None
            else (
                expected.payload.examples_seen,
                expected.payload.target_tokens_seen,
                expected.payload.history_sha256,
                expected.payload.scientific_state_sha256,
            )
        )
        with lifecycle._operation():
            run = lifecycle._run
            if (
                lifecycle._checkpoint_root is not None
                or lifecycle._pending is not None
                or lifecycle._configuration_json
                != self._header.payload.lifecycle_configuration_json
                or lifecycle._configuration_sha256
                != self._header.payload.lifecycle_configuration_sha256
                or lifecycle._state != state
                or (
                    run.examples_seen,
                    run.target_tokens_seen,
                    run.history_sha256,
                    run._committed_scientific_state_identity,
                )
                != counters
            ):
                raise ValueError("Ablation journal lost its exact lifecycle/state/cursor")

    def _validate_head(self, *, pending: bool) -> None:
        _layout(self._owned_path)
        if (
            tuple(
                _stamp(self._owned_path / name, directory=True)
                for name in (".", "events", "checkpoints")
            )
            != self._directories
            or _bounded_read(self._owned_path / _HEADER_FILE, _HEADER_LIMIT) != self._header_raw
            or _stamp(self._owned_path / _INITIAL_FILE) != self._initial_identity
        ):
            raise ValueError("Ablation execution header/initial/native directories changed")
        sequence = 0 if self._head is None else self._head.payload.sequence
        if self._head is not None and _event_content(
            self._owned_path / "events" / f"{sequence:08d}"
        ) != _event_bytes(self._head):
            raise ValueError("Committed ablation journal head changed")
        if sequence >= _MAX_EVENTS or (
            not pending and (self._owned_path / "events" / f"{sequence + 1:08d}").exists()
        ):
            raise ValueError("Unexpected ablation journal next phase")

    def advance(
        self,
        *,
        expected_previous_sha256: str | None = None,
        expected_phase: Literal["train", "validation", "checkpoint"] | None = None,
    ) -> IrisScientificAblationExecutionEventV1:
        """Commit at most one new native phase; pending retry performs DATA publication only."""
        if self._busy or self._poisoned:
            raise IrisScientificAblationExecutionError("Ablation execution is busy or poisoned")
        self._busy = True
        try:
            with _claim(self._owned_path) as claim:
                if (expected_previous_sha256 is not None or expected_phase is not None) and (
                    type(expected_previous_sha256) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", expected_previous_sha256) is None
                    or type(expected_phase) is not str
                    or expected_phase not in ("train", "validation", "checkpoint")
                ):
                    raise ValueError("Both exact expected prefix and native phase required")
                self._validate_live(
                    self._pending.event if self._pending is not None else self._head
                )
                self._validate_head(pending=self._pending is not None)
                if expected_previous_sha256 is not None:
                    prefix = self._header.sha256 if self._head is None else self._head.sha256
                    phase = (
                        self._lifecycle.state.phase
                        if self._pending is None
                        else self._pending.event.payload.before.phase
                    )
                    if prefix != expected_previous_sha256 or phase != expected_phase:
                        raise ValueError("Native phase no longer matches the expected prefix")
                if self._pending is None:
                    if self._lifecycle.state.phase == "stopped":
                        raise ValueError("Ablation execution is stopped")
                    if (
                        self._journal_bytes + _EVENT_LIMIT
                        > self._header.payload.maximum_journal_bytes
                    ):
                        raise ValueError("Reserve a full phase before native execution")
                    if shutil.disk_usage(self._owned_path).free < _EVENT_LIMIT + 1024**3:
                        raise ValueError("Insufficient free-space reserve before native phase")
                    claim.check()
                    try:
                        result = self._lifecycle.advance()
                        if isinstance(result, IrisScientificAblationLifecycleReceiptV1):
                            raise ValueError("External checkpoint publication is not journal-owned")
                    except BaseException:
                        try:
                            self._validate_live(self._head)
                        except BaseException:
                            self._poisoned = True
                        raise
                    try:
                        self._pending = _make_pending(
                            self._header, self._head, self._lifecycle, result
                        )
                    except BaseException:
                        self._poisoned = True
                        raise
                pending = self._pending
                claim.check()
                size = _append(
                    self._owned_path,
                    self._header,
                    self._head,
                    pending,
                    self._best,
                    self._journal_bytes,
                    claim,
                )
                try:
                    self._validate_live(pending.event)
                    self._validate_head(pending=True)
                    claim.check()
                except BaseException:
                    self._poisoned = True
                    raise
                self._head, self._best = pending.event, _selected_best(self._best, pending.event)
                self._journal_bytes, self._pending = size, None
                return deepcopy(self._head)
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationExecutionError(
                "Ablation phase could not commit; inspect before retrying"
            ) from None
        finally:
            self._busy = False


def create_scientific_ablation_execution_v1(
    lifecycle: IrisScientificAblationLifecycleV1,
    *,
    root: Path,
    maximum_journal_bytes: int = 128 * 1024**2,
) -> IrisScientificAblationExecutionV1:
    """Publish original initial DATA before any native phase; explicit exact lifecycle required."""
    stage: Path | None = None
    primary: BaseException | None = None
    try:
        _fresh(lifecycle)
        root = _local(root)
        initial = build_scientific_ablation_checkpoint_v1(lifecycle._run)
        payload = ScientificAblationExecutionHeaderPayloadV1(
            lifecycle_configuration_json=lifecycle.configuration_json,
            lifecycle_configuration_sha256=lifecycle.configuration_sha256,
            initial=initial.manifest,
            maximum_journal_bytes=maximum_journal_bytes,
        )
        header = IrisScientificAblationExecutionHeaderV1(
            sha256=domain_sha256(
                _HEADER_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
        content = _header_bytes(header)
        _context(header)
        if len(content) + _EVENT_LIMIT > maximum_journal_bytes:
            raise ValueError("Journal must fit its initial header and one maximum phase")
        parent = root / "scientific-ablation-executions-v1"
        for ancestor in (parent, *parent.parents):
            with suppress(FileNotFoundError):
                _regular(ancestor, directory=True)
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / header.sha256
        with _claim(parent, name=".iris-create.lock"):
            if not target.exists():
                if (
                    shutil.disk_usage(parent).free
                    < len(content) + len(initial.state_bytes) + 1024**3
                ):
                    raise ValueError("Initial ablation publication free-space reserve exceeded")
                stage = parent / f".iris-execution-{uuid4().hex}.tmp"
                os.mkdir(stage, mode=0o700)
                os.mkdir(stage / "events", mode=0o700)
                os.mkdir(stage / "checkpoints", mode=0o700)
                _write_exact_file(stage / _HEADER_FILE, content)
                _write_exact_file(stage / _INITIAL_FILE, initial.state_bytes)
                _write_exact_file(stage / _LOCK_FILE, b"1")
                if (
                    _bounded_read(stage / _HEADER_FILE, _HEADER_LIMIT) != content
                    or _bounded_read(stage / _INITIAL_FILE, 512 * 1024**2) != initial.state_bytes
                ):
                    raise ValueError("Staged original ablation state differs")
                rename_directory_with_windows_retry(stage, target)
                stage = None
        with _claim(target):
            snapshot = read_scientific_ablation_execution_v1(target)
            if (
                snapshot.header != header
                or snapshot.initial_state_bytes != initial.state_bytes
                or snapshot.events
            ):
                raise ValueError("Existing execution has different DATA or progress; use resume")
            return IrisScientificAblationExecutionV1(
                snapshot=snapshot, lifecycle=lifecycle, _token=_OPEN_TOKEN
            )
    except BaseException as error:
        primary = error
        if _has_signal(error):
            raise
        raise IrisScientificAblationExecutionError("Ablation execution creation failed") from None
    finally:
        # Only this invocation's newly created stage; never sweep unknown/history files.
        try:
            if stage is not None and stage.exists():
                _regular(stage, directory=True)
                if _member_names(stage, 5) - {
                    _HEADER_FILE,
                    _INITIAL_FILE,
                    _LOCK_FILE,
                    "events",
                    "checkpoints",
                }:
                    raise IrisScientificAblationExecutionError(
                        "Owned initial staging changed; preserve it"
                    )
                for name in (_HEADER_FILE, _INITIAL_FILE, _LOCK_FILE):
                    (stage / name).unlink(missing_ok=True)
                for name in ("events", "checkpoints"):
                    with suppress(FileNotFoundError):
                        (stage / name).rmdir()
                stage.rmdir()
        except BaseException:
            if primary is not None and _has_signal(primary):
                raise primary from None
            raise


def resume_scientific_ablation_execution_v1(
    path: Path, *, lifecycle: IrisScientificAblationLifecycleV1, maximum_replay_phases: int = 4096
) -> IrisScientificAblationExecutionV1:
    """Audit, restore last checkpoint and exactly reconstruct the bounded committed tail."""
    mutated = False
    try:
        if type(maximum_replay_phases) is not int or not 0 <= maximum_replay_phases <= _MAX_EVENTS:
            raise ValueError("Explicit bounded committed-tail replay ceiling required")
        snapshot = read_scientific_ablation_execution_v1(path)
        _fresh(lifecycle)
        with _claim(snapshot.path) as claim:
            snapshot = read_scientific_ablation_execution_v1(snapshot.path)
            header = snapshot.header
            initial = build_scientific_ablation_checkpoint_v1(lifecycle._run)
            if (
                initial.manifest != header.payload.initial
                or initial.state_bytes != snapshot.initial_state_bytes
                or lifecycle.configuration_json != header.payload.lifecycle_configuration_json
                or lifecycle.configuration_sha256 != header.payload.lifecycle_configuration_sha256
            ):
                raise ValueError(
                    "Fresh factory does not reproduce the exact initial ablation state"
                )
            checkpoints = [
                event
                for event in snapshot.events
                if type(event.payload.record) is ScientificAblationExecutionCheckpointV1
            ]
            previous = checkpoints[-1] if checkpoints else None
            start = 0 if previous is None else previous.payload.sequence
            if len(snapshot.events) - start > maximum_replay_phases:
                raise ValueError("Committed tail exceeds the explicitly allowed replay ceiling")
            claim.check()
            if previous is not None:
                best = None
                for event in snapshot.events[:start]:
                    best = _selected_best(best, event)
                mutated = True
                restore_scientific_ablation_lifecycle_v1(
                    _checkpoint_for_event(snapshot.path, header, previous, best_event=best),
                    controller=lifecycle,
                )
            # Without a checkpoint the fresh initial bytes already match.
            # Do not fabricate a lifecycle boundary or restore capability.
            for expected in snapshot.events[start:]:
                claim.check()
                mutated = True
                result = lifecycle.advance()
                if isinstance(result, IrisScientificAblationLifecycleReceiptV1):
                    raise ValueError("Committed tail must not publish external checkpoints")
                actual = _make_pending(header, previous, lifecycle, result)
                if actual.event != expected or actual.checkpoint is not None:
                    raise ValueError("Committed ablation tail replay differs")
                previous = actual.event
            claim.check()
            return IrisScientificAblationExecutionV1(
                snapshot=snapshot, lifecycle=lifecycle, _token=_OPEN_TOKEN
            )
    except BaseException as error:
        if type(lifecycle) is IrisScientificAblationLifecycleV1 and (mutated or lifecycle.poisoned):
            lifecycle._poison()
        if _has_signal(error):
            raise
        raise IrisScientificAblationExecutionError("Ablation execution resume failed") from None


__all__ = [
    "IrisScientificAblationExecutionError",
    "IrisScientificAblationExecutionSnapshotV1",
    "IrisScientificAblationExecutionV1",
    "create_scientific_ablation_execution_v1",
    "read_scientific_ablation_execution_v1",
    "resume_scientific_ablation_execution_v1",
]
