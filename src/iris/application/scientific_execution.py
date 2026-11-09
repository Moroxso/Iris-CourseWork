"""Durable scientific phase journal and exact fresh-owner replay from checkpoints.

Only explicit local corpora/graphs are used. This does not launch models, implement
AMP, reinterpret legacy experiment logs, or turn validation NLL into scientific
acceptance. The exclusive-owner store keeps committed records immutable.
"""

import os
import re
import shutil
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _bounded_read,
    _json,
    _regular,
    _verify,
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_lifecycle import (
    IrisScientificLifecycleBundleV1,
    IrisScientificLifecycleV1,
    read_scientific_lifecycle_checkpoint_v1,
    restore_scientific_lifecycle_v1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizationOutputV1,
    IrisScientificOptimizerPolicyV1,
    scientific_learning_rate_v1,
)
from iris.application.scientific_teacher import IrisScientificObjectivePolicyV1
from iris.application.scientific_validation import IrisScientificValidationV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_delivery_v1 import ScientificDeliveryValidationV1
from iris.artifacts.scientific_execution_v1 import (
    IrisScientificExecutionEventV1,
    IrisScientificExecutionHeaderV1,
    ScientificExecutionCheckpointV1,
    ScientificExecutionEventPayloadV1,
    ScientificExecutionHeaderPayloadV1,
    ScientificExecutionRecordV1,
    ScientificExecutionTrainingMetricV1,
    ScientificExecutionTrainV1,
    ScientificExecutionValidationV1,
)
from iris.artifacts.scientific_lifecycle_v1 import (
    IrisScientificLifecyclePolicyV1,
    ScientificLifecycleStateV1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_HEADER_DOMAIN = "iris/scientific-execution/v1"
_EVENT_DOMAIN = "iris/scientific-execution-event/v1"
_HEADER_LIMIT = 16 * 1024**2
_EVENT_LIMIT = 65536
_MAX_EVENTS = 200000
_INITIAL_FILE = "initial.safetensors"
_HEADER_FILE = "run.json"
_OPEN_TOKEN = object()


class IrisScientificExecutionError(RuntimeError):
    """Scientific execution journal, compatibility or durable phase failed."""


@dataclass(frozen=True, slots=True)
class IrisScientificExecutionSnapshotV1:
    path: Path
    header: IrisScientificExecutionHeaderV1
    initial_state_bytes: bytes
    events: tuple[IrisScientificExecutionEventV1, ...]
    journal_size_bytes: int


def _header_bytes(header: IrisScientificExecutionHeaderV1) -> bytes:
    header = IrisScientificExecutionHeaderV1.model_validate(header.model_dump(mode="python"))
    if (
        domain_sha256(_HEADER_DOMAIN, canonical_json_bytes(header.payload.model_dump(mode="json")))
        != header.sha256
    ):
        raise ValueError("Scientific execution header identity differs")
    data = canonical_json_file_bytes(header.model_dump(mode="json"))
    if len(data) > min(_HEADER_LIMIT, header.payload.maximum_journal_bytes):
        raise ValueError("Scientific execution header exceeds its byte budget")
    return data


def _context(
    header: IrisScientificExecutionHeaderV1,
) -> tuple[IrisScientificLifecyclePolicyV1, IrisScientificOptimizerPolicyV1, dict[str, object]]:
    _header_bytes(header)
    payload = header.payload
    initial = payload.initial.payload
    configuration = _json(payload.lifecycle_configuration_json.encode())
    run = _json(initial.configuration_json.encode())
    if (
        type(configuration) is not dict
        or set(configuration)
        != {"kind", "run_sha256", "validation_sha256", "validation_examples", "policy"}
        or configuration["kind"] != "iris-scientific-lifecycle-v1"
        or configuration["run_sha256"] != initial.configuration_sha256
        or type(configuration["validation_examples"]) is not int
        or not 1 <= configuration["validation_examples"] <= 5000
        or type(configuration["validation_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", configuration["validation_sha256"]) is None
        or domain_sha256(
            "iris/scientific-lifecycle-configuration/v1",
            payload.lifecycle_configuration_json.encode(),
        )
        != payload.lifecycle_configuration_sha256
        or initial.completed_steps
        or initial.examples_seen
        or initial.target_tokens_seen
        or initial.history_sha256
        != domain_sha256("iris/scientific-run-history/v1", initial.configuration_sha256.encode())
    ):
        raise ValueError("Scientific execution initial lifecycle configuration differs")
    lifecycle = IrisScientificLifecyclePolicyV1.model_validate(configuration["policy"])
    optimizer = IrisScientificOptimizerPolicyV1.model_validate(run["optimizer_policy"])
    if lifecycle.purpose != optimizer.purpose:
        raise ValueError("Execution policy purposes differ")
    if lifecycle.purpose == "scientific" and (
        configuration["validation_examples"] != 5000 or run["dataset_examples"] != 50000
    ):
        raise ValueError("Scientific execution requires source split sizes")
    return lifecycle, optimizer, configuration


def _event_bytes(event: IrisScientificExecutionEventV1) -> bytes:
    event = IrisScientificExecutionEventV1.model_validate(event.model_dump(mode="python"))
    if event.sha256 != domain_sha256(
        _EVENT_DOMAIN, canonical_json_bytes(event.payload.model_dump(mode="json"))
    ):
        raise ValueError("Scientific phase event identity differs")
    data = canonical_json_file_bytes(event.model_dump(mode="json"))
    if len(data) > _EVENT_LIMIT:
        raise ValueError("Scientific phase event is oversized")
    return data


def _transition(
    header: IrisScientificExecutionHeaderV1,
    previous: IrisScientificExecutionEventV1 | None,
    event: IrisScientificExecutionEventV1,
) -> None:
    """Independently derive cadence, metrics, counters and run history from the prefix."""
    lifecycle, optimizer, config = _context(header)
    _event_bytes(event)
    payload = event.payload
    before = ScientificLifecycleStateV1() if previous is None else previous.payload.after
    examples = 0 if previous is None else previous.payload.examples_seen
    tokens = 0 if previous is None else previous.payload.target_tokens_seen
    history = (
        header.payload.initial.payload.history_sha256
        if previous is None
        else previous.payload.history_sha256
    )
    if (
        payload.execution_sha256 != header.sha256
        or payload.sequence != (1 if previous is None else previous.payload.sequence + 1)
        or payload.previous_sha256 != (header.sha256 if previous is None else previous.sha256)
        or payload.before != before
        or before.phase == "stopped"
        or payload.record.kind != before.phase
    ):
        raise ValueError("Scientific phase event does not extend this exact prefix")
    values = before.model_dump(mode="python")
    run = _json(header.payload.initial.payload.configuration_json.encode())
    record = payload.record
    if type(record) is ScientificExecutionTrainV1:
        metric = record.metric
        step = before.step + 1
        weights = IrisScientificObjectivePolicyV1.model_validate(run["teacher_policy"]).weights
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
            or metric.configuration_sha256 != run["optimizer_sha256"]
            or metric.purpose != optimizer.purpose
            or metric.learning_rate != scientific_learning_rate_v1(optimizer, step=step)
            or not optimizer.minimum_target_tokens
            <= metric.supervised_token_count
            <= optimizer.maximum_target_tokens
            or metric.example_count > min(optimizer.maximum_examples, run["dataset_examples"])
            or metric.gradients_clipped != (metric.gradient_l2_norm > optimizer.max_gradient_norm)
            or metric.loss != loss
        ):
            raise ValueError("Scientific training metric contradicts its configured window")
        examples += metric.example_count
        tokens += metric.supervised_token_count
        history = domain_sha256(
            "iris/scientific-run-history/v1",
            canonical_json_bytes(
                {
                    "previous": history,
                    "result": metric.model_dump(mode="json"),
                    "examples_seen": examples,
                    "target_tokens_seen": tokens,
                }
            ),
        )
        values |= {
            "step": step,
            "phase": "validation"
            if step == optimizer.max_optimizer_steps or step % lifecycle.validate_every_steps == 0
            else "checkpoint"
            if step % lifecycle.checkpoint_every_steps == 0
            else "train",
        }
    elif type(record) is ScientificExecutionValidationV1:
        report = record.metric
        if (
            report.at_optimizer_step != before.step
            or report.run_configuration_sha256
            != header.payload.initial.payload.configuration_sha256
            or report.dataset_sha256 != config["validation_sha256"]
            or report.example_count != config["validation_examples"]
        ):
            raise ValueError("Scientific validation metric belongs to a different corpus/run/step")
        better = (
            before.best_nll_per_token is None
            or report.target_nll_per_token
            < before.best_nll_per_token - lifecycle.minimum_improvement
        )
        misses = 0 if better else before.nonimproving_validations + 1
        if better != (record.selected_checkpoint_sha256 is not None):
            raise ValueError("Best-state selection identity does not match validation improvement")
        reason = (
            "early_stopping"
            if misses >= lifecycle.early_stopping_patience
            else "step_budget"
            if before.step == optimizer.max_optimizer_steps
            else None
        )
        values |= {
            "phase": "checkpoint"
            if reason is not None or before.step % lifecycle.checkpoint_every_steps == 0
            else "train",
            "validation_count": before.validation_count + 1,
            "last_validation_step": before.step,
            "best_step": before.step if better else before.best_step,
            "best_nll_per_token": report.target_nll_per_token
            if better
            else before.best_nll_per_token,
            "nonimproving_validations": misses,
            "stop_reason": reason,
        }
    else:
        values |= {
            "phase": "stopped" if before.stop_reason is not None else "train",
            "checkpoint_count": before.checkpoint_count + 1,
            "last_checkpoint_step": before.step,
        }
    if payload.after != ScientificLifecycleStateV1.model_validate(values) or (
        payload.examples_seen,
        payload.target_tokens_seen,
        payload.history_sha256,
    ) != (examples, tokens, history):
        raise ValueError("Scientific phase state/cursor/history transition differs")


def _checkpoint_path(path: Path, identity: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("Explicit lifecycle content identity required")
    return path / "checkpoints" / "scientific-lifecycle-v1" / identity


def _checkpoint_for_event(
    path: Path,
    header: IrisScientificExecutionHeaderV1,
    event: IrisScientificExecutionEventV1,
    *,
    best_event: IrisScientificExecutionEventV1 | None,
) -> IrisScientificLifecycleBundleV1:
    record = event.payload.record
    if type(record) is not ScientificExecutionCheckpointV1:
        raise ValueError("Checkpoint phase required")
    checkpoint_path = _checkpoint_path(path, record.lifecycle_checkpoint_sha256)
    for parent in (checkpoint_path, *checkpoint_path.parents):
        _regular(parent, directory=True)
    bundle = read_scientific_lifecycle_checkpoint_v1(checkpoint_path)
    payload, current = bundle.manifest.payload, bundle.manifest.payload.current.payload
    if (
        payload.configuration_json != header.payload.lifecycle_configuration_json
        or payload.configuration_sha256 != header.payload.lifecycle_configuration_sha256
        or payload.state != event.payload.after
        or current.configuration_sha256 != header.payload.initial.payload.configuration_sha256
        or (current.examples_seen, current.target_tokens_seen, current.history_sha256)
        != (
            event.payload.examples_seen,
            event.payload.target_tokens_seen,
            event.payload.history_sha256,
        )
    ):
        raise ValueError("Scientific checkpoint reference contradicts its phase record")
    if payload.best is None:
        if best_event is not None:
            raise ValueError("Previously selected best checkpoint is missing")
    elif (
        best_event is None
        or type(best_event.payload.record) is not ScientificExecutionValidationV1
        or payload.best.sha256 != best_event.payload.record.selected_checkpoint_sha256
    ):
        raise ValueError("Saved best state differs from the exact validation-selected checkpoint")
    return bundle


def _selected_best(
    previous: IrisScientificExecutionEventV1 | None, event: IrisScientificExecutionEventV1
) -> IrisScientificExecutionEventV1 | None:
    record = event.payload.record
    if (
        type(record) is ScientificExecutionValidationV1
        and record.selected_checkpoint_sha256 is not None
    ):
        return event
    return previous


def _layout(path: Path) -> None:
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    expected = {_HEADER_FILE, _INITIAL_FILE, "events", "checkpoints"}
    entries = tuple(path.iterdir())
    if len(entries) > 1028 or not expected.issubset({entry.name for entry in entries}):
        raise ValueError("Scientific execution directory inventory differs")
    for entry in entries:
        if entry.name in expected:
            _regular(entry, directory=entry.name in {"events", "checkpoints"})
        elif re.fullmatch(r"\.iris-event-[0-9a-f]{32}\.tmp", entry.name):
            # Unacknowledged owned stages are not authoritative events or removed on open.
            _regular(entry, directory=True)
            for child in entry.iterdir():
                _regular(child, directory=False)
                if child.name != "event.json" or child.stat().st_size > _EVENT_LIMIT:
                    raise ValueError("Unexpected scientific event staging member")
        else:
            raise ValueError("Unexpected scientific execution entry")


def read_scientific_execution_v1(path: Path) -> IrisScientificExecutionSnapshotV1:
    """Full independent audit of header, initial state, chain, phases and checkpoint bytes."""
    try:
        if not isinstance(path, Path) or not path.name:
            raise ValueError("Explicit scientific execution directory required")
        path = path.absolute()
        _layout(path)
        raw = _bounded_read(path / _HEADER_FILE, _HEADER_LIMIT)
        header = IrisScientificExecutionHeaderV1.model_validate_json(raw)
        if _header_bytes(header) != raw or path.name != header.sha256:
            raise ValueError("Noncanonical scientific execution header/path")
        _context(header)
        initial_identity = _initial_stamp(path / _INITIAL_FILE)
        initial = _bounded_read(path / _INITIAL_FILE, 512 * 1024**2)
        _verify(IrisScientificCheckpointBundleV1(header.payload.initial, initial))
        entries = []
        for index, entry in enumerate((path / "events").iterdir()):
            if index >= _MAX_EVENTS:
                raise ValueError("Scientific event count budget exceeded")
            entries.append(entry)
        events = []
        previous = None
        best_event = None
        size = len(raw)
        for index, entry in enumerate(sorted(entries), start=1):
            if entry.name != f"{index:08d}.json":
                raise ValueError("Scientific event sequence must be contiguous")
            content = _bounded_read(
                entry, min(_EVENT_LIMIT, header.payload.maximum_journal_bytes - size)
            )
            event = IrisScientificExecutionEventV1.model_validate_json(content)
            if _event_bytes(event) != content:
                raise ValueError("Noncanonical scientific phase event")
            _transition(header, previous, event)
            if type(event.payload.record) is ScientificExecutionCheckpointV1:
                _checkpoint_for_event(path, header, event, best_event=best_event)
            best_event = _selected_best(best_event, event)
            events.append(event)
            size += len(content)
            previous = event
        if (
            set((path / "events").iterdir()) != set(entries)
            or _bounded_read(path / _HEADER_FILE, _HEADER_LIMIT) != raw
            or _initial_stamp(path / _INITIAL_FILE) != initial_identity
        ):
            raise ValueError("Scientific execution changed during the full audit")
        return IrisScientificExecutionSnapshotV1(path, header, initial, tuple(events), size)
    except Exception as error:
        raise IrisScientificExecutionError("Scientific execution could not be read") from error


def _cleanup_event_stage(path: Path | None) -> None:
    if path is not None and path.exists():
        _regular(path, directory=True)
        if any(child.name != "event.json" for child in path.iterdir()):
            raise ValueError("Unexpected owned event staging entry; preserve for inspection")
        (path / "event.json").unlink(missing_ok=True)
        path.rmdir()


def _append_event(
    path: Path,
    header: IrisScientificExecutionHeaderV1,
    previous: IrisScientificExecutionEventV1 | None,
    event: IrisScientificExecutionEventV1,
    *,
    journal_size_bytes: int,
    best_event: IrisScientificExecutionEventV1 | None,
) -> int:
    stage: Path | None = None
    try:
        _transition(header, previous, event)
        content = _event_bytes(event)
        if journal_size_bytes + len(content) > header.payload.maximum_journal_bytes:
            raise ValueError("Scientific journal byte budget exceeded")
        if type(event.payload.record) is ScientificExecutionCheckpointV1:
            _checkpoint_for_event(path, header, event, best_event=best_event)
        target = path / "events" / f"{event.payload.sequence:08d}.json"
        if target.exists():
            if _bounded_read(target, _EVENT_LIMIT) != content:
                raise ValueError("Committed scientific event conflicts")
            return journal_size_bytes + len(content)
        if shutil.disk_usage(path).free < len(content) + 1024**3:
            raise ValueError("Scientific journal free-space reserve exceeded")
        candidate = path / f".iris-event-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        stage = candidate  # Cleanup ownership starts only after successful creation.
        _write_exact_file(stage / "event.json", content)
        if _bounded_read(stage / "event.json", _EVENT_LIMIT) != content:
            raise ValueError("Staged scientific event bytes differ")
        try:
            # Exclusive same-filesystem publication; a committed event is never overwritten.
            os.link(stage / "event.json", target)
        except OSError:
            if not target.exists() or _bounded_read(target, _EVENT_LIMIT) != content:
                raise
        if _bounded_read(target, _EVENT_LIMIT) != content:
            raise ValueError("Published scientific event bytes differ")
        return journal_size_bytes + len(content)
    finally:
        _cleanup_event_stage(stage)


def _initial_stamp(path: Path) -> tuple[int, int, int, int]:
    _regular(path, directory=False)
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _make_event(
    header: IrisScientificExecutionHeaderV1,
    previous: IrisScientificExecutionEventV1 | None,
    lifecycle: IrisScientificLifecycleV1,
    result: IrisScientificOptimizationOutputV1 | IrisScientificValidationV1 | Path,
) -> IrisScientificExecutionEventV1:
    record: ScientificExecutionRecordV1
    if type(result) is IrisScientificOptimizationOutputV1:
        record = ScientificExecutionTrainV1(
            metric=ScientificExecutionTrainingMetricV1.model_validate(asdict(result))
        )
    elif type(result) is IrisScientificValidationV1:
        best = lifecycle.best_checkpoint
        old_step = None if previous is None else previous.payload.after.best_step
        if best is None:
            raise ValueError("Validation must retain its selected best checkpoint")
        record = ScientificExecutionValidationV1(
            metric=ScientificDeliveryValidationV1.model_validate(asdict(result)),
            selected_checkpoint_sha256=best.manifest.sha256
            if lifecycle.state.best_step != old_step
            else None,
        )
    elif isinstance(result, Path):
        record = ScientificExecutionCheckpointV1(lifecycle_checkpoint_sha256=result.name)
    else:
        raise ValueError("Unexpected scientific lifecycle phase result")
    run = lifecycle._run
    payload = ScientificExecutionEventPayloadV1(
        execution_sha256=header.sha256,
        sequence=1 if previous is None else previous.payload.sequence + 1,
        previous_sha256=header.sha256 if previous is None else previous.sha256,
        before=ScientificLifecycleStateV1() if previous is None else previous.payload.after,
        after=lifecycle.state,
        examples_seen=run.examples_seen,
        target_tokens_seen=run.target_tokens_seen,
        history_sha256=run._history_sha256,
        record=record,
    )
    event = IrisScientificExecutionEventV1(
        sha256=domain_sha256(_EVENT_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    _transition(header, previous, event)
    return event


class IrisScientificExecutionV1:
    """One opened scientific lifecycle and incremental append owner; use create/resume.

    Full open audits every record. An active exclusive owner checks its immutable
    header, initial-file identity, exact current event and next slot. It is not an
    authentication scheme or a concurrent/hostile-filesystem journal lock.
    """

    def __init__(
        self,
        *,
        snapshot: IrisScientificExecutionSnapshotV1,
        lifecycle: IrisScientificLifecycleV1,
        _token: object | None = None,
    ) -> None:
        if _token is not _OPEN_TOKEN:
            raise IrisScientificExecutionError("Use explicit scientific execution create/resume")
        self._path = snapshot.path
        self._header = snapshot.header
        self._header_content = _header_bytes(snapshot.header)
        self._initial_identity = _initial_stamp(snapshot.path / _INITIAL_FILE)
        self._head = snapshot.events[-1] if snapshot.events else None
        self._best_event: IrisScientificExecutionEventV1 | None = None
        for event in snapshot.events:
            self._best_event = _selected_best(self._best_event, event)
        self._journal_bytes = snapshot.journal_size_bytes
        self._lifecycle = lifecycle
        self._pending: IrisScientificExecutionEventV1 | None = None
        self._busy = False
        self._poisoned = False
        self._validate_live(self._head)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def header(self) -> IrisScientificExecutionHeaderV1:
        return self._header

    @property
    def last_event(self) -> IrisScientificExecutionEventV1 | None:
        return self._head

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _validate_live(self, expected: IrisScientificExecutionEventV1 | None) -> None:
        self._lifecycle._validate()
        run = self._lifecycle._run
        state = ScientificLifecycleStateV1() if expected is None else expected.payload.after
        counters = (
            (0, 0, self._header.payload.initial.payload.history_sha256)
            if expected is None
            else (
                expected.payload.examples_seen,
                expected.payload.target_tokens_seen,
                expected.payload.history_sha256,
            )
        )
        if (
            self._poisoned
            or self._lifecycle._busy
            or self._lifecycle._configuration_json
            != self._header.payload.lifecycle_configuration_json
            or self._lifecycle.state != state
            or (run.examples_seen, run.target_tokens_seen, run._history_sha256) != counters
        ):
            raise ValueError("Scientific execution lost lifecycle/cursor ownership")

    def _validate_head(self, *, pending: bool) -> None:
        for parent in (self._path, *self._path.parents):
            _regular(parent, directory=True)
        _regular(self._path / "events", directory=True)
        _regular(self._path / "checkpoints", directory=True)
        if (
            _bounded_read(self._path / _HEADER_FILE, _HEADER_LIMIT) != self._header_content
            or _initial_stamp(self._path / _INITIAL_FILE) != self._initial_identity
        ):
            raise ValueError("Scientific execution initial/header files changed")
        sequence = 0 if self._head is None else self._head.payload.sequence
        if self._head is not None and _bounded_read(
            self._path / "events" / f"{sequence:08d}.json", _EVENT_LIMIT
        ) != _event_bytes(self._head):
            raise ValueError("Scientific execution journal head changed")
        if sequence >= _MAX_EVENTS or (
            not pending and (self._path / "events" / f"{sequence + 1:08d}.json").exists()
        ):
            raise ValueError("Scientific execution journal has an unexpected next event")

    def advance(self) -> IrisScientificExecutionEventV1:
        """Commit one phase; after failed publication retry only its immutable event."""
        if self._busy or self._poisoned:
            raise IrisScientificExecutionError("Scientific execution is busy or poisoned")
        self._busy = True
        try:
            self._validate_live(self._pending if self._pending is not None else self._head)
            self._validate_head(pending=self._pending is not None)
            if self._pending is None:
                if self._lifecycle.state.phase == "stopped":
                    raise ValueError("Scientific execution is already stopped")
                # Reserve one maximum event BEFORE training, not after model mutation.
                if self._journal_bytes + _EVENT_LIMIT > self._header.payload.maximum_journal_bytes:
                    raise ValueError("Insufficient remaining journal budget for another phase")
                try:
                    result = self._lifecycle.advance(checkpoint_root=self._path / "checkpoints")
                except BaseException:
                    try:
                        self._validate_live(self._head)
                    except Exception:
                        self._poisoned = True
                    raise
                try:
                    self._pending = _make_event(self._header, self._head, self._lifecycle, result)
                except BaseException:
                    self._poisoned = True  # A phase changed, but no trustworthy DTO was retained.
                    raise
            event = self._pending
            self._journal_bytes = _append_event(
                self._path,
                self._header,
                self._head,
                event,
                journal_size_bytes=self._journal_bytes,
                best_event=self._best_event,
            )
            try:
                self._validate_live(event)
                self._validate_head(pending=True)
            except BaseException:
                self._poisoned = True
                raise
            self._head = event
            self._best_event = _selected_best(self._best_event, event)
            self._pending = None
            return event
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise IrisScientificExecutionError(
                "Scientific execution phase could not complete"
            ) from error
        finally:
            self._busy = False


def create_scientific_execution_v1(
    lifecycle: IrisScientificLifecycleV1,
    *,
    root: Path,
    maximum_journal_bytes: int = 128 * 1024**2,
) -> IrisScientificExecutionV1:
    """Persist an initial frozen run/journal before any scientific phase is executed."""
    stage: Path | None = None
    try:
        if (
            type(lifecycle) is not IrisScientificLifecycleV1
            or lifecycle._busy
            or lifecycle.state != ScientificLifecycleStateV1()
            or not isinstance(root, Path)
            or not root.name
        ):
            raise ValueError("Fresh idle scientific lifecycle and explicit root required")
        lifecycle._validate()
        initial = build_scientific_checkpoint_v1(lifecycle._run)
        payload = ScientificExecutionHeaderPayloadV1(
            lifecycle_configuration_json=lifecycle._configuration_json,
            lifecycle_configuration_sha256=lifecycle._configuration_sha256,
            initial=initial.manifest,
            maximum_journal_bytes=maximum_journal_bytes,
        )
        header = IrisScientificExecutionHeaderV1(
            sha256=domain_sha256(
                _HEADER_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
        content = _header_bytes(header)
        _context(header)
        if len(content) + _EVENT_LIMIT > maximum_journal_bytes:
            raise ValueError("Journal must have room for its header and one maximum event")
        parent = root.absolute() / "scientific-executions-v1"
        for ancestor in (parent, *parent.parents):
            with suppress(FileNotFoundError):
                _regular(ancestor, directory=True)
        parent.mkdir(parents=True, exist_ok=True)
        _regular(parent, directory=True)
        target = parent / header.sha256
        if not target.exists():
            if shutil.disk_usage(parent).free < len(content) + len(initial.state_bytes) + 1024**3:
                raise ValueError("Scientific initial publication free-space reserve exceeded")
            candidate = parent / f".iris-execution-{uuid4().hex}.tmp"
            os.mkdir(candidate, mode=0o700)
            stage = candidate
            os.mkdir(stage / "events", mode=0o700)
            os.mkdir(stage / "checkpoints", mode=0o700)
            _write_exact_file(stage / _HEADER_FILE, content)
            _write_exact_file(stage / _INITIAL_FILE, initial.state_bytes)
            if (
                _bounded_read(stage / _HEADER_FILE, _HEADER_LIMIT) != content
                or _bounded_read(stage / _INITIAL_FILE, 512 * 1024**2) != initial.state_bytes
            ):
                raise ValueError("Staged initial execution bytes differ")
            try:
                rename_directory_with_windows_retry(stage, target)
            except OSError:
                if not target.exists():
                    raise
            else:
                stage = None
        snapshot = read_scientific_execution_v1(target)
        if (
            snapshot.header != header
            or snapshot.initial_state_bytes != initial.state_bytes
            or snapshot.events
        ):
            raise ValueError("Existing execution differs or has progress; use explicit resume")
        return IrisScientificExecutionV1(snapshot=snapshot, lifecycle=lifecycle, _token=_OPEN_TOKEN)
    except Exception as error:
        raise IrisScientificExecutionError("Scientific execution could not be created") from error
    finally:
        if stage is not None and stage.exists():
            _regular(stage, directory=True)
            for name in (_HEADER_FILE, _INITIAL_FILE):
                (stage / name).unlink(missing_ok=True)
            for name in ("events", "checkpoints"):
                with suppress(FileNotFoundError):
                    (stage / name).rmdir()
            stage.rmdir()


def resume_scientific_execution_v1(
    path: Path, *, lifecycle: IrisScientificLifecycleV1
) -> IrisScientificExecutionV1:
    """Audit first, restore latest controller checkpoint, verify the exact phase tail."""
    mutated = False
    try:
        if (
            type(lifecycle) is not IrisScientificLifecycleV1
            or lifecycle._busy
            or lifecycle.state != ScientificLifecycleStateV1()
        ):
            raise ValueError("Fresh idle scientific lifecycle required for resume")
        lifecycle._validate()
        snapshot = read_scientific_execution_v1(path)
        header = snapshot.header
        if (
            lifecycle._configuration_json != header.payload.lifecycle_configuration_json
            or lifecycle._configuration_sha256 != header.payload.lifecycle_configuration_sha256
        ):
            raise ValueError("Scientific execution belongs to a different lifecycle/corpus")
        checkpoints = [
            event
            for event in snapshot.events
            if type(event.payload.record) is ScientificExecutionCheckpointV1
        ]
        previous = checkpoints[-1] if checkpoints else None
        if previous is None:
            restore_scientific_checkpoint_v1(
                IrisScientificCheckpointBundleV1(
                    header.payload.initial, snapshot.initial_state_bytes
                ),
                run=lifecycle._run,
            )
        else:
            best_event = None
            for event in snapshot.events[: previous.payload.sequence]:
                best_event = _selected_best(best_event, event)
            restore_scientific_lifecycle_v1(
                _checkpoint_for_event(snapshot.path, header, previous, best_event=best_event),
                lifecycle=lifecycle,
            )
        mutated = True
        start = 0 if previous is None else previous.payload.sequence
        for expected in snapshot.events[start:]:
            result = lifecycle.advance(checkpoint_root=snapshot.path / "checkpoints")
            actual = _make_event(header, previous, lifecycle, result)
            if actual != expected:
                raise ValueError("Scientific replay does not reproduce the committed phase event")
            previous = actual
        return IrisScientificExecutionV1(snapshot=snapshot, lifecycle=lifecycle, _token=_OPEN_TOKEN)
    except BaseException as error:
        if mutated or (type(lifecycle) is IrisScientificLifecycleV1 and lifecycle._run.poisoned):
            lifecycle._poisoned = True
            lifecycle._run._poisoned = True
            lifecycle._run._optimizer._poisoned = True
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise IrisScientificExecutionError("Scientific execution could not be resumed") from error


__all__ = [
    "IrisScientificExecutionError",
    "IrisScientificExecutionSnapshotV1",
    "IrisScientificExecutionV1",
    "create_scientific_execution_v1",
    "read_scientific_execution_v1",
    "resume_scientific_execution_v1",
]
