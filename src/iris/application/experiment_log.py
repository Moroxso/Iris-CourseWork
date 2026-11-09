"""Canonical append-only local experiment runs and metric event chains."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from re import fullmatch
from typing import Any, Literal
from uuid import uuid4

import torch

from iris.application.errors import IrisExperimentLogError
from iris.application.experiment_execution import (
    IrisExperimentEpochProgressV1,
    analyze_experiment_epoch_prefix_v1,
    extend_experiment_epoch_progress_v1,
    validate_experiment_epoch_candidate_v1,
)
from iris.application.experiment_manifest import (
    IrisPreparedExperimentExample,
    validate_experiment_manifest_v1,
)
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    build_teacher_checkpoint_v1,
    validate_teacher_checkpoint_bundle_v1,
    write_teacher_checkpoint_v1,
)
from iris.artifacts.checkpoint_v1 import (
    TeacherAdamWConfigV1,
    TeacherBackboneRuntimeV1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentCheckpointReferenceV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentEventPayloadV1,
    ExperimentEventSchemaV1,
    ExperimentExecutionPolicyV1,
    ExperimentManifestReferenceV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentModelConfigurationV1,
    ExperimentOptimizerConfigurationV1,
    ExperimentRunPayloadV1,
    ExperimentRunSchemaV1,
    ExperimentRuntimePolicyV1,
    ExperimentTargetPreparationPolicyV1,
    ExperimentTrainingConfigurationV1,
    ExperimentTrainingStepMetricV1,
    IrisExperimentEventV1,
    IrisExperimentRunV1,
)
from iris.artifacts.experiment_v1 import ExperimentSplitV1, IrisExperimentManifestV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training import (
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherModelCheckpointProvenance,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerProvenance,
    IrisTokenWeightedEvaluationSummary,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_CONFIGURATION_DOMAIN = "iris/experiment-log/v1/configuration"
_RUN_PAYLOAD_DOMAIN = "iris/experiment-log/v1/run-payload"
_EVENT_PAYLOAD_DOMAIN = "iris/experiment-log/v1/event-payload"
_RUN_FILENAME = "run.json"
_EVENTS_DIRECTORY = "events"
_MAX_RUN_BYTES = 1024 * 1024
_MAX_EVENT_BYTES = 64 * 1024
_MAX_EVENT_COUNT = 1_000_000


@dataclass(frozen=True, slots=True)
class IrisExperimentLogCreateResult:
    """Installed identity of one immutable run root and its initial checkpoint."""

    path: Path
    run_payload_sha256: str
    configuration_sha256: str
    initial_checkpoint_path: Path
    initial_checkpoint_payload_sha256: str
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class IrisExperimentEventWriteResult:
    """Installed identity of one immutable event in a run chain."""

    path: Path
    sequence: int
    event_type: Literal["training_step", "evaluation", "checkpoint"]
    payload_sha256: str
    reused_existing: bool
    referenced_checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class IrisExperimentLogSnapshotV1:
    """One fully verified immutable snapshot of a local experiment journal."""

    path: Path
    run: IrisExperimentRunV1
    events: tuple[IrisExperimentEventV1, ...]

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def optimizer_completed_steps(self) -> int:
        if self.events:
            return self.events[-1].payload.optimizer_completed_steps
        return self.run.payload.initial_checkpoint.completed_steps

    @property
    def sidecar_state_sha256(self) -> str:
        if self.events:
            return self.events[-1].payload.sidecar_state_sha256
        return self.run.payload.initial_checkpoint.sidecar_state_sha256

    @property
    def chain_head_sha256(self) -> str:
        if self.events:
            return self.events[-1].payload_sha256
        return self.run.payload_sha256


@dataclass(frozen=True, slots=True)
class IrisExperimentTrainingStateV1:
    """Validated identity of one live teacher state against an immutable run."""

    configuration_sha256: str
    chain_head_sha256: str
    event_count: int
    optimizer_completed_steps: int
    sidecar_state_sha256: str


@dataclass(frozen=True, slots=True)
class _LiveTrainingIdentity:
    configuration: ExperimentTrainingConfigurationV1
    optimizer_completed_steps: int
    sidecar_state_sha256: str


def _configuration_content(configuration: ExperimentTrainingConfigurationV1) -> dict[str, Any]:
    return {
        "execution_policy": configuration.execution_policy.model_dump(mode="json"),
        "model": configuration.model.model_dump(mode="json"),
        "optimizer": configuration.optimizer.model_dump(mode="json"),
        "runtime_policy": configuration.runtime_policy.model_dump(mode="json"),
        "target_preparation": configuration.target_preparation.model_dump(mode="json"),
    }


def _owned_execution_policy(
    policy: ExperimentExecutionPolicyV1,
) -> ExperimentExecutionPolicyV1:
    if type(policy) is ExperimentManualExecutionPolicyV1:
        return ExperimentManualExecutionPolicyV1.model_validate(policy.model_dump(mode="python"))
    if type(policy) is ExperimentEpochExecutionPolicyV1:
        return ExperimentEpochExecutionPolicyV1.model_validate(policy.model_dump(mode="python"))
    message = "execution_policy must be an exact supported experiment policy"
    raise TypeError(message)


def _configuration(
    *,
    model: ExperimentModelConfigurationV1,
    optimizer: ExperimentOptimizerConfigurationV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
) -> ExperimentTrainingConfigurationV1:
    policy = ExperimentTargetPreparationPolicyV1(
        policy="raw_target_no_special_tokens_no_truncation_plus_one_eos",
        terminal_token_id=terminal_token_id,
    )
    owned_execution_policy = _owned_execution_policy(execution_policy)
    content = {
        "execution_policy": owned_execution_policy.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
        "optimizer": optimizer.model_dump(mode="json"),
        "runtime_policy": ExperimentRuntimePolicyV1(
            initialization_seed=initialization_seed,
            deterministic=deterministic,
        ).model_dump(mode="json"),
        "target_preparation": policy.model_dump(mode="json"),
    }
    return ExperimentTrainingConfigurationV1(
        sha256=domain_sha256(_CONFIGURATION_DOMAIN, canonical_json_bytes(content)),
        model=model,
        optimizer=optimizer,
        target_preparation=policy,
        runtime_policy=ExperimentRuntimePolicyV1(
            initialization_seed=initialization_seed,
            deterministic=deterministic,
        ),
        execution_policy=owned_execution_policy,
    )


def _validate_configuration(
    configuration: ExperimentTrainingConfigurationV1,
) -> ExperimentTrainingConfigurationV1:
    owned = ExperimentTrainingConfigurationV1.model_validate(
        configuration.model_dump(mode="python")
    )
    expected = domain_sha256(
        _CONFIGURATION_DOMAIN,
        canonical_json_bytes(_configuration_content(owned)),
    )
    if owned.sha256 != expected:
        message = "experiment training configuration SHA-256 is inconsistent"
        raise ValueError(message)
    return owned


def _validate_ambient_runtime_policy(policy: ExperimentRuntimePolicyV1) -> None:
    try:
        actual_seed = torch.initial_seed()
        actual_deterministic = torch.are_deterministic_algorithms_enabled()
        actual_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    except Exception as error:
        message = "experiment logger could not inspect the ambient torch runtime policy"
        raise ValueError(message) from error
    if (
        type(actual_seed) is not int
        or actual_seed != policy.initialization_seed
        or type(actual_deterministic) is not bool
        or actual_deterministic is not policy.deterministic
        or (policy.deterministic and actual_warn_only)
    ):
        message = "ambient torch seed or strict deterministic policy disagrees with the run"
        raise ValueError(message)


def _checkpoint_reference(
    bundle: IrisTeacherCheckpointBundleV1,
) -> tuple[IrisTeacherCheckpointBundleV1, ExperimentCheckpointReferenceV1]:
    verified = validate_teacher_checkpoint_bundle_v1(bundle)
    payload = verified.manifest.payload
    return verified, ExperimentCheckpointReferenceV1(
        payload_sha256=verified.manifest.sha256,
        state_file_sha256=payload.state_file.sha256,
        sidecar_state_sha256=payload.model.sha256,
        completed_steps=payload.optimizer.completed_steps,
    )


def _configuration_from_checkpoint(
    bundle: IrisTeacherCheckpointBundleV1,
    *,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
) -> tuple[IrisTeacherCheckpointBundleV1, ExperimentTrainingConfigurationV1]:
    verified = validate_teacher_checkpoint_bundle_v1(bundle)
    checkpoint = verified.manifest.payload
    model = ExperimentModelConfigurationV1(
        backbone=checkpoint.model.backbone,
        cognitive=checkpoint.model.cognitive,
        workspace=checkpoint.model.workspace,
        runtime=checkpoint.model.runtime,
        state_names=checkpoint.model.state_names,
    )
    optimizer = ExperimentOptimizerConfigurationV1(
        optimizer_type=checkpoint.optimizer.optimizer_type,
        config=checkpoint.optimizer.config,
        parameter_names=checkpoint.optimizer.parameter_names,
    )
    return verified, _configuration(
        model=model,
        optimizer=optimizer,
        terminal_token_id=terminal_token_id,
        initialization_seed=initialization_seed,
        deterministic=deterministic,
        execution_policy=execution_policy,
    )


def _run_bytes(run: IrisExperimentRunV1) -> tuple[IrisExperimentRunV1, bytes]:
    if type(run) is not IrisExperimentRunV1:
        message = "experiment run must be exactly IrisExperimentRunV1"
        raise TypeError(message)
    owned = IrisExperimentRunV1.model_validate(run.model_dump(mode="python"))
    _validate_configuration(owned.payload.configuration)
    payload_bytes = canonical_json_bytes(owned.payload.model_dump(mode="json"))
    expected = domain_sha256(_RUN_PAYLOAD_DOMAIN, payload_bytes)
    if owned.payload_sha256 != expected:
        message = "experiment run payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_RUN_BYTES:
        message = "experiment run record exceeds the 1 MiB safety limit"
        raise ValueError(message)
    return owned, content


def build_experiment_run_v1(
    *,
    run_id: str,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
    initial_checkpoint: IrisTeacherCheckpointBundleV1,
) -> IrisExperimentRunV1:
    """Build one immutable run identity without copying manifest text or tensors."""
    try:
        if type(terminal_token_id) is not int or type(initialization_seed) is not int:
            message = "terminal_token_id and initialization_seed must be exact integers"
            raise TypeError(message)
        if type(deterministic) is not bool:
            message = "deterministic must be an exact boolean"
            raise TypeError(message)
        owned_manifest = validate_experiment_manifest_v1(manifest)
        verified_checkpoint, configuration = _configuration_from_checkpoint(
            initial_checkpoint,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=execution_policy,
        )
        _, checkpoint_reference = _checkpoint_reference(verified_checkpoint)
        payload = ExperimentRunPayloadV1(
            run_id=run_id,
            manifest=ExperimentManifestReferenceV1(
                dataset_id=owned_manifest.payload.dataset_id,
                payload_sha256=owned_manifest.payload_sha256,
                split_counts=owned_manifest.payload.split_counts,
            ),
            configuration=configuration,
            initial_checkpoint=checkpoint_reference,
        )
        run = IrisExperimentRunV1(
            schema_identity=ExperimentRunSchemaV1(
                artifact_type="iris.experiment_run",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                journal_layout="immutable_run_and_chained_events",
            ),
            payload_sha256=domain_sha256(
                _RUN_PAYLOAD_DOMAIN,
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        owned, _ = _run_bytes(run)
        return owned
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment run could not be built"
        raise IrisExperimentLogError(message) from error


def _parse_canonical_json_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        message = f"{label} must be a real regular file"
        raise ValueError(message)
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        message = f"{label} file size is invalid"
        raise ValueError(message)
    content = path.read_bytes()
    if len(content) != size:
        message = f"{label} size changed while reading"
        raise ValueError(message)
    if (
        not content.endswith(b"\n")
        or content.endswith(b"\n\n")
        or content.startswith(b"\xef\xbb\xbf")
    ):
        message = f"{label} must be canonical UTF-8 JSON with one final LF"
        raise ValueError(message)
    decoded = content[:-1].decode("utf-8", errors="strict")
    raw = json.loads(
        decoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON number is forbidden: {value}")
        ),
    )
    if type(raw) is not dict or canonical_json_file_bytes(raw) != content:
        message = f"{label} is not canonical iris-json-v1"
        raise ValueError(message)
    return raw, content


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        value[key] = item
    return value


def _event_bytes(event: IrisExperimentEventV1) -> tuple[IrisExperimentEventV1, bytes]:
    if type(event) is not IrisExperimentEventV1:
        message = "experiment event must be exactly IrisExperimentEventV1"
        raise TypeError(message)
    owned = IrisExperimentEventV1.model_validate(event.model_dump(mode="python"))
    payload_bytes = canonical_json_bytes(owned.payload.model_dump(mode="json"))
    expected = domain_sha256(_EVENT_PAYLOAD_DOMAIN, payload_bytes)
    if owned.payload_sha256 != expected:
        message = "experiment event payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_EVENT_BYTES:
        message = "experiment event exceeds the 64 KiB safety limit"
        raise ValueError(message)
    return owned, content


def _validate_event_against_prefix(
    event: IrisExperimentEventV1,
    *,
    run: IrisExperimentRunV1,
    previous: IrisExperimentEventV1 | None,
) -> None:
    payload = event.payload
    run_payload = run.payload
    expected_sequence = 1 if previous is None else previous.payload.sequence + 1
    expected_predecessor = run.payload_sha256 if previous is None else previous.payload_sha256
    previous_steps = (
        run_payload.initial_checkpoint.completed_steps
        if previous is None
        else previous.payload.optimizer_completed_steps
    )
    previous_state = (
        run_payload.initial_checkpoint.sidecar_state_sha256
        if previous is None
        else previous.payload.sidecar_state_sha256
    )
    if (
        payload.sequence != expected_sequence
        or payload.run_id != run_payload.run_id
        or payload.run_payload_sha256 != run.payload_sha256
        or payload.manifest_payload_sha256 != run_payload.manifest.payload_sha256
        or payload.configuration_sha256 != run_payload.configuration.sha256
        or payload.predecessor_sha256 != expected_predecessor
    ):
        message = "experiment event identity or predecessor chain is inconsistent"
        raise ValueError(message)

    record = payload.record
    if isinstance(record, ExperimentTrainingStepMetricV1):
        if (
            payload.optimizer_completed_steps != previous_steps + 1
            or record.split_example_count != run_payload.manifest.split_counts.train
        ):
            message = "training event does not describe the next committed run step"
            raise ValueError(message)
    elif (
        payload.optimizer_completed_steps != previous_steps
        or payload.sidecar_state_sha256 != previous_state
    ):
        message = "non-training event cannot change the recorded training state"
        raise ValueError(message)

    if isinstance(record, ExperimentEvaluationMetricV1):
        expected_count = getattr(run_payload.manifest.split_counts, record.split)
        if expected_count == 0 or record.example_count != expected_count:
            message = "evaluation metric does not cover its complete nonempty manifest split"
            raise ValueError(message)


def _read_run(path: Path) -> tuple[IrisExperimentRunV1, bytes]:
    _, content = _parse_canonical_json_file(
        path,
        maximum_bytes=_MAX_RUN_BYTES,
        label="experiment run record",
    )
    run = IrisExperimentRunV1.model_validate_json(content[:-1])
    owned, expected = _run_bytes(run)
    if expected != content:
        message = "experiment run canonical round trip changed its bytes"
        raise ValueError(message)
    return owned, content


def _event_filename(sequence: int) -> str:
    if type(sequence) is not int or sequence < 1 or sequence > _MAX_EVENT_COUNT:
        message = "experiment event sequence is outside the supported journal range"
        raise ValueError(message)
    return f"{sequence:08d}.json"


def _read_event_entry(
    entry: Path,
    *,
    expected_sequence: int,
    run: IrisExperimentRunV1,
    previous: IrisExperimentEventV1 | None,
) -> IrisExperimentEventV1:
    if entry.name != _event_filename(expected_sequence):
        message = "experiment event filenames must form one contiguous prefix"
        raise ValueError(message)
    _, content = _parse_canonical_json_file(
        entry,
        maximum_bytes=_MAX_EVENT_BYTES,
        label=f"experiment event {expected_sequence}",
    )
    event = IrisExperimentEventV1.model_validate_json(content[:-1])
    owned, expected_content = _event_bytes(event)
    if expected_content != content:
        message = "experiment event canonical round trip changed its bytes"
        raise ValueError(message)
    _validate_event_against_prefix(owned, run=run, previous=previous)
    return owned


def read_experiment_log_v1(path: Path) -> IrisExperimentLogSnapshotV1:
    """Read one exact run directory and validate its complete event prefix."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "experiment log reader requires an explicit run directory Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_dir():
            message = "experiment log path must be a real directory, not a symlink"
            raise ValueError(message)
        initial_entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in initial_entries) != (_EVENTS_DIRECTORY, _RUN_FILENAME):
            message = "experiment run directory must contain exactly events and run.json"
            raise ValueError(message)
        events_path = path / _EVENTS_DIRECTORY
        if events_path.is_symlink() or not events_path.is_dir():
            message = "experiment events path must be a real directory"
            raise ValueError(message)
        run, _ = _read_run(path / _RUN_FILENAME)
        if path.name != run.payload.run_id:
            message = "experiment run directory name does not match run_id"
            raise ValueError(message)

        initial_event_entries = tuple(sorted(events_path.iterdir(), key=lambda entry: entry.name))
        if len(initial_event_entries) > _MAX_EVENT_COUNT:
            message = "experiment journal exceeds the one-million-event safety limit"
            raise ValueError(message)
        events: list[IrisExperimentEventV1] = []
        previous: IrisExperimentEventV1 | None = None
        for expected_sequence, entry in enumerate(initial_event_entries, start=1):
            owned = _read_event_entry(
                entry,
                expected_sequence=expected_sequence,
                run=run,
                previous=previous,
            )
            events.append(owned)
            previous = owned

        final_entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        final_event_entries = tuple(sorted(events_path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in final_entries) != tuple(
            entry.name for entry in initial_entries
        ) or tuple(entry.name for entry in final_event_entries) != tuple(
            entry.name for entry in initial_event_entries
        ):
            message = "experiment journal changed while reading"
            raise ValueError(message)
        return IrisExperimentLogSnapshotV1(path=path, run=run, events=tuple(events))
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment log could not be read or verified"
        raise IrisExperimentLogError(message) from error


def refresh_experiment_log_snapshot_v1(
    snapshot: IrisExperimentLogSnapshotV1,
) -> IrisExperimentLogSnapshotV1:
    """Verify the cached head and extend one immutable snapshot with its new suffix."""
    try:
        if type(snapshot) is not IrisExperimentLogSnapshotV1:
            message = "experiment snapshot refresh requires exactly IrisExperimentLogSnapshotV1"
            raise TypeError(message)
        path = snapshot.path
        if path.is_symlink() or not path.is_dir():
            message = "experiment log path must remain a real directory"
            raise ValueError(message)
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in entries) != (_EVENTS_DIRECTORY, _RUN_FILENAME):
            message = "experiment run directory changed after its verified snapshot"
            raise ValueError(message)
        events_path = path / _EVENTS_DIRECTORY
        if events_path.is_symlink() or not events_path.is_dir():
            message = "experiment events path must remain a real directory"
            raise ValueError(message)
        run, _ = _read_run(path / _RUN_FILENAME)
        if run != snapshot.run or path.name != run.payload.run_id:
            message = "experiment run identity changed after its verified snapshot"
            raise ValueError(message)
        if snapshot.event_count > _MAX_EVENT_COUNT:
            message = "experiment snapshot exceeds the event safety limit"
            raise ValueError(message)

        events = list(snapshot.events)
        previous = events[-1] if events else None
        if previous is not None:
            verified_head = _read_event_entry(
                events_path / _event_filename(snapshot.event_count),
                expected_sequence=snapshot.event_count,
                run=run,
                previous=events[-2] if len(events) > 1 else None,
            )
            if verified_head != previous:
                message = "experiment journal head changed after its verified snapshot"
                raise ValueError(message)

        while len(events) < _MAX_EVENT_COUNT:
            sequence = len(events) + 1
            entry = events_path / _event_filename(sequence)
            if entry.is_symlink():
                message = "experiment event suffix cannot contain a symlink"
                raise ValueError(message)
            if not entry.exists():
                break
            owned = _read_event_entry(
                entry,
                expected_sequence=sequence,
                run=run,
                previous=previous,
            )
            events.append(owned)
            previous = owned

        if len(events) == snapshot.event_count:
            return snapshot
        return IrisExperimentLogSnapshotV1(path=path, run=run, events=tuple(events))
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment log snapshot could not be refreshed"
        raise IrisExperimentLogError(message) from error


class IrisExperimentLogWriterV1:
    """Explicit process-local cache for one verified immutable journal prefix."""

    __slots__ = ("_poisoned", "_snapshot")

    def __init__(self, snapshot: IrisExperimentLogSnapshotV1) -> None:
        if type(snapshot) is not IrisExperimentLogSnapshotV1:
            message = "experiment writer requires exactly IrisExperimentLogSnapshotV1"
            raise TypeError(message)
        self._snapshot = snapshot
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        """Whether an uncertain persistence or refresh failure invalidated this writer."""
        return self._poisoned

    @property
    def snapshot(self) -> IrisExperimentLogSnapshotV1:
        """Return the latest verified prefix held by this writer."""
        if self._poisoned:
            message = "experiment writer is poisoned and must be reopened"
            raise IrisExperimentLogError(message)
        return self._snapshot

    def refresh(self) -> IrisExperimentLogSnapshotV1:
        """Read and verify only the cached head plus newly installed contiguous events."""
        if self._poisoned:
            message = "experiment writer is poisoned and must be reopened"
            raise IrisExperimentLogError(message)
        try:
            self._snapshot = refresh_experiment_log_snapshot_v1(self._snapshot)
            return self._snapshot
        except BaseException:
            self._poisoned = True
            raise

    def _accept(self, snapshot: IrisExperimentLogSnapshotV1) -> None:
        if self._poisoned:
            message = "experiment writer is poisoned and cannot accept a new snapshot"
            raise IrisExperimentLogError(message)
        if (
            type(snapshot) is not IrisExperimentLogSnapshotV1
            or snapshot.path != self._snapshot.path
            or snapshot.run != self._snapshot.run
            or snapshot.event_count < self._snapshot.event_count
        ):
            self._poisoned = True
            message = "experiment writer received an invalid snapshot advance"
            raise IrisExperimentLogError(message)
        self._snapshot = snapshot

    def _poison(self) -> None:
        self._poisoned = True


def open_experiment_log_writer_v1(path: Path) -> IrisExperimentLogWriterV1:
    """Open one writer from a complete verified snapshot of an installed journal."""
    return IrisExperimentLogWriterV1(read_experiment_log_v1(path))


def _write_exact_file(path: Path, content: bytes) -> None:
    file_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "wb", buffering=0) as stream:
            remaining = memoryview(content)
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0:
                    message = "Experiment log write made no forward progress"
                    raise OSError(message)
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(file_descriptor)
        raise


def _cleanup_run_temporary(path: Path) -> None:
    # A completed rename can lose its acknowledgement; never touch its target.
    if not path.exists():
        return
    (path / _RUN_FILENAME).unlink(missing_ok=True)
    (path / _EVENTS_DIRECTORY).rmdir()
    path.rmdir()


def _existing_run_matches(path: Path, run_bytes: bytes) -> bool:
    try:
        snapshot = read_experiment_log_v1(path)
        _, actual = _run_bytes(snapshot.run)
    except IrisExperimentLogError:
        return False
    return actual == run_bytes


def _write_run(run: IrisExperimentRunV1, *, root: Path) -> tuple[Path, bool]:
    if not isinstance(root, Path) or not root.name:
        message = "experiment log writer requires an explicit root Path"
        raise TypeError(message)
    owned, content = _run_bytes(run)
    parent = root / "experiments" / "v1"
    target = parent / owned.payload.run_id
    temporary_path: Path | None = None
    try:
        if target.exists():
            if (
                target.is_symlink()
                or not target.is_dir()
                or not _existing_run_matches(target, content)
            ):
                message = "Existing experiment run conflicts with expected bytes"
                raise IrisExperimentLogError(message)
            return target, True
        parent.mkdir(parents=True, exist_ok=True)
        staging_candidate = parent / f".iris-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary_path = staging_candidate
        os.mkdir(temporary_path / _EVENTS_DIRECTORY, mode=0o700)
        _write_exact_file(temporary_path / _RUN_FILENAME, content)
        if (temporary_path / _RUN_FILENAME).read_bytes() != content:
            message = "Experiment run temporary bytes failed verification"
            raise OSError(message)
        try:
            rename_directory_with_windows_retry(temporary_path, target)
        except OSError as error:
            if not target.exists() or not _existing_run_matches(target, content):
                message = "Concurrent experiment run installation conflicted"
                raise IrisExperimentLogError(message) from error
            _cleanup_run_temporary(temporary_path)
            temporary_path = None
            return target, True
        temporary_path = None
        return target, False
    except BaseException as primary_error:
        if temporary_path is not None and temporary_path.exists():
            try:
                _cleanup_run_temporary(temporary_path)
            except BaseException as cleanup_error:
                failures = BaseExceptionGroup(
                    "Experiment run persistence and cleanup both failed",
                    [primary_error, cleanup_error],
                )
                message = "Iris experiment run failed and cleanup was incomplete"
                raise IrisExperimentLogError(message) from failures
        if isinstance(primary_error, IrisExperimentLogError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris experiment run could not be installed atomically"
            raise IrisExperimentLogError(message) from primary_error
        raise


def create_experiment_log_v1(
    *,
    run_id: str,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    root: Path,
) -> IrisExperimentLogCreateResult:
    """Persist the exact initial checkpoint and atomically create one run journal."""
    try:
        if not isinstance(root, Path) or not root.name:
            message = "experiment log creation requires an explicit root Path"
            raise TypeError(message)
        requested_runtime = ExperimentRuntimePolicyV1(
            initialization_seed=initialization_seed,
            deterministic=deterministic,
        )
        _validate_ambient_runtime_policy(requested_runtime)
        checkpoint = build_teacher_checkpoint_v1(model=model, optimizer=optimizer)
        run = build_experiment_run_v1(
            run_id=run_id,
            manifest=manifest,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=execution_policy,
            initial_checkpoint=checkpoint,
        )
        checkpoint_result = write_teacher_checkpoint_v1(checkpoint, root=root)
        path, reused = _write_run(run, root=root)
        snapshot = read_experiment_log_v1(path)
        if snapshot.run != run:
            message = "installed experiment run does not match the requested identity"
            raise ValueError(message)
        return IrisExperimentLogCreateResult(
            path=path,
            run_payload_sha256=run.payload_sha256,
            configuration_sha256=run.payload.configuration.sha256,
            initial_checkpoint_path=checkpoint_result.path,
            initial_checkpoint_payload_sha256=checkpoint_result.payload_sha256,
            reused_existing=reused,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment log could not be created"
        raise IrisExperimentLogError(message) from error


def _model_configuration_from_provenance(
    provenance: IrisTeacherModelCheckpointProvenance,
    *,
    state_names: tuple[str, ...],
) -> ExperimentModelConfigurationV1:
    if type(provenance) is not IrisTeacherModelCheckpointProvenance:
        message = "teacher model returned invalid experiment-log provenance"
        raise TypeError(message)
    backbone = BackboneConfig.model_validate_json(provenance.backbone_configuration_json)
    cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    workspace = WorkspaceConfig.model_validate_json(provenance.workspace_configuration_json)
    runtime = TeacherBackboneRuntimeV1.model_validate(
        {
            "hidden_size": provenance.hidden_size,
            "num_hidden_layers": provenance.num_hidden_layers,
            "vocabulary_size": provenance.vocabulary_size,
            "max_position_embeddings": provenance.max_position_embeddings,
            "max_prompt_tokens": provenance.max_prompt_tokens,
            "max_target_tokens": provenance.max_target_tokens,
            "device": provenance.device,
            "dtype": provenance.dtype,
        }
    )
    return ExperimentModelConfigurationV1(
        backbone=backbone,
        cognitive=cognitive,
        workspace=workspace,
        runtime=runtime,
        state_names=state_names,
    )


def _optimizer_configuration_from_provenance(
    provenance: IrisTeacherOptimizerProvenance,
) -> ExperimentOptimizerConfigurationV1:
    if type(provenance) is not IrisTeacherOptimizerProvenance:
        message = "teacher optimizer returned invalid experiment-log provenance"
        raise TypeError(message)
    config = provenance.config
    return ExperimentOptimizerConfigurationV1(
        optimizer_type="torch.optim.AdamW",
        config=TeacherAdamWConfigV1(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
            max_gradient_norm=config.max_gradient_norm,
        ),
        parameter_names=provenance.parameter_names,
    )


def _live_training_identity(
    *,
    run: IrisExperimentRunV1,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> _LiveTrainingIdentity:
    if not isinstance(model, IrisTeacherForcingModel):
        message = "experiment logging requires an IrisTeacherForcingModel"
        raise TypeError(message)
    if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
        message = "experiment logging requires the exact optimizer owner for its model"
        raise TypeError(message)
    _validate_ambient_runtime_policy(run.payload.configuration.runtime_policy)
    before = model.checkpoint_provenance
    optimizer_provenance = optimizer.provenance_snapshot()
    state_names = tuple(model.state_dict())
    after = model.checkpoint_provenance
    if before != after or optimizer.completed_steps != optimizer_provenance.completed_steps:
        message = "teacher state changed during experiment-log observation"
        raise ValueError(message)
    configuration = _configuration(
        model=_model_configuration_from_provenance(after, state_names=state_names),
        optimizer=_optimizer_configuration_from_provenance(optimizer_provenance),
        terminal_token_id=run.payload.configuration.target_preparation.terminal_token_id,
        initialization_seed=run.payload.configuration.runtime_policy.initialization_seed,
        deterministic=run.payload.configuration.runtime_policy.deterministic,
        execution_policy=run.payload.configuration.execution_policy,
    )
    if configuration != run.payload.configuration:
        message = "live teacher configuration does not match the experiment run"
        raise ValueError(message)
    return _LiveTrainingIdentity(
        configuration=configuration,
        optimizer_completed_steps=optimizer_provenance.completed_steps,
        sidecar_state_sha256=after.sidecar_state_sha256,
    )


def _manifest_for_run(
    manifest: IrisExperimentManifestV1,
    run: IrisExperimentRunV1,
) -> IrisExperimentManifestV1:
    owned = validate_experiment_manifest_v1(manifest)
    reference = run.payload.manifest
    if (
        owned.payload.dataset_id != reference.dataset_id
        or owned.payload_sha256 != reference.payload_sha256
        or owned.payload.split_counts != reference.split_counts
    ):
        message = "experiment manifest does not match the run identity"
        raise ValueError(message)
    return owned


def _require_expected_chain_head(
    snapshot: IrisExperimentLogSnapshotV1,
    expected_predecessor_sha256: str | None,
) -> None:
    if expected_predecessor_sha256 is None:
        return
    if (
        type(expected_predecessor_sha256) is not str
        or fullmatch(r"[0-9a-f]{64}", expected_predecessor_sha256) is None
    ):
        message = "expected_predecessor_sha256 must be one canonical SHA-256"
        raise TypeError(message)
    if snapshot.chain_head_sha256 != expected_predecessor_sha256:
        message = "experiment chain head changed before the requested append"
        raise ValueError(message)


def _validate_managed_candidate(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    manifest: IrisExperimentManifestV1 | None,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1,
    epoch_progress: IrisExperimentEpochProgressV1 | None,
) -> None:
    policy = snapshot.run.payload.configuration.execution_policy
    if type(policy) is ExperimentManualExecutionPolicyV1:
        if epoch_progress is not None:
            message = "manual experiment append cannot use managed epoch progress"
            raise TypeError(message)
        return
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "experiment run contains an unsupported execution policy"
        raise TypeError(message)
    if manifest is None:
        message = "managed experiment append requires its exact manifest"
        raise TypeError(message)
    if epoch_progress is None:
        cursor = analyze_experiment_epoch_prefix_v1(
            run=snapshot.run,
            events=snapshot.events,
            manifest=manifest,
        )
    else:
        current = extend_experiment_epoch_progress_v1(
            progress=epoch_progress,
            run=snapshot.run,
            events=snapshot.events,
        )
        if (
            current.verified_event_count != snapshot.event_count
            or current.chain_head_sha256 != snapshot.chain_head_sha256
        ):
            message = "managed epoch progress does not cover the experiment snapshot"
            raise ValueError(message)
        cursor = current.cursor
    validate_experiment_epoch_candidate_v1(cursor=cursor, record=record)


def inspect_experiment_training_state_v1(
    *,
    path: Path,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisExperimentTrainingStateV1:
    """Validate and identify one live teacher state against an installed run."""
    try:
        snapshot = read_experiment_log_v1(path)
        return inspect_experiment_training_snapshot_v1(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment training state could not be inspected"
        raise IrisExperimentLogError(message) from error


def inspect_experiment_training_snapshot_v1(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisExperimentTrainingStateV1:
    """Identify one live teacher state against an already verified journal snapshot."""
    try:
        if type(snapshot) is not IrisExperimentLogSnapshotV1:
            message = "training-state inspection requires exactly IrisExperimentLogSnapshotV1"
            raise TypeError(message)
        live = _live_training_identity(
            run=snapshot.run,
            model=model,
            optimizer=optimizer,
        )
        return IrisExperimentTrainingStateV1(
            configuration_sha256=live.configuration.sha256,
            chain_head_sha256=snapshot.chain_head_sha256,
            event_count=snapshot.event_count,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris experiment training snapshot could not be inspected"
        raise IrisExperimentLogError(message) from error


def _prepared_token_count(
    prepared: IrisPreparedExperimentExample,
    *,
    configuration: ExperimentTrainingConfigurationV1,
) -> int:
    teacher_input = prepared.teacher_input
    if type(teacher_input) is not IrisTeacherForcingInput:
        message = "prepared experiment example contains an invalid teacher input"
        raise TypeError(message)
    input_ids = teacher_input.target_input_ids
    attention_mask = teacher_input.target_attention_mask
    runtime = configuration.model.runtime
    terminal_token_id = configuration.target_preparation.terminal_token_id
    if (
        type(input_ids) is not torch.Tensor
        or type(attention_mask) is not torch.Tensor
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != attention_mask.shape
        or input_ids.dtype != torch.int64
        or attention_mask.dtype != torch.bool
        or str(input_ids.device) != runtime.device
        or attention_mask.device != input_ids.device
        or input_ids.shape[1] < 2
        or input_ids.shape[1] > runtime.max_target_tokens
        or not bool(torch.all(attention_mask == 1))
        or not bool(torch.all((input_ids >= 0) & (input_ids < runtime.vocabulary_size)))
        or int(input_ids[0, -1].item()) != terminal_token_id
    ):
        message = "prepared teacher target disagrees with the run tokenization policy"
        raise ValueError(message)
    return int(attention_mask.sum().item())


def _new_event(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1,
) -> IrisExperimentEventV1:
    payload = ExperimentEventPayloadV1(
        sequence=snapshot.event_count + 1,
        run_id=snapshot.run.payload.run_id,
        run_payload_sha256=snapshot.run.payload_sha256,
        manifest_payload_sha256=snapshot.run.payload.manifest.payload_sha256,
        configuration_sha256=snapshot.run.payload.configuration.sha256,
        predecessor_sha256=snapshot.chain_head_sha256,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
        record=record,
    )
    event = IrisExperimentEventV1(
        schema_identity=ExperimentEventSchemaV1(
            artifact_type="iris.experiment_event",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=domain_sha256(
            _EVENT_PAYLOAD_DOMAIN,
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    owned, _ = _event_bytes(event)
    previous = snapshot.events[-1] if snapshot.events else None
    _validate_event_against_prefix(owned, run=snapshot.run, previous=previous)
    return owned


def _event_type(
    event: IrisExperimentEventV1,
) -> Literal["training_step", "evaluation", "checkpoint"]:
    return event.payload.record.event_type


def _event_result(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    event: IrisExperimentEventV1,
    reused_existing: bool,
    checkpoint_path: Path | None,
) -> IrisExperimentEventWriteResult:
    return IrisExperimentEventWriteResult(
        path=snapshot.path / _EVENTS_DIRECTORY / _event_filename(event.payload.sequence),
        sequence=event.payload.sequence,
        event_type=_event_type(event),
        payload_sha256=event.payload_sha256,
        reused_existing=reused_existing,
        referenced_checkpoint_path=checkpoint_path,
    )


def _managed_head_duplicate(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    expected_predecessor_sha256: str | None,
) -> IrisExperimentEventV1 | None:
    if (
        type(snapshot.run.payload.configuration.execution_policy)
        is (ExperimentManualExecutionPolicyV1)
        or not snapshot.events
    ):
        return None
    head = snapshot.events[-1]
    payload = head.payload
    if (
        payload.record != record
        or payload.optimizer_completed_steps != optimizer_completed_steps
        or payload.sidecar_state_sha256 != sidecar_state_sha256
    ):
        return None
    if expected_predecessor_sha256 is not None:
        if (
            type(expected_predecessor_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", expected_predecessor_sha256) is None
        ):
            message = "expected_predecessor_sha256 must be one canonical SHA-256"
            raise TypeError(message)
        if payload.predecessor_sha256 != expected_predecessor_sha256:
            return None
    return head


def _find_duplicate(
    snapshot: IrisExperimentLogSnapshotV1,
    *,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
) -> IrisExperimentEventV1 | None:
    for event in reversed(snapshot.events):
        payload = event.payload
        if payload.optimizer_completed_steps < optimizer_completed_steps:
            break
        if (
            payload.optimizer_completed_steps == optimizer_completed_steps
            and payload.sidecar_state_sha256 == sidecar_state_sha256
            and payload.record == record
        ):
            return event
    return None


def _install_event(
    snapshot: IrisExperimentLogSnapshotV1,
    event: IrisExperimentEventV1,
    *,
    incremental: bool,
) -> tuple[IrisExperimentEventWriteResult, IrisExperimentLogSnapshotV1]:
    owned, content = _event_bytes(event)
    events_path = snapshot.path / _EVENTS_DIRECTORY
    target = events_path / _event_filename(owned.payload.sequence)
    temporary_path = events_path / f".iris-{uuid4().hex}.tmp"
    try:
        _write_exact_file(temporary_path, content)
        if temporary_path.read_bytes() != content:
            message = "Experiment event temporary bytes failed verification"
            raise OSError(message)
        try:
            os.link(temporary_path, target)
            reused = False
        except FileExistsError as error:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                message = "Concurrent experiment event conflicts with the expected next event"
                raise IrisExperimentLogError(message) from error
            reused = True
        temporary_path.unlink()
        installed = (
            refresh_experiment_log_snapshot_v1(snapshot)
            if incremental
            else read_experiment_log_v1(snapshot.path)
        )
        matching = (
            installed.events[owned.payload.sequence - 1]
            if installed.event_count >= owned.payload.sequence
            else None
        )
        if matching != owned:
            message = "installed experiment event does not match the expected chain entry"
            raise ValueError(message)
        return (
            _event_result(
                snapshot=installed,
                event=owned,
                reused_existing=reused,
                checkpoint_path=None,
            ),
            installed,
        )
    except BaseException as primary_error:
        try:
            temporary_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "Experiment event persistence and cleanup both failed",
                [primary_error, cleanup_error],
            )
            message = "Iris experiment event failed and cleanup was incomplete"
            raise IrisExperimentLogError(message) from failures
        if isinstance(primary_error, IrisExperimentLogError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris experiment event could not be installed atomically"
            raise IrisExperimentLogError(message) from primary_error
        raise


def _record_or_reuse(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    checkpoint_path: Path | None = None,
    writer: IrisExperimentLogWriterV1 | None = None,
) -> IrisExperimentEventWriteResult:
    duplicate = _find_duplicate(
        snapshot,
        record=record,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
    )
    if duplicate is not None:
        return _event_result(
            snapshot=snapshot,
            event=duplicate,
            reused_existing=True,
            checkpoint_path=checkpoint_path,
        )
    event = _new_event(
        snapshot=snapshot,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
        record=record,
    )
    try:
        result, installed = _install_event(
            snapshot,
            event,
            incremental=writer is not None,
        )
        if writer is not None:
            writer._accept(installed)
    except BaseException:
        if writer is not None:
            writer._poison()
        raise
    if checkpoint_path is None:
        return result
    return IrisExperimentEventWriteResult(
        path=result.path,
        sequence=result.sequence,
        event_type=result.event_type,
        payload_sha256=result.payload_sha256,
        reused_existing=result.reused_existing,
        referenced_checkpoint_path=checkpoint_path,
    )


def _snapshot_for_record(
    *,
    path: Path,
    writer: IrisExperimentLogWriterV1 | None,
) -> IrisExperimentLogSnapshotV1:
    if writer is None:
        return read_experiment_log_v1(path)
    if type(writer) is not IrisExperimentLogWriterV1:
        message = "experiment append writer must be exactly IrisExperimentLogWriterV1"
        raise TypeError(message)
    snapshot = writer.refresh()
    if snapshot.path != path:
        message = "experiment append path does not match its writer"
        raise ValueError(message)
    return snapshot


def record_experiment_training_step_v1(
    *,
    path: Path,
    manifest: IrisExperimentManifestV1,
    prepared: IrisPreparedExperimentExample,
    output: IrisTeacherOptimizationOutput,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    expected_predecessor_sha256: str | None = None,
    writer: IrisExperimentLogWriterV1 | None = None,
    epoch_progress: IrisExperimentEpochProgressV1 | None = None,
) -> IrisExperimentEventWriteResult:
    """Append one committed train-step metric tied to exact example and live state."""
    try:
        snapshot = _snapshot_for_record(path=path, writer=writer)
        owned_manifest = _manifest_for_run(manifest, snapshot.run)
        if type(prepared) is not IrisPreparedExperimentExample:
            message = "training log requires exactly IrisPreparedExperimentExample"
            raise TypeError(message)
        if type(output) is not IrisTeacherOptimizationOutput:
            message = "training log requires exactly IrisTeacherOptimizationOutput"
            raise TypeError(message)
        if (
            prepared.dataset_id != owned_manifest.payload.dataset_id
            or prepared.manifest_payload_sha256 != owned_manifest.payload_sha256
            or prepared.split != "train"
            or prepared.split_example_count != owned_manifest.payload.split_counts.train
            or prepared.split_ordinal < 1
            or prepared.split_ordinal > prepared.split_example_count
        ):
            message = "prepared training example does not match the run manifest"
            raise ValueError(message)
        train_examples = tuple(
            example for example in owned_manifest.payload.examples if example.split == "train"
        )
        source = train_examples[prepared.split_ordinal - 1]
        if (
            source.example_id != prepared.example_id
            or source.prompt != prepared.teacher_input.prompt
        ):
            message = "prepared training example provenance is inconsistent"
            raise ValueError(message)
        prepared_token_count = _prepared_token_count(
            prepared,
            configuration=snapshot.run.payload.configuration,
        )
        live = _live_training_identity(run=snapshot.run, model=model, optimizer=optimizer)
        if (
            output.step_index != live.optimizer_completed_steps
            or output.supervised_token_count != prepared_token_count
        ):
            message = "optimization output does not describe the current committed step"
            raise ValueError(message)
        record = ExperimentTrainingStepMetricV1(
            event_type="training_step",
            split="train",
            example_id=prepared.example_id,
            split_ordinal=prepared.split_ordinal,
            split_example_count=prepared.split_example_count,
            step_index=output.step_index,
            loss=output.loss,
            supervised_token_count=output.supervised_token_count,
            gradient_l2_norm=output.gradient_l2_norm,
            gradients_clipped=output.gradients_clipped,
        )
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused_existing=True,
                checkpoint_path=None,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=owned_manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            writer=writer,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris training-step metric could not be recorded"
        raise IrisExperimentLogError(message) from error


def record_experiment_evaluation_v1(
    *,
    path: Path,
    manifest: IrisExperimentManifestV1,
    split: ExperimentSplitV1,
    summary: IrisTokenWeightedEvaluationSummary,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    expected_predecessor_sha256: str | None = None,
    writer: IrisExperimentLogWriterV1 | None = None,
    epoch_progress: IrisExperimentEpochProgressV1 | None = None,
) -> IrisExperimentEventWriteResult:
    """Append one complete validation or test summary at the current training state."""
    try:
        snapshot = _snapshot_for_record(path=path, writer=writer)
        owned_manifest = _manifest_for_run(manifest, snapshot.run)
        if split not in {"validation", "test"} or type(split) is not str:
            message = "evaluation log split must be exactly validation or test"
            raise ValueError(message)
        if type(summary) is not IrisTokenWeightedEvaluationSummary:
            message = "evaluation log requires exactly IrisTokenWeightedEvaluationSummary"
            raise TypeError(message)
        expected_count = getattr(owned_manifest.payload.split_counts, split)
        if expected_count == 0 or summary.example_count != expected_count:
            message = "evaluation summary must cover one complete nonempty manifest split"
            raise ValueError(message)
        live = _live_training_identity(run=snapshot.run, model=model, optimizer=optimizer)
        record = ExperimentEvaluationMetricV1(
            event_type="evaluation",
            split=split,
            example_count=summary.example_count,
            supervised_token_count=summary.supervised_token_count,
            mean_loss=summary.mean_loss,
            perplexity=summary.perplexity,
        )
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused_existing=True,
                checkpoint_path=None,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=owned_manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            writer=writer,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris evaluation metric could not be recorded"
        raise IrisExperimentLogError(message) from error


def _root_from_run_path(path: Path) -> Path:
    if (
        not isinstance(path, Path)
        or path.parent.name != "v1"
        or path.parent.parent.name != "experiments"
        or not path.parent.parent.parent.name
    ):
        message = "experiment run path is outside the canonical root layout"
        raise ValueError(message)
    return path.parent.parent.parent


def record_experiment_checkpoint_v1(
    *,
    path: Path,
    checkpoint: IrisTeacherCheckpointBundleV1,
    manifest: IrisExperimentManifestV1 | None = None,
    expected_predecessor_sha256: str | None = None,
    writer: IrisExperimentLogWriterV1 | None = None,
    epoch_progress: IrisExperimentEpochProgressV1 | None = None,
) -> IrisExperimentEventWriteResult:
    """Persist and append one exact checkpoint reference at the current chain state."""
    try:
        snapshot = _snapshot_for_record(path=path, writer=writer)
        verified, reference = _checkpoint_reference(checkpoint)
        _, configuration = _configuration_from_checkpoint(
            verified,
            terminal_token_id=(
                snapshot.run.payload.configuration.target_preparation.terminal_token_id
            ),
            initialization_seed=(
                snapshot.run.payload.configuration.runtime_policy.initialization_seed
            ),
            deterministic=snapshot.run.payload.configuration.runtime_policy.deterministic,
            execution_policy=snapshot.run.payload.configuration.execution_policy,
        )
        if configuration != snapshot.run.payload.configuration:
            message = "checkpoint configuration does not match the experiment run"
            raise ValueError(message)
        if (
            reference.completed_steps != snapshot.optimizer_completed_steps
            or reference.sidecar_state_sha256 != snapshot.sidecar_state_sha256
        ):
            message = "checkpoint does not describe the current experiment chain state"
            raise ValueError(message)
        record = ExperimentCheckpointEventV1(
            event_type="checkpoint",
            checkpoint=reference,
        )
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=reference.completed_steps,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            checkpoint_result = write_teacher_checkpoint_v1(
                verified,
                root=_root_from_run_path(snapshot.path),
            )
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused_existing=True,
                checkpoint_path=checkpoint_result.path,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        checkpoint_result = write_teacher_checkpoint_v1(
            verified,
            root=_root_from_run_path(snapshot.path),
        )
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=reference.completed_steps,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            checkpoint_path=checkpoint_result.path,
            writer=writer,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "Iris checkpoint event could not be recorded"
        raise IrisExperimentLogError(message) from error


__all__ = [
    "IrisExperimentEventWriteResult",
    "IrisExperimentLogCreateResult",
    "IrisExperimentLogSnapshotV1",
    "IrisExperimentLogWriterV1",
    "IrisExperimentTrainingStateV1",
    "build_experiment_run_v1",
    "create_experiment_log_v1",
    "inspect_experiment_training_snapshot_v1",
    "inspect_experiment_training_state_v1",
    "open_experiment_log_writer_v1",
    "read_experiment_log_v1",
    "record_experiment_checkpoint_v1",
    "record_experiment_evaluation_v1",
    "record_experiment_training_step_v1",
    "refresh_experiment_log_snapshot_v1",
]
