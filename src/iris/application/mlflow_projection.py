"""Deterministic, content-redacted MLflow projection of canonical experiment journals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from iris.application.errors import IrisMlflowProjectionError
from iris.application.experiment_execution import analyze_experiment_epoch_prefix_v1
from iris.application.experiment_log import IrisExperimentLogSnapshotV1, read_experiment_log_v1
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.verification import (
    IrisExperimentJournalVerificationReportV1,
    verify_experiment_journal_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1

MlflowProjectionStatusV1 = Literal["running", "finished"]

_PROJECTION_SCHEMA = "iris.mlflow_projection.v1"
_RESERVED_PREFIX = "iris."
_METRIC_BATCH_LIMIT = 500

_TAG_SCHEMA = "iris.projection.schema"
_TAG_RUN_ID = "iris.source.run_id"
_TAG_RUN_PAYLOAD = "iris.source.run_payload_sha256"
_TAG_CONFIGURATION = "iris.source.configuration_sha256"
_TAG_MANIFEST = "iris.source.manifest_payload_sha256"
_TAG_EVENT_COUNT = "iris.projection.event_count"
_TAG_CHAIN_HEAD = "iris.projection.chain_head_sha256"
_TAG_OPTIMIZER_STEPS = "iris.projection.optimizer_completed_steps"
_TAG_SIDECAR_STATE = "iris.projection.sidecar_state_sha256"
_TAG_LATEST_CHECKPOINT = "iris.projection.latest_checkpoint_payload_sha256"
_TAG_SOURCE_STATUS = "iris.projection.source_status"
_TAG_MANAGED_PHASE = "iris.projection.managed_phase"
_TAG_COMPLETED_EPOCHS = "iris.projection.completed_epochs"


@dataclass(frozen=True, slots=True)
class IrisMlflowProjectionReportV1:
    """Content-redacted private metadata for one reconciled MLflow projection."""

    path: Path
    experiment_name: str
    experiment_id: str
    mlflow_run_id: str
    canonical_run_id: str
    run_payload_sha256: str
    configuration_sha256: str
    manifest_payload_sha256: str
    event_count: int
    chain_head_sha256: str
    metric_point_count: int
    new_metric_point_count: int
    reused_metric_point_count: int
    reused_existing_experiment: bool
    reused_existing_run: bool
    status: MlflowProjectionStatusV1


@dataclass(frozen=True, slots=True)
class _MlflowMetricPointV1:
    key: str
    value: float = field(repr=False)
    timestamp: int
    step: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.timestamp, self.step


@dataclass(frozen=True, slots=True)
class _MlflowObservedMetricV1:
    key: str
    value: float = field(repr=False)
    timestamp: int
    step: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.timestamp, self.step


@dataclass(frozen=True, slots=True)
class _MlflowExperimentV1:
    experiment_id: str
    artifact_location: str
    lifecycle_stage: str


@dataclass(frozen=True, slots=True)
class _MlflowRunV1:
    run_id: str
    lifecycle_stage: str
    status: str
    params: Mapping[str, str]
    tags: Mapping[str, str]
    metric_keys: frozenset[str]


class _MlflowGatewayV1(Protocol):
    def ensure_experiment(
        self,
        *,
        name: str,
        artifact_location: str,
    ) -> tuple[_MlflowExperimentV1, bool]: ...

    def find_runs(
        self,
        *,
        experiment_id: str,
        run_payload_sha256: str,
    ) -> tuple[_MlflowRunV1, ...]: ...

    def create_run(
        self,
        *,
        experiment_id: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> _MlflowRunV1: ...

    def get_run(self, run_id: str) -> _MlflowRunV1: ...

    def get_metric_history(
        self,
        *,
        run_id: str,
        key: str,
    ) -> tuple[_MlflowObservedMetricV1, ...]: ...

    def log_batch(
        self,
        *,
        run_id: str,
        metrics: tuple[_MlflowMetricPointV1, ...] = (),
        params: Mapping[str, str] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None: ...

    def set_terminated(self, *, run_id: str, end_time: int) -> None: ...


def _open_mlflow_gateway(tracking_uri: str) -> _MlflowGatewayV1:
    from iris.integrations.mlflow_tracking import MlflowTrackingGatewayV1

    return MlflowTrackingGatewayV1(tracking_uri=tracking_uri)


def _explicit_text(value: object, *, label: str, maximum_length: int) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum_length
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        message = f"{label} must be one explicit bounded string without control characters"
        raise ValueError(message)
    return value


def _parameter_value(value: object) -> str:
    if type(value) is bool:
        rendered = str(value).lower()
    elif type(value) is float:
        if not math.isfinite(value):
            message = "MLflow parameter floats must be finite"
            raise ValueError(message)
        rendered = repr(value)
    elif type(value) in {int, str}:
        rendered = str(value)
    else:
        message = "MLflow parameter values must be exact scalars"
        raise TypeError(message)
    return _explicit_text(rendered, label="MLflow parameter value", maximum_length=5_000)


def _identity_tags(snapshot: IrisExperimentLogSnapshotV1) -> dict[str, str]:
    payload = snapshot.run.payload
    return {
        _TAG_SCHEMA: _PROJECTION_SCHEMA,
        _TAG_RUN_ID: payload.run_id,
        _TAG_RUN_PAYLOAD: snapshot.run.payload_sha256,
        _TAG_CONFIGURATION: payload.configuration.sha256,
        _TAG_MANIFEST: payload.manifest.payload_sha256,
    }


def _projection_params(snapshot: IrisExperimentLogSnapshotV1) -> dict[str, str]:
    payload = snapshot.run.payload
    manifest = payload.manifest
    configuration = payload.configuration
    model = configuration.model
    backbone = model.backbone
    cognitive = model.cognitive
    workspace = model.workspace
    runtime = model.runtime
    optimizer = configuration.optimizer
    optimizer_config = optimizer.config
    initial = payload.initial_checkpoint
    policy = configuration.execution_policy
    values: dict[str, object] = {
        "iris.source.artifact_type": snapshot.run.schema_identity.artifact_type,
        "iris.source.schema_version": snapshot.run.schema_identity.version,
        "iris.source.run_id": payload.run_id,
        "iris.source.run_payload_sha256": snapshot.run.payload_sha256,
        "iris.source.configuration_sha256": configuration.sha256,
        "iris.source.manifest_payload_sha256": manifest.payload_sha256,
        "iris.source.dataset_id": manifest.dataset_id,
        "iris.source.train_examples": manifest.split_counts.train,
        "iris.source.validation_examples": manifest.split_counts.validation,
        "iris.source.test_examples": manifest.split_counts.test,
        "iris.execution.mode": policy.mode,
        "iris.runtime.initialization_seed": configuration.runtime_policy.initialization_seed,
        "iris.runtime.deterministic": configuration.runtime_policy.deterministic,
        "iris.target.terminal_token_id": configuration.target_preparation.terminal_token_id,
        "iris.backbone.name": backbone.name,
        "iris.backbone.repository": backbone.repository,
        "iris.backbone.revision": backbone.revision,
        "iris.backbone.tokenizer_revision": backbone.tokenizer_revision,
        "iris.backbone.dtype": backbone.dtype,
        "iris.model.hidden_size": runtime.hidden_size,
        "iris.model.num_hidden_layers": runtime.num_hidden_layers,
        "iris.model.vocabulary_size": runtime.vocabulary_size,
        "iris.model.device": runtime.device,
        "iris.model.dtype": runtime.dtype,
        "iris.cognitive.state_dim": cognitive.state_dim,
        "iris.cognitive.num_modules": cognitive.num_modules,
        "iris.cognitive.layer_fractions": ",".join(
            repr(fraction) for fraction in cognitive.layer_sampler.layer_fractions
        ),
        "iris.workspace.num_slots": workspace.num_slots,
        "iris.workspace.cycles": workspace.cycles,
        "iris.workspace.winners_per_cycle": workspace.winners_per_cycle,
        "iris.workspace.decay_rate": workspace.decay_rate,
        "iris.workspace.empty_threshold": workspace.empty_threshold,
        "iris.optimizer.type": optimizer.optimizer_type,
        "iris.optimizer.learning_rate": optimizer_config.learning_rate,
        "iris.optimizer.beta1": optimizer_config.beta1,
        "iris.optimizer.beta2": optimizer_config.beta2,
        "iris.optimizer.epsilon": optimizer_config.epsilon,
        "iris.optimizer.weight_decay": optimizer_config.weight_decay,
        "iris.optimizer.max_gradient_norm": optimizer_config.max_gradient_norm,
        "iris.initial_checkpoint.payload_sha256": initial.payload_sha256,
        "iris.initial_checkpoint.completed_steps": initial.completed_steps,
        "iris.initial_checkpoint.sidecar_state_sha256": initial.sidecar_state_sha256,
    }
    if type(policy) is ExperimentEpochExecutionPolicyV1:
        values.update(
            {
                "iris.execution.epoch_count": policy.epoch_count,
                "iris.execution.evaluate_test_after_training": (
                    policy.evaluate_test_after_training
                ),
                "iris.execution.training_order": policy.training_order,
                "iris.execution.checkpoint_policy": policy.checkpoint_policy,
                "iris.execution.validation_policy": policy.validation_policy,
                "iris.execution.resume_policy": policy.resume_policy,
            }
        )
    elif type(policy) is not ExperimentManualExecutionPolicyV1:
        message = "MLflow projection received an unsupported execution policy"
        raise TypeError(message)
    return {key: _parameter_value(value) for key, value in values.items()}


def _metric_point(
    *,
    key: str,
    value: float | int | bool,
    sequence: int,
    step: int,
) -> _MlflowMetricPointV1:
    numeric = float(value)
    if not math.isfinite(numeric):
        message = "MLflow metric values must remain finite"
        raise ValueError(message)
    return _MlflowMetricPointV1(
        key=key,
        value=numeric,
        timestamp=sequence,
        step=step,
    )


def _projection_metrics(
    snapshot: IrisExperimentLogSnapshotV1,
) -> tuple[_MlflowMetricPointV1, ...]:
    points: list[_MlflowMetricPointV1] = []
    for event in snapshot.events:
        payload = event.payload
        sequence = payload.sequence
        step = payload.optimizer_completed_steps
        record = payload.record
        points.append(
            _metric_point(
                key="iris.event.marker",
                value=1,
                sequence=sequence,
                step=step,
            )
        )
        if type(record) is ExperimentTrainingStepMetricV1:
            for key, value in (
                ("iris.train.loss", record.loss),
                ("iris.train.gradient_l2_norm", record.gradient_l2_norm),
                ("iris.train.supervised_tokens", record.supervised_token_count),
                ("iris.train.gradients_clipped", record.gradients_clipped),
            ):
                points.append(
                    _metric_point(
                        key=key,
                        value=value,
                        sequence=sequence,
                        step=step,
                    )
                )
        elif type(record) is ExperimentEvaluationMetricV1:
            prefix = f"iris.{record.split}"
            values: list[tuple[str, float | int]] = [
                (f"{prefix}.mean_loss", record.mean_loss),
                (f"{prefix}.supervised_tokens", record.supervised_token_count),
                (f"{prefix}.example_count", record.example_count),
            ]
            if record.perplexity is not None:
                values.append((f"{prefix}.perplexity", record.perplexity))
            for key, value in values:
                points.append(
                    _metric_point(
                        key=key,
                        value=value,
                        sequence=sequence,
                        step=step,
                    )
                )
        elif type(record) is ExperimentCheckpointEventV1:
            points.append(
                _metric_point(
                    key="iris.checkpoint.marker",
                    value=1,
                    sequence=sequence,
                    step=step,
                )
            )
        else:
            message = "MLflow projection received an unsupported experiment event"
            raise TypeError(message)
    return tuple(points)


def _progress_tags(
    *,
    snapshot: IrisExperimentLogSnapshotV1,
    manifest: IrisExperimentManifestV1,
    event_count: int,
) -> dict[str, str]:
    if type(event_count) is not int or event_count < 0 or event_count > snapshot.event_count:
        message = "MLflow projection event cursor is outside the source prefix"
        raise ValueError(message)
    prefix = snapshot.events[:event_count]
    run = snapshot.run
    if prefix:
        last_payload = prefix[-1].payload
        chain_head = prefix[-1].payload_sha256
        optimizer_steps = last_payload.optimizer_completed_steps
        sidecar_state = last_payload.sidecar_state_sha256
    else:
        initial = run.payload.initial_checkpoint
        chain_head = run.payload_sha256
        optimizer_steps = initial.completed_steps
        sidecar_state = initial.sidecar_state_sha256

    policy = run.payload.configuration.execution_policy
    if type(policy) is ExperimentEpochExecutionPolicyV1:
        cursor = analyze_experiment_epoch_prefix_v1(
            run=run,
            events=prefix,
            manifest=manifest,
        )
        latest_checkpoint = cursor.latest_checkpoint.payload_sha256
        source_status: MlflowProjectionStatusV1 = (
            "finished" if cursor.phase == "complete" else "running"
        )
        managed_phase: str = cursor.phase
        completed_epochs = str(cursor.completed_epochs)
    elif type(policy) is ExperimentManualExecutionPolicyV1:
        latest_checkpoint = run.payload.initial_checkpoint.payload_sha256
        for event in prefix:
            record = event.payload.record
            if type(record) is ExperimentCheckpointEventV1:
                latest_checkpoint = record.checkpoint.payload_sha256
        source_status = "running"
        managed_phase = "not_applicable"
        completed_epochs = "not_applicable"
    else:
        message = "MLflow projection received an unsupported execution policy"
        raise TypeError(message)
    return {
        _TAG_EVENT_COUNT: str(event_count),
        _TAG_CHAIN_HEAD: chain_head,
        _TAG_OPTIMIZER_STEPS: str(optimizer_steps),
        _TAG_SIDECAR_STATE: sidecar_state,
        _TAG_LATEST_CHECKPOINT: latest_checkpoint,
        _TAG_SOURCE_STATUS: source_status,
        _TAG_MANAGED_PHASE: managed_phase,
        _TAG_COMPLETED_EPOCHS: completed_epochs,
    }


def _verified_projection_source(
    *,
    path: Path,
    manifest_path: Path,
) -> tuple[
    IrisExperimentJournalVerificationReportV1,
    IrisExperimentManifestV1,
    IrisExperimentLogSnapshotV1,
]:
    report = verify_experiment_journal_v1(path=path, manifest_path=manifest_path)
    manifest = read_experiment_manifest_v1(manifest_path)
    snapshot = read_experiment_log_v1(path)
    if (
        manifest.payload_sha256 != report.manifest_payload_sha256
        or snapshot.run.payload_sha256 != report.run_payload_sha256
        or snapshot.run.payload.configuration.sha256 != report.configuration_sha256
        or snapshot.run.payload.manifest.payload_sha256 != report.manifest_payload_sha256
        or snapshot.event_count != report.event_count
        or snapshot.chain_head_sha256 != report.chain_head_sha256
    ):
        message = "experiment source changed after referential verification"
        raise ValueError(message)
    return report, manifest, snapshot


def _require_allowed_namespace(
    values: Mapping[str, object],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unexpected = sorted(
        key for key in values if key.startswith(_RESERVED_PREFIX) and key not in allowed
    )
    if unexpected:
        message = f"MLflow run contains unexpected reserved {label}"
        raise ValueError(message)


def _missing_exact_values(
    *,
    actual: Mapping[str, str],
    expected: Mapping[str, str],
    label: str,
) -> dict[str, str]:
    missing: dict[str, str] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value is None:
            missing[key] = expected_value
        elif actual_value != expected_value:
            message = f"MLflow run contains conflicting Iris {label}"
            raise ValueError(message)
    return missing


def _source_heads(snapshot: IrisExperimentLogSnapshotV1) -> tuple[str, ...]:
    return (
        snapshot.run.payload_sha256,
        *(event.payload_sha256 for event in snapshot.events),
    )


def _existing_progress_count(
    *,
    run: _MlflowRunV1,
    snapshot: IrisExperimentLogSnapshotV1,
    manifest: IrisExperimentManifestV1,
) -> int:
    raw_count = run.tags.get(_TAG_EVENT_COUNT)
    raw_head = run.tags.get(_TAG_CHAIN_HEAD)
    heads = _source_heads(snapshot)
    if raw_count is None and raw_head is None:
        progress_keys = frozenset(
            _progress_tags(snapshot=snapshot, manifest=manifest, event_count=0)
        )
        if any(key in run.tags for key in progress_keys):
            message = "MLflow projection contains progress tags without a strong cursor"
            raise ValueError(message)
        count = 0
    elif raw_count is not None:
        try:
            count = int(raw_count)
        except ValueError as error:
            message = "MLflow projection event cursor is malformed"
            raise ValueError(message) from error
        if str(count) != raw_count or count < 0 or count > snapshot.event_count:
            message = "MLflow projection event cursor is outside the source prefix"
            raise ValueError(message)
        if raw_head is not None and heads[count] != raw_head:
            message = "MLflow projection cursor head conflicts with the canonical journal"
            raise ValueError(message)
    else:
        matching = tuple(index for index, head in enumerate(heads) if head == raw_head)
        if len(matching) != 1:
            message = "MLflow projection cursor head is not one unique source prefix"
            raise ValueError(message)
        count = matching[0]

    expected = _progress_tags(snapshot=snapshot, manifest=manifest, event_count=count)
    for key, expected_value in expected.items():
        actual_value = run.tags.get(key)
        if actual_value is not None and actual_value != expected_value:
            message = "MLflow projection progress tags conflict with the canonical prefix"
            raise ValueError(message)
    return count


def _expected_metrics_by_key(
    points: tuple[_MlflowMetricPointV1, ...],
) -> dict[str, dict[tuple[int, int], _MlflowMetricPointV1]]:
    by_key: dict[str, dict[tuple[int, int], _MlflowMetricPointV1]] = {}
    for point in points:
        identities = by_key.setdefault(point.key, {})
        if point.identity in identities:
            message = "MLflow projection plan contains a duplicate metric identity"
            raise ValueError(message)
        identities[point.identity] = point
    return by_key


def _missing_metric_points(
    *,
    gateway: _MlflowGatewayV1,
    run: _MlflowRunV1,
    expected_points: tuple[_MlflowMetricPointV1, ...],
) -> tuple[_MlflowMetricPointV1, ...]:
    expected_by_key = _expected_metrics_by_key(expected_points)
    expected_keys = frozenset(expected_by_key)
    _require_allowed_namespace(
        {key: object() for key in run.metric_keys},
        allowed=expected_keys,
        label="metric keys",
    )
    missing: list[_MlflowMetricPointV1] = []
    for key in sorted(expected_by_key):
        expected = expected_by_key[key]
        observed = gateway.get_metric_history(run_id=run.run_id, key=key)
        seen: set[tuple[int, int]] = set()
        for metric in observed:
            if metric.key != key or metric.identity in seen:
                message = "MLflow metric history contains a duplicate or mismatched identity"
                raise ValueError(message)
            seen.add(metric.identity)
            expected_metric = expected.get(metric.identity)
            if expected_metric is None or metric.value != expected_metric.value:
                message = "MLflow metric history conflicts with the canonical journal"
                raise ValueError(message)
        missing.extend(point for identity, point in expected.items() if identity not in seen)
    return tuple(sorted(missing, key=lambda point: (point.timestamp, point.key)))


def _metric_batches(
    points: tuple[_MlflowMetricPointV1, ...],
) -> tuple[tuple[_MlflowMetricPointV1, ...], ...]:
    batches: list[tuple[_MlflowMetricPointV1, ...]] = []
    current: list[_MlflowMetricPointV1] = []
    index = 0
    while index < len(points):
        sequence = points[index].timestamp
        group_end = index + 1
        while group_end < len(points) and points[group_end].timestamp == sequence:
            group_end += 1
        event_group = points[index:group_end]
        if len(event_group) > _METRIC_BATCH_LIMIT:
            message = "one MLflow event projection exceeds the metric batch limit"
            raise ValueError(message)
        if current and len(current) + len(event_group) > _METRIC_BATCH_LIMIT:
            batches.append(tuple(current))
            current = []
        current.extend(event_group)
        index = group_end
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _validate_run_identity(
    *,
    run: _MlflowRunV1,
    identity_tags: Mapping[str, str],
    params: Mapping[str, str],
    progress_keys: frozenset[str],
) -> dict[str, str]:
    if run.lifecycle_stage != "active":
        message = "MLflow projection run is not active"
        raise ValueError(message)
    if run.status not in {"RUNNING", "FINISHED"}:
        message = "MLflow projection run has an unsupported terminal status"
        raise ValueError(message)
    _require_allowed_namespace(
        run.params,
        allowed=frozenset(params),
        label="parameters",
    )
    _require_allowed_namespace(
        run.tags,
        allowed=frozenset(identity_tags) | progress_keys,
        label="tags",
    )
    _missing_exact_values(actual=run.tags, expected=identity_tags, label="identity tags")
    return _missing_exact_values(actual=run.params, expected=params, label="parameters")


def _require_source_stable(
    *,
    path: Path,
    manifest_path: Path,
    manifest: IrisExperimentManifestV1,
    snapshot: IrisExperimentLogSnapshotV1,
) -> None:
    final_manifest = read_experiment_manifest_v1(manifest_path)
    final_snapshot = read_experiment_log_v1(path)
    if (
        final_manifest != manifest
        or final_snapshot.run != snapshot.run
        or final_snapshot.event_count != snapshot.event_count
        or final_snapshot.chain_head_sha256 != snapshot.chain_head_sha256
    ):
        message = "experiment source changed while MLflow projection was running"
        raise ValueError(message)


def project_experiment_journal_to_mlflow_v1(
    *,
    path: Path,
    manifest_path: Path,
    tracking_uri: str,
    experiment_name: str,
    artifact_location: str,
) -> IrisMlflowProjectionReportV1:
    """Verify and reconcile one canonical journal into an explicit MLflow target."""
    try:
        owned_tracking_uri = _explicit_text(
            tracking_uri,
            label="MLflow tracking_uri",
            maximum_length=4_096,
        )
        owned_experiment_name = _explicit_text(
            experiment_name,
            label="MLflow experiment_name",
            maximum_length=256,
        )
        owned_artifact_location = _explicit_text(
            artifact_location,
            label="MLflow artifact_location",
            maximum_length=4_096,
        )
        _, manifest, snapshot = _verified_projection_source(
            path=path,
            manifest_path=manifest_path,
        )
        identity_tags = _identity_tags(snapshot)
        params = _projection_params(snapshot)
        metric_points = _projection_metrics(snapshot)
        current_progress = _progress_tags(
            snapshot=snapshot,
            manifest=manifest,
            event_count=snapshot.event_count,
        )
        progress_keys = frozenset(current_progress)
        gateway = _open_mlflow_gateway(owned_tracking_uri)
        experiment, reused_experiment = gateway.ensure_experiment(
            name=owned_experiment_name,
            artifact_location=owned_artifact_location,
        )
        if experiment.lifecycle_stage != "active":
            message = "MLflow projection experiment is not active"
            raise ValueError(message)
        matches = gateway.find_runs(
            experiment_id=experiment.experiment_id,
            run_payload_sha256=snapshot.run.payload_sha256,
        )
        if len(matches) > 1:
            message = "MLflow experiment contains multiple runs for one canonical journal"
            raise ValueError(message)
        if matches:
            run = matches[0]
            reused_run = True
        else:
            run = gateway.create_run(
                experiment_id=experiment.experiment_id,
                run_name=snapshot.run.payload.run_id,
                tags=identity_tags,
            )
            reused_run = False
            installed = gateway.find_runs(
                experiment_id=experiment.experiment_id,
                run_payload_sha256=snapshot.run.payload_sha256,
            )
            if len(installed) != 1 or installed[0].run_id != run.run_id:
                message = "concurrent MLflow run creation conflicted with projection identity"
                raise ValueError(message)

        missing_params = _validate_run_identity(
            run=run,
            identity_tags=identity_tags,
            params=params,
            progress_keys=progress_keys,
        )
        existing_count = _existing_progress_count(
            run=run,
            snapshot=snapshot,
            manifest=manifest,
        )
        source_status_value = current_progress[_TAG_SOURCE_STATUS]
        if source_status_value == "running":
            source_status: MlflowProjectionStatusV1 = "running"
        elif source_status_value == "finished":
            source_status = "finished"
        else:
            message = "MLflow projection produced an unsupported source status"
            raise ValueError(message)
        if run.status == "FINISHED" and (
            source_status != "finished" or existing_count != snapshot.event_count
        ):
            message = "MLflow run terminated before its canonical source prefix completed"
            raise ValueError(message)

        missing_metrics = _missing_metric_points(
            gateway=gateway,
            run=run,
            expected_points=metric_points,
        )
        if missing_params:
            gateway.log_batch(run_id=run.run_id, params=missing_params)
        for batch in _metric_batches(missing_metrics):
            gateway.log_batch(run_id=run.run_id, metrics=batch)
        gateway.log_batch(
            run_id=run.run_id,
            tags=identity_tags | current_progress,
        )

        reconciled = gateway.get_run(run.run_id)
        remaining_params = _validate_run_identity(
            run=reconciled,
            identity_tags=identity_tags,
            params=params,
            progress_keys=progress_keys,
        )
        if remaining_params:
            message = "MLflow projection parameters were not installed"
            raise ValueError(message)
        _missing_exact_values(
            actual=reconciled.tags,
            expected=current_progress,
            label="progress tags",
        )
        remaining_metrics = _missing_metric_points(
            gateway=gateway,
            run=reconciled,
            expected_points=metric_points,
        )
        if remaining_metrics:
            message = "MLflow projection metrics were not installed"
            raise ValueError(message)
        if source_status == "finished" and reconciled.status != "FINISHED":
            gateway.set_terminated(run_id=run.run_id, end_time=snapshot.event_count)
            reconciled = gateway.get_run(run.run_id)
        expected_mlflow_status = "FINISHED" if source_status == "finished" else "RUNNING"
        if reconciled.status != expected_mlflow_status:
            message = "MLflow run status disagrees with the canonical source status"
            raise ValueError(message)

        _require_source_stable(
            path=path,
            manifest_path=manifest_path,
            manifest=manifest,
            snapshot=snapshot,
        )
        return IrisMlflowProjectionReportV1(
            path=path,
            experiment_name=owned_experiment_name,
            experiment_id=experiment.experiment_id,
            mlflow_run_id=run.run_id,
            canonical_run_id=snapshot.run.payload.run_id,
            run_payload_sha256=snapshot.run.payload_sha256,
            configuration_sha256=snapshot.run.payload.configuration.sha256,
            manifest_payload_sha256=manifest.payload_sha256,
            event_count=snapshot.event_count,
            chain_head_sha256=snapshot.chain_head_sha256,
            metric_point_count=len(metric_points),
            new_metric_point_count=len(missing_metrics),
            reused_metric_point_count=len(metric_points) - len(missing_metrics),
            reused_existing_experiment=reused_experiment,
            reused_existing_run=reused_run,
            status=source_status,
        )
    except IrisMlflowProjectionError:
        raise
    except Exception as error:
        message = "Iris MLflow projection failed"
        raise IrisMlflowProjectionError(message) from error


__all__ = [
    "IrisMlflowProjectionReportV1",
    "MlflowProjectionStatusV1",
    "project_experiment_journal_to_mlflow_v1",
]
