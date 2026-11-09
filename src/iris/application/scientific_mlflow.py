"""Local metadata-only MLflow projection of the authoritative scientific journal.

No model construction, training, artifact upload or legacy-log reinterpretation.
Logical event sequence is the metric timestamp, NOT a measured wall-clock time.
"""

import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal

from iris.application.mlflow_projection import (
    _explicit_text,
    _metric_batches,
    _metric_point,
    _missing_exact_values,
    _missing_metric_points,
    _MlflowMetricPointV1,
    _MlflowRunV1,
    _parameter_value,
    _require_allowed_namespace,
)
from iris.application.scientific_checkpoint import _json, _regular
from iris.application.scientific_execution import (
    IrisScientificExecutionSnapshotV1,
    read_scientific_execution_v1,
)
from iris.application.scientific_optimization import IrisScientificOptimizerPolicyV1
from iris.application.scientific_teacher import IrisScientificObjectivePolicyV1
from iris.artifacts.scientific_execution_v1 import (
    ScientificExecutionCheckpointV1,
    ScientificExecutionTrainV1,
    ScientificExecutionValidationV1,
)
from iris.artifacts.scientific_lifecycle_v1 import (
    IrisScientificLifecyclePolicyV1,
    ScientificLifecycleStateV1,
)
from iris.integrations.mlflow_tracking import MlflowTrackingGatewayV1
from iris.utils.hashing import canonical_json_bytes
from iris.utils.process_file_lock import process_file_lock

_PREFIX = "iris.scientific."
_PROGRESS = _PREFIX + "progress_json"
_SCHEMA = "iris.scientific.mlflow.v1"
_RESERVE = 1024**3


class IrisScientificMlflowError(RuntimeError):
    """Projection failed; retain the source and retry the same explicit local target."""


@dataclass(frozen=True, slots=True)
class IrisScientificMlflowReportV1:
    execution_sha256: str
    event_count: int
    chain_head_sha256: str
    mlflow_run_id: str
    experiment_id: str
    tracking_database: Path
    metric_point_count: int
    new_metric_point_count: int
    reused_existing_run: bool
    status: Literal["running", "finished"]


def _local(path: Path) -> Path:
    if (
        not isinstance(path, Path)
        or not path.name
        or str(path).startswith(("//", "\\\\"))
        or any(c in str(path) for c in ("?", "#", "\x00"))
        or "://" in str(path)
    ):
        raise ValueError("Explicit local filesystem path required")
    windows = PureWindowsPath(str(path))
    if windows.drive and not windows.root:
        raise ValueError("Drive-relative path forbidden")
    owned = Path(os.path.abspath(path))
    for parent in (owned, *owned.parents):
        with suppress(FileNotFoundError):
            _regular(parent, directory=True)
    return owned


def _store_check(root: Path, maximum: int) -> None:
    """Observed byte check between SDK operations, not a peak-filesystem quota."""
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < _RESERVE:
        raise ValueError("Scientific MLflow requires a 1GiB free-space reserve")
    size = 0
    if root.exists():
        for index, entry in enumerate(root.iterdir()):
            if index > 8:
                raise ValueError("Unexpected MLflow store inventory")
            if entry.name == "artifacts":
                _regular(entry, directory=True)
                if next(entry.iterdir(), None) is not None:
                    raise ValueError("Metadata-only scientific MLflow artifacts must remain empty")
            elif entry.name in {
                "projection.lock",
                "tracking.sqlite",
                "tracking.sqlite-wal",
                "tracking.sqlite-shm",
                "tracking.sqlite-journal",
            }:
                _regular(entry, directory=False)
                info = entry.stat()
                if info.st_nlink != 1:
                    raise ValueError("Hard-linked MLflow database/ownership files forbidden")
                size += info.st_size
            else:
                raise ValueError("Unexpected MLflow store member")
    if size > maximum:
        raise ValueError("Scientific MLflow observed storage budget exceeded")


def _identity(snapshot: IrisScientificExecutionSnapshotV1) -> dict[str, str]:
    header = snapshot.header
    return {
        _PREFIX + "schema": _SCHEMA,
        _PREFIX + "execution_sha256": header.sha256,
        _PREFIX + "run_configuration_sha256": header.payload.initial.payload.configuration_sha256,
        _PREFIX + "lifecycle_configuration_sha256": header.payload.lifecycle_configuration_sha256,
        _PREFIX + "initial_checkpoint_sha256": header.payload.initial.sha256,
    }


def _params(snapshot: IrisScientificExecutionSnapshotV1) -> dict[str, str]:
    run = _json(snapshot.header.payload.initial.payload.configuration_json.encode())
    lifecycle = _json(snapshot.header.payload.lifecycle_configuration_json.encode())
    optimizer_policy = IrisScientificOptimizerPolicyV1.model_validate(run["optimizer_policy"])
    lifecycle_policy = IrisScientificLifecyclePolicyV1.model_validate(lifecycle["policy"])
    teacher_policy = IrisScientificObjectivePolicyV1.model_validate(run["teacher_policy"])
    # Explicit scalar allowlist. Never flatten arbitrary config, probe labels or raw rows.
    values = {
        key: run[key]
        for key in (
            "seed",
            "ordering",
            "dataset_sha256",
            "dataset_examples",
            "dataset_logical_bytes",
            "terminal_token_id",
            "optimizer_sha256",
            "backbone_runtime_sha256",
        )
    }
    values |= {"arm": "D", "precision": "float32", "metric_time": "event_sequence_not_wall_time"}
    values |= {
        "validation." + key: lifecycle[key]
        for key in (
            "validation_sha256",
            "validation_examples",
        )
    }
    for group, policy in (
        ("optimizer", optimizer_policy.model_dump(mode="json")),
        ("lifecycle", lifecycle_policy.model_dump(mode="json")),
    ):
        values |= {group + "." + key: value for key, value in policy.items()}
    # Teacher policy is separately versioned; nested loss weights are scientific parameters.
    for key, value in teacher_policy.model_dump(mode="json").items():
        if isinstance(value, dict):
            values |= {"teacher." + key + "." + k: v for k, v in value.items()}
        else:
            values["teacher." + key] = value
    return {_PREFIX + key: _parameter_value(value) for key, value in values.items()}


def _progress(snapshot: IrisScientificExecutionSnapshotV1, count: int) -> str:
    if type(count) is not int or not 0 <= count <= len(snapshot.events):
        raise ValueError("MLflow cursor lies outside the scientific journal")
    prefix = snapshot.events[:count]
    state = prefix[-1].payload.after if prefix else ScientificLifecycleStateV1()
    best = checkpoint = None
    for event in prefix:
        record = event.payload.record
        if type(record) is ScientificExecutionValidationV1 and record.selected_checkpoint_sha256:
            best = record.selected_checkpoint_sha256
        if type(record) is ScientificExecutionCheckpointV1:
            checkpoint = record.lifecycle_checkpoint_sha256
    return canonical_json_bytes(
        {
            "event_count": count,
            "chain_head_sha256": prefix[-1].sha256 if prefix else snapshot.header.sha256,
            "state": state.model_dump(mode="json"),
            "best_checkpoint_sha256": best,
            "lifecycle_checkpoint_sha256": checkpoint,
            "history_sha256": prefix[-1].payload.history_sha256
            if prefix
            else snapshot.header.payload.initial.payload.history_sha256,
        }
    ).decode()


def _metrics(
    snapshot: IrisScientificExecutionSnapshotV1, limit: int
) -> tuple[_MlflowMetricPointV1, ...]:
    points: list[_MlflowMetricPointV1] = []
    for event in snapshot.events:
        payload = event.payload
        record = payload.record
        values: dict[str, float | int | bool] = {
            "progress.examples_seen": payload.examples_seen,
            "progress.target_tokens_seen": payload.target_tokens_seen,
        }
        if type(record) is ScientificExecutionTrainV1:
            metric = record.metric
            values |= {
                "train." + key: getattr(metric, key)
                for key in (
                    "loss",
                    "language_model_loss",
                    "probe_loss",
                    "diversity_loss",
                    "utilization_loss",
                    "norm_loss",
                    "learning_rate",
                    "gradient_l2_norm",
                    "gradients_clipped",
                    "example_count",
                    "supervised_token_count",
                )
            }
        elif type(record) is ScientificExecutionValidationV1:
            values |= {
                "validation." + key: getattr(record.metric, key)
                for key in (
                    "target_nll_sum",
                    "target_nll_per_token",
                    "target_token_count",
                    "example_count",
                )
            }
            values["validation.selected_best"] = record.selected_checkpoint_sha256 is not None
            if payload.after.best_step is not None and payload.after.best_nll_per_token is not None:
                values["validation.best_step"] = payload.after.best_step
                values["validation.best_nll_per_token"] = payload.after.best_nll_per_token
        elif type(record) is ScientificExecutionCheckpointV1:
            values["checkpoint.marker"] = 1
        else:
            raise ValueError("Unknown scientific record")
        if len(points) + len(values) > limit:
            raise ValueError("Scientific MLflow metric-point budget exceeded")
        points.extend(
            _metric_point(
                key=_PREFIX + key,
                value=value,
                sequence=payload.sequence,
                step=payload.after.step,
            )
            for key, value in sorted(values.items())
        )
    return tuple(points)


def _validate_run(
    run: _MlflowRunV1, snapshot: IrisScientificExecutionSnapshotV1, params: dict[str, str]
) -> tuple[dict[str, str], int]:
    identity = _identity(snapshot)
    if run.lifecycle_stage != "active" or run.status not in {"RUNNING", "FINISHED"}:
        raise ValueError("Scientific MLflow run is inactive or has an incompatible status")
    _require_allowed_namespace(run.tags, allowed=frozenset(identity) | {_PROGRESS}, label="tags")
    _require_allowed_namespace(run.params, allowed=frozenset(params), label="parameters")
    if _missing_exact_values(actual=run.tags, expected=identity, label="scientific identity"):
        raise ValueError("Scientific MLflow run has incomplete identity tags")
    count = 0
    if _PROGRESS in run.tags:
        raw = run.tags[_PROGRESS]
        if len(raw.encode()) > 8192:
            raise ValueError("Oversized scientific MLflow cursor")
        count = _json(raw.encode())["event_count"]
        if raw != _progress(snapshot, count):
            raise ValueError("Scientific MLflow cursor differs from its exact source prefix")
    final = bool(snapshot.events and snapshot.events[-1].payload.after.phase == "stopped")
    if run.status == "FINISHED" and (not final or count != len(snapshot.events)):
        raise ValueError("Scientific MLflow run was terminated before its source")
    return _missing_exact_values(actual=run.params, expected=params, label="parameters"), count


def _open_gateway(uri: str) -> MlflowTrackingGatewayV1:
    from mlflow.telemetry.utils import is_telemetry_disabled

    if not is_telemetry_disabled():
        raise ValueError("MLflow SDK telemetry policy must be disabled")
    return MlflowTrackingGatewayV1(tracking_uri=uri)


def project_scientific_execution_to_mlflow_v1(
    *,
    execution_path: Path,
    output_root: Path,
    experiment_name: str,
    maximum_metric_points: int = 1_000_000,
    maximum_store_bytes: int = 1024**3,
) -> IrisScientificMlflowReportV1:
    """Reconcile one verified prefix into explicit local SQLite, under a process lock.

    Retry after an uncertain SDK write verifies existing history before adding only
    missing values. This is metadata delivery, not independent scientific acceptance.
    Byte limits are observed between SDK calls; no SQLite transaction peak-size claim.
    """
    try:
        # Validate BEFORE importing/constructing the MLflow SDK. Never modify caller
        # environment or rely on pytest/CI implicitly disabling telemetry for us.
        if os.environ.get("MLFLOW_DISABLE_TELEMETRY", "").lower() != "true" or os.environ.get(
            "_MLFLOW_TESTING_TELEMETRY", "false"
        ).lower() not in {"false", "0"}:
            raise IrisScientificMlflowError(
                "Local scientific MLflow requires MLFLOW_DISABLE_TELEMETRY=true "
                "and no telemetry testing override"
            )
        if type(maximum_metric_points) is not int or not 1 <= maximum_metric_points <= 4_000_000:
            raise ValueError("Explicit bounded metric-point budget required")
        if type(maximum_store_bytes) is not int or not 1 <= maximum_store_bytes <= 1024**3:
            raise ValueError("Explicit observed metadata store budget <=1GiB required")
        name = _explicit_text(experiment_name, label="experiment name", maximum_length=256)
        source, root = _local(execution_path), _local(output_root)
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("MLflow target and scientific source must be disjoint")
        snapshot = read_scientific_execution_v1(source)
        params, points = _params(snapshot), _metrics(snapshot, maximum_metric_points)
        identity, progress = _identity(snapshot), _progress(snapshot, len(snapshot.events))
        _store_check(root, maximum_store_bytes)
        root.mkdir(parents=True, exist_ok=True)
        _local(root)
        database = root / "tracking.sqlite"
        with process_file_lock(root / "projection.lock") as claim:

            def guard() -> None:
                claim.check()
                _store_check(root, maximum_store_bytes)

            guard()
            gateway = _open_gateway("sqlite:///" + database.as_posix())
            guard()
            experiment, _ = gateway.ensure_experiment(
                name=name,
                artifact_location=(root / "artifacts").as_uri(),
            )
            guard()
            matches = gateway.find_scientific_runs(
                experiment_id=experiment.experiment_id,
                execution_sha256=snapshot.header.sha256,
            )
            if len(matches) > 1:
                raise ValueError("Multiple MLflow runs claim this scientific execution")
            reused = bool(matches)
            if matches:
                run = matches[0]
            else:
                run = gateway.create_run(
                    experiment_id=experiment.experiment_id,
                    run_name="scientific-" + snapshot.header.sha256[:16],
                    tags=identity,
                )
                guard()
                matches = gateway.find_scientific_runs(
                    experiment_id=experiment.experiment_id,
                    execution_sha256=snapshot.header.sha256,
                )
                if len(matches) != 1 or matches[0].run_id != run.run_id:
                    raise ValueError("Scientific MLflow creation did not establish unique identity")
            missing_params, _ = _validate_run(run, snapshot, params)
            missing = _missing_metric_points(gateway=gateway, run=run, expected_points=points)
            if missing_params:
                guard()
                gateway.log_batch(run_id=run.run_id, params=missing_params)
            for batch in _metric_batches(missing):
                guard()
                gateway.log_batch(run_id=run.run_id, metrics=batch)
            guard()
            reconciled = gateway.get_run(run.run_id)
            remaining, _ = _validate_run(reconciled, snapshot, params)
            if remaining or _missing_metric_points(
                gateway=gateway,
                run=reconciled,
                expected_points=points,
            ):
                raise ValueError("Scientific MLflow values were not fully installed")
            if read_scientific_execution_v1(source) != snapshot:
                raise ValueError("Scientific source changed during MLflow projection")
            if reconciled.tags.get(_PROGRESS) != progress:
                guard()
                gateway.log_batch(run_id=run.run_id, tags={_PROGRESS: progress})
            # Do not terminate a run on an unacknowledged/missing cursor: that would
            # make retry look like an incompatible prematurely finished projection.
            reconciled = gateway.get_run(run.run_id)
            remaining, count = _validate_run(reconciled, snapshot, params)
            if (
                remaining
                or count != len(snapshot.events)
                or reconciled.tags.get(_PROGRESS) != progress
            ):
                raise ValueError("Scientific MLflow progress cursor was not installed")
            finished = bool(
                snapshot.events and snapshot.events[-1].payload.after.phase == "stopped"
            )
            if finished and reconciled.status != "FINISHED":
                guard()
                gateway.set_terminated(run_id=run.run_id, end_time=len(snapshot.events))
            guard()
            installed = gateway.get_run(run.run_id)
            remaining, count = _validate_run(installed, snapshot, params)
            if (
                remaining
                or count != len(snapshot.events)
                or installed.tags.get(_PROGRESS) != progress
                or installed.status != ("FINISHED" if finished else "RUNNING")
                or _missing_metric_points(gateway=gateway, run=installed, expected_points=points)
            ):
                raise ValueError("Scientific MLflow final verification differs")
            if read_scientific_execution_v1(source) != snapshot:
                raise ValueError("Scientific source changed before MLflow acknowledgement")
            return IrisScientificMlflowReportV1(
                snapshot.header.sha256,
                len(snapshot.events),
                snapshot.events[-1].sha256 if snapshot.events else snapshot.header.sha256,
                run.run_id,
                experiment.experiment_id,
                database,
                len(points),
                len(missing),
                reused,
                "finished" if finished else "running",
            )
    except IrisScientificMlflowError:
        raise
    except Exception as error:
        raise IrisScientificMlflowError("Scientific local MLflow projection failed") from error


__all__ = [
    "IrisScientificMlflowError",
    "IrisScientificMlflowReportV1",
    "project_scientific_execution_to_mlflow_v1",
]
