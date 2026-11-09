"""MLflow SDK adapter for the deterministic journal-projection gateway."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from iris.application.mlflow_projection import (
    _MlflowExperimentV1,
    _MlflowMetricPointV1,
    _MlflowObservedMetricV1,
    _MlflowRunV1,
)


def _exact_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        message = f"MLflow returned an invalid {label}"
        raise TypeError(message)
    return value


def _exact_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        message = f"MLflow returned an invalid {label}"
        raise TypeError(message)
    return value


def _exact_integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        message = f"MLflow returned an invalid {label}"
        raise TypeError(message)
    return value


def _finite_float(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        message = f"MLflow returned an invalid {label}"
        raise TypeError(message)
    numeric = float(value)
    if not math.isfinite(numeric):
        message = f"MLflow returned a non-finite {label}"
        raise ValueError(message)
    return numeric


def _text_mapping(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        message = f"MLflow returned invalid {label}"
        raise TypeError(message)
    owned: dict[str, str] = {}
    for key, item in value.items():
        owned[_exact_text(key, label=f"{label} key")] = _exact_string(
            item,
            label=f"{label} value",
        )
    return owned


class MlflowTrackingGatewayV1:
    """Instance-scoped synchronous wrapper around one explicit MLflow tracking URI."""

    __slots__ = ("_client",)

    def __init__(self, *, tracking_uri: str) -> None:
        from mlflow import MlflowClient

        self._client: Any = MlflowClient(tracking_uri=tracking_uri)

    @staticmethod
    def _experiment(native: Any) -> _MlflowExperimentV1:
        return _MlflowExperimentV1(
            experiment_id=_exact_text(native.experiment_id, label="experiment_id"),
            artifact_location=_exact_text(
                native.artifact_location,
                label="artifact_location",
            ),
            lifecycle_stage=_exact_text(native.lifecycle_stage, label="experiment lifecycle"),
        )

    @staticmethod
    def _run(native: Any) -> _MlflowRunV1:
        params = _text_mapping(native.data.params, label="run parameters")
        tags = _text_mapping(native.data.tags, label="run tags")
        metrics = native.data.metrics
        if not isinstance(metrics, Mapping):
            message = "MLflow returned invalid run metrics"
            raise TypeError(message)
        metric_keys = frozenset(_exact_text(key, label="metric key") for key in metrics)
        return _MlflowRunV1(
            run_id=_exact_text(native.info.run_id, label="run_id"),
            lifecycle_stage=_exact_text(native.info.lifecycle_stage, label="run lifecycle"),
            status=_exact_text(native.info.status, label="run status"),
            params=params,
            tags=tags,
            metric_keys=metric_keys,
        )

    def ensure_experiment(
        self,
        *,
        name: str,
        artifact_location: str,
    ) -> tuple[_MlflowExperimentV1, bool]:
        native = self._client.get_experiment_by_name(name)
        reused = native is not None
        if native is None:
            try:
                experiment_id = self._client.create_experiment(
                    name,
                    artifact_location=artifact_location,
                )
            except Exception as primary_error:
                native = self._client.get_experiment_by_name(name)
                if native is None:
                    raise primary_error
                reused = True
            else:
                native = self._client.get_experiment(experiment_id)
        experiment = self._experiment(native)
        if experiment.lifecycle_stage != "active":
            message = "MLflow experiment is not active"
            raise ValueError(message)
        if experiment.artifact_location.rstrip("/") != artifact_location.rstrip("/"):
            message = "MLflow experiment artifact location conflicts with the explicit target"
            raise ValueError(message)
        return experiment, reused

    def find_runs(
        self,
        *,
        experiment_id: str,
        run_payload_sha256: str,
    ) -> tuple[_MlflowRunV1, ...]:
        from mlflow.entities import ViewType

        filter_string = f"tags.`iris.source.run_payload_sha256` = '{run_payload_sha256}'"
        native_runs = self._client.search_runs(
            [experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ALL,
            max_results=2,
        )
        return tuple(self._run(native) for native in native_runs)

    def find_scientific_runs(
        self, *, experiment_id: str, execution_sha256: str
    ) -> tuple[_MlflowRunV1, ...]:
        """Scientific execution identity never aliases the legacy run-payload tag."""
        from mlflow.entities import ViewType

        if re.fullmatch(r"[0-9a-f]{64}", execution_sha256) is None:
            raise ValueError("Exact scientific execution digest required")
        native_runs = self._client.search_runs(
            [experiment_id],
            filter_string=f"tags.`iris.scientific.execution_sha256` = '{execution_sha256}'",
            run_view_type=ViewType.ALL,
            max_results=2,
        )
        return tuple(self._run(native) for native in native_runs)

    def create_run(
        self,
        *,
        experiment_id: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> _MlflowRunV1:
        native = self._client.create_run(
            experiment_id,
            start_time=1,
            tags=dict(tags),
            run_name=run_name,
        )
        return self._run(native)

    def get_run(self, run_id: str) -> _MlflowRunV1:
        return self._run(self._client.get_run(run_id))

    def get_metric_history(
        self,
        *,
        run_id: str,
        key: str,
    ) -> tuple[_MlflowObservedMetricV1, ...]:
        history = self._client.get_metric_history(run_id, key)
        return tuple(
            _MlflowObservedMetricV1(
                key=_exact_text(metric.key, label="metric key"),
                value=_finite_float(metric.value, label="metric value"),
                timestamp=_exact_integer(metric.timestamp, label="metric timestamp"),
                step=_exact_integer(metric.step, label="metric step"),
            )
            for metric in history
        )

    def log_batch(
        self,
        *,
        run_id: str,
        metrics: tuple[_MlflowMetricPointV1, ...] = (),
        params: Mapping[str, str] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        from mlflow.entities import Metric, Param, RunTag

        metric_type: Any = Metric
        param_type: Any = Param
        tag_type: Any = RunTag

        native_metrics = tuple(
            metric_type(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
            )
            for metric in metrics
        )
        native_params = tuple(
            param_type(key=key, value=value) for key, value in sorted((params or {}).items())
        )
        native_tags = tuple(
            tag_type(key=key, value=value) for key, value in sorted((tags or {}).items())
        )
        self._client.log_batch(
            run_id,
            metrics=native_metrics,
            params=native_params,
            tags=native_tags,
            synchronous=True,
        )

    def set_terminated(self, *, run_id: str, end_time: int) -> None:
        self._client.set_terminated(
            run_id,
            status="FINISHED",
            end_time=end_time,
        )


__all__ = ["MlflowTrackingGatewayV1"]
