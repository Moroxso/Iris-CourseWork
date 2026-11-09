"""Verified, idempotent MLflow projection tests with fake and local SQL stores."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application.mlflow_projection as projection_module
from iris.application.errors import IrisMlflowProjectionError
from iris.application.mlflow_projection import (
    IrisMlflowProjectionReportV1,
    _MlflowExperimentV1,
    _MlflowMetricPointV1,
    _MlflowObservedMetricV1,
    _MlflowRunV1,
    project_experiment_journal_to_mlflow_v1,
)
from iris.application.verification import (
    IrisExperimentJournalVerificationReportV1,
    verify_experiment_journal_v1,
)
from tests.integration.test_experiment_log_flow import _tamper_json
from tests.integration.test_experiment_runner_flow import (
    _deterministic_runtime,
    _execute,
    _fresh_owner,
)
from tests.integration.test_verification_flow import _install_managed_run, _tree_snapshot

pytestmark = pytest.mark.integration

_PRIVATE_SENTINELS = (
    "offline teacher prompt",
    "longer target",
    "short target",
    "train-long",
    "train-short",
    "validation-short",
    "test-long",
)


class _FakeMlflowGateway:
    def __init__(self, *, fail_after_metric_batch: bool = False) -> None:
        self.experiment = _MlflowExperimentV1(
            experiment_id="fake-experiment-1",
            artifact_location="file:///explicit-artifacts",
            lifecycle_stage="active",
        )
        self.run_id: str | None = None
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.histories: dict[str, list[_MlflowObservedMetricV1]] = {}
        self.status = "RUNNING"
        self.fail_after_metric_batch = fail_after_metric_batch
        self.duplicate_matches = False
        self.write_calls = 0

    def _run(self) -> _MlflowRunV1:
        if self.run_id is None:
            message = "fake run has not been created"
            raise RuntimeError(message)
        return _MlflowRunV1(
            run_id=self.run_id,
            lifecycle_stage="active",
            status=self.status,
            params=dict(self.params),
            tags=dict(self.tags),
            metric_keys=frozenset(self.histories),
        )

    def ensure_experiment(
        self,
        *,
        name: str,
        artifact_location: str,
    ) -> tuple[_MlflowExperimentV1, bool]:
        assert name == "fake-iris"
        assert artifact_location == self.experiment.artifact_location
        return self.experiment, True

    def find_runs(
        self,
        *,
        experiment_id: str,
        run_payload_sha256: str,
    ) -> tuple[_MlflowRunV1, ...]:
        assert experiment_id == self.experiment.experiment_id
        if self.run_id is None or self.tags.get("iris.source.run_payload_sha256") != (
            run_payload_sha256
        ):
            return ()
        match = self._run()
        return (match, match) if self.duplicate_matches else (match,)

    def create_run(
        self,
        *,
        experiment_id: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> _MlflowRunV1:
        assert experiment_id == self.experiment.experiment_id
        assert run_name
        assert self.run_id is None
        self.run_id = "fake-run-1"
        self.tags.update(tags)
        return self._run()

    def get_run(self, run_id: str) -> _MlflowRunV1:
        assert run_id == self.run_id
        return self._run()

    def get_metric_history(
        self,
        *,
        run_id: str,
        key: str,
    ) -> tuple[_MlflowObservedMetricV1, ...]:
        assert run_id == self.run_id
        return tuple(self.histories.get(key, ()))

    def log_batch(
        self,
        *,
        run_id: str,
        metrics: tuple[_MlflowMetricPointV1, ...] = (),
        params: Mapping[str, str] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        assert run_id == self.run_id
        self.write_calls += 1
        self.params.update(params or {})
        self.tags.update(tags or {})
        for metric in metrics:
            self.histories.setdefault(metric.key, []).append(
                _MlflowObservedMetricV1(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                )
            )
        if metrics and self.fail_after_metric_batch:
            self.fail_after_metric_batch = False
            raise OSError("synthetic uncertain MLflow write")

    def set_terminated(self, *, run_id: str, end_time: int) -> None:
        assert run_id == self.run_id
        assert end_time >= 0
        self.status = "FINISHED"


def _project_with_fake(
    *,
    journal_path: Path,
    manifest_path: Path,
    gateway: _FakeMlflowGateway,
) -> IrisMlflowProjectionReportV1:
    with patch.object(projection_module, "_open_mlflow_gateway", return_value=gateway):
        return project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri="sqlite:///explicit.db",
            experiment_name="fake-iris",
            artifact_location=gateway.experiment.artifact_location,
        )


def test_uncertain_metric_write_reconciles_without_duplicates_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id="mlflow-fake-retry",
    )
    gateway = _FakeMlflowGateway(fail_after_metric_batch=True)

    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    installed_after_failure = sum(len(history) for history in gateway.histories.values())
    assert installed_after_failure > 0

    report = _project_with_fake(
        journal_path=journal_path,
        manifest_path=manifest_path,
        gateway=gateway,
    )
    installed_after_retry = sum(len(history) for history in gateway.histories.values())

    assert report.reused_existing_run
    assert report.new_metric_point_count == 0
    assert report.reused_metric_point_count == report.metric_point_count
    assert installed_after_retry == installed_after_failure == report.metric_point_count
    assert gateway.status == "FINISHED"
    projected_text = repr((gateway.params, gateway.tags, tuple(gateway.histories)))
    assert all(sentinel not in projected_text for sentinel in _PRIVATE_SENTINELS)

    original_dataset = gateway.params["iris.source.dataset_id"]
    gateway.params["iris.source.dataset_id"] = "conflicting-dataset"
    writes_before_conflict = gateway.write_calls
    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    assert gateway.write_calls == writes_before_conflict
    gateway.params["iris.source.dataset_id"] = original_dataset

    original_head = gateway.tags["iris.projection.chain_head_sha256"]
    gateway.tags["iris.projection.chain_head_sha256"] = "f" * 64
    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    assert gateway.write_calls == writes_before_conflict
    gateway.tags["iris.projection.chain_head_sha256"] = original_head

    original_count = gateway.tags.pop("iris.projection.event_count")
    original_head = gateway.tags.pop("iris.projection.chain_head_sha256")
    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    assert gateway.write_calls == writes_before_conflict
    gateway.tags["iris.projection.event_count"] = original_count
    gateway.tags["iris.projection.chain_head_sha256"] = original_head

    key = "iris.train.loss"
    gateway.histories[key][0] = replace(
        gateway.histories[key][0],
        value=gateway.histories[key][0].value + 1.0,
    )
    writes_before_conflict = gateway.write_calls
    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    assert gateway.write_calls == writes_before_conflict

    gateway.duplicate_matches = True
    with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
        _project_with_fake(
            journal_path=journal_path,
            manifest_path=manifest_path,
            gateway=gateway,
        )
    assert gateway.write_calls == writes_before_conflict


def test_invalid_source_fails_before_opening_mlflow_target(tmp_path: Path) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id="mlflow-invalid-source",
        maximum_new_training_steps=1,
    )
    _tamper_json(journal_path / "events" / "00000001.json", "f" * 64)

    with (
        patch.object(projection_module, "_open_mlflow_gateway") as open_gateway,
        pytest.raises(IrisMlflowProjectionError, match="projection failed"),
    ):
        project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri="sqlite:///must-not-open.db",
            experiment_name="must-not-open",
            artifact_location="file:///must-not-open",
        )

    open_gateway.assert_not_called()


def test_source_change_after_verification_fails_before_opening_mlflow_target(
    tmp_path: Path,
) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id="mlflow-source-race",
        maximum_new_training_steps=1,
    )
    original_verify = verify_experiment_journal_v1

    def verify_then_tamper(
        *,
        path: Path,
        manifest_path: Path,
    ) -> IrisExperimentJournalVerificationReportV1:
        report = original_verify(path=path, manifest_path=manifest_path)
        _tamper_json(path / "events" / "00000001.json", "f" * 64)
        return report

    with (
        patch.object(
            projection_module,
            "verify_experiment_journal_v1",
            side_effect=verify_then_tamper,
        ),
        patch.object(projection_module, "_open_mlflow_gateway") as open_gateway,
        pytest.raises(IrisMlflowProjectionError, match="projection failed"),
    ):
        project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri="sqlite:///must-not-open.db",
            experiment_name="must-not-open",
            artifact_location="file:///must-not-open",
        )

    open_gateway.assert_not_called()


def test_sqlite_projection_is_offline_idempotent_and_extends_one_run(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    journal_path, manifest_path = _install_managed_run(
        source_root,
        run_id="mlflow-sqlite-extension",
        maximum_new_training_steps=1,
    )
    database_path = tmp_path / "tracking.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"
    artifact_directory = tmp_path / "mlflow-artifacts"
    artifact_directory.mkdir()
    artifact_location = artifact_directory.resolve().as_uri()

    source_before = _tree_snapshot(source_root)
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("MLflow projection attempted network access"),
    ):
        first = project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri=tracking_uri,
            experiment_name="iris-local-sqlite",
            artifact_location=artifact_location,
        )
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_tag(first.mlflow_run_id, "foreign.empty", "")
        repeated_partial = project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri=tracking_uri,
            experiment_name="iris-local-sqlite",
            artifact_location=artifact_location,
        )
        conflicting_artifact_directory = tmp_path / "other-mlflow-artifacts"
        conflicting_artifact_directory.mkdir()
        with pytest.raises(IrisMlflowProjectionError, match="projection failed"):
            project_experiment_journal_to_mlflow_v1(
                path=journal_path,
                manifest_path=manifest_path,
                tracking_uri=tracking_uri,
                experiment_name="iris-local-sqlite",
                artifact_location=conflicting_artifact_directory.resolve().as_uri(),
            )
    assert _tree_snapshot(source_root) == source_before
    assert first.status == "running"
    assert first.event_count == 1
    assert first.metric_point_count == first.new_metric_point_count == 5
    assert not first.reused_existing_run
    assert repeated_partial.mlflow_run_id == first.mlflow_run_id
    assert repeated_partial.reused_existing_run
    assert repeated_partial.new_metric_point_count == 0

    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        completed = _execute(
            root=source_root,
            run_id="mlflow-sqlite-extension",
            model=model,
            optimizer=optimizer,
        )
    assert completed.status == "complete"
    completed_source = _tree_snapshot(source_root)

    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("MLflow projection attempted network access"),
    ):
        extended = project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri=tracking_uri,
            experiment_name="iris-local-sqlite",
            artifact_location=artifact_location,
        )
        repeated_complete = project_experiment_journal_to_mlflow_v1(
            path=journal_path,
            manifest_path=manifest_path,
            tracking_uri=tracking_uri,
            experiment_name="iris-local-sqlite",
            artifact_location=artifact_location,
        )
    assert _tree_snapshot(source_root) == completed_source

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(first.mlflow_run_id)
    metric_keys = tuple(sorted(run.data.metrics))
    history_count = sum(
        len(client.get_metric_history(first.mlflow_run_id, key)) for key in metric_keys
    )

    assert extended.mlflow_run_id == first.mlflow_run_id
    assert extended.reused_existing_run
    assert extended.status == "finished"
    assert extended.event_count == 5
    assert extended.metric_point_count == 22
    assert extended.new_metric_point_count == 17
    assert repeated_complete.new_metric_point_count == 0
    assert history_count == extended.metric_point_count
    assert len(client.get_metric_history(first.mlflow_run_id, "iris.event.marker")) == 5
    assert run.info.status == "FINISHED"
    assert run.info.start_time == 1
    assert run.info.end_time == 5
    assert run.data.tags["iris.projection.event_count"] == "5"
    assert run.data.tags["iris.projection.chain_head_sha256"] == extended.chain_head_sha256
    assert run.data.params["iris.source.dataset_id"] == "offline-baseline-v1"

    destination_text = repr((run.data.params, run.data.tags, metric_keys, first, extended))
    assert all(sentinel not in destination_text for sentinel in _PRIVATE_SENTINELS)
    for destination_file in tmp_path.rglob("*"):
        if destination_file.is_file() and source_root not in destination_file.parents:
            content = destination_file.read_bytes()
            assert all(sentinel.encode() not in content for sentinel in _PRIVATE_SENTINELS)
