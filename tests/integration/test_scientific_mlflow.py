"""Real scientific journals projected into fake/retryable and real local SQLite MLflow."""

import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iris.application import scientific_mlflow as projection
from iris.application.mlflow_projection import _MlflowExperimentV1, _MlflowRunV1
from iris.application.scientific_execution import (
    create_scientific_execution_v1,
    read_scientific_execution_v1,
)
from iris.cli import app
from iris.integrations.mlflow_tracking import MlflowTrackingGatewayV1
from iris.utils.hashing import canonical_json_bytes
from iris.utils.process_file_lock import process_file_lock
from tests.integration.test_mlflow_projection_flow import _FakeMlflowGateway
from tests.integration.test_scientific_lifecycle import _lifecycle
from tests.integration.test_scientific_publication import completed as completed

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def explicit_telemetry_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.delenv("_MLFLOW_TESTING_TELEMETRY", raising=False)


class _Gateway(_FakeMlflowGateway):
    def __init__(self, root: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.experiment = _MlflowExperimentV1("science-1", (root / "artifacts").as_uri(), "active")

    def ensure_experiment(
        self, *, name: str, artifact_location: str
    ) -> tuple[_MlflowExperimentV1, bool]:
        assert name == "scientific-tests"
        assert artifact_location == self.experiment.artifact_location
        return self.experiment, True

    def find_scientific_runs(
        self, *, experiment_id: str, execution_sha256: str
    ) -> tuple[_MlflowRunV1, ...]:
        assert experiment_id == self.experiment.experiment_id
        if (
            self.run_id is None
            or self.tags.get("iris.scientific.execution_sha256") != execution_sha256
        ):
            return ()
        run = self._run()
        return (run, run) if self.duplicate_matches else (run,)


def _project(
    source: Path, root: Path, gateway: _Gateway, **kwargs: Any
) -> projection.IrisScientificMlflowReportV1:
    with patch.object(projection, "_open_gateway", return_value=gateway):
        return projection.project_scientific_execution_to_mlflow_v1(
            execution_path=source,
            output_root=root,
            experiment_name="scientific-tests",
            **kwargs,
        )


def test_exact_metrics_best_identity_and_repeat_are_read_only_to_source(
    tmp_path: Path,
    completed: Path,
) -> None:
    gateway = _Gateway(tmp_path)
    before = read_scientific_execution_v1(completed)
    report = _project(completed, tmp_path, gateway)
    assert report.status == "finished" and not report.reused_existing_run
    assert report.new_metric_point_count == report.metric_point_count > 0
    assert "iris.source.run_payload_sha256" not in gateway.tags
    assert gateway.params["iris.scientific.arm"] == "D"
    assert gateway.params["iris.scientific.precision"] == "float32"
    assert gateway.params["iris.scientific.metric_time"] == "event_sequence_not_wall_time"
    for event in before.events:
        if event.payload.record.kind == "train":
            for field in (
                "loss",
                "language_model_loss",
                "probe_loss",
                "diversity_loss",
                "utilization_loss",
                "norm_loss",
            ):
                actual = next(
                    p
                    for p in gateway.histories["iris.scientific.train." + field]
                    if p.timestamp == event.payload.sequence
                )
                assert actual.value == getattr(event.payload.record.metric, field)
                assert actual.step == event.payload.after.step
    cursor = json.loads(gateway.tags[projection._PROGRESS])
    assert cursor["state"]["phase"] == "stopped" and cursor["state"]["best_step"] == 1
    best = next(
        e.payload.record.selected_checkpoint_sha256
        for e in before.events
        if e.payload.record.kind == "validation" and e.payload.record.selected_checkpoint_sha256
    )
    assert cursor["best_checkpoint_sha256"] == best
    writes = gateway.write_calls
    again = _project(completed, tmp_path, gateway)
    assert again.new_metric_point_count == 0 and again.reused_existing_run
    assert gateway.write_calls == writes
    assert read_scientific_execution_v1(completed) == before
    text = repr((gateway.params, gateway.tags))
    assert "expected_response" not in text and "probe_vocabulary" not in text
    assert "prompt" not in text and "tokenizer_json" not in text
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("stage", ["create", "metrics", "progress", "finish"])
def test_lost_sdk_acknowledgement_retries_without_duplicate_history(
    tmp_path: Path,
    completed: Path,
    stage: str,
) -> None:
    gateway = _Gateway(tmp_path, fail_after_metric_batch=(stage == "metrics"))
    method = (
        "create_run"
        if stage == "create"
        else "set_terminated"
        if stage == "finish"
        else "log_batch"
    )
    original = getattr(gateway, method)
    failed = False

    def uncertain(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        result = original(*args, **kwargs)
        if not failed and (
            stage != "progress" or projection._PROGRESS in (kwargs.get("tags") or {})
        ):
            failed = True
            raise OSError("lost acknowledgement")
        return result

    injection = (
        patch.object(gateway, method, side_effect=uncertain)
        if stage != "metrics"
        else patch.object(gateway, "log_batch", wraps=gateway.log_batch)
    )
    with injection, pytest.raises(projection.IrisScientificMlflowError):
        _project(completed, tmp_path, gateway)
    report = _project(completed, tmp_path, gateway)
    assert report.status == "finished" and report.reused_existing_run
    for history in gateway.histories.values():
        assert len(history) == len({p.identity for p in history})
    assert sum(map(len, gateway.histories.values())) == report.metric_point_count


@pytest.mark.parametrize(
    "damage",
    [
        "metric",
        "duplicate",
        "extra_metric",
        "identity",
        "param",
        "cursor",
        "missing_identity",
        "duplicate_run",
        "status",
        "partial_cursor",
    ],
)
def test_destination_conflicts_are_rejected_without_rewriting_evidence(
    tmp_path: Path,
    completed: Path,
    damage: str,
) -> None:
    gateway = _Gateway(tmp_path)
    _project(completed, tmp_path, gateway)
    key = "iris.scientific.train.loss"
    if damage == "metric":
        gateway.histories[key][0] = replace(gateway.histories[key][0], value=987.0)
    elif damage == "duplicate":
        gateway.histories[key].append(gateway.histories[key][0])
    elif damage == "extra_metric":
        gateway.histories["iris.unexpected"] = []
    elif damage == "identity":
        gateway.tags["iris.scientific.initial_checkpoint_sha256"] = "f" * 64
    elif damage == "missing_identity":
        del gateway.tags["iris.scientific.schema"]
    elif damage == "param":
        gateway.params["iris.scientific.seed"] = "987"
    elif damage == "cursor":
        cursor = json.loads(gateway.tags[projection._PROGRESS])
        cursor["chain_head_sha256"] = "f" * 64
        gateway.tags[projection._PROGRESS] = canonical_json_bytes(cursor).decode()
    elif damage == "partial_cursor":
        gateway.tags[projection._PROGRESS] = '{"event_count":1}'
    elif damage == "duplicate_run":
        gateway.duplicate_matches = True
    else:
        gateway.status = "FAILED"
    writes = gateway.write_calls
    with pytest.raises(projection.IrisScientificMlflowError):
        _project(completed, tmp_path, gateway)
    assert gateway.write_calls == writes


def test_empty_then_extended_prefix_keeps_one_run_and_only_new_metrics(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    execution = create_scientific_execution_v1(lifecycle, root=tmp_path / "source")
    root = tmp_path / "tracking"
    gateway = _Gateway(root)
    initial = _project(execution.path, root, gateway)
    assert initial.event_count == initial.metric_point_count == 0 and initial.status == "running"
    execution.advance()
    first = _project(execution.path, root, gateway)
    assert first.new_metric_point_count == first.metric_point_count > 0
    while lifecycle.state.phase != "stopped":
        execution.advance()
    final = _project(execution.path, root, gateway)
    assert final.new_metric_point_count == final.metric_point_count - first.metric_point_count
    assert initial.mlflow_run_id == first.mlflow_run_id == final.mlflow_run_id
    assert final.status == "finished"


def test_early_termination_is_not_silently_reopened(tmp_path: Path) -> None:
    execution = create_scientific_execution_v1(_lifecycle(), root=tmp_path / "source")
    root = tmp_path / "tracking"
    gateway = _Gateway(root)
    _project(execution.path, root, gateway)
    gateway.status = "FINISHED"
    with pytest.raises(projection.IrisScientificMlflowError):
        _project(execution.path, root, gateway)


def test_corrupt_source_is_rejected_before_sdk_or_target_creation(
    tmp_path: Path, completed: Path
) -> None:
    copy = tmp_path / "source" / completed.name
    shutil.copytree(completed, copy)
    event = copy / "events/00000001.json"
    event.write_bytes(event.read_bytes().replace(b'"kind":"train"', b'"kind":"bogus"'))
    root = tmp_path / "tracking"
    with (
        patch.object(projection, "_open_gateway", side_effect=AssertionError("No SDK")),
        pytest.raises(projection.IrisScientificMlflowError),
    ):
        projection.project_scientific_execution_to_mlflow_v1(
            execution_path=copy,
            output_root=root,
            experiment_name="scientific-tests",
        )
    assert not root.exists()


@pytest.mark.parametrize("budget", [0, 1, True, 4_000_001])
def test_metric_budgets_fail_before_target_creation(
    tmp_path: Path, completed: Path, budget: int
) -> None:
    root = tmp_path / "tracking"
    with pytest.raises(projection.IrisScientificMlflowError):
        _project(completed, root, _Gateway(root), maximum_metric_points=budget)
    assert not root.exists()


def test_paths_storage_reserve_and_local_lock_prevent_sdk_access(
    tmp_path: Path, completed: Path
) -> None:
    with patch.object(projection, "_open_gateway", side_effect=AssertionError("No SDK")) as sdk:
        for root in (
            completed / "tracking",
            completed.parent,
            Path("//remote/share/tracking"),
            Path("C:relative"),
        ):
            with pytest.raises(projection.IrisScientificMlflowError):
                projection.project_scientific_execution_to_mlflow_v1(
                    execution_path=completed,
                    output_root=root,
                    experiment_name="scientific-tests",
                )
        root = tmp_path / "tracking"
        root.mkdir()
        with (
            process_file_lock(root / "projection.lock"),
            pytest.raises(projection.IrisScientificMlflowError),
        ):
            projection.project_scientific_execution_to_mlflow_v1(
                execution_path=completed,
                output_root=root,
                experiment_name="scientific-tests",
            )
        (root / "unexpected.txt").write_text("retain me")
        with pytest.raises(projection.IrisScientificMlflowError):
            projection.project_scientific_execution_to_mlflow_v1(
                execution_path=completed,
                output_root=root,
                experiment_name="scientific-tests",
            )
        assert (root / "unexpected.txt").read_text() == "retain me"
        with (
            patch.object(shutil, "disk_usage", return_value=shutil._ntuple_diskusage(100, 100, 0)),
            pytest.raises(projection.IrisScientificMlflowError),
        ):
            projection.project_scientific_execution_to_mlflow_v1(
                execution_path=completed,
                output_root=tmp_path / "full",
                experiment_name="scientific-tests",
            )
        sdk.assert_not_called()


def test_real_local_sqlite_and_cli_idempotence_without_network(
    tmp_path: Path,
    completed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow.telemetry.utils as telemetry
    from mlflow import MlflowClient, get_tracking_uri

    # The operation must remain offline OUTSIDE pytest's implicit telemetry opt-out.
    monkeypatch.setattr(telemetry, "_IS_IN_CI_ENV_OR_TESTING", False)
    root = tmp_path / "real SQLite"
    original_global = get_tracking_uri()
    before = read_scientific_execution_v1(completed)
    with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
        result = CliRunner().invoke(
            app,
            [
                "scientific",
                "project-mlflow",
                "--execution",
                str(completed),
                "--output-root",
                str(root),
                "--experiment-name",
                "scientific-tests",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        report = projection.project_scientific_execution_to_mlflow_v1(
            execution_path=completed,
            output_root=root,
            experiment_name="scientific-tests",
        )
        assert report.reused_existing_run and report.new_metric_point_count == 0
        client = MlflowClient(tracking_uri="sqlite:///" + report.tracking_database.as_posix())
        run = client.get_run(report.mlflow_run_id)
        assert run.info.status == "FINISHED"
        assert len(client.search_runs([report.experiment_id])) == 1
        points = client.get_metric_history(report.mlflow_run_id, "iris.scientific.train.loss")
        assert len(points) == before.events[-1].payload.after.step
        assert all(p.timestamp < 1000 for p in points)  # Logical sequence, not UTC claims.
        assert not client.list_artifacts(report.mlflow_run_id)
        gateway = MlflowTrackingGatewayV1(
            tracking_uri="sqlite:///" + report.tracking_database.as_posix()
        )
        assert (
            gateway.find_runs(
                experiment_id=report.experiment_id, run_payload_sha256=before.header.sha256
            )
            == ()
        )
    assert get_tracking_uri() == original_global
    assert read_scientific_execution_v1(completed) == before
    assert report.tracking_database.stat().st_size < 16 * 1024**2


def test_cancellation_is_not_wrapped(tmp_path: Path, completed: Path) -> None:
    with (
        patch.object(projection, "_open_gateway", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        projection.project_scientific_execution_to_mlflow_v1(
            execution_path=completed,
            output_root=tmp_path,
            experiment_name="scientific-tests",
        )


@pytest.mark.parametrize("override", [False, True])
def test_telemetry_requires_explicit_opt_out_before_any_sdk_or_target(
    tmp_path: Path,
    completed: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
) -> None:
    if override:
        monkeypatch.setenv("_MLFLOW_TESTING_TELEMETRY", "true")
    else:
        monkeypatch.delenv("MLFLOW_DISABLE_TELEMETRY")
    root = tmp_path / "tracking"
    with (
        patch.object(projection, "_open_gateway", side_effect=AssertionError("No SDK")) as sdk,
        pytest.raises(projection.IrisScientificMlflowError, match="TELEMETRY"),
    ):
        projection.project_scientific_execution_to_mlflow_v1(
            execution_path=completed,
            output_root=root,
            experiment_name="scientific-tests",
        )
    sdk.assert_not_called()
    assert not root.exists()


def test_partial_metric_batch_and_cursor_ack_are_repaired_from_actual_history(
    tmp_path: Path,
    completed: Path,
) -> None:
    gateway = _Gateway(tmp_path)
    original = gateway.log_batch
    failed = False

    def partial(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        if kwargs.get("metrics") and not failed:
            failed = True
            original(*args, **(kwargs | {"metrics": kwargs["metrics"][:2]}))
            raise OSError("partial transaction")
        original(*args, **kwargs)

    with (
        patch.object(gateway, "log_batch", side_effect=partial),
        pytest.raises(projection.IrisScientificMlflowError),
    ):
        _project(completed, tmp_path, gateway)
    assert sum(map(len, gateway.histories.values())) == 2
    report = _project(completed, tmp_path, gateway)
    assert report.new_metric_point_count == report.metric_point_count - 2
    # The cursor alone is not proof of installed history: reconcile a missing point.
    gateway.histories["iris.scientific.train.loss"].pop()
    repaired = _project(completed, tmp_path, gateway)
    assert repaired.new_metric_point_count == 1


def test_source_changes_during_projection_prevent_progress_acknowledgement(
    tmp_path: Path,
    completed: Path,
) -> None:
    source = tmp_path / "source" / completed.name
    shutil.copytree(completed, source)
    root = tmp_path / "tracking"
    gateway = _Gateway(root)
    original = gateway.log_batch

    def alter_after_metrics(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        if kwargs.get("metrics"):
            target = source / "events/00000001.json"
            target.write_bytes(target.read_bytes() + b" ")

    with (
        patch.object(gateway, "log_batch", side_effect=alter_after_metrics),
        pytest.raises(projection.IrisScientificMlflowError),
    ):
        _project(source, root, gateway)
    assert projection._PROGRESS not in gateway.tags and gateway.status == "RUNNING"


def test_silent_dropped_cursor_is_not_acknowledged(tmp_path: Path, completed: Path) -> None:
    gateway = _Gateway(tmp_path)
    original = gateway.log_batch

    def drop_progress(*args: Any, **kwargs: Any) -> None:
        if projection._PROGRESS not in (kwargs.get("tags") or {}):
            original(*args, **kwargs)

    with (
        patch.object(gateway, "log_batch", side_effect=drop_progress),
        pytest.raises(projection.IrisScientificMlflowError),
    ):
        _project(completed, tmp_path, gateway)
    assert gateway.status == "RUNNING"
    assert _project(completed, tmp_path, gateway).status == "finished"


def test_observed_disk_budget_does_not_overwrite_or_remove_existing_files(
    tmp_path: Path,
    completed: Path,
) -> None:
    database = tmp_path / "tracking.sqlite"
    database.write_bytes(b"fixture exceeding explicit bound")
    before = database.read_bytes()
    with (
        patch.object(projection, "_open_gateway", side_effect=AssertionError("No SDK")) as sdk,
        pytest.raises(projection.IrisScientificMlflowError),
    ):
        _project(completed, tmp_path, _Gateway(tmp_path), maximum_store_bytes=1)
    sdk.assert_not_called()
    assert database.read_bytes() == before


def test_parameter_projection_revalidates_policy_allowlist(completed: Path) -> None:
    snapshot = read_scientific_execution_v1(completed)
    values = projection._params(snapshot)
    assert values["iris.scientific.teacher.weights.probe"] == "0.05"
    payload = snapshot.header.payload.initial.payload
    config = json.loads(payload.configuration_json)
    config["teacher_policy"]["private_prompt"] = "do not project unknown source content"
    changed_payload = payload.model_copy(
        update={"configuration_json": canonical_json_bytes(config).decode()}
    )
    changed_initial = snapshot.header.payload.initial.model_copy(
        update={"payload": changed_payload}
    )
    changed_header_payload = snapshot.header.payload.model_copy(update={"initial": changed_initial})
    changed = replace(
        snapshot, header=snapshot.header.model_copy(update={"payload": changed_header_payload})
    )
    # Even this private pure helper refuses non-schema policy fields; the public
    # API additionally checks all envelope hashes before reaching the allowlist.
    with pytest.raises(ValueError, match="Extra inputs"):
        projection._params(changed)
