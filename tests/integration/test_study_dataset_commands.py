"""Real dataset storage, redacted CLI verification, and legacy runner-input bridge."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import iris.application as application_api
import iris.artifacts as artifact_api
from iris.application.errors import IrisStudyDatasetError, IrisVerificationError
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.study_dataset_commands import (
    export_study_experiment_manifest_v1,
    verify_study_dataset_v1,
)
from iris.application.study_dataset_io import write_study_dataset_v1
from iris.artifacts.study_dataset_v1 import IrisStudyDatasetV1
from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_study_dataset import _dataset

pytestmark = pytest.mark.integration
runner = CliRunner()


def _persist(tmp_path: Path) -> tuple[Path, IrisStudyDatasetV1]:
    dataset = _dataset()
    path = tmp_path / "dataset.json"
    write_study_dataset_v1(dataset, path=path)
    return path, dataset


def test_verifier_reports_counts_and_identity_without_private_content(tmp_path: Path) -> None:
    path, dataset = _persist(tmp_path)
    before = path.read_bytes()
    report = verify_study_dataset_v1(path)

    assert path.read_bytes() == before
    assert report.payload_sha256 == dataset.payload_sha256
    assert report.manifest_payload_sha256 == dataset.payload.manifest.payload_sha256
    assert report.total_examples == 30
    assert report.leakage_group_count == 15
    assert report.purpose == "operational_fixture"
    assert report.scientific_launch_status == "not_assessed"
    assert report.validation_scope == "input_membership_and_exact_duplicate_checks_only"
    assert tuple(item.task_family for item in report.task_counts) == tuple(
        item.task_family for item in dataset.payload.task_definitions
    )
    assert all((item.train, item.validation, item.test) == (2, 2, 2) for item in report.task_counts)
    rendered = repr(asdict(report))
    for example, assignment in zip(
        dataset.payload.manifest.payload.examples, dataset.payload.assignments, strict=True
    ):
        assert example.prompt not in rendered
        assert example.target not in rendered
        assert assignment.source_item_id not in rendered
        assert assignment.leakage_group_id not in rendered
    assert dataset.payload.construction.split_policy_description not in rendered


def test_export_preserves_exact_training_manifest_with_all_splits(tmp_path: Path) -> None:
    source, dataset = _persist(tmp_path)
    destination = tmp_path / "export" / "manifest.json"
    before = source.read_bytes()
    first = export_study_experiment_manifest_v1(dataset_path=source, manifest_path=destination)
    second = export_study_experiment_manifest_v1(dataset_path=source, manifest_path=destination)

    assert source.read_bytes() == before
    assert read_experiment_manifest_v1(destination) == dataset.payload.manifest
    assert destination.read_bytes() == canonical_json_file_bytes(
        dataset.payload.manifest.model_dump(mode="json")
    )
    assert first.dataset_payload_sha256 == dataset.payload_sha256
    assert first.manifest_payload_sha256 == dataset.payload.manifest.payload_sha256
    assert first.size_bytes == destination.stat().st_size
    assert not first.reused_existing
    assert second.reused_existing
    assert dataset.payload.manifest.payload.split_counts.test == 10


@pytest.mark.parametrize("same_source", [False, True])
def test_export_never_overwrites_existing_file(tmp_path: Path, same_source: bool) -> None:
    source, _ = _persist(tmp_path)
    destination = source if same_source else tmp_path / "conflict.json"
    if not same_source:
        destination.write_bytes(b"user-owned bytes")
    before = destination.read_bytes()

    with pytest.raises(IrisStudyDatasetError):
        export_study_experiment_manifest_v1(dataset_path=source, manifest_path=destination)

    assert destination.read_bytes() == before


def test_invalid_source_is_rejected_before_creating_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_bytes(b'{"private_prompt":"SECRET_PRIVATE_TEXT"}\n')
    destination = tmp_path / "uncreated" / "manifest.json"

    with pytest.raises(IrisStudyDatasetError):
        export_study_experiment_manifest_v1(dataset_path=source, manifest_path=destination)
    with pytest.raises(IrisVerificationError):
        verify_study_dataset_v1(source)

    assert not destination.parent.exists()


def test_real_cli_verification_is_read_only_and_does_not_load_backbone(tmp_path: Path) -> None:
    path, dataset = _persist(tmp_path)
    before = path.read_bytes()
    with patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader:
        result = runner.invoke(app, ["verify", "study-dataset", "--path", str(path)])

    assert result.exit_code == 0, result.output
    loader.assert_not_called()
    assert "Study dataset verification" in result.stdout
    assert "not_assessed" in result.stdout
    assert "operational_fixture" in result.stdout
    assert "content-redacted" in result.stdout
    assert "Привет" not in result.stdout
    assert "Exact target" not in result.stdout
    assert path.read_bytes() == before
    assert dataset.payload.manifest.payload.split_counts.train == 10


def test_real_cli_export_and_legacy_verification(tmp_path: Path) -> None:
    source, dataset = _persist(tmp_path)
    output = tmp_path / "manifest.json"
    result = runner.invoke(
        app, ["study", "export-manifest", "--dataset", str(source), "--path", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "ALL three splits" in result.stdout
    assert "not assessed" in result.stdout
    assert "Exact target" not in result.stdout
    assert read_experiment_manifest_v1(output) == dataset.payload.manifest
    verified = runner.invoke(app, ["verify", "manifest", "--path", str(output)])
    assert verified.exit_code == 0, verified.output


@pytest.mark.parametrize("debug", [False, True])
def test_cli_validation_errors_do_not_echo_private_input(tmp_path: Path, debug: bool) -> None:
    path = tmp_path / "invalid.json"
    raw = _dataset().model_dump(mode="json")
    raw["payload"]["manifest"]["payload"]["examples"][0]["prompt"] = ["SECRET_PRIVATE_TEXT"]
    path.write_bytes(canonical_json_file_bytes(raw))
    args = ["--debug"] if debug else []
    result = runner.invoke(app, [*args, "verify", "study-dataset", "--path", str(path)])

    assert result.exit_code != 0
    assert "SECRET_PRIVATE_TEXT" not in result.output
    if not debug:
        assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["study", "--help"],
        ["study", "export-manifest", "--help"],
        ["verify", "study-dataset", "--help"],
    ],
)
def test_cli_help_is_available(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output


def test_public_apis_are_exact_exports() -> None:
    assert artifact_api.IrisStudyDatasetV1 is IrisStudyDatasetV1
    for name in (
        "build_study_dataset_v1",
        "validate_study_dataset_v1",
        "select_study_examples_v1",
        "project_study_experiment_manifest_v1",
        "read_study_dataset_v1",
        "write_study_dataset_v1",
        "verify_study_dataset_v1",
        "export_study_experiment_manifest_v1",
        "IrisStudyDatasetError",
    ):
        assert name in application_api.__all__
        assert getattr(application_api, name) is not None
    with pytest.raises(IrisStudyDatasetError):
        application_api.validate_study_dataset_v1(cast(IrisStudyDatasetV1, object()))
