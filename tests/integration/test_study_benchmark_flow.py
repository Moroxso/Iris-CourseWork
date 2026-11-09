"""Canonical private benchmark storage, pilot commands, and the unchanged runner bridge."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import iris.application as application
import iris.artifacts as artifacts
from iris.application.errors import IrisStudyBenchmarkError
from iris.application.experiment_manifest import (
    read_experiment_manifest_v1,
    write_experiment_manifest_v1,
)
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_benchmark_io import read_study_benchmark_v1, write_study_benchmark_v1
from iris.application.study_dataset import project_study_experiment_manifest_v1
from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_study_benchmark import _benchmark

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_canonical_round_trip_no_clobber_and_runner_bridge(tmp_path: Path) -> None:
    benchmark = _benchmark()
    path = tmp_path / "benchmark.json"
    written = write_study_benchmark_v1(benchmark, path=path)
    read = read_study_benchmark_v1(path)
    reused = write_study_benchmark_v1(benchmark, path=path)
    assert written.payload_sha256 == benchmark.payload_sha256
    assert not written.reused_existing
    assert reused.reused_existing
    assert read == benchmark
    assert read is not benchmark
    assert path.read_bytes() == canonical_json_file_bytes(benchmark.model_dump(mode="json"))
    manifest = project_study_experiment_manifest_v1(read.payload.dataset)
    destination = tmp_path / "runner-input.json"
    write_experiment_manifest_v1(manifest, path=destination)
    assert read_experiment_manifest_v1(destination) == benchmark.payload.dataset.payload.manifest


def test_conflicting_file_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_bytes(b"unrelated user content")
    with pytest.raises(IrisStudyBenchmarkError):
        write_study_benchmark_v1(_benchmark(), path=path)
    assert path.read_bytes() == b"unrelated user content"


@pytest.mark.parametrize(
    "kind", ["bom", "extra_lf", "missing_lf", "duplicates", "nan", "digest", "nested_digest"]
)
def test_noncanonical_or_tampered_input_is_rejected(tmp_path: Path, kind: str) -> None:
    benchmark = _benchmark()
    data = canonical_json_file_bytes(benchmark.model_dump(mode="json"))
    if kind == "bom":
        data = b"\xef\xbb\xbf" + data
    elif kind == "extra_lf":
        data += b"\n"
    elif kind == "missing_lf":
        data = data[:-1]
    elif kind == "duplicates":
        data = b'{"payload":{},"payload":{}}\n'
    elif kind == "nan":
        data = b'{"payload":NaN}\n'
    else:
        digest = (
            benchmark.payload_sha256
            if kind == "digest"
            else benchmark.payload.dataset.payload_sha256
        )
        data = data.replace(digest.encode(), b"0" * 64, 1)
    path = tmp_path / "invalid.json"
    path.write_bytes(data)
    with pytest.raises(IrisStudyBenchmarkError):
        read_study_benchmark_v1(path)


def test_invalid_benchmark_does_not_create_output_parent(tmp_path: Path) -> None:
    invalid = _benchmark().model_copy(update={"payload_sha256": "0" * 64})
    path = tmp_path / "uncreated" / "benchmark.json"
    with pytest.raises(IrisStudyBenchmarkError):
        write_study_benchmark_v1(invalid, path=path)
    assert not path.parent.exists()


def test_real_pilot_cli_builds_and_verifies_without_model_loading(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    args = [
        "study",
        "build-pilot",
        "--seed",
        "1907",
        "--problems-per-cell",
        "2",
        "--path",
        str(path),
    ]
    with patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader:
        result = runner.invoke(app, args)
        verified = runner.invoke(app, ["verify", "study-benchmark", "--path", str(path)])
    assert result.exit_code == 0, result.output
    assert verified.exit_code == 0, verified.output
    loader.assert_not_called()
    assert "operational_fixture" in result.stdout
    assert "not assessed" in verified.stdout
    assert "isolated Python backend" in verified.stdout
    assert len(read_study_benchmark_v1(path).payload.scoring) == 30
    for private_text in ("def solve", "Quoted log entries", "inputs", "expected"):
        assert private_text not in result.stdout + verified.stdout


@pytest.mark.parametrize(
    "args", [["study", "build-pilot", "--help"], ["verify", "study-benchmark", "--help"]]
)
def test_pilot_and_benchmark_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output


def test_bad_seed_fails_before_any_output_creation(tmp_path: Path) -> None:
    path = tmp_path / "uncreated" / "benchmark.json"
    result = runner.invoke(
        app,
        ["study", "build-pilot", "--seed", "-1", "--problems-per-cell", "1", "--path", str(path)],
    )
    assert result.exit_code != 0
    assert not path.parent.exists()


def test_public_export_identity() -> None:
    assert application.validate_study_benchmark_v1 is validate_study_benchmark_v1
    assert application.read_study_benchmark_v1 is read_study_benchmark_v1
    benchmark = _benchmark()
    assert type(benchmark) is artifacts.IrisStudyBenchmarkV1
    names: dict[str, Any] = {
        "build_synthetic_study_pilot_v1": application.build_synthetic_study_pilot_v1,
        "score_study_answer_v1": application.score_study_answer_v1,
        "IrisCodeEvaluationUnavailableError": application.IrisCodeEvaluationUnavailableError,
    }
    assert all(name in application.__all__ and value is not None for name, value in names.items())
