"""Public offline command, canonical input IO, and redacted failure exit codes."""

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import iris.application as application
from iris.application.errors import IrisStudyInputPreflightError
from iris.application.study_benchmark_io import write_study_benchmark_v1
from iris.application.study_input_command import (
    StudyBackboneProfileV1,
    preflight_study_input_file_v1,
)
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.backbone import BackboneAdapter
from iris.cli import app
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_backbone_adapter import FakeModel, cpu_info
from tests.unit.test_study_input_preflight import _assets, _config, _tokenizer

pytestmark = pytest.mark.integration


def test_token_ids_match_actual_adapter_prompt_and_teacher_preparation_without_forward() -> None:
    config, tokenizer = _config(), _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    with _assets(tokenizer):
        report = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    model = FakeModel()
    model.config.max_position_embeddings = report.max_position_embeddings
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        adapter = BackboneAdapter(config.backbone, cpu_info())
    for example, observation in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples, report.examples, strict=True
    ):
        prompt = adapter.tokenize(example.prompt)["input_ids"].tolist()[0]
        target = adapter.tokenize_teacher_target(
            example.target, terminal_token_id=1
        ).input_ids.tolist()[0]
        assert observation.prompt_token_ids_sha256 == domain_sha256(
            "iris/study-input-token-ids/v1", canonical_json_bytes(prompt)
        )
        assert observation.target_token_ids_sha256 == domain_sha256(
            "iris/study-input-token-ids/v1", canonical_json_bytes(target)
        )
    assert model.calls == []


@pytest.mark.parametrize("profile", ["qwen3_0_6b", "qwen3_1_7b"])
def test_packaged_command_honors_profile_and_limits_without_altering_benchmark(
    tmp_path: Path, profile: StudyBackboneProfileV1
) -> None:
    path = tmp_path / "benchmark.json"
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    write_study_benchmark_v1(benchmark, path=path)
    before = path.read_bytes()
    with _assets(_tokenizer()) as loader:
        report = preflight_study_input_file_v1(
            path,
            terminal_token_id=1,
            backbone_profile=profile,
            max_prompt_tokens=256,
            max_new_tokens=128,
        )
    assert report.inputs_fit
    assert report.max_new_tokens == 128
    assert report.benchmark_payload_sha256 == benchmark.payload_sha256
    assert path.read_bytes() == before
    assert loader.call_args.kwargs["local_files_only"] is True
    expected_revision = (
        "c1899de289a04d12100db370d81485cdf75e47ca"
        if profile == "qwen3_0_6b"
        else "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert loader.call_args.kwargs["revision"] == expected_revision


@pytest.mark.parametrize("limit,code", [(128, 0), (1, 2)])
def test_real_cli_is_redacted_and_budget_failure_is_nonzero(
    tmp_path: Path, limit: int, code: int
) -> None:
    path = tmp_path / "benchmark.json"
    write_study_benchmark_v1(build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1), path=path)
    with _assets(_tokenizer()):
        result = CliRunner().invoke(
            app,
            [
                "study",
                "preflight-inputs",
                "--benchmark",
                str(path),
                "--terminal-token-id",
                "1",
                "--backbone-profile",
                "qwen3_0_6b",
                "--max-new-tokens",
                str(limit),
            ],
        )
    assert result.exit_code == code, result.output
    assert "not_assessed" in result.output
    assert ("blocked" if code else "fit") in result.output
    for private in ("def solve", ":train:", "Remember the goal", "target_token_ids"):
        assert private not in result.output


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_override_is_rejected_before_input_read(tmp_path: Path, limit: int) -> None:
    with (
        patch("iris.application.study_input_command.read_study_benchmark_v1") as reader,
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_input_file_v1(
            tmp_path / "absent.json",
            terminal_token_id=1,
            backbone_profile="qwen3_0_6b",
            max_prompt_tokens=limit,
        )
    reader.assert_not_called()


def test_unknown_profile_is_rejected_before_input_read(tmp_path: Path) -> None:
    with (
        patch("iris.application.study_input_command.read_study_benchmark_v1") as reader,
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_input_file_v1(
            tmp_path / "absent.json",
            terminal_token_id=1,
            backbone_profile=cast(StudyBackboneProfileV1, "UNKNOWN"),
        )
    reader.assert_not_called()


def test_cli_requires_explicit_profile_and_terminal_and_exports_are_live() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["study", "preflight-inputs", "--help"]).exit_code == 0
    assert (
        runner.invoke(app, ["study", "preflight-inputs", "--benchmark", "missing.json"]).exit_code
        == 2
    )
    assert application.preflight_study_input_file_v1 is preflight_study_input_file_v1
    assert application.IrisStudyInputPreflightError is IrisStudyInputPreflightError
    for name in (
        "preflight_study_inputs_v1",
        "IrisStudyInputPreflightV1",
        "IrisStudyExampleTokenBudgetV1",
        "StudyBackboneProfileV1",
        "StudyInputViolationV1",
    ):
        assert name in application.__all__
