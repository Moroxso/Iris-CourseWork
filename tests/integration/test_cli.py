from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application.backbone import BackboneForwardReport
from iris.application.errors import (
    IrisExperimentCommandError,
    IrisMlflowProjectionError,
    IrisRequestArtifactError,
    IrisRequestCommandError,
    IrisVerificationError,
)
from iris.application.experiment_runner import IrisExperimentExecutionResult
from iris.application.experiment_runner_v2 import IrisExperimentExecutionResultV2
from iris.application.mlflow_projection import IrisMlflowProjectionReportV1
from iris.application.verification import (
    IrisExperimentJournalVerificationReportV1,
    IrisExperimentManifestVerificationReportV1,
    IrisFourArmProtocolVerificationReportV1,
    IrisRequestArtifactVerificationReportV1,
    IrisTeacherCheckpointVerificationReportV1,
)
from iris.application.verification_v2 import (
    IrisExperimentJournalVerificationReportV2,
    IrisRequestArtifactVerificationReportV2,
)
from iris.application.verification_v3 import IrisRequestArtifactVerificationReportV3
from iris.application.verification_v4 import IrisRequestArtifactVerificationReportV4
from iris.cli import app
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.training import IrisAdamWConfig

runner = CliRunner()

_PRIVATE_OUTPUT_SENTINELS = (
    "IRIS_PRIVATE_PROMPT_5dc2_Привет",
    "IRIS_PRIVATE_TARGET_f487_ответ",
    "IRIS_PRIVATE_EXAMPLE_ID_a091",
)


def _assert_verification_output_is_redacted(output: str) -> None:
    for private_value in _PRIVATE_OUTPUT_SENTINELS:
        assert private_value not in output


def test_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "hardware" in result.stdout
    assert "backbone" in result.stdout
    assert "request" in result.stdout
    assert "experiment" in result.stdout
    assert "verify" in result.stdout
    assert "Traceback" not in result.output


def test_verify_help_lists_all_read_only_artifact_commands() -> None:
    result = runner.invoke(app, ["verify", "--help"])

    assert result.exit_code == 0
    for command in ("request", "checkpoint", "manifest", "four-arm-protocol", "journal"):
        assert command in result.stdout
    assert "Traceback" not in result.output


def test_experiment_help_lists_execution_and_mlflow_projection() -> None:
    result = runner.invoke(app, ["experiment", "--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "project-mlflow" in result.stdout
    assert "Traceback" not in result.output


_EXPERIMENT_REQUIRED_OPTIONS = (
    "--manifest",
    "--root",
    "--run-id",
    "--terminal-token-id",
    "--initialization-seed",
    "--epoch-count",
    "--evaluate-test",
    "--learning-rate",
    "--beta1",
    "--beta2",
    "--epsilon",
    "--weight-decay",
    "--max-gradient-norm",
)


def _experiment_run_arguments(
    tmp_path: Path,
    *,
    test_flag: str = "--evaluate-test",
) -> list[str]:
    return [
        "experiment",
        "run",
        "--manifest",
        str(tmp_path / f"{_PRIVATE_OUTPUT_SENTINELS[1]}.json"),
        "--root",
        str(tmp_path / "runs"),
        "--run-id",
        "cli-managed-run",
        "--terminal-token-id",
        "151645",
        "--initialization-seed",
        "41",
        "--epoch-count",
        "2",
        test_flag,
        "--learning-rate",
        "0.001",
        "--beta1",
        "0.9",
        "--beta2",
        "0.999",
        "--epsilon",
        "1e-8",
        "--weight-decay",
        "0.01",
        "--max-gradient-norm",
        "1.0",
    ]


def _experiment_execution_result(
    tmp_path: Path,
    *,
    status: Literal["complete", "step_limit_reached"] = "complete",
) -> IrisExperimentExecutionResult:
    bounded = status == "step_limit_reached"
    return IrisExperimentExecutionResult(
        path=tmp_path / "runs" / "experiments" / "v1" / "cli-managed-run",
        status=status,
        run_payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        event_count=7,
        optimizer_completed_steps=3,
        completed_epochs=1,
        next_epoch_index=2 if bounded else None,
        next_training_ordinal=1 if bounded else None,
        latest_checkpoint_payload_sha256="3" * 64,
        replayed_training_steps=1,
        new_training_steps=1 if bounded else 2,
    )


def _arm_experiment_run_arguments(
    tmp_path: Path,
    *,
    arm_id: str,
    test_flag: str = "--evaluate-test",
) -> list[str]:
    base = _experiment_run_arguments(tmp_path, test_flag=test_flag)
    return ["experiment", "run-arm", "--arm", arm_id, *base[2:]]


def _arm_experiment_execution_result(
    tmp_path: Path,
    *,
    arm_id: Literal["B", "D"],
    status: Literal["complete", "step_limit_reached"] = "complete",
) -> IrisExperimentExecutionResultV2:
    bounded = status == "step_limit_reached"
    return IrisExperimentExecutionResultV2(
        arm_id=arm_id,
        path=tmp_path / "runs" / "experiments" / "v2" / "cli-managed-run",
        status=status,
        run_payload_sha256="4" * 64,
        configuration_sha256="5" * 64,
        event_count=11,
        optimizer_completed_steps=5,
        completed_epochs=1,
        next_epoch_index=2 if bounded else None,
        next_training_ordinal=1 if bounded else None,
        latest_checkpoint_schema_version=2 if arm_id == "B" else 1,
        latest_checkpoint_payload_sha256="6" * 64,
        replayed_training_steps=2,
        new_training_steps=1 if bounded else 3,
    )


def test_experiment_run_help_exposes_only_fixed_packaged_policy() -> None:
    result = runner.invoke(app, ["experiment", "run", "--help"], terminal_width=220)

    assert result.exit_code == 0
    for option in _EXPERIMENT_REQUIRED_OPTIONS:
        assert option in result.stdout
    assert "--skip-test" in result.stdout
    # Rich shortens this one long spelling visually, while the invocation tests
    # below exercise the complete option name.
    assert "--maximum-new-trainin" in result.stdout
    assert "--allow-download" in result.stdout
    for deferred in (
        "--device",
        "--deterministic",
        "--shuffle",
        "--batch-size",
        "--checkpoint-policy",
        "--validation-policy",
        "--resume-policy",
        "--hydra",
    ):
        assert deferred not in result.stdout
    assert "Traceback" not in result.output


def test_experiment_run_forwards_explicit_identity_and_renders_redacted_result(
    tmp_path: Path,
) -> None:
    report = _experiment_execution_result(tmp_path)
    arguments = _experiment_run_arguments(tmp_path)
    manifest_path = Path(arguments[arguments.index("--manifest") + 1])
    root = Path(arguments[arguments.index("--root") + 1])
    with patch(
        "iris.cli.experiment_command_service.run_packaged_experiment_v1",
        return_value=report,
    ) as run_experiment:
        result = runner.invoke(app, arguments, terminal_width=220)

    assert result.exit_code == 0
    run_experiment.assert_called_once()
    forwarded = run_experiment.call_args.kwargs
    assert forwarded == {
        "manifest_path": manifest_path,
        "root": root,
        "run_id": "cli-managed-run",
        "terminal_token_id": 151_645,
        "initialization_seed": 41,
        "epoch_count": 2,
        "evaluate_test_after_training": True,
        "optimizer_config": forwarded["optimizer_config"],
        "maximum_new_training_steps": None,
        "allow_download": False,
    }
    optimizer_config = forwarded["optimizer_config"]
    assert type(optimizer_config) is IrisAdamWConfig
    assert optimizer_config == IrisAdamWConfig(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )
    for expected in (
        "Experiment execution",
        "complete",
        "content-redacted private metadata",
        "Run payload SHA-256",
        "Configuration SHA-256",
        "Optimizer completed steps",
        "Completed epochs",
        "Latest checkpoint SHA-256",
        "Replayed training steps",
        "New training steps",
    ):
        assert expected in result.stdout
    assert str(manifest_path) not in result.output
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_experiment_run_forwards_skip_limit_and_download_opt_in(tmp_path: Path) -> None:
    report = _experiment_execution_result(tmp_path, status="step_limit_reached")
    arguments = _experiment_run_arguments(tmp_path, test_flag="--skip-test")
    arguments.extend(("--maximum-new-training-steps", "1", "--allow-download"))
    with patch(
        "iris.cli.experiment_command_service.run_packaged_experiment_v1",
        return_value=report,
    ) as run_experiment:
        result = runner.invoke(app, arguments, terminal_width=220)

    assert result.exit_code == 0
    forwarded = run_experiment.call_args.kwargs
    assert forwarded["evaluate_test_after_training"] is False
    assert forwarded["maximum_new_training_steps"] == 1
    assert forwarded["allow_download"] is True
    assert "step_limit_reached" in result.stdout
    assert "not applicable" not in result.stdout
    _assert_verification_output_is_redacted(result.output)


@pytest.mark.parametrize("missing_option", _EXPERIMENT_REQUIRED_OPTIONS)
def test_experiment_run_requires_every_identity_and_optimizer_option(
    tmp_path: Path,
    missing_option: str,
) -> None:
    arguments = _experiment_run_arguments(tmp_path)
    index = arguments.index(missing_option)
    del arguments[index]
    if missing_option != "--evaluate-test":
        del arguments[index]
    with patch("iris.cli.experiment_command_service.run_packaged_experiment_v1") as run_experiment:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert missing_option in result.output
    run_experiment.assert_not_called()


@pytest.mark.parametrize(
    "option",
    (
        "--terminal-token-id",
        "--initialization-seed",
        "--epoch-count",
        "--learning-rate",
        "--beta1",
        "--beta2",
        "--epsilon",
        "--weight-decay",
        "--max-gradient-norm",
        "--maximum-new-training-steps",
    ),
)
def test_experiment_run_rejects_malformed_numeric_options_before_service(
    tmp_path: Path,
    option: str,
) -> None:
    arguments = _experiment_run_arguments(tmp_path)
    if option == "--maximum-new-training-steps":
        arguments.extend((option, "not-a-number"))
    else:
        arguments[arguments.index(option) + 1] = "not-a-number"
    with patch("iris.cli.experiment_command_service.run_packaged_experiment_v1") as run_experiment:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert option in result.output
    run_experiment.assert_not_called()


def test_experiment_run_handles_invalid_optimizer_policy_inside_application_boundary(
    tmp_path: Path,
) -> None:
    arguments = _experiment_run_arguments(tmp_path)
    arguments[arguments.index("--learning-rate") + 1] = "0"
    with patch("iris.cli.experiment_command_service.run_packaged_experiment_v1") as run_experiment:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "learning_rate" in result.stderr
    assert "Traceback" not in result.output
    run_experiment.assert_not_called()


def test_experiment_run_failure_does_not_echo_manifest_or_nested_private_content(
    tmp_path: Path,
) -> None:
    arguments = _experiment_run_arguments(tmp_path)
    manifest_path = arguments[arguments.index("--manifest") + 1]
    nested = RuntimeError(_PRIVATE_OUTPUT_SENTINELS[0])
    failure = IrisExperimentCommandError("Iris experiment command failed during managed execution")
    failure.__cause__ = nested
    with patch(
        "iris.cli.experiment_command_service.run_packaged_experiment_v1",
        side_effect=failure,
    ):
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: Iris experiment command failed during managed execution" in result.stderr
    assert manifest_path not in result.output
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_experiment_run_arm_help_exposes_exact_label_and_fixed_packaged_policy() -> None:
    result = runner.invoke(app, ["experiment", "run-arm", "--help"], terminal_width=220)

    assert result.exit_code == 0
    assert "--arm" in result.stdout
    assert "A|B|C|D" in result.stdout
    for option in _EXPERIMENT_REQUIRED_OPTIONS:
        rendered_option = "--initialization-se" if option == "--initialization-seed" else option
        assert rendered_option in result.stdout
    assert "--skip-test" in result.stdout
    assert "--maximum-new-train" in result.stdout
    assert "--allow-download" in result.stdout
    for deferred in (
        "--device",
        "--deterministic",
        "--shuffle",
        "--batch-size",
        "--checkpoint-policy",
        "--validation-policy",
        "--resume-policy",
        "--hydra",
    ):
        assert deferred not in result.stdout
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("arm_id", "test_flag", "status", "step_limit", "allow_download"),
    [
        ("B", "--evaluate-test", "complete", None, False),
        ("D", "--skip-test", "step_limit_reached", 1, True),
    ],
)
def test_experiment_run_arm_forwards_exact_b_d_identity_and_renders_redacted_result(
    tmp_path: Path,
    arm_id: Literal["B", "D"],
    test_flag: str,
    status: Literal["complete", "step_limit_reached"],
    step_limit: int | None,
    allow_download: bool,
) -> None:
    report = _arm_experiment_execution_result(
        tmp_path,
        arm_id=arm_id,
        status=status,
    )
    arguments = _arm_experiment_run_arguments(
        tmp_path,
        arm_id=arm_id,
        test_flag=test_flag,
    )
    if step_limit is not None:
        arguments.extend(("--maximum-new-training-steps", str(step_limit)))
    if allow_download:
        arguments.append("--allow-download")
    manifest_path = Path(arguments[arguments.index("--manifest") + 1])
    root = Path(arguments[arguments.index("--root") + 1])
    with (
        patch(
            "iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2",
            return_value=report,
        ) as run_experiment,
        patch("iris.cli.experiment_command_service.run_packaged_experiment_v1") as legacy_run,
    ):
        result = runner.invoke(app, arguments, terminal_width=220)

    assert result.exit_code == 0
    run_experiment.assert_called_once()
    legacy_run.assert_not_called()
    forwarded = run_experiment.call_args.kwargs
    assert forwarded == {
        "arm_id": arm_id,
        "manifest_path": manifest_path,
        "root": root,
        "run_id": "cli-managed-run",
        "terminal_token_id": 151_645,
        "initialization_seed": 41,
        "epoch_count": 2,
        "evaluate_test_after_training": test_flag == "--evaluate-test",
        "optimizer_config": forwarded["optimizer_config"],
        "maximum_new_training_steps": step_limit,
        "allow_download": allow_download,
    }
    optimizer_config = forwarded["optimizer_config"]
    assert type(optimizer_config) is IrisAdamWConfig
    assert optimizer_config == IrisAdamWConfig(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )
    for expected in (
        "Arm-labelled experiment execution",
        status,
        "content-redacted private metadata",
        "Arm",
        arm_id,
        "Path",
        "Run payload SHA-256",
        "Configuration SHA-256",
        "Latest checkpoint schema",
        str(report.latest_checkpoint_schema_version),
        "Latest checkpoint SHA-256",
        "Replayed training steps",
        "New training steps",
    ):
        assert expected in result.stdout
    assert str(manifest_path) not in result.output
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("arm_id", "message"),
    [
        ("A", "arm A has no trainable teacher graph"),
        ("C", "arm C teacher graph is blocked"),
    ],
)
def test_experiment_run_arm_rejects_a_c_before_optimizer_or_execution_service(
    tmp_path: Path,
    arm_id: str,
    message: str,
) -> None:
    arguments = _arm_experiment_run_arguments(tmp_path, arm_id=arm_id)
    with (
        patch("iris.cli.IrisAdamWConfig") as optimizer_config,
        patch(
            "iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2"
        ) as run_experiment,
    ):
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert message in result.stderr
    assert "Traceback" not in result.output
    optimizer_config.assert_not_called()
    run_experiment.assert_not_called()


def test_experiment_run_arm_c_blocker_precedes_invalid_optimizer_policy(
    tmp_path: Path,
) -> None:
    arguments = _arm_experiment_run_arguments(tmp_path, arm_id="C")
    arguments[arguments.index("--learning-rate") + 1] = "0"
    with patch(
        "iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2"
    ) as run_experiment:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "arm C teacher graph is blocked" in result.stderr
    assert "learning_rate" not in result.output
    assert "Traceback" not in result.output
    run_experiment.assert_not_called()


def test_experiment_run_arm_rejects_lowercase_label_as_usage_error(tmp_path: Path) -> None:
    arguments = _arm_experiment_run_arguments(tmp_path, arm_id="b")
    with (
        patch(
            "iris.cli.arm_experiment_command_service.require_arm_experiment_command_profile_v2"
        ) as require_profile,
        patch(
            "iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2"
        ) as run_experiment,
    ):
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "--arm" in result.output
    assert "Traceback" not in result.output
    require_profile.assert_not_called()
    run_experiment.assert_not_called()


def test_experiment_run_arm_failure_does_not_echo_manifest_or_nested_private_content(
    tmp_path: Path,
) -> None:
    arguments = _arm_experiment_run_arguments(tmp_path, arm_id="B")
    manifest_path = arguments[arguments.index("--manifest") + 1]
    nested = RuntimeError(_PRIVATE_OUTPUT_SENTINELS[0])
    failure = IrisExperimentCommandError("arm-labelled managed execution failed")
    failure.__cause__ = nested
    with patch(
        "iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2",
        side_effect=failure,
    ):
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: arm-labelled managed execution failed" in result.stderr
    assert manifest_path not in result.output
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_experiment_run_preserves_legacy_v1_service_route(tmp_path: Path) -> None:
    report = _experiment_execution_result(tmp_path)
    arguments = _experiment_run_arguments(tmp_path)
    with (
        patch(
            "iris.cli.experiment_command_service.run_packaged_experiment_v1",
            return_value=report,
        ) as legacy_run,
        patch("iris.cli.arm_experiment_command_service.run_packaged_arm_experiment_v2") as arm_run,
    ):
        result = runner.invoke(app, arguments, terminal_width=220)

    assert result.exit_code == 0
    legacy_run.assert_called_once()
    arm_run.assert_not_called()
    assert "Experiment execution" in result.stdout
    assert "Arm-labelled experiment execution" not in result.stdout
    assert "Traceback" not in result.output


def test_hardware_inspect_reports_explicit_cpu() -> None:
    result = runner.invoke(app, ["hardware", "inspect", "--device", "cpu"])

    assert result.exit_code == 0
    assert "Requested device" in result.stdout
    assert "Resolved backend" in result.stdout
    assert "cpu" in result.stdout
    assert "Traceback" not in result.output


def test_backbone_inspect_does_not_load_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["backbone", "inspect"])

    assert result.exit_code == 0
    assert "qwen3_0_6b" in result.stdout
    assert "Loaded" in result.stdout
    assert "false" in result.stdout


def test_backbone_forward_renders_validated_report(tmp_path: Path) -> None:
    artifact_path = tmp_path / "backbone_forward.json"
    report = BackboneForwardReport(
        backbone="Qwen/Qwen3-0.6B",
        revision="a" * 40,
        device="cpu",
        dtype="float32",
        trainable_parameters=0,
        prompt_tokens=12,
        vocabulary_size=151_936,
        hidden_layers=29,
        hidden_size=1_024,
        logits_shape=(1, 12, 151_936),
        last_hidden_shape=(1, 12, 1_024),
        last_hidden_sha256="b" * 64,
        finite_logits=True,
        finite_hidden_states=True,
        forward_completed=True,
        artifact_path=artifact_path,
    )
    with patch("iris.cli.backbone_service.forward_backbone", return_value=report) as forward:
        result = runner.invoke(app, ["backbone", "forward", "--text", "hello"])

    assert result.exit_code == 0
    forward.assert_called_once_with("hello")
    for expected in (
        "Qwen/Qwen3-0.6B",
        "Revision",
        "CPU",
        "float32",
        "Trainable parameters",
        "Prompt tokens",
        "Vocabulary size",
        "Hidden layers returned",
        "Hidden size",
        "Logits shape",
        "Last hidden shape",
        "Last hidden SHA-256",
        "b" * 64,
        "Finite logits",
        "Finite hidden states",
        "Forward completed",
        str(artifact_path),
    ):
        assert expected in result.stdout
    assert "Traceback" not in result.output


def test_backbone_forward_requires_prompt_text() -> None:
    result = runner.invoke(app, ["backbone", "forward"])

    assert result.exit_code == 2
    assert "--text" in result.output
    assert "Traceback" not in result.output


def test_debug_mode_shows_traceback_for_forward_failure() -> None:
    with patch("iris.cli.backbone_service.forward_backbone", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, ["--debug", "backbone", "forward", "--text", "hello"])

    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "RuntimeError" in result.output


def test_forward_failure_is_concise_without_debug() -> None:
    with patch("iris.cli.backbone_service.forward_backbone", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, ["backbone", "forward", "--text", "hello"])

    assert result.exit_code == 1
    assert "Error: boom" in result.output
    assert "Traceback" not in result.output


def test_invalid_device_has_no_implicit_fallback_or_traceback() -> None:
    result = runner.invoke(app, ["hardware", "inspect", "--device", "auto"])

    assert result.exit_code == 1
    assert "Unsupported device" in result.output
    assert "Traceback" not in result.output


def test_unexpected_failure_is_concise_without_debug() -> None:
    with patch("iris.cli.hardware_service.inspect_hardware", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, ["hardware", "inspect"])

    assert result.exit_code == 1
    assert "Error: boom" in result.output
    assert "Traceback" not in result.output


def request_output(text: str = "response") -> IrisGeneratorOutput:
    return IrisGeneratorOutput(
        text=text,
        token_ids=torch.tensor([[5, 7]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=7,
        workspace_trace=(),
    )


def request_arguments(
    *extra: str,
    strategy: str = "greedy",
    stop_token_ids: tuple[str, ...] = ("7", "2"),
    temperature: str = "0.75",
    top_p: str = "0.875",
) -> list[str]:
    stop_options = [item for token_id in stop_token_ids for item in ("--stop-token-id", token_id)]
    return [
        "request",
        "generate",
        "--prompt",
        "Explain Iris.",
        "--strategy",
        strategy,
        *stop_options,
        "--temperature",
        temperature,
        "--top-p",
        top_p,
        *extra,
    ]


def arm_request_arguments(
    arm_id: str,
    *extra: str,
    strategy: str = "greedy",
) -> list[str]:
    base = request_arguments(*extra, strategy=strategy)
    checkpoint_options: list[str] = []
    if arm_id in {"B", "C", "D"}:
        checkpoint_options = [
            "--checkpoint-payload-sha256",
            "a" * 64,
            "--checkpoint-state-file-sha256",
            "b" * 64,
            "--checkpoint-sidecar-state-sha256",
            "c" * 64,
            "--checkpoint-completed-steps",
            "7",
        ]
    return [
        "request",
        "generate-arm",
        "--arm",
        arm_id,
        *checkpoint_options,
        *base[2:],
    ]


def test_request_generate_help_exposes_only_the_fixed_local_surface() -> None:
    result = runner.invoke(app, ["request", "generate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--prompt",
        "--strategy",
        "--stop-token-id",
        "--temperature",
        "--top-p",
        "--sampling-seed",
        "--allow-download",
    ):
        assert option in result.stdout
    for deferred_option in ("--config-dir", "--config-name", "--output", "--override"):
        assert deferred_option not in result.stdout
    assert "Traceback" not in result.output


def test_request_generate_arm_help_exposes_explicit_four_arm_boundary() -> None:
    result = runner.invoke(
        app,
        ["request", "generate-arm", "--help"],
        terminal_width=160,
    )

    assert result.exit_code == 0
    for option in (
        "--arm",
        "--prompt",
        "--strategy",
        "--stop-token-id",
        "--temperature",
        "--top-p",
        "--sampling-seed",
        "--allow-download",
    ):
        assert option in result.stdout
    for rendered_prefix in (
        "--checkpoint-payload-",
        "--checkpoint-state-fi",
        "--checkpoint-sidecar-",
        "--checkpoint-complete",
    ):
        assert rendered_prefix in result.stdout
    assert "Traceback" not in result.output


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_request_generate_arm_forwards_label_and_writes_exact_text(arm_id: str) -> None:
    output = request_output(f"arm-{arm_id}")
    with patch(
        "iris.cli.arm_request_service.generate_arm_request_v4",
        return_value=output,
    ) as generate:
        result = runner.invoke(app, arm_request_arguments(arm_id))

    assert result.exit_code == 0
    assert result.stdout == f"arm-{arm_id}"
    kwargs = generate.call_args.kwargs
    assert kwargs["arm_id"] == arm_id
    assert kwargs["strategy"] == "greedy"
    assert kwargs["stop_token_ids"] == (7, 2)
    assert kwargs["sampling_seed"] is None
    checkpoint = kwargs["checkpoint_identity"]
    if arm_id == "A":
        assert checkpoint is None
    else:
        assert checkpoint.arm_id == arm_id
        assert checkpoint.checkpoint_schema_version == {"B": 2, "C": 3, "D": 1}[arm_id]
        assert checkpoint.payload_sha256 == "a" * 64
        assert checkpoint.state_file_sha256 == "b" * 64
        assert checkpoint.sidecar_state_sha256 == "c" * 64
        assert checkpoint.completed_steps == 7


def test_request_generate_arm_rejects_lowercase_label_as_usage_error() -> None:
    with patch("iris.cli.arm_request_service.generate_arm_request_v4") as generate:
        result = runner.invoke(app, arm_request_arguments("b"))

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    generate.assert_not_called()


def test_request_generate_arm_artifact_failure_writes_no_response_stdout() -> None:
    failure = IrisRequestArtifactError("arm-labelled artifact installation failed")
    with patch(
        "iris.cli.arm_request_service.generate_arm_request_v4",
        side_effect=failure,
    ):
        result = runner.invoke(app, arm_request_arguments("B"))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: arm-labelled artifact installation failed" in result.stderr
    assert "Traceback" not in result.output


def test_request_generate_arm_requires_complete_checkpoint_identity_for_trained_arm() -> None:
    arguments = arm_request_arguments("B")
    option_index = arguments.index("--checkpoint-state-file-sha256")
    del arguments[option_index : option_index + 2]
    with patch("iris.cli.arm_request_service.generate_arm_request_v4") as generate:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "requires all four teacher checkpoint identity options" in result.stderr
    generate.assert_not_called()


def test_request_generate_arm_a_forbids_checkpoint_identity_options() -> None:
    arguments = arm_request_arguments(
        "A",
        "--checkpoint-payload-sha256",
        "a" * 64,
    )
    with patch("iris.cli.arm_request_service.generate_arm_request_v4") as generate:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Arm A forbids" in result.stderr
    generate.assert_not_called()


def test_request_generate_writes_exact_decoded_text_and_forwards_greedy_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    decoded_text = "[bold]literal markup[/bold]\n<xml>&value"
    output = request_output(decoded_text)
    with patch("iris.cli.request_service.generate_request", return_value=output) as generate:
        result = runner.invoke(app, request_arguments())

    assert result.exit_code == 0
    assert result.stdout == decoded_text
    assert result.stderr == ""
    generate.assert_called_once()
    kwargs = generate.call_args.kwargs
    assert set(kwargs) == {
        "prompt",
        "strategy",
        "stop_token_ids",
        "sampling",
        "sampling_seed",
        "allow_download",
    }
    assert kwargs["prompt"] == "Explain Iris."
    assert kwargs["strategy"] == "greedy"
    assert kwargs["stop_token_ids"] == (7, 2)
    assert kwargs["sampling"] == TemperatureTopPSamplingConfig(
        temperature=0.75,
        top_p=0.875,
    )
    assert kwargs["sampling_seed"] is None
    assert kwargs["allow_download"] is False
    assert list(tmp_path.iterdir()) == []


def test_request_generate_forwards_explicit_sampling_seed_and_download_opt_in() -> None:
    output = request_output("sampled")
    arguments = request_arguments(
        "--sampling-seed",
        "9973",
        "--allow-download",
        strategy="sampling",
    )

    with patch("iris.cli.request_service.generate_request", return_value=output) as generate:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert result.stdout == "sampled"
    kwargs = generate.call_args.kwargs
    assert kwargs["strategy"] == "sampling"
    assert kwargs["sampling_seed"] == 9_973
    assert kwargs["allow_download"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--strategy", "greedy", "--stop-token-id", "7", "--temperature", "1", "--top-p", "1"],
        ["--prompt", "hello", "--stop-token-id", "7", "--temperature", "1", "--top-p", "1"],
        ["--prompt", "hello", "--strategy", "greedy", "--temperature", "1", "--top-p", "1"],
        ["--prompt", "hello", "--strategy", "greedy", "--stop-token-id", "7", "--top-p", "1"],
        [
            "--prompt",
            "hello",
            "--strategy",
            "greedy",
            "--stop-token-id",
            "7",
            "--temperature",
            "1",
        ],
    ],
)
def test_request_generate_missing_required_policy_is_a_usage_error(
    arguments: list[str],
) -> None:
    with patch("iris.cli.request_service.generate_request") as generate:
        result = runner.invoke(app, ["request", "generate", *arguments])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    generate.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        request_arguments(strategy="beam"),
        request_arguments(stop_token_ids=("not-an-int",)),
        request_arguments(temperature="not-a-float"),
        request_arguments(top_p="not-a-float"),
        request_arguments("--sampling-seed", "not-an-int"),
    ],
)
def test_request_generate_malformed_option_is_a_usage_error(arguments: list[str]) -> None:
    with patch("iris.cli.request_service.generate_request") as generate:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    generate.assert_not_called()


def test_request_sampling_policy_validation_runs_inside_application_error_boundary() -> None:
    arguments = request_arguments(temperature="0.0")
    with patch("iris.cli.request_service.generate_request") as generate:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert "temperature" in result.output
    assert "Traceback" not in result.output
    generate.assert_not_called()


def test_request_command_failure_is_concise_without_debug() -> None:
    failure = IrisRequestCommandError("runtime bootstrap failed")
    with patch("iris.cli.request_service.generate_request", side_effect=failure):
        result = runner.invoke(app, request_arguments())

    assert result.exit_code == 1
    assert "Error: runtime bootstrap failed" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.output


def test_request_artifact_failure_writes_no_response_stdout() -> None:
    failure = IrisRequestArtifactError("artifact installation failed")
    with patch("iris.cli.request_service.generate_request", side_effect=failure):
        result = runner.invoke(app, request_arguments())

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: artifact installation failed" in result.stderr
    assert "Traceback" not in result.output


def test_request_command_failure_shows_cause_in_debug_mode() -> None:
    root_cause = RuntimeError("model load exploded")
    failure = IrisRequestCommandError("runtime bootstrap failed")
    failure.__cause__ = root_cause
    arguments = request_arguments()
    with patch("iris.cli.request_service.generate_request", side_effect=failure):
        result = runner.invoke(app, ["--debug", *arguments])

    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "IrisRequestCommandError" in result.output
    assert "model load exploded" in result.output


def test_verify_request_forwards_path_and_renders_redacted_report(tmp_path: Path) -> None:
    path = tmp_path / "requests" / "v1" / f"{'1' * 64}.json"
    report = IrisRequestArtifactVerificationReportV1(
        path=path,
        artifact_type="iris.request",
        version=1,
        payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        generated_tokens=3,
        stop_reason="stop_token",
        workspace_cycles=2,
        deterministic=True,
        backend="cpu",
        dtype="float32",
    )

    with patch(
        "iris.cli.verification_service.verify_request_artifact_v1",
        return_value=report,
    ) as verify:
        result = runner.invoke(
            app,
            ["verify", "request", "--path", str(path)],
            terminal_width=200,
        )

    assert result.exit_code == 0
    verify.assert_called_once_with(path)
    for expected in (
        "Request artifact verification",
        "Verification",
        "passed",
        "iris.request",
        "Payload SHA-256",
        "Configuration SHA-256",
        "Generated tokens",
        "stop_token",
        "Workspace cycles",
        "true",
        "cpu",
        "float32",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_request_auto_dispatches_arm_labelled_v2_report(tmp_path: Path) -> None:
    path = tmp_path / "requests" / "v2" / f"{'1' * 64}.json"
    report = IrisRequestArtifactVerificationReportV2(
        path=path,
        artifact_type="iris.request",
        version=2,
        arm_id="B",
        arm_kind="llm_modules_no_workspace",
        request_graph="independent_recurrent_modules_zero_memory_read_v1",
        execution_profile_set_version=3,
        execution_profile_set_sha256=(
            "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
        ),
        payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        generated_tokens=3,
        stop_reason="stop_token",
        trace_contract="empty_workspace_trace",
        workspace_cycles=0,
        cognitive_cycles=3,
        latent_token_count=8,
        zero_memory_read_cycles=3,
        sidecar_present=True,
        deterministic=True,
        backend="cpu",
        dtype="float32",
    )

    with (
        patch(
            "iris.cli.verification_v2_service.verify_request_artifact_v2",
            return_value=report,
        ) as verify_v2,
        patch("iris.cli.verification_service.verify_request_artifact_v1") as verify_v1,
    ):
        result = runner.invoke(
            app,
            ["verify", "request", "--path", str(path)],
            terminal_width=220,
        )

    assert result.exit_code == 0
    verify_v2.assert_called_once_with(path)
    verify_v1.assert_not_called()
    for expected in (
        "Arm-labelled request artifact verification",
        "passed",
        "llm_modules_no_workspace",
        "Request graph",
        "Execution profile version",
        "empty_workspace_trace",
        "Zero-memory-read cycles",
        "true",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_request_auto_dispatches_profile_v4_arm_c_report(tmp_path: Path) -> None:
    path = tmp_path / "requests" / "v3" / f"{'1' * 64}.json"
    report = IrisRequestArtifactVerificationReportV3(
        path=path,
        artifact_type="iris.request",
        version=3,
        arm_id="C",
        arm_kind="llm_modules_unlimited_shared_buffer",
        request_graph="append_only_all_proposals_prior_cycle_shared_buffer_v1",
        execution_profile_set_version=4,
        execution_profile_set_sha256=(
            "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
        ),
        shared_buffer_policy_sha256=(
            "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
        ),
        payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        generated_tokens=3,
        stop_reason="stop_token",
        trace_contract="append_only_shared_buffer_per_cycle_v1",
        workspace_cycles=3,
        cognitive_cycles=3,
        latent_token_count=8,
        zero_memory_read_cycles=1,
        pre_update_read_entry_counts=(0, 8, 16),
        post_cycle_entry_counts=(8, 16, 24),
        final_output_read_entry_count=24,
        sidecar_present=True,
        deterministic=True,
        backend="cpu",
        dtype="float32",
    )

    with (
        patch(
            "iris.cli.verification_v3_service.verify_request_artifact_v3",
            return_value=report,
        ) as verify_v3,
        patch("iris.cli.verification_v2_service.verify_request_artifact_v2") as verify_v2,
        patch("iris.cli.verification_service.verify_request_artifact_v1") as verify_v1,
    ):
        result = runner.invoke(
            app,
            ["verify", "request", "--path", str(path)],
            terminal_width=240,
        )

    assert result.exit_code == 0
    verify_v3.assert_called_once_with(path)
    verify_v2.assert_not_called()
    verify_v1.assert_not_called()
    for expected in (
        "Profile-v4 request artifact verification",
        "llm_modules_unlimited_shared_buffer",
        "Request graph",
        "Shared-buffer policy SHA-256",
        "Pre-update read entries",
        "(0, 8, 16)",
        "Post-cycle entries",
        "(8, 16, 24)",
        "Final output read entries",
        "24",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_request_auto_dispatches_checkpoint_bound_v4_report(tmp_path: Path) -> None:
    path = tmp_path / "requests" / "v4" / f"{'1' * 64}.json"
    report = IrisRequestArtifactVerificationReportV4(
        path=path,
        artifact_type="iris.request",
        version=4,
        arm_id="B",
        arm_kind="llm_modules_no_workspace",
        request_graph="independent_recurrent_modules_zero_memory_read_v1",
        execution_profile_set_version=4,
        execution_profile_set_sha256=(
            "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
        ),
        shared_buffer_policy_sha256=(
            "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
        ),
        payload_sha256="1" * 64,
        configuration_sha256="2" * 64,
        generated_tokens=3,
        stop_reason="stop_token",
        trace_contract="empty_workspace_trace",
        workspace_cycles=0,
        cognitive_cycles=3,
        latent_token_count=8,
        zero_memory_read_cycles=3,
        pre_update_read_entry_counts=(),
        post_cycle_entry_counts=(),
        final_output_read_entry_count=None,
        sidecar_present=True,
        deterministic=True,
        backend="cpu",
        dtype="float32",
        checkpoint_present=True,
        checkpoint_schema_version=2,
        checkpoint_payload_sha256="3" * 64,
        checkpoint_state_file_sha256="4" * 64,
        source_teacher_sidecar_sha256="5" * 64,
        request_sidecar_state_sha256="6" * 64,
        checkpoint_completed_steps=7,
        transferred_tensor_count=41,
    )

    with (
        patch(
            "iris.cli.verification_v4_service.verify_request_artifact_v4",
            return_value=report,
        ) as verify_v4,
        patch("iris.cli.verification_v3_service.verify_request_artifact_v3") as verify_v3,
        patch("iris.cli.verification_v2_service.verify_request_artifact_v2") as verify_v2,
        patch("iris.cli.verification_service.verify_request_artifact_v1") as verify_v1,
    ):
        result = runner.invoke(
            app,
            ["verify", "request", "--path", str(path)],
            terminal_width=260,
        )

    assert result.exit_code == 0
    verify_v4.assert_called_once_with(path)
    verify_v3.assert_not_called()
    verify_v2.assert_not_called()
    verify_v1.assert_not_called()
    for expected in (
        "Checkpoint-bound request artifact verification",
        "Checkpoint schema version",
        "Checkpoint payload SHA-256",
        "Checkpoint state file SHA-256",
        "Teacher sidecar SHA-256",
        "Request sidecar SHA-256",
        "Checkpoint completed steps",
        "Transferred tensors",
        "41",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_checkpoint_forwards_path_and_renders_redacted_report(tmp_path: Path) -> None:
    path = tmp_path / "teacher" / "v1" / ("3" * 64)
    report = IrisTeacherCheckpointVerificationReportV1(
        path=path,
        artifact_type="iris.teacher_checkpoint",
        version=1,
        payload_sha256="3" * 64,
        state_file_sha256="4" * 64,
        state_size_bytes=2_048,
        tensor_count=7,
        sidecar_state_sha256="5" * 64,
        completed_steps=1,
        dtype="float32",
    )

    with patch(
        "iris.cli.verification_service.verify_teacher_checkpoint_v1",
        return_value=report,
    ) as verify:
        result = runner.invoke(
            app,
            ["verify", "checkpoint", "--path", str(path)],
            terminal_width=200,
        )

    assert result.exit_code == 0
    verify.assert_called_once_with(path)
    for expected in (
        "Teacher checkpoint verification",
        "passed",
        "iris.teacher_checkpoint",
        "State file SHA-256",
        "State size bytes",
        "2048",
        "Tensor count",
        "Sidecar state SHA-256",
        "Completed steps",
        "float32",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_manifest_forwards_path_and_renders_redacted_report(tmp_path: Path) -> None:
    path = tmp_path / "offline-manifest.json"
    report = IrisExperimentManifestVerificationReportV1(
        path=path,
        artifact_type="iris.experiment_examples",
        version=1,
        payload_sha256="6" * 64,
        dataset_id="offline-baseline-v1",
        train_examples=2,
        validation_examples=1,
        test_examples=1,
        total_examples=4,
    )

    with patch(
        "iris.cli.verification_service.verify_experiment_manifest_v1",
        return_value=report,
    ) as verify:
        result = runner.invoke(
            app,
            ["verify", "manifest", "--path", str(path)],
            terminal_width=200,
        )

    assert result.exit_code == 0
    verify.assert_called_once_with(path)
    for expected in (
        "Experiment manifest verification",
        "passed",
        "iris.experiment_examples",
        "offline-baseline-v1",
        "Train examples",
        "Validation examples",
        "Test examples",
        "Total examples",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_four_arm_protocol_renders_explicit_nonexecution_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "four-arm-protocol.json"
    report = IrisFourArmProtocolVerificationReportV1(
        path=path,
        artifact_type="iris.four_arm_protocol_declaration",
        version=1,
        payload_sha256="c" * 64,
        protocol_id="iris-gw-0.1-four-arm",
        arm_count=4,
        task_family_count=5,
        unresolved_item_count=10,
        main_backbone_repository="Qwen/Qwen3-1.7B",
        debug_backbone_repository="Qwen/Qwen3-0.6B",
        execution_status="blocked_until_explicit_protocol_and_arm_implementations",
    )

    with patch(
        "iris.cli.verification_service.verify_four_arm_protocol_declaration_v1",
        return_value=report,
    ) as verify:
        result = runner.invoke(
            app,
            ["verify", "four-arm-protocol", "--path", str(path)],
            terminal_width=220,
        )

    assert result.exit_code == 0
    verify.assert_called_once_with(path)
    for expected in (
        "Four-arm protocol declaration verification",
        "passed",
        "iris.four_arm_protocol_declaration",
        "iris-gw-0.1-four-arm",
        "Arms",
        "Task families",
        "Unresolved protocol items",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-0.6B",
        "blocked",
        "A/B/C profiles and evaluation protocol required",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_journal_forwards_paths_and_renders_redacted_report(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "v1" / "managed-run"
    manifest_path = tmp_path / "offline-manifest.json"
    report = IrisExperimentJournalVerificationReportV1(
        path=path,
        artifact_type="iris.experiment_run",
        version=1,
        run_id="managed-run",
        run_payload_sha256="7" * 64,
        configuration_sha256="8" * 64,
        manifest_payload_sha256="6" * 64,
        execution_mode="deterministic_epochs",
        event_count=5,
        training_step_event_count=2,
        evaluation_event_count=2,
        checkpoint_event_count=1,
        optimizer_completed_steps=2,
        sidecar_state_sha256="9" * 64,
        chain_head_sha256="a" * 64,
        checkpoint_reference_count=2,
        unique_checkpoint_count=2,
        latest_checkpoint_payload_sha256="b" * 64,
        managed_phase="complete",
        completed_epochs=1,
        active_epoch=None,
        next_training_ordinal=None,
    )

    with patch(
        "iris.cli.verification_service.verify_experiment_journal_v1",
        return_value=report,
    ) as verify:
        result = runner.invoke(
            app,
            [
                "verify",
                "journal",
                "--path",
                str(path),
                "--manifest",
                str(manifest_path),
            ],
            terminal_width=200,
        )

    assert result.exit_code == 0
    verify.assert_called_once_with(path=path, manifest_path=manifest_path)
    for expected in (
        "Experiment journal verification",
        "passed",
        "iris.experiment_run",
        "managed-run",
        "deterministic_epochs",
        "Training step events",
        "Evaluation events",
        "Checkpoint events",
        "Checkpoint references",
        "Unique checkpoints",
        "complete",
        "not applicable",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_journal_auto_dispatches_arm_labelled_v2_report(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "v2" / "managed-arm-b"
    manifest_path = tmp_path / "offline-manifest.json"
    report = IrisExperimentJournalVerificationReportV2(
        path=path,
        artifact_type="iris.experiment_run",
        version=2,
        run_id="managed-arm-b",
        arm_id="B",
        arm_kind="llm_modules_no_workspace",
        request_graph="independent_recurrent_modules_zero_memory_read_v1",
        teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
        execution_profile_set_version=3,
        execution_profile_set_sha256=(
            "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
        ),
        run_payload_sha256="7" * 64,
        configuration_sha256="8" * 64,
        manifest_payload_sha256="6" * 64,
        execution_mode="deterministic_epochs",
        event_count=4,
        training_step_event_count=2,
        evaluation_event_count=1,
        checkpoint_event_count=1,
        optimizer_completed_steps=2,
        sidecar_state_sha256="9" * 64,
        chain_head_sha256="a" * 64,
        checkpoint_reference_count=2,
        unique_checkpoint_count=2,
        latest_checkpoint_schema_version=2,
        latest_checkpoint_payload_sha256="b" * 64,
        managed_phase="complete",
        completed_epochs=1,
        active_epoch=None,
        next_training_ordinal=None,
    )

    with (
        patch(
            "iris.cli.verification_v2_service.verify_experiment_journal_v2",
            return_value=report,
        ) as verify_v2,
        patch("iris.cli.verification_service.verify_experiment_journal_v1") as verify_v1,
    ):
        result = runner.invoke(
            app,
            [
                "verify",
                "journal",
                "--path",
                str(path),
                "--manifest",
                str(manifest_path),
            ],
            terminal_width=220,
        )

    assert result.exit_code == 0
    verify_v2.assert_called_once_with(path=path, manifest_path=manifest_path)
    verify_v1.assert_not_called()
    for expected in (
        "Arm-labelled experiment journal verification",
        "passed",
        "managed-arm-b",
        "Arm kind",
        "llm_modules_no_workspace",
        "Execution profile version",
        "Latest checkpoint schema version",
        "complete",
    ):
        assert expected in result.stdout
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


def test_verify_journal_requires_manifest_before_calling_application(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "v1" / "managed-run"
    with patch("iris.cli.verification_service.verify_experiment_journal_v1") as verify:
        result = runner.invoke(app, ["verify", "journal", "--path", str(path)])

    assert result.exit_code == 2
    assert "--manifest" in result.output
    assert "Traceback" not in result.output
    verify.assert_not_called()


def test_verification_failure_is_concise_without_debug(tmp_path: Path) -> None:
    path = tmp_path / "offline-manifest.json"
    failure = IrisVerificationError("manifest verification failed safely")
    with patch(
        "iris.cli.verification_service.verify_experiment_manifest_v1",
        side_effect=failure,
    ) as verify:
        result = runner.invoke(app, ["verify", "manifest", "--path", str(path)])

    assert result.exit_code == 1
    verify.assert_called_once_with(path)
    assert result.stdout == ""
    assert "Error: manifest verification failed safely" in result.stderr
    assert "Traceback" not in result.output


def test_mlflow_projection_forwards_explicit_target_and_omits_target_uris(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments" / "v1" / "managed-run"
    manifest_path = tmp_path / "offline-manifest.json"
    tracking_uri = "https://private-user:private-password@mlflow.invalid"
    artifact_location = "s3://private-bucket/private-prefix"
    report = IrisMlflowProjectionReportV1(
        path=path,
        experiment_name="iris-baseline",
        experiment_id="experiment-1",
        mlflow_run_id="mlflow-run-1",
        canonical_run_id="managed-run",
        run_payload_sha256="7" * 64,
        configuration_sha256="8" * 64,
        manifest_payload_sha256="6" * 64,
        event_count=5,
        chain_head_sha256="a" * 64,
        metric_point_count=22,
        new_metric_point_count=17,
        reused_metric_point_count=5,
        reused_existing_experiment=True,
        reused_existing_run=True,
        status="finished",
    )

    with patch(
        "iris.cli.mlflow_projection_service.project_experiment_journal_to_mlflow_v1",
        return_value=report,
    ) as project:
        result = runner.invoke(
            app,
            [
                "experiment",
                "project-mlflow",
                "--path",
                str(path),
                "--manifest",
                str(manifest_path),
                "--tracking-uri",
                tracking_uri,
                "--experiment-name",
                "iris-baseline",
                "--artifact-location",
                artifact_location,
            ],
            terminal_width=220,
        )

    assert result.exit_code == 0
    project.assert_called_once_with(
        path=path,
        manifest_path=manifest_path,
        tracking_uri=tracking_uri,
        experiment_name="iris-baseline",
        artifact_location=artifact_location,
    )
    for expected in (
        "MLflow experiment projection",
        "reconciled",
        "content-redacted private metadata",
        "iris-baseline",
        "experiment-1",
        "mlflow-run-1",
        "managed-run",
        "Metric points",
        "New metric points",
        "Reused metric points",
        "finished",
    ):
        assert expected in result.stdout
    assert tracking_uri not in result.output
    assert artifact_location not in result.output
    assert "private-password" not in result.output
    _assert_verification_output_is_redacted(result.output)
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "missing_option",
    ["--tracking-uri", "--experiment-name", "--artifact-location"],
)
def test_mlflow_projection_requires_every_explicit_target_option(
    tmp_path: Path,
    missing_option: str,
) -> None:
    path = tmp_path / "experiments" / "v1" / "managed-run"
    manifest_path = tmp_path / "offline-manifest.json"
    target_options = (
        ("--tracking-uri", "sqlite:///explicit.db"),
        ("--experiment-name", "iris-baseline"),
        ("--artifact-location", "file:///explicit-artifacts"),
    )
    arguments = [
        "experiment",
        "project-mlflow",
        "--path",
        str(path),
        "--manifest",
        str(manifest_path),
    ]
    for option, value in target_options:
        if option != missing_option:
            arguments.extend((option, value))
    with patch(
        "iris.cli.mlflow_projection_service.project_experiment_journal_to_mlflow_v1"
    ) as project:
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert missing_option in result.output
    project.assert_not_called()


def test_mlflow_projection_failure_does_not_echo_target_credentials(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "v1" / "managed-run"
    manifest_path = tmp_path / "offline-manifest.json"
    tracking_uri = "https://private-user:private-password@mlflow.invalid"
    failure = IrisMlflowProjectionError("Iris MLflow projection failed")
    with patch(
        "iris.cli.mlflow_projection_service.project_experiment_journal_to_mlflow_v1",
        side_effect=failure,
    ):
        result = runner.invoke(
            app,
            [
                "experiment",
                "project-mlflow",
                "--path",
                str(path),
                "--manifest",
                str(manifest_path),
                "--tracking-uri",
                tracking_uri,
                "--experiment-name",
                "iris-baseline",
                "--artifact-location",
                "file:///private-artifacts",
            ],
        )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: Iris MLflow projection failed" in result.stderr
    assert tracking_uri not in result.output
    assert "private-password" not in result.output
    assert "Traceback" not in result.output
