"""Application-boundary tests for checkpoint-bound arm request command v4."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import iris.application as application
import iris.application.arm_request_command_v4 as command
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointTransferResultV1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointIdentity,
    IrisTeacherArmId,
)
from iris.application.errors import (
    IrisArmExecutionProfileError,
    IrisRequestArtifactError,
    IrisRequestCommandError,
)
from iris.application.request_checkpoint_bootstrap import IrisArmRequestBootstrapResultV4
from tests.unit.test_arm_request_command_v3 import (
    CPU,
    STOP_TOKEN_IDS,
    _execution,
    _policy,
    _runtime,
)


def _identity(arm_id: IrisTeacherArmId) -> IrisArmTeacherCheckpointIdentity:
    if arm_id == "B":
        return IrisArmTeacherCheckpointIdentity(
            arm_id="B",
            checkpoint_schema_version=2,
            payload_sha256="c" * 64,
            state_file_sha256="d" * 64,
            sidecar_state_sha256="e" * 64,
            completed_steps=3,
        )
    if arm_id == "C":
        return IrisArmTeacherCheckpointIdentity(
            arm_id="C",
            checkpoint_schema_version=3,
            payload_sha256="c" * 64,
            state_file_sha256="d" * 64,
            sidecar_state_sha256="e" * 64,
            completed_steps=3,
        )
    return IrisArmTeacherCheckpointIdentity(
        arm_id="D",
        checkpoint_schema_version=1,
        payload_sha256="c" * 64,
        state_file_sha256="d" * 64,
        sidecar_state_sha256="e" * 64,
        completed_steps=3,
    )


def _bootstrap_result(
    arm_id: str,
    orchestrator: MagicMock,
) -> IrisArmRequestBootstrapResultV4:
    runtime = _runtime(arm_id, orchestrator)
    if arm_id == "A":
        return IrisArmRequestBootstrapResultV4(
            runtime=runtime,
            checkpoint_transfer=None,
        )
    identity = _identity(arm_id)  # type: ignore[arg-type]
    orchestrator.sidecar_state_sha256 = runtime.sidecar_state_sha256
    transfer = IrisArmRequestCheckpointTransferResultV1(
        runtime=runtime,
        checkpoint_identity=identity,
        transfer_contract="exact_teacher_state_to_profile_v4_request_sidecar_v1",
        source_teacher_sidecar_sha256=identity.sidecar_state_sha256,
        previous_request_sidecar_sha256="a" * 64,
        installed_request_sidecar_sha256="b" * 64,
        completed_steps=identity.completed_steps,
        state_names=("layer_sampler.projection.weight",),
        transferred_tensor_count=1,
    )
    return IrisArmRequestBootstrapResultV4(
        runtime=runtime,
        checkpoint_transfer=transfer,
    )


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_command_executes_once_and_persists_checkpoint_bound_artifact_v4(
    arm_id: str,
) -> None:
    execution = _execution(arm_id)
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    result = _bootstrap_result(arm_id, orchestrator)
    identity = None if arm_id == "A" else _identity(arm_id)  # type: ignore[arg-type]
    config = OmegaConf.create({"offline": True})
    artifact = object()

    with (
        patch.object(command, "_compose_request_config", return_value=config) as compose,
        patch.object(
            command,
            "bootstrap_arm_request_runtime_v4",
            return_value=result,
        ) as bootstrap,
        patch.object(command, "build_request_artifact_v4", return_value=artifact) as build,
        patch.object(command, "write_request_artifact_v4") as write,
    ):
        output = command.generate_arm_request_v4(
            arm_id=arm_id,
            checkpoint_identity=identity,
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert output is execution.generation
    compose.assert_called_once_with(allow_download=False)
    bootstrap.assert_called_once_with(
        arm_id,
        config,
        checkpoint_identity=identity,
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=_policy(),
    )
    orchestrator.execute.assert_called_once_with(
        "hello",
        strategy="greedy",
        generator=None,
    )
    assert build.call_args.kwargs["runtime"] is result.runtime
    assert build.call_args.kwargs["checkpoint_transfer"] is result.checkpoint_transfer
    assert build.call_args.kwargs["execution"] is execution
    write.assert_called_once_with(artifact)


@pytest.mark.parametrize(
    ("arm_id", "identity"),
    [
        ("A", _identity("B")),
        ("B", None),
        ("B", _identity("C")),
    ],
)
def test_checkpoint_policy_fails_before_request_validation_config_or_bootstrap(
    arm_id: str,
    identity: IrisArmTeacherCheckpointIdentity | None,
) -> None:
    with (
        patch.object(command, "_validate_request_command") as validate,
        patch.object(command, "_compose_request_config") as compose,
        patch.object(command, "bootstrap_arm_request_runtime_v4") as bootstrap,
        pytest.raises(IrisRequestCommandError, match="checkpoint"),
    ):
        command.generate_arm_request_v4(
            arm_id=arm_id,
            checkpoint_identity=identity,
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    validate.assert_not_called()
    compose.assert_not_called()
    bootstrap.assert_not_called()


@pytest.mark.parametrize("arm_id", [None, "E", 3, True])
def test_invalid_profile_fails_before_checkpoint_or_request_work(arm_id: object) -> None:
    with (
        patch.object(command, "_require_checkpoint_identity") as checkpoint,
        patch.object(command, "_validate_request_command") as validate,
        patch.object(command, "_compose_request_config") as compose,
        pytest.raises(IrisArmExecutionProfileError),
    ):
        command.generate_arm_request_v4(
            arm_id=arm_id,
            checkpoint_identity=None,
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    checkpoint.assert_not_called()
    validate.assert_not_called()
    compose.assert_not_called()


def test_wrong_bootstrap_result_fails_before_inference() -> None:
    orchestrator = MagicMock()
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v4", return_value=object()),
        pytest.raises(IrisRequestCommandError, match="exact result v4"),
    ):
        command.generate_arm_request_v4(
            arm_id="A",
            checkpoint_identity=None,
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_not_called()


def test_cross_family_execution_output_is_rejected_before_artifact() -> None:
    orchestrator = MagicMock()
    orchestrator.execute.return_value = _execution("A")
    result = _bootstrap_result("C", orchestrator)
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v4", return_value=result),
        patch.object(command, "build_request_artifact_v4") as build,
        pytest.raises(IrisRequestCommandError, match="shared-buffer execution"),
    ):
        command.generate_arm_request_v4(
            arm_id="C",
            checkpoint_identity=_identity("C"),
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_called_once()
    build.assert_not_called()


def test_sampling_uses_one_owned_generator_after_checkpoint_bootstrap() -> None:
    execution = _execution("B", text="sampled")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    result = _bootstrap_result("B", orchestrator)
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v4", return_value=result),
        patch.object(command, "build_request_artifact_v4", return_value=object()),
        patch.object(command, "write_request_artifact_v4"),
    ):
        output = command.generate_arm_request_v4(
            arm_id="B",
            checkpoint_identity=_identity("B"),
            prompt="hello",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=9973,
            allow_download=False,
        )

    assert output.text == "sampled"
    generator = orchestrator.execute.call_args.kwargs["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator is not torch.default_generator
    assert generator.device == CPU


@pytest.mark.parametrize("failure_stage", ["build", "write"])
def test_artifact_failure_never_retries_inference(failure_stage: str) -> None:
    execution = _execution("B")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    result = _bootstrap_result("B", orchestrator)
    failure = IrisRequestArtifactError(f"{failure_stage} failed")
    artifact = object()
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v4", return_value=result),
        patch.object(
            command,
            "build_request_artifact_v4",
            return_value=artifact,
            side_effect=failure if failure_stage == "build" else None,
        ) as build,
        patch.object(
            command,
            "write_request_artifact_v4",
            side_effect=failure if failure_stage == "write" else None,
        ) as write,
        pytest.raises(IrisRequestArtifactError, match=failure_stage),
    ):
        command.generate_arm_request_v4(
            arm_id="B",
            checkpoint_identity=_identity("B"),
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_called_once()
    build.assert_called_once()
    if failure_stage == "build":
        write.assert_not_called()
    else:
        write.assert_called_once_with(artifact)


def test_command_v4_is_exported() -> None:
    assert application.generate_arm_request_v4 is command.generate_arm_request_v4
    assert application.require_arm_request_command_profile_v4 is (
        command.require_arm_request_command_profile_v4
    )
