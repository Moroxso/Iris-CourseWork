"""Application-boundary tests for the profile-v4 arm request command v3."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import iris.application as application
import iris.application.arm_request_command_v3 as command
from iris.application.arm_profiles import (
    ArmRequestOrchestratorV4,
    get_arm_execution_profile_v4,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import (
    IrisArmExecutionProfileError,
    IrisRequestArtifactError,
    IrisRequestCommandError,
)
from iris.application.request import (
    IrisRequestExecutionOutput,
    IrisRequestExecutionReceiptV1,
)
from iris.application.shared_buffer_request import (
    IrisSharedBufferRequestExecutionOutputV1,
)
from iris.backbone import BackboneAdapter
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.shared_buffer import SharedBufferState
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import domain_sha256
from iris.workspace import WorkspaceReaderOutput

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (2, 7)
PROFILE_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"


def _policy() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)


def _receipt() -> IrisRequestExecutionReceiptV1:
    return IrisRequestExecutionReceiptV1(
        prompt_sha256=domain_sha256("iris/request-execution/v1/prompt", b"hello"),
        strategy="greedy",
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=_policy(),
        sampling_seed=None,
    )


def _generation(text: str = "arm response") -> IrisGeneratorOutput:
    return IrisGeneratorOutput(
        text=text,
        token_ids=torch.tensor([[5, 2]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=2,
        workspace_trace=(),
    )


def _execution(
    arm_id: str,
    *,
    text: str = "arm response",
) -> IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1:
    generation = _generation(text)
    if arm_id == "C":
        return IrisSharedBufferRequestExecutionOutputV1(
            generation=generation,
            prompt_input_ids=torch.tensor([[11]], dtype=torch.int64),
            prompt_attention_mask=torch.ones((1, 1), dtype=torch.int64),
            receipt=_receipt(),
            shared_buffer_trace=cast(
                tuple[SharedBufferState, SharedBufferState, SharedBufferState],
                (),
            ),
            pre_update_reads=cast(
                tuple[WorkspaceReaderOutput, WorkspaceReaderOutput, WorkspaceReaderOutput],
                (),
            ),
            final_output_read=cast(WorkspaceReaderOutput, object()),
        )
    return IrisRequestExecutionOutput(
        generation=generation,
        prompt_input_ids=torch.tensor([[11]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 1), dtype=torch.int64),
        receipt=_receipt(),
    )


def _runtime(
    arm_id: str,
    orchestrator: MagicMock,
) -> IrisArmRequestRuntimeV3:
    return IrisArmRequestRuntimeV3(
        profile=get_arm_execution_profile_v4(arm_id),
        execution_profile_set_sha256=PROFILE_SHA256,
        backbone=cast(BackboneAdapter, object()),
        orchestrator=cast(ArmRequestOrchestratorV4, orchestrator),
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Command V3 Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=9001,
        deterministic=True,
        configuration_json="{}",
        configuration_sha256="a" * 64,
        sidecar_state_sha256=None if arm_id == "A" else "b" * 64,
    )


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_command_executes_exact_arm_once_and_persists_v3(arm_id: str) -> None:
    execution = _execution(arm_id)
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = _runtime(arm_id, orchestrator)
    config = OmegaConf.create({"offline": True})
    artifact = object()

    with (
        patch.object(command, "_compose_request_config", return_value=config) as compose,
        patch.object(
            command,
            "bootstrap_arm_request_runtime_v3",
            return_value=runtime,
        ) as bootstrap,
        patch.object(command, "build_request_artifact_v3", return_value=artifact) as build,
        patch.object(command, "write_request_artifact_v3") as write,
    ):
        output = command.generate_arm_request_v3(
            arm_id=arm_id,
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
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=_policy(),
    )
    orchestrator.execute.assert_called_once_with(
        "hello",
        strategy="greedy",
        generator=None,
    )
    assert build.call_args.kwargs["runtime"] is runtime
    assert build.call_args.kwargs["execution"] is execution
    write.assert_called_once_with(artifact)


@pytest.mark.parametrize("arm_id", [None, "E", 3, True])
def test_invalid_profile_fails_before_validation_config_or_model_work(arm_id: object) -> None:
    with (
        patch.object(command, "_validate_request_command") as validate,
        patch.object(command, "_compose_request_config") as compose,
        patch.object(command, "bootstrap_arm_request_runtime_v3") as bootstrap,
        pytest.raises(IrisArmExecutionProfileError),
    ):
        command.generate_arm_request_v3(
            arm_id=arm_id,
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


def test_bootstrap_must_return_exact_runtime_before_inference() -> None:
    orchestrator = MagicMock()
    wrong_runtime = SimpleNamespace(
        profile=get_arm_execution_profile_v4("A"),
        orchestrator=orchestrator,
    )
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(
            command,
            "bootstrap_arm_request_runtime_v3",
            return_value=wrong_runtime,
        ),
        pytest.raises(IrisRequestCommandError, match="exact IrisArmRequestRuntimeV3"),
    ):
        command.generate_arm_request_v3(
            arm_id="A",
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_not_called()


def test_bootstrap_profile_mismatch_fails_before_inference() -> None:
    orchestrator = MagicMock()
    runtime = _runtime("B", orchestrator)
    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v3", return_value=runtime),
        pytest.raises(IrisRequestCommandError, match="different profile-v4"),
    ):
        command.generate_arm_request_v3(
            arm_id="A",
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_not_called()


@pytest.mark.parametrize(("arm_id", "wrong_output_arm"), [("A", "C"), ("C", "A")])
def test_command_rejects_cross_family_execution_output(
    arm_id: str,
    wrong_output_arm: str,
) -> None:
    orchestrator = MagicMock()
    orchestrator.execute.return_value = _execution(wrong_output_arm)
    runtime = _runtime(arm_id, orchestrator)

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v3", return_value=runtime),
        patch.object(command, "build_request_artifact_v3") as build,
        patch.object(command, "write_request_artifact_v3") as write,
        pytest.raises(IrisRequestCommandError, match="execution provenance"),
    ):
        command.generate_arm_request_v3(
            arm_id=arm_id,
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    orchestrator.execute.assert_called_once()
    build.assert_not_called()
    write.assert_not_called()


def test_sampling_uses_one_owned_generator_on_runtime_device() -> None:
    execution = _execution("C", text="sampled")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = _runtime("C", orchestrator)

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v3", return_value=runtime),
        patch.object(command, "build_request_artifact_v3", return_value=object()),
        patch.object(command, "write_request_artifact_v3"),
    ):
        output = command.generate_arm_request_v3(
            arm_id="C",
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
    expected = torch.Generator(device=CPU)
    expected.manual_seed(9973)
    assert torch.equal(generator.get_state(), expected.get_state())


@pytest.mark.parametrize("failure_stage", ["build", "write"])
def test_artifact_failure_never_retries_inference(failure_stage: str) -> None:
    execution = _execution("C")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = _runtime("C", orchestrator)
    failure = IrisRequestArtifactError(f"{failure_stage} failed")
    artifact = object()

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v3", return_value=runtime),
        patch.object(
            command,
            "build_request_artifact_v3",
            return_value=artifact,
            side_effect=failure if failure_stage == "build" else None,
        ) as build,
        patch.object(
            command,
            "write_request_artifact_v3",
            side_effect=failure if failure_stage == "write" else None,
        ) as write,
        pytest.raises(IrisRequestArtifactError, match=failure_stage),
    ):
        command.generate_arm_request_v3(
            arm_id="C",
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


def test_command_v3_is_exported_from_application_package() -> None:
    assert application.generate_arm_request_v3 is command.generate_arm_request_v3
    assert (
        application.require_arm_request_command_profile_v3
        is command.require_arm_request_command_profile_v3
    )
