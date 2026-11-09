"""Application-boundary tests for the packaged arm-labelled request command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import iris.application.arm_request_command as command
from iris.application.arm_profiles import get_arm_execution_profile_v3
from iris.application.errors import IrisArmExecutionProfileError, IrisRequestArtifactError
from iris.application.request import IrisRequestExecutionOutput
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.utils.devices import DeviceInfo

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (2, 7)


def _policy() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)


def _execution(text: str = "arm response") -> IrisRequestExecutionOutput:
    return IrisRequestExecutionOutput(
        generation=IrisGeneratorOutput(
            text=text,
            token_ids=torch.tensor([[5, 2]], dtype=torch.int64),
            stop_reason="stop_token",
            matched_stop_token_id=2,
            workspace_trace=(),
        ),
        prompt_input_ids=torch.tensor([[11]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 1), dtype=torch.int64),
    )


@pytest.mark.parametrize("arm_id", ["A", "B", "D"])
def test_command_forwards_exact_arm_and_persists_v2_before_return(arm_id: str) -> None:
    profile = get_arm_execution_profile_v3(arm_id)
    execution = _execution()
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = SimpleNamespace(
        profile=profile,
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Command Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
    )
    config = OmegaConf.create({"offline": True})
    artifact = object()

    with (
        patch.object(command, "_compose_request_config", return_value=config) as compose,
        patch.object(
            command,
            "bootstrap_arm_request_runtime_v2",
            return_value=runtime,
        ) as bootstrap,
        patch.object(command, "build_request_artifact_v2", return_value=artifact) as build,
        patch.object(command, "write_request_artifact_v2") as write,
    ):
        output = command.generate_arm_request_v2(
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
    kwargs = build.call_args.kwargs
    assert kwargs["runtime"] is runtime
    assert kwargs["execution"] is execution
    write.assert_called_once_with(artifact)


def test_unresolved_arm_c_fails_before_config_composition_or_bootstrap() -> None:
    with (
        patch.object(command, "_compose_request_config") as compose,
        patch.object(command, "bootstrap_arm_request_runtime_v2") as bootstrap,
        pytest.raises(IrisArmExecutionProfileError, match="arm C request graph is blocked"),
    ):
        command.generate_arm_request_v2(
            arm_id="C",
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    compose.assert_not_called()
    bootstrap.assert_not_called()


def test_sampling_uses_one_owned_generator_on_the_runtime_device() -> None:
    profile = get_arm_execution_profile_v3("A")
    execution = _execution("sampled")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = SimpleNamespace(
        profile=profile,
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Command Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
    )

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v2", return_value=runtime),
        patch.object(command, "build_request_artifact_v2", return_value=object()),
        patch.object(command, "write_request_artifact_v2"),
    ):
        output = command.generate_arm_request_v2(
            arm_id="A",
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
    expected = torch.Generator(device=CPU)
    expected.manual_seed(9973)
    assert torch.equal(generator.get_state(), expected.get_state())


@pytest.mark.parametrize("failure_stage", ["build", "write"])
def test_artifact_failure_never_retries_inference(failure_stage: str) -> None:
    profile = get_arm_execution_profile_v3("A")
    execution = _execution()
    orchestrator = MagicMock()
    orchestrator.execute.return_value = execution
    runtime = SimpleNamespace(
        profile=profile,
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Command Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
    )
    failure = IrisRequestArtifactError(f"{failure_stage} failed")
    artifact = object()
    build_side_effect = failure if failure_stage == "build" else None
    write_side_effect = failure if failure_stage == "write" else None

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v2", return_value=runtime),
        patch.object(
            command,
            "build_request_artifact_v2",
            return_value=artifact,
            side_effect=build_side_effect,
        ) as build,
        patch.object(
            command,
            "write_request_artifact_v2",
            side_effect=write_side_effect,
        ) as write,
        pytest.raises(IrisRequestArtifactError, match=failure_stage),
    ):
        command.generate_arm_request_v2(
            arm_id="A",
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
