"""Packaged A/B/D request command with arm-labelled artifact delivery v2."""

from __future__ import annotations

import torch

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV3,
    require_arm_request_executable_v3,
)
from iris.application.bootstrap import bootstrap_arm_request_runtime_v2
from iris.application.errors import ApplicationError, IrisRequestCommandError
from iris.application.request import IrisRequestExecutionOutput
from iris.application.request_artifact_v2 import (
    build_request_artifact_v2,
    write_request_artifact_v2,
)
from iris.application.request_command import (
    _compose_request_config,
    _create_sampling_generator,
    _validate_request_command,
    _validate_sampling_generator,
)
from iris.generation import (
    DecodingStrategy,
    IrisGeneratorOutput,
    TemperatureTopPSamplingConfig,
)


def require_arm_request_command_profile_v2(arm_id: object) -> IrisArmExecutionProfileV3:
    """Resolve one request arm before prompt, policy, configuration, or model work."""
    try:
        return require_arm_request_executable_v3(arm_id)
    except ApplicationError:
        raise
    except Exception as error:
        message = "Arm-labelled request command rejected the execution profile"
        raise IrisRequestCommandError(message) from error


def generate_arm_request_v2(
    *,
    arm_id: object,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
) -> IrisGeneratorOutput:
    """Compose, execute, and persist one explicitly labelled A, B, or D request."""
    profile = require_arm_request_command_profile_v2(arm_id)
    _validate_request_command(
        prompt=prompt,
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=allow_download,
    )
    try:
        config = _compose_request_config(allow_download=allow_download)
    except Exception as error:
        message = "Packaged arm-labelled request configuration did not compose"
        raise IrisRequestCommandError(message) from error
    try:
        runtime = bootstrap_arm_request_runtime_v2(
            profile.arm_id,
            config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = "Arm-labelled request runtime bootstrap did not complete"
        raise IrisRequestCommandError(message) from error
    if runtime.profile != profile:
        message = "Arm-labelled request bootstrap returned a different execution profile"
        raise IrisRequestCommandError(message)

    generator: torch.Generator | None = None
    if strategy == "sampling":
        if sampling_seed is None:
            message = "Validated sampling seed was unexpectedly unavailable"
            raise IrisRequestCommandError(message)
        try:
            generator = _validate_sampling_generator(
                _create_sampling_generator(
                    device=runtime.device_info.device,
                    seed=sampling_seed,
                ),
                device=runtime.device_info.device,
                seed=sampling_seed,
            )
        except Exception as error:
            message = "Sampling generator could not be created on the arm request device"
            raise IrisRequestCommandError(message) from error
    try:
        execution = runtime.orchestrator.execute(
            prompt,
            strategy=strategy,
            generator=generator,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = f"Arm-{profile.arm_id} request command did not complete"
        raise IrisRequestCommandError(message) from error
    if type(execution) is not IrisRequestExecutionOutput:
        message = "Arm-labelled request command returned invalid execution provenance"
        raise IrisRequestCommandError(message)
    output = execution.generation
    if type(output) is not IrisGeneratorOutput or type(output.text) is not str:
        message = "Arm-labelled request command returned an invalid text output"
        raise IrisRequestCommandError(message)
    artifact = build_request_artifact_v2(
        prompt=prompt,
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=allow_download,
        runtime=runtime,
        execution=execution,
    )
    write_request_artifact_v2(artifact)
    return output


__all__ = ["generate_arm_request_v2", "require_arm_request_command_profile_v2"]
