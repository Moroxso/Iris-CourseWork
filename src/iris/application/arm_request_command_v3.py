"""Packaged A/B/C/D request command with profile-v4 artifact delivery v3."""

from __future__ import annotations

import torch

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV4,
    require_arm_request_executable_v4,
)
from iris.application.bootstrap import (
    IrisArmRequestRuntimeV3,
    bootstrap_arm_request_runtime_v3,
)
from iris.application.errors import ApplicationError, IrisRequestCommandError
from iris.application.request import IrisRequestExecutionOutput
from iris.application.request_artifact_v3 import (
    build_request_artifact_v3,
    write_request_artifact_v3,
)
from iris.application.request_command import (
    _compose_request_config,
    _create_sampling_generator,
    _validate_request_command,
    _validate_sampling_generator,
)
from iris.application.shared_buffer_request import (
    IrisSharedBufferRequestExecutionOutputV1,
)
from iris.generation import (
    DecodingStrategy,
    IrisGeneratorOutput,
    TemperatureTopPSamplingConfig,
)


def require_arm_request_command_profile_v3(arm_id: object) -> IrisArmExecutionProfileV4:
    """Resolve one profile-v4 request arm before config or model work."""
    try:
        profile = require_arm_request_executable_v4(arm_id)
        if type(profile) is not IrisArmExecutionProfileV4:
            message = "Arm request command v3 requires an exact profile-v4 execution profile"
            raise TypeError(message)
        return profile
    except ApplicationError:
        raise
    except Exception as error:
        message = "Arm-labelled request command v3 rejected the execution profile"
        raise IrisRequestCommandError(message) from error


def _require_exact_runtime(
    runtime: object,
    *,
    profile: IrisArmExecutionProfileV4,
) -> IrisArmRequestRuntimeV3:
    if type(runtime) is not IrisArmRequestRuntimeV3:
        message = "Arm-labelled request bootstrap did not return exact IrisArmRequestRuntimeV3"
        raise IrisRequestCommandError(message)
    if type(runtime.profile) is not IrisArmExecutionProfileV4 or runtime.profile != profile:
        message = "Arm-labelled request bootstrap returned a different profile-v4 execution profile"
        raise IrisRequestCommandError(message)
    return runtime


def _require_exact_execution(
    execution: object,
    *,
    arm_id: str,
) -> IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1:
    if arm_id == "C":
        if type(execution) is not IrisSharedBufferRequestExecutionOutputV1:
            message = "Arm-C request command returned invalid shared-buffer execution provenance"
            raise IrisRequestCommandError(message)
        return execution
    if type(execution) is not IrisRequestExecutionOutput:
        message = f"Arm-{arm_id} request command returned invalid execution provenance"
        raise IrisRequestCommandError(message)
    return execution


def generate_arm_request_v3(
    *,
    arm_id: object,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
) -> IrisGeneratorOutput:
    """Compose, execute once, and persist one profile-v4 A/B/C/D request."""
    profile = require_arm_request_command_profile_v3(arm_id)
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
        message = "Packaged arm-labelled request v3 configuration did not compose"
        raise IrisRequestCommandError(message) from error
    try:
        bootstrapped = bootstrap_arm_request_runtime_v3(
            profile.arm_id,
            config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = "Arm-labelled request runtime v3 bootstrap did not complete"
        raise IrisRequestCommandError(message) from error
    runtime = _require_exact_runtime(bootstrapped, profile=profile)

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
        raw_execution = runtime.orchestrator.execute(
            prompt,
            strategy=strategy,
            generator=generator,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = f"Arm-{profile.arm_id} request command v3 did not complete"
        raise IrisRequestCommandError(message) from error
    execution = _require_exact_execution(raw_execution, arm_id=profile.arm_id)
    output = execution.generation
    if type(output) is not IrisGeneratorOutput or type(output.text) is not str:
        message = "Arm-labelled request command v3 returned an invalid text output"
        raise IrisRequestCommandError(message)
    artifact = build_request_artifact_v3(
        prompt=prompt,
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=allow_download,
        runtime=runtime,
        execution=execution,
    )
    write_request_artifact_v3(artifact)
    return output


__all__ = ["generate_arm_request_v3", "require_arm_request_command_profile_v3"]
