"""Checkpoint-bound packaged A/B/C/D request command with artifact v4 delivery."""

from __future__ import annotations

import torch

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV4,
    require_arm_request_executable_v4,
)
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import ApplicationError, IrisRequestCommandError
from iris.application.request import IrisRequestExecutionOutput
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.request_checkpoint_bootstrap import (
    IrisArmRequestBootstrapResultV4,
    bootstrap_arm_request_runtime_v4,
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


def require_arm_request_command_profile_v4(arm_id: object) -> IrisArmExecutionProfileV4:
    """Resolve one checkpoint-bound profile before any request or model work."""
    try:
        profile = require_arm_request_executable_v4(arm_id)
        if type(profile) is not IrisArmExecutionProfileV4:
            message = "Arm request command v4 requires an exact profile-v4 execution profile"
            raise TypeError(message)
        return profile
    except ApplicationError:
        raise
    except Exception as error:
        message = "Checkpoint-bound request command rejected the execution profile"
        raise IrisRequestCommandError(message) from error


def _require_checkpoint_identity(
    *,
    profile: IrisArmExecutionProfileV4,
    identity: IrisArmTeacherCheckpointIdentity | None,
) -> None:
    if profile.arm_id == "A":
        if identity is not None:
            message = "Arm A request command v4 forbids a teacher checkpoint identity"
            raise IrisRequestCommandError(message)
        return
    if type(identity) is not IrisArmTeacherCheckpointIdentity:
        message = f"Arm {profile.arm_id} request command v4 requires an exact checkpoint identity"
        raise IrisRequestCommandError(message)
    if identity.arm_id != profile.arm_id:
        message = "Request command profile and teacher checkpoint arms differ"
        raise IrisRequestCommandError(message)


def _require_bootstrap_result(
    result: object,
    *,
    profile: IrisArmExecutionProfileV4,
) -> IrisArmRequestBootstrapResultV4:
    if type(result) is not IrisArmRequestBootstrapResultV4:
        message = "Checkpoint-bound bootstrap did not return exact result v4"
        raise IrisRequestCommandError(message)
    if result.runtime.profile != profile:
        message = "Checkpoint-bound bootstrap returned a different execution profile"
        raise IrisRequestCommandError(message)
    transfer = result.checkpoint_transfer
    if (profile.arm_id == "A") is (transfer is not None):
        message = "Checkpoint-bound bootstrap returned the wrong transfer ownership"
        raise IrisRequestCommandError(message)
    return result


def _require_exact_execution(
    execution: object,
    *,
    arm_id: str,
) -> IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1:
    if arm_id == "C":
        if type(execution) is not IrisSharedBufferRequestExecutionOutputV1:
            message = "Arm-C checkpoint-bound command returned invalid shared-buffer execution"
            raise IrisRequestCommandError(message)
        return execution
    if type(execution) is not IrisRequestExecutionOutput:
        message = f"Arm-{arm_id} checkpoint-bound command returned invalid execution"
        raise IrisRequestCommandError(message)
    return execution


def generate_arm_request_v4(
    *,
    arm_id: object,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
) -> IrisGeneratorOutput:
    """Transfer a checkpoint, execute exactly once, and persist request artifact v4."""
    profile = require_arm_request_command_profile_v4(arm_id)
    _require_checkpoint_identity(profile=profile, identity=checkpoint_identity)
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
        message = "Checkpoint-bound request configuration did not compose"
        raise IrisRequestCommandError(message) from error
    try:
        bootstrapped = bootstrap_arm_request_runtime_v4(
            profile.arm_id,
            config,
            checkpoint_identity=checkpoint_identity,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = "Checkpoint-bound request runtime bootstrap did not complete"
        raise IrisRequestCommandError(message) from error
    result = _require_bootstrap_result(bootstrapped, profile=profile)
    runtime = result.runtime

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
            message = "Sampling generator could not be created on the request device"
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
        message = f"Arm-{profile.arm_id} checkpoint-bound request did not complete"
        raise IrisRequestCommandError(message) from error
    execution = _require_exact_execution(raw_execution, arm_id=profile.arm_id)
    output = execution.generation
    if type(output) is not IrisGeneratorOutput or type(output.text) is not str:
        message = "Checkpoint-bound request command returned an invalid text output"
        raise IrisRequestCommandError(message)
    artifact = build_request_artifact_v4(
        prompt=prompt,
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=allow_download,
        runtime=runtime,
        execution=execution,
        checkpoint_transfer=result.checkpoint_transfer,
    )
    write_request_artifact_v4(artifact)
    return output


__all__ = ["generate_arm_request_v4", "require_arm_request_command_profile_v4"]
