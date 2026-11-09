"""Packaged local-command service for one complete Iris text request."""

from math import isfinite

import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from iris.application.bootstrap import bootstrap_request_runtime
from iris.application.errors import ApplicationError, IrisRequestCommandError
from iris.application.request import IrisRequestExecutionOutput
from iris.application.request_artifact import (
    build_request_artifact_v1,
    write_request_artifact_v1,
)
from iris.generation import (
    DecodingStrategy,
    IrisGeneratorOutput,
    TemperatureTopPSamplingConfig,
)


def _validate_request_command(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
) -> None:
    if type(prompt) is not str or not prompt.strip():
        message = "prompt must contain non-whitespace text"
        raise IrisRequestCommandError(message)
    try:
        prompt.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        message = "prompt must be valid scalar Unicode encodable as UTF-8"
        raise IrisRequestCommandError(message) from error
    if type(strategy) is not str or strategy not in {"greedy", "sampling"}:
        message = "strategy must be either 'greedy' or 'sampling'"
        raise IrisRequestCommandError(message)
    if type(stop_token_ids) is not tuple or not stop_token_ids:
        message = "at least one explicit stop_token_id is required"
        raise IrisRequestCommandError(message)
    if any(type(token_id) is not int or token_id < 0 for token_id in stop_token_ids):
        message = "stop_token_ids must contain only nonnegative integers"
        raise IrisRequestCommandError(message)
    if len(set(stop_token_ids)) != len(stop_token_ids):
        message = "stop_token_ids must be unique"
        raise IrisRequestCommandError(message)
    if type(sampling) is not TemperatureTopPSamplingConfig:
        message = "sampling must be TemperatureTopPSamplingConfig"
        raise IrisRequestCommandError(message)
    if (
        type(sampling.temperature) is not float
        or not isfinite(sampling.temperature)
        or sampling.temperature <= 0.0
        or type(sampling.top_p) is not float
        or not isfinite(sampling.top_p)
        or not 0.0 < sampling.top_p <= 1.0
    ):
        message = "sampling policy must contain finite valid temperature and top_p values"
        raise IrisRequestCommandError(message)
    if type(allow_download) is not bool:
        message = "allow_download must be an explicit boolean"
        raise IrisRequestCommandError(message)
    if strategy == "greedy":
        if sampling_seed is not None:
            message = "greedy requests do not accept a sampling_seed"
            raise IrisRequestCommandError(message)
    elif type(sampling_seed) is not int or sampling_seed < 0 or sampling_seed > 2**63 - 1:
        message = "sampling requests require an explicit seed in [0, 2^63 - 1]"
        raise IrisRequestCommandError(message)


def _compose_request_config(*, allow_download: bool) -> DictConfig:
    local_files_only = "false" if allow_download else "true"
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        config = compose(
            config_name="config",
            overrides=[f"backbone.local_files_only={local_files_only}"],
        )
    if not isinstance(config, DictConfig):
        message = "Packaged Iris configuration did not compose to a DictConfig"
        raise TypeError(message)
    return config


def _create_sampling_generator(*, device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _validate_sampling_generator(
    generator: object,
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        message = "Sampling generator construction did not return torch.Generator"
        raise TypeError(message)
    if generator.device != device:
        message = "Sampling generator was created on the wrong runtime device"
        raise ValueError(message)
    if generator is torch.default_generator:
        message = "Sampling generator must not be the global default generator"
        raise ValueError(message)
    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            message = "Sampling generator requires a concrete CUDA runtime device"
            raise ValueError(message)
        if generator is torch.cuda.default_generators[device_index]:
            message = "Sampling generator must not be a CUDA default generator"
            raise ValueError(message)
    expected_generator = torch.Generator(device=device)
    expected_generator.manual_seed(seed)
    if not torch.equal(generator.get_state(), expected_generator.get_state()):
        message = "Sampling generator state does not match the explicit sampling seed"
        raise ValueError(message)
    return generator


def generate_request(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
) -> IrisGeneratorOutput:
    """Compose, execute once, persist provenance, and return one local response."""
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
        message = "Packaged Iris request configuration did not compose"
        raise IrisRequestCommandError(message) from error

    try:
        runtime = bootstrap_request_runtime(
            config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = "Iris request runtime bootstrap did not complete"
        raise IrisRequestCommandError(message) from error
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
        execution = runtime.orchestrator.execute(
            prompt,
            strategy=strategy,
            generator=generator,
        )
    except ApplicationError:
        raise
    except Exception as error:
        message = "Iris request command did not complete"
        raise IrisRequestCommandError(message) from error
    if type(execution) is not IrisRequestExecutionOutput:
        message = "Iris request command returned invalid execution provenance"
        raise IrisRequestCommandError(message)
    output = execution.generation
    if type(output) is not IrisGeneratorOutput or type(output.text) is not str:
        message = "Iris request command returned an invalid text output"
        raise IrisRequestCommandError(message)
    artifact = build_request_artifact_v1(
        prompt=prompt,
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=allow_download,
        runtime=runtime,
        execution=execution,
    )
    write_request_artifact_v1(artifact)
    return output
