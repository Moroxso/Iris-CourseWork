"""Explicit-generator temperature and nucleus sampling for Iris generation."""

from dataclasses import dataclass
from math import isfinite

import torch

from iris.backbone.adapter import BackboneAdapter
from iris.generation.greedy import IrisGenerationError, _CachedIrisDecoder
from iris.generation.outputs import (
    IrisGenerationOutput,
    SecondPassGenerationPrefixOutput,
)
from iris.workspace.state import WorkspaceState


@dataclass(frozen=True, slots=True)
class TemperatureTopPSamplingConfig:
    """Strict probability-transform settings without an implicit RNG seed."""

    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if type(self.temperature) is not float or not isfinite(self.temperature):
            message = "temperature must be a finite float greater than zero"
            raise ValueError(message)
        if self.temperature <= 0.0:
            message = "temperature must be a finite float greater than zero"
            raise ValueError(message)
        if type(self.top_p) is not float or not isfinite(self.top_p):
            message = "top_p must be a finite float in (0, 1]"
            raise ValueError(message)
        if self.top_p <= 0.0 or self.top_p > 1.0:
            message = "top_p must be a finite float in (0, 1]"
            raise ValueError(message)


def _generator_matches_device(
    generator_device: torch.device,
    expected_device: torch.device,
) -> bool:
    if generator_device.type != expected_device.type:
        return False
    if expected_device.type != "cuda":
        return generator_device == expected_device
    current_index = torch.cuda.current_device()
    generator_index = current_index if generator_device.index is None else generator_device.index
    expected_index = current_index if expected_device.index is None else expected_device.index
    return generator_index == expected_index


def _clone_owned_generator(
    generator: object,
    *,
    expected_device: torch.device,
) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        message = "Sampling requires an explicit torch.Generator"
        raise IrisGenerationError(message)
    try:
        generator_matches_device = _generator_matches_device(
            generator.device,
            expected_device,
        )
    except Exception as error:
        message = "Failed to resolve the sampling-generator device"
        raise IrisGenerationError(message) from error
    if not generator_matches_device:
        message = "Sampling generator must share the generation-prefix device"
        raise IrisGenerationError(message)
    if generator is torch.default_generator:
        message = "Sampling requires a caller-owned generator, not the global default"
        raise IrisGenerationError(message)
    if expected_device.type == "cuda":
        try:
            device_index = expected_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            is_cuda_default = generator is torch.cuda.default_generators[device_index]
        except Exception as error:
            message = "Failed to validate CUDA sampling-generator ownership"
            raise IrisGenerationError(message) from error
        if is_cuda_default:
            message = "Sampling requires a caller-owned generator, not the global default"
            raise IrisGenerationError(message)
    try:
        local_generator = torch.Generator(device=expected_device)
        local_generator.set_state(generator.get_state().detach().clone())
    except Exception as error:
        message = "Failed to clone the sampling generator state"
        raise IrisGenerationError(message) from error
    return local_generator


class SamplingIrisDecoder(_CachedIrisDecoder):
    """Sample through the shared cache loop using transactional RNG state."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        *,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...],
        sampling: TemperatureTopPSamplingConfig,
    ) -> None:
        if not isinstance(sampling, TemperatureTopPSamplingConfig):
            message = "sampling must be TemperatureTopPSamplingConfig"
            raise ValueError(message)
        super().__init__(
            backbone,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        self.sampling = sampling

    @torch.inference_mode()
    def generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        workspace_trace: tuple[WorkspaceState, ...],
        generator: torch.Generator,
    ) -> IrisGenerationOutput:
        """Sample reproducibly and commit caller RNG state only after success."""
        _, _, _, device = self._validate_prefix(prefix)
        local_generator = _clone_owned_generator(generator, expected_device=device)

        output = self._generate(
            prefix,
            workspace_trace=workspace_trace,
            select_next_token=lambda logits: self._sample_next_token(
                logits,
                generator=local_generator,
            ),
        )
        try:
            generator.set_state(local_generator.get_state().detach().clone())
        except Exception as error:
            message = "Failed to commit the sampling generator state"
            raise IrisGenerationError(message) from error
        return output

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = logits.device
        with torch.autocast(device_type=device.type, enabled=False):
            safe_logits = logits[:, -1, :].to(dtype=torch.float64)
            centered = safe_logits - safe_logits.amax(dim=-1, keepdim=True)
            temperature = torch.tensor(
                self.sampling.temperature,
                dtype=torch.float64,
                device=device,
            )
            scaled = centered / temperature
            if bool(torch.isnan(scaled).any()) or bool(torch.isposinf(scaled).any()):
                message = "Temperature scaling produced NaN or positive infinity"
                raise IrisGenerationError(message)
            scaled = torch.where(
                torch.isneginf(scaled),
                torch.full_like(scaled, torch.finfo(torch.float64).min),
                scaled,
            )
            probabilities = torch.softmax(scaled, dim=-1)
            if not bool(torch.isfinite(probabilities).all()) or not bool(
                (probabilities >= 0).all()
            ):
                message = "Temperature sampling produced invalid probabilities"
                raise IrisGenerationError(message)

            filtered_probabilities = self._apply_top_p(probabilities)
            probability_sum = filtered_probabilities.sum(dim=-1, keepdim=True)
            if not bool(torch.isfinite(probability_sum).all()) or not bool(
                (probability_sum > 0).all()
            ):
                message = "Nucleus filtering removed every sampling candidate"
                raise IrisGenerationError(message)
            normalized = filtered_probabilities / probability_sum
            if not bool(torch.isfinite(normalized).all()):
                message = "Nucleus probability normalization produced NaN or Inf"
                raise IrisGenerationError(message)
            try:
                return torch.multinomial(
                    normalized,
                    num_samples=1,
                    replacement=False,
                    generator=generator,
                )
            except Exception as error:
                message = "Temperature/top-p token sampling failed"
                raise IrisGenerationError(message) from error

    def _apply_top_p(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.sampling.top_p == 1.0:
            return probabilities
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        previous_cumulative = torch.cat(
            (
                torch.zeros_like(cumulative[:, :1]),
                cumulative[:, :-1],
            ),
            dim=-1,
        )
        threshold = torch.tensor(
            self.sampling.top_p,
            dtype=torch.float64,
            device=probabilities.device,
        )
        keep = previous_cumulative < threshold
        kept_sorted = torch.where(keep, sorted_probabilities, 0.0)
        return torch.zeros_like(probabilities).scatter(
            dim=-1,
            index=sorted_indices,
            src=kept_sorted,
        )
