"""Prompt-only request execution for the frozen arm-A baseline."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from iris.application.errors import IrisBaseLlmRequestError, IrisCompositionError
from iris.application.request import (
    IrisRequestExecutionOutput,
    _capture_request_execution_receipt_v1,
)
from iris.backbone import BackboneAdapter
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    PromptOnlyGenerationPrefixAssembler,
    TemperatureTopPSamplingConfig,
)

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    hidden_size: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_new_tokens: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _PromptProvenance:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_input_ids: torch.Tensor
    expected_attention_mask: torch.Tensor
    expected_versions: tuple[int, int]
    expected_storage: tuple[tuple[torch.device, int], tuple[torch.device, int]]


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        message = f"Backbone {name} must be a positive integer"
        raise ValueError(message)
    return value


def _read_backbone_metadata(backbone: BackboneAdapter) -> _BackboneMetadata:
    hidden_size = _positive_integer("hidden_size", backbone.hidden_size)
    vocabulary_size = _positive_integer("vocabulary_size", backbone.vocabulary_size)
    max_position_embeddings = _positive_integer(
        "max_position_embeddings",
        backbone.max_position_embeddings,
    )
    max_prompt_tokens = _positive_integer("max_prompt_tokens", backbone.max_prompt_tokens)
    max_new_tokens = _positive_integer("max_new_tokens", backbone.max_new_tokens)
    device = backbone.device
    dtype = backbone.dtype
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        message = "Backbone device must be an explicit CPU or CUDA torch.device"
        raise ValueError(message)
    if device.index is not None and device.index < 0:
        message = "Backbone device index cannot be negative"
        raise ValueError(message)
    if device.type == "cuda" and device.index is None:
        message = "Backbone CUDA device must have one concrete index"
        raise ValueError(message)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Backbone dtype must be fp16, bf16, or fp32"
        raise ValueError(message)
    if device.type == "cpu" and dtype != torch.float32:
        message = "Iris-GW-0.1 requires fp32 prompt-only execution on CPU"
        raise ValueError(message)
    if backbone.trainable_parameter_count != 0:
        message = "Arm-A backbone must remain fully frozen"
        raise ValueError(message)
    if backbone.has_parameter_gradients:
        message = "Arm-A backbone parameters must not carry gradients"
        raise ValueError(message)
    if max_prompt_tokens + max_new_tokens > max_position_embeddings:
        message = "Prompt and generation maxima exceed backbone context"
        raise ValueError(message)
    return _BackboneMetadata(
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        max_position_embeddings=max_position_embeddings,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
    )


class BaseLlmRequestOrchestrator:
    """Generate from the frozen prompt alone, with no Iris sidecar or memory."""

    arm_id = "A"
    arm_kind = "base_llm"

    def __init__(
        self,
        *,
        backbone: BackboneAdapter,
        prefix_assembler: PromptOnlyGenerationPrefixAssembler,
        iris_generator: IrisGenerator,
    ) -> None:
        if not isinstance(backbone, BackboneAdapter):
            message = "backbone must be an already-loaded BackboneAdapter"
            raise TypeError(message)
        if not isinstance(prefix_assembler, PromptOnlyGenerationPrefixAssembler):
            message = "prefix_assembler must be PromptOnlyGenerationPrefixAssembler"
            raise TypeError(message)
        if not isinstance(iris_generator, IrisGenerator):
            message = "iris_generator must be IrisGenerator"
            raise TypeError(message)
        self._backbone = backbone
        self._prefix_assembler = prefix_assembler
        self._iris_generator = iris_generator
        self._metadata = _read_backbone_metadata(backbone)
        self._validate_wiring()

    @property
    def backbone(self) -> BackboneAdapter:
        """Return the exact frozen backbone owned by the baseline graph."""
        return self._backbone

    def generate(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        """Generate one base-LLM response with an explicitly empty memory trace."""
        return self.execute(prompt, strategy=strategy, generator=generator).generation

    @torch.inference_mode()
    def execute(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisRequestExecutionOutput:
        """Run one prompt-only cached decode and preserve exact prompt tokens."""
        self._validate_request(prompt, strategy=strategy, generator=generator)
        self._validate_runtime_boundary()
        tokenized = self._run_stage("prompt tokenization", lambda: self._backbone.tokenize(prompt))
        provenance = self._capture_prompt_provenance(tokenized)
        prompt_embeddings = self._run_stage(
            "prompt token embedding",
            lambda: self._backbone.embed_tokens(provenance.input_ids),
            provenance=provenance,
        )
        prefix = self._run_stage(
            "prompt-only generation prefix",
            lambda: self._prefix_assembler(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=provenance.attention_mask,
            ),
            provenance=provenance,
        )
        try:
            generation = self._iris_generator.generate(
                prefix,
                strategy=strategy,
                workspace_trace=(),
                generator=generator,
            )
        except Exception as error:
            message = "Base-LLM text generation failed"
            raise IrisBaseLlmRequestError(message) from error
        self._validate_prompt_provenance(provenance)
        self._validate_runtime_boundary()
        if type(generation) is not IrisGeneratorOutput or generation.workspace_trace != ():
            message = "Arm-A generation must return an exact empty workspace trace"
            raise IrisBaseLlmRequestError(message)
        return IrisRequestExecutionOutput(
            generation=generation,
            prompt_input_ids=provenance.input_ids,
            prompt_attention_mask=provenance.attention_mask,
            receipt=_capture_request_execution_receipt_v1(
                prompt=prompt,
                strategy=strategy,
                generator=generator,
                iris_generator=self._iris_generator,
            ),
        )

    def _validate_wiring(self) -> None:
        current = _read_backbone_metadata(self._backbone)
        if current != self._metadata:
            message = "Arm-A backbone metadata changed after composition"
            raise ValueError(message)
        prefix = self._prefix_assembler
        if any(child.training for child in prefix.modules()):
            message = "Prompt-only prefix assembler must remain in evaluation mode"
            raise ValueError(message)
        if next(prefix.parameters(), None) is not None or next(prefix.buffers(), None) is not None:
            message = "Prompt-only prefix assembler must be stateless"
            raise ValueError(message)
        if (
            prefix.hidden_size != current.hidden_size
            or prefix.max_prompt_tokens != current.max_prompt_tokens
            or prefix.max_new_tokens != current.max_new_tokens
            or prefix.max_context_tokens != current.max_position_embeddings
        ):
            message = "Prompt-only prefix assembler disagrees with backbone limits"
            raise ValueError(message)
        generator = self._iris_generator
        if (
            generator.backbone is not self._backbone
            or generator.max_new_tokens != current.max_new_tokens
            or type(generator.stop_token_ids) is not tuple
            or any(type(token_id) is not int for token_id in generator.stop_token_ids)
            or len(set(generator.stop_token_ids)) != len(generator.stop_token_ids)
            or any(
                token_id < 0 or token_id >= current.vocabulary_size
                for token_id in generator.stop_token_ids
            )
            or type(generator.sampling) is not TemperatureTopPSamplingConfig
        ):
            message = "Prompt-only generator wiring is invalid"
            raise ValueError(message)

    @staticmethod
    def _validate_request(
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None,
    ) -> None:
        if type(prompt) is not str or not prompt.strip():
            message = "prompt must contain non-whitespace text"
            raise IrisBaseLlmRequestError(message)
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "prompt must be valid scalar Unicode encodable as UTF-8"
            raise IrisBaseLlmRequestError(message) from error
        if type(strategy) is not str or strategy not in {"greedy", "sampling"}:
            message = "strategy must be either 'greedy' or 'sampling'"
            raise IrisBaseLlmRequestError(message)
        if strategy == "greedy" and generator is not None:
            message = "greedy generation does not accept a random generator"
            raise IrisBaseLlmRequestError(message)
        if strategy == "sampling" and not isinstance(generator, torch.Generator):
            message = "sampling generation requires an explicit torch.Generator"
            raise IrisBaseLlmRequestError(message)

    @staticmethod
    def _capture_prompt_provenance(
        tokenized: object,
    ) -> _PromptProvenance:
        if type(tokenized) is not dict:
            message = "Backbone tokenization did not return an exact tensor mapping"
            raise IrisBaseLlmRequestError(message)
        input_ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
        if (
            type(input_ids) is not torch.Tensor
            or type(attention_mask) is not torch.Tensor
            or input_ids.ndim != 2
            or input_ids.shape != attention_mask.shape
            or input_ids.shape[0] != 1
            or input_ids.shape[1] <= 0
            or input_ids.dtype != torch.int64
            or attention_mask.dtype != torch.int64
            or input_ids.device != attention_mask.device
            or input_ids.requires_grad
            or attention_mask.requires_grad
        ):
            message = "Backbone prompt tokenization violates the arm-A contract"
            raise IrisBaseLlmRequestError(message)
        with torch.inference_mode(False):
            expected_input_ids = input_ids.detach().clone()
            expected_attention_mask = attention_mask.detach().clone()
            owned_input_ids = expected_input_ids.detach().clone()
            owned_attention_mask = expected_attention_mask.detach().clone()
        provenance = _PromptProvenance(
            input_ids=owned_input_ids,
            attention_mask=owned_attention_mask,
            expected_input_ids=expected_input_ids,
            expected_attention_mask=expected_attention_mask,
            expected_versions=(owned_input_ids._version, owned_attention_mask._version),
            expected_storage=(
                (owned_input_ids.device, owned_input_ids.untyped_storage().data_ptr()),
                (
                    owned_attention_mask.device,
                    owned_attention_mask.untyped_storage().data_ptr(),
                ),
            ),
        )
        source_storage = {
            (input_ids.device, input_ids.untyped_storage().data_ptr()),
            (attention_mask.device, attention_mask.untyped_storage().data_ptr()),
        }
        private_tensors = (
            expected_input_ids,
            expected_attention_mask,
            owned_input_ids,
            owned_attention_mask,
        )
        private_storage = {
            (tensor.device, tensor.untyped_storage().data_ptr()) for tensor in private_tensors
        }
        if len(private_storage) != len(private_tensors) or private_storage & source_storage:
            message = "Arm-A prompt provenance must use independent tensor storage"
            raise IrisBaseLlmRequestError(message)
        BaseLlmRequestOrchestrator._validate_prompt_provenance(provenance)
        return provenance

    @staticmethod
    def _validate_prompt_provenance(provenance: _PromptProvenance) -> None:
        for actual, expected, version, storage in (
            (
                provenance.input_ids,
                provenance.expected_input_ids,
                provenance.expected_versions[0],
                provenance.expected_storage[0],
            ),
            (
                provenance.attention_mask,
                provenance.expected_attention_mask,
                provenance.expected_versions[1],
                provenance.expected_storage[1],
            ),
        ):
            try:
                unchanged = torch.equal(actual, expected)
                actual_storage = (actual.device, actual.untyped_storage().data_ptr())
            except Exception as error:
                message = "Failed to verify exact arm-A prompt token provenance"
                raise IrisBaseLlmRequestError(message) from error
            if (
                type(actual) is not torch.Tensor
                or actual.is_inference()
                or actual.requires_grad
                or actual.grad_fn is not None
                or actual._version != version
                or actual_storage != storage
                or not unchanged
            ):
                message = "Arm-A execution mutated exact prompt token provenance"
                raise IrisBaseLlmRequestError(message)

    def _validate_runtime_boundary(self) -> None:
        try:
            self._validate_wiring()
        except Exception as error:
            message = "Arm-A request runtime boundary is not ready"
            raise IrisBaseLlmRequestError(message) from error

    def _run_stage(
        self,
        name: str,
        operation: Callable[[], _ResultT],
        *,
        provenance: _PromptProvenance | None = None,
    ) -> _ResultT:
        try:
            self._validate_runtime_boundary()
            if provenance is not None:
                self._validate_prompt_provenance(provenance)
            result = operation()
            if provenance is not None:
                self._validate_prompt_provenance(provenance)
            self._validate_runtime_boundary()
            return result
        except IrisBaseLlmRequestError:
            raise
        except Exception as error:
            message = f"Arm-A request failed during {name}"
            raise IrisBaseLlmRequestError(message) from error


def build_base_llm_request_orchestrator(
    *,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> BaseLlmRequestOrchestrator:
    """Compose the frozen prompt-only arm-A graph around one loaded backbone."""
    try:
        if not isinstance(backbone, BackboneAdapter):
            message = "backbone must be an already-loaded BackboneAdapter"
            raise TypeError(message)
        if type(stop_token_ids) is not tuple:
            message = "stop_token_ids must be an explicit tuple"
            raise TypeError(message)
        if any(type(token_id) is not int for token_id in stop_token_ids):
            message = "stop_token_ids must contain only integers"
            raise ValueError(message)
        if len(set(stop_token_ids)) != len(stop_token_ids):
            message = "stop_token_ids must be unique"
            raise ValueError(message)
        if type(sampling) is not TemperatureTopPSamplingConfig:
            message = "sampling must be exactly TemperatureTopPSamplingConfig"
            raise TypeError(message)
        metadata = _read_backbone_metadata(backbone)
        if any(token_id < 0 or token_id >= metadata.vocabulary_size for token_id in stop_token_ids):
            message = "stop_token_ids contains an out-of-vocabulary ID"
            raise ValueError(message)
        with torch.inference_mode(False), torch.device("cpu"):
            prefix_assembler = PromptOnlyGenerationPrefixAssembler(
                hidden_size=metadata.hidden_size,
                max_prompt_tokens=metadata.max_prompt_tokens,
                max_new_tokens=metadata.max_new_tokens,
                max_context_tokens=metadata.max_position_embeddings,
            )
            placed = prefix_assembler.to(device=metadata.device, dtype=metadata.dtype)
            if placed is not prefix_assembler:
                message = "Prompt-only prefix placement must preserve module identity"
                raise ValueError(message)
            evaluated = prefix_assembler.eval()
            if evaluated is not prefix_assembler:
                message = "Prompt-only prefix eval transition must preserve module identity"
                raise ValueError(message)
            iris_generator = IrisGenerator(
                backbone,
                max_new_tokens=metadata.max_new_tokens,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
            )
            orchestrator = BaseLlmRequestOrchestrator(
                backbone=backbone,
                prefix_assembler=prefix_assembler,
                iris_generator=iris_generator,
            )
        if _read_backbone_metadata(backbone) != metadata:
            message = "Backbone metadata changed during arm-A composition"
            raise ValueError(message)
        return orchestrator
    except IrisCompositionError:
        raise
    except Exception as error:
        message = "Arm-A request composition did not complete"
        raise IrisCompositionError(message) from error


__all__ = [
    "BaseLlmRequestOrchestrator",
    "build_base_llm_request_orchestrator",
]
