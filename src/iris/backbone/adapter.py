"""Adapter for frozen Hugging Face causal language models."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from iris.backbone.outputs import (
    BackboneCausalOutput,
    BackboneFirstPassOutput,
    BackboneTokenizedTarget,
)
from iris.integrations.tokenizer_semantics import tokenizer_semantics_sha256
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo, resolve_torch_dtype
from iris.utils.hashing import (
    canonical_json_bytes,
    domain_sha256,
    module_state_sha256,
    tensor_sha256,
)


class BackboneError(RuntimeError):
    """Base error for backbone integration failures."""


class BackboneLoadError(BackboneError):
    """Raised when a tokenizer or model cannot be loaded safely."""


class BackboneForwardError(BackboneError):
    """Raised when prompt preparation or model output validation fails."""


class BackboneDecodeError(BackboneError):
    """Raised when validated generated token IDs cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class _TeacherTargetMetadata:
    """Tokenizer and backbone facts that must survive one target callback."""

    tokenizer_identity: int
    vocabulary_size: int
    max_new_tokens: int
    device: torch.device
    eos_token_ids: tuple[int, ...]
    special_token_ids: tuple[int, ...]
    padding_side: str
    truncation_side: str


class BackboneAdapter:
    """Load and run a frozen causal language model on an explicit device."""

    def __init__(self, config: BackboneConfig, device_info: DeviceInfo) -> None:
        self._config = config
        try:
            self._configuration_snapshot = config.model_dump_json()
        except Exception as error:
            message = "Failed to snapshot backbone configuration"
            raise BackboneLoadError(message) from error
        self._device = self._resolve_concrete_device(device_info)
        try:
            self._dtype = resolve_torch_dtype(device_info, config.dtype)
        except Exception as error:
            message = f"Failed to resolve dtype {config.dtype!r} for {device_info.name!r}"
            raise BackboneLoadError(message) from error
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()
        _ = self.hidden_size
        _ = self.num_hidden_layers
        _ = self.max_position_embeddings

    @property
    def configuration_json(self) -> str:
        """Return the immutable validated backbone configuration snapshot."""
        try:
            current = self._config.model_dump_json()
        except Exception as error:
            message = "Failed to validate backbone configuration provenance"
            raise BackboneLoadError(message) from error
        if current != self._configuration_snapshot:
            message = "Backbone configuration changed after adapter construction"
            raise BackboneLoadError(message)
        return self._configuration_snapshot

    @property
    def hidden_size(self) -> int:
        """Return the backbone hidden dimension."""
        return self._positive_config_integer("hidden_size")

    @property
    def checkpoint_revision_token(self) -> tuple[object, ...]:
        """Cheap process-local mutation guard; not a persisted weights fingerprint."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        entries = tuple(self._model.named_parameters()) + tuple(self._model.named_buffers())
        if any(module.training for module in self._model.modules()):
            raise BackboneError("Scientific checkpoint requires every frozen module in eval mode")
        return (
            id(self._model),
            id(self._tokenizer),
            self.configuration_json,
            canonical_json_bytes(self._model.config.to_dict()),
            canonical_json_bytes(getattr(self._model.config, "_attn_implementation", None)),
            bool(getattr(self._model, "is_gradient_checkpointing", False)),
            tokenizer_semantics_sha256(self._tokenizer),
            tuple(
                (name, id(value), value._version, str(value.dtype), str(value.device))
                for name, value in entries
            ),
        )

    @property
    def checkpoint_runtime_sha256(self) -> str:
        """Fingerprint actual frozen weights, HF config and tokenizer for exact resume.

        Expensive: reads all model state, without saving weights into a sidecar
        checkpoint. Call at run attachment/checkpoint boundaries, not every token.
        """
        before = self.checkpoint_revision_token
        model_type = type(self._model)
        value = domain_sha256(
            "iris/frozen-backbone-runtime/v1",
            canonical_json_bytes(
                {
                    "configuration": self.configuration_json,
                    "hf_configuration": self._model.config.to_dict(),
                    "model_class": f"{model_type.__module__}.{model_type.__qualname__}",
                    "attention_implementation": getattr(
                        self._model.config, "_attn_implementation", None
                    ),
                    "gradient_checkpointing": bool(
                        getattr(self._model, "is_gradient_checkpointing", False)
                    ),
                    "tokenizer_semantics": tokenizer_semantics_sha256(self._tokenizer),
                    "state_sha256": module_state_sha256((("backbone", self._model),)),
                    # HF rotary and other runtime buffers may be nonpersistent.
                    "all_buffers": [
                        {
                            "name": name,
                            "shape": list(buffer.shape),
                            "sha256": tensor_sha256(buffer.detach().reshape(-1)),
                        }
                        for name, buffer in self._model.named_buffers()
                    ],
                }
            ),
        )
        if self.checkpoint_revision_token != before:
            raise BackboneError("Frozen runtime changed during fingerprinting")
        return value

    @property
    def num_hidden_layers(self) -> int:
        """Return the number of transformer blocks."""
        return self._positive_config_integer("num_hidden_layers")

    @property
    def vocabulary_size(self) -> int:
        """Return the causal LM vocabulary dimension."""
        return self._positive_config_integer("vocab_size")

    @property
    def max_position_embeddings(self) -> int:
        """Return the maximum explicit logical position supported by the backbone."""
        return self._positive_config_integer("max_position_embeddings")

    @property
    def max_prompt_tokens(self) -> int:
        """Return the configured first-pass prompt limit."""
        return self._config.limits.max_prompt_tokens

    @property
    def max_new_tokens(self) -> int:
        """Return the configured generated-token limit."""
        return self._config.limits.max_new_tokens

    @property
    def device(self) -> torch.device:
        """Return the explicit device used by every frozen backbone boundary."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating dtype required at frozen backbone boundaries."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        if self._dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            message = "Backbone dtype metadata is not supported by Iris-GW-0.1"
            raise BackboneLoadError(message)
        return self._dtype

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """Return validated tokenizer EOS IDs without exposing the tokenizer."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        try:
            value = self._tokenizer.eos_token_id
        except Exception as error:
            message = "Failed to read tokenizer EOS metadata"
            raise BackboneDecodeError(message) from error
        if type(value) is int:
            token_ids = (value,)
        elif isinstance(value, (list, tuple)):
            token_ids = tuple(value)
        else:
            message = "Tokenizer EOS metadata must contain one or more token IDs"
            raise BackboneDecodeError(message)
        if not token_ids or any(type(token_id) is not int for token_id in token_ids):
            message = "Tokenizer EOS metadata must contain integer token IDs"
            raise BackboneDecodeError(message)
        if len(set(token_ids)) != len(token_ids):
            message = "Tokenizer EOS token IDs must be unique"
            raise BackboneDecodeError(message)
        if any(token_id < 0 or token_id >= self.vocabulary_size for token_id in token_ids):
            message = "Tokenizer EOS metadata contains an out-of-vocabulary ID"
            raise BackboneDecodeError(message)
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        return token_ids

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of scalar parameters that still require gradients."""
        return sum(
            parameter.numel() for parameter in self._model.parameters() if parameter.requires_grad
        )

    @property
    def has_parameter_gradients(self) -> bool:
        """Report whether any frozen parameter unexpectedly holds a gradient."""
        return any(parameter.grad is not None for parameter in self._model.parameters())

    def _positive_config_integer(self, name: str) -> int:
        value = getattr(self._model.config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            message = f"Backbone model config has invalid {name!r}: {value!r}"
            raise BackboneLoadError(message)
        return value

    @staticmethod
    def _resolve_concrete_device(device_info: DeviceInfo) -> torch.device:
        device = device_info.device
        if not isinstance(device, torch.device):
            message = "Backbone device metadata must be a torch.device"
            raise BackboneLoadError(message)
        if device.type != device_info.backend:
            message = "Backbone device metadata disagrees with its backend"
            raise BackboneLoadError(message)
        if device.type == "cpu":
            return torch.device("cpu")
        if device.type != "cuda":
            message = "Backbone device must use the CPU or CUDA backend"
            raise BackboneLoadError(message)
        if device.index is not None:
            if device.index < 0:
                message = "Backbone CUDA device index cannot be negative"
                raise BackboneLoadError(message)
            return device
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except Exception as error:
            message = "Failed to resolve a concrete CUDA device for the backbone"
            raise BackboneLoadError(message) from error

    def _load_tokenizer(self) -> Any:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self._config.repository,
                revision=self._config.tokenizer_revision,
                trust_remote_code=False,
                local_files_only=self._config.local_files_only,
                use_fast=self._config.tokenizer.use_fast,
            )
            tokenizer.padding_side = self._config.tokenizer.padding_side
            tokenizer.truncation_side = self._config.tokenizer.truncation_side
            return tokenizer
        except Exception as error:
            message = f"Failed to load tokenizer for {self._config.repository!r}"
            raise BackboneLoadError(message) from error

    def _load_model(self) -> Any:
        try:
            model: Any = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self._config.repository,
                revision=self._config.revision,
                trust_remote_code=False,
                local_files_only=self._config.local_files_only,
                dtype=self._dtype,
                use_safetensors=True,
            )
            model.requires_grad_(False)
            model.eval()
            model.to(self._device)
            self._validate_frozen_model(model)
            self._validate_model_device(model)
            return model
        except BackboneError:
            raise
        except Exception as error:
            message = f"Failed to load backbone model {self._config.repository!r}"
            raise BackboneLoadError(message) from error

    @staticmethod
    def _validate_frozen_model(model: Any) -> None:
        if model.training:
            message = "Backbone model must be in evaluation mode"
            raise BackboneLoadError(message)
        if any(parameter.requires_grad for parameter in model.parameters()):
            message = "Backbone model must not contain trainable parameters"
            raise BackboneLoadError(message)
        if any(parameter.grad is not None for parameter in model.parameters()):
            message = "Backbone model parameters must not contain gradients"
            raise BackboneLoadError(message)

    def _validate_model_device(self, model: Any) -> None:
        parameters_on_device = all(
            self._is_expected_device(parameter.device) for parameter in model.parameters()
        )
        buffers_on_device = all(
            self._is_expected_device(buffer.device) for buffer in model.buffers()
        )
        if not parameters_on_device or not buffers_on_device:
            message = f"Backbone model is not fully placed on {self._device}"
            raise BackboneLoadError(message)

    def _is_expected_device(self, actual: torch.device) -> bool:
        return actual == self._device

    def _read_teacher_target_metadata(self) -> _TeacherTargetMetadata:
        """Read and validate every fact used by target-text preparation."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        try:
            vocabulary_size = self.vocabulary_size
            max_new_tokens = self.max_new_tokens
            device = self.device
            eos_token_ids = self.eos_token_ids
            raw_special_ids = self._tokenizer.all_special_ids
            padding_side = self._tokenizer.padding_side
            truncation_side = self._tokenizer.truncation_side
        except BackboneError:
            raise
        except Exception as error:
            message = "Failed to read tokenizer metadata for teacher-target preparation"
            raise BackboneForwardError(message) from error

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            message = "Backbone max_new_tokens metadata is invalid"
            raise BackboneForwardError(message)
        if not isinstance(device, torch.device) or device != self._device:
            message = "Backbone target-token device metadata is invalid"
            raise BackboneForwardError(message)
        if not isinstance(raw_special_ids, (list, tuple)):
            message = "Tokenizer special-token metadata must be a sequence"
            raise BackboneForwardError(message)
        special_token_ids = tuple(raw_special_ids)
        if not special_token_ids or any(
            type(token_id) is not int for token_id in special_token_ids
        ):
            message = "Tokenizer special-token metadata must contain integer IDs"
            raise BackboneForwardError(message)
        if len(set(special_token_ids)) != len(special_token_ids):
            message = "Tokenizer special-token IDs must be unique"
            raise BackboneForwardError(message)
        if any(token_id < 0 or token_id >= vocabulary_size for token_id in special_token_ids):
            message = "Tokenizer special-token metadata contains an out-of-vocabulary ID"
            raise BackboneForwardError(message)
        if not set(eos_token_ids).issubset(special_token_ids):
            message = "Tokenizer EOS metadata must be included in its special-token IDs"
            raise BackboneForwardError(message)
        if padding_side != self._config.tokenizer.padding_side:
            message = "Tokenizer padding-side metadata changed after loading"
            raise BackboneForwardError(message)
        if truncation_side != self._config.tokenizer.truncation_side:
            message = "Tokenizer truncation-side metadata changed after loading"
            raise BackboneForwardError(message)

        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        return _TeacherTargetMetadata(
            tokenizer_identity=id(self._tokenizer),
            vocabulary_size=vocabulary_size,
            max_new_tokens=max_new_tokens,
            device=device,
            eos_token_ids=eos_token_ids,
            special_token_ids=special_token_ids,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )

    def _validate_teacher_target_metadata_unchanged(
        self,
        expected: _TeacherTargetMetadata,
    ) -> None:
        if self._read_teacher_target_metadata() != expected:
            message = "Backbone or tokenizer metadata changed during target preparation"
            raise BackboneForwardError(message)

    def tokenize_teacher_target(
        self,
        text: str,
        *,
        terminal_token_id: int,
    ) -> BackboneTokenizedTarget:
        """Tokenize plain target text and append one explicit tokenizer EOS."""
        if type(text) is not str or not text.strip():
            message = "Teacher target text must contain non-whitespace text"
            raise BackboneForwardError(message)
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "Teacher target text must contain valid UTF-8 scalar text"
            raise BackboneForwardError(message) from error
        if type(terminal_token_id) is not int:
            message = "Teacher target terminal token must be an integer"
            raise BackboneForwardError(message)

        metadata = self._read_teacher_target_metadata()
        if terminal_token_id < 0 or terminal_token_id >= metadata.vocabulary_size:
            message = "Teacher target terminal token is outside the backbone vocabulary"
            raise BackboneForwardError(message)
        if terminal_token_id not in metadata.eos_token_ids:
            message = "Teacher target terminal token is not a tokenizer EOS"
            raise BackboneForwardError(message)
        if metadata.max_new_tokens < 2:
            message = "Backbone target limit cannot contain both text and a terminal token"
            raise BackboneForwardError(message)

        try:
            encoded = self._tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
        except Exception as error:
            message = "Failed to tokenize teacher target text"
            raise BackboneForwardError(message) from error
        self._validate_teacher_target_metadata_unchanged(metadata)

        if not isinstance(encoded, Mapping):
            message = "Teacher target tokenizer output must be a mapping"
            raise BackboneForwardError(message)
        try:
            encoded_names = set(encoded)
            raw_input_ids = encoded.get("input_ids")
            raw_attention_mask = encoded.get("attention_mask")
        except Exception as error:
            message = "Failed to read teacher target tokenizer output"
            raise BackboneForwardError(message) from error
        self._validate_teacher_target_metadata_unchanged(metadata)
        if encoded_names != {"input_ids", "attention_mask"}:
            message = "Teacher target tokenizer returned unexpected output fields"
            raise BackboneForwardError(message)
        if type(raw_input_ids) is not torch.Tensor or raw_input_ids.ndim != 2:
            message = "Teacher target input IDs must have shape [1, target]"
            raise BackboneForwardError(message)
        if raw_input_ids.shape[0] != 1:
            message = "Teacher target tokenizer must return exactly one sequence"
            raise BackboneForwardError(message)
        content_length = raw_input_ids.shape[1]
        if content_length <= 0:
            message = "Teacher target tokenizer returned an empty sequence"
            raise BackboneForwardError(message)
        if content_length >= metadata.max_new_tokens:
            message = "Teacher target text exceeds the limit reserved for content plus terminal"
            raise BackboneForwardError(message)
        if raw_input_ids.dtype != torch.int64:
            message = "Teacher target tokenizer input IDs must use torch.int64"
            raise BackboneForwardError(message)
        if type(raw_attention_mask) is not torch.Tensor:
            message = "Teacher target tokenizer attention mask must be a tensor"
            raise BackboneForwardError(message)
        if raw_attention_mask.shape != raw_input_ids.shape:
            message = "Teacher target tokenizer tensors must have matching shapes"
            raise BackboneForwardError(message)
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if raw_attention_mask.dtype not in integer_dtypes:
            message = "Teacher target tokenizer attention mask must be boolean or integer"
            raise BackboneForwardError(message)

        try:
            if not bool(
                torch.logical_and(
                    raw_input_ids >= 0,
                    raw_input_ids < metadata.vocabulary_size,
                ).all()
            ):
                message = "Teacher target tokenizer returned an out-of-vocabulary token"
                raise BackboneForwardError(message)
            if not bool((raw_attention_mask == 1).all()):
                message = "Unpadded teacher target attention must contain only attended tokens"
                raise BackboneForwardError(message)
            if any(
                bool((raw_input_ids == special_token_id).any())
                for special_token_id in metadata.special_token_ids
            ):
                message = "Teacher target text must not encode tokenizer special tokens"
                raise BackboneForwardError(message)
        except BackboneError:
            raise
        except Exception as error:
            message = "Failed to validate teacher target tokenizer tensors"
            raise BackboneForwardError(message) from error
        self._validate_teacher_target_metadata_unchanged(metadata)

        try:
            with torch.inference_mode(False):
                content_ids = (
                    raw_input_ids.detach()
                    .clone()
                    .to(device=metadata.device, dtype=torch.int64)
                    .contiguous()
                )
                terminal = torch.full(
                    (1, 1),
                    terminal_token_id,
                    dtype=torch.int64,
                    device=metadata.device,
                )
                input_ids = torch.cat((content_ids, terminal), dim=1).contiguous()
                attention_mask = torch.ones(
                    input_ids.shape,
                    dtype=torch.bool,
                    device=metadata.device,
                ).contiguous()
        except Exception as error:
            message = "Failed to materialize teacher target tensors on the backbone device"
            raise BackboneForwardError(message) from error
        self._validate_teacher_target_metadata_unchanged(metadata)

        expected_shape = (1, content_length + 1)
        if (
            type(input_ids) is not torch.Tensor
            or tuple(input_ids.shape) != expected_shape
            or input_ids.dtype != torch.int64
            or input_ids.device != metadata.device
            or not input_ids.is_contiguous()
            or input_ids.requires_grad
            or input_ids.grad_fn is not None
            or input_ids.is_inference()
        ):
            message = "Prepared teacher target input IDs violate the ownership contract"
            raise BackboneForwardError(message)
        if (
            type(attention_mask) is not torch.Tensor
            or tuple(attention_mask.shape) != expected_shape
            or attention_mask.dtype != torch.bool
            or attention_mask.device != metadata.device
            or not attention_mask.is_contiguous()
            or attention_mask.requires_grad
            or attention_mask.grad_fn is not None
            or attention_mask.is_inference()
            or not bool(attention_mask.all())
        ):
            message = "Prepared teacher target attention violates the ownership contract"
            raise BackboneForwardError(message)
        if not torch.equal(input_ids[:, :-1], content_ids) or not bool(
            (input_ids[:, -1] == terminal_token_id).all()
        ):
            message = "Prepared teacher target does not preserve content plus terminal order"
            raise BackboneForwardError(message)
        return BackboneTokenizedTarget(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize one non-empty prompt and move all tensors explicitly."""
        if not text.strip():
            message = "Prompt must contain non-whitespace text"
            raise BackboneForwardError(message)

        try:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.limits.max_prompt_tokens,
                return_attention_mask=True,
            )
            tensors: dict[str, torch.Tensor] = {}
            for name, value in encoded.items():
                if type(value) is not torch.Tensor:
                    message = f"Tokenizer output {name!r} is not a tensor"
                    raise BackboneForwardError(message)
                tensors[name] = value.to(self._device)
        except BackboneError:
            raise
        except Exception as error:
            message = "Failed to tokenize prompt"
            raise BackboneForwardError(message) from error

        input_ids = tensors.get("input_ids")
        attention_mask = tensors.get("attention_mask")
        if input_ids is None or attention_mask is None:
            message = "Tokenizer output must contain input_ids and attention_mask"
            raise BackboneForwardError(message)
        if input_ids.dtype != torch.long:
            message = "Tokenizer input_ids must use torch.long dtype"
            raise BackboneForwardError(message)
        valid_mask_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if attention_mask.dtype not in valid_mask_dtypes:
            message = "Tokenizer attention_mask must use a boolean or integer dtype"
            raise BackboneForwardError(message)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            message = "Tokenizer tensors must have matching [batch, sequence] shapes"
            raise BackboneForwardError(message)
        if input_ids.shape[0] != 1:
            message = "BackboneAdapter supports exactly one prompt per forward pass"
            raise BackboneForwardError(message)
        sequence_length = input_ids.shape[1]
        if sequence_length == 0 or sequence_length > self._config.limits.max_prompt_tokens:
            message = "Tokenizer sequence length is empty or exceeds max_prompt_tokens"
            raise BackboneForwardError(message)
        if not bool(torch.logical_and(input_ids >= 0, input_ids < self.vocabulary_size).all()):
            message = "Tokenizer input_ids contain an out-of-vocabulary token"
            raise BackboneForwardError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Tokenizer attention_mask must contain only zeros and ones"
            raise BackboneForwardError(message)
        if not bool(attention_mask.any()):
            message = "Tokenizer attention_mask must contain an attended token"
            raise BackboneForwardError(message)
        if any(not self._is_expected_device(tensor.device) for tensor in tensors.values()):
            message = f"Tokenizer tensors are not fully placed on {self._device}"
            raise BackboneForwardError(message)
        return tensors

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed validated token IDs without exposing the private model object."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            message = "Token IDs must have shape [batch, sequence]"
            raise BackboneForwardError(message)
        if input_ids.dtype != torch.long:
            message = "Token IDs must use torch.long dtype"
            raise BackboneForwardError(message)
        if input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
            message = "BackboneAdapter embeds exactly one non-empty token sequence"
            raise BackboneForwardError(message)
        if not self._is_expected_device(input_ids.device):
            message = f"Token IDs are not on {self._device}"
            raise BackboneForwardError(message)
        if not bool(torch.logical_and(input_ids >= 0, input_ids < self.vocabulary_size).all()):
            message = "Token IDs contain an out-of-vocabulary index"
            raise BackboneForwardError(message)

        try:
            embedding_layer = self._model.get_input_embeddings()
            if not callable(embedding_layer):
                message = "Backbone does not expose a callable input embedding layer"
                raise BackboneForwardError(message)
            with torch.no_grad():
                embeddings = embedding_layer(input_ids)
        except BackboneError:
            raise
        except Exception as error:
            message = "Backbone token embedding failed"
            raise BackboneForwardError(message) from error
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)

        expected_shape = (1, input_ids.shape[1], self.hidden_size)
        if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != expected_shape:
            message = f"Token embeddings must have shape {expected_shape}"
            raise BackboneForwardError(message)
        if not torch.is_floating_point(embeddings) or embeddings.dtype != self._dtype:
            message = f"Token embeddings must use {self._dtype}"
            raise BackboneForwardError(message)
        if not self._is_expected_device(embeddings.device):
            message = f"Token embeddings are not on {self._device}"
            raise BackboneForwardError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = "Frozen token embeddings must not track gradients"
            raise BackboneForwardError(message)
        if not bool(torch.isfinite(embeddings).all()):
            message = "Token embeddings contain NaN or Inf"
            raise BackboneForwardError(message)
        return embeddings

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode one validated generated sequence with fixed deterministic flags."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            message = "Generated token IDs must have shape [1, sequence]"
            raise BackboneDecodeError(message)
        if token_ids.dtype != torch.int64:
            message = "Generated token IDs must use torch.int64"
            raise BackboneDecodeError(message)
        if (
            token_ids.shape[0] != 1
            or token_ids.shape[1] <= 0
            or token_ids.shape[1] > self._config.limits.max_new_tokens
        ):
            message = "Generated token sequence exceeds the configured decode contract"
            raise BackboneDecodeError(message)
        if not self._is_expected_device(token_ids.device):
            message = f"Generated token IDs are not on {self._device}"
            raise BackboneDecodeError(message)
        if not bool(torch.logical_and(token_ids >= 0, token_ids < self.vocabulary_size).all()):
            message = "Generated token IDs contain an out-of-vocabulary index"
            raise BackboneDecodeError(message)
        host_token_ids = token_ids.detach().to(device="cpu").reshape(-1).tolist()
        try:
            decode = self._tokenizer.decode
            if not callable(decode):
                message = "Tokenizer does not expose a callable decode method"
                raise BackboneDecodeError(message)
            text = decode(
                host_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except BackboneError:
            raise
        except Exception as error:
            message = "Tokenizer failed to decode generated token IDs"
            raise BackboneDecodeError(message) from error
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        if not isinstance(text, str):
            message = "Tokenizer decode must return a string"
            raise BackboneDecodeError(message)
        return text

    def forward_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        logits_to_keep: Literal[0, 1] = 0,
    ) -> BackboneCausalOutput:
        """Run a frozen causal pass while preserving gradients to input embeddings."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        batch_size, current_length = self._validate_inputs_embeds(inputs_embeds)
        past_sequence_length = self._validate_causal_sequence_metadata(
            attention_mask,
            position_ids,
            batch_size=batch_size,
            current_length=current_length,
            device=inputs_embeds.device,
            past_key_values=past_key_values,
        )
        self._validate_causal_options(
            labels,
            batch_size=batch_size,
            current_length=current_length,
            device=inputs_embeds.device,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        )

        model_arguments: dict[str, object] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if labels is not None:
            model_arguments["labels"] = labels
        if past_key_values is not None:
            model_arguments["past_key_values"] = past_key_values
        if logits_to_keep == 1:
            model_arguments["logits_to_keep"] = logits_to_keep
        try:
            result = self._model(**model_arguments)
        except Exception as error:
            message = "Backbone embedding forward pass failed"
            raise BackboneForwardError(message) from error
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)

        try:
            logits = getattr(result, "logits", None)
            loss = getattr(result, "loss", None)
            returned_cache = getattr(result, "past_key_values", None)
        except Exception as error:
            message = "Failed to read backbone causal output"
            raise BackboneForwardError(message) from error
        validated_logits, validated_loss, validated_cache = self._validate_causal_output(
            logits,
            loss,
            returned_cache,
            batch_size=batch_size,
            logits_length=1 if logits_to_keep == 1 else current_length,
            inputs_require_grad=inputs_embeds.requires_grad,
            labels_supplied=labels is not None,
            use_cache=use_cache,
            expected_cache_length=past_sequence_length + current_length,
        )
        return BackboneCausalOutput(
            logits=validated_logits,
            loss=validated_loss,
            past_key_values=validated_cache,
        )

    def _validate_inputs_embeds(self, inputs_embeds: torch.Tensor) -> tuple[int, int]:
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            message = "Input embeddings must have shape [batch, sequence, hidden_size]"
            raise BackboneForwardError(message)
        batch_size, current_length, hidden_size = inputs_embeds.shape
        if batch_size != 1 or current_length <= 0 or hidden_size != self.hidden_size:
            message = (
                "Input embeddings must contain one non-empty sequence with backbone hidden width"
            )
            raise BackboneForwardError(message)
        if not torch.is_floating_point(inputs_embeds) or inputs_embeds.dtype != self._dtype:
            message = f"Input embeddings must use {self._dtype}"
            raise BackboneForwardError(message)
        if not self._is_expected_device(inputs_embeds.device):
            message = f"Input embeddings are not on {self._device}"
            raise BackboneForwardError(message)
        if not bool(torch.isfinite(inputs_embeds).all()):
            message = "Input embeddings contain NaN or Inf"
            raise BackboneForwardError(message)
        return batch_size, current_length

    def _validate_causal_sequence_metadata(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        batch_size: int,
        current_length: int,
        device: torch.device,
        past_key_values: Cache | None,
    ) -> int:
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            message = "Causal attention mask must have shape [batch, total_sequence]"
            raise BackboneForwardError(message)
        if attention_mask.dtype not in integer_dtypes:
            message = "Causal attention mask must use a boolean or integer dtype"
            raise BackboneForwardError(message)
        if attention_mask.shape[0] != batch_size or attention_mask.shape[1] < current_length:
            message = "Causal attention mask has an invalid sequence shape"
            raise BackboneForwardError(message)
        if attention_mask.device != device:
            message = "Causal attention mask must share the input-embedding device"
            raise BackboneForwardError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Causal attention mask must contain only zeros and ones"
            raise BackboneForwardError(message)
        if not bool(attention_mask.any(dim=1).all()):
            message = "Every causal sequence must contain at least one attended token"
            raise BackboneForwardError(message)
        past_sequence_length = 0
        if past_key_values is not None:
            past_sequence_length = self._cache_sequence_length(past_key_values)
        if attention_mask.shape[1] != past_sequence_length + current_length:
            message = "Causal attention mask must match cached plus current sequence length"
            raise BackboneForwardError(message)

        if (
            not isinstance(position_ids, torch.Tensor)
            or tuple(position_ids.shape) != (batch_size, current_length)
            or position_ids.dtype != torch.long
        ):
            message = "Position IDs must be int64 [batch, current_sequence]"
            raise BackboneForwardError(message)
        if position_ids.device != device:
            message = "Position IDs must share the input-embedding device"
            raise BackboneForwardError(message)
        if not bool((position_ids >= 0).all()):
            message = "Position IDs must be non-negative"
            raise BackboneForwardError(message)
        if not bool((position_ids < self.max_position_embeddings).all()):
            message = "Position IDs exceed the backbone context window"
            raise BackboneForwardError(message)
        return past_sequence_length

    def _validate_causal_options(
        self,
        labels: torch.Tensor | None,
        *,
        batch_size: int,
        current_length: int,
        device: torch.device,
        past_key_values: Cache | None,
        use_cache: bool,
        logits_to_keep: Literal[0, 1],
    ) -> None:
        if not isinstance(use_cache, bool):
            message = "use_cache must be an explicit boolean"
            raise BackboneForwardError(message)
        if type(logits_to_keep) is not int or logits_to_keep not in (0, 1):
            message = "logits_to_keep must be exactly 0 or 1"
            raise BackboneForwardError(message)
        if past_key_values is not None and not use_cache:
            message = "past_key_values require use_cache=True"
            raise BackboneForwardError(message)
        if labels is None:
            return
        if logits_to_keep != 0:
            message = "Teacher-forced labels require all causal logits"
            raise BackboneForwardError(message)
        if use_cache or past_key_values is not None:
            message = "Teacher-forced labels cannot be combined with cache state"
            raise BackboneForwardError(message)
        if (
            not isinstance(labels, torch.Tensor)
            or tuple(labels.shape) != (batch_size, current_length)
            or labels.dtype != torch.long
        ):
            message = "Labels must be int64 [batch, current_sequence]"
            raise BackboneForwardError(message)
        if labels.device != device:
            message = "Labels must share the input-embedding device"
            raise BackboneForwardError(message)
        valid_labels = (labels == -100) | ((labels >= 0) & (labels < self.vocabulary_size))
        if not bool(valid_labels.all()):
            message = "Labels contain an invalid token ID"
            raise BackboneForwardError(message)
        if current_length < 2 or not bool((labels[:, 1:] != -100).any(dim=1).all()):
            message = "Every labelled sequence must contain a shifted supervised target"
            raise BackboneForwardError(message)

    def _validate_causal_output(
        self,
        logits: object,
        loss: object,
        returned_cache: object | None,
        *,
        batch_size: int,
        logits_length: int,
        inputs_require_grad: bool,
        labels_supplied: bool,
        use_cache: bool,
        expected_cache_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
            batch_size,
            logits_length,
            self.vocabulary_size,
        ):
            message = "Causal logits have an invalid shape"
            raise BackboneForwardError(message)
        if not torch.is_floating_point(logits) or logits.dtype != self._dtype:
            message = f"Causal logits must use {self._dtype}"
            raise BackboneForwardError(message)
        if not self._is_expected_device(logits.device):
            message = f"Causal logits are not on {self._device}"
            raise BackboneForwardError(message)
        if not bool(torch.isfinite(logits).all()):
            message = "Causal logits contain NaN or Inf"
            raise BackboneForwardError(message)

        if labels_supplied:
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                message = "A labelled causal pass must return a scalar loss"
                raise BackboneForwardError(message)
            if not torch.is_floating_point(loss) or not self._is_expected_device(loss.device):
                message = "Causal loss must be floating point on the backbone device"
                raise BackboneForwardError(message)
            if not bool(torch.isfinite(loss)):
                message = "Causal loss contains NaN or Inf"
                raise BackboneForwardError(message)
            if loss.requires_grad != inputs_require_grad or (
                inputs_require_grad and loss.grad_fn is None
            ):
                message = "Causal loss has a detached or unrelated input-embedding gradient state"
                raise BackboneForwardError(message)
        elif loss is not None:
            message = "An unlabelled causal pass must not return a loss"
            raise BackboneForwardError(message)

        if use_cache:
            if returned_cache is None:
                message = "A cached causal pass must return past_key_values"
                raise BackboneForwardError(message)
            actual_cache_length = self._cache_sequence_length(returned_cache)
            if actual_cache_length != expected_cache_length:
                message = "Returned cache has an invalid sequence length"
                raise BackboneForwardError(message)
        if not use_cache and returned_cache is not None:
            message = "A cache-free causal pass returned unexpected past_key_values"
            raise BackboneForwardError(message)
        if logits.requires_grad != inputs_require_grad or (
            inputs_require_grad and logits.grad_fn is None
        ):
            message = "Causal logits have a detached or unrelated input-embedding gradient state"
            raise BackboneForwardError(message)
        validated_loss = loss if isinstance(loss, torch.Tensor) else None
        validated_cache = returned_cache if isinstance(returned_cache, Cache) else None
        return logits, validated_loss, validated_cache

    @staticmethod
    def _cache_sequence_length(cache: object) -> int:
        if not isinstance(cache, Cache):
            message = "past_key_values must implement the Transformers Cache contract"
            raise BackboneForwardError(message)
        try:
            value = cache.get_seq_length()
        except Exception as error:
            message = "Failed to read past_key_values sequence length"
            raise BackboneForwardError(message) from error
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            message = "past_key_values returned an invalid sequence length"
            raise BackboneForwardError(message)
        return value

    @torch.inference_mode()
    def forward_prompt(self, text: str) -> BackboneFirstPassOutput:
        """Run one frozen pass and retain its exact prompt-token provenance."""
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        tensors = self.tokenize(text)
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        tokenized_input_ids = tensors["input_ids"]
        tokenized_attention_mask = tensors["attention_mask"]
        input_ids_snapshot = tokenized_input_ids.detach().clone()
        attention_mask_snapshot = tokenized_attention_mask.detach().clone()
        with torch.inference_mode(False):
            model_input_ids = input_ids_snapshot.detach().clone()
            model_attention_mask = attention_mask_snapshot.detach().clone()
        model_input_versions = (model_input_ids._version, model_attention_mask._version)
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)

        try:
            result = self._model(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        except Exception as error:
            message = "Backbone forward pass failed"
            raise BackboneForwardError(message) from error

        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)

        self._validate_model_prompt_inputs_unchanged(
            model_input_ids,
            model_attention_mask,
            input_ids_snapshot,
            attention_mask_snapshot,
            model_input_versions,
        )

        try:
            logits = getattr(result, "logits", None)
            hidden_states_value = getattr(result, "hidden_states", None)
        except Exception as error:
            message = "Failed to read backbone first-pass output"
            raise BackboneForwardError(message) from error

        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        self._validate_model_prompt_inputs_unchanged(
            model_input_ids,
            model_attention_mask,
            input_ids_snapshot,
            attention_mask_snapshot,
            model_input_versions,
        )
        if not isinstance(logits, torch.Tensor):
            message = "Backbone output does not contain logits"
            raise BackboneForwardError(message)
        if not isinstance(hidden_states_value, tuple) or not all(
            isinstance(state, torch.Tensor) for state in hidden_states_value
        ):
            message = "Backbone output does not contain a hidden-state tuple"
            raise BackboneForwardError(message)

        hidden_states = hidden_states_value
        self._validate_forward_tensors(logits, hidden_states, attention_mask_snapshot)
        returned_input_ids = input_ids_snapshot.detach().clone()
        returned_attention_mask = attention_mask_snapshot.detach().clone()
        self._validate_returned_prompt_provenance(
            returned_input_ids,
            returned_attention_mask,
            input_ids_snapshot,
            attention_mask_snapshot,
            forbidden_aliases=(
                tokenized_input_ids,
                tokenized_attention_mask,
                model_input_ids,
                model_attention_mask,
                input_ids_snapshot,
                attention_mask_snapshot,
            ),
        )
        self._validate_frozen_model(self._model)
        self._validate_model_device(self._model)
        self._validate_model_prompt_inputs_unchanged(
            model_input_ids,
            model_attention_mask,
            input_ids_snapshot,
            attention_mask_snapshot,
            model_input_versions,
        )
        return BackboneFirstPassOutput(
            logits=logits,
            hidden_states=hidden_states,
            attention_mask=returned_attention_mask,
            input_ids=returned_input_ids,
        )

    @staticmethod
    def _validate_model_prompt_inputs_unchanged(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expected_input_ids: torch.Tensor,
        expected_attention_mask: torch.Tensor,
        expected_versions: tuple[int, int],
    ) -> None:
        pairs = (
            ("input_ids", input_ids, expected_input_ids, expected_versions[0]),
            (
                "attention_mask",
                attention_mask,
                expected_attention_mask,
                expected_versions[1],
            ),
        )
        for name, actual, expected, expected_version in pairs:
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
            ):
                message = f"Backbone mutated first-pass {name} metadata"
                raise BackboneForwardError(message)
            try:
                actual_version = actual._version
                unchanged = torch.equal(actual, expected)
            except Exception as error:
                message = f"Failed to verify first-pass {name} provenance"
                raise BackboneForwardError(message) from error
            if actual_version != expected_version or not unchanged:
                message = f"Backbone mutated first-pass {name} values"
                raise BackboneForwardError(message)

    @staticmethod
    def _validate_returned_prompt_provenance(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expected_input_ids: torch.Tensor,
        expected_attention_mask: torch.Tensor,
        *,
        forbidden_aliases: tuple[torch.Tensor, ...],
    ) -> None:
        pairs = (
            ("input_ids", input_ids, expected_input_ids),
            ("attention_mask", attention_mask, expected_attention_mask),
        )
        try:
            forbidden_storage = {
                (tensor.device, tensor.untyped_storage().data_ptr()) for tensor in forbidden_aliases
            }
            returned_storage: set[tuple[torch.device, int]] = set()
            for name, actual, expected in pairs:
                if (
                    type(actual) is not torch.Tensor
                    or actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or actual.device != expected.device
                    or not torch.equal(actual, expected)
                    or actual.requires_grad
                    or actual.grad_fn is not None
                    or not actual.is_inference()
                ):
                    message = f"Returned first-pass {name} does not preserve provenance"
                    raise BackboneForwardError(message)
                storage = (actual.device, actual.untyped_storage().data_ptr())
                if storage in forbidden_storage or storage in returned_storage:
                    message = f"Returned first-pass {name} aliases internal prompt storage"
                    raise BackboneForwardError(message)
                returned_storage.add(storage)
        except BackboneError:
            raise
        except Exception as error:
            message = "Failed to validate returned first-pass prompt provenance"
            raise BackboneForwardError(message) from error

    def _validate_forward_tensors(
        self,
        logits: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> None:
        sequence_length = attention_mask.shape[1]
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != sequence_length:
            message = "Logits must have shape [1, sequence, vocabulary]"
            raise BackboneForwardError(message)
        if logits.shape[2] != self.vocabulary_size:
            message = f"Logits vocabulary dimension must equal {self.vocabulary_size}"
            raise BackboneForwardError(message)
        if not torch.is_floating_point(logits) or logits.dtype != self._dtype:
            message = f"Backbone logits must use {self._dtype}"
            raise BackboneForwardError(message)
        if not self._is_expected_device(logits.device):
            message = f"Backbone logits are not on {self._device}"
            raise BackboneForwardError(message)
        if logits.requires_grad or logits.grad_fn is not None:
            message = "Backbone logits must not track gradients"
            raise BackboneForwardError(message)
        expected_hidden_states = self.num_hidden_layers + 1
        if len(hidden_states) != expected_hidden_states:
            message = f"Expected {expected_hidden_states} hidden states, got {len(hidden_states)}"
            raise BackboneForwardError(message)
        for state in hidden_states:
            expected_shape = (1, sequence_length, self.hidden_size)
            if state.ndim != 3 or tuple(state.shape) != expected_shape:
                message = f"Hidden state must have shape {expected_shape}"
                raise BackboneForwardError(message)
            if not torch.is_floating_point(state) or state.dtype != self._dtype:
                message = f"Backbone hidden states must use {self._dtype}"
                raise BackboneForwardError(message)
            if not self._is_expected_device(state.device):
                message = f"Backbone hidden states are not on {self._device}"
                raise BackboneForwardError(message)
            if state.requires_grad or state.grad_fn is not None:
                message = "Backbone hidden states must not track gradients"
                raise BackboneForwardError(message)
        if not bool(torch.isfinite(logits).all()):
            message = "Backbone logits contain NaN or Inf"
            raise BackboneForwardError(message)
        if not all(bool(torch.isfinite(state).all()) for state in hidden_states):
            message = "Backbone hidden states contain NaN or Inf"
            raise BackboneForwardError(message)
