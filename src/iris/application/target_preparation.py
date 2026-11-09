"""Strict raw-text preparation for one teacher-forced Iris example."""

from dataclasses import dataclass

import torch

from iris.application.errors import IrisTeacherForcingPreparationError
from iris.backbone import BackboneAdapter, BackboneTokenizedTarget
from iris.training import IrisTeacherForcingInput


@dataclass(frozen=True, slots=True)
class _PreparationMetadata:
    """Public backbone facts that must remain stable across tokenization."""

    vocabulary_size: int
    max_new_tokens: int
    device: torch.device
    eos_token_ids: tuple[int, ...]


def _validate_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        message = f"{name} must contain non-whitespace text"
        raise IrisTeacherForcingPreparationError(message)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        message = f"{name} must contain valid UTF-8 scalar text"
        raise IrisTeacherForcingPreparationError(message) from error
    return value


def _read_metadata(backbone: BackboneAdapter) -> _PreparationMetadata:
    try:
        vocabulary_size = backbone.vocabulary_size
        max_new_tokens = backbone.max_new_tokens
        device = backbone.device
        eos_token_ids = backbone.eos_token_ids
    except Exception as error:
        message = "Teacher-forcing preparation metadata could not be read"
        raise IrisTeacherForcingPreparationError(message) from error
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        message = "Teacher-forcing backbone vocabulary metadata is invalid"
        raise IrisTeacherForcingPreparationError(message)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        message = "Teacher-forcing target-length metadata is invalid"
        raise IrisTeacherForcingPreparationError(message)
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        message = "Teacher-forcing backbone device metadata is invalid"
        raise IrisTeacherForcingPreparationError(message)
    if (
        type(eos_token_ids) is not tuple
        or not eos_token_ids
        or any(type(token_id) is not int for token_id in eos_token_ids)
        or len(set(eos_token_ids)) != len(eos_token_ids)
        or any(token_id < 0 or token_id >= vocabulary_size for token_id in eos_token_ids)
    ):
        message = "Teacher-forcing tokenizer EOS metadata is invalid"
        raise IrisTeacherForcingPreparationError(message)
    return _PreparationMetadata(
        vocabulary_size=vocabulary_size,
        max_new_tokens=max_new_tokens,
        device=device,
        eos_token_ids=eos_token_ids,
    )


def _validate_tokenized_target(
    tokenized: object,
    *,
    metadata: _PreparationMetadata,
    terminal_token_id: int,
) -> BackboneTokenizedTarget:
    if type(tokenized) is not BackboneTokenizedTarget:
        message = "Backbone returned invalid teacher-target provenance"
        raise IrisTeacherForcingPreparationError(message)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    if type(input_ids) is not torch.Tensor or input_ids.ndim != 2:
        message = "Prepared teacher target input IDs have an invalid shape"
        raise IrisTeacherForcingPreparationError(message)
    sequence_length = input_ids.shape[1]
    expected_shape = (1, sequence_length)
    if sequence_length < 2 or sequence_length > metadata.max_new_tokens:
        message = "Prepared teacher target length violates the configured limit"
        raise IrisTeacherForcingPreparationError(message)
    if (
        tuple(input_ids.shape) != expected_shape
        or input_ids.dtype != torch.int64
        or input_ids.device != metadata.device
        or not input_ids.is_contiguous()
        or input_ids.requires_grad
        or input_ids.grad_fn is not None
        or input_ids.is_inference()
    ):
        message = "Prepared teacher target input IDs violate the tensor contract"
        raise IrisTeacherForcingPreparationError(message)
    if (
        type(attention_mask) is not torch.Tensor
        or tuple(attention_mask.shape) != expected_shape
        or attention_mask.dtype != torch.bool
        or attention_mask.device != metadata.device
        or not attention_mask.is_contiguous()
        or attention_mask.requires_grad
        or attention_mask.grad_fn is not None
        or attention_mask.is_inference()
    ):
        message = "Prepared teacher target attention violates the tensor contract"
        raise IrisTeacherForcingPreparationError(message)
    try:
        ids_valid = bool(
            torch.logical_and(
                input_ids >= 0,
                input_ids < metadata.vocabulary_size,
            ).all()
        )
        all_attended = bool(attention_mask.all())
        terminal_valid = bool((input_ids[:, -1] == terminal_token_id).all())
    except Exception as error:
        message = "Prepared teacher target tensors could not be inspected"
        raise IrisTeacherForcingPreparationError(message) from error
    if not ids_valid:
        message = "Prepared teacher target contains an out-of-vocabulary token"
        raise IrisTeacherForcingPreparationError(message)
    if not all_attended:
        message = "Prepared teacher target must be entirely attended"
        raise IrisTeacherForcingPreparationError(message)
    if not terminal_valid:
        message = "Prepared teacher target does not end in the declared terminal"
        raise IrisTeacherForcingPreparationError(message)
    return tokenized


def prepare_teacher_forcing_input(
    *,
    backbone: BackboneAdapter,
    prompt: str,
    target_text: str,
    terminal_token_id: int,
) -> IrisTeacherForcingInput:
    """Prepare one raw prompt/target pair without running or mutating the model."""
    if not isinstance(backbone, BackboneAdapter):
        message = "backbone must be an already-loaded BackboneAdapter"
        raise IrisTeacherForcingPreparationError(message)
    validated_prompt = _validate_text(prompt, name="prompt")
    validated_target = _validate_text(target_text, name="target_text")
    if type(terminal_token_id) is not int:
        message = "terminal_token_id must be an integer"
        raise IrisTeacherForcingPreparationError(message)

    metadata = _read_metadata(backbone)
    if terminal_token_id < 0 or terminal_token_id >= metadata.vocabulary_size:
        message = "terminal_token_id is outside the backbone vocabulary"
        raise IrisTeacherForcingPreparationError(message)
    if terminal_token_id not in metadata.eos_token_ids:
        message = "terminal_token_id must be one declared tokenizer EOS"
        raise IrisTeacherForcingPreparationError(message)
    if metadata.max_new_tokens < 2:
        message = "max_new_tokens cannot contain both target content and its terminal"
        raise IrisTeacherForcingPreparationError(message)

    try:
        tokenized_value = backbone.tokenize_teacher_target(
            validated_target,
            terminal_token_id=terminal_token_id,
        )
    except Exception as error:
        message = "Teacher target text could not be prepared"
        raise IrisTeacherForcingPreparationError(message) from error
    if _read_metadata(backbone) != metadata:
        message = "Backbone metadata changed during teacher-target preparation"
        raise IrisTeacherForcingPreparationError(message)
    tokenized = _validate_tokenized_target(
        tokenized_value,
        metadata=metadata,
        terminal_token_id=terminal_token_id,
    )

    try:
        with torch.inference_mode(False):
            owned_input_ids = tokenized.input_ids.detach().clone().contiguous()
            owned_attention_mask = tokenized.attention_mask.detach().clone().contiguous()
    except Exception as error:
        message = "Teacher target provenance could not be materialized safely"
        raise IrisTeacherForcingPreparationError(message) from error
    owned = BackboneTokenizedTarget(
        input_ids=owned_input_ids,
        attention_mask=owned_attention_mask,
    )
    _validate_tokenized_target(
        owned,
        metadata=metadata,
        terminal_token_id=terminal_token_id,
    )
    if (
        owned_input_ids.untyped_storage().data_ptr()
        == tokenized.input_ids.untyped_storage().data_ptr()
        or owned_attention_mask.untyped_storage().data_ptr()
        == tokenized.attention_mask.untyped_storage().data_ptr()
    ):
        message = "Teacher target output aliases backbone tokenizer storage"
        raise IrisTeacherForcingPreparationError(message)
    if _read_metadata(backbone) != metadata:
        message = "Backbone metadata changed while materializing teacher target provenance"
        raise IrisTeacherForcingPreparationError(message)

    return IrisTeacherForcingInput(
        prompt=validated_prompt,
        target_input_ids=owned_input_ids,
        target_attention_mask=owned_attention_mask,
    )
