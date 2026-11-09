from collections.abc import Callable
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

import iris.application as application
import iris.backbone as backbone_package
from iris.application.errors import IrisTeacherForcingPreparationError
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.backbone import (
    BackboneAdapter,
    BackboneError,
    BackboneForwardError,
    BackboneTokenizedTarget,
)
from iris.schemas import BackboneConfig
from iris.training import IrisTeacherForcingInput
from iris.utils.devices import DeviceInfo

CPU = torch.device("cpu")
VOCABULARY_SIZE = 16
EOS_TOKEN_ID = 14
OTHER_SPECIAL_TOKEN_ID = 15


class _StringSubclass(str):
    """Adversarial text subclass rejected by exact-type preflight."""


class _FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"

    def __init__(self) -> None:
        self.eos_token_id: object = EOS_TOKEN_ID
        self.all_special_ids: object = [EOS_TOKEN_ID, OTHER_SPECIAL_TOKEN_ID]
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.next_output: object = {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
        }
        self.callback: Callable[[], None] | None = None
        self.error: BaseException | None = None

    def __call__(self, text: str, **kwargs: object) -> object:
        self.calls.append((text, kwargs))
        if self.callback is not None:
            self.callback()
        if self.error is not None:
            raise self.error
        return self.next_output


class _FrozenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            vocab_size=VOCABULARY_SIZE,
            max_position_embeddings=32,
        )
        self.forward_calls = 0

    def forward(self, **_: object) -> Any:
        self.forward_calls += 1
        raise AssertionError("target preparation must not run the frozen model")


def _config(*, max_new_tokens: int = 4) -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "target-preparation-test",
            "repository": "offline/test",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": "fp32",
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {
                "max_prompt_tokens": 8,
                "max_new_tokens": max_new_tokens,
            },
        }
    )


def _cpu_info() -> DeviceInfo:
    return DeviceInfo(
        device=CPU,
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )


def _build_adapter(
    *,
    tokenizer: _FakeTokenizer | None = None,
    model: _FrozenModel | None = None,
    max_new_tokens: int = 4,
) -> tuple[BackboneAdapter, _FakeTokenizer, _FrozenModel]:
    tokenizer = tokenizer or _FakeTokenizer()
    model = model or _FrozenModel()
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        adapter = BackboneAdapter(_config(max_new_tokens=max_new_tokens), _cpu_info())
    return adapter, tokenizer, model


def _prepare(
    adapter: BackboneAdapter,
    *,
    prompt: str = "  exact prompt  ",
    target_text: str = "  exact target  ",
    terminal_token_id: int = EOS_TOKEN_ID,
) -> IrisTeacherForcingInput:
    return prepare_teacher_forcing_input(
        backbone=adapter,
        prompt=prompt,
        target_text=target_text,
        terminal_token_id=terminal_token_id,
    )


def test_prepares_exact_prompt_and_owned_content_plus_terminal() -> None:
    adapter, tokenizer, model = _build_adapter()
    source = tokenizer.next_output
    assert isinstance(source, dict)
    source_ids = source["input_ids"]
    source_mask = source["attention_mask"]
    assert isinstance(source_ids, torch.Tensor)
    assert isinstance(source_mask, torch.Tensor)

    output = _prepare(adapter)

    assert type(output) is IrisTeacherForcingInput
    assert output.prompt == "  exact prompt  "
    assert torch.equal(output.target_input_ids, torch.tensor([[2, 3, EOS_TOKEN_ID]]))
    assert torch.equal(output.target_attention_mask, torch.tensor([[True, True, True]]))
    assert output.target_input_ids.dtype == torch.int64
    assert output.target_attention_mask.dtype == torch.bool
    assert output.target_input_ids.device == CPU
    assert output.target_attention_mask.device == CPU
    assert output.target_input_ids.is_contiguous()
    assert output.target_attention_mask.is_contiguous()
    assert not output.target_input_ids.is_inference()
    assert not output.target_attention_mask.is_inference()
    assert (
        output.target_input_ids.untyped_storage().data_ptr()
        != source_ids.untyped_storage().data_ptr()
    )
    assert (
        output.target_attention_mask.untyped_storage().data_ptr()
        != source_mask.untyped_storage().data_ptr()
    )
    assert model.forward_calls == 0
    assert tokenizer.calls == [
        (
            "  exact target  ",
            {
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "return_attention_mask": True,
                "return_token_type_ids": False,
                "return_tensors": "pt",
            },
        )
    ]


def test_target_callback_is_deterministic_and_returns_fresh_storage() -> None:
    adapter, tokenizer, _ = _build_adapter()
    rng_before = torch.random.get_rng_state().clone()

    first = _prepare(adapter)
    second = _prepare(adapter)

    assert torch.equal(first.target_input_ids, second.target_input_ids)
    assert torch.equal(first.target_attention_mask, second.target_attention_mask)
    assert (
        first.target_input_ids.untyped_storage().data_ptr()
        != second.target_input_ids.untyped_storage().data_ptr()
    )
    assert (
        first.target_attention_mask.untyped_storage().data_ptr()
        != second.target_attention_mask.untyped_storage().data_ptr()
    )
    assert len(tokenizer.calls) == 2
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_ambient_inference_mode_still_returns_ordinary_tensors() -> None:
    adapter, _, _ = _build_adapter()

    with torch.inference_mode():
        output = _prepare(adapter)

    assert not output.target_input_ids.is_inference()
    assert not output.target_attention_mask.is_inference()
    assert output.target_input_ids._version == 0
    assert output.target_attention_mask._version == 0


def test_backbone_target_dto_is_frozen_and_public() -> None:
    value = BackboneTokenizedTarget(
        input_ids=torch.tensor([[1, EOS_TOKEN_ID]]),
        attention_mask=torch.tensor([[True, True]]),
    )

    with pytest.raises(FrozenInstanceError):
        value.input_ids = torch.tensor([[2, EOS_TOKEN_ID]])  # type: ignore[misc]

    assert backbone_package.BackboneTokenizedTarget is BackboneTokenizedTarget
    assert application.prepare_teacher_forcing_input is prepare_teacher_forcing_input
    assert application.IrisTeacherForcingPreparationError is IrisTeacherForcingPreparationError


def test_rejects_non_adapter_before_any_tokenizer_callback() -> None:
    with pytest.raises(IrisTeacherForcingPreparationError, match="already-loaded"):
        prepare_teacher_forcing_input(
            backbone=object(),  # type: ignore[arg-type]
            prompt="prompt",
            target_text="target",
            terminal_token_id=EOS_TOKEN_ID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", ""),
        ("prompt", " \t\r\n"),
        ("prompt", _StringSubclass("prompt")),
        ("prompt", "bad\ud800prompt"),
        ("target_text", ""),
        ("target_text", " \t\r\n"),
        ("target_text", _StringSubclass("target")),
        ("target_text", "bad\udffftarget"),
    ],
)
def test_rejects_invalid_text_preflight_before_tokenization(field: str, value: object) -> None:
    adapter, tokenizer, _ = _build_adapter()
    arguments: dict[str, object] = {
        "backbone": adapter,
        "prompt": "prompt",
        "target_text": "target",
        "terminal_token_id": EOS_TOKEN_ID,
    }
    arguments[field] = value

    with pytest.raises(IrisTeacherForcingPreparationError):
        prepare_teacher_forcing_input(**arguments)  # type: ignore[arg-type]

    assert tokenizer.calls == []


@pytest.mark.parametrize(
    "terminal_token_id",
    [True, 1.0, -1, VOCABULARY_SIZE, 3],
)
def test_rejects_invalid_terminal_preflight_before_tokenization(
    terminal_token_id: object,
) -> None:
    adapter, tokenizer, _ = _build_adapter()

    with pytest.raises(IrisTeacherForcingPreparationError):
        prepare_teacher_forcing_input(
            backbone=adapter,
            prompt="prompt",
            target_text="target",
            terminal_token_id=terminal_token_id,  # type: ignore[arg-type]
        )

    assert tokenizer.calls == []


def test_rejects_limit_without_room_for_content_before_tokenization() -> None:
    adapter, tokenizer, _ = _build_adapter(max_new_tokens=1)

    with pytest.raises(IrisTeacherForcingPreparationError, match="both target content"):
        _prepare(adapter)

    assert tokenizer.calls == []


def test_accepts_exact_content_limit_and_counts_terminal_inside_limit() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 3), dtype=torch.int64),
    }
    adapter, _, _ = _build_adapter(tokenizer=tokenizer, max_new_tokens=4)

    output = _prepare(adapter)

    assert output.target_input_ids.shape == (1, 4)
    assert output.target_input_ids[0, -1].item() == EOS_TOKEN_ID


def test_rejects_overlength_content_without_truncating() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 4), dtype=torch.int64),
    }
    adapter, _, _ = _build_adapter(tokenizer=tokenizer, max_new_tokens=4)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter, target_text="private overlength target")

    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0][1]["truncation"] is False
    assert "private overlength target" not in str(captured.value)


@pytest.mark.parametrize("special_token_id", [EOS_TOKEN_ID, OTHER_SPECIAL_TOKEN_ID])
def test_rejects_special_token_ids_encoded_from_raw_content(special_token_id: int) -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": torch.tensor([[2, special_token_id]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
    }
    adapter, _, _ = _build_adapter(tokenizer=tokenizer)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter, target_text="private control sequence")

    assert isinstance(captured.value.__cause__, BackboneForwardError)
    assert "private control sequence" not in str(captured.value)
    assert "private control sequence" not in str(captured.value.__cause__)


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        {"input_ids": torch.tensor([[2]], dtype=torch.int64)},
        {
            "input_ids": torch.tensor([[2]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1]], dtype=torch.int64),
            "token_type_ids": torch.tensor([[0]], dtype=torch.int64),
        },
        {
            "input_ids": [2],
            "attention_mask": torch.tensor([[1]], dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([2], dtype=torch.int64),
            "attention_mask": torch.tensor([1], dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([[2], [3]], dtype=torch.int64),
            "attention_mask": torch.ones((2, 1), dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([[2]], dtype=torch.int32),
            "attention_mask": torch.tensor([[1]], dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1]], dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([[2]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1.0]], dtype=torch.float32),
        },
        {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 0]], dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([[VOCABULARY_SIZE]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1]], dtype=torch.int64),
        },
    ],
)
def test_rejects_malformed_tokenizer_output(malformed: object) -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.next_output = malformed
    adapter, _, model = _build_adapter(tokenizer=tokenizer)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter)

    assert isinstance(captured.value.__cause__, BackboneForwardError)
    assert len(tokenizer.calls) == 1
    assert model.forward_calls == 0


@pytest.mark.parametrize(
    "special_ids",
    [
        None,
        [],
        [EOS_TOKEN_ID, True],
        [EOS_TOKEN_ID, EOS_TOKEN_ID],
        [EOS_TOKEN_ID, VOCABULARY_SIZE],
        [OTHER_SPECIAL_TOKEN_ID],
    ],
)
def test_rejects_invalid_special_token_metadata_before_callback(special_ids: object) -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.all_special_ids = special_ids
    adapter, _, _ = _build_adapter(tokenizer=tokenizer)

    with pytest.raises(IrisTeacherForcingPreparationError):
        _prepare(adapter)

    assert tokenizer.calls == []


def test_detects_tokenizer_metadata_mutation_during_callback() -> None:
    tokenizer = _FakeTokenizer()

    def mutate_metadata() -> None:
        tokenizer.padding_side = "right"

    tokenizer.callback = mutate_metadata
    adapter, _, _ = _build_adapter(tokenizer=tokenizer)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter)

    assert isinstance(captured.value.__cause__, BackboneForwardError)
    assert len(tokenizer.calls) == 1


def test_detects_frozen_model_mutation_during_callback() -> None:
    tokenizer = _FakeTokenizer()
    model = _FrozenModel()

    def switch_model_to_training() -> None:
        model.train()

    tokenizer.callback = switch_model_to_training
    adapter, _, _ = _build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter)

    assert isinstance(captured.value.__cause__, BackboneError)
    assert len(tokenizer.calls) == 1
    assert model.forward_calls == 0


class _TokenizerSentinelError(RuntimeError):
    """Distinct callback failure used to verify the public exception chain."""


def test_wraps_tokenizer_failure_with_typed_chained_errors() -> None:
    tokenizer = _FakeTokenizer()
    sentinel = _TokenizerSentinelError("opaque tokenizer failure")
    tokenizer.error = sentinel
    adapter, _, _ = _build_adapter(tokenizer=tokenizer)

    with pytest.raises(IrisTeacherForcingPreparationError) as captured:
        _prepare(adapter, target_text="private target")

    adapter_error = captured.value.__cause__
    assert isinstance(adapter_error, BackboneForwardError)
    assert adapter_error.__cause__ is sentinel
    assert "private target" not in str(captured.value)
    assert "private target" not in str(adapter_error)


def test_application_rejects_invalid_adapter_dto_and_calls_once() -> None:
    adapter, tokenizer, _ = _build_adapter()
    calls = 0

    def invalid_callback(text: str, *, terminal_token_id: int) -> BackboneTokenizedTarget:
        nonlocal calls
        calls += 1
        assert text == "target"
        assert terminal_token_id == EOS_TOKEN_ID
        return BackboneTokenizedTarget(
            input_ids=torch.tensor([[2, 3]], dtype=torch.int64),
            attention_mask=torch.tensor([[True, True]], dtype=torch.bool),
        )

    adapter.tokenize_teacher_target = invalid_callback  # type: ignore[method-assign]

    with pytest.raises(IrisTeacherForcingPreparationError, match="declared terminal"):
        _prepare(adapter, target_text="target")

    assert calls == 1
    assert tokenizer.calls == []
