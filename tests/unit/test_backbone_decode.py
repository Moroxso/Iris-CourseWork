"""Contract tests for the frozen backbone's tokenizer decode boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from iris.backbone import BackboneAdapter, BackboneDecodeError
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo

CPU = torch.device("cpu")
VOCABULARY_SIZE = 8
MAX_NEW_TOKENS = 3


@dataclass(frozen=True, slots=True)
class DecodeCall:
    token_ids: list[int]
    kwargs: dict[str, object]


class DecodeTokenizer:
    """Tokenizer double exposing independently controlled decode and EOS metadata."""

    padding_side = "right"
    truncation_side = "right"

    def __init__(
        self,
        *,
        decode_result: object = "decoded",
        eos_value: object = 7,
        decode_error: BaseException | None = None,
        eos_error: BaseException | None = None,
    ) -> None:
        self.decode_result = decode_result
        self.eos_value = eos_value
        self.decode_error = decode_error
        self.eos_error = eos_error
        self.decode_calls: list[DecodeCall] = []
        self.eos_reads = 0

    @property
    def eos_token_id(self) -> object:
        self.eos_reads += 1
        if self.eos_error is not None:
            raise self.eos_error
        return self.eos_value

    def decode(self, token_ids: object, **kwargs: object) -> object:
        assert isinstance(token_ids, list)
        assert all(type(token_id) is int for token_id in token_ids)
        self.decode_calls.append(DecodeCall(token_ids=list(token_ids), kwargs=dict(kwargs)))
        if self.decode_error is not None:
            raise self.decode_error
        return self.decode_result


class DecodeModel(torch.nn.Module):
    """Minimal frozen model carrying the adapter's validated config/device state."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCABULARY_SIZE, 4)
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            vocab_size=VOCABULARY_SIZE,
            max_position_embeddings=32,
        )
        self.generate_calls = 0

    def generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.generate_calls += 1
        raise AssertionError("adapter decode must never call model.generate")


def backbone_config() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "qwen3_0_6b",
            "repository": "Qwen/Qwen3-0.6B",
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
                "max_prompt_tokens": 16,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        }
    )


def cpu_info() -> DeviceInfo:
    return DeviceInfo(
        device=CPU,
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )


def build_adapter(
    tokenizer: DecodeTokenizer | None = None,
) -> tuple[BackboneAdapter, DecodeTokenizer, DecodeModel]:
    selected_tokenizer = tokenizer or DecodeTokenizer()
    model = DecodeModel()
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=selected_tokenizer,
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        adapter = BackboneAdapter(backbone_config(), cpu_info())
    return adapter, selected_tokenizer, model


def test_decode_passes_exact_cpu_integer_list_and_fixed_flags_without_mutation() -> None:
    tokenizer = DecodeTokenizer(decode_result="  hello \n")
    adapter, tokenizer, model = build_adapter(tokenizer)
    token_ids = torch.tensor([[0, 3, 7]], dtype=torch.int64)
    before = token_ids.clone()
    storage_pointer = token_ids.untyped_storage().data_ptr()

    text = adapter.decode_tokens(token_ids)

    assert text == "  hello \n"
    assert tokenizer.decode_calls == [
        DecodeCall(
            token_ids=[0, 3, 7],
            kwargs={
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
        )
    ]
    assert torch.equal(token_ids, before)
    assert token_ids.untyped_storage().data_ptr() == storage_pointer
    assert model.generate_calls == 0


def test_decode_preserves_empty_text_for_special_token_only_output() -> None:
    adapter, tokenizer, _ = build_adapter(DecodeTokenizer(decode_result=""))

    text = adapter.decode_tokens(torch.tensor([[7]], dtype=torch.int64))

    assert text == ""
    assert tokenizer.decode_calls[0].token_ids == [7]


def test_decode_accepts_exact_configured_length_boundary() -> None:
    adapter, tokenizer, _ = build_adapter()

    result = adapter.decode_tokens(torch.tensor([[1, 2, 3]], dtype=torch.int64))

    assert result == "decoded"
    assert tokenizer.decode_calls[0].token_ids == [1, 2, 3]


@pytest.mark.parametrize(
    "token_ids",
    [
        cast(Any, "not a tensor"),
        torch.tensor([1, 2], dtype=torch.int64),
        torch.ones(1, 1, 1, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.int64),
        torch.empty(1, 0, dtype=torch.int64),
        torch.ones(1, MAX_NEW_TOKENS + 1, dtype=torch.int64),
        torch.ones(1, 1, dtype=torch.int32),
        torch.ones(1, 1, dtype=torch.float32),
        torch.tensor([[-1]], dtype=torch.int64),
        torch.tensor([[VOCABULARY_SIZE]], dtype=torch.int64),
        torch.ones(1, 1, dtype=torch.int64, device="meta"),
    ],
)
def test_decode_rejects_malformed_empty_overlong_wrong_device_or_oov_ids(
    token_ids: object,
) -> None:
    adapter, tokenizer, _ = build_adapter()

    with pytest.raises(BackboneDecodeError):
        adapter.decode_tokens(cast(Any, token_ids))

    assert tokenizer.decode_calls == []


def test_tokenizer_exception_is_wrapped_with_original_cause() -> None:
    tokenizer_error = RuntimeError("decode callback failed")
    adapter, _, _ = build_adapter(DecodeTokenizer(decode_error=tokenizer_error))

    with pytest.raises(BackboneDecodeError, match="failed to decode") as captured:
        adapter.decode_tokens(torch.tensor([[1]], dtype=torch.int64))

    assert captured.value.__cause__ is tokenizer_error


def test_non_callable_decode_and_non_string_result_are_rejected() -> None:
    non_callable = DecodeTokenizer()
    non_callable.decode = cast(Any, None)  # type: ignore[method-assign]
    adapter, _, _ = build_adapter(non_callable)
    with pytest.raises(BackboneDecodeError, match="callable"):
        adapter.decode_tokens(torch.tensor([[1]], dtype=torch.int64))

    non_string = DecodeTokenizer(decode_result=["not", "text"])
    adapter, tokenizer, _ = build_adapter(non_string)
    with pytest.raises(BackboneDecodeError, match="return a string"):
        adapter.decode_tokens(torch.tensor([[1]], dtype=torch.int64))
    assert tokenizer.decode_calls[0].token_ids == [1]


@pytest.mark.parametrize(
    ("eos_value", "expected"),
    [
        (7, (7,)),
        ([7, 2], (7, 2)),
        ((5, 1, 7), (5, 1, 7)),
    ],
)
def test_eos_token_ids_lazily_preserve_valid_unique_metadata(
    eos_value: object,
    expected: tuple[int, ...],
) -> None:
    tokenizer = DecodeTokenizer(eos_value=eos_value)
    adapter, tokenizer, _ = build_adapter(tokenizer)

    assert tokenizer.eos_reads == 0
    result = adapter.eos_token_ids

    assert result == expected
    assert tokenizer.eos_reads == 1


@pytest.mark.parametrize(
    "eos_value",
    [
        None,
        [],
        (),
        True,
        False,
        "7",
        [7, 7],
        [7, True],
        [-1],
        [VOCABULARY_SIZE],
    ],
)
def test_invalid_eos_metadata_is_lazy_and_rejected_only_on_property_access(
    eos_value: object,
) -> None:
    tokenizer = DecodeTokenizer(eos_value=eos_value)
    adapter, tokenizer, _ = build_adapter(tokenizer)

    assert tokenizer.eos_reads == 0
    with pytest.raises(BackboneDecodeError):
        _ = adapter.eos_token_ids
    assert tokenizer.eos_reads == 1


def test_eos_metadata_exception_is_wrapped_with_original_cause() -> None:
    metadata_error = RuntimeError("EOS metadata unavailable")
    adapter, tokenizer, _ = build_adapter(DecodeTokenizer(eos_error=metadata_error))

    assert tokenizer.eos_reads == 0
    with pytest.raises(BackboneDecodeError, match="EOS metadata") as captured:
        _ = adapter.eos_token_ids

    assert captured.value.__cause__ is metadata_error
    assert tokenizer.eos_reads == 1
