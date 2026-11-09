from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from iris.backbone import (
    BackboneAdapter,
    BackboneFirstPassOutput,
    BackboneForwardError,
    BackboneForwardOutput,
    BackboneLoadError,
)
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo


class FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.next_output: dict[str, Any] | None = None

    def __call__(self, text: str, **kwargs: object) -> dict[str, Any]:
        self.calls.append((text, kwargs))
        if self.next_output is not None:
            return self.next_output
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            vocab_size=8,
            max_position_embeddings=32,
        )
        self.calls: list[dict[str, object]] = []
        self.inference_modes: list[bool] = []
        self.next_output: SimpleNamespace | None = None

    def forward(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        if self.next_output is not None:
            return self.next_output
        hidden_states = tuple(torch.ones(1, 3, 4) for _ in range(3))
        return SimpleNamespace(logits=torch.ones(1, 3, 8), hidden_states=hidden_states)


class MutatingPromptModel(FakeModel):
    """Adversarial model that overwrites one prompt metadata tensor in place."""

    def __init__(
        self,
        target: str,
        *,
        error_after_mutation: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.error_after_mutation = error_after_mutation

    def forward(self, **kwargs: object) -> SimpleNamespace:
        tensor = cast(torch.Tensor, kwargs[self.target])
        if self.target == "input_ids":
            tensor.add_(1)
        else:
            tensor.zero_()
        if self.error_after_mutation is not None:
            raise self.error_after_mutation
        return super().forward(**kwargs)


class RestoreAfterComputeModel(FakeModel):
    """Adversarial model that restores values after mutating a prompt tensor."""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.version_before: int | None = None
        self.version_after: int | None = None
        self.mutated_snapshot: torch.Tensor | None = None

    def forward(self, **kwargs: object) -> SimpleNamespace:
        tensor = cast(torch.Tensor, kwargs[self.target])
        original = tensor.detach().clone()
        self.version_before = tensor._version
        if self.target == "input_ids":
            tensor.add_(1)
        else:
            tensor.zero_()
        self.mutated_snapshot = tensor.detach().clone()

        # Materialize a structurally valid model result while the model-call input
        # is mutated, then conceal the value change before returning it.
        output = super().forward(**kwargs)
        tensor.copy_(original)
        self.version_after = tensor._version
        return output


class GetterSideEffectOutput:
    """Result object whose selected property mutates an earlier call boundary."""

    def __init__(
        self,
        model: FakeModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        trigger: str,
        action: str,
    ) -> None:
        self._model = model
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._trigger = trigger
        self._action = action

    def _apply_side_effect(self, property_name: str) -> None:
        if property_name != self._trigger:
            return
        if self._action == "input_ids":
            self._input_ids.add_(1)
        elif self._action == "attention_mask":
            self._attention_mask.zero_()
        elif self._action == "training":
            self._model.train()
        else:
            self._model.to(torch.device("meta"))

    @property
    def logits(self) -> torch.Tensor:
        self._apply_side_effect("logits")
        return torch.ones(1, 3, 8)

    @property
    def hidden_states(self) -> tuple[torch.Tensor, ...]:
        self._apply_side_effect("hidden_states")
        return tuple(torch.ones(1, 3, 4) for _ in range(3))


class GetterSideEffectModel(FakeModel):
    """Model returning a lazy result with a controlled getter side effect."""

    def __init__(self, *, trigger: str, action: str) -> None:
        super().__init__()
        self.trigger = trigger
        self.action = action

    def forward(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return GetterSideEffectOutput(
            self,
            cast(torch.Tensor, kwargs["input_ids"]),
            cast(torch.Tensor, kwargs["attention_mask"]),
            trigger=self.trigger,
            action=self.action,
        )


class SideEffectHiddenStates(tuple[torch.Tensor, ...]):
    """Tuple that mutates model input only during forward-output validation."""

    model_input_ids: torch.Tensor
    iteration_count: int

    def __new__(
        cls,
        values: tuple[torch.Tensor, ...],
        model_input_ids: torch.Tensor,
    ) -> "SideEffectHiddenStates":
        instance = super().__new__(cls, values)
        instance.model_input_ids = model_input_ids
        instance.iteration_count = 0
        return instance

    def __iter__(self) -> Any:
        self.iteration_count += 1
        # First iteration is the DTO tuple-type check. The second occurs inside
        # `_validate_forward_tensors`, after the preceding provenance barrier.
        if self.iteration_count == 2:
            self.model_input_ids.add_(1)
        return super().__iter__()


class IteratorSideEffectModel(FakeModel):
    """Model returning a tuple subclass that mutates inputs during validation."""

    def forward(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        hidden_states = SideEffectHiddenStates(
            tuple(torch.ones(1, 3, 4) for _ in range(3)),
            cast(torch.Tensor, kwargs["input_ids"]),
        )
        return SimpleNamespace(
            logits=torch.ones(1, 3, 8),
            hidden_states=hidden_states,
        )


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
            "limits": {"max_prompt_tokens": 256, "max_new_tokens": 32},
        }
    )


def cpu_info() -> DeviceInfo:
    return DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )


def build_adapter(
    tokenizer: FakeTokenizer | None = None, model: FakeModel | None = None
) -> tuple[BackboneAdapter, FakeTokenizer, FakeModel]:
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeModel()
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        adapter = BackboneAdapter(backbone_config(), cpu_info())
    return adapter, tokenizer, model


def test_loads_safe_frozen_backbone() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer
        ) as tokenizer_load,
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as model_load,
    ):
        adapter = BackboneAdapter(backbone_config(), cpu_info())

    tokenizer_load.assert_called_once_with(
        pretrained_model_name_or_path="Qwen/Qwen3-0.6B",
        revision="tokenizer-revision",
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    model_load.assert_called_once_with(
        pretrained_model_name_or_path="Qwen/Qwen3-0.6B",
        revision="model-revision",
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.float32,
        use_safetensors=True,
    )
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert adapter.hidden_size == 4
    assert adapter.num_hidden_layers == 2
    assert adapter.vocabulary_size == 8
    assert adapter.device == torch.device("cpu")
    assert adapter.dtype == torch.float32
    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False


def test_configuration_json_is_an_immutable_validated_snapshot() -> None:
    adapter, _, _ = build_adapter()

    assert adapter.configuration_json == backbone_config().model_dump_json()
    adapter._config.local_files_only = False

    with pytest.raises(BackboneLoadError, match="configuration changed"):
        _ = adapter.configuration_json


def test_reports_trainable_parameters_and_gradients() -> None:
    adapter, _, model = build_adapter()
    model.weight.requires_grad_(True)

    assert adapter.trainable_parameter_count == model.weight.numel()
    assert adapter.has_parameter_gradients is False

    model.weight.grad = torch.ones_like(model.weight)

    assert adapter.has_parameter_gradients is True


def test_resolves_indexless_cuda_metadata_to_one_concrete_device() -> None:
    device_info = DeviceInfo(
        device=torch.device("cuda"),
        backend="cuda",
        name="Test GPU",
        total_memory_bytes=1024,
        supports_fp16=True,
        supports_bf16=True,
    )

    with patch("torch.cuda.current_device", return_value=2) as current_device:
        resolved = BackboneAdapter._resolve_concrete_device(device_info)

    assert resolved == torch.device("cuda", 2)
    current_device.assert_called_once_with()


def test_tokenize_rejects_empty_and_whitespace_prompts() -> None:
    adapter, _, _ = build_adapter()

    with pytest.raises(BackboneForwardError, match="non-whitespace"):
        adapter.tokenize("")
    with pytest.raises(BackboneForwardError, match="non-whitespace"):
        adapter.tokenize("   \n")


def test_tokenize_enforces_limits_and_explicit_device() -> None:
    adapter, tokenizer, _ = build_adapter()

    tensors = adapter.tokenize("hello")

    assert tokenizer.calls == [
        (
            "hello",
            {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 256,
                "return_attention_mask": True,
            },
        )
    ]
    assert all(tensor.device.type == "cpu" for tensor in tensors.values())


@pytest.mark.parametrize(
    ("tokenizer_output", "message"),
    [
        (
            {"input_ids": torch.tensor([[1.0]]), "attention_mask": torch.tensor([[1]])},
            "torch.long",
        ),
        (
            {
                "input_ids": torch.empty((1, 0), dtype=torch.long),
                "attention_mask": torch.empty((1, 0), dtype=torch.long),
            },
            "empty or exceeds",
        ),
        (
            {
                "input_ids": torch.ones((1, 257), dtype=torch.long),
                "attention_mask": torch.ones((1, 257), dtype=torch.long),
            },
            "empty or exceeds",
        ),
        (
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[2]])},
            "zeros and ones",
        ),
        (
            {"input_ids": torch.tensor([[-1]]), "attention_mask": torch.tensor([[1]])},
            "out-of-vocabulary",
        ),
        (
            {"input_ids": torch.tensor([[8]]), "attention_mask": torch.tensor([[1]])},
            "out-of-vocabulary",
        ),
        (
            {"input_ids": "not-a-tensor", "attention_mask": torch.tensor([[1]])},
            "not a tensor",
        ),
    ],
)
def test_tokenize_rejects_malformed_tokenizer_outputs(
    tokenizer_output: dict[str, Any], message: str
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.next_output = tokenizer_output
    adapter, _, _ = build_adapter(tokenizer=tokenizer)

    with pytest.raises(BackboneForwardError, match=message):
        adapter.tokenize("hello")


def test_forward_returns_validated_tensors_under_inference_mode() -> None:
    adapter, _, model = build_adapter()

    output = adapter.forward_prompt("hello")

    assert output.logits.shape == (1, 3, 8)
    assert len(output.hidden_states) == 3
    assert output.attention_mask.shape == (1, 3)
    assert model.calls == [
        {
            "input_ids": ANY,
            "attention_mask": ANY,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
    ]
    assert model.inference_modes == [True]
    assert torch.is_inference_mode_enabled() is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_first_pass_retains_exact_nonaliased_prompt_token_provenance() -> None:
    tokenizer_input_ids = torch.tensor([[0, 2, 7]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[0, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    adapter, tokenizer, model = build_adapter(tokenizer=tokenizer)

    output = adapter.forward_prompt("hello")

    assert isinstance(output, BackboneFirstPassOutput)
    assert isinstance(output, BackboneForwardOutput)
    assert torch.equal(output.input_ids, tokenizer_input_ids)
    assert torch.equal(output.attention_mask, tokenizer_attention)
    assert len(tokenizer.calls) == 1
    model_input_ids = cast(torch.Tensor, model.calls[0]["input_ids"])
    model_attention = cast(torch.Tensor, model.calls[0]["attention_mask"])
    assert torch.equal(model_input_ids, tokenizer_input_ids)
    assert torch.equal(model_attention, tokenizer_attention)

    prompt_metadata = (
        tokenizer_input_ids,
        tokenizer_attention,
        model_input_ids,
        model_attention,
        output.input_ids,
        output.attention_mask,
    )
    assert len({tensor.untyped_storage().data_ptr() for tensor in prompt_metadata}) == len(
        prompt_metadata
    )
    for tensor in (model_input_ids, model_attention):
        assert tensor.device == torch.device("cpu")
        assert not tensor.is_inference()
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
        assert isinstance(tensor._version, int)
        assert tensor._version >= 0
    for tensor in (output.input_ids, output.attention_mask):
        assert tensor.device == torch.device("cpu")
        assert tensor.is_inference()
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
    assert output.input_ids.dtype == torch.int64
    assert output.attention_mask.dtype == torch.int64
    assert bool(torch.logical_or(output.attention_mask == 0, output.attention_mask == 1).all())
    assert bool(
        torch.logical_and(
            output.input_ids >= 0,
            output.input_ids < adapter.vocabulary_size,
        ).all()
    )

    with torch.inference_mode():
        output.input_ids.fill_(5)
        output.attention_mask.zero_()
    assert torch.equal(tokenizer_input_ids, torch.tensor([[0, 2, 7]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[0, 1, 1]]))
    assert torch.equal(model_input_ids, torch.tensor([[0, 2, 7]]))
    assert torch.equal(model_attention, torch.tensor([[0, 1, 1]]))


def test_legacy_backbone_forward_output_constructor_remains_valid() -> None:
    legacy = BackboneForwardOutput(
        logits=torch.ones(1, 2, 8),
        hidden_states=(torch.ones(1, 2, 4),),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert type(legacy) is BackboneForwardOutput
    assert not hasattr(legacy, "input_ids")
    assert legacy.logits.shape == (1, 2, 8)


@pytest.mark.parametrize("target", ["input_ids", "attention_mask"])
def test_forward_rejects_model_mutation_without_corrupting_tokenizer_provenance(
    target: str,
) -> None:
    tokenizer_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    model = MutatingPromptModel(target)
    adapter, _, built_model = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(BackboneForwardError, match=f"mutated first-pass {target}"):
        adapter.forward_prompt("hello")

    assert torch.equal(tokenizer_input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[1, 1, 1]]))
    model_value = cast(torch.Tensor, built_model.calls[0][target])
    if target == "input_ids":
        assert torch.equal(model_value, torch.tensor([[2, 3, 4]]))
    else:
        assert torch.equal(model_value, torch.zeros(1, 3, dtype=torch.int64))


@pytest.mark.parametrize("target", ["input_ids", "attention_mask"])
def test_forward_rejects_restored_model_input_values_by_version_counter(
    target: str,
) -> None:
    tokenizer_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    model = RestoreAfterComputeModel(target)
    adapter, _, _ = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(BackboneForwardError, match=f"mutated first-pass {target}"):
        adapter.forward_prompt("hello")

    assert model.version_before is not None
    assert model.version_before >= 0
    assert model.version_after is not None
    assert model.version_after > model.version_before
    assert model.mutated_snapshot is not None
    if target == "input_ids":
        assert torch.equal(model.mutated_snapshot, torch.tensor([[2, 3, 4]]))
    else:
        assert torch.equal(model.mutated_snapshot, torch.zeros(1, 3, dtype=torch.int64))

    model_value = cast(torch.Tensor, model.calls[0][target])
    tokenizer_value = tokenizer_input_ids if target == "input_ids" else tokenizer_attention
    assert torch.equal(model_value, tokenizer_value)
    assert model_value.untyped_storage().data_ptr() != tokenizer_value.untyped_storage().data_ptr()
    assert torch.equal(tokenizer_input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[1, 1, 1]]))


def test_model_exception_after_mutation_is_typed_and_preserves_original_cause() -> None:
    original = RuntimeError("model mutated its input and failed")
    tokenizer_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    model = MutatingPromptModel("input_ids", error_after_mutation=original)
    adapter, _, _ = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(BackboneForwardError, match="forward pass failed") as captured:
        adapter.forward_prompt("hello")

    assert captured.value.__cause__ is original
    assert torch.equal(tokenizer_input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[1, 1, 1]]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("training", "evaluation mode"), ("device", "not fully placed")],
)
def test_forward_revalidates_frozen_model_state_after_callback(
    mutation: str,
    message: str,
) -> None:
    class StateMutatingModel(FakeModel):
        def forward(self, **kwargs: object) -> SimpleNamespace:
            output = super().forward(**kwargs)
            if mutation == "training":
                self.train()
            else:
                self.to(torch.device("meta"))
            return output

    adapter, _, _ = build_adapter(model=StateMutatingModel())

    with pytest.raises(BackboneLoadError, match=message):
        adapter.forward_prompt("hello")


def test_first_pass_output_property_error_is_wrapped_with_original_cause() -> None:
    getter_error = RuntimeError("broken first-pass logits property")

    class RaisingFirstPassOutput:
        @property
        def logits(self) -> torch.Tensor:
            raise getter_error

    model = FakeModel()
    model.next_output = cast(Any, RaisingFirstPassOutput())
    adapter, _, _ = build_adapter(model=model)

    with pytest.raises(
        BackboneForwardError,
        match="Failed to read backbone first-pass output",
    ) as captured:
        adapter.forward_prompt("hello")

    assert captured.value.__cause__ is getter_error


@pytest.mark.parametrize(
    ("trigger", "action", "error_type", "message"),
    [
        ("logits", "input_ids", BackboneForwardError, "mutated first-pass input_ids"),
        (
            "hidden_states",
            "attention_mask",
            BackboneForwardError,
            "mutated first-pass attention_mask",
        ),
        ("logits", "training", BackboneLoadError, "evaluation mode"),
        ("hidden_states", "device", BackboneLoadError, "not fully placed"),
    ],
)
def test_forward_revalidates_boundaries_after_lazy_output_getters(
    trigger: str,
    action: str,
    error_type: type[Exception],
    message: str,
) -> None:
    tokenizer_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    model = GetterSideEffectModel(trigger=trigger, action=action)
    adapter, _, _ = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(error_type, match=message):
        adapter.forward_prompt("hello")

    assert torch.equal(tokenizer_input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[1, 1, 1]]))
    assert len(model.calls) == 1


def test_final_barrier_catches_tuple_iterator_mutation_during_output_validation() -> None:
    tokenizer_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokenizer_attention = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    tokenizer = FakeTokenizer()
    tokenizer.next_output = {
        "input_ids": tokenizer_input_ids,
        "attention_mask": tokenizer_attention,
    }
    model = IteratorSideEffectModel()
    adapter, _, _ = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(
        BackboneForwardError,
        match="mutated first-pass input_ids",
    ):
        adapter.forward_prompt("hello")

    assert torch.equal(tokenizer_input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(tokenizer_attention, torch.tensor([[1, 1, 1]]))
    assert len(model.calls) == 1


def test_tokenizer_callback_cannot_switch_model_mode_before_first_pass_call() -> None:
    model = FakeModel()

    class ModelMutatingTokenizer(FakeTokenizer):
        def __call__(self, text: str, **kwargs: object) -> dict[str, Any]:
            model.train()
            return super().__call__(text, **kwargs)

    tokenizer = ModelMutatingTokenizer()
    adapter, _, model = build_adapter(tokenizer=tokenizer, model=model)

    with pytest.raises(BackboneLoadError, match="evaluation mode"):
        adapter.forward_prompt("hello")

    assert len(tokenizer.calls) == 1
    assert model.calls == []


@pytest.mark.parametrize("target", ["tokenizer", "model"])
def test_loading_failures_preserve_original_cause(target: str) -> None:
    tokenizer_result: Any = FakeTokenizer()
    model_result: Any = FakeModel()
    original = OSError("offline")
    if target == "tokenizer":
        tokenizer_result = original
    else:
        model_result = original

    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=tokenizer_result if isinstance(tokenizer_result, Exception) else None,
            return_value=None if isinstance(tokenizer_result, Exception) else tokenizer_result,
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=model_result if isinstance(model_result, Exception) else None,
            return_value=None if isinstance(model_result, Exception) else model_result,
        ),
        pytest.raises(BackboneLoadError) as raised,
    ):
        BackboneAdapter(backbone_config(), cpu_info())

    assert raised.value.__cause__ is original


def test_unsupported_dtype_is_reported_as_load_error() -> None:
    config = backbone_config().model_copy(update={"dtype": "fp16"})

    with pytest.raises(BackboneLoadError, match="Failed to resolve dtype") as raised:
        BackboneAdapter(config, cpu_info())

    assert isinstance(raised.value.__cause__, ValueError)


def test_forward_rejects_nonfinite_logits() -> None:
    model = FakeModel()
    model.next_output = SimpleNamespace(
        logits=torch.full((1, 3, 8), torch.nan),
        hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
    )
    adapter, _, _ = build_adapter(model=model)

    with pytest.raises(BackboneForwardError, match="logits contain"):
        adapter.forward_prompt("hello")


def test_forward_rejects_wrong_hidden_state_count() -> None:
    model = FakeModel()
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 3, 8),
        hidden_states=(torch.ones(1, 3, 4),),
    )
    adapter, _, _ = build_adapter(model=model)

    with pytest.raises(BackboneForwardError, match="Expected 3 hidden states"):
        adapter.forward_prompt("hello")


@pytest.mark.parametrize(
    ("model_output", "message"),
    [
        (
            SimpleNamespace(
                logits=None,
                hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
            ),
            "does not contain logits",
        ),
        (
            SimpleNamespace(logits=torch.ones(1, 3, 8), hidden_states=None),
            "hidden-state tuple",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 7),
                hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
            ),
            "vocabulary dimension",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 8, dtype=torch.int64),
                hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
            ),
            "logits must use",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 8, device="meta"),
                hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
            ),
            "logits are not on",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 8, requires_grad=True),
                hidden_states=tuple(torch.ones(1, 3, 4) for _ in range(3)),
            ),
            "must not track gradients",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 8),
                hidden_states=(
                    torch.ones(1, 3, 4),
                    torch.ones(1, 2, 4),
                    torch.ones(1, 3, 4),
                ),
            ),
            "Hidden state must have shape",
        ),
        (
            SimpleNamespace(
                logits=torch.ones(1, 3, 8),
                hidden_states=(
                    torch.ones(1, 3, 4),
                    torch.full((1, 3, 4), torch.inf),
                    torch.ones(1, 3, 4),
                ),
            ),
            "hidden states contain",
        ),
    ],
)
def test_forward_rejects_invalid_tensor_outputs(
    model_output: SimpleNamespace, message: str
) -> None:
    model = FakeModel()
    model.next_output = model_output
    adapter, _, _ = build_adapter(model=model)

    with pytest.raises(BackboneForwardError, match=message):
        adapter.forward_prompt("hello")


def test_model_must_actually_move_to_the_requested_device() -> None:
    model = MagicMock()
    parameter = torch.nn.Parameter(torch.ones(1, device="meta"))
    parameter.requires_grad_(False)
    model.training = False
    model.parameters.side_effect = lambda: iter([parameter])
    model.buffers.side_effect = lambda: iter(())
    model.requires_grad_.return_value = model
    model.eval.return_value = model
    model.to.return_value = model
    model.config = SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=8,
        max_position_embeddings=32,
    )

    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
        pytest.raises(BackboneLoadError, match="not fully placed"),
    ):
        BackboneAdapter(backbone_config(), cpu_info())


def test_forward_is_compatible_with_tiny_real_qwen3_model() -> None:
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
        )
    )
    tokenizer = FakeTokenizer()
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        adapter = BackboneAdapter(backbone_config(), cpu_info())

    output = adapter.forward_prompt("hello")

    assert output.logits.shape == (1, 3, 8)
    assert len(output.hidden_states) == 2
    assert output.hidden_states[-1].shape == (1, 3, 8)
