"""Contract tests for the frozen backbone's public embedding-driven boundary."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache, DynamicCache

from iris.backbone import BackboneAdapter, BackboneForwardError, BackboneLoadError
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo

CPU = torch.device("cpu")


class FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"


class FakeCache(Cache):
    """Public Cache test double with an explicit logical sequence length."""

    def __init__(self, sequence_length: int = 0) -> None:
        super().__init__(layers=[])
        self.sequence_length = sequence_length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.sequence_length


class FakeModel(torch.nn.Module):
    """Small differentiable model with controllable embedding and causal outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            vocab_size=8,
            max_position_embeddings=32,
        )
        self.calls: list[dict[str, object]] = []
        self.next_embedding: object | None = None
        self.next_output: object | None = None

    def get_input_embeddings(self) -> object:
        return self.embedding if self.next_embedding is None else self.next_embedding

    def forward(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.next_output is not None:
            return self.next_output
        inputs = cast(torch.Tensor, kwargs["inputs_embeds"])
        logits = self.projection(inputs)
        if kwargs.get("logits_to_keep") == 1:
            logits = logits[:, -1:]
        labels = cast(torch.Tensor | None, kwargs.get("labels"))
        loss = None
        if labels is not None:
            supervised = labels != -100
            loss = logits[supervised].square().mean()
        cache = kwargs.get("past_key_values")
        if kwargs["use_cache"] and cache is None:
            cache = FakeCache()
        if isinstance(cache, FakeCache):
            cache.sequence_length += inputs.shape[1]
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=cache)


class GradientModeEmbeddingStub:
    """Embedding double that records the autograd mode used by the adapter."""

    def __init__(self) -> None:
        self.grad_enabled_during_calls: list[bool] = []

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_during_calls.append(torch.is_grad_enabled())
        return torch.ones(1, input_ids.shape[1], 4)


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
        device=CPU,
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )


def build_adapter(model: torch.nn.Module | None = None) -> tuple[BackboneAdapter, Any]:
    selected_model = model or FakeModel()
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=selected_model,
        ),
    ):
        adapter = BackboneAdapter(backbone_config(), cpu_info())
    return adapter, selected_model


def causal_inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    embeddings = torch.randn(1, 4, 4, requires_grad=requires_grad)
    attention_mask = torch.tensor([[False, True, True, True]])
    position_ids = torch.tensor([[0, 0, 1, 2]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    return embeddings, attention_mask, position_ids, labels


def test_embed_tokens_is_exact_no_grad_frozen_dtype_device_boundary() -> None:
    adapter, model = build_adapter()
    input_ids = torch.tensor([[0, 3, 7]])
    expected = model.embedding(input_ids).detach()

    embedded = adapter.embed_tokens(input_ids)

    torch.testing.assert_close(embedded, expected)
    assert embedded.shape == (1, 3, 4)
    assert embedded.dtype == torch.float32
    assert embedded.device == CPU
    assert not embedded.requires_grad
    assert embedded.grad_fn is None
    assert adapter.trainable_parameter_count == 0
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_exposes_validated_backbone_position_window() -> None:
    adapter, _ = build_adapter()

    assert adapter.max_position_embeddings == 32


@pytest.mark.parametrize("max_position_embeddings", [None, 0, False])
def test_rejects_invalid_model_position_window(max_position_embeddings: object) -> None:
    model = FakeModel()
    model.config.max_position_embeddings = max_position_embeddings

    with pytest.raises(BackboneLoadError, match="max_position_embeddings"):
        build_adapter(model)


def test_embed_tokens_calls_embedding_layer_with_autograd_disabled() -> None:
    adapter, model = build_adapter()
    embedding = GradientModeEmbeddingStub()
    model.next_embedding = embedding

    with torch.enable_grad():
        embedded = adapter.embed_tokens(torch.tensor([[1, 2]]))

    assert embedding.grad_enabled_during_calls == [False]
    assert not embedded.requires_grad
    assert embedded.grad_fn is None


def test_embed_tokens_rejects_embedding_callable_that_mutates_model_training_mode() -> None:
    adapter, model = build_adapter()

    class TrainingModeMutatingEmbedding:
        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            model.train()
            return torch.ones(1, input_ids.shape[1], 4)

    model.next_embedding = TrainingModeMutatingEmbedding()

    with pytest.raises(BackboneLoadError, match="evaluation mode"):
        adapter.embed_tokens(torch.tensor([[1, 2]]))


@pytest.mark.parametrize(
    "input_ids",
    [
        cast(Any, "not a tensor"),
        torch.tensor([1, 2]),
        torch.empty(1, 0, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.int64),
        torch.ones(1, 2),
        torch.tensor([[-1]]),
        torch.tensor([[8]]),
    ],
)
def test_embed_tokens_rejects_malformed_and_out_of_vocabulary_ids(input_ids: Any) -> None:
    adapter, _ = build_adapter()
    with pytest.raises(BackboneForwardError):
        adapter.embed_tokens(input_ids)


@pytest.mark.parametrize(
    "embedding_result",
    [
        cast(Any, "not a tensor"),
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 4, dtype=torch.float64),
        torch.full((1, 2, 4), float("nan")),
        torch.ones(1, 2, 4, requires_grad=True),
    ],
)
def test_embed_tokens_rejects_invalid_embedding_layer_outputs(embedding_result: Any) -> None:
    class EmbeddingStub:
        def __call__(self, input_ids: torch.Tensor) -> Any:
            return embedding_result

    adapter, model = build_adapter()
    model.next_embedding = EmbeddingStub()

    with pytest.raises(BackboneForwardError):
        adapter.embed_tokens(torch.tensor([[1, 2]]))


def test_forward_embeddings_passes_exact_teacher_forcing_arguments() -> None:
    adapter, model = build_adapter()
    embeddings, attention_mask, position_ids, labels = causal_inputs(requires_grad=True)

    output = adapter.forward_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        labels=labels,
    )

    assert model.calls == [
        {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "return_dict": True,
            "labels": labels,
        }
    ]
    assert output.loss is not None
    assert output.past_key_values is None


def test_forward_embeddings_passes_exact_cached_decode_arguments_and_returns_cache() -> None:
    adapter, model = build_adapter()
    embeddings = torch.randn(1, 1, 4)
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    position_ids = torch.tensor([[3]])
    incoming_cache = FakeCache(sequence_length=3)

    output = adapter.forward_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        past_key_values=incoming_cache,
        use_cache=True,
    )

    assert model.calls == [
        {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "return_dict": True,
            "past_key_values": incoming_cache,
        }
    ]
    assert output.loss is None
    assert output.past_key_values is incoming_cache
    assert incoming_cache.get_seq_length() == 4


def test_forward_embeddings_passes_exact_last_logit_prefill_arguments() -> None:
    adapter, model = build_adapter()
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    output = adapter.forward_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        use_cache=True,
        logits_to_keep=1,
    )

    assert model.calls == [
        {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    ]
    assert output.logits.shape == (1, 1, 8)
    assert isinstance(output.past_key_values, FakeCache)
    assert output.past_key_values.get_seq_length() == embeddings.shape[1]


def test_frozen_model_preserves_gradients_from_logits_and_loss_to_inputs() -> None:
    adapter, model = build_adapter()
    embeddings, attention_mask, position_ids, labels = causal_inputs(requires_grad=True)

    labelled = adapter.forward_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        labels=labels,
    )
    assert labelled.loss is not None
    torch.autograd.backward(labelled.loss)

    assert embeddings.grad is not None
    assert bool((embeddings.grad != 0).any())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert adapter.has_parameter_gradients is False

    logits_input = embeddings.detach().clone().requires_grad_()
    unlabelled = adapter.forward_embeddings(logits_input, attention_mask, position_ids)
    torch.autograd.backward(unlabelled.logits.square().sum())
    assert logits_input.grad is not None and bool((logits_input.grad != 0).any())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_forward_embeddings_rejects_model_moved_off_device_during_call() -> None:
    class DeviceMutatingModel(FakeModel):
        def forward(self, **kwargs: object) -> object:
            output = super().forward(**kwargs)
            self.to(torch.device("meta"))
            return output

    adapter, _ = build_adapter(DeviceMutatingModel())
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneLoadError, match="not fully placed"):
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda embeddings: cast(Any, "not a tensor"),
        lambda embeddings: embeddings[:, :, :3],
        lambda embeddings: embeddings.expand(2, -1, -1),
        lambda embeddings: embeddings[:, :0],
        lambda embeddings: embeddings.to(torch.float64),
        lambda embeddings: torch.full_like(embeddings, float("inf")),
    ],
)
def test_rejects_malformed_input_embeddings(mutator: Any) -> None:
    adapter, _ = build_adapter()
    embeddings, attention_mask, position_ids, _ = causal_inputs()
    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(mutator(embeddings), attention_mask, position_ids)


@pytest.mark.parametrize(
    ("attention_mask", "position_ids"),
    [
        (torch.ones(1, 4), torch.tensor([[0, 1, 2, 3]])),
        (torch.ones(1, 3, dtype=torch.bool), torch.tensor([[0, 1, 2, 3]])),
        (torch.tensor([[1, 1, 2, 1]]), torch.tensor([[0, 1, 2, 3]])),
        (torch.zeros(1, 4, dtype=torch.bool), torch.tensor([[0, 1, 2, 3]])),
        (torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 1, 2]])),
        (torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)),
        (torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, -1, 2, 3]])),
        (torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 1, 2, 32]])),
    ],
)
def test_rejects_malformed_attention_masks_and_position_ids(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    adapter, _ = build_adapter()
    embeddings, _, _, _ = causal_inputs()
    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([[-100, -100, 3]]),
        torch.tensor([[-100, -100, 3, 4]], dtype=torch.int32),
        torch.tensor([[-100, -100, -1, 4]]),
        torch.tensor([[-100, -100, 8, 4]]),
        torch.full((1, 4), -100, dtype=torch.int64),
        torch.tensor([[3, -100, -100, -100]]),
    ],
)
def test_rejects_malformed_labels(labels: torch.Tensor) -> None:
    adapter, _ = build_adapter()
    embeddings, attention_mask, position_ids, _ = causal_inputs()
    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            labels=labels,
        )


def test_rejects_incompatible_label_and_cache_options() -> None:
    adapter, _ = build_adapter()
    embeddings, attention_mask, position_ids, labels = causal_inputs()

    with pytest.raises(BackboneForwardError, match="labels cannot be combined"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            labels=labels,
            use_cache=True,
        )
    with pytest.raises(BackboneForwardError, match="require use_cache"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            past_key_values=DynamicCache(),
        )
    with pytest.raises(BackboneForwardError, match="explicit boolean"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            use_cache=cast(Any, 1),
        )


@pytest.mark.parametrize("logits_to_keep", [-1, 2, True, 0.0, "1", [], object()])
def test_rejects_invalid_logits_to_keep_values(logits_to_keep: object) -> None:
    adapter, _ = build_adapter()
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneForwardError, match="logits_to_keep"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            logits_to_keep=cast(Any, logits_to_keep),
        )


def test_rejects_last_logit_only_teacher_forcing() -> None:
    adapter, _ = build_adapter()
    embeddings, attention_mask, position_ids, labels = causal_inputs()

    with pytest.raises(BackboneForwardError, match="labels"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            labels=labels,
            logits_to_keep=1,
        )


def test_rejects_non_cache_objects_and_wrong_cached_attention_mask_length() -> None:
    adapter, _ = build_adapter()
    embeddings = torch.randn(1, 1, 4)
    cache = DynamicCache()

    with pytest.raises(BackboneForwardError, match="Transformers Cache"):
        adapter.forward_embeddings(
            embeddings,
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([[0]]),
            past_key_values=cast(Any, object()),
            use_cache=True,
        )
    with pytest.raises(BackboneForwardError, match="cached plus current sequence length"):
        adapter.forward_embeddings(
            embeddings,
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[0]]),
            past_key_values=cache,
            use_cache=True,
        )


@pytest.mark.parametrize(
    "detached_logits",
    [
        torch.ones(1, 4, 8),
        torch.ones(1, 4, 8, requires_grad=True),
    ],
)
def test_rejects_detached_unlabelled_logits_for_trainable_inputs(
    detached_logits: torch.Tensor,
) -> None:
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=detached_logits,
        loss=None,
        past_key_values=None,
    )
    embeddings, attention_mask, position_ids, _ = causal_inputs(requires_grad=True)

    with pytest.raises(BackboneForwardError, match="detached"):
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)


def test_rejects_unrelated_grad_logits_for_frozen_inputs() -> None:
    unrelated_logits = torch.ones(1, 4, 8, requires_grad=True).square()
    assert unrelated_logits.grad_fn is not None
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=unrelated_logits,
        loss=None,
        past_key_values=None,
    )
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneForwardError, match="unrelated"):
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)


@pytest.mark.parametrize(
    "output",
    [
        SimpleNamespace(logits=None, loss=None, past_key_values=None),
        SimpleNamespace(logits=torch.ones(1, 3, 8), loss=None, past_key_values=None),
        SimpleNamespace(
            logits=torch.ones(1, 4, 7),
            loss=None,
            past_key_values=None,
        ),
        SimpleNamespace(
            logits=torch.ones(1, 4, 8, dtype=torch.float64),
            loss=None,
            past_key_values=None,
        ),
        SimpleNamespace(
            logits=torch.full((1, 4, 8), float("nan")),
            loss=None,
            past_key_values=None,
        ),
        SimpleNamespace(
            logits=torch.ones(1, 4, 8),
            loss=torch.tensor(1.0),
            past_key_values=None,
        ),
        SimpleNamespace(
            logits=torch.ones(1, 4, 8),
            loss=None,
            past_key_values=DynamicCache(config=Qwen3Config()),
        ),
    ],
)
def test_rejects_invalid_unlabelled_causal_outputs(output: SimpleNamespace) -> None:
    adapter, model = build_adapter()
    model.next_output = output
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)


def test_wraps_causal_output_property_error_and_preserves_cause() -> None:
    getter_error = RuntimeError("broken logits property")

    class RaisingCausalOutput:
        @property
        def logits(self) -> torch.Tensor:
            raise getter_error

    adapter, model = build_adapter()
    model.next_output = RaisingCausalOutput()
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(
        BackboneForwardError,
        match="Failed to read backbone causal output",
    ) as error_info:
        adapter.forward_embeddings(embeddings, attention_mask, position_ids)

    assert error_info.value.__cause__ is getter_error


@pytest.mark.parametrize(
    "loss",
    [
        None,
        torch.ones(1),
        torch.tensor(1),
        torch.tensor(float("inf")),
        torch.tensor(1.0),
        torch.tensor(1.0, requires_grad=True),
    ],
)
def test_rejects_invalid_labelled_loss_outputs(loss: object) -> None:
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 4, 8, requires_grad=True),
        loss=loss,
        past_key_values=None,
    )
    embeddings, attention_mask, position_ids, labels = causal_inputs(requires_grad=True)

    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            labels=labels,
        )


def test_rejects_unrelated_grad_loss_for_frozen_inputs() -> None:
    unrelated_loss = torch.ones((), requires_grad=True).square()
    assert unrelated_loss.grad_fn is not None
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 4, 8),
        loss=unrelated_loss,
        past_key_values=None,
    )
    embeddings, attention_mask, position_ids, labels = causal_inputs()

    with pytest.raises(BackboneForwardError, match="unrelated"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            labels=labels,
        )


def test_rejects_missing_cache_output_when_cache_is_requested() -> None:
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 4, 8),
        loss=None,
        past_key_values=None,
    )
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneForwardError, match="must return past_key_values"):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            use_cache=True,
        )


@pytest.mark.parametrize(
    "returned_cache",
    [
        object(),
        FakeCache(sequence_length=3),
        FakeCache(sequence_length=5),
        FakeCache(sequence_length=cast(Any, True)),
    ],
)
def test_rejects_invalid_or_inexact_output_cache(returned_cache: object) -> None:
    adapter, model = build_adapter()
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 4, 8),
        loss=None,
        past_key_values=returned_cache,
    )
    embeddings, attention_mask, position_ids, _ = causal_inputs()

    with pytest.raises(BackboneForwardError):
        adapter.forward_embeddings(
            embeddings,
            attention_mask,
            position_ids,
            use_cache=True,
        )


def test_rejects_returned_input_cache_that_was_not_extended_in_place() -> None:
    adapter, model = build_adapter()
    incoming_cache = FakeCache(sequence_length=2)
    model.next_output = SimpleNamespace(
        logits=torch.ones(1, 1, 8),
        loss=None,
        past_key_values=incoming_cache,
    )

    with pytest.raises(BackboneForwardError, match="invalid sequence length"):
        adapter.forward_embeddings(
            torch.randn(1, 1, 4),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[2]]),
            past_key_values=incoming_cache,
            use_cache=True,
        )


def test_wraps_cache_sequence_length_error_and_preserves_cause() -> None:
    cache_error = RuntimeError("broken cache length")

    class RaisingCache(FakeCache):
        def get_seq_length(self, layer_idx: int = 0) -> int:
            raise cache_error

    adapter, _ = build_adapter()

    with pytest.raises(
        BackboneForwardError,
        match="Failed to read past_key_values sequence length",
    ) as error_info:
        adapter.forward_embeddings(
            torch.randn(1, 1, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[1]]),
            past_key_values=RaisingCache(sequence_length=1),
            use_cache=True,
        )

    assert error_info.value.__cause__ is cache_error


def tiny_qwen_adapter() -> tuple[BackboneAdapter, Qwen3ForCausalLM]:
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
        )
    )
    adapter, selected = build_adapter(model)
    return adapter, cast(Qwen3ForCausalLM, selected)


def test_tiny_qwen_teacher_forcing_reaches_inputs_but_not_backbone_parameters() -> None:
    adapter, model = tiny_qwen_adapter()
    token_embeddings = adapter.embed_tokens(torch.tensor([[2, 3, 4]])).detach()
    workspace_embeddings = torch.randn(1, 2, 8, requires_grad=True)
    target_embedding = adapter.embed_tokens(torch.tensor([[5]]))
    inputs_embeds = torch.cat(
        (token_embeddings, workspace_embeddings, target_embedding),
        dim=1,
    )
    labels = torch.tensor([[-100, -100, -100, -100, -100, 5]])

    output = adapter.forward_embeddings(
        inputs_embeds,
        torch.ones(1, 6, dtype=torch.bool),
        torch.arange(6).unsqueeze(0),
        labels=labels,
    )
    assert output.loss is not None
    assert output.loss.requires_grad
    assert output.loss.grad_fn is not None
    expected_loss = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, expected_loss)
    torch.autograd.backward(output.loss)

    assert workspace_embeddings.grad is not None
    assert bool((workspace_embeddings.grad != 0).any())
    assert bool((workspace_embeddings.grad[:, -1] != 0).any())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert adapter.has_parameter_gradients is False


def test_tiny_qwen_cached_decode_matches_full_recomputation() -> None:
    adapter, _ = tiny_qwen_adapter()
    prefix = adapter.embed_tokens(torch.tensor([[2, 3, 4]]))
    next_embedding = adapter.embed_tokens(torch.tensor([[5]]))
    prefix_mask = torch.tensor([[False, True, True]])
    prefix_positions = torch.tensor([[0, 0, 1]])

    cached_prefix = adapter.forward_embeddings(
        prefix,
        prefix_mask,
        prefix_positions,
        use_cache=True,
    )
    assert cached_prefix.past_key_values is not None
    assert isinstance(cached_prefix.past_key_values, Cache)
    assert cached_prefix.past_key_values.get_seq_length() == prefix.shape[1]
    cached_step = adapter.forward_embeddings(
        next_embedding,
        torch.tensor([[False, True, True, True]]),
        torch.tensor([[2]]),
        past_key_values=cached_prefix.past_key_values,
        use_cache=True,
    )
    assert cached_step.past_key_values is cached_prefix.past_key_values
    assert cached_step.past_key_values.get_seq_length() == prefix.shape[1] + 1
    recomputed = adapter.forward_embeddings(
        torch.cat((prefix, next_embedding), dim=1),
        torch.tensor([[False, True, True, True]]),
        torch.tensor([[0, 0, 1, 2]]),
    )

    torch.testing.assert_close(
        cached_step.logits[:, -1],
        recomputed.logits[:, -1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_tiny_qwen_last_logit_prefill_matches_full_logits() -> None:
    adapter, _ = tiny_qwen_adapter()
    embeddings = adapter.embed_tokens(torch.tensor([[2, 3, 4, 5]]))
    attention_mask = torch.tensor([[False, True, True, True]])
    position_ids = torch.tensor([[0, 0, 1, 2]])

    last_only = adapter.forward_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        logits_to_keep=1,
    )
    full = adapter.forward_embeddings(embeddings, attention_mask, position_ids)

    assert last_only.logits.shape == (1, 1, 17)
    torch.testing.assert_close(last_only.logits, full.logits[:, -1:], rtol=1e-5, atol=1e-6)
