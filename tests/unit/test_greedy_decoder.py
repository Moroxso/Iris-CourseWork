"""Mutation-resistant contract tests for deterministic cached Iris decoding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache

from iris.backbone import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import (
    GreedyIrisDecoder,
    IrisGenerationError,
    IrisGenerationOutput,
    SecondPassGenerationPrefixOutput,
)
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo
from iris.workspace import WorkspaceState

CPU = torch.device("cpu")
HIDDEN_SIZE = 3
VOCABULARY_SIZE = 8
PREFIX_LENGTH = 6
MAX_CONTEXT_TOKENS = 16


class FakeCache(Cache):
    """Transformers Cache double whose physical sequence length is explicit."""

    def __init__(self, sequence_length: int = 0) -> None:
        super().__init__(layers=[])
        self.sequence_length = sequence_length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.sequence_length


@dataclass(frozen=True, slots=True)
class ForwardCall:
    """One immutable observation of a scripted backbone call boundary."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor | None
    incoming_cache: Cache | None
    incoming_cache_length: int | None
    use_cache: bool
    logits_to_keep: int
    grad_enabled: bool
    inference_enabled: bool


OutputMutator = Callable[[int, BackboneCausalOutput], object]
EmbeddingMutator = Callable[[torch.Tensor], object]


class ScriptedBackbone(BackboneAdapter):
    """A public-boundary test double with deterministic logits and a mutable cache."""

    def __init__(
        self,
        planned_scores: list[torch.Tensor],
        *,
        hidden_size: int = HIDDEN_SIZE,
        vocabulary_size: int = VOCABULARY_SIZE,
        max_position_embeddings: int = MAX_CONTEXT_TOKENS,
        output_mutator: OutputMutator | None = None,
        replace_cache_on_calls: frozenset[int] = frozenset(),
        forward_error_on_call: int | None = None,
        embedding_error: BaseException | None = None,
        embedding_mutator: EmbeddingMutator | None = None,
        embedding_dtype: torch.dtype = torch.float32,
        embedding_device: torch.device = CPU,
    ) -> None:
        # Deliberately bypass model loading while retaining nominal BackboneAdapter type.
        self._script_hidden_size = hidden_size
        self._script_vocabulary_size = vocabulary_size
        self._script_max_positions = max_position_embeddings
        self.planned_scores = planned_scores
        self.output_mutator = output_mutator
        self.replace_cache_on_calls = replace_cache_on_calls
        self.forward_error_on_call = forward_error_on_call
        self.embedding_error = embedding_error
        self.embedding_mutator = embedding_mutator
        self.embedding_dtype = embedding_dtype
        self.embedding_device = embedding_device
        self.forward_calls: list[ForwardCall] = []
        self.embed_calls: list[torch.Tensor] = []
        self.embed_grad_modes: list[bool] = []
        self.embed_inference_modes: list[bool] = []
        self.returned_caches: list[Cache] = []
        self.emitted_logits: list[torch.Tensor] = []

    @property
    def hidden_size(self) -> int:
        return self._script_hidden_size

    @property
    def vocabulary_size(self) -> int:
        return self._script_vocabulary_size

    @property
    def max_position_embeddings(self) -> int:
        return self._script_max_positions

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.embed_calls.append(input_ids.detach().clone())
        self.embed_grad_modes.append(torch.is_grad_enabled())
        self.embed_inference_modes.append(torch.is_inference_mode_enabled())
        if self.embedding_error is not None:
            raise self.embedding_error
        token_values = input_ids.to(
            dtype=self.embedding_dtype,
            device=self.embedding_device,
        ).unsqueeze(-1)
        embeddings = token_values.expand(-1, -1, self.hidden_size).clone()
        if self.embedding_mutator is None:
            return embeddings
        return cast(torch.Tensor, self.embedding_mutator(embeddings))

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
        call_index = len(self.forward_calls)
        incoming_length = None if past_key_values is None else past_key_values.get_seq_length()
        self.forward_calls.append(
            ForwardCall(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask.detach().clone(),
                position_ids=position_ids.detach().clone(),
                labels=labels,
                incoming_cache=past_key_values,
                incoming_cache_length=incoming_length,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                grad_enabled=torch.is_grad_enabled(),
                inference_enabled=torch.is_inference_mode_enabled(),
            )
        )
        if self.forward_error_on_call == call_index:
            raise RuntimeError("scripted backbone failure")
        if call_index >= len(self.planned_scores):
            raise AssertionError("decoder made more model calls than the token script")

        if past_key_values is None:
            cache: Cache = FakeCache(inputs_embeds.shape[1])
        elif call_index in self.replace_cache_on_calls:
            assert incoming_length is not None
            cache = FakeCache(incoming_length + inputs_embeds.shape[1])
        else:
            assert isinstance(past_key_values, FakeCache)
            past_key_values.sequence_length += inputs_embeds.shape[1]
            cache = past_key_values
        self.returned_caches.append(cache)

        logits = self.planned_scores[call_index].to(
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        logits = logits.reshape(1, 1, self.vocabulary_size)
        self.emitted_logits.append(logits)
        output = BackboneCausalOutput(
            logits=logits,
            loss=None,
            past_key_values=cache,
        )
        if self.output_mutator is None:
            return output
        return cast(BackboneCausalOutput, self.output_mutator(call_index, output))


def token_scores(*token_ids: int, vocabulary_size: int = VOCABULARY_SIZE) -> list[torch.Tensor]:
    """Create score rows whose unique maxima select the requested tokens."""
    result: list[torch.Tensor] = []
    for token_id in token_ids:
        scores = torch.arange(vocabulary_size, dtype=torch.float32) * -0.01
        scores[token_id] = 10.0
        result.append(scores)
    return result


def make_prefix(
    *,
    hidden_size: int = HIDDEN_SIZE,
    prefix_length: int = PREFIX_LENGTH,
    dtype: torch.dtype = torch.float32,
) -> SecondPassGenerationPrefixOutput:
    embeddings = torch.arange(
        prefix_length * hidden_size,
        dtype=dtype,
    ).reshape(1, prefix_length, hidden_size)
    if prefix_length == PREFIX_LENGTH:
        attention = torch.tensor(
            [[False, False, True, True, True, True]],
            dtype=torch.bool,
        )
    else:
        attention = torch.ones(1, prefix_length, dtype=torch.bool)
    raw_positions = attention.to(dtype=torch.int64).cumsum(dim=-1) - 1
    positions = torch.where(attention, raw_positions, torch.zeros_like(raw_positions))
    return SecondPassGenerationPrefixOutput(
        inputs_embeds=embeddings,
        attention_mask=attention,
        position_ids=positions,
    )


def make_trace() -> tuple[WorkspaceState, ...]:
    first = WorkspaceState.empty(
        batch_size=1,
        num_slots=2,
        state_dim=4,
        dtype=torch.float32,
        device=CPU,
    )
    second = replace(
        first,
        slots=torch.ones_like(first.slots),
        age=torch.ones_like(first.age),
        occupancy=torch.ones_like(first.occupancy),
        source_module=torch.tensor([[0, 1]], dtype=torch.int64),
        confidence=torch.full_like(first.confidence, 0.75),
        ttl=torch.full_like(first.ttl, 2.0),
        write_count=torch.ones_like(first.write_count),
    )
    return (first, second)


def decode(
    backbone: ScriptedBackbone,
    *,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...] = (),
    prefix: SecondPassGenerationPrefixOutput | None = None,
    workspace_trace: tuple[WorkspaceState, ...] = (),
) -> IrisGenerationOutput:
    decoder = GreedyIrisDecoder(
        backbone,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
    )
    return decoder.generate(
        make_prefix() if prefix is None else prefix,
        workspace_trace=workspace_trace,
    )


def test_exact_prefill_and_incremental_calls_for_three_generated_tokens() -> None:
    backbone = ScriptedBackbone(token_scores(4, 3, 7))
    prefix = make_prefix()

    output = decode(backbone, max_new_tokens=3, prefix=prefix)

    assert output.token_ids.tolist() == [[4, 3, 7]]
    assert output.token_ids.dtype == torch.int64
    assert output.token_ids.device == prefix.inputs_embeds.device
    assert output.token_ids.requires_grad is False
    assert output.stop_reason == "max_new_tokens"
    assert output.matched_stop_token_id is None
    assert len(backbone.forward_calls) == 3
    assert len(backbone.embed_calls) == 2
    assert [ids.tolist() for ids in backbone.embed_calls] == [[[4]], [[3]]]

    prefill, first_step, second_step = backbone.forward_calls
    torch.testing.assert_close(prefill.inputs_embeds, prefix.inputs_embeds)
    assert prefill.inputs_embeds.untyped_storage().data_ptr() != (
        prefix.inputs_embeds.untyped_storage().data_ptr()
    )
    assert torch.equal(prefill.attention_mask, prefix.attention_mask)
    assert torch.equal(prefill.position_ids, prefix.position_ids)
    assert prefill.incoming_cache is None
    assert prefill.incoming_cache_length is None
    assert first_step.inputs_embeds.tolist() == [[[4.0, 4.0, 4.0]]]
    assert second_step.inputs_embeds.tolist() == [[[3.0, 3.0, 3.0]]]
    assert first_step.attention_mask.tolist() == [[False, False, True, True, True, True, True]]
    assert second_step.attention_mask.tolist() == [
        [False, False, True, True, True, True, True, True]
    ]
    assert first_step.position_ids.tolist() == [[4]]
    assert second_step.position_ids.tolist() == [[5]]
    assert first_step.incoming_cache is backbone.returned_caches[0]
    assert second_step.incoming_cache is backbone.returned_caches[0]
    assert first_step.incoming_cache_length == PREFIX_LENGTH
    assert second_step.incoming_cache_length == PREFIX_LENGTH + 1
    assert all(call.labels is None for call in backbone.forward_calls)
    assert all(call.use_cache is True for call in backbone.forward_calls)
    assert all(call.logits_to_keep == 1 for call in backbone.forward_calls)


def test_one_generated_token_uses_one_prefill_and_no_embedding_call() -> None:
    backbone = ScriptedBackbone(token_scores(6))

    output = decode(backbone, max_new_tokens=1)

    assert output.token_ids.tolist() == [[6]]
    assert output.stop_reason == "max_new_tokens"
    assert len(backbone.forward_calls) == 1
    assert backbone.embed_calls == []


def test_float64_prefix_preserves_backbone_dtype_across_cached_steps() -> None:
    prefix = make_prefix(dtype=torch.float64)
    backbone = ScriptedBackbone(
        token_scores(1, 2),
        embedding_dtype=torch.float64,
    )

    output = decode(backbone, max_new_tokens=2, prefix=prefix)

    assert output.token_ids.tolist() == [[1, 2]]
    assert output.token_ids.device == prefix.inputs_embeds.device
    assert all(call.inputs_embeds.dtype == torch.float64 for call in backbone.forward_calls)
    assert all(logits.dtype == torch.float64 for logits in backbone.emitted_logits)


def test_first_prefill_token_can_stop_and_is_included() -> None:
    backbone = ScriptedBackbone(token_scores(2, 7, 6))

    output = decode(backbone, max_new_tokens=3, stop_token_ids=(2,))

    assert output.token_ids.tolist() == [[2]]
    assert output.stop_reason == "stop_token"
    assert output.matched_stop_token_id == 2
    assert len(backbone.forward_calls) == 1
    assert backbone.embed_calls == []


def test_stop_token_wins_when_it_is_also_the_max_length_token() -> None:
    backbone = ScriptedBackbone(token_scores(1, 5))

    output = decode(backbone, max_new_tokens=2, stop_token_ids=(5,))

    assert output.token_ids.tolist() == [[1, 5]]
    assert output.stop_reason == "stop_token"
    assert output.matched_stop_token_id == 5
    assert len(backbone.forward_calls) == 2
    assert len(backbone.embed_calls) == 1


def test_exact_argmax_tie_prefers_the_lower_token_id_without_rng() -> None:
    scores = torch.full((VOCABULARY_SIZE,), -4.0)
    scores[2] = 9.0
    scores[6] = 9.0
    backbone = ScriptedBackbone([scores])
    rng_before = torch.random.get_rng_state().clone()

    output = decode(backbone, max_new_tokens=1)

    assert output.token_ids.tolist() == [[2]]
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_generation_runs_every_backbone_boundary_in_inference_mode() -> None:
    backbone = ScriptedBackbone(token_scores(1, 2, 3))

    with torch.enable_grad():
        output = decode(backbone, max_new_tokens=3)

    assert output.token_ids.is_inference()
    assert all(call.grad_enabled is False for call in backbone.forward_calls)
    assert all(call.inference_enabled is True for call in backbone.forward_calls)
    assert backbone.embed_grad_modes == [False, False]
    assert backbone.embed_inference_modes == [True, True]


def test_workspace_trace_is_passed_through_by_identity_without_mutation() -> None:
    trace = make_trace()
    before = tuple(
        tuple(getattr(state, field).clone() for field in state.__dataclass_fields__)
        for state in trace
    )
    backbone = ScriptedBackbone(token_scores(4))

    output = decode(backbone, max_new_tokens=1, workspace_trace=trace)

    assert output.workspace_trace is trace
    assert output.workspace_trace[0] is trace[0]
    for state, snapshots in zip(trace, before, strict=True):
        for field, snapshot in zip(state.__dataclass_fields__, snapshots, strict=True):
            torch.testing.assert_close(getattr(state, field), snapshot)
    assert not hasattr(output, "__dict__")
    with pytest.raises(FrozenInstanceError):
        output.stop_reason = "stop_token"  # type: ignore[misc]


def test_empty_workspace_trace_is_valid() -> None:
    output = decode(ScriptedBackbone(token_scores(4)), max_new_tokens=1)

    assert output.workspace_trace == ()


@pytest.mark.parametrize(
    "workspace_trace",
    [
        cast(Any, []),
        cast(Any, [make_trace()[0]]),
        cast(Any, (object(),)),
        cast(Any, (make_trace()[0], object())),
    ],
)
def test_rejects_malformed_workspace_trace_before_any_model_call(
    workspace_trace: object,
) -> None:
    backbone = ScriptedBackbone(token_scores(4))
    decoder = GreedyIrisDecoder(backbone, max_new_tokens=1, stop_token_ids=())

    with pytest.raises(IrisGenerationError, match="workspace_trace"):
        decoder.generate(make_prefix(), workspace_trace=cast(Any, workspace_trace))

    assert backbone.forward_calls == []


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None])
def test_constructor_requires_exact_positive_max_new_tokens(value: object) -> None:
    backbone = ScriptedBackbone(token_scores(1))

    with pytest.raises(ValueError, match="max_new_tokens"):
        GreedyIrisDecoder(
            backbone,
            max_new_tokens=cast(Any, value),
            stop_token_ids=(),
        )


@pytest.mark.parametrize(
    "stop_token_ids",
    [
        cast(Any, []),
        cast(Any, {1}),
        cast(Any, (True,)),
        cast(Any, (1.0,)),
        cast(Any, ("1",)),
        (-1,),
        (VOCABULARY_SIZE,),
        (1, 1),
    ],
)
def test_constructor_requires_unique_in_vocabulary_integer_stop_tuple(
    stop_token_ids: object,
) -> None:
    backbone = ScriptedBackbone(token_scores(1))

    with pytest.raises(ValueError, match=r"[Ss]top_token_ids|[Ss]top token"):
        GreedyIrisDecoder(
            backbone,
            max_new_tokens=1,
            stop_token_ids=cast(Any, stop_token_ids),
        )


def test_constructor_accepts_empty_and_unsorted_stop_tuple() -> None:
    empty = GreedyIrisDecoder(
        ScriptedBackbone(token_scores(1)),
        max_new_tokens=1,
        stop_token_ids=(),
    )
    unsorted = GreedyIrisDecoder(
        ScriptedBackbone(token_scores(1)),
        max_new_tokens=1,
        stop_token_ids=(7, 0, 3),
    )

    assert empty.stop_token_ids == ()
    assert unsorted.stop_token_ids == (7, 0, 3)


PrefixMutator = Callable[[SecondPassGenerationPrefixOutput], object]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda prefix: cast(Any, "not a prefix DTO"),
        lambda prefix: replace(prefix, inputs_embeds=prefix.inputs_embeds[0]),
        lambda prefix: replace(
            prefix,
            inputs_embeds=torch.ones(2, PREFIX_LENGTH, HIDDEN_SIZE),
        ),
        lambda prefix: replace(prefix, inputs_embeds=torch.ones(1, 0, HIDDEN_SIZE)),
        lambda prefix: replace(
            prefix,
            inputs_embeds=torch.ones(1, PREFIX_LENGTH, HIDDEN_SIZE + 1),
        ),
        lambda prefix: replace(
            prefix,
            inputs_embeds=prefix.inputs_embeds.to(dtype=torch.int64),
        ),
        lambda prefix: replace(
            prefix,
            inputs_embeds=torch.full_like(prefix.inputs_embeds, float("nan")),
        ),
        lambda prefix: replace(
            prefix,
            inputs_embeds=torch.full_like(prefix.inputs_embeds, float("inf")),
        ),
        lambda prefix: replace(
            prefix,
            inputs_embeds=prefix.inputs_embeds.detach().requires_grad_(),
        ),
        lambda prefix: replace(
            prefix,
            inputs_embeds=torch.ones(
                1,
                PREFIX_LENGTH,
                HIDDEN_SIZE,
                device="meta",
            ),
        ),
        lambda prefix: replace(
            prefix,
            attention_mask=prefix.attention_mask[:, :-1],
        ),
        lambda prefix: replace(
            prefix,
            attention_mask=prefix.attention_mask.to(dtype=torch.int64),
        ),
        lambda prefix: replace(
            prefix,
            attention_mask=torch.zeros_like(prefix.attention_mask),
        ),
        lambda prefix: replace(
            prefix,
            attention_mask=torch.tensor(
                [[False, False, True, True, True, False]],
                dtype=torch.bool,
            ),
        ),
        lambda prefix: replace(prefix, position_ids=prefix.position_ids[:, :-1]),
        lambda prefix: replace(
            prefix,
            position_ids=prefix.position_ids.to(dtype=torch.int32),
        ),
        lambda prefix: replace(prefix, position_ids=prefix.position_ids + 1),
        lambda prefix: replace(
            prefix,
            position_ids=torch.tensor([[0, 0, 0, 2, 3, 4]], dtype=torch.int64),
        ),
    ],
)
def test_rejects_malformed_prefix_before_any_model_call(mutator: PrefixMutator) -> None:
    backbone = ScriptedBackbone(token_scores(1))
    decoder = GreedyIrisDecoder(backbone, max_new_tokens=1, stop_token_ids=())

    with pytest.raises(IrisGenerationError):
        decoder.generate(cast(Any, mutator(make_prefix())), workspace_trace=())

    assert backbone.forward_calls == []


def test_rejects_physical_context_overflow_before_prefill() -> None:
    prefix = make_prefix(prefix_length=8)
    backbone = ScriptedBackbone(
        token_scores(1),
        max_position_embeddings=10,
    )
    decoder = GreedyIrisDecoder(backbone, max_new_tokens=3, stop_token_ids=())

    with pytest.raises(IrisGenerationError, match="context"):
        decoder.generate(prefix, workspace_trace=())

    assert backbone.forward_calls == []


def test_exact_physical_and_logical_context_boundary_is_accepted() -> None:
    prefix = make_prefix(prefix_length=7)
    backbone = ScriptedBackbone(
        token_scores(1, 2, 3),
        max_position_embeddings=10,
    )

    output = decode(backbone, max_new_tokens=3, prefix=prefix)

    assert output.token_ids.tolist() == [[1, 2, 3]]
    assert backbone.forward_calls[-1].position_ids.tolist() == [[8]]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda index, output: cast(Any, object()),
        lambda index, output: replace(output, logits=output.logits[:, 0]),
        lambda index, output: replace(output, logits=output.logits.to(torch.float64)),
        lambda index, output: replace(
            output,
            logits=torch.full_like(output.logits, float("nan")),
        ),
        lambda index, output: replace(output, loss=torch.tensor(0.0)),
        lambda index, output: replace(output, past_key_values=cast(Any, object())),
        lambda index, output: replace(output, past_key_values=None),
        lambda index, output: replace(output, past_key_values=FakeCache(0)),
    ],
)
def test_rejects_malformed_prefill_output(mutator: OutputMutator) -> None:
    backbone = ScriptedBackbone(token_scores(1), output_mutator=mutator)

    with pytest.raises(IrisGenerationError):
        decode(backbone, max_new_tokens=1)


@pytest.mark.parametrize(
    "incremental_mutator",
    [
        lambda output: cast(Any, object()),
        lambda output: replace(output, logits=output.logits[:, 0]),
        lambda output: replace(output, logits=output.logits.to(torch.float64)),
        lambda output: replace(
            output,
            logits=torch.full_like(output.logits, float("inf")),
        ),
        lambda output: replace(output, loss=torch.tensor(0.0)),
        lambda output: replace(output, past_key_values=None),
        lambda output: replace(output, past_key_values=FakeCache(0)),
    ],
)
def test_rejects_malformed_incremental_output(
    incremental_mutator: Callable[[BackboneCausalOutput], object],
) -> None:
    def mutate_only_incremental(
        call_index: int,
        output: BackboneCausalOutput,
    ) -> object:
        if call_index == 1:
            return incremental_mutator(output)
        return output

    backbone = ScriptedBackbone(
        token_scores(1, 2),
        output_mutator=mutate_only_incremental,
    )

    with pytest.raises(IrisGenerationError):
        decode(backbone, max_new_tokens=2)


@pytest.mark.parametrize(
    "embedding_mutator",
    [
        lambda embeddings: cast(Any, object()),
        lambda embeddings: embeddings[:, 0],
        lambda embeddings: embeddings.to(dtype=torch.float64),
        lambda embeddings: torch.full_like(embeddings, float("nan")),
        lambda embeddings: torch.ones(1, 1, HIDDEN_SIZE + 1),
    ],
)
def test_rejects_malformed_generated_token_embeddings(
    embedding_mutator: EmbeddingMutator,
) -> None:
    backbone = ScriptedBackbone(
        token_scores(1, 2),
        embedding_mutator=embedding_mutator,
    )

    with pytest.raises(IrisGenerationError, match="embeddings"):
        decode(backbone, max_new_tokens=2)


def test_wraps_prefill_and_generated_embedding_failures_with_causes() -> None:
    prefill_failure = ScriptedBackbone(
        token_scores(1),
        forward_error_on_call=0,
    )
    with pytest.raises(IrisGenerationError, match="prefill") as prefill_error:
        decode(prefill_failure, max_new_tokens=1)
    assert isinstance(prefill_error.value.__cause__, RuntimeError)

    embedding_failure = ScriptedBackbone(
        token_scores(1, 2),
        embedding_error=RuntimeError("embedding failed"),
    )
    with pytest.raises(IrisGenerationError, match="embed") as embedding_error:
        decode(embedding_failure, max_new_tokens=2)
    assert isinstance(embedding_error.value.__cause__, RuntimeError)


def test_wraps_incremental_forward_failure_after_one_selected_token() -> None:
    backbone = ScriptedBackbone(
        token_scores(1, 2),
        forward_error_on_call=1,
    )

    with pytest.raises(IrisGenerationError, match="cached generation step") as captured:
        decode(backbone, max_new_tokens=2)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert len(backbone.forward_calls) == 2
    assert len(backbone.embed_calls) == 1


def test_new_cache_object_transfers_sole_ownership_to_the_following_step() -> None:
    backbone = ScriptedBackbone(
        token_scores(1, 2, 3),
        replace_cache_on_calls=frozenset({1}),
    )

    output = decode(backbone, max_new_tokens=3)

    assert output.token_ids.tolist() == [[1, 2, 3]]
    assert backbone.returned_caches[1] is not backbone.returned_caches[0]
    assert backbone.forward_calls[1].incoming_cache is backbone.returned_caches[0]
    assert backbone.forward_calls[2].incoming_cache is backbone.returned_caches[1]
    assert backbone.forward_calls[2].incoming_cache is not backbone.returned_caches[0]


def test_prefix_and_trace_survive_backbone_callback_without_aliasing() -> None:
    prefix = make_prefix()
    prefix_embeddings_before = prefix.inputs_embeds.clone()
    prefix_attention_before = prefix.attention_mask.clone()
    prefix_positions_before = prefix.position_ids.clone()
    trace = make_trace()
    backbone = ScriptedBackbone(token_scores(1, 2))

    output = decode(
        backbone,
        max_new_tokens=2,
        prefix=prefix,
        workspace_trace=trace,
    )

    torch.testing.assert_close(prefix.inputs_embeds, prefix_embeddings_before)
    assert torch.equal(prefix.attention_mask, prefix_attention_before)
    assert torch.equal(prefix.position_ids, prefix_positions_before)
    assert output.workspace_trace is trace
    model_storage = {logits.untyped_storage().data_ptr() for logits in backbone.emitted_logits}
    assert output.token_ids.untyped_storage().data_ptr() not in model_storage


class MutatingInputBackbone(ScriptedBackbone):
    """Adversarial callback that overwrites every tensor it receives."""

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().embed_tokens(input_ids)
        input_ids.fill_(7)
        return embeddings

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
        output = super().forward_embeddings(
            inputs_embeds,
            attention_mask,
            position_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        )
        inputs_embeds.zero_()
        attention_mask.zero_()
        position_ids.fill_(99)
        return output


def test_adversarial_backbone_tensor_mutation_cannot_reach_prefix_storage() -> None:
    prefix = make_prefix()
    embeddings_before = prefix.inputs_embeds.clone()
    attention_before = prefix.attention_mask.clone()
    positions_before = prefix.position_ids.clone()
    backbone = MutatingInputBackbone(token_scores(3, 4, 5))

    output = decode(backbone, max_new_tokens=3, prefix=prefix)

    assert output.token_ids.tolist() == [[3, 4, 5]]
    assert [ids.tolist() for ids in backbone.embed_calls] == [[[3]], [[4]]]
    assert backbone.forward_calls[1].attention_mask.tolist() == [
        [False, False, True, True, True, True, True]
    ]
    assert backbone.forward_calls[2].attention_mask.tolist() == [
        [False, False, True, True, True, True, True, True]
    ]
    assert backbone.forward_calls[1].position_ids.tolist() == [[4]]
    assert backbone.forward_calls[2].position_ids.tolist() == [[5]]
    torch.testing.assert_close(prefix.inputs_embeds, embeddings_before)
    assert torch.equal(prefix.attention_mask, attention_before)
    assert torch.equal(prefix.position_ids, positions_before)


class FakeTokenizer:
    padding_side = "left"
    truncation_side = "left"


def tiny_backbone_config() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "qwen3_0_6b",
            "repository": "Qwen/Qwen3-0.6B",
            "revision": "test-model-revision",
            "tokenizer_revision": "test-tokenizer-revision",
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
            "limits": {"max_prompt_tokens": 3, "max_new_tokens": 3},
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


def tiny_qwen_adapter() -> BackboneAdapter:
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
        )
    )
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        return BackboneAdapter(tiny_backbone_config(), cpu_info())


def test_tiny_qwen_cached_greedy_matches_full_recomputation() -> None:
    torch.manual_seed(8917)
    adapter = tiny_qwen_adapter()
    prompt_ids = torch.tensor([[0, 2, 3]], dtype=torch.int64)
    prompt_embeddings = adapter.embed_tokens(prompt_ids)
    workspace_embeddings = torch.linspace(
        -0.5,
        0.5,
        steps=2 * adapter.hidden_size,
        dtype=prompt_embeddings.dtype,
    ).reshape(1, 2, adapter.hidden_size)
    prefix = SecondPassGenerationPrefixOutput(
        inputs_embeds=torch.cat((prompt_embeddings, workspace_embeddings), dim=1),
        attention_mask=torch.tensor(
            [[False, True, True, True, True]],
            dtype=torch.bool,
        ),
        position_ids=torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.int64),
    )
    decoder = GreedyIrisDecoder(adapter, max_new_tokens=3, stop_token_ids=())

    cached = decoder.generate(prefix, workspace_trace=())

    with torch.inference_mode():
        full_embeddings = prefix.inputs_embeds.detach().clone()
        full_attention = prefix.attention_mask.detach().clone()
        full_positions = prefix.position_ids.detach().clone()
        expected_tokens: list[torch.Tensor] = []
        for generated_count in range(1, 4):
            causal = adapter.forward_embeddings(
                full_embeddings,
                full_attention,
                full_positions,
                logits_to_keep=1,
            )
            token = torch.argmax(causal.logits, dim=-1)
            expected_tokens.append(token)
            if generated_count < 3:
                full_embeddings = torch.cat(
                    (full_embeddings, adapter.embed_tokens(token)),
                    dim=1,
                )
                full_attention = torch.cat(
                    (full_attention, torch.ones(1, 1, dtype=torch.bool)),
                    dim=1,
                )
                full_positions = torch.cat(
                    (
                        full_positions,
                        prefix.position_ids[:, -1:] + generated_count,
                    ),
                    dim=1,
                )
        expected = torch.cat(expected_tokens, dim=1)

    assert cached.stop_reason == "max_new_tokens"
    assert cached.matched_stop_token_id is None
    assert torch.equal(cached.token_ids, expected)
