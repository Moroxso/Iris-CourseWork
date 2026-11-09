"""Contract tests for transactional temperature/top-p Iris sampling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache

from iris.backbone import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import (
    IrisGenerationError,
    IrisGenerationOutput,
    SamplingIrisDecoder,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
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
    """Transformers cache test double with observable physical length."""

    def __init__(self, sequence_length: int = 0) -> None:
        super().__init__(layers=[])
        self.sequence_length = sequence_length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.sequence_length


@dataclass(frozen=True, slots=True)
class ForwardCall:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor | None
    incoming_cache: Cache | None
    incoming_cache_length: int | None
    use_cache: bool
    logits_to_keep: int


class SamplingBackbone(BackboneAdapter):
    """Frozen-backbone boundary double with scripted logits and cache growth."""

    def __init__(
        self,
        planned_logits: list[torch.Tensor],
        *,
        hidden_size: int = HIDDEN_SIZE,
        vocabulary_size: int = VOCABULARY_SIZE,
        max_position_embeddings: int = MAX_CONTEXT_TOKENS,
        embedding_dtype: torch.dtype = torch.float32,
        forward_error_on_call: int | None = None,
        mutate_received_tensors: bool = False,
    ) -> None:
        self._script_hidden_size = hidden_size
        self._script_vocabulary_size = vocabulary_size
        self._script_max_positions = max_position_embeddings
        self.embedding_dtype = embedding_dtype
        self.planned_logits = planned_logits
        self.forward_error_on_call = forward_error_on_call
        self.mutate_received_tensors = mutate_received_tensors
        self.forward_calls: list[ForwardCall] = []
        self.embed_calls: list[torch.Tensor] = []
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
        values = input_ids.to(dtype=self.embedding_dtype).unsqueeze(-1)
        embeddings = values.expand(-1, -1, self.hidden_size).clone()
        if self.mutate_received_tensors:
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
            )
        )
        if self.forward_error_on_call == call_index:
            raise RuntimeError("scripted sampling backbone failure")
        if call_index >= len(self.planned_logits):
            raise AssertionError("sampling decoder made too many model calls")

        if past_key_values is None:
            cache: Cache = FakeCache(inputs_embeds.shape[1])
        else:
            assert isinstance(past_key_values, FakeCache)
            past_key_values.sequence_length += inputs_embeds.shape[1]
            cache = past_key_values
        self.returned_caches.append(cache)
        logits = (
            self.planned_logits[call_index]
            .to(
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            .reshape(1, 1, self.vocabulary_size)
        )
        self.emitted_logits.append(logits)
        output = BackboneCausalOutput(logits=logits, loss=None, past_key_values=cache)
        if self.mutate_received_tensors:
            inputs_embeds.zero_()
            attention_mask.zero_()
            position_ids.fill_(99)
        return output


def logits_selecting(*token_ids: int) -> list[torch.Tensor]:
    """Return score vectors with one dominant token apiece."""
    result: list[torch.Tensor] = []
    for token_id in token_ids:
        logits = torch.full((VOCABULARY_SIZE,), -10.0)
        logits[token_id] = 10.0
        result.append(logits)
    return result


def make_prefix(*, dtype: torch.dtype = torch.float32) -> SecondPassGenerationPrefixOutput:
    embeddings = torch.arange(
        PREFIX_LENGTH * HIDDEN_SIZE,
        dtype=dtype,
    ).reshape(1, PREFIX_LENGTH, HIDDEN_SIZE)
    attention = torch.tensor(
        [[False, False, True, True, True, True]],
        dtype=torch.bool,
    )
    positions = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.int64)
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


def make_decoder(
    backbone: SamplingBackbone,
    *,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...] = (),
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> SamplingIrisDecoder:
    return SamplingIrisDecoder(
        backbone,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        sampling=TemperatureTopPSamplingConfig(
            temperature=temperature,
            top_p=top_p,
        ),
    )


def generate(
    backbone: SamplingBackbone,
    *,
    max_new_tokens: int,
    generator: torch.Generator,
    stop_token_ids: tuple[int, ...] = (),
    temperature: float = 1.0,
    top_p: float = 1.0,
    prefix: SecondPassGenerationPrefixOutput | None = None,
    workspace_trace: tuple[WorkspaceState, ...] = (),
) -> IrisGenerationOutput:
    decoder = make_decoder(
        backbone,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        top_p=top_p,
    )
    return decoder.generate(
        make_prefix() if prefix is None else prefix,
        workspace_trace=workspace_trace,
        generator=generator,
    )


def seeded_generator(seed: int = 1234) -> torch.Generator:
    return torch.Generator(device=CPU).manual_seed(seed)


def test_sampling_config_defaults_and_frozen_slotted_shape() -> None:
    config = TemperatureTopPSamplingConfig()

    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert [field.name for field in fields(config)] == ["temperature", "top_p"]
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.top_p = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "temperature",
    [
        cast(Any, True),
        cast(Any, False),
        cast(Any, 1),
        cast(Any, 0),
        cast(Any, "1.0"),
        cast(Any, None),
        0.0,
        -0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_sampling_config_rejects_non_float_nonfinite_or_nonpositive_temperature(
    temperature: object,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        TemperatureTopPSamplingConfig(temperature=cast(Any, temperature))


@pytest.mark.parametrize(
    "top_p",
    [
        cast(Any, True),
        cast(Any, False),
        cast(Any, 1),
        cast(Any, 0),
        cast(Any, "1.0"),
        cast(Any, None),
        0.0,
        -0.0,
        -1.0,
        1.0000000000000002,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_sampling_config_rejects_non_float_nonfinite_or_out_of_range_top_p(
    top_p: object,
) -> None:
    with pytest.raises(ValueError, match="top_p"):
        TemperatureTopPSamplingConfig(top_p=cast(Any, top_p))


def test_sampling_config_accepts_closed_upper_and_tiny_positive_boundaries() -> None:
    smallest_positive = float.fromhex("0x0.0000000000001p-1022")

    config = TemperatureTopPSamplingConfig(
        temperature=smallest_positive,
        top_p=1.0,
    )
    tiny_nucleus = TemperatureTopPSamplingConfig(
        temperature=1.0,
        top_p=smallest_positive,
    )

    assert config.temperature == smallest_positive
    assert config.top_p == 1.0
    assert tiny_nucleus.top_p == smallest_positive


def capture_probabilities(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    selected_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Generator, IrisGenerationOutput]:
    captured: list[torch.Tensor] = []
    seen_generators: list[torch.Generator | None] = []

    def fake_multinomial(
        probabilities: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del replacement, out
        assert num_samples == 1
        captured.append(probabilities.detach().clone())
        seen_generators.append(generator)
        return torch.tensor(
            [[selected_token_id]],
            dtype=torch.int64,
            device=probabilities.device,
        )

    caller_generator = seeded_generator()
    backbone = SamplingBackbone(
        [logits],
        vocabulary_size=logits.numel(),
        embedding_dtype=logits.dtype,
    )
    with patch("torch.multinomial", side_effect=fake_multinomial):
        output = generate(
            backbone,
            max_new_tokens=1,
            generator=caller_generator,
            temperature=temperature,
            top_p=top_p,
            prefix=make_prefix(dtype=logits.dtype),
        )

    assert len(captured) == 1
    assert len(seen_generators) == 1
    assert isinstance(seen_generators[0], torch.Generator)
    assert seen_generators[0] is not caller_generator
    return captured[0], caller_generator, output


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (0.5, [0.5, 0.5, 0.0, 0.0]),
        (0.5000000000000001, [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0]),
    ],
)
def test_exact_nucleus_threshold_and_stable_lower_id_ties(
    top_p: float,
    expected: list[float],
) -> None:
    probabilities, _, _ = capture_probabilities(
        torch.zeros(4),
        temperature=1.0,
        top_p=top_p,
    )

    assert probabilities.dtype == torch.float64
    torch.testing.assert_close(
        probabilities,
        torch.tensor([expected], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_exact_threshold_uses_shifted_cumulative_not_rounding_prone_subtraction() -> None:
    # The two largest probabilities sum to the threshold exactly in FP64. Computing
    # `cumulative - current` for the third candidate rounds one ULP below it and
    # would incorrectly retain token 0; shifting cumsum must exclude that token.
    logits = torch.tensor(
        [-9.643329988281465, 0.4509888547443417, -3.8026345019834267],
        dtype=torch.float64,
    )
    threshold = 0.9999592670074917

    probabilities, _, _ = capture_probabilities(
        logits,
        temperature=1.0,
        top_p=threshold,
        selected_token_id=1,
    )

    assert probabilities[0, 0].item() == 0.0
    assert bool((probabilities[0, 1:] > 0).all())
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_tiny_top_p_keeps_exactly_the_highest_probability_token() -> None:
    probabilities, _, output = capture_probabilities(
        torch.tensor([-2.0, 4.0, 3.0, 0.0]),
        temperature=1.0,
        top_p=1e-12,
        selected_token_id=1,
    )

    assert probabilities.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert output.token_ids.tolist() == [[1]]


def test_top_p_one_scatter_matches_plain_temperature_softmax_in_vocab_order() -> None:
    logits = torch.tensor([0.0, 3.0, -1.0, 2.0])
    probabilities, _, _ = capture_probabilities(
        logits,
        temperature=2.0,
        top_p=1.0,
        selected_token_id=3,
    )
    expected = torch.softmax(logits.to(torch.float64) / 2.0, dim=-1).unsqueeze(0)

    torch.testing.assert_close(probabilities, expected, rtol=1e-14, atol=1e-15)


def test_top_p_one_matches_plain_multinomial_for_the_same_generator_state() -> None:
    logits = torch.tensor([0.0, 3.0, -1.0, 2.0, 0.5, -2.0, 1.0, 0.25])
    temperature = 0.65
    caller = seeded_generator(1652)
    initial_state = caller.get_state().clone()
    reference = seeded_generator(0)
    reference.set_state(initial_state)
    expected_probabilities = torch.softmax(
        logits.to(torch.float64) / temperature,
        dim=-1,
    ).unsqueeze(0)
    expected_token = torch.multinomial(
        expected_probabilities,
        1,
        generator=reference,
    )

    output = generate(
        SamplingBackbone([logits]),
        max_new_tokens=1,
        generator=caller,
        temperature=temperature,
        top_p=1.0,
    )

    assert torch.equal(output.token_ids, expected_token)
    assert torch.equal(caller.get_state(), reference.get_state())


def test_extreme_finite_logits_and_tiny_temperature_stay_finite_and_normalized() -> None:
    limit = torch.finfo(torch.float64).max
    probabilities, _, output = capture_probabilities(
        torch.tensor([-limit, limit, 0.0, limit / 2], dtype=torch.float64),
        temperature=float.fromhex("0x0.0000000000001p-1022"),
        top_p=1.0,
        selected_token_id=1,
    )

    assert bool(torch.isfinite(probabilities).all())
    assert bool((probabilities >= 0).all())
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert probabilities.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert output.token_ids.tolist() == [[1]]


def test_same_generator_state_reproduces_tokens_and_final_state() -> None:
    logits = [
        torch.tensor([0.1, 0.3, -0.2, 1.1, 0.0, -0.4, 0.8, 0.2]),
        torch.tensor([1.2, -0.1, 0.4, 0.0, 0.7, -0.3, 0.2, 0.5]),
        torch.tensor([-0.2, 0.8, 0.1, 0.5, -0.1, 1.0, 0.3, 0.0]),
    ]
    first_generator = seeded_generator(9182)
    second_generator = seeded_generator(9182)

    first = generate(
        SamplingBackbone(logits),
        max_new_tokens=3,
        generator=first_generator,
        temperature=0.7,
        top_p=0.85,
    )
    second = generate(
        SamplingBackbone(logits),
        max_new_tokens=3,
        generator=second_generator,
        temperature=0.7,
        top_p=0.85,
    )

    assert torch.equal(first.token_ids, second.token_ids)
    assert first.stop_reason == second.stop_reason == "max_new_tokens"
    assert torch.equal(first_generator.get_state(), second_generator.get_state())


def test_success_commits_exactly_one_multinomial_draw_per_returned_token() -> None:
    generator = seeded_generator(711)
    initial_state = generator.get_state().clone()
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE) for _ in range(3)])

    output = generate(backbone, max_new_tokens=3, generator=generator)

    reference = seeded_generator(0)
    reference.set_state(initial_state)
    uniform = torch.full(
        (1, VOCABULARY_SIZE),
        1.0 / VOCABULARY_SIZE,
        dtype=torch.float64,
    )
    for _ in range(output.token_ids.shape[1]):
        torch.multinomial(uniform, 1, generator=reference)
    assert output.token_ids.shape == (1, 3)
    assert not torch.equal(generator.get_state(), initial_state)
    assert torch.equal(generator.get_state(), reference.get_state())


def test_stop_token_consumes_and_commits_its_single_sample() -> None:
    generator = seeded_generator(274)
    initial_state = generator.get_state().clone()
    all_tokens_are_stops = tuple(range(VOCABULARY_SIZE))

    output = generate(
        SamplingBackbone([torch.zeros(VOCABULARY_SIZE)]),
        max_new_tokens=3,
        generator=generator,
        stop_token_ids=all_tokens_are_stops,
    )

    reference = seeded_generator(0)
    reference.set_state(initial_state)
    torch.multinomial(
        torch.full(
            (1, VOCABULARY_SIZE),
            1.0 / VOCABULARY_SIZE,
            dtype=torch.float64,
        ),
        1,
        generator=reference,
    )
    assert output.token_ids.shape == (1, 1)
    assert output.stop_reason == "stop_token"
    assert output.matched_stop_token_id == int(output.token_ids.item())
    assert torch.equal(generator.get_state(), reference.get_state())


def test_global_rng_and_deterministic_flags_are_unchanged_on_success() -> None:
    global_rng_before = torch.random.get_rng_state().clone()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()

    generate(
        SamplingBackbone([torch.zeros(VOCABULARY_SIZE) for _ in range(2)]),
        max_new_tokens=2,
        generator=seeded_generator(44),
    )

    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before


def test_prefill_failure_rolls_back_caller_generator_state() -> None:
    generator = seeded_generator(883)
    state_before = generator.get_state().clone()
    backbone = SamplingBackbone(
        [torch.zeros(VOCABULARY_SIZE)],
        forward_error_on_call=0,
    )

    with pytest.raises(IrisGenerationError, match="prefill"):
        generate(backbone, max_new_tokens=1, generator=generator)

    assert torch.equal(generator.get_state(), state_before)


def test_mid_decode_backbone_failure_rolls_back_prior_sample() -> None:
    generator = seeded_generator(884)
    state_before = generator.get_state().clone()
    backbone = SamplingBackbone(
        [torch.zeros(VOCABULARY_SIZE), torch.zeros(VOCABULARY_SIZE)],
        forward_error_on_call=1,
    )

    with pytest.raises(IrisGenerationError, match="cached generation step"):
        generate(backbone, max_new_tokens=2, generator=generator)

    assert len(backbone.forward_calls) == 2
    assert torch.equal(generator.get_state(), state_before)


def test_mid_decode_multinomial_failure_rolls_back_all_local_samples() -> None:
    generator = seeded_generator(885)
    state_before = generator.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    original_multinomial = torch.multinomial
    call_count = 0

    def fail_second_multinomial(
        probabilities: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("multinomial failed")
        return original_multinomial(
            probabilities,
            num_samples,
            replacement,
            generator=generator,
            out=out,
        )

    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE) for _ in range(2)])
    with (
        patch("torch.multinomial", side_effect=fail_second_multinomial),
        pytest.raises(IrisGenerationError),
    ):
        generate(backbone, max_new_tokens=2, generator=generator)

    assert call_count == 2
    assert torch.equal(generator.get_state(), state_before)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_exact_model_embed_multinomial_cache_mask_and_position_counts() -> None:
    backbone = SamplingBackbone(logits_selecting(4, 3, 7))
    generator = seeded_generator(99)
    original_multinomial = torch.multinomial

    with patch("torch.multinomial", wraps=original_multinomial) as multinomial:
        output = generate(
            backbone,
            max_new_tokens=3,
            generator=generator,
            top_p=1e-12,
        )

    assert output.token_ids.tolist() == [[4, 3, 7]]
    assert multinomial.call_count == 3
    assert len(backbone.forward_calls) == 3
    assert len(backbone.embed_calls) == 2
    assert [ids.tolist() for ids in backbone.embed_calls] == [[[4]], [[3]]]
    prefill, first_step, second_step = backbone.forward_calls
    assert prefill.incoming_cache is None
    assert first_step.incoming_cache is backbone.returned_caches[0]
    assert second_step.incoming_cache is backbone.returned_caches[1]
    assert first_step.incoming_cache_length == PREFIX_LENGTH
    assert second_step.incoming_cache_length == PREFIX_LENGTH + 1
    assert first_step.attention_mask.tolist() == [[False, False, True, True, True, True, True]]
    assert second_step.attention_mask.tolist() == [
        [False, False, True, True, True, True, True, True]
    ]
    assert first_step.position_ids.tolist() == [[4]]
    assert second_step.position_ids.tolist() == [[5]]
    assert all(call.labels is None for call in backbone.forward_calls)
    assert all(call.use_cache is True for call in backbone.forward_calls)
    assert all(call.logits_to_keep == 1 for call in backbone.forward_calls)


def test_stop_overrides_max_and_final_token_is_not_embedded_or_cached() -> None:
    backbone = SamplingBackbone(logits_selecting(1, 5, 7))

    output = generate(
        backbone,
        max_new_tokens=2,
        generator=seeded_generator(100),
        stop_token_ids=(5,),
        top_p=1e-12,
    )

    assert output.token_ids.tolist() == [[1, 5]]
    assert output.stop_reason == "stop_token"
    assert output.matched_stop_token_id == 5
    assert len(backbone.forward_calls) == 2
    assert len(backbone.embed_calls) == 1
    assert backbone.returned_caches[-1].get_seq_length() == PREFIX_LENGTH + 1


def test_prefix_and_trace_are_immutable_even_under_adversarial_backbone_mutation() -> None:
    prefix = make_prefix()
    prefix_before = (
        prefix.inputs_embeds.clone(),
        prefix.attention_mask.clone(),
        prefix.position_ids.clone(),
    )
    trace = make_trace()
    trace_before = tuple(
        tuple(getattr(state, field.name).clone() for field in fields(state)) for state in trace
    )
    backbone = SamplingBackbone(
        logits_selecting(1, 2, 3),
        mutate_received_tensors=True,
    )

    output = generate(
        backbone,
        max_new_tokens=3,
        generator=seeded_generator(101),
        top_p=1e-12,
        prefix=prefix,
        workspace_trace=trace,
    )

    assert output.token_ids.tolist() == [[1, 2, 3]]
    assert output.workspace_trace is trace
    torch.testing.assert_close(prefix.inputs_embeds, prefix_before[0])
    assert torch.equal(prefix.attention_mask, prefix_before[1])
    assert torch.equal(prefix.position_ids, prefix_before[2])
    for state, snapshots in zip(trace, trace_before, strict=True):
        for field, snapshot in zip(fields(state), snapshots, strict=True):
            torch.testing.assert_close(getattr(state, field.name), snapshot)
    assert backbone.forward_calls[1].position_ids.tolist() == [[4]]
    assert backbone.forward_calls[2].position_ids.tolist() == [[5]]


def test_rejects_non_generator_before_prefill_without_state_or_rng_changes() -> None:
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])
    decoder = make_decoder(backbone, max_new_tokens=1)
    global_before = torch.random.get_rng_state().clone()

    with pytest.raises(IrisGenerationError, match=r"[Gg]enerator"):
        decoder.generate(
            make_prefix(),
            workspace_trace=(),
            generator=cast(Any, object()),
        )

    assert backbone.forward_calls == []
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_rejects_cpu_global_default_generator_without_advancing_or_prefilling() -> None:
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])
    decoder = make_decoder(backbone, max_new_tokens=1)
    global_before = torch.random.get_rng_state().clone()

    with pytest.raises(IrisGenerationError, match=r"caller-owned|global default"):
        decoder.generate(
            make_prefix(),
            workspace_trace=(),
            generator=torch.default_generator,
        )

    assert backbone.forward_calls == []
    assert torch.equal(torch.random.get_rng_state(), global_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a second generator device")
def test_rejects_generator_on_a_different_device_before_prefill() -> None:
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])
    decoder = make_decoder(backbone, max_new_tokens=1)
    cuda_generator = torch.Generator(device=torch.device("cuda"))

    with pytest.raises(IrisGenerationError, match="device"):
        decoder.generate(
            make_prefix(),
            workspace_trace=(),
            generator=cuda_generator,
        )

    assert backbone.forward_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_accepts_abstract_cuda_generator_for_concrete_current_device_prefix() -> None:
    device_index = torch.cuda.current_device()
    concrete_cuda = torch.device("cuda", device_index)
    cpu_prefix = make_prefix()
    prefix = SecondPassGenerationPrefixOutput(
        inputs_embeds=cpu_prefix.inputs_embeds.to(device=concrete_cuda),
        attention_mask=cpu_prefix.attention_mask.to(device=concrete_cuda),
        position_ids=cpu_prefix.position_ids.to(device=concrete_cuda),
    )
    # The constructor is intentionally given the abstract `cuda` device. It denotes
    # the current CUDA device and must compare equal to the concrete prefix device.
    generator = torch.Generator(device=torch.device("cuda")).manual_seed(1701)
    backbone = SamplingBackbone(logits_selecting(2))
    decoder = make_decoder(backbone, max_new_tokens=1, top_p=1e-12)

    output = decoder.generate(prefix, workspace_trace=(), generator=generator)

    assert output.token_ids.tolist() == [[2]]
    assert output.token_ids.device == concrete_cuda
    assert len(backbone.forward_calls) == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_rejects_generator_on_wrong_concrete_cuda_index_before_prefill() -> None:
    current_index = torch.cuda.current_device()
    other_index = next(
        index for index in range(torch.cuda.device_count()) if index != current_index
    )
    prefix_device = torch.device("cuda", current_index)
    cpu_prefix = make_prefix()
    prefix = SecondPassGenerationPrefixOutput(
        inputs_embeds=cpu_prefix.inputs_embeds.to(device=prefix_device),
        attention_mask=cpu_prefix.attention_mask.to(device=prefix_device),
        position_ids=cpu_prefix.position_ids.to(device=prefix_device),
    )
    wrong_generator = torch.Generator(device=torch.device("cuda", other_index))
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])
    decoder = make_decoder(backbone, max_new_tokens=1)

    with pytest.raises(IrisGenerationError, match="device"):
        decoder.generate(
            prefix,
            workspace_trace=(),
            generator=wrong_generator,
        )

    assert backbone.forward_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA default generator")
def test_rejects_cuda_global_default_generator_without_advancing_or_prefilling() -> None:
    device_index = torch.cuda.current_device()
    cuda = torch.device("cuda", device_index)
    cpu_prefix = make_prefix()
    prefix = SecondPassGenerationPrefixOutput(
        inputs_embeds=cpu_prefix.inputs_embeds.to(device=cuda),
        attention_mask=cpu_prefix.attention_mask.to(device=cuda),
        position_ids=cpu_prefix.position_ids.to(device=cuda),
    )
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])
    decoder = make_decoder(backbone, max_new_tokens=1)
    cuda_default = torch.cuda.default_generators[device_index]
    state_before = cuda_default.get_state().clone()

    with pytest.raises(IrisGenerationError, match=r"caller-owned|global default"):
        decoder.generate(
            prefix,
            workspace_trace=(),
            generator=cuda_default,
        )

    assert backbone.forward_calls == []
    assert torch.equal(cuda_default.get_state(), state_before)


def test_constructor_requires_sampling_config_instance() -> None:
    backbone = SamplingBackbone([torch.zeros(VOCABULARY_SIZE)])

    with pytest.raises(ValueError, match="sampling"):
        SamplingIrisDecoder(
            backbone,
            max_new_tokens=1,
            stop_token_ids=(),
            sampling=cast(Any, object()),
        )


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


def test_tiny_qwen_seeded_sampling_is_reproducible_and_frozen() -> None:
    torch.manual_seed(6341)
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
    decoder = SamplingIrisDecoder(
        adapter,
        max_new_tokens=3,
        stop_token_ids=(),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )
    first_generator = seeded_generator(4455)
    second_generator = seeded_generator(4455)

    first = decoder.generate(prefix, workspace_trace=(), generator=first_generator)
    second = decoder.generate(prefix, workspace_trace=(), generator=second_generator)

    assert first.token_ids.shape == (1, 3)
    assert first.token_ids.dtype == torch.int64
    assert bool((first.token_ids >= 0).all())
    assert bool((first.token_ids < adapter.vocabulary_size).all())
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    assert all(parameter.grad is None for parameter in adapter._model.parameters())
    assert adapter.has_parameter_gradients is False
