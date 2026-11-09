"""Contract tests for the thin token-to-text Iris generation facade."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from iris.backbone import BackboneAdapter, BackboneDecodeError
from iris.generation import (
    IrisGenerationOutput,
    IrisGenerator,
    IrisGeneratorError,
    SecondPassGenerationPrefixOutput,
    TemperatureTopPSamplingConfig,
)
from iris.workspace import WorkspaceState

CPU = torch.device("cpu")
VOCABULARY_SIZE = 8
MAX_NEW_TOKENS = 3
STOP_TOKEN_IDS = (6, 2)


class FacadeBackbone(BackboneAdapter):
    """Nominal adapter double that exposes only the facade-owned decode boundary."""

    def __init__(
        self,
        *,
        decode_result: object = "decoded text",
        decode_error: BaseException | None = None,
    ) -> None:
        self.decode_result = decode_result
        self.decode_error = decode_error
        self.decode_calls: list[torch.Tensor] = []
        self.eos_reads = 0
        self.model_generate_calls = 0

    @property
    def vocabulary_size(self) -> int:
        return VOCABULARY_SIZE

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        self.eos_reads += 1
        raise AssertionError("facade must not add tokenizer EOS IDs to the explicit stop tuple")

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        self.decode_calls.append(token_ids)
        if self.decode_error is not None:
            raise self.decode_error
        return cast(str, self.decode_result)


@dataclass(frozen=True, slots=True)
class DecoderCall:
    prefix: object
    workspace_trace: object
    generator: torch.Generator | None


class DecoderSpy:
    """Greedy/sampling-compatible decoder double with explicit RNG advancement."""

    def __init__(
        self,
        output: object,
        *,
        error: BaseException | None = None,
        draws_before_return: int = 0,
    ) -> None:
        self.output = output
        self.error = error
        self.draws_before_return = draws_before_return
        self.calls: list[DecoderCall] = []

    def generate(
        self,
        prefix: object,
        *,
        workspace_trace: object,
        generator: torch.Generator | None = None,
    ) -> object:
        self.calls.append(
            DecoderCall(
                prefix=prefix,
                workspace_trace=workspace_trace,
                generator=generator,
            )
        )
        if self.draws_before_return:
            assert isinstance(generator, torch.Generator)
            probabilities = torch.full((1, 4), 0.25, dtype=torch.float64)
            for _ in range(self.draws_before_return):
                torch.multinomial(probabilities, 1, generator=generator)
        if self.error is not None:
            raise self.error
        return self.output


@dataclass(frozen=True, slots=True)
class FacadeBundle:
    facade: IrisGenerator
    backbone: FacadeBackbone
    greedy: DecoderSpy
    sampling: DecoderSpy
    greedy_constructor: MagicMock
    sampling_constructor: MagicMock


def make_prefix() -> SecondPassGenerationPrefixOutput:
    return SecondPassGenerationPrefixOutput(
        inputs_embeds=torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
        attention_mask=torch.tensor(
            [[False, False, True, True, True, True]],
            dtype=torch.bool,
        ),
        position_ids=torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.int64),
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


def raw_output(
    trace: tuple[WorkspaceState, ...],
    *,
    token_ids: torch.Tensor | None = None,
    stop_reason: str = "max_new_tokens",
    matched_stop_token_id: int | None = None,
) -> IrisGenerationOutput:
    if token_ids is None:
        token_ids = torch.tensor([[1, 3, 4]], dtype=torch.int64)
    return IrisGenerationOutput(
        token_ids=token_ids,
        stop_reason=cast(Any, stop_reason),
        matched_stop_token_id=matched_stop_token_id,
        workspace_trace=trace,
    )


def build_facade(
    *,
    backbone: FacadeBackbone | None = None,
    greedy_output: object,
    sampling_output: object | None = None,
    greedy_error: BaseException | None = None,
    sampling_error: BaseException | None = None,
    sampling_draws: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stop_token_ids: tuple[int, ...] = STOP_TOKEN_IDS,
    sampling_config: TemperatureTopPSamplingConfig | None = None,
) -> FacadeBundle:
    selected_backbone = backbone or FacadeBackbone()
    if sampling_output is None:
        sampling_output = greedy_output
    greedy = DecoderSpy(greedy_output, error=greedy_error)
    sampling = DecoderSpy(
        sampling_output,
        error=sampling_error,
        draws_before_return=sampling_draws,
    )
    config = sampling_config or TemperatureTopPSamplingConfig()
    with (
        patch(
            "iris.generation.facade.GreedyIrisDecoder",
            return_value=greedy,
        ) as greedy_constructor,
        patch(
            "iris.generation.facade.SamplingIrisDecoder",
            return_value=sampling,
        ) as sampling_constructor,
    ):
        facade = IrisGenerator(
            selected_backbone,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampling=config,
        )
    return FacadeBundle(
        facade=facade,
        backbone=selected_backbone,
        greedy=greedy,
        sampling=sampling,
        greedy_constructor=greedy_constructor,
        sampling_constructor=sampling_constructor,
    )


def seeded_generator(seed: int = 1234) -> torch.Generator:
    return torch.Generator(device=CPU).manual_seed(seed)


def test_constructor_builds_both_decoders_with_exact_shared_configuration() -> None:
    trace = make_trace()
    backbone = FacadeBackbone()
    config = TemperatureTopPSamplingConfig(temperature=0.7, top_p=0.8)

    bundle = build_facade(
        backbone=backbone,
        greedy_output=raw_output(trace),
        max_new_tokens=3,
        stop_token_ids=(6, 2),
        sampling_config=config,
    )

    bundle.greedy_constructor.assert_called_once_with(
        backbone,
        max_new_tokens=3,
        stop_token_ids=(6, 2),
    )
    bundle.sampling_constructor.assert_called_once_with(
        backbone,
        max_new_tokens=3,
        stop_token_ids=(6, 2),
        sampling=config,
    )
    assert bundle.facade.max_new_tokens == 3
    assert bundle.facade.stop_token_ids == (6, 2)
    assert bundle.facade.sampling is config
    assert backbone.eos_reads == 0


def test_greedy_dispatch_and_flat_output_preserve_exact_provenance_and_text() -> None:
    trace = make_trace()
    prefix = make_prefix()
    tokens = torch.tensor([[4, 6]], dtype=torch.int64)
    raw = raw_output(
        trace,
        token_ids=tokens,
        stop_reason="stop_token",
        matched_stop_token_id=6,
    )
    backbone = FacadeBackbone(decode_result=" answer<STOP> \n")
    bundle = build_facade(backbone=backbone, greedy_output=raw)

    output = bundle.facade.generate(
        prefix,
        strategy="greedy",
        workspace_trace=trace,
    )

    assert output.text == " answer<STOP> \n"
    assert output.token_ids is tokens
    assert output.stop_reason == "stop_token"
    assert output.matched_stop_token_id == 6
    assert output.workspace_trace is trace
    assert backbone.decode_calls == [tokens]
    assert bundle.greedy.calls == [
        DecoderCall(prefix=prefix, workspace_trace=trace, generator=None)
    ]
    assert bundle.sampling.calls == []
    assert backbone.eos_reads == 0
    assert backbone.model_generate_calls == 0
    assert [field.name for field in fields(output)] == [
        "text",
        "token_ids",
        "stop_reason",
        "matched_stop_token_id",
        "workspace_trace",
    ]
    assert not hasattr(output, "__dict__")
    with pytest.raises(FrozenInstanceError):
        output.text = "changed"  # type: ignore[misc]


def test_sampling_dispatch_uses_private_local_generator_and_commits_after_text() -> None:
    trace = make_trace()
    prefix = make_prefix()
    raw = raw_output(trace)
    caller = seeded_generator(719)
    state_before = caller.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    bundle = build_facade(
        greedy_output=raw,
        sampling_output=raw,
        sampling_draws=2,
    )

    output = bundle.facade.generate(
        prefix,
        strategy="sampling",
        workspace_trace=trace,
        generator=caller,
    )

    assert output.token_ids is raw.token_ids
    assert output.workspace_trace is trace
    assert bundle.greedy.calls == []
    assert len(bundle.sampling.calls) == 1
    local_generator = bundle.sampling.calls[0].generator
    assert isinstance(local_generator, torch.Generator)
    assert local_generator is not caller
    reference = seeded_generator(0)
    reference.set_state(state_before)
    probabilities = torch.full((1, 4), 0.25, dtype=torch.float64)
    torch.multinomial(probabilities, 1, generator=reference)
    torch.multinomial(probabilities, 1, generator=reference)
    assert torch.equal(caller.get_state(), reference.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_before)


@pytest.mark.parametrize("strategy", ["beam", "Greedy", "", True, 1, None, object()])
def test_invalid_strategy_rejected_before_decoder_or_tokenizer(strategy: object) -> None:
    trace = make_trace()
    bundle = build_facade(greedy_output=raw_output(trace))

    with pytest.raises(IrisGeneratorError, match="strategy"):
        bundle.facade.generate(
            make_prefix(),
            strategy=cast(Any, strategy),
            workspace_trace=trace,
        )

    assert bundle.greedy.calls == []
    assert bundle.sampling.calls == []
    assert bundle.backbone.decode_calls == []


def test_greedy_rejects_any_generator_before_dispatch() -> None:
    trace = make_trace()
    bundle = build_facade(greedy_output=raw_output(trace))

    with pytest.raises(IrisGeneratorError, match="does not accept"):
        bundle.facade.generate(
            make_prefix(),
            strategy="greedy",
            workspace_trace=trace,
            generator=seeded_generator(),
        )

    assert bundle.greedy.calls == []
    assert bundle.backbone.decode_calls == []


@pytest.mark.parametrize("generator", [None, cast(Any, object()), cast(Any, "generator")])
def test_sampling_requires_explicit_torch_generator_before_dispatch(
    generator: object,
) -> None:
    trace = make_trace()
    bundle = build_facade(greedy_output=raw_output(trace))

    with pytest.raises(IrisGeneratorError, match=r"[Gg]enerator"):
        bundle.facade.generate(
            make_prefix(),
            strategy="sampling",
            workspace_trace=trace,
            generator=cast(Any, generator),
        )

    assert bundle.sampling.calls == []
    assert bundle.backbone.decode_calls == []


def test_sampling_rejects_global_default_generator_without_global_rng_change() -> None:
    trace = make_trace()
    bundle = build_facade(greedy_output=raw_output(trace))
    global_before = torch.random.get_rng_state().clone()

    with pytest.raises(IrisGeneratorError, match="transaction") as captured:
        bundle.facade.generate(
            make_prefix(),
            strategy="sampling",
            workspace_trace=trace,
            generator=torch.default_generator,
        )

    assert captured.value.__cause__ is not None
    assert bundle.sampling.calls == []
    assert bundle.backbone.decode_calls == []
    assert torch.equal(torch.random.get_rng_state(), global_before)


@pytest.mark.parametrize("strategy", ["greedy", "sampling"])
def test_decoder_failures_are_wrapped_with_original_cause_and_not_decoded(
    strategy: str,
) -> None:
    trace = make_trace()
    decoder_error = RuntimeError(f"{strategy} decoder failed")
    bundle = build_facade(
        greedy_output=raw_output(trace),
        greedy_error=decoder_error if strategy == "greedy" else None,
        sampling_error=decoder_error if strategy == "sampling" else None,
        sampling_draws=1 if strategy == "sampling" else 0,
    )
    generator = seeded_generator(901) if strategy == "sampling" else None
    state_before = None if generator is None else generator.get_state().clone()

    with pytest.raises(IrisGeneratorError, match="generation failed") as captured:
        bundle.facade.generate(
            make_prefix(),
            strategy=cast(Any, strategy),
            workspace_trace=trace,
            generator=generator,
        )

    assert captured.value.__cause__ is decoder_error
    assert bundle.backbone.decode_calls == []
    if generator is not None:
        assert state_before is not None
        assert torch.equal(generator.get_state(), state_before)


MalformedRawMutator = Any


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw, trace: cast(Any, object()),
        lambda raw, trace: replace(raw, workspace_trace=tuple([*trace])),
        lambda raw, trace: replace(raw, token_ids=cast(Any, "not a tensor")),
        lambda raw, trace: replace(raw, token_ids=raw.token_ids[0]),
        lambda raw, trace: replace(raw, token_ids=torch.ones(2, 1, dtype=torch.int64)),
        lambda raw, trace: replace(raw, token_ids=torch.empty(1, 0, dtype=torch.int64)),
        lambda raw, trace: replace(raw, token_ids=raw.token_ids[:, :-1]),
        lambda raw, trace: replace(raw, token_ids=torch.ones(1, 4, dtype=torch.int64)),
        lambda raw, trace: replace(raw, token_ids=raw.token_ids.to(torch.int32)),
        lambda raw, trace: replace(raw, stop_reason=cast(Any, "unknown")),
        lambda raw, trace: replace(
            raw,
            stop_reason="stop_token",
            matched_stop_token_id=None,
        ),
        lambda raw, trace: replace(
            raw,
            stop_reason="stop_token",
            matched_stop_token_id=True,
        ),
        lambda raw, trace: replace(
            raw,
            stop_reason="stop_token",
            matched_stop_token_id=7,
        ),
        lambda raw, trace: replace(
            raw,
            token_ids=torch.tensor([[1, 3]], dtype=torch.int64),
            stop_reason="stop_token",
            matched_stop_token_id=6,
        ),
        lambda raw, trace: replace(
            raw,
            token_ids=torch.tensor([[1, 6, 4]], dtype=torch.int64),
        ),
        lambda raw, trace: replace(
            raw,
            token_ids=torch.tensor([[1, 6, 2]], dtype=torch.int64),
            stop_reason="stop_token",
            matched_stop_token_id=2,
        ),
        lambda raw, trace: replace(raw, matched_stop_token_id=6),
    ],
)
def test_malformed_raw_decoder_output_is_rejected_before_decode(
    mutator: MalformedRawMutator,
) -> None:
    trace = make_trace()
    valid = raw_output(trace)
    malformed = mutator(valid, trace)
    bundle = build_facade(greedy_output=malformed)

    with pytest.raises(IrisGeneratorError, match="finalize"):
        bundle.facade.generate(
            make_prefix(),
            strategy="greedy",
            workspace_trace=trace,
        )

    assert bundle.backbone.decode_calls == []


def test_sampling_malformed_raw_output_rolls_back_rng_after_decoder_draws() -> None:
    trace = make_trace()
    malformed = replace(
        raw_output(trace),
        token_ids=torch.tensor([[1, 3]], dtype=torch.int64),
    )
    caller = seeded_generator(902)
    state_before = caller.get_state().clone()
    bundle = build_facade(
        greedy_output=raw_output(trace),
        sampling_output=malformed,
        sampling_draws=2,
    )

    with pytest.raises(IrisGeneratorError, match="finalize"):
        bundle.facade.generate(
            make_prefix(),
            strategy="sampling",
            workspace_trace=trace,
            generator=caller,
        )

    assert torch.equal(caller.get_state(), state_before)
    assert bundle.backbone.decode_calls == []


@pytest.mark.parametrize(
    "malformed",
    [
        IrisGenerationOutput(
            token_ids=torch.tensor([[1, 6, 4]], dtype=torch.int64),
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
            workspace_trace=(),
        ),
        IrisGenerationOutput(
            token_ids=torch.tensor([[1, 6, 2]], dtype=torch.int64),
            stop_reason="stop_token",
            matched_stop_token_id=2,
            workspace_trace=(),
        ),
    ],
)
def test_sampling_rejects_tokens_after_first_stop_and_rolls_back_rng(
    malformed: IrisGenerationOutput,
) -> None:
    trace = make_trace()
    malformed_with_trace = replace(malformed, workspace_trace=trace)
    caller = seeded_generator(905)
    state_before = caller.get_state().clone()
    bundle = build_facade(
        greedy_output=raw_output(trace),
        sampling_output=malformed_with_trace,
        sampling_draws=2,
    )

    with pytest.raises(IrisGeneratorError, match="finalize"):
        bundle.facade.generate(
            make_prefix(),
            strategy="sampling",
            workspace_trace=trace,
            generator=caller,
        )

    assert torch.equal(caller.get_state(), state_before)
    assert bundle.backbone.decode_calls == []


@pytest.mark.parametrize("strategy", ["greedy", "sampling"])
def test_decode_failure_is_wrapped_and_sampling_transaction_rolls_back(
    strategy: str,
) -> None:
    trace = make_trace()
    decode_error = BackboneDecodeError("tokenizer failed")
    backbone = FacadeBackbone(decode_error=decode_error)
    caller = seeded_generator(903) if strategy == "sampling" else None
    state_before = None if caller is None else caller.get_state().clone()
    bundle = build_facade(
        backbone=backbone,
        greedy_output=raw_output(trace),
        sampling_draws=2,
    )

    with pytest.raises(IrisGeneratorError, match="finalize") as captured:
        bundle.facade.generate(
            make_prefix(),
            strategy=cast(Any, strategy),
            workspace_trace=trace,
            generator=caller,
        )

    assert captured.value.__cause__ is decode_error
    assert len(backbone.decode_calls) == 1
    if caller is not None:
        assert state_before is not None
        assert torch.equal(caller.get_state(), state_before)


def test_sampling_output_construction_failure_rolls_back_after_successful_decode() -> None:
    trace = make_trace()
    caller = seeded_generator(904)
    state_before = caller.get_state().clone()
    bundle = build_facade(
        greedy_output=raw_output(trace),
        sampling_draws=1,
    )
    construction_error = RuntimeError("DTO construction failed")

    with (
        patch(
            "iris.generation.facade.IrisGeneratorOutput",
            side_effect=construction_error,
        ),
        pytest.raises(IrisGeneratorError, match="finalize") as captured,
    ):
        bundle.facade.generate(
            make_prefix(),
            strategy="sampling",
            workspace_trace=trace,
            generator=caller,
        )

    assert captured.value.__cause__ is construction_error
    assert len(bundle.backbone.decode_calls) == 1
    assert torch.equal(caller.get_state(), state_before)


def test_success_does_not_mutate_prefix_raw_tokens_or_workspace_trace() -> None:
    trace = make_trace()
    trace_before = tuple(
        tuple(getattr(state, field.name).clone() for field in fields(state)) for state in trace
    )
    prefix = make_prefix()
    prefix_before = (
        prefix.inputs_embeds.clone(),
        prefix.attention_mask.clone(),
        prefix.position_ids.clone(),
    )
    tokens = torch.tensor([[1, 3, 4]], dtype=torch.int64)
    tokens_before = tokens.clone()
    raw = raw_output(trace, token_ids=tokens)
    bundle = build_facade(greedy_output=raw)

    output = bundle.facade.generate(
        prefix,
        strategy="greedy",
        workspace_trace=trace,
    )

    assert output.token_ids is tokens
    assert torch.equal(tokens, tokens_before)
    torch.testing.assert_close(prefix.inputs_embeds, prefix_before[0])
    assert torch.equal(prefix.attention_mask, prefix_before[1])
    assert torch.equal(prefix.position_ids, prefix_before[2])
    for state, snapshots in zip(trace, trace_before, strict=True):
        for field, snapshot in zip(fields(state), snapshots, strict=True):
            torch.testing.assert_close(getattr(state, field.name), snapshot)
