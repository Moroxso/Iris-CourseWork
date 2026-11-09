"""Contract tests for request-level goal attention pooling."""

from math import sqrt

import pytest
import torch

from iris.backbone import BackboneForwardOutput
from iris.cognitive import GoalEncoder, GoalEncoderError, LayerSampler
from iris.schemas import LayerSamplerConfig


def test_zero_query_is_masked_mean_with_normalized_attention() -> None:
    encoder = GoalEncoder(state_dim=3)
    context = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [100.0, 200.0, 300.0]],
            [[7.0, 8.0, 9.0], [11.0, 12.0, 13.0], [15.0, 16.0, 17.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.int16)

    output = encoder(context, mask)

    assert output.goal_state.shape == (2, 3)
    assert output.attention_weights.shape == (2, 3)
    torch.testing.assert_close(
        output.attention_weights,
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        output.goal_state,
        torch.tensor([[2.0, 3.0, 4.0], [11.0, 12.0, 13.0]]),
    )


def test_masked_finite_padding_values_cannot_change_goal() -> None:
    encoder = GoalEncoder(state_dim=2)
    with torch.no_grad():
        encoder.pooling_query.copy_(torch.tensor([0.75, -0.25]))
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    context = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[-1.0, 2.0], [0.0, 0.0], [4.0, -3.0], [0.0, 0.0]],
        ]
    )
    changed_padding = context.clone()
    changed_padding[~mask] = torch.tensor(
        [[10_000.0, -10_000.0], [-7_777.0, 8_888.0], [123.0, -456.0], [9_999.0, 1.0]]
    )

    original = encoder(context, mask)
    perturbed = encoder(changed_padding, mask)

    torch.testing.assert_close(perturbed.goal_state, original.goal_state)
    torch.testing.assert_close(perturbed.attention_weights, original.attention_weights)
    assert torch.count_nonzero(perturbed.attention_weights.masked_select(~mask)) == 0


def test_custom_query_uses_scaled_dot_product_softmax() -> None:
    encoder = GoalEncoder(state_dim=4)
    query = torch.tensor([2.0, -1.0, 0.5, 1.0])
    with torch.no_grad():
        encoder.pooling_query.copy_(query)
    context = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0, -1.0],
                [0.0, 2.0, -1.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        ]
    )
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    output = encoder(context, mask)

    expected_scores = (context[:, :2] @ query) / sqrt(4)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_goal = torch.einsum("bt,btd->bd", expected_weights, context[:, :2])
    torch.testing.assert_close(
        output.attention_weights,
        torch.cat((expected_weights, torch.zeros(1, 1)), dim=1),
    )
    torch.testing.assert_close(output.goal_state, expected_goal)


def test_backward_reaches_query_and_only_unmasked_context_positions() -> None:
    encoder = GoalEncoder(state_dim=2)
    with torch.no_grad():
        encoder.pooling_query.copy_(torch.tensor([0.5, -1.0]))
    context = torch.tensor(
        [[[1.0, 2.0], [3.0, -1.0], [8.0, 9.0], [-2.0, 4.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)

    output = encoder(context, mask)
    output.goal_state.square().sum().backward()

    assert encoder.pooling_query.grad is not None
    assert torch.isfinite(encoder.pooling_query.grad).all()
    assert torch.count_nonzero(encoder.pooling_query.grad) > 0
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert torch.count_nonzero(context.grad[mask]) > 0
    assert torch.count_nonzero(context.grad[~mask]) == 0


def test_materializes_inference_context_for_trainable_encoder_backward() -> None:
    encoder = GoalEncoder(state_dim=2)
    with torch.no_grad():
        encoder.pooling_query.copy_(torch.tensor([0.25, -0.75]))
    with torch.inference_mode():
        context = torch.tensor([[[1.0, 2.0], [4.0, -1.0], [2.0, 3.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    assert context.is_inference()

    output = encoder(context, mask)
    output.goal_state.square().sum().backward()

    assert output.goal_state.is_inference() is False
    assert output.goal_state.requires_grad is True
    assert encoder.pooling_query.grad is not None
    assert torch.isfinite(encoder.pooling_query.grad).all()
    assert torch.count_nonzero(encoder.pooling_query.grad) > 0


def test_backward_reaches_layer_sampler_without_unfreezing_backbone_states() -> None:
    torch.manual_seed(11)
    sampler = LayerSampler(
        LayerSamplerConfig(layer_fractions=(0.25, 0.5, 0.75, 1.0)),
        backbone_hidden_size=3,
        state_dim=2,
    )
    encoder = GoalEncoder(state_dim=2)
    with torch.inference_mode():
        hidden_states = tuple(torch.randn(1, 3, 3) for _ in range(5))
        backbone_output = BackboneForwardOutput(
            logits=torch.zeros(1, 3, 7),
            hidden_states=hidden_states,
            attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
        )

    sampled = sampler(backbone_output)
    encoded = encoder(sampled.context_features, sampled.attention_mask)
    encoded.goal_state.square().sum().backward()

    assert encoder.pooling_query.grad is not None
    assert torch.isfinite(encoder.pooling_query.grad).all()
    assert sampler.mixing_logits.grad is not None
    assert torch.isfinite(sampler.mixing_logits.grad).all()
    for projection in sampler.projections:
        assert isinstance(projection, torch.nn.Linear)
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
    assert all(state.grad is None for state in hidden_states)


def test_supports_trainable_parameters_under_global_inference_mode() -> None:
    encoder = GoalEncoder(state_dim=2)

    with torch.inference_mode():
        context = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones((1, 2), dtype=torch.long)
        output = encoder(context, mask)

    assert output.goal_state.is_inference()
    assert output.attention_weights.is_inference()
    assert output.goal_state.requires_grad is False
    assert output.attention_weights.requires_grad is False
    assert torch.isfinite(output.goal_state).all()


@pytest.mark.parametrize(
    "context",
    [
        torch.ones(2, 3),
        torch.ones(0, 2, 3),
        torch.ones(1, 0, 3),
        torch.ones(1, 2, 4),
        torch.ones(1, 2, 3, dtype=torch.int64),
        torch.full((1, 2, 3), torch.nan),
        torch.full((1, 2, 3), torch.inf),
    ],
    ids=["rank", "empty-batch", "empty-sequence", "width", "integer", "nan", "inf"],
)
def test_rejects_malformed_context(context: torch.Tensor) -> None:
    encoder = GoalEncoder(state_dim=3)
    mask = torch.ones(context.shape[:2], dtype=torch.long)

    with pytest.raises(GoalEncoderError):
        encoder(context, mask)


def test_rejects_nontensor_context() -> None:
    encoder = GoalEncoder(state_dim=2)

    with pytest.raises(GoalEncoderError):
        encoder([[1.0, 2.0]], torch.ones((1, 1), dtype=torch.long))


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 3, dtype=torch.float32),
        torch.tensor([[1, 2, 1], [1, 1, 1]], dtype=torch.long),
        torch.ones(2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.long),
        torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long),
    ],
    ids=["floating-dtype", "nonbinary", "rank", "shape", "fully-masked-row"],
)
def test_rejects_malformed_attention_mask(mask: torch.Tensor) -> None:
    encoder = GoalEncoder(state_dim=2)
    context = torch.ones((2, 3, 2))

    with pytest.raises(GoalEncoderError):
        encoder(context, mask)


def test_rejects_nontensor_attention_mask() -> None:
    encoder = GoalEncoder(state_dim=2)

    with pytest.raises(GoalEncoderError):
        encoder(torch.ones((1, 2, 2)), [[1, 1]])


def test_rejects_parameter_dtype_mismatch() -> None:
    encoder = GoalEncoder(state_dim=2).to(dtype=torch.float64)

    with pytest.raises(GoalEncoderError, match="parameters must explicitly match"):
        encoder(torch.ones((1, 2, 2)), torch.ones((1, 2), dtype=torch.long))


def test_rejects_nonfinite_pooling_query() -> None:
    encoder = GoalEncoder(state_dim=2)
    with torch.no_grad():
        encoder.pooling_query[0] = torch.nan

    with pytest.raises(GoalEncoderError, match="query contains NaN or Inf"):
        encoder(torch.ones((1, 2, 2)), torch.ones((1, 2), dtype=torch.long))


@pytest.mark.parametrize("state_dim", [0, -1, True, 2.5])
def test_rejects_invalid_state_dim(state_dim: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GoalEncoder(state_dim=state_dim)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "large_value", "query_value"),
    [(torch.float16, 60_000.0, 1.0), (torch.bfloat16, 2e38, 0.3)],
    ids=["float16", "bfloat16"],
)
def test_low_precision_pooling_accumulates_safely_in_float32(
    dtype: torch.dtype, large_value: float, query_value: float
) -> None:
    encoder = GoalEncoder(state_dim=8).to(dtype=dtype)
    with torch.no_grad():
        encoder.pooling_query.fill_(query_value)
    context = torch.full((1, 8, 8), large_value, dtype=dtype)
    mask = torch.ones((1, 8), dtype=torch.long)

    output = encoder(context, mask)

    assert output.goal_state.dtype == dtype
    assert output.attention_weights.dtype == torch.float32
    assert torch.isfinite(output.goal_state).all()
    assert torch.isfinite(output.attention_weights).all()
    torch.testing.assert_close(output.goal_state, context[:, 0, :])
    torch.testing.assert_close(output.attention_weights, torch.full((1, 8), 0.125))


def test_external_autocast_cannot_lower_pooling_precision() -> None:
    encoder = GoalEncoder(state_dim=4)
    with torch.no_grad():
        encoder.pooling_query.copy_(torch.tensor([0.5, -0.25, 0.75, 1.0]))
    context = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.5], [9.0, 8.0, 7.0, 6.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    reference = encoder(context, mask)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = encoder(context, mask)

    assert actual.goal_state.dtype == torch.float32
    assert actual.attention_weights.dtype == torch.float32
    torch.testing.assert_close(actual.goal_state, reference.goal_state)
    torch.testing.assert_close(actual.attention_weights, reference.attention_weights)
