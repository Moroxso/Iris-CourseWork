"""Contract tests for relative-depth backbone feature sampling."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from iris.backbone import BackboneForwardOutput
from iris.cognitive import LayerSampler, LayerSamplerError
from iris.schemas import LayerSamplerConfig


def layer_sampler_config(
    fractions: tuple[float, float, float, float] = (0.25, 0.5, 0.75, 1.0),
) -> LayerSamplerConfig:
    return LayerSamplerConfig(layer_fractions=fractions)


def hidden_states(
    num_layers: int,
    *,
    batch_size: int = 1,
    sequence_length: int = 1,
    hidden_size: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Build embedding-plus-layer states whose value identifies their tuple position."""
    states = [
        torch.full((batch_size, sequence_length, hidden_size), float(index))
        for index in range(num_layers + 1)
    ]
    states[0].fill_(10_000.0)
    return tuple(states)


def backbone_output(
    states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None = None,
) -> BackboneForwardOutput:
    batch_size, sequence_length = states[0].shape[:2]
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    return BackboneForwardOutput(
        logits=torch.zeros((batch_size, sequence_length, 7)),
        hidden_states=states,
        attention_mask=attention_mask,
    )


def projection_layers(sampler: LayerSampler) -> list[nn.Linear]:
    layers = [module for module in sampler.modules() if isinstance(module, nn.Linear)]
    assert len(layers) == 4
    return layers


def mixing_logits(sampler: LayerSampler) -> nn.Parameter:
    projections = projection_layers(sampler)
    projection_parameter_ids = {
        id(parameter) for projection in projections for parameter in projection.parameters()
    }
    candidates = [
        parameter
        for parameter in sampler.parameters()
        if id(parameter) not in projection_parameter_ids and parameter.shape == (4,)
    ]
    assert len(candidates) == 1
    return candidates[0]


def set_scalar_identity_projections(sampler: LayerSampler) -> None:
    with torch.no_grad():
        for projection in projection_layers(sampler):
            projection.weight.fill_(1.0)
            if projection.bias is not None:
                projection.bias.zero_()


@pytest.mark.parametrize(
    ("num_layers", "expected_indices", "expected_context"),
    [
        (28, (7, 14, 21, 28), 17.5),
        (10, (3, 5, 8, 10), 6.5),
        (5, (2, 3, 4, 5), 3.5),
    ],
)
def test_samples_relative_transformer_depths_with_ceiling(
    num_layers: int,
    expected_indices: tuple[int, int, int, int],
    expected_context: float,
) -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=1, state_dim=1)
    set_scalar_identity_projections(sampler)
    with torch.no_grad():
        mixing_logits(sampler).zero_()

    output = sampler(backbone_output(hidden_states(num_layers)))

    assert output.selected_layer_indices == expected_indices
    torch.testing.assert_close(
        output.context_features,
        torch.tensor([[[expected_context]]]),
    )
    torch.testing.assert_close(output.mixing_weights, torch.full((4,), 0.25))


def test_projects_each_selected_layer_independently_before_mixing() -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=2, state_dim=1)
    states = hidden_states(4, hidden_size=2)
    selected_value = torch.tensor([[[2.0, 3.0]]])
    states = (states[0],) + (selected_value,) * 4
    projection_weights = (
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, -1.0]]),
    )
    with torch.no_grad():
        for projection, weight in zip(projection_layers(sampler), projection_weights, strict=True):
            projection.weight.copy_(weight)
            if projection.bias is not None:
                projection.bias.zero_()
        mixing_logits(sampler).zero_()

    output = sampler(backbone_output(states))

    # (2 + 3 + 5 - 1) / 4 proves four distinct H -> D maps precede mixing.
    torch.testing.assert_close(output.context_features, torch.tensor([[[2.25]]]))


def test_returns_batched_features_and_canonical_boolean_mask() -> None:
    torch.manual_seed(7)
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=4, state_dim=5)
    states = tuple(torch.randn(2, 3, 4) for _ in range(9))
    attention_mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.int16)

    output = sampler(backbone_output(states, attention_mask))

    assert output.context_features.shape == (2, 3, 5)
    assert output.context_features.dtype == torch.float32
    assert output.context_features.device.type == "cpu"
    assert torch.isfinite(output.context_features).all()
    assert output.attention_mask.dtype == torch.bool
    assert torch.equal(output.attention_mask, attention_mask.bool())
    assert output.mixing_weights.shape == (4,)
    assert bool((output.mixing_weights > 0.0).all())
    torch.testing.assert_close(output.mixing_weights.sum(), torch.tensor(1.0))


def test_uses_softmax_normalized_custom_mixing_logits() -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=1, state_dim=1)
    set_scalar_identity_projections(sampler)
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
    with torch.no_grad():
        mixing_logits(sampler).copy_(probabilities.log())

    output = sampler(backbone_output(hidden_states(4)))

    torch.testing.assert_close(output.mixing_weights, probabilities)
    torch.testing.assert_close(output.context_features, torch.tensor([[[3.0]]]))


def test_materializes_inference_tensors_for_trainable_sampler_backward() -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=2, state_dim=2)
    with torch.no_grad():
        for projection in projection_layers(sampler):
            projection.weight.copy_(torch.eye(2))
            if projection.bias is not None:
                projection.bias.zero_()
        mixing_logits(sampler).zero_()
    with torch.inference_mode():
        states = tuple(torch.full((1, 2, 2), float(index)) for index in range(5))
        output_from_backbone = BackboneForwardOutput(
            logits=torch.zeros(1, 2, 7),
            hidden_states=states,
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )

    assert all(state.is_inference() for state in states)

    output = sampler(output_from_backbone)
    loss = output.context_features.square().mean()
    loss.backward()

    assert output.context_features.is_inference() is False
    assert output.context_features.requires_grad is True
    for projection in projection_layers(sampler):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
    logits_gradient = mixing_logits(sampler).grad
    assert logits_gradient is not None
    assert torch.isfinite(logits_gradient).all()
    assert torch.count_nonzero(logits_gradient) > 0
    assert all(state.grad is None for state in states)


def test_supports_frozen_sampler_execution_under_inference_mode() -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=2, state_dim=2)
    sampler.requires_grad_(False)

    with torch.inference_mode():
        states = tuple(torch.full((1, 2, 2), float(index)) for index in range(5))
        output = sampler(backbone_output(states))

    assert output.context_features.is_inference()
    assert output.context_features.requires_grad is False
    assert torch.isfinite(output.context_features).all()


HiddenStateMutation = Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]


def replace_state(
    states: tuple[torch.Tensor, ...], index: int, value: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    mutable_states = list(states)
    mutable_states[index] = value
    return tuple(mutable_states)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda states: replace_state(states, 1, torch.ones(1, 2)),
        lambda states: replace_state(states, 2, torch.ones(1, 2, 2)),
        lambda states: replace_state(states, 3, torch.ones(1, 3, 1)),
        lambda states: replace_state(states, 4, torch.ones(1, 2, 1, dtype=torch.int64)),
        lambda states: replace_state(states, 1, torch.full((1, 2, 1), torch.nan)),
        lambda states: replace_state(states, 2, torch.ones(1, 2, 1, requires_grad=True)),
        lambda states: replace_state(states, 3, torch.ones(1, 2, 1, dtype=torch.float64)),
    ],
    ids=[
        "rank",
        "hidden-size",
        "sequence-length",
        "integer-dtype",
        "nonfinite",
        "gradient-tracking",
        "mixed-dtype",
    ],
)
def test_rejects_malformed_selected_hidden_states(
    mutation: HiddenStateMutation,
) -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=1, state_dim=1)
    states = tuple(torch.ones(1, 2, 1) for _ in range(5))

    with pytest.raises(LayerSamplerError):
        sampler(backbone_output(mutation(states)))


@pytest.mark.parametrize(
    "attention_mask",
    [
        torch.ones(2, 3, dtype=torch.float32),
        torch.tensor([[1, 2, 1], [1, 1, 1]], dtype=torch.long),
        torch.ones(2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.long),
        torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long),
    ],
    ids=["floating-dtype", "nonbinary", "rank", "shape", "empty-row"],
)
def test_rejects_malformed_attention_masks(attention_mask: torch.Tensor) -> None:
    sampler = LayerSampler(layer_sampler_config(), backbone_hidden_size=1, state_dim=1)
    states = hidden_states(4, batch_size=2, sequence_length=3)

    with pytest.raises(LayerSamplerError):
        sampler(backbone_output(states, attention_mask))


@pytest.mark.parametrize(
    ("fractions", "num_layers"),
    [
        ((0.25, 0.5, 0.75, 1.0), 3),
        ((0.1, 0.2, 0.3, 1.0), 4),
    ],
)
def test_rejects_relative_depth_collisions(
    fractions: tuple[float, float, float, float], num_layers: int
) -> None:
    sampler = LayerSampler(layer_sampler_config(fractions), backbone_hidden_size=1, state_dim=1)

    with pytest.raises(LayerSamplerError):
        sampler(backbone_output(hidden_states(num_layers)))
