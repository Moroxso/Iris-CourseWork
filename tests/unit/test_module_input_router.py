"""Contract tests for per-module soft prompt routing."""

from math import sqrt

import pytest
import torch

from iris.backbone import BackboneForwardOutput
from iris.cognitive import LayerSampler, ModuleInputRouter, ModuleInputRouterError
from iris.schemas import LayerSamplerConfig


def test_zero_queries_route_masked_mean_to_every_module() -> None:
    router = ModuleInputRouter(state_dim=3, num_modules=4)
    context = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [100.0, 200.0, 300.0]],
            [[7.0, 8.0, 9.0], [11.0, 12.0, 13.0], [15.0, 16.0, 17.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.int16)

    output = router(context, mask)

    assert output.module_inputs.shape == (2, 4, 3)
    assert output.attention_weights.shape == (2, 4, 3)
    expected_attention = torch.tensor(
        [
            [[0.5, 0.5, 0.0]] * 4,
            [[0.0, 1.0, 0.0]] * 4,
        ]
    )
    expected_inputs = torch.tensor(
        [
            [[2.0, 3.0, 4.0]] * 4,
            [[11.0, 12.0, 13.0]] * 4,
        ]
    )
    torch.testing.assert_close(output.attention_weights, expected_attention)
    torch.testing.assert_close(output.module_inputs, expected_inputs)


def test_distinct_queries_use_exact_scaled_soft_attention() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=3)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.75]])
    with torch.no_grad():
        router.module_queries.copy_(queries)
    context = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    output = router(context, mask)

    expected_scores = torch.einsum("btd,md->bmt", context[:, :3], queries) / sqrt(2)
    expected_attention = torch.softmax(expected_scores, dim=-1)
    expected_inputs = torch.einsum("bmt,btd->bmd", expected_attention, context[:, :3])
    torch.testing.assert_close(
        output.attention_weights,
        torch.cat((expected_attention, torch.zeros(1, 3, 1)), dim=-1),
    )
    torch.testing.assert_close(output.module_inputs, expected_inputs)
    assert not torch.equal(output.attention_weights[:, 0], output.attention_weights[:, 1])
    assert bool((output.attention_weights[:, :, :3] > 0).all())


def test_masked_padding_values_cannot_change_routing() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)
    with torch.no_grad():
        router.module_queries.copy_(torch.tensor([[0.75, -0.25], [-0.5, 1.0]]))
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

    original = router(context, mask)
    perturbed = router(changed_padding, mask)

    torch.testing.assert_close(perturbed.module_inputs, original.module_inputs)
    torch.testing.assert_close(perturbed.attention_weights, original.attention_weights)
    expanded_mask = mask.unsqueeze(1).expand_as(perturbed.attention_weights)
    assert torch.count_nonzero(perturbed.attention_weights.masked_select(~expanded_mask)) == 0


def test_backward_reaches_queries_and_only_unmasked_context_positions() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=3)
    with torch.no_grad():
        router.module_queries.copy_(torch.tensor([[0.5, -1.0], [1.0, 0.25], [-0.75, 0.5]]))
    context = torch.tensor(
        [[[1.0, 2.0], [3.0, -1.0], [8.0, 9.0], [-2.0, 4.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)

    output = router(context, mask)
    output.module_inputs.square().sum().backward()

    assert router.module_queries.grad is not None
    assert torch.isfinite(router.module_queries.grad).all()
    assert torch.count_nonzero(router.module_queries.grad) > 0
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert torch.count_nonzero(context.grad[mask]) > 0
    assert torch.count_nonzero(context.grad[~mask]) == 0


def test_materializes_inference_context_for_trainable_router_backward() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)
    with torch.no_grad():
        router.module_queries.copy_(torch.tensor([[0.25, -0.75], [1.0, 0.5]]))
    with torch.inference_mode():
        context = torch.tensor([[[1.0, 2.0], [4.0, -1.0], [2.0, 3.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    output = router(context, mask)
    output.module_inputs.square().sum().backward()

    assert output.module_inputs.is_inference() is False
    assert output.module_inputs.requires_grad
    assert router.module_queries.grad is not None
    assert torch.isfinite(router.module_queries.grad).all()
    assert torch.count_nonzero(router.module_queries.grad) > 0


def test_backward_reaches_layer_sampler_without_unfreezing_backbone_states() -> None:
    torch.manual_seed(17)
    sampler = LayerSampler(
        LayerSamplerConfig(layer_fractions=(0.25, 0.5, 0.75, 1.0)),
        backbone_hidden_size=3,
        state_dim=2,
    )
    router = ModuleInputRouter(state_dim=2, num_modules=3)
    with torch.no_grad():
        router.module_queries.normal_()
    with torch.inference_mode():
        hidden_states = tuple(torch.randn(1, 3, 3) for _ in range(5))
        backbone_output = BackboneForwardOutput(
            logits=torch.zeros(1, 3, 7),
            hidden_states=hidden_states,
            attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
        )

    sampled = sampler(backbone_output)
    routed = router(sampled.context_features, sampled.attention_mask)
    routed.module_inputs.square().sum().backward()

    assert router.module_queries.grad is not None
    assert torch.isfinite(router.module_queries.grad).all()
    assert sampler.mixing_logits.grad is not None
    assert torch.isfinite(sampler.mixing_logits.grad).all()
    for projection in sampler.projections:
        assert isinstance(projection, torch.nn.Linear)
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
    assert all(state.grad is None for state in hidden_states)


def test_supports_trainable_parameters_under_global_inference_mode() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)

    with torch.inference_mode():
        context = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones((1, 2), dtype=torch.long)
        output = router(context, mask)

    assert output.module_inputs.is_inference()
    assert output.attention_weights.is_inference()
    assert output.module_inputs.requires_grad is False
    assert torch.isfinite(output.module_inputs).all()


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
    router = ModuleInputRouter(state_dim=3, num_modules=2)
    mask = torch.ones(context.shape[:2], dtype=torch.long)

    with pytest.raises(ModuleInputRouterError):
        router(context, mask)


def test_rejects_nontensor_context() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)

    with pytest.raises(ModuleInputRouterError):
        router([[1.0, 2.0]], torch.ones((1, 1), dtype=torch.long))


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
    router = ModuleInputRouter(state_dim=2, num_modules=2)

    with pytest.raises(ModuleInputRouterError):
        router(torch.ones((2, 3, 2)), mask)


def test_rejects_nontensor_attention_mask() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)

    with pytest.raises(ModuleInputRouterError):
        router(torch.ones((1, 2, 2)), [[1, 1]])


def test_rejects_parameter_dtype_mismatch() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2).to(dtype=torch.float64)

    with pytest.raises(ModuleInputRouterError, match="parameters must explicitly match"):
        router(torch.ones((1, 2, 2)), torch.ones((1, 2), dtype=torch.long))


def test_rejects_nonfinite_module_query() -> None:
    router = ModuleInputRouter(state_dim=2, num_modules=2)
    with torch.no_grad():
        router.module_queries[0, 0] = torch.nan

    with pytest.raises(ModuleInputRouterError, match="queries contain NaN or Inf"):
        router(torch.ones((1, 2, 2)), torch.ones((1, 2), dtype=torch.long))


@pytest.mark.parametrize(
    ("state_dim", "num_modules"),
    [(0, 2), (-1, 2), (True, 2), (2.5, 2), (2, 0), (2, -1), (2, True), (2, 2.5)],
)
def test_rejects_invalid_dimensions(state_dim: object, num_modules: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ModuleInputRouter(state_dim=state_dim, num_modules=num_modules)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "large_value", "query_value"),
    [(torch.float16, 60_000.0, 1.0), (torch.bfloat16, 2e38, 0.3)],
    ids=["float16", "bfloat16"],
)
def test_low_precision_routing_accumulates_safely_in_float32(
    dtype: torch.dtype, large_value: float, query_value: float
) -> None:
    router = ModuleInputRouter(state_dim=8, num_modules=3).to(dtype=dtype)
    with torch.no_grad():
        router.module_queries.fill_(query_value)
    context = torch.full((1, 8, 8), large_value, dtype=dtype)
    mask = torch.ones((1, 8), dtype=torch.long)

    output = router(context, mask)

    assert output.module_inputs.dtype == dtype
    assert output.attention_weights.dtype == torch.float32
    assert torch.isfinite(output.module_inputs).all()
    assert torch.isfinite(output.attention_weights).all()
    torch.testing.assert_close(output.module_inputs, context[:, :1].expand(1, 3, 8))
    torch.testing.assert_close(output.attention_weights, torch.full((1, 3, 8), 0.125))


def test_external_autocast_cannot_lower_routing_precision() -> None:
    router = ModuleInputRouter(state_dim=4, num_modules=2)
    with torch.no_grad():
        router.module_queries.copy_(
            torch.tensor([[0.5, -0.25, 0.75, 1.0], [-1.0, 0.5, 0.25, -0.75]])
        )
    context = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.5], [9.0, 8.0, 7.0, 6.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    reference = router(context, mask)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = router(context, mask)

    assert actual.module_inputs.dtype == torch.float32
    assert actual.attention_weights.dtype == torch.float32
    torch.testing.assert_close(actual.module_inputs, reference.module_inputs)
    torch.testing.assert_close(actual.attention_weights, reference.attention_weights)
