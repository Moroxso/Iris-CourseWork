"""Contract tests for the no-workspace module-to-LLM latent bridge."""

from typing import Any

import pytest
import torch

import iris.cognitive as cognitive
from iris.cognitive import (
    ModuleLatentBridge,
    ModuleLatentBridgeError,
    ModuleLatentBridgeOutput,
)


def bridge() -> ModuleLatentBridge:
    return ModuleLatentBridge(
        state_dim=4,
        backbone_hidden_size=6,
        num_modules=3,
    )


def test_module_latent_bridge_returns_one_attended_token_per_module() -> None:
    module_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.int64)

    output = bridge()(module_states, prompt_mask)

    assert type(output) is ModuleLatentBridgeOutput
    assert output.module_embeddings.shape == (2, 3, 6)
    assert output.module_embeddings.dtype == torch.float32
    assert bool(torch.isfinite(output.module_embeddings).all())
    assert torch.equal(output.module_attention_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.equal(
        output.module_position_ids,
        torch.tensor([[2, 3, 4], [3, 4, 5]], dtype=torch.int64),
    )


def test_module_latent_bridge_preserves_gradients_to_states_and_parameters() -> None:
    component = bridge().train()
    module_states = torch.randn((1, 3, 4), dtype=torch.float32, requires_grad=True)

    output = component(module_states, torch.ones((1, 2), dtype=torch.int64))
    output.module_embeddings.square().mean().backward()

    assert module_states.grad is not None
    assert bool(torch.isfinite(module_states.grad).all())
    assert component.bridge_projection.weight.grad is not None
    assert component.module_type_embedding.grad is not None
    assert component.normalization.weight.grad is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_dim", 0),
        ("backbone_hidden_size", True),
        ("num_modules", -1),
    ],
)
def test_module_latent_bridge_rejects_invalid_constructor_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "state_dim": 4,
        "backbone_hidden_size": 6,
        "num_modules": 3,
    }
    values[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        ModuleLatentBridge(**values)


@pytest.mark.parametrize(
    "states",
    [
        torch.zeros((3, 4), dtype=torch.float32),
        torch.zeros((0, 3, 4), dtype=torch.float32),
        torch.zeros((1, 2, 4), dtype=torch.float32),
        torch.zeros((1, 3, 5), dtype=torch.float32),
        torch.zeros((1, 3, 4), dtype=torch.int64),
        torch.full((1, 3, 4), float("inf")),
    ],
)
def test_module_latent_bridge_rejects_invalid_module_states(states: torch.Tensor) -> None:
    with pytest.raises(ModuleLatentBridgeError):
        bridge()(states, torch.ones((1, 2), dtype=torch.int64))


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones((2,), dtype=torch.int64),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.zeros((1, 2), dtype=torch.int64),
        torch.tensor([[1, 0, 1]], dtype=torch.int64),
    ],
)
def test_module_latent_bridge_rejects_invalid_prompt_masks(mask: torch.Tensor) -> None:
    with pytest.raises(ModuleLatentBridgeError):
        bridge()(torch.zeros((1, 3, 4), dtype=torch.float32), mask)


def test_module_latent_bridge_rejects_parameter_dtype_mismatch() -> None:
    component = bridge().to(dtype=torch.float64)
    with pytest.raises(ModuleLatentBridgeError, match="share input dtype"):
        component(
            torch.zeros((1, 3, 4), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.int64),
        )


def test_module_latent_bridge_stays_finite_at_float32_limits() -> None:
    component = bridge()
    maximum = torch.finfo(torch.float32).max
    states = torch.tensor(
        [[[maximum, -maximum, maximum, -maximum]] * 3],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = component(states, torch.ones((1, 1), dtype=torch.int64))
    output.module_embeddings.sum().backward()

    assert bool(torch.isfinite(output.module_embeddings).all())
    assert states.grad is not None
    assert bool(torch.isfinite(states.grad).all())


def test_module_latent_bridge_materializes_inference_states_for_training() -> None:
    with torch.inference_mode():
        states = torch.ones((1, 3, 4), dtype=torch.float32)
    output = bridge()(states, torch.ones((1, 2), dtype=torch.int64))

    assert not output.module_embeddings.is_inference()
    assert output.module_embeddings.grad_fn is not None


def test_module_latent_bridge_api_is_public() -> None:
    assert cognitive.ModuleLatentBridge is ModuleLatentBridge
    assert cognitive.ModuleLatentBridgeOutput is ModuleLatentBridgeOutput
