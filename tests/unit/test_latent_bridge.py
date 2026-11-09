"""Contract tests for projecting workspace slots into a latent prefix."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch.nn import functional

from iris.workspace import LatentBridge, LatentBridgeError, WorkspaceState

CPU = torch.device("cpu")


def make_bridge(
    *,
    state_dim: int = 4,
    backbone_hidden_size: int = 6,
    num_slots: int = 3,
) -> LatentBridge:
    return LatentBridge(
        state_dim=state_dim,
        backbone_hidden_size=backbone_hidden_size,
        num_slots=num_slots,
    )


def make_workspace(
    *,
    batch_size: int = 2,
    num_slots: int = 3,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    occupancy: torch.Tensor | None = None,
    requires_grad: bool = False,
    stale_empty_content: bool = False,
) -> WorkspaceState:
    if occupancy is None:
        occupancy = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
            device=CPU,
        )[:batch_size, :num_slots]
    generator = torch.Generator(device=CPU).manual_seed(1009)
    slots = torch.randn(
        batch_size,
        num_slots,
        state_dim,
        dtype=dtype,
        device=CPU,
        generator=generator,
    )
    if not stale_empty_content:
        slots = slots * occupancy.unsqueeze(-1)
    slots.requires_grad_(requires_grad)
    source = torch.arange(num_slots, dtype=torch.int64).remainder(2).expand(batch_size, -1).clone()
    source = torch.where(occupancy, source, torch.full_like(source, -1))
    confidence = occupancy.to(dtype=dtype) * 0.6
    ttl = occupancy.to(dtype=dtype) * 2.0
    return WorkspaceState(
        slots=slots,
        age=occupancy.to(torch.int64),
        occupancy=occupancy,
        source_module=source,
        confidence=confidence,
        ttl=ttl,
        write_count=occupancy.to(torch.int64),
    )


def test_matches_projection_then_affine_layer_norm_then_type_embedding() -> None:
    bridge = make_bridge(state_dim=2, backbone_hidden_size=3, num_slots=2).to(dtype=torch.float64)
    with torch.no_grad():
        bridge.bridge_projection.weight.copy_(
            torch.tensor(
                [[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]],
                dtype=torch.float64,
            )
        )
        bridge.normalization.weight.copy_(torch.tensor([1.2, 0.5, 0.8], dtype=torch.float64))
        bridge.normalization.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
        bridge.workspace_type_embedding.copy_(
            torch.tensor([0.05, -0.15, 0.25], dtype=torch.float64)
        )
    workspace = make_workspace(
        batch_size=1,
        num_slots=2,
        state_dim=2,
        dtype=torch.float64,
        occupancy=torch.tensor([[True, False]]),
    )
    prompt_mask = torch.tensor([[False, True, True, True]], dtype=torch.bool)

    output = bridge(workspace, prompt_mask)
    canonical_slots = torch.where(
        workspace.occupancy.unsqueeze(-1),
        workspace.slots,
        torch.zeros_like(workspace.slots),
    )
    projected = functional.linear(canonical_slots, bridge.bridge_projection.weight)
    expected = (
        functional.layer_norm(
            projected,
            (3,),
            bridge.normalization.weight,
            bridge.normalization.bias,
            bridge.normalization.eps,
        )
        + bridge.workspace_type_embedding
    )

    torch.testing.assert_close(output.workspace_embeddings, expected, rtol=1e-12, atol=1e-12)
    assert output.workspace_attention_mask.tolist() == [[True, True]]
    assert output.workspace_position_ids.tolist() == [[3, 4]]
    assert bridge.bridge_projection.bias is None
    assert bridge.normalization.elementwise_affine


def test_all_slots_are_attended_and_positions_follow_logical_prompt_lengths() -> None:
    bridge = make_bridge(num_slots=3)
    workspace = make_workspace()
    prompt_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )

    output = bridge(workspace, prompt_mask)

    assert torch.equal(output.workspace_attention_mask, torch.ones(2, 3, dtype=torch.bool))
    assert output.workspace_position_ids.dtype == torch.int64
    assert output.workspace_position_ids.tolist() == [[3, 4, 5], [2, 3, 4]]


def test_left_and_right_padding_with_equal_logical_lengths_have_equal_positions() -> None:
    bridge = make_bridge()
    workspace = make_workspace()
    left = torch.tensor([[False, False, True, True], [False, True, True, True]])
    right = torch.tensor([[True, True, False, False], [True, True, True, False]])

    left_output = bridge(workspace, left)
    right_output = bridge(workspace, right)

    assert torch.equal(left_output.workspace_position_ids, right_output.workspace_position_ids)
    torch.testing.assert_close(left_output.workspace_embeddings, right_output.workspace_embeddings)


def test_binary_integer_prompt_mask_is_supported() -> None:
    output = make_bridge()(
        make_workspace(),
        torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.int64),
    )

    assert output.workspace_position_ids.tolist() == [[3, 4, 5], [2, 3, 4]]


def test_rejects_stale_unoccupied_content_at_post_decay_boundary() -> None:
    occupancy = torch.tensor([[True, False, True], [False, True, False]])
    bridge = make_bridge()
    canonical = make_workspace(occupancy=occupancy)
    stale = replace(
        canonical,
        slots=torch.where(
            occupancy.unsqueeze(-1),
            canonical.slots,
            torch.full_like(canonical.slots, 10_000.0),
        ),
    )
    prompt_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(LatentBridgeError, match="canonical slots"):
        bridge(stale, prompt_mask)
    assert bool(torch.isfinite(bridge(canonical, prompt_mask).workspace_embeddings).all())


def test_all_empty_workspace_produces_finite_shared_null_tokens() -> None:
    bridge = make_bridge()
    workspace = WorkspaceState.empty(
        batch_size=2,
        num_slots=3,
        state_dim=4,
        dtype=torch.float32,
        device=CPU,
    )

    output = bridge(workspace, torch.ones(2, 3, dtype=torch.bool))

    assert bool(torch.isfinite(output.workspace_embeddings).all())
    assert bool(output.workspace_attention_mask.all())
    torch.testing.assert_close(
        output.workspace_embeddings,
        bridge.workspace_type_embedding.expand(2, 3, -1),
    )


def test_backward_reaches_slots_and_every_trainable_bridge_parameter() -> None:
    bridge = make_bridge().to(dtype=torch.float64)
    workspace = make_workspace(dtype=torch.float64, requires_grad=True)
    output = bridge(workspace, torch.ones(2, 4, dtype=torch.bool))
    coefficients = torch.linspace(
        0.2,
        1.4,
        output.workspace_embeddings.numel(),
        dtype=torch.float64,
    ).reshape_as(output.workspace_embeddings)

    (output.workspace_embeddings * coefficients).sum().backward()

    assert workspace.slots.grad is not None
    assert bool((workspace.slots.grad[workspace.occupancy] != 0).any())
    assert torch.count_nonzero(workspace.slots.grad[~workspace.occupancy]) == 0
    for parameter in bridge.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0).any())


def test_forward_does_not_mutate_or_alias_inputs() -> None:
    bridge = make_bridge()
    workspace = make_workspace()
    prompt_mask = torch.tensor([[True, True, False], [False, True, True]])
    snapshots = {
        name: value.clone()
        for name in workspace.__dataclass_fields__
        if isinstance((value := getattr(workspace, name)), torch.Tensor)
    }
    mask_snapshot = prompt_mask.clone()

    output = bridge(workspace, prompt_mask)

    for name, expected in snapshots.items():
        assert torch.equal(getattr(workspace, name), expected)
    assert torch.equal(prompt_mask, mask_snapshot)
    assert output.workspace_embeddings.data_ptr() != workspace.slots.data_ptr()
    assert output.workspace_attention_mask.data_ptr() != prompt_mask.data_ptr()


def test_materializes_inference_workspace_for_trainable_backward() -> None:
    bridge = make_bridge()
    with torch.inference_mode():
        workspace = make_workspace()
        prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    assert workspace.slots.is_inference()

    output = bridge(workspace, prompt_mask)
    assert not output.workspace_embeddings.is_inference()
    output.workspace_embeddings.square().sum().backward()

    assert bridge.bridge_projection.weight.grad is not None
    assert bridge.normalization.weight.grad is not None
    assert bridge.workspace_type_embedding.grad is not None


def test_global_inference_mode_returns_valid_inference_tensors() -> None:
    bridge = make_bridge()
    with torch.inference_mode():
        output = bridge(make_workspace(), torch.ones(2, 4, dtype=torch.bool))

    for tensor in (
        output.workspace_embeddings,
        output.workspace_attention_mask,
        output.workspace_position_ids,
    ):
        assert tensor.is_inference()
        assert not tensor.requires_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_preserves_supported_dtype_and_produces_finite_outputs(dtype: torch.dtype) -> None:
    bridge = make_bridge().to(dtype=dtype)
    output = bridge(make_workspace(dtype=dtype), torch.ones(2, 4, dtype=torch.bool))

    assert output.workspace_embeddings.dtype == dtype
    assert bool(torch.isfinite(output.workspace_embeddings).all())
    assert output.workspace_attention_mask.dtype == torch.bool
    assert output.workspace_position_ids.dtype == torch.int64


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.float16, 60_000.0), (torch.bfloat16, 1.0e30)],
)
def test_low_precision_huge_equal_inputs_have_stable_finite_layer_norm(
    dtype: torch.dtype,
    magnitude: float,
) -> None:
    bridge = make_bridge().to(dtype=dtype)
    workspace = make_workspace(dtype=dtype, occupancy=torch.ones(2, 3, dtype=torch.bool))
    workspace = replace(workspace, slots=torch.full_like(workspace.slots, magnitude))

    output = bridge(workspace, torch.ones(2, 4, dtype=torch.bool))

    assert output.workspace_embeddings.dtype == dtype
    assert bool(torch.isfinite(output.workspace_embeddings).all())


def test_cpu_autocast_cannot_change_bridge_contract_dtype() -> None:
    bridge = make_bridge()
    workspace = make_workspace()
    prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    baseline = bridge(workspace, prompt_mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = bridge(workspace, prompt_mask)

    assert output.workspace_embeddings.dtype == torch.float32
    assert bool(torch.isfinite(output.workspace_embeddings).all())
    torch.testing.assert_close(
        output.workspace_embeddings,
        baseline.workspace_embeddings,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.bfloat16, 1.0e35),
        (torch.float32, 1.0e35),
        (torch.float64, 1.0e200),
    ],
)
def test_extreme_constant_projection_has_finite_backward(
    dtype: torch.dtype,
    magnitude: float,
) -> None:
    bridge = make_bridge(state_dim=4, backbone_hidden_size=6, num_slots=1).to(dtype=dtype)
    with torch.no_grad():
        bridge.bridge_projection.weight.zero_()
        for output_index in range(6):
            bridge.bridge_projection.weight[output_index, output_index % 4] = 1
        bridge.normalization.weight.fill_(1)
        bridge.normalization.bias.zero_()
        bridge.workspace_type_embedding.zero_()
    workspace = make_workspace(
        batch_size=1,
        num_slots=1,
        state_dim=4,
        dtype=dtype,
        occupancy=torch.ones(1, 1, dtype=torch.bool),
        requires_grad=True,
    )
    workspace = replace(workspace, slots=torch.full_like(workspace.slots, magnitude))
    workspace.slots.requires_grad_(True)

    output = bridge(workspace, torch.ones(1, 2, dtype=torch.bool))
    coefficients = torch.arange(1, 7, dtype=dtype).reshape(1, 1, 6)
    (output.workspace_embeddings * coefficients).sum().backward()

    assert workspace.slots.grad is not None
    assert bool(torch.isfinite(workspace.slots.grad).all())
    assert bridge.bridge_projection.weight.grad is not None
    assert bool(torch.isfinite(bridge.bridge_projection.weight.grad).all())
    assert bridge.normalization.weight.grad is not None
    assert bool(torch.isfinite(bridge.normalization.weight.grad).all())


def test_stable_layer_norm_handles_mixed_sign_float32_limits() -> None:
    bridge = make_bridge(state_dim=4, backbone_hidden_size=4, num_slots=1)
    projected = torch.tensor([[[3.0e38, -3.0e38, -3.0e38, -3.0e38]]])

    normalized = bridge._stable_layer_norm(
        projected,
        weight=torch.ones(4),
        bias=torch.zeros(4),
    )

    assert bool(torch.isfinite(normalized).all())
    torch.testing.assert_close(
        normalized,
        torch.tensor(
            [[[3.0**0.5, -((1.0 / 3.0) ** 0.5), -((1.0 / 3.0) ** 0.5), -((1.0 / 3.0) ** 0.5)]]]
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_type_embedding_initialization_uses_hidden_width_scaled_normal() -> None:
    torch.manual_seed(1701)
    bridge = make_bridge(backbone_hidden_size=4096)

    initialized_embedding = bridge.workspace_type_embedding.detach()
    assert abs(float(initialized_embedding.mean())) < 2.0e-3
    assert float(initialized_embedding.std()) == pytest.approx(
        4096**-0.5,
        rel=0.04,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_dim": 0},
        {"state_dim": True},
        {"state_dim": 2.5},
        {"backbone_hidden_size": 0},
        {"backbone_hidden_size": True},
        {"num_slots": -1},
        {"num_slots": 2.5},
    ],
)
def test_rejects_invalid_constructor_dimensions(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_bridge(**kwargs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: cast(Any, "not a workspace"),
        lambda state: replace(state, slots=state.slots[:, :, :-1]),
        lambda state: replace(state, slots=state.slots[:, :2]),
        lambda state: replace(state, slots=state.slots.to(torch.int64)),
        lambda state: replace(state, slots=torch.full_like(state.slots, float("nan"))),
        lambda state: replace(state, occupancy=state.occupancy.to(torch.int64)),
        lambda state: replace(state, age=state.age[:, :2]),
        lambda state: replace(state, source_module=state.source_module.to(torch.int32)),
        lambda state: replace(state, confidence=state.confidence.to(torch.float64)),
        lambda state: replace(state, ttl=torch.full_like(state.ttl, float("inf"))),
        lambda state: replace(state, write_count=state.write_count.to(torch.int32)),
    ],
)
def test_rejects_malformed_workspace(mutator: Any) -> None:
    with pytest.raises(LatentBridgeError):
        make_bridge()(mutator(make_workspace()), torch.ones(2, 4, dtype=torch.bool))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: replace(state, age=torch.full_like(state.age, -1)),
        lambda state: replace(state, write_count=torch.full_like(state.write_count, -1)),
        lambda state: replace(state, source_module=torch.full_like(state.source_module, -2)),
        lambda state: replace(
            state,
            source_module=torch.where(
                state.occupancy,
                torch.full_like(state.source_module, -1),
                state.source_module,
            ),
        ),
        lambda state: replace(state, confidence=torch.full_like(state.confidence, float("nan"))),
        lambda state: replace(state, confidence=torch.full_like(state.confidence, 1.1)),
        lambda state: replace(state, ttl=torch.full_like(state.ttl, -0.1)),
        lambda state: replace(
            state,
            ttl=torch.where(~state.occupancy, torch.ones_like(state.ttl), state.ttl),
        ),
        lambda state: replace(
            state,
            age=torch.where(~state.occupancy, torch.ones_like(state.age), state.age),
        ),
        lambda state: replace(
            state,
            confidence=torch.where(
                ~state.occupancy,
                torch.ones_like(state.confidence),
                state.confidence,
            ),
        ),
        lambda state: replace(
            state,
            write_count=torch.where(
                ~state.occupancy,
                torch.ones_like(state.write_count),
                state.write_count,
            ),
        ),
    ],
)
def test_rejects_invalid_workspace_metadata_values(mutator: Any) -> None:
    with pytest.raises(LatentBridgeError):
        make_bridge()(mutator(make_workspace()), torch.ones(2, 4, dtype=torch.bool))


@pytest.mark.parametrize(
    "prompt_mask",
    [
        cast(Any, "not a tensor"),
        torch.ones(2, 4),
        torch.tensor([[1, 1, 2, 0], [1, 1, 1, 1]], dtype=torch.int64),
        torch.ones(2, 4, 1, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(2, 0, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    ],
)
def test_rejects_malformed_prompt_attention_mask(prompt_mask: Any) -> None:
    with pytest.raises(LatentBridgeError):
        make_bridge()(make_workspace(), prompt_mask)


def test_rejects_prompt_mask_with_internal_holes() -> None:
    with pytest.raises(LatentBridgeError, match="contiguous"):
        make_bridge()(
            make_workspace(),
            torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool),
        )


def test_rejects_parameter_dtype_and_finiteness_boundary_violations() -> None:
    bridge = make_bridge().to(dtype=torch.float64)
    with pytest.raises(LatentBridgeError, match="parameters must share"):
        bridge(make_workspace(), torch.ones(2, 4, dtype=torch.bool))

    bridge = make_bridge()
    with torch.no_grad():
        bridge.bridge_projection.weight[0, 0] = float("nan")
    with pytest.raises(LatentBridgeError, match=r"parameter .* NaN or Inf"):
        bridge(make_workspace(), torch.ones(2, 4, dtype=torch.bool))


def test_output_has_exact_shape_dtype_device_and_value_invariants() -> None:
    bridge = make_bridge()
    output = bridge(make_workspace(), torch.ones(2, 4, dtype=torch.bool))

    assert output.workspace_embeddings.shape == (2, 3, 6)
    assert output.workspace_attention_mask.shape == (2, 3)
    assert output.workspace_position_ids.shape == (2, 3)
    assert output.workspace_embeddings.device == CPU
    assert output.workspace_attention_mask.device == CPU
    assert output.workspace_position_ids.device == CPU
    assert bool(torch.isfinite(output.workspace_embeddings).all())
    assert bool(output.workspace_attention_mask.all())
    assert bool((output.workspace_position_ids >= 0).all())
