"""Tests for token-matched final-read projection in arm C."""

from dataclasses import replace

import pytest
import torch

from iris.shared_buffer import SharedBufferLatentBridge, SharedBufferLatentBridgeError
from iris.workspace import WorkspaceReaderOutput


def _read(
    *,
    dtype: torch.dtype = torch.float32,
    entry_count: int = 9,
    requires_grad: bool = False,
) -> WorkspaceReaderOutput:
    torch.manual_seed(314159)
    reads = torch.randn(2, 3, 4, dtype=dtype).requires_grad_(requires_grad)
    attention_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    attention = torch.full(
        (2, 3, entry_count),
        1.0 / entry_count,
        dtype=attention_dtype,
    )
    return WorkspaceReaderOutput(
        workspace_reads=reads,
        attention_weights=attention,
    )


def _bridge(*, dtype: torch.dtype = torch.float32) -> SharedBufferLatentBridge:
    return SharedBufferLatentBridge(
        state_dim=4,
        backbone_hidden_size=6,
        num_modules=3,
        cycles=3,
    ).to(dtype=dtype)


@pytest.mark.parametrize(
    "arguments",
    [
        {"state_dim": 0, "backbone_hidden_size": 6, "num_modules": 3, "cycles": 3},
        {"state_dim": 4, "backbone_hidden_size": 6, "num_modules": True, "cycles": 3},
        {"state_dim": 4, "backbone_hidden_size": 6, "num_modules": 3, "cycles": -1},
    ],
)
def test_bridge_rejects_invalid_dimensions(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SharedBufferLatentBridge(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "prompt_mask",
    [
        torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.int64),
        torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
    ],
)
def test_bridge_emits_exactly_one_token_per_module_after_logical_prompt(
    prompt_mask: torch.Tensor,
) -> None:
    output = _bridge()(_read(), prompt_mask)
    logical_lengths = prompt_mask.to(dtype=torch.int64).sum(dim=-1, keepdim=True)
    expected_positions = logical_lengths + torch.arange(3)

    assert output.workspace_embeddings.shape == (2, 3, 6)
    assert output.workspace_attention_mask.shape == (2, 3)
    assert bool(output.workspace_attention_mask.all())
    assert torch.equal(output.workspace_position_ids, expected_positions)
    assert bool(torch.isfinite(output.workspace_embeddings).all())


def test_bridge_rejects_a_read_that_does_not_cover_all_m_times_c_entries() -> None:
    with pytest.raises(SharedBufferLatentBridgeError, match="every configured"):
        _bridge()(_read(entry_count=6), torch.ones(2, 2, dtype=torch.bool))


def test_bridge_rejects_invalid_final_attention_values() -> None:
    source = _read()
    bad_sum = replace(source, attention_weights=source.attention_weights * 0.5)
    with pytest.raises(SharedBufferLatentBridgeError, match="sum to one"):
        _bridge()(bad_sum, torch.ones(2, 2, dtype=torch.bool))
    negative = source.attention_weights.clone()
    negative[0, 0, 0] = -1.0
    with pytest.raises(SharedBufferLatentBridgeError, match="cover every"):
        _bridge()(
            replace(source, attention_weights=negative),
            torch.ones(2, 2, dtype=torch.bool),
        )
    nonfinite = source.attention_weights.clone()
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises(SharedBufferLatentBridgeError, match="cover every"):
        _bridge()(
            replace(source, attention_weights=nonfinite),
            torch.ones(2, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_bridge_requires_fp32_attention_and_preserves_storage_dtype(
    dtype: torch.dtype,
) -> None:
    output = _bridge(dtype=dtype)(
        _read(dtype=dtype),
        torch.ones(2, 2, dtype=torch.bool),
    )
    assert output.workspace_embeddings.dtype == dtype
    assert bool(torch.isfinite(output.workspace_embeddings).all())
    wrong_attention = _read(dtype=dtype)
    wrong_attention = replace(
        wrong_attention,
        attention_weights=wrong_attention.attention_weights.to(dtype=dtype),
    )
    with pytest.raises(SharedBufferLatentBridgeError, match="cover every"):
        _bridge(dtype=dtype)(wrong_attention, torch.ones(2, 2, dtype=torch.bool))


def test_bridge_preserves_gradients_to_final_reads_and_parameters() -> None:
    bridge = _bridge()
    final_read = _read(requires_grad=True)
    output = bridge(final_read, torch.ones(2, 2, dtype=torch.bool))
    weights = torch.linspace(0.1, 1.0, output.workspace_embeddings.numel()).reshape_as(
        output.workspace_embeddings
    )
    (output.workspace_embeddings * weights).sum().backward()

    assert final_read.workspace_reads.grad is not None
    assert bool(torch.isfinite(final_read.workspace_reads.grad).all())
    assert all(parameter.grad is not None for parameter in bridge.parameters())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in bridge.parameters()
    )


def test_bridge_materializes_inference_reads_before_trainable_projection() -> None:
    with torch.inference_mode():
        final_read = _read()
    output = _bridge()(final_read, torch.ones(2, 2, dtype=torch.bool))
    assert final_read.workspace_reads.is_inference()
    assert not output.workspace_embeddings.is_inference()
    assert output.workspace_embeddings.requires_grad


def test_bridge_remains_finite_for_large_finite_reads_and_gradients() -> None:
    bridge = _bridge()
    source = _read(requires_grad=True)
    large_reads = source.workspace_reads.detach().clone().fill_(1e20).requires_grad_(True)
    source = replace(source, workspace_reads=large_reads)
    output = bridge(source, torch.ones(2, 2, dtype=torch.bool))
    output.workspace_embeddings.sum().backward()
    assert bool(torch.isfinite(output.workspace_embeddings).all())
    assert large_reads.grad is not None
    assert bool(torch.isfinite(large_reads.grad).all())


def test_bridge_rejects_bad_read_mask_and_parameter_boundaries() -> None:
    with pytest.raises(SharedBufferLatentBridgeError, match="exact"):
        _bridge()(object(), torch.ones(2, 2, dtype=torch.bool))
    malformed = replace(_read(), workspace_reads=torch.zeros(2, 2, 4))
    with pytest.raises(SharedBufferLatentBridgeError, match="shape"):
        _bridge()(malformed, torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(SharedBufferLatentBridgeError, match="contiguous"):
        _bridge()(_read(), torch.tensor([[1, 0, 1], [1, 1, 1]]))
    mismatched = _bridge(dtype=torch.float64)
    with pytest.raises(SharedBufferLatentBridgeError, match="parameters"):
        mismatched(_read(), torch.ones(2, 2, dtype=torch.bool))
    bridge = _bridge()
    with torch.no_grad():
        bridge.shared_buffer_type_embedding[0] = torch.inf
    with pytest.raises(SharedBufferLatentBridgeError, match="NaN or Inf"):
        bridge(_read(), torch.ones(2, 2, dtype=torch.bool))
