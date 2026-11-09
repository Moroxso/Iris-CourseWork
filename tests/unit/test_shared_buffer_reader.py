"""Tests for all-entry learned broadcast reads in arm C."""

from dataclasses import replace

import pytest
import torch

from iris.shared_buffer import SharedBufferReader, SharedBufferReaderError, SharedBufferState


def _empty(*, dtype: torch.dtype = torch.float32) -> SharedBufferState:
    return SharedBufferState.empty(
        batch_size=2,
        state_dim=4,
        num_modules=3,
        cycles=3,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def _state(*, completed_cycles: int = 2, dtype: torch.dtype = torch.float32) -> SharedBufferState:
    entry_count = completed_cycles * 3
    entries = torch.arange(2 * entry_count * 4, dtype=dtype).reshape(2, entry_count, 4)
    positions = torch.arange(entry_count, dtype=torch.int64)
    return SharedBufferState(
        entries=entries,
        source_module=positions.remainder(3).expand(2, -1).clone(),
        write_cycle=(positions.div(3, rounding_mode="floor") + 1).expand(2, -1).clone(),
        state_dim=4,
        num_modules=3,
        cycles=3,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"state_dim": 0, "num_modules": 3, "cycles": 3},
        {"state_dim": 4, "num_modules": True, "cycles": 3},
        {"state_dim": 4, "num_modules": 3, "cycles": -1},
    ],
)
def test_reader_rejects_invalid_dimensions(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SharedBufferReader(**arguments)  # type: ignore[arg-type]


def test_empty_buffer_returns_exact_zero_reads_and_zero_width_attention() -> None:
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    output = reader(_empty())

    assert output.workspace_reads.shape == (2, 3, 4)
    assert output.attention_weights.shape == (2, 3, 0)
    assert output.workspace_reads.dtype == torch.float32
    assert output.attention_weights.dtype == torch.float32
    assert torch.equal(output.workspace_reads, torch.zeros_like(output.workspace_reads))
    assert output.attention_weights.numel() == 0


def test_zero_queries_read_uniform_mean_over_every_existing_entry() -> None:
    state = _state()
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    output = reader(state)

    expected_weights = torch.full((2, 3, 6), 1.0 / 6.0)
    expected_reads = state.entries.mean(dim=1, keepdim=True).expand(-1, 3, -1)
    assert torch.allclose(output.attention_weights, expected_weights)
    assert torch.allclose(output.workspace_reads, expected_reads)


def test_every_module_query_has_soft_access_to_every_entry() -> None:
    state = _state(completed_cycles=3)
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    with torch.no_grad():
        reader.read_queries.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    output = reader(state)

    assert output.attention_weights.shape == (2, 3, 9)
    assert bool((output.attention_weights > 0).all())
    assert torch.allclose(
        output.attention_weights.sum(dim=-1),
        torch.ones(2, 3),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_reads_accumulate_attention_in_fp32(dtype: torch.dtype) -> None:
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3).to(dtype=dtype)
    output = reader(_state(dtype=dtype))
    assert output.workspace_reads.dtype == dtype
    assert output.attention_weights.dtype == torch.float32
    assert bool(torch.isfinite(output.workspace_reads).all())


def test_reader_preserves_autograd_to_entries_and_queries() -> None:
    base = _state()
    entries = base.entries.detach().clone().requires_grad_(True)
    state = replace(base, entries=entries)
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    with torch.no_grad():
        reader.read_queries.normal_(mean=0.0, std=0.1)

    output = reader(state)
    output.workspace_reads.square().mean().backward()

    assert entries.grad is not None
    assert bool(torch.isfinite(entries.grad).all())
    assert reader.read_queries.grad is not None
    assert bool(torch.isfinite(reader.read_queries.grad).all())


def test_reader_materializes_inference_entries_before_training_use() -> None:
    with torch.inference_mode():
        state = _state()
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    output = reader(state)
    assert state.entries.is_inference()
    assert not output.workspace_reads.is_inference()
    assert output.workspace_reads.requires_grad


def test_reader_rejects_malformed_state_before_attention() -> None:
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    malformed = replace(_state(), source_module=torch.zeros(2, 6, dtype=torch.int64))
    with pytest.raises(SharedBufferReaderError, match="canonical"):
        reader(malformed)


def test_reader_rejects_parameter_dtype_mismatch_and_nonfinite_query() -> None:
    state = _state()
    mismatched = SharedBufferReader(state_dim=4, num_modules=3, cycles=3).to(dtype=torch.float64)
    with pytest.raises(SharedBufferReaderError, match="dtype and device"):
        mismatched(state)

    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    with torch.no_grad():
        reader.read_queries[0, 0] = torch.nan
    with pytest.raises(SharedBufferReaderError, match="NaN or Inf"):
        reader(state)


def test_reader_rejects_nonfinite_entries() -> None:
    state = _state()
    entries = state.entries.clone()
    entries[0, 0, 0] = torch.inf
    with pytest.raises(SharedBufferReaderError, match="NaN or Inf"):
        SharedBufferReader(state_dim=4, num_modules=3, cycles=3)(replace(state, entries=entries))
