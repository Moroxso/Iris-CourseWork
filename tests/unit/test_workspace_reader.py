"""Contract tests for independent global-workspace reads."""

from dataclasses import replace
from math import sqrt
from typing import cast

import pytest
import torch

from iris.workspace import WorkspaceReader, WorkspaceReaderError, WorkspaceState


def state_from_slots(slots: torch.Tensor, occupancy: torch.Tensor) -> WorkspaceState:
    state = WorkspaceState.empty(
        batch_size=slots.shape[0],
        num_slots=slots.shape[1],
        state_dim=slots.shape[2],
        dtype=slots.dtype,
        device=slots.device,
    )
    return replace(state, slots=slots, occupancy=occupancy)


def test_empty_workspace_returns_exact_zero_reads_and_attention() -> None:
    state = WorkspaceState.empty(
        batch_size=2,
        num_slots=8,
        state_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    reader = WorkspaceReader(state_dim=4, num_modules=8, num_slots=8)

    output = reader(state)

    assert output.workspace_reads.shape == (2, 8, 4)
    assert output.attention_weights.shape == (2, 8, 8)
    assert torch.count_nonzero(output.workspace_reads) == 0
    assert torch.count_nonzero(output.attention_weights) == 0
    assert torch.isfinite(output.workspace_reads).all()
    assert torch.isfinite(output.attention_weights).all()


def test_reader_preserves_continuous_fractional_ttl_contract() -> None:
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    fractional_ttl = torch.tensor([[0.0, 1.25, 2.75]])
    state = replace(
        state,
        occupancy=torch.tensor([[False, True, True]]),
        ttl=fractional_ttl,
    )
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)

    reader(state)

    torch.testing.assert_close(state.ttl, fractional_ttl, rtol=0, atol=0)


def test_single_occupied_slot_is_read_exactly_by_every_module() -> None:
    slots = torch.tensor([[[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]]])
    occupancy = torch.tensor([[0, 1, 0]], dtype=torch.bool)
    state = state_from_slots(slots, occupancy)
    reader = WorkspaceReader(state_dim=2, num_modules=3, num_slots=3)

    output = reader(state)

    torch.testing.assert_close(output.workspace_reads, slots[:, 1:2].expand(1, 3, 2))
    torch.testing.assert_close(
        output.attention_weights,
        torch.tensor([[[0.0, 1.0, 0.0]] * 3]),
    )


def test_distinct_queries_use_exact_scaled_attention_over_occupied_slots() -> None:
    slots = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [100.0, 100.0]]])
    occupancy = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    state = state_from_slots(slots, occupancy)
    reader = WorkspaceReader(state_dim=2, num_modules=3, num_slots=4)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.75]])
    with torch.no_grad():
        reader.read_queries.copy_(queries)

    output = reader(state)

    expected_scores = torch.einsum("bkd,md->bmk", slots[:, :3], queries) / sqrt(2)
    expected_attention = torch.softmax(expected_scores, dim=-1)
    expected_reads = torch.einsum("bmk,bkd->bmd", expected_attention, slots[:, :3])
    torch.testing.assert_close(
        output.attention_weights,
        torch.cat((expected_attention, torch.zeros(1, 3, 1)), dim=-1),
    )
    torch.testing.assert_close(output.workspace_reads, expected_reads)
    assert not torch.equal(output.attention_weights[:, 0], output.attention_weights[:, 1])


def test_mixed_empty_and_nonempty_batch_has_zero_or_normalized_attention() -> None:
    slots = torch.tensor(
        [
            [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]],
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
        ]
    )
    occupancy = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.bool)
    state = state_from_slots(slots, occupancy)
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)

    output = reader(state)

    assert torch.count_nonzero(output.workspace_reads[0]) == 0
    assert torch.count_nonzero(output.attention_weights[0]) == 0
    torch.testing.assert_close(output.attention_weights[1].sum(dim=-1), torch.ones(2))
    assert torch.count_nonzero(output.attention_weights[1, :, 2]) == 0
    assert torch.isfinite(output.workspace_reads).all()
    assert torch.isfinite(output.attention_weights).all()


def test_stale_unoccupied_content_cannot_change_reads() -> None:
    occupancy = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    base_slots = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])
    stale_slots = base_slots.clone()
    stale_slots[:, 1] = torch.tensor([10_000.0, -10_000.0])
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)
    with torch.no_grad():
        reader.read_queries.copy_(torch.tensor([[0.5, -1.0], [-0.25, 0.75]]))

    base = reader(state_from_slots(base_slots, occupancy))
    stale = reader(state_from_slots(stale_slots, occupancy))

    torch.testing.assert_close(stale.workspace_reads, base.workspace_reads)
    torch.testing.assert_close(stale.attention_weights, base.attention_weights)


def test_backward_reaches_queries_and_only_occupied_slots() -> None:
    slots = torch.tensor([[[1.0, 2.0], [8.0, 9.0], [3.0, -1.0]]], requires_grad=True)
    occupancy = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)
    with torch.no_grad():
        reader.read_queries.copy_(torch.tensor([[0.5, -1.0], [1.0, 0.25]]))

    output = reader(state_from_slots(slots, occupancy))
    output.workspace_reads.square().sum().backward()

    assert reader.read_queries.grad is not None
    assert torch.isfinite(reader.read_queries.grad).all()
    assert torch.count_nonzero(reader.read_queries.grad) > 0
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert torch.count_nonzero(slots.grad[occupancy]) > 0
    assert torch.count_nonzero(slots.grad[~occupancy]) == 0


def test_materializes_inference_slots_for_trainable_reader_backward() -> None:
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)
    with torch.no_grad():
        reader.read_queries.copy_(torch.tensor([[0.25, -0.75], [1.0, 0.5]]))
    with torch.inference_mode():
        slots = torch.tensor([[[1.0, 2.0], [4.0, -1.0], [2.0, 3.0]]])
        occupancy = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        state = state_from_slots(slots, occupancy)

    output = reader(state)
    output.workspace_reads.square().sum().backward()

    assert output.workspace_reads.is_inference() is False
    assert output.workspace_reads.requires_grad
    assert reader.read_queries.grad is not None
    assert torch.isfinite(reader.read_queries.grad).all()


def test_supports_trainable_parameters_under_global_inference_mode() -> None:
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=2)

    with torch.inference_mode():
        state = state_from_slots(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[1, 1]], dtype=torch.bool),
        )
        output = reader(state)

    assert output.workspace_reads.is_inference()
    assert output.attention_weights.is_inference()
    assert not output.workspace_reads.requires_grad


@pytest.mark.parametrize(
    ("state_dim", "num_modules", "num_slots"),
    [(0, 2, 3), (2, 0, 3), (2, 2, 0), (True, 2, 3), (2, 2.5, 3)],
)
def test_rejects_invalid_reader_dimensions(
    state_dim: object, num_modules: object, num_slots: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WorkspaceReader(
            state_dim=state_dim,  # type: ignore[arg-type]
            num_modules=num_modules,  # type: ignore[arg-type]
            num_slots=num_slots,  # type: ignore[arg-type]
        )


def test_rejects_non_workspace_state() -> None:
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)

    with pytest.raises(WorkspaceReaderError, match="WorkspaceState"):
        reader(torch.ones(1, 3, 2))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("slots", "not-a-tensor"),
        ("slots", torch.ones(1, 3)),
        ("slots", torch.ones(1, 2, 2)),
        ("slots", torch.ones(1, 3, 3)),
        ("slots", torch.ones(1, 3, 2, dtype=torch.int64)),
        ("slots", torch.full((1, 3, 2), torch.nan)),
        ("age", torch.zeros(1, 2, dtype=torch.int64)),
        ("age", torch.zeros(1, 3, dtype=torch.float32)),
        ("occupancy", torch.zeros(1, 3, dtype=torch.int64)),
        ("source_module", torch.zeros(1, 3, dtype=torch.int32)),
        ("confidence", torch.zeros(1, 3, dtype=torch.float64)),
        ("confidence", torch.full((1, 3), torch.inf)),
        ("confidence", torch.full((1, 3), -0.1)),
        ("confidence", torch.full((1, 3), 1.1)),
        ("ttl", torch.zeros(1, 3, dtype=torch.int64)),
        ("ttl", torch.zeros(1, 3, dtype=torch.float64)),
        ("ttl", torch.full((1, 3), torch.nan)),
        ("ttl", torch.full((1, 3), -0.5)),
        ("ttl", torch.tensor([[0.0, 1.0, 0.0]])),
        ("write_count", torch.zeros(1, 2, dtype=torch.int64)),
    ],
)
def test_rejects_malformed_workspace_state(field: str, replacement: object) -> None:
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)

    with pytest.raises(WorkspaceReaderError):
        reader(replace(state, **{field: cast(torch.Tensor, replacement)}))


def test_rejects_parameter_dtype_mismatch_and_nonfinite_query() -> None:
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=3,
        state_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3).to(dtype=torch.float64)
    with pytest.raises(WorkspaceReaderError, match="explicitly match"):
        reader(state)

    reader = WorkspaceReader(state_dim=2, num_modules=2, num_slots=3)
    with torch.no_grad():
        reader.read_queries[0, 0] = torch.nan
    with pytest.raises(WorkspaceReaderError, match="queries contain"):
        reader(state)


@pytest.mark.parametrize(
    ("dtype", "large_value", "query_value"),
    [(torch.float16, 60_000.0, 1.0), (torch.bfloat16, 2e38, 0.3)],
    ids=["float16", "bfloat16"],
)
def test_low_precision_reader_avoids_dot_product_overflow(
    dtype: torch.dtype, large_value: float, query_value: float
) -> None:
    reader = WorkspaceReader(state_dim=8, num_modules=3, num_slots=8).to(dtype=dtype)
    with torch.no_grad():
        reader.read_queries.fill_(query_value)
    slots = torch.full((1, 8, 8), large_value, dtype=dtype)
    occupancy = torch.ones((1, 8), dtype=torch.bool)

    output = reader(state_from_slots(slots, occupancy))

    assert output.workspace_reads.dtype == dtype
    assert output.attention_weights.dtype == torch.float32
    assert torch.isfinite(output.workspace_reads).all()
    assert torch.isfinite(output.attention_weights).all()
    torch.testing.assert_close(output.workspace_reads, slots[:, :1].expand(1, 3, 8))
    torch.testing.assert_close(output.attention_weights, torch.full((1, 3, 8), 0.125))


def test_external_autocast_cannot_lower_reader_precision() -> None:
    reader = WorkspaceReader(state_dim=4, num_modules=2, num_slots=3)
    with torch.no_grad():
        reader.read_queries.copy_(torch.tensor([[0.5, -0.25, 0.75, 1.0], [-1.0, 0.5, 0.25, -0.75]]))
    state = state_from_slots(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.5], [9.0, 8.0, 7.0, 6.0]]]),
        torch.tensor([[1, 1, 0]], dtype=torch.bool),
    )
    reference = reader(state)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = reader(state)

    assert actual.workspace_reads.dtype == torch.float32
    assert actual.attention_weights.dtype == torch.float32
    torch.testing.assert_close(actual.workspace_reads, reference.workspace_reads)
    torch.testing.assert_close(actual.attention_weights, reference.attention_weights)
