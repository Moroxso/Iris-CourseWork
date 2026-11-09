"""Contract tests for functional global-workspace state."""

import pytest
import torch

from iris.workspace import WorkspaceState


def test_empty_workspace_has_canonical_content_and_metadata() -> None:
    state = WorkspaceState.empty(
        batch_size=2,
        num_slots=8,
        state_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert state.slots.shape == (2, 8, 4)
    assert state.slots.dtype == torch.float32
    assert state.occupancy.shape == (2, 8)
    assert state.occupancy.dtype == torch.bool
    assert not bool(state.occupancy.any())
    for tensor in (state.slots, state.age, state.confidence, state.ttl, state.write_count):
        assert torch.count_nonzero(tensor) == 0
    assert state.source_module.dtype == torch.int64
    assert bool((state.source_module == -1).all())
    assert state.age.dtype == torch.int64
    assert state.ttl.dtype == torch.float32
    assert state.write_count.dtype == torch.int64
    assert state.confidence.dtype == torch.float32
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            state.slots,
            state.age,
            state.occupancy,
            state.source_module,
            state.confidence,
            state.ttl,
            state.write_count,
        )
    )


def test_continuous_metadata_preserves_slot_dtype() -> None:
    state = WorkspaceState.empty(
        batch_size=1,
        num_slots=2,
        state_dim=3,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert state.confidence.dtype == torch.float64
    assert state.ttl.dtype == torch.float64


@pytest.mark.parametrize(
    ("batch_size", "num_slots", "state_dim"),
    [(0, 8, 4), (1, 0, 4), (1, 8, 0), (True, 8, 4)],
)
def test_empty_workspace_rejects_invalid_dimensions(
    batch_size: int, num_slots: int, state_dim: int
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WorkspaceState.empty(
            batch_size=batch_size,
            num_slots=num_slots,
            state_dim=state_dim,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_empty_workspace_rejects_nonfloating_slot_dtype() -> None:
    with pytest.raises(ValueError, match="floating point"):
        WorkspaceState.empty(
            batch_size=1,
            num_slots=8,
            state_dim=4,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("dtype", "device", "message"),
    [(None, torch.device("cpu"), "dtype"), (torch.float32, None, "device")],
)
def test_empty_workspace_requires_explicit_dtype_and_device(
    dtype: object, device: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WorkspaceState.empty(
            batch_size=1,
            num_slots=8,
            state_dim=4,
            dtype=dtype,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
        )
