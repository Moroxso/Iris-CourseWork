"""Contract tests for the native append-only arm-C buffer state."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from iris.shared_buffer import SharedBufferState, SharedBufferStateError
from iris.shared_buffer.state import validate_shared_buffer_state


def _state(*, completed_cycles: int = 2, requires_grad: bool = False) -> SharedBufferState:
    batch_size, num_modules, state_dim, cycles = 2, 3, 4, 3
    entry_count = completed_cycles * num_modules
    entries = torch.arange(
        batch_size * entry_count * state_dim,
        dtype=torch.float32,
    ).reshape(batch_size, entry_count, state_dim)
    entries.requires_grad_(requires_grad)
    positions = torch.arange(entry_count, dtype=torch.int64)
    return SharedBufferState(
        entries=entries,
        source_module=positions.remainder(num_modules).expand(batch_size, -1).clone(),
        write_cycle=(positions.div(num_modules, rounding_mode="floor") + 1)
        .expand(batch_size, -1)
        .clone(),
        state_dim=state_dim,
        num_modules=num_modules,
        cycles=cycles,
    )


def _validate(state: object) -> None:
    validate_shared_buffer_state(
        state,
        state_dim=4,
        num_modules=3,
        cycles=3,
        error_type=SharedBufferStateError,
        component="test state",
    )


def test_empty_state_has_exact_zero_entry_native_contract() -> None:
    state = SharedBufferState.empty(
        batch_size=2,
        state_dim=4,
        num_modules=3,
        cycles=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert tuple(field.name for field in fields(SharedBufferState)) == (
        "entries",
        "source_module",
        "write_cycle",
        "state_dim",
        "num_modules",
        "cycles",
    )
    assert state.entries.shape == (2, 0, 4)
    assert state.source_module.shape == (2, 0)
    assert state.write_cycle.shape == (2, 0)
    assert state.entries.dtype == torch.float32
    assert state.source_module.dtype == torch.int64
    assert state.entry_count == 0
    assert state.current_cycle == 0
    assert state.max_entries == 9
    _validate(state)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("state_dim", -1),
        ("num_modules", True),
        ("cycles", 0),
    ],
)
def test_empty_rejects_invalid_dimensions(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "batch_size": 1,
        "state_dim": 4,
        "num_modules": 3,
        "cycles": 3,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    arguments[name] = value
    with pytest.raises(ValueError, match="positive integer"):
        SharedBufferState.empty(**arguments)  # type: ignore[arg-type]


def test_empty_requires_explicit_floating_dtype_and_device() -> None:
    with pytest.raises(ValueError, match="floating"):
        SharedBufferState.empty(
            batch_size=1,
            state_dim=4,
            num_modules=3,
            cycles=3,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match=r"torch\.dtype"):
        SharedBufferState.empty(
            batch_size=1,
            state_dim=4,
            num_modules=3,
            cycles=3,
            dtype="float32",  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )


def test_state_is_frozen_and_reports_complete_cycle_count() -> None:
    state = _state()
    _validate(state)
    assert state.entry_count == 6
    assert state.current_cycle == 2
    with pytest.raises(FrozenInstanceError):
        state.cycles = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "candidate",
    [
        object(),
        replace(_state(), state_dim=5),
        replace(_state(), entries=torch.zeros(2, 6, 5)),
        replace(_state(), entries=torch.zeros(2, 5, 4)),
        replace(_state(), entries=torch.full((2, 6, 4), torch.nan)),
        replace(_state(), source_module=torch.zeros(2, 6, dtype=torch.int32)),
        replace(_state(), write_cycle=torch.zeros(2, 6, dtype=torch.int64)),
        replace(_state(), source_module=torch.zeros(2, 6, dtype=torch.int64)),
    ],
)
def test_validator_rejects_malformed_or_noncanonical_state(candidate: object) -> None:
    with pytest.raises(SharedBufferStateError):
        _validate(candidate)


def test_validator_rejects_post_horizon_state() -> None:
    state = _state(completed_cycles=3)
    extra_entries = torch.zeros(2, 3, 4)
    candidate = replace(
        state,
        entries=torch.cat((state.entries, extra_entries), dim=1),
        source_module=torch.cat(
            (state.source_module, torch.arange(3).expand(2, -1)),
            dim=1,
        ),
        write_cycle=torch.cat(
            (state.write_cycle, torch.full((2, 3), 4, dtype=torch.int64)),
            dim=1,
        ),
    )
    with pytest.raises(SharedBufferStateError, match="exceeds"):
        _validate(candidate)


def test_workspace_projection_is_detached_non_aliasing_and_explicitly_synthetic() -> None:
    state = _state(requires_grad=True)
    projected = state.to_workspace_state()

    assert projected.slots.shape == (2, 6, 4)
    assert torch.equal(projected.slots, state.entries)
    assert torch.equal(projected.source_module, state.source_module)
    assert torch.equal(
        projected.age,
        torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]),
    )
    assert bool(projected.occupancy.all())
    assert bool((projected.confidence == 1).all())
    assert bool((projected.ttl == 1).all())
    assert bool((projected.write_count == 1).all())
    assert not projected.slots.requires_grad
    assert projected.slots.grad_fn is None
    assert (
        projected.slots.untyped_storage().data_ptr() != state.entries.untyped_storage().data_ptr()
    )
    projected.slots.zero_()
    assert bool((state.entries != 0).any())


def test_projection_materializes_ordinary_tensors_from_inference_state() -> None:
    with torch.inference_mode():
        source = _state()
    projected = source.to_workspace_state()
    assert source.entries.is_inference()
    assert not projected.slots.is_inference()
    assert not projected.source_module.is_inference()


def test_empty_state_cannot_claim_a_post_write_workspace_trace() -> None:
    state = SharedBufferState.empty(
        batch_size=1,
        state_dim=4,
        num_modules=3,
        cycles=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(SharedBufferStateError, match="empty"):
        state.to_workspace_state()
