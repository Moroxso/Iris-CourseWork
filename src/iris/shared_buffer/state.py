"""Immutable append-only state for the Iris arm-C shared buffer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from iris.workspace.state import WorkspaceState


class SharedBufferStateError(ValueError):
    """Raised when shared-buffer state violates its canonical contract."""


@dataclass(frozen=True, slots=True)
class SharedBufferState:
    """A dynamically growing, cycle-major shared-buffer snapshot.

    Every completed cognitive cycle contributes exactly ``num_modules`` entries.
    Entries are ordered first by one-indexed cycle and then by zero-indexed source
    module. The native state intentionally has no occupancy, confidence, TTL,
    age, or write-count lifecycle metadata.
    """

    entries: torch.Tensor
    source_module: torch.Tensor
    write_cycle: torch.Tensor
    state_dim: int
    num_modules: int
    cycles: int

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        state_dim: int,
        num_modules: int,
        cycles: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> SharedBufferState:
        """Create the canonical zero-entry state on an explicit dtype/device."""
        for name, value in {
            "batch_size": batch_size,
            "state_dim": state_dim,
            "num_modules": num_modules,
            "cycles": cycles,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if not isinstance(dtype, torch.dtype):
            message = "dtype must be an explicit torch.dtype"
            raise ValueError(message)
        if not isinstance(device, torch.device):
            message = "device must be an explicit torch.device"
            raise ValueError(message)
        if not torch.empty((), dtype=dtype).is_floating_point():
            message = "Shared-buffer entry dtype must be floating point"
            raise ValueError(message)

        scalar_shape = (batch_size, 0)
        return cls(
            entries=torch.empty(
                (batch_size, 0, state_dim),
                dtype=dtype,
                device=device,
            ),
            source_module=torch.empty(scalar_shape, dtype=torch.int64, device=device),
            write_cycle=torch.empty(scalar_shape, dtype=torch.int64, device=device),
            state_dim=state_dim,
            num_modules=num_modules,
            cycles=cycles,
        )

    @property
    def max_entries(self) -> int:
        """Maximum entries reachable in the finite configured execution horizon."""
        return self.num_modules * self.cycles

    @property
    def entry_count(self) -> int:
        """Current dynamic cardinality shared by every batch item."""
        if not isinstance(self.entries, torch.Tensor) or self.entries.ndim < 2:
            message = "Shared-buffer entries must expose an entry axis"
            raise SharedBufferStateError(message)
        return self.entries.shape[1]

    @property
    def current_cycle(self) -> int:
        """Number of complete append cycles represented by this snapshot."""
        entry_count = self.entry_count
        if type(self.num_modules) is not int or self.num_modules <= 0:
            message = "Shared-buffer num_modules must be a positive integer"
            raise SharedBufferStateError(message)
        if entry_count % self.num_modules != 0:
            message = "Shared-buffer state cannot contain a partial cognitive cycle"
            raise SharedBufferStateError(message)
        return entry_count // self.num_modules

    def to_workspace_state(self) -> WorkspaceState:
        """Return a detached, non-aliasing trace snapshot in the legacy state type.

        This projection exists only for typed public traces. It does not introduce
        limited-workspace semantics: all dynamic entries are occupied, every write
        count is one, and age is derived from append-cycle provenance. Empty states
        are intentionally rejected because public traces are recorded post-write.
        """
        contract = validate_shared_buffer_state(
            self,
            state_dim=self.state_dim,
            num_modules=self.num_modules,
            cycles=self.cycles,
            error_type=SharedBufferStateError,
            component="Shared-buffer trace projection",
        )
        if contract.entry_count == 0:
            message = "Cannot project an empty shared buffer into a post-write trace"
            raise SharedBufferStateError(message)

        # A trace is a value snapshot, never part of the trainable graph. Explicitly
        # disabling inference mode also materializes ordinary tensors when the
        # request itself was executed under torch.inference_mode().
        with torch.inference_mode(False):
            slots = self.entries.detach().clone()
            source_module = self.source_module.detach().clone()
            write_cycle = self.write_cycle.detach().clone()
            occupancy = torch.ones(
                (contract.batch_size, contract.entry_count),
                dtype=torch.bool,
                device=contract.device,
            )
            write_count = torch.ones(
                (contract.batch_size, contract.entry_count),
                dtype=torch.int64,
                device=contract.device,
            )
            age = torch.full_like(write_cycle, self.current_cycle) - write_cycle
            # WorkspaceState requires these fields. They are constant projection
            # placeholders, not arm-C buffer lifecycle or proposal provenance.
            confidence = torch.ones(
                (contract.batch_size, contract.entry_count),
                dtype=contract.dtype,
                device=contract.device,
            )
            ttl = torch.ones(
                (contract.batch_size, contract.entry_count),
                dtype=contract.dtype,
                device=contract.device,
            )

        return WorkspaceState(
            slots=slots,
            age=age,
            occupancy=occupancy,
            source_module=source_module,
            confidence=confidence,
            ttl=ttl,
            write_count=write_count,
        )


@dataclass(frozen=True, slots=True)
class SharedBufferContract:
    """Validated dimensions and storage boundary for a shared-buffer state."""

    batch_size: int
    entry_count: int
    dtype: torch.dtype
    device: torch.device


def validate_shared_buffer_state(
    state: object,
    *,
    state_dim: int,
    num_modules: int,
    cycles: int,
    error_type: type[Exception],
    component: str,
) -> SharedBufferContract:
    """Validate exact dynamic shape, metadata, capacity, and canonical ordering."""

    def fail(detail: str) -> None:
        message = f"{component}: {detail}"
        raise error_type(message)

    if type(state) is not SharedBufferState:
        fail("input must be exactly SharedBufferState")
    assert isinstance(state, SharedBufferState)

    for name, actual, expected in (
        ("state_dim", state.state_dim, state_dim),
        ("num_modules", state.num_modules, num_modules),
        ("cycles", state.cycles, cycles),
    ):
        if type(actual) is not int or actual != expected:
            fail(f"{name} does not match the component contract")
    if type(state_dim) is not int or state_dim <= 0:
        fail("state_dim must be a positive integer")
    if type(num_modules) is not int or num_modules <= 0:
        fail("num_modules must be a positive integer")
    if type(cycles) is not int or cycles <= 0:
        fail("cycles must be a positive integer")

    entries = state.entries
    if not isinstance(entries, torch.Tensor) or entries.ndim != 3:
        fail("entries must have shape [batch, entries, state_dim]")
    batch_size, entry_count, actual_state_dim = entries.shape
    if batch_size <= 0 or actual_state_dim != state_dim:
        fail("entry shape does not match the component contract")
    if not torch.is_floating_point(entries):
        fail("entries must use a floating dtype")
    if not bool(torch.isfinite(entries).all()):
        fail("entries contain NaN or Inf")
    max_entries = num_modules * cycles
    if entry_count > max_entries:
        fail("entry count exceeds num_modules * cycles")
    if entry_count % num_modules != 0:
        fail("entry count must contain only complete cognitive cycles")

    matrix_shape = (batch_size, entry_count)
    for name, tensor in (
        ("source_module", state.source_module),
        ("write_cycle", state.write_cycle),
    ):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != matrix_shape:
            fail(f"{name} must have shape [batch, entries]")
        if tensor.dtype != torch.int64:
            fail(f"{name} has an invalid dtype")
        if tensor.device != entries.device:
            fail("all state tensors must share one device")

    positions = torch.arange(entry_count, dtype=torch.int64, device=entries.device)
    expected_sources = positions.remainder(num_modules).expand(batch_size, -1)
    expected_cycles = (torch.div(positions, num_modules, rounding_mode="floor") + 1).expand(
        batch_size, -1
    )
    if not torch.equal(state.source_module, expected_sources):
        fail("source_module must follow canonical module order in every cycle")
    if not torch.equal(state.write_cycle, expected_cycles):
        fail("write_cycle must be contiguous, one-indexed, and cycle-major")

    return SharedBufferContract(
        batch_size=batch_size,
        entry_count=entry_count,
        dtype=entries.dtype,
        device=entries.device,
    )
