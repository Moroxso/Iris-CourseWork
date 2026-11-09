"""Typed functional state for the Iris global workspace."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WorkspaceState:
    """Workspace content and slot metadata for one cognitive cycle.

    Confidence and continuous TTL share the slot floating dtype. Age and write
    count are discrete integer counters.
    """

    slots: torch.Tensor
    age: torch.Tensor
    occupancy: torch.Tensor
    source_module: torch.Tensor
    confidence: torch.Tensor
    ttl: torch.Tensor
    write_count: torch.Tensor

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        num_slots: int,
        state_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "WorkspaceState":
        """Create a canonical empty workspace without hidden device fallback."""
        for name, value in {
            "batch_size": batch_size,
            "num_slots": num_slots,
            "state_dim": state_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if not isinstance(dtype, torch.dtype):
            message = "dtype must be an explicit torch.dtype"
            raise ValueError(message)
        if not isinstance(device, torch.device):
            message = "device must be an explicit torch.device"
            raise ValueError(message)
        if not torch.empty((), dtype=dtype).is_floating_point():
            message = "Workspace slot dtype must be floating point"
            raise ValueError(message)

        slot_shape = (batch_size, num_slots)
        return cls(
            slots=torch.zeros((*slot_shape, state_dim), dtype=dtype, device=device),
            age=torch.zeros(slot_shape, dtype=torch.int64, device=device),
            occupancy=torch.zeros(slot_shape, dtype=torch.bool, device=device),
            source_module=torch.full(slot_shape, -1, dtype=torch.int64, device=device),
            confidence=torch.zeros(slot_shape, dtype=dtype, device=device),
            ttl=torch.zeros(slot_shape, dtype=dtype, device=device),
            write_count=torch.zeros(slot_shape, dtype=torch.int64, device=device),
        )
