"""Functional lifecycle decay for Iris global-workspace slots."""

from math import isfinite

import torch
from torch import nn

from iris.workspace.state import WorkspaceState


class WorkspaceDecayError(RuntimeError):
    """Raised when a workspace cannot satisfy the decay contract."""


class WorkspaceDecay(nn.Module):
    """Age occupied slots, decrement TTL, decay expired content, and clear it."""

    def __init__(
        self,
        *,
        state_dim: int,
        num_modules: int,
        num_slots: int,
        decay_rate: float,
        empty_threshold: float,
    ) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "num_modules": num_modules,
            "num_slots": num_slots,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.num_slots = num_slots
        self.decay_rate = self._validate_decay_rate(decay_rate)
        self.empty_threshold = self._validate_threshold(empty_threshold)

    def forward(self, workspace: WorkspaceState) -> WorkspaceState:
        """Return the canonical workspace state at the next cycle boundary."""
        batch_size, dtype, device = self._validate_state(workspace)
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

        slots = self._materialize(workspace.slots)
        age = self._materialize(workspace.age)
        occupancy = self._materialize(workspace.occupancy)
        source_module = self._materialize(workspace.source_module)
        confidence = self._materialize(workspace.confidence)
        ttl = self._materialize(workspace.ttl)
        write_count = self._materialize(workspace.write_count)

        if bool((age.masked_select(occupancy) == torch.iinfo(torch.int64).max).any()):
            message = "Occupied workspace age cannot be incremented without int64 overflow"
            raise WorkspaceDecayError(message)

        canonical_slots = torch.where(occupancy.unsqueeze(-1), slots, torch.zeros_like(slots))
        canonical_age = torch.where(occupancy, age, torch.zeros_like(age))
        canonical_source = torch.where(
            occupancy,
            source_module,
            torch.full_like(source_module, -1),
        )
        canonical_confidence = torch.where(
            occupancy,
            confidence,
            torch.zeros_like(confidence),
        )
        canonical_ttl = torch.where(occupancy, ttl, torch.zeros_like(ttl))
        canonical_write_count = torch.where(
            occupancy,
            write_count,
            torch.zeros_like(write_count),
        )

        with torch.autocast(device_type=device.type, enabled=False):
            safe_slots = canonical_slots.to(dtype=accumulation_dtype)
            raw_ttl = canonical_ttl.to(dtype=accumulation_dtype) - occupancy.to(
                dtype=accumulation_dtype
            )
            expired = occupancy & (raw_ttl <= 0)
            decayed_slots = torch.where(
                expired.unsqueeze(-1),
                safe_slots * self.decay_rate,
                safe_slots,
            )
            stored_slots = decayed_slots.to(dtype=dtype)
            if not bool(torch.isfinite(stored_slots).all()):
                message = "Decayed workspace slots contain NaN or Inf after storage cast"
                raise WorkspaceDecayError(message)
            below_threshold = self._below_threshold(stored_slots.to(dtype=accumulation_dtype))
            survives = occupancy & ~below_threshold
            next_ttl = torch.where(
                expired,
                torch.zeros_like(raw_ttl),
                raw_ttl,
            ).to(dtype=dtype)

        next_age = canonical_age + occupancy.to(dtype=torch.int64)
        output = WorkspaceState(
            slots=torch.where(
                survives.unsqueeze(-1),
                stored_slots,
                torch.zeros_like(stored_slots),
            ),
            age=torch.where(survives, next_age, torch.zeros_like(next_age)),
            occupancy=survives,
            source_module=torch.where(
                survives,
                canonical_source,
                torch.full_like(canonical_source, -1),
            ),
            confidence=torch.where(
                survives,
                canonical_confidence,
                torch.zeros_like(canonical_confidence),
            ),
            ttl=torch.where(survives, next_ttl, torch.zeros_like(next_ttl)),
            write_count=torch.where(
                survives,
                canonical_write_count,
                torch.zeros_like(canonical_write_count),
            ),
        )
        self._validate_output(
            output,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return output

    @staticmethod
    def _validate_decay_rate(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = "decay_rate must be a finite number in (0, 1)"
            raise ValueError(message)
        result = float(value)
        if not isfinite(result) or result <= 0 or result >= 1:
            message = "decay_rate must be a finite number in (0, 1)"
            raise ValueError(message)
        return result

    @staticmethod
    def _validate_threshold(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = "empty_threshold must be a finite positive number"
            raise ValueError(message)
        result = float(value)
        if not isfinite(result) or result <= 0:
            message = "empty_threshold must be a finite positive number"
            raise ValueError(message)
        return result

    def _validate_state(
        self,
        workspace: WorkspaceState,
    ) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(workspace, WorkspaceState):
            message = "Workspace decay input must be WorkspaceState"
            raise WorkspaceDecayError(message)
        slots = workspace.slots
        if not isinstance(slots, torch.Tensor) or slots.ndim != 3:
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise WorkspaceDecayError(message)
        batch_size = slots.shape[0]
        slot_shape = (batch_size, self.num_slots)
        if batch_size <= 0 or tuple(slots.shape) != (*slot_shape, self.state_dim):
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise WorkspaceDecayError(message)
        if not torch.is_floating_point(slots):
            message = "Workspace slots must use a floating dtype"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise WorkspaceDecayError(message)
        dtype = slots.dtype
        device = slots.device
        for name, tensor, expected_dtype in (
            ("age", workspace.age, torch.int64),
            ("occupancy", workspace.occupancy, torch.bool),
            ("source_module", workspace.source_module, torch.int64),
            ("confidence", workspace.confidence, dtype),
            ("ttl", workspace.ttl, dtype),
            ("write_count", workspace.write_count, torch.int64),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != slot_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise WorkspaceDecayError(message)
            if tensor.dtype != expected_dtype or tensor.device != device:
                message = f"Workspace {name} has an invalid dtype or device"
                raise WorkspaceDecayError(message)
        if not bool((workspace.age.masked_select(workspace.occupancy) >= 0).all()) or not bool(
            (workspace.write_count.masked_select(workspace.occupancy) >= 0).all()
        ):
            message = "Occupied workspace counters must be non-negative"
            raise WorkspaceDecayError(message)
        occupied_sources = workspace.source_module.masked_select(workspace.occupancy)
        if not bool(
            torch.logical_and(occupied_sources >= 0, occupied_sources < self.num_modules).all()
        ):
            message = "Occupied workspace slots must have a valid source module"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(workspace.confidence).all()):
            message = "Workspace confidence must be finite"
            raise WorkspaceDecayError(message)
        occupied_confidence = workspace.confidence.masked_select(workspace.occupancy)
        if not bool(torch.logical_and(occupied_confidence >= 0, occupied_confidence <= 1).all()):
            message = "Occupied workspace confidence must lie in [0, 1]"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(workspace.ttl).all()) or not bool((workspace.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise WorkspaceDecayError(message)
        if not bool((workspace.ttl.masked_select(~workspace.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise WorkspaceDecayError(message)
        return batch_size, dtype, device

    def _below_threshold(self, slots: torch.Tensor) -> torch.Tensor:
        maximum = slots.abs().amax(dim=-1)
        nonzero = maximum > 0
        safe_maximum = torch.where(nonzero, maximum, torch.ones_like(maximum))
        scaled_norm = torch.linalg.vector_norm(slots / safe_maximum.unsqueeze(-1), dim=-1)
        scaled_threshold = self.empty_threshold / safe_maximum
        below_threshold: torch.Tensor = ~nonzero | (scaled_norm < scaled_threshold)
        return below_threshold

    def _validate_output(
        self,
        state: WorkspaceState,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        slot_shape = (batch_size, self.num_slots)
        if tuple(state.slots.shape) != (*slot_shape, self.state_dim):
            message = "Decayed workspace slots have an invalid shape"
            raise WorkspaceDecayError(message)
        if state.slots.dtype != dtype or state.slots.device != device:
            message = "Decayed workspace slots have an invalid dtype or device"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(state.slots).all()):
            message = "Decayed workspace slots contain NaN or Inf"
            raise WorkspaceDecayError(message)
        for name, tensor, expected_dtype in (
            ("age", state.age, torch.int64),
            ("occupancy", state.occupancy, torch.bool),
            ("source_module", state.source_module, torch.int64),
            ("confidence", state.confidence, dtype),
            ("ttl", state.ttl, dtype),
            ("write_count", state.write_count, torch.int64),
        ):
            if (
                tuple(tensor.shape) != slot_shape
                or tensor.dtype != expected_dtype
                or tensor.device != device
            ):
                message = f"Decayed workspace {name} has an invalid shape, dtype, or device"
                raise WorkspaceDecayError(message)
        if not bool((state.age >= 0).all()) or not bool((state.write_count >= 0).all()):
            message = "Decayed workspace counters must be non-negative"
            raise WorkspaceDecayError(message)
        if not bool(
            torch.logical_and(
                state.source_module >= -1, state.source_module < self.num_modules
            ).all()
        ):
            message = "Decayed workspace source_module is out of range"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(state.confidence).all()) or not bool(
            torch.logical_and(state.confidence >= 0, state.confidence <= 1).all()
        ):
            message = "Decayed workspace confidence must lie in [0, 1]"
            raise WorkspaceDecayError(message)
        if not bool(torch.isfinite(state.ttl).all()) or not bool((state.ttl >= 0).all()):
            message = "Decayed workspace TTL must be finite and non-negative"
            raise WorkspaceDecayError(message)
        empty = ~state.occupancy
        for name, tensor, expected in (
            ("slots", state.slots, 0),
            ("age", state.age, 0),
            ("source_module", state.source_module, -1),
            ("confidence", state.confidence, 0),
            ("ttl", state.ttl, 0),
            ("write_count", state.write_count, 0),
        ):
            mask = empty.unsqueeze(-1) if name == "slots" else empty
            if not bool((tensor.masked_select(mask) == expected).all()):
                message = f"Empty decayed slots must have canonical {name}"
                raise WorkspaceDecayError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor
