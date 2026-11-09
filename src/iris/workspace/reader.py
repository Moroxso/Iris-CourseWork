"""Independent attention reads from the Iris global workspace."""

from math import sqrt

import torch
from torch import nn

from iris.workspace.outputs import WorkspaceReaderOutput
from iris.workspace.state import WorkspaceState


class WorkspaceReaderError(RuntimeError):
    """Raised when workspace state cannot satisfy the reader contract."""


class WorkspaceReader(nn.Module):
    """Read occupied workspace slots with one learned query per module."""

    def __init__(self, *, state_dim: int, num_modules: int, num_slots: int) -> None:
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
        self.read_queries = nn.Parameter(torch.zeros(num_modules, state_dim))

    def forward(self, state: WorkspaceState) -> WorkspaceReaderOutput:
        """Read slot values, returning exact zeros for empty workspaces."""
        batch_size, dtype, device = self._validate_state(state)
        self._validate_parameter_boundary(dtype=dtype, device=device)
        slots = state.slots
        if slots.is_inference() and not torch.is_inference_mode_enabled():
            slots = slots.detach().clone()

        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        with torch.autocast(device_type=device.type, enabled=False):
            safe_slots = slots.masked_fill(~state.occupancy.unsqueeze(-1), 0).to(
                dtype=accumulation_dtype
            )
            scores = torch.einsum(
                "bkd,md->bmk",
                safe_slots,
                self.read_queries.to(dtype=accumulation_dtype) / sqrt(self.state_dim),
            )
            if not bool(torch.isfinite(scores).all()):
                message = "Workspace read scores contain NaN or Inf"
                raise WorkspaceReaderError(message)
            occupancy = state.occupancy.unsqueeze(1)
            scores = scores.masked_fill(~occupancy, -torch.inf)
            has_content = state.occupancy.any(dim=1)
            scores = torch.where(has_content[:, None, None], scores, torch.zeros_like(scores))
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = torch.where(
                has_content[:, None, None],
                attention_weights,
                torch.zeros_like(attention_weights),
            )
            workspace_reads = torch.einsum("bmk,bkd->bmd", attention_weights, safe_slots).to(
                dtype=dtype
            )

        self._validate_output(
            workspace_reads,
            attention_weights,
            state.occupancy,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return WorkspaceReaderOutput(
            workspace_reads=workspace_reads,
            attention_weights=attention_weights,
        )

    def _validate_state(self, state: WorkspaceState) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(state, WorkspaceState):
            message = "Workspace reader input must be WorkspaceState"
            raise WorkspaceReaderError(message)
        if not isinstance(state.slots, torch.Tensor):
            message = "Workspace slots must be a tensor"
            raise WorkspaceReaderError(message)
        if state.slots.ndim != 3:
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise WorkspaceReaderError(message)
        batch_size, num_slots, state_dim = state.slots.shape
        if batch_size <= 0 or num_slots != self.num_slots or state_dim != self.state_dim:
            message = "Workspace slot shape does not match the reader contract"
            raise WorkspaceReaderError(message)
        if not torch.is_floating_point(state.slots):
            message = "Workspace slots must use a floating dtype"
            raise WorkspaceReaderError(message)
        if not bool(torch.isfinite(state.slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise WorkspaceReaderError(message)

        matrix_shape = (batch_size, self.num_slots)
        expected_metadata: tuple[tuple[str, torch.Tensor, torch.dtype], ...] = (
            ("age", state.age, torch.int64),
            ("occupancy", state.occupancy, torch.bool),
            ("source_module", state.source_module, torch.int64),
            ("confidence", state.confidence, state.slots.dtype),
            ("ttl", state.ttl, state.slots.dtype),
            ("write_count", state.write_count, torch.int64),
        )
        for name, tensor, expected_dtype in expected_metadata:
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != matrix_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise WorkspaceReaderError(message)
            if tensor.dtype != expected_dtype:
                message = f"Workspace {name} has an invalid dtype"
                raise WorkspaceReaderError(message)
            if tensor.device != state.slots.device:
                message = "Workspace state tensors must share one device"
                raise WorkspaceReaderError(message)
        if not bool(torch.isfinite(state.confidence).all()) or not bool(
            torch.logical_and(state.confidence >= 0, state.confidence <= 1).all()
        ):
            message = "Workspace confidence must be finite and lie in [0, 1]"
            raise WorkspaceReaderError(message)
        if not bool(torch.isfinite(state.ttl).all()) or not bool((state.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise WorkspaceReaderError(message)
        if not bool((state.ttl.masked_select(~state.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise WorkspaceReaderError(message)
        return batch_size, state.slots.dtype, state.slots.device

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if self.read_queries.dtype != dtype or self.read_queries.device != device:
            message = "WorkspaceReader parameters must explicitly match the slot dtype and device"
            raise WorkspaceReaderError(message)
        if not bool(torch.isfinite(self.read_queries).all()):
            message = "WorkspaceReader queries contain NaN or Inf"
            raise WorkspaceReaderError(message)

    def _validate_output(
        self,
        workspace_reads: torch.Tensor,
        attention_weights: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(workspace_reads.shape) != (
            batch_size,
            self.num_modules,
            self.state_dim,
        ):
            message = "Workspace reads must have shape [batch, modules, state_dim]"
            raise WorkspaceReaderError(message)
        expected_attention = (batch_size, self.num_modules, self.num_slots)
        if tuple(attention_weights.shape) != expected_attention:
            message = "Workspace attention must have shape [batch, modules, slots]"
            raise WorkspaceReaderError(message)
        if workspace_reads.dtype != dtype or workspace_reads.device != device:
            message = "Workspace reads must preserve the slot dtype and device"
            raise WorkspaceReaderError(message)
        if not bool(torch.isfinite(workspace_reads).all()) or not bool(
            torch.isfinite(attention_weights).all()
        ):
            message = "Workspace reader outputs contain NaN or Inf"
            raise WorkspaceReaderError(message)
        if not bool((attention_weights >= 0).all()):
            message = "Workspace attention must be non-negative"
            raise WorkspaceReaderError(message)
        expanded_occupancy = occupancy.unsqueeze(1).expand_as(attention_weights)
        if not bool((attention_weights.masked_select(~expanded_occupancy) == 0).all()):
            message = "Unoccupied slots must receive zero read attention"
            raise WorkspaceReaderError(message)
        expected_sums = occupancy.any(dim=1, keepdim=True).expand(batch_size, self.num_modules)
        if not torch.allclose(
            attention_weights.sum(dim=-1),
            expected_sums.to(dtype=attention_weights.dtype),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Workspace attention must sum to one or zero for empty state"
            raise WorkspaceReaderError(message)
