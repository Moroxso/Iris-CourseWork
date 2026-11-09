"""Learned per-module broadcast reads over every arm-C buffer entry."""

from math import sqrt

import torch
from torch import nn

from iris.shared_buffer.state import SharedBufferState, validate_shared_buffer_state
from iris.workspace.outputs import WorkspaceReaderOutput


class SharedBufferReaderError(RuntimeError):
    """Raised when a shared buffer cannot satisfy the broadcast-read contract."""


class SharedBufferReader(nn.Module):
    """Read all preceding entries with one learned query per cognitive module."""

    def __init__(self, *, state_dim: int, num_modules: int, cycles: int) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "num_modules": num_modules,
            "cycles": cycles,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.cycles = cycles
        self.max_entries = num_modules * cycles
        self.read_queries = nn.Parameter(torch.zeros(num_modules, state_dim))

    def forward(self, state: SharedBufferState) -> WorkspaceReaderOutput:
        """Return independent module reads; an empty state yields exact zeros."""
        contract = validate_shared_buffer_state(
            state,
            state_dim=self.state_dim,
            num_modules=self.num_modules,
            cycles=self.cycles,
            error_type=SharedBufferReaderError,
            component="Shared-buffer reader",
        )
        self._validate_parameter_boundary(dtype=contract.dtype, device=contract.device)
        entries = self._materialize(state.entries)
        accumulation_dtype = (
            torch.float32 if contract.dtype in {torch.float16, torch.bfloat16} else contract.dtype
        )

        with torch.autocast(device_type=contract.device.type, enabled=False):
            if contract.entry_count == 0:
                workspace_reads = torch.zeros(
                    (contract.batch_size, self.num_modules, self.state_dim),
                    dtype=contract.dtype,
                    device=contract.device,
                )
                attention_weights = torch.zeros(
                    (contract.batch_size, self.num_modules, 0),
                    dtype=accumulation_dtype,
                    device=contract.device,
                )
            else:
                safe_entries = entries.to(dtype=accumulation_dtype)
                scores = torch.einsum(
                    "bed,md->bme",
                    safe_entries,
                    self.read_queries.to(dtype=accumulation_dtype) / sqrt(self.state_dim),
                )
                if not bool(torch.isfinite(scores).all()):
                    message = "Shared-buffer read scores contain NaN or Inf"
                    raise SharedBufferReaderError(message)
                attention_weights = torch.softmax(scores, dim=-1)
                workspace_reads = torch.einsum(
                    "bme,bed->bmd",
                    attention_weights,
                    safe_entries,
                ).to(dtype=contract.dtype)

        self._validate_output(
            workspace_reads,
            attention_weights,
            batch_size=contract.batch_size,
            entry_count=contract.entry_count,
            dtype=contract.dtype,
            device=contract.device,
        )
        return WorkspaceReaderOutput(
            workspace_reads=workspace_reads,
            attention_weights=attention_weights,
        )

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if self.read_queries.dtype != dtype or self.read_queries.device != device:
            message = "SharedBufferReader parameters must match the entry dtype and device"
            raise SharedBufferReaderError(message)
        if not bool(torch.isfinite(self.read_queries).all()):
            message = "SharedBufferReader queries contain NaN or Inf"
            raise SharedBufferReaderError(message)

    def _validate_output(
        self,
        workspace_reads: torch.Tensor,
        attention_weights: torch.Tensor,
        *,
        batch_size: int,
        entry_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(workspace_reads.shape) != (batch_size, self.num_modules, self.state_dim):
            message = "Shared-buffer reads must have shape [batch, modules, state_dim]"
            raise SharedBufferReaderError(message)
        if tuple(attention_weights.shape) != (batch_size, self.num_modules, entry_count):
            message = "Shared-buffer attention must cover every dynamic entry"
            raise SharedBufferReaderError(message)
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        if workspace_reads.dtype != dtype or workspace_reads.device != device:
            message = "Shared-buffer reads must preserve entry dtype and device"
            raise SharedBufferReaderError(message)
        if attention_weights.dtype != accumulation_dtype or attention_weights.device != device:
            message = "Shared-buffer attention has an invalid accumulation dtype or device"
            raise SharedBufferReaderError(message)
        if not bool(torch.isfinite(workspace_reads).all()) or not bool(
            torch.isfinite(attention_weights).all()
        ):
            message = "Shared-buffer reader outputs contain NaN or Inf"
            raise SharedBufferReaderError(message)
        if not bool((attention_weights >= 0).all()):
            message = "Shared-buffer attention must be non-negative"
            raise SharedBufferReaderError(message)
        expected_sum = 0.0 if entry_count == 0 else 1.0
        if not torch.allclose(
            attention_weights.sum(dim=-1),
            torch.full(
                (batch_size, self.num_modules),
                expected_sum,
                dtype=attention_weights.dtype,
                device=device,
            ),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Shared-buffer attention must sum to one, or zero when empty"
            raise SharedBufferReaderError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor
