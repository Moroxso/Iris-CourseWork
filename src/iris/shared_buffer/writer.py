"""Gated append-only writes for the Iris arm-C shared buffer."""

import torch
import torch.nn.functional as functional
from torch import nn

from iris.cognitive.outputs import ModuleProposal
from iris.shared_buffer.state import SharedBufferState, validate_shared_buffer_state


class SharedBufferWriterError(RuntimeError):
    """Raised when an append would violate the shared-buffer contract."""


class SharedBufferWriter(nn.Module):
    """Apply D-compatible empty-slot transforms and append every proposal."""

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
        # Equation-compatible with WorkspaceWriter's empty-slot branch. C
        # removes allocation and overwrite, not the transformation itself.
        self.gate_projection = nn.Linear(3 * state_dim, state_dim, bias=False)
        self.candidate_projection = nn.Linear(2 * state_dim, state_dim, bias=False)
        self.normalization = nn.LayerNorm(state_dim)

    def forward(
        self,
        proposal: ModuleProposal,
        state: SharedBufferState,
        goal_state: torch.Tensor,
        *,
        cycle_number: int,
    ) -> SharedBufferState:
        """Transform and append exactly one proposal per module."""
        batch_size, dtype, device = self._validate_proposal(proposal)
        contract = validate_shared_buffer_state(
            state,
            state_dim=self.state_dim,
            num_modules=self.num_modules,
            cycles=self.cycles,
            error_type=SharedBufferWriterError,
            component="Shared-buffer writer",
        )
        if contract.batch_size != batch_size:
            message = "Proposal and shared-buffer batch sizes must match"
            raise SharedBufferWriterError(message)
        if contract.dtype != dtype or contract.device != device:
            message = "Proposal and shared-buffer entries must share one dtype and device"
            raise SharedBufferWriterError(message)
        expected_cycle = contract.entry_count // self.num_modules + 1
        if type(cycle_number) is not int or cycle_number != expected_cycle:
            message = "cycle_number must be the next one-based append cycle"
            raise SharedBufferWriterError(message)
        if cycle_number > self.cycles or contract.entry_count == self.max_entries:
            message = "Shared buffer already contains every configured cognitive cycle"
            raise SharedBufferWriterError(message)
        self._validate_goal(
            goal_state,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        content = self._materialize(proposal.content)
        entries = self._materialize(state.entries)
        source_module = self._materialize(state.source_module)
        write_cycle = self._materialize(state.write_cycle)
        goal = self._materialize(goal_state)
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

        with torch.autocast(device_type=device.type, enabled=False):
            safe_content = content.to(dtype=accumulation_dtype)
            empty_slots = torch.zeros_like(safe_content)
            goal_grid = (
                goal.to(dtype=accumulation_dtype)
                .unsqueeze(1)
                .expand(
                    -1,
                    self.num_modules,
                    -1,
                )
            )
            gate_input = torch.cat((safe_content, empty_slots, goal_grid), dim=-1)
            candidate_input = torch.cat((safe_content, empty_slots), dim=-1)
            gate_logits = functional.linear(
                gate_input,
                self.gate_projection.weight.to(dtype=accumulation_dtype),
            )
            candidate_logits = functional.linear(
                candidate_input,
                self.candidate_projection.weight.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(gate_logits).all()):
                message = "Shared-buffer write gate logits contain NaN or Inf"
                raise SharedBufferWriterError(message)
            if not bool(torch.isfinite(candidate_logits).all()):
                message = "Shared-buffer write candidate logits contain NaN or Inf"
                raise SharedBufferWriterError(message)
            transformed = torch.sigmoid(gate_logits) * torch.tanh(candidate_logits)
            normalized = self._stable_layer_norm(
                transformed,
                weight=self.normalization.weight.to(dtype=accumulation_dtype),
                bias=self.normalization.bias.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(normalized).all()):
                message = "Normalized shared-buffer writes contain NaN or Inf"
                raise SharedBufferWriterError(message)
            appended_entries = normalized.to(dtype=dtype)

        if not bool(torch.isfinite(appended_entries).all()):
            message = "Shared-buffer writes contain NaN or Inf after storage cast"
            raise SharedBufferWriterError(message)
        appended_sources = torch.arange(
            self.num_modules,
            dtype=torch.int64,
            device=device,
        ).expand(batch_size, -1)
        appended_cycles = torch.full(
            (batch_size, self.num_modules),
            cycle_number,
            dtype=torch.int64,
            device=device,
        )
        output = SharedBufferState(
            entries=torch.cat((entries, appended_entries), dim=1),
            source_module=torch.cat((source_module, appended_sources), dim=1),
            write_cycle=torch.cat((write_cycle, appended_cycles), dim=1),
            state_dim=self.state_dim,
            num_modules=self.num_modules,
            cycles=self.cycles,
        )
        output_contract = validate_shared_buffer_state(
            output,
            state_dim=self.state_dim,
            num_modules=self.num_modules,
            cycles=self.cycles,
            error_type=SharedBufferWriterError,
            component="Shared-buffer writer output",
        )
        if output_contract.entry_count != contract.entry_count + self.num_modules:
            message = "Shared-buffer append must add exactly one entry per module"
            raise SharedBufferWriterError(message)
        if contract.entry_count and (
            not torch.equal(output.entries[:, : contract.entry_count], state.entries)
            or not torch.equal(
                output.source_module[:, : contract.entry_count],
                state.source_module,
            )
            or not torch.equal(output.write_cycle[:, : contract.entry_count], state.write_cycle)
        ):
            message = "Shared-buffer append changed an existing entry or provenance field"
            raise SharedBufferWriterError(message)
        return output

    def _validate_proposal(self, proposal: ModuleProposal) -> tuple[int, torch.dtype, torch.device]:
        if type(proposal) is not ModuleProposal:
            message = "Shared-buffer writer input must be exactly ModuleProposal"
            raise SharedBufferWriterError(message)
        content = proposal.content
        if type(content) is not torch.Tensor or content.ndim != 3:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise SharedBufferWriterError(message)
        batch_size = content.shape[0]
        vector_shape = (batch_size, self.num_modules, self.state_dim)
        scalar_shape = (batch_size, self.num_modules)
        if batch_size <= 0 or tuple(content.shape) != vector_shape:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise SharedBufferWriterError(message)
        for name, tensor, expected_shape in (
            ("content", proposal.content, vector_shape),
            ("key", proposal.key, vector_shape),
            ("bid", proposal.bid, scalar_shape),
            ("confidence", proposal.confidence, scalar_shape),
            ("ttl", proposal.ttl, scalar_shape),
        ):
            if type(tensor) is not torch.Tensor or tuple(tensor.shape) != expected_shape:
                message = f"Proposal {name} has an invalid shape"
                raise SharedBufferWriterError(message)
            if not torch.is_floating_point(tensor):
                message = f"Proposal {name} must use a floating dtype"
                raise SharedBufferWriterError(message)
            if tensor.dtype != content.dtype or tensor.device != content.device:
                message = "Proposal tensors must share one dtype and device"
                raise SharedBufferWriterError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Proposal {name} contains NaN or Inf"
                raise SharedBufferWriterError(message)
        if not bool(torch.logical_and(proposal.confidence >= 0, proposal.confidence <= 1).all()):
            message = "Proposal confidence must lie in [0, 1]"
            raise SharedBufferWriterError(message)
        if not bool(torch.logical_and(proposal.ttl >= 1, proposal.ttl <= self.cycles).all()):
            message = "Proposal TTL must lie in [1, cycles]"
            raise SharedBufferWriterError(message)
        return batch_size, content.dtype, content.device

    def _validate_goal(
        self,
        goal_state: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if type(goal_state) is not torch.Tensor or tuple(goal_state.shape) != (
            batch_size,
            self.state_dim,
        ):
            message = "Goal state must have shape [batch, state_dim]"
            raise SharedBufferWriterError(message)
        if (
            not torch.is_floating_point(goal_state)
            or goal_state.dtype != dtype
            or goal_state.device != device
            or not bool(torch.isfinite(goal_state).all())
        ):
            message = "Goal state must be finite and match proposal dtype and device"
            raise SharedBufferWriterError(message)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = "SharedBufferWriter parameters must match entry dtype and device"
                raise SharedBufferWriterError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"SharedBufferWriter parameter {name!r} contains NaN or Inf"
                raise SharedBufferWriterError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor

    def _stable_layer_norm(
        self,
        tensor: torch.Tensor,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        maximum = tensor.abs().amax(dim=-1, keepdim=True)
        epsilon_scale = torch.full_like(maximum, self.normalization.eps**0.5)
        scale = torch.maximum(maximum, epsilon_scale)
        scaled = tensor / scale
        centered = scaled - scaled.mean(dim=-1, keepdim=True)
        standard_deviation = torch.linalg.vector_norm(
            centered,
            dim=-1,
            keepdim=True,
        ) / (self.state_dim**0.5)
        denominator = torch.hypot(standard_deviation, epsilon_scale / scale)
        normalized = centered / denominator
        return normalized * weight + bias
