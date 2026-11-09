"""Functional gated writes into the Iris global workspace."""

import torch
import torch.nn.functional as functional
from torch import nn

from iris.cognitive.outputs import ModuleProposal
from iris.workspace.outputs import SlotAllocatorOutput
from iris.workspace.state import WorkspaceState


class WorkspaceWriterError(RuntimeError):
    """Raised when a gated workspace write violates the state contract."""


class WorkspaceWriter(nn.Module):
    """Apply simultaneous gated proposal writes to unique allocated slots."""

    def __init__(
        self,
        *,
        state_dim: int,
        num_modules: int,
        num_slots: int,
        winners_per_cycle: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "num_modules": num_modules,
            "num_slots": num_slots,
            "winners_per_cycle": winners_per_cycle,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if winners_per_cycle > num_modules:
            message = "winners_per_cycle cannot exceed num_modules"
            raise ValueError(message)
        if winners_per_cycle > num_slots:
            message = "winners_per_cycle cannot exceed num_slots"
            raise ValueError(message)

        self.state_dim = state_dim
        self.num_modules = num_modules
        self.num_slots = num_slots
        self.winners_per_cycle = winners_per_cycle
        self.gate_projection = nn.Linear(3 * state_dim, state_dim, bias=False)
        self.candidate_projection = nn.Linear(2 * state_dim, state_dim, bias=False)
        self.normalization = nn.LayerNorm(state_dim)

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        allocation: SlotAllocatorOutput,
        goal_state: torch.Tensor,
    ) -> WorkspaceState:
        """Return a new state after exactly one write per allocated target."""
        batch_size, dtype, device = self._validate_proposal(proposal)
        self._validate_workspace(
            workspace,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        self._validate_allocation(
            allocation,
            batch_size=batch_size,
            dtype=accumulation_dtype,
            device=device,
        )
        self._validate_goal(
            goal_state,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        content = self._materialize(proposal.content)
        confidence = self._materialize(proposal.confidence)
        proposal_ttl = self._materialize(proposal.ttl)
        slots = self._materialize(workspace.slots)
        occupancy = self._materialize(workspace.occupancy)
        source_module = self._materialize(workspace.source_module)
        workspace_confidence = self._materialize(workspace.confidence)
        workspace_ttl = self._materialize(workspace.ttl)
        age = self._materialize(workspace.age)
        write_count = self._materialize(workspace.write_count)
        assignment_mask = self._materialize(allocation.assignment_mask)
        assignment_weights = self._materialize(allocation.assignment_weights)
        goal_state = self._materialize(goal_state)

        base_slots = torch.where(occupancy.unsqueeze(-1), slots, torch.zeros_like(slots))
        base_confidence = torch.where(
            occupancy,
            workspace_confidence,
            torch.zeros_like(workspace_confidence),
        )
        base_ttl = torch.where(occupancy, workspace_ttl, torch.zeros_like(workspace_ttl))
        selected_slots = assignment_mask.any(dim=1)
        selected_modules = assignment_mask.to(dtype=torch.int64).argmax(dim=1)

        with torch.autocast(device_type=device.type, enabled=False):
            safe_content = content.to(dtype=accumulation_dtype)
            safe_slots = base_slots.to(dtype=accumulation_dtype)
            safe_goal = goal_state.to(dtype=accumulation_dtype)
            module_grid = safe_content.unsqueeze(2).expand(-1, -1, self.num_slots, -1)
            slot_grid = safe_slots.unsqueeze(1).expand(-1, self.num_modules, -1, -1)
            goal_grid = safe_goal[:, None, None, :].expand(
                -1,
                self.num_modules,
                self.num_slots,
                -1,
            )
            gate_input = torch.cat((module_grid, slot_grid, goal_grid), dim=-1)
            candidate_input = torch.cat((module_grid, slot_grid), dim=-1)
            gate_logits = functional.linear(
                gate_input,
                self.gate_projection.weight.to(dtype=accumulation_dtype),
            )
            candidate_logits = functional.linear(
                candidate_input,
                self.candidate_projection.weight.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(gate_logits).all()):
                message = "Workspace write gate logits contain NaN or Inf"
                raise WorkspaceWriterError(message)
            if not bool(torch.isfinite(candidate_logits).all()):
                message = "Workspace write candidate logits contain NaN or Inf"
                raise WorkspaceWriterError(message)

            gates = torch.sigmoid(gate_logits)
            candidates = torch.tanh(candidate_logits)
            blended = (1.0 - gates) * slot_grid + gates * candidates
            safe_assignments = assignment_weights.to(dtype=accumulation_dtype)
            slot_delta = torch.einsum(
                "bmk,bmkd->bkd",
                safe_assignments,
                blended - slot_grid,
            )
            pre_normalized_slots = safe_slots + slot_delta
            if not bool(torch.isfinite(pre_normalized_slots).all()):
                message = "Gated workspace writes contain NaN or Inf"
                raise WorkspaceWriterError(message)
            normalized_slots = self._stable_layer_norm(
                pre_normalized_slots,
                weight=self.normalization.weight.to(dtype=accumulation_dtype),
                bias=self.normalization.bias.to(dtype=accumulation_dtype),
            )
            if not bool(torch.isfinite(normalized_slots).all()):
                message = "Normalized workspace writes contain NaN or Inf"
                raise WorkspaceWriterError(message)
            selected_float = selected_slots.to(dtype=accumulation_dtype)
            updated_slots = safe_slots + selected_float.unsqueeze(-1) * (
                normalized_slots - safe_slots
            )
            confidence_delta = torch.einsum(
                "bmk,bmk->bk",
                safe_assignments,
                confidence.to(dtype=accumulation_dtype).unsqueeze(-1)
                - base_confidence.to(dtype=accumulation_dtype).unsqueeze(1),
            )
            ttl_delta = torch.einsum(
                "bmk,bmk->bk",
                safe_assignments,
                proposal_ttl.to(dtype=accumulation_dtype).unsqueeze(-1)
                - base_ttl.to(dtype=accumulation_dtype).unsqueeze(1),
            )
            updated_confidence = base_confidence.to(dtype=accumulation_dtype) + confidence_delta
            updated_ttl = base_ttl.to(dtype=accumulation_dtype) + ttl_delta

        new_slots = updated_slots.to(dtype=dtype)
        new_confidence = updated_confidence.to(dtype=dtype)
        new_ttl = updated_ttl.to(dtype=dtype)
        new_occupancy = occupancy | selected_slots
        canonical_age = torch.where(occupancy, age, torch.zeros_like(age))
        canonical_source = torch.where(
            occupancy,
            source_module,
            torch.full_like(source_module, -1),
        )
        canonical_write_count = torch.where(
            occupancy,
            write_count,
            torch.zeros_like(write_count),
        )
        new_state = WorkspaceState(
            slots=new_slots,
            age=torch.where(selected_slots, torch.zeros_like(canonical_age), canonical_age),
            occupancy=new_occupancy,
            source_module=torch.where(selected_slots, selected_modules, canonical_source),
            confidence=new_confidence,
            ttl=new_ttl,
            write_count=canonical_write_count + selected_slots.to(dtype=torch.int64),
        )
        self._validate_output(
            new_state,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return new_state

    def _validate_proposal(self, proposal: ModuleProposal) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(proposal, ModuleProposal):
            message = "Workspace writer input must be ModuleProposal"
            raise WorkspaceWriterError(message)
        content = proposal.content
        if not isinstance(content, torch.Tensor) or content.ndim != 3:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise WorkspaceWriterError(message)
        batch_size = content.shape[0]
        vector_shape = (batch_size, self.num_modules, self.state_dim)
        scalar_shape = (batch_size, self.num_modules)
        if batch_size <= 0 or tuple(content.shape) != vector_shape:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise WorkspaceWriterError(message)
        for name, tensor, expected_shape in (
            ("content", proposal.content, vector_shape),
            ("key", proposal.key, vector_shape),
            ("bid", proposal.bid, scalar_shape),
            ("confidence", proposal.confidence, scalar_shape),
            ("ttl", proposal.ttl, scalar_shape),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                message = f"Proposal {name} has an invalid shape"
                raise WorkspaceWriterError(message)
            if not torch.is_floating_point(tensor):
                message = f"Proposal {name} must use a floating dtype"
                raise WorkspaceWriterError(message)
            if tensor.dtype != content.dtype or tensor.device != content.device:
                message = "Proposal tensors must share one dtype and device"
                raise WorkspaceWriterError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Proposal {name} contains NaN or Inf"
                raise WorkspaceWriterError(message)
        if not bool(torch.logical_and(proposal.confidence >= 0, proposal.confidence <= 1).all()):
            message = "Proposal confidence must lie in [0, 1]"
            raise WorkspaceWriterError(message)
        if not bool((proposal.ttl >= 1).all()):
            message = "Proposal TTL must be at least one cycle"
            raise WorkspaceWriterError(message)
        return batch_size, content.dtype, content.device

    def _validate_workspace(
        self,
        workspace: WorkspaceState,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(workspace, WorkspaceState):
            message = "Workspace writer input must be WorkspaceState"
            raise WorkspaceWriterError(message)
        slot_shape = (batch_size, self.num_slots)
        if not isinstance(workspace.slots, torch.Tensor) or tuple(workspace.slots.shape) != (
            *slot_shape,
            self.state_dim,
        ):
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise WorkspaceWriterError(message)
        if not torch.is_floating_point(workspace.slots):
            message = "Workspace slots must use a floating dtype"
            raise WorkspaceWriterError(message)
        if workspace.slots.dtype != dtype or workspace.slots.device != device:
            message = "Workspace slots must share the proposal dtype and device"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(workspace.slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise WorkspaceWriterError(message)
        if (
            not isinstance(workspace.occupancy, torch.Tensor)
            or tuple(workspace.occupancy.shape) != slot_shape
            or workspace.occupancy.dtype != torch.bool
            or workspace.occupancy.device != device
        ):
            message = "Workspace occupancy must be boolean [batch, slots] on the input device"
            raise WorkspaceWriterError(message)
        for name, tensor, expected_dtype in (
            ("age", workspace.age, torch.int64),
            ("source_module", workspace.source_module, torch.int64),
            ("confidence", workspace.confidence, dtype),
            ("ttl", workspace.ttl, dtype),
            ("write_count", workspace.write_count, torch.int64),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != slot_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise WorkspaceWriterError(message)
            if tensor.dtype != expected_dtype or tensor.device != device:
                message = f"Workspace {name} has an invalid dtype or device"
                raise WorkspaceWriterError(message)
        if not bool((workspace.age >= 0).all()) or not bool((workspace.write_count >= 0).all()):
            message = "Workspace counters must be non-negative"
            raise WorkspaceWriterError(message)
        if not bool(
            torch.logical_and(
                workspace.source_module >= -1,
                workspace.source_module < self.num_modules,
            ).all()
        ):
            message = "Workspace source_module must be -1 or a valid module index"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(workspace.confidence).all()) or not bool(
            torch.logical_and(workspace.confidence >= 0, workspace.confidence <= 1).all()
        ):
            message = "Workspace confidence must be finite and lie in [0, 1]"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(workspace.ttl).all()) or not bool((workspace.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise WorkspaceWriterError(message)
        if not bool((workspace.ttl.masked_select(~workspace.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise WorkspaceWriterError(message)

    def _validate_allocation(
        self,
        allocation: SlotAllocatorOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(allocation, SlotAllocatorOutput):
            message = "Workspace writer input must be SlotAllocatorOutput"
            raise WorkspaceWriterError(message)
        dense_shape = (batch_size, self.num_modules, self.num_slots)
        for name, tensor, expected_shape in (
            ("allocation_scores", allocation.allocation_scores, dense_shape),
            ("retention_scores", allocation.retention_scores, (batch_size, self.num_slots)),
            ("assignment_probabilities", allocation.assignment_probabilities, dense_shape),
            ("assignment_weights", allocation.assignment_weights, dense_shape),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                message = f"Allocation {name} has an invalid shape"
                raise WorkspaceWriterError(message)
            if tensor.dtype != dtype or tensor.device != device:
                message = "Allocation floating tensors must share one dtype and device"
                raise WorkspaceWriterError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Allocation {name} contains NaN or Inf"
                raise WorkspaceWriterError(message)
        if (
            not isinstance(allocation.target_mask, torch.Tensor)
            or tuple(allocation.target_mask.shape) != (batch_size, self.num_slots)
            or allocation.target_mask.dtype != torch.bool
            or allocation.target_mask.device != device
        ):
            message = "Allocation target mask must be boolean [batch, slots]"
            raise WorkspaceWriterError(message)
        if (
            not isinstance(allocation.assignment_mask, torch.Tensor)
            or tuple(allocation.assignment_mask.shape) != dense_shape
            or allocation.assignment_mask.dtype != torch.bool
            or allocation.assignment_mask.device != device
        ):
            message = "Allocation assignment mask must be boolean [batch, modules, slots]"
            raise WorkspaceWriterError(message)
        index_shape = (batch_size, self.winners_per_cycle)
        for name, tensor, upper_bound in (
            ("module_indices", allocation.module_indices, self.num_modules),
            ("slot_indices", allocation.slot_indices, self.num_slots),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != index_shape
                or tensor.dtype != torch.int64
                or tensor.device != device
            ):
                message = f"Allocation {name} must be int64 [batch, winners]"
                raise WorkspaceWriterError(message)
            if not bool(torch.logical_and(tensor >= 0, tensor < upper_bound).all()):
                message = f"Allocation {name} contains an out-of-range index"
                raise WorkspaceWriterError(message)

        hard = allocation.assignment_mask.to(dtype=dtype)
        if not torch.equal(allocation.assignment_weights, hard):
            message = "Allocation weights must have the hard assignment forward value"
            raise WorkspaceWriterError(message)
        row_mask = allocation.assignment_mask.any(dim=-1)
        column_mask = allocation.assignment_mask.any(dim=1)
        expected_count = torch.full(
            (batch_size,),
            self.winners_per_cycle,
            dtype=torch.int64,
            device=device,
        )
        if not torch.equal(allocation.assignment_mask.sum(dim=(1, 2)), expected_count):
            message = "Allocation must contain exactly one write per winner"
            raise WorkspaceWriterError(message)
        if bool((allocation.assignment_mask.sum(dim=-1) > 1).any()) or bool(
            (allocation.assignment_mask.sum(dim=1) > 1).any()
        ):
            message = "Allocation assignments must be one-to-one"
            raise WorkspaceWriterError(message)
        if not torch.equal(column_mask, allocation.target_mask):
            message = "Allocation target mask must equal assigned slot columns"
            raise WorkspaceWriterError(message)
        if not torch.equal(
            row_mask.gather(1, allocation.module_indices),
            torch.ones_like(allocation.module_indices, dtype=torch.bool),
        ):
            message = "Allocation module indices must identify assigned rows"
            raise WorkspaceWriterError(message)
        reconstructed_mask = torch.zeros_like(allocation.assignment_mask)
        reconstructed_mask[
            torch.arange(batch_size, device=device).unsqueeze(1),
            allocation.module_indices,
            allocation.slot_indices,
        ] = True
        if not torch.equal(reconstructed_mask, allocation.assignment_mask):
            message = "Allocation rank indices must exactly reconstruct the assigned pairs"
            raise WorkspaceWriterError(message)
        probability_sums = allocation.assignment_probabilities.sum(dim=-1)
        if not bool((allocation.assignment_probabilities >= 0).all()) or not torch.allclose(
            probability_sums,
            row_mask.to(dtype=dtype),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Winner allocation probabilities must sum to one"
            raise WorkspaceWriterError(message)
        if bool(
            (
                allocation.assignment_probabilities.masked_select(
                    ~allocation.target_mask.unsqueeze(1).expand_as(
                        allocation.assignment_probabilities
                    )
                )
                != 0
            ).any()
        ):
            message = "Allocation probabilities must be confined to target slots"
            raise WorkspaceWriterError(message)

    def _validate_goal(
        self,
        goal_state: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(goal_state, torch.Tensor) or tuple(goal_state.shape) != (
            batch_size,
            self.state_dim,
        ):
            message = "Goal state must have shape [batch, state_dim]"
            raise WorkspaceWriterError(message)
        if not torch.is_floating_point(goal_state):
            message = "Goal state must use a floating dtype"
            raise WorkspaceWriterError(message)
        if goal_state.dtype != dtype or goal_state.device != device:
            message = "Goal state must share the proposal dtype and device"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(goal_state).all()):
            message = "Goal state contains NaN or Inf"
            raise WorkspaceWriterError(message)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = "Writer parameters must share the proposal dtype and device"
                raise WorkspaceWriterError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"Writer parameter {name} contains NaN or Inf"
                raise WorkspaceWriterError(message)

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
            message = "Written workspace slots have an invalid shape"
            raise WorkspaceWriterError(message)
        if state.slots.dtype != dtype or state.slots.device != device:
            message = "Written workspace slots have an invalid dtype or device"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(state.slots).all()):
            message = "Written workspace slots contain NaN or Inf"
            raise WorkspaceWriterError(message)
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
                message = f"Written workspace {name} has an invalid shape, dtype, or device"
                raise WorkspaceWriterError(message)
        if not bool((state.age >= 0).all()) or not bool((state.write_count >= 0).all()):
            message = "Written workspace counters must be non-negative"
            raise WorkspaceWriterError(message)
        if not bool(
            torch.logical_and(
                state.source_module >= -1, state.source_module < self.num_modules
            ).all()
        ):
            message = "Written workspace source_module is out of range"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(state.confidence).all()) or not bool(
            torch.logical_and(state.confidence >= 0, state.confidence <= 1).all()
        ):
            message = "Written workspace confidence must lie in [0, 1]"
            raise WorkspaceWriterError(message)
        if not bool(torch.isfinite(state.ttl).all()) or not bool((state.ttl >= 0).all()):
            message = "Written workspace TTL must be finite and non-negative"
            raise WorkspaceWriterError(message)
        empty = ~state.occupancy
        if not bool((state.slots.masked_select(empty.unsqueeze(-1)) == 0).all()):
            message = "Unoccupied written slots must have zero content"
            raise WorkspaceWriterError(message)
        for name, tensor, expected in (
            ("age", state.age, 0),
            ("source_module", state.source_module, -1),
            ("confidence", state.confidence, 0),
            ("ttl", state.ttl, 0),
            ("write_count", state.write_count, 0),
        ):
            if not bool((tensor.masked_select(empty) == expected).all()):
                message = f"Unoccupied written slots must have canonical {name}"
                raise WorkspaceWriterError(message)

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
        """Apply LayerNorm without squaring the unscaled input magnitude."""
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
