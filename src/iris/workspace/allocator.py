"""Deterministic multi-winner slot allocation for the Iris workspace."""

from math import isfinite, sqrt

import torch
import torch.nn.functional as functional
from torch import nn

from iris.cognitive.outputs import ModuleProposal
from iris.schemas import SlotAllocatorConfig
from iris.workspace.outputs import SlotAllocatorOutput, WorkspaceArbiterOutput
from iris.workspace.state import WorkspaceState


class SlotAllocatorError(RuntimeError):
    """Raised when winners cannot be assigned to valid workspace slots."""


class SlotAllocator(nn.Module):
    """Assign hard winners to unique slots with a soft training surrogate.

    Empty slots form the target set first. If fewer than ``R`` empty slots are
    available, the remaining targets are the occupied slots with the lowest
    confidence-minus-age retention. Winner order comes from WorkspaceArbiter;
    each winner greedily selects one remaining target by the recovered
    compatibility score.
    """

    def __init__(
        self,
        config: SlotAllocatorConfig,
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
        self.empty_bonus = self._validate_coefficient("empty_bonus", config.empty_bonus)
        self.allocation_age_penalty = self._validate_coefficient(
            "allocation_age_penalty", config.allocation_age_penalty
        )
        self.retention_age_penalty = self._validate_coefficient(
            "retention_age_penalty", config.retention_age_penalty
        )
        self.slot_projection = nn.Linear(state_dim, state_dim, bias=False)

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        arbitration: WorkspaceArbiterOutput,
    ) -> SlotAllocatorOutput:
        """Allocate exactly one unique target slot to every hard winner."""
        batch_size, dtype, device = self._validate_proposal(proposal)
        self._validate_workspace(
            workspace,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        self._validate_arbitration(
            arbitration,
            batch_size=batch_size,
            dtype=accumulation_dtype,
            device=device,
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        keys = self._materialize(proposal.key)
        slots = self._materialize(workspace.slots)
        confidence = self._materialize(workspace.confidence)
        winner_indices = self._materialize(arbitration.winner_indices)
        selection_weights = self._materialize(arbitration.selection_weights)

        occupied_slots = torch.where(
            workspace.occupancy.unsqueeze(-1),
            slots,
            torch.zeros_like(slots),
        )

        with torch.autocast(device_type=device.type, enabled=False):
            safe_keys = keys.to(dtype=accumulation_dtype) / sqrt(self.state_dim)
            projected_slots = functional.linear(
                occupied_slots.to(dtype=accumulation_dtype),
                self.slot_projection.weight.to(dtype=accumulation_dtype),
            )
            compatibility = torch.einsum("bmd,bkd->bmk", safe_keys, projected_slots)
            age = torch.where(
                workspace.occupancy,
                workspace.age,
                torch.zeros_like(workspace.age),
            ).to(dtype=accumulation_dtype)
            confidence = torch.where(
                workspace.occupancy,
                confidence,
                torch.zeros_like(confidence),
            )
            empty = ~workspace.occupancy
            allocation_scores = (
                compatibility
                + self.empty_bonus * empty.unsqueeze(1).to(dtype=accumulation_dtype)
                - self.allocation_age_penalty * age.unsqueeze(1)
            )
            retention_scores = confidence.to(dtype=accumulation_dtype) - (
                self.retention_age_penalty * age
            )
            if not bool(torch.isfinite(allocation_scores).all()):
                message = "Slot allocation scores contain NaN or Inf"
                raise SlotAllocatorError(message)
            if not bool(torch.isfinite(retention_scores).all()):
                message = "Slot retention scores contain NaN or Inf"
                raise SlotAllocatorError(message)

            target_mask = self._select_target_slots(
                occupancy=workspace.occupancy,
                retention_scores=retention_scores,
            )
            (
                assignment_mask,
                assignment_probabilities,
                assignment_weights,
                slot_indices,
            ) = self._assign_winners(
                allocation_scores=allocation_scores,
                target_mask=target_mask,
                winner_indices=winner_indices,
                selection_weights=selection_weights,
            )

        output = SlotAllocatorOutput(
            allocation_scores=allocation_scores,
            retention_scores=retention_scores,
            target_mask=target_mask,
            assignment_mask=assignment_mask,
            assignment_probabilities=assignment_probabilities,
            assignment_weights=assignment_weights,
            module_indices=winner_indices,
            slot_indices=slot_indices,
        )
        self._validate_output(
            output,
            arbitration=arbitration,
            batch_size=batch_size,
            dtype=accumulation_dtype,
            device=device,
        )
        return output

    def _select_target_slots(
        self,
        *,
        occupancy: torch.Tensor,
        retention_scores: torch.Tensor,
    ) -> torch.Tensor:
        empty = ~occupancy
        empty_rank = empty.to(dtype=torch.int64).cumsum(dim=-1)
        selected_empty = empty & (empty_rank <= self.winners_per_cycle)
        remaining = self.winners_per_cycle - selected_empty.sum(dim=-1)

        occupied_retention = retention_scores.masked_fill(~occupancy, torch.inf)
        occupied_order = torch.argsort(
            occupied_retention,
            dim=-1,
            descending=False,
            stable=True,
        )
        order_positions = torch.arange(
            self.num_slots,
            dtype=torch.int64,
            device=occupancy.device,
        ).expand_as(occupied_order)
        selected_in_order = order_positions < remaining.unsqueeze(-1)
        selected_occupied = torch.zeros_like(occupancy).scatter(
            dim=1,
            index=occupied_order,
            src=selected_in_order,
        )
        return selected_empty | (selected_occupied & occupancy)

    def _assign_winners(
        self,
        *,
        allocation_scores: torch.Tensor,
        target_mask: torch.Tensor,
        winner_indices: torch.Tensor,
        selection_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        remaining_targets = target_mask
        probabilities: list[torch.Tensor] = []
        slot_weights: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []

        for winner_position in range(self.winners_per_cycle):
            module_index = winner_indices[:, winner_position]
            scores = allocation_scores.gather(
                dim=1,
                index=module_index[:, None, None].expand(-1, 1, self.num_slots),
            ).squeeze(1)
            masked_scores = scores.masked_fill(~remaining_targets, -torch.inf)
            soft = torch.softmax(masked_scores, dim=-1)
            slot_index = torch.argmax(masked_scores, dim=-1)
            hard = functional.one_hot(slot_index, num_classes=self.num_slots).to(
                dtype=allocation_scores.dtype
            )
            assignment = hard + (soft - soft.detach()) if self.training else hard
            probabilities.append(soft)
            slot_weights.append(assignment)
            indices.append(slot_index)
            remaining_targets = remaining_targets & ~hard.to(dtype=torch.bool)

        rank_probabilities = torch.stack(probabilities, dim=1)
        rank_slot_weights = torch.stack(slot_weights, dim=1)
        slot_indices = torch.stack(indices, dim=1)
        module_selector = functional.one_hot(
            winner_indices,
            num_classes=self.num_modules,
        ).to(dtype=allocation_scores.dtype)
        hard_slots = functional.one_hot(slot_indices, num_classes=self.num_slots).to(
            dtype=allocation_scores.dtype
        )
        assignment_mask = torch.einsum("brm,brk->bmk", module_selector, hard_slots).to(
            dtype=torch.bool
        )
        assignment_probabilities = torch.einsum(
            "brm,brk->bmk",
            module_selector,
            rank_probabilities,
        )
        if self.training:
            winner_weights = selection_weights.gather(1, winner_indices)
            combined_rank_weights = rank_slot_weights * winner_weights.unsqueeze(-1)
            assignment_weights = torch.einsum(
                "brm,brk->bmk",
                module_selector,
                combined_rank_weights,
            )
        else:
            assignment_weights = assignment_mask.to(dtype=allocation_scores.dtype)
        return (
            assignment_mask,
            assignment_probabilities,
            assignment_weights,
            slot_indices,
        )

    @staticmethod
    def _validate_coefficient(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = f"{name} must be a finite non-negative number"
            raise ValueError(message)
        result = float(value)
        if not isfinite(result) or result < 0:
            message = f"{name} must be a finite non-negative number"
            raise ValueError(message)
        return result

    def _validate_proposal(self, proposal: ModuleProposal) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(proposal, ModuleProposal):
            message = "Slot allocator input must be ModuleProposal"
            raise SlotAllocatorError(message)
        content = proposal.content
        if not isinstance(content, torch.Tensor) or content.ndim != 3:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise SlotAllocatorError(message)
        batch_size = content.shape[0]
        vector_shape = (batch_size, self.num_modules, self.state_dim)
        scalar_shape = (batch_size, self.num_modules)
        if batch_size <= 0 or tuple(content.shape) != vector_shape:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise SlotAllocatorError(message)
        for name, tensor, expected_shape in (
            ("content", proposal.content, vector_shape),
            ("key", proposal.key, vector_shape),
            ("bid", proposal.bid, scalar_shape),
            ("confidence", proposal.confidence, scalar_shape),
            ("ttl", proposal.ttl, scalar_shape),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                message = f"Proposal {name} has an invalid shape"
                raise SlotAllocatorError(message)
            if not torch.is_floating_point(tensor):
                message = f"Proposal {name} must use a floating dtype"
                raise SlotAllocatorError(message)
            if tensor.dtype != content.dtype or tensor.device != content.device:
                message = "Proposal tensors must share one dtype and device"
                raise SlotAllocatorError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Proposal {name} contains NaN or Inf"
                raise SlotAllocatorError(message)
        if not bool(torch.logical_and(proposal.confidence >= 0, proposal.confidence <= 1).all()):
            message = "Proposal confidence must lie in [0, 1]"
            raise SlotAllocatorError(message)
        if not bool((proposal.ttl >= 1).all()):
            message = "Proposal TTL must be at least one cycle"
            raise SlotAllocatorError(message)
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
            message = "Slot allocator input must be WorkspaceState"
            raise SlotAllocatorError(message)
        slot_shape = (batch_size, self.num_slots)
        if not isinstance(workspace.slots, torch.Tensor) or tuple(workspace.slots.shape) != (
            *slot_shape,
            self.state_dim,
        ):
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise SlotAllocatorError(message)
        if not torch.is_floating_point(workspace.slots):
            message = "Workspace slots must use a floating dtype"
            raise SlotAllocatorError(message)
        if workspace.slots.dtype != dtype or workspace.slots.device != device:
            message = "Workspace slots must share the proposal dtype and device"
            raise SlotAllocatorError(message)
        if not bool(torch.isfinite(workspace.slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise SlotAllocatorError(message)
        if (
            not isinstance(workspace.occupancy, torch.Tensor)
            or tuple(workspace.occupancy.shape) != slot_shape
            or workspace.occupancy.dtype != torch.bool
            or workspace.occupancy.device != device
        ):
            message = "Workspace occupancy must be boolean [batch, slots] on the input device"
            raise SlotAllocatorError(message)
        for name, tensor, expected_dtype in (
            ("age", workspace.age, torch.int64),
            ("source_module", workspace.source_module, torch.int64),
            ("confidence", workspace.confidence, dtype),
            ("ttl", workspace.ttl, dtype),
            ("write_count", workspace.write_count, torch.int64),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != slot_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise SlotAllocatorError(message)
            if tensor.dtype != expected_dtype or tensor.device != device:
                message = f"Workspace {name} has an invalid dtype or device"
                raise SlotAllocatorError(message)
        if not bool((workspace.age >= 0).all()) or not bool((workspace.write_count >= 0).all()):
            message = "Workspace counters must be non-negative"
            raise SlotAllocatorError(message)
        if not bool(
            torch.logical_and(
                workspace.source_module >= -1,
                workspace.source_module < self.num_modules,
            ).all()
        ):
            message = "Workspace source_module must be -1 or a valid module index"
            raise SlotAllocatorError(message)
        if not bool(torch.isfinite(workspace.confidence).all()) or not bool(
            torch.logical_and(workspace.confidence >= 0, workspace.confidence <= 1).all()
        ):
            message = "Workspace confidence must be finite and lie in [0, 1]"
            raise SlotAllocatorError(message)
        if not bool(torch.isfinite(workspace.ttl).all()) or not bool((workspace.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise SlotAllocatorError(message)
        if not bool((workspace.ttl.masked_select(~workspace.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise SlotAllocatorError(message)

    def _validate_arbitration(
        self,
        arbitration: WorkspaceArbiterOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(arbitration, WorkspaceArbiterOutput):
            message = "Slot allocator input must be WorkspaceArbiterOutput"
            raise SlotAllocatorError(message)
        matrix_shape = (batch_size, self.num_modules)
        for name, tensor in (
            ("scores", arbitration.scores),
            ("novelty", arbitration.novelty),
            ("relevance", arbitration.relevance),
            ("soft_probabilities", arbitration.soft_probabilities),
            ("selection_weights", arbitration.selection_weights),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != matrix_shape:
                message = f"Arbitration {name} must have shape [batch, modules]"
                raise SlotAllocatorError(message)
            if tensor.dtype != dtype or tensor.device != device:
                message = "Arbitration floating tensors must share one dtype and device"
                raise SlotAllocatorError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Arbitration {name} contains NaN or Inf"
                raise SlotAllocatorError(message)
        if (
            not isinstance(arbitration.winner_mask, torch.Tensor)
            or tuple(arbitration.winner_mask.shape) != matrix_shape
            or arbitration.winner_mask.dtype != torch.bool
            or arbitration.winner_mask.device != device
        ):
            message = "Winner mask must be boolean [batch, modules]"
            raise SlotAllocatorError(message)
        expected_index_shape = (batch_size, self.winners_per_cycle)
        if (
            not isinstance(arbitration.winner_indices, torch.Tensor)
            or tuple(arbitration.winner_indices.shape) != expected_index_shape
            or arbitration.winner_indices.dtype != torch.int64
            or arbitration.winner_indices.device != device
        ):
            message = "Winner indices must be int64 [batch, winners]"
            raise SlotAllocatorError(message)
        expected_indices = torch.argsort(
            arbitration.scores,
            dim=-1,
            descending=True,
            stable=True,
        )[:, : self.winners_per_cycle]
        if not torch.equal(arbitration.winner_indices, expected_indices):
            message = "Winner indices must follow deterministic descending score order"
            raise SlotAllocatorError(message)
        expected_mask = torch.zeros_like(arbitration.winner_mask).scatter(
            dim=1,
            index=expected_indices,
            src=torch.ones_like(expected_indices, dtype=torch.bool),
        )
        if not torch.equal(arbitration.winner_mask, expected_mask):
            message = "Winner mask and indices must identify the same modules"
            raise SlotAllocatorError(message)
        if not torch.equal(
            arbitration.selection_weights,
            arbitration.winner_mask.to(dtype=dtype),
        ):
            message = "Arbitration selection weights must have a hard forward value"
            raise SlotAllocatorError(message)
        if not bool((arbitration.soft_probabilities >= 0).all()) or not torch.allclose(
            arbitration.soft_probabilities.sum(dim=-1),
            torch.ones(batch_size, dtype=dtype, device=device),
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Arbitration soft probabilities must form a distribution"
            raise SlotAllocatorError(message)

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        weight = self.slot_projection.weight
        if weight.dtype != dtype or weight.device != device:
            message = "Allocator parameters must share the proposal dtype and device"
            raise SlotAllocatorError(message)
        if not bool(torch.isfinite(weight).all()):
            message = "Allocator parameters contain NaN or Inf"
            raise SlotAllocatorError(message)

    def _validate_output(
        self,
        output: SlotAllocatorOutput,
        *,
        arbitration: WorkspaceArbiterOutput,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assignment_shape = (batch_size, self.num_modules, self.num_slots)
        for name, tensor, expected_shape in (
            ("allocation_scores", output.allocation_scores, assignment_shape),
            ("retention_scores", output.retention_scores, (batch_size, self.num_slots)),
            ("assignment_probabilities", output.assignment_probabilities, assignment_shape),
            ("assignment_weights", output.assignment_weights, assignment_shape),
        ):
            if (
                tuple(tensor.shape) != expected_shape
                or tensor.dtype != dtype
                or tensor.device != device
            ):
                message = f"Allocator {name} has an invalid shape, dtype, or device"
                raise SlotAllocatorError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Allocator {name} contains NaN or Inf"
                raise SlotAllocatorError(message)
        if (
            tuple(output.target_mask.shape) != (batch_size, self.num_slots)
            or output.target_mask.dtype != torch.bool
            or output.target_mask.device != device
        ):
            message = "Allocator target mask must be boolean [batch, slots]"
            raise SlotAllocatorError(message)
        if not torch.equal(
            output.target_mask.sum(dim=-1),
            torch.full(
                (batch_size,),
                self.winners_per_cycle,
                dtype=torch.int64,
                device=device,
            ),
        ):
            message = "Allocator target mask must contain exactly one target per winner"
            raise SlotAllocatorError(message)
        index_shape = (batch_size, self.winners_per_cycle)
        for name, tensor in (
            ("module_indices", output.module_indices),
            ("slot_indices", output.slot_indices),
        ):
            if (
                tuple(tensor.shape) != index_shape
                or tensor.dtype != torch.int64
                or tensor.device != device
            ):
                message = f"Allocator {name} must be int64 [batch, winners]"
                raise SlotAllocatorError(message)
        if not torch.equal(output.module_indices, arbitration.winner_indices):
            message = "Allocator module indices must preserve arbiter winner order"
            raise SlotAllocatorError(message)
        if not bool(
            torch.logical_and(output.slot_indices >= 0, output.slot_indices < self.num_slots).all()
        ):
            message = "Allocated slot index is out of range"
            raise SlotAllocatorError(message)
        if not bool(output.target_mask.gather(1, output.slot_indices).all()):
            message = "Every allocated slot must belong to the target set"
            raise SlotAllocatorError(message)
        sorted_indices = torch.sort(output.slot_indices, dim=-1).values
        if self.winners_per_cycle > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            message = "Allocated slot indices must be unique within each batch item"
            raise SlotAllocatorError(message)
        if (
            tuple(output.assignment_mask.shape) != assignment_shape
            or output.assignment_mask.dtype != torch.bool
            or output.assignment_mask.device != device
        ):
            message = "Allocator assignment mask must be boolean [batch, modules, slots]"
            raise SlotAllocatorError(message)
        if not torch.equal(output.assignment_mask.any(dim=-1), arbitration.winner_mask):
            message = "Assignment rows must exactly match the arbiter winner mask"
            raise SlotAllocatorError(message)
        if not torch.equal(output.assignment_mask.sum(dim=1).to(torch.bool), output.target_mask):
            message = "Assignment columns must exactly match the reserved target slots"
            raise SlotAllocatorError(message)
        expected_probability_sums = arbitration.winner_mask.to(dtype=dtype)
        if not bool((output.assignment_probabilities >= 0).all()) or not torch.allclose(
            output.assignment_probabilities.sum(dim=-1),
            expected_probability_sums,
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Winner assignment probabilities must sum to one"
            raise SlotAllocatorError(message)
        if not torch.equal(output.assignment_weights, output.assignment_mask.to(dtype=dtype)):
            message = "Straight-through assignment weights must have a hard forward value"
            raise SlotAllocatorError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor
