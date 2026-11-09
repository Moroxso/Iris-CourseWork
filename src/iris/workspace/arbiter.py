"""Competitive proposal selection for the Iris global workspace."""

from math import isfinite

import torch
from torch import nn

from iris.cognitive.outputs import ModuleProposal
from iris.schemas import WorkspaceArbiterConfig
from iris.workspace.outputs import WorkspaceArbiterOutput
from iris.workspace.state import WorkspaceState


class WorkspaceArbiterError(RuntimeError):
    """Raised when proposals cannot satisfy the workspace competition contract."""


class WorkspaceArbiter(nn.Module):
    """Score module proposals and select deterministic hard top-k winners."""

    def __init__(
        self,
        config: WorkspaceArbiterConfig,
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
        self.confidence_weight = self._validate_score_weight(
            "confidence_weight", config.confidence_weight
        )
        self.novelty_weight = self._validate_score_weight("novelty_weight", config.novelty_weight)
        self.relevance_weight = self._validate_score_weight(
            "relevance_weight", config.relevance_weight
        )

    def forward(
        self,
        proposal: ModuleProposal,
        workspace: WorkspaceState,
        goal_state: torch.Tensor,
    ) -> WorkspaceArbiterOutput:
        """Run hard competition with a soft straight-through training gradient."""
        batch_size, dtype, device = self._validate_proposal(proposal)
        self._validate_workspace(
            workspace,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._validate_goal(
            goal_state,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        content = self._materialize(proposal.content)
        bid = self._materialize(proposal.bid)
        confidence = self._materialize(proposal.confidence)
        slots = self._materialize(workspace.slots)
        goal_state = self._materialize(goal_state)
        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

        with torch.autocast(device_type=device.type, enabled=False):
            normalized_content = self._safe_normalize(content.to(dtype=accumulation_dtype))
            normalized_goal = self._safe_normalize(goal_state.to(dtype=accumulation_dtype))
            normalized_slots = self._safe_normalize(slots.to(dtype=accumulation_dtype))
            relevance = torch.einsum("bmd,bd->bm", normalized_content, normalized_goal).clamp(
                min=-1.0, max=1.0
            )
            similarities = torch.einsum("bmd,bkd->bmk", normalized_content, normalized_slots).clamp(
                min=-1.0, max=1.0
            )
            occupancy = workspace.occupancy.unsqueeze(1)
            has_workspace = workspace.occupancy.any(dim=1)
            similarities = similarities.masked_fill(~occupancy, -2.0)
            similarities = torch.where(
                has_workspace[:, None, None],
                similarities,
                torch.zeros_like(similarities),
            )
            maximum_similarity = similarities.max(dim=-1).values
            novelty = torch.where(
                has_workspace[:, None],
                1.0 - maximum_similarity,
                torch.ones_like(maximum_similarity),
            )
            scores = (
                bid.to(dtype=accumulation_dtype)
                + self.confidence_weight * confidence.to(dtype=accumulation_dtype)
                + self.novelty_weight * novelty
                + self.relevance_weight * relevance
            )
            if not bool(torch.isfinite(scores).all()):
                message = "Workspace arbitration scores contain NaN or Inf"
                raise WorkspaceArbiterError(message)

            soft_probabilities = torch.softmax(scores, dim=-1)
            top_values = torch.topk(
                scores,
                k=self.winners_per_cycle,
                dim=-1,
                largest=True,
                sorted=True,
            ).values
            cutoff = top_values[:, -1:]
            above_cutoff = scores > cutoff
            remaining = self.winners_per_cycle - above_cutoff.sum(dim=-1)
            at_cutoff = scores == cutoff
            cutoff_rank = at_cutoff.to(dtype=torch.int64).cumsum(dim=-1)
            winner_mask = above_cutoff | (at_cutoff & (cutoff_rank <= remaining.unsqueeze(-1)))
            winner_indices = torch.argsort(
                scores,
                dim=-1,
                descending=True,
                stable=True,
            )[:, : self.winners_per_cycle]
            hard_weights = winner_mask.to(dtype=accumulation_dtype)
            if self.training:
                selection_weights = hard_weights + (
                    soft_probabilities - soft_probabilities.detach()
                )
            else:
                selection_weights = hard_weights

        output = WorkspaceArbiterOutput(
            scores=scores,
            novelty=novelty,
            relevance=relevance,
            soft_probabilities=soft_probabilities,
            selection_weights=selection_weights,
            winner_mask=winner_mask,
            winner_indices=winner_indices,
        )
        self._validate_output(
            output,
            batch_size=batch_size,
            dtype=accumulation_dtype,
            device=device,
        )
        return output

    @staticmethod
    def _validate_score_weight(name: str, value: float) -> float:
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
            message = "Workspace arbiter input must be ModuleProposal"
            raise WorkspaceArbiterError(message)
        content = proposal.content
        if not isinstance(content, torch.Tensor) or content.ndim != 3:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise WorkspaceArbiterError(message)
        batch_size = content.shape[0]
        vector_shape = (batch_size, self.num_modules, self.state_dim)
        scalar_shape = (batch_size, self.num_modules)
        if batch_size <= 0 or tuple(content.shape) != vector_shape:
            message = "Proposal content must have shape [batch, modules, state_dim]"
            raise WorkspaceArbiterError(message)
        for name, tensor, expected_shape in (
            ("content", proposal.content, vector_shape),
            ("key", proposal.key, vector_shape),
            ("bid", proposal.bid, scalar_shape),
            ("confidence", proposal.confidence, scalar_shape),
            ("ttl", proposal.ttl, scalar_shape),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                message = f"Proposal {name} has an invalid shape"
                raise WorkspaceArbiterError(message)
            if not torch.is_floating_point(tensor):
                message = f"Proposal {name} must use a floating dtype"
                raise WorkspaceArbiterError(message)
            if tensor.dtype != content.dtype or tensor.device != content.device:
                message = "Proposal tensors must share one dtype and device"
                raise WorkspaceArbiterError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Proposal {name} contains NaN or Inf"
                raise WorkspaceArbiterError(message)
        if not bool(torch.logical_and(proposal.confidence >= 0, proposal.confidence <= 1).all()):
            message = "Proposal confidence must lie in [0, 1]"
            raise WorkspaceArbiterError(message)
        if not bool((proposal.ttl >= 1).all()):
            message = "Proposal TTL must be at least one cycle"
            raise WorkspaceArbiterError(message)
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
            message = "Workspace arbiter input must be WorkspaceState"
            raise WorkspaceArbiterError(message)
        if not isinstance(workspace.slots, torch.Tensor) or tuple(workspace.slots.shape) != (
            batch_size,
            self.num_slots,
            self.state_dim,
        ):
            message = "Workspace slots must have shape [batch, slots, state_dim]"
            raise WorkspaceArbiterError(message)
        if not torch.is_floating_point(workspace.slots):
            message = "Workspace slots must use a floating dtype"
            raise WorkspaceArbiterError(message)
        if workspace.slots.dtype != dtype or workspace.slots.device != device:
            message = "Workspace slots must share the proposal dtype and device"
            raise WorkspaceArbiterError(message)
        if not bool(torch.isfinite(workspace.slots).all()):
            message = "Workspace slots contain NaN or Inf"
            raise WorkspaceArbiterError(message)
        if (
            not isinstance(workspace.occupancy, torch.Tensor)
            or tuple(workspace.occupancy.shape) != (batch_size, self.num_slots)
            or workspace.occupancy.dtype != torch.bool
            or workspace.occupancy.device != device
        ):
            message = "Workspace occupancy must be boolean [batch, slots] on the input device"
            raise WorkspaceArbiterError(message)
        matrix_shape = (batch_size, self.num_slots)
        for name, tensor, expected_dtype in (
            ("age", workspace.age, torch.int64),
            ("source_module", workspace.source_module, torch.int64),
            ("confidence", workspace.confidence, dtype),
            ("ttl", workspace.ttl, dtype),
            ("write_count", workspace.write_count, torch.int64),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != matrix_shape:
                message = f"Workspace {name} must have shape [batch, slots]"
                raise WorkspaceArbiterError(message)
            if tensor.dtype != expected_dtype or tensor.device != device:
                message = f"Workspace {name} has an invalid dtype or device"
                raise WorkspaceArbiterError(message)
        if not bool((workspace.age >= 0).all()) or not bool((workspace.write_count >= 0).all()):
            message = "Workspace counters must be non-negative"
            raise WorkspaceArbiterError(message)
        if not bool(
            torch.logical_and(
                workspace.source_module >= -1,
                workspace.source_module < self.num_modules,
            ).all()
        ):
            message = "Workspace source_module must be -1 or a valid module index"
            raise WorkspaceArbiterError(message)
        if not bool(torch.isfinite(workspace.confidence).all()) or not bool(
            torch.logical_and(workspace.confidence >= 0, workspace.confidence <= 1).all()
        ):
            message = "Workspace confidence must be finite and lie in [0, 1]"
            raise WorkspaceArbiterError(message)
        if not bool(torch.isfinite(workspace.ttl).all()) or not bool((workspace.ttl >= 0).all()):
            message = "Workspace TTL must be finite and non-negative"
            raise WorkspaceArbiterError(message)
        if not bool((workspace.ttl.masked_select(~workspace.occupancy) == 0).all()):
            message = "Unoccupied workspace slots must have zero TTL"
            raise WorkspaceArbiterError(message)

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
            raise WorkspaceArbiterError(message)
        if not torch.is_floating_point(goal_state):
            message = "Goal state must use a floating dtype"
            raise WorkspaceArbiterError(message)
        if goal_state.dtype != dtype or goal_state.device != device:
            message = "Goal state must share the proposal dtype and device"
            raise WorkspaceArbiterError(message)
        if not bool(torch.isfinite(goal_state).all()):
            message = "Goal state contains NaN or Inf"
            raise WorkspaceArbiterError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor

    @staticmethod
    def _safe_normalize(tensor: torch.Tensor) -> torch.Tensor:
        maximum = tensor.abs().amax(dim=-1, keepdim=True)
        nonzero = maximum > 0
        safe_maximum = torch.where(nonzero, maximum, torch.ones_like(maximum))
        scaled = tensor / safe_maximum
        norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        safe_norm = torch.where(nonzero, norm, torch.ones_like(norm))
        normalized = scaled / safe_norm
        return torch.where(nonzero, normalized, torch.zeros_like(normalized))

    def _validate_output(
        self,
        output: WorkspaceArbiterOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        matrix_shape = (batch_size, self.num_modules)
        for name, tensor in (
            ("scores", output.scores),
            ("novelty", output.novelty),
            ("relevance", output.relevance),
            ("soft_probabilities", output.soft_probabilities),
            ("selection_weights", output.selection_weights),
        ):
            if tuple(tensor.shape) != matrix_shape:
                message = f"Arbiter {name} must have shape [batch, modules]"
                raise WorkspaceArbiterError(message)
            if tensor.dtype != dtype or tensor.device != device:
                message = "Arbiter floating outputs must share one dtype and device"
                raise WorkspaceArbiterError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Arbiter {name} contains NaN or Inf"
                raise WorkspaceArbiterError(message)
        if not bool(torch.logical_and(output.novelty >= 0, output.novelty <= 2).all()):
            message = "Proposal novelty must lie in [0, 2]"
            raise WorkspaceArbiterError(message)
        if not bool(torch.logical_and(output.relevance >= -1, output.relevance <= 1).all()):
            message = "Proposal relevance must lie in [-1, 1]"
            raise WorkspaceArbiterError(message)
        if not bool((output.soft_probabilities >= 0).all()):
            message = "Soft competition probabilities must be non-negative"
            raise WorkspaceArbiterError(message)
        expected_probability_sums = torch.ones(batch_size, dtype=dtype, device=device)
        if not torch.allclose(
            output.soft_probabilities.sum(dim=-1),
            expected_probability_sums,
            rtol=1e-4,
            atol=1e-6,
        ):
            message = "Soft competition probabilities must sum to one"
            raise WorkspaceArbiterError(message)
        if (
            tuple(output.winner_mask.shape) != matrix_shape
            or output.winner_mask.dtype != torch.bool
            or output.winner_mask.device != device
        ):
            message = "Winner mask must be boolean [batch, modules]"
            raise WorkspaceArbiterError(message)
        index_shape = (batch_size, self.winners_per_cycle)
        if (
            tuple(output.winner_indices.shape) != index_shape
            or output.winner_indices.dtype != torch.int64
            or output.winner_indices.device != device
        ):
            message = "Winner indices must be int64 [batch, winners]"
            raise WorkspaceArbiterError(message)
        expected_winner_counts = torch.full(
            (batch_size,),
            self.winners_per_cycle,
            dtype=torch.int64,
            device=device,
        )
        if not torch.equal(output.winner_mask.sum(dim=-1), expected_winner_counts):
            message = "Winner mask must select exactly winners_per_cycle modules"
            raise WorkspaceArbiterError(message)
        if not torch.equal(
            output.selection_weights,
            output.winner_mask.to(dtype=dtype),
        ):
            message = "Straight-through forward weights must equal the hard winner mask"
            raise WorkspaceArbiterError(message)
        if not bool(output.winner_mask.gather(1, output.winner_indices).all()):
            message = "Winner indices and mask must describe the same modules"
            raise WorkspaceArbiterError(message)
