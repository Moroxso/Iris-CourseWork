"""Latent proposal heads for Iris cognitive modules."""

import torch
from torch import nn
from torch.nn import functional

from iris.cognitive.outputs import ModuleProposal


class ProposalHeadError(RuntimeError):
    """Raised when module states cannot satisfy the proposal contract."""


class ProposalHead(nn.Module):
    """Map every module state to a latent workspace proposal."""

    def __init__(self, *, state_dim: int, num_modules: int, cycles: int) -> None:
        super().__init__()
        for name, value in {
            "state_dim": state_dim,
            "num_modules": num_modules,
            "cycles": cycles,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.cycles = cycles

        # The architecture specifies shared W_v/W_k/W_b/W_c/W_t maps. Biases
        # would add terms absent from those equations, so every head is bias-free.
        self.content_head = nn.Linear(state_dim, state_dim, bias=False)
        self.key_head = nn.Linear(state_dim, state_dim, bias=False)
        self.bid_head = nn.Linear(state_dim, 1, bias=False)
        self.confidence_head = nn.Linear(state_dim, 1, bias=False)
        self.ttl_head = nn.Linear(state_dim, 1, bias=False)

    def forward(self, module_states: torch.Tensor) -> ModuleProposal:
        """Create one continuous latent proposal per module."""
        batch_size, dtype, device = self._validate_input(module_states)
        self._validate_parameter_boundary(dtype=dtype, device=device)
        if module_states.is_inference() and not torch.is_inference_mode_enabled():
            module_states = module_states.detach().clone()

        accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        with torch.autocast(device_type=device.type, enabled=False):
            safe_states = module_states.to(dtype=accumulation_dtype)
            raw_content = self._project(safe_states, self.content_head, accumulation_dtype)
            raw_key = self._project(safe_states, self.key_head, accumulation_dtype)
            raw_bid = self._project(safe_states, self.bid_head, accumulation_dtype).squeeze(-1)
            confidence_logits = self._project(
                safe_states, self.confidence_head, accumulation_dtype
            ).squeeze(-1)
            ttl_logits = self._project(safe_states, self.ttl_head, accumulation_dtype).squeeze(-1)
            for name, tensor in (
                ("content", raw_content),
                ("key", raw_key),
                ("bid", raw_bid),
                ("confidence", confidence_logits),
                ("ttl", ttl_logits),
            ):
                if not bool(torch.isfinite(tensor).all()):
                    message = f"Proposal {name} logits contain NaN or Inf"
                    raise ProposalHeadError(message)
            content = raw_content.to(dtype=dtype)
            key = raw_key.to(dtype=dtype)
            bid = raw_bid.to(dtype=dtype)
            confidence = torch.sigmoid(confidence_logits).to(dtype=dtype)
            ttl = (1.0 + (self.cycles - 1) * torch.sigmoid(ttl_logits)).to(dtype=dtype)

        proposal = ModuleProposal(
            content=content,
            key=key,
            bid=bid,
            confidence=confidence,
            ttl=ttl,
        )
        self._validate_output(
            proposal,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return proposal

    @staticmethod
    def _project(
        inputs: torch.Tensor,
        head: nn.Linear,
        accumulation_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = functional.linear(
            inputs,
            head.weight.to(dtype=accumulation_dtype),
        )
        if not isinstance(output, torch.Tensor):
            message = "Proposal projection must return a tensor"
            raise ProposalHeadError(message)
        return output

    def _validate_input(self, module_states: torch.Tensor) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(module_states, torch.Tensor) or module_states.ndim != 3:
            message = "Module states must have shape [batch, modules, state_dim]"
            raise ProposalHeadError(message)
        batch_size = module_states.shape[0]
        if batch_size <= 0 or tuple(module_states.shape) != (
            batch_size,
            self.num_modules,
            self.state_dim,
        ):
            message = "Module states must have shape [batch, modules, state_dim]"
            raise ProposalHeadError(message)
        if not torch.is_floating_point(module_states):
            message = "Module states must use a floating dtype"
            raise ProposalHeadError(message)
        if not bool(torch.isfinite(module_states).all()):
            message = "Module states contain NaN or Inf"
            raise ProposalHeadError(message)
        return batch_size, module_states.dtype, module_states.device

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = (
                    "ProposalHead parameters must explicitly match the module-state "
                    "dtype and device"
                )
                raise ProposalHeadError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"ProposalHead parameter {name!r} contains NaN or Inf"
                raise ProposalHeadError(message)

    def _validate_output(
        self,
        proposal: ModuleProposal,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        vector_shape = (batch_size, self.num_modules, self.state_dim)
        scalar_shape = (batch_size, self.num_modules)
        for name, tensor, expected_shape in (
            ("content", proposal.content, vector_shape),
            ("key", proposal.key, vector_shape),
            ("bid", proposal.bid, scalar_shape),
            ("confidence", proposal.confidence, scalar_shape),
            ("ttl", proposal.ttl, scalar_shape),
        ):
            if tuple(tensor.shape) != expected_shape:
                message = f"Proposal {name} has an invalid shape"
                raise ProposalHeadError(message)
            if tensor.dtype != dtype or tensor.device != device:
                message = "Proposal tensors must preserve module-state dtype and device"
                raise ProposalHeadError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"Proposal {name} contains NaN or Inf"
                raise ProposalHeadError(message)
        if not bool(torch.logical_and(proposal.confidence >= 0, proposal.confidence <= 1).all()):
            message = "Proposal confidence must lie in [0, 1]"
            raise ProposalHeadError(message)
        if not bool(torch.logical_and(proposal.ttl >= 1, proposal.ttl <= self.cycles).all()):
            message = "Proposal TTL must lie in [1, cycles]"
            raise ProposalHeadError(message)
