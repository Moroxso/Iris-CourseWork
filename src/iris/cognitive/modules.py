"""Independent recurrent cognitive modules for Iris-GW."""

import torch
from torch import nn

from iris.cognitive.outputs import CognitiveModuleOutput


class CognitiveModuleError(RuntimeError):
    """Raised when cognitive module inputs violate the recurrent-state contract."""


class _CognitiveModuleCell(nn.Module):
    """One parameter-independent recurrent cell in the cognitive module bank."""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        fused_dim = 4 * state_dim
        self.identity_embedding = nn.Parameter(torch.empty(state_dim))
        nn.init.normal_(self.identity_embedding, mean=0.0, std=state_dim**-0.5)
        self.input_norm = nn.LayerNorm(fused_dim)
        self.input_projection = nn.Linear(fused_dim, state_dim)
        self.activation = nn.SiLU()
        self.recurrent_cell = nn.GRUCell(state_dim, state_dim)
        self.output_projection = nn.Linear(state_dim, state_dim)

    def forward(
        self,
        prompt_context: torch.Tensor,
        workspace_read: torch.Tensor,
        goal_state: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one cell, using the projected result as its next local state."""
        identity = self.identity_embedding.unsqueeze(0).expand(prompt_context.shape[0], -1)
        fused = torch.cat((prompt_context, workspace_read, goal_state, identity), dim=-1)
        prepared = self.activation(self.input_projection(self.input_norm(fused)))
        recurrent_state = self.recurrent_cell(prepared, previous_state)
        next_state = self.output_projection(recurrent_state)
        if not isinstance(next_state, torch.Tensor):
            message = "Cognitive module output projection must return a tensor"
            raise CognitiveModuleError(message)
        return next_state


class CognitiveModule(nn.Module):
    """Run architecturally identical modules with independent parameters."""

    def __init__(self, *, state_dim: int, num_modules: int) -> None:
        super().__init__()
        for name, value in {"state_dim": state_dim, "num_modules": num_modules}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.module_cells = nn.ModuleList(
            _CognitiveModuleCell(state_dim=state_dim) for _ in range(num_modules)
        )

    def initial_state(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create explicit zero local state for the first cognitive cycle."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            message = "batch_size must be a positive integer"
            raise ValueError(message)
        if not isinstance(dtype, torch.dtype):
            message = "dtype must be an explicit torch.dtype"
            raise ValueError(message)
        if not isinstance(device, torch.device):
            message = "device must be an explicit torch.device"
            raise ValueError(message)
        if not torch.empty((), dtype=dtype).is_floating_point():
            message = "Cognitive module state dtype must be floating point"
            raise ValueError(message)
        return torch.zeros(
            (batch_size, self.num_modules, self.state_dim),
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None = None,
    ) -> CognitiveModuleOutput:
        """Advance every module by one cognitive cycle."""
        batch_size, dtype, device = self._validate_inputs(
            module_inputs, workspace_reads, goal_state, previous_states
        )
        self._validate_parameter_boundary(dtype=dtype, device=device)

        materialized = tuple(
            tensor.detach().clone()
            if tensor.is_inference() and not torch.is_inference_mode_enabled()
            else tensor
            for tensor in (module_inputs, workspace_reads, goal_state)
        )
        module_inputs, workspace_reads, goal_state = materialized
        if previous_states is None:
            previous_states = module_inputs.new_zeros(
                (batch_size, self.num_modules, self.state_dim)
            )
        elif previous_states.is_inference() and not torch.is_inference_mode_enabled():
            previous_states = previous_states.detach().clone()

        with torch.autocast(device_type=device.type, enabled=False):
            next_states = torch.stack(
                [
                    cell(
                        module_inputs[:, index],
                        workspace_reads[:, index],
                        goal_state,
                        previous_states[:, index],
                    )
                    for index, cell in enumerate(self.module_cells)
                ],
                dim=1,
            )
        self._validate_output(
            next_states,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return CognitiveModuleOutput(module_states=next_states)

    def _validate_inputs(
        self,
        module_inputs: torch.Tensor,
        workspace_reads: torch.Tensor,
        goal_state: torch.Tensor,
        previous_states: torch.Tensor | None,
    ) -> tuple[int, torch.dtype, torch.device]:
        if not isinstance(module_inputs, torch.Tensor) or module_inputs.ndim != 3:
            message = "Module inputs must have shape [batch, modules, state_dim]"
            raise CognitiveModuleError(message)
        batch_size = module_inputs.shape[0]
        batched_shape = (batch_size, self.num_modules, self.state_dim)
        if batch_size <= 0 or tuple(module_inputs.shape) != batched_shape:
            message = "Module inputs must have shape [batch, modules, state_dim]"
            raise CognitiveModuleError(message)
        if (
            not isinstance(workspace_reads, torch.Tensor)
            or tuple(workspace_reads.shape) != batched_shape
        ):
            message = "Workspace reads must have shape [batch, modules, state_dim]"
            raise CognitiveModuleError(message)
        if not isinstance(goal_state, torch.Tensor) or tuple(goal_state.shape) != (
            batch_size,
            self.state_dim,
        ):
            message = "Goal state must have shape [batch, state_dim]"
            raise CognitiveModuleError(message)
        if previous_states is not None and (
            not isinstance(previous_states, torch.Tensor)
            or tuple(previous_states.shape) != batched_shape
        ):
            message = "Previous states must have shape [batch, modules, state_dim]"
            raise CognitiveModuleError(message)

        named_inputs = {
            "module_inputs": module_inputs,
            "workspace_reads": workspace_reads,
            "goal_state": goal_state,
        }
        if previous_states is not None:
            named_inputs["previous_states"] = previous_states
        for name, tensor in named_inputs.items():
            if not torch.is_floating_point(tensor):
                message = f"{name} must use a floating dtype"
                raise CognitiveModuleError(message)
            if tensor.dtype != module_inputs.dtype or tensor.device != module_inputs.device:
                message = "Cognitive module inputs must share one dtype and device"
                raise CognitiveModuleError(message)
            if not bool(torch.isfinite(tensor).all()):
                message = f"{name} contains NaN or Inf"
                raise CognitiveModuleError(message)
        return batch_size, module_inputs.dtype, module_inputs.device

    def _validate_parameter_boundary(self, *, dtype: torch.dtype, device: torch.device) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != dtype or parameter.device != device:
                message = (
                    "CognitiveModule parameters must explicitly match the input dtype and device"
                )
                raise CognitiveModuleError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"CognitiveModule parameter {name!r} contains NaN or Inf"
                raise CognitiveModuleError(message)

    def _validate_output(
        self,
        module_states: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(module_states.shape) != (
            batch_size,
            self.num_modules,
            self.state_dim,
        ):
            message = "Module states must have shape [batch, modules, state_dim]"
            raise CognitiveModuleError(message)
        if module_states.dtype != dtype or module_states.device != device:
            message = "Module states must preserve the input dtype and device"
            raise CognitiveModuleError(message)
        if not bool(torch.isfinite(module_states).all()):
            message = "Module states contain NaN or Inf"
            raise CognitiveModuleError(message)
