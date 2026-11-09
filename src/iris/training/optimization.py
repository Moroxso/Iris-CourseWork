"""Strict one-example AdamW ownership for the Iris teacher-forced sidecar."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from iris.training.model import IrisTeacherForcingModel
from iris.training.no_workspace_model import IrisNoWorkspaceTeacherForcingModel
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherForcingInput
from iris.training.shared_buffer_model import IrisSharedBufferTeacherForcingModel


class IrisOptimizationError(RuntimeError):
    """Raised when one owned sidecar optimization step cannot complete safely."""


@dataclass(frozen=True, slots=True)
class IrisAdamWConfig:
    """Fully explicit AdamW and global-gradient-clipping policy."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_gradient_norm: float

    def __post_init__(self) -> None:
        _validate_positive_float("learning_rate", self.learning_rate)
        _validate_unit_interval_float("beta1", self.beta1)
        _validate_unit_interval_float("beta2", self.beta2)
        _validate_positive_float("epsilon", self.epsilon)
        _validate_nonnegative_float("weight_decay", self.weight_decay)
        _validate_positive_float("max_gradient_norm", self.max_gradient_norm)


@dataclass(frozen=True, slots=True)
class IrisTeacherOptimizationOutput:
    """Detached scalar measurements from one committed optimizer update."""

    step_index: int
    loss: float
    supervised_token_count: int
    gradient_l2_norm: float
    gradients_clipped: bool


@dataclass(frozen=True, slots=True)
class IrisAdamWParameterSnapshot:
    """Owned CPU copy of canonical AdamW moments for one sidecar parameter."""

    parameter_name: str
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


@dataclass(frozen=True, slots=True)
class IrisTeacherOptimizerSnapshot:
    """Complete checkpoint state for one owned AdamW boundary."""

    completed_steps: int
    config: IrisAdamWConfig
    parameter_names: tuple[str, ...]
    parameter_states: tuple[IrisAdamWParameterSnapshot, ...]


@dataclass(frozen=True, slots=True)
class IrisTeacherOptimizerProvenance:
    """Lightweight validated identity of one stable AdamW owner."""

    completed_steps: int
    config: IrisAdamWConfig
    parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParameterTopology:
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


def _validate_positive_float(name: str, value: object) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        message = f"{name} must be a finite exact float greater than zero"
        raise ValueError(message)


def _validate_nonnegative_float(name: str, value: object) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        message = f"{name} must be a finite exact float greater than or equal to zero"
        raise ValueError(message)


def _validate_unit_interval_float(name: str, value: object) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0 or value >= 1.0:
        message = f"{name} must be a finite exact float in [0, 1)"
        raise ValueError(message)


class IrisTeacherOptimizer:
    """Own exactly one AdamW instance and the sidecar gradient lifecycle.

    This boundary deliberately performs one microbatch-one update at a time. It
    neither tokenizes examples nor accumulates gradients, schedules learning
    rates, uses AMP, or owns checkpoints and experiment logging.
    """

    _MAX_EXACT_ADAMW_STEPS = 2**24

    __slots__ = (
        "_completed_steps",
        "_config",
        "_config_snapshot",
        "_model",
        "_optimizer",
        "_parameter_topology",
        "_poisoned",
    )

    def __init__(
        self,
        *,
        model: (
            IrisTeacherForcingModel
            | IrisNoWorkspaceTeacherForcingModel
            | IrisSharedBufferTeacherForcingModel
        ),
        config: IrisAdamWConfig,
    ) -> None:
        if not isinstance(
            model,
            (
                IrisTeacherForcingModel,
                IrisNoWorkspaceTeacherForcingModel,
                IrisSharedBufferTeacherForcingModel,
            ),
        ):
            message = (
                "model must be an IrisTeacherForcingModel, "
                "IrisNoWorkspaceTeacherForcingModel, or "
                "IrisSharedBufferTeacherForcingModel"
            )
            raise TypeError(message)
        if type(config) is not IrisAdamWConfig:
            message = "config must be exactly IrisAdamWConfig"
            raise TypeError(message)
        topology = self._read_parameter_topology(model)
        self._validate_training_mode(model)
        self._validate_parameters(topology)
        if any(parameter.grad is not None for parameter in topology.parameters):
            message = "Teacher optimizer attachment requires gradient-free parameters"
            raise IrisOptimizationError(message)

        try:
            optimizer = torch.optim.AdamW(
                topology.parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=False,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
        except Exception as error:
            message = "Failed to construct the owned Iris AdamW optimizer"
            raise IrisOptimizationError(message) from error

        self._model = model
        self._config = config
        self._config_snapshot = self._snapshot_config(config)
        self._parameter_topology = topology
        self._optimizer = optimizer
        self._completed_steps = 0
        self._poisoned = False
        self._validate_optimizer()

    @property
    def config(self) -> IrisAdamWConfig:
        """Return the immutable policy object attached to this optimizer."""
        return self._config

    @property
    def model(
        self,
    ) -> (
        IrisTeacherForcingModel
        | IrisNoWorkspaceTeacherForcingModel
        | IrisSharedBufferTeacherForcingModel
    ):
        """Return the exact teacher graph owned by this optimizer boundary."""
        return self._model

    @property
    def completed_steps(self) -> int:
        """Return the number of fully committed and validated updates."""
        return self._completed_steps

    def checkpoint_snapshot(self) -> IrisTeacherOptimizerSnapshot:
        """Return detached CPU copies of canonical AdamW checkpoint state."""
        if self._poisoned:
            message = "Cannot checkpoint a poisoned teacher optimizer"
            raise IrisOptimizationError(message)
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "Cannot checkpoint while sidecar gradients are present"
            raise IrisOptimizationError(message)
        self._validate_stable_contract()

        snapshots: list[IrisAdamWParameterSnapshot] = []
        for name, parameter in zip(
            self._parameter_topology.names,
            self._parameter_topology.parameters,
            strict=True,
        ):
            state = self._optimizer.state.get(parameter)
            if state is None:
                continue
            step, exp_avg, exp_avg_sq = self._canonical_adamw_state(
                name,
                parameter,
                state,
                completed_steps=self._completed_steps,
            )
            with torch.inference_mode(False), torch.no_grad():
                snapshots.append(
                    IrisAdamWParameterSnapshot(
                        parameter_name=name,
                        step=step,
                        exp_avg=exp_avg.detach().to(device="cpu").contiguous().clone(),
                        exp_avg_sq=exp_avg_sq.detach().to(device="cpu").contiguous().clone(),
                    )
                )
        if self._completed_steps == 0 and snapshots:
            message = "A pristine teacher optimizer cannot contain AdamW state"
            raise IrisOptimizationError(message)
        if self._completed_steps > 0 and not snapshots:
            message = "A stepped teacher optimizer must contain AdamW state"
            raise IrisOptimizationError(message)
        return IrisTeacherOptimizerSnapshot(
            completed_steps=self._completed_steps,
            config=self._config,
            parameter_names=self._parameter_topology.names,
            parameter_states=tuple(snapshots),
        )

    def provenance_snapshot(self) -> IrisTeacherOptimizerProvenance:
        """Return validated policy, topology, and counter without copying moments."""
        if self._poisoned:
            message = "Cannot inspect a poisoned teacher optimizer"
            raise IrisOptimizationError(message)
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "Cannot inspect a teacher optimizer while gradients are present"
            raise IrisOptimizationError(message)
        self._validate_stable_contract()

        active_state_count = 0
        for name, parameter in zip(
            self._parameter_topology.names,
            self._parameter_topology.parameters,
            strict=True,
        ):
            state = self._optimizer.state.get(parameter)
            if state is None:
                continue
            self._canonical_adamw_state(
                name,
                parameter,
                state,
                completed_steps=self._completed_steps,
            )
            active_state_count += 1
        if self._completed_steps == 0 and active_state_count != 0:
            message = "Zero-step AdamW provenance cannot contain parameter state"
            raise IrisOptimizationError(message)
        if self._completed_steps > 0 and active_state_count == 0:
            message = "Stepped AdamW provenance requires active parameter state"
            raise IrisOptimizationError(message)
        return IrisTeacherOptimizerProvenance(
            completed_steps=self._completed_steps,
            config=self._config,
            parameter_names=self._parameter_topology.names,
        )

    def restore_checkpoint_snapshot(self, snapshot: IrisTeacherOptimizerSnapshot) -> None:
        """Install validated AdamW state into one pristine optimizer owner."""
        if type(snapshot) is not IrisTeacherOptimizerSnapshot:
            message = "snapshot must be exactly IrisTeacherOptimizerSnapshot"
            raise IrisOptimizationError(message)
        if self._poisoned:
            message = "Cannot restore a poisoned teacher optimizer"
            raise IrisOptimizationError(message)
        if self._completed_steps != 0 or self._optimizer.state:
            message = "AdamW checkpoint restore requires a pristine optimizer owner"
            raise IrisOptimizationError(message)
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "AdamW checkpoint restore requires gradient-free parameters"
            raise IrisOptimizationError(message)
        self._validate_stable_contract()
        candidate_state = self._materialize_snapshot(snapshot)

        self._optimizer.state = cast(
            defaultdict[torch.Tensor, Any],
            candidate_state,
        )
        self._completed_steps = snapshot.completed_steps
        try:
            self._validate_stable_contract()
            restored = self.checkpoint_snapshot()
            self._validate_snapshot_equivalence(restored, snapshot)
        except Exception as error:
            self._poisoned = True
            message = "Restored AdamW state failed post-installation validation"
            raise IrisOptimizationError(message) from error

    def validate_checkpoint_snapshot(self, snapshot: IrisTeacherOptimizerSnapshot) -> None:
        """Validate and materialize a restore candidate without changing owned state."""
        if type(snapshot) is not IrisTeacherOptimizerSnapshot:
            message = "snapshot must be exactly IrisTeacherOptimizerSnapshot"
            raise IrisOptimizationError(message)
        if self._poisoned:
            message = "Cannot restore a poisoned teacher optimizer"
            raise IrisOptimizationError(message)
        if self._completed_steps != 0 or self._optimizer.state:
            message = "AdamW checkpoint restore requires a pristine optimizer owner"
            raise IrisOptimizationError(message)
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "AdamW checkpoint restore requires gradient-free parameters"
            raise IrisOptimizationError(message)
        self._validate_stable_contract()
        _ = self._materialize_snapshot(snapshot)

    def step(self, example: IrisTeacherForcingInput) -> IrisTeacherOptimizationOutput:
        """Run forward, backward, global clipping, and exactly one AdamW update."""
        stage = "preflight validation"
        optimizer_step_started = False
        candidate_output: IrisTeacherOptimizationOutput | None = None
        primary_error: BaseException | None = None

        try:
            self._validate_step_preflight(example)
            stage = "initial gradient clearing"
            self._optimizer.zero_grad(set_to_none=True)
            self._validate_gradients_cleared()

            versions_before = tuple(
                parameter._version for parameter in self._parameter_topology.parameters
            )
            stage = "teacher-forced forward"
            forward_output = self._model.training_forward(example)
            self._validate_forward_output(forward_output)
            self._validate_parameter_versions(versions_before)
            loss_value = float(forward_output.loss.detach().item())

            stage = "loss backward"
            torch.autograd.backward((forward_output.loss,))
            self._validate_parameter_versions(versions_before)

            stage = "gradient validation"
            gradients = self._read_gradients()
            gradient_norm = self._gradient_l2_norm(gradients)

            stage = "global gradient clipping"
            gradients_clipped = self._clip_gradients(
                gradients,
                gradient_norm=gradient_norm,
            )
            self._validate_gradient_tensors(gradients)

            stage = "AdamW update"
            optimizer_step_started = True
            self._optimizer.step()
            optimizer_step_started = False

            stage = "post-update validation"
            self._validate_stable_contract()
            next_step = self._completed_steps + 1
            candidate_output = IrisTeacherOptimizationOutput(
                step_index=next_step,
                loss=loss_value,
                supervised_token_count=forward_output.supervised_token_count,
                gradient_l2_norm=gradient_norm,
                gradients_clipped=gradients_clipped,
            )
        except BaseException as error:
            if optimizer_step_started or stage == "post-update validation":
                self._poisoned = True
            primary_error = IrisOptimizationError(
                f"Iris teacher optimization failed during {stage}: {error}"
            )
            primary_error.__cause__ = error

        cleanup_error: BaseException | None = None
        try:
            self._optimizer.zero_grad(set_to_none=True)
            self._validate_gradients_cleared()
        except BaseException as error:
            self._poisoned = True
            cleanup_error = error

        if primary_error is not None:
            if cleanup_error is not None:
                failure_group = BaseExceptionGroup(
                    "Optimization and gradient cleanup both failed",
                    [primary_error.__cause__ or primary_error, cleanup_error],
                )
                raise IrisOptimizationError(
                    "Iris teacher optimization failed and gradient cleanup was incomplete"
                ) from failure_group
            raise primary_error
        if cleanup_error is not None:
            raise IrisOptimizationError(
                "Iris teacher optimization committed but gradient cleanup failed"
            ) from cleanup_error
        if candidate_output is None:
            message = "Iris teacher optimization produced no committed result"
            raise IrisOptimizationError(message)

        self._completed_steps = candidate_output.step_index
        return candidate_output

    def _validate_step_preflight(self, example: IrisTeacherForcingInput) -> None:
        if self._poisoned:
            message = "Teacher optimizer is poisoned after an uncertain update"
            raise IrisOptimizationError(message)
        if self._completed_steps >= self._MAX_EXACT_ADAMW_STEPS:
            message = "AdamW step counter exhausted its exact float32 range"
            raise IrisOptimizationError(message)
        if type(example) is not IrisTeacherForcingInput:
            message = "example must be exactly IrisTeacherForcingInput"
            raise IrisOptimizationError(message)
        if torch.is_inference_mode_enabled():
            message = "Teacher optimization cannot run under torch.inference_mode"
            raise IrisOptimizationError(message)
        if not torch.is_grad_enabled():
            message = "Teacher optimization requires autograd to be enabled"
            raise IrisOptimizationError(message)
        device_type = self._parameter_topology.parameters[0].device.type
        try:
            autocast_enabled = torch.is_autocast_enabled(device_type)
        except Exception as error:
            message = "Failed to inspect ambient autocast state"
            raise IrisOptimizationError(message) from error
        if autocast_enabled:
            message = "Automatic mixed precision is not supported by this optimizer boundary"
            raise IrisOptimizationError(message)
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "Gradient accumulation and externally supplied gradients are not supported"
            raise IrisOptimizationError(message)
        self._validate_stable_contract()

    def _validate_stable_contract(self) -> None:
        if self._snapshot_config(self._config) != self._config_snapshot:
            message = "AdamW configuration changed after optimizer construction"
            raise IrisOptimizationError(message)
        current_topology = self._read_parameter_topology(self._model)
        expected = self._parameter_topology
        if current_topology.names != expected.names or any(
            current is not original
            for current, original in zip(
                current_topology.parameters,
                expected.parameters,
                strict=True,
            )
        ):
            message = "Teacher sidecar parameter topology changed after optimizer construction"
            raise IrisOptimizationError(message)
        self._validate_training_mode(self._model)
        self._validate_parameters(expected)
        self._validate_optimizer()

    @staticmethod
    def _snapshot_config(config: IrisAdamWConfig) -> tuple[float, ...]:
        return (
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
            config.weight_decay,
            config.max_gradient_norm,
        )

    @staticmethod
    def _read_parameter_topology(model: nn.Module) -> _ParameterTopology:
        try:
            entries = tuple(model.named_parameters(remove_duplicate=False))
        except Exception as error:
            message = "Failed to enumerate teacher sidecar parameters"
            raise IrisOptimizationError(message) from error
        if not entries:
            message = "Teacher sidecar must expose at least one trainable parameter"
            raise IrisOptimizationError(message)
        names = tuple(name for name, _ in entries)
        parameters = tuple(parameter for _, parameter in entries)
        invalid_names = any(type(name) is not str or not name for name in names)
        if invalid_names or len(set(names)) != len(names):
            message = "Teacher sidecar parameter names must be nonempty and unique"
            raise IrisOptimizationError(message)
        if any(type(parameter) is not nn.Parameter for parameter in parameters):
            message = "Teacher sidecar entries must be exact torch.nn.Parameter instances"
            raise IrisOptimizationError(message)
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            message = "Teacher sidecar parameters must not be aliased"
            raise IrisOptimizationError(message)
        return _ParameterTopology(names=names, parameters=parameters)

    @staticmethod
    def _validate_training_mode(model: nn.Module) -> None:
        if not model.training:
            message = "Teacher model must be in training mode for optimization"
            raise IrisOptimizationError(message)
        mismatched = next((module for module in model.modules() if not module.training), None)
        if mismatched is not None:
            message = "Teacher model and all registered children must remain in training mode"
            raise IrisOptimizationError(message)

    @staticmethod
    def _validate_parameters(topology: _ParameterTopology) -> None:
        reference = topology.parameters[0]
        for name, parameter in zip(topology.names, topology.parameters, strict=True):
            if not parameter.requires_grad:
                message = f"Teacher sidecar parameter {name!r} must remain trainable"
                raise IrisOptimizationError(message)
            if parameter.is_inference():
                message = f"Teacher sidecar parameter {name!r} must be an ordinary tensor"
                raise IrisOptimizationError(message)
            if parameter.layout != torch.strided or not torch.is_floating_point(parameter):
                message = f"Teacher sidecar parameter {name!r} must be dense and floating"
                raise IrisOptimizationError(message)
            if parameter.device != reference.device or parameter.dtype != reference.dtype:
                message = "Every teacher sidecar parameter must share one device and dtype"
                raise IrisOptimizationError(message)
            if not bool(torch.isfinite(parameter).all()):
                message = f"Teacher sidecar parameter {name!r} contains NaN or Inf"
                raise IrisOptimizationError(message)

    def _validate_optimizer(self) -> None:
        optimizer = self._optimizer
        if type(optimizer) is not torch.optim.AdamW:
            message = "Owned optimizer must remain exactly torch.optim.AdamW"
            raise IrisOptimizationError(message)
        if type(optimizer.param_groups) is not list or len(optimizer.param_groups) != 1:
            message = "Owned AdamW optimizer must contain exactly one parameter group"
            raise IrisOptimizationError(message)
        group = optimizer.param_groups[0]
        if type(group) is not dict:
            message = "Owned AdamW parameter group must be an exact dictionary"
            raise IrisOptimizationError(message)
        group_parameters = group.get("params")
        expected_parameters = self._parameter_topology.parameters
        if type(group_parameters) is not list or len(group_parameters) != len(expected_parameters):
            message = "Owned AdamW parameter group does not cover the exact sidecar"
            raise IrisOptimizationError(message)
        if any(
            actual is not expected
            for actual, expected in zip(group_parameters, expected_parameters, strict=True)
        ):
            message = "Owned AdamW parameter order or identity changed"
            raise IrisOptimizationError(message)

        config = self._config
        expected_values: tuple[tuple[str, object], ...] = (
            ("lr", config.learning_rate),
            ("betas", (config.beta1, config.beta2)),
            ("eps", config.epsilon),
            ("weight_decay", config.weight_decay),
            ("amsgrad", False),
            ("maximize", False),
            ("foreach", False),
            ("capturable", False),
            ("differentiable", False),
            ("fused", False),
        )
        for key, expected_value in expected_values:
            actual_value = group.get(key)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                message = f"Owned AdamW parameter-group field {key!r} changed"
                raise IrisOptimizationError(message)

        owned_ids = {id(parameter) for parameter in expected_parameters}
        for state_parameter, state in optimizer.state.items():
            if id(state_parameter) not in owned_ids:
                message = "AdamW state contains a parameter outside the teacher sidecar"
                raise IrisOptimizationError(message)
            if type(state) is not dict:
                message = "AdamW per-parameter state must be an exact dictionary"
                raise IrisOptimizationError(message)
            for state_name, value in state.items():
                if type(state_name) is not str or not state_name:
                    message = "AdamW state names must be nonempty exact strings"
                    raise IrisOptimizationError(message)
                if type(value) is not torch.Tensor:
                    message = f"AdamW state {state_name!r} must be an exact tensor"
                    raise IrisOptimizationError(message)
                if value.is_inference() or value.layout != torch.strided:
                    message = f"AdamW state {state_name!r} must be an ordinary dense tensor"
                    raise IrisOptimizationError(message)
                if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                    message = f"AdamW state {state_name!r} contains NaN or Inf"
                    raise IrisOptimizationError(message)

    def _validate_forward_output(self, output: object) -> None:
        if type(output) is not IrisTeacherForcedOutput:
            message = "Teacher model must return exactly IrisTeacherForcedOutput"
            raise IrisOptimizationError(message)
        loss = output.loss
        if (
            type(loss) is not torch.Tensor
            or loss.ndim != 0
            or not torch.is_floating_point(loss)
            or not loss.requires_grad
            or loss.grad_fn is None
            or loss.is_inference()
            or loss.device != self._parameter_topology.parameters[0].device
            or loss.dtype != self._parameter_topology.parameters[0].dtype
            or not bool(torch.isfinite(loss))
            or bool(loss.detach() < 0)
        ):
            message = "Teacher optimization loss must be finite, nonnegative, and graph-attached"
            raise IrisOptimizationError(message)
        if type(output.supervised_token_count) is not int or output.supervised_token_count <= 0:
            message = "Teacher optimization token count must be an exact positive integer"
            raise IrisOptimizationError(message)

    def _validate_parameter_versions(self, expected_versions: tuple[int, ...]) -> None:
        actual_versions = tuple(
            parameter._version for parameter in self._parameter_topology.parameters
        )
        if actual_versions != expected_versions:
            message = "Teacher forward/backward mutated a sidecar parameter in place"
            raise IrisOptimizationError(message)

    def _read_gradients(self) -> tuple[torch.Tensor, ...]:
        gradients: list[torch.Tensor] = []
        for name, parameter in zip(
            self._parameter_topology.names,
            self._parameter_topology.parameters,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is not None:
                self._validate_one_gradient(name, parameter, gradient)
                gradients.append(gradient)
        if not gradients:
            message = "Teacher loss produced no sidecar gradients"
            raise IrisOptimizationError(message)
        return tuple(gradients)

    def _validate_gradient_tensors(self, gradients: tuple[torch.Tensor, ...]) -> None:
        current_gradients: list[torch.Tensor] = []
        for name, parameter in zip(
            self._parameter_topology.names,
            self._parameter_topology.parameters,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is not None:
                self._validate_one_gradient(name, parameter, gradient)
                current_gradients.append(gradient)
        if len(current_gradients) != len(gradients) or any(
            current is not expected
            for current, expected in zip(current_gradients, gradients, strict=True)
        ):
            message = "Sidecar gradient identity changed during clipping"
            raise IrisOptimizationError(message)

    @staticmethod
    def _validate_one_gradient(
        name: str,
        parameter: nn.Parameter,
        gradient: torch.Tensor,
    ) -> None:
        if (
            type(gradient) is not torch.Tensor
            or gradient.is_inference()
            or gradient.layout != torch.strided
            or gradient.requires_grad
            or not torch.is_floating_point(gradient)
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all())
        ):
            message = f"Gradient for sidecar parameter {name!r} is invalid"
            raise IrisOptimizationError(message)

    def _gradient_l2_norm(self, gradients: tuple[torch.Tensor, ...]) -> float:
        squared_norms: list[float] = []
        try:
            with torch.no_grad():
                for gradient in gradients:
                    with torch.autocast(device_type=gradient.device.type, enabled=False):
                        value = gradient.detach().to(dtype=torch.float64)
                        squared_norms.append(float(torch.sum(value * value).item()))
            squared_total = math.fsum(squared_norms)
            total = math.sqrt(squared_total)
        except Exception as error:
            message = "Failed to compute the sidecar global L2 gradient norm"
            raise IrisOptimizationError(message) from error
        if not math.isfinite(total) or total < 0.0:
            message = "Sidecar global L2 gradient norm is not finite"
            raise IrisOptimizationError(message)
        return total

    def _clip_gradients(
        self,
        gradients: tuple[torch.Tensor, ...],
        *,
        gradient_norm: float,
    ) -> bool:
        maximum = self._config.max_gradient_norm
        if gradient_norm <= maximum:
            return False
        scale = maximum / gradient_norm
        if not math.isfinite(scale) or scale <= 0.0 or scale >= 1.0:
            message = "Global gradient clipping produced an invalid scale"
            raise IrisOptimizationError(message)
        try:
            with torch.no_grad():
                for gradient in gradients:
                    gradient.mul_(scale)
        except Exception as error:
            message = "Failed to apply global L2 gradient clipping"
            raise IrisOptimizationError(message) from error
        return True

    def _validate_gradients_cleared(self) -> None:
        if any(parameter.grad is not None for parameter in self._parameter_topology.parameters):
            message = "Owned AdamW optimizer failed to clear every sidecar gradient"
            raise IrisOptimizationError(message)

    def _canonical_adamw_state(
        self,
        parameter_name: str,
        parameter: nn.Parameter,
        state: dict[str, object],
        *,
        completed_steps: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            message = f"AdamW state for {parameter_name!r} is not canonical"
            raise IrisOptimizationError(message)
        step_tensor = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if (
            type(step_tensor) is not torch.Tensor
            or step_tensor.shape != torch.Size([])
            or step_tensor.dtype != torch.float32
            or step_tensor.device != torch.device("cpu")
            or step_tensor.requires_grad
            or step_tensor.is_inference()
            or not bool(torch.isfinite(step_tensor))
        ):
            message = f"AdamW step state for {parameter_name!r} is invalid"
            raise IrisOptimizationError(message)
        step_value = float(step_tensor.item())
        if (
            not step_value.is_integer()
            or step_value < 1.0
            or step_value > completed_steps
            or step_value > self._MAX_EXACT_ADAMW_STEPS
        ):
            message = f"AdamW step state for {parameter_name!r} is inconsistent"
            raise IrisOptimizationError(message)
        for state_name, tensor in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if (
                type(tensor) is not torch.Tensor
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or tensor.device != parameter.device
                or tensor.layout != torch.strided
                or tensor.requires_grad
                or tensor.is_inference()
                or not bool(torch.isfinite(tensor).all())
            ):
                message = f"AdamW {state_name} state for {parameter_name!r} is invalid"
                raise IrisOptimizationError(message)
        return int(step_value), cast(torch.Tensor, exp_avg), cast(torch.Tensor, exp_avg_sq)

    def _materialize_snapshot(
        self,
        snapshot: IrisTeacherOptimizerSnapshot,
    ) -> defaultdict[nn.Parameter, dict[str, torch.Tensor]]:
        if type(snapshot.completed_steps) is not int or not (
            0 <= snapshot.completed_steps <= self._MAX_EXACT_ADAMW_STEPS
        ):
            message = "Checkpoint completed_steps is invalid"
            raise IrisOptimizationError(message)
        if type(snapshot.config) is not IrisAdamWConfig or snapshot.config != self._config:
            message = "Checkpoint AdamW configuration does not match the owner"
            raise IrisOptimizationError(message)
        if type(snapshot.parameter_names) is not tuple or (
            snapshot.parameter_names != self._parameter_topology.names
        ):
            message = "Checkpoint parameter topology does not match the owner"
            raise IrisOptimizationError(message)
        if type(snapshot.parameter_states) is not tuple:
            message = "Checkpoint parameter_states must be an exact tuple"
            raise IrisOptimizationError(message)
        if snapshot.completed_steps == 0 and snapshot.parameter_states:
            message = "A zero-step checkpoint cannot contain AdamW moments"
            raise IrisOptimizationError(message)
        if snapshot.completed_steps > 0 and not snapshot.parameter_states:
            message = "A stepped checkpoint must contain AdamW moments"
            raise IrisOptimizationError(message)

        by_name = dict(
            zip(
                self._parameter_topology.names,
                self._parameter_topology.parameters,
                strict=True,
            )
        )
        seen: set[str] = set()
        candidate: defaultdict[nn.Parameter, dict[str, torch.Tensor]] = defaultdict(dict)
        for parameter_state in snapshot.parameter_states:
            if type(parameter_state) is not IrisAdamWParameterSnapshot:
                message = "Checkpoint contains an invalid AdamW parameter snapshot"
                raise IrisOptimizationError(message)
            name = parameter_state.parameter_name
            if type(name) is not str or name not in by_name or name in seen:
                message = "Checkpoint AdamW parameter names are invalid or duplicated"
                raise IrisOptimizationError(message)
            seen.add(name)
            parameter = by_name[name]
            step = parameter_state.step
            if type(step) is not int or not (1 <= step <= snapshot.completed_steps):
                message = f"Checkpoint AdamW step for {name!r} is invalid"
                raise IrisOptimizationError(message)
            tensors = (
                ("exp_avg", parameter_state.exp_avg),
                ("exp_avg_sq", parameter_state.exp_avg_sq),
            )
            materialized: dict[str, torch.Tensor] = {}
            for state_name, tensor in tensors:
                if (
                    type(tensor) is not torch.Tensor
                    or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype
                    or tensor.device != torch.device("cpu")
                    or tensor.layout != torch.strided
                    or tensor.requires_grad
                    or tensor.is_inference()
                    or not bool(torch.isfinite(tensor).all())
                ):
                    message = f"Checkpoint {state_name} for {name!r} is invalid"
                    raise IrisOptimizationError(message)
                with torch.inference_mode(False), torch.no_grad():
                    materialized[state_name] = (
                        tensor.detach()
                        .to(device=parameter.device, dtype=parameter.dtype)
                        .contiguous()
                        .clone()
                    )
            with torch.inference_mode(False), torch.no_grad():
                materialized["step"] = torch.tensor(float(step), dtype=torch.float32)
            candidate[parameter] = {
                "step": materialized["step"],
                "exp_avg": materialized["exp_avg"],
                "exp_avg_sq": materialized["exp_avg_sq"],
            }
        return candidate

    @staticmethod
    def _validate_snapshot_equivalence(
        actual: IrisTeacherOptimizerSnapshot,
        expected: IrisTeacherOptimizerSnapshot,
    ) -> None:
        if (
            actual.completed_steps != expected.completed_steps
            or actual.config != expected.config
            or actual.parameter_names != expected.parameter_names
            or len(actual.parameter_states) != len(expected.parameter_states)
        ):
            message = "Restored AdamW checkpoint metadata changed"
            raise IrisOptimizationError(message)
        for actual_state, expected_state in zip(
            actual.parameter_states,
            expected.parameter_states,
            strict=True,
        ):
            if (
                actual_state.parameter_name != expected_state.parameter_name
                or actual_state.step != expected_state.step
                or not torch.equal(actual_state.exp_avg, expected_state.exp_avg)
                or not torch.equal(actual_state.exp_avg_sq, expected_state.exp_avg_sq)
            ):
                message = "Restored AdamW checkpoint tensors changed"
                raise IrisOptimizationError(message)
