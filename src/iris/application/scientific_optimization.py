"""Bounded two-pass token-window AdamW for the row-bound scientific teacher.

The first pass keeps only detached scalars/distributions/RNG states. Replay uses
the exact derivative of KL of the window means, not the mean of example KLs.
This FP32 owner does not implement AMP or persistent checkpoint/run IO yet.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.application.scientific_ablation_teacher import (
    IrisScientificAblationTeacherOutputV1,
    IrisScientificAblationTeacherV1,
)
from iris.application.scientific_teacher import (
    IrisScientificTeacherOutputV1,
    IrisScientificTeacherV1,
)
from iris.data.schemas import IrisExample
from iris.training.losses import uniform_utilization_loss
from iris.training.optimization import IrisTeacherOptimizer
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state


class IrisScientificOptimizationError(RuntimeError):
    """A token-window update did not complete; uncertain writes poison the owner."""


class IrisScientificOptimizerPolicyV1(BaseModel):
    """Source defaults, explicit smoke exception and bounded complete input windows."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )
    kind: Literal["iris-scientific-optimizer-v1"] = "iris-scientific-optimizer-v1"
    purpose: Literal["scientific", "smoke"] = "scientific"
    minimum_target_tokens: int = Field(default=16384, ge=1, le=1048576)
    maximum_target_tokens: int = Field(default=65536, ge=1, le=1048576)
    maximum_examples: int = Field(default=4096, ge=1, le=4096)
    maximum_logical_bytes: int = Field(default=64 * 1024**2, ge=1, le=512 * 1024**2)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_gradient_norm: float = Field(default=1.0, gt=0.0)
    max_optimizer_steps: int = Field(default=30000, ge=2, le=2**24)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    min_learning_rate_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    precision: Literal["fp32"] = "fp32"
    usage_simplex_policy: Literal["fp64_per_example_unit_sum_closure"] = (
        "fp64_per_example_unit_sum_closure"
    )
    objective_reduction: Literal["example_mean_with_window_mean_usage_v1"] = (
        "example_mean_with_window_mean_usage_v1"
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.purpose == "scientific" and self.minimum_target_tokens < 16384:
            raise ValueError("Scientific windows require at least 16384 target tokens")
        if self.maximum_target_tokens < self.minimum_target_tokens:
            raise ValueError("Window maximum cannot be below its minimum")
        return self


def scientific_learning_rate_v1(policy: IrisScientificOptimizerPolicyV1, *, step: int) -> float:
    """One-based linear warmup, then cosine reaching the floor at the final step."""
    if type(policy) is not IrisScientificOptimizerPolicyV1 or type(step) is not int:
        raise IrisScientificOptimizationError(
            "Exact learning-rate policy and integer step required"
        )
    owned = IrisScientificOptimizerPolicyV1.model_validate(policy.model_dump(mode="python"))
    if not 1 <= step <= owned.max_optimizer_steps:
        raise IrisScientificOptimizationError("Learning-rate step is outside the declared schedule")
    warmup = min(
        owned.max_optimizer_steps - 1, math.ceil(owned.max_optimizer_steps * owned.warmup_ratio)
    )
    if warmup and step <= warmup:
        ratio = step / warmup
    else:
        progress = (
            (step - warmup) / (owned.max_optimizer_steps - warmup)
            if warmup
            else (step - 1) / (owned.max_optimizer_steps - 1)
        )
        ratio = (
            owned.min_learning_rate_ratio
            + (1.0 - owned.min_learning_rate_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
        )
    value = owned.learning_rate * ratio
    if not math.isfinite(value) or value <= 0:
        raise IrisScientificOptimizationError("Scheduled learning rate must remain finite/positive")
    return value


@dataclass(frozen=True, slots=True)
class IrisScientificOptimizationOutputV1:
    """Detached measurements for one committed window, not a persisted receipt."""

    step_index: int
    example_count: int
    supervised_token_count: int
    learning_rate: float
    gradient_l2_norm: float
    gradients_clipped: bool
    loss: float
    language_model_loss: float
    probe_loss: float
    diversity_loss: float
    utilization_loss: float
    norm_loss: float
    configuration_sha256: str
    window_sha256: str
    purpose: Literal["scientific", "smoke"]


@dataclass(frozen=True, slots=True)
class _Observation:
    identity: str
    token_count: int
    values: tuple[float, float, float, float]
    modules: torch.Tensor
    slots: torch.Tensor
    before: RuntimeStateSnapshot
    invocation_sha256: str | None = None


type _ScientificTeacherModel = IrisScientificTeacherV1 | IrisScientificAblationTeacherV1
type _ScientificTeacherOutput = (
    IrisScientificTeacherOutputV1 | IrisScientificAblationTeacherOutputV1
)


@dataclass(frozen=True, slots=True)
class _WindowValues:
    """Private shared numerical result; has no baseline or ablation identity."""

    step_index: int
    example_count: int
    supervised_token_count: int
    learning_rate: float
    gradient_l2_norm: float
    gradients_clipped: bool
    loss: float
    language_model_loss: float
    probe_loss: float
    diversity_loss: float
    utilization_loss: float
    norm_loss: float
    participation: tuple[int, int, int] | None


def _same_runtime(left: RuntimeStateSnapshot, right: RuntimeStateSnapshot) -> bool:
    return (
        left.python_rng_state == right.python_rng_state
        and torch.equal(left.cpu_rng_state, right.cpu_rng_state)
        and left.cuda_device == right.cuda_device
        and (
            (left.cuda_rng_state is None and right.cuda_rng_state is None)
            or (
                left.cuda_rng_state is not None
                and right.cuda_rng_state is not None
                and torch.equal(left.cuda_rng_state, right.cuda_rng_state)
            )
        )
        and left.deterministic_algorithms == right.deterministic_algorithms
        and left.deterministic_warn_only == right.deterministic_warn_only
        and left.cudnn_deterministic == right.cudnn_deterministic
        and left.cudnn_benchmark == right.cudnn_benchmark
    )


def _closed_distribution(value: torch.Tensor) -> torch.Tensor:
    """Close accepted FP32 simplex roundoff in FP64, preserving its Jacobian."""
    promoted = value.to(dtype=torch.float64)
    return promoted / promoted.sum()


class _ScientificOptimizerCore[ModelT: _ScientificTeacherModel, OutputT](ABC):
    """Own complete FP32 windows and all registered teacher/probe parameters.

    Caller supplies an explicit bounded tuple, not an iterator silently consumed
    past its boundary. Rows must be unique train rows. An underfilled/overfull
    window never updates weights. Only one live autograd example graph is kept.
    This boundary requires exclusive ownership of the process runtime during step.
    """

    def __init__(
        self,
        *,
        model: ModelT,
        policy: IrisScientificOptimizerPolicyV1,
        configuration_domain: str,
        configuration_extra: dict[str, object],
    ) -> None:
        if type(policy) is not IrisScientificOptimizerPolicyV1:
            raise IrisScientificOptimizationError(
                "Exact scientific model and explicit policy required"
            )
        model._validate_boundary(training=True)
        self._policy = IrisScientificOptimizerPolicyV1.model_validate(
            policy.model_dump(mode="python")
        )
        self._policy_json = self._policy.model_dump_json()
        self._model = model
        self._topology = IrisTeacherOptimizer._read_parameter_topology(model)
        IrisTeacherOptimizer._validate_parameters(self._topology)
        if self._topology.parameters[0].dtype != torch.float32:
            raise IrisScientificOptimizationError("This optimizer profile requires FP32 parameters")
        if any(parameter.grad is not None for parameter in self._topology.parameters):
            raise IrisScientificOptimizationError("Optimizer attachment requires clean gradients")
        self._versions = tuple(parameter._version for parameter in self._topology.parameters)
        self._device = self._topology.parameters[0].device
        self._model_identity = model.configuration_sha256
        configuration: dict[str, object] = {
            "policy": self._policy.model_dump(mode="json"),
            "model_configuration_sha256": self._model_identity,
            "parameter_names": self._topology.names,
        }
        configuration.update(configuration_extra)
        self._configuration_json = canonical_json_bytes(configuration).decode("utf-8")
        self._configuration_sha256 = domain_sha256(
            configuration_domain, self._configuration_json.encode("utf-8")
        )
        self._current_lr = scientific_learning_rate_v1(self._policy, step=1)
        self._optimizer = torch.optim.AdamW(
            self._topology.parameters,
            lr=self._current_lr,
            betas=(self._policy.beta1, self._policy.beta2),
            eps=self._policy.epsilon,
            weight_decay=self._policy.weight_decay,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        self._group_settings = {
            key: value
            for key, value in self._optimizer.param_groups[0].items()
            if key not in ("params", "lr")
        }
        self._completed_steps = 0
        self._parameter_steps = (0,) * len(self._topology.parameters)
        self._poisoned = False
        self._busy = False
        self._validate_contract()

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _validate_contract(self, *, parameter_steps: tuple[int, ...] | None = None) -> None:
        if self._poisoned or self._policy.model_dump_json() != self._policy_json:
            raise IrisScientificOptimizationError("Optimizer is poisoned or its policy changed")
        self._model._validate_boundary(training=True)
        current = IrisTeacherOptimizer._read_parameter_topology(self._model)
        if (
            self._model.configuration_sha256 != self._model_identity
            or current.names != self._topology.names
            or any(
                a is not b
                for a, b in zip(current.parameters, self._topology.parameters, strict=True)
            )
        ):
            raise IrisScientificOptimizationError("Owned scientific parameter topology changed")
        IrisTeacherOptimizer._validate_parameters(current)
        if len(self._optimizer.param_groups) != 1:
            raise IrisScientificOptimizationError("Exactly one owned AdamW group is required")
        group = self._optimizer.param_groups[0]
        if (
            len(group["params"]) != len(current.parameters)
            or any(a is not b for a, b in zip(group["params"], current.parameters, strict=True))
            or group["lr"] != self._current_lr
            or {key: value for key, value in group.items() if key not in ("params", "lr")}
            != self._group_settings
        ):
            raise IrisScientificOptimizationError("Owned AdamW group policy or parameters changed")
        identities = {id(parameter) for parameter in current.parameters}
        expected_steps = self._parameter_steps if parameter_steps is None else parameter_steps
        expected_by_id = dict(zip(map(id, current.parameters), expected_steps, strict=True))
        if {id(parameter) for parameter in self._optimizer.state} != {
            identity for identity, count in expected_by_id.items() if count
        }:
            raise IrisScientificOptimizationError(
                "AdamW state membership differs from owned updates"
            )
        for parameter, state in self._optimizer.state.items():
            if id(parameter) not in identities or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise IrisScientificOptimizationError("Noncanonical AdamW state")
            step = state["step"]
            if (
                type(step) is not torch.Tensor
                or step.ndim != 0
                or step.dtype != torch.float32
                or step.device.type != "cpu"
                or step.requires_grad
                or step.is_inference()
                or not bool(torch.isfinite(step))
                or float(step) != expected_by_id[id(parameter)]
            ):
                raise IrisScientificOptimizationError("Invalid AdamW counter")
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if (
                    type(value) is not torch.Tensor
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                    or value.requires_grad
                    or value.is_inference()
                    or value.layout != torch.strided
                    or not bool(torch.isfinite(value).all())
                    or (name == "exp_avg_sq" and bool((value < 0).any()))
                ):
                    raise IrisScientificOptimizationError("Invalid AdamW moment state")

    def _own_rows(self, examples: tuple[IrisExample, ...]) -> tuple[IrisExample, ...]:
        if type(examples) is not tuple or not 1 <= len(examples) <= self._policy.maximum_examples:
            raise IrisScientificOptimizationError("An explicit bounded complete window is required")
        rows = []
        logical_bytes = 0
        identities: set[str] = set()
        for example in examples:
            if type(example) is not IrisExample:
                raise IrisScientificOptimizationError("Exact scientific rows required")
            row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
            if row.split != "train" or row.id in identities:
                raise IrisScientificOptimizationError("Windows require unique train rows")
            identities.add(row.id)
            logical_bytes += len(canonical_json_bytes(row.model_dump(mode="json")))
            if logical_bytes > self._policy.maximum_logical_bytes:
                raise IrisScientificOptimizationError("Window exceeds its logical byte budget")
            rows.append(row)
        return tuple(rows)

    def _observe_common(
        self,
        output: _ScientificTeacherOutput,
        row: IrisExample,
        before: RuntimeStateSnapshot,
        *,
        output_type: type[_ScientificTeacherOutput],
        example_domain: str,
    ) -> _Observation:
        if type(output) is not output_type:
            raise IrisScientificOptimizationError("Exact scientific output required")
        expected = domain_sha256(example_domain, canonical_json_bytes(row.model_dump(mode="json")))
        if (
            output.example_id != row.id
            or output.example_sha256 != expected
            or output.configuration_sha256 != self._model_identity
            or type(output.supervised_token_count) is not int
            or output.supervised_token_count < 1
        ):
            raise IrisScientificOptimizationError(
                "Scientific output row/configuration/count differs"
            )
        values = []
        for value in (
            output.losses.language_model,
            output.losses.probe,
            output.losses.diversity,
            output.losses.norm,
        ):
            if (
                type(value) is not torch.Tensor
                or value.ndim != 0
                or value.device != self._device
                or not value.requires_grad
                or value.grad_fn is None
                or not bool(torch.isfinite(value))
                or bool(value.detach() < 0)
            ):
                raise IrisScientificOptimizationError("Invalid scientific objective graph")
            values.append(float(value.detach()))
        uniform_utilization_loss(output.usage.module_probabilities, output.usage.slot_probabilities)
        probabilities = []
        for value in (output.usage.module_probabilities, output.usage.slot_probabilities):
            if not value.requires_grad or value.device != self._device:
                raise IrisScientificOptimizationError("Window usage must retain its graph")
            probabilities.append(_closed_distribution(value.detach().to(device="cpu")).clone())
        uniform_utilization_loss(*probabilities)
        return _Observation(
            expected,
            output.supervised_token_count,
            (values[0], values[1], values[2], values[3]),
            probabilities[0],
            probabilities[1],
            before,
        )

    @abstractmethod
    def _observe(
        self, output: _ScientificTeacherOutput, row: IrisExample, before: RuntimeStateSnapshot
    ) -> _Observation:
        """Closed public variant validates its exact type and provenance."""

    @abstractmethod
    def _make_output(
        self, values: _WindowValues, observations: list[_Observation], terminal_token_id: int
    ) -> OutputT:
        """Closed public variant creates only its own result domain/type."""

    def _gradient_participation(self) -> tuple[int, int, int] | None:
        return None

    def _finish_success(self) -> None:
        """Variant-specific commit guard after successful gradient cleanup."""
        return None

    def _failure_was_stateful(self) -> bool:
        return False

    def step(self, examples: tuple[IrisExample, ...], *, terminal_token_id: int) -> OutputT:
        if self._busy:
            raise IrisScientificOptimizationError(
                "Concurrent/reentrant optimizer steps are forbidden"
            )
        self._busy = True
        initial = None
        update_started = False
        successful = False
        try:
            self._validate_contract()
            if self._completed_steps >= self._policy.max_optimizer_steps:
                raise IrisScientificOptimizationError("Optimizer step budget is exhausted")
            if any(parameter.grad is not None for parameter in self._topology.parameters):
                raise IrisScientificOptimizationError("Externally supplied gradients are forbidden")
            if (
                tuple(parameter._version for parameter in self._topology.parameters)
                != self._versions
            ):
                self._poisoned = True
                raise IrisScientificOptimizationError("Parameters changed outside their owner")
            rows = self._own_rows(examples)
            initial = capture_runtime_state(self._device)
            observations = []
            token_count = 0
            for row in rows:
                before = capture_runtime_state(self._device)
                output = self._model.training_forward(row, terminal_token_id=terminal_token_id)
                observation = self._observe(output, row, before)
                token_count += observation.token_count
                if token_count > self._policy.maximum_target_tokens:
                    raise IrisScientificOptimizationError("Window exceeds target-token maximum")
                observations.append(observation)
                del output  # No example autograd graph survives into the next forward.
            if token_count < self._policy.minimum_target_tokens:
                raise IrisScientificOptimizationError("Window has too few target tokens")
            after_scan = capture_runtime_state(self._device)
            count = len(rows)
            module_mean = (
                torch.stack([item.modules for item in observations]).mean(0).requires_grad_()
            )
            slot_mean = torch.stack([item.slots for item in observations]).mean(0).requires_grad_()
            utilization = uniform_utilization_loss(module_mean, slot_mean)
            module_derivative, slot_derivative = torch.autograd.grad(
                utilization, (module_mean, slot_mean)
            )
            weights = self._model.policy.weights
            means = tuple(
                math.fsum(item.values[index] for item in observations) / count for index in range(4)
            )
            for index, (row, observed) in enumerate(zip(rows, observations, strict=True)):
                restore_runtime_state(observed.before)
                output = self._model.training_forward(row, terminal_token_id=terminal_token_id)
                replay = self._observe(output, row, observed.before)
                if (
                    replay.identity != observed.identity
                    or replay.invocation_sha256 != observed.invocation_sha256
                    or replay.token_count != observed.token_count
                    or replay.values != observed.values
                    or not torch.equal(replay.modules, observed.modules)
                    or not torch.equal(replay.slots, observed.slots)
                ):
                    raise IrisScientificOptimizationError(
                        "Scientific replay differs from its first pass"
                    )
                expected_runtime = (
                    observations[index + 1].before if index + 1 < count else after_scan
                )
                if not _same_runtime(capture_runtime_state(self._device), expected_runtime):
                    raise IrisScientificOptimizationError(
                        "Scientific forward RNG/runtime replay differs"
                    )
                terms = output.losses
                other = (
                    terms.language_model.double()
                    + weights.probe * terms.probe.double()
                    + weights.diversity * terms.diversity.double()
                    + weights.norm * terms.norm.double()
                )
                linearized = (
                    _closed_distribution(output.usage.module_probabilities)
                    * module_derivative.to(self._device)
                ).sum() + (
                    _closed_distribution(output.usage.slot_probabilities)
                    * slot_derivative.to(self._device)
                ).sum()
                torch.autograd.backward((other + weights.utilization * linearized) / count)
                del output, terms, other, linearized
                if tuple(
                    parameter._version for parameter in self._topology.parameters
                ) != self._versions or not _same_runtime(
                    capture_runtime_state(self._device), expected_runtime
                ):
                    raise IrisScientificOptimizationError(
                        "Scientific backward mutated parameters or runtime"
                    )
            restore_runtime_state(after_scan)  # Replay consumes no extra caller RNG draws.
            gradients = []
            for name, parameter in zip(
                self._topology.names, self._topology.parameters, strict=True
            ):
                if parameter.grad is not None:
                    IrisTeacherOptimizer._validate_one_gradient(name, parameter, parameter.grad)
                    gradients.append(parameter.grad)
            if not gradients:
                raise IrisScientificOptimizationError("Scientific window produced no gradients")
            norm = math.sqrt(
                math.fsum(
                    float(gradient.detach().double().square().sum()) for gradient in gradients
                )
            )
            if not math.isfinite(norm):
                raise IrisScientificOptimizationError("Nonfinite global gradient norm")
            participation = self._gradient_participation()
            clipped = norm > self._policy.max_gradient_norm
            if clipped:
                with torch.no_grad():
                    for gradient in gradients:
                        gradient.mul_(self._policy.max_gradient_norm / norm)
            self._validate_contract()
            step = self._completed_steps + 1
            self._current_lr = scientific_learning_rate_v1(self._policy, step=step)
            self._optimizer.param_groups[0]["lr"] = self._current_lr
            next_parameter_steps = tuple(
                previous + int(parameter.grad is not None)
                for previous, parameter in zip(
                    self._parameter_steps, self._topology.parameters, strict=True
                )
            )
            update_started = True
            self._optimizer.step()
            self._validate_contract(parameter_steps=next_parameter_steps)
            util_value = float(utilization.detach())
            values = _WindowValues(
                step_index=step,
                example_count=count,
                supervised_token_count=token_count,
                learning_rate=self._current_lr,
                gradient_l2_norm=norm,
                gradients_clipped=clipped,
                loss=means[0]
                + weights.probe * means[1]
                + weights.diversity * means[2]
                + weights.norm * means[3]
                + weights.utilization * util_value,
                language_model_loss=means[0],
                probe_loss=means[1],
                diversity_loss=means[2],
                utilization_loss=util_value,
                norm_loss=means[3],
                participation=participation,
            )
            result = self._make_output(values, observations, terminal_token_id)
            successful = True
            return result
        except BaseException as error:
            if (
                update_started
                or tuple(parameter._version for parameter in self._topology.parameters)
                != self._versions
                or self._failure_was_stateful()
            ):
                self._poisoned = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificOptimizationError(
                "Scientific optimizer window could not complete"
            ) from None
        finally:
            try:
                self._optimizer.zero_grad(set_to_none=True)
                if any(parameter.grad is not None for parameter in self._topology.parameters):
                    raise IrisScientificOptimizationError(
                        "Scientific gradient cleanup was incomplete"
                    )
                if initial is not None and not successful and not update_started:
                    restore_runtime_state(initial)
                if successful:
                    # Commit only after state validation AND gradient cleanup.
                    self._validate_contract(parameter_steps=next_parameter_steps)
                    self._finish_success()
                    self._parameter_steps = next_parameter_steps
                    self._versions = tuple(
                        parameter._version for parameter in self._topology.parameters
                    )
                    self._completed_steps = step
            except BaseException:
                self._poisoned = True
                raise IrisScientificOptimizationError(
                    "Scientific optimizer cleanup/runtime restoration failed"
                ) from None
            finally:
                self._busy = False


class IrisScientificOptimizerV1(
    _ScientificOptimizerCore[IrisScientificTeacherV1, IrisScientificOptimizationOutputV1]
):
    """Historical exact baseline owner; its public policy/result/domains are unchanged."""

    def __init__(
        self, *, model: IrisScientificTeacherV1, policy: IrisScientificOptimizerPolicyV1
    ) -> None:
        if (
            type(model) is not IrisScientificTeacherV1
            or type(policy) is not IrisScientificOptimizerPolicyV1
        ):
            raise IrisScientificOptimizationError(
                "Exact scientific model and explicit policy required"
            )
        super().__init__(
            model=model,
            policy=policy,
            configuration_domain="iris/scientific-optimizer-configuration/v1",
            configuration_extra={},
        )

    def _observe(
        self, output: _ScientificTeacherOutput, row: IrisExample, before: RuntimeStateSnapshot
    ) -> _Observation:
        return self._observe_common(
            output,
            row,
            before,
            output_type=IrisScientificTeacherOutputV1,
            example_domain="iris/scientific-teacher-example/v1",
        )

    def _make_output(
        self, values: _WindowValues, observations: list[_Observation], terminal_token_id: int
    ) -> IrisScientificOptimizationOutputV1:
        return IrisScientificOptimizationOutputV1(
            step_index=values.step_index,
            example_count=values.example_count,
            supervised_token_count=values.supervised_token_count,
            learning_rate=values.learning_rate,
            gradient_l2_norm=values.gradient_l2_norm,
            gradients_clipped=values.gradients_clipped,
            loss=values.loss,
            language_model_loss=values.language_model_loss,
            probe_loss=values.probe_loss,
            diversity_loss=values.diversity_loss,
            utilization_loss=values.utilization_loss,
            norm_loss=values.norm_loss,
            configuration_sha256=self.configuration_sha256,
            window_sha256=domain_sha256(
                "iris/scientific-optimizer-window/v1",
                canonical_json_bytes(
                    {
                        "rows": [item.identity for item in observations],
                        "terminal_token_id": terminal_token_id,
                        "target_counts": [item.token_count for item in observations],
                    }
                ),
            ),
            purpose=self._policy.purpose,
        )


__all__ = [
    "IrisScientificOptimizationError",
    "IrisScientificOptimizationOutputV1",
    "IrisScientificOptimizerPolicyV1",
    "IrisScientificOptimizerV1",
    "scientific_learning_rate_v1",
]
