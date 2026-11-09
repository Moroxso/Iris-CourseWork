"""Distinct strict cognitive-ablation controller policy and committed phase state."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import _Strict


class IrisScientificAblationLifecyclePolicyV1(_Strict):
    kind: Literal["iris-scientific-ablation-lifecycle-v1"] = "iris-scientific-ablation-lifecycle-v1"
    purpose: Literal["scientific", "smoke"] = "scientific"
    validate_every_steps: int = Field(default=500, ge=1, le=2**24)
    checkpoint_every_steps: int = Field(default=1000, ge=1, le=2**24)
    early_stopping_patience: int = Field(default=8, ge=1, le=65536)
    selection_metric: Literal["validation_target_nll_per_token"] = "validation_target_nll_per_token"
    minimum_improvement: float = Field(default=0.0, ge=0.0)
    terminal_policy: Literal["validate_and_checkpoint"] = "validate_and_checkpoint"
    maximum_storage_bytes: int = Field(default=8 * 1024**3, ge=1, le=16 * 1024**3)


class ScientificAblationLifecycleStateV1(_Strict):
    phase: Literal["train", "validation", "checkpoint", "stopped"] = "train"
    step: int = Field(default=0, ge=0, le=2**24)
    validation_count: int = Field(default=0, ge=0, le=2**24)
    last_validation_step: int = Field(default=0, ge=0, le=2**24)
    best_step: int | None = Field(default=None, ge=1, le=2**24)
    best_nll_per_token: float | None = Field(default=None, ge=0.0)
    nonimproving_validations: int = Field(default=0, ge=0, le=65536)
    checkpoint_count: int = Field(default=0, ge=0, le=2**24)
    last_checkpoint_step: int = Field(default=0, ge=0, le=2**24)
    stop_reason: Literal["early_stopping", "step_budget"] | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if (
            self.last_validation_step > self.step
            or self.last_checkpoint_step > self.step
            or self.validation_count > self.step
            or self.checkpoint_count > self.step
            or self.nonimproving_validations > self.validation_count
            or ((self.best_step is None) != (self.best_nll_per_token is None))
            or ((self.validation_count == 0) != (self.best_step is None))
            or ((self.validation_count == 0) != (self.last_validation_step == 0))
            or ((self.checkpoint_count == 0) != (self.last_checkpoint_step == 0))
            or (self.best_step is not None and self.best_step > self.last_validation_step)
            or (self.phase == "stopped" and self.stop_reason is None)
            or (self.stop_reason is not None and self.phase not in ("checkpoint", "stopped"))
        ):
            raise ValueError("Inconsistent scientific ablation controller state")
        return self


__all__ = ["IrisScientificAblationLifecyclePolicyV1", "ScientificAblationLifecycleStateV1"]
