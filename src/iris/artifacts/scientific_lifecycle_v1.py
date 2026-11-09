"""Versioned lifecycle policy, state and self-contained checkpoint envelope."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import IrisScientificCheckpointManifestV1

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class IrisScientificLifecyclePolicyV1(_Strict):
    kind: Literal["iris-scientific-lifecycle-v1"] = "iris-scientific-lifecycle-v1"
    purpose: Literal["scientific", "smoke"] = "scientific"
    validate_every_steps: int = Field(default=500, ge=1, le=2**24)
    checkpoint_every_steps: int = Field(default=1000, ge=1, le=2**24)
    early_stopping_patience: int = Field(default=8, ge=1, le=65536)
    selection_metric: Literal["validation_target_nll_per_token"] = "validation_target_nll_per_token"
    minimum_improvement: float = Field(default=0.0, ge=0.0)
    terminal_policy: Literal["validate_and_checkpoint"] = "validate_and_checkpoint"
    maximum_storage_bytes: int = Field(default=8 * 1024**3, ge=1, le=16 * 1024**3)


class ScientificLifecycleStateV1(_Strict):
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
            or (self.best_step is not None and self.best_step > self.last_validation_step)
            or ((self.validation_count == 0) != (self.best_step is None))
            or ((self.validation_count == 0) != (self.last_validation_step == 0))
            or ((self.checkpoint_count == 0) != (self.last_checkpoint_step == 0))
            or (self.phase == "stopped" and self.stop_reason is None)
            or (self.stop_reason is not None and self.phase not in ("checkpoint", "stopped"))
        ):
            raise ValueError("Inconsistent scientific lifecycle state")
        return self


class ScientificLifecycleCheckpointPayloadV1(_Strict):
    kind: Literal["iris-scientific-lifecycle-checkpoint-v1"] = (
        "iris-scientific-lifecycle-checkpoint-v1"
    )
    configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    configuration_sha256: Digest
    state: ScientificLifecycleStateV1
    current: IrisScientificCheckpointManifestV1
    best: IrisScientificCheckpointManifestV1 | None

    @model_validator(mode="after")
    def validate_complete_boundary(self) -> Self:
        state = self.state
        if (
            state.phase not in ("train", "stopped")
            or state.last_checkpoint_step != state.step
            or state.checkpoint_count < 1
            or state.step != self.current.payload.completed_steps
            or ((state.best_step is None) != (self.best is None))
        ):
            raise ValueError("Lifecycle checkpoint must describe a committed clean boundary")
        if self.best is not None and (
            self.best.payload.completed_steps != state.best_step
            or self.best.payload.configuration_sha256 != self.current.payload.configuration_sha256
        ):
            raise ValueError("Best scientific checkpoint does not match its lifecycle")
        return self


class IrisScientificLifecycleCheckpointManifestV1(_Strict):
    sha256: Digest
    payload: ScientificLifecycleCheckpointPayloadV1


__all__ = [
    "IrisScientificLifecycleCheckpointManifestV1",
    "IrisScientificLifecyclePolicyV1",
    "ScientificLifecycleCheckpointPayloadV1",
    "ScientificLifecycleStateV1",
]
