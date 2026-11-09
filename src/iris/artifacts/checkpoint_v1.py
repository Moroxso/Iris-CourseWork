"""Persisted schema for content-addressed Iris teacher checkpoints."""

from math import isfinite, prod
from re import fullmatch
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

CheckpointFloatDTypeV1 = Literal["float16", "bfloat16", "float32"]


class _CheckpointModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class _HasSha256(_CheckpointModel):
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "sha256 must contain exactly 64 lower-case hexadecimal characters"
            raise ValueError(message)
        return value


def _validate_names(values: tuple[str, ...], *, label: str, allow_empty: bool = False) -> None:
    if type(values) is not tuple or (not values and not allow_empty):
        message = f"{label} must be a nonempty exact tuple"
        raise ValueError(message)
    if any(type(value) is not str or not value or "\0" in value for value in values):
        message = f"{label} must contain nonempty exact strings without NUL"
        raise ValueError(message)
    if len(set(values)) != len(values):
        message = f"{label} must not contain duplicates"
        raise ValueError(message)


class TeacherCheckpointSchemaV1(_CheckpointModel):
    artifact_type: Literal["iris.teacher_checkpoint"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    tensor_encoding: Literal["safetensors-v1"]
    hash_algorithm: Literal["sha256"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "teacher checkpoint version must be an exact integer"
            raise ValueError(message)
        return value


class TeacherCheckpointTensorV1(_HasSha256):
    key: str = Field(min_length=1)
    owner: Literal["sidecar", "adamw"]
    parameter_name: str | None
    state_name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: CheckpointFloatDTypeV1
    byte_length: int = Field(ge=0, strict=True)

    @field_validator("key", "state_name")
    @classmethod
    def validate_nonempty_name(cls, value: str) -> str:
        if "\0" in value:
            message = "checkpoint tensor names cannot contain NUL"
            raise ValueError(message)
        return value

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if type(value) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in value
        ):
            message = "checkpoint tensor shape must contain nonnegative exact integers"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_tensor_identity(self) -> Self:
        if self.owner == "sidecar":
            if self.parameter_name is not None:
                message = "sidecar state descriptors cannot contain parameter_name"
                raise ValueError(message)
        elif (
            type(self.parameter_name) is not str
            or not self.parameter_name
            or "\0" in self.parameter_name
            or self.state_name not in {"exp_avg", "exp_avg_sq"}
        ):
            message = "AdamW tensor descriptor identity is invalid"
            raise ValueError(message)
        element_size = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }[self.dtype]
        if self.byte_length != prod(self.shape) * element_size:
            message = "checkpoint tensor byte_length does not match shape and dtype"
            raise ValueError(message)
        return self


class TeacherCheckpointStateFileV1(_HasSha256):
    filename: Literal["state.safetensors"]
    size_bytes: int = Field(gt=0, strict=True)
    tensor_count: int = Field(gt=0, strict=True)


class TeacherBackboneRuntimeV1(_CheckpointModel):
    hidden_size: int = Field(gt=0, strict=True)
    num_hidden_layers: int = Field(gt=0, strict=True)
    vocabulary_size: int = Field(gt=0, strict=True)
    max_position_embeddings: int = Field(gt=0, strict=True)
    max_prompt_tokens: int = Field(gt=0, strict=True)
    max_target_tokens: int = Field(gt=0, strict=True)
    device: str = Field(min_length=1)
    dtype: CheckpointFloatDTypeV1


class TeacherCheckpointModelIdentityV1(_HasSha256):
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="sidecar state_names")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> Self:
        runtime = self.runtime
        if self.backbone.limits.max_prompt_tokens != runtime.max_prompt_tokens:
            message = "checkpoint prompt limit disagrees with backbone configuration"
            raise ValueError(message)
        if self.backbone.limits.max_new_tokens != runtime.max_target_tokens:
            message = "checkpoint target limit disagrees with backbone configuration"
            raise ValueError(message)
        configured_dtype = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[self.backbone.dtype]
        if configured_dtype != runtime.dtype:
            message = "checkpoint runtime dtype disagrees with backbone configuration"
            raise ValueError(message)
        if self.workspace.winners_per_cycle > self.cognitive.num_modules:
            message = "checkpoint workspace winners exceed cognitive module count"
            raise ValueError(message)
        if (
            runtime.max_prompt_tokens + self.workspace.num_slots + runtime.max_target_tokens
            > runtime.max_position_embeddings
        ):
            message = "checkpoint sequence maxima exceed backbone context"
            raise ValueError(message)
        return self


class TeacherAdamWConfigV1(_CheckpointModel):
    learning_rate: float = Field(gt=0.0, strict=True)
    beta1: float = Field(ge=0.0, lt=1.0, strict=True)
    beta2: float = Field(ge=0.0, lt=1.0, strict=True)
    epsilon: float = Field(gt=0.0, strict=True)
    weight_decay: float = Field(ge=0.0, strict=True)
    max_gradient_norm: float = Field(gt=0.0, strict=True)

    @model_validator(mode="after")
    def validate_finite_values(self) -> Self:
        if not all(
            isfinite(value)
            for value in (
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.weight_decay,
                self.max_gradient_norm,
            )
        ):
            message = "checkpoint AdamW configuration must be finite"
            raise ValueError(message)
        return self


class TeacherAdamWParameterStateV1(_CheckpointModel):
    parameter_name: str = Field(min_length=1)
    step: int = Field(ge=1, le=2**24, strict=True)

    @field_validator("parameter_name")
    @classmethod
    def validate_parameter_name(cls, value: str) -> str:
        if "\0" in value:
            message = "AdamW parameter_name cannot contain NUL"
            raise ValueError(message)
        return value


class TeacherCheckpointOptimizerV1(_CheckpointModel):
    optimizer_type: Literal["torch.optim.AdamW"]
    completed_steps: int = Field(ge=0, le=2**24, strict=True)
    config: TeacherAdamWConfigV1
    parameter_names: tuple[str, ...]
    parameter_states: tuple[TeacherAdamWParameterStateV1, ...]

    @field_validator("parameter_names")
    @classmethod
    def validate_parameter_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="optimizer parameter_names")
        return value

    @model_validator(mode="after")
    def validate_optimizer_state(self) -> Self:
        state_names = tuple(state.parameter_name for state in self.parameter_states)
        _validate_names(
            state_names,
            label="optimizer state parameter names",
            allow_empty=True,
        )
        if any(name not in self.parameter_names for name in state_names):
            message = "AdamW state contains a parameter outside optimizer ownership"
            raise ValueError(message)
        if self.completed_steps == 0 and self.parameter_states:
            message = "zero-step checkpoint cannot contain AdamW parameter state"
            raise ValueError(message)
        if self.completed_steps > 0 and not self.parameter_states:
            message = "stepped checkpoint must contain AdamW parameter state"
            raise ValueError(message)
        if any(state.step > self.completed_steps for state in self.parameter_states):
            message = "AdamW parameter step exceeds completed_steps"
            raise ValueError(message)
        return self


class TeacherCheckpointSoftwareV1(_CheckpointModel):
    iris: str = Field(min_length=1)
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    safetensors: str = Field(min_length=1)


class TeacherCheckpointPayloadV1(_CheckpointModel):
    model: TeacherCheckpointModelIdentityV1
    optimizer: TeacherCheckpointOptimizerV1
    state_file: TeacherCheckpointStateFileV1
    tensors: tuple[TeacherCheckpointTensorV1, ...]
    software: TeacherCheckpointSoftwareV1

    @model_validator(mode="after")
    def validate_tensor_table(self) -> Self:
        if not self.tensors or len(self.tensors) != self.state_file.tensor_count:
            message = "checkpoint tensor table does not match state-file tensor_count"
            raise ValueError(message)
        keys = tuple(tensor.key for tensor in self.tensors)
        _validate_names(keys, label="checkpoint tensor keys")
        if keys != tuple(sorted(keys)):
            message = "checkpoint tensor descriptors must use sorted key order"
            raise ValueError(message)

        sidecar_names = tuple(
            tensor.state_name for tensor in self.tensors if tensor.owner == "sidecar"
        )
        if sidecar_names != self.model.state_names:
            message = "checkpoint sidecar tensor table does not match model state_names"
            raise ValueError(message)

        expected_adamw = {
            (state.parameter_name, state_name)
            for state in self.optimizer.parameter_states
            for state_name in ("exp_avg", "exp_avg_sq")
        }
        actual_adamw = {
            (tensor.parameter_name, tensor.state_name)
            for tensor in self.tensors
            if tensor.owner == "adamw"
        }
        adamw_tensor_count = sum(tensor.owner == "adamw" for tensor in self.tensors)
        if actual_adamw != expected_adamw or adamw_tensor_count != len(expected_adamw):
            message = "checkpoint AdamW tensor table is incomplete or duplicated"
            raise ValueError(message)
        return self


class IrisTeacherCheckpointEnvelopeV1(_HasSha256):
    schema_identity: TeacherCheckpointSchemaV1
    payload: TeacherCheckpointPayloadV1


__all__ = [
    "CheckpointFloatDTypeV1",
    "IrisTeacherCheckpointEnvelopeV1",
    "TeacherAdamWConfigV1",
    "TeacherAdamWParameterStateV1",
    "TeacherBackboneRuntimeV1",
    "TeacherCheckpointModelIdentityV1",
    "TeacherCheckpointOptimizerV1",
    "TeacherCheckpointPayloadV1",
    "TeacherCheckpointSchemaV1",
    "TeacherCheckpointSoftwareV1",
    "TeacherCheckpointStateFileV1",
    "TeacherCheckpointTensorV1",
]
