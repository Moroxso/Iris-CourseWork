"""Arm-labelled persisted checkpoint schema for no-workspace teacher arm B."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from iris.artifacts.checkpoint_v1 import (
    TeacherAdamWConfigV1,
    TeacherAdamWParameterStateV1,
    TeacherBackboneRuntimeV1,
    TeacherCheckpointOptimizerV1,
    TeacherCheckpointSoftwareV1,
    TeacherCheckpointStateFileV1,
    TeacherCheckpointTensorV1,
    _CheckpointModel,
    _HasSha256,
    _validate_names,
)
from iris.schemas import BackboneConfig, CognitiveConfig

_PROFILE_SET_V3_SHA256 = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)


class NoWorkspaceTeacherCheckpointSchemaV2(_CheckpointModel):
    """Exact schema identity for the arm-B checkpoint successor format."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    artifact_type: Literal["iris.teacher_checkpoint"]
    version: Literal[2]
    canonicalization: Literal["iris-json-v1"]
    tensor_encoding: Literal["safetensors-v1"]
    hash_algorithm: Literal["sha256"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-B teacher checkpoint version must be an exact integer"
            raise ValueError(message)
        return value


class NoWorkspaceTeacherCheckpointModelIdentityV2(_HasSha256):
    """No-workspace sidecar identity with no representable workspace config."""

    arm_id: Literal["B"]
    arm_kind: Literal["llm_modules_no_workspace"]
    teacher_graph: Literal["independent_recurrent_modules_zero_memory_read_teacher_v1"]
    execution_profile_set_sha256: Literal[
        "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
    ]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    cognitive_cycles: Literal[3]
    latent_token_count: int = Field(gt=0, strict=True)
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("cognitive_cycles", mode="before")
    @classmethod
    def validate_exact_cycles(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-B cognitive_cycles must be an exact integer"
            raise ValueError(message)
        return value

    @field_validator("component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-B checkpoint component_names")
        if value != _COMPONENT_NAMES:
            message = "arm-B checkpoint component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-B sidecar state_names")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> Self:
        runtime = self.runtime
        if self.execution_profile_set_sha256 != _PROFILE_SET_V3_SHA256:
            message = "arm-B checkpoint is not bound to execution profile set v3"
            raise ValueError(message)
        if self.latent_token_count != self.cognitive.num_modules:
            message = "arm-B latent token count must equal cognitive module count"
            raise ValueError(message)
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
        if (
            runtime.max_prompt_tokens + self.latent_token_count + runtime.max_target_tokens
            > runtime.max_position_embeddings
        ):
            message = "arm-B checkpoint sequence maxima exceed backbone context"
            raise ValueError(message)
        return self


class NoWorkspaceTeacherCheckpointPayloadV2(_CheckpointModel):
    """Canonical arm-B model, AdamW, tensor, and software payload."""

    model: NoWorkspaceTeacherCheckpointModelIdentityV2
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
        _validate_names(keys, label="arm-B checkpoint tensor keys")
        if keys != tuple(sorted(keys)):
            message = "arm-B checkpoint tensor descriptors must use sorted key order"
            raise ValueError(message)
        sidecar_names = tuple(
            tensor.state_name for tensor in self.tensors if tensor.owner == "sidecar"
        )
        if sidecar_names != self.model.state_names:
            message = "arm-B tensor table does not match model state_names"
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
        adamw_count = sum(tensor.owner == "adamw" for tensor in self.tensors)
        if actual_adamw != expected_adamw or adamw_count != len(expected_adamw):
            message = "arm-B checkpoint AdamW tensor table is incomplete or duplicated"
            raise ValueError(message)
        return self


class IrisNoWorkspaceTeacherCheckpointEnvelopeV2(_HasSha256):
    """Content-addressed envelope for exactly one arm-B teacher checkpoint."""

    schema_identity: NoWorkspaceTeacherCheckpointSchemaV2
    payload: NoWorkspaceTeacherCheckpointPayloadV2


__all__ = [
    "IrisNoWorkspaceTeacherCheckpointEnvelopeV2",
    "NoWorkspaceTeacherCheckpointModelIdentityV2",
    "NoWorkspaceTeacherCheckpointPayloadV2",
    "NoWorkspaceTeacherCheckpointSchemaV2",
    "TeacherAdamWConfigV1",
    "TeacherAdamWParameterStateV1",
    "TeacherBackboneRuntimeV1",
    "TeacherCheckpointOptimizerV1",
    "TeacherCheckpointSoftwareV1",
    "TeacherCheckpointStateFileV1",
    "TeacherCheckpointTensorV1",
]
