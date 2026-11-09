"""Arm-C persisted teacher checkpoint schema for the shared-buffer graph."""

from re import fullmatch
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

_POLICY_SHA256 = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
_PROFILE_SET_V4_SHA256 = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_PROTOCOL_PAYLOAD_SHA256 = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
    "sequence_assembler",
)


class SharedBufferTeacherCheckpointSchemaV3(_CheckpointModel):
    """Exact schema identity for the arm-C checkpoint successor format."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    artifact_type: Literal["iris.teacher_checkpoint"]
    version: Literal[3]
    canonicalization: Literal["iris-json-v1"]
    tensor_encoding: Literal["safetensors-v1"]
    hash_algorithm: Literal["sha256"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-C teacher checkpoint version must be an exact integer"
            raise ValueError(message)
        return value


class SharedBufferTeacherCheckpointModelIdentityV3(_HasSha256):
    """Complete arm-C graph identity with no representable workspace config."""

    arm_id: Literal["C"]
    arm_kind: Literal["llm_modules_unlimited_shared_buffer"]
    teacher_graph: Literal["append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"]
    execution_profile_set_sha256: Literal[
        "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
    ]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    cognitive_cycles: Literal[3]
    pre_update_read_count: Literal[3]
    final_output_read_count: Literal[1]
    entries_per_cycle: int = Field(gt=0, strict=True)
    final_entry_count: int = Field(gt=0, strict=True)
    latent_token_count: int = Field(gt=0, strict=True)
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator(
        "cognitive_cycles",
        "pre_update_read_count",
        "final_output_read_count",
        mode="before",
    )
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-C lifecycle counts must be exact integers"
            raise ValueError(message)
        return value

    @field_validator("execution_profile_set_sha256", mode="before")
    @classmethod
    def validate_profile_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "execution profile set SHA-256 must be canonical lower-case hex"
            raise ValueError(message)
        return value

    @field_validator("component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-C checkpoint component_names")
        if value != _COMPONENT_NAMES:
            message = "arm-C checkpoint component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-C sidecar state_names")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> Self:
        runtime = self.runtime
        modules = self.cognitive.num_modules
        if self.shared_buffer_policy_sha256 != _POLICY_SHA256:
            message = "arm-C checkpoint does not bind canonical ADR-0037 policy"
            raise ValueError(message)
        if self.execution_profile_set_sha256 != _PROFILE_SET_V4_SHA256:
            message = "arm-C checkpoint is not bound to execution profile set v4"
            raise ValueError(message)
        if self.protocol_declaration_payload_sha256 != _PROTOCOL_PAYLOAD_SHA256:
            message = "arm-C checkpoint does not bind the recovered protocol payload"
            raise ValueError(message)
        if (
            self.entries_per_cycle != modules
            or self.final_entry_count != modules * self.cognitive_cycles
            or self.latent_token_count != modules
        ):
            message = "arm-C checkpoint buffer cardinalities are not canonical"
            raise ValueError(message)
        if self.pre_update_read_count != self.cognitive_cycles:
            message = "arm-C pre-update read count must equal cognitive cycles"
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
            message = "arm-C checkpoint sequence maxima exceed backbone context"
            raise ValueError(message)
        return self


class SharedBufferTeacherCheckpointPayloadV3(_CheckpointModel):
    """Canonical arm-C model, AdamW, tensor, and software payload."""

    model: SharedBufferTeacherCheckpointModelIdentityV3
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
        _validate_names(keys, label="arm-C checkpoint tensor keys")
        if keys != tuple(sorted(keys)):
            message = "arm-C checkpoint tensor descriptors must use sorted key order"
            raise ValueError(message)
        sidecar_names = tuple(
            tensor.state_name for tensor in self.tensors if tensor.owner == "sidecar"
        )
        if sidecar_names != self.model.state_names:
            message = "arm-C tensor table does not match model state_names"
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
            message = "arm-C checkpoint AdamW tensor table is incomplete or duplicated"
            raise ValueError(message)
        return self


class IrisSharedBufferTeacherCheckpointEnvelopeV3(_HasSha256):
    """Content-addressed envelope for exactly one arm-C teacher checkpoint."""

    schema_identity: SharedBufferTeacherCheckpointSchemaV3
    payload: SharedBufferTeacherCheckpointPayloadV3


__all__ = [
    "IrisSharedBufferTeacherCheckpointEnvelopeV3",
    "SharedBufferTeacherCheckpointModelIdentityV3",
    "SharedBufferTeacherCheckpointPayloadV3",
    "SharedBufferTeacherCheckpointSchemaV3",
    "TeacherAdamWConfigV1",
    "TeacherAdamWParameterStateV1",
    "TeacherBackboneRuntimeV1",
    "TeacherCheckpointOptimizerV1",
    "TeacherCheckpointSoftwareV1",
    "TeacherCheckpointStateFileV1",
    "TeacherCheckpointTensorV1",
]
