"""Checkpoint-bound request-artifact schema for profile-v4 A/B/C/D evaluation."""

from __future__ import annotations

from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.request_v3 import (
    IrisRequestArtifactPayloadV3,
    RequestArmATopologyV3,
    RequestArmBTopologyV3,
    RequestArmCTopologyV3,
    RequestArmDTopologyV3,
)

_STATE_COMPONENTS = {
    "B": frozenset(
        {
            "layer_sampler",
            "goal_encoder",
            "module_input_router",
            "cognitive_modules",
            "module_latent_bridge",
        }
    ),
    "C": frozenset(
        {
            "layer_sampler",
            "goal_encoder",
            "module_input_router",
            "shared_buffer_reader",
            "cognitive_modules",
            "proposal_head",
            "shared_buffer_writer",
            "shared_buffer_latent_bridge",
        }
    ),
    "D": frozenset(
        {
            "layer_sampler",
            "goal_encoder",
            "module_input_router",
            "workspace_reader",
            "cognitive_modules",
            "proposal_head",
            "workspace_arbiter",
            "slot_allocator",
            "workspace_writer",
            "workspace_decay",
            "latent_bridge",
        }
    ),
}


class _RequestArtifactModelV4(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
        populate_by_name=True,
    )


def _validate_sha256(value: object, *, label: str) -> object:
    if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
        message = f"{label} must contain exactly 64 lower-case hexadecimal characters"
        raise ValueError(message)
    return value


class RequestArtifactSchemaV4(_RequestArtifactModelV4):
    """Schema identity for a checkpoint-bound request artifact."""

    artifact_type: Literal["iris.request"]
    version: Literal[4]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_values_sha256_only"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "request artifact v4 version must be an exact integer"
            raise ValueError(message)
        return value


class _RequestCheckpointTransferV4(_RequestArtifactModelV4):
    transfer_contract: Literal["exact_teacher_state_to_profile_v4_request_sidecar_v1"]
    arm_id: Literal["B", "C", "D"]
    checkpoint_schema_version: Literal[1, 2, 3]
    checkpoint_payload_sha256: str
    checkpoint_state_file_sha256: str
    source_teacher_sidecar_sha256: str
    request_sidecar_state_sha256: str
    completed_steps: int = Field(ge=0, le=2**24, strict=True)
    transferred_state_names: tuple[str, ...]
    transferred_tensor_count: int = Field(gt=0, strict=True)

    @field_validator("checkpoint_schema_version", mode="before")
    @classmethod
    def validate_exact_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "checkpoint schema version must be an exact integer"
            raise ValueError(message)
        return value

    @field_validator(
        "checkpoint_payload_sha256",
        "checkpoint_state_file_sha256",
        "source_teacher_sidecar_sha256",
        "request_sidecar_state_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: object) -> object:
        return _validate_sha256(value, label="request checkpoint-transfer SHA-256")

    @field_validator("transferred_state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            type(value) is not tuple
            or not value
            or any(type(name) is not str or not name or "\0" in name for name in value)
            or len(set(value)) != len(value)
        ):
            message = "transferred state names must be a nonempty unique exact tuple"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        expected_schema = {"B": 2, "C": 3, "D": 1}[self.arm_id]
        if self.checkpoint_schema_version != expected_schema:
            message = f"arm {self.arm_id} requires checkpoint schema {expected_schema}"
            raise ValueError(message)
        if self.transferred_tensor_count != len(self.transferred_state_names):
            message = "transferred tensor count differs from state-name topology"
            raise ValueError(message)
        allowed = _STATE_COMPONENTS[self.arm_id]
        for name in self.transferred_state_names:
            component, separator, local_name = name.partition(".")
            if not separator or not local_name or component not in allowed:
                message = f"arm {self.arm_id} transfer contains a non-overlap state name"
                raise ValueError(message)
        return self


class RequestArmBCheckpointTransferV4(_RequestCheckpointTransferV4):
    arm_id: Literal["B"]
    checkpoint_schema_version: Literal[2]


class RequestArmCCheckpointTransferV4(_RequestCheckpointTransferV4):
    arm_id: Literal["C"]
    checkpoint_schema_version: Literal[3]


class RequestArmDCheckpointTransferV4(_RequestCheckpointTransferV4):
    arm_id: Literal["D"]
    checkpoint_schema_version: Literal[1]


RequestArmCheckpointTransferV4 = Annotated[
    RequestArmBCheckpointTransferV4
    | RequestArmCCheckpointTransferV4
    | RequestArmDCheckpointTransferV4,
    Field(discriminator="arm_id"),
]


class IrisRequestArtifactPayloadV4(IrisRequestArtifactPayloadV3):
    """Profile-v4 request provenance plus exact trained-checkpoint transfer."""

    checkpoint_transfer: RequestArmCheckpointTransferV4 | None

    @model_validator(mode="after")
    def validate_checkpoint_binding(self) -> Self:
        topology = self.topology
        transfer = self.checkpoint_transfer
        if isinstance(topology, RequestArmATopologyV3):
            if transfer is not None:
                message = "arm A cannot carry teacher-checkpoint transfer provenance"
                raise ValueError(message)
            return self
        if transfer is None:
            message = f"arm {topology.arm_id} requires teacher-checkpoint transfer provenance"
            raise ValueError(message)
        if transfer.arm_id != topology.arm_id:
            message = "request topology and checkpoint-transfer arms differ"
            raise ValueError(message)
        if self.runtime.sidecar_state_sha256 != transfer.request_sidecar_state_sha256:
            message = "runtime and checkpoint-transfer request sidecar hashes differ"
            raise ValueError(message)
        if isinstance(
            topology,
            RequestArmBTopologyV3 | RequestArmCTopologyV3 | RequestArmDTopologyV3,
        ) and topology.sidecar_state_sha256 != transfer.request_sidecar_state_sha256:
            message = "topology and checkpoint-transfer request sidecar hashes differ"
            raise ValueError(message)
        return self


class IrisRequestArtifactEnvelopeV4(_RequestArtifactModelV4):
    """Canonical envelope for one checkpoint-bound request artifact v4."""

    schema_identity: RequestArtifactSchemaV4 = Field(alias="schema")
    payload_sha256: str
    payload: IrisRequestArtifactPayloadV4

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="request artifact v4 payload_sha256")


__all__ = [
    "IrisRequestArtifactEnvelopeV4",
    "IrisRequestArtifactPayloadV4",
    "RequestArmBCheckpointTransferV4",
    "RequestArmCCheckpointTransferV4",
    "RequestArmCheckpointTransferV4",
    "RequestArmDCheckpointTransferV4",
    "RequestArtifactSchemaV4",
]
