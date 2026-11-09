"""Strict arm-labelled schema for privacy-preserving request artifacts v2."""

from __future__ import annotations

from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    GenerationProvenanceV1,
    RequestProvenanceV1,
    SoftwareProvenanceV1,
    WorkspaceCycleProvenanceV1,
)

_PROTOCOL_DECLARATION_PAYLOAD_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_EXECUTION_PROFILE_SET_SHA256: Literal[
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
] = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"

_ARM_B_SIDECAR_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "prefix_assembler",
)
_ARM_D_SIDECAR_COMPONENTS = (
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
    "prefix_assembler",
)

type RequestArmIdV2 = Literal["A", "B", "D"]
type RequestArmKindV2 = Literal[
    "base_llm",
    "llm_modules_no_workspace",
    "llm_modules_limited_competitive_workspace",
]
type RequestGraphV2 = Literal[
    "frozen_backbone_prompt_only_v1",
    "independent_recurrent_modules_zero_memory_read_v1",
    "limited_competitive_workspace_v1",
]
type RequestTraceContractV2 = Literal[
    "empty_workspace_trace",
    "workspace_state_per_cycle",
]


class _RequestArtifactModelV2(BaseModel):
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


def _validate_exact_integer(value: object, *, label: str) -> object:
    if type(value) is not int:
        message = f"{label} must be an exact integer"
        raise ValueError(message)
    return value


class RequestArtifactSchemaV2(_RequestArtifactModelV2):
    """Schema identity for one immutable arm-labelled request artifact."""

    artifact_type: Literal["iris.request"]
    version: Literal[2]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_text_sha256_only"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        return _validate_exact_integer(value, label="request artifact version")


class RequestArmATopologyV2(_RequestArtifactModelV2):
    """Prompt-only baseline topology with no cognitive sidecar or latent tokens."""

    arm_id: Literal["A"]
    arm_kind: Literal["base_llm"]
    request_graph: Literal["frozen_backbone_prompt_only_v1"]
    memory_model: Literal["none"]
    trace_contract: Literal["empty_workspace_trace"]
    cognitive_cycles: Literal[0]
    latent_token_count: Literal[0]
    zero_memory_read_cycles: Literal[0]
    sidecar_component_names: tuple[()]
    sidecar_state_sha256: None

    @field_validator(
        "cognitive_cycles",
        "latent_token_count",
        "zero_memory_read_cycles",
        mode="before",
    )
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        return _validate_exact_integer(value, label="arm-A topology count")


class RequestArmBTopologyV2(_RequestArtifactModelV2):
    """Independent recurrent modules with exact zero shared-memory reads."""

    arm_id: Literal["B"]
    arm_kind: Literal["llm_modules_no_workspace"]
    request_graph: Literal["independent_recurrent_modules_zero_memory_read_v1"]
    memory_model: Literal["none"]
    trace_contract: Literal["empty_workspace_trace"]
    cognitive_cycles: Literal[3]
    latent_token_count: int = Field(gt=0, strict=True)
    zero_memory_read_cycles: Literal[3]
    sidecar_component_names: tuple[str, ...]
    sidecar_state_sha256: str

    @field_validator("cognitive_cycles", "zero_memory_read_cycles", mode="before")
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        return _validate_exact_integer(value, label="arm-B topology count")

    @field_validator("sidecar_component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != _ARM_B_SIDECAR_COMPONENTS:
            message = "arm-B request sidecar component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("sidecar_state_sha256", mode="before")
    @classmethod
    def validate_sidecar_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="arm-B sidecar_state_sha256")


class RequestArmDTopologyV2(_RequestArtifactModelV2):
    """Limited competitive workspace request topology."""

    arm_id: Literal["D"]
    arm_kind: Literal["llm_modules_limited_competitive_workspace"]
    request_graph: Literal["limited_competitive_workspace_v1"]
    memory_model: Literal["limited_competitive_workspace"]
    trace_contract: Literal["workspace_state_per_cycle"]
    cognitive_cycles: int = Field(gt=0, strict=True)
    latent_token_count: int = Field(gt=0, strict=True)
    zero_memory_read_cycles: Literal[0]
    sidecar_component_names: tuple[str, ...]
    sidecar_state_sha256: str

    @field_validator("zero_memory_read_cycles", mode="before")
    @classmethod
    def validate_exact_zero_reads(cls, value: object) -> object:
        return _validate_exact_integer(value, label="arm-D zero_memory_read_cycles")

    @field_validator("sidecar_component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != _ARM_D_SIDECAR_COMPONENTS:
            message = "arm-D request sidecar component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("sidecar_state_sha256", mode="before")
    @classmethod
    def validate_sidecar_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="arm-D sidecar_state_sha256")


RequestArmTopologyV2 = Annotated[
    RequestArmATopologyV2 | RequestArmBTopologyV2 | RequestArmDTopologyV2,
    Field(discriminator="arm_id"),
]


class RequestRuntimeProvenanceV2(_RequestArtifactModelV2):
    """Resolved runtime, including an honest optional sidecar identity."""

    initialization_seed: int = Field(ge=0, le=2**63 - 1, strict=True)
    deterministic: bool = Field(strict=True)
    backend: Literal["cpu", "cuda"]
    device: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    dtype: Literal["float16", "bfloat16", "float32"]
    sidecar_state_sha256: str | None
    software: SoftwareProvenanceV1

    @field_validator("sidecar_state_sha256", mode="before")
    @classmethod
    def validate_optional_sidecar_sha256(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_sha256(value, label="runtime sidecar_state_sha256")


class EmptyWorkspaceTraceProvenanceV2(_RequestArtifactModelV2):
    """Canonical empty workspace trace for A and B."""

    trace_contract: Literal["empty_workspace_trace"]
    cycle_count: Literal[0]
    cycles: tuple[()]
    sha256: str

    @field_validator("cycle_count", mode="before")
    @classmethod
    def validate_exact_cycle_count(cls, value: object) -> object:
        return _validate_exact_integer(value, label="empty trace cycle_count")

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="empty trace sha256")


class WorkspaceTraceProvenanceV2(_RequestArtifactModelV2):
    """Nonempty per-cycle limited-workspace trace for D."""

    trace_contract: Literal["workspace_state_per_cycle"]
    cycle_count: int = Field(gt=0, strict=True)
    cycles: tuple[WorkspaceCycleProvenanceV1, ...]
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="workspace trace sha256")

    @model_validator(mode="after")
    def validate_cycles(self) -> Self:
        if len(self.cycles) != self.cycle_count:
            message = "workspace trace cycle_count must equal the number of cycles"
            raise ValueError(message)
        expected = tuple(range(1, self.cycle_count + 1))
        if tuple(cycle.cycle for cycle in self.cycles) != expected:
            message = "workspace trace cycles must be contiguous and one-indexed"
            raise ValueError(message)
        return self


RequestWorkspaceTraceProvenanceV2 = Annotated[
    EmptyWorkspaceTraceProvenanceV2 | WorkspaceTraceProvenanceV2,
    Field(discriminator="trace_contract"),
]


class IrisRequestArtifactPayloadV2(_RequestArtifactModelV2):
    """Arm-labelled request identity, private fingerprints, and exact trace contract."""

    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    execution_profile_set_version: Literal[3]
    execution_profile_set_sha256: Literal[
        "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
    ]
    topology: RequestArmTopologyV2
    configuration: ConfigurationProvenanceV1
    request: RequestProvenanceV1
    runtime: RequestRuntimeProvenanceV2
    generation: GenerationProvenanceV1
    workspace_trace: RequestWorkspaceTraceProvenanceV2

    @field_validator("execution_profile_set_version", mode="before")
    @classmethod
    def validate_exact_profile_version(cls, value: object) -> object:
        return _validate_exact_integer(value, label="execution profile set version")

    @model_validator(mode="after")
    def validate_cross_component_provenance(self) -> Self:
        config = self.configuration.value
        topology = self.topology
        trace = self.workspace_trace
        runtime = self.runtime
        decoding = self.request.decoding
        expected_dtype = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[config.hardware.precision]
        if (
            runtime.initialization_seed != config.runtime.seed
            or runtime.deterministic is not config.runtime.deterministic
            or runtime.backend != config.hardware.device
            or runtime.dtype != expected_dtype
            or config.backbone.local_files_only is decoding.allow_download
        ):
            message = "request runtime or download provenance disagrees with configuration"
            raise ValueError(message)
        if (runtime.backend == "cpu" and runtime.device != "cpu") or (
            runtime.backend == "cuda" and fullmatch(r"cuda:(0|[1-9][0-9]*)", runtime.device) is None
        ):
            message = "request runtime device does not identify its resolved backend"
            raise ValueError(message)
        if self.request.physical_prompt_tokens > config.backbone.limits.max_prompt_tokens:
            message = "request prompt provenance exceeds the configured prompt limit"
            raise ValueError(message)
        if self.generation.generated_tokens > config.backbone.limits.max_new_tokens:
            message = "request generation provenance exceeds the configured generation limit"
            raise ValueError(message)
        if (
            self.generation.stop_reason == "max_new_tokens"
            and self.generation.generated_tokens != config.backbone.limits.max_new_tokens
        ):
            message = "maximum-length request did not reach the configured generation limit"
            raise ValueError(message)
        if (
            self.generation.stop_reason == "stop_token"
            and self.generation.matched_stop_token_id not in decoding.stop_token_ids
        ):
            message = "matched generation stop token is absent from the decoding policy"
            raise ValueError(message)
        if (
            self.request.prompt_input_ids.device != runtime.device
            or self.generation.token_ids.device != runtime.device
        ):
            message = "request token provenance disagrees with the resolved runtime device"
            raise ValueError(message)
        if topology.trace_contract != trace.trace_contract:
            message = "request topology and workspace trace contracts disagree"
            raise ValueError(message)

        if isinstance(topology, RequestArmATopologyV2):
            if runtime.sidecar_state_sha256 is not None or not isinstance(
                trace, EmptyWorkspaceTraceProvenanceV2
            ):
                message = "arm A requires no sidecar and an exact empty workspace trace"
                raise ValueError(message)
        elif isinstance(topology, RequestArmBTopologyV2):
            if (
                runtime.sidecar_state_sha256 != topology.sidecar_state_sha256
                or config.cognitive.num_modules != 8
                or config.cognitive.state_dim != 256
                or topology.latent_token_count != config.cognitive.num_modules
                or not isinstance(trace, EmptyWorkspaceTraceProvenanceV2)
            ):
                message = "arm B topology disagrees with sidecar, cognitive config, or trace"
                raise ValueError(message)
        else:
            if (
                runtime.sidecar_state_sha256 != topology.sidecar_state_sha256
                or config.cognitive.num_modules != 8
                or config.cognitive.state_dim != 256
                or config.workspace.num_slots != 8
                or config.workspace.winners_per_cycle != 2
                or config.workspace.cycles != 3
                or topology.cognitive_cycles != config.workspace.cycles
                or topology.latent_token_count != config.workspace.num_slots
                or not isinstance(trace, WorkspaceTraceProvenanceV2)
                or trace.cycle_count != config.workspace.cycles
            ):
                message = "arm D topology disagrees with sidecar, workspace config, or trace"
                raise ValueError(message)
            expected_slots_shape = (
                1,
                config.workspace.num_slots,
                config.cognitive.state_dim,
            )
            for cycle in trace.cycles:
                if (
                    cycle.slots.shape != expected_slots_shape
                    or cycle.slots.dtype != expected_dtype
                    or cycle.slots.device != runtime.device
                ):
                    message = "arm D workspace trace layout disagrees with configuration"
                    raise ValueError(message)
        return self


class IrisRequestArtifactEnvelopeV2(_RequestArtifactModelV2):
    """Canonical envelope for one request artifact v2."""

    schema_identity: RequestArtifactSchemaV2 = Field(alias="schema")
    payload_sha256: str
    payload: IrisRequestArtifactPayloadV2

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="request artifact payload_sha256")


__all__ = [
    "EmptyWorkspaceTraceProvenanceV2",
    "IrisRequestArtifactEnvelopeV2",
    "IrisRequestArtifactPayloadV2",
    "RequestArmATopologyV2",
    "RequestArmBTopologyV2",
    "RequestArmDTopologyV2",
    "RequestArmIdV2",
    "RequestArmKindV2",
    "RequestArmTopologyV2",
    "RequestArtifactSchemaV2",
    "RequestGraphV2",
    "RequestRuntimeProvenanceV2",
    "RequestTraceContractV2",
    "RequestWorkspaceTraceProvenanceV2",
    "WorkspaceTraceProvenanceV2",
]
