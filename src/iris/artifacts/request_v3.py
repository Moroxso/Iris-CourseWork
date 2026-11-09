"""Immutable A/B/C/D request-artifact schema bound to execution profiles v4."""

from __future__ import annotations

from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    GenerationProvenanceV1,
    RequestProvenanceV1,
    TensorFingerprintV1,
)
from iris.artifacts.request_v2 import (
    EmptyWorkspaceTraceProvenanceV2,
    RequestArmATopologyV2,
    RequestArmBTopologyV2,
    RequestArmDTopologyV2,
    RequestRuntimeProvenanceV2,
    WorkspaceTraceProvenanceV2,
)

_PROTOCOL_SHA256 = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_PROFILE_SHA256 = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_POLICY_SHA256 = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
_ARM_C_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
    "prefix_assembler",
)

type RequestArmIdV3 = Literal["A", "B", "C", "D"]
type RequestArmKindV3 = Literal[
    "base_llm",
    "llm_modules_no_workspace",
    "llm_modules_unlimited_shared_buffer",
    "llm_modules_limited_competitive_workspace",
]
type RequestGraphV3 = Literal[
    "frozen_backbone_prompt_only_v1",
    "independent_recurrent_modules_zero_memory_read_v1",
    "append_only_all_proposals_prior_cycle_shared_buffer_v1",
    "limited_competitive_workspace_v1",
]
type RequestTraceContractV3 = Literal[
    "empty_workspace_trace",
    "append_only_shared_buffer_per_cycle_v1",
    "workspace_state_per_cycle",
]


class _RequestArtifactModelV3(BaseModel):
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


class RequestArtifactSchemaV3(_RequestArtifactModelV3):
    """Schema identity for one immutable A/B/C/D request artifact."""

    artifact_type: Literal["iris.request"]
    version: Literal[3]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_values_sha256_only"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        return _validate_exact_integer(value, label="request artifact version")


class RequestArmATopologyV3(RequestArmATopologyV2):
    """Version-3 ownership of the unchanged prompt-only topology."""


class RequestArmBTopologyV3(RequestArmBTopologyV2):
    """Version-3 ownership of the unchanged no-workspace topology."""


class RequestArmDTopologyV3(RequestArmDTopologyV2):
    """Version-3 ownership of the unchanged limited-workspace topology."""


class RequestArmCTopologyV3(_RequestArtifactModelV3):
    """Exact append-only, all-proposals, prior-cycle shared-buffer topology."""

    arm_id: Literal["C"]
    arm_kind: Literal["llm_modules_unlimited_shared_buffer"]
    request_graph: Literal["append_only_all_proposals_prior_cycle_shared_buffer_v1"]
    memory_model: Literal["unlimited_shared_buffer"]
    trace_contract: Literal["append_only_shared_buffer_per_cycle_v1"]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    cognitive_cycles: Literal[3]
    pre_update_read_count: Literal[3]
    final_read_count: Literal[1]
    entries_per_cycle: int = Field(gt=0, strict=True)
    final_entry_count: int = Field(gt=0, strict=True)
    latent_token_count: int = Field(gt=0, strict=True)
    sidecar_component_names: tuple[str, ...]
    sidecar_state_sha256: str

    @field_validator(
        "cognitive_cycles",
        "pre_update_read_count",
        "final_read_count",
        mode="before",
    )
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        return _validate_exact_integer(value, label="arm-C topology count")

    @field_validator("sidecar_component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != _ARM_C_COMPONENTS:
            message = "arm-C request sidecar component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("sidecar_state_sha256", mode="before")
    @classmethod
    def validate_sidecar_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="arm-C sidecar_state_sha256")

    @model_validator(mode="after")
    def validate_cardinality(self) -> Self:
        if (
            self.final_entry_count != self.cognitive_cycles * self.entries_per_cycle
            or self.latent_token_count != self.entries_per_cycle
        ):
            message = "arm-C topology cardinalities do not encode 0→M→2M→3M and M latents"
            raise ValueError(message)
        return self


RequestArmTopologyV3 = Annotated[
    RequestArmATopologyV3 | RequestArmBTopologyV3 | RequestArmCTopologyV3 | RequestArmDTopologyV3,
    Field(discriminator="arm_id"),
]


class RequestRuntimeProvenanceV3(RequestRuntimeProvenanceV2):
    """Version-3 ownership of resolved runtime provenance."""


class EmptyWorkspaceTraceProvenanceV3(EmptyWorkspaceTraceProvenanceV2):
    """Version-3 ownership of the exact empty A/B trace."""


class WorkspaceTraceProvenanceV3(WorkspaceTraceProvenanceV2):
    """Version-3 ownership of the unchanged limited-workspace D trace."""


class SharedBufferStateProvenanceV3(_RequestArtifactModelV3):
    """Fingerprints for one native post-cycle append-only buffer state."""

    cycle: int = Field(gt=0, strict=True)
    entry_count: int = Field(gt=0, strict=True)
    entries: TensorFingerprintV1
    source_module: TensorFingerprintV1
    write_cycle: TensorFingerprintV1
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="shared-buffer state sha256")

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if (
            len(self.entries.shape) != 3
            or self.entries.shape[0] != 1
            or self.entries.shape[1] != self.entry_count
            or self.entries.dtype not in {"float16", "bfloat16", "float32"}
        ):
            message = "shared-buffer entries fingerprint has an invalid layout"
            raise ValueError(message)
        metadata_shape = self.entries.shape[:2]
        for name, tensor in (
            ("source_module", self.source_module),
            ("write_cycle", self.write_cycle),
        ):
            if (
                tensor.shape != metadata_shape
                or tensor.dtype != "int64"
                or tensor.device != self.entries.device
            ):
                message = f"shared-buffer {name} fingerprint disagrees with entries"
                raise ValueError(message)
        return self


class SharedBufferReadProvenanceV3(_RequestArtifactModelV3):
    """Private fingerprints for one all-entry module broadcast read."""

    read_kind: Literal["pre_update", "final_output"]
    cycle: int = Field(gt=0, strict=True)
    entry_count: int = Field(ge=0, strict=True)
    workspace_reads: TensorFingerprintV1
    attention_weights: TensorFingerprintV1
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="shared-buffer read sha256")

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        reads = self.workspace_reads
        attention = self.attention_weights
        if (
            len(reads.shape) != 3
            or reads.shape[0] != 1
            or reads.shape[1] <= 0
            or reads.shape[2] <= 0
            or reads.dtype not in {"float16", "bfloat16", "float32"}
            or attention.shape != (1, reads.shape[1], self.entry_count)
            or attention.dtype not in {"float16", "bfloat16", "float32"}
            or attention.device != reads.device
        ):
            message = "shared-buffer reader fingerprints have an invalid layout"
            raise ValueError(message)
        return self


class SharedBufferCycleProvenanceV3(_RequestArtifactModelV3):
    """One pre-update read followed by its same-cycle post-write native state."""

    cycle: int = Field(gt=0, strict=True)
    pre_update_read: SharedBufferReadProvenanceV3
    post_cycle_state: SharedBufferStateProvenanceV3
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="shared-buffer cycle sha256")

    @model_validator(mode="after")
    def validate_cycle_identity(self) -> Self:
        if (
            self.pre_update_read.read_kind != "pre_update"
            or self.pre_update_read.cycle != self.cycle
            or self.post_cycle_state.cycle != self.cycle
        ):
            message = "shared-buffer cycle read/state provenance is not aligned"
            raise ValueError(message)
        return self


class SharedBufferTraceProvenanceV3(_RequestArtifactModelV3):
    """Complete 0/M/2M reads, M/2M/3M states, and final 3M read."""

    trace_contract: Literal["append_only_shared_buffer_per_cycle_v1"]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    cycle_count: Literal[3]
    pre_update_read_count: Literal[3]
    final_read_count: Literal[1]
    entries_per_cycle: int = Field(gt=0, strict=True)
    final_entry_count: int = Field(gt=0, strict=True)
    latent_token_count: int = Field(gt=0, strict=True)
    cycles: tuple[
        SharedBufferCycleProvenanceV3,
        SharedBufferCycleProvenanceV3,
        SharedBufferCycleProvenanceV3,
    ]
    final_output_read: SharedBufferReadProvenanceV3
    sha256: str

    @field_validator(
        "cycle_count",
        "pre_update_read_count",
        "final_read_count",
        mode="before",
    )
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        return _validate_exact_integer(value, label="shared-buffer trace count")

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="shared-buffer trace sha256")

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if tuple(cycle.cycle for cycle in self.cycles) != (1, 2, 3):
            message = "shared-buffer trace cycles must be contiguous and one-indexed"
            raise ValueError(message)
        if self.final_entry_count != self.cycle_count * self.entries_per_cycle:
            message = "shared-buffer final entry count must equal 3M"
            raise ValueError(message)
        if self.latent_token_count != self.entries_per_cycle:
            message = "shared-buffer final read must bridge exactly M latent tokens"
            raise ValueError(message)
        for cycle in self.cycles:
            if (
                cycle.pre_update_read.entry_count != (cycle.cycle - 1) * self.entries_per_cycle
                or cycle.post_cycle_state.entry_count != cycle.cycle * self.entries_per_cycle
            ):
                message = "shared-buffer trace does not encode 0/M/2M reads and M/2M/3M writes"
                raise ValueError(message)
        if (
            self.final_output_read.read_kind != "final_output"
            or self.final_output_read.cycle != self.cycle_count
            or self.final_output_read.entry_count != self.final_entry_count
        ):
            message = "shared-buffer final output read must cover the complete 3M buffer"
            raise ValueError(message)
        return self


RequestWorkspaceTraceProvenanceV3 = Annotated[
    EmptyWorkspaceTraceProvenanceV3 | SharedBufferTraceProvenanceV3 | WorkspaceTraceProvenanceV3,
    Field(discriminator="trace_contract"),
]


class IrisRequestArtifactPayloadV3(_RequestArtifactModelV3):
    """Profile-v4 request identity plus private values represented only by hashes."""

    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    execution_profile_set_version: Literal[4]
    execution_profile_set_sha256: Literal[
        "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
    ]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    topology: RequestArmTopologyV3
    configuration: ConfigurationProvenanceV1
    request: RequestProvenanceV1
    runtime: RequestRuntimeProvenanceV3
    generation: GenerationProvenanceV1
    workspace_trace: RequestWorkspaceTraceProvenanceV3

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
            or self.request.prompt_attention_mask.device != runtime.device
            or self.generation.token_ids.device != runtime.device
        ):
            message = "request tensor provenance disagrees with the resolved runtime device"
            raise ValueError(message)
        if topology.trace_contract != trace.trace_contract:
            message = "request topology and trace contracts disagree"
            raise ValueError(message)

        if isinstance(topology, RequestArmATopologyV3):
            if runtime.sidecar_state_sha256 is not None or not isinstance(
                trace, EmptyWorkspaceTraceProvenanceV3
            ):
                message = "arm A requires no sidecar and an exact empty trace"
                raise ValueError(message)
        elif isinstance(topology, RequestArmBTopologyV3):
            if (
                runtime.sidecar_state_sha256 != topology.sidecar_state_sha256
                or config.cognitive.num_modules != 8
                or config.cognitive.state_dim != 256
                or topology.latent_token_count != config.cognitive.num_modules
                or not isinstance(trace, EmptyWorkspaceTraceProvenanceV3)
            ):
                message = "arm B topology disagrees with sidecar, cognitive config, or trace"
                raise ValueError(message)
        elif isinstance(topology, RequestArmCTopologyV3):
            if (
                runtime.sidecar_state_sha256 != topology.sidecar_state_sha256
                or topology.shared_buffer_policy_sha256 != self.shared_buffer_policy_sha256
                or config.cognitive.num_modules != 8
                or config.cognitive.state_dim != 256
                or topology.entries_per_cycle != config.cognitive.num_modules
                or topology.final_entry_count != 3 * config.cognitive.num_modules
                or topology.latent_token_count != config.cognitive.num_modules
                or not isinstance(trace, SharedBufferTraceProvenanceV3)
                or trace.shared_buffer_policy_sha256 != self.shared_buffer_policy_sha256
                or trace.entries_per_cycle != config.cognitive.num_modules
                or trace.final_entry_count != 3 * config.cognitive.num_modules
                or trace.latent_token_count != config.cognitive.num_modules
            ):
                message = (
                    "arm C topology disagrees with policy, sidecar, cognitive config, or trace"
                )
                raise ValueError(message)
            expected_reads_shape = (1, config.cognitive.num_modules, config.cognitive.state_dim)
            for cycle in trace.cycles:
                state = cycle.post_cycle_state
                read = cycle.pre_update_read
                if (
                    state.entries.shape
                    != (1, cycle.cycle * config.cognitive.num_modules, config.cognitive.state_dim)
                    or state.entries.dtype != expected_dtype
                    or state.entries.device != runtime.device
                    or read.workspace_reads.shape != expected_reads_shape
                    or read.workspace_reads.dtype != expected_dtype
                    or read.workspace_reads.device != runtime.device
                    or read.attention_weights.dtype
                    != ("float32" if expected_dtype != "float32" else expected_dtype)
                ):
                    message = "arm C native state or reader layout disagrees with configuration"
                    raise ValueError(message)
            final_read = trace.final_output_read
            if (
                final_read.workspace_reads.shape != expected_reads_shape
                or final_read.workspace_reads.dtype != expected_dtype
                or final_read.workspace_reads.device != runtime.device
                or final_read.attention_weights.dtype
                != ("float32" if expected_dtype != "float32" else expected_dtype)
            ):
                message = "arm C final reader layout disagrees with configuration"
                raise ValueError(message)
        else:
            assert isinstance(topology, RequestArmDTopologyV3)
            if (
                runtime.sidecar_state_sha256 != topology.sidecar_state_sha256
                or config.cognitive.num_modules != 8
                or config.cognitive.state_dim != 256
                or config.workspace.num_slots != 8
                or config.workspace.winners_per_cycle != 2
                or config.workspace.cycles != 3
                or topology.cognitive_cycles != config.workspace.cycles
                or topology.latent_token_count != config.workspace.num_slots
                or not isinstance(trace, WorkspaceTraceProvenanceV3)
                or trace.cycle_count != config.workspace.cycles
            ):
                message = "arm D topology disagrees with sidecar, workspace config, or trace"
                raise ValueError(message)
            expected_slots_shape = (
                1,
                config.workspace.num_slots,
                config.cognitive.state_dim,
            )
            for workspace_cycle in trace.cycles:
                if (
                    workspace_cycle.slots.shape != expected_slots_shape
                    or workspace_cycle.slots.dtype != expected_dtype
                    or workspace_cycle.slots.device != runtime.device
                ):
                    message = "arm D workspace trace layout disagrees with configuration"
                    raise ValueError(message)
        return self


class IrisRequestArtifactEnvelopeV3(_RequestArtifactModelV3):
    """Canonical envelope for one request artifact v3."""

    schema_identity: RequestArtifactSchemaV3 = Field(alias="schema")
    payload_sha256: str
    payload: IrisRequestArtifactPayloadV3

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="request artifact payload_sha256")


__all__ = [
    "EmptyWorkspaceTraceProvenanceV3",
    "IrisRequestArtifactEnvelopeV3",
    "IrisRequestArtifactPayloadV3",
    "RequestArmATopologyV3",
    "RequestArmBTopologyV3",
    "RequestArmCTopologyV3",
    "RequestArmDTopologyV3",
    "RequestArmIdV3",
    "RequestArmKindV3",
    "RequestArmTopologyV3",
    "RequestArtifactSchemaV3",
    "RequestGraphV3",
    "RequestRuntimeProvenanceV3",
    "RequestTraceContractV3",
    "RequestWorkspaceTraceProvenanceV3",
    "SharedBufferCycleProvenanceV3",
    "SharedBufferReadProvenanceV3",
    "SharedBufferStateProvenanceV3",
    "SharedBufferTraceProvenanceV3",
    "WorkspaceTraceProvenanceV3",
]
