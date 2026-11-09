"""Strict persisted schema for the privacy-preserving request artifact v1."""

from math import isfinite, prod
from re import fullmatch
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.schemas import IrisBootstrapConfig


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
        populate_by_name=True,
    )


class SchemaIdentityV1(_ArtifactModel):
    artifact_type: Literal["iris.request"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "request artifact version must be an exact integer"
            raise ValueError(message)
        return value


class _HasSha256(_ArtifactModel):
    sha256: str

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "sha256 must be exactly 64 lower-case hexadecimal characters"
            raise ValueError(message)
        return value


class PrivateTextFingerprintV1(_HasSha256):
    representation: Literal["sha256_only"]
    encoding: Literal["utf-8"]
    normalization: Literal["none"]
    byte_length: int = Field(ge=0, strict=True)


class TensorFingerprintV1(_HasSha256):
    shape: tuple[int, ...]
    dtype: Literal["float16", "bfloat16", "float32", "int64", "bool"]
    device: str = Field(min_length=1)
    byte_length: int = Field(ge=0, strict=True)

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(type(dimension) is not int or dimension < 0 for dimension in value):
            message = "tensor shape must contain nonnegative exact integer dimensions"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_byte_length(self) -> Self:
        element_size = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int64": 8,
            "bool": 1,
        }[self.dtype]
        if self.byte_length != prod(self.shape) * element_size:
            message = "tensor byte_length must match shape and dtype"
            raise ValueError(message)
        return self


class ConfigurationProvenanceV1(_HasSha256):
    value: IrisBootstrapConfig


class DecodingProvenanceV1(_ArtifactModel):
    strategy: Literal["greedy", "sampling"]
    stop_token_ids: tuple[int, ...]
    temperature: float = Field(gt=0.0, strict=True)
    top_p: float = Field(gt=0.0, le=1.0, strict=True)
    sampling_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    allow_download: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if not self.stop_token_ids:
            message = "decoding provenance requires at least one stop token"
            raise ValueError(message)
        if any(type(token_id) is not int or token_id < 0 for token_id in self.stop_token_ids):
            message = "stop token IDs must be nonnegative exact integers"
            raise ValueError(message)
        if len(set(self.stop_token_ids)) != len(self.stop_token_ids):
            message = "stop token IDs must be unique"
            raise ValueError(message)
        if not isfinite(self.temperature) or not isfinite(self.top_p):
            message = "sampling policy values must be finite"
            raise ValueError(message)
        if self.strategy == "greedy" and self.sampling_seed is not None:
            message = "greedy provenance cannot contain a sampling seed"
            raise ValueError(message)
        if self.strategy == "sampling" and self.sampling_seed is None:
            message = "sampling provenance requires an explicit sampling seed"
            raise ValueError(message)
        return self


class RequestProvenanceV1(_ArtifactModel):
    prompt: PrivateTextFingerprintV1
    prompt_input_ids: TensorFingerprintV1
    prompt_attention_mask: TensorFingerprintV1
    physical_prompt_tokens: int = Field(gt=0, strict=True)
    attended_prompt_tokens: int = Field(gt=0, strict=True)
    decoding: DecodingProvenanceV1

    @model_validator(mode="after")
    def validate_token_provenance(self) -> Self:
        input_ids = self.prompt_input_ids
        attention_mask = self.prompt_attention_mask
        if (
            input_ids.shape != attention_mask.shape
            or len(input_ids.shape) != 2
            or input_ids.shape[0] != 1
            or input_ids.dtype != "int64"
            or attention_mask.dtype != "int64"
            or input_ids.device != attention_mask.device
            or self.physical_prompt_tokens != input_ids.shape[1]
            or self.attended_prompt_tokens > self.physical_prompt_tokens
        ):
            message = "request prompt token provenance is internally inconsistent"
            raise ValueError(message)
        return self


class SoftwareProvenanceV1(_ArtifactModel):
    iris: str = Field(min_length=1)
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    transformers: str = Field(min_length=1)
    tokenizers: str = Field(min_length=1)
    safetensors: str = Field(min_length=1)
    hydra_core: str = Field(min_length=1)
    omegaconf: str = Field(min_length=1)


class RuntimeProvenanceV1(_ArtifactModel):
    initialization_seed: int = Field(ge=0, le=2**63 - 1, strict=True)
    deterministic: bool = Field(strict=True)
    backend: Literal["cpu", "cuda"]
    device: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    dtype: Literal["float16", "bfloat16", "float32"]
    sidecar_state_sha256: str
    software: SoftwareProvenanceV1

    @field_validator("sidecar_state_sha256", mode="before")
    @classmethod
    def validate_sidecar_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "sidecar_state_sha256 must be lower-case full SHA-256"
            raise ValueError(message)
        return value


class GenerationProvenanceV1(_ArtifactModel):
    text: PrivateTextFingerprintV1
    token_ids: TensorFingerprintV1
    generated_tokens: int = Field(gt=0, strict=True)
    stop_reason: Literal["stop_token", "max_new_tokens"]
    matched_stop_token_id: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_generation_metadata(self) -> Self:
        if (
            len(self.token_ids.shape) != 2
            or self.token_ids.shape[0] != 1
            or self.token_ids.dtype != "int64"
            or self.generated_tokens != self.token_ids.shape[1]
        ):
            message = "generation token provenance is internally inconsistent"
            raise ValueError(message)
        if self.stop_reason == "stop_token" and self.matched_stop_token_id is None:
            message = "stop-token provenance requires a matched stop token"
            raise ValueError(message)
        if self.stop_reason == "max_new_tokens" and self.matched_stop_token_id is not None:
            message = "maximum-length provenance cannot contain a matched stop token"
            raise ValueError(message)
        return self


class WorkspaceCycleProvenanceV1(_HasSha256):
    cycle: int = Field(gt=0, strict=True)
    slots: TensorFingerprintV1
    age: TensorFingerprintV1
    occupancy: TensorFingerprintV1
    source_module: TensorFingerprintV1
    confidence: TensorFingerprintV1
    ttl: TensorFingerprintV1
    write_count: TensorFingerprintV1

    @model_validator(mode="after")
    def validate_state_layout(self) -> Self:
        if len(self.slots.shape) != 3 or self.slots.shape[0] != 1:
            message = "workspace slots fingerprint must have shape [1,K,D]"
            raise ValueError(message)
        metadata_shape = self.slots.shape[:2]
        metadata = {
            "age": (self.age, "int64"),
            "occupancy": (self.occupancy, "bool"),
            "source_module": (self.source_module, "int64"),
            "confidence": (self.confidence, self.slots.dtype),
            "ttl": (self.ttl, self.slots.dtype),
            "write_count": (self.write_count, "int64"),
        }
        if self.slots.dtype not in {"float16", "bfloat16", "float32"}:
            message = "workspace slot fingerprint must use a floating dtype"
            raise ValueError(message)
        for name, (tensor, dtype) in metadata.items():
            if (
                tensor.shape != metadata_shape
                or tensor.dtype != dtype
                or tensor.device != self.slots.device
            ):
                message = f"workspace {name} fingerprint disagrees with slot layout"
                raise ValueError(message)
        return self


class WorkspaceTraceProvenanceV1(_HasSha256):
    cycle_count: int = Field(gt=0, strict=True)
    cycles: tuple[WorkspaceCycleProvenanceV1, ...]

    @model_validator(mode="after")
    def validate_cycles(self) -> Self:
        if len(self.cycles) != self.cycle_count:
            message = "workspace cycle_count must equal the number of cycle records"
            raise ValueError(message)
        if tuple(cycle.cycle for cycle in self.cycles) != tuple(range(1, self.cycle_count + 1)):
            message = "workspace cycles must be contiguous and one-indexed"
            raise ValueError(message)
        return self


class IrisRequestArtifactPayloadV1(_ArtifactModel):
    configuration: ConfigurationProvenanceV1
    request: RequestProvenanceV1
    runtime: RuntimeProvenanceV1
    generation: GenerationProvenanceV1
    workspace_trace: WorkspaceTraceProvenanceV1

    @model_validator(mode="after")
    def validate_cross_component_provenance(self) -> Self:
        config = self.configuration.value
        decoding = self.request.decoding
        expected_dtype = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[config.hardware.precision]
        if (
            self.runtime.initialization_seed != config.runtime.seed
            or self.runtime.deterministic is not config.runtime.deterministic
            or self.runtime.backend != config.hardware.device
            or self.runtime.dtype != expected_dtype
            or config.backbone.local_files_only is decoding.allow_download
        ):
            message = "runtime or download provenance disagrees with effective configuration"
            raise ValueError(message)
        if self.request.physical_prompt_tokens > config.backbone.limits.max_prompt_tokens:
            message = "prompt provenance exceeds the configured prompt limit"
            raise ValueError(message)
        if self.generation.generated_tokens > config.backbone.limits.max_new_tokens:
            message = "generation provenance exceeds the configured generation limit"
            raise ValueError(message)
        if (
            self.generation.stop_reason == "max_new_tokens"
            and self.generation.generated_tokens != config.backbone.limits.max_new_tokens
        ):
            message = "maximum-length provenance has not reached the configured limit"
            raise ValueError(message)
        if self.workspace_trace.cycle_count != config.workspace.cycles:
            message = "workspace trace provenance disagrees with configured cycles"
            raise ValueError(message)
        expected_slots_shape = (
            1,
            config.workspace.num_slots,
            config.cognitive.state_dim,
        )
        for cycle in self.workspace_trace.cycles:
            if cycle.slots.shape != expected_slots_shape or cycle.slots.dtype != expected_dtype:
                message = "workspace trace layout disagrees with effective configuration"
                raise ValueError(message)
            if cycle.slots.device != self.runtime.device:
                message = "workspace trace device disagrees with resolved runtime"
                raise ValueError(message)
        if self.request.prompt_input_ids.device != self.runtime.device:
            message = "prompt token provenance device disagrees with resolved runtime"
            raise ValueError(message)
        if self.generation.token_ids.device != self.runtime.device:
            message = "generation token provenance device disagrees with resolved runtime"
            raise ValueError(message)
        return self


class IrisRequestArtifactEnvelopeV1(_ArtifactModel):
    schema_identity: SchemaIdentityV1 = Field(alias="schema")
    payload_sha256: str
    payload: IrisRequestArtifactPayloadV1

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "payload_sha256 must be lower-case full SHA-256"
            raise ValueError(message)
        return value
