"""Validated configuration schemas for Iris."""

from itertools import pairwise
from math import isfinite
from re import fullmatch
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator, model_validator


class HardwareConfig(BaseModel):
    """Explicit hardware selection for one Iris run."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    name: str
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "fp16", "bf16"]
    max_memory_gb: float = Field(gt=0, strict=True)
    allow_gpu: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_cpu_precision(self) -> Self:
        """Reject reduced precision on CPU in Iris-GW-0.1."""
        if self.device == "cpu" and self.precision != "fp32":
            message = "Iris-GW-0.1 supports only fp32 precision on CPU"
            raise ValueError(message)
        if self.device == "cuda" and not self.allow_gpu:
            message = "CUDA hardware selection requires allow_gpu=true"
            raise ValueError(message)
        return self


class TokenizerConfig(BaseModel):
    """Tokenizer behavior for deterministic prompt preparation."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    use_fast: bool = Field(strict=True)
    padding_side: Literal["left", "right"]
    truncation_side: Literal["left", "right"]


class BackboneLimitsConfig(BaseModel):
    """Token limits enforced around development backbone calls."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    max_prompt_tokens: int = Field(gt=0, strict=True)
    max_new_tokens: int = Field(gt=0, strict=True)


class BackboneConfig(BaseModel):
    """Frozen Hugging Face backbone loading configuration."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    name: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    trust_remote_code: Literal[False] = False
    local_files_only: bool = False
    output_hidden_states: Literal[True] = True
    use_cache: Literal[False] = False
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    tokenizer: TokenizerConfig
    limits: BackboneLimitsConfig

    @field_validator(
        "trust_remote_code",
        "local_files_only",
        "output_hidden_states",
        "use_cache",
        mode="before",
    )
    @classmethod
    def validate_strict_runtime_flags(cls, value: object) -> object:
        """Reject numeric and string coercions at the model-loading boundary."""
        if type(value) is not bool:
            message = "Backbone runtime flags must be explicit booleans"
            raise ValueError(message)
        return value

    @field_validator("name", "repository", "revision", "tokenizer_revision")
    @classmethod
    def validate_nonblank_identity(cls, value: str) -> str:
        """Reject loader identities that contain no usable characters."""
        if not value.strip():
            message = "Backbone identity fields must not be blank"
            raise ValueError(message)
        return value


class LayerSamplerConfig(BaseModel):
    """Relative transformer depths sampled for Iris context features."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    layer_fractions: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]

    @field_validator("layer_fractions")
    @classmethod
    def validate_layer_fractions(
        cls, fractions: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Require four unique depths ending at the final transformer layer."""
        if any(
            not isfinite(fraction) or fraction <= 0.0 or fraction > 1.0 for fraction in fractions
        ):
            message = "Layer fractions must be in the interval (0, 1]"
            raise ValueError(message)
        if any(left >= right for left, right in pairwise(fractions)):
            message = "Layer fractions must be strictly increasing"
            raise ValueError(message)
        if fractions[-1] != 1.0:
            message = "The final layer fraction must equal 1.0"
            raise ValueError(message)
        return fractions


class CognitiveConfig(BaseModel):
    """Shared latent dimensions for the Iris-GW cognitive sidecar."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    name: str
    state_dim: int = Field(gt=0, strict=True)
    num_modules: int = Field(gt=0, strict=True)
    layer_sampler: LayerSamplerConfig


class WorkspaceArbiterConfig(BaseModel):
    """Fixed score weights for Iris-GW workspace competition."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    confidence_weight: float = Field(ge=0.0, strict=True)
    novelty_weight: float = Field(ge=0.0, strict=True)
    relevance_weight: float = Field(ge=0.0, strict=True)


class SlotAllocatorConfig(BaseModel):
    """Fixed coefficients for winner-to-slot assignment and replacement."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    empty_bonus: float = Field(ge=0.0, strict=True)
    allocation_age_penalty: float = Field(ge=0.0, strict=True)
    retention_age_penalty: float = Field(ge=0.0, strict=True)


class WorkspaceConfig(BaseModel):
    """Global-workspace dimensions and lifecycle constants for Iris-GW."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    name: str
    num_slots: int = Field(gt=0, strict=True)
    cycles: int = Field(gt=0, strict=True)
    winners_per_cycle: int = Field(gt=0, strict=True)
    decay_rate: float = Field(gt=0.0, lt=1.0, strict=True)
    empty_threshold: float = Field(gt=0.0, strict=True)
    arbiter: WorkspaceArbiterConfig
    allocator: SlotAllocatorConfig

    @model_validator(mode="after")
    def validate_winner_capacity(self) -> Self:
        """Keep the number of winners within available workspace capacity."""
        if self.winners_per_cycle > self.num_slots:
            message = "winners_per_cycle cannot exceed num_slots"
            raise ValueError(message)
        return self


class RuntimeConfig(BaseModel):
    """Deterministic execution and artifact settings."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    seed: int = Field(strict=True, ge=0, le=2**63 - 1)
    deterministic: bool = Field(strict=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    output_dir: str = Field(min_length=1)

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: str) -> str:
        """Reject a path that contains only whitespace."""
        if not value.strip():
            message = "output_dir must contain a non-whitespace path"
            raise ValueError(message)
        return value


class ProjectConfig(BaseModel):
    """Identity metadata carried by one composed Iris configuration."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class IrisBootstrapConfig(BaseModel):
    """Owned validated root of one normal Hydra request composition."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    project: ProjectConfig
    hardware: HardwareConfig
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def validate_cross_component_policy(self) -> Self:
        """Reject policy mismatches before device or model initialization."""
        if self.hardware.precision != self.backbone.dtype:
            message = "hardware precision must match backbone dtype"
            raise ValueError(message)
        if self.workspace.winners_per_cycle > self.cognitive.num_modules:
            message = "workspace winners_per_cycle cannot exceed cognitive num_modules"
            raise ValueError(message)
        for field_name, revision in (
            ("revision", self.backbone.revision),
            ("tokenizer_revision", self.backbone.tokenizer_revision),
        ):
            if fullmatch(r"[0-9a-f]{40}", revision) is None:
                message = f"backbone {field_name} must be an immutable 40-hex commit SHA"
                raise ValueError(message)
        return self
