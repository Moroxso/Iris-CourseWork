"""Strict persisted full-scientific-run checkpoint schema; no pickle payloads."""

from math import prod
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=512, pattern=r"^[A-Za-z0-9_./]+$")]


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class ScientificTensorV1(_Strict):
    key: Name
    shape: tuple[Annotated[int, Field(ge=0, le=2**24)], ...] = Field(max_length=8)
    dtype: Literal["float32", "uint8"]
    byte_length: int = Field(ge=0, le=2**31)

    @model_validator(mode="after")
    def validate_bytes(self) -> Self:
        if self.byte_length != prod(self.shape) * (4 if self.dtype == "float32" else 1):
            raise ValueError("Tensor size differs from dtype/shape")
        return self


class ScientificRuntimeV1(_Strict):
    python_version: int = Field(default=3, ge=3, le=3)
    python_state: tuple[Annotated[int, Field(ge=0, le=2**32 - 1)], ...] = Field(
        min_length=625, max_length=625
    )
    python_gaussian: float | None
    device: str = Field(pattern=r"^(cpu|cuda:[0-9]+)$")
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool

    @model_validator(mode="after")
    def validate_python_rng(self) -> Self:
        if self.python_state[-1] > 624:
            raise ValueError("Invalid Python RNG position")
        return self


class ScientificCheckpointPayloadV1(_Strict):
    kind: Literal["iris-scientific-checkpoint-v1"] = "iris-scientific-checkpoint-v1"
    configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    configuration_sha256: Digest
    completed_steps: int = Field(ge=0, le=2**24)
    examples_seen: int = Field(ge=0, le=2**48)
    target_tokens_seen: int = Field(ge=0, le=2**48)
    history_sha256: Digest
    parameter_names: tuple[Name, ...] = Field(min_length=1, max_length=4096)
    parameter_steps: tuple[Annotated[int, Field(ge=0, le=2**24)], ...] = Field(
        min_length=1, max_length=4096
    )
    model_state_names: tuple[Name, ...] = Field(min_length=1, max_length=8192)
    runtime: ScientificRuntimeV1
    tensors: tuple[ScientificTensorV1, ...] = Field(min_length=1, max_length=16384)
    state_sha256: Digest
    state_size_bytes: int = Field(gt=0, le=2**31)

    @model_validator(mode="after")
    def validate_tables(self) -> Self:
        if (
            len(set(self.parameter_names)) != len(self.parameter_names)
            or len(set(self.model_state_names)) != len(self.model_state_names)
            or len(self.parameter_steps) != len(self.parameter_names)
            or any(count > self.completed_steps for count in self.parameter_steps)
            or max(self.parameter_steps) != self.completed_steps
            or not set(self.parameter_names).issubset(self.model_state_names)
        ):
            raise ValueError("Checkpoint state topology/counters are inconsistent")
        expected = {f"model/{name}" for name in self.model_state_names} | {"runtime/cpu"}
        for index, count in enumerate(self.parameter_steps):
            if count:
                expected |= {f"adamw/{index}/exp_avg", f"adamw/{index}/exp_avg_sq"}
        if self.runtime.device != "cpu":
            expected.add("runtime/cuda")
        keys = tuple(tensor.key for tensor in self.tensors)
        if (
            keys != tuple(sorted(expected))
            or sum(t.byte_length for t in self.tensors) >= self.state_size_bytes
        ):
            raise ValueError("Checkpoint tensor table membership/size differs")
        if not self.completed_steps and (
            self.examples_seen or self.target_tokens_seen or any(self.parameter_steps)
        ):
            raise ValueError("An initial checkpoint cannot have training progress")
        return self


class IrisScientificCheckpointManifestV1(_Strict):
    sha256: Digest
    payload: ScientificCheckpointPayloadV1


__all__ = [
    "IrisScientificCheckpointManifestV1",
    "ScientificCheckpointPayloadV1",
    "ScientificRuntimeV1",
    "ScientificTensorV1",
]
