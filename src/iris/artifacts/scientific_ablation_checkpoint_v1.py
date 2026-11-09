"""Distinct, strict persisted scientific cognitive-ablation checkpoint; no pickle."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import (
    Digest,
    Name,
    ScientificRuntimeV1,
    ScientificTensorV1,
    _Strict,
)


class ScientificAblationCheckpointPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-checkpoint-v1"] = (
        "iris-scientific-ablation-checkpoint-v1"
    )
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
    scientific_state_sha256: Digest
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
            raise ValueError("Ablation checkpoint topology/counters differ")
        expected = {f"model/{name}" for name in self.model_state_names} | {"runtime/cpu"}
        for index, count in enumerate(self.parameter_steps):
            if count:
                expected |= {f"adamw/{index}/exp_avg", f"adamw/{index}/exp_avg_sq"}
        if self.runtime.device != "cpu":
            expected.add("runtime/cuda")
        if (
            tuple(entry.key for entry in self.tensors) != tuple(sorted(expected))
            or sum(entry.byte_length for entry in self.tensors) >= self.state_size_bytes
        ):
            raise ValueError("Ablation checkpoint tensor inventory/size differs")
        if not self.completed_steps and (self.examples_seen or self.target_tokens_seen):
            raise ValueError("Initial ablation checkpoint has training progress")
        return self


class IrisScientificAblationCheckpointManifestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCheckpointPayloadV1


__all__ = [
    "IrisScientificAblationCheckpointManifestV1",
    "ScientificAblationCheckpointPayloadV1",
]
