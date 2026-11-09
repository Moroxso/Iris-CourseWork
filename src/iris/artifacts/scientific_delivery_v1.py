"""Source-section22 training delivery configuration and optional validation metric."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import (
    Digest,
    IrisScientificCheckpointManifestV1,
    _Strict,
)
from iris.training.probes import IrisProbeVocabulary


class ScientificDeliveryConfigurationV1(_Strict):
    kind: Literal["iris-scientific-delivery-v1"] = "iris-scientific-delivery-v1"
    checkpoint: IrisScientificCheckpointManifestV1
    teacher_configuration_json: str = Field(min_length=1, max_length=1024**2)
    probe_configuration_json: str = Field(min_length=1, max_length=1024**2)
    probe_vocabulary: IrisProbeVocabulary
    tokenizer_json_sha256: Digest
    tokenizer_semantics_sha256: Digest


class ScientificDeliveryValidationV1(_Strict):
    dataset_sha256: Digest
    example_count: int = Field(ge=1, le=5000)
    target_token_count: int = Field(ge=1, le=2**40)
    target_nll_sum: float = Field(ge=0.0)
    target_nll_per_token: float = Field(ge=0.0)
    at_optimizer_step: int = Field(ge=0, le=2**24)
    run_configuration_sha256: Digest

    @model_validator(mode="after")
    def validate_reduction(self) -> Self:
        if self.target_nll_per_token != self.target_nll_sum / self.target_token_count:
            raise ValueError("Validation metric reduction differs")
        return self


class ScientificDeliveryMetricsV1(_Strict):
    kind: Literal["iris-scientific-delivery-metrics-v1"] = "iris-scientific-delivery-metrics-v1"
    validation: ScientificDeliveryValidationV1 | None = None


__all__ = ["ScientificDeliveryConfigurationV1", "ScientificDeliveryMetricsV1"]
