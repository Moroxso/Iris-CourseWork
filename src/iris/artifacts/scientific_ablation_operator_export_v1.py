"""Explicit persisted-boundary export intent and result, not a training capability."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict

Selection = Literal["current", "selected_best"]


class ScientificAblationBestPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-operator-best-v1"] = (
        "iris-scientific-ablation-operator-best-v1"
    )
    operator_sha256: Digest
    execution_sha256: Digest
    head_sha256: Digest
    validated_best_event_sha256: Digest | None
    validated_best_checkpoint_sha256: Digest | None
    validated_best_step: int | None = Field(ge=1, le=2**24)
    persisted_checkpoint_event_sha256: Digest | None
    persisted_lifecycle_sha256: Digest | None
    persisted_best_checkpoint_sha256: Digest | None
    persisted_best_step: int | None = Field(ge=1, le=2**24)
    validated_best_is_persisted: bool

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            len(
                {
                    self.validated_best_event_sha256 is None,
                    self.validated_best_checkpoint_sha256 is None,
                    self.validated_best_step is None,
                }
            )
            != 1
            or (
                (self.persisted_checkpoint_event_sha256 is None)
                != (self.persisted_lifecycle_sha256 is None)
            )
            or (
                (self.persisted_best_checkpoint_sha256 is None)
                != (self.persisted_best_step is None)
            )
            or (
                self.persisted_best_checkpoint_sha256 is not None
                and self.persisted_lifecycle_sha256 is None
            )
            or self.validated_best_is_persisted
            != (
                self.validated_best_checkpoint_sha256 is not None
                and self.validated_best_checkpoint_sha256 == self.persisted_best_checkpoint_sha256
                and self.validated_best_step == self.persisted_best_step
            )
        ):
            raise ValueError("Validated-best metadata and persisted boundary differ")
        return self


class IrisScientificAblationBestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationBestPayloadV1


class ScientificAblationExportRequestPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-operator-export-request-v1"] = (
        "iris-scientific-ablation-operator-export-request-v1"
    )
    operator_sha256: Digest
    execution_sha256: Digest
    checkpoint_event_sha256: Digest
    lifecycle_sha256: Digest
    source_checkpoint_sha256: Digest
    selection: Selection
    purpose: Literal["scientific", "smoke"]
    maximum_storage_bytes: int = Field(ge=1, le=16 * 1024**3)
    minimum_free_bytes: int = Field(ge=0, le=16 * 1024**3)


class IrisScientificAblationExportRequestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationExportRequestPayloadV1


class ScientificAblationExportResultPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-operator-export-result-v1"] = (
        "iris-scientific-ablation-operator-export-result-v1"
    )
    request: IrisScientificAblationExportRequestV1
    inference_sha256: Digest
    selected_completed_steps: int = Field(ge=1, le=2**24)


class IrisScientificAblationExportResultV1(_Strict):
    sha256: Digest
    payload: ScientificAblationExportResultPayloadV1


__all__ = [
    "IrisScientificAblationBestV1",
    "IrisScientificAblationExportRequestV1",
    "IrisScientificAblationExportResultV1",
    "ScientificAblationBestPayloadV1",
    "ScientificAblationExportRequestPayloadV1",
    "ScientificAblationExportResultPayloadV1",
    "Selection",
]
