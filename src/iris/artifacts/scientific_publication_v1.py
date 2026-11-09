"""Non-authoritative delivery receipts bound to committed scientific checkpoints."""

from typing import Literal

from pydantic import Field

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificPublishedSelectionV1(_Strict):
    checkpoint_sha256: Digest
    delivery_sha256: Digest
    validation_event_sha256: Digest | None


class ScientificPublicationPayloadV1(_Strict):
    kind: Literal["iris-scientific-publication-v1"] = "iris-scientific-publication-v1"
    execution_sha256: Digest
    event_sequence: int = Field(ge=1, le=200000)
    event_sha256: Digest
    lifecycle_checkpoint_sha256: Digest
    current: ScientificPublishedSelectionV1
    best: ScientificPublishedSelectionV1 | None


class IrisScientificPublicationV1(_Strict):
    sha256: Digest
    payload: ScientificPublicationPayloadV1
