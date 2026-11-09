"""Strict scientific execution header and phase journal, separate from legacy logs."""

from typing import Annotated, Literal

from pydantic import Field

from iris.artifacts.scientific_checkpoint_v1 import (
    Digest,
    IrisScientificCheckpointManifestV1,
    _Strict,
)
from iris.artifacts.scientific_delivery_v1 import ScientificDeliveryValidationV1
from iris.artifacts.scientific_lifecycle_v1 import ScientificLifecycleStateV1


class ScientificExecutionHeaderPayloadV1(_Strict):
    kind: Literal["iris-scientific-execution-v1"] = "iris-scientific-execution-v1"
    lifecycle_configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    lifecycle_configuration_sha256: Digest
    initial: IrisScientificCheckpointManifestV1
    maximum_journal_bytes: int = Field(default=128 * 1024**2, ge=1, le=1024**3)


class IrisScientificExecutionHeaderV1(_Strict):
    sha256: Digest
    payload: ScientificExecutionHeaderPayloadV1


class ScientificExecutionTrainingMetricV1(_Strict):
    step_index: int = Field(ge=1, le=2**24)
    example_count: int = Field(ge=1, le=4096)
    supervised_token_count: int = Field(ge=1, le=1048576)
    learning_rate: float = Field(gt=0.0)
    gradient_l2_norm: float = Field(ge=0.0)
    gradients_clipped: bool
    loss: float = Field(ge=0.0)
    language_model_loss: float = Field(ge=0.0)
    probe_loss: float = Field(ge=0.0)
    diversity_loss: float = Field(ge=0.0)
    utilization_loss: float = Field(ge=0.0)
    norm_loss: float = Field(ge=0.0)
    configuration_sha256: Digest
    window_sha256: Digest
    purpose: Literal["scientific", "smoke"]


class ScientificExecutionTrainV1(_Strict):
    kind: Literal["train"] = "train"
    metric: ScientificExecutionTrainingMetricV1


class ScientificExecutionValidationV1(_Strict):
    kind: Literal["validation"] = "validation"
    metric: ScientificDeliveryValidationV1
    selected_checkpoint_sha256: Digest | None


class ScientificExecutionCheckpointV1(_Strict):
    kind: Literal["checkpoint"] = "checkpoint"
    lifecycle_checkpoint_sha256: Digest


ScientificExecutionRecordV1 = Annotated[
    ScientificExecutionTrainV1 | ScientificExecutionValidationV1 | ScientificExecutionCheckpointV1,
    Field(discriminator="kind"),
]


class ScientificExecutionEventPayloadV1(_Strict):
    kind: Literal["iris-scientific-execution-event-v1"] = "iris-scientific-execution-event-v1"
    execution_sha256: Digest
    sequence: int = Field(ge=1, le=200000)
    previous_sha256: Digest
    before: ScientificLifecycleStateV1
    after: ScientificLifecycleStateV1
    examples_seen: int = Field(ge=0, le=2**48)
    target_tokens_seen: int = Field(ge=0, le=2**48)
    history_sha256: Digest
    record: ScientificExecutionRecordV1


class IrisScientificExecutionEventV1(_Strict):
    sha256: Digest
    payload: ScientificExecutionEventPayloadV1


__all__ = [
    "IrisScientificExecutionEventV1",
    "IrisScientificExecutionHeaderV1",
    "ScientificExecutionCheckpointV1",
    "ScientificExecutionEventPayloadV1",
    "ScientificExecutionHeaderPayloadV1",
    "ScientificExecutionRecordV1",
    "ScientificExecutionTrainV1",
    "ScientificExecutionTrainingMetricV1",
    "ScientificExecutionValidationV1",
]
