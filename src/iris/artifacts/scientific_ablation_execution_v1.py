"""Distinct cognitive-ablation phase journal DATA; never baseline/native authority."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_checkpoint_v1 import (
    IrisScientificAblationCheckpointManifestV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import ScientificAblationLifecycleStateV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationExecutionHeaderPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-execution-v1"] = "iris-scientific-ablation-execution-v1"
    lifecycle_configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    lifecycle_configuration_sha256: Digest
    initial: IrisScientificAblationCheckpointManifestV1
    maximum_journal_bytes: int = Field(default=128 * 1024**2, ge=1, le=1024**3)


class IrisScientificAblationExecutionHeaderV1(_Strict):
    sha256: Digest
    payload: ScientificAblationExecutionHeaderPayloadV1


class ScientificAblationExecutionTrainingMetricV1(_Strict):
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
    ablation_policy_sha256: Digest
    model_configuration_sha256: Digest
    scientific_state_before_sha256: Digest
    scientific_state_after_sha256: Digest
    registered_parameter_count: int = Field(ge=1, le=2**31)
    gradient_present_parameter_count: int = Field(ge=0, le=2**31)
    nonzero_gradient_parameter_count_before_clip: int = Field(ge=0, le=2**31)

    @model_validator(mode="after")
    def participation(self) -> Self:
        if not (
            self.nonzero_gradient_parameter_count_before_clip
            <= self.gradient_present_parameter_count
            <= self.registered_parameter_count
        ):
            raise ValueError("Gradient participation exceeds registered scalar parameters")
        return self


class ScientificAblationExecutionValidationMetricV1(_Strict):
    dataset_sha256: Digest
    example_count: int = Field(ge=1, le=5000)
    target_token_count: int = Field(ge=1, le=2**48)
    target_nll_sum: float = Field(ge=0.0)
    target_nll_per_token: float = Field(ge=0.0)
    at_optimizer_step: int = Field(ge=0, le=2**24)
    run_configuration_sha256: Digest
    model_configuration_sha256: Digest
    ablation_policy_sha256: Digest
    scientific_state_sha256: Digest

    @model_validator(mode="after")
    def complete_target_mean(self) -> Self:
        if self.target_nll_per_token != self.target_nll_sum / self.target_token_count:
            raise ValueError("Validation NLL is not the complete token-weighted mean")
        return self


class ScientificAblationExecutionTrainV1(_Strict):
    kind: Literal["train"] = "train"
    metric: ScientificAblationExecutionTrainingMetricV1


class ScientificAblationExecutionValidationV1(_Strict):
    kind: Literal["validation"] = "validation"
    metric: ScientificAblationExecutionValidationMetricV1
    selected_checkpoint_sha256: Digest | None


class ScientificAblationExecutionCheckpointV1(_Strict):
    kind: Literal["checkpoint"] = "checkpoint"
    lifecycle_checkpoint_sha256: Digest


ScientificAblationExecutionRecordV1 = Annotated[
    ScientificAblationExecutionTrainV1
    | ScientificAblationExecutionValidationV1
    | ScientificAblationExecutionCheckpointV1,
    Field(discriminator="kind"),
]


class ScientificAblationExecutionEventPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-execution-event-v1"] = (
        "iris-scientific-ablation-execution-event-v1"
    )
    execution_sha256: Digest
    sequence: int = Field(ge=1, le=200000)
    previous_sha256: Digest
    before: ScientificAblationLifecycleStateV1
    after: ScientificAblationLifecycleStateV1
    examples_seen: int = Field(ge=0, le=2**48)
    target_tokens_seen: int = Field(ge=0, le=2**48)
    history_sha256: Digest
    scientific_state_sha256: Digest
    record: ScientificAblationExecutionRecordV1


class IrisScientificAblationExecutionEventV1(_Strict):
    sha256: Digest
    payload: ScientificAblationExecutionEventPayloadV1


__all__ = [
    "IrisScientificAblationExecutionEventV1",
    "IrisScientificAblationExecutionHeaderV1",
    "ScientificAblationExecutionCheckpointV1",
    "ScientificAblationExecutionEventPayloadV1",
    "ScientificAblationExecutionHeaderPayloadV1",
    "ScientificAblationExecutionRecordV1",
    "ScientificAblationExecutionTrainV1",
    "ScientificAblationExecutionTrainingMetricV1",
    "ScientificAblationExecutionValidationMetricV1",
    "ScientificAblationExecutionValidationV1",
]
