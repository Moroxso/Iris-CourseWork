"""Strict schemas for one immutable Iris experiment run and its event chain."""

from __future__ import annotations

import math
import sys
from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.checkpoint_v1 import (
    TeacherAdamWConfigV1,
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
)
from iris.artifacts.experiment_v1 import ExperimentSplitCountsV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

_IDENTITY_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_RUN_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_MAX_EXACT_ADAMW_STEPS = 2**24
_MAX_EVENT_SEQUENCE = 1_000_000
_MAX_FLOAT64_LOG = math.log(sys.float_info.max)


class _ExperimentLogModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _validate_portable_identity(value: object, *, label: str) -> object:
    if type(value) is not str or fullmatch(_IDENTITY_PATTERN, value) is None:
        message = f"{label} must be a 1-128 character portable ASCII identity"
        raise ValueError(message)
    return value


def _validate_sha256(value: object, *, label: str) -> object:
    if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
        message = f"{label} must contain exactly 64 lower-case hexadecimal characters"
        raise ValueError(message)
    return value


def validate_experiment_run_id_v1(value: object) -> str:
    """Return one ASCII run ID safe as a path component on supported platforms."""
    if (
        type(value) is not str
        or fullmatch(_RUN_ID_PATTERN, value) is None
        or value.endswith(".")
        or value.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_STEMS
    ):
        message = "run_id must be a portable filesystem identity"
        raise ValueError(message)
    return value


def _validate_names(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        message = f"{label} must be a nonempty exact tuple"
        raise ValueError(message)
    if any(type(value) is not str or not value or "\0" in value for value in values):
        message = f"{label} must contain nonempty exact strings without NUL"
        raise ValueError(message)
    if len(set(values)) != len(values):
        message = f"{label} must not contain duplicates"
        raise ValueError(message)
    return values


class ExperimentRunSchemaV1(_ExperimentLogModel):
    artifact_type: Literal["iris.experiment_run"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    journal_layout: Literal["immutable_run_and_chained_events"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "experiment run version must be an exact integer"
            raise ValueError(message)
        return value


class ExperimentEventSchemaV1(_ExperimentLogModel):
    artifact_type: Literal["iris.experiment_event"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    chain_algorithm: Literal["sha256_predecessor"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "experiment event version must be an exact integer"
            raise ValueError(message)
        return value


class ExperimentManifestReferenceV1(_ExperimentLogModel):
    dataset_id: str
    payload_sha256: str
    split_counts: ExperimentSplitCountsV1

    @field_validator("dataset_id", mode="before")
    @classmethod
    def validate_dataset_id(cls, value: object) -> object:
        return _validate_portable_identity(value, label="dataset_id")

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="manifest payload_sha256")


class ExperimentModelConfigurationV1(_ExperimentLogModel):
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_names(value, label="experiment sidecar state_names")

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        TeacherCheckpointModelIdentityV1(
            backbone=self.backbone,
            cognitive=self.cognitive,
            workspace=self.workspace,
            runtime=self.runtime,
            state_names=self.state_names,
            sha256="0" * 64,
        )
        return self


class ExperimentOptimizerConfigurationV1(_ExperimentLogModel):
    optimizer_type: Literal["torch.optim.AdamW"]
    config: TeacherAdamWConfigV1
    parameter_names: tuple[str, ...]

    @field_validator("parameter_names")
    @classmethod
    def validate_parameter_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_names(value, label="experiment optimizer parameter_names")


class ExperimentTargetPreparationPolicyV1(_ExperimentLogModel):
    policy: Literal["raw_target_no_special_tokens_no_truncation_plus_one_eos"]
    terminal_token_id: int = Field(ge=0, strict=True)


class ExperimentRuntimePolicyV1(_ExperimentLogModel):
    initialization_seed: int = Field(ge=0, le=2**63 - 1, strict=True)
    deterministic: bool = Field(strict=True)


class ExperimentManualExecutionPolicyV1(_ExperimentLogModel):
    mode: Literal["manual_event_chain"]


class ExperimentEpochExecutionPolicyV1(_ExperimentLogModel):
    mode: Literal["deterministic_epochs"]
    epoch_count: int = Field(ge=1, le=1_000_000, strict=True)
    training_order: Literal["manifest_order_each_epoch"]
    checkpoint_policy: Literal["after_each_complete_epoch"]
    validation_policy: Literal["after_each_complete_epoch"]
    evaluate_test_after_training: bool = Field(strict=True)
    resume_policy: Literal["latest_checkpoint_plus_verified_replay"]


ExperimentExecutionPolicyV1 = Annotated[
    ExperimentManualExecutionPolicyV1 | ExperimentEpochExecutionPolicyV1,
    Field(discriminator="mode"),
]


class ExperimentTrainingConfigurationV1(_ExperimentLogModel):
    sha256: str
    model: ExperimentModelConfigurationV1
    optimizer: ExperimentOptimizerConfigurationV1
    target_preparation: ExperimentTargetPreparationPolicyV1
    runtime_policy: ExperimentRuntimePolicyV1
    execution_policy: ExperimentExecutionPolicyV1

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="training configuration sha256")

    @model_validator(mode="after")
    def validate_target_token(self) -> Self:
        if self.target_preparation.terminal_token_id >= self.model.runtime.vocabulary_size:
            message = "experiment terminal_token_id exceeds the configured vocabulary"
            raise ValueError(message)
        return self


class ExperimentCheckpointReferenceV1(_ExperimentLogModel):
    payload_sha256: str
    state_file_sha256: str
    sidecar_state_sha256: str
    completed_steps: int = Field(ge=0, le=_MAX_EXACT_ADAMW_STEPS, strict=True)

    @field_validator("payload_sha256", "state_file_sha256", "sidecar_state_sha256", mode="before")
    @classmethod
    def validate_hashes(cls, value: object) -> object:
        return _validate_sha256(value, label="checkpoint reference SHA-256")


class ExperimentRunPayloadV1(_ExperimentLogModel):
    run_id: str
    manifest: ExperimentManifestReferenceV1
    configuration: ExperimentTrainingConfigurationV1
    initial_checkpoint: ExperimentCheckpointReferenceV1

    @field_validator("run_id", mode="before")
    @classmethod
    def validate_run_id(cls, value: object) -> object:
        return validate_experiment_run_id_v1(value)

    @model_validator(mode="after")
    def validate_execution_bounds(self) -> Self:
        policy = self.configuration.execution_policy
        if isinstance(policy, ExperimentEpochExecutionPolicyV1):
            if not self.configuration.runtime_policy.deterministic:
                message = "epoch execution requires deterministic torch algorithms"
                raise ValueError(message)
            train_count = self.manifest.split_counts.train
            if self.manifest.split_counts.validation == 0:
                message = "epoch execution cannot evaluate an empty validation split"
                raise ValueError(message)
            final_step = self.initial_checkpoint.completed_steps + policy.epoch_count * train_count
            if final_step > _MAX_EXACT_ADAMW_STEPS:
                message = "epoch execution exceeds the exact AdamW step range"
                raise ValueError(message)
            event_count = policy.epoch_count * (train_count + 2)
            if policy.evaluate_test_after_training:
                if self.manifest.split_counts.test == 0:
                    message = "epoch execution cannot evaluate an empty test split"
                    raise ValueError(message)
                event_count += 1
            if event_count > _MAX_EVENT_SEQUENCE:
                message = "epoch execution exceeds the experiment event safety limit"
                raise ValueError(message)
        return self


class IrisExperimentRunV1(_ExperimentLogModel):
    schema_identity: ExperimentRunSchemaV1
    payload_sha256: str
    payload: ExperimentRunPayloadV1

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment run payload_sha256")


class ExperimentTrainingStepMetricV1(_ExperimentLogModel):
    event_type: Literal["training_step"]
    split: Literal["train"]
    example_id: str
    split_ordinal: int = Field(ge=1, strict=True)
    split_example_count: int = Field(ge=1, strict=True)
    step_index: int = Field(ge=1, le=_MAX_EXACT_ADAMW_STEPS, strict=True)
    loss: float = Field(ge=0.0, strict=True)
    supervised_token_count: int = Field(ge=1, strict=True)
    gradient_l2_norm: float = Field(ge=0.0, strict=True)
    gradients_clipped: bool = Field(strict=True)

    @field_validator("example_id", mode="before")
    @classmethod
    def validate_example_id(cls, value: object) -> object:
        return _validate_portable_identity(value, label="training example_id")

    @model_validator(mode="after")
    def validate_ordinal(self) -> Self:
        if self.split_ordinal > self.split_example_count:
            message = "training split_ordinal exceeds split_example_count"
            raise ValueError(message)
        return self


class ExperimentEvaluationMetricV1(_ExperimentLogModel):
    event_type: Literal["evaluation"]
    split: Literal["validation", "test"]
    example_count: int = Field(ge=1, strict=True)
    supervised_token_count: int = Field(ge=1, strict=True)
    mean_loss: float = Field(ge=0.0, strict=True)
    perplexity: float | None = Field(default=None, ge=1.0, strict=True)

    @model_validator(mode="after")
    def validate_perplexity(self) -> Self:
        if self.mean_loss > _MAX_FLOAT64_LOG:
            if self.perplexity is not None:
                message = "overflowed evaluation perplexity must be null"
                raise ValueError(message)
        elif self.perplexity != math.exp(self.mean_loss):
            message = "evaluation perplexity must exactly equal exp(mean_loss)"
            raise ValueError(message)
        return self


class ExperimentCheckpointEventV1(_ExperimentLogModel):
    event_type: Literal["checkpoint"]
    checkpoint: ExperimentCheckpointReferenceV1


ExperimentEventRecordV1 = Annotated[
    ExperimentTrainingStepMetricV1 | ExperimentEvaluationMetricV1 | ExperimentCheckpointEventV1,
    Field(discriminator="event_type"),
]


class ExperimentEventPayloadV1(_ExperimentLogModel):
    sequence: int = Field(ge=1, le=_MAX_EVENT_SEQUENCE, strict=True)
    run_id: str
    run_payload_sha256: str
    manifest_payload_sha256: str
    configuration_sha256: str
    predecessor_sha256: str
    optimizer_completed_steps: int = Field(
        ge=0,
        le=_MAX_EXACT_ADAMW_STEPS,
        strict=True,
    )
    sidecar_state_sha256: str
    record: ExperimentEventRecordV1

    @field_validator("run_id", mode="before")
    @classmethod
    def validate_run_id(cls, value: object) -> object:
        return _validate_portable_identity(value, label="event run_id")

    @field_validator(
        "run_payload_sha256",
        "manifest_payload_sha256",
        "configuration_sha256",
        "predecessor_sha256",
        "sidecar_state_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment event SHA-256")

    @model_validator(mode="after")
    def validate_record_state(self) -> Self:
        if (
            isinstance(self.record, ExperimentTrainingStepMetricV1)
            and self.record.step_index != self.optimizer_completed_steps
        ):
            message = "training metric step_index disagrees with optimizer_completed_steps"
            raise ValueError(message)
        if isinstance(self.record, ExperimentCheckpointEventV1):
            checkpoint = self.record.checkpoint
            if (
                checkpoint.completed_steps != self.optimizer_completed_steps
                or checkpoint.sidecar_state_sha256 != self.sidecar_state_sha256
            ):
                message = "checkpoint event disagrees with the recorded training state"
                raise ValueError(message)
        return self


class IrisExperimentEventV1(_ExperimentLogModel):
    schema_identity: ExperimentEventSchemaV1
    payload_sha256: str
    payload: ExperimentEventPayloadV1

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment event payload_sha256")


__all__ = [
    "ExperimentCheckpointEventV1",
    "ExperimentCheckpointReferenceV1",
    "ExperimentEpochExecutionPolicyV1",
    "ExperimentEvaluationMetricV1",
    "ExperimentEventPayloadV1",
    "ExperimentEventRecordV1",
    "ExperimentEventSchemaV1",
    "ExperimentExecutionPolicyV1",
    "ExperimentManifestReferenceV1",
    "ExperimentManualExecutionPolicyV1",
    "ExperimentModelConfigurationV1",
    "ExperimentOptimizerConfigurationV1",
    "ExperimentRunPayloadV1",
    "ExperimentRunSchemaV1",
    "ExperimentRuntimePolicyV1",
    "ExperimentTargetPreparationPolicyV1",
    "ExperimentTrainingConfigurationV1",
    "ExperimentTrainingStepMetricV1",
    "IrisExperimentEventV1",
    "IrisExperimentRunV1",
    "validate_experiment_run_id_v1",
]
