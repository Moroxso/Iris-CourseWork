"""Strict arm-labelled schemas for experiment run and event journal version 2."""

from __future__ import annotations

from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.checkpoint_v1 import (
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
)
from iris.artifacts.checkpoint_v2 import NoWorkspaceTeacherCheckpointModelIdentityV2
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentExecutionPolicyV1,
    ExperimentManifestReferenceV1,
    ExperimentOptimizerConfigurationV1,
    ExperimentRuntimePolicyV1,
    ExperimentTargetPreparationPolicyV1,
    ExperimentTrainingStepMetricV1,
    validate_experiment_run_id_v1,
)
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

_PROTOCOL_DECLARATION_PAYLOAD_SHA256 = (
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
)
_EXECUTION_PROFILE_SET_SHA256: Literal[
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
] = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
_MAX_EXACT_ADAMW_STEPS = 2**24
_MAX_EVENT_SEQUENCE = 1_000_000
_ARM_B_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)
_ARM_D_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "workspace_reader",
    "cognitive_modules",
    "proposal_head",
    "workspace_arbiter",
    "slot_allocator",
    "workspace_writer",
    "workspace_decay",
    "latent_bridge",
    "sequence_assembler",
)

type ExperimentArmIdV2 = Literal["B", "D"]
type ExperimentArmKindV2 = Literal[
    "llm_modules_no_workspace",
    "llm_modules_limited_competitive_workspace",
]
type ExperimentRequestGraphV2 = Literal[
    "independent_recurrent_modules_zero_memory_read_v1",
    "limited_competitive_workspace_v1",
]
type ExperimentTeacherGraphV2 = Literal[
    "independent_recurrent_modules_zero_memory_read_teacher_v1",
    "limited_competitive_workspace_teacher_v1",
]


class _ExperimentLogModelV2(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _validate_sha256(value: object, *, label: str) -> object:
    if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
        message = f"{label} must contain exactly 64 lower-case hexadecimal characters"
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


class ExperimentRunSchemaV2(_ExperimentLogModelV2):
    """Schema identity for one immutable arm-labelled experiment run."""

    artifact_type: Literal["iris.experiment_run"]
    version: Literal[2]
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


class ExperimentEventSchemaV2(_ExperimentLogModelV2):
    """Schema identity for one arm-labelled chained event."""

    artifact_type: Literal["iris.experiment_event"]
    version: Literal[2]
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


class ExperimentArmBModelConfigurationV2(_ExperimentLogModelV2):
    """Exact no-workspace model topology for arm B."""

    arm_id: Literal["B"]
    arm_kind: Literal["llm_modules_no_workspace"]
    teacher_graph: Literal["independent_recurrent_modules_zero_memory_read_teacher_v1"]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    cognitive_cycles: Literal[3]
    latent_token_count: int = Field(gt=0, strict=True)
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("cognitive_cycles", mode="before")
    @classmethod
    def validate_exact_cycles(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-B cognitive_cycles must be an exact integer"
            raise ValueError(message)
        return value

    @field_validator("component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-B experiment component_names")
        if value != _ARM_B_COMPONENT_NAMES:
            message = "arm-B experiment component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_names(value, label="arm-B experiment state_names")

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        NoWorkspaceTeacherCheckpointModelIdentityV2(
            arm_id=self.arm_id,
            arm_kind=self.arm_kind,
            teacher_graph=self.teacher_graph,
            execution_profile_set_sha256=_EXECUTION_PROFILE_SET_SHA256,
            backbone=self.backbone,
            cognitive=self.cognitive,
            cognitive_cycles=self.cognitive_cycles,
            latent_token_count=self.latent_token_count,
            component_names=self.component_names,
            runtime=self.runtime,
            state_names=self.state_names,
            sha256="0" * 64,
        )
        return self


class ExperimentArmDModelConfigurationV2(_ExperimentLogModelV2):
    """Exact limited competitive workspace model topology for arm D."""

    arm_id: Literal["D"]
    arm_kind: Literal["llm_modules_limited_competitive_workspace"]
    teacher_graph: Literal["limited_competitive_workspace_teacher_v1"]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-D experiment component_names")
        if value != _ARM_D_COMPONENT_NAMES:
            message = "arm-D experiment component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_names(value, label="arm-D experiment state_names")

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


ExperimentModelConfigurationV2 = Annotated[
    ExperimentArmBModelConfigurationV2 | ExperimentArmDModelConfigurationV2,
    Field(discriminator="arm_id"),
]


class ExperimentTrainingConfigurationV2(_ExperimentLogModelV2):
    """Hash-labelled model, optimizer, target, runtime, and execution policy."""

    sha256: str
    model: ExperimentModelConfigurationV2
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


class _ExperimentCheckpointReferenceV2(_ExperimentLogModelV2):
    payload_sha256: str
    state_file_sha256: str
    sidecar_state_sha256: str
    completed_steps: int = Field(ge=0, le=_MAX_EXACT_ADAMW_STEPS, strict=True)

    @field_validator(
        "payload_sha256",
        "state_file_sha256",
        "sidecar_state_sha256",
        mode="before",
    )
    @classmethod
    def validate_hashes(cls, value: object) -> object:
        return _validate_sha256(value, label="checkpoint reference SHA-256")


class ExperimentArmBCheckpointReferenceV2(_ExperimentCheckpointReferenceV2):
    """Reference to exactly one no-workspace arm-B checkpoint v2."""

    checkpoint_schema_version: Literal[2]
    arm_id: Literal["B"]


class ExperimentArmDCheckpointReferenceV2(_ExperimentCheckpointReferenceV2):
    """Reference to exactly one limited-workspace arm-D checkpoint v1."""

    checkpoint_schema_version: Literal[1]
    arm_id: Literal["D"]


ExperimentCheckpointReferenceV2 = Annotated[
    ExperimentArmBCheckpointReferenceV2 | ExperimentArmDCheckpointReferenceV2,
    Field(discriminator="checkpoint_schema_version"),
]


class ExperimentRunPayloadV2(_ExperimentLogModelV2):
    """Arm-labelled immutable run identity and initial checkpoint."""

    run_id: str
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    execution_profile_set_version: Literal[3]
    execution_profile_set_sha256: Literal[
        "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
    ]
    arm_id: ExperimentArmIdV2
    arm_kind: ExperimentArmKindV2
    request_graph: ExperimentRequestGraphV2
    teacher_graph: ExperimentTeacherGraphV2
    manifest: ExperimentManifestReferenceV1
    configuration: ExperimentTrainingConfigurationV2
    initial_checkpoint: ExperimentCheckpointReferenceV2

    @field_validator("run_id", mode="before")
    @classmethod
    def validate_run_id(cls, value: object) -> object:
        return validate_experiment_run_id_v1(value)

    @field_validator("execution_profile_set_version", mode="before")
    @classmethod
    def validate_exact_profile_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "execution profile set version must be an exact integer"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_arm_and_execution_bounds(self) -> Self:
        expected_identity: dict[
            ExperimentArmIdV2,
            tuple[ExperimentArmKindV2, ExperimentRequestGraphV2, ExperimentTeacherGraphV2],
        ] = {
            "B": (
                "llm_modules_no_workspace",
                "independent_recurrent_modules_zero_memory_read_v1",
                "independent_recurrent_modules_zero_memory_read_teacher_v1",
            ),
            "D": (
                "llm_modules_limited_competitive_workspace",
                "limited_competitive_workspace_v1",
                "limited_competitive_workspace_teacher_v1",
            ),
        }
        observed_identity = (self.arm_kind, self.request_graph, self.teacher_graph)
        if observed_identity != expected_identity[self.arm_id]:
            message = "experiment arm identity disagrees with execution profile set v3"
            raise ValueError(message)
        if (
            self.configuration.model.arm_id != self.arm_id
            or self.configuration.model.arm_kind != self.arm_kind
            or self.configuration.model.teacher_graph != self.teacher_graph
        ):
            message = "experiment model configuration disagrees with run arm identity"
            raise ValueError(message)
        if self.initial_checkpoint.arm_id != self.arm_id:
            message = "initial checkpoint disagrees with run arm identity"
            raise ValueError(message)

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


class IrisExperimentRunV2(_ExperimentLogModelV2):
    """Canonical envelope for one experiment run v2."""

    schema_identity: ExperimentRunSchemaV2
    payload_sha256: str
    payload: ExperimentRunPayloadV2

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment run payload_sha256")


class ExperimentCheckpointEventV2(_ExperimentLogModelV2):
    """Checkpoint event carrying an arm-discriminated reference."""

    event_type: Literal["checkpoint"]
    checkpoint: ExperimentCheckpointReferenceV2


ExperimentEventRecordV2 = Annotated[
    ExperimentTrainingStepMetricV1 | ExperimentEvaluationMetricV1 | ExperimentCheckpointEventV2,
    Field(discriminator="event_type"),
]


class ExperimentEventPayloadV2(_ExperimentLogModelV2):
    """One arm-labelled event linked to a version-2 run and predecessor."""

    sequence: int = Field(ge=1, le=_MAX_EVENT_SEQUENCE, strict=True)
    run_id: str
    arm_id: ExperimentArmIdV2
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
    record: ExperimentEventRecordV2

    @field_validator("run_id", mode="before")
    @classmethod
    def validate_run_id(cls, value: object) -> object:
        return validate_experiment_run_id_v1(value)

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
        if isinstance(self.record, ExperimentCheckpointEventV2):
            checkpoint = self.record.checkpoint
            if checkpoint.arm_id != self.arm_id:
                message = "checkpoint event disagrees with event arm identity"
                raise ValueError(message)
            if (
                checkpoint.completed_steps != self.optimizer_completed_steps
                or checkpoint.sidecar_state_sha256 != self.sidecar_state_sha256
            ):
                message = "checkpoint event disagrees with the recorded training state"
                raise ValueError(message)
        return self


class IrisExperimentEventV2(_ExperimentLogModelV2):
    """Canonical envelope for one experiment event v2."""

    schema_identity: ExperimentEventSchemaV2
    payload_sha256: str
    payload: ExperimentEventPayloadV2

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment event payload_sha256")


__all__ = [
    "ExperimentArmBCheckpointReferenceV2",
    "ExperimentArmBModelConfigurationV2",
    "ExperimentArmDCheckpointReferenceV2",
    "ExperimentArmDModelConfigurationV2",
    "ExperimentArmIdV2",
    "ExperimentArmKindV2",
    "ExperimentCheckpointEventV2",
    "ExperimentCheckpointReferenceV2",
    "ExperimentEventPayloadV2",
    "ExperimentEventRecordV2",
    "ExperimentEventSchemaV2",
    "ExperimentModelConfigurationV2",
    "ExperimentRequestGraphV2",
    "ExperimentRunPayloadV2",
    "ExperimentRunSchemaV2",
    "ExperimentTeacherGraphV2",
    "ExperimentTrainingConfigurationV2",
    "IrisExperimentEventV2",
    "IrisExperimentRunV2",
]
