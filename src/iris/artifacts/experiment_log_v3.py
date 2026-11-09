"""Strict profile-v4 schemas for B/C/D experiment runs and event journals."""

from __future__ import annotations

from re import fullmatch
from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from iris.artifacts.checkpoint_v1 import (
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
)
from iris.artifacts.checkpoint_v2 import NoWorkspaceTeacherCheckpointModelIdentityV2
from iris.artifacts.checkpoint_v3 import SharedBufferTeacherCheckpointModelIdentityV3
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

_PROTOCOL_DECLARATION_PAYLOAD_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_EXECUTION_PROFILE_SET_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_PREDECESSOR_PROFILE_SET_SHA256: Literal[
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
_ARM_C_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
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

type ExperimentArmIdV3 = Literal["B", "C", "D"]
type ExperimentArmKindV3 = Literal[
    "llm_modules_no_workspace",
    "llm_modules_unlimited_shared_buffer",
    "llm_modules_limited_competitive_workspace",
]
type ExperimentRequestGraphV3 = Literal[
    "independent_recurrent_modules_zero_memory_read_v1",
    "append_only_all_proposals_prior_cycle_shared_buffer_v1",
    "limited_competitive_workspace_v1",
]
type ExperimentTeacherGraphV3 = Literal[
    "independent_recurrent_modules_zero_memory_read_teacher_v1",
    "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
    "limited_competitive_workspace_teacher_v1",
]


class _ExperimentLogModelV3(BaseModel):
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


def _normalize_json_tuple(value: object, info: ValidationInfo) -> object:
    """Restore JSON arrays as tuples without weakening strict Python validation."""
    if info.mode == "json" and type(value) is list:
        return tuple(value)
    return value


def _validate_exact_wrapped(
    value: object,
    *,
    handler: ValidatorFunctionWrapHandler,
    exact_types: tuple[type[BaseModel], ...],
    label: str,
) -> object:
    if isinstance(value, BaseModel) and type(value) not in exact_types:
        message = f"{label} must use an exact version-3 schema type"
        raise ValueError(message)
    result = handler(value)
    if type(result) not in exact_types:
        message = f"{label} must use an exact version-3 schema type"
        raise ValueError(message)
    return result


class ExperimentRunSchemaV3(_ExperimentLogModelV3):
    """Schema identity for one immutable profile-v4 experiment run."""

    artifact_type: Literal["iris.experiment_run"]
    version: Literal[3]
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


class ExperimentEventSchemaV3(_ExperimentLogModelV3):
    """Schema identity for one profile-v4 chained experiment event."""

    artifact_type: Literal["iris.experiment_event"]
    version: Literal[3]
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


class ExperimentArmBModelConfigurationV3(_ExperimentLogModelV3):
    """Exact cognitive-only, zero-memory-read teacher topology for arm B."""

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

    @field_validator("component_names", "state_names", mode="before")
    @classmethod
    def normalize_json_names(cls, value: object, info: ValidationInfo) -> object:
        return _normalize_json_tuple(value, info)

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
        # B/v2 remains the canonical checkpoint format. Its topology is unchanged in
        # profile v4, so the predecessor profile identity is supplied only to reuse
        # the checkpoint's complete topology and context invariant validation.
        NoWorkspaceTeacherCheckpointModelIdentityV2(
            arm_id=self.arm_id,
            arm_kind=self.arm_kind,
            teacher_graph=self.teacher_graph,
            execution_profile_set_sha256=_PREDECESSOR_PROFILE_SET_SHA256,
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


class ExperimentArmCModelConfigurationV3(_ExperimentLogModelV3):
    """Exact cognitive-only append-only shared-buffer teacher topology for arm C."""

    arm_id: Literal["C"]
    arm_kind: Literal["llm_modules_unlimited_shared_buffer"]
    teacher_graph: Literal["append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    cognitive_cycles: Literal[3]
    pre_update_read_count: Literal[3]
    final_output_read_count: Literal[1]
    entries_per_cycle: int = Field(gt=0, strict=True)
    final_entry_count: int = Field(gt=0, strict=True)
    latent_token_count: int = Field(gt=0, strict=True)
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator(
        "cognitive_cycles",
        "pre_update_read_count",
        "final_output_read_count",
        mode="before",
    )
    @classmethod
    def validate_exact_counts(cls, value: object) -> object:
        if type(value) is not int:
            message = "arm-C lifecycle counts must be exact integers"
            raise ValueError(message)
        return value

    @field_validator("component_names", "state_names", mode="before")
    @classmethod
    def normalize_json_names(cls, value: object, info: ValidationInfo) -> object:
        return _normalize_json_tuple(value, info)

    @field_validator("component_names")
    @classmethod
    def validate_component_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_names(value, label="arm-C experiment component_names")
        if value != _ARM_C_COMPONENT_NAMES:
            message = "arm-C experiment component ownership is not canonical"
            raise ValueError(message)
        return value

    @field_validator("state_names")
    @classmethod
    def validate_state_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_names(value, label="arm-C experiment state_names")

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        SharedBufferTeacherCheckpointModelIdentityV3(
            arm_id=self.arm_id,
            arm_kind=self.arm_kind,
            teacher_graph=self.teacher_graph,
            execution_profile_set_sha256=_EXECUTION_PROFILE_SET_SHA256,
            shared_buffer_policy_sha256=self.shared_buffer_policy_sha256,
            protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            backbone=self.backbone,
            cognitive=self.cognitive,
            cognitive_cycles=self.cognitive_cycles,
            pre_update_read_count=self.pre_update_read_count,
            final_output_read_count=self.final_output_read_count,
            entries_per_cycle=self.entries_per_cycle,
            final_entry_count=self.final_entry_count,
            latent_token_count=self.latent_token_count,
            component_names=self.component_names,
            runtime=self.runtime,
            state_names=self.state_names,
            sha256="0" * 64,
        )
        return self


class ExperimentArmDModelConfigurationV3(_ExperimentLogModelV3):
    """Exact limited competitive workspace teacher topology for arm D."""

    arm_id: Literal["D"]
    arm_kind: Literal["llm_modules_limited_competitive_workspace"]
    teacher_graph: Literal["limited_competitive_workspace_teacher_v1"]
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    component_names: tuple[str, ...]
    runtime: TeacherBackboneRuntimeV1
    state_names: tuple[str, ...]

    @field_validator("component_names", "state_names", mode="before")
    @classmethod
    def normalize_json_names(cls, value: object, info: ValidationInfo) -> object:
        return _normalize_json_tuple(value, info)

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


ExperimentModelConfigurationV3 = Annotated[
    ExperimentArmBModelConfigurationV3
    | ExperimentArmCModelConfigurationV3
    | ExperimentArmDModelConfigurationV3,
    Field(discriminator="arm_id"),
]
_MODEL_CONFIGURATION_TYPES: tuple[type[BaseModel], ...] = (
    ExperimentArmBModelConfigurationV3,
    ExperimentArmCModelConfigurationV3,
    ExperimentArmDModelConfigurationV3,
)


class ExperimentTrainingConfigurationV3(_ExperimentLogModelV3):
    """Hash-labelled model, optimizer, target, runtime, and execution policy."""

    sha256: str
    model: ExperimentModelConfigurationV3
    optimizer: ExperimentOptimizerConfigurationV1
    target_preparation: ExperimentTargetPreparationPolicyV1
    runtime_policy: ExperimentRuntimePolicyV1
    execution_policy: ExperimentExecutionPolicyV1

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="training configuration sha256")

    @field_validator("model", mode="wrap")
    @classmethod
    def validate_exact_model_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentModelConfigurationV3:
        return cast(
            "ExperimentModelConfigurationV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=_MODEL_CONFIGURATION_TYPES,
                label="experiment model configuration",
            ),
        )

    @field_validator("optimizer", mode="before")
    @classmethod
    def normalize_json_optimizer(cls, value: object, info: ValidationInfo) -> object:
        if info.mode != "json" or type(value) is not dict:
            return value
        parameter_names = value.get("parameter_names")
        if type(parameter_names) is not list:
            return value
        normalized = dict(value)
        normalized["parameter_names"] = tuple(parameter_names)
        return normalized

    @model_validator(mode="after")
    def validate_target_token(self) -> Self:
        if type(self.model) not in _MODEL_CONFIGURATION_TYPES:
            message = "experiment model configuration must use an exact version-3 schema type"
            raise ValueError(message)
        if self.target_preparation.terminal_token_id >= self.model.runtime.vocabulary_size:
            message = "experiment terminal_token_id exceeds the configured vocabulary"
            raise ValueError(message)
        return self


class _ExperimentCheckpointReferenceV3(_ExperimentLogModelV3):
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


class ExperimentArmBCheckpointReferenceV3(_ExperimentCheckpointReferenceV3):
    """Reference to exactly one no-workspace arm-B checkpoint v2."""

    checkpoint_schema_version: Literal[2]
    arm_id: Literal["B"]


class ExperimentArmCCheckpointReferenceV3(_ExperimentCheckpointReferenceV3):
    """Reference to exactly one shared-buffer arm-C checkpoint v3."""

    checkpoint_schema_version: Literal[3]
    arm_id: Literal["C"]


class ExperimentArmDCheckpointReferenceV3(_ExperimentCheckpointReferenceV3):
    """Reference to exactly one limited-workspace arm-D checkpoint v1."""

    checkpoint_schema_version: Literal[1]
    arm_id: Literal["D"]


ExperimentCheckpointReferenceV3 = Annotated[
    ExperimentArmBCheckpointReferenceV3
    | ExperimentArmCCheckpointReferenceV3
    | ExperimentArmDCheckpointReferenceV3,
    Field(discriminator="checkpoint_schema_version"),
]
_CHECKPOINT_REFERENCE_TYPES: tuple[type[BaseModel], ...] = (
    ExperimentArmBCheckpointReferenceV3,
    ExperimentArmCCheckpointReferenceV3,
    ExperimentArmDCheckpointReferenceV3,
)


class ExperimentRunPayloadV3(_ExperimentLogModelV3):
    """Profile-v4 run identity and its exact arm-labelled initial checkpoint."""

    run_id: str
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    execution_profile_set_version: Literal[4]
    execution_profile_set_sha256: Literal[
        "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
    ]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    arm_id: ExperimentArmIdV3
    arm_kind: ExperimentArmKindV3
    request_graph: ExperimentRequestGraphV3
    teacher_graph: ExperimentTeacherGraphV3
    manifest: ExperimentManifestReferenceV1
    configuration: ExperimentTrainingConfigurationV3
    initial_checkpoint: ExperimentCheckpointReferenceV3

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

    @field_validator("configuration", mode="wrap")
    @classmethod
    def validate_exact_configuration_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentTrainingConfigurationV3:
        return cast(
            "ExperimentTrainingConfigurationV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=(ExperimentTrainingConfigurationV3,),
                label="experiment training configuration",
            ),
        )

    @field_validator("initial_checkpoint", mode="wrap")
    @classmethod
    def validate_exact_checkpoint_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentCheckpointReferenceV3:
        return cast(
            "ExperimentCheckpointReferenceV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=_CHECKPOINT_REFERENCE_TYPES,
                label="initial checkpoint reference",
            ),
        )

    @model_validator(mode="after")
    def validate_arm_and_execution_bounds(self) -> Self:
        expected_identity: dict[
            ExperimentArmIdV3,
            tuple[ExperimentArmKindV3, ExperimentRequestGraphV3, ExperimentTeacherGraphV3],
        ] = {
            "B": (
                "llm_modules_no_workspace",
                "independent_recurrent_modules_zero_memory_read_v1",
                "independent_recurrent_modules_zero_memory_read_teacher_v1",
            ),
            "C": (
                "llm_modules_unlimited_shared_buffer",
                "append_only_all_proposals_prior_cycle_shared_buffer_v1",
                "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
            ),
            "D": (
                "llm_modules_limited_competitive_workspace",
                "limited_competitive_workspace_v1",
                "limited_competitive_workspace_teacher_v1",
            ),
        }
        expected_model_types: dict[ExperimentArmIdV3, type[BaseModel]] = {
            "B": ExperimentArmBModelConfigurationV3,
            "C": ExperimentArmCModelConfigurationV3,
            "D": ExperimentArmDModelConfigurationV3,
        }
        expected_checkpoint_types: dict[ExperimentArmIdV3, type[BaseModel]] = {
            "B": ExperimentArmBCheckpointReferenceV3,
            "C": ExperimentArmCCheckpointReferenceV3,
            "D": ExperimentArmDCheckpointReferenceV3,
        }
        observed_identity = (self.arm_kind, self.request_graph, self.teacher_graph)
        if observed_identity != expected_identity[self.arm_id]:
            message = "experiment arm identity disagrees with execution profile set v4"
            raise ValueError(message)

        model = self.configuration.model
        if (
            type(self.configuration) is not ExperimentTrainingConfigurationV3
            or type(model) is not expected_model_types[self.arm_id]
            or model.arm_id != self.arm_id
            or model.arm_kind != self.arm_kind
            or model.teacher_graph != self.teacher_graph
        ):
            message = "experiment model configuration disagrees with run arm identity"
            raise ValueError(message)
        if (
            type(self.initial_checkpoint) is not expected_checkpoint_types[self.arm_id]
            or self.initial_checkpoint.arm_id != self.arm_id
        ):
            message = "initial checkpoint disagrees with run arm identity"
            raise ValueError(message)
        if (
            type(model) is ExperimentArmCModelConfigurationV3
            and model.shared_buffer_policy_sha256 != self.shared_buffer_policy_sha256
        ):
            message = "arm-C model configuration disagrees with shared-buffer policy"
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


class IrisExperimentRunV3(_ExperimentLogModelV3):
    """Canonical envelope for one profile-v4 experiment run."""

    schema_identity: ExperimentRunSchemaV3
    payload_sha256: str
    payload: ExperimentRunPayloadV3

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment run payload_sha256")

    @field_validator("schema_identity", mode="wrap")
    @classmethod
    def validate_exact_schema_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentRunSchemaV3:
        return cast(
            "ExperimentRunSchemaV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=(ExperimentRunSchemaV3,),
                label="experiment run schema identity",
            ),
        )

    @field_validator("payload", mode="wrap")
    @classmethod
    def validate_exact_payload_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentRunPayloadV3:
        return cast(
            "ExperimentRunPayloadV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=(ExperimentRunPayloadV3,),
                label="experiment run payload",
            ),
        )

    @model_validator(mode="after")
    def validate_exact_types(self) -> Self:
        if (
            type(self.schema_identity) is not ExperimentRunSchemaV3
            or type(self.payload) is not ExperimentRunPayloadV3
        ):
            message = "experiment run must use exact version-3 schema and payload types"
            raise ValueError(message)
        return self


class ExperimentCheckpointEventV3(_ExperimentLogModelV3):
    """Checkpoint event carrying an exact arm-discriminated reference."""

    event_type: Literal["checkpoint"]
    checkpoint: ExperimentCheckpointReferenceV3

    @field_validator("checkpoint", mode="wrap")
    @classmethod
    def validate_exact_checkpoint_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentCheckpointReferenceV3:
        return cast(
            "ExperimentCheckpointReferenceV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=_CHECKPOINT_REFERENCE_TYPES,
                label="event checkpoint reference",
            ),
        )

    @model_validator(mode="after")
    def validate_exact_checkpoint(self) -> Self:
        if type(self.checkpoint) not in _CHECKPOINT_REFERENCE_TYPES:
            message = "event checkpoint reference must use an exact version-3 schema type"
            raise ValueError(message)
        return self


ExperimentEventRecordV3 = Annotated[
    ExperimentTrainingStepMetricV1 | ExperimentEvaluationMetricV1 | ExperimentCheckpointEventV3,
    Field(discriminator="event_type"),
]
_EVENT_RECORD_TYPES: tuple[type[BaseModel], ...] = (
    ExperimentTrainingStepMetricV1,
    ExperimentEvaluationMetricV1,
    ExperimentCheckpointEventV3,
)


class ExperimentEventPayloadV3(_ExperimentLogModelV3):
    """One arm-labelled event linked to a version-3 run and predecessor."""

    sequence: int = Field(ge=1, le=_MAX_EVENT_SEQUENCE, strict=True)
    run_id: str
    arm_id: ExperimentArmIdV3
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
    record: ExperimentEventRecordV3

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

    @field_validator("record", mode="wrap")
    @classmethod
    def validate_exact_record_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentEventRecordV3:
        return cast(
            "ExperimentEventRecordV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=_EVENT_RECORD_TYPES,
                label="experiment event record",
            ),
        )

    @model_validator(mode="after")
    def validate_record_state(self) -> Self:
        if type(self.record) not in _EVENT_RECORD_TYPES:
            message = "experiment event record must use an exact persisted schema type"
            raise ValueError(message)
        if (
            type(self.record) is ExperimentTrainingStepMetricV1
            and self.record.step_index != self.optimizer_completed_steps
        ):
            message = "training metric step_index disagrees with optimizer_completed_steps"
            raise ValueError(message)
        if type(self.record) is ExperimentCheckpointEventV3:
            checkpoint = self.record.checkpoint
            expected_checkpoint_types: dict[ExperimentArmIdV3, type[BaseModel]] = {
                "B": ExperimentArmBCheckpointReferenceV3,
                "C": ExperimentArmCCheckpointReferenceV3,
                "D": ExperimentArmDCheckpointReferenceV3,
            }
            if (
                type(checkpoint) is not expected_checkpoint_types[self.arm_id]
                or checkpoint.arm_id != self.arm_id
            ):
                message = "checkpoint event disagrees with event arm identity"
                raise ValueError(message)
            if (
                checkpoint.completed_steps != self.optimizer_completed_steps
                or checkpoint.sidecar_state_sha256 != self.sidecar_state_sha256
            ):
                message = "checkpoint event disagrees with the recorded training state"
                raise ValueError(message)
        return self


class IrisExperimentEventV3(_ExperimentLogModelV3):
    """Canonical envelope for one profile-v4 experiment event."""

    schema_identity: ExperimentEventSchemaV3
    payload_sha256: str
    payload: ExperimentEventPayloadV3

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        return _validate_sha256(value, label="experiment event payload_sha256")

    @field_validator("schema_identity", mode="wrap")
    @classmethod
    def validate_exact_schema_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentEventSchemaV3:
        return cast(
            "ExperimentEventSchemaV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=(ExperimentEventSchemaV3,),
                label="experiment event schema identity",
            ),
        )

    @field_validator("payload", mode="wrap")
    @classmethod
    def validate_exact_payload_type(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
    ) -> ExperimentEventPayloadV3:
        return cast(
            "ExperimentEventPayloadV3",
            _validate_exact_wrapped(
                value,
                handler=handler,
                exact_types=(ExperimentEventPayloadV3,),
                label="experiment event payload",
            ),
        )

    @model_validator(mode="after")
    def validate_exact_types(self) -> Self:
        if (
            type(self.schema_identity) is not ExperimentEventSchemaV3
            or type(self.payload) is not ExperimentEventPayloadV3
        ):
            message = "experiment event must use exact version-3 schema and payload types"
            raise ValueError(message)
        return self


__all__ = [
    "ExperimentArmBCheckpointReferenceV3",
    "ExperimentArmBModelConfigurationV3",
    "ExperimentArmCCheckpointReferenceV3",
    "ExperimentArmCModelConfigurationV3",
    "ExperimentArmDCheckpointReferenceV3",
    "ExperimentArmDModelConfigurationV3",
    "ExperimentArmIdV3",
    "ExperimentArmKindV3",
    "ExperimentCheckpointEventV3",
    "ExperimentCheckpointReferenceV3",
    "ExperimentEventPayloadV3",
    "ExperimentEventRecordV3",
    "ExperimentEventSchemaV3",
    "ExperimentModelConfigurationV3",
    "ExperimentRequestGraphV3",
    "ExperimentRunPayloadV3",
    "ExperimentRunSchemaV3",
    "ExperimentTeacherGraphV3",
    "ExperimentTrainingConfigurationV3",
    "IrisExperimentEventV3",
    "IrisExperimentRunV3",
]
