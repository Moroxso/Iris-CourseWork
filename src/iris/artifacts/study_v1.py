"""Strict schema for the recovered Iris-GW-0.1 four-arm protocol declaration."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

StudyArmIdV1 = Literal["A", "B", "C", "D"]
StudyArmKindV1 = Literal[
    "base_llm",
    "llm_modules_no_workspace",
    "llm_modules_unlimited_shared_buffer",
    "llm_modules_limited_competitive_workspace",
]
StudyTaskFamilyV1 = Literal[
    "split_information",
    "delayed_planning",
    "code_generation_and_verification",
    "hidden_goal_retention",
    "module_conflict",
]
StudyUnresolvedItemV1 = Literal[
    "arm_a_execution_semantics",
    "arm_b_execution_semantics",
    "arm_c_execution_semantics",
    "unlimited_shared_buffer_semantics",
    "main_backbone_pinned_revision",
    "dataset_construction_and_split_policy",
    "task_specific_evaluation_protocol",
    "seeds_repeats_epochs_and_optimizer",
    "comparison_aggregation_uncertainty_and_threshold",
    "study_storage_budget",
]

_EXPECTED_ARMS: tuple[tuple[StudyArmIdV1, StudyArmKindV1], ...] = (
    ("A", "base_llm"),
    ("B", "llm_modules_no_workspace"),
    ("C", "llm_modules_unlimited_shared_buffer"),
    ("D", "llm_modules_limited_competitive_workspace"),
)
_EXPECTED_TASK_FAMILIES: tuple[StudyTaskFamilyV1, ...] = (
    "split_information",
    "delayed_planning",
    "code_generation_and_verification",
    "hidden_goal_retention",
    "module_conflict",
)
_EXPECTED_UNRESOLVED_ITEMS: tuple[StudyUnresolvedItemV1, ...] = (
    "arm_a_execution_semantics",
    "arm_b_execution_semantics",
    "arm_c_execution_semantics",
    "unlimited_shared_buffer_semantics",
    "main_backbone_pinned_revision",
    "dataset_construction_and_split_policy",
    "task_specific_evaluation_protocol",
    "seeds_repeats_epochs_and_optimizer",
    "comparison_aggregation_uncertainty_and_threshold",
    "study_storage_budget",
)


class _StudyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class FourArmProtocolSchemaV1(_StudyModel):
    artifact_type: Literal["iris.four_arm_protocol_declaration"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    declaration_scope: Literal["recovered_facts_and_explicit_gaps"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "four-arm protocol version must be an exact integer"
            raise ValueError(message)
        return value


class FourArmProtocolSourceV1(_StudyModel):
    conversation_id: Literal["6a748d3d-5864-83ed-9d15-3cd16ea8580e"]
    arm_definition_turn_id: Literal["0b2e3f67-511d-40f1-a652-167abd8c1d58"]
    arm_definition_message_id: Literal["f6e454fa-171c-4240-83d9-79be11765cf5"]
    final_specification_turn_id: Literal["fb2807c0-b18c-47fd-b950-88784c477c8f"]
    final_specification_message_id: Literal["cfb7cc81-ece7-4241-8ba1-889cc51997b9"]
    final_specification_recovery: Literal["prefix_only_20000_characters"]


class FourArmDefinitionV1(_StudyModel):
    arm_id: StudyArmIdV1
    arm_kind: StudyArmKindV1


class FourArmArchitectureConstantsV1(_StudyModel):
    main_backbone_repository: Literal["Qwen/Qwen3-1.7B"]
    debug_backbone_repository: Literal["Qwen/Qwen3-0.6B"]
    cognitive_module_count: Literal[8]
    cognitive_state_width: Literal[256]
    limited_workspace_slot_count: Literal[8]
    limited_workspace_winners_per_cycle: Literal[2]
    cognitive_cycle_count: Literal[3]
    fast_weights_enabled: Literal[False]
    ciel_enabled: Literal[False]
    snn_enabled: Literal[False]

    @field_validator(
        "cognitive_module_count",
        "cognitive_state_width",
        "limited_workspace_slot_count",
        "limited_workspace_winners_per_cycle",
        "cognitive_cycle_count",
        mode="before",
    )
    @classmethod
    def validate_exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            message = "four-arm architecture constants must be exact integers"
            raise ValueError(message)
        return value

    @field_validator(
        "fast_weights_enabled",
        "ciel_enabled",
        "snn_enabled",
        mode="before",
    )
    @classmethod
    def validate_exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            message = "four-arm feature gates must be exact booleans"
            raise ValueError(message)
        return value


class FourArmMechanisticCriterionV1(_StudyModel):
    treatment_arm: Literal["D"]
    control_arms: tuple[Literal["B"], Literal["C"]]
    relation: Literal["strictly_outperforms"]
    interpretation: Literal["supports_competition_and_global_broadcast_usefulness"]
    quantitative_rule_status: Literal["not_recovered"]

    @field_validator("control_arms")
    @classmethod
    def validate_control_arms(
        cls,
        value: tuple[Literal["B"], Literal["C"]],
    ) -> tuple[Literal["B"], Literal["C"]]:
        if type(value) is not tuple or value != ("B", "C"):
            message = "mechanistic controls must be the exact ordered tuple ('B', 'C')"
            raise ValueError(message)
        return value


class FourArmProtocolPayloadV1(_StudyModel):
    protocol_id: Literal["iris-gw-0.1-four-arm"]
    research_question: Literal["limited_competitive_workspace_advantage"]
    source: FourArmProtocolSourceV1
    architecture: FourArmArchitectureConstantsV1
    arms: tuple[
        FourArmDefinitionV1,
        FourArmDefinitionV1,
        FourArmDefinitionV1,
        FourArmDefinitionV1,
    ]
    task_families: tuple[
        StudyTaskFamilyV1,
        StudyTaskFamilyV1,
        StudyTaskFamilyV1,
        StudyTaskFamilyV1,
        StudyTaskFamilyV1,
    ]
    mechanistic_criterion: FourArmMechanisticCriterionV1
    unresolved_items: tuple[
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
        StudyUnresolvedItemV1,
    ]
    execution_status: Literal["blocked_until_explicit_protocol_and_arm_implementations"]

    @model_validator(mode="after")
    def validate_recovered_protocol(self) -> Self:
        observed_arms = tuple((arm.arm_id, arm.arm_kind) for arm in self.arms)
        if observed_arms != _EXPECTED_ARMS:
            message = "four-arm protocol requires exact ordered A/B/C/D definitions"
            raise ValueError(message)
        if self.task_families != _EXPECTED_TASK_FAMILIES:
            message = "four-arm protocol requires all recovered task families in source order"
            raise ValueError(message)
        if self.unresolved_items != _EXPECTED_UNRESOLVED_ITEMS:
            message = "four-arm protocol unresolved items must remain explicit and ordered"
            raise ValueError(message)
        return self


class IrisFourArmProtocolDeclarationV1(_StudyModel):
    schema_identity: FourArmProtocolSchemaV1
    payload_sha256: str
    payload: FourArmProtocolPayloadV1

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            message = "four-arm protocol payload_sha256 must be lower-case hexadecimal SHA-256"
            raise ValueError(message)
        return value


__all__ = [
    "FourArmArchitectureConstantsV1",
    "FourArmDefinitionV1",
    "FourArmMechanisticCriterionV1",
    "FourArmProtocolPayloadV1",
    "FourArmProtocolSchemaV1",
    "FourArmProtocolSourceV1",
    "IrisFourArmProtocolDeclarationV1",
    "StudyArmIdV1",
    "StudyArmKindV1",
    "StudyTaskFamilyV1",
    "StudyUnresolvedItemV1",
]
