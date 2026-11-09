"""Versioned request-graph profiles for the recovered four-arm study."""

from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, model_validator

from iris.application.base_request import (
    BaseLlmRequestOrchestrator,
    build_base_llm_request_orchestrator,
)
from iris.application.composition import build_request_orchestrator
from iris.application.errors import IrisArmExecutionProfileError
from iris.application.no_workspace_request import (
    CognitiveNoWorkspaceRequestOrchestrator,
    build_no_workspace_request_orchestrator,
)
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.request import IrisRequestOrchestrator
from iris.application.shared_buffer_request import (
    CognitiveSharedBufferRequestOrchestrator,
    build_shared_buffer_request_orchestrator,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts import StudyArmIdV1, StudyArmKindV1
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256

_PROTOCOL_DECLARATION_PAYLOAD_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
_PROFILE_SET_DOMAIN = "iris/arm-execution-profile-set/v1"
_PROFILE_SET_DOMAIN_V2 = "iris/arm-execution-profile-set/v2"
_PROFILE_SET_DOMAIN_V3 = "iris/arm-execution-profile-set/v3"
_PROFILE_SET_DOMAIN_V4 = "iris/arm-execution-profile-set/v4"
_SHARED_BUFFER_POLICY_SHA256: Literal[
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
] = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"

type ArmRequestExecutionStatusV1 = Literal["executable", "blocked"]
type ArmTeacherExecutionStatusV1 = Literal["executable", "not_applicable", "blocked"]
type ArmRequestGraphV1 = Literal[
    "frozen_backbone_prompt_only_v1",
    "unresolved_modules_no_workspace",
    "unresolved_modules_unlimited_shared_buffer",
    "limited_competitive_workspace_v1",
]
type ArmTeacherGraphV1 = Literal[
    "none_frozen_backbone",
    "unresolved",
    "limited_competitive_workspace_teacher_v1",
]
type ArmMemoryModelV1 = Literal[
    "none",
    "unlimited_shared_buffer",
    "limited_competitive_workspace",
]
type ArmTraceContractV1 = Literal[
    "empty_workspace_trace",
    "unresolved",
    "workspace_state_per_cycle",
]
type ArmProfileBlockerV1 = Literal[
    "module_output_to_backbone_bridge",
    "cognitive_cycle_feedback_semantics",
    "checkpoint_component_ownership",
    "unlimited_buffer_capacity_and_growth",
    "unlimited_buffer_read_write_and_broadcast_lifecycle",
    "unlimited_buffer_latent_bridge_cardinality",
]
type ArmRequestGraphV2 = Literal[
    "frozen_backbone_prompt_only_v1",
    "independent_recurrent_modules_zero_memory_read_v1",
    "unresolved_modules_unlimited_shared_buffer",
    "limited_competitive_workspace_v1",
]
type ArmTeacherGraphV3 = Literal[
    "none_frozen_backbone",
    "unresolved",
    "independent_recurrent_modules_zero_memory_read_teacher_v1",
    "limited_competitive_workspace_teacher_v1",
]
type ArmRequestGraphV4 = Literal[
    "frozen_backbone_prompt_only_v1",
    "independent_recurrent_modules_zero_memory_read_v1",
    "append_only_all_proposals_prior_cycle_shared_buffer_v1",
    "limited_competitive_workspace_v1",
]
type ArmTeacherGraphV4 = Literal[
    "none_frozen_backbone",
    "independent_recurrent_modules_zero_memory_read_teacher_v1",
    "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
    "limited_competitive_workspace_teacher_v1",
]
type ArmTraceContractV4 = Literal[
    "empty_workspace_trace",
    "append_only_shared_buffer_per_cycle_v1",
    "workspace_state_per_cycle",
]


class _ProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class IrisArmExecutionProfileV1(_ProfileModel):
    """One exact request/training graph resolution for a recovered study arm."""

    schema_version: Literal[1]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    arm_id: StudyArmIdV1
    arm_kind: StudyArmKindV1
    request_status: ArmRequestExecutionStatusV1
    request_graph: ArmRequestGraphV1
    teacher_status: ArmTeacherExecutionStatusV1
    teacher_graph: ArmTeacherGraphV1
    memory_model: ArmMemoryModelV1
    trace_contract: ArmTraceContractV1
    blocker_codes: tuple[ArmProfileBlockerV1, ...]

    @model_validator(mode="after")
    def validate_exact_profile(self) -> Self:
        expected = _expected_profile_values(self.arm_id)
        observed = (
            self.arm_kind,
            self.request_status,
            self.request_graph,
            self.teacher_status,
            self.teacher_graph,
            self.memory_model,
            self.trace_contract,
            self.blocker_codes,
        )
        if observed != expected:
            message = f"arm {self.arm_id} execution profile differs from version 1"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileSetV1(_ProfileModel):
    """Ordered profile registry that remains fail-closed for unresolved arms."""

    schema_version: Literal[1]
    profile_set_id: Literal["iris-gw-0.1-arm-execution-profiles"]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    request_execution_status: Literal["arms_a_and_d_executable_b_and_c_blocked"]
    scientific_launch_status: Literal["blocked"]
    profiles: tuple[
        IrisArmExecutionProfileV1,
        IrisArmExecutionProfileV1,
        IrisArmExecutionProfileV1,
        IrisArmExecutionProfileV1,
    ]

    @model_validator(mode="after")
    def validate_exact_profile_set(self) -> Self:
        if tuple(profile.arm_id for profile in self.profiles) != ("A", "B", "C", "D"):
            message = "arm execution profiles must preserve exact A/B/C/D order"
            raise ValueError(message)
        expected = tuple(_build_profile(arm_id) for arm_id in ("A", "B", "C", "D"))
        if self.profiles != expected:
            message = "arm execution profile set differs from version 1"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileV2(_ProfileModel):
    """One exact profile after resolving the arm-B request graph."""

    schema_version: Literal[2]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    arm_id: StudyArmIdV1
    arm_kind: StudyArmKindV1
    request_status: ArmRequestExecutionStatusV1
    request_graph: ArmRequestGraphV2
    teacher_status: ArmTeacherExecutionStatusV1
    teacher_graph: ArmTeacherGraphV1
    memory_model: ArmMemoryModelV1
    trace_contract: ArmTraceContractV1
    blocker_codes: tuple[ArmProfileBlockerV1, ...]

    @model_validator(mode="after")
    def validate_exact_profile(self) -> Self:
        expected = _expected_profile_values_v2(self.arm_id)
        observed = (
            self.arm_kind,
            self.request_status,
            self.request_graph,
            self.teacher_status,
            self.teacher_graph,
            self.memory_model,
            self.trace_contract,
            self.blocker_codes,
        )
        if observed != expected:
            message = f"arm {self.arm_id} execution profile differs from version 2"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileSetV2(_ProfileModel):
    """Profile registry with request graphs A/B/D and fail-closed C."""

    schema_version: Literal[2]
    profile_set_id: Literal["iris-gw-0.1-arm-execution-profiles"]
    predecessor_profile_set_sha256: Literal[
        "8f2bd7782f545076a0579854f46ca9a391e2eab0ae0557062f2b124c1c943d32"
    ]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    request_execution_status: Literal["arms_a_b_and_d_executable_c_blocked"]
    scientific_launch_status: Literal["blocked"]
    profiles: tuple[
        IrisArmExecutionProfileV2,
        IrisArmExecutionProfileV2,
        IrisArmExecutionProfileV2,
        IrisArmExecutionProfileV2,
    ]

    @model_validator(mode="after")
    def validate_exact_profile_set(self) -> Self:
        if tuple(profile.arm_id for profile in self.profiles) != ("A", "B", "C", "D"):
            message = "arm execution profiles v2 must preserve exact A/B/C/D order"
            raise ValueError(message)
        expected = tuple(_build_profile_v2(arm_id) for arm_id in ("A", "B", "C", "D"))
        if self.profiles != expected:
            message = "arm execution profile set differs from version 2"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileV3(_ProfileModel):
    """One exact profile after resolving arm-B teacher and checkpoint ownership."""

    schema_version: Literal[3]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    arm_id: StudyArmIdV1
    arm_kind: StudyArmKindV1
    request_status: ArmRequestExecutionStatusV1
    request_graph: ArmRequestGraphV2
    teacher_status: ArmTeacherExecutionStatusV1
    teacher_graph: ArmTeacherGraphV3
    memory_model: ArmMemoryModelV1
    trace_contract: ArmTraceContractV1
    blocker_codes: tuple[ArmProfileBlockerV1, ...]

    @model_validator(mode="after")
    def validate_exact_profile(self) -> Self:
        expected = _expected_profile_values_v3(self.arm_id)
        observed = (
            self.arm_kind,
            self.request_status,
            self.request_graph,
            self.teacher_status,
            self.teacher_graph,
            self.memory_model,
            self.trace_contract,
            self.blocker_codes,
        )
        if observed != expected:
            message = f"arm {self.arm_id} execution profile differs from version 3"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileSetV3(_ProfileModel):
    """Profile registry with A/B/D requests, B/D teachers, and fail-closed C."""

    schema_version: Literal[3]
    profile_set_id: Literal["iris-gw-0.1-arm-execution-profiles"]
    predecessor_profile_set_sha256: Literal[
        "d3166e1dd70732fc6e540fe3ac4bf27928e79ae1470af3cc95bc0030d839f3c3"
    ]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    request_execution_status: Literal["arms_a_b_and_d_executable_c_blocked"]
    scientific_launch_status: Literal["blocked"]
    profiles: tuple[
        IrisArmExecutionProfileV3,
        IrisArmExecutionProfileV3,
        IrisArmExecutionProfileV3,
        IrisArmExecutionProfileV3,
    ]

    @model_validator(mode="after")
    def validate_exact_profile_set(self) -> Self:
        if tuple(profile.arm_id for profile in self.profiles) != ("A", "B", "C", "D"):
            message = "arm execution profiles v3 must preserve exact A/B/C/D order"
            raise ValueError(message)
        expected = tuple(_build_profile_v3(arm_id) for arm_id in ("A", "B", "C", "D"))
        if self.profiles != expected:
            message = "arm execution profile set differs from version 3"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileV4(_ProfileModel):
    """One exact profile after resolving the complete arm-C graph."""

    schema_version: Literal[4]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    arm_id: StudyArmIdV1
    arm_kind: StudyArmKindV1
    request_status: ArmRequestExecutionStatusV1
    request_graph: ArmRequestGraphV4
    teacher_status: ArmTeacherExecutionStatusV1
    teacher_graph: ArmTeacherGraphV4
    memory_model: ArmMemoryModelV1
    trace_contract: ArmTraceContractV4
    blocker_codes: tuple[ArmProfileBlockerV1, ...]

    @model_validator(mode="after")
    def validate_exact_profile(self) -> Self:
        expected = _expected_profile_values_v4(self.arm_id)
        observed = (
            self.arm_kind,
            self.request_status,
            self.request_graph,
            self.teacher_status,
            self.teacher_graph,
            self.memory_model,
            self.trace_contract,
            self.blocker_codes,
        )
        if observed != expected:
            message = f"arm {self.arm_id} execution profile differs from version 4"
            raise ValueError(message)
        return self


class IrisArmExecutionProfileSetV4(_ProfileModel):
    """All request graphs and every applicable teacher graph, with launch blocked."""

    schema_version: Literal[4]
    profile_set_id: Literal["iris-gw-0.1-arm-execution-profiles"]
    predecessor_profile_set_sha256: Literal[
        "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
    ]
    protocol_declaration_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    shared_buffer_policy_sha256: Literal[
        "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
    ]
    request_execution_status: Literal["arms_a_b_c_and_d_executable"]
    scientific_launch_status: Literal["blocked"]
    profiles: tuple[
        IrisArmExecutionProfileV4,
        IrisArmExecutionProfileV4,
        IrisArmExecutionProfileV4,
        IrisArmExecutionProfileV4,
    ]

    @model_validator(mode="after")
    def validate_exact_profile_set(self) -> Self:
        if tuple(profile.arm_id for profile in self.profiles) != ("A", "B", "C", "D"):
            message = "arm execution profiles v4 must preserve exact A/B/C/D order"
            raise ValueError(message)
        expected = tuple(_build_profile_v4(arm_id) for arm_id in ("A", "B", "C", "D"))
        if self.profiles != expected:
            message = "arm execution profile set differs from version 4"
            raise ValueError(message)
        if self.shared_buffer_policy_sha256 != shared_buffer_policy_sha256_v1():
            message = "arm execution profile set v4 has a different shared-buffer policy"
            raise ValueError(message)
        arm_c = self.profiles[2]
        if (
            arm_c.arm_id != "C"
            or arm_c.request_graph != "append_only_all_proposals_prior_cycle_shared_buffer_v1"
            or arm_c.teacher_graph
            != "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
            or arm_c.trace_contract != "append_only_shared_buffer_per_cycle_v1"
        ):
            message = "arm execution profile set v4 does not bind policy semantics to arm C"
            raise ValueError(message)
        return self


type ArmRequestOrchestratorV1 = BaseLlmRequestOrchestrator | IrisRequestOrchestrator
type ArmRequestOrchestratorV2 = (
    BaseLlmRequestOrchestrator | CognitiveNoWorkspaceRequestOrchestrator | IrisRequestOrchestrator
)
type ArmRequestOrchestratorV3 = ArmRequestOrchestratorV2
type ArmTeacherForcingModelV3 = IrisNoWorkspaceTeacherForcingModel | IrisTeacherForcingModel
type ArmRequestOrchestratorV4 = (
    BaseLlmRequestOrchestrator
    | CognitiveNoWorkspaceRequestOrchestrator
    | CognitiveSharedBufferRequestOrchestrator
    | IrisRequestOrchestrator
)
type ArmTeacherForcingModelV4 = (
    IrisNoWorkspaceTeacherForcingModel
    | IrisSharedBufferTeacherForcingModel
    | IrisTeacherForcingModel
)


def _expected_profile_values(
    arm_id: StudyArmIdV1,
) -> tuple[
    StudyArmKindV1,
    ArmRequestExecutionStatusV1,
    ArmRequestGraphV1,
    ArmTeacherExecutionStatusV1,
    ArmTeacherGraphV1,
    ArmMemoryModelV1,
    ArmTraceContractV1,
    tuple[ArmProfileBlockerV1, ...],
]:
    if arm_id == "A":
        return (
            "base_llm",
            "executable",
            "frozen_backbone_prompt_only_v1",
            "not_applicable",
            "none_frozen_backbone",
            "none",
            "empty_workspace_trace",
            (),
        )
    if arm_id == "B":
        return (
            "llm_modules_no_workspace",
            "blocked",
            "unresolved_modules_no_workspace",
            "blocked",
            "unresolved",
            "none",
            "unresolved",
            (
                "module_output_to_backbone_bridge",
                "cognitive_cycle_feedback_semantics",
                "checkpoint_component_ownership",
            ),
        )
    if arm_id == "C":
        return (
            "llm_modules_unlimited_shared_buffer",
            "blocked",
            "unresolved_modules_unlimited_shared_buffer",
            "blocked",
            "unresolved",
            "unlimited_shared_buffer",
            "unresolved",
            (
                "unlimited_buffer_capacity_and_growth",
                "unlimited_buffer_read_write_and_broadcast_lifecycle",
                "unlimited_buffer_latent_bridge_cardinality",
                "checkpoint_component_ownership",
            ),
        )
    return (
        "llm_modules_limited_competitive_workspace",
        "executable",
        "limited_competitive_workspace_v1",
        "executable",
        "limited_competitive_workspace_teacher_v1",
        "limited_competitive_workspace",
        "workspace_state_per_cycle",
        (),
    )


def _build_profile(arm_id: StudyArmIdV1) -> IrisArmExecutionProfileV1:
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values(arm_id)
    return IrisArmExecutionProfileV1(
        schema_version=1,
        protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
        arm_id=arm_id,
        arm_kind=arm_kind,
        request_status=request_status,
        request_graph=request_graph,
        teacher_status=teacher_status,
        teacher_graph=teacher_graph,
        memory_model=memory_model,
        trace_contract=trace_contract,
        blocker_codes=blocker_codes,
    )


def _expected_profile_values_v2(
    arm_id: StudyArmIdV1,
) -> tuple[
    StudyArmKindV1,
    ArmRequestExecutionStatusV1,
    ArmRequestGraphV2,
    ArmTeacherExecutionStatusV1,
    ArmTeacherGraphV1,
    ArmMemoryModelV1,
    ArmTraceContractV1,
    tuple[ArmProfileBlockerV1, ...],
]:
    if arm_id == "A":
        return (
            "base_llm",
            "executable",
            "frozen_backbone_prompt_only_v1",
            "not_applicable",
            "none_frozen_backbone",
            "none",
            "empty_workspace_trace",
            (),
        )
    if arm_id == "B":
        return (
            "llm_modules_no_workspace",
            "executable",
            "independent_recurrent_modules_zero_memory_read_v1",
            "blocked",
            "unresolved",
            "none",
            "empty_workspace_trace",
            ("checkpoint_component_ownership",),
        )
    if arm_id == "C":
        return (
            "llm_modules_unlimited_shared_buffer",
            "blocked",
            "unresolved_modules_unlimited_shared_buffer",
            "blocked",
            "unresolved",
            "unlimited_shared_buffer",
            "unresolved",
            (
                "unlimited_buffer_capacity_and_growth",
                "unlimited_buffer_read_write_and_broadcast_lifecycle",
                "unlimited_buffer_latent_bridge_cardinality",
                "checkpoint_component_ownership",
            ),
        )
    return (
        "llm_modules_limited_competitive_workspace",
        "executable",
        "limited_competitive_workspace_v1",
        "executable",
        "limited_competitive_workspace_teacher_v1",
        "limited_competitive_workspace",
        "workspace_state_per_cycle",
        (),
    )


def _build_profile_v2(arm_id: StudyArmIdV1) -> IrisArmExecutionProfileV2:
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values_v2(arm_id)
    return IrisArmExecutionProfileV2(
        schema_version=2,
        protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
        arm_id=arm_id,
        arm_kind=arm_kind,
        request_status=request_status,
        request_graph=request_graph,
        teacher_status=teacher_status,
        teacher_graph=teacher_graph,
        memory_model=memory_model,
        trace_contract=trace_contract,
        blocker_codes=blocker_codes,
    )


def _expected_profile_values_v3(
    arm_id: StudyArmIdV1,
) -> tuple[
    StudyArmKindV1,
    ArmRequestExecutionStatusV1,
    ArmRequestGraphV2,
    ArmTeacherExecutionStatusV1,
    ArmTeacherGraphV3,
    ArmMemoryModelV1,
    ArmTraceContractV1,
    tuple[ArmProfileBlockerV1, ...],
]:
    if arm_id == "B":
        return (
            "llm_modules_no_workspace",
            "executable",
            "independent_recurrent_modules_zero_memory_read_v1",
            "executable",
            "independent_recurrent_modules_zero_memory_read_teacher_v1",
            "none",
            "empty_workspace_trace",
            (),
        )
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values_v2(arm_id)
    return (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        cast(ArmTeacherGraphV3, teacher_graph),
        memory_model,
        trace_contract,
        blocker_codes,
    )


def _build_profile_v3(arm_id: StudyArmIdV1) -> IrisArmExecutionProfileV3:
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values_v3(arm_id)
    return IrisArmExecutionProfileV3(
        schema_version=3,
        protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
        arm_id=arm_id,
        arm_kind=arm_kind,
        request_status=request_status,
        request_graph=request_graph,
        teacher_status=teacher_status,
        teacher_graph=teacher_graph,
        memory_model=memory_model,
        trace_contract=trace_contract,
        blocker_codes=blocker_codes,
    )


def _expected_profile_values_v4(
    arm_id: StudyArmIdV1,
) -> tuple[
    StudyArmKindV1,
    ArmRequestExecutionStatusV1,
    ArmRequestGraphV4,
    ArmTeacherExecutionStatusV1,
    ArmTeacherGraphV4,
    ArmMemoryModelV1,
    ArmTraceContractV4,
    tuple[ArmProfileBlockerV1, ...],
]:
    if arm_id == "C":
        return (
            "llm_modules_unlimited_shared_buffer",
            "executable",
            "append_only_all_proposals_prior_cycle_shared_buffer_v1",
            "executable",
            "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
            "unlimited_shared_buffer",
            "append_only_shared_buffer_per_cycle_v1",
            (),
        )
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values_v3(arm_id)
    return (
        arm_kind,
        request_status,
        cast(ArmRequestGraphV4, request_graph),
        teacher_status,
        cast(ArmTeacherGraphV4, teacher_graph),
        memory_model,
        cast(ArmTraceContractV4, trace_contract),
        blocker_codes,
    )


def _build_profile_v4(arm_id: StudyArmIdV1) -> IrisArmExecutionProfileV4:
    (
        arm_kind,
        request_status,
        request_graph,
        teacher_status,
        teacher_graph,
        memory_model,
        trace_contract,
        blocker_codes,
    ) = _expected_profile_values_v4(arm_id)
    return IrisArmExecutionProfileV4(
        schema_version=4,
        protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
        arm_id=arm_id,
        arm_kind=arm_kind,
        request_status=request_status,
        request_graph=request_graph,
        teacher_status=teacher_status,
        teacher_graph=teacher_graph,
        memory_model=memory_model,
        trace_contract=trace_contract,
        blocker_codes=blocker_codes,
    )


def _validate_arm_id(arm_id: object) -> StudyArmIdV1:
    if type(arm_id) is not str or arm_id not in {"A", "B", "C", "D"}:
        message = "arm_id must be exactly one of A, B, C, or D"
        raise IrisArmExecutionProfileError(message)
    return cast(StudyArmIdV1, arm_id)


def build_arm_execution_profile_set_v1() -> IrisArmExecutionProfileSetV1:
    """Build the immutable profile set bound to the recovered protocol hash."""
    try:
        return IrisArmExecutionProfileSetV1(
            schema_version=1,
            profile_set_id="iris-gw-0.1-arm-execution-profiles",
            protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            request_execution_status="arms_a_and_d_executable_b_and_c_blocked",
            scientific_launch_status="blocked",
            profiles=(
                _build_profile("A"),
                _build_profile("B"),
                _build_profile("C"),
                _build_profile("D"),
            ),
        )
    except Exception as error:
        message = "arm execution profile set v1 could not be built"
        raise IrisArmExecutionProfileError(message) from error


def arm_execution_profile_set_sha256_v1() -> str:
    """Return the domain-separated canonical identity of profile set v1."""
    profile_set = build_arm_execution_profile_set_v1()
    canonical = canonical_json_bytes(profile_set.model_dump(mode="json"))
    return domain_sha256(_PROFILE_SET_DOMAIN, canonical)


def get_arm_execution_profile_v1(arm_id: object) -> IrisArmExecutionProfileV1:
    """Return one owned exact profile without silently resolving blocked semantics."""
    owned_arm_id = _validate_arm_id(arm_id)
    profile = build_arm_execution_profile_set_v1().profiles[
        {"A": 0, "B": 1, "C": 2, "D": 3}[owned_arm_id]
    ]
    return IrisArmExecutionProfileV1.model_validate(profile.model_dump(mode="python"))


def require_arm_request_executable_v1(arm_id: object) -> IrisArmExecutionProfileV1:
    """Reject B/C before any model or configuration object is inspected."""
    profile = get_arm_execution_profile_v1(arm_id)
    if profile.request_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} request graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def build_arm_request_orchestrator_v1(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmRequestOrchestratorV1:
    """Compose A or D and fail before touching runtime inputs for blocked B/C."""
    profile = require_arm_request_executable_v1(arm_id)
    if profile.arm_id == "A":
        if cognitive_config is not None or workspace_config is not None:
            message = "arm A rejects cognitive and workspace configuration"
            raise IrisArmExecutionProfileError(message)
        return build_base_llm_request_orchestrator(
            backbone=backbone,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
    )


def build_arm_execution_profile_set_v2() -> IrisArmExecutionProfileSetV2:
    """Build the immutable successor profile set with arm-B request semantics."""
    try:
        return IrisArmExecutionProfileSetV2(
            schema_version=2,
            profile_set_id="iris-gw-0.1-arm-execution-profiles",
            predecessor_profile_set_sha256=(
                "8f2bd7782f545076a0579854f46ca9a391e2eab0ae0557062f2b124c1c943d32"
            ),
            protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            request_execution_status="arms_a_b_and_d_executable_c_blocked",
            scientific_launch_status="blocked",
            profiles=(
                _build_profile_v2("A"),
                _build_profile_v2("B"),
                _build_profile_v2("C"),
                _build_profile_v2("D"),
            ),
        )
    except Exception as error:
        message = "arm execution profile set v2 could not be built"
        raise IrisArmExecutionProfileError(message) from error


def arm_execution_profile_set_sha256_v2() -> str:
    """Return the domain-separated canonical identity of profile set v2."""
    profile_set = build_arm_execution_profile_set_v2()
    canonical = canonical_json_bytes(profile_set.model_dump(mode="json"))
    return domain_sha256(_PROFILE_SET_DOMAIN_V2, canonical)


def get_arm_execution_profile_v2(arm_id: object) -> IrisArmExecutionProfileV2:
    """Return one owned profile from the A/B/D request-capable successor set."""
    owned_arm_id = _validate_arm_id(arm_id)
    profile = build_arm_execution_profile_set_v2().profiles[
        {"A": 0, "B": 1, "C": 2, "D": 3}[owned_arm_id]
    ]
    return IrisArmExecutionProfileV2.model_validate(profile.model_dump(mode="python"))


def require_arm_request_executable_v2(arm_id: object) -> IrisArmExecutionProfileV2:
    """Reject unresolved C before any runtime object is inspected."""
    profile = get_arm_execution_profile_v2(arm_id)
    if profile.request_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} request graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def build_arm_request_orchestrator_v2(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmRequestOrchestratorV2:
    """Compose A/B/D and fail before runtime inspection for unresolved C."""
    profile = require_arm_request_executable_v2(arm_id)
    if profile.arm_id == "A":
        if cognitive_config is not None or workspace_config is not None:
            message = "arm A rejects cognitive and workspace configuration"
            raise IrisArmExecutionProfileError(message)
        return build_base_llm_request_orchestrator(
            backbone=backbone,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if profile.arm_id == "B":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm B requires exact cognitive config and rejects workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_no_workspace_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
    )


def build_arm_execution_profile_set_v3() -> IrisArmExecutionProfileSetV3:
    """Build the immutable successor with arm-B teacher/checkpoint ownership."""
    try:
        return IrisArmExecutionProfileSetV3(
            schema_version=3,
            profile_set_id="iris-gw-0.1-arm-execution-profiles",
            predecessor_profile_set_sha256=(
                "d3166e1dd70732fc6e540fe3ac4bf27928e79ae1470af3cc95bc0030d839f3c3"
            ),
            protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            request_execution_status="arms_a_b_and_d_executable_c_blocked",
            scientific_launch_status="blocked",
            profiles=(
                _build_profile_v3("A"),
                _build_profile_v3("B"),
                _build_profile_v3("C"),
                _build_profile_v3("D"),
            ),
        )
    except Exception as error:
        message = "arm execution profile set v3 could not be built"
        raise IrisArmExecutionProfileError(message) from error


def arm_execution_profile_set_sha256_v3() -> str:
    """Return the domain-separated canonical identity of profile set v3."""
    profile_set = build_arm_execution_profile_set_v3()
    canonical = canonical_json_bytes(profile_set.model_dump(mode="json"))
    return domain_sha256(_PROFILE_SET_DOMAIN_V3, canonical)


def get_arm_execution_profile_v3(arm_id: object) -> IrisArmExecutionProfileV3:
    """Return one owned profile from the arm-B-teacher-capable successor set."""
    owned_arm_id = _validate_arm_id(arm_id)
    profile = build_arm_execution_profile_set_v3().profiles[
        {"A": 0, "B": 1, "C": 2, "D": 3}[owned_arm_id]
    ]
    return IrisArmExecutionProfileV3.model_validate(profile.model_dump(mode="python"))


def require_arm_request_executable_v3(arm_id: object) -> IrisArmExecutionProfileV3:
    """Reject unresolved C before any request runtime object is inspected."""
    profile = get_arm_execution_profile_v3(arm_id)
    if profile.request_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} request graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def require_arm_teacher_executable_v3(arm_id: object) -> IrisArmExecutionProfileV3:
    """Resolve teacher applicability before inspecting configs or a backbone."""
    profile = get_arm_execution_profile_v3(arm_id)
    if profile.teacher_status == "not_applicable":
        message = f"arm {profile.arm_id} has no trainable teacher graph"
        raise IrisArmExecutionProfileError(message)
    if profile.teacher_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} teacher graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def build_arm_request_orchestrator_v3(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmRequestOrchestratorV3:
    """Compose A/B/D requests under profile v3 and fail closed for C."""
    profile = require_arm_request_executable_v3(arm_id)
    if profile.arm_id == "A":
        if cognitive_config is not None or workspace_config is not None:
            message = "arm A rejects cognitive and workspace configuration"
            raise IrisArmExecutionProfileError(message)
        return build_base_llm_request_orchestrator(
            backbone=backbone,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if profile.arm_id == "B":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm B requires exact cognitive config and rejects workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_no_workspace_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
    )


def build_arm_teacher_forcing_model_v3(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmTeacherForcingModelV3:
    """Compose B or D teacher graphs and reject A/C before runtime inspection."""
    profile = require_arm_teacher_executable_v3(arm_id)
    if profile.arm_id == "B":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm B teacher requires exact cognitive config and no workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_no_workspace_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive_config,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D teacher requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_teacher_forcing_model(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
    )


def build_arm_execution_profile_set_v4() -> IrisArmExecutionProfileSetV4:
    """Build the immutable successor with executable arm-C request and teacher graphs."""
    try:
        policy_sha256 = shared_buffer_policy_sha256_v1()
        if policy_sha256 != _SHARED_BUFFER_POLICY_SHA256:
            message = "canonical arm-C shared-buffer policy identity changed"
            raise ValueError(message)
        return IrisArmExecutionProfileSetV4(
            schema_version=4,
            profile_set_id="iris-gw-0.1-arm-execution-profiles",
            predecessor_profile_set_sha256=(
                "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
            ),
            protocol_declaration_payload_sha256=_PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            shared_buffer_policy_sha256=_SHARED_BUFFER_POLICY_SHA256,
            request_execution_status="arms_a_b_c_and_d_executable",
            scientific_launch_status="blocked",
            profiles=(
                _build_profile_v4("A"),
                _build_profile_v4("B"),
                _build_profile_v4("C"),
                _build_profile_v4("D"),
            ),
        )
    except Exception as error:
        message = "arm execution profile set v4 could not be built"
        raise IrisArmExecutionProfileError(message) from error


def arm_execution_profile_set_sha256_v4() -> str:
    """Return the domain-separated canonical identity of profile set v4."""
    profile_set = build_arm_execution_profile_set_v4()
    canonical = canonical_json_bytes(profile_set.model_dump(mode="json"))
    return domain_sha256(_PROFILE_SET_DOMAIN_V4, canonical)


def get_arm_execution_profile_v4(arm_id: object) -> IrisArmExecutionProfileV4:
    """Return one owned exact profile from the complete request-graph set."""
    owned_arm_id = _validate_arm_id(arm_id)
    profile = build_arm_execution_profile_set_v4().profiles[
        {"A": 0, "B": 1, "C": 2, "D": 3}[owned_arm_id]
    ]
    return IrisArmExecutionProfileV4.model_validate(profile.model_dump(mode="python"))


def require_arm_request_executable_v4(arm_id: object) -> IrisArmExecutionProfileV4:
    """Resolve one of the four executable request graphs before runtime inspection."""
    profile = get_arm_execution_profile_v4(arm_id)
    if profile.request_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} request graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def require_arm_teacher_executable_v4(arm_id: object) -> IrisArmExecutionProfileV4:
    """Resolve B/C/D teacher execution and reject the frozen arm-A baseline."""
    profile = get_arm_execution_profile_v4(arm_id)
    if profile.teacher_status == "not_applicable":
        message = f"arm {profile.arm_id} has no trainable teacher graph"
        raise IrisArmExecutionProfileError(message)
    if profile.teacher_status != "executable":
        blockers = ", ".join(profile.blocker_codes)
        message = f"arm {profile.arm_id} teacher graph is blocked: {blockers}"
        raise IrisArmExecutionProfileError(message)
    return profile


def build_arm_request_orchestrator_v4(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmRequestOrchestratorV4:
    """Compose the exact A/B/C/D request graph selected by profile set v4."""
    profile = require_arm_request_executable_v4(arm_id)
    if profile.arm_id == "A":
        if cognitive_config is not None or workspace_config is not None:
            message = "arm A rejects cognitive and workspace configuration"
            raise IrisArmExecutionProfileError(message)
        return build_base_llm_request_orchestrator(
            backbone=backbone,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if profile.arm_id == "B":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm B requires exact cognitive config and rejects workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_no_workspace_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if profile.arm_id == "C":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm C requires exact cognitive config and rejects workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_shared_buffer_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive_config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
    )


def build_arm_teacher_forcing_model_v4(
    *,
    arm_id: object,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig | None = None,
    workspace_config: WorkspaceConfig | None = None,
) -> ArmTeacherForcingModelV4:
    """Compose the exact B/C/D teacher graph selected by profile set v4."""
    profile = require_arm_teacher_executable_v4(arm_id)
    if profile.arm_id == "B":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm B teacher requires exact cognitive config and no workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_no_workspace_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive_config,
        )
    if profile.arm_id == "C":
        if type(cognitive_config) is not CognitiveConfig or workspace_config is not None:
            message = "arm C teacher requires exact cognitive config and no workspace config"
            raise IrisArmExecutionProfileError(message)
        return build_shared_buffer_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive_config,
        )
    if (
        type(cognitive_config) is not CognitiveConfig
        or type(workspace_config) is not WorkspaceConfig
    ):
        message = "arm D teacher requires exact cognitive and workspace configurations"
        raise IrisArmExecutionProfileError(message)
    return build_teacher_forcing_model(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
    )


__all__ = [
    "ArmMemoryModelV1",
    "ArmProfileBlockerV1",
    "ArmRequestExecutionStatusV1",
    "ArmRequestGraphV1",
    "ArmRequestGraphV2",
    "ArmRequestGraphV4",
    "ArmRequestOrchestratorV1",
    "ArmRequestOrchestratorV2",
    "ArmRequestOrchestratorV3",
    "ArmRequestOrchestratorV4",
    "ArmTeacherExecutionStatusV1",
    "ArmTeacherForcingModelV3",
    "ArmTeacherForcingModelV4",
    "ArmTeacherGraphV1",
    "ArmTeacherGraphV3",
    "ArmTeacherGraphV4",
    "ArmTraceContractV1",
    "ArmTraceContractV4",
    "IrisArmExecutionProfileSetV1",
    "IrisArmExecutionProfileSetV2",
    "IrisArmExecutionProfileSetV3",
    "IrisArmExecutionProfileSetV4",
    "IrisArmExecutionProfileV1",
    "IrisArmExecutionProfileV2",
    "IrisArmExecutionProfileV3",
    "IrisArmExecutionProfileV4",
    "arm_execution_profile_set_sha256_v1",
    "arm_execution_profile_set_sha256_v2",
    "arm_execution_profile_set_sha256_v3",
    "arm_execution_profile_set_sha256_v4",
    "build_arm_execution_profile_set_v1",
    "build_arm_execution_profile_set_v2",
    "build_arm_execution_profile_set_v3",
    "build_arm_execution_profile_set_v4",
    "build_arm_request_orchestrator_v1",
    "build_arm_request_orchestrator_v2",
    "build_arm_request_orchestrator_v3",
    "build_arm_request_orchestrator_v4",
    "build_arm_teacher_forcing_model_v3",
    "build_arm_teacher_forcing_model_v4",
    "get_arm_execution_profile_v1",
    "get_arm_execution_profile_v2",
    "get_arm_execution_profile_v3",
    "get_arm_execution_profile_v4",
    "require_arm_request_executable_v1",
    "require_arm_request_executable_v2",
    "require_arm_request_executable_v3",
    "require_arm_request_executable_v4",
    "require_arm_teacher_executable_v3",
    "require_arm_teacher_executable_v4",
]
