"""Contract tests for the immutable arm-C-capable execution profile set v4."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import iris.application as application
import iris.application.arm_profiles as profiles_service
from iris.application.arm_profiles import (
    arm_execution_profile_set_sha256_v1,
    arm_execution_profile_set_sha256_v2,
    arm_execution_profile_set_sha256_v3,
    arm_execution_profile_set_sha256_v4,
    build_arm_execution_profile_set_v4,
    build_arm_request_orchestrator_v4,
    build_arm_teacher_forcing_model_v4,
    get_arm_execution_profile_v4,
    require_arm_request_executable_v4,
    require_arm_teacher_executable_v4,
)
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.errors import IrisArmExecutionProfileError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import IrisRequestOrchestrator
from iris.application.shared_buffer_request import CognitiveSharedBufferRequestOrchestrator
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
)
from iris.utils.hashing import canonical_json_file_bytes

_REPOSITORY_ROOT = Path(__file__).parents[2]
_CHECKED_IN_PROFILE_SET_V4 = (
    _REPOSITORY_ROOT / "docs" / "protocols" / "iris-gw-0.1-arm-execution-profiles.v4.json"
)
_PROFILE_SET_SHA256_V1 = "8f2bd7782f545076a0579854f46ca9a391e2eab0ae0557062f2b124c1c943d32"
_PROFILE_SET_SHA256_V2 = "d3166e1dd70732fc6e540fe3ac4bf27928e79ae1470af3cc95bc0030d839f3c3"
_PROFILE_SET_SHA256_V3 = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
_PROFILE_SET_SHA256_V4 = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"


def cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "profile-v4-test-cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "profile-v4-test-workspace",
            "num_slots": 2,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.5,
            "empty_threshold": 0.01,
            "arbiter": {
                "confidence_weight": 0.25,
                "novelty_weight": 0.1,
                "relevance_weight": 0.25,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def test_v4_resolves_only_arm_c_and_keeps_scientific_launch_blocked() -> None:
    profile_set = build_arm_execution_profile_set_v4()
    profiles = profile_set.profiles

    assert application.build_arm_execution_profile_set_v4 is build_arm_execution_profile_set_v4
    assert (
        application.arm_execution_profile_set_sha256_v4
        is arm_execution_profile_set_sha256_v4
    )
    assert profile_set.predecessor_profile_set_sha256 == _PROFILE_SET_SHA256_V3
    assert profile_set.request_execution_status == "arms_a_b_c_and_d_executable"
    assert profile_set.scientific_launch_status == "blocked"
    assert profile_set.shared_buffer_policy_sha256 == shared_buffer_policy_sha256_v1()
    assert tuple(profile.arm_id for profile in profiles) == ("A", "B", "C", "D")
    assert tuple(profile.request_status for profile in profiles) == (
        "executable",
        "executable",
        "executable",
        "executable",
    )
    assert tuple(profile.teacher_status for profile in profiles) == (
        "not_applicable",
        "executable",
        "executable",
        "executable",
    )
    assert all(profile.blocker_codes == () for profile in profiles)

    arm_c = get_arm_execution_profile_v4("C")
    assert arm_c.request_graph == "append_only_all_proposals_prior_cycle_shared_buffer_v1"
    assert arm_c.teacher_graph == ("append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1")
    assert arm_c.trace_contract == "append_only_shared_buffer_per_cycle_v1"
    assert arm_c.memory_model == "unlimited_shared_buffer"
    assert sum("shared_buffer" in profile.request_graph for profile in profiles) == 1
    assert sum("shared_buffer" in profile.teacher_graph for profile in profiles) == 1
    assert sum("shared_buffer" in profile.trace_contract for profile in profiles) == 1


def test_v4_is_canonical_hash_bound_and_preserves_all_predecessor_identities() -> None:
    profile_set = build_arm_execution_profile_set_v4()

    assert arm_execution_profile_set_sha256_v1() == _PROFILE_SET_SHA256_V1
    assert arm_execution_profile_set_sha256_v2() == _PROFILE_SET_SHA256_V2
    assert arm_execution_profile_set_sha256_v3() == _PROFILE_SET_SHA256_V3
    assert arm_execution_profile_set_sha256_v4() == _PROFILE_SET_SHA256_V4
    assert _CHECKED_IN_PROFILE_SET_V4.read_bytes() == canonical_json_file_bytes(
        profile_set.model_dump(mode="json")
    )


def test_v4_rejects_a_changed_shared_buffer_policy_identity() -> None:
    with (
        patch.object(
            profiles_service,
            "shared_buffer_policy_sha256_v1",
            return_value="0" * 64,
        ),
        pytest.raises(IrisArmExecutionProfileError, match="v4 could not be built"),
    ):
        build_arm_execution_profile_set_v4()


def test_v4_profile_lookup_is_owned_and_every_request_is_executable() -> None:
    first = get_arm_execution_profile_v4("C")
    second = get_arm_execution_profile_v4("C")

    assert first == second
    assert first is not second
    assert tuple(require_arm_request_executable_v4(arm).arm_id for arm in "ABCD") == tuple("ABCD")
    with pytest.raises(IrisArmExecutionProfileError, match="exactly one of"):
        get_arm_execution_profile_v4("c")


def test_v4_arm_a_teacher_is_not_applicable_before_runtime_inspection() -> None:
    explosive = cast(Any, object())
    with pytest.raises(IrisArmExecutionProfileError, match="no trainable"):
        build_arm_teacher_forcing_model_v4(
            arm_id="A",
            backbone=explosive,
            cognitive_config=explosive,
            workspace_config=explosive,
        )
    with pytest.raises(IrisArmExecutionProfileError, match="no trainable"):
        require_arm_teacher_executable_v4("A")


def test_v4_arm_c_request_requires_exact_cognitive_config_and_no_workspace() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="requires exact cognitive"):
        build_arm_request_orchestrator_v4(
            arm_id="C",
            backbone=cast(Any, object()),
            stop_token_ids=(2,),
            sampling=cast(Any, object()),
        )
    with pytest.raises(IrisArmExecutionProfileError, match="rejects workspace"):
        build_arm_request_orchestrator_v4(
            arm_id="C",
            backbone=cast(Any, object()),
            stop_token_ids=(2,),
            sampling=cast(Any, object()),
            cognitive_config=cognitive_config(),
            workspace_config=workspace_config(),
        )


def test_v4_arm_c_request_dispatches_only_to_shared_buffer_builder() -> None:
    expected = cast(CognitiveSharedBufferRequestOrchestrator, object())
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    cognitive = cognitive_config()
    with (
        patch.object(
            profiles_service,
            "build_shared_buffer_request_orchestrator",
            return_value=expected,
        ) as build_c,
        patch.object(profiles_service, "build_no_workspace_request_orchestrator") as build_b,
        patch.object(profiles_service, "build_request_orchestrator") as build_d,
    ):
        observed = build_arm_request_orchestrator_v4(
            arm_id="C",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
        )

    assert observed is expected
    build_c.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        stop_token_ids=(2,),
        sampling=sampling,
    )
    build_b.assert_not_called()
    build_d.assert_not_called()


def test_v4_arm_c_teacher_requires_exact_cognitive_config_and_no_workspace() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="requires exact cognitive"):
        build_arm_teacher_forcing_model_v4(
            arm_id="C",
            backbone=cast(Any, object()),
        )
    with pytest.raises(IrisArmExecutionProfileError, match="no workspace"):
        build_arm_teacher_forcing_model_v4(
            arm_id="C",
            backbone=cast(Any, object()),
            cognitive_config=cognitive_config(),
            workspace_config=workspace_config(),
        )


def test_v4_arm_c_teacher_dispatches_only_to_shared_buffer_builder() -> None:
    expected = cast(IrisSharedBufferTeacherForcingModel, object())
    backbone = cast(Any, object())
    cognitive = cognitive_config()
    with (
        patch.object(
            profiles_service,
            "build_shared_buffer_teacher_forcing_model",
            return_value=expected,
        ) as build_c,
        patch.object(profiles_service, "build_no_workspace_teacher_forcing_model") as build_b,
        patch.object(profiles_service, "build_teacher_forcing_model") as build_d,
    ):
        observed = build_arm_teacher_forcing_model_v4(
            arm_id="C",
            backbone=backbone,
            cognitive_config=cognitive,
        )

    assert observed is expected
    build_c.assert_called_once_with(backbone=backbone, cognitive_config=cognitive)
    build_b.assert_not_called()
    build_d.assert_not_called()


def test_v4_preserves_a_b_d_request_dispatch() -> None:
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    cognitive = cognitive_config()
    workspace = workspace_config()
    expected_a = cast(BaseLlmRequestOrchestrator, object())
    expected_b = cast(CognitiveNoWorkspaceRequestOrchestrator, object())
    expected_d = cast(IrisRequestOrchestrator, object())
    with (
        patch.object(
            profiles_service,
            "build_base_llm_request_orchestrator",
            return_value=expected_a,
        ) as build_a,
        patch.object(
            profiles_service,
            "build_no_workspace_request_orchestrator",
            return_value=expected_b,
        ) as build_b,
        patch.object(
            profiles_service,
            "build_request_orchestrator",
            return_value=expected_d,
        ) as build_d,
    ):
        observed_a = build_arm_request_orchestrator_v4(
            arm_id="A", backbone=backbone, stop_token_ids=(2,), sampling=sampling
        )
        observed_b = build_arm_request_orchestrator_v4(
            arm_id="B",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
        )
        observed_d = build_arm_request_orchestrator_v4(
            arm_id="D",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
            workspace_config=workspace,
        )

    assert (observed_a, observed_b, observed_d) == (expected_a, expected_b, expected_d)
    build_a.assert_called_once_with(backbone=backbone, stop_token_ids=(2,), sampling=sampling)
    build_b.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        stop_token_ids=(2,),
        sampling=sampling,
    )
    build_d.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        workspace_config=workspace,
        stop_token_ids=(2,),
        sampling=sampling,
    )


def test_v4_preserves_b_d_teacher_dispatch() -> None:
    backbone = cast(Any, object())
    cognitive = cognitive_config()
    workspace = workspace_config()
    expected_b = cast(IrisNoWorkspaceTeacherForcingModel, object())
    expected_d = cast(IrisTeacherForcingModel, object())
    with (
        patch.object(
            profiles_service,
            "build_no_workspace_teacher_forcing_model",
            return_value=expected_b,
        ) as build_b,
        patch.object(
            profiles_service,
            "build_teacher_forcing_model",
            return_value=expected_d,
        ) as build_d,
    ):
        observed_b = build_arm_teacher_forcing_model_v4(
            arm_id="B", backbone=backbone, cognitive_config=cognitive
        )
        observed_d = build_arm_teacher_forcing_model_v4(
            arm_id="D",
            backbone=backbone,
            cognitive_config=cognitive,
            workspace_config=workspace,
        )

    assert (observed_b, observed_d) == (expected_b, expected_d)
    build_b.assert_called_once_with(backbone=backbone, cognitive_config=cognitive)
    build_d.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        workspace_config=workspace,
    )
