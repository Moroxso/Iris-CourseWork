"""Contract tests for versioned four-arm request-graph profiles."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import iris.application as application
import iris.application.arm_profiles as profiles_service
from iris.application import (
    BaseLlmRequestOrchestrator,
    CognitiveNoWorkspaceRequestOrchestrator,
    IrisArmExecutionProfileError,
    IrisRequestOrchestrator,
    arm_execution_profile_set_sha256_v1,
    arm_execution_profile_set_sha256_v2,
    arm_execution_profile_set_sha256_v3,
    build_arm_execution_profile_set_v1,
    build_arm_execution_profile_set_v2,
    build_arm_execution_profile_set_v3,
    build_arm_request_orchestrator_v1,
    build_arm_request_orchestrator_v2,
    build_arm_request_orchestrator_v3,
    build_arm_teacher_forcing_model_v3,
    get_arm_execution_profile_v1,
    get_arm_execution_profile_v2,
    get_arm_execution_profile_v3,
    require_arm_request_executable_v1,
    require_arm_request_executable_v2,
    require_arm_request_executable_v3,
    require_arm_teacher_executable_v3,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training import IrisNoWorkspaceTeacherForcingModel, IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_file_bytes

_CHECKED_IN_PROFILE_SET = (
    Path(__file__).parents[2] / "docs" / "protocols" / "iris-gw-0.1-arm-execution-profiles.v1.json"
)
_PROFILE_SET_SHA256 = "8f2bd7782f545076a0579854f46ca9a391e2eab0ae0557062f2b124c1c943d32"
_CHECKED_IN_PROFILE_SET_V2 = (
    Path(__file__).parents[2] / "docs" / "protocols" / "iris-gw-0.1-arm-execution-profiles.v2.json"
)
_PROFILE_SET_SHA256_V2 = "d3166e1dd70732fc6e540fe3ac4bf27928e79ae1470af3cc95bc0030d839f3c3"
_CHECKED_IN_PROFILE_SET_V3 = (
    Path(__file__).parents[2] / "docs" / "protocols" / "iris-gw-0.1-arm-execution-profiles.v3.json"
)
_PROFILE_SET_SHA256_V3 = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"


def cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "profile-test-cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "profile-test-workspace",
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


def test_profile_set_is_exact_ordered_and_protocol_bound() -> None:
    profile_set = build_arm_execution_profile_set_v1()

    assert tuple(profile.arm_id for profile in profile_set.profiles) == ("A", "B", "C", "D")
    assert tuple(profile.arm_kind for profile in profile_set.profiles) == (
        "base_llm",
        "llm_modules_no_workspace",
        "llm_modules_unlimited_shared_buffer",
        "llm_modules_limited_competitive_workspace",
    )
    assert tuple(profile.request_status for profile in profile_set.profiles) == (
        "executable",
        "blocked",
        "blocked",
        "executable",
    )
    assert profile_set.scientific_launch_status == "blocked"
    assert profile_set.protocol_declaration_payload_sha256 == (
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    )


def test_profile_set_hash_is_deterministic_and_domain_separated() -> None:
    first = arm_execution_profile_set_sha256_v1()
    second = arm_execution_profile_set_sha256_v1()

    assert first == second
    assert first == _PROFILE_SET_SHA256
    assert first != build_arm_execution_profile_set_v1().protocol_declaration_payload_sha256


def test_checked_in_profile_set_is_the_exact_canonical_builder_output() -> None:
    profile_set = build_arm_execution_profile_set_v1()

    assert _CHECKED_IN_PROFILE_SET.read_bytes() == canonical_json_file_bytes(
        profile_set.model_dump(mode="json")
    )


def test_profile_set_v2_resolves_only_arm_b_request_semantics() -> None:
    profile_set = build_arm_execution_profile_set_v2()
    arm_b = get_arm_execution_profile_v2("B")

    assert profile_set.predecessor_profile_set_sha256 == _PROFILE_SET_SHA256
    assert profile_set.request_execution_status == "arms_a_b_and_d_executable_c_blocked"
    assert tuple(profile.request_status for profile in profile_set.profiles) == (
        "executable",
        "executable",
        "blocked",
        "executable",
    )
    assert arm_b.request_graph == "independent_recurrent_modules_zero_memory_read_v1"
    assert arm_b.request_status == "executable"
    assert arm_b.teacher_status == "blocked"
    assert arm_b.memory_model == "none"
    assert arm_b.trace_contract == "empty_workspace_trace"
    assert arm_b.blocker_codes == ("checkpoint_component_ownership",)
    assert get_arm_execution_profile_v1("B").request_status == "blocked"


def test_checked_in_profile_set_v2_is_exact_and_hash_bound() -> None:
    profile_set = build_arm_execution_profile_set_v2()

    assert arm_execution_profile_set_sha256_v2() == _PROFILE_SET_SHA256_V2
    assert _CHECKED_IN_PROFILE_SET_V2.read_bytes() == canonical_json_file_bytes(
        profile_set.model_dump(mode="json")
    )


def test_profile_set_v3_resolves_only_arm_b_teacher_and_checkpoint_blocker() -> None:
    profile_set = build_arm_execution_profile_set_v3()
    arm_b = get_arm_execution_profile_v3("B")
    assert profile_set.predecessor_profile_set_sha256 == _PROFILE_SET_SHA256_V2
    assert profile_set.request_execution_status == "arms_a_b_and_d_executable_c_blocked"
    assert profile_set.scientific_launch_status == "blocked"
    assert arm_b.request_graph == "independent_recurrent_modules_zero_memory_read_v1"
    assert arm_b.request_status == "executable"
    assert arm_b.teacher_status == "executable"
    assert arm_b.teacher_graph == ("independent_recurrent_modules_zero_memory_read_teacher_v1")
    assert arm_b.memory_model == "none"
    assert arm_b.trace_contract == "empty_workspace_trace"
    assert arm_b.blocker_codes == ()
    assert get_arm_execution_profile_v2("B").blocker_codes == ("checkpoint_component_ownership",)


def test_profile_set_v3_preserves_every_request_and_non_b_teacher_semantic() -> None:
    v2 = build_arm_execution_profile_set_v2()
    v3 = build_arm_execution_profile_set_v3()
    for older, newer in zip(v2.profiles, v3.profiles, strict=True):
        assert newer.arm_id == older.arm_id
        assert newer.arm_kind == older.arm_kind
        assert newer.request_status == older.request_status
        assert newer.request_graph == older.request_graph
        assert newer.memory_model == older.memory_model
        assert newer.trace_contract == older.trace_contract
        if newer.arm_id != "B":
            assert newer.teacher_status == older.teacher_status
            assert newer.teacher_graph == older.teacher_graph
            assert newer.blocker_codes == older.blocker_codes


def test_checked_in_profile_set_v3_is_exact_hash_bound_successor() -> None:
    profile_set = build_arm_execution_profile_set_v3()
    assert arm_execution_profile_set_sha256_v3() == _PROFILE_SET_SHA256_V3
    assert profile_set.predecessor_profile_set_sha256 == arm_execution_profile_set_sha256_v2()
    assert _CHECKED_IN_PROFILE_SET_V3.read_bytes() == canonical_json_file_bytes(
        profile_set.model_dump(mode="json")
    )


def test_get_profile_returns_an_owned_exact_model() -> None:
    first = get_arm_execution_profile_v1("A")
    second = get_arm_execution_profile_v1("A")

    assert first == second
    assert first is not second
    assert first.request_graph == "frozen_backbone_prompt_only_v1"
    assert first.teacher_status == "not_applicable"
    assert first.memory_model == "none"
    assert first.trace_contract == "empty_workspace_trace"
    assert first.blocker_codes == ()


@pytest.mark.parametrize("arm_id", [None, 1, True, "", "a", "E"])
def test_profile_lookup_rejects_noncanonical_arm_ids(arm_id: object) -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="exactly one of"):
        get_arm_execution_profile_v1(arm_id)


@pytest.mark.parametrize("arm_id", ["B", "C"])
def test_unresolved_profile_fails_before_runtime_inputs_are_touched(arm_id: str) -> None:
    explosive = cast(Any, object())

    with pytest.raises(
        IrisArmExecutionProfileError, match=f"arm {arm_id} request graph is blocked"
    ):
        build_arm_request_orchestrator_v1(
            arm_id=arm_id,
            backbone=explosive,
            stop_token_ids=explosive,
            sampling=explosive,
            cognitive_config=explosive,
            workspace_config=explosive,
        )


def test_v2_unresolved_arm_c_fails_before_runtime_inputs_are_touched() -> None:
    explosive = cast(Any, object())
    with pytest.raises(IrisArmExecutionProfileError, match="arm C request graph is blocked"):
        build_arm_request_orchestrator_v2(
            arm_id="C",
            backbone=explosive,
            stop_token_ids=explosive,
            sampling=explosive,
            cognitive_config=explosive,
            workspace_config=explosive,
        )


def test_v3_unresolved_arm_c_request_fails_before_runtime_inputs_are_touched() -> None:
    explosive = cast(Any, object())
    with pytest.raises(IrisArmExecutionProfileError, match="arm C request graph is blocked"):
        build_arm_request_orchestrator_v3(
            arm_id="C",
            backbone=explosive,
            stop_token_ids=explosive,
            sampling=explosive,
            cognitive_config=explosive,
            workspace_config=explosive,
        )


def test_blocked_profiles_report_distinct_missing_semantics() -> None:
    arm_b = get_arm_execution_profile_v1("B")
    arm_c = get_arm_execution_profile_v1("C")

    assert "module_output_to_backbone_bridge" in arm_b.blocker_codes
    assert "unlimited_buffer_capacity_and_growth" in arm_c.blocker_codes
    assert arm_b.blocker_codes != arm_c.blocker_codes
    with pytest.raises(IrisArmExecutionProfileError, match="module_output_to_backbone_bridge"):
        require_arm_request_executable_v1("B")


def test_arm_a_dispatch_rejects_sidecar_configuration() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="rejects cognitive"):
        build_arm_request_orchestrator_v1(
            arm_id="A",
            backbone=cast(Any, object()),
            stop_token_ids=(),
            sampling=cast(Any, object()),
            cognitive_config=cast(Any, object()),
        )


def test_arm_a_dispatches_only_to_prompt_only_builder() -> None:
    expected = cast(BaseLlmRequestOrchestrator, object())
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    with patch.object(
        profiles_service,
        "build_base_llm_request_orchestrator",
        return_value=expected,
    ) as build_a:
        observed = build_arm_request_orchestrator_v1(
            arm_id="A",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
        )

    assert observed is expected
    build_a.assert_called_once_with(
        backbone=backbone,
        stop_token_ids=(2,),
        sampling=sampling,
    )


def test_arm_d_requires_exact_sidecar_configs_before_dispatch() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="requires exact"):
        build_arm_request_orchestrator_v1(
            arm_id="D",
            backbone=cast(Any, object()),
            stop_token_ids=(2,),
            sampling=cast(Any, object()),
        )


def test_arm_d_dispatches_to_existing_competitive_graph() -> None:
    expected = cast(IrisRequestOrchestrator, object())
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    cognitive = cognitive_config()
    workspace = workspace_config()
    with patch.object(
        profiles_service,
        "build_request_orchestrator",
        return_value=expected,
    ) as build_d:
        observed = build_arm_request_orchestrator_v1(
            arm_id="D",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
            workspace_config=workspace,
        )

    assert observed is expected
    build_d.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        workspace_config=workspace,
        stop_token_ids=(2,),
        sampling=sampling,
    )


def test_arm_b_v2_requires_cognitive_config_and_rejects_workspace() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="requires exact cognitive"):
        build_arm_request_orchestrator_v2(
            arm_id="B",
            backbone=cast(Any, object()),
            stop_token_ids=(2,),
            sampling=cast(Any, object()),
        )
    with pytest.raises(IrisArmExecutionProfileError, match="rejects workspace"):
        build_arm_request_orchestrator_v2(
            arm_id="B",
            backbone=cast(Any, object()),
            stop_token_ids=(2,),
            sampling=cast(Any, object()),
            cognitive_config=cognitive_config(),
            workspace_config=workspace_config(),
        )


def test_arm_b_v2_dispatches_to_no_workspace_builder() -> None:
    expected = cast(CognitiveNoWorkspaceRequestOrchestrator, object())
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    cognitive = cognitive_config()
    with patch.object(
        profiles_service,
        "build_no_workspace_request_orchestrator",
        return_value=expected,
    ) as build_b:
        observed = build_arm_request_orchestrator_v2(
            arm_id="B",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
        )

    assert observed is expected
    build_b.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        stop_token_ids=(2,),
        sampling=sampling,
    )


def test_arm_b_v3_request_dispatch_is_identical_to_v2() -> None:
    expected = cast(CognitiveNoWorkspaceRequestOrchestrator, object())
    backbone = cast(Any, object())
    sampling = cast(Any, object())
    cognitive = cognitive_config()
    with patch.object(
        profiles_service,
        "build_no_workspace_request_orchestrator",
        return_value=expected,
    ) as build_b:
        observed = build_arm_request_orchestrator_v3(
            arm_id="B",
            backbone=backbone,
            stop_token_ids=(2,),
            sampling=sampling,
            cognitive_config=cognitive,
        )
    assert observed is expected
    build_b.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        stop_token_ids=(2,),
        sampling=sampling,
    )


@pytest.mark.parametrize(("arm_id", "match"), [("A", "no trainable"), ("C", "blocked")])
def test_v3_non_teacher_arms_fail_before_runtime_inspection(
    arm_id: str,
    match: str,
) -> None:
    explosive = cast(Any, object())
    with pytest.raises(IrisArmExecutionProfileError, match=match):
        build_arm_teacher_forcing_model_v3(
            arm_id=arm_id,
            backbone=explosive,
            cognitive_config=explosive,
            workspace_config=explosive,
        )


def test_arm_b_v3_teacher_requires_exact_cognitive_and_no_workspace() -> None:
    with pytest.raises(IrisArmExecutionProfileError, match="requires exact cognitive"):
        build_arm_teacher_forcing_model_v3(
            arm_id="B",
            backbone=cast(Any, object()),
        )
    with pytest.raises(IrisArmExecutionProfileError, match="no workspace"):
        build_arm_teacher_forcing_model_v3(
            arm_id="B",
            backbone=cast(Any, object()),
            cognitive_config=cognitive_config(),
            workspace_config=workspace_config(),
        )


def test_arm_b_v3_teacher_dispatches_only_to_no_workspace_builder() -> None:
    expected = cast(IrisNoWorkspaceTeacherForcingModel, object())
    backbone = cast(Any, object())
    cognitive = cognitive_config()
    with (
        patch.object(
            profiles_service,
            "build_no_workspace_teacher_forcing_model",
            return_value=expected,
        ) as build_b,
        patch.object(profiles_service, "build_teacher_forcing_model") as build_d,
    ):
        observed = build_arm_teacher_forcing_model_v3(
            arm_id="B",
            backbone=backbone,
            cognitive_config=cognitive,
        )
    assert observed is expected
    build_b.assert_called_once_with(backbone=backbone, cognitive_config=cognitive)
    build_d.assert_not_called()


def test_arm_d_v3_teacher_dispatches_only_to_competitive_builder() -> None:
    expected = cast(IrisTeacherForcingModel, object())
    backbone = cast(Any, object())
    cognitive = cognitive_config()
    workspace = workspace_config()
    with (
        patch.object(
            profiles_service,
            "build_teacher_forcing_model",
            return_value=expected,
        ) as build_d,
        patch.object(
            profiles_service,
            "build_no_workspace_teacher_forcing_model",
        ) as build_b,
    ):
        observed = build_arm_teacher_forcing_model_v3(
            arm_id="D",
            backbone=backbone,
            cognitive_config=cognitive,
            workspace_config=workspace,
        )
    assert observed is expected
    build_d.assert_called_once_with(
        backbone=backbone,
        cognitive_config=cognitive,
        workspace_config=workspace,
    )
    build_b.assert_not_called()


def test_arm_profile_api_is_public() -> None:
    assert application.build_arm_execution_profile_set_v1 is build_arm_execution_profile_set_v1
    assert application.build_arm_request_orchestrator_v1 is build_arm_request_orchestrator_v1
    assert application.get_arm_execution_profile_v1 is get_arm_execution_profile_v1
    assert application.build_arm_execution_profile_set_v2 is build_arm_execution_profile_set_v2
    assert application.build_arm_request_orchestrator_v2 is build_arm_request_orchestrator_v2
    assert application.get_arm_execution_profile_v2 is get_arm_execution_profile_v2
    assert application.require_arm_request_executable_v2 is require_arm_request_executable_v2
    assert application.build_arm_execution_profile_set_v3 is build_arm_execution_profile_set_v3
    assert application.build_arm_request_orchestrator_v3 is build_arm_request_orchestrator_v3
    assert application.build_arm_teacher_forcing_model_v3 is build_arm_teacher_forcing_model_v3
    assert application.get_arm_execution_profile_v3 is get_arm_execution_profile_v3
    assert application.require_arm_request_executable_v3 is require_arm_request_executable_v3
    assert application.require_arm_teacher_executable_v3 is require_arm_teacher_executable_v3
