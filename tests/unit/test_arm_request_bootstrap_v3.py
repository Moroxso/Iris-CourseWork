"""Transactional profile-v4 request bootstrap tests for all four study arms."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import Literal, cast
from unittest.mock import PropertyMock, patch

import pytest
import torch

import iris.application as application
import iris.application.bootstrap as bootstrap
from iris.application.arm_profiles import require_arm_request_executable_v4
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import (
    IrisArmRequestRuntimeV3,
    bootstrap_arm_request_runtime_v3,
)
from iris.application.errors import IrisBootstrapError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import IrisRequestOrchestrator
from iris.application.shared_buffer_request import CognitiveSharedBufferRequestOrchestrator
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.unit.test_bootstrap import (
    CPU,
    CPU_INFO,
    SIDECAR_STATE_SHA256,
    STOP_TOKEN_IDS,
    OfflineBootstrapBackbone,
    assert_current_state_matches,
    validated_config,
)

PROFILE_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"


@pytest.fixture(autouse=True)
def preserve_process_runtime_state() -> Iterator[None]:
    snapshot = capture_runtime_state(CPU)
    try:
        yield
    finally:
        restore_runtime_state(snapshot)


def _sampling() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)


@pytest.mark.parametrize(
    ("arm_id", "orchestrator_type", "sidecar"),
    [
        ("A", BaseLlmRequestOrchestrator, None),
        ("B", CognitiveNoWorkspaceRequestOrchestrator, SIDECAR_STATE_SHA256),
        ("C", CognitiveSharedBufferRequestOrchestrator, SIDECAR_STATE_SHA256),
        ("D", IrisRequestOrchestrator, SIDECAR_STATE_SHA256),
    ],
)
def test_bootstrap_dispatches_all_exact_v4_graphs_without_configuration_leakage(
    arm_id: str,
    orchestrator_type: type[object],
    sidecar: str | None,
) -> None:
    config = validated_config()
    backbone = OfflineBootstrapBackbone()
    orchestrator = object.__new__(orchestrator_type)

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", return_value=backbone) as load,
        patch.object(
            bootstrap,
            "build_arm_request_orchestrator_v4",
            return_value=orchestrator,
        ) as build,
        patch.object(
            bootstrap,
            "_read_arm_sidecar_state_sha256_v4",
            return_value=sidecar,
        ) as read_sidecar,
    ):
        runtime = bootstrap_arm_request_runtime_v3(
            arm_id,
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert type(runtime) is IrisArmRequestRuntimeV3
    assert runtime.profile.arm_id == arm_id
    assert runtime.execution_profile_set_sha256 == PROFILE_SHA256
    assert runtime.backbone is backbone
    assert runtime.orchestrator is orchestrator
    assert runtime.sidecar_state_sha256 == sidecar
    load.assert_called_once()
    assert read_sidecar.call_count == 2
    kwargs = build.call_args.kwargs
    assert kwargs["arm_id"] == arm_id
    assert kwargs["backbone"] is backbone
    if arm_id == "A":
        assert kwargs["cognitive_config"] is None
        assert kwargs["workspace_config"] is None
    else:
        assert type(kwargs["cognitive_config"]) is CognitiveConfig
        assert kwargs["cognitive_config"] == config.cognitive
        assert kwargs["cognitive_config"] is not config.cognitive
        if arm_id == "D":
            assert type(kwargs["workspace_config"]) is WorkspaceConfig
            assert kwargs["workspace_config"] == config.workspace
            assert kwargs["workspace_config"] is not config.workspace
        else:
            assert kwargs["workspace_config"] is None


@pytest.mark.parametrize(
    ("target", "drifted", "cause_match"),
    [
        (
            "arm_execution_profile_set_sha256_v4",
            "0" * 64,
            "profile set v4 identity",
        ),
        (
            "shared_buffer_policy_sha256_v1",
            "0" * 64,
            "shared-buffer policy v1 identity",
        ),
    ],
)
def test_identity_drift_fails_closed_before_config_device_or_backbone_inspection(
    target: str,
    drifted: str,
    cause_match: str,
) -> None:
    with (
        patch.object(bootstrap, target, return_value=drifted),
        patch.object(bootstrap, "_own_config") as own,
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "_create_backbone") as load,
        pytest.raises(IrisBootstrapError, match="execution profile preflight") as raised,
    ):
        bootstrap_arm_request_runtime_v3(
            "C",
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert cause_match in str(raised.value.__cause__)
    own.assert_not_called()
    inspect.assert_not_called()
    load.assert_not_called()


def test_invalid_arm_fails_closed_before_config_device_or_backbone_inspection() -> None:
    with (
        patch.object(bootstrap, "_own_config") as own,
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "_create_backbone") as load,
        pytest.raises(IrisBootstrapError, match="execution profile preflight"),
    ):
        bootstrap_arm_request_runtime_v3(
            "E",
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    own.assert_not_called()
    inspect.assert_not_called()
    load.assert_not_called()


def test_arm_c_sidecar_reader_requires_exact_graph_backbone_and_canonical_policy() -> None:
    profile = require_arm_request_executable_v4("C")
    backbone = cast(BackboneAdapter, OfflineBootstrapBackbone())
    orchestrator = object.__new__(CognitiveSharedBufferRequestOrchestrator)

    with (
        patch.object(
            CognitiveSharedBufferRequestOrchestrator,
            "backbone",
            new_callable=PropertyMock,
            return_value=backbone,
        ),
        patch.object(
            CognitiveSharedBufferRequestOrchestrator,
            "shared_buffer_policy_sha256",
            new_callable=PropertyMock,
            return_value=shared_buffer_policy_sha256_v1(),
        ),
        patch.object(
            CognitiveSharedBufferRequestOrchestrator,
            "sidecar_state_sha256",
            new_callable=PropertyMock,
            return_value=SIDECAR_STATE_SHA256,
        ),
    ):
        digest = bootstrap._read_arm_sidecar_state_sha256_v4(
            profile=profile,
            orchestrator=orchestrator,
            backbone=backbone,
        )

    assert digest == SIDECAR_STATE_SHA256


def test_arm_c_sidecar_reader_rejects_noncanonical_policy() -> None:
    profile = require_arm_request_executable_v4("C")
    backbone = cast(BackboneAdapter, OfflineBootstrapBackbone())
    orchestrator = object.__new__(CognitiveSharedBufferRequestOrchestrator)

    with (
        patch.object(
            CognitiveSharedBufferRequestOrchestrator,
            "backbone",
            new_callable=PropertyMock,
            return_value=backbone,
        ),
        patch.object(
            CognitiveSharedBufferRequestOrchestrator,
            "shared_buffer_policy_sha256",
            new_callable=PropertyMock,
            return_value="0" * 64,
        ),
        pytest.raises(ValueError, match="non-canonical shared-buffer policy"),
    ):
        bootstrap._read_arm_sidecar_state_sha256_v4(
            profile=profile,
            orchestrator=orchestrator,
            backbone=backbone,
        )


def test_sidecar_drift_rolls_back_committed_global_runtime_state() -> None:
    before = capture_runtime_state(CPU)
    orchestrator = object.__new__(CognitiveSharedBufferRequestOrchestrator)
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(
            bootstrap,
            "build_arm_request_orchestrator_v4",
            return_value=orchestrator,
        ),
        patch.object(
            bootstrap,
            "_read_arm_sidecar_state_sha256_v4",
            side_effect=(SIDECAR_STATE_SHA256, "d" * 64),
        ),
        pytest.raises(IrisBootstrapError, match="final arm-labelled provenance validation"),
    ):
        bootstrap_arm_request_runtime_v3(
            "C",
            validated_config(seed=9001),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert_current_state_matches(before)


def test_composition_failure_rolls_back_and_does_not_retry_backbone_load() -> None:
    before = capture_runtime_state(CPU)
    failure = RuntimeError("arm-C composition exploded")
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ) as load,
        patch.object(
            bootstrap,
            "build_arm_request_orchestrator_v4",
            side_effect=failure,
        ) as build,
        pytest.raises(IrisBootstrapError, match="request-graph v4 composition") as raised,
    ):
        bootstrap_arm_request_runtime_v3(
            "C",
            validated_config(seed=9001),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert raised.value.__cause__ is failure
    load.assert_called_once()
    build.assert_called_once()
    assert_current_state_matches(before)


def test_runtime_is_frozen_and_owns_profile_v4_identity() -> None:
    backbone = cast(BackboneAdapter, OfflineBootstrapBackbone())
    orchestrator = object.__new__(BaseLlmRequestOrchestrator)
    config = validated_config()
    runtime = IrisArmRequestRuntimeV3(
        profile=require_arm_request_executable_v4("A"),
        execution_profile_set_sha256=PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=CPU_INFO,
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=True,
        configuration_json=bootstrap._canonical_config_json(config),
        configuration_sha256="a" * 64,
        sidecar_state_sha256=None,
    )

    with pytest.raises(FrozenInstanceError):
        runtime.sidecar_state_sha256 = SIDECAR_STATE_SHA256  # type: ignore[misc]


def test_runtime_v3_is_exported_from_application_package() -> None:
    assert application.IrisArmRequestRuntimeV3 is IrisArmRequestRuntimeV3
    assert application.bootstrap_arm_request_runtime_v3 is bootstrap_arm_request_runtime_v3
