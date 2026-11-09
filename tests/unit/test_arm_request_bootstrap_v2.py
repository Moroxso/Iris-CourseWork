"""Transactional tests for the A/B/D arm-labelled request bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch

import iris.application.bootstrap as bootstrap
from iris.application.arm_profiles import require_arm_request_executable_v3
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import (
    IrisArmRequestRuntimeV2,
    bootstrap_arm_request_runtime_v2,
)
from iris.application.errors import IrisBootstrapError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import IrisRequestOrchestrator
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
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

PROFILE_SHA256: Literal["5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"] = (
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
)


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
        ("D", IrisRequestOrchestrator, SIDECAR_STATE_SHA256),
    ],
)
def test_bootstrap_dispatches_exact_executable_arm_without_config_leakage(
    arm_id: str,
    orchestrator_type: type[object],
    sidecar: str | None,
) -> None:
    config = validated_config()
    backbone = OfflineBootstrapBackbone()
    orchestrator = object.__new__(orchestrator_type)

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", return_value=backbone),
        patch.object(
            bootstrap,
            "build_arm_request_orchestrator_v3",
            return_value=orchestrator,
        ) as build,
        patch.object(
            bootstrap,
            "_read_arm_sidecar_state_sha256",
            return_value=sidecar,
        ) as read_sidecar,
    ):
        runtime = bootstrap_arm_request_runtime_v2(
            arm_id,
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert type(runtime) is IrisArmRequestRuntimeV2
    assert runtime.profile.arm_id == arm_id
    assert runtime.execution_profile_set_sha256 == PROFILE_SHA256
    assert runtime.backbone is backbone
    assert runtime.orchestrator is orchestrator
    assert runtime.sidecar_state_sha256 == sidecar
    assert read_sidecar.call_count == 2
    kwargs = build.call_args.kwargs
    assert kwargs["arm_id"] == arm_id
    assert kwargs["backbone"] is backbone
    if arm_id == "A":
        assert kwargs["cognitive_config"] is None
        assert kwargs["workspace_config"] is None
    elif arm_id == "B":
        assert kwargs["cognitive_config"] == config.cognitive
        assert kwargs["cognitive_config"] is not config.cognitive
        assert kwargs["workspace_config"] is None
    else:
        assert kwargs["cognitive_config"] == config.cognitive
        assert kwargs["workspace_config"] == config.workspace
        assert kwargs["workspace_config"] is not config.workspace


def test_arm_c_is_rejected_before_config_device_or_backbone_inspection() -> None:
    with (
        patch.object(bootstrap, "_own_config") as own,
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "_create_backbone") as load,
        pytest.raises(IrisBootstrapError, match="execution profile") as raised,
    ):
        bootstrap_arm_request_runtime_v2(
            "C",
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert "arm C request graph is blocked" in str(raised.value.__cause__)
    own.assert_not_called()
    inspect.assert_not_called()
    load.assert_not_called()


def test_wrong_exact_orchestrator_type_rolls_back_committed_runtime_policy() -> None:
    before = capture_runtime_state(CPU)
    wrong_for_a = object.__new__(CognitiveNoWorkspaceRequestOrchestrator)
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(
            bootstrap,
            "build_arm_request_orchestrator_v3",
            return_value=wrong_for_a,
        ),
        pytest.raises(IrisBootstrapError, match="sidecar provenance capture"),
    ):
        bootstrap_arm_request_runtime_v2(
            "A",
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert_current_state_matches(before)


def test_arm_runtime_is_frozen_and_owns_the_profile_identity() -> None:
    backbone = cast(BackboneAdapter, OfflineBootstrapBackbone())
    orchestrator = object.__new__(BaseLlmRequestOrchestrator)
    config = validated_config()
    runtime = IrisArmRequestRuntimeV2(
        profile=require_arm_request_executable_v3("A"),
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
