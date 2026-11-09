"""Transactional checkpoint-aware profile-v4 request bootstrap tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

import iris.application as application
import iris.application.request_checkpoint_bootstrap as checkpoint_bootstrap
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointProjectionV1,
    IrisArmRequestCheckpointTransferResultV1,
)
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisBootstrapError
from iris.application.request_checkpoint_bootstrap import (
    IrisArmRequestBootstrapResultV4,
    bootstrap_arm_request_runtime_v4,
)
from iris.artifacts.checkpoint_v1 import TeacherBackboneRuntimeV1
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, IrisBootstrapConfig
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.unit.test_bootstrap import (
    CPU,
    CPU_INFO,
    STOP_TOKEN_IDS,
    OfflineBootstrapBackbone,
    assert_current_state_matches,
    validated_config,
)

_SOURCE_SHA256 = "a" * 64
_INITIAL_SHA256 = "b" * 64
_INSTALLED_SHA256 = "c" * 64
_STATE_NAMES = ("layer_sampler.projection.weight",)


@pytest.fixture(autouse=True)
def preserve_process_runtime_state() -> Iterator[None]:
    snapshot = capture_runtime_state(CPU)
    try:
        yield
    finally:
        restore_runtime_state(snapshot)


def _sampling() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)


def _identity() -> IrisArmTeacherCheckpointIdentity:
    return IrisArmTeacherCheckpointIdentity(
        arm_id="B",
        checkpoint_schema_version=2,
        payload_sha256="d" * 64,
        state_file_sha256="e" * 64,
        sidecar_state_sha256=_SOURCE_SHA256,
        completed_steps=5,
    )


def _projection(
    config: IrisBootstrapConfig,
    identity: IrisArmTeacherCheckpointIdentity,
) -> IrisArmRequestCheckpointProjectionV1:
    return IrisArmRequestCheckpointProjectionV1(
        checkpoint_identity=identity,
        backbone=config.backbone,
        cognitive=config.cognitive,
        workspace=None,
        runtime=TeacherBackboneRuntimeV1(
            hidden_size=8,
            num_hidden_layers=4,
            vocabulary_size=32,
            max_position_embeddings=1024,
            max_prompt_tokens=config.backbone.limits.max_prompt_tokens,
            max_target_tokens=config.backbone.limits.max_new_tokens,
            device="cpu",
            dtype="float32",
        ),
        source_teacher_sidecar_sha256=_SOURCE_SHA256,
        request_sidecar_state_sha256=_INSTALLED_SHA256,
        completed_steps=5,
        state_names=_STATE_NAMES,
        tensor_count=1,
    )


def _successful_transfer(
    *,
    identity: IrisArmTeacherCheckpointIdentity,
    orchestrator: MagicMock,
) -> Callable[..., IrisArmRequestCheckpointTransferResultV1]:
    def transfer(**kwargs: object) -> IrisArmRequestCheckpointTransferResultV1:
        runtime = kwargs["runtime"]
        if type(runtime) is not IrisArmRequestRuntimeV3:
            raise AssertionError("bootstrap must transfer an exact provisional runtime")
        orchestrator.sidecar_state_sha256 = _INSTALLED_SHA256
        updated = replace(runtime, sidecar_state_sha256=_INSTALLED_SHA256)
        return IrisArmRequestCheckpointTransferResultV1(
            runtime=updated,
            checkpoint_identity=identity,
            transfer_contract="exact_teacher_state_to_profile_v4_request_sidecar_v1",
            source_teacher_sidecar_sha256=_SOURCE_SHA256,
            previous_request_sidecar_sha256=_INITIAL_SHA256,
            installed_request_sidecar_sha256=_INSTALLED_SHA256,
            completed_steps=5,
            state_names=_STATE_NAMES,
            transferred_tensor_count=1,
        )

    return transfer


def test_checkpoint_is_read_before_backbone_and_only_transferred_runtime_is_published() -> None:
    config = validated_config()
    identity = _identity()
    projection = _projection(config, identity)
    checkpoint = object()
    orchestrator = MagicMock()
    call_order: list[str] = []

    def read(*args: object, **kwargs: object) -> object:
        del args, kwargs
        call_order.append("checkpoint")
        return checkpoint

    def create(*args: object, **kwargs: object) -> OfflineBootstrapBackbone:
        del args, kwargs
        call_order.append("backbone")
        return OfflineBootstrapBackbone()

    with (
        patch.object(
            checkpoint_bootstrap,
            "read_arm_teacher_checkpoint",
            side_effect=read,
        ) as read_call,
        patch.object(
            checkpoint_bootstrap,
            "project_arm_teacher_checkpoint_to_request_v1",
            return_value=projection,
        ) as project,
        patch.object(
            checkpoint_bootstrap,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(checkpoint_bootstrap, "_create_backbone", side_effect=create),
        patch.object(
            checkpoint_bootstrap,
            "build_arm_request_orchestrator_v4",
            return_value=orchestrator,
        ),
        patch.object(
            checkpoint_bootstrap,
            "_read_arm_sidecar_state_sha256_v4",
            side_effect=(_INITIAL_SHA256, _INSTALLED_SHA256),
        ),
        patch.object(
            checkpoint_bootstrap,
            "transfer_arm_teacher_checkpoint_to_request_v1",
            side_effect=_successful_transfer(identity=identity, orchestrator=orchestrator),
        ) as transfer,
    ):
        result = bootstrap_arm_request_runtime_v4(
            "B",
            config,
            checkpoint_identity=identity,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert type(result) is IrisArmRequestBootstrapResultV4
    assert type(result.checkpoint_transfer) is IrisArmRequestCheckpointTransferResultV1
    assert result.checkpoint_transfer.runtime is result.runtime
    assert result.runtime.sidecar_state_sha256 == _INSTALLED_SHA256
    assert call_order == ["checkpoint", "backbone", "checkpoint"]
    assert read_call.call_count == 2
    assert project.call_count == 2
    transfer.assert_called_once()


@pytest.mark.parametrize(
    ("arm_id", "identity"),
    [
        ("A", _identity()),
        ("B", None),
        (
            "B",
            IrisArmTeacherCheckpointIdentity(
                arm_id="C",
                checkpoint_schema_version=3,
                payload_sha256="d" * 64,
                state_file_sha256="e" * 64,
                sidecar_state_sha256="f" * 64,
                completed_steps=0,
            ),
        ),
    ],
)
def test_checkpoint_identity_policy_fails_before_configuration_or_model_work(
    arm_id: str,
    identity: IrisArmTeacherCheckpointIdentity | None,
) -> None:
    with (
        patch.object(checkpoint_bootstrap, "_own_config") as own,
        patch.object(checkpoint_bootstrap, "read_arm_teacher_checkpoint") as read,
        patch.object(checkpoint_bootstrap, "_create_backbone") as create,
        pytest.raises(IrisBootstrapError, match="profile or checkpoint identity"),
    ):
        bootstrap_arm_request_runtime_v4(
            arm_id,
            validated_config(),
            checkpoint_identity=identity,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    own.assert_not_called()
    read.assert_not_called()
    create.assert_not_called()


def test_checkpoint_config_mismatch_fails_before_device_or_backbone_work() -> None:
    config = validated_config()
    identity = _identity()
    projection = _projection(config, identity)
    cognitive_values = config.cognitive.model_dump(mode="python")
    cognitive_values["name"] = "mismatched-cognitive-config"
    changed = replace(
        projection,
        cognitive=CognitiveConfig.model_validate(cognitive_values),
    )
    with (
        patch.object(checkpoint_bootstrap, "read_arm_teacher_checkpoint", return_value=object()),
        patch.object(
            checkpoint_bootstrap,
            "project_arm_teacher_checkpoint_to_request_v1",
            return_value=changed,
        ),
        patch.object(checkpoint_bootstrap, "_resolve_device") as resolve,
        patch.object(checkpoint_bootstrap, "_create_backbone") as create,
        pytest.raises(IrisBootstrapError, match="configuration compatibility") as raised,
    ):
        bootstrap_arm_request_runtime_v4(
            "B",
            config,
            checkpoint_identity=identity,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert "model configuration differs" in str(raised.value.__cause__)
    resolve.assert_not_called()
    create.assert_not_called()


def test_transfer_failure_restores_process_runtime_and_does_not_reread_checkpoint() -> None:
    before = capture_runtime_state(CPU)
    config = validated_config(seed=9001)
    identity = _identity()
    projection = _projection(config, identity)
    failure = RuntimeError("simulated transfer failure")
    with (
        patch.object(
            checkpoint_bootstrap,
            "read_arm_teacher_checkpoint",
            return_value=object(),
        ) as read,
        patch.object(
            checkpoint_bootstrap,
            "project_arm_teacher_checkpoint_to_request_v1",
            return_value=projection,
        ),
        patch.object(
            checkpoint_bootstrap,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(
            checkpoint_bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(
            checkpoint_bootstrap,
            "build_arm_request_orchestrator_v4",
            return_value=MagicMock(),
        ),
        patch.object(
            checkpoint_bootstrap,
            "_read_arm_sidecar_state_sha256_v4",
            return_value=_INITIAL_SHA256,
        ),
        patch.object(
            checkpoint_bootstrap,
            "transfer_arm_teacher_checkpoint_to_request_v1",
            side_effect=failure,
        ),
        pytest.raises(IrisBootstrapError, match="teacher checkpoint transfer") as raised,
    ):
        bootstrap_arm_request_runtime_v4(
            "B",
            config,
            checkpoint_identity=identity,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert raised.value.__cause__ is failure
    assert read.call_count == 1
    assert_current_state_matches(before)


def test_arm_a_keeps_evaluation_parity_without_checkpoint_access() -> None:
    config = validated_config()
    orchestrator = MagicMock()
    with (
        patch.object(checkpoint_bootstrap, "read_arm_teacher_checkpoint") as read,
        patch.object(
            checkpoint_bootstrap,
            "project_arm_teacher_checkpoint_to_request_v1",
        ) as project,
        patch.object(
            checkpoint_bootstrap,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(
            checkpoint_bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(
            checkpoint_bootstrap,
            "build_arm_request_orchestrator_v4",
            return_value=orchestrator,
        ),
        patch.object(
            checkpoint_bootstrap,
            "_read_arm_sidecar_state_sha256_v4",
            side_effect=(None, None),
        ),
        patch.object(
            checkpoint_bootstrap,
            "transfer_arm_teacher_checkpoint_to_request_v1",
        ) as transfer,
    ):
        result = bootstrap_arm_request_runtime_v4(
            "A",
            config,
            checkpoint_identity=None,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=_sampling(),
        )

    assert result.runtime.profile.arm_id == "A"
    assert result.runtime.sidecar_state_sha256 is None
    assert result.checkpoint_transfer is None
    read.assert_not_called()
    project.assert_not_called()
    transfer.assert_not_called()


def test_checkpoint_bound_bootstrap_is_exported() -> None:
    assert application.IrisArmRequestBootstrapResultV4 is IrisArmRequestBootstrapResultV4
    assert application.bootstrap_arm_request_runtime_v4 is bootstrap_arm_request_runtime_v4
