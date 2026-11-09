from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import Mock

import pytest

import iris.application.arm_training_checkpoint as arm_checkpoint
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherCheckpointIdentity,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
    read_arm_teacher_checkpoint,
    restore_arm_teacher_checkpoint,
    validate_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
)
from iris.application.shared_buffer_training_checkpoint import (
    IrisSharedBufferTeacherCheckpointBundleV3,
)
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    IrisTeacherCheckpointRestoreResult,
    IrisTeacherCheckpointWriteResult,
)
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
)

_PAYLOAD_SHA256 = "1" * 64
_STATE_FILE_SHA256 = "2" * 64
_SIDECAR_STATE_SHA256 = "3" * 64
_COMPLETED_STEPS = 7
_STATE_BYTES = b"strict-arm-checkpoint-state"


def _schema_version(arm_id: IrisTeacherArmId) -> Literal[1, 2, 3]:
    if arm_id == "B":
        return 2
    if arm_id == "C":
        return 3
    return 1


def _raw_payload(
    arm_id: IrisTeacherArmId,
) -> (
    IrisNoWorkspaceTeacherCheckpointBundleV2
    | IrisSharedBufferTeacherCheckpointBundleV3
    | IrisTeacherCheckpointBundleV1
):
    manifest = SimpleNamespace(
        schema_identity=SimpleNamespace(version=_schema_version(arm_id)),
        sha256=_PAYLOAD_SHA256,
        payload=SimpleNamespace(
            state_file=SimpleNamespace(sha256=_STATE_FILE_SHA256),
            model=SimpleNamespace(sha256=_SIDECAR_STATE_SHA256),
            optimizer=SimpleNamespace(completed_steps=_COMPLETED_STEPS),
        ),
    )
    if arm_id == "B":
        return IrisNoWorkspaceTeacherCheckpointBundleV2(
            manifest=cast(Any, manifest),
            state_bytes=_STATE_BYTES,
        )
    if arm_id == "C":
        return IrisSharedBufferTeacherCheckpointBundleV3(
            manifest=cast(Any, manifest),
            state_bytes=_STATE_BYTES,
        )
    return IrisTeacherCheckpointBundleV1(
        manifest=cast(Any, manifest),
        state_bytes=_STATE_BYTES,
    )


def _operational_bundle(arm_id: IrisTeacherArmId) -> IrisArmTeacherCheckpointBundle:
    payload = _raw_payload(arm_id)
    return IrisArmTeacherCheckpointBundle(
        identity=IrisArmTeacherCheckpointIdentity(
            arm_id=arm_id,
            checkpoint_schema_version=_schema_version(arm_id),
            payload_sha256=_PAYLOAD_SHA256,
            state_file_sha256=_STATE_FILE_SHA256,
            sidecar_state_sha256=_SIDECAR_STATE_SHA256,
            completed_steps=_COMPLETED_STEPS,
        ),
        payload=payload,
    )


def _owner(
    arm_id: IrisTeacherArmId,
) -> tuple[
    IrisNoWorkspaceTeacherForcingModel
    | IrisSharedBufferTeacherForcingModel
    | IrisTeacherForcingModel,
    IrisTeacherOptimizer,
]:
    model: (
        IrisNoWorkspaceTeacherForcingModel
        | IrisSharedBufferTeacherForcingModel
        | IrisTeacherForcingModel
    )
    if arm_id == "B":
        model = object.__new__(IrisNoWorkspaceTeacherForcingModel)
    elif arm_id == "C":
        model = object.__new__(IrisSharedBufferTeacherForcingModel)
    else:
        model = object.__new__(IrisTeacherForcingModel)
    optimizer = object.__new__(IrisTeacherOptimizer)
    object.__setattr__(optimizer, "_model", model)
    return model, optimizer


@pytest.mark.parametrize(
    ("arm_id", "version"),
    [("B", 2), ("C", 3), ("D", 1)],
)
def test_arm_adapter_dispatches_every_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    version: int,
) -> None:
    payload = _raw_payload(arm_id)
    model, optimizer = _owner(arm_id)
    build_name = {
        "B": "build_no_workspace_teacher_checkpoint_v2",
        "C": "build_shared_buffer_teacher_checkpoint_v3",
        "D": "build_teacher_checkpoint_v1",
    }[arm_id]
    validate_name = {
        "B": "validate_no_workspace_teacher_checkpoint_bundle_v2",
        "C": "validate_shared_buffer_teacher_checkpoint_bundle_v3",
        "D": "validate_teacher_checkpoint_bundle_v1",
    }[arm_id]
    write_name = {
        "B": "write_no_workspace_teacher_checkpoint_v2",
        "C": "write_shared_buffer_teacher_checkpoint_v3",
        "D": "write_teacher_checkpoint_v1",
    }[arm_id]
    read_name = {
        "B": "read_no_workspace_teacher_checkpoint_v2",
        "C": "read_shared_buffer_teacher_checkpoint_v3",
        "D": "read_teacher_checkpoint_v1",
    }[arm_id]
    restore_name = {
        "B": "restore_no_workspace_teacher_checkpoint_v2",
        "C": "restore_shared_buffer_teacher_checkpoint_v3",
        "D": "restore_teacher_checkpoint_v1",
    }[arm_id]
    build_backend = Mock(return_value=payload)
    validate_backend = Mock(return_value=payload)
    expected_path = tmp_path / "teacher" / f"v{version}" / _PAYLOAD_SHA256
    write_backend = Mock(
        return_value=IrisTeacherCheckpointWriteResult(
            path=expected_path,
            payload_sha256=_PAYLOAD_SHA256,
            state_file_sha256=_STATE_FILE_SHA256,
            state_size_bytes=len(_STATE_BYTES),
            reused_existing=False,
        )
    )
    read_backend = Mock(return_value=payload)
    restore_backend = Mock(
        return_value=IrisTeacherCheckpointRestoreResult(
            payload_sha256=_PAYLOAD_SHA256,
            sidecar_state_sha256=_SIDECAR_STATE_SHA256,
            completed_steps=_COMPLETED_STEPS,
        )
    )
    monkeypatch.setattr(arm_checkpoint, build_name, build_backend)
    monkeypatch.setattr(arm_checkpoint, validate_name, validate_backend)
    monkeypatch.setattr(arm_checkpoint, write_name, write_backend)
    monkeypatch.setattr(arm_checkpoint, read_name, read_backend)
    monkeypatch.setattr(arm_checkpoint, restore_name, restore_backend)

    built = build_arm_teacher_checkpoint(
        arm_id=arm_id,
        model=model,
        optimizer=optimizer,
    )
    assert built.identity.arm_id == arm_id
    assert built.identity.checkpoint_schema_version == version
    assert built.identity.payload_sha256 == _PAYLOAD_SHA256
    assert built.identity.state_file_sha256 == _STATE_FILE_SHA256
    assert built.identity.sidecar_state_sha256 == _SIDECAR_STATE_SHA256
    assert built.identity.completed_steps == _COMPLETED_STEPS

    validated = validate_arm_teacher_checkpoint(built)
    written = write_arm_teacher_checkpoint(validated, root=tmp_path)
    assert written.identity == built.identity
    assert written.path == expected_path
    assert not written.reused_existing

    read = read_arm_teacher_checkpoint(built.identity, root=tmp_path)
    restored = restore_arm_teacher_checkpoint(
        read,
        model=model,
        optimizer=optimizer,
    )
    assert restored.payload_sha256 == _PAYLOAD_SHA256
    assert restored.sidecar_state_sha256 == _SIDECAR_STATE_SHA256
    assert restored.completed_steps == _COMPLETED_STEPS
    build_backend.assert_called_once()
    assert validate_backend.call_count == 3
    write_backend.assert_called_once()
    read_backend.assert_called_once_with(expected_path)
    restore_backend.assert_called_once()


@pytest.mark.parametrize(
    ("checkpoint_arm", "target_arm"),
    [("B", "C"), ("C", "D"), ("D", "B")],
)
def test_restore_rejects_cross_arm_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_arm: IrisTeacherArmId,
    target_arm: IrisTeacherArmId,
) -> None:
    bundle = _operational_bundle(checkpoint_arm)
    model, optimizer = _owner(target_arm)
    restore_name = {
        "B": "restore_no_workspace_teacher_checkpoint_v2",
        "C": "restore_shared_buffer_teacher_checkpoint_v3",
        "D": "restore_teacher_checkpoint_v1",
    }[checkpoint_arm]
    restore_backend = Mock()
    monkeypatch.setattr(arm_checkpoint, restore_name, restore_backend)

    with pytest.raises(TypeError, match="requires exact model type"):
        restore_arm_teacher_checkpoint(
            bundle,
            model=model,
            optimizer=optimizer,
        )

    restore_backend.assert_not_called()


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_build_rejects_model_subclass_and_wrong_optimizer_owner_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    arm_id: IrisTeacherArmId,
) -> None:
    model, _optimizer = _owner(arm_id)
    build_name = {
        "B": "build_no_workspace_teacher_checkpoint_v2",
        "C": "build_shared_buffer_teacher_checkpoint_v3",
        "D": "build_teacher_checkpoint_v1",
    }[arm_id]
    build_backend = Mock()
    monkeypatch.setattr(arm_checkpoint, build_name, build_backend)

    derived_model: (
        IrisNoWorkspaceTeacherForcingModel
        | IrisSharedBufferTeacherForcingModel
        | IrisTeacherForcingModel
    )
    if arm_id == "B":

        class DerivedNoWorkspaceModel(IrisNoWorkspaceTeacherForcingModel):
            pass

        derived_model = object.__new__(DerivedNoWorkspaceModel)

    elif arm_id == "C":

        class DerivedSharedBufferModel(IrisSharedBufferTeacherForcingModel):
            pass

        derived_model = object.__new__(DerivedSharedBufferModel)

    else:

        class DerivedWorkspaceModel(IrisTeacherForcingModel):
            pass

        derived_model = object.__new__(DerivedWorkspaceModel)
    derived_optimizer = object.__new__(IrisTeacherOptimizer)
    object.__setattr__(derived_optimizer, "_model", derived_model)
    with pytest.raises(TypeError, match="requires exact model type"):
        build_arm_teacher_checkpoint(
            arm_id=arm_id,
            model=derived_model,
            optimizer=derived_optimizer,
        )

    other_model, _ = _owner(arm_id)
    wrong_owner = object.__new__(IrisTeacherOptimizer)
    object.__setattr__(wrong_owner, "_model", other_model)
    with pytest.raises(TypeError, match="must exactly own"):
        build_arm_teacher_checkpoint(
            arm_id=arm_id,
            model=model,
            optimizer=wrong_owner,
        )

    build_backend.assert_not_called()


@pytest.mark.parametrize(
    ("arm_id", "wrong_payload_arm"),
    [("B", "C"), ("C", "D"), ("D", "B")],
)
def test_operational_bundle_rejects_cross_arm_payload_type(
    arm_id: IrisTeacherArmId,
    wrong_payload_arm: IrisTeacherArmId,
) -> None:
    identity = _operational_bundle(arm_id).identity

    with pytest.raises(TypeError, match="requires checkpoint payload type"):
        IrisArmTeacherCheckpointBundle(
            identity=identity,
            payload=_raw_payload(wrong_payload_arm),
        )


@pytest.mark.parametrize(
    ("arm_id", "version"),
    [("B", 2), ("C", 3), ("D", 1)],
)
def test_identity_is_frozen_and_owns_the_exact_expected_path(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    version: int,
) -> None:
    identity = _operational_bundle(arm_id).identity

    assert identity.expected_path(root=tmp_path) == (
        tmp_path / "teacher" / f"v{version}" / _PAYLOAD_SHA256
    )
    with pytest.raises(FrozenInstanceError):
        identity.completed_steps = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("arm_id", "version"),
    [("B", 1), ("C", 2), ("D", 3)],
)
def test_identity_rejects_cross_arm_schema_versions(
    arm_id: IrisTeacherArmId,
    version: Any,
) -> None:
    with pytest.raises(ValueError, match="requires checkpoint schema version"):
        IrisArmTeacherCheckpointIdentity(
            arm_id=arm_id,
            checkpoint_schema_version=version,
            payload_sha256=_PAYLOAD_SHA256,
            state_file_sha256=_STATE_FILE_SHA256,
            sidecar_state_sha256=_SIDECAR_STATE_SHA256,
            completed_steps=0,
        )
