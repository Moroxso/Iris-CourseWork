"""Strict arm-aware dispatch over the versioned teacher checkpoint formats.

This module is an application boundary, not a persisted artifact schema.  It
keeps the arm discriminator next to the exact v1/v2/v3 checkpoint identity while
delegating all serialization and state mutation to the format owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
    build_no_workspace_teacher_checkpoint_v2,
    read_no_workspace_teacher_checkpoint_v2,
    restore_no_workspace_teacher_checkpoint_v2,
    validate_no_workspace_teacher_checkpoint_bundle_v2,
    write_no_workspace_teacher_checkpoint_v2,
)
from iris.application.shared_buffer_training_checkpoint import (
    IrisSharedBufferTeacherCheckpointBundleV3,
    build_shared_buffer_teacher_checkpoint_v3,
    read_shared_buffer_teacher_checkpoint_v3,
    restore_shared_buffer_teacher_checkpoint_v3,
    validate_shared_buffer_teacher_checkpoint_bundle_v3,
    write_shared_buffer_teacher_checkpoint_v3,
)
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    IrisTeacherCheckpointRestoreResult,
    IrisTeacherCheckpointWriteResult,
    build_teacher_checkpoint_v1,
    read_teacher_checkpoint_v1,
    restore_teacher_checkpoint_v1,
    validate_teacher_checkpoint_bundle_v1,
    write_teacher_checkpoint_v1,
)
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
)

type IrisTeacherArmId = Literal["B", "C", "D"]
type IrisArmTeacherModel = (
    IrisNoWorkspaceTeacherForcingModel
    | IrisSharedBufferTeacherForcingModel
    | IrisTeacherForcingModel
)
type IrisArmTeacherCheckpointPayload = (
    IrisNoWorkspaceTeacherCheckpointBundleV2
    | IrisSharedBufferTeacherCheckpointBundleV3
    | IrisTeacherCheckpointBundleV1
)

_MAX_EXACT_ADAMW_STEPS = 2**24
_LOWERCASE_HEX = frozenset("0123456789abcdef")


def _require_arm_id(value: object) -> IrisTeacherArmId:
    if type(value) is not str or value not in ("B", "C", "D"):
        message = "teacher checkpoint arm_id must be exactly 'B', 'C', or 'D'"
        raise ValueError(message)
    return cast(IrisTeacherArmId, value)


def _schema_version_for_arm(arm_id: IrisTeacherArmId) -> Literal[1, 2, 3]:
    if arm_id == "B":
        return 2
    if arm_id == "C":
        return 3
    return 1


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        message = f"{field} must be exactly 64 lowercase hexadecimal characters"
        raise ValueError(message)


def _require_root(root: object) -> Path:
    if not isinstance(root, Path) or not root.name:
        message = "arm checkpoint operation requires an explicit root Path"
        raise TypeError(message)
    return root


@dataclass(frozen=True, slots=True)
class IrisArmTeacherCheckpointIdentity:
    """Immutable operational identity shared by the B/C/D checkpoint adapters."""

    arm_id: IrisTeacherArmId
    checkpoint_schema_version: Literal[1, 2, 3]
    payload_sha256: str
    state_file_sha256: str
    sidecar_state_sha256: str
    completed_steps: int

    def __post_init__(self) -> None:
        arm_id = _require_arm_id(self.arm_id)
        expected_version = _schema_version_for_arm(arm_id)
        if (
            type(self.checkpoint_schema_version) is not int
            or self.checkpoint_schema_version != expected_version
        ):
            message = f"arm {arm_id} requires checkpoint schema version {expected_version}"
            raise ValueError(message)
        _require_sha256(self.payload_sha256, field="payload_sha256")
        _require_sha256(self.state_file_sha256, field="state_file_sha256")
        _require_sha256(self.sidecar_state_sha256, field="sidecar_state_sha256")
        if (
            type(self.completed_steps) is not int
            or self.completed_steps < 0
            or self.completed_steps > _MAX_EXACT_ADAMW_STEPS
        ):
            message = "completed_steps must be an exact integer in [0, 2**24]"
            raise ValueError(message)

    def expected_path(self, *, root: Path) -> Path:
        """Return the sole content-addressed path permitted for this identity."""
        checked_root = _require_root(root)
        return checked_root / "teacher" / f"v{self.checkpoint_schema_version}" / self.payload_sha256


def _require_payload_type(
    arm_id: IrisTeacherArmId,
    payload: object,
) -> IrisArmTeacherCheckpointPayload:
    expected_type: type[object]
    if arm_id == "B":
        expected_type = IrisNoWorkspaceTeacherCheckpointBundleV2
    elif arm_id == "C":
        expected_type = IrisSharedBufferTeacherCheckpointBundleV3
    else:
        expected_type = IrisTeacherCheckpointBundleV1
    if type(payload) is not expected_type:
        message = f"arm {arm_id} requires checkpoint payload type {expected_type.__name__}"
        raise TypeError(message)
    return payload


def _identity_from_payload(
    arm_id: IrisTeacherArmId,
    payload: IrisArmTeacherCheckpointPayload,
) -> IrisArmTeacherCheckpointIdentity:
    checked_payload = _require_payload_type(arm_id, payload)
    if arm_id == "B":
        manifest_b = cast(
            IrisNoWorkspaceTeacherCheckpointBundleV2,
            checked_payload,
        ).manifest
        return IrisArmTeacherCheckpointIdentity(
            arm_id=arm_id,
            checkpoint_schema_version=manifest_b.schema_identity.version,
            payload_sha256=manifest_b.sha256,
            state_file_sha256=manifest_b.payload.state_file.sha256,
            sidecar_state_sha256=manifest_b.payload.model.sha256,
            completed_steps=manifest_b.payload.optimizer.completed_steps,
        )
    if arm_id == "C":
        manifest_c = cast(
            IrisSharedBufferTeacherCheckpointBundleV3,
            checked_payload,
        ).manifest
        return IrisArmTeacherCheckpointIdentity(
            arm_id=arm_id,
            checkpoint_schema_version=manifest_c.schema_identity.version,
            payload_sha256=manifest_c.sha256,
            state_file_sha256=manifest_c.payload.state_file.sha256,
            sidecar_state_sha256=manifest_c.payload.model.sha256,
            completed_steps=manifest_c.payload.optimizer.completed_steps,
        )
    manifest_d = cast(IrisTeacherCheckpointBundleV1, checked_payload).manifest
    return IrisArmTeacherCheckpointIdentity(
        arm_id=arm_id,
        checkpoint_schema_version=manifest_d.schema_identity.version,
        payload_sha256=manifest_d.sha256,
        state_file_sha256=manifest_d.payload.state_file.sha256,
        sidecar_state_sha256=manifest_d.payload.model.sha256,
        completed_steps=manifest_d.payload.optimizer.completed_steps,
    )


@dataclass(frozen=True, slots=True)
class IrisArmTeacherCheckpointBundle:
    """An arm discriminator paired with one exact existing checkpoint bundle."""

    identity: IrisArmTeacherCheckpointIdentity
    payload: IrisArmTeacherCheckpointPayload

    def __post_init__(self) -> None:
        if type(self.identity) is not IrisArmTeacherCheckpointIdentity:
            message = "identity must be exactly IrisArmTeacherCheckpointIdentity"
            raise TypeError(message)
        actual = _identity_from_payload(self.identity.arm_id, self.payload)
        if actual != self.identity:
            message = "checkpoint payload identity disagrees with its arm identity"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class IrisArmTeacherCheckpointWriteResult:
    """Verified filesystem result for one arm-aware checkpoint installation."""

    identity: IrisArmTeacherCheckpointIdentity
    path: Path
    state_size_bytes: int
    reused_existing: bool

    def __post_init__(self) -> None:
        if type(self.identity) is not IrisArmTeacherCheckpointIdentity:
            message = "identity must be exactly IrisArmTeacherCheckpointIdentity"
            raise TypeError(message)
        if not isinstance(self.path, Path) or not self.path.name:
            message = "checkpoint write result path must be an explicit Path"
            raise TypeError(message)
        expected_suffix = (
            "teacher",
            f"v{self.identity.checkpoint_schema_version}",
            self.identity.payload_sha256,
        )
        actual_suffix = (
            self.path.parent.parent.name,
            self.path.parent.name,
            self.path.name,
        )
        if actual_suffix != expected_suffix:
            message = "checkpoint write result violates the arm identity path"
            raise ValueError(message)
        if type(self.state_size_bytes) is not int or self.state_size_bytes <= 0:
            message = "state_size_bytes must be an exact positive integer"
            raise ValueError(message)
        if type(self.reused_existing) is not bool:
            message = "reused_existing must be exactly bool"
            raise TypeError(message)


def _require_bundle(value: object) -> IrisArmTeacherCheckpointBundle:
    if type(value) is not IrisArmTeacherCheckpointBundle:
        message = "checkpoint must be exactly IrisArmTeacherCheckpointBundle"
        raise TypeError(message)
    bundle = value
    if type(bundle.identity) is not IrisArmTeacherCheckpointIdentity:
        message = "checkpoint identity must be exact before dispatch"
        raise TypeError(message)
    _require_payload_type(bundle.identity.arm_id, bundle.payload)
    actual = _identity_from_payload(bundle.identity.arm_id, bundle.payload)
    if actual != bundle.identity:
        message = "checkpoint payload identity changed before dispatch"
        raise ValueError(message)
    return bundle


def _require_owner(
    arm_id: IrisTeacherArmId,
    model: object,
    optimizer: object,
) -> tuple[IrisArmTeacherModel, IrisTeacherOptimizer]:
    expected_model_type: type[object]
    if arm_id == "B":
        expected_model_type = IrisNoWorkspaceTeacherForcingModel
    elif arm_id == "C":
        expected_model_type = IrisSharedBufferTeacherForcingModel
    else:
        expected_model_type = IrisTeacherForcingModel
    if type(model) is not expected_model_type:
        message = f"arm {arm_id} requires exact model type {expected_model_type.__name__}"
        raise TypeError(message)
    if type(optimizer) is not IrisTeacherOptimizer:
        message = "arm checkpoint optimizer must be exactly IrisTeacherOptimizer"
        raise TypeError(message)
    checked_optimizer = optimizer
    if checked_optimizer.model is not model:
        message = "arm checkpoint optimizer must exactly own the supplied model"
        raise TypeError(message)
    return model, checked_optimizer


def _bundle_from_payload(
    arm_id: IrisTeacherArmId,
    payload: IrisArmTeacherCheckpointPayload,
) -> IrisArmTeacherCheckpointBundle:
    checked_payload = _require_payload_type(arm_id, payload)
    return IrisArmTeacherCheckpointBundle(
        identity=_identity_from_payload(arm_id, checked_payload),
        payload=checked_payload,
    )


def build_arm_teacher_checkpoint(
    *,
    arm_id: IrisTeacherArmId,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisArmTeacherCheckpointBundle:
    """Build a verified B/v2, C/v3, or D/v1 bundle using its exact owner."""
    checked_arm_id = _require_arm_id(arm_id)
    checked_model, checked_optimizer = _require_owner(
        checked_arm_id,
        model,
        optimizer,
    )
    payload: IrisArmTeacherCheckpointPayload
    if checked_arm_id == "B":
        payload = build_no_workspace_teacher_checkpoint_v2(
            model=cast(IrisNoWorkspaceTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    elif checked_arm_id == "C":
        payload = build_shared_buffer_teacher_checkpoint_v3(
            model=cast(IrisSharedBufferTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    else:
        payload = build_teacher_checkpoint_v1(
            model=cast(IrisTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    return _bundle_from_payload(checked_arm_id, payload)


def validate_arm_teacher_checkpoint(
    bundle: IrisArmTeacherCheckpointBundle,
) -> IrisArmTeacherCheckpointBundle:
    """Fully validate an operational bundle without changing either format."""
    checked = _require_bundle(bundle)
    payload: IrisArmTeacherCheckpointPayload
    if checked.identity.arm_id == "B":
        payload = validate_no_workspace_teacher_checkpoint_bundle_v2(
            cast(IrisNoWorkspaceTeacherCheckpointBundleV2, checked.payload)
        )
    elif checked.identity.arm_id == "C":
        payload = validate_shared_buffer_teacher_checkpoint_bundle_v3(
            cast(IrisSharedBufferTeacherCheckpointBundleV3, checked.payload)
        )
    else:
        payload = validate_teacher_checkpoint_bundle_v1(
            cast(IrisTeacherCheckpointBundleV1, checked.payload)
        )
    validated = _bundle_from_payload(checked.identity.arm_id, payload)
    if validated.identity != checked.identity:
        message = "validated checkpoint identity differs from the supplied identity"
        raise ValueError(message)
    return validated


def write_arm_teacher_checkpoint(
    bundle: IrisArmTeacherCheckpointBundle,
    *,
    root: Path,
) -> IrisArmTeacherCheckpointWriteResult:
    """Validate then atomically install an arm checkpoint at its exact path."""
    checked_root = _require_root(root)
    validated = validate_arm_teacher_checkpoint(bundle)
    if validated.identity.arm_id == "B":
        raw_result = write_no_workspace_teacher_checkpoint_v2(
            cast(IrisNoWorkspaceTeacherCheckpointBundleV2, validated.payload),
            root=checked_root,
        )
    elif validated.identity.arm_id == "C":
        raw_result = write_shared_buffer_teacher_checkpoint_v3(
            cast(IrisSharedBufferTeacherCheckpointBundleV3, validated.payload),
            root=checked_root,
        )
    else:
        raw_result = write_teacher_checkpoint_v1(
            cast(IrisTeacherCheckpointBundleV1, validated.payload),
            root=checked_root,
        )
    if type(raw_result) is not IrisTeacherCheckpointWriteResult:
        message = "checkpoint writer returned an unexpected result type"
        raise TypeError(message)
    expected_path = validated.identity.expected_path(root=checked_root)
    if (
        raw_result.path != expected_path
        or raw_result.payload_sha256 != validated.identity.payload_sha256
        or raw_result.state_file_sha256 != validated.identity.state_file_sha256
        or raw_result.state_size_bytes != len(validated.payload.state_bytes)
    ):
        message = "checkpoint writer result disagrees with the validated identity"
        raise ValueError(message)
    return IrisArmTeacherCheckpointWriteResult(
        identity=validated.identity,
        path=raw_result.path,
        state_size_bytes=raw_result.state_size_bytes,
        reused_existing=raw_result.reused_existing,
    )


def read_arm_teacher_checkpoint(
    identity: IrisArmTeacherCheckpointIdentity,
    *,
    root: Path,
) -> IrisArmTeacherCheckpointBundle:
    """Read only the exact content-addressed path named by an arm identity."""
    if type(identity) is not IrisArmTeacherCheckpointIdentity:
        message = "identity must be exactly IrisArmTeacherCheckpointIdentity"
        raise TypeError(message)
    path = identity.expected_path(root=_require_root(root))
    payload: IrisArmTeacherCheckpointPayload
    if identity.arm_id == "B":
        payload = read_no_workspace_teacher_checkpoint_v2(path)
    elif identity.arm_id == "C":
        payload = read_shared_buffer_teacher_checkpoint_v3(path)
    else:
        payload = read_teacher_checkpoint_v1(path)
    bundle = _bundle_from_payload(identity.arm_id, payload)
    if bundle.identity != identity:
        message = "checkpoint read from the expected path has a different identity"
        raise ValueError(message)
    return bundle


def restore_arm_teacher_checkpoint(
    bundle: IrisArmTeacherCheckpointBundle,
    *,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisTeacherCheckpointRestoreResult:
    """Restore a B/v2, C/v3, or D/v1 bundle into its exact fresh owner."""
    checked = _require_bundle(bundle)
    checked_model, checked_optimizer = _require_owner(
        checked.identity.arm_id,
        model,
        optimizer,
    )
    validated = validate_arm_teacher_checkpoint(checked)
    if validated.identity.arm_id == "B":
        result = restore_no_workspace_teacher_checkpoint_v2(
            cast(IrisNoWorkspaceTeacherCheckpointBundleV2, validated.payload),
            model=cast(IrisNoWorkspaceTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    elif validated.identity.arm_id == "C":
        result = restore_shared_buffer_teacher_checkpoint_v3(
            cast(IrisSharedBufferTeacherCheckpointBundleV3, validated.payload),
            model=cast(IrisSharedBufferTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    else:
        result = restore_teacher_checkpoint_v1(
            cast(IrisTeacherCheckpointBundleV1, validated.payload),
            model=cast(IrisTeacherForcingModel, checked_model),
            optimizer=checked_optimizer,
        )
    if type(result) is not IrisTeacherCheckpointRestoreResult:
        message = "checkpoint restorer returned an unexpected result type"
        raise TypeError(message)
    if (
        result.payload_sha256 != validated.identity.payload_sha256
        or result.sidecar_state_sha256 != validated.identity.sidecar_state_sha256
        or result.completed_steps != validated.identity.completed_steps
    ):
        message = "checkpoint restore result disagrees with the validated identity"
        raise ValueError(message)
    return result


__all__ = [
    "IrisArmTeacherCheckpointBundle",
    "IrisArmTeacherCheckpointIdentity",
    "IrisArmTeacherCheckpointPayload",
    "IrisArmTeacherCheckpointWriteResult",
    "IrisArmTeacherModel",
    "IrisTeacherArmId",
    "build_arm_teacher_checkpoint",
    "read_arm_teacher_checkpoint",
    "restore_arm_teacher_checkpoint",
    "validate_arm_teacher_checkpoint",
    "write_arm_teacher_checkpoint",
]
