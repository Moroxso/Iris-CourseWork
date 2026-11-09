"""Read-only verifier tests for checkpoint-bound request artifacts v4."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application as application
import iris.application.verification_v4 as verification_v4
from iris.application.arm_request_checkpoint import IrisArmRequestCheckpointProjectionV1
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import IrisVerificationError
from iris.application.request_artifact_v4 import write_request_artifact_v4
from iris.application.verification_v4 import (
    IrisRequestArtifactVerificationReportV4,
    verify_request_artifact_v4,
)
from iris.artifacts.checkpoint_v1 import TeacherBackboneRuntimeV1
from iris.artifacts.request_v4 import IrisRequestArtifactEnvelopeV4
from tests.unit.test_request_artifact_v4 import build_artifact_v4


def _projection(artifact: IrisRequestArtifactEnvelopeV4) -> IrisArmRequestCheckpointProjectionV1:
    transfer = artifact.payload.checkpoint_transfer
    if transfer is None:
        raise AssertionError("projection fixture requires arm B, C, or D")
    config = artifact.payload.configuration.value
    return IrisArmRequestCheckpointProjectionV1(
        checkpoint_identity=IrisArmTeacherCheckpointIdentity(
            arm_id=transfer.arm_id,
            checkpoint_schema_version=transfer.checkpoint_schema_version,
            payload_sha256=transfer.checkpoint_payload_sha256,
            state_file_sha256=transfer.checkpoint_state_file_sha256,
            sidecar_state_sha256=transfer.source_teacher_sidecar_sha256,
            completed_steps=transfer.completed_steps,
        ),
        backbone=config.backbone,
        cognitive=config.cognitive,
        workspace=config.workspace if transfer.arm_id == "D" else None,
        runtime=TeacherBackboneRuntimeV1(
            hidden_size=8,
            num_hidden_layers=4,
            vocabulary_size=32,
            max_position_embeddings=1024,
            max_prompt_tokens=config.backbone.limits.max_prompt_tokens,
            max_target_tokens=config.backbone.limits.max_new_tokens,
            device=artifact.payload.runtime.device,
            dtype=artifact.payload.runtime.dtype,
        ),
        source_teacher_sidecar_sha256=transfer.source_teacher_sidecar_sha256,
        request_sidecar_state_sha256=transfer.request_sidecar_state_sha256,
        completed_steps=transfer.completed_steps,
        state_names=transfer.transferred_state_names,
        tensor_count=transfer.transferred_tensor_count,
    )


def test_arm_a_verifies_without_checkpoint_access(tmp_path: Path) -> None:
    artifact = build_artifact_v4(tmp_path, "A")
    written = write_request_artifact_v4(artifact)
    with patch.object(verification_v4, "read_arm_teacher_checkpoint") as checkpoint_read:
        report = verify_request_artifact_v4(written.path)

    assert type(report) is IrisRequestArtifactVerificationReportV4
    assert report.arm_id == "A"
    assert not report.checkpoint_present
    assert report.checkpoint_schema_version is None
    assert report.checkpoint_payload_sha256 is None
    assert report.transferred_tensor_count == 0
    checkpoint_read.assert_not_called()


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_trained_arms_verify_checkpoint_projection_twice(
    tmp_path: Path,
    arm_id: str,
) -> None:
    artifact = build_artifact_v4(tmp_path, arm_id)
    written = write_request_artifact_v4(artifact)
    projection = _projection(artifact)
    checkpoint = object()
    with (
        patch.object(
            verification_v4,
            "read_arm_teacher_checkpoint",
            return_value=checkpoint,
        ) as checkpoint_read,
        patch.object(
            verification_v4,
            "project_arm_teacher_checkpoint_to_request_v1",
            return_value=projection,
        ) as project,
    ):
        report = verify_request_artifact_v4(written.path)

    transfer = artifact.payload.checkpoint_transfer
    assert transfer is not None
    assert report.arm_id == arm_id
    assert report.checkpoint_present
    assert report.checkpoint_schema_version == transfer.checkpoint_schema_version
    assert report.checkpoint_payload_sha256 == transfer.checkpoint_payload_sha256
    assert report.checkpoint_state_file_sha256 == transfer.checkpoint_state_file_sha256
    assert report.source_teacher_sidecar_sha256 == transfer.source_teacher_sidecar_sha256
    assert report.request_sidecar_state_sha256 == transfer.request_sidecar_state_sha256
    assert report.checkpoint_completed_steps == transfer.completed_steps
    assert report.transferred_tensor_count == transfer.transferred_tensor_count
    assert checkpoint_read.call_count == 2
    assert project.call_count == 2


def test_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    artifact = build_artifact_v4(tmp_path, "B")
    written = write_request_artifact_v4(artifact)
    with (
        patch.object(
            verification_v4,
            "read_arm_teacher_checkpoint",
            side_effect=FileNotFoundError("checkpoint removed"),
        ),
        pytest.raises(IrisVerificationError, match="verification failed") as raised,
    ):
        verify_request_artifact_v4(written.path)

    assert "checkpoint removed" in str(raised.value.__cause__)


def test_projected_tensor_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = build_artifact_v4(tmp_path, "C")
    written = write_request_artifact_v4(artifact)
    projection = _projection(artifact)
    changed = IrisArmRequestCheckpointProjectionV1(
        checkpoint_identity=projection.checkpoint_identity,
        backbone=projection.backbone,
        cognitive=projection.cognitive,
        workspace=projection.workspace,
        runtime=projection.runtime.model_copy(
            update={"hidden_size": projection.runtime.hidden_size + 1}
        ),
        source_teacher_sidecar_sha256=projection.source_teacher_sidecar_sha256,
        request_sidecar_state_sha256="f" * 64,
        completed_steps=projection.completed_steps,
        state_names=projection.state_names,
        tensor_count=projection.tensor_count,
    )
    with (
        patch.object(verification_v4, "read_arm_teacher_checkpoint", return_value=object()),
        patch.object(
            verification_v4,
            "project_arm_teacher_checkpoint_to_request_v1",
            return_value=changed,
        ),
        pytest.raises(IrisVerificationError, match="verification failed") as raised,
    ):
        verify_request_artifact_v4(written.path)

    assert "does not match projected checkpoint tensor state" in str(raised.value.__cause__)


def test_checkpoint_change_between_reads_fails_closed(tmp_path: Path) -> None:
    artifact = build_artifact_v4(tmp_path, "D")
    written = write_request_artifact_v4(artifact)
    projection = _projection(artifact)
    changed = IrisArmRequestCheckpointProjectionV1(
        checkpoint_identity=projection.checkpoint_identity,
        backbone=projection.backbone,
        cognitive=projection.cognitive,
        workspace=projection.workspace,
        runtime=projection.runtime.model_copy(
            update={"hidden_size": projection.runtime.hidden_size + 1}
        ),
        source_teacher_sidecar_sha256=projection.source_teacher_sidecar_sha256,
        request_sidecar_state_sha256=projection.request_sidecar_state_sha256,
        completed_steps=projection.completed_steps,
        state_names=projection.state_names,
        tensor_count=projection.tensor_count,
    )
    with (
        patch.object(verification_v4, "read_arm_teacher_checkpoint", return_value=object()),
        patch.object(
            verification_v4,
            "project_arm_teacher_checkpoint_to_request_v1",
            side_effect=(projection, changed),
        ),
        pytest.raises(IrisVerificationError, match="verification failed") as raised,
    ):
        verify_request_artifact_v4(written.path)

    assert "checkpoint changed" in str(raised.value.__cause__)


def test_v4_verifier_is_exported() -> None:
    assert application.IrisRequestArtifactVerificationReportV4 is (
        IrisRequestArtifactVerificationReportV4
    )
    assert application.verify_request_artifact_v4 is verify_request_artifact_v4
