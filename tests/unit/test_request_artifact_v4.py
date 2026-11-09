"""Checkpoint-bound request artifact v4 schema, builder, and persistence tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import iris.application as application
import iris.application.request_artifact_v4 as request_artifact_v4
import iris.artifacts as artifacts
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointTransferResultV1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointIdentity,
    IrisTeacherArmId,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestArtifactError
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    read_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.artifacts.request_v4 import (
    IrisRequestArtifactEnvelopeV4,
    RequestArmBCheckpointTransferV4,
    RequestArmCCheckpointTransferV4,
    RequestArmDCheckpointTransferV4,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_request_artifact_v3 import (
    SECRET,
    _runtime_property_patches,
    artifact_inputs,
)

_SOURCE_SHA256 = "d" * 64
_PREVIOUS_SHA256 = "e" * 64


def _identity(arm_id: IrisTeacherArmId) -> IrisArmTeacherCheckpointIdentity:
    schema_version: Literal[1, 2, 3]
    if arm_id == "B":
        schema_version = 2
    elif arm_id == "C":
        schema_version = 3
    else:
        schema_version = 1
    return IrisArmTeacherCheckpointIdentity(
        arm_id=arm_id,
        checkpoint_schema_version=schema_version,
        payload_sha256="a" * 64,
        state_file_sha256="b" * 64,
        sidecar_state_sha256=_SOURCE_SHA256,
        completed_steps=7,
    )


def _transfer(
    runtime: IrisArmRequestRuntimeV3,
) -> IrisArmRequestCheckpointTransferResultV1:
    arm_id = cast(IrisTeacherArmId, runtime.profile.arm_id)
    return IrisArmRequestCheckpointTransferResultV1(
        runtime=runtime,
        checkpoint_identity=_identity(arm_id),
        transfer_contract="exact_teacher_state_to_profile_v4_request_sidecar_v1",
        source_teacher_sidecar_sha256=_SOURCE_SHA256,
        previous_request_sidecar_sha256=_PREVIOUS_SHA256,
        installed_request_sidecar_sha256=cast(str, runtime.sidecar_state_sha256),
        completed_steps=7,
        state_names=("layer_sampler.projection.weight",),
        transferred_tensor_count=1,
    )


def build_artifact_v4(tmp_path: Path, arm_id: str) -> IrisRequestArtifactEnvelopeV4:
    runtime, _, kwargs = artifact_inputs(tmp_path, arm_id)
    with _runtime_property_patches(arm_id):
        transfer = None if arm_id == "A" else _transfer(runtime)
        return build_request_artifact_v4(
            **kwargs,  # type: ignore[arg-type]
            checkpoint_transfer=transfer,
        )


@pytest.mark.parametrize(
    ("arm_id", "transfer_type", "checkpoint_schema_version"),
    [
        ("A", None, None),
        ("B", RequestArmBCheckpointTransferV4, 2),
        ("C", RequestArmCCheckpointTransferV4, 3),
        ("D", RequestArmDCheckpointTransferV4, 1),
    ],
)
def test_builder_binds_all_arms_to_exact_checkpoint_policy(
    tmp_path: Path,
    arm_id: str,
    transfer_type: type[object] | None,
    checkpoint_schema_version: int | None,
) -> None:
    artifact = build_artifact_v4(tmp_path, arm_id)

    assert artifact.schema_identity.version == 4
    assert artifact.payload.topology.arm_id == arm_id
    transfer = artifact.payload.checkpoint_transfer
    if transfer_type is None:
        assert transfer is None
    else:
        assert type(transfer) is transfer_type
        assert transfer is not None
        assert transfer.checkpoint_schema_version == checkpoint_schema_version
        assert transfer.source_teacher_sidecar_sha256 == _SOURCE_SHA256
        assert transfer.request_sidecar_state_sha256 == (
            artifact.payload.runtime.sidecar_state_sha256
        )
        assert transfer.completed_steps == 7
    assert SECRET.encode() not in canonical_json_bytes(artifact.model_dump(mode="json"))


def test_builder_requires_transfer_before_building_base_provenance(tmp_path: Path) -> None:
    _, _, kwargs = artifact_inputs(tmp_path, "B")
    with (
        patch.object(request_artifact_v4, "build_request_artifact_v3") as base_builder,
        pytest.raises(IrisRequestArtifactError, match="v4 could not be built") as raised,
    ):
        build_request_artifact_v4(
            **kwargs,  # type: ignore[arg-type]
            checkpoint_transfer=None,
        )

    assert "requires an exact checkpoint transfer" in str(raised.value.__cause__)
    base_builder.assert_not_called()


def test_builder_rejects_a_transfer_for_a_and_nonidentical_runtime(tmp_path: Path) -> None:
    runtime_b, _, kwargs_b = artifact_inputs(tmp_path, "B")
    with _runtime_property_patches("B"):
        transfer = _transfer(runtime_b)
    _, _, kwargs_a = artifact_inputs(tmp_path, "A")
    with (
        patch.object(request_artifact_v4, "build_request_artifact_v3") as base_builder,
        pytest.raises(IrisRequestArtifactError, match="v4 could not be built") as raised_a,
    ):
        build_request_artifact_v4(
            **kwargs_a,  # type: ignore[arg-type]
            checkpoint_transfer=transfer,
        )
    assert "arm A" in str(raised_a.value.__cause__)
    base_builder.assert_not_called()

    clone = replace(runtime_b)
    kwargs_b["runtime"] = clone
    with (
        patch.object(request_artifact_v4, "build_request_artifact_v3") as base_builder,
        pytest.raises(IrisRequestArtifactError, match="v4 could not be built") as raised_clone,
    ):
        build_request_artifact_v4(
            **kwargs_b,  # type: ignore[arg-type]
            checkpoint_transfer=transfer,
        )
    assert "exact runtime returned" in str(raised_clone.value.__cause__)
    base_builder.assert_not_called()


def test_writer_reader_round_trip_reuse_and_checkpoint_tamper_detection(
    tmp_path: Path,
) -> None:
    artifact = build_artifact_v4(tmp_path, "C")
    first = write_request_artifact_v4(artifact)
    second = write_request_artifact_v4(artifact)

    assert first.path.parent.name == "v4"
    assert not first.reused_existing
    assert second.reused_existing
    assert read_request_artifact_v4(first.path) == artifact

    content = first.path.read_text(encoding="utf-8")
    first.path.write_text(
        content.replace('"transferred_tensor_count":1', '"transferred_tensor_count":2', 1),
        encoding="utf-8",
    )
    with pytest.raises(IrisRequestArtifactError, match="could not be read or verified"):
        read_request_artifact_v4(first.path)


def test_writer_revalidates_embedded_v3_trace_integrity(tmp_path: Path) -> None:
    artifact = build_artifact_v4(tmp_path, "C")
    trace = artifact.payload.workspace_trace.model_copy(update={"sha256": "f" * 64})
    payload = artifact.payload.model_copy(update={"workspace_trace": trace})
    payload_sha256 = domain_sha256(
        "iris/request-artifact/v4/payload",
        canonical_json_bytes(payload.model_dump(mode="json")),
    )
    changed = artifact.model_copy(
        update={"payload": payload, "payload_sha256": payload_sha256}
    )

    with pytest.raises(IrisRequestArtifactError, match="could not be persisted") as raised:
        write_request_artifact_v4(changed)
    assert "trace digest is inconsistent" in str(raised.value.__cause__)


def test_transfer_schema_rejects_python_list_but_round_trips_json_tuple() -> None:
    raw: dict[str, object] = {
        "transfer_contract": "exact_teacher_state_to_profile_v4_request_sidecar_v1",
        "arm_id": "B",
        "checkpoint_schema_version": 2,
        "checkpoint_payload_sha256": "a" * 64,
        "checkpoint_state_file_sha256": "b" * 64,
        "source_teacher_sidecar_sha256": "c" * 64,
        "request_sidecar_state_sha256": "d" * 64,
        "completed_steps": 1,
        "transferred_state_names": ["layer_sampler.projection.weight"],
        "transferred_tensor_count": 1,
    }
    with pytest.raises(ValidationError):
        RequestArmBCheckpointTransferV4.model_validate(raw)

    parsed = RequestArmBCheckpointTransferV4.model_validate_json(json.dumps(raw))
    assert parsed.transferred_state_names == ("layer_sampler.projection.weight",)


def test_v4_public_api_is_exported() -> None:
    assert artifacts.IrisRequestArtifactEnvelopeV4 is IrisRequestArtifactEnvelopeV4
    assert artifacts.RequestArmBCheckpointTransferV4 is RequestArmBCheckpointTransferV4
    assert application.build_request_artifact_v4 is build_request_artifact_v4
    assert application.read_request_artifact_v4 is read_request_artifact_v4
    assert application.write_request_artifact_v4 is write_request_artifact_v4
