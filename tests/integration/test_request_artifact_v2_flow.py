"""Offline verification flow for persisted A/B/D request artifacts v2."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.application.errors import IrisVerificationError
from iris.application.request_artifact_v2 import write_request_artifact_v2
from iris.application.verification_v2 import verify_request_artifact_v2
from tests.unit.test_request_artifact_v2 import _build


@pytest.mark.parametrize(
    ("arm_id", "workspace_cycles", "sidecar_present"),
    [("A", 0, False), ("B", 0, True), ("D", 3, True)],
)
def test_verifier_reads_each_arm_twice_without_mutation(
    tmp_path: Path,
    arm_id: str,
    workspace_cycles: int,
    sidecar_present: bool,
) -> None:
    artifact = _build(tmp_path, arm_id)
    result = write_request_artifact_v2(artifact)
    before = result.path.read_bytes()
    before_stat = result.path.stat()

    report = verify_request_artifact_v2(result.path)

    assert report.path == result.path
    assert report.artifact_type == "iris.request"
    assert report.version == 2
    assert report.arm_id == arm_id
    assert report.payload_sha256 == artifact.payload_sha256
    assert report.workspace_cycles == workspace_cycles
    assert report.sidecar_present is sidecar_present
    assert report.execution_profile_set_version == 3
    assert result.path.read_bytes() == before
    after_stat = result.path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_verifier_rejects_a_valid_artifact_under_the_wrong_digest_name(tmp_path: Path) -> None:
    result = write_request_artifact_v2(_build(tmp_path, "A"))
    wrong_path = result.path.with_name(f"{'0' * 64}.json")
    wrong_path.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v2(wrong_path)

    assert "filename" in str(raised.value.__cause__)


def test_verifier_rejects_v2_bytes_outside_the_canonical_layout(tmp_path: Path) -> None:
    result = write_request_artifact_v2(_build(tmp_path, "B"))
    wrong_directory = tmp_path / "requests" / "copy"
    wrong_directory.mkdir(parents=True)
    wrong_path = wrong_directory / result.path.name
    wrong_path.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v2(wrong_path)

    assert "canonical" in str(raised.value.__cause__)


def test_verifier_rejects_relocation_to_a_different_canonical_root(tmp_path: Path) -> None:
    result = write_request_artifact_v2(_build(tmp_path / "original", "A"))
    relocated = tmp_path / "relocated" / "requests" / "v2" / result.path.name
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v2(relocated)

    assert "configured output root" in str(raised.value.__cause__)


def test_verifier_rejects_tampered_private_fingerprint(tmp_path: Path) -> None:
    result = write_request_artifact_v2(_build(tmp_path, "D"))
    content = result.path.read_bytes()
    result.path.write_bytes(content.replace(b'"sha256":"', b'"sha256":"0', 1))

    with pytest.raises(IrisVerificationError, match="verification failed"):
        verify_request_artifact_v2(result.path)
