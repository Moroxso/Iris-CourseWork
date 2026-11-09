"""Offline read-only verification flow for persisted A/B/C/D request artifacts v3."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application as application
from iris.application.errors import IrisVerificationError
from iris.application.request_artifact_v3 import write_request_artifact_v3
from iris.application.verification_v3 import verify_request_artifact_v3
from tests.unit.test_request_artifact_v3 import SECRET, build_artifact

pytestmark = pytest.mark.integration


def _tree_snapshot(root: Path) -> tuple[tuple[str, bytes | None, int], ...]:
    entries: list[tuple[str, bytes | None, int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        stat = path.stat()
        entries.append(
            (
                path.relative_to(root).as_posix(),
                None if path.is_dir() else path.read_bytes(),
                stat.st_mtime_ns,
            )
        )
    return tuple(entries)


@pytest.mark.parametrize(
    ("arm_id", "workspace_cycles", "zero_memory_reads", "sidecar_present"),
    [
        ("A", 0, 0, False),
        ("B", 0, 3, True),
        ("C", 3, 1, True),
        ("D", 3, 0, True),
    ],
)
def test_verifier_reads_every_profile_v4_arm_twice_without_mutation(
    tmp_path: Path,
    arm_id: str,
    workspace_cycles: int,
    zero_memory_reads: int,
    sidecar_present: bool,
) -> None:
    artifact = build_artifact(tmp_path, arm_id)
    result = write_request_artifact_v3(artifact)
    artifact_root = tmp_path / "artifacts"
    before = _tree_snapshot(artifact_root)

    report = verify_request_artifact_v3(result.path)

    assert report.path == result.path
    assert report.artifact_type == "iris.request"
    assert report.version == 3
    assert report.arm_id == arm_id
    assert report.payload_sha256 == artifact.payload_sha256
    assert report.execution_profile_set_version == 4
    assert report.workspace_cycles == workspace_cycles
    assert report.zero_memory_read_cycles == zero_memory_reads
    assert report.sidecar_present is sidecar_present
    assert SECRET not in repr(report)
    assert _tree_snapshot(artifact_root) == before


def test_arm_c_report_proves_public_cardinalities_without_private_fingerprints(
    tmp_path: Path,
) -> None:
    artifact = build_artifact(tmp_path, "C")
    result = write_request_artifact_v3(artifact)

    report = verify_request_artifact_v3(result.path)

    assert report.request_graph == "append_only_all_proposals_prior_cycle_shared_buffer_v1"
    assert report.trace_contract == "append_only_shared_buffer_per_cycle_v1"
    assert report.cognitive_cycles == 3
    assert report.latent_token_count == 8
    assert report.pre_update_read_entry_counts == (0, 8, 16)
    assert report.post_cycle_entry_counts == (8, 16, 24)
    assert report.final_output_read_entry_count == 24
    assert artifact.payload.request.prompt.sha256 not in repr(report)
    assert artifact.payload.generation.text.sha256 not in repr(report)
    trace = artifact.payload.workspace_trace
    assert hasattr(trace, "final_output_read")
    assert trace.final_output_read.workspace_reads.sha256 not in repr(report)


def test_verifier_rejects_a_valid_artifact_under_the_wrong_digest_name(tmp_path: Path) -> None:
    result = write_request_artifact_v3(build_artifact(tmp_path, "A"))
    wrong_path = result.path.with_name(f"{'0' * 64}.json")
    wrong_path.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v3(wrong_path)

    assert "filename" in str(raised.value.__cause__)


def test_verifier_rejects_v3_bytes_outside_the_canonical_layout(tmp_path: Path) -> None:
    result = write_request_artifact_v3(build_artifact(tmp_path, "B"))
    wrong_directory = tmp_path / "artifacts" / "requests" / "copy"
    wrong_directory.mkdir(parents=True)
    wrong_path = wrong_directory / result.path.name
    wrong_path.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v3(wrong_path)

    assert "canonical" in str(raised.value.__cause__)


def test_verifier_rejects_relocation_to_a_different_canonical_root(tmp_path: Path) -> None:
    result = write_request_artifact_v3(build_artifact(tmp_path / "original", "D"))
    relocated = tmp_path / "relocated" / "requests" / "v3" / result.path.name
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(result.path.read_bytes())

    with pytest.raises(IrisVerificationError, match="verification failed") as raised:
        verify_request_artifact_v3(relocated)

    assert "configured output root" in str(raised.value.__cause__)


def test_verifier_rejects_tampered_nested_arm_c_fingerprint(tmp_path: Path) -> None:
    result = write_request_artifact_v3(build_artifact(tmp_path, "C"))
    content = result.path.read_bytes()
    result.path.write_bytes(content.replace(b'"sha256":"', b'"sha256":"0', 1))

    with pytest.raises(IrisVerificationError, match="verification failed"):
        verify_request_artifact_v3(result.path)


def test_verifier_rejects_an_artifact_that_changes_between_reads(tmp_path: Path) -> None:
    first = build_artifact(tmp_path, "A")
    second = build_artifact(tmp_path, "B")
    result = write_request_artifact_v3(first)

    with (
        patch(
            "iris.application.verification_v3.read_request_artifact_v3",
            side_effect=(first, second),
        ),
        pytest.raises(IrisVerificationError, match="verification failed") as raised,
    ):
        verify_request_artifact_v3(result.path)

    assert "changed during verification" in str(raised.value.__cause__)


def test_verifier_fails_closed_if_installed_profile_v4_identity_drifts(tmp_path: Path) -> None:
    result = write_request_artifact_v3(build_artifact(tmp_path, "A"))

    with (
        patch(
            "iris.application.verification_v3.arm_execution_profile_set_sha256_v4",
            return_value="0" * 64,
        ),
        pytest.raises(IrisVerificationError, match="verification failed") as raised,
    ):
        verify_request_artifact_v3(result.path)

    assert "profile-set identity" in str(raised.value.__cause__)


def test_request_v3_verifier_is_exported_from_application_package() -> None:
    assert application.verify_request_artifact_v3 is verify_request_artifact_v3
    assert (
        application.IrisRequestArtifactVerificationReportV3.__module__
        == "iris.application.verification_v3"
    )
