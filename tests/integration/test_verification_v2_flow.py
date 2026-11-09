"""Read-only verification of complete arm-labelled B/D journal stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.application.arm_training_checkpoint import IrisTeacherArmId
from iris.application.errors import IrisVerificationError
from iris.application.experiment_log_v2 import read_experiment_log_v2
from iris.application.experiment_manifest import (
    build_experiment_manifest_v1,
    write_experiment_manifest_v1,
)
from iris.application.verification_v2 import verify_experiment_journal_v2
from iris.artifacts.experiment_log_v2 import ExperimentCheckpointEventV2
from iris.artifacts.experiment_v1 import ExperimentExampleV1
from tests.integration.test_experiment_iteration import _manifest
from tests.integration.test_experiment_runner_v2_flow import (
    _deterministic_runtime,
    _execute,
    _fresh_owner,
)

pytestmark = pytest.mark.integration


def _tree_snapshot(root: Path) -> tuple[tuple[str, bytes | None], ...]:
    entries: list[tuple[str, bytes | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        entries.append(
            (
                path.relative_to(root).as_posix(),
                None if path.is_dir() else path.read_bytes(),
            )
        )
    return tuple(entries)


@pytest.mark.parametrize(("arm_id", "checkpoint_version"), [("B", 2), ("D", 1)])
def test_verifier_reads_complete_arm_store_without_mutation(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    write_experiment_manifest_v1(manifest, path=manifest_path)
    with _deterministic_runtime():
        model, optimizer = _fresh_owner(arm_id)
        result = _execute(
            root=tmp_path,
            run_id=f"verify-complete-{arm_id.lower()}",
            arm_id=arm_id,
            model=model,
            optimizer=optimizer,
        )
    before = _tree_snapshot(tmp_path)

    report = verify_experiment_journal_v2(
        path=result.path,
        manifest_path=manifest_path,
    )

    assert report.path == result.path
    assert report.artifact_type == "iris.experiment_run"
    assert report.version == 2
    assert report.arm_id == arm_id
    assert report.execution_profile_set_version == 3
    assert report.run_payload_sha256 == result.run_payload_sha256
    assert report.configuration_sha256 == result.configuration_sha256
    assert report.manifest_payload_sha256 == manifest.payload_sha256
    assert report.execution_mode == "deterministic_epochs"
    assert report.event_count == 5
    assert report.training_step_event_count == 2
    assert report.evaluation_event_count == 2
    assert report.checkpoint_event_count == 1
    assert report.optimizer_completed_steps == 2
    assert report.checkpoint_reference_count == 2
    assert report.unique_checkpoint_count == 2
    assert report.latest_checkpoint_schema_version == checkpoint_version
    assert report.latest_checkpoint_payload_sha256 == result.latest_checkpoint_payload_sha256
    assert report.managed_phase == "complete"
    assert report.completed_epochs == 1
    assert report.active_epoch is None
    assert report.next_training_ordinal is None
    assert _tree_snapshot(tmp_path) == before


def test_verifier_rejects_wrong_manifest_without_checkpoint_mutation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    wrong_manifest = build_experiment_manifest_v1(
        dataset_id="wrong-dataset",
        examples=(
            ExperimentExampleV1(
                example_id="train",
                split="train",
                prompt="private prompt",
                target="private target",
            ),
            ExperimentExampleV1(
                example_id="validation",
                split="validation",
                prompt="private prompt",
                target="private target",
            ),
        ),
    )
    write_experiment_manifest_v1(wrong_manifest, path=manifest_path)
    with _deterministic_runtime():
        model, optimizer = _fresh_owner("B")
        result = _execute(
            root=tmp_path,
            run_id="verify-wrong-manifest",
            arm_id="B",
            model=model,
            optimizer=optimizer,
        )
    before = _tree_snapshot(tmp_path / "teacher")

    with pytest.raises(IrisVerificationError, match="verification failed"):
        verify_experiment_journal_v2(
            path=result.path,
            manifest_path=manifest_path,
        )

    assert _tree_snapshot(tmp_path / "teacher") == before


def test_verifier_rejects_tampered_referenced_checkpoint(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_experiment_manifest_v1(_manifest(), path=manifest_path)
    with _deterministic_runtime():
        model, optimizer = _fresh_owner("D")
        result = _execute(
            root=tmp_path,
            run_id="verify-tampered-checkpoint",
            arm_id="D",
            model=model,
            optimizer=optimizer,
        )
    snapshot = read_experiment_log_v2(result.path)
    reference = snapshot.events[2].payload.record
    assert type(reference) is ExperimentCheckpointEventV2
    payload_sha256 = reference.checkpoint.payload_sha256
    state_path = tmp_path / "teacher" / "v1" / payload_sha256 / "state.safetensors"
    state_bytes = state_path.read_bytes()
    state_path.write_bytes(state_bytes[:-1] + bytes([state_bytes[-1] ^ 1]))

    with pytest.raises(IrisVerificationError, match="verification failed"):
        verify_experiment_journal_v2(
            path=result.path,
            manifest_path=manifest_path,
        )
