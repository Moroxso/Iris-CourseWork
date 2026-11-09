"""Read-only artifact verification and referential-integrity tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import iris.application.verification as verification_module
from iris.application.errors import IrisVerificationError
from iris.application.experiment_log import read_experiment_log_v1
from iris.application.experiment_manifest import (
    read_experiment_manifest_v1,
    write_experiment_manifest_v1,
)
from iris.application.request_artifact import write_request_artifact_v1
from iris.application.training_checkpoint import write_teacher_checkpoint_v1
from iris.application.verification import (
    verify_experiment_journal_v1,
    verify_experiment_manifest_v1,
    verify_request_artifact_v1,
    verify_teacher_checkpoint_v1,
)
from iris.artifacts.experiment_log_v1 import ExperimentCheckpointEventV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from tests.integration.test_experiment_iteration import _manifest
from tests.integration.test_experiment_log_flow import _create
from tests.integration.test_experiment_runner_flow import (
    _deterministic_runtime,
    _execute,
    _fresh_owner,
)
from tests.integration.test_teacher_checkpoint_flow import _trained_bundle
from tests.unit.test_request_artifact_writer import _minimal_artifact

pytestmark = pytest.mark.integration

_EVENT_PAYLOAD_DOMAIN = "iris/experiment-log/v1/event-payload"


@contextmanager
def _manual_runtime() -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        torch.use_deterministic_algorithms(False)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(
                previous_enabled,
                warn_only=previous_warn_only,
            )


def _tree_snapshot(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    entries = tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))
    files = {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    return entries, files


def _install_managed_run(
    root: Path,
    *,
    run_id: str,
    maximum_new_training_steps: int | None = None,
) -> tuple[Path, Path]:
    manifest = _manifest()
    manifest_path = root / f"{run_id}-manifest.json"
    write_experiment_manifest_v1(manifest, path=manifest_path)
    with _deterministic_runtime():
        model, optimizer = _fresh_owner()
        result = _execute(
            root=root,
            run_id=run_id,
            model=model,
            optimizer=optimizer,
            maximum_new_training_steps=maximum_new_training_steps,
        )
    return result.path, manifest_path


def test_all_verifiers_return_content_redacted_reports_without_writes(tmp_path: Path) -> None:
    request_artifact = _minimal_artifact(tmp_path)
    request_write = write_request_artifact_v1(request_artifact)

    with torch.random.fork_rng(devices=[]):
        _, _, _, checkpoint = _trained_bundle()
    checkpoint_write = write_teacher_checkpoint_v1(checkpoint, root=tmp_path)

    journal_path, manifest_path = _install_managed_run(tmp_path, run_id="verify-managed")
    before = _tree_snapshot(tmp_path)

    request_report = verify_request_artifact_v1(request_write.path)
    checkpoint_report = verify_teacher_checkpoint_v1(checkpoint_write.path)
    manifest_report = verify_experiment_manifest_v1(manifest_path)
    journal_report = verify_experiment_journal_v1(
        path=journal_path,
        manifest_path=manifest_path,
    )

    assert _tree_snapshot(tmp_path) == before
    assert request_report.payload_sha256 == request_artifact.payload_sha256
    assert request_report.generated_tokens == 1
    assert request_report.workspace_cycles == 1
    assert checkpoint_report.payload_sha256 == checkpoint.manifest.sha256
    assert checkpoint_report.state_file_sha256 == checkpoint.manifest.payload.state_file.sha256
    assert checkpoint_report.completed_steps == 1
    assert manifest_report.payload_sha256 == _manifest().payload_sha256
    assert (
        manifest_report.train_examples,
        manifest_report.validation_examples,
        manifest_report.test_examples,
        manifest_report.total_examples,
    ) == (2, 1, 1, 4)
    assert journal_report.execution_mode == "deterministic_epochs"
    assert journal_report.event_count == 5
    assert journal_report.training_step_event_count == 2
    assert journal_report.evaluation_event_count == 2
    assert journal_report.checkpoint_event_count == 1
    assert journal_report.optimizer_completed_steps == 2
    assert journal_report.checkpoint_reference_count == 2
    assert journal_report.unique_checkpoint_count == 2
    assert journal_report.managed_phase == "complete"
    assert journal_report.completed_epochs == 1
    assert journal_report.active_epoch is None
    assert journal_report.next_training_ordinal is None

    redacted = "\n".join(
        map(repr, (request_report, checkpoint_report, manifest_report, journal_report))
    )
    for private_value in (
        "offline teacher prompt",
        "longer target",
        "short target",
        "train-long",
        "train-short",
        "validation-short",
        "test-long",
    ):
        assert private_value not in redacted


def test_manual_journal_still_verifies_manifest_and_initial_checkpoint(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manual-manifest.json"
    write_experiment_manifest_v1(_manifest(), path=manifest_path)
    with _manual_runtime():
        _, _, created = _create(tmp_path)

    report = verify_experiment_journal_v1(
        path=created.path,
        manifest_path=manifest_path,
    )

    assert report.execution_mode == "manual_event_chain"
    assert report.event_count == 0
    assert report.checkpoint_reference_count == 1
    assert report.unique_checkpoint_count == 1
    assert report.latest_checkpoint_payload_sha256 == created.initial_checkpoint_payload_sha256
    assert report.managed_phase is None
    assert report.completed_epochs is None
    assert report.active_epoch is None
    assert report.next_training_ordinal is None


def test_journal_verification_rejects_a_different_valid_manifest(tmp_path: Path) -> None:
    journal_path, _ = _install_managed_run(tmp_path, run_id="verify-manifest-mismatch")
    other_manifest_path = tmp_path / "other-manifest.json"
    write_experiment_manifest_v1(_manifest(include_test=False), path=other_manifest_path)

    with pytest.raises(IrisVerificationError, match="journal verification failed"):
        verify_experiment_journal_v1(
            path=journal_path,
            manifest_path=other_manifest_path,
        )


def test_journal_verification_rejects_rehashed_manifest_reference_tamper(
    tmp_path: Path,
) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id="verify-order-tamper",
        maximum_new_training_steps=1,
    )
    event_path = journal_path / "events" / "00000001.json"
    raw = json.loads(event_path.read_text(encoding="utf-8"))
    raw["payload"]["record"]["example_id"] = "train-short"
    raw["payload_sha256"] = domain_sha256(
        _EVENT_PAYLOAD_DOMAIN,
        canonical_json_bytes(raw["payload"]),
    )
    event_path.write_bytes(canonical_json_file_bytes(raw))
    assert read_experiment_log_v1(journal_path).event_count == 1

    with pytest.raises(IrisVerificationError, match="journal verification failed"):
        verify_experiment_journal_v1(
            path=journal_path,
            manifest_path=manifest_path,
        )


def test_journal_verification_rejects_manifest_or_chain_change_before_return(
    tmp_path: Path,
) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id="verify-stability",
    )
    snapshot = read_experiment_log_v1(journal_path)
    manifest = read_experiment_manifest_v1(manifest_path)

    with (
        patch(
            "iris.application.verification.read_experiment_log_v1",
            side_effect=[snapshot, replace(snapshot, events=snapshot.events[:-1])],
        ),
        pytest.raises(IrisVerificationError) as changed_journal,
    ):
        verify_experiment_journal_v1(
            path=journal_path,
            manifest_path=manifest_path,
        )
    assert changed_journal.value.__cause__ is not None
    assert "journal changed" in str(changed_journal.value.__cause__)

    with (
        patch.object(
            verification_module,
            "read_experiment_manifest_v1",
            side_effect=[manifest, _manifest(include_test=False)],
        ),
        pytest.raises(IrisVerificationError) as changed_manifest,
    ):
        verify_experiment_journal_v1(
            path=journal_path,
            manifest_path=manifest_path,
        )
    assert changed_manifest.value.__cause__ is not None
    assert "manifest changed" in str(changed_manifest.value.__cause__)


@pytest.mark.parametrize("reference_location", ["initial", "event"])
def test_journal_verification_reads_and_rejects_each_tampered_checkpoint_reference(
    tmp_path: Path,
    reference_location: str,
) -> None:
    journal_path, manifest_path = _install_managed_run(
        tmp_path,
        run_id=f"verify-checkpoint-{reference_location}",
    )
    snapshot = read_experiment_log_v1(journal_path)
    if reference_location == "initial":
        reference = snapshot.run.payload.initial_checkpoint
    else:
        record = snapshot.events[2].payload.record
        assert type(record) is ExperimentCheckpointEventV1
        reference = record.checkpoint
    state_path = tmp_path / "teacher" / "v1" / reference.payload_sha256 / "state.safetensors"
    state_bytes = state_path.read_bytes()
    state_path.write_bytes(bytes((state_bytes[0] ^ 1,)) + state_bytes[1:])

    with pytest.raises(IrisVerificationError, match="journal verification failed"):
        verify_experiment_journal_v1(
            path=journal_path,
            manifest_path=manifest_path,
        )


def test_standalone_verifiers_reject_valid_content_outside_canonical_store_layout(
    tmp_path: Path,
) -> None:
    request_artifact = _minimal_artifact(tmp_path)
    request_write = write_request_artifact_v1(request_artifact)
    copied_request = tmp_path / "copied-request.json"
    copied_request.write_bytes(request_write.path.read_bytes())
    other_request = (
        tmp_path / "other-root" / "requests" / "v1" / f"{request_artifact.payload_sha256}.json"
    )
    other_request.parent.mkdir(parents=True)
    other_request.write_bytes(request_write.path.read_bytes())

    with torch.random.fork_rng(devices=[]):
        _, _, _, checkpoint = _trained_bundle()
    checkpoint_write = write_teacher_checkpoint_v1(checkpoint, root=tmp_path)
    copied_checkpoint = tmp_path / checkpoint.manifest.sha256
    copied_checkpoint.mkdir()
    for source in checkpoint_write.path.iterdir():
        (copied_checkpoint / source.name).write_bytes(source.read_bytes())

    with pytest.raises(IrisVerificationError, match="request artifact verification failed"):
        verify_request_artifact_v1(copied_request)
    with pytest.raises(IrisVerificationError, match="request artifact verification failed"):
        verify_request_artifact_v1(other_request)
    with pytest.raises(IrisVerificationError, match="checkpoint verification failed"):
        verify_teacher_checkpoint_v1(copied_checkpoint)
