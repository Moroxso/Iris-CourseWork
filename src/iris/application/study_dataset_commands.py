"""Content-redacted dataset verification and explicit plaintext manifest export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.errors import IrisStudyDatasetError, IrisVerificationError
from iris.application.experiment_manifest import write_experiment_manifest_v1
from iris.application.study_dataset import project_study_experiment_manifest_v1
from iris.application.study_dataset_io import read_study_dataset_v1
from iris.artifacts.study_dataset_v1 import StudyDatasetPurposeV1
from iris.artifacts.study_v1 import StudyTaskFamilyV1


@dataclass(frozen=True, slots=True)
class IrisStudyTaskSplitCountsV1:
    task_family: StudyTaskFamilyV1
    train: int
    validation: int
    test: int


@dataclass(frozen=True, slots=True)
class IrisStudyDatasetVerificationReportV1:
    """Private identities/counts only; no source descriptions, examples, or group labels."""

    path: Path
    artifact_type: Literal["iris.study_dataset"]
    version: Literal[1]
    payload_sha256: str
    manifest_payload_sha256: str
    protocol_declaration_payload_sha256: str
    dataset_id: str
    purpose: StudyDatasetPurposeV1
    task_counts: tuple[IrisStudyTaskSplitCountsV1, ...]
    total_examples: int
    leakage_group_count: int
    validation_scope: Literal["input_membership_and_exact_duplicate_checks_only"]
    scientific_launch_status: Literal["not_assessed"]


@dataclass(frozen=True, slots=True)
class IrisStudyDatasetManifestExportResultV1:
    """Exact dataset-to-training-manifest binding; includes all three splits."""

    path: Path
    dataset_payload_sha256: str
    manifest_payload_sha256: str
    size_bytes: int
    reused_existing: bool


def verify_study_dataset_v1(path: Path) -> IrisStudyDatasetVerificationReportV1:
    """Verify the dataset without loading a model, executing source code, or scoring tasks."""
    try:
        dataset = read_study_dataset_v1(path)
        payload = dataset.payload
        counts = {
            task.task_family: {"train": 0, "validation": 0, "test": 0}
            for task in payload.task_definitions
        }
        for example, assignment in zip(
            payload.manifest.payload.examples, payload.assignments, strict=True
        ):
            counts[assignment.task_family][example.split] += 1
        return IrisStudyDatasetVerificationReportV1(
            path=path,
            artifact_type=dataset.schema_identity.artifact_type,
            version=dataset.schema_identity.version,
            payload_sha256=dataset.payload_sha256,
            manifest_payload_sha256=payload.manifest.payload_sha256,
            protocol_declaration_payload_sha256=payload.protocol_declaration_payload_sha256,
            dataset_id=payload.manifest.payload.dataset_id,
            purpose=payload.purpose,
            task_counts=tuple(
                IrisStudyTaskSplitCountsV1(
                    task_family=task.task_family,
                    train=counts[task.task_family]["train"],
                    validation=counts[task.task_family]["validation"],
                    test=counts[task.task_family]["test"],
                )
                for task in payload.task_definitions
            ),
            total_examples=len(payload.assignments),
            leakage_group_count=len({item.leakage_group_id for item in payload.assignments}),
            validation_scope=dataset.schema_identity.validation_scope,
            scientific_launch_status="not_assessed",
        )
    except Exception as error:
        raise IrisVerificationError("Iris study dataset verification failed") from error


def export_study_experiment_manifest_v1(
    *, dataset_path: Path, manifest_path: Path
) -> IrisStudyDatasetManifestExportResultV1:
    """Export ALL private inputs/targets for the existing runner without changing its API."""
    try:
        dataset = read_study_dataset_v1(dataset_path)
        manifest = project_study_experiment_manifest_v1(dataset)
        result = write_experiment_manifest_v1(manifest, path=manifest_path)
        return IrisStudyDatasetManifestExportResultV1(
            path=result.path,
            dataset_payload_sha256=dataset.payload_sha256,
            manifest_payload_sha256=result.payload_sha256,
            size_bytes=result.size_bytes,
            reused_existing=result.reused_existing,
        )
    except Exception as error:
        raise IrisStudyDatasetError("Iris study dataset manifest could not be exported") from error


__all__ = [
    "IrisStudyDatasetManifestExportResultV1",
    "IrisStudyDatasetVerificationReportV1",
    "IrisStudyTaskSplitCountsV1",
    "export_study_experiment_manifest_v1",
    "verify_study_dataset_v1",
]
