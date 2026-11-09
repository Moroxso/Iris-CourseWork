"""Owned, hash-verified task datasets with a lossless bridge to the training runner."""

from dataclasses import dataclass
from typing import get_args

from iris.application.errors import IrisStudyDatasetError
from iris.application.experiment_manifest import validate_experiment_manifest_v1
from iris.application.study_declaration import build_four_arm_protocol_declaration_v1
from iris.artifacts.experiment_v1 import (
    ExperimentExampleV1,
    ExperimentSplitV1,
    IrisExperimentManifestV1,
)
from iris.artifacts.study_dataset_v1 import (
    IrisStudyDatasetV1,
    StudyDatasetConstructionV1,
    StudyDatasetPayloadV1,
    StudyDatasetPurposeV1,
    StudyDatasetSchemaV1,
    StudyExampleAssignmentV1,
    StudyTaskDefinitionV1,
)
from iris.artifacts.study_v1 import StudyTaskFamilyV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/study-dataset/v1/payload"
_MAX_DATASET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IrisStudyExampleV1:
    """Owned pair in original manifest order; contains private prompt and target text."""

    example: ExperimentExampleV1
    assignment: StudyExampleAssignmentV1


def _validate_integrity(dataset: IrisStudyDatasetV1) -> IrisStudyDatasetV1:
    if type(dataset) is not IrisStudyDatasetV1:
        raise TypeError("dataset must be exactly IrisStudyDatasetV1")
    owned = IrisStudyDatasetV1.model_validate(dataset.model_dump(mode="python"))
    # The enclosing hash is no substitute for checking the original manifest's hash.
    validate_experiment_manifest_v1(owned.payload.manifest)
    protocol = build_four_arm_protocol_declaration_v1()
    if owned.payload.protocol_declaration_payload_sha256 != protocol.payload_sha256:
        raise ValueError("study dataset must bind the exact recovered four-arm protocol")
    expected_sha256 = domain_sha256(
        _PAYLOAD_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    )
    if owned.payload_sha256 != expected_sha256:
        raise ValueError("study dataset payload SHA-256 is inconsistent")
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_DATASET_BYTES:
        raise ValueError("study dataset exceeds the 64 MiB safety limit")
    return owned


def build_study_dataset_v1(
    *,
    manifest: IrisExperimentManifestV1,
    construction: StudyDatasetConstructionV1,
    task_definitions: tuple[StudyTaskDefinitionV1, ...],
    assignments: tuple[StudyExampleAssignmentV1, ...],
    purpose: StudyDatasetPurposeV1,
) -> IrisStudyDatasetV1:
    """Bind declared provenance and membership; do not authorize a scientific launch.

    Construction policy, source identity, and task intent are supplied explicitly.
    No example is shuffled, rendered, normalized, or relabelled by this operation.
    """
    try:
        if type(construction) is not StudyDatasetConstructionV1:
            raise TypeError("construction must be exactly StudyDatasetConstructionV1")
        if type(task_definitions) is not tuple or any(
            type(item) is not StudyTaskDefinitionV1 for item in task_definitions
        ):
            raise TypeError("task definitions must be an exact tuple of StudyTaskDefinitionV1")
        if type(assignments) is not tuple or any(
            type(item) is not StudyExampleAssignmentV1 for item in assignments
        ):
            raise TypeError("assignments must be an exact tuple of StudyExampleAssignmentV1")
        payload = StudyDatasetPayloadV1(
            protocol_declaration_payload_sha256=(
                build_four_arm_protocol_declaration_v1().payload_sha256
            ),
            purpose=purpose,
            construction=construction,
            task_definitions=task_definitions,
            manifest=validate_experiment_manifest_v1(manifest),
            assignments=assignments,
        )
        dataset = IrisStudyDatasetV1(
            schema_identity=StudyDatasetSchemaV1(
                artifact_type="iris.study_dataset",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                validation_scope="input_membership_and_exact_duplicate_checks_only",
            ),
            payload_sha256=domain_sha256(
                _PAYLOAD_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
        return _validate_integrity(dataset)
    except Exception:
        # Validation exceptions can include private prompts/targets in their input details.
        raise IrisStudyDatasetError("Iris study dataset could not be built") from None


def validate_study_dataset_v1(dataset: IrisStudyDatasetV1) -> IrisStudyDatasetV1:
    """Revalidate schema, both payload digests, protocol binding, membership, and size."""
    try:
        return _validate_integrity(dataset)
    except Exception:
        raise IrisStudyDatasetError("Iris study dataset could not be verified") from None


def project_study_experiment_manifest_v1(dataset: IrisStudyDatasetV1) -> IrisExperimentManifestV1:
    """Return the exact owned nested manifest accepted by the existing B/C/D runner.

    This preserves train, validation AND held-out test membership. It is not a
    checkpoint-selection or test-access policy, which the launch must enforce.
    """
    return validate_study_dataset_v1(dataset).payload.manifest


def select_study_examples_v1(
    dataset: IrisStudyDatasetV1,
    *,
    task_family: StudyTaskFamilyV1,
    split: ExperimentSplitV1,
) -> tuple[IrisStudyExampleV1, ...]:
    """Select one family/split in original manifest order, never exposing another split."""
    try:
        if type(task_family) is not str or task_family not in get_args(StudyTaskFamilyV1):
            raise ValueError("unknown study task family")
        if type(split) is not str or split not in get_args(ExperimentSplitV1):
            raise ValueError("unknown study split")
        owned = _validate_integrity(dataset)
        return tuple(
            IrisStudyExampleV1(example=example, assignment=assignment)
            for example, assignment in zip(
                owned.payload.manifest.payload.examples, owned.payload.assignments, strict=True
            )
            if example.split == split and assignment.task_family == task_family
        )
    except Exception:
        raise IrisStudyDatasetError("Iris study dataset examples could not be selected") from None


__all__ = [
    "IrisStudyExampleV1",
    "build_study_dataset_v1",
    "project_study_experiment_manifest_v1",
    "select_study_examples_v1",
    "validate_study_dataset_v1",
]
