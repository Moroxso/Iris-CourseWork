"""Task-labelled inputs and split membership, not a scientific launch approval."""

from hashlib import sha256
from re import fullmatch
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.experiment_v1 import IrisExperimentManifestV1
from iris.artifacts.study_v1 import StudyTaskFamilyV1

StudyDatasetPurposeV1 = Literal["operational_fixture", "study_candidate"]
StudyTaskMeasurementScopeV1 = Literal[
    "cross_fragment_information",
    "delayed_instruction_planning",
    "functional_code_generation",
    "within_prompt_goal_retention",
    "within_prompt_conflict_resolution_proxy",
]
_MEASUREMENT_SCOPES: dict[StudyTaskFamilyV1, StudyTaskMeasurementScopeV1] = {
    "split_information": "cross_fragment_information",
    "delayed_planning": "delayed_instruction_planning",
    "code_generation_and_verification": "functional_code_generation",
    "hidden_goal_retention": "within_prompt_goal_retention",
    "module_conflict": "within_prompt_conflict_resolution_proxy",
}


def _validate_identity(value: object) -> object:
    if type(value) is not str or fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
        raise ValueError("dataset identity must be 1-128 portable ASCII characters")
    return value


def _validate_digest(value: object) -> object:
    if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("dataset digest must be a lower-case hexadecimal SHA-256")
    return value


def _validate_description(value: object) -> object:
    if type(value) is not str or not value.strip() or "\0" in value:
        raise ValueError("dataset description must be nonblank Unicode without NUL")
    if len(value.encode("utf-8", errors="strict")) > 16 * 1024:
        raise ValueError("dataset description exceeds the 16 KiB UTF-8 safety limit")
    return value


class _StudyDatasetModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyDatasetSchemaV1(_StudyDatasetModel):
    artifact_type: Literal["iris.study_dataset"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    validation_scope: Literal["input_membership_and_exact_duplicate_checks_only"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("study dataset version must be an exact integer")
        return value


class StudyDatasetConstructionV1(_StudyDatasetModel):
    """Declared source identity; its hash does not authenticate source or task validity."""

    method: Literal["manual_curation", "deterministic_generator"]
    source_id: str
    source_revision: str
    source_sha256: str
    generation_seed: int | None = Field(ge=0, le=2**63 - 1, strict=True)
    split_policy: Literal["preassigned_group_disjoint_v1"]
    split_policy_description: str

    _identity = field_validator("source_id", "source_revision", mode="before")(_validate_identity)
    _digest = field_validator("source_sha256", mode="before")(_validate_digest)
    _description = field_validator("split_policy_description", mode="before")(_validate_description)

    @model_validator(mode="after")
    def validate_generation_seed(self) -> Self:
        generated = self.method == "deterministic_generator"
        if generated != (self.generation_seed is not None):
            raise ValueError("only deterministic generation requires a generation seed")
        return self


class StudyTaskDefinitionV1(_StudyDatasetModel):
    """Human-auditable task intent; executable answer scoring is a separate contract."""

    task_family: StudyTaskFamilyV1
    measurement_scope: StudyTaskMeasurementScopeV1
    description: str
    target_semantics: str

    _description = field_validator("description", "target_semantics", mode="before")(
        _validate_description
    )

    @model_validator(mode="after")
    def validate_measurement_scope(self) -> Self:
        if self.measurement_scope != _MEASUREMENT_SCOPES[self.task_family]:
            raise ValueError("task measurement scope does not match its family")
        return self


class StudyExampleAssignmentV1(_StudyDatasetModel):
    example_id: str
    task_family: StudyTaskFamilyV1
    source_item_id: str
    leakage_group_id: str

    _identity = field_validator("example_id", "source_item_id", "leakage_group_id", mode="before")(
        _validate_identity
    )


class StudyDatasetPayloadV1(_StudyDatasetModel):
    protocol_declaration_payload_sha256: str
    purpose: StudyDatasetPurposeV1
    construction: StudyDatasetConstructionV1
    task_definitions: tuple[StudyTaskDefinitionV1, ...] = Field(min_length=5, max_length=5)
    manifest: IrisExperimentManifestV1
    assignments: tuple[StudyExampleAssignmentV1, ...] = Field(min_length=15, max_length=1_000_000)

    _digest = field_validator("protocol_declaration_payload_sha256", mode="before")(
        _validate_digest
    )

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        families = get_args(StudyTaskFamilyV1)
        if tuple(task.task_family for task in self.task_definitions) != families:
            raise ValueError("study dataset requires all five task definitions in protocol order")
        examples = self.manifest.payload.examples
        if tuple(item.example_id for item in self.assignments) != tuple(
            item.example_id for item in examples
        ):
            raise ValueError("study assignments must match exact manifest example order")
        coverage: set[tuple[str, str]] = set()
        group_splits: dict[str, str] = {}
        source_groups: dict[str, str] = {}
        prompt_splits: dict[bytes, str] = {}
        for example, assignment in zip(examples, self.assignments, strict=True):
            coverage.add((assignment.task_family, example.split))
            if group_splits.setdefault(assignment.leakage_group_id, example.split) != example.split:
                raise ValueError("a leakage group must not cross dataset splits")
            if (
                source_groups.setdefault(assignment.source_item_id, assignment.leakage_group_id)
                != assignment.leakage_group_id
            ):
                raise ValueError("a source item must belong to exactly one leakage group")
            prompt_hash = sha256(example.prompt.encode("utf-8")).digest()
            if prompt_splits.setdefault(prompt_hash, example.split) != example.split:
                raise ValueError("an exact prompt must not occur in different dataset splits")
        expected_coverage = {
            (family, split) for family in families for split in ("train", "validation", "test")
        }
        if coverage != expected_coverage:
            raise ValueError("every task family needs nonempty train, validation, and test splits")
        return self


class IrisStudyDatasetV1(_StudyDatasetModel):
    schema_identity: StudyDatasetSchemaV1
    payload_sha256: str
    payload: StudyDatasetPayloadV1

    _digest = field_validator("payload_sha256", mode="before")(_validate_digest)


__all__ = [
    "IrisStudyDatasetV1",
    "StudyDatasetConstructionV1",
    "StudyDatasetPayloadV1",
    "StudyDatasetPurposeV1",
    "StudyDatasetSchemaV1",
    "StudyExampleAssignmentV1",
    "StudyTaskDefinitionV1",
    "StudyTaskMeasurementScopeV1",
]
