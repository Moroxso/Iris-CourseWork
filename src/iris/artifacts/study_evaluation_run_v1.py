"""Immutable complete-split evaluation plan and metadata-only complete aggregate."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v4 import RequestArmCheckpointTransferV4
from iris.artifacts.study_v1 import StudyArmIdV1, StudyTaskFamilyV1

type _Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _RunModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyEvaluationRunSchemaV1(_RunModel):
    artifact_type: Literal["iris.study_evaluation_run", "iris.study_evaluation_summary"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_metadata_and_hashes_only"]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("evaluation-run version must be an exact integer")
        return value


class StudyEvaluationPlannedAnswerV1(_RunModel):
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: StudyTaskFamilyV1
    request_payload_sha256: _Sha256
    output_sha256: _Sha256
    output_byte_length: int = Field(strict=True, ge=0, le=64 * 1024 * 1024)


class StudyEvaluationRunPayloadV1(_RunModel):
    generation_payload_sha256: _Sha256
    benchmark_payload_sha256: _Sha256
    dataset_payload_sha256: _Sha256
    manifest_payload_sha256: _Sha256
    protocol_declaration_payload_sha256: _Sha256
    configuration_sha256: _Sha256
    execution_profile_set_sha256: _Sha256
    generation_input_preflight_sha256: _Sha256
    terminal_token_id: int = Field(strict=True, ge=0)
    arm_id: StudyArmIdV1
    split: ExperimentSplitV1
    checkpoint_transfer: RequestArmCheckpointTransferV4 | None
    code_policy: PythonSandboxPolicyV1
    max_storage_bytes: int = Field(strict=True, ge=4096, le=20 * 1024**3)
    answers: tuple[StudyEvaluationPlannedAnswerV1, ...] = Field(min_length=5, max_length=100000)
    membership_policy: Literal["all_generated_answers_in_manifest_order"]
    recovery_policy: Literal["verify_committed_prefix_before_executing_missing_answers"]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def exact_membership_and_arm(self) -> Self:
        if len({item.example_id for item in self.answers}) != len(self.answers):
            raise ValueError("evaluation plan requires unique examples")
        if len({item.task_family for item in self.answers}) != 5:
            raise ValueError("evaluation plan must retain all five task families")
        if self.arm_id == "A":
            if self.checkpoint_transfer is not None:
                raise ValueError("A cannot have a checkpoint transfer")
        elif self.checkpoint_transfer is None or self.checkpoint_transfer.arm_id != self.arm_id:
            raise ValueError("trainable arms require exact checkpoint transfer")
        return self


class IrisStudyEvaluationRunV1(_RunModel):
    schema_identity: StudyEvaluationRunSchemaV1
    payload_sha256: _Sha256
    payload: StudyEvaluationRunPayloadV1

    @model_validator(mode="after")
    def exact_artifact_type(self) -> Self:
        if self.schema_identity.artifact_type != "iris.study_evaluation_run":
            raise ValueError("wrong evaluation run artifact type")
        return self


class StudyEvaluationResultReferenceV1(_RunModel):
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: StudyTaskFamilyV1
    artifact_type: Literal["iris.study_evaluation", "iris.study_code_evaluation"]
    evaluation_payload_sha256: _Sha256
    file_sha256: _Sha256
    file_byte_length: int = Field(strict=True, ge=1, le=12 * 1024 * 1024)
    outcome: Literal["correct", "incorrect", "invalid_output"]

    @model_validator(mode="after")
    def exact_family(self) -> Self:
        if (self.task_family == "code_generation_and_verification") != (
            self.artifact_type == "iris.study_code_evaluation"
        ):
            raise ValueError("evaluation family and artifact type disagree")
        return self


class StudyEvaluationCountsV1(_RunModel):
    total: int = Field(strict=True, ge=1, le=100000)
    correct: int = Field(strict=True, ge=0, le=100000)
    incorrect: int = Field(strict=True, ge=0, le=100000)
    invalid_output: int = Field(strict=True, ge=0, le=100000)

    @model_validator(mode="after")
    def complete_denominator(self) -> Self:
        if self.total != self.correct + self.incorrect + self.invalid_output:
            raise ValueError("all evaluated outputs must remain in the denominator")
        return self


class StudyEvaluationFamilyCountsV1(StudyEvaluationCountsV1):
    task_family: StudyTaskFamilyV1


class StudyEvaluationSummaryPayloadV1(_RunModel):
    run_payload_sha256: _Sha256
    results: tuple[StudyEvaluationResultReferenceV1, ...] = Field(min_length=5, max_length=100000)
    overall: StudyEvaluationCountsV1
    by_family: tuple[StudyEvaluationFamilyCountsV1, ...] = Field(min_length=5, max_length=5)
    status: Literal["complete_all_generated_answers_evaluated"]
    aggregation_policy: Literal["unweighted_example_counts_including_invalid_outputs"]
    verification_scope: Literal["bindings_and_recomputed_scores_not_execution_attestation"]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def recomputed_counts(self) -> Self:
        if len({item.example_id for item in self.results}) != len(self.results):
            raise ValueError("summary results must be unique")
        families = tuple(dict.fromkeys(item.task_family for item in self.results))
        if tuple(item.task_family for item in self.by_family) != families:
            raise ValueError("summary families must follow first occurrence in manifest order")
        for group, items in [
            (self.overall, self.results),
            *[
                (
                    group,
                    tuple(item for item in self.results if item.task_family == group.task_family),
                )
                for group in self.by_family
            ],
        ]:
            if group.total != len(items) or any(
                getattr(group, outcome) != sum(item.outcome == outcome for item in items)
                for outcome in ("correct", "incorrect", "invalid_output")
            ):
                raise ValueError("summary counts do not match every referenced result")
        return self


class IrisStudyEvaluationSummaryV1(_RunModel):
    schema_identity: StudyEvaluationRunSchemaV1
    payload_sha256: _Sha256
    payload: StudyEvaluationSummaryPayloadV1

    @model_validator(mode="after")
    def exact_artifact_type(self) -> Self:
        if self.schema_identity.artifact_type != "iris.study_evaluation_summary":
            raise ValueError("wrong evaluation summary artifact type")
        return self


__all__ = [
    "IrisStudyEvaluationRunV1",
    "IrisStudyEvaluationSummaryV1",
    "StudyEvaluationCountsV1",
    "StudyEvaluationFamilyCountsV1",
    "StudyEvaluationPlannedAnswerV1",
    "StudyEvaluationResultReferenceV1",
    "StudyEvaluationRunPayloadV1",
    "StudyEvaluationRunSchemaV1",
    "StudyEvaluationSummaryPayloadV1",
]
