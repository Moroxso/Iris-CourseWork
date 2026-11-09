"""Private immutable generation plans and durable raw-answer/request records."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.artifacts.request_v4 import IrisRequestArtifactEnvelopeV4
from iris.artifacts.study_generation_v1 import StudyGeneratedAnswerV1
from iris.artifacts.study_v1 import StudyArmIdV1, StudyTaskFamilyV1
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256

type _Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _RunModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyGenerationRunSchemaV1(_RunModel):
    artifact_type: Literal["iris.study_generation_run", "iris.study_generation_answer"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_generation_plan_and_answer_request_envelopes"]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("generation-run version must be an exact integer")
        return value


class StudyGenerationCheckpointV1(_RunModel):
    arm_id: Literal["B", "C", "D"]
    checkpoint_schema_version: int = Field(strict=True, ge=1, le=3)
    payload_sha256: _Sha256
    state_file_sha256: _Sha256
    sidecar_state_sha256: _Sha256
    completed_steps: int = Field(strict=True, ge=0, le=2**24)

    @model_validator(mode="after")
    def exact_arm_version(self) -> Self:
        if self.checkpoint_schema_version != {"B": 2, "C": 3, "D": 1}[self.arm_id]:
            raise ValueError("generation checkpoint arm and schema disagree")
        return self


class StudyGenerationPlannedAnswerV1(_RunModel):
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: StudyTaskFamilyV1
    decoding: DecodingProvenanceV1


class StudyGenerationRunPayloadV1(_RunModel):
    benchmark_payload_sha256: _Sha256
    dataset_payload_sha256: _Sha256
    manifest_payload_sha256: _Sha256
    protocol_declaration_payload_sha256: _Sha256
    configuration: IrisBootstrapConfig
    configuration_sha256: _Sha256
    execution_profile_set_sha256: _Sha256
    input_preflight_sha256: _Sha256
    terminal_token_id: int = Field(strict=True, ge=0)
    arm_id: StudyArmIdV1
    split: ExperimentSplitV1
    checkpoint_identity: StudyGenerationCheckpointV1 | None
    directory: str
    request_root: str
    max_artifact_bytes: int = Field(strict=True, ge=1024, le=64 * 1024**2)
    max_record_bytes: int = Field(strict=True, ge=4096, le=64 * 1024**2)
    max_storage_bytes: int = Field(strict=True, ge=4096, le=20 * 1024**3)
    answers: tuple[StudyGenerationPlannedAnswerV1, ...] = Field(min_length=5, max_length=100000)
    membership_policy: Literal["all_examples_in_selected_split_in_manifest_order"]
    recovery_policy: Literal["verify_prefix_and_pending_before_generating_missing_suffix"]
    output_policy: Literal["generation_v1_in_managed_directory"]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def bound_plan(self) -> Self:
        if self.configuration_sha256 != domain_sha256(
            "iris/configuration/v1",
            canonical_json_bytes(self.configuration.model_dump(mode="json")),
        ):
            raise ValueError("run configuration hash disagrees")
        for raw in (self.directory, self.request_root):
            path = Path(raw)
            if not path.is_absolute() or not path.name or ".." in path.parts:
                raise ValueError("generation plan paths must be explicit absolute non-root paths")
        if Path(self.configuration.runtime.output_dir) != Path(self.request_root):
            raise ValueError("request root differs from configuration")
        if (
            Path(self.directory) == Path(self.request_root)
            or Path(self.directory) in Path(self.request_root).parents
        ):
            raise ValueError("external request storage must not be inside the managed run")
        if len({a.example_id for a in self.answers}) != len(self.answers):
            raise ValueError("generation plan examples must be unique")
        if len({a.task_family for a in self.answers}) != 5:
            raise ValueError("generation plan must retain all five families")
        first = self.answers[0].decoding.model_dump(exclude={"sampling_seed"})
        if any(
            a.decoding.allow_download or a.decoding.model_dump(exclude={"sampling_seed"}) != first
            for a in self.answers
        ):
            raise ValueError("only explicit sampling seeds may vary; downloads are forbidden")
        if self.arm_id == "A":
            if self.checkpoint_identity is not None:
                raise ValueError("A cannot have a checkpoint")
        elif self.checkpoint_identity is None or self.checkpoint_identity.arm_id != self.arm_id:
            raise ValueError("trainable arms require an exact checkpoint identity")
        return self


class IrisStudyGenerationRunV1(_RunModel):
    schema_identity: StudyGenerationRunSchemaV1
    payload_sha256: _Sha256
    payload: StudyGenerationRunPayloadV1

    @model_validator(mode="after")
    def exact_type(self) -> Self:
        if self.schema_identity.artifact_type != "iris.study_generation_run":
            raise ValueError("incorrect generation-run artifact type")
        return self


class StudyGenerationAnswerPayloadV1(_RunModel):
    run_payload_sha256: _Sha256
    index: int = Field(strict=True, ge=0, lt=100000)
    answer: StudyGeneratedAnswerV1
    request: IrisRequestArtifactEnvelopeV4

    @model_validator(mode="after")
    def exact_request_reference(self) -> Self:
        if self.answer.request_payload_sha256 != self.request.payload_sha256:
            raise ValueError("durable answer and embedded request disagree")
        return self


class IrisStudyGenerationAnswerV1(_RunModel):
    schema_identity: StudyGenerationRunSchemaV1
    payload_sha256: _Sha256
    payload: StudyGenerationAnswerPayloadV1

    @model_validator(mode="after")
    def exact_type(self) -> Self:
        if self.schema_identity.artifact_type != "iris.study_generation_answer":
            raise ValueError("incorrect durable-answer artifact type")
        return self


__all__ = [
    "IrisStudyGenerationAnswerV1",
    "IrisStudyGenerationRunV1",
    "StudyGenerationAnswerPayloadV1",
    "StudyGenerationCheckpointV1",
    "StudyGenerationPlannedAnswerV1",
    "StudyGenerationRunPayloadV1",
    "StudyGenerationRunSchemaV1",
]
