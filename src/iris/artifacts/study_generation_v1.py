"""Private, complete-split generation capture; not a scoring or launch artifact."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.artifacts.request_v4 import RequestArmCheckpointTransferV4
from iris.artifacts.study_v1 import StudyArmIdV1, StudyTaskFamilyV1
from iris.utils.hashing import bytes_sha256

type _Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _GenerationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyGenerationSchemaV1(_GenerationModel):
    artifact_type: Literal["iris.study_generation"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_generated_answers_and_metadata"]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("generation version must be an exact integer")
        return value


class StudyGeneratedAnswerV1(_GenerationModel):
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: StudyTaskFamilyV1
    request_payload_sha256: _Sha256
    decoding: DecodingProvenanceV1
    output_text: str
    output_sha256: _Sha256
    output_byte_length: int = Field(strict=True, ge=0, le=64 * 1024 * 1024)

    @model_validator(mode="after")
    def exact_private_output(self) -> Self:
        encoded = self.output_text.encode("utf-8", errors="strict")
        if self.output_byte_length != len(encoded) or self.output_sha256 != bytes_sha256(encoded):
            raise ValueError("generated output fingerprint disagrees with its exact bytes")
        if self.decoding.allow_download:
            raise ValueError("study generation is offline-only")
        return self


class StudyGenerationPayloadV1(_GenerationModel):
    benchmark_payload_sha256: _Sha256
    dataset_payload_sha256: _Sha256
    manifest_payload_sha256: _Sha256
    protocol_declaration_payload_sha256: _Sha256
    configuration_sha256: _Sha256
    execution_profile_set_sha256: _Sha256
    input_preflight_sha256: _Sha256
    terminal_token_id: int = Field(strict=True, ge=0)
    arm_id: StudyArmIdV1
    split: ExperimentSplitV1
    checkpoint_transfer: RequestArmCheckpointTransferV4 | None
    max_artifact_bytes: int = Field(strict=True, ge=1024, le=64 * 1024 * 1024)
    answers: tuple[StudyGeneratedAnswerV1, ...] = Field(min_length=5, max_length=100000)
    membership_policy: Literal["all_examples_in_selected_split_in_manifest_order"]
    status: Literal["complete_generation_not_scored"]
    verification_scope: Literal["bindings_not_execution_attestation"]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def unique_membership_and_arm(self) -> Self:
        if len({item.example_id for item in self.answers}) != len(self.answers):
            raise ValueError("generation example IDs must be unique")
        if len({item.task_family for item in self.answers}) != 5:
            raise ValueError("generation must retain all five families")
        transfer = self.checkpoint_transfer
        if self.arm_id == "A":
            if transfer is not None:
                raise ValueError("arm A forbids a checkpoint transfer")
        elif transfer is None or transfer.arm_id != self.arm_id:
            raise ValueError("trainable arms require an arm-exact checkpoint transfer")
        first = self.answers[0].decoding.model_dump(exclude={"sampling_seed"})
        if any(
            item.decoding.model_dump(exclude={"sampling_seed"}) != first for item in self.answers
        ):
            raise ValueError("only explicit per-example seeds may vary within one batch")
        return self


class IrisStudyGenerationV1(_GenerationModel):
    schema_identity: StudyGenerationSchemaV1
    payload_sha256: _Sha256
    payload: StudyGenerationPayloadV1


__all__ = [
    "IrisStudyGenerationV1",
    "StudyGeneratedAnswerV1",
    "StudyGenerationPayloadV1",
    "StudyGenerationSchemaV1",
]
