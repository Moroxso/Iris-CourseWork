"""Hash-only, request/checkpoint-bound exact-text answer evaluation metadata."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v4 import RequestArmCheckpointTransferV4
from iris.artifacts.study_benchmark_v1 import StudyExactTextRuleV1
from iris.artifacts.study_v1 import StudyArmIdV1, StudyTaskFamilyV1

type _Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyEvaluationSchemaV1(_EvaluationModel):
    artifact_type: Literal["iris.study_evaluation"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_metadata_and_hashes_only"]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("evaluation version must be an exact integer")
        return value


class StudyEvaluationInputBindingV1(_EvaluationModel):
    preflight_sha256: _Sha256
    tokenizer_semantics_sha256: _Sha256
    transformers_version: str = Field(min_length=1, max_length=128)
    tokenizers_version: str = Field(min_length=1, max_length=128)
    terminal_token_id: int = Field(strict=True, ge=0)
    corpus_examples: int = Field(strict=True, ge=15)
    prompt_tokens: int = Field(strict=True, ge=1)
    target_tokens_including_terminal: int = Field(strict=True, ge=2)
    prompt_token_ids_sha256: _Sha256
    target_token_ids_sha256: _Sha256
    request_prompt_input_ids_sha256: _Sha256
    request_prompt_attention_mask_sha256: _Sha256
    scope: Literal["all_benchmark_inputs_fit_and_selected_request_prompt_matches"]


class StudyEvaluationPayloadV1(_EvaluationModel):
    benchmark_payload_sha256: _Sha256
    dataset_payload_sha256: _Sha256
    manifest_payload_sha256: _Sha256
    protocol_declaration_payload_sha256: _Sha256
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: StudyTaskFamilyV1
    split: ExperimentSplitV1
    request_payload_sha256: _Sha256
    configuration_sha256: _Sha256
    execution_profile_set_sha256: _Sha256
    arm_id: StudyArmIdV1
    checkpoint_transfer: RequestArmCheckpointTransferV4 | None
    input_binding: StudyEvaluationInputBindingV1
    scoring_rule: StudyExactTextRuleV1
    output_sha256: _Sha256
    output_byte_length: int = Field(strict=True, ge=0)
    outcome: Literal["correct", "incorrect", "invalid_output"]
    verification_scope: Literal["bindings_and_recomputed_text_score_not_execution_attestation"]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def exact_arm_and_scoring_scope(self) -> Self:
        if self.task_family == "code_generation_and_verification":
            raise ValueError("functional Python tasks cannot acquire an exact-text evaluation")
        transfer = self.checkpoint_transfer
        if self.arm_id == "A":
            if transfer is not None:
                raise ValueError("arm A cannot carry a checkpoint transfer")
        elif transfer is None or transfer.arm_id != self.arm_id:
            raise ValueError("trainable arms require their exact checkpoint transfer")
        return self


class IrisStudyEvaluationV1(_EvaluationModel):
    schema_identity: StudyEvaluationSchemaV1
    payload_sha256: _Sha256
    payload: StudyEvaluationPayloadV1


__all__ = [
    "IrisStudyEvaluationV1",
    "StudyEvaluationInputBindingV1",
    "StudyEvaluationPayloadV1",
    "StudyEvaluationSchemaV1",
]
