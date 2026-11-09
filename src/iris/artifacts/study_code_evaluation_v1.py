"""Private bounded worker observations and a hash-only request-bound code receipt."""

import base64
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v4 import RequestArmCheckpointTransferV4
from iris.artifacts.study_evaluation_v1 import StudyEvaluationInputBindingV1
from iris.artifacts.study_v1 import StudyArmIdV1

type _Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_MAX_STREAM_BYTES = 8 * 1024 * 1024


class _CodeEvaluationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyCodeEvaluationSchemaV1(_CodeEvaluationModel):
    artifact_type: Literal["iris.study_code_evaluation"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_worker_observations_and_hash_only_receipt"]

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("code evaluation version must be an exact integer")
        return value


class StudyCodeStreamV1(_CodeEvaluationModel):
    """Private bytes, NOT encryption, trusted text, or executable serialization."""

    data_base64: str = Field(max_length=4 * ((_MAX_STREAM_BYTES + 2) // 3))
    byte_length: int = Field(strict=True, ge=0, le=_MAX_STREAM_BYTES)
    sha256: _Sha256

    @model_validator(mode="after")
    def canonical_bounded_bytes(self) -> Self:
        if len(self.data_base64) != 4 * ((self.byte_length + 2) // 3):
            raise ValueError("stream encoded length disagrees")
        raw = base64.b64decode(self.data_base64, validate=True)
        if (
            len(raw) != self.byte_length
            or base64.b64encode(raw).decode("ascii") != self.data_base64
            or sha256(raw).hexdigest() != self.sha256
        ):
            raise ValueError("stream bytes or digest disagree")
        return self


class StudyCodeObservationV1(_CodeEvaluationModel):
    policy: PythonSandboxPolicyV1
    policy_sha256: _Sha256
    worker_sha256: _Sha256
    image_reference: Literal[
        "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
    ]
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_version: Literal["release-20260831.0"]
    docker_version: Literal["29.8.0"]
    kernel_release: str = Field(min_length=1, max_length=256)
    submitted_input_sha256: _Sha256
    stdout: StudyCodeStreamV1
    stderr: StudyCodeStreamV1
    transport_returncode: int = Field(strict=True, ge=-(2**31), le=2**32 - 1)
    transport_termination: Literal["exited", "timeout", "output_limit"]
    container_exit_code: int = Field(strict=True, ge=0, le=255)
    oom_killed: bool

    @model_validator(mode="after")
    def bounded_observation(self) -> Self:
        if self.stdout.byte_length + self.stderr.byte_length > self.policy.max_output_bytes:
            raise ValueError("combined capture exceeds explicit policy")
        if self.container_exit_code == 80:
            raise ValueError("reserved worker failure cannot acquire a receipt")
        return self


class StudyCodeReceiptV1(_CodeEvaluationModel):
    benchmark_payload_sha256: _Sha256
    dataset_payload_sha256: _Sha256
    manifest_payload_sha256: _Sha256
    protocol_declaration_payload_sha256: _Sha256
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    task_family: Literal["code_generation_and_verification"]
    split: ExperimentSplitV1
    request_payload_sha256: _Sha256
    configuration_sha256: _Sha256
    execution_profile_set_sha256: _Sha256
    arm_id: StudyArmIdV1
    checkpoint_transfer: RequestArmCheckpointTransferV4 | None
    input_binding: StudyEvaluationInputBindingV1
    scoring_rule_sha256: _Sha256
    observation_sha256: _Sha256
    output_sha256: _Sha256
    output_byte_length: int = Field(strict=True, ge=0, le=64 * 1024 * 1024)
    source_submitted: bool
    case_count: int = Field(strict=True, ge=4, le=128)
    passed_case_count: int = Field(strict=True, ge=0, le=128)
    outcome: Literal["correct", "incorrect", "invalid_output"]
    reason: Literal[
        "all_cases_pass",
        "case_mismatch",
        "invalid_source",
        "syntax_error",
        "missing_entrypoint",
        "load_exception",
        "protocol_error",
        "execution_terminated",
        "memory_limit",
        "output_limit",
        "wall_time_limit",
    ]
    verification_scope: Literal[
        "bindings_and_recomputed_recorded_observations_not_execution_attestation"
    ]
    scientific_launch_status: Literal["not_assessed"]

    @model_validator(mode="after")
    def exact_arm_and_score(self) -> Self:
        transfer = self.checkpoint_transfer
        if self.arm_id == "A":
            if transfer is not None:
                raise ValueError("arm A cannot carry a checkpoint transfer")
        elif transfer is None or transfer.arm_id != self.arm_id:
            raise ValueError("trainable arms require their exact checkpoint transfer")
        invalid = self.reason in ("invalid_source", "syntax_error", "missing_entrypoint")
        if (
            self.passed_case_count > self.case_count
            or (self.outcome == "correct") != (self.reason == "all_cases_pass")
            or (self.outcome == "correct") != (self.passed_case_count == self.case_count)
            or (self.outcome == "invalid_output") != invalid
            or (self.reason not in ("case_mismatch", "all_cases_pass") and self.passed_case_count)
            or (self.reason == "invalid_source" and self.source_submitted)
            or (self.reason in ("syntax_error", "missing_entrypoint") and not self.source_submitted)
        ):
            raise ValueError("score counts, status or source scope disagree")
        return self


class StudyCodeEvaluationPayloadV1(_CodeEvaluationModel):
    receipt: StudyCodeReceiptV1
    observation: StudyCodeObservationV1


class IrisStudyCodeEvaluationV1(_CodeEvaluationModel):
    schema_identity: StudyCodeEvaluationSchemaV1
    payload_sha256: _Sha256
    payload: StudyCodeEvaluationPayloadV1


__all__ = [
    "IrisStudyCodeEvaluationV1",
    "StudyCodeEvaluationPayloadV1",
    "StudyCodeEvaluationSchemaV1",
    "StudyCodeObservationV1",
    "StudyCodeReceiptV1",
    "StudyCodeStreamV1",
]
