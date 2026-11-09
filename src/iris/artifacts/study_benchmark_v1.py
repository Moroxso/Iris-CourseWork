"""Study inputs bound to executable-answer intent and private functional code cases."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris.artifacts.study_dataset_v1 import IrisStudyDatasetV1


class _BenchmarkModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class StudyExactTextRuleV1(_BenchmarkModel):
    kind: Literal["exact_text_v1"]
    reference: Literal["manifest_teacher_target"]
    normalization: Literal["strip_ascii_space_tab_cr_lf_only"]


class StudyPythonIntegerListCaseV1(_BenchmarkModel):
    inputs: tuple[Annotated[int, Field(strict=True, ge=-(2**31), le=2**31 - 1)], ...] = Field(
        max_length=1024
    )
    expected: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)


class StudyPythonIntegerListRuleV1(_BenchmarkModel):
    """Real Python function behavior, never a source-text matching rule or a sandbox claim."""

    kind: Literal["python_integer_list_function_v1"]
    language: Literal["python3"]
    entrypoint: Literal["solve"]
    source_extraction: Literal["raw_python_source_only"]
    execution_requirement: Literal["isolated_python_backend"]
    comparison: Literal["exact_integer_excluding_bool"]
    input_policy: Literal["input_list_unchanged"]
    aggregation: Literal["all_cases_pass"]
    cases: tuple[StudyPythonIntegerListCaseV1, ...] = Field(min_length=4, max_length=128)

    @model_validator(mode="after")
    def validate_case_uniqueness(self) -> Self:
        if len({case.inputs for case in self.cases}) != len(self.cases):
            raise ValueError("functional code cases must have unique argument lists")
        return self


StudyScoringRuleV1 = Annotated[
    StudyExactTextRuleV1 | StudyPythonIntegerListRuleV1, Field(discriminator="kind")
]


class StudyExampleScoringV1(_BenchmarkModel):
    example_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    rule: StudyScoringRuleV1


class StudyBenchmarkSchemaV1(_BenchmarkModel):
    artifact_type: Literal["iris.study_benchmark"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    content_policy: Literal["private_inputs_targets_and_code_cases"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("benchmark version must be an exact integer")
        return value


class StudyBenchmarkPayloadV1(_BenchmarkModel):
    dataset: IrisStudyDatasetV1
    scoring: tuple[StudyExampleScoringV1, ...] = Field(min_length=15, max_length=1_000_000)

    @model_validator(mode="after")
    def validate_scoring_membership(self) -> Self:
        assignments = self.dataset.payload.assignments
        if tuple(item.example_id for item in self.scoring) != tuple(
            item.example_id for item in assignments
        ):
            raise ValueError("scoring rules must match the exact ordered dataset membership")
        for example, assignment, scoring in zip(
            self.dataset.payload.manifest.payload.examples, assignments, self.scoring, strict=True
        ):
            is_code = assignment.task_family == "code_generation_and_verification"
            if is_code != isinstance(scoring.rule, StudyPythonIntegerListRuleV1):
                raise ValueError("code tasks require functional code cases, not exact text scoring")
            if not is_code and not example.target.strip(" \t\r\n"):
                raise ValueError("exact-text task reference must be nonempty after normalization")
        return self


class IrisStudyBenchmarkV1(_BenchmarkModel):
    schema_identity: StudyBenchmarkSchemaV1
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: StudyBenchmarkPayloadV1


__all__ = [
    "IrisStudyBenchmarkV1",
    "StudyBenchmarkPayloadV1",
    "StudyBenchmarkSchemaV1",
    "StudyExactTextRuleV1",
    "StudyExampleScoringV1",
    "StudyPythonIntegerListCaseV1",
    "StudyPythonIntegerListRuleV1",
    "StudyScoringRuleV1",
]
