"""Strict local plaintext experiment-example manifest schema."""

from re import fullmatch
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ExperimentSplitV1 = Literal["train", "validation", "test"]
_IDENTITY_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_MAX_TEXT_UTF8_BYTES = 1024 * 1024


class _ExperimentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ExperimentManifestSchemaV1(_ExperimentModel):
    artifact_type: Literal["iris.experiment_examples"]
    version: Literal[1]
    canonicalization: Literal["iris-json-v1"]
    hash_algorithm: Literal["sha256"]
    text_encoding: Literal["utf-8"]
    text_normalization: Literal["none"]
    iteration_order: Literal["manifest_order_within_split"]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            message = "experiment manifest version must be an exact integer"
            raise ValueError(message)
        return value


class ExperimentExampleV1(_ExperimentModel):
    example_id: str
    split: ExperimentSplitV1
    prompt: str
    target: str

    @field_validator("example_id", mode="before")
    @classmethod
    def validate_example_id(cls, value: object) -> object:
        if type(value) is not str or fullmatch(_IDENTITY_PATTERN, value) is None:
            message = "example_id must be a 1-128 character portable ASCII identity"
            raise ValueError(message)
        return value

    @field_validator("prompt", "target", mode="before")
    @classmethod
    def validate_text(cls, value: object) -> object:
        if type(value) is not str or not value.strip() or "\0" in value:
            message = "experiment text must be nonblank exact Unicode without NUL"
            raise ValueError(message)
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "experiment text must contain valid UTF-8 scalar values"
            raise ValueError(message) from error
        if len(encoded) > _MAX_TEXT_UTF8_BYTES:
            message = "experiment text exceeds the 1 MiB UTF-8 safety limit"
            raise ValueError(message)
        return value


class ExperimentSplitCountsV1(_ExperimentModel):
    train: int = Field(ge=1, strict=True)
    validation: int = Field(ge=1, strict=True)
    test: int = Field(ge=0, strict=True)


class ExperimentManifestPayloadV1(_ExperimentModel):
    dataset_id: str
    split_counts: ExperimentSplitCountsV1
    examples: tuple[ExperimentExampleV1, ...] = Field(min_length=2, max_length=1_000_000)

    @field_validator("dataset_id", mode="before")
    @classmethod
    def validate_dataset_id(cls, value: object) -> object:
        if type(value) is not str or fullmatch(_IDENTITY_PATTERN, value) is None:
            message = "dataset_id must be a 1-128 character portable ASCII identity"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def validate_examples(self) -> Self:
        identities = tuple(example.example_id for example in self.examples)
        if len(set(identities)) != len(identities):
            message = "experiment example IDs must be globally unique"
            raise ValueError(message)
        observed = {
            split: sum(example.split == split for example in self.examples)
            for split in ("train", "validation", "test")
        }
        if observed != self.split_counts.model_dump(mode="python"):
            message = "experiment split_counts do not match example membership"
            raise ValueError(message)
        return self


class IrisExperimentManifestV1(_ExperimentModel):
    schema_identity: ExperimentManifestSchemaV1
    payload_sha256: str
    payload: ExperimentManifestPayloadV1

    @field_validator("payload_sha256", mode="before")
    @classmethod
    def validate_payload_sha256(cls, value: object) -> object:
        if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
            message = "payload_sha256 must be 64 lower-case hexadecimal characters"
            raise ValueError(message)
        return value


__all__ = [
    "ExperimentExampleV1",
    "ExperimentManifestPayloadV1",
    "ExperimentManifestSchemaV1",
    "ExperimentSplitCountsV1",
    "ExperimentSplitV1",
    "IrisExperimentManifestV1",
]
