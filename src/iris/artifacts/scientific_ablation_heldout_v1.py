"""Distinct redacted mode-bound test preflight and private per-row evidence."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_generation_v1 import (
    ScientificDecodingPolicyV1,
    scientific_row_sampling_seed_v1,
)
from iris.data.schemas import SyntheticTask


class ScientificAblationHeldoutInputPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-heldout-input-v1"] = (
        "iris-scientific-ablation-heldout-input-v1"
    )
    scope: Literal["preflight_not_capture_scoring_or_launch_approval"] = (
        "preflight_not_capture_scoring_or_launch_approval"
    )
    reconstruction_sha256: Digest
    source_sha256: Digest
    ablation_policy_sha256: Digest
    selected_checkpoint_sha256: Digest
    purpose: Literal["scientific", "smoke"]
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    example_count: int = Field(ge=1, le=5000)
    target_token_count: int = Field(ge=1, le=2**48)
    decoding: ScientificDecodingPolicyV1
    max_new_tokens: int = Field(ge=1, le=65536)
    vocabulary_size: int = Field(ge=1)
    eos_token_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_input(self) -> Self:
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids) or any(
            token >= self.vocabulary_size for token in self.eos_token_ids
        ):
            raise ValueError("Invalid actual EOS inventory")
        if self.purpose == "scientific" and self.example_count != 5000:
            raise ValueError("Scientific held-out input requires the complete split")
        return self


class IrisScientificAblationHeldoutInputV1(_Strict):
    sha256: Digest
    payload: ScientificAblationHeldoutInputPayloadV1


class ScientificAblationHeldoutRowPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-heldout-row-v1"] = (
        "iris-scientific-ablation-heldout-row-v1"
    )
    scope: Literal["actual_row_not_durable_ack_or_complete_evaluation"] = (
        "actual_row_not_durable_ack_or_complete_evaluation"
    )
    input: IrisScientificAblationHeldoutInputV1
    row_index: int = Field(ge=0, le=4999)
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    task_type: SyntheticTask
    sampling_seed: int | None = Field(ge=0, le=2**63 - 1)
    output_text: str = Field(max_length=1048576, repr=False)
    token_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1, max_length=65536)
    stop_reason: Literal["stop_token", "max_new_tokens"]
    matched_stop_token_id: int | None = Field(ge=0)
    target_token_count: int = Field(ge=1, le=65536)
    target_nll_sum: float = Field(ge=0.0)
    teacher_invocation_sha256: Digest
    teacher_target_sha256: Digest
    scientific_state_sha256: Digest
    teacher_configuration_sha256: Digest
    request_receipt_sha256: Digest
    request_receipt_json: str = Field(min_length=1, max_length=65536, repr=False)

    @model_validator(mode="after")
    def complete_row(self) -> Self:
        p = self.input.payload
        expected_seed = (
            scientific_row_sampling_seed_v1(p.decoding.seed, self.example_id, self.prompt_sha256)
            if p.decoding.strategy == "sampling"
            else None
        )
        if self.row_index >= p.example_count or self.sampling_seed != expected_seed:
            raise ValueError("Row membership/index or input-only sampling seed differs")
        if len(self.token_ids) > p.max_new_tokens or any(
            token >= p.vocabulary_size for token in self.token_ids
        ):
            raise ValueError("Generated tokens exceed actual limits")
        if any(token in p.eos_token_ids for token in self.token_ids[:-1]):
            raise ValueError("Generation continued after actual EOS")
        if self.stop_reason == "stop_token":
            if (
                self.matched_stop_token_id != self.token_ids[-1]
                or self.token_ids[-1] not in p.eos_token_ids
            ):
                raise ValueError("EOS receipt contradicts actual final token")
        elif (
            self.matched_stop_token_id is not None
            or len(self.token_ids) != p.max_new_tokens
            or self.token_ids[-1] in p.eos_token_ids
        ):
            raise ValueError("Full-cap termination contradicts actual tokens")
        self.output_text.encode("utf-8", errors="strict")
        return self


class IrisScientificAblationHeldoutRowV1(_Strict):
    sha256: Digest
    payload: ScientificAblationHeldoutRowPayloadV1 = Field(repr=False)


__all__ = [
    "IrisScientificAblationHeldoutInputV1",
    "IrisScientificAblationHeldoutRowV1",
    "ScientificAblationHeldoutInputPayloadV1",
    "ScientificAblationHeldoutRowPayloadV1",
]
