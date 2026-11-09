"""Private held-out generations, separate from legacy five-family study records."""

from math import fsum, isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class ScientificDecodingPolicyV1(_Strict):
    strategy: Literal["greedy", "sampling"]
    seed: int = Field(ge=0, le=2**63 - 1)
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def canonical_greedy(self) -> Self:
        if self.strategy == "greedy" and (self.temperature, self.top_p) != (1.0, 1.0):
            raise ValueError("Greedy policy cannot declare unused sampling transforms")
        return self


def scientific_row_sampling_seed_v1(seed: int, example_id: str, prompt_sha256: str) -> int:
    """Paired across arms/order; reference labels never influence sampling RNG."""
    digest = domain_sha256(
        "iris/scientific-generation-row-seed/v1",
        canonical_json_bytes([seed, example_id, prompt_sha256]),
    )
    return int(digest[:16], 16) & (2**63 - 1)


class ScientificGeneratedRowV1(_Strict):
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


class ScientificGenerationPayloadV1(_Strict):
    kind: Literal["iris-scientific-generation-v1"] = "iris-scientific-generation-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["scientific", "smoke"]
    data_scope: Literal["candidate_not_launch_approval"] = "candidate_not_launch_approval"
    execution_sha256: Digest
    terminal_event_sha256: Digest
    best_validation_event_sha256: Digest
    best_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1)
    terminal_step: int = Field(ge=1)
    inference_sha256: Digest
    teacher_configuration_sha256: Digest
    backbone_runtime_sha256: Digest
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    decoding: ScientificDecodingPolicyV1
    max_new_tokens: int = Field(ge=1, le=65536)
    vocabulary_size: int = Field(ge=1)
    eos_token_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1)
    rows: tuple[ScientificGeneratedRowV1, ...] = Field(min_length=1, max_length=5000, repr=False)

    @model_validator(mode="after")
    def check_capture(self) -> Self:
        if self.selected_step > self.terminal_step:
            raise ValueError("Selected checkpoint is later than terminal training")
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids) or any(
            token >= self.vocabulary_size for token in self.eos_token_ids
        ):
            raise ValueError("Invalid EOS inventory")
        if len({row.example_id for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate held-out rows")
        if self.purpose == "scientific":
            counts = {
                task: sum(row.task_type == task for row in self.rows)
                for task in (
                    "split_spec_repair",
                    "delayed_goal",
                    "conflict_resolution",
                    "next_tool_prediction",
                )
            }
            if counts != dict(zip(counts, (2000, 1000, 1000, 1000), strict=True)):
                raise ValueError("Scientific generation requires the complete 5000-row split")
        for row in self.rows:
            expected_seed = (
                scientific_row_sampling_seed_v1(
                    self.decoding.seed, row.example_id, row.prompt_sha256
                )
                if self.decoding.strategy == "sampling"
                else None
            )
            if row.sampling_seed != expected_seed:
                raise ValueError("Row sampling seed differs from declared input-only policy")
            if len(row.token_ids) > self.max_new_tokens or any(
                token >= self.vocabulary_size for token in row.token_ids
            ):
                raise ValueError("Generated tokens exceed declared limits")
            if any(token in self.eos_token_ids for token in row.token_ids[:-1]):
                raise ValueError("Generation continued after EOS")
            if row.stop_reason == "stop_token":
                if (
                    row.matched_stop_token_id != row.token_ids[-1]
                    or row.token_ids[-1] not in self.eos_token_ids
                ):
                    raise ValueError("Reported EOS does not match the final token")
            elif (
                row.matched_stop_token_id is not None
                or len(row.token_ids) != self.max_new_tokens
                or row.token_ids[-1] in self.eos_token_ids
            ):
                raise ValueError("Length termination contradicts generated tokens")
            row.output_text.encode("utf-8", errors="strict")
        try:
            total = fsum(row.target_nll_sum for row in self.rows)
        except OverflowError:
            raise ValueError("Held-out NLL aggregation overflowed") from None
        if not isfinite(total):
            raise ValueError("Held-out NLL aggregation is nonfinite")
        return self

    @property
    def target_token_count(self) -> int:
        return sum(row.target_token_count for row in self.rows)

    @property
    def target_nll_per_token(self) -> float:
        return fsum(row.target_nll_sum for row in self.rows) / self.target_token_count


class IrisScientificGenerationV1(_Strict):
    sha256: Digest
    payload: ScientificGenerationPayloadV1 = Field(repr=False)


__all__ = [
    "IrisScientificGenerationV1",
    "ScientificDecodingPolicyV1",
    "ScientificGeneratedRowV1",
    "ScientificGenerationPayloadV1",
    "scientific_row_sampling_seed_v1",
]
