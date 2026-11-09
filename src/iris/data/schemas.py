"""Logical rows and structured answers from specification sections 12 and 14."""

import json
from hashlib import sha256
from itertools import pairwise
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from iris.utils.hashing import canonical_json_bytes

SyntheticTask = Literal[
    "split_spec_repair", "delayed_goal", "conflict_resolution", "next_tool_prediction"
]
SyntheticSplit = Literal["train", "validation", "test"]
SyntheticTemplate = Literal["iris-synthetic-01-v1", "iris-synthetic-01-v2"]
NextAction = Literal[
    "read_file", "search_symbol", "edit_file", "run_tests", "inspect_error", "finish"
]
FactValue = StrictStr | StrictInt | StrictBool


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False, revalidate_instances="always"
    )


class IrisStructuredResponse(_StrictModel):
    """One answer envelope; confidence is not an exact-match correctness target."""

    decision: Literal["patch", "answer", "tool", "conflict"]
    content: str = Field(min_length=1, max_length=262144)
    confidence: float = Field(ge=0.0, le=1.0)
    next_action: NextAction

    @field_validator("confidence", mode="before")
    @classmethod
    def exact_confidence(cls, value: object) -> object:
        if type(value) not in (int, float):
            raise ValueError("confidence must be a finite JSON number, not a boolean")
        return float(cast(int | float, value))

    @field_validator("content")
    @classmethod
    def valid_content(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("content must be nonblank and contain no NUL")
        value.encode("utf-8", errors="strict")
        return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_structured_response(text: str) -> IrisStructuredResponse:
    """Parse bounded JSON only; never execute code or accept fenced/prefixed text."""
    try:
        if type(text) is not str or len(text.encode("utf-8", errors="strict")) > 1048576:
            raise ValueError("invalid answer text")
        raw = json.loads(text, object_pairs_hook=_unique_object)
        return IrisStructuredResponse.model_validate(raw)
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise ValueError("Invalid Iris structured response") from None


class SyntheticPythonCase(_StrictModel):
    inputs: tuple[Annotated[int, Field(strict=True, ge=-(2**31), le=2**31 - 1)], ...] = Field(
        max_length=1024
    )
    expected: int = Field(ge=-(2**63), le=2**63 - 1)


class IrisExample(_StrictModel):
    """Private research row, including labels never supplied to inference as inputs.

    Specification fields are retained. Split/template/group identities and typed
    functional cases are explicit metadata extensions, not five-family aliases.
    """

    id: str = Field(pattern=r"^(train|validation|test)-[0-9]+$")
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)
    seed: int = Field(ge=0, le=2**63 - 1)
    prompt: str = Field(min_length=1, max_length=262144)
    expected_response: str = Field(min_length=1, max_length=262144)
    hidden_facts: dict[str, FactValue]
    required_modules: tuple[str, ...]
    distractor_count: int = Field(ge=0, le=128)
    executable: bool
    test_code: str | None
    timeout_seconds: Literal[5]
    split: SyntheticSplit
    template_id: SyntheticTemplate
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_cases: tuple[SyntheticPythonCase, ...]

    @model_validator(mode="after")
    def validate_relations(self) -> Self:
        if self.id != f"{self.split}-{self.seed}":
            raise ValueError("row ID must bind split and seed")
        expected_problem = sha256(
            canonical_json_bytes({"task": self.task_type, "facts": self.hidden_facts})
        ).hexdigest()
        if self.problem_sha256 != expected_problem:
            raise ValueError("problem identity does not match task facts")
        expected = parse_structured_response(self.expected_response)
        decisions = {
            "split_spec_repair": "patch",
            "delayed_goal": "answer",
            "conflict_resolution": "conflict",
            "next_tool_prediction": "tool",
        }
        if expected.decision != decisions[self.task_type]:
            raise ValueError("answer decision does not match task")
        is_code = self.task_type == "split_spec_repair"
        if self.executable != is_code or is_code != bool(self.code_cases):
            raise ValueError("functional cases are required only for repair tasks")
        if is_code != (self.test_code is not None):
            raise ValueError("test_code is required only for repair tasks")
        if is_code and not 4 <= len(self.code_cases) <= 128:
            raise ValueError("repair tasks require 4 to 128 functional cases")
        if len({case.inputs for case in self.code_cases}) != len(self.code_cases):
            raise ValueError("duplicate functional inputs")
        if self.required_modules:
            raise ValueError("module semantic roles are not assigned in Iris-GW-0.1")
        required = {"active_constraint", "operation", "conflicting_conditions", "result_type"}
        if not required <= self.hidden_facts.keys():
            raise ValueError("all four diagnostic fact targets must be present")
        if any(
            type(self.hidden_facts[key]) is not str for key in required - {"conflicting_conditions"}
        ):
            raise ValueError("diagnostic categorical facts must be strings")
        if type(self.hidden_facts["conflicting_conditions"]) is not bool:
            raise ValueError("conflicting_conditions must be boolean")
        for value in (self.prompt, self.expected_response, self.test_code or ""):
            if "\0" in value:
                raise ValueError("row text cannot contain NUL")
            value.encode("utf-8", errors="strict")
        return self


class SyntheticSplitPlan(_StrictModel):
    split: SyntheticSplit
    first_seed: int = Field(ge=0, le=2**63 - 1)
    count: int = Field(ge=15, le=100000)

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.count % 5 or self.first_seed + self.count > 2**63:
            raise ValueError("split requires complete 40/20/20/20 cells and a valid seed interval")
        return self


class SyntheticDatasetPlan(_StrictModel):
    """Explicit dataset seeds; never inferred from training repetition seeds."""

    splits: tuple[SyntheticSplitPlan, SyntheticSplitPlan, SyntheticSplitPlan]

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if tuple(item.split for item in self.splits) != ("train", "validation", "test"):
            raise ValueError("split order must be train, validation, test")
        intervals = sorted((item.first_seed, item.first_seed + item.count) for item in self.splits)
        if any(right[0] < left[1] for left, right in pairwise(intervals)):
            raise ValueError("dataset seed intervals must be disjoint")
        return self
