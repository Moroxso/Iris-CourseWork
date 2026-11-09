"""Distinct controlled held-out evidence; never an ordinary capture or score."""

import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    validate_ablation_control_plan_v1,
)
from iris.ablations.policy import cognitive_ablation_policy_sha256_v1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_generation_v1 import (
    ScientificDecodingPolicyV1,
    scientific_row_sampling_seed_v1,
)
from iris.data.schemas import SyntheticTask
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

_MAX_CONTROL = 64 * 1024**2
_CONTROL_DOMAIN = "iris/scientific-ablation-controlled-generation/v1/plan"
_REQUEST_DOMAIN = "iris/scientific-ablation-controlled-heldout-row-request/v1"


class ScientificAblationControlledHeldoutInputPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-heldout-input-v1"] = (
        "iris-scientific-ablation-controlled-heldout-input-v1"
    )
    scope: Literal["preflight_not_capture_scoring_or_launch_approval"] = (
        "preflight_not_capture_scoring_or_launch_approval"
    )
    reconstruction_sha256: Digest
    source_sha256: Digest
    ablation_policy_sha256: Digest
    generation_control_plan_sha256: Digest
    generation_control_plan_json: str = Field(min_length=1, max_length=_MAX_CONTROL, repr=False)
    target_nll_semantics: Literal["selected_best_cognitive_mode_without_workspace_control"] = (
        "selected_best_cognitive_mode_without_workspace_control"
    )
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
        raw = self.generation_control_plan_json.encode("utf-8", errors="strict")
        if len(raw) > _MAX_CONTROL:
            raise ValueError("Complete controlled intent exceeds byte quota")
        plan = validate_ablation_control_plan_v1(
            CognitiveAblationControlPlanV1.model_validate_json(raw)
        )
        if (
            type(plan.control) is not WorkspaceInterventionPlanV1
            or canonical_json_bytes(plan.model_dump(mode="json")) != raw
            or domain_sha256(_CONTROL_DOMAIN, raw) != self.generation_control_plan_sha256
            or cognitive_ablation_policy_sha256_v1(plan.ablation) != self.ablation_policy_sha256
        ):
            raise ValueError("Exact complete generation control/policy/domain binding differs")
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids) or any(
            token >= self.vocabulary_size for token in self.eos_token_ids
        ):
            raise ValueError("Invalid actual EOS inventory")
        if self.purpose == "scientific" and self.example_count != 5000:
            raise ValueError("Scientific held-out input requires the complete split")
        return self


class IrisScientificAblationControlledHeldoutInputV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledHeldoutInputPayloadV1


class ScientificAblationControlledHeldoutRowPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-heldout-row-v1"] = (
        "iris-scientific-ablation-controlled-heldout-row-v1"
    )
    scope: Literal["actual_row_not_durable_ack_or_complete_evaluation"] = (
        "actual_row_not_durable_ack_or_complete_evaluation"
    )
    input: IrisScientificAblationControlledHeldoutInputV1
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
    request_receipt_json: str = Field(min_length=1, max_length=_MAX_CONTROL, repr=False)

    @model_validator(mode="after")
    def complete_row(self) -> Self:
        p = self.input.payload
        raw_receipt = self.request_receipt_json.encode("utf-8", errors="strict")
        receipt = json.loads(raw_receipt)
        if (
            len(raw_receipt) > _MAX_CONTROL
            or type(receipt) is not dict
            or canonical_json_bytes(receipt) != raw_receipt
            or domain_sha256(_REQUEST_DOMAIN, raw_receipt) != self.request_receipt_sha256
            or receipt.get("kind") != "iris-scientific-ablation-controlled-request-execution-v1"
            or receipt.get("semantics")
            != "prompt-only-workspace-control-full-caller-state-transaction-v1"
            or receipt.get("control_plan_sha256") != p.generation_control_plan_sha256
            or receipt.get("control_plan_json") != p.generation_control_plan_json
        ):
            raise ValueError("Actual controlled receipt must match this whole held-out intent")
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


class IrisScientificAblationControlledHeldoutRowV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledHeldoutRowPayloadV1 = Field(repr=False)


__all__ = [
    "IrisScientificAblationControlledHeldoutInputV1",
    "IrisScientificAblationControlledHeldoutRowV1",
    "ScientificAblationControlledHeldoutInputPayloadV1",
    "ScientificAblationControlledHeldoutRowPayloadV1",
]
