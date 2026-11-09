"""Distinct durable held-out plan, row acknowledgement and complete coverage."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_controlled_generation_reconstruction_v1 import (
    IrisScientificAblationControlledGenerationReconstructionV1,
)
from iris.artifacts.scientific_ablation_controlled_heldout_v1 import (
    IrisScientificAblationControlledHeldoutInputV1,
)
from iris.artifacts.scientific_ablation_generation_source_v1 import (
    IrisScientificAblationGenerationSourceManifestV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class ScientificAblationControlledCaptureExpectationV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    request_prompt_sha256: Digest
    cognitive_prompt_sha256: Digest
    prompt_input_ids_sha256: Digest
    prompt_attention_mask_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    teacher_target_sha256: Digest
    teacher_invocation_sha256: Digest
    sampling_initial_state_sha256: Digest | None
    task_type: SyntheticTask


class ScientificAblationControlledCapturePlanPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-capture-plan-v1"] = (
        "iris-scientific-ablation-controlled-capture-plan-v1"
    )
    scope: Literal["owned_test_plan_not_scoring_or_launch_approval"] = (
        "owned_test_plan_not_scoring_or_launch_approval"
    )
    input: IrisScientificAblationControlledHeldoutInputV1
    source: IrisScientificAblationGenerationSourceManifestV1
    reconstruction: IrisScientificAblationControlledGenerationReconstructionV1
    scientific_state_sha256: Digest
    teacher_graph_sha256: Digest
    cycles: int = Field(ge=1, le=1024)
    maximum_row_file_bytes: int = Field(ge=1, le=8 * 1024**2)
    rows: tuple[ScientificAblationControlledCaptureExpectationV1, ...] = Field(
        min_length=1, max_length=5000, repr=False
    )

    @model_validator(mode="after")
    def complete_plan(self) -> Self:
        for manifest, domain in (
            (self.input, "iris/scientific-ablation-controlled-heldout-input/v1"),
            (self.source, "iris/scientific-ablation-generation-source/v1/payload"),
            (
                self.reconstruction,
                "iris/scientific-ablation-controlled-generation-reconstruction/v1/payload",
            ),
        ):
            if manifest.sha256 != domain_sha256(
                domain, canonical_json_bytes(manifest.payload.model_dump(mode="json"))
            ):
                raise ValueError("Capture input/source/reconstruction checksum differs")
        p, s, r = self.input.payload, self.source.payload, self.reconstruction.payload
        if (
            len(self.rows) != p.example_count
            or tuple(row.row_index for row in self.rows) != tuple(range(len(self.rows)))
            or len({row.example_id for row in self.rows}) != len(self.rows)
            or sum(row.target_token_count for row in self.rows) != p.target_token_count
            or p.source_sha256 != self.source.sha256
            or r.source_sha256 != self.source.sha256
            or p.reconstruction_sha256 != self.reconstruction.sha256
            or p.selected_checkpoint_sha256 != s.best_checkpoint_sha256
            or r.selected_checkpoint_sha256 != s.best_checkpoint_sha256
            or p.ablation_policy_sha256 != s.ablation_policy_sha256
            or r.ablation_policy_sha256 != s.ablation_policy_sha256
            or p.purpose != s.purpose
            or r.lifecycle_sha256 != s.lifecycle_sha256
            or r.run_configuration_sha256 != s.run_configuration_sha256
            or r.model_configuration_sha256 != s.model_configuration_sha256
            or r.inference_sha256 != s.inference_sha256
            or r.selected_step != s.selected_step
            or r.backbone_runtime_sha256 != s.backbone_runtime_sha256
            or r.request_sidecar_sha256 != s.request_sidecar_sha256
        ):
            raise ValueError("Full owned capture plan/source/best inventory differs")
        if any(
            (row.sampling_initial_state_sha256 is None) != (p.decoding.strategy == "greedy")
            for row in self.rows
        ):
            raise ValueError("Capture sampling-state inventory differs")
        return self


class IrisScientificAblationControlledCapturePlanV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledCapturePlanPayloadV1


class ScientificAblationControlledCaptureAckPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-capture-row-ack-v1"] = (
        "iris-scientific-ablation-controlled-capture-row-ack-v1"
    )
    status: Literal["acknowledged_success"] = "acknowledged_success"
    plan_sha256: Digest
    row_index: int = Field(ge=0, le=4999)
    previous_ack_sha256: Digest
    row_sha256: Digest
    row_file_sha256: Digest
    row_file_bytes: int = Field(ge=1, le=8 * 1024**2)
    answer_sha256: Digest


class IrisScientificAblationControlledCaptureAckV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledCaptureAckPayloadV1


class ScientificAblationControlledCaptureCompletePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-controlled-capture-complete-v1"] = (
        "iris-scientific-ablation-controlled-capture-complete-v1"
    )
    status: Literal["complete_not_scored_or_launch_approved"] = (
        "complete_not_scored_or_launch_approved"
    )
    plan_sha256: Digest
    input_sha256: Digest
    generation_control_plan_sha256: Digest
    target_nll_semantics: Literal["selected_best_cognitive_mode_without_workspace_control"] = (
        "selected_best_cognitive_mode_without_workspace_control"
    )
    acknowledgement_head_sha256: Digest
    example_count: int = Field(ge=1, le=5000)
    target_token_count: int = Field(ge=2, le=2**48)
    target_nll_sum: float = Field(ge=0.0)


class IrisScientificAblationControlledCaptureCompleteV1(_Strict):
    sha256: Digest
    payload: ScientificAblationControlledCaptureCompletePayloadV1


__all__ = [
    "IrisScientificAblationControlledCaptureAckV1",
    "IrisScientificAblationControlledCaptureCompleteV1",
    "IrisScientificAblationControlledCapturePlanV1",
    "ScientificAblationControlledCaptureAckPayloadV1",
    "ScientificAblationControlledCaptureCompletePayloadV1",
    "ScientificAblationControlledCaptureExpectationV1",
    "ScientificAblationControlledCapturePlanPayloadV1",
]
