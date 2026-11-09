"""Retainable kernel-prefix DATA view, never permission to execute or adopt DATA."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_workflow_v1 import ScientificAblationWorkflowPhaseV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class IrisScientificAblationWorkflowKernelProgressV1(_Strict):
    kind: Literal["iris-scientific-ablation-workflow-kernel-progress-v1"] = (
        "iris-scientific-ablation-workflow-kernel-progress-v1"
    )
    workflow_plan_sha256: Digest
    phase: ScientificAblationWorkflowPhaseV1
    completed_phase_sha256: tuple[Digest, ...] = Field(max_length=134)
    phase_is_acknowledged: bool
    kernel_plan_sha256: Digest
    acknowledgement_head_sha256: Digest
    acknowledged_rows: int = Field(ge=0, le=5000)
    total_rows: int = Field(ge=1, le=5000)
    kernel_complete_sha256: Digest | None
    uncommitted_stage_count: int = Field(ge=0, le=700000)
    native_execution_authority: Literal["none_data_only"] = "none_data_only"
    new_native_execution_authorized: Literal[False] = False
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def coherent_prefix(self) -> Self:
        if (
            self.phase.kind not in ("capture", "score")
            or self.phase.index > len(self.completed_phase_sha256)
            or self.phase_is_acknowledged != (self.phase.index < len(self.completed_phase_sha256))
            or self.acknowledged_rows > self.total_rows
            or (
                self.kernel_complete_sha256 is not None
                and self.acknowledged_rows != self.total_rows
            )
            or (self.phase_is_acknowledged and self.kernel_complete_sha256 is None)
        ):
            raise ValueError("Exact current/past kernel prefix DATA required")
        return self


__all__ = ["IrisScientificAblationWorkflowKernelProgressV1"]
