"""Explicit initial/retained kernel planning request DATA, never ownership."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    IrisScientificAblationCommandWorkflowDefinitionV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import _Envelope
from iris.artifacts.scientific_ablation_workflow_v1 import ScientificAblationWorkflowPhaseV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationWorkflowKernelRequestPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-kernel-request-v1"] = (
        "scientific-ablation-workflow-kernel-request-v1"
    )
    namespace: IrisScientificAblationCommandWorkflowDefinitionV1
    phase: ScientificAblationWorkflowPhaseV1
    completed_phase_sha256: tuple[Digest, ...] = Field(max_length=134)
    route: Literal["initial", "retained"]
    rows_per_call: int = Field(ge=1, le=5000)
    expected_kernel_plan_sha256: Digest | None
    expected_acknowledgement_head_sha256: Digest | None
    expected_acknowledged_rows: int = Field(ge=0, le=5000)
    authority: Literal["planning_request_data_only"] = "planning_request_data_only"

    @model_validator(mode="after")
    def explicit_prefix(self) -> Self:
        if (
            self.phase.kind not in {"capture", "score"}
            or len(self.completed_phase_sha256) != self.phase.index
        ):
            raise ValueError("Exact due capture/score phase prefix required for kernel planning")
        if self.route == "initial":
            if (
                self.expected_kernel_plan_sha256 is not None
                or self.expected_acknowledgement_head_sha256 is not None
                or self.expected_acknowledged_rows != 0
            ):
                raise ValueError(
                    "Initial planning requires explicit zero rows and no guessed anchors"
                )
        elif (
            self.expected_kernel_plan_sha256 is None
            or self.expected_acknowledgement_head_sha256 is None
        ):
            raise ValueError("Retained planning requires both independently saved kernel anchors")
        return self


class IrisScientificAblationWorkflowKernelRequestV1(
    _Envelope[ScientificAblationWorkflowKernelRequestPayloadV1]
):
    """Constructible saved request; cannot recreate a factory, claim or native owner."""


__all__ = [
    "IrisScientificAblationWorkflowKernelRequestV1",
    "ScientificAblationWorkflowKernelRequestPayloadV1",
]
