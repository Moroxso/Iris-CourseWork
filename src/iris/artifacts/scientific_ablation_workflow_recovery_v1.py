"""Distinct recovered kernel DATA, NEVER a fabricated original returned result."""

from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    _Envelope,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPhaseDataV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationWorkflowRecoveryIntentPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-recovery-intent-v1"] = (
        "scientific-ablation-workflow-recovery-intent-v1"
    )
    original_command: IrisScientificAblationWorkflowCommandIntentV1
    authority: Literal["recovery_request_data_only"] = "recovery_request_data_only"

    @model_validator(mode="after")
    def resumable_only(self) -> Self:
        if self.original_command.payload.expected_chunk is None:
            raise ValueError("Recovery requires the unchanged original capture/score interval")
        return self


class ScientificAblationWorkflowRecoveredFileV1(_Strict):
    name: Literal["capture.json", "evaluation.json"]
    content: str = Field(min_length=1, max_length=128 * 1024)


class ScientificAblationWorkflowRecoveredResultPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-recovered-kernel-data-v1"] = (
        "scientific-ablation-workflow-recovered-kernel-data-v1"
    )
    recovery_request_sha256: Digest
    original_request_sha256: Digest
    kernel_progress: IrisScientificAblationWorkflowKernelProgressV1
    observed_acknowledged_rows: int = Field(ge=0, le=5000)
    resumed_from_row: int | None = Field(ge=0, le=5000)
    route: Literal["continued_original_interval", "verified_kernel_data"]
    phase_record: IrisScientificAblationWorkflowPhaseDataV1 | None
    phase_files: tuple[ScientificAblationWorkflowRecoveredFileV1, ...] = Field(max_length=1)
    original_return: Literal["unavailable_not_reconstructed"] = "unavailable_not_reconstructed"
    authority: Literal["recovered_data_not_native_ownership"] = (
        "recovered_data_not_native_ownership"
    )

    @model_validator(mode="after")
    def exact_data(self) -> Self:
        kernel = self.kernel_progress
        if (
            kernel.phase_is_acknowledged
            or self.observed_acknowledged_rows < kernel.acknowledged_rows
            or self.observed_acknowledged_rows > kernel.total_rows
            or (self.route == "verified_kernel_data") != (self.resumed_from_row is None)
            or (
                self.resumed_from_row is not None
                and not 0
                <= self.resumed_from_row
                <= kernel.acknowledged_rows
                == self.observed_acknowledged_rows
            )
        ):
            raise ValueError("Explicit bounded recovered DATA and route required")
        record = self.phase_record
        if record is None:
            if self.phase_files:
                raise ValueError("Partial recovered DATA cannot invent phase files")
        else:
            files: dict[str, bytes] = {f.name: f.content.encode("utf-8") for f in self.phase_files}
            if (
                record.payload.origin != "kernel_data"
                or record.payload.phase != kernel.phase
                or record.payload.plan_sha256 != kernel.workflow_plan_sha256
                or kernel.acknowledged_rows != kernel.total_rows
                or kernel.kernel_complete_sha256 is None
                or len(files) != 1
                or set(files) != {f.name for f in record.payload.files}
                or any(
                    len(files[p.name]) != p.bytes or sha256(files[p.name]).hexdigest() != p.sha256
                    for p in record.payload.files
                )
            ):
                raise ValueError("Exact complete kernel DATA file pins required")
        return self


class ScientificAblationWorkflowRecoveryAckPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-recovery-operator-ack-v1"] = (
        "scientific-ablation-workflow-recovery-operator-ack-v1"
    )
    recovery_request_sha256: Digest
    original_request_sha256: Digest
    recovered_result_sha256: Digest
    completed_phase_sha256: tuple[Digest, ...] = Field(max_length=134)
    authority: Literal["saved_recovered_data_not_execution_permission"] = (
        "saved_recovered_data_not_execution_permission"
    )


class IrisScientificAblationWorkflowRecoveryIntentV1(
    _Envelope[ScientificAblationWorkflowRecoveryIntentPayloadV1]
):
    """Constructible request DATA, not kernel/native permission."""


class IrisScientificAblationWorkflowRecoveredResultV1(
    _Envelope[ScientificAblationWorkflowRecoveredResultPayloadV1]
):
    """Recovered bounded kernel DATA, not the original lost return."""


class IrisScientificAblationWorkflowRecoveryAckV1(
    _Envelope[ScientificAblationWorkflowRecoveryAckPayloadV1]
):
    """Retain independently; DATA ACK does not restore a live producer."""
