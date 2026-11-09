"""Strict immutable handoff wire DATA. No command/native/stage ownership token."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    IrisScientificAblationCommandWorkflowDefinitionV1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPhaseDataV1,
    ScientificAblationWorkflowPhaseV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class ScientificAblationWorkflowCommandIntentPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-command-intent-v1"] = (
        "scientific-ablation-workflow-command-intent-v1"
    )
    namespace: IrisScientificAblationCommandWorkflowDefinitionV1
    phase: ScientificAblationWorkflowPhaseV1
    completed_phase_sha256: tuple[Digest, ...] = Field(max_length=134)
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None
    rows_per_call: int | None = Field(ge=1, le=5000)
    authority: Literal["request_data_only"] = "request_data_only"

    @model_validator(mode="after")
    def exact_original_request(self) -> Self:
        if len(self.completed_phase_sha256) != self.phase.index:
            raise ValueError("Exact original phase prefix length required")
        if self.phase.kind in ("capture", "score"):
            if (
                self.expected_chunk is None
                or self.rows_per_call != self.expected_chunk.rows_per_call
            ):
                raise ValueError("Exact original kernel interval and row budget required")
        elif self.expected_chunk is not None or self.rows_per_call is not None:
            raise ValueError("Native/publication command has no kernel interval")
        return self


class ScientificAblationWorkflowReceivedPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-original-result-received-v1"] = (
        "scientific-ablation-workflow-original-result-received-v1"
    )
    request_sha256: Digest
    original_phase_record: IrisScientificAblationWorkflowPhaseDataV1 | None
    original_kernel_progress: IrisScientificAblationWorkflowKernelProgressV1 | None
    original_files_location: str | None = Field(
        pattern=r"^(\.iris-workflow-[0-9a-f]{32}\.tmp|phase-[0-9]{3})$"
    )
    authority: Literal["received_data_not_native_ownership"] = "received_data_not_native_ownership"

    @model_validator(mode="after")
    def detached_files(self) -> Self:
        if (self.original_phase_record is None) != (self.original_files_location is None):
            raise ValueError("Only a complete original phase has a DATA file location")
        if self.original_phase_record is None and self.original_kernel_progress is None:
            raise ValueError("A partial result requires the original kernel snapshot")
        return self


class ScientificAblationWorkflowProducerAckPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-original-producer-ack-v1"] = (
        "scientific-ablation-workflow-original-producer-ack-v1"
    )
    request_sha256: Digest
    received_sha256: Digest
    authority: Literal["original_reference_received_not_native_restoration"] = (
        "original_reference_received_not_native_restoration"
    )


class ScientificAblationWorkflowCommandProgressV1(_Strict):
    plan_sha256: Digest
    completed_phase_sha256: tuple[Digest, ...] = Field(max_length=134)
    next_phase: str | None = Field(min_length=1, max_length=64)
    complete: bool
    recovered_prefix_is_data_only: bool

    @model_validator(mode="after")
    def completion(self) -> Self:
        if self.complete != (self.next_phase is None):
            raise ValueError("Original complete progress has no next phase")
        return self


class ScientificAblationWorkflowReturnedPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-original-returned-result-v1"] = (
        "scientific-ablation-workflow-original-returned-result-v1"
    )
    request_sha256: Digest
    producer_ack_sha256: Digest
    progress: ScientificAblationWorkflowCommandProgressV1
    authority: Literal["original_return_data_only"] = "original_return_data_only"


class ScientificAblationWorkflowOperatorAckPayloadV1(_Strict):
    kind: Literal["scientific-ablation-workflow-original-operator-ack-v1"] = (
        "scientific-ablation-workflow-original-operator-ack-v1"
    )
    request_sha256: Digest
    returned_result_sha256: Digest
    producer_ack_sha256: Digest
    authority: Literal["saved_original_data_not_execution_permission"] = (
        "saved_original_data_not_execution_permission"
    )


class _Envelope[P: _Strict](_Strict):
    sha256: Digest
    payload: P

    @model_validator(mode="after")
    def exact_wire_digest(self) -> Self:
        kind = self.payload.model_dump(mode="json")["kind"]
        if type(kind) is not str or self.sha256 != domain_sha256(
            "iris/" + kind, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Original handoff wire checksum differs")
        return self


class IrisScientificAblationWorkflowCommandIntentV1(
    _Envelope[ScientificAblationWorkflowCommandIntentPayloadV1]
):
    """Constructible request DATA, not native authority or permission to resume."""


class IrisScientificAblationWorkflowReceivedV1(
    _Envelope[ScientificAblationWorkflowReceivedPayloadV1]
):
    """Original receiver DATA; cannot recreate its live result/stage capability."""


class IrisScientificAblationWorkflowProducerAckV1(
    _Envelope[ScientificAblationWorkflowProducerAckPayloadV1]
):
    """Independent original producer ACK DATA, never an original owner."""


class IrisScientificAblationWorkflowReturnedV1(
    _Envelope[ScientificAblationWorkflowReturnedPayloadV1]
):
    """Persisted exact original progress, never a prediction from a newer head."""


class IrisScientificAblationWorkflowOperatorAckV1(
    _Envelope[ScientificAblationWorkflowOperatorAckPayloadV1]
):
    """Saved original DATA acknowledgement; no new native execution is authorized."""
