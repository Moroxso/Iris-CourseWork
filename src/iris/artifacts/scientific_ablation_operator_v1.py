"""Strict operator intent/results; no native capabilities or launch approval."""

from typing import Literal

from pydantic import Field

from iris.artifacts.scientific_ablation_execution_v1 import IrisScientificAblationExecutionEventV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict

Phase = Literal["train", "validation", "checkpoint"]


class ScientificAblationOperatorPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-operator-v1"] = "iris-scientific-ablation-operator-v1"
    recipe_json: str = Field(min_length=1, max_length=128 * 1024)
    recipe_sha256: Digest
    maximum_trainable_parameter_bytes: int = Field(ge=1, le=512 * 1024**2)
    execution_sha256: Digest
    initial_checkpoint_sha256: Digest


class IrisScientificAblationOperatorV1(_Strict):
    sha256: Digest
    payload: ScientificAblationOperatorPayloadV1


class ScientificAblationPhaseRequestPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-phase-request-v1"] = (
        "iris-scientific-ablation-phase-request-v1"
    )
    operator_sha256: Digest
    execution_sha256: Digest
    expected_previous_sha256: Digest
    phase: Phase
    purpose: Literal["scientific", "smoke"]


class IrisScientificAblationPhaseRequestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationPhaseRequestPayloadV1


class ScientificAblationPhaseResultPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-phase-result-v1"] = (
        "iris-scientific-ablation-phase-result-v1"
    )
    request: IrisScientificAblationPhaseRequestV1
    event: IrisScientificAblationExecutionEventV1


class IrisScientificAblationPhaseResultV1(_Strict):
    sha256: Digest
    payload: ScientificAblationPhaseResultPayloadV1


__all__ = [
    "IrisScientificAblationOperatorV1",
    "IrisScientificAblationPhaseRequestV1",
    "IrisScientificAblationPhaseResultV1",
    "Phase",
    "ScientificAblationOperatorPayloadV1",
    "ScientificAblationPhaseRequestPayloadV1",
    "ScientificAblationPhaseResultPayloadV1",
]
