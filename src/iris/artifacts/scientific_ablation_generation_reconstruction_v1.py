"""Redacted observations of actual fresh original-context reconstruction."""

from typing import Literal

from pydantic import Field

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationGenerationReconstructionPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-generation-reconstruction-v1"] = (
        "iris-scientific-ablation-generation-reconstruction-v1"
    )
    scope: Literal["reconstructed_source_not_heldout_capture_or_launch_approval"] = (
        "reconstructed_source_not_heldout_capture_or_launch_approval"
    )
    source_sha256: Digest
    lifecycle_sha256: Digest
    lifecycle_configuration_sha256: Digest
    run_configuration_sha256: Digest
    model_configuration_sha256: Digest
    ablation_policy_sha256: Digest
    fresh_initial_checkpoint_sha256: Digest
    initial_scientific_state_sha256: Digest
    training_corpus_sha256: Digest
    training_examples: int = Field(ge=1, le=200000)
    validation_corpus_sha256: Digest
    validation_examples: int = Field(ge=1, le=5000)
    validation_target_tokens: int = Field(ge=1, le=2**48)
    selected_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1, le=2**24)
    inference_sha256: Digest
    backbone_runtime_sha256: Digest
    request_sidecar_sha256: Digest


class IrisScientificAblationGenerationReconstructionV1(_Strict):
    sha256: Digest
    payload: ScientificAblationGenerationReconstructionPayloadV1


__all__ = [
    "IrisScientificAblationGenerationReconstructionV1",
    "ScientificAblationGenerationReconstructionPayloadV1",
]
