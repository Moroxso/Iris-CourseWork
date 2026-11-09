"""Redacted terminal native-best binding, not test/capture/launch approval."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationGenerationSourcePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-generation-source-v1"] = (
        "iris-scientific-ablation-generation-source-v1"
    )
    arm_id: Literal["D"] = "D"
    protocol_role: Literal["post_main_ablation"] = "post_main_ablation"
    scope: Literal["terminal_source_binding_not_test_or_launch_approval"] = (
        "terminal_source_binding_not_test_or_launch_approval"
    )
    purpose: Literal["scientific", "smoke"]
    ablation: CognitiveAblationPolicyV1
    ablation_policy_sha256: Digest
    lifecycle_sha256: Digest
    lifecycle_configuration_sha256: Digest
    current_checkpoint_sha256: Digest
    best_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1, le=2**24)
    terminal_step: int = Field(ge=1, le=2**24)
    stop_reason: Literal["step_budget", "early_stopping"]
    best_validation_nll_per_token: float = Field(ge=0.0)
    run_configuration_sha256: Digest
    model_configuration_sha256: Digest
    initial_scientific_state_sha256: Digest
    training_corpus_sha256: Digest
    training_examples: int = Field(ge=1, le=200000)
    validation_corpus_sha256: Digest
    validation_examples: int = Field(ge=1, le=5000)
    validation_target_tokens: int = Field(ge=1, le=2**48)
    backbone_runtime_sha256: Digest
    inference_sha256: Digest
    teacher_sidecar_sha256: Digest
    request_sidecar_sha256: Digest

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if (
            self.selected_step > self.terminal_step
            or self.ablation_policy_sha256 != cognitive_ablation_policy_sha256_v1(self.ablation)
            or (
                (self.best_checkpoint_sha256 == self.current_checkpoint_sha256)
                != (self.selected_step == self.terminal_step)
            )
            or (
                self.purpose == "scientific"
                and (self.training_examples != 50000 or self.validation_examples != 5000)
            )
        ):
            raise ValueError("Terminal mode-bound selected-best source differs")
        return self


class IrisScientificAblationGenerationSourceManifestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationGenerationSourcePayloadV1


__all__ = [
    "IrisScientificAblationGenerationSourceManifestV1",
    "ScientificAblationGenerationSourcePayloadV1",
]
