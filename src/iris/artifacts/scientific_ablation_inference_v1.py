"""Distinct inference-only cognitive-ablation state; never a resumable checkpoint."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.artifacts.checkpoint_v1 import TeacherCheckpointModelIdentityV1
from iris.artifacts.scientific_ablation_lifecycle_v1 import ScientificAblationLifecycleStateV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, ScientificTensorV1, _Strict
from iris.artifacts.scientific_inference_v1 import REQUEST_COMPONENTS_V1


class ScientificAblationInferenceGenerationPolicyV1(_Strict):
    """Execution intent, not a sampler seed/state or held-out eligibility claim."""

    kind: Literal["iris-scientific-ablation-inference-generation-v1"] = (
        "iris-scientific-ablation-inference-generation-v1"
    )
    session: Literal["fresh_per_request"] = "fresh_per_request"
    prompt_domain: Literal["iris/cognitive-ablation-prompt/v1"] = (
        "iris/cognitive-ablation-prompt/v1"
    )
    sampling_rng: Literal["explicit_caller_full_state_transaction_v1"] = (
        "explicit_caller_full_state_transaction_v1"
    )
    terminal_token_id: int = Field(ge=0, le=2**31 - 1)


class ScientificAblationInferenceSelectionV1(_Strict):
    """Redacted coherent-boundary provenance, authenticated by external bindings.

    Controller counters are selection metadata, not a runnable cursor. Neither
    native manifests nor private training RNG/optimizer state are embedded here.
    Self-consistent rehashing does not authenticate this record's claimed source.
    """

    kind: Literal["current", "selected_best"]
    lifecycle_sha256: Digest
    lifecycle_configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    lifecycle_configuration_sha256: Digest
    current_checkpoint_sha256: Digest
    current_completed_steps: int = Field(ge=1, le=2**24)
    best_checkpoint_sha256: Digest | None
    state: ScientificAblationLifecycleStateV1

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if (
            self.state.phase not in ("train", "stopped")
            or self.state.step != self.current_completed_steps
            or self.state.last_checkpoint_step != self.state.step
            or self.state.checkpoint_count < 1
            or ((self.best_checkpoint_sha256 is None) != (self.state.best_step is None))
            or (self.kind == "selected_best" and self.best_checkpoint_sha256 is None)
        ):
            raise ValueError("Inference selection requires a complete coherent boundary")
        return self


class ScientificAblationInferencePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-inference-v1"] = "iris-scientific-ablation-inference-v1"
    arm_id: Literal["D"] = "D"
    protocol_role: Literal["post_main_ablation"] = "post_main_ablation"
    purpose: Literal["scientific", "smoke"]
    ablation: CognitiveAblationPolicyV1
    ablation_policy_sha256: Digest
    selection: ScientificAblationInferenceSelectionV1
    source_checkpoint_sha256: Digest
    source_run_configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    source_run_configuration_sha256: Digest
    source_model_configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    source_model_configuration_sha256: Digest
    source_teacher_sidecar_sha256: Digest
    backbone_runtime_sha256: Digest
    completed_steps: int = Field(ge=1, le=2**24)
    generation: ScientificAblationInferenceGenerationPolicyV1
    model: TeacherCheckpointModelIdentityV1
    tensors: tuple[ScientificTensorV1, ...] = Field(min_length=1, max_length=4096)
    state_sha256: Digest
    state_size_bytes: int = Field(gt=0, lt=200_000_000)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        names = self.model.state_names
        selected = self.selection
        source_digest = (
            selected.current_checkpoint_sha256
            if selected.kind == "current"
            else selected.best_checkpoint_sha256
        )
        source_step = (
            selected.current_completed_steps
            if selected.kind == "current"
            else selected.state.best_step
        )
        if (
            self.source_checkpoint_sha256 != source_digest
            or self.completed_steps != source_step
            or self.ablation_policy_sha256 != cognitive_ablation_policy_sha256_v1(self.ablation)
            or not 1 <= len(names) <= 4096
            or any(
                name.partition(".")[0] not in REQUEST_COMPONENTS_V1[:-1]
                or not name.partition(".")[2]
                for name in names
            )
            or tuple(t.key for t in self.tensors) != tuple(sorted(names))
            or any(t.dtype != "float32" for t in self.tensors)
            or sum(t.byte_length for t in self.tensors) >= self.state_size_bytes
            or self.model.runtime.dtype != "float32"
            or self.generation.terminal_token_id >= self.model.runtime.vocabulary_size
        ):
            raise ValueError("Mode-bound inference selection/policy/FP32 projection differs")
        return self


class IrisScientificAblationInferenceManifestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationInferencePayloadV1


__all__ = [
    "IrisScientificAblationInferenceManifestV1",
    "ScientificAblationInferenceGenerationPolicyV1",
    "ScientificAblationInferencePayloadV1",
    "ScientificAblationInferenceSelectionV1",
]
