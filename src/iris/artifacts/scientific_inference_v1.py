"""Inference-only D weights and provenance, distinct from resumable run checkpoints."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.checkpoint_v1 import TeacherCheckpointModelIdentityV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, ScientificTensorV1, _Strict

REQUEST_COMPONENTS_V1 = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "workspace_reader",
    "cognitive_modules",
    "proposal_head",
    "workspace_arbiter",
    "slot_allocator",
    "workspace_writer",
    "workspace_decay",
    "latent_bridge",
    "prefix_assembler",
)


class ScientificInferencePayloadV1(_Strict):
    kind: Literal["iris-scientific-inference-v1"] = "iris-scientific-inference-v1"
    arm_id: Literal["D"] = "D"
    purpose: Literal["scientific", "smoke"]
    source_checkpoint_sha256: Digest
    source_run_configuration_sha256: Digest
    source_teacher_configuration_sha256: Digest
    source_teacher_sidecar_sha256: Digest
    backbone_runtime_sha256: Digest
    completed_steps: int = Field(ge=0, le=2**24)
    model: TeacherCheckpointModelIdentityV1
    tensors: tuple[ScientificTensorV1, ...] = Field(min_length=1, max_length=4096)
    state_sha256: Digest
    state_size_bytes: int = Field(gt=0, lt=200_000_000)

    @model_validator(mode="after")
    def validate_inference_state(self) -> Self:
        names = self.model.state_names
        if (
            not 1 <= len(names) <= 4096
            or any(
                name.partition(".")[0] not in REQUEST_COMPONENTS_V1[:-1]
                or not name.partition(".")[2]
                for name in names
            )
            or tuple(t.key for t in self.tensors) != tuple(sorted(names))
            or any(t.dtype != "float32" for t in self.tensors)
            or sum(t.byte_length for t in self.tensors) >= self.state_size_bytes
            or self.model.runtime.dtype != "float32"
        ):
            raise ValueError("Inference payload is not the FP32 D parameter projection")
        return self


class IrisScientificInferenceManifestV1(_Strict):
    sha256: Digest
    payload: ScientificInferencePayloadV1


__all__ = ["IrisScientificInferenceManifestV1", "ScientificInferencePayloadV1"]
