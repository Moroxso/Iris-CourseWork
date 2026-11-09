"""Distinct strict persisted ablation lifecycle, containing current and selected best."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_checkpoint_v1 import (
    IrisScientificAblationCheckpointManifestV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import ScientificAblationLifecycleStateV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationLifecycleCheckpointPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-lifecycle-checkpoint-v1"] = (
        "iris-scientific-ablation-lifecycle-checkpoint-v1"
    )
    configuration_json: str = Field(min_length=1, max_length=4 * 1024**2)
    configuration_sha256: Digest
    state: ScientificAblationLifecycleStateV1
    current: IrisScientificAblationCheckpointManifestV1
    best: IrisScientificAblationCheckpointManifestV1 | None

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if (
            self.state.phase not in ("train", "stopped")
            or self.state.checkpoint_count < 1
            or self.state.last_checkpoint_step != self.state.step
            or self.state.step != self.current.payload.completed_steps
            or ((self.state.best_step is None) != (self.best is None))
            or (
                self.best is not None
                and (
                    self.best.payload.completed_steps != self.state.best_step
                    or self.best.payload.configuration_sha256
                    != self.current.payload.configuration_sha256
                )
            )
        ):
            raise ValueError("Ablation lifecycle requires a complete checkpoint boundary")
        return self


class IrisScientificAblationLifecycleCheckpointManifestV1(_Strict):
    sha256: Digest
    payload: ScientificAblationLifecycleCheckpointPayloadV1


__all__ = [
    "IrisScientificAblationLifecycleCheckpointManifestV1",
    "ScientificAblationLifecycleCheckpointPayloadV1",
]
