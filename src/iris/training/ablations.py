"""Distinct graph-attached ablation teacher output; never an ordinary baseline row."""

from dataclasses import dataclass
from typing import Literal

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.training.outputs import IrisTeacherTracedOutputV1


@dataclass(frozen=True, slots=True)
class IrisAblatedTeacherOutputV1:
    mode: Literal["training", "evaluation"]
    traced: IrisTeacherTracedOutputV1
    policy: CognitiveAblationPolicyV1
    policy_sha256: str
    invocation_sha256: str
    prompt_sha256: str
    target_sha256: str
    teacher_graph_sha256: str
    sidecar_state_sha256: str
