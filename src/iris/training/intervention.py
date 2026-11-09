"""Distinct evaluation-only NLL output; never an ordinary training/evaluation row."""

from dataclasses import dataclass
from re import fullmatch
from typing import Literal

import torch

from iris.interventions.session import WORKSPACE_INTERVENTION_SEMANTICS_V1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceV1


def teacher_target_sha256_v1(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    """Bind the complete prepared target tensors, including padding/mask placement."""
    return domain_sha256(
        "iris/intervened-teacher-target/v1",
        canonical_json_bytes(
            {
                "input_ids": tensor_sha256(input_ids),
                "attention_mask": tensor_sha256(attention_mask),
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class IrisIntervenedTeacherOutputV1:
    """Actual detached mean target NLL and changed post-write workspace states.

    Same six workspace phases as intervened inference; targets enter only the
    subsequent causal pass. This is not a five-term training objective or a
    differentiable/soft-utilization trace, and has no baseline DTO inheritance.
    Identity hashes check consistency, not external attestation of execution.
    """

    loss: torch.Tensor
    supervised_token_count: int
    workspace_trace: tuple[WorkspaceState, ...]
    prompt_sha256: str
    target_sha256: str
    intervention_plan_sha256: str
    teacher_graph_sha256: str
    sidecar_state_sha256: str
    semantics: str = WORKSPACE_INTERVENTION_SEMANTICS_V1
    kind: Literal["iris-intervened-teacher-output-v1"] = "iris-intervened-teacher-output-v1"
    diagnostic_trace: WorkspaceTraceV1 | None = None

    def __post_init__(self) -> None:
        if (
            self.kind != "iris-intervened-teacher-output-v1"
            or self.semantics != WORKSPACE_INTERVENTION_SEMANTICS_V1
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count < 1
            or type(self.loss) is not torch.Tensor
            or self.loss.ndim != 0
            or self.loss.dtype not in (torch.float32, torch.float64)
            or self.loss.requires_grad
            or self.loss.grad_fn is not None
            or self.loss.is_inference()
            or not bool(torch.isfinite(self.loss))
            or float(self.loss) < 0
            or type(self.workspace_trace) is not tuple
            or not self.workspace_trace
            or any(type(state) is not WorkspaceState for state in self.workspace_trace)
        ):
            raise ValueError("An exact detached intervened teacher measurement is required")
        for value in (
            self.prompt_sha256,
            self.target_sha256,
            self.intervention_plan_sha256,
            self.teacher_graph_sha256,
            self.sidecar_state_sha256,
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Intervened teacher identities must be lowercase SHA256")


__all__ = ["IrisIntervenedTeacherOutputV1", "teacher_target_sha256_v1"]
