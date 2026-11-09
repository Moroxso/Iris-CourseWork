"""Distinct evaluation receipt for actual collected/transplanted proposal content."""

from dataclasses import dataclass
from re import fullmatch
from typing import Literal

import torch

from iris.interventions.messages import ModuleMessageSnapshotV1
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceV1


@dataclass(frozen=True, slots=True)
class IrisMessageTeacherOutputV1:
    loss: torch.Tensor
    supervised_token_count: int
    workspace_trace: tuple[WorkspaceState, ...]
    diagnostic_trace: WorkspaceTraceV1
    messages: tuple[ModuleMessageSnapshotV1, ...]
    prompt_sha256: str
    target_sha256: str
    message_plan_sha256: str
    teacher_graph_sha256: str
    sidecar_state_sha256: str
    kind: Literal["iris-message-teacher-output-v1"] = "iris-message-teacher-output-v1"

    def __post_init__(self) -> None:
        if (
            self.kind != "iris-message-teacher-output-v1"
            or type(self.loss) is not torch.Tensor
            or self.loss.ndim != 0
            or self.loss.dtype not in (torch.float32, torch.float64)
            or self.loss.requires_grad
            or self.loss.grad_fn is not None
            or self.loss.is_inference()
            or not bool(torch.isfinite(self.loss))
            or float(self.loss) < 0
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count < 1
            or type(self.diagnostic_trace) is not WorkspaceTraceV1
            or self.diagnostic_trace.options.mode != "full"
            or type(self.messages) is not tuple
            or type(self.workspace_trace) is not tuple
        ):
            raise ValueError("Exact detached message evaluation receipt required")
        trace = WorkspaceTraceV1.model_validate(self.diagnostic_trace)
        if (
            trace.intervention_plan_sha256 != self.message_plan_sha256
            or len(self.workspace_trace) != trace.cycle_count
            or any(type(s) is not WorkspaceState for s in self.workspace_trace)
        ):
            raise ValueError("Message execution trace/plan differs")
        expected = tuple(
            (cycle, module)
            for cycle in range(1, trace.cycle_count + 1)
            for module in range(trace.num_modules)
        )
        if tuple((m.cycle, m.module) for m in self.messages) != expected:
            raise ValueError("Incomplete or unordered proposal message capture")
        for value in (
            self.prompt_sha256,
            self.target_sha256,
            self.message_plan_sha256,
            self.teacher_graph_sha256,
            self.sidecar_state_sha256,
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Message receipt requires lowercase SHA256 identities")
        for message in self.messages:
            message = ModuleMessageSnapshotV1.model_validate(message)
            if (
                message.prompt_sha256 != self.prompt_sha256
                or message.teacher_graph_sha256 != self.teacher_graph_sha256
                or message.sidecar_state_sha256 != self.sidecar_state_sha256
                or message.example_id != trace.options.example_id
                or len(message.values) != trace.state_dim
            ):
                raise ValueError("Message snapshot does not match the owned evaluation")
