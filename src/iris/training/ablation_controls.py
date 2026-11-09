"""Distinct detached mode+control teacher receipt; no five-term objective claim."""

from dataclasses import dataclass, fields
from re import fullmatch
from typing import Literal

import torch

from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    ablation_control_plan_sha256_v1,
    validate_ablation_control_plan_v1,
)
from iris.ablations.policy import cognitive_ablation_policy_sha256_v1
from iris.interventions.messages import (
    ModuleMessageSnapshotV1,
    WorkspaceMessagePlanV1,
    message_plan_sha256_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, intervention_plan_sha256_v1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceV1


def controlled_teacher_invocation_sha256_v1(
    *,
    plan_sha256: str,
    prompt_sha256: str,
    target_sha256: str,
    teacher_graph_sha256: str,
    sidecar_state_sha256: str,
    supervised_token_count: int,
) -> str:
    return domain_sha256(
        "iris/cognitive-ablation-controlled-teacher-invocation/v1",
        canonical_json_bytes(
            dict(
                plan_sha256=plan_sha256,
                prompt_sha256=prompt_sha256,
                target_sha256=target_sha256,
                teacher_graph_sha256=teacher_graph_sha256,
                sidecar_state_sha256=sidecar_state_sha256,
                supervised_token_count=supervised_token_count,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class IrisAblationControlledTeacherOutputV1:
    loss: torch.Tensor
    supervised_token_count: int
    workspace_trace: tuple[WorkspaceState, ...]
    diagnostic_trace: WorkspaceTraceV1
    messages: tuple[ModuleMessageSnapshotV1, ...]
    plan: CognitiveAblationControlPlanV1
    plan_sha256: str
    ablation_policy_sha256: str
    control_plan_sha256: str
    prompt_sha256: str
    target_sha256: str
    teacher_graph_sha256: str
    sidecar_state_sha256: str
    invocation_sha256: str
    kind: Literal["iris-ablation-controlled-teacher-output-v1"] = (
        "iris-ablation-controlled-teacher-output-v1"
    )

    def __post_init__(self) -> None:
        if (
            self.kind != "iris-ablation-controlled-teacher-output-v1"
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
            or type(self.workspace_trace) is not tuple
            or type(self.messages) is not tuple
            or type(self.diagnostic_trace) is not WorkspaceTraceV1
        ):
            raise ValueError("Exact detached mode/control teacher receipt required")
        plan = validate_ablation_control_plan_v1(self.plan)
        message_mode = type(plan.control) is WorkspaceMessagePlanV1
        if type(plan.control) is WorkspaceMessagePlanV1:
            control_sha = message_plan_sha256_v1(plan.control)
        else:
            assert type(plan.control) is WorkspaceInterventionPlanV1
            control_sha = intervention_plan_sha256_v1(plan.control)
        trace = WorkspaceTraceV1.model_validate(self.diagnostic_trace)
        if (
            self.plan_sha256 != ablation_control_plan_sha256_v1(plan)
            or self.ablation_policy_sha256 != cognitive_ablation_policy_sha256_v1(plan.ablation)
            or self.control_plan_sha256 != control_sha
            or trace.intervention_plan_sha256 != self.plan_sha256
            or len(self.workspace_trace) != trace.cycle_count
            or (message_mode and trace.options.mode != "full")
            or (not message_mode and self.messages)
            or self.invocation_sha256
            != controlled_teacher_invocation_sha256_v1(
                plan_sha256=self.plan_sha256,
                prompt_sha256=self.prompt_sha256,
                target_sha256=self.target_sha256,
                teacher_graph_sha256=self.teacher_graph_sha256,
                sidecar_state_sha256=self.sidecar_state_sha256,
                supervised_token_count=self.supervised_token_count,
            )
        ):
            raise ValueError("Mode/control receipt identities or trace differ")
        for state in self.workspace_trace:
            if (
                type(state) is not WorkspaceState
                or tuple(state.slots.shape) != (1, trace.num_slots, trace.state_dim)
                or any(
                    type(getattr(state, f.name)) is not torch.Tensor
                    or getattr(state, f.name).requires_grad
                    or getattr(state, f.name).is_inference()
                    for f in fields(state)
                )
            ):
                raise ValueError("Detached complete actual workspace states required")
        for digest in (
            self.plan_sha256,
            self.ablation_policy_sha256,
            self.control_plan_sha256,
            self.prompt_sha256,
            self.target_sha256,
            self.teacher_graph_sha256,
            self.sidecar_state_sha256,
            self.invocation_sha256,
        ):
            if type(digest) is not str or fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Lowercase SHA256 mode/control identities required")
        if message_mode:
            expected = tuple(
                (cycle, module)
                for cycle in range(1, trace.cycle_count + 1)
                for module in range(trace.num_modules)
            )
            if tuple((m.cycle, m.module) for m in self.messages) != expected:
                raise ValueError("All actual ordered mode-bound proposal snapshots required")
            for snapshot in self.messages:
                if type(snapshot) is not ModuleMessageSnapshotV1:
                    raise TypeError("Exact mode-bound proposal snapshot required")
                message = ModuleMessageSnapshotV1.model_validate(snapshot)
                if (
                    message.teacher_graph_sha256 != self.teacher_graph_sha256
                    or message.sidecar_state_sha256 != self.sidecar_state_sha256
                    or message.prompt_sha256 != self.prompt_sha256
                    or message.example_id != trace.options.example_id
                    or len(message.values) != trace.state_dim
                ):
                    raise ValueError("Proposal snapshot belongs to another mode/execution")


__all__ = ["IrisAblationControlledTeacherOutputV1"]
