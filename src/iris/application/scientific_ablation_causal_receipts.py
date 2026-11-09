"""Private native-pass retention seals and faithful ALL7 JSON projections.

Seals are taken immediately when a trusted teacher returns, before a later row or
trial can mutate retained tensors. They are not execution attestations or public
reports. No prompt, target IDs, logits or private RNG enter the JSON projection.
"""

from dataclasses import dataclass, fields
from re import fullmatch
from typing import Any

import torch

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import _own
from iris.interventions.messages import ModuleMessageSnapshotV1
from iris.interventions.state import validate_intervention_state
from iris.training.ablation_controls import IrisAblationControlledTeacherOutputV1
from iris.training.ablation_observation import (
    AblationWorkspaceWriteObservationV1,
    IrisAblationObservedTeacherOutputV1,
)
from iris.training.outputs import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceV1


def _state_json(state: WorkspaceState) -> dict[str, Any]:
    if type(state) is not WorkspaceState:
        raise TypeError("Exact actual ALL7 state required")
    validate_intervention_state(state)
    result = {}
    for field in fields(WorkspaceState):
        tensor = getattr(state, field.name)
        if type(tensor) is not torch.Tensor or tensor.requires_grad or tensor.grad_fn is not None:
            raise ValueError("Actual detached native state tensors required")
        result[field.name] = dict(
            values=tensor.detach().cpu().tolist(),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=list(tensor.shape),
            tensor_sha256=tensor_sha256(tensor),
        )
    return result


def _observed_json(output: IrisAblationObservedTeacherOutputV1) -> dict[str, Any]:
    """Own exact metadata before invoking receipt validators or serializers."""
    if (
        type(output) is not IrisAblationObservedTeacherOutputV1
        or type(output.controlled) is not IrisAblationControlledTeacherOutputV1
        or type(output.observations) is not tuple
        or any(type(o) is not AblationWorkspaceWriteObservationV1 for o in output.observations)
    ):
        raise TypeError("Exact native observed teacher receipt required")
    controlled = output.controlled
    plan = _own(controlled.plan, CognitiveAblationControlPlanV1)
    trace = _own(controlled.diagnostic_trace, WorkspaceTraceV1)
    if type(controlled.messages) is not tuple:
        raise ValueError("Exact native proposal inventory required")
    messages = tuple(_own(m, ModuleMessageSnapshotV1) for m in controlled.messages)
    output.__post_init__()
    return dict(
        kind=output.kind,
        controlled=dict(
            kind=controlled.kind,
            loss=float(controlled.loss),
            supervised_token_count=controlled.supervised_token_count,
            plan=plan.model_dump(mode="json"),
            plan_sha256=controlled.plan_sha256,
            ablation_policy_sha256=controlled.ablation_policy_sha256,
            control_plan_sha256=controlled.control_plan_sha256,
            prompt_sha256=controlled.prompt_sha256,
            target_sha256=controlled.target_sha256,
            teacher_graph_sha256=controlled.teacher_graph_sha256,
            sidecar_state_sha256=controlled.sidecar_state_sha256,
            invocation_sha256=controlled.invocation_sha256,
            trace=trace.model_dump(mode="json"),
            messages=[m.model_dump(mode="json") for m in messages],
            states=[_state_json(s) for s in controlled.workspace_trace],
        ),
        maximum_observation_scalars=output.maximum_observation_scalars,
        actual_observation_scalars=output.actual_observation_scalars,
        invocation_sha256=output.invocation_sha256,
        observations=[
            dict(
                cycle=o.cycle,
                before_write=_state_json(o.before_write),
                native_written=_state_json(o.native_written),
                before_write_sha256=o.before_write_sha256,
                native_written_sha256=o.native_written_sha256,
            )
            for o in output.observations
        ],
    )


def _tensors(
    output: IrisAblationObservedTeacherOutputV1, prepared: IrisTeacherForcingInput
) -> tuple[torch.Tensor, ...]:
    return (
        prepared.target_input_ids,
        prepared.target_attention_mask,
        output.controlled.loss,
        *(
            getattr(state, field.name)
            for state in (
                *output.controlled.workspace_trace,
                *(s for o in output.observations for s in (o.before_write, o.native_written)),
            )
            for field in fields(WorkspaceState)
        ),
    )


def _fingerprint(
    output: IrisAblationObservedTeacherOutputV1, prepared: IrisTeacherForcingInput
) -> str:
    if type(prepared) is not IrisTeacherForcingInput or type(prepared.prompt) is not str:
        raise TypeError("Original exact privately prepared teacher input required")
    return domain_sha256(
        "iris/scientific-ablation-native-pass-retention/v1",
        canonical_json_bytes(
            dict(
                output=_observed_json(output),
                prompt_sha256=domain_sha256(
                    "iris/private-native-prompt/v1", prepared.prompt.encode()
                ),
                target_input_ids_sha256=tensor_sha256(prepared.target_input_ids),
                target_attention_mask_sha256=tensor_sha256(prepared.target_attention_mask),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _ObservedPassSeal:
    output: IrisAblationObservedTeacherOutputV1
    prepared: IrisTeacherForcingInput
    fingerprint: str
    tensors: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self) is not _ObservedPassSeal
            or type(self.output) is not IrisAblationObservedTeacherOutputV1
            or type(self.prepared) is not IrisTeacherForcingInput
            or type(self.fingerprint) is not str
            or fullmatch(r"[0-9a-f]{64}", self.fingerprint) is None
            or type(self.tensors) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or pair[0] <= 0
                or pair[1] < 0
                for pair in self.tensors
            )
        ):
            raise ValueError("Exact original native retention metadata families required")

    def audit(self) -> None:
        self.__post_init__()
        if type(self) is not _ObservedPassSeal or (
            _fingerprint(self.output, self.prepared) != self.fingerprint
            or tuple((id(t), t._version) for t in _tensors(self.output, self.prepared))
            != self.tensors
        ):
            raise ValueError("Original retained native pass/input changed after teacher return")


def _seal_observed(
    output: IrisAblationObservedTeacherOutputV1, prepared: IrisTeacherForcingInput
) -> _ObservedPassSeal:
    fingerprint = _fingerprint(output, prepared)
    return _ObservedPassSeal(
        output,
        prepared,
        fingerprint,
        tuple((id(t), t._version) for t in _tensors(output, prepared)),
    )
