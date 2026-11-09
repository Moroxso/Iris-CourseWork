"""Distinct row-bound section17 cognitive-ablation five-term scientific graph.

This owns the existing source objective/probes without pretending to be the old
scientific-D model. Durable ablation optimizer/run/checkpoint/request IO is separate.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from iris.ablations import (
    CognitiveAblationPolicyV1,
    cognitive_ablation_policy_sha256_v1,
    validate_cognitive_ablation_policy_v1,
)
from iris.application.scientific_teacher import (
    IrisScientificObjectivePolicyV1,
    IrisScientificTeacherV1,
    _objective_from_trace_v1,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.data.schemas import IrisExample
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.losses import IrisLossBreakdown
from iris.training.model import IrisTeacherForcingModel
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherTracedOutputV1
from iris.training.probes import WorkspaceProbeHeads, prepare_probe_targets
from iris.training.usage import IrisWorkspaceUsageV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1

type ScientificAblationMode = Literal["training", "evaluation"]


class IrisScientificAblationTeacherError(RuntimeError):
    """Owned scientific ablation graph, row or objective did not complete."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationTeacherOutputV1:
    """Graph-attached complete objective, not a baseline or durable run receipt."""

    mode: ScientificAblationMode
    example_id: str
    example_sha256: str
    configuration_sha256: str
    supervised_token_count: int
    losses: IrisLossBreakdown
    usage: IrisWorkspaceUsageV1
    teacher: IrisAblatedTeacherOutputV1
    scientific_state_sha256: str
    invocation_sha256: str

    @property
    def trace(self) -> IrisTeacherTracedOutputV1:
        return self.teacher.traced

    @property
    def ablation_policy_sha256(self) -> str:
        return self.teacher.policy_sha256


class IrisScientificAblationTeacherV1(nn.Module):
    """Explicit exclusive owner of a D objective graph and immutable cognitive mode.

    Construction requires an exact train-mode scientific graph. Parameters and
    probe initialization are reused, not copied or reinitialized; unused ablated
    parameters remain registered. Historical owners reject this distinct type.
    Only complete owned rows supply prompt/target/labels. Training rejects held-out
    rows. Precision is explicit and ambient AMP/inference_mode are not supported.
    """

    configuration_identity: torch.Tensor

    def __init__(
        self, *, base: IrisScientificTeacherV1, ablation: CognitiveAblationPolicyV1
    ) -> None:
        super().__init__()
        if type(base) is not IrisScientificTeacherV1:
            raise IrisScientificAblationTeacherError("Exact scientific base required")
        try:
            base._validate_boundary(training=True)
            owned = validate_cognitive_ablation_policy_v1(ablation)
            ablation_sha256 = cognitive_ablation_policy_sha256_v1(owned)
            configuration = {
                "kind": "iris-scientific-ablation-teacher-v1",
                "arm_id": "D",
                "protocol_role": "post_main_ablation",
                "base_configuration": json.loads(base.configuration_json),
                "base_configuration_sha256": base.configuration_sha256,
                "ablation": owned.model_dump(mode="json"),
                "ablation_policy_sha256": ablation_sha256,
            }
        except Exception:
            raise IrisScientificAblationTeacherError("Ablation graph construction failed") from None
        self._ablation_json = owned.model_dump_json()
        self._ablation_sha256 = ablation_sha256
        self._base_configuration_sha256 = base.configuration_sha256
        self._configuration_json = canonical_json_bytes(configuration).decode("utf-8")
        self._configuration_sha256 = domain_sha256(
            "iris/scientific-ablation-teacher-configuration/v1",
            self._configuration_json.encode("utf-8"),
        )
        self.base = base
        self._base_id = id(base)
        self._parameter_identity = tuple((name, id(p)) for name, p in self.named_parameters())
        self.register_buffer(
            "configuration_identity",
            torch.tensor(
                list(bytes.fromhex(self._configuration_sha256)),
                dtype=torch.uint8,
                device=self.teacher.backbone.device,
            ),
        )
        self.register_load_state_dict_pre_hook(self._check_loading_identity)  # type: ignore[no-untyped-call]
        self._validate_boundary(training=True)

    @property
    def teacher(self) -> IrisTeacherForcingModel:
        return self.base.teacher

    @property
    def probes(self) -> WorkspaceProbeHeads:
        return self.base.probes

    @property
    def policy(self) -> IrisScientificObjectivePolicyV1:
        return self.base.policy

    @property
    def ablation(self) -> CognitiveAblationPolicyV1:
        return CognitiveAblationPolicyV1.model_validate_json(self._ablation_json)

    @property
    def ablation_policy_sha256(self) -> str:
        return self._ablation_sha256

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def configuration_json(self) -> str:
        if (
            domain_sha256(
                "iris/scientific-ablation-teacher-configuration/v1",
                self._configuration_json.encode("utf-8"),
            )
            != self._configuration_sha256
        ):
            raise IrisScientificAblationTeacherError("Scientific ablation configuration changed")
        return self._configuration_json

    def _check_loading_identity(
        self,
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Reject mode/base/probe mismatch before ANY child weights can be copied.
        matches = IrisScientificTeacherV1._matches_identity
        if (
            module is not self
            or local_metadata.get("assign_to_params_buffers", False)
            or not matches(
                state_dict.get(prefix + "configuration_identity"), self.configuration_sha256
            )
            or not matches(
                state_dict.get(prefix + "base.configuration_identity"),
                self._base_configuration_sha256,
            )
            or not matches(
                state_dict.get(prefix + "base.probes.configuration_identity"),
                self.probes.configuration_sha256,
            )
        ):
            raise IrisScientificAblationTeacherError("Scientific ablation state identity differs")

    def _validate_boundary(self, *, training: bool) -> None:
        if (
            tuple(self._modules) != ("base",)
            or type(self.base) is not IrisScientificTeacherV1
            or id(self.base) != self._base_id
            or tuple(self._buffers) != ("configuration_identity",)
            or any(module.training is not training for module in self.modules())
            or self.base.configuration_sha256 != self._base_configuration_sha256
            or cognitive_ablation_policy_sha256_v1(self.ablation) != self._ablation_sha256
            or not IrisScientificTeacherV1._matches_identity(
                self.configuration_identity, self.configuration_sha256
            )
            or tuple((name, id(p)) for name, p in self.named_parameters(remove_duplicate=False))
            != self._parameter_identity
        ):
            raise IrisScientificAblationTeacherError("Scientific ablation ownership changed")
        # Both complete snapshots remain in their respective immutable domains.
        _ = self.configuration_json, self.base.configuration_json
        self.base._validate_boundary(training=training)

    def forward(
        self, example: IrisExample, *, terminal_token_id: int
    ) -> IrisScientificAblationTeacherOutputV1:
        return self.training_forward(example, terminal_token_id=terminal_token_id)

    def training_forward(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificAblationTeacherOutputV1:
        return self._forward_owned(
            example, terminal_token_id=terminal_token_id, training=True, diagnostics=diagnostics
        )

    def evaluation_forward(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificAblationTeacherOutputV1:
        return self._forward_owned(
            example, terminal_token_id=terminal_token_id, training=False, diagnostics=diagnostics
        )

    def _forward_owned(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        training: bool,
        diagnostics: WorkspaceTraceOptionsV1 | None,
    ) -> IrisScientificAblationTeacherOutputV1:
        caller = None
        try:
            self._validate_boundary(training=training)
            caller = capture_runtime_state(self.teacher.backbone.device)
            if training:
                return self._execute(
                    example,
                    terminal_token_id=terminal_token_id,
                    training=True,
                    diagnostics=diagnostics,
                )
            with torch.no_grad():
                return self._execute(
                    example,
                    terminal_token_id=terminal_token_id,
                    training=False,
                    diagnostics=diagnostics,
                )
        except Exception:
            raise IrisScientificAblationTeacherError("Scientific ablation forward failed") from None
        finally:
            if caller is not None:
                restore_runtime_state(caller)

    def _execute(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        training: bool,
        diagnostics: WorkspaceTraceOptionsV1 | None,
    ) -> IrisScientificAblationTeacherOutputV1:
        if type(example) is not IrisExample:
            raise ValueError("Exact complete IrisExample required")
        row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
        if training and row.split != "train":
            raise ValueError("Held-out rows cannot train ablation models")
        if diagnostics is not None:
            if type(diagnostics) is not WorkspaceTraceOptionsV1:
                raise ValueError("Exact diagnostics required")
            diagnostics = WorkspaceTraceOptionsV1.model_validate(
                diagnostics.model_dump(mode="python", warnings=False)
            )
            if diagnostics.example_id != row.id:
                raise ValueError("Diagnostic row differs")
        row_sha256 = domain_sha256(
            "iris/scientific-ablation-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
        )
        state_sha256 = module_state_sha256((("scientific_ablation_teacher", self),))
        tensors = tuple(self.parameters()) + tuple(self.buffers())
        versions = tuple(p._version for p in tensors)
        labels = prepare_probe_targets(
            (row,),
            vocabulary=self.probes.vocabulary,
            purpose="training" if training else "analysis",
            device=self.teacher.backbone.device,
        )
        prepared = prepare_teacher_forcing_input(
            backbone=self.teacher.backbone,
            prompt=row.prompt,
            target_text=row.expected_response,
            terminal_token_id=terminal_token_id,
        )
        self._validate_boundary(training=training)
        if tuple(p._version for p in tensors) != versions:
            raise ValueError("Preparation changed registered state")
        mode: ScientificAblationMode = "training" if training else "evaluation"
        provenance = (
            self.teacher.checkpoint_provenance if training else self.teacher.evaluation_provenance
        )
        if prepared.prompt != row.prompt:
            raise ValueError("Preparation changed owned prompt")
        prompt_sha256 = domain_sha256(
            "iris/cognitive-ablation-prompt/v1", row.prompt.encode("utf-8")
        )
        target_sha256 = teacher_target_sha256_v1(
            prepared.target_input_ids, prepared.target_attention_mask
        )
        teacher_graph_sha256 = domain_sha256(
            "iris/cognitive-ablation-teacher-graph/v1",
            canonical_json_bytes(
                {
                    "backbone": provenance.backbone_configuration_json,
                    "cognitive": provenance.cognitive_configuration_json,
                    "workspace": provenance.workspace_configuration_json,
                    "policy": self.ablation.model_dump(mode="json"),
                }
            ),
        )
        expected_teacher_invocation = domain_sha256(
            "iris/cognitive-ablation-teacher-invocation/v1",
            canonical_json_bytes(
                {
                    "mode": mode,
                    "policy_sha256": self.ablation_policy_sha256,
                    "prompt_sha256": prompt_sha256,
                    "target_sha256": target_sha256,
                    "teacher_graph_sha256": teacher_graph_sha256,
                    "sidecar_state_sha256": provenance.sidecar_state_sha256,
                }
            ),
        )
        result = (
            self.teacher.training_forward_ablated(
                prepared, policy=self.ablation, diagnostics=diagnostics
            )
            if training
            else self.teacher.evaluation_forward_ablated(
                prepared, policy=self.ablation, diagnostics=diagnostics
            )
        )
        if (
            type(result) is not IrisAblatedTeacherOutputV1
            or result.mode != mode
            or result.policy_sha256 != self.ablation_policy_sha256
            or cognitive_ablation_policy_sha256_v1(result.policy) != self.ablation_policy_sha256
            or result.prompt_sha256 != prompt_sha256
            or result.target_sha256 != target_sha256
            or result.teacher_graph_sha256 != teacher_graph_sha256
            or result.sidecar_state_sha256 != provenance.sidecar_state_sha256
            or result.invocation_sha256 != expected_teacher_invocation
            or type(result.traced) is not IrisTeacherTracedOutputV1
            or type(result.traced.teacher) is not IrisTeacherForcedOutput
            or result.traced.teacher.supervised_token_count
            != int(prepared.target_attention_mask.sum())
        ):
            raise ValueError("Teacher result belongs to a different ablation")
        terms, usage = _objective_from_trace_v1(
            result.traced,
            cycles=self.teacher.proposal_head.cycles,
            probes=self.probes,
            labels=labels,
            policy=self.policy,
        )
        self._validate_boundary(training=training)
        if (
            tuple(p._version for p in tensors) != versions
            or module_state_sha256((("scientific_ablation_teacher", self),)) != state_sha256
            or terms.total.requires_grad is not training
            or (training and terms.total.grad_fn is None)
        ):
            raise ValueError("Scientific objective changed state or gradient mode")
        invocation = domain_sha256(
            "iris/scientific-ablation-invocation/v1",
            canonical_json_bytes(
                {
                    "mode": mode,
                    "example_sha256": row_sha256,
                    "configuration_sha256": self.configuration_sha256,
                    "teacher_invocation_sha256": result.invocation_sha256,
                    "scientific_state_sha256": state_sha256,
                }
            ),
        )
        return IrisScientificAblationTeacherOutputV1(
            mode=mode,
            example_id=row.id,
            example_sha256=row_sha256,
            configuration_sha256=self.configuration_sha256,
            supervised_token_count=result.traced.teacher.supervised_token_count,
            losses=terms,
            usage=usage,
            teacher=result,
            scientific_state_sha256=state_sha256,
            invocation_sha256=invocation,
        )


__all__ = [
    "IrisScientificAblationTeacherError",
    "IrisScientificAblationTeacherOutputV1",
    "IrisScientificAblationTeacherV1",
]
