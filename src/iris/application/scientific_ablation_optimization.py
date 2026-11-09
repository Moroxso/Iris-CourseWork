"""Mode-bound two-pass scientific ablation optimization, without durable run IO.

The numerical kernel is shared privately with the historical baseline owner.
Public types and configuration/window identities remain distinct. Disconnected
parameters stay registered, with zero AdamW counters and no manufactured gradients.
"""

import json
from dataclasses import dataclass, replace
from typing import Literal

from iris.ablations import cognitive_ablation_policy_sha256_v1
from iris.application.scientific_ablation_teacher import (
    IrisScientificAblationTeacherOutputV1,
    IrisScientificAblationTeacherV1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizationError,
    IrisScientificOptimizerPolicyV1,
    _Observation,
    _ScientificOptimizerCore,
    _ScientificTeacherOutput,
    _WindowValues,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.data.schemas import IrisExample
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import RuntimeStateSnapshot


class IrisScientificAblationOptimizationError(IrisScientificOptimizationError):
    """A mode-bound window failed; uncertain stateful writes poison its owner."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationOptimizationOutputV1:
    """Detached committed-window measurements, not a durable checkpoint receipt."""

    step_index: int
    example_count: int
    supervised_token_count: int
    learning_rate: float
    gradient_l2_norm: float
    gradients_clipped: bool
    loss: float
    language_model_loss: float
    probe_loss: float
    diversity_loss: float
    utilization_loss: float
    norm_loss: float
    configuration_sha256: str
    window_sha256: str
    purpose: Literal["scientific", "smoke"]
    ablation_policy_sha256: str
    model_configuration_sha256: str
    scientific_state_before_sha256: str
    scientific_state_after_sha256: str
    registered_parameter_count: int
    gradient_present_parameter_count: int
    nonzero_gradient_parameter_count_before_clip: int


class IrisScientificAblationOptimizerV1(
    _ScientificOptimizerCore[
        IrisScientificAblationTeacherV1, IrisScientificAblationOptimizationOutputV1
    ]
):
    """Exclusive FP32 owner of an exact scientific cognitive-ablation teacher.

    Only complete bounded unique train windows are accepted. Source-sized windows
    use the unchanged numeric policy, window-mean utilization KL and one live graph.
    The cheap frozen-backbone revision guard is process-local, not a durable weights
    hash; future checkpoint/resume must bind the actual pretrained frozen runtime.
    """

    def __init__(
        self, *, model: IrisScientificAblationTeacherV1, policy: IrisScientificOptimizerPolicyV1
    ) -> None:
        if (
            type(model) is not IrisScientificAblationTeacherV1
            or type(policy) is not IrisScientificOptimizerPolicyV1
        ):
            raise IrisScientificAblationOptimizationError(
                "Exact scientific ablation model and numeric policy required"
            )
        try:
            model._validate_boundary(training=True)
            self._ablation_identity = model.ablation_policy_sha256
            self._backbone_revision = model.teacher.backbone.checkpoint_revision_token
            base_configuration = json.loads(model.base.configuration_json)
            self._teacher_graph_identity = domain_sha256(
                "iris/cognitive-ablation-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": base_configuration["backbone_configuration_json"],
                        "cognitive": base_configuration["cognitive_configuration_json"],
                        "workspace": base_configuration["workspace_configuration_json"],
                        "policy": model.ablation.model_dump(mode="json"),
                    }
                ),
            )
            # Initialized before the core invokes this variant's contract guard.
            self._committed_scientific_state_sha256 = module_state_sha256(
                (("scientific_ablation_teacher", model),)
            )
            self._pending_update_state_sha256: str | None = None
            self._active_terminal_token_id: int | None = None
            self._active_teacher_state_sha256: str | None = None
            self._attached_run_token: object | None = None
            super().__init__(
                model=model,
                policy=policy,
                configuration_domain="iris/scientific-ablation-optimizer-configuration/v1",
                configuration_extra={
                    "kind": "iris-scientific-ablation-optimizer-v1",
                    "ablation_policy_sha256": self._ablation_identity,
                    "ablation": model.ablation.model_dump(mode="json"),
                },
            )
        except Exception:
            raise IrisScientificAblationOptimizationError(
                "Scientific ablation optimizer attachment failed"
            ) from None

    @property
    def configuration_json(self) -> str:
        if (
            domain_sha256(
                "iris/scientific-ablation-optimizer-configuration/v1",
                self._configuration_json.encode("utf-8"),
            )
            != self.configuration_sha256
        ):
            raise IrisScientificAblationOptimizationError(
                "Ablation optimizer configuration changed"
            )
        return self._configuration_json

    def _state_sha256(self) -> str:
        return module_state_sha256((("scientific_ablation_teacher", self._model),))

    def _validate_contract(self, *, parameter_steps: tuple[int, ...] | None = None) -> None:
        super()._validate_contract(parameter_steps=parameter_steps)
        _ = self.configuration_json
        if (
            self._model.ablation_policy_sha256 != self._ablation_identity
            or self._model.teacher.backbone.checkpoint_revision_token != self._backbone_revision
            or (
                parameter_steps is None
                and self._state_sha256() != self._committed_scientific_state_sha256
            )
        ):
            self._poisoned = True
            raise IrisScientificAblationOptimizationError("Owned ablation graph/state changed")

    def step(
        self, examples: tuple[IrisExample, ...], *, terminal_token_id: int
    ) -> IrisScientificAblationOptimizationOutputV1:
        # Do not alter the outer call's transient provenance during reentry.
        if self._busy:
            raise IrisScientificAblationOptimizationError("Concurrent/reentrant windows forbidden")
        try:
            self._validate_contract()
            if (
                type(terminal_token_id) is not int
                or terminal_token_id not in self._model.teacher.backbone.eos_token_ids
            ):
                raise IrisScientificAblationOptimizationError("Exact declared terminal required")
            self._active_terminal_token_id = terminal_token_id
            self._active_teacher_state_sha256 = (
                self._model.teacher.checkpoint_provenance.sidecar_state_sha256
            )
            return super().step(examples, terminal_token_id=terminal_token_id)
        except BaseException as error:
            if self._failure_was_stateful():
                self._poisoned = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificAblationOptimizationError(
                "Scientific ablation optimizer window could not complete"
            ) from None
        finally:
            self._active_terminal_token_id = None
            self._active_teacher_state_sha256 = None
            self._pending_update_state_sha256 = None

    def _observe(
        self, output: _ScientificTeacherOutput, row: IrisExample, before: RuntimeStateSnapshot
    ) -> _Observation:
        if type(output) is not IrisScientificAblationTeacherOutputV1:
            raise IrisScientificAblationOptimizationError(
                "Exact scientific ablation output required"
            )
        common = self._observe_common(
            output,
            row,
            before,
            output_type=IrisScientificAblationTeacherOutputV1,
            example_domain="iris/scientific-ablation-example/v1",
        )
        result = output.teacher
        if (
            output.mode != "training"
            or output.ablation_policy_sha256 != self._ablation_identity
            or output.scientific_state_sha256 != self._committed_scientific_state_sha256
            or type(result) is not IrisAblatedTeacherOutputV1
            or self._active_terminal_token_id is None
            or self._active_teacher_state_sha256 is None
        ):
            raise IrisScientificAblationOptimizationError("Scientific ablation provenance differs")
        prepared = prepare_teacher_forcing_input(
            backbone=self._model.teacher.backbone,
            prompt=row.prompt,
            target_text=row.expected_response,
            terminal_token_id=self._active_terminal_token_id,
        )
        prompt_sha256 = domain_sha256(
            "iris/cognitive-ablation-prompt/v1", row.prompt.encode("utf-8")
        )
        target_sha256 = teacher_target_sha256_v1(
            prepared.target_input_ids, prepared.target_attention_mask
        )
        invocation = domain_sha256(
            "iris/cognitive-ablation-teacher-invocation/v1",
            canonical_json_bytes(
                {
                    "mode": "training",
                    "policy_sha256": self._ablation_identity,
                    "prompt_sha256": prompt_sha256,
                    "target_sha256": target_sha256,
                    "teacher_graph_sha256": self._teacher_graph_identity,
                    "sidecar_state_sha256": self._active_teacher_state_sha256,
                }
            ),
        )
        scientific_invocation = domain_sha256(
            "iris/scientific-ablation-invocation/v1",
            canonical_json_bytes(
                {
                    "mode": "training",
                    "example_sha256": common.identity,
                    "configuration_sha256": self._model_identity,
                    "teacher_invocation_sha256": invocation,
                    "scientific_state_sha256": self._committed_scientific_state_sha256,
                }
            ),
        )
        if (
            result.mode != "training"
            or cognitive_ablation_policy_sha256_v1(result.policy) != self._ablation_identity
            or result.prompt_sha256 != prompt_sha256
            or result.target_sha256 != target_sha256
            or result.teacher_graph_sha256 != self._teacher_graph_identity
            or result.sidecar_state_sha256 != self._active_teacher_state_sha256
            or result.invocation_sha256 != invocation
            or output.invocation_sha256 != scientific_invocation
            or output.supervised_token_count != int(prepared.target_attention_mask.sum())
            or result.traced.teacher.supervised_token_count != output.supervised_token_count
        ):
            raise IrisScientificAblationOptimizationError("Ablation row/invocation/state differs")
        return replace(common, invocation_sha256=scientific_invocation)

    def _gradient_participation(self) -> tuple[int, int, int]:
        parameters = self._topology.parameters
        registered = sum(parameter.numel() for parameter in parameters)
        present = sum(parameter.numel() for parameter in parameters if parameter.grad is not None)
        nonzero = sum(
            int(parameter.grad.count_nonzero())
            for parameter in parameters
            if parameter.grad is not None
        )
        if not 0 <= nonzero <= present <= registered:
            raise IrisScientificAblationOptimizationError("Invalid gradient participation")
        return registered, present, nonzero

    def _make_output(
        self, values: _WindowValues, observations: list[_Observation], terminal_token_id: int
    ) -> IrisScientificAblationOptimizationOutputV1:
        if values.participation is None:
            raise IrisScientificAblationOptimizationError("Missing ablation gradient participation")
        registered, present, nonzero = values.participation
        after = self._state_sha256()
        self._pending_update_state_sha256 = after
        return IrisScientificAblationOptimizationOutputV1(
            step_index=values.step_index,
            example_count=values.example_count,
            supervised_token_count=values.supervised_token_count,
            learning_rate=values.learning_rate,
            gradient_l2_norm=values.gradient_l2_norm,
            gradients_clipped=values.gradients_clipped,
            loss=values.loss,
            language_model_loss=values.language_model_loss,
            probe_loss=values.probe_loss,
            diversity_loss=values.diversity_loss,
            utilization_loss=values.utilization_loss,
            norm_loss=values.norm_loss,
            configuration_sha256=self.configuration_sha256,
            window_sha256=domain_sha256(
                "iris/scientific-ablation-optimizer-window/v1",
                canonical_json_bytes(
                    {
                        "rows": [item.identity for item in observations],
                        "target_counts": [item.token_count for item in observations],
                        "invocations": [item.invocation_sha256 for item in observations],
                        "terminal_token_id": terminal_token_id,
                        "model_configuration_sha256": self._model_identity,
                        "ablation_policy_sha256": self._ablation_identity,
                        "scientific_state_before_sha256": self._committed_scientific_state_sha256,
                    }
                ),
            ),
            purpose=self._policy.purpose,
            ablation_policy_sha256=self._ablation_identity,
            model_configuration_sha256=self._model_identity,
            scientific_state_before_sha256=self._committed_scientific_state_sha256,
            scientific_state_after_sha256=after,
            registered_parameter_count=registered,
            gradient_present_parameter_count=present,
            nonzero_gradient_parameter_count_before_clip=nonzero,
        )

    def _finish_success(self) -> None:
        if (
            self._pending_update_state_sha256 is None
            or self._state_sha256() != self._pending_update_state_sha256
        ):
            raise IrisScientificAblationOptimizationError(
                "Ablation commit state changed in cleanup"
            )
        self._committed_scientific_state_sha256 = self._pending_update_state_sha256

    def _failure_was_stateful(self) -> bool:
        try:
            return (
                self._state_sha256() != self._committed_scientific_state_sha256
                or self._model.teacher.backbone.checkpoint_revision_token != self._backbone_revision
            )
        except Exception:
            return True


__all__ = [
    "IrisScientificAblationOptimizationError",
    "IrisScientificAblationOptimizationOutputV1",
    "IrisScientificAblationOptimizerV1",
]
