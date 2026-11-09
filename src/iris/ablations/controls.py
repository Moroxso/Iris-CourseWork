"""Distinct closed mode+control intent and a fresh evaluation-only kernel session.

The old prohibition against passing an ablation alongside a legacy intervention
remains unchanged. Only this new exact session composes the two trusted phase
engines, and binds both identities to a new plan/graph domain.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.interventions.messages import (
    ModuleMessageSnapshotV1,
    WorkspaceMessagePlanV1,
    WorkspaceMessageSessionV1,
    message_plan_sha256_v1,
)
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class CognitiveAblationControlPlanV1(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    kind: Literal["iris-cognitive-ablation-control-plan-v1"] = (
        "iris-cognitive-ablation-control-plan-v1"
    )
    ablation: CognitiveAblationPolicyV1
    control: Annotated[
        WorkspaceInterventionPlanV1 | WorkspaceMessagePlanV1, Field(discriminator="kind")
    ]
    maximum_message_scalars: int | None = Field(default=None, ge=1, le=2_000_000)

    @model_validator(mode="after")
    def exact_budget(self) -> Self:
        if isinstance(self.control, WorkspaceMessagePlanV1) != (
            self.maximum_message_scalars is not None
        ):
            raise ValueError("Only message controls require an explicit message scalar budget")
        return self


def validate_ablation_control_plan_v1(value: object) -> CognitiveAblationControlPlanV1:
    """Reject bypassed/subclass/injected DTOs before taking a fresh owned snapshot."""
    allowed = (
        CognitiveAblationControlPlanV1,
        CognitiveAblationPolicyV1,
        WorkspaceInterventionPlanV1,
        WorkspaceInterventionSpecV1,
        WorkspaceMessagePlanV1,
        ModuleMessageSnapshotV1,
    )

    def exact(item: object) -> None:
        if isinstance(item, BaseModel):
            if (
                type(item) not in allowed
                or set(vars(item)) != set(type(item).model_fields)
                or item.__pydantic_extra__ is not None
                or item.__pydantic_private__ is not None
            ):
                raise TypeError("Exact complete closed mode/control model tree required")
            for child in vars(item).values():
                exact(child)
        elif type(item) is tuple:
            for child in item:
                exact(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise TypeError("Mode/control intent cannot contain callbacks or mutable containers")

    if type(value) is not CognitiveAblationControlPlanV1:
        raise TypeError("Exact distinct mode/control plan required")
    exact(value)
    owned = CognitiveAblationControlPlanV1.model_validate(value)
    # A newly validated model does not retain a caller-owned nested DTO.
    return CognitiveAblationControlPlanV1.model_validate_json(
        canonical_json_bytes(owned.model_dump(mode="json"))
    )


def ablation_control_plan_sha256_v1(value: CognitiveAblationControlPlanV1) -> str:
    owned = validate_ablation_control_plan_v1(value)
    return domain_sha256(
        "iris/cognitive-ablation-control-plan/v1",
        canonical_json_bytes(owned.model_dump(mode="json")),
    )


def ablation_control_graph_sha256_v1(
    *, backbone: str, cognitive: str, workspace: str, ablation: CognitiveAblationPolicyV1
) -> str:
    # Graph identity contains the cognitive mode, not the particular intervention:
    # paired same-mode donor messages must be reusable across recipient control plans.
    return domain_sha256(
        "iris/cognitive-ablation-control-teacher-graph/v1",
        canonical_json_bytes(
            {
                "backbone": backbone,
                "cognitive": cognitive,
                "workspace": workspace,
                "ablation_policy_sha256": cognitive_ablation_policy_sha256_v1(ablation),
            }
        ),
    )


class _AblationControlSessionV1:
    """Private fresh composition, never accepted by either legacy public method."""

    control: WorkspaceInterventionSessionV1 | WorkspaceMessageSessionV1

    def __init__(
        self,
        plan: CognitiveAblationControlPlanV1,
        *,
        example_id: str,
        prompt_sha256: str,
        target_sha256: str,
        teacher_graph_sha256: str,
        sidecar_state_sha256: str,
        num_modules: int,
        num_slots: int,
        state_dim: int,
        cycles: int,
        winners_per_cycle: int,
    ) -> None:
        self.plan = validate_ablation_control_plan_v1(plan)
        self.plan_sha256 = ablation_control_plan_sha256_v1(self.plan)
        self.policy_sha256 = cognitive_ablation_policy_sha256_v1(self.plan.ablation)
        geometry = dict(
            num_modules=num_modules,
            num_slots=num_slots,
            state_dim=state_dim,
            cycles=cycles,
            winners_per_cycle=winners_per_cycle,
        )
        self.ablation = CognitiveAblationSessionV1(
            self.plan.ablation,
            prompt_sha256=prompt_sha256,
            num_modules=num_modules,
            num_slots=num_slots,
            state_dim=state_dim,
            cycles=cycles,
        )
        if type(self.plan.control) is WorkspaceMessagePlanV1:
            assert self.plan.maximum_message_scalars is not None
            self.control_sha256 = message_plan_sha256_v1(self.plan.control)
            self.control = WorkspaceMessageSessionV1(
                self.plan.control,
                example_id=example_id,
                prompt_sha256=prompt_sha256,
                teacher_graph_sha256=teacher_graph_sha256,
                sidecar_state_sha256=sidecar_state_sha256,
                maximum_message_scalars=self.plan.maximum_message_scalars,
                **geometry,
            )
        else:
            assert type(self.plan.control) is WorkspaceInterventionPlanV1
            self.control_sha256 = intervention_plan_sha256_v1(self.plan.control)
            self.control = WorkspaceInterventionSessionV1(self.plan.control, **geometry)
        self._context = (
            example_id,
            prompt_sha256,
            target_sha256,
            teacher_graph_sha256,
            sidecar_state_sha256,
            num_modules,
            num_slots,
            state_dim,
            cycles,
            winners_per_cycle,
        )
        self._inner_ids = (id(self.ablation), id(self.control))
        self._claimed = False

    def claim(
        self,
        *,
        example_id: str,
        prompt_sha256: str,
        target_sha256: str,
        teacher_graph_sha256: str,
        sidecar_state_sha256: str,
        num_modules: int,
        num_slots: int,
        state_dim: int,
        cycles: int,
        winners_per_cycle: int,
    ) -> None:
        if type(self) is not _AblationControlSessionV1 or self._claimed is not False:
            raise ValueError("Mode/control session is closed or already consumed")
        self._claimed = True
        if (
            type(self) is not _AblationControlSessionV1
            or set(vars(self))
            != {
                "plan",
                "plan_sha256",
                "policy_sha256",
                "control_sha256",
                "ablation",
                "control",
                "_context",
                "_inner_ids",
                "_claimed",
            }
            or self._context
            != (
                example_id,
                prompt_sha256,
                target_sha256,
                teacher_graph_sha256,
                sidecar_state_sha256,
                num_modules,
                num_slots,
                state_dim,
                cycles,
                winners_per_cycle,
            )
            or self._inner_ids != (id(self.ablation), id(self.control))
            or self.plan_sha256 != ablation_control_plan_sha256_v1(self.plan)
            or self.policy_sha256 != cognitive_ablation_policy_sha256_v1(self.plan.ablation)
            or self.control.plan_sha256 != self.control_sha256
        ):
            raise ValueError("Mode/control session is not fresh or belongs to another execution")
        self.check_intent()
        self.ablation._validate_start(
            prompt_sha256=prompt_sha256,
            num_modules=num_modules,
            num_slots=num_slots,
            state_dim=state_dim,
            cycles=cycles,
        )
        inner = self.control
        if type(inner) is WorkspaceMessageSessionV1:
            if (
                type(self.plan.control) is not WorkspaceMessagePlanV1
                or inner._messages
                or inner._prepared != 0
                or inner._finished
                or inner._failed
                or inner._context
                != dict(
                    example_id=example_id,
                    prompt_sha256=prompt_sha256,
                    teacher_graph_sha256=teacher_graph_sha256,
                    sidecar_state_sha256=sidecar_state_sha256,
                )
            ):
                raise ValueError("Message control session is not fresh")
            inner = inner._inner
        elif type(inner) is not WorkspaceInterventionSessionV1:
            raise TypeError("Exact trusted phase engine required")
        if (
            inner._cycle != 1
            or inner._phase != 0
            or inner._failed
            or inner._finished
            or inner._geometry is not None
        ):
            raise ValueError("Workspace control session is not fresh")

    def check_intent(self) -> None:
        """Actual child engines retain captured intent even after completion."""
        if type(self.ablation) is not CognitiveAblationSessionV1:
            raise TypeError("Exact cognitive phase engine required")
        actual_control: WorkspaceMessagePlanV1 | WorkspaceInterventionPlanV1
        if type(self.control) is WorkspaceMessageSessionV1:
            actual_control = self.control.plan
        else:
            if type(self.control) is not WorkspaceInterventionSessionV1:
                raise TypeError("Exact workspace phase engine required")
            actual_control = self.control._plan
        actual = self.plan.model_copy(
            update={"ablation": self.ablation._policy, "control": actual_control}
        )
        if ablation_control_plan_sha256_v1(actual) != self.plan_sha256:
            raise ValueError("Actual child phase engines differ from the combined plan")


__all__ = [
    "CognitiveAblationControlPlanV1",
    "ablation_control_plan_sha256_v1",
    "validate_ablation_control_plan_v1",
]
