"""Private prompt-only generation phase composition; NOT request authorization.

This reuses the closed neutral phase engines, not their teacher entrypoint or
teacher evidence. The ordinary owned request must keep rejecting this type.
A distinct generation owner/capability/receipt is still required to execute it.
"""

from dataclasses import asdict, dataclass, replace
from re import fullmatch
from typing import Any, Literal

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    _AblationControlSessionV1,
    validate_ablation_control_plan_v1,
)
from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

_DOMAIN = "iris/scientific-ablation-controlled-generation/v1"
_NO_TARGET = domain_sha256(_DOMAIN + "/no-target", b"")


@dataclass(frozen=True, slots=True)
class _GenerationControlBindingV1:
    """Data expected by a future native owner; never an execution capability."""

    artifact_sha256: str
    source_checkpoint_sha256: str
    lifecycle_sha256: str
    sidecar_state_sha256: str
    prompt_sha256: str
    geometry: tuple[int, int, int, int, int]  # modules, slots, width, cycles, winners

    def __post_init__(self) -> None:
        for value in (
            self.artifact_sha256,
            self.source_checkpoint_sha256,
            self.lifecycle_sha256,
            self.sidecar_state_sha256,
            self.prompt_sha256,
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Exact canonical generation source/prompt bindings required")
        if (
            type(self.geometry) is not tuple
            or len(self.geometry) != 5
            or any(type(value) is not int or value < 1 for value in self.geometry)
            or self.geometry[4] > self.geometry[0]
        ):
            raise ValueError("Exact positive generation geometry/winner bound required")


class _GenerationWorkspaceControlSessionV1:
    """One fresh prompt-only phase session; no labels, targets, loaders or callbacks.

    The reused private kernel's historical target argument receives ONLY a fixed
    generation-domain no-target sentinel. It is not a target digest or a teacher
    execution claim. The outer identity and graph domain are generation-specific.
    """

    __slots__ = ("_binding", "_binding_json", "_ids", "_kernel", "_plan_json", "_state")

    def __init__(
        self,
        ablation: CognitiveAblationPolicyV1,
        control: WorkspaceInterventionPlanV1,
        *,
        binding: _GenerationControlBindingV1,
    ) -> None:
        if type(binding) is not _GenerationControlBindingV1:
            raise TypeError("Exact prompt-only generation binding required")
        if type(control) is not WorkspaceInterventionPlanV1:
            raise TypeError("Exact workspace generation control required; no teacher/message alias")
        # Preserve the caller's raw nested tree until the existing exact-tree
        # validator checks it; ordinary Pydantic construction could launder extras.
        raw = CognitiveAblationControlPlanV1.model_construct(ablation=ablation, control=control)
        owned = validate_ablation_control_plan_v1(raw)
        self._binding = replace(binding)
        self._binding_json = canonical_json_bytes(asdict(self._binding))
        self._plan_json = canonical_json_bytes(owned.model_dump(mode="json"))
        self._kernel = _AblationControlSessionV1(owned, **self._arguments())
        self._ids = id(self._kernel), id(self._kernel.ablation), id(self._kernel.control)
        self._state: Literal["pending", "claimed", "completed", "failed"] = "pending"
        self._check_intent()

    def _arguments(self) -> dict[str, Any]:
        modules, slots, width, cycles, winners = self._binding.geometry
        graph = asdict(self._binding)
        del graph["prompt_sha256"]
        return dict(
            example_id="prompt:" + self._binding.prompt_sha256,
            prompt_sha256=self._binding.prompt_sha256,
            target_sha256=_NO_TARGET,
            teacher_graph_sha256=domain_sha256(_DOMAIN + "/graph", canonical_json_bytes(graph)),
            sidecar_state_sha256=self._binding.sidecar_state_sha256,
            num_modules=modules,
            num_slots=slots,
            state_dim=width,
            cycles=cycles,
            winners_per_cycle=winners,
        )

    def _check_intent(self) -> None:
        if type(self) is not _GenerationWorkspaceControlSessionV1:
            raise TypeError("Exact distinct generation session required")
        if self._state == "failed":
            raise ValueError("Generation session is permanently failed")
        try:
            if type(self._kernel) is not _AblationControlSessionV1:
                raise TypeError("Exact neutral phase kernel required")
            owned_plan = validate_ablation_control_plan_v1(self._kernel.plan)
            if (
                self._state not in {"pending", "claimed", "completed"}
                or type(self._binding) is not _GenerationControlBindingV1
                or type(self._kernel) is not _AblationControlSessionV1
                or type(self._kernel.ablation) is not CognitiveAblationSessionV1
                or type(self._kernel.control) is not WorkspaceInterventionSessionV1
                or self._ids
                != (id(self._kernel), id(self._kernel.ablation), id(self._kernel.control))
                or canonical_json_bytes(asdict(self._binding)) != self._binding_json
                or canonical_json_bytes(owned_plan.model_dump(mode="json")) != self._plan_json
                or self._kernel._context != tuple(self._arguments().values())
            ):
                raise ValueError("Generation session/source/intent/engine binding changed")
            self._binding.__post_init__()
            self._kernel.check_intent()
        except BaseException:
            self._state = "failed"
            raise

    @property
    def plan_sha256(self) -> str:
        self._check_intent()
        return domain_sha256(_DOMAIN + "/plan", self._plan_json)

    @property
    def ablation(self) -> CognitiveAblationSessionV1:
        self._check_intent()
        return self._kernel.ablation

    @property
    def workspace(self) -> WorkspaceInterventionSessionV1:
        self._check_intent()
        inner = self._kernel.control
        if type(inner) is not WorkspaceInterventionSessionV1:
            raise TypeError("Exact workspace generation phase engine required")
        return inner

    def claim(self, *, binding: _GenerationControlBindingV1) -> None:
        self._check_intent()
        try:
            if self._state != "pending" or type(binding) is not _GenerationControlBindingV1:
                raise ValueError("Generation session is not fresh or has another binding type")
            binding.__post_init__()
            if canonical_json_bytes(asdict(binding)) != self._binding_json:
                raise ValueError("Actual generation invocation differs from the owned binding")
            self._state = "claimed"
            self._kernel.claim(**self._arguments())
        except BaseException:
            self._state = "failed"
            raise

    def complete(self) -> None:
        """Check BOTH actually finished phase engines, not a caller completion flag."""
        self._check_intent()
        try:
            cognitive, workspace = self._kernel.ablation, self.workspace
            cycles = self._binding.geometry[3]
            if (
                self._state != "claimed"
                or cognitive._finished is not True
                or cognitive._failed is not False
                or type(cognitive._phase) is not int
                or cognitive._phase != cycles + 3
                or workspace._finished is not True
                or workspace._failed is not False
                or type(workspace._phase) is not int
                or workspace._phase != 0
                or type(workspace._cycle) is not int
                or workspace._cycle != cycles + 1
            ):
                raise ValueError("Generation phases did not both complete exactly once")
            self._state = "completed"
        except BaseException:
            self._state = "failed"
            raise
