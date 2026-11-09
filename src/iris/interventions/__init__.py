"""Source section18 intervention intent and phase-aware inference primitives.

The D orchestrator has a distinct intervened-request path. Scientific held-out
paired capture and answer-quality reporting exist separately from these primitives.
Teacher intervention NLL and scientific causal acceptance remain unmeasured.
"""

from iris.interventions.phases import (
    BlockModuleRead,
    BlockModuleWrite,
    ForceModuleWinner,
    ScaleModuleBid,
)
from iris.interventions.policy import (
    InterventionKindV1,
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
    validate_intervention_plan_v1,
)
from iris.interventions.state import (
    FreezeSlot,
    GaussianNoise,
    ReduceWorkspaceCapacity,
    ReplaceSlot,
    ShuffleSlots,
    SwapSlots,
    WorkspaceIntervention,
    WorkspaceInterventionError,
    ZeroSlot,
    ZeroWorkspace,
    validate_intervention_state,
)

__all__ = [
    "BlockModuleRead",
    "BlockModuleWrite",
    "ForceModuleWinner",
    "FreezeSlot",
    "GaussianNoise",
    "InterventionKindV1",
    "ReduceWorkspaceCapacity",
    "ReplaceSlot",
    "ScaleModuleBid",
    "ShuffleSlots",
    "SwapSlots",
    "WorkspaceIntervention",
    "WorkspaceInterventionError",
    "WorkspaceInterventionPlanV1",
    "WorkspaceInterventionSpecV1",
    "ZeroSlot",
    "ZeroWorkspace",
    "intervention_plan_sha256_v1",
    "validate_intervention_plan_v1",
    "validate_intervention_state",
]
