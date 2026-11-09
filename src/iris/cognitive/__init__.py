"""Trainable cognitive components for Iris."""

from iris.cognitive.goal_encoder import GoalEncoder, GoalEncoderError
from iris.cognitive.layer_sampler import LayerSampler, LayerSamplerError
from iris.cognitive.module_input_router import ModuleInputRouter, ModuleInputRouterError
from iris.cognitive.module_latent_bridge import ModuleLatentBridge, ModuleLatentBridgeError
from iris.cognitive.modules import CognitiveModule, CognitiveModuleError
from iris.cognitive.no_workspace_topology import NO_WORKSPACE_COGNITIVE_CYCLES_V1
from iris.cognitive.outputs import (
    CognitiveModuleOutput,
    GoalEncoderOutput,
    LayerSamplerOutput,
    ModuleInputRouterOutput,
    ModuleLatentBridgeOutput,
    ModuleProposal,
)
from iris.cognitive.proposal_head import ProposalHead, ProposalHeadError

__all__ = [
    "NO_WORKSPACE_COGNITIVE_CYCLES_V1",
    "CognitiveModule",
    "CognitiveModuleError",
    "CognitiveModuleOutput",
    "GoalEncoder",
    "GoalEncoderError",
    "GoalEncoderOutput",
    "LayerSampler",
    "LayerSamplerError",
    "LayerSamplerOutput",
    "ModuleInputRouter",
    "ModuleInputRouterError",
    "ModuleInputRouterOutput",
    "ModuleLatentBridge",
    "ModuleLatentBridgeError",
    "ModuleLatentBridgeOutput",
    "ModuleProposal",
    "ProposalHead",
    "ProposalHeadError",
]
