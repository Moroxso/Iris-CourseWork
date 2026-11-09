"""Typed inputs and outputs for Iris teacher-forced execution."""

from dataclasses import dataclass
from typing import Literal

import torch

from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceV1


@dataclass(frozen=True, slots=True)
class IrisTeacherForcingInput:
    """One prompt and one explicitly prepared teacher target."""

    prompt: str
    target_input_ids: torch.Tensor
    target_attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class IrisTeacherForcedOutput:
    """Mean natural-log causal loss over counted target tokens and its exact trace."""

    loss: torch.Tensor
    supervised_token_count: int
    workspace_trace: tuple[WorkspaceState, ...]
    diagnostic_trace: WorkspaceTraceV1 | None = None


@dataclass(frozen=True, slots=True)
class IrisTeacherCycleTraceV1:
    """Call-owned graph copies before each write; hard masks are diagnostics only."""

    module_probabilities: torch.Tensor
    allocation_scores: torch.Tensor
    winner_mask: torch.Tensor
    assignment_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class IrisTeacherTracedOutputV1:
    """Opt-in auxiliary-training output, not a change to the legacy LM-only DTO."""

    teacher: IrisTeacherForcedOutput
    cycles: tuple[IrisTeacherCycleTraceV1, ...]


@dataclass(frozen=True, slots=True)
class IrisTeacherModelCheckpointProvenance:
    """Validated configuration and placement identity for sidecar checkpoints."""

    backbone_configuration_json: str
    cognitive_configuration_json: str
    workspace_configuration_json: str
    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_target_tokens: int
    device: str
    dtype: str
    sidecar_state_sha256: str


@dataclass(frozen=True, slots=True)
class IrisNoWorkspaceTeacherCheckpointProvenanceV1:
    """Arm-labelled ownership and runtime identity for an arm-B sidecar."""

    arm_id: Literal["B"]
    teacher_graph: Literal["independent_recurrent_modules_zero_memory_read_teacher_v1"]
    cognitive_cycles: int
    latent_token_count: int
    component_names: tuple[str, ...]
    backbone_configuration_json: str
    cognitive_configuration_json: str
    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_target_tokens: int
    device: str
    dtype: str
    sidecar_state_sha256: str


@dataclass(frozen=True, slots=True)
class IrisSharedBufferTeacherCheckpointProvenanceV1:
    """Arm-labelled ownership and runtime identity for an arm-C sidecar."""

    arm_id: Literal["C"]
    teacher_graph: Literal["append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"]
    shared_buffer_policy_sha256: str
    cognitive_cycles: int
    pre_update_read_count: int
    final_output_read_count: int
    entries_per_cycle: int
    final_entry_count: int
    latent_token_count: int
    component_names: tuple[str, ...]
    backbone_configuration_json: str
    cognitive_configuration_json: str
    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_target_tokens: int
    device: str
    dtype: str
    sidecar_state_sha256: str
