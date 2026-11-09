"""Distinct in-memory mode-bound execution provenance, not scientific acceptance."""

from dataclasses import dataclass
from re import fullmatch
from typing import Literal

import torch

from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.utils.hashing import canonical_json_bytes
from iris.workspace.trace import WorkspaceTraceV1


@dataclass(frozen=True, slots=True)
class IrisScientificAblationRequestReceiptV1:
    """Actual source/selection/policy and FULL initial/final explicit sampling state.

    Returned only after the owner transaction succeeds. No RNG bytes, mutable
    policy/session, native checkpoint, optimizer or held-out eligibility claim.
    This type is intentionally neither a baseline nor an intervention receipt.
    """

    artifact_sha256: str
    source_checkpoint_sha256: str
    lifecycle_sha256: str
    source_run_configuration_sha256: str
    source_model_configuration_sha256: str
    backbone_runtime_sha256: str
    ablation_policy_sha256: str
    ablation_policy_json: str
    sidecar_state_sha256: str
    selection: Literal["current", "selected_best"]
    completed_steps: int
    purpose: Literal["scientific", "smoke"]
    prompt_sha256: str
    cognitive_prompt_sha256: str
    prompt_input_ids_sha256: str
    prompt_attention_mask_sha256: str
    generated_token_ids_sha256: str
    strategy: DecodingStrategy
    stop_token_ids: tuple[int, ...]
    sampling: TemperatureTopPSamplingConfig
    max_new_tokens: int
    cycles: int
    sampling_seed: int | None
    sampling_initial_state_sha256: str | None
    sampling_final_state_sha256: str | None
    kind: str = "iris-scientific-ablation-request-execution-v1"
    semantics: str = "fresh-session-full-caller-state-transaction-v1"

    def __post_init__(self) -> None:
        for value in (
            self.artifact_sha256,
            self.source_checkpoint_sha256,
            self.lifecycle_sha256,
            self.source_run_configuration_sha256,
            self.source_model_configuration_sha256,
            self.backbone_runtime_sha256,
            self.ablation_policy_sha256,
            self.sidecar_state_sha256,
            self.prompt_sha256,
            self.cognitive_prompt_sha256,
            self.prompt_input_ids_sha256,
            self.prompt_attention_mask_sha256,
            self.generated_token_ids_sha256,
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Mode-bound receipt requires canonical SHA256 identities")
        if (
            type(self.ablation_policy_json) is not str
            or not 1 <= len(self.ablation_policy_json.encode("utf-8")) <= 65536
        ):
            raise ValueError("Bounded immutable cognitive policy JSON required")
        policy = CognitiveAblationPolicyV1.model_validate_json(self.ablation_policy_json)
        if (
            canonical_json_bytes(policy.model_dump(mode="json")).decode("utf-8")
            != self.ablation_policy_json
            or cognitive_ablation_policy_sha256_v1(policy) != self.ablation_policy_sha256
        ):
            raise ValueError("Receipt policy JSON/hash binding differs")
        if (
            type(self.kind) is not str
            or self.kind != "iris-scientific-ablation-request-execution-v1"
            or type(self.semantics) is not str
            or self.semantics != "fresh-session-full-caller-state-transaction-v1"
            or type(self.selection) is not str
            or self.selection not in {"current", "selected_best"}
            or type(self.purpose) is not str
            or self.purpose not in {"scientific", "smoke"}
            or any(
                type(value) is not int or value < 1
                for value in (self.completed_steps, self.max_new_tokens, self.cycles)
            )
        ):
            raise ValueError("Invalid distinct mode-bound receipt semantics/selection/limits")
        if (
            type(self.strategy) is not str
            or self.strategy not in {"greedy", "sampling"}
            or type(self.stop_token_ids) is not tuple
            or any(type(token) is not int or token < 0 for token in self.stop_token_ids)
            or len(set(self.stop_token_ids)) != len(self.stop_token_ids)
            or type(self.sampling) is not TemperatureTopPSamplingConfig
        ):
            raise ValueError("Exact decoding strategy/stop tokens/sampling policy required")
        TemperatureTopPSamplingConfig(self.sampling.temperature, self.sampling.top_p)
        if self.strategy == "greedy":
            if any(
                value is not None
                for value in (
                    self.sampling_seed,
                    self.sampling_initial_state_sha256,
                    self.sampling_final_state_sha256,
                )
            ):
                raise ValueError("Greedy receipt cannot contain sampling RNG provenance")
        elif (
            type(self.sampling_seed) is not int
            or not 0 <= self.sampling_seed <= 2**64 - 1
            or any(
                type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (self.sampling_initial_state_sha256, self.sampling_final_state_sha256)
            )
        ):
            raise ValueError("Sampling receipt requires full initial/final state identities")


@dataclass(frozen=True, slots=True)
class IrisScientificAblationRequestOutputV1:
    generation: IrisGeneratorOutput
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    receipt: IrisScientificAblationRequestReceiptV1
    diagnostic_trace: WorkspaceTraceV1 | None = None

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not IrisScientificAblationRequestReceiptV1
            or type(self.generation) is not IrisGeneratorOutput
            or type(self.prompt_input_ids) is not torch.Tensor
            or type(self.prompt_attention_mask) is not torch.Tensor
        ):
            raise TypeError("Exact distinct mode-bound output facts/receipt required")


__all__ = ["IrisScientificAblationRequestOutputV1", "IrisScientificAblationRequestReceiptV1"]
