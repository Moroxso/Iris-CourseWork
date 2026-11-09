"""Distinct in-memory provenance for intervened requests; never a baseline receipt."""

from dataclasses import dataclass
from re import fullmatch

import torch

from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.interventions.session import WORKSPACE_INTERVENTION_SEMANTICS_V1
from iris.workspace.trace import WorkspaceTraceV1


@dataclass(frozen=True, slots=True)
class IrisIntervenedRequestExecutionReceiptV1:
    """Bind exact plan/phase semantics, graph, weights and initial sampling state.

    This is not a persisted scientific result or an attestation. A future capture
    format must explicitly accept this type; existing baseline writers must not.
    """

    prompt_sha256: str
    strategy: DecodingStrategy
    stop_token_ids: tuple[int, ...]
    sampling: TemperatureTopPSamplingConfig
    sampling_seed: int | None
    sampling_rng_state_sha256: str | None
    intervention_plan_sha256: str
    graph_sha256: str
    sidecar_state_sha256: str
    cycles: int
    semantics: str = WORKSPACE_INTERVENTION_SEMANTICS_V1
    kind: str = "iris-intervened-request-execution-v1"

    def __post_init__(self) -> None:
        for value in (
            self.prompt_sha256,
            self.intervention_plan_sha256,
            self.graph_sha256,
            self.sidecar_state_sha256,
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Intervened receipt identities must be lowercase SHA256")
        if (
            type(self.kind) is not str
            or self.kind != "iris-intervened-request-execution-v1"
            or type(self.semantics) is not str
            or self.semantics != WORKSPACE_INTERVENTION_SEMANTICS_V1
        ):
            raise ValueError("Unknown intervened request kind or phase semantics")
        if type(self.cycles) is not int or self.cycles < 1:
            raise ValueError("Intervened receipt needs an exact positive cycle count")
        if type(self.strategy) is not str or self.strategy not in {"greedy", "sampling"}:
            raise ValueError("Invalid intervened request strategy")
        if (
            type(self.stop_token_ids) is not tuple
            or any(type(token) is not int or token < 0 for token in self.stop_token_ids)
            or len(set(self.stop_token_ids)) != len(self.stop_token_ids)
        ):
            raise ValueError("Stop-token IDs must be unique exact non-negative integers")
        if type(self.sampling) is not TemperatureTopPSamplingConfig:
            raise TypeError("Intervened receipt needs the exact sampling policy type")
        TemperatureTopPSamplingConfig(
            temperature=self.sampling.temperature, top_p=self.sampling.top_p
        )
        if self.strategy == "greedy":
            if self.sampling_seed is not None or self.sampling_rng_state_sha256 is not None:
                raise ValueError("Greedy intervened receipts cannot contain sampling RNG state")
        elif (
            type(self.sampling_seed) is not int
            or not 0 <= self.sampling_seed <= 2**63 - 1
            or type(self.sampling_rng_state_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", self.sampling_rng_state_sha256) is None
        ):
            raise ValueError("Sampling receipt needs an owned initial RNG state and 63-bit seed")


@dataclass(frozen=True, slots=True)
class IrisIntervenedRequestExecutionOutput:
    """Intervened generation, exact prompt provenance and required distinct receipt."""

    generation: IrisGeneratorOutput
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    receipt: IrisIntervenedRequestExecutionReceiptV1
    diagnostic_trace: WorkspaceTraceV1 | None = None

    def __post_init__(self) -> None:
        if type(self.receipt) is not IrisIntervenedRequestExecutionReceiptV1:
            raise TypeError("Intervened output requires an exact intervention-bound receipt")
