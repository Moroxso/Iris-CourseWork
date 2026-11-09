"""Whole causal numeric preflight when EVERY controlled pass is observed.

This extends, rather than relabels or rewrites, the accepted input-only design.
It reserves the same ordinary/empty/shuffle/zero/block/donor passes, plus both
genuine ALL7 snapshots for each controlled pass. It performs no model execution,
source authentication or complete result-publication byte preflight. The counts
are planned trace/proposal/observation volume, NOT peak model memory.
"""

from typing import Literal, Self, TypedDict

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_causal_design_v1 import (
    IrisScientificAblationCausalDesignV1,
    scientific_ablation_causal_design_bytes_v1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import _tree as _design_tree
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.training.ablation_observation import ablation_workspace_observation_scalars_v1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

DOMAIN = "iris/scientific-ablation-causal-observed-budget/v1"
_MAX_METADATA_BYTES = 4 * 1024**2


class _Counts(TypedDict):
    planned_ordinary_teacher_executions: int
    planned_controlled_teacher_executions: int
    planned_total_teacher_executions: int
    planned_observed_teacher_executions: int
    observation_scalar_bound_per_pass: int
    total_observation_scalar_bound: int
    original_trace_proposal_scalar_bound: int
    total_trace_proposal_observation_scalar_bound: int


def _tree(value: object) -> None:
    if (
        type(value) is ScientificAblationCausalObservedBudgetPayloadV1
        or type(value) is IrisScientificAblationCausalObservedBudgetV1
    ):
        if (
            set(vars(value)) != set(type(value).model_fields)
            or value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
        ):
            raise ValueError("Exact closed observed-budget metadata required")
        for child in vars(value).values():
            _tree(child)
    else:
        _design_tree(value)


def _counts(
    design: IrisScientificAblationCausalDesignV1, maximum_observation_scalars: int
) -> _Counts:
    if type(design) is not IrisScientificAblationCausalDesignV1:
        raise ValueError("Exact native-ablation causal design required")
    if (
        type(maximum_observation_scalars) is not int
        or not 1 <= maximum_observation_scalars <= 2_000_000
    ):
        raise ValueError("Exact bounded per-pass observation limit required")
    # Guard constructed/copy-updated/subclass/hidden values before serialization,
    # then own/revalidate the entire design, including its original budget.
    owned = IrisScientificAblationCausalDesignV1.model_validate_json(
        scientific_ablation_causal_design_bytes_v1(design)
    )
    p, b, g = owned.payload, owned.payload.budget, owned.payload.geometry
    observation = ablation_workspace_observation_scalars_v1(
        cycles=g.cycle_count, slots=g.num_slots, width=g.state_dim
    )
    if observation > maximum_observation_scalars:
        raise ValueError("Per-pass ALL7 observation limit exceeded before any forward")
    total_observation = b.controlled_executions * observation
    total = b.total_numeric_scalars + total_observation
    if total > p.policy.maximum_total_numeric_scalars:
        raise ValueError("Whole trace/proposal/ALL7 budget exceeded before any forward")
    return _Counts(
        planned_ordinary_teacher_executions=b.ordinary_executions,
        planned_controlled_teacher_executions=b.controlled_executions,
        planned_total_teacher_executions=b.total_executions,
        planned_observed_teacher_executions=b.controlled_executions,
        observation_scalar_bound_per_pass=observation,
        total_observation_scalar_bound=total_observation,
        original_trace_proposal_scalar_bound=b.total_numeric_scalars,
        total_trace_proposal_observation_scalar_bound=total,
    )


class ScientificAblationCausalObservedBudgetPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-causal-observed-budget-v1"] = (
        "iris-scientific-ablation-causal-observed-budget-v1"
    )
    scope: Literal[
        "all-controlled-passes-observed-numeric-preflight-not-measurements-or-byte-guarantee"
    ] = "all-controlled-passes-observed-numeric-preflight-not-measurements-or-byte-guarantee"
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    design: IrisScientificAblationCausalDesignV1
    maximum_observation_scalars: int = Field(ge=1, le=2_000_000)
    planned_ordinary_teacher_executions: int = Field(ge=1, le=768)
    planned_controlled_teacher_executions: int = Field(ge=1, le=8192)
    planned_total_teacher_executions: int = Field(ge=1, le=8192)
    planned_observed_teacher_executions: int = Field(ge=1, le=8192)
    observation_scalar_bound_per_pass: int = Field(ge=1, le=2_000_000)
    total_observation_scalar_bound: int = Field(ge=1, le=64_000_000)
    original_trace_proposal_scalar_bound: int = Field(ge=1, le=64_000_000)
    total_trace_proposal_observation_scalar_bound: int = Field(ge=1, le=64_000_000)

    @model_validator(mode="after")
    def complete_observed_volume(self) -> Self:
        _tree(self)
        expected = _counts(self.design, self.maximum_observation_scalars)
        if any(getattr(self, name) != count for name, count in expected.items()):
            raise ValueError("Every planned controlled pass must reserve both ALL7 snapshots")
        return self


class IrisScientificAblationCausalObservedBudgetV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCausalObservedBudgetPayloadV1

    @model_validator(mode="after")
    def checksum(self) -> Self:
        _tree(self)
        # A nested model_construct/model_copy must not waive its own validation.
        payload = ScientificAblationCausalObservedBudgetPayloadV1.model_validate(
            self.payload.model_dump(mode="python", warnings=False)
        )
        if self.sha256 != domain_sha256(
            DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
        ):
            raise ValueError("Distinct observed-budget checksum differs")
        return self


def scientific_ablation_causal_observed_budget_bytes_v1(
    value: IrisScientificAblationCausalObservedBudgetV1,
) -> bytes:
    if type(value) is not IrisScientificAblationCausalObservedBudgetV1:
        raise ValueError("Exact distinct observed-budget root required")
    _tree(value)
    owned = IrisScientificAblationCausalObservedBudgetV1.model_validate(
        value.model_dump(mode="python", warnings=False)
    )
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_METADATA_BYTES:
        raise ValueError("Observed-budget metadata exceeds its byte limit")
    return content


def build_scientific_ablation_causal_observed_budget_v1(
    design: IrisScientificAblationCausalDesignV1,
    *,
    maximum_observation_scalars: int,
) -> IrisScientificAblationCausalObservedBudgetV1:
    """Before ANY teacher pass, reject insufficient volume without shrinking input.

    The future source executor must use observed EMPTY controls in place of V1
    controls, observe every subsequent controlled pass, count actual executions,
    and separately preflight the complete result's canonical byte bound. This
    helper neither runs that executor nor claims its results have been measured.
    """
    counts = _counts(design, maximum_observation_scalars)
    owned = IrisScientificAblationCausalDesignV1.model_validate_json(
        scientific_ablation_causal_design_bytes_v1(design)
    )
    payload = ScientificAblationCausalObservedBudgetPayloadV1(
        design=owned, maximum_observation_scalars=maximum_observation_scalars, **counts
    )
    result = IrisScientificAblationCausalObservedBudgetV1(
        sha256=domain_sha256(DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    scientific_ablation_causal_observed_budget_bytes_v1(result)
    return result
