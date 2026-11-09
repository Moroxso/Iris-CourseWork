"""Input-only native-ablation causal design; NOT source authentication or results.

All original members remain visible. Caps never shrink to fit a budget. These
distinct domains cannot relabel baseline workspace/message experiments. The
source-owned executor must independently derive/audit membership and bind this
design to the original best/capture/complete score ledger before any forward.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.schemas import SyntheticTask
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.trace import _scalar_bound

_MAX_BYTES = 4 * 1024**2
_DOMAIN = "iris/scientific-ablation-causal-design/v1"


class ScientificAblationCausalMemberV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    example_id: str = Field(pattern=r"^test-[0-9]+$", max_length=256)
    example_sha256: Digest
    prompt_sha256: Digest
    cognitive_prompt_sha256: Digest
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)


class ScientificAblationCausalGeometryV1(_Strict):
    num_modules: int = Field(ge=2, le=64)
    num_slots: int = Field(ge=1, le=64)
    state_dim: int = Field(ge=1, le=4096)
    cycle_count: int = Field(ge=2, le=64)
    writers_per_cycle: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def actual_allocation(self) -> Self:
        if self.writers_per_cycle > min(self.num_modules, self.num_slots):
            raise ValueError("Declared writer count exceeds the causal graph")
        return self


class _CausalPolicy(_Strict):
    selection_seed: int = Field(ge=0, le=2**63 - 1)
    shuffle_seed: int = Field(ge=0, le=2**63 - 1)
    ordinary_trace_mode: Literal["metadata", "full"] = "metadata"
    maximum_executions_per_example: int = Field(default=64, ge=1, le=8192)
    maximum_executions: int = Field(default=1024, ge=1, le=8192)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_numeric_scalars: int = Field(default=16_000_000, ge=1, le=64_000_000)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    numerical_tolerance: float = Field(default=1e-6, ge=0, lt=1)


class ScientificAblationWorkspaceCausalPolicyV1(_CausalPolicy):
    kind: Literal["iris-ablation-workspace-record-design-policy-v1"] = (
        "iris-ablation-workspace-record-design-policy-v1"
    )
    selection: Literal["input-ranked-task-difficulty-cap-v1"] = (
        "input-ranked-task-difficulty-cap-v1"
    )
    examples_per_stratum: int = Field(default=4, ge=1, le=64)
    maximum_examples: int = Field(default=768, ge=1, le=768)


class ScientificAblationMessageCausalPolicyV1(_CausalPolicy):
    kind: Literal["iris-ablation-paired-message-design-policy-v1"] = (
        "iris-ablation-paired-message-design-policy-v1"
    )
    selection: Literal["input-ranked-largest-even-stratum-prefix-v1"] = (
        "input-ranked-largest-even-stratum-prefix-v1"
    )
    examples_per_stratum: int = Field(default=4, ge=2, le=64)
    maximum_examples: int = Field(default=64, ge=2, le=512)
    maximum_message_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    inferential_unit: Literal["disjoint-donor-pair-not-recipient-or-cycle"] = (
        "disjoint-donor-pair-not-recipient-or-cycle"
    )
    independence: Literal["input-uniqueness-does-not-prove-task-independence"] = (
        "input-uniqueness-does-not-prove-task-independence"
    )

    @model_validator(mode="after")
    def even_cap(self) -> Self:
        if self.examples_per_stratum % 2 or self.maximum_examples % 2:
            raise ValueError("Both message subset limits must be even")
        return self


type CausalDesignPolicy = (
    ScientificAblationWorkspaceCausalPolicyV1 | ScientificAblationMessageCausalPolicyV1
)


class ScientificAblationCausalExclusionV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    reason: Literal["stratum_cap", "odd_rank_tail", "singleton_stratum"]


class ScientificAblationDonorPairV1(_Strict):
    sha256: Digest
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)
    first_row_index: int = Field(ge=0, le=4999)
    second_row_index: int = Field(ge=0, le=4999)


class ScientificAblationCausalGroupV1(_Strict):
    task_type: SyntheticTask | None
    difficulty: int | None = Field(ge=1, le=3)
    available_examples: int = Field(ge=0, le=5000)
    selected_examples: int = Field(ge=0, le=768)
    excluded_examples: int = Field(ge=0, le=5000)


class ScientificAblationCausalBudgetV1(_Strict):
    examples: int = Field(ge=1, le=768)
    ordinary_executions: int = Field(ge=1)
    controlled_executions: int = Field(ge=1)
    total_executions: int = Field(ge=1)
    full_trace_executions: int = Field(ge=1)
    metadata_trace_executions: int = Field(ge=0)
    message_collect_executions: int = Field(ge=0)
    ordinary_trace_scalars: int = Field(ge=1)
    full_trace_scalars: int = Field(ge=1)
    message_scalars_per_collection: int = Field(ge=0)
    total_trace_scalars: int = Field(ge=1)
    total_message_scalars: int = Field(ge=0)
    total_numeric_scalars: int = Field(ge=1)
    maximum_effect_rows: int = Field(ge=1)


def _tree(value: object, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("Causal design nesting exceeds its closed contract")
    if isinstance(value, BaseModel):
        allowed = (
            CognitiveAblationPolicyV1,
            ScientificAblationCausalMemberV1,
            ScientificAblationCausalGeometryV1,
            ScientificAblationWorkspaceCausalPolicyV1,
            ScientificAblationMessageCausalPolicyV1,
            ScientificAblationCausalExclusionV1,
            ScientificAblationDonorPairV1,
            ScientificAblationCausalGroupV1,
            ScientificAblationCausalBudgetV1,
            ScientificAblationCausalDesignPayloadV1,
            IrisScientificAblationCausalDesignV1,
        )
        if (
            type(value) not in allowed
            or set(vars(value)) != set(type(value).model_fields)
            or value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
        ):
            raise ValueError("Exact closed causal design models required")
        for item in vars(value).values():
            _tree(item, depth + 1)
    elif type(value) is tuple:
        if len(value) > 5000:
            raise ValueError("Causal design tuple exceeds original test inventory")
        for item in value:
            _tree(item, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("Mutable or foreign values are not causal design metadata")


def _own[T: BaseModel](value: T, expected: type[T]) -> T:
    if type(value) is not expected:
        raise ValueError("Exact declared causal design model required")
    _tree(value)
    return expected.model_validate(value.model_dump(mode="python", warnings=False))


def scientific_ablation_causal_budget_v1(
    policy: CausalDesignPolicy, *, examples: int, geometry: ScientificAblationCausalGeometryV1
) -> ScientificAblationCausalBudgetV1:
    """Count every ordinary/control/shuffle/block/donor pass BEFORE any forward.

    Numeric trace/proposal volume is not a file-byte or peak-allocation guarantee.
    The final source-owned service must separately preflight publication bytes.
    """
    if type(policy) not in (
        ScientificAblationWorkspaceCausalPolicyV1,
        ScientificAblationMessageCausalPolicyV1,
    ):
        raise ValueError("Exact distinct ablation causal policy required")
    policy = _own(policy, type(policy))
    geometry = _own(geometry, ScientificAblationCausalGeometryV1)
    if type(examples) is not int or not 1 <= examples <= policy.maximum_examples:
        raise ValueError("Selected inventory exceeds the declared limit")
    m, k, d, c, r = (
        geometry.num_modules,
        geometry.num_slots,
        geometry.state_dim,
        geometry.cycle_count,
        geometry.writers_per_cycle,
    )
    ordinary = _scalar_bound(m, k, d, c, policy.ordinary_trace_mode == "full")
    full = _scalar_bound(m, k, d, c, True)
    if isinstance(policy, ScientificAblationMessageCausalPolicyV1):
        if examples < 2 or examples % 2:
            raise ValueError("Disjoint donor design needs an even selected inventory")
        controlled = 1 + 2 * (c - 1) * r
        collections = 1 + (c - 1) * r
        messages = c * m * d
        if messages > policy.maximum_message_scalars:
            raise ValueError("Per-pass proposal scalar budget exceeded")
    else:
        # Empty combined control, then one record shuffle and R zero-slot passes.
        controlled = 1 + (c - 1) * (r + 1)
        collections = messages = 0
    calls = 1 + controlled
    trace_total = examples * (ordinary + controlled * full)
    message_total = examples * collections * messages
    total = trace_total + message_total
    if (
        max(ordinary, full) > policy.maximum_trace_scalars
        or calls > policy.maximum_executions_per_example
        or examples * calls > policy.maximum_executions
        or total > policy.maximum_total_numeric_scalars
    ):
        raise ValueError("Whole causal design exceeds declared transient budgets")
    full_calls = examples * (controlled + (policy.ordinary_trace_mode == "full"))
    return ScientificAblationCausalBudgetV1(
        examples=examples,
        ordinary_executions=examples,
        controlled_executions=examples * controlled,
        total_executions=examples * calls,
        full_trace_executions=full_calls,
        metadata_trace_executions=examples * calls - full_calls,
        message_collect_executions=examples * collections,
        ordinary_trace_scalars=ordinary,
        full_trace_scalars=full,
        message_scalars_per_collection=messages,
        total_trace_scalars=trace_total,
        total_message_scalars=message_total,
        total_numeric_scalars=total,
        maximum_effect_rows=examples * (c - 1) * r * (m - 1),
    )


def _input(member: ScientificAblationCausalMemberV1) -> dict[str, str | int]:
    return {
        "example_id": member.example_id,
        "prompt_sha256": member.prompt_sha256,
        "task_type": member.task_type,
        "difficulty": member.difficulty,
    }


def _derive(
    members: tuple[ScientificAblationCausalMemberV1, ...], policy: CausalDesignPolicy
) -> tuple[
    tuple[int, ...],
    tuple[ScientificAblationCausalExclusionV1, ...],
    tuple[ScientificAblationDonorPairV1, ...],
    tuple[ScientificAblationCausalGroupV1, ...],
]:
    if not 1 <= len(members) <= 5000 or tuple(m.row_index for m in members) != tuple(
        range(len(members))
    ):
        raise ValueError("Complete dense original test row inventory required")
    if any(
        len({getattr(m, field) for m in members}) != len(members)
        for field in ("example_id", "prompt_sha256", "cognitive_prompt_sha256")
    ):
        raise ValueError("Every original member must have unique input identity")
    message = isinstance(policy, ScientificAblationMessageCausalPolicyV1)
    protocol = "paired-message" if message else "workspace-record"
    selected: list[int] = []
    exclusions: list[ScientificAblationCausalExclusionV1] = []
    pairs: list[ScientificAblationDonorPairV1] = []
    for task in TASKS:
        for difficulty in (1, 2, 3):
            ranked = sorted(
                (m for m in members if (m.task_type, m.difficulty) == (task, difficulty)),
                key=lambda m: (
                    domain_sha256(
                        f"iris/scientific-ablation-{protocol}-selection/v1",
                        canonical_json_bytes({"seed": policy.selection_seed, "input": _input(m)}),
                    ),
                    m.example_id,
                ),
            )
            capped = min(len(ranked), policy.examples_per_stratum)
            count = capped - capped % 2 if message else capped
            chosen = ranked[:count]
            selected.extend(m.row_index for m in chosen)
            for rank, member in enumerate(ranked[count:], count):
                reason: Literal["stratum_cap", "odd_rank_tail", "singleton_stratum"] = (
                    "singleton_stratum"
                    if message and len(ranked) == 1
                    else "odd_rank_tail"
                    if message and rank < capped
                    else "stratum_cap"
                )
                exclusions.append(
                    ScientificAblationCausalExclusionV1(row_index=member.row_index, reason=reason)
                )
            if message:
                shuffled = sorted(
                    chosen,
                    key=lambda m: (
                        domain_sha256(
                            "iris/scientific-ablation-message-donor-rank/v1",
                            canonical_json_bytes({"seed": policy.shuffle_seed, "input": _input(m)}),
                        ),
                        m.example_id,
                    ),
                )
                for left, right in zip(shuffled[::2], shuffled[1::2], strict=True):
                    pairs.append(
                        ScientificAblationDonorPairV1(
                            sha256=domain_sha256(
                                "iris/scientific-ablation-message-donor-pair/v1",
                                canonical_json_bytes(
                                    {
                                        "seed": policy.shuffle_seed,
                                        "first": _input(left),
                                        "second": _input(right),
                                    }
                                ),
                            ),
                            task_type=task,
                            difficulty=difficulty,
                            first_row_index=left.row_index,
                            second_row_index=right.row_index,
                        )
                    )
    if not selected or len(selected) > policy.maximum_examples:
        raise ValueError("Declared subset is empty or exceeds its cap; no silent shrinking")
    selectors: tuple[tuple[SyntheticTask | None, int | None], ...] = (
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    )
    groups = []
    selected_set = set(selected)
    for group_task, group_difficulty in selectors:
        available = tuple(
            m.row_index
            for m in members
            if (group_task is None or m.task_type == group_task)
            and (group_difficulty is None or m.difficulty == group_difficulty)
        )
        observed = sum(index in selected_set for index in available)
        groups.append(
            ScientificAblationCausalGroupV1(
                task_type=group_task,
                difficulty=group_difficulty,
                available_examples=len(available),
                selected_examples=observed,
                excluded_examples=len(available) - observed,
            )
        )
    return tuple(selected), tuple(exclusions), tuple(pairs), tuple(groups)


class ScientificAblationCausalDesignPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-causal-design-v1"] = (
        "iris-scientific-ablation-causal-design-v1"
    )
    scope: Literal["input-design-not-source-authentication-or-measurements"] = (
        "input-design-not-source-authentication-or-measurements"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    purpose: Literal["smoke", "scientific"]
    ablation: CognitiveAblationPolicyV1
    ablation_policy_sha256: Digest
    policy: Annotated[CausalDesignPolicy, Field(discriminator="kind")]
    policy_sha256: Digest
    geometry: ScientificAblationCausalGeometryV1
    members: tuple[ScientificAblationCausalMemberV1, ...] = Field(min_length=1, max_length=5000)
    selected_row_indices: tuple[int, ...] = Field(min_length=1, max_length=768)
    exclusions: tuple[ScientificAblationCausalExclusionV1, ...] = Field(max_length=5000)
    donor_pairs: tuple[ScientificAblationDonorPairV1, ...] = Field(max_length=256)
    groups: tuple[ScientificAblationCausalGroupV1, ...] = Field(min_length=20, max_length=20)
    budget: ScientificAblationCausalBudgetV1

    @model_validator(mode="after")
    def reproduce_design(self) -> Self:
        if self.purpose == "scientific" and tuple(
            sum(m.task_type == task for m in self.members) for task in TASKS
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Declared scientific design requires all5000 original members")
        if (
            self.ablation_policy_sha256 != cognitive_ablation_policy_sha256_v1(self.ablation)
            or self.policy_sha256
            != domain_sha256(
                "iris/scientific-ablation-causal-design-policy/v1",
                canonical_json_bytes(self.policy.model_dump(mode="json")),
            )
            or (self.selected_row_indices, self.exclusions, self.donor_pairs, self.groups)
            != _derive(self.members, self.policy)
            or self.budget
            != scientific_ablation_causal_budget_v1(
                self.policy, examples=len(self.selected_row_indices), geometry=self.geometry
            )
        ):
            raise ValueError("Declared mode/input/pair/group/budget design does not reproduce")
        return self


class IrisScientificAblationCausalDesignV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCausalDesignPayloadV1

    @model_validator(mode="after")
    def checksum(self) -> Self:
        if self.sha256 != domain_sha256(
            _DOMAIN, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Distinct ablation causal design checksum differs")
        return self


def scientific_ablation_causal_design_bytes_v1(
    design: IrisScientificAblationCausalDesignV1,
) -> bytes:
    """Own and bound exact closed metadata; not native source evidence."""
    owned = _own(design, IrisScientificAblationCausalDesignV1)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Causal design byte bound exceeded")
    return content


def build_scientific_ablation_causal_design_v1(
    members: tuple[ScientificAblationCausalMemberV1, ...],
    *,
    purpose: Literal["smoke", "scientific"],
    ablation: CognitiveAblationPolicyV1,
    policy: CausalDesignPolicy,
    geometry: ScientificAblationCausalGeometryV1,
) -> IrisScientificAblationCausalDesignV1:
    """Deterministic input-only design; source executor must own full audit later."""
    if type(members) is not tuple or not 1 <= len(members) <= 5000:
        raise ValueError("Exact bounded complete member tuple required")
    if type(policy) not in (
        ScientificAblationWorkspaceCausalPolicyV1,
        ScientificAblationMessageCausalPolicyV1,
    ):
        raise ValueError("Distinct exact native-ablation causal policy required")
    checked = tuple(
        sorted(
            (_own(m, ScientificAblationCausalMemberV1) for m in members),
            key=lambda m: m.row_index,
        )
    )
    ablation = _own(ablation, CognitiveAblationPolicyV1)
    policy = _own(policy, type(policy))
    geometry = _own(geometry, ScientificAblationCausalGeometryV1)
    selected, exclusions, pairs, groups = _derive(checked, policy)
    payload = ScientificAblationCausalDesignPayloadV1(
        purpose=purpose,
        ablation=ablation,
        ablation_policy_sha256=cognitive_ablation_policy_sha256_v1(ablation),
        policy=policy,
        policy_sha256=domain_sha256(
            "iris/scientific-ablation-causal-design-policy/v1",
            canonical_json_bytes(policy.model_dump(mode="json")),
        ),
        geometry=geometry,
        members=checked,
        selected_row_indices=selected,
        exclusions=exclusions,
        donor_pairs=pairs,
        groups=groups,
        budget=scientific_ablation_causal_budget_v1(
            policy, examples=len(selected), geometry=geometry
        ),
    )
    result = IrisScientificAblationCausalDesignV1(
        sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    scientific_ablation_causal_design_bytes_v1(result)
    return result
