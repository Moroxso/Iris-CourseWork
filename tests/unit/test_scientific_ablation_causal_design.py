"""Complete input design/budget contracts; synthetic members, zero model forwards."""

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    CausalDesignPolicy,
    IrisScientificAblationCausalDesignV1,
    ScientificAblationCausalGeometryV1,
    ScientificAblationCausalMemberV1,
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
    build_scientific_ablation_causal_design_v1,
    scientific_ablation_causal_budget_v1,
    scientific_ablation_causal_design_bytes_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_workspace_v1 import ScientificWorkspacePolicyV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

type Protocol = Literal["workspace", "message"]
ROUTING: tuple[Literal["learned", "random", "uniform"], ...] = ("learned", "random", "uniform")
FLAGS: tuple[Literal["enabled", "disabled"], ...] = ("enabled", "disabled")
MODES = tuple(
    CognitiveAblationPolicyV1(
        routing=r, routing_seed=42 if r == "random" else None, goal=g, broadcast=b
    )
    for r in ROUTING
    for g in FLAGS
    for b in FLAGS
)


def members(repeats: int = 3) -> tuple[ScientificAblationCausalMemberV1, ...]:
    return tuple(
        ScientificAblationCausalMemberV1(
            row_index=n,
            example_id=f"test-{n}",
            example_sha256=f"{n:064x}",
            prompt_sha256=f"{n + 5000:064x}",
            cognitive_prompt_sha256=f"{n + 10000:064x}",
            task_type=TASKS[n // (3 * repeats)],
            difficulty=(n // repeats) % 3 + 1,
        )
        for n in range(12 * repeats)
    )


def geometry(**changes: Any) -> ScientificAblationCausalGeometryV1:
    return ScientificAblationCausalGeometryV1.model_validate(
        dict(num_modules=4, num_slots=2, state_dim=8, cycle_count=3, writers_per_cycle=2) | changes
    )


def policy(protocol: Protocol = "message", /, **changes: Any) -> CausalDesignPolicy:
    cls = (
        ScientificAblationMessageCausalPolicyV1
        if protocol == "message"
        else ScientificAblationWorkspaceCausalPolicyV1
    )
    return cls.model_validate(dict(selection_seed=42, shuffle_seed=1337) | changes)


def build(
    protocol: Protocol = "message",
    *,
    inventory: tuple[ScientificAblationCausalMemberV1, ...] | None = None,
    config: CausalDesignPolicy | None = None,
    ablation: CognitiveAblationPolicyV1 | None = None,
    shape: ScientificAblationCausalGeometryV1 | None = None,
    purpose: Literal["smoke", "scientific"] = "smoke",
) -> IrisScientificAblationCausalDesignV1:
    return build_scientific_ablation_causal_design_v1(
        members() if inventory is None else inventory,
        purpose=purpose,
        ablation=MODES[0] if ablation is None else ablation,
        policy=policy(protocol) if config is None else config,
        geometry=geometry() if shape is None else shape,
    )


def selected_ids(design: IrisScientificAblationCausalDesignV1) -> tuple[str, ...]:
    p = design.payload
    return tuple(p.members[index].example_id for index in p.selected_row_indices)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("ablation", MODES)
def test_all12_distinct_mode_designs_reproduce_selection_groups_and_budgets(
    protocol: Protocol, ablation: CognitiveAblationPolicyV1, tmp_path: Path
) -> None:
    result = build(protocol, ablation=ablation)
    p = result.payload
    assert p.ablation == ablation and p.scientific_acceptance == "not_assessed"
    assert p.scope == "input-design-not-source-authentication-or-measurements"
    assert len(p.groups) == 20 and p.groups[0].available_examples == 36
    selected, excluded = set(p.selected_row_indices), {e.row_index for e in p.exclusions}
    assert not selected & excluded and selected | excluded == set(range(36))
    assert selected_ids(result) == selected_ids(build(protocol))
    assert p.donor_pairs == build(protocol).payload.donor_pairs
    content = scientific_ablation_causal_design_bytes_v1(result)
    assert IrisScientificAblationCausalDesignV1.model_validate_json(content) == result
    assert result == build(protocol, inventory=tuple(reversed(members())), ablation=ablation)
    if protocol == "message":
        assert len(selected) == 24 and len(p.donor_pairs) == 12
        assert {e.reason for e in p.exclusions} == {"odd_rank_tail"}
        assert p.budget.total_executions == 240 and p.budget.message_collect_executions == 120
    else:
        assert len(selected) == 36 and not p.donor_pairs and not p.exclusions
        assert p.budget.total_executions == 288 and p.budget.message_collect_executions == 0
    (tmp_path / "design.json").write_bytes(content)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_original_label_digest_changes_design_not_selection_or_pair_assignment(
    protocol: Protocol,
) -> None:
    original = build(protocol)
    changed = build(
        protocol,
        inventory=tuple(m.model_copy(update={"example_sha256": "f" * 64}) for m in members()),
    )
    assert changed.sha256 != original.sha256
    assert selected_ids(changed) == selected_ids(original)
    assert changed.payload.donor_pairs == original.payload.donor_pairs
    assert changed.payload.budget == original.payload.budget


def test_input_rank_and_pair_seeds_are_distinct_and_pairs_are_disjoint_inside_cells() -> None:
    inventory = members(6)
    original = build(
        inventory=inventory, config=policy(examples_per_stratum=6, maximum_examples=72)
    )
    shuffled = build(
        inventory=inventory,
        config=policy(examples_per_stratum=6, maximum_examples=72, shuffle_seed=1338),
    )
    ranked = build(
        inventory=inventory,
        config=policy(examples_per_stratum=4, maximum_examples=72, selection_seed=43),
    )
    assert selected_ids(original) == selected_ids(shuffled)
    assert original.payload.donor_pairs != shuffled.payload.donor_pairs
    assert set(selected_ids(ranked)) < set(selected_ids(original))
    owned: list[int] = []
    for pair in original.payload.donor_pairs:
        a, b = inventory[pair.first_row_index], inventory[pair.second_row_index]
        assert (a.task_type, a.difficulty) == (b.task_type, b.difficulty)
        assert (pair.task_type, pair.difficulty) == (a.task_type, a.difficulty)
        assert a.example_id != b.example_id
        owned.extend((pair.first_row_index, pair.second_row_index))
    assert len(owned) == len(set(owned)) == 72


def test_original_row_reindexing_keeps_input_pair_hashes_and_selected_ids() -> None:
    original = build()
    reordered = tuple(
        m.model_copy(update={"row_index": n}) for n, m in enumerate(reversed(members()))
    )
    changed = build(inventory=reordered)
    assert original.sha256 != changed.sha256
    assert selected_ids(original) == selected_ids(changed)
    assert tuple(p.sha256 for p in original.payload.donor_pairs) == tuple(
        p.sha256 for p in changed.payload.donor_pairs
    )


def test_singletons_odd_tails_caps_and_missing_groups_remain_explicit() -> None:
    # Three in the first cell, one singleton in the next; all other cells empty.
    original = members()[:4]
    result = build(inventory=original, config=policy(examples_per_stratum=2))
    assert len(result.payload.selected_row_indices) == 2
    assert {e.reason for e in result.payload.exclusions} == {"stratum_cap", "singleton_stratum"}
    assert result.payload.groups[0].excluded_examples == 2
    assert any(
        g.available_examples == 0 and g.selected_examples == 0 for g in result.payload.groups
    )
    odd = build(inventory=original, config=policy(examples_per_stratum=4))
    assert {e.reason for e in odd.payload.exclusions} == {"odd_rank_tail", "singleton_stratum"}
    with pytest.raises(ValueError):
        build(inventory=members(1))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize(
    "field", ("row_index", "example_id", "prompt_sha256", "cognitive_prompt_sha256")
)
def test_full_inventory_duplicates_fail_even_for_an_unselected_row(
    protocol: Protocol, field: str
) -> None:
    inventory = members()
    bad = (*inventory[:-1], inventory[-1].model_copy(update={field: getattr(inventory[0], field)}))
    with pytest.raises(ValueError):
        build(protocol, inventory=bad, config=policy(protocol, examples_per_stratum=2))


@pytest.mark.parametrize("value", (-1, 5000, True, 0.0))
def test_undeclared_dense_row_inventory_does_not_repair_invalid_indices(value: Any) -> None:
    with pytest.raises(ValueError):
        build(inventory=(*members()[:-1], members()[-1].model_copy(update={"row_index": value})))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_design_does_not_shrink_coverage_to_fit_total_caps(protocol: Protocol) -> None:
    with pytest.raises(ValueError):
        build(protocol, config=policy(protocol, maximum_examples=2))
    with pytest.raises(ValueError):
        build(protocol, config=policy(protocol, maximum_executions=1))
    with pytest.raises(ValueError):
        build(protocol, config=policy(protocol, maximum_total_numeric_scalars=1))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("ordinary", ("metadata", "full"))
def test_exact_c3_r2_budgets_count_all_transient_traces_and_proposals(
    protocol: Protocol, ordinary: Literal["metadata", "full"]
) -> None:
    b = scientific_ablation_causal_budget_v1(
        policy(protocol, ordinary_trace_mode=ordinary), examples=4, geometry=geometry()
    )
    ordinary_scalars = 279 if ordinary == "metadata" else 567
    controlled = 9 if protocol == "message" else 7
    assert b.ordinary_trace_scalars == ordinary_scalars and b.full_trace_scalars == 567
    assert b.total_executions == 4 * (1 + controlled)
    assert b.total_trace_scalars == 4 * (ordinary_scalars + controlled * 567)
    assert b.message_collect_executions == (20 if protocol == "message" else 0)
    assert b.total_message_scalars == (1920 if protocol == "message" else 0)
    assert b.total_numeric_scalars == b.total_trace_scalars + b.total_message_scalars
    assert b.maximum_effect_rows == 48
    assert b.metadata_trace_executions == (4 if ordinary == "metadata" else 0)
    assert b.full_trace_executions + b.metadata_trace_executions == b.total_executions


@pytest.mark.parametrize(
    "changes",
    (
        {"maximum_executions_per_example": 9},
        {"maximum_executions": 39},
        {"maximum_trace_scalars": 566},
        {"maximum_message_scalars": 95},
        {"maximum_total_numeric_scalars": 23447},
        {"maximum_examples": 2},
    ),
)
def test_each_message_budget_off_by_one_is_denied_before_execution(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        scientific_ablation_causal_budget_v1(policy(**changes), examples=4, geometry=geometry())


def test_exact_budget_edge_is_accepted_without_dropping_transient_proposals() -> None:
    b = scientific_ablation_causal_budget_v1(
        policy(
            maximum_executions_per_example=10,
            maximum_executions=40,
            maximum_trace_scalars=567,
            maximum_message_scalars=96,
            maximum_total_numeric_scalars=23448,
        ),
        examples=4,
        geometry=geometry(),
    )
    assert b.total_numeric_scalars == 23448


@pytest.mark.parametrize("examples", (0, 1, 3, 65, True, 2.0))
def test_exact_even_message_inventory_budget(examples: Any) -> None:
    with pytest.raises(ValueError):
        scientific_ablation_causal_budget_v1(policy(), examples=examples, geometry=geometry())


@pytest.mark.parametrize(
    "changes",
    (
        {"cycle_count": 1},
        {"num_modules": 1},
        {"num_slots": 0},
        {"state_dim": True},
        {"state_dim": 4097},
        {"writers_per_cycle": 3},
    ),
)
def test_no_next_cycle_or_invalid_graph_geometry(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        geometry(**changes)


@pytest.mark.parametrize("changes", ({"examples_per_stratum": 3}, {"maximum_examples": 63}))
def test_both_message_selection_caps_are_even(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("mutation", ("root", "member", "policy", "geometry", "ablation"))
def test_closed_tree_rejects_hidden_attributes_before_serialization(mutation: str) -> None:
    result = build()
    values = {
        "root": result,
        "member": result.payload.members[0],
        "policy": result.payload.policy,
        "geometry": result.payload.geometry,
        "ablation": result.payload.ablation,
    }
    object.__setattr__(values[mutation], "foreign_callback", lambda: None)
    with pytest.raises(ValueError):
        scientific_ablation_causal_design_bytes_v1(result)


def test_exact_models_not_subclasses_lists_raw_roots_or_baseline_policy() -> None:
    class ForeignMember(ScientificAblationCausalMemberV1):
        pass

    foreign = ForeignMember.model_validate(members()[0].model_dump())
    with pytest.raises(ValueError):
        build(inventory=(foreign, *members()[1:]))
    invalid: Any = list(members())
    with pytest.raises(ValueError):
        build(inventory=invalid)
    invalid = build().model_dump()
    with pytest.raises(ValueError):
        scientific_ablation_causal_design_bytes_v1(invalid)
    invalid = ScientificWorkspacePolicyV1(selection_seed=42)
    with pytest.raises(ValueError):
        build(config=invalid)


@pytest.mark.parametrize(
    "field", ("selected_row_indices", "exclusions", "donor_pairs", "groups", "budget")
)
def test_rehashed_forged_design_cannot_validate(field: str) -> None:
    result = build()
    raw = result.model_dump(mode="json")
    p = raw["payload"]
    if field == "selected_row_indices":
        p[field].reverse()
    elif field == "exclusions":
        p[field][0]["reason"] = "stratum_cap"
    elif field == "donor_pairs":
        p[field][0]["second_row_index"] = p[field][0]["first_row_index"]
    elif field == "groups":
        p[field][0]["excluded_examples"] -= 1
    else:
        p[field]["total_message_scalars"] -= 1
    raw["sha256"] = domain_sha256(
        "iris/scientific-ablation-causal-design/v1", canonical_json_bytes(p)
    )
    with pytest.raises(ValueError):
        IrisScientificAblationCausalDesignV1.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_full5000_scientific_design_and_all20_groups_not_actual_source_audit(
    protocol: Protocol, tmp_path: Path
) -> None:
    inventory: list[ScientificAblationCausalMemberV1] = []
    for task, count in zip(TASKS, (2000, 1000, 1000, 1000), strict=True):
        for local in range(count):
            n = len(inventory)
            inventory.append(
                ScientificAblationCausalMemberV1(
                    row_index=n,
                    example_id=f"test-{n}",
                    example_sha256=f"{n:064x}",
                    prompt_sha256=f"{n + 5000:064x}",
                    cognitive_prompt_sha256=f"{n + 10000:064x}",
                    task_type=task,
                    difficulty=local % 3 + 1,
                )
            )
    result = build(protocol, inventory=tuple(inventory), purpose="scientific")
    assert len(result.payload.members) == 5000 and len(result.payload.selected_row_indices) == 48
    assert len(result.payload.exclusions) == 4952 and len(result.payload.groups) == 20
    assert result.payload.groups[0].available_examples == 5000
    assert tuple(g.available_examples for g in result.payload.groups[1:5]) == (
        2000,
        1000,
        1000,
        1000,
    )
    assert all(g.selected_examples == 4 for g in result.payload.groups[8:])
    content = scientific_ablation_causal_design_bytes_v1(result)
    assert len(content) < 4 * 1024**2
    (tmp_path / "design.json").write_bytes(content)
    with pytest.raises(ValueError):
        build(protocol, purpose="scientific")
