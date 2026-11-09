"""Disjoint donor pairing, pair-clustered statistics and complete measurement design."""

from typing import Any

import pytest

from iris.workspace.message_causal import (
    MessageCausalEffectV1,
    MessageCausalSubjectV1,
    WorkspaceMessageCausalCaptureV1,
    WorkspaceMessageCausalPolicyV1,
    message_causal_budget_v1,
    message_causal_statistics_v1,
    pair_message_subjects_v1,
)


def subjects(count: int = 16) -> tuple[MessageCausalSubjectV1, ...]:
    return tuple(
        MessageCausalSubjectV1(
            example_id=f"example-{n:03d}", prompt_sha256=f"{n:064x}", stratum="task-1"
        )
        for n in range(count)
    )


def capture(
    count: int = 16, *, genuine: float = 0.4, donor: float = 0.1, cycles: int = 2
) -> WorkspaceMessageCausalCaptureV1:
    policy = WorkspaceMessageCausalPolicyV1(shuffle_seed=42)
    members = subjects(count)
    pairs = pair_message_subjects_v1(members, policy)
    mapped = {}
    for pair in pairs:
        mapped[pair.first_example_id] = (pair.second_example_id, pair.sha256)
        mapped[pair.second_example_id] = (pair.first_example_id, pair.sha256)
    effects = tuple(
        MessageCausalEffectV1(
            example_id=s.example_id,
            donor_example_id=mapped[s.example_id][0],
            donor_pair_sha256=mapped[s.example_id][1],
            write_cycle=cycle,
            source_module=0,
            slot=0,
            target_module=1,
            overwrote_existing_record=cycle > 1,
            survives_to_next_read=True,
            genuine_write_distance=genuine,
            donor_write_distance=donor,
            recipient_message_sha256="a" * 64,
            donor_message_sha256="b" * 64,
            block_plan_sha256="c" * 64,
            transfer_plan_sha256="d" * 64,
        )
        for s in members
        for cycle in range(1, cycles)
    )
    calls, volume = message_causal_budget_v1(
        policy, examples=count, modules=2, slots=1, width=1, cycles=cycles, writers=1
    )
    return WorkspaceMessageCausalCaptureV1(
        policy=policy,
        teacher_graph_sha256="e" * 64,
        sidecar_state_sha256="f" * 64,
        seed=1337,
        num_modules=2,
        num_slots=1,
        state_dim=1,
        cycle_count=cycles,
        writers_per_cycle=1,
        executions=calls,
        trace_scalar_bound=volume,
        subjects=members,
        donor_pairs=pairs,
        effects=effects,
        statistics=message_causal_statistics_v1(pairs, effects, modules=2, policy=policy),
    )


def test_pairing_is_bijective_stratified_order_independent_and_input_only() -> None:
    members = subjects()
    policy = WorkspaceMessageCausalPolicyV1(shuffle_seed=42)
    pairs = pair_message_subjects_v1(members, policy)
    assert len(pairs) == 8
    assert pair_message_subjects_v1(tuple(reversed(members)), policy) == pairs
    assert {id for pair in pairs for id in (pair.first_example_id, pair.second_example_id)} == {
        s.example_id for s in members
    }
    assert all(pair.first_example_id != pair.second_example_id for pair in pairs)
    assert (
        pair_message_subjects_v1(members, policy.model_copy(update={"shuffle_seed": 43})) != pairs
    )
    stratified = tuple(
        s.model_copy(update={"stratum": "task-2" if n >= 8 else "task-1"})
        for n, s in enumerate(members)
    )
    owned = {s.example_id: s.stratum for s in stratified}
    assert all(
        owned[p.first_example_id] == owned[p.second_example_id] == p.stratum
        for p in pair_message_subjects_v1(stratified, policy)
    )


@pytest.mark.parametrize(
    "bad",
    [
        subjects(1),
        subjects(3),
        (subjects(2)[0], subjects(2)[0]),
        (subjects(2)[0], subjects(2)[1].model_copy(update={"prompt_sha256": "0" * 64})),
        tuple(s.model_copy(update={"stratum": f"stratum-{n}"}) for n, s in enumerate(subjects(2))),
    ],
)
def test_odd_singleton_or_repeated_donor_units_reject(
    bad: tuple[MessageCausalSubjectV1, ...],
) -> None:
    with pytest.raises(ValueError):
        pair_message_subjects_v1(bad, WorkspaceMessageCausalPolicyV1(shuffle_seed=42))


def test_statistics_use_eight_disjoint_pairs_not_sixteen_recipients() -> None:
    result = capture()
    observed, missing = result.statistics
    assert (
        observed.independent_donor_pairs,
        observed.recipient_examples,
        observed.write_trials,
    ) == (8, 16, 16)
    assert observed.positive_pairs == 8
    assert observed.sign_test_p == 1 / 256
    assert observed.holm_adjusted_p == 1 / 128
    assert observed.significant_positive_effect
    assert missing.independent_donor_pairs == 0
    assert missing.sign_test_p == missing.holm_adjusted_p == 1
    assert missing.mean_excess is None
    assert result.scientific_acceptance == "not_assessed"


def test_cycle_repetitions_do_not_double_statistical_sample_size() -> None:
    result = capture(cycles=3)
    pair = result.statistics[0]
    assert pair.write_trials == 32 and pair.independent_donor_pairs == 8
    assert pair.overwritten_trials == 16
    assert pair.sign_test_p == 1 / 256


def test_opposite_effects_inside_one_pair_average_before_testing() -> None:
    result = capture(count=2)
    effects = tuple(
        e.model_copy(
            update={
                "genuine_write_distance": 0.6
                if e.example_id == result.donor_pairs[0].first_example_id
                else 0.2,
                "donor_write_distance": 0.2
                if e.example_id == result.donor_pairs[0].first_example_id
                else 0.6,
            }
        )
        for e in result.effects
    )
    records = message_causal_statistics_v1(
        result.donor_pairs, effects, modules=2, policy=result.policy
    )
    assert (
        records[0].tied_pairs == 1 and records[0].positive_pairs == records[0].negative_pairs == 0
    )
    assert records[0].sign_test_p == 1


@pytest.mark.parametrize("genuine,donor", [(0.1, 0.4), (0.4, 0.4), (0.4000001, 0.4)])
def test_negative_equal_and_numerical_tie_effects_are_not_significant(
    genuine: float, donor: float
) -> None:
    observed = capture(genuine=genuine, donor=donor).statistics[0]
    assert observed.sign_test_p == 1 and not observed.significant_positive_effect


@pytest.mark.parametrize(
    "field", ["executions", "trace_scalar_bound", "donor_pairs", "statistics", "effects"]
)
def test_complete_capture_rejects_rehashed_or_forged_design(field: str) -> None:
    result = capture(count=2)
    raw = result.model_dump()
    if field in ("executions", "trace_scalar_bound"):
        raw[field] += 1
    elif field == "donor_pairs":
        raw[field][0]["sha256"] = "0" * 64
    elif field == "statistics":
        raw[field][0]["sign_test_p"] = 0.0
    else:
        raw[field] = ()
    with pytest.raises(ValueError):
        WorkspaceMessageCausalCaptureV1.model_validate(raw)


@pytest.mark.parametrize(
    "updates",
    [
        {"donor_example_id": "missing"},
        {"target_module": 0},
        {"slot": 1},
        {"write_cycle": 2},
        {"donor_pair_sha256": "0" * 64},
    ],
)
def test_trial_domain_must_match_actual_pair_and_next_cycle(updates: dict[str, Any]) -> None:
    result = capture(count=2)
    raw = result.model_dump()
    raw["effects"][0].update(updates)
    with pytest.raises(ValueError):
        WorkspaceMessageCausalCaptureV1.model_validate(raw)


@pytest.mark.parametrize(
    "updates",
    [
        {"maximum_executions": 47},
        {"maximum_trace_scalars": 1},
        {"maximum_message_scalars": 1},
        {"maximum_total_trace_scalars": 1},
        {"maximum_examples": 2},
    ],
)
def test_budgets_preflight_the_complete_design(updates: dict[str, Any]) -> None:
    policy = WorkspaceMessageCausalPolicyV1.model_validate({"shuffle_seed": 42, **updates})
    with pytest.raises(ValueError):
        message_causal_budget_v1(
            policy, examples=16, modules=2, slots=1, width=1, cycles=2, writers=1
        )
