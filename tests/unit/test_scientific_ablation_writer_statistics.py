"""Synthetic exact statistics/coverage contracts, NEVER native measurements."""

import json
from collections.abc import Callable
from itertools import product
from math import fsum, isnan
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.artifacts import scientific_ablation_writer_statistics_v1 as module
from iris.artifacts.scientific_ablation_causal_design_v1 import IrisScientificAblationCausalDesignV1
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_file_bytes
from iris.workspace.causal import WorkspaceCausalEffectV1
from tests.unit.test_scientific_ablation_causal_design import (
    MODES,
    Protocol,
    build,
    geometry,
    members,
    policy,
)

type Schedule = Callable[[int, int, int], int]
type Values = Callable[[int, int, int, int], tuple[float, float]]


def digest(value: int) -> str:
    return f"{value:064x}"


def effects(
    design: IrisScientificAblationCausalDesignV1,
    *,
    schedule: Schedule | None = None,
    values: Values | None = None,
) -> tuple[module.WriterEffect, ...]:
    p, g = design.payload, design.payload.geometry
    donor = {
        row: (other, pair.sha256)
        for pair in p.donor_pairs
        for row, other in (
            (pair.first_row_index, pair.second_row_index),
            (pair.second_row_index, pair.first_row_index),
        )
    }
    rows: list[module.WriterEffect] = []
    for row in p.selected_row_indices:
        for cycle in range(1, g.cycle_count):
            for slot in range(g.writers_per_cycle):
                source = (
                    (row + cycle + slot) % g.num_modules
                    if schedule is None
                    else schedule(row, cycle, slot)
                )
                for target in range(g.num_modules):
                    if target == source:
                        continue
                    a, b = (0.75, 0.25) if values is None else values(row, cycle, source, target)
                    kwargs: dict[str, Any] = dict(
                        row_index=row,
                        write_cycle=cycle,
                        source_module=source,
                        target_module=target,
                        slot=slot,
                        survives_to_next_read=(row + cycle) % 3 != 0,
                        overwrote_existing_record=(row + cycle) % 2 == 0,
                        genuine_distance=a,
                        negative_control_distance=b,
                        empty_invocation_sha256=f"{row + 100:064x}",
                        intervention_invocation_sha256=digest(
                            10000 + row * 1000 + cycle * 100 + source
                        ),
                        negative_control_invocation_sha256=digest(20000 + row * 1000 + cycle * 100),
                    )
                    if donor:
                        other, sha = donor[row]
                        kwargs.update(
                            donor_row_index=other,
                            donor_pair_sha256=sha,
                            recipient_message_sha256=digest(
                                40000 + row * 1000 + cycle * 100 + source
                            ),
                            donor_message_sha256=digest(
                                40000 + other * 1000 + cycle * 100 + source
                            ),
                            negative_control_invocation_sha256=digest(
                                30000 + row * 1000 + cycle * 100 + source
                            ),
                        )
                        rows.append(module.ScientificAblationMessageWriterEffectV1(**kwargs))
                    else:
                        rows.append(module.ScientificAblationWorkspaceWriterEffectV1(**kwargs))
    return tuple(rows)


def pair(result: module.WriterStatistics, source: int = 0, target: int = 1) -> Any:
    return next(p for p in result.pairs if (p.source_module, p.target_module) == (source, target))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("ablation", MODES)
def test_all12_synthetic_complete_stat_domains_no_teacher_passes(
    protocol: Protocol,
    ablation: Any,
    tmp_path: Path,
) -> None:
    design = build(protocol, ablation=ablation)
    rows = effects(design)
    with patch.object(IrisTeacherForcingModel, "_execute") as execute:
        result = module.build_scientific_ablation_writer_statistics_v1(design, rows)
        content = module.scientific_ablation_writer_statistics_bytes_v1(result)
        execute.assert_not_called()
    assert result.kind == f"iris-scientific-ablation-{protocol}-writer-statistics-v1"
    assert result.alpha == design.payload.policy.alpha
    assert result.numerical_tolerance == design.payload.policy.numerical_tolerance
    assert len(result.pairs) == 12
    assert sum(p.write_trials for p in result.pairs) == len(rows)
    assert sum(p.overwritten_trials for p in result.pairs) == sum(
        e.overwrote_existing_record for e in rows
    )
    assert sum(p.expired_trials for p in result.pairs) == sum(
        not e.survives_to_next_read for e in rows
    )
    assert (
        module.build_scientific_ablation_writer_statistics_v1(design, tuple(reversed(rows)))
        == result
    )
    witness = dict(
        kind="iris-tests-synthetic-native-ablation-writer-statistics-v1",
        native_measurements=False,
        teacher_executions=0,
        scientific_acceptance="not_assessed",
        design=design.model_dump(mode="json"),
        effects=[e.model_dump(mode="json") for e in rows],
        statistics=json.loads(content),
    )
    (tmp_path / "statistics-specimen.json").write_bytes(canonical_json_file_bytes(witness))
    (tmp_path / "design.json").write_bytes(
        canonical_json_file_bytes(design.model_dump(mode="json"))
    )


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_cycles_and_available_recipients_are_not_independent_units(protocol: Protocol) -> None:
    design = build(
        protocol,
        inventory=members()[:2],
        config=policy(protocol, examples_per_stratum=2),
        shape=geometry(num_modules=3, num_slots=1, writers_per_cycle=1),
    )
    first = (
        design.payload.donor_pairs[0].first_row_index
        if protocol == "message"
        else design.payload.selected_row_indices[0]
    )

    def schedule(row: int, cycle: int, slot: int) -> int:
        return 0 if row == first or cycle == 1 else 2

    def values(row: int, cycle: int, source: int, target: int) -> tuple[float, float]:
        return (1.0 if row == first and cycle == 1 else 0.0, 0.4)

    result = module.build_scientific_ablation_writer_statistics_v1(
        design, effects(design, schedule=schedule, values=values)
    )
    p = pair(result)
    assert p.write_trials == 3 and p.recipient_examples == 2
    assert p.mean_genuine_distance == 0.25 and p.mean_negative_control_distance == 0.4
    assert p.mean_excess == pytest.approx(-0.15)
    assert p.mean_genuine_distance != 1 / 3
    if protocol == "message":
        assert p.independent_units == 1 and p.negative_units == 1 and p.sign_test_p == 1.0
    else:
        assert p.independent_units == 2 and p.positive_units == p.negative_units == 1
        assert p.sign_test_p == 0.75


def test_one_recipient_donor_not_winner_still_contributes_pair() -> None:
    design = build(
        "message",
        inventory=members()[:2],
        shape=geometry(num_modules=3, num_slots=1, writers_per_cycle=1),
    )
    first = design.payload.donor_pairs[0].first_row_index
    result = module.build_scientific_ablation_writer_statistics_v1(
        design, effects(design, schedule=lambda row, cycle, slot: 0 if row == first else 2)
    )
    p = pair(result)
    assert p.independent_units == p.recipient_examples == 1 and p.write_trials == 2
    assert p.mean_genuine_distance == 0.75 and p.mean_negative_control_distance == 0.25
    assert p.positive_units == 1 and p.sign_test_p == 0.5


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_tolerance_ties_stay_in_descriptive_means_and_missing_pairs_in_holm(
    protocol: Protocol,
) -> None:
    design = build(protocol, config=policy(protocol, numerical_tolerance=0.125))
    result = module.build_scientific_ablation_writer_statistics_v1(
        design,
        effects(
            design,
            schedule=lambda row, cycle, slot: slot,
            values=lambda row, cycle, source, target: (0.5, 0.375),
        ),
    )
    observed, missing = pair(result), pair(result, 2, 0)
    assert observed.tied_units == observed.independent_units > 0
    assert observed.mean_excess == 0.125 and observed.sign_test_p == 1.0
    assert missing.independent_units == missing.write_trials == 0
    assert (
        missing.mean_genuine_distance
        is missing.mean_negative_control_distance
        is missing.mean_excess
        is None
    )
    assert missing.sign_test_p == missing.holm_adjusted_p == 1.0
    assert not any(p.significant_positive_effect for p in result.pairs)


@pytest.mark.parametrize("positive", range(7))
@pytest.mark.parametrize("negative", range(7))
def test_exact_sign_tail_matches_exhaustive_coin_sequences(positive: int, negative: int) -> None:
    n = positive + negative
    successes = sum(sum(bits) >= positive for bits in product((0, 1), repeat=n))
    assert module._sign_tail(positive, negative) == successes / 2**n


def test_large_exact_sign_and_full_holm_missing_family() -> None:
    assert module._sign_tail(768, 0) == 2.0**-768
    assert module._sign_tail(0, 768) == 1.0
    assert module._holm((0.01, 0.03, 0.03, 1.0, 1.0, 1.0)) == (0.06, 0.15, 0.15, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize(
    "kind",
    (
        "subset",
        "duplicate",
        "self",
        "geometry",
        "slot",
        "unselected",
        "bool",
        "int_distance",
        "nan",
        "hidden",
        "private",
        "background",
        "flag",
        "donor",
        "wrong_kind",
        "foreign_tuple",
    ),
)
def test_closed_complete_inventory_rejects_bypassed_or_changed_effects(
    protocol: Protocol, kind: str
) -> None:
    design = build(protocol)
    rows = list(effects(design))
    e = rows[0]
    changes: dict[str, Any] = {}
    if kind == "subset":
        rows.pop()
    elif kind == "duplicate":
        rows[1] = e
    elif kind == "self":
        changes["target_module"] = e.source_module
    elif kind == "geometry":
        changes["write_cycle"] = design.payload.geometry.cycle_count
    elif kind == "slot":
        changes["slot"] = design.payload.geometry.num_slots
    elif kind == "unselected":
        changes["row_index"] = 4999
    elif kind == "bool":
        changes["write_cycle"] = True
    elif kind == "int_distance":
        changes["genuine_distance"] = 1
    elif kind == "nan":
        changes["genuine_distance"] = float("nan")
    elif kind == "hidden":
        changes["unused"] = 1
    elif kind == "private":
        object.__setattr__(e, "__pydantic_private__", {"hidden": 1})
    elif kind == "background":
        changes["empty_invocation_sha256"] = "f" * 64
    elif kind == "flag":
        changes["survives_to_next_read"] = not e.survives_to_next_read
    elif kind == "donor":
        changes["donor_row_index"] = e.row_index
    elif kind == "wrong_kind":
        changes["kind"] = "iris-workspace-causal-effect-v1"
    if changes:
        rows[0] = e.model_copy(update=changes)
    actual: Any = tuple(rows)
    if kind == "foreign_tuple":

        class Foreign(tuple[module.WriterEffect, ...]):
            def __iter__(self) -> Any:
                pytest.fail("Foreign tuple callback must not execute")

        actual = Foreign(rows)
    with pytest.raises((ValueError, TypeError)):
        module.build_scientific_ablation_writer_statistics_v1(design, actual)


def test_foreign_effect_and_legacy_neutral_effect_never_relabel_or_execute_hooks() -> None:
    design = build("workspace")
    rows = effects(design)

    class Foreign(module.ScientificAblationWorkspaceWriterEffectV1):
        def __getattribute__(self, name: str) -> Any:
            pytest.fail("Foreign native effect callback must not execute")

    foreign = Foreign.model_construct(**rows[0].model_dump())
    neutral = WorkspaceCausalEffectV1(
        write_cycle=1,
        source_module=0,
        slot=0,
        target_module=1,
        survives_to_next_read=True,
        overwrote_existing_record=False,
        ablated_distance=0.5,
        shuffled_distance=0.1,
        ablation_plan_sha256="a" * 64,
        shuffle_plan_sha256="b" * 64,
    )
    untrusted_values: tuple[Any, ...] = (foreign, neutral)
    for value in untrusted_values:
        with pytest.raises(ValueError):
            module.build_scientific_ablation_writer_statistics_v1(design, (value, *rows[1:]))


@pytest.mark.parametrize("kind", ("p", "holm", "missing", "ties", "hidden", "alpha", "foreign"))
def test_stats_revalidate_complete_family_and_exact_tails_on_encoding(kind: str) -> None:
    design = build("workspace")
    result = module.build_scientific_ablation_writer_statistics_v1(design, effects(design))
    pairs = list(result.pairs)
    if kind == "p":
        pairs[0] = pairs[0].model_copy(update={"sign_test_p": 0.0})
    elif kind == "holm":
        pairs[0] = pairs[0].model_copy(update={"holm_adjusted_p": 0.0})
    elif kind == "missing":
        pairs.pop()
    elif kind == "ties":
        pairs[0] = pairs[0].model_copy(update={"tied_units": 123})
    elif kind == "hidden":
        pairs[0] = pairs[0].model_copy(update={"hidden": 1})
    elif kind == "alpha":
        object.__setattr__(result, "alpha", False)
    else:

        class Foreign(module.ScientificAblationWriterPairStatisticsV1):
            def __getattribute__(self, name: str) -> Any:
                pytest.fail("Foreign statistics must be denied before callbacks")

        pairs[0] = Foreign.model_construct(**pairs[0].model_dump())
    object.__setattr__(result, "pairs", tuple(pairs))
    with pytest.raises(ValueError):
        module.scientific_ablation_writer_statistics_bytes_v1(result)


def test_legitimate_whole_effect_inventory_can_exceed_source_metadata_tuple_limit(
    tmp_path: Path,
) -> None:
    design = build(
        "workspace",
        inventory=members(16),
        config=policy("workspace", examples_per_stratum=16),
        shape=geometry(
            num_modules=64, num_slots=1, state_dim=1, cycle_count=2, writers_per_cycle=1
        ),
    )
    budget = build_scientific_ablation_causal_observed_budget_v1(
        design, maximum_observation_scalars=28
    )
    rows = effects(design, schedule=lambda row, cycle, slot: 0)
    assert len(rows) == 12096 > 8192
    assert budget.payload.planned_total_teacher_executions == 768
    assert (
        budget.payload.total_trace_proposal_observation_scalar_bound
        <= design.payload.policy.maximum_total_numeric_scalars
    )
    result = module.build_scientific_ablation_writer_statistics_v1(design, rows)
    assert len(result.pairs) == 4032 and fsum(p.write_trials for p in result.pairs) == 12096
    assert pair(result).independent_units == 192 and not isnan(pair(result).sign_test_p)
    witness = dict(
        kind="iris-tests-synthetic-native-ablation-writer-statistics-v1",
        native_measurements=False,
        teacher_executions=0,
        scientific_acceptance="not_assessed",
        design=design.model_dump(mode="json"),
        effects=[e.model_dump(mode="json") for e in rows],
        statistics=json.loads(module.scientific_ablation_writer_statistics_bytes_v1(result)),
    )
    (tmp_path / "statistics-specimen.json").write_bytes(canonical_json_file_bytes(witness))
    (tmp_path / "design.json").write_bytes(
        canonical_json_file_bytes(design.model_dump(mode="json"))
    )
