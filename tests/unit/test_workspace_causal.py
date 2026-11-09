"""Hand-computed causal statistics; cycles are not independent samples."""

from hashlib import sha256
from typing import Any

import pytest

from iris.workspace.causal import (
    WorkspaceCausalCaptureV1,
    WorkspaceCausalEffectV1,
    WorkspaceCausalPolicyV1,
    relative_state_distance_v1,
    summarize_workspace_causal_v1,
)


def capture(index: int, ablated: float = 0.2, shuffled: float = 0.0) -> WorkspaceCausalCaptureV1:
    return WorkspaceCausalCaptureV1(
        policy=WorkspaceCausalPolicyV1(shuffle_seed=42),
        example_id=f"row-{index}",
        prompt_sha256=sha256(str(index).encode()).hexdigest(),
        teacher_graph_sha256="1" * 64,
        sidecar_state_sha256="2" * 64,
        seed=42,
        num_modules=2,
        num_slots=2,
        cycle_count=3,
        writers_per_cycle=1,
        executions=5,
        effects=tuple(
            WorkspaceCausalEffectV1(
                write_cycle=c,
                source_module=0,
                slot=0,
                target_module=1,
                survives_to_next_read=True,
                overwrote_existing_record=False,
                ablated_distance=ablated,
                shuffled_distance=shuffled,
                ablation_plan_sha256="3" * 64,
                shuffle_plan_sha256="4" * 64,
            )
            for c in (1, 2)
        ),
    )


def test_sign_test_clusters_cycles_and_holm_counts_absent_pairs() -> None:
    result = summarize_workspace_causal_v1(tuple(capture(i) for i in range(8)))
    observed, absent = result.pairs
    assert (observed.source_module, observed.target_module) == (0, 1)
    assert observed.examples == observed.positive_examples == 8
    assert observed.write_trials == 16
    assert observed.sign_test_p == 1 / 256  # NOT 1 / 65536 for 16 pseudo-samples.
    assert observed.holm_adjusted_p == 2 / 256
    assert observed.significant_positive_effect
    assert observed.mean_excess == pytest.approx(0.2)
    assert absent.examples == absent.write_trials == 0
    assert absent.sign_test_p == absent.holm_adjusted_p == 1
    assert absent.mean_excess is None and not absent.significant_positive_effect
    assert result.scientific_acceptance == "not_assessed"
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_ties_and_negative_effects_are_not_silently_dropped_from_means() -> None:
    result = summarize_workspace_causal_v1((capture(0), capture(1, 0.0, 0.2), capture(2, 1e-8)))
    row = result.pairs[0]
    assert (row.positive_examples, row.negative_examples, row.tied_examples) == (1, 1, 1)
    assert row.sign_test_p == 0.75
    assert row.mean_excess == pytest.approx(1e-8 / 3, abs=1e-16)
    assert not row.significant_positive_effect
    tied = summarize_workspace_causal_v1((capture(0, 0.0),)).pairs[0]
    assert tied.sign_test_p == 1 and tied.tied_examples == 1


def test_order_independent_and_no_cross_seed_or_repeated_prompt_pooling() -> None:
    rows = tuple(capture(i) for i in range(8))
    assert summarize_workspace_causal_v1(rows) == summarize_workspace_causal_v1(rows[::-1])
    with pytest.raises(ValueError, match="Repeated"):
        summarize_workspace_causal_v1((rows[0], rows[0]))
    data = rows[1].model_dump()
    data["prompt_sha256"] = rows[0].prompt_sha256
    with pytest.raises(ValueError, match="Repeated"):
        summarize_workspace_causal_v1((rows[0], WorkspaceCausalCaptureV1.model_validate(data)))
    for field, value in (("seed", 1337), ("sidecar_state_sha256", "f" * 64)):
        data = rows[1].model_dump()
        data[field] = value
        with pytest.raises(ValueError, match="Cannot pool"):
            summarize_workspace_causal_v1((rows[0], WorkspaceCausalCaptureV1.model_validate(data)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("executions", 4),
        ("num_modules", 3),
        ("cycle_count", 1),
        ("effects", ()),
        ("seed", True),
        ("teacher_graph_sha256", "bad"),
    ],
)
def test_invalid_capture_geometry_and_context_rejected(field: str, value: Any) -> None:
    data = capture(0).model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        WorkspaceCausalCaptureV1.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("write_cycle", 3),
        ("source_module", 1),
        ("target_module", 0),
        ("slot", 2),
        ("ablated_distance", float("nan")),
        ("shuffled_distance", 1.1),
    ],
)
def test_invalid_effect_rejected(field: str, value: Any) -> None:
    data = capture(0).model_dump()
    data["effects"][0][field] = value
    with pytest.raises(ValueError):
        WorkspaceCausalCaptureV1.model_validate(data)


@pytest.mark.parametrize("scale", [1e-310, 1.0, 1e308])
def test_distance_is_scale_stable_and_bounded(scale: float) -> None:
    assert relative_state_distance_v1((scale, 0.0), (scale, 0.0)) == 0
    assert relative_state_distance_v1((scale, 0.0), (-scale, 0.0)) == 1
    assert relative_state_distance_v1((scale, 0.0), (0.0, scale)) == pytest.approx(2**-0.5)
    assert relative_state_distance_v1((scale, 0.0), (0.0, 0.0)) == 1


@pytest.mark.parametrize("right", [(0.0, float("nan")), (float("inf"), 0.0), (), (1.0,)])
def test_distance_rejects_invalid_vectors(right: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        relative_state_distance_v1((0.0, 0.0), right)


def test_empty_inputs_forged_capture_and_invalid_statistics_policy_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_workspace_causal_v1(())
    with pytest.raises(ValueError):
        summarize_workspace_causal_v1((capture(0).model_copy(update={"effects": ()}),))
    for policy in ({"alpha": 0.0}, {"alpha": float("nan")}, {"numerical_tolerance": -1.0}):
        with pytest.raises(ValueError):
            summarize_workspace_causal_v1((capture(0),), **policy)
