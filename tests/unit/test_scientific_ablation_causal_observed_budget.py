"""All observed controlled passes need extra ALL7 volume; no model forwards."""

from pathlib import Path
from typing import Any, NoReturn

import pytest

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationCausalMemberV1,
    scientific_ablation_causal_design_bytes_v1,
)
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    DOMAIN,
    IrisScientificAblationCausalObservedBudgetV1,
    build_scientific_ablation_causal_observed_budget_v1,
    scientific_ablation_causal_observed_budget_bytes_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_scientific_ablation_causal_design import (
    MODES,
    Protocol,
    build,
    geometry,
    policy,
)

COUNTS = (
    "planned_ordinary_teacher_executions",
    "planned_controlled_teacher_executions",
    "planned_total_teacher_executions",
    "planned_observed_teacher_executions",
    "observation_scalar_bound_per_pass",
    "total_observation_scalar_bound",
    "original_trace_proposal_scalar_bound",
    "total_trace_proposal_observation_scalar_bound",
)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("ablation", MODES)
def test_all12_both_protocols_reserve_observations_without_extra_hidden_passes(
    protocol: Protocol, ablation: CognitiveAblationPolicyV1, tmp_path: Path
) -> None:
    design = build(protocol, ablation=ablation)
    original = design.model_dump(mode="json")
    result = build_scientific_ablation_causal_observed_budget_v1(
        design, maximum_observation_scalars=168
    )
    p, b = result.payload, design.payload.budget
    assert p.design == design and design.model_dump(mode="json") == original
    assert p.design is not design
    assert p.scientific_acceptance == "not_assessed"
    assert p.observation_scalar_bound_per_pass == 168  # 2*C*K*(D+6), C3/K2/D8
    assert p.planned_total_teacher_executions == b.total_executions
    assert p.planned_total_teacher_executions // b.examples == (
        8 if protocol == "workspace" else 10
    )
    assert p.planned_observed_teacher_executions == b.controlled_executions
    assert p.total_observation_scalar_bound == 168 * b.controlled_executions
    assert p.original_trace_proposal_scalar_bound == b.total_numeric_scalars
    assert p.total_trace_proposal_observation_scalar_bound == (
        b.total_numeric_scalars + 168 * b.controlled_executions
    )
    content = scientific_ablation_causal_observed_budget_bytes_v1(result)
    assert IrisScientificAblationCausalObservedBudgetV1.model_validate_json(content) == result
    assert result.sha256 != domain_sha256(
        "iris/scientific-ablation-causal-design/v1",
        canonical_json_bytes(result.payload.model_dump(mode="json")),
    )
    (tmp_path / "observed-budget.json").write_bytes(content)
    (tmp_path / "design.json").write_bytes(scientific_ablation_causal_design_bytes_v1(design))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_full_ordinary_trace_is_counted_but_not_mistaken_for_an_observed_control(
    protocol: Protocol, tmp_path: Path
) -> None:
    design = build(protocol, config=policy(protocol, ordinary_trace_mode="full"))
    result = build_scientific_ablation_causal_observed_budget_v1(
        design, maximum_observation_scalars=168
    )
    b = design.payload.budget
    assert b.ordinary_trace_scalars == b.full_trace_scalars
    assert b.metadata_trace_executions == 0
    assert result.payload.planned_observed_teacher_executions == (
        b.total_executions - b.ordinary_executions
    )
    (tmp_path / "observed-budget.json").write_bytes(
        scientific_ablation_causal_observed_budget_bytes_v1(result)
    )
    (tmp_path / "design.json").write_bytes(scientific_ablation_causal_design_bytes_v1(design))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_full5000_declared_m8_k8_d256_budget_not_actual_scientific_measurement(
    protocol: Protocol, tmp_path: Path
) -> None:
    inventory: list[ScientificAblationCausalMemberV1] = []
    for task, count in zip(TASKS, (2000, 1000, 1000, 1000), strict=True):
        for local in range(count):
            index = len(inventory)
            inventory.append(
                ScientificAblationCausalMemberV1(
                    row_index=index,
                    example_id=f"test-{index}",
                    example_sha256=f"{index:064x}",
                    prompt_sha256=f"{index + 5000:064x}",
                    cognitive_prompt_sha256=f"{index + 10000:064x}",
                    task_type=task,
                    difficulty=local % 3 + 1,
                )
            )
    design = build(
        protocol,
        inventory=tuple(inventory),
        purpose="scientific",
        shape=geometry(num_modules=8, num_slots=8, state_dim=256),
        config=policy(protocol, maximum_total_numeric_scalars=64_000_000),
    )
    result = build_scientific_ablation_causal_observed_budget_v1(
        design, maximum_observation_scalars=12_576
    )
    assert len(result.payload.design.payload.members) == 5000
    assert len(result.payload.design.payload.groups) == 20
    assert len(result.payload.design.payload.selected_row_indices) == 48
    assert result.payload.observation_scalar_bound_per_pass == 12_576
    assert result.payload.planned_total_teacher_executions == 48 * (
        8 if protocol == "workspace" else 10
    )
    assert result.payload.scientific_acceptance == "not_assessed"
    (tmp_path / "observed-budget.json").write_bytes(
        scientific_ablation_causal_observed_budget_bytes_v1(result)
    )
    (tmp_path / "design.json").write_bytes(scientific_ablation_causal_design_bytes_v1(design))


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_complete_numeric_cap_includes_every_observed_control_and_does_not_shrink(
    protocol: Protocol,
) -> None:
    initial = build(protocol)
    required = initial.payload.budget.total_numeric_scalars + (
        168 * initial.payload.budget.controlled_executions
    )
    exact = build(protocol, config=policy(protocol, maximum_total_numeric_scalars=required))
    assert (
        build_scientific_ablation_causal_observed_budget_v1(
            exact, maximum_observation_scalars=168
        ).payload.total_trace_proposal_observation_scalar_bound
        == required
    )
    insufficient = build(
        protocol, config=policy(protocol, maximum_total_numeric_scalars=required - 1)
    )
    selected = insufficient.payload.selected_row_indices
    with pytest.raises(ValueError, match="Whole trace/proposal/ALL7"):
        build_scientific_ablation_causal_observed_budget_v1(
            insufficient, maximum_observation_scalars=168
        )
    assert insufficient.payload.selected_row_indices == selected
    # The older design legitimately reserved fewer scalars. Keep it unchanged;
    # that prior acceptance is not complete observed execution preflight.
    old_only = build(
        protocol,
        config=policy(
            protocol, maximum_total_numeric_scalars=initial.payload.budget.total_numeric_scalars
        ),
    )
    with pytest.raises(ValueError, match="Whole trace/proposal/ALL7"):
        build_scientific_ablation_causal_observed_budget_v1(
            old_only, maximum_observation_scalars=168
        )


@pytest.mark.parametrize("maximum", (True, False, 168.0, "168", 0, 2_000_001, object()))
def test_observation_limit_requires_exact_bounded_integer(maximum: Any) -> None:
    with pytest.raises(ValueError, match="Exact bounded"):
        build_scientific_ablation_causal_observed_budget_v1(
            build(), maximum_observation_scalars=maximum
        )


def test_int_subclass_and_one_below_per_pass_limit_are_rejected() -> None:
    class ForeignInt(int):
        pass

    with pytest.raises(ValueError, match="Exact bounded"):
        build_scientific_ablation_causal_observed_budget_v1(
            build(), maximum_observation_scalars=ForeignInt(168)
        )
    with pytest.raises(ValueError, match="Per-pass ALL7"):
        build_scientific_ablation_causal_observed_budget_v1(
            build(), maximum_observation_scalars=167
        )


def _rehashed(
    result: IrisScientificAblationCausalObservedBudgetV1, **changes: Any
) -> IrisScientificAblationCausalObservedBudgetV1:
    payload = result.payload.model_copy(update=changes)
    return IrisScientificAblationCausalObservedBudgetV1.model_construct(
        sha256=domain_sha256(DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )


@pytest.mark.parametrize("field", COUNTS)
def test_rehashed_under_or_overcount_cannot_waive_derived_volume(field: str) -> None:
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    for difference in (-1, 1):
        forged = _rehashed(result, **{field: getattr(result.payload, field) + difference})
        with pytest.raises(ValueError):
            scientific_ablation_causal_observed_budget_bytes_v1(forged)


@pytest.mark.parametrize("field", COUNTS)
def test_rehashed_bool_does_not_masquerade_as_an_execution_or_scalar_count(field: str) -> None:
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    with pytest.raises(ValueError):
        scientific_ablation_causal_observed_budget_bytes_v1(_rehashed(result, **{field: True}))


@pytest.mark.parametrize("field", ("kind", "scope", "scientific_acceptance"))
def test_foreign_string_primitive_is_rejected_without_overloaded_comparison(field: str) -> None:
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    calls: list[str] = []

    class ForeignString(str):
        def __ne__(self, other: object) -> bool:
            calls.append("untrusted comparison")
            return False

    payload = result.payload.model_copy(
        update={field: ForeignString(getattr(result.payload, field))}
    )
    forged = result.model_copy(update={"payload": payload})
    with pytest.raises(ValueError):
        scientific_ablation_causal_observed_budget_bytes_v1(forged)
    assert not calls


@pytest.mark.parametrize(
    "change",
    (
        {"scope": "source-owned-empty-controls-not-causal-effects-or-answer-quality"},
        {"scientific_acceptance": "accepted"},
        {"kind": "iris-workspace-causal-capture-v1"},
    ),
)
def test_legacy_scope_and_scientific_claim_relabel_are_rejected(change: dict[str, str]) -> None:
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    with pytest.raises(ValueError):
        scientific_ablation_causal_observed_budget_bytes_v1(_rehashed(result, **change))


def test_hidden_budget_metadata_and_nested_design_drift_are_rejected() -> None:
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    object.__setattr__(result.payload, "hidden", "not part of the closed contract")
    with pytest.raises(ValueError, match="Exact closed"):
        scientific_ablation_causal_observed_budget_bytes_v1(result)
    result = build_scientific_ablation_causal_observed_budget_v1(
        build(), maximum_observation_scalars=168
    )
    object.__setattr__(result.payload.design.payload.geometry, "state_dim", 9)
    with pytest.raises(ValueError):
        scientific_ablation_causal_observed_budget_bytes_v1(result)


def test_numeric_preflight_calls_no_teacher_and_owns_its_input_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        calls.append("teacher forward")
        raise AssertionError("Numeric preflight must not execute the model")

    monkeypatch.setattr(IrisTeacherForcingModel, "_execute", forbidden)
    design = build()
    result = build_scientific_ablation_causal_observed_budget_v1(
        design, maximum_observation_scalars=168
    )
    object.__setattr__(design.payload.geometry, "state_dim", 9)
    assert result.payload.design.payload.geometry.state_dim == 8
    assert not calls
