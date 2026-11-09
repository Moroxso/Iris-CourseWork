"""Real original metadata; synthetic encoding specimens, NOT writer measurements.

Fixture setup trains/captures/scores one genuine earlier-best source. Preflight
and synthetic-layout validation then execute ZERO additional teacher/model passes.
No claim of all12 source execution, READY-preserving public preflight or effects.
"""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_causal_controls as source_module
from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_diagnostics import _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make

pytestmark = pytest.mark.integration
type Preflight = layout.ScientificAblationCausalReportBytePreflightV1


def rebuild(preflight: Preflight, **changes: Any) -> Preflight:
    kwargs = {
        name: getattr(preflight, name)
        for name in (
            "evaluation_plan",
            "evaluation",
            "evaluation_complete",
            "run_seed",
            "ordinary_teacher_graph_sha256",
            "controlled_teacher_graph_sha256",
            "sidecar_sha256",
            "maximum_report_bytes",
        )
    }
    return layout.build_scientific_ablation_causal_report_byte_preflight_v1(
        preflight.observed_budget, **(kwargs | changes)
    )


@pytest.fixture(scope="module")
def preflights(tmp_path_factory: pytest.TempPathFactory) -> tuple[Preflight, Preflight]:
    root = tmp_path_factory.mktemp("report-layout-original-source")
    ex = _experiment(_make(root / "case", rows=_rows()), root / "scores")
    owner = _heldout(ex)
    results = []
    for message in (False, True):
        with source_module._source_transaction(
            owner,
            input=_input(ex),
            policy=_policy(message),
            maximum_report_bytes=64 * 1024**2,
        ) as source:
            budget = build_scientific_ablation_causal_observed_budget_v1(
                source.design, maximum_observation_scalars=336
            )
            assert source.progress.evaluation is not None and source.progress.complete is not None
            with patch.object(source.model.teacher, "_execute") as execute:
                result = layout.build_scientific_ablation_causal_report_byte_preflight_v1(
                    budget,
                    evaluation_plan=source.progress.plan,
                    evaluation=source.progress.evaluation,
                    evaluation_complete=source.progress.complete,
                    run_seed=source.seed,
                    ordinary_teacher_graph_sha256=source.progress.plan.payload.generation_plan.payload.teacher_graph_sha256,
                    controlled_teacher_graph_sha256=source.controlled_graph_sha256,
                    sidecar_sha256=source.sidecar_sha256,
                    maximum_report_bytes=layout.MAX_REPORT_BYTES,
                )
                execute.assert_not_called()
            assert result.observed_budget is not budget
            assert result.evaluation_plan is not source.progress.plan
            assert source.progress.evaluation.payload.plan.payload.source.payload.selected_step == 1
            assert source.progress.evaluation.payload.plan.payload.source.payload.terminal_step == 3
            results.append(result)
        assert owner._state == owner._guard.context._state == "ready"
    assert len(results) == 2
    return results[0], results[1]


def specimen(shape: Any) -> Any:
    if type(shape) is layout._Fixed:
        return json.loads(shape.encoded)
    if type(shape) is layout._Scalar:
        return {
            "float": -sys.float_info.max,
            "int": -(2**63),
            "bool": False,
            "digest": "f" * 64,
            "text": "\0" * shape.limit,
        }[shape.kind]
    if type(shape) is layout._Array:
        return [specimen(shape.item) for _ in range(shape.count)]
    if type(shape) is layout._Object:
        return {name: specimen(child) for name, child in shape.members}
    assert type(shape) is layout._Choice
    return specimen(max(shape.choices, key=lambda child: child.bound))


def report_specimen(preflight: Preflight) -> dict[str, Any]:
    result: dict[str, Any] = specimen(preflight._shape)
    result["payload"]["byte_preflight"]["layout_sha256"] = preflight.layout_sha256
    result["payload"]["byte_preflight"]["report_byte_preflight_bound"] = (
        preflight.report_byte_preflight_bound
    )
    return result


@pytest.mark.parametrize("index", (0, 1))
def test_whole_encoding_specimen_fits_complete_before_forward_reservation(
    preflights: tuple[Preflight, Preflight], index: int, tmp_path: Path
) -> None:
    preflight = preflights[index]
    raw = report_specimen(preflight)
    content = layout.causal_report_layout_bytes_v1(raw, preflight)
    assert content == canonical_json_file_bytes(raw)
    assert len(content) <= preflight.report_byte_preflight_bound <= preflight.maximum_report_bytes
    assert preflight.report_byte_preflight_bound == sum(
        (
            preflight.source_metadata_bytes,
            preflight.ordinary_rows_byte_bound,
            preflight.controlled_passes_byte_bound,
            preflight.effects_byte_bound,
            preflight.statistics_byte_bound,
            preflight.framing_byte_bound,
        )
    )
    p, g = (
        preflight.observed_budget.payload,
        preflight.observed_budget.payload.design.payload.geometry,
    )
    assert preflight.post_intervention_scalar_bound == (
        p.planned_controlled_teacher_executions * g.cycle_count * g.num_slots * (g.state_dim + 6)
    )
    assert raw["payload"]["actual_teacher_executions"] == p.planned_total_teacher_executions
    assert len(raw["payload"]["statistics"]["pairs"]) == g.num_modules * (g.num_modules - 1)
    # Wrapper explicitly distinguishes synthetic encoding from measured reports.
    (tmp_path / "encoding-specimen.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-tests-complete-report-layout-encoding-only-v1",
                native_measurements=False,
                scientific_acceptance="not_assessed",
                actual_preflight_teacher_executions=0,
                complete_bound=preflight.report_byte_preflight_bound,
                actual_canonical_bytes=len(content),
                layout_sha256=preflight.layout_sha256,
                complete_layout=layout._descriptor(preflight._shape),
                source_metadata_bytes=preflight.source_metadata_bytes,
                ordinary_rows_byte_bound=preflight.ordinary_rows_byte_bound,
                controlled_passes_byte_bound=preflight.controlled_passes_byte_bound,
                effects_byte_bound=preflight.effects_byte_bound,
                statistics_byte_bound=preflight.statistics_byte_bound,
                framing_byte_bound=preflight.framing_byte_bound,
                post_intervention_scalar_bound=preflight.post_intervention_scalar_bound,
                example=raw,
            )
        )
    )


@pytest.mark.parametrize("index", (0, 1))
def test_quota_exact_boundary_and_one_below_never_shrink_design(
    preflights: tuple[Preflight, Preflight],
    index: int,
) -> None:
    preflight = preflights[index]
    exact = rebuild(preflight, maximum_report_bytes=preflight.report_byte_preflight_bound)
    for _ in range(3):
        if exact.maximum_report_bytes == exact.report_byte_preflight_bound:
            break
        exact = rebuild(exact, maximum_report_bytes=exact.report_byte_preflight_bound)
    assert exact.maximum_report_bytes == exact.report_byte_preflight_bound
    assert exact.observed_budget == preflight.observed_budget
    layout.causal_report_layout_bytes_v1(report_specimen(exact), exact)
    with pytest.raises(ValueError, match="before any forward"):
        rebuild(exact, maximum_report_bytes=exact.maximum_report_bytes - 1)


@pytest.mark.parametrize(
    "field",
    (
        "ordinary_rows",
        "effects",
        "statistics",
        "controlled_passes",
        "source_metadata",
        "byte_preflight",
        "tensor_geometry",
        "tensor_dtype",
        "hidden_tensor_field",
    ),
)
def test_late_changed_or_missing_layout_cannot_format(
    preflights: tuple[Preflight, Preflight],
    field: str,
) -> None:
    preflight = preflights[1]
    raw = report_specimen(preflight)
    p = raw["payload"]
    if field in ("ordinary_rows", "effects"):
        p[field].pop()
    elif field == "statistics":
        p[field]["pairs"].pop()
    elif field == "controlled_passes":
        p[field]["trials"]["donor"].pop()
    elif field == "source_metadata":
        p[field]["run_seed"] += 1
    elif field == "byte_preflight":
        p[field]["report_byte_preflight_bound"] -= 1
    else:
        tensor = p["controlled_passes"]["empty"][0]["output"]["controlled"]["states"][0]["slots"]
        if field == "tensor_geometry":
            tensor["values"][0].pop()
        elif field == "tensor_dtype":
            tensor["dtype"] = "float128"
        else:
            tensor["hidden"] = 0
    with pytest.raises(ValueError):
        layout.causal_report_layout_bytes_v1(raw, preflight)


@pytest.mark.parametrize(
    "field,value",
    (
        ("run_seed", True),
        ("run_seed", -1),
        ("ordinary_teacher_graph_sha256", "0" * 64),
        ("sidecar_sha256", "0" * 64),
        ("maximum_report_bytes", True),
        ("maximum_report_bytes", 0),
        ("maximum_report_bytes", layout.MAX_REPORT_BYTES + 1),
    ),
)
def test_preflight_metadata_and_cap_are_exact(
    preflights: tuple[Preflight, Preflight],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError):
        rebuild(preflights[0], **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "report_byte_preflight_bound",
        "source_metadata_bytes",
        "post_intervention_scalar_bound",
        "layout_sha256",
        "_shape",
    ),
)
def test_bypassed_immutable_preflight_is_revalidated(
    preflights: tuple[Preflight, Preflight],
    field: str,
) -> None:
    original = preflights[0]
    damaged = replace(original)
    value = getattr(damaged, field)
    if field == "_shape":
        value = layout._object(payload=layout.N)
    else:
        value = "0" * 64 if field == "layout_sha256" else value + 1
    object.__setattr__(damaged, field, value)
    with pytest.raises(ValueError):
        layout.causal_report_layout_bytes_v1(report_specimen(original), damaged)


def test_hidden_source_metadata_and_unselected_membership_never_preflight(
    preflights: tuple[Preflight, Preflight],
) -> None:
    original = preflights[0]
    hidden = copy.deepcopy(original.evaluation_plan)
    hidden.__dict__["hidden"] = None
    with pytest.raises(ValueError, match="closed"):
        rebuild(original, evaluation_plan=hidden)
    design = original.observed_budget.payload.design
    members = list(design.payload.members)
    omitted = next(i for i in range(len(members)) if i not in design.payload.selected_row_indices)
    members[omitted] = members[omitted].model_copy(update={"example_sha256": "e" * 64})
    from iris.artifacts.scientific_ablation_causal_design_v1 import (
        build_scientific_ablation_causal_design_v1,
    )

    foreign_design = build_scientific_ablation_causal_design_v1(
        tuple(members),
        purpose=design.payload.purpose,
        ablation=design.payload.ablation,
        policy=design.payload.policy,
        geometry=design.payload.geometry,
    )
    foreign_budget = build_scientific_ablation_causal_observed_budget_v1(
        foreign_design, maximum_observation_scalars=336
    )
    with pytest.raises(ValueError, match="membership"):
        rebuild(replace(original, observed_budget=foreign_budget))
