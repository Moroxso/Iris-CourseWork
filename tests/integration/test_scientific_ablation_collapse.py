"""Actual selected-best mode-bound full-test counts; never host candidate execution."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_collapse as module
from iris.application import scientific_ablation_diagnostics as diagnostics
from iris.application import scientific_ablation_evaluation_v2 as scoring
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_collapse_v1 import (
    IrisScientificAblationCollapseV1,
    ScientificAblationCollapsePayloadV1,
    ScientificAblationCollapsePolicyV1,
    scientific_ablation_collapse_groups_v1,
    scientific_ablation_collapse_witness_v1,
)
from iris.artifacts.scientific_ablation_diagnostics_v1 import IrisScientificAblationDiagnosticsV1
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_diagnostics import (
    Experiment,
    _experiment,
    _heldout,
    _kwargs,
)
from tests.integration.test_scientific_ablation_evaluation_ledger import _plan, _run
from tests.integration.test_scientific_ablation_evaluation_v2 import _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
Error = module.IrisScientificAblationCollapseError
POLICY = ScientificAblationCollapsePolicyV1()


def _measure(
    experiment: Experiment, heldout: IrisScientificAblationHeldoutV1, **changes: Any
) -> IrisScientificAblationCollapseV1:
    return module.measure_scientific_ablation_collapse_v1(
        heldout, **{**_kwargs(experiment), "policy": POLICY, **changes}
    )


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("actual-native-collapse-source")
    return _experiment(_make(root / "case"), root / "scores")


@pytest.fixture(scope="module")
def result(experiment: Experiment) -> IrisScientificAblationCollapseV1:
    return _measure(experiment, _heldout(experiment))


def test_actual_four_families_all_twenty_groups_best_c3_and_compact_trace_parity(
    experiment: Experiment, tmp_path: Path
) -> None:
    heldout = _heldout(experiment)
    ordinary: list[IrisScientificAblationDiagnosticsV1] = []
    original = diagnostics.measure_scientific_ablation_diagnostics_v1
    constructions: list[str] = []
    initialize = torch.nn.Module.__init__

    def capture(*args: Any, **kwargs: Any) -> IrisScientificAblationDiagnosticsV1:
        report = original(*args, **kwargs)
        ordinary.append(report)
        return report

    def construct(self: torch.nn.Module) -> None:
        constructions.append(type(self).__name__)
        assert type(self) is torch.nn.ModuleList
        initialize(self)

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(module, "measure_scientific_ablation_diagnostics_v1", new=capture),
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as generate,
        patch.object(scoring, "score_synthetic_answer_v2") as score,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as execute,
        patch.object(torch.nn.Module, "__init__", new=construct),
    ):
        report = _measure(experiment, heldout)
    generate.assert_not_called()
    score.assert_not_called()
    execute.assert_not_called()
    assert constructions == ["ModuleList"] * 8
    p = report.payload
    assert p.ordinary_diagnostics_sha256 == ordinary[0].sha256
    assert p.teacher_executions == len(p.rows) == len(p.members) == 4
    source = p.evaluation.payload.plan.payload.source.payload
    assert (source.selected_step, source.terminal_step) == (1, 3)
    assert tuple(member.task_type for member in p.members) == TASKS
    assert tuple(member.difficulty for member in p.members) == tuple(
        row.difficulty for row in heldout._guard.rows
    )
    for row, diagnostic in zip(p.rows, ordinary[0].payload.rows, strict=True):
        assert (
            row.metadata_trace_sha256,
            row.context,
            row.cycles,
        ) == scientific_ablation_collapse_witness_v1(
            diagnostic.trace, expected_ablation_policy_sha256=source.ablation_policy_sha256
        )
        assert row.context.cycle_count == 3
        assert row.target_nll_sum == diagnostic.target_nll_sum
        assert row.evaluation_row_sha256 == experiment.progress.rows[row.row_index].sha256
        assert (
            row.evaluation_ack_sha256 == experiment.progress.acknowledgements[row.row_index].sha256
        )
    summary = p.groups[0].summary
    assert summary is not None
    assert summary.module_win_counts == ordinary[0].payload.workspace.module_win_counts
    assert summary.slot_allocation_counts == ordinary[0].payload.workspace.slot_allocation_counts
    assert (
        p.source_20_4.outcome == "not_assessed_smoke" and p.scientific_acceptance == "not_assessed"
    )
    encoded = module._checked(report)
    for field in (
        b'"trace"',
        b'"read_attention"',
        b'"probes"',
        b'"expected_response"',
        b'"output_text"',
        b'"full"',
        b'"ttl"',
    ):
        assert field not in encoded
    assert len(encoded) < len(diagnostics._checked(ordinary[0]))
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert heldout.manifest == experiment.case.capture.plan.payload.input
    (tmp_path / "collapse.json").write_bytes(encoded)
    (tmp_path / "ordinary.json").write_bytes(diagnostics._checked(ordinary[0]))


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_best_not_terminal_compact_pass(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    case = _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:])
    experiment = _experiment(case, tmp_path / "scores")
    actual: list[IrisScientificAblationDiagnosticsV1] = []
    original = diagnostics.measure_scientific_ablation_diagnostics_v1

    def capture(*args: Any, **kwargs: Any) -> IrisScientificAblationDiagnosticsV1:
        result = original(*args, **kwargs)
        actual.append(result)
        return result

    with patch.object(module, "measure_scientific_ablation_diagnostics_v1", new=capture):
        report = _measure(experiment, _heldout(experiment, routing, goal=goal, broadcast=broadcast))
    p = report.payload
    source = p.evaluation.payload.plan.payload.source.payload
    assert (source.ablation.routing, source.ablation.goal, source.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert (source.selected_step, source.terminal_step, p.cycle_count) == (1, 3, 3)
    assert p.rows[0].context.ablation_policy_sha256 == source.ablation_policy_sha256
    assert p.rows[0].target_nll_sum == case.capture.rows[0].payload.target_nll_sum
    assert len(p.groups) == 20 and p.groups[0].examples == 1
    assert any(group.summary is None for group in p.groups)
    (tmp_path / "collapse.json").write_bytes(module._checked(report))
    assert len(actual) == 1 and p.ordinary_diagnostics_sha256 == actual[0].sha256
    (tmp_path / "ordinary.json").write_bytes(diagnostics._checked(actual[0]))


@pytest.mark.parametrize(
    "field",
    (
        "maximum_examples",
        "maximum_teacher_executions",
        "maximum_trace_scalars",
        "maximum_total_trace_scalars",
    ),
)
def test_full_budget_rejects_before_any_teacher_call(experiment: Experiment, field: str) -> None:
    heldout = _heldout(experiment)
    policy = ScientificAblationCollapsePolicyV1.model_validate({field: 1})
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, heldout, policy=policy)
    forward.assert_not_called()
    assert heldout._state == heldout._guard.context._state == "ready"


@pytest.mark.parametrize(
    "field",
    (
        "expected_plan_sha256",
        "expected_capture_sha256",
        "expected_evaluation_plan_sha256",
        "expected_evaluation_sha256",
    ),
)
def test_four_independent_anchors_reject_before_any_forward(
    experiment: Experiment, field: str
) -> None:
    heldout = _heldout(experiment)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, heldout, **{field: "0" * 64})
    forward.assert_not_called()


@pytest.mark.parametrize(
    "problem",
    ("nll", "ack", "mode", "membership", "writer", "groups", "threshold", "policy", "trace_hash"),
)
def test_rehashed_but_inconsistent_compact_reports_rejected(
    result: IrisScientificAblationCollapseV1, problem: str
) -> None:
    raw = result.payload.model_dump()
    if problem == "nll":
        raw["rows"][0]["target_nll_sum"] += 1
    elif problem == "ack":
        raw["rows"][0]["evaluation_ack_sha256"] = "0" * 64
    elif problem == "mode":
        raw["rows"][0]["context"]["ablation_policy_sha256"] = "0" * 64
    elif problem == "membership":
        raw["members"] = raw["members"][::-1]
        raw["membership_sha256"] = domain_sha256(
            module._MEMBERS_DOMAIN, canonical_json_bytes(raw["members"])
        )
    elif problem == "writer":
        raw["writers_per_cycle"] += 1
    elif problem == "groups":
        raw["groups"][0]["summary"]["effective_active_modules"] += 1
    elif problem == "threshold":
        raw["source_20_4"]["minimum_effective_modules"] = 1
    elif problem == "policy":
        raw["policy_sha256"] = "0" * 64
    else:
        raw["rows"][0]["metadata_trace_sha256"] = "bad"
    with pytest.raises(ValueError):
        ScientificAblationCollapsePayloadV1.model_validate(raw)


@pytest.mark.parametrize("problem", ("extra", "missing", "subclass", "wrapper_sha"))
def test_exact_tree_cannot_erase_unchecked_metadata(
    result: IrisScientificAblationCollapseV1, problem: str
) -> None:
    forged = result.model_copy(deep=True)
    if problem == "extra":
        forged.payload.__dict__["raw_prompt"] = "private"
    elif problem == "missing":
        del forged.payload.__dict__["run_seed"]
    elif problem == "subclass":

        class Other(IrisScientificAblationCollapseV1):
            pass

        forged = Other.model_construct(**result.model_dump())
    else:
        object.__setattr__(forged, "sha256", "0" * 64)
    with pytest.raises(ValueError):
        module._checked(forged)


def test_projection_busy_reentry_is_rejected_without_poison(experiment: Experiment) -> None:
    heldout = _heldout(experiment)
    original = scientific_ablation_collapse_groups_v1
    reentries = []

    def groups(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(Error):
            _measure(experiment, heldout)
        reentries.append(heldout._state)
        return original(*args, **kwargs)

    with patch.object(module, "scientific_ablation_collapse_groups_v1", new=groups):
        _measure(experiment, heldout)
    assert reentries == ["busy"] and heldout._state == "ready"


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit))
def test_projection_first_signal_preserved_and_native_owner_poisoned(
    experiment: Experiment, signal: type[BaseException]
) -> None:
    heldout = _heldout(experiment)
    primary = signal("first")
    with (
        patch.object(module, "scientific_ablation_collapse_groups_v1", side_effect=primary),
        patch.object(module, "_restore_caller", side_effect=RuntimeError("cleanup")),
        pytest.raises(signal) as captured,
    ):
        _measure(experiment, heldout)
    assert captured.value is primary
    assert heldout._state == heldout._guard.context._state == "poisoned"


def test_late_physical_test_drift_after_projection_returns_no_report(
    experiment: Experiment, tmp_path: Path
) -> None:
    copied = _copy(experiment.case, tmp_path / "case")
    local = _experiment(copied, tmp_path / "scores")
    heldout = _heldout(local)
    original = scientific_ablation_collapse_groups_v1

    def groups(*args: Any, **kwargs: Any) -> Any:
        report = original(*args, **kwargs)
        path = copied.root / copied.test.path
        path.write_bytes(path.read_bytes() + b"changed")
        return report

    with (
        patch.object(module, "scientific_ablation_collapse_groups_v1", new=groups),
        pytest.raises(Error),
    ):
        _measure(local, heldout)
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_caller_runtime_dtype_device_and_exact_retry(
    experiment: Experiment, default: str, inference: bool
) -> None:
    heldout = _heldout(experiment)
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(default), torch.inference_mode(inference):
            assert _measure(experiment, heldout) == _measure(experiment, heldout)
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device(default)
            assert torch.is_inference_mode_enabled() is inference
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)


def test_partial_original_score_ledger_is_not_complete_test_evidence(
    experiment: Experiment, tmp_path: Path
) -> None:
    copied = _copy(experiment.case, tmp_path / "case")
    partial = _run(copied, _plan(copied), tmp_path / "partial-scores", rows_per_call=1)
    assert partial.complete is None and partial.evaluation is None
    heldout = _heldout(Experiment(copied, experiment.progress))
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(Experiment(copied, experiment.progress), heldout, evaluation_path=partial.path)
    forward.assert_not_called()
    assert heldout._state == "ready"


def test_late_private_difficulty_change_is_not_projected_as_original_metadata(
    experiment: Experiment,
) -> None:
    heldout = _heldout(experiment)
    original = scientific_ablation_collapse_groups_v1

    def groups(*args: Any, **kwargs: Any) -> Any:
        report = original(*args, **kwargs)
        row = heldout._guard.rows[0]
        object.__setattr__(row, "difficulty", 1 + row.difficulty % 3)
        return report

    with (
        patch.object(module, "scientific_ablation_collapse_groups_v1", new=groups),
        pytest.raises(Error),
    ):
        _measure(experiment, heldout)
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize("problem", ("baseline", "different_mode", "full"))
def test_witness_never_nulls_policy_or_accepts_baseline_other_mode_or_full_trace(
    experiment: Experiment, problem: str
) -> None:
    report = diagnostics.measure_scientific_ablation_diagnostics_v1(
        _heldout(experiment), **_kwargs(experiment)
    )
    trace = report.payload.rows[0].trace
    expected = trace.intervention_plan_sha256
    assert expected is not None
    if problem == "baseline":
        trace = trace.model_copy(update={"intervention_plan_sha256": None})
    elif problem == "different_mode":
        expected = "0" * 64
    else:
        # Invalid full-mode metadata is denied during trace revalidation, not relabelled.
        trace = trace.model_copy(
            update={"options": trace.options.model_copy(update={"mode": "full"})}
        )
    with pytest.raises(ValueError):
        scientific_ablation_collapse_witness_v1(trace, expected_ablation_policy_sha256=expected)
