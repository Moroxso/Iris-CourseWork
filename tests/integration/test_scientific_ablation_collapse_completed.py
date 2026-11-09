"""Actual one-pass native ordinary+compact DATA, not whole workflow/science."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_collapse as module
from iris.application import scientific_ablation_diagnostics as diagnostics
from iris.application import synthetic_scoring_v2 as sandbox_scoring
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_collapse_v1 import (
    IrisScientificAblationCollapseV1,
    ScientificAblationCollapsePolicyV1,
)
from iris.artifacts.scientific_ablation_diagnostics_v1 import IrisScientificAblationDiagnosticsV1
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_diagnostics import (
    Experiment,
    _experiment,
    _heldout,
    _kwargs,
)
from tests.integration.test_scientific_ablation_evaluation_v2 import _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
POLICY = ScientificAblationCollapsePolicyV1()


@pytest.fixture(scope="module", autouse=True)
def no_real_candidates_or_loaders() -> Iterator[None]:
    with (
        patch.object(
            sandbox_scoring, "execute_python_job_v2", side_effect=AssertionError("No sandbox")
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No backbone loading"),
        ),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer loading"),
        ),
    ):
        yield


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("actual-retained-native-diagnostics")
    return _experiment(_make(root / "case"), root / "scores")


@dataclass(frozen=True)
class Measured:
    data: module.ScientificAblationCollapseCompletedV1
    owner: IrisScientificAblationHeldoutV1
    original: IrisScientificAblationDiagnosticsV1
    teacher_calls: int
    diagnostic_calls: int
    caller_restored: bool


@pytest.fixture(scope="module")
def measured(experiment: Experiment) -> Measured:
    owner = _heldout(experiment)
    original_measure = diagnostics.measure_scientific_ablation_diagnostics_v1
    observations: list[IrisScientificAblationDiagnosticsV1] = []

    def observe(*args: Any, **kwargs: Any) -> IrisScientificAblationDiagnosticsV1:
        report = original_measure(*args, **kwargs)
        observations.append(report)
        return report

    caller = capture_runtime_state(torch.device("cpu"))
    original_forward = IrisScientificAblationTeacherV1.evaluation_forward
    with (
        patch.object(
            module, "measure_scientific_ablation_diagnostics_v1", side_effect=observe
        ) as passes,
        patch.object(
            IrisScientificAblationTeacherV1,
            "evaluation_forward",
            autospec=True,
            side_effect=original_forward,
        ) as forwards,
        patch.object(
            IrisScientificAblationHeldoutV1,
            "execute_row",
            side_effect=AssertionError("No generation"),
        ),
        patch.object(
            sandbox_scoring, "score_synthetic_answer_v2", side_effect=AssertionError("No rescoring")
        ),
    ):
        result = module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY}
        )
    assert len(observations) == 1
    return Measured(
        result,
        owner,
        observations[0],
        forwards.call_count,
        passes.call_count,
        _same_runtime(caller, capture_runtime_state(torch.device("cpu"))),
    )


def test_actual_one_native_pass_returns_both_full_bound_reports(
    experiment: Experiment, measured: Measured, tmp_path: Path
) -> None:
    pair, owner = measured.data, measured.owner
    assert type(pair) is module.ScientificAblationCollapseCompletedV1
    d, c = pair.diagnostics.payload, pair.collapse.payload
    assert measured.teacher_calls == d.teacher_executions == c.teacher_executions == 4
    assert measured.diagnostic_calls == 1 and measured.caller_restored
    assert owner._state == owner._guard.context._state == "ready"
    assert pair.diagnostics is not measured.original
    assert pair.diagnostics.payload is not measured.original.payload
    assert diagnostics._checked(pair.diagnostics) == diagnostics._checked(measured.original)
    assert c.ordinary_diagnostics_sha256 == pair.diagnostics.sha256
    assert d.evaluation == c.evaluation == experiment.progress.evaluation
    assert len(c.groups) == 20 and tuple(m.task_type for m in c.members) == TASKS
    source = c.evaluation.payload.plan.payload.source.payload
    assert (source.selected_step, source.terminal_step, c.cycle_count) == (1, 3, 3)
    for row, ordinary in zip(c.rows, d.rows, strict=True):
        assert row.target_nll_sum == ordinary.target_nll_sum
        assert row.evaluation_ack_sha256 == ordinary.evaluation_ack_sha256
    (tmp_path / "retained-ordinary.json").write_bytes(diagnostics._checked(pair.diagnostics))
    (tmp_path / "retained-collapse.json").write_bytes(module._checked(pair.collapse))


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_retained_pair_keeps_original_mode_best_and_complete_rows(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    case = _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:])
    experiment = _experiment(case, tmp_path / "scores")
    owner = _heldout(experiment, routing, goal=goal, broadcast=broadcast)
    original = IrisScientificAblationTeacherV1.evaluation_forward
    with patch.object(
        IrisScientificAblationTeacherV1, "evaluation_forward", autospec=True, side_effect=original
    ) as observed:
        pair = module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY}
        )
    assert observed.call_count == pair.collapse.payload.teacher_executions == 1
    source = pair.collapse.payload.evaluation.payload.plan.payload.source.payload
    assert (source.ablation.routing, source.ablation.goal, source.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert (source.selected_step, source.terminal_step) == (1, 3)
    assert pair.collapse.payload.ordinary_diagnostics_sha256 == pair.diagnostics.sha256
    assert len(pair.collapse.payload.groups) == 20
    (tmp_path / "retained-ordinary.json").write_bytes(diagnostics._checked(pair.diagnostics))
    (tmp_path / "retained-collapse.json").write_bytes(module._checked(pair.collapse))


def test_legacy_projection_retains_old_return_without_detaching_full_report(
    experiment: Experiment, measured: Measured
) -> None:
    owner = _heldout(experiment)
    with patch.object(
        module, "_detach_completed", side_effect=AssertionError("Legacy must not copy full report")
    ) as detach:
        result = module.measure_scientific_ablation_collapse_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY}
        )
    detach.assert_not_called()
    assert result == measured.data.collapse


@pytest.mark.parametrize(
    "field",
    (
        "maximum_examples",
        "maximum_teacher_executions",
        "maximum_trace_scalars",
        "maximum_total_trace_scalars",
    ),
)
def test_retained_complete_budget_fails_before_any_forward(
    experiment: Experiment, field: str
) -> None:
    owner = _heldout(experiment)
    policy = ScientificAblationCollapsePolicyV1.model_validate({field: 1})
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(module.IrisScientificAblationCollapseError),
    ):
        module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": policy}
        )
    forward.assert_not_called()
    assert owner._state == "ready"


@pytest.mark.parametrize(
    "field",
    (
        "expected_plan_sha256",
        "expected_capture_sha256",
        "expected_evaluation_plan_sha256",
        "expected_evaluation_sha256",
    ),
)
def test_retained_wrong_original_anchor_fails_before_any_forward(
    experiment: Experiment, field: str
) -> None:
    owner = _heldout(experiment)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(module.IrisScientificAblationCollapseError),
    ):
        module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY, field: "0" * 64}
        )
    forward.assert_not_called()


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit))
def test_retained_detach_signal_and_late_cleanup_never_return_partial_data(
    experiment: Experiment, signal: type[BaseException]
) -> None:
    owner = _heldout(experiment)
    primary = signal("first")
    with (
        patch.object(module, "_detach_completed", side_effect=primary),
        patch.object(module, "_restore_caller", side_effect=RuntimeError("cleanup")),
        pytest.raises(signal) as caught,
    ):
        module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY}
        )
    assert caught.value is primary
    assert owner._state == owner._guard.context._state == "poisoned"


@pytest.mark.parametrize("changed", ("test_parquet", "score_row", "capture_row", "private_row"))
def test_late_source_drift_during_retained_detachment_returns_no_data(
    experiment: Experiment, tmp_path: Path, changed: str
) -> None:
    _late_drift(experiment, tmp_path, changed, cleanup=False, retained=True)


@pytest.mark.parametrize("retained", (False, True), ids=("legacy", "retained"))
@pytest.mark.parametrize("changed", ("test_parquet", "score_row", "capture_row", "private_row"))
def test_late_cleanup_source_drift_returns_no_compact_or_retained_data(
    experiment: Experiment, tmp_path: Path, changed: str, retained: bool
) -> None:
    _late_drift(experiment, tmp_path, changed, cleanup=True, retained=retained)


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit))
def test_final_source_audit_signal_keeps_identity_after_real_native_phase(
    experiment: Experiment, signal: type[BaseException]
) -> None:
    owner = _heldout(experiment)
    primary = signal("final-source-audit")
    caller = capture_runtime_state(torch.device("cpu"))
    original_forward = IrisScientificAblationTeacherV1.evaluation_forward
    with (
        patch.object(module, "_finish", side_effect=primary) as audit,
        patch.object(
            IrisScientificAblationTeacherV1,
            "evaluation_forward",
            autospec=True,
            side_effect=original_forward,
        ) as forwards,
        pytest.raises(signal) as caught,
    ):
        module.measure_scientific_ablation_collapse_completed_v1(
            owner, **{**_kwargs(experiment), "policy": POLICY}
        )
    assert caught.value is primary and audit.call_count == 1 and forwards.call_count == 4
    assert owner._state == owner._guard.context._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def _late_drift(
    experiment: Experiment, tmp_path: Path, changed: str, *, cleanup: bool, retained: bool
) -> None:
    copied = _copy(experiment.case, tmp_path / "case")
    local = _experiment(copied, tmp_path / "scores")
    owner = _heldout(local)
    original_detach, original_restore = module._detach_completed, _restore_caller
    calls: list[str] = []

    def mutate() -> None:
        assert owner._state == owner._guard.context._state == "busy"
        if changed == "private_row":
            row = owner._guard.rows[0]
            object.__setattr__(row, "difficulty", 1 + row.difficulty % 3)
        else:
            path = {
                "test_parquet": copied.root / copied.test.path,
                "score_row": local.progress.path / "score-00000/row.json",
                "capture_row": copied.capture.path / "row-00000/row.json",
            }[changed]
            assert path.is_file()
            path.write_bytes(path.read_bytes() + b"changed-during-detachment")
        calls.append(changed)

    def detach(
        ordinary: IrisScientificAblationDiagnosticsV1,
        compact: IrisScientificAblationCollapseV1,
    ) -> module.ScientificAblationCollapseCompletedV1:
        pair = original_detach(ordinary, compact)
        if not cleanup:
            mutate()
        return pair

    def restore(*args: Any, **kwargs: Any) -> None:
        original_restore(*args, **kwargs)
        if cleanup:
            mutate()

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(module, "_detach_completed", new=detach),
        patch.object(module, "_restore_caller", new=restore),
        pytest.raises(module.IrisScientificAblationCollapseError),
    ):
        if retained:
            module.measure_scientific_ablation_collapse_completed_v1(
                owner, **{**_kwargs(local), "policy": POLICY}
            )
        else:
            module.measure_scientific_ablation_collapse_v1(
                owner, **{**_kwargs(local), "policy": POLICY}
            )
    assert calls == [changed]
    assert owner._state == owner._guard.context._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "problem",
    ("ordinary_sha", "collapse_sha", "ordinary_extra", "collapse_extra", "ordinary_missing"),
)
def test_detachment_does_not_erase_forged_or_unchecked_data(
    measured: Measured, problem: str
) -> None:
    d, c = (
        measured.data.diagnostics.model_copy(deep=True),
        measured.data.collapse.model_copy(deep=True),
    )
    if problem == "ordinary_sha":
        object.__setattr__(d, "sha256", "0" * 64)
    elif problem == "collapse_sha":
        object.__setattr__(c, "sha256", "0" * 64)
    elif problem == "ordinary_extra":
        d.payload.__dict__["hidden"] = "private"
    elif problem == "collapse_extra":
        c.payload.__dict__["hidden"] = "private"
    else:
        del d.payload.__dict__["run_seed"]
    with pytest.raises(ValueError):
        module._detach_completed(d, c)
