"""Actual source-owned complete reports; tiny CPU engineering, NOT answer quality."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_writer_reports as service
from iris.application.scientific_ablation_causal_controls import (
    IrisScientificAblationCausalControlsError,
    ScientificAblationCausalCompleteLimitsV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    build_scientific_ablation_writer_report_v1,
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make
from tests.integration.test_scientific_ablation_generation_source import Routing

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("source-owned-writer-reports")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_both_public_source_owned_reports_full_schedule_after_ready(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    ex = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(ex, routing, goal=goal, broadcast=broadcast)
    context = owner._guard.context
    run = context._run
    model = run._optimizer._model
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    modes = tuple((child, child.training) for child in model.modules())
    default_dtype = torch.get_default_dtype()
    captured: list[Any] = []
    original = service._snapshot

    def snapshot(*args: Any, **kwargs: Any) -> Any:
        assert owner._state == context._state == context._request._state == "busy"
        assert run._busy
        source = args[0]
        captured.append(source.complete_preflight)
        return original(*args, **kwargs)

    runtime = ScientificAblationWriterCPURuntimeV1(
        torch_version=str(torch.__version__), torch_git_version=torch.version.git_version
    )
    with (
        patch.object(service, "_snapshot", new=snapshot),
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_controlled") as legacy,
        patch.object(type(context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            before_calls = execute.call_count
            measure = (
                service.measure_scientific_ablation_message_writer_report_v1
                if message
                else service.measure_scientific_ablation_workspace_writer_report_v1
            )
            report = measure(
                owner,
                input=_input(ex),
                policy=_policy(message),
                limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
            )
            ready(owner)
            assert all(child.training is was for child, was in modes)
            assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
            assert _same_runtime(private, run._runtime)
            assert torch.get_default_dtype() is default_dtype
            pf = captured[-1]
            assert pf is not None and captured[-2] is pf
            content = scientific_ablation_writer_report_bytes_v1(
                report, preflight=pf, runtime=runtime
            )
            reread = read_scientific_ablation_writer_report_v1(
                content, preflight=pf, runtime=runtime
            )
            assert canonical_json_file_bytes(reread.model_dump(mode="json")) == content
            p = report.payload
            n = 2 if message else 3
            assert (
                execute.call_count - before_calls
                == p.actual_teacher_executions
                == n * (10 if message else 8)
            )
            assert len(p.ordinary_rows) == n and len(p.effects) == n * 2 * 2 * 3
            assert len(p.statistics.pairs) == 12
            source = p.source_metadata.evaluation.payload.plan.payload.source.payload
            assert (source.selected_step, source.terminal_step) == (1, 3)
            assert len(p.source_metadata.observed_budget.payload.design.payload.members) == 4
            if broadcast == "disabled":
                assert all(
                    effect.genuine_distance == effect.negative_control_distance == 0.0
                    for effect in p.effects
                )
            protocol = "message" if message else "workspace"
            (tmp_path / f"public-writer-{protocol}.json").write_bytes(content)
        assert execute.call_count == 44
        legacy.assert_not_called()
        generate.assert_not_called()
        candidate.assert_not_called()
        loader.assert_not_called()
        tokenizer.assert_not_called()


@pytest.mark.parametrize("message", (False, True))
@pytest.mark.parametrize("quota", ("observation", "bytes", "runtime"))
def test_public_complete_quota_and_stream_denials_are_ready_zero_forward(
    experiment: Experiment, message: bool, quota: str
) -> None:
    owner = _heldout(experiment)
    model = owner._guard.context._run._optimizer._model
    limits = ScientificAblationCausalCompleteLimitsV1(
        maximum_observation_scalars=335 if quota == "observation" else 336,
        maximum_report_bytes=1 if quota == "bytes" else 512 * 1024**2,
    )
    measure = (
        service.measure_scientific_ablation_message_writer_report_v1
        if message
        else service.measure_scientific_ablation_workspace_writer_report_v1
    )
    version = "unknown-unproved-stream" if quota == "runtime" else torch.__version__
    with (
        patch.object(model.teacher, "_execute") as execute,
        patch.object(torch, "__version__", version),
    ):
        with pytest.raises(
            ValueError if quota == "runtime" else IrisScientificAblationCausalControlsError
        ):
            measure(owner, input=_input(experiment), policy=_policy(message), limits=limits)
        execute.assert_not_called()
    ready(owner)


@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
def test_construction_failure_no_report_and_first_signal_original_cleanup(
    experiment: Experiment, signal: type[BaseException]
) -> None:
    owner = _heldout(experiment)
    run = owner._guard.context._run
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    failure = signal("whole report construction failed")
    with (
        patch.object(service, "build_scientific_ablation_writer_report_v1", side_effect=failure),
        pytest.raises(
            IrisScientificAblationCausalControlsError if signal is ValueError else signal
        ) as caught,
    ):
        service.measure_scientific_ablation_message_writer_report_v1(
            owner,
            input=_input(experiment),
            policy=_policy(True),
            limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
        )
    assert (
        (caught.value.__cause__ is failure) if signal is ValueError else (caught.value is failure)
    )
    assert (
        owner._state
        == owner._guard.context._state
        == owner._guard.context._request._state
        == "poisoned"
    )
    assert run._poisoned and run._optimizer._poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, run._runtime)


@pytest.mark.parametrize("change", ("native-confidence", "mutate-restore", "report-counter"))
def test_actual_late_construction_drift_no_public_return_original_seals_rechecked(
    experiment: Experiment, change: str
) -> None:
    owner = _heldout(experiment)
    retained: list[Any] = []
    original_snapshot = service._snapshot
    original_builder = build_scientific_ablation_writer_report_v1

    def snapshot(*args: Any, **kwargs: Any) -> Any:
        retained.append(args[1])
        return original_snapshot(*args, **kwargs)

    def build(*args: Any, **kwargs: Any) -> Any:
        report = original_builder(*args, **kwargs)
        if change == "report-counter":
            object.__setattr__(
                report.payload,
                "actual_teacher_executions",
                report.payload.actual_teacher_executions - 1,
            )
        else:
            output = retained[0].passes[0].output
            tensor = output.controlled.workspace_trace[0].confidence
            previous = tensor.clone()
            tensor.masked_fill_(output.controlled.workspace_trace[0].occupancy, 0.125)
            if change == "mutate-restore":
                tensor.copy_(previous)
        return report

    with (
        patch.object(service, "_snapshot", new=snapshot),
        patch.object(service, "build_scientific_ablation_writer_report_v1", new=build),
        pytest.raises(IrisScientificAblationCausalControlsError),
    ):
        service.measure_scientific_ablation_message_writer_report_v1(
            owner,
            input=_input(experiment),
            policy=_policy(True),
            limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
        )
    assert (
        owner._state
        == owner._guard.context._state
        == owner._guard.context._request._state
        == "poisoned"
    )
    assert owner._guard.context._run._poisoned and not owner._guard.context._run._busy
