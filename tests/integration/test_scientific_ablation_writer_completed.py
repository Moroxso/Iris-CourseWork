"""Fresh staged CPU handoff checks; no publication or scientific acceptance."""

from dataclasses import FrozenInstanceError, fields
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as controls
from iris.application import scientific_ablation_writer_reports as service
from iris.application.scientific_ablation_causal_controls import _source_transaction
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_writer_effects import _measure_writer_effects
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    build_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    build_scientific_ablation_writer_report_v1,
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    build_scientific_ablation_writer_source_v1,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
    scientific_ablation_writer_source_preflight_v1,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("completed-writer-source")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


def _measure(experiment: Experiment, owner: Any, message: bool) -> service.WriterCompleted:
    measure = (
        service.measure_scientific_ablation_message_writer_completed_v1
        if message
        else service.measure_scientific_ablation_workspace_writer_completed_v1
    )
    return measure(
        owner,
        input=_input(experiment),
        policy=_policy(message),
        limits=controls.ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
    )


@pytest.mark.parametrize("message", (False, True))
def test_fresh_both_handoff_one_native_stage_after_cleanup(
    experiment: Experiment, message: bool
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    run = context._run
    model = run._optimizer._model
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    modes = tuple((child, child.training) for child in model.modules())
    dtype = torch.get_default_dtype()
    captured: list[Any] = []
    original_snapshot = service._snapshot
    original_init = torch.nn.Module.__init__

    def initialize(module: torch.nn.Module) -> None:
        # Qwen slices its existing ModuleList during ordinary teacher execution.
        # Permit only these borrowed containers, never a new parameter-owning graph.
        assert type(module) is torch.nn.ModuleList, type(module).__name__
        original_init(module)

    def snapshot(*args: Any, **kwargs: Any) -> Any:
        assert owner._state == context._state == context._request._state == "busy"
        assert run._busy
        captured.append(args[0].complete_preflight)
        return original_snapshot(*args, **kwargs)

    with (
        patch.object(service, "_snapshot", new=snapshot),
        patch.object(service, "_source_transaction", wraps=_source_transaction) as tx,
        patch.object(service, "_measure_writer_effects", wraps=_measure_writer_effects) as stage,
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(torch.nn.Module, "__init__", new=initialize),
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_controlled") as legacy,
        patch.object(type(context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        completed = _measure(experiment, owner, message)
        ready(owner)
        assert tx.call_count == stage.call_count == 1
        assert execute.call_count == completed.report.payload.actual_teacher_executions
        assert execute.call_count == (20 if message else 24)
        legacy.assert_not_called()
        generate.assert_not_called()
        candidate.assert_not_called()
        loader.assert_not_called()
        tokenizer.assert_not_called()
    assert len(captured) == 2 and captured[0] is captured[1]
    rebuilt = scientific_ablation_writer_source_preflight_v1(
        completed.source, expected_sha256=completed.source.sha256
    )
    assert rebuilt == captured[0]
    assert rebuilt is not captured[0]
    assert rebuilt.observed_budget is not captured[0].observed_budget
    body = scientific_ablation_writer_source_bytes_v1(
        completed.source, expected_sha256=completed.source.sha256
    )
    assert (
        read_scientific_ablation_writer_source_v1(
            body, expected_sha256=completed.source.sha256, maximum_source_bytes=len(body)
        )
        == completed.source
    )
    content = scientific_ablation_writer_report_bytes_v1(
        completed.report, preflight=rebuilt, runtime=completed.runtime
    )
    assert (
        read_scientific_ablation_writer_report_v1(
            content, preflight=rebuilt, runtime=completed.runtime
        )
        == completed.report
    )
    assert {field.name for field in fields(completed)} == {"report", "source", "runtime"}
    assert not hasattr(completed, "__dict__")
    for field in fields(completed):
        with pytest.raises(FrozenInstanceError):
            setattr(completed, field.name, getattr(completed, field.name))
    assert all(child.training is mode for child, mode in modes)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, run._runtime)
    assert torch.get_default_dtype() is dtype


@pytest.mark.parametrize("message", (False, True))
@pytest.mark.parametrize("point", ("source-builder", "report-builder", "late-cleanup"))
@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
def test_failed_body_or_late_restore_never_constructs_completed_result(
    experiment: Experiment, message: bool, point: str, signal: type[BaseException]
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    run = context._run
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    failure = signal("completed writer injection: " + point)
    original_restore = _restore_caller

    def restore(*args: Any, **kwargs: Any) -> None:
        original_restore(*args, **kwargs)
        raise failure

    target = (
        "build_scientific_ablation_writer_source_v1"
        if point == "source-builder"
        else "build_scientific_ablation_writer_report_v1"
    )
    replacement = (
        patch.object(controls, "_restore_caller", new=restore)
        if point == "late-cleanup"
        else patch.object(service, target, side_effect=failure)
    )
    with (
        patch.object(service, "ScientificAblationWorkspaceWriterCompletedV1") as workspace,
        patch.object(service, "ScientificAblationMessageWriterCompletedV1") as msg,
        replacement,
        pytest.raises(
            controls.IrisScientificAblationCausalControlsError if signal is ValueError else signal
        ) as caught,
    ):
        _measure(experiment, owner, message)
    workspace.assert_not_called()
    msg.assert_not_called()
    assert (
        (caught.value.__cause__ is failure) if signal is ValueError else (caught.value is failure)
    )
    assert owner._state == context._state == context._request._state == "poisoned"
    assert run._poisoned and run._optimizer._poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, run._runtime)


@pytest.mark.parametrize("message", (False, True))
def test_coherent_replacement_source_cannot_choose_original_expectation(
    experiment: Experiment, message: bool
) -> None:
    owner = _heldout(experiment)

    def foreign(pf: Any) -> Any:
        original = build_scientific_ablation_writer_source_v1(pf)
        metadata = original.payload.source_metadata
        replacement = build_scientific_ablation_causal_report_byte_preflight_v1(
            metadata.observed_budget,
            evaluation_plan=metadata.evaluation_plan,
            evaluation=metadata.evaluation,
            evaluation_complete=metadata.evaluation_complete,
            run_seed=metadata.run_seed,
            ordinary_teacher_graph_sha256=metadata.ordinary_teacher_graph_sha256,
            controlled_teacher_graph_sha256=metadata.controlled_teacher_graph_sha256,
            sidecar_sha256=metadata.sidecar_sha256,
            maximum_report_bytes=pf.maximum_report_bytes - 1,
        )
        result = build_scientific_ablation_writer_source_v1(replacement)
        assert result.sha256 != original.sha256
        return result

    with (
        patch.object(service, "build_scientific_ablation_writer_source_v1", new=foreign),
        patch.object(service, "build_scientific_ablation_writer_report_v1") as builder,
        pytest.raises(controls.IrisScientificAblationCausalControlsError) as caught,
    ):
        _measure(experiment, owner, message)
    builder.assert_not_called()
    assert "original native reservation" in str(caught.value.__cause__)
    assert owner._state == "poisoned" and not owner._guard.context._run._busy


@pytest.mark.parametrize("message", (False, True))
def test_portable_source_mutation_during_report_build_is_late_denied(
    experiment: Experiment, message: bool
) -> None:
    owner = _heldout(experiment)
    retained: list[Any] = []
    original_report = build_scientific_ablation_writer_report_v1

    def source(*args: Any, **kwargs: Any) -> Any:
        result = build_scientific_ablation_writer_source_v1(*args, **kwargs)
        retained.append(result)
        return result

    def report(*args: Any, **kwargs: Any) -> Any:
        result = original_report(*args, **kwargs)
        object.__setattr__(retained[0].payload, "maximum_report_bytes", 1)
        return result

    with (
        patch.object(service, "build_scientific_ablation_writer_source_v1", new=source),
        patch.object(service, "build_scientific_ablation_writer_report_v1", new=report),
        pytest.raises(controls.IrisScientificAblationCausalControlsError),
    ):
        _measure(experiment, owner, message)
    assert owner._state == "poisoned" and not owner._guard.context._run._busy
