"""Real random-Qwen execution: metadata is observational, never a replacement graph."""

from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import build_request_orchestrator
from iris.application.intervention_receipt import IrisIntervenedRequestExecutionOutput
from iris.application.request import IrisRequestExecutionOutput
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import IrisScientificTeacherError
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.intervention import IrisIntervenedTeacherOutputV1
from iris.training.outputs import IrisTeacherForcedOutput
from iris.utils.hashing import module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceRecorderV1, summarize_workspace_traces_v1
from tests.integration.test_intervened_request import plan
from tests.integration.test_intervened_teacher import OPERATIONS, prepared
from tests.integration.test_scientific_teacher import _build, _rows
from tests.unit.test_workspace_trace import options

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "specification",
    [None, {}, *OPERATIONS],
    ids=lambda spec: "baseline" if spec is None else spec.get("kind", "empty"),
)
def test_all_phases_match_real_teacher_and_request_without_changing_results(
    specification: dict[str, Any] | None,
) -> None:
    model, _, lm = _build()
    teacher = model.teacher
    provenance = teacher.checkpoint_provenance
    with torch.random.fork_rng(devices=[]):
        request = build_request_orchestrator(
            backbone=teacher.backbone,
            cognitive_config=CognitiveConfig.model_validate_json(
                provenance.cognitive_configuration_json
            ),
            workspace_config=WorkspaceConfig.model_validate_json(
                provenance.workspace_configuration_json
            ),
            stop_token_ids=tuple(range(teacher.backbone.vocabulary_size)),
            sampling=TemperatureTopPSamplingConfig(),
        )
    request.restore_teacher_sidecar_state_v1(dict(teacher.state_dict()))
    model.eval()
    example = prepared(model)
    before = module_state_sha256((("model", model), ("lm", lm)))
    runtime = capture_runtime_state(torch.device("cpu"))
    plain: IrisTeacherForcedOutput | IrisIntervenedTeacherOutputV1
    output: IrisTeacherForcedOutput | IrisIntervenedTeacherOutputV1
    generated: IrisRequestExecutionOutput | IrisIntervenedRequestExecutionOutput
    ordinary: IrisRequestExecutionOutput | IrisIntervenedRequestExecutionOutput
    if specification is None:
        plain = teacher.evaluation_forward(example)
        output = teacher.evaluation_forward(example, diagnostics=options("full"))
        generated = request.execute(example.prompt, strategy="greedy", diagnostics=options("full"))
        ordinary = request.execute(example.prompt, strategy="greedy")
    else:
        selected = plan() if not specification else plan(specification)
        plain = teacher.evaluation_forward_intervened(example, plan=selected)
        output = teacher.evaluation_forward_intervened(
            example, plan=selected, diagnostics=options("full")
        )
        generated = request.execute_intervened(
            example.prompt, plan=selected, strategy="greedy", diagnostics=options("full")
        )
        ordinary = request.execute_intervened(example.prompt, plan=selected, strategy="greedy")
    assert plain.diagnostic_trace is None and ordinary.diagnostic_trace is None
    assert torch.equal(plain.loss, output.loss)
    assert generated.generation.text == ordinary.generation.text
    assert output.diagnostic_trace == generated.diagnostic_trace
    trace = output.diagnostic_trace
    assert trace is not None and len(trace.cycles) == 3
    assert trace.cycles[0].read_attention == ((0.0,) * 4,) * 4
    assert before == module_state_sha256((("model", model), ("lm", lm)))
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    metrics = summarize_workspace_traces_v1((trace,))
    assert metrics.allocated_writes == 6 and sum(metrics.module_win_counts) == 6
    if specification and specification["kind"] == "block_module_read":
        assert all(row.read_attention[1] == (0.0,) * 4 for row in trace.cycles)
    if specification and specification["kind"] == "zero_workspace":
        assert metrics.occupied_slot_observations == metrics.active_reads == 0
        assert metrics.mean_ttl is None and metrics.mean_slot_cosine_similarity is None


def test_diagnostics_leave_training_gradients_identical_and_preserve_auxiliary_graph() -> None:
    model, _, _ = _build()
    example = prepared(model)
    plain = model.teacher.training_forward_with_trace(example)
    torch.autograd.backward(plain.teacher.loss)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    output = model.teacher.training_forward_with_trace(example, diagnostics=options("metadata"))
    assert torch.equal(plain.teacher.loss, output.teacher.loss)
    assert output.teacher.diagnostic_trace is not None
    assert all(row.full is None for row in output.teacher.diagnostic_trace.cycles)
    assert all(c.module_probabilities.requires_grad for c in output.cycles)
    torch.autograd.backward(output.teacher.loss)
    for name, parameter in model.named_parameters():
        expected = gradients[name]
        assert (expected is None and parameter.grad is None) or (
            expected is not None
            and parameter.grad is not None
            and torch.equal(expected, parameter.grad)
        )


def test_none_is_zero_recording_work_and_oversize_fails_before_backbone() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    with patch.object(WorkspaceTraceRecorderV1, "record", side_effect=AssertionError("unexpected")):
        assert (
            model.teacher.evaluation_forward(example, diagnostics=options("none")).diagnostic_trace
            is None
        )
    with patch.object(
        model.teacher.backbone, "forward_prompt", side_effect=AssertionError("unexpected")
    ) as forward:
        with pytest.raises(ValueError, match="budget"):
            model.teacher.evaluation_forward(
                example, diagnostics=options("full", maximum_scalars=1)
            )
        forward.assert_not_called()


def test_five_term_scientific_training_diagnostics_preserve_full_objective_and_gradients() -> None:
    model, _, _ = _build()
    row = _rows()[0]
    plain = model.training_forward(row, terminal_token_id=1)
    torch.autograd.backward(plain.losses.total)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    diagnostic = options("full", example_id=row.id)
    output = model.training_forward(row, terminal_token_id=1, diagnostics=diagnostic)
    assert torch.equal(output.losses.total, plain.losses.total)
    torch.autograd.backward(output.losses.total)
    for name, value in model.named_parameters():
        expected = gradients[name]
        assert (expected is None and value.grad is None) or (
            expected is not None and value.grad is not None and torch.equal(expected, value.grad)
        )
    model.eval()
    ordinary = model.evaluation_forward(row, terminal_token_id=1, diagnostics=diagnostic)
    changed = model.evaluation_forward_intervened(
        row, terminal_token_id=1, plan=plan(), diagnostics=diagnostic
    )
    assert ordinary.trace.teacher.diagnostic_trace is not None
    assert changed.teacher.diagnostic_trace is not None
    assert ordinary.trace.teacher.diagnostic_trace.cycles == changed.teacher.diagnostic_trace.cycles
    with pytest.raises(IrisScientificTeacherError):
        model.evaluation_forward(row, terminal_token_id=1, diagnostics=options())


def test_metadata_ownership_and_target_noninterference() -> None:
    model, _, _ = _build()
    model.eval()
    first = model.teacher.evaluation_forward(prepared(model), diagnostics=options("full"))
    second = model.teacher.evaluation_forward(
        prepared(model, "different target"), diagnostics=options("full")
    )
    assert first.diagnostic_trace == second.diagnostic_trace
    trace = first.diagnostic_trace
    assert trace is not None
    encoded = trace.model_dump_json()
    for state in first.workspace_trace:
        state.slots.fill_(123)
    assert trace.model_dump_json() == encoded
