"""Genuine tiny-Qwen message writes, transplants and physical pathway nulls."""

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application.scientific_optimization import _same_runtime
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.workspace_message_causal import (
    PreparedMessageExampleV1,
    measure_workspace_message_causal_v1,
)
from iris.interventions.messages import WorkspaceMessagePlanV1
from iris.training.messages import IrisMessageTeacherOutputV1
from iris.training.model import IrisTeacherForcingError
from iris.utils.hashing import module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.message_causal import (
    WorkspaceMessageCausalCaptureV1,
    WorkspaceMessageCausalPolicyV1,
)
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_scientific_teacher import _build, _rows
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration


def prepared(model: Any, *, target: str | None = None) -> tuple[PreparedMessageExampleV1, ...]:
    return tuple(
        PreparedMessageExampleV1(
            example_id=f"example-{n}",
            stratum="same-task-level",
            example=prepare_teacher_forcing_input(
                backbone=model.teacher.backbone,
                prompt=row.prompt,
                target_text=row.expected_response if target is None else target,
                terminal_token_id=1,
            ),
        )
        for n, row in enumerate(_rows()[:2])
    )


def test_empty_message_capture_matches_ordinary_teacher_and_restores_runtime() -> None:
    model, _, lm = _build(slots=2)
    torch.autograd.backward(model.training_forward(_rows()[0], terminal_token_id=1).losses.total)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    model.eval()
    example = prepared(model)[0].example
    options = WorkspaceTraceOptionsV1(
        mode="full", example_id="example-0", checkpoint_id="selected", seed=42
    )
    identity = module_state_sha256((("model", model), ("qwen", lm)))
    runtime = capture_runtime_state(torch.device("cpu"))
    ordinary = model.teacher.evaluation_forward(example, diagnostics=options)
    captured = model.teacher.evaluation_forward_messages(
        example, plan=WorkspaceMessagePlanV1(), diagnostics=options
    )
    assert type(captured) is IrisMessageTeacherOutputV1
    assert torch.equal(captured.loss, ordinary.loss)
    assert captured.supervised_token_count == ordinary.supervised_token_count
    assert ordinary.diagnostic_trace is not None
    assert captured.diagnostic_trace.cycles == ordinary.diagnostic_trace.cycles
    assert len(captured.messages) == 12
    assert {(m.cycle, m.module) for m in captured.messages} == {
        (c, m) for c in (1, 2, 3) for m in range(4)
    }
    for a, b in zip(captured.workspace_trace, ordinary.workspace_trace, strict=True):
        assert_same(a, b)
    assert identity == module_state_sha256((("model", model), ("qwen", lm)))
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    for name, p in model.named_parameters():
        expected = gradients[name]
        assert (expected is None and p.grad is None) or (
            expected is not None and p.grad is not None and torch.equal(expected, p.grad)
        )


def test_reinserting_own_actual_message_leaves_all_cycles_and_loss_unchanged() -> None:
    model, _, _ = _build(slots=2)
    model.eval()
    example = prepared(model)[0].example
    options = WorkspaceTraceOptionsV1(
        mode="full", example_id="example-0", checkpoint_id="selected", seed=42
    )
    original = model.teacher.evaluation_forward_messages(
        example, plan=WorkspaceMessagePlanV1(), diagnostics=options
    )
    donor = original.messages[0]
    reinserted = model.teacher.evaluation_forward_messages(
        example, plan=WorkspaceMessagePlanV1(donor=donor), diagnostics=options
    )
    assert torch.equal(original.loss, reinserted.loss)
    assert original.diagnostic_trace.cycles == reinserted.diagnostic_trace.cycles
    assert original.messages == reinserted.messages


def test_real_pair_experiment_measures_all_writes_and_changes_actual_content() -> None:
    model, _, lm = _build(slots=2)
    model.eval()
    examples = prepared(model)
    policy = WorkspaceMessageCausalPolicyV1(shuffle_seed=42)
    identity = module_state_sha256((("model", model), ("qwen", lm)))
    rng = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            model.teacher,
            "evaluation_forward_messages",
            wraps=model.teacher.evaluation_forward_messages,
        ) as message_calls,
        patch.object(
            model.teacher,
            "evaluation_forward_intervened",
            wraps=model.teacher.evaluation_forward_intervened,
        ) as block_calls,
    ):
        result = measure_workspace_message_causal_v1(
            model.teacher, examples, seed=1337, policy=policy
        )
    assert result.executions == message_calls.call_count + block_calls.call_count == 18
    assert message_calls.call_count == 10 and block_calls.call_count == 8
    assert len(result.effects) == 24
    assert any(e.genuine_write_distance > 1e-6 for e in result.effects)
    assert any(e.donor_write_distance > 1e-6 for e in result.effects)
    assert any(e.overwrote_existing_record for e in result.effects)
    assert all(e.example_id != e.donor_example_id for e in result.effects)
    assert len(result.donor_pairs) == 1 and all(
        s.independent_donor_pairs <= 1 for s in result.statistics
    )
    assert WorkspaceMessageCausalCaptureV1.model_validate_json(result.model_dump_json()) == result
    assert identity == module_state_sha256((("model", model), ("qwen", lm)))
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert all(not m.training for m in model.modules())
    assert not any(s.significant_positive_effect for s in result.statistics)
    repeated = measure_workspace_message_causal_v1(
        model.teacher,
        tuple(reversed(prepared(model, target="another answer"))),
        seed=1337,
        policy=policy,
    )
    assert repeated == result
    serialized = result.model_dump_json()
    assert (
        '"values"' not in serialized
        and '"target"' not in serialized
        and '"prompt"' not in serialized
    )


def test_genuinely_disconnected_reader_path_has_zero_write_and_donor_effects() -> None:
    model, _, _ = _build(slots=2)
    with torch.no_grad():
        for cell in model.teacher.cognitive_modules.module_cells:
            assert isinstance(cell.input_projection, torch.nn.Linear)
            cell.input_projection.weight.zero_()
    model.eval()
    result = measure_workspace_message_causal_v1(
        model.teacher,
        prepared(model),
        seed=42,
        policy=WorkspaceMessageCausalPolicyV1(shuffle_seed=42),
    )
    assert all(e.genuine_write_distance == e.donor_write_distance == 0 for e in result.effects)
    assert all(s.sign_test_p == 1 for s in result.statistics)


def test_constant_real_proposal_content_cannot_look_input_specific() -> None:
    model, _, _ = _build(slots=2)
    with torch.no_grad():
        model.teacher.proposal_head.content_head.weight.zero_()
        model.teacher.workspace_writer.normalization.bias.copy_(
            torch.arange(8, dtype=torch.float32) * 0.2
        )
    model.eval()
    result = measure_workspace_message_causal_v1(
        model.teacher,
        prepared(model),
        seed=42,
        policy=WorkspaceMessageCausalPolicyV1(shuffle_seed=42),
    )
    assert all(e.genuine_write_distance == e.donor_write_distance for e in result.effects)
    assert all(s.sign_test_p == 1 and not s.significant_positive_effect for s in result.statistics)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_executions": 17},
        {"maximum_trace_scalars": 1},
        {"maximum_message_scalars": 1},
        {"maximum_total_trace_scalars": 1},
    ],
)
def test_whole_experiment_budget_rejects_before_any_model_pass(changes: dict[str, int]) -> None:
    model, _, _ = _build(slots=2)
    model.eval()
    policy = WorkspaceMessageCausalPolicyV1.model_validate({"shuffle_seed": 42, **changes})
    with (
        patch.object(model.teacher, "evaluation_forward_messages") as forward,
        pytest.raises(ValueError, match="budget"),
    ):
        measure_workspace_message_causal_v1(model.teacher, prepared(model), seed=42, policy=policy)
    forward.assert_not_called()


def test_wrong_donor_model_is_rejected_before_forward() -> None:
    model, _, lm = _build(slots=2)
    model.eval()
    example = prepared(model)[0].example
    options = WorkspaceTraceOptionsV1(
        mode="full", example_id="example-0", checkpoint_id="selected", seed=42
    )
    original = model.teacher.evaluation_forward_messages(
        example, plan=WorkspaceMessagePlanV1(), diagnostics=options
    )
    donor = original.messages[0].model_copy(update={"sidecar_state_sha256": "0" * 64})
    with patch.object(lm, "forward") as forward, pytest.raises(IrisTeacherForcingError):
        model.teacher.evaluation_forward_messages(
            example, plan=WorkspaceMessagePlanV1(donor=donor), diagnostics=options
        )
    forward.assert_not_called()


def test_wrong_message_receipt_aborts_without_partial_capture() -> None:
    model, _, _ = _build(slots=2)
    model.eval()
    actual = model.teacher.evaluation_forward_messages

    def wrong(*args: Any, **kwargs: Any) -> Any:
        return replace(actual(*args, **kwargs), prompt_sha256="0" * 64)

    with (
        patch.object(model.teacher, "evaluation_forward_messages", side_effect=wrong),
        pytest.raises(ValueError),
    ):
        measure_workspace_message_causal_v1(
            model.teacher,
            prepared(model),
            seed=42,
            policy=WorkspaceMessageCausalPolicyV1(shuffle_seed=42),
        )
