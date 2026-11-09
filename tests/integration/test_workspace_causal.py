"""Real random-Qwen causal reruns, including an actual disconnected-module null."""

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application.scientific_optimization import _same_runtime
from iris.application.workspace_causal import measure_workspace_causal_v1
from iris.utils.hashing import module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.causal import (
    WorkspaceCausalCaptureV1,
    WorkspaceCausalPolicyV1,
    summarize_workspace_causal_v1,
)
from tests.integration.test_intervened_teacher import prepared
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def test_real_next_cycle_effects_permutation_control_and_unchanged_runtime() -> None:
    model, _, lm = _build()
    torch.autograd.backward(model.training_forward(_rows()[0], terminal_token_id=1).losses.total)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    provenance = model.teacher.checkpoint_provenance
    model.eval()
    assert model.teacher.evaluation_provenance == provenance
    runtime = capture_runtime_state(torch.device("cpu"))
    identity = module_state_sha256((("model", model), ("qwen", lm)))
    with patch.object(
        model.teacher,
        "evaluation_forward_intervened",
        wraps=model.teacher.evaluation_forward_intervened,
    ) as calls:
        result = measure_workspace_causal_v1(
            model.teacher,
            prepared(model),
            example_id="example-1",
            seed=42,
            policy=WorkspaceCausalPolicyV1(shuffle_seed=9),
        )
    assert calls.call_count == result.executions == 7
    assert len(result.effects) == 12  # 2 cycles * 2 actual writers * 3 other modules.
    assert {e.write_cycle for e in result.effects} == {1, 2}
    assert all(e.source_module != e.target_module for e in result.effects)
    assert any(e.ablated_distance > 1e-5 for e in result.effects)
    assert all(e.shuffled_distance < 1e-6 for e in result.effects)
    assert WorkspaceCausalCaptureV1.model_validate_json(result.model_dump_json()) == result
    assert identity == module_state_sha256((("model", model), ("qwen", lm)))
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    assert all(not module.training for module in model.modules())
    for name, parameter in model.named_parameters():
        expected = gradients[name]
        assert (expected is None and parameter.grad is None) or (
            expected is not None
            and parameter.grad is not None
            and torch.equal(expected, parameter.grad)
        )
    assert all(
        not pair.significant_positive_effect
        for pair in summarize_workspace_causal_v1((result,)).pairs
    )
    # Targets affect only the subsequent LM pass, never the captured module effects/seeds.
    repeated = measure_workspace_causal_v1(
        model.teacher,
        prepared(model, "another answer"),
        example_id="example-1",
        seed=42,
        policy=WorkspaceCausalPolicyV1(shuffle_seed=9),
    )
    assert result == repeated


def test_actual_disconnected_module_weights_have_no_next_cycle_effect() -> None:
    model, _, _ = _build()
    with torch.no_grad():
        # Real parameter change, not mocked states: each GRU now receives a constant
        # projected input, so workspace cannot affect its next recurrent state.
        for cell in model.teacher.cognitive_modules.module_cells:
            projection = cell.input_projection
            assert isinstance(projection, torch.nn.Linear)
            projection.weight.zero_()
    model.eval()
    result = measure_workspace_causal_v1(
        model.teacher,
        prepared(model),
        example_id="disconnected",
        seed=42,
        policy=WorkspaceCausalPolicyV1(shuffle_seed=9),
    )
    assert all(e.ablated_distance == e.shuffled_distance == 0 for e in result.effects)
    summary = summarize_workspace_causal_v1((result,))
    assert all(e.sign_test_p == 1 and not e.significant_positive_effect for e in summary.pairs)


@pytest.mark.parametrize("change", [{"maximum_executions": 6}, {"maximum_trace_scalars": 1}])
def test_budget_rejected_before_first_model_pass(change: dict[str, int]) -> None:
    model, _, _ = _build()
    model.eval()
    with patch.object(
        model.teacher, "evaluation_forward_intervened", side_effect=AssertionError("unexpected")
    ) as forward:
        with pytest.raises(ValueError, match="budget"):
            measure_workspace_causal_v1(
                model.teacher,
                prepared(model),
                example_id="preflight",
                seed=42,
                policy=WorkspaceCausalPolicyV1.model_validate({"shuffle_seed": 9, **change}),
            )
        forward.assert_not_called()


def test_training_mode_and_single_cycle_are_rejected() -> None:
    model, _, _ = _build(cycles=1)
    policy = WorkspaceCausalPolicyV1(shuffle_seed=9)
    with pytest.raises(ValueError, match="eval mode"):
        measure_workspace_causal_v1(
            model.teacher, prepared(model), example_id="bad", seed=42, policy=policy
        )
    model.eval()
    with pytest.raises(ValueError, match="no other"):
        measure_workspace_causal_v1(
            model.teacher, prepared(model), example_id="bad", seed=42, policy=policy
        )


def test_wrong_receipt_is_rejected_without_returning_partial_capture() -> None:
    model, _, _ = _build()
    model.eval()
    original = model.teacher.evaluation_forward_intervened

    def corrupted(*args: Any, **kwargs: Any) -> Any:
        return replace(original(*args, **kwargs), prompt_sha256="f" * 64)

    with patch.object(
        model.teacher, "evaluation_forward_intervened", side_effect=corrupted
    ) as forward:
        with pytest.raises(ValueError, match="receipt"):
            measure_workspace_causal_v1(
                model.teacher,
                prepared(model),
                example_id="bad",
                seed=42,
                policy=WorkspaceCausalPolicyV1(shuffle_seed=9),
            )
        assert forward.call_count == 1
