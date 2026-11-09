"""Real CPU/meta allocation contexts and terminal signal-owned sessions; no CUDA proof."""

from typing import Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.application.scientific_optimization import _same_runtime
from iris.cognitive import ModuleInputRouter
from iris.cognitive.outputs import GoalEncoderOutput
from iris.utils.seeds import capture_runtime_state
from iris.workspace.outputs import WorkspaceReaderOutput


def _session(routing: Literal["learned", "random", "uniform"]) -> CognitiveAblationSessionV1:
    return CognitiveAblationSessionV1(
        CognitiveAblationPolicyV1(
            routing=routing, routing_seed=42 if routing == "random" else None
        ),
        prompt_sha256="1" * 64,
        num_modules=3,
        num_slots=2,
        state_dim=2,
        cycles=3,
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_actual_cpu_routes_ignore_default_device_preserve_rng_and_complete_session(
    routing: Literal["learned", "random", "uniform"],
    default: str,
    inference: bool,
    dtype: torch.dtype,
) -> None:
    with torch.device("cpu"):
        router = ModuleInputRouter(state_dim=2, num_modules=3).to(dtype=dtype)
    context = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype, device="cpu")
    mask = torch.ones((1, 2), dtype=torch.int64, device="cpu")
    goal = GoalEncoderOutput(
        goal_state=torch.ones((1, 2), dtype=dtype, device="cpu"),
        attention_weights=torch.full((1, 2), 0.5, dtype=dtype, device="cpu"),
    )
    reads = WorkspaceReaderOutput(
        workspace_reads=torch.ones((1, 3, 2), dtype=dtype, device="cpu"),
        attention_weights=torch.full((1, 3, 2), 0.5, dtype=dtype, device="cpu"),
    )
    reference_session = _session(routing)
    reference_session.after_goal(goal)
    with torch.device("cpu"), torch.inference_mode(inference):
        reference = reference_session.route(router, context, mask)
    session = _session(routing)
    session.after_goal(goal)
    caller = capture_runtime_state(torch.device("cpu"))
    caller_default = str(torch.get_default_device())
    with torch.device(default), torch.inference_mode(inference):
        actual = session.route(router, context, mask)
        for cycle in range(1, 4):
            session.after_read(reads, cycle)
        session.finish()
    assert actual.module_inputs.device.type == actual.attention_weights.device.type == "cpu"
    assert torch.equal(actual.module_inputs, reference.module_inputs)
    assert torch.equal(actual.attention_weights, reference.attention_weights)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert str(torch.get_default_device()) == caller_default
    assert session._finished and session._phase == 6 and not session._failed


@pytest.mark.parametrize(
    "failure",
    (
        KeyboardInterrupt("session"),
        SystemExit(17),
        BaseExceptionGroup("session", [ValueError("other"), KeyboardInterrupt("group")]),
    ),
)
def test_first_signal_closes_session_and_preserves_the_exact_primary(
    failure: BaseException,
) -> None:
    session = _session("random")
    goal = GoalEncoderOutput(
        goal_state=torch.ones((1, 2), device="cpu"),
        attention_weights=torch.full((1, 2), 0.5, device="cpu"),
    )
    with (
        patch.object(session, "_floating", side_effect=failure),
        pytest.raises(type(failure)) as caught,
    ):
        session.after_goal(goal)
    assert caught.value is failure and session._failed and session._phase == 0
    with pytest.raises(ValueError, match="closed"):
        session.after_goal(goal)
