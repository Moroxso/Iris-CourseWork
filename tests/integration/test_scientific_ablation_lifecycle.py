"""Actual mode-bound best selection, phase cadence, caller state and failure ownership."""

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import iris.application as application
import iris.artifacts as artifacts
from iris.application import scientific_ablation_lifecycle as module
from iris.application.scientific_ablation_checkpoint import (
    build_scientific_ablation_checkpoint_v1 as build,
)
from iris.application.scientific_ablation_validation import (
    IrisScientificAblationValidationV1,
    _evaluate_scientific_ablation_validation_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1 as Policy,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    ScientificAblationLifecycleStateV1 as State,
)
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_run import _run, _stochastic
from tests.integration.test_scientific_checkpoint import _run as _old_run
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration
Controller = module.IrisScientificAblationLifecycleV1
Error = module.IrisScientificAblationLifecycleError
Boundary = module.IrisScientificAblationLifecycleBoundaryV1


def _controller(
    routing: Any = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    maximum: int = 5,
    cycles: int = 3,
    **updates: Any,
) -> module.IrisScientificAblationLifecycleV1:
    return Controller(
        run=_run(
            routing,
            goal=goal,
            broadcast=broadcast,
            cycles=cycles,
            policy=_policy(minimum_target_tokens=1, max_optimizer_steps=maximum),
        ),
        validation=_validation(),
        policy=Policy.model_validate(
            {
                "purpose": "smoke",
                "validate_every_steps": 1,
                "checkpoint_every_steps": 2,
                "early_stopping_patience": 2,
                "minimum_improvement": 1e6,
                **updates,
            }
        ),
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_c3_modes_best_patience_current_boundary_and_private_rng(
    routing: Any,
    goal: str,
    broadcast: str,
) -> None:
    controller = _controller(routing, goal=goal, broadcast=broadcast)
    caller = capture_runtime_state(torch.device("cpu"))
    boundaries = []
    first_best = None
    while controller.state.phase != "stopped":
        phase = controller.state.phase
        with _stochastic(controller._run):
            result = controller.advance()
        if phase == "validation":
            assert type(result) is IrisScientificAblationValidationV1
            assert result.ablation_policy_sha256 == controller._run._ablation_identity
            if first_best is None:
                first_best = controller.best_checkpoint
        elif phase == "checkpoint":
            assert type(result) is Boundary
            boundaries.append(result)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    state = controller.state
    assert (state.step, state.validation_count, state.best_step) == (3, 3, 1)
    assert state.nonimproving_validations == 2 and state.stop_reason == "early_stopping"
    assert state.checkpoint_count == 2 and [item.state.step for item in boundaries] == [2, 3]
    assert controller.best_checkpoint == first_best
    assert boundaries[-1].current == build(controller._run)
    assert boundaries[-1].best == first_best and first_best is not None
    assert first_best.manifest.payload.completed_steps == 1
    assert first_best != boundaries[-1].current
    assert controller._run._runtime.python_rng_state[2] is not None
    assert all(child.training for child in controller._run._optimizer._model.modules())
    with pytest.raises(Error):
        controller.advance()
    assert not controller.poisoned


@pytest.mark.parametrize("cycles", (1, 3))
def test_terminal_off_cadence_and_checkpoints_before_first_validation(cycles: int) -> None:
    controller = _controller(cycles=cycles, validate_every_steps=3, checkpoint_every_steps=2)
    validation_steps, boundaries = [], []
    while controller.state.phase != "stopped":
        result = controller.advance()
        if type(result) is IrisScientificAblationValidationV1:
            validation_steps.append(result.at_optimizer_step)
        elif type(result) is Boundary:
            boundaries.append(result)
    assert validation_steps == [3, 5]
    assert [item.state.step for item in boundaries] == [2, 4, 5]
    assert boundaries[0].best is None
    assert controller.state.stop_reason == "step_budget"
    assert controller.state.checkpoint_count == 3


def test_source_patience_eight_uses_actual_nonimproving_validations() -> None:
    controller = _controller(maximum=12, checkpoint_every_steps=20, early_stopping_patience=8)
    while controller.state.phase != "stopped":
        controller.advance()
    assert controller.state.step == 9 and controller.state.validation_count == 9
    assert controller.state.best_step == 1 and controller.state.nonimproving_validations == 8
    assert controller.state.stop_reason == "early_stopping"


def test_best_and_state_exports_are_defensive_and_equal_current_shares_one_bundle() -> None:
    controller = _controller(checkpoint_every_steps=1)
    controller.advance()
    controller.advance()
    best = controller.best_checkpoint
    assert best is not None
    vars(best.manifest.payload)["not_a_field"] = True
    state = controller.state
    object.__setattr__(state, "step", 123)
    assert controller.state.step == 1
    assert controller.best_checkpoint is not best
    boundary = controller.advance()
    assert type(boundary) is Boundary and boundary.best is boundary.current
    vars(boundary.current.manifest.payload)["not_a_field"] = True
    assert controller.best_checkpoint is not None and not controller.poisoned


@pytest.mark.parametrize(
    "failure", (ValueError("PRIVATE"), KeyboardInterrupt("primary"), SystemExit(83))
)
@pytest.mark.parametrize("phase", ("validation", "checkpoint"))
def test_untouched_phase_failure_is_retryable_without_retraining(
    phase: str,
    failure: BaseException,
) -> None:
    controller = _controller(checkpoint_every_steps=1)
    controller.advance()
    if phase == "checkpoint":
        controller.advance()
    state = controller.state
    before = build(controller._run)
    caller = capture_runtime_state(torch.device("cpu"))
    expected = Error if isinstance(failure, Exception) else type(failure)
    target = (
        "_evaluate_scientific_ablation_validation_v1"
        if phase == "validation"
        else "build_scientific_ablation_checkpoint_v1"
    )
    with patch.object(module, target, side_effect=failure), pytest.raises(expected):
        controller.advance()
    assert not controller.poisoned and not controller._busy and not controller._run._busy
    assert controller.state == state and build(controller._run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    with patch.object(controller._run, "_step_authorized") as training:
        controller.advance()
    training.assert_not_called()


@pytest.mark.parametrize("kind", ("state", "policy", "validation", "train", "cursor", "moment"))
def test_external_owned_change_rejects_and_poisons_all_owners(kind: str) -> None:
    controller = _controller()
    controller.advance()
    if kind == "state":
        vars(controller._state)["not_a_field"] = True
    elif kind == "policy":
        vars(controller._policy)["not_a_field"] = True
    elif kind == "validation":
        controller._validation[0].hidden_facts["result_type"] = "changed"
    elif kind == "train":
        controller._run._rows[0].hidden_facts["result_type"] = "changed"
    elif kind == "cursor":
        controller._run._examples_seen += 1
        controller._run._commit_owner()
    else:
        moment = next(iter(controller._run._optimizer._optimizer.state.values()))
        moment["exp_avg"].data.reshape(-1)[0].add_(0.125)
    caller = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(Error):
        controller.advance()
    assert controller.poisoned and controller._run._poisoned and controller._run._optimizer.poisoned
    assert not controller._busy and not controller._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_single_live_controller_and_serialization_reentry_are_rejected() -> None:
    controller = _controller()
    with pytest.raises(Error):
        Controller(run=controller._run, validation=_validation(), policy=controller._policy)
    original = controller._run._optimizer._model.training_forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(Error):
            controller.advance()
        with pytest.raises(Error):
            _ = controller.state
        return original(*args, **kwargs)

    with patch.object(controller._run._optimizer._model, "training_forward", side_effect=forward):
        controller.advance()
    assert controller.state.step == 1 and not controller.poisoned


@pytest.mark.parametrize("kind", ("tokens", "step_bool", "sum_bool", "config", "mode"))
def test_untrusted_validation_result_cannot_select_best(kind: str) -> None:
    controller = _controller()
    controller.advance()
    original = _evaluate_scientific_ablation_validation_v1
    before = build(controller._run)

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        report = original(*args, **kwargs)
        if kind == "tokens":
            count = report.target_token_count + 1
            return replace(
                report, target_token_count=count, target_nll_per_token=report.target_nll_sum / count
            )
        if kind == "step_bool":
            return replace(report, at_optimizer_step=True)
        if kind == "sum_bool":
            return replace(report, target_nll_sum=False, target_nll_per_token=0.0)
        if kind == "config":
            return replace(report, run_configuration_sha256="0" * 64)
        return replace(report, ablation_policy_sha256="0" * 64)

    with (
        patch.object(module, "_evaluate_scientific_ablation_validation_v1", side_effect=corrupt),
        pytest.raises(Error),
    ):
        controller.advance()
    assert not controller.poisoned and controller.state.validation_count == 0
    assert controller.best_checkpoint is None and build(controller._run) == before


def test_canonical_exports_defaults_and_historical_run_are_distinct() -> None:
    for name in module.__all__:
        assert name in application.__all__ and getattr(application, name) is getattr(module, name)
    assert artifacts.IrisScientificAblationLifecyclePolicyV1 is Policy
    assert artifacts.ScientificAblationLifecycleStateV1 is State
    policy = Policy()
    assert (
        policy.validate_every_steps,
        policy.checkpoint_every_steps,
        policy.early_stopping_patience,
    ) == (500, 1000, 8)
    with pytest.raises(Error):
        Controller(run=_old_run(), validation=_validation(), policy=Policy(purpose="smoke"))  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ("policy_unknown", "wrong_purpose", "bad_validation"))
def test_invalid_attachment_preserves_run_and_allows_valid_constructor_retry(kind: str) -> None:
    run = _run(policy=_policy(minimum_target_tokens=1))
    before = build(run)
    policy = Policy(purpose="smoke")
    validation = _validation()
    if kind == "policy_unknown":
        policy = policy.model_copy(update={"not_a_field": True})
    elif kind == "wrong_purpose":
        policy = Policy()
    else:
        validation = run._rows[:1]
    with pytest.raises(Error):
        Controller(run=run, validation=validation, policy=policy)
    assert not run.poisoned and not run._busy and build(run) == before
    assert (
        Controller(run=run, validation=_validation(), policy=Policy(purpose="smoke")).state.step
        == 0
    )
