"""Primary cleanup signals and uncertain partial controller commits on actual owners."""

from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_lifecycle as module
from iris.application.scientific_ablation_validation import _snapshot
from iris.application.scientific_optimization import _same_runtime
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_lifecycle import _controller
from tests.integration.test_scientific_ablation_run import _run
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration


def _signals() -> tuple[BaseException, ...]:
    return (
        KeyboardInterrupt("first signal"),
        SystemExit(83),
        BaseExceptionGroup("first group", [ValueError("ordinary"), KeyboardInterrupt("signal")]),
    )


@pytest.mark.parametrize("failure", _signals())
@pytest.mark.parametrize("secondary_restore_error", (False, True))
def test_attachment_cleanup_signal_keeps_signal_type_and_releases_run(
    failure: BaseException,
    secondary_restore_error: bool,
) -> None:
    run = _run(policy=_policy(minimum_target_tokens=1))
    caller = capture_runtime_state(torch.device("cpu"))
    calls = 0

    def snapshot(current: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise failure
        return _snapshot(current)

    def restore(snapshot: Any) -> None:
        restore_runtime_state(snapshot)
        if secondary_restore_error:
            raise ValueError("secondary acknowledgement lost")

    from iris.artifacts.scientific_ablation_lifecycle_v1 import (
        IrisScientificAblationLifecyclePolicyV1,
    )
    from tests.integration.test_scientific_lifecycle import _validation

    with (
        patch.object(module, "_snapshot", side_effect=snapshot),
        patch.object(module, "restore_runtime_state", side_effect=restore),
        pytest.raises(type(failure)) as caught,
    ):
        module.IrisScientificAblationLifecycleV1(
            run=run,
            validation=_validation(),
            policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
        )
    assert caught.value is failure
    assert calls == 3 and run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("failure", _signals())
def test_first_cleanup_signal_wins_over_secondary_caller_restore_error(
    failure: BaseException,
) -> None:
    controller = _controller()
    caller = capture_runtime_state(torch.device("cpu"))
    calls = 0

    def snapshot(current: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return _snapshot(current)

    def uncertain(snapshot: Any) -> None:
        restore_runtime_state(snapshot)
        raise ValueError("secondary acknowledgement lost")

    with (
        patch.object(module, "_snapshot", side_effect=snapshot),
        patch.object(module, "restore_runtime_state", side_effect=uncertain),
        pytest.raises(type(failure)) as caught,
    ):
        _ = controller.state
    assert caught.value is failure
    assert controller.poisoned and controller._run._poisoned and controller._run._optimizer.poisoned
    assert not controller._busy and not controller._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("failure", _signals())
def test_primary_phase_signal_wins_over_both_cleanup_failures(failure: BaseException) -> None:
    controller = _controller()
    controller.advance()
    caller = capture_runtime_state(torch.device("cpu"))
    calls = 0

    def snapshot(current: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:  # Operation entry, explicit delegation entry, then cleanup.
            raise KeyboardInterrupt("secondary cleanup signal")
        return _snapshot(current)

    def uncertain(snapshot: Any) -> None:
        restore_runtime_state(snapshot)
        raise ValueError("tertiary acknowledgement lost")

    with (
        patch.object(module, "_snapshot", side_effect=snapshot),
        patch.object(module, "restore_runtime_state", side_effect=uncertain),
        patch.object(module, "_evaluate_scientific_ablation_validation_v1", side_effect=failure),
        pytest.raises(type(failure)) as caught,
    ):
        controller.advance()
    assert caught.value is failure
    assert controller.poisoned and controller._run._optimizer.poisoned
    assert controller._run.completed_steps == 1 and controller._state.validation_count == 0
    assert not controller._busy and not controller._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("failure", (ValueError("partial"), KeyboardInterrupt("partial")))
def test_unknown_partial_phase_commit_poisons_all_and_preserves_signal(
    failure: BaseException,
) -> None:
    controller = _controller()
    caller = capture_runtime_state(torch.device("cpu"))

    def partial(state: Any, best: Any) -> None:
        controller._state = state
        raise failure

    expected = (
        module.IrisScientificAblationLifecycleError
        if isinstance(failure, Exception)
        else type(failure)
    )
    with patch.object(controller, "_commit", side_effect=partial), pytest.raises(expected):
        controller.advance()
    assert controller.poisoned and controller._run._poisoned and controller._run._optimizer.poisoned
    assert controller._run.completed_steps == 1
    assert not controller._busy and not controller._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
