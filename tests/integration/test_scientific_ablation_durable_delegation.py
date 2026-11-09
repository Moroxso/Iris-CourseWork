"""Retained-run exclusion, explicit one-call delegation and attachment rollback."""

from pathlib import Path
from typing import Any
from unittest.mock import patch
from weakref import ReferenceType, ref

import pytest
import torch

from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_lifecycle as lifecycle
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application import scientific_ablation_validation as validation
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunError, _ScientificRunCore
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1,
)
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable
from tests.integration.test_scientific_ablation_lifecycle import _controller
from tests.integration.test_scientific_ablation_run import _assert_equal, _run, _stochastic
from tests.integration.test_scientific_checkpoint import _run as _old_run
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration


def _assert_external_operations_rejected(
    controller: lifecycle.IrisScientificAblationLifecycleV1,
) -> None:
    run = controller._run
    before = native.build_scientific_ablation_checkpoint_v1(run)
    state = controller.state
    caller = capture_runtime_state(torch.device("cpu"))
    attachment = run._controller_attachment
    with (
        patch.object(run._optimizer, "step") as training,
        patch.object(run._optimizer._model, "evaluation_forward") as evaluation,
        patch.object(native, "_verify") as verify,
        patch.object(native, "restore_runtime_state") as restore_caller,
        patch.object(validation, "restore_runtime_state") as evaluation_runtime,
    ):
        with pytest.raises(IrisScientificRunError):
            run.step()
        with pytest.raises(IrisScientificRunError):
            _ScientificRunCore.step(run)
        with pytest.raises(validation.IrisScientificAblationValidationError):
            validation.evaluate_scientific_ablation_validation_v1(run, controller._validation)
        with pytest.raises(native.IrisScientificAblationCheckpointError):
            native.restore_scientific_ablation_checkpoint_v1(before, run=run)
    training.assert_not_called()
    evaluation.assert_not_called()
    verify.assert_not_called()
    restore_caller.assert_not_called()
    evaluation_runtime.assert_not_called()
    assert controller.state == state
    assert native.build_scientific_ablation_checkpoint_v1(run) == before
    assert run._controller_attachment is attachment
    assert run._controller_attachment_guard is attachment
    assert controller._delegation is None
    assert not controller.poisoned and not controller._busy and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("durable", (False, True), ids=("memory", "disk"))
@pytest.mark.parametrize("phase", ("train", "validation", "checkpoint", "stopped"))
def test_attached_run_public_execution_is_excluded_at_every_idle_phase(
    tmp_path: Path, durable: bool, phase: str
) -> None:
    controller = (
        _durable(tmp_path, maximum=2, checkpoint=1)
        if durable
        else _controller(maximum=2, checkpoint_every_steps=1, early_stopping_patience=8)
    )
    while controller.state.phase != phase:
        controller.advance()
    _assert_external_operations_rejected(controller)


@pytest.mark.parametrize("terminal", (False, True), ids=("nonterminal", "terminal"))
@pytest.mark.parametrize("kind", ("ordinary", "signal", "group"))
def test_lost_ack_pending_excludes_retained_run_but_allows_snapshots_and_exact_retry(
    tmp_path: Path, terminal: bool, kind: str
) -> None:
    controller = _durable(tmp_path, maximum=2, checkpoint=2 if terminal else 1)
    while controller.state.phase != "checkpoint":
        controller.advance()
    failure: BaseException = (
        OSError("lost acknowledgement")
        if kind == "ordinary"
        else KeyboardInterrupt("lost acknowledgement")
        if kind == "signal"
        else BaseExceptionGroup("lost acknowledgement", [SystemExit(83)])
    )
    published: list[io.IrisScientificAblationLifecycleBundleV1] = []

    def lost_ack(bundle: io.IrisScientificAblationLifecycleBundleV1, *, root: Path) -> Path:
        published.append(bundle)
        io.write_scientific_ablation_lifecycle_checkpoint_v1(bundle, root=root)
        raise failure

    expected = (
        lifecycle.IrisScientificAblationLifecycleError if kind == "ordinary" else type(failure)
    )
    with (
        patch.object(
            lifecycle, "write_scientific_ablation_lifecycle_checkpoint_v1", side_effect=lost_ack
        ),
        pytest.raises(expected) as caught,
    ):
        controller.advance()
    if kind != "ordinary":
        assert caught.value is failure
    pending = controller._pending
    assert pending is not None and controller.publication_pending
    assert (pending.manifest.payload.state.phase == "stopped") is terminal
    _assert_external_operations_rejected(controller)
    assert controller._pending is pending and controller.best_checkpoint is not None
    with (
        patch.object(controller._run, "_step_authorized") as training,
        patch.object(lifecycle, "_evaluate_scientific_ablation_validation_v1") as evaluation,
        patch.object(lifecycle, "build_scientific_ablation_checkpoint_v1") as rebuild,
        patch.object(
            lifecycle,
            "write_scientific_ablation_lifecycle_checkpoint_v1",
            wraps=io.write_scientific_ablation_lifecycle_checkpoint_v1,
        ) as writer,
    ):
        receipt = controller.advance()
    training.assert_not_called()
    evaluation.assert_not_called()
    rebuild.assert_not_called()
    assert writer.call_count == 1 and writer.call_args.args[0] is pending
    assert published == [pending]
    assert type(receipt) is io.IrisScientificAblationLifecycleReceiptV1
    assert receipt.manifest == pending.manifest and not controller.publication_pending
    assert controller.state.phase == ("stopped" if terminal else "train")


def test_delegation_is_explicit_and_consumed_once_not_an_ambient_bypass() -> None:
    controller = _controller(maximum=2, checkpoint_every_steps=1)
    run = controller._run
    authorize = run._authorize_operation
    observed: list[object] = []

    def delegated(operation: str, capability: object | None) -> None:
        if capability is not None:
            assert controller._delegation is capability
            observed.append(capability)
            # No public call receives the offered internal capability.
            with pytest.raises(IrisScientificRunError):
                run.step()
            with pytest.raises(validation.IrisScientificAblationValidationError):
                validation.evaluate_scientific_ablation_validation_v1(run, controller._validation)
            with pytest.raises(native.IrisScientificAblationCheckpointError):
                native.restore_scientific_ablation_checkpoint_v1(candidate, run=run)
            assert controller._delegation is capability
        authorize(operation, capability)

    candidate = native.build_scientific_ablation_checkpoint_v1(run)
    with patch.object(run, "_authorize_operation", side_effect=delegated):
        result = controller.advance()
    assert type(result) is IrisScientificAblationOptimizationOutputV1
    assert result.step_index == 1 and len(observed) == 1
    assert controller._delegation is None
    before = native.build_scientific_ablation_checkpoint_v1(run)
    with pytest.raises(IrisScientificRunError):
        run._step_authorized(observed[0])
    assert native.build_scientific_ablation_checkpoint_v1(run) == before
    assert not controller.poisoned and controller.state.phase == "validation"


def test_independent_ablation_run_train_validation_restore_and_next_update_are_unchanged() -> None:
    source = _run(policy=_policy(minimum_target_tokens=1, max_optimizer_steps=3))
    with _stochastic(source):
        source.step()
    candidate = native.build_scientific_ablation_checkpoint_v1(source)
    report = validation.evaluate_scientific_ablation_validation_v1(source, _validation())
    assert report.at_optimizer_step == 1
    assert native.build_scientific_ablation_checkpoint_v1(source) == candidate
    resumed = _run(policy=_policy(minimum_target_tokens=1, max_optimizer_steps=3))
    native.restore_scientific_ablation_checkpoint_v1(candidate, run=resumed)
    assert source._controller_attachment is resumed._controller_attachment is None
    with _stochastic(source), _stochastic(resumed):
        assert source.step() == resumed.step()
    _assert_equal(source, resumed)


def test_historical_scientific_d_unbound_core_training_remains_available() -> None:
    run = _old_run()
    result = _ScientificRunCore.step(run)
    assert result.step_index == run.completed_steps == 1 and not run.poisoned


@pytest.mark.parametrize("kind", ("ordinary", "signal", "group"))
def test_untouched_constructor_failure_rolls_back_claims_and_allows_retry(kind: str) -> None:
    run = _run(policy=_policy(minimum_target_tokens=1))
    before = native.build_scientific_ablation_checkpoint_v1(run)
    caller = capture_runtime_state(torch.device("cpu"))
    failure: BaseException = (
        ValueError("invalid rows")
        if kind == "ordinary"
        else KeyboardInterrupt("primary")
        if kind == "signal"
        else BaseExceptionGroup("primary", [SystemExit(83)])
    )
    expected = (
        lifecycle.IrisScientificAblationLifecycleError if kind == "ordinary" else type(failure)
    )
    with (
        patch.object(lifecycle, "own_scientific_ablation_validation_rows_v1", side_effect=failure),
        pytest.raises(expected) as caught,
    ):
        lifecycle.IrisScientificAblationLifecycleV1(
            run=run,
            validation=_validation(),
            policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
        )
    if kind != "ordinary":
        assert caught.value is failure
    assert lifecycle._OWNERS.get(run) is None
    assert run._controller_attachment is run._controller_attachment_guard is None
    assert native.build_scientific_ablation_checkpoint_v1(run) == before
    assert not run.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    attached = lifecycle.IrisScientificAblationLifecycleV1(
        run=run,
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
    )
    assert attached.state.step == 0


@pytest.mark.parametrize("kind", ("ordinary", "signal", "group"))
def test_constructor_caller_cleanup_failure_does_not_commit_attachment(kind: str) -> None:
    run = _run(policy=_policy(minimum_target_tokens=1))
    before = validation._snapshot(run)
    caller = capture_runtime_state(torch.device("cpu"))
    failure: BaseException = (
        ValueError("uncertain cleanup")
        if kind == "ordinary"
        else KeyboardInterrupt("cleanup signal")
        if kind == "signal"
        else BaseExceptionGroup("cleanup group", [SystemExit(83)])
    )

    def uncertain(snapshot: Any) -> None:
        restore_runtime_state(snapshot)
        raise failure

    expected = (
        lifecycle.IrisScientificAblationLifecycleError if kind == "ordinary" else type(failure)
    )
    with (
        patch.object(lifecycle, "restore_runtime_state", side_effect=uncertain),
        pytest.raises(expected) as caught,
    ):
        lifecycle.IrisScientificAblationLifecycleV1(
            run=run,
            validation=_validation(),
            policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
        )
    if kind != "ordinary":
        assert caught.value is failure
    assert run.poisoned and run._optimizer.poisoned and not run._busy
    assert lifecycle._OWNERS.get(run) is None
    assert run._controller_attachment is run._controller_attachment_guard is None
    assert validation._snapshot(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ("ordinary", "signal", "group"))
def test_partial_registry_install_is_rolled_back_without_poisoning_untouched_run(kind: str) -> None:
    run = _run(policy=_policy(minimum_target_tokens=1))
    before = native.build_scientific_ablation_checkpoint_v1(run)
    registry_type = type(lifecycle._OWNERS)
    install = registry_type.__setitem__
    failure: BaseException = (
        ValueError("after registry install")
        if kind == "ordinary"
        else KeyboardInterrupt("primary")
        if kind == "signal"
        else BaseExceptionGroup("primary", [SystemExit(83)])
    )

    def partial(
        registry: Any,
        key: IrisScientificAblationRunV1,
        value: tuple[object, ReferenceType[object]],
    ) -> None:
        install(registry, key, value)
        if key is run:
            raise failure

    expected = (
        lifecycle.IrisScientificAblationLifecycleError if kind == "ordinary" else type(failure)
    )
    with (
        patch.object(registry_type, "__setitem__", new=partial),
        pytest.raises(expected) as caught,
    ):
        lifecycle.IrisScientificAblationLifecycleV1(
            run=run,
            validation=_validation(),
            policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
        )
    if kind != "ordinary":
        assert caught.value is failure
    assert lifecycle._OWNERS.get(run) is None
    assert run._controller_attachment is run._controller_attachment_guard is None
    assert not run.poisoned and not run._busy
    assert native.build_scientific_ablation_checkpoint_v1(run) == before
    attached = lifecycle.IrisScientificAblationLifecycleV1(
        run=run,
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(purpose="smoke"),
    )
    assert attached.state.step == 0


@pytest.mark.parametrize("kind", ("local", "guard", "registry_missing", "registry_replaced"))
def test_attachment_tampering_cannot_be_resealed_by_cursor_commit(kind: str) -> None:
    controller = _controller()
    run: IrisScientificAblationRunV1 = controller._run
    cursor = run._cursor()
    state = validation._snapshot(run)
    if kind == "local":
        run._controller_attachment = None
    elif kind == "guard":
        run._controller_attachment_guard = None
    elif kind == "registry_missing":
        del lifecycle._OWNERS[run]
    else:
        lifecycle._OWNERS[run] = object(), ref(controller)
    run._commit_owner()
    with (
        patch.object(run._optimizer, "step") as training,
        pytest.raises(IrisScientificRunError),
    ):
        run.step()
    training.assert_not_called()
    with pytest.raises(lifecycle.IrisScientificAblationLifecycleError):
        controller.advance()
    assert controller.poisoned and run.poisoned and run._optimizer.poisoned
    assert run._cursor() == cursor and validation._snapshot(run) == state
    assert not controller._busy and not run._busy
