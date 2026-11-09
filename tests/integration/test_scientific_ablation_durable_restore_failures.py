"""Actual lifecycle restore failures poison uncertain writes and preserve caller state."""

import random
from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_lifecycle as controller_module
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_lifecycle_v1 import ScientificAblationLifecycleStateV1
from iris.artifacts.scientific_checkpoint_v1 import ScientificRuntimeV1
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt

pytestmark = pytest.mark.integration

type SignalKind = Literal["keyboard_interrupt", "system_exit", "base_exception_group"]
type FailureKind = SignalKind | Literal["value_error"]

_SIGNALS: tuple[SignalKind, ...] = (
    "keyboard_interrupt",
    "system_exit",
    "base_exception_group",
)
_FAILURES: tuple[FailureKind, ...] = ("value_error", *_SIGNALS)


def _failure(kind: FailureKind, message: str) -> BaseException:
    if kind == "keyboard_interrupt":
        return KeyboardInterrupt(message)
    if kind == "system_exit":
        return SystemExit(83)
    if kind == "base_exception_group":
        return BaseExceptionGroup(message, [ValueError("ordinary child"), SystemExit(84)])
    return ValueError(message)


def _perturb_caller() -> None:
    random.gauss(0.0, 1.0)
    torch.rand(3)


def _assert_owner_cleanup(controller: IrisScientificAblationLifecycleV1) -> None:
    run = controller._run
    optimizer = run._optimizer
    assert not controller._busy and not run._busy and not optimizer._busy
    assert controller._delegation is None
    assert run._controller_attachment is controller._run_attachment
    assert run._controller_attachment_guard is controller._run_attachment
    registered = controller_module._OWNERS.get(run)
    assert registered is not None
    assert registered[0] is controller._attachment_token and registered[1]() is controller
    assert optimizer._attached_run_token is run._attachment_token
    assert optimizer._pending_update_state_sha256 is None
    assert optimizer._active_terminal_token_id is None
    assert optimizer._active_teacher_state_sha256 is None
    assert all(parameter.grad is None for parameter in optimizer._topology.parameters)


def _assert_poisoned(controller: IrisScientificAblationLifecycleV1) -> None:
    assert controller._poisoned and controller._run._poisoned
    assert controller._run._optimizer.poisoned and controller.poisoned
    _assert_owner_cleanup(controller)
    with pytest.raises(controller_module.IrisScientificAblationLifecycleError):
        controller.advance()
    _assert_owner_cleanup(controller)


@pytest.fixture(scope="module")
def disk_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> io.IrisScientificAblationLifecycleBundleV1:
    root = tmp_path_factory.mktemp("trusted-ablation-durable-restore")
    controller = _durable(root)
    receipt = _receipt(controller)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    assert bundle.manifest == receipt.manifest and bundle.best_state_bytes is not None
    assert bundle.manifest.payload.best is not None
    assert bundle.manifest.payload.best.sha256 != bundle.manifest.payload.current.sha256
    return bundle


@pytest.mark.parametrize("phase", ("copy", "native_commit", "controller_commit"))
@pytest.mark.parametrize("failure_kind", _FAILURES)
def test_uncertain_lifecycle_restore_write_or_commit_poisons_all_owners(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    phase: Literal["copy", "native_commit", "controller_commit"],
    failure_kind: FailureKind,
) -> None:
    controller = _durable(tmp_path / "resumed")
    run = controller._run
    failure = _failure(failure_kind, "PRIVATE partial lifecycle restore")
    caller = capture_runtime_state(torch.device("cpu"))
    original_copy = torch.Tensor.copy_
    original_versions = tuple(
        value._version for value in run._optimizer._model.state_dict().values()
    )
    copies = 0
    commits: list[str] = []

    def partial_copy(
        value: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        nonlocal copies
        result = original_copy(value, source, non_blocking=non_blocking)
        copies += 1
        if copies == 3:
            assert controller._delegation is None
            _perturb_caller()
            raise failure
        return result

    def partial_native_commit() -> None:
        assert controller._delegation is None
        run._committed_cursor = run._cursor()
        commits.append("native_commit")
        _perturb_caller()
        raise failure

    def partial_controller_commit(
        state: ScientificAblationLifecycleStateV1,
        best: native.IrisScientificAblationCheckpointBundleV1 | None,
        *,
        clear_pending: bool = False,
    ) -> None:
        assert controller._delegation is None
        controller._state = state
        commits.append("controller_commit")
        _perturb_caller()
        raise failure

    injection: AbstractContextManager[object]
    if phase == "copy":
        injection = patch.object(torch.Tensor, "copy_", new=partial_copy)
    elif phase == "native_commit":
        injection = patch.object(run, "_commit_owner", side_effect=partial_native_commit)
    else:
        injection = patch.object(controller, "_commit", side_effect=partial_controller_commit)
    expected = (
        io.IrisScientificAblationLifecycleCheckpointError
        if isinstance(failure, Exception)
        else type(failure)
    )
    with injection, pytest.raises(expected) as caught:
        io.restore_scientific_ablation_lifecycle_v1(disk_bundle, controller=controller)

    if failure_kind != "value_error":
        assert caught.value is failure
    else:
        assert "PRIVATE" not in str(caught.value)
    if phase == "copy":
        assert copies == 3 and not commits
        restored_versions = tuple(
            value._version for value in run._optimizer._model.state_dict().values()
        )
        assert restored_versions != original_versions
    else:
        assert commits == [phase]
        assert run.completed_steps == disk_bundle.manifest.payload.current.payload.completed_steps
        if phase == "controller_commit":
            assert controller._state == disk_bundle.manifest.payload.state
    _assert_poisoned(controller)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "resumed").exists()


@pytest.mark.parametrize("layer", ("native", "controller"))
@pytest.mark.parametrize("failure_kind", _FAILURES)
def test_restore_cleanup_failure_poisons_all_owners_and_releases_capability(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    layer: Literal["native", "controller"],
    failure_kind: FailureKind,
) -> None:
    controller = _durable(tmp_path / "resumed")
    failure = _failure(failure_kind, "PRIVATE restore cleanup")
    caller = capture_runtime_state(torch.device("cpu"))
    calls: list[RuntimeStateSnapshot] = []

    def uncertain_cleanup(snapshot: RuntimeStateSnapshot) -> None:
        _perturb_caller()
        restore_runtime_state(snapshot)
        calls.append(snapshot)
        raise failure

    target = native if layer == "native" else controller_module
    expected = (
        io.IrisScientificAblationLifecycleCheckpointError
        if isinstance(failure, Exception)
        else type(failure)
    )
    with (
        patch.object(target, "restore_runtime_state", side_effect=uncertain_cleanup),
        pytest.raises(expected) as caught,
    ):
        io.restore_scientific_ablation_lifecycle_v1(disk_bundle, controller=controller)

    assert len(calls) == 1
    if failure_kind != "value_error":
        assert caught.value is failure
    else:
        assert "PRIVATE" not in str(caught.value)
    assert (
        controller._run.completed_steps
        == disk_bundle.manifest.payload.current.payload.completed_steps
    )
    _assert_poisoned(controller)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "resumed").exists()


@pytest.mark.parametrize("primary_kind", _SIGNALS)
@pytest.mark.parametrize("secondary_kind", _FAILURES)
def test_first_restore_signal_survives_native_and_controller_cleanup_failures(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    primary_kind: SignalKind,
    secondary_kind: FailureKind,
) -> None:
    controller = _durable(tmp_path / "resumed")
    primary = _failure(primary_kind, "primary native commit")
    secondary = _failure(secondary_kind, "secondary native caller cleanup")
    tertiary = BaseExceptionGroup(
        "tertiary controller caller cleanup", [ValueError("ordinary child"), KeyboardInterrupt()]
    )
    caller = capture_runtime_state(torch.device("cpu"))
    cleanup_calls: list[str] = []

    def interrupted_commit() -> None:
        assert controller._delegation is None
        controller._run._committed_cursor = controller._run._cursor()
        _perturb_caller()
        raise primary

    def native_cleanup(snapshot: RuntimeStateSnapshot) -> None:
        restore_runtime_state(snapshot)
        cleanup_calls.append("native")
        raise secondary

    def controller_cleanup(snapshot: RuntimeStateSnapshot) -> None:
        _perturb_caller()
        restore_runtime_state(snapshot)
        cleanup_calls.append("controller")
        raise tertiary

    with (
        patch.object(controller._run, "_commit_owner", side_effect=interrupted_commit),
        patch.object(native, "restore_runtime_state", side_effect=native_cleanup),
        patch.object(controller_module, "restore_runtime_state", side_effect=controller_cleanup),
        pytest.raises(type(primary)) as caught,
    ):
        io.restore_scientific_ablation_lifecycle_v1(disk_bundle, controller=controller)

    assert caught.value is primary
    assert cleanup_calls == ["native", "controller"]
    _assert_poisoned(controller)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "resumed").exists()


@pytest.mark.parametrize("failure_kind", _FAILURES)
def test_selected_best_runtime_preflight_failure_preserves_reusable_fresh_controller(
    disk_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    failure_kind: FailureKind,
) -> None:
    controller = _durable(tmp_path / "resumed")
    run = controller._run
    before = native.build_scientific_ablation_checkpoint_v1(run)
    before_state = deepcopy(controller._state)
    before_seal = controller._control_seal
    before_private = run._clone_runtime(controller._private_runtime)
    caller = capture_runtime_state(torch.device("cpu"))
    failure = _failure(failure_kind, "PRIVATE selected-best runtime preflight")
    preflight_calls: list[ScientificRuntimeV1] = []

    def failed_best_runtime(
        record: ScientificRuntimeV1, tensors: dict[str, torch.Tensor]
    ) -> RuntimeStateSnapshot:
        preflight_calls.append(record)
        _perturb_caller()
        raise failure

    expected = (
        io.IrisScientificAblationLifecycleCheckpointError
        if isinstance(failure, Exception)
        else type(failure)
    )
    with (
        patch.object(io, "_runtime", side_effect=failed_best_runtime),
        patch.object(io, "_restore_scientific_ablation_checkpoint_v1") as restore,
        patch.object(controller, "_commit") as commit,
        pytest.raises(expected) as caught,
    ):
        io.restore_scientific_ablation_lifecycle_v1(disk_bundle, controller=controller)

    restore.assert_not_called()
    commit.assert_not_called()
    best_manifest = disk_bundle.manifest.payload.best
    assert best_manifest is not None
    assert preflight_calls == [best_manifest.payload.runtime]
    if failure_kind != "value_error":
        assert caught.value is failure
    else:
        assert "PRIVATE" not in str(caught.value)
    assert not controller._poisoned and not run._poisoned and not run._optimizer.poisoned
    _assert_owner_cleanup(controller)
    assert controller._state == before_state and controller._control_seal == before_seal
    assert controller._best is None and controller._pending is None
    assert _same_runtime(before_private, controller._private_runtime)
    assert native.build_scientific_ablation_checkpoint_v1(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))

    io.restore_scientific_ablation_lifecycle_v1(disk_bundle, controller=controller)
    _, current, best = io._bundle_parts(disk_bundle)
    assert controller.state == disk_bundle.manifest.payload.state
    assert controller.best_checkpoint == best
    assert native.build_scientific_ablation_checkpoint_v1(run) == current
    assert not controller.poisoned and not controller.publication_pending
    _assert_owner_cleanup(controller)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "resumed").exists()
