"""Actual mode-bound lifecycle disk restore and exact pending publication retry."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

import iris.application as application
import iris.artifacts as artifacts
from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_lifecycle as controller_module
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts import scientific_ablation_lifecycle_checkpoint_v1 as schema
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_run import _run, _stochastic
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration


def _durable(
    root: Path,
    routing: Literal["learned", "random", "uniform"] = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    cycles: int = 3,
    maximum: int = 5,
    validate: int = 1,
    checkpoint: int = 3,
    patience: int = 8,
    storage: int = 8 * 1024**3,
) -> IrisScientificAblationLifecycleV1:
    return IrisScientificAblationLifecycleV1(
        run=_run(
            routing,
            goal=goal,
            broadcast=broadcast,
            cycles=cycles,
            policy=_policy(minimum_target_tokens=1, max_optimizer_steps=maximum),
        ),
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=validate,
            checkpoint_every_steps=checkpoint,
            early_stopping_patience=patience,
            minimum_improvement=1e6,
            maximum_storage_bytes=storage,
        ),
        checkpoint_root=root,
    )


def _receipt(
    controller: IrisScientificAblationLifecycleV1,
) -> io.IrisScientificAblationLifecycleReceiptV1:
    for _ in range(30):
        with _stochastic(controller._run):
            result = controller.advance()
        if type(result) is io.IrisScientificAblationLifecycleReceiptV1:
            return result
    raise AssertionError("No actual lifecycle receipt")


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_c3_disk_restore_all_modes_matches_next_stochastic_update(
    tmp_path: Path,
    routing: Literal["learned", "random", "uniform"],
    goal: str,
    broadcast: str,
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    original = _durable(tmp_path / "original", routing, goal=goal, broadcast=broadcast)
    receipt = _receipt(original)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    assert receipt.manifest == bundle.manifest and original.state.step == 3
    assert bundle.manifest.payload.best is not None
    assert bundle.manifest.payload.best.payload.completed_steps == 1
    assert (
        bundle.best_state_bytes is not None
        and original._run._runtime.python_rng_state[2] is not None
    )
    current = build_scientific_ablation_checkpoint_v1(original._run)
    resumed = _durable(tmp_path / "resumed", routing, goal=goal, broadcast=broadcast)
    io.restore_scientific_ablation_lifecycle_v1(bundle, controller=resumed)
    assert original.state == resumed.state and original.best_checkpoint == resumed.best_checkpoint
    assert build_scientific_ablation_checkpoint_v1(resumed._run) == current
    assert not resumed.publication_pending
    with _stochastic(original._run), _stochastic(resumed._run):
        left, right = original.advance(), resumed.advance()
    assert left == right and original.state == resumed.state
    assert build_scientific_ablation_checkpoint_v1(
        original._run
    ) == build_scientific_ablation_checkpoint_v1(resumed._run)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "failure",
    (
        OSError("ack lost"),
        KeyboardInterrupt("ack lost"),
        BaseExceptionGroup("ack lost", [ValueError("other"), SystemExit(83)]),
    ),
)
def test_lost_ack_retries_exact_pending_without_train_eval_or_rebuild(
    tmp_path: Path, failure: BaseException
) -> None:
    controller = _durable(tmp_path)
    while controller.state.phase != "checkpoint":
        controller.advance()
    before = controller.state
    writes: list[io.IrisScientificAblationLifecycleBundleV1] = []
    paths: list[Path] = []

    def uncertain(bundle: io.IrisScientificAblationLifecycleBundleV1, *, root: Path) -> Path:
        writes.append(bundle)
        paths.append(io.write_scientific_ablation_lifecycle_checkpoint_v1(bundle, root=root))
        raise failure

    expected = (
        controller_module.IrisScientificAblationLifecycleError
        if isinstance(failure, Exception)
        else type(failure)
    )
    with (
        patch.object(
            controller_module,
            "write_scientific_ablation_lifecycle_checkpoint_v1",
            side_effect=uncertain,
        ),
        pytest.raises(expected) as caught,
    ):
        controller.advance()
    if not isinstance(failure, Exception):
        assert caught.value is failure
    assert controller.state == before and controller.publication_pending and not controller.poisoned
    assert controller._pending is writes[0]
    with (
        patch.object(controller._run, "_step_authorized") as train,
        patch.object(controller_module, "_evaluate_scientific_ablation_validation_v1") as evaluate,
        patch.object(controller_module, "build_scientific_ablation_checkpoint_v1") as build,
    ):
        receipt = controller.advance()
    train.assert_not_called()
    evaluate.assert_not_called()
    build.assert_not_called()
    assert type(receipt) is io.IrisScientificAblationLifecycleReceiptV1
    assert receipt.path == paths[0] and receipt.manifest == writes[0].manifest
    assert not controller.publication_pending and controller.state.checkpoint_count == 1


@pytest.mark.parametrize("cycles", (1, 3))
@pytest.mark.parametrize("kind", ("shared", "no_best", "terminal"))
def test_shared_absent_best_and_off_cadence_terminal_inventory(
    tmp_path: Path, cycles: int, kind: str
) -> None:
    def case_controller(root: Path) -> IrisScientificAblationLifecycleV1:
        if kind == "shared":
            return _durable(root, cycles=cycles, checkpoint=1)
        if kind == "no_best":
            return _durable(root, cycles=cycles, validate=5, checkpoint=2)
        return _durable(root, cycles=cycles, maximum=2, validate=5, checkpoint=3)

    caller = capture_runtime_state(torch.device("cpu"))
    controller = case_controller(tmp_path)
    receipt = _receipt(controller)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    assert bundle.best_state_bytes is None
    assert {path.name for path in receipt.path.iterdir()} == {
        "manifest.json",
        "current.safetensors",
    }
    if kind == "no_best":
        assert controller.best_checkpoint is None and bundle.manifest.payload.best is None
    else:
        assert bundle.manifest.payload.best == bundle.manifest.payload.current
    if kind == "terminal":
        assert controller.state.phase == "stopped" and controller.state.stop_reason == "step_budget"
        assert controller.state.step == 2 and controller.state.validation_count == 1
    assert (
        io.write_scientific_ablation_lifecycle_checkpoint_v1(bundle, root=tmp_path) == receipt.path
    )
    resumed = case_controller(tmp_path / "fresh")
    io.restore_scientific_ablation_lifecycle_v1(bundle, controller=resumed)
    assert (
        resumed.state == controller.state and resumed.best_checkpoint == controller.best_checkpoint
    )
    assert build_scientific_ablation_checkpoint_v1(
        resumed._run
    ) == build_scientific_ablation_checkpoint_v1(controller._run)
    if kind == "terminal":
        with (
            patch.object(resumed._run._optimizer, "step") as train,
            pytest.raises(controller_module.IrisScientificAblationLifecycleError),
        ):
            resumed.advance()
        train.assert_not_called()
    else:
        with _stochastic(controller._run), _stochastic(resumed._run):
            assert controller.advance() == resumed.advance()
        assert build_scientific_ablation_checkpoint_v1(
            controller._run
        ) == build_scientific_ablation_checkpoint_v1(resumed._run)
        assert resumed.state == controller.state
    assert not resumed.publication_pending and not resumed.poisoned
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("cycles", (1, 3))
def test_off_cadence_early_stopping_restores_selected_best_and_terminal_phase(
    tmp_path: Path, cycles: int
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    controller = _durable(tmp_path, cycles=cycles, patience=1)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(_receipt(controller).path)
    assert controller.state.phase == "stopped" and controller.state.stop_reason == "early_stopping"
    assert controller.state.step == 2 and controller.state.best_step == 1
    assert controller.state.checkpoint_count == 1 and controller.state.nonimproving_validations == 1
    assert bundle.best_state_bytes is not None
    resumed = _durable(tmp_path / "fresh", cycles=cycles, patience=1)
    io.restore_scientific_ablation_lifecycle_v1(bundle, controller=resumed)
    assert (
        resumed.state == controller.state and resumed.best_checkpoint == controller.best_checkpoint
    )
    assert build_scientific_ablation_checkpoint_v1(
        resumed._run
    ) == build_scientific_ablation_checkpoint_v1(controller._run)
    with (
        patch.object(resumed._run._optimizer, "step") as train,
        pytest.raises(controller_module.IrisScientificAblationLifecycleError),
    ):
        resumed.advance()
    train.assert_not_called()
    assert not resumed.poisoned and not resumed.publication_pending
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ("outer", "payload", "state", "current", "best"))
def test_hidden_unknown_fields_reject_before_parent_dump(tmp_path: Path, kind: str) -> None:
    controller = _durable(tmp_path)
    bundle = deepcopy(
        io.read_scientific_ablation_lifecycle_checkpoint_v1(_receipt(controller).path)
    )
    assert bundle.manifest.payload.best is not None
    targets = {
        "outer": bundle.manifest,
        "payload": bundle.manifest.payload,
        "state": bundle.manifest.payload.state,
        "current": bundle.manifest.payload.current.payload,
        "best": bundle.manifest.payload.best.payload,
    }
    vars(targets[kind])["not_a_field"] = True
    before = build_scientific_ablation_checkpoint_v1(controller._run)
    with pytest.raises(ValueError):
        io._bundle_parts(bundle)
    assert build_scientific_ablation_checkpoint_v1(controller._run) == before


def test_native_restore_preserves_primary_group_over_secondary_cleanup(tmp_path: Path) -> None:
    controller = _durable(tmp_path)
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(_receipt(controller).path)
    resumed = _run(policy=_policy(minimum_target_tokens=1, max_optimizer_steps=5))
    caller = capture_runtime_state(torch.device("cpu"))
    primary = BaseExceptionGroup("primary", [KeyboardInterrupt("signal")])

    def uncertain(snapshot: Any) -> None:
        restore_runtime_state(snapshot)
        raise ValueError("secondary cleanup")

    with (
        patch.object(native, "_verify", side_effect=primary),
        patch.object(native, "restore_runtime_state", side_effect=uncertain),
        pytest.raises(BaseExceptionGroup) as caught,
    ):
        native.restore_scientific_ablation_checkpoint_v1(
            io._bundle_parts(bundle)[1],
            run=resumed,
        )
    assert caught.value is primary and resumed.poisoned and not resumed._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_canonical_new_exports() -> None:
    for name in io.__all__:
        assert name in application.__all__ and getattr(application, name) is getattr(io, name)
    for name in schema.__all__:
        assert name in artifacts.__all__ and getattr(artifacts, name) is getattr(schema, name)
