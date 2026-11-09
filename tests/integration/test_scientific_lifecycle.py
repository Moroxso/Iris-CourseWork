"""Genuine tiny-Qwen lifecycle cadence, exact resume and validation boundaries."""

from math import fsum
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application.scientific_checkpoint import build_scientific_checkpoint_v1
from iris.application.scientific_lifecycle import (
    IrisScientificLifecycleBundleV1,
    IrisScientificLifecycleError,
    IrisScientificLifecycleV1,
    read_scientific_lifecycle_checkpoint_v1,
    restore_scientific_lifecycle_v1,
)
from iris.application.scientific_optimization import IrisScientificOptimizerV1
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_validation import evaluate_scientific_validation_v1
from iris.artifacts.scientific_lifecycle_v1 import (
    IrisScientificLifecycleCheckpointManifestV1,
    IrisScientificLifecyclePolicyV1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.integration.test_scientific_checkpoint import _run
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _validation() -> tuple[IrisExample, ...]:
    return tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            template_id="iris-synthetic-01-v2",
        )
    )[:4]


def _lifecycle() -> IrisScientificLifecycleV1:
    return IrisScientificLifecycleV1(
        run=_run(),
        validation=_validation(),
        policy=IrisScientificLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=1,
            checkpoint_every_steps=2,
            early_stopping_patience=2,
            minimum_improvement=1e6,
        ),
    )


def test_validation_uses_actual_token_weighted_nll_and_does_not_mutate_run() -> None:
    run = _run()
    before = build_scientific_checkpoint_v1(run)
    report = evaluate_scientific_validation_v1(run, _validation())
    assert report.example_count == 4 and report.target_token_count > 4
    assert report.target_nll_per_token == report.target_nll_sum / report.target_token_count
    assert build_scientific_checkpoint_v1(run) == before
    assert all(parameter.grad is None for parameter in run._optimizer._model.parameters())
    assert all(module.training for module in run._optimizer._model.modules())
    model = run._optimizer._model
    model.eval()
    try:
        outputs = [model.evaluation_forward(row, terminal_token_id=1) for row in _validation()]
        counts = [out.supervised_token_count for out in outputs]
        sums = [float(out.losses.language_model) for out in outputs]
        assert len(set(counts)) > 1
        assert report.target_nll_per_token == fsum(sums) / sum(counts)
        assert report.target_nll_per_token != pytest.approx(
            fsum(value / count for value, count in zip(sums, counts, strict=True)) / len(counts),
            abs=1e-8,
        )
    finally:
        model.train(True)


def test_early_stopping_best_state_disk_resume_and_no_duplicate_training(tmp_path: Path) -> None:
    uninterrupted = _lifecycle()
    first_path = None
    while uninterrupted.state.phase != "stopped":
        result = uninterrupted.advance(checkpoint_root=tmp_path / "continuous")
        if isinstance(result, Path) and first_path is None:
            first_path = result
    state = uninterrupted.state
    assert state.step == 3 and state.stop_reason == "early_stopping"
    assert state.best_step == 1 and state.nonimproving_validations == 2
    assert state.checkpoint_count == 2 and first_path is not None
    restored = _lifecycle()
    restore_scientific_lifecycle_v1(
        read_scientific_lifecycle_checkpoint_v1(first_path), lifecycle=restored
    )
    restored_state = restored.state
    assert restored_state.step == 2 and restored_state.phase == "train"
    while restored.state.phase != "stopped":
        restored.advance(checkpoint_root=tmp_path / "resumed")
    assert restored.state == state
    assert restored.best_checkpoint == uninterrupted.best_checkpoint
    assert build_scientific_checkpoint_v1(restored._run) == build_scientific_checkpoint_v1(
        uninterrupted._run
    )
    with pytest.raises(IrisScientificLifecycleError):
        restored.advance(checkpoint_root=tmp_path / "resumed")


def test_checkpoint_failure_retries_same_pending_bytes_without_forward(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    while lifecycle.state.phase != "checkpoint":
        lifecycle.advance(checkpoint_root=tmp_path)
    before = build_scientific_checkpoint_v1(lifecycle._run)
    with (
        patch("iris.application.scientific_lifecycle._publish", side_effect=OSError("private")),
        pytest.raises(IrisScientificLifecycleError),
    ):
        lifecycle.advance(checkpoint_root=tmp_path)
    pending = lifecycle._pending
    assert pending is not None and lifecycle.state.phase == "checkpoint"
    with patch.object(lifecycle._run, "step") as step:
        result = lifecycle.advance(checkpoint_root=tmp_path)
    step.assert_not_called()
    assert isinstance(result, Path) and read_scientific_lifecycle_checkpoint_v1(result) == pending
    assert build_scientific_checkpoint_v1(lifecycle._run) == before


def test_default_source_cadences_are_explicit() -> None:
    policy = IrisScientificLifecyclePolicyV1()
    assert (
        policy.validate_every_steps,
        policy.checkpoint_every_steps,
        policy.early_stopping_patience,
    ) == (500, 1000, 8)
    assert policy.selection_metric == "validation_target_nll_per_token"
    assert policy.maximum_storage_bytes == 8 * 1024**3


def _short_lifecycle(**updates: Any) -> IrisScientificLifecycleV1:
    model, _, _ = _build()
    run = IrisScientificRunV1(
        optimizer=IrisScientificOptimizerV1(model=model, policy=_policy(max_optimizer_steps=5)),
        examples=_rows()[:5],
        terminal_token_id=1,
        seed=901,
    )
    policy = IrisScientificLifecyclePolicyV1.model_validate(
        {
            "purpose": "smoke",
            "validate_every_steps": 3,
            "checkpoint_every_steps": 2,
            "early_stopping_patience": 8,
            **updates,
        }
    )
    return IrisScientificLifecycleV1(run=run, validation=_validation(), policy=policy)


def test_terminal_off_cadence_validation_and_checkpoint_resume(tmp_path: Path) -> None:
    lifecycle = _short_lifecycle()
    snapshots = []
    validation_steps = []
    while lifecycle.state.phase != "stopped":
        phase = lifecycle.state.phase
        result = lifecycle.advance(checkpoint_root=tmp_path)
        if phase == "validation":
            validation_steps.append(lifecycle.state.step)
        if isinstance(result, Path):
            snapshots.append(result)
    assert validation_steps == [3, 5]
    assert lifecycle.state.stop_reason == "step_budget"
    assert lifecycle.state.checkpoint_count == 3
    assert [
        read_scientific_lifecycle_checkpoint_v1(path).manifest.payload.state.step
        for path in snapshots
    ] == [2, 4, 5]
    for path in snapshots:
        other = _short_lifecycle()
        restore_scientific_lifecycle_v1(
            read_scientific_lifecycle_checkpoint_v1(path), lifecycle=other
        )
        assert (
            other.state.step
            == read_scientific_lifecycle_checkpoint_v1(path).manifest.payload.state.step
        )


def test_uncertain_successful_publication_is_reused_without_retraining(tmp_path: Path) -> None:
    from iris.application.scientific_lifecycle import _publish

    lifecycle = _short_lifecycle()
    while lifecycle.state.phase != "checkpoint":
        lifecycle.advance(checkpoint_root=tmp_path)

    def uncertain(*args: Any, **kwargs: Any) -> Path:
        _publish(*args, **kwargs)
        raise OSError("PRIVATE acknowledgement lost")

    with (
        patch("iris.application.scientific_lifecycle._publish", side_effect=uncertain),
        pytest.raises(IrisScientificLifecycleError),
    ):
        lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle.state.phase == "checkpoint" and lifecycle.state.step == 2
    pending = lifecycle._pending
    with patch.object(lifecycle._run, "step") as training:
        result = lifecycle.advance(checkpoint_root=tmp_path)
    training.assert_not_called()
    assert isinstance(result, Path) and read_scientific_lifecycle_checkpoint_v1(result) == pending
    assert lifecycle.state.checkpoint_count == 1


def test_storage_budget_failure_keeps_pending_state_and_creates_no_checkpoint(
    tmp_path: Path,
) -> None:
    lifecycle = _short_lifecycle(maximum_storage_bytes=1)
    while lifecycle.state.phase != "checkpoint":
        lifecycle.advance(checkpoint_root=tmp_path)
    with pytest.raises(IrisScientificLifecycleError):
        lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle._pending is not None and lifecycle.state.step == 2
    assert tuple((tmp_path / "scientific-lifecycle-v1").iterdir()) == ()


def test_validation_rejects_train_test_and_seed_overlap_without_changing_weights() -> None:
    run = _run()
    baseline = build_scientific_checkpoint_v1(run)
    row = _validation()[0]
    for examples in (
        (_rows()[0],),
        (row.model_copy(update={"split": "test", "id": f"test-{row.seed}"}),),
        (_rows()[0].model_copy(update={"split": "validation", "id": "validation-0"}),),
    ):
        with pytest.raises(RuntimeError):
            evaluate_scientific_validation_v1(run, examples)
        assert build_scientific_checkpoint_v1(run) == baseline


def test_corrupted_lifecycle_current_state_cannot_be_read(tmp_path: Path) -> None:
    lifecycle = _short_lifecycle()
    while lifecycle.state.phase != "checkpoint":
        lifecycle.advance(checkpoint_root=tmp_path)
    path = lifecycle.advance(checkpoint_root=tmp_path)
    assert isinstance(path, Path)
    state_file = path / "current.safetensors"
    value = state_file.read_bytes()
    state_file.write_bytes(value[:-1])
    with pytest.raises(IrisScientificLifecycleError):
        read_scientific_lifecycle_checkpoint_v1(path)


@pytest.mark.parametrize(
    "kind", ["checkpoint_count", "validation_count", "best_step", "stop_reason"]
)
def test_rehashed_invalid_lifecycle_control_is_rejected_before_copy(
    tmp_path: Path, kind: str
) -> None:
    source = _lifecycle()
    path = None
    while source.state.phase != "stopped":
        result = source.advance(checkpoint_root=tmp_path)
        if isinstance(result, Path):
            path = result
    assert path is not None
    bundle = read_scientific_lifecycle_checkpoint_v1(path)
    raw = bundle.manifest.payload.model_dump(mode="python")
    if kind == "checkpoint_count":
        raw["state"]["checkpoint_count"] = 3
    elif kind == "validation_count":
        raw["state"]["validation_count"] = 2
        raw["state"]["nonimproving_validations"] = 1
        raw["state"]["stop_reason"] = "step_budget"
    elif kind == "best_step":
        raw["state"]["best_step"] = 2
        # This mismatch must fail schema validation before restore.
    else:
        raw["state"]["stop_reason"] = "step_budget"
        raw["state"]["nonimproving_validations"] = 1
    payload = bundle.manifest.payload.model_construct(**raw)
    # Construct an intentionally untrusted manifest without pre-validating it.
    manifest = IrisScientificLifecycleCheckpointManifestV1.model_construct(
        sha256=domain_sha256("iris/scientific-lifecycle-checkpoint/v1", canonical_json_bytes(raw)),
        payload=payload,
    )
    target = _lifecycle()
    before = build_scientific_checkpoint_v1(target._run)
    invalid = IrisScientificLifecycleBundleV1(
        manifest, bundle.current_state_bytes, bundle.best_state_bytes
    )
    with pytest.raises(IrisScientificLifecycleError):
        restore_scientific_lifecycle_v1(invalid, lifecycle=target)
    assert build_scientific_checkpoint_v1(target._run) == before


def test_evaluation_failure_is_retryable_without_advancing_lifecycle(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    lifecycle.advance(checkpoint_root=tmp_path)
    before = build_scientific_checkpoint_v1(lifecycle._run)
    with (
        patch.object(
            lifecycle._run._optimizer._model,
            "evaluation_forward",
            side_effect=RuntimeError("PRIVATE"),
        ),
        pytest.raises(IrisScientificLifecycleError),
    ):
        lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle.state.phase == "validation" and lifecycle.state.validation_count == 0
    assert build_scientific_checkpoint_v1(lifecycle._run) == before
    lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle.state.validation_count == 1


def test_best_equal_current_has_one_tensor_file_and_roundtrips(tmp_path: Path) -> None:
    lifecycle = _short_lifecycle(validate_every_steps=1, checkpoint_every_steps=1)
    lifecycle.advance(checkpoint_root=tmp_path)
    lifecycle.advance(checkpoint_root=tmp_path)
    path = lifecycle.advance(checkpoint_root=tmp_path)
    assert isinstance(path, Path)
    bundle = read_scientific_lifecycle_checkpoint_v1(path)
    assert bundle.best_state_bytes is None
    assert bundle.manifest.payload.best == bundle.manifest.payload.current
    assert {file.name for file in path.iterdir()} == {"manifest.json", "current.safetensors"}
    target = _short_lifecycle(validate_every_steps=1, checkpoint_every_steps=1)
    restore_scientific_lifecycle_v1(bundle, lifecycle=target)
    assert target.best_checkpoint == lifecycle.best_checkpoint


def test_source_patience_eight_stops_after_eight_nonimproving_validations(tmp_path: Path) -> None:
    lifecycle = IrisScientificLifecycleV1(
        run=_run(),
        validation=_validation(),
        policy=IrisScientificLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=1,
            checkpoint_every_steps=20,
            early_stopping_patience=8,
            minimum_improvement=1e6,
        ),
    )
    while lifecycle.state.phase != "stopped":
        lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle.state.step == 9 and lifecycle.state.validation_count == 9
    assert lifecycle.state.best_step == 1 and lifecycle.state.nonimproving_validations == 8
    assert lifecycle.state.stop_reason == "early_stopping" and lifecycle.state.checkpoint_count == 1
