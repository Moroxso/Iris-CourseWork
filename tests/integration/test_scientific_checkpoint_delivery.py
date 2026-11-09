"""Saved current/best delivery without replacing a live context's training state."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors

from iris import application
from iris.application import scientific_delivery as delivery
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_delivery import (
    IrisScientificDeliveryError,
    build_scientific_checkpoint_delivery_v1,
    build_scientific_delivery_v1,
    read_scientific_delivery_v1,
    restore_scientific_delivery_v1,
    write_scientific_delivery_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_validation import evaluate_scientific_validation_v1
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _run
from tests.integration.test_scientific_lifecycle import _lifecycle, _validation

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("context_steps", [0, 1, 3])
def test_saved_delivery_matches_restored_owner_without_touching_context(
    tmp_path: Path,
    context_steps: int,
) -> None:
    saved = _run()
    saved.step()
    metric = evaluate_scientific_validation_v1(saved, _validation())
    checkpoint = build_scientific_checkpoint_v1(saved)
    expected = build_scientific_delivery_v1(saved, validation=metric)
    context = _run()
    for _ in range(context_steps):
        context.step()
    before = build_scientific_checkpoint_v1(context)
    runtime = capture_runtime_state(torch.device("cpu"))
    with patch("torch.load", side_effect=AssertionError("No incoming pickle")):
        actual = build_scientific_checkpoint_delivery_v1(
            checkpoint,
            context=context,
            validation=metric,
        )
    assert actual == expected
    assert build_scientific_checkpoint_v1(context) == before
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), runtime)
    path = write_scientific_delivery_v1(actual, root=tmp_path)
    assert read_scientific_delivery_v1(path) == actual
    restored = _run()
    restore_scientific_delivery_v1(actual, run=restored)
    assert build_scientific_checkpoint_v1(restored) == checkpoint


def test_lifecycle_best_retains_its_own_weights_and_same_step_metric(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    lifecycle.advance(checkpoint_root=tmp_path)  # train1
    first_metric = evaluate_scientific_validation_v1(lifecycle._run, _validation())
    lifecycle.advance(checkpoint_root=tmp_path)  # selects step1
    while lifecycle.state.phase != "stopped":
        lifecycle.advance(checkpoint_root=tmp_path)
    assert lifecycle.state.step == 3 and lifecycle.state.best_step == 1
    saved = lifecycle.best_checkpoint
    assert saved is not None
    context = lifecycle._run
    current = build_scientific_checkpoint_v1(context)
    actual = build_scientific_checkpoint_delivery_v1(
        saved,
        context=context,
        validation=first_metric,
    )
    expected_owner = _run()
    restore_scientific_checkpoint_v1(saved, run=expected_owner)
    assert actual == build_scientific_delivery_v1(expected_owner, validation=first_metric)
    files = dict(actual.files)
    assert json.loads(files["config.yaml"])["checkpoint"]["sha256"] == saved.manifest.sha256
    assert json.loads(files["metrics.json"])["validation"]["at_optimizer_step"] == 1
    saved_tensors = load_tensors(saved.state_bytes)
    exported = load_tensors(files["iris_model.safetensors"])
    assert all(
        torch.equal(value, saved_tensors[f"model/{name}"]) for name, value in exported.items()
    )
    assert build_scientific_checkpoint_v1(context) == current
    current_metric = evaluate_scientific_validation_v1(context, _validation())
    with pytest.raises(IrisScientificDeliveryError):
        build_scientific_checkpoint_delivery_v1(saved, context=context, validation=current_metric)


@pytest.mark.parametrize("mismatch", ["corpus", "optimizer", "weights", "metric", "busy"])
def test_incompatible_saved_state_is_rejected_without_context_mutation(mismatch: str) -> None:
    source = _run()
    source.step()
    checkpoint = build_scientific_checkpoint_v1(source)
    context = _run(reverse=mismatch == "corpus", minimum=70 if mismatch == "optimizer" else 1)
    metric = None
    if mismatch == "weights":
        checkpoint = IrisScientificCheckpointBundleV1(checkpoint.manifest, b"corrupt")
    if mismatch == "metric":
        metric = replace(
            evaluate_scientific_validation_v1(source, _validation()), at_optimizer_step=0
        )
    before = build_scientific_checkpoint_v1(context)
    if mismatch == "busy":
        context._busy = True
    try:
        with pytest.raises(IrisScientificDeliveryError):
            build_scientific_checkpoint_delivery_v1(checkpoint, context=context, validation=metric)
        assert context._busy is (mismatch == "busy")
    finally:
        context._busy = False
    assert not context.poisoned
    assert build_scientific_checkpoint_v1(context) == before


def test_saved_checkpoint_delivery_public_export() -> None:
    assert (
        application.build_scientific_checkpoint_delivery_v1
        is build_scientific_checkpoint_delivery_v1
    )


def test_live_builder_still_rejects_source_advancing_after_its_snapshot() -> None:
    run = _run()

    def advance_after_snapshot(source: IrisScientificRunV1) -> IrisScientificCheckpointBundleV1:
        checkpoint = build_scientific_checkpoint_v1(source)
        source.step()
        return checkpoint

    with (
        patch.object(
            delivery, "build_scientific_checkpoint_v1", side_effect=advance_after_snapshot
        ),
        pytest.raises(IrisScientificDeliveryError, match="source changed"),
    ):
        build_scientific_delivery_v1(run)
    assert run.completed_steps == 1 and not run._busy and not run.poisoned
