"""Full-run exact continuation, corpus cursor and failure checks on real tiny Qwen."""

import random
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    IrisScientificCheckpointError,
    build_scientific_checkpoint_v1,
    read_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
    write_scientific_checkpoint_v1,
)
from iris.application.scientific_optimization import IrisScientificOptimizerV1, _same_runtime
from iris.application.scientific_run import IrisScientificRunError, IrisScientificRunV1
from iris.artifacts.scientific_checkpoint_v1 import (
    IrisScientificCheckpointManifestV1,
    ScientificCheckpointPayloadV1,
)
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _run(*, model_seed: int = 1907, reverse: bool = False, minimum: int = 1) -> IrisScientificRunV1:
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        random.seed(901)
        # Include an occupied Box-Muller Gaussian cache in the saved Python state.
        random.gauss(0, 1)
        torch.manual_seed(321)
        model, _, _ = _build(seed=model_seed)
        rows = _rows()[:5]
        return IrisScientificRunV1(
            optimizer=IrisScientificOptimizerV1(
                model=model, policy=_policy(minimum_target_tokens=minimum)
            ),
            examples=tuple(reversed(rows)) if reverse else rows,
            terminal_token_id=1,
            seed=901,
        )
    finally:
        restore_runtime_state(caller)


@contextmanager
def _stochastic(run: IrisScientificRunV1) -> Iterator[None]:
    model = run._optimizer._model
    original = model.training_forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        factor = 0.9 + 0.1 * float(torch.rand(())) + 0.01 * random.gauss(0, 1)
        # Real loss/gradient dependence on both RNG streams; replay must reproduce it.
        return replace(
            output,
            losses=replace(output.losses, language_model=output.losses.language_model * factor),
        )

    with patch.object(model, "training_forward", side_effect=forward):
        yield


@pytest.mark.parametrize("cut", [0, 1, 3])
def test_disk_checkpoint_resume_is_bitwise_equal_to_uninterrupted_training(
    tmp_path: Path, cut: int
) -> None:
    continuous = _run(minimum=70)
    interrupted = _run(minimum=70)
    ambient = capture_runtime_state(torch.device("cpu"))
    with _stochastic(continuous):
        expected = [continuous.step() for _ in range(4)]
    with _stochastic(interrupted):
        for index in range(cut):
            assert interrupted.step() == expected[index]
    bundle = build_scientific_checkpoint_v1(interrupted)
    location = write_scientific_checkpoint_v1(bundle, root=tmp_path / "checkpoints")
    assert write_scientific_checkpoint_v1(bundle, root=tmp_path / "checkpoints") == location
    loaded = read_scientific_checkpoint_v1(location)
    assert loaded == bundle
    resumed = _run(minimum=70)
    restore_scientific_checkpoint_v1(loaded, run=resumed)
    with _stochastic(resumed):
        for index in range(cut, 4):
            assert resumed.step() == expected[index]
    assert build_scientific_checkpoint_v1(resumed) == build_scientific_checkpoint_v1(continuous)
    assert resumed.examples_seen > len(resumed._rows)  # At least one actual epoch boundary.
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    assert all(
        p.grad is None and not p.requires_grad for p in resumed._backbone._model.parameters()
    )


def test_saved_bundle_does_not_alias_later_training_or_restored_owner() -> None:
    run = _run()
    run.step()
    bundle = build_scientific_checkpoint_v1(run)
    state = bundle.state_bytes
    other = _run()
    restore_scientific_checkpoint_v1(bundle, run=other)
    other.step()
    assert bundle.state_bytes == state and run.completed_steps == 1
    assert build_scientific_checkpoint_v1(other) != bundle


@pytest.mark.parametrize(
    "kind", ["different_backbone", "reordered_corpus", "different_window", "nonfresh"]
)
def test_incompatible_restore_fails_before_any_target_mutation(kind: str) -> None:
    source = _run()
    source.step()
    bundle = build_scientific_checkpoint_v1(source)
    target = _run(
        model_seed=1908 if kind == "different_backbone" else 1907,
        reverse=kind == "reordered_corpus",
        minimum=2 if kind == "different_window" else 1,
    )
    if kind == "nonfresh":
        target.step()
    before = deepcopy(target._optimizer._model.state_dict())
    steps = target.completed_steps
    with pytest.raises(IrisScientificCheckpointError):
        restore_scientific_checkpoint_v1(bundle, run=target)
    assert not target.poisoned and target.completed_steps == steps
    assert all(
        torch.equal(value, target._optimizer._model.state_dict()[key])
        for key, value in before.items()
    )


@pytest.mark.parametrize(
    "kind",
    ["state_hash", "manifest_hash", "truncated", "extra_file", "duplicate_json", "wrong_path"],
)
def test_corrupt_checkpoint_files_and_manifests_are_rejected(tmp_path: Path, kind: str) -> None:
    bundle = build_scientific_checkpoint_v1(_run())
    location = write_scientific_checkpoint_v1(bundle, root=tmp_path / "checkpoints")
    state = location / "state.safetensors"
    manifest = location / "manifest.json"
    if kind == "state_hash":
        data = state.read_bytes()
        state.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif kind == "manifest_hash":
        manifest.write_bytes(
            manifest.read_bytes().replace(bundle.manifest.sha256.encode(), b"0" * 64)
        )
    elif kind == "truncated":
        state.write_bytes(state.read_bytes()[:8])
    elif kind == "extra_file":
        (location / "extra.txt").write_text("invalid", encoding="utf-8")
    elif kind == "duplicate_json":
        content = manifest.read_bytes()
        manifest.write_bytes(b'{"sha256":"' + bundle.manifest.sha256.encode() + b'",' + content[1:])
    else:
        destination = location.parent / ("0" * 64)
        location.rename(destination)
        location = destination
    with pytest.raises(IrisScientificCheckpointError):
        read_scientific_checkpoint_v1(location)


def test_partial_publish_cleans_only_owned_staging_and_never_overwrites(tmp_path: Path) -> None:
    bundle = build_scientific_checkpoint_v1(_run())
    root = tmp_path / "checkpoints"
    root.mkdir()
    unrelated = root / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    from iris.application import scientific_checkpoint as module
    from iris.application.training_checkpoint import _write_exact_file

    original = _write_exact_file

    def fail_manifest(path: Path, content: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("PRIVATE failure")
        original(path, content)

    with (
        patch.object(module, "_write_exact_file", side_effect=fail_manifest),
        pytest.raises(IrisScientificCheckpointError),
    ):
        write_scientific_checkpoint_v1(bundle, root=root)
    assert tuple(root.iterdir()) == (unrelated,)
    assert unrelated.read_text(encoding="utf-8") == "keep"
    path = write_scientific_checkpoint_v1(bundle, root=root)
    (path / "manifest.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(IrisScientificCheckpointError):
        write_scientific_checkpoint_v1(bundle, root=root)
    assert (path / "manifest.json").read_text(encoding="utf-8") == "corrupted"


def test_run_window_failure_does_not_advance_cursor_or_runtime() -> None:
    run = _run(minimum=10000)
    before = build_scientific_checkpoint_v1(run)
    with pytest.raises(IrisScientificRunError):
        run.step()
    assert not run.poisoned and build_scientific_checkpoint_v1(run) == before


def test_outside_optimizer_or_backbone_update_is_rejected() -> None:
    run = _run()
    run._optimizer.step(_rows()[:1], terminal_token_id=1)
    with pytest.raises(IrisScientificRunError):
        run.step()
    with pytest.raises(IrisScientificCheckpointError):
        build_scientific_checkpoint_v1(run)
    run = _run()
    with torch.no_grad():
        next(run._backbone._model.parameters()).add_(0.1)
    with pytest.raises(IrisScientificRunError):
        run.step()


def test_invalid_bundle_never_mutates_target() -> None:
    target = _run()
    bundle = build_scientific_checkpoint_v1(target)
    invalid = IrisScientificCheckpointBundleV1(bundle.manifest, bundle.state_bytes[:-1])
    with pytest.raises(IrisScientificCheckpointError):
        restore_scientific_checkpoint_v1(invalid, run=target)
    assert build_scientific_checkpoint_v1(target) == bundle


def test_failure_after_copy_poisoning_preserves_caller_runtime() -> None:
    source = _run()
    source.step()
    bundle = build_scientific_checkpoint_v1(source)
    target = _run()
    original = target._validate_boundary
    calls = 0
    ambient = capture_runtime_state(torch.device("cpu"))

    def fail_after_copy() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("PRIVATE post-copy failure")
        original()

    with (
        patch.object(target, "_validate_boundary", side_effect=fail_after_copy),
        pytest.raises(IrisScientificCheckpointError) as caught,
    ):
        restore_scientific_checkpoint_v1(bundle, run=target)
    assert "PRIVATE" not in str(caught.value) and calls == 3
    assert target.poisoned and target._optimizer.poisoned
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    with pytest.raises(IrisScientificRunError):
        target.step()


@pytest.mark.parametrize("kind", ["buffer", "negative_moment", "invalid_rng"])
def test_rehashed_but_semantically_invalid_tensor_state_rejected_before_copy(kind: str) -> None:
    source = _run()
    source.step()
    bundle = build_scientific_checkpoint_v1(source)
    tensors = load_tensors(bundle.state_bytes)
    if kind == "buffer":
        tensors["model/configuration_identity"][0] ^= 1
    elif kind == "negative_moment":
        next(value for name, value in tensors.items() if name.endswith("/exp_avg_sq")).fill_(-1)
    else:
        tensors["runtime/cpu"].zero_()
    state = save_tensors(
        dict(sorted(tensors.items())), metadata={"format": "iris.scientific-checkpoint.v1"}
    )
    payload = bundle.manifest.payload.model_copy(
        update={"state_sha256": bytes_sha256(state), "state_size_bytes": len(state)}
    )
    manifest = IrisScientificCheckpointManifestV1(
        sha256=domain_sha256(
            "iris/scientific-checkpoint/v1/payload",
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    invalid = IrisScientificCheckpointBundleV1(manifest, state)
    target = _run()
    before = build_scientific_checkpoint_v1(target)
    with pytest.raises(IrisScientificCheckpointError):
        restore_scientific_checkpoint_v1(invalid, run=target)
    assert not target.poisoned and build_scientific_checkpoint_v1(target) == before


@pytest.mark.parametrize(
    "kind",
    [
        "bool_steps",
        "bad_hash",
        "negative_cursor",
        "duplicate_name",
        "bad_python_position",
        "nonfinite_gaussian",
        "bad_tensor_shape",
    ],
)
def test_persisted_schema_rejects_noncanonical_state(kind: str) -> None:
    payload = build_scientific_checkpoint_v1(_run()).manifest.payload.model_dump(mode="python")
    if kind == "bool_steps":
        payload["completed_steps"] = False
    elif kind == "bad_hash":
        payload["configuration_sha256"] = "A" * 64
    elif kind == "negative_cursor":
        payload["examples_seen"] = -1
    elif kind == "duplicate_name":
        payload["parameter_names"] = (payload["parameter_names"][0],) * len(
            payload["parameter_names"]
        )
    elif kind == "bad_python_position":
        payload["runtime"]["python_state"] = (*payload["runtime"]["python_state"][:-1], 625)
    elif kind == "nonfinite_gaussian":
        payload["runtime"]["python_gaussian"] = float("nan")
    else:
        payload["tensors"][0]["shape"] = (-1,)
    with pytest.raises(ValidationError):
        ScientificCheckpointPayloadV1.model_validate(payload)


def test_tokenizer_mutation_and_caller_corpus_aliases_are_detected_or_isolated() -> None:
    model, tokenizer, _ = _build()
    rows = _rows()[:5]
    run = IrisScientificRunV1(
        optimizer=IrisScientificOptimizerV1(model=model, policy=_policy()),
        examples=rows,
        terminal_token_id=1,
        seed=901,
    )
    original_fact = run._rows[0].hidden_facts["operation"]
    rows[0].hidden_facts["operation"] = "invalid caller mutation"
    assert run._rows[0].hidden_facts["operation"] == original_fact
    tokenizer.padding_side = "right"
    with pytest.raises(IrisScientificRunError):
        run.step()


def test_selected_corpus_order_continues_without_dropping_tail_rows() -> None:
    run = _run(minimum=70)
    ids: list[str] = []
    original = run._optimizer.step

    def observe(examples: Any, **kwargs: Any) -> Any:
        ids.extend(row.id for row in examples)
        return original(examples, **kwargs)

    with patch.object(run._optimizer, "step", side_effect=observe):
        for _ in range(3):
            run.step()
    assert ids == [run._rows[index % len(run._rows)].id for index in range(run.examples_seen)]
    assert len(ids) > len(run._rows)


def test_frozen_identity_includes_nonpersistent_buffers_and_eval_modes() -> None:
    model, _, frozen = _build()
    backbone = model.teacher.backbone
    buffer = torch.tensor(1.0)
    frozen.register_buffer("test_runtime_buffer", buffer, persistent=False)
    before = backbone.checkpoint_runtime_sha256
    buffer.add_(1.0)
    assert backbone.checkpoint_runtime_sha256 != before
    next(child for child in frozen.modules() if child is not frozen).train(True)
    with pytest.raises(RuntimeError):
        _ = backbone.checkpoint_runtime_sha256


def test_attention_implementation_omitted_by_hf_config_still_binds_run() -> None:
    run = _run()
    config = run._backbone._model.config
    before_json = config.to_dict()
    before_identity = run._backbone.checkpoint_runtime_sha256
    config._attn_implementation = "eager" if config._attn_implementation != "eager" else "sdpa"
    assert config.to_dict() == before_json  # The installed Transformers deliberately omits it.
    assert run._backbone.checkpoint_runtime_sha256 != before_identity
    with pytest.raises(IrisScientificRunError):
        run.step()


def test_float32_backend_policy_drift_is_rejected_before_execution() -> None:
    run = _run()
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high" if previous != "high" else "highest")
        with pytest.raises(IrisScientificRunError):
            run.step()
    finally:
        torch.set_float32_matmul_precision(previous)
    assert not run.poisoned
