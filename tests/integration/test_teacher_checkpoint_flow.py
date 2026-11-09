"""Offline end-to-end checkpoint, persistence, tamper, and resume tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch

import iris.application.training_checkpoint as checkpoint_module
from iris.application import prepare_teacher_forcing_input
from iris.application.errors import IrisTeacherCheckpointError
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    build_teacher_checkpoint_v1,
    read_teacher_checkpoint_v1,
    restore_teacher_checkpoint_v1,
    validate_teacher_checkpoint_bundle_v1,
    write_teacher_checkpoint_v1,
)
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerSnapshot,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_teacher_forcing_sidecar import (
    _build_offline_model,
    _TinyFrozenCausalLM,
    _TinyTokenizer,
)

pytestmark = pytest.mark.integration


def _optimizer_config(*, max_gradient_norm: float = 1.0) -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=max_gradient_norm,
    )


def _build_with_owner(
    *,
    seed: int = 1907,
    config: IrisAdamWConfig | None = None,
) -> tuple[IrisTeacherForcingModel, IrisTeacherOptimizer, _TinyTokenizer, _TinyFrozenCausalLM]:
    torch.manual_seed(seed)
    untyped_teacher, tokenizer, frozen_model = _build_offline_model()
    teacher = cast(IrisTeacherForcingModel, untyped_teacher)
    optimizer = IrisTeacherOptimizer(
        model=teacher,
        config=config or _optimizer_config(),
    )
    return teacher, optimizer, tokenizer, frozen_model


def _example(teacher: IrisTeacherForcingModel) -> IrisTeacherForcingInput:
    return prepare_teacher_forcing_input(
        backbone=teacher._backbone,
        prompt="offline checkpoint prompt",
        target_text="longer target",
        terminal_token_id=15,
    )


def _trained_bundle(
    *,
    seed: int = 1907,
) -> tuple[
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
    _TinyFrozenCausalLM,
    IrisTeacherCheckpointBundleV1,
]:
    teacher, optimizer, _, frozen_model = _build_with_owner(seed=seed)
    optimizer.step(_example(teacher))
    bundle = build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)
    return teacher, optimizer, frozen_model, bundle


def _assert_optimizer_snapshots_equal(
    left: IrisTeacherOptimizerSnapshot,
    right: IrisTeacherOptimizerSnapshot,
) -> None:
    assert left.completed_steps == right.completed_steps
    assert left.config == right.config
    assert left.parameter_names == right.parameter_names
    assert len(left.parameter_states) == len(right.parameter_states)
    for left_state, right_state in zip(
        left.parameter_states,
        right.parameter_states,
        strict=True,
    ):
        assert left_state.parameter_name == right_state.parameter_name
        assert left_state.step == right_state.step
        assert torch.equal(left_state.exp_avg, right_state.exp_avg)
        assert torch.equal(left_state.exp_avg_sq, right_state.exp_avg_sq)


def test_checkpoint_capture_is_deterministic_hash_only_and_sidecar_exclusive() -> None:
    with torch.random.fork_rng(devices=[]):
        teacher, optimizer, frozen_model, first = _trained_bundle()
        second = build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)

    assert first.manifest == second.manifest
    assert first.state_bytes == second.state_bytes
    assert first.manifest.payload.optimizer.completed_steps == 1
    assert first.manifest.payload.model.state_names == tuple(teacher.state_dict())
    assert first.manifest.payload.state_file.tensor_count == (
        len(teacher.state_dict()) + 2 * len(optimizer.checkpoint_snapshot().parameter_states)
    )
    assert all(
        not name.startswith(("embedding", "lm_head"))
        for name in first.manifest.payload.model.state_names
    )
    manifest_bytes = canonical_json_file_bytes(first.manifest.model_dump(mode="json"))
    for private_text in (b"offline checkpoint prompt", b"longer target"):
        assert private_text not in manifest_bytes
        assert private_text not in first.state_bytes
    assert all(not parameter.requires_grad for parameter in frozen_model.parameters())
    assert all(parameter.grad is None for parameter in frozen_model.parameters())


def test_public_bundle_validator_owns_verified_state_and_rejects_drift() -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()

    validated = validate_teacher_checkpoint_bundle_v1(bundle)

    assert validated == bundle
    assert validated is not bundle
    mutated = IrisTeacherCheckpointBundleV1(
        manifest=bundle.manifest.model_copy(update={"sha256": "f" * 64}),
        state_bytes=bundle.state_bytes,
    )
    with pytest.raises(IrisTeacherCheckpointError, match="verified"):
        validate_teacher_checkpoint_bundle_v1(mutated)


def test_atomic_write_read_and_idempotent_reuse(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
    root = tmp_path / "checkpoints"

    first = write_teacher_checkpoint_v1(bundle, root=root)
    loaded = read_teacher_checkpoint_v1(first.path)
    second = write_teacher_checkpoint_v1(bundle, root=root)

    assert first.path == root / "teacher" / "v1" / bundle.manifest.sha256
    assert first.payload_sha256 == bundle.manifest.sha256
    assert first.state_file_sha256 == bundle.manifest.payload.state_file.sha256
    assert first.state_size_bytes == len(bundle.state_bytes)
    assert first.reused_existing is False
    assert second.path == first.path
    assert second.reused_existing is True
    assert loaded.manifest == bundle.manifest
    assert loaded.state_bytes == bundle.state_bytes
    assert sorted(path.name for path in first.path.iterdir()) == [
        "manifest.json",
        "state.safetensors",
    ]
    assert not tuple(first.path.parent.glob(".iris-*.tmp"))


def test_restore_reproduces_sidecar_moments_and_the_next_real_update() -> None:
    seed = 1907
    with torch.random.fork_rng(devices=[]):
        source, source_optimizer, source_backbone, bundle = _trained_bundle(seed=seed)
        source_backbone_before = {
            name: parameter.detach().clone()
            for name, parameter in source_backbone.named_parameters()
        }
        target, target_optimizer, _, target_backbone = _build_with_owner(seed=seed)
        with torch.no_grad():
            next(target.parameters()).add_(7.0)
        target_before_restore = {
            name: tensor.detach().clone() for name, tensor in target.state_dict().items()
        }

        result = restore_teacher_checkpoint_v1(
            bundle,
            model=target,
            optimizer=target_optimizer,
        )

        assert result.payload_sha256 == bundle.manifest.sha256
        assert result.sidecar_state_sha256 == bundle.manifest.payload.model.sha256
        assert result.completed_steps == 1
        assert any(
            not torch.equal(tensor, target_before_restore[name])
            for name, tensor in target.state_dict().items()
        )
        for name, tensor in source.state_dict().items():
            assert torch.equal(tensor, target.state_dict()[name])
        _assert_optimizer_snapshots_equal(
            source_optimizer.checkpoint_snapshot(),
            target_optimizer.checkpoint_snapshot(),
        )

        source_output = source_optimizer.step(_example(source))
        target_output = target_optimizer.step(_example(target))

        assert target_output == source_output
        for name, tensor in source.state_dict().items():
            assert torch.equal(tensor, target.state_dict()[name])
        _assert_optimizer_snapshots_equal(
            source_optimizer.checkpoint_snapshot(),
            target_optimizer.checkpoint_snapshot(),
        )
        for name, parameter in source_backbone.named_parameters():
            assert torch.equal(parameter, source_backbone_before[name])
            assert parameter.grad is None
        for (source_name, source_parameter), (target_name, target_parameter) in zip(
            source_backbone.named_parameters(),
            target_backbone.named_parameters(),
            strict=True,
        ):
            assert source_name == target_name
            assert torch.equal(source_parameter, target_parameter)
            assert target_parameter.grad is None


def test_zero_step_checkpoint_restores_a_pristine_owner() -> None:
    with torch.random.fork_rng(devices=[]):
        source, source_optimizer, _, _ = _build_with_owner(seed=1907)
        bundle = build_teacher_checkpoint_v1(model=source, optimizer=source_optimizer)
        target, target_optimizer, _, _ = _build_with_owner(seed=1907)
        with torch.no_grad():
            next(target.parameters()).mul_(0.0)

        result = restore_teacher_checkpoint_v1(
            bundle,
            model=target,
            optimizer=target_optimizer,
        )
        source_output = source_optimizer.step(_example(source))
        target_output = target_optimizer.step(_example(target))

    assert result.completed_steps == 0
    assert bundle.manifest.payload.optimizer.parameter_states == ()
    assert target_output == source_output
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name])


def test_restore_rejects_optimizer_policy_mismatch_and_rolls_back_model() -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
        target, optimizer, _, _ = _build_with_owner(config=_optimizer_config(max_gradient_norm=2.0))
        before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

        with pytest.raises(IrisTeacherCheckpointError):
            restore_teacher_checkpoint_v1(bundle, model=target, optimizer=optimizer)

    assert optimizer.completed_steps == 0
    assert not optimizer._optimizer.state
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_restore_rejects_nonpristine_optimizer_and_rolls_back_model() -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
        target, optimizer, _, _ = _build_with_owner()
        optimizer.step(_example(target))
        before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
        snapshot_before = optimizer.checkpoint_snapshot()

        with pytest.raises(IrisTeacherCheckpointError):
            restore_teacher_checkpoint_v1(bundle, model=target, optimizer=optimizer)

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name])
    _assert_optimizer_snapshots_equal(snapshot_before, optimizer.checkpoint_snapshot())


def test_restore_rolls_back_sidecar_if_optimizer_install_raises() -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
        target, optimizer, _, _ = _build_with_owner()
        with torch.no_grad():
            next(target.parameters()).add_(9.0)
        before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

        with (
            patch.object(
                IrisTeacherOptimizer,
                "restore_checkpoint_snapshot",
                side_effect=RuntimeError("synthetic restore failure"),
            ),
            pytest.raises(IrisTeacherCheckpointError),
        ):
            restore_teacher_checkpoint_v1(bundle, model=target, optimizer=optimizer)

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert optimizer.completed_steps == 0
    assert not optimizer._optimizer.state


def test_build_rejects_an_optimizer_owned_by_another_model() -> None:
    with torch.random.fork_rng(devices=[]):
        first, optimizer, _, _ = _build_with_owner(seed=1)
        second, _, _, _ = _build_with_owner(seed=1)

        with pytest.raises(IrisTeacherCheckpointError):
            build_teacher_checkpoint_v1(model=second, optimizer=optimizer)

    assert optimizer.model is first


def test_build_detects_backbone_configuration_mutation() -> None:
    with torch.random.fork_rng(devices=[]):
        teacher, optimizer, _, _ = _build_with_owner()
        backbone = teacher._backbone
        backbone._config.local_files_only = False

        with pytest.raises(IrisTeacherCheckpointError):
            build_teacher_checkpoint_v1(model=teacher, optimizer=optimizer)


@pytest.mark.parametrize("tamper", ["state", "manifest", "duplicate", "extra", "dirname"])
def test_reader_rejects_tampered_checkpoint_directory(tmp_path: Path, tamper: str) -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
    result = write_teacher_checkpoint_v1(bundle, root=tmp_path / "checkpoints")
    path = result.path
    if tamper == "state":
        state_path = path / "state.safetensors"
        content = bytearray(state_path.read_bytes())
        content[-1] ^= 1
        state_path.write_bytes(bytes(content))
    elif tamper == "manifest":
        manifest_path = path / "manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes()[:-1] + b" \n")
    elif tamper == "duplicate":
        manifest_path = path / "manifest.json"
        manifest_content = manifest_path.read_bytes()
        manifest_path.write_bytes(
            b'{"sha256":"' + bundle.manifest.sha256.encode() + b'",' + manifest_content[1:]
        )
    elif tamper == "extra":
        (path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        renamed = path.with_name("b" * 64)
        path.rename(renamed)
        path = renamed

    with pytest.raises(IrisTeacherCheckpointError):
        read_teacher_checkpoint_v1(path)


def test_writer_rejects_conflicting_existing_directory_without_temp_leak(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
    root = tmp_path / "checkpoints"
    result = write_teacher_checkpoint_v1(bundle, root=root)
    (result.path / "state.safetensors").write_bytes(b"conflict")

    with pytest.raises(IrisTeacherCheckpointError, match="conflicts"):
        write_teacher_checkpoint_v1(bundle, root=root)

    assert not tuple(result.path.parent.glob(".iris-*.tmp"))


def test_writer_cleans_temporary_directory_after_second_file_failure(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
    root = tmp_path / "checkpoints"
    original_write = checkpoint_module._write_exact_file

    def failing_write(path: Path, content: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("synthetic manifest failure")
        original_write(path, content)

    with (
        patch.object(checkpoint_module, "_write_exact_file", side_effect=failing_write),
        pytest.raises(IrisTeacherCheckpointError),
    ):
        write_teacher_checkpoint_v1(bundle, root=root)

    parent = root / "teacher" / "v1"
    assert parent.is_dir()
    assert not tuple(parent.iterdir())


def test_writer_revalidates_mutated_in_memory_manifest(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        _, _, _, bundle = _trained_bundle()
    mutated_manifest = bundle.manifest.model_copy(update={"sha256": "b" * 64})
    mutated = IrisTeacherCheckpointBundleV1(
        manifest=mutated_manifest,
        state_bytes=bundle.state_bytes,
    )

    with pytest.raises(IrisTeacherCheckpointError):
        write_teacher_checkpoint_v1(mutated, root=tmp_path / "checkpoints")
