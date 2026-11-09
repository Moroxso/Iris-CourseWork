"""End-to-end Arm-C checkpoint v3 identity, persistence, and exact restore."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from iris.application.errors import IrisTeacherCheckpointError
from iris.application.no_workspace_training_checkpoint import (
    build_no_workspace_teacher_checkpoint_v2,
)
from iris.application.shared_buffer_training_checkpoint import (
    IrisSharedBufferTeacherCheckpointBundleV3,
    build_shared_buffer_teacher_checkpoint_v3,
    read_shared_buffer_teacher_checkpoint_v3,
    restore_shared_buffer_teacher_checkpoint_v3,
    validate_shared_buffer_teacher_checkpoint_bundle_v3,
    write_shared_buffer_teacher_checkpoint_v3,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.training_checkpoint import build_teacher_checkpoint_v1
from iris.artifacts.checkpoint_v3 import (
    IrisSharedBufferTeacherCheckpointEnvelopeV3,
    SharedBufferTeacherCheckpointModelIdentityV3,
)
from iris.training import (
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerSnapshot,
)
from tests.integration.test_teacher_checkpoint_flow import (
    _build_with_owner as build_d_owner,
)
from tests.unit.test_no_workspace_teacher_forcing import make_harness as make_b_harness
from tests.unit.test_shared_buffer_teacher_forcing import (
    ArmCBackbone,
)
from tests.unit.test_shared_buffer_teacher_forcing import (
    make_harness as make_c_harness,
)
from tests.unit.test_shared_buffer_training_optimization import _config
from tests.unit.test_teacher_forcing import (
    EventControl,
    nominal_input,
)

pytestmark = pytest.mark.integration

PROFILE_SET_V4_SHA256 = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
POLICY_SHA256 = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
PROTOCOL_SHA256 = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
    "sequence_assembler",
)


def _owner(
    *,
    stepped: bool,
) -> tuple[IrisSharedBufferTeacherForcingModel, IrisTeacherOptimizer]:
    model = make_c_harness().model
    optimizer = IrisTeacherOptimizer(model=model, config=_config())
    if stepped:
        optimizer.step(nominal_input())
    return model, optimizer


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _state_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(tensor._version for tensor in model.state_dict(keep_vars=True).values())


def _assert_state_equal(
    model: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = model.state_dict()
    assert tuple(actual) == tuple(expected)
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected.items())


def _assert_optimizer_snapshots_equal(
    actual: IrisTeacherOptimizerSnapshot,
    expected: IrisTeacherOptimizerSnapshot,
) -> None:
    assert actual.completed_steps == expected.completed_steps
    assert actual.config == expected.config
    assert actual.parameter_names == expected.parameter_names
    assert len(actual.parameter_states) == len(expected.parameter_states)
    for actual_state, expected_state in zip(
        actual.parameter_states,
        expected.parameter_states,
        strict=True,
    ):
        assert actual_state.parameter_name == expected_state.parameter_name
        assert actual_state.step == expected_state.step
        assert torch.equal(actual_state.exp_avg, expected_state.exp_avg)
        assert torch.equal(actual_state.exp_avg_sq, expected_state.exp_avg_sq)


def test_v3_schema_binds_exact_arm_c_graph_policy_profile_and_lifecycle() -> None:
    model, optimizer = _owner(stepped=False)
    bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=model,
        optimizer=optimizer,
    )

    assert bundle.manifest.schema_identity.version == 3
    identity = bundle.manifest.payload.model
    assert type(identity) is SharedBufferTeacherCheckpointModelIdentityV3
    assert identity.arm_id == "C"
    assert identity.arm_kind == "llm_modules_unlimited_shared_buffer"
    assert identity.teacher_graph == (
        "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
    )
    assert identity.execution_profile_set_sha256 == PROFILE_SET_V4_SHA256
    assert identity.shared_buffer_policy_sha256 == POLICY_SHA256
    assert identity.protocol_declaration_payload_sha256 == PROTOCOL_SHA256
    assert identity.cognitive_cycles == 3
    assert identity.pre_update_read_count == 3
    assert identity.final_output_read_count == 1
    assert identity.entries_per_cycle == 3
    assert identity.final_entry_count == 9
    assert identity.latent_token_count == 3
    assert identity.component_names == COMPONENT_NAMES
    assert identity.state_names == tuple(model.state_dict())
    assert "workspace" not in identity.model_dump(mode="json")
    assert bundle.manifest.payload.optimizer.completed_steps == 0
    assert bundle.manifest.payload.optimizer.parameter_states == ()

    values = identity.model_dump(mode="python")
    values["execution_profile_set_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        SharedBufferTeacherCheckpointModelIdentityV3.model_validate(values)
    values = identity.model_dump(mode="python")
    values["arm_id"] = "B"
    with pytest.raises(ValidationError):
        SharedBufferTeacherCheckpointModelIdentityV3.model_validate(values)


def test_stepped_v3_write_read_validate_and_idempotent_reuse(tmp_path: Path) -> None:
    model, optimizer = _owner(stepped=True)
    bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=model,
        optimizer=optimizer,
    )

    validated = validate_shared_buffer_teacher_checkpoint_bundle_v3(bundle)
    assert validated == bundle
    assert validated is not bundle
    first = write_shared_buffer_teacher_checkpoint_v3(bundle, root=tmp_path)
    assert first.path == tmp_path / "teacher" / "v3" / bundle.manifest.sha256
    assert first.reused_existing is False
    assert tuple(sorted(path.name for path in first.path.iterdir())) == (
        "manifest.json",
        "state.safetensors",
    )
    assert read_shared_buffer_teacher_checkpoint_v3(first.path) == bundle
    second = write_shared_buffer_teacher_checkpoint_v3(bundle, root=tmp_path)
    assert second.path == first.path
    assert second.payload_sha256 == first.payload_sha256
    assert second.state_file_sha256 == first.state_file_sha256
    assert second.state_size_bytes == first.state_size_bytes
    assert second.reused_existing is True


def test_v3_restores_sidecar_and_adamw_into_fresh_exact_arm_c_owner() -> None:
    source_model, source_optimizer = _owner(stepped=True)
    bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=source_model,
        optimizer=source_optimizer,
    )
    source_state = _state_copy(source_model)
    source_snapshot = source_optimizer.checkpoint_snapshot()

    target_model, target_optimizer = _owner(stepped=False)
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.add_(7.0)
    result = restore_shared_buffer_teacher_checkpoint_v3(
        bundle,
        model=target_model,
        optimizer=target_optimizer,
    )

    assert result.payload_sha256 == bundle.manifest.sha256
    assert result.sidecar_state_sha256 == bundle.manifest.payload.model.sha256
    assert result.completed_steps == 1
    _assert_state_equal(target_model, source_state)
    _assert_optimizer_snapshots_equal(
        target_optimizer.checkpoint_snapshot(),
        source_snapshot,
    )


def test_bad_c_identity_and_nonfresh_owner_fail_before_model_mutation() -> None:
    source_model, source_optimizer = _owner(stepped=True)
    bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=source_model,
        optimizer=source_optimizer,
    )

    target_model, target_optimizer = _owner(stepped=True)
    state_before = _state_copy(target_model)
    versions_before = _state_versions(target_model)
    snapshot_before = target_optimizer.checkpoint_snapshot()
    with pytest.raises(IrisTeacherCheckpointError, match="could not be restored"):
        restore_shared_buffer_teacher_checkpoint_v3(
            bundle,
            model=target_model,
            optimizer=target_optimizer,
        )
    _assert_state_equal(target_model, state_before)
    assert _state_versions(target_model) == versions_before
    _assert_optimizer_snapshots_equal(
        target_optimizer.checkpoint_snapshot(),
        snapshot_before,
    )

    source_config = source_model._cognitive_config
    mismatched_model = build_shared_buffer_teacher_forcing_model(
        backbone=ArmCBackbone(EventControl.empty()),
        cognitive_config=source_config.model_copy(update={"name": "identity-mismatch"}),
    )
    mismatched_optimizer = IrisTeacherOptimizer(
        model=mismatched_model,
        config=_config(),
    )
    state_before = _state_copy(mismatched_model)
    versions_before = _state_versions(mismatched_model)
    with pytest.raises(IrisTeacherCheckpointError, match="could not be restored"):
        restore_shared_buffer_teacher_checkpoint_v3(
            bundle,
            model=mismatched_model,
            optimizer=mismatched_optimizer,
        )
    _assert_state_equal(mismatched_model, state_before)
    assert _state_versions(mismatched_model) == versions_before
    assert mismatched_optimizer.completed_steps == 0


def test_v3_rejects_b_v2_and_d_v1_sources_and_targets_before_mutation() -> None:
    c_model, c_optimizer = _owner(stepped=False)
    c_bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=c_model,
        optimizer=c_optimizer,
    )
    c_before = _state_copy(c_model)
    c_versions = _state_versions(c_model)

    b_model = make_b_harness().model
    b_optimizer = IrisTeacherOptimizer(model=b_model, config=_config())
    b_bundle = build_no_workspace_teacher_checkpoint_v2(
        model=b_model,
        optimizer=b_optimizer,
    )
    d_model, d_optimizer, _, _ = build_d_owner()
    d_bundle = build_teacher_checkpoint_v1(model=d_model, optimizer=d_optimizer)

    for foreign_bundle in (b_bundle, d_bundle):
        with pytest.raises(IrisTeacherCheckpointError, match="could not be restored"):
            restore_shared_buffer_teacher_checkpoint_v3(
                cast(IrisSharedBufferTeacherCheckpointBundleV3, foreign_bundle),
                model=c_model,
                optimizer=c_optimizer,
            )
        _assert_state_equal(c_model, c_before)
        assert _state_versions(c_model) == c_versions
        assert c_optimizer.completed_steps == 0

    for foreign_model, foreign_optimizer in (
        (b_model, b_optimizer),
        (d_model, d_optimizer),
    ):
        before = _state_copy(foreign_model)
        versions = _state_versions(foreign_model)
        with pytest.raises(IrisTeacherCheckpointError, match="could not be restored"):
            restore_shared_buffer_teacher_checkpoint_v3(
                c_bundle,
                model=cast(IrisSharedBufferTeacherForcingModel, foreign_model),
                optimizer=foreign_optimizer,
            )
        _assert_state_equal(foreign_model, before)
        assert _state_versions(foreign_model) == versions
        assert foreign_optimizer.completed_steps == 0


def test_v3_tampered_bytes_and_manifest_envelope_fail_closed() -> None:
    model, optimizer = _owner(stepped=False)
    bundle = build_shared_buffer_teacher_checkpoint_v3(
        model=model,
        optimizer=optimizer,
    )
    tampered = IrisSharedBufferTeacherCheckpointBundleV3(
        manifest=bundle.manifest,
        state_bytes=bundle.state_bytes[:-1] + bytes([bundle.state_bytes[-1] ^ 1]),
    )
    with pytest.raises(IrisTeacherCheckpointError, match="could not be verified"):
        validate_shared_buffer_teacher_checkpoint_bundle_v3(tampered)

    bad_manifest = bundle.manifest.model_copy(update={"sha256": "0" * 64})
    assert type(bad_manifest) is IrisSharedBufferTeacherCheckpointEnvelopeV3
    bad_bundle = IrisSharedBufferTeacherCheckpointBundleV3(
        manifest=bad_manifest,
        state_bytes=bundle.state_bytes,
    )
    with pytest.raises(IrisTeacherCheckpointError, match="could not be verified"):
        validate_shared_buffer_teacher_checkpoint_bundle_v3(bad_bundle)


def test_build_requires_exact_arm_c_model_and_its_exact_optimizer_owner() -> None:
    c_model, c_optimizer = _owner(stepped=False)
    b_model = make_b_harness().model
    b_optimizer = IrisTeacherOptimizer(model=b_model, config=_config())

    with pytest.raises(IrisTeacherCheckpointError, match="could not be built"):
        build_shared_buffer_teacher_checkpoint_v3(
            model=cast(IrisSharedBufferTeacherForcingModel, b_model),
            optimizer=b_optimizer,
        )
    with pytest.raises(IrisTeacherCheckpointError, match="could not be built"):
        build_shared_buffer_teacher_checkpoint_v3(
            model=c_model,
            optimizer=b_optimizer,
        )
    assert c_optimizer.completed_steps == 0
    assert b_optimizer.completed_steps == 0
