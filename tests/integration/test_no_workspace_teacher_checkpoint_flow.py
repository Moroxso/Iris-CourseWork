from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import ValidationError

import iris.application as application
import iris.artifacts as artifacts
from iris.application.errors import IrisTeacherCheckpointError
from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
    build_no_workspace_teacher_checkpoint_v2,
    read_no_workspace_teacher_checkpoint_v2,
    restore_no_workspace_teacher_checkpoint_v2,
    validate_no_workspace_teacher_checkpoint_bundle_v2,
    write_no_workspace_teacher_checkpoint_v2,
)
from iris.application.training_checkpoint import build_teacher_checkpoint_v1
from iris.artifacts.checkpoint_v2 import (
    IrisNoWorkspaceTeacherCheckpointEnvelopeV2,
    NoWorkspaceTeacherCheckpointModelIdentityV2,
)
from iris.training import (
    IrisAdamWConfig,
    IrisNoWorkspaceTeacherForcingModel,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
)
from tests.unit.test_no_workspace_teacher_forcing import make_harness as make_b_harness
from tests.unit.test_teacher_forcing import make_harness as make_d_harness
from tests.unit.test_teacher_forcing import nominal_input

PROFILE_SET_V3_SHA256 = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"


def _config() -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )


def _owner(
    *,
    stepped: bool,
) -> tuple[IrisNoWorkspaceTeacherForcingModel, IrisTeacherOptimizer]:
    model = make_b_harness().model
    optimizer = IrisTeacherOptimizer(model=model, config=_config())
    if stepped:
        optimizer.step(nominal_input())
    return model, optimizer


def test_public_api_and_schema_are_explicitly_arm_b_without_workspace() -> None:
    model, optimizer = _owner(stepped=False)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=model,
        optimizer=optimizer,
    )
    assert application.IrisNoWorkspaceTeacherCheckpointBundleV2 is (
        IrisNoWorkspaceTeacherCheckpointBundleV2
    )
    assert application.build_no_workspace_teacher_checkpoint_v2 is (
        build_no_workspace_teacher_checkpoint_v2
    )
    assert artifacts.IrisNoWorkspaceTeacherCheckpointEnvelopeV2 is (
        IrisNoWorkspaceTeacherCheckpointEnvelopeV2
    )
    assert bundle.manifest.schema_identity.version == 2
    identity = bundle.manifest.payload.model
    assert type(identity) is NoWorkspaceTeacherCheckpointModelIdentityV2
    assert identity.arm_id == "B"
    assert identity.arm_kind == "llm_modules_no_workspace"
    assert identity.teacher_graph == ("independent_recurrent_modules_zero_memory_read_teacher_v1")
    assert identity.execution_profile_set_sha256 == PROFILE_SET_V3_SHA256
    assert identity.cognitive_cycles == 3
    assert identity.latent_token_count == 3
    assert identity.component_names == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "cognitive_modules",
        "module_latent_bridge",
        "sequence_assembler",
    )
    model_payload = identity.model_dump(mode="json")
    assert "workspace" not in model_payload
    assert "workspace_configuration_json" not in model_payload
    assert bundle.manifest.payload.optimizer.completed_steps == 0
    assert bundle.manifest.payload.optimizer.parameter_states == ()


def test_stepped_checkpoint_write_read_validate_and_idempotent_reuse(tmp_path: Path) -> None:
    model, optimizer = _owner(stepped=True)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=model,
        optimizer=optimizer,
    )
    validated = validate_no_workspace_teacher_checkpoint_bundle_v2(bundle)
    assert validated.manifest == bundle.manifest
    assert validated.state_bytes == bundle.state_bytes
    first = write_no_workspace_teacher_checkpoint_v2(bundle, root=tmp_path)
    assert first.path == tmp_path / "teacher" / "v2" / bundle.manifest.sha256
    assert not first.reused_existing
    assert first.path.is_dir()
    assert tuple(sorted(path.name for path in first.path.iterdir())) == (
        "manifest.json",
        "state.safetensors",
    )
    read = read_no_workspace_teacher_checkpoint_v2(first.path)
    assert read == bundle
    second = write_no_workspace_teacher_checkpoint_v2(bundle, root=tmp_path)
    assert second == type(second)(
        path=first.path,
        payload_sha256=first.payload_sha256,
        state_file_sha256=first.state_file_sha256,
        state_size_bytes=first.state_size_bytes,
        reused_existing=True,
    )


def test_checkpoint_restores_weights_and_adamw_into_fresh_arm_b_owner() -> None:
    source_model, source_optimizer = _owner(stepped=True)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=source_model,
        optimizer=source_optimizer,
    )
    source_state = {
        name: tensor.detach().clone() for name, tensor in source_model.state_dict().items()
    }
    source_snapshot = source_optimizer.checkpoint_snapshot()

    target_model, target_optimizer = _owner(stepped=False)
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.add_(7.0)
    result = restore_no_workspace_teacher_checkpoint_v2(
        bundle,
        model=target_model,
        optimizer=target_optimizer,
    )
    assert result.payload_sha256 == bundle.manifest.sha256
    assert result.sidecar_state_sha256 == bundle.manifest.payload.model.sha256
    assert result.completed_steps == 1
    assert target_optimizer.completed_steps == 1
    assert all(
        torch.equal(target_model.state_dict()[name], expected)
        for name, expected in source_state.items()
    )
    restored_snapshot = target_optimizer.checkpoint_snapshot()
    assert restored_snapshot.completed_steps == source_snapshot.completed_steps
    assert restored_snapshot.parameter_names == source_snapshot.parameter_names
    for actual, expected in zip(
        restored_snapshot.parameter_states,
        source_snapshot.parameter_states,
        strict=True,
    ):
        assert actual.parameter_name == expected.parameter_name
        assert actual.step == expected.step
        assert torch.equal(actual.exp_avg, expected.exp_avg)
        assert torch.equal(actual.exp_avg_sq, expected.exp_avg_sq)


def test_v2_rejects_d_target_and_v1_rejects_b_source_before_mutation() -> None:
    b_model, b_optimizer = _owner(stepped=False)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=b_model,
        optimizer=b_optimizer,
    )
    d_model = make_d_harness().model
    d_optimizer = IrisTeacherOptimizer(model=d_model, config=_config())
    d_before = {name: tensor.detach().clone() for name, tensor in d_model.state_dict().items()}
    with pytest.raises(IrisTeacherCheckpointError, match="could not be restored") as captured:
        restore_no_workspace_teacher_checkpoint_v2(
            bundle,
            model=cast(IrisNoWorkspaceTeacherForcingModel, d_model),
            optimizer=d_optimizer,
        )
    assert captured.value.__cause__ is not None
    assert "exactly arm-B" in str(captured.value.__cause__)
    assert all(
        torch.equal(d_model.state_dict()[name], expected) for name, expected in d_before.items()
    )
    with pytest.raises(IrisTeacherCheckpointError, match="could not be built") as captured:
        build_teacher_checkpoint_v1(
            model=cast(IrisTeacherForcingModel, b_model),
            optimizer=b_optimizer,
        )
    assert captured.value.__cause__ is not None
    assert "IrisTeacherForcingModel" in str(captured.value.__cause__)


def test_manifest_cannot_represent_workspace_or_d_arm() -> None:
    model, optimizer = _owner(stepped=False)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=model,
        optimizer=optimizer,
    )
    values = bundle.manifest.payload.model.model_dump(mode="python")
    values["workspace"] = {"name": "forbidden"}
    with pytest.raises(ValidationError):
        NoWorkspaceTeacherCheckpointModelIdentityV2.model_validate(values)
    values = bundle.manifest.payload.model.model_dump(mode="python")
    values["arm_id"] = "D"
    with pytest.raises(ValidationError):
        NoWorkspaceTeacherCheckpointModelIdentityV2.model_validate(values)


def test_tampered_state_bytes_and_profile_identity_fail_closed() -> None:
    model, optimizer = _owner(stepped=False)
    bundle = build_no_workspace_teacher_checkpoint_v2(
        model=model,
        optimizer=optimizer,
    )
    tampered = IrisNoWorkspaceTeacherCheckpointBundleV2(
        manifest=bundle.manifest,
        state_bytes=bundle.state_bytes[:-1] + bytes([bundle.state_bytes[-1] ^ 1]),
    )
    with pytest.raises(IrisTeacherCheckpointError, match="could not be verified"):
        validate_no_workspace_teacher_checkpoint_bundle_v2(tampered)
    values = bundle.manifest.payload.model.model_dump(mode="python")
    values["execution_profile_set_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        NoWorkspaceTeacherCheckpointModelIdentityV2.model_validate(values)
