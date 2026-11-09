"""Offline B/C/D teacher-checkpoint to request-sidecar transfer tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_safetensors

import iris.application.arm_request_checkpoint as request_checkpoint
from iris.application.arm_profiles import (
    arm_execution_profile_set_sha256_v4,
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointProjectionV1,
    IrisArmRequestCheckpointTransferResultV1,
    load_arm_teacher_checkpoint_into_request_v1,
    project_arm_teacher_checkpoint_to_request_v1,
    transfer_arm_teacher_checkpoint_to_request_v1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestCheckpointError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
)
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.request import IrisRequestOrchestrator
from iris.application.shared_buffer_request import CognitiveSharedBufferRequestOrchestrator
from iris.application.shared_buffer_training_checkpoint import (
    IrisSharedBufferTeacherCheckpointBundleV3,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.training_checkpoint import IrisTeacherCheckpointBundleV1
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import BackboneConfig, CognitiveConfig, IrisBootstrapConfig, WorkspaceConfig
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.integration.test_teacher_forcing_sidecar import (
    _backbone_config,
    _cognitive_config,
    _TinyFrozenCausalLM,
    _TinyTokenizer,
    _workspace_config,
)

pytestmark = pytest.mark.integration

_PROFILE_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_SEED = 1907

type _SidecarRequestOrchestrator = (
    CognitiveNoWorkspaceRequestOrchestrator
    | CognitiveSharedBufferRequestOrchestrator
    | IrisRequestOrchestrator
)


def _require_sidecar_orchestrator(orchestrator: object) -> _SidecarRequestOrchestrator:
    if type(orchestrator) not in (
        CognitiveNoWorkspaceRequestOrchestrator,
        CognitiveSharedBufferRequestOrchestrator,
        IrisRequestOrchestrator,
    ):
        raise AssertionError("test runtime must own a sidecar request graph")
    return cast(_SidecarRequestOrchestrator, orchestrator)


def _valid_backbone_config() -> BackboneConfig:
    values = _backbone_config().model_dump(mode="python")
    values["revision"] = "1" * 40
    values["tokenizer_revision"] = "2" * 40
    return BackboneConfig.model_validate(values)


def _arm_cognitive_config(arm_id: IrisTeacherArmId) -> CognitiveConfig:
    values = _cognitive_config().model_dump(mode="python")
    if arm_id in {"B", "C"}:
        values["num_modules"] = 2
    return CognitiveConfig.model_validate(values)


def _fresh_owner(
    arm_id: IrisTeacherArmId,
) -> tuple[IrisArmTeacherModel, IrisTeacherOptimizer]:
    backbone_config = _valid_backbone_config()
    device_info = DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name="Offline test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=_TinyFrozenCausalLM(),
        ),
    ):
        backbone = BackboneAdapter(backbone_config, device_info)
    cognitive = _arm_cognitive_config(arm_id)
    model: IrisArmTeacherModel
    if arm_id == "B":
        model = build_no_workspace_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive,
        )
    elif arm_id == "C":
        model = build_shared_buffer_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive,
        )
    else:
        model = build_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=cognitive,
            workspace_config=_workspace_config(),
        )
    optimizer = IrisTeacherOptimizer(
        model=model,
        config=IrisAdamWConfig(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        ),
    )
    return model, optimizer


def _trained_checkpoint(
    arm_id: IrisTeacherArmId,
) -> tuple[IrisArmTeacherModel, IrisTeacherOptimizer, IrisArmTeacherCheckpointBundle]:
    model, optimizer = _fresh_owner(arm_id)
    optimizer.step(
        IrisTeacherForcingInput(
            prompt="offline teacher prompt",
            target_input_ids=torch.tensor([[5, 15]], dtype=torch.int64),
            target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    )
    return model, optimizer, build_arm_teacher_checkpoint(
        arm_id=arm_id,
        model=model,
        optimizer=optimizer,
    )


def _checkpoint_configs(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> tuple[BackboneConfig, CognitiveConfig, WorkspaceConfig]:
    if checkpoint.identity.arm_id == "B":
        payload_b = cast(IrisNoWorkspaceTeacherCheckpointBundleV2, checkpoint.payload)
        model_b = payload_b.manifest.payload.model
        return model_b.backbone, model_b.cognitive, _workspace_config()
    if checkpoint.identity.arm_id == "C":
        payload_c = cast(IrisSharedBufferTeacherCheckpointBundleV3, checkpoint.payload)
        model_c = payload_c.manifest.payload.model
        return model_c.backbone, model_c.cognitive, _workspace_config()
    payload_d = cast(IrisTeacherCheckpointBundleV1, checkpoint.payload)
    model_d = payload_d.manifest.payload.model
    return model_d.backbone, model_d.cognitive, model_d.workspace


def _runtime(
    *,
    model: IrisArmTeacherModel,
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> IrisArmRequestRuntimeV3:
    arm_id = checkpoint.identity.arm_id
    backbone_config, cognitive, workspace = _checkpoint_configs(checkpoint)
    config = IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "offline-transfer", "version": "0.1.0"},
            "hardware": {
                "name": "offline-cpu",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 1.0,
                "allow_gpu": False,
            },
            "backbone": backbone_config.model_dump(mode="python"),
            "cognitive": cognitive.model_dump(mode="python"),
            "workspace": workspace.model_dump(mode="python"),
            "runtime": {
                "seed": _SEED,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": "unused-offline-transfer",
            },
        }
    )
    sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
    orchestrator = build_arm_request_orchestrator_v4(
        arm_id=arm_id,
        backbone=model.backbone,
        stop_token_ids=(15,),
        sampling=sampling,
        cognitive_config=cognitive,
        workspace_config=(workspace if arm_id == "D" else None),
    )
    sidecar_orchestrator = _require_sidecar_orchestrator(orchestrator)
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode("utf-8")
    return IrisArmRequestRuntimeV3(
        profile=require_arm_request_executable_v4(arm_id),
        execution_profile_set_sha256=_PROFILE_SHA256,
        backbone=model.backbone,
        orchestrator=sidecar_orchestrator,
        device_info=DeviceInfo(
            device=torch.device("cpu"),
            backend="cpu",
            name="Offline test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=_SEED,
        deterministic=True,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256(
            "iris/configuration/v1",
            configuration_json.encode("utf-8"),
        ),
        sidecar_state_sha256=sidecar_orchestrator.sidecar_state_sha256,
    )


def _request_state(runtime: IrisArmRequestRuntimeV3) -> dict[str, torch.Tensor]:
    sidecar_orchestrator = _require_sidecar_orchestrator(runtime.orchestrator)
    modules = sidecar_orchestrator._sidecar_modules
    state: dict[str, torch.Tensor] = {}
    for component_name, module in modules:
        for state_name, tensor in module.state_dict().items():
            state[f"{component_name}.{state_name}"] = tensor.detach().clone()
    return state


def _checkpoint_state(checkpoint: IrisArmTeacherCheckpointBundle) -> dict[str, torch.Tensor]:
    payload = checkpoint.payload
    if type(payload) not in (
        IrisNoWorkspaceTeacherCheckpointBundleV2,
        IrisSharedBufferTeacherCheckpointBundleV3,
        IrisTeacherCheckpointBundleV1,
    ):
        raise AssertionError("test checkpoint payload must be B/v2, C/v3, or D/v1")
    manifest_payload = payload.manifest.payload
    state_bytes = payload.state_bytes
    decoded = load_safetensors(state_bytes)
    return {
        descriptor.state_name: decoded[descriptor.key].detach().clone()
        for descriptor in manifest_payload.tensors
        if descriptor.owner == "sidecar"
    }


@pytest.mark.parametrize(
    ("arm_id", "checkpoint_version"),
    [("B", 2), ("C", 3), ("D", 1)],
)
def test_trained_checkpoint_weights_transfer_exactly_into_request_graph(
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SEED)
        model, _, checkpoint = _trained_checkpoint(arm_id)
        runtime = _runtime(model=model, checkpoint=checkpoint)
        previous_hash = runtime.sidecar_state_sha256
        result = transfer_arm_teacher_checkpoint_to_request_v1(
            runtime=runtime,
            checkpoint=checkpoint,
        )
        projection = project_arm_teacher_checkpoint_to_request_v1(checkpoint)

    expected = _checkpoint_state(checkpoint)
    installed = _request_state(result.runtime)
    assert type(result) is IrisArmRequestCheckpointTransferResultV1
    assert type(projection) is IrisArmRequestCheckpointProjectionV1
    assert result.checkpoint_identity == checkpoint.identity
    assert result.transfer_contract == "exact_teacher_state_to_profile_v4_request_sidecar_v1"
    assert result.checkpoint_identity.checkpoint_schema_version == checkpoint_version
    assert result.source_teacher_sidecar_sha256 == checkpoint.identity.sidecar_state_sha256
    assert result.previous_request_sidecar_sha256 == previous_hash
    assert result.installed_request_sidecar_sha256 == result.runtime.sidecar_state_sha256
    assert (
        _require_sidecar_orchestrator(result.runtime.orchestrator).sidecar_state_sha256
        == result.runtime.sidecar_state_sha256
    )
    assert result.completed_steps == 1
    assert result.state_names == tuple(expected) == tuple(installed)
    assert result.transferred_tensor_count == len(expected)
    assert all(torch.equal(installed[name], tensor) for name, tensor in expected.items())
    assert result.source_teacher_sidecar_sha256 != result.installed_request_sidecar_sha256
    assert projection.checkpoint_identity == result.checkpoint_identity
    assert projection.source_teacher_sidecar_sha256 == result.source_teacher_sidecar_sha256
    assert projection.request_sidecar_state_sha256 == result.installed_request_sidecar_sha256
    assert projection.state_names == result.state_names
    assert projection.tensor_count == result.transferred_tensor_count
    assert arm_execution_profile_set_sha256_v4() == _PROFILE_SHA256


def test_cross_arm_checkpoint_rejects_before_request_mutation() -> None:
    with torch.random.fork_rng(devices=[]):
        model_b, _, checkpoint_b = _trained_checkpoint("B")
        runtime_b = _runtime(model=model_b, checkpoint=checkpoint_b)
        _, _, checkpoint_c = _trained_checkpoint("C")
        before_hash = _require_sidecar_orchestrator(
            runtime_b.orchestrator
        ).sidecar_state_sha256
        before_state = _request_state(runtime_b)

        with pytest.raises(IrisRequestCheckpointError, match="incompatible") as raised:
            transfer_arm_teacher_checkpoint_to_request_v1(
                runtime=runtime_b,
                checkpoint=checkpoint_c,
            )

    assert "arm does not match" in str(raised.value.__cause__)
    assert (
        _require_sidecar_orchestrator(runtime_b.orchestrator).sidecar_state_sha256
        == before_hash
    )
    assert all(
        torch.equal(_request_state(runtime_b)[name], tensor)
        for name, tensor in before_state.items()
    )


def test_configuration_mismatch_rejects_same_shape_checkpoint_before_mutation() -> None:
    with torch.random.fork_rng(devices=[]):
        model, _, checkpoint = _trained_checkpoint("B")
        runtime = _runtime(model=model, checkpoint=checkpoint)
        config = IrisBootstrapConfig.model_validate_json(runtime.configuration_json)
        cognitive_values = config.cognitive.model_dump(mode="python")
        cognitive_values["name"] = "same-shape-but-different-cognitive-config"
        changed = config.model_copy(
            update={"cognitive": CognitiveConfig.model_validate(cognitive_values)}
        )
        changed_json = canonical_json_bytes(changed.model_dump(mode="json")).decode("utf-8")
        mismatched = replace(
            runtime,
            configuration_json=changed_json,
            configuration_sha256=domain_sha256(
                "iris/configuration/v1",
                changed_json.encode("utf-8"),
            ),
        )
        before_hash = _require_sidecar_orchestrator(
            runtime.orchestrator
        ).sidecar_state_sha256
        before_state = _request_state(runtime)

        with pytest.raises(IrisRequestCheckpointError, match="incompatible") as raised:
            transfer_arm_teacher_checkpoint_to_request_v1(
                runtime=mismatched,
                checkpoint=checkpoint,
            )

    assert "configuration does not match" in str(raised.value.__cause__)
    assert (
        _require_sidecar_orchestrator(runtime.orchestrator).sidecar_state_sha256
        == before_hash
    )
    assert all(
        torch.equal(_request_state(runtime)[name], tensor)
        for name, tensor in before_state.items()
    )


def test_content_addressed_loader_preflights_arm_before_filesystem_read(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        model_b, _, checkpoint_b = _trained_checkpoint("B")
        runtime_b = _runtime(model=model_b, checkpoint=checkpoint_b)
        _, _, checkpoint_c = _trained_checkpoint("C")

        with (
            patch.object(request_checkpoint, "read_arm_teacher_checkpoint") as read,
            pytest.raises(IrisRequestCheckpointError, match="could not be loaded") as raised,
        ):
            load_arm_teacher_checkpoint_into_request_v1(
                runtime=runtime_b,
                identity=checkpoint_c.identity,
                root=tmp_path,
            )

    assert "before filesystem access" in str(raised.value.__cause__)
    read.assert_not_called()


def test_content_addressed_loader_reads_and_transfers_exact_arm_c_checkpoint(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        model, _, checkpoint = _trained_checkpoint("C")
        written = write_arm_teacher_checkpoint(checkpoint, root=tmp_path)
        runtime = _runtime(model=model, checkpoint=checkpoint)
        result = load_arm_teacher_checkpoint_into_request_v1(
            runtime=runtime,
            identity=checkpoint.identity,
            root=tmp_path,
        )

    assert written.path == checkpoint.identity.expected_path(root=tmp_path)
    assert result.checkpoint_identity == checkpoint.identity
    assert result.completed_steps == 1
    assert (
        _require_sidecar_orchestrator(result.runtime.orchestrator).sidecar_state_sha256
        == result.installed_request_sidecar_sha256
    )
