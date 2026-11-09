"""Real checkpoint-to-request-artifact v4 verification flow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch

from iris.application.arm_profiles import (
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import (
    transfer_arm_teacher_checkpoint_to_request_v1,
)
from iris.application.arm_training_checkpoint import (
    build_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.verification_v4 import verify_request_artifact_v4
from iris.backbone import BackboneAdapter
from iris.schemas import IrisBootstrapConfig
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.integration.test_teacher_forcing_sidecar import (
    _TinyFrozenCausalLM,
    _TinyTokenizer,
)
from tests.unit.test_request_artifact_schema import valid_bootstrap_config
from tests.unit.test_request_artifact_v3 import (
    SAMPLING,
    SECRET,
    STOP_TOKEN_IDS,
    _execution,
)

pytestmark = pytest.mark.integration

_PROFILE_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"


def _config(root: Path) -> IrisBootstrapConfig:
    values = valid_bootstrap_config().model_dump(mode="python")
    runtime = values["runtime"]
    if not isinstance(runtime, dict):
        raise AssertionError("runtime fixture must be a dictionary")
    runtime["output_dir"] = str(root)
    return IrisBootstrapConfig.model_validate(values)


def test_real_trained_arm_b_checkpoint_verifies_through_request_artifact_v4(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    config = _config(root)
    device_info = DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name="Verifier v4 integration CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )
    tiny_model = _TinyFrozenCausalLM()
    tiny_model.config.max_position_embeddings = 1024
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=tiny_model,
        ),
    ):
        backbone = BackboneAdapter(config.backbone, device_info)

    teacher = build_no_workspace_teacher_forcing_model(
        backbone=backbone,
        cognitive_config=config.cognitive,
    )
    optimizer = IrisTeacherOptimizer(
        model=teacher,
        config=IrisAdamWConfig(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        ),
    )
    optimizer.step(
        IrisTeacherForcingInput(
            prompt="verifier v4 teacher prompt",
            target_input_ids=torch.tensor([[5, 15]], dtype=torch.int64),
            target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    )
    checkpoint = build_arm_teacher_checkpoint(
        arm_id="B",
        model=teacher,
        optimizer=optimizer,
    )
    checkpoint_write = write_arm_teacher_checkpoint(checkpoint, root=root)

    orchestrator = build_arm_request_orchestrator_v4(
        arm_id="B",
        backbone=backbone,
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=SAMPLING,
        cognitive_config=config.cognitive,
        workspace_config=None,
    )
    arm_b_orchestrator = cast(CognitiveNoWorkspaceRequestOrchestrator, orchestrator)
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode()
    runtime = IrisArmRequestRuntimeV3(
        profile=require_arm_request_executable_v4("B"),
        execution_profile_set_sha256=_PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=device_info,
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256(
            "iris/configuration/v1",
            configuration_json.encode(),
        ),
        sidecar_state_sha256=arm_b_orchestrator.sidecar_state_sha256,
    )
    transfer = transfer_arm_teacher_checkpoint_to_request_v1(
        runtime=runtime,
        checkpoint=checkpoint,
    )
    artifact = build_request_artifact_v4(
        prompt=SECRET,
        strategy="greedy",
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=SAMPLING,
        sampling_seed=None,
        allow_download=False,
        runtime=transfer.runtime,
        execution=_execution("B"),
        checkpoint_transfer=transfer,
    )
    artifact_write = write_request_artifact_v4(artifact)
    report = verify_request_artifact_v4(artifact_write.path)

    assert checkpoint_write.path == checkpoint.identity.expected_path(root=root)
    assert report.arm_id == "B"
    assert report.checkpoint_present
    assert report.checkpoint_schema_version == 2
    assert report.checkpoint_payload_sha256 == checkpoint.identity.payload_sha256
    assert report.checkpoint_state_file_sha256 == checkpoint.identity.state_file_sha256
    assert report.source_teacher_sidecar_sha256 == checkpoint.identity.sidecar_state_sha256
    assert report.request_sidecar_state_sha256 == transfer.installed_request_sidecar_sha256
    assert report.checkpoint_completed_steps == 1
    assert report.transferred_tensor_count == transfer.transferred_tensor_count
