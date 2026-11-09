"""Real B/C/D checkpoint projections: cached/uncached equivalence and mutation rejection."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from iris.application import study_checkpoint_scope as scope
from iris.application import verification_v4 as verifier
from iris.application.arm_profiles import (
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import (
    project_arm_teacher_checkpoint_to_request_v1,
    transfer_arm_teacher_checkpoint_to_request_v1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
    read_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisVerificationError
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.integration.test_arm_request_checkpoint_flow import _require_sidecar_orchestrator
from tests.integration.test_study_generation_flow import _model
from tests.integration.test_verification_v4_flow import _PROFILE_SHA256, _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_study_input_preflight import _tokenizer

pytestmark = pytest.mark.integration


def _requests(tmp_path: Path, arm_id: IrisTeacherArmId) -> tuple[tuple[Path, ...], Path]:
    root = tmp_path / "artifacts"
    config = _config(root)
    config.backbone.limits.max_new_tokens = 128
    model, tokenizer = _model(), _tokenizer()
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        backbone = BackboneAdapter(config.backbone, CPU_INFO)
    teacher: IrisArmTeacherModel
    if arm_id == "B":
        teacher = build_no_workspace_teacher_forcing_model(
            backbone=backbone, cognitive_config=config.cognitive
        )
    elif arm_id == "C":
        teacher = build_shared_buffer_teacher_forcing_model(
            backbone=backbone, cognitive_config=config.cognitive
        )
    else:
        teacher = build_teacher_forcing_model(
            backbone=backbone, cognitive_config=config.cognitive, workspace_config=config.workspace
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
            prompt="checkpoint scope training fixture",
            target_input_ids=torch.tensor([[0, 1]], dtype=torch.int64),
            target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    )
    checkpoint = build_arm_teacher_checkpoint(arm_id=arm_id, model=teacher, optimizer=optimizer)
    directory = write_arm_teacher_checkpoint(checkpoint, root=root).path
    stops = tuple(range(16))
    sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
    orchestrator = build_arm_request_orchestrator_v4(
        arm_id=arm_id,
        backbone=backbone,
        stop_token_ids=stops,
        sampling=sampling,
        cognitive_config=config.cognitive,
        workspace_config=config.workspace if arm_id == "D" else None,
    )
    config_bytes = canonical_json_bytes(config.model_dump(mode="json"))
    runtime = IrisArmRequestRuntimeV3(
        profile=require_arm_request_executable_v4(arm_id),
        execution_profile_set_sha256=_PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=CPU_INFO,
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=config_bytes.decode(),
        configuration_sha256=domain_sha256("iris/configuration/v1", config_bytes),
        sidecar_state_sha256=_require_sidecar_orchestrator(orchestrator).sidecar_state_sha256,
    )
    transfer = transfer_arm_teacher_checkpoint_to_request_v1(runtime=runtime, checkpoint=checkpoint)
    paths = []
    for index in range(5):
        prompt = f"private checkpoint scope prompt {index}"
        execution = transfer.runtime.orchestrator.execute(prompt, strategy="greedy")
        artifact = build_request_artifact_v4(
            prompt=prompt,
            strategy="greedy",
            stop_token_ids=stops,
            sampling=sampling,
            sampling_seed=None,
            allow_download=False,
            runtime=transfer.runtime,
            execution=execution,
            checkpoint_transfer=transfer,
        )
        paths.append(write_request_artifact_v4(artifact).path)
    assert len(set(paths)) == 5
    return tuple(paths), directory


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_real_distinct_request_reports_match_with_one_deep_checkpoint_read(
    tmp_path: Path, arm_id: IrisTeacherArmId
) -> None:
    paths, _ = _requests(tmp_path, arm_id)
    with (
        patch.object(
            verifier, "read_arm_teacher_checkpoint", wraps=read_arm_teacher_checkpoint
        ) as read,
        patch.object(
            verifier,
            "project_arm_teacher_checkpoint_to_request_v1",
            wraps=project_arm_teacher_checkpoint_to_request_v1,
        ) as project,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
    ):
        expected = tuple(verifier.verify_request_artifact_v4(path) for path in paths)
        assert read.call_count == project.call_count == 10
        read.reset_mock()
        project.reset_mock()
        with scope._checkpoint_verification_context():
            observed = tuple(verifier.verify_request_artifact_v4(path) for path in paths)
        assert observed == expected
        assert read.call_count == project.call_count == 1
        # No cache survives the batch lifetime; standalone verification remains twice.
        assert verifier.verify_request_artifact_v4(paths[0]) == expected[0]
        assert read.call_count == project.call_count == 3
        loader.assert_not_called()


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
@pytest.mark.parametrize("filename", ["manifest.json", "state.safetensors"])
def test_real_cached_checkpoint_corruption_rejected_even_with_restored_mtime(
    tmp_path: Path, arm_id: IrisTeacherArmId, filename: str
) -> None:
    paths, directory = _requests(tmp_path, arm_id)
    target = directory / filename
    with pytest.raises(IrisVerificationError), scope._checkpoint_verification_context():
        verifier.verify_request_artifact_v4(paths[0])
        before = target.stat()
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        verifier.verify_request_artifact_v4(paths[1])
    assert scope._current_checkpoint_scope() is None


def test_cached_projection_does_not_skip_current_request_semantics(tmp_path: Path) -> None:
    paths, _ = _requests(tmp_path, "B")
    with pytest.raises(IrisVerificationError), scope._checkpoint_verification_context():
        verifier.verify_request_artifact_v4(paths[0])
        with patch.object(
            verifier, "_validate_profile_v4_binding", side_effect=ValueError("profile changed")
        ) as profile:
            verifier.verify_request_artifact_v4(paths[1])
    profile.assert_called_once()
