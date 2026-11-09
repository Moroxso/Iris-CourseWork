"""Actual tiny-random-Qwen full A-D generation, real gVisor scoring and no-execution resume."""

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from iris.application import (
    IrisStudyEvaluationError,
    evaluate_study_split_v1,
    read_study_evaluation_run_v1,
    verify_study_evaluation_run_v1,
)
from iris.application import study_evaluation_run as service
from iris.application import study_evaluation_run_io as storage
from iris.application import verification_v4 as request_verifier
from iris.application.arm_request_checkpoint import project_arm_teacher_checkpoint_to_request_v1
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    build_arm_teacher_checkpoint,
    read_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.study_generation import generate_study_split_v1, verify_study_generation_v1
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.study_v1 import StudyArmIdV1
from iris.backbone import BackboneAdapter
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from tests.integration.test_study_code_evaluation_smoke import _empty_inventory
from tests.integration.test_study_generation_flow import _model, _policies
from tests.integration.test_verification_v4_flow import _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_study_code_evaluation import _POLICY
from tests.unit.test_study_input_preflight import _tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="requires the verified opt-in IrisCodeEval runtime",
    ),
]


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_all_families_real_arm_checkpoint_generation_scoring_and_resume(
    tmp_path: Path, arm_id: StudyArmIdV1
) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    root = tmp_path / "artifacts"
    config = _config(root)
    config.backbone.limits.max_new_tokens = 128
    model, tokenizer = _model(), _tokenizer()
    checkpoint = None
    checkpoint_path = None
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        if arm_id != "A":
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
                    backbone=backbone,
                    cognitive_config=config.cognitive,
                    workspace_config=config.workspace,
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
                    prompt="training fixture",
                    target_input_ids=torch.tensor([[0, 1]], dtype=torch.int64),
                    target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
                )
            )
            checkpoint = build_arm_teacher_checkpoint(
                arm_id=arm_id, model=teacher, optimizer=optimizer
            )
            checkpoint_path = write_arm_teacher_checkpoint(checkpoint, root=root).path
        generation_path = tmp_path / "generation.json"
        generate_study_split_v1(
            benchmark,
            config=config,
            arm_id=arm_id,
            checkpoint_identity=None if checkpoint is None else checkpoint.identity,
            split="test",
            decoding=_policies(),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
            path=generation_path,
        )
    generation = read_study_generation_v1(generation_path)
    directory = tmp_path / "evaluation"
    original = storage._publish

    def stop_after_code(destination: Path, content: bytes, **kwargs: Any) -> None:
        original(destination, content, **kwargs)
        if destination.name == "000002.json":
            raise KeyboardInterrupt

    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as no_model,
    ):
        with patch.object(service, "_publish", stop_after_code), pytest.raises(KeyboardInterrupt):
            evaluate_study_split_v1(
                generation,
                benchmark=benchmark,
                request_root=root,
                directory=directory,
                code_policy=_POLICY,
                max_storage_bytes=32 * 1024 * 1024,
                resume=False,
            )
        assert len(list((directory / "receipts").iterdir())) == 3
        assert not (directory / "summary.json").exists()
        with patch("iris.application.study_code_evaluation.execute_python_job_v1") as no_code:
            result = evaluate_study_split_v1(
                generation,
                benchmark=benchmark,
                request_root=root,
                directory=directory,
                code_policy=_POLICY,
                max_storage_bytes=32 * 1024 * 1024,
                resume=True,
            )
            assert result.reused_evaluations == 3 and result.evaluated_now == 2
            saved_bytes = {
                path.relative_to(directory): path.read_bytes()
                for path in directory.rglob("*")
                if path.is_file()
            }
            with (
                patch.object(
                    request_verifier,
                    "read_arm_teacher_checkpoint",
                    wraps=read_arm_teacher_checkpoint,
                ) as read,
                patch.object(
                    request_verifier,
                    "project_arm_teacher_checkpoint_to_request_v1",
                    wraps=project_arm_teacher_checkpoint_to_request_v1,
                ) as project,
            ):
                summary = verify_study_evaluation_run_v1(
                    directory, generation=generation, benchmark=benchmark, request_root=root
                )
                assert read.call_count == project.call_count == (0 if arm_id == "A" else 1)
                read.reset_mock()
                project.reset_mock()
                # Test-only uncached reference: all existing deep reads and semantic
                # checks run. This is not an exposed API option or trust assertion.
                with patch.object(request_verifier, "_current_checkpoint_scope", return_value=None):
                    uncached = verify_study_evaluation_run_v1(
                        directory, generation=generation, benchmark=benchmark, request_root=root
                    )
                assert read.call_count == project.call_count == (0 if arm_id == "A" else 64)
                assert uncached == summary
            assert saved_bytes == {
                path.relative_to(directory): path.read_bytes()
                for path in directory.rglob("*")
                if path.is_file()
            }
            assert summary.payload.overall.total == 5
            assert len(summary.payload.by_family) == 5
            plan = read_study_evaluation_run_v1(directory)
            assert plan.payload.arm_id == arm_id
            if checkpoint is not None:
                assert plan.payload.checkpoint_transfer is not None
                assert (
                    plan.payload.checkpoint_transfer.checkpoint_payload_sha256
                    == checkpoint.identity.payload_sha256
                )
                assert checkpoint_path is not None
                state_path = checkpoint_path / "state.safetensors"
                original_state = state_path.read_bytes()
                original_stat = state_path.stat()
                damaged = bytearray(original_state)
                damaged[-1] ^= 1

                # Mutate after the last nested generation check returns, while
                # the outer whole-run context still owns its verified projection.
                # Only the new final full-byte check can catch this boundary.
                original_verify = verify_study_generation_v1
                generation_checks = 0

                def mutate_after_final_generation(*args: Any, **kwargs: Any) -> Any:
                    nonlocal generation_checks
                    checked = original_verify(*args, **kwargs)
                    generation_checks += 1
                    if generation_checks == 2:
                        state_path.write_bytes(damaged)
                        os.utime(
                            state_path,
                            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                        )
                    return checked

                try:
                    with (
                        patch.object(
                            service,
                            "verify_study_generation_v1",
                            side_effect=mutate_after_final_generation,
                        ),
                        pytest.raises(IrisStudyEvaluationError),
                    ):
                        verify_study_evaluation_run_v1(
                            directory, generation=generation, benchmark=benchmark, request_root=root
                        )
                    assert generation_checks == 2
                finally:
                    # Restore only this test-owned fixture; no production repair.
                    state_path.write_bytes(original_state)
                assert saved_bytes == {
                    path.relative_to(directory): path.read_bytes()
                    for path in directory.rglob("*")
                    if path.is_file()
                }
                state_path.write_bytes(damaged)
                with pytest.raises(IrisStudyEvaluationError):
                    verify_study_evaluation_run_v1(
                        directory, generation=generation, benchmark=benchmark, request_root=root
                    )
            no_code.assert_not_called()
        no_model.assert_not_called()
    _empty_inventory()
