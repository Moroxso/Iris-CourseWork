"""Actual tiny A-D graphs, seeded sampling, durable pending recovery and v1 equivalence."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from iris.application import study_generation_run as service
from iris.application import study_generation_run_io as storage
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    build_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.errors import IrisStudyGenerationError
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.study_generation import generate_study_split_v1
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.study_v1 import StudyArmIdV1
from iris.backbone import BackboneAdapter
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_study_generation_flow import _model, _policies
from tests.integration.test_verification_v4_flow import _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_study_input_preflight import _tokenizer

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
@pytest.mark.parametrize("sampling", [False, True])
def test_real_arms_recover_staged_answer_without_reinference_and_match_legacy(
    tmp_path: Path, arm_id: StudyArmIdV1, sampling: bool
) -> None:
    root, directory = tmp_path / "artifacts", tmp_path / "run"
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    config = _config(root)
    config.backbone.limits.max_new_tokens = 128
    model, tokenizer = _model(), _tokenizer()
    identity = None
    checkpoint_path: Path | None = None
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as loader,
        patch(
            "iris.application.study_code_evaluation.execute_python_job_v1",
            side_effect=AssertionError("generation cannot score code"),
        ),
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
                    prompt="durable generation training fixture",
                    target_input_ids=torch.tensor([[0, 1]]),
                    target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
                )
            )
            checkpoint = build_arm_teacher_checkpoint(
                arm_id=arm_id, model=teacher, optimizer=optimizer
            )
            checkpoint_path = write_arm_teacher_checkpoint(checkpoint, root=root).path
            identity = checkpoint.identity
        options: dict[str, Any] = dict(
            config=config,
            arm_id=arm_id,
            checkpoint_identity=identity,
            split="test",
            decoding=_policies(sampling=sampling),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
        )
        budgets: dict[str, Any] = dict(
            directory=directory, max_record_bytes=1024 * 1024, max_storage_bytes=16 * 1024 * 1024
        )
        original_stage = storage._stage

        def stop_after_staging(pending: Path, content: bytes, **kwargs: Any) -> None:
            original_stage(pending, content, **kwargs)
            if (
                pending.name == ".answer-pending.json"
                and json.loads(content)["payload"]["index"] == 2
            ):
                raise KeyboardInterrupt

        before = capture_runtime_state(CPU_INFO.device)
        initial_loads = loader.call_count
        with (
            patch.object(service, "_stage", side_effect=stop_after_staging),
            pytest.raises(KeyboardInterrupt),
        ):
            service.generate_study_split_run_v1(benchmark, **options, **budgets, resume=False)
        assert loader.call_count == initial_loads + 1
        interrupted = capture_runtime_state(CPU_INFO.device)
        assert torch.equal(before.cpu_rng_state, interrupted.cpu_rng_state)
        assert before.python_rng_state == interrupted.python_rng_state
        pending = storage._read_answer(directory / ".answer-pending.json", 1024 * 1024)
        pending_request = (
            root / "requests" / "v4" / f"{pending.payload.answer.request_payload_sha256}.json"
        )
        assert not pending_request.exists()
        assert len(tuple((directory / "answers").iterdir())) == 2
        result = service.generate_study_split_run_v1(benchmark, **options, **budgets, resume=True)
        assert (result.generated_now, result.reused_answers) == (2, 3)
        assert pending_request.is_file()
        assert loader.call_count == initial_loads + 2
        resumed = read_study_generation_v1(result.path)
        assert resumed.payload.answers[2] == pending.payload.answer
        restored = capture_runtime_state(CPU_INFO.device)
        assert torch.equal(before.cpu_rng_state, restored.cpu_rng_state)
        assert before.python_rng_state == restored.python_rng_state
        # The original all-at-once API produces the exact same private artifact,
        # despite a fresh runtime being used for the resumed suffix above.
        legacy = generate_study_split_v1(benchmark, **options, path=tmp_path / "legacy.json")
        assert legacy.payload_sha256 == resumed.payload_sha256
        assert legacy.path.read_bytes() == result.path.read_bytes()
        loads_before_complete = loader.call_count
        with patch.object(service, "_iter_study_generation_answers") as no_inference:
            again = service.generate_study_split_run_v1(
                benchmark, **options, **budgets, resume=True
            )
            assert (again.generated_now, again.reused_answers) == (0, 5)
            assert service.verify_study_generation_run_v1(directory, benchmark=benchmark) == resumed
            no_inference.assert_not_called()
        assert loader.call_count == loads_before_complete
        if checkpoint_path is not None:
            state = checkpoint_path / "state.safetensors"
            metadata = state.stat()
            content = state.read_bytes()
            state.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
            os.utime(state, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            with patch.object(service, "_iter_study_generation_answers") as forbidden:
                with pytest.raises(IrisStudyGenerationError):
                    service.generate_study_split_run_v1(
                        benchmark, **options, **budgets, resume=True
                    )
                forbidden.assert_not_called()
            assert loader.call_count == loads_before_complete
