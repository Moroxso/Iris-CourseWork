"""Actual tiny A-D inference/checkpoints through packaged generation and verifier commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from iris.application import study_generation_command as file_service
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    build_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
)
from iris.application.study_benchmark_io import write_study_benchmark_v1
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_generation_run_io import read_study_generation_run_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.study_v1 import StudyArmIdV1
from iris.backbone import BackboneAdapter
from iris.cli import app
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_study_generation_commands import _args, _paths, runner
from tests.integration.test_study_generation_flow import _model, _policies
from tests.integration.test_verification_v4_flow import _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_study_generation_run import _saved
from tests.unit.test_study_input_preflight import _tokenizer

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_real_four_arm_sampling_new_resume_and_read_only_cli(
    tmp_path: Path,
    arm_id: StudyArmIdV1,
) -> None:
    root = tmp_path / "artifacts"
    paths = _paths(tmp_path)
    config = _config(root)
    config.backbone.limits.max_new_tokens = 128
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    write_study_benchmark_v1(benchmark, path=paths["benchmark_path"])
    paths["configuration_path"].write_bytes(
        canonical_json_file_bytes(config.model_dump(mode="json"))
    )
    dataset = benchmark.payload.dataset.payload
    selected = tuple(
        (example, assignment)
        for example, assignment in zip(
            dataset.manifest.payload.examples, dataset.assignments, strict=True
        )
        if example.split == "test"
    )
    policies = _policies(sampling=True)
    paths["decoding_path"].write_bytes(
        canonical_json_file_bytes(
            [
                {
                    "example_id": example.example_id,
                    "task_family": assignment.task_family,
                    "decoding": policy.model_dump(mode="json"),
                }
                for (example, assignment), policy in zip(selected, policies, strict=True)
            ]
        )
    )
    model = _model()
    options = {"--arm": arm_id, "--split": "test"}
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=_tokenizer()),
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
                    prompt="packaged generation training fixture",
                    target_input_ids=torch.tensor([[0, 1]]),
                    target_attention_mask=torch.ones((1, 2), dtype=torch.bool),
                )
            )
            checkpoint = build_arm_teacher_checkpoint(
                arm_id=arm_id, model=teacher, optimizer=optimizer
            )
            write_arm_teacher_checkpoint(checkpoint, root=root)
            identity = checkpoint.identity
            options.update(
                {
                    "--checkpoint-payload-sha256": identity.payload_sha256,
                    "--checkpoint-state-file-sha256": identity.state_file_sha256,
                    "--checkpoint-sidecar-state-sha256": identity.sidecar_state_sha256,
                    "--checkpoint-completed-steps": str(identity.completed_steps),
                }
            )
        initial_loads = loader.call_count
        result = runner.invoke(app, _args(tmp_path, changes=options))
        assert result.exit_code == 0, result.output
        assert loader.call_count == initial_loads + 1
        for call in loader.call_args_list:
            assert call.kwargs["local_files_only"] is True
        run = read_study_generation_run_v1(paths["directory"])
        generation = read_study_generation_v1(paths["directory"] / "generation.json")
        assert run.payload.arm_id == generation.payload.arm_id == arm_id
        assert tuple(item.decoding for item in run.payload.answers) == policies
        before = _saved(paths["directory"])
        with patch(
            "iris.application.study_generation_run._iter_study_generation_answers",
            side_effect=AssertionError("complete resume cannot infer"),
        ):
            resumed = runner.invoke(app, _args(tmp_path, changes={**options, "--mode": "resume"}))
            assert resumed.exit_code == 0, resumed.output
            verified = runner.invoke(app, _args(tmp_path, verify=True))
            assert verified.exit_code == 0, verified.output
            report = file_service.verify_study_generation_files_v1(
                benchmark_path=paths["benchmark_path"], directory=paths["directory"]
            )
        assert loader.call_count == initial_loads + 1
        assert _saved(paths["directory"]) == before
        assert report.arm_id == arm_id and report.total_examples == 5
        assert report.generation_payload_sha256 == generation.payload_sha256
        assert "not_assessed" in verified.output
