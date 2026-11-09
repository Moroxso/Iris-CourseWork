"""Offline A-D inference and B/C/D one-step checkpoint-bound code evaluations in real gVisor.

Uses a tiny randomly initialized Qwen, never downloaded/pretrained model weights.
The all-vocabulary stop set keeps test inference to exactly one generated token.
These are engineering tests, not accuracy measurements or scientific experiments.
"""

import os
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from iris.application import (
    IrisStudyEvaluationError,
    evaluate_study_request_python_answer_v1,
    read_study_code_evaluation_v1,
    verify_study_code_evaluation_v1,
    write_study_code_evaluation_v1,
)
from iris.application.arm_profiles import (
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import transfer_arm_teacher_checkpoint_to_request_v1
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherModel,
    build_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
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
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.study_v1 import StudyArmIdV1
from iris.backbone import BackboneAdapter
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.integration.test_arm_request_checkpoint_flow import _require_sidecar_orchestrator
from tests.integration.test_study_code_evaluation_smoke import _empty_inventory
from tests.integration.test_verification_v4_flow import _PROFILE_SHA256, _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_request_artifact_v3 import SAMPLING
from tests.unit.test_study_code_evaluation import _POLICY
from tests.unit.test_study_code_receipt import _fixture
from tests.unit.test_study_input_preflight import _assets, _tokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
        reason="requires the verified opt-in IrisCodeEval gVisor runtime",
    ),
]


@pytest.mark.parametrize("arm_id", ["A", "B", "C", "D"])
def test_actual_four_arm_code_request_capture_and_checkpoint_reverification(
    tmp_path: Path, arm_id: StudyArmIdV1
) -> None:
    root = tmp_path / "artifacts"
    config = _config(root)
    config.backbone.limits.max_new_tokens = 128
    tokenizer = _tokenizer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        tiny_model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=1024,
                bos_token_id=2,
                eos_token_id=1,
                tie_word_embeddings=True,
            )
        )
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=tiny_model
        ),
    ):
        backbone = BackboneAdapter(config.backbone, CPU_INFO)

    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    example = benchmark.payload.dataset.payload.manifest.payload.examples[2]
    checkpoint = None
    checkpoint_path = None
    if arm_id != "A":
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
        target = tokenizer(example.target, add_special_tokens=False)["input_ids"] + [1]
        optimizer.step(
            IrisTeacherForcingInput(
                prompt=example.prompt,
                target_input_ids=torch.tensor([target], dtype=torch.int64),
                target_attention_mask=torch.ones((1, len(target)), dtype=torch.bool),
            )
        )
        checkpoint = build_arm_teacher_checkpoint(arm_id=arm_id, model=teacher, optimizer=optimizer)
        checkpoint_path = write_arm_teacher_checkpoint(checkpoint, root=root).path

    stops = tuple(range(16))
    orchestrator = build_arm_request_orchestrator_v4(
        arm_id=arm_id,
        backbone=backbone,
        stop_token_ids=stops,
        sampling=SAMPLING,
        cognitive_config=config.cognitive if arm_id != "A" else None,
        workspace_config=config.workspace if arm_id == "D" else None,
    )
    configuration = canonical_json_bytes(config.model_dump(mode="json"))
    runtime = IrisArmRequestRuntimeV3(
        profile=require_arm_request_executable_v4(arm_id),
        execution_profile_set_sha256=_PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=CPU_INFO,
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=configuration.decode(),
        configuration_sha256=domain_sha256("iris/configuration/v1", configuration),
        sidecar_state_sha256=(
            None
            if arm_id == "A"
            else _require_sidecar_orchestrator(orchestrator).sidecar_state_sha256
        ),
    )
    transfer = None
    if checkpoint is not None:
        transfer = transfer_arm_teacher_checkpoint_to_request_v1(
            runtime=runtime, checkpoint=checkpoint
        )
        runtime = transfer.runtime
    execution = runtime.orchestrator.execute(example.prompt, strategy="greedy")
    request = build_request_artifact_v4(
        prompt=example.prompt,
        strategy="greedy",
        stop_token_ids=stops,
        sampling=SAMPLING,
        sampling_seed=None,
        allow_download=False,
        runtime=runtime,
        execution=execution,
        checkpoint_transfer=transfer,
    )
    request_path = write_request_artifact_v4(request).path
    assert request.payload.generation.generated_tokens == 1
    assert backbone.trainable_parameter_count == 0
    assert not backbone.has_parameter_gradients

    with (
        patch.object(AutoConfig, "from_pretrained", return_value=tiny_model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained"
        ) as forbidden_model_loader,
    ):
        evaluation = evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=request_path,
            example_id=example.example_id,
            output_text=execution.generation.text,
            terminal_token_id=1,
            policy=_POLICY,
        )
        evaluation_path = tmp_path / "evaluation.json"
        write_study_code_evaluation_v1(evaluation, path=evaluation_path)
        with patch("iris.application.study_code_evaluation.execute_python_job_v1") as no_execution:
            restored = read_study_code_evaluation_v1(evaluation_path)
            assert restored.payload.receipt.arm_id == arm_id
            assert restored.payload.receipt.request_payload_sha256 == request.payload_sha256
            assert restored.payload.receipt.outcome in {"incorrect", "invalid_output"}
            assert (
                verify_study_code_evaluation_v1(
                    restored,
                    benchmark=benchmark,
                    request_path=request_path,
                    output_text=execution.generation.text,
                )
                == restored
            )
            if checkpoint is None:
                assert restored.payload.receipt.checkpoint_transfer is None
            else:
                binding = restored.payload.receipt.checkpoint_transfer
                assert binding is not None
                assert binding.checkpoint_payload_sha256 == checkpoint.identity.payload_sha256
                assert binding.completed_steps == 1
                # Corrupt only this test's checkpoint; schema/hash-only receipt reading
                # still works, but dependency verification must now fail.
                state_path = cast(Path, checkpoint_path) / "state.safetensors"
                content = bytearray(state_path.read_bytes())
                content[-1] ^= 1
                state_path.write_bytes(content)
                assert read_study_code_evaluation_v1(evaluation_path) == restored
                with pytest.raises(IrisStudyEvaluationError):
                    verify_study_code_evaluation_v1(
                        restored,
                        benchmark=benchmark,
                        request_path=request_path,
                        output_text=execution.generation.text,
                    )
            forbidden_model_loader.assert_not_called()
        no_execution.assert_not_called()
    _empty_inventory()


@pytest.mark.parametrize("variant", ["reference", "nondeterministic_diagnostics", "syntax"])
def test_real_reference_and_private_binary_diagnostics_replay_without_execution(
    tmp_path: Path, variant: str
) -> None:
    benchmark, _, _, source = _fixture(tmp_path / "reference")
    if variant == "nondeterministic_diagnostics":
        source += "\nimport os\nos.write(2, os.urandom(64))\n"
    elif variant == "syntax":
        source = "def solve(values) invalid syntax\n"
    benchmark, path, identity, source = _fixture(tmp_path / "actual", output=source)
    with _assets(_tokenizer()):
        result = evaluate_study_request_python_answer_v1(
            benchmark,
            request_path=path,
            example_id=identity,
            output_text=source,
            terminal_token_id=1,
            policy=_POLICY,
        )
        assert result.payload.receipt.outcome == (
            "invalid_output" if variant == "syntax" else "correct"
        )
        if variant == "nondeterministic_diagnostics":
            assert result.payload.observation.stderr.byte_length == 64
        destination = tmp_path / "private-code.json"
        write_study_code_evaluation_v1(result, path=destination)
        with patch("iris.application.study_code_evaluation.execute_python_job_v1") as no_execution:
            assert (
                verify_study_code_evaluation_v1(
                    read_study_code_evaluation_v1(destination),
                    benchmark=benchmark,
                    request_path=path,
                    output_text=source,
                )
                == result
            )
        no_execution.assert_not_called()
    _empty_inventory()
