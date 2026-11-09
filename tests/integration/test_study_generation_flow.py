"""Offline complete-split A-D generation using tiny random Qwen weights only."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from iris.application import study_generation as service
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
from iris.application.study_generation import generate_study_split_v1, verify_study_generation_v1
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.artifacts.study_v1 import StudyArmIdV1
from iris.backbone import BackboneAdapter
from iris.training import IrisAdamWConfig, IrisTeacherForcingInput, IrisTeacherOptimizer
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_verification_v4_flow import _config
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_study_input_preflight import _tokenizer

pytestmark = pytest.mark.integration


def _model() -> Qwen3ForCausalLM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        return Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
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


def _policies(*, sampling: bool = False) -> tuple[DecodingProvenanceV1, ...]:
    return tuple(
        DecodingProvenanceV1(
            strategy="sampling" if sampling else "greedy",
            stop_token_ids=tuple(range(16)),
            temperature=0.75,
            top_p=0.875,
            sampling_seed=1907 + index if sampling else None,
            allow_download=False,
        )
        for index in range(5)
    )


@pytest.mark.parametrize(
    "arm_id,sampling", [("A", False), ("B", False), ("C", False), ("D", False), ("A", True)]
)
def test_complete_split_uses_one_model_load_and_preserves_all_private_answers(
    tmp_path: Path,
    arm_id: StudyArmIdV1,
    sampling: bool,
) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    config = _config(tmp_path / "artifacts")
    config.backbone.limits.max_new_tokens = 128
    model, tokenizer = _model(), _tokenizer()
    checkpoint = None
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as loader,
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
            write_arm_teacher_checkpoint(checkpoint, root=Path(config.runtime.output_dir))
        before_loads = loader.call_count
        before = capture_runtime_state(CPU_INFO.device)
        path = tmp_path / "generation.json"
        result = generate_study_split_v1(
            benchmark,
            config=config,
            arm_id=arm_id,
            checkpoint_identity=None if checkpoint is None else checkpoint.identity,
            split="test",
            decoding=_policies(sampling=sampling),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
            path=path,
        )
        after = capture_runtime_state(CPU_INFO.device)
        assert torch.equal(before.cpu_rng_state, after.cpu_rng_state)
        assert before.python_rng_state == after.python_rng_state
        assert before.deterministic_algorithms == after.deterministic_algorithms
        assert before.deterministic_warn_only == after.deterministic_warn_only
        assert before.cudnn_deterministic == after.cudnn_deterministic
        assert before.cudnn_benchmark == after.cudnn_benchmark
        assert loader.call_count == before_loads + 1
        loaded = read_study_generation_v1(path)
        assert result.payload_sha256 == loaded.payload_sha256
        selected = tuple(
            item
            for item in benchmark.payload.dataset.payload.manifest.payload.examples
            if item.split == "test"
        )
        assert tuple(item.example_id for item in loaded.payload.answers) == tuple(
            item.example_id for item in selected
        )
        assert loaded.payload.arm_id == arm_id
        assert len(loaded.payload.answers) == 5
        assert len({item.task_family for item in loaded.payload.answers}) == 5
        assert loaded.payload.status == "complete_generation_not_scored"
        assert tuple(item.decoding for item in loaded.payload.answers) == _policies(
            sampling=sampling
        )
        assert (
            verify_study_generation_v1(
                loaded, benchmark=benchmark, request_root=Path(config.runtime.output_dir)
            )
            == loaded
        )
        assert loader.call_count == before_loads + 1
        assert len(tuple((Path(config.runtime.output_dir) / "requests" / "v4").glob("*.json"))) == 5


def test_full_corpus_overflow_prevents_weight_loading_even_for_one_split(tmp_path: Path) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    config = _config(tmp_path / "artifacts")
    config.backbone.limits.max_new_tokens = 2
    model, tokenizer = _model(), _tokenizer()
    path = tmp_path / "uncreated" / "generation.json"
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        pytest.raises(IrisStudyGenerationError),
    ):
        generate_study_split_v1(
            benchmark,
            config=config,
            arm_id="A",
            checkpoint_identity=None,
            split="test",
            decoding=_policies(),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
            path=path,
        )
    loader.assert_not_called()
    assert not path.parent.exists()
    assert not Path(config.runtime.output_dir).exists()


def test_publication_failure_does_not_leave_global_runtime_policy_changed(tmp_path: Path) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    config = _config(tmp_path / "artifacts")
    config.backbone.limits.max_new_tokens = 128
    model, tokenizer = _model(), _tokenizer()
    path = tmp_path / "generation.json"
    before = capture_runtime_state(CPU_INFO.device)
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=model.config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch.object(
            service, "write_request_artifact_v4", side_effect=OSError("fixture disk failure")
        ),
        pytest.raises(IrisStudyGenerationError),
    ):
        generate_study_split_v1(
            benchmark,
            config=config,
            arm_id="A",
            checkpoint_identity=None,
            split="test",
            decoding=_policies(),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
            path=path,
        )
    after = capture_runtime_state(CPU_INFO.device)
    assert torch.equal(before.cpu_rng_state, after.cpu_rng_state)
    assert before.python_rng_state == after.python_rng_state
    assert before.deterministic_algorithms == after.deterministic_algorithms
    assert not path.exists()
