import gc
import os
import re
from collections.abc import Iterator
from importlib.resources import files

import pytest
import torch
from omegaconf import OmegaConf

from iris.application import (
    build_base_llm_request_orchestrator,
    build_no_workspace_request_orchestrator,
    build_no_workspace_teacher_forcing_model,
    prepare_teacher_forcing_input,
)
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import BackboneConfig, CognitiveConfig
from iris.utils.devices import inspect_device

RUN_ENVIRONMENT_VARIABLE = "IRIS_RUN_BACKBONE_SMOKE"
CONTROL_PROMPTS = (
    "Напиши функцию Python add(a, b), которая возвращает сумму двух чисел.",
    "Объясни одним предложением, что такое глобальное рабочее пространство.",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get(RUN_ENVIRONMENT_VARIABLE) != "1",
        reason=f"set {RUN_ENVIRONMENT_VARIABLE}=1 to run the real backbone smoke test",
    ),
]


def _load_packaged_config() -> BackboneConfig:
    resource = files("iris.configs").joinpath("backbone", "qwen3_0_6b.yaml")
    config = OmegaConf.create(resource.read_text(encoding="utf-8"))
    return BackboneConfig.model_validate(OmegaConf.to_container(config))


def _load_packaged_cognitive_config() -> CognitiveConfig:
    resource = files("iris.configs").joinpath("cognitive", "iris_gw_0_1.yaml")
    config = OmegaConf.create(resource.read_text(encoding="utf-8"))
    return CognitiveConfig.model_validate(OmegaConf.to_container(config))


@pytest.fixture(scope="module")
def real_backbone() -> Iterator[tuple[BackboneConfig, BackboneAdapter]]:
    """Load the pinned model once for both canonical CPU smoke prompts."""
    packaged = _load_packaged_config()
    smoke_limits = packaged.limits.model_copy(update={"max_new_tokens": 2})
    config = packaged.model_copy(update={"limits": smoke_limits})
    adapter = BackboneAdapter(config, inspect_device("cpu"))
    yield config, adapter
    del adapter
    gc.collect()


def test_backbone_smoke_uses_immutable_revision() -> None:
    config = _load_packaged_config()

    assert config.revision == config.tokenizer_revision
    assert config.revision != "main"
    assert re.fullmatch(r"[0-9a-f]{40}", config.revision)
    assert config.dtype == "fp32"
    assert config.trust_remote_code is False


@pytest.mark.parametrize("prompt", CONTROL_PROMPTS)
def test_pinned_qwen_cpu_forward(
    real_backbone: tuple[BackboneConfig, BackboneAdapter], prompt: str
) -> None:
    config, adapter = real_backbone

    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False

    output = adapter.forward_prompt(prompt)
    sequence_length = output.attention_mask.shape[1]

    assert output.logits.ndim == 3
    assert output.logits.shape == (1, sequence_length, adapter.vocabulary_size)
    assert output.logits.device.type == "cpu"
    assert output.logits.dtype == torch.float32
    assert output.logits.requires_grad is False
    assert output.logits.grad_fn is None
    assert bool(torch.isfinite(output.logits).all())

    assert 0 < sequence_length <= config.limits.max_prompt_tokens
    assert len(output.hidden_states) == adapter.num_hidden_layers + 1
    for hidden_state in output.hidden_states:
        assert hidden_state.shape == (1, sequence_length, adapter.hidden_size)
        assert hidden_state.device.type == "cpu"
        assert hidden_state.dtype == torch.float32
        assert hidden_state.requires_grad is False
        assert hidden_state.grad_fn is None
        assert bool(torch.isfinite(hidden_state).all())

    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False


def test_pinned_qwen_cpu_arm_a_prompt_only_generation(
    real_backbone: tuple[BackboneConfig, BackboneAdapter],
) -> None:
    config, adapter = real_backbone
    graph = build_base_llm_request_orchestrator(
        backbone=adapter,
        stop_token_ids=adapter.eos_token_ids,
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )

    output = graph.execute(CONTROL_PROMPTS[0], strategy="greedy")

    assert config.limits.max_new_tokens == 2
    assert output.generation.token_ids.shape[0] == 1
    assert 1 <= output.generation.token_ids.shape[1] <= 2
    assert output.generation.workspace_trace == ()
    assert output.prompt_input_ids.shape == output.prompt_attention_mask.shape
    assert output.prompt_input_ids.shape[0] == 1
    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False


def test_pinned_qwen_cpu_arm_b_no_workspace_generation(
    real_backbone: tuple[BackboneConfig, BackboneAdapter],
) -> None:
    config, adapter = real_backbone
    graph = build_no_workspace_request_orchestrator(
        backbone=adapter,
        cognitive_config=_load_packaged_cognitive_config(),
        stop_token_ids=adapter.eos_token_ids,
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )

    output = graph.execute(CONTROL_PROMPTS[0], strategy="greedy")

    assert config.limits.max_new_tokens == 2
    assert graph.cycles == 3
    assert len(graph.sidecar_state_sha256) == 64
    assert output.generation.token_ids.shape[0] == 1
    assert 1 <= output.generation.token_ids.shape[1] <= 2
    assert output.generation.workspace_trace == ()
    assert output.prompt_input_ids.shape == output.prompt_attention_mask.shape
    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False


def test_pinned_qwen_cpu_arm_b_teacher_forcing(
    real_backbone: tuple[BackboneConfig, BackboneAdapter],
) -> None:
    config, adapter = real_backbone
    teacher = build_no_workspace_teacher_forcing_model(
        backbone=adapter,
        cognitive_config=_load_packaged_cognitive_config(),
    )
    example = prepare_teacher_forcing_input(
        backbone=adapter,
        prompt="2+2=",
        target_text="4",
        terminal_token_id=adapter.eos_token_ids[0],
    )
    provenance = teacher.checkpoint_provenance

    assert config.limits.max_new_tokens == 2
    assert example.target_input_ids.shape == (1, 2)
    assert example.target_attention_mask.shape == (1, 2)
    assert teacher.cycles == provenance.cognitive_cycles == 3
    assert provenance.arm_id == "B"
    assert provenance.teacher_graph == "independent_recurrent_modules_zero_memory_read_teacher_v1"
    assert all("workspace" not in name for name in provenance.component_names)
    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False
    assert all(parameter.grad is None for parameter in teacher.parameters())

    output = teacher.training_forward(example)

    assert output.workspace_trace == ()
    assert output.supervised_token_count == 2
    assert output.loss.ndim == 0
    assert output.loss.requires_grad
    assert output.loss.grad_fn is not None
    assert bool(torch.isfinite(output.loss))

    torch.autograd.backward((output.loss,))

    sidecar_gradients = tuple(
        parameter.grad for parameter in teacher.parameters() if parameter.grad is not None
    )
    assert sidecar_gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in sidecar_gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in sidecar_gradients)
    assert adapter.trainable_parameter_count == 0
    assert adapter.has_parameter_gradients is False
