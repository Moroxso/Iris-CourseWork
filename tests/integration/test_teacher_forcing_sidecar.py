"""Offline CPU integration for the differentiable teacher-forced sidecar."""

import math
from types import SimpleNamespace
from typing import Any, Never, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional

from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone import BackboneAdapter
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingInput,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationAccumulator,
)
from iris.utils.devices import DeviceInfo

pytestmark = pytest.mark.integration

CPU = torch.device("cpu")
HIDDEN_SIZE = 6
STATE_DIM = 4
NUM_MODULES = 3
NUM_SLOTS = 2
CYCLES = 3
WINNERS = 2
VOCABULARY_SIZE = 16


class _TinyTokenizer:
    padding_side = "right"
    truncation_side = "right"
    eos_token_id = 15
    all_special_ids = (15,)

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append((text, kwargs))
        if kwargs.get("add_special_tokens") is False:
            target_ids = {
                "short target": [5],
                "longer target": [5, 6],
            }[text]
            return {
                "input_ids": torch.tensor([target_ids], dtype=torch.int64),
                "attention_mask": torch.ones((1, len(target_ids)), dtype=torch.int64),
            }
        return {
            "input_ids": torch.tensor([[0, 2, 7]], dtype=torch.int64),
            "attention_mask": torch.tensor([[0, 1, 1]], dtype=torch.int64),
        }


class _TinyFrozenCausalLM(nn.Module):
    """A tiny frozen LM that retains gradients only to supplied embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY_SIZE, HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCABULARY_SIZE, bias=False)
        self.config = SimpleNamespace(
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=4,
            vocab_size=VOCABULARY_SIZE,
            max_position_embeddings=10,
        )
        self.prompt_calls = 0
        self.causal_calls: list[dict[str, object]] = []
        self.last_causal_logits: torch.Tensor | None = None
        self.last_labels: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def generate(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("teacher forcing must not call model.generate")

    def forward(self, **kwargs: object) -> SimpleNamespace:
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            self.prompt_calls += 1
            embedding_state = self.embedding(input_ids)
            hidden_states = (
                embedding_state,
                *(torch.tanh(embedding_state + 0.125 * index) for index in range(1, 5)),
            )
            return SimpleNamespace(
                logits=self.lm_head(hidden_states[-1]),
                hidden_states=hidden_states,
            )

        self.causal_calls.append(dict(kwargs))
        inputs_embeds = cast(torch.Tensor, kwargs["inputs_embeds"])
        labels = cast(torch.Tensor, kwargs["labels"])
        logits = self.lm_head(inputs_embeds)
        loss = functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCABULARY_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        self.last_causal_logits = logits
        self.last_labels = labels
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=None)


def _backbone_config() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "tiny-frozen",
            "repository": "offline/tiny",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": "fp32",
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {"max_prompt_tokens": 5, "max_new_tokens": 3},
        }
    )


def _cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": NUM_MODULES,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-workspace",
            "num_slots": NUM_SLOTS,
            "cycles": CYCLES,
            "winners_per_cycle": WINNERS,
            "decay_rate": 0.5,
            "empty_threshold": 1e-4,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 1.0,
                "relevance_weight": 1.0,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def _build_offline_model() -> tuple[Any, _TinyTokenizer, _TinyFrozenCausalLM]:
    tokenizer = _TinyTokenizer()
    frozen_model = _TinyFrozenCausalLM()
    device_info = DeviceInfo(
        device=CPU,
        backend="cpu",
        name="Test CPU",
        total_memory_bytes=None,
        supports_fp16=False,
        supports_bf16=False,
    )
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            return_value=frozen_model,
        ),
    ):
        backbone = BackboneAdapter(_backbone_config(), device_info)
    teacher = build_teacher_forcing_model(
        backbone=backbone,
        cognitive_config=_cognitive_config(),
        workspace_config=_workspace_config(),
    )
    return teacher, tokenizer, frozen_model


def _assert_finite_gradient(parameter: nn.Parameter) -> None:
    assert parameter.grad is not None
    assert bool(torch.isfinite(parameter.grad).all())


def test_real_sidecar_backpropagates_through_frozen_causal_lm_offline() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, tokenizer, frozen_model = _build_offline_model()
        example = IrisTeacherForcingInput(
            prompt="offline teacher prompt",
            target_input_ids=torch.tensor([[5, 6]], dtype=torch.int64),
            target_attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        )

        output = teacher(example)

        assert output.loss.ndim == 0
        assert output.loss.requires_grad
        assert output.loss.grad_fn is not None
        assert output.supervised_token_count == 2
        assert len(output.workspace_trace) == CYCLES
        assert frozen_model.prompt_calls == 1
        assert len(frozen_model.causal_calls) == 1
        causal_call = frozen_model.causal_calls[0]
        assert causal_call["use_cache"] is False
        assert "past_key_values" not in causal_call
        assert "logits_to_keep" not in causal_call
        labels = cast(torch.Tensor, causal_call["labels"])
        assert torch.equal(
            labels,
            torch.tensor([[-100, -100, -100, -100, -100, 5, 6]]),
        )
        assert tokenizer.calls == [
            (
                "offline teacher prompt",
                {
                    "return_tensors": "pt",
                    "truncation": True,
                    "max_length": 5,
                    "return_attention_mask": True,
                },
            )
        ]

        output.loss.backward()

        _assert_finite_gradient(teacher.latent_bridge.bridge_projection.weight)
        _assert_finite_gradient(teacher.layer_sampler.projections[0].weight)
        _assert_finite_gradient(teacher.proposal_head.bid_head.weight)
        _assert_finite_gradient(teacher.slot_allocator.slot_projection.weight)
        assert all(not parameter.requires_grad for parameter in frozen_model.parameters())
        assert all(parameter.grad is None for parameter in frozen_model.parameters())


def test_owned_adamw_step_updates_only_the_real_sidecar_offline() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, tokenizer, frozen_model = _build_offline_model()
        example = prepare_teacher_forcing_input(
            backbone=cast(BackboneAdapter, teacher._backbone),
            prompt="offline teacher prompt",
            target_text="longer target",
            terminal_token_id=15,
        )
        optimizer = IrisTeacherOptimizer(
            model=teacher,
            config=IrisAdamWConfig(
                learning_rate=1e-3,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.0,
                max_gradient_norm=1e-12,
            ),
        )
        sidecar_before = {
            name: parameter.detach().clone() for name, parameter in teacher.named_parameters()
        }
        backbone_before = {
            name: parameter.detach().clone() for name, parameter in frozen_model.named_parameters()
        }

        result = optimizer.step(example)

        assert result.step_index == 1
        assert result.supervised_token_count == 3
        assert math.isfinite(result.loss) and result.loss >= 0.0
        assert math.isfinite(result.gradient_l2_norm) and result.gradient_l2_norm > 0.0
        assert result.gradients_clipped is True
        assert optimizer.completed_steps == 1
        assert any(
            not torch.equal(parameter, sidecar_before[name])
            for name, parameter in teacher.named_parameters()
        )
        assert all(parameter.grad is None for parameter in teacher.parameters())
        for name, parameter in frozen_model.named_parameters():
            assert torch.equal(parameter, backbone_before[name])
            assert not parameter.requires_grad
            assert parameter.grad is None
        sidecar_ids = {id(parameter) for parameter in teacher.parameters()}
        backbone_ids = {id(parameter) for parameter in frozen_model.parameters()}
        optimized_ids = {
            id(parameter)
            for group in optimizer._optimizer.param_groups
            for parameter in group["params"]
        }
        assert optimized_ids == sidecar_ids
        assert optimized_ids.isdisjoint(backbone_ids)
        assert frozen_model.prompt_calls == 1
        assert len(frozen_model.causal_calls) == 1
        assert [text for text, _ in tokenizer.calls] == [
            "longer target",
            "offline teacher prompt",
        ]


def test_evaluation_matches_hard_forward_without_mutating_state_rng_or_gradients() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, _, frozen_model = _build_offline_model()
        example = IrisTeacherForcingInput(
            prompt="offline teacher prompt",
            target_input_ids=torch.tensor([[5, 6]], dtype=torch.int64),
            target_attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        )
        train_output = teacher.training_forward(example)
        train_output.loss.backward()
        parameters_before = {
            name: parameter.detach().clone() for name, parameter in teacher.named_parameters()
        }
        gradients_before = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in teacher.named_parameters()
        }
        rng_before = torch.get_rng_state().clone()

        teacher.eval()
        evaluation = teacher.evaluation_forward(example)

        assert not teacher.training
        assert all(not module.training for module in teacher.modules())
        assert not evaluation.loss.requires_grad
        assert evaluation.loss.grad_fn is None
        torch.testing.assert_close(evaluation.loss, train_output.loss.detach())
        assert evaluation.supervised_token_count == train_output.supervised_token_count
        for training_state, evaluation_state in zip(
            train_output.workspace_trace,
            evaluation.workspace_trace,
            strict=True,
        ):
            torch.testing.assert_close(evaluation_state.slots, training_state.slots.detach())
        assert torch.equal(torch.get_rng_state(), rng_before)
        for name, parameter in teacher.named_parameters():
            assert torch.equal(parameter, parameters_before[name])
            expected_gradient = gradients_before[name]
            if expected_gradient is None:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None
                assert torch.equal(parameter.grad, expected_gradient)

        assert frozen_model.last_causal_logits is not None
        assert frozen_model.last_labels is not None
        manual_loss = functional.cross_entropy(
            frozen_model.last_causal_logits[:, :-1, :].reshape(-1, VOCABULARY_SIZE),
            frozen_model.last_labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(evaluation.loss, manual_loss)


def test_prepared_text_targets_aggregate_real_evaluation_by_supervised_tokens() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, tokenizer, _ = _build_offline_model()
        teacher.eval()
        accumulator = IrisTokenWeightedEvaluationAccumulator()
        observed: list[tuple[float, int]] = []

        for target_text in ("short target", "longer target"):
            example = prepare_teacher_forcing_input(
                backbone=cast(BackboneAdapter, teacher._backbone),
                prompt="offline teacher prompt",
                target_text=target_text,
                terminal_token_id=15,
            )
            output = teacher.evaluation_forward(example)
            accumulator.update(output)
            observed.append((float(output.loss), output.supervised_token_count))

        summary = accumulator.summary()

    expected_tokens = sum(token_count for _, token_count in observed)
    expected_loss = sum(loss * token_count for loss, token_count in observed) / expected_tokens
    assert [token_count for _, token_count in observed] == [2, 3]
    assert summary.example_count == 2
    assert summary.supervised_token_count == expected_tokens
    assert summary.mean_loss == pytest.approx(expected_loss)
    assert summary.perplexity == pytest.approx(torch.exp(torch.tensor(expected_loss)).item())
    assert [text for text, _ in tokenizer.calls] == [
        "short target",
        "offline teacher prompt",
        "longer target",
        "offline teacher prompt",
    ]
