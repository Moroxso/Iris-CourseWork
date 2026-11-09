"""Full five-term teacher on genuine tiny random Qwen, with no pretrained downloads."""

from contextlib import AbstractContextManager
from copy import deepcopy
from hashlib import sha256
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from iris.application.scientific_teacher import (
    IrisScientificObjectivePolicyV1,
    IrisScientificTeacherError,
    IrisScientificTeacherV1,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone import BackboneAdapter
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training.losses import IrisLossWeights, workspace_diversity_loss, workspace_norm_loss
from iris.training.probes import (
    prepare_probe_targets,
    synthetic_probe_vocabulary_v1,
    workspace_probe_objective,
)
from iris.utils.hashing import canonical_json_bytes
from tests.unit.test_bootstrap import CPU_INFO

pytestmark = pytest.mark.integration


def _rows() -> tuple[IrisExample, ...]:
    return tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=15),
            template_id="iris-synthetic-01-v2",
        )
    )


def _build(
    *,
    seed: int = 1907,
    slots: int = 4,
    cycles: int = 3,
    policy: IrisScientificObjectivePolicyV1 | None = None,
) -> tuple[IrisScientificTeacherV1, PreTrainedTokenizerFast, Qwen3ForCausalLM]:
    pretokenizer = Whitespace()
    words = sorted(
        {
            word
            for row in _rows()
            for text in (row.prompt, row.expected_response)
            for word, _ in pretokenizer.pre_tokenize_str(text)
        }
    )
    vocabulary = {word: index + 3 for index, word in enumerate(words)}
    vocabulary |= {"[UNK]": 0, "<eos>": 1, "<bos>": 2}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pretokenizer
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        eos_token="<eos>",
        bos_token="<bos>",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        lm = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
            Qwen3Config(
                vocab_size=len(vocabulary),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=2048,
                bos_token_id=2,
                eos_token_id=1,
                tie_word_embeddings=True,
                attention_dropout=0.0,
            )
        )
        config = BackboneConfig.model_validate(
            {
                "name": "scientific-teacher-random-qwen",
                "repository": "offline/random-qwen3",
                "revision": "random-fixture-v1",
                "tokenizer_revision": "wordlevel-fixture-v1",
                "trust_remote_code": False,
                "local_files_only": True,
                "output_hidden_states": True,
                "use_cache": False,
                "dtype": "fp32",
                "tokenizer": {"use_fast": True, "padding_side": "left", "truncation_side": "left"},
                "limits": {"max_prompt_tokens": 768, "max_new_tokens": 256},
            }
        )
        with (
            patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=lm),
        ):
            backbone = BackboneAdapter(config, CPU_INFO)
        teacher = build_teacher_forcing_model(
            backbone=backbone,
            cognitive_config=CognitiveConfig.model_validate(
                {
                    "name": "scientific-tiny",
                    "state_dim": 8,
                    "num_modules": 4,
                    "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
                }
            ),
            workspace_config=WorkspaceConfig.model_validate(
                {
                    "name": "scientific-tiny",
                    "num_slots": slots,
                    "cycles": cycles,
                    "winners_per_cycle": min(2, slots),
                    "decay_rate": 0.1,
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
            ),
        )
        model = IrisScientificTeacherV1(
            teacher=teacher,
            vocabulary=synthetic_probe_vocabulary_v1(),
            probe_init_seed=42,
            policy=policy or IrisScientificObjectivePolicyV1(),
        )
    return model, tokenizer, lm


def test_full_qwen_objective_has_source_coefficients_and_updates_sidecar_and_probes() -> None:
    model, _, lm = _build()
    row = _rows()[0]
    named = dict(model.named_parameters())
    assert any(name.startswith("teacher.") for name in named)
    assert any(name.startswith("probes.") for name in named)
    assert set(map(id, lm.parameters())).isdisjoint(map(id, named.values()))
    frozen_before = deepcopy(lm.state_dict())
    before = {name: value.detach().clone() for name, value in named.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    with patch.object(lm, "forward", wraps=lm.forward) as forward:
        output = model(row, terminal_token_id=1)
    assert forward.call_count == 2  # One frozen prompt pass and one differentiable causal pass.
    losses = output.losses
    assert torch.equal(
        losses.language_model, output.trace.teacher.loss * output.supervised_token_count
    )
    assert torch.equal(
        losses.diversity, workspace_diversity_loss(output.trace.teacher.workspace_trace[-1].slots)
    )
    assert torch.equal(
        losses.norm, workspace_norm_loss(output.trace.teacher.workspace_trace[-1].slots)
    )
    expected = (
        losses.language_model
        + 0.05 * losses.probe
        + 0.01 * losses.diversity
        + 0.01 * losses.utilization
        + 0.001 * losses.norm
    )
    assert torch.allclose(losses.total, expected)
    assert output.example_id == row.id and len(output.example_sha256) == 64
    assert (
        output.usage.module_probabilities.requires_grad
        and output.usage.slot_probabilities.requires_grad
    )
    torch.autograd.backward(losses.total)
    # Existing TTL is consumed only by a hard expiry comparison. Do not conceal
    # that disconnected head with an arbitrary artificial auxiliary target.
    assert {name for name, parameter in named.items() if parameter.grad is None} == {
        "teacher.proposal_head.ttl_head.weight"
    }
    for prefix in (
        "teacher.layer_sampler",
        "teacher.goal_encoder",
        "teacher.cognitive_modules",
        "teacher.proposal_head",
        "teacher.latent_bridge",
        "probes.heads",
    ):
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if name.startswith(prefix) and name != "teacher.proposal_head.ttl_head.weight"
        ]
        assert gradients and all(
            value is not None and bool(torch.isfinite(value).all()) for value in gradients
        )
        assert any(value is not None and bool((value.abs() > 0).any()) for value in gradients)
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in lm.parameters()
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    for prefix in ("teacher.", "probes."):
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in named.items()
            if name.startswith(prefix)
        )
    assert all(torch.equal(value, lm.state_dict()[name]) for name, value in frozen_before.items())


def test_auxiliary_only_gradients_reach_real_sidecar_without_language_model_loss() -> None:
    model, _, lm = _build()
    output = model(_rows()[0], terminal_token_id=1)
    auxiliary = (
        0.05 * output.losses.probe
        + 0.01 * output.losses.diversity
        + 0.01 * output.losses.utilization
        + 0.001 * output.losses.norm
    )
    torch.autograd.backward(auxiliary)
    assert any(
        parameter.grad is not None and bool((parameter.grad.abs() > 0).any())
        for parameter in model.teacher.cognitive_modules.parameters()
    )
    assert all(parameter.grad is not None for parameter in model.probes.parameters())
    assert all(parameter.grad is None for parameter in lm.parameters())


def test_trace_path_preserves_legacy_forward_rng_and_current_state_exactly() -> None:
    model, _, _ = _build()
    row = _rows()[0]
    prepared = prepare_teacher_forcing_input(
        backbone=model.teacher.backbone,
        prompt=row.prompt,
        target_text=row.expected_response,
        terminal_token_id=1,
    )
    before = deepcopy(model.teacher.state_dict())
    rng = torch.random.get_rng_state().clone()
    legacy = model.teacher.training_forward(prepared)
    traced = model.teacher.training_forward_with_trace(prepared)
    assert torch.equal(legacy.loss, traced.teacher.loss)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(
        torch.equal(value, model.teacher.state_dict()[name]) for name, value in before.items()
    )
    assert all(
        torch.equal(left.slots, right.slots)
        for left, right in zip(legacy.workspace_trace, traced.teacher.workspace_trace, strict=True)
    )
    assert len(traced.cycles) == 3
    assert all(
        cycle.module_probabilities.requires_grad and cycle.allocation_scores.requires_grad
        for cycle in traced.cycles
    )


def test_full_row_target_and_probe_labels_are_bound_without_hidden_fact_input() -> None:
    model, tokenizer, _ = _build()
    row = _rows()[0]
    with (
        patch.object(
            model.teacher.backbone,
            "tokenize_teacher_target",
            wraps=model.teacher.backbone.tokenize_teacher_target,
        ) as target,
        patch.object(
            model.teacher.backbone, "forward_prompt", wraps=model.teacher.backbone.forward_prompt
        ) as prompt,
    ):
        output = model(row, terminal_token_id=1)
    target.assert_called_once_with(row.expected_response, terminal_token_id=1)
    prompt.assert_called_once_with(row.prompt)
    labels = prepare_probe_targets(
        (row,), vocabulary=model.probes.vocabulary, purpose="training", device=torch.device("cpu")
    )
    expected = workspace_probe_objective(
        model.probes, output.trace.teacher.workspace_trace[-1].slots, labels, head_reduction="sum"
    )
    assert torch.equal(output.losses.probe, expected)
    encoded = tokenizer(row.expected_response, add_special_tokens=False)["input_ids"]
    assert output.supervised_token_count == len(encoded) + 1
    with pytest.raises(TypeError):
        model(row, terminal_token_id=1, labels=labels)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_training_fails_before_any_target_or_backbone_call(split: str) -> None:
    model, _, _ = _build()
    row = _rows()[0].model_copy(update={"split": split, "id": f"{split}-0"})
    with (
        patch.object(model.teacher.backbone, "tokenize_teacher_target") as target,
        patch.object(model.teacher.backbone, "forward_prompt") as prompt,
        pytest.raises(IrisScientificTeacherError),
    ):
        model(row, terminal_token_id=1)
    target.assert_not_called()
    prompt.assert_not_called()


def test_analysis_is_no_grad_preserves_parameters_existing_gradients_and_modes() -> None:
    model, _, lm = _build()
    row = _rows()[0].model_copy(update={"split": "validation", "id": "validation-0"})
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = deepcopy(model.state_dict())
    model.eval()
    with torch.enable_grad():
        output = model.evaluation_forward(row, terminal_token_id=1)
    assert not output.losses.total.requires_grad and not output.losses.probe.requires_grad
    assert not output.usage.module_probabilities.requires_grad
    assert all(not child.training for child in model.modules())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(
        parameter.grad is not None and bool((parameter.grad == 1).all())
        for parameter in model.parameters()
    )
    assert all(parameter.grad is None for parameter in lm.parameters())


def test_bad_owned_row_and_callback_errors_are_content_redacted() -> None:
    model, _, _ = _build()
    row = _rows()[0].model_copy(update={"id": "PRIVATE-invalid"})
    with pytest.raises(IrisScientificTeacherError) as caught:
        model(row, terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value) and caught.value.__suppress_context__
    with (
        patch.object(
            model.teacher.backbone,
            "tokenize_teacher_target",
            side_effect=ValueError("PRIVATE callback payload"),
        ),
        pytest.raises(IrisScientificTeacherError) as caught,
    ):
        model(_rows()[0], terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value) and caught.value.__suppress_context__


@pytest.mark.parametrize("kind", ["policy", "probe_identity", "missing", "root_identity"])
def test_incompatible_state_rejected_before_teacher_child_weights_are_loaded(kind: str) -> None:
    model, _, _ = _build()
    state = deepcopy(model.state_dict())
    if kind == "policy":
        changed, _, _ = _build(
            policy=IrisScientificObjectivePolicyV1(weights=IrisLossWeights(probe=0.1))
        )
        state = deepcopy(changed.state_dict())
    elif kind == "missing":
        del state["configuration_identity"]
    else:
        key = (
            "probes.configuration_identity"
            if kind == "probe_identity"
            else "configuration_identity"
        )
        state[key][0] ^= 1
    first = next(iter(dict(model.teacher.named_parameters())))
    state["teacher." + first].fill_(987)
    before = deepcopy(model.state_dict())
    with pytest.raises(IrisScientificTeacherError):
        model.load_state_dict(state, strict=False)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_compatible_registered_state_roundtrip_preserves_objective() -> None:
    model, _, _ = _build()
    restored, _, _ = _build()
    restored.load_state_dict(deepcopy(model.state_dict()))
    assert torch.equal(
        model(_rows()[0], terminal_token_id=1).losses.total,
        restored(_rows()[0], terminal_token_id=1).losses.total,
    )


@pytest.mark.parametrize("mode", ["eval", "no_grad", "inference", "autocast", "child_eval"])
def test_bad_training_context_fails_before_tokenization(mode: str) -> None:
    model, _, _ = _build()
    if mode == "eval":
        model.eval()
    if mode == "child_eval":
        model.probes.eval()
    context: AbstractContextManager[Any] = torch.enable_grad()
    if mode == "no_grad":
        context = torch.no_grad()
    elif mode == "inference":
        context = torch.inference_mode()
    elif mode == "autocast":
        context = torch.autocast("cpu", dtype=torch.bfloat16)
    with (
        context,
        patch.object(model.teacher.backbone, "tokenize_teacher_target") as target,
        pytest.raises(IrisScientificTeacherError),
    ):
        model(_rows()[0], terminal_token_id=1)
    target.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("lm_reduction", "mean"),
        ("probe_head_reduction", "mean"),
        ("diversity_margin", float("nan")),
        ("extra", True),
    ],
)
def test_policy_rejects_silent_alternative_reductions(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        IrisScientificObjectivePolicyV1.model_validate({field: value})


@pytest.mark.parametrize("slots,cycles", [(1, 1), (8, 1), (4, 3)])
def test_partial_occupancy_and_capacity_variants_are_finite(slots: int, cycles: int) -> None:
    model, _, _ = _build(slots=slots, cycles=cycles)
    output = model(_rows()[0], terminal_token_id=1)
    assert bool(torch.isfinite(output.losses.total))
    assert sum(parameter.numel() for parameter in model.probes.parameters()) == 22 * 9
    if cycles == 1:
        assert torch.allclose(output.usage.slot_probabilities, torch.full((slots,), 1 / slots))
    if slots == 1:
        assert output.losses.diversity.item() == 0


def test_usage_trace_is_call_owned_and_slot_surrogate_has_a_real_gradient_after_occupancy() -> None:
    model, _, _ = _build(slots=4, cycles=5)
    captured = []
    handle = model.teacher.slot_allocator.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    try:
        result = model(_rows()[0], terminal_token_id=1)
    finally:
        handle.remove()
    for raw, cycle in zip(captured, result.trace.cycles, strict=True):
        assert raw.allocation_scores.data_ptr() != cycle.allocation_scores.data_ptr()
        assert torch.equal(raw.allocation_scores, cycle.allocation_scores)
    torch.autograd.backward(result.losses.utilization)
    weight = model.teacher.slot_allocator.slot_projection.weight
    assert weight.grad is not None and bool(torch.isfinite(weight.grad).all())
    assert bool((weight.grad.abs() > 0).any())


@pytest.mark.parametrize(
    "family",
    [
        "split_spec_repair",
        "delayed_goal",
        "conflict_resolution",
        "next_tool_prediction",
    ],
)
def test_complete_row_from_each_scientific_family_reaches_the_objective(family: str) -> None:
    model, _, _ = _build()
    row = next(row for row in _rows() if row.task_type == family)
    output = model(row, terminal_token_id=1)
    assert output.example_id == row.id and bool(torch.isfinite(output.losses.total))
    torch.autograd.backward(output.losses.total)
    assert all(parameter.grad is not None for parameter in model.probes.parameters())


def test_labels_cannot_leak_into_cognition_and_caller_rows_are_not_changed() -> None:
    model, _, _ = _build()
    row = _rows()[0]
    original = row.model_dump(mode="python")
    # Controlled label intervention, not a proposed dataset reference: keep prompt
    # and LM target fixed while updating the schema's fact-bound problem identity.
    facts = {**row.hidden_facts, "operation": "maximum"}
    changed = IrisExample.model_validate(
        {
            **original,
            "hidden_facts": facts,
            "problem_sha256": sha256(
                canonical_json_bytes({"task": row.task_type, "facts": facts})
            ).hexdigest(),
        }
    )
    first = model(row, terminal_token_id=1)
    second = model(changed, terminal_token_id=1)
    assert first.example_sha256 != second.example_sha256
    assert torch.equal(first.losses.language_model, second.losses.language_model)
    assert all(
        torch.equal(left.slots, right.slots)
        for left, right in zip(
            first.trace.teacher.workspace_trace,
            second.trace.teacher.workspace_trace,
            strict=True,
        )
    )
    assert not torch.equal(first.losses.probe, second.losses.probe)
    assert row.model_dump(mode="python") == original


@pytest.mark.parametrize(
    "mutation", ["buffer", "probe_buffer", "extra_parameter", "policy", "frozen_probe"]
)
def test_mutated_owner_fails_before_target_preparation(mutation: str) -> None:
    model, _, _ = _build()
    if mutation == "buffer":
        model.configuration_identity[0] ^= 1
    elif mutation == "probe_buffer":
        model.probes.configuration_identity[0] ^= 1
    elif mutation == "extra_parameter":
        model.register_parameter("foreign", torch.nn.Parameter(torch.zeros(1)))
    elif mutation == "policy":
        model._policy = model.policy.model_copy(update={"diversity_margin": 0.4})
    else:
        next(model.probes.parameters()).requires_grad_(False)
    with (
        patch.object(model.teacher.backbone, "tokenize_teacher_target") as target,
        pytest.raises(IrisScientificTeacherError),
    ):
        model(_rows()[0], terminal_token_id=1)
    target.assert_not_called()


def test_parameter_mutation_during_target_preparation_fails_before_model_execution() -> None:
    model, _, _ = _build()
    original = model.teacher.backbone.tokenize_teacher_target

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return result

    with (
        patch.object(model.teacher.backbone, "tokenize_teacher_target", side_effect=mutate),
        patch.object(model.teacher.backbone, "forward_prompt") as prompt,
        pytest.raises(IrisScientificTeacherError),
    ):
        model(_rows()[0], terminal_token_id=1)
    prompt.assert_not_called()
