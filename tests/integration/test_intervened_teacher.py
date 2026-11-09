"""Actual causal NLL and inference/teacher phase parity on offline random Qwen."""

import random
from dataclasses import fields
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from iris.application import build_request_orchestrator
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import (
    IrisScientificIntervenedTeacherOutputV1,
    IrisScientificTeacherError,
    IrisScientificTeacherV1,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.generation import TemperatureTopPSamplingConfig
from iris.interventions import intervention_plan_sha256_v1
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.evaluation import (
    IrisEvaluationAggregationError,
    IrisTokenWeightedEvaluationAccumulator,
)
from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.training.model import IrisTeacherForcingError
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_intervened_request import plan
from tests.integration.test_scientific_teacher import _build, _rows
from tests.unit.test_workspace_intervention_state import assert_same, clone

pytestmark = pytest.mark.integration

OPERATIONS: tuple[dict[str, Any], ...] = (
    {"kind": "zero_workspace"},
    {"kind": "zero_slot", "slot": 0},
    {"kind": "freeze_slot", "slot": 0},
    {"kind": "swap_slots", "slot": 0, "other_slot": 1},
    {"kind": "shuffle_slots", "seed": 42},
    {"kind": "gaussian_noise", "sigma": 0.25, "seed": 42},
    {
        "kind": "replace_slot",
        "slot": 0,
        "values": (1.0, -1.0, 0.5, 0.25) * 2,
        "source_module": 2,
        "confidence": 0.75,
        "ttl": 4.0,
    },
    {"kind": "block_module_write", "module": 0},
    {"kind": "block_module_read", "module": 1},
    {"kind": "force_module_winner", "module": 2},
    {"kind": "scale_module_bid", "module": 0, "factor": 10.0},
    {"kind": "reduce_workspace_capacity", "capacity": 1},
)


def prepared(model: IrisScientificTeacherV1, text: str | None = None) -> IrisTeacherForcingInput:
    row = _rows()[0]
    return prepare_teacher_forcing_input(
        backbone=model.teacher.backbone,
        prompt=row.prompt,
        target_text=row.expected_response if text is None else text,
        terminal_token_id=1,
    )


def test_empty_plan_matches_baseline_without_changing_weights_gradients_modes_or_rng() -> None:
    model, _, lm = _build()
    example = prepared(model)
    configuration = model.configuration_sha256
    provenance = model.teacher.checkpoint_provenance
    torch.autograd.backward(model.training_forward(_rows()[0], terminal_token_id=1).losses.total)
    gradients = {
        name: None if value.grad is None else value.grad.clone()
        for name, value in model.named_parameters()
    }
    model.eval()
    before = module_state_sha256((("model", model), ("qwen", lm)))
    runtime = capture_runtime_state(torch.device("cpu"))
    baseline = model.teacher.evaluation_forward(example)
    output = model.teacher.evaluation_forward_intervened(example, plan=plan())
    assert type(output) is IrisIntervenedTeacherOutputV1
    assert not isinstance(output, IrisTeacherForcedOutput)
    assert torch.equal(output.loss, baseline.loss)
    assert output.supervised_token_count == baseline.supervised_token_count
    assert len(output.workspace_trace) == 3
    for left, right in zip(output.workspace_trace, baseline.workspace_trace, strict=True):
        assert_same(left, right)
        for field in fields(left):
            tensor = getattr(left, field.name)
            assert not tensor.requires_grad and not tensor.is_inference()
    assert not output.loss.requires_grad and not output.loss.is_inference()
    assert output.sidecar_state_sha256 == provenance.sidecar_state_sha256
    assert output.target_sha256 == teacher_target_sha256_v1(
        example.target_input_ids, example.target_attention_mask
    )
    assert output.prompt_sha256 == domain_sha256(
        "iris/intervened-teacher-prompt/v1", example.prompt.encode()
    )
    assert output.teacher_graph_sha256 == domain_sha256(
        "iris/intervened-teacher-graph/v1",
        canonical_json_bytes(
            {
                "backbone": provenance.backbone_configuration_json,
                "cognitive": provenance.cognitive_configuration_json,
                "workspace": provenance.workspace_configuration_json,
            }
        ),
    )
    assert before == module_state_sha256((("model", model), ("qwen", lm)))
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    assert all(not module.training for module in model.modules())
    for name, parameter in model.named_parameters():
        expected = gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.equal(parameter.grad, expected)
    with pytest.raises(IrisEvaluationAggregationError, match="exactly"):
        IrisTokenWeightedEvaluationAccumulator().update(cast(IrisTeacherForcedOutput, output))
    model.train()
    assert model.teacher.checkpoint_provenance == provenance
    assert model.configuration_sha256 == configuration
    model.zero_grad(set_to_none=True)
    torch.autograd.backward(model.training_forward(_rows()[0], terminal_token_id=1).losses.total)
    assert any(p.grad is not None and bool(p.grad.any()) for p in model.parameters())
    assert all(p.grad is None and not p.requires_grad for p in lm.parameters())


@pytest.mark.parametrize("specification", OPERATIONS, ids=lambda spec: spec["kind"])
def test_all_twelve_operators_match_actual_inference_workspace_and_latent_prefix(
    specification: dict[str, Any],
) -> None:
    model, _, _ = _build()
    teacher = model.teacher
    provenance = teacher.checkpoint_provenance
    with torch.random.fork_rng(devices=[]):
        request = build_request_orchestrator(
            backbone=teacher.backbone,
            cognitive_config=CognitiveConfig.model_validate_json(
                provenance.cognitive_configuration_json
            ),
            workspace_config=WorkspaceConfig.model_validate_json(
                provenance.workspace_configuration_json
            ),
            # Stop after the first genuine generated token; no mock decoder.
            stop_token_ids=tuple(range(teacher.backbone.vocabulary_size)),
            sampling=TemperatureTopPSamplingConfig(),
        )
    request.restore_teacher_sidecar_state_v1(dict(teacher.state_dict()))
    model.eval()
    example = prepared(model)
    selected = plan(specification)
    embeddings: list[torch.Tensor] = []
    original = teacher.backbone.forward_embeddings

    def record(*args: Any, **kwargs: Any) -> Any:
        embeddings.append(args[0].detach().clone())
        return original(*args, **kwargs)

    before = capture_runtime_state(torch.device("cpu"))
    with patch.object(teacher.backbone, "forward_embeddings", side_effect=record):
        inferred = request.execute_intervened(example.prompt, plan=selected, strategy="greedy")
        output = teacher.evaluation_forward_intervened(example, plan=selected)
    assert len(embeddings) == 2
    assert torch.equal(embeddings[0], embeddings[1][:, : embeddings[0].shape[1]])
    for actual, expected in zip(
        output.workspace_trace, inferred.generation.workspace_trace, strict=True
    ):
        assert_same(actual, expected)
    assert output.intervention_plan_sha256 == inferred.receipt.intervention_plan_sha256
    assert output.intervention_plan_sha256 == intervention_plan_sha256_v1(selected)
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("specification", [OPERATIONS[0], OPERATIONS[5], OPERATIONS[6]])
def test_changed_nll_is_recomputed_from_real_causal_logits_and_exact_target_labels(
    specification: dict[str, Any],
) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    baseline = model.teacher.evaluation_forward(example)
    records: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = model.teacher.backbone.forward_embeddings

    def record(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        records.append((result.logits.detach().clone(), kwargs["labels"].clone()))
        return result

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=record):
        output = model.teacher.evaluation_forward_intervened(example, plan=plan(specification))
    assert len(records) == 1
    logits, labels = records[0]
    shifted_labels = labels[:, 1:].reshape(-1)
    expected_sum = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        shifted_labels,
        ignore_index=-100,
        reduction="sum",
    )
    count = int((shifted_labels != -100).sum())
    assert count == output.supervised_token_count == int(example.target_attention_mask.sum())
    assert torch.equal(
        labels[labels != -100], example.target_input_ids[example.target_attention_mask.bool()]
    )
    torch.testing.assert_close(output.loss * count, expected_sum)
    assert not torch.equal(output.loss, baseline.loss)


def test_targets_are_excluded_from_workspace_and_changed_target_gets_a_distinct_identity() -> None:
    model, _, _ = _build()
    model.eval()
    selected = plan(OPERATIONS[5])
    first = model.teacher.evaluation_forward_intervened(prepared(model, "0"), plan=selected)
    other = model.teacher.evaluation_forward_intervened(prepared(model, "1 2"), plan=selected)
    for left, right in zip(first.workspace_trace, other.workspace_trace, strict=True):
        assert_same(left, right)
    assert first.prompt_sha256 == other.prompt_sha256
    assert first.target_sha256 != other.target_sha256
    assert first.supervised_token_count != other.supervised_token_count
    assert not torch.equal(first.loss, other.loss)


def test_freeze_owns_new_state_on_each_call_and_schedule_starts_at_selected_cycle() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    selected = plan({"kind": "freeze_slot", "slot": 0, "cycles": (2, 3)})
    first = model.teacher.evaluation_forward_intervened(example, plan=selected)
    originals = tuple(clone(state) for state in first.workspace_trace)
    for field in fields(first.workspace_trace[0]):
        assert torch.equal(
            getattr(originals[1], field.name)[:, 0], getattr(originals[2], field.name)[:, 0]
        )
    first.workspace_trace[1].slots.fill_(19)
    again = model.teacher.evaluation_forward_intervened(example, plan=selected)
    for actual, expected in zip(again.workspace_trace, originals, strict=True):
        assert_same(actual, expected)


def test_phase_constraints_reach_actual_modules_and_final_state() -> None:
    model, _, _ = _build()
    model.eval()
    teacher = model.teacher
    selected = plan(
        {"kind": "block_module_read", "module": 1},
        {"kind": "reduce_workspace_capacity", "capacity": 1},
        {"kind": "force_module_winner", "module": 2},
        {"kind": "block_module_write", "module": 2},
        {**OPERATIONS[6], "slot": 3},
    )
    with (
        patch.object(
            teacher.cognitive_modules, "forward", wraps=teacher.cognitive_modules.forward
        ) as reads,
        patch.object(
            teacher.slot_allocator, "forward", wraps=teacher.slot_allocator.forward
        ) as allocations,
    ):
        output = teacher.evaluation_forward_intervened(prepared(model), plan=selected)
    assert reads.call_count == allocations.call_count == 3
    for call in reads.call_args_list:
        assert not bool(call.args[1][:, 1].any())
    for call in allocations.call_args_list:
        assert bool(call.args[2].winner_mask[:, 2].all())
    for state in output.workspace_trace:
        assert not bool(state.occupancy[:, 1:].any())
        assert not bool((state.source_module == 2).any())


def test_blocking_every_writer_is_a_canonical_empty_workspace() -> None:
    model, _, _ = _build()
    model.eval()
    selected = plan(*({"kind": "block_module_write", "module": n} for n in range(4)))
    output = model.teacher.evaluation_forward_intervened(prepared(model), plan=selected)
    for state in output.workspace_trace:
        assert not bool(state.occupancy.any()) and not bool(state.slots.any())
        assert not bool(state.write_count.any()) and bool((state.source_module == -1).all())


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "zero_slot", "slot": 4},
        {"kind": "block_module_read", "module": 4},
        {"kind": "zero_workspace", "cycles": (4,)},
        {"kind": "reduce_workspace_capacity", "capacity": 5},
        {**OPERATIONS[6], "values": (1.0,)},
    ],
)
def test_incompatible_plan_is_rejected_before_any_backbone_call(bad: dict[str, Any]) -> None:
    model, _, lm = _build()
    model.eval()
    example = prepared(model)
    with (
        patch.object(lm, "forward", wraps=lm.forward) as calls,
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_intervened(example, plan=plan(bad))
    assert calls.call_count == 0
    assert model.teacher.evaluation_forward_intervened(example, plan=plan()).loss.isfinite()


@pytest.mark.parametrize("boundary", ["training", "inference", "autocast"])
def test_invalid_mode_is_rejected_without_model_execution(boundary: str) -> None:
    model, _, lm = _build()
    example = prepared(model)
    model.train(boundary == "training")
    with (
        torch.inference_mode(boundary == "inference"),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=boundary == "autocast"),
        patch.object(lm, "forward", wraps=lm.forward) as calls,
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_intervened(example, plan=plan())
    assert calls.call_count == 0


@pytest.mark.parametrize("failure", [False, True])
def test_success_and_failure_restore_caller_runtime(failure: bool) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    before = capture_runtime_state(torch.device("cpu"))
    original = model.teacher.backbone.forward_embeddings

    def disturb(*args: Any, **kwargs: Any) -> Any:
        random.random()
        torch.rand(3)
        torch.backends.cudnn.benchmark = not before.cudnn_benchmark
        if failure:
            raise RuntimeError("injected private failure")
        return original(*args, **kwargs)

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=disturb):
        if failure:
            with pytest.raises(IrisTeacherForcingError) as error:
                model.teacher.evaluation_forward_intervened(example, plan=plan(OPERATIONS[5]))
            assert "private" not in str(error.value)
        else:
            model.teacher.evaluation_forward_intervened(example, plan=plan(OPERATIONS[5]))
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("mutation", ["weight", "target", "prompt"])
def test_post_causal_mutation_cannot_return_a_mislabeled_measurement(mutation: str) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    before = capture_runtime_state(torch.device("cpu"))
    original = model.teacher.backbone.forward_embeddings

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if mutation == "weight":
            next(model.teacher.parameters()).add_(0.01)
        elif mutation == "target":
            example.target_input_ids[0, 0] = 0
        else:
            object.__setattr__(example, "prompt", "unrelated prompt")
        return result

    with (
        patch.object(model.teacher.backbone, "forward_embeddings", side_effect=mutate),
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_intervened(example, plan=plan())
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    # Test-injected corruption is deliberately not described as rolled back.


def test_scientific_wrapper_binds_owned_row_and_actual_nll_without_probe_surrogate() -> None:
    model, _, lm = _build()
    model.eval()
    row = _rows()[0]
    before = module_state_sha256((("scientific", model), ("qwen", lm)))
    baseline = model.evaluation_forward(row, terminal_token_id=1)
    with (
        patch.object(model.probes, "forward", side_effect=AssertionError("No probe surrogate")),
        patch.object(lm, "forward", wraps=lm.forward) as calls,
    ):
        output = model.evaluation_forward_intervened(row, terminal_token_id=1, plan=plan())
    assert type(output) is IrisScientificIntervenedTeacherOutputV1
    assert calls.call_count == 2
    assert output.example_id == baseline.example_id
    assert output.example_sha256 == baseline.example_sha256
    assert output.configuration_sha256 == model.configuration_sha256
    assert torch.equal(output.target_nll_sum, baseline.losses.language_model)
    assert output.supervised_token_count == baseline.supervised_token_count
    assert before == module_state_sha256((("scientific", model), ("qwen", lm)))


@pytest.mark.parametrize("stage", ["preparation", "causal"])
def test_scientific_probe_weight_mutation_aborts_and_restores_rng(stage: str) -> None:
    model, _, _ = _build()
    model.eval()
    before = capture_runtime_state(torch.device("cpu"))
    original = (
        prepare_teacher_forcing_input
        if stage == "preparation"
        else model.teacher.backbone.forward_embeddings
    )

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        next(model.probes.parameters()).add_(0.125)
        torch.rand(2)
        return result

    target = "iris.application.scientific_teacher.prepare_teacher_forcing_input"
    context = (
        patch(target, side_effect=mutate)
        if stage == "preparation"
        else patch.object(model.teacher.backbone, "forward_embeddings", side_effect=mutate)
    )
    with context, pytest.raises(IrisScientificTeacherError):
        model.evaluation_forward_intervened(_rows()[0], terminal_token_id=1, plan=plan())
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
