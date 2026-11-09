"""Explicit probe labels, workspace-only heads and differentiable CE wiring."""

from copy import deepcopy
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan, SyntheticTemplate
from iris.training.losses import PROBE_FACT_FIELDS
from iris.training.probes import (
    IrisProbeError,
    IrisProbeVocabulary,
    WorkspaceProbeHeads,
    prepare_probe_targets,
    probe_vocabulary_sha256,
    synthetic_probe_vocabulary_v1,
    workspace_probe_objective,
)


def _rows(
    template: SyntheticTemplate = "iris-synthetic-01-v2", *, count: int = 15
) -> tuple[IrisExample, ...]:
    return tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=count), template_id=template
        )
    )


def _heads(**updates: Any) -> WorkspaceProbeHeads:
    arguments: dict[str, Any] = {
        "vocabulary": synthetic_probe_vocabulary_v1(),
        "state_dim": 4,
        "pooling": "mean_all_slots",
        "init_seed": 42,
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }
    return WorkspaceProbeHeads(**(arguments | updates))


def test_declared_vocabulary_has_source_domains_and_fixed_boolean_order() -> None:
    vocabulary = synthetic_probe_vocabulary_v1()
    assert tuple(len(getattr(vocabulary, name)) for name in PROBE_FACT_FIELDS) == (4, 12, 2, 4)
    assert vocabulary.conflicting_conditions == (False, True)
    assert len(probe_vocabulary_sha256(vocabulary)) == 64
    assert vocabulary == synthetic_probe_vocabulary_v1()
    assert IrisProbeVocabulary.model_validate_json(vocabulary.model_dump_json()) == vocabulary
    assert probe_vocabulary_sha256(vocabulary) == probe_vocabulary_sha256(
        synthetic_probe_vocabulary_v1()
    )
    changed = vocabulary.model_copy(update={"operation": tuple(reversed(vocabulary.operation))})
    assert probe_vocabulary_sha256(changed) != probe_vocabulary_sha256(vocabulary)


@pytest.mark.parametrize("template", ["iris-synthetic-01-v1", "iris-synthetic-01-v2"])
def test_generated_fact_labels_encode_and_decode_exactly_without_scanning_to_fit(
    template: SyntheticTemplate,
) -> None:
    rows, vocabulary = _rows(template, count=90), synthetic_probe_vocabulary_v1()
    prepared = prepare_probe_targets(
        rows, vocabulary=vocabulary, purpose="training", device=torch.device("cpu")
    )
    assert prepared.example_ids == tuple(row.id for row in rows)
    assert prepared.purpose == "training"
    assert prepared.vocabulary_sha256 == probe_vocabulary_sha256(vocabulary)
    assert tuple(prepared.labels) == PROBE_FACT_FIELDS
    for name, tensor in prepared.labels.items():
        assert tensor.dtype == torch.int64 and tensor.device.type == "cpu"
        assert tensor.shape == (90,) and not tensor.requires_grad
        values = getattr(vocabulary, name)
        assert [values[index] for index in tensor.tolist()] == [
            row.hidden_facts[name] for row in rows
        ]


def test_unknown_fact_is_rejected_instead_of_growing_the_vocabulary() -> None:
    rows = _rows()
    vocabulary = synthetic_probe_vocabulary_v1()
    reduced = vocabulary.model_copy(update={"operation": ("resolve_conflict",)})
    before = probe_vocabulary_sha256(reduced)
    with pytest.raises(IrisProbeError):
        prepare_probe_targets(
            rows, vocabulary=reduced, purpose="training", device=torch.device("cpu")
        )
    assert probe_vocabulary_sha256(reduced) == before


def test_training_rejects_a_late_heldout_row_while_analysis_is_explicitly_allowed() -> None:
    validation = next(
        generate_synthetic_split(
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = (*_rows()[:2], validation)
    with pytest.raises(IrisProbeError):
        prepare_probe_targets(
            rows,
            vocabulary=synthetic_probe_vocabulary_v1(),
            purpose="training",
            device=torch.device("cpu"),
        )
    prepared = prepare_probe_targets(
        rows,
        vocabulary=synthetic_probe_vocabulary_v1(),
        purpose="analysis",
        device=torch.device("cpu"),
    )
    assert prepared.purpose == "analysis" and prepared.example_ids[-1] == validation.id


@pytest.mark.parametrize(
    "updates",
    [
        {"active_constraint": ("duplicate", "duplicate")},
        {"operation": ()},
        {"operation": ("bad label",)},
        {"conflicting_conditions": (0, 1)},
        {"conflicting_conditions": (True, False)},
    ],
)
def test_vocabulary_revalidates_bypassed_frozen_models(updates: dict[str, Any]) -> None:
    bad = synthetic_probe_vocabulary_v1().model_copy(update=updates)
    with pytest.raises(IrisProbeError):
        _heads(vocabulary=bad)
    with pytest.raises(IrisProbeError):
        probe_vocabulary_sha256(bad)


def test_label_preparation_owns_facts_and_does_not_alias_other_calls() -> None:
    rows = _rows()[:2]
    arguments: dict[str, Any] = {
        "vocabulary": synthetic_probe_vocabulary_v1(),
        "purpose": "training",
        "device": torch.device("cpu"),
    }
    first = prepare_probe_targets(rows, **arguments)
    second = prepare_probe_targets(rows, **arguments)
    first.labels["operation"].fill_(999)
    assert not torch.equal(first.labels["operation"], second.labels["operation"])
    assert first.example_ids == second.example_ids


def test_local_initialization_is_deterministic_and_leaves_ambient_rng_unchanged() -> None:
    before = torch.random.get_rng_state().clone()
    first = _heads()
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    second = _heads()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    different = _heads(init_seed=1337)
    assert first.configuration_sha256 != different.configuration_sha256
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), different.parameters(), strict=True)
    )


def test_heads_use_all_slot_mean_and_parameter_count_does_not_depend_on_capacity() -> None:
    heads = _heads()
    expected_count = (4 + 12 + 2 + 4) * (4 + 1)
    assert sum(parameter.numel() for parameter in heads.parameters()) == expected_count
    slots = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 24
    original = slots.clone()
    logits = heads(slots)
    for name in PROBE_FACT_FIELDS:
        head = heads.heads[name]
        weight, bias = head.weight, head.bias
        assert isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)
        expected = slots.mean(dim=1) @ weight.T + bias
        assert torch.allclose(logits[name], expected, atol=1e-12, rtol=1e-12)
        assert torch.allclose(logits[name], heads(slots.flip(dims=(1,)))[name])
        assert torch.allclose(logits[name], heads(slots.repeat(1, 2, 1))[name])
    assert torch.equal(slots, original)
    # A canonical D=256 profile has 5654 probe parameters at every K.
    assert sum(parameter.numel() for parameter in _heads(state_dim=256).parameters()) == 5654
    with pytest.raises(TypeError):
        heads(slots, hidden_facts={})


def test_heads_and_workspace_receive_probe_gradients_and_owned_parameters_can_step() -> None:
    heads = _heads()
    rows = _rows()[:2]
    labels = prepare_probe_targets(
        rows, vocabulary=heads.vocabulary, purpose="training", device=torch.device("cpu")
    )
    slots = nn.Parameter(torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 24)
    optimizer = torch.optim.AdamW([slots, *heads.parameters()], lr=1e-3)
    before = tuple(parameter.detach().clone() for parameter in heads.parameters())
    loss = workspace_probe_objective(heads, slots, labels, head_reduction="mean")
    torch.autograd.backward(loss)
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())
    assert bool((slots.grad != 0).any())
    for parameter in heads.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    optimizer.step()
    assert any(
        not torch.equal(old, new) for old, new in zip(before, heads.parameters(), strict=True)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_heads_accumulate_outside_autocast(dtype: torch.dtype) -> None:
    heads = _heads(dtype=dtype)
    slots = torch.ones(2, 8, 4, dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = heads(slots)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(value.dtype == expected_dtype for value in logits.values())
    torch.autograd.backward(torch.stack([value.sum() for value in logits.values()]).sum())
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())


def test_state_roundtrip_and_nested_prefix_preserve_the_class_identity() -> None:
    source, restored = _heads(), _heads()
    state = deepcopy(source.state_dict())
    restored.load_state_dict(state)
    slots = torch.ones(1, 8, 4, dtype=torch.float64)
    for name in PROBE_FACT_FIELDS:
        assert torch.equal(source(slots)[name], restored(slots)[name])
    parent_source, parent_target = nn.Sequential(source), nn.Sequential(_heads())
    parent_target.load_state_dict(deepcopy(parent_source.state_dict()))
    assert parent_target[0].configuration_sha256 == source.configuration_sha256


@pytest.mark.parametrize("kind", ["class_order", "seed", "missing", "tampered"])
def test_incompatible_checkpoint_fails_before_copying_any_head_parameters(kind: str) -> None:
    target = _heads()
    state = deepcopy(target.state_dict())
    if kind == "class_order":
        vocabulary = target.vocabulary
        changed = vocabulary.model_copy(update={"operation": tuple(reversed(vocabulary.operation))})
        state = deepcopy(_heads(vocabulary=changed).state_dict())
    elif kind == "seed":
        state = deepcopy(_heads(init_seed=1337).state_dict())
    elif kind == "missing":
        del state["configuration_identity"]
    else:
        state["configuration_identity"][0] ^= 1
    state["heads.operation.bias"].fill_(987)
    before = deepcopy(target.state_dict())
    with pytest.raises(IrisProbeError):
        target.load_state_dict(state, strict=False)
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in before.items())


def test_mutated_identity_or_head_registration_is_rejected_before_forward() -> None:
    slots = torch.ones(1, 8, 4, dtype=torch.float64)
    heads = _heads()
    heads.configuration_identity[0] ^= 1
    with pytest.raises(IrisProbeError):
        heads(slots)
    heads = _heads()
    heads.heads["operation"] = nn.Identity()
    with pytest.raises(IrisProbeError):
        heads(slots)


@pytest.mark.parametrize(
    "updates",
    [
        {"state_dim": True},
        {"state_dim": 0},
        {"state_dim": 4097},
        {"init_seed": True},
        {"init_seed": -1},
        {"init_seed": 2**63},
        {"pooling": "implicit"},
        {"dtype": torch.int64},
        {"device": "cpu"},
    ],
)
def test_invalid_constructor_contract_is_rejected(updates: dict[str, Any]) -> None:
    with pytest.raises(IrisProbeError):
        _heads(**updates)


def test_vocabulary_property_is_owned_and_immutable() -> None:
    heads = _heads()
    first, second = heads.vocabulary, heads.vocabulary
    assert first is not second and first == second
    with pytest.raises(ValidationError):
        first.operation = ("changed",)
    assert heads.vocabulary == synthetic_probe_vocabulary_v1()


@pytest.mark.parametrize(
    "slots",
    [
        torch.ones(1, 4, dtype=torch.float64),
        torch.ones(1, 8, 5, dtype=torch.float64),
        torch.ones(1, 8, 4, dtype=torch.float32),
        torch.empty(1, 0, 4, dtype=torch.float64),
        torch.full((1, 8, 4), float("nan"), dtype=torch.float64),
    ],
)
def test_invalid_workspace_boundary_is_rejected(slots: torch.Tensor) -> None:
    with pytest.raises(IrisProbeError):
        _heads()(slots)


def test_objective_rejects_same_shape_reordered_labels_and_detached_training() -> None:
    heads, rows = _heads(), _rows()[:2]
    slots = torch.ones(2, 8, 4, dtype=torch.float64, requires_grad=True)
    vocabulary = heads.vocabulary
    reordered = vocabulary.model_copy(update={"operation": tuple(reversed(vocabulary.operation))})
    wrong = prepare_probe_targets(
        rows, vocabulary=reordered, purpose="training", device=torch.device("cpu")
    )
    with pytest.raises(IrisProbeError):
        workspace_probe_objective(heads, slots, wrong, head_reduction="mean")
    labels = prepare_probe_targets(
        rows, vocabulary=vocabulary, purpose="training", device=torch.device("cpu")
    )
    with torch.no_grad(), pytest.raises(IrisProbeError):
        workspace_probe_objective(heads, slots, labels, head_reduction="mean")
    heads.eval()
    with pytest.raises(IrisProbeError):
        workspace_probe_objective(heads, slots, labels, head_reduction="mean")
    heads.train()
    heads.requires_grad_(False)
    with pytest.raises(IrisProbeError):
        workspace_probe_objective(heads, slots, labels, head_reduction="mean")


def test_analysis_keeps_modes_parameters_and_existing_gradients_unchanged() -> None:
    heads, rows = _heads(), _rows()[:2]
    labels = prepare_probe_targets(
        rows, vocabulary=heads.vocabulary, purpose="analysis", device=torch.device("cpu")
    )
    slots = torch.ones(2, 8, 4, dtype=torch.float64, requires_grad=True)
    with pytest.raises(IrisProbeError):
        workspace_probe_objective(heads, slots, labels, head_reduction="mean")
    heads.eval()
    for parameter in heads.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = deepcopy(heads.state_dict())
    loss = workspace_probe_objective(heads, slots, labels, head_reduction="mean")
    assert not loss.requires_grad and not heads.training
    assert slots.grad is None
    for parameter in heads.parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert all(torch.equal(value, heads.state_dict()[name]) for name, value in before.items())


def test_analysis_target_allocation_does_not_create_inference_tensors() -> None:
    rows = _rows()[:2]
    with torch.inference_mode():
        labels = prepare_probe_targets(
            rows,
            vocabulary=synthetic_probe_vocabulary_v1(),
            purpose="analysis",
            device=torch.device("cpu"),
        )
    assert all(not value.is_inference() for value in labels.labels.values())
    with torch.inference_mode(), pytest.raises(IrisProbeError):
        _heads()
