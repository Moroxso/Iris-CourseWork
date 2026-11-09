"""Hand-computed section-19/21 metrics, strict metadata and bounded owned vectors."""

from dataclasses import replace
from math import hypot, isclose, log
from typing import Any, Literal

import pytest
import torch
from pydantic import ValidationError

from iris.cognitive.outputs import ModuleProposal
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import (
    WorkspaceTraceOptionsV1,
    WorkspaceTraceRecorderV1,
    WorkspaceTraceV1,
    summarize_workspace_traces_v1,
)


def options(
    mode: Literal["none", "metadata", "full"] = "metadata", **changes: Any
) -> WorkspaceTraceOptionsV1:
    return WorkspaceTraceOptionsV1.model_validate(
        {
            "mode": mode,
            "example_id": "test-1",
            "checkpoint_id": "checkpoint-1",
            "seed": 42,
            **changes,
        }
    )


def state() -> WorkspaceState:
    return WorkspaceState.empty(
        batch_size=1, num_slots=2, state_dim=2, dtype=torch.float64, device=torch.device("cpu")
    )


def record(
    recorder: WorkspaceTraceRecorderV1,
    cycle: int,
    before: WorkspaceState,
    after: WorkspaceState,
    module: int,
    slot: int,
    attention: torch.Tensor,
    module_states: torch.Tensor | None = None,
) -> None:
    mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    mask[0, module, slot] = True
    probability = torch.full((1, 2), 0.5, dtype=torch.float64)
    scores = torch.zeros(1, 2, 2, dtype=torch.float64)
    recorder.record(
        cycle=cycle,
        read_state=before,
        before_write=before,
        workspace=after,
        reads=WorkspaceReaderOutput(torch.zeros(1, 2, 2), attention),
        module_states=(
            torch.eye(2).unsqueeze(0).requires_grad_() if module_states is None else module_states
        ),
        proposal=ModuleProposal(
            content=scores, key=scores, bid=probability, confidence=probability, ttl=probability
        ),
        arbitration=WorkspaceArbiterOutput(
            scores=probability,
            novelty=probability,
            relevance=probability,
            soft_probabilities=probability,
            selection_weights=probability,
            winner_mask=mask.any(-1),
            winner_indices=torch.tensor([[module]]),
        ),
        allocation=SlotAllocatorOutput(
            allocation_scores=scores,
            retention_scores=probability,
            target_mask=mask.any(1),
            assignment_mask=mask,
            assignment_probabilities=scores,
            assignment_weights=scores,
            module_indices=torch.tensor([[module]]),
            slot_indices=torch.tensor([[slot]]),
        ),
    )


def trace(mode: Literal["metadata", "full"] = "metadata") -> WorkspaceTraceV1:
    recorder = WorkspaceTraceRecorderV1(
        options(mode), num_modules=2, num_slots=2, state_dim=2, cycles=2
    )
    empty = state()
    first = replace(
        empty,
        slots=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        occupancy=torch.tensor([[True, False]]),
        source_module=torch.tensor([[0, -1]]),
        ttl=torch.tensor([[2.0, 0.0]]),
        write_count=torch.tensor([[1, 0]]),
    )
    record(recorder, 1, empty, first, 0, 0, torch.zeros(1, 2, 2))
    second = replace(
        first,
        slots=torch.eye(2).unsqueeze(0),
        occupancy=torch.ones(1, 2, dtype=torch.bool),
        source_module=torch.tensor([[0, 1]]),
        ttl=torch.tensor([[1.0, 3.0]]),
        write_count=torch.tensor([[1, 1]]),
        age=torch.tensor([[1, 0]]),
    )
    record(recorder, 2, first, second, 1, 1, torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]))
    result = recorder.finish()
    assert result is not None
    with pytest.raises(ValueError, match="completed"):
        recorder.finish()
    return result


def test_hand_calculated_count_weighted_usage_reading_and_cosine() -> None:
    result = summarize_workspace_traces_v1((trace(),))
    assert result.module_win_counts == result.slot_allocation_counts == (1, 1)
    assert result.module_win_rates == (0.5, 0.5)
    assert result.module_entropy_nats == result.slot_entropy_nats == pytest.approx(log(2))
    assert result.effective_active_modules == result.effective_active_slots == pytest.approx(2)
    assert result.used_slots == 2 and result.allocated_writes == 2
    assert result.overwrites == 0 and result.overwrite_rate == 0
    assert result.occupied_slot_observations == 3
    assert result.mean_ttl == 2 and result.mean_write_count == 1
    assert result.cosine_pairs == 1 and result.mean_slot_cosine_similarity == 0
    assert result.read_opportunities == 4 and result.active_reads == 2
    assert result.cross_module_reads == 1 and result.cross_module_read_rate == 0.5
    assert (
        result.cross_module_attention_mass == 1 and result.mean_cross_module_attention_mass == 0.5
    )
    assert (
        result.causal_influence == "not_measured" and result.scientific_acceptance == "not_assessed"
    )


@pytest.mark.parametrize("mode", ["metadata", "full"])
def test_json_roundtrip_and_plain_immutable_owned_values(mode: Any) -> None:
    result = trace(mode)
    encoded = result.model_dump_json()
    assert WorkspaceTraceV1.model_validate_json(encoded) == result
    assert (result.cycles[0].full is not None) == (mode == "full")
    assert "prompt" not in encoded and "target" not in encoded
    with pytest.raises(ValidationError):
        result.num_slots = 8


def test_none_does_not_record_and_budget_rejects_before_collection() -> None:
    recorder = WorkspaceTraceRecorderV1(
        options("none", maximum_scalars=1), num_modules=8, num_slots=8, state_dim=256, cycles=3
    )
    assert not recorder.enabled and recorder.finish() is None
    with pytest.raises(ValueError, match="before model"):
        WorkspaceTraceRecorderV1(
            options("full", maximum_scalars=100),
            num_modules=8,
            num_slots=8,
            state_dim=256,
            cycles=3,
        )
    with pytest.raises(TypeError):
        WorkspaceTraceRecorderV1(
            {"mode": "metadata"},  # type: ignore[arg-type]
            num_modules=2,
            num_slots=2,
            state_dim=2,
            cycles=1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycle", 3),
        ("module_bids", (1.0,)),
        ("module_confidence", (float("nan"), 1.0)),
        ("selected_modules", (0, 0)),
        ("selected_slots", (2,)),
        ("write_modules", (1,)),
        ("slot_sources", (2, -1)),
        ("slot_write_counts", (1, 2)),
        ("read_attention", ((1.0, 0.0), (1.0, 0.0))),
        ("slot_ttl", (-1.0, 0.0)),
        ("workspace_cosine_matrix", ((1.0, 1.0), (0.0, 0.0))),
        ("slot_occupancy", (1, False)),
        ("full", {"workspace_slots": ()}),
    ],
)
def test_malformed_metadata_is_rejected(field: str, value: Any) -> None:
    data = trace().model_dump()
    data["cycles"][0][field] = value
    with pytest.raises(ValidationError):
        WorkspaceTraceV1.model_validate(data)


def test_pool_raw_counts_and_reject_duplicates_mixed_context_and_forged_models() -> None:
    first = trace()
    data = first.model_dump()
    data["options"]["example_id"] = "test-2"
    second = WorkspaceTraceV1.model_validate(data)
    result = summarize_workspace_traces_v1((first, second))
    assert result.module_win_counts == (2, 2) and result.effective_active_modules == pytest.approx(
        2
    )
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_workspace_traces_v1((first, first))
    data["options"]["seed"] = 1337
    with pytest.raises(ValueError, match="different"):
        summarize_workspace_traces_v1((first, WorkspaceTraceV1.model_validate(data)))
    with pytest.raises(ValueError):
        summarize_workspace_traces_v1((first.model_copy(update={"num_modules": 0}),))


def test_trace_full_budget_and_mode_revalidated_from_untrusted_json() -> None:
    data = trace("full").model_dump()
    data["options"]["maximum_scalars"] = 1
    with pytest.raises(ValueError, match="budget"):
        WorkspaceTraceV1.model_validate(data)
    data["options"]["maximum_scalars"] = 65536
    data["options"]["mode"] = "metadata"
    with pytest.raises(ValueError, match="Full vectors"):
        WorkspaceTraceV1.model_validate(data)


@pytest.mark.parametrize("value", [0.0, 1e-310, 6e307])
def test_finite_extreme_norms_cosines_and_pooled_means(value: float) -> None:
    empty = state()
    vectors = torch.tensor([[[value, value], [-value, value]]], dtype=torch.float64)
    populated = replace(
        empty,
        slots=vectors,
        occupancy=torch.ones(1, 2, dtype=torch.bool),
        source_module=torch.tensor([[0, 1]]),
        ttl=torch.full((1, 2), value, dtype=torch.float64),
        write_count=torch.ones(1, 2, dtype=torch.int64),
    )
    recorder = WorkspaceTraceRecorderV1(
        options("full"), num_modules=2, num_slots=2, state_dim=2, cycles=1
    )
    record(recorder, 1, empty, populated, 0, 0, torch.zeros(1, 2, 2), module_states=vectors)
    first = recorder.finish()
    assert first is not None
    expected = hypot(value, value)
    for actual in (*first.cycles[0].slot_norms, *first.cycles[0].module_state_norms):
        # Default pytest.approx absolute tolerance would conceal subnormal errors.
        assert isclose(actual, expected, rel_tol=1e-12, abs_tol=0)
    for i, row in enumerate(first.cycles[0].workspace_cosine_matrix):
        for j, actual in enumerate(row):
            assert actual == pytest.approx(float(value != 0 and i == j), abs=1e-15)
    data = first.model_dump()
    data["options"]["example_id"] = "test-2"
    second = WorkspaceTraceV1.model_validate(data)
    pooled = summarize_workspace_traces_v1((first, second))
    assert isclose(pooled.mean_workspace_norm, hypot(expected, expected), rel_tol=1e-12, abs_tol=0)
    assert pooled.mean_ttl is not None
    assert isclose(pooled.mean_ttl, value, rel_tol=1e-12, abs_tol=0)
    assert WorkspaceTraceV1.model_validate_json(first.model_dump_json()) == first
