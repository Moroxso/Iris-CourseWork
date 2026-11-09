"""Actual state-phase effects, canonical records, ownership and paired randomness."""

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from iris.interventions import (
    FreezeSlot,
    GaussianNoise,
    ReduceWorkspaceCapacity,
    ReplaceSlot,
    ShuffleSlots,
    SwapSlots,
    WorkspaceIntervention,
    WorkspaceInterventionError,
    WorkspaceInterventionSpecV1,
    ZeroSlot,
    ZeroWorkspace,
    validate_intervention_state,
)
from iris.utils.hashing import tensor_sha256
from iris.workspace import LatentBridge, WorkspaceDecay, WorkspaceReader, WorkspaceState

KINDS = ("zero", "slot", "freeze", "swap", "shuffle", "noise", "replace", "capacity")
DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def state_fixture(dtype: torch.dtype = torch.float32) -> WorkspaceState:
    occupied = torch.tensor([[True, False, True, True], [True, True, False, True]])
    counters = torch.arange(1, 9).reshape(2, 4)
    return WorkspaceState(
        slots=torch.arange(1, 25)
        .reshape(2, 4, 3)
        .to(dtype)
        .masked_fill(~occupied.unsqueeze(-1), 0),
        age=counters.masked_fill(~occupied, 0),
        occupancy=occupied,
        source_module=torch.arange(4).expand(2, -1).clone().masked_fill(~occupied, -1),
        confidence=(counters.to(dtype) / 10).masked_fill(~occupied, 0),
        ttl=(counters.to(dtype) + 0.5).masked_fill(~occupied, 0),
        write_count=(counters + 3).masked_fill(~occupied, 0),
    )


def operation(kind: str, cycles: tuple[int, ...] | None = None) -> WorkspaceIntervention:
    if kind == "zero":
        return ZeroWorkspace(cycles=cycles)
    if kind == "slot":
        return ZeroSlot(1, cycles=cycles)
    if kind == "freeze":
        return FreezeSlot(1, cycles=cycles)
    if kind == "swap":
        return SwapSlots(0, 1, cycles=cycles)
    if kind == "shuffle":
        return ShuffleSlots(42, cycles=cycles)
    if kind == "noise":
        return GaussianNoise(0.25, 42, cycles=cycles)
    if kind == "replace":
        return ReplaceSlot(
            1, (9.0, 8.0, 7.0), source_module=3, confidence=0.75, ttl=2.5, cycles=cycles
        )
    if kind == "capacity":
        return ReduceWorkspaceCapacity(2, cycles=cycles)
    raise AssertionError(kind)


def clone(state: WorkspaceState) -> WorkspaceState:
    return WorkspaceState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def assert_same(left: WorkspaceState, right: WorkspaceState) -> None:
    for field in fields(left):
        actual, expected = getattr(left, field.name), getattr(right, field.name)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual, expected), field.name


def assert_slot(left: WorkspaceState, index: int, right: WorkspaceState, other: int) -> None:
    for field in fields(left):
        assert torch.equal(
            getattr(left, field.name)[:, index], getattr(right, field.name)[:, other]
        ), field.name


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", KINDS)
def test_each_state_operation_is_typed_canonical_owned_and_functional(
    dtype: torch.dtype, kind: str
) -> None:
    state = state_fixture(dtype)
    before = clone(state)
    op = operation(kind)
    assert isinstance(op, WorkspaceIntervention)
    output = op.apply(state, 1)
    validate_intervention_state(output)
    assert output.slots.shape == state.slots.shape
    assert_same(state, before)
    for field in fields(state):
        result = getattr(output, field.name)
        assert result.dtype == getattr(state, field.name).dtype
        assert (
            result.untyped_storage().data_ptr()
            != getattr(state, field.name).untyped_storage().data_ptr()
        )
        result.zero_()
    assert_same(state, before)


@pytest.mark.parametrize("kind", KINDS)
def test_inactive_cycle_returns_an_independent_exact_copy(kind: str) -> None:
    state = state_fixture()
    output = operation(kind, cycles=(2, 4)).apply(state, 1)
    assert_same(output, state)
    output.slots.zero_()
    assert bool((state.slots != 0).any())


def test_zeroes_and_capacity_clear_every_field_and_only_selected_records() -> None:
    state = state_fixture()
    empty = WorkspaceState.empty(
        batch_size=2, num_slots=4, state_dim=3, dtype=state.slots.dtype, device=state.slots.device
    )
    assert_same(ZeroWorkspace().apply(state, 1), empty)
    output = ZeroSlot(1).apply(state, 1)
    for slot in range(4):
        assert_slot(output, slot, empty if slot == 1 else state, slot)
    output = ReduceWorkspaceCapacity(2).apply(state, 1)
    for slot in range(4):
        assert_slot(output, slot, state if slot < 2 else empty, slot)
    assert_same(ReduceWorkspaceCapacity(4).apply(state, 1), state)


def test_swap_moves_complete_records_including_empty_records_and_is_an_involution() -> None:
    state = state_fixture()
    op = SwapSlots(0, 1)
    output = op.apply(state, 1)
    assert_slot(output, 0, state, 1)
    assert_slot(output, 1, state, 0)
    assert_slot(output, 2, state, 2)
    assert_slot(output, 3, state, 3)
    assert_same(op.apply(output, 2), state)


def test_shuffle_is_a_batched_bijection_of_entire_records_not_vectors_only() -> None:
    state = state_fixture()
    output = ShuffleSlots(42).apply(state, 1)
    for row in range(2):
        before = [
            tuple(
                getattr(state, field.name)[row, slot].flatten().tolist()[0:]
                for field in fields(state)
            )
            for slot in range(4)
        ]
        after = [
            tuple(
                getattr(output, field.name)[row, slot].flatten().tolist()[0:]
                for field in fields(state)
            )
            for slot in range(4)
        ]
        assert sorted(map(str, before)) == sorted(map(str, after))
    assert not torch.equal(output.slots, state.slots)


@pytest.mark.parametrize("kind", ["noise", "shuffle"])
def test_random_operations_are_repeatable_cycle_keyed_and_do_not_consume_global_rng(
    kind: str,
) -> None:
    state = state_fixture(torch.float64)
    before = torch.random.get_rng_state().clone()
    first = operation(kind).apply(state, 1)
    assert_same(operation(kind).apply(state, 1), first)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not torch.equal(operation(kind).apply(state, 2).slots, first.slots)
    if kind == "noise":
        different_seed = GaussianNoise(0.25, 43).apply(state, 1)
    else:
        different_seed = ShuffleSlots(43).apply(state, 1)
    assert not torch.equal(different_seed.slots, first.slots)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_noise_changes_only_occupied_values_and_zero_sigma_is_byte_exact(
    dtype: torch.dtype,
) -> None:
    state = state_fixture(dtype)
    output = GaussianNoise(0.25, 42).apply(state, 1)
    for field in fields(state):
        if field.name != "slots":
            assert torch.equal(getattr(output, field.name), getattr(state, field.name))
    assert bool((output.slots.masked_select(~state.occupancy.unsqueeze(-1)) == 0).all())
    assert not torch.equal(output.slots, state.slots)
    state.slots[0, 0, 0] = -0.0
    assert tensor_sha256(GaussianNoise(0.0, 42).apply(state, 1).slots) == tensor_sha256(state.slots)


def test_noise_draw_is_occupancy_independent_and_amplitudes_are_paired() -> None:
    state = state_fixture(torch.float64)
    full = ReplaceSlot(1, (0.0, 0.0, 0.0), source_module=0, confidence=0.5, ttl=1.0).apply(state, 1)
    original = GaussianNoise(0.25, 42).apply(state, 1).slots - state.slots
    changed = GaussianNoise(0.25, 42).apply(full, 1).slots - full.slots
    assert torch.equal(original[:, 0], changed[:, 0])
    assert torch.equal(original[:, 3], changed[:, 3])
    larger = GaussianNoise(0.5, 42).apply(state, 1).slots - state.slots
    torch.testing.assert_close(larger, 2 * original, rtol=0, atol=4e-15)


def test_noise_has_declared_scale_on_a_large_deterministic_fixture() -> None:
    shape = (128, 32)
    state = WorkspaceState(
        slots=torch.zeros(*shape, 4, dtype=torch.float64),
        age=torch.zeros(shape, dtype=torch.int64),
        occupancy=torch.ones(shape, dtype=torch.bool),
        source_module=torch.zeros(shape, dtype=torch.int64),
        confidence=torch.ones(shape, dtype=torch.float64),
        ttl=torch.ones(shape, dtype=torch.float64),
        write_count=torch.ones(shape, dtype=torch.int64),
    )
    noise = GaussianNoise(0.25, 42).apply(state, 1).slots
    assert abs(float(noise.mean())) < 0.01
    assert abs(float(noise.std()) - 0.25) < 0.01


@pytest.mark.parametrize("dtype", DTYPES)
def test_replacement_is_explicit_for_empty_and_occupied_rows(dtype: torch.dtype) -> None:
    state = state_fixture(dtype)
    output = ReplaceSlot(1, (9.0, 8.0, 7.0), source_module=3, confidence=0.75, ttl=2.5).apply(
        state, 1
    )
    assert torch.equal(output.slots[:, 1], torch.tensor([[9, 8, 7], [9, 8, 7]], dtype=dtype))
    assert bool(output.occupancy[:, 1].all())
    assert bool((output.age[:, 1] == 0).all())
    assert bool((output.source_module[:, 1] == 3).all())
    assert bool((output.confidence[:, 1] == 0.75).all())
    assert bool((output.ttl[:, 1] == 2.5).all())
    assert torch.equal(output.write_count[:, 1], state.write_count[:, 1] + 1)
    for slot in (0, 2, 3):
        assert_slot(output, slot, state, slot)


def test_freeze_private_snapshot_survives_input_output_changes_and_later_decay() -> None:
    state = state_fixture()
    before = clone(state)
    freeze = FreezeSlot(0)
    first = freeze.apply(state, 1)
    first.slots.zero_()
    state.slots.zero_()
    changed = ZeroWorkspace().apply(before, 1)
    restored = freeze.restore_before_read(changed, 2)
    assert_slot(restored, 0, before, 0)
    restored.slots.zero_()
    second = freeze.apply(changed, 2)
    assert_slot(second, 0, before, 0)
    for slot in (1, 2, 3):
        assert_slot(second, slot, changed, slot)
    with pytest.raises(WorkspaceInterventionError, match="increasing"):
        freeze.apply(before, 1)
    with pytest.raises(WorkspaceInterventionError, match="precede"):
        freeze.restore_before_read(before, 2)
    freeze.reset()
    assert_same(freeze.apply(changed, 1), changed)
    assert_slot(freeze.apply(before, 2), 0, changed, 0)


def test_freeze_schedule_captures_first_selected_cycle_and_only_restores_selected_cycles() -> None:
    state = state_fixture()
    freeze = FreezeSlot(0, cycles=(2, 4))
    assert_same(freeze.apply(state, 1), state)
    selected = ZeroWorkspace().apply(state, 1)
    assert_same(freeze.restore_before_read(selected, 2), selected)
    freeze.apply(selected, 2)
    assert_same(freeze.restore_before_read(state, 3), state)
    assert_same(freeze.apply(state, 3), state)
    assert_slot(freeze.restore_before_read(state, 4), 0, selected, 0)
    assert_slot(freeze.apply(state, 4), 0, selected, 0)


def test_freeze_rejects_changed_request_geometry_or_intent_even_on_inactive_cycle() -> None:
    state = state_fixture()
    freeze = FreezeSlot(0, cycles=(1, 3))
    freeze.apply(state, 1)
    for changed in (
        state_fixture(torch.float64),
        WorkspaceState.empty(
            batch_size=2, num_slots=5, state_dim=3, dtype=torch.float32, device=torch.device("cpu")
        ),
        WorkspaceState.empty(
            batch_size=1, num_slots=4, state_dim=3, dtype=torch.float32, device=torch.device("cpu")
        ),
    ):
        with pytest.raises(WorkspaceInterventionError, match="reset"):
            freeze.apply(changed, 2)
    freeze._spec = WorkspaceInterventionSpecV1(kind="freeze_slot", slot=1)
    with pytest.raises(WorkspaceInterventionError, match="reset"):
        freeze.restore_before_read(state, 2)


def test_real_reader_and_decay_see_restored_or_cleared_information() -> None:
    state = state_fixture()
    reader = WorkspaceReader(state_dim=3, num_modules=4, num_slots=4).eval()
    decay = WorkspaceDecay(
        state_dim=3, num_modules=4, num_slots=4, decay_rate=0.5, empty_threshold=1e-3
    ).eval()
    baseline = reader(state)
    cleared = reader(ZeroWorkspace().apply(state, 1))
    assert bool((baseline.workspace_reads != 0).any())
    assert not bool(cleared.workspace_reads.any()) and not bool(cleared.attention_weights.any())
    freeze = FreezeSlot(0)
    first = freeze.apply(state, 1)
    decayed = decay(first)
    assert not torch.equal(decayed.age[:, 0], first.age[:, 0])
    restored = freeze.restore_before_read(decayed, 2)
    assert_slot(restored, 0, first, 0)
    assert bool((reader(restored).workspace_reads != 0).any())


@pytest.mark.parametrize(
    "kind", ["zero", "slot", "swap", "shuffle", "noise", "replace", "capacity"]
)
def test_actual_latent_bridge_preserves_prefix_geometry_but_sees_intervention(kind: str) -> None:
    state = state_fixture(torch.float64)
    with torch.random.fork_rng(devices=[]):
        bridge = LatentBridge(state_dim=3, backbone_hidden_size=4, num_slots=4).double().eval()
    with torch.no_grad():
        bridge.bridge_projection.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, -0.5, 0.0]],
                dtype=torch.float64,
            )
        )
        bridge.workspace_type_embedding.zero_()
    mask = torch.tensor([[False, True, True], [True, True, True]])
    baseline = bridge(state, mask)
    changed = bridge(operation(kind).apply(state, 1), mask)
    assert changed.workspace_embeddings.shape == baseline.workspace_embeddings.shape == (2, 4, 4)
    assert torch.equal(changed.workspace_attention_mask, baseline.workspace_attention_mask)
    assert bool(changed.workspace_attention_mask.all())
    assert torch.equal(changed.workspace_position_ids, baseline.workspace_position_ids)
    assert not torch.equal(changed.workspace_embeddings, baseline.workspace_embeddings)


def test_whole_slot_permutation_does_not_invent_a_read_effect() -> None:
    # Ordinary content-only attention is permutation invariant. The causal effect
    # of shuffle must be measured at the positional prefix/output, not fabricated
    # by changing provenance metadata or asserting that every read must change.
    state = state_fixture(torch.float64)
    reader = WorkspaceReader(state_dim=3, num_modules=4, num_slots=4).double().eval()
    with torch.no_grad():
        reader.read_queries.copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 100)
    torch.testing.assert_close(
        reader(ShuffleSlots(42).apply(state, 1)).workspace_reads,
        reader(state).workspace_reads,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_noncontiguous_and_inference_owned_inputs_remain_usable(kind: str) -> None:
    state = state_fixture()
    noncontiguous = WorkspaceState(
        **{
            field.name: getattr(state, field.name).transpose(0, 1).contiguous().transpose(0, 1)
            for field in fields(state)
        }
    )
    assert not noncontiguous.slots.is_contiguous()
    with torch.inference_mode():
        inference_state = clone(noncontiguous)
        inference_result = operation(kind).apply(inference_state, 1)
    validate_intervention_state(inference_result)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = operation(kind).apply(inference_state, 1)
    assert_same(result, operation(kind).apply(state, 1))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cycle", [0, -1, True, 1.0])
def test_reject_invalid_cycle_before_applying(kind: str, cycle: Any) -> None:
    with pytest.raises(ValueError):
        operation(kind).apply(state_fixture(), cycle)


@pytest.mark.parametrize(
    "field_name", ["slots", "age", "source_module", "confidence", "ttl", "write_count"]
)
def test_reject_noncanonical_empty_records_even_on_an_inactive_cycle(field_name: str) -> None:
    state = state_fixture()
    getattr(state, field_name)[0, 1] = 0.5 if field_name == "confidence" else 3
    with pytest.raises(WorkspaceInterventionError, match="canonical"):
        ZeroWorkspace(cycles=(2,)).apply(state, 1)


@pytest.mark.parametrize("field_name", ["slots", "confidence", "ttl"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_state(field_name: str, value: float) -> None:
    state = state_fixture()
    getattr(state, field_name)[0, 0] = value
    with pytest.raises(WorkspaceInterventionError, match="finite"):
        ZeroWorkspace().apply(state, 1)


@pytest.mark.parametrize("field_name", ["age", "write_count", "ttl", "source_module"])
def test_reject_negative_occupied_metadata(field_name: str) -> None:
    state = state_fixture()
    getattr(state, field_name)[0, 0] = -1
    with pytest.raises(WorkspaceInterventionError):
        ZeroWorkspace().apply(state, 1)


@pytest.mark.parametrize(
    "field_name", ["age", "occupancy", "source_module", "confidence", "ttl", "write_count"]
)
def test_reject_metadata_shape_dtype_and_device(field_name: str) -> None:
    state = state_fixture()
    original = getattr(state, field_name)
    for tensor in (original[:, :1], original.to(torch.int32), original.to("meta")):
        with pytest.raises(WorkspaceInterventionError, match="shape/dtype/device"):
            ZeroWorkspace().apply(replace(state, **{field_name: tensor}), 1)


def test_reject_invalid_state_storage_and_out_of_range_operator_bounds() -> None:
    state = state_fixture()
    for slots in (
        state.slots.to(torch.int64),
        state.slots[:, :0],
        state.slots.to_sparse(),
        state.slots.to("meta"),
    ):
        with pytest.raises(WorkspaceInterventionError):
            ZeroWorkspace().apply(replace(state, slots=slots), 1)
    with pytest.raises(WorkspaceInterventionError):
        ZeroWorkspace().apply(None, 1)  # type: ignore[arg-type]
    for op in (
        ZeroSlot(4),
        SwapSlots(0, 4),
        ReduceWorkspaceCapacity(5),
        ReplaceSlot(0, (1.0,), source_module=0, confidence=0.5, ttl=1.0),
    ):
        with pytest.raises(WorkspaceInterventionError):
            op.apply(state, 1)


def test_overflow_fails_without_modifying_inputs_or_freeze_session() -> None:
    state = state_fixture(torch.float16)
    before = clone(state)
    for op in (
        GaussianNoise(1e100, 42),
        ReplaceSlot(0, (1e100, 0.0, 0.0), source_module=0, confidence=0.5, ttl=1.0),
        ReplaceSlot(0, (0.0, 0.0, 0.0), source_module=0, confidence=0.5, ttl=1e100),
        ReplaceSlot(0, (0.0, 0.0, 0.0), source_module=2**80, confidence=0.5, ttl=1.0),
    ):
        with pytest.raises(WorkspaceInterventionError):
            op.apply(state, 1)
        assert_same(state, before)
    state.write_count[:, 0] = torch.iinfo(torch.int64).max
    with pytest.raises(WorkspaceInterventionError, match="overflow"):
        ReplaceSlot(0, (1.0, 2.0, 3.0), source_module=0, confidence=0.5, ttl=1.0).apply(state, 1)
    freeze = FreezeSlot(0)
    invalid = clone(before)
    invalid.slots[0, 0, 0] = torch.inf
    with pytest.raises(WorkspaceInterventionError):
        freeze.apply(invalid, 1)
    assert_same(freeze.apply(before, 1), before)


def test_operator_specs_are_revalidated_after_unsafe_model_copy() -> None:
    op = ZeroSlot(1)
    op._spec = op.spec.model_copy(update={"slot": -1})
    with pytest.raises(ValidationError):
        op.apply(state_fixture(), 1)
    op._spec = WorkspaceInterventionSpecV1(kind="zero_workspace")
    with pytest.raises(WorkspaceInterventionError, match="kind"):
        op.apply(state_fixture(), 1)


def test_pure_transforms_preserve_gradients_on_untouched_records() -> None:
    state = state_fixture(torch.float64)
    state.slots.requires_grad_(True)
    transformed = GaussianNoise(0.25, 42).apply(ZeroSlot(0).apply(state, 1), 1)
    cast(Callable[[], None], transformed.slots.sum().backward)()
    assert state.slots.grad is not None
    assert not bool(state.slots.grad[:, 0].any())
    assert bool((state.slots.grad[:, 1:] == 1).all())
