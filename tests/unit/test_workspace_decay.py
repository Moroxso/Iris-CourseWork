"""Contract tests for functional workspace lifecycle decay."""

from dataclasses import fields, replace
from typing import Any, cast

import pytest
import torch

from iris.workspace import WorkspaceDecay, WorkspaceDecayError, WorkspaceState


def make_decay(
    *,
    state_dim: int = 2,
    num_modules: int = 4,
    num_slots: int = 4,
    decay_rate: float = 0.85,
    empty_threshold: float = 1.0e-3,
) -> WorkspaceDecay:
    """Create a compact lifecycle transition."""
    return WorkspaceDecay(
        state_dim=state_dim,
        num_modules=num_modules,
        num_slots=num_slots,
        decay_rate=decay_rate,
        empty_threshold=empty_threshold,
    )


def make_state(
    *,
    batch_size: int = 1,
    num_slots: int = 4,
    state_dim: int = 2,
    dtype: torch.dtype = torch.float32,
    slots: torch.Tensor | None = None,
    occupancy: torch.Tensor | None = None,
    age: torch.Tensor | None = None,
    source_module: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    ttl: torch.Tensor | None = None,
    write_count: torch.Tensor | None = None,
    requires_grad: bool = False,
) -> WorkspaceState:
    """Build a valid state with explicit occupied-slot metadata."""
    if slots is None:
        slots = torch.zeros(batch_size, num_slots, state_dim, dtype=dtype)
        slots[..., 0] = 1.0
    device = slots.device
    if occupancy is None:
        occupancy = torch.ones(batch_size, num_slots, dtype=torch.bool, device=device)
    shape = (batch_size, num_slots)
    if age is None:
        age = torch.zeros(shape, dtype=torch.int64, device=device)
    if source_module is None:
        source_module = (
            torch.arange(num_slots, dtype=torch.int64, device=device)
            .remainder(4)
            .expand(batch_size, -1)
            .clone()
        )
        source_module = torch.where(
            occupancy,
            source_module,
            torch.full_like(source_module, -1),
        )
    if confidence is None:
        confidence = occupancy.to(dtype=dtype) * 0.5
    if ttl is None:
        ttl = occupancy.to(dtype=dtype) * 2.0
    if write_count is None:
        write_count = occupancy.to(dtype=torch.int64)
    slots = slots.requires_grad_(requires_grad)
    confidence = confidence.requires_grad_(requires_grad)
    ttl = ttl.requires_grad_(requires_grad)
    return WorkspaceState(
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source_module,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
    )


def assert_canonical_empty(state: WorkspaceState) -> None:
    """Require every empty slot to use the canonical representation."""
    empty = ~state.occupancy
    assert torch.equal(
        state.slots.masked_select(empty.unsqueeze(-1)),
        torch.zeros_like(state.slots.masked_select(empty.unsqueeze(-1))),
    )
    assert torch.equal(
        state.age.masked_select(empty), torch.zeros_like(state.age.masked_select(empty))
    )
    assert torch.equal(
        state.source_module.masked_select(empty),
        torch.full_like(state.source_module.masked_select(empty), -1),
    )
    assert torch.equal(
        state.confidence.masked_select(empty),
        torch.zeros_like(state.confidence.masked_select(empty)),
    )
    assert torch.equal(
        state.ttl.masked_select(empty), torch.zeros_like(state.ttl.masked_select(empty))
    )
    assert torch.equal(
        state.write_count.masked_select(empty),
        torch.zeros_like(state.write_count.masked_select(empty)),
    )


def test_exact_mixed_lifecycle_order_and_metadata_transition() -> None:
    decay = make_decay()
    slots = torch.tensor([[[3.0, 4.0], [1.0, 0.0], [2.0, 0.0], [5e-4, 0.0]]])
    state = make_state(
        slots=slots,
        age=torch.tensor([[0, 2, 4, 6]]),
        source_module=torch.tensor([[0, 1, 2, 3]]),
        confidence=torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
        ttl=torch.tensor([[2.0, 1.0, 0.5, 2.0]]),
        write_count=torch.tensor([[1, 2, 3, 4]]),
    )

    output = decay(state)

    expected_slots = torch.tensor([[[3.0, 4.0], [0.85, 0.0], [1.70, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(output.slots, expected_slots, rtol=0, atol=1e-7)
    torch.testing.assert_close(output.ttl, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(output.age, torch.tensor([[1, 3, 5, 0]]))
    assert torch.equal(output.occupancy, torch.tensor([[True, True, True, False]]))
    assert torch.equal(output.source_module, torch.tensor([[0, 1, 2, -1]]))
    torch.testing.assert_close(output.confidence, torch.tensor([[0.2, 0.3, 0.4, 0.0]]))
    assert torch.equal(output.write_count, torch.tensor([[1, 2, 3, 0]]))
    assert_canonical_empty(output)


def test_ttl_expiry_decays_but_does_not_immediately_remove_a_healthy_slot() -> None:
    state = make_state(
        num_slots=1,
        slots=torch.tensor([[[4.0, 3.0]]]),
        ttl=torch.tensor([[1.0]]),
        source_module=torch.tensor([[2]]),
        confidence=torch.tensor([[0.7]]),
        write_count=torch.tensor([[9]]),
    )
    output = make_decay(num_slots=1)(state)

    torch.testing.assert_close(output.slots, torch.tensor([[[3.4, 2.55]]]))
    assert output.occupancy.item()
    assert output.ttl.item() == 0
    assert output.source_module.item() == 2
    assert output.confidence.item() == pytest.approx(0.7)
    assert output.write_count.item() == 9


def test_threshold_is_strict_and_evaluated_after_decay() -> None:
    decay = make_decay(
        state_dim=1,
        num_slots=3,
        decay_rate=0.5,
        empty_threshold=1e-3,
    )
    state = make_state(
        num_slots=3,
        state_dim=1,
        dtype=torch.float64,
        slots=torch.tensor([[[0.0018], [0.0020], [0.0022]]], dtype=torch.float64),
        ttl=torch.ones(1, 3, dtype=torch.float64),
    )

    output = decay(state)

    assert torch.equal(output.occupancy, torch.tensor([[False, True, True]]))
    torch.testing.assert_close(
        output.slots,
        torch.tensor([[[0.0], [0.0010], [0.0011]]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_threshold_also_clears_nonexpired_low_norm_content() -> None:
    state = make_state(
        num_slots=1,
        state_dim=1,
        slots=torch.tensor([[[5.0e-4]]]),
        ttl=torch.tensor([[3.0]]),
    )
    output = make_decay(state_dim=1, num_slots=1)(state)

    assert not output.occupancy.item()
    assert_canonical_empty(output)


def test_multiple_expired_cycles_apply_rho_power_until_canonical_clear() -> None:
    decay = make_decay(
        state_dim=1,
        num_slots=1,
        decay_rate=0.5,
        empty_threshold=0.1,
    )
    state = make_state(
        num_slots=1,
        state_dim=1,
        slots=torch.tensor([[[1.0]]]),
        ttl=torch.tensor([[0.0]]),
        age=torch.tensor([[7]]),
        write_count=torch.tensor([[5]]),
    )

    first = decay(state)
    second = decay(first)
    third = decay(second)
    fourth = decay(third)

    for index, current in enumerate((first, second, third), start=1):
        assert current.occupancy.item()
        assert current.slots.item() == pytest.approx(0.5**index)
        assert current.ttl.item() == 0
        assert current.age.item() == 7 + index
        assert current.write_count.item() == 5
    assert not fourth.occupancy.item()
    assert_canonical_empty(fourth)


def test_empty_slots_do_not_age_and_finite_stale_metadata_is_repaired() -> None:
    occupancy = torch.tensor([[True, False, False, True]])
    state = make_state(
        slots=torch.tensor([[[1.0, 0.0], [9.0, -8.0], [7.0, 6.0], [2.0, 0.0]]]),
        occupancy=occupancy,
        age=torch.tensor([[2, -90, -91, 3]]),
        source_module=torch.tensor([[0, 99, -20, 1]]),
        confidence=torch.tensor([[0.2, 8.0, -9.0, 0.3]]),
        ttl=torch.tensor([[2.0, 0.0, 0.0, 2.0]]),
        write_count=torch.tensor([[1, -20, -21, 2]]),
    )

    output = make_decay()(state)

    assert torch.equal(output.age, torch.tensor([[3, 0, 0, 4]]))
    assert_canonical_empty(output)


def test_forward_is_functional_and_does_not_mutate_or_alias_input_tensors() -> None:
    state = make_state()
    before = {
        field.name: cast(torch.Tensor, getattr(state, field.name)).clone()
        for field in fields(state)
    }

    output = make_decay()(state)

    for field in fields(state):
        tensor = cast(torch.Tensor, getattr(state, field.name))
        assert torch.equal(tensor, before[field.name])
        assert cast(torch.Tensor, getattr(output, field.name)).data_ptr() != tensor.data_ptr()


def test_backward_factors_are_identity_decay_and_zero_for_cleared_slots() -> None:
    slots = torch.tensor([[[2.0, 1.0], [3.0, 1.0], [5e-4, 0.0]]], dtype=torch.float64)
    confidence = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64)
    ttl = torch.tensor([[2.0, 1.0, 2.0]], dtype=torch.float64)
    state = make_state(
        num_slots=3,
        dtype=torch.float64,
        slots=slots,
        confidence=confidence,
        ttl=ttl,
        requires_grad=True,
    )

    output = make_decay(num_slots=3)(state)
    (output.slots.sum() + output.confidence.sum() + output.ttl.sum()).backward()

    torch.testing.assert_close(
        state.slots.grad,
        torch.tensor([[[1.0, 1.0], [0.85, 0.85], [0.0, 0.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state.confidence.grad,
        torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state.ttl.grad,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )


def test_threshold_uses_euclidean_l2_norm_not_maximum_component() -> None:
    state = make_state(
        num_slots=1,
        slots=torch.tensor([[[8.0e-4, 8.0e-4]]]),
        ttl=torch.tensor([[2.0]]),
    )

    output = make_decay(num_slots=1)(state)

    assert output.occupancy.item()
    torch.testing.assert_close(output.slots, state.slots)


def test_materializes_inference_state_outside_global_inference_mode() -> None:
    with torch.inference_mode():
        state = make_state()
    output = make_decay()(state)

    for field in fields(output):
        assert not cast(torch.Tensor, getattr(output, field.name)).is_inference()


def test_supports_global_inference_mode() -> None:
    with torch.inference_mode():
        output = make_decay()(make_state())

    for field in fields(output):
        tensor = cast(torch.Tensor, getattr(output, field.name))
        assert tensor.is_inference()
        assert not tensor.requires_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_preserves_supported_storage_dtype(dtype: torch.dtype) -> None:
    state = make_state(dtype=dtype)
    output = make_decay()(state)

    assert output.slots.dtype == dtype
    assert output.confidence.dtype == dtype
    assert output.ttl.dtype == dtype
    assert bool(torch.isfinite(output.slots).all())


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.float16, 60_000.0), (torch.bfloat16, 1.0e30)],
)
def test_large_low_precision_values_decay_without_norm_overflow(
    dtype: torch.dtype,
    magnitude: float,
) -> None:
    state = make_state(
        num_slots=1,
        state_dim=256,
        dtype=dtype,
        slots=torch.full((1, 1, 256), magnitude, dtype=dtype),
        ttl=torch.zeros(1, 1, dtype=dtype),
    )
    output = make_decay(state_dim=256, num_slots=1)(state)

    assert output.occupancy.item()
    assert bool(torch.isfinite(output.slots).all())
    torch.testing.assert_close(
        output.slots,
        (state.slots.float() * 0.85).to(dtype=dtype),
        rtol=0,
        atol=0,
    )


def test_fp16_emptiness_uses_the_actual_post_cast_stored_value() -> None:
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float16), torch.tensor(1.0, dtype=torch.float16)
    )
    state = make_state(
        num_slots=1,
        state_dim=1,
        dtype=torch.float16,
        slots=smallest.reshape(1, 1, 1),
        ttl=torch.zeros(1, 1, dtype=torch.float16),
    )
    output = make_decay(
        state_dim=1,
        num_slots=1,
        decay_rate=0.5,
        empty_threshold=1e-20,
    )(state)

    assert output.slots.item() == 0
    assert not output.occupancy.item()


def test_external_autocast_cannot_change_decay_result() -> None:
    state = make_state(ttl=torch.tensor([[2.0, 1.0, 0.5, 2.0]]))
    decay = make_decay()
    reference = decay(state)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = decay(state)

    for field in fields(reference):
        torch.testing.assert_close(
            cast(torch.Tensor, getattr(actual, field.name)),
            cast(torch.Tensor, getattr(reference, field.name)),
            rtol=0,
            atol=0,
        )


def test_decay_is_parameter_free_deterministic_and_rng_free() -> None:
    decay = make_decay()
    state = make_state()
    assert list(decay.parameters()) == []
    torch.manual_seed(8128)
    rng_before = torch.random.get_rng_state().clone()

    first = decay(state)
    rng_after = torch.random.get_rng_state()
    second = decay(state)

    assert torch.equal(rng_before, rng_after)
    for field in fields(first):
        assert torch.equal(
            cast(torch.Tensor, getattr(first, field.name)),
            cast(torch.Tensor, getattr(second, field.name)),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("state_dim", 0),
        ("state_dim", True),
        ("num_modules", 0),
        ("num_slots", -1),
    ],
)
def test_rejects_invalid_dimensions(name: str, value: object) -> None:
    arguments: dict[str, Any] = {
        "state_dim": 2,
        "num_modules": 4,
        "num_slots": 4,
        "decay_rate": 0.85,
        "empty_threshold": 1e-3,
    }
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        WorkspaceDecay(**arguments)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, float("nan"), float("inf"), True])
def test_rejects_invalid_decay_rate(value: object) -> None:
    with pytest.raises(ValueError, match="decay_rate"):
        WorkspaceDecay(
            state_dim=2,
            num_modules=4,
            num_slots=4,
            decay_rate=cast(float, value),
            empty_threshold=1e-3,
        )


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf"), True])
def test_rejects_invalid_empty_threshold(value: object) -> None:
    with pytest.raises(ValueError, match="empty_threshold"):
        WorkspaceDecay(
            state_dim=2,
            num_modules=4,
            num_slots=4,
            decay_rate=0.85,
            empty_threshold=cast(float, value),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("slots", "not-a-tensor", "slots"),
        ("slots", torch.zeros(1, 4), "slots"),
        ("slots", torch.zeros(0, 4, 2), "slots"),
        ("slots", torch.zeros(1, 4, 2, dtype=torch.int64), "floating"),
        ("slots", torch.full((1, 4, 2), float("nan")), "NaN or Inf"),
        ("age", torch.zeros(1, 3, dtype=torch.int64), "age"),
        ("age", torch.zeros(1, 4), "age"),
        ("occupancy", torch.zeros(1, 4, dtype=torch.int64), "occupancy"),
        ("source_module", torch.zeros(1, 4), "source_module"),
        ("confidence", torch.zeros(1, 4, dtype=torch.float64), "confidence"),
        ("ttl", torch.zeros(1, 4, dtype=torch.float64), "ttl"),
        ("write_count", torch.zeros(1, 4), "write_count"),
    ],
)
def test_rejects_malformed_field_boundaries(
    field: str,
    replacement: object,
    message: str,
) -> None:
    state = make_state()
    with pytest.raises(WorkspaceDecayError, match=message):
        make_decay()(replace(state, **{field: cast(Any, replacement)}))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("age", torch.tensor([[-1, 0, 0, 0]]), "non-negative"),
        ("write_count", torch.tensor([[0, -1, 0, 0]]), "non-negative"),
        ("source_module", torch.tensor([[4, 1, 2, 3]]), "valid source module"),
        ("source_module", torch.tensor([[-1, 1, 2, 3]]), "valid source module"),
        ("confidence", torch.tensor([[1.1, 0.5, 0.5, 0.5]]), r"\[0, 1\]"),
        ("confidence", torch.tensor([[float("inf"), 0.5, 0.5, 0.5]]), "finite"),
        ("ttl", torch.tensor([[-0.1, 2.0, 2.0, 2.0]]), "non-negative"),
        ("ttl", torch.tensor([[float("nan"), 2.0, 2.0, 2.0]]), "finite"),
    ],
)
def test_rejects_invalid_state_values(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(WorkspaceDecayError, match=message):
        make_decay()(replace(make_state(), **{field: replacement}))


def test_rejects_nonzero_ttl_on_an_empty_slot() -> None:
    occupancy = torch.tensor([[True, False, True, True]])
    state = make_state(occupancy=occupancy, ttl=torch.tensor([[2.0, 1.0, 2.0, 2.0]]))
    with pytest.raises(WorkspaceDecayError, match="zero TTL"):
        make_decay()(state)


def test_rejects_age_increment_overflow() -> None:
    age = torch.tensor([[torch.iinfo(torch.int64).max, 0, 0, 0]])
    with pytest.raises(WorkspaceDecayError, match="overflow"):
        make_decay()(replace(make_state(), age=age))


def test_rejects_wrong_input_type_with_typed_error() -> None:
    with pytest.raises(WorkspaceDecayError, match="WorkspaceState"):
        make_decay()(cast(Any, object()))
