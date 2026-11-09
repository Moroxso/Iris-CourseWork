"""Saved chunk DATA validation, not a native or journal/idempotence attestation."""

from typing import Any

import pytest

from iris.application.scientific_ablation_kernel_chunk import (
    check_kernel_chunk_prefix_v1,
    checked_kernel_chunk_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)

PLAN, HEAD = "a" * 64, "b" * 64


def intent(**changes: Any) -> Chunk:
    return Chunk.model_validate(
        dict(
            kernel_plan_sha256=PLAN,
            acknowledgement_head_sha256=HEAD,
            acknowledged_rows=1,
            total_rows=3,
            rows_per_call=1,
            stop_row_index=2,
        )
        | changes
    )


@pytest.mark.parametrize("start,total,budget,stop", ((0, 3, 1, 1), (1, 3, 5, 3), (3, 3, 1, 3)))
def test_exact_interval_and_snapshot(start: int, total: int, budget: int, stop: int) -> None:
    value = intent(
        acknowledged_rows=start, total_rows=total, rows_per_call=budget, stop_row_index=stop
    )
    checked = checked_kernel_chunk_v1(value, expected_plan_sha256=PLAN, rows_per_call=budget)
    assert checked == value and checked is not value
    assert Chunk.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "changes",
    (
        {"acknowledged_rows": True},
        {"acknowledged_rows": -1},
        {"acknowledged_rows": 4},
        {"total_rows": False},
        {"total_rows": 0},
        {"total_rows": 5001},
        {"rows_per_call": True},
        {"rows_per_call": 0},
        {"rows_per_call": 5001},
        {"stop_row_index": True},
        {"stop_row_index": 1},
        {"stop_row_index": 3},
        {"stop_row_index": 5001},
        {"kernel_plan_sha256": "A" * 64},
        {"acknowledgement_head_sha256": b"b" * 64},
        {"new_native_execution_authorized": True},
    ),
)
def test_invalid_saved_intent_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        intent(**changes)


@pytest.mark.parametrize("budget", (None, True, 0, 2, "1"))
def test_saved_budget_is_not_default_or_another_interval(budget: Any) -> None:
    with pytest.raises(ValueError):
        checked_kernel_chunk_v1(intent(), expected_plan_sha256=PLAN, rows_per_call=budget)


@pytest.mark.parametrize(
    "changes",
    (
        {"plan_sha256": "c" * 64},
        {"acknowledgement_head_sha256": "c" * 64},
        {"acknowledged_rows": 2},
        {"acknowledged_rows": True},
        {"total_rows": 4},
        {"total_rows": True},
    ),
)
def test_exact_locked_prefix_is_not_a_moving_head(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = (
        dict(
            plan_sha256=PLAN,
            acknowledgement_head_sha256=HEAD,
            acknowledged_rows=1,
            total_rows=3,
            rows_per_call=1,
        )
        | changes
    )
    with pytest.raises(ValueError):
        check_kernel_chunk_prefix_v1(intent(), **arguments)


@pytest.mark.parametrize("poison", ("extra", "private", "unchecked", "bool", "subclass"))
def test_unchecked_hidden_or_subclass_intent_is_not_trusted(poison: str) -> None:
    value = intent()
    if poison == "extra":
        vars(value)["unknown"] = "hidden"
    elif poison == "private":
        object.__setattr__(value, "__pydantic_private__", {})
    elif poison == "unchecked":
        value = Chunk.model_construct(**(value.model_dump() | {"stop_row_index": 3}))
    elif poison == "bool":
        vars(value)["rows_per_call"] = True
    else:

        class Subclass(Chunk):
            pass

        value = Subclass.model_validate(value.model_dump())
    with pytest.raises((TypeError, ValueError)):
        checked_kernel_chunk_v1(value, expected_plan_sha256=PLAN, rows_per_call=1)


def test_optional_guard_preserves_original_no_guard_contract() -> None:
    assert checked_kernel_chunk_v1(None, expected_plan_sha256="unused", rows_per_call=None) is None
    check_kernel_chunk_prefix_v1(
        None,
        plan_sha256="unused",
        acknowledgement_head_sha256="unused",
        acknowledged_rows=0,
        total_rows=1,
        rows_per_call=None,
    )
