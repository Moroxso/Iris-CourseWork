"""Exact wire consistency and trusted CPU byte parity; no model execution."""

import json
import struct
from copy import deepcopy
from dataclasses import fields
from typing import Any

import pytest
import torch
from pydantic import BaseModel

from iris.artifacts.scientific_ablation_writer_tensors_v1 import (
    ScientificAblationBooleanMetadataV1,
    ScientificAblationFloatMetadataV1,
    ScientificAblationFloatSlotsV1,
    ScientificAblationIntegerMetadataV1,
    ScientificAblationWriterStateV1,
    writer_state_sha256_v1,
    writer_state_wire_v1,
    writer_tensor_wire_v1,
)
from iris.training.ablation_observation import ablation_workspace_state_sha256_v1
from iris.utils.hashing import tensor_sha256
from iris.workspace.state import WorkspaceState

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def tensor_wire(tensor: torch.Tensor) -> dict[str, Any]:
    return dict(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        values=tensor.tolist(),
        tensor_sha256=tensor_sha256(tensor),
    )


def state_wire(dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    return dict(
        slots=tensor_wire(torch.tensor([[[0.5, -0.0], [0.0, -0.0]]], dtype=dtype)),
        age=tensor_wire(torch.tensor([[2, 0]], dtype=torch.int64)),
        confidence=tensor_wire(torch.tensor([[0.5, -0.0]], dtype=dtype)),
        ttl=tensor_wire(torch.tensor([[1.0, -0.0]], dtype=dtype)),
        occupancy=tensor_wire(torch.tensor([[True, False]], dtype=torch.bool)),
        source_module=tensor_wire(torch.tensor([[1, -1]], dtype=torch.int64)),
        write_count=tensor_wire(torch.tensor([[3, 0]], dtype=torch.int64)),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_rank_two_floating_metadata_is_a_separate_exact_family(dtype: torch.dtype) -> None:
    raw = tensor_wire(torch.tensor([[0.5, -0.0]], dtype=dtype))
    value = ScientificAblationFloatMetadataV1.model_validate(raw)
    assert writer_tensor_wire_v1(value) == raw
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)


def test_zero_sign_changes_digest_even_when_numeric_values_compare_equal() -> None:
    raw = tensor_wire(torch.tensor([[[-0.0]]], dtype=torch.float32))
    raw["values"] = [[[0.0]]]
    with pytest.raises(ValueError, match="digest"):
        ScientificAblationFloatSlotsV1.model_validate(raw)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_four_types_keep_exact_bytes_signed_zero_and_native_state_hash(
    dtype: torch.dtype,
) -> None:
    raw = state_wire(dtype)
    value = ScientificAblationWriterStateV1.model_validate(raw)
    wire = writer_state_wire_v1(value, num_modules=4)
    assert wire == raw
    # Numeric equality alone ignores signed zero; canonical bytes cannot.
    assert json.dumps(wire, sort_keys=True) == json.dumps(raw, sort_keys=True)
    assert ScientificAblationWriterStateV1.model_validate_json(json.dumps(wire)) == value
    native = WorkspaceState(
        **{
            field.name: torch.tensor(
                raw[field.name]["values"], dtype=getattr(torch, raw[field.name]["dtype"])
            )
            for field in fields(WorkspaceState)
        }
    )
    assert writer_state_sha256_v1(value, num_modules=4) == ablation_workspace_state_sha256_v1(
        native
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_extremes_subnormals_and_both_zero_signs_have_native_byte_parity(
    dtype: torch.dtype,
) -> None:
    info = torch.finfo(dtype)
    small = torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)).item()
    raw = tensor_wire(
        torch.tensor([[[info.max, -info.max, info.tiny, small, -small, 0.0, -0.0]]], dtype=dtype)
    )
    value = ScientificAblationFloatSlotsV1.model_validate(raw)
    assert json.dumps(writer_tensor_wire_v1(value), sort_keys=True) == json.dumps(
        raw, sort_keys=True
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_every_finite_16_bit_pattern_matches_trusted_cpu_tensor_bytes(dtype: torch.dtype) -> None:
    words = [
        word
        for word in range(65536)
        if (word & (0x7C00 if dtype is torch.float16 else 0x7F80))
        != (0x7C00 if dtype is torch.float16 else 0x7F80)
    ]
    expected_count = 63488 if dtype is torch.float16 else 65280
    assert len(words) == expected_count
    raw_bytes = bytearray(b"".join(struct.pack("<H", word) for word in words))
    tensor = torch.frombuffer(raw_bytes, dtype=dtype).reshape(1, 64, expected_count // 64)
    raw = tensor_wire(tensor)
    value = ScientificAblationFloatSlotsV1.model_validate(raw)
    assert writer_tensor_wire_v1(value)["tensor_sha256"] == tensor_sha256(tensor)
    assert json.dumps(writer_tensor_wire_v1(value), sort_keys=True) == json.dumps(
        raw, sort_keys=True
    )


@pytest.mark.parametrize("dtype", DTYPES[:3])
def test_rehashed_decimal_alias_cannot_be_rounded_to_declared_type(dtype: torch.dtype) -> None:
    raw = tensor_wire(torch.tensor([[[0.1]]], dtype=dtype))
    raw["values"] = [[[0.1]]]
    with pytest.raises(ValueError, match="rounding"):
        ScientificAblationFloatSlotsV1.model_validate(raw)


def test_float64_decimal_is_not_silently_narrowed() -> None:
    raw = tensor_wire(torch.tensor([[[0.1]]], dtype=torch.float64))
    assert writer_tensor_wire_v1(ScientificAblationFloatSlotsV1.model_validate(raw)) == raw


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bad", (0, True, "0.0", None, float("nan"), float("inf"), -float("inf")))
def test_float_family_is_checked_before_pydantic_coercion(dtype: torch.dtype, bad: Any) -> None:
    raw = tensor_wire(torch.zeros((1, 1, 1), dtype=dtype))
    raw["values"] = [[[bad]]]
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)


@pytest.mark.parametrize("dtype", DTYPES[:3])
def test_finite_python_value_overflow_is_denied(dtype: torch.dtype) -> None:
    raw = tensor_wire(torch.zeros((1, 1, 1), dtype=dtype))
    raw["values"] = [[[1e300]]]
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)


@pytest.mark.parametrize("bad", (False, 0.0, "0", -(2**63) - 1, 2**63))
def test_exact_int64_family_and_bounds(bad: Any) -> None:
    raw = tensor_wire(torch.zeros((1, 1), dtype=torch.int64))
    raw["values"] = [[bad]]
    with pytest.raises(ValueError):
        ScientificAblationIntegerMetadataV1.model_validate(raw)


def test_int64_boundary_bytes_are_not_float_converted() -> None:
    raw = tensor_wire(torch.tensor([[-(2**63), 2**63 - 1]], dtype=torch.int64))
    assert writer_tensor_wire_v1(ScientificAblationIntegerMetadataV1.model_validate(raw)) == raw


@pytest.mark.parametrize("bad", (0, 1, 0.0, "false", None))
def test_exact_boolean_family(bad: Any) -> None:
    raw = tensor_wire(torch.zeros((1, 1), dtype=torch.bool))
    raw["values"] = [[bad]]
    with pytest.raises(ValueError):
        ScientificAblationBooleanMetadataV1.model_validate(raw)


@pytest.mark.parametrize(
    "shape",
    (
        [True, 1, 1],
        [1.0, 1, 1],
        [2, 1, 1],
        [1, 65, 1],
        [1, 1, 4097],
        [1, 0, 1],
        [1, 1],
        [1, 1, 1, 1],
    ),
)
def test_rank_and_generic_declared_geometry_before_values(shape: Any) -> None:
    raw = tensor_wire(torch.zeros((1, 1, 1)))
    raw["shape"] = shape
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)


def test_full_declared_slots_and_width_not_tiny_fixture_geometry() -> None:
    raw = tensor_wire(torch.zeros((1, 64, 4096), dtype=torch.float32))
    assert writer_tensor_wire_v1(ScientificAblationFloatSlotsV1.model_validate(raw)) == raw


@pytest.mark.parametrize(
    "changes",
    (
        {"values": [[[0.0, 0.0]]]},
        {"values": [[0.0]]},
        {"values": []},
        {"dtype": "float"},
        {"dtype": "int64"},
        {"tensor_sha256": "A" * 64},
        {"tensor_sha256": "a" * 64},
        {"rank": 3},
        {"shape": [1, 2, 1]},
    ),
)
def test_closed_tensor_and_rehashed_dtype_shape_value_mismatch(changes: dict[str, Any]) -> None:
    raw = tensor_wire(torch.zeros((1, 1, 1))) | changes
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)


@pytest.mark.parametrize("field", ("dtype", "shape", "values", "tensor_sha256"))
def test_tensor_formatter_rechecks_reflection_and_model_construct_bypass(field: str) -> None:
    raw = tensor_wire(torch.zeros((1, 1, 1)))
    value = ScientificAblationFloatSlotsV1.model_validate(raw)
    bad = dict(dtype="int64", shape=(1, 1), values=(((0,),),), tensor_sha256="a" * 64)[field]
    object.__setattr__(value, field, bad)
    with pytest.raises(ValueError):
        writer_tensor_wire_v1(value)
    bypass = ScientificAblationFloatSlotsV1.model_construct(**(raw | {field: bad}))
    with pytest.raises(ValueError):
        writer_tensor_wire_v1(bypass)


@pytest.mark.parametrize(
    "attribute,bad",
    (
        ("__pydantic_extra__", {}),
        ("__pydantic_private__", {}),
        ("__pydantic_fields_set__", set()),
        ("__pydantic_fields_set__", []),
        ("shape", [1, 1, 1]),
        ("values", [[[0.0]]]),
        ("hidden", True),
    ),
)
def test_hidden_extra_private_and_mutable_reflection_are_denied(attribute: str, bad: Any) -> None:
    value = ScientificAblationFloatSlotsV1.model_validate(tensor_wire(torch.zeros((1, 1, 1))))
    object.__setattr__(value, attribute, bad)
    with pytest.raises(ValueError):
        writer_tensor_wire_v1(value)


@pytest.mark.parametrize(
    "name,value",
    (
        ("age", -1),
        ("write_count", -1),
        ("ttl", -0.5),
        ("confidence", 1.5),
        ("source_module", -1),
        ("source_module", 64),
    ),
)
def test_locally_rehashed_state_semantics_are_not_just_tensor_checksums(
    name: str, value: Any
) -> None:
    raw = state_wire()
    tensor = torch.tensor(raw[name]["values"], dtype=getattr(torch, raw[name]["dtype"]))
    tensor[0, 0] = value
    raw[name] = tensor_wire(tensor)
    with pytest.raises(ValueError):
        ScientificAblationWriterStateV1.model_validate(raw)


@pytest.mark.parametrize(
    "name", ("slots", "age", "confidence", "ttl", "source_module", "write_count")
)
def test_empty_records_are_canonical_in_every_field_even_after_rehash(name: str) -> None:
    raw = state_wire()
    tensor = torch.tensor(raw[name]["values"], dtype=getattr(torch, raw[name]["dtype"]))
    tensor[0, 1] = 1
    raw[name] = tensor_wire(tensor)
    with pytest.raises(ValueError, match="canonical"):
        ScientificAblationWriterStateV1.model_validate(raw)


def test_cross_field_shape_and_float_dtype_must_agree() -> None:
    for name in ("confidence", "ttl"):
        raw = state_wire()
        raw[name] = tensor_wire(torch.tensor([[0.5, 0.0]], dtype=torch.float64))
        with pytest.raises(ValueError, match="dtype"):
            ScientificAblationWriterStateV1.model_validate(raw)
    raw = state_wire()
    raw["age"] = tensor_wire(torch.zeros((1, 1), dtype=torch.int64))
    with pytest.raises(ValueError, match="geometry"):
        ScientificAblationWriterStateV1.model_validate(raw)


@pytest.mark.parametrize("bad", (True, 1.0, "4", 1, 65))
def test_separate_trusted_module_geometry_is_strict(bad: Any) -> None:
    with pytest.raises(ValueError):
        writer_state_wire_v1(
            ScientificAblationWriterStateV1.model_validate(state_wire()), num_modules=bad
        )


def test_source_index_is_bound_to_trusted_geometry_not_self_described_state() -> None:
    raw = state_wire()
    raw["source_module"] = tensor_wire(torch.tensor([[4, -1]], dtype=torch.int64))
    value = ScientificAblationWriterStateV1.model_validate(raw)
    assert writer_state_wire_v1(value, num_modules=5) == raw
    with pytest.raises(ValueError, match="trusted"):
        writer_state_wire_v1(value, num_modules=4)


def test_all_seven_fields_mandatory_and_no_unreserved_rank_or_digest_field() -> None:
    for name in state_wire():
        raw = state_wire()
        del raw[name]
        with pytest.raises(ValueError):
            ScientificAblationWriterStateV1.model_validate(raw)
    with pytest.raises(ValueError):
        ScientificAblationWriterStateV1.model_validate(state_wire() | {"state_sha256": "a" * 64})


def test_nested_reflection_cannot_evade_state_formatter() -> None:
    value = ScientificAblationWriterStateV1.model_validate(state_wire())
    object.__setattr__(value.age, "values", ((True, 0),))
    with pytest.raises(ValueError):
        writer_state_wire_v1(value, num_modules=4)


@pytest.mark.parametrize("name", tuple(state_wire()))
def test_state_formatter_denies_any_missing_or_wrong_nested_family(name: str) -> None:
    value = ScientificAblationWriterStateV1.model_validate(state_wire())
    object.__setattr__(value, name, None)
    with pytest.raises(ValueError):
        writer_state_wire_v1(value, num_modules=4)
    object.__delattr__(value, name)
    with pytest.raises(ValueError):
        writer_state_sha256_v1(value, num_modules=4)


def test_foreign_roots_denied_before_metaclass_and_serializer_callbacks() -> None:
    calls: list[str] = []

    class ForeignMeta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("class equality")
            raise AssertionError("foreign class equality called")

        def __hash__(cls) -> int:
            calls.append("class hash")
            raise AssertionError("foreign class hash called")

    class Foreign(metaclass=ForeignMeta):
        def __getattribute__(self, name: str) -> Any:
            calls.append("attribute")
            raise AssertionError("foreign attribute called")

    with pytest.raises(ValueError):
        writer_tensor_wire_v1(Foreign())
    with pytest.raises(ValueError):
        writer_state_wire_v1(Foreign(), num_modules=4)
    assert calls == []


def test_subclasses_denied_before_candidate_serialization() -> None:
    calls: list[str] = []

    class ForeignSlots(ScientificAblationFloatSlotsV1):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append("serialize")
            raise AssertionError("candidate serializer called")

    candidate = ForeignSlots.model_construct(**tensor_wire(torch.zeros((1, 1, 1))))
    with pytest.raises(ValueError):
        writer_tensor_wire_v1(candidate)
    raw = state_wire()
    raw["slots"] = candidate
    with pytest.raises(ValueError):
        ScientificAblationWriterStateV1.model_validate(raw)
    assert calls == []


def test_foreign_dictionary_keys_and_scalar_callbacks_not_invoked() -> None:
    calls: list[str] = []

    class ForeignDict(dict[str, Any]):
        def items(self) -> Any:
            calls.append("dictionary")
            raise AssertionError("foreign dictionary called")

    class ForeignFloat(float):
        def __eq__(self, other: object) -> bool:
            calls.append("float equality")
            raise AssertionError("foreign float called")

    class ForeignKey(str):
        def __hash__(self) -> int:
            calls.append("key hash")
            return str.__hash__(self)

    raw = tensor_wire(torch.zeros((1, 1, 1)))
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(ForeignDict(raw))
    keyed = deepcopy(raw)
    keyed[ForeignKey("hidden")] = True
    calls.clear()
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(keyed)
    raw["values"] = [[[ForeignFloat(0.0)]]]
    with pytest.raises(ValueError):
        ScientificAblationFloatSlotsV1.model_validate(raw)
    assert calls == []


def test_other_tensor_family_cannot_relabel_nested_field() -> None:
    raw = state_wire()
    raw["age"] = ScientificAblationBooleanMetadataV1.model_validate(raw["occupancy"])
    with pytest.raises(ValueError):
        ScientificAblationWriterStateV1.model_validate(raw)


def test_models_are_frozen_and_owned_output_does_not_alias_input() -> None:
    raw = state_wire()
    value = ScientificAblationWriterStateV1.model_validate(raw)
    with pytest.raises(ValueError):
        value.age = value.write_count
    wire = writer_state_wire_v1(value, num_modules=4)
    raw["age"]["values"][0][0] = 99
    assert wire != raw
    assert writer_state_wire_v1(value, num_modules=4) == wire
    assert isinstance(value.slots, BaseModel)
