"""Exact tensor wire components for the complete native writer reports.

These components preserve all four native floating dtypes and all seven state
fields. They establish representation/consistency, NOT execution or provenance.
The full report service must separately bind the original source and measurements.
No tensor, private input, prompt, model or RNG is accepted by this wire boundary.
"""

import json
import struct
from hashlib import sha256
from math import copysign, isfinite
from re import fullmatch
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

type FloatingDtype = Literal["float16", "bfloat16", "float32", "float64"]
_FLOATS = ("float16", "bfloat16", "float32", "float64")
_TENSOR_FIELDS = ("dtype", "shape", "values", "tensor_sha256")
_STATE_FIELDS = ("slots", "age", "confidence", "ttl", "occupancy", "source_module", "write_count")


def _dictionary(value: object, names: tuple[str, ...]) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("Exact wire dictionary required")
    assert isinstance(value, dict)
    if any(type(key) is not str for key in value) or set(value) != set(names):
        raise ValueError("Exact closed wire field inventory required")
    return {name: value[name] for name in names}


def _owned_data(
    value: object, expected: type[BaseModel], names: tuple[str, ...]
) -> dict[str, object]:
    # Identity must precede attributes, class hashing/equality or serialization.
    if type(value) is not expected:
        raise ValueError("Exact declared wire model required")
    assert isinstance(value, BaseModel)
    data = _dictionary(vars(value), names)
    declared = value.__pydantic_fields_set__
    if (
        value.__pydantic_extra__ is not None
        or value.__pydantic_private__ is not None
        or type(declared) is not set
        or any(type(name) is not str for name in declared)
        or declared != set(names)
    ):
        raise ValueError("Hidden or incomplete wire model metadata forbidden")
    return data


def _sequence(value: object, *, immutable: bool) -> tuple[object, ...]:
    if type(value) is not tuple and (immutable or type(value) is not list):
        raise ValueError("Exact dense wire sequence required")
    assert isinstance(value, (tuple, list))
    return tuple(value)


def _scalar_bytes(value: object, dtype: str) -> bytes:
    if dtype == "bool":
        if type(value) is not bool:
            raise ValueError("Exact boolean tensor values required")
        return b"\x01" if value else b"\x00"
    if dtype == "int64":
        if type(value) is not int:
            raise ValueError("Exact int64 tensor values required")
        assert isinstance(value, int)
        if not -(2**63) <= value < 2**63:
            raise ValueError("Tensor integer exceeds int64")
        return struct.pack("<q", value)
    if type(value) is not float:
        raise ValueError("Exact floating tensor values required")
    assert isinstance(value, float)
    if not isfinite(value):
        raise ValueError("Finite tensor values required")
    fmt = (
        "<f" if dtype == "bfloat16" else {"float16": "<e", "float32": "<f", "float64": "<d"}[dtype]
    )
    try:
        raw = struct.pack(fmt, value)
    except (OverflowError, struct.error) as error:
        raise ValueError("Tensor value exceeds its declared floating dtype") from error
    restored = struct.unpack(fmt, raw)[0]
    if restored != value or copysign(1.0, restored) != copysign(1.0, value):
        raise ValueError("Tensor value would require rounding to its declared dtype")
    if dtype == "bfloat16":
        if raw[:2] != b"\x00\x00":
            raise ValueError("Tensor value would require bfloat16 rounding")
        raw = raw[2:]
    return raw


def _dense(
    value: object, shape: tuple[int, ...], dtype: str, *, immutable: bool
) -> tuple[object, bytes]:
    if not shape:
        return value, _scalar_bytes(value, dtype)
    children = _sequence(value, immutable=immutable)
    if len(children) != shape[0]:
        raise ValueError("Dense tensor values differ from their exact shape")
    if len(shape) == 1:
        # _sequence already snapshots exact lists/tuples. Keep scalar validation
        # and byte order unchanged, but avoid a recursive pair per dense leaf.
        return tuple(child for child in children), b"".join(
            _scalar_bytes(child, dtype) for child in children
        )
    parsed = tuple(_dense(child, shape[1:], dtype, immutable=immutable) for child in children)
    return tuple(child for child, _ in parsed), b"".join(raw for _, raw in parsed)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(*parts: bytes) -> str:
    digest = sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


class _Wire(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class _TensorWire(_Wire):
    @model_validator(mode="before")
    @classmethod
    def exact_input(cls, value: object) -> object:
        return _tensor_input(value, cls, immutable=False)


class ScientificAblationFloatSlotsV1(_TensorWire):
    dtype: FloatingDtype
    shape: tuple[int, int, int]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    tensor_sha256: str


class ScientificAblationFloatMetadataV1(_TensorWire):
    dtype: FloatingDtype
    shape: tuple[int, int]
    values: tuple[tuple[float, ...], ...]
    tensor_sha256: str


class ScientificAblationIntegerMetadataV1(_TensorWire):
    dtype: Literal["int64"]
    shape: tuple[int, int]
    values: tuple[tuple[int, ...], ...]
    tensor_sha256: str


class ScientificAblationBooleanMetadataV1(_TensorWire):
    dtype: Literal["bool"]
    shape: tuple[int, int]
    values: tuple[tuple[bool, ...], ...]
    tensor_sha256: str


def _spec(expected: type[BaseModel]) -> tuple[int, tuple[str, ...]]:
    if expected is ScientificAblationFloatSlotsV1:
        return 3, _FLOATS
    if expected is ScientificAblationFloatMetadataV1:
        return 2, _FLOATS
    if expected is ScientificAblationIntegerMetadataV1:
        return 2, ("int64",)
    if expected is ScientificAblationBooleanMetadataV1:
        return 2, ("bool",)
    raise ValueError("Unknown tensor wire family")


def _tensor_input(
    value: object, expected: type[BaseModel], *, immutable: bool
) -> dict[str, object]:
    rank, dtypes = _spec(expected)
    data = (
        _dictionary(value, _TENSOR_FIELDS)
        if type(value) is dict
        else _owned_data(value, expected, _TENSOR_FIELDS)
    )
    dtype = data["dtype"]
    if type(dtype) is not str or dtype not in dtypes:
        raise ValueError("Tensor dtype differs from its declared wire family")
    assert isinstance(dtype, str)
    dimensions = _sequence(data["shape"], immutable=immutable)
    if len(dimensions) != rank or any(type(n) is not int for n in dimensions):
        raise ValueError("Exact tensor rank and integer dimensions required")
    shape: list[int] = []
    for index, dimension in enumerate(dimensions):
        assert isinstance(dimension, int)
        limit = 1 if index == 0 else (64 if index == 1 else 4096)
        if not 1 <= dimension <= limit:
            raise ValueError("Tensor exceeds declared batch/slot/width geometry")
        shape.append(dimension)
    normalized, raw = _dense(data["values"], tuple(shape), dtype, immutable=immutable)
    supplied = data["tensor_sha256"]
    if type(supplied) is not str or fullmatch(r"[0-9a-f]{64}", supplied) is None:
        raise ValueError("Exact lower-case tensor digest required")
    actual = _fingerprint(b"iris/tensor/v1", _canonical(dict(dtype=dtype, shape=shape)), raw)
    if supplied != actual:
        raise ValueError("Tensor digest differs from its exact dtype/shape/value bytes")
    return dict(dtype=dtype, shape=tuple(shape), values=normalized, tensor_sha256=actual)


def _json_values(value: object) -> object:
    if type(value) is tuple:
        assert isinstance(value, tuple)
        return [_json_values(child) for child in value]
    return value


def writer_tensor_wire_v1(value: object) -> dict[str, object]:
    """Revalidate owned immutable metadata before producing plain JSON values."""
    for expected in (
        ScientificAblationFloatSlotsV1,
        ScientificAblationFloatMetadataV1,
        ScientificAblationIntegerMetadataV1,
        ScientificAblationBooleanMetadataV1,
    ):
        if type(value) is expected:
            data = _owned_data(value, expected, _TENSOR_FIELDS)
            checked = _tensor_input(data, expected, immutable=True)
            return {name: _json_values(child) for name, child in checked.items()}
    raise ValueError("Exact declared tensor wire root required")


def _state_kinds() -> tuple[tuple[str, type[BaseModel]], ...]:
    return (
        ("slots", ScientificAblationFloatSlotsV1),
        ("age", ScientificAblationIntegerMetadataV1),
        ("confidence", ScientificAblationFloatMetadataV1),
        ("ttl", ScientificAblationFloatMetadataV1),
        ("occupancy", ScientificAblationBooleanMetadataV1),
        ("source_module", ScientificAblationIntegerMetadataV1),
        ("write_count", ScientificAblationIntegerMetadataV1),
    )


class ScientificAblationWriterStateV1(_Wire):
    slots: ScientificAblationFloatSlotsV1
    age: ScientificAblationIntegerMetadataV1
    confidence: ScientificAblationFloatMetadataV1
    ttl: ScientificAblationFloatMetadataV1
    occupancy: ScientificAblationBooleanMetadataV1
    source_module: ScientificAblationIntegerMetadataV1
    write_count: ScientificAblationIntegerMetadataV1

    @model_validator(mode="before")
    @classmethod
    def exact_input(cls, value: object) -> object:
        if cls is not ScientificAblationWriterStateV1:
            raise ValueError("Exact declared seven-field state root required")
        data = (
            _dictionary(value, _STATE_FIELDS)
            if type(value) is dict
            else _owned_data(value, ScientificAblationWriterStateV1, _STATE_FIELDS)
        )
        result: dict[str, BaseModel] = {}
        for name, expected in _state_kinds():
            child = data[name]
            if type(child) is not dict:
                child = _owned_data(child, expected, _TENSOR_FIELDS)
                _tensor_input(child, expected, immutable=True)
            result[name] = expected.model_validate(child)
        _state_semantics(result, num_modules=64)
        return result


def _state_semantics(data: dict[str, BaseModel], *, num_modules: int) -> None:
    slots, age, confidence, ttl = data["slots"], data["age"], data["confidence"], data["ttl"]
    occupied, sources, writes = data["occupancy"], data["source_module"], data["write_count"]
    assert isinstance(slots, ScientificAblationFloatSlotsV1)
    assert isinstance(age, ScientificAblationIntegerMetadataV1)
    assert isinstance(confidence, ScientificAblationFloatMetadataV1)
    assert isinstance(ttl, ScientificAblationFloatMetadataV1)
    assert isinstance(occupied, ScientificAblationBooleanMetadataV1)
    assert isinstance(sources, ScientificAblationIntegerMetadataV1)
    assert isinstance(writes, ScientificAblationIntegerMetadataV1)
    if any(
        child.shape != slots.shape[:2]
        for child in (age, confidence, ttl, occupied, sources, writes)
    ):
        raise ValueError("All seven state fields must share the original slot geometry")
    if confidence.dtype != slots.dtype or ttl.dtype != slots.dtype:
        raise ValueError("State floating fields must share their original native dtype")
    for index in range(slots.shape[1]):
        if age.values[0][index] < 0 or writes.values[0][index] < 0 or ttl.values[0][index] < 0:
            raise ValueError("State counters and TTL must be nonnegative")
        if not 0 <= confidence.values[0][index] <= 1:
            raise ValueError("State confidence must lie in [0,1]")
        source = sources.values[0][index]
        if occupied.values[0][index]:
            if not 0 <= source < num_modules:
                raise ValueError("Occupied state source is outside the trusted module domain")
        elif (
            source != -1
            or age.values[0][index] != 0
            or writes.values[0][index] != 0
            or confidence.values[0][index] != 0
            or ttl.values[0][index] != 0
            or any(value != 0 for value in slots.values[0][index])
        ):
            raise ValueError("Empty state records must be canonical across all seven fields")


def writer_state_wire_v1(value: object, *, num_modules: int) -> dict[str, object]:
    """Bind occupied sources to separately trusted geometry; not source attestation."""
    if type(num_modules) is not int or not 2 <= num_modules <= 64:
        raise ValueError("Exact trusted module count in [2,64] required")
    data = _owned_data(value, ScientificAblationWriterStateV1, _STATE_FIELDS)
    checked: dict[str, BaseModel] = {}
    wire: dict[str, object] = {}
    for name, expected in _state_kinds():
        _owned_data(data[name], expected, _TENSOR_FIELDS)
        wire[name] = writer_tensor_wire_v1(data[name])
        checked[name] = expected.model_validate(wire[name])
    _state_semantics(checked, num_modules=num_modules)
    return wire


def writer_state_sha256_v1(value: object, *, num_modules: int) -> str:
    """Match the native observed-state domain without adding a wire field."""
    wire = writer_state_wire_v1(value, num_modules=num_modules)
    digests: dict[str, object] = {}
    for name, child in wire.items():
        assert isinstance(child, dict)
        digests[name] = child["tensor_sha256"]
    return _fingerprint(b"iris/ablation-workspace-observed-state/v1", _canonical(digests))
