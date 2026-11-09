"""Scalar/dense performance paths preserve the pre-fastpath byte/type contract."""

import json
import math
import random
import struct
from typing import Any, NoReturn

import pytest

from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.artifacts import scientific_ablation_writer_tensors_v1 as tensors
from iris.utils.hashing import canonical_json_bytes


def _legacy_dense(
    value: object, shape: tuple[int, ...], dtype: str, *, immutable: bool
) -> tuple[object, bytes]:
    """The original recursive implementation, with unchanged scalar validator."""
    if not shape:
        return value, tensors._scalar_bytes(value, dtype)
    children = tensors._sequence(value, immutable=immutable)
    if len(children) != shape[0]:
        raise ValueError("Dense tensor values differ from their exact shape")
    parsed = tuple(
        _legacy_dense(child, shape[1:], dtype, immutable=immutable) for child in children
    )
    return tuple(child for child, _ in parsed), b"".join(raw for _, raw in parsed)


@pytest.mark.parametrize(
    "value",
    (0.0, -0.0, 5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308, 1e-4, 1e16),
)
def test_float_ascii_repr_matches_both_stdlib_encoder_paths(value: float) -> None:
    encoded = repr(value).encode("ascii")
    assert encoded == canonical_json_bytes(value)
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    assert encoded == "".join(encoder.iterencode(value, _one_shot=False)).encode()
    layout.F.check(value)


def test_seeded_binary64_and_int64_numeric_representation_equivalence() -> None:
    rng = random.Random(43137)
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    for _ in range(10000):
        value = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(value):
            expected = canonical_json_bytes(value)
            assert expected == repr(value).encode("ascii")
            assert expected == "".join(encoder.iterencode(value, _one_shot=False)).encode()
            assert len(expected) <= layout.F.bound
            layout.F.check(value)
        integer = rng.randrange(-(2**63), 2**63)
        assert canonical_json_bytes(integer) == repr(integer).encode("ascii")
        layout.N.check(integer)


@pytest.mark.parametrize(
    "value", (0.0, -0.0, 5e-324, 1.7976931348623157e308, -(2**63), 0, 2**63 - 1)
)
def test_numeric_leaf_does_not_construct_a_canonical_encoder(
    monkeypatch: pytest.MonkeyPatch, value: int | float
) -> None:
    def forbidden(_: object) -> NoReturn:
        raise AssertionError("Numeric leaves must not allocate a JSON encoder")

    monkeypatch.setattr(layout, "canonical_json_bytes", forbidden)
    (layout.F if type(value) is float else layout.N).check(value)


@pytest.mark.parametrize("value", (0.0, -0.0, 1.25, -(2**63), 0, 2**63 - 1))
def test_numeric_encoding_reservation_is_still_explicit(
    monkeypatch: pytest.MonkeyPatch, value: int | float
) -> None:
    expected_size = len(canonical_json_bytes(value))
    scalar = layout.F if type(value) is float else layout.N
    monkeypatch.setattr(layout._Scalar, "bound", property(lambda self: expected_size))
    scalar.check(value)
    monkeypatch.setattr(layout._Scalar, "bound", property(lambda self: expected_size - 1))
    with pytest.raises(ValueError, match="encoding exceeds reservation"):
        scalar.check(value)


@pytest.mark.parametrize("value", (False, True, "Ж😀\n\0", "a" * 64, "\ud800"))
def test_other_primitive_encodings_still_use_the_canonical_boundary(
    monkeypatch: pytest.MonkeyPatch, value: bool | str
) -> None:
    calls: list[object] = []

    def observed(item: object) -> bytes:
        calls.append(item)
        return canonical_json_bytes(item)

    monkeypatch.setattr(layout, "canonical_json_bytes", observed)
    scalar = layout._Scalar("bool" if type(value) is bool else "text")
    if value == "\ud800":
        with pytest.raises(UnicodeError):
            scalar.check(value)
    else:
        scalar.check(value)
    assert calls == [value]


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int64", "bool"))
@pytest.mark.parametrize("immutable", (False, True))
@pytest.mark.parametrize("rank", (1, 2, 3))
def test_dense_reference_equality_for_every_dtype_rank_and_container_mode(
    dtype: str, immutable: bool, rank: int
) -> None:
    values: tuple[object, ...]
    if dtype == "bool":
        values = (False, True, False, True)
    elif dtype == "int64":
        values = (-(2**63), -1, 0, 2**63 - 1)
    else:
        values = (-0.0, 0.0, 1.5, -2.0)
    candidate: object = values if immutable else list(values)
    shape: tuple[int, ...] = (4,)
    for _ in range(rank - 1):
        candidate = (candidate, candidate) if immutable else [candidate, candidate]
        shape = (2, *shape)
    expected = _legacy_dense(candidate, shape, dtype, immutable=immutable)
    actual = tensors._dense(candidate, shape, dtype, immutable=immutable)
    assert actual == expected
    assert actual[1] == expected[1]
    if dtype.startswith("float") or dtype == "bfloat16":
        size = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}[dtype]
        assert (
            actual[1][:size]
            == {2: b"\x00\x80", 4: b"\x00\x00\x00\x80", 8: b"\x00" * 7 + b"\x80"}[size]
        )


@pytest.mark.parametrize(
    "dtype,value,shape,immutable",
    (
        ("float16", [0.1], (1,), False),
        ("bfloat16", [1.0001], (1,), False),
        ("float32", [0.1], (1,), False),
        ("float32", [1e300], (1,), False),
        ("float64", [math.inf], (1,), False),
        ("float64", [math.nan], (1,), False),
        ("float64", [1], (1,), False),
        ("int64", [True], (1,), False),
        ("int64", [2**63], (1,), False),
        ("bool", [0], (1,), False),
        ("float32", [0.0], (2,), False),
        ("float32", [[0.0]], (1,), False),
        ("float32", [0.0], (1,), True),
        ("float32", [0.0, (1.0,)], (2, 1), False),
    ),
)
def test_dense_rejections_match_original_exactly(
    dtype: str, value: Any, shape: tuple[int, ...], immutable: bool
) -> None:
    with pytest.raises(ValueError) as reference:
        _legacy_dense(value, shape, dtype, immutable=immutable)
    with pytest.raises(ValueError) as actual:
        tensors._dense(value, shape, dtype, immutable=immutable)
    assert str(actual.value) == str(reference.value)


def test_terminal_dense_vector_has_no_per_scalar_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = tensors._dense

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("A dense vector must not recurse per scalar")

    monkeypatch.setattr(tensors, "_dense", forbidden)
    actual = dense([0.0, -0.0, 1.5], (3,), "float32", immutable=False)
    assert actual == ((0.0, -0.0, 1.5), struct.pack("<fff", 0.0, -0.0, 1.5))


def test_dense_output_does_not_alias_mutable_input_and_keeps_original_tuple_ownership() -> None:
    mutable = [0.0, -0.0, 1.5]
    parsed, raw = tensors._dense(mutable, (3,), "float64", immutable=False)
    mutable[0] = 99.0
    assert parsed == (0.0, -0.0, 1.5)
    assert raw == struct.pack("<ddd", 0.0, -0.0, 1.5)
    immutable = (0.0, -0.0, 1.5)
    owned, _ = tensors._dense(immutable, (3,), "float64", immutable=True)
    assert owned == immutable and owned is not immutable


def test_foreign_numeric_and_sequence_callbacks_are_never_invoked() -> None:
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        calls.append("callback")
        raise AssertionError("Candidate callback invoked")

    class ForeignFloat(float):
        __repr__ = forbidden
        __float__ = forbidden
        __eq__ = forbidden

    class ForeignInt(int):
        __repr__ = forbidden
        __int__ = forbidden

    class ForeignList(list[object]):
        __iter__ = forbidden
        __len__ = forbidden
        __getitem__ = forbidden

    for scalar, value in ((layout.F, ForeignFloat(0.0)), (layout.N, ForeignInt(0))):
        with pytest.raises(ValueError):
            scalar.check(value)
    with pytest.raises(ValueError):
        tensors._dense([ForeignFloat(0.0)], (1,), "float64", immutable=False)
    with pytest.raises(ValueError):
        tensors._dense(ForeignList([0.0]), (1,), "float64", immutable=False)
    assert calls == []


def test_dense_stops_at_first_bad_scalar_without_inspecting_later_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tensors._scalar_bytes
    bad = object()
    seen: list[object] = []

    def observed(value: object, dtype: str) -> bytes:
        seen.append(value)
        return original(value, dtype)

    monkeypatch.setattr(tensors, "_scalar_bytes", observed)
    with pytest.raises(ValueError, match="Exact floating tensor values"):
        tensors._dense([0.0, bad, 1.0], (3,), "float64", immutable=False)
    assert len(seen) == 2 and seen[0] == 0.0 and seen[1] is bad
