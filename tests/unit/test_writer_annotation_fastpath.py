"""Primitive dispatch skips typing reflection without relaxing writer DATA rules."""

import math
import random
import struct
from typing import Annotated, Any, Literal, NoReturn

import pytest
from pydantic import BaseModel

from iris.artifacts import scientific_ablation_writer_observations_v1 as writer


def forbidden_reflection(_: object) -> NoReturn:
    raise AssertionError("Primitive leaves must not repeatedly reflect typing annotations")


@pytest.mark.parametrize("immutable", (False, True))
@pytest.mark.parametrize(
    "annotation,value",
    (
        (str, "plain"),
        (str, ""),
        (str, "\u043f\u0440\u0438\u043c\u0435\u0440"),
        (int, 0),
        (int, -(2**63)),
        (int, 2**100),
        (float, 0.0),
        (float, -0.0),
        (float, math.nextafter(0.0, 1.0)),
        (float, float.fromhex("0x1.fffffffffffffp+1023")),
        (bool, False),
        (bool, True),
        (type(None), None),
    ),
)
def test_exact_primitives_do_not_reflect(
    monkeypatch: pytest.MonkeyPatch, immutable: bool, annotation: object, value: object
) -> None:
    monkeypatch.setattr(writer, "get_origin", forbidden_reflection)
    monkeypatch.setattr(writer, "get_args", forbidden_reflection)
    assert writer._annotation(annotation, value, immutable=immutable, depth=32) is value


@pytest.mark.parametrize("immutable", (False, True))
@pytest.mark.parametrize(
    "annotation,value",
    (
        (int, True),
        (int, 1.0),
        (float, 1),
        (float, True),
        (bool, 0),
        (bool, 1),
        (str, b"text"),
        (type(None), False),
    ),
)
def test_wrong_primitive_families_still_refuse(
    immutable: bool, annotation: object, value: object
) -> None:
    with pytest.raises(ValueError, match="Exact declared writer primitive family"):
        writer._annotation(annotation, value, immutable=immutable, depth=0)


@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_nonfinite_leaves_still_refuse(value: float) -> None:
    with pytest.raises(ValueError, match="Finite writer floating metadata"):
        writer._annotation(float, value, immutable=True, depth=0)


def test_float_bits_keep_exact_values_and_signed_zero() -> None:
    rng = random.Random(137)
    for _ in range(1000):
        value = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(value):
            result = writer._annotation(float, value, immutable=False, depth=1)
            assert type(result) is float
            assert struct.pack("<d", result) == struct.pack("<d", value)
        else:
            with pytest.raises(ValueError, match="Finite writer floating metadata"):
                writer._annotation(float, value, immutable=False, depth=1)


def test_depth_guard_remains_before_primitive_dispatch() -> None:
    with pytest.raises(ValueError, match="Writer metadata depth exceeded"):
        writer._annotation(float, "wrong-family", immutable=False, depth=33)


def test_alias_annotated_union_and_fixed_variadic_tuples() -> None:
    type alias = float
    for annotation in (alias, Annotated[float, "unchanged metadata"], float | None):
        assert writer._annotation(annotation, 1.25, immutable=True, depth=0) == 1.25
    assert writer._annotation(tuple[int, str], [2, "a"], immutable=False, depth=0) == (2, "a")
    assert writer._annotation(tuple[float, ...], [1.0, 2.0], immutable=False, depth=0) == (1.0, 2.0)
    assert writer._annotation(tuple[float, ...], (1.0,), immutable=True, depth=0) == (1.0,)
    with pytest.raises(ValueError, match="Exact immutable writer sequence"):
        writer._annotation(tuple[float, ...], [1.0], immutable=True, depth=0)
    with pytest.raises(ValueError, match="tuple length"):
        writer._annotation(tuple[int, str], [2], immutable=False, depth=0)
    with pytest.raises(ValueError, match="component bound"):
        writer._annotation(tuple[float, ...], [1.0] * 8193, immutable=False, depth=0)


@pytest.mark.parametrize(
    "annotation,value", ((Literal[1], True), (Literal[True], 1), (Literal[False], 0))
)
def test_literal_family_is_not_coerced(annotation: object, value: object) -> None:
    with pytest.raises(ValueError, match="Exact declared writer literal"):
        writer._annotation(annotation, value, immutable=False, depth=0)


def test_foreign_primitive_type_does_not_execute_callbacks() -> None:
    class ForeignMeta(type):
        def __hash__(cls) -> int:
            raise AssertionError("Foreign class hashing must not run")

        def __eq__(cls, other: object) -> bool:
            raise AssertionError("Foreign class equality must not run")

    class ForeignFloat(float, metaclass=ForeignMeta):
        def __float__(self) -> float:
            raise AssertionError("No primitive coercion")

    with pytest.raises(ValueError, match="Exact declared writer primitive family"):
        writer._annotation(float, ForeignFloat(1.0), immutable=True, depth=0)


def test_foreign_sequence_does_not_execute_iteration() -> None:
    class ForeignTuple(tuple[Any, ...]):
        def __iter__(self) -> NoReturn:
            raise AssertionError("Foreign iteration must not run")

    with pytest.raises(ValueError, match="Exact immutable writer sequence"):
        writer._annotation(tuple[float, ...], ForeignTuple((1.0,)), immutable=False, depth=0)


def test_raw_and_owned_declared_models_keep_the_same_rules() -> None:
    class Declared(BaseModel):
        value: float

    class Foreign(Declared):
        pass

    assert writer._annotation(Declared, {"value": 1.0}, immutable=False, depth=0) == {"value": 1.0}
    assert writer._annotation(Declared, Declared(value=1.0), immutable=True, depth=0) == {
        "value": 1.0
    }
    with pytest.raises(ValueError, match="exact immutable model family"):
        writer._annotation(Declared, Foreign(value=1.0), immutable=True, depth=0)
