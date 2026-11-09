"""Encoding/closed-container proofs, not measured causal effects or source identity."""

import sys
from typing import Any

import pytest

from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes


@pytest.mark.parametrize(
    "value",
    (0.0, -0.0, sys.float_info.max, -sys.float_info.max, 5e-324, -5e-324, 1.0000000000000002e-308),
)
def test_finite_binary64_encoding_is_reserved(value: float) -> None:
    layout.F.check(value)
    assert len(canonical_json_bytes(value)) <= layout.F.bound == 32


@pytest.mark.parametrize("value", (-(2**63), -1, 0, 1, 2**63 - 1))
def test_signed_int64_encoding_is_reserved(value: int) -> None:
    layout.N.check(value)
    assert len(canonical_json_bytes(value)) <= layout.N.bound == 32


@pytest.mark.parametrize("text", ("x", '"', "\\", "\0", "\n", "Ж", "😀"))
def test_escaped_utf8_and_supplementary_text_boundary(text: str) -> None:
    value = text * 256
    layout.T.check(value)
    assert len(canonical_json_bytes(value)) <= layout.T.bound == 1538
    with pytest.raises(ValueError):
        layout.T.check(value + text)


@pytest.mark.parametrize(
    "kind,value",
    (
        ("float", float("nan")),
        ("float", float("inf")),
        ("float", float("-inf")),
        ("float", True),
        ("float", 1),
        ("int", True),
        ("int", 1.0),
        ("int", 2**63),
        ("int", -(2**63) - 1),
        ("bool", 1),
        ("digest", "A" * 64),
        ("digest", "0" * 63),
        ("text", object()),
    ),
)
def test_scalar_types_and_bounds_are_exact(kind: Any, value: Any) -> None:
    with pytest.raises(ValueError):
        layout._Scalar(kind).check(value)


def test_primitive_subclasses_and_lone_surrogates_are_not_encodable() -> None:
    class ForeignInt(int):
        pass

    class ForeignStr(str):
        pass

    with pytest.raises(ValueError):
        layout.N.check(ForeignInt(1))
    with pytest.raises(ValueError):
        layout.T.check(ForeignStr("abc"))
    with pytest.raises(UnicodeError):
        layout.T.check("\ud800")


def test_structural_bound_counts_all_escaped_keys_arrays_and_final_lf() -> None:
    shape = layout._Object(
        (
            ('"\0😀', layout._Array(layout.T, 2)),
            ("numbers", layout._Array(layout.N, 3)),
            ("optional", layout._optional(layout.F)),
        )
    )
    value = {
        '"\0😀': ["\0" * 256, "\0" * 256],
        "numbers": [-(2**63)] * 3,
        "optional": -sys.float_info.max,
    }
    shape.check(value)
    assert len(canonical_json_file_bytes(value)) <= shape.bound + 1
    assert shape.bound == (
        2
        + 2
        + sum(len(canonical_json_bytes(name)) + 1 + child.bound for name, child in shape.members)
    )


def test_array_maximum_and_choice_do_not_shrink_or_allocate_geometry() -> None:
    # Descriptor arithmetic only. Do not materialize the billion-element array.
    shape = layout._Array(layout.F, 1_000_000_000)
    assert shape.bound == 2 + 32_000_000_000 + 999_999_999
    with pytest.raises(ValueError):
        shape.check([])
    variable = layout._Array(layout.N, 2, 0)
    for value in ([], [0], [0, 0]):
        variable.check(value)
        assert len(canonical_json_bytes(value)) <= variable.bound
    with pytest.raises(ValueError):
        variable.check([0, 0, 0])
    for optional in (None, 0.0):
        layout._optional(layout.F).check(optional)


def test_closed_objects_reject_hidden_mutable_and_foreign_containers() -> None:
    shape = layout._object(values=layout._Array(layout.N, 1))

    class ForeignDict(dict[str, Any]):
        pass

    class ForeignList(list[int]):
        pass

    for value in (
        {"values": [1], "hidden": 1},
        {"values": (1,)},
        ForeignDict(values=[1]),
        {"values": ForeignList([1])},
    ):
        with pytest.raises(ValueError):
            shape.check(value)
    with pytest.raises(ValueError):
        layout._fixed({"source": [1]}).check(ForeignDict(source=[1]))


def test_layout_descriptor_binds_names_not_only_total_bytes() -> None:
    first = layout._object(first=layout.N)
    other = layout._object(other=layout.N)
    assert first.bound == other.bound
    assert canonical_json_bytes(layout._descriptor(first)) != canonical_json_bytes(
        layout._descriptor(other)
    )
    with pytest.raises(ValueError):
        layout._descriptor(layout._Object((("a", layout.N), ("a", layout.N))))
    with pytest.raises(ValueError):
        layout._descriptor(layout._Array(layout.N, True))
    with pytest.raises(ValueError):
        layout._descriptor(object())


def test_cyclic_descriptor_and_source_models_fail_boundedly() -> None:
    from pydantic import BaseModel

    class CyclicSource(BaseModel):
        child: Any = None

    source = CyclicSource()
    object.__setattr__(source, "child", source)
    with pytest.raises(ValueError, match="depth"):
        layout._closed_models(source)
    shape = layout._object(child=layout.N)
    object.__setattr__(shape, "members", (("child", shape),))
    with pytest.raises(ValueError, match="depth"):
        layout._descriptor(shape)


def test_geometry_extremes_are_arithmetic_not_vector_allocations() -> None:
    state = layout._state(64, 4096)
    trace = layout._trace(64, 64, 4096, 64, 64, full=True)
    assert state.bound > 64 * 4096 * 32
    assert trace.bound > 64 * 2 * (64 + 64) * 4096 * 32
    assert trace.bound > layout.MAX_REPORT_BYTES
    layout._descriptor(trace)


def test_foreign_descriptor_never_calls_its_attributes_or_equality() -> None:
    class Foreign:
        def __getattr__(self, name: str) -> Any:
            pytest.fail("Foreign descriptor attribute must never execute")

        def __eq__(self, other: object) -> bool:
            pytest.fail("Foreign descriptor equality must never execute")

    with pytest.raises(ValueError, match="Foreign"):
        layout._descriptor(Foreign())
