"""Observer-only primitive closure regression; no source/model factories."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

path = (
    Path(__file__).resolve().parents[2]
    / "scripts/verify_scientific_ablation_causal_report_layout.py"
)
spec = importlib.util.spec_from_file_location("iris_complete_layout_reader_test", path)
assert spec is not None and spec.loader is not None
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.mark.parametrize(
    "kind,value",
    (
        ("text", "\0" * 256),
        ("digest", "a" * 64),
        ("float", -1.7976931348623157e308),
        ("int", -(2**63)),
        ("bool", False),
    ),
)
def test_original_scalar_family_positive(kind: str, value: Any) -> None:
    assert reader.node_bound(dict(kind="scalar", scalar=kind, limit=256), value, check=True) > 0


@pytest.mark.parametrize(
    "kind,value",
    (
        ("text", "x" * 257),
        ("digest", "A" * 64),
        ("digest", "a" * 63),
        ("float", True),
        ("int", True),
        ("bool", 0),
        ("float", float("inf")),
    ),
)
def test_closed_reader_does_not_confuse_byte_room_with_primitive_domain(
    kind: str, value: Any
) -> None:
    with pytest.raises(ValueError):
        reader.node_bound(dict(kind="scalar", scalar=kind, limit=256), value, check=True)


def test_fixed_byte_count_is_not_a_boolean_alias() -> None:
    with pytest.raises(ValueError):
        reader.node_bound(dict(kind="fixed", bytes=True, encoded="0"))
