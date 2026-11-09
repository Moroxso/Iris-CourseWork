"""Exact declared source types must deny foreign callbacks BEFORE serialization."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic._internal._model_construction import ModelMetaclass

from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    _json_types,
    _model_types,
    _own,
)


class Child(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, extra="forbid", revalidate_instances="always"
    )
    value: int


class Root(Child):
    child: Child


type Children = tuple[Child, ...]


class AliasRoot(Child):
    children: Children


@pytest.mark.parametrize(
    "hook", ("__dict__", "__pydantic_extra__", "__pydantic_private__", "model_dump")
)
def test_nested_foreign_model_is_denied_before_its_attributes(hook: str) -> None:
    calls: list[str] = []

    class Foreign(Child):
        def __getattribute__(self, name: str) -> Any:
            if name == hook:
                calls.append(name)
                raise RuntimeError("FOREIGN_HOOK_CALLED")
            return super().__getattribute__(name)

    foreign = Foreign.model_construct(value=1)
    value = Root.model_construct(value=0, child=foreign)
    calls.clear()
    with pytest.raises(ValueError):
        _own(value, Root)
    assert not calls


def test_root_subclass_is_denied_without_attribute_inspection() -> None:
    class Foreign(Root):
        def __getattribute__(self, name: str) -> Any:
            pytest.fail("Foreign root attributes must never execute")

    value = Foreign.model_construct(value=0, child=Child(value=1))
    with pytest.raises(ValueError):
        _own(value, Root)


def test_declared_pep695_tuple_alias_is_resolved_and_owned() -> None:
    value = AliasRoot(value=0, children=(Child(value=1), Child(value=2)))
    owned = _own(value, AliasRoot)
    assert owned == value and owned is not value
    assert all(a is not b for a, b in zip(owned.children, value.children, strict=True))
    assert _model_types(AliasRoot) == frozenset((AliasRoot, Child))


def test_foreign_tuple_does_not_execute_iteration() -> None:
    class ForeignTuple(tuple[Child, ...]):
        def __iter__(self) -> Any:
            pytest.fail("Foreign tuple iteration must never execute")

    value = AliasRoot.model_construct(value=0, children=ForeignTuple((Child(value=1),)))
    with pytest.raises(ValueError):
        _own(value, AliasRoot)


def test_unrelated_model_cannot_supply_its_own_schema_or_dump() -> None:
    class Foreign(BaseModel):
        value: int

        def __getattribute__(self, name: str) -> Any:
            pytest.fail("Unrelated source model must never be inspected")

    value = Root.model_construct(value=0, child=Foreign.model_construct(value=1))
    with pytest.raises(ValueError):
        _own(value, Root)


@pytest.mark.parametrize("hook", ("__hash__", "__eq__"))
def test_nested_foreign_metaclass_never_executes_type_membership(hook: str) -> None:
    armed = False

    class ForeignMeta(ModelMetaclass):
        def __hash__(cls) -> int:
            if armed and hook == "__hash__":
                pytest.fail("Foreign metaclass hash must not execute")
            return type.__hash__(cls)

        def __eq__(cls, other: object) -> bool:
            if armed and hook == "__eq__":
                pytest.fail("Foreign metaclass equality must not execute")
            return cls is other

    class Foreign(Child, metaclass=ForeignMeta):
        pass

    value = Root.model_construct(value=0, child=Foreign.model_construct(value=1))
    armed = True
    with pytest.raises(ValueError):
        _own(value, Root)


def test_foreign_model_dictionary_never_executes_container_methods() -> None:
    class ForeignDict(dict[str, object]):
        def __iter__(self) -> Any:
            pytest.fail("Foreign model dictionary must not execute iteration")

        def items(self) -> Any:
            pytest.fail("Foreign model dictionary must not execute items")

    value = Root(value=0, child=Child(value=1))
    object.__setattr__(value, "__dict__", ForeignDict(vars(value)))
    with pytest.raises(ValueError):
        _own(value, Root)


def test_foreign_model_field_key_never_executes_equality() -> None:
    armed = False

    class ForeignKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            if armed:
                pytest.fail("Foreign field-key equality must not execute")
            return str.__eq__(self, other)

    value = Root(value=0, child=Child(value=1))
    object.__setattr__(value, "__dict__", {ForeignKey("value"): 0, "child": value.child})
    armed = True
    with pytest.raises(ValueError):
        _own(value, Root)


def test_foreign_json_metaclass_never_executes_primitive_type_equality() -> None:
    class ForeignMeta(type):
        def __eq__(cls, other: object) -> bool:
            pytest.fail("Foreign JSON metaclass equality must not execute")

    class Foreign(metaclass=ForeignMeta):
        pass

    with pytest.raises(ValueError):
        _json_types({"leaf": Foreign()})
