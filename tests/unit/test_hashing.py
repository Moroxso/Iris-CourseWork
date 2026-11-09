"""Unit tests for canonical Iris provenance fingerprints."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pytest
import torch
from torch import nn

from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    module_state_sha256,
    tensor_sha256,
)


class _StateModule(nn.Module):
    counter: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.register_buffer("counter", torch.tensor([5], dtype=torch.int64))


class _StringSubclass(str):
    pass


class _TupleSubclass(tuple[Any, ...]):
    pass


class _RegistrationOrderModule(nn.Module):
    def __init__(self, *, reverse: bool) -> None:
        super().__init__()
        first = nn.Parameter(torch.tensor([1.0]))
        second = nn.Parameter(torch.tensor([2.0]))
        names_and_parameters = (
            (("second", second), ("first", first))
            if reverse
            else (
                ("first", first),
                ("second", second),
            )
        )
        for name, parameter in names_and_parameters:
            self.register_parameter(name, parameter)


def _framed(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big", signed=False) + value


def test_canonical_json_bytes_are_sorted_compact_and_utf8() -> None:
    value = {"z": "Привет", "a": [True, None, 2], "nested": {"b": 1, "a": 0}}

    encoded = canonical_json_bytes(value)

    assert encoded == ('{"a":[true,null,2],"nested":{"a":0,"b":1},"z":"Привет"}'.encode())
    assert not encoded.endswith(b"\n")


def test_canonical_json_file_bytes_append_exactly_one_lf() -> None:
    value = {"line": "already ends with an escaped newline\n"}
    canonical = canonical_json_bytes(value)

    file_bytes = canonical_json_file_bytes(value)

    assert file_bytes == canonical + b"\n"
    assert file_bytes[-1:] == b"\n"
    assert file_bytes[-2:] != b"\n\n"


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_bytes({"value": float("nan")})


def test_bytes_sha256_hashes_exact_bytes() -> None:
    assert (
        bytes_sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_domain_sha256_uses_length_prefixed_domain_and_payload() -> None:
    domain = "iris/данные/v1"
    value = b"payload\x00bytes"
    domain_bytes = domain.encode()
    expected = sha256(_framed(domain_bytes) + _framed(value)).hexdigest()

    assert domain_sha256(domain, value) == expected


def test_domain_sha256_separates_domains_and_ambiguous_concatenations() -> None:
    assert domain_sha256("iris/a", b"payload") != domain_sha256("iris/b", b"payload")
    assert domain_sha256("ab", b"c") != domain_sha256("a", b"bc")


@pytest.mark.parametrize("domain", ["", "iris\x00tensor", 1, _StringSubclass("iris/tensor/v1")])
def test_domain_sha256_rejects_invalid_domains(domain: Any) -> None:
    with pytest.raises(ValueError, match="nonempty exact string without NUL"):
        domain_sha256(domain, b"payload")


def test_tensor_sha256_is_deterministic_and_domain_separated() -> None:
    tensor = torch.tensor([[1.0, -2.0], [3.5, 0.0]], dtype=torch.float32)

    first = tensor_sha256(tensor)

    assert first == tensor_sha256(tensor.clone())
    assert first != tensor_sha256(tensor, domain="iris/tensor/other-v1")


def test_tensor_sha256_includes_dtype_shape_and_values() -> None:
    raw = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    same_bytes_other_dtype = raw.view(torch.float32)
    same_bytes_other_shape = raw.reshape(2, 2)
    changed_value = raw.clone()
    changed_value[-1] = 4

    original = tensor_sha256(raw)

    assert original != tensor_sha256(same_bytes_other_dtype)
    assert original != tensor_sha256(same_bytes_other_shape)
    assert original != tensor_sha256(changed_value)


def test_tensor_sha256_matches_noncontiguous_and_contiguous_logical_values() -> None:
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not noncontiguous.is_contiguous()

    assert tensor_sha256(noncontiguous) == tensor_sha256(noncontiguous.contiguous())


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_tensor_sha256_supports_canonical_numeric_and_boolean_dtypes(
    dtype: torch.dtype,
) -> None:
    tensor = torch.tensor([0, 1], dtype=dtype)

    digest = tensor_sha256(tensor)

    assert len(digest) == 64
    assert digest == digest.lower()
    assert digest == tensor_sha256(tensor.clone())


def test_tensor_sha256_rejects_sparse_tensor() -> None:
    sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [1]]),
        values=torch.tensor([2.0]),
        size=(2, 2),
        check_invariants=True,
    )

    with pytest.raises(ValueError, match="strided layout"):
        tensor_sha256(sparse)


def test_tensor_sha256_rejects_meta_tensor() -> None:
    with pytest.raises(NotImplementedError):
        tensor_sha256(torch.empty((2, 3), device="meta"))


def test_tensor_sha256_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match=r"exact torch.Tensor"):
        tensor_sha256(object())  # type: ignore[arg-type]


def test_tensor_sha256_rejects_tensor_subclasses() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(TypeError, match=r"exact torch.Tensor"):
        tensor_sha256(parameter)


def test_module_state_sha256_is_deterministic() -> None:
    module = _StateModule()
    equivalent = _StateModule()

    digest = module_state_sha256((("reader", module),))

    assert digest == module_state_sha256((("reader", module),))
    assert digest == module_state_sha256((("reader", equivalent),))


def test_module_state_sha256_changes_after_parameter_or_buffer_mutation() -> None:
    module = _StateModule()
    original = module_state_sha256((("reader", module),))

    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    after_parameter = module_state_sha256((("reader", module),))
    module.counter.add_(1)
    after_buffer = module_state_sha256((("reader", module),))

    assert after_parameter != original
    assert after_buffer != after_parameter


def test_module_state_sha256_includes_component_names() -> None:
    module = _StateModule()

    assert module_state_sha256((("reader", module),)) != module_state_sha256((("writer", module),))


def test_module_state_sha256_preserves_component_order() -> None:
    first = _StateModule()
    second = _StateModule()
    with torch.no_grad():
        second.weight.add_(10.0)

    forward = module_state_sha256((("first", first), ("second", second)))
    reversed_order = module_state_sha256((("second", second), ("first", first)))

    assert forward != reversed_order


def test_module_state_sha256_sorts_state_names() -> None:
    normal = _RegistrationOrderModule(reverse=False)
    reversed_registration = _RegistrationOrderModule(reverse=True)

    assert module_state_sha256((("component", normal),)) == module_state_sha256(
        (("component", reversed_registration),)
    )


@pytest.mark.parametrize("modules", [(), [], _TupleSubclass((("reader", _StateModule()),))])
def test_module_state_sha256_requires_nonempty_exact_tuple(modules: Any) -> None:
    with pytest.raises(TypeError, match="nonempty ordered tuple"):
        module_state_sha256(modules)


@pytest.mark.parametrize("name", ["", 1, _StringSubclass("reader")])
def test_module_state_sha256_rejects_invalid_component_names(name: Any) -> None:
    with pytest.raises(TypeError, match="nonempty exact strings"):
        module_state_sha256(((name, _StateModule()),))


def test_module_state_sha256_rejects_duplicate_component_names() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        module_state_sha256((("reader", _StateModule()), ("reader", _StateModule())))


def test_module_state_sha256_rejects_non_module_values() -> None:
    with pytest.raises(TypeError, match=r"reader must be a torch.nn.Module"):
        module_state_sha256((("reader", object()),))  # type: ignore[arg-type]
