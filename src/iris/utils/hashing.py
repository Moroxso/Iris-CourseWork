"""Canonical byte fingerprints for Iris provenance records."""

import json
from hashlib import sha256
from typing import Any

import torch
from torch import nn


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value with the Iris canonical UTF-8 representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def canonical_json_file_bytes(value: Any) -> bytes:
    """Return one canonical JSON value followed by exactly one LF byte."""
    return canonical_json_bytes(value) + b"\n"


def bytes_sha256(value: bytes) -> str:
    """Return the lower-case SHA-256 identity of exact bytes."""
    return sha256(value).hexdigest()


def domain_sha256(domain: str, value: bytes) -> str:
    """Fingerprint bytes in one explicit, unambiguous Iris domain."""
    if type(domain) is not str or not domain or "\0" in domain:
        message = "hash domain must be a nonempty exact string without NUL"
        raise ValueError(message)
    hasher = sha256()
    _update_framed(hasher, domain.encode())
    _update_framed(hasher, value)
    return hasher.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if type(tensor) is not torch.Tensor:
        message = "fingerprinted values must be exact torch.Tensor instances"
        raise TypeError(message)
    if tensor.layout != torch.strided:
        message = "fingerprinted tensors must use the strided layout"
        raise ValueError(message)
    with torch.inference_mode():
        contiguous_bytes = tensor.detach().to(device="cpu").contiguous().view(torch.uint8)
        return contiguous_bytes.numpy().tobytes()


def _update_framed(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def tensor_sha256(
    tensor: torch.Tensor,
    *,
    domain: str = "iris/tensor/v1",
) -> str:
    """Fingerprint dtype, shape, and exact row-major tensor bytes."""
    if type(tensor) is not torch.Tensor:
        message = "fingerprinted values must be exact torch.Tensor instances"
        raise TypeError(message)
    if type(domain) is not str or not domain or "\0" in domain:
        message = "tensor hash domain must be a nonempty exact string without NUL"
        raise ValueError(message)
    hasher = sha256()
    _update_framed(hasher, domain.encode())
    metadata = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
    )
    _update_framed(hasher, metadata)
    _update_framed(hasher, _tensor_bytes(tensor))
    return hasher.hexdigest()


def module_state_sha256(modules: tuple[tuple[str, nn.Module], ...]) -> str:
    """Fingerprint an ordered, named sidecar module state without private access."""
    if type(modules) is not tuple or not modules:
        message = "module fingerprinting requires a nonempty ordered tuple"
        raise TypeError(message)
    names: list[str] = []
    hasher = sha256()
    _update_framed(hasher, b"iris/sidecar-state/v1")
    for component_name, module in modules:
        if type(component_name) is not str or not component_name:
            message = "module fingerprint names must be nonempty exact strings"
            raise TypeError(message)
        if component_name in names:
            message = "module fingerprint names must be unique"
            raise ValueError(message)
        if not isinstance(module, nn.Module):
            message = f"{component_name} must be a torch.nn.Module"
            raise TypeError(message)
        names.append(component_name)
        _update_framed(hasher, component_name.encode("utf-8"))
        state = module.state_dict()
        for state_name in sorted(state):
            tensor = state[state_name]
            qualified_name = f"{component_name}.{state_name}".encode()
            _update_framed(hasher, qualified_name)
            _update_framed(
                hasher,
                canonical_json_bytes(
                    {
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "shape": list(tensor.shape),
                    }
                ),
            )
            _update_framed(hasher, _tensor_bytes(tensor))
    return hasher.hexdigest()


def named_module_state_sha256(
    *,
    component_names: tuple[str, ...],
    state: dict[str, torch.Tensor],
) -> str:
    """Fingerprint qualified tensor state under an ordered module topology."""
    if (
        type(component_names) is not tuple
        or not component_names
        or any(type(name) is not str or not name or "\0" in name for name in component_names)
    ):
        message = "named module fingerprinting requires nonempty exact component names"
        raise TypeError(message)
    if len(set(component_names)) != len(component_names):
        message = "named module fingerprint component names must be unique"
        raise ValueError(message)
    if type(state) is not dict:
        message = "named module fingerprint state must be an exact dict"
        raise TypeError(message)

    grouped: dict[str, dict[str, torch.Tensor]] = {name: {} for name in component_names}
    for qualified_name, tensor in state.items():
        if type(qualified_name) is not str or not qualified_name or "\0" in qualified_name:
            message = "named module fingerprint state names must be exact nonempty strings"
            raise TypeError(message)
        component_name, separator, state_name = qualified_name.partition(".")
        if not separator or not state_name or component_name not in grouped:
            message = f"state {qualified_name!r} is outside the named module topology"
            raise ValueError(message)
        if type(tensor) is not torch.Tensor:
            message = f"state {qualified_name!r} must be an exact torch.Tensor"
            raise TypeError(message)
        grouped[component_name][state_name] = tensor

    hasher = sha256()
    _update_framed(hasher, b"iris/sidecar-state/v1")
    for component_name in component_names:
        _update_framed(hasher, component_name.encode("utf-8"))
        component_state = grouped[component_name]
        for state_name in sorted(component_state):
            tensor = component_state[state_name]
            qualified_bytes = f"{component_name}.{state_name}".encode()
            _update_framed(hasher, qualified_bytes)
            _update_framed(
                hasher,
                canonical_json_bytes(
                    {
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "shape": list(tensor.shape),
                    }
                ),
            )
            _update_framed(hasher, _tensor_bytes(tensor))
    return hasher.hexdigest()
