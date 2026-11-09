"""Explicit device inspection and precision resolution."""

import platform
from dataclasses import dataclass
from typing import Literal

import torch

type DeviceBackend = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Capabilities of one explicitly selected compute device."""

    device: torch.device
    backend: DeviceBackend
    name: str
    total_memory_bytes: int | None
    supports_fp16: bool
    supports_bf16: bool


def inspect_device(requested_device: str) -> DeviceInfo:
    """Inspect a requested device without falling back to another backend."""
    if requested_device == "cpu":
        return DeviceInfo(
            device=torch.device("cpu"),
            backend="cpu",
            name=platform.processor().strip() or "CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        )

    if requested_device != "cuda":
        message = f"Unsupported device {requested_device!r}; expected 'cpu' or 'cuda'"
        raise ValueError(message)

    if not torch.cuda.is_available():
        message = "CUDA was explicitly requested, but it is not available"
        raise RuntimeError(message)

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    return DeviceInfo(
        device=device,
        backend="cuda",
        name=torch.cuda.get_device_name(device),
        total_memory_bytes=properties.total_memory,
        supports_fp16=True,
        supports_bf16=torch.cuda.is_bf16_supported(),
    )


def resolve_torch_dtype(device_info: DeviceInfo, precision: str) -> torch.dtype:
    """Resolve a configured precision after checking device capabilities."""
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        if device_info.supports_fp16:
            return torch.float16
        message = f"Device {device_info.name!r} does not support fp16"
        raise ValueError(message)
    if precision == "bf16":
        if device_info.supports_bf16:
            return torch.bfloat16
        message = f"Device {device_info.name!r} does not support bf16"
        raise ValueError(message)

    message = f"Unsupported precision {precision!r}; expected 'fp32', 'fp16', or 'bf16'"
    raise ValueError(message)
