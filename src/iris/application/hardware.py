"""Hardware inspection application service."""

from dataclasses import dataclass

import torch

from iris.utils.devices import inspect_device


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """Presentation-neutral report of one hardware inspection."""

    requested_device: str
    resolved_backend: str
    device_name: str
    total_memory_bytes: int | None
    supports_fp16: bool
    supports_bf16: bool
    torch_version: str
    cuda_runtime: str | None


def inspect_hardware(requested_device: str) -> HardwareReport:
    """Inspect one explicitly requested device for a delivery interface."""
    info = inspect_device(requested_device)
    return HardwareReport(
        requested_device=requested_device,
        resolved_backend=info.backend,
        device_name=info.name,
        total_memory_bytes=info.total_memory_bytes,
        supports_fp16=info.supports_fp16,
        supports_bf16=info.supports_bf16,
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
    )
