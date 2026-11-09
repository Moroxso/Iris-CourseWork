import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from iris.utils.devices import DeviceInfo, inspect_device, resolve_torch_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_inspect_cpu_reports_conservative_capabilities() -> None:
    info = inspect_device("cpu")

    assert info.device == torch.device("cpu")
    assert info.backend == "cpu"
    assert info.name
    assert info.total_memory_bytes is None
    assert info.supports_fp16 is False
    assert info.supports_bf16 is False


def test_cpu_resolves_only_fp32() -> None:
    info = inspect_device("cpu")

    assert resolve_torch_dtype(info, "fp32") is torch.float32
    with pytest.raises(ValueError, match="does not support fp16"):
        resolve_torch_dtype(info, "fp16")
    with pytest.raises(ValueError, match="does not support bf16"):
        resolve_torch_dtype(info, "bf16")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        inspect_device("auto")


def test_unavailable_cuda_does_not_fall_back_to_cpu() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="explicitly requested"),
    ):
        inspect_device("cuda")


def test_available_cuda_reports_runtime_capabilities() -> None:
    properties = SimpleNamespace(total_memory=8 * 1024**3)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=1),
        patch("torch.cuda.get_device_properties", return_value=properties) as get_properties,
        patch("torch.cuda.get_device_name", return_value="Test GPU") as get_name,
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        info = inspect_device("cuda")

    assert info.device == torch.device("cuda", 1)
    assert info.backend == "cuda"
    assert info.name == "Test GPU"
    assert info.total_memory_bytes == 8 * 1024**3
    assert info.supports_fp16 is True
    assert info.supports_bf16 is True
    get_properties.assert_called_once_with(torch.device("cuda", 1))
    get_name.assert_called_once_with(torch.device("cuda", 1))


def test_supported_cuda_precisions_resolve_to_torch_dtypes() -> None:
    info = DeviceInfo(
        device=torch.device("cuda"),
        backend="cuda",
        name="Test GPU",
        total_memory_bytes=1024,
        supports_fp16=True,
        supports_bf16=True,
    )

    assert resolve_torch_dtype(info, "fp16") is torch.float16
    assert resolve_torch_dtype(info, "bf16") is torch.bfloat16


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_torch_dtype(inspect_device("cpu"), "int8")


def test_inspect_hardware_script_reports_cpu_capabilities() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/inspect_hardware.py", "--device", "cpu"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Requested device: cpu" in result.stdout
    assert "Resolved backend: cpu" in result.stdout
    assert "Device name:" in result.stdout
    assert "Total memory: not reported" in result.stdout
    assert "FP16 support: false" in result.stdout
    assert "BF16 support: false" in result.stdout
    assert "PyTorch version:" in result.stdout
    assert "CUDA runtime:" in result.stdout
