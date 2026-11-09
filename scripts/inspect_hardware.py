"""Inspect the explicitly requested hardware backend."""

import argparse

import torch

from iris.utils.devices import inspect_device


def parse_args() -> argparse.Namespace:
    """Parse an explicit device request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    """Print capabilities of the requested compute device."""
    args = parse_args()
    info = inspect_device(args.device)
    total_memory = (
        "not reported" if info.total_memory_bytes is None else str(info.total_memory_bytes)
    )

    print(f"Requested device: {args.device}")
    print(f"Resolved backend: {info.backend}")
    print(f"Device name: {info.name}")
    print(f"Total memory: {total_memory}")
    print(f"FP16 support: {str(info.supports_fp16).lower()}")
    print(f"BF16 support: {str(info.supports_bf16).lower()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'not available'}")


if __name__ == "__main__":
    main()
