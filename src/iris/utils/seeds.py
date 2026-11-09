"""Transactional process-global runtime policy for Iris startup."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class RuntimeStateSnapshot:
    """Process-global state owned by one exclusive bootstrap transaction."""

    python_rng_state: tuple[Any, ...]
    cpu_rng_state: torch.Tensor
    cuda_device: torch.device | None
    cuda_rng_state: torch.Tensor | None
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


def capture_runtime_state(device: torch.device) -> RuntimeStateSnapshot:
    """Capture the exact global state that bootstrap is allowed to mutate."""
    cuda_device = device if device.type == "cuda" else None
    cuda_rng_state = None
    if cuda_device is not None:
        cuda_rng_state = torch.cuda.get_rng_state(cuda_device).detach().clone()
    return RuntimeStateSnapshot(
        python_rng_state=random.getstate(),
        cpu_rng_state=torch.get_rng_state().detach().clone(),
        cuda_device=cuda_device,
        cuda_rng_state=cuda_rng_state,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )


def apply_runtime_policy(
    *,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> None:
    """Commit an explicit seed and deterministic policy to owned globals."""
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False

    random.seed(seed, version=2)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state().detach().clone())
    if device.type == "cuda":
        cuda_generator = torch.Generator(device=device)
        cuda_generator.manual_seed(seed)
        torch.cuda.set_rng_state(
            cuda_generator.get_state().detach().clone(),
            device=device,
        )


def restore_runtime_state(snapshot: RuntimeStateSnapshot) -> None:
    """Restore every owned global and report all restoration failures."""
    failures: list[BaseException] = []

    operations: list[Callable[[], None]] = [
        lambda: random.setstate(snapshot.python_rng_state),
        lambda: torch.set_rng_state(snapshot.cpu_rng_state.detach().clone()),
        lambda: torch.use_deterministic_algorithms(
            snapshot.deterministic_algorithms,
            warn_only=snapshot.deterministic_warn_only,
        ),
        lambda: setattr(
            torch.backends.cudnn,
            "deterministic",
            snapshot.cudnn_deterministic,
        ),
        lambda: setattr(
            torch.backends.cudnn,
            "benchmark",
            snapshot.cudnn_benchmark,
        ),
    ]
    cuda_device = snapshot.cuda_device
    cuda_rng_state = snapshot.cuda_rng_state
    if cuda_device is not None and cuda_rng_state is not None:

        def restore_cuda_rng_state() -> None:
            torch.cuda.set_rng_state(
                cuda_rng_state.detach().clone(),
                device=cuda_device,
            )

        operations.insert(
            2,
            restore_cuda_rng_state,
        )

    for operation in operations:
        try:
            operation()
        except BaseException as error:
            failures.append(error)
    if failures:
        message = "Failed to restore one or more Iris runtime globals"
        raise BaseExceptionGroup(message, failures)
