"""Explicit versioned policy for the first restricted Python runtime profile."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PythonOutcome = Literal["correct", "incorrect", "invalid_output"]
PythonReason = Literal[
    "all_cases_pass",
    "case_mismatch",
    "invalid_source",
    "syntax_error",
    "missing_entrypoint",
    "load_exception",
    "protocol_error",
    "execution_terminated",
    "memory_limit",
    "output_limit",
    "wall_time_limit",
]


class PythonSandboxPolicyV1(BaseModel):
    """Caller-selected engineering policy, not an inferred scientific budget.

    The named immutable profile fixes image/runtime, isolation and resource
    settings. A different profile requires a new reviewed implementation.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    profile: Literal["iris_wsl_gvisor_python312_restricted_v1"]
    wall_time_ms: int = Field(strict=True, ge=1, le=600_000)
    max_output_bytes: int = Field(strict=True, ge=1024, le=8 * 1024 * 1024)


__all__ = ["PythonOutcome", "PythonReason", "PythonSandboxPolicyV1"]
