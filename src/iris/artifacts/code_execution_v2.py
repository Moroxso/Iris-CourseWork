"""Explicit section-15 resource policy; immutable engineering-v1 is separate."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PythonSandboxPolicyV2(BaseModel):
    """One CPU, 512 MB (512000000 bytes), 32 processes and 5 seconds.

    The named profile also fixes the existing pinned image/runtime and isolation
    settings, with an empty candidate environment. Output capture is bounded, not
    attestation or permission to use arbitrary host/container configurations.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    profile: Literal["iris_wsl_gvisor_python312_source_v2"]
    wall_time_ms: int = Field(strict=True, ge=5000, le=5000)
    max_output_bytes: int = Field(strict=True, ge=1024, le=8 * 1024 * 1024)


__all__ = ["PythonSandboxPolicyV2"]
