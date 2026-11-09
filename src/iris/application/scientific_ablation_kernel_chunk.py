"""Validate saved DATA intent; compare the real prefix inside each kernel lock.

This is neither a journal nor an independent producer ACK. Callers still need
the original native owner (capture) or verified saved corpus/policy (score).
"""

from iris.application.scientific_ablation_inference import _bounded_metadata, _digest, _exact_tree
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)


def checked_kernel_chunk_v1(
    value: ScientificAblationKernelChunkExpectationV1 | None,
    *,
    expected_plan_sha256: str,
    rows_per_call: int | None,
) -> ScientificAblationKernelChunkExpectationV1 | None:
    """Snapshot exact bounded scalars BEFORE inputs, directories or native work."""
    if value is None:
        return None
    _exact_tree(value, ScientificAblationKernelChunkExpectationV1)
    if value.__pydantic_private__ is not None:
        raise ValueError("Hidden chunk metadata forbidden")
    _bounded_metadata(value, maximum=4096)
    checked = ScientificAblationKernelChunkExpectationV1.model_validate(
        value.model_dump(mode="python", warnings=False)
    )
    _digest(expected_plan_sha256)
    if (
        type(rows_per_call) is not int
        or checked.rows_per_call != rows_per_call
        or checked.kernel_plan_sha256 != expected_plan_sha256
    ):
        raise ValueError("Exact explicit saved kernel plan and chunk budget required")
    return checked


def check_kernel_chunk_prefix_v1(
    value: ScientificAblationKernelChunkExpectationV1 | None,
    *,
    plan_sha256: str,
    acknowledgement_head_sha256: str,
    acknowledged_rows: int,
    total_rows: int,
    rows_per_call: int | None,
) -> None:
    """Only call with the actual plan/row ACKs under the ORIGINAL mutation lock."""
    checked = checked_kernel_chunk_v1(
        value, expected_plan_sha256=plan_sha256, rows_per_call=rows_per_call
    )
    if checked is None:
        return
    _digest(acknowledgement_head_sha256)
    if (
        type(acknowledged_rows) is not int
        or type(total_rows) is not int
        or checked.acknowledgement_head_sha256 != acknowledgement_head_sha256
        or checked.acknowledged_rows != acknowledged_rows
        or checked.total_rows != total_rows
    ):
        raise ValueError(
            "Saved chunk prefix differs from actual locked kernel row acknowledgements"
        )


__all__ = ["check_kernel_chunk_prefix_v1", "checked_kernel_chunk_v1"]
