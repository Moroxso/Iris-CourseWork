"""Exact saved chunk intent; DATA only, not ownership or durable completion ACK."""

from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict


class ScientificAblationKernelChunkExpectationV1(_Strict):
    """Bind one exact prefix and interval; never choose a newer moving head.

    Retrying after a committed row changes this prefix and must fail, not silently
    compute another chunk. An unacknowledged stage is still charged/preserved by
    the original resumable kernel; this schema does not promise compute-once.
    """

    kind: Literal["iris-scientific-ablation-kernel-chunk-expectation-v1"] = (
        "iris-scientific-ablation-kernel-chunk-expectation-v1"
    )
    kernel_plan_sha256: Digest
    acknowledgement_head_sha256: Digest
    acknowledged_rows: int = Field(ge=0, le=5000)
    total_rows: int = Field(ge=1, le=5000)
    rows_per_call: int = Field(ge=1, le=5000)
    stop_row_index: int = Field(ge=0, le=5000)

    @model_validator(mode="after")
    def exact_interval(self) -> Self:
        if self.acknowledged_rows > self.total_rows or self.stop_row_index != min(
            self.total_rows, self.acknowledged_rows + self.rows_per_call
        ):
            raise ValueError("Saved kernel chunk must bind its exact prefix and stop index")
        return self


__all__ = ["ScientificAblationKernelChunkExpectationV1"]
