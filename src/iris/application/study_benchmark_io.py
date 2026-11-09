"""Canonical benchmark storage, sharing the existing bounded local artifact primitives."""

import json
from dataclasses import dataclass
from pathlib import Path

from iris.application.errors import IrisStudyBenchmarkError
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_dataset_io import (
    _explicit_path,
    _install_no_clobber,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.utils.hashing import canonical_json_file_bytes


@dataclass(frozen=True, slots=True)
class IrisStudyBenchmarkWriteResultV1:
    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def write_study_benchmark_v1(
    benchmark: IrisStudyBenchmarkV1, *, path: str | Path
) -> IrisStudyBenchmarkWriteResultV1:
    """Write private inputs, targets, and hidden code cases without overwriting a file."""
    try:
        output_path = _explicit_path(path)
        owned = validate_study_benchmark_v1(benchmark)
        content = canonical_json_file_bytes(owned.model_dump(mode="json"))
        reused = _install_no_clobber(output_path, content)
        return IrisStudyBenchmarkWriteResultV1(
            path=output_path,
            payload_sha256=owned.payload_sha256,
            size_bytes=len(content),
            reused_existing=reused,
        )
    except Exception:
        raise IrisStudyBenchmarkError("Iris study benchmark could not be persisted") from None


def read_study_benchmark_v1(path: str | Path) -> IrisStudyBenchmarkV1:
    """Read bounded canonical JSON; do not execute code or infer launch readiness."""
    try:
        content = _read_regular_bytes(_explicit_path(path))
        decoded = content.decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        if canonical_json_file_bytes(raw) != content:
            raise ValueError("benchmark bytes are not canonical iris-json-v1")
        owned = validate_study_benchmark_v1(IrisStudyBenchmarkV1.model_validate_json(decoded))
        if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
            raise ValueError("benchmark changed during canonical round trip")
        return owned
    except Exception:
        raise IrisStudyBenchmarkError(
            "Iris study benchmark could not be read or verified"
        ) from None


__all__ = [
    "IrisStudyBenchmarkWriteResultV1",
    "read_study_benchmark_v1",
    "write_study_benchmark_v1",
]
