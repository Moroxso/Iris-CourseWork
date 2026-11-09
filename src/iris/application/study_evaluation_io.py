"""Canonical no-clobber evaluation storage; dependency verification is a separate API."""

import json
from dataclasses import dataclass
from pathlib import Path

from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_dataset_io import (
    _explicit_path,
    _install_no_clobber,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.application.study_evaluation import validate_study_evaluation_v1
from iris.artifacts.study_evaluation_v1 import IrisStudyEvaluationV1
from iris.utils.hashing import canonical_json_file_bytes


@dataclass(frozen=True, slots=True)
class IrisStudyEvaluationWriteResultV1:
    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def write_study_evaluation_v1(
    evaluation: IrisStudyEvaluationV1, *, path: str | Path
) -> IrisStudyEvaluationWriteResultV1:
    """Persist metadata only, never overwrite an existing conflicting file."""
    try:
        destination = _explicit_path(path)
        owned = validate_study_evaluation_v1(evaluation)
        content = canonical_json_file_bytes(owned.model_dump(mode="json"))
        reused = _install_no_clobber(destination, content)
        return IrisStudyEvaluationWriteResultV1(
            destination, owned.payload_sha256, len(content), reused
        )
    except Exception:
        raise IrisStudyEvaluationError("Iris study evaluation could not be persisted") from None


def read_study_evaluation_v1(path: str | Path) -> IrisStudyEvaluationV1:
    """Check bounded canonical bytes and envelope integrity, not dependencies or truth."""
    try:
        content = _read_regular_bytes(_explicit_path(path), max_bytes=1024 * 1024)
        decoded = content.decode("utf-8", errors="strict")
        raw = json.loads(
            decoded, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite
        )
        if canonical_json_file_bytes(raw) != content:
            raise ValueError("evaluation is not canonical JSON")
        owned = validate_study_evaluation_v1(IrisStudyEvaluationV1.model_validate_json(decoded))
        if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
            raise ValueError("evaluation changed during canonical round trip")
        return owned
    except Exception:
        raise IrisStudyEvaluationError("Iris study evaluation could not be read") from None


__all__ = [
    "IrisStudyEvaluationWriteResultV1",
    "read_study_evaluation_v1",
    "write_study_evaluation_v1",
]
