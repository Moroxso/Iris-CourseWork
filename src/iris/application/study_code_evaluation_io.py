"""Atomic private code evidence bundles; base64 is neither encryption nor attestation."""

import json
from dataclasses import dataclass
from pathlib import Path

from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_code_receipt import (
    _MAX_CODE_EVALUATION_BYTES,
    validate_study_code_evaluation_v1,
)
from iris.application.study_dataset_io import (
    _explicit_path,
    _install_no_clobber,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.artifacts.study_code_evaluation_v1 import IrisStudyCodeEvaluationV1
from iris.utils.hashing import canonical_json_file_bytes


@dataclass(frozen=True, slots=True)
class IrisStudyCodeEvaluationWriteResultV1:
    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def write_study_code_evaluation_v1(
    evaluation: IrisStudyCodeEvaluationV1, *, path: str | Path
) -> IrisStudyCodeEvaluationWriteResultV1:
    """Explicitly persist private observations and receipt together without overwrite."""
    try:
        destination = _explicit_path(path)
        owned = validate_study_code_evaluation_v1(evaluation)
        content = canonical_json_file_bytes(owned.model_dump(mode="json"))
        reused = _install_no_clobber(destination, content)
        return IrisStudyCodeEvaluationWriteResultV1(
            destination, owned.payload_sha256, len(content), reused
        )
    except Exception:
        raise IrisStudyEvaluationError("Iris code evaluation could not be persisted") from None


def read_study_code_evaluation_v1(path: str | Path) -> IrisStudyCodeEvaluationV1:
    """Read bounded canonical data only; never verify dependencies or execute a candidate."""
    try:
        content = _read_regular_bytes(_explicit_path(path), max_bytes=_MAX_CODE_EVALUATION_BYTES)
        decoded = content.decode("utf-8", errors="strict")
        raw = json.loads(
            decoded, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite
        )
        if canonical_json_file_bytes(raw) != content:
            raise ValueError("code evaluation is not canonical JSON")
        owned = validate_study_code_evaluation_v1(
            IrisStudyCodeEvaluationV1.model_validate_json(decoded)
        )
        if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
            raise ValueError("code evaluation changed during canonical round trip")
        return owned
    except Exception:
        raise IrisStudyEvaluationError("Iris code evaluation could not be read") from None


__all__ = [
    "IrisStudyCodeEvaluationWriteResultV1",
    "read_study_code_evaluation_v1",
    "write_study_code_evaluation_v1",
]
