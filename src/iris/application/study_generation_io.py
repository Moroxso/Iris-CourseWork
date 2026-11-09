"""Canonical bounded private generation storage, with no-clobber publication."""

import json
from dataclasses import dataclass
from pathlib import Path

from iris.application.errors import IrisStudyGenerationError
from iris.application.study_dataset_io import (
    _explicit_path,
    _install_no_clobber,
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/study-generation/v1/payload"


@dataclass(frozen=True, slots=True)
class IrisStudyGenerationWriteResultV1:
    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def validate_study_generation_v1(generation: IrisStudyGenerationV1) -> IrisStudyGenerationV1:
    """Own and check schema/hash/budget only; does not recheck request dependencies."""
    try:
        if type(generation) is not IrisStudyGenerationV1:
            raise TypeError("generation must be exactly IrisStudyGenerationV1")
        owned = IrisStudyGenerationV1.model_validate(generation.model_dump(mode="python"))
        if owned.payload_sha256 != domain_sha256(
            _PAYLOAD_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
        ):
            raise ValueError("generation payload hash is inconsistent")
        if (
            len(canonical_json_file_bytes(owned.model_dump(mode="json")))
            > owned.payload.max_artifact_bytes
        ):
            raise ValueError("generation artifact exceeds its explicit byte budget")
        return owned
    except Exception:
        raise IrisStudyGenerationError("Iris study generation integrity check failed") from None


def write_study_generation_v1(
    generation: IrisStudyGenerationV1, *, path: str | Path
) -> IrisStudyGenerationWriteResultV1:
    try:
        destination = _explicit_path(path)
        owned = validate_study_generation_v1(generation)
        content = canonical_json_file_bytes(owned.model_dump(mode="json"))
        reused = _install_no_clobber(destination, content)
        return IrisStudyGenerationWriteResultV1(
            destination, owned.payload_sha256, len(content), reused
        )
    except Exception:
        raise IrisStudyGenerationError("Iris study generation could not be persisted") from None


def read_study_generation_v1(path: str | Path) -> IrisStudyGenerationV1:
    """Read private answers without executing them; verify schema/hash only."""
    try:
        content = _read_regular_bytes(_explicit_path(path))
        decoded = content.decode("utf-8", errors="strict")
        raw = json.loads(
            decoded, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite
        )
        if canonical_json_file_bytes(raw) != content:
            raise ValueError("generation is not canonical JSON")
        owned = validate_study_generation_v1(IrisStudyGenerationV1.model_validate_json(decoded))
        if canonical_json_file_bytes(owned.model_dump(mode="json")) != content:
            raise ValueError("generation canonical round trip changed bytes")
        return owned
    except Exception:
        raise IrisStudyGenerationError("Iris study generation could not be read") from None


__all__ = [
    "IrisStudyGenerationWriteResultV1",
    "read_study_generation_v1",
    "validate_study_generation_v1",
    "write_study_generation_v1",
]
