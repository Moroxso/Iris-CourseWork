"""Bounded private publication of complete paired intervention captures."""

import os
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from iris.application.scientific_checkpoint import _bounded_read, _json, _regular
from iris.application.scientific_command import _local_path
from iris.application.scientific_generation import IrisScientificGenerationError
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_intervention_v1 import IrisScientificInterventionV1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-intervention/v1"
_MAX_BYTES = 64 * 1024**2
_FILE = "intervention.json"


def _checked(result: IrisScientificInterventionV1) -> bytes:
    if type(result) is not IrisScientificInterventionV1:
        raise ValueError("Exact paired intervention capture required")
    result = IrisScientificInterventionV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Paired capture identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Paired capture exceeds its byte budget")
    return content


def read_scientific_intervention_v1(path: Path) -> IrisScientificInterventionV1:
    """Verify exact complete private bytes, not the authenticity of outside results."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Paired capture inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificInterventionV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Paired capture identity/canonical bytes differ")
        return result
    except Exception:
        raise IrisScientificGenerationError(
            "Scientific intervention capture could not be read"
        ) from None


def write_scientific_intervention_v1(
    result: IrisScientificInterventionV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish only whole paired captures in a distinct content-addressed store."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded paired capture storage quota required")
        root = _local_path(root) / "scientific-intervention-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_intervention_v1(target) != result:
                raise ValueError("Existing paired capture conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        if (
            occupied + len(content) > maximum_storage_bytes
            or shutil.disk_usage(root).free < len(content) + 1024**3
        ):
            raise ValueError("Paired capture quota/free-space reserve exceeded")
        candidate = root / f".iris-intervention-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged paired capture differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_intervention_v1(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificGenerationError(
            "Scientific intervention capture could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()
