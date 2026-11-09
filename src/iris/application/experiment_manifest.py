"""Canonical local example manifests and deterministic split iteration."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch

from iris.application.errors import (
    IrisExperimentIterationError,
    IrisExperimentManifestError,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.experiment_v1 import (
    ExperimentExampleV1,
    ExperimentManifestPayloadV1,
    ExperimentManifestSchemaV1,
    ExperimentSplitCountsV1,
    ExperimentSplitV1,
    IrisExperimentManifestV1,
)
from iris.backbone import BackboneAdapter
from iris.training import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/experiment-manifest/v1/payload"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IrisExperimentManifestWriteResult:
    """Identity of one explicit local plaintext manifest file."""

    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class IrisPreparedExperimentExample:
    """One split-ordered example paired with its prepared teacher input."""

    dataset_id: str
    manifest_payload_sha256: str
    split: ExperimentSplitV1
    split_ordinal: int
    split_example_count: int
    example_id: str
    teacher_input: IrisTeacherForcingInput


@dataclass(frozen=True, slots=True)
class _BackbonePreparationIdentity:
    configuration_json: str
    vocabulary_size: int
    max_target_tokens: int
    eos_token_ids: tuple[int, ...]
    device: torch.device


def _validate_integrity(
    manifest: IrisExperimentManifestV1,
) -> tuple[IrisExperimentManifestV1, bytes]:
    if type(manifest) is not IrisExperimentManifestV1:
        message = "manifest must be exactly IrisExperimentManifestV1"
        raise TypeError(message)
    owned = IrisExperimentManifestV1.model_validate(manifest.model_dump(mode="python"))
    payload_bytes = canonical_json_bytes(owned.payload.model_dump(mode="json"))
    expected_sha256 = domain_sha256(_PAYLOAD_DOMAIN, payload_bytes)
    if owned.payload_sha256 != expected_sha256:
        message = "experiment manifest payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_MANIFEST_BYTES:
        message = "experiment manifest exceeds the 64 MiB safety limit"
        raise ValueError(message)
    return owned, content


def build_experiment_manifest_v1(
    *,
    dataset_id: str,
    examples: tuple[ExperimentExampleV1, ...],
) -> IrisExperimentManifestV1:
    """Build one canonical in-memory manifest while preserving exact order and text."""
    try:
        if type(examples) is not tuple:
            message = "experiment examples must be an exact tuple"
            raise TypeError(message)
        owned_examples: list[ExperimentExampleV1] = []
        for example in examples:
            if type(example) is not ExperimentExampleV1:
                message = "every experiment example must be exactly ExperimentExampleV1"
                raise TypeError(message)
            owned_examples.append(
                ExperimentExampleV1.model_validate(example.model_dump(mode="python"))
            )
        counts = ExperimentSplitCountsV1(
            train=sum(example.split == "train" for example in owned_examples),
            validation=sum(example.split == "validation" for example in owned_examples),
            test=sum(example.split == "test" for example in owned_examples),
        )
        payload = ExperimentManifestPayloadV1(
            dataset_id=dataset_id,
            split_counts=counts,
            examples=tuple(owned_examples),
        )
        manifest = IrisExperimentManifestV1(
            schema_identity=ExperimentManifestSchemaV1(
                artifact_type="iris.experiment_examples",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                text_encoding="utf-8",
                text_normalization="none",
                iteration_order="manifest_order_within_split",
            ),
            payload_sha256=domain_sha256(
                _PAYLOAD_DOMAIN,
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        owned, _ = _validate_integrity(manifest)
        return owned
    except IrisExperimentManifestError:
        raise
    except Exception as error:
        message = "Iris experiment manifest could not be built"
        raise IrisExperimentManifestError(message) from error


def validate_experiment_manifest_v1(
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentManifestV1:
    """Return one deeply owned manifest after schema, hash, and size validation."""
    try:
        owned, _ = _validate_integrity(manifest)
        return owned
    except IrisExperimentManifestError:
        raise
    except Exception as error:
        message = "Iris experiment manifest could not be verified"
        raise IrisExperimentManifestError(message) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        value[key] = item
    return value


def read_experiment_manifest_v1(path: Path) -> IrisExperimentManifestV1:
    """Read and verify one canonical local plaintext example manifest."""
    try:
        if not isinstance(path, Path) or path.suffix != ".json" or not path.name:
            message = "experiment manifest reader requires an explicit .json file Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_file():
            message = "experiment manifest path must be a real regular file"
            raise ValueError(message)
        size = path.stat().st_size
        if size <= 0 or size > _MAX_MANIFEST_BYTES:
            message = "experiment manifest file size is invalid"
            raise ValueError(message)
        content = path.read_bytes()
        if len(content) != size:
            message = "experiment manifest size changed while reading"
            raise ValueError(message)
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "experiment manifest must be canonical UTF-8 JSON with one final LF"
            raise ValueError(message)
        decoded = content[:-1].decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number is forbidden: {value}")
            ),
        )
        if canonical_json_file_bytes(raw) != content:
            message = "experiment manifest file is not canonical iris-json-v1"
            raise ValueError(message)
        manifest = IrisExperimentManifestV1.model_validate_json(decoded)
        owned, expected_content = _validate_integrity(manifest)
        if expected_content != content:
            message = "experiment manifest canonical round trip changed its bytes"
            raise ValueError(message)
        return owned
    except IrisExperimentManifestError:
        raise
    except Exception as error:
        message = "Iris experiment manifest could not be read or verified"
        raise IrisExperimentManifestError(message) from error


def _existing_matches(path: Path, expected: bytes) -> bool:
    try:
        return path.read_bytes() == expected
    except (FileNotFoundError, OSError):
        return False


def _install_no_clobber(path: Path, content: bytes) -> bool:
    if path.exists():
        if path.is_symlink() or not path.is_file() or not _existing_matches(path, content):
            message = "Existing experiment manifest conflicts with expected bytes"
            raise IrisExperimentManifestError(message)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".iris-{uuid4().hex}.tmp"
    try:
        file_descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(file_descriptor, "wb", buffering=0) as stream:
                remaining = memoryview(content)
                while remaining:
                    written = stream.write(remaining)
                    if written is None or written <= 0:
                        message = "Experiment manifest write made no forward progress"
                        raise OSError(message)
                    remaining = remaining[written:]
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        if temporary_path.read_bytes() != content:
            message = "Experiment manifest temporary bytes failed verification"
            raise OSError(message)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or not _existing_matches(path, content):
                message = "Concurrent experiment manifest conflicts with expected bytes"
                raise IrisExperimentManifestError(message) from error
            reused = True
        temporary_path.unlink()
        return reused
    except BaseException as primary_error:
        try:
            temporary_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "Experiment manifest persistence and cleanup both failed",
                [primary_error, cleanup_error],
            )
            message = "Iris experiment manifest failed and cleanup was incomplete"
            raise IrisExperimentManifestError(message) from failures
        if isinstance(primary_error, IrisExperimentManifestError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris experiment manifest could not be installed atomically"
            raise IrisExperimentManifestError(message) from primary_error
        raise


def write_experiment_manifest_v1(
    manifest: IrisExperimentManifestV1,
    *,
    path: Path,
) -> IrisExperimentManifestWriteResult:
    """Write one explicit plaintext manifest atomically without overwrite."""
    try:
        if not isinstance(path, Path) or path.suffix != ".json" or not path.name:
            message = "experiment manifest writer requires an explicit .json file Path"
            raise TypeError(message)
        owned, content = _validate_integrity(manifest)
        reused = _install_no_clobber(path, content)
        return IrisExperimentManifestWriteResult(
            path=path,
            payload_sha256=owned.payload_sha256,
            size_bytes=len(content),
            reused_existing=reused,
        )
    except IrisExperimentManifestError:
        raise
    except Exception as error:
        message = "Iris experiment manifest could not be persisted"
        raise IrisExperimentManifestError(message) from error


def _backbone_identity(backbone: BackboneAdapter) -> _BackbonePreparationIdentity:
    try:
        configuration_json = backbone.configuration_json
        vocabulary_size = backbone.vocabulary_size
        max_target_tokens = backbone.max_new_tokens
        eos_token_ids = backbone.eos_token_ids
        device = backbone.device
    except Exception as error:
        message = "Experiment iterator could not read backbone preparation metadata"
        raise IrisExperimentIterationError(message) from error
    if (
        type(configuration_json) is not str
        or not configuration_json
        or type(vocabulary_size) is not int
        or vocabulary_size <= 0
        or type(max_target_tokens) is not int
        or max_target_tokens < 2
        or type(eos_token_ids) is not tuple
        or not eos_token_ids
        or any(type(token_id) is not int for token_id in eos_token_ids)
        or len(set(eos_token_ids)) != len(eos_token_ids)
        or any(token_id < 0 or token_id >= vocabulary_size for token_id in eos_token_ids)
        or not isinstance(device, torch.device)
    ):
        message = "Experiment iterator backbone preparation metadata is invalid"
        raise IrisExperimentIterationError(message)
    return _BackbonePreparationIdentity(
        configuration_json=configuration_json,
        vocabulary_size=vocabulary_size,
        max_target_tokens=max_target_tokens,
        eos_token_ids=eos_token_ids,
        device=device,
    )


class IrisExperimentExampleIterator:
    """Single-pass deterministic preparation in physical manifest order."""

    _backbone: BackboneAdapter
    _backbone_snapshot: _BackbonePreparationIdentity
    _examples: tuple[ExperimentExampleV1, ...]
    _index: int
    _manifest: IrisExperimentManifestV1
    _poisoned: bool
    _split: ExperimentSplitV1
    _terminal_token_id: int

    __slots__ = (
        "_backbone",
        "_backbone_snapshot",
        "_examples",
        "_index",
        "_manifest",
        "_poisoned",
        "_split",
        "_terminal_token_id",
    )

    def __init__(
        self,
        *,
        manifest: IrisExperimentManifestV1,
        split: ExperimentSplitV1,
        backbone: BackboneAdapter,
        terminal_token_id: int,
    ) -> None:
        try:
            owned, _ = _validate_integrity(manifest)
        except Exception as error:
            message = "Experiment iterator requires a valid manifest"
            raise IrisExperimentIterationError(message) from error
        if type(split) is not str or split not in {"train", "validation", "test"}:
            message = "Experiment iterator split must be train, validation, or test"
            raise IrisExperimentIterationError(message)
        if not isinstance(backbone, BackboneAdapter):
            message = "Experiment iterator requires an already-loaded BackboneAdapter"
            raise IrisExperimentIterationError(message)
        if type(terminal_token_id) is not int:
            message = "Experiment iterator terminal_token_id must be an exact integer"
            raise IrisExperimentIterationError(message)
        backbone_snapshot = _backbone_identity(backbone)
        if terminal_token_id not in backbone_snapshot.eos_token_ids:
            message = "Experiment iterator terminal_token_id must be a declared tokenizer EOS"
            raise IrisExperimentIterationError(message)
        typed_split = cast(ExperimentSplitV1, split)
        examples = tuple(
            example for example in owned.payload.examples if example.split == typed_split
        )
        expected_count = getattr(owned.payload.split_counts, typed_split)
        if len(examples) != expected_count:
            message = "Experiment iterator split membership is inconsistent"
            raise IrisExperimentIterationError(message)

        self._manifest = owned
        self._split = typed_split
        self._backbone = backbone
        self._backbone_snapshot = backbone_snapshot
        self._terminal_token_id = terminal_token_id
        self._examples = examples
        self._index = 0
        self._poisoned = False

    @property
    def example_count(self) -> int:
        return len(self._examples)

    @property
    def remaining_examples(self) -> int:
        return len(self._examples) - self._index

    def __iter__(self) -> IrisExperimentExampleIterator:
        return self

    def __next__(self) -> IrisPreparedExperimentExample:
        if self._poisoned:
            message = "Experiment iterator is poisoned after a preparation failure"
            raise IrisExperimentIterationError(message)
        if self._index >= len(self._examples):
            raise StopIteration
        example = self._examples[self._index]
        try:
            if _backbone_identity(self._backbone) != self._backbone_snapshot:
                message = "Backbone preparation identity changed between experiment examples"
                raise IrisExperimentIterationError(message)
            teacher_input = prepare_teacher_forcing_input(
                backbone=self._backbone,
                prompt=example.prompt,
                target_text=example.target,
                terminal_token_id=self._terminal_token_id,
            )
            if _backbone_identity(self._backbone) != self._backbone_snapshot:
                message = "Backbone preparation identity changed during experiment iteration"
                raise IrisExperimentIterationError(message)
            if type(teacher_input) is not IrisTeacherForcingInput:
                message = "Experiment target preparation returned an invalid teacher input"
                raise IrisExperimentIterationError(message)
        except Exception as error:
            self._poisoned = True
            message = f"Experiment example {example.example_id!r} could not be prepared"
            raise IrisExperimentIterationError(message) from error

        self._index += 1
        return IrisPreparedExperimentExample(
            dataset_id=self._manifest.payload.dataset_id,
            manifest_payload_sha256=self._manifest.payload_sha256,
            split=self._split,
            split_ordinal=self._index,
            split_example_count=len(self._examples),
            example_id=example.example_id,
            teacher_input=teacher_input,
        )


def iterate_experiment_split_v1(
    *,
    manifest: IrisExperimentManifestV1,
    split: ExperimentSplitV1,
    backbone: BackboneAdapter,
    terminal_token_id: int,
) -> IrisExperimentExampleIterator:
    """Create one validated single-pass split iterator without tokenizing yet."""
    return IrisExperimentExampleIterator(
        manifest=manifest,
        split=split,
        backbone=backbone,
        terminal_token_id=terminal_token_id,
    )


__all__ = [
    "IrisExperimentExampleIterator",
    "IrisExperimentManifestWriteResult",
    "IrisPreparedExperimentExample",
    "build_experiment_manifest_v1",
    "iterate_experiment_split_v1",
    "read_experiment_manifest_v1",
    "validate_experiment_manifest_v1",
    "write_experiment_manifest_v1",
]
