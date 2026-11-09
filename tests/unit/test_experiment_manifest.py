"""Build, canonical I/O, and atomic-install tests for experiment manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from iris.application.errors import IrisExperimentManifestError
from iris.application.experiment_manifest import (
    IrisExperimentManifestWriteResult,
    build_experiment_manifest_v1,
    read_experiment_manifest_v1,
    validate_experiment_manifest_v1,
    write_experiment_manifest_v1,
)
from iris.artifacts.experiment_v1 import ExperimentExampleV1, IrisExperimentManifestV1
from iris.utils.hashing import canonical_json_file_bytes, domain_sha256


def _example(example_id: str, split: str, prompt: str, target: str) -> ExperimentExampleV1:
    return ExperimentExampleV1.model_validate(
        {
            "example_id": example_id,
            "split": split,
            "prompt": prompt,
            "target": target,
        }
    )


def _examples() -> tuple[ExperimentExampleV1, ...]:
    return (
        _example("train-2", "train", " prompt two ", " target two "),
        _example("validation-1", "validation", "validation prompt", "validation target"),
        _example("train-1", "train", "prompt one", "target one"),
        _example("test-1", "test", "test prompt", "test target"),
    )


def _manifest() -> IrisExperimentManifestV1:
    return build_experiment_manifest_v1(dataset_id="iris-local-v1", examples=_examples())


def test_builder_preserves_physical_order_text_and_computes_counts_and_hash() -> None:
    examples = _examples()

    manifest = build_experiment_manifest_v1(
        dataset_id="iris-local-v1",
        examples=examples,
    )

    assert manifest.payload.dataset_id == "iris-local-v1"
    assert tuple(example.example_id for example in manifest.payload.examples) == (
        "train-2",
        "validation-1",
        "train-1",
        "test-1",
    )
    assert manifest.payload.examples[0].prompt == " prompt two "
    assert manifest.payload.examples[0].target == " target two "
    assert manifest.payload.split_counts.model_dump(mode="python") == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    expected_payload_hash = domain_sha256(
        "iris/experiment-manifest/v1/payload",
        canonical_json_file_bytes(manifest.payload.model_dump(mode="json"))[:-1],
    )
    assert manifest.payload_sha256 == expected_payload_hash


def test_public_validator_returns_owned_manifest_and_rejects_hash_drift() -> None:
    manifest = _manifest()

    validated = validate_experiment_manifest_v1(manifest)

    assert validated == manifest
    assert validated is not manifest
    with pytest.raises(IrisExperimentManifestError, match="verified"):
        validate_experiment_manifest_v1(manifest.model_copy(update={"payload_sha256": "b" * 64}))


def test_builder_is_deterministic_and_order_is_hash_significant() -> None:
    first = _manifest()
    second = _manifest()
    reordered = build_experiment_manifest_v1(
        dataset_id="iris-local-v1",
        examples=tuple(reversed(_examples())),
    )

    assert first == second
    assert canonical_json_file_bytes(first.model_dump(mode="json")) == canonical_json_file_bytes(
        second.model_dump(mode="json")
    )
    assert reordered.payload_sha256 != first.payload_sha256
    assert reordered.payload.examples != first.payload.examples


def test_builder_owns_examples_against_post_build_mutation() -> None:
    examples = _examples()
    manifest = build_experiment_manifest_v1(dataset_id="iris-local-v1", examples=examples)

    object.__setattr__(examples[0], "prompt", "mutated")

    assert manifest.payload.examples[0].prompt == " prompt two "


def test_builder_rejects_non_tuple_subclasses_and_missing_required_split() -> None:
    with pytest.raises(IrisExperimentManifestError):
        build_experiment_manifest_v1(
            dataset_id="iris-local-v1",
            examples=cast(tuple[ExperimentExampleV1, ...], list(_examples())),
        )

    class ExampleSubclass(ExperimentExampleV1):
        pass

    subclass = ExampleSubclass.model_validate(_examples()[0].model_dump(mode="python"))
    with pytest.raises(IrisExperimentManifestError):
        build_experiment_manifest_v1(
            dataset_id="iris-local-v1",
            examples=(subclass, _examples()[1]),
        )

    with pytest.raises(IrisExperimentManifestError):
        build_experiment_manifest_v1(
            dataset_id="iris-local-v1",
            examples=tuple(example for example in _examples() if example.split != "validation"),
        )


def test_writer_reader_round_trip_and_exact_reuse(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "data" / "experiment.json"

    first = write_experiment_manifest_v1(manifest, path=path)
    loaded = read_experiment_manifest_v1(path)
    second = write_experiment_manifest_v1(manifest, path=path)

    assert type(first) is IrisExperimentManifestWriteResult
    assert first.path == path
    assert first.payload_sha256 == manifest.payload_sha256
    assert first.size_bytes == path.stat().st_size
    assert first.reused_existing is False
    assert second.reused_existing is True
    assert loaded == manifest
    assert path.read_bytes() == canonical_json_file_bytes(manifest.model_dump(mode="json"))
    assert b" prompt two " in path.read_bytes()
    assert not tuple(path.parent.glob(".iris-*.tmp"))


@pytest.mark.parametrize(
    "tamper",
    ["bom", "missing_lf", "double_lf", "whitespace", "duplicate", "hash", "unknown"],
)
def test_reader_rejects_noncanonical_or_tampered_files(tmp_path: Path, tamper: str) -> None:
    manifest = _manifest()
    path = tmp_path / "experiment.json"
    content = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    if tamper == "bom":
        content = b"\xef\xbb\xbf" + content
    elif tamper == "missing_lf":
        content = content[:-1]
    elif tamper == "double_lf":
        content += b"\n"
    elif tamper == "whitespace":
        content = content[:-1] + b" \n"
    elif tamper == "duplicate":
        content = b'{"payload_sha256":"' + manifest.payload_sha256.encode() + b'",' + content[1:]
    elif tamper == "hash":
        content = content.replace(manifest.payload_sha256.encode(), b"b" * 64, 1)
    else:
        content = content[:-2] + b',"unknown":true}\n'
    path.write_bytes(content)

    with pytest.raises(IrisExperimentManifestError):
        read_experiment_manifest_v1(path)


def test_reader_writer_require_explicit_json_regular_file_paths(tmp_path: Path) -> None:
    manifest = _manifest()
    for invalid in (tmp_path / "manifest.txt", tmp_path / "no-suffix"):
        with pytest.raises(IrisExperimentManifestError):
            write_experiment_manifest_v1(manifest, path=invalid)
        invalid.write_text("x", encoding="utf-8")
        with pytest.raises(IrisExperimentManifestError):
            read_experiment_manifest_v1(invalid)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(IrisExperimentManifestError):
        read_experiment_manifest_v1(directory)


def test_writer_revalidates_mutated_manifest_before_filesystem_change(tmp_path: Path) -> None:
    manifest = _manifest()
    mutated = manifest.model_copy(update={"payload_sha256": "b" * 64})
    path = tmp_path / "experiment.json"

    with pytest.raises(IrisExperimentManifestError):
        write_experiment_manifest_v1(mutated, path=path)

    assert not path.exists()


def test_writer_rejects_existing_conflict_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_bytes(b"existing private data")

    with pytest.raises(IrisExperimentManifestError, match="conflicts"):
        write_experiment_manifest_v1(_manifest(), path=path)

    assert path.read_bytes() == b"existing private data"
    assert not tuple(tmp_path.glob(".iris-*.tmp"))


def test_writer_cleans_temporary_file_after_link_failure(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"

    with (
        patch.object(os, "link", side_effect=OSError("synthetic link failure")),
        pytest.raises(IrisExperimentManifestError),
    ):
        write_experiment_manifest_v1(_manifest(), path=path)

    assert not path.exists()
    assert not tuple(tmp_path.glob(".iris-*.tmp"))


def test_concurrent_identical_install_reuses_winner_and_cleans_temp(tmp_path: Path) -> None:
    manifest = _manifest()
    expected = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    path = tmp_path / "experiment.json"

    def install_winner(source: Path, target: Path) -> None:
        del source
        target.write_bytes(expected)
        raise FileExistsError

    with patch.object(os, "link", side_effect=install_winner):
        result = write_experiment_manifest_v1(manifest, path=path)

    assert result.reused_existing is True
    assert path.read_bytes() == expected
    assert not tuple(tmp_path.glob(".iris-*.tmp"))
