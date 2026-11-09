"""Strict schema tests for local experiment-example manifests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from iris.artifacts.experiment_v1 import (
    ExperimentExampleV1,
    ExperimentManifestPayloadV1,
    ExperimentManifestSchemaV1,
    ExperimentSplitCountsV1,
    IrisExperimentManifestV1,
)

SHA = "a" * 64


def _example(
    example_id: str,
    split: str,
    *,
    prompt: str = " exact prompt \n",
    target: str = " exact target \r\n",
) -> ExperimentExampleV1:
    return ExperimentExampleV1.model_validate(
        {
            "example_id": example_id,
            "split": split,
            "prompt": prompt,
            "target": target,
        }
    )


def _payload() -> ExperimentManifestPayloadV1:
    return ExperimentManifestPayloadV1(
        dataset_id="iris-baseline:1",
        split_counts=ExperimentSplitCountsV1(train=1, validation=1, test=1),
        examples=(
            _example("train.001", "train"),
            _example("validation.001", "validation"),
            _example("test.001", "test"),
        ),
    )


def test_valid_manifest_preserves_exact_unicode_text_order_and_splits() -> None:
    prompt = "  Привет, Iris 👁️\r\n"
    target = "\t正確な答え ✨\n"
    payload = ExperimentManifestPayloadV1(
        dataset_id="unicode-v1",
        split_counts=ExperimentSplitCountsV1(train=1, validation=1, test=0),
        examples=(
            _example("z-train", "train", prompt=prompt, target=target),
            _example("a-validation", "validation"),
        ),
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
        payload_sha256=SHA,
        payload=payload,
    )

    assert manifest.payload.examples[0].prompt == prompt
    assert manifest.payload.examples[0].target == target
    assert tuple(example.example_id for example in manifest.payload.examples) == (
        "z-train",
        "a-validation",
    )
    with pytest.raises(ValidationError):
        manifest.payload_sha256 = "b" * 64


@pytest.mark.parametrize(
    "example_id",
    ["", " white", "white space", "é", "-leading", "a/b", "a" * 129, 1, True],
)
def test_example_id_rejects_nonportable_or_coerced_values(example_id: object) -> None:
    with pytest.raises(ValidationError, match="example_id"):
        ExperimentExampleV1.model_validate(
            {
                "example_id": example_id,
                "split": "train",
                "prompt": "prompt",
                "target": "target",
            }
        )


@pytest.mark.parametrize("dataset_id", ["", "white space", "_leading", "é", "a" * 129, 7])
def test_dataset_id_rejects_nonportable_values(dataset_id: object) -> None:
    values = _payload().model_dump(mode="python")
    values["dataset_id"] = dataset_id
    with pytest.raises(ValidationError, match="dataset_id"):
        ExperimentManifestPayloadV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", ""),
        ("prompt", " \t\r\n"),
        ("prompt", "bad\0text"),
        ("prompt", "\ud800"),
        ("prompt", 1),
        ("target", ""),
        ("target", "\u0000"),
        ("target", "\udfff"),
        ("target", True),
    ],
)
def test_example_text_rejects_blank_nul_surrogate_or_coercion(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "example_id": "example-1",
        "split": "train",
        "prompt": "prompt",
        "target": "target",
    }
    values[field] = value
    with pytest.raises(ValidationError, match="experiment text"):
        ExperimentExampleV1.model_validate(values)


def test_example_text_enforces_utf8_byte_limit_not_character_count() -> None:
    allowed = "я" * (1024 * 1024 // 2)
    too_large = allowed + "я"

    assert _example("large", "train", prompt=allowed).prompt == allowed
    with pytest.raises(ValidationError, match="1 MiB"):
        _example("too-large", "train", prompt=too_large)


@pytest.mark.parametrize("split", ["eval", "Validation", "", 1, True, None])
def test_split_is_one_exact_literal(split: object) -> None:
    with pytest.raises(ValidationError):
        ExperimentExampleV1.model_validate(
            {
                "example_id": "example",
                "split": split,
                "prompt": "prompt",
                "target": "target",
            }
        )


def test_payload_rejects_duplicate_ids_even_across_splits() -> None:
    with pytest.raises(ValidationError, match="globally unique"):
        ExperimentManifestPayloadV1(
            dataset_id="duplicate-test",
            split_counts=ExperimentSplitCountsV1(train=1, validation=1, test=0),
            examples=(
                _example("same", "train"),
                _example("same", "validation"),
            ),
        )


@pytest.mark.parametrize(
    "counts",
    [
        {"train": 2, "validation": 1, "test": 0},
        {"train": 1, "validation": 2, "test": 0},
        {"train": 1, "validation": 1, "test": 1},
    ],
)
def test_payload_rejects_incorrect_redundant_split_counts(counts: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="split_counts"):
        ExperimentManifestPayloadV1(
            dataset_id="count-test",
            split_counts=ExperimentSplitCountsV1(**counts),
            examples=(
                _example("train", "train"),
                _example("validation", "validation"),
            ),
        )


@pytest.mark.parametrize(
    "counts",
    [
        {"train": 0, "validation": 1, "test": 0},
        {"train": 1, "validation": 0, "test": 0},
        {"train": True, "validation": 1, "test": 0},
        {"train": 1, "validation": 1, "test": -1},
    ],
)
def test_split_counts_require_train_validation_and_nonnegative_test(
    counts: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ExperimentSplitCountsV1.model_validate(counts)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 1, True])
def test_manifest_payload_sha_is_strict_lower_hex(digest: object) -> None:
    with pytest.raises(ValidationError, match="payload_sha256"):
        IrisExperimentManifestV1.model_validate(
            {
                "schema_identity": {
                    "artifact_type": "iris.experiment_examples",
                    "version": 1,
                    "canonicalization": "iris-json-v1",
                    "hash_algorithm": "sha256",
                    "text_encoding": "utf-8",
                    "text_normalization": "none",
                    "iteration_order": "manifest_order_within_split",
                },
                "payload_sha256": digest,
                "payload": _payload().model_dump(mode="python"),
            }
        )


def test_schema_forbids_unknown_fields_and_literal_drift() -> None:
    values: dict[str, Any] = {
        "artifact_type": "iris.experiment_examples",
        "version": 1,
        "canonicalization": "iris-json-v1",
        "hash_algorithm": "sha256",
        "text_encoding": "utf-8",
        "text_normalization": "none",
        "iteration_order": "manifest_order_within_split",
    }
    with pytest.raises(ValidationError):
        ExperimentManifestSchemaV1.model_validate({**values, "unknown": True})
    with pytest.raises(ValidationError):
        ExperimentManifestSchemaV1.model_validate({**values, "version": True})
    with pytest.raises(ValidationError):
        ExperimentManifestSchemaV1.model_validate({**values, "iteration_order": "sorted_id"})
