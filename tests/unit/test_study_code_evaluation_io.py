"""Bounded private atomic code bundles and distinct public API, not execution proof."""

import os
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application as application
import iris.artifacts as artifacts
from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_code_evaluation_io import (
    read_study_code_evaluation_v1,
    write_study_code_evaluation_v1,
)
from iris.application.study_code_receipt import _MAX_CODE_EVALUATION_BYTES
from iris.artifacts.study_code_evaluation_v1 import IrisStudyCodeEvaluationV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_study_code_receipt import _evaluate, _fixture, _rehash, _verify


def test_public_api_is_separate_from_unchanged_exact_text_v1() -> None:
    assert artifacts.IrisStudyCodeEvaluationV1 is IrisStudyCodeEvaluationV1
    assert application.read_study_code_evaluation_v1 is read_study_code_evaluation_v1
    assert application.write_study_code_evaluation_v1 is write_study_code_evaluation_v1
    for name in (
        "IrisStudyCodeEvaluationWriteResultV1",
        "evaluate_study_request_python_answer_v1",
        "validate_study_code_evaluation_v1",
        "verify_study_code_evaluation_v1",
    ):
        assert name in application.__all__ and getattr(application, name) is not None


def test_canonical_single_file_roundtrip_idempotence_and_recomputation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evaluation = _evaluate(fixture)
    path = tmp_path / "private" / "code.json"
    first = write_study_code_evaluation_v1(evaluation, path=path)
    second = write_study_code_evaluation_v1(evaluation, path=str(path))
    assert first.path == second.path == path
    assert not first.reused_existing and second.reused_existing
    assert first.payload_sha256 == evaluation.payload_sha256
    assert first.size_bytes == len(path.read_bytes())
    assert path.read_bytes() == canonical_json_file_bytes(evaluation.model_dump(mode="json"))
    assert list(path.parent.iterdir()) == [path]
    assert read_study_code_evaluation_v1(path) == evaluation
    assert _verify(read_study_code_evaluation_v1(path), fixture) == evaluation


def test_conflict_is_preserved_and_errors_never_contain_private_bytes(tmp_path: Path) -> None:
    evaluation = _evaluate(_fixture(tmp_path))
    path = tmp_path / "conflict.json"
    private = b"PRIVATE_CONFLICTING_CODE"
    path.write_bytes(private)
    with pytest.raises(IrisStudyEvaluationError) as caught:
        write_study_code_evaluation_v1(evaluation, path=path)
    assert path.read_bytes() == private
    assert private.decode() not in "".join(traceback.format_exception(caught.value))
    assert not list(tmp_path.glob(".iris-study-*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "crlf",
        "bom",
        "whitespace",
        "duplicate",
        "nan",
        "empty",
        "utf8",
        "version_bool",
    ],
)
def test_noncanonical_or_corrupted_envelope_is_rejected(tmp_path: Path, mutation: str) -> None:
    evaluation = _evaluate(_fixture(tmp_path))
    content = canonical_json_file_bytes(evaluation.model_dump(mode="json"))
    if mutation == "hash":
        content = content.replace(evaluation.payload_sha256.encode(), b"0" * 64, 1)
    elif mutation == "crlf":
        content = content[:-1] + b"\r\n"
    elif mutation == "bom":
        content = b"\xef\xbb\xbf" + content
    elif mutation == "whitespace":
        content = b" " + content
    elif mutation == "duplicate":
        content = b'{"payload":null,' + content[1:]
    elif mutation == "nan":
        content = b'{"payload":NaN}\n'
    elif mutation == "empty":
        content = b""
    elif mutation == "version_bool":
        content = content.replace(b'"version":1', b'"version":true')
    else:
        content = b"\xff\n"
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(IrisStudyEvaluationError):
        read_study_code_evaluation_v1(path)


def test_reader_enforces_twelve_mib_before_open(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * (_MAX_CODE_EVALUATION_BYTES + 1))
    with patch.object(os, "open") as opened, pytest.raises(IrisStudyEvaluationError):
        read_study_code_evaluation_v1(path)
    opened.assert_not_called()


@pytest.mark.parametrize("name", ["missing-extension", "code.JSON", "../code.json"])
def test_only_explicit_json_paths_without_traversal(tmp_path: Path, name: str) -> None:
    evaluation = _evaluate(_fixture(tmp_path))
    with pytest.raises(IrisStudyEvaluationError):
        write_study_code_evaluation_v1(evaluation, path=tmp_path / name)
    with pytest.raises(IrisStudyEvaluationError):
        read_study_code_evaluation_v1(tmp_path / name)


def test_invalid_envelope_cannot_create_destination_directory(tmp_path: Path) -> None:
    evaluation = _evaluate(_fixture(tmp_path)).model_copy(update={"payload_sha256": "0" * 64})
    path = tmp_path / "uncreated" / "code.json"
    with pytest.raises(IrisStudyEvaluationError):
        write_study_code_evaluation_v1(evaluation, path=path)
    assert not path.parent.exists()


def test_static_reading_does_not_recompute_or_claim_runtime_attestation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw = _evaluate(fixture).model_dump(mode="python")
    raw["payload"]["receipt"].update(
        outcome="incorrect", reason="case_mismatch", passed_case_count=0
    )
    forged = _rehash(raw)
    path = tmp_path / "forged.json"
    write_study_code_evaluation_v1(forged, path=path)
    assert read_study_code_evaluation_v1(path) == forged
    with pytest.raises(IrisStudyEvaluationError):
        _verify(forged, fixture)


@pytest.mark.parametrize("kind", ["leaf", "parent"])
def test_link_or_reparse_metadata_cannot_be_used_for_publication_or_reading(
    tmp_path: Path, kind: str
) -> None:
    evaluation = _evaluate(_fixture(tmp_path))
    path = tmp_path / "code.json"
    write_study_code_evaluation_v1(evaluation, path=path)
    real_lstat = Path.lstat
    target = path if kind == "leaf" else path.parent

    def linked_metadata(observed: Path) -> os.stat_result:
        metadata = real_lstat(observed)
        if observed == target:
            values = list(metadata)
            values[0] = 0o120777
            return os.stat_result(values)
        return metadata

    with patch.object(Path, "lstat", linked_metadata):
        with pytest.raises(IrisStudyEvaluationError):
            read_study_code_evaluation_v1(path)
        with pytest.raises(IrisStudyEvaluationError):
            write_study_code_evaluation_v1(evaluation, path=path)
    assert read_study_code_evaluation_v1(path) == evaluation
