"""Bounded canonical evaluation receipts, no-clobber publication, and public API."""

import os
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application as application
import iris.artifacts as artifacts
from iris.application.errors import IrisStudyEvaluationError
from iris.application.study_evaluation import validate_study_evaluation_v1
from iris.application.study_evaluation_io import (
    read_study_evaluation_v1,
    write_study_evaluation_v1,
)
from iris.artifacts.study_evaluation_v1 import IrisStudyEvaluationV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from tests.unit.test_study_evaluation import _evaluate, _request


def test_public_api_exports_exact_evaluation_types_and_services() -> None:
    assert application.read_study_evaluation_v1 is read_study_evaluation_v1
    assert application.write_study_evaluation_v1 is write_study_evaluation_v1
    assert application.validate_study_evaluation_v1 is validate_study_evaluation_v1
    assert application.IrisStudyEvaluationError is IrisStudyEvaluationError
    assert artifacts.IrisStudyEvaluationV1 is IrisStudyEvaluationV1
    for name in (
        "IrisStudyEvaluationWriteResultV1",
        "evaluate_study_request_answer_v1",
        "verify_study_evaluation_v1",
    ):
        assert name in application.__all__ and getattr(application, name) is not None


def test_canonical_round_trip_and_idempotent_no_clobber_storage(tmp_path: Path) -> None:
    evaluation = _evaluate(_request(tmp_path))
    path = tmp_path / "private" / "evaluation.json"
    first = write_study_evaluation_v1(evaluation, path=path)
    content = path.read_bytes()
    second = write_study_evaluation_v1(evaluation, path=str(path))
    assert not first.reused_existing and second.reused_existing
    assert first.path == second.path == path
    assert first.payload_sha256 == evaluation.payload_sha256
    assert first.size_bytes == len(content)
    assert content == canonical_json_file_bytes(evaluation.model_dump(mode="json"))
    assert read_study_evaluation_v1(path) == evaluation
    assert list(path.parent.iterdir()) == [path]


def test_conflicting_file_is_preserved_and_private_errors_are_redacted(tmp_path: Path) -> None:
    evaluation = _evaluate(_request(tmp_path))
    path = tmp_path / "conflicting.json"
    private = b"PRIVATE_CONFLICTING_CONTENT"
    path.write_bytes(private)
    with pytest.raises(IrisStudyEvaluationError) as caught:
        write_study_evaluation_v1(evaluation, path=path)
    assert path.read_bytes() == private
    assert private.decode() not in "".join(traceback.format_exception(caught.value))
    assert not list(tmp_path.glob(".iris-study-*.tmp"))


@pytest.mark.parametrize(
    "mutation", ["hash", "crlf", "bom", "whitespace", "duplicate", "nan", "empty", "utf8"]
)
def test_reader_rejects_corrupted_or_noncanonical_files(tmp_path: Path, mutation: str) -> None:
    evaluation = _evaluate(_request(tmp_path))
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
    else:
        content = b"\xff\n"
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(IrisStudyEvaluationError):
        read_study_evaluation_v1(path)


def test_reader_enforces_one_mib_cap_before_opening_file(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with patch.object(os, "open") as open_file, pytest.raises(IrisStudyEvaluationError):
        read_study_evaluation_v1(path)
    open_file.assert_not_called()


@pytest.mark.parametrize("name", ["no-extension", "evaluation.JSON", "../evaluation.json"])
def test_paths_must_be_explicit_json_without_traversal(tmp_path: Path, name: str) -> None:
    evaluation = _evaluate(_request(tmp_path))
    with pytest.raises(IrisStudyEvaluationError):
        write_study_evaluation_v1(evaluation, path=tmp_path / name)
    with pytest.raises(IrisStudyEvaluationError):
        read_study_evaluation_v1(tmp_path / name)


def test_invalid_envelope_is_rejected_before_creating_directories(tmp_path: Path) -> None:
    evaluation = _evaluate(_request(tmp_path)).model_copy(update={"payload_sha256": "0" * 64})
    path = tmp_path / "uncreated" / "evaluation.json"
    with pytest.raises(IrisStudyEvaluationError):
        write_study_evaluation_v1(evaluation, path=path)
    assert not path.parent.exists()


def test_reader_does_not_claim_dependency_verification(tmp_path: Path) -> None:
    raw = _evaluate(_request(tmp_path)).model_dump(mode="python")
    raw["payload"]["outcome"] = "incorrect"
    raw["payload_sha256"] = domain_sha256(
        "iris/study-evaluation/v1/payload", canonical_json_bytes(raw["payload"])
    )
    forged = IrisStudyEvaluationV1.model_validate(raw)
    path = tmp_path / "forged.json"
    write_study_evaluation_v1(forged, path=path)
    assert read_study_evaluation_v1(path) == forged
    # Deliberately schema/hash-only. A separate service test proves recomputation rejects this.


@pytest.mark.parametrize("kind", ["leaf", "parent"])
def test_link_or_reparse_metadata_is_rejected_for_reads_and_writes(
    tmp_path: Path, kind: str
) -> None:
    evaluation = _evaluate(_request(tmp_path))
    path = tmp_path / "evaluation.json"
    write_study_evaluation_v1(evaluation, path=path)
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
            read_study_evaluation_v1(path)
        with pytest.raises(IrisStudyEvaluationError):
            write_study_evaluation_v1(evaluation, path=path)
    assert read_study_evaluation_v1(path) == evaluation
