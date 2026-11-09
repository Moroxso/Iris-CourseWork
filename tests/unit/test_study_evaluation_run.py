"""Complete membership, private prefix recovery and aggregates; Python observations are mocked."""

import os
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from iris.application import study_code_evaluation as scorer
from iris.application import study_evaluation_run as service
from iris.application import study_evaluation_run_io as storage
from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_evaluation_run_v1 import (
    IrisStudyEvaluationRunV1,
    IrisStudyEvaluationSummaryV1,
)
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.integrations.python_sandbox import PythonSandboxError, PythonSandboxObservationV1, _worker
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import process_file_lock
from tests.unit.test_study_code_evaluation import _POLICY, _observation
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_input_preflight import _assets, _tokenizer

type _Fixture = tuple[IrisStudyGenerationV1, IrisStudyBenchmarkV1, Path]
_BUDGET = 32 * 1024 * 1024


@contextmanager
def _runtime() -> Iterator[AsyncMock]:
    async def execute(
        *, request_bytes: bytes, policy: PythonSandboxPolicyV1
    ) -> PythonSandboxObservationV1:
        assert policy == _POLICY
        return replace(
            _observation(b'{"kind":"syntax_error"}', request_bytes),
            worker_sha256=sha256(_worker()).hexdigest(),
        )

    mock = AsyncMock(side_effect=execute)
    with _assets(_tokenizer()), patch.object(scorer, "execute_python_job_v1", mock):
        yield mock


def _evaluate(
    fixture: _Fixture, directory: Path, *, resume: bool = False, **kwargs: Any
) -> service.IrisStudyEvaluationRunResultV1:
    arguments: dict[str, Any] = dict(
        benchmark=fixture[1],
        request_root=fixture[2],
        directory=directory,
        code_policy=_POLICY,
        max_storage_bytes=_BUDGET,
        resume=resume,
    )
    arguments.update(kwargs)
    return service.evaluate_study_split_v1(fixture[0], **arguments)


def _verify(fixture: _Fixture, directory: Path) -> IrisStudyEvaluationSummaryV1:
    return service.verify_study_evaluation_run_v1(
        directory, generation=fixture[0], benchmark=fixture[1], request_root=fixture[2]
    )


def test_complete_five_family_run_and_no_execution_resume(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime() as runtime:
        first = _evaluate(fixture, directory)
        assert runtime.await_count == 1
        summary = _verify(fixture, directory)
        again = _evaluate(fixture, directory, resume=True)
        assert runtime.await_count == 1
    assert first.evaluated_now == 5 and first.reused_evaluations == 0
    assert again.evaluated_now == 0 and again.reused_evaluations == 5
    assert first.summary_payload_sha256 == again.summary_payload_sha256 == summary.payload_sha256
    assert first.retained_file_bytes == sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    assert summary.payload.overall.model_dump() == dict(
        total=5, correct=4, incorrect=0, invalid_output=1
    )
    assert len(summary.payload.by_family) == 5
    assert tuple(item.example_id for item in summary.payload.results) == tuple(
        item.example_id for item in fixture[0].payload.answers
    )
    assert summary.payload.scientific_launch_status == "not_assessed"
    assert not list(directory.glob("*-pending.json"))


def test_runtime_failure_preserves_prefix_and_cannot_publish_partial_summary(
    tmp_path: Path,
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with (
        _assets(_tokenizer()),
        patch.object(
            scorer,
            "execute_python_job_v1",
            AsyncMock(side_effect=PythonSandboxError("PRIVATE_FAILURE")),
        ),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        _evaluate(fixture, directory)
    assert "PRIVATE_FAILURE" not in "".join(traceback.format_exception(caught.value))
    assert sorted(path.name for path in (directory / "receipts").iterdir()) == [
        "000000.json",
        "000001.json",
    ]
    assert not (directory / "summary.json").exists()
    saved = (directory / "receipts/000000.json").read_bytes()
    with _runtime():
        result = _evaluate(fixture, directory, resume=True)
        assert _verify(fixture, directory).payload.overall.total == 5
    assert result.evaluated_now == 3 and result.reused_evaluations == 2
    assert (directory / "receipts/000000.json").read_bytes() == saved


@pytest.mark.parametrize("boundary", ["partial_write", "before_link", "after_link"])
def test_interrupted_code_publication_recovers_without_reexecuting_committed_or_complete_pending(
    tmp_path: Path, boundary: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    original_publish = storage._publish
    original_write, original_link, original_unlink = os.write, os.link, Path.unlink

    def interrupted(destination: Path, content: bytes, **kwargs: Any) -> None:
        if destination.name != "000002.json":
            original_publish(destination, content, **kwargs)
            return

        def partial(fd: int, data: Any) -> int:
            original_write(fd, data[:23])
            raise KeyboardInterrupt

        def no_link(*args: Any, **options: Any) -> Any:
            raise KeyboardInterrupt

        def no_unlink(path: Path, **options: Any) -> None:
            if path.name == ".receipt-pending.json":
                raise KeyboardInterrupt
            original_unlink(path, **options)

        with (
            patch.object(os, "write", partial if boundary == "partial_write" else original_write),
            patch.object(os, "link", no_link if boundary == "before_link" else original_link),
            patch.object(
                Path, "unlink", no_unlink if boundary == "after_link" else original_unlink
            ),
        ):
            original_publish(destination, content, **kwargs)

    with _runtime() as runtime:
        with patch.object(service, "_publish", interrupted), pytest.raises(KeyboardInterrupt):
            _evaluate(fixture, directory)
        assert runtime.await_count == 1
        assert (directory / ".receipt-pending.json").exists()
        assert not (directory / "summary.json").exists()
        result = _evaluate(fixture, directory, resume=True)
        assert runtime.await_count == (2 if boundary == "partial_write" else 1)
        assert _verify(fixture, directory).payload.overall.total == 5
    assert result.reused_evaluations == (2 if boundary == "partial_write" else 3)
    interrupted_files = list((directory / "interrupted").glob("*.bin"))
    assert len(interrupted_files) == (1 if boundary == "partial_write" else 0)
    if interrupted_files:
        assert interrupted_files[0].stat().st_size == 23
    assert not (directory / ".receipt-pending.json").exists()


def test_partial_initialization_can_resume_only_exact_plan_prefix(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    original_write = os.write

    def partial(fd: int, data: Any) -> int:
        if bytes(data) == b"1":
            return original_write(fd, data)
        original_write(fd, data[:23])
        raise KeyboardInterrupt

    with _runtime() as runtime:
        with patch.object(os, "write", partial), pytest.raises(KeyboardInterrupt):
            _evaluate(fixture, directory)
        assert runtime.await_count == 0
        assert not (directory / "run.json").exists()
        assert _evaluate(fixture, directory, resume=True).evaluated_now == 5
        assert len(list((directory / "interrupted").iterdir())) == 1
        assert _verify(fixture, directory).payload.overall.total == 5


def test_interrupted_summary_recovers_without_new_scoring(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    original = storage._publish

    def stop(destination: Path, content: bytes, **kwargs: Any) -> None:
        if destination.name == "summary.json":
            (directory / ".summary-pending.json").write_bytes(content[:23])
            raise KeyboardInterrupt
        original(destination, content, **kwargs)

    with _runtime() as runtime:
        with patch.object(service, "_publish", stop), pytest.raises(KeyboardInterrupt):
            _evaluate(fixture, directory)
        assert runtime.await_count == 1
        result = _evaluate(fixture, directory, resume=True)
        assert result.evaluated_now == 0 and runtime.await_count == 1
        assert _verify(fixture, directory).payload.overall.total == 5


@pytest.mark.parametrize(
    "change", ["policy", "budget", "resume", "foreign", "gap", "receipt", "summary", "request"]
)
def test_resume_never_overwrites_conflicts_or_reexecutes_to_hide_them(
    tmp_path: Path, change: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    kwargs: dict[str, Any] = {"resume": True}
    if change == "policy":
        kwargs["code_policy"] = _POLICY.model_copy(update={"wall_time_ms": 2001})
    elif change == "budget":
        kwargs["max_storage_bytes"] = _BUDGET + 1
    elif change == "resume":
        kwargs["resume"] = False
    elif change == "foreign":
        (directory / "foreign-private.json").write_bytes(b"PRESERVE_ME")
    elif change == "gap":
        (directory / "receipts/000001.json").rename(directory / "receipts/000008.json")
    elif change == "receipt":
        (directory / "receipts/000000.json").write_bytes(b"PRIVATE_CORRUPT_RECEIPT")
    elif change == "summary":
        (directory / "summary.json").write_bytes(b"PRIVATE_CORRUPT_SUMMARY")
    else:
        path = (
            fixture[2]
            / "requests/v4"
            / f"{fixture[0].payload.answers[0].request_payload_sha256}.json"
        )
        path.write_bytes(b"PRIVATE_CORRUPT_REQUEST")
    before = {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    with _runtime() as runtime, pytest.raises(IrisStudyEvaluationError):
        _evaluate(fixture, directory, **kwargs)
    runtime.assert_not_called()
    after = {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_active_owner_prevents_another_run_before_scoring(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    directory.mkdir()
    with (
        process_file_lock(directory / ".lock"),
        _runtime() as runtime,
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture, directory)
    runtime.assert_not_called()
    assert list(directory.iterdir()) == [directory / ".lock"]


def test_budget_prevents_code_execution_without_silently_dropping_it(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime() as runtime, pytest.raises(IrisStudyEvaluationError):
        _evaluate(fixture, directory, max_storage_bytes=4 * 1024 * 1024)
    runtime.assert_not_called()
    assert len(list((directory / "receipts").iterdir())) == 2
    assert not (directory / "summary.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_storage_bytes", True),
        ("max_storage_bytes", 100),
        ("resume", 1),
        ("directory", Path("relative")),
    ],
)
def test_invalid_controls_cannot_execute_source(tmp_path: Path, field: str, value: Any) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    args: dict[str, Any] = {field: value}
    if field == "directory":
        directory = args.pop(field)
    with _runtime() as runtime, pytest.raises(IrisStudyEvaluationError):
        _evaluate(fixture, directory, **args)
    runtime.assert_not_called()
    assert not (tmp_path / "evaluation").exists()


def test_static_summary_can_be_forged_but_independent_verifier_recomputes_every_result(
    tmp_path: Path,
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    raw = storage.read_study_evaluation_summary_v1(directory).model_dump(mode="python")
    raw["payload"]["results"][0]["file_sha256"] = "0" * 64
    raw["payload_sha256"] = domain_sha256(
        storage._SUMMARY_DOMAIN, canonical_json_bytes(raw["payload"])
    )
    forged = IrisStudyEvaluationSummaryV1.model_validate(raw)
    (directory / "summary.json").write_bytes(
        canonical_json_file_bytes(forged.model_dump(mode="json"))
    )
    assert storage.read_study_evaluation_summary_v1(directory) == forged
    with _runtime() as runtime, pytest.raises(IrisStudyEvaluationError):
        _verify(fixture, directory)
    runtime.assert_not_called()


@pytest.mark.parametrize("field", ["total", "correct", "incorrect", "invalid_output"])
def test_summary_counts_cannot_drop_or_misclassify_answers(tmp_path: Path, field: str) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    raw = storage.read_study_evaluation_summary_v1(directory).model_dump(mode="python")
    raw["payload"]["overall"][field] += 1
    with pytest.raises(ValidationError):
        IrisStudyEvaluationSummaryV1.model_validate(raw)


def test_run_and_summary_are_distinct_artifact_types(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    raw = storage.read_study_evaluation_run_v1(directory).model_dump(mode="python")
    raw["schema_identity"]["artifact_type"] = "iris.study_evaluation_summary"
    with pytest.raises(ValidationError):
        IrisStudyEvaluationRunV1.model_validate(raw)


def test_public_api_exports_run_summary_and_services() -> None:
    import iris.application as application
    import iris.artifacts as artifacts

    assert artifacts.IrisStudyEvaluationRunV1 is IrisStudyEvaluationRunV1
    assert artifacts.IrisStudyEvaluationSummaryV1 is IrisStudyEvaluationSummaryV1
    assert application.evaluate_study_split_v1 is service.evaluate_study_split_v1
    assert application.verify_study_evaluation_run_v1 is service.verify_study_evaluation_run_v1
    assert application.read_study_evaluation_run_v1 is storage.read_study_evaluation_run_v1
    assert application.read_study_evaluation_summary_v1 is storage.read_study_evaluation_summary_v1


def test_death_after_archiving_partial_initialization_can_resume_again(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    original_publish = storage._publish

    def interrupt_initialization(destination: Path, content: bytes, **kwargs: Any) -> None:
        if destination.name == "run.json":
            if not (directory / "interrupted").exists():
                (directory / ".run-pending.json").write_bytes(content[:20])
            raise KeyboardInterrupt
        original_publish(destination, content, **kwargs)

    with _runtime() as runtime:
        with (
            patch.object(service, "_publish", interrupt_initialization),
            pytest.raises(KeyboardInterrupt),
        ):
            _evaluate(fixture, directory)
        with (
            patch.object(service, "_publish", interrupt_initialization),
            pytest.raises(KeyboardInterrupt),
        ):
            _evaluate(fixture, directory, resume=True)
        assert not (directory / ".run-pending.json").exists()
        assert not (directory / "run.json").exists()
        result = _evaluate(fixture, directory, resume=True)
        assert result.evaluated_now == 5 and runtime.await_count == 1


def test_pending_summary_with_missing_receipts_is_not_a_license_to_rerun(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    (directory / "summary.json").rename(directory / ".summary-pending.json")
    (directory / "receipts/000004.json").unlink()
    with _runtime() as runtime, pytest.raises(IrisStudyEvaluationError):
        _evaluate(fixture, directory, resume=True)
    runtime.assert_not_called()
    assert (directory / ".summary-pending.json").exists()


def test_mutation_during_summary_publication_fails_final_independent_check(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    original = storage._publish

    def corrupted(destination: Path, content: bytes, **kwargs: Any) -> None:
        original(destination, content, **kwargs)
        if destination.name == "summary.json":
            (directory / "receipts/000000.json").write_bytes(b"CORRUPTED_AFTER_PRECHECK")

    with (
        _runtime(),
        patch.object(service, "_publish", corrupted),
        pytest.raises(IrisStudyEvaluationError),
    ):
        _evaluate(fixture, directory)


@pytest.mark.parametrize("name", ["run.json", "summary.json"])
@pytest.mark.parametrize("mutation", ["hash", "duplicate", "whitespace", "bool_version"])
def test_canonical_metadata_readers_reject_corruption(
    tmp_path: Path, name: str, mutation: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    path = directory / name
    content = path.read_bytes()
    if mutation == "hash":
        content = content.replace(b'"payload_sha256":"', b'"payload_sha256":"0', 1)
    elif mutation == "duplicate":
        content = b'{"payload":null,' + content[1:]
    elif mutation == "whitespace":
        content = b" " + content
    else:
        content = content.replace(b'"version":1', b'"version":true')
    path.write_bytes(content)
    reader = (
        storage.read_study_evaluation_run_v1
        if name == "run.json"
        else storage.read_study_evaluation_summary_v1
    )
    with pytest.raises(IrisStudyEvaluationError):
        reader(directory)


def test_independent_verifier_does_not_mutate_files_or_acquire_a_lock(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "evaluation"
    with _runtime():
        _evaluate(fixture, directory)
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    with _runtime() as runtime, patch.object(service, "process_file_lock") as claim:
        _verify(fixture, directory)
    runtime.assert_not_called()
    claim.assert_not_called()
    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert before == after
