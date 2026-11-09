"""Real canonical answer bindings with substituted inference only; no candidate execution."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import study_generation_run as service
from iris.application import study_generation_run_io as storage
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import read_request_artifact_v4, write_request_artifact_v4
from iris.application.study_generation import _GeneratedStudyAnswer, _PreparedStudyGeneration
from iris.application.study_generation_io import read_study_generation_v1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.utils.process_file_lock import process_file_lock
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_input_preflight import _assets, _tokenizer

type _Fixture = tuple[IrisStudyGenerationV1, IrisStudyBenchmarkV1, Path]


def _run(
    fixture: _Fixture, directory: Path, *, resume: bool = False, **changes: Any
) -> service.IrisStudyGenerationRunResultV1:
    generation, benchmark, root = fixture
    first = read_request_artifact_v4(
        root / "requests" / "v4" / f"{generation.payload.answers[0].request_payload_sha256}.json"
    )
    options: dict[str, Any] = dict(
        config=first.payload.configuration.value,
        arm_id="A",
        checkpoint_identity=None,
        split=generation.payload.split,
        decoding=tuple(a.decoding for a in generation.payload.answers),
        terminal_token_id=1,
        max_artifact_bytes=generation.payload.max_artifact_bytes,
        max_record_bytes=1024 * 1024,
        max_storage_bytes=16 * 1024 * 1024,
        directory=directory,
        resume=resume,
    )
    options.update(changes)
    return service.generate_study_split_run_v1(benchmark, **options)


@contextmanager
def _inference(fixture: _Fixture) -> Iterator[list[int]]:
    seen: list[int] = []

    def generate(
        prepared: _PreparedStudyGeneration, *, start_index: int, answer_bytes: int
    ) -> Iterator[_GeneratedStudyAnswer]:
        assert answer_bytes >= 0
        for index in range(start_index, len(prepared.selected)):
            answer = fixture[0].payload.answers[index]
            request = read_request_artifact_v4(
                fixture[2] / "requests" / "v4" / f"{answer.request_payload_sha256}.json"
            )
            seen.append(index)
            yield _GeneratedStudyAnswer(request, answer)

    with (
        _assets(_tokenizer()),
        patch.object(service, "_iter_study_generation_answers", side_effect=generate),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("forbidden model load"),
        ),
        patch(
            "iris.application.study_code_evaluation.execute_python_job_v1",
            side_effect=AssertionError("forbidden scoring"),
        ),
    ):
        yield seen


def _saved(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_complete_run_is_compatible_and_complete_resume_never_infers(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture) as seen:
        first = _run(fixture, directory)
        assert seen == list(range(5))
        assert read_study_generation_v1(first.path) == fixture[0]
        before = _saved(directory)
        again = _run(fixture, directory, resume=True)
        assert seen == list(range(5))
        assert service.verify_study_generation_run_v1(directory, benchmark=fixture[1]) == fixture[0]
        assert _saved(directory) == before
    assert (first.generated_now, first.reused_answers) == (5, 0)
    assert (again.generated_now, again.reused_answers) == (0, 5)
    assert (
        first.generation_payload_sha256
        == again.generation_payload_sha256
        == fixture[0].payload_sha256
    )
    assert first.retained_file_bytes == sum(len(value) for value in before.values())


@pytest.mark.parametrize(
    "boundary", ["staged", "request", "linked", "committed", "partial", "final"]
)
def test_interrupted_publication_recovers_only_missing_answers(
    tmp_path: Path, boundary: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    original_stage, original_publish, original_link = storage._stage, storage._publish, os.link

    def stage(pending: Path, content: bytes, **kwargs: Any) -> None:
        target = (
            pending.name == ".answer-pending.json" and json.loads(content)["payload"]["index"] == 2
        )
        if boundary == "partial" and target:
            pending.write_bytes(content[:23])
            raise KeyboardInterrupt
        original_stage(pending, content, **kwargs)
        if (boundary == "staged" and target) or (
            boundary == "final" and pending.name == ".generation-pending.json"
        ):
            raise KeyboardInterrupt

    def request(artifact: Any) -> Any:
        result = write_request_artifact_v4(artifact)
        if (
            boundary == "request"
            and artifact.payload_sha256 == fixture[0].payload.answers[2].request_payload_sha256
        ):
            raise KeyboardInterrupt
        return result

    def link(source: Any, destination: Any, **kwargs: Any) -> None:
        original_link(source, destination, **kwargs)
        if boundary == "linked" and Path(destination).name == "000002.json":
            raise KeyboardInterrupt

    def publish(destination: Path, content: bytes, **kwargs: Any) -> None:
        original_publish(destination, content, **kwargs)
        if boundary == "committed" and destination.name == "000002.json":
            raise KeyboardInterrupt

    with _inference(fixture) as seen:
        with (
            patch.object(storage, "_stage", side_effect=stage),
            patch.object(service, "_stage", side_effect=stage),
            patch.object(service, "write_request_artifact_v4", side_effect=request),
            patch.object(os, "link", side_effect=link),
            patch.object(service, "_publish", side_effect=publish),
            pytest.raises(KeyboardInterrupt),
        ):
            _run(fixture, directory)
        assert not (directory / "generation.json").exists()
        before_resume = len(seen)
        result = _run(fixture, directory, resume=True)
        missing = 0 if boundary == "final" else 3 if boundary == "partial" else 2
        assert seen[before_resume:] == list(range(5 - missing, 5))
        assert (result.generated_now, result.reused_answers) == (missing, 5 - missing)
        assert service.verify_study_generation_run_v1(directory, benchmark=fixture[1]) == fixture[0]
        if boundary == "partial":
            archived = tuple((directory / "interrupted").iterdir())
            assert len(archived) == 1 and len(archived[0].read_bytes()) == 23
            assert archived[0].stem == bytes_sha256(archived[0].read_bytes())
        assert not any((directory / name).exists() for name in storage._PENDING_NAMES)


@pytest.mark.parametrize("change", ["output", "index", "run", "missing", "request"])
def test_corrupted_committed_prefix_cannot_trigger_new_inference(
    tmp_path: Path, change: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture):
        _run(fixture, directory)
    path = directory / "answers" / "000001.json"
    if change == "missing":
        path.rename(tmp_path / "preserved-answer.json")
    elif change == "request":
        request = (
            fixture[2]
            / "requests"
            / "v4"
            / f"{fixture[0].payload.answers[1].request_payload_sha256}.json"
        )
        request.write_bytes(b"corrupt")
    else:
        raw = json.loads(path.read_bytes())
        if change == "output":
            raw["payload"]["answer"].update(
                output_text="altered", output_byte_length=7, output_sha256=bytes_sha256(b"altered")
            )
        elif change == "index":
            raw["payload"]["index"] = 0
        else:
            raw["payload"]["run_payload_sha256"] = "0" * 64
        raw["payload_sha256"] = domain_sha256(
            storage._ANSWER_DOMAIN, canonical_json_bytes(raw["payload"])
        )
        path.write_bytes(canonical_json_file_bytes(raw))
    saved = _saved(directory)
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, directory, resume=True)
    assert not seen and _saved(directory) == saved


@pytest.mark.parametrize(
    "control", ["max_artifact_bytes", "max_record_bytes", "max_storage_bytes", "resume"]
)
def test_controls_require_exact_types_before_inference(tmp_path: Path, control: str) -> None:
    fixture = _fixture(tmp_path)
    changes: dict[str, Any] = {control: 1 if control == "resume" else True}
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, tmp_path / "run", **changes)
    assert not seen


def test_conflicting_budget_and_native_owner_cannot_infer(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture):
        _run(fixture, directory)
    before = _saved(directory)
    with _inference(fixture) as seen:
        with pytest.raises(IrisStudyGenerationError):
            _run(fixture, directory, resume=True, max_storage_bytes=20 * 1024 * 1024)
        with process_file_lock(directory / ".lock"), pytest.raises(IrisStudyGenerationError):
            _run(fixture, directory, resume=True)
        assert not seen
    assert _saved(directory) == before
