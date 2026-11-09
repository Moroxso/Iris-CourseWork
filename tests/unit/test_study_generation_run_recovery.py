"""Initialization, corrupt pending data, conservative budgets and read-only boundaries."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import study_generation_run as service
from iris.application import study_generation_run_io as storage
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_generation_run import _inference, _run, _saved


@pytest.mark.parametrize("length", [0, 23, None])
def test_interrupted_plan_initialization_recovers_only_the_exact_plan(
    tmp_path: Path, length: int | None
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"

    def partial(pending: Path, content: bytes, **kwargs: Any) -> None:
        assert pending.name == ".run-pending.json"
        pending.write_bytes(content if length is None else content[:length])
        raise KeyboardInterrupt

    with _inference(fixture) as seen:
        with patch.object(storage, "_stage", side_effect=partial), pytest.raises(KeyboardInterrupt):
            _run(fixture, directory)
        assert not seen and not (directory / "run.json").exists()
        result = _run(fixture, directory, resume=True)
        assert result.generated_now == 5 and seen == list(range(5))
    if length is not None:
        archived = tuple((directory / "interrupted").iterdir())
        assert len(archived) == 1 and archived[0].stat().st_size == length


@pytest.mark.parametrize("change", ["hash", "run", "output", "index"])
def test_complete_but_corrupt_pending_answer_is_never_archived_or_reinferred(
    tmp_path: Path, change: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    original_stage = storage._stage

    def stop(pending: Path, content: bytes, **kwargs: Any) -> None:
        original_stage(pending, content, **kwargs)
        if pending.name == ".answer-pending.json" and json.loads(content)["payload"]["index"] == 2:
            raise KeyboardInterrupt

    with (
        _inference(fixture),
        patch.object(service, "_stage", side_effect=stop),
        pytest.raises(KeyboardInterrupt),
    ):
        _run(fixture, directory)
    path = directory / ".answer-pending.json"
    raw = json.loads(path.read_bytes())
    if change == "hash":
        raw["payload_sha256"] = "0" * 64
    else:
        if change == "run":
            raw["payload"]["run_payload_sha256"] = "0" * 64
        elif change == "index":
            raw["payload"]["index"] = 1
        else:
            raw["payload"]["answer"].update(
                output_text="altered", output_byte_length=7, output_sha256=bytes_sha256(b"altered")
            )
        raw["payload_sha256"] = domain_sha256(
            storage._ANSWER_DOMAIN, canonical_json_bytes(raw["payload"])
        )
    path.write_bytes(canonical_json_file_bytes(raw))
    before = _saved(directory)
    with _inference(fixture) as seen:
        with pytest.raises(IrisStudyGenerationError):
            _run(fixture, directory, resume=True)
        with pytest.raises(IrisStudyGenerationError):
            service.verify_study_generation_run_v1(directory, benchmark=fixture[1])
        assert not seen
    assert _saved(directory) == before


def test_reserve_prevents_inference_without_room_for_a_full_record_and_final_output(
    tmp_path: Path,
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, directory, max_storage_bytes=4 * 1024 * 1024)
    assert not seen and (directory / "run.json").is_file()
    assert not (directory / "answers").exists()


def test_record_cap_is_enforced_without_publishing_a_false_complete_answer(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, directory, max_record_bytes=4096)
    assert seen == [0]
    assert not (directory / "answers").exists()
    assert not (directory / ".answer-pending.json").exists()
    assert not (directory / "generation.json").exists()


def test_final_generation_is_reread_after_record_verification(tmp_path: Path) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture):
        _run(fixture, directory)
    original = service._unchanged_records

    def mutate(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        (directory / "generation.json").write_bytes(b"changed after first final-generation read")

    with (
        _inference(fixture) as seen,
        patch.object(service, "_unchanged_records", side_effect=mutate),
        pytest.raises(IrisStudyGenerationError),
    ):
        service.verify_study_generation_run_v1(directory, benchmark=fixture[1])
    assert not seen


def test_request_store_cannot_be_inside_managed_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, tmp_path)
    assert not seen


@pytest.mark.parametrize(
    "change", ["configuration", "benchmark", "decoding", "record_budget", "artifact_budget"]
)
def test_resume_rejects_changed_inputs_before_inference(tmp_path: Path, change: str) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    with _inference(fixture):
        _run(fixture, directory)
    original = _saved(directory)
    arguments: dict[str, Any] = {}
    if change == "configuration":
        first = fixture[0].payload.answers[0]
        config = read_request_artifact_v4(
            fixture[2] / "requests" / "v4" / f"{first.request_payload_sha256}.json"
        ).payload.configuration.value
        config.runtime.seed += 1
        arguments["config"] = config
    elif change == "benchmark":
        fixture = (
            fixture[0],
            build_synthetic_study_pilot_v1(seed=1908, problems_per_cell=1),
            fixture[2],
        )
    elif change == "decoding":
        arguments["decoding"] = tuple(
            DecodingProvenanceV1.model_validate({**a.decoding.model_dump(), "temperature": 0.5})
            for a in fixture[0].payload.answers
        )
    elif change == "record_budget":
        arguments["max_record_bytes"] = 2 * 1024 * 1024
    else:
        arguments["max_artifact_bytes"] = 2 * 1024 * 1024
    with _inference(fixture) as seen, pytest.raises(IrisStudyGenerationError):
        _run(fixture, directory, resume=True, **arguments)
    assert not seen and _saved(directory) == original
