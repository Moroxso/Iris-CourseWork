"""Public durable-generation exports and explicit controls, without an implicit launch policy."""

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

import iris.application as application
import iris.artifacts as artifacts
from iris.application import study_generation_run as service
from iris.application import study_generation_run_io as storage
from iris.application.errors import IrisStudyGenerationError
from iris.artifacts import study_generation_run_v1 as schemas
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_generation_run import _inference, _run


def test_public_exports_reference_the_implemented_services_and_schemas() -> None:
    for name in service.__all__:
        assert name in application.__all__
        assert getattr(application, name) is getattr(service, name)
    assert application.read_study_generation_run_v1 is storage.read_study_generation_run_v1
    assert "read_study_generation_run_v1" in application.__all__
    for name in schemas.__all__:
        assert name in artifacts.__all__
        assert getattr(artifacts, name) is getattr(schemas, name)


def test_public_generation_has_no_implicit_budget_mode_or_decoding_defaults() -> None:
    parameters = inspect.signature(service.generate_study_split_run_v1).parameters
    assert set(parameters) == {
        "benchmark",
        "config",
        "arm_id",
        "checkpoint_identity",
        "split",
        "decoding",
        "terminal_token_id",
        "max_artifact_bytes",
        "max_record_bytes",
        "max_storage_bytes",
        "directory",
        "resume",
    }
    assert all(value.default is inspect.Parameter.empty for value in parameters.values())


def test_generation_failure_redacts_private_exception_text(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with (
        _inference(fixture),
        patch.object(
            service,
            "_iter_study_generation_answers",
            side_effect=OSError("PRIVATE_INFERENCE_BYTES"),
        ),
        pytest.raises(IrisStudyGenerationError) as caught,
    ):
        _run(fixture, tmp_path / "run")
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__suppress_context__
