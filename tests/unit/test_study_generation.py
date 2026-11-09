"""Complete private generation membership and independent request binding tests."""

from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import iris.application as application
import iris.artifacts as artifacts
from iris.application import study_generation as service
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_generation import generate_study_split_v1, verify_study_generation_v1
from iris.application.study_generation_io import (
    read_study_generation_v1,
    validate_study_generation_v1,
    write_study_generation_v1,
)
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_generation_v1 import (
    IrisStudyGenerationV1,
    StudyGeneratedAnswerV1,
    StudyGenerationPayloadV1,
    StudyGenerationSchemaV1,
)
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256
from tests.unit.test_study_evaluation import _request
from tests.unit.test_study_input_preflight import _assets, _tokenizer


def _fixture(tmp_path: Path) -> tuple[IrisStudyGenerationV1, IrisStudyBenchmarkV1, Path]:
    answers = []
    for index in range(5):
        benchmark, path, identity, output = _request(tmp_path, example_index=index)
        request = read_request_artifact_v4(path)
        assignment = benchmark.payload.dataset.payload.assignments[index]
        answers.append(
            StudyGeneratedAnswerV1(
                example_id=identity,
                task_family=assignment.task_family,
                request_payload_sha256=request.payload_sha256,
                decoding=request.payload.request.decoding,
                output_text=output,
                output_sha256=bytes_sha256(output.encode()),
                output_byte_length=len(output.encode()),
            )
        )
    with _assets(_tokenizer()):
        preflight = preflight_study_inputs_v1(
            benchmark,
            config=request.payload.configuration.value,
            terminal_token_id=1,
        )
    dataset = benchmark.payload.dataset.payload
    payload = StudyGenerationPayloadV1(
        benchmark_payload_sha256=benchmark.payload_sha256,
        dataset_payload_sha256=benchmark.payload.dataset.payload_sha256,
        manifest_payload_sha256=dataset.manifest.payload_sha256,
        protocol_declaration_payload_sha256=dataset.protocol_declaration_payload_sha256,
        configuration_sha256=preflight.configuration_sha256,
        execution_profile_set_sha256=preflight.execution_profile_set_sha256,
        input_preflight_sha256=domain_sha256(
            "iris/study-generation/v1/input-preflight", canonical_json_bytes(asdict(preflight))
        ),
        terminal_token_id=1,
        arm_id="A",
        split="train",
        checkpoint_transfer=None,
        max_artifact_bytes=1024 * 1024,
        answers=tuple(answers),
        membership_policy="all_examples_in_selected_split_in_manifest_order",
        status="complete_generation_not_scored",
        verification_scope="bindings_not_execution_attestation",
        scientific_launch_status="not_assessed",
    )
    generation = IrisStudyGenerationV1(
        schema_identity=StudyGenerationSchemaV1(
            artifact_type="iris.study_generation",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            content_policy="private_generated_answers_and_metadata",
        ),
        payload_sha256=domain_sha256(
            "iris/study-generation/v1/payload",
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    return generation, benchmark, tmp_path / "artifacts"


def _rehash(raw: dict[str, Any]) -> IrisStudyGenerationV1:
    raw["payload_sha256"] = domain_sha256(
        "iris/study-generation/v1/payload",
        canonical_json_bytes(raw["payload"]),
    )
    return IrisStudyGenerationV1.model_validate(raw)


def test_capture_preserves_all_families_raw_answers_and_canonical_round_trip(
    tmp_path: Path,
) -> None:
    generation, benchmark, root = _fixture(tmp_path)
    assert len(generation.payload.answers) == 5
    assert len({answer.task_family for answer in generation.payload.answers}) == 5
    assert generation.payload.answers[2].output_text.startswith("\tdef solve")
    with _assets(_tokenizer()):
        verified = verify_study_generation_v1(generation, benchmark=benchmark, request_root=root)
    assert verified == generation and verified is not generation
    path = tmp_path / "generation.json"
    first = write_study_generation_v1(verified, path=path)
    second = write_study_generation_v1(verified, path=path)
    assert not first.reused_existing and second.reused_existing
    assert read_study_generation_v1(path) == generation


@pytest.mark.parametrize("change", ["order", "split", "output", "request", "family", "preflight"])
def test_recomputed_outer_hash_does_not_prove_generation_bindings(
    tmp_path: Path, change: str
) -> None:
    generation, benchmark, root = _fixture(tmp_path)
    raw = generation.model_dump(mode="python")
    if change == "order":
        raw["payload"]["answers"] = tuple(reversed(raw["payload"]["answers"]))
    elif change == "split":
        raw["payload"]["split"] = "test"
    elif change == "output":
        answer = raw["payload"]["answers"][0]
        answer.update(
            output_text="altered", output_sha256=bytes_sha256(b"altered"), output_byte_length=7
        )
    elif change == "request":
        raw["payload"]["answers"][0]["request_payload_sha256"] = "0" * 64
    elif change == "family":
        first, second = raw["payload"]["answers"][:2]
        first["task_family"], second["task_family"] = second["task_family"], first["task_family"]
    else:
        raw["payload"]["input_preflight_sha256"] = "0" * 64
    forged = _rehash(raw)
    assert validate_study_generation_v1(forged) == forged
    with _assets(_tokenizer()), pytest.raises(IrisStudyGenerationError):
        verify_study_generation_v1(forged, benchmark=benchmark, request_root=root)


@pytest.mark.parametrize(
    "change",
    ["version", "output_hash", "duplicate", "drop_code", "seed", "budget", "arm", "policy"],
)
def test_schema_or_integrity_rejects_malformed_capture(tmp_path: Path, change: str) -> None:
    generation, _, _ = _fixture(tmp_path)
    raw = generation.model_dump(mode="python")
    if change == "version":
        raw["schema_identity"]["version"] = True
    elif change == "output_hash":
        raw["payload"]["answers"][0]["output_sha256"] = "0" * 64
    elif change == "duplicate":
        raw["payload"]["answers"][1]["example_id"] = raw["payload"]["answers"][0]["example_id"]
    elif change == "drop_code":
        raw["payload"]["answers"] = tuple(
            item
            for item in raw["payload"]["answers"]
            if item["task_family"] != "code_generation_and_verification"
        )
    elif change == "seed":
        raw["payload"]["answers"][0]["decoding"]["sampling_seed"] = True
    elif change == "budget":
        raw["payload"]["max_artifact_bytes"] = 1024
    elif change == "arm":
        raw["payload"]["arm_id"] = "C"
    else:
        raw["payload"]["answers"][0]["decoding"]["top_p"] = 0.5
    with pytest.raises((ValidationError, IrisStudyGenerationError)):
        validate_study_generation_v1(_rehash(raw))


def test_existing_destination_prevents_model_work_and_preserves_bytes(tmp_path: Path) -> None:
    generation, benchmark, root = _fixture(tmp_path)
    path = tmp_path / "existing.json"
    path.write_bytes(b"EXISTING_PRIVATE_BYTES")
    config = read_request_artifact_v4(
        root / "requests" / "v4" / f"{generation.payload.answers[0].request_payload_sha256}.json"
    ).payload.configuration.value
    with (
        patch.object(service, "bootstrap_arm_request_runtime_v4") as bootstrap,
        pytest.raises(IrisStudyGenerationError),
    ):
        generate_study_split_v1(
            benchmark,
            config=config,
            arm_id="A",
            checkpoint_identity=None,
            split="train",
            decoding=tuple(item.decoding for item in generation.payload.answers),
            terminal_token_id=1,
            max_artifact_bytes=1024 * 1024,
            path=path,
        )
    bootstrap.assert_not_called()
    assert path.read_bytes() == b"EXISTING_PRIVATE_BYTES"


@pytest.mark.parametrize(
    "change",
    [
        "missing_policy",
        "policy_list",
        "online",
        "budget_bool",
        "budget_small",
        "root",
        "split",
        "missing_checkpoint",
        "policy_drift",
    ],
)
def test_invalid_batch_inputs_fail_before_tokenizer_or_model_loading(
    tmp_path: Path, change: str
) -> None:
    generation, benchmark, root = _fixture(tmp_path)
    config = read_request_artifact_v4(
        root / "requests" / "v4" / f"{generation.payload.answers[0].request_payload_sha256}.json"
    ).payload.configuration.value
    args: dict[str, Any] = dict(
        config=config,
        arm_id="A",
        checkpoint_identity=None,
        split="train",
        decoding=tuple(item.decoding for item in generation.payload.answers),
        terminal_token_id=1,
        max_artifact_bytes=1024 * 1024,
        path=tmp_path / "new.json",
    )
    if change == "missing_policy":
        args["decoding"] = args["decoding"][:-1]
    elif change == "policy_list":
        args["decoding"] = list(args["decoding"])
    elif change == "online":
        args["decoding"] = tuple(
            item.model_copy(update={"allow_download": True}) for item in args["decoding"]
        )
    elif change == "budget_bool":
        args["max_artifact_bytes"] = True
    elif change == "budget_small":
        args["max_artifact_bytes"] = 1024
    elif change == "root":
        config.runtime.output_dir = "relative-root"
    elif change == "split":
        args["split"] = "missing"
    elif change == "missing_checkpoint":
        args["arm_id"] = "B"
    else:
        args["decoding"] = (
            args["decoding"][0].model_copy(update={"top_p": 0.5}),
            *args["decoding"][1:],
        )
    with (
        patch.object(service, "preflight_study_inputs_v1") as preflight,
        patch.object(service, "bootstrap_arm_request_runtime_v4") as bootstrap,
        pytest.raises(IrisStudyGenerationError),
    ):
        generate_study_split_v1(benchmark, **args)
    preflight.assert_not_called()
    bootstrap.assert_not_called()
    assert not (tmp_path / "new.json").exists()


def test_public_api_exposes_exact_generation_services_and_types() -> None:
    assert application.generate_study_split_v1 is generate_study_split_v1
    assert application.verify_study_generation_v1 is verify_study_generation_v1
    assert application.read_study_generation_v1 is read_study_generation_v1
    assert application.write_study_generation_v1 is write_study_generation_v1
    assert application.validate_study_generation_v1 is validate_study_generation_v1
    assert application.IrisStudyGenerationError is IrisStudyGenerationError
    assert artifacts.IrisStudyGenerationV1 is IrisStudyGenerationV1
    assert artifacts.StudyGeneratedAnswerV1 is StudyGeneratedAnswerV1
