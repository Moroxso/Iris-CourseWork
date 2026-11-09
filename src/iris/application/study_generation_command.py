"""Explicit file-backed durable generation and content-redacted read-only verification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import Field, TypeAdapter

from iris.application.arm_request_command_v4 import (
    _require_checkpoint_identity,
    require_arm_request_command_profile_v4,
)
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import IrisStudyGenerationError
from iris.application.study_benchmark_io import read_study_benchmark_v1
from iris.application.study_dataset_io import (
    _read_regular_bytes,
    _reject_duplicate_keys,
    _reject_nonfinite,
)
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_generation_run import (
    IrisStudyGenerationRunResultV1,
    generate_study_split_run_v1,
    verify_study_generation_run_v1,
)
from iris.application.study_generation_run_io import _decode, read_study_generation_run_v1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_generation_run_v1 import (
    IrisStudyGenerationRunV1,
    StudyGenerationPlannedAnswerV1,
)
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.artifacts.study_v1 import StudyArmIdV1, StudyTaskFamilyV1
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_file_bytes

_CONFIG_BYTES = 1024 * 1024
_DECODING_BYTES = 64 * 1024 * 1024
_DECODING: TypeAdapter[tuple[StudyGenerationPlannedAnswerV1, ...]] = TypeAdapter(
    Annotated[tuple[StudyGenerationPlannedAnswerV1, ...], Field(min_length=5, max_length=100000)]
)


@dataclass(frozen=True, slots=True)
class IrisStudyGenerationFileReportV1:
    """Private identities/counts, not prompts, IDs, answers, seeds or quality claims."""

    directory: Path
    path: Path
    run_payload_sha256: str
    generation_payload_sha256: str
    benchmark_payload_sha256: str
    configuration_sha256: str
    input_preflight_sha256: str
    arm_id: StudyArmIdV1
    split: ExperimentSplitV1
    total_examples: int
    by_family: tuple[tuple[StudyTaskFamilyV1, int], ...]
    generated_now: int | None
    reused_answers: int | None
    retained_file_bytes: int | None
    verification_scope: Literal["complete_bindings_not_historical_execution_attestation"]
    scientific_launch_status: Literal["not_assessed"]


def _paths(*paths: Path) -> None:
    # Never resolve away traversal or reparse parents before the owned reader checks them.
    if any(
        not isinstance(path, Path) or not path.is_absolute() or not path.name or ".." in path.parts
        for path in paths
    ):
        raise ValueError("explicit absolute non-root Paths without traversal are required")


def _inputs(
    configuration_path: Path,
    decoding_path: Path,
    benchmark: IrisStudyBenchmarkV1,
    split: ExperimentSplitV1,
) -> tuple[IrisBootstrapConfig, tuple[StudyGenerationPlannedAnswerV1, ...], bytes, bytes]:
    config_bytes = _read_regular_bytes(configuration_path, max_bytes=_CONFIG_BYTES)
    # Canonical model round-trip requires EVERY serialized field, including defaults.
    config = _decode(config_bytes, IrisBootstrapConfig)
    if not config.backbone.local_files_only:
        raise ValueError("generation configuration must explicitly forbid downloads")
    decoding_bytes = _read_regular_bytes(decoding_path, max_bytes=_DECODING_BYTES)
    raw = json.loads(
        decoding_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if canonical_json_file_bytes(raw) != decoding_bytes:
        raise ValueError("decoding input must use canonical iris-json-v1")
    planned = _DECODING.validate_json(decoding_bytes, strict=True)
    if (
        canonical_json_file_bytes([item.model_dump(mode="json") for item in planned])
        != decoding_bytes
    ):
        raise ValueError("decoding input must explicitly provide every policy field")
    dataset = benchmark.payload.dataset.payload
    expected = tuple(
        (example.example_id, assignment.task_family)
        for example, assignment in zip(
            dataset.manifest.payload.examples, dataset.assignments, strict=True
        )
        if example.split == split
    )
    if tuple((item.example_id, item.task_family) for item in planned) != expected:
        raise ValueError("decoding membership must exactly match the complete ordered split")
    first = planned[0].decoding.model_dump(exclude={"sampling_seed"})
    if any(
        item.decoding.allow_download or item.decoding.model_dump(exclude={"sampling_seed"}) != first
        for item in planned
    ):
        raise ValueError(
            "only explicit per-example sampling seeds may vary; downloads are forbidden"
        )
    return config, planned, config_bytes, decoding_bytes


def _report(
    directory: Path,
    run: IrisStudyGenerationRunV1,
    generation: IrisStudyGenerationV1,
    execution: IrisStudyGenerationRunResultV1 | None,
) -> IrisStudyGenerationFileReportV1:
    return IrisStudyGenerationFileReportV1(
        directory=directory,
        path=directory / "generation.json",
        run_payload_sha256=run.payload_sha256,
        generation_payload_sha256=generation.payload_sha256,
        benchmark_payload_sha256=generation.payload.benchmark_payload_sha256,
        configuration_sha256=run.payload.configuration_sha256,
        input_preflight_sha256=run.payload.input_preflight_sha256,
        arm_id=generation.payload.arm_id,
        split=generation.payload.split,
        total_examples=len(generation.payload.answers),
        by_family=tuple(
            (family, sum(answer.task_family == family for answer in generation.payload.answers))
            for family in get_args(StudyTaskFamilyV1)
        ),
        generated_now=None if execution is None else execution.generated_now,
        reused_answers=None if execution is None else execution.reused_answers,
        retained_file_bytes=None if execution is None else execution.retained_file_bytes,
        verification_scope="complete_bindings_not_historical_execution_attestation",
        scientific_launch_status="not_assessed",
    )


def _unchanged_outputs(
    directory: Path,
    run: IrisStudyGenerationRunV1,
    generation: IrisStudyGenerationV1,
) -> None:
    if (
        read_study_generation_run_v1(directory) != run
        or read_study_generation_v1(directory / "generation.json") != generation
    ):
        raise ValueError("selected generation output changed during the command")


def generate_study_split_files_v1(
    *,
    benchmark_path: Path,
    configuration_path: Path,
    decoding_path: Path,
    arm_id: object,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    split: ExperimentSplitV1,
    terminal_token_id: int,
    directory: Path,
    max_artifact_bytes: int,
    max_record_bytes: int,
    max_storage_bytes: int,
    resume: bool,
) -> IrisStudyGenerationFileReportV1:
    """Run ADR 0051 with explicit original input files; no downloads, scoring or seed defaults."""
    try:
        _paths(benchmark_path, configuration_path, decoding_path, directory)
        if (
            type(resume) is not bool
            or type(split) is not str
            or split not in get_args(ExperimentSplitV1)
        ):
            raise ValueError("explicit generation mode and split are required")
        if type(terminal_token_id) is not int or terminal_token_id < 0:
            raise ValueError("terminal token must be an exact nonnegative integer")
        for value, lower, upper in (
            (max_artifact_bytes, 1024, 64 * 1024**2),
            (max_record_bytes, 4096, 64 * 1024**2),
            (max_storage_bytes, 4096, 20 * 1024**3),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("explicit generation budget is outside its supported range")
        profile = require_arm_request_command_profile_v4(arm_id)
        _require_checkpoint_identity(profile=profile, identity=checkpoint_identity)
        for path in (benchmark_path, configuration_path, decoding_path):
            if directory == path or directory in path.parents:
                raise ValueError("original input files must be outside the managed run")
        benchmark = read_study_benchmark_v1(benchmark_path)
        config, planned, config_bytes, decoding_bytes = _inputs(
            configuration_path, decoding_path, benchmark, split
        )
        execution = generate_study_split_run_v1(
            benchmark,
            config=config,
            arm_id=profile.arm_id,
            checkpoint_identity=checkpoint_identity,
            split=split,
            decoding=tuple(item.decoding for item in planned),
            terminal_token_id=terminal_token_id,
            max_artifact_bytes=max_artifact_bytes,
            max_record_bytes=max_record_bytes,
            max_storage_bytes=max_storage_bytes,
            directory=directory,
            resume=resume,
        )
        # The runner already performed whole-run verification. Bind its exact returned identities.
        run = read_study_generation_run_v1(directory)
        generation = read_study_generation_v1(directory / "generation.json")
        if (
            run.payload_sha256 != execution.run_payload_sha256
            or generation.payload_sha256 != execution.generation_payload_sha256
        ):
            raise ValueError("published generation identity changed")
        if (
            read_study_benchmark_v1(benchmark_path) != benchmark
            or _read_regular_bytes(configuration_path, max_bytes=_CONFIG_BYTES) != config_bytes
            or _read_regular_bytes(decoding_path, max_bytes=_DECODING_BYTES) != decoding_bytes
        ):
            raise ValueError("selected generation inputs changed during the command")
        _unchanged_outputs(directory, run, generation)
        return _report(directory, run, generation, execution)
    except Exception:
        raise IrisStudyGenerationError(
            "Iris study generation file command did not complete"
        ) from None


def verify_study_generation_files_v1(
    *,
    benchmark_path: Path,
    directory: Path,
) -> IrisStudyGenerationFileReportV1:
    """Verify the complete saved plan/answers/dependencies; no ownership, repairs or inference."""
    try:
        _paths(benchmark_path, directory)
        benchmark = read_study_benchmark_v1(benchmark_path)
        run = read_study_generation_run_v1(directory)
        generation = verify_study_generation_run_v1(directory, benchmark=benchmark)
        if read_study_benchmark_v1(benchmark_path) != benchmark:
            raise ValueError("selected benchmark changed during generation verification")
        _unchanged_outputs(directory, run, generation)
        return _report(directory, run, generation, None)
    except Exception:
        raise IrisStudyGenerationError(
            "Iris study generation files could not be independently verified"
        ) from None


__all__ = [
    "IrisStudyGenerationFileReportV1",
    "generate_study_split_files_v1",
    "verify_study_generation_files_v1",
]
