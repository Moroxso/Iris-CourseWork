"""File-backed evaluation commands with explicit policies and content-redacted reports."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.study_benchmark_io import read_study_benchmark_v1
from iris.application.study_evaluation_run import (
    IrisStudyEvaluationRunResultV1,
    evaluate_study_split_v1,
    verify_study_evaluation_run_v1,
)
from iris.application.study_evaluation_run_io import read_study_evaluation_summary_v1
from iris.application.study_generation_io import read_study_generation_v1
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_evaluation_run_v1 import (
    IrisStudyEvaluationSummaryV1,
    StudyEvaluationCountsV1,
    StudyEvaluationFamilyCountsV1,
)
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.artifacts.study_v1 import StudyArmIdV1


@dataclass(frozen=True, slots=True)
class IrisStudyEvaluationFileReportV1:
    """Private paths/hashes/counts only, never example IDs, answers or observations."""

    directory: Path
    generation_payload_sha256: str
    benchmark_payload_sha256: str
    run_payload_sha256: str
    summary_payload_sha256: str
    arm_id: StudyArmIdV1
    split: ExperimentSplitV1
    overall: StudyEvaluationCountsV1
    by_family: tuple[StudyEvaluationFamilyCountsV1, ...]
    evaluated_now: int | None
    reused_evaluations: int | None
    retained_file_bytes: int | None
    verification_scope: Literal["bindings_and_recomputed_scores_not_execution_attestation"]
    scientific_launch_status: Literal["not_assessed"]


def _paths(*paths: Path) -> None:
    # Do not resolve away a symlink or parent traversal before the storage checks.
    if any(
        not isinstance(path, Path) or not path.is_absolute() or not path.name or ".." in path.parts
        for path in paths
    ):
        raise ValueError("explicit absolute non-root Paths without traversal are required")


def _unchanged_inputs(
    generation_path: Path,
    benchmark_path: Path,
    generation: IrisStudyGenerationV1,
    benchmark: IrisStudyBenchmarkV1,
) -> None:
    # The core verifies owned values and request/checkpoint dependencies. The
    # file command also checks that its selected input files still name them.
    if (
        read_study_generation_v1(generation_path) != generation
        or read_study_benchmark_v1(benchmark_path) != benchmark
    ):
        raise ValueError("selected evaluation inputs changed during the command")


def _report(
    directory: Path,
    generation: IrisStudyGenerationV1,
    summary: IrisStudyEvaluationSummaryV1,
    execution: IrisStudyEvaluationRunResultV1 | None,
) -> IrisStudyEvaluationFileReportV1:
    return IrisStudyEvaluationFileReportV1(
        directory=directory,
        generation_payload_sha256=generation.payload_sha256,
        benchmark_payload_sha256=generation.payload.benchmark_payload_sha256,
        run_payload_sha256=summary.payload.run_payload_sha256,
        summary_payload_sha256=summary.payload_sha256,
        arm_id=generation.payload.arm_id,
        split=generation.payload.split,
        overall=summary.payload.overall,
        by_family=summary.payload.by_family,
        evaluated_now=None if execution is None else execution.evaluated_now,
        reused_evaluations=None if execution is None else execution.reused_evaluations,
        retained_file_bytes=None if execution is None else execution.retained_file_bytes,
        verification_scope=summary.payload.verification_scope,
        scientific_launch_status=summary.payload.scientific_launch_status,
    )


def evaluate_study_generation_file_v1(
    *,
    generation_path: Path,
    benchmark_path: Path,
    request_root: Path,
    directory: Path,
    code_profile: str,
    code_wall_time_ms: int,
    code_max_output_bytes: int,
    max_storage_bytes: int,
    resume: bool,
) -> IrisStudyEvaluationFileReportV1:
    """Evaluate or explicitly resume ADR 0048; no inference, installation or downloads."""
    try:
        _paths(generation_path, benchmark_path, request_root, directory)
        policy = PythonSandboxPolicyV1.model_validate(
            {
                "profile": code_profile,
                "wall_time_ms": code_wall_time_ms,
                "max_output_bytes": code_max_output_bytes,
            }
        )
        if type(resume) is not bool or type(max_storage_bytes) is not int:
            raise TypeError("resume and storage budget require exact types")
        if not 4096 <= max_storage_bytes <= 20 * 1024**3:
            raise ValueError("storage budget is outside the supported range")
        generation = read_study_generation_v1(generation_path)
        benchmark = read_study_benchmark_v1(benchmark_path)
        execution = evaluate_study_split_v1(
            generation,
            benchmark=benchmark,
            request_root=request_root,
            directory=directory,
            code_policy=policy,
            max_storage_bytes=max_storage_bytes,
            resume=resume,
        )
        # The runner independently verified this exact summary before returning.
        summary = read_study_evaluation_summary_v1(directory)
        if (
            summary.payload_sha256 != execution.summary_payload_sha256
            or summary.payload.run_payload_sha256 != execution.run_payload_sha256
        ):
            raise ValueError("published evaluation identity changed")
        _unchanged_inputs(generation_path, benchmark_path, generation, benchmark)
        return _report(directory, generation, summary, execution)
    except IrisCodeEvaluationUnavailableError:
        raise IrisCodeEvaluationUnavailableError(
            "Iris isolated Python evaluation is unavailable; no complete result was reported"
        ) from None
    except Exception:
        raise IrisStudyEvaluationError("Iris study evaluation command did not complete") from None


def verify_study_evaluation_files_v1(
    *, generation_path: Path, benchmark_path: Path, request_root: Path, directory: Path
) -> IrisStudyEvaluationFileReportV1:
    """Read-only full binding/regrading check; no runtime start, repairs or model weights."""
    try:
        _paths(generation_path, benchmark_path, request_root, directory)
        generation = read_study_generation_v1(generation_path)
        benchmark = read_study_benchmark_v1(benchmark_path)
        summary = verify_study_evaluation_run_v1(
            directory, generation=generation, benchmark=benchmark, request_root=request_root
        )
        _unchanged_inputs(generation_path, benchmark_path, generation, benchmark)
        return _report(directory, generation, summary, None)
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris study evaluation files could not be independently verified"
        ) from None


__all__ = [
    "IrisStudyEvaluationFileReportV1",
    "evaluate_study_generation_file_v1",
    "verify_study_evaluation_files_v1",
]
