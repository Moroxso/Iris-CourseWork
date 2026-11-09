"""Owned benchmark integrity and deterministic standalone text-answer scoring."""

from dataclasses import dataclass
from typing import Literal

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyBenchmarkError
from iris.application.study_dataset import validate_study_dataset_v1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.study_benchmark_v1 import (
    IrisStudyBenchmarkV1,
    StudyBenchmarkPayloadV1,
    StudyBenchmarkSchemaV1,
    StudyExampleScoringV1,
    StudyPythonIntegerListRuleV1,
)
from iris.artifacts.study_dataset_v1 import IrisStudyDatasetV1
from iris.artifacts.study_v1 import StudyTaskFamilyV1
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)

_PAYLOAD_DOMAIN = "iris/study-benchmark/v1/payload"
_MAX_BENCHMARK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IrisStudyAnswerScoreV1:
    """Standalone score, not a request-bound evaluation receipt or a four-arm result."""

    benchmark_payload_sha256: str
    example_id: str
    task_family: StudyTaskFamilyV1
    split: ExperimentSplitV1
    output_sha256: str
    output_byte_length: int
    outcome: Literal["correct", "incorrect", "invalid_output"]


def _validated(benchmark: IrisStudyBenchmarkV1) -> IrisStudyBenchmarkV1:
    if type(benchmark) is not IrisStudyBenchmarkV1:
        raise TypeError("benchmark must be exactly IrisStudyBenchmarkV1")
    owned = IrisStudyBenchmarkV1.model_validate(benchmark.model_dump(mode="python"))
    validate_study_dataset_v1(owned.payload.dataset)
    if owned.payload_sha256 != domain_sha256(
        _PAYLOAD_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("benchmark payload digest is inconsistent")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > _MAX_BENCHMARK_BYTES:
        raise ValueError("benchmark exceeds the 64 MiB safety limit")
    return owned


def build_study_benchmark_v1(
    *, dataset: IrisStudyDatasetV1, scoring: tuple[StudyExampleScoringV1, ...]
) -> IrisStudyBenchmarkV1:
    """Bind every example to a scoring rule without changing dataset membership."""
    try:
        if type(scoring) is not tuple or any(
            type(item) is not StudyExampleScoringV1 for item in scoring
        ):
            raise TypeError("scoring must be an exact tuple of StudyExampleScoringV1")
        payload = StudyBenchmarkPayloadV1(
            dataset=validate_study_dataset_v1(dataset), scoring=scoring
        )
        return _validated(
            IrisStudyBenchmarkV1(
                schema_identity=StudyBenchmarkSchemaV1(
                    artifact_type="iris.study_benchmark",
                    version=1,
                    canonicalization="iris-json-v1",
                    hash_algorithm="sha256",
                    content_policy="private_inputs_targets_and_code_cases",
                ),
                payload_sha256=domain_sha256(
                    _PAYLOAD_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
        )
    except Exception:
        raise IrisStudyBenchmarkError("Iris study benchmark could not be built") from None


def validate_study_benchmark_v1(benchmark: IrisStudyBenchmarkV1) -> IrisStudyBenchmarkV1:
    """Recheck all three nested payload identities and take deep ownership."""
    try:
        return _validated(benchmark)
    except Exception:
        raise IrisStudyBenchmarkError("Iris study benchmark could not be verified") from None


def score_study_answer_v1(
    benchmark: IrisStudyBenchmarkV1, *, example_id: str, output_text: str
) -> IrisStudyAnswerScoreV1:
    """Legacy exact-text scoring API; code requires the separate explicit-policy API.

    No candidate code, reference code, subprocess, imports from candidate text,
    or shell command is executed. A missing execution backend is an infrastructure
    error, never a zero score attributed to a model's Python answer.
    Use score_study_python_answer_v1 to explicitly opt into the isolated backend.
    """
    try:
        if type(example_id) is not str or type(output_text) is not str:
            raise TypeError("example ID and output text must be exact strings")
        owned = _validated(benchmark)
        dataset = owned.payload.dataset.payload
        for example, assignment, scoring in zip(
            dataset.manifest.payload.examples,
            dataset.assignments,
            owned.payload.scoring,
            strict=True,
        ):
            if example.example_id != example_id:
                continue
            if isinstance(scoring.rule, StudyPythonIntegerListRuleV1):
                raise IrisCodeEvaluationUnavailableError(
                    "Functional Python scoring requires an explicitly isolated backend"
                )
            encoded = output_text.encode("utf-8", errors="strict")
            normalized = output_text.strip(" \t\r\n")
            outcome: Literal["correct", "incorrect", "invalid_output"]
            if not normalized or "\0" in output_text or len(encoded) > 1024 * 1024:
                outcome = "invalid_output"
            elif normalized == example.target.strip(" \t\r\n"):
                outcome = "correct"
            else:
                outcome = "incorrect"
            return IrisStudyAnswerScoreV1(
                benchmark_payload_sha256=owned.payload_sha256,
                example_id=example_id,
                task_family=assignment.task_family,
                split=example.split,
                output_sha256=bytes_sha256(encoded),
                output_byte_length=len(encoded),
                outcome=outcome,
            )
        raise ValueError("example is absent from the benchmark")
    except IrisCodeEvaluationUnavailableError:
        raise
    except Exception:
        raise IrisStudyBenchmarkError("Iris study answer could not be scored") from None


__all__ = [
    "IrisStudyAnswerScoreV1",
    "build_study_benchmark_v1",
    "score_study_answer_v1",
    "validate_study_benchmark_v1",
]
