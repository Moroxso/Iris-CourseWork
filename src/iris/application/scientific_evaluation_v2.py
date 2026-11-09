"""Source-bound scoring of complete scientific captures, without loading a model."""

import os
import shutil
from contextlib import suppress
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _local_path,
)
from iris.application.scientific_evaluation import IrisScientificEvaluationError, _audit_inputs
from iris.application.scientific_generation import read_scientific_generation_v1
from iris.application.synthetic_scoring import _ConflictContent, _content
from iris.application.synthetic_scoring_v2 import (
    IrisSyntheticAnswerScoreV2,
    score_synthetic_answer_v2,
)
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    ScientificFamilyQualityV1,
    scientific_quality_counts_v1,
)
from iris.artifacts.scientific_evaluation_v2 import (
    IrisScientificEvaluationV2,
    ScientificAnswerMeasurementsV2,
    ScientificCodeEvidenceV2,
    ScientificEvaluationPayloadV2,
    ScientificPythonLimitsV2,
    ScientificScoredRowV2,
)
from iris.artifacts.scientific_generation_v1 import (
    ScientificGeneratedRowV1,
)
from iris.data.schemas import IrisExample, parse_structured_response
from iris.integrations.python_sandbox import _worker
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-evaluation/v2"
_MAX_BYTES = 32 * 1024**2
_FILE = "evaluation.json"


def _answer_measurements(
    row: IrisExample,
    *,
    generation_example_sha256: str,
    output_text: str,
    score: IrisSyntheticAnswerScoreV2,
    policy: PythonSandboxPolicyV2 | None,
) -> ScientificAnswerMeasurementsV2:
    output = output_text.encode("utf-8")
    if (
        type(score) is not IrisSyntheticAnswerScoreV2
        or score.example_id != row.id
        or score.task_type != row.task_type
        or score.split != "test"
        or score.scoring_profile != "iris-synthetic-01-scoring-v2"
        or score.example_sha256
        != domain_sha256(
            "iris/synthetic-example/v1",
            canonical_json_bytes(row.model_dump(mode="json")),
        )
        or score.output_sha256 != sha256(output).hexdigest()
        or score.output_byte_length != len(output)
        or score.case_count != len(row.code_cases)
    ):
        raise ValueError("Scorer did not return the exact submitted row/output binding")
    expected = parse_structured_response(row.expected_response)
    try:
        answer = parse_structured_response(output_text)
    except ValueError:
        answer = None
    exact = answer is not None and (answer.decision, answer.content, answer.next_action) == (
        expected.decision,
        expected.content,
        expected.next_action,
    )
    tool = None
    if row.task_type == "next_tool_prediction":
        tool = answer is not None and answer.next_action == expected.next_action
    conflict = None
    if row.task_type == "conflict_resolution":
        conflict = False
        if answer is not None and answer.decision == "conflict":
            with suppress(ValueError, TypeError, RecursionError, OverflowError):
                actual_content = _content(row.task_type, answer)
                expected_content = _content(row.task_type, expected)
                if (
                    type(actual_content) is _ConflictContent
                    and type(expected_content) is _ConflictContent
                ):
                    conflict = actual_content.conflict == expected_content.conflict
    evidence = None
    if score.execution is not None:
        evidence = ScientificCodeEvidenceV2.model_validate(asdict(score.execution))
        if answer is None or policy is None or evidence.policy != policy:
            raise ValueError("Unexpected Python execution evidence")
        submitted = canonical_json_bytes(
            {
                "kind": "python_integer_list_v1",
                "source": answer.content,
                "inputs": [list(case.inputs) for case in row.code_cases],
            }
        )
        if (
            evidence.submitted_input_sha256 != sha256(submitted).hexdigest()
            or evidence.worker_sha256 != sha256(_worker("python_worker_v2.txt")).hexdigest()
        ):
            raise ValueError("Code evidence binds another source/input/worker")
    return ScientificAnswerMeasurementsV2(
        example_id=row.id,
        generation_example_sha256=generation_example_sha256,
        scoring_example_sha256=score.example_sha256,
        task_type=row.task_type,
        output_sha256=score.output_sha256,
        output_byte_length=score.output_byte_length,
        outcome=score.outcome,
        reason=score.reason,
        envelope_valid=answer is not None,
        structured_exact_match=exact,
        tool_action_match=tool,
        conflict_flag_match=conflict,
        case_count=score.case_count,
        passed_case_count=score.passed_case_count,
        execution=evidence,
    )


def _scored_row(
    row: IrisExample,
    generated: ScientificGeneratedRowV1,
    score: IrisSyntheticAnswerScoreV2,
    policy: PythonSandboxPolicyV2 | None,
) -> ScientificScoredRowV2:
    measurements = _answer_measurements(
        row,
        generation_example_sha256=generated.example_sha256,
        output_text=generated.output_text,
        score=score,
        policy=policy,
    )
    return ScientificScoredRowV2(
        **measurements.model_dump(mode="python"),
        target_token_count=generated.target_token_count,
        target_nll_sum=generated.target_nll_sum,
    )


def evaluate_scientific_generation_v2(
    *,
    generation_path: Path,
    execution_path: Path,
    config_path: Path,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
) -> IrisScientificEvaluationV2:
    """Score a complete source-bound capture; no model load or regenerated answers.

    NLL is carried from the bound capture, not independently recomputed here.
    Source identities are consistency checks, not an attestation of generation.
    Any isolation infrastructure failure aborts this report without a model score.
    """
    try:
        test = ScientificParquetInputV1.model_validate(test.model_dump(mode="python"))
        capture = read_scientific_generation_v1(generation_path)
        rows, snapshot = _audit_inputs(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        has_code = any(row.executable for row in rows)
        if has_code != (python_policy is not None):
            raise ValueError(
                "Explicit sandbox policy required exactly when repair rows are present"
            )
        if python_policy is not None:
            if type(python_policy) is not PythonSandboxPolicyV2:
                raise ValueError("Exact sandbox policy required")
            python_policy = PythonSandboxPolicyV2.model_validate(
                python_policy.model_dump(mode="python")
            )
            if python_policy.wall_time_ms != 5000:
                raise ValueError("Repair wall budget must be exactly five seconds")
        scored = []
        for row, generated in zip(rows, capture.payload.rows, strict=True):
            policy = python_policy if row.executable else None
            score = score_synthetic_answer_v2(
                row, output_text=generated.output_text, python_policy=policy
            )
            scored.append(_scored_row(row, generated, score, policy))
        records = tuple(scored)
        # Re-audit recipe/data bindings too, not just the journal and output file.
        final_rows, final_snapshot = _audit_inputs(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        if (
            final_snapshot != snapshot
            or read_scientific_generation_v1(generation_path) != capture
            or final_rows != rows
        ):
            raise ValueError("Source/capture/test changed during evaluation")
        source = capture.payload
        payload = ScientificEvaluationPayloadV2(
            purpose=source.purpose,
            python_resource_scope="source_section15_v2" if has_code else "not_applicable_no_code",
            python_limits=ScientificPythonLimitsV2() if has_code else None,
            generation_sha256=capture.sha256,
            execution_sha256=source.execution_sha256,
            terminal_event_sha256=source.terminal_event_sha256,
            best_checkpoint_sha256=source.best_checkpoint_sha256,
            selected_step=source.selected_step,
            test_parquet_sha256=source.test_parquet_sha256,
            test_corpus_sha256=source.test_corpus_sha256,
            python_policy=python_policy,
            rows=records,
            overall=scientific_quality_counts_v1(records),
            families=tuple(
                ScientificFamilyQualityV1(
                    task_type=task,
                    counts=scientific_quality_counts_v1(
                        tuple(row for row in records if row.task_type == task)
                    ),
                )
                for task in TASKS
            ),
        )
        result = IrisScientificEvaluationV2(
            sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        _checked(result)
        return result
    except IrisCodeEvaluationUnavailableError:
        raise IrisCodeEvaluationUnavailableError(
            "Scientific Python evaluation infrastructure is unavailable; "
            "no complete score was produced"
        ) from None
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific answer evaluation did not complete"
        ) from None


def _checked(result: IrisScientificEvaluationV2) -> bytes:
    if type(result) is not IrisScientificEvaluationV2:
        raise ValueError("Exact scientific evaluation required")
    result = IrisScientificEvaluationV2.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Scientific evaluation identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Scientific evaluation exceeds its byte budget")
    return content


def read_scientific_evaluation_v2(path: Path) -> IrisScientificEvaluationV2:
    """Verify canonical hashes/counts, not source history or execution authenticity."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Evaluation directory inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificEvaluationV2.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Evaluation canonical identity differs")
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific evaluation report could not be read"
        ) from None


def write_scientific_evaluation_v2(
    result: IrisScientificEvaluationV2,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish a bounded hash-only report; never overwrite or remove old reports."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded evaluation store quota required")
        root = _local_path(root) / "scientific-evaluation-v2"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_evaluation_v2(target) != result:
                raise ValueError("Existing evaluation report conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        if (
            occupied + len(content) > maximum_storage_bytes
            or shutil.disk_usage(root).free < len(content) + 1024**3
        ):
            raise ValueError("Evaluation quota/free-space reserve exceeded")
        candidate = root / f".iris-evaluation-v2-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged evaluation differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_evaluation_v2(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific evaluation report could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "IrisScientificEvaluationError",
    "evaluate_scientific_generation_v2",
    "read_scientific_evaluation_v2",
    "write_scientific_evaluation_v2",
]
