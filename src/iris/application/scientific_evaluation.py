"""Source-bound scoring of complete scientific captures, without loading a model."""

import os
import shutil
from contextlib import suppress
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from iris.application.errors import ApplicationError, IrisCodeEvaluationUnavailableError
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular, _verify
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _local_path,
    _rows,
    read_scientific_command_config_v1,
)
from iris.application.scientific_execution import (
    IrisScientificExecutionSnapshotV1,
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_generation import _own_test_rows, read_scientific_generation_v1
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_validation import own_scientific_validation_rows_v1
from iris.application.synthetic_scoring import (
    IrisSyntheticAnswerScoreV1,
    _ConflictContent,
    _content,
    score_synthetic_answer_v1,
)
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    IrisScientificEvaluationV1,
    ScientificCodeEvidenceV1,
    ScientificEvaluationPayloadV1,
    ScientificFamilyQualityV1,
    ScientificScoredRowV1,
    scientific_quality_counts_v1,
)
from iris.artifacts.scientific_execution_v1 import ScientificExecutionCheckpointV1
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificGeneratedRowV1,
)
from iris.data.schemas import IrisExample, parse_structured_response
from iris.integrations.python_sandbox import _worker
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-evaluation/v1"
_MAX_BYTES = 32 * 1024**2
_FILE = "evaluation.json"


class IrisScientificEvaluationError(ApplicationError):
    """Incomplete or invalid evaluation, redacted of private input/output content."""


def _audit_inputs(
    capture: IrisScientificGenerationV1,
    *,
    config_path: Path,
    execution_path: Path,
    test: ScientificParquetInputV1,
) -> tuple[tuple[IrisExample, ...], IrisScientificExecutionSnapshotV1]:
    config = read_scientific_command_config_v1(config_path)
    snapshot = read_scientific_execution_v1(_local_path(execution_path))
    payload = capture.payload
    if not snapshot.events:
        raise ValueError("Terminal source execution required")
    terminal = snapshot.events[-1]
    if (
        terminal.payload.after.phase != "stopped"
        or type(terminal.payload.record) is not ScientificExecutionCheckpointV1
    ):
        raise ValueError("Source execution lacks its terminal checkpoint")
    selection = None
    for event in snapshot.events:
        selection = _selected_best(selection, event)
    _, best = _bundle_parts(
        _checkpoint_for_event(
            snapshot.path,
            snapshot.header,
            terminal,
            best_event=selection,
        )
    )
    if best is None or selection is None:
        raise ValueError("No saved validation-selected state")
    _, tensors = _verify(best)
    run_config = _json(best.manifest.payload.configuration_json.encode())
    lifecycle_config = _json(snapshot.header.payload.lifecycle_configuration_json.encode())
    teacher_hash = bytes(tensors["model/configuration_identity"].tolist()).hex()
    if (
        payload.execution_sha256 != snapshot.header.sha256
        or payload.terminal_event_sha256 != terminal.sha256
        or payload.best_validation_event_sha256 != selection.sha256
        or payload.best_checkpoint_sha256 != best.manifest.sha256
        or payload.selected_step != best.manifest.payload.completed_steps
        or payload.terminal_step != terminal.payload.after.step
        or payload.teacher_configuration_sha256 != teacher_hash
        or payload.backbone_runtime_sha256 != run_config["backbone_runtime_sha256"]
        or payload.purpose != config.optimizer.purpose
        or run_config["optimizer_policy"] != config.optimizer.model_dump(mode="json")
        or run_config["teacher_policy"] != config.objective.model_dump(mode="json")
        or lifecycle_config["policy"] != config.lifecycle.model_dump(mode="json")
        or run_config["seed"] != config.run_seed
        or run_config["terminal_token_id"] != config.terminal_token_id
        or config.terminal_token_id not in payload.eos_token_ids
        or payload.max_new_tokens != config.backbone.limits.max_new_tokens
        or config.maximum_journal_bytes != snapshot.header.payload.maximum_journal_bytes
        or domain_sha256(
            "iris/scientific-optimizer-configuration/v1",
            canonical_json_bytes(
                {
                    "policy": run_config["optimizer_policy"],
                    "model_configuration_sha256": teacher_hash,
                    "parameter_names": best.manifest.payload.parameter_names,
                }
            ),
        )
        != run_config["optimizer_sha256"]
    ):
        raise ValueError("Capture/source checkpoint/teacher/runtime identities disagree")
    base = config_path.absolute().parent
    train, validation = _rows(config.train, base), _rows(config.validation, base)
    digest = sha256(b"iris/scientific-train-corpus/v1\0")
    logical_bytes = 0
    if any(row.split != "train" for row in train) or len({row.id for row in train}) != len(train):
        raise ValueError("Original unique training corpus required")
    for row in train:
        content = canonical_json_bytes(row.model_dump(mode="json"))
        logical_bytes += len(content)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    validation, validation_sha = own_scientific_validation_rows_v1(validation, training_rows=train)
    if (
        (digest.hexdigest(), len(train), logical_bytes)
        != (
            run_config["dataset_sha256"],
            run_config["dataset_examples"],
            run_config["dataset_logical_bytes"],
        )
        or (validation_sha, len(validation))
        != (lifecycle_config["validation_sha256"], lifecycle_config["validation_examples"])
        or (payload.purpose == "scientific" and (len(train), len(validation)) != (50000, 5000))
        or test.maximum_rows > 5000
        or test.sha256 != payload.test_parquet_sha256
    ):
        raise ValueError("Original train/validation/test data identities differ")
    rows, corpus_sha = _own_test_rows(
        _rows(test, base),
        excluded_seeds={row.seed for row in (*train, *validation)},
        scientific=payload.purpose == "scientific",
    )
    if corpus_sha != payload.test_corpus_sha256 or len(rows) != len(payload.rows):
        raise ValueError("Incomplete or different held-out corpus")
    # Audit EVERY row before any sandbox dispatch, including mismatches late in the split.
    for row, generated in zip(rows, payload.rows, strict=True):
        if (
            row.id != generated.example_id
            or row.task_type != generated.task_type
            or sha256(row.prompt.encode()).hexdigest() != generated.prompt_sha256
            or generated.example_sha256
            != domain_sha256(
                "iris/scientific-teacher-example/v1",
                canonical_json_bytes(row.model_dump(mode="json")),
            )
        ):
            raise ValueError("Generated row membership/order/content binding differs")
    return rows, snapshot


def _scored_row(
    row: IrisExample,
    generated: ScientificGeneratedRowV1,
    score: IrisSyntheticAnswerScoreV1,
    policy: PythonSandboxPolicyV1 | None,
) -> ScientificScoredRowV1:
    output = generated.output_text.encode("utf-8")
    if (
        type(score) is not IrisSyntheticAnswerScoreV1
        or score.example_id != row.id
        or score.task_type != row.task_type
        or score.split != "test"
        or score.scoring_profile != "iris-synthetic-01-scoring-v1"
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
        answer = parse_structured_response(generated.output_text)
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
        evidence = ScientificCodeEvidenceV1.model_validate(asdict(score.execution))
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
            or evidence.worker_sha256 != sha256(_worker()).hexdigest()
        ):
            raise ValueError("Code evidence binds another source/input/worker")
    return ScientificScoredRowV1(
        example_id=row.id,
        generation_example_sha256=generated.example_sha256,
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
        target_token_count=generated.target_token_count,
        target_nll_sum=generated.target_nll_sum,
    )


def evaluate_scientific_generation_v1(
    *,
    generation_path: Path,
    execution_path: Path,
    config_path: Path,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV1 | None,
) -> IrisScientificEvaluationV1:
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
            if type(python_policy) is not PythonSandboxPolicyV1:
                raise ValueError("Exact sandbox policy required")
            python_policy = PythonSandboxPolicyV1.model_validate(
                python_policy.model_dump(mode="python")
            )
            if python_policy.wall_time_ms != 5000:
                raise ValueError("Repair wall budget must be exactly five seconds")
        scored = []
        for row, generated in zip(rows, capture.payload.rows, strict=True):
            policy = python_policy if row.executable else None
            score = score_synthetic_answer_v1(
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
        payload = ScientificEvaluationPayloadV1(
            purpose=source.purpose,
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
        result = IrisScientificEvaluationV1(
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


def _checked(result: IrisScientificEvaluationV1) -> bytes:
    if type(result) is not IrisScientificEvaluationV1:
        raise ValueError("Exact scientific evaluation required")
    result = IrisScientificEvaluationV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Scientific evaluation identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Scientific evaluation exceeds its byte budget")
    return content


def read_scientific_evaluation_v1(path: Path) -> IrisScientificEvaluationV1:
    """Verify canonical hashes/counts, not source history or execution authenticity."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Evaluation directory inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificEvaluationV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Evaluation canonical identity differs")
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific evaluation report could not be read"
        ) from None


def write_scientific_evaluation_v1(
    result: IrisScientificEvaluationV1,
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
        root = _local_path(root) / "scientific-evaluation-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_evaluation_v1(target) != result:
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
        candidate = root / f".iris-evaluation-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged evaluation differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_evaluation_v1(target) != result:
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
    "evaluate_scientific_generation_v1",
    "read_scientific_evaluation_v1",
    "write_scientific_evaluation_v1",
]
