"""Complete-split scoring with immutable plans, verified prefix recovery and bounded storage."""

from dataclasses import dataclass
from pathlib import Path

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_checkpoint_scope import _scoped_evaluation
from iris.application.study_code_evaluation_io import read_study_code_evaluation_v1
from iris.application.study_code_receipt import (
    evaluate_study_request_python_answer_v1,
    verify_study_code_evaluation_v1,
)
from iris.application.study_dataset_io import _check_parents, _read_regular_bytes
from iris.application.study_evaluation import (
    evaluate_study_request_answer_v1,
    verify_study_evaluation_v1,
)
from iris.application.study_evaluation_io import read_study_evaluation_v1
from iris.application.study_evaluation_run_io import (
    _MAX_METADATA_BYTES,
    _PENDING_NAMES,
    _RUN_DOMAIN,
    _SUMMARY_DOMAIN,
    _archive_pending,
    _bounded_bytes,
    _directory,
    _inventory,
    _publish,
    _validated_run,
    _validated_summary,
    read_study_evaluation_run_v1,
    read_study_evaluation_summary_v1,
)
from iris.application.study_generation import verify_study_generation_v1
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_code_evaluation_v1 import IrisStudyCodeEvaluationV1, StudyCodeReceiptV1
from iris.artifacts.study_evaluation_run_v1 import (
    IrisStudyEvaluationRunV1,
    IrisStudyEvaluationSummaryV1,
    StudyEvaluationCountsV1,
    StudyEvaluationFamilyCountsV1,
    StudyEvaluationResultReferenceV1,
    StudyEvaluationRunPayloadV1,
    StudyEvaluationRunSchemaV1,
    StudyEvaluationSummaryPayloadV1,
)
from iris.artifacts.study_evaluation_v1 import IrisStudyEvaluationV1, StudyEvaluationPayloadV1
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

type _Receipt = IrisStudyEvaluationV1 | IrisStudyCodeEvaluationV1
_CODE_FAMILY = "code_generation_and_verification"


@dataclass(frozen=True, slots=True)
class IrisStudyEvaluationRunResultV1:
    directory: Path
    run_payload_sha256: str
    summary_payload_sha256: str
    evaluated_now: int
    reused_evaluations: int
    retained_file_bytes: int


def _plan(
    generation: IrisStudyGenerationV1, policy: PythonSandboxPolicyV1, budget: int
) -> IrisStudyEvaluationRunV1:
    if type(policy) is not PythonSandboxPolicyV1:
        raise TypeError("an explicit exact code policy is required")
    source = generation.payload
    payload = StudyEvaluationRunPayloadV1.model_validate(
        {
            "generation_payload_sha256": generation.payload_sha256,
            **{
                key: getattr(source, key)
                for key in (
                    "benchmark_payload_sha256",
                    "dataset_payload_sha256",
                    "manifest_payload_sha256",
                    "protocol_declaration_payload_sha256",
                    "configuration_sha256",
                    "execution_profile_set_sha256",
                    "terminal_token_id",
                    "arm_id",
                    "split",
                    "checkpoint_transfer",
                )
            },
            "generation_input_preflight_sha256": source.input_preflight_sha256,
            "code_policy": policy,
            "max_storage_bytes": budget,
            "answers": tuple(
                {
                    key: getattr(item, key)
                    for key in (
                        "example_id",
                        "task_family",
                        "request_payload_sha256",
                        "output_sha256",
                        "output_byte_length",
                    )
                }
                for item in source.answers
            ),
            "membership_policy": "all_generated_answers_in_manifest_order",
            "recovery_policy": "verify_committed_prefix_before_executing_missing_answers",
            "scientific_launch_status": "not_assessed",
        }
    )
    return _validated_run(
        IrisStudyEvaluationRunV1(
            schema_identity=StudyEvaluationRunSchemaV1(
                artifact_type="iris.study_evaluation_run",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_metadata_and_hashes_only",
            ),
            payload_sha256=domain_sha256(
                _RUN_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
    )


def _reserve(directory: Path, run: IrisStudyEvaluationRunV1, additional: int = 0) -> int:
    retained = _inventory(directory, answers=len(run.payload.answers))
    if retained + additional > run.payload.max_storage_bytes:
        raise ValueError("evaluation run has insufficient explicit retained-byte budget")
    return retained


def _receipt_path(directory: Path, index: int) -> Path:
    return directory / "receipts" / f"{index:06d}.json"


def _prefix(directory: Path, run: IrisStudyEvaluationRunV1) -> int:
    _reserve(directory, run)
    folder = directory / "receipts"
    names = {item.name for item in folder.iterdir()} if folder.exists() else set()
    if names != {f"{index:06d}.json" for index in range(len(names))}:
        raise ValueError("committed evaluations must be one contiguous ordered prefix")
    return len(names)


def _load_receipt(path: Path, *, code: bool) -> _Receipt:
    return read_study_code_evaluation_v1(path) if code else read_study_evaluation_v1(path)


def _verify_receipt(
    receipt: _Receipt,
    *,
    index: int,
    run: IrisStudyEvaluationRunV1,
    generation: IrisStudyGenerationV1,
    benchmark: IrisStudyBenchmarkV1,
    request_root: Path,
) -> _Receipt:
    planned = run.payload.answers[index]
    answer = generation.payload.answers[index]
    request = request_root / "requests" / "v4" / f"{planned.request_payload_sha256}.json"
    recorded: StudyCodeReceiptV1 | StudyEvaluationPayloadV1
    if isinstance(receipt, IrisStudyCodeEvaluationV1):
        owned: _Receipt = verify_study_code_evaluation_v1(
            receipt, benchmark=benchmark, request_path=request, output_text=answer.output_text
        )
        recorded = receipt.payload.receipt
        if receipt.payload.observation.policy != run.payload.code_policy:
            raise ValueError("code receipt policy differs from immutable run plan")
    else:
        owned = verify_study_evaluation_v1(
            receipt, benchmark=benchmark, request_path=request, output_text=answer.output_text
        )
        recorded = receipt.payload
    if (
        recorded.example_id != planned.example_id
        or recorded.task_family != planned.task_family
        or recorded.request_payload_sha256 != planned.request_payload_sha256
        or recorded.output_sha256 != planned.output_sha256
        or recorded.output_byte_length != planned.output_byte_length
        or recorded.input_binding.terminal_token_id != run.payload.terminal_token_id
        or any(
            getattr(recorded, key) != getattr(run.payload, key)
            for key in (
                "benchmark_payload_sha256",
                "dataset_payload_sha256",
                "manifest_payload_sha256",
                "protocol_declaration_payload_sha256",
                "configuration_sha256",
                "execution_profile_set_sha256",
                "arm_id",
                "split",
                "checkpoint_transfer",
            )
        )
    ):
        raise ValueError("receipt does not belong to this exact generation and run")
    return owned


def _reference(path: Path, receipt: _Receipt) -> StudyEvaluationResultReferenceV1:
    content = _read_regular_bytes(path, max_bytes=12 * 1024 * 1024)
    if content != canonical_json_file_bytes(receipt.model_dump(mode="json")):
        raise ValueError("receipt bytes changed after verification")
    record = (
        receipt.payload.receipt
        if isinstance(receipt, IrisStudyCodeEvaluationV1)
        else receipt.payload
    )
    return StudyEvaluationResultReferenceV1.model_validate(
        {
            "example_id": record.example_id,
            "task_family": record.task_family,
            "artifact_type": receipt.schema_identity.artifact_type,
            "evaluation_payload_sha256": receipt.payload_sha256,
            "file_sha256": bytes_sha256(content),
            "file_byte_length": len(content),
            "outcome": record.outcome,
        }
    )


def _all_references(
    directory: Path,
    run: IrisStudyEvaluationRunV1,
    generation: IrisStudyGenerationV1,
    benchmark: IrisStudyBenchmarkV1,
    request_root: Path,
) -> tuple[StudyEvaluationResultReferenceV1, ...]:
    result = []
    for index in range(_prefix(directory, run)):
        path = _receipt_path(directory, index)
        receipt = _load_receipt(path, code=run.payload.answers[index].task_family == _CODE_FAMILY)
        checked = _verify_receipt(
            receipt,
            index=index,
            run=run,
            generation=generation,
            benchmark=benchmark,
            request_root=request_root,
        )
        result.append(_reference(path, checked))
    return tuple(result)


def _summary(
    run: IrisStudyEvaluationRunV1, results: tuple[StudyEvaluationResultReferenceV1, ...]
) -> IrisStudyEvaluationSummaryV1:
    if len(results) != len(run.payload.answers):
        raise ValueError("a complete aggregate requires every planned answer")

    def counts(items: tuple[StudyEvaluationResultReferenceV1, ...]) -> dict[str, int]:
        return {
            "total": len(items),
            **{
                outcome: sum(item.outcome == outcome for item in items)
                for outcome in ("correct", "incorrect", "invalid_output")
            },
        }

    families = tuple(dict.fromkeys(item.task_family for item in results))
    payload = StudyEvaluationSummaryPayloadV1(
        run_payload_sha256=run.payload_sha256,
        results=results,
        overall=StudyEvaluationCountsV1.model_validate(counts(results)),
        by_family=tuple(
            StudyEvaluationFamilyCountsV1.model_validate(
                {
                    "task_family": family,
                    **counts(tuple(item for item in results if item.task_family == family)),
                }
            )
            for family in families
        ),
        status="complete_all_generated_answers_evaluated",
        aggregation_policy="unweighted_example_counts_including_invalid_outputs",
        verification_scope="bindings_and_recomputed_scores_not_execution_attestation",
        scientific_launch_status="not_assessed",
    )
    return _validated_summary(
        IrisStudyEvaluationSummaryV1(
            schema_identity=StudyEvaluationRunSchemaV1(
                artifact_type="iris.study_evaluation_summary",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_metadata_and_hashes_only",
            ),
            payload_sha256=domain_sha256(
                _SUMMARY_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
    )


def _recover_exact_slot(
    pending: Path,
    expected: bytes,
    *,
    directory: Path,
    run: IrisStudyEvaluationRunV1,
    claim: ProcessFileLock,
    require_prefix: bool = False,
) -> None:
    if not pending.exists():
        return
    content = _bounded_bytes(pending, maximum=_MAX_METADATA_BYTES, empty=True)
    if content != expected:
        if require_prefix and not expected.startswith(content):
            raise ValueError("interrupted initialization conflicts with this run plan")
        _reserve(directory, run, len(content))
        _archive_pending(pending, claim=claim, maximum=_MAX_METADATA_BYTES)


def _recover_receipt(
    directory: Path,
    run: IrisStudyEvaluationRunV1,
    generation: IrisStudyGenerationV1,
    benchmark: IrisStudyBenchmarkV1,
    request_root: Path,
    claim: ProcessFileLock,
    count: int,
) -> None:
    pending = directory / ".receipt-pending.json"
    if not pending.exists():
        return
    content = _bounded_bytes(pending, maximum=12 * 1024 * 1024, empty=True)
    # Death after hard-link publication: keep the committed value and remove only its extra link.
    if count and content == _read_regular_bytes(_receipt_path(directory, count - 1)):
        _publish(_receipt_path(directory, count - 1), content, pending=pending, claim=claim)
        return
    if count == len(run.payload.answers):
        raise ValueError("unexpected pending receipt after complete prefix")
    try:
        receipt = _load_receipt(
            pending, code=run.payload.answers[count].task_family == _CODE_FAMILY
        )
    except IrisStudyEvaluationError:
        _reserve(directory, run, len(content))
        _archive_pending(pending, claim=claim, maximum=12 * 1024 * 1024)
        return
    checked = _verify_receipt(
        receipt,
        index=count,
        run=run,
        generation=generation,
        benchmark=benchmark,
        request_root=request_root,
    )
    content = canonical_json_file_bytes(checked.model_dump(mode="json"))
    _reserve(directory, run, len(content))
    _publish(_receipt_path(directory, count), content, pending=pending, claim=claim)


@_scoped_evaluation
def evaluate_study_split_v1(
    generation: IrisStudyGenerationV1,
    *,
    benchmark: IrisStudyBenchmarkV1,
    request_root: Path,
    directory: Path,
    code_policy: PythonSandboxPolicyV1,
    max_storage_bytes: int,
    resume: bool,
) -> IrisStudyEvaluationRunResultV1:
    """Score all five families; explicit resume revalidates committed results, never reruns them.

    Uncommitted incomplete bytes are preserved in interrupted/ and may require a
    new execution. This is at-least-once for uncommitted work, not exactly-once or
    durable sandbox-container recovery. A stale live runtime claim fails closed.
    """
    try:
        if type(resume) is not bool:
            raise TypeError("resume must be an explicit boolean")
        destination = _directory(directory)
        study = validate_study_benchmark_v1(benchmark)
        owned = verify_study_generation_v1(generation, benchmark=study, request_root=request_root)
        run = _plan(owned, code_policy, max_storage_bytes)
        run_bytes = canonical_json_file_bytes(run.model_dump(mode="json"))
        if 1 + 2 * len(run_bytes) > run.payload.max_storage_bytes:
            raise ValueError("insufficient budget for atomic run initialization")
        _check_parents(destination / ".lock", create=True)
        with process_file_lock(destination / ".lock") as claim:
            _reserve(destination, run)
            manifest = destination / "run.json"
            if manifest.exists():
                if not resume or read_study_evaluation_run_v1(destination) != run:
                    raise ValueError(
                        "existing run requires explicit resume of its exact immutable plan"
                    )
            else:
                names = {item.name for item in destination.iterdir()}
                if not names <= {".lock", ".run-pending.json", "interrupted"}:
                    raise ValueError("cannot initialize an occupied run directory")
                if resume and not names & {".run-pending.json", "interrupted"}:
                    raise ValueError("no existing run initialization to resume")
                archived = destination / "interrupted"
                if archived.exists() and any(
                    not run_bytes.startswith(
                        _bounded_bytes(path, maximum=_MAX_METADATA_BYTES, empty=True)
                    )
                    for path in archived.iterdir()
                ):
                    raise ValueError("interrupted initialization belongs to another plan")
            _recover_exact_slot(
                destination / ".run-pending.json",
                run_bytes,
                directory=destination,
                run=run,
                claim=claim,
                require_prefix=True,
            )
            if not manifest.exists() or (destination / ".run-pending.json").exists():
                _reserve(destination, run, 2 * len(run_bytes))
                _publish(
                    manifest, run_bytes, pending=destination / ".run-pending.json", claim=claim
                )

            references = _all_references(destination, run, owned, study, request_root)
            if any(
                (destination / name).exists() for name in ("summary.json", ".summary-pending.json")
            ) and len(references) != len(run.payload.answers):
                raise ValueError("complete summary cannot coexist with a missing receipt")
            _recover_receipt(destination, run, owned, study, request_root, claim, len(references))
            reused = _prefix(destination, run)
            evaluated_now = 0
            # Conservative peak logical-byte reserve includes the temporary hard-link name.
            summary_reservation = 4096 + 1024 * len(run.payload.answers)
            for index in range(reused, len(run.payload.answers)):
                planned = run.payload.answers[index]
                code = planned.task_family == _CODE_FAMILY
                receipt_cap = (12 if code else 1) * 1024 * 1024
                _reserve(destination, run, 2 * receipt_cap + 2 * summary_reservation)
                claim.check()
                answer = owned.payload.answers[index]
                request_path = (
                    request_root / "requests" / "v4" / f"{planned.request_payload_sha256}.json"
                )
                if code:
                    evaluated: _Receipt = evaluate_study_request_python_answer_v1(
                        study,
                        request_path=request_path,
                        example_id=planned.example_id,
                        output_text=answer.output_text,
                        terminal_token_id=run.payload.terminal_token_id,
                        policy=run.payload.code_policy,
                    )
                else:
                    evaluated = evaluate_study_request_answer_v1(
                        study,
                        request_path=request_path,
                        example_id=planned.example_id,
                        output_text=answer.output_text,
                        terminal_token_id=run.payload.terminal_token_id,
                    )
                checked = _verify_receipt(
                    evaluated,
                    index=index,
                    run=run,
                    generation=owned,
                    benchmark=study,
                    request_root=request_root,
                )
                content = canonical_json_file_bytes(checked.model_dump(mode="json"))
                _reserve(destination, run, 2 * len(content) + 2 * summary_reservation)
                _publish(
                    _receipt_path(destination, index),
                    content,
                    pending=destination / ".receipt-pending.json",
                    claim=claim,
                )
                evaluated_now += 1

            references = _all_references(destination, run, owned, study, request_root)
            if (
                verify_study_generation_v1(owned, benchmark=study, request_root=request_root)
                != owned
            ):
                raise ValueError("generation changed during evaluation")
            summary = _summary(run, references)
            summary_bytes = canonical_json_file_bytes(summary.model_dump(mode="json"))
            _recover_exact_slot(
                destination / ".summary-pending.json",
                summary_bytes,
                directory=destination,
                run=run,
                claim=claim,
            )
            if (destination / "summary.json").exists() and read_study_evaluation_summary_v1(
                destination
            ) != summary:
                raise ValueError("existing summary conflicts with recomputed results")
            if (
                not (destination / "summary.json").exists()
                or (destination / ".summary-pending.json").exists()
            ):
                _reserve(destination, run, 2 * len(summary_bytes))
                _publish(
                    destination / "summary.json",
                    summary_bytes,
                    pending=destination / ".summary-pending.json",
                    claim=claim,
                )
            if read_study_evaluation_run_v1(destination) != run:
                raise ValueError("run changed during evaluation")
            if (
                verify_study_evaluation_run_v1(
                    destination, generation=owned, benchmark=study, request_root=request_root
                )
                != summary
            ):
                raise ValueError("published evaluation run failed independent verification")
            return IrisStudyEvaluationRunResultV1(
                destination,
                run.payload_sha256,
                summary.payload_sha256,
                evaluated_now,
                reused,
                _reserve(destination, run),
            )
    except IrisCodeEvaluationUnavailableError:
        raise
    except Exception:
        raise IrisStudyEvaluationError("Iris complete-split evaluation did not complete") from None


@_scoped_evaluation
def verify_study_evaluation_run_v1(
    directory: Path,
    *,
    generation: IrisStudyGenerationV1,
    benchmark: IrisStudyBenchmarkV1,
    request_root: Path,
) -> IrisStudyEvaluationSummaryV1:
    """Read-only verification of the whole complete run, every receipt and all counts.

    Does not create/claim a lock, start the sandbox, load weights, or repair files.
    A concurrent write or pending publication fails verification; retry after the owner finishes.
    """
    try:
        destination = _directory(directory)
        run = read_study_evaluation_run_v1(destination)
        summary = read_study_evaluation_summary_v1(destination)
        study = validate_study_benchmark_v1(benchmark)
        owned = verify_study_generation_v1(generation, benchmark=study, request_root=request_root)
        if _plan(owned, run.payload.code_policy, run.payload.max_storage_bytes) != run:
            raise ValueError("run plan does not match the complete generation")
        _reserve(destination, run)
        if any((destination / name).exists() for name in _PENDING_NAMES):
            raise ValueError("pending publications require explicit resume")
        references = _all_references(destination, run, owned, study, request_root)
        if _summary(run, references) != summary:
            raise ValueError("summary differs from independently recomputed scores")
        if (
            read_study_evaluation_run_v1(destination) != run
            or read_study_evaluation_summary_v1(destination) != summary
            or any((destination / name).exists() for name in _PENDING_NAMES)
        ):
            raise ValueError("evaluation run changed during verification")
        for index, reference in enumerate(references):
            content = _read_regular_bytes(
                _receipt_path(destination, index), max_bytes=12 * 1024 * 1024
            )
            if (
                len(content) != reference.file_byte_length
                or bytes_sha256(content) != reference.file_sha256
            ):
                raise ValueError("receipt changed during verification")
        verify_study_generation_v1(owned, benchmark=study, request_root=request_root)
        _reserve(destination, run)
        return summary
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris evaluation run could not be independently verified"
        ) from None


__all__ = [
    "IrisStudyEvaluationRunResultV1",
    "evaluate_study_split_v1",
    "verify_study_evaluation_run_v1",
]
