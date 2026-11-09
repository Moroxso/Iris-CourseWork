"""Durable complete-split generation: verify the prefix, infer only the missing suffix."""

import json
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import (
    _validate_integrity,
    read_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.request_checkpoint_bootstrap import _validate_projection_config
from iris.application.study_checkpoint_scope import _scoped_generation
from iris.application.study_dataset_io import _check_parents, _read_regular_bytes
from iris.application.study_generation import (
    _assemble_study_generation,
    _check_request,
    _GeneratedStudyAnswer,
    _iter_study_generation_answers,
    _preflight_digest,
    _prepare_study_generation,
    _PreparedStudyGeneration,
)
from iris.application.study_generation_io import read_study_generation_v1
from iris.application.study_generation_run_io import (
    _ANSWER_DOMAIN,
    _MAX_BYTES,
    _PENDING_NAMES,
    _RUN_DOMAIN,
    _archive_pending,
    _decode,
    _directory,
    _inventory,
    _names,
    _pending_bytes,
    _publish,
    _read_answer,
    _stage,
    _validated,
    read_study_generation_run_v1,
)
from iris.application.verification_v4 import (
    _checkpoint_projection,
    _validate_checkpoint_binding,
    _validate_profile_v4_binding,
    verify_request_artifact_v4,
)
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_generation_run_v1 import (
    IrisStudyGenerationAnswerV1,
    IrisStudyGenerationRunV1,
    StudyGenerationAnswerPayloadV1,
    StudyGenerationCheckpointV1,
    StudyGenerationPlannedAnswerV1,
    StudyGenerationRunPayloadV1,
    StudyGenerationRunSchemaV1,
)
from iris.artifacts.study_generation_v1 import IrisStudyGenerationV1
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock


@dataclass(frozen=True, slots=True)
class IrisStudyGenerationRunResultV1:
    directory: Path
    path: Path
    run_payload_sha256: str
    generation_payload_sha256: str
    generated_now: int
    reused_answers: int
    retained_file_bytes: int


def _schema(*, answer: bool) -> StudyGenerationRunSchemaV1:
    return StudyGenerationRunSchemaV1(
        artifact_type="iris.study_generation_answer" if answer else "iris.study_generation_run",
        version=1,
        canonicalization="iris-json-v1",
        hash_algorithm="sha256",
        content_policy="private_generation_plan_and_answer_request_envelopes",
    )


def _plan(
    prepared: _PreparedStudyGeneration, directory: Path, record_budget: int, storage_budget: int
) -> IrisStudyGenerationRunV1:
    study, preflight, dataset = (
        prepared.study,
        prepared.preflight,
        prepared.study.payload.dataset.payload,
    )
    identity = prepared.checkpoint_identity
    projection = None if identity is None else _checkpoint_projection(identity, root=prepared.root)
    _validate_projection_config(
        profile=prepared.profile, config=prepared.config, projection=projection
    )
    payload = StudyGenerationRunPayloadV1(
        benchmark_payload_sha256=study.payload_sha256,
        dataset_payload_sha256=study.payload.dataset.payload_sha256,
        manifest_payload_sha256=dataset.manifest.payload_sha256,
        protocol_declaration_payload_sha256=dataset.protocol_declaration_payload_sha256,
        configuration=prepared.config,
        configuration_sha256=preflight.configuration_sha256,
        execution_profile_set_sha256=preflight.execution_profile_set_sha256,
        input_preflight_sha256=_preflight_digest(preflight),
        terminal_token_id=prepared.terminal_token_id,
        arm_id=prepared.profile.arm_id,
        split=prepared.split,
        checkpoint_identity=None
        if identity is None
        else StudyGenerationCheckpointV1.model_validate(asdict(identity)),
        directory=str(directory),
        request_root=str(prepared.root),
        max_artifact_bytes=prepared.max_artifact_bytes,
        max_record_bytes=record_budget,
        max_storage_bytes=storage_budget,
        answers=tuple(
            StudyGenerationPlannedAnswerV1(
                example_id=example.example_id, task_family=assignment.task_family, decoding=policy
            )
            for policy, (_, example, assignment) in zip(
                prepared.policies, prepared.selected, strict=True
            )
        ),
        membership_policy="all_examples_in_selected_split_in_manifest_order",
        recovery_policy="verify_prefix_and_pending_before_generating_missing_suffix",
        output_policy="generation_v1_in_managed_directory",
        scientific_launch_status="not_assessed",
    )
    return _validated(
        IrisStudyGenerationRunV1(
            schema_identity=_schema(answer=False),
            payload=payload,
            payload_sha256=domain_sha256(
                _RUN_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
        ),
        IrisStudyGenerationRunV1,
        _RUN_DOMAIN,
    )


def _reserve(directory: Path, run: IrisStudyGenerationRunV1, additional: int = 0) -> int:
    retained = _inventory(directory, run)
    if retained + additional > run.payload.max_storage_bytes:
        raise ValueError("generation run exceeds its explicit logical-byte budget")
    return retained


def _answer_path(directory: Path, index: int) -> Path:
    return directory / "answers" / f"{index:06d}.json"


def _record(
    run: IrisStudyGenerationRunV1, index: int, item: _GeneratedStudyAnswer
) -> IrisStudyGenerationAnswerV1:
    payload = StudyGenerationAnswerPayloadV1(
        run_payload_sha256=run.payload_sha256, index=index, answer=item.answer, request=item.request
    )
    return _validated(
        IrisStudyGenerationAnswerV1(
            schema_identity=_schema(answer=True),
            payload=payload,
            payload_sha256=domain_sha256(
                _ANSWER_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
        ),
        IrisStudyGenerationAnswerV1,
        _ANSWER_DOMAIN,
        run.payload.max_record_bytes,
    )


def _check_record(
    record: IrisStudyGenerationAnswerV1,
    *,
    index: int,
    run: IrisStudyGenerationRunV1,
    prepared: _PreparedStudyGeneration,
    installed: bool,
) -> IrisStudyGenerationAnswerV1:
    owned = _validated(
        record, IrisStudyGenerationAnswerV1, _ANSWER_DOMAIN, run.payload.max_record_bytes
    )
    payload = owned.payload
    planned = run.payload.answers[index]
    answer = payload.answer
    request, _ = _validate_integrity(payload.request)
    if (
        payload.index != index
        or payload.run_payload_sha256 != run.payload_sha256
        or answer.example_id != planned.example_id
        or answer.task_family != planned.task_family
        or answer.decoding != planned.decoding
        or request.payload.topology.arm_id != run.payload.arm_id
        or request.payload.protocol_declaration_payload_sha256
        != run.payload.protocol_declaration_payload_sha256
        or request.payload.configuration.value != prepared.config
    ):
        raise ValueError("durable answer does not belong to this exact plan and index")
    _validate_profile_v4_binding(request.payload)
    projection = _validate_checkpoint_binding(payload=request.payload, root=prepared.root)
    if (
        None if projection is None else projection.checkpoint_identity
    ) != prepared.checkpoint_identity:
        raise ValueError("durable answer checkpoint differs from the immutable run")
    source_index, example, _ = prepared.selected[index]
    _check_request(
        request,
        example=example,
        answer=answer,
        observation=prepared.preflight.examples[source_index],
        preflight=prepared.preflight,
    )
    if installed:
        path = prepared.root / "requests" / "v4" / f"{request.payload_sha256}.json"
        _check_parents(path, create=False)
        report = verify_request_artifact_v4(path)
        if (
            report.payload_sha256 != request.payload_sha256
            or read_request_artifact_v4(path) != request
        ):
            raise ValueError("installed request differs from its durable answer envelope")
    return owned


def _prefix(
    directory: Path, run: IrisStudyGenerationRunV1, prepared: _PreparedStudyGeneration
) -> tuple[IrisStudyGenerationAnswerV1, ...]:
    _reserve(directory, run)
    folder = directory / "answers"
    names = set(_names(folder, len(run.payload.answers))) if folder.exists() else set()
    if names != {f"{index:06d}.json" for index in range(len(names))}:
        raise ValueError("durable answers must form one contiguous planned prefix")
    return tuple(
        _check_record(
            _read_answer(_answer_path(directory, index), run.payload.max_record_bytes),
            index=index,
            run=run,
            prepared=prepared,
            installed=True,
        )
        for index in range(len(names))
    )


def _unchanged_records(
    directory: Path, records: tuple[IrisStudyGenerationAnswerV1, ...], run: IrisStudyGenerationRunV1
) -> None:
    if len(_names(directory / "answers", len(run.payload.answers))) != len(records):
        raise ValueError("durable answer membership changed")
    for index, record in enumerate(records):
        if _read_regular_bytes(
            _answer_path(directory, index), max_bytes=run.payload.max_record_bytes
        ) != canonical_json_file_bytes(record.model_dump(mode="json")):
            raise ValueError("durable answer changed during verification")


def _commit_answer(
    record: IrisStudyGenerationAnswerV1,
    directory: Path,
    run: IrisStudyGenerationRunV1,
    prepared: _PreparedStudyGeneration,
    claim: ProcessFileLock,
) -> None:
    index = record.payload.index
    checked = _check_record(record, index=index, run=run, prepared=prepared, installed=False)
    content = canonical_json_file_bytes(checked.model_dump(mode="json"))
    _reserve(directory, run, 2 * len(content))
    pending = directory / ".answer-pending.json"
    _stage(pending, content, claim=claim)
    # The full raw text and request envelope are now recoverable before the
    # separate request store is touched. Existing request bytes cannot be replaced.
    claim.check()
    request_path = (
        prepared.root / "requests" / "v4" / f"{checked.payload.request.payload_sha256}.json"
    )
    _check_parents(request_path, create=True)
    written = write_request_artifact_v4(checked.payload.request)
    if written.path != request_path:
        raise ValueError("request installation escaped the immutable artifact root")
    _check_record(checked, index=index, run=run, prepared=prepared, installed=True)
    _publish(_answer_path(directory, index), content, pending=pending, claim=claim)


def _recover_exact(
    pending: Path,
    expected: bytes,
    directory: Path,
    run: IrisStudyGenerationRunV1,
    claim: ProcessFileLock,
) -> None:
    if not pending.exists():
        return
    content = _pending_bytes(pending, _MAX_BYTES)
    if content != expected:
        if not expected.startswith(content):
            raise ValueError("interrupted known publication conflicts with the immutable value")
        _reserve(directory, run, len(content))
        _archive_pending(pending, claim=claim, maximum=_MAX_BYTES)


def _recover_answer(
    directory: Path,
    run: IrisStudyGenerationRunV1,
    prepared: _PreparedStudyGeneration,
    claim: ProcessFileLock,
    records: tuple[IrisStudyGenerationAnswerV1, ...],
) -> None:
    pending = directory / ".answer-pending.json"
    if not pending.exists():
        return
    content = _pending_bytes(pending, run.payload.max_record_bytes)
    if records and content == canonical_json_file_bytes(records[-1].model_dump(mode="json")):
        _publish(_answer_path(directory, len(records) - 1), content, pending=pending, claim=claim)
        return
    if len(records) == len(run.payload.answers):
        raise ValueError("unexpected pending answer after a complete committed prefix")
    try:
        record = _decode(content, IrisStudyGenerationAnswerV1)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Uncommitted malformed/truncated bytes cannot reconstruct an answer.
        # Preserve them; a later inference may retry only this uncommitted index.
        _reserve(directory, run, len(content))
        _archive_pending(pending, claim=claim, maximum=run.payload.max_record_bytes)
        return
    # Structurally complete but wrong-plan/hash/semantic records fail closed.
    _check_record(record, index=len(records), run=run, prepared=prepared, installed=False)
    _commit_answer(record, directory, run, prepared, claim)


def _prepared_from_run(
    run: IrisStudyGenerationRunV1, benchmark: IrisStudyBenchmarkV1
) -> _PreparedStudyGeneration:
    payload = run.payload
    identity = payload.checkpoint_identity
    return _prepare_study_generation(
        benchmark,
        config=payload.configuration,
        arm_id=payload.arm_id,
        checkpoint_identity=None
        if identity is None
        else IrisArmTeacherCheckpointIdentity(**identity.model_dump()),
        split=payload.split,
        decoding=tuple(item.decoding for item in payload.answers),
        terminal_token_id=payload.terminal_token_id,
        max_artifact_bytes=payload.max_artifact_bytes,
    )


@_scoped_generation
def verify_study_generation_run_v1(
    directory: Path, *, benchmark: IrisStudyBenchmarkV1
) -> IrisStudyGenerationV1:
    """Read-only complete-run audit; no claim, repair, model load, inference or scoring."""
    try:
        destination = _directory(directory)
        run = read_study_generation_run_v1(destination)
        prepared = _prepared_from_run(run, benchmark)
        if (
            _plan(
                prepared, destination, run.payload.max_record_bytes, run.payload.max_storage_bytes
            )
            != run
        ):
            raise ValueError("generation run differs from current complete preflight and plan")
        if any((destination / name).exists() for name in _PENDING_NAMES):
            raise ValueError("pending generation publications require explicit resume")
        records = _prefix(destination, run, prepared)
        if len(records) != len(run.payload.answers):
            raise ValueError("incomplete generation has no complete result")
        expected = _assemble_study_generation(
            prepared,
            tuple(record.payload.answer for record in records),
            records[0].payload.request.payload.checkpoint_transfer,
        )
        if read_study_generation_v1(destination / "generation.json") != expected:
            raise ValueError("final generation differs from its verified durable answers")
        _unchanged_records(destination, records, run)
        if (
            read_study_generation_run_v1(destination) != run
            or read_study_generation_v1(destination / "generation.json") != expected
            or any((destination / name).exists() for name in _PENDING_NAMES)
        ):
            raise ValueError("generation run changed during verification")
        _reserve(destination, run)
        return expected
    except Exception:
        raise IrisStudyGenerationError(
            "Iris generation run could not be independently verified"
        ) from None


@_scoped_generation
def generate_study_split_run_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    config: IrisBootstrapConfig,
    arm_id: object,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    split: ExperimentSplitV1,
    decoding: tuple[DecodingProvenanceV1, ...],
    terminal_token_id: int,
    max_artifact_bytes: int,
    max_record_bytes: int,
    max_storage_bytes: int,
    directory: Path,
    resume: bool,
) -> IrisStudyGenerationRunResultV1:
    """Persist the complete answer/request before installation; resume only missing work.

    All budgets and resume mode are explicit. Incomplete uncommitted bytes are
    preserved but can require a new inference, not an exactly-once guarantee.
    The final compatible generation-v1 artifact lives at directory/generation.json.
    """
    try:
        if type(resume) is not bool:
            raise TypeError("generation resume must be an explicit boolean")
        destination = _directory(directory)
        prepared = _prepare_study_generation(
            benchmark,
            config=config,
            arm_id=arm_id,
            checkpoint_identity=checkpoint_identity,
            split=split,
            decoding=decoding,
            terminal_token_id=terminal_token_id,
            max_artifact_bytes=max_artifact_bytes,
        )
        run = _plan(prepared, destination, max_record_bytes, max_storage_bytes)
        run_bytes = canonical_json_file_bytes(run.model_dump(mode="json"))
        if 1 + 2 * len(run_bytes) > max_storage_bytes:
            raise ValueError("insufficient generation budget for run initialization")
        _check_parents(destination / ".lock", create=True)
        with process_file_lock(destination / ".lock") as claim:
            _reserve(destination, run)
            manifest = destination / "run.json"
            if manifest.exists():
                if not resume or read_study_generation_run_v1(destination) != run:
                    raise ValueError("existing generation run requires explicit identical resume")
            else:
                names = set(_names(destination, 8))
                if not names <= {".lock", ".run-pending.json", "interrupted"}:
                    raise ValueError("cannot initialize an occupied generation directory")
                interrupted = bool(names & {".run-pending.json", "interrupted"})
                if resume != interrupted:
                    raise ValueError("generation initialization mode disagrees with retained state")
                if (destination / "interrupted").exists():
                    for name in _names(destination / "interrupted", 100000):
                        if not run_bytes.startswith(
                            _pending_bytes(destination / "interrupted" / name, _MAX_BYTES)
                        ):
                            raise ValueError("interrupted initialization belongs to another plan")
            _recover_exact(destination / ".run-pending.json", run_bytes, destination, run, claim)
            if not manifest.exists() or (destination / ".run-pending.json").exists():
                _reserve(destination, run, 2 * len(run_bytes))
                _publish(
                    manifest, run_bytes, pending=destination / ".run-pending.json", claim=claim
                )
            records = _prefix(destination, run, prepared)
            if len(records) != len(run.payload.answers) and any(
                (destination / name).exists()
                for name in ("generation.json", ".generation-pending.json")
            ):
                raise ValueError("final generation cannot coexist with missing durable answers")
            _recover_answer(destination, run, prepared, claim, records)
            records = _prefix(destination, run, prepared)
            reused = len(records)
            if reused < len(run.payload.answers):
                # Reserve the largest allowed record's two hard-link names plus
                # final-output publication headroom before each new inference.
                headroom = 2 * max_record_bytes + 2 * max_artifact_bytes
                _reserve(destination, run, headroom)
                answer_bytes = sum(
                    len(canonical_json_bytes(r.payload.answer.model_dump(mode="json")))
                    for r in records
                )
                with closing(
                    _iter_study_generation_answers(
                        prepared, start_index=reused, answer_bytes=answer_bytes
                    )
                ) as generated:
                    for index, item in enumerate(generated, start=reused):
                        _commit_answer(_record(run, index, item), destination, run, prepared, claim)
                        if index + 1 < len(run.payload.answers):
                            _reserve(destination, run, headroom)
                            claim.check()
            final_records = _prefix(destination, run, prepared)
            if final_records[:reused] != records:
                raise ValueError("previously verified generation prefix changed")
            generation = _assemble_study_generation(
                prepared,
                tuple(record.payload.answer for record in final_records),
                final_records[0].payload.request.payload.checkpoint_transfer,
            )
            content = canonical_json_file_bytes(generation.model_dump(mode="json"))
            _recover_exact(
                destination / ".generation-pending.json", content, destination, run, claim
            )
            if (destination / "generation.json").exists() and read_study_generation_v1(
                destination / "generation.json"
            ) != generation:
                raise ValueError("existing final generation conflicts with durable answers")
            if (
                not (destination / "generation.json").exists()
                or (destination / ".generation-pending.json").exists()
            ):
                _reserve(destination, run, 2 * len(content))
                _publish(
                    destination / "generation.json",
                    content,
                    pending=destination / ".generation-pending.json",
                    claim=claim,
                )
            if verify_study_generation_run_v1(destination, benchmark=prepared.study) != generation:
                raise ValueError("published generation failed whole-run verification")
            return IrisStudyGenerationRunResultV1(
                destination,
                destination / "generation.json",
                run.payload_sha256,
                generation.payload_sha256,
                len(final_records) - reused,
                reused,
                _reserve(destination, run),
            )
    except Exception:
        raise IrisStudyGenerationError(
            "Iris durable complete-split generation did not complete"
        ) from None


__all__ = [
    "IrisStudyGenerationRunResultV1",
    "generate_study_split_run_v1",
    "verify_study_generation_run_v1",
]
