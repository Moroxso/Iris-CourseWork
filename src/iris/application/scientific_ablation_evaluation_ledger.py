"""Owned score-plan/ack ledger and model-free resume; never execute candidate code on host."""

import os
import re
import shutil
import stat
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from iris.application._scientific_ablation_kernel_recovery import (
    _checked_interval_permission,
    _KernelIntervalPermission,
    _locked_interval_stop,
)
from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_ablation_evaluation_v2 import (
    _MAX_REPORT,
    IrisScientificAblationEvaluationError,
    _assemble,
    _checked,
    _EvaluationInputs,
    _finish,
    _prepare,
    _score_row,
)
from iris.application.scientific_ablation_inference import _digest, _exact_tree
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _finish_descriptor,
    _identity,
    _path,
    _read_file,
    _regular,
    _signature,
    _write_bytes,
)
from iris.application.scientific_ablation_kernel_chunk import (
    check_kernel_chunk_prefix_v1,
    checked_kernel_chunk_v1,
)
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_command import ScientificParquetInputV1
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationAckV2,
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationAckPayloadV2,
    ScientificAblationEvaluationCompletePayloadV2,
    ScientificAblationEvaluationExpectationV2,
    ScientificAblationEvaluationPlanPayloadV2,
    ScientificAblationEvaluationRowPayloadV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import (
    IrisScientificAblationEvaluationV2,
    ScientificAblationScoredRowV2,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_STORE = "scientific-ablation-evaluation-ledger-v2"
_LOCK = ".iris-owner.lock"
_PLAN_DOMAIN = "iris/scientific-ablation-evaluation-plan/v2"
_ROW_DOMAIN = "iris/scientific-ablation-evaluation-row/v2"
_ACK_DOMAIN = "iris/scientific-ablation-evaluation-ack/v2"
_COMPLETE_DOMAIN = "iris/scientific-ablation-evaluation-complete/v2"
_STAGE = re.compile(r"\.iris-score-[0-9a-f]{32}\.tmp")
_ROW = re.compile(r"score-[0-9]{5}")
_FILES = {
    "plan.json",
    "attempt.json",
    "row.json",
    "ack.json",
    "complete.json",
    "evaluation.json",
    "failure.json",
}
_ROW_FILES = {"attempt.json", "row.json", "ack.json"}
_MAX_PLAN = 16 * 1024**2
_MAX_ROW = 65536
_MAX_STORAGE = 16 * 1024**3
_MAX_DIRS = 20000
_MAX_FILES = 60000


@dataclass(frozen=True, slots=True)
class IrisScientificAblationEvaluationProgressV2:
    path: Path
    plan: IrisScientificAblationEvaluationPlanV2
    rows: tuple[IrisScientificAblationEvaluationRowV2, ...]
    acknowledgements: tuple[IrisScientificAblationEvaluationAckV2, ...]
    complete: IrisScientificAblationEvaluationCompleteV2 | None
    evaluation: IrisScientificAblationEvaluationV2 | None
    uncommitted_stage_count: int

    @property
    def next_row_index(self) -> int:
        return len(self.rows)


def _fail(primary: BaseException, secondary: list[BaseException]) -> None:
    if not isinstance(primary, Exception):
        if secondary:
            raise primary from BaseExceptionGroup("Score ledger cleanup failed", secondary)
        raise primary
    for error in secondary:
        if not isinstance(error, Exception):
            raise error from BaseExceptionGroup(
                "Score ledger work/cleanup failed", [primary, *secondary]
            )
    if isinstance(primary, IrisCodeEvaluationUnavailableError):
        raise IrisCodeEvaluationUnavailableError(
            "Mode-bound score isolation unavailable; no row acknowledged"
        ) from None
    raise IrisScientificAblationEvaluationError(
        "Mode-bound score ledger operation failed"
    ) from None


def _sha(domain: str, payload: BaseModel) -> str:
    return domain_sha256(domain, canonical_json_bytes(payload.model_dump(mode="json")))


def _file(value: BaseModel) -> bytes:
    return canonical_json_file_bytes(value.model_dump(mode="json"))


def _plan(inputs: _EvaluationInputs) -> IrisScientificAblationEvaluationPlanV2:
    if inputs.capture.complete is None:
        raise ValueError("Complete captured inputs required")
    payload = ScientificAblationEvaluationPlanPayloadV2(
        generation_plan=inputs.capture.plan,
        capture=inputs.capture.complete,
        parquet_spec_sha256=(
            _sha("iris/scientific-ablation-evaluation-input-spec/v2", inputs.specs[0]),
            _sha("iris/scientific-ablation-evaluation-input-spec/v2", inputs.specs[1]),
            _sha("iris/scientific-ablation-evaluation-input-spec/v2", inputs.specs[2]),
        ),
        python_policy=inputs.policy,
        worker_sha256=inputs.worker_sha256,
        rows=tuple(
            ScientificAblationEvaluationExpectationV2(
                row_index=index,
                capture_row_sha256=generated.sha256,
                capture_ack_sha256=ack.sha256,
                scoring_example_sha256=domain_sha256(
                    "iris/synthetic-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
                ),
                output_sha256=sha256(generated.payload.output_text.encode()).hexdigest(),
                output_byte_length=len(generated.payload.output_text.encode()),
                case_count=len(row.code_cases),
                target_nll_sum=generated.payload.target_nll_sum,
            )
            for index, (row, generated, ack) in enumerate(
                zip(inputs.rows, inputs.capture.rows, inputs.capture.acknowledgements, strict=True)
            )
        ),
    )
    result = IrisScientificAblationEvaluationPlanV2(
        sha256=_sha(_PLAN_DOMAIN, payload), payload=payload
    )
    if len(_file(result)) > _MAX_PLAN:
        raise ValueError("Score plan byte bound exceeded")
    return result


def build_scientific_ablation_evaluation_plan_v2(
    *,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
    maximum_capture_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationEvaluationPlanV2:
    """Read/audit exact source/capture/data; return a plan without scoring or writes."""
    try:
        inputs = _prepare(
            capture_path=capture_path,
            expected_plan_sha256=expected_plan_sha256,
            expected_capture_sha256=expected_capture_sha256,
            lifecycle_path=lifecycle_path,
            inference_path=inference_path,
            inputs_root=inputs_root,
            training=training,
            validation=validation,
            test=test,
            python_policy=python_policy,
            maximum_capture_bytes=maximum_capture_bytes,
        )
        result = _plan(inputs)
        _finish(inputs)
        return result
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error


def _check_score(
    plan: IrisScientificAblationEvaluationPlanV2, row: ScientificAblationScoredRowV2, index: int
) -> None:
    _exact_tree(row, ScientificAblationScoredRowV2)
    row = ScientificAblationScoredRowV2.model_validate(row.model_dump(mode="python"))
    e, g = plan.payload.rows[index], plan.payload.generation_plan.payload.rows[index]
    if (
        row.row_index != index
        or row.example_id != g.example_id
        or row.task_type != g.task_type
        or row.generation_example_sha256 != g.example_sha256
        or row.teacher_invocation_sha256 != g.teacher_invocation_sha256
        or row.teacher_target_sha256 != g.teacher_target_sha256
        or row.target_token_count != g.target_token_count
        or row.target_nll_sum != e.target_nll_sum
        or row.capture_row_sha256 != e.capture_row_sha256
        or row.capture_ack_sha256 != e.capture_ack_sha256
        or row.scoring_example_sha256 != e.scoring_example_sha256
        or row.output_sha256 != e.output_sha256
        or row.output_byte_length != e.output_byte_length
        or row.case_count != e.case_count
        or (
            row.execution is not None
            and (
                row.execution.policy != plan.payload.python_policy
                or row.execution.worker_sha256 != plan.payload.worker_sha256
            )
        )
    ):
        raise ValueError("Cached score differs from complete owned source/capture/row inventory")


def _genesis(plan: str) -> str:
    return domain_sha256(_ACK_DOMAIN + "/genesis", plan.encode())


def _attempt(plan: str, index: int, head: str) -> dict[str, object]:
    return {
        "kind": "iris-scientific-ablation-score-attempt-v2",
        "plan_sha256": plan,
        "row_index": index,
        "previous_ack_sha256": head,
        "status": "owned_before_scoring",
    }


def _ack(
    plan: str, index: int, head: str, row: IrisScientificAblationEvaluationRowV2
) -> IrisScientificAblationEvaluationAckV2:
    content = _file(row)
    payload = ScientificAblationEvaluationAckPayloadV2(
        plan_sha256=plan,
        row_index=index,
        previous_ack_sha256=head,
        row_sha256=row.sha256,
        row_file_sha256=sha256(content).hexdigest(),
        row_file_bytes=len(content),
    )
    return IrisScientificAblationEvaluationAckV2(sha256=_sha(_ACK_DOMAIN, payload), payload=payload)


def _complete(
    plan: IrisScientificAblationEvaluationPlanV2,
    head: str,
    report: IrisScientificAblationEvaluationV2,
) -> IrisScientificAblationEvaluationCompleteV2:
    payload = ScientificAblationEvaluationCompletePayloadV2(
        plan_sha256=plan.sha256,
        capture_sha256=plan.payload.capture.sha256,
        acknowledgement_head_sha256=head,
        evaluation_sha256=report.sha256,
        example_count=len(plan.payload.rows),
    )
    return IrisScientificAblationEvaluationCompleteV2(
        sha256=_sha(_COMPLETE_DOMAIN, payload), payload=payload
    )


def _scan(path: Path, depth: int, maximum: int) -> dict[Path, tuple[int, ...]]:
    entries: dict[Path, tuple[int, ...]] = {}
    size, directories, files = 0, 0, 0

    def visit(directory: Path, level: int, stage: bool) -> None:
        nonlocal size, directories, files
        _chain(directory)
        before = _signature(_regular(directory, directory=True))
        entries[directory] = before
        for entry in directory.iterdir():
            meta = entry.lstat()
            if stat.S_ISDIR(meta.st_mode):
                directories += 1
                staged = _STAGE.fullmatch(entry.name) is not None
                valid = not stage and (
                    (level == 0 and (staged or re.fullmatch(r"[0-9a-f]{64}", entry.name)))
                    or (
                        level == 1
                        and (staged or entry.name == "complete" or _ROW.fullmatch(entry.name))
                    )
                )
                if not valid or level >= 2 or directories > _MAX_DIRS:
                    raise ValueError("Foreign/deep/excessive score-store directory")
                visit(entry, level + 1, staged)
            else:
                files += 1
                if entry.name not in ({_LOCK} if level == 0 else _FILES) or files > _MAX_FILES:
                    raise ValueError("Foreign/excessive score-store file")
                meta = _regular(entry, directory=False)
                size += meta.st_size
                if size > maximum:
                    raise ValueError("Score ledger byte quota exceeded")
                entries[entry] = _signature(meta)
        if _signature(_regular(directory, directory=True)) != before:
            raise ValueError("Score store changed during inventory")

    visit(path, depth, False)
    return entries


def _canonical(path: Path, maximum: int) -> bytes:
    content = _read_file(path, maximum)
    if not content.endswith(b"\n"):
        raise ValueError("Canonical score JSON plus LF required")
    _json(content[:-1])
    return content


def _members(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir()}


def _read(
    path: Path, plan_sha: str, generation_sha: str, capture_sha: str, maximum: int
) -> IrisScientificAblationEvaluationProgressV2:
    snapshot = _scan(path, 1, maximum)
    content = _canonical(path / "plan.json", _MAX_PLAN)
    plan = IrisScientificAblationEvaluationPlanV2.model_validate_json(content[:-1])
    if (
        plan.sha256 != plan_sha
        or _sha(_PLAN_DOMAIN, plan.payload) != plan_sha
        or plan.payload.generation_plan.sha256 != generation_sha
        or plan.payload.capture.sha256 != capture_sha
        or _file(plan) != content
    ):
        raise ValueError("Independent evaluation/generation/complete-capture plan anchors differ")
    names = _members(path)
    row_names = sorted(name for name in names if _ROW.fullmatch(name))
    if row_names != [f"score-{index:05d}" for index in range(len(row_names))] or len(
        row_names
    ) > len(plan.payload.rows):
        raise ValueError("Score acknowledgement prefix reordered/gapped/duplicated")
    stages = sum(_STAGE.fullmatch(name) is not None for name in names)
    if names != {
        "plan.json",
        *row_names,
        *[name for name in names if _STAGE.fullmatch(name)],
        *({"complete"} if "complete" in names else set()),
    }:
        raise ValueError("Foreign score plan member")
    rows: list[IrisScientificAblationEvaluationRowV2] = []
    acks: list[IrisScientificAblationEvaluationAckV2] = []
    head = _genesis(plan.sha256)
    for index, name in enumerate(row_names):
        directory = path / name
        if _members(directory) != _ROW_FILES:
            raise ValueError("Score commit requires exact attempt/row/ack")
        if _canonical(directory / "attempt.json", 4096) != canonical_json_file_bytes(
            _attempt(plan.sha256, index, head)
        ):
            raise ValueError("Score computation attempt differs from acknowledgement prefix")
        row_content = _canonical(directory / "row.json", _MAX_ROW)
        row = IrisScientificAblationEvaluationRowV2.model_validate_json(row_content[:-1])
        if (
            row.payload.plan_sha256 != plan.sha256
            or _sha(_ROW_DOMAIN, row.payload) != row.sha256
            or _file(row) != row_content
        ):
            raise ValueError("Scored row/domain/canonical bytes differ")
        _check_score(plan, row.payload.measurements, index)
        ack_content = _canonical(directory / "ack.json", 4096)
        ack = IrisScientificAblationEvaluationAckV2.model_validate_json(ack_content[:-1])
        if ack != _ack(plan.sha256, index, head, row) or _file(ack) != ack_content:
            raise ValueError("Score acknowledgement/content chain differs")
        rows.append(row)
        acks.append(ack)
        head = ack.sha256
    complete, report = None, None
    if "complete" in names:
        directory = path / "complete"
        if len(rows) != len(plan.payload.rows) or _members(directory) != {
            "evaluation.json",
            "complete.json",
        }:
            raise ValueError("Complete score commit requires full coverage/report")
        content = _canonical(directory / "evaluation.json", _MAX_REPORT)
        report = IrisScientificAblationEvaluationV2.model_validate_json(content[:-1])
        if _checked(report) != content or (
            report.payload.plan != plan.payload.generation_plan
            or report.payload.capture != plan.payload.capture
            or report.payload.python_policy != plan.payload.python_policy
            or report.payload.rows != tuple(row.payload.measurements for row in rows)
        ):
            raise ValueError("Complete report differs from actual acknowledged scores")
        content = _canonical(directory / "complete.json", 4096)
        complete = IrisScientificAblationEvaluationCompleteV2.model_validate_json(content[:-1])
        if complete != _complete(plan, head, report) or _file(complete) != content:
            raise ValueError("Complete score/report/ack-head identity differs")
    if _scan(path, 1, maximum) != snapshot:
        raise ValueError("Score ledger changed during readback")
    return IrisScientificAblationEvaluationProgressV2(
        path, plan, tuple(rows), tuple(acks), complete, report, stages
    )


def read_scientific_ablation_evaluation_progress_v2(
    path: Path,
    *,
    expected_evaluation_plan_sha256: str,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    expected_acknowledgement_head_sha256: str | None = None,
    maximum_storage_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationEvaluationProgressV2:
    """Integrity view of a score prefix, not sandbox attestation or source-data re-audit."""
    try:
        for value in (
            expected_evaluation_plan_sha256,
            expected_plan_sha256,
            expected_capture_sha256,
        ):
            _digest(value)
        if expected_acknowledgement_head_sha256 is not None:
            _digest(expected_acknowledgement_head_sha256)
        _budgets(maximum_storage_bytes, 0)
        path = _path(path)
        if path.name != expected_evaluation_plan_sha256 or path.parent.name != _STORE:
            raise ValueError("Distinct score ledger/trusted plan directory required")
        result = _read(
            path,
            expected_evaluation_plan_sha256,
            expected_plan_sha256,
            expected_capture_sha256,
            maximum_storage_bytes,
        )
        head = (
            result.acknowledgements[-1].sha256
            if result.acknowledgements
            else _genesis(result.plan.sha256)
        )
        if (
            expected_acknowledgement_head_sha256 is not None
            and head != expected_acknowledgement_head_sha256
        ):
            raise ValueError("Independently trusted score acknowledgement head differs")
        return result
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error


def _budgets(maximum: int, reserve: int) -> None:
    if (
        type(maximum) is not int
        or not 1 <= maximum <= _MAX_STORAGE
        or type(reserve) is not int
        or not 0 <= reserve <= _MAX_STORAGE
    ):
        raise ValueError("Exact bounded score quota/free reserve required")


@dataclass(frozen=True, slots=True)
class _Claim:
    store: Path
    identity: tuple[int, int]
    lock: ProcessFileLock

    def check(self) -> None:
        self.lock.check()
        _chain(self.store)
        if _identity(_regular(self.store, directory=True)) != self.identity:
            raise ValueError("Score store replaced")


@dataclass(slots=True)
class _Budget:
    claim: _Claim
    maximum: int
    reserve: int
    bytes: int = 0
    directories: int = 0
    files: int = 0

    def reload(self) -> None:
        self.claim.check()
        snapshot = _scan(self.claim.store, 0, self.maximum)
        self.bytes = sum(value[2] for path, value in snapshot.items() if path.is_file())
        self.files = sum(path.is_file() for path in snapshot)
        self.directories = len(snapshot) - self.files - 1
        self.claim.check()

    def ensure(self, size: int, *, directories: int = 0, files: int = 0) -> None:
        self.claim.check()
        if (
            self.bytes + size > self.maximum
            or self.directories + directories > _MAX_DIRS
            or self.files + files > _MAX_FILES
            or shutil.disk_usage(self.claim.store).free < size + self.reserve
        ):
            raise ValueError("Score quota/count/free reserve exceeded before work/write")


@dataclass(slots=True)
class _Stage:
    path: Path
    identity: tuple[int, int]
    parent_identity: tuple[int, int]
    files: dict[str, tuple[int, int]] = field(default_factory=dict)

    def check(self, claim: _Claim) -> None:
        claim.check()
        _chain(self.path)
        if (
            _identity(_regular(self.path, directory=True)) != self.identity
            or _identity(_regular(self.path.parent, directory=True)) != self.parent_identity
            or _members(self.path) != set(self.files)
            or any(
                _identity(_regular(self.path / name, directory=False)) != identity
                for name, identity in self.files.items()
            )
        ):
            raise ValueError("Owned score stage/files/parent replaced")


def _stage(parent: Path, budget: _Budget) -> _Stage:
    budget.ensure(0, directories=1)
    parent_identity = _identity(_regular(parent, directory=True))
    path = parent / f".iris-score-{uuid4().hex}.tmp"
    os.mkdir(path, 0o700)
    budget.directories += 1
    return _Stage(path, _identity(_regular(path, directory=True)), parent_identity)


def _write(stage: _Stage, name: str, content: bytes, budget: _Budget) -> None:
    stage.check(budget.claim)
    if name not in _FILES:
        raise ValueError("Unknown score stage member")
    budget.ensure(len(content), files=1)
    path = stage.path / name
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    budget.files += 1
    budget.bytes += len(content)
    primary = None
    try:
        identity = _identity(os.fstat(descriptor))
        if identity != _identity(_regular(path, directory=False)):
            raise ValueError("Opened score file identity differs")
        stage.files[name] = identity
        _write_bytes(descriptor, content)
        stage.check(budget.claim)
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, lambda: os.close(descriptor))
    stage.check(budget.claim)
    if _read_file(path, len(content)) != content:
        raise ValueError("Exact score write readback differs")
    stage.check(budget.claim)


def _publish(stage: _Stage, target: Path, files: dict[str, bytes], budget: _Budget) -> None:
    stage.check(budget.claim)
    if set(stage.files) != set(files):
        raise ValueError("Exact owned score publication inventory differs")
    for name, content in files.items():
        if _read_file(stage.path / name, len(content)) != content:
            raise ValueError("Owned score publication bytes changed")
    stage.check(budget.claim)
    uncertain = False
    try:
        rename_directory_with_windows_retry(stage.path, target)
    except OSError:
        uncertain = True
        if not os.path.lexists(target):
            raise
    _chain(target)
    before = _signature(_regular(target, directory=True))
    owns_target = not uncertain or not os.path.lexists(stage.path)

    def check_published_identity() -> None:
        if owns_target and (
            _identity(_regular(target, directory=True)) != stage.identity
            or any(
                _identity(_regular(target / name, directory=False)) != identity
                for name, identity in stage.files.items()
            )
        ):
            raise ValueError("Published owned score directory/file identity was replaced")

    check_published_identity()
    if _members(target) != set(files):
        raise ValueError("Uncertain score publication inventory conflicts")
    for name, content in files.items():
        if _read_file(target / name, len(content)) != content:
            raise ValueError("Uncertain score publication bytes conflict")
    if _signature(_regular(target, directory=True)) != before:
        raise ValueError("Score commit changed during publication readback")
    check_published_identity()
    if os.path.lexists(stage.path):
        stage.check(budget.claim)
        # Only unusual outside-protocol identical winner needs a new global scan.
        # Ordinary rename, including lost return acknowledgement, is O(1) metadata.
        budget.reload()
    budget.claim.check()


def _capture_stamp(path: Path) -> dict[Path, tuple[int, ...]]:
    _chain(path)
    if _members(path) != _ROW_FILES:
        raise ValueError("Current captured-row members differ")
    return {
        path: _signature(_regular(path, directory=True)),
        **{path / name: _signature(_regular(path / name, directory=False)) for name in _ROW_FILES},
    }


def evaluate_scientific_ablation_capture_resumable_v2(
    *,
    root: Path,
    expected_evaluation_plan_sha256: str,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
    rows_per_call: int | None = None,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    _recovery: _KernelIntervalPermission | None = None,
    maximum_capture_bytes: int = 512 * 1024**2,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> IrisScientificAblationEvaluationProgressV2:
    """Score only unacknowledged rows under one owned immutable source/data plan.

    Full source/capture/corpus verification runs twice per invocation. The normal
    row loop uses constant-size source/current-row identity checks and incremental
    quota accounting; it never regenerates outputs, recomputes NLL or loads models.
    Stages survive interruption and consume quota. This is not execution attestation.
    """
    try:
        expected_chunk = checked_kernel_chunk_v1(
            expected_chunk,
            expected_plan_sha256=expected_evaluation_plan_sha256,
            rows_per_call=rows_per_call,
        )
        _checked_interval_permission(
            _recovery,
            root=root,
            namespace=_STORE,
            expected_chunk=expected_chunk,
            plan_sha256=expected_evaluation_plan_sha256,
            rows_per_call=rows_per_call,
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        _digest(expected_evaluation_plan_sha256)
        _budgets(maximum_storage_bytes, minimum_free_bytes)
        if rows_per_call is not None and (
            type(rows_per_call) is not int or not 1 <= rows_per_call <= 5000
        ):
            raise ValueError("Explicit bounded positive scoring chunk required")
        inputs = _prepare(
            capture_path=capture_path,
            expected_plan_sha256=expected_plan_sha256,
            expected_capture_sha256=expected_capture_sha256,
            lifecycle_path=lifecycle_path,
            inference_path=inference_path,
            inputs_root=inputs_root,
            training=training,
            validation=validation,
            test=test,
            python_policy=python_policy,
            maximum_capture_bytes=maximum_capture_bytes,
        )
        plan = _plan(inputs)
        if plan.sha256 != expected_evaluation_plan_sha256:
            raise ValueError("Independent score plan differs before filesystem writes")
        stamps = tuple(
            _capture_stamp(inputs.capture.path / f"row-{index:05d}")
            for index in range(len(inputs.rows))
        )
        inputs.check()
        store = _path(root) / _STORE
        for parent in (store, *store.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _chain(store)
        identity = _identity(_regular(store, directory=True))
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    result = None
    try:
        with process_file_lock(store / _LOCK) as lock:
            claim = _Claim(store, identity, lock)
            try:
                budget = _Budget(claim, maximum_storage_bytes, minimum_free_bytes)
                budget.reload()
                target = store / plan.sha256
                if not os.path.lexists(target):
                    if _recovery is not None:
                        raise ValueError(
                            "Recovery cannot initialize a missing original kernel plan"
                        )
                    check_kernel_chunk_prefix_v1(
                        expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=_genesis(plan.sha256),
                        acknowledged_rows=0,
                        total_rows=len(inputs.rows),
                        rows_per_call=rows_per_call,
                    )
                    content = _file(plan)
                    stage = _stage(store, budget)
                    _write(stage, "plan.json", content, budget)
                    _publish(stage, target, {"plan.json": content}, budget)
                progress = _read(
                    target,
                    plan.sha256,
                    expected_plan_sha256,
                    expected_capture_sha256,
                    maximum_storage_bytes,
                )
                if progress.plan != plan:
                    raise ValueError("Existing score job differs from fully owned source/data plan")
                rows, acks = list(progress.rows), list(progress.acknowledgements)
                head = acks[-1].sha256 if acks else _genesis(plan.sha256)
                if _recovery is None:
                    check_kernel_chunk_prefix_v1(
                        expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=head,
                        acknowledged_rows=len(rows),
                        total_rows=len(inputs.rows),
                        rows_per_call=rows_per_call,
                    )
                    stop = (
                        len(inputs.rows)
                        if rows_per_call is None
                        else min(len(inputs.rows), len(rows) + rows_per_call)
                    )
                else:
                    stop = _locked_interval_stop(
                        _recovery,
                        root=root,
                        namespace=_STORE,
                        expected_chunk=expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=head,
                        acknowledged_rows=len(rows),
                        total_rows=len(inputs.rows),
                        rows_per_call=rows_per_call,
                        acknowledgement_prefix=tuple(ack.sha256 for ack in acks),
                        kernel_lock=lock,
                        maximum_storage_bytes=maximum_storage_bytes,
                        minimum_free_bytes=minimum_free_bytes,
                    )
                if stop == len(inputs.rows) and progress.complete is None:
                    # Reserve bounded completion before expensive remaining sandbox work.
                    budget.ensure(
                        min(_MAX_REPORT, len(_file(plan)) + len(inputs.rows) * _MAX_ROW + 16384)
                        + 4096,
                        directories=1,
                        files=2,
                    )
                for index in range(len(rows), stop):
                    inputs.check()
                    capture_row_path = inputs.capture.path / f"row-{index:05d}"
                    if _capture_stamp(capture_row_path) != stamps[index]:
                        raise ValueError("Captured current-row identity changed before scoring")
                    budget.ensure(_MAX_ROW + 12288, directories=1, files=4)
                    stage = _stage(target, budget)
                    attempt = canonical_json_file_bytes(_attempt(plan.sha256, index, head))
                    _write(stage, "attempt.json", attempt, budget)
                    try:
                        score = _score_row(inputs, index)
                        _check_score(plan, score, index)
                        if _capture_stamp(capture_row_path) != stamps[index]:
                            raise ValueError("Captured current-row changed during scoring")
                        payload = ScientificAblationEvaluationRowPayloadV2(
                            plan_sha256=plan.sha256, measurements=score
                        )
                        row = IrisScientificAblationEvaluationRowV2(
                            sha256=_sha(_ROW_DOMAIN, payload), payload=payload
                        )
                        row_content = _file(row)
                        if len(row_content) > _MAX_ROW:
                            raise ValueError("Score-row byte budget exceeded")
                        ack = _ack(plan.sha256, index, head, row)
                        files = {
                            "attempt.json": attempt,
                            "row.json": row_content,
                            "ack.json": _file(ack),
                        }
                        _write(stage, "row.json", row_content, budget)
                        _write(stage, "ack.json", files["ack.json"], budget)
                        inputs.check()
                        _publish(stage, target / f"score-{index:05d}", files, budget)
                    except BaseException as error:
                        primary = error
                        if os.path.lexists(stage.path):
                            try:
                                _write(
                                    stage,
                                    "failure.json",
                                    canonical_json_file_bytes(
                                        {
                                            "kind": "iris-scientific-ablation-score-failure-v2",
                                            "status": "unacknowledged",
                                            "plan_sha256": plan.sha256,
                                            "row_index": index,
                                            "exception_type": type(error).__name__,
                                        }
                                    ),
                                    budget,
                                )
                            except BaseException as failure:
                                secondary.append(failure)
                        raise
                    rows.append(row)
                    acks.append(ack)
                    head = ack.sha256
                _finish(inputs)
                if len(rows) == len(inputs.rows) and progress.complete is None:
                    report = _assemble(inputs, tuple(row.payload.measurements for row in rows))
                    complete = _complete(plan, head, report)
                    files = {"evaluation.json": _checked(report), "complete.json": _file(complete)}
                    budget.ensure(sum(map(len, files.values())), directories=1, files=2)
                    stage = _stage(target, budget)
                    for name, content in files.items():
                        _write(stage, name, content, budget)
                    inputs.check()
                    _publish(stage, target / "complete", files, budget)
                budget.reload()
                result = _read(
                    target,
                    plan.sha256,
                    expected_plan_sha256,
                    expected_capture_sha256,
                    maximum_storage_bytes,
                )
                inputs.check()
                claim.check()
            except BaseException as error:
                if primary is None:
                    primary = error
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            secondary.append(error)
    if primary is not None:
        _fail(primary, secondary)
    if result is None:
        raise IrisScientificAblationEvaluationError("Score ledger produced no owned progress")
    return result


__all__ = [
    "IrisScientificAblationEvaluationProgressV2",
    "build_scientific_ablation_evaluation_plan_v2",
    "evaluate_scientific_ablation_capture_resumable_v2",
    "read_scientific_ablation_evaluation_progress_v2",
]
