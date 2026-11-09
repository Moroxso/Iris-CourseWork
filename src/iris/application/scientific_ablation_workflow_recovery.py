"""Explicit original-stop continuation and DISTINCT recovered DATA/ACK publication.

No original return is inferred. Existing recovery results take a storage-only
route; JSON never mints a kernel permission or adopts an unknown stage. Native
continuation requires the actual bound original workflow and live outer claim.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from iris.application import _scientific_ablation_kernel_recovery as interval
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application._scientific_ablation_kernel_finalization import (
    _finalize_acknowledged_kernel_data,
)
from iris.application.scientific_ablation_capture import IrisScientificAblationCaptureProgressV1
from iris.application.scientific_ablation_controlled_capture import (
    IrisScientificAblationControlledCaptureProgressV1,
)
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _read_file,
    _regular,
)
from iris.application.scientific_ablation_operator_workflow import _revalidate_operator_binding
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowV1,
    _entry,
    _prefix,
    _Seal,
)
from iris.application.scientific_ablation_workflow_io import (
    _RECOVERY_FILES,
    ScientificAblationWorkflowDataV1,
    _Claim,
    checked_workflow_model_v1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_recovery_v1 import (
    IrisScientificAblationWorkflowRecoveredResultV1,
    IrisScientificAblationWorkflowRecoveryAckV1,
    IrisScientificAblationWorkflowRecoveryIntentV1,
    ScientificAblationWorkflowRecoveredFileV1,
    ScientificAblationWorkflowRecoveredResultPayloadV1,
    ScientificAblationWorkflowRecoveryAckPayloadV1,
    ScientificAblationWorkflowRecoveryIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowAttemptV1,
    ScientificAblationWorkflowCaptureDataV1,
    ScientificAblationWorkflowScoreDataV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.process_file_lock import process_file_lock

_MAX_RECORD = 512 * 1024
_CAPACITY = len(_RECOVERY_FILES) * _MAX_RECORD


def _envelope[T: BaseModel](payload: BaseModel, cls: type[T]) -> T:
    value = payload.model_dump(mode="json")
    return cls.model_validate(
        {
            "sha256": domain_sha256("iris/" + value["kind"], canonical_json_bytes(value)),
            "payload": payload,
        }
    )


def _metadata[T: BaseModel](path: Path, cls: type[T]) -> T:
    raw = _read_file(path, _MAX_RECORD)
    value = cls.model_validate_json(raw)
    if steps.encoded(value) != raw:
        raise ValueError("Exact canonical recovered DATA required")
    return value


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkflowRecoveryReplyV1:
    """Retainable recovered DATA only, never an original-return/native token."""

    intent_json: bytes
    result_json: bytes
    operator_ack_json: bytes

    @property
    def result(self) -> IrisScientificAblationWorkflowRecoveredResultV1:
        return IrisScientificAblationWorkflowRecoveredResultV1.model_validate_json(self.result_json)

    @property
    def operator_ack(self) -> IrisScientificAblationWorkflowRecoveryAckV1:
        return IrisScientificAblationWorkflowRecoveryAckV1.model_validate_json(
            self.operator_ack_json
        )


class _Journal:
    """Bounded metadata publisher under the SAME live workflow claim/quota."""

    def __init__(
        self, seal: _Seal, command: IrisScientificAblationWorkflowCommandIntentV1, claim: _Claim
    ) -> None:
        self.seal, self.command, self.claim = seal, command, claim
        self.intent = _envelope(
            ScientificAblationWorkflowRecoveryIntentPayloadV1(original_command=command),
            IrisScientificAblationWorkflowRecoveryIntentV1,
        )
        self.path = seal.path / "recoveries" / self.intent.sha256[:2] / self.intent.sha256
        self.identities: tuple[tuple[int, int], ...] | None = None
        self._check()

    def _check(self) -> None:
        seal = _entry(self.seal.job, busy=True)
        if (
            seal.store is not self.seal.store
            or seal.factory is not self.seal.factory
            or seal.operator_binding is not self.seal.operator_binding
            or seal.storage_definition != self.seal.storage_definition
        ):
            raise ValueError("Original recovery source/factory/namespace changed")
        seal.store._check_claim(self.claim)
        _revalidate_operator_binding(seal.job, seal)
        original = seal.path / "handoffs" / self.command.sha256[:2] / self.command.sha256
        _chain(original)
        if {p.name for p in original.iterdir()} != {"intent.json"} or _read_file(
            original / "intent.json", _MAX_RECORD
        ) != steps.encoded(self.command):
            raise ValueError(
                "Exact intent-only original request required; retained DATA is separate"
            )
        attempt_raw = _read_file(
            seal.path / f"phase-{self.command.payload.phase.index:03d}.attempt.json", _MAX_RECORD
        )
        attempt = IrisScientificAblationWorkflowAttemptV1.model_validate_json(attempt_raw)
        request = self.command.payload
        if (
            steps.encoded(attempt) != attempt_raw
            or attempt.payload.plan_sha256 != seal.store.plan.sha256
            or attempt.payload.phase != request.phase
            or attempt.payload.predecessor_sha256
            != (request.completed_phase_sha256[-1] if request.completed_phase_sha256 else None)
            or attempt.payload.recovery != "acknowledged_kernel_prefix_only"
        ):
            raise ValueError("Actual original resumable attempt must match recovered DATA")
        if self.identities is not None and self.identities != tuple(
            _identity(_regular(p, directory=True))
            for p in (self.path.parent.parent, self.path.parent, self.path)
        ):
            raise ValueError("Original recovery journal directory identity changed")

    def hold(self) -> _Claim:
        self._check()
        used = sum(
            len(_read_file(self.path / name, _MAX_RECORD))
            for name in _RECOVERY_FILES
            if os.path.lexists(self.path / name)
        )
        return self.claim.with_reserved_bytes(_CAPACITY - used)

    def write(self, name: str, value: BaseModel) -> None:
        self._check()
        raw = steps.encoded(value)
        if name not in _RECOVERY_FILES or not 1 <= len(raw) <= _MAX_RECORD:
            raise ValueError("Closed bounded recovery record required")
        self.hold().reserve(0)
        self.path.mkdir(parents=True, exist_ok=True)
        _chain(self.path)
        identities = tuple(
            _identity(_regular(p, directory=True))
            for p in (self.path.parent.parent, self.path.parent, self.path)
        )
        if self.identities is not None and self.identities != identities:
            raise ValueError("Recovery directory replaced during publication")
        self.identities = identities
        target = self.path / name
        if os.path.lexists(target):
            if _read_file(target, _MAX_RECORD) != raw:
                raise ValueError("Existing recovered DATA conflicts; never overwrite it")
            return
        hold = self.hold()
        view = replace(hold, reserved_bytes=hold.reserved_bytes - len(raw))
        view.reserve(len(raw))
        stage = self.seal.path / f".iris-recovery-{uuid4().hex}.tmp"
        identity = self.seal.store._write_new(stage, raw, view)
        self._check()
        if os.path.lexists(target) or _identity(_regular(stage, directory=False)) != identity:
            raise ValueError("Recovery destination/stage changed; unknown stages preserved")
        os.rename(stage, target)
        if (
            _identity(_regular(target, directory=False)) != identity
            or _read_file(target, _MAX_RECORD) != raw
        ):
            raise ValueError("Published recovered DATA identity/bytes differ")
        self.hold().reserve(0)


@dataclass(frozen=True, slots=True)
class _Observation:
    kernel: IrisScientificAblationWorkflowKernelProgressV1
    observed_rows: int
    phase_data: ScientificAblationWorkflowDataV1 | None


def _complete_data(
    seal: _Seal,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    result: steps.CaptureProgress | steps.ScoreProgress,
) -> ScientificAblationWorkflowDataV1 | None:
    chunk = command.payload.expected_chunk
    assert chunk is not None
    if chunk.stop_row_index != chunk.total_rows:
        return None
    if result.complete is None:
        raise ValueError("All row ACKs exist but aggregate DATA still needs storage finalization")
    phase = command.payload.phase
    protocol: Literal["ordinary", "controlled"] = (
        "ordinary" if phase.stream_index is None else "controlled"
    )
    files: dict[str, bytes]
    if isinstance(
        result,
        (
            IrisScientificAblationCaptureProgressV1,
            IrisScientificAblationControlledCaptureProgressV1,
        ),
    ):
        if phase.kind != "capture":
            raise ValueError("Capture DATA cannot replace an original score phase")
        files = {
            "capture.json": steps.encoded(
                ScientificAblationWorkflowCaptureDataV1(
                    protocol=protocol,
                    plan_sha256=result.plan.sha256,
                    input_sha256=result.plan.payload.input.sha256,
                    capture_sha256=result.complete.sha256,
                    examples=len(result.rows),
                )
            )
        }
    else:
        if phase.kind != "score":
            raise ValueError("Score DATA cannot replace an original capture phase")
        if result.evaluation is None:
            raise ValueError("Complete score DATA has no original evaluation")
        args = steps._score_args(seal.store, prefix[-1])
        files = {
            "evaluation.json": steps.encoded(
                ScientificAblationWorkflowScoreDataV1(
                    protocol=protocol,
                    evaluation_plan_sha256=result.plan.sha256,
                    evaluation_sha256=result.evaluation.sha256,
                    complete_sha256=result.complete.sha256,
                    capture_plan_sha256=args["expected_plan_sha256"],
                    capture_sha256=args["expected_capture_sha256"],
                    examples=len(result.rows),
                )
            )
        }
    data = seal.store.make_data(
        phase,
        command.payload.completed_phase_sha256[-1] if prefix else None,
        origin="kernel_data",
        files=files,
    )
    steps.validate_data(seal.store, data, prefix)
    return data


def _observe(
    seal: _Seal,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    claim: _Claim,
) -> _Observation:
    """Read original full chain under its lock, then BOUND DATA to the old stop."""
    phase, chunk = command.payload.phase, command.payload.expected_chunk
    assert chunk is not None
    namespace, _, domain = interval._PROTOCOLS[phase.kind, phase.stream_index is not None]
    root = steps._kernel(seal.store, phase) / namespace
    _chain(root)
    _regular(root, directory=True)
    claim.reserve(0)
    with process_file_lock(root / ".iris-owner.lock") as lock:
        result: steps.CaptureProgress | steps.ScoreProgress
        result = (
            kernels._capture_result(seal.store, phase, chunk.kernel_plan_sha256)
            if phase.kind == "capture"
            else kernels._score_result(seal.store, phase, prefix, chunk.kernel_plan_sha256)
        )
        lock.check()
        acks = tuple(a.sha256 for a in result.acknowledgements)
        genesis = domain_sha256(domain + "/genesis", chunk.kernel_plan_sha256.encode())
        if (
            result.plan.sha256 != chunk.kernel_plan_sha256
            or len(result.rows) != len(acks)
            or len(result.plan.payload.rows) != chunk.total_rows
            or len(acks) < chunk.acknowledged_rows
            or (acks[chunk.acknowledged_rows - 1] if chunk.acknowledged_rows else genesis)
            != chunk.acknowledgement_head_sha256
        ):
            raise ValueError("Original starting ACK/plan/total differs from actual verified chain")
        count = min(len(acks), chunk.stop_row_index)
        if count == chunk.total_rows and result.complete is None:
            result = _finalize_acknowledged_kernel_data(
                seal.store,
                command,
                prefix,
                claim,
                lock,
                expected_acknowledgement_head_sha256=acks[-1],
            )
        kernel = IrisScientificAblationWorkflowKernelProgressV1(
            workflow_plan_sha256=seal.store.plan.sha256,
            phase=phase,
            completed_phase_sha256=command.payload.completed_phase_sha256,
            phase_is_acknowledged=False,
            kernel_plan_sha256=chunk.kernel_plan_sha256,
            acknowledgement_head_sha256=acks[count - 1] if count else genesis,
            acknowledged_rows=count,
            total_rows=chunk.total_rows,
            kernel_complete_sha256=(
                result.complete.sha256
                if count == chunk.total_rows and result.complete is not None
                else None
            ),
            uncommitted_stage_count=result.uncommitted_stage_count,
        )
        data = _complete_data(seal, command, prefix, result) if count == chunk.total_rows else None
        lock.check()
        claim.reserve(0)
        return _Observation(kernel, len(acks), data)


def _checked_result(
    journal: _Journal, result: IrisScientificAblationWorkflowRecoveredResultV1
) -> None:
    p, request = result.payload, journal.command.payload
    chunk = request.expected_chunk
    assert chunk is not None
    k = p.kernel_progress
    if (
        p.recovery_request_sha256 != journal.intent.sha256
        or p.original_request_sha256 != journal.command.sha256
        or k.workflow_plan_sha256 != journal.seal.store.plan.sha256
        or k.phase != request.phase
        or k.completed_phase_sha256 != request.completed_phase_sha256
        or k.kernel_plan_sha256 != chunk.kernel_plan_sha256
        or k.total_rows != chunk.total_rows
        or k.acknowledged_rows != chunk.stop_row_index
        or (
            p.resumed_from_row is not None
            and not chunk.acknowledged_rows <= p.resumed_from_row <= chunk.stop_row_index
        )
        or (
            p.phase_record is not None
            and p.phase_record.payload.predecessor_sha256
            != (request.completed_phase_sha256[-1] if request.completed_phase_sha256 else None)
        )
    ):
        raise ValueError("Recovered DATA differs from the SAME original immutable request")


def _recover_kernel_command_v1(
    job: IrisScientificAblationWorkflowV1,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    *,
    expected_operator_ack_sha256: str | None = None,
) -> ScientificAblationWorkflowRecoveryReplyV1:
    """Explicit recovery, never entered by ordinary dispatch/stale v1 retry."""
    command = checked_workflow_model_v1(command, IrisScientificAblationWorkflowCommandIntentV1)
    if expected_operator_ack_sha256 is not None:
        _digest(expected_operator_ack_sha256)
    seal = _entry(job)
    request, chunk = command.payload, command.payload.expected_chunk
    if (
        chunk is None
        or seal.operator_binding is None
        or seal.ready is not None
        or steps.encoded(request.namespace) != seal.storage_definition
        or seal.store.phase(request.phase) != request.phase
        or seal.anchors[: request.phase.index] != request.completed_phase_sha256
        or len(seal.anchors) not in (request.phase.index, request.phase.index + 1)
    ):
        raise ValueError("Original bound kernel owner/request/BEFORE anchors required")
    job._state = "busy"
    try:
        with seal.store.claim() as claim:
            prefix = _prefix(seal)[: request.phase.index]
            _revalidate_operator_binding(job, seal)
            journal = _Journal(seal, command, claim)
            result_path = journal.path / "result.json"
            ack_path = journal.path / "operator-ack.json"
            if expected_operator_ack_sha256 is not None:
                ack = _metadata(ack_path, IrisScientificAblationWorkflowRecoveryAckV1)
                if ack.sha256 != expected_operator_ack_sha256:
                    raise ValueError("Independent recovered-DATA ACK differs")
            if os.path.lexists(ack_path) and not os.path.lexists(result_path):
                raise ValueError("Recovered ACK has no retained result")
            if os.path.lexists(result_path) and not os.path.lexists(journal.path / "intent.json"):
                raise ValueError("Recovered result has no retained exact recovery intent")
            journal.write("intent.json", journal.intent)
            held = journal.hold()
            retained_result = os.path.lexists(result_path)
            if retained_result:
                result = _metadata(result_path, IrisScientificAblationWorkflowRecoveredResultV1)
                _checked_result(journal, result)
                observed = _observe(seal, command, prefix, held)
                if (
                    observed.kernel.acknowledged_rows != chunk.stop_row_index
                    or observed.observed_rows < result.payload.observed_acknowledged_rows
                    or observed.kernel.acknowledgement_head_sha256
                    != result.payload.kernel_progress.acknowledgement_head_sha256
                    or observed.kernel.kernel_complete_sha256
                    != result.payload.kernel_progress.kernel_complete_sha256
                    or (None if observed.phase_data is None else observed.phase_data.record)
                    != result.payload.phase_record
                    or (() if observed.phase_data is None else observed.phase_data.files)
                    != tuple(
                        (f.name, f.content.encode("utf-8")) for f in result.payload.phase_files
                    )
                ):
                    raise ValueError("Retained recovery DATA differs from verified original kernel")
            else:
                if seal.anchors != request.completed_phase_sha256:
                    raise ValueError("Lost recovery result cannot be inferred from a newer phase")
                observed = _observe(seal, command, prefix, held)
                resumed: int | None = None
                if observed.observed_rows < chunk.stop_row_index:
                    if os.path.lexists(seal.path / f"phase-{request.phase.index:03d}"):
                        raise ValueError(
                            "Unacknowledged phase DATA cannot authorize more computation"
                        )
                    with interval._original_interval_permission(seal, command, held) as permission:
                        produced = steps.execute_produced(
                            seal.store,
                            request.phase,
                            prefix,
                            seal.factory,
                            held,
                            request.rows_per_call,
                            expected_chunk=chunk,
                            _recovery=permission,
                        )
                        resumed = interval._interval_start_rows(permission)
                    if produced.kernel_progress is None:
                        raise ValueError("Original continued kernel did not return a DATA snapshot")
                    kernel = IrisScientificAblationWorkflowKernelProgressV1.model_validate_json(
                        produced.kernel_progress
                    )
                    observed = _Observation(kernel, kernel.acknowledged_rows, produced.data())
                data = observed.phase_data
                result = _envelope(
                    ScientificAblationWorkflowRecoveredResultPayloadV1(
                        recovery_request_sha256=journal.intent.sha256,
                        original_request_sha256=command.sha256,
                        kernel_progress=observed.kernel,
                        observed_acknowledged_rows=observed.observed_rows,
                        resumed_from_row=resumed,
                        route="verified_kernel_data"
                        if resumed is None
                        else "continued_original_interval",
                        phase_record=None if data is None else data.record,
                        phase_files=tuple(
                            ScientificAblationWorkflowRecoveredFileV1.model_validate(
                                dict(name=name, content=raw.decode("utf-8"))
                            )
                            for name, raw in (() if data is None else data.files)
                        ),
                    ),
                    IrisScientificAblationWorkflowRecoveredResultV1,
                )
                _checked_result(journal, result)
                journal.write("result.json", result)
            data = observed.phase_data
            after = request.completed_phase_sha256
            if data is not None:
                steps.validate_data(seal.store, data, prefix)
                committed = seal.store.commit(data, journal.hold())
                if committed != data:
                    raise ValueError("Recovered compact phase publication differs")
                after = (*after, data.record.sha256)
            live_after = after
            if seal.anchors not in (request.completed_phase_sha256, after):
                # A retained PARTIAL result/ACK describes the historical original
                # stop, not the current workflow cursor. A distinct later command
                # may already have completed this SAME phase. _prefix above and
                # below authenticates that independently anchored physical DATA;
                # _observe has verified the full original kernel while bounding
                # the returned snapshot to the old stop. Keep BOTH truths: the
                # exact old recovered ACK and the current independent prefix.
                # Missing results, incomplete kernels, unrelated/later phases or
                # changed predecessor anchors never take this storage-only route.
                if not (
                    retained_result
                    and data is None
                    and chunk.stop_row_index < chunk.total_rows
                    and observed.observed_rows == chunk.total_rows
                    and len(seal.anchors) == request.phase.index + 1
                    and seal.anchors[: request.phase.index] == request.completed_phase_sha256
                ):
                    raise ValueError("Original independent phase prefix changed during recovery")
                live_after = seal.anchors
            ack = _envelope(
                ScientificAblationWorkflowRecoveryAckPayloadV1(
                    recovery_request_sha256=journal.intent.sha256,
                    original_request_sha256=command.sha256,
                    recovered_result_sha256=result.sha256,
                    completed_phase_sha256=after,
                ),
                IrisScientificAblationWorkflowRecoveryAckV1,
            )
            journal.write("operator-ack.json", ack)
            if (
                expected_operator_ack_sha256 is not None
                and ack.sha256 != expected_operator_ack_sha256
            ):
                raise ValueError("Exact independently retained recovery ACK required")
            seal = replace(seal, anchors=live_after, recovered=True)
            job._seal = seal
            _prefix(seal)
            claim.reserve(0)
            return ScientificAblationWorkflowRecoveryReplyV1(
                steps.encoded(journal.intent), steps.encoded(result), steps.encoded(ack)
            )
    finally:
        if getattr(job, "_seal", None) is seal and getattr(job, "_state", None) == "busy":
            job._state = "idle"
        else:
            job._state = "failed"
