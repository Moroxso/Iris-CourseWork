"""Original-factory compound workflow with explicit native/DATA crash boundaries.

One advance performs one existing kernel chunk or original native phase. Retain
returned phase anchors independently before the next advance. Stored DATA never
restores native authority. A hard exit with a native attempt but no independently
retained ready DATA remains uncertain, not a request to measure automatically.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _path,
    _read_file,
    _regular,
)
from iris.application.scientific_ablation_kernel_chunk import checked_kernel_chunk_v1
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
    checked_workflow_model_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPhaseDataV1,
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)

if TYPE_CHECKING:
    from iris.application._scientific_ablation_workflow_handoff import _Completion, _Receiver
    from iris.application.scientific_ablation_operator_workflow import _OperatorWorkflowBinding


class IrisScientificAblationWorkflowError(RuntimeError):
    """Wrong or changed process-local workflow owner; no implicit native repair."""


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkflowProgressV1:
    """Retainable DATA acknowledgements, never a native job or scientific verdict."""

    plan_sha256: str
    completed_phase_sha256: tuple[str, ...]
    next_phase: str | None
    complete: bool
    recovered_prefix_is_data_only: bool


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkflowAdvanceResultV1:
    """Immutable original-return DATA; NOT a durable command or producer ACK.

    Kernel bytes describe the exact original return at the requested BEFORE
    prefix, even if a standalone writer has subsequently advanced its head.
    Native phase bytes are the original result, never a new measurement.
    Constructing/deserializing this DATA grants no execution or recovery rights.
    """

    progress: ScientificAblationWorkflowProgressV1
    requested_phase_json: bytes
    completed_phase_sha256_before: tuple[str, ...]
    expected_chunk_json: bytes | None
    rows_per_call: int | None
    kernel_progress_json: bytes | None
    phase_record_json: bytes | None
    phase_files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True, slots=True)
class _Ready:
    record: bytes | None
    files: tuple[tuple[str, bytes], ...]
    kernel_chunk: bytes | None = None
    kernel_progress: bytes | None = None
    producer_result_requested: bool = False
    rows_per_call: int | None = None
    produced: steps._Produced | None = None
    receiver: "_Receiver | None" = None
    handoff_token: object | None = None

    def data(self) -> ScientificAblationWorkflowDataV1 | None:
        if self.record is None:
            if self.files or self.receiver is None or self.produced is None:
                raise IrisScientificAblationWorkflowError(
                    "Only an actual pending partial handoff has no phase DATA"
                )
            return None
        record = IrisScientificAblationWorkflowPhaseDataV1.model_validate_json(self.record)
        if steps.encoded(record) != self.record:
            raise IrisScientificAblationWorkflowError("Original private ready DATA bytes differ")
        return ScientificAblationWorkflowDataV1(record, self.files)


@dataclass(frozen=True, slots=True, eq=False)
class _Seal:
    job: "IrisScientificAblationWorkflowV1"
    store: ScientificAblationWorkflowStoreV1
    path: Path
    plan: bytes
    factory: steps.Factory | None
    anchors: tuple[str, ...]
    recovered: bool
    ready: _Ready | None = None
    operator_binding: "_OperatorWorkflowBinding | None" = None
    storage_definition: bytes | None = None
    completed_handoff: "_Completion | None" = None


@dataclass(frozen=True, slots=True)
class _Expected:
    phase: bytes
    anchors: tuple[str, ...]
    kernel_chunk: bytes | None = None
    producer_result_requested: bool = False


class IrisScientificAblationWorkflowV1:
    """Explicit trusted factory; no public constructor or JSON native restoration."""

    __slots__ = ("_seal", "_state")
    _seal: _Seal
    _state: str

    def __init__(self) -> None:
        raise IrisScientificAblationWorkflowError("Use create_scientific_ablation_workflow_v1")

    @property
    def progress(self) -> ScientificAblationWorkflowProgressV1:
        seal = _entry(self)
        _prefix(seal)
        return _progress(seal)

    def advance(self, *, rows_per_call: int | None = None) -> ScientificAblationWorkflowProgressV1:
        """Advance once; publication/commit retry NEVER reruns a native producer."""
        return _advance(self, rows_per_call)

    def inspect_kernel(
        self,
        *,
        expected_phase: ScientificAblationWorkflowPhaseV1,
        expected_kernel_plan_sha256: str,
        expected_acknowledgement_head_sha256: str | None = None,
    ) -> IrisScientificAblationWorkflowKernelProgressV1:
        """Read explicit existing row-ACK DATA; never advance or adopt a ready phase.

        This is an integrity snapshot, not a mutation lock or execution permission.
        A durable command must separately bind/revalidate its exact chunk and ACK.
        """
        from iris.application.scientific_ablation_workflow_kernel import (
            read_scientific_ablation_workflow_kernel_progress_v1,
        )

        return read_scientific_ablation_workflow_kernel_progress_v1(
            self,
            expected_phase=expected_phase,
            expected_kernel_plan_sha256=expected_kernel_plan_sha256,
            expected_acknowledgement_head_sha256=expected_acknowledgement_head_sha256,
        )

    def advance_expected(
        self,
        *,
        expected_phase: ScientificAblationWorkflowPhaseV1,
        expected_completed_phase_sha256: tuple[str, ...],
        rows_per_call: int | None = None,
        expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    ) -> ScientificAblationWorkflowProgressV1:
        """Execute only this explicit phase at this independently retained prefix.

        Check before storage and again inside the original claim. A partial row
        chunk may leave the phase prefix unchanged; this is NOT an idempotent
        durable command receipt. A CLI must additionally retain its own command
        acknowledgements, not replay this method to discover lost native DATA.
        An exact chunk is forwarded to the ORIGINAL kernel's locked prefix
        check. Existing row ACKs are also checked before factory/phase attempts.
        A first capture still reconstructs its original owner before its kernel
        authenticates the new plan; this is not model-free native planning.
        """
        expected = _expected_for(
            self, expected_phase, expected_completed_phase_sha256, rows_per_call, expected_chunk
        )
        return _advance(self, rows_per_call, expected=expected)

    def advance_expected_with_result(
        self,
        *,
        expected_phase: ScientificAblationWorkflowPhaseV1,
        expected_completed_phase_sha256: tuple[str, ...],
        rows_per_call: int | None = None,
        expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    ) -> ScientificAblationWorkflowAdvanceResultV1:
        """Return this original execution's exact outcome, not a later inspection.

        Capture/score require an explicit exact chunk; native/publication phases
        require no row budget. Process-local storage-only retries preserve the
        SAME result bytes and request. The legacy namespace has no independent
        handoff. The private concrete command namespace additionally receives
        the actual result and stores producer/operator ACKs INSIDE this claim.
        Public command dispatch and saved-request DATA replay are separate
        wrappers; bounded multi-row/cold native recovery is NOT implied.
        """
        expected = _expected_for(
            self,
            expected_phase,
            expected_completed_phase_sha256,
            rows_per_call,
            expected_chunk,
            producer_result_requested=True,
        )
        return _advance(self, rows_per_call, expected=expected, with_result=True)


def _expected_for(
    job: IrisScientificAblationWorkflowV1,
    expected_phase: ScientificAblationWorkflowPhaseV1,
    expected_completed_phase_sha256: tuple[str, ...],
    rows_per_call: int | None,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None,
    *,
    producer_result_requested: bool = False,
) -> _Expected:
    seal = _entry(job)
    phase = checked_workflow_model_v1(expected_phase, ScientificAblationWorkflowPhaseV1)
    if (
        type(expected_completed_phase_sha256) is not tuple
        or len(expected_completed_phase_sha256) > 134
    ):
        raise ValueError("Exact bounded independent workflow prefix required")
    for digest in expected_completed_phase_sha256:
        _digest(digest)
    if producer_result_requested:
        if phase.kind in ("capture", "score") and expected_chunk is None:
            raise ValueError("Exact original producer result requires an explicit kernel chunk")
        if phase.kind not in ("capture", "score") and rows_per_call is not None:
            raise ValueError("A native or publication phase has no row-chunk budget")
    chunk = None
    if expected_chunk is not None:
        if phase.kind not in ("capture", "score"):
            raise ValueError("An exact kernel chunk requires a capture or score phase")
        owned_chunk = checked_workflow_model_v1(
            expected_chunk, ScientificAblationKernelChunkExpectationV1
        )
        checked_chunk = checked_kernel_chunk_v1(
            owned_chunk,
            expected_plan_sha256=owned_chunk.kernel_plan_sha256,
            rows_per_call=rows_per_call,
        )
        assert checked_chunk is not None
        chunk = steps.encoded(checked_chunk)
    expected = _Expected(
        steps.encoded(phase),
        expected_completed_phase_sha256,
        chunk,
        producer_result_requested,
    )
    _check_expected(seal, expected)
    return expected


def _entry(job: IrisScientificAblationWorkflowV1, *, busy: bool = False) -> _Seal:
    if type(job) is not IrisScientificAblationWorkflowV1:
        raise IrisScientificAblationWorkflowError("Exact original workflow handle required")
    seal = getattr(job, "_seal", None)
    if (
        type(seal) is not _Seal
        or seal.job is not job
        or getattr(job, "_state", None) != ("busy" if busy else "idle")
        or type(seal.store) is not ScientificAblationWorkflowStoreV1
        or seal.store.path != seal.path
        or steps.encoded(seal.store.plan) != seal.plan
        or seal.store.namespace_definition_json != seal.storage_definition
        or seal.store._namespace_path() != seal.path
    ):
        raise IrisScientificAblationWorkflowError("Original workflow owner/state/plan changed")
    return seal


def _same(job: IrisScientificAblationWorkflowV1, seal: _Seal) -> None:
    if _entry(job, busy=True) is not seal:
        raise IrisScientificAblationWorkflowError("Workflow was replaced during original work")


def _check_operator_source(job: IrisScientificAblationWorkflowV1, seal: _Seal) -> None:
    """Revalidate the concrete original operator binding, never a caller callback.

    Generic trusted-factory workflows retain their original contract. Operator
    workflows must also recheck source/test intent for model-free phases and
    storage-only retries. This check grants no result/ACK or native recovery.
    """
    if seal.operator_binding is not None:
        from iris.application.scientific_ablation_operator_workflow import (
            _revalidate_operator_binding,
        )

        _revalidate_operator_binding(job, seal)
        if job._seal is not seal:
            raise IrisScientificAblationWorkflowError("Operator preflight replaced the workflow")


def _prefix(seal: _Seal) -> tuple[ScientificAblationWorkflowDataV1, ...]:
    if os.path.lexists(seal.path):
        _chain(seal.path)
        identity = _identity(_regular(seal.path, directory=True))
        if seal.store._identity is not None and identity != seal.store._identity:
            raise IrisScientificAblationWorkflowError("Original workflow root identity changed")
        seal.store._layout()
        plan_path = seal.path / "plan.json"
        if os.path.lexists(plan_path) and _read_file(plan_path, len(seal.plan)) != seal.plan:
            raise IrisScientificAblationWorkflowError("Original workflow on-disk intent changed")
    phases = scientific_ablation_workflow_phases_v1(seal.store.plan)
    if type(seal.anchors) is not tuple or len(seal.anchors) > len(phases):
        raise IrisScientificAblationWorkflowError(
            "Exact independent original phase anchors required"
        )
    result: tuple[ScientificAblationWorkflowDataV1, ...] = ()
    for phase, digest in zip(phases, seal.anchors, strict=False):
        _digest(digest)
        data = seal.store.read(phase, expected_sha256=digest)
        previous = result[-1].record.sha256 if result else None
        if data.record.payload.predecessor_sha256 != previous:
            raise IrisScientificAblationWorkflowError(
                "Physical workflow predecessor history differs"
            )
        steps.validate_data(seal.store, data, result)
        result += (data,)
    # A future attempt/completion is not a valid linear continuation of this prefix.
    for phase in phases[len(result) + 1 :]:
        if any(
            os.path.lexists(seal.path / name)
            for name in (
                f"phase-{phase.index:03d}",
                f"phase-{phase.index:03d}.attempt.json",
            )
        ):
            raise IrisScientificAblationWorkflowError("Out-of-order physical workflow history")
    return result


def _progress(seal: _Seal) -> ScientificAblationWorkflowProgressV1:
    phases = scientific_ablation_workflow_phases_v1(seal.store.plan)
    complete = len(seal.anchors) == len(phases)
    return ScientificAblationWorkflowProgressV1(
        seal.store.plan.sha256,
        seal.anchors,
        None if complete else phases[len(seal.anchors)].name,
        complete,
        seal.recovered,
    )


def _check_expected(seal: _Seal, expected: _Expected | None) -> None:
    if seal.ready is not None and seal.ready.producer_result_requested != (
        expected is not None and expected.producer_result_requested
    ):
        raise IrisScientificAblationWorkflowError(
            "Process-local producer result cannot switch to or from legacy phase-only retry"
        )
    if expected is None:
        return
    phases = scientific_ablation_workflow_phases_v1(seal.store.plan)
    if (
        seal.anchors != expected.anchors
        or len(seal.anchors) >= len(phases)
        or steps.encoded(phases[len(seal.anchors)]) != expected.phase
    ):
        raise IrisScientificAblationWorkflowError(
            "Explicit expected workflow phase or independent prefix differs"
        )
    if (
        expected.kernel_chunk is not None
        and seal.ready is not None
        and seal.ready.kernel_chunk != expected.kernel_chunk
    ):
        raise IrisScientificAblationWorkflowError(
            "Process-local ready DATA belongs to a different exact chunk request"
        )


def _expected_chunk(
    expected: _Expected | None,
) -> ScientificAblationKernelChunkExpectationV1 | None:
    if expected is None or expected.kernel_chunk is None:
        return None
    return ScientificAblationKernelChunkExpectationV1.model_validate_json(expected.kernel_chunk)


def _check_chunk(
    seal: _Seal,
    expected: _Expected | None,
    rows_per_call: int | None,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...] | None = None,
) -> None:
    chunk = _expected_chunk(expected)
    if chunk is None or seal.ready is not None:
        return
    from iris.application.scientific_ablation_workflow_kernel import _check_chunk_before_native

    assert expected is not None
    phase = ScientificAblationWorkflowPhaseV1.model_validate_json(expected.phase)
    _check_chunk_before_native(
        seal.store, phase, _prefix(seal) if prefix is None else prefix, chunk, rows_per_call
    )


def create_scientific_ablation_workflow_v1(
    *,
    root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    make_fresh_run: steps.Factory | None = None,
    completed_phase_sha256: tuple[str, ...] = (),
) -> IrisScientificAblationWorkflowV1:
    """Snapshot intent and independently retained DATA prefix, without native work.

    A factory is trusted caller code, not metadata or a completion callback. It
    must reproduce the original standalone initial run. None allows only phases
    that already require no original native owner (score/publication/inspection).
    """
    owned = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
    return _create_workflow_with_store(
        ScientificAblationWorkflowStoreV1(_path(root), owned),
        make_fresh_run=make_fresh_run,
        completed_phase_sha256=completed_phase_sha256,
    )


def _create_workflow_with_store(
    store: ScientificAblationWorkflowStoreV1,
    *,
    make_fresh_run: steps.Factory | None,
    completed_phase_sha256: tuple[str, ...],
) -> IrisScientificAblationWorkflowV1:
    """Internal original constructor, including the explicit opt-in namespace.

    No deserialized owner/factory or generic injected storage implementation is
    accepted. Command-mode production callers must use the concrete operator
    composition. This helper alone grants no independent result or ACK.
    """
    if (
        type(store) is not ScientificAblationWorkflowStoreV1
        or store.path != store._namespace_path()
    ):
        raise ValueError("Exact original workflow store/namespace required")
    owned = checked_workflow_model_v1(store.plan, IrisScientificAblationWorkflowPlanV1)
    if make_fresh_run is not None and not callable(make_fresh_run):
        raise ValueError("Explicit original callable factory or None required")
    if type(completed_phase_sha256) is not tuple:
        raise ValueError("Exact independent immutable phase-anchor tuple required")
    if not all(
        Path(path).is_absolute()
        for path in (
            owned.payload.lifecycle_path,
            owned.payload.inference_path,
            owned.payload.inputs_root,
        )
    ):
        raise ValueError("Workflow source roots must be absolute, independent of later caller CWD")
    job = object.__new__(IrisScientificAblationWorkflowV1)
    job._state = "idle"
    job._seal = _Seal(
        job,
        store,
        store.path,
        steps.encoded(owned),
        make_fresh_run,
        completed_phase_sha256,
        bool(completed_phase_sha256),
        storage_definition=store.namespace_definition_json,
    )
    _prefix(job._seal)
    return job


def _returned_advance(
    seal: _Seal,
    expected: _Expected,
    rows_per_call: int | None,
    produced: steps._Produced | _Ready,
) -> ScientificAblationWorkflowAdvanceResultV1:
    return ScientificAblationWorkflowAdvanceResultV1(
        _progress(seal),
        expected.phase,
        expected.anchors,
        expected.kernel_chunk,
        rows_per_call,
        produced.kernel_progress,
        produced.record,
        produced.files,
    )


@overload
def _advance(
    job: IrisScientificAblationWorkflowV1,
    rows_per_call: int | None,
    *,
    expected: _Expected | None = None,
    with_result: Literal[False] = False,
) -> ScientificAblationWorkflowProgressV1: ...


@overload
def _advance(
    job: IrisScientificAblationWorkflowV1,
    rows_per_call: int | None,
    *,
    expected: _Expected,
    with_result: Literal[True],
) -> ScientificAblationWorkflowAdvanceResultV1: ...


def _advance(
    job: IrisScientificAblationWorkflowV1,
    rows_per_call: int | None,
    *,
    expected: _Expected | None = None,
    with_result: bool = False,
) -> ScientificAblationWorkflowProgressV1 | ScientificAblationWorkflowAdvanceResultV1:
    seal = _entry(job)
    command_handoff = seal.storage_definition is not None
    if command_handoff and (not with_result or seal.operator_binding is None):
        raise IrisScientificAblationWorkflowError(
            "Command namespace requires the concrete operator and explicit original result request"
        )
    if rows_per_call is not None and (
        type(rows_per_call) is not int or not 1 <= rows_per_call <= 5000
    ):
        raise ValueError("Exact complete-coverage chunk budget required")
    _check_expected(seal, expected)
    if with_result != (expected is not None and expected.producer_result_requested):
        raise IrisScientificAblationWorkflowError("Original producer-result request mode differs")
    if seal.ready is not None and with_result and seal.ready.rows_per_call != rows_per_call:
        raise IrisScientificAblationWorkflowError("Storage-only retry changed the exact row budget")
    _check_operator_source(job, seal)
    _check_chunk(seal, expected, rows_per_call)
    job._state = "busy"
    try:
        with seal.store.claim() as original_claim:
            claim = original_claim
            prefix = _prefix(seal)
            _same(job, seal)
            _check_expected(seal, expected)
            _check_operator_source(job, seal)
            _same(job, seal)
            _check_chunk(seal, expected, rows_per_call, prefix)
            phases = scientific_ablation_workflow_phases_v1(seal.store.plan)
            if len(prefix) == len(phases):
                if with_result:
                    raise IrisScientificAblationWorkflowError("An explicit due phase is required")
                return _progress(seal)
            phase = phases[len(prefix)]
            previous = prefix[-1].record.sha256 if prefix else None
            receiver = None
            if command_handoff:
                from iris.application._scientific_ablation_workflow_handoff import _Receiver

                assert expected is not None
                receiver = (
                    _Receiver._begin(seal, expected, rows_per_call, original_claim)
                    if seal.ready is None
                    else seal.ready.receiver
                )
                if (
                    type(receiver) is not _Receiver
                    or receiver._expected != expected
                    or receiver._rows != rows_per_call
                ):
                    raise IrisScientificAblationWorkflowError(
                        "Original pending handoff request changed"
                    )
                if seal.ready is not None:
                    receiver._resume_claim(seal, original_claim)
                receiver._check(seal, original_claim)
                claim = receiver._hold(original_claim)
            if seal.ready is None:
                if os.path.lexists(seal.path / f"phase-{phase.index:03d}"):
                    raise IrisScientificAblationWorkflowUncertainError(
                        "Stored ready DATA requires independently retained original phase anchors"
                    )
                if (
                    phase.kind in ("capture", "diagnostics", "workspace", "message")
                    and seal.factory is None
                ):
                    raise ValueError("Explicit original factory required BEFORE a native attempt")
                seal.store.begin(phase, previous, claim)
                chunk = _expected_chunk(expected)
                produced: steps._Produced | None = None
                if with_result:
                    produced = steps.execute_produced(
                        seal.store,
                        phase,
                        prefix,
                        seal.factory,
                        claim,
                        rows_per_call,
                        expected_chunk=chunk,
                    )
                    data = produced.data()
                elif chunk is None:
                    data = steps.execute(
                        seal.store, phase, prefix, seal.factory, claim, rows_per_call
                    )
                else:
                    data = steps.execute(
                        seal.store,
                        phase,
                        prefix,
                        seal.factory,
                        claim,
                        rows_per_call,
                        expected_chunk=chunk,
                    )
                _same(job, seal)
                if _prefix(seal) != prefix:
                    raise IrisScientificAblationWorkflowError(
                        "Original history changed during phase work"
                    )
                if data is None:
                    if with_result and receiver is None:
                        assert expected is not None and produced is not None
                        return _returned_advance(seal, expected, rows_per_call, produced)
                    if not with_result:
                        return _progress(seal)
                # Immutable bytes survive storage-only failures in THIS owner.
                ready = _Ready(
                    None if data is None else steps.encoded(data.record),
                    () if data is None else data.files,
                    expected.kernel_chunk if expected is not None else None,
                    None if produced is None else produced.kernel_progress,
                    with_result,
                    rows_per_call,
                    produced,
                    receiver,
                    None if receiver is None else receiver._token,
                )
                seal = replace(seal, ready=ready)
                job._seal = seal
            data = seal.ready.data() if seal.ready is not None else None
            if data is None and receiver is not None:
                assert (
                    seal.ready is not None
                    and seal.ready.produced is not None
                    and expected is not None
                )
                receiver._receive(seal.ready.produced, None, seal, original_claim)
                actual_return = _returned_advance(seal, expected, rows_per_call, seal.ready)
                if receiver._returned is None:
                    receiver._returned = actual_return
                elif receiver._returned != actual_return:
                    raise IrisScientificAblationWorkflowError(
                        "Original partial return changed on storage retry"
                    )
                receiver._finish(receiver._returned, seal, original_claim)
                actual_return = receiver._returned
                assert receiver._completion is not None
                seal = replace(seal, ready=None, completed_handoff=receiver._completion)
                job._seal = seal
                original_claim.reserve(0)
                return actual_return
            if (
                data is None
                or data.record.payload.phase != phase
                or data.record.payload.predecessor_sha256 != previous
            ):
                raise IrisScientificAblationWorkflowError(
                    "Original process-local ready phase differs"
                )
            steps.validate_data(seal.store, data, prefix)
            if receiver is not None:
                assert seal.ready is not None and seal.ready.produced is not None
                prepared = seal.store._prepare(data, claim)
                receiver._receive(seal.ready.produced, prepared, seal, original_claim)
                claim = receiver._hold(original_claim)
            result = seal.store.commit(data, claim)
            _same(job, seal)
            if result != data or _prefix(seal) != prefix:
                raise IrisScientificAblationWorkflowError(
                    "Committed phase or predecessor history changed"
                )
            steps.validate_data(seal.store, result, prefix)
            saved_ready = seal.ready
            assert saved_ready is not None
            next_seal = replace(seal, anchors=(*seal.anchors, result.record.sha256), ready=None)
            if with_result:
                assert expected is not None
                actual_return = _returned_advance(next_seal, expected, rows_per_call, saved_ready)
                if receiver is not None:
                    if receiver._returned is None:
                        receiver._returned = actual_return
                    elif receiver._returned != actual_return:
                        raise IrisScientificAblationWorkflowError(
                            "Original return changed on storage retry"
                        )
                    receiver._finish(receiver._returned, seal, original_claim)
                    actual_return = receiver._returned
                    assert receiver._completion is not None
                    next_seal = replace(next_seal, completed_handoff=receiver._completion)
            seal = next_seal
            job._seal = seal
            original_claim.reserve(0)
        if with_result:
            return actual_return
        return _progress(seal)
    finally:
        if getattr(job, "_seal", None) is seal and getattr(job, "_state", None) == "busy":
            job._state = "idle"
        else:
            job._state = "failed"


__all__ = [
    "IrisScientificAblationWorkflowError",
    "IrisScientificAblationWorkflowV1",
    "ScientificAblationWorkflowAdvanceResultV1",
    "ScientificAblationWorkflowProgressV1",
    "create_scientific_ablation_workflow_v1",
]
