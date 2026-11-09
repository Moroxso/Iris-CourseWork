"""Concrete original in-claim receiver and live completion; no cold native authority.

Native blobs are the ORIGINAL prepared phase DATA, not a second large copy.
The separate receiver/producer/operator metadata binds that DATA by exact pins.
Only the live original produced reference can enter this receiver. JSON files
never reconstruct that reference, a prepared-stage capability or native owner.
"""

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _read_file,
    _regular,
)
from iris.application.scientific_ablation_workflow_io import (
    _HANDOFF_FILES,
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    _PreparedPhase,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock

if TYPE_CHECKING:
    from pathlib import Path

    from iris.application.scientific_ablation_operator_workflow import _OperatorWorkflowBinding
    from iris.application.scientific_ablation_workflow import (
        IrisScientificAblationWorkflowV1,
        ScientificAblationWorkflowAdvanceResultV1,
        _Expected,
        _Seal,
    )

_MAX_RECORD = 512 * 1024
_CAPACITY = len(_HANDOFF_FILES) * _MAX_RECORD


def _value(raw: bytes | None) -> object:
    return None if raw is None else json.loads(raw)


def _record(kind: str, payload: dict[str, object]) -> bytes:
    value = {"kind": kind, **payload}
    raw = canonical_json_file_bytes(
        {"sha256": domain_sha256("iris/" + kind, canonical_json_bytes(value)), "payload": value}
    )
    if len(raw) > _MAX_RECORD:
        raise ValueError("Closed handoff metadata record bound exceeded")
    return raw


@dataclass(frozen=True, slots=True, eq=False)
class _Completion:
    """Live receiver-minted ACK/result reference, not constructible JSON ownership.

    Minted only after the real durable operator ACK, inside the SAME original
    claim. A duplicate dataclass or restored bytes is not receiver._completion.
    No additional file or second native blob copy is made.
    """

    receiver: "_Receiver"
    token: object
    intent_json: bytes
    returned_json: bytes
    operator_ack_json: bytes
    result: "ScientificAblationWorkflowAdvanceResultV1"


class _Receiver:
    """Private live original receiver, not a constructible metadata capability."""

    def __init__(self) -> None:
        raise TypeError("Only the original in-claim workflow may create a receiver")

    @classmethod
    def _begin(
        cls, seal: "_Seal", expected: "_Expected", rows: int | None, claim: _Claim
    ) -> "_Receiver":
        if cls is not _Receiver or seal.operator_binding is None or seal.storage_definition is None:
            raise ValueError("Concrete original operator/command namespace binding required")
        receiver = object.__new__(cls)
        receiver._job = seal.job
        receiver._claim_lock = claim.lock
        receiver._claim_identity = claim.identity
        receiver._store = seal.store
        receiver._factory = seal.factory
        receiver._binding = seal.operator_binding
        receiver._namespace = seal.storage_definition
        receiver._expected = expected
        receiver._rows = rows
        receiver._token = object()
        receiver._produced = None
        receiver._returned = None
        receiver._completion = None
        receiver._received = None
        receiver._producer_ack = None
        receiver._remaining = _CAPACITY
        receiver._written = {}
        receiver._publishing = {}
        receiver._directories = None
        receiver._intent = _record(
            "scientific-ablation-workflow-command-intent-v1",
            dict(
                namespace=_value(seal.storage_definition),
                phase=_value(expected.phase),
                completed_phase_sha256=expected.anchors,
                expected_chunk=_value(expected.kernel_chunk),
                rows_per_call=rows,
                authority="request_data_only",
            ),
        )
        receiver._sha = json.loads(receiver._intent)["sha256"]
        receiver._path = seal.path / "handoffs" / receiver._sha[:2] / receiver._sha
        receiver._check(seal, claim)
        # Existing intent is DATA only. An independent old result is NOT a live
        # produced reference. Its cold DATA recovery/replay needs a separate API.
        if any(
            os.path.lexists(receiver._path / n)
            for n in ("received.json", "producer-ack.json", "returned.json", "operator-ack.json")
        ):
            raise IrisScientificAblationWorkflowUncertainError(
                "Retained handoff DATA requires explicit independent DATA recovery, never remeasure"
            )
        receiver._hold(claim)
        receiver._write("intent.json", receiver._intent, seal, claim)
        return receiver

    def _check(self, seal: "_Seal", claim: _Claim) -> None:
        from iris.application.scientific_ablation_workflow import _entry

        if (
            type(self) is not _Receiver
            or _entry(self._job, busy=True) is not seal
            or seal.store is not self._store
            or seal.factory is not self._factory
            or seal.operator_binding is not self._binding
            or seal.storage_definition != self._namespace
            or claim.store is not self._store
            or claim.lock is not self._claim_lock
            or claim.identity != self._claim_identity
        ):
            raise ValueError("Original receiver/job/seal/factory/namespace/claim changed")
        self._store._check_claim(claim)

    def _resume_claim(self, seal: "_Seal", claim: _Claim) -> None:
        """Only this live process retains the original object/token on retry."""
        ready = seal.ready
        if (
            ready is None
            or ready.receiver is not self
            or ready.handoff_token is not self._token
            or ready.produced is None
            or (self._produced is not None and ready.produced is not self._produced)
            or claim.store is not self._store
            or claim.identity != self._claim_identity
        ):
            raise ValueError("Original live result capability required for storage-only retry")
        self._store._check_claim(claim)
        self._claim_lock = claim.lock
        self._check(seal, claim)

    def _hold(self, claim: _Claim) -> _Claim:
        return claim.with_reserved_bytes(self._remaining)

    def _write(self, name: str, raw: bytes, seal: "_Seal", claim: _Claim) -> None:
        self._check(seal, claim)
        if name not in _HANDOFF_FILES or type(raw) is not bytes or not 1 <= len(raw) <= _MAX_RECORD:
            raise ValueError("Closed exact handoff metadata required")
        self._path.mkdir(parents=True, exist_ok=True)
        _chain(self._path)
        directories = tuple(
            _identity(_regular(p, directory=True))
            for p in (self._path.parent.parent, self._path.parent, self._path)
        )
        if self._directories is not None and self._directories != directories:
            raise ValueError("Original receiver directory identity changed")
        self._directories = directories
        target = self._path / name
        if name in self._written:
            if self._written[name] != raw or _read_file(target, _MAX_RECORD) != raw:
                raise ValueError("Original receiver metadata changed")
            return
        if name in self._publishing:
            pending_raw, identity, stage = self._publishing[name]
            if pending_raw != raw:
                raise ValueError("Original pending metadata bytes changed")
        elif self._remaining < len(raw):
            raise ValueError("Original receiver metadata reservation exhausted")
        # Convert precisely these bytes from the hold into real occupancy under
        # the SAME lock. Unknown/failed stages remain charged, never waived.
        elif os.path.lexists(target):
            if name != "intent.json" or _read_file(target, _MAX_RECORD) != raw:
                raise ValueError("Unowned existing receiver metadata cannot be adopted")
            self._remaining -= len(raw)
            self._written[name] = raw
            self._hold(claim).reserve(0)
            return
        else:
            view = replace(claim, reserved_bytes=claim.reserved_bytes + self._remaining - len(raw))
            view.reserve(len(raw))
            stage = self._store.path / f".iris-handoff-{uuid4().hex}.tmp"
            identity = self._store._write_new(stage, raw, view)
            self._publishing[name] = (raw, identity, stage)
            self._remaining -= len(raw)
        self._check(seal, claim)
        if os.path.lexists(stage):
            if os.path.lexists(target) or _identity(_regular(stage, directory=False)) != identity:
                raise ValueError("Receiver metadata destination/stage changed")
            os.rename(stage, target)  # Cooperating writers share this original OS claim.
        if _identity(_regular(target, directory=False)) != identity:
            raise ValueError("Published receiver metadata identity differs")
        if _read_file(target, _MAX_RECORD) != raw:
            raise ValueError("Published receiver metadata bytes differ")
        self._written[name] = raw
        del self._publishing[name]
        self._hold(claim).reserve(0)

    def _receive(
        self,
        produced: steps._Produced,
        prepared: _PreparedPhase | None,
        seal: "_Seal",
        claim: _Claim,
    ) -> None:
        self._check(seal, claim)
        ready = seal.ready
        if (
            type(produced) is not steps._Produced
            or ready is None
            or ready.produced is not produced
            or ready.receiver is not self
            or ready.handoff_token is not self._token
            or (self._produced is not None and self._produced is not produced)
        ):
            raise ValueError("Exact actual original produced reference/capability required")
        self._produced = produced
        data = produced.data()
        if (data is None) != (prepared is None):
            raise ValueError("Partial original return cannot invent prepared phase DATA")
        location = None
        if prepared is not None:
            location = self._store._check_prepared(prepared, claim).name
            if prepared.data != data:
                raise ValueError("Original prepared DATA differs from the actual producer")
        if self._received is None:
            self._received = _record(
                "scientific-ablation-workflow-original-result-received-v1",
                dict(
                    request_sha256=self._sha,
                    original_phase_record=_value(produced.record),
                    original_kernel_progress=_value(produced.kernel_progress),
                    original_files_location=location,
                    authority="received_data_not_native_ownership",
                ),
            )
        self._write("received.json", self._received, seal, claim)
        if self._producer_ack is None:
            self._producer_ack = _record(
                "scientific-ablation-workflow-original-producer-ack-v1",
                dict(
                    request_sha256=self._sha,
                    received_sha256=json.loads(self._received)["sha256"],
                    authority="original_reference_received_not_native_restoration",
                ),
            )
        self._write("producer-ack.json", self._producer_ack, seal, claim)

    def _finish(
        self, result: "ScientificAblationWorkflowAdvanceResultV1", seal: "_Seal", claim: _Claim
    ) -> None:
        self._check(seal, claim)
        produced = self._produced
        if (
            self._returned is not result
            or produced is None
            or self._producer_ack is None
            or (
                result.requested_phase_json != self._expected.phase
                or result.completed_phase_sha256_before != self._expected.anchors
                or result.expected_chunk_json != self._expected.kernel_chunk
                or result.rows_per_call != self._rows
                or result.phase_record_json != produced.record
                or result.kernel_progress_json != produced.kernel_progress
                or result.phase_files != produced.files
            )
        ):
            raise ValueError(
                "Original returned result differs from the independently received DATA"
            )
        data = produced.data()
        anchors = self._expected.anchors
        if data is not None:
            physical = self._store.read(
                data.record.payload.phase, expected_sha256=data.record.sha256
            )
            if physical != data:
                raise ValueError("Actual original phase publication differs before operator ACK")
            anchors = (*anchors, data.record.sha256)
        if result.progress.completed_phase_sha256 != anchors:
            raise ValueError("An operator ACK cannot predict a future phase publication")
        returned = _record(
            "scientific-ablation-workflow-original-returned-result-v1",
            dict(
                request_sha256=self._sha,
                producer_ack_sha256=json.loads(self._producer_ack)["sha256"],
                progress=asdict(result.progress),
                authority="original_return_data_only",
            ),
        )
        self._write("returned.json", returned, seal, claim)
        ack = _record(
            "scientific-ablation-workflow-original-operator-ack-v1",
            dict(
                request_sha256=self._sha,
                returned_result_sha256=json.loads(returned)["sha256"],
                producer_ack_sha256=json.loads(self._producer_ack)["sha256"],
                authority="saved_original_data_not_execution_permission",
            ),
        )
        self._write("operator-ack.json", ack, seal, claim)
        if self._completion is None:
            self._completion = _Completion(self, self._token, self._intent, returned, ack, result)
        elif (
            self._completion.receiver is not self
            or self._completion.token is not self._token
            or self._completion.intent_json != self._intent
            or self._completion.returned_json != returned
            or self._completion.operator_ack_json != ack
            or self._completion.result is not result
        ):
            raise ValueError("Original live completed handoff changed during storage-only retry")

    _job: "IrisScientificAblationWorkflowV1"
    _store: ScientificAblationWorkflowStoreV1
    _factory: steps.Factory | None
    _binding: "_OperatorWorkflowBinding"
    _namespace: bytes
    _expected: "_Expected"
    _rows: int | None
    _token: object
    _produced: steps._Produced | None
    _returned: "ScientificAblationWorkflowAdvanceResultV1 | None"
    _completion: _Completion | None
    _received: bytes | None
    _producer_ack: bytes | None
    _remaining: int
    _written: dict[str, bytes]
    _publishing: dict[str, tuple[bytes, tuple[int, int], "Path"]]
    _directories: tuple[tuple[int, int], ...] | None
    _intent: bytes
    _sha: str
    _path: "Path"
    _claim_lock: ProcessFileLock
    _claim_identity: tuple[int, int]
