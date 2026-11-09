"""Read exact OLD acknowledged command DATA, without owner/factory/advance/claim.

An independently retained operator ACK is mandatory. Missing final ACK is NOT
automatic recovery or permission to remeasure. Partial-range recovery and a
mutating command dispatcher are separate APIs; this reader restores neither.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _read_file,
    _regular,
)
from iris.application.scientific_ablation_operator_export import _checked_request
from iris.application.scientific_ablation_operator_workflow import _bind, _command_definition
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_ablation_workflow import (
    ScientificAblationWorkflowAdvanceResultV1,
    ScientificAblationWorkflowProgressV1,
)
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
    checked_workflow_model_v1,
)
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    IrisScientificAblationWorkflowOperatorAckV1,
    IrisScientificAblationWorkflowProducerAckV1,
    IrisScientificAblationWorkflowReceivedV1,
    IrisScientificAblationWorkflowReturnedV1,
    ScientificAblationWorkflowCommandIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)

_MAX_METADATA = 512 * 1024
_FILES = {"intent.json", "received.json", "producer-ack.json", "returned.json", "operator-ack.json"}


class IrisScientificAblationWorkflowReplayError(ApplicationError):
    """Wrong/missing original DATA/anchor; never silently execute a new command."""


def build_scientific_ablation_workflow_command_intent_v1(
    payload: ScientificAblationWorkflowCommandIntentPayloadV1,
) -> IrisScientificAblationWorkflowCommandIntentV1:
    """Snapshot pure immutable request DATA. No IO, model or resume authority."""
    owned = checked_workflow_model_v1(payload, ScientificAblationWorkflowCommandIntentPayloadV1)
    result = IrisScientificAblationWorkflowCommandIntentV1(
        sha256=domain_sha256(
            "iris/" + owned.kind, canonical_json_bytes(owned.model_dump(mode="json"))
        ),
        payload=owned,
    )
    if len(steps.encoded(result)) > _MAX_METADATA:
        raise ValueError("Original closed command metadata bound exceeded")
    return result


def _metadata[T: BaseModel](path: Path, model: type[T]) -> T:
    raw = _read_file(path, _MAX_METADATA)
    value = model.model_validate_json(raw)
    if steps.encoded(value) != raw:
        raise ValueError("Exact canonical complete original metadata required")
    return value


def _envelope(path: Path, domain: str, maximum: int) -> dict[str, Any]:
    raw = _read_file(path, maximum)
    value = json.loads(raw)
    if (
        type(value) is not dict
        or set(value) != {"sha256", "payload"}
        or type(value["payload"]) is not dict
        or canonical_json_file_bytes(value) != raw
        or domain_sha256(domain, canonical_json_bytes(value["payload"])) != value["sha256"]
    ):
        raise ValueError("Exact original acknowledged kernel DATA envelope required")
    _digest(value["sha256"])
    return value


def _kernel_snapshot(
    store: ScientificAblationWorkflowStoreV1,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    received: IrisScientificAblationWorkflowReceivedV1,
) -> None:
    request, saved = command.payload, received.payload.original_kernel_progress
    if request.phase.kind not in ("capture", "score"):
        if saved is not None:
            raise ValueError("Native/publication result cannot invent kernel progress")
        return
    chunk = request.expected_chunk
    if (
        saved is None
        or chunk is None
        or (
            saved.workflow_plan_sha256 != store.plan.sha256
            or saved.phase != request.phase
            or saved.completed_phase_sha256 != request.completed_phase_sha256
            or saved.phase_is_acknowledged
            or saved.kernel_plan_sha256 != chunk.kernel_plan_sha256
            or saved.acknowledged_rows != chunk.stop_row_index
            or saved.total_rows != chunk.total_rows
        )
    ):
        raise ValueError("Saved ORIGINAL kernel return differs from the requested BEFORE prefix")
    if (saved.kernel_complete_sha256 is None) != (received.payload.original_phase_record is None):
        raise ValueError("Original complete kernel return must retain its complete phase DATA")
    ordinary = request.phase.stream_index is None
    if request.phase.kind == "capture":
        namespace = (
            "scientific-ablation-capture-v1"
            if ordinary
            else "scientific-ablation-controlled-capture-v1"
        )
        ack_domain = (
            "iris/scientific-ablation-capture-row-ack/v1"
            if ordinary
            else "iris/scientific-ablation-controlled-capture-row-ack/v1"
        )
        row_domain = (
            "iris/scientific-ablation-heldout-row/v1"
            if ordinary
            else "iris/scientific-ablation-controlled-heldout-row/v1"
        )
        complete_domain = (
            "iris/scientific-ablation-capture-complete/v1"
            if ordinary
            else "iris/scientific-ablation-controlled-capture-complete/v1"
        )
        prefix, maximum = "row", 8 * 1024**2
    else:
        namespace = (
            "scientific-ablation-evaluation-ledger-v2"
            if ordinary
            else "scientific-ablation-controlled-evaluation-ledger-v1"
        )
        ack_domain = (
            "iris/scientific-ablation-evaluation-ack/v2"
            if ordinary
            else "iris/scientific-ablation-controlled-evaluation-ack/v1"
        )
        row_domain = (
            "iris/scientific-ablation-evaluation-row/v2"
            if ordinary
            else "iris/scientific-ablation-controlled-evaluation-row/v1"
        )
        complete_domain = (
            "iris/scientific-ablation-evaluation-complete/v2"
            if ordinary
            else "iris/scientific-ablation-controlled-evaluation-complete/v1"
        )
        prefix, maximum = "score", 65536
    root = steps._kernel(store, request.phase) / namespace / chunk.kernel_plan_sha256
    _chain(root)
    head = domain_sha256(ack_domain + "/genesis", chunk.kernel_plan_sha256.encode())
    if chunk.acknowledged_rows == 0 and chunk.acknowledgement_head_sha256 != head:
        raise ValueError("Original command's kernel genesis differs")
    # Read ONLY the original returned prefix. Later rows/current moving heads
    # must neither replace this result nor make an acknowledged OLD result stale.
    for index in range(saved.acknowledged_rows):
        directory = root / f"{prefix}-{index:05d}"
        ack = _envelope(directory / "ack.json", ack_domain, _MAX_METADATA)
        payload = ack["payload"]
        raw = _read_file(directory / "row.json", maximum)
        row = _envelope(directory / "row.json", row_domain, maximum)
        if (
            payload["plan_sha256"] != chunk.kernel_plan_sha256
            or type(payload["row_index"]) is not int
            or payload["row_index"] != index
            or payload["previous_ack_sha256"] != head
            or payload["row_sha256"] != row["sha256"]
            or payload["row_file_sha256"] != bytes_sha256(raw)
            or type(payload["row_file_bytes"]) is not int
            or payload["row_file_bytes"] != len(raw)
        ):
            raise ValueError("Original returned row-ACK prefix differs")
        head = ack["sha256"]
        if index + 1 == chunk.acknowledged_rows and head != chunk.acknowledgement_head_sha256:
            raise ValueError("Original requested starting head differs")
    if head != saved.acknowledgement_head_sha256:
        raise ValueError("Saved original ending head differs")
    if saved.kernel_complete_sha256 is not None:
        complete = _envelope(root / "complete/complete.json", complete_domain, _MAX_METADATA)
        if complete["sha256"] != saved.kernel_complete_sha256 or (
            complete["payload"]["plan_sha256"] != chunk.kernel_plan_sha256
            or complete["payload"]["acknowledgement_head_sha256"] != head
            or type(complete["payload"]["example_count"]) is not int
            or complete["payload"]["example_count"] != chunk.total_rows
        ):
            raise ValueError("Original returned completion differs")


def read_scientific_ablation_operator_workflow_command_result_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
    root: Path,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    expected_operator_ack_sha256: str,
) -> ScientificAblationWorkflowAdvanceResultV1:
    """Return exact saved OLD DATA, even after later progress, without writes.

    The caller retains the operator ACK independently. No current workflow
    progress inspection, original job, claim, factory, graph, scorer or candidate
    is constructed/invoked. Missing ACK never means advance/recover/remeasure.
    This is NOT the mutating command dispatcher or unacknowledged cold recovery.
    """
    try:
        command = checked_workflow_model_v1(command, IrisScientificAblationWorkflowCommandIntentV1)
        plan = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
        request = _checked_request(request)
        _digest(expected_operator_ack_sha256)
        if not all(
            type(p) is type(Path()) and p.is_absolute() for p in (path, export_root, base, root)
        ):
            raise ValueError("Explicit absolute original replay roots required")
        phases = scientific_ablation_workflow_phases_v1(plan)
        index = command.payload.phase.index
        if index >= len(phases) or command.payload.phase != phases[index]:
            raise ValueError("Requested phase differs from the exact source-bound plan")
        owned, source = _bind(path, request, export_root, plan, base)
        definition = _command_definition(owned, request, source)
        if command.payload.namespace != definition:
            raise ValueError("Saved command is bound to another operator/export/plan")
        store = ScientificAblationWorkflowStoreV1._for_commands(root, owned, definition)
        _chain(store.path)
        store._layout()
        if _read_file(store.path / "definition.json", _MAX_METADATA) != steps.encoded(definition):
            raise ValueError("Original physical command namespace definition differs or is missing")
        if _read_file(store.path / "plan.json", 4 * 1024**2) != steps.encoded(owned):
            raise ValueError("Original physical workflow plan differs")
        folder = store.path / "handoffs" / command.sha256[:2] / command.sha256
        _chain(folder)
        identity = _identity(_regular(folder, directory=True))
        if {p.name for p in folder.iterdir()} != _FILES:
            raise ValueError("Complete exact original operator-acknowledged command required")
        intent = _metadata(folder / "intent.json", IrisScientificAblationWorkflowCommandIntentV1)
        received = _metadata(folder / "received.json", IrisScientificAblationWorkflowReceivedV1)
        producer = _metadata(
            folder / "producer-ack.json", IrisScientificAblationWorkflowProducerAckV1
        )
        returned = _metadata(folder / "returned.json", IrisScientificAblationWorkflowReturnedV1)
        ack = _metadata(folder / "operator-ack.json", IrisScientificAblationWorkflowOperatorAckV1)
        if (
            intent != command
            or ack.sha256 != expected_operator_ack_sha256
            or (
                any(
                    p.request_sha256 != command.sha256
                    for p in (received.payload, producer.payload, returned.payload, ack.payload)
                )
                or producer.payload.received_sha256 != received.sha256
                or returned.payload.producer_ack_sha256 != producer.sha256
                or ack.payload.producer_ack_sha256 != producer.sha256
                or ack.payload.returned_result_sha256 != returned.sha256
            )
        ):
            raise ValueError("Independently retained original producer/operator ACK chain differs")
        previous = None
        for phase, digest in zip(
            phases[:index], command.payload.completed_phase_sha256, strict=True
        ):
            data = store.read(phase, expected_sha256=digest)
            if data.record.payload.predecessor_sha256 != previous:
                raise ValueError("Original BEFORE phase prefix differs")
            previous = digest
        _kernel_snapshot(store, command, received)
        record = received.payload.original_phase_record
        prefix = command.payload.completed_phase_sha256
        files: tuple[tuple[str, bytes], ...] = ()
        if record is not None:
            if (
                record.payload.phase != command.payload.phase
                or record.payload.predecessor_sha256 != previous
            ):
                raise ValueError("Original complete phase/predecessor differs")
            data = store.read(command.payload.phase, expected_sha256=record.sha256)
            if data.record != record:
                raise ValueError("Original acknowledged physical phase record differs")
            files = data.files
            prefix = (*prefix, record.sha256)
        elif command.payload.phase.kind not in ("capture", "score"):
            raise ValueError("Native/publication original result requires complete phase DATA")
        progress = returned.payload.progress
        complete = len(prefix) == len(phases)
        if (
            progress.plan_sha256 != plan.sha256
            or progress.completed_phase_sha256 != prefix
            or (
                progress.complete != complete
                or progress.next_phase != (None if complete else phases[len(prefix)].name)
            )
        ):
            raise ValueError("Saved original progress differs; never substitute a newer head")
        result = ScientificAblationWorkflowAdvanceResultV1(
            ScientificAblationWorkflowProgressV1(
                progress.plan_sha256,
                progress.completed_phase_sha256,
                progress.next_phase,
                progress.complete,
                progress.recovered_prefix_is_data_only,
            ),
            steps.encoded(command.payload.phase),
            command.payload.completed_phase_sha256,
            None
            if command.payload.expected_chunk is None
            else steps.encoded(command.payload.expected_chunk),
            command.payload.rows_per_call,
            None
            if received.payload.original_kernel_progress is None
            else steps.encoded(received.payload.original_kernel_progress),
            None if record is None else steps.encoded(record),
            files,
        )
        if (
            _identity(_regular(folder, directory=True)) != identity
            or {p.name for p in folder.iterdir()} != _FILES
        ):
            raise ValueError("Original acknowledged result directory changed during read")
        if store._identity != _identity(_regular(store.path, directory=True)):
            raise ValueError("Original workflow root changed during replay")
        if _read_file(store.path / "plan.json", 4 * 1024**2) != steps.encoded(owned):
            raise ValueError("Original workflow plan changed during replay")
        if _read_file(store.path / "definition.json", _MAX_METADATA) != steps.encoded(definition):
            raise ValueError("Original command namespace definition changed during replay")
        after, source_after = _bind(path, request, export_root, owned, base)
        if after != owned or _command_definition(after, request, source_after) != definition:
            raise ValueError("Original operator/export/source changed during replay")
        return result
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationWorkflowReplayError(
            "Original acknowledged command DATA replay failed; no execution or recovery requested"
        ) from None


__all__ = [
    "IrisScientificAblationWorkflowReplayError",
    "build_scientific_ablation_workflow_command_intent_v1",
    "read_scientific_ablation_operator_workflow_command_result_v1",
]
