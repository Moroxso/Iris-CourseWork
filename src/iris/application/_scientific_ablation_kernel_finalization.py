"""Original-lock all-ACK aggregate publication; DATA only, never row/native replay.

Only fresh stages are created. Unknown retained stages remain charged and are
never adopted/reaped. The public recovery coordinator retains the original job,
namespace, immutable command and outer workflow claim; this module mints no owner.
"""

from pathlib import Path

from iris.application import _scientific_ablation_kernel_recovery as interval
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation as controlled_evaluation
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_evaluation_v2 as evaluation
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _regular,
    _signature,
)
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    checked_workflow_model_v1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
)
from iris.utils.hashing import domain_sha256
from iris.utils.process_file_lock import ProcessFileLock

Progress = steps.CaptureProgress | steps.ScoreProgress
_PHASE_RESERVE = 4 * 1024**2  # Same original compact-phase allowance; never an increased quota.


def _aggregate_files(result: Progress) -> dict[str, bytes]:
    """Derive exact normal-path bytes from verified complete ACK coverage only."""
    if (
        not result.rows
        or len(result.rows) != len(result.plan.payload.rows)
        or len(result.rows) != len(result.acknowledgements)
    ):
        raise ValueError("Aggregate requires every original row and ACK")
    head = result.acknowledgements[-1].sha256
    if isinstance(result, capture.IrisScientificAblationCaptureProgressV1):
        return {"complete.json": capture._file(capture._complete(result.plan, result.rows, head))}
    if isinstance(result, controlled_capture.IrisScientificAblationControlledCaptureProgressV1):
        return {
            "complete.json": controlled_capture._file(
                controlled_capture._complete(result.plan, result.rows, head)
            )
        }
    if isinstance(result, score.IrisScientificAblationEvaluationProgressV2):
        p = result.plan.payload
        report = evaluation._assemble_data(
            p.generation_plan,
            p.capture,
            p.python_policy,
            tuple(row.payload.measurements for row in result.rows),
        )
        return {
            "evaluation.json": evaluation._checked(report),
            "complete.json": score._file(score._complete(result.plan, head, report)),
        }
    if isinstance(result, controlled_score.IrisScientificAblationControlledEvaluationProgressV1):
        cp = result.plan.payload
        controlled_report = controlled_evaluation._assemble_data(
            cp.generation_plan,
            cp.capture,
            cp.python_policy,
            tuple(row.payload.measurements for row in result.rows),
            maximum_report_file_bytes=cp.maximum_report_file_bytes,
        )
        return {
            "evaluation.json": controlled_evaluation._checked(controlled_report),
            "complete.json": controlled_score._file(
                controlled_score._complete(result.plan, head, controlled_report)
            ),
        }
    raise ValueError("Exact original capture/score progress required")


def _row_stamps(result: Progress) -> dict[Path, tuple[int, ...]]:
    prefix = (
        "row"
        if isinstance(
            result,
            (
                capture.IrisScientificAblationCaptureProgressV1,
                controlled_capture.IrisScientificAblationControlledCaptureProgressV1,
            ),
        )
        else "score"
    )
    paths = [(result.path / "plan.json", False)]
    for index in range(len(result.rows)):
        directory = result.path / f"{prefix}-{index:05d}"
        paths.append((directory, True))
        paths.extend((directory / name, False) for name in ("attempt.json", "row.json", "ack.json"))
    return {path: _signature(_regular(path, directory=directory)) for path, directory in paths}


def _publish(
    result: Progress,
    files: dict[str, bytes],
    namespace: Path,
    lock: ProcessFileLock,
    maximum: int,
    minimum: int,
) -> None:
    """Use actual original kernel claims/stages/accounting, never constructed inputs."""
    identity = _identity(_regular(namespace, directory=True))
    size = sum(map(len, files.values()))
    if isinstance(result, capture.IrisScientificAblationCaptureProgressV1):
        claim = capture._Claim(namespace, identity, lock)
        capture._reserve(claim, size, maximum, minimum)
        stage = capture._stage(result.path, claim)
        for name, content in files.items():
            capture._write(stage, name, content, claim)
        capture._publish(stage, result.path / "complete", files, claim)
    elif isinstance(result, controlled_capture.IrisScientificAblationControlledCaptureProgressV1):
        controlled_claim = controlled_capture._Claim(namespace, identity, lock)
        controlled_capture._reserve(controlled_claim, size, maximum, minimum)
        controlled_stage = controlled_capture._stage(result.path, controlled_claim)
        for name, content in files.items():
            controlled_capture._write(controlled_stage, name, content, controlled_claim)
        controlled_capture._publish(
            controlled_stage, result.path / "complete", files, controlled_claim
        )
    elif isinstance(result, score.IrisScientificAblationEvaluationProgressV2):
        budget = score._Budget(score._Claim(namespace, identity, lock), maximum, minimum)
        budget.reload()
        budget.ensure(size, directories=1, files=2)
        score_stage = score._stage(result.path, budget)
        for name, content in files.items():
            score._write(score_stage, name, content, budget)
        score._publish(score_stage, result.path / "complete", files, budget)
    elif isinstance(result, controlled_score.IrisScientificAblationControlledEvaluationProgressV1):
        controlled_budget = controlled_score._Budget(
            controlled_score._Claim(namespace, identity, lock),
            maximum,
            minimum,
        )
        controlled_budget.reload()
        controlled_budget.ensure(size, directories=1, files=2)
        controlled_score_stage = controlled_score._stage(result.path, controlled_budget)
        for name, content in files.items():
            controlled_score._write(controlled_score_stage, name, content, controlled_budget)
        controlled_score._publish(
            controlled_score_stage, result.path / "complete", files, controlled_budget
        )
    else:
        raise ValueError("Exact original kernel DATA required")


def _finalize_acknowledged_kernel_data(
    store: ScientificAblationWorkflowStoreV1,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    claim: _Claim,
    lock: ProcessFileLock,
    *,
    expected_acknowledgement_head_sha256: str,
) -> Progress:
    """All rows already ACKed; no factory/producer/scorer/model path exists here."""
    command = checked_workflow_model_v1(command, IrisScientificAblationWorkflowCommandIntentV1)
    _digest(expected_acknowledgement_head_sha256)
    phase, chunk = command.payload.phase, command.payload.expected_chunk
    if (
        chunk is None
        or chunk.stop_row_index != chunk.total_rows
        or claim.store is not store
        or store.phase(phase) != phase
        or len(prefix) != phase.index
        or tuple(item.record.sha256 for item in prefix) != command.payload.completed_phase_sha256
        or steps.encoded(command.payload.namespace) != store.namespace_definition_json
        or claim.namespace_definition != store.namespace_definition_json
    ):
        raise ValueError("Original all-row interval/namespace/prefix/claim required")
    namespace_name, _, ack_domain = interval._PROTOCOLS[phase.kind, phase.stream_index is not None]
    root = steps._kernel(store, phase)
    namespace = root / namespace_name
    _chain(namespace)
    if lock.path != namespace / ".iris-owner.lock":
        raise ValueError("Actual original namespace lock required")
    lock.check()
    claim.reserve(0)

    def read() -> Progress:
        return (
            kernels._capture_result(store, phase, chunk.kernel_plan_sha256)
            if phase.kind == "capture"
            else kernels._score_result(store, phase, prefix, chunk.kernel_plan_sha256)
        )

    result = read()
    head = domain_sha256(ack_domain + "/genesis", chunk.kernel_plan_sha256.encode())
    if (
        result.path != namespace / chunk.kernel_plan_sha256
        or len(result.rows) != len(result.acknowledgements)
        or len(result.rows) != chunk.total_rows
        or len(result.plan.payload.rows) != chunk.total_rows
        or result.plan.sha256 != chunk.kernel_plan_sha256
    ):
        raise ValueError("Every original canonical row/ACK/plan must exist before aggregate writes")
    acks = tuple(value.sha256 for value in result.acknowledgements)
    if acks[-1] != expected_acknowledgement_head_sha256:
        raise ValueError("Observed locked complete ACK prefix changed before finalization")
    actual_start = acks[chunk.acknowledged_rows - 1] if chunk.acknowledged_rows else head
    if actual_start != chunk.acknowledgement_head_sha256:
        raise ValueError("Original immutable starting ACK differs")
    lock.check()
    if result.complete is not None:
        claim.reserve(0)
        return result
    root_identity = _identity(_regular(result.path, directory=True))
    stamps = _row_stamps(result)
    maximum = steps.budget(store, root, claim, _PHASE_RESERVE)
    minimum = claim.minimum_free_for(_PHASE_RESERVE)
    files = _aggregate_files(result)
    claim.reserve(_PHASE_RESERVE + sum(map(len, files.values())))
    lock.check()
    if _row_stamps(result) != stamps:
        raise ValueError("Acknowledged row identities changed before aggregate publication")
    _publish(result, files, namespace, lock, maximum, minimum)
    lock.check()
    claim.reserve(_PHASE_RESERVE)
    checked = read()
    if (
        checked.complete is None
        or checked.plan != result.plan
        or checked.rows != result.rows
        or checked.acknowledgements != result.acknowledgements
        or _identity(_regular(result.path, directory=True)) != root_identity
        or _row_stamps(checked) != stamps
    ):
        raise ValueError("Published aggregate or original acknowledged rows changed")
    claim.reserve(0)
    return checked
