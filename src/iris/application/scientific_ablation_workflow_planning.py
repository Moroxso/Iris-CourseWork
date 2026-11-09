"""Explicit original-source kernel planning; no rows, attempts or result adoption.

Initial capture reconstructs the ORIGINAL owner under its original outer claim
and byte budget. It is NOT model-free. Retained planning reads exact DATA only.
The returned chunk is an observation, never a native permission or durable ACK;
dispatch must still revalidate it under the original outer/inner mutation locks.
"""

import os

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application.scientific_ablation_controlled_evaluation_ledger import (
    build_scientific_ablation_controlled_evaluation_plan_v1 as build_controlled_score_plan,
)
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowV1,
    _check_operator_source,
    _entry,
    _prefix,
    _same,
)
from iris.application.scientific_ablation_workflow_io import checked_workflow_model_v1
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    IrisScientificAblationWorkflowKernelRequestV1,
    ScientificAblationWorkflowKernelRequestPayloadV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def build_scientific_ablation_workflow_kernel_request_v1(
    payload: ScientificAblationWorkflowKernelRequestPayloadV1,
) -> IrisScientificAblationWorkflowKernelRequestV1:
    """Pure strict snapshot; no filesystem or ownership construction."""
    owned = checked_workflow_model_v1(payload, ScientificAblationWorkflowKernelRequestPayloadV1)
    result = IrisScientificAblationWorkflowKernelRequestV1(
        sha256=domain_sha256(
            "iris/" + owned.kind, canonical_json_bytes(owned.model_dump(mode="json"))
        ),
        payload=owned,
    )
    if len(steps.encoded(result)) > 512 * 1024:
        raise ValueError("Bounded planning request metadata required")
    return result


def plan_scientific_ablation_workflow_kernel_chunk_v1(
    job: IrisScientificAblationWorkflowV1, request: IrisScientificAblationWorkflowKernelRequestV1
) -> ScientificAblationKernelChunkExpectationV1:
    request = checked_workflow_model_v1(request, IrisScientificAblationWorkflowKernelRequestV1)
    seal = _entry(job)
    wanted = request.payload
    phase = seal.store.phase(wanted.phase)
    if (
        seal.operator_binding is None
        or seal.storage_definition is None
        or steps.encoded(wanted.namespace) != seal.storage_definition
        or wanted.completed_phase_sha256 != seal.anchors
        or phase.index != len(seal.anchors)
        or seal.ready is not None
    ):
        raise ValueError(
            "Original concrete due command workflow with no pending native result required"
        )
    prefix = _prefix(seal)
    kernel_root = steps._kernel(seal.store, phase)
    phase_path = seal.path / f"phase-{phase.index:03d}"
    attempt_path = seal.path / f"phase-{phase.index:03d}.attempt.json"

    def initial_absent() -> None:
        if (
            os.path.lexists(kernel_root)
            or os.path.lexists(attempt_path)
            or os.path.lexists(phase_path)
        ):
            raise ValueError("Initial planning cannot adopt an existing kernel, attempt or phase")

    def retained() -> tuple[str, str, int, int, str | None, int]:
        expected = wanted.expected_kernel_plan_sha256
        assert expected is not None
        value = (
            kernels._capture(seal.store, phase, expected)
            if phase.kind == "capture"
            else kernels._score(seal.store, phase, prefix, expected)
        )
        if (
            type(value) is not tuple
            or len(value) != 6
            or type(value[2]) is not int
            or type(value[3]) is not int
            or type(value[5]) is not int
            or value[0] != expected
            or value[1] != wanted.expected_acknowledgement_head_sha256
            or value[2] != wanted.expected_acknowledged_rows
            or value[5] != 0
        ):
            raise ValueError(
                "Retained planning requires exact independent DATA and no unknown stages"
            )
        return value

    if os.path.lexists(phase_path):
        raise ValueError("Unacknowledged phase DATA cannot be adopted by planning")
    if wanted.route == "initial":
        initial_absent()
    _check_operator_source(job, seal)
    job._state = "busy"
    try:
        with seal.store.claim() as claim:
            _same(job, seal)
            if _prefix(seal) != prefix:
                raise ValueError("Independent phase prefix changed before planning")
            _check_operator_source(job, seal)
            if wanted.route == "initial":
                initial_absent()
                if phase.kind == "capture":
                    # SAME original pre-factory whole budget, no reserve increase.
                    steps.budget(seal.store, kernel_root, claim, 4 * 1024**2)
                    owner = steps._owner(seal.store.plan, phase, seal.factory)
                    if (
                        phase.stream_index is None
                        and type(owner) is IrisScientificAblationHeldoutV1
                    ):
                        plan = capture.build_scientific_ablation_capture_plan_v1(
                            owner, expected_input_sha256=owner.manifest.sha256
                        )
                        plan_sha, total, head = (
                            plan.sha256,
                            len(plan.payload.rows),
                            capture._genesis(plan.sha256),
                        )
                    elif (
                        phase.stream_index is not None
                        and type(owner) is IrisScientificAblationControlledHeldoutV1
                    ):
                        controlled = (
                            controlled_capture.build_scientific_ablation_controlled_capture_plan_v1(
                                owner, expected_input_sha256=owner.manifest.sha256
                            )
                        )
                        plan_sha, total, head = (
                            controlled.sha256,
                            len(controlled.payload.rows),
                            controlled_capture._genesis(controlled.sha256),
                        )
                    else:
                        raise ValueError(
                            "Exact original source-bound ordinary/controlled owner required"
                        )
                    steps._ready(owner)
                    claim.reserve(4 * 1024**2)
                else:
                    args = steps._score_args(seal.store, prefix[-1])
                    if phase.stream_index is None:
                        score_plan = score.build_scientific_ablation_evaluation_plan_v2(**args)
                        plan_sha, total, head = (
                            score_plan.sha256,
                            len(score_plan.payload.rows),
                            score._genesis(score_plan.sha256),
                        )
                    else:
                        controlled_plan = build_controlled_score_plan(**args)
                        plan_sha, total, head = (
                            controlled_plan.sha256,
                            len(controlled_plan.payload.rows),
                            controlled_score._genesis(controlled_plan.sha256),
                        )
                rows = 0
                initial_absent()
            else:
                observed = retained()
                plan_sha, head, rows, total, _, _ = observed
                if retained() != observed:
                    raise ValueError("Retained kernel DATA changed during planning")
            _check_operator_source(job, seal)
            _same(job, seal)
            if _prefix(seal) != prefix or os.path.lexists(phase_path):
                raise ValueError("Independent workflow changed during planning")
            claim.check()
            return ScientificAblationKernelChunkExpectationV1(
                kernel_plan_sha256=plan_sha,
                acknowledgement_head_sha256=head,
                acknowledged_rows=rows,
                total_rows=total,
                rows_per_call=wanted.rows_per_call,
                stop_row_index=min(total, rows + wanted.rows_per_call),
            )
    finally:
        if getattr(job, "_seal", None) is seal and getattr(job, "_state", None) == "busy":
            job._state = "idle"


__all__ = [
    "build_scientific_ablation_workflow_kernel_request_v1",
    "plan_scientific_ablation_workflow_kernel_chunk_v1",
]
