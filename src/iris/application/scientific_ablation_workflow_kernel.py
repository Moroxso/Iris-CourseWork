"""Inspect explicit existing capture/score prefix through an ORIGINAL workflow.

No claim, writes, graph construction, row execution, stage adoption or moving-head
advance. The supplied kernel plan/head are DATA anchors, never native ownership.
A later command must revalidate its exact row prefix under the real mutation lock;
this observation alone is NOT an idempotent command or durable producer ACK.
"""

import os
import re

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import _chain, _regular
from iris.application.scientific_ablation_kernel_chunk import check_kernel_chunk_prefix_v1
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowError,
    IrisScientificAblationWorkflowV1,
    _entry,
    _prefix,
)
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
)
from iris.application.scientific_command import _rows
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import ScientificAblationWorkflowPhaseV1

type _Observed = tuple[str, str, int, int, str | None, int]


def _check_chunk_before_native(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    chunk: ScientificAblationKernelChunkExpectationV1,
    rows_per_call: int | None,
) -> None:
    """Read-only admission, repeated under the outer original workflow claim.

    The original kernel ALSO compares its actual prefix under its own mutation
    lock, closing races with standalone kernel writers. A fresh capture plan's
    identity needs original native reconstruction; no graph-free planning claim.
    """
    phase = store.phase(phase)
    if phase.kind not in ("capture", "score") or phase.index != len(prefix):
        raise ValueError("Exact due capture/score phase required for a kernel chunk")
    expected = chunk.kernel_plan_sha256
    if phase.kind == "capture":
        module = capture if phase.stream_index is None else controlled_capture
        namespace = steps._kernel(store, phase) / module._STORE
        target = namespace / expected
        if os.path.lexists(target):
            observed = _capture(store, phase, expected)
        else:
            if os.path.lexists(namespace):
                _chain(namespace)
                _regular(namespace, directory=True)
                if any(re.fullmatch(r"[0-9a-f]{64}", entry.name) for entry in namespace.iterdir()):
                    raise ValueError(
                        "Another materialized capture plan differs from exact chunk intent"
                    )
            steps.source(store.plan)
            corpus = steps._corpus(store.plan)
            total = len(_rows(corpus["test"], corpus["inputs_root"]))
            observed = expected, module._genesis(expected), 0, total, None, 0
    else:
        args = steps._score_args(store, prefix[phase.index - 1])
        if phase.stream_index is None:
            plan = score.build_scientific_ablation_evaluation_plan_v2(**args)
            plan_sha, plan_rows = plan.sha256, len(plan.payload.rows)
            score_namespace = score._STORE
            genesis = score._genesis(plan.sha256)
        else:
            controlled_plan = (
                controlled_score.build_scientific_ablation_controlled_evaluation_plan_v1(**args)
            )
            plan_sha, plan_rows = controlled_plan.sha256, len(controlled_plan.payload.rows)
            score_namespace = controlled_score._STORE
            genesis = controlled_score._genesis(controlled_plan.sha256)
        if plan_sha != expected:
            raise ValueError("Exact score chunk plan differs from actual capture/corpus/policy")
        target = steps._kernel(store, phase) / score_namespace / expected
        observed = (
            _score(store, phase, prefix, expected)
            if os.path.lexists(target)
            else (expected, genesis, 0, plan_rows, None, 0)
        )
    plan_sha, head, rows, total, _, _ = observed
    check_kernel_chunk_prefix_v1(
        chunk,
        plan_sha256=plan_sha,
        acknowledgement_head_sha256=head,
        acknowledged_rows=rows,
        total_rows=total,
        rows_per_call=rows_per_call,
    )


def read_scientific_ablation_workflow_kernel_progress_v1(
    job: IrisScientificAblationWorkflowV1,
    *,
    expected_phase: ScientificAblationWorkflowPhaseV1,
    expected_kernel_plan_sha256: str,
    expected_acknowledgement_head_sha256: str | None = None,
) -> IrisScientificAblationWorkflowKernelProgressV1:
    seal = _entry(job)
    phase = seal.store.phase(expected_phase)
    _digest(expected_kernel_plan_sha256)
    if expected_acknowledgement_head_sha256 is not None:
        _digest(expected_acknowledgement_head_sha256)
    if phase.kind not in ("capture", "score") or phase.index > len(seal.anchors):
        raise ValueError("Only an explicit due or independently acknowledged capture/score phase")
    prefix = _prefix(seal)
    if phase.index == len(prefix) and os.path.lexists(seal.path / f"phase-{phase.index:03d}"):
        raise IrisScientificAblationWorkflowUncertainError(
            "Unacknowledged ready phase DATA is not adopted by kernel inspection"
        )
    if phase.kind == "capture":
        observed = _capture(seal.store, phase, expected_kernel_plan_sha256)
    else:
        observed = _score(seal.store, phase, prefix, expected_kernel_plan_sha256)
    kernel_plan, head, rows, total, complete, stages = observed
    if kernel_plan != expected_kernel_plan_sha256:
        raise ValueError("Original reader returned another explicit kernel plan")
    if (
        expected_acknowledgement_head_sha256 is not None
        and head != expected_acknowledgement_head_sha256
    ):
        raise ValueError("Independently retained kernel acknowledgement head differs")
    if _entry(job) is not seal or _prefix(seal) != prefix:
        raise IrisScientificAblationWorkflowError(
            "Original workflow changed during DATA inspection"
        )
    return IrisScientificAblationWorkflowKernelProgressV1(
        workflow_plan_sha256=seal.store.plan.sha256,
        phase=phase,
        completed_phase_sha256=seal.anchors,
        phase_is_acknowledged=phase.index < len(seal.anchors),
        kernel_plan_sha256=kernel_plan,
        acknowledgement_head_sha256=head,
        acknowledged_rows=rows,
        total_rows=total,
        kernel_complete_sha256=complete,
        uncommitted_stage_count=stages,
    )


def _capture_result(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    expected: str,
) -> steps.CaptureProgress:
    root = steps._kernel(store, phase)
    if phase.stream_index is None:
        result: steps.CaptureProgress = capture.read_scientific_ablation_capture_v1(
            root / capture._STORE / expected,
            expected_plan_sha256=expected,
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    else:
        result = controlled_capture.read_scientific_ablation_controlled_capture_v1(
            root / controlled_capture._STORE / expected,
            expected_plan_sha256=expected,
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    p, actual = store.plan.payload, result.plan.payload
    if (
        result.plan.sha256 != expected
        or actual.source != steps.source(store.plan).manifest
        or actual.input.payload.decoding != p.decoding
        or actual.input.payload.test_parquet_sha256 != p.test.sha256
    ):
        raise ValueError("Actual capture kernel source/input differs from workflow intent")
    if phase.stream_index is not None:
        from iris.ablations.controls import CognitiveAblationControlPlanV1

        if type(result) is not controlled_capture.IrisScientificAblationControlledCaptureProgressV1:
            raise ValueError("Exact original controlled progress required")
        control = CognitiveAblationControlPlanV1.model_validate_json(
            result.plan.payload.input.payload.generation_control_plan_json
        )
        if control.control != p.controls[phase.stream_index]:
            raise ValueError("Actual controlled kernel intent differs")
    return result


def _view(result: steps.CaptureProgress | steps.ScoreProgress, genesis: str) -> _Observed:
    return (
        result.plan.sha256,
        result.acknowledgements[-1].sha256 if result.acknowledgements else genesis,
        len(result.rows),
        len(result.plan.payload.rows),
        None if result.complete is None else result.complete.sha256,
        result.uncommitted_stage_count,
    )


def _capture(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    expected: str,
) -> _Observed:
    module = capture if phase.stream_index is None else controlled_capture
    return _view(_capture_result(store, phase, expected), module._genesis(expected))


def _score_result(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    expected: str,
) -> steps.ScoreProgress:
    if phase.index == 0:
        raise ValueError("Score requires its independently acknowledged capture predecessor")
    args = steps._score_args(store, prefix[phase.index - 1])
    root = steps._kernel(store, phase)
    if phase.stream_index is None:
        plan = score.build_scientific_ablation_evaluation_plan_v2(**args)
        if plan.sha256 != expected:
            raise ValueError("Expected score plan differs from actual capture/corpus/policy")
        result: steps.ScoreProgress = score.read_scientific_ablation_evaluation_progress_v2(
            root / score._STORE / expected,
            expected_evaluation_plan_sha256=expected,
            expected_plan_sha256=args["expected_plan_sha256"],
            expected_capture_sha256=args["expected_capture_sha256"],
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    else:
        controlled_plan = controlled_score.build_scientific_ablation_controlled_evaluation_plan_v1(
            **args
        )
        if controlled_plan.sha256 != expected:
            raise ValueError(
                "Expected controlled score plan differs from actual capture/corpus/policy"
            )
        result = controlled_score.read_scientific_ablation_controlled_evaluation_progress_v1(
            root / controlled_score._STORE / expected,
            expected_evaluation_plan_sha256=expected,
            expected_plan_sha256=args["expected_plan_sha256"],
            expected_capture_sha256=args["expected_capture_sha256"],
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    if result.plan.sha256 != expected:
        raise ValueError("Exact original evaluation plan required")
    return result


def _score(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    expected: str,
) -> _Observed:
    module = score if phase.stream_index is None else controlled_score
    return _view(_score_result(store, phase, prefix, expected), module._genesis(expected))


__all__ = ["read_scientific_ablation_workflow_kernel_progress_v1"]
