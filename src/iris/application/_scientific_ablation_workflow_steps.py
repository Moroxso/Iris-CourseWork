"""Original native/kernel seams for the compound owner, not caller callbacks.

All persisted values remain DATA. Only the explicit original factory constructs
fresh owners; no writer job or native completion authority is restored from JSON.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_collapse as collapse
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_diagnostics as diagnostics
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application._scientific_ablation_kernel_recovery import _KernelIntervalPermission
from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
    ScientificAblationCausalInputV1,
)
from iris.application.scientific_ablation_controlled_generation_context import (
    build_scientific_ablation_controlled_generation_context_v1,
)
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
    build_scientific_ablation_controlled_heldout_v1,
)
from iris.application.scientific_ablation_controlled_heldout import _entry as controlled_entry
from iris.application.scientific_ablation_generation_context import (
    build_scientific_ablation_generation_context_v1,
)
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
    read_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_heldout import (
    IrisScientificAblationHeldoutV1,
    build_scientific_ablation_heldout_v1,
)
from iris.application.scientific_ablation_heldout import _entry as heldout_entry
from iris.application.scientific_ablation_kernel_chunk import checked_kernel_chunk_v1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    _tree_size,
    checked_workflow_model_v1,
)
from iris.application.scientific_ablation_writer_io import (
    read_scientific_ablation_writer_bundle_v1,
    write_scientific_ablation_writer_bundle_v1,
)
from iris.application.scientific_ablation_writer_jobs import (
    create_scientific_ablation_writer_job_v1,
)
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_command import ScientificParquetInputV1, _rows
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_collapse_v1 import IrisScientificAblationCollapseV1
from iris.artifacts.scientific_ablation_diagnostics_v1 import IrisScientificAblationDiagnosticsV1
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPhaseDataV1,
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowCaptureDataV1,
    ScientificAblationWorkflowPhaseV1,
    ScientificAblationWorkflowScoreDataV1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    WriterReport,
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    IrisScientificAblationWriterSourceV1,
    WriterSourceProtocol,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
    scientific_ablation_writer_source_preflight_v1,
)
from iris.generation.sampling import TemperatureTopPSamplingConfig
from iris.utils.hashing import canonical_json_file_bytes

type Factory = Callable[[], IrisScientificAblationRunV1]
type CaptureProgress = (
    capture.IrisScientificAblationCaptureProgressV1
    | controlled_capture.IrisScientificAblationControlledCaptureProgressV1
)
type ScoreProgress = (
    score.IrisScientificAblationEvaluationProgressV2
    | controlled_score.IrisScientificAblationControlledEvaluationProgressV1
)


@dataclass(frozen=True, slots=True)
class _Produced:
    """Exact returned producer DATA, not a durable receipt or native authority."""

    record: bytes | None
    files: tuple[tuple[str, bytes], ...]
    kernel_progress: bytes | None

    def data(self) -> ScientificAblationWorkflowDataV1 | None:
        if self.record is None:
            if self.files:
                raise ValueError("A partial producer cannot invent phase files")
            return None
        record = IrisScientificAblationWorkflowPhaseDataV1.model_validate_json(self.record)
        if encoded(record) != self.record:
            raise ValueError("Original producer phase bytes changed")
        return ScientificAblationWorkflowDataV1(record, self.files)


def _returned_kernel(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    result: CaptureProgress | ScoreProgress,
    expected: ScientificAblationKernelChunkExpectationV1 | None,
) -> bytes:
    """Snapshot ONLY the original return; never inspect a later moving head."""
    exact: type[CaptureProgress | ScoreProgress]
    if phase.kind == "capture":
        module = capture if phase.stream_index is None else controlled_capture
        exact = (
            capture.IrisScientificAblationCaptureProgressV1
            if phase.stream_index is None
            else controlled_capture.IrisScientificAblationControlledCaptureProgressV1
        )
    elif phase.kind == "score":
        module = score if phase.stream_index is None else controlled_score
        exact = (
            score.IrisScientificAblationEvaluationProgressV2
            if phase.stream_index is None
            else controlled_score.IrisScientificAblationControlledEvaluationProgressV1
        )
    else:
        raise ValueError("Only an original capture/score return has a kernel prefix")
    if (
        type(result) is not exact
        or result.path != _kernel(store, phase) / module._STORE / result.plan.sha256
        or len(result.rows) != len(result.acknowledgements)
        or phase.index != len(prefix)
    ):
        raise ValueError("Exact original returned kernel and workflow prefix required")
    if expected is not None and (
        result.plan.sha256 != expected.kernel_plan_sha256
        or len(result.rows) != expected.stop_row_index
        or len(result.plan.payload.rows) != expected.total_rows
    ):
        raise ValueError("Returned kernel differs from this exact requested interval")
    return encoded(
        IrisScientificAblationWorkflowKernelProgressV1(
            workflow_plan_sha256=store.plan.sha256,
            phase=phase,
            completed_phase_sha256=tuple(data.record.sha256 for data in prefix),
            phase_is_acknowledged=False,
            kernel_plan_sha256=result.plan.sha256,
            acknowledgement_head_sha256=(
                result.acknowledgements[-1].sha256
                if result.acknowledgements
                else module._genesis(result.plan.sha256)
            ),
            acknowledged_rows=len(result.rows),
            total_rows=len(result.plan.payload.rows),
            kernel_complete_sha256=None if result.complete is None else result.complete.sha256,
            uncommitted_stage_count=result.uncommitted_stage_count,
        )
    )


class _Corpus(TypedDict):
    lifecycle_path: Path
    inference_path: Path
    inputs_root: Path
    training: ScientificParquetInputV1
    validation: ScientificParquetInputV1
    test: ScientificParquetInputV1


class _ScoreArguments(_Corpus):
    capture_path: Path
    expected_plan_sha256: str
    expected_capture_sha256: str
    python_policy: PythonSandboxPolicyV2 | None
    maximum_capture_bytes: int


class _ChunkArguments(TypedDict, total=False):
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None
    _recovery: _KernelIntervalPermission


def encoded(model: BaseModel) -> bytes:
    return canonical_json_file_bytes(model.model_dump(mode="json"))


def _model[T: BaseModel](raw: bytes, cls: type[T]) -> T:
    if not raw.endswith(b"\n"):
        raise ValueError("Canonical complete phase DATA requires one LF")
    _json(raw[:-1])
    value = cls.model_validate_json(raw[:-1])
    if encoded(value) != raw:
        raise ValueError("Phase DATA is not canonical")
    return value


def _corpus(plan: IrisScientificAblationWorkflowPlanV1) -> _Corpus:
    p = plan.payload
    return dict(
        lifecycle_path=Path(p.lifecycle_path),
        inference_path=Path(p.inference_path),
        inputs_root=Path(p.inputs_root),
        training=ScientificParquetInputV1.model_validate(p.training.model_dump(mode="python")),
        validation=ScientificParquetInputV1.model_validate(p.validation.model_dump(mode="python")),
        test=ScientificParquetInputV1.model_validate(p.test.model_dump(mode="python")),
    )


def source(
    plan: IrisScientificAblationWorkflowPlanV1,
) -> IrisScientificAblationGenerationSourceBundleV1:
    p = plan.payload
    result = read_scientific_ablation_generation_source_v1(
        lifecycle_path=Path(p.lifecycle_path),
        inference_path=Path(p.inference_path),
        expected_lifecycle_sha256=p.lifecycle_sha256,
        expected_inference_sha256=p.inference_sha256,
        expected_source_checkpoint_sha256=p.selected_checkpoint_sha256,
    )
    if result.manifest.sha256 != p.source_sha256:
        raise ValueError("Original workflow source differs from independent intent")
    return result


def _owner(
    plan: IrisScientificAblationWorkflowPlanV1,
    phase: ScientificAblationWorkflowPhaseV1,
    factory: Factory | None,
) -> IrisScientificAblationHeldoutV1 | IrisScientificAblationControlledHeldoutV1:
    if factory is None:
        raise ValueError("This original native phase requires an explicit trusted factory")
    bundle = source(plan)
    p, corpus = plan.payload, _corpus(plan)
    validation = _rows(corpus["validation"], corpus["inputs_root"])
    sampling = TemperatureTopPSamplingConfig(p.decoding.temperature, p.decoding.top_p)
    if phase.stream_index is None:
        context = build_scientific_ablation_generation_context_v1(
            bundle,
            expected_sha256=p.source_sha256,
            expected_lifecycle_sha256=p.lifecycle_sha256,
            expected_inference_sha256=p.inference_sha256,
            expected_source_checkpoint_sha256=p.selected_checkpoint_sha256,
            make_fresh_run=factory,
            validation=validation,
            sampling=sampling,
        )
        return build_scientific_ablation_heldout_v1(
            context,
            expected_reconstruction_sha256=context.reconstruction.sha256,
            test=corpus["test"],
            base=corpus["inputs_root"],
            decoding=p.decoding,
        )
    controlled = build_scientific_ablation_controlled_generation_context_v1(
        bundle,
        expected_sha256=p.source_sha256,
        expected_lifecycle_sha256=p.lifecycle_sha256,
        expected_inference_sha256=p.inference_sha256,
        expected_source_checkpoint_sha256=p.selected_checkpoint_sha256,
        make_fresh_run=factory,
        validation=validation,
        sampling=sampling,
    )
    return build_scientific_ablation_controlled_heldout_v1(
        controlled,
        expected_reconstruction_sha256=controlled.reconstruction.sha256,
        test=corpus["test"],
        base=corpus["inputs_root"],
        decoding=p.decoding,
        control=p.controls[phase.stream_index],
    )


def _ready(
    owner: IrisScientificAblationHeldoutV1 | IrisScientificAblationControlledHeldoutV1,
) -> None:
    if type(owner) is IrisScientificAblationHeldoutV1:
        heldout_entry(owner)
    elif type(owner) is IrisScientificAblationControlledHeldoutV1:
        controlled_entry(owner)
    else:
        raise ValueError("Original exact native held-out owner required")


def _kernel(
    store: ScientificAblationWorkflowStoreV1, phase: ScientificAblationWorkflowPhaseV1
) -> Path:
    return store.path / "kernels" / phase.name


def budget(
    store: ScientificAblationWorkflowStoreV1,
    root: Path,
    claim: _Claim,
    reserve: int,
    *,
    kernel: bool = True,
) -> int:
    """Bound one existing subtree while reserving ALL other occupied workflow data."""
    claim.reserve(reserve)
    claim.minimum_free_for(reserve)  # Validate the original kernel limit BEFORE a factory.
    if not root.is_relative_to(store.path) or root == store.path:
        raise ValueError("Only an owned workflow subtree may receive a kernel quota")
    occupied = _tree_size(store.path)[0]
    subtree = _tree_size(root)[0] if os.path.lexists(root) else 0
    resources = store.plan.payload.resources
    available = (
        resources.maximum_storage_bytes - (occupied - subtree) - reserve - claim.reserved_bytes
    )
    result = min(resources.maximum_kernel_storage_bytes, available) if kernel else available
    if result < 1:
        raise ValueError("No remaining whole-workflow budget for the complete kernel")
    claim.check()
    return result


def _capture_path(
    store: ScientificAblationWorkflowStoreV1, phase: ScientificAblationWorkflowPhaseV1, digest: str
) -> Path:
    kind = capture._STORE if phase.stream_index is None else controlled_capture._STORE
    return _kernel(store, phase) / kind / digest


def _capture_data(
    data: ScientificAblationWorkflowDataV1,
) -> ScientificAblationWorkflowCaptureDataV1:
    return _model(dict(data.files)["capture.json"], ScientificAblationWorkflowCaptureDataV1)


def _capture_read(
    store: ScientificAblationWorkflowStoreV1, data: ScientificAblationWorkflowDataV1
) -> CaptureProgress:
    phase, descriptor = data.record.payload.phase, _capture_data(data)
    path = _capture_path(store, phase, descriptor.plan_sha256)
    if phase.stream_index is None:
        result: CaptureProgress = capture.read_scientific_ablation_capture_v1(
            path,
            expected_plan_sha256=descriptor.plan_sha256,
            expected_capture_sha256=descriptor.capture_sha256,
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    else:
        result = controlled_capture.read_scientific_ablation_controlled_capture_v1(
            path,
            expected_plan_sha256=descriptor.plan_sha256,
            expected_capture_sha256=descriptor.capture_sha256,
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    p, actual = store.plan.payload, result.plan.payload
    if (
        descriptor.protocol != ("ordinary" if phase.stream_index is None else "controlled")
        or actual.source != source(store.plan).manifest
        or actual.input.sha256 != descriptor.input_sha256
        or actual.input.payload.decoding != p.decoding
        or actual.input.payload.test_parquet_sha256 != p.test.sha256
        or len(result.rows) != descriptor.examples
        or result.complete is None
    ):
        raise ValueError("Complete original capture differs from workflow DATA intent")
    if type(result) is controlled_capture.IrisScientificAblationControlledCaptureProgressV1:
        if phase.stream_index is None:
            raise ValueError("Controlled capture cannot replace an ordinary stream")
        from iris.ablations.controls import CognitiveAblationControlPlanV1

        intent = CognitiveAblationControlPlanV1.model_validate_json(
            result.plan.payload.input.payload.generation_control_plan_json
        )
        if intent.control != p.controls[phase.stream_index]:
            raise ValueError("Original controlled-generation intent differs")
    return result


def _score_args(
    store: ScientificAblationWorkflowStoreV1, data: ScientificAblationWorkflowDataV1
) -> _ScoreArguments:
    descriptor = _capture_data(data)
    return dict(
        **_corpus(store.plan),
        capture_path=_capture_path(store, data.record.payload.phase, descriptor.plan_sha256),
        expected_plan_sha256=descriptor.plan_sha256,
        expected_capture_sha256=descriptor.capture_sha256,
        python_policy=store.plan.payload.python_policy,
        maximum_capture_bytes=512 * 1024**2,
    )


def _score_read(
    store: ScientificAblationWorkflowStoreV1,
    data: ScientificAblationWorkflowDataV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
) -> ScoreProgress:
    phase = data.record.payload.phase
    descriptor = _model(dict(data.files)["evaluation.json"], ScientificAblationWorkflowScoreDataV1)
    args = _score_args(store, prefix[phase.index - 1])
    if phase.stream_index is None:
        plan = score.build_scientific_ablation_evaluation_plan_v2(**args)
        result: ScoreProgress = score.read_scientific_ablation_evaluation_progress_v2(
            _kernel(store, phase) / score._STORE / plan.sha256,
            expected_evaluation_plan_sha256=plan.sha256,
            expected_plan_sha256=args["expected_plan_sha256"],
            expected_capture_sha256=args["expected_capture_sha256"],
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    else:
        controlled_plan = controlled_score.build_scientific_ablation_controlled_evaluation_plan_v1(
            **args
        )
        result = controlled_score.read_scientific_ablation_controlled_evaluation_progress_v1(
            _kernel(store, phase) / controlled_score._STORE / controlled_plan.sha256,
            expected_evaluation_plan_sha256=controlled_plan.sha256,
            expected_plan_sha256=args["expected_plan_sha256"],
            expected_capture_sha256=args["expected_capture_sha256"],
            maximum_storage_bytes=store.plan.payload.resources.maximum_kernel_storage_bytes,
        )
    if (
        result.evaluation is None
        or result.complete is None
        or descriptor.protocol != ("ordinary" if phase.stream_index is None else "controlled")
        or descriptor.evaluation_plan_sha256 != result.plan.sha256
        or descriptor.evaluation_sha256 != result.evaluation.sha256
        or descriptor.complete_sha256 != result.complete.sha256
        or descriptor.capture_plan_sha256 != args["expected_plan_sha256"]
        or descriptor.capture_sha256 != args["expected_capture_sha256"]
        or descriptor.examples != len(result.rows)
    ):
        raise ValueError("Original complete score ledger differs from workflow DATA")
    return result


def _ordinary(
    store: ScientificAblationWorkflowStoreV1, prefix: tuple[ScientificAblationWorkflowDataV1, ...]
) -> score.IrisScientificAblationEvaluationProgressV2:
    result = _score_read(store, prefix[1], prefix)
    if type(result) is not score.IrisScientificAblationEvaluationProgressV2:
        raise ValueError("Original ordinary evaluation is required for native diagnostics")
    return result


def _causal(
    store: ScientificAblationWorkflowStoreV1, prefix: tuple[ScientificAblationWorkflowDataV1, ...]
) -> ScientificAblationCausalInputV1:
    actual = _ordinary(store, prefix)
    if actual.evaluation is None:
        raise ValueError("Original ordinary complete evaluation required")
    descriptor = _capture_data(prefix[0])
    return ScientificAblationCausalInputV1(
        **_corpus(store.plan),
        evaluation_path=actual.path,
        expected_evaluation_plan_sha256=actual.plan.sha256,
        expected_evaluation_sha256=actual.evaluation.sha256,
        capture_path=_capture_path(store, prefix[0].record.payload.phase, descriptor.plan_sha256),
        expected_plan_sha256=descriptor.plan_sha256,
        expected_capture_sha256=descriptor.capture_sha256,
    )


def _writer(
    store: ScientificAblationWorkflowStoreV1,
    data: ScientificAblationWorkflowDataV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
) -> tuple[
    IrisScientificAblationWriterSourceV1, WriterReport, ScientificAblationWriterCPURuntimeV1
]:
    files = dict(data.files)
    runtime_data = _json(files["runtime.json"][:-1])
    if set(runtime_data) != {"torch_version", "torch_git_version"}:
        raise ValueError("Exact retained original writer CPU binding required")
    runtime = ScientificAblationWriterCPURuntimeV1(**runtime_data)
    if canonical_json_file_bytes(runtime_data) != files["runtime.json"]:
        raise ValueError("Canonical retained writer runtime required")
    # The independently retained PHASE digest authenticates all these file bytes.
    source_model = IrisScientificAblationWriterSourceV1.model_validate_json(files["source.json"])
    original = read_scientific_ablation_writer_source_v1(
        files["source.json"],
        expected_sha256=source_model.sha256,
        maximum_source_bytes=len(files["source.json"]),
    )
    pf = scientific_ablation_writer_source_preflight_v1(original, expected_sha256=original.sha256)
    report = read_scientific_ablation_writer_report_v1(
        files["report.json"], preflight=pf, runtime=runtime
    )
    # The reader already validates all report semantics and compares its
    # canonical serialization with these exact bytes. Keep the workflow
    # bindings below, but do not serialize the same detached report again.
    actual = _ordinary(store, prefix)
    p = original.payload
    policy = (
        store.plan.payload.workspace
        if data.record.payload.phase.kind == "workspace"
        else store.plan.payload.message
    )
    if (
        p.protocol != data.record.payload.phase.kind
        or p.source_metadata.evaluation_plan != actual.plan
        or p.source_metadata.evaluation != actual.evaluation
        or p.source_metadata.evaluation_complete != actual.complete
        or pf.observed_budget.payload.design.payload.policy != policy
        or p.maximum_report_bytes != store.plan.payload.resources.maximum_writer_report_bytes
        or len(files["report.json"]) > p.maximum_report_bytes
    ):
        raise ValueError("Original writer DATA differs from full workflow source/policy/score")
    return original, report, runtime


def _na(
    plan: IrisScientificAblationWorkflowPlanV1, phase: ScientificAblationWorkflowPhaseV1
) -> bytes:
    if source(plan).inference.manifest.payload.model.workspace.cycles != 1:
        raise ValueError("Only actual C1 lacks a next-cycle writer endpoint")
    return canonical_json_file_bytes(
        dict(
            kind="iris-scientific-ablation-workflow-writer-not-applicable-v1",
            plan_sha256=plan.sha256,
            phase=phase.name,
            cycles=1,
            reason="no_next_cycle_writer_endpoint",
            scientific_acceptance="not_assessed",
        )
    )


def validate_data(
    store: ScientificAblationWorkflowStoreV1,
    data: ScientificAblationWorkflowDataV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
) -> None:
    """Full physical original prefix semantics; never calls a factory or executes rows."""
    phase = data.record.payload.phase
    if len(prefix) != phase.index:
        raise ValueError("The entire exact ordered predecessor prefix is required")
    if phase.kind == "capture":
        _capture_read(store, data)
    elif phase.kind == "score":
        _score_read(store, data, prefix)
    elif phase.kind == "diagnostics":
        files = dict(data.files)
        ordinary = IrisScientificAblationDiagnosticsV1.model_validate_json(files["ordinary.json"])
        compact = IrisScientificAblationCollapseV1.model_validate_json(files["collapse.json"])
        pair = collapse._detach_completed(ordinary, compact)
        actual = _ordinary(store, prefix)
        if (
            diagnostics._checked(pair.diagnostics) != files["ordinary.json"]
            or collapse._checked(pair.collapse) != files["collapse.json"]
            or pair.diagnostics.payload.evaluation_plan != actual.plan
            or pair.diagnostics.payload.evaluation != actual.evaluation
            or pair.diagnostics.payload.evaluation_complete != actual.complete
            or pair.collapse.payload.policy != store.plan.payload.collapse
        ):
            raise ValueError("Original one-pass diagnostic DATA differs from the actual evaluation")
    elif phase.kind in ("workspace", "message"):
        if data.record.payload.origin == "not_applicable_no_next_cycle":
            if data.files != (("summary.json", _na(store.plan, phase)),):
                raise ValueError("Original next-cycle N/A DATA differs")
        else:
            _writer(store, data, prefix)
    elif phase.kind == "publish" and data.files != (
        ("summary.json", _summary(store, prefix, inspect=True)),
    ):
        raise ValueError("Original complete DATA publication differs")


def _summary(
    store: ScientificAblationWorkflowStoreV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    *,
    inspect: bool,
) -> bytes:
    published: dict[str, str] = {}
    for data in prefix:
        phase = data.record.payload.phase
        if (
            phase.kind not in ("workspace", "message")
            or data.record.payload.origin == "not_applicable_no_next_cycle"
        ):
            continue
        original, report, runtime = _writer(store, data, prefix[: phase.index])
        protocol: WriterSourceProtocol = "workspace" if phase.kind == "workspace" else "message"
        path = (
            store.path
            / "published"
            / phase.name
            / f"scientific-ablation-{protocol}-writer-v1"
            / report.sha256
        )
        if inspect:
            read_scientific_ablation_writer_bundle_v1(
                path,
                expected_protocol=protocol,
                expected_report_sha256=report.sha256,
                expected_source_sha256=original.sha256,
                runtime=runtime,
                maximum_source_bytes=len(dict(data.files)["source.json"]),
                maximum_report_bytes=original.payload.maximum_report_bytes,
            )
        published[phase.name] = path.relative_to(store.path).as_posix()
    return canonical_json_file_bytes(
        dict(
            kind="iris-scientific-ablation-workflow-publication-v1",
            plan_sha256=store.plan.sha256,
            completed_phase_sha256=tuple(data.record.sha256 for data in prefix),
            published=published,
            scope="stored_workflow_data_not_restored_native_authority",
            scientific_acceptance="not_assessed",
        )
    )


def _native_reservation(
    plan: IrisScientificAblationWorkflowPlanV1, phase: ScientificAblationWorkflowPhaseV1
) -> tuple[int, bool]:
    """Byte reservation only, NEVER a native ownership or completion token.

    A C1writer has no next-cycle report. Authenticate the physical original
    source before selecting that smaller reservation; execution separately
    checks the actual original owner. Ordinary diagnostics remain measurable.
    """
    metadata = 4 * 1024**2
    if phase.kind == "diagnostics":
        return 128 * 1024**2 + metadata, False
    if phase.kind not in ("workspace", "message"):
        raise ValueError("Only an original native phase has a native byte reservation")
    cycles = source(plan).inference.manifest.payload.model.workspace.cycles
    if type(cycles) is not int or not 1 <= cycles <= 64:
        raise ValueError("Exact authenticated original cycle count required")
    if cycles == 1:
        return metadata, True
    return MAX_REPORT_BYTES + plan.payload.resources.maximum_writer_report_bytes + metadata, False


def _execute(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    factory: Factory | None,
    claim: _Claim,
    rows_per_call: int | None,
    *,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    snapshot_kernel: bool = False,
    _recovery: _KernelIntervalPermission | None = None,
) -> _Produced:
    """One original phase/chunk. The caller must journal its attempt BEFORE entry."""
    chunk_arguments: _ChunkArguments = {}
    if _recovery is not None:
        if expected_chunk is None or not snapshot_kernel:
            raise ValueError("Recovery requires the exact original interval and DATA snapshot")
        chunk_arguments["_recovery"] = _recovery
    if expected_chunk is not None:
        if phase.kind not in ("capture", "score"):
            raise ValueError("Only capture/score kernels accept an exact chunk request")
        owned_chunk = checked_workflow_model_v1(
            expected_chunk, ScientificAblationKernelChunkExpectationV1
        )
        chunk_arguments["expected_chunk"] = checked_kernel_chunk_v1(
            owned_chunk,
            expected_plan_sha256=owned_chunk.kernel_plan_sha256,
            rows_per_call=rows_per_call,
        )
    previous = prefix[-1].record.sha256 if prefix else None
    p = store.plan.payload
    files: dict[str, bytes]
    kernel_progress: bytes | None = None
    origin = "kernel_data"
    if phase.kind == "capture":
        quota = budget(store, _kernel(store, phase), claim, 4 * 1024**2)
        owner = _owner(store.plan, phase, factory)
        if type(owner) is IrisScientificAblationHeldoutV1:
            plan = capture.build_scientific_ablation_capture_plan_v1(
                owner, expected_input_sha256=owner.manifest.sha256
            )
            progress: CaptureProgress = capture.capture_scientific_ablation_heldout_v1(
                owner,
                root=_kernel(store, phase),
                expected_input_sha256=owner.manifest.sha256,
                expected_plan_sha256=plan.sha256,
                rows_per_call=rows_per_call,
                maximum_storage_bytes=quota,
                minimum_free_bytes=claim.minimum_free_for(4 * 1024**2),
                **chunk_arguments,
            )
        elif type(owner) is IrisScientificAblationControlledHeldoutV1:
            controlled_plan = (
                controlled_capture.build_scientific_ablation_controlled_capture_plan_v1(
                    owner, expected_input_sha256=owner.manifest.sha256
                )
            )
            progress = controlled_capture.capture_scientific_ablation_controlled_heldout_v1(
                owner,
                root=_kernel(store, phase),
                expected_input_sha256=owner.manifest.sha256,
                expected_plan_sha256=controlled_plan.sha256,
                rows_per_call=rows_per_call,
                maximum_storage_bytes=quota,
                minimum_free_bytes=claim.minimum_free_for(4 * 1024**2),
                **chunk_arguments,
            )
        else:
            raise ValueError("Exact original ordinary or controlled owner required")
        _ready(owner)
        source(store.plan)
        claim.reserve(4 * 1024**2)
        if snapshot_kernel:
            kernel_progress = _returned_kernel(store, phase, prefix, progress, expected_chunk)
        if progress.complete is None:
            return _Produced(None, (), kernel_progress)
        files = {
            "capture.json": encoded(
                ScientificAblationWorkflowCaptureDataV1(
                    protocol="ordinary" if phase.stream_index is None else "controlled",
                    plan_sha256=progress.plan.sha256,
                    input_sha256=progress.plan.payload.input.sha256,
                    capture_sha256=progress.complete.sha256,
                    examples=len(progress.rows),
                )
            )
        }
    elif phase.kind == "score":
        args = _score_args(store, prefix[-1])
        quota = budget(store, _kernel(store, phase), claim, 4 * 1024**2)
        if phase.stream_index is None:
            score_plan = score.build_scientific_ablation_evaluation_plan_v2(**args)
            evaluated: ScoreProgress = score.evaluate_scientific_ablation_capture_resumable_v2(
                **args,
                root=_kernel(store, phase),
                expected_evaluation_plan_sha256=score_plan.sha256,
                rows_per_call=rows_per_call,
                maximum_storage_bytes=quota,
                minimum_free_bytes=claim.minimum_free_for(4 * 1024**2),
                **chunk_arguments,
            )
        else:
            controlled_score_plan = (
                controlled_score.build_scientific_ablation_controlled_evaluation_plan_v1(**args)
            )
            evaluated = (
                controlled_score.evaluate_scientific_ablation_controlled_capture_resumable_v1(
                    **args,
                    root=_kernel(store, phase),
                    expected_evaluation_plan_sha256=controlled_score_plan.sha256,
                    rows_per_call=rows_per_call,
                    maximum_storage_bytes=quota,
                    minimum_free_bytes=claim.minimum_free_for(4 * 1024**2),
                    **chunk_arguments,
                )
            )
        claim.reserve(4 * 1024**2)
        if snapshot_kernel:
            kernel_progress = _returned_kernel(store, phase, prefix, evaluated, expected_chunk)
        if evaluated.complete is None or evaluated.evaluation is None:
            return _Produced(None, (), kernel_progress)
        files = {
            "evaluation.json": encoded(
                ScientificAblationWorkflowScoreDataV1(
                    protocol="ordinary" if phase.stream_index is None else "controlled",
                    evaluation_plan_sha256=evaluated.plan.sha256,
                    evaluation_sha256=evaluated.evaluation.sha256,
                    complete_sha256=evaluated.complete.sha256,
                    capture_plan_sha256=args["expected_plan_sha256"],
                    capture_sha256=args["expected_capture_sha256"],
                    examples=len(evaluated.rows),
                )
            )
        }
    elif phase.kind in ("diagnostics", "workspace", "message"):
        origin = "original_native_result_data"
        reservation, no_next_cycle = _native_reservation(store.plan, phase)
        claim.reserve(reservation)
        ordinary_owner = _owner(store.plan, phase, factory)
        if type(ordinary_owner) is not IrisScientificAblationHeldoutV1:
            raise ValueError("Original native diagnostics require an ordinary owner")
        if phase.kind != "diagnostics" and no_next_cycle != (
            ordinary_owner._guard.context._run._optimizer._model.teacher._cycles == 1
        ):
            raise ValueError("Actual original owner differs from authenticated writer geometry")
        causal = _causal(store, prefix)
        if phase.kind == "diagnostics":
            completed = collapse.measure_scientific_ablation_collapse_completed_v1(
                ordinary_owner,
                **_corpus(store.plan),
                evaluation_path=causal.evaluation_path,
                expected_evaluation_plan_sha256=causal.expected_evaluation_plan_sha256,
                expected_evaluation_sha256=causal.expected_evaluation_sha256,
                capture_path=causal.capture_path,
                expected_plan_sha256=causal.expected_plan_sha256,
                expected_capture_sha256=causal.expected_capture_sha256,
                policy=p.collapse,
            )
            files = {
                "ordinary.json": diagnostics._checked(completed.diagnostics),
                "collapse.json": collapse._checked(completed.collapse),
            }
        elif no_next_cycle:
            origin, files = "not_applicable_no_next_cycle", {"summary.json": _na(store.plan, phase)}
        else:
            protocol: WriterSourceProtocol = "workspace" if phase.kind == "workspace" else "message"
            job = create_scientific_ablation_writer_job_v1(
                ordinary_owner,
                input=causal,
                policy=p.workspace if protocol == "workspace" else p.message,
                limits=ScientificAblationCausalCompleteLimitsV1(
                    maximum_observation_scalars=p.resources.maximum_observation_scalars,
                    maximum_report_bytes=p.resources.maximum_writer_report_bytes,
                ),
                expected_protocol=protocol,
            )
            writer = job.measure()
            pf = scientific_ablation_writer_source_preflight_v1(
                writer.source, expected_sha256=writer.source.sha256
            )
            files = {
                "source.json": scientific_ablation_writer_source_bytes_v1(
                    writer.source, expected_sha256=writer.source.sha256
                ),
                "report.json": scientific_ablation_writer_report_bytes_v1(
                    writer.report, preflight=pf, runtime=writer.runtime
                ),
                "runtime.json": canonical_json_file_bytes(
                    dict(
                        torch_version=writer.runtime.torch_version,
                        torch_git_version=writer.runtime.torch_git_version,
                    )
                ),
            }
        _ready(ordinary_owner)
        source(store.plan)
        _ready(
            ordinary_owner
        )  # Serialization and final source audit ran while caller ownership is READY.
    else:
        origin = "publication_data"
        for data in prefix:
            kind = data.record.payload.phase.kind
            if (
                kind not in ("workspace", "message")
                or data.record.payload.origin == "not_applicable_no_next_cycle"
            ):
                continue
            original, report, runtime = _writer(
                store, data, prefix[: data.record.payload.phase.index]
            )
            protocol = "workspace" if kind == "workspace" else "message"
            root = store.path / "published" / kind
            required = sum(len(content) for name, content in data.files if name != "runtime.json")
            quota = budget(store, root, claim, 4 * 1024**2, kernel=False)
            target = root / f"scientific-ablation-{protocol}-writer-v1" / report.sha256
            claim.reserve((0 if os.path.lexists(target) else required) + 4 * 1024**2)
            write_scientific_ablation_writer_bundle_v1(
                report,
                source=original,
                root=root,
                expected_protocol=protocol,
                expected_report_sha256=report.sha256,
                expected_source_sha256=original.sha256,
                runtime=runtime,
                maximum_source_bytes=len(dict(data.files)["source.json"]),
                maximum_report_bytes=original.payload.maximum_report_bytes,
                maximum_storage_bytes=quota,
                minimum_free_bytes=claim.minimum_free_for(4 * 1024**2),
            )
        files = {"summary.json": _summary(store, prefix, inspect=True)}
    result = store.make_data(phase, previous, origin=origin, files=files)
    validate_data(store, result, prefix)
    claim.reserve(0)
    return _Produced(encoded(result.record), result.files, kernel_progress)


def execute(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    factory: Factory | None,
    claim: _Claim,
    rows_per_call: int | None,
    *,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
) -> ScientificAblationWorkflowDataV1 | None:
    """Legacy phase-only result, with unchanged original execution and claims."""
    return _execute(
        store, phase, prefix, factory, claim, rows_per_call, expected_chunk=expected_chunk
    ).data()


def execute_produced(
    store: ScientificAblationWorkflowStoreV1,
    phase: ScientificAblationWorkflowPhaseV1,
    prefix: tuple[ScientificAblationWorkflowDataV1, ...],
    factory: Factory | None,
    claim: _Claim,
    rows_per_call: int | None,
    *,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    _recovery: _KernelIntervalPermission | None = None,
) -> _Produced:
    """Original producer seam, NOT a caller callback or an independently durable ACK."""
    return _execute(
        store,
        phase,
        prefix,
        factory,
        claim,
        rows_per_call,
        expected_chunk=expected_chunk,
        snapshot_kernel=True,
        _recovery=_recovery,
    )
