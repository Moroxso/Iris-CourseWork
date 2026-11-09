"""Bind original workflow to an actual terminal operator/export and concrete factory.

Reading/validating and creating a workflow owner do not construct a native graph.
The original workflow still owns phase execution, independent ACKs and UNCERTAIN
boundaries. This does not add an implicit phase dispatcher or native JSON authority.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_command_composition import (
    create_scientific_ablation_command_run_v1,
)
from iris.application.scientific_ablation_execution import _local
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
    read_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_operator import (
    _recipe,
    inspect_scientific_ablation_operator_v1,
)
from iris.application.scientific_ablation_operator_export import (
    _checked_request,
    inspect_scientific_ablation_operator_export_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowV1,
    _create_workflow_with_store,
    _Seal,
    create_scientific_ablation_workflow_v1,
)
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
    checked_workflow_model_v1,
)
from iris.application.scientific_command import ScientificParquetInputV1, _rows
from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    COMMAND_WORKFLOW_DEFINITION_DOMAIN,
    IrisScientificAblationCommandWorkflowDefinitionV1,
    ScientificAblationCommandWorkflowDefinitionPayloadV1,
)
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import IrisScientificAblationWorkflowPlanV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256


class IrisScientificAblationOperatorWorkflowError(ApplicationError):
    """Redacted operator/export/plan mismatch, not permission to remeasure an attempt."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationOperatorGenerationSourceV1:
    """Verified DATA only; constructing this record does not grant native ownership."""

    operator_sha256: str
    execution_sha256: str
    export_result_sha256: str
    export_path: Path
    lifecycle_path: Path
    inference_path: Path
    bundle: IrisScientificAblationGenerationSourceBundleV1 = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class _OperatorWorkflowBinding:
    """Original process-local source binding, never a serialized result/owner."""

    job: IrisScientificAblationWorkflowV1
    operator_path: Path
    request_json: bytes
    export_root: Path
    base: Path
    descriptor_json: bytes
    execution_sha256: str
    export_result_sha256: str
    factory: steps.Factory


def _command_definition(
    plan: IrisScientificAblationWorkflowPlanV1,
    request: IrisScientificAblationExportRequestV1,
    source: IrisScientificAblationOperatorGenerationSourceV1,
) -> IrisScientificAblationCommandWorkflowDefinitionV1:
    payload = ScientificAblationCommandWorkflowDefinitionPayloadV1(
        operator_sha256=source.operator_sha256,
        execution_sha256=source.execution_sha256,
        export_request_sha256=request.sha256,
        export_result_sha256=source.export_result_sha256,
        workflow_plan_sha256=plan.sha256,
    )
    return IrisScientificAblationCommandWorkflowDefinitionV1(
        sha256=domain_sha256(
            COMMAND_WORKFLOW_DEFINITION_DOMAIN,
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )


def _revalidate_operator_binding(job: IrisScientificAblationWorkflowV1, seal: _Seal) -> None:
    """Full original binding before EVERY mutation, including score/publication.

    A concrete private binding stays attached to the original sealed job/factory.
    No user callback, constructible source DTO or saved result restores that job.
    This is preflight only, not an independent producer ACK or journal protocol.
    """
    try:
        binding = seal.operator_binding
        if (
            type(binding) is not _OperatorWorkflowBinding
            or binding.job is not job
            or job._seal is not seal
            or binding.factory is not seal.factory
        ):
            raise ValueError("Original operator/workflow binding identity differs")
        request = IrisScientificAblationExportRequestV1.model_validate_json(binding.request_json)
        if canonical_json_file_bytes(request.model_dump(mode="json")) != binding.request_json:
            raise ValueError("Original export request bytes changed")
        owned, source = _bind(
            binding.operator_path, request, binding.export_root, seal.store.plan, binding.base
        )
        current = inspect_scientific_ablation_operator_v1(binding.operator_path)
        if (
            steps.encoded(owned) != seal.plan
            or canonical_json_file_bytes(current.descriptor.model_dump(mode="json"))
            != binding.descriptor_json
            or source.operator_sha256 != current.descriptor.sha256
            or source.execution_sha256 != binding.execution_sha256
            or source.export_result_sha256 != binding.export_result_sha256
        ):
            raise ValueError("Original operator/export/source binding changed")
        if (
            seal.storage_definition is not None
            and steps.encoded(_command_definition(owned, request, source))
            != seal.storage_definition
        ):
            raise ValueError("Command namespace differs from the original operator binding")
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorWorkflowError(
            "Ablation workflow operation source/plan preflight failed"
        ) from None


def read_scientific_ablation_operator_generation_source_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
) -> IrisScientificAblationOperatorGenerationSourceV1:
    """Actual terminal selected-best source, no implicit export/publication or model."""
    try:
        request = _checked_request(request)
        if request.payload.selection != "selected_best":
            raise ValueError("Generation requires explicit terminal selected-best export")
        snapshot = inspect_scientific_ablation_operator_v1(path)
        events = snapshot.execution.events
        if (
            not events
            or events[-1].sha256 != request.payload.checkpoint_event_sha256
            or events[-1].payload.after.phase != "stopped"
        ):
            raise ValueError("An actual final acknowledged stopped checkpoint is required")
        receipt = inspect_scientific_ablation_operator_export_v1(
            path, request=request, root=export_root
        )
        lifecycle = (
            snapshot.execution.path
            / "checkpoints/scientific-ablation-lifecycle-v1"
            / request.payload.lifecycle_sha256
        )
        bundle = read_scientific_ablation_generation_source_v1(
            lifecycle_path=lifecycle,
            inference_path=receipt.inference_path,
            expected_lifecycle_sha256=request.payload.lifecycle_sha256,
            expected_inference_sha256=receipt.result.payload.inference_sha256,
            expected_source_checkpoint_sha256=request.payload.source_checkpoint_sha256,
        )
        if inspect_scientific_ablation_operator_v1(path).descriptor != snapshot.descriptor:
            raise ValueError("Operator descriptor changed while reading terminal source")
        return IrisScientificAblationOperatorGenerationSourceV1(
            snapshot.descriptor.sha256,
            snapshot.execution.header.sha256,
            receipt.result.sha256,
            receipt.path,
            lifecycle,
            receipt.inference_path,
            bundle,
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorWorkflowError(
            "Ablation operator generation source is not a published terminal selected-best pair"
        ) from None


def _bind(
    path: Path,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
) -> tuple[IrisScientificAblationWorkflowPlanV1, IrisScientificAblationOperatorGenerationSourceV1]:
    plan = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
    request = _checked_request(request)
    base = _local(base)
    payload = plan.payload
    if not all(
        Path(p).is_absolute()
        for p in (payload.inputs_root, payload.lifecycle_path, payload.inference_path)
    ):
        raise ValueError("Absolute workflow roots independent of later CWD are required")
    source = read_scientific_ablation_operator_generation_source_v1(
        path, request=request, export_root=export_root
    )
    snapshot = inspect_scientific_ablation_operator_v1(path)
    recipe = _recipe(snapshot.descriptor)
    if (
        snapshot.descriptor.sha256 != source.operator_sha256
        or payload.source_sha256 != source.bundle.manifest.sha256
        or payload.lifecycle_sha256 != request.payload.lifecycle_sha256
        or payload.inference_sha256 != source.bundle.inference.manifest.sha256
        or payload.selected_checkpoint_sha256 != request.payload.source_checkpoint_sha256
        or _local(Path(payload.inputs_root)) != base
        or _local(Path(payload.lifecycle_path)) != source.lifecycle_path
        or _local(Path(payload.inference_path)) != source.inference_path
        or payload.training.model_dump(mode="json") != recipe.train.model_dump(mode="json")
        or payload.validation.model_dump(mode="json") != recipe.validation.model_dump(mode="json")
    ):
        raise ValueError("Workflow plan does not bind this operator/actual export/recipe inputs")
    if not 1 <= payload.test.maximum_rows <= 5000:
        raise ValueError("Held-out input must preserve its explicit complete5000 ceiling")
    rows = _rows(
        ScientificParquetInputV1.model_validate(payload.test.model_dump(mode="python")), base
    )
    if recipe.purpose == "scientific" and (payload.test.maximum_rows != 5000 or len(rows) != 5000):
        raise ValueError("Scientific workflow requires the original complete5000 test split")
    return plan, source


def bind_scientific_ablation_operator_workflow_plan_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
) -> IrisScientificAblationWorkflowPlanV1:
    """Validate actual source/corpora before a model; returned DATA is not an owner."""
    try:
        owned, _ = _bind(path, request, export_root, plan, base)
        return owned
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorWorkflowError(
            "Ablation workflow source/plan binding failed"
        ) from None


def create_scientific_ablation_operator_workflow_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
    root: Path,
    completed_phase_sha256: tuple[str, ...] = (),
) -> IrisScientificAblationWorkflowV1:
    """Create the ORIGINAL workflow with a concrete recipe factory; execute no phase.

    Callers must retain original phase anchors independently. Existing native
    attempts without those anchors remain UNCERTAIN; never adopt/reap/replay them.
    No callback or allocation-ceiling override is accepted from serialized DATA.
    """
    return _create_operator_workflow(
        path,
        request=request,
        export_root=export_root,
        plan=plan,
        base=base,
        root=root,
        completed_phase_sha256=completed_phase_sha256,
        command_namespace=False,
    )


def _create_operator_workflow(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
    root: Path,
    completed_phase_sha256: tuple[str, ...] = (),
    command_namespace: bool,
) -> IrisScientificAblationWorkflowV1:
    """Concrete internal composition for the future command coordinator.

    Namespace opt-in is NOT a durable command API. No arbitrary factory, owner,
    namespace definition or storage path is recovered from serialized DATA.
    """
    try:
        if type(command_namespace) is not bool:
            raise ValueError("Exact explicit command namespace mode required")
        request = _checked_request(request)
        base, root = _local(base), _local(root)
        owned, source = _bind(path, request, export_root, plan, base)
        snapshot = inspect_scientific_ablation_operator_v1(path)
        if snapshot.descriptor.sha256 != source.operator_sha256:
            raise ValueError("Operator changed before original workflow creation")
        command_store = (
            ScientificAblationWorkflowStoreV1._for_commands(
                root, owned, _command_definition(owned, request, source)
            )
            if command_namespace
            else None
        )
        target = (
            command_store.path
            if command_store is not None
            else root / "scientific-ablation-native-workflow-v1" / owned.sha256
        )
        for protected in (
            snapshot.path.parent,
            _local(export_root) / "scientific-ablation-operator-exports-v1",
            source.export_path,
            source.lifecycle_path,
            source.inference_path,
        ):
            if target.is_relative_to(protected) or protected.is_relative_to(target):
                raise ValueError("Workflow output may not overlap original source artifacts")
        recipe = _recipe(snapshot.descriptor)
        maximum = snapshot.descriptor.payload.maximum_trainable_parameter_bytes

        def fresh() -> IrisScientificAblationRunV1:
            # This is trusted source code, not a factory/owner recovered from JSON.
            _revalidate_operator_binding(job, job._seal)
            return create_scientific_ablation_command_run_v1(
                config=recipe, base=base, maximum_trainable_parameter_bytes=maximum
            )

        job = (
            _create_workflow_with_store(
                command_store,
                make_fresh_run=fresh,
                completed_phase_sha256=completed_phase_sha256,
            )
            if command_store is not None
            else create_scientific_ablation_workflow_v1(
                root=root,
                plan=owned,
                make_fresh_run=fresh,
                completed_phase_sha256=completed_phase_sha256,
            )
        )
        job._seal = replace(
            job._seal,
            operator_binding=_OperatorWorkflowBinding(
                job=job,
                operator_path=snapshot.path,
                request_json=canonical_json_file_bytes(request.model_dump(mode="json")),
                export_root=_local(export_root),
                base=base,
                descriptor_json=canonical_json_file_bytes(
                    snapshot.descriptor.model_dump(mode="json")
                ),
                execution_sha256=source.execution_sha256,
                export_result_sha256=source.export_result_sha256,
                factory=fresh,
            ),
        )
        return job
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorWorkflowError(
            "Ablation operator workflow creation failed"
        ) from None


__all__ = [
    "IrisScientificAblationOperatorGenerationSourceV1",
    "IrisScientificAblationOperatorWorkflowError",
    "bind_scientific_ablation_operator_workflow_plan_v1",
    "create_scientific_ablation_operator_workflow_v1",
    "read_scientific_ablation_operator_generation_source_v1",
]
