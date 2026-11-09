"""Original source-owned causal transaction and genuine empty teacher controls.

This common phase is intended for reuse inside complete writer-effect services, not
called twice and hidden from their budgets. No constructor, generation, rescoring,
candidate execution, publication or causal-effect acceptance is performed here.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from hashlib import sha256
from pathlib import Path

import torch
from pydantic import BaseModel

from iris.ablations.controls import (
    ablation_control_graph_sha256_v1,
    ablation_control_plan_sha256_v1,
)
from iris.application._scientific_ablation_request_owner import _AblationRequestOwner
from iris.application.scientific_ablation_causal_receipts import _ObservedPassSeal, _seal_observed
from iris.application.scientific_ablation_diagnostics import _teacher
from iris.application.scientific_ablation_evaluation_ledger import (
    IrisScientificAblationEvaluationProgressV2,
    read_scientific_ablation_evaluation_progress_v2,
)
from iris.application.scientific_ablation_evaluation_ledger import (
    _plan as _score_plan,
)
from iris.application.scientific_ablation_evaluation_v2 import _EvaluationInputs, _finish, _prepare
from iris.application.scientific_ablation_generation_context import (
    IrisScientificAblationGenerationContextV1,
    _ContextSeal,
    _restore_caller,
)
from iris.application.scientific_ablation_generation_context import (
    _entry as _context_entry,
)
from iris.application.scientific_ablation_heldout import (
    IrisScientificAblationHeldoutV1,
    _audit_file,
    _audit_original,
    _TestSeal,
)
from iris.application.scientific_ablation_heldout import (
    _entry as _heldout_entry,
)
from iris.application.scientific_ablation_inference import _digest, _exact_tree
from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_ablation_validation import _complete_row_fields
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_command import ScientificParquetInputV1
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_causal_controls_v1 import (
    DOMAIN,
    IrisScientificAblationCausalControlsV1,
    ScientificAblationCausalControlRowV1,
    ScientificAblationCausalControlsPayloadV1,
    _empty_plan,
    causal_control_report_byte_bound_v1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    CausalDesignPolicy,
    IrisScientificAblationCausalDesignV1,
    ScientificAblationCausalGeometryV1,
    ScientificAblationCausalMemberV1,
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
    build_scientific_ablation_causal_design_v1,
    scientific_ablation_causal_design_bytes_v1,
)
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    IrisScientificAblationCausalObservedBudgetV1,
    build_scientific_ablation_causal_observed_budget_v1,
    scientific_ablation_causal_observed_budget_bytes_v1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    MAX_REPORT_BYTES,
    ScientificAblationCausalReportBytePreflightV1,
    build_scientific_ablation_causal_report_byte_preflight_v1,
    validate_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutInputV1
from iris.training.ablation_controls import IrisAblationControlledTeacherOutputV1
from iris.training.ablation_observation import IrisAblationObservedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.outputs import IrisTeacherForcingInput
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceV1


class IrisScientificAblationCausalControlsError(RuntimeError):
    """Changed source/state or incomplete controls cannot yield a partial report."""


@dataclass(frozen=True, slots=True)
class ScientificAblationCausalInputV1:
    evaluation_path: Path
    expected_evaluation_plan_sha256: str
    expected_evaluation_sha256: str
    capture_path: Path
    expected_plan_sha256: str
    expected_capture_sha256: str
    lifecycle_path: Path
    inference_path: Path
    inputs_root: Path
    training: ScientificParquetInputV1
    validation: ScientificParquetInputV1
    test: ScientificParquetInputV1
    maximum_capture_bytes: int = 512 * 1024**2
    maximum_evaluation_bytes: int = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class ScientificAblationCausalCompleteLimitsV1:
    """Only quotas, not caller-supplied provenance or measurement callbacks."""

    maximum_observation_scalars: int = 2_000_000
    maximum_report_bytes: int = MAX_REPORT_BYTES


def _complete_limits(
    value: ScientificAblationCausalCompleteLimitsV1,
) -> ScientificAblationCausalCompleteLimitsV1:
    if type(value) is not ScientificAblationCausalCompleteLimitsV1:
        raise ValueError("Exact closed complete causal limits required")
    if (
        type(value.maximum_observation_scalars) is not int
        or not 1 <= value.maximum_observation_scalars <= 2_000_000
        or type(value.maximum_report_bytes) is not int
        or not 1 <= value.maximum_report_bytes <= MAX_REPORT_BYTES
    ):
        raise ValueError("Exact bounded complete causal quotas required")
    return replace(value)


def _input(value: ScientificAblationCausalInputV1) -> ScientificAblationCausalInputV1:
    if type(value) is not ScientificAblationCausalInputV1:
        raise ValueError("Exact original source/capture/score inputs required")
    for name in (
        "expected_evaluation_plan_sha256",
        "expected_evaluation_sha256",
        "expected_plan_sha256",
        "expected_capture_sha256",
    ):
        _digest(getattr(value, name))
    for name in (
        "evaluation_path",
        "capture_path",
        "lifecycle_path",
        "inference_path",
        "inputs_root",
    ):
        if not isinstance(getattr(value, name), Path):
            raise ValueError("Explicit local source input paths required")
    for limit in (value.maximum_capture_bytes, value.maximum_evaluation_bytes):
        if type(limit) is not int or not 1 <= limit <= 512 * 1024**2:
            raise ValueError("Exact bounded capture/evaluation storage limits required")
    specs = []
    for spec in (value.training, value.validation, value.test):
        _exact_tree(spec, ScientificParquetInputV1)
        if spec.__pydantic_private__ is not None:
            raise ValueError("Hidden private source input metadata forbidden")
        specs.append(ScientificParquetInputV1.model_validate(spec.model_dump(mode="python")))
    return replace(value, training=specs[0], validation=specs[1], test=specs[2])


def _control_signal(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(_control_signal(child) for child in error.exceptions)
    return not isinstance(error, Exception)


def _fail(primary: BaseException | None, cleanup: list[BaseException]) -> None:
    if primary is not None and _control_signal(primary):
        raise primary from (
            BaseExceptionGroup("Causal control cleanup failed", cleanup) if cleanup else None
        )
    signal = next((e for e in cleanup if _control_signal(e)), None)
    if signal is not None:
        raise signal from primary
    errors = ([] if primary is None else [primary]) + cleanup
    if errors:
        raise IrisScientificAblationCausalControlsError(
            "Source-owned causal controls did not complete"
        ) from (
            errors[0] if len(errors) == 1 else BaseExceptionGroup("Causal control failures", errors)
        )


def _checked(result: IrisScientificAblationCausalControlsV1) -> bytes:
    # Existing inference's annotation guard intentionally does not accept PEP695
    # aliases. Keep it unchanged; the design has its own exact closed-tree guard.
    if type(result) is not IrisScientificAblationCausalControlsV1:
        raise ValueError("Exact source-owned control report root required")
    for value, expected in (
        (result, IrisScientificAblationCausalControlsV1),
        (result.payload, ScientificAblationCausalControlsPayloadV1),
    ):
        if (
            type(value) is not expected
            or set(vars(value)) != set(expected.model_fields)
            or value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
        ):
            raise ValueError("Exact closed source-owned control report required")
    _digest(result.sha256)
    scientific_ablation_causal_design_bytes_v1(result.payload.design)
    for name, field in ScientificAblationCausalControlsPayloadV1.model_fields.items():
        if name != "design":
            _exact_tree(getattr(result.payload, name), field.annotation)
    pending: list[object] = [result]
    while pending:
        item = pending.pop()
        if isinstance(item, BaseModel):
            if item.__pydantic_private__ is not None:
                raise ValueError("Hidden private report metadata forbidden")
            pending.extend(vars(item).values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.values())
    result = IrisScientificAblationCausalControlsV1.model_validate(
        result.model_dump(mode="python", warnings=False)
    )
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > min(
        result.payload.maximum_report_bytes, result.payload.report_byte_preflight_bound
    ):
        raise ValueError("Canonical control report exceeded the preflighted byte limit")
    return content


@dataclass(frozen=True, slots=True)
class _Source:
    owner: IrisScientificAblationHeldoutV1
    seal: _TestSeal
    manifest: IrisScientificAblationHeldoutInputV1
    context: IrisScientificAblationGenerationContextV1
    context_seal: _ContextSeal
    request_guard: object
    request_owner: _AblationRequestOwner
    run: IrisScientificAblationRunV1
    optimizer: IrisScientificAblationOptimizerV1
    model: IrisScientificAblationTeacherV1
    inputs: _EvaluationInputs
    spec: ScientificAblationCausalInputV1
    progress: IrisScientificAblationEvaluationProgressV2
    design: IrisScientificAblationCausalDesignV1
    seed: int
    controlled_graph_sha256: str
    sidecar_sha256: str
    report_byte_bound: int
    maximum_report_bytes: int
    complete_limits: ScientificAblationCausalCompleteLimitsV1 | None = None
    complete_preflight: ScientificAblationCausalReportBytePreflightV1 | None = None

    def identity(self) -> None:
        if (
            self.owner._guard is not self.seal
            or self.seal.context is not self.context
            or self.context._guard is not self.context_seal
            or self.context._run is not self.run
            or self.context._source is not self.context_seal.source
            or self.context._request is not self.context_seal.request
            or self.context_seal.request._guard is not self.request_guard
            or self.context_seal.request._owner is not self.request_owner
            or self.context._validation is not self.context_seal.validation
            or self.run._optimizer is not self.optimizer
            or self.optimizer._model is not self.model
        ):
            raise ValueError("Original sealed causal owners/graph changed")

    def poison(self) -> None:
        # All references were captured AFTER genuine native entry validation.
        # Never poison a foreign optimizer/request substituted into live fields.
        self.owner._state = self.context._state = "poisoned"
        self.run._poisoned = self.optimizer._poisoned = True
        self.request_owner._poison()
        self.context_seal.request._state = "poisoned"

    def read_scores(self) -> IrisScientificAblationEvaluationProgressV2:
        s = self.spec
        return read_scientific_ablation_evaluation_progress_v2(
            s.evaluation_path,
            expected_evaluation_plan_sha256=s.expected_evaluation_plan_sha256,
            expected_plan_sha256=s.expected_plan_sha256,
            expected_capture_sha256=s.expected_capture_sha256,
            maximum_storage_bytes=s.maximum_evaluation_bytes,
        )

    def audit(self, *, measuring: bool = False) -> None:
        self.identity()
        if measuring and (
            self.owner._state != "busy"
            or self.context._state != "busy"
            or self.context_seal.request._state != "busy"
            or not self.run._busy
            or any(module.training for module in self.model.modules())
            or torch.get_default_dtype() is not torch.float32
        ):
            raise ValueError("Original measurement ownership/mode/default dtype changed")
        for row, fingerprint in zip(self.seal.rows, self.seal.fingerprints, strict=True):
            if (
                not _complete_row_fields(row)
                or canonical_json_bytes(row.model_dump(mode="json")) != fingerprint
            ):
                raise ValueError("ANY original private causal row changed, including unselected")
        _audit_file(self.seal)
        _audit_original(self.context)
        _finish(self.inputs)
        if self.read_scores() != self.progress:
            raise ValueError("Entire COMPLETE original score ledger changed")
        if self.complete_limits is None and self.complete_preflight is None:
            return
        if self.complete_limits is None or self.complete_preflight is None:
            raise ValueError("Complete source quotas/reservation must remain paired")
        limits = _complete_limits(self.complete_limits)
        preflight = validate_scientific_ablation_causal_report_byte_preflight_v1(
            self.complete_preflight
        )
        teacher = self.model.teacher
        g = self.design.payload.geometry
        configuration = _json(self.model.base.configuration_json.encode())
        graph = ablation_control_graph_sha256_v1(
            backbone=configuration["backbone_configuration_json"],
            cognitive=configuration["cognitive_configuration_json"],
            workspace=configuration["workspace_configuration_json"],
            ablation=self.model.ablation,
        )
        if (
            preflight.observed_budget.payload.design != self.design
            or preflight.evaluation_plan != self.progress.plan
            or preflight.evaluation != self.progress.evaluation
            or preflight.evaluation_complete != self.progress.complete
            or preflight.run_seed != self.seed
            or self.seed != _json(self.run.configuration_json.encode())["seed"]
            or preflight.ordinary_teacher_graph_sha256
            != self.progress.plan.payload.generation_plan.payload.teacher_graph_sha256
            or preflight.controlled_teacher_graph_sha256 != self.controlled_graph_sha256
            or self.controlled_graph_sha256 != graph
            or preflight.sidecar_sha256 != self.sidecar_sha256
            or self.sidecar_sha256
            != self.context_seal.source.manifest.payload.teacher_sidecar_sha256
            or preflight.maximum_report_bytes != limits.maximum_report_bytes
            or self.maximum_report_bytes != limits.maximum_report_bytes
            or self.report_byte_bound != preflight.report_byte_preflight_bound
            or preflight.observed_budget.payload.maximum_observation_scalars
            != limits.maximum_observation_scalars
            or (g.num_modules, g.num_slots, g.state_dim, g.cycle_count, g.writers_per_cycle)
            != (
                teacher._num_modules,
                teacher._num_slots,
                teacher._state_dim,
                teacher._cycles,
                teacher._winners_per_cycle,
            )
        ):
            raise ValueError("Complete reservation differs from authenticated original source")


@contextmanager
def _source_transaction(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    maximum_report_bytes: int,
    complete_limits: ScientificAblationCausalCompleteLimitsV1 | None = None,
) -> Iterator[_Source]:
    """Shared complete-experiment transaction; only trusted internal stages yield."""
    source = None
    caller = None
    entered = False
    primary: BaseException | None = None
    cleanup: list[BaseException] = []
    caller_dtype = torch.get_default_dtype()
    modes: tuple[tuple[torch.nn.Module, bool], ...] = ()
    preflight = None
    layout_sha = None
    device = torch.device("cpu")
    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        try:
            spec = _input(input)
            if type(policy) not in (
                ScientificAblationWorkspaceCausalPolicyV1,
                ScientificAblationMessageCausalPolicyV1,
            ):
                raise ValueError("Exact distinct source-owned causal policy required")
            _exact_tree(policy, type(policy))
            if policy.__pydantic_private__ is not None:
                raise ValueError("Hidden private source control policy forbidden")
            policy = type(policy).model_validate(policy.model_dump(mode="python"))
            limits = None if complete_limits is None else _complete_limits(complete_limits)
            if (
                type(maximum_report_bytes) is not int
                or not 1
                <= maximum_report_bytes
                <= (64 * 1024**2 if limits is None else MAX_REPORT_BYTES)
                or (limits is not None and maximum_report_bytes != limits.maximum_report_bytes)
            ):
                raise ValueError("Explicit bounded causal report byte limit required")
            seal, manifest = _heldout_entry(heldout)
            context = seal.context
            context_seal = _context_entry(context)
            run, optimizer = context_seal.run, context_seal.run._optimizer
            model, teacher = optimizer._model, optimizer._model.teacher
            device = run._backbone.device
            caller = capture_runtime_state(device)
            modes = tuple((module, module.training) for module in model.modules())
            progress = read_scientific_ablation_evaluation_progress_v2(
                spec.evaluation_path,
                expected_evaluation_plan_sha256=spec.expected_evaluation_plan_sha256,
                expected_plan_sha256=spec.expected_plan_sha256,
                expected_capture_sha256=spec.expected_capture_sha256,
                maximum_storage_bytes=spec.maximum_evaluation_bytes,
            )
            if (
                progress.complete is None
                or progress.evaluation is None
                or progress.evaluation.sha256 != spec.expected_evaluation_sha256
            ):
                raise ValueError(
                    "Independently trusted COMPLETE original score ledger/report required"
                )
            inputs = _prepare(
                capture_path=spec.capture_path,
                expected_plan_sha256=spec.expected_plan_sha256,
                expected_capture_sha256=spec.expected_capture_sha256,
                lifecycle_path=spec.lifecycle_path,
                inference_path=spec.inference_path,
                inputs_root=spec.inputs_root,
                training=spec.training,
                validation=spec.validation,
                test=spec.test,
                python_policy=progress.plan.payload.python_policy,
                maximum_capture_bytes=spec.maximum_capture_bytes,
            )
            if (
                _score_plan(inputs) != progress.plan
                or inputs.capture.plan.payload.input != manifest
                or inputs.capture.plan.payload.source != context_seal.source.manifest
                or inputs.rows != seal.rows
                or tuple(r.target_token_count for r in progress.evaluation.payload.rows)
                != seal.target_counts
            ):
                raise ValueError("Original native best/test/capture/complete score source differs")
            _audit_file(seal)
            _audit_original(context)
            # Membership derives only from the fully owned ORIGINAL Parquet rows.
            members = tuple(
                ScientificAblationCausalMemberV1(
                    row_index=index,
                    example_id=row.id,
                    example_sha256=domain_sha256(
                        "iris/scientific-ablation-example/v1", seal.fingerprints[index]
                    ),
                    prompt_sha256=sha256(row.prompt.encode()).hexdigest(),
                    cognitive_prompt_sha256=domain_sha256(
                        "iris/cognitive-ablation-prompt/v1", row.prompt.encode()
                    ),
                    task_type=row.task_type,
                    difficulty=row.difficulty,
                )
                for index, row in enumerate(seal.rows)
            )
            design = build_scientific_ablation_causal_design_v1(
                members,
                purpose=manifest.payload.purpose,
                ablation=model.ablation,
                policy=policy,
                geometry=ScientificAblationCausalGeometryV1(
                    num_modules=teacher._num_modules,
                    num_slots=teacher._num_slots,
                    state_dim=teacher._state_dim,
                    cycle_count=teacher._cycles,
                    writers_per_cycle=teacher._winners_per_cycle,
                ),
            )
            bound = (
                causal_control_report_byte_bound_v1(
                    design, progress.plan, progress.evaluation, progress.complete
                )
                if limits is None
                else 0
            )
            if bound > maximum_report_bytes:
                raise ValueError("Control report byte quota exceeded BEFORE any teacher pass")
            seed = _json(run.configuration_json.encode())["seed"]
            if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
                raise ValueError("Original native causal run seed invalid")
            # Native owners have already authenticated the original complete
            # graph/best snapshot. Reading evaluation_provenance before entering
            # eval would violate their preserved original training-mode boundary.
            configuration = _json(model.base.configuration_json.encode())
            graph = ablation_control_graph_sha256_v1(
                backbone=configuration["backbone_configuration_json"],
                cognitive=configuration["cognitive_configuration_json"],
                workspace=configuration["workspace_configuration_json"],
                ablation=model.ablation,
            )
            source = _Source(
                heldout,
                seal,
                manifest,
                context,
                context_seal,
                context_seal.request._guard,
                context_seal.request._owner,
                run,
                optimizer,
                model,
                inputs,
                spec,
                progress,
                design,
                seed,
                graph,
                context_seal.source.manifest.payload.teacher_sidecar_sha256,
                bound,
                maximum_report_bytes,
            )
            source.audit()
            if limits is not None:
                # Capture ORIGINAL cleanup/poison references before either new
                # quota check. Ordinary preflight denial leaves READY untouched;
                # uncertain caller restoration still poisons only these owners.
                budget = build_scientific_ablation_causal_observed_budget_v1(
                    design, maximum_observation_scalars=limits.maximum_observation_scalars
                )
                preflight = build_scientific_ablation_causal_report_byte_preflight_v1(
                    budget,
                    evaluation_plan=progress.plan,
                    evaluation=progress.evaluation,
                    evaluation_complete=progress.complete,
                    run_seed=source.seed,
                    ordinary_teacher_graph_sha256=(
                        progress.plan.payload.generation_plan.payload.teacher_graph_sha256
                    ),
                    controlled_teacher_graph_sha256=source.controlled_graph_sha256,
                    sidecar_sha256=source.sidecar_sha256,
                    maximum_report_bytes=limits.maximum_report_bytes,
                )
                source = replace(
                    source,
                    report_byte_bound=preflight.report_byte_preflight_bound,
                    complete_limits=limits,
                    complete_preflight=preflight,
                )
                layout_sha = preflight.layout_sha256
                source.audit()
            heldout._state = context._state = "busy"
            context_seal.request._state = "busy"
            run._busy = entered = True
            torch.set_default_dtype(torch.float32)
            model.eval()
            with torch.device(device):
                yield source
            source.audit(measuring=True)
            if (
                source.complete_limits is not limits
                or source.complete_preflight is not preflight
                or (preflight is not None and preflight.layout_sha256 != layout_sha)
            ):
                raise ValueError("Original complete source reservation identity changed")
        except BaseException as error:
            primary = error
        finally:
            if entered:
                try:
                    for module, mode in modes:
                        module.training = mode
                    if any(module.training is not mode for module, mode in modes):
                        raise ValueError("Original native model modes not restored")
                except BaseException as error:
                    cleanup.append(error)
            if caller is not None:
                try:
                    _restore_caller(caller, caller_dtype, device)
                except BaseException as error:
                    cleanup.append(error)
            if entered and source is not None:
                source.run._busy = False
                source.owner._state = source.context._state = "ready"
                source.context_seal.request._state = "ready"
                try:
                    # Reject alias drift BEFORE old boundary helpers can inspect
                    # or poison any substituted optimizer/request ownership.
                    source.identity()
                    _heldout_entry(source.owner)
                    _context_entry(source.context)
                except BaseException as error:
                    cleanup.append(error)
            if source is not None and (entered or cleanup) and (primary is not None or cleanup):
                try:
                    source.poison()
                except BaseException as error:
                    cleanup.append(error)
    _fail(primary, cleanup)


@dataclass(frozen=True, slots=True)
class _ControlMeasurement:
    row: ScientificAblationCausalControlRowV1
    prepared: IrisTeacherForcingInput | None
    observation: IrisAblationObservedTeacherOutputV1 | None
    retention_seal: _ObservedPassSeal | None = None


def _control_rows(
    source: _Source, *, observed_budget: IrisScientificAblationCausalObservedBudgetV1 | None = None
) -> tuple[_ControlMeasurement, ...]:
    """ONE ordinary + ONE empty pass per original selected row; no callback.

    The observed variant is an internal stage of the future complete executor,
    not a second baseline or a complete effect/publication report. A complete
    source has already reserved its whole numeric/JSON volume BEFORE BUSY.
    """
    if source.complete_preflight is not None and observed_budget is None:
        raise ValueError("Complete source requires its preflighted observed stage")
    maximum_observation_scalars = None
    if observed_budget is not None:
        # Own/revalidate the entire declared budget BEFORE either forward. Never
        # accept a copied/rehashed/subclass DTO in place of the original design.
        observed_budget = IrisScientificAblationCausalObservedBudgetV1.model_validate_json(
            scientific_ablation_causal_observed_budget_bytes_v1(observed_budget)
        )
        if observed_budget.payload.design != source.design:
            raise ValueError("Observed budget must bind the complete original source design")
        source.audit(measuring=True)
        if (
            source.complete_preflight is not None
            and observed_budget != source.complete_preflight.observed_budget
        ):
            raise ValueError("Observed stage differs from the original complete preflight")
        maximum_observation_scalars = observed_budget.payload.maximum_observation_scalars
    design, teacher = source.design.payload, source.model.teacher
    plan = _empty_plan(source.design)
    plan_sha = ablation_control_plan_sha256_v1(plan)
    evaluation = source.progress.evaluation
    assert evaluation is not None
    rows = []
    for index in design.selected_row_indices:
        source.identity()
        row = source.seal.rows[index]
        if canonical_json_bytes(row.model_dump(mode="json")) != source.seal.fingerprints[index]:
            raise ValueError("Original selected row changed before control forward")
        options = WorkspaceTraceOptionsV1(
            mode=design.policy.ordinary_trace_mode,
            example_id=row.id,
            checkpoint_id=source.manifest.payload.selected_checkpoint_sha256,
            seed=source.seed,
            maximum_scalars=design.policy.maximum_trace_scalars,
        )
        restore_runtime_state(source.run._runtime)
        ordinary = _teacher(source.context, row, source.seal.target_counts[index], options)
        score = evaluation.payload.rows[index]
        if (
            ordinary.invocation_sha256 != score.teacher_invocation_sha256
            or ordinary.teacher.target_sha256 != score.teacher_target_sha256
            or float(ordinary.losses.language_model) != score.target_nll_sum
        ):
            raise ValueError("Original saved selected-best ordinary EOS NLL was not reproduced")
        prepared = prepare_teacher_forcing_input(
            backbone=source.run._backbone,
            prompt=row.prompt,
            target_text=row.expected_response,
            terminal_token_id=source.run._terminal_token_id,
        )
        target = teacher_target_sha256_v1(prepared.target_input_ids, prepared.target_attention_mask)
        if (
            target != score.teacher_target_sha256
            or int(prepared.target_attention_mask.sum()) != score.target_token_count
        ):
            raise ValueError("Original controlled target/EOS inventory differs")
        restore_runtime_state(source.run._runtime)
        full_options = options.model_copy(update={"mode": "full"})
        observed = None
        retention_seal = None
        if maximum_observation_scalars is None:
            controlled = teacher.evaluation_forward_ablation_controlled(
                prepared, plan=plan, diagnostics=full_options
            )
        else:
            observed = teacher.evaluation_forward_ablation_observed(
                prepared,
                plan=plan,
                diagnostics=full_options,
                maximum_observation_scalars=maximum_observation_scalars,
            )
            if type(observed) is not IrisAblationObservedTeacherOutputV1:
                raise ValueError("Distinct actual native observed EMPTY receipt required")
            retention_seal = _seal_observed(observed, prepared)
            controlled = observed.controlled
        if type(controlled) is not IrisAblationControlledTeacherOutputV1:
            raise ValueError("Distinct actual native mode/control teacher receipt required")
        controlled.__post_init__()
        normal = ordinary.teacher.traced.teacher
        trace = normal.diagnostic_trace
        if type(trace) is not WorkspaceTraceV1:
            raise ValueError("Actual ordinary causal control trace required")
        if (
            not torch.equal(normal.loss, controlled.loss)
            or (controlled.target_sha256, controlled.supervised_token_count)
            != (target, score.target_token_count)
            or controlled.plan_sha256 != plan_sha
            or controlled.plan != plan
            or controlled.teacher_graph_sha256 != source.controlled_graph_sha256
            or controlled.sidecar_state_sha256 != source.sidecar_sha256
            or controlled.prompt_sha256 != design.members[index].cognitive_prompt_sha256
            or controlled.diagnostic_trace.options != full_options
        ):
            raise ValueError(
                "Native empty control differs from original best/mode/ordinary teacher"
            )
        for left, right in zip(normal.workspace_trace, controlled.workspace_trace, strict=True):
            if any(
                not torch.equal(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
            ):
                raise ValueError("Empty combined control changed an actual workspace record")
        for state, cycle in zip(
            controlled.workspace_trace, controlled.diagnostic_trace.cycles, strict=True
        ):
            if (
                cycle.full is None
                or tuple(tuple(vector) for vector in state.slots[0].tolist())
                != cycle.full.workspace_slots
            ):
                raise ValueError("Controlled full vectors differ from actual workspace tensors")
            for tensor_name, trace_name in (
                ("age", "slot_ages"),
                ("ttl", "slot_ttl"),
                ("occupancy", "slot_occupancy"),
                ("source_module", "slot_sources"),
                ("write_count", "slot_write_counts"),
            ):
                if tuple(getattr(state, tensor_name)[0].tolist()) != getattr(cycle, trace_name):
                    raise ValueError("Controlled slot metadata differs from actual workspace")
        rows.append(
            _ControlMeasurement(
                row=ScientificAblationCausalControlRowV1(
                    row_index=index,
                    capture_row_sha256=score.capture_row_sha256,
                    evaluation_row_sha256=source.progress.rows[index].sha256,
                    evaluation_ack_sha256=source.progress.acknowledgements[index].sha256,
                    ordinary_teacher_invocation_sha256=ordinary.invocation_sha256,
                    target_sha256=target,
                    target_token_count=score.target_token_count,
                    target_nll_sum=score.target_nll_sum,
                    ordinary_mean_target_nll=float(normal.loss),
                    controlled_mean_target_nll=float(controlled.loss),
                    ordinary_trace=trace,
                    controlled_trace=controlled.diagnostic_trace,
                    controlled_invocation_sha256=controlled.invocation_sha256,
                    messages=controlled.messages,
                ),
                prepared=None if observed is None else prepared,
                observation=observed,
                retention_seal=retention_seal,
            )
        )
    return tuple(rows)


def _controls(source: _Source) -> tuple[ScientificAblationCausalControlRowV1, ...]:
    """Preserve the existing public V1 report and its original two-pass stage."""
    return tuple(result.row for result in _control_rows(source))


def _observed_controls(
    source: _Source, *, budget: IrisScientificAblationCausalObservedBudgetV1
) -> tuple[_ControlMeasurement, ...]:
    """Trusted full-executor stage: substitute observed EMPTY, never repeat V1.

    Private prepared inputs and ALL7 observations stay inside the original BUSY
    transaction for subsequent scheduled record/message trials. No report is
    published by this helper; all late transaction audits still have to succeed.
    """
    result = _control_rows(source, observed_budget=budget)
    source.audit(measuring=True)
    for measurement in result:
        prepared, observed, row = (
            measurement.prepared,
            measurement.observation,
            measurement.row,
        )
        if type(prepared) is not IrisTeacherForcingInput or type(observed) is not (
            IrisAblationObservedTeacherOutputV1
        ):
            raise ValueError("Every selected original row must retain its observed EMPTY pass")
        # Earlier rows remain mutable tensor-bearing values while later rows
        # execute. Revalidate the WHOLE retained stage before subsequent trials.
        retention_seal = measurement.retention_seal
        if (
            type(retention_seal) is not _ObservedPassSeal
            or retention_seal.output is not observed
            or retention_seal.prepared is not prepared
        ):
            raise ValueError("Original immediate observed EMPTY retention seal required")
        retention_seal.audit()
        if (
            prepared.prompt != source.seal.rows[row.row_index].prompt
            or teacher_target_sha256_v1(prepared.target_input_ids, prepared.target_attention_mask)
            != row.target_sha256
            or int(prepared.target_attention_mask.sum()) != row.target_token_count
            or observed.controlled.invocation_sha256 != row.controlled_invocation_sha256
            or observed.controlled.diagnostic_trace != row.controlled_trace
            or observed.controlled.messages != row.messages
        ):
            raise ValueError("Retained original observed EMPTY stage changed during measurement")
    return result


def measure_scientific_ablation_causal_controls_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    maximum_report_bytes: int = 64 * 1024**2,
) -> IrisScientificAblationCausalControlsV1:
    """Authenticate ALL original rows/ledgers and measure selected empty controls.

    This is NOT a writer-effect report; its actual two passes per selected row
    are distinguished from the WHOLE future experiment budget in the design.
    """
    with _source_transaction(
        heldout, input=input, policy=policy, maximum_report_bytes=maximum_report_bytes
    ) as source:
        rows = _controls(source)
        assert source.progress.evaluation is not None and source.progress.complete is not None
        b = source.design.payload.budget
        payload = ScientificAblationCausalControlsPayloadV1(
            design=source.design,
            evaluation_plan=source.progress.plan,
            evaluation=source.progress.evaluation,
            evaluation_complete=source.progress.complete,
            run_seed=source.seed,
            ordinary_teacher_graph_sha256=source.run._teacher_graph_identity,
            controlled_teacher_graph_sha256=source.controlled_graph_sha256,
            sidecar_sha256=source.sidecar_sha256,
            empty_plan_sha256=ablation_control_plan_sha256_v1(_empty_plan(source.design)),
            actual_teacher_executions=2 * len(rows),
            actual_trace_proposal_scalar_bound=len(rows)
            * (b.ordinary_trace_scalars + b.full_trace_scalars + b.message_scalars_per_collection),
            maximum_report_bytes=maximum_report_bytes,
            report_byte_preflight_bound=source.report_byte_bound,
            rows=rows,
        )
        result = IrisScientificAblationCausalControlsV1(
            sha256=domain_sha256(DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        _checked(result)
    # No report escapes if late source/score/mode/state or restoration failed.
    return result
