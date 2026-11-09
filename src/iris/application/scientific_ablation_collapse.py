"""Measure every native-best ablation row, then retain only compact allocations.

No caller-supplied diagnostic DTO, baseline cast, subset, regeneration, sandbox
execution, publication or model constructor. Legacy delivery retains only count
witnesses; explicit completed delivery also retains the SAME native diagnostic
pass, with no repeated teacher execution. Both paths preserve complete budgets.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from iris.application.scientific_ablation_diagnostics import (
    _checked as _checked_diagnostics,
)
from iris.application.scientific_ablation_diagnostics import (
    measure_scientific_ablation_diagnostics_v1,
)
from iris.application.scientific_ablation_evaluation_ledger import (
    _plan as _score_plan,
)
from iris.application.scientific_ablation_evaluation_ledger import (
    read_scientific_ablation_evaluation_progress_v2,
)
from iris.application.scientific_ablation_evaluation_v2 import _finish, _prepare
from iris.application.scientific_ablation_generation_context import _poison as _poison_context
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_heldout import _entry as _heldout_entry
from iris.application.scientific_ablation_inference import _bounded_metadata, _digest, _exact_tree
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_command import ScientificParquetInputV1
from iris.artifacts.scientific_ablation_collapse_v1 import (
    IrisScientificAblationCollapseV1,
    ScientificAblationCollapseMemberV1,
    ScientificAblationCollapsePayloadV1,
    ScientificAblationCollapsePolicyV1,
    ScientificAblationCollapseRowV1,
    scientific_ablation_collapse_budget_v1,
    scientific_ablation_collapse_groups_v1,
    scientific_ablation_collapse_witness_v1,
)
from iris.artifacts.scientific_ablation_diagnostics_v1 import (
    IrisScientificAblationDiagnosticsV1,
    ScientificAblationDiagnosticPolicyV1,
)
from iris.artifacts.scientific_collapse_v1 import scientific_collapse_criterion_v1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state

_DOMAIN = "iris/scientific-ablation-collapse/v1"
_POLICY_DOMAIN = "iris/scientific-ablation-collapse-policy/v1"
_MEMBERS_DOMAIN = "iris/scientific-ablation-collapse-members/v1"
_MAX_REPORT = 64 * 1024**2


class IrisScientificAblationCollapseError(RuntimeError):
    """Incomplete/changed complete-test evidence cannot become a collapse report."""


@dataclass(frozen=True, slots=True)
class ScientificAblationCollapseCompletedV1:
    """Detached DATA from one cleaned native phase, NOT a native execution token.

    Construction of this value never authorizes a measurement or source restore.
    A compound owner must capture its own original call and independently bind
    these reports; stored DATA recovery is not restored native-owner authority.
    """

    diagnostics: IrisScientificAblationDiagnosticsV1
    collapse: IrisScientificAblationCollapseV1


def _fail(primary: BaseException | None, cleanup: list[BaseException]) -> None:
    if primary is not None and _has_signal(primary):
        raise primary from (
            BaseExceptionGroup("Ablation collapse cleanup failed", cleanup) if cleanup else None
        )
    signal = next((error for error in cleanup if _has_signal(error)), None)
    if signal is not None:
        raise signal from primary
    failures = ([] if primary is None else [primary]) + cleanup
    if failures:
        raise IrisScientificAblationCollapseError("Complete ablation collapse failed") from (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup("Ablation collapse failures", failures)
        )


def _checked(result: IrisScientificAblationCollapseV1) -> bytes:
    _exact_tree(result, IrisScientificAblationCollapseV1)
    _bounded_metadata(result, maximum=_MAX_REPORT)
    owned = IrisScientificAblationCollapseV1.model_validate(
        result.model_dump(mode="python", warnings=False)
    )
    if owned.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("Distinct compact ablation checksum differs")
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_REPORT:
        raise ValueError("Compact ablation report byte limit exceeded")
    return content


def _detach_completed(
    ordinary: IrisScientificAblationDiagnosticsV1,
    compact: IrisScientificAblationCollapseV1,
) -> ScientificAblationCollapseCompletedV1:
    """Validate both exact full reports and detach before native cleanup releases ownership."""
    diagnostic_bytes, collapse_bytes = _checked_diagnostics(ordinary), _checked(compact)
    diagnostics = IrisScientificAblationDiagnosticsV1.model_validate_json(diagnostic_bytes)
    collapse = IrisScientificAblationCollapseV1.model_validate_json(collapse_bytes)
    d, c = diagnostics.payload, collapse.payload
    if (
        c.ordinary_diagnostics_sha256 != diagnostics.sha256
        or d.policy.trace_mode != "metadata"
        or d.evaluation_plan != c.evaluation_plan
        or d.evaluation != c.evaluation
        or d.evaluation_complete != c.evaluation_complete
        or d.run_seed != c.run_seed
        or d.teacher_graph_sha256 != c.teacher_graph_sha256
        or d.teacher_executions != c.teacher_executions
        or d.trace_scalar_bound != c.trace_scalar_bound
        or len(d.rows) != len(c.rows)
    ):
        raise ValueError("Original ordinary/collapse completed phase bindings differ")
    source = c.evaluation.payload.plan.payload.source.payload
    for diagnostic, row in zip(d.rows, c.rows, strict=True):
        if any(
            getattr(diagnostic, name) != getattr(row, name)
            for name in (
                "row_index",
                "capture_row_sha256",
                "evaluation_row_sha256",
                "evaluation_ack_sha256",
                "teacher_invocation_sha256",
                "teacher_target_sha256",
                "target_token_count",
                "target_nll_sum",
            )
        ) or (
            row.metadata_trace_sha256,
            row.context,
            row.cycles,
        ) != scientific_ablation_collapse_witness_v1(
            diagnostic.trace, expected_ablation_policy_sha256=source.ablation_policy_sha256
        ):
            raise ValueError("Completed count witness differs from its original actual diagnostic")
    return ScientificAblationCollapseCompletedV1(diagnostics, collapse)


def _measure_collapse(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    retain_diagnostics: bool,
    evaluation_path: Path,
    expected_evaluation_plan_sha256: str,
    expected_evaluation_sha256: str,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    policy: ScientificAblationCollapsePolicyV1,
    maximum_capture_bytes: int = 512 * 1024**2,
    maximum_evaluation_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationCollapseV1 | ScientificAblationCollapseCompletedV1:
    """Real ALL-row metadata forwards, original difficulty, twenty pooled groups.

    All four independent trusted hashes are mandatory. The ordinary producer
    owns the forward phase; a separately exclusive projection phase re-audits
    original source/corpus/capture/score inputs before returning the count report.
    """
    try:
        if type(retain_diagnostics) is not bool:
            raise ValueError("Exact internal retained-data selection required")
        for anchor in (
            expected_evaluation_plan_sha256,
            expected_evaluation_sha256,
            expected_plan_sha256,
            expected_capture_sha256,
        ):
            _digest(anchor)
        _exact_tree(policy, ScientificAblationCollapsePolicyV1)
        policy = ScientificAblationCollapsePolicyV1.model_validate(policy.model_dump(mode="python"))
        seal, manifest = _heldout_entry(heldout)
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error
    context = seal.context
    run, model = context._run, context._run._optimizer._model
    caller_dtype = torch.get_default_dtype()
    modes = tuple((module, module.training) for module in model.modules())
    caller = None
    entered = measured = False
    primary: BaseException | None = None
    cleanup: list[BaseException] = []
    result = None
    completed: ScientificAblationCollapseCompletedV1 | None = None
    with torch.inference_mode(False), torch.enable_grad(), torch.device(run._backbone.device):
        try:
            caller = capture_runtime_state(run._backbone.device)
            teacher = model.teacher
            geometry = (
                teacher._num_modules,
                teacher._num_slots,
                teacher._state_dim,
                teacher._cycles,
            )
            writers = teacher._winners_per_cycle
            budget = scientific_ablation_collapse_budget_v1(
                policy,
                examples=len(seal.rows),
                modules=geometry[0],
                slots=geometry[1],
                width=geometry[2],
                cycles=geometry[3],
                writers=writers,
            )
            ordinary = measure_scientific_ablation_diagnostics_v1(
                heldout,
                evaluation_path=evaluation_path,
                expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
                expected_evaluation_sha256=expected_evaluation_sha256,
                capture_path=capture_path,
                expected_plan_sha256=expected_plan_sha256,
                expected_capture_sha256=expected_capture_sha256,
                lifecycle_path=lifecycle_path,
                inference_path=inference_path,
                inputs_root=inputs_root,
                training=training,
                validation=validation,
                test=test,
                policy=ScientificAblationDiagnosticPolicyV1(
                    trace_mode="metadata",
                    maximum_teacher_executions=policy.maximum_teacher_executions,
                    maximum_trace_scalars=policy.maximum_trace_scalars,
                    maximum_total_trace_scalars=policy.maximum_total_trace_scalars,
                ),
                maximum_capture_bytes=maximum_capture_bytes,
                maximum_evaluation_bytes=maximum_evaluation_bytes,
            )
            measured = True
            _checked_diagnostics(ordinary)
            if _heldout_entry(heldout) != (seal, manifest):
                raise ValueError("Original complete held-out owner changed after forwards")
            progress = read_scientific_ablation_evaluation_progress_v2(
                evaluation_path,
                expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
                expected_plan_sha256=expected_plan_sha256,
                expected_capture_sha256=expected_capture_sha256,
                maximum_storage_bytes=maximum_evaluation_bytes,
            )
            p = ordinary.payload
            if (
                progress.plan != p.evaluation_plan
                or progress.evaluation != p.evaluation
                or progress.complete != p.evaluation_complete
                or p.evaluation.sha256 != expected_evaluation_sha256
                or (p.teacher_executions, p.trace_scalar_bound) != budget
                or heldout._guard is not seal
                or context._run is not run
                or run._optimizer._model is not model
            ):
                raise ValueError("Original measurement/score/native ownership changed")
            heldout._state = context._state = "busy"
            run._busy = entered = True
            inputs = _prepare(
                capture_path=capture_path,
                expected_plan_sha256=expected_plan_sha256,
                expected_capture_sha256=expected_capture_sha256,
                lifecycle_path=lifecycle_path,
                inference_path=inference_path,
                inputs_root=inputs_root,
                training=training,
                validation=validation,
                test=test,
                python_policy=progress.plan.payload.python_policy,
                maximum_capture_bytes=maximum_capture_bytes,
            )
            if (
                _score_plan(inputs) != progress.plan
                or inputs.capture.plan.payload.input != manifest
                or inputs.capture.plan.payload.source != context._source.manifest
                or inputs.rows != seal.rows
            ):
                raise ValueError("Original physical source/capture/test inventory differs")
            members, witnesses = [], []
            for index, (original, expected, diagnostic) in enumerate(
                zip(inputs.rows, inputs.capture.plan.payload.rows, p.rows, strict=True)
            ):
                if (
                    canonical_json_bytes(original.model_dump(mode="json"))
                    != seal.fingerprints[index]
                ):
                    raise ValueError("Original full test metadata changed during projection")
                members.append(
                    ScientificAblationCollapseMemberV1(
                        row_index=index,
                        example_id=original.id,
                        example_sha256=expected.example_sha256,
                        prompt_sha256=expected.prompt_sha256,
                        cognitive_prompt_sha256=expected.cognitive_prompt_sha256,
                        task_type=original.task_type,
                        difficulty=original.difficulty,
                    )
                )
                trace_sha, trace_context, cycles = scientific_ablation_collapse_witness_v1(
                    diagnostic.trace,
                    expected_ablation_policy_sha256=context._source.manifest.payload.ablation_policy_sha256,
                )
                witnesses.append(
                    ScientificAblationCollapseRowV1(
                        row_index=index,
                        capture_row_sha256=diagnostic.capture_row_sha256,
                        evaluation_row_sha256=diagnostic.evaluation_row_sha256,
                        evaluation_ack_sha256=diagnostic.evaluation_ack_sha256,
                        teacher_invocation_sha256=diagnostic.teacher_invocation_sha256,
                        teacher_target_sha256=diagnostic.teacher_target_sha256,
                        target_token_count=diagnostic.target_token_count,
                        target_nll_sum=diagnostic.target_nll_sum,
                        metadata_trace_sha256=trace_sha,
                        context=trace_context,
                        cycles=cycles,
                    )
                )
            groups = scientific_ablation_collapse_groups_v1(
                tuple(members), tuple(witnesses), modules=geometry[0], slots=geometry[1]
            )
            summary = groups[0].summary
            assert summary is not None
            payload = ScientificAblationCollapsePayloadV1(
                evaluation_plan=p.evaluation_plan,
                evaluation=p.evaluation,
                evaluation_complete=p.evaluation_complete,
                policy=policy,
                policy_sha256=domain_sha256(
                    _POLICY_DOMAIN, canonical_json_bytes(policy.model_dump(mode="json"))
                ),
                membership_sha256=domain_sha256(
                    _MEMBERS_DOMAIN,
                    canonical_json_bytes(
                        tuple(member.model_dump(mode="json") for member in members)
                    ),
                ),
                ordinary_diagnostics_sha256=ordinary.sha256,
                run_seed=p.run_seed,
                teacher_graph_sha256=p.teacher_graph_sha256,
                num_modules=geometry[0],
                num_slots=geometry[1],
                state_dim=geometry[2],
                cycle_count=geometry[3],
                writers_per_cycle=writers,
                members=tuple(members),
                rows=tuple(witnesses),
                teacher_executions=budget[0],
                trace_scalar_bound=budget[1],
                groups=groups,
                source_20_4=scientific_collapse_criterion_v1(
                    purpose=manifest.payload.purpose, summary=summary
                ),
            )
            result = IrisScientificAblationCollapseV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            _checked(result)
            if retain_diagnostics:
                # ONE original diagnostic pass already occurred above. No new
                # diagnostic/teacher/generation/scoring invocation here. Legacy
                # compact-only calls do not pay for retaining/copying this DATA.
                completed = _detach_completed(ordinary, result)
        except BaseException as error:
            primary = error
        finally:
            if measured:
                try:
                    for module, mode in modes:
                        module.training = mode
                    if any(module.training is not mode for module, mode in modes):
                        raise ValueError("Native best module modes not restored")
                except BaseException as error:
                    cleanup.append(error)
            if caller is not None:
                try:
                    _restore_caller(caller, caller_dtype, run._backbone.device)
                except BaseException as error:
                    cleanup.append(error)
            if entered and primary is None and not cleanup:
                try:
                    # The final deep audit must follow ALL projection, retained
                    # DATA detachment and caller restoration. Still hold busy
                    # ownership; no additional model/scoring execution occurs.
                    _finish(inputs)
                    if read_scientific_ablation_evaluation_progress_v2(
                        evaluation_path,
                        expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
                        expected_plan_sha256=expected_plan_sha256,
                        expected_capture_sha256=expected_capture_sha256,
                        maximum_storage_bytes=maximum_evaluation_bytes,
                    ) != progress or any(
                        canonical_json_bytes(row.model_dump(mode="json")) != fingerprint
                        for row, fingerprint in zip(seal.rows, seal.fingerprints, strict=True)
                    ):
                        raise ValueError("Original test/score ledger changed before return")
                except BaseException as error:
                    cleanup.append(error)
            if entered:
                run._busy = False
                heldout._state = context._state = "ready"
                try:
                    if _heldout_entry(heldout) != (seal, manifest):
                        raise ValueError("Native-best ownership changed during projection cleanup")
                except BaseException as error:
                    cleanup.append(error)
            if (measured or cleanup) and (primary is not None or cleanup):
                heldout._state = "poisoned"
                _poison_context(context)
    _fail(primary, cleanup)
    if retain_diagnostics:
        if completed is None:
            raise IrisScientificAblationCollapseError("No complete retained native diagnostic DATA")
        return completed
    if result is None:
        raise IrisScientificAblationCollapseError("No complete compact ablation report")
    return result


def measure_scientific_ablation_collapse_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    evaluation_path: Path,
    expected_evaluation_plan_sha256: str,
    expected_evaluation_sha256: str,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    policy: ScientificAblationCollapsePolicyV1,
    maximum_capture_bytes: int = 512 * 1024**2,
    maximum_evaluation_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationCollapseV1:
    """Legacy exact compact-only API; no retained full-report copy or second pass."""
    result = _measure_collapse(
        heldout,
        retain_diagnostics=False,
        evaluation_path=evaluation_path,
        expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
        expected_evaluation_sha256=expected_evaluation_sha256,
        capture_path=capture_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_capture_sha256=expected_capture_sha256,
        lifecycle_path=lifecycle_path,
        inference_path=inference_path,
        inputs_root=inputs_root,
        training=training,
        validation=validation,
        test=test,
        policy=policy,
        maximum_capture_bytes=maximum_capture_bytes,
        maximum_evaluation_bytes=maximum_evaluation_bytes,
    )
    if type(result) is not IrisScientificAblationCollapseV1:
        raise IrisScientificAblationCollapseError("Unexpected compact-only phase result")
    return result


def measure_scientific_ablation_collapse_completed_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    evaluation_path: Path,
    expected_evaluation_plan_sha256: str,
    expected_evaluation_sha256: str,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    policy: ScientificAblationCollapsePolicyV1,
    maximum_capture_bytes: int = 512 * 1024**2,
    maximum_evaluation_bytes: int = 512 * 1024**2,
) -> ScientificAblationCollapseCompletedV1:
    """Return both bounded reports from ONE owned, fully cleaned native phase.

    No supplied diagnostic DTO or measurement callbacks are accepted. The
    retained metadata report and compact count report each retain their existing
    64MiB limit; no fields/rows are clipped. This DATA is not a durable native
    execution token, publication, uncertain native resume or scientific approval.
    """
    result = _measure_collapse(
        heldout,
        retain_diagnostics=True,
        evaluation_path=evaluation_path,
        expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
        expected_evaluation_sha256=expected_evaluation_sha256,
        capture_path=capture_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_capture_sha256=expected_capture_sha256,
        lifecycle_path=lifecycle_path,
        inference_path=inference_path,
        inputs_root=inputs_root,
        training=training,
        validation=validation,
        test=test,
        policy=policy,
        maximum_capture_bytes=maximum_capture_bytes,
        maximum_evaluation_bytes=maximum_evaluation_bytes,
    )
    if type(result) is not ScientificAblationCollapseCompletedV1:
        raise IrisScientificAblationCollapseError("Unexpected completed native phase result")
    return result


__all__ = [
    "IrisScientificAblationCollapseError",
    "ScientificAblationCollapseCompletedV1",
    "measure_scientific_ablation_collapse_completed_v1",
    "measure_scientific_ablation_collapse_v1",
]
