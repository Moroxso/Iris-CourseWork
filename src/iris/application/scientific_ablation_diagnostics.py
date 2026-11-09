"""Complete ordinary native-best diagnostics, not causal/message or launch evidence.

No construction, generation, rescoring, candidate execution or publication here.
Runtime ownership is exclusive like the native held-out context, not thread-safe.
"""

from pathlib import Path
from typing import Literal

import torch

from iris.ablations import cognitive_ablation_policy_sha256_v1
from iris.application.scientific_ablation_evaluation_ledger import (
    IrisScientificAblationEvaluationProgressV2,
    read_scientific_ablation_evaluation_progress_v2,
)
from iris.application.scientific_ablation_evaluation_ledger import (
    _plan as _score_plan,
)
from iris.application.scientific_ablation_evaluation_v2 import _finish, _prepare
from iris.application.scientific_ablation_generation_context import (
    IrisScientificAblationGenerationContextV1,
    _restore_caller,
)
from iris.application.scientific_ablation_generation_context import (
    _entry as _context_entry,
)
from iris.application.scientific_ablation_generation_context import (
    _poison as _poison_context,
)
from iris.application.scientific_ablation_heldout import (
    IrisScientificAblationHeldoutV1,
    _audit_file,
    _audit_original,
)
from iris.application.scientific_ablation_heldout import (
    _entry as _heldout_entry,
)
from iris.application.scientific_ablation_inference import _digest, _exact_tree
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherOutputV1
from iris.application.scientific_ablation_validation import _complete_row_fields, _has_signal
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_command import ScientificParquetInputV1
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_diagnostics_v1 import (
    PROBE_NAMES,
    IrisScientificAblationDiagnosticsV1,
    ScientificAblationDiagnosticPolicyV1,
    ScientificAblationDiagnosticRowV1,
    ScientificAblationDiagnosticsPayloadV1,
    ScientificAblationProbeObservationV1,
)
from iris.data.schemas import IrisExample
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherTracedOutputV1
from iris.training.probes import prepare_probe_targets, probe_vocabulary_sha256
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.trace import (
    WorkspaceTraceOptionsV1,
    WorkspaceTraceV1,
    _scalar_bound,
    summarize_workspace_traces_v1,
)

_DOMAIN = "iris/scientific-ablation-diagnostics/v1"
_POLICY_DOMAIN = "iris/scientific-ablation-diagnostic-policy/v1"
_MAX_REPORT = 64 * 1024**2


class IrisScientificAblationDiagnosticsError(RuntimeError):
    """Incomplete or changed ordinary controls cannot be returned as a report."""


def _fail(primary: BaseException | None, cleanup: list[BaseException]) -> None:
    if primary is not None and _has_signal(primary):
        raise primary from (
            BaseExceptionGroup("Ablation diagnostic cleanup failed", cleanup) if cleanup else None
        )
    signal = next((error for error in cleanup if _has_signal(error)), None)
    if signal is not None:
        raise signal from primary
    failures = ([] if primary is None else [primary]) + cleanup
    if failures:
        raise IrisScientificAblationDiagnosticsError("Ordinary ablation diagnostics failed") from (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup("Ablation diagnostic failures", failures)
        )


def _checked(result: IrisScientificAblationDiagnosticsV1) -> bytes:
    _exact_tree(result, IrisScientificAblationDiagnosticsV1)
    owned = IrisScientificAblationDiagnosticsV1.model_validate(
        result.model_dump(mode="python", warnings=False)
    )
    if owned.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("Distinct ordinary diagnostic checksum differs")
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_REPORT:
        raise ValueError("Ordinary diagnostic report byte limit exceeded")
    return content


def _teacher(
    context: IrisScientificAblationGenerationContextV1,
    row: IrisExample,
    expected_count: int,
    options: WorkspaceTraceOptionsV1,
) -> IrisScientificAblationTeacherOutputV1:
    """Actual traced counterpart of held-out EOS NLL, with the same native bindings."""
    run, model = context._run, context._run._optimizer._model
    prepared = prepare_teacher_forcing_input(
        backbone=run._backbone,
        prompt=row.prompt,
        target_text=row.expected_response,
        terminal_token_id=run._terminal_token_id,
    )
    target_sha = teacher_target_sha256_v1(prepared.target_input_ids, prepared.target_attention_mask)
    if int(prepared.target_attention_mask.sum()) != expected_count:
        raise ValueError("Complete original teacher target count changed")
    prompt_sha = domain_sha256("iris/cognitive-ablation-prompt/v1", row.prompt.encode())
    sidecar_sha = model.teacher.evaluation_provenance.sidecar_state_sha256
    core_invocation = domain_sha256(
        "iris/cognitive-ablation-teacher-invocation/v1",
        canonical_json_bytes(
            {
                "mode": "evaluation",
                "policy_sha256": model.ablation_policy_sha256,
                "prompt_sha256": prompt_sha,
                "target_sha256": target_sha,
                "teacher_graph_sha256": run._teacher_graph_identity,
                "sidecar_state_sha256": sidecar_sha,
            }
        ),
    )
    example_sha = domain_sha256(
        "iris/scientific-ablation-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
    )
    state_sha = run._optimizer._committed_scientific_state_sha256
    invocation = domain_sha256(
        "iris/scientific-ablation-invocation/v1",
        canonical_json_bytes(
            {
                "mode": "evaluation",
                "example_sha256": example_sha,
                "configuration_sha256": model.configuration_sha256,
                "teacher_invocation_sha256": core_invocation,
                "scientific_state_sha256": state_sha,
            }
        ),
    )
    output = model.evaluation_forward(
        row,
        terminal_token_id=run._terminal_token_id,
        diagnostics=options,
    )
    if (
        type(output) is not IrisScientificAblationTeacherOutputV1
        or output.mode != "evaluation"
        or output.example_id != row.id
        or output.example_sha256 != example_sha
        or output.configuration_sha256 != model.configuration_sha256
        or output.scientific_state_sha256 != state_sha
        or output.invocation_sha256 != invocation
        or output.ablation_policy_sha256 != model.ablation_policy_sha256
        or type(output.supervised_token_count) is not int
        or output.supervised_token_count != expected_count
        or type(output.teacher) is not IrisAblatedTeacherOutputV1
    ):
        raise ValueError("Actual native-best scientific teacher output binding differs")
    teacher = output.teacher
    if (
        teacher.mode != "evaluation"
        or teacher.invocation_sha256 != core_invocation
        or teacher.prompt_sha256 != prompt_sha
        or teacher.target_sha256 != target_sha
        or teacher.teacher_graph_sha256 != run._teacher_graph_identity
        or teacher.sidecar_state_sha256 != sidecar_sha
        or cognitive_ablation_policy_sha256_v1(teacher.policy) != model.ablation_policy_sha256
        or type(teacher.traced) is not IrisTeacherTracedOutputV1
        or type(teacher.traced.teacher) is not IrisTeacherForcedOutput
        or teacher.traced.teacher.supervised_token_count != expected_count
    ):
        raise ValueError("Actual mode-bound core teacher/EOS/invocation differs")
    loss = output.losses.language_model
    if (
        type(loss) is not torch.Tensor
        or loss.ndim != 0
        or loss.requires_grad
        or loss.grad_fn is not None
        or loss.is_inference()
        or loss.dtype not in (torch.float32, torch.float64)
        or loss.device != run._backbone.device
        or not bool(torch.isfinite(loss))
        or float(loss) < 0
    ):
        raise ValueError("Actual detached complete target NLL is invalid")
    return output


def measure_scientific_ablation_diagnostics_v1(
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
    policy: ScientificAblationDiagnosticPolicyV1,
    maximum_capture_bytes: int = 512 * 1024**2,
    maximum_evaluation_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationDiagnosticsV1:
    """Reproduce EVERY saved EOS NLL and measure real ordinary traces/frozen probes.

    Independently trusted generation/complete-capture/score-plan/report hashes
    are required. Hash integrity is not external execution attestation. All
    budgets are preflighted; an insufficient budget never selects a subset.
    """
    try:
        for anchor in (
            expected_evaluation_plan_sha256,
            expected_evaluation_sha256,
            expected_plan_sha256,
            expected_capture_sha256,
        ):
            _digest(anchor)
        _exact_tree(policy, ScientificAblationDiagnosticPolicyV1)
        policy = ScientificAblationDiagnosticPolicyV1.model_validate(
            policy.model_dump(mode="python")
        )
        seal, manifest = _heldout_entry(heldout)
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error
    context = seal.context
    run, model = context._run, context._run._optimizer._model
    caller_dtype = torch.get_default_dtype()
    caller = None
    modes = tuple((module, module.training) for module in model.modules())
    entered = False
    primary: BaseException | None = None
    cleanup: list[BaseException] = []
    result = None

    def read_scores() -> IrisScientificAblationEvaluationProgressV2:
        return read_scientific_ablation_evaluation_progress_v2(
            evaluation_path,
            expected_evaluation_plan_sha256=expected_evaluation_plan_sha256,
            expected_plan_sha256=expected_plan_sha256,
            expected_capture_sha256=expected_capture_sha256,
            maximum_storage_bytes=maximum_evaluation_bytes,
        )

    def audit_rows() -> None:
        if (
            heldout._guard is not seal
            or context._run is not run
            or run._optimizer._model is not model
        ):
            raise ValueError("Original held-out/native teacher ownership changed")
        for row, fingerprint in zip(seal.rows, seal.fingerprints, strict=True):
            if (
                not _complete_row_fields(row)
                or canonical_json_bytes(row.model_dump(mode="json")) != fingerprint
            ):
                raise ValueError("Original private complete diagnostic row changed")

    with torch.inference_mode(False), torch.enable_grad(), torch.device(run._backbone.device):
        try:
            caller = capture_runtime_state(run._backbone.device)
            progress = read_scores()
            if (
                progress.complete is None
                or progress.evaluation is None
                or progress.evaluation.sha256 != expected_evaluation_sha256
            ):
                raise ValueError("Independently trusted COMPLETE score ledger/report required")
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
                or tuple(row.target_token_count for row in progress.evaluation.payload.rows)
                != seal.target_counts
            ):
                raise ValueError("Owned native best/test/capture/score plan differs")
            _audit_file(seal)
            _audit_original(context)
            audit_rows()
            teacher = model.teacher
            geometry = (
                teacher._num_modules,
                teacher._num_slots,
                teacher._state_dim,
                teacher._cycles,
            )
            per_trace = _scalar_bound(*geometry, policy.trace_mode == "full")
            total = len(seal.rows) * per_trace
            if (
                len(seal.rows) > policy.maximum_teacher_executions
                or per_trace > policy.maximum_trace_scalars
                or total > policy.maximum_total_trace_scalars
            ):
                raise ValueError("Complete diagnostic execution/scalar budgets insufficient")
            seed = _json(run.configuration_json.encode())["seed"]
            if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
                raise ValueError("Original native run seed is invalid")
            vocabulary = model.probes.vocabulary
            vocabulary_sha = probe_vocabulary_sha256(vocabulary)
            probe_config = model.probes.configuration_sha256
            probe_json = model.probes.configuration_json
            if (
                _json(probe_json.encode())["vocabulary_sha256"] != vocabulary_sha
                or _json(probe_json.encode())["pooling"] != "mean_all_slots"
                or _json(probe_json.encode())["state_dim"] != teacher._state_dim
            ):
                raise ValueError("Original frozen probe profile/class vocabulary differs")
            class_counts = tuple(len(getattr(vocabulary, name)) for name in PROBE_NAMES)
            # Validate every label BEFORE any teacher forward; no labels enter a head.
            for row in seal.rows:
                prepare_probe_targets(
                    (row,), vocabulary=vocabulary, purpose="analysis", device=run._backbone.device
                )
            torch.set_default_dtype(torch.float32)
            heldout._state = context._state = "busy"
            run._busy = entered = True
            model.eval()
            witnesses: list[ScientificAblationDiagnosticRowV1] = []
            for index, row in enumerate(seal.rows):
                if canonical_json_bytes(row.model_dump(mode="json")) != seal.fingerprints[index]:
                    raise ValueError("Owned diagnostic row changed before forward")
                options = WorkspaceTraceOptionsV1(
                    mode=policy.trace_mode,
                    example_id=row.id,
                    checkpoint_id=manifest.payload.selected_checkpoint_sha256,
                    seed=seed,
                    maximum_scalars=policy.maximum_trace_scalars,
                )
                restore_runtime_state(run._runtime)
                output = _teacher(context, row, seal.target_counts[index], options)
                score = progress.evaluation.payload.rows[index]
                if (
                    output.invocation_sha256 != score.teacher_invocation_sha256
                    or output.teacher.target_sha256 != score.teacher_target_sha256
                    or float(output.losses.language_model) != score.target_nll_sum
                ):
                    raise ValueError(
                        "Actual original EOS target NLL/invocation differs from saved capture"
                    )
                traced = output.teacher.traced.teacher
                trace = traced.diagnostic_trace
                if type(trace) is not WorkspaceTraceV1:
                    raise ValueError("Actual complete diagnostic trace required")
                _exact_tree(trace, WorkspaceTraceV1)
                trace = WorkspaceTraceV1.model_validate(trace.model_dump(mode="python"))
                if (
                    trace.options != options
                    or trace.intervention_plan_sha256 != model.ablation_policy_sha256
                    or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
                    != geometry
                    or len(traced.workspace_trace) != teacher._cycles
                ):
                    raise ValueError("Actual ordinary mode-bound trace geometry/options differs")
                targets = prepare_probe_targets(
                    (row,),
                    vocabulary=vocabulary,
                    purpose="analysis",
                    device=run._backbone.device,
                )
                with torch.no_grad():
                    logits = model.probes(traced.workspace_trace[-1].slots)
                if set(logits) != set(PROBE_NAMES):
                    raise ValueError("Every original frozen probe head required")
                probes = []
                for name, classes in zip(PROBE_NAMES, class_counts, strict=True):
                    value = logits[name]
                    if (
                        type(value) is not torch.Tensor
                        or tuple(value.shape) != (1, classes)
                        or value.device != run._backbone.device
                        or value.requires_grad
                        or value.grad_fn is not None
                        or value.is_inference()
                        or value.dtype not in (torch.float32, torch.float64)
                        or not bool(torch.isfinite(value).all())
                    ):
                        raise ValueError("Actual frozen probe predictions invalid")
                    predicted, target = int(value.argmax(dim=1)[0]), int(targets.labels[name][0])
                    probes.append(
                        ScientificAblationProbeObservationV1(
                            field=name,
                            class_count=classes,
                            predicted_class=predicted,
                            target_sha256=domain_sha256(
                                "iris/scientific-ablation-probe-target/v1",
                                canonical_json_bytes(
                                    {
                                        "example_sha256": score.generation_example_sha256,
                                        "vocabulary_sha256": vocabulary_sha,
                                        "field": name,
                                        "class_index": target,
                                    }
                                ),
                            ),
                            correct=predicted == target,
                        )
                    )
                if (
                    canonical_json_bytes(row.model_dump(mode="json")) != seal.fingerprints[index]
                    or model.probes.configuration_sha256 != probe_config
                    or model.probes.configuration_json != probe_json
                    or probe_vocabulary_sha256(model.probes.vocabulary) != vocabulary_sha
                ):
                    raise ValueError("Original diagnostic row/frozen probe profile changed")
                witnesses.append(
                    ScientificAblationDiagnosticRowV1(
                        row_index=index,
                        capture_row_sha256=score.capture_row_sha256,
                        evaluation_row_sha256=progress.rows[index].sha256,
                        evaluation_ack_sha256=progress.acknowledgements[index].sha256,
                        teacher_invocation_sha256=output.invocation_sha256,
                        teacher_target_sha256=output.teacher.target_sha256,
                        target_token_count=output.supervised_token_count,
                        target_nll_sum=float(output.losses.language_model),
                        trace=trace,
                        probes=tuple(probes),
                    )
                )
            audit_rows()
            _audit_file(seal)
            _audit_original(context)
            _finish(inputs)
            if read_scores() != progress:
                raise ValueError("Original complete score ledger changed during diagnostics")
            workspace = summarize_workspace_traces_v1(tuple(row.trace for row in witnesses))
            outcome: Literal["not_assessed_smoke", "not_applicable_geometry", "met", "not_met"] = (
                "not_assessed_smoke"
                if manifest.payload.purpose == "smoke"
                else "not_applicable_geometry"
                if geometry[:2] != (8, 8)
                else "met"
                if workspace.effective_active_modules >= 4 and workspace.effective_active_slots >= 4
                else "not_met"
            )
            correct = tuple(sum(row.probes[i].correct for row in witnesses) for i in range(4))
            payload = ScientificAblationDiagnosticsPayloadV1(
                evaluation_plan=progress.plan,
                evaluation=progress.evaluation,
                evaluation_complete=progress.complete,
                policy=policy,
                policy_sha256=domain_sha256(
                    _POLICY_DOMAIN, canonical_json_bytes(policy.model_dump(mode="json"))
                ),
                run_seed=seed,
                teacher_graph_sha256=run._teacher_graph_identity,
                probe_configuration_sha256=probe_config,
                probe_vocabulary_sha256=vocabulary_sha,
                probe_class_counts=(
                    class_counts[0],
                    class_counts[1],
                    class_counts[2],
                    class_counts[3],
                ),
                teacher_executions=len(witnesses),
                trace_scalar_bound=total,
                rows=tuple(witnesses),
                workspace=workspace,
                probe_correct_counts=(correct[0], correct[1], correct[2], correct[3]),
                source_20_4=outcome,
            )
            result = IrisScientificAblationDiagnosticsV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            _checked(result)
        except BaseException as error:
            primary = error
        finally:
            if entered:
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
            if entered:
                run._busy = False
                context._state = heldout._state = "ready"
                try:
                    _heldout_entry(heldout)
                    _context_entry(context)
                except BaseException as error:
                    cleanup.append(error)
            if (entered or cleanup) and (primary is not None or cleanup):
                heldout._state = "poisoned"
                _poison_context(context)
    _fail(primary, cleanup)
    if result is None:
        raise IrisScientificAblationDiagnosticsError("No complete ordinary diagnostic report")
    return result


__all__ = ["IrisScientificAblationDiagnosticsError", "measure_scientific_ablation_diagnostics_v1"]
