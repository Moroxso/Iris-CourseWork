"""Actual trained earlier-best seven-field write observations in all12 modes.

This tests the native seam inside the original source transaction. It is NOT
the final full-effect experiment, its statistics, or generated-answer quality.
"""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application import scientific_ablation_causal_controls as source_module
from iris.application.scientific_optimization import _same_runtime
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.interventions.messages import WorkspaceMessagePlanV1
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.training.ablation_observation import (
    IrisAblationObservedTeacherOutputV1,
    _AblationWorkspaceObservationSessionV1,
)
from iris.training.model import IrisTeacherForcingError
from iris.utils.hashing import canonical_json_file_bytes, tensor_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_cognitive_ablation_controls import _control, _json, _options
from tests.integration.test_intervened_teacher import prepared
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_diagnostics import _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.integration.test_scientific_teacher import _build
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration


def _state_json(value: WorkspaceState) -> dict[str, Any]:
    return {
        f.name: dict(
            values=getattr(value, f.name).tolist(),
            dtype=str(getattr(value, f.name).dtype).removeprefix("torch."),
            shape=list(getattr(value, f.name).shape),
            tensor_sha256=tensor_sha256(getattr(value, f.name)),
        )
        for f in fields(value)
    }


def _observed_json(value: IrisAblationObservedTeacherOutputV1) -> dict[str, Any]:
    return dict(
        kind=value.kind,
        controlled=_json(value.controlled),
        maximum_observation_scalars=value.maximum_observation_scalars,
        actual_observation_scalars=value.actual_observation_scalars,
        invocation_sha256=value.invocation_sha256,
        observations=[
            dict(
                cycle=o.cycle,
                before_write=_state_json(o.before_write),
                native_written=_state_json(o.native_written),
                before_write_sha256=o.before_write_sha256,
                native_written_sha256=o.native_written_sha256,
            )
            for o in value.observations
        ],
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_actual_original_best_source_prewrite_and_native_written_isolation(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    experiment = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(experiment, routing, goal=goal, broadcast=broadcast)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        source_module._source_transaction(
            owner,
            input=_input(experiment),
            policy=_policy(),
            maximum_report_bytes=64 * 1024**2,
        ) as source,
        patch.object(
            source.model.teacher, "_execute", wraps=source.model.teacher._execute
        ) as execute,
    ):
        row_index = source.design.payload.selected_row_indices[0]
        row = source.seal.rows[row_index]
        example = prepare_teacher_forcing_input(
            backbone=source.run._backbone,
            prompt=row.prompt,
            target_text=row.expected_response,
            terminal_token_id=source.run._terminal_token_id,
        )
        options = WorkspaceTraceOptionsV1(
            mode="full",
            example_id=row.id,
            checkpoint_id=source.manifest.payload.selected_checkpoint_sha256,
            seed=source.seed,
        )
        plan = CognitiveAblationControlPlanV1(
            ablation=source.design.payload.ablation,
            control=WorkspaceInterventionPlanV1(interventions=()),
        )
        teacher = source.model.teacher
        restore_runtime_state(source.run._runtime)
        baseline = teacher.evaluation_forward_ablation_observed(
            example,
            plan=plan,
            diagnostics=options,
            maximum_observation_scalars=336,
        )
        assert type(baseline) is IrisAblationObservedTeacherOutputV1
        assert baseline.actual_observation_scalars == 336
        assert baseline.controlled.teacher_graph_sha256 == source.controlled_graph_sha256
        assert baseline.controlled.sidecar_state_sha256 == source.sidecar_sha256
        assert source.progress.evaluation is not None
        saved = source.progress.evaluation.payload.rows[row_index]
        assert baseline.controlled.target_sha256 == saved.teacher_target_sha256
        assert baseline.controlled.supervised_token_count == saved.target_token_count
        assert float(baseline.controlled.loss) * saved.target_token_count == pytest.approx(
            saved.target_nll_sum, rel=2e-7, abs=1e-7
        )
        restore_runtime_state(source.run._runtime)
        messages = teacher.evaluation_forward_ablation_observed(
            example,
            plan=CognitiveAblationControlPlanV1(
                ablation=plan.ablation,
                control=WorkspaceMessagePlanV1(),
                maximum_message_scalars=96,
            ),
            diagnostics=options,
            maximum_observation_scalars=336,
        )
        assert len(messages.controlled.messages) == 12
        assert torch.equal(messages.controlled.loss, baseline.controlled.loss)
        for left, right in zip(messages.observations, baseline.observations, strict=True):
            assert_same(left.before_write, right.before_write)
            assert_same(left.native_written, right.native_written)
        selected = baseline.controlled.diagnostic_trace.cycles[1]
        writer, slot = selected.selected_modules[0], selected.selected_slots[0]
        blocked_plan = plan.model_copy(
            update={
                "control": WorkspaceInterventionPlanV1(
                    interventions=(
                        WorkspaceInterventionSpecV1(
                            kind="block_module_write",
                            module=writer,
                            cycles=(2,),
                        ),
                    ),
                )
            }
        )
        restore_runtime_state(source.run._runtime)
        blocked = teacher.evaluation_forward_ablation_observed(
            example,
            plan=blocked_plan,
            diagnostics=options,
            maximum_observation_scalars=336,
        )
        assert_same(baseline.observations[1].before_write, blocked.observations[1].before_write)
        assert_same(baseline.observations[1].native_written, blocked.observations[1].native_written)
        actual, observation = blocked.controlled.workspace_trace[1], blocked.observations[1]
        for f in fields(WorkspaceState):
            assert torch.equal(
                getattr(actual, f.name)[:, slot], getattr(observation.before_write, f.name)[:, slot]
            )
            for other in range(4):
                if other != slot:
                    assert torch.equal(
                        getattr(actual, f.name)[:, other],
                        getattr(observation.native_written, f.name)[:, other],
                    )
        for actual_baseline, o in zip(
            baseline.controlled.workspace_trace, baseline.observations, strict=True
        ):
            assert_same(actual_baseline, o.native_written)
        assert owner._state == source.context._state == source.context_seal.request._state == "busy"
        assert source.run._busy
        assert execute.call_count == 3
        source.audit(measuring=True)
        assert experiment.progress.evaluation is not None
        original_source = experiment.progress.evaluation.payload.plan.payload.source.payload
        assert (original_source.selected_step, original_source.terminal_step) == (1, 3)
        witness = dict(
            kind="iris-tests-native-ablation-observation-witness-v1",
            original_source=original_source.model_dump(mode="json"),
            design=source.design.model_dump(mode="json"),
            evaluation_plan_sha256=experiment.progress.plan.sha256,
            generation_plan_sha256=experiment.progress.evaluation.payload.plan.sha256,
            row_index=row_index,
            writer=writer,
            slot=slot,
            cycle=2,
            before_slot_preexisting=bool(observation.before_write.occupancy[0, slot]),
            actual_observed_executions=execute.call_count,
            baseline=_observed_json(baseline),
            messages=_observed_json(messages),
            blocked=_observed_json(blocked),
            scientific_acceptance="not_assessed",
            full_r015_complete=False,
        )
    assert owner._state == owner._guard.context._state == "ready"
    assert not owner._guard.context._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    (tmp_path / "observations.json").write_bytes(canonical_json_file_bytes(witness))
    (tmp_path / "design.json").write_bytes(canonical_json_file_bytes(witness["design"]))


def test_real_preexisting_overwrite_restores_every_field_not_an_empty_record(
    tmp_path: Path,
) -> None:
    model, _, _ = _build(slots=2)
    model.eval()
    example = prepared(model)
    plan = _control(CognitiveAblationPolicyV1())
    baseline = model.teacher.evaluation_forward_ablation_observed(
        example,
        plan=plan,
        diagnostics=_options(),
        maximum_observation_scalars=168,
    )
    selected = baseline.controlled.diagnostic_trace.cycles[1]
    writer, slot = selected.selected_modules[0], selected.selected_slots[0]
    before = baseline.observations[1].before_write
    assert before.occupancy[0, slot] and before.age[0, slot] > 0
    assert before.confidence[0, slot] > 0 and before.ttl[0, slot] > 0
    assert before.write_count[0, slot] > 0 and before.slots[0, slot].any()
    blocked = model.teacher.evaluation_forward_ablation_observed(
        example,
        plan=_control(plan.ablation, dict(kind="block_module_write", module=writer, cycles=(2,))),
        diagnostics=_options(),
        maximum_observation_scalars=168,
    )
    for f in fields(WorkspaceState):
        assert torch.equal(
            getattr(blocked.controlled.workspace_trace[1], f.name)[:, slot],
            getattr(blocked.observations[1].before_write, f.name)[:, slot],
        )
    (tmp_path / "overwrite.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                configuration=dict(
                    backbone=model.teacher.evaluation_provenance.backbone_configuration_json,
                    cognitive=model.teacher.evaluation_provenance.cognitive_configuration_json,
                    workspace=model.teacher.evaluation_provenance.workspace_configuration_json,
                ),
                fixture="offline-random-tiny-qwen-not-trained-source",
                writer=writer,
                slot=slot,
                baseline=_observed_json(baseline),
                blocked=_observed_json(blocked),
            )
        )
    )


@pytest.mark.parametrize("budget", (0, True, 335, 2_000_001, 336.0))
def test_observation_budget_denies_before_native_model_execution(budget: Any) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    with patch.object(model.teacher, "_execute") as execute, pytest.raises(IrisTeacherForcingError):
        model.teacher.evaluation_forward_ablation_observed(
            example,
            plan=_control(CognitiveAblationPolicyV1()),
            diagnostics=_options(),
            maximum_observation_scalars=budget,
        )
    execute.assert_not_called()


@pytest.mark.parametrize("field", tuple(f.name for f in fields(WorkspaceState)))
def test_revalidation_detects_mutated_retained_observation(field: str) -> None:
    model, _, _ = _build()
    model.eval()
    value = model.teacher.evaluation_forward_ablation_observed(
        prepared(model),
        plan=_control(CognitiveAblationPolicyV1()),
        diagnostics=_options(),
        maximum_observation_scalars=336,
    )
    tensor = getattr(value.observations[1].native_written, field)
    if tensor.dtype == torch.bool:
        tensor.logical_not_()
    else:
        tensor.add_(1)
    with pytest.raises((ValueError, RuntimeError)):
        value.__post_init__()


@pytest.mark.parametrize("change", ("invocation", "limit", "order", "alias"))
def test_distinct_receipt_cannot_rebind_or_alias_native_tensors(change: str) -> None:
    model, _, _ = _build()
    model.eval()
    value = model.teacher.evaluation_forward_ablation_observed(
        prepared(model),
        plan=_control(CognitiveAblationPolicyV1()),
        diagnostics=_options(),
        maximum_observation_scalars=336,
    )
    changes_by_kind: dict[str, dict[str, Any]] = {
        "invocation": dict(invocation_sha256="a" * 64),
        "limit": dict(maximum_observation_scalars=337),
        "order": dict(observations=tuple(reversed(value.observations))),
        "alias": dict(
            observations=(
                replace(
                    value.observations[0],
                    native_written=value.observations[0].before_write,
                    native_written_sha256=value.observations[0].before_write_sha256,
                ),
                *value.observations[1:],
            )
        ),
    }
    with pytest.raises(ValueError):
        replace(value, **changes_by_kind[change])


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_native_observer_signals_preserve_identity_and_caller_rng(signal_type: Any) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    signal = signal_type("private observer signal")
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(_AblationWorkspaceObservationSessionV1, "before_write", side_effect=signal),
        pytest.raises(signal_type) as caught,
    ):
        model.teacher.evaluation_forward_ablation_observed(
            example,
            plan=_control(CognitiveAblationPolicyV1()),
            diagnostics=_options(),
            maximum_observation_scalars=336,
        )
    assert caught.value is signal
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
