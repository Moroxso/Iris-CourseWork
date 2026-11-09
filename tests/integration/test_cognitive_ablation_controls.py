"""Actual offline tiny-Qwen mode+control math; not source/scientific acceptance."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application.scientific_optimization import _same_runtime
from iris.application.workspace_causal import _same_pre_write
from iris.interventions.messages import WorkspaceMessagePlanV1
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.training.ablation_controls import IrisAblationControlledTeacherOutputV1
from iris.training.messages import IrisMessageTeacherOutputV1
from iris.training.model import IrisTeacherForcingError
from iris.utils.hashing import canonical_json_file_bytes, module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_cognitive_ablation_teacher import _policy
from tests.integration.test_intervened_teacher import OPERATIONS, prepared
from tests.integration.test_message_causal import prepared as message_examples
from tests.integration.test_scientific_teacher import _build
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration
MODES = tuple(
    _policy(routing, goal=goal, broadcast=broadcast)
    for routing in ("learned", "random", "uniform")
    for goal in ("enabled", "disabled")
    for broadcast in ("enabled", "disabled")
)


def _control(
    ablation: CognitiveAblationPolicyV1, *specs: dict[str, Any]
) -> CognitiveAblationControlPlanV1:
    return CognitiveAblationControlPlanV1(
        ablation=ablation,
        control=WorkspaceInterventionPlanV1(
            interventions=tuple(WorkspaceInterventionSpecV1.model_validate(s) for s in specs)
        ),
    )


def _options(example_id: str = "recipient") -> WorkspaceTraceOptionsV1:
    return WorkspaceTraceOptionsV1(
        mode="full", example_id=example_id, checkpoint_id="random-fixture", seed=42
    )


def _json(output: IrisAblationControlledTeacherOutputV1) -> dict[str, Any]:
    return dict(
        loss=float(output.loss),
        supervised_token_count=output.supervised_token_count,
        plan=output.plan.model_dump(mode="json"),
        plan_sha256=output.plan_sha256,
        ablation_policy_sha256=output.ablation_policy_sha256,
        control_plan_sha256=output.control_plan_sha256,
        prompt_sha256=output.prompt_sha256,
        target_sha256=output.target_sha256,
        teacher_graph_sha256=output.teacher_graph_sha256,
        sidecar_state_sha256=output.sidecar_state_sha256,
        invocation_sha256=output.invocation_sha256,
        trace=output.diagnostic_trace.model_dump(mode="json"),
        messages=[m.model_dump(mode="json") for m in output.messages],
        states=[
            {field.name: getattr(state, field.name).tolist() for field in fields(state)}
            for state in output.workspace_trace
        ],
        kind=output.kind,
    )


def _save(path: Path, *, scenario: str, teacher: Any, **receipts: Any) -> None:
    provenance = teacher.evaluation_provenance
    value = dict(
        kind="iris-tests-ablation-controls-witness-v1",
        scenario=scenario,
        backbone=provenance.backbone_configuration_json,
        cognitive=provenance.cognitive_configuration_json,
        workspace=provenance.workspace_configuration_json,
        scientific_acceptance="not_assessed",
        fixture="offline-random-tiny-qwen",
        receipts=receipts,
    )
    path.write_bytes(canonical_json_file_bytes(value))


@pytest.mark.parametrize("ablation", MODES)
def test_all12_empty_control_and_message_collection_match_actual_ablation_bitwise(
    ablation: CognitiveAblationPolicyV1,
    tmp_path: Path,
) -> None:
    model, _, lm = _build()
    example = prepared(model)
    torch.autograd.backward(model.teacher.training_forward(example).loss)
    gradients = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    model.eval()
    before = module_state_sha256((("model", model), ("lm", lm)))
    runtime = capture_runtime_state(torch.device("cpu"))
    ordinary = model.teacher.evaluation_forward_ablated(
        example, policy=ablation, diagnostics=_options()
    )
    output = model.teacher.evaluation_forward_ablation_controlled(
        example, plan=_control(ablation), diagnostics=_options()
    )
    messages = model.teacher.evaluation_forward_ablation_controlled(
        example,
        plan=CognitiveAblationControlPlanV1(
            ablation=ablation, control=WorkspaceMessagePlanV1(), maximum_message_scalars=65536
        ),
        diagnostics=_options(),
    )
    assert type(output) is type(messages) is IrisAblationControlledTeacherOutputV1
    assert torch.equal(ordinary.traced.teacher.loss, output.loss) and torch.equal(
        output.loss, messages.loss
    )
    assert output.supervised_token_count == ordinary.traced.teacher.supervised_token_count
    assert ordinary.traced.teacher.diagnostic_trace is not None
    assert (
        ordinary.traced.teacher.diagnostic_trace.cycles
        == output.diagnostic_trace.cycles
        == messages.diagnostic_trace.cycles
    )
    for actual in (output, messages):
        assert actual.diagnostic_trace.intervention_plan_sha256 == actual.plan_sha256
        assert actual.plan_sha256 != actual.ablation_policy_sha256 != actual.control_plan_sha256
        for left, right in zip(
            ordinary.traced.teacher.workspace_trace, actual.workspace_trace, strict=True
        ):
            assert_same(left, right)
    assert len(messages.messages) == 12 and not output.messages
    assert before == module_state_sha256((("model", model), ("lm", lm)))
    assert _same_runtime(runtime, capture_runtime_state(torch.device("cpu")))
    for n, p in model.named_parameters():
        expected = gradients[n]
        assert (expected is None and p.grad is None) or (
            expected is not None and p.grad is not None and torch.equal(expected, p.grad)
        )
    assert all(not m.training for m in model.modules())
    _save(
        tmp_path / "controls.json",
        scenario="empty",
        teacher=model.teacher,
        workspace=_json(output),
        messages=_json(messages),
        ordinary_trace=ordinary.traced.teacher.diagnostic_trace.model_dump(mode="json"),
        ordinary_loss=float(ordinary.traced.teacher.loss),
    )


@pytest.mark.parametrize("ablation", MODES)
@pytest.mark.parametrize("spec", OPERATIONS, ids=lambda s: s["kind"])
def test_all12_operations_across_all12_modes_have_real_fresh_ordered_math(
    ablation: CognitiveAblationPolicyV1,
    spec: dict[str, Any],
    tmp_path: Path,
) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    ordinary = model.teacher.evaluation_forward_ablated(
        example, policy=ablation, diagnostics=_options()
    )
    selected = _control(ablation, {**spec, "cycles": (2,)})
    first = model.teacher.evaluation_forward_ablation_controlled(
        example, plan=selected, diagnostics=_options()
    )
    repeated = model.teacher.evaluation_forward_ablation_controlled(
        example, plan=selected, diagnostics=_options()
    )
    assert (
        torch.equal(first.loss, repeated.loss)
        and first.invocation_sha256 == repeated.invocation_sha256
    )
    assert first.diagnostic_trace == repeated.diagnostic_trace
    assert_same(first.workspace_trace[0], ordinary.traced.teacher.workspace_trace[0])
    for left, right in zip(first.workspace_trace, repeated.workspace_trace, strict=True):
        assert_same(left, right)
    row = first.diagnostic_trace.cycles[1]
    if spec["kind"] == "zero_workspace":
        assert (
            not first.workspace_trace[1].occupancy.any()
            and not first.workspace_trace[1].slots.any()
        )
    elif spec["kind"] == "zero_slot":
        assert (
            not first.workspace_trace[1].occupancy[0, 0]
            and not first.workspace_trace[1].slots[0, 0].any()
        )
    elif spec["kind"] == "force_module_winner":
        assert spec["module"] in row.selected_modules
    elif spec["kind"] == "block_module_read":
        assert not any(row.read_attention[spec["module"]])
    elif spec["kind"] == "reduce_workspace_capacity":
        assert not any(row.slot_occupancy[spec["capacity"] :])
    elif spec["kind"] == "replace_slot":
        assert tuple(first.workspace_trace[1].slots[0, 0].tolist()) == spec["values"]
        assert row.slot_sources[0] == spec["source_module"]
    if ablation == _policy():
        assert type(selected.control) is WorkspaceInterventionPlanV1
        legacy = model.teacher.evaluation_forward_intervened(
            example, plan=selected.control, diagnostics=_options()
        )
        assert legacy.diagnostic_trace is not None
        assert (
            torch.equal(first.loss, legacy.loss)
            and first.diagnostic_trace.cycles == legacy.diagnostic_trace.cycles
        )
    _save(
        tmp_path / "controls.json",
        scenario="operation",
        teacher=model.teacher,
        changed=_json(first),
        ordinary_trace=ordinary.traced.teacher.diagnostic_trace.model_dump(mode="json")
        if ordinary.traced.teacher.diagnostic_trace
        else None,
    )


@pytest.mark.parametrize("ablation", MODES)
def test_all12_actual_paired_donor_and_write_block_isolate_recipient_routing(
    ablation: CognitiveAblationPolicyV1,
    tmp_path: Path,
) -> None:
    model, _, _ = _build()
    model.eval()
    recipient, donor = message_examples(model)
    collecting = CognitiveAblationControlPlanV1(
        ablation=ablation, control=WorkspaceMessagePlanV1(), maximum_message_scalars=65536
    )
    control = model.teacher.evaluation_forward_ablation_controlled(
        recipient.example, plan=collecting, diagnostics=_options(recipient.example_id)
    )
    original_donor = model.teacher.evaluation_forward_ablation_controlled(
        donor.example, plan=collecting, diagnostics=_options(donor.example_id)
    )
    module = control.diagnostic_trace.cycles[0].write_modules[0]
    slot = control.diagnostic_trace.cycles[0].selected_slots[0]
    snapshot = next(m for m in original_donor.messages if m.cycle == 1 and m.module == module)
    changed = model.teacher.evaluation_forward_ablation_controlled(
        recipient.example,
        plan=CognitiveAblationControlPlanV1(
            ablation=ablation,
            control=WorkspaceMessagePlanV1(donor=snapshot),
            maximum_message_scalars=65536,
        ),
        diagnostics=_options(recipient.example_id),
    )
    blocked = model.teacher.evaluation_forward_ablation_controlled(
        recipient.example,
        plan=_control(ablation, {"kind": "block_module_write", "module": module, "cycles": (1,)}),
        diagnostics=_options(recipient.example_id),
    )
    _same_pre_write(control.diagnostic_trace, changed.diagnostic_trace, 1)
    _same_pre_write(control.diagnostic_trace, blocked.diagnostic_trace, 1)
    for field in fields(control.workspace_trace[0]):
        left, right = (
            getattr(control.workspace_trace[0], field.name),
            getattr(changed.workspace_trace[0], field.name),
        )
        other = [i for i in range(left.shape[1]) if i != slot]
        assert torch.equal(left[:, other], right[:, other])
        if field.name != "slots":
            assert torch.equal(left, right)
    assert tuple(m for m in changed.messages if m.cycle == 1) == tuple(
        m for m in control.messages if m.cycle == 1
    )
    assert not blocked.workspace_trace[0].occupancy[0, slot]
    if ablation.broadcast == "disabled":
        assert (
            control.diagnostic_trace.cycles[1].full is not None
            and changed.diagnostic_trace.cycles[1].full is not None
        )
        assert (
            control.diagnostic_trace.cycles[1].full.module_states
            == changed.diagnostic_trace.cycles[1].full.module_states
        )
        assert blocked.diagnostic_trace.cycles[1].full is not None
        assert (
            control.diagnostic_trace.cycles[1].full.module_states
            == blocked.diagnostic_trace.cycles[1].full.module_states
        )
        assert control.workspace_trace[-1].occupancy.any()  # Final LM bridge is retained.
    _save(
        tmp_path / "controls.json",
        scenario="donor",
        teacher=model.teacher,
        control=_json(control),
        donor=_json(original_donor),
        changed=_json(changed),
        blocked=_json(blocked),
    )


@pytest.mark.parametrize(
    "problem",
    (
        "train",
        "inference",
        "amp",
        "raw-plan",
        "legacy-plan",
        "dimension",
        "trace-budget",
        "message-budget",
        "metadata-message",
        "none-trace",
    ),
)
def test_invalid_control_boundaries_fail_before_backbone(problem: str) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    plan: Any = _control(_policy())
    options = _options()
    if problem == "train":
        model.train()
    elif problem == "raw-plan":
        plan = plan.model_dump()
    elif problem == "legacy-plan":
        plan = plan.control
    elif problem == "dimension":
        plan = _control(_policy(), {"kind": "zero_slot", "slot": 4})
    elif problem == "trace-budget":
        options = options.model_copy(update={"maximum_scalars": 1})
    elif problem in ("message-budget", "metadata-message"):
        plan = CognitiveAblationControlPlanV1(
            ablation=_policy(),
            control=WorkspaceMessagePlanV1(),
            maximum_message_scalars=1 if problem == "message-budget" else 65536,
        )
        if problem == "metadata-message":
            options = options.model_copy(update={"mode": "metadata"})
    elif problem == "none-trace":
        options = options.model_copy(update={"mode": "none"})
    with (
        patch.object(model.teacher.backbone, "forward_prompt") as forward,
        pytest.raises(IrisTeacherForcingError),
    ):
        if problem == "inference":
            with torch.inference_mode():
                model.teacher.evaluation_forward_ablation_controlled(
                    example, plan=plan, diagnostics=options
                )
        elif problem == "amp":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model.teacher.evaluation_forward_ablation_controlled(
                    example, plan=plan, diagnostics=options
                )
        else:
            model.teacher.evaluation_forward_ablation_controlled(
                example, plan=plan, diagnostics=options
            )
    forward.assert_not_called()


def test_legacy_mixed_sessions_still_fail_before_first_pass() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    cognitive = CognitiveAblationSessionV1(
        _policy(), prompt_sha256="a" * 64, num_modules=4, num_slots=4, state_dim=8, cycles=3
    )
    legacy = WorkspaceInterventionSessionV1(
        WorkspaceInterventionPlanV1(interventions=()),
        num_modules=4,
        num_slots=4,
        state_dim=8,
        cycles=3,
        winners_per_cycle=2,
    )
    with (
        patch.object(model.teacher.backbone, "forward_prompt") as forward,
        pytest.raises(IrisTeacherForcingError, match="legacy"),
    ):
        model.teacher._execute(
            example, expected_training=False, ablation=cognitive, intervention=legacy
        )
    forward.assert_not_called()


@pytest.mark.parametrize("mutation", ("weights", "gradient", "target"))
def test_real_mutation_aborts_without_a_receipt_and_restores_rng(mutation: str) -> None:
    model, _, _ = _build()
    example = prepared(model)
    torch.autograd.backward(model.teacher.training_forward(example).loss)
    model.eval()
    before = capture_runtime_state(torch.device("cpu"))
    original = model.teacher.workspace_reader.forward

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        torch.rand(3)
        with torch.no_grad():
            if mutation == "weights":
                model.teacher.workspace_reader.read_queries.add_(0.01)
            elif mutation == "gradient":
                parameter = next(p for p in model.teacher.parameters() if p.grad is not None)
                assert parameter.grad is not None
                parameter.grad.add_(0.01)
            else:
                example.target_input_ids.add_(0)
        return result

    with (
        patch.object(model.teacher.workspace_reader, "forward", side_effect=changed),
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_ablation_controlled(
            example, plan=_control(_policy()), diagnostics=_options()
        )
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


def test_target_changes_nll_identity_but_not_random_control_prompt_path() -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    other = replace(
        example, target_input_ids=torch.tensor([[1]]), target_attention_mask=torch.tensor([[1]])
    )
    plan = _control(_policy("random"), {"kind": "zero_slot", "slot": 0, "cycles": (2,)})
    first = model.teacher.evaluation_forward_ablation_controlled(
        example, plan=plan, diagnostics=_options()
    )
    second = model.teacher.evaluation_forward_ablation_controlled(
        other, plan=plan, diagnostics=_options()
    )
    assert (
        first.target_sha256 != second.target_sha256
        and first.invocation_sha256 != second.invocation_sha256
    )
    assert (
        first.diagnostic_trace == second.diagnostic_trace
        and first.plan_sha256 == second.plan_sha256
    )


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_first_signal_identity_survives_phase_and_cleanup_failure(cleanup_fails: bool) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    signal = KeyboardInterrupt("first")
    before = capture_runtime_state(torch.device("cpu"))
    with patch.object(CognitiveAblationSessionV1, "route", side_effect=signal):
        if cleanup_fails:
            with (
                patch(
                    "iris.training.model.restore_runtime_state", side_effect=RuntimeError("cleanup")
                ),
                pytest.raises(KeyboardInterrupt) as caught,
            ):
                model.teacher.evaluation_forward_ablation_controlled(
                    example, plan=_control(_policy()), diagnostics=_options()
                )
        else:
            with pytest.raises(KeyboardInterrupt) as caught:
                model.teacher.evaluation_forward_ablation_controlled(
                    example, plan=_control(_policy()), diagnostics=_options()
                )
    assert caught.value is signal
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("phase", ("before_read", "after_proposal", "finish"))
def test_phase_failure_is_redacted_and_restores_runtime(phase: str) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    before = capture_runtime_state(torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(3)
        raise RuntimeError(example.prompt)

    with (
        patch.object(WorkspaceInterventionSessionV1, phase, side_effect=fail),
        pytest.raises(IrisTeacherForcingError) as caught,
    ):
        model.teacher.evaluation_forward_ablation_controlled(
            example, plan=_control(_policy()), diagnostics=_options()
        )
    assert example.prompt not in str(caught.value) and caught.value.__suppress_context__
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "problem", ("mode", "baseline", "sidecar", "width", "module", "cycle", "content")
)
def test_donor_identity_and_geometry_fail_before_recipient_first_pass(problem: str) -> None:
    model, _, _ = _build()
    model.eval()
    example = prepared(model)
    mode = _policy("random")
    collecting = CognitiveAblationControlPlanV1(
        ablation=mode, control=WorkspaceMessagePlanV1(), maximum_message_scalars=65536
    )
    source: IrisMessageTeacherOutputV1 | IrisAblationControlledTeacherOutputV1
    if problem == "baseline":
        source = model.teacher.evaluation_forward_messages(
            example, plan=WorkspaceMessagePlanV1(), diagnostics=_options()
        )
    else:
        source = model.teacher.evaluation_forward_ablation_controlled(
            example, plan=collecting, diagnostics=_options()
        )
    snapshot = source.messages[0]
    if problem == "mode":
        mode = _policy("uniform")
    elif problem != "baseline":
        donor_changes: dict[str, dict[str, Any]] = {
            "sidecar": {"sidecar_state_sha256": "f" * 64},
            "width": {"values": snapshot.values[:-1]},
            "module": {"module": 4},
            "cycle": {"cycle": 4},
            "content": {"content_sha256": "f" * 64},
        }
        changes = donor_changes[problem]
        snapshot = snapshot.model_copy(update=changes)
    plan = collecting.model_copy(
        update={"ablation": mode, "control": WorkspaceMessagePlanV1.model_construct(donor=snapshot)}
    )
    with (
        patch.object(model.teacher.backbone, "forward_prompt") as forward,
        pytest.raises(IrisTeacherForcingError),
    ):
        model.teacher.evaluation_forward_ablation_controlled(
            example, plan=plan, diagnostics=_options()
        )
    forward.assert_not_called()


@pytest.mark.parametrize(
    "problem",
    ("plan", "ablation", "control", "target", "invocation", "count", "trace", "messages", "loss"),
)
def test_rehashed_or_relabelled_receipt_cannot_hide_context_or_coverage(problem: str) -> None:
    model, _, _ = _build()
    model.eval()
    result = model.teacher.evaluation_forward_ablation_controlled(
        prepared(model), plan=_control(_policy()), diagnostics=_options()
    )
    receipt_changes: dict[str, dict[str, Any]] = {
        "plan": {"plan_sha256": "f" * 64},
        "ablation": {"ablation_policy_sha256": "f" * 64},
        "control": {"control_plan_sha256": "f" * 64},
        "target": {"target_sha256": "f" * 64},
        "invocation": {"invocation_sha256": "f" * 64},
        "count": {"supervised_token_count": True},
        "trace": {
            "diagnostic_trace": result.diagnostic_trace.model_copy(
                update={"intervention_plan_sha256": None}
            )
        },
        "messages": {"messages": (object(),)},
        "loss": {"loss": torch.tensor(-1.0)},
    }
    changes = receipt_changes[problem]
    with pytest.raises((ValueError, TypeError)):
        replace(result, **changes)
