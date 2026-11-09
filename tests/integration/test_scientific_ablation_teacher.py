"""Mode-bound actual five-term tiny-Qwen objectives, not pretrained study results."""

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

import iris.application as application
from iris.ablations import CognitiveAblationPolicyV1
from iris.application.scientific_ablation_teacher import (
    IrisScientificAblationTeacherError,
    IrisScientificAblationTeacherOutputV1,
    IrisScientificAblationTeacherV1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizationError,
    IrisScientificOptimizerPolicyV1,
    IrisScientificOptimizerV1,
    _same_runtime,
)
from iris.data.schemas import IrisExample
from iris.training.losses import IrisLossBreakdown
from iris.training.probes import workspace_probe_objective
from iris.utils.hashing import canonical_json_bytes, module_state_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_cognitive_ablation_teacher import _policy
from tests.integration.test_scientific_teacher import _build, _rows
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration


def _wrap(
    policy: CognitiveAblationPolicyV1 | None = None,
) -> IrisScientificAblationTeacherV1:
    base, _, _ = _build()
    return IrisScientificAblationTeacherV1(base=base, ablation=policy or _policy())


def _losses_equal(left: IrisLossBreakdown, right: IrisLossBreakdown) -> None:
    for field in fields(left):
        if field.name == "weights":
            assert left.weights == right.weights
        else:
            assert torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name


def _diagnostics(row: IrisExample) -> WorkspaceTraceOptionsV1:
    return WorkspaceTraceOptionsV1(
        mode="metadata", example_id=row.id, checkpoint_id="ablation-fixture", seed=42
    )


def test_normal_control_preserves_all_five_losses_states_usage_and_gradients_bitwise() -> None:
    base, _, _ = _build()
    row = _rows()[0]
    normal = base(row, terminal_token_id=1)
    torch.autograd.backward(normal.losses.total)
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in base.named_parameters()
    }
    base.zero_grad(set_to_none=True)
    parameters = tuple(base.parameters())
    baseline = module_state_sha256((("base", base),))
    model = IrisScientificAblationTeacherV1(base=base, ablation=_policy())
    assert tuple(model.parameters()) == parameters
    result = model(row, terminal_token_id=1)
    assert type(result) is IrisScientificAblationTeacherOutputV1
    assert result.configuration_sha256 != normal.configuration_sha256
    assert result.example_sha256 != normal.example_sha256
    assert result.trace is result.teacher.traced
    _losses_equal(normal.losses, result.losses)
    assert torch.equal(normal.usage.module_probabilities, result.usage.module_probabilities)
    assert torch.equal(normal.usage.slot_probabilities, result.usage.slot_probabilities)
    for a, b in zip(
        normal.trace.teacher.workspace_trace, result.trace.teacher.workspace_trace, strict=True
    ):
        assert_same(a, b)
    torch.autograd.backward(result.losses.total)
    for name, p in base.named_parameters():
        expected = gradients[name]
        assert (p.grad is None and expected is None) or (
            p.grad is not None and expected is not None and torch.equal(p.grad, expected)
        ), name
    assert module_state_sha256((("base", base),)) == baseline
    assert application.IrisScientificAblationTeacherV1 is IrisScientificAblationTeacherV1
    assert (
        application.IrisScientificAblationTeacherOutputV1 is IrisScientificAblationTeacherOutputV1
    )


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_all_twelve_real_five_term_modes_eval_parity_gradient_cuts_and_adamw_update(
    routing: Literal["learned", "random", "uniform"], goal: str, broadcast: str
) -> None:
    model = _wrap(_policy(routing, goal=goal, broadcast=broadcast))
    row = _rows()[0]
    count = sum(p.numel() for p in model.parameters())
    assert count == sum(p.numel() for p in model.base.parameters())
    before = module_state_sha256((("model", model),))
    caller = capture_runtime_state(torch.device("cpu"))
    result = model.training_forward(row, terminal_token_id=1, diagnostics=_diagnostics(row))
    assert result.mode == "training" and result.losses.total.requires_grad
    assert result.scientific_state_sha256 == module_state_sha256(
        (("scientific_ablation_teacher", model),)
    )
    assert result.ablation_policy_sha256 == model.ablation_policy_sha256
    assert torch.equal(
        result.losses.language_model, result.trace.teacher.loss * result.supervised_token_count
    )
    assert (
        result.usage.module_probabilities.requires_grad
        and result.usage.slot_probabilities.requires_grad
    )
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    torch.autograd.backward(result.losses.total)
    assert (model.teacher.module_input_router.module_queries.grad is None) == (routing != "learned")
    assert (next(model.teacher.goal_encoder.parameters()).grad is None) == (goal == "disabled")
    assert (model.teacher.workspace_reader.read_queries.grad is None) == (broadcast == "disabled")
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.probes.parameters()
    )
    assert any(
        p.grad is not None and bool((p.grad.abs() > 0).any()) for p in model.probes.parameters()
    )
    assert any(p.grad is not None for p in model.teacher.latent_bridge.parameters())
    gradients = {
        name: (p.grad, None if p.grad is None else p.grad.clone())
        for name, p in model.named_parameters()
    }
    model.eval()
    evaluated = model.evaluation_forward(row, terminal_token_id=1, diagnostics=_diagnostics(row))
    repeated = model.evaluation_forward(row, terminal_token_id=1, diagnostics=_diagnostics(row))
    _losses_equal(result.losses, evaluated.losses)
    assert not evaluated.losses.total.requires_grad
    assert evaluated.configuration_sha256 == result.configuration_sha256
    assert evaluated.invocation_sha256 != result.invocation_sha256
    assert evaluated.invocation_sha256 == repeated.invocation_sha256
    assert evaluated.trace.teacher.diagnostic_trace == result.trace.teacher.diagnostic_trace
    diagnostic = evaluated.trace.teacher.diagnostic_trace
    assert (
        diagnostic is not None
        and diagnostic.intervention_plan_sha256 == model.ablation_policy_sha256
    )
    if broadcast == "disabled":
        assert all(c.read_attention == ((0.0,) * 4,) * 4 for c in diagnostic.cycles)
    for name, p in model.named_parameters():
        original, copy = gradients[name]
        assert p.grad is original
        assert (copy is None and p.grad is None) or (
            copy is not None and p.grad is not None and torch.equal(copy, p.grad)
        )
    assert before == module_state_sha256((("model", model),))
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
    optimizer.step()
    assert any(
        not torch.equal(weights[name], p)
        for name, p in model.named_parameters()
        if name.startswith("base.probes.")
    )
    assert any(
        not torch.equal(weights[name], p)
        for name, p in model.named_parameters()
        if name.startswith("base.teacher.cognitive_modules.")
    )
    assert all(
        torch.equal(weights[name], p)
        for name, p in model.named_parameters()
        if gradients[name][0] is None
    )
    assert model.teacher.backbone.trainable_parameter_count == 0
    assert not model.teacher.backbone.has_parameter_gradients


def test_configuration_binds_all_modes_seed_budget_and_preserves_complete_base_snapshot() -> None:
    policies = [
        _policy(),
        _policy("uniform"),
        _policy("random"),
        _policy("random").model_copy(update={"routing_seed": 1337}),
        _policy(goal="disabled"),
        _policy(broadcast="disabled"),
        _policy().model_copy(update={"maximum_routing_scalars": 65535}),
    ]
    identities = set()
    for policy in policies:
        model = _wrap(policy)
        config = json.loads(model.configuration_json)
        assert config["kind"] == "iris-scientific-ablation-teacher-v1"
        assert config["ablation"] == policy.model_dump(mode="json")
        assert config["base_configuration"] == json.loads(model.base.configuration_json)
        assert config["base_configuration_sha256"] == model.base.configuration_sha256
        assert model.ablation is not model.ablation and model.ablation == policy
        identities.add(model.configuration_sha256)
    assert len(identities) == len(policies)


def test_labels_change_only_probe_objective_and_new_row_identity_not_cognition() -> None:
    model = _wrap(_policy("random", goal="disabled", broadcast="disabled"))
    model.eval()
    row = _rows()[0]
    original = row.model_dump(mode="python")
    facts = {**row.hidden_facts, "operation": "maximum"}
    changed = IrisExample.model_validate(
        {
            **original,
            "hidden_facts": facts,
            "problem_sha256": sha256(
                canonical_json_bytes({"task": row.task_type, "facts": facts})
            ).hexdigest(),
        }
    )
    first = model.evaluation_forward(row, terminal_token_id=1)
    second = model.evaluation_forward(changed, terminal_token_id=1)
    assert first.example_sha256 != second.example_sha256
    assert first.invocation_sha256 != second.invocation_sha256
    assert first.teacher.invocation_sha256 == second.teacher.invocation_sha256
    assert torch.equal(first.losses.language_model, second.losses.language_model)
    assert not torch.equal(first.losses.probe, second.losses.probe)
    for a, b in zip(
        first.trace.teacher.workspace_trace, second.trace.teacher.workspace_trace, strict=True
    ):
        assert_same(a, b)
    assert row.model_dump(mode="python") == original


def test_actual_probe_weights_bind_scientific_but_not_low_level_invocation() -> None:
    model = _wrap(_policy("random"))
    model.eval()
    first = model.evaluation_forward(_rows()[0], terminal_token_id=1)
    with torch.no_grad():
        next(model.probes.parameters()).add_(0.1)
    second = model.evaluation_forward(_rows()[0], terminal_token_id=1)
    assert first.configuration_sha256 == second.configuration_sha256
    assert first.teacher.invocation_sha256 == second.teacher.invocation_sha256
    assert first.scientific_state_sha256 != second.scientific_state_sha256
    assert first.invocation_sha256 != second.invocation_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        "root_buffer",
        "base_buffer",
        "probe_buffer",
        "policy",
        "extra_parameter",
        "extra_buffer",
        "base_extra_module",
        "frozen_probe",
        "base_replaced",
        "parameter_replaced",
        "configuration_json",
    ],
)
def test_mutated_graph_fails_before_preparation(mutation: str) -> None:
    model = _wrap()
    if mutation == "root_buffer":
        model.configuration_identity[0] ^= 1
    elif mutation == "base_buffer":
        model.base.configuration_identity[0] ^= 1
    elif mutation == "probe_buffer":
        model.probes.configuration_identity[0] ^= 1
    elif mutation == "policy":
        model._ablation_json = _policy(goal="disabled").model_dump_json()
    elif mutation == "extra_parameter":
        model.register_parameter("foreign", torch.nn.Parameter(torch.zeros(1)))
    elif mutation == "extra_buffer":
        model.register_buffer("foreign", torch.zeros(1))
    elif mutation == "base_extra_module":
        model.base.add_module("foreign", torch.nn.Identity())
    elif mutation == "frozen_probe":
        next(model.probes.parameters()).requires_grad_(False)
    elif mutation == "base_replaced":
        model.base = _build()[0]
    elif mutation == "parameter_replaced":
        model.teacher.module_input_router.module_queries = torch.nn.Parameter(
            model.teacher.module_input_router.module_queries.detach().clone()
        )
    else:
        model._configuration_json += " "
    with (
        patch.object(model.teacher.backbone, "tokenize_teacher_target") as target,
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        model(_rows()[0], terminal_token_id=1)
    target.assert_not_called()


@pytest.mark.parametrize(
    "invalid",
    [
        "training_on_eval",
        "evaluation_on_train",
        "mixed_mode",
        "no_grad",
        "inference_mode",
        "autocast",
        "held_out",
        "raw_row",
        "copied_row",
        "diagnostic_row",
        "copied_diagnostics",
    ],
)
def test_invalid_input_or_runtime_rejects_before_preparation(invalid: str) -> None:
    model = _wrap()
    row: Any = _rows()[0]
    options = None
    context: Any = nullcontext()
    evaluation = False
    if invalid == "training_on_eval":
        model.eval()
    elif invalid == "evaluation_on_train":
        evaluation = True
    elif invalid == "mixed_mode":
        model.probes.eval()
    elif invalid == "no_grad":
        context = torch.no_grad()
    elif invalid == "inference_mode":
        context = torch.inference_mode()
    elif invalid == "autocast":
        context = torch.autocast("cpu", dtype=torch.bfloat16)
    elif invalid == "held_out":
        row = IrisExample.model_validate(
            {**row.model_dump(mode="python"), "split": "validation", "id": f"validation-{row.seed}"}
        )
    elif invalid == "raw_row":
        row = row.model_dump(mode="python")
    elif invalid == "copied_row":
        row = row.model_copy(update={"split": "foreign"})
    elif invalid == "diagnostic_row":
        options = _diagnostics(_rows()[1])
    else:
        options = _diagnostics(row).model_copy(update={"seed": True})
    with (
        context,
        patch.object(model.teacher.backbone, "tokenize_teacher_target") as target,
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        method = model.evaluation_forward if evaluation else model.training_forward
        method(row, terminal_token_id=1, diagnostics=options)
    target.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "policy_sha256",
        "prompt_sha256",
        "target_sha256",
        "teacher_graph_sha256",
        "sidecar_state_sha256",
        "invocation_sha256",
        "policy",
        "supervised_token_count",
        "cycles",
        "stale_row",
    ],
)
def test_wrong_or_stale_low_level_receipt_cannot_be_bound_to_owned_scientific_row(
    field: str,
) -> None:
    model = _wrap(_policy("random"))
    model.eval()
    result = model.evaluation_forward(_rows()[0], terminal_token_id=1).teacher
    row = _rows()[0]
    if field == "mode":
        result = replace(result, mode="training")
    elif field == "policy":
        result = replace(result, policy=_policy("uniform"))
    elif field == "supervised_token_count":
        result = replace(
            result,
            traced=replace(
                result.traced,
                teacher=replace(
                    result.traced.teacher,
                    supervised_token_count=result.traced.teacher.supervised_token_count + 1,
                ),
            ),
        )
    elif field == "cycles":
        result = replace(result, traced=replace(result.traced, cycles=result.traced.cycles[:-1]))
    elif field == "stale_row":
        row = _rows()[1]
    else:
        mutation: dict[str, Any] = {field: "0" * 64}
        result = replace(result, **mutation)
    with (
        patch.object(model.teacher, "evaluation_forward_ablated", return_value=result),
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        model.evaluation_forward(row, terminal_token_id=1)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("stage", ["preparation", "probes"])
def test_runtime_restores_and_errors_redact_if_preparation_or_probe_stage_fails(
    training: bool, stage: str
) -> None:
    model = _wrap()
    if not training:
        model.eval()
    caller = capture_runtime_state(torch.device("cpu"))
    before = module_state_sha256((("model", model),))
    marker = "private-prompt-marker"

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(5)
        raise RuntimeError(marker)

    path = (
        "iris.application.scientific_ablation_teacher.prepare_teacher_forcing_input"
        if stage == "preparation"
        else "iris.application.scientific_teacher.workspace_probe_objective"
    )
    with (
        patch(path, side_effect=fail),
        pytest.raises(IrisScientificAblationTeacherError) as failure,
    ):
        method = model.training_forward if training else model.evaluation_forward
        method(_rows()[0], terminal_token_id=1)
    assert marker not in str(failure.value) and failure.value.__cause__ is None
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert before == module_state_sha256((("model", model),))


@pytest.mark.parametrize("training", [True, False])
def test_finite_probe_mutation_in_preparation_fails_before_actual_teacher(training: bool) -> None:
    model = _wrap()
    if not training:
        model.eval()
    original = model.teacher.backbone.tokenize_teacher_target
    caller = capture_runtime_state(torch.device("cpu"))

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        torch.rand(2)
        with torch.no_grad():
            next(model.probes.parameters()).add_(0.1)
        return result

    with (
        patch.object(model.teacher.backbone, "tokenize_teacher_target", side_effect=mutate),
        patch.object(model.teacher.backbone, "forward_prompt") as prompt,
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        method = model.training_forward if training else model.evaluation_forward
        method(_rows()[0], terminal_token_id=1)
    prompt.assert_not_called()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "identity",
    ["configuration_identity", "base.configuration_identity", "base.probes.configuration_identity"],
)
@pytest.mark.parametrize("strict", [True, False])
def test_loading_rejects_all_identity_mismatches_before_any_child_copy(
    identity: str, strict: bool
) -> None:
    model = _wrap()
    state = deepcopy(model.state_dict())
    state[identity][0] ^= 1
    for value in state.values():
        if value.is_floating_point():
            value.add_(0.1)
    before = module_state_sha256((("model", model),))
    with pytest.raises(IrisScientificAblationTeacherError):
        model.load_state_dict(state, strict=strict)
    assert before == module_state_sha256((("model", model),))


@pytest.mark.parametrize("source", ["different_mode", "assign", "legacy"])
def test_foreign_state_or_assign_replacement_is_rejected_without_writes(source: str) -> None:
    model = _wrap()
    state = (
        _wrap(_policy(goal="disabled")).state_dict()
        if source == "different_mode"
        else (model.base.state_dict() if source == "legacy" else model.state_dict())
    )
    before = module_state_sha256((("model", model),))
    with pytest.raises(IrisScientificAblationTeacherError):
        model.load_state_dict(state, assign=source == "assign")
    assert before == module_state_sha256((("model", model),))


def test_same_mode_state_reload_preserves_policy_registered_identity_and_actual_objective() -> None:
    source = _wrap(_policy("random", goal="disabled"))
    destination = _wrap(source.ablation)
    with torch.no_grad():
        next(source.probes.parameters()).add_(0.1)
    ids = tuple(id(p) for p in destination.parameters())
    destination.load_state_dict(source.state_dict())
    assert ids == tuple(id(p) for p in destination.parameters())
    assert module_state_sha256((("model", source),)) == module_state_sha256(
        (("model", destination),)
    )
    source.eval()
    destination.eval()
    first = source.evaluation_forward(_rows()[0], terminal_token_id=1)
    second = destination.evaluation_forward(_rows()[0], terminal_token_id=1)
    assert first.invocation_sha256 == second.invocation_sha256
    _losses_equal(first.losses, second.losses)


def test_legacy_scientific_optimizer_cannot_accept_new_ablated_graph() -> None:
    model = _wrap()
    with pytest.raises(IrisScientificOptimizationError):
        IrisScientificOptimizerV1(model=model, policy=IrisScientificOptimizerPolicyV1())  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["raw", "copied", "missing_random_seed", "wrong_base"])
def test_constructor_rejects_noncanonical_or_bypassed_inputs(invalid: str) -> None:
    base, _, _ = _build()
    policy: Any = _policy()
    owner: Any = base
    if invalid == "raw":
        policy = policy.model_dump(mode="python")
    elif invalid == "copied":
        policy = policy.model_copy(update={"goal": "foreign"})
    elif invalid == "missing_random_seed":
        policy = policy.model_copy(update={"routing": "random"})
    else:
        owner = base.teacher
    with pytest.raises(IrisScientificAblationTeacherError):
        IrisScientificAblationTeacherV1(base=owner, ablation=policy)


def test_valid_held_out_evaluation_is_no_grad_and_owned_row_remains_unchanged() -> None:
    model = _wrap(_policy("random", goal="disabled", broadcast="disabled"))
    model.eval()
    original = _rows()[0]
    held_out = IrisExample.model_validate(
        {
            **original.model_dump(mode="python"),
            "split": "validation",
            "id": f"validation-{original.seed}",
        }
    )
    before = held_out.model_dump(mode="python")
    state = module_state_sha256((("model", model),))
    measured = model.evaluation_forward(
        held_out, terminal_token_id=1, diagnostics=_diagnostics(held_out)
    )
    assert measured.example_id == held_out.id and measured.mode == "evaluation"
    assert not measured.losses.total.requires_grad and bool(torch.isfinite(measured.losses.total))
    assert measured.teacher.policy == model.ablation
    assert held_out.model_dump(mode="python") == before
    assert state == module_state_sha256((("model", model),))


def test_target_only_change_rebinds_nll_invocation_but_not_prompt_keyed_cognition_or_probes() -> (
    None
):
    model = _wrap(_policy("random"))
    model.eval()
    row = _rows()[0]
    response = json.loads(row.expected_response)
    response["content"] += " maximum"
    changed = IrisExample.model_validate(
        {**row.model_dump(mode="python"), "expected_response": json.dumps(response)}
    )
    first = model.evaluation_forward(row, terminal_token_id=1)
    second = model.evaluation_forward(changed, terminal_token_id=1)
    assert first.teacher.prompt_sha256 == second.teacher.prompt_sha256
    assert first.teacher.target_sha256 != second.teacher.target_sha256
    assert first.example_sha256 != second.example_sha256
    assert first.invocation_sha256 != second.invocation_sha256
    assert first.scientific_state_sha256 == second.scientific_state_sha256
    for a, b in zip(
        first.trace.teacher.workspace_trace, second.trace.teacher.workspace_trace, strict=True
    ):
        assert_same(a, b)
    for field in ("probe", "diversity", "utilization", "norm"):
        assert torch.equal(getattr(first.losses, field), getattr(second.losses, field))


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mutation", ["parameter", "buffer"])
def test_finite_probe_stage_parameter_or_buffer_mutation_cannot_return_result(
    training: bool, mutation: str
) -> None:
    model = _wrap()
    if not training:
        model.eval()
    caller = capture_runtime_state(torch.device("cpu"))

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = workspace_probe_objective(*args, **kwargs)
        torch.rand(2)
        with torch.no_grad():
            if mutation == "parameter":
                next(model.probes.parameters()).add_(0.1)
            else:
                model.probes.configuration_identity[0] ^= 1
        return result

    with (
        patch("iris.application.scientific_teacher.workspace_probe_objective", side_effect=mutate),
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        method = model.training_forward if training else model.evaluation_forward
        method(_rows()[0], terminal_token_id=1)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("slots,cycles", [(1, 1), (1, 5), (16, 5)])
def test_scientific_capacity_and_cycle_extremes_preserve_complete_real_objective(
    slots: int, cycles: int
) -> None:
    base, _, _ = _build(slots=slots, cycles=cycles)
    model = IrisScientificAblationTeacherV1(
        base=base, ablation=_policy("random", goal="disabled", broadcast="disabled")
    )
    row = _rows()[0]
    trained = model.training_forward(row, terminal_token_id=1)
    assert len(trained.trace.cycles) == len(trained.trace.teacher.workspace_trace) == cycles
    assert trained.usage.slot_probabilities.shape == (slots,)
    if slots == 1:
        assert trained.losses.diversity.item() == 0
        config = json.loads(model.base.configuration_json)
        assert json.loads(config["workspace_configuration_json"])["winners_per_cycle"] == 1
    torch.autograd.backward(trained.losses.total)
    assert all(p.grad is not None for p in model.probes.parameters())
    model.eval()
    evaluated = model.evaluation_forward(row, terminal_token_id=1)
    _losses_equal(trained.losses, evaluated.losses)
    assert trained.invocation_sha256 != evaluated.invocation_sha256


def test_old_baseline_output_is_not_accepted_as_a_low_level_ablation_receipt() -> None:
    model = _wrap()
    model.eval()
    ordinary = model.base.evaluation_forward(_rows()[0], terminal_token_id=1)
    with (
        patch.object(model.teacher, "evaluation_forward_ablated", return_value=ordinary),
        pytest.raises(IrisScientificAblationTeacherError),
    ):
        model.evaluation_forward(_rows()[0], terminal_token_id=1)
