"""Genuine trained-best source/control phase, not measured writer effects or CUDA."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as module
from iris.application.errors import IrisRequestError
from iris.application.scientific_ablation_diagnostics import _teacher
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import (
    IrisScientificAblationHeldoutError,
    IrisScientificAblationHeldoutV1,
)
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_causal_controls_v1 import DOMAIN
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_diagnostics import (
    Experiment,
    _experiment,
    _heldout,
    _kwargs,
)
from tests.integration.test_scientific_ablation_evaluation_v2 import _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing

pytestmark = pytest.mark.integration
Error = module.IrisScientificAblationCausalControlsError


def _rows() -> tuple[IrisExample, ...]:
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    pair = tuple(
        row for row in pool if row.task_type == "next_tool_prediction" and row.difficulty == 1
    )[:2]
    assert len(pair) == 2
    other = tuple(
        next(row for row in pool if row.task_type == task)
        for task in ("delayed_goal", "conflict_resolution")
    )
    return (*pair, *other)


def _input(experiment: Experiment) -> module.ScientificAblationCausalInputV1:
    kwargs = _kwargs(experiment)
    kwargs.pop("policy")
    return module.ScientificAblationCausalInputV1(**kwargs)


def _policy(message: bool = True, **changes: Any) -> Any:
    cls = (
        ScientificAblationMessageCausalPolicyV1
        if message
        else ScientificAblationWorkspaceCausalPolicyV1
    )
    return cls.model_validate(
        dict(selection_seed=42, shuffle_seed=1337, examples_per_stratum=2 if message else 1)
        | changes
    )


def _measure(
    experiment: Experiment, heldout: IrisScientificAblationHeldoutV1, **changes: Any
) -> Any:
    return module.measure_scientific_ablation_causal_controls_v1(
        heldout, **(dict(input=_input(experiment), policy=_policy()) | changes)
    )


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("source-owned-causal-controls")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_actual_best1_terminal3_both_source_owned_control_protocols(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    ex = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(ex, routing, goal=goal, broadcast=broadcast)
    context = owner._guard.context
    before = capture_runtime_state(torch.device("cpu"))
    counts = {"ordinary": 0, "controlled": 0}
    actual_ordinary = IrisScientificAblationTeacherV1.evaluation_forward
    actual_controlled = IrisTeacherForcingModel.evaluation_forward_ablation_controlled

    def ordinary(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["ordinary"] += 1
        assert owner._state == context._state == context._request._state == "busy"
        assert context._run._busy
        return actual_ordinary(self, *args, **kwargs)

    def controlled(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["controlled"] += 1
        return actual_controlled(self, *args, **kwargs)

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=ordinary),
        patch.object(
            IrisTeacherForcingModel, "evaluation_forward_ablation_controlled", new=controlled
        ),
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as execute,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            result = _measure(ex, owner, policy=_policy(message))
            p = result.payload
            assert ex.progress.evaluation is not None
            assert (
                p.evaluation == ex.progress.evaluation
                and p.evaluation_complete == ex.progress.complete
            )
            source = p.evaluation.payload.plan.payload.source.payload
            assert (source.selected_step, source.terminal_step) == (1, 3)
            assert p.design.payload.ablation == source.ablation
            assert len(p.design.payload.members) == 4 and len(p.design.payload.groups) == 20
            assert len(p.rows) == (2 if message else 3)
            assert p.actual_teacher_executions == 2 * len(p.rows)
            assert p.design.payload.budget.total_executions > p.actual_teacher_executions
            for row in p.rows:
                assert (
                    row.target_nll_sum
                    == ex.progress.evaluation.payload.rows[row.row_index].target_nll_sum
                )
                assert row.ordinary_mean_target_nll == row.controlled_mean_target_nll
                assert len(row.messages) == (12 if message else 0)
            encoded = module._checked(result)
            assert len(encoded) <= p.report_byte_preflight_bound <= p.maximum_report_bytes
            for private in (
                b'"prompt"',
                b'"expected_response"',
                b'"hidden_facts"',
                b'"output_text"',
                b'"python_state"',
            ):
                assert private not in encoded
            (tmp_path / f"controls-{'message' if message else 'workspace'}.json").write_bytes(
                encoded
            )
            (tmp_path / f"design-{'message' if message else 'workspace'}.json").write_bytes(
                canonical_json_file_bytes(p.design.model_dump(mode="json"))
            )
            assert owner._state == context._state == context._request._state == "ready"
            assert not context._run._busy and not context._run._poisoned
    for mocked in (generate, execute, loader, tokenizer):
        mocked.assert_not_called()
    assert counts == {"ordinary": 5, "controlled": 5}
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "limit",
    (
        "maximum_executions",
        "maximum_trace_scalars",
        "maximum_message_scalars",
        "maximum_total_numeric_scalars",
    ),
)
def test_complete_future_experiment_budget_denies_before_either_control_forward(
    experiment: Experiment, limit: str
) -> None:
    owner = _heldout(experiment)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as ordinary,
        patch.object(
            IrisTeacherForcingModel, "evaluation_forward_ablation_controlled"
        ) as controlled,
        pytest.raises(Error),
    ):
        changes: dict[str, Any] = {limit: 1}
        _measure(experiment, owner, policy=_policy(**changes))
    ordinary.assert_not_called()
    controlled.assert_not_called()
    assert owner._state == owner._guard.context._state == "ready"


@pytest.mark.parametrize("limit", (0, True, 1, 64 * 1024**2 + 1))
def test_control_report_byte_quota_denies_before_first_forward(
    experiment: Experiment, limit: Any
) -> None:
    owner = _heldout(experiment)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as ordinary,
        pytest.raises(Error),
    ):
        _measure(experiment, owner, maximum_report_bytes=limit)
    ordinary.assert_not_called()


@pytest.mark.parametrize(
    "anchor",
    (
        "expected_evaluation_plan_sha256",
        "expected_evaluation_sha256",
        "expected_plan_sha256",
        "expected_capture_sha256",
    ),
)
def test_each_trusted_anchor_is_required_before_forward(
    experiment: Experiment, anchor: str
) -> None:
    owner = _heldout(experiment)
    changes: dict[str, Any] = {anchor: "f" * 64}
    spec = replace(_input(experiment), **changes)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, owner, input=spec)
    forward.assert_not_called()


@pytest.mark.parametrize("when", ("before", "during"))
@pytest.mark.parametrize(
    "what", ("unselected_row", "training", "validation", "weights", "gradient")
)
def test_complete_original_memory_drift_never_returns_partial_controls(
    experiment: Experiment, when: str, what: str
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    run, model = context._run, context._run._optimizer._model
    called = False

    def mutate() -> None:
        if what == "unselected_row":
            row = owner._guard.rows[-1]
            object.__setattr__(row, "difficulty", row.difficulty % 3 + 1)
        elif what in ("training", "validation"):
            row = run._rows[-1] if what == "training" else context._validation[-1]
            object.__setattr__(row, "expected_response", row.expected_response + " ")
        else:
            parameter = next(model.parameters())
            with torch.no_grad():
                if what == "weights":
                    parameter.add_(0.01)
                else:
                    parameter.grad = torch.ones_like(parameter)

    original = _teacher

    def altered(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        output = original(*args, **kwargs)
        if not called:
            called = True
            mutate()
        return output

    before = capture_runtime_state(torch.device("cpu"))
    if when == "before":
        mutate()
    with patch.object(module, "_teacher", side_effect=altered), pytest.raises(Error):
        _measure(experiment, owner)
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    if when == "during":
        assert called, "A late mutation must actually enter its original teacher pass"
        assert owner._state == context._state == context._request._state == "poisoned"
        assert run._poisoned


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_first_signal_identity_survives_cleanup_failure_and_poison(
    experiment: Experiment, signal_type: Any
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    signal = signal_type("private-first-signal")
    original_restore = _restore_caller

    def fail(*args: Any, **kwargs: Any) -> None:
        torch.rand(3)
        raise signal

    def restore(*args: Any, **kwargs: Any) -> None:
        original_restore(*args, **kwargs)
        raise RuntimeError("private-cleanup-failure")

    before = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(module, "_teacher", side_effect=fail),
        patch.object(module, "_restore_caller", side_effect=restore),
        pytest.raises(signal_type) as caught,
    ):
        _measure(experiment, owner)
    assert caught.value is signal
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    assert owner._state == context._state == context._request._state == "poisoned"


def test_reentry_denies_heldout_and_request_without_poisoning_outer_measurement(
    experiment: Experiment,
) -> None:
    owner = _heldout(experiment)
    request = owner._guard.context._request
    original = _teacher

    def attempted(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(IrisScientificAblationHeldoutError, match="busy"):
            owner.execute_row(0)
        with pytest.raises(IrisRequestError):
            request.execute("must not generate", strategy="greedy")
        return original(*args, **kwargs)

    with patch.object(module, "_teacher", side_effect=attempted):
        result = _measure(experiment, owner)
    assert len(result.payload.rows) == 2
    assert owner._state == request._state == "ready"


@pytest.mark.parametrize("kind", ("optimizer", "request_owner", "context_run"))
def test_foreign_substitution_poison_targets_only_original_sealed_owners(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    run, optimizer, request = context._run, context._run._optimizer, context._request
    original_request_owner = request._owner
    foreign = type("Foreign", (), {"_poisoned": False})()
    original = _teacher

    def changed(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        if kind == "optimizer":
            run._optimizer = foreign
        elif kind == "request_owner":
            request._owner = foreign
        else:
            context._run = foreign
        return output

    with patch.object(module, "_teacher", side_effect=changed), pytest.raises(Error):
        _measure(experiment, owner)
    assert owner._state == context._state == request._state == "poisoned"
    assert run._poisoned and optimizer._poisoned and not foreign._poisoned
    assert original_request_owner._state == "poisoned"


@pytest.mark.parametrize(
    "field",
    (
        "ordinary_mean_target_nll",
        "controlled_mean_target_nll",
        "controlled_invocation_sha256",
        "evaluation_ack_sha256",
        "both_mean_target_nll",
    ),
)
def test_fully_rehashed_wrong_control_row_still_fails(experiment: Experiment, field: str) -> None:
    result = _measure(experiment, _heldout(experiment))
    raw = result.model_dump(mode="json")
    if field == "both_mean_target_nll":
        raw["payload"]["rows"][0]["ordinary_mean_target_nll"] = 0.0
        raw["payload"]["rows"][0]["controlled_mean_target_nll"] = 0.0
    else:
        raw["payload"]["rows"][0][field] = 0.0 if field.endswith("nll") else "f" * 64
    raw["sha256"] = domain_sha256(DOMAIN, canonical_json_bytes(raw["payload"]))
    with pytest.raises(ValueError):
        type(result).model_validate_json(json.dumps(raw))


def test_late_ordinary_failure_redacts_public_message_and_restores_modes(
    experiment: Experiment,
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    model = context._run._optimizer._model
    modes = tuple((m, m.training) for m in model.modules())
    with (
        patch.object(module, "_teacher", side_effect=RuntimeError("private candidate text")),
        pytest.raises(Error) as caught,
    ):
        _measure(experiment, owner)
    assert "private" not in str(caught.value)
    assert all(m.training is mode for m, mode in modes)


@pytest.mark.parametrize(
    "kind",
    (
        "test",
        "train",
        "validation",
        "best",
        "unselected_capture",
        "unselected_score",
        "unselected_ack",
        "score_complete",
    ),
)
def test_late_complete_physical_source_or_unselected_chain_drift_is_not_partial_success(
    experiment: Experiment, tmp_path: Path, kind: str
) -> None:
    case = _copy(experiment.case, tmp_path / "case")
    ledger = (
        tmp_path
        / "scores/scientific-ablation-evaluation-ledger-v2"
        / experiment.progress.plan.sha256
    )
    shutil.copytree(experiment.progress.path, ledger)
    local = replace(experiment, case=case, progress=replace(experiment.progress, path=ledger))
    owner = _heldout(local)
    context = owner._guard.context
    actual_controls = module._controls
    paths = {
        "test": case.root / "test.parquet",
        "train": case.root / "train.parquet",
        "validation": case.root / "validation.parquet",
        "best": case.source.native_path / "best.safetensors",
        "unselected_capture": case.capture.path / "row-00003/row.json",
        "unselected_score": ledger / "score-00003/row.json",
        "unselected_ack": ledger / "score-00003/ack.json",
        "score_complete": ledger / "complete/complete.json",
    }
    measured = False

    def late(*args: Any, **kwargs: Any) -> Any:
        nonlocal measured
        rows = actual_controls(*args, **kwargs)
        assert len(rows) == 2 and all(row.row_index != 3 for row in rows)
        measured = True
        with paths[kind].open("ab") as stream:
            stream.write(b"late original source drift")
        return rows

    caller = capture_runtime_state(torch.device("cpu"))
    with patch.object(module, "_controls", side_effect=late), pytest.raises(Error):
        _measure(local, owner)
    assert measured
    assert owner._state == context._state == context._request._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "kind", ("mode", "dtype", "private_rng", "request_busy", "heldout_busy", "run_busy")
)
def test_late_measurement_state_cannot_be_hidden_by_restoration(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    model, run = context._run._optimizer._model, context._run
    modes = tuple((m, m.training) for m in model.modules())
    dtype = torch.get_default_dtype()
    caller = capture_runtime_state(torch.device("cpu"))
    actual_controls = module._controls
    measured = False

    def late(*args: Any, **kwargs: Any) -> Any:
        nonlocal measured
        rows = actual_controls(*args, **kwargs)
        measured = True
        if kind == "mode":
            next(model.modules()).training = True
        elif kind == "dtype":
            torch.set_default_dtype(torch.float64)
        elif kind == "private_rng":
            torch.rand(3)
            run._runtime = capture_runtime_state(torch.device("cpu"))
        elif kind == "request_busy":
            context._request._state = "ready"
        elif kind == "heldout_busy":
            owner._state = "ready"
        else:
            run._busy = False
        return rows

    with patch.object(module, "_controls", side_effect=late), pytest.raises(Error):
        _measure(experiment, owner)
    assert measured
    assert owner._state == context._state == context._request._state == "poisoned"
    assert all(m.training is mode for m, mode in modes)
    assert torch.get_default_dtype() is dtype
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("default_device", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_caller_inference_default_device_dtype_rng_and_repeatability_are_preserved(
    experiment: Experiment, default_device: str, inference: bool
) -> None:
    owner = _heldout(experiment)
    dtype = torch.get_default_dtype()
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(default_device), torch.inference_mode(inference):
            first = _measure(experiment, owner)
            second = _measure(experiment, owner)
            assert first == second
            assert torch.get_default_dtype() is torch.float64
            assert torch.get_default_device() == torch.device(default_device)
            assert torch.is_inference_mode_enabled() is inference
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize(
    "kind", ("policy_private", "spec_private", "foreign_input", "foreign_report")
)
def test_exact_closed_source_control_boundaries_reject_hidden_or_foreign_objects(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    policy, spec = _policy(), _input(experiment)
    if kind == "policy_private":
        object.__setattr__(policy, "__pydantic_private__", {"secret": "hidden"})
    elif kind == "spec_private":
        spec = replace(spec, test=spec.test.model_copy(deep=True))
        object.__setattr__(spec.test, "__pydantic_private__", {"secret": "hidden"})
    elif kind == "foreign_input":
        spec = type("ForeignInput", (), {})()
    else:
        foreign: Any = type("ForeignReport", (), {})()
        with pytest.raises(ValueError, match="root"):
            module._checked(foreign)
        return
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as ordinary,
        patch.object(
            IrisTeacherForcingModel, "evaluation_forward_ablation_controlled"
        ) as controlled,
        pytest.raises(Error),
    ):
        _measure(experiment, owner, input=spec, policy=policy)
    ordinary.assert_not_called()
    controlled.assert_not_called()


@pytest.mark.parametrize("kind", ("full_vector", "loss"))
def test_actual_control_receipt_must_match_original_tensors_and_loss(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    actual = IrisTeacherForcingModel.evaluation_forward_ablation_controlled
    called = False

    def altered(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal called
        output = actual(self, *args, **kwargs)
        called = True
        if kind == "loss":
            return replace(output, loss=output.loss + 1)
        cycle = output.diagnostic_trace.cycles[0]
        assert cycle.full is not None
        vectors = list(cycle.full.workspace_slots)
        vectors[0] = tuple(x + 1 for x in vectors[0])
        full = cycle.full.model_copy(update={"workspace_slots": tuple(vectors)})
        first = cycle.model_copy(update={"full": full})
        trace = output.diagnostic_trace.model_copy(
            update={"cycles": (first, *output.diagnostic_trace.cycles[1:])}
        )
        return replace(output, diagnostic_trace=trace)

    with (
        patch.object(
            IrisTeacherForcingModel, "evaluation_forward_ablation_controlled", new=altered
        ),
        pytest.raises(Error),
    ):
        _measure(experiment, owner)
    assert called
    assert owner._state == owner._guard.context._state == "poisoned"
