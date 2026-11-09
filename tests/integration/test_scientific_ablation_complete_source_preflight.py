"""Native source reservation BEFORE BUSY, not full writer effects or science."""

import json
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as module
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
    scientific_ablation_causal_observed_budget_bytes_v1,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from iris.workspace.state import WorkspaceState
from tests.integration.test_ablation_observation import _observed_json
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration
Limits = module.ScientificAblationCausalCompleteLimitsV1


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("complete-source-preflight")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


def transaction(
    experiment: Experiment, owner: IrisScientificAblationHeldoutV1, **changes: Any
) -> Any:
    limits = changes.pop("complete_limits", Limits(maximum_observation_scalars=336))
    return module._source_transaction(
        owner,
        **(
            dict(
                input=_input(experiment),
                policy=_policy(),
                maximum_report_bytes=layout.MAX_REPORT_BYTES,
                complete_limits=limits,
            )
            | changes
        ),
    )


def ready(owner: IrisScientificAblationHeldoutV1) -> None:
    context = owner._guard.context
    assert owner._state == context._state == context._request._state == "ready"
    assert not context._run._busy and not context._run._poisoned
    assert not context._run._optimizer._poisoned


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_complete_original_source_preflights_ready_before_both_stages(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    ex = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(ex, routing, goal=goal, broadcast=broadcast)
    context, run = owner._guard.context, owner._guard.context._run
    model = run._optimizer._model
    modes = tuple((m, m.training) for m in model.modules())
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    actual_budget = build_scientific_ablation_causal_observed_budget_v1
    actual_layout = layout.build_scientific_ablation_causal_report_byte_preflight_v1
    actual_ordinary = IrisScientificAblationTeacherV1.evaluation_forward
    actual_observed = IrisTeacherForcingModel.evaluation_forward_ablation_observed
    preflight_calls: list[str] = []
    before_observations: list[dict[str, Any]] = []
    stage_calls = {"ordinary": 0, "observed": 0}

    def before(name: str) -> None:
        ready(owner)
        assert all(m.training is state for m, state in modes)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert _same_runtime(private, run._runtime)
        preflight_calls.append(name)
        before_observations.append(
            dict(
                stage=name,
                states=[owner._state, context._state, context._request._state],
                run_busy=run._busy,
                original_modes_restored=all(m.training is state for m, state in modes),
                caller_rng_unchanged=_same_runtime(
                    caller, capture_runtime_state(torch.device("cpu"))
                ),
                private_rng_unchanged=_same_runtime(private, run._runtime),
            )
        )

    def numeric(*args: Any, **kwargs: Any) -> Any:
        before("numeric")
        with patch.object(model.teacher, "_execute", side_effect=AssertionError("NO_FORWARD")):
            return actual_budget(*args, **kwargs)

    def bytes_before_busy(*args: Any, **kwargs: Any) -> Any:
        before("bytes")
        with patch.object(model.teacher, "_execute", side_effect=AssertionError("NO_FORWARD")):
            return actual_layout(*args, **kwargs)

    def busy_stage(name: str) -> None:
        assert owner._state == context._state == context._request._state == "busy"
        assert run._busy and not any(m.training for m in model.modules())
        stage_calls[name] += 1

    def ordinary(self: Any, *args: Any, **kwargs: Any) -> Any:
        busy_stage("ordinary")
        return actual_ordinary(self, *args, **kwargs)

    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        busy_stage("observed")
        return actual_observed(self, *args, **kwargs)

    with (
        patch.object(module, "build_scientific_ablation_causal_observed_budget_v1", new=numeric),
        patch.object(
            module,
            "build_scientific_ablation_causal_report_byte_preflight_v1",
            new=bytes_before_busy,
        ),
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=ordinary),
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=observed),
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_controlled") as old,
        patch.object(type(context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            protocol = "message" if message else "workspace"
            with transaction(ex, owner, policy=_policy(message)) as source:
                assert owner._state == context._state == context._request._state == "busy"
                assert run._busy and not any(m.training for m in model.modules())
                pf = source.complete_preflight
                assert pf is not None and source.complete_limits is not None
                assert pf.observed_budget.payload.design == source.design
                assert pf.maximum_report_bytes == layout.MAX_REPORT_BYTES
                assert source.report_byte_bound == pf.report_byte_preflight_bound
                assert len(source.design.payload.members) == 4
                assert len(source.design.payload.groups) == 20
                assert ex.progress.evaluation is not None and ex.progress.complete is not None
                original = ex.progress.evaluation.payload.plan.payload.source.payload
                assert (original.selected_step, original.terminal_step) == (1, 3)
                descriptor = layout._descriptor(pf._shape)
                root = dict(pf._shape.members)
                root_payload = root["payload"]
                assert isinstance(root_payload, layout._Object)
                payload = dict(root_payload.members)
                fixed_source = payload["source_metadata"]
                assert isinstance(fixed_source, layout._Fixed)
                metadata = json.loads(fixed_source.encoded)
                reserved = dict(
                    kind="iris-tests-original-complete-source-before-busy-v1",
                    scope="trusted-ready-numeric-and-json-reservation-not-full-effects",
                    actual_preflight_teacher_executions=0,
                    before_observations=before_observations[-2:],
                    complete_layout=descriptor,
                    source_metadata=metadata,
                    **{
                        f.name: getattr(pf, f.name)
                        for f in fields(pf)
                        if f.name.endswith("_bound") or f.name.endswith("_bytes")
                    },
                    layout_sha256=pf.layout_sha256,
                    scientific_acceptance="not_assessed",
                    full_r015_complete=False,
                )
                before_calls = execute.call_count
                result = module._observed_controls(source, budget=pf.observed_budget)
                assert len(result) == (2 if message else 3)
                assert execute.call_count - before_calls == 2 * len(result)
                assert pf.observed_budget.payload.planned_total_teacher_executions == (
                    (10 if message else 8) * len(result)
                )
                rows = []
                for measurement in result:
                    o, row = measurement.observation, measurement.row
                    assert o is not None and measurement.prepared is not None
                    o.__post_init__()
                    assert o.actual_observation_scalars == 336
                    for state, observation in zip(
                        o.controlled.workspace_trace, o.observations, strict=True
                    ):
                        assert_same(state, observation.native_written)
                        for f in fields(WorkspaceState):
                            a, b = (
                                getattr(state, f.name),
                                getattr(observation.native_written, f.name),
                            )
                            assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                    rows.append(
                        dict(row=row.model_dump(mode="json"), observation=_observed_json(o))
                    )
                stage = dict(
                    kind="iris-tests-original-source-observed-empty-stage-v1",
                    scope="two-pass-internal-stage-not-full-effects-or-report-byte-preflight",
                    design=source.design.model_dump(mode="json"),
                    observed_budget=pf.observed_budget.model_dump(mode="json"),
                    evaluation_plan=ex.progress.plan.model_dump(mode="json"),
                    evaluation=ex.progress.evaluation.model_dump(mode="json"),
                    evaluation_complete=ex.progress.complete.model_dump(mode="json"),
                    run_seed=source.seed,
                    actual_ordinary_executions=len(result),
                    actual_observed_executions=len(result),
                    actual_teacher_executions=execute.call_count - before_calls,
                    rows=rows,
                    scientific_acceptance="not_assessed",
                    full_r015_complete=False,
                )
            ready(owner)
            assert all(m.training is state for m, state in modes)
            for name, value in (("stage", stage), ("preflight", reserved)):
                (tmp_path / f"{name}-{protocol}.json").write_bytes(canonical_json_file_bytes(value))
            (tmp_path / f"design-{protocol}.json").write_bytes(
                canonical_json_file_bytes(source.design.model_dump(mode="json"))
            )
            (tmp_path / f"budget-{protocol}.json").write_bytes(
                scientific_ablation_causal_observed_budget_bytes_v1(pf.observed_budget)
            )
    assert execute.call_count == 10 and preflight_calls == ["numeric", "bytes"] * 2
    assert stage_calls == {"ordinary": 5, "observed": 5}
    for mocked in (old, generate, candidate, loader, tokenizer):
        mocked.assert_not_called()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, run._runtime)


@pytest.mark.parametrize("message", (False, True))
@pytest.mark.parametrize("kind", ("bytes", "observation", "numeric", "cap_mismatch", "legacy_cap"))
def test_complete_quota_denial_leaves_original_ready_before_any_forward(
    experiment: Experiment, message: bool, kind: str
) -> None:
    owner = _heldout(experiment)
    run = owner._guard.context._run
    model = run._optimizer._model
    modes = tuple((m, m.training) for m in model.modules())
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    changes: dict[str, Any] = dict(policy=_policy(message))
    if kind == "bytes":
        changes.update(complete_limits=Limits(336, 1), maximum_report_bytes=1)
    elif kind == "observation":
        changes["complete_limits"] = Limits(335)
    elif kind == "numeric":
        with transaction(experiment, owner, **changes) as source:
            assert source.complete_preflight is not None
            b = source.complete_preflight.observed_budget.payload
            cap = b.total_trace_proposal_observation_scalar_bound - 1
        changes["policy"] = _policy(message, maximum_total_numeric_scalars=cap)
    elif kind == "cap_mismatch":
        changes["maximum_report_bytes"] = 64 * 1024**2
    else:
        changes["complete_limits"] = None
    with (
        patch.object(
            model.teacher, "_execute", side_effect=AssertionError("NO_FORWARD")
        ) as execute,
        pytest.raises(module.IrisScientificAblationCausalControlsError),
        transaction(experiment, owner, **changes),
    ):
        pytest.fail("Insufficient original complete quota must not enter BUSY")
    execute.assert_not_called()
    ready(owner)
    assert all(m.training is state for m, state in modes)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, run._runtime)


@pytest.mark.parametrize("message", (False, True))
def test_complete_exact_byte_cap_and_one_below_without_forward(
    experiment: Experiment, message: bool
) -> None:
    owner = _heldout(experiment)
    with transaction(experiment, owner, policy=_policy(message)) as source:
        initial = source.report_byte_bound
    # The quota's JSON decimal length is itself reserved. Iterate to the smallest
    # stable exact cap; this is a bounded metadata calculation, not a model pass.
    for _ in range(4):
        with transaction(
            experiment,
            owner,
            policy=_policy(message),
            complete_limits=Limits(336, initial),
            maximum_report_bytes=initial,
        ) as source:
            bound = source.report_byte_bound
        if bound == initial:
            break
        initial = bound
    assert bound == initial
    model = owner._guard.context._run._optimizer._model
    with patch.object(model.teacher, "_execute") as execute:
        with transaction(
            experiment,
            owner,
            policy=_policy(message),
            complete_limits=Limits(336, initial),
            maximum_report_bytes=initial,
        ) as source:
            assert source.report_byte_bound == initial
        with (
            pytest.raises(module.IrisScientificAblationCausalControlsError),
            transaction(
                experiment,
                owner,
                policy=_policy(message),
                complete_limits=Limits(336, initial - 1),
                maximum_report_bytes=initial - 1,
            ),
        ):
            pytest.fail("One byte below cannot shrink the original design")
        execute.assert_not_called()
    ready(owner)


@pytest.mark.parametrize(
    "kind",
    ("drop", "replace", "bound", "seed", "graph", "sidecar", "geometry", "budget", "foreign"),
)
def test_late_complete_preflight_or_binding_drift_cannot_exit(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    model = owner._guard.context._run._optimizer._model
    with (
        patch.object(model.teacher, "_execute") as execute,
        pytest.raises(module.IrisScientificAblationCausalControlsError),
        transaction(experiment, owner) as source,
    ):
        pf = source.complete_preflight
        assert pf is not None
        if kind == "drop":
            object.__setattr__(source, "complete_limits", None)
            object.__setattr__(source, "complete_preflight", None)
        elif kind == "replace":
            object.__setattr__(source, "complete_preflight", replace(pf))
        elif kind == "bound":
            object.__setattr__(source, "report_byte_bound", source.report_byte_bound - 1)
        elif kind in ("seed", "graph", "sidecar"):
            name = {
                "seed": "seed",
                "graph": "controlled_graph_sha256",
                "sidecar": "sidecar_sha256",
            }[kind]
            object.__setattr__(source, name, source.seed + 1 if kind == "seed" else "f" * 64)
        elif kind == "geometry":
            altered = source.design.payload.geometry.model_copy(update={"state_dim": 16})
            object.__setattr__(source.design.payload, "geometry", altered)
        elif kind == "budget":
            object.__setattr__(pf.observed_budget.payload, "maximum_observation_scalars", 337)
        else:

            class Foreign:
                def __getattribute__(self, name: str) -> Any:
                    pytest.fail("Foreign preflight callback must not execute")

            object.__setattr__(source, "complete_preflight", Foreign())
    execute.assert_not_called()
    assert owner._state == owner._guard.context._state == "poisoned"


@pytest.mark.parametrize("kind", ("old_stage", "other_budget"))
def test_complete_stage_denies_unobserved_or_different_budget_before_forward(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    model = owner._guard.context._run._optimizer._model
    with (
        patch.object(model.teacher, "_execute") as execute,
        pytest.raises(module.IrisScientificAblationCausalControlsError),
        transaction(experiment, owner) as source,
    ):
        if kind == "old_stage":
            module._controls(source)
        else:
            budget = build_scientific_ablation_causal_observed_budget_v1(
                source.design, maximum_observation_scalars=337
            )
            module._observed_controls(source, budget=budget)
    execute.assert_not_called()
    assert owner._state == "poisoned"


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("point", ("numeric", "bytes"))
@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_preflight_first_signal_survives_failed_caller_cleanup(
    experiment: Experiment, signal_type: Any, point: str, cleanup_fails: bool
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    run, original_optimizer, original_request = (
        context._run,
        context._run._optimizer,
        context._request,
    )
    signal = signal_type("first-preflight-signal")
    caller = capture_runtime_state(torch.device("cpu"))

    def failed(*args: Any, **kwargs: Any) -> None:
        ready(owner)
        torch.rand(3)
        raise signal

    def restore(*args: Any, **kwargs: Any) -> None:
        _restore_caller(*args, **kwargs)
        if cleanup_fails:
            raise RuntimeError("cleanup-failed")

    role = "observed_budget" if point == "numeric" else "report_byte_preflight"
    name = f"build_scientific_ablation_causal_{role}_v1"
    with (
        patch.object(module, name, new=failed),
        patch.object(module, "_restore_caller", new=restore),
        patch.object(original_optimizer._model.teacher, "_execute") as execute,
        pytest.raises(signal_type) as caught,
        transaction(experiment, owner),
    ):
        pytest.fail("Interrupted preflight must not yield")
    assert caught.value is signal
    if cleanup_fails:
        assert owner._state == context._state == original_request._state == "poisoned"
        assert run._poisoned and original_optimizer._poisoned and not run._busy
    else:
        ready(owner)
    execute.assert_not_called()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("default_device", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_preflight_preserves_original_modes_grad_rng_and_caller_contexts(
    experiment: Experiment, default_device: str, inference: bool
) -> None:
    owner = _heldout(experiment)
    model = owner._guard.context._run._optimizer._model
    modes = tuple((m, m.training) for m in model.modules())
    gradients = tuple((p, p.grad) for p in model.parameters())
    dtype = torch.get_default_dtype()
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(default_device), torch.inference_mode(inference), torch.no_grad():
            with transaction(experiment, owner) as source:
                assert source.complete_preflight is not None
                assert torch.get_default_dtype() is torch.float32
            assert torch.get_default_dtype() is torch.float64
            assert torch.get_default_device() == torch.device(default_device)
            assert torch.is_inference_mode_enabled() is inference
            assert not torch.is_grad_enabled()
        ready(owner)
        assert all(m.training is state for m, state in modes)
        assert all(p.grad is grad for p, grad in gradients)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("kind", ("optimizer", "request", "run"))
def test_failed_preflight_cleanup_poisons_only_captured_original_owners(
    experiment: Experiment, kind: str
) -> None:
    owner, foreign = _heldout(experiment), _heldout(experiment)
    context, foreign_context = owner._guard.context, foreign._guard.context
    original_run, original_request = context._run, context._request
    original_optimizer = original_run._optimizer

    def failed(*args: Any, **kwargs: Any) -> None:
        ready(owner)
        if kind == "optimizer":
            original_run._optimizer = foreign_context._run._optimizer
        elif kind == "request":
            context._request = foreign_context._request
        else:
            context._run = foreign_context._run
        raise RuntimeError("preflight-failed-after-foreign-substitution")

    def restore(*args: Any, **kwargs: Any) -> None:
        _restore_caller(*args, **kwargs)
        raise RuntimeError("cleanup-failed")

    with (
        patch.object(module, "build_scientific_ablation_causal_observed_budget_v1", new=failed),
        patch.object(module, "_restore_caller", new=restore),
        patch.object(original_optimizer._model.teacher, "_execute") as execute,
        pytest.raises(module.IrisScientificAblationCausalControlsError),
        transaction(experiment, owner),
    ):
        pytest.fail("Failed original preflight must not yield")
    execute.assert_not_called()
    assert owner._state == context._state == original_request._state == "poisoned"
    assert original_run._poisoned and original_optimizer._poisoned and not original_run._busy
    ready(foreign)


@pytest.mark.parametrize("kind", ("best", "unselected_capture"))
def test_complete_preflight_cannot_hide_late_original_physical_drift(
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
    path = (
        case.source.native_path / "best.safetensors"
        if kind == "best"
        else case.capture.path / "row-00003/row.json"
    )
    model = owner._guard.context._run._optimizer._model
    with (
        patch.object(model.teacher, "_execute") as execute,
        pytest.raises(module.IrisScientificAblationCausalControlsError),
        transaction(local, owner) as source,
    ):
        assert source.complete_preflight is not None
        assert 3 not in source.design.payload.selected_row_indices
        with path.open("ab") as stream:
            stream.write(b"late original physical drift")
    execute.assert_not_called()
    assert owner._state == owner._guard.context._state == "poisoned"
