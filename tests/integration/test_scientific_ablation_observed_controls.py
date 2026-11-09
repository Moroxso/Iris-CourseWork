"""One ordinary/observed EMPTY stage in genuine original source transactions.

Not the complete writer-effect executor, its byte preflight or scientific test.
"""

import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as module
from iris.application.errors import IrisRequestError
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutError
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    build_scientific_ablation_causal_design_v1,
)
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    DOMAIN,
    build_scientific_ablation_causal_observed_budget_v1,
    scientific_ablation_causal_observed_budget_bytes_v1,
)
from iris.training.ablation_observation import IrisAblationObservedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.state import WorkspaceState
from tests.integration.test_ablation_observation import _observed_json
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.unit.test_workspace_intervention_state import assert_same

pytestmark = pytest.mark.integration
Error = module.IrisScientificAblationCausalControlsError


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("source-observed-empty-controls")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


def _stage(experiment: Experiment, owner: Any, *, policy: Any = None) -> Any:
    with module._source_transaction(
        owner,
        input=_input(experiment),
        policy=_policy() if policy is None else policy,
        maximum_report_bytes=64 * 1024**2,
    ) as source:
        budget = build_scientific_ablation_causal_observed_budget_v1(
            source.design, maximum_observation_scalars=336
        )
        return module._observed_controls(source, budget=budget)


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_original_best1_both_observed_empty_stages_without_repeated_baseline(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    ex = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(ex, routing, goal=goal, broadcast=broadcast)
    context = owner._guard.context
    model, run = context._run._optimizer._model, context._run
    modes = tuple((m, m.training) for m in model.modules())
    caller = capture_runtime_state(torch.device("cpu"))
    counts = {"ordinary": 0, "observed": 0}
    actual_ordinary = IrisScientificAblationTeacherV1.evaluation_forward
    actual_observed = IrisTeacherForcingModel.evaluation_forward_ablation_observed

    def busy() -> None:
        assert owner._state == context._state == context._request._state == "busy"
        assert run._busy

    def ordinary(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["ordinary"] += 1
        busy()
        return actual_ordinary(self, *args, **kwargs)

    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["observed"] += 1
        busy()
        return actual_observed(self, *args, **kwargs)

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=ordinary),
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=observed),
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(
            IrisTeacherForcingModel, "evaluation_forward_ablation_controlled"
        ) as old_control,
        patch.object(type(context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            protocol = "message" if message else "workspace"
            with module._source_transaction(
                owner,
                input=_input(ex),
                policy=_policy(message),
                maximum_report_bytes=64 * 1024**2,
            ) as source:
                budget = build_scientific_ablation_causal_observed_budget_v1(
                    source.design, maximum_observation_scalars=336
                )
                before_calls = execute.call_count
                result = module._observed_controls(source, budget=budget)
                assert tuple(m.row.row_index for m in result) == (
                    source.design.payload.selected_row_indices
                )
                assert len(result) == (2 if message else 3)
                assert execute.call_count - before_calls == 2 * len(result)
                assert budget.payload.planned_total_teacher_executions > 2 * len(result)
                assert ex.progress.evaluation is not None and ex.progress.complete is not None
                original_source = ex.progress.evaluation.payload.plan.payload.source.payload
                assert (original_source.selected_step, original_source.terminal_step) == (1, 3)
                assert len(source.design.payload.members) == 4
                assert len(source.design.payload.groups) == 20
                rows = []
                for measurement in result:
                    p, o, row = (
                        measurement.prepared,
                        measurement.observation,
                        measurement.row,
                    )
                    assert p is not None and type(o) is IrisAblationObservedTeacherOutputV1
                    saved = ex.progress.evaluation.payload.rows[row.row_index]
                    assert p.prompt == source.seal.rows[row.row_index].prompt
                    assert (
                        teacher_target_sha256_v1(p.target_input_ids, p.target_attention_mask)
                        == row.target_sha256
                        == saved.teacher_target_sha256
                    )
                    assert row.target_token_count == saved.target_token_count
                    assert row.target_nll_sum == saved.target_nll_sum
                    assert row.ordinary_mean_target_nll == row.controlled_mean_target_nll
                    assert o.controlled.invocation_sha256 == row.controlled_invocation_sha256
                    assert o.controlled.teacher_graph_sha256 == source.controlled_graph_sha256
                    assert o.controlled.sidecar_state_sha256 == source.sidecar_sha256
                    assert o.actual_observation_scalars == o.maximum_observation_scalars == 336
                    assert len(row.messages) == (12 if message else 0)
                    o.__post_init__()
                    for state, observation in zip(
                        o.controlled.workspace_trace, o.observations, strict=True
                    ):
                        assert_same(state, observation.native_written)
                        for f in fields(WorkspaceState):
                            native, recorded = (
                                getattr(state, f.name),
                                getattr(observation.native_written, f.name),
                            )
                            assert native.untyped_storage().data_ptr() != (
                                recorded.untyped_storage().data_ptr()
                            )
                            assert not recorded.requires_grad and recorded.grad_fn is None
                            assert not recorded.is_inference()
                    rows.append(
                        dict(row=row.model_dump(mode="json"), observation=_observed_json(o))
                    )
                source.audit(measuring=True)
                busy()
                witness = dict(
                    kind="iris-tests-original-source-observed-empty-stage-v1",
                    scope="two-pass-internal-stage-not-full-effects-or-report-byte-preflight",
                    design=source.design.model_dump(mode="json"),
                    observed_budget=budget.model_dump(mode="json"),
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
            assert owner._state == context._state == context._request._state == "ready"
            assert not run._busy and not run._poisoned
            assert all(m.training is mode for m, mode in modes)
            (tmp_path / f"stage-{protocol}.json").write_bytes(canonical_json_file_bytes(witness))
            (tmp_path / f"design-{protocol}.json").write_bytes(
                canonical_json_file_bytes(source.design.model_dump(mode="json"))
            )
            (tmp_path / f"budget-{protocol}.json").write_bytes(
                scientific_ablation_causal_observed_budget_bytes_v1(budget)
            )
    for mocked in (old_control, generate, candidate, loader, tokenizer):
        mocked.assert_not_called()
    assert counts == {"ordinary": 5, "observed": 5} and execute.call_count == 10
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ("foreign", "private", "nested", "rehashed", "binding", "cap"))
def test_owned_observed_budget_denies_before_either_forward(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as ordinary,
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed") as observed,
        pytest.raises(Error),
        module._source_transaction(
            owner, input=_input(experiment), policy=_policy(), maximum_report_bytes=64 * 1024**2
        ) as source,
    ):
        budget: Any = build_scientific_ablation_causal_observed_budget_v1(
            source.design, maximum_observation_scalars=335 if kind == "cap" else 336
        )
        if kind == "foreign":
            budget = type("ForeignBudget", (), {})()
        elif kind == "private":
            object.__setattr__(budget, "__pydantic_private__", {"hidden": True})
        elif kind == "nested":
            object.__setattr__(budget.payload.design.payload, "__pydantic_private__", {"x": 1})
        elif kind == "rehashed":
            payload = budget.payload.model_copy(update={"planned_observed_teacher_executions": 1})
            budget = budget.model_copy(
                update={
                    "payload": payload,
                    "sha256": domain_sha256(
                        DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                }
            )
        elif kind == "binding":
            d = source.design.payload
            foreign_design = build_scientific_ablation_causal_design_v1(
                d.members,
                purpose=d.purpose,
                ablation=d.ablation,
                geometry=d.geometry,
                policy=_policy(selection_seed=43),
            )
            budget = build_scientific_ablation_causal_observed_budget_v1(
                foreign_design, maximum_observation_scalars=336
            )
        module._observed_controls(source, budget=budget)
    ordinary.assert_not_called()
    observed.assert_not_called()
    # This private stage is already INSIDE an entered BUSY transaction. Its
    # failures poison the original owners; no new public preflight API exists.
    assert owner._state == context._state == context._request._state == "poisoned"


@pytest.mark.parametrize(
    "kind", ("foreign", "digest", "tensor", "prepared", "earlier", "unselected")
)
def test_observed_or_retained_source_drift_never_completes_the_stage(
    experiment: Experiment, kind: str
) -> None:
    owner = _heldout(experiment)
    context = owner._guard.context
    original = IrisTeacherForcingModel.evaluation_forward_ablation_observed
    first: Any = None
    calls = 0

    def altered(self: Any, prepared: Any, **kwargs: Any) -> Any:
        nonlocal first, calls
        output = original(self, prepared, **kwargs)
        calls += 1
        if kind == "foreign":
            return type("ForeignObserved", (), {})()
        if kind == "digest":
            object.__setattr__(output, "invocation_sha256", "f" * 64)
        elif kind == "tensor":
            output.observations[0].native_written.slots.add_(1)
        elif kind == "prepared":
            prepared.target_input_ids.add_(1)
        elif kind == "earlier":
            if first is None:
                first = output
            else:
                first.observations[0].native_written.slots.add_(1)
        elif kind == "unselected":
            row = owner._guard.rows[-1]
            object.__setattr__(row, "difficulty", row.difficulty % 3 + 1)
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=altered),
        pytest.raises(Error),
    ):
        _stage(experiment, owner)
    assert calls > 0
    assert owner._state == context._state == context._request._state == "poisoned"
    assert context._run._poisoned
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_observed_first_signal_survives_cleanup_failure(
    experiment: Experiment, signal_type: Any
) -> None:
    owner = _heldout(experiment)
    signal = signal_type("first-observed-signal")

    def failed(*args: Any, **kwargs: Any) -> None:
        torch.rand(3)
        raise signal

    def restore(*args: Any, **kwargs: Any) -> None:
        _restore_caller(*args, **kwargs)
        raise RuntimeError("cleanup-failed")

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=failed),
        patch.object(module, "_restore_caller", side_effect=restore),
        pytest.raises(signal_type) as caught,
    ):
        _stage(experiment, owner)
    assert caught.value is signal
    assert owner._state == owner._guard.context._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_observed_stage_denies_reentry_but_outer_transaction_stays_ready(
    experiment: Experiment,
) -> None:
    owner = _heldout(experiment)
    request = owner._guard.context._request
    original = IrisTeacherForcingModel.evaluation_forward_ablation_observed

    def attempted(self: Any, *args: Any, **kwargs: Any) -> Any:
        with pytest.raises(IrisScientificAblationHeldoutError, match="busy"):
            owner.execute_row(0)
        with pytest.raises(IrisRequestError):
            request.execute("no generation", strategy="greedy")
        return original(self, *args, **kwargs)

    with patch.object(
        IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=attempted
    ):
        result = _stage(experiment, owner)
    assert len(result) == 2 and owner._state == request._state == "ready"


@pytest.mark.parametrize("default_device", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_observed_stage_restores_caller_and_original_modes(
    experiment: Experiment, default_device: str, inference: bool
) -> None:
    owner = _heldout(experiment)
    model = owner._guard.context._run._optimizer._model
    modes = tuple((m, m.training) for m in model.modules())
    dtype = torch.get_default_dtype()
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(default_device), torch.inference_mode(inference):
            result = _stage(experiment, owner)
            assert len(result) == 2
            assert torch.get_default_dtype() is torch.float64
            assert torch.get_default_device() == torch.device(default_device)
            assert torch.is_inference_mode_enabled() is inference
        assert all(m.training is mode for m, mode in modes)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("kind", ("best", "unselected_capture"))
def test_late_physical_source_drift_prevents_observed_stage_exit(
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
    actual = module._observed_controls
    path = (
        case.source.native_path / "best.safetensors"
        if kind == "best"
        else case.capture.path / "row-00003/row.json"
    )
    measured = False

    def late(*args: Any, **kwargs: Any) -> Any:
        nonlocal measured
        result = actual(*args, **kwargs)
        assert all(m.row.row_index != 3 for m in result)
        measured = True
        with path.open("ab") as stream:
            stream.write(b"late original physical drift")
        return result

    with patch.object(module, "_observed_controls", side_effect=late), pytest.raises(Error):
        _stage(local, owner)
    assert measured and owner._state == owner._guard.context._state == "poisoned"


@pytest.mark.parametrize("message", (False, True))
def test_observed_stage_full_ordinary_trace_matches_original_empty_observation(
    experiment: Experiment, message: bool
) -> None:
    result = _stage(
        experiment, _heldout(experiment), policy=_policy(message, ordinary_trace_mode="full")
    )
    for measurement in result:
        assert measurement.row.ordinary_trace.cycles == measurement.row.controlled_trace.cycles
