"""Actual ordinary native-best diagnostic passes; never candidate execution on host."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_diagnostics as module
from iris.application import scientific_ablation_evaluation_v2 as scoring
from iris.application.scientific_ablation_capture import (
    _ack as _capture_ack,
)
from iris.application.scientific_ablation_capture import (
    _attempt as _capture_attempt,
)
from iris.application.scientific_ablation_capture import (
    _complete as _capture_complete,
)
from iris.application.scientific_ablation_capture import (
    _file as _capture_file,
)
from iris.application.scientific_ablation_capture import (
    _genesis as _capture_genesis,
)
from iris.application.scientific_ablation_capture import (
    read_scientific_ablation_capture_v1,
)
from iris.application.scientific_ablation_evaluation_ledger import (
    IrisScientificAblationEvaluationProgressV2,
    read_scientific_ablation_evaluation_progress_v2,
)
from iris.application.scientific_ablation_evaluation_v2 import _finish, _prepare
from iris.application.scientific_ablation_generation_context import _restore_caller
from iris.application.scientific_ablation_heldout import (
    IrisScientificAblationHeldoutV1,
    _audit_original,
    build_scientific_ablation_heldout_v1,
)
from iris.application.scientific_ablation_teacher import (
    IrisScientificAblationTeacherOutputV1,
    IrisScientificAblationTeacherV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_diagnostics_v1 import (
    IrisScientificAblationDiagnosticsV1,
    ScientificAblationDiagnosticPolicyV1,
)
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutRowV1
from iris.training.probes import prepare_probe_targets, probe_vocabulary_sha256
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import _scalar_bound, summarize_workspace_traces_v1
from tests.integration.test_scientific_ablation_evaluation_ledger import _arguments, _plan, _run
from tests.integration.test_scientific_ablation_evaluation_v2 import Case, _copy, _make
from tests.integration.test_scientific_ablation_generation_source import Routing, _read
from tests.integration.test_scientific_ablation_heldout import _context
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
Error = module.IrisScientificAblationDiagnosticsError
POLICY = ScientificAblationDiagnosticPolicyV1()


@dataclass(frozen=True)
class Experiment:
    case: Case
    progress: IrisScientificAblationEvaluationProgressV2


def _experiment(case: Case, root: Path) -> Experiment:
    plan = _plan(case)
    progress = _run(case, plan, root)
    assert progress.evaluation is not None and progress.complete is not None
    return Experiment(case, progress)


def _heldout(
    experiment: Experiment,
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
) -> IrisScientificAblationHeldoutV1:
    case = experiment.case
    context = _context(_read(case.source), routing=routing, goal=goal, broadcast=broadcast)
    return build_scientific_ablation_heldout_v1(
        context,
        expected_reconstruction_sha256=context.reconstruction.sha256,
        test=case.test,
        base=case.root,
        decoding=case.capture.plan.payload.input.payload.decoding,
    )


def _kwargs(experiment: Experiment) -> dict[str, Any]:
    values = _arguments(experiment.case)
    values.pop("python_policy")
    assert experiment.progress.evaluation is not None
    return {
        **values,
        "evaluation_path": experiment.progress.path,
        "expected_evaluation_plan_sha256": experiment.progress.plan.sha256,
        "expected_evaluation_sha256": experiment.progress.evaluation.sha256,
        "policy": POLICY,
    }


def _measure(
    experiment: Experiment,
    heldout: IrisScientificAblationHeldoutV1,
    **changes: Any,
) -> IrisScientificAblationDiagnosticsV1:
    return module.measure_scientific_ablation_diagnostics_v1(
        heldout, **{**_kwargs(experiment), **changes}
    )


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("actual-native-diagnostics-source")
    return _experiment(_make(root / "case"), root / "scores")


def test_actual_four_family_best_c3_trace_nll_probe_and_complete_chain(
    experiment: Experiment,
    tmp_path: Path,
) -> None:
    heldout = _heldout(experiment)
    model = heldout._guard.context._run._optimizer._model
    actual: list[IrisScientificAblationTeacherOutputV1] = []
    original = IrisScientificAblationTeacherV1.evaluation_forward
    module_init = torch.nn.Module.__init__
    temporary_modules: list[str] = []

    def initialize(self: torch.nn.Module) -> None:
        temporary_modules.append(type(self).__name__)
        # Qwen's existing decoder slices its ModuleList twice per teacher call.
        # Permit only these containers, never another parameter-owning graph.
        assert type(self) is torch.nn.ModuleList, type(self).__name__
        module_init(self)

    def forward(
        self: IrisScientificAblationTeacherV1, *args: Any, **kwargs: Any
    ) -> IrisScientificAblationTeacherOutputV1:
        output = original(self, *args, **kwargs)
        actual.append(output)
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=forward),
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as generate,
        patch.object(scoring, "score_synthetic_answer_v2") as score,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as execute,
        patch.object(torch.nn.Module, "__init__", new=initialize),
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        try:
            result = _measure(experiment, heldout)
        except Error as error:
            chain = []
            current: BaseException | None = error
            while current is not None:
                chain.append((type(current).__name__, str(current)))
                current = current.__cause__ or current.__context__
            raise AssertionError((chain, temporary_modules)) from error
    generate.assert_not_called()
    score.assert_not_called()
    execute.assert_not_called()
    loader.assert_not_called()
    tokenizer.assert_not_called()
    p = result.payload
    assert len(actual) == p.teacher_executions == 4
    assert temporary_modules == ["ModuleList"] * 8
    assert p.evaluation == experiment.progress.evaluation
    assert p.evaluation_complete == experiment.progress.complete
    source = p.evaluation.payload.plan.payload.source.payload
    assert (source.selected_step, source.terminal_step) == (1, 3)
    assert p.probe_class_counts == (4, 12, 2, 4)
    assert p.workspace == summarize_workspace_traces_v1(tuple(row.trace for row in p.rows))
    for index, (witness, output, original_row) in enumerate(
        zip(p.rows, actual, heldout._guard.rows, strict=True)
    ):
        assert witness.row_index == index
        assert witness.trace == output.teacher.traced.teacher.diagnostic_trace
        assert witness.trace.cycle_count == 3
        assert all(cycle.full is None for cycle in witness.trace.cycles)
        assert witness.trace.intervention_plan_sha256 == source.ablation_policy_sha256
        assert witness.target_nll_sum == float(output.losses.language_model)
        assert witness.evaluation_row_sha256 == experiment.progress.rows[index].sha256
        assert witness.evaluation_ack_sha256 == experiment.progress.acknowledgements[index].sha256
        targets = prepare_probe_targets(
            (original_row,),
            vocabulary=model.probes.vocabulary,
            purpose="analysis",
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            logits = model.probes(output.teacher.traced.teacher.workspace_trace[-1].slots)
        for probe in witness.probes:
            target = int(targets.labels[probe.field][0])
            assert probe.predicted_class == int(logits[probe.field].argmax(dim=1)[0])
            assert probe.correct == (probe.predicted_class == target)
            assert probe.target_sha256 == domain_sha256(
                "iris/scientific-ablation-probe-target/v1",
                canonical_json_bytes(
                    {
                        "example_sha256": output.example_sha256,
                        "vocabulary_sha256": probe_vocabulary_sha256(model.probes.vocabulary),
                        "field": probe.field,
                        "class_index": target,
                    }
                ),
            )
    assert p.trace_scalar_bound == 4 * _scalar_bound(4, 4, 8, 3, False)
    assert p.probe_correct_counts == tuple(
        sum(row.probes[i].correct for row in p.rows) for i in range(4)
    )
    assert p.source_20_4 == "not_assessed_smoke"
    assert p.scientific_acceptance == "not_assessed"
    encoded = module._checked(result)
    (tmp_path / "diagnostics.json").write_bytes(encoded)
    for private in (
        b'"prompt"',
        b'"hidden_facts"',
        b'"output_text"',
        b'"expected_response"',
        b'"python_state"',
    ):
        assert private not in encoded
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert heldout.manifest == experiment.case.capture.plan.payload.input


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_mode_c3_earlier_best_bound_ordinary_pass(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    case = _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:])
    experiment = _experiment(case, tmp_path / "scores")
    heldout = _heldout(experiment, routing, goal=goal, broadcast=broadcast)
    caller = capture_runtime_state(torch.device("cpu"))
    result = _measure(experiment, heldout)
    p = result.payload
    assert p.teacher_executions == 1 and p.rows[0].trace.cycle_count == 3
    assert p.evaluation.payload.plan.payload.source.payload.ablation.routing == routing
    assert p.evaluation.payload.plan.payload.source.payload.ablation.goal == goal
    assert p.evaluation.payload.plan.payload.source.payload.ablation.broadcast == broadcast
    assert p.rows[0].target_nll_sum == case.capture.rows[0].payload.target_nll_sum
    if broadcast == "disabled":
        assert all(
            all(weight == 0 for weights in cycle.read_attention for weight in weights)
            for cycle in p.rows[0].trace.cycles
        )
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    (tmp_path / "diagnostics.json").write_bytes(module._checked(result))


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_caller_dtype_rng_device_inference_and_exact_retry(
    experiment: Experiment,
    tmp_path: Path,
    default: str,
    inference: bool,
) -> None:
    heldout = _heldout(experiment)
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(default), torch.inference_mode(inference):
            first = _measure(experiment, heldout)
            second = _measure(experiment, heldout)
            assert first == second
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device(default)
            assert torch.is_inference_mode_enabled() is inference
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)
    (tmp_path / "diagnostics.json").write_bytes(module._checked(first))


@pytest.mark.parametrize("mode", ("metadata", "full"))
def test_explicit_full_vectors_have_actual_trace_nll_parity(
    experiment: Experiment,
    tmp_path: Path,
    mode: str,
) -> None:
    heldout = _heldout(experiment)
    policy = ScientificAblationDiagnosticPolicyV1.model_validate({"trace_mode": mode})
    result = _measure(experiment, heldout, policy=policy)
    assert all(
        (cycle.full is not None) == (mode == "full")
        for row in result.payload.rows
        for cycle in row.trace.cycles
    )
    (tmp_path / "diagnostics.json").write_bytes(module._checked(result))


@pytest.mark.parametrize(
    "field", ("maximum_teacher_executions", "maximum_trace_scalars", "maximum_total_trace_scalars")
)
def test_insufficient_complete_budget_fails_before_any_forward_without_subset(
    experiment: Experiment,
    field: str,
) -> None:
    heldout = _heldout(experiment)
    policy = ScientificAblationDiagnosticPolicyV1.model_validate({field: 1})
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, heldout, policy=policy)
    forward.assert_not_called()
    assert heldout._state == heldout._guard.context._state == "ready"


@pytest.mark.parametrize(
    "anchor",
    (
        "expected_evaluation_plan_sha256",
        "expected_evaluation_sha256",
        "expected_plan_sha256",
        "expected_capture_sha256",
    ),
)
def test_wrong_independent_anchor_rejected_before_forward(
    experiment: Experiment, anchor: str
) -> None:
    heldout = _heldout(experiment)
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, heldout, **{anchor: "0" * 64})
    forward.assert_not_called()


def test_busy_reentry_is_rejected_without_poisoning_successful_owner(
    experiment: Experiment,
) -> None:
    heldout = _heldout(experiment)
    original = IrisScientificAblationTeacherV1.evaluation_forward
    calls = 0

    def forward(
        self: IrisScientificAblationTeacherV1, *args: Any, **kwargs: Any
    ) -> IrisScientificAblationTeacherOutputV1:
        nonlocal calls
        calls += 1
        with pytest.raises(Error):
            _measure(experiment, heldout)
        assert heldout._state == heldout._guard.context._state == "busy"
        return original(self, *args, **kwargs)

    with patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=forward):
        assert _measure(experiment, heldout).payload.teacher_executions == calls == 4
    assert heldout._state == heldout._guard.context._state == "ready"


@pytest.mark.parametrize("kind", ("nll", "target", "invocation", "no_trace", "trace_mode"))
def test_foreign_actual_teacher_or_trace_binding_is_not_accepted(
    experiment: Experiment,
    kind: str,
) -> None:
    heldout = _heldout(experiment)
    original = IrisScientificAblationTeacherV1.evaluation_forward

    def forward(
        self: IrisScientificAblationTeacherV1, *args: Any, **kwargs: Any
    ) -> IrisScientificAblationTeacherOutputV1:
        output = original(self, *args, **kwargs)
        teacher = output.teacher
        if kind == "nll":
            return replace(
                output,
                losses=replace(output.losses, language_model=output.losses.language_model + 1),
            )
        if kind == "target":
            return replace(output, teacher=replace(teacher, target_sha256="0" * 64))
        if kind == "invocation":
            return replace(output, invocation_sha256="0" * 64)
        traced = teacher.traced.teacher
        trace = traced.diagnostic_trace
        assert trace is not None
        if kind == "trace_mode":
            trace = trace.model_copy(update={"intervention_plan_sha256": None})
        else:
            trace = None
        return replace(
            output,
            teacher=replace(
                teacher,
                traced=replace(teacher.traced, teacher=replace(traced, diagnostic_trace=trace)),
            ),
        )

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=forward),
        pytest.raises(Error),
    ):
        _measure(experiment, heldout)
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit))
def test_first_signal_identity_survives_restore_failure(
    experiment: Experiment,
    signal_type: type[BaseException],
) -> None:
    heldout = _heldout(experiment)
    signal = signal_type("first diagnostic signal")
    original_restore = _restore_caller

    def restore(*args: Any, **kwargs: Any) -> None:
        original_restore(*args, **kwargs)
        raise OSError("after real restore")

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", side_effect=signal),
        patch.object(module, "_restore_caller", new=restore),
        pytest.raises(signal_type) as caught,
    ):
        _measure(experiment, heldout)
    assert caught.value is signal
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize(
    "kind", ("test", "train", "validation", "best", "ledger", "earlier_row", "weights", "gradients")
)
def test_late_source_corpus_ledger_and_native_mutation_cannot_return_report(
    experiment: Experiment,
    tmp_path: Path,
    kind: str,
) -> None:
    case = _copy(experiment.case, tmp_path / "case")
    local = _experiment(case, tmp_path / "scores")
    heldout = _heldout(local)
    original = IrisScientificAblationTeacherV1.evaluation_forward
    calls = 0

    def forward(
        self: IrisScientificAblationTeacherV1, *args: Any, **kwargs: Any
    ) -> IrisScientificAblationTeacherOutputV1:
        nonlocal calls
        calls += 1
        output = original(self, *args, **kwargs)
        if calls == 4:
            paths = {
                "test": case.root / "test.parquet",
                "train": case.root / "train.parquet",
                "validation": case.root / "validation.parquet",
                "best": case.source.native_path / "best.safetensors",
                "ledger": local.progress.path / "score-00000" / "row.json",
            }
            if kind in paths:
                with paths[kind].open("ab") as file:
                    file.write(b"late drift")
            elif kind == "earlier_row":
                heldout._guard.rows[0].hidden_facts["operation"] = "late_foreign"
            else:
                parameter = next(self.probes.parameters())
                if kind == "weights":
                    with torch.no_grad():
                        parameter.add_(1)
                else:
                    parameter.grad = torch.ones_like(parameter)
        return output

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=forward),
        pytest.raises(Error),
    ):
        _measure(local, heldout)
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize(
    "field",
    (
        "evaluation_row_sha256",
        "evaluation_ack_sha256",
        "teacher_target_sha256",
        "target_nll_sum",
        "trace_scalar_bound",
        "probe_correct_counts",
        "source_20_4",
    ),
)
def test_rehashed_report_does_not_waive_raw_chain_trace_or_aggregate_integrity(
    experiment: Experiment,
    field: str,
) -> None:
    result = _measure(experiment, _heldout(experiment))
    raw = json.loads(module._checked(result))
    payload = raw["payload"]
    if field in ("evaluation_row_sha256", "evaluation_ack_sha256", "teacher_target_sha256"):
        payload["rows"][0][field] = "0" * 64
    elif field == "target_nll_sum":
        payload["rows"][0][field] += 1
    elif field == "trace_scalar_bound":
        payload[field] += 1
    elif field == "probe_correct_counts":
        payload[field][0] += 1
    else:
        payload[field] = "met"
    raw["sha256"] = domain_sha256(module._DOMAIN, canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        IrisScientificAblationDiagnosticsV1.model_validate_json(json.dumps(raw))


def test_full_corpus_deep_audit_is_start_end_not_per_forward(experiment: Experiment) -> None:
    heldout = _heldout(experiment)
    with (
        patch.object(module, "_prepare", wraps=_prepare) as prepare,
        patch.object(module, "_finish", wraps=_finish) as finish,
        patch.object(module, "_audit_original", wraps=_audit_original) as audit,
        patch.object(
            module,
            "read_scientific_ablation_evaluation_progress_v2",
            wraps=read_scientific_ablation_evaluation_progress_v2,
        ) as read,
    ):
        result = _measure(experiment, heldout)
    assert result.payload.teacher_executions == 4
    assert prepare.call_count == finish.call_count == 1
    assert audit.call_count == read.call_count == 2


def test_preflight_cleanup_uncertainty_poisons_even_before_first_forward(
    experiment: Experiment,
) -> None:
    heldout = _heldout(experiment)
    original_restore = _restore_caller

    def restore(*args: Any, **kwargs: Any) -> None:
        original_restore(*args, **kwargs)
        raise OSError("uncertain after real caller restore")

    with (
        patch.object(module, "_restore_caller", new=restore),
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(
            experiment,
            heldout,
            policy=ScientificAblationDiagnosticPolicyV1(maximum_teacher_executions=1),
        )
    forward.assert_not_called()
    assert heldout._state == heldout._guard.context._state == "poisoned"


@pytest.mark.parametrize("field", ("correct", "target_sha256", "predicted_class"))
def test_rehashed_probe_claims_still_require_class_domain_and_boolean_correctness(
    experiment: Experiment,
    field: str,
) -> None:
    result = _measure(experiment, _heldout(experiment))
    raw = json.loads(module._checked(result))
    probe = raw["payload"]["rows"][0]["probes"][0]
    if field == "correct":
        probe[field] = not probe[field]
        raw["payload"]["probe_correct_counts"][0] += 1 if probe[field] else -1
    elif field == "target_sha256":
        probe[field] = "0" * 64
    else:
        probe[field] = probe["class_count"]
    raw["sha256"] = domain_sha256(module._DOMAIN, canonical_json_bytes(raw["payload"]))
    with pytest.raises(ValueError):
        IrisScientificAblationDiagnosticsV1.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("kind", ("extra", "missing", "foreign_wrapper", "hash"))
def test_unchecked_artifact_objects_do_not_erase_foreign_or_missing_fields(
    experiment: Experiment,
    kind: str,
) -> None:
    result = _measure(experiment, _heldout(experiment))
    if kind == "extra":
        result.payload.rows[0].__dict__["foreign"] = "discarded_by_model_dump"
    elif kind == "missing":
        del result.payload.rows[0].__dict__["row_index"]
    elif kind == "foreign_wrapper":

        class Foreign(IrisScientificAblationDiagnosticsV1):
            pass

        result = Foreign.model_construct(sha256=result.sha256, payload=result.payload)
    else:
        result = result.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError):
        module._checked(result)


def test_partial_score_ledger_is_not_a_complete_diagnostic_source(
    experiment: Experiment,
    tmp_path: Path,
) -> None:
    heldout = _heldout(experiment)
    partial = _run(experiment.case, experiment.progress.plan, tmp_path / "partial", rows_per_call=1)
    assert partial.next_row_index == 1 and partial.evaluation is None
    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward") as forward,
        pytest.raises(Error),
    ):
        _measure(experiment, heldout, evaluation_path=partial.path)
    forward.assert_not_called()


def test_fully_rehashed_capture_and_score_nll_still_require_actual_native_forward(
    experiment: Experiment,
    tmp_path: Path,
) -> None:
    # Integrity-consistent altered measurement is NOT evidence of real teacher NLL.
    case = _copy(experiment.case, tmp_path / "case")
    rows = []
    head = _capture_genesis(case.capture.plan.sha256)
    for index, generated in enumerate(case.capture.rows):
        payload = generated.payload.model_copy(
            update={
                "target_nll_sum": generated.payload.target_nll_sum + (1.0 if index == 0 else 0.0)
            }
        )
        generated = IrisScientificAblationHeldoutRowV1(
            sha256=domain_sha256(
                "iris/scientific-ablation-heldout-row/v1",
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        data = _capture_file(generated)
        ack = _capture_ack(case.capture.plan.sha256, index, head, generated, data)
        directory = case.capture.path / f"row-{index:05d}"
        (directory / "attempt.json").write_bytes(
            canonical_json_file_bytes(_capture_attempt(case.capture.plan.sha256, index, head))
        )
        (directory / "row.json").write_bytes(data)
        (directory / "ack.json").write_bytes(_capture_file(ack))
        rows.append(generated)
        head = ack.sha256
    complete = _capture_complete(case.capture.plan, tuple(rows), head)
    (case.capture.path / "complete/complete.json").write_bytes(_capture_file(complete))
    case = replace(
        case,
        capture=read_scientific_ablation_capture_v1(
            case.capture.path,
            expected_plan_sha256=case.capture.plan.sha256,
            expected_capture_sha256=complete.sha256,
        ),
    )
    forged = _experiment(case, tmp_path / "scores")
    heldout = _heldout(forged)
    original = IrisScientificAblationTeacherV1.evaluation_forward
    calls = 0

    def forward(
        self: IrisScientificAblationTeacherV1, *args: Any, **kwargs: Any
    ) -> IrisScientificAblationTeacherOutputV1:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=forward),
        pytest.raises(Error) as caught,
    ):
        _measure(forged, heldout)
    assert calls == 1
    assert isinstance(caught.value.__cause__, ValueError)
    assert "actual original eos target nll" in str(caught.value.__cause__).lower()
    assert heldout._state == heldout._guard.context._state == "poisoned"
