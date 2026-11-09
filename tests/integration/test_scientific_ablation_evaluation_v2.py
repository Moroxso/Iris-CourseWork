"""Actual model-free mode-bound scores; controlled answers are NOT model quality."""

import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import torch

from iris.application import scientific_ablation_evaluation_v2 as module
from iris.application import synthetic_scoring_v2 as scoring
from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_ablation_capture import (
    IrisScientificAblationCaptureProgressV1,
    _ack,
    _attempt,
    _complete,
    _file,
    _genesis,
    read_scientific_ablation_capture_v1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_command import ScientificParquetInputV1
from iris.application.scientific_optimization import _same_runtime
from iris.application.synthetic_scoring_v2 import IrisSyntheticAnswerScoreV2
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_evaluation_v2 import (
    IrisScientificAblationEvaluationV2,
    ScientificAblationEvaluationPayloadV2,
)
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutRowV1
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.schemas import IrisExample, parse_structured_response
from iris.integrations.python_sandbox import PythonSandboxError
from iris.integrations.python_sandbox_v2 import PythonSandboxObservationV2
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_capture import _capture
from tests.integration.test_scientific_ablation_generation_source import (
    Prepared,
    Routing,
    _prepared,
    _read,
)
from tests.integration.test_scientific_ablation_heldout import _context, _plan
from tests.integration.test_scientific_generation import _test_rows, _test_spec
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_teacher import _rows as training_rows
from tests.unit.test_synthetic_scoring import _answer, _observations
from tests.unit.test_synthetic_scoring_v2 import _POLICY, _observation

pytestmark = pytest.mark.integration
Error = module.IrisScientificAblationEvaluationError


@dataclass(frozen=True)
class Case:
    root: Path
    source: Prepared
    capture: IrisScientificAblationCaptureProgressV1
    training: ScientificParquetInputV1
    validation: ScientificParquetInputV1
    test: ScientificParquetInputV1
    rows: tuple[IrisExample, ...]


def _make(
    root: Path,
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    rows: tuple[IrisExample, ...] | None = None,
) -> Case:
    root.mkdir(parents=True, exist_ok=True)
    source = _prepared(root / "source", routing, goal=goal, broadcast=broadcast)
    context = _context(_read(source), routing=routing, goal=goal, broadcast=broadcast)
    rows = _test_rows() if rows is None else rows
    heldout = _plan(context, root / "test.parquet", rows)
    from iris.application.scientific_ablation_capture import (
        build_scientific_ablation_capture_plan_v1,
    )

    plan = build_scientific_ablation_capture_plan_v1(
        heldout, expected_input_sha256=heldout.manifest.sha256
    )
    capture = _capture(heldout, root / "captures", plan)
    return Case(
        root,
        source,
        capture,
        _test_spec(root / "train.parquet", training_rows()[:5]),
        _test_spec(root / "validation.parquet", _validation()),
        ScientificParquetInputV1.model_validate(
            {
                "path": "test.parquet",
                "sha256": plan.payload.input.payload.test_parquet_sha256,
                "maximum_rows": len(rows),
                "maximum_file_bytes": 1024**2,
                "maximum_logical_bytes": 1024**2,
            }
        ),
        rows,
    )


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return _make(tmp_path_factory.mktemp("actual-mode-capture"))


def _copy(case: Case, root: Path) -> Case:
    shutil.copytree(case.root, root)
    capture = replace(case.capture, path=root / case.capture.path.relative_to(case.root))
    source = replace(
        case.source,
        native_path=root / case.source.native_path.relative_to(case.root),
        inference_path=root / case.source.inference_path.relative_to(case.root),
    )
    return replace(case, root=root, capture=capture, source=source)


def _evaluate(case: Case) -> IrisScientificAblationEvaluationV2:
    assert case.capture.complete is not None
    return module.evaluate_scientific_ablation_capture_v2(
        capture_path=case.capture.path,
        expected_plan_sha256=case.capture.plan.sha256,
        expected_capture_sha256=case.capture.complete.sha256,
        lifecycle_path=case.source.native_path,
        inference_path=case.source.inference_path,
        inputs_root=case.root,
        training=case.training,
        validation=case.validation,
        test=case.test,
        python_policy=_POLICY if any(row.executable for row in case.rows) else None,
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_native_capture_model_free_score(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    case = _make(tmp_path, routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:])
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as execute,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        result = _evaluate(case)
    execute.assert_not_called()
    model.assert_not_called()
    tokenizer.assert_not_called()
    assert result.payload.plan == case.capture.plan
    assert result.payload.capture == case.capture.complete
    assert (
        result.payload.plan.payload.source.payload.selected_step,
        result.payload.plan.payload.source.payload.terminal_step,
    ) == (1, 3)
    assert result.payload.overall.examples == result.payload.overall.invalid_output == 1
    assert result.payload.python_policy is None
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_actual_four_family_capture_preserves_full_eos_nll_and_private_bytes(case: Case) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    with patch.object(scoring, "execute_python_job_v2") as execute:
        result = _evaluate(case)
    execute.assert_not_called()  # Actual tiny model emits invalid envelopes, not code.
    assert result.payload.overall.examples == result.payload.overall.invalid_output == 4
    assert tuple(f.task_type for f in result.payload.families) == TASKS
    assert tuple(f.counts.examples for f in result.payload.families) == (1, 1, 1, 1)
    assert case.capture.complete is not None
    assert result.payload.overall.target_tokens == case.capture.complete.payload.target_token_count
    assert result.payload.overall.target_nll_sum == case.capture.complete.payload.target_nll_sum
    assert len({row.target_token_count for row in result.payload.rows}) > 1
    encoded = module._checked(result)
    for forbidden in (
        b'"output_text"',
        b'"expected_response"',
        b'"prompt"',
        b'"hidden_facts"',
        b'"python_state"',
        b'"request_receipt_json"',
    ):
        assert forbidden not in encoded
    assert result.payload.scientific_launch_status == "not_assessed"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "kind",
    (
        "plan",
        "capture",
        "partial",
        "train",
        "validation",
        "test",
        "late_test",
        "native",
        "inference",
        "extra_spec",
    ),
)
def test_wrong_source_data_capture_rejects_before_any_score(
    case: Case, tmp_path: Path, kind: str
) -> None:
    bad = _copy(case, tmp_path / "case")
    assert bad.capture.complete is not None
    original_complete = bad.capture.complete
    if kind in {"plan", "capture"}:
        if kind == "plan":
            bad = replace(
                bad,
                capture=replace(
                    bad.capture, plan=bad.capture.plan.model_copy(update={"sha256": "0" * 64})
                ),
            )
        else:
            bad = replace(
                bad,
                capture=replace(
                    bad.capture,
                    complete=original_complete.model_copy(update={"sha256": "0" * 64}),
                ),
            )
    elif kind == "partial":
        shutil.rmtree(bad.capture.path / "complete")  # Test-owned copied fixture only.
    elif kind in {"train", "validation", "test", "late_test"}:
        if kind == "train":
            bad = replace(
                bad, training=_test_spec(bad.root / "train.parquet", training_rows()[:5][::-1])
            )
        elif kind == "validation":
            bad = replace(
                bad, validation=_test_spec(bad.root / "validation.parquet", _validation()[::-1])
            )
        else:
            rows = (
                bad.rows[::-1]
                if kind == "test"
                else (
                    *bad.rows[:-1],
                    bad.rows[-1].model_copy(update={"prompt": bad.rows[-1].prompt + " changed"}),
                )
            )
            bad = replace(bad, test=_test_spec(bad.root / "test.parquet", rows))
    elif kind in {"native", "inference"}:
        path = bad.source.native_path if kind == "native" else bad.source.inference_path
        path = path / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        spec = bad.test.model_copy()
        vars(spec)["foreign"] = True
        bad = replace(bad, test=spec)
    with patch.object(module, "score_synthetic_answer_v2") as scorer, pytest.raises(Error):
        _evaluate(bad)
    scorer.assert_not_called()


def _controlled(case: Case, root: Path, variant: str = "reference") -> Case:
    """Scorer-only fixture answers with new trusted SHA; NOT actual model generation."""
    case = _copy(case, root)
    rows: list[IrisScientificAblationHeldoutRowV1] = []
    acks = []
    head = _genesis(case.capture.plan.sha256)
    for source, generated in zip(case.rows, case.capture.rows, strict=True):
        content = parse_structured_response(source.expected_response).content
        if source.executable and variant != "reference":
            content = (
                content + "\n_original = solve\ndef solve(values): return _original(values) + 1\n"
                if variant == "incorrect"
                else "def solve(values) broken\n"
            )
        payload = generated.payload.model_copy(
            update={"output_text": _answer(source, content=content)}
        )
        generated = IrisScientificAblationHeldoutRowV1(
            sha256=domain_sha256(
                "iris/scientific-ablation-heldout-row/v1",
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        content_bytes = _file(generated)
        ack = _ack(case.capture.plan.sha256, payload.row_index, head, generated, content_bytes)
        directory = case.capture.path / f"row-{payload.row_index:05d}"
        (directory / "attempt.json").write_bytes(
            canonical_json_file_bytes(_attempt(case.capture.plan.sha256, payload.row_index, head))
        )
        (directory / "row.json").write_bytes(content_bytes)
        (directory / "ack.json").write_bytes(_file(ack))
        rows.append(generated)
        acks.append(ack)
        head = ack.sha256
    complete = _complete(case.capture.plan, tuple(rows), head)
    (case.capture.path / "complete" / "complete.json").write_bytes(_file(complete))
    checked = read_scientific_ablation_capture_v1(
        case.capture.path,
        expected_plan_sha256=case.capture.plan.sha256,
        expected_capture_sha256=complete.sha256,
    )
    return replace(case, capture=checked)


def _worker(case: Case, *, partial: bool = False) -> AsyncMock:
    row = next(row for row in case.rows if row.executable)
    raw = _observations(row)
    if partial:
        data = json.loads(raw)
        data["cases"][0]["result"] += 1
        raw = canonical_json_bytes(data)

    async def execute(*, request_bytes: bytes, policy: object) -> PythonSandboxObservationV2:
        assert policy == _POLICY
        assert json.loads(request_bytes) == {
            "kind": "python_integer_list_v1",
            "source": parse_structured_response(row.expected_response).content,
            "inputs": [list(case.inputs) for case in row.code_cases],
        }
        return _observation(raw, request_bytes)

    return AsyncMock(side_effect=execute)


@pytest.mark.parametrize("partial", (False, True))
def test_controlled_four_family_rates_and_reference_equal_requires_worker(
    case: Case, tmp_path: Path, partial: bool
) -> None:
    controlled = _controlled(case, tmp_path / "controlled")
    worker = _worker(controlled, partial=partial)
    with patch.object(scoring, "execute_python_job_v2", worker):
        result = _evaluate(controlled)
    worker.assert_awaited_once()
    counts = result.payload.overall
    assert counts.correct == (3 if partial else 4)
    assert counts.incorrect == int(partial)
    assert counts.code_executed == 1 and counts.code_correct == int(not partial)
    assert counts.structured_exact_match == 4
    assert counts.code_cases_passed == counts.code_cases - int(partial)
    assert counts.rates["tool_choice_accuracy"] == counts.rates["conflict_flag_accuracy"] == 1.0
    assert result.payload.families[1].counts.rates["test_pass_rate"] is None
    assert result.payload.rows[0].execution is not None


def test_infrastructure_failure_is_redacted_not_incorrect(case: Case, tmp_path: Path) -> None:
    controlled = _controlled(case, tmp_path / "controlled")
    worker = AsyncMock(side_effect=PythonSandboxError("PRIVATE_RUNTIME_ERROR"))
    with (
        patch.object(scoring, "execute_python_job_v2", worker),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        _evaluate(controlled)
    worker.assert_awaited_once()
    assert "PRIVATE" not in str(caught.value) and caught.value.__cause__ is None


@pytest.mark.parametrize("kind", ("input", "worker", "policy"))
def test_wrong_execution_receipt_produces_no_report(case: Case, tmp_path: Path, kind: str) -> None:
    controlled = _controlled(case, tmp_path / "controlled")
    worker = _worker(controlled)

    async def wrong(*, request_bytes: bytes, policy: object) -> PythonSandboxObservationV2:
        actual = await worker(request_bytes=request_bytes, policy=policy)
        if type(actual) is not PythonSandboxObservationV2:
            raise AssertionError("Exact controlled worker observation required")
        if kind == "input":
            return replace(actual, submitted_input_sha256="0" * 64)
        if kind == "worker":
            return replace(actual, worker_sha256="0" * 64)
        return replace(actual, policy_sha256="0" * 64)

    with (
        patch.object(scoring, "execute_python_job_v2", AsyncMock(side_effect=wrong)),
        pytest.raises(IrisCodeEvaluationUnavailableError),
    ):
        _evaluate(controlled)


@pytest.mark.parametrize("kind", ("test", "validation", "native", "capture"))
def test_changed_inputs_during_scoring_withholds_report(
    case: Case, tmp_path: Path, kind: str
) -> None:
    copied = _copy(case, tmp_path / "case")
    original = scoring.score_synthetic_answer_v2

    def changed(
        example: IrisExample, *, output_text: str, python_policy: PythonSandboxPolicyV2 | None
    ) -> object:
        result = original(example, output_text=output_text, python_policy=python_policy)
        path = (
            copied.root / f"{kind}.parquet"
            if kind in {"test", "validation"}
            else copied.source.native_path / "manifest.json"
            if kind == "native"
            else copied.capture.path / "plan.json"
        )
        path.write_bytes(path.read_bytes() + b" ")
        return result

    with (
        patch.object(module, "score_synthetic_answer_v2", side_effect=changed),
        pytest.raises(Error),
    ):
        _evaluate(copied)


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit))
def test_primary_score_signal_is_preserved(case: Case, signal: type[BaseException]) -> None:
    with (
        patch.object(module, "score_synthetic_answer_v2", side_effect=signal("primary")),
        pytest.raises(signal),
    ):
        _evaluate(case)


def test_mixed_primary_signal_group_is_preserved(case: Case) -> None:
    primary = BaseExceptionGroup("primary", [ValueError("ordinary"), KeyboardInterrupt("signal")])
    with (
        patch.object(module, "score_synthetic_answer_v2", side_effect=primary),
        pytest.raises(BaseExceptionGroup) as caught,
    ):
        _evaluate(case)
    assert caught.value is primary


@pytest.mark.parametrize("kind", ("missing", "budget", "hidden"))
def test_exact_repair_policy_checked_before_any_score(case: Case, kind: str) -> None:
    policy = None if kind == "missing" else _POLICY.model_copy()
    if kind == "budget":
        assert policy is not None
        policy = policy.model_copy(update={"wall_time_ms": 4999})
    elif kind == "hidden":
        assert policy is not None
        vars(policy)["foreign"] = True
    assert case.capture.complete is not None
    with patch.object(module, "score_synthetic_answer_v2") as scorer, pytest.raises(Error):
        module.evaluate_scientific_ablation_capture_v2(
            capture_path=case.capture.path,
            expected_plan_sha256=case.capture.plan.sha256,
            expected_capture_sha256=case.capture.complete.sha256,
            lifecycle_path=case.source.native_path,
            inference_path=case.source.inference_path,
            inputs_root=case.root,
            training=case.training,
            validation=case.validation,
            test=case.test,
            python_policy=policy,
        )
    scorer.assert_not_called()


@pytest.mark.parametrize("field", ("example_sha256", "output_sha256", "case_count"))
def test_wrong_scorer_binding_is_not_a_report(case: Case, field: str) -> None:
    original = scoring.score_synthetic_answer_v2

    def corrupt(
        example: IrisExample, *, output_text: str, python_policy: PythonSandboxPolicyV2 | None
    ) -> IrisSyntheticAnswerScoreV2:
        score = original(example, output_text=output_text, python_policy=python_policy)
        if field == "case_count":
            return replace(score, case_count=score.case_count + 1)
        if field == "output_sha256":
            return replace(score, output_sha256="0" * 64)
        return replace(score, example_sha256="0" * 64)

    with (
        patch.object(module, "score_synthetic_answer_v2", side_effect=corrupt),
        pytest.raises(Error),
    ):
        _evaluate(case)


@pytest.mark.parametrize("kind", ("count", "family", "nll", "row", "teacher", "ack"))
def test_fully_rehashed_report_cannot_launder_counts_or_inventory(case: Case, kind: str) -> None:
    result = _evaluate(case)
    raw = result.payload.model_dump(mode="python")
    if kind == "count":
        raw["overall"]["correct"] += 1
    elif kind == "family":
        raw["families"] = tuple(reversed(raw["families"]))
    elif kind == "nll":
        raw["rows"][0]["target_nll_sum"] += 1
    elif kind == "row":
        raw["rows"] = tuple(reversed(raw["rows"]))
    elif kind == "teacher":
        raw["rows"][0]["teacher_invocation_sha256"] = "0" * 64
    else:
        raw["rows"][-1]["capture_ack_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ScientificAblationEvaluationPayloadV2.model_validate(raw)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit real IrisCodeEval sandbox gate",
)
@pytest.mark.parametrize("variant", ("reference", "incorrect", "syntax"))
def test_real_isolation_controlled_answers_not_model_quality(
    case: Case, tmp_path: Path, variant: str
) -> None:
    controlled = _controlled(case, tmp_path / "controlled", variant)
    result = _evaluate(controlled)
    counts = result.payload.overall
    assert counts.code_executed == 1
    assert counts.code_correct == int(variant == "reference")
    assert counts.correct == (4 if variant == "reference" else 3)
    repair = next(row for row in result.payload.rows if row.task_type == "split_spec_repair")
    assert (
        repair.reason
        == {"reference": "all_cases_pass", "incorrect": "case_mismatch", "syntax": "syntax_error"}[
            variant
        ]
    )
    assert repair.execution is not None
    assert repair.execution.kernel_release != "mock-kernel"
    (tmp_path / f"actual-isolated-{variant}-report.json").write_bytes(module._checked(result))
