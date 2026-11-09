"""Real tiny-Qwen capture integration; fixture completions test scoring, not learning."""

import json
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch

from iris.application import scientific_command as command
from iris.application import scientific_evaluation as evaluation
from iris.application import scientific_generation as generation
from iris.application import synthetic_scoring as scoring
from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    IrisScientificEvaluationV1,
    ScientificEvaluationPayloadV1,
    ScientificScoredRowV1,
    scientific_quality_counts_v1,
)
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificGenerationPayloadV1,
)
from iris.data.schemas import IrisExample, parse_structured_response
from iris.integrations.python_sandbox import PythonSandboxError, _worker
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_command import _edit, _files, _input, _recipe
from tests.integration.test_scientific_generation import _generate, _test_rows, _test_spec
from tests.integration.test_scientific_teacher import _build
from tests.unit.test_synthetic_scoring import _POLICY, _answer, _observation, _observations

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class CaptureCase:
    recipe: Path
    execution: Path
    test: command.ScientificParquetInputV1
    generation_path: Path
    capture: IrisScientificGenerationV1
    rows: tuple[IrisExample, ...]


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> CaptureCase:
    root = tmp_path_factory.mktemp("real-scientific-capture")
    recipe = _recipe(root)
    _edit(
        recipe,
        lambda raw: raw["lifecycle"].update(minimum_improvement=1e6, early_stopping_patience=2),
    )
    _, tokenizer, model = _build(cycles=1, slots=2)
    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=lambda *a, **k: deepcopy(tokenizer),
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=lambda *a, **k: deepcopy(model),
        ),
    ):
        created = command.create_scientific_command_v1(
            config_path=recipe, output_root=root / "runs"
        )
        done = command.resume_scientific_command_v1(
            config_path=recipe, execution_path=created.path, phases=1000
        )
        assert (done.state.step, done.state.best_step, done.state.phase) == (3, 1, "stopped")
        rows = _test_rows()
        test = _test_spec(root / "test.parquet", rows)
        capture = _generate((recipe, done.path), test)
    path = generation.write_scientific_generation_v1(capture, root=root / "captures")
    return CaptureCase(recipe, done.path, test, path, capture, rows)


@pytest.fixture
def case(tmp_path: Path, captured: CaptureCase) -> CaptureCase:
    # A private copy permits hostile source/recipe edits without affecting other tests.
    source = captured.recipe.parent
    target = tmp_path / "source"
    shutil.copytree(source, target)
    return replace(
        captured,
        recipe=target / captured.recipe.name,
        execution=target / captured.execution.relative_to(source),
        generation_path=target / captured.generation_path.relative_to(source),
    )


def _evaluate(case: CaptureCase) -> IrisScientificEvaluationV1:
    return evaluation.evaluate_scientific_generation_v1(
        generation_path=case.generation_path,
        execution_path=case.execution,
        config_path=case.recipe,
        test=case.test,
        python_policy=_POLICY if any(row.executable for row in case.rows) else None,
    )


def _recapture(case: CaptureCase, **updates: Any) -> CaptureCase:
    """Rehash deliberate counterexamples: testing only a stale hash is insufficient."""
    raw = case.capture.payload.model_dump(mode="python")
    raw.update(updates)
    payload = ScientificGenerationPayloadV1.model_validate(raw)
    capture = IrisScientificGenerationV1(
        sha256=domain_sha256(
            "iris/scientific-generation/v1", canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )
    path = generation.write_scientific_generation_v1(capture, root=case.recipe.parent / "modified")
    return replace(case, capture=capture, generation_path=path)


def _fixture_answers(case: CaptureCase) -> CaptureCase:
    # These are controlled scorer inputs, NOT outputs or quality evidence of a trained model.
    rows = tuple(
        generated.model_copy(update={"output_text": row.expected_response})
        for row, generated in zip(case.rows, case.capture.payload.rows, strict=True)
    )
    return _recapture(case, rows=rows)


def _fake_worker(case: CaptureCase, *, partial: bool = False) -> AsyncMock:
    row = next(row for row in case.rows if row.executable)
    raw = _observations(row)
    if partial:
        data = json.loads(raw)
        data["cases"][0]["result"] += 1
        raw = canonical_json_bytes(data)

    async def execute(*, request_bytes: bytes, policy: Any) -> Any:
        assert policy == _POLICY
        submitted = json.loads(request_bytes)
        assert submitted == {
            "kind": "python_integer_list_v1",
            "source": parse_structured_response(row.expected_response).content,
            "inputs": [list(c.inputs) for c in row.code_cases],
        }
        return replace(
            _observation(raw, request_bytes), worker_sha256=sha256(_worker()).hexdigest()
        )

    return AsyncMock(side_effect=execute)


def test_actual_capture_scores_all_families_without_model_reload(case: CaptureCase) -> None:
    source_before = _files(case.execution)
    rng = capture_runtime_state(torch.device("cpu"))
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch.object(command, "_composed") as compose,
        patch.object(scoring, "execute_python_job_v1") as sandbox,
    ):
        result = _evaluate(case)
    tokenizer.assert_not_called()
    model.assert_not_called()
    compose.assert_not_called()
    sandbox.assert_not_called()  # This random tiny model emits invalid envelopes.
    assert result.payload.selected_step == 1
    assert result.payload.best_checkpoint_sha256 == case.capture.payload.best_checkpoint_sha256
    assert tuple(f.task_type for f in result.payload.families) == TASKS
    assert tuple(f.counts.examples for f in result.payload.families) == (1, 1, 1, 1)
    counts = result.payload.overall
    assert counts.examples == counts.invalid_output == 4
    assert counts.correct == counts.incorrect == counts.code_executed == 0
    assert counts.rates["invalid_output_rate"] == 1.0
    assert counts.rates["json_schema_validity"] == 0.0
    assert counts.target_tokens == case.capture.payload.target_token_count
    assert counts.rates["target_nll_per_token"] == case.capture.payload.target_nll_per_token
    assert _files(case.execution) == source_before
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    serialized = result.model_dump_json()
    for forbidden in ("output_text", "expected_response", "hidden_facts", '"prompt"', '"source"'):
        assert forbidden not in serialized
    assert result.payload.scientific_launch_status == "not_assessed"


@pytest.mark.parametrize(
    "field",
    [
        "execution_sha256",
        "terminal_event_sha256",
        "best_validation_event_sha256",
        "best_checkpoint_sha256",
        "teacher_configuration_sha256",
        "backbone_runtime_sha256",
        "test_corpus_sha256",
        "test_parquet_sha256",
        "selected_step",
        "terminal_step",
    ],
)
def test_rehashed_wrong_source_identity_rejects_before_scoring(
    case: CaptureCase, field: str
) -> None:
    bad = _recapture(case, **{field: 2 if field.endswith("step") else "0" * 64})
    with (
        patch.object(evaluation, "score_synthetic_answer_v1") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(bad)
    scorer.assert_not_called()


def test_noncode_only_capture_needs_no_sandbox_and_reports_missing_families(
    case: CaptureCase,
) -> None:
    source_rows = tuple(row for row in case.rows if not row.executable)
    generated = tuple(
        row for row in case.capture.payload.rows if row.task_type != "split_spec_repair"
    )
    spec = _test_spec(case.recipe.parent / "noncode.parquet", source_rows)
    _, digest = generation._own_test_rows(source_rows, excluded_seeds=set(), scientific=False)
    case = _recapture(
        case, rows=generated, test_parquet_sha256=spec.sha256, test_corpus_sha256=digest
    )
    case = replace(case, rows=source_rows, test=spec)
    with patch.object(scoring, "execute_python_job_v1") as worker:
        result = _evaluate(case)
    worker.assert_not_called()
    assert result.payload.python_policy is None
    assert result.payload.overall.code_examples == 0
    assert result.payload.overall.rates["code_pass_rate"] is None
    assert result.payload.overall.rates["test_pass_rate"] is None
    assert result.payload.families[0].task_type == "split_spec_repair"
    assert result.payload.families[0].counts.examples == 0
    assert all(rate is None for rate in result.payload.families[0].counts.rates.values())
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        evaluation.evaluate_scientific_generation_v1(
            generation_path=case.generation_path,
            execution_path=case.execution,
            config_path=case.recipe,
            test=case.test,
            python_policy=_POLICY,
        )


def test_wrong_worker_hash_is_not_a_valid_execution_receipt(case: CaptureCase) -> None:
    case = _fixture_answers(case)
    worker = _fake_worker(case)

    async def wrong(*args: Any, **kwargs: Any) -> Any:
        return replace(await worker(*args, **kwargs), worker_sha256="0" * 64)

    with (
        patch.object(scoring, "execute_python_job_v1", AsyncMock(side_effect=wrong)),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(case)


def test_staging_collision_does_not_remove_other_owners_files(
    case: CaptureCase, tmp_path: Path
) -> None:
    from uuid import UUID

    result = _evaluate(case)
    other = (
        tmp_path
        / "reports"
        / "scientific-evaluation-v1"
        / (".iris-evaluation-" + "0" * 32 + ".tmp")
    )
    other.mkdir(parents=True)
    marker = other / "keep.txt"
    marker.write_text("another owner", encoding="utf-8")
    with (
        patch.object(evaluation, "uuid4", return_value=UUID(int=0)),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluation.write_scientific_evaluation_v1(result, root=tmp_path / "reports")
    assert marker.read_text(encoding="utf-8") == "another owner"
    assert list(other.parent.iterdir()) == [other]


def test_public_api_exports() -> None:
    import iris.application as public

    assert public.evaluate_scientific_generation_v1 is evaluation.evaluate_scientific_generation_v1
    assert public.read_scientific_evaluation_v1 is evaluation.read_scientific_evaluation_v1
    assert public.write_scientific_evaluation_v1 is evaluation.write_scientific_evaluation_v1
    assert public.IrisScientificEvaluationV1 is IrisScientificEvaluationV1


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit real IrisCodeEval sandbox gate",
)
@pytest.mark.parametrize("variant", ["reference", "incorrect", "syntax"])
def test_real_isolation_scores_controlled_answers_not_model_quality(
    case: CaptureCase, variant: str
) -> None:
    case = _fixture_answers(case)
    if variant != "reference":
        rows = []
        for source, generated in zip(case.rows, case.capture.payload.rows, strict=True):
            if source.executable:
                content = parse_structured_response(source.expected_response).content
                content = (
                    content
                    + "\n_original = solve\ndef solve(values): return _original(values) + 1\n"
                    if variant == "incorrect"
                    else "def solve(values) broken\n"
                )
                generated = generated.model_copy(
                    update={"output_text": _answer(source, content=content)}
                )
            rows.append(generated)
        case = _recapture(case, rows=tuple(rows))
    result = _evaluate(case)
    counts = result.payload.overall
    assert counts.code_executed == 1
    assert counts.correct == (4 if variant == "reference" else 3)
    assert counts.code_correct == int(variant == "reference")
    assert counts.invalid_output == int(variant == "syntax")
    repair = next(row for row in result.payload.rows if row.task_type == "split_spec_repair")
    assert (
        repair.reason
        == {"reference": "all_cases_pass", "incorrect": "case_mismatch", "syntax": "syntax_error"}[
            variant
        ]
    )
    assert repair.execution is not None
    assert repair.execution.worker_sha256 == sha256(_worker()).hexdigest()
    path = evaluation.write_scientific_evaluation_v1(
        result, root=case.recipe.parent / "real-reports"
    )
    assert evaluation.read_scientific_evaluation_v1(path) == result


@pytest.mark.parametrize("kind", ["last_prompt", "last_example", "last_task", "reverse", "missing"])
def test_every_row_is_bound_before_any_scoring(case: CaptureCase, kind: str) -> None:
    rows = case.capture.payload.rows
    if kind == "reverse":
        rows = rows[::-1]
    elif kind == "missing":
        rows = rows[:-1]
    else:
        update = {
            "last_prompt": {"prompt_sha256": "0" * 64},
            "last_example": {"example_sha256": "0" * 64},
            "last_task": {"task_type": "split_spec_repair"},
        }[kind]
        rows = (*rows[:-1], rows[-1].model_copy(update=update))
    bad = _recapture(case, rows=rows)
    with (
        patch.object(evaluation, "score_synthetic_answer_v1") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(bad)
    scorer.assert_not_called()


@pytest.mark.parametrize("kind", ["train", "validation", "lifecycle", "optimizer", "run_seed"])
def test_changed_source_recipe_or_order_rejects(case: CaptureCase, kind: str) -> None:
    if kind in {"train", "validation"}:
        config = command.read_scientific_command_config_v1(case.recipe)
        rows = command._rows(getattr(config, kind), case.recipe.parent)
        replacement = _input(case.recipe.parent / f"{kind}.parquet", rows[::-1])
        _edit(case.recipe, lambda raw: raw.update({kind: replacement}))
    elif kind == "run_seed":
        _edit(case.recipe, lambda raw: raw.update(run_seed=9000))
    else:
        _edit(
            case.recipe,
            lambda raw: raw[kind].update(
                {"minimum_improvement": 42.0} if kind == "lifecycle" else {"learning_rate": 0.02}
            ),
        )
    with (
        patch.object(evaluation, "score_synthetic_answer_v1") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(case)
    scorer.assert_not_called()


def test_test_seed_overlap_rejects_even_with_rehashed_corpus(case: CaptureCase) -> None:
    row = case.rows[-1]
    # A structurally valid row still cannot reuse a training seed.
    raw = row.model_dump(mode="python")
    raw.update(id="test-0", seed=0)
    replacement = IrisExample.model_validate(raw)
    rows = (*case.rows[:-1], replacement)
    spec = _test_spec(case.recipe.parent / "overlap.parquet", rows)
    owned, digest = generation._own_test_rows(rows, excluded_seeds=set(), scientific=False)
    generated = tuple(
        g.model_copy(
            update={
                "example_id": r.id,
                "example_sha256": domain_sha256(
                    "iris/scientific-teacher-example/v1",
                    canonical_json_bytes(r.model_dump(mode="json")),
                ),
            }
        )
        for r, g in zip(owned, case.capture.payload.rows, strict=True)
    )
    bad = _recapture(
        case, rows=generated, test_parquet_sha256=spec.sha256, test_corpus_sha256=digest
    )
    bad = replace(bad, rows=rows, test=spec)
    with (
        patch.object(evaluation, "score_synthetic_answer_v1") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(bad)
    scorer.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
def test_fixture_answers_measure_cases_and_family_rates(case: CaptureCase, partial: bool) -> None:
    case = _fixture_answers(case)
    worker = _fake_worker(case, partial=partial)
    with patch.object(scoring, "execute_python_job_v1", worker):
        result = _evaluate(case)
    worker.assert_awaited_once()
    counts = result.payload.overall
    assert counts.correct == (3 if partial else 4)
    assert counts.incorrect == int(partial)
    assert counts.envelope_valid == counts.structured_exact_match == 4
    assert counts.code_examples == counts.code_executed == 1
    assert counts.code_correct == int(not partial)
    assert counts.code_cases_passed == counts.code_cases - int(partial)
    assert counts.rates["test_pass_rate"] == (counts.code_cases - int(partial)) / counts.code_cases
    assert counts.rates["tool_choice_accuracy"] == counts.rates["conflict_flag_accuracy"] == 1.0
    assert result.payload.families[1].counts.rates["code_pass_rate"] is None
    repair = next(r for r in result.payload.rows if r.task_type == "split_spec_repair")
    assert repair.generation_example_sha256 != repair.scoring_example_sha256
    assert repair.execution is not None
    assert repair.execution.worker_sha256 == sha256(_worker()).hexdigest()


def test_semantic_match_is_distinct_from_exact_text_and_confidence(case: CaptureCase) -> None:
    case = _fixture_answers(case)
    altered = []
    for source, generated in zip(case.rows, case.capture.payload.rows, strict=True):
        content = parse_structured_response(source.expected_response).content
        if source.task_type == "delayed_goal":
            content = " \n" + content + "\t"
        elif source.task_type in {"conflict_resolution", "next_tool_prediction"}:
            content = json.dumps(json.loads(content), indent=2)
        altered.append(
            generated.model_copy(
                update={"output_text": _answer(source, content=content, confidence=0.01)}
            )
        )
    case = _recapture(case, rows=tuple(altered))
    with patch.object(scoring, "execute_python_job_v1", _fake_worker(case)):
        result = _evaluate(case)
    assert result.payload.overall.correct == 4
    assert result.payload.overall.structured_exact_match == 1  # confidence is not a target


def test_infrastructure_failure_produces_no_report_and_is_redacted(case: CaptureCase) -> None:
    case = _fixture_answers(case)
    worker = AsyncMock(side_effect=PythonSandboxError("PRIVATE_RUNTIME_DETAILS"))
    with (
        patch.object(scoring, "execute_python_job_v1", worker),
        pytest.raises(IrisCodeEvaluationUnavailableError) as caught,
    ):
        _evaluate(case)
    assert "PRIVATE" not in str(caught.value) and caught.value.__cause__ is None
    worker.assert_awaited_once()


@pytest.mark.parametrize("field", ["output_sha256", "example_sha256", "case_count"])
def test_wrong_scorer_binding_cannot_enter_report(case: CaptureCase, field: str) -> None:
    original = scoring.score_synthetic_answer_v1

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        score = original(*args, **kwargs)
        if field == "case_count":
            return replace(score, case_count=0)
        if field == "output_sha256":
            return replace(score, output_sha256="0" * 64)
        return replace(score, example_sha256="0" * 64)

    with (
        patch.object(evaluation, "score_synthetic_answer_v1", side_effect=corrupt),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(case)


def test_source_recipe_change_during_scoring_aborts_report(case: CaptureCase) -> None:
    original = scoring.score_synthetic_answer_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        _edit(case.recipe, lambda raw: raw.update(run_seed=9000))
        return result

    with (
        patch.object(evaluation, "score_synthetic_answer_v1", side_effect=changed),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        _evaluate(case)


def test_roundtrip_quota_corruption_and_lost_acknowledgement(
    case: CaptureCase, tmp_path: Path
) -> None:
    result = _evaluate(case)
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        evaluation.write_scientific_evaluation_v1(
            result, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    assert list((tmp_path / "quota" / "scientific-evaluation-v1").iterdir()) == []

    def uncertain(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        raise OSError("Lost acknowledgement")

    with patch.object(evaluation, "rename_directory_with_windows_retry", side_effect=uncertain):
        path = evaluation.write_scientific_evaluation_v1(result, root=tmp_path / "reports")
    assert evaluation.read_scientific_evaluation_v1(path) == result
    assert evaluation.write_scientific_evaluation_v1(result, root=tmp_path / "reports") == path
    assert len(list(path.parent.iterdir())) == 1
    file = path / "evaluation.json"
    corrupt = file.read_bytes() + b" "
    file.write_bytes(corrupt)
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        evaluation.read_scientific_evaluation_v1(path)
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        evaluation.write_scientific_evaluation_v1(result, root=tmp_path / "reports")
    assert file.read_bytes() == corrupt


def test_report_rejects_recomputed_digest_with_contradictory_counts(case: CaptureCase) -> None:
    result = _evaluate(case)
    for field in ("correct", "invalid_output", "code_cases", "target_tokens", "target_nll_sum"):
        raw = result.payload.model_dump(mode="python")
        raw["overall"][field] += 1
        with pytest.raises(ValueError, match="counts differ"):
            ScientificEvaluationPayloadV1.model_validate(raw)
    raw = result.payload.model_dump(mode="python")
    raw["families"] = tuple(reversed(raw["families"]))
    with pytest.raises(ValueError, match="counts differ"):
        ScientificEvaluationPayloadV1.model_validate(raw)
    raw = result.payload.model_dump(mode="python")
    raw["purpose"] = "scientific"
    with pytest.raises(ValueError, match="5000"):
        ScientificEvaluationPayloadV1.model_validate(raw)
    row = result.payload.rows[0].model_copy(update={"target_nll_sum": 1e308})
    with pytest.raises(ValueError, match="overflowed"):
        scientific_quality_counts_v1((row, row))


def test_resource_observations_cannot_contradict_success(case: CaptureCase) -> None:
    case = _fixture_answers(case)
    with patch.object(scoring, "execute_python_job_v1", _fake_worker(case)):
        result = _evaluate(case)
    repair = next(row for row in result.payload.rows if row.execution is not None)
    for update in (
        {"oom_killed": True},
        {"transport_termination": "timeout"},
        {"transport_termination": "output_limit"},
        {"container_exit_code": 1},
    ):
        raw = repair.model_dump(mode="python")
        raw["execution"].update(update)
        with pytest.raises(ValueError, match="resource-failure"):
            ScientificScoredRowV1.model_validate(raw)


@pytest.mark.parametrize("policy", [None, _POLICY.model_copy(update={"wall_time_ms": 4999})])
def test_explicit_repair_policy_is_required_even_for_invalid_envelopes(
    case: CaptureCase, policy: Any
) -> None:
    with (
        patch.object(evaluation, "score_synthetic_answer_v1") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluation.evaluate_scientific_generation_v1(
            generation_path=case.generation_path,
            execution_path=case.execution,
            config_path=case.recipe,
            test=case.test,
            python_policy=policy,
        )
    scorer.assert_not_called()
