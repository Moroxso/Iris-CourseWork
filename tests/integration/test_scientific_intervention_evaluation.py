"""Actual tiny-model pairs plus explicit controlled scorer inputs, never quality claims."""

import json
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
import torch

from iris.application import scientific_command as command
from iris.application import scientific_evaluation_v2 as baseline_evaluation
from iris.application import scientific_generation as generation
from iris.application import scientific_intervention as captures
from iris.application import scientific_intervention_evaluation as evaluation
from iris.application import synthetic_scoring_v2 as scoring
from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificDecodingPolicyV1,
)
from iris.artifacts.scientific_inference_v1 import IrisScientificInferenceManifestV1
from iris.artifacts.scientific_intervention_evaluation_v1 import (
    IrisScientificInterventionEvaluationV1,
    ScientificInterventionEvaluationPayloadV1,
)
from iris.artifacts.scientific_intervention_v1 import (
    IrisScientificInterventionV1,
    ScientificInterventionPayloadV1,
)
from iris.data.schemas import parse_structured_response
from iris.integrations.python_sandbox import PythonSandboxError, _worker
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_command import _edit, _files
from tests.integration.test_scientific_evaluation import CaptureCase
from tests.integration.test_scientific_evaluation import captured as captured
from tests.integration.test_scientific_generation import _test_spec
from tests.integration.test_scientific_intervention import plan
from tests.integration.test_scientific_teacher import _build
from tests.unit.test_synthetic_scoring import _answer, _observations
from tests.unit.test_synthetic_scoring_v2 import _POLICY, _observation

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class PairCase:
    source: CaptureCase
    capture: IrisScientificInterventionV1
    path: Path


@pytest.fixture(scope="module")
def paired(captured: CaptureCase) -> PairCase:
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
        capture = generation.generate_scientific_intervention_v1(
            config_path=captured.recipe,
            execution_path=captured.execution,
            test=captured.test,
            decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=1337),
            plan=plan({"kind": "zero_workspace"}),
        )
    assert capture.payload.baseline == captured.capture
    path = captures.write_scientific_intervention_v1(
        capture, root=captured.recipe.parent / "paired"
    )
    return PairCase(captured, capture, path)


@pytest.fixture
def case(tmp_path: Path, paired: PairCase) -> PairCase:
    original = paired.source.recipe.parent
    root = tmp_path / "source"
    shutil.copytree(original, root)
    return PairCase(
        replace(
            paired.source,
            recipe=root / paired.source.recipe.name,
            execution=root / paired.source.execution.relative_to(original),
            generation_path=root / paired.source.generation_path.relative_to(original),
        ),
        paired.capture,
        root / paired.path.relative_to(original),
    )


def evaluate(case: PairCase) -> IrisScientificInterventionEvaluationV1:
    return evaluation.evaluate_scientific_intervention_v1(
        intervention_path=case.path,
        config_path=case.source.recipe,
        execution_path=case.source.execution,
        test=case.source.test,
        python_policy=_POLICY if any(row.executable for row in case.source.rows) else None,
    )


def recapture(case: PairCase, payload: ScientificInterventionPayloadV1) -> PairCase:
    capture = IrisScientificInterventionV1(
        sha256=domain_sha256(
            "iris/scientific-intervention/v1", canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )
    path = captures.write_scientific_intervention_v1(
        capture, root=case.source.recipe.parent / "modified-pairs"
    )
    return replace(case, capture=capture, path=path)


def rebaseline(baseline: IrisScientificGenerationV1, **changes: Any) -> IrisScientificGenerationV1:
    payload = baseline.payload.model_copy(update=changes)
    return IrisScientificGenerationV1(
        sha256=domain_sha256(
            "iris/scientific-generation/v1", canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )


def fixture_answers(case: PairCase, *, degrade: bool) -> PairCase:
    # These artificial completions exercise scoring; they are NOT learned answers.
    payload = case.capture.payload
    baseline = rebaseline(
        payload.baseline,
        rows=tuple(
            row.model_copy(update={"output_text": source.expected_response})
            for row, source in zip(payload.baseline.payload.rows, case.source.rows, strict=True)
        ),
    )
    changed = tuple(
        row.model_copy(
            update={
                "output_text": "not JSON"
                if degrade and not source.executable
                else source.expected_response
            }
        )
        for row, source in zip(payload.rows, case.source.rows, strict=True)
    )
    return recapture(case, payload.model_copy(update={"baseline": baseline, "rows": changed}))


def fake_worker(case: PairCase) -> AsyncMock:
    row = next(row for row in case.source.rows if row.executable)

    async def execute(*, request_bytes: bytes, policy: Any) -> Any:
        assert policy == _POLICY
        assert json.loads(request_bytes) == {
            "kind": "python_integer_list_v1",
            "source": parse_structured_response(row.expected_response).content,
            "inputs": [list(c.inputs) for c in row.code_cases],
        }
        return replace(
            _observation(_observations(row), request_bytes),
            worker_sha256=sha256(_worker("python_worker_v2.txt")).hexdigest(),
        )

    return AsyncMock(side_effect=execute)


def test_real_saved_pairs_score_without_model_reload_and_preserve_sources(case: PairCase) -> None:
    before = (_files(case.source.execution), _files(case.path))
    rng = capture_runtime_state(torch.device("cpu"))
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch.object(command, "_composed") as compose,
        patch.object(scoring, "execute_python_job_v2") as sandbox,
    ):
        report = evaluate(case)
    for mock in (tokenizer, model, compose, sandbox):
        mock.assert_not_called()
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert (_files(case.source.execution), _files(case.path)) == before
    assert report.payload.selected_step == 1
    assert report.payload.overall.control.invalid_output == 4
    assert report.payload.overall.intervened.invalid_output == 4
    assert report.payload.overall.neither_correct == 4
    assert report.payload.overall.relative_task_success_drop is None
    assert report.payload.overall.rate_deltas["task_success_rate"] == 0
    assert tuple(group.quality.control.examples for group in report.payload.families) == (
        1,
        1,
        1,
        1,
    )
    assert tuple(row.difficulty for row in report.payload.rows) == tuple(
        row.difficulty for row in case.source.rows
    )
    assert len(report.payload.family_difficulties) == 12
    assert report.payload.scientific_launch_status == "not_assessed"
    for forbidden in (
        '"output_text"',
        '"prompt"',
        '"source"',
        '"expected_response"',
        '"hidden_facts"',
        '"replacement"',
        '"target_nll_sum"',
        '"target_token_count"',
    ):
        assert forbidden not in report.model_dump_json()


def test_controlled_quality_deltas_and_both_python_branches_use_actual_scorer(
    case: PairCase,
) -> None:
    case = fixture_answers(case, degrade=True)
    worker = fake_worker(case)
    with patch.object(scoring, "execute_python_job_v2", worker):
        report = evaluate(case)
    assert worker.await_count == 2  # Even identical source must really be evaluated in each branch.
    counts = report.payload.overall
    assert (counts.control.correct, counts.intervened.correct) == (4, 1)
    assert (
        counts.both_correct,
        counts.control_only_correct,
        counts.intervened_only_correct,
        counts.neither_correct,
    ) == (1, 3, 0, 0)
    assert counts.rate_deltas["task_success_rate"] == -0.75
    assert counts.relative_task_success_drop == 0.75
    assert counts.control.code_cases == counts.intervened.code_cases
    assert counts.control.code_cases_passed == counts.control.code_cases
    for task_group in report.payload.families:
        if task_group.task_type != "split_spec_repair":
            assert task_group.quality.rate_deltas["code_pass_rate"] is None
    for group in report.payload.family_difficulties:
        selected = tuple(
            row
            for row in report.payload.rows
            if row.control.task_type == group.task_type and row.difficulty == group.difficulty
        )
        assert group.quality.control.examples == len(selected)
        if not selected:
            assert all(value is None for value in group.quality.rate_deltas.values())


def test_infrastructure_failure_in_second_branch_aborts_not_a_wrong_model_score(
    case: PairCase,
) -> None:
    case = fixture_answers(case, degrade=False)
    worker = fake_worker(case)

    async def second_fails(**kwargs: Any) -> Any:
        if worker.await_count:
            raise PythonSandboxError("PRIVATE failure")
        return await worker(**kwargs)

    with (
        patch.object(scoring, "execute_python_job_v2", AsyncMock(side_effect=second_fails)),
        pytest.raises(IrisCodeEvaluationUnavailableError) as error,
    ):
        evaluate(case)
    assert "PRIVATE" not in str(error.value)
    assert worker.await_count == 1


@pytest.mark.parametrize(
    "field", ["source_run_configuration_sha256", "source_teacher_sidecar_sha256", "state_sha256"]
)
def test_rehashed_export_forgery_rejected_before_any_scoring(case: PairCase, field: str) -> None:
    export_payload = case.capture.payload.inference_manifest.payload.model_copy(
        update={field: "0" * 64}
    )
    export = IrisScientificInferenceManifestV1(
        sha256=domain_sha256(
            "iris/scientific-inference/v1/payload",
            canonical_json_bytes(export_payload.model_dump(mode="json")),
        ),
        payload=export_payload,
    )
    baseline = rebaseline(case.capture.payload.baseline, inference_sha256=export.sha256)
    case = recapture(
        case,
        case.capture.payload.model_copy(
            update={"inference_manifest": export, "baseline": baseline}
        ),
    )
    with (
        patch.object(evaluation, "score_synthetic_answer_v2") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluate(case)
    scorer.assert_not_called()


@pytest.mark.parametrize("field", ["workspace", "cognitive", "probe_initialization_seed"])
def test_changed_recipe_graph_or_probe_fails_before_scoring(case: PairCase, field: str) -> None:
    def change(raw: dict[str, Any]) -> None:
        if field == "workspace":
            raw[field]["decay_rate"] *= 0.5
        elif field == "cognitive":
            raw[field]["layer_sampler"]["layer_fractions"][0] *= 0.5
        else:
            raw[field] += 1

    _edit(case.source.recipe, change)
    with (
        patch.object(evaluation, "score_synthetic_answer_v2") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluate(case)
    scorer.assert_not_called()


@pytest.mark.parametrize("target", ["recipe", "capture", "test"])
def test_late_source_mutation_aborts_report(case: PairCase, target: str) -> None:
    original = scoring.score_synthetic_answer_v2
    calls = 0

    def mutate(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2 * len(case.source.rows):
            file = (
                case.source.recipe
                if target == "recipe"
                else case.path / "intervention.json"
                if target == "capture"
                else case.source.recipe.parent / case.source.test.path
            )
            file.write_bytes(file.read_bytes() + b"\n")
        return result

    with (
        patch.object(evaluation, "score_synthetic_answer_v2", side_effect=mutate),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluate(case)
    assert calls == 8


@pytest.mark.parametrize("change", ["graph", "runtime", "sidecar"])
def test_coordinated_rehashed_declarations_cannot_replace_source_checkpoint(
    case: PairCase, change: str
) -> None:
    original = case.capture.payload
    export_payload = original.inference_manifest.payload
    model = export_payload.model
    graph_sha = original.request_graph_sha256
    sidecar_sha = original.request_sidecar_sha256
    if change == "graph":
        workspace = model.workspace.model_copy(
            update={"decay_rate": model.workspace.decay_rate / 2}
        )
        model = model.model_copy(update={"workspace": workspace})
        _edit(
            case.source.recipe, lambda raw: raw["workspace"].update(decay_rate=workspace.decay_rate)
        )
        graph_sha = domain_sha256(
            "iris/intervened-request-graph/v1",
            canonical_json_bytes(
                {
                    "cognitive": model.cognitive.model_dump(mode="json"),
                    "workspace": model.workspace.model_dump(mode="json"),
                }
            ),
        )
    elif change == "runtime":
        model = model.model_copy(
            update={
                "runtime": model.runtime.model_copy(
                    update={
                        "max_position_embeddings": model.runtime.max_position_embeddings + 1,
                    }
                )
            }
        )
    else:
        sidecar_sha = "0" * 64
        model = model.model_copy(update={"sha256": sidecar_sha})
    export_payload = export_payload.model_copy(update={"model": model})
    export = IrisScientificInferenceManifestV1(
        sha256=domain_sha256(
            "iris/scientific-inference/v1/payload",
            canonical_json_bytes(export_payload.model_dump(mode="json")),
        ),
        payload=export_payload,
    )
    baseline = rebaseline(original.baseline, inference_sha256=export.sha256)
    case = recapture(
        case,
        original.model_copy(
            update={
                "inference_manifest": export,
                "baseline": baseline,
                "request_graph_sha256": graph_sha,
                "request_sidecar_sha256": sidecar_sha,
            }
        ),
    )
    with (
        patch.object(evaluation, "score_synthetic_answer_v2") as scorer,
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluate(case)
    scorer.assert_not_called()


def test_foreign_staging_is_never_removed(case: PairCase, tmp_path: Path) -> None:
    report = evaluate(case)
    fixed = type("Fixed", (), {"hex": "e" * 32})()
    foreign = (
        tmp_path
        / "reports/scientific-intervention-evaluation-v1"
        / f".iris-paired-evaluation-{fixed.hex}.tmp"
    )
    foreign.mkdir(parents=True)
    marker = foreign / "keep.txt"
    marker.write_text("foreign owner", encoding="utf-8")
    with (
        patch.object(evaluation, "uuid4", return_value=fixed),
        pytest.raises(evaluation.IrisScientificEvaluationError),
    ):
        evaluation.write_scientific_intervention_evaluation_v1(report, root=tmp_path / "reports")
    assert marker.read_text(encoding="utf-8") == "foreign owner"
    assert list(foreign.parent.iterdir()) == [foreign]


def test_noncode_subsplit_has_no_fake_code_rates_or_policy(case: PairCase) -> None:
    source_rows = tuple(row for row in case.source.rows if not row.executable)
    ids = {row.id for row in source_rows}
    test = _test_spec(case.source.recipe.parent / "noncode.parquet", source_rows)
    _, corpus_sha = generation._own_test_rows(source_rows, excluded_seeds=set(), scientific=False)
    baseline = rebaseline(
        case.capture.payload.baseline,
        rows=tuple(
            row for row in case.capture.payload.baseline.payload.rows if row.example_id in ids
        ),
        test_parquet_sha256=test.sha256,
        test_corpus_sha256=corpus_sha,
    )
    case = recapture(
        case,
        case.capture.payload.model_copy(
            update={
                "baseline": baseline,
                "rows": tuple(row for row in case.capture.payload.rows if row.example_id in ids),
            }
        ),
    )
    case = replace(case, source=replace(case.source, test=test, rows=source_rows))
    report = evaluate(case)
    assert report.payload.python_limits is None
    assert report.payload.python_policy is None
    assert report.payload.overall.control.code_examples == 0
    assert report.payload.overall.rate_deltas["code_pass_rate"] is None


def test_hash_only_roundtrip_and_rehashed_aggregate_tampering(
    case: PairCase, tmp_path: Path
) -> None:
    report = evaluate(case)
    path = evaluation.write_scientific_intervention_evaluation_v1(report, root=tmp_path / "reports")
    assert evaluation.read_scientific_intervention_evaluation_v1(path) == report
    assert (
        evaluation.write_scientific_intervention_evaluation_v1(report, root=tmp_path / "reports")
        == path
    )
    assert path.parent.name == "scientific-intervention-evaluation-v1"
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        baseline_evaluation.write_scientific_evaluation_v2(
            cast(Any, report), root=tmp_path / "wrong"
        )
    with pytest.raises(evaluation.IrisScientificEvaluationError):
        evaluation.write_scientific_intervention_evaluation_v1(
            report, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    raw = report.payload.model_dump(mode="python")
    for field, value in (
        ("families", tuple(reversed(raw["families"]))),
        ("difficulties", tuple(reversed(raw["difficulties"]))),
        ("rows", (raw["rows"][0], *raw["rows"])),
        ("python_policy", None),
    ):
        changed = {**raw, field: value}
        with pytest.raises(ValueError):
            ScientificInterventionEvaluationPayloadV1.model_validate(changed)
    invalid = deepcopy(raw)
    invalid["overall"]["rate_deltas"]["task_success_rate"] = 0.5
    with pytest.raises(ValueError):
        ScientificInterventionEvaluationPayloadV1.model_validate(invalid)
    invalid = deepcopy(raw)
    invalid["rows"][0]["difficulty"] = 1 if invalid["rows"][0]["difficulty"] != 1 else 2
    with pytest.raises(ValueError):
        ScientificInterventionEvaluationPayloadV1.model_validate(invalid)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit real IrisCodeEval sandbox gate",
)
@pytest.mark.parametrize("variant", ["reference", "incorrect", "syntax"])
def test_real_isolation_on_both_branches_is_not_learned_model_quality(
    case: PairCase, variant: str
) -> None:
    case = fixture_answers(case, degrade=False)
    if variant != "reference":
        changed = []
        for row, source in zip(case.capture.payload.rows, case.source.rows, strict=True):
            if source.executable:
                content = parse_structured_response(source.expected_response).content
                content = (
                    content
                    + "\n_original = solve\ndef solve(values): return _original(values) + 1\n"
                    if variant == "incorrect"
                    else "def solve(values) broken\n"
                )
                row = row.model_copy(update={"output_text": _answer(source, content=content)})
            changed.append(row)
        case = recapture(case, case.capture.payload.model_copy(update={"rows": tuple(changed)}))
    report = evaluate(case)
    control, changed_counts = report.payload.overall.control, report.payload.overall.intervened
    assert control.code_executed == changed_counts.code_executed == 1
    assert control.correct == 4 and control.code_correct == 1
    assert changed_counts.code_correct == int(variant == "reference")
    assert changed_counts.correct == (4 if variant == "reference" else 3)
    assert changed_counts.invalid_output == int(variant == "syntax")
    repair = next(
        row for row in report.payload.rows if row.control.task_type == "split_spec_repair"
    )
    assert repair.control.reason == "all_cases_pass"
    assert (
        repair.intervened.reason
        == {"reference": "all_cases_pass", "incorrect": "case_mismatch", "syntax": "syntax_error"}[
            variant
        ]
    )
    assert repair.control.execution is not None and repair.intervened.execution is not None
    assert repair.control.execution.policy == repair.intervened.execution.policy == _POLICY
    path = evaluation.write_scientific_intervention_evaluation_v1(
        report, root=case.source.recipe.parent / "real-reports"
    )
    assert evaluation.read_scientific_intervention_evaluation_v1(path) == report


def test_public_api_exports() -> None:
    import iris.application as public

    assert (
        public.evaluate_scientific_intervention_v1 is evaluation.evaluate_scientific_intervention_v1
    )
    assert (
        public.read_scientific_intervention_evaluation_v1
        is evaluation.read_scientific_intervention_evaluation_v1
    )
    assert (
        public.write_scientific_intervention_evaluation_v1
        is evaluation.write_scientific_intervention_evaluation_v1
    )
