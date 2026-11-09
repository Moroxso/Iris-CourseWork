"""Exact-best, complete-split actual NLL; saved answers are never regenerated."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import intervention_command as command
from iris.application import scientific_generation as generation
from iris.application import scientific_intervention_nll as nll
from iris.application.request import IrisRequestOrchestrator
from iris.application.scientific_evaluation import IrisScientificEvaluationError
from iris.application.scientific_intervention import write_scientific_intervention_v1
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import IrisScientificTeacherV1
from iris.artifacts.scientific_intervention_nll_v1 import IrisScientificInterventionNllV1
from iris.cli import app
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_intervention_command import _prepare, _redacted
from tests.integration.test_scientific_command import _edit, _files
from tests.integration.test_scientific_evaluation import captured as captured
from tests.integration.test_scientific_intervention import plan
from tests.integration.test_scientific_intervention_evaluation import (
    PairCase,
    rebaseline,
    recapture,
)
from tests.integration.test_scientific_intervention_evaluation import case as case
from tests.integration.test_scientific_intervention_evaluation import paired as paired
from tests.integration.test_scientific_teacher import _build

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def local_loaders() -> Iterator[None]:
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
        yield


def measure(case: PairCase) -> IrisScientificInterventionNllV1:
    return nll.measure_scientific_intervention_nll_v1(
        intervention_path=case.path,
        config_path=case.source.recipe,
        execution_path=case.source.execution,
        test=case.source.test,
    )


def test_exact_selected_best_actual_nll_no_regeneration_or_candidate_execution(
    case: PairCase,
) -> None:
    source_before = _files(case.source.recipe.parent)
    rng = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            IrisRequestOrchestrator, "generate", side_effect=AssertionError("No decoding")
        ),
        patch.object(
            IrisRequestOrchestrator, "execute_intervened", side_effect=AssertionError("No decoding")
        ),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ),
    ):
        result = measure(case)
    payload = result.payload
    assert payload.selected_step == 1
    assert case.capture.payload.baseline.payload.terminal_step == 3
    assert (
        payload.best_checkpoint_sha256
        == case.capture.payload.baseline.payload.best_checkpoint_sha256
    )
    assert payload.intervention_sha256 == case.capture.sha256
    assert payload.plan_sha256 == case.capture.payload.plan_sha256
    assert len(payload.rows) == 4
    for record, saved, changed in zip(
        payload.rows,
        case.capture.payload.baseline.payload.rows,
        case.capture.payload.rows,
        strict=True,
    ):
        assert (record.example_id, record.example_sha256) == (
            saved.example_id,
            saved.example_sha256,
        )
        assert record.control_nll_sum == saved.target_nll_sum
        assert record.target_token_count == saved.target_token_count
        assert record.bound_plan_sha256 == changed.bound_plan_sha256
    assert any(row.control_nll_sum != row.intervened_nll_sum for row in payload.rows)
    assert payload.groups[0].nll.examples == 4
    assert [g.nll.examples for g in payload.groups[1:5]] == [1, 1, 1, 1]
    assert payload.scientific_launch_status == "not_assessed"
    assert _files(case.source.recipe.parent) == source_before
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert case.capture.payload.intervened_target_nll == "not_measured"


@pytest.mark.parametrize("kind", ["empty", "gaussian_noise"])
def test_actual_empty_plan_and_row_bound_noise_measurements(case: PairCase, kind: str) -> None:
    selected = plan() if kind == "empty" else plan({"kind": kind, "sigma": 0.25, "seed": 42})
    capture = generation.generate_scientific_intervention_v1(
        config_path=case.source.recipe,
        execution_path=case.source.execution,
        test=case.source.test,
        decoding=case.capture.payload.baseline.payload.decoding,
        plan=selected,
    )
    path = write_scientific_intervention_v1(capture, root=case.path.parent.parent)
    selected_case = replace(case, capture=capture, path=path)
    result = measure(selected_case)
    repeated = measure(selected_case)
    assert result == repeated
    for row, saved in zip(result.payload.rows, capture.payload.rows, strict=True):
        assert row.bound_plan_sha256 == saved.bound_plan_sha256
        if kind == "empty":
            assert row.control_nll_sum == row.intervened_nll_sum
    if kind == "gaussian_noise":
        assert len({row.bound_plan_sha256 for row in result.payload.rows}) == len(
            result.payload.rows
        )


def test_saved_control_nll_must_be_reproduced_not_copied(case: PairCase) -> None:
    baseline = case.capture.payload.baseline
    altered = (
        baseline.payload.rows[0].model_copy(
            update={"target_nll_sum": baseline.payload.rows[0].target_nll_sum + 1}
        ),
        *baseline.payload.rows[1:],
    )
    altered_case = recapture(
        case,
        case.capture.payload.model_copy(update={"baseline": rebaseline(baseline, rows=altered)}),
    )
    with pytest.raises(IrisScientificEvaluationError):
        measure(altered_case)


def test_recipe_mismatch_is_rejected_before_loading_model(case: PairCase) -> None:
    _edit(case.source.recipe, lambda raw: raw.update(run_seed=raw["run_seed"] + 1))
    with patch.object(nll, "_composed") as composed, pytest.raises(IrisScientificEvaluationError):
        measure(case)
    composed.assert_not_called()


@pytest.mark.parametrize(
    "field",
    ["target_sha256", "intervention_plan_sha256", "teacher_graph_sha256", "sidecar_state_sha256"],
)
def test_wrong_teacher_receipt_aborts_even_with_valid_finite_nll(
    case: PairCase, field: str
) -> None:
    original = IrisScientificTeacherV1.evaluation_forward_intervened

    def wrong(self: IrisScientificTeacherV1, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        changes: dict[str, Any] = {field: "0" * 64}
        return replace(result, teacher=replace(result.teacher, **changes))

    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward_intervened", wrong),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


def test_late_failure_returns_no_partial_measurement_and_restores_runtime(case: PairCase) -> None:
    original = IrisScientificTeacherV1.evaluation_forward_intervened
    calls = 0
    rng = capture_runtime_state(torch.device("cpu"))

    def fail(self: IrisScientificTeacherV1, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            torch.rand(4)
            raise RuntimeError("private row data")
        return original(self, *args, **kwargs)

    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward_intervened", fail),
        pytest.raises(IrisScientificEvaluationError) as error,
    ):
        measure(case)
    assert calls == 3 and "private" not in str(error.value)
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("mutation", ["recipe", "capture"])
def test_late_source_change_aborts_a_complete_measurement(case: PairCase, mutation: str) -> None:
    original = IrisScientificTeacherV1.evaluation_forward_intervened
    calls = 0

    def mutate(self: IrisScientificTeacherV1, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 4:
            path = case.source.recipe if mutation == "recipe" else case.path / "intervention.json"
            path.write_bytes(path.read_bytes() + b" ")
        return result

    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward_intervened", mutate),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


@pytest.fixture
def result(case: PairCase) -> IrisScientificInterventionNllV1:
    return measure(case)


def test_report_roundtrip_contains_only_hashes_and_measurements(
    result: IrisScientificInterventionNllV1, tmp_path: Path
) -> None:
    path = nll.write_scientific_intervention_nll_v1(result, root=tmp_path / "reports")
    assert nll.read_scientific_intervention_nll_v1(path) == result
    assert nll.write_scientific_intervention_nll_v1(result, root=tmp_path / "reports") == path
    content = (path / "nll.json").read_bytes()
    assert content == canonical_json_file_bytes(result.model_dump(mode="json"))
    for private in (
        b'"output_text"',
        b'"token_ids"',
        b'"expected_response"',
        b'"prompt"',
        b'"source"',
    ):
        assert private not in content


@pytest.mark.parametrize("corruption", ["trailing", "extra_file", "digest", "duplicate_key"])
def test_reader_rejects_noncanonical_or_corrupt_report(
    result: IrisScientificInterventionNllV1, tmp_path: Path, corruption: str
) -> None:
    path = nll.write_scientific_intervention_nll_v1(result, root=tmp_path)
    report = path / "nll.json"
    content = report.read_bytes()
    if corruption == "trailing":
        report.write_bytes(content + b" ")
    elif corruption == "extra_file":
        (path / "unexpected").write_bytes(b"keep")
    elif corruption == "digest":
        report.write_bytes(content.replace(result.sha256.encode(), b"0" * 64, 1))
    else:
        report.write_bytes(b'{"sha256":"' + result.sha256.encode() + b'",' + content[1:])
    with pytest.raises(IrisScientificEvaluationError):
        nll.read_scientific_intervention_nll_v1(path)


@pytest.mark.parametrize("quota", [True, 0, 1, 16 * 1024**3 + 1])
def test_writer_enforces_bounded_storage(
    result: IrisScientificInterventionNllV1, tmp_path: Path, quota: int
) -> None:
    with pytest.raises(IrisScientificEvaluationError):
        nll.write_scientific_intervention_nll_v1(result, root=tmp_path, maximum_storage_bytes=quota)
    assert not list(tmp_path.rglob("nll.json"))


def test_writer_enforces_free_space_reserve(
    result: IrisScientificInterventionNllV1, tmp_path: Path
) -> None:
    with (
        patch(
            "iris.application.scientific_intervention_nll.shutil.disk_usage",
            return_value=SimpleNamespace(free=0),
        ),
        pytest.raises(IrisScientificEvaluationError),
    ):
        nll.write_scientific_intervention_nll_v1(result, root=tmp_path)
    assert not list(tmp_path.rglob("nll.json"))


def test_writer_accepts_verified_publication_with_lost_acknowledgement(
    result: IrisScientificInterventionNllV1, tmp_path: Path
) -> None:
    rename = rename_directory_with_windows_retry

    def lost(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("lost acknowledgement")

    with patch.object(nll, "rename_directory_with_windows_retry", side_effect=lost):
        path = nll.write_scientific_intervention_nll_v1(result, root=tmp_path)
    assert nll.read_scientific_intervention_nll_v1(path) == result
    assert not list(path.parent.glob(".iris-*.tmp"))


def test_writer_revalidates_forged_aggregate_despite_rehashed_payload(
    result: IrisScientificInterventionNllV1, tmp_path: Path
) -> None:
    group = result.payload.groups[0]
    changed = group.model_copy(update={"nll": group.nll.model_copy(update={"delta_mean_nll": 9.0})})
    payload = result.payload.model_copy(update={"groups": (changed, *result.payload.groups[1:])})
    forged = result.model_copy(
        update={
            "payload": payload,
            "sha256": domain_sha256(
                nll._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
        }
    )
    with pytest.raises(IrisScientificEvaluationError):
        nll.write_scientific_intervention_nll_v1(forged, root=tmp_path)


def test_staging_collision_never_removes_a_foreign_directory(
    result: IrisScientificInterventionNllV1,
    tmp_path: Path,
) -> None:
    foreign = tmp_path / nll._STORE / ".iris-intervention-nll-fixed.tmp"
    foreign.mkdir(parents=True)
    sentinel = foreign / "keep"
    sentinel.write_bytes(b"user-owned")
    with (
        patch.object(nll, "uuid4", return_value=SimpleNamespace(hex="fixed")),
        pytest.raises(IrisScientificEvaluationError),
    ):
        nll.write_scientific_intervention_nll_v1(result, root=tmp_path)
    assert sentinel.read_bytes() == b"user-owned"


def test_short_stage_write_is_removed_without_publishing_partial_report(
    result: IrisScientificInterventionNllV1,
    tmp_path: Path,
) -> None:
    def short(path: Path, content: bytes) -> None:
        path.write_bytes(content[:-1])

    with (
        patch.object(nll, "_write_exact_file", side_effect=short),
        pytest.raises(IrisScientificEvaluationError),
    ):
        nll.write_scientific_intervention_nll_v1(result, root=tmp_path)
    assert not list((tmp_path / nll._STORE).iterdir())


def test_source_cli_measures_saved_pair_without_python_policy_and_inspects_without_model(
    case: PairCase,
    tmp_path: Path,
) -> None:
    prepared = _prepare(case, tmp_path)
    _edit(prepared.context, lambda raw: raw.update(python_policy=None))
    args = [
        "intervene",
        "mode=measure-nll",
        f"checkpoint={prepared.checkpoint}",
        f"capture={case.path}",
    ]
    before = _files(case.source.execution)
    with patch.object(
        command,
        "evaluate_scientific_intervention_v1",
        side_effect=AssertionError("No answer scoring"),
    ):
        result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Paired target NLL" in result.output and "not_assessed" in result.output
    _redacted(result.output, case)
    (path,) = (prepared.output / nll._STORE).iterdir()
    report = nll.read_scientific_intervention_nll_v1(path)
    assert report.payload.intervention_sha256 == case.capture.sha256
    assert report.payload.selected_step == 1
    assert _files(case.source.execution) == before
    assert not (prepared.output / "scientific-intervention-evaluation-v1").exists()
    with (
        patch.object(nll, "_composed", side_effect=AssertionError("No model loading")),
        patch.object(command, "_read_context", side_effect=AssertionError("No context access")),
        patch.object(nll, "_audit_paired_source", side_effect=AssertionError("No source audit")),
    ):
        inspected = CliRunner().invoke(app, ["intervene", "mode=inspect-nll", f"report={path}"])
    assert inspected.exit_code == 0, inspected.output
    assert "source not re-audited" in inspected.output and "Paired target NLL" in inspected.output
    _redacted(inspected.output, case)


def test_cli_rechecks_context_before_publishing_nll(
    case: PairCase,
    result: IrisScientificInterventionNllV1,
    tmp_path: Path,
) -> None:
    prepared = _prepare(case, tmp_path)

    def changed(**kwargs: Any) -> IrisScientificInterventionNllV1:
        prepared.context.write_bytes(prepared.context.read_bytes() + b" ")
        return result

    with patch.object(command, "measure_scientific_intervention_nll_v1", side_effect=changed):
        invoked = CliRunner().invoke(
            app,
            [
                "intervene",
                "mode=measure-nll",
                f"checkpoint={prepared.checkpoint}",
                f"capture={case.path}",
            ],
        )
    assert invoked.exit_code != 0
    assert not list(prepared.output.rglob("nll.json"))
