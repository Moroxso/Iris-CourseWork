"""Actual selected-best tiny-Qwen diagnostics; no pretrained downloads or candidate execution."""

import json
import shutil
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import scientific_command as command
from iris.application import scientific_generation as generation
from iris.application import scientific_workspace as workspace
from iris.application import scientific_workspace_command as workspace_command
from iris.application.scientific_answer_command import IrisScientificAnswerCommandError
from iris.application.scientific_evaluation import IrisScientificEvaluationError
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import IrisScientificTeacherV1
from iris.application.workspace_causal import measure_workspace_causal_v1
from iris.artifacts.scientific_workspace_v1 import (
    IrisScientificWorkspaceV1,
    ScientificWorkspacePayloadV1,
    ScientificWorkspacePolicyV1,
)
from iris.cli import app
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.causal import WorkspaceCausalPolicyV1
from tests.integration.test_scientific_command import _edit, _files, _recipe
from tests.integration.test_scientific_evaluation import CaptureCase, _recapture
from tests.integration.test_scientific_generation import _generate, _test_rows, _test_spec
from tests.integration.test_scientific_teacher import _build

pytestmark = pytest.mark.integration

_POLICY = ScientificWorkspacePolicyV1(
    selection_seed=42,
    examples_per_stratum=1,
    causal=WorkspaceCausalPolicyV1(shuffle_seed=1337),
)


@pytest.fixture(scope="module", autouse=True)
def local_loaders() -> Iterator[None]:
    _, tokenizer, model = _build(cycles=2, slots=2)
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


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory, local_loaders: None) -> CaptureCase:
    root = tmp_path_factory.mktemp("workspace-source")
    recipe = _recipe(root)
    _edit(recipe, lambda raw: raw["workspace"].update(cycles=2))
    _edit(
        recipe,
        lambda raw: raw["lifecycle"].update(minimum_improvement=1e6, early_stopping_patience=2),
    )
    created = command.create_scientific_command_v1(config_path=recipe, output_root=root / "runs")
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
    source, target = captured.recipe.parent, tmp_path / "source"
    shutil.copytree(source, target)
    return replace(
        captured,
        recipe=target / captured.recipe.name,
        execution=target / captured.execution.relative_to(source),
        generation_path=target / captured.generation_path.relative_to(source),
    )


def _measure(
    case: CaptureCase, policy: ScientificWorkspacePolicyV1 = _POLICY
) -> IrisScientificWorkspaceV1:
    return workspace.measure_scientific_workspace_v1(
        generation_path=case.generation_path,
        execution_path=case.execution,
        config_path=case.recipe,
        test=case.test,
        policy=policy,
    )


@pytest.fixture(scope="module")
def reported(captured: CaptureCase) -> IrisScientificWorkspaceV1:
    return _measure(captured)


def test_selected_best_real_measurement_is_reproducible_and_preserves_source_rng(
    case: CaptureCase,
    reported: IrisScientificWorkspaceV1,
) -> None:
    before, rng = _files(case.recipe.parent), capture_runtime_state(torch.device("cpu"))
    with (
        patch(
            "iris.application.request.IrisRequestOrchestrator.generate",
            side_effect=AssertionError("no generation"),
        ),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("no code execution"),
        ),
    ):
        result = _measure(case)
    assert result == reported
    assert _files(case.recipe.parent) == before
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    p = result.payload
    assert (p.selected_step, p.terminal_step, p.executions, p.trace_scalar_bound) == (
        1,
        3,
        20,
        6792,
    )
    assert p.best_checkpoint_sha256 == case.capture.payload.best_checkpoint_sha256
    assert p.causal_summary is not None and p.causal_summary.examples == 4
    assert p.scientific_acceptance == "not_assessed"
    assert p.causal_summary.scientific_acceptance == "not_assessed"
    saved = {r.example_id: r for r in case.capture.payload.rows}
    for row in p.rows:
        assert row.control_nll_sum == saved[row.example_id].target_nll_sum
        assert row.causal is not None and row.causal.executions == 4
        assert any(effect.ablated_distance > 1e-6 for effect in row.causal.effects)
        assert all(effect.shuffled_distance < 1e-6 for effect in row.causal.effects)
        assert all(cycle.full is None for cycle in row.trace.cycles)
    assert len(p.groups) == 20 and p.groups[0].available_examples == 4
    assert [g.selected_examples for g in p.groups[1:5]] == [1, 1, 1, 1]
    assert any(g.available_examples == 0 and g.diagnostics is None for g in p.groups)
    for forbidden in ('"prompt"', '"expected_response"', '"output_text"', '"hidden_facts"'):
        assert forbidden not in result.model_dump_json()


def test_full_is_opt_in_and_causal_measurement_can_be_disabled(
    case: CaptureCase, reported: IrisScientificWorkspaceV1
) -> None:
    policy = ScientificWorkspacePolicyV1(
        selection_seed=42, examples_per_stratum=1, trace_mode="full"
    )
    result = _measure(case, policy)
    assert result.payload.causal_summary is None
    assert result.payload.executions == 4
    assert result.payload.groups == reported.payload.groups
    assert all(
        r.causal is None and all(c.full is not None for c in r.trace.cycles)
        for r in result.payload.rows
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_executions": 19},
        {"maximum_total_trace_scalars": 6791},
        {"maximum_trace_scalars": 1},
    ],
)
def test_total_budgets_precede_model_loading(case: CaptureCase, changes: dict[str, Any]) -> None:
    policy = ScientificWorkspacePolicyV1.model_validate({**_POLICY.model_dump(), **changes})
    with (
        patch.object(workspace, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case, policy)
    compose.assert_not_called()


@pytest.mark.parametrize(
    "field", ["best_checkpoint_sha256", "teacher_configuration_sha256", "test_corpus_sha256"]
)
def test_rehashed_wrong_source_binding_rejects_before_loading(
    case: CaptureCase, field: str
) -> None:
    case = _recapture(case, **{field: "0" * 64})
    with (
        patch.object(workspace, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)
    compose.assert_not_called()


def test_inference_export_must_match_selected_best(case: CaptureCase) -> None:
    case = _recapture(case, inference_sha256="0" * 64)
    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward") as forward,
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)
    forward.assert_not_called()


def test_rehashed_saved_nll_is_actually_reproduced(case: CaptureCase) -> None:
    rows = tuple(
        r.model_copy(update={"target_nll_sum": r.target_nll_sum + 1.0})
        for r in case.capture.payload.rows
    )
    case = _recapture(case, rows=rows)
    with pytest.raises(IrisScientificEvaluationError):
        _measure(case)


def test_changed_initial_recipe_rejects_before_teacher_execution(case: CaptureCase) -> None:
    _edit(case.recipe, lambda raw: raw.update(initialization_seed=99))
    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward") as forward,
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)
    forward.assert_not_called()


def test_failure_restores_rng_and_redacts_private_content(case: CaptureCase) -> None:
    rng, before = capture_runtime_state(torch.device("cpu")), _files(case.recipe.parent)

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(9)
        raise RuntimeError(case.rows[0].prompt)

    with (
        patch.object(workspace, "measure_workspace_causal_v1", side_effect=fail),
        pytest.raises(IrisScientificEvaluationError) as error,
    ):
        _measure(case)
    assert case.rows[0].prompt not in str(error.value)
    assert error.value.__suppress_context__
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert _files(case.recipe.parent) == before


def test_late_recipe_change_aborts_after_actual_measurement(case: CaptureCase) -> None:
    actual = measure_workspace_causal_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = actual(*args, **kwargs)
        case.recipe.write_bytes(case.recipe.read_bytes() + b" ")
        return result

    with (
        patch.object(workspace, "measure_workspace_causal_v1", side_effect=changed),
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)


def test_wrong_causal_context_rejects_even_with_well_formed_capture(case: CaptureCase) -> None:
    actual = measure_workspace_causal_v1

    def wrong(*args: Any, **kwargs: Any) -> Any:
        return actual(*args, **kwargs).model_copy(update={"sidecar_state_sha256": "0" * 64})

    with (
        patch.object(workspace, "measure_workspace_causal_v1", side_effect=wrong),
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)


@pytest.mark.parametrize(
    "field",
    [
        "run_seed",
        "best_checkpoint_sha256",
        "teacher_graph_sha256",
        "teacher_sidecar_sha256",
        "executions",
        "trace_scalar_bound",
        "purpose",
    ],
)
def test_report_rejects_rehashed_mismatching_context(
    reported: IrisScientificWorkspaceV1, field: str
) -> None:
    raw = reported.payload.model_dump()
    raw[field] = (
        "scientific" if field == "purpose" else 99 if isinstance(raw[field], int) else "0" * 64
    )
    with pytest.raises(ValueError):
        ScientificWorkspacePayloadV1.model_validate(raw)


def test_report_rejects_forged_aggregate_and_subset(reported: IrisScientificWorkspaceV1) -> None:
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda raw: raw["groups"][0].update(selected_examples=99),
        lambda raw: raw.update(rows=tuple(reversed(raw["rows"]))),
        lambda raw: raw["causal_summary"]["pairs"][0].update(sign_test_p=0.0),
    )
    for mutate in mutations:
        raw = reported.payload.model_dump()
        mutate(raw)
        with pytest.raises(ValueError):
            ScientificWorkspacePayloadV1.model_validate(raw)


def test_store_roundtrip_quota_and_lost_rename_ack(
    reported: IrisScientificWorkspaceV1, tmp_path: Path
) -> None:
    with pytest.raises(IrisScientificEvaluationError):
        workspace.write_scientific_workspace_v1(
            reported, root=tmp_path / "tiny", maximum_storage_bytes=1
        )

    def renamed(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        raise OSError("lost acknowledgment")

    with patch.object(workspace, "rename_directory_with_windows_retry", side_effect=renamed):
        path = workspace.write_scientific_workspace_v1(reported, root=tmp_path / "store")
    assert workspace.read_scientific_workspace_v1(path) == reported
    assert workspace.write_scientific_workspace_v1(reported, root=tmp_path / "store") == path
    assert tuple(path.parent.iterdir()) == (path,)


@pytest.mark.parametrize("corruption", ["duplicate", "noncanonical", "aggregate", "extra"])
def test_reader_rejects_corruption_including_rehashed_aggregate(
    reported: IrisScientificWorkspaceV1, tmp_path: Path, corruption: str
) -> None:
    path = workspace.write_scientific_workspace_v1(reported, root=tmp_path)
    file = path / "workspace.json"
    if corruption == "extra":
        (path / "foreign").write_bytes(b"preserve")
    elif corruption == "duplicate":
        content = file.read_bytes()
        file.write_bytes(b'{"sha256":"' + reported.sha256.encode() + b'",' + content[1:])
    elif corruption == "noncanonical":
        file.write_bytes(file.read_bytes() + b" ")
    else:
        raw = json.loads(file.read_bytes())
        raw["payload"]["groups"][0]["selected_examples"] = 999
        raw["sha256"] = domain_sha256(
            "iris/scientific-workspace/v1", canonical_json_bytes(raw["payload"])
        )
        file.write_bytes(canonical_json_bytes(raw) + b"\n")
        new = path.with_name(raw["sha256"])
        rename_directory_with_windows_retry(path, new)
        path = new
    with pytest.raises(IrisScientificEvaluationError):
        workspace.read_scientific_workspace_v1(path)


def test_free_reserve_and_short_write_leave_no_report(
    reported: IrisScientificWorkspaceV1, tmp_path: Path
) -> None:
    with (
        patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
        pytest.raises(IrisScientificEvaluationError),
    ):
        workspace.write_scientific_workspace_v1(reported, root=tmp_path / "reserve")

    def short(path: Path, content: bytes) -> None:
        path.write_bytes(content[:-1])

    with (
        patch.object(workspace, "_write_exact_file", side_effect=short),
        pytest.raises(IrisScientificEvaluationError),
    ):
        workspace.write_scientific_workspace_v1(reported, root=tmp_path / "short")
    assert not tuple((tmp_path / "short" / workspace._STORE).iterdir())


def test_staging_collision_never_removes_foreign_files(
    reported: IrisScientificWorkspaceV1, tmp_path: Path
) -> None:
    candidate = tmp_path / workspace._STORE / ".iris-workspace-fixed.tmp"
    candidate.mkdir(parents=True)
    sentinel = candidate / "foreign"
    sentinel.write_bytes(b"preserve")
    with (
        patch.object(workspace, "uuid4", return_value=SimpleNamespace(hex="fixed")),
        pytest.raises(IrisScientificEvaluationError),
    ):
        workspace.write_scientific_workspace_v1(reported, root=tmp_path)
    assert sentinel.read_bytes() == b"preserve"


def _command_files(case: CaptureCase) -> tuple[Path, Path]:
    policy = case.recipe.parent / "diagnostic-policy.json"
    descriptor = case.recipe.parent / "test-input.json"
    policy.write_text(_POLICY.model_dump_json(), encoding="utf-8")
    descriptor.write_text(case.test.model_dump_json(), encoding="utf-8")
    return policy, descriptor


def test_real_cli_measure_and_model_free_inspection(case: CaptureCase, tmp_path: Path) -> None:
    policy, descriptor = _command_files(case)
    output = tmp_path / "cli-results"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scientific",
            "diagnose-workspace",
            "--config",
            str(case.recipe),
            "--execution",
            str(case.execution),
            "--generation",
            str(case.generation_path),
            "--test-input",
            str(descriptor),
            "--policy",
            str(policy),
            "--output-root",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not_assessed" in result.output
    (path,) = (output / workspace._STORE).iterdir()
    captured = workspace.read_scientific_workspace_v1(path)
    assert captured.payload.selected_step == 1
    with (
        patch.object(workspace, "_composed", side_effect=AssertionError("no model")),
        patch.object(workspace, "_audit_inputs", side_effect=AssertionError("no source")),
    ):
        inspected = runner.invoke(app, ["scientific", "inspect-workspace", str(path)])
    assert inspected.exit_code == 0, inspected.output
    assert "source not re-audited" in inspected.output
    assert case.rows[0].prompt not in inspected.output


@pytest.mark.parametrize(
    "bad",
    [
        b'{"selection_seed":42,"selection_seed":43}',
        b'{"selection_seed":true}',
        b'{"selection_seed":42,"unknown":1}',
        b"x" * 16385,
    ],
)
def test_bad_policy_rejected_before_model_load(
    case: CaptureCase, tmp_path: Path, bad: bytes
) -> None:
    policy, descriptor = _command_files(case)
    policy.write_bytes(bad)
    with (
        patch.object(workspace, "_composed") as compose,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        workspace_command.diagnose_scientific_workspace_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    compose.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("changed", ["policy", "test"])
def test_command_detects_late_descriptor_changes_before_publication(
    case: CaptureCase, tmp_path: Path, changed: str
) -> None:
    policy, descriptor = _command_files(case)

    def measure(**kwargs: Any) -> IrisScientificWorkspaceV1:
        result = workspace.measure_scientific_workspace_v1(**kwargs)
        target = policy if changed == "policy" else descriptor
        target.write_bytes(target.read_bytes() + b" ")
        return result

    with (
        patch.object(workspace_command, "measure_scientific_workspace_v1", side_effect=measure),
        patch.object(workspace_command, "write_scientific_workspace_v1") as write,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        workspace_command.diagnose_scientific_workspace_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    write.assert_not_called()


def test_output_store_cannot_overlap_source(case: CaptureCase) -> None:
    policy, descriptor = _command_files(case)
    before = _files(case.execution)
    with (
        patch.object(workspace_command, "measure_scientific_workspace_v1") as measure,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        workspace_command.diagnose_scientific_workspace_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=case.execution,
        )
    measure.assert_not_called()
    assert _files(case.execution) == before


def test_public_api_is_canonical() -> None:
    import iris.application as application

    assert application.measure_scientific_workspace_v1 is workspace.measure_scientific_workspace_v1
    assert application.read_scientific_workspace_v1 is workspace.read_scientific_workspace_v1
    assert application.write_scientific_workspace_v1 is workspace.write_scientific_workspace_v1
    assert (
        application.diagnose_scientific_workspace_command_v1
        is workspace_command.diagnose_scientific_workspace_command_v1
    )


def test_actual_subset_runs_only_selected_rows_but_audits_unselected_members(
    case: CaptureCase,
) -> None:
    more = generate_synthetic_split(
        SyntheticSplitPlan(split="test", first_seed=2000100, count=60),
        template_id="iris-synthetic-01-v2",
    )
    first = case.rows[0]
    extras = tuple(
        r for r in more if (r.task_type, r.difficulty) == (first.task_type, first.difficulty)
    )[:3]
    assert len(extras) == 3
    rows = (*case.rows, *extras)
    spec = _test_spec(case.recipe.parent / "extended.parquet", rows)
    capture = _generate((case.recipe, case.execution), spec)
    path = generation.write_scientific_generation_v1(capture, root=case.recipe.parent / "extended")
    case = replace(case, rows=rows, test=spec, capture=capture, generation_path=path)
    seen: list[str] = []
    actual = IrisScientificTeacherV1.evaluation_forward

    def forward(self: IrisScientificTeacherV1, example: Any, **kwargs: Any) -> Any:
        seen.append(example.id)
        return actual(self, example, **kwargs)

    with patch.object(IrisScientificTeacherV1, "evaluation_forward", new=forward):
        report = _measure(case, _POLICY.model_copy(update={"causal": None}))
    assert len(report.payload.members) == 7 and len(report.payload.rows) == 4
    assert seen == [r.example_id for r in report.payload.rows]
    assert report.payload.groups[0].available_examples == 7
    unselected = next(r.example_id for r in capture.payload.rows if r.example_id not in seen)
    altered = tuple(
        r.model_copy(update={"example_sha256": "0" * 64}) if r.example_id == unselected else r
        for r in capture.payload.rows
    )
    case = _recapture(case, rows=altered)
    with (
        patch.object(workspace, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        _measure(case)
    compose.assert_not_called()
