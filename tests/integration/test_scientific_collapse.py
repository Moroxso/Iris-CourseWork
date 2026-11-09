"""Every held-out row at actual tiny-Qwen selected best, compact private CLI/store."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import scientific_collapse as collapse
from iris.application import scientific_collapse_command as collapse_command
from iris.application import scientific_command as command
from iris.application import scientific_generation as generation
from iris.application.scientific_answer_command import IrisScientificAnswerCommandError
from iris.application.scientific_evaluation import IrisScientificEvaluationError
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import IrisScientificTeacherV1
from iris.artifacts.scientific_collapse_v1 import (
    IrisScientificCollapseV1,
    ScientificCollapsePayloadV1,
    ScientificCollapsePolicyV1,
    scientific_collapse_witness_v1,
)
from iris.cli import app
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceV1, summarize_workspace_traces_v1
from tests.integration.test_scientific_command import _edit, _files, _recipe
from tests.integration.test_scientific_evaluation import CaptureCase, _recapture
from tests.integration.test_scientific_generation import _generate, _test_rows, _test_spec
from tests.integration.test_scientific_workspace import local_loaders as local_loaders

pytestmark = pytest.mark.integration
_POLICY = ScientificCollapsePolicyV1()


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory, local_loaders: None) -> CaptureCase:
    root = tmp_path_factory.mktemp("collapse-source")
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
    base = _test_rows()
    available = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000100, count=60),
            template_id="iris-synthetic-01-v2",
        )
    )
    extras = tuple(
        r
        for r in available
        if (r.task_type, r.difficulty) == (base[0].task_type, base[0].difficulty)
    )[:3]
    assert len(extras) == 3
    rows = (*base, *extras)
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


def measure(
    case: CaptureCase, policy: ScientificCollapsePolicyV1 = _POLICY
) -> IrisScientificCollapseV1:
    return collapse.measure_scientific_collapse_v1(
        generation_path=case.generation_path,
        execution_path=case.execution,
        config_path=case.recipe,
        test=case.test,
        policy=policy,
    )


@pytest.fixture(scope="module")
def reported(captured: CaptureCase) -> IrisScientificCollapseV1:
    return measure(captured)


def test_actual_full_test_executes_even_overrepresented_strata_and_matches_raw_traces(
    case: CaptureCase, reported: IrisScientificCollapseV1
) -> None:
    before, rng = _files(case.recipe.parent), capture_runtime_state(torch.device("cpu"))
    traces: list[WorkspaceTraceV1] = []
    seen: list[str] = []
    actual = IrisScientificTeacherV1.evaluation_forward

    def forward(self: IrisScientificTeacherV1, example: Any, **kwargs: Any) -> Any:
        output = actual(self, example, **kwargs)
        seen.append(example.id)
        trace = output.trace.teacher.diagnostic_trace
        assert type(trace) is WorkspaceTraceV1
        traces.append(trace)
        return output

    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward", new=forward),
        patch(
            "iris.application.request.IrisRequestOrchestrator.generate",
            side_effect=AssertionError("no generation"),
        ),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("no execution"),
        ),
    ):
        result = measure(case)
    assert result == reported
    assert seen == [r.id for r in case.rows] and len(seen) == 7
    assert _files(case.recipe.parent) == before
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    p = result.payload
    assert (len(p.members), len(p.rows), p.executions, p.trace_scalar_bound) == (7, 7, 7, 1302)
    assert (p.selected_step, p.terminal_step) == (1, 3)
    assert p.best_checkpoint_sha256 == case.capture.payload.best_checkpoint_sha256
    assert (
        p.source_20_4.outcome == "not_assessed_smoke" and p.scientific_acceptance == "not_assessed"
    )
    pooled = summarize_workspace_traces_v1(tuple(traces))
    summary = p.groups[0].summary
    assert summary is not None
    for field in (
        "examples",
        "cycles",
        "allocated_writes",
        "module_win_counts",
        "slot_allocation_counts",
        "module_entropy_nats",
        "slot_entropy_nats",
        "effective_active_modules",
        "effective_active_slots",
    ):
        assert getattr(summary, field) == getattr(pooled, field)
    for row, trace, saved in zip(p.rows, traces, case.capture.payload.rows, strict=True):
        assert (
            row.ordinary_trace_sha256,
            row.ordinary_context,
            row.cycles,
        ) == scientific_collapse_witness_v1(trace)
        assert row.control_nll_sum == saved.target_nll_sum
        assert row.target_token_count == saved.target_token_count
    serialized = result.model_dump_json()
    for example in case.rows:
        assert example.prompt not in serialized and example.expected_response not in serialized
    assert all(
        key not in serialized
        for key in ('"full"', '"module_states"', '"read_attention"', '"workspace_slots"')
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_examples": 6},
        {"maximum_executions": 6},
        {"maximum_trace_scalars": 185},
        {"maximum_total_trace_scalars": 1301},
    ],
)
def test_no_budget_can_turn_full_test_into_partial_measurement(
    case: CaptureCase, changes: Any
) -> None:
    with (
        patch.object(collapse, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case, _POLICY.model_copy(update=changes))
    compose.assert_not_called()


def test_last_member_is_source_audited_before_any_model(case: CaptureCase) -> None:
    rows = (
        *case.capture.payload.rows[:-1],
        case.capture.payload.rows[-1].model_copy(update={"example_sha256": "f" * 64}),
    )
    case = _recapture(case, rows=rows)
    with (
        patch.object(collapse, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    compose.assert_not_called()


@pytest.mark.parametrize("change", ["initialization", "inference"])
def test_exact_initialization_and_best_export_checked_before_teacher(
    case: CaptureCase, change: str
) -> None:
    if change == "initialization":
        _edit(case.recipe, lambda raw: raw.update(initialization_seed=99))
    else:
        case = _recapture(case, inference_sha256="f" * 64)
    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward") as forward,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    forward.assert_not_called()


def test_last_saved_nll_is_actually_reproduced_not_just_hashed(case: CaptureCase) -> None:
    last = case.capture.payload.rows[-1]
    case = _recapture(
        case,
        rows=(
            *case.capture.payload.rows[:-1],
            last.model_copy(update={"target_nll_sum": last.target_nll_sum + 1.0}),
        ),
    )
    with pytest.raises(IrisScientificEvaluationError):
        measure(case)


def test_failure_restores_rng_redacts_text_and_preserves_source(case: CaptureCase) -> None:
    before, rng = _files(case.recipe.parent), capture_runtime_state(torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(7)
        raise RuntimeError(case.rows[0].prompt)

    with (
        patch.object(collapse, "scientific_collapse_witness_v1", side_effect=fail),
        pytest.raises(IrisScientificEvaluationError) as error,
    ):
        measure(case)
    assert case.rows[0].prompt not in str(error.value) and error.value.__suppress_context__
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert _files(case.recipe.parent) == before


def test_mutating_model_after_last_forward_fails_exact_selected_state_check(
    case: CaptureCase,
) -> None:
    actual = IrisScientificTeacherV1.evaluation_forward

    def changed(self: IrisScientificTeacherV1, example: Any, **kwargs: Any) -> Any:
        result = actual(self, example, **kwargs)
        if example.id == case.rows[-1].id:
            with torch.no_grad():
                next(p for p in self.parameters() if p.requires_grad).add_(1.0)
        return result

    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward", new=changed),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


def test_recipe_bytes_reaudited_after_real_allocation_measurement(case: CaptureCase) -> None:
    def changed(trace: WorkspaceTraceV1) -> Any:
        result = scientific_collapse_witness_v1(trace)
        if trace.options.example_id == case.rows[-1].id:
            case.recipe.write_bytes(case.recipe.read_bytes() + b" ")
        return result

    with (
        patch.object(collapse, "scientific_collapse_witness_v1", side_effect=changed),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


@pytest.mark.parametrize(
    "bad",
    ["omit", "cycles", "writer", "seed", "checkpoint", "geometry", "purpose", "gate", "group"],
)
def test_rehashed_report_cannot_hide_missing_rows_wrong_context_or_forged_gate(
    reported: IrisScientificCollapseV1, bad: str
) -> None:
    raw = reported.payload.model_dump()
    if bad == "omit":
        raw["rows"] = raw["rows"][:-1]
    elif bad == "cycles":
        raw["rows"][0]["cycles"] = raw["rows"][0]["cycles"][:-1]
    elif bad == "writer":
        raw["rows"][0]["cycles"][0]["write_modules"] = (0,)
    elif bad == "seed":
        raw["run_seed"] = 1337
    elif bad == "checkpoint":
        raw["best_checkpoint_sha256"] = "f" * 64
    elif bad == "geometry":
        raw["num_modules"] = 8
    elif bad == "purpose":
        raw["purpose"] = "scientific"
    elif bad == "gate":
        raw["source_20_4"]["outcome"] = "met"
    else:
        raw["groups"][0]["summary"]["effective_active_modules"] = 4.0
        raw["groups"][0]["summary"]["module_win_counts"] = (999,) * 4
    with pytest.raises(ValueError):
        ScientificCollapsePayloadV1.model_validate(raw)


def test_store_roundtrip_idempotence_and_lost_publication_ack(
    reported: IrisScientificCollapseV1, tmp_path: Path
) -> None:
    def lost(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        raise OSError("lost acknowledgement")

    with patch.object(collapse, "rename_directory_with_windows_retry", side_effect=lost):
        path = collapse.write_scientific_collapse_v1(reported, root=tmp_path)
    assert collapse.read_scientific_collapse_v1(path) == reported
    assert collapse.write_scientific_collapse_v1(reported, root=tmp_path) == path
    assert tuple(path.parent.iterdir()) == (path,)


@pytest.mark.parametrize("bad", ["duplicate", "noncanonical", "extra", "rehashed-count"])
def test_reader_rejects_corruption_even_with_new_payload_hash(
    reported: IrisScientificCollapseV1, tmp_path: Path, bad: str
) -> None:
    path = collapse.write_scientific_collapse_v1(reported, root=tmp_path)
    file = path / "collapse.json"
    if bad == "duplicate":
        file.write_bytes(b'{"sha256":"' + reported.sha256.encode() + b'",' + file.read_bytes()[1:])
    elif bad == "noncanonical":
        file.write_bytes(file.read_bytes() + b" ")
    elif bad == "extra":
        (path / "foreign").write_bytes(b"preserve")
    else:
        raw = json.loads(file.read_bytes())
        raw["payload"]["groups"][0]["summary"]["allocated_writes"] += 1
        raw["sha256"] = domain_sha256(collapse._DOMAIN, canonical_json_bytes(raw["payload"]))
        file.write_bytes(canonical_json_bytes(raw) + b"\n")
        renamed = path.with_name(raw["sha256"])
        rename_directory_with_windows_retry(path, renamed)
        path = renamed
    with pytest.raises(IrisScientificEvaluationError):
        collapse.read_scientific_collapse_v1(path)


def test_storage_limits_short_write_and_foreign_staging_preservation(
    reported: IrisScientificCollapseV1, tmp_path: Path
) -> None:
    with pytest.raises(IrisScientificEvaluationError):
        collapse.write_scientific_collapse_v1(
            reported, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    with (
        patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
        pytest.raises(IrisScientificEvaluationError),
    ):
        collapse.write_scientific_collapse_v1(reported, root=tmp_path / "reserve")

    def short(path: Path, content: bytes) -> None:
        path.write_bytes(content[:-1])

    with (
        patch.object(collapse, "_write_exact_file", side_effect=short),
        pytest.raises(IrisScientificEvaluationError),
    ):
        collapse.write_scientific_collapse_v1(reported, root=tmp_path / "short")
    assert not tuple((tmp_path / "short" / collapse._STORE).iterdir())
    candidate = tmp_path / "foreign" / collapse._STORE / ".iris-collapse-fixed.tmp"
    candidate.mkdir(parents=True)
    sentinel = candidate / "sentinel"
    sentinel.write_bytes(b"preserve")
    with (
        patch.object(collapse, "uuid4", return_value=SimpleNamespace(hex="fixed")),
        pytest.raises(IrisScientificEvaluationError),
    ):
        collapse.write_scientific_collapse_v1(reported, root=tmp_path / "foreign")
    assert sentinel.read_bytes() == b"preserve"


def command_files(case: CaptureCase) -> tuple[Path, Path]:
    policy, descriptor = (
        case.recipe.parent / "collapse-policy.json",
        case.recipe.parent / "test-input.json",
    )
    policy.write_text(_POLICY.model_dump_json(), encoding="utf-8")
    descriptor.write_text(case.test.model_dump_json(), encoding="utf-8")
    return policy, descriptor


def test_real_cli_complete_measurement_and_model_free_inspection(
    case: CaptureCase, tmp_path: Path
) -> None:
    policy, descriptor = command_files(case)
    output = tmp_path / "cli-output"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scientific",
            "diagnose-collapse",
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
    assert "not_assessed_smoke" in result.output and "not_assessed" in result.output
    (path,) = (output / collapse._STORE).iterdir()
    assert len(collapse.read_scientific_collapse_v1(path).payload.rows) == 7
    with (
        patch.object(collapse, "_composed", side_effect=AssertionError("no model")),
        patch.object(collapse, "_audit_inputs", side_effect=AssertionError("no source")),
    ):
        inspected = runner.invoke(app, ["scientific", "inspect-collapse", str(path)])
    assert inspected.exit_code == 0, inspected.output
    assert (
        "source not re-audited" in inspected.output and case.rows[0].prompt not in inspected.output
    )


@pytest.mark.parametrize(
    "bad",
    [
        b'{"maximum_examples":7,"maximum_examples":6}',
        b'{"maximum_examples":true}',
        b'{"trace_mode":"full"}',
        b'{"selection_seed":42}',
        b'{"minimum_effective_slots":1}',
        b"x" * 16385,
    ],
)
def test_invalid_policy_rejects_before_command_measurement(
    case: CaptureCase, tmp_path: Path, bad: bytes
) -> None:
    policy, descriptor = command_files(case)
    policy.write_bytes(bad)
    with (
        patch.object(collapse_command, "measure_scientific_collapse_v1") as measure_call,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        collapse_command.diagnose_scientific_collapse_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    measure_call.assert_not_called()


@pytest.mark.parametrize("changed", ["policy", "descriptor"])
def test_changed_raw_command_inputs_reject_before_publication(
    case: CaptureCase, reported: IrisScientificCollapseV1, tmp_path: Path, changed: str
) -> None:
    policy, descriptor = command_files(case)

    def mutated(**kwargs: Any) -> IrisScientificCollapseV1:
        target = policy if changed == "policy" else descriptor
        target.write_bytes(target.read_bytes() + b" ")
        return reported

    with (
        patch.object(collapse_command, "measure_scientific_collapse_v1", side_effect=mutated),
        patch.object(collapse_command, "write_scientific_collapse_v1") as write,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        collapse_command.diagnose_scientific_collapse_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    write.assert_not_called()


def test_output_store_does_not_overlap_immutable_sources(case: CaptureCase) -> None:
    policy, descriptor = command_files(case)
    before = _files(case.execution)
    with (
        patch.object(collapse_command, "measure_scientific_collapse_v1") as measure_call,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        collapse_command.diagnose_scientific_collapse_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=case.execution,
        )
    measure_call.assert_not_called()
    assert _files(case.execution) == before


def test_public_api_is_canonical() -> None:
    import iris.application as application

    assert application.measure_scientific_collapse_v1 is collapse.measure_scientific_collapse_v1
    assert application.read_scientific_collapse_v1 is collapse.read_scientific_collapse_v1
    assert application.write_scientific_collapse_v1 is collapse.write_scientific_collapse_v1
    assert (
        application.diagnose_scientific_collapse_command_v1
        is collapse_command.diagnose_scientific_collapse_command_v1
    )
