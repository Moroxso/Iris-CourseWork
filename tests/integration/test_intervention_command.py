"""Actual tiny-Qwen CLI pairs, exact-best provenance and model-free retry/inspection."""

import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from iris.application import intervention_command as command
from iris.application import scientific_command as training
from iris.application import scientific_delivery as delivery
from iris.application import scientific_intervention as captures
from iris.application import scientific_intervention_evaluation as evaluation
from iris.application import synthetic_scoring_v2 as scoring
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    read_scientific_checkpoint_v1,
    write_scientific_checkpoint_v1,
)
from iris.application.scientific_execution import (
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_lifecycle import _bundle_parts
from iris.cli import app
from iris.integrations.python_sandbox import PythonSandboxError
from tests.integration.test_scientific_answer_commands import _descriptor
from tests.integration.test_scientific_command import _edit, _files
from tests.integration.test_scientific_command import loaders as loaders
from tests.integration.test_scientific_evaluation import captured as captured
from tests.integration.test_scientific_intervention_evaluation import (
    PairCase,
    fake_worker,
    fixture_answers,
)
from tests.integration.test_scientific_intervention_evaluation import case as case
from tests.integration.test_scientific_intervention_evaluation import paired as paired
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = pytest.mark.integration
runner = CliRunner()


@dataclass(frozen=True)
class Prepared:
    case: PairCase
    checkpoint: Path
    context: Path
    descriptor: Path
    output: Path


def _selected(
    case: PairCase,
) -> tuple[IrisScientificCheckpointBundleV1, IrisScientificCheckpointBundleV1]:
    snapshot = read_scientific_execution_v1(case.source.execution)
    selection = None
    for event in snapshot.events:
        selection = _selected_best(selection, event)
    current, best = _bundle_parts(
        _checkpoint_for_event(
            snapshot.path, snapshot.header, snapshot.events[-1], best_event=selection
        )
    )
    assert best is not None
    return current, best


def _prepare(case: PairCase, root: Path) -> Prepared:
    _, best = _selected(case)
    checkpoint = write_scientific_checkpoint_v1(best, root=root / "checkpoints")
    context = checkpoint.with_name(checkpoint.name + ".intervention.json")
    descriptor = _descriptor(case.source, root / "separate-data")
    output = root / "output"
    context.write_text(
        json.dumps(
            {
                "kind": "iris-intervention-command-context-v1",
                "checkpoint_sha256": best.manifest.sha256,
                "config": os.path.relpath(case.source.recipe, context.parent),
                "execution": os.path.relpath(case.source.execution, context.parent),
                "suites": {
                    key: os.path.relpath(descriptor, context.parent)
                    for key in ("core", "integration")
                },
                "output_root": os.path.relpath(output, context.parent),
                "decoding": case.capture.payload.baseline.payload.decoding.model_dump(mode="json"),
                "operator_seed": 42,
                "python_policy": _POLICY.model_dump(mode="json"),
                "maximum_storage_bytes": 16 * 1024**2,
            }
        ),
        encoding="utf-8",
    )
    return Prepared(case, checkpoint, context, descriptor, output)


@pytest.fixture
def prepared(case: PairCase, tmp_path: Path) -> Prepared:
    return _prepare(case, tmp_path)


def _args(prepared: Prepared, *arguments: str) -> list[str]:
    return ["intervene", f"checkpoint={prepared.checkpoint}", *arguments]


def _redacted(output: str, case: PairCase) -> None:
    assert "PRIVATE_MARKER" not in output and "hidden_facts" not in output
    assert "expected_response" not in output
    for source, control, changed in zip(
        case.source.rows,
        case.capture.payload.baseline.payload.rows,
        case.capture.payload.rows,
        strict=True,
    ):
        assert source.prompt not in output and source.expected_response not in output
        for text in (control.output_text, changed.output_text):
            if len(text) > 16:
                assert text not in output


@pytest.mark.parametrize("operation", ["zero_workspace", "block_module_write", "gaussian_noise"])
def test_source_cli_runs_actual_paired_generation_and_scoring(
    prepared: Prepared, loaders: Any, operation: str
) -> None:
    extras = {
        "zero_workspace": ["suite=integration"],
        "block_module_write": ["intervention.module_id=3"],
        "gaussian_noise": ["intervention.sigma=0.25"],
    }[operation]
    source_before = _files(prepared.case.source.execution)
    with patch.object(scoring, "execute_python_job_v2") as sandbox:
        result = runner.invoke(app, _args(prepared, f"intervention={operation}", *extras))
    assert result.exit_code == 0, result.output
    (capture_path,) = (prepared.output / "scientific-intervention-v1").iterdir()
    actual = captures.read_scientific_intervention_v1(capture_path)
    (report_path,) = (prepared.output / "scientific-intervention-evaluation-v1").iterdir()
    report = evaluation.read_scientific_intervention_evaluation_v1(report_path)
    assert actual.payload.baseline == prepared.case.capture.payload.baseline
    assert actual.payload.plan.interventions[0].kind == operation
    assert report.payload.intervention_sha256 == actual.sha256
    assert (
        report.payload.overall.control.examples == report.payload.overall.intervened.examples == 4
    )
    assert report.payload.selected_step == 1
    assert actual.payload.baseline.payload.terminal_step == 3
    if operation == "zero_workspace":
        assert actual == prepared.case.capture
    if operation == "gaussian_noise":
        assert actual.payload.plan.interventions[0].seed == 42
    sandbox.assert_not_called()  # Tiny random model emits invalid JSON; no source is executed.
    assert _files(prepared.case.source.execution) == source_before
    assert "Paired quality rates" in result.output and "not_assessed" in result.output
    _redacted(result.output, prepared.case)


def test_generate_then_evaluate_same_store_and_inspect_without_model(
    prepared: Prepared, loaders: Any
) -> None:
    _edit(prepared.context, lambda raw: raw.update(python_policy=None))
    with patch.object(scoring, "execute_python_job_v2") as sandbox:
        generated = runner.invoke(
            app, _args(prepared, "mode=generate", "intervention=zero_workspace")
        )
    assert generated.exit_code == 0, generated.output
    sandbox.assert_not_called()
    assert "not run" in generated.output
    assert not (prepared.output / "scientific-intervention-evaluation-v1").exists()
    (path,) = (prepared.output / "scientific-intervention-v1").iterdir()
    _edit(prepared.context, lambda raw: raw.update(python_policy=_POLICY.model_dump(mode="json")))
    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch.object(command, "generate_scientific_intervention_v1") as generate,
        patch.object(scoring, "execute_python_job_v2") as sandbox,
    ):
        args = _args(prepared, "mode=evaluate", f"capture={path}")
        first = runner.invoke(app, args)
        second = runner.invoke(app, args)
        assert first.exit_code == second.exit_code == 0, first.output
        (report_path,) = (prepared.output / "scientific-intervention-evaluation-v1").iterdir()
        with patch.object(command, "_assert_best") as source_audit:
            inspected = runner.invoke(app, ["intervene", "mode=inspect", f"report={report_path}"])
        assert inspected.exit_code == 0, inspected.output
        source_audit.assert_not_called()
    for mock in (tokenizer, model, generate, sandbox):
        mock.assert_not_called()
    assert "source not re-audited" in inspected.output
    for result in (generated, first, second, inspected):
        _redacted(result.output, prepared.case)


def test_source_named_nine_file_best_keeps_strict_store_reader_unchanged(
    prepared: Prepared, loaders: Any
) -> None:
    best = read_scientific_checkpoint_v1(prepared.checkpoint)
    config = training.read_scientific_command_config_v1(prepared.case.source.recipe)
    with training._composed(config, base=prepared.case.source.recipe.parent) as lifecycle:
        bundle = delivery.build_scientific_checkpoint_delivery_v1(best, context=lifecycle._run)
    addressed = delivery.write_scientific_delivery_v1(
        bundle, root=prepared.output.parent / "exports"
    )
    named = prepared.checkpoint.parent / "igw-best"
    shutil.copytree(addressed, named)
    context = named.with_name("igw-best.intervention.json")
    context.write_bytes(prepared.context.read_bytes())
    with pytest.raises(delivery.IrisScientificDeliveryError):
        delivery.read_scientific_delivery_v1(named)
    with patch("torch.load", side_effect=AssertionError("No incoming pickle")):
        result = runner.invoke(
            app,
            _args(
                replace(prepared, checkpoint=named),
                "mode=evaluate",
                f"capture={prepared.case.path}",
            ),
        )
    assert result.exit_code == 0, result.output
    assert set(_files(named)) == set(dict(bundle.files))


@pytest.mark.parametrize("kind", ["current", "digest", "corrupt", "extra"])
def test_wrong_checkpoint_fails_before_generation(prepared: Prepared, kind: str) -> None:
    path = prepared.checkpoint
    if kind == "current":
        current, _ = _selected(prepared.case)
        path = write_scientific_checkpoint_v1(current, root=path.parent)
        _edit(prepared.context, lambda raw: raw.update(checkpoint_sha256=current.manifest.sha256))
    elif kind == "digest":
        _edit(prepared.context, lambda raw: raw.update(checkpoint_sha256="0" * 64))
    elif kind == "corrupt":
        state = path / "state.safetensors"
        state.write_bytes(state.read_bytes() + b"PRIVATE_MARKER")
    else:
        (path / "PRIVATE_MARKER").write_text("extra", encoding="utf-8")
    with patch.object(command, "generate_scientific_intervention_v1") as generate:
        result = runner.invoke(
            app,
            _args(
                replace(prepared, checkpoint=path),
                f"context={prepared.context}",
                "intervention=zero_workspace",
            ),
        )
    assert result.exit_code == 1
    generate.assert_not_called()
    assert not prepared.output.exists()
    _redacted(result.output, prepared.case)


@pytest.mark.parametrize(
    "kind",
    ["missing", "duplicate", "oversize", "extra", "seed_type", "policy", "nonlocal", "suite"],
)
def test_invalid_context_or_suite_fails_before_loading(prepared: Prepared, kind: str) -> None:
    args = _args(prepared, "intervention=zero_workspace")
    if kind == "missing":
        prepared.context.unlink()
    elif kind == "duplicate":
        text = prepared.context.read_text(encoding="utf-8")
        prepared.context.write_text(text[:-1] + ',"operator_seed":43}', encoding="utf-8")
    elif kind == "oversize":
        prepared.context.write_text("PRIVATE_MARKER" * 20000, encoding="utf-8")
    elif kind == "extra":
        _edit(prepared.context, lambda raw: raw.update(PRIVATE_MARKER="not allowed"))
    elif kind == "seed_type":
        _edit(prepared.context, lambda raw: raw.update(operator_seed="42"))
    elif kind == "policy":
        _edit(prepared.context, lambda raw: raw.update(python_policy=None))
    elif kind == "nonlocal":
        _edit(prepared.context, lambda raw: raw.update(output_root="https://PRIVATE_MARKER/output"))
    else:
        args += ["suite=PRIVATE_MARKER"]
    with patch.object(command, "generate_scientific_intervention_v1") as generate:
        result = runner.invoke(app, args)
    assert result.exit_code == 1
    generate.assert_not_called()
    assert not prepared.output.exists()
    _redacted(result.output, prepared.case)


@pytest.mark.parametrize("mode", ["run", "generate", "evaluate"])
def test_overlapping_output_store_is_rejected(prepared: Prepared, mode: str) -> None:
    _edit(prepared.context, lambda raw: raw.update(output_root=str(prepared.case.source.execution)))
    before = _files(prepared.case.source.execution)
    extra = f"capture={prepared.case.path}" if mode == "evaluate" else "intervention=zero_workspace"
    with (
        patch.object(command, "generate_scientific_intervention_v1") as generate,
        patch.object(command, "evaluate_scientific_intervention_v1") as evaluate,
    ):
        result = runner.invoke(app, _args(prepared, f"mode={mode}", extra))
    assert result.exit_code == 1
    generate.assert_not_called()
    evaluate.assert_not_called()
    assert _files(prepared.case.source.execution) == before


@pytest.mark.parametrize("kind", ["context", "recipe", "descriptor", "checkpoint"])
def test_post_generation_input_mutation_prevents_publication(prepared: Prepared, kind: str) -> None:
    def changed(**kwargs: Any) -> Any:
        if kind == "context":
            _edit(prepared.context, lambda raw: raw.update(operator_seed=1337))
        elif kind == "recipe":
            _edit(prepared.case.source.recipe, lambda raw: raw.update(run_seed=raw["run_seed"] + 1))
        elif kind == "descriptor":
            _edit(prepared.descriptor, lambda raw: raw.update(maximum_rows=99))
        else:
            (prepared.checkpoint / "state.safetensors").write_bytes(b"PRIVATE_MARKER")
        return prepared.case.capture

    with patch.object(command, "generate_scientific_intervention_v1", side_effect=changed):
        result = runner.invoke(app, _args(prepared, "mode=generate", "intervention=zero_workspace"))
    assert result.exit_code == 1
    assert not prepared.output.exists()
    _redacted(result.output, prepared.case)


def test_post_scoring_context_mutation_does_not_publish_report(prepared: Prepared) -> None:
    original = evaluation.evaluate_scientific_intervention_v1

    def changed(**kwargs: Any) -> Any:
        report = original(**kwargs)
        _edit(prepared.context, lambda raw: raw.update(operator_seed=1337))
        return report

    with patch.object(command, "evaluate_scientific_intervention_v1", side_effect=changed):
        result = runner.invoke(
            app, _args(prepared, "mode=evaluate", f"capture={prepared.case.path}")
        )
    assert result.exit_code == 1
    assert not prepared.output.exists()
    assert prepared.case.path.exists()


@pytest.mark.parametrize(
    "parameters",
    [
        ("intervention=zero_slot", "intervention.slot_id=2"),
        ("intervention=block_module_read", "intervention.module_id=4"),
        ("intervention=reduce_workspace_capacity", "intervention.capacity=3"),
        ("intervention=zero_workspace", "intervention.cycles=[2]"),
        (
            "intervention=replace_slot",
            "intervention.slot_id=0",
            "intervention.values=[0.1]",
            "intervention.source_module=0",
            "intervention.confidence=0.5",
            "intervention.ttl=1",
        ),
    ],
)
def test_actual_graph_dimensions_are_checked_before_generation(
    prepared: Prepared, parameters: tuple[str, ...]
) -> None:
    with patch.object(command, "generate_scientific_intervention_v1") as generate:
        result = runner.invoke(app, _args(prepared, *parameters))
    assert result.exit_code == 1
    generate.assert_not_called()
    assert not prepared.output.exists()


def test_explicit_relocated_context_uses_its_own_relative_paths(prepared: Prepared) -> None:
    relocated = prepared.output.parent / "elsewhere" / "context.json"
    relocated.parent.mkdir()
    raw = json.loads(prepared.context.read_text(encoding="utf-8"))
    for key in ("config", "execution", "output_root"):
        raw[key] = os.path.relpath(prepared.context.parent / raw[key], relocated.parent)
    raw["suites"] = {
        key: os.path.relpath(prepared.context.parent / value, relocated.parent)
        for key, value in raw["suites"].items()
    }
    relocated.write_text(json.dumps(raw), encoding="utf-8")
    prepared.context.unlink()
    result = runner.invoke(
        app,
        _args(
            prepared,
            f"context={relocated}",
            "suite=integration",
            "mode=evaluate",
            f"capture={prepared.case.path}",
        ),
    )
    assert result.exit_code == 0, result.output
    assert len(list((prepared.output / "scientific-intervention-evaluation-v1").iterdir())) == 1


def test_saved_pair_must_match_context_decoding(prepared: Prepared) -> None:
    _edit(prepared.context, lambda raw: raw["decoding"].update(seed=2026))
    with patch.object(command, "evaluate_scientific_intervention_v1") as evaluate:
        result = runner.invoke(
            app, _args(prepared, "mode=evaluate", f"capture={prepared.case.path}")
        )
    assert result.exit_code == 1
    evaluate.assert_not_called()
    assert not prepared.output.exists()


def test_context_parent_symlink_is_rejected(prepared: Prepared) -> None:
    alias = prepared.output.parent / "context-alias"
    try:
        alias.symlink_to(prepared.context.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    with patch.object(command, "generate_scientific_intervention_v1") as generate:
        result = runner.invoke(
            app,
            _args(
                prepared,
                f"context={alias / prepared.context.name}",
                "intervention=zero_workspace",
            ),
        )
    assert result.exit_code == 1
    generate.assert_not_called()
    assert not prepared.output.exists()


def test_scoring_failure_retains_capture_for_model_free_retry(
    case: PairCase, tmp_path: Path
) -> None:
    case = fixture_answers(case, degrade=True)  # Deliberate scorer inputs, not learned answers.
    prepared = _prepare(case, tmp_path)
    with (
        patch.object(command, "generate_scientific_intervention_v1", return_value=case.capture),
        patch.object(
            scoring,
            "execute_python_job_v2",
            AsyncMock(side_effect=PythonSandboxError("PRIVATE_MARKER")),
        ),
    ):
        failed = runner.invoke(app, _args(prepared, "intervention=zero_workspace"))
    assert failed.exit_code == 3, failed.output
    assert "mode=evaluate" in failed.output and "no complete report" in failed.output
    _redacted(failed.output, case)
    (path,) = (prepared.output / "scientific-intervention-v1").iterdir()
    assert captures.read_scientific_intervention_v1(path) == case.capture
    assert not (prepared.output / "scientific-intervention-evaluation-v1").exists()
    worker = fake_worker(case)
    with (
        patch.object(command, "generate_scientific_intervention_v1") as generate,
        patch.object(scoring, "execute_python_job_v2", worker),
    ):
        retry = runner.invoke(app, _args(prepared, "mode=evaluate", f"capture={path}"))
    assert retry.exit_code == 0, retry.output
    generate.assert_not_called()
    assert worker.await_count == 2
    (report_path,) = (prepared.output / "scientific-intervention-evaluation-v1").iterdir()
    report = evaluation.read_scientific_intervention_evaluation_v1(report_path)
    assert report.payload.overall.control.correct == 4
    assert report.payload.overall.intervened.correct == 1
    assert report.payload.overall.relative_task_success_drop == 0.75
    _redacted(retry.output, case)


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="Explicit installed Windows/WSL gVisor source-v2 opt-in required",
)
def test_real_gvisor_source_cli_evaluates_both_controlled_branches(
    case: PairCase, tmp_path: Path
) -> None:
    case = fixture_answers(case, degrade=True)  # Controlled scorer contract, NOT learned quality.
    prepared = _prepare(case, tmp_path)
    with patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model:
        result = runner.invoke(app, _args(prepared, "mode=evaluate", f"capture={case.path}"))
    assert result.exit_code == 0, result.output
    model.assert_not_called()
    (path,) = (prepared.output / "scientific-intervention-evaluation-v1").iterdir()
    report = evaluation.read_scientific_intervention_evaluation_v1(path)
    assert report.payload.overall.control.correct == 4
    assert report.payload.overall.intervened.correct == 1
    repair = next(
        row for row in report.payload.rows if row.control.task_type == "split_spec_repair"
    )
    assert repair.control.execution is not None and repair.intervened.execution is not None
    assert repair.control.execution.policy == repair.intervened.execution.policy == _POLICY
    _redacted(result.output, case)
