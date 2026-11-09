"""Actual native operator/export binding and original recipe-factory capture/score.

The paired non-Python rows augment, never replace, original full-family/native
workflow scopes. No generated candidate is executed on Windows.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import scientific_ablation_operator as operator
from iris.application import scientific_ablation_operator_export as export
from iris.application import scientific_ablation_operator_workflow as workflow
from iris.application import scientific_ablation_workflow as original_workflow
from iris.application.scientific_ablation_command_composition import (
    create_scientific_ablation_command_run_v1,
)
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_command import _rows as original_test_rows
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_execution import _tree
from tests.integration.test_scientific_ablation_execution import cache_ready as cache_ready
from tests.integration.test_scientific_ablation_operator_export import _intent
from tests.integration.test_scientific_ablation_operator_export import completed as completed
from tests.integration.test_scientific_generation import _test_spec
from tests.unit._workflow_data_setup import intent as data_intent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bound(
    completed: tuple[Path, Path],
    cache_ready: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1]:
    path, base = completed
    root = tmp_path_factory.mktemp("operator-workflow-export")
    request = _intent(path)
    receipt = export.export_scientific_ablation_operator_v1(
        path, request=request, base=base, root=root
    )
    source = workflow.read_scientific_ablation_operator_generation_source_v1(
        path, request=request, export_root=root
    )
    recipe = operator._recipe(operator.inspect_scientific_ablation_operator_v1(path).descriptor)
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row for row in pool if row.task_type == "next_tool_prediction" and row.difficulty == 1
    )[:2]
    assert len(rows) == 2 and not any(row.executable for row in rows)
    test_path = base / "operator-workflow-test.parquet"
    assert not test_path.exists()
    test = _test_spec(test_path, rows)
    raw = data_intent(base).model_dump(mode="python") | dict(
        source_sha256=source.bundle.manifest.sha256,
        lifecycle_sha256=request.payload.lifecycle_sha256,
        inference_sha256=receipt.result.payload.inference_sha256,
        selected_checkpoint_sha256=request.payload.source_checkpoint_sha256,
        lifecycle_path=str(source.lifecycle_path),
        inference_path=str(source.inference_path),
        inputs_root=str(base),
        training=recipe.train.model_dump(mode="python"),
        validation=recipe.validation.model_dump(mode="python"),
        test=test.model_dump(mode="python"),
    )
    plan = build_scientific_ablation_workflow_plan_v1(
        ScientificAblationWorkflowIntentV1.model_validate(raw)
    )
    assert cache_ready.is_dir()
    return path, base, root, plan


def test_actual_published_source_and_workflow_creation_are_read_only_and_lazy(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
) -> None:
    path, base, root, plan = bound
    request = _intent(path)
    before, exports = _tree(path), _tree(root)
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("Creation is lazy"),
        ) as factory,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
    ):
        receipt = export.inspect_scientific_ablation_operator_export_v1(
            path, request=request, root=root
        )
        source = workflow.read_scientific_ablation_operator_generation_source_v1(
            path, request=request, export_root=root
        )
        assert source.export_result_sha256 == receipt.result.sha256
        assert (
            workflow.bind_scientific_ablation_operator_workflow_plan_v1(
                path, request=request, export_root=root, plan=plan, base=base
            )
            == plan
        )
        job = workflow.create_scientific_ablation_operator_workflow_v1(
            path,
            request=request,
            export_root=root,
            plan=plan,
            base=base,
            root=tmp_path / "workflows",
        )
        progress = job.progress
    assert factory.call_count == graph.call_count == 0
    assert progress.next_phase == "ordinary-capture" and progress.completed_phase_sha256 == ()
    assert (
        not (tmp_path / "workflows").exists() and _tree(path) == before and _tree(root) == exports
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "lifecycle_sha256",
        "inference_sha256",
        "selected_checkpoint_sha256",
        "training",
        "validation",
        "inputs_root",
        "inference_path",
        "lifecycle_path",
    ],
)
def test_resealed_wrong_workflow_binding_denies_before_model_or_output(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
    field: str,
) -> None:
    path, base, root, plan = bound
    raw = plan.payload.model_dump(mode="python")
    if field in {"training", "validation"}:
        raw[field]["sha256"] = "0" * 64
    elif field.endswith("sha256"):
        raw[field] = "0" * 64
    else:
        raw[field] = str(tmp_path / "foreign")
    wrong = build_scientific_ablation_workflow_plan_v1(
        ScientificAblationWorkflowIntentV1.model_validate(raw)
    )
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1"
        ) as factory,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        workflow.create_scientific_ablation_operator_workflow_v1(
            path,
            request=_intent(path),
            export_root=root,
            plan=wrong,
            base=base,
            root=tmp_path / "workflows",
        )
    assert factory.call_count == 0 and not (tmp_path / "workflows").exists()


def test_inspection_never_publishes_missing_export_or_adopts_complete_staging(
    completed: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    path, base = completed
    request = _intent(path)
    root = tmp_path / "exports"
    with (
        patch(
            "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1"
        ) as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.inspect_scientific_ablation_operator_export_v1(path, request=request, root=root)
    assert factory.call_count == 0 and not root.exists()
    with (
        patch(
            "iris.application.scientific_ablation_operator_export.rename_directory_with_windows_retry",
            side_effect=OSError("preserved stage"),
        ),
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.export_scientific_ablation_operator_v1(path, request=request, base=base, root=root)
    before = _tree(root)
    with (
        patch(
            "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1"
        ) as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.inspect_scientific_ablation_operator_export_v1(path, request=request, root=root)
    assert factory.call_count == 0 and _tree(root) == before


def test_nonterminal_best_export_is_not_generation_source_before_export_read(
    completed: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    path, _ = completed
    request = _intent(path, index=0)
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.inspect_scientific_ablation_operator_export_v1",
            side_effect=AssertionError("Do not read nonterminal export"),
        ) as inspect,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        workflow.read_scientific_ablation_operator_generation_source_v1(
            path, request=request, export_root=tmp_path / "exports"
        )
    assert inspect.call_count == 0 and not (tmp_path / "exports").exists()


@pytest.mark.parametrize("where", ["operator", "export", "export_namespace"])
def test_workflow_outputs_cannot_overlap_immutable_sources(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    where: str,
) -> None:
    path, base, root, plan = bound
    request = _intent(path)
    receipt = export.inspect_scientific_ablation_operator_export_v1(
        path, request=request, root=root
    )
    output = {
        "operator": path,
        "export": receipt.path,
        "export_namespace": root / "scientific-ablation-operator-exports-v1",
    }[where]
    before_source, before_export = _tree(path), _tree(root)
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1"
        ) as factory,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        workflow.create_scientific_ablation_operator_workflow_v1(
            path, request=request, export_root=root, plan=plan, base=base, root=output
        )
    assert factory.call_count == 0 and _tree(path) == before_source and _tree(root) == before_export


def test_actual_recipe_factory_capture_and_data_only_original_score(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
) -> None:
    path, base, export_root, plan = bound
    request = _intent(path)
    before = _tree(path)
    root = tmp_path / "workflows"
    job = workflow.create_scientific_ablation_operator_workflow_v1(
        path, request=request, export_root=export_root, plan=plan, base=base, root=root
    )
    original = Qwen3ForCausalLM.forward
    forwards = []

    def forward(self: Qwen3ForCausalLM, *args: Any, **kwargs: Any) -> Any:
        forwards.append(1)
        return original(self, *args, **kwargs)

    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1",
            wraps=create_scientific_ablation_command_run_v1,
        ) as factory,
        patch.object(Qwen3ForCausalLM, "forward", new=forward),
    ):
        first = job.advance(rows_per_call=1)
        assert first.next_phase == "ordinary-capture" and first.completed_phase_sha256 == ()
        captured = job.advance(rows_per_call=1)
    assert factory.call_count == 2 and len(forwards) > 0
    assert len(captured.completed_phase_sha256) == 1 and captured.next_phase == "ordinary-score"
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("No model on score"),
        ) as factory,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No score forward")
        ) as score_forward,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidates on Windows"),
        ) as candidate,
    ):
        resumed = workflow.create_scientific_ablation_operator_workflow_v1(
            path,
            request=request,
            export_root=export_root,
            plan=plan,
            base=base,
            root=root,
            completed_phase_sha256=captured.completed_phase_sha256,
        )
        partial_score = resumed.advance(rows_per_call=1)
        assert partial_score.next_phase == "ordinary-score"
        assert partial_score.completed_phase_sha256 == captured.completed_phase_sha256
        scored = resumed.advance(rows_per_call=1)
    assert factory.call_count == score_forward.call_count == candidate.call_count == 0
    assert len(scored.completed_phase_sha256) == 2 and scored.next_phase == "diagnostics"
    assert _tree(path) == before
    (tmp_path / "actual-operator-workflow-binding-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                export_root=str(export_root),
                workflow_root=str(root),
                plan=plan.model_dump(mode="json"),
                capture_anchors=list(captured.completed_phase_sha256),
                score_anchors=list(scored.completed_phase_sha256),
                concrete_recipe_factory_calls=2,
                actual_qwen_forwards=len(forwards),
                data_only_score_no_model_or_candidate=True,
                paired_nonpython_rows=2,
                installed_or_scientific_acceptance=False,
            )
        )
    )


@pytest.fixture(scope="module")
def preflight_captured(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, tuple[str, ...]]:
    """NEW real capture, never an adopted old root or invented phase prefix."""
    path, base, exports, plan = bound
    root = tmp_path_factory.mktemp("operation-preflight-real-capture")
    job = workflow.create_scientific_ablation_operator_workflow_v1(
        path, request=_intent(path), export_root=exports, plan=plan, base=base, root=root
    )
    job.advance(rows_per_call=1)
    captured = job.advance(rows_per_call=1)
    assert len(captured.completed_phase_sha256) == 1 and captured.next_phase == "ordinary-score"
    return root, captured.completed_phase_sha256


def _operation_job(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
    phase: str,
    captured: tuple[Path, tuple[str, ...]],
) -> original_workflow.IrisScientificAblationWorkflowV1:
    path, base, exports, plan = bound
    root, anchors = captured if phase == "score" else (tmp_path / "workflows", ())
    return workflow.create_scientific_ablation_operator_workflow_v1(
        path,
        request=_intent(path),
        export_root=exports,
        plan=plan,
        base=base,
        root=root,
        completed_phase_sha256=anchors,
    )


def _operation_fault_path(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1], target: str
) -> Path:
    path, base, exports, plan = bound
    if target == "test-corpus":
        return base / plan.payload.test.path
    if target == "operator":
        return path / "operator.json"
    receipt = export.inspect_scientific_ablation_operator_export_v1(
        path, request=_intent(path), root=exports
    )
    return receipt.path / "receipt.json"


@contextmanager
def _physical_fault(path: Path, witness: Path) -> Iterator[None]:
    """Corrupt ONLY this run's fixture, retain both versions, then restore it.

    No historical native DATA is touched. Restoration errors never replace a
    primary signal/failure. This is fault injection, not a concurrent OS race.
    """
    assert path.resolve().is_relative_to(witness.parent.parent.resolve())
    before = path.read_bytes()
    identity = (path.stat().st_dev, path.stat().st_ino)
    assert path.stat().st_nlink == 1 and not path.is_symlink()
    changed = b"deliberately-invalid-operation-preflight-source\n"
    witness.mkdir()
    (witness / "original.bin").write_bytes(before)
    (witness / "fault.bin").write_bytes(changed)
    primary: BaseException | None = None
    try:
        path.write_bytes(changed)
        assert path.read_bytes() == changed
        yield
    except BaseException as error:
        primary = error
    try:
        assert (path.stat().st_dev, path.stat().st_ino) == identity
        path.write_bytes(before)
        assert path.read_bytes() == before
    except BaseException as cleanup:
        if primary is not None:
            raise primary from cleanup
        raise
    if primary is not None:
        raise primary
    (witness / "restored.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                path=str(path),
                original_sha256=sha256(before).hexdigest(),
                fault_sha256=sha256(changed).hexdigest(),
                restored_sha256=sha256(path.read_bytes()).hexdigest(),
                original_inode_preserved=True,
                physical_fault_only_not_OS_race=True,
            )
        )
    )


@pytest.mark.parametrize("phase", ("capture", "score"))
@pytest.mark.parametrize("target", ("test-corpus", "operator", "export-receipt"))
def test_physical_source_change_after_creation_denies_before_claim_or_model(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    preflight_captured: tuple[Path, tuple[str, ...]],
    tmp_path: Path,
    phase: str,
    target: str,
) -> None:
    job = _operation_job(bound, tmp_path, phase, preflight_captured)
    before = _tree(job._seal.path) if job._seal.path.exists() else {}
    changed = _operation_fault_path(bound, target)
    with (
        _physical_fault(changed, tmp_path / "physical-fault"),
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(workflow, "create_scientific_ablation_command_run_v1") as factory,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        job.advance(rows_per_call=1)
    assert claim.call_count == factory.call_count == graph.call_count == candidate.call_count == 0
    assert (_tree(job._seal.path) if job._seal.path.exists() else {}) == before
    assert job._state == "idle"


@pytest.mark.parametrize("phase", ("capture", "score"))
@pytest.mark.parametrize("target", ("operator", "export-receipt"))
def test_physical_source_change_under_real_claim_denies_before_phase_attempt(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    preflight_captured: tuple[Path, tuple[str, ...]],
    tmp_path: Path,
    phase: str,
    target: str,
) -> None:
    job = _operation_job(bound, tmp_path, phase, preflight_captured)
    changed = _operation_fault_path(bound, target)
    original_claim = ScientificAblationWorkflowStoreV1.claim

    @contextmanager
    def held_claim(store: ScientificAblationWorkflowStoreV1) -> Iterator[_Claim]:
        assert store is job._seal.store
        with original_claim(store) as actual, _physical_fault(changed, tmp_path / "physical-fault"):
            actual.check()
            yield actual

    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim", new=held_claim),
        patch.object(workflow, "_bind", wraps=workflow._bind) as bind,
        patch.object(ScientificAblationWorkflowStoreV1, "begin") as begin,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(workflow, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(IrisScientificAblationWorkflowStorageError),
    ):
        job.advance(rows_per_call=1)
    assert bind.call_count == 2 and begin.call_count == factory.call_count == graph.call_count == 0
    assert not (job._seal.path / f"phase-{len(job._seal.anchors):03d}.attempt.json").exists()
    assert job._state == "idle"


@pytest.mark.parametrize("fault", (KeyboardInterrupt(), SystemExit(29)))
def test_real_source_preflight_primary_signal_inside_claim_stays_primary(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    preflight_captured: tuple[Path, tuple[str, ...]],
    tmp_path: Path,
    fault: BaseException,
) -> None:
    job = _operation_job(bound, tmp_path, "score", preflight_captured)
    calls = 0

    def read_then_signal(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fault
        return original_test_rows(*args, **kwargs)

    with (
        patch.object(workflow, "_rows", new=read_then_signal),
        patch.object(ScientificAblationWorkflowStoreV1, "begin") as begin,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(workflow, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(type(fault)) as raised,
    ):
        job.advance(rows_per_call=1)
    assert raised.value is fault and calls == 2
    assert begin.call_count == graph.call_count == factory.call_count == 0 and job._state == "idle"
