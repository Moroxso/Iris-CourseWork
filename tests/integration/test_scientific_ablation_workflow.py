"""Actual original-factory compound CPU workflow; no scientific/GPU acceptance."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_collapse as native_collapse
from iris.application import scientific_ablation_workflow as workflow
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_collapse_v1 import ScientificAblationCollapsePolicyV1
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutRowV1
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    ScientificAblationWorkflowParquetV1,
    ScientificAblationWorkflowResourcesV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_generation_context import _fresh
from tests.integration.test_scientific_ablation_generation_source import _prepared, _read
from tests.integration.test_scientific_generation import _test_spec
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_teacher import _rows as training_rows
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Case:
    root: Path
    plan: IrisScientificAblationWorkflowPlanV1


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    root = tmp_path_factory.mktemp("compound-native-original-source")
    prepared = _prepared(root / "source")
    bundle = _read(prepared)
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row
        for task in TASKS
        for row in tuple(
            candidate
            for candidate in pool
            if candidate.task_type == task and candidate.difficulty == 1
        )[:2]
    )
    assert len(rows) == 8 and {row.task_type for row in rows} == set(TASKS)
    train = _test_spec(root / "train.parquet", training_rows()[:5])
    valid = _test_spec(root / "validation.parquet", _validation())
    test = _test_spec(root / "test.parquet", rows)
    intent = ScientificAblationWorkflowIntentV1(
        source_sha256=bundle.manifest.sha256,
        lifecycle_sha256=bundle.lifecycle.manifest.sha256,
        inference_sha256=bundle.inference.manifest.sha256,
        selected_checkpoint_sha256=bundle.manifest.payload.best_checkpoint_sha256,
        lifecycle_path=str(prepared.native_path),
        inference_path=str(prepared.inference_path),
        inputs_root=str(root),
        training=ScientificAblationWorkflowParquetV1.model_validate(
            train.model_dump(mode="python")
        ),
        validation=ScientificAblationWorkflowParquetV1.model_validate(
            valid.model_dump(mode="python")
        ),
        test=ScientificAblationWorkflowParquetV1.model_validate(test.model_dump(mode="python")),
        decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=42),
        controls=(WorkspaceInterventionPlanV1(interventions=()),),
        collapse=ScientificAblationCollapsePolicyV1(),
        workspace=ScientificAblationWorkspaceCausalPolicyV1(selection_seed=42, shuffle_seed=1337),
        message=ScientificAblationMessageCausalPolicyV1(
            selection_seed=42, shuffle_seed=1337, examples_per_stratum=2
        ),
        resources=ScientificAblationWorkflowResourcesV1(
            maximum_storage_bytes=4 * 1024**3, minimum_free_bytes=0
        ),
        python_policy=_POLICY,
    )
    return Case(root, build_scientific_ablation_workflow_plan_v1(intent))


@dataclass(frozen=True)
class Completed:
    store: ScientificAblationWorkflowStoreV1
    anchors: tuple[str, ...]


@pytest.fixture(scope="module")
def completed(case: Case) -> Completed:
    job = workflow.create_scientific_ablation_workflow_v1(
        root=case.root / "complete",
        plan=case.plan,
        make_fresh_run=_fresh,
    )
    with (
        patch.object(
            native_collapse,
            "measure_scientific_ablation_collapse_completed_v1",
            wraps=native_collapse.measure_scientific_ablation_collapse_completed_v1,
        ) as diagnostic,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("Tiny native outputs must not invoke candidate jobs"),
        ) as candidate,
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No pretrained load"),
        ),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer load"),
        ),
    ):
        progress = job.progress
        for index in range(8):
            assert len(progress.completed_phase_sha256) == index
            progress = job.advance()
        assert progress.complete and progress.next_phase is None
        assert diagnostic.call_count == 1 and candidate.call_count == 0
    (case.root / "complete-workflow-anchors.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                complete_native_workflow_engineering_only=True,
                scientific_acceptance="not_assessed",
            )
        )
    )
    return Completed(job._seal.store, progress.completed_phase_sha256)


def _copy_prefix(
    case: Case, completed: Completed, root: Path, count: int
) -> ScientificAblationWorkflowStoreV1:
    """Copy independently anchored DATA only; never pretend it restores an owner."""
    store = ScientificAblationWorkflowStoreV1(root, case.plan)
    with store.claim():
        pass
    for phase in scientific_ablation_workflow_phases_v1(case.plan)[:count]:
        shutil.copy2(completed.store.path / f"phase-{phase.index:03d}.attempt.json", store.path)
        shutil.copytree(
            completed.store.path / f"phase-{phase.index:03d}",
            store.path / f"phase-{phase.index:03d}",
        )
        if phase.kind in ("capture", "score"):
            shutil.copytree(
                completed.store.path / "kernels" / phase.name, store.path / "kernels" / phase.name
            )
    return store


def test_complete_original_ordinary_controlled_score_diagnostics_both_writers_and_publication(
    case: Case, completed: Completed
) -> None:
    phases = scientific_ablation_workflow_phases_v1(case.plan)
    assert len(completed.anchors) == len(phases) == 8
    for phase, digest in zip(phases, completed.anchors, strict=True):
        data = completed.store.read(phase, expected_sha256=digest)
        assert data.record.payload.native_execution_authority == "none_data_only"
    for name in ("workspace", "message"):
        assert (completed.store.path / "published" / name).is_dir()


def test_cold_data_publication_has_no_factory_generation_teacher_or_scoring(
    case: Case, completed: Completed, tmp_path: Path
) -> None:
    _copy_prefix(case, completed, tmp_path, 7)
    with patch.object(
        steps, "_owner", side_effect=AssertionError("DATA publication must not reconstruct a model")
    ):
        cold = workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path, plan=case.plan, completed_phase_sha256=completed.anchors[:7]
        )
        result = cold.advance()
        assert result.complete and result.recovered_prefix_is_data_only
        assert cold.advance() == result and cold.progress == result


def test_ready_data_without_independent_phase_anchor_is_not_adopted(
    case: Case, completed: Completed, tmp_path: Path
) -> None:
    _copy_prefix(case, completed, tmp_path, 5)
    with patch.object(
        steps, "_owner", side_effect=AssertionError("Do not remeasure stored ready DATA")
    ):
        cold = workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path, plan=case.plan, completed_phase_sha256=completed.anchors[:4]
        )
        with pytest.raises(IrisScientificAblationWorkflowUncertainError):
            cold.advance()


def test_native_attempt_is_uncertain_and_never_remeasured(
    case: Case, completed: Completed, tmp_path: Path
) -> None:
    store = _copy_prefix(case, completed, tmp_path, 4)
    phase = scientific_ablation_workflow_phases_v1(case.plan)[4]
    with store.claim() as claim:
        store.begin(phase, completed.anchors[3], claim)
    with patch.object(
        steps, "_owner", side_effect=AssertionError("Existing native attempt must not reconstruct")
    ):
        cold = workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path,
            plan=case.plan,
            make_fresh_run=_fresh,
            completed_phase_sha256=completed.anchors[:4],
        )
        with pytest.raises(IrisScientificAblationWorkflowUncertainError):
            cold.advance()


def test_native_success_then_commit_failure_retries_only_private_detached_data(
    case: Case, completed: Completed, tmp_path: Path
) -> None:
    _copy_prefix(case, completed, tmp_path, 4)
    job = workflow.create_scientific_ablation_workflow_v1(
        root=tmp_path,
        plan=case.plan,
        make_fresh_run=_fresh,
        completed_phase_sha256=completed.anchors[:4],
    )
    original = ScientificAblationWorkflowStoreV1.commit
    attempts = 0

    def commit(
        store: ScientificAblationWorkflowStoreV1,
        data: ScientificAblationWorkflowDataV1,
        claim: _Claim,
    ) -> ScientificAblationWorkflowDataV1:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("labelled storage-only failure after actual native completion")
        return original(store, data, claim)

    with patch.object(ScientificAblationWorkflowStoreV1, "commit", new=commit):
        with pytest.raises(IrisScientificAblationWorkflowStorageError):
            job.advance()
        assert job._seal.ready is not None
        with patch.object(
            steps, "_owner", side_effect=AssertionError("No second native owner/measurement")
        ):
            result = job.advance()
    assert attempts == 2 and len(result.completed_phase_sha256) == 5


def test_actual_capture_chunk_cold_handle_resumes_only_missing_rows(
    case: Case, tmp_path: Path
) -> None:
    seen: list[int] = []
    execute = IrisScientificAblationHeldoutV1.execute_row

    def row(
        owner: IrisScientificAblationHeldoutV1, index: int
    ) -> IrisScientificAblationHeldoutRowV1:
        seen.append(index)
        return execute(owner, index)

    with patch.object(IrisScientificAblationHeldoutV1, "execute_row", new=row):
        first = workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path, plan=case.plan, make_fresh_run=_fresh
        )
        assert first.advance(rows_per_call=1).completed_phase_sha256 == ()
        second = workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path, plan=case.plan, make_fresh_run=_fresh
        )
        assert len(second.advance().completed_phase_sha256) == 1
    assert seen == list(range(8))


def test_out_of_order_physical_history_rejected_before_original_factory(
    case: Case, completed: Completed, tmp_path: Path
) -> None:
    _copy_prefix(case, completed, tmp_path, 6)
    with pytest.raises(workflow.IrisScientificAblationWorkflowError):
        workflow.create_scientific_ablation_workflow_v1(
            root=tmp_path,
            plan=case.plan,
            make_fresh_run=_fresh,
            completed_phase_sha256=completed.anchors[:2],
        )
