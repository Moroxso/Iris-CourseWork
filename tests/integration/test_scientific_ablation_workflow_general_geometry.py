"""Actual general dimensions/all24C1C5 modes; never replaces original four-family71."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from iris.ablations.controls import CognitiveAblationControlPlanV1
from iris.application import scientific_ablation_collapse as native_collapse
from iris.application import scientific_ablation_controlled_capture as controlled
from iris.application import scientific_ablation_writer_jobs as writer_jobs
from iris.application import scientific_ablation_writer_reports as writer_reports
from iris.application.scientific_ablation_workflow import create_scientific_ablation_workflow_v1
from iris.artifacts.scientific_ablation_collapse_v1 import IrisScientificAblationCollapseV1
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowCaptureDataV1,
    ScientificAblationWorkflowScoreDataV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_general_geometry_setup import GEOMETRIES, Geometry, prepare

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "geometry",
    GEOMETRIES,
    ids=tuple(geometry.key for geometry in GEOMETRIES),
)
def test_actual_general_geometry_modes_two_nonempty_controls_complete_pipeline(
    tmp_path: Path,
    geometry: Geometry,
) -> None:
    case = prepare(tmp_path / "case", geometry)
    phases = scientific_ablation_workflow_phases_v1(case.plan)
    assert len(phases) == 10
    job = create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow",
        plan=case.plan,
        make_fresh_run=case.factory,
    )
    with (
        patch.object(
            native_collapse,
            "measure_scientific_ablation_collapse_completed_v1",
            wraps=native_collapse.measure_scientific_ablation_collapse_completed_v1,
        ) as diagnostic,
        patch.object(
            writer_jobs,
            "measure_scientific_ablation_workspace_writer_completed_v1",
            wraps=writer_reports.measure_scientific_ablation_workspace_writer_completed_v1,
        ) as workspace,
        patch.object(
            writer_jobs,
            "measure_scientific_ablation_message_writer_completed_v1",
            wraps=writer_reports.measure_scientific_ablation_message_writer_completed_v1,
        ) as message,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No generated Python in paired non-code geometry fixture"),
        ) as candidate,
    ):
        progress = job.progress
        for index in range(len(phases)):
            assert len(progress.completed_phase_sha256) == index
            progress = job.advance()
        assert progress.complete and progress.next_phase is None
        assert diagnostic.call_count == 1 and candidate.call_count == 0
        assert workspace.call_count == message.call_count == int(geometry.cycles != 1)
    store = job._seal.store
    for phase, digest in zip(phases, progress.completed_phase_sha256, strict=True):
        data = store.read(phase, expected_sha256=digest)
        assert data.record.payload.phase == phase
        assert data.record.payload.native_execution_authority == "none_data_only"
        files = dict(data.files)
        if phase.kind == "capture":
            descriptor = ScientificAblationWorkflowCaptureDataV1.model_validate_json(
                files["capture.json"]
            )
            assert descriptor.examples == 2
            if phase.stream_index is not None:
                captured = controlled.read_scientific_ablation_controlled_capture_v1(
                    store.path
                    / "kernels"
                    / phase.name
                    / "scientific-ablation-controlled-capture-v1"
                    / descriptor.plan_sha256,
                    expected_plan_sha256=descriptor.plan_sha256,
                    expected_capture_sha256=descriptor.capture_sha256,
                    maximum_storage_bytes=case.plan.payload.resources.maximum_kernel_storage_bytes,
                )
                assert captured.plan.payload.cycles == geometry.cycles and len(captured.rows) == 2
                actual_control = CognitiveAblationControlPlanV1.model_validate_json(
                    captured.plan.payload.input.payload.generation_control_plan_json,
                )
                assert actual_control.control == case.plan.payload.controls[phase.stream_index]
                assert actual_control.control is not None and actual_control.control.interventions
        elif phase.kind == "score":
            descriptor_score = ScientificAblationWorkflowScoreDataV1.model_validate_json(
                files["evaluation.json"]
            )
            assert descriptor_score.examples == 2
        elif phase.kind == "diagnostics":
            collapse = IrisScientificAblationCollapseV1.model_validate_json(files["collapse.json"])
            assert (
                collapse.payload.num_modules,
                collapse.payload.num_slots,
                collapse.payload.state_dim,
                collapse.payload.cycle_count,
                collapse.payload.writers_per_cycle,
            ) == (
                geometry.modules,
                geometry.slots,
                geometry.state_dim,
                geometry.cycles,
                geometry.winners,
            )
            assert collapse.payload.teacher_executions == 2
            assert collapse.payload.scientific_acceptance == "not_assessed"
        elif phase.kind in ("workspace", "message"):
            if geometry.cycles == 1:
                assert data.record.payload.origin == "not_applicable_no_next_cycle"
                assert (
                    set(files) == {"summary.json"}
                    and json.loads(files["summary.json"])["cycles"] == 1
                )
            else:
                assert data.record.payload.origin == "original_native_result_data"
                assert set(files) == {"source.json", "report.json", "runtime.json"}
    assert (store.path / "published").exists() is (geometry.cycles != 1)
    (tmp_path / "actual-geometry-complete.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                modules=geometry.modules,
                slots=geometry.slots,
                state_dim=geometry.state_dim,
                cycles=geometry.cycles,
                winners=geometry.winners,
                routing=geometry.routing,
                goal=geometry.goal,
                broadcast=geometry.broadcast,
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                selected_step=1,
                terminal_step=3,
                real_original_native_workflow=True,
                source_scope="tiny_actual_general_geometry_augmentation_not_scientific5000",
                full_installed_or_scientific_acceptance=False,
            )
        )
    )


@pytest.mark.parametrize(
    ("modules", "slots", "width", "cycles", "winners"),
    (
        (0, 4, 8, 3, 2),
        (4, 0, 8, 3, 2),
        (4, 4, 0, 3, 2),
        (4, 4, 8, 0, 2),
        (4, 4, 8, 3, 0),
        (1, 4, 8, 3, 2),
        (4, 1, 8, 3, 2),
        (True, 4, 8, 3, 2),
        (4, 4, 8, False, 2),
    ),
)
def test_invalid_geometry_intent_rejected_before_any_backbone(
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    winners: int,
) -> None:
    with patch(
        "tests.integration._workflow_general_geometry_setup._build",
        side_effect=AssertionError("Intent validation has no native construction"),
    ) as factory:
        with pytest.raises(ValueError):
            Geometry(modules, slots, width, cycles, winners)
        factory.assert_not_called()
