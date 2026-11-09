"""Actual complete C1/C5 controlled workflows through explicit expected phase/prefix.

All four original paired geometries augment, never replace, the full29/four-family
or original scientific5000 scopes. Reopening here is same-process, not cold OS.
"""

import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowError,
    create_scientific_ablation_workflow_v1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, Geometry, prepare
from tests.integration.test_scientific_ablation_execution import _tree

pytestmark = pytest.mark.integration


def _snapshot(root: Path) -> dict[str, bytes] | None:
    """Creation is read-only: explicit absence differs from an existing empty tree."""
    return _tree(root) if os.path.lexists(root) else None


@pytest.mark.parametrize(
    "geometry", GEOMETRIES, ids=("c1-learned", "c1-uniform", "c5-learned", "c5-random")
)
def test_actual_full_pipeline_explicit_phase_prefix_and_one_row_chunks(
    tmp_path: Path, geometry: Geometry
) -> None:
    case = prepare(tmp_path / "case", geometry)
    phases = scientific_ablation_workflow_phases_v1(case.plan)
    assert len(phases) == 10 and len(case.plan.payload.controls) == 2
    root = tmp_path / "workflow"
    anchors: tuple[str, ...] = ()
    records: list[dict[str, Any]] = []
    with patch(
        "iris.application.synthetic_scoring_v2.execute_python_job_v2",
        side_effect=AssertionError("No generated Python on the host in paired non-code tests"),
    ) as candidate:
        for index, phase in enumerate(phases):
            job = create_scientific_ablation_workflow_v1(
                root=root,
                plan=case.plan,
                make_fresh_run=None if phase.kind in ("score", "publish") else case.factory,
                completed_phase_sha256=anchors,
            )
            before = _snapshot(job._seal.path)
            wrong = phases[(index + 1) % len(phases)]
            with (
                patch.object(steps, "execute") as execute,
                pytest.raises(IrisScientificAblationWorkflowError),
            ):
                job.advance_expected(expected_phase=wrong, expected_completed_phase_sha256=anchors)
            assert execute.call_count == 0 and _snapshot(job._seal.path) == before
            if anchors:
                with (
                    patch.object(steps, "execute") as execute,
                    pytest.raises(IrisScientificAblationWorkflowError),
                ):
                    job.advance_expected(
                        expected_phase=phase, expected_completed_phase_sha256=anchors[:-1]
                    )
                assert execute.call_count == 0 and _snapshot(job._seal.path) == before
            budget = 2 if phase.kind in ("capture", "score") else 1
            for chunk in range(budget):
                forward_guard = (
                    patch.object(
                        Qwen3ForCausalLM,
                        "forward",
                        side_effect=AssertionError("Score/publication is DATA only"),
                    )
                    if phase.kind in ("score", "publish")
                    else nullcontext()
                )
                with forward_guard:
                    progress = job.advance_expected(
                        expected_phase=phase,
                        expected_completed_phase_sha256=anchors,
                        rows_per_call=1,
                    )
                records.append(
                    dict(
                        phase=phase.model_dump(mode="json"),
                        before_anchors=anchors,
                        chunk=chunk,
                        actual_progress=asdict(progress),
                    )
                )
                if chunk < budget - 1:
                    assert (
                        progress.next_phase == phase.name
                        and progress.completed_phase_sha256 == anchors
                    )
                else:
                    assert len(progress.completed_phase_sha256) == len(anchors) + 1
            anchors = progress.completed_phase_sha256
        assert candidate.call_count == 0 and progress.complete and progress.next_phase is None
    final_tree = _tree(job._seal.path)
    with (
        patch.object(steps, "execute") as execute,
        pytest.raises(IrisScientificAblationWorkflowError),
    ):
        job.advance_expected(expected_phase=phases[-1], expected_completed_phase_sha256=anchors)
    assert execute.call_count == 0 and _tree(job._seal.path) == final_tree
    assert len(anchors) == 10 and len(records) == 16
    (tmp_path / "actual-geometry-complete.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                cycles=geometry.cycles,
                routing=geometry.routing,
                goal=geometry.goal,
                broadcast=geometry.broadcast,
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=anchors,
                selected_step=1,
                terminal_step=3,
                real_original_native_workflow=True,
                source_scope="tiny_paired_geometry_augmentation_not_scientific5000",
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
    (tmp_path / "actual-expected-workflow-phase-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-expected-workflow-phase-native-witness-v1",
                plan_sha256=case.plan.sha256,
                requests_and_actual_chunks=records,
                completed_phase_sha256=anchors,
                wrong_phase_and_stale_prefix_denied_before_execute=True,
                reopened_with_independent_completed_phase_anchors=True,
                actual_os_cold_process=False,
                durable_cli_command_idempotence=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
