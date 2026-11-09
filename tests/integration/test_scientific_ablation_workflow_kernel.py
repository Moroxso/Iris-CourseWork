"""Actual native C1/C5 partial/complete row-prefix inspection; no CLI ACK claim."""

import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture_kernel
from iris.application import scientific_ablation_controlled_capture as controlled_capture_kernel
from iris.application import (
    scientific_ablation_controlled_evaluation_ledger as controlled_score_kernel,
)
from iris.application import scientific_ablation_evaluation_ledger as score_kernel
from iris.application.scientific_ablation_workflow import create_scientific_ablation_workflow_v1
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, Geometry, prepare
from tests.integration.test_scientific_ablation_execution import _tree

pytestmark = pytest.mark.integration
_LOCKS = {
    capture_kernel._STORE: capture_kernel._LOCK,
    controlled_capture_kernel._STORE: controlled_capture_kernel._LOCK,
    score_kernel._STORE: score_kernel._LOCK,
    controlled_score_kernel._STORE: controlled_score_kernel._LOCK,
}


@pytest.mark.parametrize("geometry", (GEOMETRIES[0], GEOMETRIES[2]), ids=("c1", "c5"))
def test_actual_kernel_prefixes_are_read_only_through_all_ten_phases(
    tmp_path: Path, geometry: Geometry
) -> None:
    case = prepare(tmp_path / "case", geometry)
    job = create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow", plan=case.plan, make_fresh_run=case.factory
    )
    phases = scientific_ablation_workflow_phases_v1(case.plan)
    records = []
    with patch(
        "iris.application.synthetic_scoring_v2.execute_python_job_v2",
        side_effect=AssertionError("No generated Python on host"),
    ) as candidate:
        for phase in phases:
            budget = 2 if phase.kind in ("capture", "score") else 1
            for chunk in range(budget):
                progress = job.advance_expected(
                    expected_phase=phase,
                    expected_completed_phase_sha256=job.progress.completed_phase_sha256,
                    rows_per_call=1,
                )
                if phase.kind not in ("capture", "score"):
                    continue
                namespace = (
                    (
                        "scientific-ablation-capture-v1"
                        if phase.stream_index is None
                        else "scientific-ablation-controlled-capture-v1"
                    )
                    if phase.kind == "capture"
                    else (
                        "scientific-ablation-evaluation-ledger-v2"
                        if phase.stream_index is None
                        else "scientific-ablation-controlled-evaluation-ledger-v1"
                    )
                )
                parent = job._seal.path / "kernels" / phase.name / namespace
                entries = tuple(parent.iterdir())
                folders = tuple(p for p in entries if re.fullmatch(r"[0-9a-f]{64}", p.name))
                assert len(folders) == 1
                expected = folders[0].name
                assert {p.name for p in entries} == {expected, _LOCKS[namespace]}
                for node, directory in ((folders[0], True), (parent / _LOCKS[namespace], False)):
                    metadata = node.lstat()
                    assert not node.is_symlink()
                    assert not getattr(metadata, "st_file_attributes", 0) & 0x400
                    if directory:
                        assert stat.S_ISDIR(metadata.st_mode)
                    else:
                        assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
                before = _tree(job._seal.path)
                with (
                    patch.object(
                        steps, "_owner", side_effect=AssertionError("No factory")
                    ) as factory,
                    patch(
                        "torch.nn.Module.__init__", side_effect=AssertionError("No graph")
                    ) as graph,
                    patch.object(
                        Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
                    ) as forward,
                ):
                    snapshot = job.inspect_kernel(
                        expected_phase=phase, expected_kernel_plan_sha256=expected
                    )
                    repeat = job.inspect_kernel(
                        expected_phase=phase,
                        expected_kernel_plan_sha256=expected,
                        expected_acknowledgement_head_sha256=snapshot.acknowledgement_head_sha256,
                    )
                    with pytest.raises(ValueError):
                        job.inspect_kernel(
                            expected_phase=phase,
                            expected_kernel_plan_sha256=expected,
                            expected_acknowledgement_head_sha256="0" * 64,
                        )
                assert factory.call_count == graph.call_count == forward.call_count == 0
                assert snapshot == repeat and _tree(job._seal.path) == before
                assert snapshot.acknowledged_rows == chunk + 1 and snapshot.total_rows == 2
                assert snapshot.phase_is_acknowledged is (chunk == 1)
                assert (snapshot.kernel_complete_sha256 is not None) is (chunk == 1)
                assert snapshot.completed_phase_sha256 == progress.completed_phase_sha256
                assert snapshot.new_native_execution_authorized is False
                records.append(snapshot.model_dump(mode="json"))
        assert candidate.call_count == 0
    assert progress.complete and len(progress.completed_phase_sha256) == 10 and len(records) == 12
    (tmp_path / "actual-workflow-kernel-inspection-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-workflow-kernel-inspection-native-witness-v1",
                cycles=geometry.cycles,
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                inspections=records,
                actual_inspection_factory_graph_forward_calls_zero=True,
                all_inspections_preserved_workflow_bytes=True,
                actual_os_cold_process=False,
                durable_cli_command_ack=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
