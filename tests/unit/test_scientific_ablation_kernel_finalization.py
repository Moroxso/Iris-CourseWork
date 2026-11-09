"""LABELLED strict JSON + REAL storage/locks; mocked source binding, NOT native proof."""

import json
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_kernel_finalization as finalization
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow_io as storage
from iris.application import scientific_ablation_workflow_kernel as kernels
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.process_file_lock import process_file_lock
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_handoff import labelled_job

FIXTURE = Path(__file__).parent / "fixtures/aggregate_serialization_original"
NAMES = ("ordinary-capture", "ordinary-score", "control-00-capture", "control-00-score")


def fixture_root(index: int) -> Path:
    values = [path for path in (FIXTURE / NAMES[index]).glob("*/*") if path.is_dir()]
    assert len(values) == 1
    return values[0]


def read(index: int, path: Path) -> finalization.Progress:
    plan = json.loads((path / "plan.json").read_bytes())
    if index == 0:
        return capture.read_scientific_ablation_capture_v1(
            path, expected_plan_sha256=plan["sha256"]
        )
    if index == 2:
        return controlled_capture.read_scientific_ablation_controlled_capture_v1(
            path,
            expected_plan_sha256=plan["sha256"],
        )
    args = dict(
        expected_evaluation_plan_sha256=plan["sha256"],
        expected_plan_sha256=plan["payload"]["generation_plan"]["sha256"],
        expected_capture_sha256=plan["payload"]["capture"]["sha256"],
    )
    if index == 1:
        return score.read_scientific_ablation_evaluation_progress_v2(path, **args)
    return controlled_score.read_scientific_ablation_controlled_evaluation_progress_v1(path, **args)


@pytest.mark.parametrize("index", range(4))
def test_exact_original_aggregate_bytes_from_strict_saved_json_only(index: int) -> None:
    path = fixture_root(index)
    result = read(index, path)
    expected = {file.name: file.read_bytes() for file in (path / "complete").iterdir()}
    assert finalization._aggregate_files(result) == expected


@dataclass(frozen=True)
class Labelled:
    store: storage.ScientificAblationWorkflowStoreV1
    command: IrisScientificAblationWorkflowCommandIntentV1
    prefix: tuple[storage.ScientificAblationWorkflowDataV1, ...]
    path: Path
    namespace: Path
    expected: dict[str, bytes]
    head: str


@contextmanager
def labelled(tmp_path: Path, index: int, *, missing_last: bool = False) -> Iterator[Labelled]:
    # The underlying strict JSON reader, disk and both OS locks are REAL.
    # Authentic operator/source binding is mocked and is NOT proved by this test.
    _, _, _, source, _ = labelled_job(tmp_path)
    plan = storage.build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    definition = operator._command_definition(plan, _request(), source)
    store = storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "aggregate-output",
        plan,
        definition,
    )
    phases = scientific_ablation_workflow_phases_v1(plan)
    phase = phases[index]
    source_path = fixture_root(index)
    namespace = steps._kernel(store, phase) / source_path.parent.name
    target = namespace / source_path.name
    prefix: list[storage.ScientificAblationWorkflowDataV1] = []
    for previous in phases[:index]:
        name = "capture.json" if previous.kind == "capture" else "evaluation.json"
        prefix.append(
            store.make_data(
                previous,
                prefix[-1].record.sha256 if prefix else None,
                origin="kernel_data",
                files={name: b'"LABELLED DATA"\n'},
            )
        )
    with ExitStack() as stack:
        with store.claim():
            shutil.copytree(
                source_path,
                target,
                ignore=shutil.ignore_patterns(
                    "complete",
                    *(("row-00003" if index in (0, 2) else "score-00003",) if missing_last else ()),
                ),
            )
        original = read(index, source_path)
        head = original.acknowledgements[0].payload.previous_ack_sha256
        chunk = Chunk(
            kernel_plan_sha256=original.plan.sha256,
            acknowledgement_head_sha256=head,
            acknowledged_rows=0,
            total_rows=4,
            rows_per_call=4,
            stop_row_index=4,
        )
        command = build(
            Intent(
                namespace=definition,
                phase=phase,
                completed_phase_sha256=tuple(v.record.sha256 for v in prefix),
                expected_chunk=chunk,
                rows_per_call=4,
            )
        )
        for function in ("_capture_result", "_score_result"):
            stack.enter_context(
                patch.object(kernels, function, side_effect=lambda *a, **k: read(index, target))
            )
        stack.enter_context(patch.object(steps, "_owner", side_effect=AssertionError("No factory")))
        stack.enter_context(
            patch.object(steps, "execute_produced", side_effect=AssertionError("No producer"))
        )
        yield Labelled(
            store,
            command,
            tuple(prefix),
            target,
            namespace,
            {f.name: f.read_bytes() for f in (source_path / "complete").iterdir()},
            original.acknowledgements[-1].sha256,
        )


@pytest.mark.parametrize("index", range(4))
def test_real_original_stages_publish_once_preserve_rows_and_replay_data_only(
    tmp_path: Path, index: int
) -> None:
    with labelled(tmp_path, index) as item:
        before = finalization._row_stamps(read(index, item.path))
        with (
            item.store.claim() as claim,
            process_file_lock(item.namespace / ".iris-owner.lock") as lock,
        ):
            result = finalization._finalize_acknowledged_kernel_data(
                item.store,
                item.command,
                item.prefix,
                claim,
                lock,
                expected_acknowledgement_head_sha256=item.head,
            )
            assert result.complete is not None
            assert {
                f.name: f.read_bytes() for f in (item.path / "complete").iterdir()
            } == item.expected
            assert finalization._row_stamps(result) == before
            with patch.object(
                finalization, "_publish", side_effect=AssertionError("No second publication")
            ):
                assert (
                    finalization._finalize_acknowledged_kernel_data(
                        item.store,
                        item.command,
                        item.prefix,
                        claim,
                        lock,
                        expected_acknowledgement_head_sha256=item.head,
                    )
                    == result
                )


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize(
    "fault",
    (
        "missing_last",
        "old_stop",
        "wrong_head",
        "locked_head",
        "held_quota",
        "free_floor",
        "foreign_lock",
    ),
)
def test_closed_interval_chain_claim_and_quota_fences(
    tmp_path: Path, index: int, fault: str
) -> None:
    with labelled(tmp_path, index, missing_last=fault == "missing_last") as item:
        command = item.command
        p = command.payload
        assert p.expected_chunk is not None
        if fault in {"old_stop", "wrong_head"}:
            chunk = p.expected_chunk.model_copy(
                update=(
                    dict(rows_per_call=3, stop_row_index=3)
                    if fault == "old_stop"
                    else dict(acknowledgement_head_sha256="f" * 64)
                )
            )
            command = build(
                Intent(
                    namespace=p.namespace,
                    phase=p.phase,
                    completed_phase_sha256=p.completed_phase_sha256,
                    expected_chunk=chunk,
                    rows_per_call=chunk.rows_per_call,
                )
            )
        lock_name = "foreign.lock" if fault == "foreign_lock" else ".iris-owner.lock"
        with item.store.claim() as original, process_file_lock(item.namespace / lock_name) as lock:
            claim = (
                replace(original, reserved_bytes=15 * 1024**2)
                if fault == "held_quota"
                else original
            )
            with ExitStack() as stack:
                if fault == "free_floor":
                    stack.enter_context(
                        patch.object(
                            shutil,
                            "disk_usage",
                            return_value=SimpleNamespace(free=0),
                        )
                    )
                with pytest.raises((ValueError, OSError)):
                    finalization._finalize_acknowledged_kernel_data(
                        item.store,
                        command,
                        item.prefix,
                        claim,
                        lock,
                        expected_acknowledgement_head_sha256="f" * 64
                        if fault == "locked_head"
                        else item.head,
                    )
        assert not (item.path / "complete").exists()
