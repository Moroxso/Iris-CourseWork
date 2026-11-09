"""LABELLED controller/reader doubles + REAL storage/locks, NOT fresh-process proof.

The actual native03 replay failure motivates these all-four-protocol regressions.
Physical prefix semantics and real fresh-process/data-only behavior still require
the unchanged whole native matrix and full independent physical/math readback.
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow_command as public
from iris.application import scientific_ablation_workflow_recovery as recovery
from tests.unit.test_scientific_ablation_kernel_recovery import PROTOCOLS
from tests.unit.test_scientific_ablation_workflow_recovery import labelled_recovery


def metadata_pins(path: Path) -> dict[str, tuple[int, str]]:
    return {
        file.relative_to(path).as_posix(): (
            file.stat().st_size,
            sha256(file.read_bytes()).hexdigest(),
        )
        for file in path.rglob("*")
        if file.is_file()
    }


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_retained_old_stop_keeps_advanced_independent_phase_prefix(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, total=4) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        old_prefix = item.job._seal.anchors
        assert reply.result.payload.kernel_progress.acknowledged_rows == 3
        assert reply.operator_ack.payload.completed_phase_sha256 == old_prefix
        item.state["count"] = 4
        advanced = (*old_prefix, "9" * 64)
        # LABELLED independent prefix: real physical prefix/source validation is
        # deliberately doubled by labelled_recovery and is NOT proved here.
        item.job._seal = replace(item.job._seal, anchors=advanced)
        pins = metadata_pins(item.store.path)
        before_calls = item.producer.call_count
        with patch.object(steps, "execute_produced", side_effect=AssertionError("No native retry")):
            repeated = item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
        assert repeated == reply
        assert item.job._seal.anchors == advanced and item.job._state == "idle"
        assert item.facade.progress.completed_phase_sha256 == advanced
        assert repeated.result.payload.kernel_progress.acknowledged_rows == 3
        assert repeated.result.payload.kernel_progress.kernel_complete_sha256 is None
        assert repeated.result.payload.phase_record is None
        assert repeated.operator_ack.payload.completed_phase_sha256 == old_prefix
        assert item.producer.call_count == before_calls
        assert metadata_pins(item.store.path) == pins


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_retained_old_stop_does_not_adopt_unacknowledged_physical_phase(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, total=4) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        old_prefix = item.job._seal.anchors
        item.state["count"] = 4
        pins = metadata_pins(item.store.path)
        with patch.object(steps, "execute_produced", side_effect=AssertionError("No native retry")):
            repeated = item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
        assert repeated == reply and item.job._seal.anchors == old_prefix
        assert metadata_pins(item.store.path) == pins


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_advanced_prefix_with_incomplete_kernel_is_refused_without_new_work(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, total=4) as item:
        reply = item.facade.recover(item.command, operation=item.operation)
        advanced = (*item.job._seal.anchors, "9" * 64)
        item.job._seal = replace(item.job._seal, anchors=advanced)
        pins = metadata_pins(item.store.path)
        with (
            patch.object(steps, "execute_produced", side_effect=AssertionError("No native retry")),
            pytest.raises(public.IrisScientificAblationWorkflowCommandError),
        ):
            item.facade.recover(
                item.command,
                operation=item.operation,
                expected_operator_ack_sha256=reply.operator_ack.sha256,
            )
        assert item.job._seal.anchors == advanced and item.job._state == "idle"
        assert metadata_pins(item.store.path) == pins


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_missing_result_cannot_be_inferred_from_advanced_independent_prefix(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, total=4) as item:
        advanced = (*item.job._seal.anchors, "9" * 64)
        item.job._seal = replace(item.job._seal, anchors=advanced)
        item.state["count"] = 4
        with (
            patch.object(steps, "execute_produced", side_effect=AssertionError("No native retry")),
            pytest.raises(public.IrisScientificAblationWorkflowCommandError),
        ):
            item.facade.recover(item.command, operation=item.operation)
        assert not list(item.store.path.glob("recoveries/*/*/result.json"))
        assert not list(item.store.path.glob("recoveries/*/*/operator-ack.json"))
        assert item.job._seal.anchors == advanced and item.job._state == "idle"


@pytest.mark.parametrize("kind,controlled", PROTOCOLS)
def test_same_old_ack_is_still_required_after_phase_advance(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    with labelled_recovery(tmp_path, kind, controlled, total=4) as item:
        item.facade.recover(item.command, operation=item.operation)
        advanced = (*item.job._seal.anchors, "9" * 64)
        item.job._seal = replace(item.job._seal, anchors=advanced)
        item.state["count"] = 4
        pins = metadata_pins(item.store.path)
        with (
            patch.object(steps, "execute_produced", side_effect=AssertionError("No native retry")),
            pytest.raises(public.IrisScientificAblationWorkflowCommandError),
        ):
            item.facade.recover(
                item.command, operation=item.operation, expected_operator_ack_sha256="f" * 64
            )
        assert item.job._seal.anchors == advanced and item.job._state == "idle"
        assert metadata_pins(item.store.path) == pins


def test_actual_retained_recovery_coordinator_is_used() -> None:
    assert vars(public)["_recover_kernel_command_v1"] is recovery._recover_kernel_command_v1
