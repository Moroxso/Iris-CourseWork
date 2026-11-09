"""Labelled reservation-branch DATA tests; real C1/C5 native proof is separate."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as module
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from tests.unit._workflow_data_setup import plan


def _data_only_source(cycles: object) -> SimpleNamespace:
    value = SimpleNamespace(cycles=cycles)
    for name in ("workspace", "model", "payload", "manifest", "inference"):
        value = SimpleNamespace(**{name: value})
    return value


@pytest.mark.parametrize("kind", ("workspace", "message"))
@pytest.mark.parametrize("cycles", (1, 3, 5))
def test_original_authenticated_cycle_controls_only_byte_reservation(
    tmp_path: Path,
    kind: str,
    cycles: int,
) -> None:
    intent = plan(tmp_path)
    phase = next(p for p in scientific_ablation_workflow_phases_v1(intent) if p.kind == kind)
    with patch.object(module, "source", return_value=_data_only_source(cycles)) as read:
        reservation, no_next_cycle = module._native_reservation(intent, phase)
    read.assert_called_once_with(intent)
    assert no_next_cycle is (cycles == 1)
    assert reservation == (
        4 * 1024**2
        if cycles == 1
        else MAX_REPORT_BYTES + intent.payload.resources.maximum_writer_report_bytes + 4 * 1024**2
    )


@pytest.mark.parametrize("cycles", (True, False, 0, 65, 1.0, "1", None))
def test_malformed_source_cycle_cannot_select_c1_reservation(
    tmp_path: Path, cycles: object
) -> None:
    intent = plan(tmp_path)
    phase = next(p for p in scientific_ablation_workflow_phases_v1(intent) if p.kind == "workspace")
    with (
        patch.object(module, "source", return_value=_data_only_source(cycles)),
        pytest.raises(ValueError),
    ):
        module._native_reservation(intent, phase)


def test_diagnostics_keep_both_full_report_bounds_even_for_c1(tmp_path: Path) -> None:
    intent = plan(tmp_path)
    phase = next(
        p for p in scientific_ablation_workflow_phases_v1(intent) if p.kind == "diagnostics"
    )
    with patch.object(module, "source", side_effect=AssertionError("No fake native permission")):
        assert module._native_reservation(intent, phase) == (132 * 1024**2, False)


def test_capture_is_not_a_native_diagnostic_reservation(tmp_path: Path) -> None:
    intent = plan(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(intent)[0]
    with (
        patch.object(module, "source", side_effect=AssertionError("No native source read")),
        pytest.raises(ValueError),
    ):
        module._native_reservation(intent, phase)
