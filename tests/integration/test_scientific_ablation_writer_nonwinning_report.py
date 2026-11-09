"""Fixed input-diverse paired donor source, not outcome-selected or fabricated data."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.training.model import IrisTeacherForcingModel
from tests.integration import scientific_ablation_writer_geometry_fixture as fixture
from tests.integration.scientific_ablation_writer_geometry_fixture import Geometry
from tests.integration.test_scientific_ablation_writer_geometry_reports import _measure

pytestmark = pytest.mark.integration


def test_actual_fixed_input_diverse_paired_proposal_donor_need_not_win(tmp_path: Path) -> None:
    # These source seeds/first-two-per-stratum rules are fixed BEFORE any model,
    # bidding, allocation, effect or generated answer is observed. No retry/reroll.
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=3000),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row
        for task in TASKS
        for difficulty in (1, 2, 3)
        for row in tuple(r for r in pool if r.task_type == task and r.difficulty == difficulty)[:2]
    )
    assert len(rows) == 24 and len({row.id for row in rows}) == 24

    def symmetric_initial_teacher(*args: Any, **kwargs: Any) -> IrisTeacherForcingModel:
        # INITIALIZE only a newly constructed native graph, BEFORE scientific
        # owner/optimizer/run/initial-state checksum and ANY train/capture/forward.
        # No bid/winner/target/effect is queried. Cells retain distinct parameters
        # and storage; equal initial values remove fixed module identity advantage.
        teacher = build_teacher_forcing_model(*args, **kwargs)
        cells = teacher.cognitive_modules.module_cells
        with torch.no_grad():
            dict(cells[0].named_parameters())["identity_embedding"].zero_()
            initial = {
                name: value.detach().clone() for name, value in cells[0].state_dict().items()
            }
            for cell in cells[1:]:
                cell.load_state_dict(initial, strict=True)
        parameters = [parameter for cell in cells for parameter in cell.parameters()]
        assert len({id(parameter) for parameter in parameters}) == len(parameters)
        assert len({parameter.untyped_storage().data_ptr() for parameter in parameters}) == len(
            parameters
        )
        return teacher

    # Every ORIGINAL construction and fresh restore factory uses this same fixed
    # analytic initialization. Real independent cells train normally afterward.
    with patch.object(fixture, "build_teacher_forcing_model", new=symmetric_initial_teacher):
        result = _measure(tmp_path, Geometry(), "random", "disabled", "enabled", rows)
    assert result["message"]["actual_teacher_executions"] == 144
    assert result["workspace"]["actual_teacher_executions"] == 72
    assert result["message"]["nonwinning_donor_frames"] > 0
    assert result["message"]["occupied_block_frames"] > 0
