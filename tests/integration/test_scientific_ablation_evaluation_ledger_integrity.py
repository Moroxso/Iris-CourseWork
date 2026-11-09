"""Owned publication identity and trusted resume source/anchor edge cases."""

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_evaluation_ledger as ledger
from iris.application import scientific_ablation_evaluation_v2 as scorer
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationPlanV2,
)
from iris.integrations.python_sandbox import _worker
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_evaluation_ledger import (
    Error,
    _plan,
    _read,
    _run,
    _snapshot,
)
from tests.integration.test_scientific_ablation_evaluation_v2 import Case, _copy, _make

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return _make(tmp_path_factory.mktemp("actual-score-ledger-integrity-source"))


@pytest.fixture(scope="module")
def plan(case: Case) -> IrisScientificAblationEvaluationPlanV2:
    return _plan(case)


@pytest.mark.parametrize("phase", ("plan", "row", "complete"))
@pytest.mark.parametrize("member", ("directory", "file"))
def test_equal_byte_replacement_after_actual_rename_is_not_our_commit(
    case: Case,
    plan: IrisScientificAblationEvaluationPlanV2,
    tmp_path: Path,
    phase: str,
    member: str,
) -> None:
    replaced = False
    phase_target = {"plan": plan.sha256, "row": "score-00000", "complete": "complete"}[phase]

    def swap(source: Path, target: Path) -> None:
        nonlocal replaced
        rename_directory_with_windows_retry(source, target)
        if target.name == phase_target:
            if member == "directory":
                displaced = tmp_path / "displaced-owned-directory"
                target.rename(displaced)
                shutil.copytree(displaced, target)
                assert target.stat().st_ino != displaced.stat().st_ino
            else:
                file = (
                    target
                    / {"plan": "plan.json", "row": "row.json", "complete": "evaluation.json"}[phase]
                )
                content = file.read_bytes()
                displaced = tmp_path / "displaced-owned-file"
                file.rename(displaced)
                file.write_bytes(content)
                assert file.stat().st_ino != displaced.stat().st_ino
            replaced = True

    with (
        patch.object(ledger, "rename_directory_with_windows_retry", side_effect=swap),
        pytest.raises(Error),
    ):
        _run(case, plan, tmp_path)
    assert replaced


@pytest.mark.parametrize(
    "field",
    (
        "expected_evaluation_plan_sha256",
        "expected_plan_sha256",
        "expected_capture_sha256",
        "expected_acknowledgement_head_sha256",
    ),
)
def test_reader_requires_independent_plan_capture_and_optional_head(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, field: str
) -> None:
    partial = _run(case, plan, tmp_path, rows_per_call=1)
    before = _snapshot(tmp_path)
    with pytest.raises(Error):
        _read(plan, partial.path, **{field: "0" * 64})
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ("worker", "policy", "source", "receipt"))
def test_old_cache_never_bypasses_current_source_policy_or_worker_verification(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    owned = _copy(case, tmp_path / "source")
    root = tmp_path / "scores"
    partial = _run(owned, plan, root, rows_per_call=1)
    before = _snapshot(root)
    options: dict[str, Any] = {}
    if kind == "policy":
        assert plan.payload.python_policy is not None
        options["python_policy"] = plan.payload.python_policy.model_copy(
            update={"max_output_bytes": 1024}
        )
    elif kind == "source":
        file = owned.source.native_path / "best.safetensors"
        file.write_bytes(file.read_bytes() + b"late source drift")
    elif kind == "receipt":
        file = partial.path / "score-00000" / "row.json"
        value = json.loads(file.read_bytes())
        value["payload"]["measurements"]["execution"] = {"foreign": True}
        file.write_bytes(canonical_json_file_bytes(value))
        before = _snapshot(root)
    with (
        patch.object(
            scorer,
            "_worker",
            return_value=_worker("python_worker_v2.txt")
            + (b"changed" if kind == "worker" else b""),
        ),
        patch.object(ledger, "_score_row") as score,
        pytest.raises(Error),
    ):
        _run(owned, plan, root, **options)
    score.assert_not_called()
    assert _snapshot(root) == before
