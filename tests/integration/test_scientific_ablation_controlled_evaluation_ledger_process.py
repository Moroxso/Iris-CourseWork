"""Actual hard death/cold resume, including separately gated genuine gVisor jobs."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

import iris
from iris.application import scientific_ablation_controlled_evaluation_ledger as ledger
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._sandbox_startup_trace import assert_safe_sandbox_startup_trace
from tests.integration.test_scientific_ablation_controlled_evaluation import (
    Case,
    _controlled,
    _evaluate,
    _make,
)
from tests.integration.test_scientific_ablation_controlled_evaluation_ledger import (
    _arguments,
    _plan,
    _read,
)
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
_WORKER = Path(__file__).with_name("_scientific_ablation_controlled_evaluation_worker.py")


def _child(root: Path, job: dict[str, Any], name: str) -> subprocess.CompletedProcess[str]:
    assert iris.__file__ is not None
    package_root = Path(iris.__file__).resolve().parent.parent
    content = canonical_json_file_bytes(job)
    job_path = root / f"{name}.json"
    job_path.write_bytes(content)
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(_WORKER),
            str(package_root),
            str(job_path),
            hashlib.sha256(content).hexdigest(),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert len(result.stdout.encode()) <= 128 * 1024 and len(result.stderr.encode()) <= 128 * 1024
    (root / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode in {0, 87}:
        packet = json.loads(result.stdout)
        assert packet["job_sha256"] == hashlib.sha256(content).hexdigest()
        assert packet["worker_sha256"] == hashlib.sha256(_WORKER.read_bytes()).hexdigest()
        assert packet["package_root"] == str(package_root)
        assert packet["actual_iris_origins"]
        for relative in packet["actual_iris_origins"].values():
            assert (package_root / relative).resolve().is_relative_to(package_root)
    return result


def _proof(case: Case, root: Path, *, isolated: bool, variant: str = "reference") -> None:
    plan = _plan(case)
    job: dict[str, Any] = {
        **{
            key: str(value)
            if isinstance(value, Path)
            else value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
            for key, value in _arguments(case).items()
        },
        "root": str(root / "scores"),
        "expected_evaluation_plan_sha256": plan.sha256,
        "action": "die_after_row_commit",
    }
    crashed = _child(root, job, "hard-exit")
    assert crashed.returncode == 87, crashed.stderr
    first = json.loads(crashed.stdout)
    assert first["kind"] == "iris-controlled-score-worker-hard-exit-v1"
    assert first["pid"] != os.getpid() and first["scored_indices"] == [0]
    assert first["sandbox_indices"] == ([0] if isolated else [])
    assert first["evaluation_plan_sha256"] == plan.sha256
    assert first["committed_row"] == "score-00000"
    if isolated:
        assert_safe_sandbox_startup_trace(first["linux_phases"])
    else:
        assert first["linux_phases"] == []
    path = root / "scores" / ledger._STORE / plan.sha256
    partial = _read(plan, path)
    assert partial.next_row_index == 1 and partial.complete is None and partial.evaluation is None
    score_bytes = (path / "score-00000" / "row.json").read_bytes()
    score_inode = (path / "score-00000" / "row.json").stat().st_ino
    job["action"] = "resume"
    resumed = _child(root, job, "cold-resume")
    assert resumed.returncode == 0, resumed.stderr
    observed = json.loads(resumed.stdout)
    assert observed["kind"] == "iris-controlled-score-worker-complete-v1"
    assert observed["pid"] not in {first["pid"], os.getpid()}
    assert observed["scored_indices"] == list(range(1, len(case.rows)))
    assert observed["sandbox_indices"] == [] and observed["linux_phases"] == []
    assert observed["caller_restored"] and not observed["model_constructed"]
    assert not observed["host_candidate_execution"]
    current = _read(plan, path)
    assert (
        current.rows[0] == partial.rows[0]
        and current.acknowledgements[0] == partial.acknowledgements[0]
    )
    assert (path / "score-00000" / "row.json").read_bytes() == score_bytes
    assert (path / "score-00000" / "row.json").stat().st_ino == score_inode
    assert current.evaluation is not None and current.complete is not None
    assert observed["evaluation_sha256"] == current.evaluation.sha256
    assert observed["complete_sha256"] == current.complete.sha256
    assert observed["row_sha256"] == [row.sha256 for row in current.rows]
    assert observed["ack_sha256"] == [ack.sha256 for ack in current.acknowledgements]
    if isolated:
        repair = current.rows[0].payload.measurements
        assert repair.execution is not None
        assert (
            repair.outcome
            == {"reference": "correct", "incorrect": "incorrect", "syntax": "invalid_output"}[
                variant
            ]
        )
        assert (
            repair.reason
            == {
                "reference": "all_cases_pass",
                "incorrect": "case_mismatch",
                "syntax": "syntax_error",
            }[variant]
        )
        assert current.evaluation.payload.overall.code_executed == 1
        assert current.evaluation.payload.overall.correct == (4 if variant == "reference" else 3)
    else:
        assert current.evaluation == _evaluate(case)
    retry = _child(root, job, "complete-retry")
    assert retry.returncode == 0, retry.stderr
    repeated = json.loads(retry.stdout)
    assert (
        repeated["scored_indices"] == repeated["sandbox_indices"] == repeated["linux_phases"] == []
    )
    assert repeated["evaluation_sha256"] == observed["evaluation_sha256"]


@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
@pytest.mark.parametrize("sampling", (False, True))
def test_real_hard_exit_os_releases_lock_cold_model_free_resume(
    tmp_path: Path, kind: Literal["empty", "zero", "shuffle"], sampling: bool
) -> None:
    case = _make(tmp_path / "source", control_kind=kind, sampling=sampling)
    _proof(case, tmp_path, isolated=False)


@pytest.fixture(scope="module", params=("empty", "zero", "shuffle"))
def code_case(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Case:
    rows = tuple(sorted(_test_rows(), key=lambda row: not row.executable))
    return _make(
        tmp_path_factory.mktemp("actual-code-first-capture"), rows=rows, control_kind=request.param
    )


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("IRIS_RUN_CODE_ISOLATION_SMOKE") != "1",
    reason="requires explicit real IrisCodeEval/gVisor gate",
)
@pytest.mark.parametrize("variant", ("reference", "incorrect", "syntax"))
def test_real_gvisor_ack_survives_hard_exit_and_is_never_executed_again(
    code_case: Case, tmp_path: Path, variant: str
) -> None:
    controlled = _controlled(code_case, tmp_path / "controlled", variant)
    assert controlled.rows[0].executable
    _proof(controlled, tmp_path, isolated=True, variant=variant)
