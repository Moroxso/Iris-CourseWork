"""Real hard-death after durable row commit, cold source-only process recovery.

Authored tiny CPU consumers get disk sources/data and independently supplied job
SHA, never expected outputs. This is not installed package/public CLI evidence.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

import iris
from iris.application.scientific_ablation_controlled_capture import (
    read_scientific_ablation_controlled_capture_v1,
)
from iris.application.scientific_command import ScientificParquetInputV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_controlled_capture import _capture, _setup
from tests.integration.test_scientific_ablation_controlled_request import (
    control as workspace_control,
)
from tests.integration.test_scientific_ablation_generation_source import _prepared, _read

pytestmark = pytest.mark.integration
_WORKER = Path(__file__).with_name("_scientific_ablation_controlled_capture_worker.py")


def _child(root: Path, job: dict[str, Any], name: str) -> subprocess.CompletedProcess[str]:
    assert iris.__file__ is not None
    package_root = Path(iris.__file__).resolve().parent.parent
    raw = canonical_json_file_bytes(job)
    path = root / f"{name}.json"
    path.write_bytes(raw)
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
            str(path),
            hashlib.sha256(raw).hexdigest(),
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
    return result


@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
@pytest.mark.parametrize("sampling", (False, True), ids=("greedy", "sampling"))
def test_actual_hard_exit_releases_process_lock_cold_consumer_skips_ack_and_matches_uninterrupted(
    tmp_path: Path,
    sampling: bool,
    kind: Literal["empty", "zero", "shuffle"],
) -> None:
    from tests.integration.test_scientific_generation import _test_rows

    prepared = _prepared(tmp_path / "source")
    source = _read(prepared)
    heldout, plan = _setup(
        source,
        tmp_path / "test.parquet",
        rows=_test_rows()[-2:],
        sampling=sampling,
        control=workspace_control(kind),
    )
    spec = ScientificParquetInputV1(
        path="test.parquet",
        sha256=plan.payload.input.payload.test_parquet_sha256,
        maximum_rows=2,
        maximum_file_bytes=1024**2,
        maximum_logical_bytes=1024**2,
    )
    root = tmp_path / "resumed"
    job: dict[str, Any] = {
        "action": "die_after_row_commit",
        "lifecycle_path": str(prepared.native_path),
        "inference_path": str(prepared.inference_path),
        "lifecycle_sha256": source.manifest.payload.lifecycle_sha256,
        "inference_sha256": source.manifest.payload.inference_sha256,
        "checkpoint_sha256": source.manifest.payload.best_checkpoint_sha256,
        "source_sha256": source.manifest.sha256,
        "input_sha256": plan.payload.input.sha256,
        "plan_sha256": plan.sha256,
        "test": spec.model_dump(mode="json"),
        "test_base": str(tmp_path),
        "root": str(root),
        "routing": "learned",
        "goal": "enabled",
        "broadcast": "enabled",
        "decoding": plan.payload.input.payload.decoding.model_dump(mode="json"),
        "control": json.loads(plan.payload.input.payload.generation_control_plan_json)["control"],
    }
    crashed = _child(tmp_path, job, "hard-exit")
    assert crashed.returncode == 87, crashed.stderr
    first = json.loads(crashed.stdout)
    assert first["kind"] == "iris-controlled-capture-worker-hard-exit-v1"
    assert first["executed_indices"] == [0] and first["committed_row"] == "row-00000"
    assert first["plan_sha256"] == plan.sha256 and first["pid"] != os.getpid()
    path = root / "scientific-ablation-controlled-capture-v1" / plan.sha256
    partial = read_scientific_ablation_controlled_capture_v1(path, expected_plan_sha256=plan.sha256)
    assert partial.next_row_index == 1 and partial.complete is None
    job["action"] = "resume"
    resumed = _child(tmp_path, job, "cold-resume")
    assert resumed.returncode == 0, resumed.stderr
    observed = json.loads(resumed.stdout)
    assert observed["kind"] == "iris-controlled-capture-worker-complete-v1"
    assert observed["executed_indices"] == [1] and observed["pid"] != first["pid"]
    assert observed["caller_restored"] and not observed["source_regenerated"]
    current = read_scientific_ablation_controlled_capture_v1(
        path, expected_plan_sha256=plan.sha256, expected_capture_sha256=observed["capture_sha256"]
    )
    assert current.rows[0] == partial.rows[0]
    uninterrupted = _capture(heldout, tmp_path / "uninterrupted", plan)
    assert uninterrupted.complete is not None and current.complete == uninterrupted.complete
    assert (
        current.rows == uninterrupted.rows
        and current.acknowledgements == uninterrupted.acknowledgements
    )
    assert observed["row_sha256"] == [row.sha256 for row in uninterrupted.rows]
    assert observed["ack_sha256"] == [ack.sha256 for ack in uninterrupted.acknowledgements]
    assert observed["target_token_count"] == uninterrupted.complete.payload.target_token_count
    assert observed["target_nll_sum"] == uninterrupted.complete.payload.target_nll_sum
