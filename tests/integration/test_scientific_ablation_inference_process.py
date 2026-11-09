"""Actual executable-only consumer: all12 modes, trained earlier best and full decode.

Only disk artifact, independent frozen backbone and explicit caller sampling state
cross the child boundary. Expected outputs/teacher/labels stay in this parent.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save as save_tensors

import iris
import iris.application as application
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import bytes_sha256, canonical_json_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration._scientific_ablation_inference_observation import (
    compact_outputs,
    execute_observation,
    workspace_observation,
)
from tests.integration.test_scientific_ablation_inference_projection import Routing, _fresh, _source
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration
_WORKER = Path(__file__).with_name("_scientific_ablation_inference_worker.py")
_MAX_RESULT = 512 * 1024


class _ExpectedHashes(TypedDict):
    expected_sha256: str
    expected_source_checkpoint_sha256: str
    expected_lifecycle_sha256: str


def _generator() -> torch.Generator:
    value = torch.Generator(device="cpu").manual_seed(1234)
    torch.rand(17, generator=value, device="cpu")
    return value


def _package_root() -> Path:
    assert iris.__file__ is not None
    return Path(iris.__file__).resolve().parent.parent


def _producer(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base, tokenizer, lm = _build()
    backbone = base.teacher.backbone
    raw_weights = save_tensors(
        {name: value.detach().contiguous().clone() for name, value in lm.state_dict().items()}
    )
    external_weights = tmp_path / "explicit-frozen-backbone.safetensors"
    external_weights.write_bytes(raw_weights)
    caller_generator = _generator()
    job: dict[str, Any] = {
        "backbone_weights": str(external_weights),
        "backbone_weights_sha256": bytes_sha256(raw_weights),
        "backbone_configuration": backbone.configuration_json,
        "tokenizer_json": tokenizer.backend_tokenizer.to_str(),
        "lm_configuration": lm.config.to_dict(),
        "sampling_seed": caller_generator.initial_seed(),
        "sampling_state_hex": bytes(caller_generator.get_state().tolist()).hex(),
        "cases": [],
    }
    expected: dict[str, Any] = {}
    modes: tuple[Routing, ...] = ("learned", "random", "uniform")
    for routing, goal, broadcast in product(
        modes, ("enabled", "disabled"), ("enabled", "disabled")
    ):
        mode = f"{goal}-{broadcast}-{routing}"
        source, controller = _source(routing, goal=goal, broadcast=broadcast)
        current = build_scientific_ablation_checkpoint_v1(controller._run)
        restored: list[IrisScientificAblationRunV1] = []
        factory = partial(_fresh, routing, goal=goal, broadcast=broadcast)

        def make(
            fresh_factory: Callable[[], IrisScientificAblationRunV1] = factory,
            collected: list[IrisScientificAblationRunV1] = restored,
        ) -> IrisScientificAblationRunV1:
            fresh = fresh_factory()
            collected.append(fresh)
            return fresh

        bundle = application.build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=make
        )
        anchors = (
            bundle.manifest.sha256,
            bundle.manifest.payload.source_checkpoint_sha256,
            source.manifest.sha256,
        )
        bindings = _ExpectedHashes(
            expected_sha256=anchors[0],
            expected_source_checkpoint_sha256=anchors[1],
            expected_lifecycle_sha256=anchors[2],
        )
        model = restored[0]._optimizer._model
        model.eval()
        row = _rows()[0]
        logits: list[torch.Tensor] = []
        native_teacher = model.teacher.backbone.forward_embeddings

        def teacher(
            *args: Any,
            original: Callable[..., BackboneCausalOutput] = native_teacher,
            collected_logits: list[torch.Tensor] = logits,
            **kwargs: Any,
        ) -> BackboneCausalOutput:
            actual = original(*args, **kwargs)
            first_target = int(torch.nonzero(kwargs["labels"][0] != -100)[0, 0])
            collected_logits.append(actual.logits[:, first_target - 1, :].detach().clone())
            return actual

        with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=teacher):
            reference = model.evaluation_forward(row, terminal_token_id=1)
        path = application.write_scientific_ablation_inference_v1(
            bundle, root=tmp_path / "published", **bindings
        )
        read = application.read_scientific_ablation_inference_v1(path, **bindings)
        request = application.build_scientific_ablation_request_v1(
            read,
            backbone=backbone,
            stop_token_ids=(),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
            **bindings,
        )
        caller = capture_runtime_state(torch.device("cpu"))
        generator = _generator()
        outputs = [
            execute_observation(request, backbone, row.prompt, strategy, generator)
            for strategy in ("greedy", "greedy", "sampling", "sampling")
        ]
        for observed in outputs:
            assert observed["workspace"] == workspace_observation(
                reference.trace.teacher.workspace_trace
            )
            torch.testing.assert_close(
                torch.tensor(observed["prefill_logits"]), logits[0], rtol=1e-5, atol=1e-6
            )
        assert outputs[0] == outputs[1]
        assert outputs[0]["tokens"][0][0] == int(logits[0].argmax(-1)[0])
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert build_scientific_ablation_checkpoint_v1(controller._run) == current
        assert len(logits) == 1
        expected[mode] = {**compact_outputs(outputs), "teacher_logits": logits[0]}
        job["cases"].append(
            {"mode": mode, "path": str(path), "anchors": anchors, "prompt": row.prompt}
        )
    return job, expected


def _invoke(
    job: dict[str, Any], tmp_path: Path, *, digest: str | None = None
) -> tuple[subprocess.CompletedProcess[bytes], int]:
    encoded = canonical_json_bytes(job)
    assert len(encoded) <= 1024**2
    path = tmp_path / "test-inputs.json"
    path.write_bytes(encoded)
    cwd = tmp_path / "independent-cwd"
    cwd.mkdir()
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    argv = [
        sys.executable,
        "-I",
        "-B",
        str(_WORKER),
        str(_package_root()),
        str(path),
        bytes_sha256(encoded) if digest is None else digest,
    ]
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            # Only the owned launch handle is stopped here. Descendant termination
            # is not certified on timeout; that run cannot be acceptance evidence.
            process.kill()
            process.wait(timeout=10)
            raise
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        launch_pid = process.pid
    # Preserve actual evidence before any late parent-side comparison can fail.
    (tmp_path / "actual-consumer.json").write_bytes(result.stdout)
    (tmp_path / "actual-consumer.stderr.txt").write_bytes(result.stderr)
    (tmp_path / "actual-launch.json").write_bytes(
        canonical_json_bytes(
            {
                "kind": "iris-owned-inference-test-launch-v1",
                "parent_pid": os.getpid(),
                "launch_pid": launch_pid,
                "returncode": result.returncode,
                "stdout_sha256": bytes_sha256(result.stdout),
                "stderr_sha256": bytes_sha256(result.stderr),
                "job_sha256": bytes_sha256(encoded),
            }
        )
    )
    return result, launch_pid


def test_actual_executable_consumer_all12_without_training_owner_and_full_sampling(
    tmp_path: Path,
) -> None:
    job, expected = _producer(tmp_path)
    root = _package_root()
    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "iris").rglob("*.py")
    }
    disk = {
        str(path): (path.stat().st_ino, path.read_bytes())
        for path in (tmp_path / "published").rglob("*")
        if path.is_file()
    }
    caller = capture_runtime_state(torch.device("cpu"))
    result, launch_pid = _invoke(job, tmp_path)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert len(result.stdout) <= _MAX_RESULT
    observed = json.loads(result.stdout)
    assert observed["pid"] != os.getpid()
    assert (observed["pid"] == launch_pid and observed["parent_pid"] == os.getpid()) or (
        observed["pid"] != launch_pid and observed["parent_pid"] == launch_pid
    )
    assert observed["isolated"] == 1
    assert Path(observed["cwd"]) == tmp_path / "independent-cwd"
    assert Path(observed["package_root"]) == root
    assert observed["training_owner_instances"] == 0
    assert observed["training_owner_constructors_called"] == 0
    assert {case["mode"] for case in observed["cases"]} == set(expected)
    assert len(observed["cases"]) == 12
    for case in observed["cases"]:
        reference = expected[case["mode"]]
        assert case["outputs"] == reference["outputs"]
        assert case["prefill_logits"] == reference["prefill_logits"]
        torch.testing.assert_close(
            torch.tensor(case["prefill_logits"]),
            reference["teacher_logits"],
            rtol=1e-5,
            atol=1e-6,
        )
    assert observed["trust_anchor_denials"] == [0, 1, 2]
    assert {
        (fault["kind"], fault["secondary"]) for fault in observed["primary_signal_faults"]
    } == set(product(("interrupt", "exit", "group"), (False, True)))
    assert len(observed["primary_signal_faults"]) == 6
    assert len(observed["loaded_package_files"]) > 100
    for binding in observed["loaded_package_files"]:
        assert before[binding["path"]] == binding["sha256"]
        assert (
            hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
        )
    assert {
        str(path): (path.stat().st_ino, path.read_bytes())
        for path in (tmp_path / "published").rglob("*")
        if path.is_file()
    } == disk
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_worker_rejects_untrusted_job_digest_before_iris_import(tmp_path: Path) -> None:
    result, _ = _invoke({}, tmp_path, digest="0" * 64)
    assert result.returncode != 0
    assert b"AssertionError" in result.stderr and not result.stdout
