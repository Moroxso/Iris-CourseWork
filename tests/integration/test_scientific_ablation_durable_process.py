"""Independent spawned-process proof of a real durable stochastic continuation."""

import json
import multiprocessing
import os
import random
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic
from typing import Literal, cast

import pytest
import torch

import iris
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    build_scientific_ablation_checkpoint_v1,
)
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt
from tests.integration.test_scientific_ablation_run import _stochastic

pytestmark = pytest.mark.integration

_MAX_MESSAGE_BYTES = 32 * 1024
_CASE_TIMEOUT_SECONDS = 210.0
_Routing = Literal["learned", "random", "uniform"]
_Role = Literal["producer", "restorer"]


@dataclass(frozen=True, slots=True)
class _Evidence:
    pid: int
    module_root: str
    path: str
    boundary: str
    next_output: str
    following: str
    caller_preserved: bool


def _observation(controller: IrisScientificAblationLifecycleV1) -> str:
    current = build_scientific_ablation_checkpoint_v1(controller._run)
    best = controller.best_checkpoint
    assert best is not None
    assert not controller.publication_pending and not controller.poisoned
    # Hash actual canonical manifest and tensor bytes, not only their claimed identities.
    return canonical_json_bytes(
        {
            "state": controller.state.model_dump(mode="json"),
            "current_manifest": bytes_sha256(
                canonical_json_bytes(current.manifest.model_dump(mode="json"))
            ),
            "current_bytes": bytes_sha256(current.state_bytes),
            "best_manifest": bytes_sha256(
                canonical_json_bytes(best.manifest.model_dump(mode="json"))
            ),
            "best_bytes": bytes_sha256(best.state_bytes),
        }
    ).decode()


def _assert_disk_boundary(
    controller: IrisScientificAblationLifecycleV1,
    bundle: io.IrisScientificAblationLifecycleBundleV1,
    path: Path,
) -> None:
    payload = bundle.manifest.payload
    assert controller.state == payload.state
    assert controller.state.step == 3 and controller.state.phase == "train"
    assert controller.state.checkpoint_count == 1 and controller.state.best_step == 1
    assert payload.current.payload.completed_steps == 3
    assert payload.best is not None and payload.best.payload.completed_steps == 1
    assert payload.current.sha256 != payload.best.sha256
    assert bundle.best_state_bytes is not None
    assert payload.current.payload.runtime.python_gaussian is not None
    assert controller._run._runtime.python_rng_state[2] is not None
    assert {entry.name for entry in path.iterdir()} == {
        "manifest.json",
        "current.safetensors",
        "best.safetensors",
    }
    assert (path / "manifest.json").read_bytes() == canonical_json_file_bytes(
        bundle.manifest.model_dump(mode="json")
    )
    assert (path / "current.safetensors").read_bytes() == bundle.current_state_bytes
    assert (path / "best.safetensors").read_bytes() == bundle.best_state_bytes
    assert build_scientific_ablation_checkpoint_v1(
        controller._run
    ) == IrisScientificAblationCheckpointBundleV1(payload.current, bundle.current_state_bytes)
    assert controller.best_checkpoint == IrisScientificAblationCheckpointBundleV1(
        payload.best, bundle.best_state_bytes
    )


def _child_evidence(
    role: _Role,
    root: str,
    durable_path: str,
    routing: _Routing,
    goal: str,
    broadcast: str,
) -> _Evidence:
    # Separate, deliberately different ambient streams must not affect either owner.
    torch.set_num_threads(1)
    random.seed(7301 if role == "producer" else 9203)
    random.gauss(0, 1)
    torch.manual_seed(7307 if role == "producer" else 9209)
    caller = capture_runtime_state(torch.device("cpu"))
    controller = _durable(Path(root), routing, goal=goal, broadcast=broadcast, cycles=3)
    assert controller.state.step == 0 and controller.best_checkpoint is None
    assert controller._run.completed_steps == 0
    assert controller._run._runtime.python_rng_state[2] is None
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    if role == "producer":
        receipt = _receipt(controller)
        path = receipt.path
        bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(path)
        assert receipt.manifest == bundle.manifest
    else:
        # No producer owner, bundle, model, optimizer or expected result crosses the pipe.
        path = Path(durable_path)
        bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(path)
        io.restore_scientific_ablation_lifecycle_v1(bundle, controller=controller)
    _assert_disk_boundary(controller, bundle, path)
    boundary = _observation(controller)
    best = controller.best_checkpoint
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    with _stochastic(controller._run):
        output = controller.advance()
    assert type(output) is IrisScientificAblationOptimizationOutputV1
    assert output.step_index == 4 and controller.state.step == 4
    assert controller.state.phase == "validation"
    assert controller._run._runtime.python_rng_state[2] is None
    following_current = build_scientific_ablation_checkpoint_v1(controller._run)
    assert following_current.manifest.payload.completed_steps == 4
    assert (
        following_current.manifest.payload.scientific_state_sha256
        != bundle.manifest.payload.current.payload.scientific_state_sha256
    )
    assert controller.best_checkpoint == best
    following = _observation(controller)
    assert boundary != following
    preserved = _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert preserved
    assert iris.__file__ is not None
    return _Evidence(
        pid=os.getpid(),
        module_root=str(Path(iris.__file__).resolve().parent),
        path=str(path),
        boundary=boundary,
        next_output=canonical_json_bytes(asdict(output)).decode(),
        following=following,
        caller_preserved=preserved,
    )


def _child_entry(
    sender: Connection,
    role: _Role,
    root: str,
    durable_path: str,
    routing: _Routing,
    goal: str,
    broadcast: str,
) -> None:
    try:
        evidence = _child_evidence(role, root, durable_path, routing, goal, broadcast)
        encoded = canonical_json_bytes(asdict(evidence))
        assert len(encoded) <= _MAX_MESSAGE_BYTES
        sender.send_bytes(encoded)
    except BaseException:
        sender.send_bytes(canonical_json_bytes({"error": traceback.format_exc()[-4096:]}))
        raise
    finally:
        sender.close()


def _string(message: dict[str, object], key: str) -> str:
    value = message[key]
    assert isinstance(value, str)
    return value


def _decode_evidence(encoded: bytes, role: _Role) -> _Evidence:
    decoded: object = json.loads(encoded)
    assert isinstance(decoded, dict)
    message = cast(dict[str, object], decoded)
    error = message.get("error")
    if isinstance(error, str):
        pytest.fail(f"Spawned {role} failed:\n{error}")
    assert set(message) == {
        "pid",
        "module_root",
        "path",
        "boundary",
        "next_output",
        "following",
        "caller_preserved",
    }
    pid, preserved = message["pid"], message["caller_preserved"]
    assert isinstance(pid, int) and not isinstance(pid, bool)
    assert isinstance(preserved, bool)
    return _Evidence(
        pid=pid,
        module_root=_string(message, "module_root"),
        path=_string(message, "path"),
        boundary=_string(message, "boundary"),
        next_output=_string(message, "next_output"),
        following=_string(message, "following"),
        caller_preserved=preserved,
    )


def _spawn_evidence(
    role: _Role,
    root: Path,
    durable_path: str,
    routing: _Routing,
    goal: str,
    broadcast: str,
    deadline: float,
) -> _Evidence:
    # spawn inherits the caller's sys.path; do not force a checkout src path here.
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry,
        args=(sender, role, str(root), durable_path, routing, goal, broadcast),
        name=f"iris-durable-{role}",
    )
    try:
        process.start()
        sender.close()
        while True:
            remaining = deadline - monotonic()
            assert remaining > 0, f"Spawned {role} exceeded the case deadline"
            if receiver.poll(min(remaining, 1.0)):
                try:
                    encoded = receiver.recv_bytes(maxlength=_MAX_MESSAGE_BYTES)
                except (EOFError, OSError) as error:
                    pytest.fail(f"Spawned {role} closed its bounded pipe: {error}")
                break
            assert process.is_alive(), f"Spawned {role} exited without evidence: {process.exitcode}"
        process.join(timeout=max(0.0, deadline - monotonic()))
        assert not process.is_alive(), f"Spawned {role} did not exit after sending evidence"
        evidence = _decode_evidence(encoded, role)
        assert process.exitcode == 0, f"Spawned {role} exited with code {process.exitcode}"
        return evidence
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                with suppress(OSError):
                    process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                with suppress(OSError):
                    process.kill()
                process.join(timeout=5.0)
            assert not process.is_alive(), f"Could not terminate owned {role} process"
            process.close()


@pytest.mark.parametrize(
    ("routing", "goal", "broadcast"),
    (
        pytest.param("learned", "enabled", "enabled", id="learned-c3"),
        pytest.param("random", "disabled", "enabled", id="random-no-goal-c3"),
        pytest.param("uniform", "enabled", "disabled", id="uniform-no-broadcast-c3"),
    ),
)
def test_durable_boundary_survives_independent_fresh_process_and_exact_next_update(
    tmp_path: Path,
    routing: _Routing,
    goal: str,
    broadcast: str,
    record_property: Callable[[str, object], None],
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    deadline = monotonic() + _CASE_TIMEOUT_SECONDS
    producer = _spawn_evidence(
        "producer", tmp_path / "producer", "", routing, goal, broadcast, deadline
    )
    # The producing process has exited before an independently initialized reader exists.
    restorer = _spawn_evidence(
        "restorer", tmp_path / "restorer", producer.path, routing, goal, broadcast, deadline
    )
    assert len({os.getpid(), producer.pid, restorer.pid}) == 3
    assert iris.__file__ is not None
    module_root = str(Path(iris.__file__).resolve().parent)
    assert producer.module_root == restorer.module_root == module_root
    assert producer.path == restorer.path
    assert producer.boundary == restorer.boundary
    assert producer.next_output == restorer.next_output
    assert producer.following == restorer.following
    assert producer.caller_preserved and restorer.caller_preserved
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    record_property("durable_parent_pid", os.getpid())
    record_property("durable_producer_pid", producer.pid)
    record_property("durable_restorer_pid", restorer.pid)
    record_property("durable_parent_iris_root", module_root)
    record_property("durable_producer_iris_root", producer.module_root)
    record_property("durable_restorer_iris_root", restorer.module_root)
