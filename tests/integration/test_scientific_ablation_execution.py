"""Real tiny ablation phases, original offline HF loaders, durable/cold journal recovery."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_execution as execution
from iris.application.scientific_ablation_command_composition import (
    create_scientific_ablation_command_lifecycle_v1,
)
from iris.application.scientific_ablation_command_config import (
    IrisScientificAblationCommandConfigV1,
)
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_execution import (
    IrisScientificExecutionError,
    read_scientific_execution_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration._offline_composition_fixture import tiny_offline_cache
from tests.integration.test_scientific_ablation_command_composition import _recipe
from tests.integration.test_scientific_teacher import _build

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cache_ready() -> Path:
    """Original offline loaders; one complete immutable tiny fixture across modules."""
    return tiny_offline_cache(lambda: _build(cycles=1))


def _config(
    base: Path,
    *,
    cycles: int = 3,
    routing: str = "learned",
    goal: str = "enabled",
    broadcast: str = "enabled",
) -> IrisScientificAblationCommandConfigV1:
    def mutate(raw: dict[str, Any]) -> None:
        raw["workspace"]["cycles"] = cycles
        raw["ablation"].update(routing=routing, goal=goal, broadcast=broadcast)
        if routing == "random":
            raw["ablation"]["routing_seed"] = 42
        raw["lifecycle"].update(checkpoint_every_steps=2, minimum_improvement=1e6)

    return _recipe(base, mutate=mutate)


def _owner(
    config: IrisScientificAblationCommandConfigV1, base: Path
) -> IrisScientificAblationLifecycleV1:
    return create_scientific_ablation_command_lifecycle_v1(config=config, base=base)


def _finish(owner: execution.IrisScientificAblationExecutionV1) -> None:
    for _ in range(8):
        head = owner.last_event
        if head is not None and head.payload.after.phase == "stopped":
            break
        owner.advance()
    assert owner.last_event is not None and owner.last_event.payload.after.phase == "stopped"


def _tree(root: Path) -> dict[str, bytes]:
    def reject(error: OSError) -> None:
        raise error

    result = {}
    for directory, directories, files in os.walk(root, onerror=reject, followlinks=False):
        parent = Path(directory)
        for node in (parent, *(parent / name for name in directories)):
            assert (
                not node.is_symlink() and not getattr(node.lstat(), "st_file_attributes", 0) & 0x400
            )
        for name in files:
            path = parent / name
            assert not path.is_symlink() and path.lstat().st_nlink == 1
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


@pytest.fixture(scope="module")
def reference(
    tmp_path_factory: pytest.TempPathFactory, cache_ready: Path
) -> tuple[Path, dict[str, bytes]]:
    base = tmp_path_factory.mktemp("ablation-execution-reference")
    config = _config(base)
    native = _owner(config, base)
    owner = execution.create_scientific_ablation_execution_v1(native, root=base / "run")
    _finish(owner)
    snapshot = execution.read_scientific_ablation_execution_v1(owner.path)
    assert cache_ready.is_dir() and len(snapshot.events) == 8
    assert snapshot.events[-1].payload.after.step == 3
    assert snapshot.events[-1].payload.after.best_step == 1
    return owner.path, _tree(owner.path)


@pytest.mark.parametrize("cycles", [1, 5])
@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_full_actual_journal_all24_modes_and_selected_best(
    tmp_path: Path, cache_ready: Path, cycles: int, routing: str, goal: str, broadcast: str
) -> None:
    config = _config(tmp_path, cycles=cycles, routing=routing, goal=goal, broadcast=broadcast)
    native = _owner(config, tmp_path)
    caller = capture_runtime_state(torch.device("cpu"))
    with patch(
        "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No initial forward")
    ) as forward:
        owner = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    assert forward.call_count == 0 and owner.last_event is None
    _finish(owner)
    snapshot = execution.read_scientific_ablation_execution_v1(owner.path)
    assert cache_ready.is_dir() and len(snapshot.events) == 8
    assert snapshot.events[-1].payload.after.step == 3
    assert snapshot.events[-1].payload.after.best_step == 1
    assert snapshot.header.payload.initial.payload.completed_steps == 0
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    before = _tree(owner.path)
    with (
        patch(
            "torch.nn.Module.__init__", side_effect=AssertionError("No model in DATA reader")
        ) as graph,
        pytest.raises(IrisScientificExecutionError),
    ):
        read_scientific_execution_v1(owner.path)
    assert graph.call_count == 0
    with patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph:
        assert execution.read_scientific_ablation_execution_v1(owner.path).events == snapshot.events
    assert graph.call_count == 0 and _tree(owner.path) == before
    (tmp_path / "actual-execution-witness.json").write_text(
        json.dumps(
            dict(
                cycles=cycles,
                routing=routing,
                goal=goal,
                broadcast=broadcast,
                path=str(owner.path),
                header_sha256=snapshot.header.sha256,
                events=[event.sha256 for event in snapshot.events],
                current_step=3,
                best_step=1,
                phase_count=8,
                full_operator_installed_or_scientific_acceptance=False,
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("cut", list(range(9)))
def test_every_phase_cut_restores_and_exactly_replays_to_uninterrupted_bytes(
    tmp_path: Path, cache_ready: Path, reference: tuple[Path, dict[str, bytes]], cut: int
) -> None:
    config = _config(tmp_path)
    native = _owner(config, tmp_path)
    first = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    for _ in range(cut):
        first.advance()
    prefix = _tree(first.path)
    fresh = _owner(config, tmp_path)
    resumed = execution.resume_scientific_ablation_execution_v1(first.path, lifecycle=fresh)
    assert _tree(first.path) == prefix and cache_ready.is_dir()
    _finish(resumed)
    assert _tree(resumed.path) == reference[1]
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("Stale owner")
        ) as forward,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        first.advance()
    assert forward.call_count == 0
    with (tmp_path / "actual-phase-cut-witness.json").open("xb") as stream:
        stream.write(
            canonical_json_file_bytes(
                dict(
                    cut=cut,
                    path=str(resumed.path),
                    reference_path=str(reference[0]),
                    prefix_sha256={name: sha256(raw).hexdigest() for name, raw in prefix.items()},
                    final_tree_byte_parity=True,
                    stale_owner_forward_calls=forward.call_count,
                )
            )
        )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_failed_phase_publication_retry_never_repeats_native_work(
    tmp_path: Path, cache_ready: Path, checkpoint: bool
) -> None:
    config = _config(tmp_path)
    native = _owner(config, tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    for _ in range(4 if checkpoint else 0):
        owner.advance()
    function = (
        "write_scientific_ablation_lifecycle_checkpoint_v1"
        if checkpoint
        else "rename_directory_with_windows_retry"
    )
    with (
        patch.object(execution, function, side_effect=OSError("owned publication fault")),
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        owner.advance()
    assert owner.publication_pending and not owner.poisoned
    step = native.state.step
    with patch(
        "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No native replay")
    ) as forward:
        event = owner.advance()
    assert forward.call_count == 0 and native.state.step == step
    assert event.payload.sequence == (5 if checkpoint else 1)
    assert not owner.publication_pending and cache_ready.is_dir()
    _finish(owner)


def test_unrecorded_actual_phase_signal_poison_denies_reuse(
    tmp_path: Path, cache_ready: Path
) -> None:
    config = _config(tmp_path)
    native = _owner(config, tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    original = native.advance
    signal = KeyboardInterrupt("after actual native commit")

    def interrupted() -> Any:
        original()
        raise signal

    with (
        patch.object(native, "advance", new=interrupted),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        owner.advance()
    assert caught.value is signal and owner.poisoned and native.state.step == 1
    assert execution.read_scientific_ablation_execution_v1(owner.path).events == ()
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("Poisoned")
        ) as forward,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        owner.advance()
    assert forward.call_count == 0 and cache_ready.is_dir()


def test_reentry_rejected_without_a_second_native_phase(tmp_path: Path, cache_ready: Path) -> None:
    config = _config(tmp_path)
    native = _owner(config, tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    original = native.advance
    calls = []

    def nested() -> Any:
        with pytest.raises(execution.IrisScientificAblationExecutionError):
            owner.advance()
        calls.append(True)
        return original()

    with patch.object(native, "advance", new=nested):
        event = owner.advance()
    assert calls == [True] and event.payload.sequence == 1 and native.state.step == 1
    assert cache_ready.is_dir()


@pytest.mark.parametrize("ceiling", [0, -1, True, 200001])
def test_replay_ceiling_denial_precedes_any_restore_or_forward(
    tmp_path: Path, cache_ready: Path, ceiling: int
) -> None:
    config = _config(tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(
        _owner(config, tmp_path), root=tmp_path / "run"
    )
    owner.advance()
    fresh = _owner(config, tmp_path)
    before = _tree(owner.path)
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No replay")
        ) as forward,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        execution.resume_scientific_ablation_execution_v1(
            owner.path, lifecycle=fresh, maximum_replay_phases=ceiling
        )
    assert forward.call_count == 0 and fresh.state.step == 0 and not fresh.poisoned
    assert _tree(owner.path) == before and cache_ready.is_dir()


def test_native_initial_mismatch_and_external_publisher_are_not_accepted(
    tmp_path: Path, cache_ready: Path
) -> None:
    config = _config(tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(
        _owner(config, tmp_path), root=tmp_path / "run"
    )
    raw = json.loads(config.model_dump_json())
    raw["probe_initialization_seed"] += 1
    other = IrisScientificAblationCommandConfigV1.model_validate_json(json.dumps(raw))
    fresh = _owner(other, tmp_path)
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No replay")
        ) as forward,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        execution.resume_scientific_ablation_execution_v1(owner.path, lifecycle=fresh)
    assert forward.call_count == 0 and fresh.state.step == 0
    published = create_scientific_ablation_command_lifecycle_v1(
        config=config, base=tmp_path, checkpoint_root=tmp_path / "other-publisher"
    )
    with pytest.raises(execution.IrisScientificAblationExecutionError):
        execution.create_scientific_ablation_execution_v1(published, root=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists() and cache_ready.is_dir()


def _rewrite_chain(path: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    paths = sorted((path / "events").iterdir())
    records = [json.loads((folder / "event.json").read_bytes()) for folder in paths]
    mutate(records)
    previous = json.loads((path / "run.json").read_bytes())["sha256"]
    for folder, value in zip(paths, records, strict=True):
        value["payload"]["previous_sha256"] = previous
        value["sha256"] = domain_sha256(
            "iris/scientific-ablation-execution-event/v1", canonical_json_bytes(value["payload"])
        )
        (folder / "event.json").write_bytes(canonical_json_file_bytes(value))
        previous = value["sha256"]


@pytest.mark.parametrize(
    "fault",
    [
        "mode",
        "model",
        "scientific-state",
        "examples",
        "loss",
        "participation",
        "validation-count",
        "selected-best",
        "baseline-kind",
        "current-bytes",
        "best-bytes",
    ],
)
def test_rehashed_foreign_or_corrupt_phases_fail_read_only_audit(
    tmp_path: Path, reference: tuple[Path, dict[str, bytes]], fault: str
) -> None:
    path = tmp_path / reference[0].name
    shutil.copytree(reference[0], path)
    if fault in ("current-bytes", "best-bytes"):
        name = "current.safetensors" if fault == "current-bytes" else "best.safetensors"
        target = next((path / "checkpoints/scientific-ablation-lifecycle-v1").glob("*/" + name))
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    else:

        def mutate(records: list[dict[str, Any]]) -> None:
            payload = records[0]["payload"]
            metric = payload["record"]["metric"]
            if fault == "mode":
                metric["ablation_policy_sha256"] = "0" * 64
            elif fault == "model":
                metric["model_configuration_sha256"] = "0" * 64
            elif fault == "scientific-state":
                payload["scientific_state_sha256"] = "0" * 64
            elif fault == "examples":
                payload["examples_seen"] += 1
            elif fault == "loss":
                metric["loss"] += 1.0
            elif fault == "participation":
                metric["registered_parameter_count"] += 1
            elif fault == "validation-count":
                metric = records[1]["payload"]["record"]["metric"]
                metric["target_token_count"] += 1
                metric["target_nll_per_token"] = (
                    metric["target_nll_sum"] / metric["target_token_count"]
                )
            elif fault == "selected-best":
                records[1]["payload"]["record"]["selected_checkpoint_sha256"] = "0" * 64
            elif fault == "baseline-kind":
                payload["kind"] = "iris-scientific-execution-event-v1"

        _rewrite_chain(path, mutate)
    before = _tree(path)
    with (
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        execution.read_scientific_ablation_execution_v1(path)
    assert graph.call_count == 0 and _tree(path) == before


def test_cross_process_os_lock_denies_a_second_native_writer(
    tmp_path: Path, cache_ready: Path
) -> None:
    config = _config(tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(
        _owner(config, tmp_path), root=tmp_path / "run"
    )
    stage = Path(__file__).resolve().parents[2]
    code = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from iris.utils.process_file_lock import process_file_lock
with process_file_lock(Path(sys.argv[2])):
    print('LOCKED',flush=True)
    sys.stdin.buffer.read(1)
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-u",
            "-c",
            code,
            str(stage),
            str(owner.path / ".iris-owner.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == b"LOCKED"
        with (
            patch(
                "transformers.Qwen3ForCausalLM.forward",
                side_effect=AssertionError("No second native writer"),
            ) as forward,
            pytest.raises(execution.IrisScientificAblationExecutionError),
        ):
            owner.advance()
        assert forward.call_count == 0
    finally:
        child.communicate(input=b"x", timeout=120)
    assert child.returncode == 0 and cache_ready.is_dir()
    assert owner.advance().payload.sequence == 1


COLD = """import json,os,sys
from pathlib import Path
stage,config_path,root,job_path=map(Path,sys.argv[1:5])
mode,cut=sys.argv[5],int(sys.argv[6])
sys.path.insert(0,str(stage/'src'))
assert not any(n=='iris' or n.startswith('iris.') for n in sys.modules)
import iris.application.scientific_ablation_execution as journal
import iris.application.scientific_ablation_command_config as recipes
import iris.application.scientific_ablation_command_composition as composition
from unittest.mock import patch
config=recipes.read_scientific_ablation_command_config_v1(config_path)
native=composition.create_scientific_ablation_command_lifecycle_v1(
    config=config,base=config_path.parent)
def emit(path):
    for name,module in tuple(sys.modules.items()):
        if name=='iris' or name.startswith('iris.'):
            assert Path(module.__file__).resolve().is_relative_to(stage/'src')
    value=dict(mode=mode,cut=cut,pid=os.getpid(),parent_pid=os.getppid(),path=str(path))
    with job_path.open('xb') as f:
        f.write(json.dumps(value,sort_keys=True).encode()+b'\\n');f.flush();os.fsync(f.fileno())
    print('ACTUAL_ABLATION_COLD '+json.dumps(value),flush=True)
if mode=='abrupt':
    owner=journal.create_scientific_ablation_execution_v1(native,root=root)
    original=journal.rename_directory_with_windows_retry
    def publish(source,target):
        original(source,target)
        if target.parent.name=='events' and target.name==f'{cut:08d}':
            emit(owner.path)
            os._exit(83)
    with patch.object(journal,'rename_directory_with_windows_retry',new=publish):
        for _ in range(cut):owner.advance()
    raise AssertionError('Required actual post-publication hard exit did not occur')
else:
    candidates=list((root/'scientific-ablation-executions-v1').iterdir())
    path=next(p for p in candidates if p.is_dir())
    owner=journal.resume_scientific_ablation_execution_v1(path,lifecycle=native)
    for _ in range(8):
        if owner.last_event is not None and owner.last_event.payload.after.phase=='stopped':break
        owner.advance()
    assert owner.last_event.payload.after.phase=='stopped'
    emit(owner.path)
"""


@pytest.mark.parametrize("cut", [1, 5])
def test_actual_hard_exit_locked_after_phase_publication_and_cold_recovery(
    tmp_path: Path, cache_ready: Path, reference: tuple[Path, dict[str, bytes]], cut: int
) -> None:
    config = _config(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(config.model_dump_json(), encoding="utf-8")
    stage = Path(__file__).resolve().parents[2]
    root = tmp_path / "run"
    original_files: dict[Path, tuple[bytes, tuple[int, ...]]] = {}
    prefix_before: dict[str, Any] = {}
    for mode, expected in (("abrupt", 83), ("resume", 0)):
        job = tmp_path / (mode + ".native.json")
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-u",
            "-c",
            COLD,
            str(stage),
            str(recipe),
            str(root),
            str(job),
            mode,
            str(cut),
        ]
        child = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = child.communicate(timeout=180)
        (tmp_path / (mode + ".stdout.log")).write_bytes(stdout)
        (tmp_path / (mode + ".stderr.log")).write_bytes(stderr)
        assert child.returncode == expected, stderr.decode()
        value = json.loads(job.read_bytes())
        assert (value["pid"] == child.pid and value["parent_pid"] == os.getpid()) or (
            value["pid"] != child.pid and value["parent_pid"] == child.pid
        )
        (tmp_path / (mode + ".process.json")).write_text(
            json.dumps(
                dict(
                    actual_exit_code=child.returncode,
                    actual_launch_pid=child.pid,
                    observer_pid=os.getpid(),
                    argv=argv,
                    native=value,
                )
            ),
            encoding="utf-8",
        )
        path = Path(value["path"])
        if mode == "abrupt":
            snapshot = execution.read_scientific_ablation_execution_v1(path)
            assert len(snapshot.events) == cut
            for relative in _tree(path):
                if relative.startswith("events/"):
                    file = path / relative
                    stat = file.lstat()
                    original_files[file] = (
                        file.read_bytes(),
                        (
                            stat.st_dev,
                            stat.st_ino,
                            stat.st_size,
                            stat.st_mtime_ns,
                            stat.st_ctime_ns,
                        ),
                    )
                    content, identity = original_files[file]
                    prefix_before[file.relative_to(path).as_posix()] = dict(
                        bytes=len(content), sha256=sha256(content).hexdigest(), native=identity
                    )
            with (tmp_path / "prefix.before.json").open("xb") as stream:
                stream.write(canonical_json_file_bytes(prefix_before))
        else:
            assert _tree(path) == reference[1]
            prefix_after: dict[str, Any] = {}
            for file, (content, identity) in original_files.items():
                stat = file.lstat()
                assert file.read_bytes() == content
                assert (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                ) == identity
                prefix_after[file.relative_to(path).as_posix()] = dict(
                    bytes=stat.st_size,
                    sha256=sha256(file.read_bytes()).hexdigest(),
                    native=(
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    ),
                )
            assert prefix_after == prefix_before and len(prefix_after) == cut
            with (tmp_path / "prefix.after.json").open("xb") as stream:
                stream.write(canonical_json_file_bytes(prefix_after))
    assert cache_ready.is_dir()


def test_actual_scalar_participation_is_not_named_tensor_count(
    tmp_path: Path, cache_ready: Path
) -> None:
    config = _config(tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(
        _owner(config, tmp_path), root=tmp_path / "run"
    )
    event = owner.advance()
    assert event.payload.record.kind == "train"
    initial = owner.header.payload.initial.payload
    shapes = {entry.key: entry.shape for entry in initial.tensors}
    scalar_count = sum(prod(shapes["model/" + name]) for name in initial.parameter_names)
    tensor_count = len(initial.parameter_names)
    assert scalar_count == 4874 and tensor_count == 74
    assert event.payload.record.metric.registered_parameter_count == scalar_count
    assert cache_ready.is_dir()

    corrupt = tmp_path / "corrupt" / owner.path.name
    shutil.copytree(owner.path, corrupt)

    def substitute_tensor_count(events: list[dict[str, Any]]) -> None:
        payload = events[0]["payload"]
        metric = payload["record"]["metric"]
        metric.update(
            registered_parameter_count=tensor_count,
            gradient_present_parameter_count=0,
            nonzero_gradient_parameter_count_before_clip=0,
        )
        # Rehash history too: the scalar-count contract must reject coherent DATA,
        # not merely a stale event/history checksum or an impossible gradient ratio.
        payload["history_sha256"] = domain_sha256(
            "iris/scientific-ablation-run-history/v1",
            canonical_json_bytes(
                dict(
                    previous=initial.history_sha256,
                    result=metric,
                    examples_seen=payload["examples_seen"],
                    target_tokens_seen=payload["target_tokens_seen"],
                )
            ),
        )

    _rewrite_chain(corrupt, substitute_tensor_count)
    before = _tree(corrupt)
    with (
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        execution.read_scientific_ablation_execution_v1(corrupt)
    assert graph.call_count == 0 and _tree(corrupt) == before
