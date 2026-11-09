"""Actual tiny offline operators, one-phase CAS, DATA retries and process recovery."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_execution as execution
from iris.application import scientific_ablation_operator as operator
from iris.artifacts.scientific_ablation_operator_v1 import (
    IrisScientificAblationPhaseRequestV1,
    Phase,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from tests.integration.test_scientific_ablation_execution import (
    _config,
    _finish,
    _owner,
    _tree,
)
from tests.integration.test_scientific_ablation_execution import (
    cache_ready as cache_ready,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def reference(
    tmp_path_factory: pytest.TempPathFactory, cache_ready: Path
) -> tuple[Path, dict[str, bytes]]:
    """Real uninterrupted native oracle with the SAME explicit operator recipe quota."""
    base = tmp_path_factory.mktemp("ablation-execution-reference")
    config = _config(base)
    (base / "reference.recipe.json").write_bytes(
        canonical_json_file_bytes(config.model_dump(mode="json"))
    )
    native = _owner(config, base)
    owner = execution.create_scientific_ablation_execution_v1(
        native, root=base / "run", maximum_journal_bytes=config.maximum_journal_bytes
    )
    _finish(owner)
    snapshot = execution.read_scientific_ablation_execution_v1(owner.path)
    assert snapshot.header.payload.maximum_journal_bytes == config.maximum_journal_bytes
    assert len(snapshot.events) == 8 and snapshot.events[-1].payload.after.phase == "stopped"
    assert snapshot.events[-1].payload.after.step == 3
    assert snapshot.events[-1].payload.after.best_step == 1 and cache_ready.is_dir()
    return owner.path, _tree(owner.path)


def _sealed(value: dict[str, Any]) -> IrisScientificAblationPhaseRequestV1:
    value["sha256"] = domain_sha256(
        "iris/scientific-ablation-phase-request/v1", canonical_json_bytes(value["payload"])
    )
    return IrisScientificAblationPhaseRequestV1.model_validate_json(json.dumps(value))


def _due(snapshot: operator.IrisScientificAblationOperatorSnapshotV1) -> Phase:
    value = (
        "train"
        if not snapshot.execution.events
        else snapshot.execution.events[-1].payload.after.phase
    )
    if value == "train":
        return "train"
    if value == "validation":
        return "validation"
    if value == "checkpoint":
        return "checkpoint"
    raise AssertionError("Stopped operator has no due phase")


@pytest.mark.parametrize("cycles", [1, 5])
@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_full_actual_operator_all24_modes_and_identical_old_request_retries(
    tmp_path: Path,
    cache_ready: Path,
    cycles: int,
    routing: str,
    goal: str,
    broadcast: str,
) -> None:
    config = _config(tmp_path, cycles=cycles, routing=routing, goal=goal, broadcast=broadcast)
    created = operator.create_scientific_ablation_operator_v1(
        config=config, base=tmp_path, root=tmp_path / "operators"
    )
    requests = []
    results = []
    for _ in range(8):
        current = operator.inspect_scientific_ablation_operator_v1(created.path)
        previous = current.execution.events[-1] if current.execution.events else None
        phase = _due(current)
        request = operator.make_scientific_ablation_phase_request_v1(
            created.path,
            expected_previous_sha256=(
                current.execution.header.sha256 if previous is None else previous.sha256
            ),
            phase=phase,
        )
        result = operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
        requests.append(request)
        results.append(result)
        with patch.object(
            operator,
            "create_scientific_ablation_command_lifecycle_v1",
            side_effect=AssertionError("DATA retry cannot build"),
        ) as factory:
            assert (
                operator.advance_scientific_ablation_operator_v1(
                    created.path, request=request, base=tmp_path
                )
                == result
            )
        assert factory.call_count == 0
    finished = operator.inspect_scientific_ablation_operator_v1(created.path)
    assert len(finished.execution.events) == 8
    assert finished.execution.events[-1].payload.after.phase == "stopped"
    assert finished.execution.events[-1].payload.after.best_step == 1
    before = _tree(created.path)
    with (
        patch.object(
            operator,
            "create_scientific_ablation_command_lifecycle_v1",
            side_effect=AssertionError("No new native owner"),
        ) as factory,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
    ):
        repeat_create = operator.create_scientific_ablation_operator_v1(
            config=config, base=tmp_path, root=tmp_path / "operators"
        )
        assert repeat_create.descriptor == created.descriptor
        for request, expected in zip(requests, results, strict=True):
            assert (
                operator.advance_scientific_ablation_operator_v1(
                    created.path, request=request, base=tmp_path
                )
                == expected
            )
    assert factory.call_count == graph.call_count == 0 and _tree(created.path) == before
    assert cache_ready.is_dir()
    (tmp_path / "actual-operator-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                path=str(created.path),
                cycles=cycles,
                routing=routing,
                goal=goal,
                broadcast=broadcast,
                descriptor_sha256=created.descriptor.sha256,
                execution_path=str(finished.execution.path),
                requests=[r.model_dump(mode="json") for r in requests],
                results=[r.model_dump(mode="json") for r in results],
                identical_immediate_and_late_retries=True,
                actual_phases=8,
                full_operator_installed_or_scientific_acceptance=False,
            )
        )
    )


@pytest.mark.parametrize("phase", ["validation", "checkpoint"])
def test_wrong_initial_phase_has_no_factory_or_forward(
    tmp_path: Path, cache_ready: Path, phase: Any
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase=phase
    )
    before = _tree(created.path)
    with (
        patch.object(
            operator,
            "create_scientific_ablation_command_lifecycle_v1",
            side_effect=AssertionError("No build"),
        ) as factory,
        pytest.raises(operator.IrisScientificAblationOperatorError),
    ):
        operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
    assert factory.call_count == 0 and _tree(created.path) == before and cache_ready.is_dir()


@pytest.mark.parametrize("ceiling", [-1, True, 200001, "1"])
def test_invalid_replay_control_is_not_ignored_on_committed_retry(
    tmp_path: Path,
    cache_ready: Path,
    ceiling: Any,
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase="train"
    )
    operator.advance_scientific_ablation_operator_v1(created.path, request=request, base=tmp_path)
    with (
        patch.object(operator, "create_scientific_ablation_command_lifecycle_v1") as factory,
        pytest.raises(operator.IrisScientificAblationOperatorError),
    ):
        operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path, maximum_replay_phases=ceiling
        )
    assert factory.call_count == 0 and cache_ready.is_dir()


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator_sha256", "0" * 64),
        ("execution_sha256", "0" * 64),
        ("expected_previous_sha256", "0" * 64),
        ("purpose", "scientific"),
    ],
)
def test_rehashed_foreign_request_never_builds_a_model(
    tmp_path: Path,
    cache_ready: Path,
    field: str,
    value: str,
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase="train"
    ).model_dump(mode="json")
    request["payload"][field] = value
    with (
        patch.object(operator, "create_scientific_ablation_command_lifecycle_v1") as factory,
        pytest.raises(operator.IrisScientificAblationOperatorError),
    ):
        operator.advance_scientific_ablation_operator_v1(
            created.path, request=_sealed(request), base=tmp_path
        )
    assert factory.call_count == 0 and cache_ready.is_dir()


@pytest.mark.parametrize("fault", [OSError("publication"), KeyboardInterrupt(), SystemExit(9)])
def test_completed_initial_publication_retry_is_data_only_and_preserves_first_signal(
    tmp_path: Path,
    cache_ready: Path,
    fault: BaseException,
) -> None:
    config = _config(tmp_path)
    expected = (
        operator.IrisScientificAblationOperatorError
        if isinstance(fault, Exception)
        else type(fault)
    )
    with (
        patch.object(operator, "rename_directory_with_windows_retry", side_effect=fault),
        pytest.raises(expected),
    ):
        operator.create_scientific_ablation_operator_v1(
            config=config, base=tmp_path, root=tmp_path / "operators"
        )
    with patch.object(
        operator,
        "create_scientific_ablation_command_lifecycle_v1",
        side_effect=AssertionError("Retry cannot build"),
    ) as factory:
        recovered = operator.create_scientific_ablation_operator_v1(
            config=config, base=tmp_path, root=tmp_path / "operators"
        )
    assert factory.call_count == 0 and not recovered.execution.events and cache_ready.is_dir()


@pytest.mark.parametrize("signal", [KeyboardInterrupt(), SystemExit(17)])
def test_actual_committed_phase_lost_ack_returns_same_result_without_native_retry(
    tmp_path: Path,
    cache_ready: Path,
    signal: BaseException,
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase="train"
    )
    with patch.object(operator, "_result", side_effect=signal), pytest.raises(type(signal)):
        operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
    snapshot = operator.inspect_scientific_ablation_operator_v1(created.path)
    assert len(snapshot.execution.events) == 1
    before = _tree(created.path)
    with patch.object(
        operator,
        "create_scientific_ablation_command_lifecycle_v1",
        side_effect=AssertionError("No repeat"),
    ) as factory:
        result = operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
    assert result.payload.event == snapshot.execution.events[0]
    assert factory.call_count == 0 and _tree(created.path) == before and cache_ready.is_dir()


@pytest.mark.parametrize("fault", ["prefix", "phase", "prefix-only", "phase-only"])
def test_locked_journal_compare_and_advance_denies_before_native_work(
    tmp_path: Path,
    cache_ready: Path,
    fault: str,
) -> None:
    native = _owner(_config(tmp_path), tmp_path)
    owner = execution.create_scientific_ablation_execution_v1(native, root=tmp_path / "run")
    prefix = "0" * 64 if fault == "prefix" else owner.header.sha256
    phase: Phase = "validation" if fault == "phase" else "train"
    with (
        patch.object(native, "advance", side_effect=AssertionError("No native phase")) as work,
        pytest.raises(execution.IrisScientificAblationExecutionError),
    ):
        owner.advance(
            expected_previous_sha256=None if fault == "phase-only" else prefix,
            expected_phase=None if fault == "prefix-only" else phase,
        )
    assert work.call_count == 0 and owner.last_event is None and not owner.poisoned
    assert cache_ready.is_dir()


WORKER = r"""import json,os,sys
from pathlib import Path
from unittest.mock import patch
stage,input_path,root,job_path=map(Path,sys.argv[1:5])
mode,cut=sys.argv[5],int(sys.argv[6])
sys.path[:0]=[str(stage/'src')]
import iris.application.scientific_ablation_operator as op
from iris.application.scientific_ablation_command_config import (
    read_scientific_ablation_command_config_v1)

def emit(snapshot,receipt,status='ok',no_factory=False):
    for name,module in tuple(sys.modules.items()):
        if name=='iris' or name.startswith('iris.'):
            assert Path(module.__file__).resolve().is_relative_to(stage/'src')
    value=dict(mode=mode,cut=cut,pid=os.getpid(),parent_pid=os.getppid(),
        path=str(snapshot.path),execution_path=str(snapshot.execution.path),
        receipt_sha256=receipt,status=status,retry_no_factory=no_factory)
    with job_path.open('xb') as stream:
        stream.write(json.dumps(value,sort_keys=True).encode()+b'\n')
        stream.flush();os.fsync(stream.fileno())
    print('ACTUAL_OPERATOR_COLD '+json.dumps(value),flush=True)

def request(snapshot):
    last=snapshot.execution.events[-1] if snapshot.execution.events else None
    return op.make_scientific_ablation_phase_request_v1(snapshot.path,
        expected_previous_sha256=snapshot.execution.header.sha256 if last is None else last.sha256,
        phase='train' if last is None else last.payload.after.phase)

if mode=='abrupt':
    config=read_scientific_ablation_command_config_v1(input_path)
    snapshot=op.create_scientific_ablation_operator_v1(config=config,base=input_path.parent,root=root)
    original=op._result
    def lose_ack(req,event):
        result=original(req,event)
        if event.payload.sequence==cut:
            emit(op.inspect_scientific_ablation_operator_v1(snapshot.path),result.sha256)
            os._exit(83)
        return result
    with patch.object(op,'_result',new=lose_ack):
        for _ in range(cut):
            current=op.inspect_scientific_ablation_operator_v1(snapshot.path)
            req=request(current)
            (input_path.parent/'last.request.json').write_text(req.model_dump_json(),encoding='utf-8')
            op.advance_scientific_ablation_operator_v1(snapshot.path,request=req,base=input_path.parent)
    raise AssertionError('Expected real committed phase hard exit')
elif mode=='resume':
    path=next(p for p in (root/'scientific-ablation-operators-v1').iterdir() if p.is_dir())
    req=op.read_scientific_ablation_phase_request_v1(input_path.parent/'last.request.json')
    with patch.object(op,'create_scientific_ablation_command_lifecycle_v1',
                      side_effect=AssertionError('Cold retry is DATA-only')) as factory:
        repeated=op.advance_scientific_ablation_operator_v1(path,request=req,base=input_path.parent)
    assert factory.call_count==0
    for _ in range(8):
        current=op.inspect_scientific_ablation_operator_v1(path)
        if current.execution.events[-1].payload.after.phase=='stopped':break
        op.advance_scientific_ablation_operator_v1(path,request=request(current),base=input_path.parent)
    final=op.inspect_scientific_ablation_operator_v1(path)
    assert len(final.execution.events)==8
    assert final.execution.events[-1].payload.after.phase=='stopped'
    emit(final,repeated.sha256,no_factory=True)
else:
    req=op.read_scientific_ablation_phase_request_v1(input_path)
    prepared=op.inspect_scientific_ablation_operator_v1(root)
    print('READY',flush=True)
    sys.stdin.buffer.read(1)
    try:
        result=op.advance_scientific_ablation_operator_v1(root,request=req,base=input_path.parent)
    except op.IrisScientificAblationOperatorError:
        emit(prepared,None,status='busy_or_conflict')
        raise SystemExit(23)
    emit(op.inspect_scientific_ablation_operator_v1(root),result.sha256)
"""


def _child(
    tmp_path: Path, mode: str, cut: int, input_path: Path, root: Path
) -> tuple[subprocess.Popen[bytes], list[str], Path]:
    job = tmp_path / (mode + ".native.json")
    stage = Path(__file__).resolve().parents[2]
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-u",
        "-c",
        WORKER,
        str(stage),
        str(input_path),
        str(root),
        str(job),
        mode,
        str(cut),
    ]
    return (
        subprocess.Popen(
            argv,
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ),
        argv,
        job,
    )


def _collect(
    tmp_path: Path,
    mode: str,
    child: subprocess.Popen[bytes],
    argv: list[str],
    job: Path,
    expected: set[int],
) -> dict[str, Any]:
    stdout, stderr = child.communicate(timeout=360)
    (tmp_path / (mode + ".stdout.log")).write_bytes(stdout)
    (tmp_path / (mode + ".stderr.log")).write_bytes(stderr)
    assert child.returncode in expected, stderr.decode()
    value: dict[str, Any] = json.loads(job.read_bytes())
    assert (value["pid"] == child.pid and value["parent_pid"] == os.getpid()) or (
        value["pid"] != child.pid and value["parent_pid"] == child.pid
    )
    (tmp_path / (mode + ".process.json")).write_bytes(
        canonical_json_file_bytes(
            dict(
                actual_launch_pid=child.pid,
                observer_pid=os.getpid(),
                actual_exit_code=child.returncode,
                argv=argv,
                native=value,
            )
        )
    )
    return value


@pytest.mark.parametrize("cut", [1, 5])
def test_actual_hard_exit_after_committed_operator_phase_and_data_only_cold_retry(
    tmp_path: Path,
    cache_ready: Path,
    reference: tuple[Path, dict[str, bytes]],
    cut: int,
) -> None:
    config = _config(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(config.model_dump_json(), encoding="utf-8")
    root = tmp_path / "operators"
    child, argv, job = _child(tmp_path, "abrupt", cut, recipe, root)
    first = _collect(tmp_path, "abrupt", child, argv, job, {83})
    path = Path(first["execution_path"])
    before = {}
    for relative in _tree(path):
        if relative.startswith("events/"):
            file = path / relative
            stat = file.lstat()
            before[relative] = dict(
                content=file.read_bytes().hex(),
                native=(
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                ),
            )
    assert len(before) == cut
    (tmp_path / "prefix.before.json").write_bytes(canonical_json_file_bytes(before))
    child, argv, job = _child(tmp_path, "resume", cut, recipe, root)
    final = _collect(tmp_path, "resume", child, argv, job, {0})
    assert final["receipt_sha256"] == first["receipt_sha256"] and final["retry_no_factory"] is True
    assert _tree(path) == reference[1]
    after = {}
    for relative in before:
        file = path / relative
        stat = file.lstat()
        after[relative] = dict(
            content=file.read_bytes().hex(),
            native=(
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        )
    assert after == before and cache_ready.is_dir()
    (tmp_path / "prefix.after.json").write_bytes(canonical_json_file_bytes(after))


@pytest.mark.parametrize("incompatible", [False, True])
def test_actual_two_process_prefix_competition_never_commits_a_follow_on_phase(
    tmp_path: Path,
    cache_ready: Path,
    incompatible: bool,
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase="train"
    )
    other = operator.make_scientific_ablation_phase_request_v1(
        created.path,
        expected_previous_sha256=created.execution.header.sha256,
        phase="validation" if incompatible else "train",
    )
    workers = []
    for index, req in enumerate((request, other)):
        input_path = tmp_path / f"race{index}.request.json"
        input_path.write_text(req.model_dump_json(), encoding="utf-8")
        mode = f"race{index}"
        child, argv, job = _child(tmp_path, mode, 0, input_path, created.path)
        assert child.stdout is not None
        ready = child.stdout.readline()
        (tmp_path / (mode + ".ready.stdout.log")).write_bytes(ready)
        assert ready.strip() == b"READY"
        workers.append((mode, child, argv, job))
    for _, child, _, _ in workers:
        assert child.stdin is not None
        child.stdin.write(b"x")
        child.stdin.flush()
    for mode, child, argv, job in workers:
        _collect(tmp_path, mode, child, argv, job, {0, 23})
    final = operator.inspect_scientific_ablation_operator_v1(created.path)
    assert len(final.execution.events) == 1
    with patch.object(
        operator,
        "create_scientific_ablation_command_lifecycle_v1",
        side_effect=AssertionError("Committed retry cannot build"),
    ) as factory:
        repeated = operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
    assert repeated.payload.event == final.execution.events[0]
    assert factory.call_count == 0 and cache_ready.is_dir()


@pytest.mark.parametrize("fault", ["initialization", "backbone", "cycles", "routing"])
def test_rehashed_changed_recipe_cannot_reuse_an_existing_operator_address(
    tmp_path: Path,
    cache_ready: Path,
    fault: str,
) -> None:
    created = operator.create_scientific_ablation_operator_v1(
        config=_config(tmp_path), base=tmp_path, root=tmp_path / "operators"
    )
    corrupt = tmp_path / "corrupt" / created.path.name
    shutil.copytree(created.path, corrupt)
    value = created.descriptor.model_dump(mode="json")
    recipe = json.loads(value["payload"]["recipe_json"])
    if fault == "initialization":
        recipe["initialization_seed"] += 1
    elif fault == "backbone":
        recipe["backbone"]["revision"] = "f" * 40
    elif fault == "cycles":
        recipe["workspace"]["cycles"] = 5
    else:
        recipe["ablation"].update(routing="random", routing_seed=42)
    raw = canonical_json_bytes(recipe)
    value["payload"]["recipe_json"] = raw.decode()
    value["payload"]["recipe_sha256"] = domain_sha256(
        "iris/scientific-ablation-operator-recipe/v1", raw
    )
    value["sha256"] = domain_sha256(
        "iris/scientific-ablation-operator/v1", canonical_json_bytes(value["payload"])
    )
    (corrupt / "operator.json").write_bytes(canonical_json_file_bytes(value))
    with (
        patch.object(operator, "create_scientific_ablation_command_lifecycle_v1") as factory,
        pytest.raises(operator.IrisScientificAblationOperatorError),
    ):
        operator.inspect_scientific_ablation_operator_v1(corrupt)
    assert factory.call_count == 0 and cache_ready.is_dir()


def test_real_competing_native_phase_between_preflight_and_commit_returns_same_event(
    tmp_path: Path,
    cache_ready: Path,
) -> None:
    config = _config(tmp_path)
    created = operator.create_scientific_ablation_operator_v1(
        config=config, base=tmp_path, root=tmp_path / "operators"
    )
    request = operator.make_scientific_ablation_phase_request_v1(
        created.path, expected_previous_sha256=created.execution.header.sha256, phase="train"
    )
    competitor = execution.resume_scientific_ablation_execution_v1(
        created.execution.path, lifecycle=_owner(config, tmp_path)
    )
    original = execution.resume_scientific_ablation_execution_v1
    completed = []

    def race(*args: Any, **kwargs: Any) -> execution.IrisScientificAblationExecutionV1:
        owner = original(*args, **kwargs)
        completed.append(competitor.advance())
        return owner

    with patch.object(operator, "resume_scientific_ablation_execution_v1", new=race):
        result = operator.advance_scientific_ablation_operator_v1(
            created.path, request=request, base=tmp_path
        )
    final = operator.inspect_scientific_ablation_operator_v1(created.path)
    assert len(completed) == len(final.execution.events) == 1
    assert result.payload.event == completed[0] == final.execution.events[0]
    assert cache_ready.is_dir()


def test_uninterrupted_operator_reference_uses_exact_recipe_journal_quota(
    reference: tuple[Path, dict[str, bytes]],
) -> None:
    path, original_bytes = reference
    config = json.loads((path.parents[2] / "reference.recipe.json").read_bytes())
    snapshot = execution.read_scientific_ablation_execution_v1(path)
    assert snapshot.header.payload.maximum_journal_bytes == config["maximum_journal_bytes"]
    assert snapshot.header.payload.maximum_journal_bytes == 4 * 1024**2
    assert len(snapshot.events) == 8 and snapshot.events[-1].payload.after.phase == "stopped"
    assert _tree(path) == original_bytes
