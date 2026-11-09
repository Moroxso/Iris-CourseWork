"""Actual offline persisted-best/current export, immutable retry and publication faults."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3ForCausalLM

from iris.application import scientific_ablation_execution as execution
from iris.application import scientific_ablation_operator as operator
from iris.application import scientific_ablation_operator_export as export
from iris.application.scientific_ablation_command_composition import (
    create_scientific_ablation_command_run_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_execution_v1 import ScientificAblationExecutionCheckpointV1
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
    ScientificAblationExportRequestPayloadV1,
    Selection,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_execution import _config, _owner, _tree
from tests.integration.test_scientific_ablation_execution import cache_ready as cache_ready

pytestmark = pytest.mark.integration


def _new(
    base: Path, **modes: Any
) -> tuple[
    operator.IrisScientificAblationOperatorSnapshotV1, execution.IrisScientificAblationExecutionV1
]:
    config = _config(base, **modes)
    created = operator.create_scientific_ablation_operator_v1(
        config=config, base=base, root=base / "operators"
    )
    owner = execution.resume_scientific_ablation_execution_v1(
        created.execution.path, lifecycle=_owner(config, base)
    )
    return created, owner


def _intent(
    path: Path, selection: Selection = "selected_best", index: int = -1
) -> IrisScientificAblationExportRequestV1:
    source = operator.inspect_scientific_ablation_operator_v1(path)
    event = [
        e
        for e in source.execution.events
        if type(e.payload.record) is ScientificAblationExecutionCheckpointV1
    ][index]
    record = event.payload.record
    assert type(record) is ScientificAblationExecutionCheckpointV1
    return export.make_scientific_ablation_export_request_v1(
        path,
        checkpoint_event_sha256=event.sha256,
        expected_lifecycle_sha256=record.lifecycle_checkpoint_sha256,
        selection=selection,
        maximum_storage_bytes=4 * 1024**2,
        minimum_free_bytes=0,
    )


def _ids(root: Path) -> dict[str, tuple[int, int]]:
    return {
        name: (metadata.st_dev, metadata.st_ino)
        for name in _tree(root)
        for metadata in [(root / name).lstat()]
    }


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory, cache_ready: Path) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("ablation-export-source")
    created, owner = _new(base)
    for _ in range(8):
        owner.advance()
    assert cache_ready.is_dir()
    return created.path, base


@pytest.mark.parametrize("cycles", [1, 5])
@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_actual_best_current_export_all24_modes_and_late_data_retry(
    tmp_path: Path,
    cache_ready: Path,
    cycles: int,
    routing: str,
    goal: str,
    broadcast: str,
) -> None:
    created, owner = _new(tmp_path, cycles=cycles, routing=routing, goal=goal, broadcast=broadcast)
    initial = export.inspect_scientific_ablation_best_v1(created.path)
    assert initial.payload.validated_best_step is None
    for _ in range(2):
        owner.advance()
    selected = export.inspect_scientific_ablation_best_v1(created.path)
    assert (
        selected.payload.validated_best_step == 1
        and not selected.payload.validated_best_is_persisted
    )
    assert selected.payload.persisted_lifecycle_sha256 is None
    for _ in range(3):
        owner.advance()
    current = export.inspect_scientific_ablation_best_v1(created.path)
    assert current.payload.validated_best_is_persisted and current.payload.persisted_best_step == 1
    requests = [_intent(created.path, selection) for selection in ("current", "selected_best")]
    receipts = []
    state_bytes = []
    before_source = _tree(created.path)
    caller = capture_runtime_state(torch.device("cpu"))
    for request in requests:
        with (
            patch.object(
                Qwen3ForCausalLM, "forward", side_effect=AssertionError("Export never forwards")
            ) as forward,
            patch.object(
                export,
                "create_scientific_ablation_command_run_v1",
                wraps=create_scientific_ablation_command_run_v1,
            ) as factory,
        ):
            receipt = export.export_scientific_ablation_operator_v1(
                created.path, request=request, base=tmp_path, root=tmp_path / "exports"
            )
        assert factory.call_count == 1 and forward.call_count == 0
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        receipts.append(receipt)
        state_bytes.append((receipt.inference_path / "state.safetensors").read_bytes())
    assert [r.result.payload.selected_completed_steps for r in receipts] == [2, 1]
    assert state_bytes[0] != state_bytes[1] and _tree(created.path) == before_source
    for _ in range(3):
        owner.advance()
    frozen = _tree(created.path)
    frozen_ids = _ids(created.path)
    export_frozen = _tree(tmp_path / "exports")
    export_ids = _ids(tmp_path / "exports")
    with (
        patch.object(
            export,
            "create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("No native retry"),
        ) as factory,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
    ):
        for request, receipt in zip(requests, receipts, strict=True):
            assert (
                export.inspect_scientific_ablation_operator_export_v1(
                    created.path, request=request, root=tmp_path / "exports"
                )
                == receipt
            )
            assert (
                export.export_scientific_ablation_operator_v1(
                    created.path, request=request, base=tmp_path, root=tmp_path / "exports"
                )
                == receipt
            )
    assert factory.call_count == graph.call_count == 0
    assert _tree(created.path) == frozen and _ids(created.path) == frozen_ids
    assert _tree(tmp_path / "exports") == export_frozen and _ids(tmp_path / "exports") == export_ids
    for selection in ("current", "selected_best"):
        request = _intent(created.path, selection)
        with patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No extra phase")
        ) as forward:
            receipt = export.export_scientific_ablation_operator_v1(
                created.path, request=request, base=tmp_path, root=tmp_path / "exports"
            )
        assert forward.call_count == 0
        assert receipt.result.payload.selected_completed_steps == (
            3 if selection == "current" else 1
        )
        if selection == "selected_best":
            assert (receipt.inference_path / "state.safetensors").read_bytes() == state_bytes[1]
        else:
            assert (receipt.inference_path / "state.safetensors").read_bytes() != state_bytes[0]
        requests.append(request)
        receipts.append(receipt)
    assert (
        _tree(created.path) == frozen and _ids(created.path) == frozen_ids and cache_ready.is_dir()
    )
    (tmp_path / "actual-operator-export-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(created.path),
                export_root=str(tmp_path / "exports"),
                cycles=cycles,
                routing=routing,
                goal=goal,
                broadcast=broadcast,
                requests=[r.model_dump(mode="json") for r in requests],
                results=[r.result.model_dump(mode="json") for r in receipts],
                inference_paths=[str(r.inference_path) for r in receipts],
                earlier_current2_best1_then_current3_best1=True,
                late_retry_bytes_and_native_ids_unchanged=True,
                no_export_forward=True,
                installed_or_scientific_acceptance=False,
            )
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator_sha256", "0" * 64),
        ("execution_sha256", "0" * 64),
        ("checkpoint_event_sha256", "0" * 64),
        ("lifecycle_sha256", "0" * 64),
        ("source_checkpoint_sha256", "0" * 64),
        ("purpose", "scientific"),
    ],
)
def test_resealed_foreign_export_intents_fail_before_model_or_filesystem(
    completed: tuple[Path, Path],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path, base = completed
    raw = _intent(path).model_dump(mode="json")
    raw["payload"][field] = value
    raw["sha256"] = domain_sha256(export._REQUEST, canonical_json_bytes(raw["payload"]))
    request = IrisScientificAblationExportRequestV1.model_validate_json(json.dumps(raw))
    with (
        patch.object(export, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    assert factory.call_count == 0 and not (tmp_path / "exports").exists()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 6])
def test_train_or_validation_metadata_is_not_a_persisted_export_boundary(
    completed: tuple[Path, Path], index: int
) -> None:
    path, _ = completed
    snapshot = operator.inspect_scientific_ablation_operator_v1(path)
    with (
        patch.object(export, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.make_scientific_ablation_export_request_v1(
            path,
            checkpoint_event_sha256=snapshot.execution.events[index].sha256,
            expected_lifecycle_sha256="0" * 64,
        )
    assert factory.call_count == 0


@pytest.mark.parametrize("fault", [OSError("publication"), KeyboardInterrupt(), SystemExit(9)])
def test_complete_staged_export_retries_without_another_native_factory(
    completed: tuple[Path, Path],
    tmp_path: Path,
    fault: BaseException,
) -> None:
    path, base = completed
    request = _intent(path)
    with (
        patch.object(export, "rename_directory_with_windows_retry", side_effect=fault),
        pytest.raises(
            export.IrisScientificAblationOperatorExportError
            if isinstance(fault, Exception)
            else type(fault)
        ),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    with patch.object(
        export,
        "create_scientific_ablation_command_run_v1",
        side_effect=AssertionError("DATA retry"),
    ) as factory:
        receipt = export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    assert factory.call_count == 0 and receipt.result.payload.request == request


@pytest.mark.parametrize("fault", [OSError("lost ACK"), KeyboardInterrupt(), SystemExit(9)])
def test_actual_published_lost_ack_preserves_primary_then_exact_data_retry(
    completed: tuple[Path, Path],
    tmp_path: Path,
    fault: BaseException,
) -> None:
    path, base = completed
    request = _intent(path)
    rename = rename_directory_with_windows_retry

    def lose(source: Path, target: Path) -> None:
        rename(source, target)
        raise fault

    with (
        patch.object(export, "rename_directory_with_windows_retry", side_effect=lose),
        pytest.raises(
            export.IrisScientificAblationOperatorExportError
            if isinstance(fault, Exception)
            else type(fault)
        ),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    before, ids = _tree(tmp_path / "exports"), _ids(tmp_path / "exports")
    with patch.object(
        export, "create_scientific_ablation_command_run_v1", side_effect=AssertionError("No graph")
    ) as factory:
        result = export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    assert factory.call_count == 0 and result.result.payload.request == request
    assert _tree(tmp_path / "exports") == before and _ids(tmp_path / "exports") == ids


@pytest.mark.parametrize("fault", [OSError("partial"), KeyboardInterrupt(), SystemExit(9)])
def test_partial_unacknowledged_stage_is_preserved_not_native_reexecuted(
    completed: tuple[Path, Path],
    tmp_path: Path,
    fault: BaseException,
) -> None:
    path, base = completed
    request = _intent(path)

    def partial(target: Path, raw: bytes) -> None:
        target.write_bytes(raw[:1])
        raise fault

    with (
        patch.object(export, "_write_exact_file", side_effect=partial),
        pytest.raises(
            export.IrisScientificAblationOperatorExportError
            if isinstance(fault, Exception)
            else type(fault)
        ),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    before, ids = _tree(tmp_path / "exports"), _ids(tmp_path / "exports")
    with (
        patch.object(
            export,
            "create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("Preserve incomplete stage"),
        ) as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    assert (
        factory.call_count == 0
        and _tree(tmp_path / "exports") == before
        and _ids(tmp_path / "exports") == ids
    )


def test_export_reentry_and_actual_namespace_lock_deny_second_graph(
    completed: tuple[Path, Path], tmp_path: Path
) -> None:
    path, base = completed
    request, root = _intent(path), tmp_path / "exports"
    factory = create_scientific_ablation_command_run_v1

    def nested(**kwargs: Any) -> Any:
        with pytest.raises(export.IrisScientificAblationOperatorExportError):
            export.export_scientific_ablation_operator_v1(
                path, request=request, base=base, root=root
            )
        return factory(**kwargs)

    with patch.object(
        export, "create_scientific_ablation_command_run_v1", side_effect=nested
    ) as create:
        receipt = export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=root
        )
    assert create.call_count == 1 and receipt.result.payload.request == request


def test_inference_factory_cannot_return_an_attached_lifecycle_owner(
    completed: tuple[Path, Path], tmp_path: Path
) -> None:
    path, base = completed
    snapshot = operator.inspect_scientific_ablation_operator_v1(path)
    config = operator._recipe(snapshot.descriptor)
    lifecycle = _owner(config, base)
    with (
        patch.object(
            export, "create_scientific_ablation_command_run_v1", return_value=lifecycle._run
        ),
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=_intent(path), base=base, root=tmp_path / "exports"
        )
    assert not list((tmp_path / "exports" / export._STORE).glob(".iris-export-*.tmp"))


def test_actual_staging_replacement_is_preserved_without_foreign_writes(
    completed: tuple[Path, Path], tmp_path: Path
) -> None:
    path, base = completed
    request = _intent(path)
    original = _write_exact_file
    moved = tmp_path / "owned-stage-preserved"

    def replace(target: Path, raw: bytes) -> None:
        original(target, raw)
        stage = target.parent
        stage.rename(moved)
        stage.mkdir()
        (stage / "foreign.txt").write_bytes(b"preserve this")

    with (
        patch.object(export, "_write_exact_file", side_effect=replace),
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.export_scientific_ablation_operator_v1(
            path, request=request, base=base, root=tmp_path / "exports"
        )
    stage = tmp_path / "exports" / export._STORE / f".iris-export-{request.sha256}.tmp"
    assert _tree(stage) == {"foreign.txt": b"preserve this"} and (moved / "receipt.json").is_file()


@pytest.mark.parametrize("policy", ["quota", "reserve"])
def test_bounded_store_failure_withholds_export_and_preserves_source(
    completed: tuple[Path, Path], tmp_path: Path, policy: str
) -> None:
    path, base = completed
    original = _intent(path)
    payload = original.payload.model_dump(mode="python") | (
        {"maximum_storage_bytes": 1} if policy == "quota" else {"minimum_free_bytes": 16 * 1024**3}
    )
    value = ScientificAblationExportRequestPayloadV1.model_validate(payload)
    request = IrisScientificAblationExportRequestV1(
        sha256=domain_sha256(export._REQUEST, canonical_json_bytes(value.model_dump(mode="json"))),
        payload=value,
    )
    before = _tree(path)
    if policy == "reserve":
        usage = SimpleNamespace(total=1024, used=1023, free=1)
        with (
            patch("shutil.disk_usage", return_value=usage),
            patch.object(export, "create_scientific_ablation_command_run_v1") as factory,
            pytest.raises(export.IrisScientificAblationOperatorExportError),
        ):
            export.export_scientific_ablation_operator_v1(
                path, request=request, base=base, root=tmp_path / "exports"
            )
        assert factory.call_count == 0
    else:
        with pytest.raises(export.IrisScientificAblationOperatorExportError):
            export.export_scientific_ablation_operator_v1(
                path, request=request, base=base, root=tmp_path / "exports"
            )
    assert _tree(path) == before


WORKER = r"""import json, os, sys
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
stage, operator_path, intent_path, base, root = map(Path, sys.argv[1:6])
mode, ready_path = sys.argv[6], Path(sys.argv[7])
manifest_path, manifest_sha = Path(sys.argv[8]), sys.argv[9]
sys.path[:0] = [str(stage/'src'),str(stage)]
from tests._source_boundary import PackageBoundary, read_snapshot
expected = read_snapshot(manifest_path, manifest_sha)
finder = PackageBoundary(stage/'src', expected)
sys.meta_path.insert(0, finder)
import torch
from transformers import Qwen3ForCausalLM
from iris.application import scientific_ablation_operator_export as export
intent = export.read_scientific_ablation_export_request_v1(intent_path)
ready = dict(kind='iris-operator-export-cold-ready-v1',pid=os.getpid(),parent_pid=os.getppid(),
             argv=sys.argv,mode=mode,request_sha256=intent.sha256)
with ready_path.open('xb') as stream:
    stream.write(json.dumps(ready,sort_keys=True,separators=(',',':')).encode()+b'\n')
print('IRIS_EXPORT_READY '+json.dumps(ready,sort_keys=True),flush=True)
with (
    patch('socket.socket.connect',side_effect=AssertionError('No network')) as network,
    patch('iris.application.synthetic_scoring_v2.execute_python_job_v2',
          side_effect=AssertionError('Never execute candidate code')) as candidate,
    patch.object(Qwen3ForCausalLM,'forward',
                 side_effect=AssertionError('Export never forwards')) as forward,
):
    if mode == 'cut':
        rename = export.rename_directory_with_windows_retry
        def cut(source,target):
            rename(source,target)
            os._exit(83)
        with patch.object(export,'rename_directory_with_windows_retry',side_effect=cut):
            export.export_scientific_ablation_operator_v1(operator_path,request=intent,base=base,root=root)
        raise AssertionError('Abrupt publication cut did not execute')
    assert mode == 'resume'
    with (
        patch.object(export,'create_scientific_ablation_command_run_v1',
                     side_effect=AssertionError('No native cold retry')) as factory,
        patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('No graph')) as graph,
    ):
        receipt = export.export_scientific_ablation_operator_v1(
            operator_path,request=intent,base=base,root=root)
    assert factory.call_count == graph.call_count == forward.call_count == 0
    assert network.call_count == candidate.call_count == 0
for name,module in tuple(sys.modules.items()):
    if name == 'iris' or name.startswith('iris.'):
        finder.check(name,getattr(module,'__file__',None))
finder.verify()
print('IRIS_EXPORT_RESULT '+json.dumps(dict(kind='iris-operator-export-cold-result-v1',
    pid=os.getpid(),parent_pid=os.getppid(),request_sha256=intent.sha256,
    result_sha256=receipt.result.sha256,inference_path=str(receipt.inference_path),
    all_package_bytes_unchanged=True,package_files=len(expected),
    package_manifest_sha256=manifest_sha,data_only_retry=True),sort_keys=True),flush=True)
"""


@pytest.mark.parametrize("selection", ["current", "selected_best"])
def test_actual_abrupt83_after_export_publication_and_cold_data_retry(
    completed: tuple[Path, Path],
    tmp_path: Path,
    selection: Selection,
) -> None:
    path, base = completed
    request = _intent(path, selection)
    intent = tmp_path / "saved-export-intent.json"
    intent.write_bytes(canonical_json_file_bytes(request.model_dump(mode="json")))
    root = tmp_path / "exports"
    source_bytes, source_ids = _tree(path), _ids(path)
    stage = Path(__file__).resolve().parents[2]
    from tests._source_boundary import snapshot_package, write_snapshot

    package_pins = snapshot_package(stage / "src")
    package_manifest = tmp_path / "source-package-snapshot.json"
    package_sha = write_snapshot(package_manifest, package_pins)
    records = []
    export_bytes = None
    export_ids = None
    for mode, expected_exit in (("cut", 83), ("resume", 0)):
        ready = tmp_path / f"{mode}.ready.json"
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-u",
            "-c",
            WORKER,
            str(stage),
            str(path),
            str(intent),
            str(base),
            str(root),
            mode,
            str(ready),
            str(package_manifest),
            package_sha,
        ]
        child = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = child.communicate(timeout=180)
        (tmp_path / f"{mode}.stdout.log").write_bytes(stdout)
        (tmp_path / f"{mode}.stderr.log").write_bytes(stderr)
        assert child.returncode == expected_exit, stderr.decode(errors="replace")
        packet = json.loads(ready.read_bytes())
        assert packet["request_sha256"] == request.sha256
        assert (packet["pid"] == child.pid and packet["parent_pid"] == os.getpid()) or (
            packet["pid"] != child.pid and packet["parent_pid"] == child.pid
        )
        assert packet["argv"] == ["-c", *argv[6:]]
        lines = [
            line for line in stdout.decode().splitlines() if line.startswith("IRIS_EXPORT_READY ")
        ]
        assert len(lines) == 1 and json.loads(lines[0].removeprefix("IRIS_EXPORT_READY ")) == packet
        records.append(
            dict(
                mode=mode,
                actual_launch_pid=child.pid,
                observer_pid=os.getpid(),
                actual_child_pid=packet["pid"],
                actual_parent_pid=packet["parent_pid"],
                actual_exit_code=child.returncode,
                argv=argv,
                ready_path=str(ready),
                stdout_path=str(tmp_path / f"{mode}.stdout.log"),
                stderr_path=str(tmp_path / f"{mode}.stderr.log"),
            )
        )
        if mode == "cut":
            export_bytes, export_ids = _tree(root), _ids(root)
            assert any(name.endswith("receipt.json") for name in export_bytes)
        else:
            assert _tree(root) == export_bytes and _ids(root) == export_ids
            responses = [
                line
                for line in stdout.decode().splitlines()
                if line.startswith("IRIS_EXPORT_RESULT ")
            ]
            assert len(responses) == 1
            response = json.loads(responses[0].removeprefix("IRIS_EXPORT_RESULT "))
            assert (
                response["data_only_retry"] is True and response["request_sha256"] == request.sha256
            )
            assert response["all_package_bytes_unchanged"] is True
            assert response["package_files"] == len(package_pins)
            assert response["package_manifest_sha256"] == package_sha
    assert _tree(path) == source_bytes and _ids(path) == source_ids
    (tmp_path / "actual-operator-export-cold-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-operator-export-cold-witness-v1",
                operator_path=str(path),
                export_root=str(root),
                request=request.model_dump(mode="json"),
                records=records,
                source_native_ids=source_ids,
                exported_native_ids=export_ids,
                all_source_and_export_bytes_and_native_ids_stable=True,
                actual_cut_exit83_then_cold_exit0=True,
                selection=selection,
                installed_or_scientific_acceptance=False,
            )
        )
    )
