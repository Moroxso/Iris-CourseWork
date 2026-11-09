"""Genuine captured/scored reports and adversarial immutable local storage checks."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import iris.utils.process_file_lock as locks
from iris.application import scientific_ablation_evaluation_io as io
from iris.application import scientific_ablation_evaluation_v2 as evaluation
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_inference_io import _finish_descriptor, _read_file
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_evaluation_v2 import _evaluate, _make
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
Error = evaluation.IrisScientificAblationEvaluationError


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> IrisScientificAblationEvaluationV2:
    return _evaluate(_make(tmp_path_factory.mktemp("actual-storage-scored-source")))


def _write(
    report: IrisScientificAblationEvaluationV2,
    root: Path,
    *,
    maximum: int = 512 * 1024**2,
    reserve: int = 0,
) -> Path:
    return io.write_scientific_ablation_evaluation_v2(
        report,
        root=root,
        expected_sha256=report.sha256,
        expected_plan_sha256=report.payload.plan.sha256,
        expected_capture_sha256=report.payload.capture.sha256,
        maximum_storage_bytes=maximum,
        minimum_free_bytes=reserve,
    )


def _read(
    report: IrisScientificAblationEvaluationV2, path: Path, *, maximum: int = evaluation._MAX_REPORT
) -> IrisScientificAblationEvaluationV2:
    return io.read_scientific_ablation_evaluation_v2(
        path,
        expected_sha256=report.sha256,
        expected_plan_sha256=report.payload.plan.sha256,
        expected_capture_sha256=report.payload.capture.sha256,
        maximum_report_bytes=maximum,
    )


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def _stage(root: Path) -> Path:
    path = root / io._STORE / (".iris-evaluation-" + "a" * 32 + ".tmp")
    path.mkdir(parents=True)
    return path


def _signal(kind: str) -> BaseException:
    if kind == "interrupt":
        return KeyboardInterrupt("first storage signal")
    if kind == "exit":
        return SystemExit(79)
    return BaseExceptionGroup("first signal group", [SystemExit(81), ValueError("ordinary")])


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_actual_scored_roundtrip_exact_retry_no_models_and_caller_state(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, default: str, inference: bool
) -> None:
    (tmp_path / "unrelated-user.txt").write_bytes(b"keep unrelated data")
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        torch.device(default),
        torch.inference_mode(inference),
        patch.object(evaluation, "evaluate_scientific_ablation_capture_v2") as score,
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as execute,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        target = _write(report, tmp_path)
        before = _snapshot(tmp_path)
        with patch.object(io, "_write") as write:
            assert _write(report, tmp_path) == target
        write.assert_not_called()
        restored = _read(report, target)
        assert _snapshot(tmp_path) == before
        assert torch.get_default_device() == torch.device(default)
        assert torch.is_inference_mode_enabled() is inference
    score.assert_not_called()
    execute.assert_not_called()
    model.assert_not_called()
    tokenizer.assert_not_called()
    assert restored == report and restored is not report
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert (target / io._FILE).read_bytes() == canonical_json_file_bytes(
        report.model_dump(mode="json")
    )
    assert {path.name for path in target.parent.iterdir()} == {report.sha256, io._LOCK}
    assert (target.parent / io._LOCK).read_bytes() == b"1"


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_earlier_best_capture_score_disk_binding(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    case = _make(
        tmp_path / "source", routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:]
    )
    report = _evaluate(case)
    target = _write(report, tmp_path / "reports")
    restored = _read(report, target)
    assert restored == report
    assert restored.payload.plan == case.capture.plan
    assert restored.payload.capture == case.capture.complete
    assert restored.payload.plan.payload.source == case.capture.plan.payload.source
    source = restored.payload.plan.payload.source.payload
    assert source.selected_step == 1 and source.terminal_step == 3
    assert restored.payload.plan.payload.cycles == 3
    assert (source.ablation.routing, source.ablation.goal, source.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert len(restored.payload.rows) == 1 and restored.payload.overall.target_tokens > 0
    assert restored.payload.scientific_launch_status == "not_assessed"


@pytest.mark.parametrize("anchor", ("artifact", "plan", "capture"))
def test_wrong_trusted_writer_anchor_before_any_write(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, anchor: str
) -> None:
    with patch.object(os, "mkdir") as mkdir, pytest.raises(Error):
        io.write_scientific_ablation_evaluation_v2(
            report,
            root=tmp_path / "absent",
            expected_sha256="0" * 64 if anchor == "artifact" else report.sha256,
            expected_plan_sha256="0" * 64 if anchor == "plan" else report.payload.plan.sha256,
            expected_capture_sha256="0" * 64
            if anchor == "capture"
            else report.payload.capture.sha256,
        )
    mkdir.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("anchor", ("artifact", "plan", "capture"))
def test_malformed_reader_anchor_before_file_read(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, anchor: str
) -> None:
    with patch.object(io, "_read_file") as read, pytest.raises(Error):
        io.read_scientific_ablation_evaluation_v2(
            tmp_path / io._STORE / report.sha256,
            expected_sha256="../bad" if anchor == "artifact" else report.sha256,
            expected_plan_sha256="BAD" if anchor == "plan" else report.payload.plan.sha256,
            expected_capture_sha256="" if anchor == "capture" else report.payload.capture.sha256,
        )
    read.assert_not_called()


@pytest.mark.parametrize("variant", ("hidden", "domain", "counts"))
def test_invalid_input_tree_never_creates_store(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    candidate = report.model_copy(deep=True)
    if variant == "hidden":
        candidate.payload.rows[0].__dict__["raw_candidate"] = "must not be erased"
    elif variant == "domain":
        candidate.__dict__["sha256"] = "0" * 64
    else:
        candidate.payload.overall.__dict__["examples"] += 1
    with pytest.raises(Error):
        _write(candidate, tmp_path / "absent")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "variant", ("zero", "bool", "float", "too_large", "negative_reserve", "bool_reserve")
)
def test_exact_quota_validation_before_mkdir(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    maximum = {"zero": 0, "bool": True, "float": 1.5, "too_large": io._MAX_STORAGE + 1}.get(
        variant, 512 * 1024**2
    )
    reserve = (
        -1
        if variant == "negative_reserve"
        else variant == "bool_reserve"
        if variant == "bool_reserve"
        else 0
    )
    with pytest.raises(Error), patch.object(os, "mkdir") as mkdir:
        _write(report, tmp_path / "absent", maximum=maximum, reserve=reserve)  # type: ignore[arg-type]
    mkdir.assert_not_called()


@pytest.mark.parametrize(
    "variant",
    (
        "no_lf",
        "pretty",
        "duplicate",
        "nonfinite",
        "corrupt",
        "foreign",
        "nested",
        "hardlink",
        "wrong_store",
        "wrong_directory",
        "read_budget",
        "plan",
        "capture",
    ),
)
def test_reader_rejects_untrusted_or_noncanonical_disk_without_mutation(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    target = _write(report, tmp_path)
    file = target / io._FILE
    data = file.read_bytes()
    if variant == "no_lf":
        file.write_bytes(data[:-1])
    elif variant == "pretty":
        file.write_bytes(json.dumps(json.loads(data), indent=2).encode() + b"\n")
    elif variant == "duplicate":
        file.write_bytes(b'{"sha256":"' + report.sha256.encode() + b'",' + data[1:])
    elif variant == "nonfinite":
        file.write_bytes(data.replace(b'"target_nll_sum":', b'"target_nll_sum":NaN,"old":', 1))
    elif variant == "corrupt":
        file.write_bytes(data.replace(report.sha256.encode(), b"0" * 64, 1))
    elif variant == "foreign":
        (target / "foreign.txt").write_bytes(b"keep")
    elif variant == "nested":
        (target / "nested").mkdir()
    elif variant == "hardlink":
        os.link(file, tmp_path / "linked-report")
    elif variant == "wrong_store":
        target.parent.rename(tmp_path / "scientific-evaluation-v2")
        target = tmp_path / "scientific-evaluation-v2" / report.sha256
    elif variant == "wrong_directory":
        target.rename(target.parent / ("0" * 64))
        target = target.parent / ("0" * 64)
    before = _snapshot(tmp_path)
    with pytest.raises(Error):
        io.read_scientific_ablation_evaluation_v2(
            target,
            expected_sha256=report.sha256,
            expected_plan_sha256="0" * 64 if variant == "plan" else report.payload.plan.sha256,
            expected_capture_sha256="0" * 64
            if variant == "capture"
            else report.payload.capture.sha256,
            maximum_report_bytes=len(data) - 1
            if variant == "read_budget"
            else evaluation._MAX_REPORT,
        )
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "variant",
    (
        "foreign",
        "deep",
        "stage_member",
        "quota",
        "reserve",
        "directory_count",
        "file_count",
        "hardlink",
    ),
)
def test_store_rejections_preserve_old_stages_and_user_files_before_new_staging(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    stage = _stage(tmp_path)
    file = stage / io._FILE
    file.write_bytes(b"partial old report")
    if variant == "foreign":
        (stage.parent / "foreign.txt").write_bytes(b"keep")
    elif variant == "deep":
        (stage / "nested").mkdir()
    elif variant == "stage_member":
        (stage / "foreign.txt").write_bytes(b"keep")
    elif variant == "hardlink":
        os.link(file, tmp_path / "other-link")
    maximum = len(evaluation._checked(report)) + 1 if variant == "quota" else 512 * 1024**2
    with (
        patch.object(io, "_MAX_DIRECTORIES", 0 if variant == "directory_count" else 4096),
        patch.object(io, "_MAX_FILES", 1 if variant == "file_count" else 4097),
        patch.object(
            shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=0 if variant == "reserve" else 16 * 1024**3),
        ),
        patch.object(io, "_write") as write,
        pytest.raises(Error),
    ):
        _write(report, tmp_path, maximum=maximum)
    write.assert_not_called()
    assert stage.is_dir() and file.read_bytes() == b"partial old report"
    assert {path.name for path in stage.parent.iterdir()} <= {stage.name, io._LOCK, "foreign.txt"}


def test_empty_and_partial_old_stages_preserved_on_success(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    first = _stage(tmp_path)
    second = first.parent / (".iris-evaluation-" + "b" * 32 + ".tmp")
    second.mkdir()
    (second / io._FILE).write_bytes(b"unacknowledged")
    before = _snapshot(tmp_path)
    target = _write(report, tmp_path)
    assert _read(report, target) == report
    after = _snapshot(tmp_path)
    assert all(after[name] == value for name, value in before.items())


@pytest.mark.parametrize("variant", ("bytes", "foreign"))
def test_existing_immutable_target_conflict_is_not_overwritten(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    target = _write(report, tmp_path)
    if variant == "bytes":
        (target / io._FILE).write_bytes(b"corrupt keep")
    else:
        (target / "foreign").write_bytes(b"keep")
    before = _snapshot(tmp_path)
    with patch.object(io, "_write") as write, pytest.raises(Error):
        _write(report, tmp_path)
    write.assert_not_called()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("winner", ("moved", "identical", "conflicting"))
def test_ordinary_uncertain_rename_requires_exact_target_and_preserves_stages(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, winner: str
) -> None:
    def rename(source: Path, target: Path) -> None:
        if winner == "moved":
            os.rename(source, target)
        else:
            target.mkdir()
            (target / io._FILE).write_bytes(
                (source / io._FILE).read_bytes() if winner == "identical" else b"foreign keep"
            )
        raise OSError("lost rename acknowledgement")

    with patch.object(io, "rename_directory_with_windows_retry", side_effect=rename):
        if winner == "conflicting":
            with pytest.raises(Error):
                _write(report, tmp_path)
        else:
            assert _read(report, _write(report, tmp_path)) == report
    store = tmp_path / io._STORE
    stages = list(store.glob(".iris-evaluation-*.tmp"))
    assert len(stages) == (0 if winner == "moved" else 1)
    if stages:
        assert (stages[0] / io._FILE).read_bytes() == evaluation._checked(report)
    if winner == "conflicting":
        assert (store / report.sha256 / io._FILE).read_bytes() == b"foreign keep"
    else:
        with patch.object(io, "_write") as write:
            assert _write(report, tmp_path) == store / report.sha256
        write.assert_not_called()


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
@pytest.mark.parametrize("phase", ("write", "rename", "after_rename"))
def test_first_signal_never_becomes_publication_success(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, kind: str, phase: str
) -> None:
    primary = _signal(kind)

    def rename(source: Path, target: Path) -> None:
        if phase == "after_rename":
            os.rename(source, target)
        raise primary

    with (
        patch.object(io, "_write_bytes", side_effect=primary)
        if phase == "write"
        else patch.object(io, "rename_directory_with_windows_retry", side_effect=rename),
        pytest.raises(BaseException) as caught,
    ):
        _write(report, tmp_path)
    assert caught.value is primary
    store = tmp_path / io._STORE
    if phase == "after_rename":
        assert _read(report, store / report.sha256) == report
        assert _write(report, tmp_path) == store / report.sha256  # Explicit later exact retry.
    else:
        assert not (store / report.sha256).exists()
        stages = list(store.glob(".iris-evaluation-*.tmp"))
        assert len(stages) == 1 and (stages[0] / io._FILE).is_file()


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
def test_first_signal_survives_actual_lock_release_failure(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, kind: str
) -> None:
    primary = _signal(kind)
    native = locks._lock

    def lock(descriptor: int, *, acquire: bool) -> None:
        native(descriptor, acquire=acquire)
        if not acquire:
            raise OSError("actual release acknowledgement lost")

    with (
        patch.object(io, "_write_bytes", side_effect=primary),
        patch.object(locks, "_lock", side_effect=lock),
        pytest.raises(BaseException) as caught,
    ):
        _write(report, tmp_path)
    assert caught.value is primary and isinstance(primary.__cause__, BaseExceptionGroup)


def test_actual_persistent_lock_denies_second_writer_without_staging(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    target = _write(report, tmp_path)
    before = _snapshot(tmp_path)
    with process_file_lock(target.parent / io._LOCK), pytest.raises(Error):
        _write(report, tmp_path)
    assert _snapshot(tmp_path) == before


def test_real_fresh_process_observes_nonblocking_lock_and_owner_release(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    target = _write(report, tmp_path)
    # Authored diagnostic, never candidate/generated Python. No filesystem writes.
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from iris.utils.process_file_lock import process_file_lock; "
        "\ntry:\n with process_file_lock(Path(sys.argv[2])): print('acquired')\n"
        "except OSError: print('busy'); sys.exit(23)\n"
    )
    args = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        script,
        str(Path(__file__).parents[2] / "src"),
        str(target.parent / io._LOCK),
    ]
    before = _snapshot(tmp_path)
    with process_file_lock(target.parent / io._LOCK):
        busy = subprocess.run(args, check=False, capture_output=True, timeout=30)
    acquired = subprocess.run(args, check=False, capture_output=True, timeout=30)
    assert busy.returncode == 23 and busy.stdout.strip() == b"busy"
    assert acquired.returncode == 0 and acquired.stdout.strip() == b"acquired"
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("variant", ("stage", "file"))
def test_actual_replacement_during_write_preserves_foreign_data_and_withholds_report(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, variant: str
) -> None:
    original = io._write
    replaced = tmp_path / "preserved-original"

    def write(owned: io._Stage, content: bytes, claim: io._Claim) -> None:
        original(owned, content, claim)
        store = tmp_path / io._STORE
        stage = next(store.glob(".iris-evaluation-*.tmp"))
        if variant == "stage":
            stage.rename(replaced)
            stage.mkdir()
            (stage / io._FILE).write_bytes(b"keep replacement")
        else:
            (stage / io._FILE).rename(replaced)
            (stage / io._FILE).write_bytes(content)  # Even equal-byte new inode is rejected.

    with patch.object(io, "_write", side_effect=write), pytest.raises(Error):
        _write(report, tmp_path)
    assert replaced.exists()
    assert not (tmp_path / io._STORE / report.sha256).exists()


def test_actual_root_replacement_between_identity_snapshot_and_lock_acquisition(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    preserved = tmp_path / "preserved-original-store"

    @contextmanager
    def replaced_root(path: Path) -> Iterator[ProcessFileLock]:
        # Real Windows replacement BEFORE the handle opens; no waived lock guard.
        store = path.parent
        store.rename(preserved)
        store.mkdir()
        (store / "foreign.txt").write_bytes(b"keep replacement")
        with process_file_lock(path) as lock:
            lock.check()  # Current named/opened lock really agrees; OLD root does not.
            yield lock

    with (
        patch.object(io, "process_file_lock", side_effect=replaced_root),
        patch.object(io, "_write") as write,
        pytest.raises(Error),
    ):
        _write(report, tmp_path)
    write.assert_not_called()
    assert preserved.is_dir()
    assert (tmp_path / io._STORE / "foreign.txt").read_bytes() == b"keep replacement"
    assert not (tmp_path / io._STORE / report.sha256).exists()


@pytest.mark.parametrize("phase", ("after_close", "after_stage_read"))
def test_equal_byte_file_replacement_at_close_or_publication_boundary_fails_closed(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, phase: str
) -> None:
    before_read = _read_file
    original_finish = _finish_descriptor
    preserved = tmp_path / "preserved-original-report"
    replaced: list[Path] = []

    def replace() -> None:
        stage = next((tmp_path / io._STORE).glob(".iris-evaluation-*.tmp"))
        path = stage / io._FILE
        data = path.read_bytes()
        path.rename(preserved)
        path.write_bytes(data)
        replaced.append(path)

    def finish(primary: BaseException | None, close: Callable[[], None]) -> None:
        # Callback executed only after the real descriptor has closed.
        original_finish(primary, close)
        replace()

    reads = 0

    def read(path: Path, maximum: int) -> bytes:
        nonlocal reads
        content = before_read(path, maximum)
        reads += 1
        if reads == 2:  # _write readback, then _publish's canonical staged read.
            replace()
        return content

    with (
        patch.object(io, "_finish_descriptor", side_effect=finish)
        if phase == "after_close"
        else patch.object(io, "_read_file", side_effect=read),
        patch.object(io, "rename_directory_with_windows_retry") as rename,
        pytest.raises(Error),
    ):
        _write(report, tmp_path)
    rename.assert_not_called()
    assert len(replaced) == 1
    assert replaced[0].read_bytes() == preserved.read_bytes() == evaluation._checked(report)
    assert replaced[0].stat().st_ino != preserved.stat().st_ino


def test_equal_byte_replacement_during_reader_is_rejected(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    target = _write(report, tmp_path)
    original = _read_file
    file = target / io._FILE

    def read(path: Path, maximum: int) -> bytes:
        content = original(path, maximum)
        path.rename(tmp_path / "preserved-old-report")
        path.write_bytes(content)
        return content

    with patch.object(io, "_read_file", side_effect=read), pytest.raises(Error):
        _read(report, target)
    assert file.read_bytes() == (tmp_path / "preserved-old-report").read_bytes()


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
def test_first_write_signal_survives_actual_descriptor_close_failure(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path, kind: str
) -> None:
    primary = _signal(kind)
    descriptors: list[int] = []
    native = os.close

    def write(descriptor: int, content: bytes) -> None:
        descriptors.append(descriptor)
        os.write(descriptor, content[:19])
        raise primary

    def close(descriptor: int) -> None:
        native(descriptor)
        if descriptors and descriptor == descriptors[0]:
            raise OSError("actual file close acknowledgement lost")

    with (
        patch.object(io, "_write_bytes", side_effect=write),
        patch.object(os, "close", side_effect=close),
        pytest.raises(BaseException) as caught,
    ):
        _write(report, tmp_path)
    assert caught.value is primary
    stages = list((tmp_path / io._STORE).glob(".iris-evaluation-*.tmp"))
    assert len(stages) == 1
    assert (stages[0] / io._FILE).read_bytes() == evaluation._checked(report)[:19]


def test_free_reserve_drift_after_write_withholds_publication_and_preserves_stage(
    report: IrisScientificAblationEvaluationV2, tmp_path: Path
) -> None:
    with (
        patch.object(
            shutil,
            "disk_usage",
            side_effect=[SimpleNamespace(free=16 * 1024**3), SimpleNamespace(free=0)],
        ),
        patch.object(io, "_publish") as publish,
        pytest.raises(Error),
    ):
        _write(report, tmp_path, reserve=1024)
    publish.assert_not_called()
    stages = list((tmp_path / io._STORE).glob(".iris-evaluation-*.tmp"))
    assert len(stages) == 1 and (stages[0] / io._FILE).read_bytes() == evaluation._checked(report)
