"""Actual mode-bound disk publication/read/request and ordinary ownership faults."""

import json
import os
import shutil
from dataclasses import fields
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import iris.utils.process_file_lock as locks
from iris.application import scientific_ablation_inference as projection
from iris.application import scientific_ablation_inference_io as io
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_inference import (
    _MAX_MANIFEST,
    IrisScientificAblationInferenceBundleV1,
    IrisScientificAblationInferenceError,
    build_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_inference_io import (
    read_scientific_ablation_inference_v1,
    write_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_request import build_scientific_ablation_request_v1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.process_file_lock import process_file_lock
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_inference_projection import _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration
type Trusted = tuple[IrisScientificAblationInferenceBundleV1, tuple[str, str, str]]


@pytest.fixture(scope="module")
def trusted() -> Trusted:
    source, _ = _source()
    bundle = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=_fresh
    )
    return bundle, (
        bundle.manifest.sha256,
        bundle.manifest.payload.source_checkpoint_sha256,
        source.manifest.sha256,
    )


def _write(trusted: Trusted, root: Path, **kwargs: Any) -> Path:
    bundle, anchors = trusted
    return write_scientific_ablation_inference_v1(
        bundle,
        root=root,
        expected_sha256=anchors[0],
        expected_source_checkpoint_sha256=anchors[1],
        expected_lifecycle_sha256=anchors[2],
        **kwargs,
    )


def _read(trusted: Trusted, path: Path) -> IrisScientificAblationInferenceBundleV1:
    _, anchors = trusted
    return read_scientific_ablation_inference_v1(
        path,
        expected_sha256=anchors[0],
        expected_source_checkpoint_sha256=anchors[1],
        expected_lifecycle_sha256=anchors[2],
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "explicit-output"
    root.mkdir()
    (root / "user.txt").write_bytes(b"keep unrelated user data\n")
    return root


def _assert_user(root: Path) -> None:
    assert (root / "user.txt").read_bytes() == b"keep unrelated user data\n"


def _signal(kind: str) -> BaseException:
    if kind == "interrupt":
        return KeyboardInterrupt("first")
    if kind == "exit":
        return SystemExit(73)
    return BaseExceptionGroup("first group", [SystemExit(74), ValueError("ordinary")])


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_real_disk_canonical_exact_retry_and_caller_context(
    trusted: Trusted, tmp_path: Path, default: str, inference: bool
) -> None:
    root = _root(tmp_path)
    caller = capture_runtime_state(torch.device("cpu"))
    with torch.device(default), torch.inference_mode(inference):
        target = _write(trusted, root)
        before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()}
        restored = _read(trusted, target)
        with patch.object(io, "_write_owned") as write:
            assert _write(trusted, root) == target
        write.assert_not_called()
        assert torch.get_default_device() == torch.device(default)
        assert torch.is_inference_mode_enabled() is inference
    assert restored == trusted[0] and restored.manifest is not trusted[0].manifest
    assert target.parent.name == "scientific-ablation-inference-v1"
    assert target.name == trusted[1][0]
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()} == before
    assert (target / "manifest.json").read_bytes() == canonical_json_file_bytes(
        restored.manifest.model_dump(mode="json")
    )
    assert {p.name for p in target.parent.iterdir()} == {target.name, ".iris-owner.lock"}
    assert (target.parent / ".iris-owner.lock").read_bytes() == b"1"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    _assert_user(root)


@pytest.mark.parametrize("anchor", (0, 1, 2))
def test_writer_bad_external_anchor_before_decode_or_filesystem(
    trusted: Trusted, tmp_path: Path, anchor: int
) -> None:
    bundle, good = trusted
    wrong = list(good)
    wrong[anchor] = "0" * 64
    with (
        patch.object(projection, "load_tensors") as decode,
        patch.object(os, "mkdir") as mkdir,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        write_scientific_ablation_inference_v1(
            bundle,
            root=tmp_path / "not-created",
            expected_sha256=wrong[0],
            expected_source_checkpoint_sha256=wrong[1],
            expected_lifecycle_sha256=wrong[2],
        )
    decode.assert_not_called()
    mkdir.assert_not_called()


@pytest.mark.parametrize("anchor", (0, 1, 2))
def test_reader_bad_external_anchor_before_weight_read_or_decode(
    trusted: Trusted, tmp_path: Path, anchor: int
) -> None:
    target = _write(trusted, _root(tmp_path))
    wrong = list(trusted[1])
    wrong[anchor] = "0" * 64
    with (
        patch.object(io, "_read_file", wraps=io._read_file) as read,
        patch.object(projection, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        read_scientific_ablation_inference_v1(
            target,
            expected_sha256=wrong[0],
            expected_source_checkpoint_sha256=wrong[1],
            expected_lifecycle_sha256=wrong[2],
        )
    assert all(call.args[0].name == "manifest.json" for call in read.call_args_list)
    decode.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    (
        ("maximum_storage_bytes", True),
        ("maximum_storage_bytes", 0),
        ("maximum_storage_bytes", 16 * 1024**3 + 1),
        ("minimum_free_bytes", -1),
        ("minimum_free_bytes", False),
    ),
)
def test_exact_budget_validation_before_mkdir(
    trusted: Trusted, tmp_path: Path, field: str, value: Any
) -> None:
    root = tmp_path / "not-created"
    with pytest.raises(IrisScientificAblationInferenceError):
        _write(trusted, root, **{field: value})
    assert not root.exists()


@pytest.mark.parametrize(
    "reason", ("quota", "reserve", "entry_bound", "file_bound", "foreign", "hardlink")
)
def test_store_budget_and_foreign_inventory_preserved_before_staging(
    trusted: Trusted, tmp_path: Path, reason: str
) -> None:
    root = _root(tmp_path)
    store = root / io._STORE
    store.mkdir()
    if reason == "foreign":
        (store / "keep.txt").write_bytes(b"foreign")
    if reason in {"entry_bound", "file_bound", "hardlink"}:
        staged = store / (io._STAGE_PREFIX + "a" * 32 + ".tmp")
        staged.mkdir()
        (staged / "manifest.json").write_bytes(b"old orphan")
        if reason == "hardlink":
            os.link(staged / "manifest.json", root / "alias")
    with patch.object(io, "_write_owned") as write:
        if reason == "quota":
            with pytest.raises(IrisScientificAblationInferenceError):
                _write(trusted, root, maximum_storage_bytes=1)
        elif reason == "reserve":
            with (
                patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=0)),
                pytest.raises(IrisScientificAblationInferenceError),
            ):
                _write(trusted, root)
        elif reason == "entry_bound":
            with (
                patch.object(io, "_MAX_STORE_ENTRIES", 0),
                pytest.raises(IrisScientificAblationInferenceError),
            ):
                _write(trusted, root)
        elif reason == "file_bound":
            with (
                patch.object(io, "_MAX_STORE_FILES", 0),
                pytest.raises(IrisScientificAblationInferenceError),
            ):
                _write(trusted, root)
        else:
            with pytest.raises(IrisScientificAblationInferenceError):
                _write(trusted, root)
    write.assert_not_called()
    _assert_user(root)
    if reason in {"entry_bound", "file_bound", "hardlink"}:
        assert (staged / "manifest.json").read_bytes() == b"old orphan"
    if reason == "foreign":
        assert (store / "keep.txt").read_bytes() == b"foreign"


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "missing",
        "noncanonical",
        "metadata_extra",
        "oversize_manifest",
        "state_corrupt",
        "state_truncated",
        "hardlink",
    ),
)
def test_disk_malformed_candidates_are_read_only_rejections(
    trusted: Trusted, tmp_path: Path, fault: str
) -> None:
    root = _root(tmp_path)
    target = _write(trusted, root)
    manifest, weights = target / "manifest.json", target / "state.safetensors"
    if fault == "extra":
        (target / "foreign.txt").write_bytes(b"keep foreign")
    elif fault == "missing":
        weights.rename(root / "retained-missing-state")
    elif fault == "noncanonical":
        manifest.write_bytes(b" " + manifest.read_bytes())
    elif fault == "metadata_extra":
        value = json.loads(manifest.read_bytes())
        value["ghost"] = 1
        manifest.write_bytes(canonical_json_file_bytes(value))
    elif fault == "oversize_manifest":
        manifest.write_bytes(b"x" * (_MAX_MANIFEST + 1))
    elif fault == "state_corrupt":
        value = bytearray(weights.read_bytes())
        value[-1] ^= 1
        weights.write_bytes(value)
    elif fault == "state_truncated":
        weights.write_bytes(weights.read_bytes()[:-1])
    else:
        os.link(weights, root / "hardlink-state")
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()}
    with pytest.raises(IrisScientificAblationInferenceError):
        _read(trusted, target)
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()} == before
    _assert_user(root)


@pytest.mark.parametrize("when", ("before_open", "between_files"))
def test_equal_byte_file_replacement_detected_without_tensor_decode_or_cleanup(
    trusted: Trusted, tmp_path: Path, when: str
) -> None:
    root = _root(tmp_path)
    target = _write(trusted, root)
    weights = target / "state.safetensors"
    original = weights.read_bytes()
    old_identity = weights.stat().st_ino
    native_open, native_read = os.open, io._read_file
    replaced = False

    def replace_file() -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            weights.rename(root / "original-state")
            weights.write_bytes(original)

    def open_file(path: Any, *args: Any, **kwargs: Any) -> int:
        if Path(path) == weights:
            replace_file()
        return native_open(path, *args, **kwargs)

    def read_file(path: Path, maximum: int) -> bytes:
        result = native_read(path, maximum)
        if path.name == "manifest.json":
            replace_file()
        return result

    with pytest.raises(IrisScientificAblationInferenceError):
        if when == "before_open":
            with (
                patch.object(os, "open", side_effect=open_file),
                patch.object(projection, "load_tensors") as decode,
            ):
                _read(trusted, target)
            decode.assert_not_called()
        else:
            with patch.object(io, "_read_file", side_effect=read_file):
                _read(trusted, target)
    assert replaced and weights.stat().st_ino != old_identity
    assert weights.read_bytes() == original and (root / "original-state").read_bytes() == original
    _assert_user(root)


def test_uncertain_rename_actual_success_and_exact_retry_never_rewrite(
    trusted: Trusted, tmp_path: Path
) -> None:
    root = _root(tmp_path)

    def renamed(source: Path, target: Path) -> None:
        os.rename(source, target)
        raise OSError("successful move not acknowledged")

    with patch.object(io, "rename_directory_with_windows_retry", side_effect=renamed):
        target = _write(trusted, root)
    assert _read(trusted, target) == trusted[0]
    with patch.object(io, "_write_owned") as write:
        assert _write(trusted, root) == target
    write.assert_not_called()
    assert {p.name for p in target.parent.iterdir()} == {target.name, io._LOCK}
    _assert_user(root)


@pytest.mark.parametrize("same", (False, True))
def test_concurrent_winner_requires_verified_exact_bytes_and_preserves_both_owners(
    trusted: Trusted, tmp_path: Path, same: bool
) -> None:
    root = _root(tmp_path)

    def winner(stage: Path, target: Path) -> None:
        target.mkdir()
        for name in io._FILES:
            (target / name).write_bytes((stage / name).read_bytes())
        if not same:
            (target / "state.safetensors").write_bytes(b"foreign winner")
        raise FileExistsError("winner appeared")

    with patch.object(io, "rename_directory_with_windows_retry", side_effect=winner):
        if same:
            target = _write(trusted, root)
            assert _read(trusted, target) == trusted[0]
        else:
            with pytest.raises(IrisScientificAblationInferenceError):
                _write(trusted, root)
            target = root / io._STORE / trusted[1][0]
            assert (target / "state.safetensors").read_bytes() == b"foreign winner"
    assert {p.name for p in target.parent.iterdir()} == {target.name, io._LOCK}
    _assert_user(root)


def test_mkdir_collision_never_grants_cleanup_ownership(trusted: Trusted, tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = root / io._STORE
    store.mkdir()
    colliding = store / (io._STAGE_PREFIX + "a" * 32 + ".tmp")
    colliding.mkdir()
    (colliding / "manifest.json").write_bytes(b"existing caller file")
    with (
        patch.object(io, "uuid4", return_value=SimpleNamespace(hex="a" * 32)),
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _write(trusted, root)
    assert (colliding / "manifest.json").read_bytes() == b"existing caller file"
    _assert_user(root)


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
@pytest.mark.parametrize(
    "fault", ("foreign_entry", "equal_replaced_file", "replaced_directory", "cleanup_signal")
)
def test_actual_primary_write_signal_retained_with_foreign_or_uncertain_cleanup(
    trusted: Trusted, tmp_path: Path, kind: str, fault: str
) -> None:
    root = _root(tmp_path)
    primary = _signal(kind)
    native = io._write_owned
    stages: list[io._Stage] = []

    def interrupted(stage: io._Stage, name: str, content: bytes, claim: io._StoreClaim) -> None:
        native(stage, name, content, claim)
        stages.append(stage)
        if fault == "foreign_entry":
            (stage.path / "foreign.bin").write_bytes(b"keep foreign")
        elif fault == "equal_replaced_file":
            (stage.path / name).rename(root / "original-manifest")
            (stage.path / name).write_bytes(content)
        elif fault == "replaced_directory":
            stage.path.rename(root / "original-stage")
            stage.path.mkdir()
            (stage.path / name).write_bytes(b"keep replacement directory")
        raise primary

    with patch.object(io, "_write_owned", side_effect=interrupted):
        if fault == "cleanup_signal":
            with (
                patch.object(io, "_cleanup", side_effect=SystemExit(99)),
                pytest.raises(BaseException) as caught,
            ):
                _write(trusted, root)
        else:
            with pytest.raises(BaseException) as caught:
                _write(trusted, root)
    assert caught.value is primary and primary.__cause__ is not None
    assert len(stages) == 1 and stages[0].path.is_dir()
    assert (stages[0].path / "manifest.json").exists()
    if fault == "foreign_entry":
        assert (stages[0].path / "foreign.bin").read_bytes() == b"keep foreign"
    elif fault == "equal_replaced_file":
        assert (root / "original-manifest").stat().st_ino == stages[0].files["manifest.json"][1]
        assert (stages[0].path / "manifest.json").stat().st_ino != stages[0].files["manifest.json"][
            1
        ]
    elif fault == "replaced_directory":
        assert (root / "original-stage" / "manifest.json").exists()
        assert (stages[0].path / "manifest.json").read_bytes() == b"keep replacement directory"
    _assert_user(root)


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
def test_partial_file_write_first_signal_survives_actual_descriptor_close_failure(
    trusted: Trusted, tmp_path: Path, kind: str
) -> None:
    root = _root(tmp_path)
    primary = _signal(kind)
    native_close = os.close
    interrupted: list[int] = []

    def write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:17])
        interrupted.append(descriptor)
        raise primary

    def close(descriptor: int) -> None:
        native_close(descriptor)
        if descriptor in interrupted:
            interrupted.remove(descriptor)
            raise OSError("actual close returned but then failed")

    with (
        patch.object(io, "_write_bytes", side_effect=write),
        patch.object(os, "close", side_effect=close),
        pytest.raises(BaseException) as caught,
    ):
        _write(trusted, root)
    assert caught.value is primary and isinstance(primary.__cause__, OSError)
    assert {p.name for p in (root / io._STORE).iterdir()} == {io._LOCK}
    _assert_user(root)


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
def test_first_signal_not_overwritten_by_lock_release_failure(
    trusted: Trusted, tmp_path: Path, kind: str
) -> None:
    root = _root(tmp_path)
    primary = _signal(kind)
    native = locks._lock

    def lock(descriptor: int, *, acquire: bool) -> None:
        native(descriptor, acquire=acquire)
        if not acquire:
            raise OSError("release acknowledgement failed")

    with (
        patch.object(io, "_write_bytes", side_effect=primary),
        patch.object(locks, "_lock", side_effect=lock),
        pytest.raises(BaseException) as caught,
    ):
        _write(trusted, root)
    assert caught.value is primary and isinstance(primary.__cause__, BaseExceptionGroup)
    assert {p.name for p in (root / io._STORE).iterdir()} == {io._LOCK}
    _assert_user(root)


def test_real_store_lock_denies_second_writer_without_touching_artifacts(
    trusted: Trusted, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    target = _write(trusted, root)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with (
        process_file_lock(target.parent / io._LOCK),
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _write(trusted, root)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    _assert_user(root)


@pytest.mark.parametrize("platform", ("nt", "posix"))
def test_exact_named_opened_timestamp_semantics_without_tolerance(platform: str) -> None:
    named: Any = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_size=10,
        st_mtime_ns=15,
        st_ctime_ns=11,
        st_birthtime_ns=11,
        st_nlink=1,
    )
    opened: Any = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_size=10,
        st_mtime_ns=15,
        st_ctime_ns=19,
        st_birthtime_ns=11,
        st_nlink=1,
    )
    with patch.object(os, "name", platform):
        if platform == "nt":
            assert io._signature(named) == io._signature(opened)
        else:
            assert io._signature(named) != io._signature(opened)
        opened.st_mtime_ns += 1
        assert io._signature(named) != io._signature(opened)


def test_exact_windows_birthtime_unavailable_fails_closed() -> None:
    named: Any = SimpleNamespace(
        st_dev=1, st_ino=2, st_size=10, st_mtime_ns=15, st_ctime_ns=11, st_nlink=1
    )
    with patch.object(os, "name", "nt"), pytest.raises(OSError, match="birthtime"):
        io._signature(named)


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_persisted_earlier_best_all12_c3_teacher_public_request_parity(
    tmp_path: Path, routing: Any, goal: str, broadcast: str
) -> None:
    source, controller = _source(routing, goal=goal, broadcast=broadcast)
    current = build_scientific_ablation_checkpoint_v1(controller._run)
    restored: list[IrisScientificAblationRunV1] = []
    factory = partial(_fresh, routing, goal=goal, broadcast=broadcast)

    def make() -> IrisScientificAblationRunV1:
        run = factory()
        restored.append(run)
        return run

    exported = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=make
    )
    anchors = (
        exported.manifest.sha256,
        exported.manifest.payload.source_checkpoint_sha256,
        source.manifest.sha256,
    )
    model = restored[0]._optimizer._model
    model.eval()
    row = _rows()[0]
    logits: list[torch.Tensor] = []
    native_teacher = model.teacher.backbone.forward_embeddings

    def teacher(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        result = native_teacher(*args, **kwargs)
        first_target = int(torch.nonzero(kwargs["labels"][0] != -100)[0, 0])
        logits.append(result.logits[:, first_target - 1, :].detach().clone())
        return result

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=teacher):
        expected = model.evaluation_forward(row, terminal_token_id=1)
    backbone = factory()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    root = _root(tmp_path)
    target = _write((exported, anchors), root)
    read = _read((exported, anchors), target)
    request = build_scientific_ablation_request_v1(
        read,
        expected_sha256=anchors[0],
        expected_source_checkpoint_sha256=anchors[1],
        expected_lifecycle_sha256=anchors[2],
        backbone=backbone,
        stop_token_ids=tuple(range(backbone.vocabulary_size)),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )
    native_request = backbone.forward_embeddings
    actual_logits: list[torch.Tensor] = []

    def cached(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        result = native_request(*args, **kwargs)
        actual_logits.append(result.logits[:, -1, :].detach().clone())
        return result

    for _ in range(2):
        with patch.object(backbone, "forward_embeddings", side_effect=cached):
            actual = request.execute(row.prompt, strategy="greedy")
        for observed, reference in zip(
            actual.generation.workspace_trace, expected.trace.teacher.workspace_trace, strict=True
        ):
            for field in fields(observed):
                assert torch.equal(getattr(observed, field.name), getattr(reference, field.name))
        torch.testing.assert_close(actual_logits[-1], logits[0], rtol=1e-5, atol=1e-6)
        assert actual.generation.token_ids.tolist() == [logits[0].argmax(-1).tolist()]
        assert actual.receipt.artifact_sha256 == anchors[0]
        assert actual.receipt.source_checkpoint_sha256 == anchors[1]
        assert actual.receipt.selection == "selected_best" and actual.receipt.completed_steps == 1
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert len(logits) == 1 and len(actual_logits) == 2
    assert build_scientific_ablation_checkpoint_v1(controller._run) == current
    _assert_user(root)
