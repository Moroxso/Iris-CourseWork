"""Genuine tiny-Qwen scientific phase persistence and exact controller replay."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application import scientific_execution as execution
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _verify,
    build_scientific_checkpoint_v1,
)
from iris.application.scientific_lifecycle import (
    IrisScientificLifecycleBundleV1,
    IrisScientificLifecycleV1,
    _publish,
    read_scientific_lifecycle_checkpoint_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_checkpoint_v1 import IrisScientificCheckpointManifestV1
from iris.artifacts.scientific_execution_v1 import ScientificExecutionCheckpointV1
from iris.artifacts.scientific_lifecycle_v1 import IrisScientificLifecycleCheckpointManifestV1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _run, _stochastic
from tests.integration.test_scientific_lifecycle import _lifecycle, _short_lifecycle, _validation

pytestmark = pytest.mark.integration


@pytest.fixture
def randomness() -> Iterator[ExitStack]:
    with ExitStack() as stack:
        yield stack


@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_phase_tail_resume_matches_uninterrupted_training(
    tmp_path: Path, cut: int, randomness: ExitStack, completed: Path
) -> None:
    ambient = capture_runtime_state(torch.device("cpu"))
    lifecycle = _lifecycle()
    randomness.enter_context(_stochastic(lifecycle._run))
    source = execution.create_scientific_execution_v1(lifecycle, root=tmp_path / "source")
    for _ in range(cut):
        source.advance()
    prefix = execution.read_scientific_execution_v1(source.path)
    fresh = _lifecycle()
    randomness.enter_context(_stochastic(fresh._run))
    with patch.object(fresh._run, "step", wraps=fresh._run.step) as steps:
        resumed = execution.resume_scientific_execution_v1(source.path, lifecycle=fresh)
    latest = max(
        (
            event.payload.sequence
            for event in prefix.events
            if type(event.payload.record) is ScientificExecutionCheckpointV1
        ),
        default=0,
    )
    assert steps.call_count == sum(
        event.payload.record.kind == "train" for event in prefix.events[latest:]
    )
    assert execution.read_scientific_execution_v1(source.path) == prefix
    assert fresh.state == lifecycle.state
    assert fresh.best_checkpoint == lifecycle.best_checkpoint
    assert build_scientific_checkpoint_v1(fresh._run) == build_scientific_checkpoint_v1(
        lifecycle._run
    )

    while fresh.state.phase != "stopped":
        resumed.advance()
    actual = execution.read_scientific_execution_v1(resumed.path)
    expected = execution.read_scientific_execution_v1(completed)
    assert actual.header == expected.header and actual.events == expected.events
    assert len(actual.events) == 8
    assert fresh.state.step == 3 and fresh.state.best_step == 1
    assert fresh.state.stop_reason == "early_stopping"
    final_record = expected.events[-1].payload.record
    assert type(final_record) is ScientificExecutionCheckpointV1
    final_bundle = read_scientific_lifecycle_checkpoint_v1(
        completed / "checkpoints/scientific-lifecycle-v1" / final_record.lifecycle_checkpoint_sha256
    )
    assert build_scientific_checkpoint_v1(fresh._run) == IrisScientificCheckpointBundleV1(
        final_bundle.manifest.payload.current, final_bundle.current_state_bytes
    )
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    with pytest.raises(execution.IrisScientificExecutionError):
        resumed.advance()


def test_event_failure_retries_pending_record_without_retraining(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    with (
        patch.object(os, "link", side_effect=OSError("event publication failed")),
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        writer.advance()
    assert lifecycle.state.step == 1 and writer.last_event is None and not writer.poisoned
    assert execution.read_scientific_execution_v1(writer.path).events == ()
    state = build_scientific_checkpoint_v1(lifecycle._run)
    with patch.object(lifecycle, "advance", side_effect=AssertionError("do not retrain")) as phases:
        event = writer.advance()
    phases.assert_not_called()
    assert event.payload.sequence == 1 and event.payload.after.step == 1
    assert build_scientific_checkpoint_v1(lifecycle._run) == state
    assert execution.read_scientific_execution_v1(writer.path).events == (event,)


def test_successful_link_with_lost_acknowledgement_is_exactly_once(tmp_path: Path) -> None:
    writer = execution.create_scientific_execution_v1(_lifecycle(), root=tmp_path)
    link = os.link

    def uncertain(source: Path, target: Path) -> None:
        link(source, target)
        raise OSError("lost acknowledgement")

    with patch.object(os, "link", side_effect=uncertain) as publish:
        event = writer.advance()
    assert publish.call_count == 1
    assert execution.read_scientific_execution_v1(writer.path).events == (event,)
    assert {entry.name for entry in writer.path.iterdir()} == {
        "run.json",
        "initial.safetensors",
        "events",
        "checkpoints",
    }


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(
        lifecycle, root=tmp_path_factory.mktemp("completed-execution")
    )
    with _stochastic(lifecycle._run):
        while lifecycle.state.phase != "stopped":
            writer.advance()
    return writer.path


def _copy(completed: Path, root: Path) -> Path:
    target = root / completed.name
    shutil.copytree(completed, target)
    return target


def _rehash_event(path: Path, transform: Any) -> None:
    data = json.loads(path.read_bytes())
    transform(data["payload"])
    data["sha256"] = domain_sha256(
        "iris/scientific-execution-event/v1", canonical_json_bytes(data["payload"])
    )
    path.write_bytes(canonical_json_file_bytes(data))


@pytest.mark.parametrize(
    "kind",
    [
        "header",
        "initial",
        "gap",
        "extra",
        "duplicate_json",
        "state",
        "lr",
        "loss",
        "cursor",
        "history",
        "validation_corpus",
        "validation_count",
        "missing_best",
        "checkpoint_reference",
    ],
)
def test_reader_rejects_corrupted_or_rehashed_contradictory_history(
    tmp_path: Path, completed: Path, kind: str
) -> None:
    target = _copy(completed, tmp_path)
    first = target / "events/00000001.json"
    if kind in ("header", "initial"):
        file = target / ("run.json" if kind == "header" else "initial.safetensors")
        data = file.read_bytes()
        file.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif kind == "gap":
        first.unlink()  # Only the fresh test-owned copy is modified.
    elif kind == "extra":
        (target / "events/extra.json").write_bytes(b"{}\n")
    elif kind == "duplicate_json":
        data = first.read_bytes()
        first.write_bytes(b'{"sha256":"' + b"0" * 64 + b'",' + data[1:])
    elif kind == "state":
        _rehash_event(first, lambda payload: payload["after"].update(phase="train"))
    elif kind == "lr":
        _rehash_event(first, lambda payload: payload["record"]["metric"].update(learning_rate=0.3))
    elif kind == "loss":
        _rehash_event(first, lambda payload: payload["record"]["metric"].update(loss=0.0))
    elif kind == "cursor":
        _rehash_event(first, lambda payload: payload.update(examples_seen=999))
    elif kind == "history":
        _rehash_event(first, lambda payload: payload.update(history_sha256="0" * 64))
    elif kind.startswith("validation"):
        update = (
            {"dataset_sha256": "0" * 64} if kind == "validation_corpus" else {"example_count": 5}
        )
        _rehash_event(
            target / "events/00000002.json",
            lambda payload: payload["record"]["metric"].update(update),
        )
    elif kind == "missing_best":
        _rehash_event(
            target / "events/00000002.json",
            lambda payload: payload["record"].update(selected_checkpoint_sha256=None),
        )
    else:
        _rehash_event(
            target / "events/00000008.json",
            lambda payload: payload["record"].update(lifecycle_checkpoint_sha256="0" * 64),
        )
    with pytest.raises(execution.IrisScientificExecutionError):
        execution.read_scientific_execution_v1(target)


@pytest.mark.parametrize("phase", ["train", "validation", "checkpoint"])
def test_failed_event_ack_after_commit_retries_without_another_phase(
    tmp_path: Path, phase: str
) -> None:
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    while lifecycle.state.phase != phase:
        writer.advance()
    append = execution._append_event

    def uncertain(*args: Any, **kwargs: Any) -> int:
        append(*args, **kwargs)
        raise OSError("lost event acknowledgement after append")

    with (
        patch.object(execution, "_append_event", side_effect=uncertain),
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        writer.advance()
    pending = writer._pending
    assert pending is not None and not writer.poisoned
    state = build_scientific_checkpoint_v1(lifecycle._run)
    with patch.object(
        lifecycle, "advance", side_effect=AssertionError("no repeated phase")
    ) as advance:
        assert writer.advance() == pending
    advance.assert_not_called()
    assert build_scientific_checkpoint_v1(lifecycle._run) == state
    snapshot = execution.read_scientific_execution_v1(writer.path)
    assert snapshot.events[-1] == pending
    assert snapshot.journal_size_bytes == len(execution._header_bytes(writer.header)) + sum(
        len(execution._event_bytes(event)) for event in snapshot.events
    )


def test_interrupted_event_publication_retains_exact_retry(tmp_path: Path) -> None:
    writer = execution.create_scientific_execution_v1(_lifecycle(), root=tmp_path)
    link = os.link

    def interrupt(source: Path, target: Path) -> None:
        link(source, target)
        raise KeyboardInterrupt("committed event interruption")

    with patch.object(os, "link", side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        writer.advance()
    pending = writer._pending
    assert pending is not None and not writer.poisoned
    assert execution.read_scientific_execution_v1(writer.path).events == (pending,)
    assert writer.advance() == pending


@pytest.mark.parametrize("where", ["initial", "event"])
def test_failed_mkdir_never_owns_an_existing_collision(tmp_path: Path, where: str) -> None:
    lifecycle = _lifecycle()
    writer = (
        None
        if where == "initial"
        else execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    )
    parent = tmp_path / "scientific-executions-v1" if writer is None else writer.path
    collision = parent / f".iris-{'execution' if writer is None else 'event'}-{'c' * 32}.tmp"
    collision.mkdir(parents=True)
    marker = collision / ("run.json" if writer is None else "event.json")
    marker.write_bytes(b"existing bytes must survive")
    with (
        patch.object(execution, "uuid4", return_value=SimpleNamespace(hex="c" * 32)),
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        if writer is None:
            execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
        else:
            writer.advance()
    assert collision.is_dir() and marker.read_bytes() == b"existing bytes must survive"
    if writer is not None:
        writer.advance()
        assert marker.read_bytes() == b"existing bytes must survive"


def test_journal_budget_rejects_before_the_next_lifecycle_phase(tmp_path: Path) -> None:
    first = execution.create_scientific_execution_v1(_lifecycle(), root=tmp_path / "sizing")
    budget = len(execution._header_bytes(first.header)) + 65536 + 256
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(
        lifecycle, root=tmp_path / "bounded", maximum_journal_bytes=budget
    )
    event = writer.advance()
    with (
        patch.object(
            lifecycle, "advance", side_effect=AssertionError("budget before training")
        ) as phase,
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        writer.advance()
    phase.assert_not_called()
    assert writer.last_event == event and not writer.poisoned


def test_exact_resume_in_a_separate_python_process(tmp_path: Path, completed: Path) -> None:
    target = _copy(completed, tmp_path)
    before = execution.read_scientific_execution_v1(target)
    for sequence in (7, 8):
        (target / "events" / f"{sequence:08d}.json").unlink()
    script = """
import sys
from pathlib import Path
from iris.application.scientific_execution import resume_scientific_execution_v1
from iris.application import scientific_execution
from tests.integration.test_scientific_checkpoint import _stochastic
from tests.integration.test_scientific_lifecycle import _lifecycle
lifecycle = _lifecycle()
assert Path(scientific_execution.__file__).resolve().is_relative_to(Path(sys.argv[2]))
with _stochastic(lifecycle._run):
    writer = resume_scientific_execution_v1(Path(sys.argv[1]), lifecycle=lifecycle)
    while lifecycle.state.phase != "stopped":
        writer.advance()
print(writer.last_event.sha256)
"""
    project = Path(__file__).resolve().parents[2]
    package_root = Path(execution.__file__).resolve().parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(package_root), str(project))),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(target), str(package_root)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == before.events[-1].sha256
    assert execution.read_scientific_execution_v1(target) == before


def test_rehashed_alternate_best_weights_do_not_replace_validation_selection(
    tmp_path: Path, completed: Path
) -> None:
    target = _copy(completed, tmp_path)
    snapshot = execution.read_scientific_execution_v1(target)
    record = snapshot.events[-1].payload.record
    assert type(record) is ScientificExecutionCheckpointV1
    old = read_scientific_lifecycle_checkpoint_v1(
        target / "checkpoints/scientific-lifecycle-v1" / record.lifecycle_checkpoint_sha256
    )
    best = old.manifest.payload.best
    assert best is not None and old.best_state_bytes is not None
    tensors = load_tensors(old.best_state_bytes)
    tensors[f"model/{best.payload.parameter_names[0]}"].add_(0.1)
    changed = save_tensors(
        dict(sorted(tensors.items())), metadata={"format": "iris.scientific-checkpoint.v1"}
    )
    best_payload = best.payload.model_copy(
        update={"state_sha256": bytes_sha256(changed), "state_size_bytes": len(changed)}
    )
    new_best = IrisScientificCheckpointManifestV1(
        sha256=domain_sha256(
            "iris/scientific-checkpoint/v1/payload",
            canonical_json_bytes(best_payload.model_dump(mode="json")),
        ),
        payload=best_payload,
    )
    _verify(IrisScientificCheckpointBundleV1(new_best, changed))
    payload = old.manifest.payload.model_copy(update={"best": new_best})
    manifest = IrisScientificLifecycleCheckpointManifestV1(
        sha256=domain_sha256(
            "iris/scientific-lifecycle-checkpoint/v1",
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    replacement = IrisScientificLifecycleBundleV1(manifest, old.current_state_bytes, changed)
    published = _publish(replacement, target / "checkpoints", budget=1024**3)
    assert read_scientific_lifecycle_checkpoint_v1(published) == replacement
    _rehash_event(
        target / "events/00000008.json",
        lambda event: event["record"].update(lifecycle_checkpoint_sha256=manifest.sha256),
    )
    with pytest.raises(execution.IrisScientificExecutionError) as caught:
        execution.read_scientific_execution_v1(target)
    assert "exact validation-selected checkpoint" in str(caught.value.__cause__)


def test_changed_corpus_fails_before_copying_or_poisoning_fresh_owner(
    tmp_path: Path, completed: Path
) -> None:
    target = _copy(completed, tmp_path)
    fresh = IrisScientificLifecycleV1(
        run=_run(reverse=True), validation=_validation(), policy=_lifecycle()._policy
    )
    before = build_scientific_checkpoint_v1(fresh._run)
    with pytest.raises(execution.IrisScientificExecutionError):
        execution.resume_scientific_execution_v1(target, lifecycle=fresh)
    assert not fresh._poisoned and not fresh._run.poisoned
    assert build_scientific_checkpoint_v1(fresh._run) == before


def test_replay_mismatch_poisons_restored_owner_without_changing_journal(
    tmp_path: Path, completed: Path
) -> None:
    target = _copy(completed, tmp_path)
    for sequence in range(2, 9):
        (target / "events" / f"{sequence:08d}.json").unlink()
    before = execution.read_scientific_execution_v1(target)
    fresh = _lifecycle()  # No stochastic forward: same graph/config, different actual replay.
    with pytest.raises(execution.IrisScientificExecutionError):
        execution.resume_scientific_execution_v1(target, lifecycle=fresh)
    assert fresh._poisoned and fresh._run.poisoned and fresh._run._optimizer.poisoned
    assert execution.read_scientific_execution_v1(target) == before


def test_checkpoint_publication_failure_keeps_pending_controller_phase(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    while lifecycle.state.phase != "checkpoint":
        writer.advance()
    before = execution.read_scientific_execution_v1(writer.path)
    with (
        patch(
            "iris.application.scientific_lifecycle._publish",
            side_effect=OSError("checkpoint failed"),
        ),
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        writer.advance()
    assert lifecycle._pending is not None and not writer.poisoned
    assert execution.read_scientific_execution_v1(writer.path) == before
    with patch.object(
        lifecycle._run, "step", side_effect=AssertionError("no repeated training")
    ) as step:
        event = writer.advance()
    step.assert_not_called()
    assert event.payload.record.kind == "checkpoint"
    assert execution.read_scientific_execution_v1(writer.path).events[-1] == event


def test_public_exports_and_factory_only_owner(tmp_path: Path) -> None:
    import iris.application as application

    for name in execution.__all__:
        assert getattr(application, name) is getattr(execution, name)
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    snapshot = execution.read_scientific_execution_v1(writer.path)
    with pytest.raises(execution.IrisScientificExecutionError):
        execution.IrisScientificExecutionV1(snapshot=snapshot, lifecycle=lifecycle)


@pytest.mark.parametrize("interruption", [False, True])
def test_initial_publication_uncertain_ack_preserves_reusable_exact_target(
    tmp_path: Path, interruption: bool
) -> None:
    lifecycle = _lifecycle()

    def uncertain(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        if interruption:
            raise KeyboardInterrupt("initial move completed")
        raise OSError("initial move completed")

    with patch.object(execution, "rename_directory_with_windows_retry", side_effect=uncertain):
        if interruption:
            with pytest.raises(KeyboardInterrupt):
                execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
        else:
            execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    assert execution.read_scientific_execution_v1(writer.path).events == ()
    assert tuple((tmp_path / "scientific-executions-v1").iterdir()) == (writer.path,)
    assert lifecycle.state.step == 0


@pytest.mark.parametrize("cut", [3, 5, 7, 9])
def test_off_cadence_budget_terminal_and_initial_checkpoint_without_best(
    tmp_path: Path, cut: int
) -> None:
    original = _short_lifecycle()
    writer = execution.create_scientific_execution_v1(original, root=tmp_path / "original")
    for _ in range(cut):
        writer.advance()
    fresh = _short_lifecycle()
    resumed = execution.resume_scientific_execution_v1(writer.path, lifecycle=fresh)
    assert fresh.state == original.state
    assert fresh.best_checkpoint == original.best_checkpoint
    assert build_scientific_checkpoint_v1(fresh._run) == build_scientific_checkpoint_v1(
        original._run
    )
    while fresh.state.phase != "stopped":
        resumed.advance()
    snapshot = execution.read_scientific_execution_v1(writer.path)
    assert len(snapshot.events) == 10
    assert snapshot.events[2].payload.after.best_step is None
    assert snapshot.events[-1].payload.after.stop_reason == "step_budget"
    assert snapshot.events[-1].payload.after.step == 5
    assert snapshot.events[-1].payload.after.last_validation_step == 5


def test_selected_best_can_share_current_checkpoint_storage(tmp_path: Path) -> None:
    lifecycle = _short_lifecycle(validate_every_steps=1, checkpoint_every_steps=1)
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    for _ in range(3):
        writer.advance()
    snapshot = execution.read_scientific_execution_v1(writer.path)
    record = snapshot.events[-1].payload.record
    assert type(record) is ScientificExecutionCheckpointV1
    bundle = read_scientific_lifecycle_checkpoint_v1(
        writer.path / "checkpoints/scientific-lifecycle-v1" / record.lifecycle_checkpoint_sha256
    )
    assert bundle.manifest.payload.current == bundle.manifest.payload.best
    assert bundle.best_state_bytes is None
    fresh = _short_lifecycle(validate_every_steps=1, checkpoint_every_steps=1)
    execution.resume_scientific_execution_v1(writer.path, lifecycle=fresh)
    assert fresh.state == lifecycle.state and fresh.best_checkpoint == lifecycle.best_checkpoint


def test_lost_phase_result_poisons_runner_and_fresh_resume_discards_uncommitted_tail(
    tmp_path: Path,
) -> None:
    lifecycle = _lifecycle()
    writer = execution.create_scientific_execution_v1(lifecycle, root=tmp_path)
    advance = lifecycle.advance

    def uncertain(*args: Any, **kwargs: Any) -> Any:
        advance(*args, **kwargs)
        raise OSError("phase updated but result was lost")

    with (
        patch.object(lifecycle, "advance", side_effect=uncertain),
        pytest.raises(execution.IrisScientificExecutionError),
    ):
        writer.advance()
    assert writer.poisoned and lifecycle.state.step == 1
    assert execution.read_scientific_execution_v1(writer.path).events == ()
    with pytest.raises(execution.IrisScientificExecutionError):
        writer.advance()
    fresh = _lifecycle()
    resumed = execution.resume_scientific_execution_v1(writer.path, lifecycle=fresh)
    assert fresh.state.step == 0
    resumed.advance()
    assert build_scientific_checkpoint_v1(fresh._run) == build_scientific_checkpoint_v1(
        lifecycle._run
    )
