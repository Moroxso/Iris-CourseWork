"""Real codecs must distinguish an uncertain rename from an invalid publication.

Only the rename acknowledgement is injected. Serialization, target validation,
filesystem moves/copies and staging cleanup are real, within fresh pytest paths.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.utils.directory_publication import rename_directory_with_windows_retry
from tests.integration.test_experiment_log_flow import _create as create_v1
from tests.integration.test_experiment_log_v2_flow import _create as create_v2
from tests.integration.test_experiment_log_v3_flow import _create as create_v3
from tests.integration.test_no_workspace_teacher_checkpoint_flow import _owner as owner_b
from tests.integration.test_scientific_checkpoint import _run
from tests.integration.test_scientific_lifecycle import _short_lifecycle
from tests.integration.test_shared_buffer_teacher_checkpoint_flow import _owner as owner_c
from tests.integration.test_teacher_checkpoint_flow import _trained_bundle

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Publication:
    module: ModuleType
    rename_name: str
    publish: Callable[[Path], Path]
    read: Callable[[Path], object]
    expected: object
    error: type[Exception]


def _prepare(kind: str, seed_root: Path) -> _Publication:
    module = import_module(f"iris.application.{kind}")
    rename_name = "rename_directory_with_windows_retry"
    extra: dict[str, object] = {}
    bundle: Any
    if kind == "scientific_lifecycle":
        lifecycle = _short_lifecycle()
        while lifecycle.state.phase != "checkpoint":
            lifecycle.advance(checkpoint_root=seed_root)
        path = lifecycle.advance(checkpoint_root=seed_root)
        assert isinstance(path, Path)
        reader = module.read_scientific_lifecycle_checkpoint_v1
        bundle, writer = reader(path), module._publish
        extra = {"budget": 1024**3}
    elif kind.startswith("scientific_"):
        run = _run()
        run.step()
        builder = getattr(module, f"build_{kind}_v1")
        bundle = builder(run)
        writer, reader = (
            getattr(module, f"write_{kind}_v1"),
            getattr(module, f"read_{kind}_v1"),
        )
    elif "training_checkpoint" in kind:
        if kind == "training_checkpoint":
            bundle = _trained_bundle()[-1]
            stem = "teacher_checkpoint_v1"
        elif kind == "no_workspace_training_checkpoint":
            model_b, optimizer_b = owner_b(stepped=True)
            bundle = module.build_no_workspace_teacher_checkpoint_v2(
                model=model_b, optimizer=optimizer_b
            )
            stem = "no_workspace_teacher_checkpoint_v2"
        else:
            model_c, optimizer_c = owner_c(stepped=True)
            bundle = module.build_shared_buffer_teacher_checkpoint_v3(
                model=model_c, optimizer=optimizer_c
            )
            stem = "shared_buffer_teacher_checkpoint_v3"
        writer, reader = getattr(module, f"write_{stem}"), getattr(module, f"read_{stem}")
    else:
        if kind == "experiment_log":
            torch.manual_seed(1907)
            path = create_v1(seed_root)[-1].path
            version = 1
        elif kind == "experiment_log_v2":
            path = create_v2(root=seed_root, arm_id="B", run_id="acknowledgement-v2")[-1].path
            version = 2
        else:
            path = create_v3(root=seed_root, arm_id="C", run_id="acknowledgement-v3")[-1].path
            version = 3
        if version != 1:
            rename_name = "_rename_directory_with_permission_retry"
        journal_reader = getattr(module, f"read_experiment_log_v{version}")

        def read_run(location: Path) -> object:
            return journal_reader(location).run

        reader = read_run
        bundle, writer = reader(path), module._write_run

    def publish(root: Path) -> Path:
        result = writer(bundle, root=root, **extra)
        location = (
            result
            if isinstance(result, Path)
            else result[0]
            if isinstance(result, tuple)
            else result.path
        )
        assert isinstance(location, Path)
        return location

    error_name = (
        {
            "scientific_checkpoint": "IrisScientificCheckpointError",
            "scientific_lifecycle": "IrisScientificLifecycleError",
            "scientific_inference": "IrisScientificInferenceError",
            "scientific_delivery": "IrisScientificDeliveryError",
        }[kind]
        if kind.startswith("scientific_")
        else "IrisExperimentLogError"
        if kind.startswith("experiment_log")
        else "IrisTeacherCheckpointError"
    )
    return _Publication(module, rename_name, publish, reader, bundle, getattr(module, error_name))


@pytest.fixture(
    scope="module",
    params=[
        "scientific_checkpoint",
        "scientific_lifecycle",
        "scientific_inference",
        "scientific_delivery",
        "training_checkpoint",
        "no_workspace_training_checkpoint",
        "shared_buffer_training_checkpoint",
        "experiment_log",
        "experiment_log_v2",
        "experiment_log_v3",
    ],
)
def publication(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> _Publication:
    with torch.random.fork_rng(devices=[]):
        return _prepare(str(request.param), tmp_path_factory.mktemp("publication-seed"))


def _tree(path: Path) -> dict[str, bytes | None]:
    return {
        member.relative_to(path).as_posix(): None if member.is_dir() else member.read_bytes()
        for member in path.rglob("*")
    }


@pytest.mark.parametrize(
    "mode",
    ["moved", "copied", "missing", "conflicting", "corrupt_moved", "extra_moved", "interrupted"],
)
def test_exact_target_survives_uncertain_acknowledgement(
    tmp_path: Path, publication: _Publication, mode: str
) -> None:
    targets: list[Path] = []
    stages: list[Path] = []
    installed: dict[str, bytes | None] = {}
    rename = getattr(publication.module, publication.rename_name)

    def uncertain(source: Path, target: Path) -> None:
        nonlocal installed
        stages.append(source)
        targets.append(target)
        if mode in ("moved", "corrupt_moved", "extra_moved", "interrupted"):
            rename(source, target)
        elif mode in ("copied", "conflicting"):
            shutil.copytree(source, target)  # Another winner; owned staging remains.
        if mode in ("conflicting", "corrupt_moved"):
            member = next(path for path in target.iterdir() if path.is_file())
            content = member.read_bytes()
            member.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        if mode == "extra_moved":
            (target / "unexpected.txt").write_bytes(b"invalid extra member")
        if target.exists():
            installed = _tree(target)
        if mode == "interrupted":
            raise KeyboardInterrupt("interrupted after rename")
        raise OSError("lost rename acknowledgement")

    succeeds = mode in ("moved", "copied")
    with patch.object(publication.module, publication.rename_name, side_effect=uncertain) as move:
        if succeeds:
            result = publication.publish(tmp_path)
            assert result == targets[0]
        else:
            expected_error = KeyboardInterrupt if mode == "interrupted" else publication.error
            with pytest.raises(expected_error):
                publication.publish(tmp_path)
    assert move.call_count == 1
    assert len(targets) == len(stages) == 1 and not stages[0].exists()
    target = targets[0]
    if mode == "missing":
        assert not target.exists() and tuple(target.parent.iterdir()) == ()
        recovered = publication.publish(tmp_path)
        assert publication.read(recovered) == publication.expected
        return
    assert tuple(target.parent.iterdir()) == (target,)
    assert _tree(target) == installed, "never clean or replace the published target"
    with patch.object(
        publication.module, publication.rename_name, side_effect=AssertionError("no second rename")
    ) as retry:
        if succeeds or mode == "interrupted":
            assert publication.read(target) == publication.expected
            assert publication.publish(tmp_path) == target
        else:
            with pytest.raises(publication.error):
                publication.read(target)
            with pytest.raises(publication.error):
                publication.publish(tmp_path)
    retry.assert_not_called()
    assert _tree(target) == installed


def test_lifecycle_commits_exactly_once_after_uncertain_rename(tmp_path: Path) -> None:
    from iris.application import scientific_lifecycle as module
    from iris.application.scientific_checkpoint import build_scientific_checkpoint_v1

    lifecycle = _short_lifecycle()
    while lifecycle.state.phase != "checkpoint":
        lifecycle.advance(checkpoint_root=tmp_path)
    before = build_scientific_checkpoint_v1(lifecycle._run)
    rename = rename_directory_with_windows_retry

    def uncertain(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("lost acknowledgement after successful rename")

    with (
        patch.object(module, "rename_directory_with_windows_retry", side_effect=uncertain) as move,
        patch.object(
            lifecycle._run, "step", side_effect=AssertionError("no repeated training")
        ) as step,
    ):
        path = lifecycle.advance(checkpoint_root=tmp_path)
    step.assert_not_called()
    assert move.call_count == 1 and isinstance(path, Path)
    current = lifecycle.state
    assert current.phase == "train" and current.checkpoint_count == 1
    assert lifecycle._pending is None and not lifecycle._poisoned
    assert lifecycle._run.completed_steps == lifecycle.state.step == 2
    assert build_scientific_checkpoint_v1(lifecycle._run) == before
    persisted = module.read_scientific_lifecycle_checkpoint_v1(path)
    assert persisted.manifest.payload.state == lifecycle.state
    fresh = _short_lifecycle()
    module.restore_scientific_lifecycle_v1(persisted, lifecycle=fresh)
    assert fresh.state == lifecycle.state
    assert build_scientific_checkpoint_v1(fresh._run) == before
