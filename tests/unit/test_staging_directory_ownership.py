"""A failed exclusive mkdir must never grant cleanup ownership of an existing path.

Only preflight serialization is stubbed, to isolate the real filesystem fence.
Every fixture and deliberately colliding directory is below pytest's tmp_path.
"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

_SUFFIX = "c" * 32


@pytest.mark.parametrize(
    "kind",
    [
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
def test_failed_mkdir_never_cleans_an_existing_staging_directory(tmp_path: Path, kind: str) -> None:
    module = import_module(f"iris.application.{kind}")
    manifest = SimpleNamespace(sha256="1" * 64, model_dump=lambda **kwargs: {})
    bundle: Any = SimpleNamespace(manifest=manifest, sha256=manifest.sha256, state_bytes=b"new")
    extra: dict[str, Any] = {}
    verified: Any
    names: tuple[str, ...]
    if kind == "scientific_checkpoint":
        parent, prefix, operation = tmp_path, ".iris-scientific-", "write_scientific_checkpoint_v1"
        preflight, verified = "_verify", (manifest, {})
        names = ("manifest.json", "state.safetensors")
    elif kind == "scientific_lifecycle":
        parent, prefix, operation = tmp_path / "scientific-lifecycle-v1", ".iris-", "_publish"
        preflight, verified = "_files", {"manifest.json": b"new", "current.safetensors": b"new"}
        names = ("manifest.json", "current.safetensors", "best.safetensors")
        extra = {"budget": 1024**3}
    elif kind == "scientific_inference":
        parent, prefix, operation = (
            tmp_path / "scientific-inference-v1",
            ".iris-inference-",
            "write_scientific_inference_v1",
        )
        preflight, verified = "_checked", (manifest, {})
        names = ("manifest.json", "iris_model.safetensors")
    elif kind == "scientific_delivery":
        parent, prefix, operation = (
            tmp_path / "scientific-delivery-v1",
            ".iris-delivery-",
            "write_scientific_delivery_v1",
        )
        preflight, verified = "_checked", None
        names = tuple(module._FILE_LIMITS)
        bundle.files = tuple((name, b"new") for name in names)
    elif "training_checkpoint" in kind:
        version = {
            "training_checkpoint": 1,
            "no_workspace_training_checkpoint": 2,
            "shared_buffer_training_checkpoint": 3,
        }[kind]
        parent, prefix = tmp_path / "teacher" / f"v{version}", ".iris-"
        operation = {
            1: "write_teacher_checkpoint_v1",
            2: "write_no_workspace_teacher_checkpoint_v2",
            3: "write_shared_buffer_teacher_checkpoint_v3",
        }[version]
        preflight, verified = "_validate_bundle_integrity", SimpleNamespace(manifest=manifest)
        names = (module._MANIFEST_FILENAME, module._STATE_FILENAME)
    else:
        version = {"experiment_log": 1, "experiment_log_v2": 2, "experiment_log_v3": 3}[kind]
        parent, prefix, operation = tmp_path / "experiments" / f"v{version}", ".iris-", "_write_run"
        preflight, verified = (
            "_run_bytes",
            (SimpleNamespace(payload=SimpleNamespace(run_id="new-run")), b"new"),
        )
        names = (module._RUN_FILENAME,)
    collision = parent / f"{prefix}{_SUFFIX}.tmp"
    collision.mkdir(parents=True)
    for name in names:
        (collision / name).write_bytes(b"existing content must survive")
    if kind.startswith("experiment_log"):
        (collision / module._EVENTS_DIRECTORY).mkdir()
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
    expected_error: type[Exception] = getattr(module, error_name)
    before = {
        p.relative_to(collision).as_posix(): None if p.is_dir() else p.read_bytes()
        for p in collision.rglob("*")
    }
    with (
        patch.object(module, "uuid4", return_value=SimpleNamespace(hex=_SUFFIX)),
        patch.object(module, preflight, return_value=verified),
        pytest.raises(expected_error),
    ):
        getattr(module, operation)(bundle, root=tmp_path, **extra)
    assert collision.is_dir(), "failed mkdir must not delete the pre-existing directory"
    after = {
        p.relative_to(collision).as_posix(): None if p.is_dir() else p.read_bytes()
        for p in collision.rglob("*")
    }
    assert after == before, "failed mkdir must not remove any pre-existing member"
