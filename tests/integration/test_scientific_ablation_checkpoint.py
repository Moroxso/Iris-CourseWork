"""Distinct mode-bound disk recovery on real tiny Qwen, not scientific acceptance."""

import json
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

import iris.application as application
import iris.artifacts as artifacts
from iris.application import scientific_ablation_checkpoint as module
from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    IrisScientificAblationCheckpointError,
)
from iris.application.scientific_ablation_checkpoint import (
    build_scientific_ablation_checkpoint_v1 as build,
)
from iris.application.scientific_ablation_checkpoint import (
    read_scientific_ablation_checkpoint_v1 as read,
)
from iris.application.scientific_ablation_checkpoint import (
    restore_scientific_ablation_checkpoint_v1 as restore,
)
from iris.application.scientific_ablation_checkpoint import (
    write_scientific_ablation_checkpoint_v1 as write,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointError,
    build_scientific_checkpoint_v1,
    read_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
    write_scientific_checkpoint_v1,
)
from iris.application.scientific_optimization import IrisScientificOptimizerPolicyV1, _same_runtime
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_checkpoint_v1 import (
    IrisScientificAblationCheckpointManifestV1,
    ScientificAblationCheckpointPayloadV1,
)
from iris.artifacts.scientific_checkpoint_v1 import ScientificTensorV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    domain_sha256,
    named_module_state_sha256,
)
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_run import _assert_equal, _run, _stochastic
from tests.integration.test_scientific_checkpoint import _run as _old_run
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


def _repack(
    bundle: IrisScientificAblationCheckpointBundleV1,
    *,
    updates: dict[str, Any] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, str] | None = None,
) -> IrisScientificAblationCheckpointBundleV1:
    changes = {} if updates is None else dict(updates)
    state = bundle.state_bytes
    if tensors is not None:
        state = save_tensors(
            dict(sorted(tensors.items())), metadata=module._META if metadata is None else metadata
        )
        changes.update(
            state_sha256=bytes_sha256(state),
            state_size_bytes=len(state),
            tensors=tuple(
                ScientificTensorV1(
                    key=key,
                    shape=tuple(value.shape),
                    dtype="float32" if value.dtype == torch.float32 else "uint8",
                    byte_length=value.numel() * value.element_size(),
                )
                for key, value in sorted(tensors.items())
            ),
        )
    payload = bundle.manifest.payload.model_copy(update=changes)
    manifest = IrisScientificAblationCheckpointManifestV1.model_construct(
        sha256=domain_sha256(module._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    return IrisScientificAblationCheckpointBundleV1(manifest, state)


@pytest.fixture(scope="module")
def stepped() -> IrisScientificAblationCheckpointBundleV1:
    owner = _run("random", goal="disabled", broadcast="disabled")
    with _stochastic(owner):
        owner.step()
    return build(owner)


def _fresh() -> IrisScientificAblationRunV1:
    return _run("random", goal="disabled", broadcast="disabled")


def test_canonical_exports_initial_bundle_and_detached_inventory(tmp_path: Path) -> None:
    for name in module.__all__:
        assert name in application.__all__
        assert getattr(application, name) is getattr(module, name)
    assert (
        artifacts.IrisScientificAblationCheckpointManifestV1
        is IrisScientificAblationCheckpointManifestV1
    )
    run = _fresh()
    ambient = capture_runtime_state(torch.device("cpu"))
    bundle = build(run)
    payload = bundle.manifest.payload
    assert type(payload) is ScientificAblationCheckpointPayloadV1
    assert payload.kind == "iris-scientific-ablation-checkpoint-v1"
    assert not any(payload.parameter_steps)
    assert not any(entry.key.startswith("adamw/") for entry in payload.tensors)
    assert payload.scientific_state_sha256 == run._optimizer._state_sha256()
    path = write(bundle, root=tmp_path / "checkpoints")
    assert read(path) == bundle and write(bundle, root=path.parent) == path
    assert {entry.name for entry in path.iterdir()} == {"manifest.json", "state.safetensors"}
    destination = _fresh()
    restore(bundle, run=destination)
    assert build(destination) == bundle
    _assert_equal(run, destination)
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
@pytest.mark.parametrize("cut", [0, 1, 3])
def test_all_modes_stochastic_disk_resume_is_bitwise_across_epoch_tail(
    tmp_path: Path,
    routing: Literal["learned", "random", "uniform"],
    goal: str,
    broadcast: str,
    cut: int,
) -> None:
    def owner() -> IrisScientificAblationRunV1:
        return _run(
            routing, goal=goal, broadcast=broadcast, policy=_policy(minimum_target_tokens=1)
        )

    continuous, interrupted = owner(), owner()
    ambient = capture_runtime_state(torch.device("cpu"))
    with _stochastic(continuous):
        expected = [continuous.step() for _ in range(6)]
    with _stochastic(interrupted):
        for index in range(cut):
            assert interrupted.step() == expected[index]
    bundle = build(interrupted)
    if cut:
        assert bundle.manifest.payload.runtime.python_gaussian is not None
    initial_identity = json.loads(interrupted.configuration_json)["initial_scientific_state_sha256"]
    location = write(bundle, root=tmp_path / "checkpoints")
    loaded = read(location)
    assert loaded == bundle
    resumed = owner()
    attachment = resumed._attachment_token
    parameters = tuple(resumed._optimizer._topology.parameters)
    restore(loaded, run=resumed)
    assert resumed._attachment_token is attachment
    assert resumed._optimizer._attached_run_token is attachment
    assert all(
        p is q for p, q in zip(parameters, resumed._optimizer._topology.parameters, strict=True)
    )
    assert (
        json.loads(resumed.configuration_json)["initial_scientific_state_sha256"]
        == initial_identity
    )
    _assert_equal(interrupted, resumed)
    assert resumed._optimizer._current_lr == interrupted._optimizer._current_lr
    with _stochastic(resumed):
        for index in range(cut, 6):
            assert resumed.step() == expected[index]
    _assert_equal(continuous, resumed)
    assert build(continuous) == build(resumed)
    assert resumed.examples_seen == 6 > len(resumed._rows)
    assert not resumed.poisoned and not resumed._busy
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    assert any(count == 0 for count in resumed._optimizer._parameter_steps)
    assert all(
        (p in resumed._optimizer._optimizer.state) == bool(count)
        for p, count in zip(parameters, resumed._optimizer._parameter_steps, strict=True)
    )
    assert all(
        p.grad is None and not p.requires_grad for p in resumed._backbone._model.parameters()
    )


def test_bundle_has_no_alias_with_later_training_or_restored_owner(
    stepped: IrisScientificAblationCheckpointBundleV1,
) -> None:
    target = _fresh()
    restore(stepped, run=target)
    before = build(target)
    with _stochastic(target):
        target.step()
    assert stepped == before and build(target) != stepped


@pytest.mark.parametrize(
    "kind",
    [
        "mode",
        "goal",
        "broadcast",
        "routing_seed",
        "run_seed",
        "model_seed",
        "corpus_order",
        "numeric_policy",
        "nonfresh",
    ],
)
def test_incompatible_destination_is_untouched(
    stepped: IrisScientificAblationCheckpointBundleV1,
    kind: str,
) -> None:
    destination = _run(
        "uniform" if kind == "mode" else "random",
        goal="enabled" if kind == "goal" else "disabled",
        broadcast="enabled" if kind == "broadcast" else "disabled",
        routing_seed=1337 if kind == "routing_seed" else 42,
        run_seed=1337 if kind == "run_seed" else 901,
        model_seed=1908 if kind == "model_seed" else 1907,
        examples=tuple(reversed(_rows()[:5])) if kind == "corpus_order" else None,
        policy=_policy(minimum_target_tokens=71) if kind == "numeric_policy" else None,
    )
    if kind == "nonfresh":
        destination.step()
    before = build(destination)
    token = destination._attachment_token
    versions = destination._optimizer._versions
    ambient = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(IrisScientificAblationCheckpointError, match="could not be restored"):
        restore(stepped, run=destination)
    assert build(destination) == before and not destination.poisoned and not destination._busy
    assert destination._attachment_token is token and destination._optimizer._versions == versions
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "kind",
    [
        "state_hash",
        "state_size",
        "scientific_hash",
        "configuration_hash",
        "manifest_hash",
        "examples",
        "tokens",
        "unknown_field",
        "kind",
        "duplicate_parameter",
        "duplicate_model",
        "negative_counter",
        "counter_overflow",
        "max_counter",
        "missing_tensor",
        "extra_tensor",
        "nan",
        "negative_moment",
        "moment_shape",
        "cpu_rng",
        "cpu_rng_dtype",
        "metadata",
    ],
)
def test_malformed_candidate_fails_closed_before_any_destination_write(
    stepped: IrisScientificAblationCheckpointBundleV1,
    kind: str,
) -> None:
    payload = stepped.manifest.payload
    updates: dict[str, Any] = {}
    tensors: dict[str, torch.Tensor] | None = None
    metadata: dict[str, str] | None = None
    if kind in ("state_hash", "scientific_hash", "configuration_hash"):
        updates[
            {
                "state_hash": "state_sha256",
                "scientific_hash": "scientific_state_sha256",
                "configuration_hash": "configuration_sha256",
            }[kind]
        ] = "0" * 64
    elif kind == "state_size":
        updates["state_size_bytes"] = len(stepped.state_bytes) + 1
    elif kind == "examples":
        updates["examples_seen"] = 0
    elif kind == "tokens":
        updates["target_tokens_seen"] = 1
    elif kind == "unknown_field":
        updates["unknown"] = 1
    elif kind == "kind":
        updates["kind"] = "iris-scientific-checkpoint-v1"
    elif kind == "duplicate_parameter":
        updates["parameter_names"] = (payload.parameter_names[0],) * len(payload.parameter_names)
    elif kind == "duplicate_model":
        updates["model_state_names"] = (*payload.model_state_names, payload.model_state_names[0])
    elif kind in ("negative_counter", "counter_overflow", "max_counter"):
        updates["parameter_steps"] = (
            ((-1 if kind == "negative_counter" else 2), *payload.parameter_steps[1:])
            if kind != "max_counter"
            else (0,) * len(payload.parameter_steps)
        )
    elif kind != "manifest_hash":
        tensors = load_tensors(stepped.state_bytes)
        moment = next(name for name in tensors if name.endswith("/exp_avg_sq"))
        if kind == "missing_tensor":
            tensors.pop("runtime/cpu")
        elif kind == "extra_tensor":
            tensors["unknown/tensor"] = torch.ones(1)
        elif kind == "nan":
            tensors[moment].flatten()[0] = float("nan")
        elif kind == "negative_moment":
            tensors[moment].flatten()[0] = -1
        elif kind == "moment_shape":
            tensors[moment] = torch.zeros(1)
        elif kind == "cpu_rng":
            tensors["runtime/cpu"] = torch.zeros(8, dtype=torch.uint8)
        elif kind == "cpu_rng_dtype":
            tensors["runtime/cpu"] = torch.zeros(8)
        elif kind == "metadata":
            metadata = {"format": "iris.scientific-checkpoint.v1"}
    damaged = _repack(stepped, updates=updates, tensors=tensors, metadata=metadata)
    if kind == "manifest_hash":
        damaged = replace(
            damaged, manifest=damaged.manifest.model_copy(update={"sha256": "0" * 64})
        )
    target = _fresh()
    before = build(target)
    with pytest.raises(IrisScientificAblationCheckpointError):
        restore(damaged, run=target)
    assert build(target) == before and not target.poisoned


@pytest.mark.parametrize(
    "kind",
    [
        "kind",
        "ordering",
        "extra",
        "ablation_sha",
        "random_seed",
        "software",
        "runtime_flags",
        "noncanonical",
        "duplicate",
    ],
)
def test_configuration_is_strict_canonical_and_mode_bound(
    stepped: IrisScientificAblationCheckpointBundleV1,
    kind: str,
    tmp_path: Path,
) -> None:
    configuration = json.loads(stepped.manifest.payload.configuration_json)
    if kind == "kind":
        configuration["kind"] = "iris-scientific-run-v1"
    elif kind == "ordering":
        configuration["ordering"] = "shuffled"
    elif kind == "extra":
        configuration["unknown"] = 1
    elif kind == "ablation_sha":
        configuration["ablation_policy_sha256"] = "0" * 64
    elif kind == "random_seed":
        configuration["ablation"]["routing_seed"] = None
    elif kind == "software":
        configuration["software"]["unknown"] = configuration["software"].pop("torch")
    elif kind == "runtime_flags":
        configuration["runtime_policy"]["deterministic_algorithms"] = not configuration[
            "runtime_policy"
        ]["deterministic_algorithms"]
    encoded = canonical_json_bytes(configuration)
    if kind == "noncanonical":
        encoded += b" "
    elif kind == "duplicate":
        encoded = b'{"seed":901,' + encoded[1:]
    damaged = _repack(
        stepped,
        updates={
            "configuration_json": encoded.decode(),
            "configuration_sha256": domain_sha256(module._CONFIG_DOMAIN, encoded),
        },
    )
    with pytest.raises(IrisScientificAblationCheckpointError):
        write(damaged, root=tmp_path / "checkpoints")
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "state",
        "duplicate_json",
        "extra_file",
        "wrong_path",
        "newline",
        "unknown_manifest",
        "truncated",
    ],
)
def test_disk_tampering_is_rejected(
    stepped: IrisScientificAblationCheckpointBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    location = write(stepped, root=tmp_path / "checkpoints")
    path = location / "manifest.json"
    if kind == "state":
        state = location / "state.safetensors"
        content = state.read_bytes()
        state.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif kind == "duplicate_json":
        path.write_bytes(
            b'{"sha256":"' + stepped.manifest.sha256.encode() + b'",' + path.read_bytes()[1:]
        )
    elif kind == "extra_file":
        (location / "foreign.txt").write_text("preserve", encoding="utf-8")
    elif kind == "wrong_path":
        other = location.parent / ("0" * 64)
        location.rename(other)
        location = other
    elif kind == "newline":
        path.write_bytes(path.read_bytes() + b"\n")
    elif kind == "unknown_manifest":
        path.write_bytes(b'{"unknown":1,' + path.read_bytes()[1:])
    else:
        (location / "state.safetensors").write_bytes(b"bad")
    with pytest.raises(IrisScientificAblationCheckpointError, match="could not be read"):
        read(location)


@pytest.mark.parametrize("phase", ["copy", "lr", "commit"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_partial_restore_poison_both_owners_and_preserve_primary_signals(
    stepped: IrisScientificAblationCheckpointBundleV1,
    phase: str,
    error_type: type[BaseException],
) -> None:
    target = _fresh()
    ambient = capture_runtime_state(torch.device("cpu"))
    original = torch.Tensor.copy_
    copies = 0

    def copy(value: torch.Tensor, source: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal copies
        result = original(value, source, *args, **kwargs)
        copies += 1
        if copies == 3:
            raise error_type("PRIVATE candidate copy failure")
        return result

    injection = (
        patch.object(torch.Tensor, "copy_", new=copy)
        if phase == "copy"
        else patch.object(
            module, "scientific_learning_rate_v1", side_effect=error_type("PRIVATE LR failure")
        )
        if phase == "lr"
        else patch.object(target, "_commit_owner", side_effect=error_type("PRIVATE commit failure"))
    )
    expected = (
        error_type
        if error_type in (KeyboardInterrupt, SystemExit)
        else IrisScientificAblationCheckpointError
    )
    with injection, pytest.raises(expected) as caught:
        restore(stepped, run=target)
    assert target.poisoned and target._optimizer.poisoned and not target._busy
    assert target._optimizer._attached_run_token is target._attachment_token
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    if expected is IrisScientificAblationCheckpointError:
        assert "PRIVATE" not in str(caught.value)
    with pytest.raises(IrisScientificAblationCheckpointError):
        build(target)


def test_preflight_failure_restores_caller_without_poisoning_or_mutation(
    stepped: IrisScientificAblationCheckpointBundleV1,
) -> None:
    target = _fresh()
    before = build(target)
    ambient = capture_runtime_state(torch.device("cpu"))

    def fail(_: Any) -> Any:
        random.random()
        torch.rand(3)
        raise ValueError("PRIVATE preflight failure")

    with (
        patch.object(module, "_verify", side_effect=fail),
        pytest.raises(IrisScientificAblationCheckpointError),
    ):
        restore(stepped, run=target)
    assert build(target) == before and not target.poisoned and not target._busy
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("primary", [False, True])
def test_uncertain_caller_restoration_poison_and_signal_priority(
    stepped: IrisScientificAblationCheckpointBundleV1,
    primary: bool,
) -> None:
    target = _fresh()
    expected = KeyboardInterrupt if primary else IrisScientificAblationCheckpointError
    injection = (
        patch.object(target, "_commit_owner", side_effect=KeyboardInterrupt("PRIMARY"))
        if primary
        else patch.object(target, "_commit_owner", wraps=target._commit_owner)
    )
    with (
        injection,
        patch.object(module, "restore_runtime_state", side_effect=ValueError("PRIVATE restore")),
        pytest.raises(expected),
    ):
        restore(stepped, run=target)
    assert target.poisoned and target._optimizer.poisoned and not target._busy


@pytest.mark.parametrize(
    "kind", ["partial", "collision", "lost_ack", "foreign_staging", "conflict"]
)
def test_publication_preserves_unowned_files_and_idempotent_exact_bytes(
    stepped: IrisScientificAblationCheckpointBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    if kind == "collision":
        candidate = root / ".iris-scientific-ablation-collision.tmp"
        candidate.mkdir()
        foreign = candidate / "manifest.json"
        foreign.write_text("foreign", encoding="utf-8")
        with (
            patch.object(module, "uuid4", return_value=SimpleNamespace(hex="collision")),
            pytest.raises(IrisScientificAblationCheckpointError),
        ):
            write(stepped, root=root)
        assert foreign.read_text(encoding="utf-8") == "foreign"
    elif kind in ("partial", "foreign_staging"):
        original = _write_exact_file

        def fail(path: Path, content: bytes) -> None:
            if path.name == "manifest.json":
                if kind == "foreign_staging":
                    (path.parent / "foreign.txt").write_text("foreign", encoding="utf-8")
                raise OSError("PRIVATE publish failure")
            original(path, content)

        with (
            patch.object(module, "_write_exact_file", side_effect=fail),
            pytest.raises(IrisScientificAblationCheckpointError),
        ):
            write(stepped, root=root)
        if kind == "partial":
            assert list(root.iterdir()) == [keep]
        else:
            staging = next(path for path in root.iterdir() if path.is_dir())
            assert (staging / "foreign.txt").read_text(encoding="utf-8") == "foreign"
            assert (staging / "state.safetensors").read_bytes() == stepped.state_bytes
    elif kind == "lost_ack":
        original_rename = rename_directory_with_windows_retry

        def uncertain(source: Path, target: Path) -> None:
            original_rename(source, target)
            raise OSError("PRIVATE lost acknowledgement")

        with patch.object(module, "rename_directory_with_windows_retry", side_effect=uncertain):
            location = write(stepped, root=root)
        assert read(location) == stepped
        assert write(stepped, root=root) == location
        assert {path.name for path in root.iterdir()} == {"keep.txt", stepped.manifest.sha256}
    else:
        location = write(stepped, root=root)
        (location / "manifest.json").write_text("foreign conflict", encoding="utf-8")
        with pytest.raises(IrisScientificAblationCheckpointError):
            write(stepped, root=root)
        assert (location / "manifest.json").read_text(encoding="utf-8") == "foreign conflict"
    assert keep.read_text(encoding="utf-8") == "preserve"


def test_old_and_new_public_io_and_run_types_reject_each_other(tmp_path: Path) -> None:
    old_run, new_run = _old_run(), _fresh()
    old_bundle, new_bundle = build_scientific_checkpoint_v1(old_run), build(new_run)
    with pytest.raises(IrisScientificCheckpointError):
        build_scientific_checkpoint_v1(new_run)  # type: ignore[arg-type]
    with pytest.raises(IrisScientificAblationCheckpointError):
        build(old_run)  # type: ignore[arg-type]
    with pytest.raises(IrisScientificCheckpointError):
        write_scientific_checkpoint_v1(new_bundle, root=tmp_path / "old")  # type: ignore[arg-type]
    with pytest.raises(IrisScientificAblationCheckpointError):
        write(old_bundle, root=tmp_path / "new")  # type: ignore[arg-type]
    with pytest.raises(IrisScientificCheckpointError):
        restore_scientific_checkpoint_v1(new_bundle, run=old_run)  # type: ignore[arg-type]
    with pytest.raises(IrisScientificAblationCheckpointError):
        restore(old_bundle, run=new_run)  # type: ignore[arg-type]
    new_path = write(new_bundle, root=tmp_path / "new")
    old_path = write_scientific_checkpoint_v1(old_bundle, root=tmp_path / "old")
    with pytest.raises(IrisScientificCheckpointError):
        read_scientific_checkpoint_v1(new_path)
    with pytest.raises(IrisScientificAblationCheckpointError):
        read(old_path)
    assert build(new_run) == new_bundle and build_scientific_checkpoint_v1(old_run) == old_bundle


@pytest.mark.parametrize("kind", ["bool", "unknown", "version", "position", "gaussian_nan"])
def test_schema_rejects_unsafe_runtime_fields(
    stepped: IrisScientificAblationCheckpointBundleV1, kind: str
) -> None:
    data = stepped.manifest.payload.model_dump(mode="python")
    if kind == "bool":
        data["completed_steps"] = True
    elif kind == "unknown":
        data["runtime"]["unknown"] = 1
    elif kind == "version":
        data["runtime"]["python_version"] = 2
    elif kind == "position":
        data["runtime"]["python_state"] = (*data["runtime"]["python_state"][:-1], 625)
    else:
        data["runtime"]["python_gaussian"] = float("nan")
    with pytest.raises(ValidationError):
        ScientificAblationCheckpointPayloadV1.model_validate(data)


@pytest.mark.parametrize("kind", ["cursor", "history", "cpu_rng", "gaussian", "moment"])
def test_commit_cannot_reseal_state_that_differs_from_verified_candidate(
    stepped: IrisScientificAblationCheckpointBundleV1,
    kind: str,
) -> None:
    target = _fresh()
    original = target._commit_owner
    ambient = capture_runtime_state(torch.device("cpu"))

    def corrupt() -> None:
        if kind == "cursor":
            target._examples_seen += 1
        elif kind == "history":
            target._history_sha256 = "0" * 64
        elif kind == "cpu_rng":
            target._runtime.cpu_rng_state[0] ^= 1
        elif kind == "gaussian":
            python_state = target._runtime.python_rng_state
            target._runtime = replace(target._runtime, python_rng_state=(*python_state[:2], 0.25))
        else:
            adamw_state = next(iter(target._optimizer._optimizer.state.values()))
            adamw_state["exp_avg"].add_(1)
        original()  # A self-rebound seal alone must not validate these changes.

    with (
        patch.object(target, "_commit_owner", side_effect=corrupt),
        pytest.raises(IrisScientificAblationCheckpointError),
    ):
        restore(stepped, run=target)
    assert target.poisoned and target._optimizer.poisoned and not target._busy
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ["immutable_buffer", "model_shape"])
def test_well_encoded_but_incompatible_model_candidate_fails_before_write(
    stepped: IrisScientificAblationCheckpointBundleV1,
    kind: str,
) -> None:
    payload = stepped.manifest.payload
    tensors = load_tensors(stepped.state_bytes)
    if kind == "immutable_buffer":
        name = next(
            name for name in payload.model_state_names if name not in payload.parameter_names
        )
        tensors[f"model/{name}"].flatten()[0] += 1
    else:
        index = next(
            index
            for index, name in enumerate(payload.parameter_names)
            if tensors[f"model/{name}"].ndim == 2 and payload.parameter_steps[index]
        )
        name = payload.parameter_names[index]
        tensors[f"model/{name}"] = tensors[f"model/{name}"].flatten()
        for suffix in ("exp_avg", "exp_avg_sq"):
            tensors[f"adamw/{index}/{suffix}"] = tensors[f"adamw/{index}/{suffix}"].flatten()
    state_sha = named_module_state_sha256(
        component_names=("scientific_ablation_teacher",),
        state={
            f"scientific_ablation_teacher.{name}": tensors[f"model/{name}"]
            for name in payload.model_state_names
        },
    )
    candidate = _repack(stepped, tensors=tensors, updates={"scientific_state_sha256": state_sha})
    module._verify(candidate)  # All standalone canonical/hash/moment checks really pass.
    target = _fresh()
    before = build(target)
    with pytest.raises(IrisScientificAblationCheckpointError):
        restore(candidate, run=target)
    assert build(target) == before and not target.poisoned and not target._busy


@pytest.mark.parametrize("kind", ["state", "manifest"])
def test_byte_limits_reject_before_tensor_decoder(
    stepped: IrisScientificAblationCheckpointBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    limit = "_MAX_STATE" if kind == "state" else "_MAX_MANIFEST"
    with (
        patch.object(module, limit, 1),
        patch.object(module, "load_tensors") as decoder,
        pytest.raises(IrisScientificAblationCheckpointError),
    ):
        write(stepped, root=tmp_path / "checkpoints")
    decoder.assert_not_called()
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("kind", ["python", "cpu", "optimizer_digest"])
def test_initial_checkpoint_requires_seeded_runtime_and_derived_optimizer_identity(
    tmp_path: Path,
    kind: str,
) -> None:
    initial = build(_fresh())
    updates: dict[str, Any] = {}
    tensors: dict[str, torch.Tensor] | None = None
    if kind == "python":
        updates["runtime"] = initial.manifest.payload.runtime.model_copy(
            update={"python_state": random.Random(1).getstate()[1]}
        )
    elif kind == "cpu":
        tensors = load_tensors(initial.state_bytes)
        tensors["runtime/cpu"] = torch.Generator(device="cpu").manual_seed(1).get_state()
    else:
        configuration = json.loads(initial.manifest.payload.configuration_json)
        configuration["optimizer_sha256"] = "0" * 64
        encoded = canonical_json_bytes(configuration)
        digest = domain_sha256(module._CONFIG_DOMAIN, encoded)
        updates.update(
            configuration_json=encoded.decode(),
            configuration_sha256=digest,
            history_sha256=domain_sha256(module._HISTORY_DOMAIN, digest.encode("ascii")),
        )
    candidate = _repack(initial, updates=updates, tensors=tensors)
    with pytest.raises(IrisScientificAblationCheckpointError):
        write(candidate, root=tmp_path / "checkpoints")
    assert not (tmp_path / "checkpoints").exists()


def test_default_source_sized_disk_checkpoint_resume_crosses_epoch_exactly(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    rows = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=600),
            template_id="iris-synthetic-01-v2",
        )
    )
    policy = IrisScientificOptimizerPolicyV1()

    def owner() -> IrisScientificAblationRunV1:
        return _run(
            "random", goal="disabled", broadcast="disabled", cycles=1, examples=rows, policy=policy
        )

    ambient = capture_runtime_state(torch.device("cpu"))
    source = owner()
    assert source._optimizer._policy == IrisScientificOptimizerPolicyV1()
    counts = [
        int(
            source._backbone.tokenize_teacher_target(
                row.expected_response, terminal_token_id=1
            ).attention_mask.sum()
        )
        for row in rows
    ]

    def window(start: int) -> tuple[tuple[str, ...], int]:
        ids: list[str] = []
        tokens = 0
        while tokens < policy.minimum_target_tokens:
            index = (start + len(ids)) % len(rows)
            ids.append(rows[index].id)
            tokens += counts[index]
        return tuple(ids), tokens

    first_ids, first_tokens = window(0)
    assert len(first_ids) == 452 and first_tokens == 16420
    with (
        _stochastic(source),
        patch.object(source._optimizer, "step", wraps=source._optimizer.step) as step,
    ):
        first = source.step()
    assert tuple(row.id for row in step.call_args.args[0]) == first_ids
    assert first.example_count == 452 and first.supervised_token_count == first_tokens
    saved = build(source)
    path = write(saved, root=tmp_path / "checkpoints")
    resumed = owner()
    with patch("torch.load", side_effect=AssertionError("Incoming pickle forbidden")):
        restore(read(path), run=resumed)
    assert build(resumed) == saved
    second_ids, second_tokens = window(len(first_ids))
    assert len(second_ids) == 452 and second_tokens == 16388
    assert len(set(second_ids)) == len(second_ids)
    assert rows[-1].id in second_ids and rows[0].id in second_ids
    with (
        _stochastic(source),
        patch.object(source._optimizer, "step", wraps=source._optimizer.step) as step,
    ):
        expected = source.step()
    assert tuple(row.id for row in step.call_args.args[0]) == second_ids
    with (
        _stochastic(resumed),
        patch.object(resumed._optimizer, "step", wraps=resumed._optimizer.step) as step,
    ):
        actual = resumed.step()
    assert tuple(row.id for row in step.call_args.args[0]) == second_ids
    assert actual == expected
    _assert_equal(source, resumed)
    assert resumed.examples_seen == 904 and resumed.target_tokens_seen == 32808
    assert build(source) == build(resumed)
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    for name, value in {
        "cycles": 1,
        "mode": "random-goal-disabled-broadcast-disabled",
        "first_examples": 452,
        "first_target_tokens": first_tokens,
        "second_examples": 452,
        "second_target_tokens": second_tokens,
        "examples_seen": resumed.examples_seen,
        "target_tokens_seen": resumed.target_tokens_seen,
        "checkpoint_state_bytes": len(saved.state_bytes),
    }.items():
        record_testsuite_property(f"scientific_ablation_checkpoint_{name}", value)
