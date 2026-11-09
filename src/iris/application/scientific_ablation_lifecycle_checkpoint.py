"""Bounded ablation current/best lifecycle IO and explicit fresh-controller restore."""

import os
import shutil
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import torch
from pydantic import Field

from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    _directory_chain,
    _restore_scientific_ablation_checkpoint_v1,
    _RunConfiguration,
    _runtime,
    _verify,
)
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_lifecycle_checkpoint_v1 import (
    IrisScientificAblationLifecycleCheckpointManifestV1,
    ScientificAblationLifecycleCheckpointPayloadV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1,
    ScientificAblationLifecycleStateV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

if TYPE_CHECKING:
    from iris.application.scientific_ablation_lifecycle import (
        IrisScientificAblationLifecycleBoundaryV1,
        IrisScientificAblationLifecycleV1,
    )

_DOMAIN = "iris/scientific-ablation-lifecycle-checkpoint/v1"
_CONFIG_DOMAIN = "iris/scientific-ablation-lifecycle-configuration/v1"
_MAX_MANIFEST = 16 * 1024**2
_MAX_STATE = 512 * 1024**2
_FILES = {"manifest.json", "current.safetensors", "best.safetensors"}
_STORE = "scientific-ablation-lifecycle-v1"
_LOCK = ".iris-owner.lock"


class IrisScientificAblationLifecycleCheckpointError(RuntimeError):
    """A lifecycle candidate, filesystem publication or explicit restore failed."""


class _Configuration(_Strict):
    kind: Literal["iris-scientific-ablation-lifecycle-v1"]
    run_sha256: Digest
    ablation_policy_sha256: Digest
    validation_sha256: Digest
    validation_examples: int = Field(ge=1, le=5000)
    validation_target_tokens: int = Field(ge=1, le=2**48)
    policy: IrisScientificAblationLifecyclePolicyV1


@dataclass(frozen=True, slots=True)
class IrisScientificAblationLifecycleBundleV1:
    manifest: IrisScientificAblationLifecycleCheckpointManifestV1
    current_state_bytes: bytes
    best_state_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class IrisScientificAblationLifecycleReceiptV1:
    """Verified publication path and defensive manifest, not an arbitrary label."""

    path: Path
    manifest: IrisScientificAblationLifecycleCheckpointManifestV1


def _bundle_parts(
    bundle: IrisScientificAblationLifecycleBundleV1,
) -> tuple[
    IrisScientificAblationLifecycleCheckpointManifestV1,
    IrisScientificAblationCheckpointBundleV1,
    IrisScientificAblationCheckpointBundleV1 | None,
]:
    from iris.application.scientific_ablation_lifecycle import _exact, _validate_cadence

    if type(bundle) is not IrisScientificAblationLifecycleBundleV1:
        raise ValueError("Exact ablation lifecycle bundle required")
    _exact(bundle.manifest, IrisScientificAblationLifecycleCheckpointManifestV1)
    _exact(bundle.manifest.payload, ScientificAblationLifecycleCheckpointPayloadV1)
    _exact(bundle.manifest.payload.state, ScientificAblationLifecycleStateV1)
    payload = bundle.manifest.payload
    current = IrisScientificAblationCheckpointBundleV1(payload.current, bundle.current_state_bytes)
    _, current_tensors = _verify(current)  # Nested exact guards before parent model_dump.
    best = None
    best_tensors: dict[str, torch.Tensor] | None = None
    if payload.best is not None:
        if payload.best.sha256 == payload.current.sha256:
            if bundle.best_state_bytes is not None or payload.best != payload.current:
                raise ValueError("Shared current/best must have one exact state encoding")
            _, best_tensors = _verify(
                IrisScientificAblationCheckpointBundleV1(payload.best, bundle.current_state_bytes)
            )
            best = current
        else:
            if type(bundle.best_state_bytes) is not bytes:
                raise ValueError("Separate best state required")
            best = IrisScientificAblationCheckpointBundleV1(payload.best, bundle.best_state_bytes)
            _, best_tensors = _verify(best)
    elif bundle.best_state_bytes is not None:
        raise ValueError("Unexpected best state")
    manifest = IrisScientificAblationLifecycleCheckpointManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    if (
        len(canonical_json_file_bytes(manifest.model_dump(mode="json"))) > _MAX_MANIFEST
        or manifest.sha256
        != domain_sha256(_DOMAIN, canonical_json_bytes(manifest.payload.model_dump(mode="json")))
        or payload.configuration_sha256
        != domain_sha256(_CONFIG_DOMAIN, payload.configuration_json.encode())
    ):
        raise ValueError("Ablation lifecycle canonical identity differs")
    configuration = _Configuration.model_validate(_json(payload.configuration_json.encode()))
    run_configuration = _RunConfiguration.model_validate(
        _json(current.manifest.payload.configuration_json.encode())
    )
    if (
        configuration.run_sha256 != current.manifest.payload.configuration_sha256
        or configuration.ablation_policy_sha256 != run_configuration.ablation_policy_sha256
        or configuration.policy.purpose != run_configuration.optimizer_policy.purpose
        or (
            configuration.policy.purpose == "scientific"
            and (
                run_configuration.dataset_examples != 50000
                or configuration.validation_examples != 5000
            )
        )
    ):
        raise ValueError("Ablation lifecycle configuration/mode/purpose differs")
    _validate_cadence(
        payload.state, configuration.policy, run_configuration.optimizer_policy.max_optimizer_steps
    )
    if best is not None:
        if best_tensors is None:
            raise ValueError("Selected-best verified tensors required")
        left, right = current.manifest.payload, best.manifest.payload
        if (
            left.configuration_json != right.configuration_json
            or left.parameter_names != right.parameter_names
            or left.model_state_names != right.model_state_names
            or left.runtime.device != right.runtime.device
            or tuple(item for item in left.tensors if item.key.startswith("model/"))
            != tuple(item for item in right.tensors if item.key.startswith("model/"))
            or any(
                not torch.equal(current_tensors[f"model/{name}"], best_tensors[f"model/{name}"])
                for name in left.model_state_names
                if name not in left.parameter_names
            )
            or right.examples_seen > left.examples_seen
            or right.target_tokens_seen > left.target_tokens_seen
            or any(a < b for a, b in zip(left.parameter_steps, right.parameter_steps, strict=True))
        ):
            raise ValueError(
                "Ablation selected-best configuration/device/topology/progress differs"
            )
    return manifest, current, best


def build_scientific_ablation_lifecycle_checkpoint_v1(
    boundary: "IrisScientificAblationLifecycleBoundaryV1",
) -> IrisScientificAblationLifecycleBundleV1:
    from iris.application.scientific_ablation_lifecycle import (
        IrisScientificAblationLifecycleBoundaryV1,
        _exact,
    )

    try:
        if type(boundary) is not IrisScientificAblationLifecycleBoundaryV1:
            raise ValueError("Exact actual ablation checkpoint boundary required")
        _exact(boundary.state, ScientificAblationLifecycleStateV1)
        current, _ = _verify(boundary.current)
        best = None if boundary.best is None else _verify(boundary.best)[0]
        payload = ScientificAblationLifecycleCheckpointPayloadV1(
            configuration_json=boundary.configuration_json,
            configuration_sha256=boundary.configuration_sha256,
            state=deepcopy(boundary.state),
            current=current,
            best=best,
        )
        manifest = IrisScientificAblationLifecycleCheckpointManifestV1(
            sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        bundle = IrisScientificAblationLifecycleBundleV1(
            manifest,
            boundary.current.state_bytes,
            boundary.best.state_bytes
            if boundary.best is not None and best is not None and best.sha256 != current.sha256
            else None,
        )
        _bundle_parts(bundle)
        return bundle
    except Exception:
        raise IrisScientificAblationLifecycleCheckpointError(
            "Ablation lifecycle boundary rejected"
        ) from None


def _files(bundle: IrisScientificAblationLifecycleBundleV1) -> dict[str, bytes]:
    manifest, _, _ = _bundle_parts(bundle)
    files = {
        "manifest.json": canonical_json_file_bytes(manifest.model_dump(mode="json")),
        "current.safetensors": bundle.current_state_bytes,
    }
    if bundle.best_state_bytes is not None:
        files["best.safetensors"] = bundle.best_state_bytes
    return files


def read_scientific_ablation_lifecycle_checkpoint_v1(
    path: Path,
) -> IrisScientificAblationLifecycleBundleV1:
    try:
        if not isinstance(path, Path):
            raise ValueError("Explicit lifecycle checkpoint path required")
        _directory_chain(path)
        content = _bounded_read(path / "manifest.json", _MAX_MANIFEST)
        if not content.endswith(b"\n"):
            raise ValueError("Canonical manifest requires one LF")
        _json(content[:-1])
        manifest = IrisScientificAblationLifecycleCheckpointManifestV1.model_validate_json(
            content[:-1]
        )
        if (
            path.name != manifest.sha256
            or canonical_json_file_bytes(manifest.model_dump(mode="json")) != content
        ):
            raise ValueError("Lifecycle directory/canonical manifest differs")
        separate = (
            manifest.payload.best is not None
            and manifest.payload.best.sha256 != manifest.payload.current.sha256
        )
        expected = {"manifest.json", "current.safetensors"} | (
            {"best.safetensors"} if separate else set()
        )
        if {entry.name for entry in path.iterdir()} != expected:
            raise ValueError("Lifecycle file inventory differs")
        bundle = IrisScientificAblationLifecycleBundleV1(
            manifest,
            _bounded_read(path / "current.safetensors", _MAX_STATE),
            _bounded_read(path / "best.safetensors", _MAX_STATE) if separate else None,
        )
        _bundle_parts(bundle)
        _directory_chain(path)
        return bundle
    except Exception:
        raise IrisScientificAblationLifecycleCheckpointError(
            "Ablation lifecycle checkpoint could not be read"
        ) from None


def _occupied(root: Path, claim: ProcessFileLock) -> int:
    occupied, count = 0, 0
    for entry in root.iterdir():
        claim.check()
        if entry.name == _LOCK:
            continue
        _directory_chain(entry)
        count += 1
        if count > 4096:
            raise ValueError("Lifecycle store directory bound exceeded")
        for child in entry.iterdir():
            _regular(child, directory=False)
            count += 1
            if count > 16384:
                raise ValueError("Lifecycle store inventory bound exceeded")
            occupied += child.stat().st_size
    return occupied


def write_scientific_ablation_lifecycle_checkpoint_v1(
    bundle: IrisScientificAblationLifecycleBundleV1,
    *,
    root: Path,
) -> Path:
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    primary_signal: BaseException | None = None
    try:
        files = _files(bundle)
        configuration = _Configuration.model_validate(
            _json(bundle.manifest.payload.configuration_json.encode())
        )
        if not isinstance(root, Path) or not root.name:
            raise ValueError("Explicit lifecycle storage root required")
        store = root.absolute() / _STORE
        for parent in (store, *store.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _directory_chain(store)
        with process_file_lock(store / _LOCK) as claim:
            try:
                target = store / bundle.manifest.sha256
                if os.path.lexists(target):
                    if read_scientific_ablation_lifecycle_checkpoint_v1(target) != bundle:
                        raise ValueError("Existing lifecycle checkpoint conflicts")
                    return target
                required = sum(map(len, files.values()))
                if (
                    _occupied(store, claim) + required > configuration.policy.maximum_storage_bytes
                    or shutil.disk_usage(store).free < required + 1024**3
                ):
                    raise ValueError("Lifecycle storage quota/free-space reserve exceeded")
                claim.check()
                candidate = store / f".iris-ablation-lifecycle-{uuid4().hex}.tmp"
                os.mkdir(candidate, mode=0o700)
                temporary = candidate
                metadata = temporary.lstat()
                temporary_identity = metadata.st_dev, metadata.st_ino
                for name, content in files.items():
                    claim.check()
                    _write_exact_file(temporary / name, content)
                    if _bounded_read(temporary / name, len(content)) != content:
                        raise ValueError("Staged lifecycle bytes differ")
                if {entry.name for entry in temporary.iterdir()} != set(files):
                    raise ValueError("Staging inventory differs")
                claim.check()
                try:
                    rename_directory_with_windows_retry(temporary, target)
                except OSError:
                    if (
                        not target.exists()
                        or read_scientific_ablation_lifecycle_checkpoint_v1(target) != bundle
                    ):
                        raise
                else:
                    temporary = None
                if read_scientific_ablation_lifecycle_checkpoint_v1(target) != bundle:
                    raise ValueError("Published lifecycle differs")
                claim.check()
                return target
            except BaseException as error:
                if _has_signal(error):
                    primary_signal = error
                raise
    except BaseException as error:
        if primary_signal is not None:
            raise primary_signal from None
        if _has_signal(error):
            primary_signal = error
            raise
        raise IrisScientificAblationLifecycleCheckpointError(
            "Ablation lifecycle publication failed"
        ) from None
    finally:
        if temporary is not None:
            try:
                if os.path.lexists(temporary):
                    _directory_chain(temporary)
                    metadata = temporary.lstat()
                    if (metadata.st_dev, metadata.st_ino) != temporary_identity:
                        raise ValueError("Unowned replaced staging directory")
                    if not {entry.name for entry in temporary.iterdir()}.issubset(_FILES):
                        raise ValueError("Foreign staging entry; refusing cleanup")
                    for name in sorted(_FILES):
                        path = temporary / name
                        if os.path.lexists(path):
                            _regular(path, directory=False)
                            path.unlink()
                    temporary.rmdir()
            except BaseException as error:
                if primary_signal is None:
                    if _has_signal(error):
                        raise
                    raise IrisScientificAblationLifecycleCheckpointError(
                        "Ablation lifecycle owned cleanup failed"
                    ) from None


def restore_scientific_ablation_lifecycle_v1(
    bundle: IrisScientificAblationLifecycleBundleV1,
    *,
    controller: "IrisScientificAblationLifecycleV1",
) -> None:
    from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1

    copied = False
    try:
        if type(controller) is not IrisScientificAblationLifecycleV1:
            raise ValueError("Exact fresh ablation controller required")
        with controller._operation():
            if (
                controller._state != ScientificAblationLifecycleStateV1()
                or controller._pending is not None
            ):
                raise ValueError("Fresh controller without pending publication required")
            manifest, current, best = _bundle_parts(bundle)
            if (
                manifest.payload.configuration_json != controller.configuration_json
                or manifest.payload.configuration_sha256 != controller.configuration_sha256
                or current.manifest.payload.runtime.device
                != str(controller._run._optimizer._device)
            ):
                raise ValueError("Restored lifecycle configuration differs")
            if best is not None:
                # Offline verification is model-free; CUDA generator usability is
                # target-runtime preflight and must precede every current-run write.
                verified_best, best_tensors = _verify(best)
                best_runtime = _runtime(verified_best.payload.runtime, best_tensors)
                if best_runtime.cuda_device is not None:
                    if best_runtime.cuda_rng_state is None:
                        raise ValueError("Selected-best CUDA state required")
                    torch.Generator(device=best_runtime.cuda_device).set_state(
                        best_runtime.cuda_rng_state
                    )
            state = deepcopy(manifest.payload.state)
            best = deepcopy(best)
            controller._run._busy = False
            try:
                with controller._delegate("restore") as capability:
                    _restore_scientific_ablation_checkpoint_v1(
                        current, run=controller._run, capability=capability
                    )
                copied = True
            finally:
                controller._run._busy = True
            controller._commit(state, best)
    except BaseException as error:
        if type(controller) is IrisScientificAblationLifecycleV1 and (
            copied or controller.poisoned
        ):
            controller._poison()
        if _has_signal(error):
            raise
        raise IrisScientificAblationLifecycleCheckpointError(
            "Ablation lifecycle could not be restored"
        ) from None


__all__ = [
    "IrisScientificAblationLifecycleBundleV1",
    "IrisScientificAblationLifecycleCheckpointError",
    "IrisScientificAblationLifecycleReceiptV1",
    "build_scientific_ablation_lifecycle_checkpoint_v1",
    "read_scientific_ablation_lifecycle_checkpoint_v1",
    "restore_scientific_ablation_lifecycle_v1",
    "write_scientific_ablation_lifecycle_checkpoint_v1",
]
