"""Validation, best-state selection, scheduled checkpoints and early stopping."""

import json
import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _bounded_read,
    _json,
    _regular,
    _verify,
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_optimization import IrisScientificOptimizationOutputV1
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_validation import (
    IrisScientificValidationV1,
    evaluate_scientific_validation_v1,
    own_scientific_validation_rows_v1,
)
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_lifecycle_v1 import (
    IrisScientificLifecycleCheckpointManifestV1,
    IrisScientificLifecyclePolicyV1,
    ScientificLifecycleCheckpointPayloadV1,
    ScientificLifecycleStateV1,
)
from iris.data.schemas import IrisExample
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-lifecycle-checkpoint/v1"
_CONFIG_DOMAIN = "iris/scientific-lifecycle-configuration/v1"


class IrisScientificLifecycleError(RuntimeError):
    """Lifecycle state, validation or persistence could not complete."""


@dataclass(frozen=True, slots=True)
class IrisScientificLifecycleBundleV1:
    manifest: IrisScientificLifecycleCheckpointManifestV1
    current_state_bytes: bytes
    best_state_bytes: bytes | None


def _bundle_parts(
    bundle: IrisScientificLifecycleBundleV1,
) -> tuple[IrisScientificCheckpointBundleV1, IrisScientificCheckpointBundleV1 | None]:
    if type(bundle) is not IrisScientificLifecycleBundleV1:
        raise ValueError("Exact lifecycle bundle required")
    manifest = IrisScientificLifecycleCheckpointManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python", warnings=False)
    )
    payload = manifest.payload
    if manifest.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
    ) or payload.configuration_sha256 != domain_sha256(
        _CONFIG_DOMAIN, payload.configuration_json.encode()
    ):
        raise ValueError("Lifecycle checkpoint identity differs")
    current = IrisScientificCheckpointBundleV1(payload.current, bundle.current_state_bytes)
    _verify(current)
    configuration = _json(payload.configuration_json.encode())
    if (
        type(configuration) is not dict
        or set(configuration)
        != {"kind", "run_sha256", "validation_sha256", "validation_examples", "policy"}
        or configuration["kind"] != "iris-scientific-lifecycle-v1"
        or configuration["run_sha256"] != current.manifest.payload.configuration_sha256
        or type(configuration["validation_examples"]) is not int
        or not 1 <= configuration["validation_examples"] <= 5000
        or type(configuration["validation_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", configuration["validation_sha256"]) is None
    ):
        raise ValueError("Lifecycle configuration structure differs")
    IrisScientificLifecyclePolicyV1.model_validate(configuration["policy"])
    best = None
    if payload.best is not None:
        if payload.best.sha256 == payload.current.sha256:
            if bundle.best_state_bytes is not None:
                raise ValueError("Shared best/current state must not be duplicated")
            best = current
        else:
            if type(bundle.best_state_bytes) is not bytes:
                raise ValueError("Missing separate best state")
            best = IrisScientificCheckpointBundleV1(payload.best, bundle.best_state_bytes)
            _verify(best)
    elif bundle.best_state_bytes is not None:
        raise ValueError("Unexpected best state")
    return current, best


def _files(bundle: IrisScientificLifecycleBundleV1) -> dict[str, bytes]:
    _bundle_parts(bundle)
    files = {
        "manifest.json": canonical_json_file_bytes(bundle.manifest.model_dump(mode="json")),
        "current.safetensors": bundle.current_state_bytes,
    }
    if bundle.best_state_bytes is not None:
        files["best.safetensors"] = bundle.best_state_bytes
    if len(files["manifest.json"]) > 16 * 1024**2:
        raise ValueError("Lifecycle manifest byte budget exceeded")
    return files


def read_scientific_lifecycle_checkpoint_v1(path: Path) -> IrisScientificLifecycleBundleV1:
    try:
        _regular(path, directory=True)
        content = _bounded_read(path / "manifest.json", 16 * 1024**2)
        manifest = IrisScientificLifecycleCheckpointManifestV1.model_validate_json(content)
        if (
            canonical_json_file_bytes(manifest.model_dump(mode="json")) != content
            or path.name != manifest.sha256
        ):
            raise ValueError("Noncanonical lifecycle manifest")
        payload = manifest.payload
        separate = payload.best is not None and payload.best.sha256 != payload.current.sha256
        expected = {"manifest.json", "current.safetensors"} | (
            {"best.safetensors"} if separate else set()
        )
        if {entry.name for entry in path.iterdir()} != expected:
            raise ValueError("Lifecycle file inventory differs")
        bundle = IrisScientificLifecycleBundleV1(
            manifest,
            _bounded_read(path / "current.safetensors", 512 * 1024**2),
            _bounded_read(path / "best.safetensors", 512 * 1024**2) if separate else None,
        )
        _bundle_parts(bundle)
        return bundle
    except Exception:
        raise IrisScientificLifecycleError("Lifecycle checkpoint could not be read") from None


def _publish(bundle: IrisScientificLifecycleBundleV1, root: Path, *, budget: int) -> Path:
    temporary: Path | None = None
    try:
        files = _files(bundle)
        root = root.absolute() / "scientific-lifecycle-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        _regular(root, directory=True)
        target = root / bundle.manifest.sha256
        if target.exists():
            if read_scientific_lifecycle_checkpoint_v1(target) != bundle:
                raise ValueError("Existing lifecycle checkpoint conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        required = sum(map(len, files.values()))
        if occupied + required > budget or shutil.disk_usage(root).free < required + 1024**3:
            raise ValueError("Lifecycle storage budget/free-space reserve exceeded")
        staging_candidate = root / f".iris-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary = staging_candidate  # Cleanup ownership begins only after exclusive creation.
        for name, content in files.items():
            _write_exact_file(temporary / name, content)
            if _bounded_read(temporary / name, len(content)) != content:
                raise ValueError("Staged lifecycle bytes differ")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_lifecycle_checkpoint_v1(target) != bundle:
                raise
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificLifecycleError("Lifecycle checkpoint publication failed") from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            for name in ("manifest.json", "current.safetensors", "best.safetensors"):
                (temporary / name).unlink(missing_ok=True)
            temporary.rmdir()


class IrisScientificLifecycleV1:
    """Advance one owned phase at a time; publication failure never retrains a window."""

    def __init__(
        self,
        *,
        run: IrisScientificRunV1,
        validation: tuple[IrisExample, ...],
        policy: IrisScientificLifecyclePolicyV1,
    ) -> None:
        if type(run) is not IrisScientificRunV1 or run._busy or run.completed_steps:
            raise IrisScientificLifecycleError("A fresh idle scientific run is required")
        run._validate_boundary()
        self._policy = IrisScientificLifecyclePolicyV1.model_validate(
            policy.model_dump(mode="python")
        )
        if self._policy.purpose != run._optimizer._policy.purpose:
            raise IrisScientificLifecycleError("Lifecycle and optimizer purposes differ")
        rows, digest = own_scientific_validation_rows_v1(validation, training_rows=run._rows)
        if self._policy.purpose == "scientific" and (len(run._rows) != 50000 or len(rows) != 5000):
            raise IrisScientificLifecycleError(
                "Scientific lifecycle requires source 50000/5000 splits"
            )
        self._run = run
        self._validation = rows
        self._configuration_json = canonical_json_bytes(
            {
                "kind": "iris-scientific-lifecycle-v1",
                "run_sha256": run.configuration_sha256,
                "validation_sha256": digest,
                "validation_examples": len(rows),
                "policy": self._policy.model_dump(mode="json"),
            }
        ).decode()
        self._configuration_sha256 = domain_sha256(
            _CONFIG_DOMAIN, self._configuration_json.encode()
        )
        self._state = ScientificLifecycleStateV1()
        self._best: IrisScientificCheckpointBundleV1 | None = None
        self._pending: IrisScientificLifecycleBundleV1 | None = None
        self._poisoned = False
        self._busy = False

    @property
    def state(self) -> ScientificLifecycleStateV1:
        return ScientificLifecycleStateV1.model_validate(self._state.model_dump(mode="python"))

    @property
    def best_checkpoint(self) -> IrisScientificCheckpointBundleV1 | None:
        return self._best

    def _validate(self) -> None:
        if self._poisoned or self._run.completed_steps != self._state.step:
            raise IrisScientificLifecycleError("Lifecycle no longer owns the run boundary")
        self._run._validate_boundary()
        _ = self.state
        if self._policy.model_dump(mode="json") != json.loads(self._configuration_json)["policy"]:
            raise IrisScientificLifecycleError("Lifecycle policy changed")

    def advance(
        self, *, checkpoint_root: Path
    ) -> IrisScientificOptimizationOutputV1 | IrisScientificValidationV1 | Path:
        if self._busy:
            raise IrisScientificLifecycleError("Reentrant lifecycle operation forbidden")
        self._validate()
        if self._state.phase == "stopped":
            raise IrisScientificLifecycleError("Scientific lifecycle has stopped")
        self._busy = True
        try:
            state = self._state
            if state.phase == "train":
                result = self._run.step()
                step = result.step_index
                final = step == self._run._optimizer._policy.max_optimizer_steps
                phase = (
                    "validation"
                    if final or step % self._policy.validate_every_steps == 0
                    else "checkpoint"
                    if step % self._policy.checkpoint_every_steps == 0
                    else "train"
                )
                self._state = ScientificLifecycleStateV1.model_validate(
                    state.model_dump(mode="python") | {"step": step, "phase": phase}
                )
                return result
            if state.phase == "validation":
                report = evaluate_scientific_validation_v1(self._run, self._validation)
                better = (
                    state.best_nll_per_token is None
                    or report.target_nll_per_token
                    < state.best_nll_per_token - self._policy.minimum_improvement
                )
                best = build_scientific_checkpoint_v1(self._run) if better else self._best
                misses = 0 if better else state.nonimproving_validations + 1
                reason = (
                    "early_stopping"
                    if misses >= self._policy.early_stopping_patience
                    else "step_budget"
                    if state.step == self._run._optimizer._policy.max_optimizer_steps
                    else None
                )
                phase = (
                    "checkpoint"
                    if reason is not None or state.step % self._policy.checkpoint_every_steps == 0
                    else "train"
                )
                self._state = ScientificLifecycleStateV1.model_validate(
                    state.model_dump(mode="python")
                    | {
                        "phase": phase,
                        "validation_count": state.validation_count + 1,
                        "last_validation_step": state.step,
                        "best_step": state.step if better else state.best_step,
                        "best_nll_per_token": report.target_nll_per_token
                        if better
                        else state.best_nll_per_token,
                        "nonimproving_validations": misses,
                        "stop_reason": reason,
                    }
                )
                self._best = best
                return report
            if self._pending is None:
                current = build_scientific_checkpoint_v1(self._run)
                following = ScientificLifecycleStateV1.model_validate(
                    state.model_dump(mode="python")
                    | {
                        "phase": "stopped" if state.stop_reason is not None else "train",
                        "checkpoint_count": state.checkpoint_count + 1,
                        "last_checkpoint_step": state.step,
                    }
                )
                payload = ScientificLifecycleCheckpointPayloadV1(
                    configuration_json=self._configuration_json,
                    configuration_sha256=self._configuration_sha256,
                    state=following,
                    current=current.manifest,
                    best=None if self._best is None else self._best.manifest,
                )
                manifest = IrisScientificLifecycleCheckpointManifestV1(
                    sha256=domain_sha256(
                        _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                    payload=payload,
                )
                best_bytes = (
                    None if self._best is None or self._best == current else self._best.state_bytes
                )
                self._pending = IrisScientificLifecycleBundleV1(
                    manifest, current.state_bytes, best_bytes
                )
            path = _publish(
                self._pending, checkpoint_root, budget=self._policy.maximum_storage_bytes
            )
            self._validate()
            self._state = self._pending.manifest.payload.state
            self._pending = None
            return path
        except BaseException as error:
            if self._run.poisoned or self._run.completed_steps != self._state.step:
                self._poisoned = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificLifecycleError(
                "Scientific lifecycle phase could not complete"
            ) from None
        finally:
            self._busy = False


def restore_scientific_lifecycle_v1(
    bundle: IrisScientificLifecycleBundleV1, *, lifecycle: IrisScientificLifecycleV1
) -> None:
    copied = False
    locked = False
    try:
        if (
            type(lifecycle) is not IrisScientificLifecycleV1
            or lifecycle._busy
            or lifecycle.state != ScientificLifecycleStateV1()
        ):
            raise ValueError("Fresh idle lifecycle required")
        lifecycle._validate()
        lifecycle._busy = locked = True
        current, best = _bundle_parts(bundle)
        payload = bundle.manifest.payload
        if (
            payload.configuration_sha256 != lifecycle._configuration_sha256
            or payload.configuration_json != lifecycle._configuration_json
        ):
            raise ValueError("Lifecycle configuration differs")
        policy, state = lifecycle._policy, payload.state
        if (state.nonimproving_validations >= policy.early_stopping_patience) != (
            state.stop_reason == "early_stopping"
        ):
            raise ValueError("Lifecycle stopping state differs from policy")
        maximum = lifecycle._run._optimizer._policy.max_optimizer_steps
        if state.step > maximum or (state.step == maximum and state.phase != "stopped"):
            raise ValueError("Lifecycle step budget differs")
        terminal = state.phase == "stopped"
        expected_validations = state.step // policy.validate_every_steps + int(
            terminal and state.step % policy.validate_every_steps != 0
        )
        expected_checkpoints = state.step // policy.checkpoint_every_steps + int(
            terminal and state.step % policy.checkpoint_every_steps != 0
        )
        expected_last_validation = (
            state.step
            if terminal
            else state.step // policy.validate_every_steps * policy.validate_every_steps
        )
        if (
            state.validation_count != expected_validations
            or state.checkpoint_count != expected_checkpoints
            or state.last_validation_step != expected_last_validation
            or (state.stop_reason == "step_budget" and state.step != maximum)
            or state.nonimproving_validations > policy.early_stopping_patience
        ):
            raise ValueError("Lifecycle cadence counters differ from policy")
        if state.best_step is not None:
            best_index = state.best_step // policy.validate_every_steps + int(
                state.best_step == maximum and state.best_step % policy.validate_every_steps != 0
            )
            if (
                state.best_step % policy.validate_every_steps and state.best_step != maximum
            ) or state.nonimproving_validations != state.validation_count - best_index:
                raise ValueError("Lifecycle staleness differs from best validation")
        if best is not None:
            left, right = current.manifest.payload, best.manifest.payload
            if (
                left.parameter_names != right.parameter_names
                or left.model_state_names != right.model_state_names
                or tuple(t for t in left.tensors if t.key.startswith("model/"))
                != tuple(t for t in right.tensors if t.key.startswith("model/"))
            ):
                raise ValueError("Best checkpoint topology differs from current run")
        restore_scientific_checkpoint_v1(current, run=lifecycle._run)
        copied = True
        lifecycle._best = best
        lifecycle._state = state
        lifecycle._validate()
    except BaseException as error:
        if type(lifecycle) is IrisScientificLifecycleV1 and (copied or lifecycle._run.poisoned):
            lifecycle._poisoned = True
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise IrisScientificLifecycleError("Lifecycle checkpoint could not be restored") from None
    finally:
        if locked:
            lifecycle._busy = False


__all__ = [
    "IrisScientificLifecycleBundleV1",
    "IrisScientificLifecycleError",
    "IrisScientificLifecycleV1",
    "read_scientific_lifecycle_checkpoint_v1",
    "restore_scientific_lifecycle_v1",
]
