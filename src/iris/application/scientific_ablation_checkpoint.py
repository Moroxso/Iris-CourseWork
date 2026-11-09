"""Bounded immutable ablation checkpoints and exact fresh-owner continuation.

Historical scientific-D payloads/bundles remain a separate exact-type boundary.
Recovery validates a complete candidate before writes. Uncertain partial writes
poison both owners; they are not rolled back or silently used as warm starts.
"""

import os
import random
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import torch
from pydantic import Field
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.ablations import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_checkpoint import (
    _bounded_read,
    _clone,
    _json,
    _regular,
    _runtime_record,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizerPolicyV1,
    _same_runtime,
    scientific_learning_rate_v1,
)
from iris.application.scientific_teacher import IrisScientificObjectivePolicyV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_checkpoint_v1 import (
    IrisScientificAblationCheckpointManifestV1,
    ScientificAblationCheckpointPayloadV1,
)
from iris.artifacts.scientific_checkpoint_v1 import (
    Digest,
    ScientificRuntimeV1,
    ScientificTensorV1,
    _Strict,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    named_module_state_sha256,
)
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state

_META = {"format": "iris.scientific-ablation-checkpoint.v1"}
_DOMAIN = "iris/scientific-ablation-checkpoint/v1/payload"
_CONFIG_DOMAIN = "iris/scientific-ablation-run-configuration/v1"
_HISTORY_DOMAIN = "iris/scientific-ablation-run-history/v1"
_MAX_MANIFEST = 16 * 1024**2
_MAX_STATE = 512 * 1024**2
_FILES = {"manifest.json", "state.safetensors"}


class IrisScientificAblationCheckpointError(RuntimeError):
    """A mode-bound checkpoint could not be verified, published or restored."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationCheckpointBundleV1:
    manifest: IrisScientificAblationCheckpointManifestV1
    state_bytes: bytes


class _RuntimePolicy(_Strict):
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


class _RunConfiguration(_Strict):
    kind: Literal["iris-scientific-ablation-run-v1"]
    ordering: Literal["cyclic_fixed_corpus_order_v1"]
    seed: int = Field(ge=0, le=2**63 - 1)
    dataset_sha256: Digest
    dataset_examples: int = Field(ge=1, le=200000)
    dataset_logical_bytes: int = Field(ge=1, le=512 * 1024**2)
    terminal_token_id: int = Field(ge=0, le=2**31 - 1)
    optimizer_sha256: Digest
    optimizer_policy: IrisScientificOptimizerPolicyV1
    teacher_policy: IrisScientificObjectivePolicyV1
    backbone_runtime_sha256: Digest
    software: dict[
        Annotated[str, Field(min_length=1, max_length=128)],
        Annotated[str, Field(min_length=1, max_length=4096)],
    ] = Field(min_length=11, max_length=11)
    ablation: CognitiveAblationPolicyV1
    ablation_policy_sha256: Digest
    model_configuration_sha256: Digest
    initial_scientific_state_sha256: Digest
    runtime_policy: _RuntimePolicy


def _scientific_state(tensors: dict[str, torch.Tensor], names: tuple[str, ...]) -> str:
    return named_module_state_sha256(
        component_names=("scientific_ablation_teacher",),
        state={f"scientific_ablation_teacher.{name}": tensors[f"model/{name}"] for name in names},
    )


def _runtime(record: ScientificRuntimeV1, tensors: dict[str, torch.Tensor]) -> RuntimeStateSnapshot:
    python_state = (record.python_version, tuple(record.python_state), record.python_gaussian)
    random.Random().setstate(python_state)  # Private validation; no global RNG consumption.
    cpu = tensors["runtime/cpu"]
    if cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Invalid ablation CPU RNG state")
    torch.Generator(device="cpu").set_state(cpu)
    cuda = tensors.get("runtime/cuda")
    if cuda is not None and (cuda.dtype != torch.uint8 or cuda.ndim != 1):
        raise ValueError("Invalid ablation CUDA RNG state")
    return RuntimeStateSnapshot(
        python_state,
        cpu.clone(),
        None if record.device == "cpu" else torch.device(record.device),
        None if cuda is None else cuda.clone(),
        record.deterministic_algorithms,
        record.deterministic_warn_only,
        record.cudnn_deterministic,
        record.cudnn_benchmark,
    )


def _verify(
    bundle: IrisScientificAblationCheckpointBundleV1,
) -> tuple[IrisScientificAblationCheckpointManifestV1, dict[str, torch.Tensor]]:
    if (
        type(bundle) is not IrisScientificAblationCheckpointBundleV1
        or type(bundle.manifest) is not IrisScientificAblationCheckpointManifestV1
        or type(bundle.manifest.payload) is not ScientificAblationCheckpointPayloadV1
        or type(bundle.state_bytes) is not bytes
    ):
        raise ValueError("Exact scientific ablation checkpoint required")
    for instance, expected_type in (
        (bundle.manifest, IrisScientificAblationCheckpointManifestV1),
        (bundle.manifest.payload, ScientificAblationCheckpointPayloadV1),
        (bundle.manifest.payload.runtime, ScientificRuntimeV1),
        *((entry, ScientificTensorV1) for entry in bundle.manifest.payload.tensors),
    ):
        # Unchecked model_copy/model_construct may hide unknown fields in model_dump.
        if (
            type(instance) is not expected_type
            or set(vars(instance)) != set(expected_type.model_fields)
            or instance.__pydantic_extra__ is not None
        ):
            raise ValueError("Exact ablation manifest instance fields required")
    manifest = IrisScientificAblationCheckpointManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    payload = manifest.payload
    if (
        len(canonical_json_file_bytes(manifest.model_dump(mode="json"))) > _MAX_MANIFEST
        or not 1 <= len(bundle.state_bytes) <= _MAX_STATE
    ):
        raise ValueError("Ablation checkpoint byte budget exceeded")
    if (
        manifest.sha256
        != domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json")))
        or payload.state_size_bytes != len(bundle.state_bytes)
        or payload.state_sha256 != bytes_sha256(bundle.state_bytes)
        or payload.configuration_sha256
        != domain_sha256(_CONFIG_DOMAIN, payload.configuration_json.encode("utf-8"))
    ):
        raise ValueError("Ablation checkpoint digest differs")
    configuration = _RunConfiguration.model_validate(_json(payload.configuration_json.encode()))
    if (
        configuration.ablation_policy_sha256
        != cognitive_ablation_policy_sha256_v1(configuration.ablation)
        or configuration.optimizer_sha256
        != domain_sha256(
            "iris/scientific-ablation-optimizer-configuration/v1",
            canonical_json_bytes(
                {
                    "kind": "iris-scientific-ablation-optimizer-v1",
                    "policy": configuration.optimizer_policy.model_dump(mode="json"),
                    "model_configuration_sha256": configuration.model_configuration_sha256,
                    "parameter_names": payload.parameter_names,
                    "ablation_policy_sha256": configuration.ablation_policy_sha256,
                    "ablation": configuration.ablation.model_dump(mode="json"),
                }
            ),
        )
        or set(configuration.software)
        != {
            "iris",
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "platform",
            "torch_threads",
            "torch_interop_threads",
            "float32_matmul_precision",
            "mkldnn_enabled",
        }
        or configuration.runtime_policy.model_dump()
        != {name: getattr(payload.runtime, name) for name in _RuntimePolicy.model_fields}
    ):
        raise ValueError("Ablation checkpoint mode/software/runtime policy differs")
    policy = configuration.optimizer_policy
    steps = payload.completed_steps
    if (
        steps > policy.max_optimizer_steps
        or not steps
        <= payload.examples_seen
        <= steps * min(policy.maximum_examples, configuration.dataset_examples)
        or not steps * policy.minimum_target_tokens
        <= payload.target_tokens_seen
        <= steps * policy.maximum_target_tokens
        or (
            not steps
            and (
                payload.history_sha256
                != domain_sha256(_HISTORY_DOMAIN, payload.configuration_sha256.encode("ascii"))
                or payload.scientific_state_sha256 != configuration.initial_scientific_state_sha256
            )
        )
    ):
        raise ValueError("Ablation checkpoint progress differs from numeric policy")
    tensors = load_tensors(bundle.state_bytes)
    if set(tensors) != {entry.key for entry in payload.tensors}:
        raise ValueError("Ablation checkpoint tensor inventory differs")
    for entry in payload.tensors:
        value = tensors[entry.key]
        _clone(value)
        if (
            tuple(value.shape) != entry.shape
            or str(value.dtype).removeprefix("torch.") != entry.dtype
            or value.numel() * value.element_size() != entry.byte_length
            or (entry.key.endswith("/exp_avg_sq") and bool((value < 0).any()))
        ):
            raise ValueError("Ablation tensor descriptor/moment differs")
    for index, (name, count) in enumerate(
        zip(payload.parameter_names, payload.parameter_steps, strict=True)
    ):
        parameter = tensors[f"model/{name}"]
        if parameter.dtype != torch.float32:
            raise ValueError("Ablation parameters require FP32")
        if count:
            for suffix in ("exp_avg", "exp_avg_sq"):
                moment = tensors[f"adamw/{index}/{suffix}"]
                if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError("Ablation moment topology differs")
    if (
        _scientific_state(tensors, payload.model_state_names) != payload.scientific_state_sha256
        or save_tensors(dict(sorted(tensors.items())), metadata=_META) != bundle.state_bytes
    ):
        raise ValueError("Ablation scientific state/canonical encoding differs")
    runtime = _runtime(payload.runtime, tensors)
    if not steps and (
        runtime.python_rng_state != random.Random(configuration.seed).getstate()
        or not torch.equal(
            runtime.cpu_rng_state,
            torch.Generator(device="cpu").manual_seed(configuration.seed).get_state(),
        )
    ):
        raise ValueError("Initial ablation runtime differs from explicit run seed")
    return manifest, tensors


def build_scientific_ablation_checkpoint_v1(
    run: IrisScientificAblationRunV1,
) -> IrisScientificAblationCheckpointBundleV1:
    locked = False
    try:
        if type(run) is not IrisScientificAblationRunV1 or run._busy:
            raise ValueError("Exact idle ablation run required")
        run._validate_boundary()
        run._busy = locked = True
        if run._backbone.checkpoint_runtime_sha256 != run._backbone_identity:
            raise ValueError("Frozen ablation runtime identity differs")
        optimizer = run._optimizer
        tensors = {
            f"model/{name}": _clone(value) for name, value in optimizer._model.state_dict().items()
        }
        for index, (parameter, count) in enumerate(
            zip(optimizer._topology.parameters, optimizer._parameter_steps, strict=True)
        ):
            if count:
                for name in ("exp_avg", "exp_avg_sq"):
                    tensors[f"adamw/{index}/{name}"] = _clone(
                        optimizer._optimizer.state[parameter][name]
                    )
        tensors["runtime/cpu"] = _clone(run._runtime.cpu_rng_state)
        if run._runtime.cuda_rng_state is not None:
            tensors["runtime/cuda"] = _clone(run._runtime.cuda_rng_state)
        if sum(value.numel() * value.element_size() for value in tensors.values()) > _MAX_STATE:
            raise ValueError("Ablation state exceeds its byte budget")
        state = save_tensors(dict(sorted(tensors.items())), metadata=_META)
        payload = ScientificAblationCheckpointPayloadV1(
            configuration_json=run.configuration_json,
            configuration_sha256=run.configuration_sha256,
            completed_steps=run.completed_steps,
            examples_seen=run.examples_seen,
            target_tokens_seen=run.target_tokens_seen,
            history_sha256=run.history_sha256,
            parameter_names=optimizer._topology.names,
            parameter_steps=optimizer._parameter_steps,
            model_state_names=tuple(optimizer._model.state_dict()),
            runtime=_runtime_record(run._runtime, optimizer._device),
            tensors=tuple(
                ScientificTensorV1(
                    key=key,
                    shape=tuple(value.shape),
                    dtype="float32" if value.dtype == torch.float32 else "uint8",
                    byte_length=value.numel() * value.element_size(),
                )
                for key, value in sorted(tensors.items())
            ),
            scientific_state_sha256=optimizer._committed_scientific_state_sha256,
            state_sha256=bytes_sha256(state),
            state_size_bytes=len(state),
        )
        bundle = IrisScientificAblationCheckpointBundleV1(
            IrisScientificAblationCheckpointManifestV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            ),
            state,
        )
        _verify(bundle)
        run._validate_boundary()
        return bundle
    except Exception:
        raise IrisScientificAblationCheckpointError(
            "Scientific ablation checkpoint could not be built"
        ) from None
    finally:
        if locked:
            run._busy = False


def _directory_chain(path: Path) -> None:
    for parent in (path.absolute(), *path.absolute().parents):
        _regular(parent, directory=True)


def read_scientific_ablation_checkpoint_v1(path: Path) -> IrisScientificAblationCheckpointBundleV1:
    try:
        if not isinstance(path, Path):
            raise ValueError("Explicit ablation checkpoint path required")
        _directory_chain(path)
        if {entry.name for entry in path.iterdir()} != _FILES:
            raise ValueError("Ablation checkpoint directory inventory differs")
        content = _bounded_read(path / "manifest.json", _MAX_MANIFEST)
        if not content.endswith(b"\n"):
            raise ValueError("Ablation checkpoint manifest requires one LF")
        _json(content[:-1])
        manifest = IrisScientificAblationCheckpointManifestV1.model_validate_json(content[:-1])
        if (
            canonical_json_file_bytes(manifest.model_dump(mode="json")) != content
            or path.name != manifest.sha256
        ):
            raise ValueError("Ablation checkpoint canonical identity differs")
        bundle = IrisScientificAblationCheckpointBundleV1(
            manifest, _bounded_read(path / "state.safetensors", _MAX_STATE)
        )
        _verify(bundle)
        _directory_chain(path)
        return bundle
    except Exception:
        raise IrisScientificAblationCheckpointError(
            "Scientific ablation checkpoint could not be read"
        ) from None


def write_scientific_ablation_checkpoint_v1(
    bundle: IrisScientificAblationCheckpointBundleV1,
    *,
    root: Path,
) -> Path:
    temporary: Path | None = None
    primary_signal: BaseException | None = None
    try:
        manifest, _ = _verify(bundle)
        if not isinstance(root, Path) or not root.name:
            raise ValueError("Explicit ablation checkpoint root required")
        root = root.absolute()
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        _directory_chain(root)
        target = root / manifest.sha256
        if target.exists():
            if read_scientific_ablation_checkpoint_v1(target) != bundle:
                raise ValueError("Existing ablation checkpoint differs")
            return target
        candidate = root / f".iris-scientific-ablation-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate  # Ownership starts only after successful exclusive creation.
        _write_exact_file(temporary / "state.safetensors", bundle.state_bytes)
        content = canonical_json_file_bytes(manifest.model_dump(mode="json"))
        _write_exact_file(temporary / "manifest.json", content)
        if (
            _bounded_read(temporary / "state.safetensors", _MAX_STATE) != bundle.state_bytes
            or _bounded_read(temporary / "manifest.json", _MAX_MANIFEST) != content
        ):
            raise ValueError("Staged ablation checkpoint bytes differ")
        if {entry.name for entry in temporary.iterdir()} != _FILES:
            raise ValueError("Staged ablation checkpoint inventory differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_ablation_checkpoint_v1(target) != bundle:
                raise
        else:
            temporary = None
        if read_scientific_ablation_checkpoint_v1(target) != bundle:
            raise ValueError("Published ablation checkpoint differs")
        return target
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            primary_signal = error
            raise
        raise IrisScientificAblationCheckpointError(
            "Scientific ablation checkpoint could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            try:
                _directory_chain(temporary)
                if not {entry.name for entry in temporary.iterdir()}.issubset(_FILES):
                    raise ValueError("Unowned staging entry; refusing cleanup")
                for name in sorted(_FILES):
                    path = temporary / name
                    if path.exists():
                        _regular(path, directory=False)
                        path.unlink()
                temporary.rmdir()
            except BaseException:
                if primary_signal is None:
                    raise IrisScientificAblationCheckpointError(
                        "Scientific ablation staging cleanup failed"
                    ) from None


def restore_scientific_ablation_checkpoint_v1(
    bundle: IrisScientificAblationCheckpointBundleV1,
    *,
    run: IrisScientificAblationRunV1,
) -> None:
    _restore_scientific_ablation_checkpoint_v1(bundle, run=run, capability=None)


def _restore_scientific_ablation_checkpoint_v1(
    bundle: IrisScientificAblationCheckpointBundleV1,
    *,
    run: IrisScientificAblationRunV1,
    capability: object | None,
) -> None:
    from iris.application.scientific_ablation_validation import _has_signal

    mutation_started = locked = False
    caller: RuntimeStateSnapshot | None = None
    primary_signal: BaseException | None = None
    try:
        if type(run) is not IrisScientificAblationRunV1 or run._busy:
            raise ValueError("Exact fresh idle ablation run required")
        run._authorize_operation("restore", capability)
        run._validate_boundary()
        optimizer = run._optimizer
        if (
            run.completed_steps
            or run.examples_seen
            or run.target_tokens_seen
            or any(optimizer._parameter_steps)
            or optimizer._optimizer.state
            or optimizer._pending_update_state_sha256 is not None
            or optimizer._active_terminal_token_id is not None
            or optimizer._active_teacher_state_sha256 is not None
        ):
            raise ValueError("Cannot restore over a nonfresh ablation run")
        caller = capture_runtime_state(optimizer._device)
        run._busy = locked = True
        manifest, tensors = _verify(bundle)
        payload = manifest.payload
        if (
            payload.configuration_json != run.configuration_json
            or payload.configuration_sha256 != run.configuration_sha256
            or payload.parameter_names != optimizer._topology.names
            or payload.runtime.device != str(optimizer._device)
            or run._backbone.checkpoint_runtime_sha256 != run._backbone_identity
        ):
            raise ValueError("Ablation checkpoint fresh configuration differs")
        targets = optimizer._model.state_dict()
        if tuple(targets) != payload.model_state_names:
            raise ValueError("Ablation model state inventory differs")
        candidates: dict[str, torch.Tensor] = {}
        for name, target in targets.items():
            source = tensors[f"model/{name}"]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError("Ablation model shape/dtype differs")
            if name not in optimizer._topology.names and not torch.equal(
                source, target.detach().cpu()
            ):
                raise ValueError("Ablation immutable configuration buffer differs")
            candidates[name] = source.to(device=target.device).clone()
        moments: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for index, (parameter, count) in enumerate(
            zip(optimizer._topology.parameters, payload.parameter_steps, strict=True)
        ):
            if count:
                state = {"step": torch.tensor(float(count), dtype=torch.float32, device="cpu")}
                for name in ("exp_avg", "exp_avg_sq"):
                    state[name] = (
                        tensors[f"adamw/{index}/{name}"].to(device=parameter.device).clone()
                    )
                moments[parameter] = state
        runtime = _runtime(payload.runtime, tensors)
        if runtime.cuda_device is not None:
            assert runtime.cuda_rng_state is not None
            torch.Generator(device=runtime.cuda_device).set_state(runtime.cuda_rng_state)
        run._validate_boundary()  # Revalidate immediately before the first write.
        mutation_started = True
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(candidates[name])
        for name, target in targets.items():
            if not torch.equal(target.detach().cpu(), tensors[f"model/{name}"]):
                raise ValueError("Ablation restored tensor bytes differ")
        if optimizer._state_sha256() != payload.scientific_state_sha256:
            raise ValueError("Ablation restored scientific state differs")
        optimizer._optimizer.state.update(moments)
        optimizer._parameter_steps = payload.parameter_steps
        optimizer._completed_steps = payload.completed_steps
        optimizer._current_lr = scientific_learning_rate_v1(
            optimizer._policy, step=max(1, payload.completed_steps)
        )
        optimizer._optimizer.param_groups[0]["lr"] = optimizer._current_lr
        optimizer._versions = tuple(
            parameter._version for parameter in optimizer._topology.parameters
        )
        optimizer._committed_scientific_state_sha256 = payload.scientific_state_sha256
        run._completed_steps = payload.completed_steps
        run._examples_seen = payload.examples_seen
        run._target_tokens_seen = payload.target_tokens_seen
        run._history_sha256 = payload.history_sha256
        run._runtime = runtime
        run._commit_owner()  # Explicit validated restore; preserve attachment/topology identities.
        run._validate_boundary()
        if (
            run._cursor()
            != (
                payload.completed_steps,
                payload.examples_seen,
                payload.target_tokens_seen,
                payload.history_sha256,
            )
            or optimizer._parameter_steps != payload.parameter_steps
            or optimizer._committed_scientific_state_sha256 != payload.scientific_state_sha256
            or not _same_runtime(run._runtime, _runtime(payload.runtime, tensors))
        ):
            raise ValueError("Ablation restored cursor/runtime/state differs from candidate")
        for index, (parameter, count) in enumerate(
            zip(optimizer._topology.parameters, payload.parameter_steps, strict=True)
        ):
            if count:
                for name in ("exp_avg", "exp_avg_sq"):
                    if not torch.equal(
                        optimizer._optimizer.state[parameter][name].detach().cpu(),
                        tensors[f"adamw/{index}/{name}"],
                    ):
                        raise ValueError("Ablation restored moment bytes differ from candidate")
    except BaseException as error:
        if mutation_started:
            run._poisoned = run._optimizer._poisoned = True
        if _has_signal(error):
            primary_signal = error
            raise
        raise IrisScientificAblationCheckpointError(
            "Scientific ablation checkpoint could not be restored"
        ) from None
    finally:
        try:
            if caller is not None:
                restore_runtime_state(caller)
        except BaseException as error:
            run._poisoned = run._optimizer._poisoned = True
            if primary_signal is None:
                if _has_signal(error):
                    raise
                raise IrisScientificAblationCheckpointError(
                    "Scientific ablation caller runtime restoration failed"
                ) from None
        finally:
            if locked:
                run._busy = False


__all__ = [
    "IrisScientificAblationCheckpointBundleV1",
    "IrisScientificAblationCheckpointError",
    "build_scientific_ablation_checkpoint_v1",
    "read_scientific_ablation_checkpoint_v1",
    "restore_scientific_ablation_checkpoint_v1",
    "write_scientific_ablation_checkpoint_v1",
]
