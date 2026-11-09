"""Packaged deterministic command service for one managed Iris experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from re import fullmatch

import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from iris.application import bootstrap as bootstrap_service
from iris.application.errors import IrisExperimentCommandError
from iris.application.experiment_log import read_experiment_log_v1
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.experiment_runner import (
    IrisExperimentExecutionResult,
    execute_experiment_v1,
)
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    validate_experiment_run_id_v1,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1
from iris.backbone import BackboneAdapter
from iris.schemas import IrisBootstrapConfig
from iris.training import IrisAdamWConfig, IrisTeacherForcingModel, IrisTeacherOptimizer
from iris.utils.devices import DeviceInfo
from iris.utils.seeds import (
    RuntimeStateSnapshot,
    apply_runtime_policy,
    capture_runtime_state,
    restore_runtime_state,
)

_MAX_EXACT_ADAMW_STEPS = 2**24
_MAX_MANAGED_EVENTS = 1_000_000
_MAX_EPOCH_COUNT = 1_000_000
_MAX_SEED = 2**63 - 1


@dataclass(frozen=True, slots=True)
class _OwnedExperimentCommand:
    manifest_path: Path
    root: Path
    run_id: str
    terminal_token_id: int
    initialization_seed: int
    epoch_count: int
    evaluate_test_after_training: bool
    optimizer_config: IrisAdamWConfig
    maximum_new_training_steps: int | None
    allow_download: bool


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisExperimentCommandError:
        raise
    except Exception as error:
        message = f"Iris experiment command failed during {name}"
        raise IrisExperimentCommandError(message) from error


def _owned_path(value: object, *, label: str, suffix: str | None = None) -> Path:
    if not isinstance(value, Path) or not value.name:
        message = f"{label} must be an explicit Path"
        raise TypeError(message)
    if suffix is not None and value.suffix != suffix:
        message = f"{label} must have the {suffix} suffix"
        raise ValueError(message)
    return Path(value)


def _owned_optimizer_config(value: object) -> IrisAdamWConfig:
    if type(value) is not IrisAdamWConfig:
        message = "optimizer_config must be exactly IrisAdamWConfig"
        raise TypeError(message)
    return IrisAdamWConfig(
        learning_rate=value.learning_rate,
        beta1=value.beta1,
        beta2=value.beta2,
        epsilon=value.epsilon,
        weight_decay=value.weight_decay,
        max_gradient_norm=value.max_gradient_norm,
    )


def _own_command(
    *,
    manifest_path: object,
    root: object,
    run_id: object,
    terminal_token_id: object,
    initialization_seed: object,
    epoch_count: object,
    evaluate_test_after_training: object,
    optimizer_config: object,
    maximum_new_training_steps: object,
    allow_download: object,
) -> _OwnedExperimentCommand:
    owned_manifest_path = _owned_path(manifest_path, label="manifest_path", suffix=".json")
    owned_root = _owned_path(root, label="root")
    if owned_root.exists() and (owned_root.is_symlink() or not owned_root.is_dir()):
        message = "root must be absent or a real directory"
        raise ValueError(message)
    owned_run_id = validate_experiment_run_id_v1(run_id)
    if type(terminal_token_id) is not int or terminal_token_id < 0:
        message = "terminal_token_id must be an exact nonnegative integer"
        raise ValueError(message)
    if (
        type(initialization_seed) is not int
        or initialization_seed < 0
        or initialization_seed > _MAX_SEED
    ):
        message = "initialization_seed must be an exact integer in [0, 2^63 - 1]"
        raise ValueError(message)
    if type(epoch_count) is not int or not 1 <= epoch_count <= _MAX_EPOCH_COUNT:
        message = "epoch_count must be an exact integer in [1, 1_000_000]"
        raise ValueError(message)
    if type(evaluate_test_after_training) is not bool:
        message = "evaluate_test_after_training must be an explicit boolean"
        raise TypeError(message)
    if maximum_new_training_steps is not None and (
        type(maximum_new_training_steps) is not int or maximum_new_training_steps < 1
    ):
        message = "maximum_new_training_steps must be an exact positive integer"
        raise ValueError(message)
    if type(allow_download) is not bool:
        message = "allow_download must be an explicit boolean"
        raise TypeError(message)
    return _OwnedExperimentCommand(
        manifest_path=owned_manifest_path,
        root=owned_root,
        run_id=owned_run_id,
        terminal_token_id=terminal_token_id,
        initialization_seed=initialization_seed,
        epoch_count=epoch_count,
        evaluate_test_after_training=evaluate_test_after_training,
        optimizer_config=_owned_optimizer_config(optimizer_config),
        maximum_new_training_steps=maximum_new_training_steps,
        allow_download=allow_download,
    )


def _execution_policy(
    command: _OwnedExperimentCommand,
) -> ExperimentEpochExecutionPolicyV1:
    return ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=command.epoch_count,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=command.evaluate_test_after_training,
        resume_policy="latest_checkpoint_plus_verified_replay",
    )


def _validate_schedule(
    *,
    command: _OwnedExperimentCommand,
    manifest: IrisExperimentManifestV1,
) -> None:
    counts = manifest.payload.split_counts
    if command.evaluate_test_after_training and counts.test == 0:
        message = "evaluate_test_after_training requires a nonempty test split"
        raise ValueError(message)
    final_step = command.epoch_count * counts.train
    if final_step > _MAX_EXACT_ADAMW_STEPS:
        message = "managed experiment exceeds the exact AdamW step range"
        raise ValueError(message)
    event_count = command.epoch_count * (counts.train + 2)
    if command.evaluate_test_after_training:
        event_count += 1
    if event_count > _MAX_MANAGED_EVENTS:
        message = "managed experiment exceeds the event safety limit"
        raise ValueError(message)


def _compose_experiment_config(
    *,
    allow_download: bool,
    initialization_seed: int,
) -> DictConfig:
    local_files_only = "false" if allow_download else "true"
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        config = compose(
            config_name="config",
            overrides=[
                f"backbone.local_files_only={local_files_only}",
                f"runtime.seed={initialization_seed}",
            ],
        )
    if not isinstance(config, DictConfig):
        message = "Packaged Iris experiment configuration did not compose to a DictConfig"
        raise TypeError(message)
    return config


def _own_config(
    config: DictConfig,
    *,
    command: _OwnedExperimentCommand,
) -> IrisBootstrapConfig:
    if not isinstance(config, DictConfig):
        message = "config must be a composed DictConfig"
        raise TypeError(message)
    snapshot = OmegaConf.to_yaml(config, resolve=False, sort_keys=False)
    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if type(container) is not dict:
        message = "Composed Hydra root must be a mapping"
        raise TypeError(message)
    owned = IrisBootstrapConfig.model_validate(container)
    if OmegaConf.to_yaml(config, resolve=False, sort_keys=False) != snapshot:
        message = "Packaged experiment configuration changed during validation"
        raise ValueError(message)
    if owned.runtime.seed != command.initialization_seed:
        message = "Packaged experiment configuration ignored the requested seed"
        raise ValueError(message)
    if not owned.runtime.deterministic:
        message = "Packaged experiment execution requires deterministic=true"
        raise ValueError(message)
    if owned.backbone.local_files_only is command.allow_download:
        message = "Packaged experiment configuration ignored the download gate"
        raise ValueError(message)
    return owned


def _optimizer_values(config: IrisAdamWConfig) -> tuple[float, ...]:
    return (
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.epsilon,
        config.weight_decay,
        config.max_gradient_norm,
    )


def _preflight_existing_run(
    *,
    command: _OwnedExperimentCommand,
    manifest: IrisExperimentManifestV1,
    config: IrisBootstrapConfig,
    policy: ExperimentEpochExecutionPolicyV1,
) -> None:
    path = command.root / "experiments" / "v1" / command.run_id
    if not path.exists():
        return
    snapshot = read_experiment_log_v1(path)
    run = snapshot.run.payload
    reference = run.manifest
    training = run.configuration
    existing_optimizer = training.optimizer
    existing_values = (
        existing_optimizer.config.learning_rate,
        existing_optimizer.config.beta1,
        existing_optimizer.config.beta2,
        existing_optimizer.config.epsilon,
        existing_optimizer.config.weight_decay,
        existing_optimizer.config.max_gradient_norm,
    )
    if (
        run.run_id != command.run_id
        or reference.dataset_id != manifest.payload.dataset_id
        or reference.payload_sha256 != manifest.payload_sha256
        or reference.split_counts != manifest.payload.split_counts
        or training.target_preparation.terminal_token_id != command.terminal_token_id
        or training.runtime_policy.initialization_seed != command.initialization_seed
        or not training.runtime_policy.deterministic
        or training.execution_policy != policy
        or training.model.backbone != config.backbone
        or training.model.cognitive != config.cognitive
        or training.model.workspace != config.workspace
        or existing_optimizer.optimizer_type != "torch.optim.AdamW"
        or existing_values != _optimizer_values(command.optimizer_config)
    ):
        message = "requested experiment identity conflicts with the installed run"
        raise ValueError(message)


def _require_manifest_stable(
    path: Path,
    expected: IrisExperimentManifestV1,
) -> None:
    observed = read_experiment_manifest_v1(path)
    if observed != expected:
        message = "experiment manifest changed during command execution"
        raise ValueError(message)


def _create_backbone(config: IrisBootstrapConfig, device_info: DeviceInfo) -> BackboneAdapter:
    return BackboneAdapter(config.backbone, device_info)


def _validate_ambient_training_context(device_info: DeviceInfo) -> None:
    if torch.is_inference_mode_enabled():
        message = "managed experiment cannot run under torch.inference_mode"
        raise ValueError(message)
    if not torch.is_grad_enabled():
        message = "managed experiment requires autograd to be enabled"
        raise ValueError(message)
    try:
        autocast_enabled = torch.is_autocast_enabled(device_info.device.type)
    except Exception as error:
        message = "managed experiment could not inspect ambient autocast state"
        raise ValueError(message) from error
    if autocast_enabled:
        message = "managed experiment does not support ambient automatic mixed precision"
        raise ValueError(message)


def _validate_runtime_backbone(
    backbone: BackboneAdapter,
    *,
    config: IrisBootstrapConfig,
    device_info: DeviceInfo,
    dtype: torch.dtype,
    terminal_token_id: int,
) -> None:
    bootstrap_service._validate_backbone(
        backbone,
        config=config,
        device_info=device_info,
        dtype=dtype,
        stop_token_ids=(terminal_token_id,),
    )


def _validate_teacher_target_policy(
    backbone: BackboneAdapter,
    *,
    terminal_token_id: int,
) -> None:
    eos_token_ids = backbone.eos_token_ids
    if terminal_token_id not in eos_token_ids:
        message = "terminal_token_id must be one declared tokenizer EOS"
        raise ValueError(message)
    if backbone.max_new_tokens < 2:
        message = "backbone target limit cannot contain content plus its terminal"
        raise ValueError(message)


def _validate_pristine_training_owner(
    *,
    backbone: BackboneAdapter,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    optimizer_config: IrisAdamWConfig,
) -> None:
    if not isinstance(model, IrisTeacherForcingModel) or model.backbone is not backbone:
        message = "teacher model does not own the loaded backbone"
        raise TypeError(message)
    if not model.training:
        message = "teacher model must begin in training mode"
        raise ValueError(message)
    if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
        message = "optimizer does not own the composed teacher model"
        raise TypeError(message)
    before = model.checkpoint_provenance
    provenance = optimizer.provenance_snapshot()
    after = model.checkpoint_provenance
    if (
        before != after
        or provenance.completed_steps != 0
        or optimizer.completed_steps != 0
        or provenance.config != optimizer_config
    ):
        message = "teacher model and optimizer did not form a stable pristine owner"
        raise ValueError(message)


def _validate_live_training_owner(
    *,
    config: IrisBootstrapConfig,
    backbone: BackboneAdapter,
    device_info: DeviceInfo,
    dtype: torch.dtype,
    terminal_token_id: int,
    model: IrisTeacherForcingModel,
    optimizer: IrisTeacherOptimizer,
    initialization_seed: int,
) -> None:
    _validate_runtime_backbone(
        backbone,
        config=config,
        device_info=device_info,
        dtype=dtype,
        terminal_token_id=terminal_token_id,
    )
    if torch.are_deterministic_algorithms_enabled() is not True:
        message = "managed experiment changed the deterministic-algorithm policy"
        raise ValueError(message)
    if torch.is_deterministic_algorithms_warn_only_enabled():
        message = "managed experiment enabled deterministic warn-only behavior"
        raise ValueError(message)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark:
        message = "managed experiment changed the deterministic cuDNN policy"
        raise ValueError(message)
    if torch.initial_seed() != initialization_seed:
        message = "managed experiment changed the process torch initial seed"
        raise ValueError(message)
    if model.backbone is not backbone or optimizer.model is not model or not model.training:
        message = "managed experiment changed the live training owner"
        raise ValueError(message)
    before = model.checkpoint_provenance
    optimizer.provenance_snapshot()
    after = model.checkpoint_provenance
    if before != after:
        message = "managed experiment state changed during final observation"
        raise ValueError(message)


def _validate_execution_result(
    result: object,
    *,
    command: _OwnedExperimentCommand,
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentExecutionResult:
    if type(result) is not IrisExperimentExecutionResult:
        message = "experiment runner returned an invalid result type"
        raise TypeError(message)
    expected_path = command.root / "experiments" / "v1" / command.run_id
    hashes = (
        result.run_payload_sha256,
        result.configuration_sha256,
        result.latest_checkpoint_payload_sha256,
    )
    counters = (
        result.event_count,
        result.optimizer_completed_steps,
        result.completed_epochs,
        result.replayed_training_steps,
        result.new_training_steps,
    )
    if result.path != expected_path:
        message = "experiment runner returned the wrong run path"
        raise ValueError(message)
    if any(type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        message = "experiment runner returned an invalid SHA-256 identity"
        raise ValueError(message)
    if any(type(value) is not int or value < 0 for value in counters):
        message = "experiment runner returned an invalid progress counter"
        raise ValueError(message)
    if (
        result.event_count > _MAX_MANAGED_EVENTS
        or result.optimizer_completed_steps > _MAX_EXACT_ADAMW_STEPS
        or result.completed_epochs > command.epoch_count
    ):
        message = "experiment runner returned progress outside requested bounds"
        raise ValueError(message)
    if command.maximum_new_training_steps is None:
        if result.status != "complete":
            message = "unbounded managed execution did not complete"
            raise ValueError(message)
    elif result.new_training_steps > command.maximum_new_training_steps:
        message = "experiment runner exceeded the requested training-step limit"
        raise ValueError(message)
    if result.status == "complete":
        if result.next_epoch_index is not None or result.next_training_ordinal is not None:
            message = "complete experiment result contains a live cursor"
            raise ValueError(message)
    elif result.status == "step_limit_reached":
        if (
            type(result.next_epoch_index) is not int
            or not 1 <= result.next_epoch_index <= command.epoch_count
            or type(result.next_training_ordinal) is not int
            or not 1 <= result.next_training_ordinal <= manifest.payload.split_counts.train
        ):
            message = "bounded experiment result contains an invalid live cursor"
            raise ValueError(message)
    else:
        message = "experiment runner returned an invalid status"
        raise ValueError(message)
    return result


def _restore_after_failure(
    snapshot: RuntimeStateSnapshot,
    primary_error: BaseException,
) -> None:
    try:
        restore_runtime_state(snapshot)
    except BaseException as rollback_error:
        failures = BaseExceptionGroup(
            "Iris experiment command failed and runtime rollback was incomplete",
            [primary_error, rollback_error],
        )
        message = "Iris experiment command failed and process runtime state is uncertain"
        raise IrisExperimentCommandError(message) from failures


def _restore_after_success(snapshot: RuntimeStateSnapshot) -> None:
    try:
        restore_runtime_state(snapshot)
    except BaseException as error:
        message = "Iris experiment completed but process runtime state is uncertain"
        raise IrisExperimentCommandError(message) from error


def run_packaged_experiment_v1(
    *,
    manifest_path: Path,
    root: Path,
    run_id: str,
    terminal_token_id: int,
    initialization_seed: int,
    epoch_count: int,
    evaluate_test_after_training: bool,
    optimizer_config: IrisAdamWConfig,
    maximum_new_training_steps: int | None,
    allow_download: bool,
) -> IrisExperimentExecutionResult:
    """Compose, execute or resume, and restore one deterministic process runtime."""
    command = _stage(
        "input validation",
        lambda: _own_command(
            manifest_path=manifest_path,
            root=root,
            run_id=run_id,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            epoch_count=epoch_count,
            evaluate_test_after_training=evaluate_test_after_training,
            optimizer_config=optimizer_config,
            maximum_new_training_steps=maximum_new_training_steps,
            allow_download=allow_download,
        ),
    )
    manifest = _stage(
        "manifest verification",
        lambda: read_experiment_manifest_v1(command.manifest_path),
    )
    _stage(
        "schedule validation",
        lambda: _validate_schedule(command=command, manifest=manifest),
    )
    policy = _stage("execution-policy construction", lambda: _execution_policy(command))
    composed = _stage(
        "packaged configuration composition",
        lambda: _compose_experiment_config(
            allow_download=command.allow_download,
            initialization_seed=command.initialization_seed,
        ),
    )
    config = _stage(
        "configuration validation",
        lambda: _own_config(composed, command=command),
    )
    _stage(
        "existing-run preflight",
        lambda: _preflight_existing_run(
            command=command,
            manifest=manifest,
            config=config,
            policy=policy,
        ),
    )
    _stage(
        "CUDA deterministic preflight",
        lambda: bootstrap_service._validate_cuda_determinism(config),
    )
    device_info, dtype = _stage(
        "device resolution",
        lambda: bootstrap_service._resolve_device(config),
    )
    _stage(
        "ambient training-context preflight",
        lambda: _validate_ambient_training_context(device_info),
    )
    runtime_snapshot = _stage(
        "runtime-state capture",
        lambda: capture_runtime_state(device_info.device),
    )

    try:
        _stage(
            "runtime-policy application",
            lambda: apply_runtime_policy(
                seed=command.initialization_seed,
                deterministic=True,
                device=device_info.device,
            ),
        )
        backbone = _stage(
            "backbone loading",
            lambda: _create_backbone(config, device_info),
        )
        _stage(
            "backbone validation",
            lambda: _validate_runtime_backbone(
                backbone,
                config=config,
                device_info=device_info,
                dtype=dtype,
                terminal_token_id=command.terminal_token_id,
            ),
        )
        _stage(
            "teacher-target policy validation",
            lambda: _validate_teacher_target_policy(
                backbone,
                terminal_token_id=command.terminal_token_id,
            ),
        )
        model = _stage(
            "teacher-model composition",
            lambda: build_teacher_forcing_model(
                backbone=backbone,
                cognitive_config=config.cognitive,
                workspace_config=config.workspace,
            ),
        )
        optimizer = _stage(
            "optimizer construction",
            lambda: IrisTeacherOptimizer(
                model=model,
                config=command.optimizer_config,
            ),
        )
        _stage(
            "pristine training-owner validation",
            lambda: _validate_pristine_training_owner(
                backbone=backbone,
                model=model,
                optimizer=optimizer,
                optimizer_config=command.optimizer_config,
            ),
        )
        _stage(
            "pre-execution manifest stability check",
            lambda: _require_manifest_stable(command.manifest_path, manifest),
        )
        raw_result = _stage(
            "managed execution",
            lambda: execute_experiment_v1(
                run_id=command.run_id,
                manifest=manifest,
                terminal_token_id=command.terminal_token_id,
                initialization_seed=command.initialization_seed,
                deterministic=True,
                execution_policy=policy,
                model=model,
                optimizer=optimizer,
                root=command.root,
                maximum_new_training_steps=command.maximum_new_training_steps,
            ),
        )
        result = _stage(
            "execution-result validation",
            lambda: _validate_execution_result(
                raw_result,
                command=command,
                manifest=manifest,
            ),
        )
        _stage(
            "post-execution manifest stability check",
            lambda: _require_manifest_stable(command.manifest_path, manifest),
        )
        _stage(
            "final runtime validation",
            lambda: _validate_live_training_owner(
                config=config,
                backbone=backbone,
                device_info=device_info,
                dtype=dtype,
                terminal_token_id=command.terminal_token_id,
                model=model,
                optimizer=optimizer,
                initialization_seed=command.initialization_seed,
            ),
        )
    except BaseException as error:
        _restore_after_failure(runtime_snapshot, error)
        raise

    _restore_after_success(runtime_snapshot)
    return result


__all__ = ["run_packaged_experiment_v1"]
