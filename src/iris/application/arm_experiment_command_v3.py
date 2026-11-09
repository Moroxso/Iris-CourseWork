"""Packaged deterministic profile-v4 B/C/D command over experiment runner v3."""

from __future__ import annotations

from pathlib import Path
from re import fullmatch

import torch

from iris.application import bootstrap as bootstrap_service
from iris.application.arm_profiles import (
    IrisArmExecutionProfileV4,
    build_arm_teacher_forcing_model_v4,
    require_arm_teacher_executable_v4,
)
from iris.application.arm_training_checkpoint import IrisArmTeacherModel, IrisTeacherArmId
from iris.application.errors import ApplicationError, IrisExperimentCommandError
from iris.application.experiment_command import (
    _MAX_EXACT_ADAMW_STEPS,
    _MAX_MANAGED_EVENTS,
    _compose_experiment_config,
    _create_backbone,
    _execution_policy,
    _optimizer_values,
    _own_command,
    _own_config,
    _OwnedExperimentCommand,
    _require_manifest_stable,
    _restore_after_failure,
    _restore_after_success,
    _stage,
    _validate_ambient_training_context,
    _validate_runtime_backbone,
    _validate_schedule,
    _validate_teacher_target_policy,
)
from iris.application.experiment_log_v3 import read_experiment_log_v3
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.experiment_runner_v3 import (
    IrisExperimentExecutionResultV3,
    execute_experiment_v3,
)
from iris.artifacts.experiment_log_v1 import ExperimentEpochExecutionPolicyV1
from iris.artifacts.experiment_log_v3 import (
    ExperimentArmBModelConfigurationV3,
    ExperimentArmCModelConfigurationV3,
    ExperimentArmDModelConfigurationV3,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1
from iris.backbone import BackboneAdapter
from iris.schemas import IrisBootstrapConfig
from iris.training import (
    IrisAdamWConfig,
    IrisNoWorkspaceTeacherForcingModel,
    IrisSharedBufferTeacherForcingModel,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
)
from iris.utils.devices import DeviceInfo
from iris.utils.seeds import apply_runtime_policy, capture_runtime_state


def require_arm_experiment_command_profile_v3(
    arm_id: object,
) -> IrisArmExecutionProfileV4:
    """Resolve B/C/D teacher applicability before manifest, policy, or runtime work."""
    try:
        return require_arm_teacher_executable_v4(arm_id)
    except ApplicationError:
        raise
    except Exception as error:
        message = "Arm-labelled experiment command rejected the execution profile"
        raise IrisExperimentCommandError(message) from error


def _teacher_arm_id(profile: IrisArmExecutionProfileV4) -> IrisTeacherArmId:
    if profile.arm_id not in {"B", "C", "D"}:
        message = "arm-labelled experiment profile did not resolve to B, C, or D"
        raise IrisExperimentCommandError(message)
    return profile.arm_id


def _model_matches_config(
    *,
    arm_id: IrisTeacherArmId,
    model: object,
    config: IrisBootstrapConfig,
) -> bool:
    if arm_id == "B":
        return (
            type(model) is ExperimentArmBModelConfigurationV3
            and model.backbone == config.backbone
            and model.cognitive == config.cognitive
        )
    if arm_id == "C":
        return (
            type(model) is ExperimentArmCModelConfigurationV3
            and model.backbone == config.backbone
            and model.cognitive == config.cognitive
        )
    return (
        type(model) is ExperimentArmDModelConfigurationV3
        and model.backbone == config.backbone
        and model.cognitive == config.cognitive
        and model.workspace == config.workspace
    )


def _preflight_existing_run_v3(
    *,
    command: _OwnedExperimentCommand,
    profile: IrisArmExecutionProfileV4,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
    config: IrisBootstrapConfig,
    policy: ExperimentEpochExecutionPolicyV1,
) -> None:
    path = command.root / "experiments" / "v3" / command.run_id
    if not path.exists():
        return
    snapshot = read_experiment_log_v3(path)
    run = snapshot.run.payload
    if (
        run.run_id != command.run_id
        or run.arm_id != arm_id
        or run.arm_kind != profile.arm_kind
        or run.request_graph != profile.request_graph
        or run.teacher_graph != profile.teacher_graph
    ):
        message = "requested arm-labelled experiment identity conflicts with the installed run"
        raise ValueError(message)
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
        reference.dataset_id != manifest.payload.dataset_id
        or reference.payload_sha256 != manifest.payload_sha256
        or reference.split_counts != manifest.payload.split_counts
        or training.target_preparation.terminal_token_id != command.terminal_token_id
        or training.runtime_policy.initialization_seed != command.initialization_seed
        or not training.runtime_policy.deterministic
        or training.execution_policy != policy
        or not _model_matches_config(arm_id=arm_id, model=training.model, config=config)
        or existing_optimizer.optimizer_type != "torch.optim.AdamW"
        or existing_values != _optimizer_values(command.optimizer_config)
    ):
        message = "requested arm-labelled experiment identity conflicts with the installed run"
        raise ValueError(message)


def _expected_model_type(
    arm_id: IrisTeacherArmId,
) -> type[torch.nn.Module]:
    expected_types: dict[IrisTeacherArmId, type[torch.nn.Module]] = {
        "B": IrisNoWorkspaceTeacherForcingModel,
        "C": IrisSharedBufferTeacherForcingModel,
        "D": IrisTeacherForcingModel,
    }
    return expected_types[arm_id]


def _validate_pristine_training_owner_v3(
    *,
    arm_id: IrisTeacherArmId,
    backbone: BackboneAdapter,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    optimizer_config: IrisAdamWConfig,
) -> None:
    expected_model_type = _expected_model_type(arm_id)
    if type(model) is not expected_model_type or model.backbone is not backbone:
        message = f"arm {arm_id} teacher model does not own the loaded backbone"
        raise TypeError(message)
    if not model.training:
        message = f"arm {arm_id} teacher model must begin in training mode"
        raise ValueError(message)
    if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
        message = "arm-labelled optimizer does not own the composed teacher model"
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
        message = "arm-labelled model and optimizer did not form a stable pristine owner"
        raise ValueError(message)


def _validate_live_training_owner_v3(
    *,
    arm_id: IrisTeacherArmId,
    config: IrisBootstrapConfig,
    backbone: BackboneAdapter,
    device_info: DeviceInfo,
    dtype: torch.dtype,
    terminal_token_id: int,
    model: IrisArmTeacherModel,
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
        message = "arm-labelled experiment changed the deterministic-algorithm policy"
        raise ValueError(message)
    if torch.is_deterministic_algorithms_warn_only_enabled():
        message = "arm-labelled experiment enabled deterministic warn-only behavior"
        raise ValueError(message)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark:
        message = "arm-labelled experiment changed the deterministic cuDNN policy"
        raise ValueError(message)
    if torch.initial_seed() != initialization_seed:
        message = "arm-labelled experiment changed the process torch initial seed"
        raise ValueError(message)
    expected_model_type = _expected_model_type(arm_id)
    if (
        type(model) is not expected_model_type
        or model.backbone is not backbone
        or optimizer.model is not model
        or not model.training
    ):
        message = "arm-labelled experiment changed the live training owner"
        raise ValueError(message)
    before = model.checkpoint_provenance
    optimizer.provenance_snapshot()
    after = model.checkpoint_provenance
    if before != after:
        message = "arm-labelled experiment state changed during final observation"
        raise ValueError(message)


def _validate_execution_result_v3(
    result: object,
    *,
    command: _OwnedExperimentCommand,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
) -> IrisExperimentExecutionResultV3:
    if type(result) is not IrisExperimentExecutionResultV3:
        message = "arm-labelled experiment runner returned an invalid result type"
        raise TypeError(message)
    expected_path = command.root / "experiments" / "v3" / command.run_id
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
    if result.arm_id != arm_id or result.path != expected_path:
        message = "arm-labelled experiment runner returned the wrong arm or run path"
        raise ValueError(message)
    if any(type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        message = "arm-labelled experiment runner returned an invalid SHA-256 identity"
        raise ValueError(message)
    if any(type(value) is not int or value < 0 for value in counters):
        message = "arm-labelled experiment runner returned an invalid progress counter"
        raise ValueError(message)
    if (
        result.event_count > _MAX_MANAGED_EVENTS
        or result.optimizer_completed_steps > _MAX_EXACT_ADAMW_STEPS
        or result.completed_epochs > command.epoch_count
        or result.latest_checkpoint_schema_version != {"B": 2, "C": 3, "D": 1}[arm_id]
    ):
        message = "arm-labelled experiment runner returned progress outside requested bounds"
        raise ValueError(message)
    if command.maximum_new_training_steps is None:
        if result.status != "complete":
            message = "unbounded arm-labelled execution did not complete"
            raise ValueError(message)
    elif result.new_training_steps > command.maximum_new_training_steps:
        message = "arm-labelled experiment runner exceeded the requested training-step limit"
        raise ValueError(message)
    if result.status == "complete":
        if result.next_epoch_index is not None or result.next_training_ordinal is not None:
            message = "complete arm-labelled result contains a live cursor"
            raise ValueError(message)
    elif result.status == "step_limit_reached":
        if (
            type(result.next_epoch_index) is not int
            or not 1 <= result.next_epoch_index <= command.epoch_count
            or type(result.next_training_ordinal) is not int
            or not 1 <= result.next_training_ordinal <= manifest.payload.split_counts.train
        ):
            message = "bounded arm-labelled result contains an invalid live cursor"
            raise ValueError(message)
    else:
        message = "arm-labelled experiment runner returned an invalid status"
        raise ValueError(message)
    return result


def run_packaged_arm_experiment_v3(
    *,
    arm_id: object,
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
) -> IrisExperimentExecutionResultV3:
    """Compose, execute or resume one deterministic profile-v4 B/C/D runtime."""
    profile = require_arm_experiment_command_profile_v3(arm_id)
    teacher_arm_id = _teacher_arm_id(profile)
    command = _stage(
        "arm-labelled input validation",
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
        "arm-labelled manifest verification",
        lambda: read_experiment_manifest_v1(command.manifest_path),
    )
    _stage(
        "arm-labelled schedule validation",
        lambda: _validate_schedule(command=command, manifest=manifest),
    )
    policy = _stage(
        "arm-labelled execution-policy construction",
        lambda: _execution_policy(command),
    )
    composed = _stage(
        "arm-labelled packaged configuration composition",
        lambda: _compose_experiment_config(
            allow_download=command.allow_download,
            initialization_seed=command.initialization_seed,
        ),
    )
    config = _stage(
        "arm-labelled configuration validation",
        lambda: _own_config(composed, command=command),
    )
    _stage(
        "arm-labelled existing-run preflight",
        lambda: _preflight_existing_run_v3(
            command=command,
            profile=profile,
            arm_id=teacher_arm_id,
            manifest=manifest,
            config=config,
            policy=policy,
        ),
    )
    _stage(
        "arm-labelled CUDA deterministic preflight",
        lambda: bootstrap_service._validate_cuda_determinism(config),
    )
    device_info, dtype = _stage(
        "arm-labelled device resolution",
        lambda: bootstrap_service._resolve_device(config),
    )
    _stage(
        "arm-labelled ambient training-context preflight",
        lambda: _validate_ambient_training_context(device_info),
    )
    runtime_snapshot = _stage(
        "arm-labelled runtime-state capture",
        lambda: capture_runtime_state(device_info.device),
    )

    try:
        _stage(
            "arm-labelled runtime-policy application",
            lambda: apply_runtime_policy(
                seed=command.initialization_seed,
                deterministic=True,
                device=device_info.device,
            ),
        )
        backbone = _stage(
            "arm-labelled backbone loading",
            lambda: _create_backbone(config, device_info),
        )
        _stage(
            "arm-labelled backbone validation",
            lambda: _validate_runtime_backbone(
                backbone,
                config=config,
                device_info=device_info,
                dtype=dtype,
                terminal_token_id=command.terminal_token_id,
            ),
        )
        _stage(
            "arm-labelled teacher-target policy validation",
            lambda: _validate_teacher_target_policy(
                backbone,
                terminal_token_id=command.terminal_token_id,
            ),
        )
        model = _stage(
            "arm-labelled teacher-model composition",
            lambda: build_arm_teacher_forcing_model_v4(
                arm_id=teacher_arm_id,
                backbone=backbone,
                cognitive_config=config.cognitive,
                workspace_config=config.workspace if teacher_arm_id == "D" else None,
            ),
        )
        optimizer = _stage(
            "arm-labelled optimizer construction",
            lambda: IrisTeacherOptimizer(model=model, config=command.optimizer_config),
        )
        _stage(
            "arm-labelled pristine training-owner validation",
            lambda: _validate_pristine_training_owner_v3(
                arm_id=teacher_arm_id,
                backbone=backbone,
                model=model,
                optimizer=optimizer,
                optimizer_config=command.optimizer_config,
            ),
        )
        _stage(
            "arm-labelled pre-execution manifest stability check",
            lambda: _require_manifest_stable(command.manifest_path, manifest),
        )
        raw_result = _stage(
            "arm-labelled managed execution",
            lambda: execute_experiment_v3(
                run_id=command.run_id,
                arm_id=teacher_arm_id,
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
            "arm-labelled execution-result validation",
            lambda: _validate_execution_result_v3(
                raw_result,
                command=command,
                arm_id=teacher_arm_id,
                manifest=manifest,
            ),
        )
        _stage(
            "arm-labelled post-execution manifest stability check",
            lambda: _require_manifest_stable(command.manifest_path, manifest),
        )
        _stage(
            "arm-labelled final runtime validation",
            lambda: _validate_live_training_owner_v3(
                arm_id=teacher_arm_id,
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


__all__ = [
    "require_arm_experiment_command_profile_v3",
    "run_packaged_arm_experiment_v3",
]
