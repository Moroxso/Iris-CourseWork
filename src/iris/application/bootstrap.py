"""Exclusive startup transaction for one configured Iris request runtime."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from re import fullmatch
from typing import Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf

from iris.application.arm_profiles import (
    ArmRequestOrchestratorV3,
    ArmRequestOrchestratorV4,
    IrisArmExecutionProfileV3,
    IrisArmExecutionProfileV4,
    arm_execution_profile_set_sha256_v3,
    arm_execution_profile_set_sha256_v4,
    build_arm_request_orchestrator_v3,
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v3,
    require_arm_request_executable_v4,
)
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.composition import build_request_orchestrator
from iris.application.errors import IrisBootstrapError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import IrisRequestOrchestrator
from iris.application.shared_buffer_request import CognitiveSharedBufferRequestOrchestrator
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import BackboneConfig, IrisBootstrapConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.utils.devices import DeviceInfo, inspect_device, resolve_torch_dtype
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import (
    apply_runtime_policy,
    capture_runtime_state,
    restore_runtime_state,
)


@dataclass(frozen=True, slots=True)
class IrisRequestRuntime:
    """Validated objects and committed policy for one request process."""

    backbone: BackboneAdapter
    orchestrator: IrisRequestOrchestrator
    device_info: DeviceInfo
    dtype: torch.dtype
    seed: int
    deterministic: bool
    configuration_json: str
    configuration_sha256: str
    sidecar_state_sha256: str


@dataclass(frozen=True, slots=True)
class IrisArmRequestRuntimeV2:
    """Validated A/B/D request runtime bound to execution profile set v3."""

    profile: IrisArmExecutionProfileV3
    execution_profile_set_sha256: Literal[
        "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
    ]
    backbone: BackboneAdapter
    orchestrator: ArmRequestOrchestratorV3
    device_info: DeviceInfo
    dtype: torch.dtype
    seed: int
    deterministic: bool
    configuration_json: str
    configuration_sha256: str
    sidecar_state_sha256: str | None


@dataclass(frozen=True, slots=True)
class IrisArmRequestRuntimeV3:
    """Validated A/B/C/D request runtime bound to execution profile set v4."""

    profile: IrisArmExecutionProfileV4
    execution_profile_set_sha256: Literal[
        "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
    ]
    backbone: BackboneAdapter
    orchestrator: ArmRequestOrchestratorV4
    device_info: DeviceInfo
    dtype: torch.dtype
    seed: int
    deterministic: bool
    configuration_json: str
    configuration_sha256: str
    sidecar_state_sha256: str | None


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisBootstrapError:
        raise
    except Exception as error:
        message = f"Iris request bootstrap failed during {name}"
        raise IrisBootstrapError(message) from error


def _snapshot_input(config: DictConfig | IrisBootstrapConfig) -> str:
    if isinstance(config, DictConfig):
        return OmegaConf.to_yaml(config, resolve=False, sort_keys=False)
    if isinstance(config, IrisBootstrapConfig):
        return config.model_dump_json()
    message = "config must be a composed DictConfig or validated IrisBootstrapConfig"
    raise TypeError(message)


def _canonical_config_json(config: IrisBootstrapConfig) -> str:
    return canonical_json_bytes(config.model_dump(mode="json")).decode("utf-8")


def _own_config(
    config: DictConfig | IrisBootstrapConfig,
) -> tuple[IrisBootstrapConfig, str]:
    input_snapshot = _snapshot_input(config)
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
        )
        if not isinstance(container, dict):
            message = "Composed Hydra root must be a mapping"
            raise TypeError(message)
    else:
        container = config.model_dump(mode="python")
    owned = IrisBootstrapConfig.model_validate(container)
    if _snapshot_input(config) != input_snapshot:
        message = "Bootstrap configuration changed during validation"
        raise ValueError(message)
    return owned, input_snapshot


def _validate_generation_policy(
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> tuple[float, float]:
    if type(stop_token_ids) is not tuple:
        message = "stop_token_ids must be an explicit tuple"
        raise TypeError(message)
    if any(type(token_id) is not int for token_id in stop_token_ids):
        message = "stop_token_ids must contain only integers"
        raise ValueError(message)
    if len(set(stop_token_ids)) != len(stop_token_ids):
        message = "stop_token_ids must be unique"
        raise ValueError(message)
    if any(token_id < 0 for token_id in stop_token_ids):
        message = "stop_token_ids cannot contain a negative ID"
        raise ValueError(message)
    if type(sampling) is not TemperatureTopPSamplingConfig:
        message = "sampling must be TemperatureTopPSamplingConfig"
        raise TypeError(message)
    temperature = sampling.temperature
    top_p = sampling.top_p
    if type(temperature) is not float or not isfinite(temperature) or temperature <= 0.0:
        message = "sampling temperature must be a finite float greater than zero"
        raise ValueError(message)
    if type(top_p) is not float or not isfinite(top_p) or not 0.0 < top_p <= 1.0:
        message = "sampling top_p must be a finite float in (0, 1]"
        raise ValueError(message)
    return temperature, top_p


def _validate_cuda_determinism(config: IrisBootstrapConfig) -> None:
    if config.hardware.device != "cuda" or not config.runtime.deterministic:
        return
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {":4096:8", ":16:8"}:
        message = (
            "Deterministic CUDA bootstrap requires CUBLAS_WORKSPACE_CONFIG "
            "to be ':4096:8' or ':16:8' before device initialization"
        )
        raise ValueError(message)
    cuda_is_initialized = cast("Callable[[], bool]", torch.cuda.is_initialized)
    if cuda_is_initialized():
        message = (
            "Deterministic CUDA bootstrap must run before CUDA initialization "
            "so the cuBLAS workspace policy is effective"
        )
        raise ValueError(message)


def _resolve_device(config: IrisBootstrapConfig) -> tuple[DeviceInfo, torch.dtype]:
    device_info = inspect_device(config.hardware.device)
    if not isinstance(device_info, DeviceInfo):
        message = "Device inspection did not return DeviceInfo"
        raise TypeError(message)
    if device_info.backend != config.hardware.device:
        message = "Device inspection returned the wrong backend"
        raise ValueError(message)
    if not isinstance(device_info.device, torch.device):
        message = "DeviceInfo must contain a torch.device"
        raise TypeError(message)
    if device_info.device.type != device_info.backend:
        message = "DeviceInfo device disagrees with its backend"
        raise ValueError(message)
    if device_info.backend == "cpu":
        if device_info.device != torch.device("cpu"):
            message = "CPU bootstrap requires the canonical indexless CPU device"
            raise ValueError(message)
    elif device_info.device.index is None or device_info.device.index < 0:
        message = "CUDA bootstrap requires one concrete nonnegative device index"
        raise ValueError(message)
    dtype = resolve_torch_dtype(device_info, config.hardware.precision)
    return device_info, dtype


def _read_sidecar_state_sha256(orchestrator: IrisRequestOrchestrator) -> str:
    digest = orchestrator.sidecar_state_sha256
    if type(digest) is not str or fullmatch(r"[0-9a-f]{64}", digest) is None:
        message = "Request graph returned an invalid sidecar state SHA-256"
        raise ValueError(message)
    return digest


def _read_arm_sidecar_state_sha256(
    *,
    profile: IrisArmExecutionProfileV3,
    orchestrator: ArmRequestOrchestratorV3,
) -> str | None:
    if profile.arm_id == "A":
        if type(orchestrator) is not BaseLlmRequestOrchestrator:
            message = "Arm-A request composition returned the wrong orchestrator type"
            raise TypeError(message)
        return None
    expected_type = (
        CognitiveNoWorkspaceRequestOrchestrator
        if profile.arm_id == "B"
        else IrisRequestOrchestrator
    )
    if type(orchestrator) is not expected_type:
        message = f"Arm-{profile.arm_id} request composition returned the wrong orchestrator type"
        raise TypeError(message)
    digest = orchestrator.sidecar_state_sha256
    if type(digest) is not str or fullmatch(r"[0-9a-f]{64}", digest) is None:
        message = f"Arm-{profile.arm_id} request graph returned an invalid sidecar SHA-256"
        raise ValueError(message)
    return digest


_PROFILE_SET_SHA256_V4: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_SHARED_BUFFER_POLICY_SHA256_V1: Literal[
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
] = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"


def _preflight_arm_request_v4(arm_id: object) -> IrisArmExecutionProfileV4:
    """Resolve all immutable v4 identities before inspecting runtime inputs."""
    try:
        profile = require_arm_request_executable_v4(arm_id)
        profile_set_sha256 = arm_execution_profile_set_sha256_v4()
        if (
            type(profile_set_sha256) is not str
            or profile_set_sha256 != _PROFILE_SET_SHA256_V4
        ):
            message = "Arm execution profile set v4 identity changed unexpectedly"
            raise ValueError(message)
        policy_sha256 = shared_buffer_policy_sha256_v1()
        if (
            type(policy_sha256) is not str
            or policy_sha256 != _SHARED_BUFFER_POLICY_SHA256_V1
        ):
            message = "Arm-C shared-buffer policy v1 identity changed unexpectedly"
            raise ValueError(message)
        return profile
    except Exception as error:
        message = "Arm-labelled request bootstrap v3 rejected the execution profile preflight"
        raise IrisBootstrapError(message) from error


def _read_arm_sidecar_state_sha256_v4(
    *,
    profile: IrisArmExecutionProfileV4,
    orchestrator: ArmRequestOrchestratorV4,
    backbone: BackboneAdapter,
) -> str | None:
    """Validate one exact v4 graph and return its immutable sidecar identity."""
    if type(profile) is not IrisArmExecutionProfileV4:
        message = "Arm request profile must be exactly IrisArmExecutionProfileV4"
        raise TypeError(message)
    expected_type: type[
        BaseLlmRequestOrchestrator
        | CognitiveNoWorkspaceRequestOrchestrator
        | CognitiveSharedBufferRequestOrchestrator
        | IrisRequestOrchestrator
    ]
    if profile.arm_id == "A":
        expected_type = BaseLlmRequestOrchestrator
    elif profile.arm_id == "B":
        expected_type = CognitiveNoWorkspaceRequestOrchestrator
    elif profile.arm_id == "C":
        expected_type = CognitiveSharedBufferRequestOrchestrator
    else:
        expected_type = IrisRequestOrchestrator
    if type(orchestrator) is not expected_type:
        message = f"Arm-{profile.arm_id} request composition returned the wrong orchestrator type"
        raise TypeError(message)
    if orchestrator.backbone is not backbone:
        message = f"Arm-{profile.arm_id} request graph does not own the loaded backbone"
        raise ValueError(message)
    if profile.arm_id == "A":
        return None
    sidecar_orchestrator: (
        CognitiveNoWorkspaceRequestOrchestrator
        | CognitiveSharedBufferRequestOrchestrator
        | IrisRequestOrchestrator
    )
    if profile.arm_id == "B":
        sidecar_orchestrator = cast("CognitiveNoWorkspaceRequestOrchestrator", orchestrator)
    elif profile.arm_id == "C":
        shared_buffer = cast("CognitiveSharedBufferRequestOrchestrator", orchestrator)
        sidecar_orchestrator = shared_buffer
        policy_sha256 = shared_buffer.shared_buffer_policy_sha256
        if (
            type(policy_sha256) is not str
            or policy_sha256 != _SHARED_BUFFER_POLICY_SHA256_V1
            or policy_sha256 != shared_buffer_policy_sha256_v1()
        ):
            message = "Arm-C request graph returned a non-canonical shared-buffer policy"
            raise ValueError(message)
    else:
        sidecar_orchestrator = cast("IrisRequestOrchestrator", orchestrator)
    digest = sidecar_orchestrator.sidecar_state_sha256
    if type(digest) is not str or fullmatch(r"[0-9a-f]{64}", digest) is None:
        message = f"Arm-{profile.arm_id} request graph returned an invalid sidecar SHA-256"
        raise ValueError(message)
    return digest


def _create_backbone(config: BackboneConfig, device_info: DeviceInfo) -> BackboneAdapter:
    return BackboneAdapter(config, device_info)


def _validate_backbone(
    backbone: BackboneAdapter,
    *,
    config: IrisBootstrapConfig,
    device_info: DeviceInfo,
    dtype: torch.dtype,
    stop_token_ids: tuple[int, ...],
) -> None:
    if not isinstance(backbone, BackboneAdapter):
        message = "Backbone construction did not return BackboneAdapter"
        raise TypeError(message)
    if backbone.device != device_info.device:
        message = "Loaded backbone is not on the inspected concrete device"
        raise ValueError(message)
    if backbone.dtype != dtype:
        message = "Loaded backbone dtype disagrees with hardware precision"
        raise ValueError(message)
    if type(backbone.max_prompt_tokens) is not int or (
        backbone.max_prompt_tokens != config.backbone.limits.max_prompt_tokens
    ):
        message = "Loaded backbone prompt limit disagrees with configuration"
        raise ValueError(message)
    if type(backbone.max_new_tokens) is not int or (
        backbone.max_new_tokens != config.backbone.limits.max_new_tokens
    ):
        message = "Loaded backbone generation limit disagrees with configuration"
        raise ValueError(message)
    vocabulary_size = backbone.vocabulary_size
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        message = "Loaded backbone vocabulary size must be a positive integer"
        raise ValueError(message)
    if any(token_id >= vocabulary_size for token_id in stop_token_ids):
        message = "stop_token_ids contains an out-of-vocabulary ID"
        raise ValueError(message)
    if type(backbone.trainable_parameter_count) is not int or (
        backbone.trainable_parameter_count != 0
    ):
        message = "Loaded backbone must not contain trainable parameters"
        raise ValueError(message)
    if backbone.has_parameter_gradients is not False:
        message = "Loaded backbone parameters must not contain gradients"
        raise ValueError(message)


def _validate_stability(
    original: DictConfig | IrisBootstrapConfig,
    *,
    input_snapshot: str,
    config: IrisBootstrapConfig,
    owned_snapshot: str,
    backbone: BackboneAdapter,
    device_info: DeviceInfo,
    dtype: torch.dtype,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_snapshot: tuple[float, float],
) -> None:
    if _snapshot_input(original) != input_snapshot:
        message = "Caller configuration changed during bootstrap"
        raise ValueError(message)
    if _canonical_config_json(config) != owned_snapshot:
        message = "Owned validated configuration changed during bootstrap"
        raise ValueError(message)
    if _validate_generation_policy(stop_token_ids, sampling) != sampling_snapshot:
        message = "Sampling policy changed during bootstrap"
        raise ValueError(message)
    _validate_backbone(
        backbone,
        config=config,
        device_info=device_info,
        dtype=dtype,
        stop_token_ids=stop_token_ids,
    )
    if torch.are_deterministic_algorithms_enabled() is not config.runtime.deterministic:
        message = "PyTorch deterministic-algorithm policy changed during bootstrap"
        raise ValueError(message)
    if torch.is_deterministic_algorithms_warn_only_enabled():
        message = "Iris bootstrap requires deterministic warn_only=false"
        raise ValueError(message)
    if torch.backends.cudnn.deterministic is not config.runtime.deterministic:
        message = "cuDNN deterministic policy changed during bootstrap"
        raise ValueError(message)
    if config.runtime.deterministic and torch.backends.cudnn.benchmark:
        message = "Deterministic Iris bootstrap requires cudnn.benchmark=false"
        raise ValueError(message)


def bootstrap_request_runtime(
    config: DictConfig | IrisBootstrapConfig,
    *,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisRequestRuntime:
    """Validate, seed, load, and compose one exclusive request runtime."""
    owned, input_snapshot = _stage("configuration validation", lambda: _own_config(config))
    owned_snapshot = _canonical_config_json(owned)
    configuration_sha256 = domain_sha256(
        "iris/configuration/v1",
        owned_snapshot.encode(),
    )
    sampling_snapshot = _stage(
        "generation-policy validation",
        lambda: _validate_generation_policy(stop_token_ids, sampling),
    )
    _stage("CUDA deterministic preflight", lambda: _validate_cuda_determinism(owned))
    device_info, dtype = _stage("device resolution", lambda: _resolve_device(owned))
    runtime_snapshot = _stage(
        "runtime-state capture",
        lambda: capture_runtime_state(device_info.device),
    )

    try:
        _stage(
            "runtime-policy application",
            lambda: apply_runtime_policy(
                seed=owned.runtime.seed,
                deterministic=owned.runtime.deterministic,
                device=device_info.device,
            ),
        )
        backbone = _stage(
            "backbone loading",
            lambda: _create_backbone(owned.backbone, device_info),
        )
        _stage(
            "backbone validation",
            lambda: _validate_backbone(
                backbone,
                config=owned,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
            ),
        )
        orchestrator = _stage(
            "request-graph composition",
            lambda: build_request_orchestrator(
                backbone=backbone,
                cognitive_config=owned.cognitive,
                workspace_config=owned.workspace,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
            ),
        )
        if not isinstance(orchestrator, IrisRequestOrchestrator):
            message = "Composition did not return IrisRequestOrchestrator"
            raise IrisBootstrapError(message)
        sidecar_state_sha256 = _stage(
            "sidecar provenance capture",
            lambda: _read_sidecar_state_sha256(orchestrator),
        )
        _stage(
            "final runtime validation",
            lambda: _validate_stability(
                config,
                input_snapshot=input_snapshot,
                config=owned,
                owned_snapshot=owned_snapshot,
                backbone=backbone,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                sampling_snapshot=sampling_snapshot,
            ),
        )
    except BaseException as error:
        try:
            restore_runtime_state(runtime_snapshot)
        except BaseException as rollback_error:
            failures = BaseExceptionGroup(
                "Iris bootstrap failed and runtime rollback was incomplete",
                [error, rollback_error],
            )
            message = "Iris bootstrap failed and process runtime state is uncertain"
            raise IrisBootstrapError(message) from failures
        raise

    return IrisRequestRuntime(
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=device_info,
        dtype=dtype,
        seed=owned.runtime.seed,
        deterministic=owned.runtime.deterministic,
        configuration_json=owned_snapshot,
        configuration_sha256=configuration_sha256,
        sidecar_state_sha256=sidecar_state_sha256,
    )


def bootstrap_arm_request_runtime_v2(
    arm_id: object,
    config: DictConfig | IrisBootstrapConfig,
    *,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisArmRequestRuntimeV2:
    """Bootstrap one executable A/B/D graph and reject unresolved C before loading."""
    try:
        profile = require_arm_request_executable_v3(arm_id)
    except Exception as error:
        message = "Arm-labelled request bootstrap rejected the execution profile"
        raise IrisBootstrapError(message) from error
    owned, input_snapshot = _stage("configuration validation", lambda: _own_config(config))
    owned_snapshot = _canonical_config_json(owned)
    configuration_sha256 = domain_sha256(
        "iris/configuration/v1",
        owned_snapshot.encode(),
    )
    sampling_snapshot = _stage(
        "generation-policy validation",
        lambda: _validate_generation_policy(stop_token_ids, sampling),
    )
    _stage("CUDA deterministic preflight", lambda: _validate_cuda_determinism(owned))
    device_info, dtype = _stage("device resolution", lambda: _resolve_device(owned))
    runtime_snapshot = _stage(
        "runtime-state capture",
        lambda: capture_runtime_state(device_info.device),
    )

    try:
        _stage(
            "runtime-policy application",
            lambda: apply_runtime_policy(
                seed=owned.runtime.seed,
                deterministic=owned.runtime.deterministic,
                device=device_info.device,
            ),
        )
        backbone = _stage(
            "backbone loading",
            lambda: _create_backbone(owned.backbone, device_info),
        )
        _stage(
            "backbone validation",
            lambda: _validate_backbone(
                backbone,
                config=owned,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
            ),
        )
        orchestrator = _stage(
            "arm-labelled request-graph composition",
            lambda: build_arm_request_orchestrator_v3(
                arm_id=profile.arm_id,
                backbone=backbone,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                cognitive_config=(None if profile.arm_id == "A" else owned.cognitive),
                workspace_config=(owned.workspace if profile.arm_id == "D" else None),
            ),
        )
        sidecar_state_sha256 = _stage(
            "arm-labelled sidecar provenance capture",
            lambda: _read_arm_sidecar_state_sha256(
                profile=profile,
                orchestrator=orchestrator,
            ),
        )
        profile_set_sha256 = _stage(
            "execution-profile identity validation",
            arm_execution_profile_set_sha256_v3,
        )
        if profile_set_sha256 != (
            "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
        ):
            message = "Arm execution profile set v3 identity changed unexpectedly"
            raise ValueError(message)
        _stage(
            "final runtime validation",
            lambda: _validate_stability(
                config,
                input_snapshot=input_snapshot,
                config=owned,
                owned_snapshot=owned_snapshot,
                backbone=backbone,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                sampling_snapshot=sampling_snapshot,
            ),
        )
        _stage(
            "final arm-labelled sidecar validation",
            lambda: _read_arm_sidecar_state_sha256(
                profile=profile,
                orchestrator=orchestrator,
            ),
        )
    except BaseException as error:
        try:
            restore_runtime_state(runtime_snapshot)
        except BaseException as rollback_error:
            failures = BaseExceptionGroup(
                "Arm request bootstrap failed and runtime rollback was incomplete",
                [error, rollback_error],
            )
            message = "Arm request bootstrap failed and process runtime state is uncertain"
            raise IrisBootstrapError(message) from failures
        raise

    return IrisArmRequestRuntimeV2(
        profile=profile,
        execution_profile_set_sha256=(
            "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
        ),
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=device_info,
        dtype=dtype,
        seed=owned.runtime.seed,
        deterministic=owned.runtime.deterministic,
        configuration_json=owned_snapshot,
        configuration_sha256=configuration_sha256,
        sidecar_state_sha256=sidecar_state_sha256,
    )


def bootstrap_arm_request_runtime_v3(
    arm_id: object,
    config: DictConfig | IrisBootstrapConfig,
    *,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisArmRequestRuntimeV3:
    """Bootstrap one profile-v4 A/B/C/D request graph transactionally."""
    profile = _preflight_arm_request_v4(arm_id)
    owned, input_snapshot = _stage("configuration validation", lambda: _own_config(config))
    owned_snapshot = _canonical_config_json(owned)
    configuration_sha256 = domain_sha256(
        "iris/configuration/v1",
        owned_snapshot.encode(),
    )
    sampling_snapshot = _stage(
        "generation-policy validation",
        lambda: _validate_generation_policy(stop_token_ids, sampling),
    )
    _stage("CUDA deterministic preflight", lambda: _validate_cuda_determinism(owned))
    device_info, dtype = _stage("device resolution", lambda: _resolve_device(owned))
    runtime_snapshot = _stage(
        "runtime-state capture",
        lambda: capture_runtime_state(device_info.device),
    )

    try:
        _stage(
            "runtime-policy application",
            lambda: apply_runtime_policy(
                seed=owned.runtime.seed,
                deterministic=owned.runtime.deterministic,
                device=device_info.device,
            ),
        )
        backbone = _stage(
            "backbone loading",
            lambda: _create_backbone(owned.backbone, device_info),
        )
        _stage(
            "backbone validation",
            lambda: _validate_backbone(
                backbone,
                config=owned,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
            ),
        )
        orchestrator = _stage(
            "arm-labelled request-graph v4 composition",
            lambda: build_arm_request_orchestrator_v4(
                arm_id=profile.arm_id,
                backbone=backbone,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                cognitive_config=(None if profile.arm_id == "A" else owned.cognitive),
                workspace_config=(owned.workspace if profile.arm_id == "D" else None),
            ),
        )
        sidecar_state_sha256 = _stage(
            "arm-labelled sidecar provenance capture",
            lambda: _read_arm_sidecar_state_sha256_v4(
                profile=profile,
                orchestrator=orchestrator,
                backbone=backbone,
            ),
        )
        _stage(
            "final runtime validation",
            lambda: _validate_stability(
                config,
                input_snapshot=input_snapshot,
                config=owned,
                owned_snapshot=owned_snapshot,
                backbone=backbone,
                device_info=device_info,
                dtype=dtype,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                sampling_snapshot=sampling_snapshot,
            ),
        )

        def validate_final_arm_state() -> None:
            final_profile = _preflight_arm_request_v4(profile.arm_id)
            if final_profile != profile:
                message = "Arm execution profile v4 changed during bootstrap"
                raise ValueError(message)
            final_sidecar = _read_arm_sidecar_state_sha256_v4(
                profile=profile,
                orchestrator=orchestrator,
                backbone=backbone,
            )
            if final_sidecar != sidecar_state_sha256:
                message = "Arm request sidecar provenance changed during bootstrap"
                raise ValueError(message)

        _stage("final arm-labelled provenance validation", validate_final_arm_state)
    except BaseException as error:
        try:
            restore_runtime_state(runtime_snapshot)
        except BaseException as rollback_error:
            failures = BaseExceptionGroup(
                "Arm request bootstrap v3 failed and runtime rollback was incomplete",
                [error, rollback_error],
            )
            message = "Arm request bootstrap v3 failed and process runtime state is uncertain"
            raise IrisBootstrapError(message) from failures
        raise

    return IrisArmRequestRuntimeV3(
        profile=profile,
        execution_profile_set_sha256=_PROFILE_SET_SHA256_V4,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=device_info,
        dtype=dtype,
        seed=owned.runtime.seed,
        deterministic=owned.runtime.deterministic,
        configuration_json=owned_snapshot,
        configuration_sha256=configuration_sha256,
        sidecar_state_sha256=sidecar_state_sha256,
    )
