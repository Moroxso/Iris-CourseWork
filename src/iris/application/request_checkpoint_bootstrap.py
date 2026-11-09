"""Transactional profile-v4 request bootstrap bound to one teacher checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from re import fullmatch
from typing import Literal

from omegaconf import DictConfig

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV4,
    arm_execution_profile_set_sha256_v4,
    build_arm_request_orchestrator_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointProjectionV1,
    IrisArmRequestCheckpointTransferResultV1,
    project_arm_teacher_checkpoint_to_request_v1,
    transfer_arm_teacher_checkpoint_to_request_v1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherCheckpointIdentity,
    read_arm_teacher_checkpoint,
)
from iris.application.bootstrap import (
    IrisArmRequestRuntimeV3,
    _canonical_config_json,
    _create_backbone,
    _own_config,
    _read_arm_sidecar_state_sha256_v4,
    _resolve_device,
    _snapshot_input,
    _validate_backbone,
    _validate_cuda_determinism,
    _validate_generation_policy,
    _validate_stability,
)
from iris.application.errors import IrisBootstrapError
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.shared_buffer import shared_buffer_policy_sha256_v1
from iris.utils.hashing import domain_sha256
from iris.utils.seeds import apply_runtime_policy, capture_runtime_state, restore_runtime_state

_PROFILE_SET_SHA256_V4: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_SHARED_BUFFER_POLICY_SHA256_V1 = (
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
)


@dataclass(frozen=True, slots=True)
class IrisArmRequestBootstrapResultV4:
    """A provisional v3 graph published only after checkpoint transfer succeeds."""

    runtime: IrisArmRequestRuntimeV3
    checkpoint_transfer: IrisArmRequestCheckpointTransferResultV1 | None

    def __post_init__(self) -> None:
        if type(self.runtime) is not IrisArmRequestRuntimeV3:
            message = "checkpointed request runtime must own exact IrisArmRequestRuntimeV3"
            raise TypeError(message)
        arm_id = self.runtime.profile.arm_id
        if arm_id == "A":
            if self.checkpoint_transfer is not None:
                message = "arm A checkpointed request runtime cannot own a transfer"
                raise ValueError(message)
            return
        if type(self.checkpoint_transfer) is not IrisArmRequestCheckpointTransferResultV1:
            message = f"arm {arm_id} checkpointed request runtime requires an exact transfer"
            raise TypeError(message)
        if self.checkpoint_transfer.runtime is not self.runtime:
            message = "checkpointed request runtime must publish the exact transferred runtime"
            raise ValueError(message)


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisBootstrapError:
        raise
    except Exception as error:
        message = f"Checkpoint-bound request bootstrap failed during {name}"
        raise IrisBootstrapError(message) from error


def _preflight_identity(
    *,
    arm_id: object,
    identity: IrisArmTeacherCheckpointIdentity | None,
) -> IrisArmExecutionProfileV4:
    try:
        profile = require_arm_request_executable_v4(arm_id)
        if type(profile) is not IrisArmExecutionProfileV4:
            message = "checkpoint-bound request bootstrap requires an exact profile v4"
            raise TypeError(message)
        if arm_execution_profile_set_sha256_v4() != _PROFILE_SET_SHA256_V4:
            message = "execution profile set v4 identity changed unexpectedly"
            raise ValueError(message)
        if shared_buffer_policy_sha256_v1() != _SHARED_BUFFER_POLICY_SHA256_V1:
            message = "shared-buffer policy v1 identity changed unexpectedly"
            raise ValueError(message)
        if profile.arm_id == "A":
            if identity is not None:
                message = "arm A cannot accept a teacher checkpoint identity"
                raise ValueError(message)
        elif type(identity) is not IrisArmTeacherCheckpointIdentity:
            message = f"arm {profile.arm_id} requires an exact teacher checkpoint identity"
            raise TypeError(message)
        elif identity.arm_id != profile.arm_id:
            message = "teacher checkpoint identity arm differs from requested profile"
            raise ValueError(message)
        return profile
    except IrisBootstrapError:
        raise
    except Exception as error:
        message = "Checkpoint-bound request bootstrap rejected its profile or checkpoint identity"
        raise IrisBootstrapError(message) from error


def _read_projection(
    *,
    profile: IrisArmExecutionProfileV4,
    identity: IrisArmTeacherCheckpointIdentity | None,
    root: Path,
) -> tuple[IrisArmTeacherCheckpointBundle | None, IrisArmRequestCheckpointProjectionV1 | None]:
    if profile.arm_id == "A":
        return None, None
    if type(identity) is not IrisArmTeacherCheckpointIdentity:
        message = "trained arm checkpoint identity disappeared after preflight"
        raise TypeError(message)
    checkpoint = read_arm_teacher_checkpoint(identity, root=root)
    projection = project_arm_teacher_checkpoint_to_request_v1(checkpoint)
    if projection.checkpoint_identity != identity:
        message = "checkpoint projection identity differs from the requested checkpoint"
        raise ValueError(message)
    return checkpoint, projection


def _validate_projection_config(
    *,
    profile: IrisArmExecutionProfileV4,
    config: IrisBootstrapConfig,
    projection: IrisArmRequestCheckpointProjectionV1 | None,
) -> None:
    if profile.arm_id == "A":
        if projection is not None:
            message = "arm A unexpectedly produced checkpoint projection"
            raise ValueError(message)
        return
    if type(projection) is not IrisArmRequestCheckpointProjectionV1:
        message = f"arm {profile.arm_id} requires exact checkpoint projection"
        raise TypeError(message)
    if projection.checkpoint_identity.arm_id != profile.arm_id:
        message = "checkpoint projection arm differs from execution profile"
        raise ValueError(message)
    if projection.backbone != config.backbone or projection.cognitive != config.cognitive:
        message = "checkpoint model configuration differs from request configuration"
        raise ValueError(message)
    if profile.arm_id == "D":
        if projection.workspace != config.workspace:
            message = "arm-D checkpoint workspace differs from request configuration"
            raise ValueError(message)
    elif projection.workspace is not None:
        message = f"arm-{profile.arm_id} checkpoint unexpectedly owns workspace configuration"
        raise ValueError(message)
    expected_dtype = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }[config.hardware.precision]
    runtime = projection.runtime
    if (
        runtime.max_prompt_tokens != config.backbone.limits.max_prompt_tokens
        or runtime.max_target_tokens != config.backbone.limits.max_new_tokens
        or runtime.dtype != expected_dtype
    ):
        message = "checkpoint runtime limits or dtype differ from request configuration"
        raise ValueError(message)
    if (config.hardware.device == "cpu" and runtime.device != "cpu") or (
        config.hardware.device == "cuda"
        and fullmatch(r"cuda:(0|[1-9][0-9]*)", runtime.device) is None
    ):
        message = "checkpoint runtime device differs from requested hardware backend"
        raise ValueError(message)


def _validate_transfer(
    *,
    profile: IrisArmExecutionProfileV4,
    projection: IrisArmRequestCheckpointProjectionV1 | None,
    transfer: IrisArmRequestCheckpointTransferResultV1 | None,
) -> None:
    if profile.arm_id == "A":
        if projection is not None or transfer is not None:
            message = "arm A cannot publish checkpoint provenance"
            raise ValueError(message)
        return
    if (
        type(projection) is not IrisArmRequestCheckpointProjectionV1
        or type(transfer) is not IrisArmRequestCheckpointTransferResultV1
    ):
        message = f"arm {profile.arm_id} requires exact projection and transfer"
        raise TypeError(message)
    if (
        transfer.checkpoint_identity != projection.checkpoint_identity
        or transfer.source_teacher_sidecar_sha256
        != projection.source_teacher_sidecar_sha256
        or transfer.installed_request_sidecar_sha256
        != projection.request_sidecar_state_sha256
        or transfer.completed_steps != projection.completed_steps
        or transfer.state_names != projection.state_names
        or transfer.transferred_tensor_count != projection.tensor_count
    ):
        message = "committed request transfer differs from checkpoint projection"
        raise ValueError(message)


def bootstrap_arm_request_runtime_v4(
    arm_id: object,
    config: DictConfig | IrisBootstrapConfig,
    *,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisArmRequestBootstrapResultV4:
    """Read a checkpoint first, then privately compose, transfer, and publish one graph."""
    profile = _preflight_identity(arm_id=arm_id, identity=checkpoint_identity)
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
    root = Path(owned.runtime.output_dir)
    if not root.name:
        message = "checkpoint-bound request bootstrap requires an explicit artifact root"
        raise IrisBootstrapError(message)
    checkpoint, projection = _stage(
        "checkpoint read and projection",
        lambda: _read_projection(
            profile=profile,
            identity=checkpoint_identity,
            root=root,
        ),
    )
    _stage(
        "checkpoint configuration compatibility",
        lambda: _validate_projection_config(
            profile=profile,
            config=owned,
            projection=projection,
        ),
    )
    _stage("CUDA deterministic preflight", lambda: _validate_cuda_determinism(owned))
    device_info, dtype = _stage("device resolution", lambda: _resolve_device(owned))
    if projection is not None and (
        projection.runtime.device != str(device_info.device)
        or projection.runtime.dtype != str(dtype).removeprefix("torch.")
    ):
        message = "checkpoint placement differs from resolved request device"
        raise IrisBootstrapError(message)
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
            "provisional profile-v4 request-graph composition",
            lambda: build_arm_request_orchestrator_v4(
                arm_id=profile.arm_id,
                backbone=backbone,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                cognitive_config=(None if profile.arm_id == "A" else owned.cognitive),
                workspace_config=(owned.workspace if profile.arm_id == "D" else None),
            ),
        )
        initial_sidecar_sha256 = _stage(
            "provisional sidecar provenance capture",
            lambda: _read_arm_sidecar_state_sha256_v4(
                profile=profile,
                orchestrator=orchestrator,
                backbone=backbone,
            ),
        )
        _stage(
            "provisional runtime stability validation",
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
        runtime = IrisArmRequestRuntimeV3(
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
            sidecar_state_sha256=initial_sidecar_sha256,
        )
        transfer: IrisArmRequestCheckpointTransferResultV1 | None = None
        if checkpoint is not None:
            transfer = _stage(
                "teacher checkpoint transfer",
                lambda: transfer_arm_teacher_checkpoint_to_request_v1(
                    runtime=runtime,
                    checkpoint=checkpoint,
                ),
            )
            runtime = transfer.runtime
        _stage(
            "committed transfer validation",
            lambda: _validate_transfer(
                profile=profile,
                projection=projection,
                transfer=transfer,
            ),
        )
        if (
            _snapshot_input(config) != input_snapshot
            or _canonical_config_json(owned) != owned_snapshot
        ):
            message = "request configuration changed during checkpoint-bound bootstrap"
            raise ValueError(message)
        if _validate_generation_policy(stop_token_ids, sampling) != sampling_snapshot:
            message = "generation policy changed during checkpoint-bound bootstrap"
            raise ValueError(message)
        final_profile = _preflight_identity(
            arm_id=profile.arm_id,
            identity=checkpoint_identity,
        )
        if final_profile != profile:
            message = "execution profile changed during checkpoint-bound bootstrap"
            raise ValueError(message)
        sidecar_sha256 = _read_arm_sidecar_state_sha256_v4(
            profile=profile,
            orchestrator=runtime.orchestrator,
            backbone=runtime.backbone,
        )
        if sidecar_sha256 != runtime.sidecar_state_sha256:
            message = "published runtime sidecar differs after checkpoint transfer"
            raise ValueError(message)
        if projection is not None:
            final_checkpoint, final_projection = _read_projection(
                profile=profile,
                identity=projection.checkpoint_identity,
                root=root,
            )
            if final_checkpoint is None or final_projection != projection:
                message = "teacher checkpoint changed during request bootstrap"
                raise ValueError(message)
        result = IrisArmRequestBootstrapResultV4(
            runtime=runtime,
            checkpoint_transfer=transfer,
        )
    except BaseException as error:
        try:
            restore_runtime_state(runtime_snapshot)
        except BaseException as rollback_error:
            failures = BaseExceptionGroup(
                "Checkpoint-bound bootstrap failed and runtime rollback was incomplete",
                [error, rollback_error],
            )
            message = "Checkpoint-bound request runtime state is uncertain"
            raise IrisBootstrapError(message) from failures
        raise
    return result


__all__ = ["IrisArmRequestBootstrapResultV4", "bootstrap_arm_request_runtime_v4"]
