"""Fail-closed teacher-checkpoint transfer into profile-v4 request graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from re import fullmatch
from typing import Literal, cast

import torch
from safetensors.torch import load as load_safetensors

from iris.application.arm_profiles import IrisArmExecutionProfileV4
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherCheckpointIdentity,
    IrisTeacherArmId,
    read_arm_teacher_checkpoint,
    validate_arm_teacher_checkpoint,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestCheckpointError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
)
from iris.application.request import IrisRequestOrchestrator
from iris.application.request_sidecar_state import IrisRequestSidecarInstallResultV1
from iris.application.shared_buffer_request import CognitiveSharedBufferRequestOrchestrator
from iris.application.shared_buffer_training_checkpoint import (
    IrisSharedBufferTeacherCheckpointBundleV3,
)
from iris.application.training_checkpoint import IrisTeacherCheckpointBundleV1
from iris.artifacts.checkpoint_v1 import (
    TeacherBackboneRuntimeV1,
    TeacherCheckpointTensorV1,
)
from iris.schemas import BackboneConfig, CognitiveConfig, IrisBootstrapConfig, WorkspaceConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, named_module_state_sha256

_PROFILE_SET_V4_SHA256 = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
_ARM_B_TEACHER_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)
_ARM_C_TEACHER_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
    "sequence_assembler",
)
_ARM_D_TEACHER_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "workspace_reader",
    "cognitive_modules",
    "proposal_head",
    "workspace_arbiter",
    "slot_allocator",
    "workspace_writer",
    "workspace_decay",
    "latent_bridge",
    "sequence_assembler",
)
_ARM_REQUEST_COMPONENTS = {
    "B": (*_ARM_B_TEACHER_COMPONENTS[:-1], "prefix_assembler"),
    "C": (*_ARM_C_TEACHER_COMPONENTS[:-1], "prefix_assembler"),
    "D": (*_ARM_D_TEACHER_COMPONENTS[:-1], "prefix_assembler"),
}


@dataclass(frozen=True, slots=True)
class IrisArmRequestCheckpointTransferResultV1:
    """One committed checkpoint-to-request state transition and its two hashes."""

    runtime: IrisArmRequestRuntimeV3
    checkpoint_identity: IrisArmTeacherCheckpointIdentity
    transfer_contract: Literal["exact_teacher_state_to_profile_v4_request_sidecar_v1"]
    source_teacher_sidecar_sha256: str
    previous_request_sidecar_sha256: str
    installed_request_sidecar_sha256: str
    completed_steps: int
    state_names: tuple[str, ...]
    transferred_tensor_count: int

    def __post_init__(self) -> None:
        if type(self.runtime) is not IrisArmRequestRuntimeV3:
            message = "checkpoint transfer runtime must be exact IrisArmRequestRuntimeV3"
            raise TypeError(message)
        if type(self.checkpoint_identity) is not IrisArmTeacherCheckpointIdentity:
            message = "checkpoint transfer identity must be exact"
            raise TypeError(message)
        if self.transfer_contract != "exact_teacher_state_to_profile_v4_request_sidecar_v1":
            message = "checkpoint transfer contract is not the supported version"
            raise ValueError(message)
        if self.runtime.profile.arm_id != self.checkpoint_identity.arm_id:
            message = "checkpoint transfer runtime and checkpoint arms differ"
            raise ValueError(message)
        for field, value in (
            ("source_teacher_sidecar_sha256", self.source_teacher_sidecar_sha256),
            ("previous_request_sidecar_sha256", self.previous_request_sidecar_sha256),
            ("installed_request_sidecar_sha256", self.installed_request_sidecar_sha256),
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                message = f"{field} must be a canonical SHA-256 digest"
                raise ValueError(message)
        if self.source_teacher_sidecar_sha256 != self.checkpoint_identity.sidecar_state_sha256:
            message = "checkpoint transfer source sidecar hash differs from checkpoint identity"
            raise ValueError(message)
        if self.runtime.sidecar_state_sha256 != self.installed_request_sidecar_sha256:
            message = "checkpoint transfer runtime carries a different installed sidecar hash"
            raise ValueError(message)
        orchestrator = self.runtime.orchestrator
        if not hasattr(orchestrator, "sidecar_state_sha256") or (
            orchestrator.sidecar_state_sha256 != self.installed_request_sidecar_sha256
        ):
            message = "checkpoint transfer orchestrator carries a different installed sidecar hash"
            raise ValueError(message)
        if (
            type(self.completed_steps) is not int
            or self.completed_steps != self.checkpoint_identity.completed_steps
        ):
            message = "checkpoint transfer step count differs from checkpoint identity"
            raise ValueError(message)
        if (
            type(self.state_names) is not tuple
            or not self.state_names
            or len(set(self.state_names)) != len(self.state_names)
        ):
            message = "checkpoint transfer state names must be a nonempty unique tuple"
            raise ValueError(message)
        if (
            type(self.transferred_tensor_count) is not int
            or self.transferred_tensor_count != len(self.state_names)
        ):
            message = "transferred tensor count must equal the state-name count"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class IrisArmRequestCheckpointProjectionV1:
    """Redacted pure projection of one teacher checkpoint into a request topology."""

    checkpoint_identity: IrisArmTeacherCheckpointIdentity
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig | None
    runtime: TeacherBackboneRuntimeV1
    source_teacher_sidecar_sha256: str
    request_sidecar_state_sha256: str
    completed_steps: int
    state_names: tuple[str, ...]
    tensor_count: int

    def __post_init__(self) -> None:
        if type(self.checkpoint_identity) is not IrisArmTeacherCheckpointIdentity:
            message = "checkpoint projection identity must be exact"
            raise TypeError(message)
        if type(self.backbone) is not BackboneConfig or type(self.cognitive) is not CognitiveConfig:
            message = "checkpoint projection model configurations must be exact"
            raise TypeError(message)
        if self.workspace is not None and type(self.workspace) is not WorkspaceConfig:
            message = "checkpoint projection workspace configuration must be exact"
            raise TypeError(message)
        if type(self.runtime) is not TeacherBackboneRuntimeV1:
            message = "checkpoint projection runtime must be exact"
            raise TypeError(message)
        for field, value in (
            ("source_teacher_sidecar_sha256", self.source_teacher_sidecar_sha256),
            ("request_sidecar_state_sha256", self.request_sidecar_state_sha256),
        ):
            if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
                message = f"{field} must be a canonical SHA-256 digest"
                raise ValueError(message)
        if self.source_teacher_sidecar_sha256 != self.checkpoint_identity.sidecar_state_sha256:
            message = "checkpoint projection source hash differs from checkpoint identity"
            raise ValueError(message)
        if self.completed_steps != self.checkpoint_identity.completed_steps:
            message = "checkpoint projection steps differ from checkpoint identity"
            raise ValueError(message)
        if (
            type(self.state_names) is not tuple
            or not self.state_names
            or len(set(self.state_names)) != len(self.state_names)
            or type(self.tensor_count) is not int
            or self.tensor_count != len(self.state_names)
        ):
            message = "checkpoint projection state topology is invalid"
            raise ValueError(message)
        if (self.checkpoint_identity.arm_id == "D") is (self.workspace is None):
            message = "checkpoint projection workspace ownership differs from its arm"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _CheckpointSidecarView:
    arm_id: IrisTeacherArmId
    identity: IrisArmTeacherCheckpointIdentity
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig | None
    runtime: TeacherBackboneRuntimeV1
    component_names: tuple[str, ...]
    state_names: tuple[str, ...]
    sidecar_state_sha256: str
    completed_steps: int
    descriptors: tuple[TeacherCheckpointTensorV1, ...]
    state_bytes: bytes


def _require_runtime(runtime: object) -> IrisArmRequestRuntimeV3:
    if type(runtime) is not IrisArmRequestRuntimeV3:
        message = "checkpoint transfer requires exact IrisArmRequestRuntimeV3"
        raise TypeError(message)
    if type(runtime.profile) is not IrisArmExecutionProfileV4:
        message = "checkpoint transfer requires exact profile-v4 provenance"
        raise TypeError(message)
    if runtime.profile.arm_id not in ("B", "C", "D"):
        message = "arm A has no teacher checkpoint to transfer"
        raise ValueError(message)
    if runtime.execution_profile_set_sha256 != _PROFILE_SET_V4_SHA256:
        message = "request runtime is not bound to execution profile set v4"
        raise ValueError(message)
    if (
        type(runtime.sidecar_state_sha256) is not str
        or fullmatch(r"[0-9a-f]{64}", runtime.sidecar_state_sha256) is None
    ):
        message = "request runtime has no canonical sidecar state identity"
        raise ValueError(message)
    expected_type: type[object]
    if runtime.profile.arm_id == "B":
        expected_type = CognitiveNoWorkspaceRequestOrchestrator
    elif runtime.profile.arm_id == "C":
        expected_type = CognitiveSharedBufferRequestOrchestrator
    else:
        expected_type = IrisRequestOrchestrator
    if type(runtime.orchestrator) is not expected_type:
        message = f"arm {runtime.profile.arm_id} runtime owns the wrong request graph type"
        raise TypeError(message)
    if runtime.orchestrator.backbone is not runtime.backbone:
        message = "request graph does not own the runtime backbone"
        raise ValueError(message)
    if runtime.orchestrator.sidecar_state_sha256 != runtime.sidecar_state_sha256:
        message = "request runtime sidecar identity is stale before checkpoint transfer"
        raise ValueError(message)
    return runtime


def _runtime_config(runtime: IrisArmRequestRuntimeV3) -> IrisBootstrapConfig:
    try:
        config = IrisBootstrapConfig.model_validate_json(runtime.configuration_json)
    except Exception as error:
        message = "request runtime configuration JSON is invalid"
        raise ValueError(message) from error
    canonical = canonical_json_bytes(config.model_dump(mode="json")).decode("utf-8")
    if canonical != runtime.configuration_json:
        message = "request runtime configuration JSON is not canonical"
        raise ValueError(message)
    expected_sha256 = domain_sha256("iris/configuration/v1", canonical.encode("utf-8"))
    if runtime.configuration_sha256 != expected_sha256:
        message = "request runtime configuration hash is invalid"
        raise ValueError(message)
    try:
        backbone_config = BackboneConfig.model_validate_json(runtime.backbone.configuration_json)
    except Exception as error:
        message = "request backbone configuration provenance is invalid"
        raise ValueError(message) from error
    if backbone_config != config.backbone:
        message = "request runtime and loaded backbone configurations differ"
        raise ValueError(message)
    return config


def _checkpoint_view(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> _CheckpointSidecarView:
    verified = validate_arm_teacher_checkpoint(checkpoint)
    identity = verified.identity
    if identity.arm_id == "B":
        bundle_b = cast(IrisNoWorkspaceTeacherCheckpointBundleV2, verified.payload)
        payload = bundle_b.manifest.payload
        model = payload.model
        return _CheckpointSidecarView(
            arm_id="B",
            identity=identity,
            backbone=model.backbone,
            cognitive=model.cognitive,
            workspace=None,
            runtime=model.runtime,
            component_names=model.component_names,
            state_names=model.state_names,
            sidecar_state_sha256=model.sha256,
            completed_steps=payload.optimizer.completed_steps,
            descriptors=payload.tensors,
            state_bytes=bundle_b.state_bytes,
        )
    if identity.arm_id == "C":
        bundle_c = cast(IrisSharedBufferTeacherCheckpointBundleV3, verified.payload)
        payload_c = bundle_c.manifest.payload
        model_c = payload_c.model
        return _CheckpointSidecarView(
            arm_id="C",
            identity=identity,
            backbone=model_c.backbone,
            cognitive=model_c.cognitive,
            workspace=None,
            runtime=model_c.runtime,
            component_names=model_c.component_names,
            state_names=model_c.state_names,
            sidecar_state_sha256=model_c.sha256,
            completed_steps=payload_c.optimizer.completed_steps,
            descriptors=payload_c.tensors,
            state_bytes=bundle_c.state_bytes,
        )
    bundle_d = cast(IrisTeacherCheckpointBundleV1, verified.payload)
    payload_d = bundle_d.manifest.payload
    model_d = payload_d.model
    return _CheckpointSidecarView(
        arm_id="D",
        identity=identity,
        backbone=model_d.backbone,
        cognitive=model_d.cognitive,
        workspace=model_d.workspace,
        runtime=model_d.runtime,
        component_names=_ARM_D_TEACHER_COMPONENTS,
        state_names=model_d.state_names,
        sidecar_state_sha256=model_d.sha256,
        completed_steps=payload_d.optimizer.completed_steps,
        descriptors=payload_d.tensors,
        state_bytes=bundle_d.state_bytes,
    )


def _validate_compatibility(
    *,
    runtime: IrisArmRequestRuntimeV3,
    config: IrisBootstrapConfig,
    checkpoint: _CheckpointSidecarView,
) -> None:
    if checkpoint.arm_id != runtime.profile.arm_id:
        message = "teacher checkpoint arm does not match request runtime arm"
        raise ValueError(message)
    expected_components = {
        "B": _ARM_B_TEACHER_COMPONENTS,
        "C": _ARM_C_TEACHER_COMPONENTS,
        "D": _ARM_D_TEACHER_COMPONENTS,
    }[checkpoint.arm_id]
    if checkpoint.component_names != expected_components:
        message = "teacher checkpoint component ownership is not canonical"
        raise ValueError(message)
    if checkpoint.backbone != config.backbone or checkpoint.cognitive != config.cognitive:
        message = "teacher checkpoint model configuration does not match request runtime"
        raise ValueError(message)
    if checkpoint.arm_id == "D":
        if checkpoint.workspace != config.workspace:
            message = "arm-D checkpoint workspace configuration does not match request runtime"
            raise ValueError(message)
    elif checkpoint.workspace is not None:
        message = f"arm-{checkpoint.arm_id} checkpoint cannot own workspace configuration"
        raise ValueError(message)
    backbone = runtime.backbone
    actual_runtime = (
        backbone.hidden_size,
        backbone.num_hidden_layers,
        backbone.vocabulary_size,
        backbone.max_position_embeddings,
        backbone.max_prompt_tokens,
        backbone.max_new_tokens,
        str(runtime.device_info.device),
        str(runtime.dtype).removeprefix("torch."),
    )
    expected_runtime = (
        checkpoint.runtime.hidden_size,
        checkpoint.runtime.num_hidden_layers,
        checkpoint.runtime.vocabulary_size,
        checkpoint.runtime.max_position_embeddings,
        checkpoint.runtime.max_prompt_tokens,
        checkpoint.runtime.max_target_tokens,
        checkpoint.runtime.device,
        checkpoint.runtime.dtype,
    )
    if actual_runtime != expected_runtime:
        message = "teacher checkpoint runtime does not match request backbone placement"
        raise ValueError(message)


def _sidecar_state(checkpoint: _CheckpointSidecarView) -> dict[str, torch.Tensor]:
    try:
        decoded = load_safetensors(checkpoint.state_bytes)
    except Exception as error:
        message = "validated teacher checkpoint state bytes could not be decoded"
        raise ValueError(message) from error
    descriptors = {
        descriptor.state_name: descriptor
        for descriptor in checkpoint.descriptors
        if descriptor.owner == "sidecar"
    }
    if tuple(descriptors) != checkpoint.state_names:
        message = "teacher checkpoint tensor table does not match state topology"
        raise ValueError(message)
    state: dict[str, torch.Tensor] = {}
    for name in checkpoint.state_names:
        descriptor = descriptors[name]
        source = decoded.get(descriptor.key)
        if type(source) is not torch.Tensor:
            message = f"teacher checkpoint is missing sidecar tensor {name!r}"
            raise ValueError(message)
        state[name] = source.detach().contiguous().clone()
    return state


def _projection(
    checkpoint: _CheckpointSidecarView,
    state: dict[str, torch.Tensor],
) -> IrisArmRequestCheckpointProjectionV1:
    request_sha256 = named_module_state_sha256(
        component_names=_ARM_REQUEST_COMPONENTS[checkpoint.arm_id],
        state=state,
    )
    return IrisArmRequestCheckpointProjectionV1(
        checkpoint_identity=checkpoint.identity,
        backbone=checkpoint.backbone,
        cognitive=checkpoint.cognitive,
        workspace=checkpoint.workspace,
        runtime=checkpoint.runtime,
        source_teacher_sidecar_sha256=checkpoint.sidecar_state_sha256,
        request_sidecar_state_sha256=request_sha256,
        completed_steps=checkpoint.completed_steps,
        state_names=tuple(state),
        tensor_count=len(state),
    )


def project_arm_teacher_checkpoint_to_request_v1(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> IrisArmRequestCheckpointProjectionV1:
    """Validate a B/C/D checkpoint and derive its redacted request-sidecar identity."""
    try:
        view = _checkpoint_view(checkpoint)
        state = _sidecar_state(view)
        return _projection(view, state)
    except IrisRequestCheckpointError:
        raise
    except Exception as error:
        message = "Teacher checkpoint could not be projected into the request topology"
        raise IrisRequestCheckpointError(message) from error


def _install(
    runtime: IrisArmRequestRuntimeV3,
    state: dict[str, torch.Tensor],
    *,
    expected_installed_sha256: str,
) -> IrisRequestSidecarInstallResultV1:
    orchestrator = runtime.orchestrator
    if runtime.profile.arm_id == "B":
        return cast(
            CognitiveNoWorkspaceRequestOrchestrator,
            orchestrator,
        ).restore_teacher_sidecar_state_v1(
            state,
            expected_installed_sha256=expected_installed_sha256,
        )
    if runtime.profile.arm_id == "C":
        return cast(
            CognitiveSharedBufferRequestOrchestrator,
            orchestrator,
        ).restore_teacher_sidecar_state_v1(
            state,
            expected_installed_sha256=expected_installed_sha256,
        )
    return cast(IrisRequestOrchestrator, orchestrator).restore_teacher_sidecar_state_v1(
        state,
        expected_installed_sha256=expected_installed_sha256,
    )


def transfer_arm_teacher_checkpoint_to_request_v1(
    *,
    runtime: IrisArmRequestRuntimeV3,
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> IrisArmRequestCheckpointTransferResultV1:
    """Validate and commit one B/v2, C/v3, or D/v1 sidecar into a request graph."""
    try:
        checked_runtime = _require_runtime(runtime)
        config = _runtime_config(checked_runtime)
        view = _checkpoint_view(checkpoint)
        _validate_compatibility(runtime=checked_runtime, config=config, checkpoint=view)
        state = _sidecar_state(view)
        projection = _projection(view, state)
        installation = _install(
            checked_runtime,
            state,
            expected_installed_sha256=projection.request_sidecar_state_sha256,
        )
        updated_runtime = replace(
            checked_runtime,
            sidecar_state_sha256=installation.installed_state_sha256,
        )
        return IrisArmRequestCheckpointTransferResultV1(
            runtime=updated_runtime,
            checkpoint_identity=view.identity,
            transfer_contract="exact_teacher_state_to_profile_v4_request_sidecar_v1",
            source_teacher_sidecar_sha256=projection.source_teacher_sidecar_sha256,
            previous_request_sidecar_sha256=installation.previous_state_sha256,
            installed_request_sidecar_sha256=installation.installed_state_sha256,
            completed_steps=projection.completed_steps,
            state_names=installation.state_names,
            transferred_tensor_count=installation.tensor_count,
        )
    except IrisRequestCheckpointError:
        raise
    except Exception as error:
        message = "Teacher checkpoint is incompatible with the profile-v4 request runtime"
        raise IrisRequestCheckpointError(message) from error


def load_arm_teacher_checkpoint_into_request_v1(
    *,
    runtime: IrisArmRequestRuntimeV3,
    identity: IrisArmTeacherCheckpointIdentity,
    root: Path,
) -> IrisArmRequestCheckpointTransferResultV1:
    """Read one exact content-addressed checkpoint and transfer only its sidecar state."""
    try:
        checked_runtime = _require_runtime(runtime)
        if type(identity) is not IrisArmTeacherCheckpointIdentity:
            message = "checkpoint identity must be exact before filesystem access"
            raise TypeError(message)
        if identity.arm_id != checked_runtime.profile.arm_id:
            message = (
                "checkpoint identity arm differs from request runtime before filesystem access"
            )
            raise ValueError(message)
        if not isinstance(root, Path) or not root.name:
            message = "checkpoint loading requires an explicit root Path"
            raise TypeError(message)
        checkpoint = read_arm_teacher_checkpoint(identity, root=root)
        return transfer_arm_teacher_checkpoint_to_request_v1(
            runtime=checked_runtime,
            checkpoint=checkpoint,
        )
    except IrisRequestCheckpointError:
        raise
    except Exception as error:
        message = "Teacher checkpoint could not be loaded into the request runtime"
        raise IrisRequestCheckpointError(message) from error


__all__ = [
    "IrisArmRequestCheckpointProjectionV1",
    "IrisArmRequestCheckpointTransferResultV1",
    "load_arm_teacher_checkpoint_into_request_v1",
    "project_arm_teacher_checkpoint_to_request_v1",
    "transfer_arm_teacher_checkpoint_to_request_v1",
]
