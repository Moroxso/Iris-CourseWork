"""Read-only verification for checkpoint-bound profile-v4 request artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.arm_profiles import (
    arm_execution_profile_set_sha256_v4,
    build_arm_execution_profile_set_v4,
    require_arm_request_executable_v4,
)
from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointProjectionV1,
    project_arm_teacher_checkpoint_to_request_v1,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointIdentity,
    read_arm_teacher_checkpoint,
)
from iris.application.errors import IrisVerificationError
from iris.application.request_artifact import _path_is_link_or_reparse
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_checkpoint_scope import _current_checkpoint_scope
from iris.artifacts.request_v3 import (
    RequestArmCTopologyV3,
    RequestArmIdV3,
    RequestArmKindV3,
    RequestGraphV3,
    RequestTraceContractV3,
    SharedBufferTraceProvenanceV3,
)
from iris.artifacts.request_v4 import IrisRequestArtifactPayloadV4


@dataclass(frozen=True, slots=True)
class IrisRequestArtifactVerificationReportV4:
    """Content-redacted summary of one request and its exact teacher source."""

    path: Path
    artifact_type: Literal["iris.request"]
    version: Literal[4]
    arm_id: RequestArmIdV3
    arm_kind: RequestArmKindV3
    request_graph: RequestGraphV3
    execution_profile_set_version: Literal[4]
    execution_profile_set_sha256: str
    shared_buffer_policy_sha256: str
    payload_sha256: str
    configuration_sha256: str
    generated_tokens: int
    stop_reason: Literal["stop_token", "max_new_tokens"]
    trace_contract: RequestTraceContractV3
    workspace_cycles: int
    cognitive_cycles: int
    latent_token_count: int
    zero_memory_read_cycles: int
    pre_update_read_entry_counts: tuple[int, ...]
    post_cycle_entry_counts: tuple[int, ...]
    final_output_read_entry_count: int | None
    sidecar_present: bool
    deterministic: bool
    backend: Literal["cpu", "cuda"]
    dtype: Literal["float16", "bfloat16", "float32"]
    checkpoint_present: bool
    checkpoint_schema_version: Literal[1, 2, 3] | None
    checkpoint_payload_sha256: str | None
    checkpoint_state_file_sha256: str | None
    source_teacher_sidecar_sha256: str | None
    request_sidecar_state_sha256: str | None
    checkpoint_completed_steps: int | None
    transferred_tensor_count: int


def _require_real_file(path: Path, *, label: str) -> None:
    if not isinstance(path, Path) or not path.name:
        message = f"{label} verification requires an explicit file Path"
        raise TypeError(message)
    if _path_is_link_or_reparse(path) or not path.is_file():
        message = f"{label} path must be a real regular file"
        raise ValueError(message)


def _canonical_request_root_v4(path: Path) -> Path:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v4" or category_directory.name != "requests" or not root.name:
        message = "request path must use the canonical <root>/requests/v4 layout"
        raise ValueError(message)
    for directory, label in (
        (root, "artifact root"),
        (category_directory, "requests directory"),
        (version_directory, "requests version directory"),
    ):
        if _path_is_link_or_reparse(directory) or not directory.is_dir():
            message = f"{label} must be a real directory"
            raise ValueError(message)
    return root


def _validate_profile_v4_binding(payload: IrisRequestArtifactPayloadV4) -> None:
    topology = payload.topology
    profile_set = build_arm_execution_profile_set_v4()
    if payload.execution_profile_set_sha256 != arm_execution_profile_set_sha256_v4():
        message = "request artifact profile-set identity is not the installed profile v4"
        raise ValueError(message)
    if payload.shared_buffer_policy_sha256 != profile_set.shared_buffer_policy_sha256:
        message = "request artifact shared-buffer policy is not the installed profile-v4 policy"
        raise ValueError(message)
    profile = require_arm_request_executable_v4(topology.arm_id)
    if (
        topology.arm_kind != profile.arm_kind
        or topology.request_graph != profile.request_graph
        or topology.trace_contract != profile.trace_contract
    ):
        message = "request artifact topology is not the installed executable profile v4"
        raise ValueError(message)


def _checkpoint_projection(
    identity: IrisArmTeacherCheckpointIdentity, *, root: Path
) -> IrisArmRequestCheckpointProjectionV1:
    def load() -> IrisArmRequestCheckpointProjectionV1:
        return project_arm_teacher_checkpoint_to_request_v1(
            read_arm_teacher_checkpoint(identity, root=root)
        )

    scope = _current_checkpoint_scope()
    return load() if scope is None else scope.projection(identity, root, load)


def _validate_checkpoint_binding(
    *,
    payload: IrisRequestArtifactPayloadV4,
    root: Path,
) -> IrisArmRequestCheckpointProjectionV1 | None:
    transfer = payload.checkpoint_transfer
    if transfer is None:
        if payload.topology.arm_id != "A":
            message = "only arm A may omit checkpoint transfer provenance"
            raise ValueError(message)
        return None
    identity = IrisArmTeacherCheckpointIdentity(
        arm_id=transfer.arm_id,
        checkpoint_schema_version=transfer.checkpoint_schema_version,
        payload_sha256=transfer.checkpoint_payload_sha256,
        state_file_sha256=transfer.checkpoint_state_file_sha256,
        sidecar_state_sha256=transfer.source_teacher_sidecar_sha256,
        completed_steps=transfer.completed_steps,
    )
    projection = _checkpoint_projection(identity, root=root)
    if projection.checkpoint_identity != identity:
        message = "request artifact checkpoint identity differs from canonical checkpoint"
        raise ValueError(message)
    if (
        projection.source_teacher_sidecar_sha256 != transfer.source_teacher_sidecar_sha256
        or projection.request_sidecar_state_sha256 != transfer.request_sidecar_state_sha256
        or projection.completed_steps != transfer.completed_steps
        or projection.state_names != transfer.transferred_state_names
        or projection.tensor_count != transfer.transferred_tensor_count
    ):
        message = "request artifact transfer does not match projected checkpoint tensor state"
        raise ValueError(message)

    config = payload.configuration.value
    if projection.backbone != config.backbone or projection.cognitive != config.cognitive:
        message = "request configuration differs from its teacher checkpoint"
        raise ValueError(message)
    if transfer.arm_id == "D":
        if projection.workspace != config.workspace:
            message = "arm-D request workspace configuration differs from its checkpoint"
            raise ValueError(message)
    elif projection.workspace is not None:
        message = f"arm-{transfer.arm_id} checkpoint unexpectedly owns workspace configuration"
        raise ValueError(message)

    checkpoint_runtime = projection.runtime
    request_runtime = payload.runtime
    if (
        checkpoint_runtime.max_prompt_tokens != config.backbone.limits.max_prompt_tokens
        or checkpoint_runtime.max_target_tokens != config.backbone.limits.max_new_tokens
        or checkpoint_runtime.device != request_runtime.device
        or checkpoint_runtime.dtype != request_runtime.dtype
    ):
        message = "request runtime placement or limits differ from its teacher checkpoint"
        raise ValueError(message)
    return projection


def _trace_counts(
    payload: IrisRequestArtifactPayloadV4,
) -> tuple[tuple[int, ...], tuple[int, ...], int | None, int]:
    topology = payload.topology
    trace = payload.workspace_trace
    if isinstance(topology, RequestArmCTopologyV3):
        if not isinstance(trace, SharedBufferTraceProvenanceV3):
            message = "arm-C request artifact requires its native shared-buffer trace"
            raise TypeError(message)
        pre_update = tuple(cycle.pre_update_read.entry_count for cycle in trace.cycles)
        post_cycle = tuple(cycle.post_cycle_state.entry_count for cycle in trace.cycles)
        return (
            pre_update,
            post_cycle,
            trace.final_output_read.entry_count,
            sum(count == 0 for count in pre_update),
        )
    return (), (), None, topology.zero_memory_read_cycles


def verify_request_artifact_v4(path: Path) -> IrisRequestArtifactVerificationReportV4:
    """Check request/checkpoint bindings at both boundaries without private output.

    Standalone calls deeply read/project the checkpoint twice. An owning study
    context may reuse its verified projection after rehashing all dependency bytes.
    """
    try:
        _require_real_file(path, label="request artifact")
        root = _canonical_request_root_v4(path)
        artifact = read_request_artifact_v4(path)
        if path.suffix != ".json" or path.stem != artifact.payload_sha256:
            message = "request artifact filename must equal its payload SHA-256"
            raise ValueError(message)

        payload = artifact.payload
        topology = payload.topology
        trace = payload.workspace_trace
        configured_root = Path(payload.configuration.value.runtime.output_dir)
        if configured_root.is_absolute() and configured_root != root.absolute():
            message = "request artifact path disagrees with its absolute configured output root"
            raise ValueError(message)
        _validate_profile_v4_binding(payload)
        projection = _validate_checkpoint_binding(payload=payload, root=root)
        pre_update, post_cycle, final_read, zero_reads = _trace_counts(payload)

        final_artifact = read_request_artifact_v4(path)
        if final_artifact != artifact:
            message = "request artifact changed during verification"
            raise ValueError(message)
        if projection is not None:
            final_projection = _checkpoint_projection(projection.checkpoint_identity, root=root)
            if final_projection != projection:
                message = "teacher checkpoint changed during request verification"
                raise ValueError(message)

        transfer = payload.checkpoint_transfer
        runtime = payload.runtime
        return IrisRequestArtifactVerificationReportV4(
            path=path,
            artifact_type=artifact.schema_identity.artifact_type,
            version=artifact.schema_identity.version,
            arm_id=topology.arm_id,
            arm_kind=topology.arm_kind,
            request_graph=topology.request_graph,
            execution_profile_set_version=payload.execution_profile_set_version,
            execution_profile_set_sha256=payload.execution_profile_set_sha256,
            shared_buffer_policy_sha256=payload.shared_buffer_policy_sha256,
            payload_sha256=artifact.payload_sha256,
            configuration_sha256=payload.configuration.sha256,
            generated_tokens=payload.generation.generated_tokens,
            stop_reason=payload.generation.stop_reason,
            trace_contract=topology.trace_contract,
            workspace_cycles=trace.cycle_count,
            cognitive_cycles=topology.cognitive_cycles,
            latent_token_count=topology.latent_token_count,
            zero_memory_read_cycles=zero_reads,
            pre_update_read_entry_counts=pre_update,
            post_cycle_entry_counts=post_cycle,
            final_output_read_entry_count=final_read,
            sidecar_present=runtime.sidecar_state_sha256 is not None,
            deterministic=runtime.deterministic,
            backend=runtime.backend,
            dtype=runtime.dtype,
            checkpoint_present=transfer is not None,
            checkpoint_schema_version=(
                None if transfer is None else transfer.checkpoint_schema_version
            ),
            checkpoint_payload_sha256=(
                None if transfer is None else transfer.checkpoint_payload_sha256
            ),
            checkpoint_state_file_sha256=(
                None if transfer is None else transfer.checkpoint_state_file_sha256
            ),
            source_teacher_sidecar_sha256=(
                None if transfer is None else transfer.source_teacher_sidecar_sha256
            ),
            request_sidecar_state_sha256=(
                None if transfer is None else transfer.request_sidecar_state_sha256
            ),
            checkpoint_completed_steps=(None if transfer is None else transfer.completed_steps),
            transferred_tensor_count=(0 if transfer is None else transfer.transferred_tensor_count),
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris checkpoint-bound request artifact verification failed"
        raise IrisVerificationError(message) from error


__all__ = ["IrisRequestArtifactVerificationReportV4", "verify_request_artifact_v4"]
