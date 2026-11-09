"""Content-redacted read-only verification for profile-v4 delivery artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.arm_profiles import (
    arm_execution_profile_set_sha256_v4,
    build_arm_execution_profile_set_v4,
    require_arm_request_executable_v4,
    require_arm_teacher_executable_v4,
)
from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    read_arm_teacher_checkpoint,
)
from iris.application.errors import IrisVerificationError
from iris.application.experiment_execution_v3 import (
    ExperimentEpochPhaseV3,
    analyze_experiment_epoch_prefix_v3,
)
from iris.application.experiment_log_v3 import (
    IrisExperimentLogSnapshotV3,
    checkpoint_identity_from_reference_v3,
    read_experiment_log_v3,
    validate_experiment_checkpoint_for_run_v3,
)
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.request_artifact import _path_is_link_or_reparse
from iris.application.request_artifact_v3 import read_request_artifact_v3
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v3 import (
    ExperimentArmIdV3,
    ExperimentArmKindV3,
    ExperimentCheckpointEventV3,
    ExperimentCheckpointReferenceV3,
    ExperimentEventRecordV3,
    ExperimentRequestGraphV3,
    ExperimentTeacherGraphV3,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1
from iris.artifacts.request_v3 import (
    RequestArmCTopologyV3,
    RequestArmIdV3,
    RequestArmKindV3,
    RequestGraphV3,
    RequestTraceContractV3,
    SharedBufferTraceProvenanceV3,
)


@dataclass(frozen=True, slots=True)
class IrisRequestArtifactVerificationReportV3:
    """Content-redacted summary for one profile-v4 A/B/C/D request artifact."""

    path: Path
    artifact_type: Literal["iris.request"]
    version: Literal[3]
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


@dataclass(frozen=True, slots=True)
class IrisExperimentJournalVerificationReportV3:
    """Content-redacted summary for one profile-v4 B/C/D journal."""

    path: Path
    artifact_type: Literal["iris.experiment_run"]
    version: Literal[3]
    run_id: str
    arm_id: ExperimentArmIdV3
    arm_kind: ExperimentArmKindV3
    request_graph: ExperimentRequestGraphV3
    teacher_graph: ExperimentTeacherGraphV3
    execution_profile_set_version: Literal[4]
    execution_profile_set_sha256: str
    shared_buffer_policy_sha256: str
    run_payload_sha256: str
    configuration_sha256: str
    manifest_payload_sha256: str
    execution_mode: Literal["manual_event_chain", "deterministic_epochs"]
    event_count: int
    training_step_event_count: int
    evaluation_event_count: int
    checkpoint_event_count: int
    optimizer_completed_steps: int
    sidecar_state_sha256: str
    chain_head_sha256: str
    checkpoint_reference_count: int
    unique_checkpoint_count: int
    latest_checkpoint_schema_version: Literal[1, 2, 3]
    latest_checkpoint_payload_sha256: str
    managed_phase: ExperimentEpochPhaseV3 | None
    completed_epochs: int | None
    active_epoch: int | None
    next_training_ordinal: int | None


def _require_real_file(path: Path, *, label: str) -> None:
    if not isinstance(path, Path) or not path.name:
        message = f"{label} verification requires an explicit file Path"
        raise TypeError(message)
    if _path_is_link_or_reparse(path) or not path.is_file():
        message = f"{label} path must be a real regular file"
        raise ValueError(message)


def _canonical_request_root_v3(path: Path) -> Path:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v3" or category_directory.name != "requests" or not root.name:
        message = "request path must use the canonical <root>/requests/v3 layout"
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


def _validate_profile_v4_binding(
    *,
    arm_id: RequestArmIdV3,
    arm_kind: RequestArmKindV3,
    request_graph: RequestGraphV3,
    trace_contract: RequestTraceContractV3,
    execution_profile_set_sha256: str,
    shared_buffer_policy_sha256: str,
) -> None:
    profile_set = build_arm_execution_profile_set_v4()
    if execution_profile_set_sha256 != arm_execution_profile_set_sha256_v4():
        message = "request artifact profile-set identity is not the installed profile v4"
        raise ValueError(message)
    if shared_buffer_policy_sha256 != profile_set.shared_buffer_policy_sha256:
        message = "request artifact shared-buffer policy is not the installed profile-v4 policy"
        raise ValueError(message)
    profile = require_arm_request_executable_v4(arm_id)
    if (
        arm_kind != profile.arm_kind
        or request_graph != profile.request_graph
        or trace_contract != profile.trace_contract
    ):
        message = "request artifact topology is not the installed executable profile v4"
        raise ValueError(message)


def verify_request_artifact_v3(path: Path) -> IrisRequestArtifactVerificationReportV3:
    """Verify one canonical request v3 twice without revealing private values or tensor hashes."""
    try:
        _require_real_file(path, label="request artifact")
        root = _canonical_request_root_v3(path)
        artifact = read_request_artifact_v3(path)
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
        _validate_profile_v4_binding(
            arm_id=topology.arm_id,
            arm_kind=topology.arm_kind,
            request_graph=topology.request_graph,
            trace_contract=topology.trace_contract,
            execution_profile_set_sha256=payload.execution_profile_set_sha256,
            shared_buffer_policy_sha256=payload.shared_buffer_policy_sha256,
        )

        pre_update_read_entry_counts: tuple[int, ...] = ()
        post_cycle_entry_counts: tuple[int, ...] = ()
        final_output_read_entry_count: int | None = None
        if isinstance(topology, RequestArmCTopologyV3):
            if not isinstance(trace, SharedBufferTraceProvenanceV3):
                message = "arm-C request artifact requires its native shared-buffer trace"
                raise TypeError(message)
            pre_update_read_entry_counts = tuple(
                cycle.pre_update_read.entry_count for cycle in trace.cycles
            )
            post_cycle_entry_counts = tuple(
                cycle.post_cycle_state.entry_count for cycle in trace.cycles
            )
            final_output_read_entry_count = trace.final_output_read.entry_count
            zero_memory_read_cycles = sum(count == 0 for count in pre_update_read_entry_counts)
        else:
            zero_memory_read_cycles = topology.zero_memory_read_cycles

        final_artifact = read_request_artifact_v3(path)
        if final_artifact != artifact:
            message = "request artifact changed during verification"
            raise ValueError(message)

        runtime = payload.runtime
        return IrisRequestArtifactVerificationReportV3(
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
            zero_memory_read_cycles=zero_memory_read_cycles,
            pre_update_read_entry_counts=pre_update_read_entry_counts,
            post_cycle_entry_counts=post_cycle_entry_counts,
            final_output_read_entry_count=final_output_read_entry_count,
            sidecar_present=runtime.sidecar_state_sha256 is not None,
            deterministic=runtime.deterministic,
            backend=runtime.backend,
            dtype=runtime.dtype,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris profile-v4 request artifact verification failed"
        raise IrisVerificationError(message) from error


def _canonical_experiment_root_v3(path: Path) -> Path:
    if not isinstance(path, Path) or not path.name:
        message = "journal verification requires an explicit run directory Path"
        raise TypeError(message)
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v3" or category_directory.name != "experiments" or not root.name:
        message = "journal path must use the canonical <root>/experiments/v3 layout"
        raise ValueError(message)
    events_directory = path / "events"
    for directory, label in (
        (root, "artifact root"),
        (category_directory, "experiments directory"),
        (version_directory, "experiments version directory"),
        (path, "experiment run directory"),
        (events_directory, "experiment events directory"),
    ):
        if _path_is_link_or_reparse(directory) or not directory.is_dir():
            message = f"{label} must be a real directory"
            raise ValueError(message)
    _require_real_file(path / "run.json", label="experiment run")
    for event_path in events_directory.iterdir():
        _require_real_file(event_path, label="experiment event")
    return root


def _validate_experiment_profile_v4_binding(snapshot: IrisExperimentLogSnapshotV3) -> None:
    payload = snapshot.run.payload
    profile_set = build_arm_execution_profile_set_v4()
    if payload.execution_profile_set_sha256 != arm_execution_profile_set_sha256_v4():
        message = "experiment journal profile-set identity is not the installed profile v4"
        raise ValueError(message)
    if (
        payload.protocol_declaration_payload_sha256
        != profile_set.protocol_declaration_payload_sha256
    ):
        message = "experiment journal protocol identity is not the installed profile-v4 protocol"
        raise ValueError(message)
    if payload.shared_buffer_policy_sha256 != profile_set.shared_buffer_policy_sha256:
        message = "experiment journal shared-buffer policy is not the installed profile-v4 policy"
        raise ValueError(message)
    profile = require_arm_teacher_executable_v4(payload.arm_id)
    if (
        payload.arm_kind != profile.arm_kind
        or payload.request_graph != profile.request_graph
        or payload.teacher_graph != profile.teacher_graph
    ):
        message = "experiment journal topology is not the installed executable profile v4"
        raise ValueError(message)


def _validate_manifest_reference_v3(
    *,
    manifest: IrisExperimentManifestV1,
    snapshot: IrisExperimentLogSnapshotV3,
) -> None:
    reference = snapshot.run.payload.manifest
    if (
        manifest.payload.dataset_id != reference.dataset_id
        or manifest.payload_sha256 != reference.payload_sha256
        or manifest.payload.split_counts != reference.split_counts
    ):
        message = "experiment manifest does not match the profile-v4 journal"
        raise ValueError(message)


def _validate_event_manifest_references_v3(
    *,
    manifest: IrisExperimentManifestV1,
    records: tuple[ExperimentEventRecordV3, ...],
) -> None:
    counts = manifest.payload.split_counts
    train_ids = tuple(
        example.example_id for example in manifest.payload.examples if example.split == "train"
    )
    for record in records:
        if type(record) is ExperimentTrainingStepMetricV1:
            if (
                record.split_example_count != counts.train
                or record.split_ordinal < 1
                or record.split_ordinal > len(train_ids)
                or train_ids[record.split_ordinal - 1] != record.example_id
            ):
                message = "training event does not reference its manifest-ordered example"
                raise ValueError(message)
        elif type(record) is ExperimentEvaluationMetricV1:
            expected_count = counts.validation if record.split == "validation" else counts.test
            if record.example_count != expected_count:
                message = "evaluation event count does not match its manifest split"
                raise ValueError(message)


def _read_and_validate_checkpoints_v3(
    *,
    root: Path,
    snapshot: IrisExperimentLogSnapshotV3,
    records: tuple[ExperimentEventRecordV3, ...],
) -> tuple[
    tuple[ExperimentCheckpointReferenceV3, ...],
    dict[tuple[ExperimentArmIdV3, int, str], IrisArmTeacherCheckpointBundle],
]:
    references = [snapshot.run.payload.initial_checkpoint]
    references.extend(
        record.checkpoint for record in records if type(record) is ExperimentCheckpointEventV3
    )
    verified: dict[
        tuple[ExperimentArmIdV3, int, str], IrisArmTeacherCheckpointBundle
    ] = {}
    for reference in references:
        identity = checkpoint_identity_from_reference_v3(reference)
        key = (
            identity.arm_id,
            identity.checkpoint_schema_version,
            identity.payload_sha256,
        )
        checkpoint = verified.get(key)
        if checkpoint is None:
            checkpoint = read_arm_teacher_checkpoint(identity, root=root)
            verified[key] = checkpoint
        if checkpoint.identity != identity:
            message = "referenced checkpoint identity changed between journal references"
            raise ValueError(message)
        validate_experiment_checkpoint_for_run_v3(
            checkpoint=checkpoint,
            run=snapshot.run,
        )
    return tuple(references), verified


def _reread_checkpoints_v3(
    *,
    root: Path,
    snapshot: IrisExperimentLogSnapshotV3,
    verified: dict[tuple[ExperimentArmIdV3, int, str], IrisArmTeacherCheckpointBundle],
) -> None:
    for checkpoint in verified.values():
        final_checkpoint = read_arm_teacher_checkpoint(checkpoint.identity, root=root)
        final_checkpoint = validate_experiment_checkpoint_for_run_v3(
            checkpoint=final_checkpoint,
            run=snapshot.run,
        )
        if final_checkpoint.identity != checkpoint.identity:
            message = "experiment checkpoint changed during journal verification"
            raise ValueError(message)


def verify_experiment_journal_v3(
    *,
    path: Path,
    manifest_path: Path,
) -> IrisExperimentJournalVerificationReportV3:
    """Verify one canonical B/C/D journal, manifest, FSM, and every checkpoint twice."""
    try:
        root = _canonical_experiment_root_v3(path)
        _require_real_file(manifest_path, label="experiment manifest")
        manifest = read_experiment_manifest_v1(manifest_path)
        snapshot = read_experiment_log_v3(path)
        _validate_experiment_profile_v4_binding(snapshot)
        _validate_manifest_reference_v3(manifest=manifest, snapshot=snapshot)

        run = snapshot.run
        configuration = run.payload.configuration
        policy = configuration.execution_policy
        records = tuple(event.payload.record for event in snapshot.events)
        _validate_event_manifest_references_v3(manifest=manifest, records=records)
        checkpoint_references, verified_checkpoints = _read_and_validate_checkpoints_v3(
            root=root,
            snapshot=snapshot,
            records=records,
        )

        managed_phase: ExperimentEpochPhaseV3 | None = None
        completed_epochs: int | None = None
        active_epoch: int | None = None
        next_training_ordinal: int | None = None
        if type(policy) is ExperimentEpochExecutionPolicyV1:
            cursor = analyze_experiment_epoch_prefix_v3(
                run=run,
                events=snapshot.events,
                manifest=manifest,
            )
            managed_phase = cursor.phase
            completed_epochs = cursor.completed_epochs
            active_epoch = cursor.epoch_index
            next_training_ordinal = cursor.next_training_ordinal
            if cursor.latest_checkpoint != checkpoint_references[-1]:
                message = "managed cursor disagrees with the latest checkpoint reference"
                raise ValueError(message)
        elif type(policy) is not ExperimentManualExecutionPolicyV1:
            message = "experiment journal contains an unsupported execution policy"
            raise TypeError(message)

        final_manifest = read_experiment_manifest_v1(manifest_path)
        final_snapshot = read_experiment_log_v3(path)
        if final_manifest != manifest:
            message = "experiment manifest changed during journal verification"
            raise ValueError(message)
        if final_snapshot != snapshot:
            message = "experiment journal changed during verification"
            raise ValueError(message)
        _reread_checkpoints_v3(
            root=root,
            snapshot=final_snapshot,
            verified=verified_checkpoints,
        )

        latest_checkpoint = checkpoint_references[-1]
        payload = run.payload
        return IrisExperimentJournalVerificationReportV3(
            path=path,
            artifact_type=run.schema_identity.artifact_type,
            version=run.schema_identity.version,
            run_id=payload.run_id,
            arm_id=payload.arm_id,
            arm_kind=payload.arm_kind,
            request_graph=payload.request_graph,
            teacher_graph=payload.teacher_graph,
            execution_profile_set_version=payload.execution_profile_set_version,
            execution_profile_set_sha256=payload.execution_profile_set_sha256,
            shared_buffer_policy_sha256=payload.shared_buffer_policy_sha256,
            run_payload_sha256=run.payload_sha256,
            configuration_sha256=configuration.sha256,
            manifest_payload_sha256=manifest.payload_sha256,
            execution_mode=policy.mode,
            event_count=snapshot.event_count,
            training_step_event_count=sum(
                type(record) is ExperimentTrainingStepMetricV1 for record in records
            ),
            evaluation_event_count=sum(
                type(record) is ExperimentEvaluationMetricV1 for record in records
            ),
            checkpoint_event_count=sum(
                type(record) is ExperimentCheckpointEventV3 for record in records
            ),
            optimizer_completed_steps=snapshot.optimizer_completed_steps,
            sidecar_state_sha256=snapshot.sidecar_state_sha256,
            chain_head_sha256=snapshot.chain_head_sha256,
            checkpoint_reference_count=len(checkpoint_references),
            unique_checkpoint_count=len(verified_checkpoints),
            latest_checkpoint_schema_version=latest_checkpoint.checkpoint_schema_version,
            latest_checkpoint_payload_sha256=latest_checkpoint.payload_sha256,
            managed_phase=managed_phase,
            completed_epochs=completed_epochs,
            active_epoch=active_epoch,
            next_training_ordinal=next_training_ordinal,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris profile-v4 experiment journal verification failed"
        raise IrisVerificationError(message) from error


__all__ = [
    "IrisExperimentJournalVerificationReportV3",
    "IrisRequestArtifactVerificationReportV3",
    "verify_experiment_journal_v3",
    "verify_request_artifact_v3",
]
