"""Content-redacted read-only verification for arm-labelled artifacts v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    read_arm_teacher_checkpoint,
)
from iris.application.errors import IrisVerificationError
from iris.application.experiment_execution_v2 import (
    ExperimentEpochPhaseV2,
    analyze_experiment_epoch_prefix_v2,
)
from iris.application.experiment_log_v2 import (
    IrisExperimentLogSnapshotV2,
    checkpoint_identity_from_reference_v2,
    read_experiment_log_v2,
    validate_experiment_checkpoint_for_run_v2,
)
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.request_artifact import _path_is_link_or_reparse
from iris.application.request_artifact_v2 import read_request_artifact_v2
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmIdV2,
    ExperimentArmKindV2,
    ExperimentCheckpointEventV2,
    ExperimentCheckpointReferenceV2,
    ExperimentEventRecordV2,
    ExperimentRequestGraphV2,
    ExperimentTeacherGraphV2,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1
from iris.artifacts.request_v2 import (
    RequestArmIdV2,
    RequestArmKindV2,
    RequestGraphV2,
    RequestTraceContractV2,
)


@dataclass(frozen=True, slots=True)
class IrisRequestArtifactVerificationReportV2:
    """Content-redacted summary for one A/B/D request artifact v2."""

    path: Path
    artifact_type: Literal["iris.request"]
    version: Literal[2]
    arm_id: RequestArmIdV2
    arm_kind: RequestArmKindV2
    request_graph: RequestGraphV2
    execution_profile_set_version: Literal[3]
    execution_profile_set_sha256: str
    payload_sha256: str
    configuration_sha256: str
    generated_tokens: int
    stop_reason: Literal["stop_token", "max_new_tokens"]
    trace_contract: RequestTraceContractV2
    workspace_cycles: int
    cognitive_cycles: int
    latent_token_count: int
    zero_memory_read_cycles: int
    sidecar_present: bool
    deterministic: bool
    backend: Literal["cpu", "cuda"]
    dtype: Literal["float16", "bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class IrisExperimentJournalVerificationReportV2:
    """Content-redacted summary for one B/D journal and every checkpoint."""

    path: Path
    artifact_type: Literal["iris.experiment_run"]
    version: Literal[2]
    run_id: str
    arm_id: ExperimentArmIdV2
    arm_kind: ExperimentArmKindV2
    request_graph: ExperimentRequestGraphV2
    teacher_graph: ExperimentTeacherGraphV2
    execution_profile_set_version: Literal[3]
    execution_profile_set_sha256: str
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
    latest_checkpoint_schema_version: Literal[1, 2]
    latest_checkpoint_payload_sha256: str
    managed_phase: ExperimentEpochPhaseV2 | None
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


def _canonical_request_root_v2(path: Path) -> Path:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v2" or category_directory.name != "requests" or not root.name:
        message = "request path must use the canonical <root>/requests/v2 layout"
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


def verify_request_artifact_v2(path: Path) -> IrisRequestArtifactVerificationReportV2:
    """Verify one canonical request v2 twice and reveal no prompt or response text."""
    try:
        _require_real_file(path, label="request artifact")
        root = _canonical_request_root_v2(path)
        artifact = read_request_artifact_v2(path)
        if path.suffix != ".json" or path.stem != artifact.payload_sha256:
            message = "request artifact filename must equal its payload SHA-256"
            raise ValueError(message)
        configured_root = Path(artifact.payload.configuration.value.runtime.output_dir)
        if configured_root.is_absolute() and configured_root != root.absolute():
            message = "request artifact path disagrees with its absolute configured output root"
            raise ValueError(message)
        final_artifact = read_request_artifact_v2(path)
        if final_artifact != artifact:
            message = "request artifact changed during verification"
            raise ValueError(message)
        payload = artifact.payload
        topology = payload.topology
        return IrisRequestArtifactVerificationReportV2(
            path=path,
            artifact_type=artifact.schema_identity.artifact_type,
            version=artifact.schema_identity.version,
            arm_id=topology.arm_id,
            arm_kind=topology.arm_kind,
            request_graph=topology.request_graph,
            execution_profile_set_version=payload.execution_profile_set_version,
            execution_profile_set_sha256=payload.execution_profile_set_sha256,
            payload_sha256=artifact.payload_sha256,
            configuration_sha256=payload.configuration.sha256,
            generated_tokens=payload.generation.generated_tokens,
            stop_reason=payload.generation.stop_reason,
            trace_contract=topology.trace_contract,
            workspace_cycles=payload.workspace_trace.cycle_count,
            cognitive_cycles=topology.cognitive_cycles,
            latent_token_count=topology.latent_token_count,
            zero_memory_read_cycles=topology.zero_memory_read_cycles,
            sidecar_present=payload.runtime.sidecar_state_sha256 is not None,
            deterministic=payload.runtime.deterministic,
            backend=payload.runtime.backend,
            dtype=payload.runtime.dtype,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris arm-labelled request artifact verification failed"
        raise IrisVerificationError(message) from error


def _canonical_artifact_root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.name:
        message = "journal verification requires an explicit run Path"
        raise TypeError(message)
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v2" or category_directory.name != "experiments" or not root.name:
        message = "journal path must use the canonical <root>/experiments/v2 layout"
        raise ValueError(message)
    for directory, label in (
        (root, "artifact root"),
        (category_directory, "experiments directory"),
        (version_directory, "experiments version directory"),
    ):
        if directory.is_symlink() or not directory.is_dir():
            message = f"{label} must be a real directory"
            raise ValueError(message)
    return root


def _validate_manifest_reference(
    *,
    manifest: IrisExperimentManifestV1,
    snapshot: IrisExperimentLogSnapshotV2,
) -> None:
    reference = snapshot.run.payload.manifest
    if (
        manifest.payload.dataset_id != reference.dataset_id
        or manifest.payload_sha256 != reference.payload_sha256
        or manifest.payload.split_counts != reference.split_counts
    ):
        message = "experiment manifest does not match the arm-labelled journal"
        raise ValueError(message)


def _validate_event_manifest_references(
    *,
    manifest: IrisExperimentManifestV1,
    records: tuple[ExperimentEventRecordV2, ...],
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


def _read_and_validate_checkpoints(
    *,
    root: Path,
    snapshot: IrisExperimentLogSnapshotV2,
    records: tuple[ExperimentEventRecordV2, ...],
) -> tuple[
    tuple[ExperimentCheckpointReferenceV2, ...],
    dict[tuple[int, str], IrisArmTeacherCheckpointBundle],
]:
    references = [snapshot.run.payload.initial_checkpoint]
    references.extend(
        record.checkpoint for record in records if type(record) is ExperimentCheckpointEventV2
    )
    verified: dict[tuple[int, str], IrisArmTeacherCheckpointBundle] = {}
    for reference in references:
        identity = checkpoint_identity_from_reference_v2(reference)
        key = (identity.checkpoint_schema_version, identity.payload_sha256)
        checkpoint = verified.get(key)
        if checkpoint is None:
            checkpoint = read_arm_teacher_checkpoint(identity, root=root)
            verified[key] = checkpoint
        if checkpoint.identity != identity:
            message = "referenced checkpoint identity changed between journal references"
            raise ValueError(message)
        validate_experiment_checkpoint_for_run_v2(
            checkpoint=checkpoint,
            run=snapshot.run,
        )
    return tuple(references), verified


def verify_experiment_journal_v2(
    *,
    path: Path,
    manifest_path: Path,
) -> IrisExperimentJournalVerificationReportV2:
    """Verify one B/D journal, exact manifest, FSM, and all checkpoints."""
    try:
        root = _canonical_artifact_root(path)
        _require_real_file(manifest_path, label="experiment manifest")
        manifest = read_experiment_manifest_v1(manifest_path)
        snapshot = read_experiment_log_v2(path)
        _validate_manifest_reference(manifest=manifest, snapshot=snapshot)

        run = snapshot.run
        configuration = run.payload.configuration
        policy = configuration.execution_policy
        records = tuple(event.payload.record for event in snapshot.events)
        _validate_event_manifest_references(manifest=manifest, records=records)
        checkpoint_references, verified_checkpoints = _read_and_validate_checkpoints(
            root=root,
            snapshot=snapshot,
            records=records,
        )

        managed_phase: ExperimentEpochPhaseV2 | None = None
        completed_epochs: int | None = None
        active_epoch: int | None = None
        next_training_ordinal: int | None = None
        if type(policy) is ExperimentEpochExecutionPolicyV1:
            cursor = analyze_experiment_epoch_prefix_v2(
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
        final_snapshot = read_experiment_log_v2(path)
        if final_manifest != manifest:
            message = "experiment manifest changed during journal verification"
            raise ValueError(message)
        if (
            final_snapshot.run != snapshot.run
            or final_snapshot.event_count != snapshot.event_count
            or final_snapshot.chain_head_sha256 != snapshot.chain_head_sha256
        ):
            message = "experiment journal changed during verification"
            raise ValueError(message)

        latest_checkpoint = checkpoint_references[-1]
        payload = run.payload
        return IrisExperimentJournalVerificationReportV2(
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
                type(record) is ExperimentCheckpointEventV2 for record in records
            ),
            optimizer_completed_steps=snapshot.optimizer_completed_steps,
            sidecar_state_sha256=snapshot.sidecar_state_sha256,
            chain_head_sha256=snapshot.chain_head_sha256,
            checkpoint_reference_count=len(checkpoint_references),
            unique_checkpoint_count=len(verified_checkpoints),
            latest_checkpoint_schema_version=(latest_checkpoint.checkpoint_schema_version),
            latest_checkpoint_payload_sha256=latest_checkpoint.payload_sha256,
            managed_phase=managed_phase,
            completed_epochs=completed_epochs,
            active_epoch=active_epoch,
            next_training_ordinal=next_training_ordinal,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris arm-labelled experiment journal verification failed"
        raise IrisVerificationError(message) from error


__all__ = [
    "IrisExperimentJournalVerificationReportV2",
    "IrisRequestArtifactVerificationReportV2",
    "verify_experiment_journal_v2",
    "verify_request_artifact_v2",
]
