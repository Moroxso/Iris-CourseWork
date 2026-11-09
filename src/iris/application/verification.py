"""Content-redacted, read-only verification reports for installed Iris artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.errors import IrisVerificationError
from iris.application.experiment_execution import (
    ExperimentEpochPhaseV1,
    analyze_experiment_epoch_prefix_v1,
)
from iris.application.experiment_log import (
    IrisExperimentLogSnapshotV1,
    read_experiment_log_v1,
)
from iris.application.experiment_manifest import read_experiment_manifest_v1
from iris.application.request_artifact import read_request_artifact_v1
from iris.application.study_declaration import read_four_arm_protocol_declaration_v1
from iris.application.training_checkpoint import (
    IrisTeacherCheckpointBundleV1,
    read_teacher_checkpoint_v1,
)
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentCheckpointReferenceV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentTrainingConfigurationV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_v1 import IrisExperimentManifestV1


@dataclass(frozen=True, slots=True)
class IrisRequestArtifactVerificationReportV1:
    """Content-redacted verification summary for one request artifact."""

    path: Path
    artifact_type: Literal["iris.request"]
    version: Literal[1]
    payload_sha256: str
    configuration_sha256: str
    generated_tokens: int
    stop_reason: Literal["stop_token", "max_new_tokens"]
    workspace_cycles: int
    deterministic: bool
    backend: Literal["cpu", "cuda"]
    dtype: Literal["float16", "bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class IrisTeacherCheckpointVerificationReportV1:
    """Content-redacted verification summary for one teacher checkpoint."""

    path: Path
    artifact_type: Literal["iris.teacher_checkpoint"]
    version: Literal[1]
    payload_sha256: str
    state_file_sha256: str
    state_size_bytes: int
    tensor_count: int
    sidecar_state_sha256: str
    completed_steps: int
    dtype: Literal["float16", "bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class IrisExperimentManifestVerificationReportV1:
    """Content-redacted verification summary for one plaintext example manifest."""

    path: Path
    artifact_type: Literal["iris.experiment_examples"]
    version: Literal[1]
    payload_sha256: str
    dataset_id: str
    train_examples: int
    validation_examples: int
    test_examples: int
    total_examples: int


@dataclass(frozen=True, slots=True)
class IrisExperimentJournalVerificationReportV1:
    """Content-redacted summary for one journal and all of its references."""

    path: Path
    artifact_type: Literal["iris.experiment_run"]
    version: Literal[1]
    run_id: str
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
    latest_checkpoint_payload_sha256: str
    managed_phase: ExperimentEpochPhaseV1 | None
    completed_epochs: int | None
    active_epoch: int | None
    next_training_ordinal: int | None


@dataclass(frozen=True, slots=True)
class IrisFourArmProtocolVerificationReportV1:
    """Content-redacted summary for the recovered four-arm protocol facts."""

    path: Path
    artifact_type: Literal["iris.four_arm_protocol_declaration"]
    version: Literal[1]
    payload_sha256: str
    protocol_id: Literal["iris-gw-0.1-four-arm"]
    arm_count: Literal[4]
    task_family_count: Literal[5]
    unresolved_item_count: Literal[10]
    main_backbone_repository: Literal["Qwen/Qwen3-1.7B"]
    debug_backbone_repository: Literal["Qwen/Qwen3-0.6B"]
    execution_status: Literal["blocked_until_explicit_protocol_and_arm_implementations"]


def _require_real_file(path: Path, *, label: str) -> None:
    if not isinstance(path, Path) or not path.name:
        message = f"{label} verification requires an explicit file Path"
        raise TypeError(message)
    if path.is_symlink() or not path.is_file():
        message = f"{label} path must be a real regular file"
        raise ValueError(message)


def _canonical_artifact_root(path: Path, *, category: str, label: str) -> Path:
    if not isinstance(path, Path) or not path.name:
        message = f"{label} verification requires an explicit Path"
        raise TypeError(message)
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if version_directory.name != "v1" or category_directory.name != category or not root.name:
        message = f"{label} path must use the canonical <root>/{category}/v1 layout"
        raise ValueError(message)
    for directory, label in (
        (root, "artifact root"),
        (category_directory, f"{category} directory"),
        (version_directory, f"{category} version directory"),
    ):
        if directory.is_symlink() or not directory.is_dir():
            message = f"{label} must be a real directory"
            raise ValueError(message)
    return root


def _validate_manifest_reference(
    *,
    manifest: IrisExperimentManifestV1,
    snapshot: IrisExperimentLogSnapshotV1,
) -> None:
    owned_payload = manifest.payload
    reference = snapshot.run.payload.manifest
    if (
        owned_payload.dataset_id != reference.dataset_id
        or manifest.payload_sha256 != reference.payload_sha256
        or owned_payload.split_counts != reference.split_counts
    ):
        message = "experiment manifest does not match the journal run identity"
        raise ValueError(message)


def _validate_event_manifest_references(
    *,
    manifest: IrisExperimentManifestV1,
    records: tuple[
        ExperimentTrainingStepMetricV1 | ExperimentEvaluationMetricV1 | ExperimentCheckpointEventV1,
        ...,
    ],
) -> None:
    counts = manifest.payload.split_counts
    train_ids = tuple(
        example.example_id for example in manifest.payload.examples if example.split == "train"
    )
    for record in records:
        if type(record) is ExperimentTrainingStepMetricV1:
            if (
                record.split_example_count != counts.train
                or train_ids[record.split_ordinal - 1] != record.example_id
            ):
                message = "training event does not reference its manifest-ordered example"
                raise ValueError(message)
        elif type(record) is ExperimentEvaluationMetricV1:
            expected_count = counts.validation if record.split == "validation" else counts.test
            if record.example_count != expected_count:
                message = "evaluation event example_count does not match the manifest split"
                raise ValueError(message)


def _validate_checkpoint_reference(
    *,
    checkpoint: IrisTeacherCheckpointBundleV1,
    reference: ExperimentCheckpointReferenceV1,
) -> None:
    manifest = checkpoint.manifest
    payload = manifest.payload
    if (
        manifest.sha256 != reference.payload_sha256
        or payload.state_file.sha256 != reference.state_file_sha256
        or payload.model.sha256 != reference.sidecar_state_sha256
        or payload.optimizer.completed_steps != reference.completed_steps
    ):
        message = "referenced checkpoint does not match the experiment journal"
        raise ValueError(message)


def _validate_checkpoint_configuration(
    *,
    checkpoint: IrisTeacherCheckpointBundleV1,
    configuration: ExperimentTrainingConfigurationV1,
) -> None:
    payload = checkpoint.manifest.payload
    model = payload.model
    expected_model = configuration.model
    optimizer = payload.optimizer
    expected_optimizer = configuration.optimizer
    if (
        model.backbone != expected_model.backbone
        or model.cognitive != expected_model.cognitive
        or model.workspace != expected_model.workspace
        or model.runtime != expected_model.runtime
        or model.state_names != expected_model.state_names
        or optimizer.optimizer_type != expected_optimizer.optimizer_type
        or optimizer.config != expected_optimizer.config
        or optimizer.parameter_names != expected_optimizer.parameter_names
    ):
        message = "referenced checkpoint configuration does not match the experiment run"
        raise ValueError(message)


def verify_request_artifact_v1(path: Path) -> IrisRequestArtifactVerificationReportV1:
    """Verify one request artifact without returning private prompt or generated text."""
    try:
        _require_real_file(path, label="request artifact")
        root = _canonical_artifact_root(path, category="requests", label="request artifact")
        artifact = read_request_artifact_v1(path)
        if path.name != f"{artifact.payload_sha256}.json":
            message = "request artifact filename does not match its payload SHA-256"
            raise ValueError(message)
        payload = artifact.payload
        configured_root = Path(payload.configuration.value.runtime.output_dir)
        if configured_root.is_absolute() and configured_root != root.absolute():
            message = "request artifact path disagrees with its absolute configured output root"
            raise ValueError(message)
        runtime = payload.runtime
        return IrisRequestArtifactVerificationReportV1(
            path=path,
            artifact_type=artifact.schema_identity.artifact_type,
            version=artifact.schema_identity.version,
            payload_sha256=artifact.payload_sha256,
            configuration_sha256=payload.configuration.sha256,
            generated_tokens=payload.generation.generated_tokens,
            stop_reason=payload.generation.stop_reason,
            workspace_cycles=payload.workspace_trace.cycle_count,
            deterministic=runtime.deterministic,
            backend=runtime.backend,
            dtype=runtime.dtype,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris request artifact verification failed"
        raise IrisVerificationError(message) from error


def verify_teacher_checkpoint_v1(path: Path) -> IrisTeacherCheckpointVerificationReportV1:
    """Verify a checkpoint manifest and every encoded tensor without exposing values."""
    try:
        _canonical_artifact_root(path, category="teacher", label="teacher checkpoint")
        checkpoint = read_teacher_checkpoint_v1(path)
        manifest = checkpoint.manifest
        payload = manifest.payload
        return IrisTeacherCheckpointVerificationReportV1(
            path=path,
            artifact_type=manifest.schema_identity.artifact_type,
            version=manifest.schema_identity.version,
            payload_sha256=manifest.sha256,
            state_file_sha256=payload.state_file.sha256,
            state_size_bytes=payload.state_file.size_bytes,
            tensor_count=payload.state_file.tensor_count,
            sidecar_state_sha256=payload.model.sha256,
            completed_steps=payload.optimizer.completed_steps,
            dtype=payload.model.runtime.dtype,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris teacher checkpoint verification failed"
        raise IrisVerificationError(message) from error


def verify_experiment_manifest_v1(
    path: Path,
) -> IrisExperimentManifestVerificationReportV1:
    """Verify a plaintext manifest while omitting its private example content."""
    try:
        manifest = read_experiment_manifest_v1(path)
        counts = manifest.payload.split_counts
        return IrisExperimentManifestVerificationReportV1(
            path=path,
            artifact_type=manifest.schema_identity.artifact_type,
            version=manifest.schema_identity.version,
            payload_sha256=manifest.payload_sha256,
            dataset_id=manifest.payload.dataset_id,
            train_examples=counts.train,
            validation_examples=counts.validation,
            test_examples=counts.test,
            total_examples=counts.train + counts.validation + counts.test,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris experiment manifest verification failed"
        raise IrisVerificationError(message) from error


def verify_four_arm_protocol_declaration_v1(
    path: Path,
) -> IrisFourArmProtocolVerificationReportV1:
    """Verify recovered four-arm facts without claiming execution readiness."""
    try:
        declaration = read_four_arm_protocol_declaration_v1(path)
        payload = declaration.payload
        return IrisFourArmProtocolVerificationReportV1(
            path=path,
            artifact_type=declaration.schema_identity.artifact_type,
            version=declaration.schema_identity.version,
            payload_sha256=declaration.payload_sha256,
            protocol_id=payload.protocol_id,
            arm_count=4,
            task_family_count=5,
            unresolved_item_count=10,
            main_backbone_repository=payload.architecture.main_backbone_repository,
            debug_backbone_repository=payload.architecture.debug_backbone_repository,
            execution_status=payload.execution_status,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris four-arm protocol declaration verification failed"
        raise IrisVerificationError(message) from error


def verify_experiment_journal_v1(
    *,
    path: Path,
    manifest_path: Path,
) -> IrisExperimentJournalVerificationReportV1:
    """Verify a journal, its exact manifest, and every checkpoint reference."""
    try:
        root = _canonical_artifact_root(path, category="experiments", label="journal")
        _require_real_file(manifest_path, label="experiment manifest")
        manifest = read_experiment_manifest_v1(manifest_path)
        snapshot = read_experiment_log_v1(path)
        _validate_manifest_reference(manifest=manifest, snapshot=snapshot)

        run = snapshot.run
        configuration = run.payload.configuration
        policy = configuration.execution_policy
        records = tuple(event.payload.record for event in snapshot.events)
        _validate_event_manifest_references(manifest=manifest, records=records)
        checkpoint_references = [run.payload.initial_checkpoint]
        checkpoint_references.extend(
            record.checkpoint for record in records if type(record) is ExperimentCheckpointEventV1
        )

        teacher_directory = root / "teacher"
        teacher_version_directory = teacher_directory / "v1"
        for directory in (teacher_directory, teacher_version_directory):
            if directory.is_symlink() or not directory.is_dir():
                message = "journal checkpoint store must use real <root>/teacher/v1 directories"
                raise ValueError(message)
        verified_checkpoints: dict[str, IrisTeacherCheckpointBundleV1] = {}
        for reference in checkpoint_references:
            checkpoint = verified_checkpoints.get(reference.payload_sha256)
            if checkpoint is None:
                checkpoint = read_teacher_checkpoint_v1(
                    teacher_version_directory / reference.payload_sha256
                )
                verified_checkpoints[reference.payload_sha256] = checkpoint
            _validate_checkpoint_reference(checkpoint=checkpoint, reference=reference)
            _validate_checkpoint_configuration(
                checkpoint=checkpoint,
                configuration=configuration,
            )

        managed_phase: ExperimentEpochPhaseV1 | None = None
        completed_epochs: int | None = None
        active_epoch: int | None = None
        next_training_ordinal: int | None = None
        if type(policy) is ExperimentEpochExecutionPolicyV1:
            cursor = analyze_experiment_epoch_prefix_v1(
                run=run,
                events=snapshot.events,
                manifest=manifest,
            )
            managed_phase = cursor.phase
            completed_epochs = cursor.completed_epochs
            active_epoch = cursor.epoch_index
            next_training_ordinal = cursor.next_training_ordinal
            if cursor.latest_checkpoint != checkpoint_references[-1]:
                message = "managed epoch cursor disagrees with the latest checkpoint reference"
                raise ValueError(message)
        elif type(policy) is not ExperimentManualExecutionPolicyV1:
            message = "experiment journal contains an unsupported execution policy"
            raise TypeError(message)

        final_manifest = read_experiment_manifest_v1(manifest_path)
        final_snapshot = read_experiment_log_v1(path)
        if final_manifest != manifest:
            message = "experiment manifest changed while journal verification was running"
            raise ValueError(message)
        if (
            final_snapshot.run != snapshot.run
            or final_snapshot.event_count != snapshot.event_count
            or final_snapshot.chain_head_sha256 != snapshot.chain_head_sha256
        ):
            message = "experiment journal changed while verification was running"
            raise ValueError(message)

        latest_checkpoint = checkpoint_references[-1]
        return IrisExperimentJournalVerificationReportV1(
            path=path,
            artifact_type=run.schema_identity.artifact_type,
            version=run.schema_identity.version,
            run_id=run.payload.run_id,
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
                type(record) is ExperimentCheckpointEventV1 for record in records
            ),
            optimizer_completed_steps=snapshot.optimizer_completed_steps,
            sidecar_state_sha256=snapshot.sidecar_state_sha256,
            chain_head_sha256=snapshot.chain_head_sha256,
            checkpoint_reference_count=len(checkpoint_references),
            unique_checkpoint_count=len(verified_checkpoints),
            latest_checkpoint_payload_sha256=latest_checkpoint.payload_sha256,
            managed_phase=managed_phase,
            completed_epochs=completed_epochs,
            active_epoch=active_epoch,
            next_training_ordinal=next_training_ordinal,
        )
    except IrisVerificationError:
        raise
    except Exception as error:
        message = "Iris experiment journal verification failed"
        raise IrisVerificationError(message) from error


__all__ = [
    "IrisExperimentJournalVerificationReportV1",
    "IrisExperimentManifestVerificationReportV1",
    "IrisFourArmProtocolVerificationReportV1",
    "IrisRequestArtifactVerificationReportV1",
    "IrisTeacherCheckpointVerificationReportV1",
    "verify_experiment_journal_v1",
    "verify_experiment_manifest_v1",
    "verify_four_arm_protocol_declaration_v1",
    "verify_request_artifact_v1",
    "verify_teacher_checkpoint_v1",
]
