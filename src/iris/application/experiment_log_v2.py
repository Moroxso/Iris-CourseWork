"""Canonical arm-labelled B/D experiment journals and event chains v2."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from re import fullmatch
from time import sleep
from typing import Any, Literal, cast
from uuid import uuid4

import torch

from iris.application.arm_training_checkpoint import (
    IrisArmTeacherCheckpointBundle,
    IrisArmTeacherCheckpointIdentity,
    IrisArmTeacherCheckpointWriteResult,
    IrisArmTeacherModel,
    IrisTeacherArmId,
    build_arm_teacher_checkpoint,
    validate_arm_teacher_checkpoint,
    write_arm_teacher_checkpoint,
)
from iris.application.errors import IrisExperimentLogError, IrisTeacherCheckpointError
from iris.application.experiment_execution_v2 import (
    IrisExperimentEpochProgressV2,
    analyze_experiment_epoch_prefix_v2,
    extend_experiment_epoch_progress_v2,
    validate_experiment_epoch_candidate_v2,
)
from iris.application.experiment_manifest import (
    IrisPreparedExperimentExample,
    validate_experiment_manifest_v1,
)
from iris.application.no_workspace_training_checkpoint import (
    IrisNoWorkspaceTeacherCheckpointBundleV2,
)
from iris.application.training_checkpoint import IrisTeacherCheckpointBundleV1
from iris.artifacts.checkpoint_v1 import TeacherAdamWConfigV1, TeacherBackboneRuntimeV1
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentExecutionPolicyV1,
    ExperimentManifestReferenceV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentOptimizerConfigurationV1,
    ExperimentRuntimePolicyV1,
    ExperimentTargetPreparationPolicyV1,
    ExperimentTrainingStepMetricV1,
)
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmBCheckpointReferenceV2,
    ExperimentArmBModelConfigurationV2,
    ExperimentArmDCheckpointReferenceV2,
    ExperimentArmDModelConfigurationV2,
    ExperimentArmIdV2,
    ExperimentCheckpointEventV2,
    ExperimentCheckpointReferenceV2,
    ExperimentEventPayloadV2,
    ExperimentEventRecordV2,
    ExperimentEventSchemaV2,
    ExperimentModelConfigurationV2,
    ExperimentRunPayloadV2,
    ExperimentRunSchemaV2,
    ExperimentTrainingConfigurationV2,
    IrisExperimentEventV2,
    IrisExperimentRunV2,
)
from iris.artifacts.experiment_v1 import ExperimentSplitV1, IrisExperimentManifestV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training import (
    IrisNoWorkspaceTeacherForcingModel,
    IrisTeacherForcingInput,
    IrisTeacherForcingModel,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerProvenance,
    IrisTokenWeightedEvaluationSummary,
)
from iris.training.outputs import (
    IrisNoWorkspaceTeacherCheckpointProvenanceV1,
    IrisTeacherModelCheckpointProvenance,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_CONFIGURATION_DOMAIN = "iris/experiment-log/v2/configuration"
_RUN_PAYLOAD_DOMAIN = "iris/experiment-log/v2/run-payload"
_EVENT_PAYLOAD_DOMAIN = "iris/experiment-log/v2/event-payload"
_PROTOCOL_DECLARATION_PAYLOAD_SHA256 = (
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
)
_EXECUTION_PROFILE_SET_SHA256 = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
_RUN_FILENAME = "run.json"
_EVENTS_DIRECTORY = "events"
_MAX_RUN_BYTES = 1 << 20
_MAX_EVENT_BYTES = 64 << 10
_MAX_EVENT_COUNT = 1_000_000
_WINDOWS_PERMISSION_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16)
_ARM_D_COMPONENT_NAMES = (
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


@dataclass(frozen=True, slots=True)
class IrisExperimentLogCreateResultV2:
    """Installed arm-labelled run identity and its exact initial checkpoint."""

    arm_id: ExperimentArmIdV2
    path: Path
    run_payload_sha256: str
    configuration_sha256: str
    initial_checkpoint_path: Path
    initial_checkpoint_payload_sha256: str
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class IrisExperimentEventWriteResultV2:
    """Result of one canonical event install or exact retry."""

    arm_id: ExperimentArmIdV2
    path: Path
    sequence: int
    event_type: Literal["training_step", "evaluation", "checkpoint"]
    payload_sha256: str
    reused_existing: bool
    referenced_checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class IrisExperimentLogSnapshotV2:
    """One fully verified immutable v2 run and contiguous event prefix."""

    path: Path
    run: IrisExperimentRunV2
    events: tuple[IrisExperimentEventV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.name:
            message = "experiment snapshot path must be an explicit Path"
            raise TypeError(message)
        if type(self.run) is not IrisExperimentRunV2:
            message = "experiment snapshot requires exactly IrisExperimentRunV2"
            raise TypeError(message)
        if type(self.events) is not tuple or any(
            type(event) is not IrisExperimentEventV2 for event in self.events
        ):
            message = "experiment snapshot requires an exact v2 event tuple"
            raise TypeError(message)

    @property
    def arm_id(self) -> ExperimentArmIdV2:
        return self.run.payload.arm_id

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def chain_head_sha256(self) -> str:
        return self.events[-1].payload_sha256 if self.events else self.run.payload_sha256

    @property
    def optimizer_completed_steps(self) -> int:
        if self.events:
            return self.events[-1].payload.optimizer_completed_steps
        return self.run.payload.initial_checkpoint.completed_steps

    @property
    def sidecar_state_sha256(self) -> str:
        if self.events:
            return self.events[-1].payload.sidecar_state_sha256
        return self.run.payload.initial_checkpoint.sidecar_state_sha256


@dataclass(frozen=True, slots=True)
class IrisExperimentTrainingStateV2:
    """Content-redacted identity of one live B/D owner against a v2 journal."""

    arm_id: ExperimentArmIdV2
    configuration_sha256: str
    chain_head_sha256: str
    event_count: int
    optimizer_completed_steps: int
    sidecar_state_sha256: str


@dataclass(frozen=True, slots=True)
class _LiveTrainingIdentity:
    configuration: ExperimentTrainingConfigurationV2
    optimizer_completed_steps: int
    sidecar_state_sha256: str


def _require_arm_id(value: object) -> ExperimentArmIdV2:
    if type(value) is not str or value not in {"B", "D"}:
        message = "experiment arm_id must be exactly B or D"
        raise ValueError(message)
    return cast(ExperimentArmIdV2, value)


def _owned_execution_policy(policy: ExperimentExecutionPolicyV1) -> ExperimentExecutionPolicyV1:
    if type(policy) is ExperimentManualExecutionPolicyV1:
        return ExperimentManualExecutionPolicyV1.model_validate(policy.model_dump(mode="python"))
    if type(policy) is ExperimentEpochExecutionPolicyV1:
        return ExperimentEpochExecutionPolicyV1.model_validate(policy.model_dump(mode="python"))
    message = "execution_policy must be an exact supported experiment policy"
    raise TypeError(message)


def _configuration_content(configuration: ExperimentTrainingConfigurationV2) -> dict[str, Any]:
    return {
        "execution_policy": configuration.execution_policy.model_dump(mode="json"),
        "model": configuration.model.model_dump(mode="json"),
        "optimizer": configuration.optimizer.model_dump(mode="json"),
        "runtime_policy": configuration.runtime_policy.model_dump(mode="json"),
        "target_preparation": configuration.target_preparation.model_dump(mode="json"),
    }


def _configuration(
    *,
    model: ExperimentModelConfigurationV2,
    optimizer: ExperimentOptimizerConfigurationV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
) -> ExperimentTrainingConfigurationV2:
    target_policy = ExperimentTargetPreparationPolicyV1(
        policy="raw_target_no_special_tokens_no_truncation_plus_one_eos",
        terminal_token_id=terminal_token_id,
    )
    runtime_policy = ExperimentRuntimePolicyV1(
        initialization_seed=initialization_seed,
        deterministic=deterministic,
    )
    owned_execution_policy = _owned_execution_policy(execution_policy)
    content = {
        "execution_policy": owned_execution_policy.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
        "optimizer": optimizer.model_dump(mode="json"),
        "runtime_policy": runtime_policy.model_dump(mode="json"),
        "target_preparation": target_policy.model_dump(mode="json"),
    }
    return ExperimentTrainingConfigurationV2(
        sha256=domain_sha256(_CONFIGURATION_DOMAIN, canonical_json_bytes(content)),
        model=model,
        optimizer=optimizer,
        target_preparation=target_policy,
        runtime_policy=runtime_policy,
        execution_policy=owned_execution_policy,
    )


def _validate_configuration(
    configuration: ExperimentTrainingConfigurationV2,
) -> ExperimentTrainingConfigurationV2:
    owned = ExperimentTrainingConfigurationV2.model_validate(
        configuration.model_dump(mode="python")
    )
    expected = domain_sha256(
        _CONFIGURATION_DOMAIN,
        canonical_json_bytes(_configuration_content(owned)),
    )
    if owned.sha256 != expected:
        message = "arm-labelled experiment configuration SHA-256 is inconsistent"
        raise ValueError(message)
    return owned


def _validate_ambient_runtime_policy(policy: ExperimentRuntimePolicyV1) -> None:
    try:
        actual_seed = torch.initial_seed()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    except Exception as error:
        message = "experiment logger could not inspect ambient torch runtime policy"
        raise ValueError(message) from error
    if (
        type(actual_seed) is not int
        or actual_seed != policy.initialization_seed
        or type(deterministic) is not bool
        or deterministic is not policy.deterministic
        or (policy.deterministic and warn_only)
    ):
        message = "ambient torch seed or strict deterministic policy disagrees with the run"
        raise ValueError(message)


def _runtime_from_values(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    vocabulary_size: int,
    max_position_embeddings: int,
    max_prompt_tokens: int,
    max_target_tokens: int,
    device: str,
    dtype: str,
) -> TeacherBackboneRuntimeV1:
    return TeacherBackboneRuntimeV1.model_validate(
        {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "vocabulary_size": vocabulary_size,
            "max_position_embeddings": max_position_embeddings,
            "max_prompt_tokens": max_prompt_tokens,
            "max_target_tokens": max_target_tokens,
            "device": device,
            "dtype": dtype,
        }
    )


def _model_configuration_from_checkpoint(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> ExperimentModelConfigurationV2:
    verified = validate_arm_teacher_checkpoint(checkpoint)
    if verified.identity.arm_id == "B":
        arm_b_payload = cast(
            IrisNoWorkspaceTeacherCheckpointBundleV2, verified.payload
        ).manifest.payload
        arm_b_model_identity = arm_b_payload.model
        return ExperimentArmBModelConfigurationV2(
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
            backbone=arm_b_model_identity.backbone,
            cognitive=arm_b_model_identity.cognitive,
            cognitive_cycles=arm_b_model_identity.cognitive_cycles,
            latent_token_count=arm_b_model_identity.latent_token_count,
            component_names=arm_b_model_identity.component_names,
            runtime=arm_b_model_identity.runtime,
            state_names=arm_b_model_identity.state_names,
        )
    arm_d_payload = cast(IrisTeacherCheckpointBundleV1, verified.payload).manifest.payload
    arm_d_model_identity = arm_d_payload.model
    return ExperimentArmDModelConfigurationV2(
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        teacher_graph="limited_competitive_workspace_teacher_v1",
        backbone=arm_d_model_identity.backbone,
        cognitive=arm_d_model_identity.cognitive,
        workspace=arm_d_model_identity.workspace,
        component_names=_ARM_D_COMPONENT_NAMES,
        runtime=arm_d_model_identity.runtime,
        state_names=arm_d_model_identity.state_names,
    )


def _optimizer_configuration_from_checkpoint(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> ExperimentOptimizerConfigurationV1:
    verified = validate_arm_teacher_checkpoint(checkpoint)
    if verified.identity.arm_id == "B":
        optimizer = cast(
            IrisNoWorkspaceTeacherCheckpointBundleV2, verified.payload
        ).manifest.payload.optimizer
    else:
        optimizer = cast(IrisTeacherCheckpointBundleV1, verified.payload).manifest.payload.optimizer
    return ExperimentOptimizerConfigurationV1(
        optimizer_type=optimizer.optimizer_type,
        config=optimizer.config,
        parameter_names=optimizer.parameter_names,
    )


def _checkpoint_reference(
    checkpoint: IrisArmTeacherCheckpointBundle,
) -> ExperimentCheckpointReferenceV2:
    verified = validate_arm_teacher_checkpoint(checkpoint)
    identity = verified.identity
    if identity.arm_id == "B":
        return ExperimentArmBCheckpointReferenceV2(
            checkpoint_schema_version=2,
            arm_id="B",
            payload_sha256=identity.payload_sha256,
            state_file_sha256=identity.state_file_sha256,
            sidecar_state_sha256=identity.sidecar_state_sha256,
            completed_steps=identity.completed_steps,
        )
    return ExperimentArmDCheckpointReferenceV2(
        checkpoint_schema_version=1,
        arm_id="D",
        payload_sha256=identity.payload_sha256,
        state_file_sha256=identity.state_file_sha256,
        sidecar_state_sha256=identity.sidecar_state_sha256,
        completed_steps=identity.completed_steps,
    )


def _configuration_from_checkpoint(
    checkpoint: IrisArmTeacherCheckpointBundle,
    *,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
) -> ExperimentTrainingConfigurationV2:
    return _configuration(
        model=_model_configuration_from_checkpoint(checkpoint),
        optimizer=_optimizer_configuration_from_checkpoint(checkpoint),
        terminal_token_id=terminal_token_id,
        initialization_seed=initialization_seed,
        deterministic=deterministic,
        execution_policy=execution_policy,
    )


def _run_bytes(run: IrisExperimentRunV2) -> tuple[IrisExperimentRunV2, bytes]:
    if type(run) is not IrisExperimentRunV2:
        message = "experiment run must be exactly IrisExperimentRunV2"
        raise TypeError(message)
    owned = IrisExperimentRunV2.model_validate(run.model_dump(mode="python"))
    _validate_configuration(owned.payload.configuration)
    expected = domain_sha256(
        _RUN_PAYLOAD_DOMAIN,
        canonical_json_bytes(owned.payload.model_dump(mode="json")),
    )
    if owned.payload_sha256 != expected:
        message = "arm-labelled experiment run payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_RUN_BYTES:
        message = "arm-labelled experiment run exceeds the 1 MiB safety limit"
        raise ValueError(message)
    return owned, content


def build_experiment_run_v2(
    *,
    run_id: str,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
    initial_checkpoint: IrisArmTeacherCheckpointBundle,
) -> IrisExperimentRunV2:
    """Build one profile-v3-bound B/D run without persisting data text or tensors."""
    try:
        checked_arm = _require_arm_id(arm_id)
        if type(terminal_token_id) is not int or type(initialization_seed) is not int:
            message = "terminal_token_id and initialization_seed must be exact integers"
            raise TypeError(message)
        if type(deterministic) is not bool:
            message = "deterministic must be exactly bool"
            raise TypeError(message)
        checkpoint = validate_arm_teacher_checkpoint(initial_checkpoint)
        if checkpoint.identity.arm_id != checked_arm:
            message = "initial checkpoint arm disagrees with requested experiment arm"
            raise ValueError(message)
        owned_manifest = validate_experiment_manifest_v1(manifest)
        configuration = _configuration_from_checkpoint(
            checkpoint,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=execution_policy,
        )
        reference = _checkpoint_reference(checkpoint)
        common: dict[str, Any] = {
            "run_id": run_id,
            "protocol_declaration_payload_sha256": _PROTOCOL_DECLARATION_PAYLOAD_SHA256,
            "execution_profile_set_version": 3,
            "execution_profile_set_sha256": _EXECUTION_PROFILE_SET_SHA256,
            "manifest": ExperimentManifestReferenceV1(
                dataset_id=owned_manifest.payload.dataset_id,
                payload_sha256=owned_manifest.payload_sha256,
                split_counts=owned_manifest.payload.split_counts,
            ),
            "configuration": configuration,
            "initial_checkpoint": reference,
        }
        if checked_arm == "B":
            payload = ExperimentRunPayloadV2(
                arm_id="B",
                arm_kind="llm_modules_no_workspace",
                request_graph="independent_recurrent_modules_zero_memory_read_v1",
                teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
                **common,
            )
        else:
            payload = ExperimentRunPayloadV2(
                arm_id="D",
                arm_kind="llm_modules_limited_competitive_workspace",
                request_graph="limited_competitive_workspace_v1",
                teacher_graph="limited_competitive_workspace_teacher_v1",
                **common,
            )
        run = IrisExperimentRunV2(
            schema_identity=ExperimentRunSchemaV2(
                artifact_type="iris.experiment_run",
                version=2,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                journal_layout="immutable_run_and_chained_events",
            ),
            payload_sha256=domain_sha256(
                _RUN_PAYLOAD_DOMAIN,
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        return _run_bytes(run)[0]
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled experiment run could not be built"
        raise IrisExperimentLogError(message) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        result[key] = value
    return result


def _parse_canonical_json_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        message = f"{label} must be a real regular file"
        raise ValueError(message)
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        message = f"{label} file size is invalid"
        raise ValueError(message)
    content = path.read_bytes()
    if len(content) != size:
        message = f"{label} size changed while reading"
        raise ValueError(message)
    if (
        not content.endswith(b"\n")
        or content.endswith(b"\n\n")
        or content.startswith(b"\xef\xbb\xbf")
    ):
        message = f"{label} must be canonical UTF-8 JSON with one final LF"
        raise ValueError(message)
    raw = json.loads(
        content[:-1].decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON number is forbidden: {value}")
        ),
    )
    if type(raw) is not dict or canonical_json_file_bytes(raw) != content:
        message = f"{label} is not canonical iris-json-v1"
        raise ValueError(message)
    return raw, content


def _event_bytes(event: IrisExperimentEventV2) -> tuple[IrisExperimentEventV2, bytes]:
    if type(event) is not IrisExperimentEventV2:
        message = "experiment event must be exactly IrisExperimentEventV2"
        raise TypeError(message)
    owned = IrisExperimentEventV2.model_validate(event.model_dump(mode="python"))
    expected = domain_sha256(
        _EVENT_PAYLOAD_DOMAIN,
        canonical_json_bytes(owned.payload.model_dump(mode="json")),
    )
    if owned.payload_sha256 != expected:
        message = "arm-labelled experiment event payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_EVENT_BYTES:
        message = "arm-labelled experiment event exceeds the 64 KiB safety limit"
        raise ValueError(message)
    return owned, content


def _validate_event_against_prefix(
    event: IrisExperimentEventV2,
    *,
    run: IrisExperimentRunV2,
    previous: IrisExperimentEventV2 | None,
) -> None:
    payload = event.payload
    expected_sequence = 1 if previous is None else previous.payload.sequence + 1
    expected_predecessor = run.payload_sha256 if previous is None else previous.payload_sha256
    previous_steps = (
        run.payload.initial_checkpoint.completed_steps
        if previous is None
        else previous.payload.optimizer_completed_steps
    )
    previous_state = (
        run.payload.initial_checkpoint.sidecar_state_sha256
        if previous is None
        else previous.payload.sidecar_state_sha256
    )
    if (
        payload.sequence != expected_sequence
        or payload.run_id != run.payload.run_id
        or payload.arm_id != run.payload.arm_id
        or payload.run_payload_sha256 != run.payload_sha256
        or payload.manifest_payload_sha256 != run.payload.manifest.payload_sha256
        or payload.configuration_sha256 != run.payload.configuration.sha256
        or payload.predecessor_sha256 != expected_predecessor
    ):
        message = "experiment event identity, arm, or predecessor chain is inconsistent"
        raise ValueError(message)
    record = payload.record
    if type(record) is ExperimentTrainingStepMetricV1:
        if (
            payload.optimizer_completed_steps != previous_steps + 1
            or record.split_example_count != run.payload.manifest.split_counts.train
        ):
            message = "training event does not describe the next committed run step"
            raise ValueError(message)
    elif (
        payload.optimizer_completed_steps != previous_steps
        or payload.sidecar_state_sha256 != previous_state
    ):
        message = "non-training event cannot change the recorded training state"
        raise ValueError(message)
    if type(record) is ExperimentEvaluationMetricV1:
        expected_count = getattr(run.payload.manifest.split_counts, record.split)
        if expected_count == 0 or record.example_count != expected_count:
            message = "evaluation metric does not cover its complete nonempty split"
            raise ValueError(message)


def _event_filename(sequence: int) -> str:
    if type(sequence) is not int or sequence < 1 or sequence > _MAX_EVENT_COUNT:
        message = "experiment event sequence is outside the supported range"
        raise ValueError(message)
    return f"{sequence:08d}.json"


def _read_run(path: Path) -> IrisExperimentRunV2:
    _, content = _parse_canonical_json_file(
        path,
        maximum_bytes=_MAX_RUN_BYTES,
        label="arm-labelled experiment run",
    )
    run = IrisExperimentRunV2.model_validate_json(content[:-1])
    owned, expected = _run_bytes(run)
    if content != expected:
        message = "experiment run canonical round trip changed its bytes"
        raise ValueError(message)
    return owned


def _read_event(
    path: Path,
    *,
    sequence: int,
    run: IrisExperimentRunV2,
    previous: IrisExperimentEventV2 | None,
) -> IrisExperimentEventV2:
    if path.name != _event_filename(sequence):
        message = "experiment event filenames must form one contiguous prefix"
        raise ValueError(message)
    _, content = _parse_canonical_json_file(
        path,
        maximum_bytes=_MAX_EVENT_BYTES,
        label=f"arm-labelled experiment event {sequence}",
    )
    event = IrisExperimentEventV2.model_validate_json(content[:-1])
    owned, expected = _event_bytes(event)
    if content != expected:
        message = "experiment event canonical round trip changed its bytes"
        raise ValueError(message)
    _validate_event_against_prefix(owned, run=run, previous=previous)
    return owned


def read_experiment_log_v2(path: Path) -> IrisExperimentLogSnapshotV2:
    """Read and fully verify one exact v2 run and its contiguous event prefix."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "experiment log reader requires an explicit run directory Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_dir():
            message = "experiment log path must be a real directory"
            raise ValueError(message)
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in entries) != (_EVENTS_DIRECTORY, _RUN_FILENAME):
            message = "experiment run directory must contain exactly events and run.json"
            raise ValueError(message)
        events_path = path / _EVENTS_DIRECTORY
        if events_path.is_symlink() or not events_path.is_dir():
            message = "experiment events path must be a real directory"
            raise ValueError(message)
        run = _read_run(path / _RUN_FILENAME)
        if path.name != run.payload.run_id:
            message = "experiment run directory name does not match run_id"
            raise ValueError(message)
        event_entries = tuple(sorted(events_path.iterdir(), key=lambda entry: entry.name))
        if len(event_entries) > _MAX_EVENT_COUNT:
            message = "experiment journal exceeds the event safety limit"
            raise ValueError(message)
        events: list[IrisExperimentEventV2] = []
        previous: IrisExperimentEventV2 | None = None
        for sequence, entry in enumerate(event_entries, start=1):
            event = _read_event(
                entry,
                sequence=sequence,
                run=run,
                previous=previous,
            )
            events.append(event)
            previous = event
        final_entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
        final_event_entries = tuple(sorted(events_path.iterdir(), key=lambda entry: entry.name))
        if tuple(entry.name for entry in final_entries) != tuple(
            entry.name for entry in entries
        ) or tuple(entry.name for entry in final_event_entries) != tuple(
            entry.name for entry in event_entries
        ):
            message = "experiment journal changed while reading"
            raise ValueError(message)
        return IrisExperimentLogSnapshotV2(path=path, run=run, events=tuple(events))
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled experiment log could not be read or verified"
        raise IrisExperimentLogError(message) from error


def _write_exact_file(path: Path, content: bytes) -> None:
    file_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "wb", buffering=0) as stream:
            remaining = memoryview(content)
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0:
                    message = "experiment v2 write made no forward progress"
                    raise OSError(message)
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(file_descriptor)
        raise


def _cleanup_directory(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        message = "temporary run path is not a real directory"
        raise OSError(message)
    entries = tuple(path.iterdir())
    if any(child.name not in {_RUN_FILENAME, _EVENTS_DIRECTORY} for child in entries):
        message = "temporary run directory contains an unexpected entry"
        raise OSError(message)
    run_path = path / _RUN_FILENAME
    if run_path.exists():
        if run_path.is_symlink() or not run_path.is_file():
            message = "temporary run file is not a real file"
            raise OSError(message)
        run_path.unlink()
    events_path = path / _EVENTS_DIRECTORY
    if events_path.exists():
        if events_path.is_symlink() or not events_path.is_dir() or tuple(events_path.iterdir()):
            message = "temporary events path is not one empty real directory"
            raise OSError(message)
        events_path.rmdir()
    path.rmdir()


def _existing_run_matches(path: Path, content: bytes) -> bool:
    try:
        snapshot = read_experiment_log_v2(path)
        _, actual = _run_bytes(snapshot.run)
    except IrisExperimentLogError:
        return False
    return actual == content


def _rename_directory_with_permission_retry(source: Path, target: Path) -> None:
    for delay in _WINDOWS_PERMISSION_RETRY_DELAYS:
        try:
            os.rename(source, target)
            return
        except PermissionError:
            sleep(delay)
    os.rename(source, target)


def _caused_by_permission_error(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, PermissionError):
            return True
        current = current.__cause__
    return False


def _write_checkpoint_with_permission_retry(
    checkpoint: IrisArmTeacherCheckpointBundle,
    *,
    root: Path,
) -> IrisArmTeacherCheckpointWriteResult:
    for delay in _WINDOWS_PERMISSION_RETRY_DELAYS:
        try:
            return write_arm_teacher_checkpoint(checkpoint, root=root)
        except IrisTeacherCheckpointError as error:
            if not _caused_by_permission_error(error):
                raise
            sleep(delay)
    return write_arm_teacher_checkpoint(checkpoint, root=root)


def _write_run(run: IrisExperimentRunV2, *, root: Path) -> tuple[Path, bool]:
    if not isinstance(root, Path) or not root.name:
        message = "experiment log writer requires an explicit root Path"
        raise TypeError(message)
    owned, content = _run_bytes(run)
    parent = root / "experiments" / "v2"
    target = parent / owned.payload.run_id
    temporary: Path | None = None
    try:
        if target.exists():
            if not _existing_run_matches(target, content):
                message = "existing experiment v2 run conflicts with expected bytes"
                raise IrisExperimentLogError(message)
            return target, True
        parent.mkdir(parents=True, exist_ok=True)
        staging_candidate = parent / f".iris-{uuid4().hex}.tmp"
        os.mkdir(staging_candidate, mode=0o700)
        temporary = staging_candidate
        os.mkdir(temporary / _EVENTS_DIRECTORY, mode=0o700)
        _write_exact_file(temporary / _RUN_FILENAME, content)
        if (temporary / _RUN_FILENAME).read_bytes() != content:
            message = "temporary experiment run bytes failed verification"
            raise OSError(message)
        try:
            _rename_directory_with_permission_retry(temporary, target)
        except OSError as error:
            if not target.exists() or not _existing_run_matches(target, content):
                message = "concurrent experiment v2 run installation conflicted"
                raise IrisExperimentLogError(message) from error
            _cleanup_directory(temporary)
            temporary = None
            return target, True
        temporary = None
        return target, False
    except BaseException as primary_error:
        if temporary is not None and temporary.exists():
            try:
                _cleanup_directory(temporary)
            except BaseException as cleanup_error:
                failures = BaseExceptionGroup(
                    "experiment run persistence and cleanup both failed",
                    [primary_error, cleanup_error],
                )
                raise IrisExperimentLogError(
                    "arm-labelled experiment run cleanup was incomplete"
                ) from failures
        if isinstance(primary_error, IrisExperimentLogError):
            raise
        if isinstance(primary_error, Exception):
            raise IrisExperimentLogError(
                "arm-labelled experiment run could not be installed atomically"
            ) from primary_error
        raise


def create_experiment_log_v2(
    *,
    run_id: str,
    arm_id: IrisTeacherArmId,
    manifest: IrisExperimentManifestV1,
    terminal_token_id: int,
    initialization_seed: int,
    deterministic: bool,
    execution_policy: ExperimentExecutionPolicyV1,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    root: Path,
) -> IrisExperimentLogCreateResultV2:
    """Persist the exact B/D initial checkpoint and atomically create a v2 journal."""
    try:
        checked_arm = _require_arm_id(arm_id)
        if not isinstance(root, Path) or not root.name:
            message = "experiment log creation requires an explicit root Path"
            raise TypeError(message)
        runtime_policy = ExperimentRuntimePolicyV1(
            initialization_seed=initialization_seed,
            deterministic=deterministic,
        )
        _validate_ambient_runtime_policy(runtime_policy)
        checkpoint = build_arm_teacher_checkpoint(
            arm_id=checked_arm,
            model=model,
            optimizer=optimizer,
        )
        run = build_experiment_run_v2(
            run_id=run_id,
            arm_id=checked_arm,
            manifest=manifest,
            terminal_token_id=terminal_token_id,
            initialization_seed=initialization_seed,
            deterministic=deterministic,
            execution_policy=execution_policy,
            initial_checkpoint=checkpoint,
        )
        checkpoint_result = _write_checkpoint_with_permission_retry(checkpoint, root=root)
        path, reused = _write_run(run, root=root)
        snapshot = read_experiment_log_v2(path)
        if snapshot.run != run:
            message = "installed v2 run does not match requested identity"
            raise ValueError(message)
        return IrisExperimentLogCreateResultV2(
            arm_id=checked_arm,
            path=path,
            run_payload_sha256=run.payload_sha256,
            configuration_sha256=run.payload.configuration.sha256,
            initial_checkpoint_path=checkpoint_result.path,
            initial_checkpoint_payload_sha256=checkpoint_result.identity.payload_sha256,
            reused_existing=reused,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled experiment log could not be created"
        raise IrisExperimentLogError(message) from error


def _optimizer_configuration_from_provenance(
    provenance: IrisTeacherOptimizerProvenance,
) -> ExperimentOptimizerConfigurationV1:
    if type(provenance) is not IrisTeacherOptimizerProvenance:
        message = "teacher optimizer returned invalid v2 log provenance"
        raise TypeError(message)
    config = provenance.config
    return ExperimentOptimizerConfigurationV1(
        optimizer_type="torch.optim.AdamW",
        config=TeacherAdamWConfigV1(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
            max_gradient_norm=config.max_gradient_norm,
        ),
        parameter_names=provenance.parameter_names,
    )


def _arm_b_model_from_provenance(
    provenance: IrisNoWorkspaceTeacherCheckpointProvenanceV1,
    *,
    state_names: tuple[str, ...],
) -> ExperimentArmBModelConfigurationV2:
    if type(provenance) is not IrisNoWorkspaceTeacherCheckpointProvenanceV1:
        message = "arm-B teacher returned invalid v2 log provenance"
        raise TypeError(message)
    return ExperimentArmBModelConfigurationV2(
        arm_id="B",
        arm_kind="llm_modules_no_workspace",
        teacher_graph=provenance.teacher_graph,
        backbone=BackboneConfig.model_validate_json(provenance.backbone_configuration_json),
        cognitive=CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json),
        cognitive_cycles=3,
        latent_token_count=provenance.latent_token_count,
        component_names=provenance.component_names,
        runtime=_runtime_from_values(
            hidden_size=provenance.hidden_size,
            num_hidden_layers=provenance.num_hidden_layers,
            vocabulary_size=provenance.vocabulary_size,
            max_position_embeddings=provenance.max_position_embeddings,
            max_prompt_tokens=provenance.max_prompt_tokens,
            max_target_tokens=provenance.max_target_tokens,
            device=provenance.device,
            dtype=provenance.dtype,
        ),
        state_names=state_names,
    )


def _arm_d_model_from_provenance(
    provenance: IrisTeacherModelCheckpointProvenance,
    *,
    state_names: tuple[str, ...],
) -> ExperimentArmDModelConfigurationV2:
    if type(provenance) is not IrisTeacherModelCheckpointProvenance:
        message = "arm-D teacher returned invalid v2 log provenance"
        raise TypeError(message)
    return ExperimentArmDModelConfigurationV2(
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        teacher_graph="limited_competitive_workspace_teacher_v1",
        backbone=BackboneConfig.model_validate_json(provenance.backbone_configuration_json),
        cognitive=CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json),
        workspace=WorkspaceConfig.model_validate_json(provenance.workspace_configuration_json),
        component_names=_ARM_D_COMPONENT_NAMES,
        runtime=_runtime_from_values(
            hidden_size=provenance.hidden_size,
            num_hidden_layers=provenance.num_hidden_layers,
            vocabulary_size=provenance.vocabulary_size,
            max_position_embeddings=provenance.max_position_embeddings,
            max_prompt_tokens=provenance.max_prompt_tokens,
            max_target_tokens=provenance.max_target_tokens,
            device=provenance.device,
            dtype=provenance.dtype,
        ),
        state_names=state_names,
    )


def _require_owner(
    arm_id: ExperimentArmIdV2,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> None:
    expected_model = (
        IrisNoWorkspaceTeacherForcingModel if arm_id == "B" else IrisTeacherForcingModel
    )
    if type(model) is not expected_model:
        message = f"arm {arm_id} experiment requires exact model type {expected_model.__name__}"
        raise TypeError(message)
    if type(optimizer) is not IrisTeacherOptimizer or optimizer.model is not model:
        message = "experiment requires the exact optimizer owner for its model"
        raise TypeError(message)


def _live_training_identity(
    *,
    run: IrisExperimentRunV2,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> _LiveTrainingIdentity:
    arm_id = run.payload.arm_id
    _require_owner(arm_id, model, optimizer)
    _validate_ambient_runtime_policy(run.payload.configuration.runtime_policy)
    state_names = tuple(model.state_dict())
    if arm_id == "B":
        arm_b_model = cast(IrisNoWorkspaceTeacherForcingModel, model)
        before = arm_b_model.checkpoint_provenance
        optimizer_provenance = optimizer.provenance_snapshot()
        after = arm_b_model.checkpoint_provenance
        if before != after:
            message = "arm-B teacher state changed during v2 log observation"
            raise ValueError(message)
        model_configuration: ExperimentModelConfigurationV2 = _arm_b_model_from_provenance(
            after,
            state_names=state_names,
        )
        sidecar_state_sha256 = after.sidecar_state_sha256
    else:
        arm_d_model = cast(IrisTeacherForcingModel, model)
        before_d = arm_d_model.checkpoint_provenance
        optimizer_provenance = optimizer.provenance_snapshot()
        after_d = arm_d_model.checkpoint_provenance
        if before_d != after_d:
            message = "arm-D teacher state changed during v2 log observation"
            raise ValueError(message)
        model_configuration = _arm_d_model_from_provenance(
            after_d,
            state_names=state_names,
        )
        sidecar_state_sha256 = after_d.sidecar_state_sha256
    if optimizer.completed_steps != optimizer_provenance.completed_steps:
        message = "optimizer counter changed during v2 log observation"
        raise ValueError(message)
    configuration = _configuration(
        model=model_configuration,
        optimizer=_optimizer_configuration_from_provenance(optimizer_provenance),
        terminal_token_id=run.payload.configuration.target_preparation.terminal_token_id,
        initialization_seed=run.payload.configuration.runtime_policy.initialization_seed,
        deterministic=run.payload.configuration.runtime_policy.deterministic,
        execution_policy=run.payload.configuration.execution_policy,
    )
    if configuration != run.payload.configuration:
        message = "live teacher configuration does not match the arm-labelled run"
        raise ValueError(message)
    return _LiveTrainingIdentity(
        configuration=configuration,
        optimizer_completed_steps=optimizer_provenance.completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
    )


def inspect_experiment_training_snapshot_v2(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisExperimentTrainingStateV2:
    """Identify one exact live B/D owner against a verified v2 snapshot."""
    try:
        if type(snapshot) is not IrisExperimentLogSnapshotV2:
            message = "training inspection requires exactly IrisExperimentLogSnapshotV2"
            raise TypeError(message)
        live = _live_training_identity(run=snapshot.run, model=model, optimizer=optimizer)
        return IrisExperimentTrainingStateV2(
            arm_id=snapshot.arm_id,
            configuration_sha256=live.configuration.sha256,
            chain_head_sha256=snapshot.chain_head_sha256,
            event_count=snapshot.event_count,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled experiment training state could not be inspected"
        raise IrisExperimentLogError(message) from error


def inspect_experiment_training_state_v2(
    *,
    path: Path,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
) -> IrisExperimentTrainingStateV2:
    return inspect_experiment_training_snapshot_v2(
        snapshot=read_experiment_log_v2(path),
        model=model,
        optimizer=optimizer,
    )


def _manifest_for_run(
    manifest: IrisExperimentManifestV1,
    run: IrisExperimentRunV2,
) -> IrisExperimentManifestV1:
    owned = validate_experiment_manifest_v1(manifest)
    reference = run.payload.manifest
    if (
        owned.payload.dataset_id != reference.dataset_id
        or owned.payload_sha256 != reference.payload_sha256
        or owned.payload.split_counts != reference.split_counts
    ):
        message = "experiment manifest does not match the run identity"
        raise ValueError(message)
    return owned


def _require_expected_chain_head(
    snapshot: IrisExperimentLogSnapshotV2,
    expected_predecessor_sha256: str | None,
) -> None:
    if expected_predecessor_sha256 is None:
        return
    if (
        type(expected_predecessor_sha256) is not str
        or fullmatch(r"[0-9a-f]{64}", expected_predecessor_sha256) is None
    ):
        message = "expected_predecessor_sha256 must be one canonical SHA-256"
        raise TypeError(message)
    if snapshot.chain_head_sha256 != expected_predecessor_sha256:
        message = "experiment chain head changed before the requested append"
        raise ValueError(message)


def _validate_managed_candidate(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    manifest: IrisExperimentManifestV1 | None,
    record: ExperimentEventRecordV2,
    epoch_progress: IrisExperimentEpochProgressV2 | None,
) -> None:
    policy = snapshot.run.payload.configuration.execution_policy
    if type(policy) is ExperimentManualExecutionPolicyV1:
        if epoch_progress is not None:
            message = "manual experiment append cannot use managed epoch progress"
            raise TypeError(message)
        return
    if type(policy) is not ExperimentEpochExecutionPolicyV1:
        message = "experiment run contains an unsupported execution policy"
        raise TypeError(message)
    if manifest is None:
        message = "managed experiment append requires its exact manifest"
        raise TypeError(message)
    if epoch_progress is None:
        cursor = analyze_experiment_epoch_prefix_v2(
            run=snapshot.run,
            events=snapshot.events,
            manifest=manifest,
        )
    else:
        current = extend_experiment_epoch_progress_v2(
            progress=epoch_progress,
            run=snapshot.run,
            events=snapshot.events,
        )
        if (
            current.verified_event_count != snapshot.event_count
            or current.chain_head_sha256 != snapshot.chain_head_sha256
        ):
            message = "managed epoch progress does not cover the experiment snapshot"
            raise ValueError(message)
        cursor = current.cursor
    validate_experiment_epoch_candidate_v2(cursor=cursor, record=record)


def _prepared_token_count(
    prepared: IrisPreparedExperimentExample,
    *,
    configuration: ExperimentTrainingConfigurationV2,
) -> int:
    teacher_input = prepared.teacher_input
    if type(teacher_input) is not IrisTeacherForcingInput:
        message = "prepared experiment example contains an invalid teacher input"
        raise TypeError(message)
    input_ids = teacher_input.target_input_ids
    mask = teacher_input.target_attention_mask
    runtime = configuration.model.runtime
    terminal_token_id = configuration.target_preparation.terminal_token_id
    if (
        type(input_ids) is not torch.Tensor
        or type(mask) is not torch.Tensor
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != mask.shape
        or input_ids.dtype != torch.int64
        or mask.dtype != torch.bool
        or str(input_ids.device) != runtime.device
        or mask.device != input_ids.device
        or input_ids.shape[1] < 2
        or input_ids.shape[1] > runtime.max_target_tokens
        or not bool(torch.all(mask == 1))
        or not bool(torch.all((input_ids >= 0) & (input_ids < runtime.vocabulary_size)))
        or int(input_ids[0, -1].item()) != terminal_token_id
    ):
        message = "prepared teacher target disagrees with run tokenization policy"
        raise ValueError(message)
    return int(mask.sum().item())


def _new_event(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    record: ExperimentEventRecordV2,
) -> IrisExperimentEventV2:
    payload = ExperimentEventPayloadV2(
        sequence=snapshot.event_count + 1,
        run_id=snapshot.run.payload.run_id,
        arm_id=snapshot.arm_id,
        run_payload_sha256=snapshot.run.payload_sha256,
        manifest_payload_sha256=snapshot.run.payload.manifest.payload_sha256,
        configuration_sha256=snapshot.run.payload.configuration.sha256,
        predecessor_sha256=snapshot.chain_head_sha256,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
        record=record,
    )
    event = IrisExperimentEventV2(
        schema_identity=ExperimentEventSchemaV2(
            artifact_type="iris.experiment_event",
            version=2,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=domain_sha256(
            _EVENT_PAYLOAD_DOMAIN,
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    owned, _ = _event_bytes(event)
    previous = snapshot.events[-1] if snapshot.events else None
    _validate_event_against_prefix(owned, run=snapshot.run, previous=previous)
    return owned


def _event_result(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    event: IrisExperimentEventV2,
    reused: bool,
    checkpoint_path: Path | None,
) -> IrisExperimentEventWriteResultV2:
    return IrisExperimentEventWriteResultV2(
        arm_id=snapshot.arm_id,
        path=snapshot.path / _EVENTS_DIRECTORY / _event_filename(event.payload.sequence),
        sequence=event.payload.sequence,
        event_type=event.payload.record.event_type,
        payload_sha256=event.payload_sha256,
        reused_existing=reused,
        referenced_checkpoint_path=checkpoint_path,
    )


def _managed_head_duplicate(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    record: ExperimentEventRecordV2,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    expected_predecessor_sha256: str | None,
) -> IrisExperimentEventV2 | None:
    if (
        type(snapshot.run.payload.configuration.execution_policy)
        is ExperimentManualExecutionPolicyV1
        or not snapshot.events
    ):
        return None
    head = snapshot.events[-1]
    payload = head.payload
    if (
        payload.record != record
        or payload.optimizer_completed_steps != optimizer_completed_steps
        or payload.sidecar_state_sha256 != sidecar_state_sha256
    ):
        return None
    if expected_predecessor_sha256 is not None:
        if (
            type(expected_predecessor_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", expected_predecessor_sha256) is None
        ):
            message = "expected_predecessor_sha256 must be one canonical SHA-256"
            raise TypeError(message)
        if payload.predecessor_sha256 != expected_predecessor_sha256:
            return None
    return head


def _find_duplicate(
    snapshot: IrisExperimentLogSnapshotV2,
    *,
    record: ExperimentEventRecordV2,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
) -> IrisExperimentEventV2 | None:
    for event in reversed(snapshot.events):
        payload = event.payload
        if payload.optimizer_completed_steps < optimizer_completed_steps:
            break
        if (
            payload.optimizer_completed_steps == optimizer_completed_steps
            and payload.sidecar_state_sha256 == sidecar_state_sha256
            and payload.record == record
        ):
            return event
    return None


def _install_event(
    snapshot: IrisExperimentLogSnapshotV2,
    event: IrisExperimentEventV2,
) -> IrisExperimentEventWriteResultV2:
    owned, content = _event_bytes(event)
    events_path = snapshot.path / _EVENTS_DIRECTORY
    target = events_path / _event_filename(owned.payload.sequence)
    temporary = snapshot.path.parent / f".iris-event-{uuid4().hex}.tmp"
    try:
        _write_exact_file(temporary, content)
        if temporary.read_bytes() != content:
            message = "temporary experiment event bytes failed verification"
            raise OSError(message)
        try:
            os.link(temporary, target)
            reused = False
        except FileExistsError as error:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                message = "concurrent experiment event conflicts with expected next event"
                raise IrisExperimentLogError(message) from error
            reused = True
        temporary.unlink()
        installed = read_experiment_log_v2(snapshot.path)
        if (
            installed.event_count < owned.payload.sequence
            or installed.events[owned.payload.sequence - 1] != owned
        ):
            message = "installed experiment event does not match expected chain entry"
            raise ValueError(message)
        return _event_result(
            snapshot=installed,
            event=owned,
            reused=reused,
            checkpoint_path=None,
        )
    except BaseException as primary_error:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "experiment event persistence and cleanup both failed",
                [primary_error, cleanup_error],
            )
            raise IrisExperimentLogError(
                "arm-labelled experiment event cleanup was incomplete"
            ) from failures
        if isinstance(primary_error, IrisExperimentLogError):
            raise
        if isinstance(primary_error, Exception):
            raise IrisExperimentLogError(
                "arm-labelled experiment event could not be installed atomically"
            ) from primary_error
        raise


def _record_or_reuse(
    *,
    snapshot: IrisExperimentLogSnapshotV2,
    record: ExperimentEventRecordV2,
    optimizer_completed_steps: int,
    sidecar_state_sha256: str,
    expected_predecessor_sha256: str | None,
    checkpoint_path: Path | None = None,
) -> IrisExperimentEventWriteResultV2:
    duplicate = _find_duplicate(
        snapshot,
        record=record,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
    )
    if duplicate is not None and (
        expected_predecessor_sha256 is None
        or duplicate.payload.predecessor_sha256 == expected_predecessor_sha256
    ):
        return _event_result(
            snapshot=snapshot,
            event=duplicate,
            reused=True,
            checkpoint_path=checkpoint_path,
        )
    _require_expected_chain_head(snapshot, expected_predecessor_sha256)
    event = _new_event(
        snapshot=snapshot,
        optimizer_completed_steps=optimizer_completed_steps,
        sidecar_state_sha256=sidecar_state_sha256,
        record=record,
    )
    result = _install_event(snapshot, event)
    if checkpoint_path is None:
        return result
    return IrisExperimentEventWriteResultV2(
        arm_id=result.arm_id,
        path=result.path,
        sequence=result.sequence,
        event_type=result.event_type,
        payload_sha256=result.payload_sha256,
        reused_existing=result.reused_existing,
        referenced_checkpoint_path=checkpoint_path,
    )


def record_experiment_training_step_v2(
    *,
    path: Path,
    manifest: IrisExperimentManifestV1,
    prepared: IrisPreparedExperimentExample,
    output: IrisTeacherOptimizationOutput,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    expected_predecessor_sha256: str | None = None,
    epoch_progress: IrisExperimentEpochProgressV2 | None = None,
) -> IrisExperimentEventWriteResultV2:
    """Append one committed train-step metric tied to exact arm/example/live state."""
    try:
        snapshot = read_experiment_log_v2(path)
        owned_manifest = _manifest_for_run(manifest, snapshot.run)
        if type(prepared) is not IrisPreparedExperimentExample:
            message = "training log requires exactly IrisPreparedExperimentExample"
            raise TypeError(message)
        if type(output) is not IrisTeacherOptimizationOutput:
            message = "training log requires exactly IrisTeacherOptimizationOutput"
            raise TypeError(message)
        train_examples = tuple(
            example for example in owned_manifest.payload.examples if example.split == "train"
        )
        if (
            prepared.dataset_id != owned_manifest.payload.dataset_id
            or prepared.manifest_payload_sha256 != owned_manifest.payload_sha256
            or prepared.split != "train"
            or prepared.split_example_count != len(train_examples)
            or prepared.split_ordinal < 1
            or prepared.split_ordinal > len(train_examples)
        ):
            message = "prepared training example does not match the run manifest"
            raise ValueError(message)
        source = train_examples[prepared.split_ordinal - 1]
        if (
            source.example_id != prepared.example_id
            or source.prompt != prepared.teacher_input.prompt
        ):
            message = "prepared training example provenance is inconsistent"
            raise ValueError(message)
        token_count = _prepared_token_count(
            prepared,
            configuration=snapshot.run.payload.configuration,
        )
        live = _live_training_identity(run=snapshot.run, model=model, optimizer=optimizer)
        if (
            output.step_index != live.optimizer_completed_steps
            or output.supervised_token_count != token_count
        ):
            message = "optimization output does not describe current committed state"
            raise ValueError(message)
        record = ExperimentTrainingStepMetricV1(
            event_type="training_step",
            split="train",
            example_id=prepared.example_id,
            split_ordinal=prepared.split_ordinal,
            split_example_count=prepared.split_example_count,
            step_index=output.step_index,
            loss=output.loss,
            supervised_token_count=output.supervised_token_count,
            gradient_l2_norm=output.gradient_l2_norm,
            gradients_clipped=output.gradients_clipped,
        )
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused=True,
                checkpoint_path=None,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=owned_manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled training-step metric could not be recorded"
        raise IrisExperimentLogError(message) from error


def validate_experiment_checkpoint_for_run_v2(
    *,
    checkpoint: IrisArmTeacherCheckpointBundle,
    run: IrisExperimentRunV2,
) -> IrisArmTeacherCheckpointBundle:
    """Verify one B/D checkpoint and its complete immutable run configuration."""
    try:
        if type(run) is not IrisExperimentRunV2:
            message = "checkpoint/run validation requires exactly IrisExperimentRunV2"
            raise TypeError(message)
        verified = validate_arm_teacher_checkpoint(checkpoint)
        if verified.identity.arm_id != run.payload.arm_id:
            message = "checkpoint arm does not match the experiment run"
            raise ValueError(message)
        configuration = _configuration_from_checkpoint(
            verified,
            terminal_token_id=run.payload.configuration.target_preparation.terminal_token_id,
            initialization_seed=run.payload.configuration.runtime_policy.initialization_seed,
            deterministic=run.payload.configuration.runtime_policy.deterministic,
            execution_policy=run.payload.configuration.execution_policy,
        )
        if configuration != run.payload.configuration:
            message = "checkpoint configuration does not match the experiment run"
            raise ValueError(message)
        return verified
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled checkpoint/run validation failed"
        raise IrisExperimentLogError(message) from error


def record_experiment_evaluation_v2(
    *,
    path: Path,
    manifest: IrisExperimentManifestV1,
    split: ExperimentSplitV1,
    summary: IrisTokenWeightedEvaluationSummary,
    model: IrisArmTeacherModel,
    optimizer: IrisTeacherOptimizer,
    expected_predecessor_sha256: str | None = None,
    epoch_progress: IrisExperimentEpochProgressV2 | None = None,
) -> IrisExperimentEventWriteResultV2:
    """Append one complete validation/test summary at unchanged B/D state."""
    try:
        snapshot = read_experiment_log_v2(path)
        owned_manifest = _manifest_for_run(manifest, snapshot.run)
        if type(split) is not str or split not in {"validation", "test"}:
            message = "evaluation split must be exactly validation or test"
            raise ValueError(message)
        if type(summary) is not IrisTokenWeightedEvaluationSummary:
            message = "evaluation log requires exact token-weighted summary"
            raise TypeError(message)
        expected_count = getattr(owned_manifest.payload.split_counts, split)
        if expected_count == 0 or summary.example_count != expected_count:
            message = "evaluation summary must cover one complete nonempty split"
            raise ValueError(message)
        live = _live_training_identity(run=snapshot.run, model=model, optimizer=optimizer)
        record = ExperimentEvaluationMetricV1(
            event_type="evaluation",
            split=split,
            example_count=summary.example_count,
            supervised_token_count=summary.supervised_token_count,
            mean_loss=summary.mean_loss,
            perplexity=summary.perplexity,
        )
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused=True,
                checkpoint_path=None,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=owned_manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=live.optimizer_completed_steps,
            sidecar_state_sha256=live.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled evaluation metric could not be recorded"
        raise IrisExperimentLogError(message) from error


def _root_from_run_path(path: Path) -> Path:
    if (
        not isinstance(path, Path)
        or path.parent.name != "v2"
        or path.parent.parent.name != "experiments"
        or not path.parent.parent.parent.name
    ):
        message = "experiment run path is outside canonical experiments/v2 layout"
        raise ValueError(message)
    return path.parent.parent.parent


def record_experiment_checkpoint_v2(
    *,
    path: Path,
    checkpoint: IrisArmTeacherCheckpointBundle,
    manifest: IrisExperimentManifestV1 | None = None,
    expected_predecessor_sha256: str | None = None,
    epoch_progress: IrisExperimentEpochProgressV2 | None = None,
) -> IrisExperimentEventWriteResultV2:
    """Persist and append one exact B/v2 or D/v1 checkpoint reference."""
    try:
        snapshot = read_experiment_log_v2(path)
        try:
            verified = validate_experiment_checkpoint_for_run_v2(
                checkpoint=checkpoint,
                run=snapshot.run,
            )
        except IrisExperimentLogError as error:
            message = "checkpoint is incompatible with the experiment run"
            raise ValueError(message) from error
        reference = _checkpoint_reference(verified)
        if (
            reference.completed_steps != snapshot.optimizer_completed_steps
            or reference.sidecar_state_sha256 != snapshot.sidecar_state_sha256
        ):
            message = "checkpoint does not describe current experiment chain state"
            raise ValueError(message)
        record = ExperimentCheckpointEventV2(event_type="checkpoint", checkpoint=reference)
        duplicate = _managed_head_duplicate(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=reference.completed_steps,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
        if duplicate is not None:
            checkpoint_result = _write_checkpoint_with_permission_retry(
                verified,
                root=_root_from_run_path(snapshot.path),
            )
            return _event_result(
                snapshot=snapshot,
                event=duplicate,
                reused=True,
                checkpoint_path=checkpoint_result.path,
            )
        _require_expected_chain_head(snapshot, expected_predecessor_sha256)
        _validate_managed_candidate(
            snapshot=snapshot,
            manifest=manifest,
            record=record,
            epoch_progress=epoch_progress,
        )
        root = _root_from_run_path(snapshot.path)
        checkpoint_result = _write_checkpoint_with_permission_retry(verified, root=root)
        return _record_or_reuse(
            snapshot=snapshot,
            record=record,
            optimizer_completed_steps=reference.completed_steps,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
            checkpoint_path=checkpoint_result.path,
        )
    except IrisExperimentLogError:
        raise
    except Exception as error:
        message = "arm-labelled checkpoint event could not be recorded"
        raise IrisExperimentLogError(message) from error


def checkpoint_identity_from_reference_v2(
    reference: ExperimentCheckpointReferenceV2,
) -> IrisArmTeacherCheckpointIdentity:
    """Translate one persisted v2 reference into the exact operational read identity."""
    if type(reference) is ExperimentArmBCheckpointReferenceV2:
        return IrisArmTeacherCheckpointIdentity(
            arm_id="B",
            checkpoint_schema_version=2,
            payload_sha256=reference.payload_sha256,
            state_file_sha256=reference.state_file_sha256,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            completed_steps=reference.completed_steps,
        )
    if type(reference) is ExperimentArmDCheckpointReferenceV2:
        return IrisArmTeacherCheckpointIdentity(
            arm_id="D",
            checkpoint_schema_version=1,
            payload_sha256=reference.payload_sha256,
            state_file_sha256=reference.state_file_sha256,
            sidecar_state_sha256=reference.sidecar_state_sha256,
            completed_steps=reference.completed_steps,
        )
    message = "checkpoint reference must be an exact arm-labelled v2 variant"
    raise TypeError(message)


__all__ = [
    "IrisExperimentEventWriteResultV2",
    "IrisExperimentLogCreateResultV2",
    "IrisExperimentLogSnapshotV2",
    "IrisExperimentTrainingStateV2",
    "build_experiment_run_v2",
    "checkpoint_identity_from_reference_v2",
    "create_experiment_log_v2",
    "inspect_experiment_training_snapshot_v2",
    "inspect_experiment_training_state_v2",
    "read_experiment_log_v2",
    "record_experiment_checkpoint_v2",
    "record_experiment_evaluation_v2",
    "record_experiment_training_step_v2",
    "validate_experiment_checkpoint_for_run_v2",
]
