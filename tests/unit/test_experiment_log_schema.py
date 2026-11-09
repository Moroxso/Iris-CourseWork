"""Strict persisted-schema tests for experiment run and event records."""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from iris.artifacts.checkpoint_v1 import TeacherAdamWConfigV1, TeacherBackboneRuntimeV1
from iris.artifacts.experiment_log_v1 import (
    ExperimentCheckpointEventV1,
    ExperimentCheckpointReferenceV1,
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
    ExperimentEventPayloadV1,
    ExperimentEventSchemaV1,
    ExperimentManifestReferenceV1,
    ExperimentManualExecutionPolicyV1,
    ExperimentModelConfigurationV1,
    ExperimentOptimizerConfigurationV1,
    ExperimentRunPayloadV1,
    ExperimentRunSchemaV1,
    ExperimentRuntimePolicyV1,
    ExperimentTargetPreparationPolicyV1,
    ExperimentTrainingConfigurationV1,
    ExperimentTrainingStepMetricV1,
    IrisExperimentEventV1,
    IrisExperimentRunV1,
)
from iris.artifacts.experiment_v1 import ExperimentSplitCountsV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _backbone() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "tiny",
            "repository": "offline/tiny",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": "fp32",
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {"max_prompt_tokens": 5, "max_new_tokens": 3},
        }
    )


def _cognitive() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _workspace() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-workspace",
            "num_slots": 2,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.5,
            "empty_threshold": 1e-4,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 1.0,
                "relevance_weight": 1.0,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def _model_configuration() -> ExperimentModelConfigurationV1:
    return ExperimentModelConfigurationV1(
        backbone=_backbone(),
        cognitive=_cognitive(),
        workspace=_workspace(),
        runtime=TeacherBackboneRuntimeV1(
            hidden_size=6,
            num_hidden_layers=4,
            vocabulary_size=16,
            max_position_embeddings=10,
            max_prompt_tokens=5,
            max_target_tokens=3,
            device="cpu",
            dtype="float32",
        ),
        state_names=("layer.weight",),
    )


def _optimizer_configuration() -> ExperimentOptimizerConfigurationV1:
    return ExperimentOptimizerConfigurationV1(
        optimizer_type="torch.optim.AdamW",
        config=TeacherAdamWConfigV1(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        ),
        parameter_names=("layer.weight",),
    )


def _configuration() -> ExperimentTrainingConfigurationV1:
    return ExperimentTrainingConfigurationV1(
        sha256=SHA_C,
        model=_model_configuration(),
        optimizer=_optimizer_configuration(),
        target_preparation=ExperimentTargetPreparationPolicyV1(
            policy="raw_target_no_special_tokens_no_truncation_plus_one_eos",
            terminal_token_id=15,
        ),
        runtime_policy=ExperimentRuntimePolicyV1(
            initialization_seed=1907,
            deterministic=True,
        ),
        execution_policy=ExperimentManualExecutionPolicyV1(
            mode="manual_event_chain",
        ),
    )


def _epoch_policy(
    *,
    epoch_count: int = 1,
    evaluate_test: bool = False,
) -> ExperimentEpochExecutionPolicyV1:
    return ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=epoch_count,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=evaluate_test,
        resume_policy="latest_checkpoint_plus_verified_replay",
    )


def _checkpoint(*, completed_steps: int = 0) -> ExperimentCheckpointReferenceV1:
    return ExperimentCheckpointReferenceV1(
        payload_sha256=SHA_A,
        state_file_sha256=SHA_B,
        sidecar_state_sha256=SHA_C,
        completed_steps=completed_steps,
    )


def _run() -> IrisExperimentRunV1:
    return IrisExperimentRunV1(
        schema_identity=ExperimentRunSchemaV1(
            artifact_type="iris.experiment_run",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            journal_layout="immutable_run_and_chained_events",
        ),
        payload_sha256=SHA_A,
        payload=ExperimentRunPayloadV1(
            run_id="offline.run-1",
            manifest=ExperimentManifestReferenceV1(
                dataset_id="offline-data",
                payload_sha256=SHA_B,
                split_counts=ExperimentSplitCountsV1(train=2, validation=1, test=0),
            ),
            configuration=_configuration(),
            initial_checkpoint=_checkpoint(),
        ),
    )


def _training_record() -> ExperimentTrainingStepMetricV1:
    return ExperimentTrainingStepMetricV1(
        event_type="training_step",
        split="train",
        example_id="train-1",
        split_ordinal=1,
        split_example_count=2,
        step_index=1,
        loss=2.0,
        supervised_token_count=3,
        gradient_l2_norm=0.5,
        gradients_clipped=False,
    )


def _event(
    *,
    record: ExperimentTrainingStepMetricV1
    | ExperimentEvaluationMetricV1
    | ExperimentCheckpointEventV1
    | None = None,
    optimizer_completed_steps: int = 1,
    sidecar_state_sha256: str = SHA_B,
) -> IrisExperimentEventV1:
    return IrisExperimentEventV1(
        schema_identity=ExperimentEventSchemaV1(
            artifact_type="iris.experiment_event",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=SHA_A,
        payload=ExperimentEventPayloadV1(
            sequence=1,
            run_id="offline.run-1",
            run_payload_sha256=SHA_A,
            manifest_payload_sha256=SHA_B,
            configuration_sha256=SHA_C,
            predecessor_sha256=SHA_A,
            optimizer_completed_steps=optimizer_completed_steps,
            sidecar_state_sha256=sidecar_state_sha256,
            record=record or _training_record(),
        ),
    )


def test_complete_run_and_event_models_are_strict_frozen_and_round_trip() -> None:
    run = _run()
    event = _event()

    assert IrisExperimentRunV1.model_validate_json(run.model_dump_json()) == run
    assert IrisExperimentEventV1.model_validate_json(event.model_dump_json()) == event
    assert event.payload.record.event_type == "training_step"
    with pytest.raises(ValidationError, match="frozen_instance"):
        run.payload.run_id = "changed"
    with pytest.raises(ValidationError, match="frozen_instance"):
        event.payload.sequence = 2


@pytest.mark.parametrize("schema", ["run", "event"])
@pytest.mark.parametrize("version", [True, "1", 1.0, 2])
def test_schema_versions_reject_nonexact_or_unsupported_values(
    schema: str,
    version: object,
) -> None:
    if schema == "run":
        values = _run().schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentRunSchemaV1.model_validate(values)
    else:
        values = _event().schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentEventSchemaV1.model_validate(values)


@pytest.mark.parametrize(
    "value",
    ["", ".bad", "bad/name", "a" * 129, 1, True],
)
def test_run_and_dataset_identities_are_portable(value: object) -> None:
    run_values = _run().payload.model_dump(mode="python")
    run_values["run_id"] = value
    with pytest.raises(ValidationError, match="run_id"):
        ExperimentRunPayloadV1.model_validate(run_values)

    manifest_values = _run().payload.manifest.model_dump(mode="python")
    manifest_values["dataset_id"] = value
    with pytest.raises(ValidationError, match="dataset_id"):
        ExperimentManifestReferenceV1.model_validate(manifest_values)


@pytest.mark.parametrize("value", ["bad:name", "CON", "con.json", "trail."])
def test_run_identity_is_a_windows_safe_path_component(value: str) -> None:
    values = _run().payload.model_dump(mode="python")
    values["run_id"] = value

    with pytest.raises(ValidationError, match="filesystem"):
        ExperimentRunPayloadV1.model_validate(values)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 1, True])
def test_every_hash_reference_rejects_noncanonical_digest(digest: object) -> None:
    values = _checkpoint().model_dump(mode="python")
    values["payload_sha256"] = digest
    with pytest.raises(ValidationError, match="SHA-256"):
        ExperimentCheckpointReferenceV1.model_validate(values)

    event_values = _event().payload.model_dump(mode="python")
    event_values["predecessor_sha256"] = digest
    with pytest.raises(ValidationError, match="SHA-256"):
        ExperimentEventPayloadV1.model_validate(event_values)


@pytest.mark.parametrize("value", [True, -1, 16, "15"])
def test_target_terminal_token_is_exact_and_inside_vocabulary(value: object) -> None:
    values = _configuration().model_dump(mode="python")
    values["target_preparation"]["terminal_token_id"] = value
    with pytest.raises(ValidationError):
        ExperimentTrainingConfigurationV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initialization_seed", True),
        ("initialization_seed", -1),
        ("initialization_seed", 2**63),
        ("deterministic", 1),
        ("deterministic", "true"),
    ],
)
def test_runtime_policy_requires_explicit_seed_and_determinism(
    field: str,
    value: object,
) -> None:
    values = ExperimentRuntimePolicyV1(
        initialization_seed=1907,
        deterministic=True,
    ).model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        ExperimentRuntimePolicyV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch_count", True),
        ("epoch_count", 0),
        ("epoch_count", 1_000_001),
        ("evaluate_test_after_training", 1),
        ("training_order", "shuffled"),
        ("checkpoint_policy", "after_each_step"),
        ("validation_policy", "after_training"),
        ("resume_policy", "best_effort"),
    ],
)
def test_epoch_execution_policy_is_explicit_and_strict(
    field: str,
    value: object,
) -> None:
    values = _epoch_policy(epoch_count=2).model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        ExperimentEpochExecutionPolicyV1.model_validate(values)


def test_execution_policy_discriminator_rejects_unknown_mode() -> None:
    values = _configuration().model_dump(mode="python")
    values["execution_policy"] = {"mode": "automatic"}

    with pytest.raises(ValidationError):
        ExperimentTrainingConfigurationV1.model_validate(values)


@pytest.mark.parametrize(
    ("validation_count", "test_count", "evaluate_test", "message"),
    [
        (0, 0, False, "validation"),
        (1, 0, True, "test"),
    ],
)
def test_epoch_run_requires_every_scheduled_evaluation_split_to_be_nonempty(
    validation_count: int,
    test_count: int,
    evaluate_test: bool,
    message: str,
) -> None:
    values = _run().payload.model_dump(mode="python")
    values["manifest"]["split_counts"] = {
        "train": 2,
        "validation": validation_count,
        "test": test_count,
    }
    values["configuration"]["execution_policy"] = {
        "mode": "deterministic_epochs",
        "epoch_count": 1,
        "training_order": "manifest_order_each_epoch",
        "checkpoint_policy": "after_each_complete_epoch",
        "validation_policy": "after_each_complete_epoch",
        "evaluate_test_after_training": evaluate_test,
        "resume_policy": "latest_checkpoint_plus_verified_replay",
    }

    with pytest.raises(ValidationError, match=message):
        ExperimentRunPayloadV1.model_validate(values)


def test_epoch_run_requires_strict_deterministic_runtime_policy() -> None:
    values = _run().payload.model_dump(mode="python")
    values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    values["configuration"]["runtime_policy"]["deterministic"] = False

    with pytest.raises(ValidationError, match="deterministic"):
        ExperimentRunPayloadV1.model_validate(values)


def test_epoch_run_accepts_exact_adamw_and_event_safety_boundaries() -> None:
    step_values = _run().payload.model_dump(mode="python")
    step_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    step_values["initial_checkpoint"]["completed_steps"] = 2**24 - 2
    assert (
        ExperimentRunPayloadV1.model_validate(step_values).initial_checkpoint.completed_steps
        == 2**24 - 2
    )

    event_values = _run().payload.model_dump(mode="python")
    event_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    event_values["manifest"]["split_counts"] = {
        "train": 999_998,
        "validation": 1,
        "test": 0,
    }
    assert (
        ExperimentRunPayloadV1.model_validate(event_values).manifest.split_counts.train == 999_998
    )


def test_epoch_run_rejects_adamw_step_and_event_count_overflow() -> None:
    step_values = _run().payload.model_dump(mode="python")
    step_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    step_values["initial_checkpoint"]["completed_steps"] = 2**24 - 1
    with pytest.raises(ValidationError, match="AdamW step"):
        ExperimentRunPayloadV1.model_validate(step_values)

    event_values = _run().payload.model_dump(mode="python")
    event_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    event_values["manifest"]["split_counts"] = {
        "train": 999_999,
        "validation": 1,
        "test": 0,
    }
    with pytest.raises(ValidationError, match="event safety"):
        ExperimentRunPayloadV1.model_validate(event_values)


@pytest.mark.parametrize(
    "names",
    [(), ("",), ("bad\0name",), ("same", "same"), ["weight"]],
)
def test_configuration_topology_names_are_nonempty_unique_exact_tuples(
    names: object,
) -> None:
    model_values = _model_configuration().model_dump(mode="python")
    model_values["state_names"] = names
    with pytest.raises(ValidationError):
        ExperimentModelConfigurationV1.model_validate(model_values)

    optimizer_values = _optimizer_configuration().model_dump(mode="python")
    optimizer_values["parameter_names"] = names
    with pytest.raises(ValidationError):
        ExperimentOptimizerConfigurationV1.model_validate(optimizer_values)


def test_model_configuration_reuses_checkpoint_cross_component_invariants() -> None:
    values = _model_configuration().model_dump(mode="python")
    values["runtime"]["max_target_tokens"] = 2

    with pytest.raises(ValidationError, match="target limit"):
        ExperimentModelConfigurationV1.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"split_ordinal": 0},
        {"split_ordinal": 3},
        {"step_index": True},
        {"loss": -1.0},
        {"loss": float("inf")},
        {"supervised_token_count": 0},
        {"gradient_l2_norm": float("nan")},
        {"gradients_clipped": 1},
    ],
)
def test_training_metric_rejects_invalid_measurements(changes: dict[str, Any]) -> None:
    values = _training_record().model_dump(mode="python")
    values.update(changes)

    with pytest.raises(ValidationError):
        ExperimentTrainingStepMetricV1.model_validate(values)


def test_evaluation_metric_accepts_exact_summary_and_overflow_sentinel() -> None:
    finite = ExperimentEvaluationMetricV1(
        event_type="evaluation",
        split="validation",
        example_count=2,
        supervised_token_count=5,
        mean_loss=2.0,
        perplexity=math.exp(2.0),
    )
    overflow_loss = math.nextafter(math.log(sys.float_info.max), math.inf)
    overflow = ExperimentEvaluationMetricV1(
        event_type="evaluation",
        split="test",
        example_count=1,
        supervised_token_count=2,
        mean_loss=overflow_loss,
        perplexity=None,
    )

    assert finite.perplexity == math.exp(2.0)
    assert overflow.perplexity is None


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "train"},
        {"example_count": 0},
        {"supervised_token_count": True},
        {"mean_loss": -0.1},
        {"mean_loss": 2.0, "perplexity": 7.0},
        {"mean_loss": 2.0, "perplexity": None},
        {"mean_loss": float("inf"), "perplexity": None},
    ],
)
def test_evaluation_metric_rejects_incoherent_summary(changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "event_type": "evaluation",
        "split": "validation",
        "example_count": 2,
        "supervised_token_count": 5,
        "mean_loss": 2.0,
        "perplexity": math.exp(2.0),
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        ExperimentEvaluationMetricV1.model_validate(values)


def test_event_payload_requires_training_step_to_match_common_counter() -> None:
    values = _event().payload.model_dump(mode="python")
    values["optimizer_completed_steps"] = 2

    with pytest.raises(ValidationError, match="step_index"):
        ExperimentEventPayloadV1.model_validate(values)


def test_checkpoint_event_requires_exact_common_state() -> None:
    record = ExperimentCheckpointEventV1(
        event_type="checkpoint",
        checkpoint=_checkpoint(completed_steps=2),
    )

    with pytest.raises(ValidationError, match="checkpoint event"):
        _event(
            record=record,
            optimizer_completed_steps=1,
            sidecar_state_sha256=SHA_C,
        )
    with pytest.raises(ValidationError, match="checkpoint event"):
        _event(
            record=record,
            optimizer_completed_steps=2,
            sidecar_state_sha256=SHA_B,
        )


def test_event_discriminator_rejects_unknown_type_and_record_extras() -> None:
    values = _event().payload.model_dump(mode="python")
    values["record"]["event_type"] = "metric"
    with pytest.raises(ValidationError):
        ExperimentEventPayloadV1.model_validate(values)

    values = _event().payload.model_dump(mode="python")
    values["record"]["unknown"] = 1
    with pytest.raises(ValidationError):
        ExperimentEventPayloadV1.model_validate(values)


@pytest.mark.parametrize("level", ["envelope", "schema", "payload", "configuration"])
def test_run_rejects_unknown_fields_at_every_level(level: str) -> None:
    values = _run().model_dump(mode="python")
    if level == "envelope":
        values["unknown"] = 1
    elif level == "schema":
        values["schema_identity"]["unknown"] = 1
    elif level == "payload":
        values["payload"]["unknown"] = 1
    else:
        values["payload"]["configuration"]["unknown"] = 1

    with pytest.raises(ValidationError):
        IrisExperimentRunV1.model_validate(values)


def test_nested_instances_are_revalidated() -> None:
    configuration = _configuration()
    object.__setattr__(configuration, "sha256", "invalid")

    with pytest.raises(ValidationError, match="sha256"):
        ExperimentRunPayloadV1(
            run_id="run-1",
            manifest=_run().payload.manifest,
            configuration=configuration,
            initial_checkpoint=_checkpoint(),
        )
