from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import ValidationError

from iris.artifacts.checkpoint_v1 import TeacherAdamWConfigV1, TeacherBackboneRuntimeV1
from iris.artifacts.experiment_log_v1 import (
    ExperimentEpochExecutionPolicyV1,
    ExperimentEvaluationMetricV1,
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
from iris.artifacts.experiment_v1 import ExperimentSplitCountsV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
PROTOCOL_SHA256: Literal["b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"] = (
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
)
PROFILE_SET_V3_SHA256: Literal[
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
] = "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
ARM_B_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)
ARM_D_COMPONENTS = (
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


def _runtime() -> TeacherBackboneRuntimeV1:
    return TeacherBackboneRuntimeV1(
        hidden_size=6,
        num_hidden_layers=4,
        vocabulary_size=16,
        max_position_embeddings=16,
        max_prompt_tokens=5,
        max_target_tokens=3,
        device="cpu",
        dtype="float32",
    )


def _model(arm_id: ExperimentArmIdV2) -> ExperimentModelConfigurationV2:
    if arm_id == "B":
        return ExperimentArmBModelConfigurationV2(
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
            backbone=_backbone(),
            cognitive=_cognitive(),
            cognitive_cycles=3,
            latent_token_count=3,
            component_names=ARM_B_COMPONENTS,
            runtime=_runtime(),
            state_names=("layer.weight",),
        )
    return ExperimentArmDModelConfigurationV2(
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        teacher_graph="limited_competitive_workspace_teacher_v1",
        backbone=_backbone(),
        cognitive=_cognitive(),
        workspace=_workspace(),
        component_names=ARM_D_COMPONENTS,
        runtime=_runtime(),
        state_names=("layer.weight",),
    )


def _optimizer() -> ExperimentOptimizerConfigurationV1:
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


def _configuration(arm_id: ExperimentArmIdV2) -> ExperimentTrainingConfigurationV2:
    return ExperimentTrainingConfigurationV2(
        sha256=SHA_C,
        model=_model(arm_id),
        optimizer=_optimizer(),
        target_preparation=ExperimentTargetPreparationPolicyV1(
            policy="raw_target_no_special_tokens_no_truncation_plus_one_eos",
            terminal_token_id=15,
        ),
        runtime_policy=ExperimentRuntimePolicyV1(
            initialization_seed=1907,
            deterministic=True,
        ),
        execution_policy=ExperimentManualExecutionPolicyV1(mode="manual_event_chain"),
    )


def _checkpoint(
    arm_id: ExperimentArmIdV2,
    *,
    completed_steps: int = 0,
) -> ExperimentCheckpointReferenceV2:
    if arm_id == "B":
        return ExperimentArmBCheckpointReferenceV2(
            checkpoint_schema_version=2,
            arm_id="B",
            payload_sha256=SHA_A,
            state_file_sha256=SHA_B,
            sidecar_state_sha256=SHA_C,
            completed_steps=completed_steps,
        )
    return ExperimentArmDCheckpointReferenceV2(
        checkpoint_schema_version=1,
        arm_id="D",
        payload_sha256=SHA_A,
        state_file_sha256=SHA_B,
        sidecar_state_sha256=SHA_C,
        completed_steps=completed_steps,
    )


def _manifest(
    *,
    train: int = 2,
    validation: int = 1,
    test: int = 0,
) -> ExperimentManifestReferenceV1:
    return ExperimentManifestReferenceV1(
        dataset_id="offline-data",
        payload_sha256=SHA_B,
        split_counts=ExperimentSplitCountsV1(
            train=train,
            validation=validation,
            test=test,
        ),
    )


def _run_payload(arm_id: ExperimentArmIdV2) -> ExperimentRunPayloadV2:
    if arm_id == "B":
        return ExperimentRunPayloadV2(
            run_id="offline.run-1",
            protocol_declaration_payload_sha256=PROTOCOL_SHA256,
            execution_profile_set_version=3,
            execution_profile_set_sha256=PROFILE_SET_V3_SHA256,
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            request_graph="independent_recurrent_modules_zero_memory_read_v1",
            teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
            manifest=_manifest(),
            configuration=_configuration("B"),
            initial_checkpoint=_checkpoint("B"),
        )
    return ExperimentRunPayloadV2(
        run_id="offline.run-1",
        protocol_declaration_payload_sha256=PROTOCOL_SHA256,
        execution_profile_set_version=3,
        execution_profile_set_sha256=PROFILE_SET_V3_SHA256,
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        request_graph="limited_competitive_workspace_v1",
        teacher_graph="limited_competitive_workspace_teacher_v1",
        manifest=_manifest(),
        configuration=_configuration("D"),
        initial_checkpoint=_checkpoint("D"),
    )


def _run(arm_id: ExperimentArmIdV2) -> IrisExperimentRunV2:
    return IrisExperimentRunV2(
        schema_identity=ExperimentRunSchemaV2(
            artifact_type="iris.experiment_run",
            version=2,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            journal_layout="immutable_run_and_chained_events",
        ),
        payload_sha256=SHA_A,
        payload=_run_payload(arm_id),
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
    arm_id: ExperimentArmIdV2,
    *,
    record: ExperimentEventRecordV2 | None = None,
    optimizer_completed_steps: int = 1,
    sidecar_state_sha256: str = SHA_B,
) -> IrisExperimentEventV2:
    return IrisExperimentEventV2(
        schema_identity=ExperimentEventSchemaV2(
            artifact_type="iris.experiment_event",
            version=2,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=SHA_A,
        payload=ExperimentEventPayloadV2(
            sequence=1,
            run_id="offline.run-1",
            arm_id=arm_id,
            run_payload_sha256=SHA_A,
            manifest_payload_sha256=SHA_B,
            configuration_sha256=SHA_C,
            predecessor_sha256=SHA_A,
            optimizer_completed_steps=optimizer_completed_steps,
            sidecar_state_sha256=sidecar_state_sha256,
            record=record or _training_record(),
        ),
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_arm_runs_and_events_are_strict_frozen_and_round_trip(
    arm_id: ExperimentArmIdV2,
) -> None:
    run = _run(arm_id)
    event = _event(arm_id)

    assert IrisExperimentRunV2.model_validate_json(run.model_dump_json()) == run
    assert IrisExperimentEventV2.model_validate_json(event.model_dump_json()) == event
    assert run.payload.configuration.model.arm_id == arm_id
    assert run.payload.initial_checkpoint.arm_id == arm_id
    with pytest.raises(ValidationError, match="frozen_instance"):
        run.payload.run_id = "changed"
    with pytest.raises(ValidationError, match="frozen_instance"):
        event.payload.sequence = 2


@pytest.mark.parametrize("schema", ["run", "event"])
@pytest.mark.parametrize("version", [True, "2", 2.0, 1, 3])
def test_schema_v2_identity_rejects_nonexact_or_unsupported_versions(
    schema: str,
    version: object,
) -> None:
    if schema == "run":
        values = _run("B").schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentRunSchemaV2.model_validate(values)
    else:
        values = _event("B").schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentEventSchemaV2.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_declaration_payload_sha256", SHA_A),
        ("execution_profile_set_version", True),
        ("execution_profile_set_version", "3"),
        ("execution_profile_set_version", 3.0),
        ("execution_profile_set_version", 2),
        ("execution_profile_set_sha256", SHA_A),
    ],
)
def test_run_is_exactly_bound_to_protocol_and_profile_v3(field: str, value: object) -> None:
    values = _run_payload("B").model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        ExperimentRunPayloadV2.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "D"])
@pytest.mark.parametrize("field", ["arm_kind", "request_graph", "teacher_graph"])
def test_run_rejects_every_cross_arm_profile_identity(
    arm_id: ExperimentArmIdV2,
    field: str,
) -> None:
    values = _run_payload(arm_id).model_dump(mode="python")
    alternate = _run_payload("D" if arm_id == "B" else "B").model_dump(mode="python")
    values[field] = alternate[field]

    with pytest.raises(ValidationError):
        ExperimentRunPayloadV2.model_validate(values)


def test_model_discriminator_keeps_workspace_unrepresentable_for_arm_b() -> None:
    values = _model("B").model_dump(mode="python")
    values["workspace"] = _workspace().model_dump(mode="python")

    with pytest.raises(ValidationError, match="workspace"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration("B").model_dump(mode="python"), "model": values}
        )


def test_model_discriminator_requires_workspace_for_arm_d() -> None:
    values = _model("D").model_dump(mode="python")
    del values["workspace"]

    with pytest.raises(ValidationError, match="workspace"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration("D").model_dump(mode="python"), "model": values}
        )


@pytest.mark.parametrize("arm_id", ["B", "D"])
@pytest.mark.parametrize("components", [(), ("layer_sampler",), ("goal_encoder", "layer_sampler")])
def test_model_configuration_requires_exact_component_ownership(
    arm_id: ExperimentArmIdV2,
    components: tuple[str, ...],
) -> None:
    values = _model(arm_id).model_dump(mode="python")
    values["component_names"] = components

    with pytest.raises(ValidationError, match="component"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration(arm_id).model_dump(mode="python"), "model": values}
        )


@pytest.mark.parametrize("cycles", [True, 2, "3"])
def test_arm_b_requires_exact_three_zero_read_cycles(cycles: object) -> None:
    values = _model("B").model_dump(mode="python")
    values["cognitive_cycles"] = cycles

    with pytest.raises(ValidationError, match="cognitive_cycles"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration("B").model_dump(mode="python"), "model": values}
        )


def test_arm_b_latent_cardinality_equals_cognitive_module_count() -> None:
    values = _model("B").model_dump(mode="python")
    values["latent_token_count"] = 2

    with pytest.raises(ValidationError, match="latent token"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration("B").model_dump(mode="python"), "model": values}
        )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_model_configuration_reuses_checkpoint_context_invariants(
    arm_id: ExperimentArmIdV2,
) -> None:
    values = _model(arm_id).model_dump(mode="python")
    values["runtime"]["max_target_tokens"] = 2

    with pytest.raises(ValidationError, match="target limit"):
        ExperimentTrainingConfigurationV2.model_validate(
            {**_configuration(arm_id).model_dump(mode="python"), "model": values}
        )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_checkpoint_reference_discriminator_selects_exact_arm_schema(
    arm_id: ExperimentArmIdV2,
) -> None:
    checkpoint = _checkpoint(arm_id)

    if arm_id == "B":
        assert isinstance(checkpoint, ExperimentArmBCheckpointReferenceV2)
        assert checkpoint.checkpoint_schema_version == 2
    else:
        assert isinstance(checkpoint, ExperimentArmDCheckpointReferenceV2)
        assert checkpoint.checkpoint_schema_version == 1


@pytest.mark.parametrize(
    ("arm_id", "checkpoint_version"),
    [("B", 1), ("B", True), ("B", "2"), ("B", 3), ("D", 2), ("D", False), ("D", "1")],
)
def test_checkpoint_reference_rejects_cross_arm_or_nonexact_version(
    arm_id: ExperimentArmIdV2,
    checkpoint_version: object,
) -> None:
    values = _checkpoint(arm_id).model_dump(mode="python")
    values["checkpoint_schema_version"] = checkpoint_version

    with pytest.raises(ValidationError):
        ExperimentCheckpointEventV2.model_validate(
            {"event_type": "checkpoint", "checkpoint": values}
        )


@pytest.mark.parametrize("completed_steps", [True, "0", -1, 2**24 + 1])
def test_checkpoint_reference_enforces_exact_completed_step_bounds(
    completed_steps: object,
) -> None:
    values = _checkpoint("B").model_dump(mode="python")
    values["completed_steps"] = completed_steps

    with pytest.raises(ValidationError):
        ExperimentCheckpointEventV2.model_validate(
            {"event_type": "checkpoint", "checkpoint": values}
        )


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 1, True])
def test_checkpoint_and_event_hashes_are_canonical(digest: object) -> None:
    checkpoint_values = _checkpoint("B").model_dump(mode="python")
    checkpoint_values["payload_sha256"] = digest
    with pytest.raises(ValidationError, match="SHA-256"):
        ExperimentCheckpointEventV2.model_validate(
            {"event_type": "checkpoint", "checkpoint": checkpoint_values}
        )

    event_values = _event("B").payload.model_dump(mode="python")
    event_values["predecessor_sha256"] = digest
    with pytest.raises(ValidationError, match="SHA-256"):
        ExperimentEventPayloadV2.model_validate(event_values)


@pytest.mark.parametrize("run_arm", ["B", "D"])
def test_run_rejects_other_arm_model_and_checkpoint(run_arm: ExperimentArmIdV2) -> None:
    other_arm: ExperimentArmIdV2 = "D" if run_arm == "B" else "B"
    model_values = _run_payload(run_arm).model_dump(mode="python")
    model_values["configuration"]["model"] = _model(other_arm).model_dump(mode="python")
    with pytest.raises(ValidationError, match="model configuration"):
        ExperimentRunPayloadV2.model_validate(model_values)

    checkpoint_values = _run_payload(run_arm).model_dump(mode="python")
    checkpoint_values["initial_checkpoint"] = _checkpoint(other_arm).model_dump(mode="python")
    with pytest.raises(ValidationError, match="initial checkpoint"):
        ExperimentRunPayloadV2.model_validate(checkpoint_values)


@pytest.mark.parametrize("value", [True, -1, 16, "15"])
def test_target_terminal_token_is_exact_and_inside_vocabulary(value: object) -> None:
    values = _configuration("B").model_dump(mode="python")
    values["target_preparation"]["terminal_token_id"] = value

    with pytest.raises(ValidationError):
        ExperimentTrainingConfigurationV2.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_epoch_run_accepts_exact_step_and_event_safety_boundaries(
    arm_id: ExperimentArmIdV2,
) -> None:
    step_values = _run_payload(arm_id).model_dump(mode="python")
    step_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    step_values["initial_checkpoint"]["completed_steps"] = 2**24 - 2
    assert (
        ExperimentRunPayloadV2.model_validate(step_values).initial_checkpoint.completed_steps
        == 2**24 - 2
    )

    event_values = _run_payload(arm_id).model_dump(mode="python")
    event_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    event_values["manifest"] = _manifest(train=999_998).model_dump(mode="python")
    assert (
        ExperimentRunPayloadV2.model_validate(event_values).manifest.split_counts.train == 999_998
    )


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_epoch_run_rejects_step_and_event_count_overflow(
    arm_id: ExperimentArmIdV2,
) -> None:
    step_values = _run_payload(arm_id).model_dump(mode="python")
    step_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    step_values["initial_checkpoint"]["completed_steps"] = 2**24 - 1
    with pytest.raises(ValidationError, match="AdamW step"):
        ExperimentRunPayloadV2.model_validate(step_values)

    event_values = _run_payload(arm_id).model_dump(mode="python")
    event_values["configuration"]["execution_policy"] = _epoch_policy().model_dump(mode="python")
    event_values["manifest"] = _manifest(train=999_999).model_dump(mode="python")
    with pytest.raises(ValidationError, match="event safety"):
        ExperimentRunPayloadV2.model_validate(event_values)


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_epoch_run_requires_determinism_and_nonempty_scheduled_test(
    arm_id: ExperimentArmIdV2,
) -> None:
    nondeterministic = _run_payload(arm_id).model_dump(mode="python")
    nondeterministic["configuration"]["execution_policy"] = _epoch_policy().model_dump(
        mode="python"
    )
    nondeterministic["configuration"]["runtime_policy"]["deterministic"] = False
    with pytest.raises(ValidationError, match="deterministic"):
        ExperimentRunPayloadV2.model_validate(nondeterministic)

    empty_test = _run_payload(arm_id).model_dump(mode="python")
    empty_test["configuration"]["execution_policy"] = _epoch_policy(evaluate_test=True).model_dump(
        mode="python"
    )
    with pytest.raises(ValidationError, match="test split"):
        ExperimentRunPayloadV2.model_validate(empty_test)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence", True),
        ("sequence", 0),
        ("sequence", 1_000_001),
        ("optimizer_completed_steps", True),
        ("optimizer_completed_steps", -1),
        ("optimizer_completed_steps", 2**24 + 1),
    ],
)
def test_event_payload_enforces_exact_counter_bounds(field: str, value: object) -> None:
    values = _event("B").payload.model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        ExperimentEventPayloadV2.model_validate(values)


def test_event_requires_training_step_to_match_common_counter() -> None:
    values = _event("B").payload.model_dump(mode="python")
    values["optimizer_completed_steps"] = 2

    with pytest.raises(ValidationError, match="step_index"):
        ExperimentEventPayloadV2.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_checkpoint_event_uses_new_reference_and_exact_common_state(
    arm_id: ExperimentArmIdV2,
) -> None:
    checkpoint = _checkpoint(arm_id, completed_steps=2)
    record = ExperimentCheckpointEventV2(event_type="checkpoint", checkpoint=checkpoint)
    event = _event(
        arm_id,
        record=record,
        optimizer_completed_steps=2,
        sidecar_state_sha256=SHA_C,
    )
    assert event.payload.record == record

    wrong_steps = event.payload.model_dump(mode="python")
    wrong_steps["optimizer_completed_steps"] = 1
    with pytest.raises(ValidationError, match="training state"):
        ExperimentEventPayloadV2.model_validate(wrong_steps)

    wrong_state = event.payload.model_dump(mode="python")
    wrong_state["sidecar_state_sha256"] = SHA_B
    with pytest.raises(ValidationError, match="training state"):
        ExperimentEventPayloadV2.model_validate(wrong_state)


@pytest.mark.parametrize("event_arm", ["B", "D"])
def test_checkpoint_event_rejects_other_arm(event_arm: ExperimentArmIdV2) -> None:
    other_arm: ExperimentArmIdV2 = "D" if event_arm == "B" else "B"
    record = ExperimentCheckpointEventV2(
        event_type="checkpoint",
        checkpoint=_checkpoint(other_arm, completed_steps=1),
    )

    with pytest.raises(ValidationError, match="arm identity"):
        _event(event_arm, record=record, sidecar_state_sha256=SHA_C)


def test_event_discriminator_rejects_unknown_type_and_record_extras() -> None:
    values = _event("B").payload.model_dump(mode="python")
    values["record"]["event_type"] = "metric"
    with pytest.raises(ValidationError):
        ExperimentEventPayloadV2.model_validate(values)

    values = _event("B").payload.model_dump(mode="python")
    values["record"]["unknown"] = 1
    with pytest.raises(ValidationError):
        ExperimentEventPayloadV2.model_validate(values)


def test_v1_metric_shapes_remain_valid_inside_v2_events() -> None:
    evaluation = ExperimentEvaluationMetricV1(
        event_type="evaluation",
        split="validation",
        example_count=2,
        supervised_token_count=5,
        mean_loss=0.0,
        perplexity=1.0,
    )

    assert _event("B", record=evaluation).payload.record == evaluation
    assert _event("D", record=_training_record()).payload.record.event_type == "training_step"


@pytest.mark.parametrize(
    "level",
    ["envelope", "schema", "payload", "configuration", "model", "checkpoint"],
)
def test_run_rejects_unknown_fields_at_every_v2_level(level: str) -> None:
    values: dict[str, Any] = _run("B").model_dump(mode="python")
    if level == "envelope":
        values["unknown"] = 1
    elif level == "schema":
        values["schema_identity"]["unknown"] = 1
    elif level == "payload":
        values["payload"]["unknown"] = 1
    elif level == "configuration":
        values["payload"]["configuration"]["unknown"] = 1
    elif level == "model":
        values["payload"]["configuration"]["model"]["unknown"] = 1
    else:
        values["payload"]["initial_checkpoint"]["unknown"] = 1

    with pytest.raises(ValidationError):
        IrisExperimentRunV2.model_validate(values)


def test_nested_v2_instances_are_revalidated() -> None:
    configuration = _configuration("B")
    object.__setattr__(configuration, "sha256", "invalid")

    with pytest.raises(ValidationError, match="sha256"):
        ExperimentRunPayloadV2(
            run_id="offline.run-1",
            protocol_declaration_payload_sha256=PROTOCOL_SHA256,
            execution_profile_set_version=3,
            execution_profile_set_sha256=PROFILE_SET_V3_SHA256,
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            request_graph="independent_recurrent_modules_zero_memory_read_v1",
            teacher_graph="independent_recurrent_modules_zero_memory_read_teacher_v1",
            manifest=_manifest(),
            configuration=configuration,
            initial_checkpoint=_checkpoint("B"),
        )
