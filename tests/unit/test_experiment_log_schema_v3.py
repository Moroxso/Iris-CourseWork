from __future__ import annotations

from typing import Any, Literal, cast

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
from iris.artifacts.experiment_log_v2 import ExperimentRunSchemaV2
from iris.artifacts.experiment_log_v3 import (
    ExperimentArmBCheckpointReferenceV3,
    ExperimentArmBModelConfigurationV3,
    ExperimentArmCCheckpointReferenceV3,
    ExperimentArmCModelConfigurationV3,
    ExperimentArmDCheckpointReferenceV3,
    ExperimentArmDModelConfigurationV3,
    ExperimentArmIdV3,
    ExperimentCheckpointEventV3,
    ExperimentCheckpointReferenceV3,
    ExperimentEventPayloadV3,
    ExperimentEventRecordV3,
    ExperimentEventSchemaV3,
    ExperimentModelConfigurationV3,
    ExperimentRunPayloadV3,
    ExperimentRunSchemaV3,
    ExperimentTrainingConfigurationV3,
    IrisExperimentEventV3,
    IrisExperimentRunV3,
)
from iris.artifacts.experiment_v1 import ExperimentSplitCountsV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
PROTOCOL_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
PROFILE_SET_V4_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
POLICY_SHA256: Literal[
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
] = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
ARM_B_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)
ARM_C_COMPONENTS = (
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


def _model(arm_id: ExperimentArmIdV3) -> ExperimentModelConfigurationV3:
    if arm_id == "B":
        return ExperimentArmBModelConfigurationV3(
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
    if arm_id == "C":
        return ExperimentArmCModelConfigurationV3(
            arm_id="C",
            arm_kind="llm_modules_unlimited_shared_buffer",
            teacher_graph="append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
            shared_buffer_policy_sha256=POLICY_SHA256,
            backbone=_backbone(),
            cognitive=_cognitive(),
            cognitive_cycles=3,
            pre_update_read_count=3,
            final_output_read_count=1,
            entries_per_cycle=3,
            final_entry_count=9,
            latent_token_count=3,
            component_names=ARM_C_COMPONENTS,
            runtime=_runtime(),
            state_names=("layer.weight",),
        )
    return ExperimentArmDModelConfigurationV3(
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


def _configuration(arm_id: ExperimentArmIdV3) -> ExperimentTrainingConfigurationV3:
    return ExperimentTrainingConfigurationV3(
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
    arm_id: ExperimentArmIdV3,
    *,
    completed_steps: int = 0,
) -> ExperimentCheckpointReferenceV3:
    if arm_id == "B":
        return ExperimentArmBCheckpointReferenceV3(
            checkpoint_schema_version=2,
            arm_id="B",
            payload_sha256=SHA_A,
            state_file_sha256=SHA_B,
            sidecar_state_sha256=SHA_C,
            completed_steps=completed_steps,
        )
    if arm_id == "C":
        return ExperimentArmCCheckpointReferenceV3(
            checkpoint_schema_version=3,
            arm_id="C",
            payload_sha256=SHA_A,
            state_file_sha256=SHA_B,
            sidecar_state_sha256=SHA_C,
            completed_steps=completed_steps,
        )
    return ExperimentArmDCheckpointReferenceV3(
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


def _profile_identity(
    arm_id: ExperimentArmIdV3,
) -> tuple[str, str, str]:
    if arm_id == "B":
        return (
            "llm_modules_no_workspace",
            "independent_recurrent_modules_zero_memory_read_v1",
            "independent_recurrent_modules_zero_memory_read_teacher_v1",
        )
    if arm_id == "C":
        return (
            "llm_modules_unlimited_shared_buffer",
            "append_only_all_proposals_prior_cycle_shared_buffer_v1",
            "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1",
        )
    return (
        "llm_modules_limited_competitive_workspace",
        "limited_competitive_workspace_v1",
        "limited_competitive_workspace_teacher_v1",
    )


def _run_payload(arm_id: ExperimentArmIdV3) -> ExperimentRunPayloadV3:
    arm_kind, request_graph, teacher_graph = _profile_identity(arm_id)
    return ExperimentRunPayloadV3.model_validate(
        {
            "run_id": "offline.run-1",
            "protocol_declaration_payload_sha256": PROTOCOL_SHA256,
            "execution_profile_set_version": 4,
            "execution_profile_set_sha256": PROFILE_SET_V4_SHA256,
            "shared_buffer_policy_sha256": POLICY_SHA256,
            "arm_id": arm_id,
            "arm_kind": arm_kind,
            "request_graph": request_graph,
            "teacher_graph": teacher_graph,
            "manifest": _manifest(),
            "configuration": _configuration(arm_id),
            "initial_checkpoint": _checkpoint(arm_id),
        }
    )


def _run(arm_id: ExperimentArmIdV3) -> IrisExperimentRunV3:
    return IrisExperimentRunV3(
        schema_identity=ExperimentRunSchemaV3(
            artifact_type="iris.experiment_run",
            version=3,
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
    arm_id: ExperimentArmIdV3,
    *,
    record: ExperimentEventRecordV3 | None = None,
    optimizer_completed_steps: int = 1,
    sidecar_state_sha256: str = SHA_B,
) -> IrisExperimentEventV3:
    return IrisExperimentEventV3(
        schema_identity=ExperimentEventSchemaV3(
            artifact_type="iris.experiment_event",
            version=3,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            chain_algorithm="sha256_predecessor",
        ),
        payload_sha256=SHA_A,
        payload=ExperimentEventPayloadV3(
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


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_profile_v4_trainable_runs_and_events_are_frozen_round_trips(
    arm_id: ExperimentArmIdV3,
) -> None:
    run = _run(arm_id)
    event = _event(arm_id)

    assert IrisExperimentRunV3.model_validate_json(run.model_dump_json()) == run
    assert IrisExperimentEventV3.model_validate_json(event.model_dump_json()) == event
    assert run.payload.configuration.model.arm_id == arm_id
    assert run.payload.initial_checkpoint.arm_id == arm_id
    with pytest.raises(ValidationError, match="frozen_instance"):
        run.payload.run_id = "changed"
    with pytest.raises(ValidationError, match="frozen_instance"):
        event.payload.sequence = 2


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_json_arrays_round_trip_as_tuples_without_accepting_python_lists(
    arm_id: ExperimentArmIdV3,
) -> None:
    model_json = _model(arm_id).model_dump_json()
    assert type(type(_model(arm_id)).model_validate_json(model_json).component_names) is tuple

    model_values = _model(arm_id).model_dump(mode="python")
    model_values["component_names"] = list(model_values["component_names"])
    with pytest.raises(ValidationError, match="tuple"):
        type(_model(arm_id)).model_validate(model_values)

    configuration_values = _configuration(arm_id).model_dump(mode="python")
    configuration_values["optimizer"]["parameter_names"] = ["layer.weight"]
    with pytest.raises(ValidationError, match="tuple"):
        ExperimentTrainingConfigurationV3.model_validate(configuration_values)


@pytest.mark.parametrize("schema", ["run", "event"])
@pytest.mark.parametrize("version", [True, "3", 3.0, 1, 2, 4])
def test_schema_v3_rejects_nonexact_or_unsupported_versions(
    schema: str,
    version: object,
) -> None:
    if schema == "run":
        values = _run("B").schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentRunSchemaV3.model_validate(values)
    else:
        values = _event("B").schema_identity.model_dump(mode="python")
        values["version"] = version
        with pytest.raises(ValidationError, match="version"):
            ExperimentEventSchemaV3.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_declaration_payload_sha256", SHA_A),
        ("execution_profile_set_version", True),
        ("execution_profile_set_version", "4"),
        ("execution_profile_set_version", 4.0),
        ("execution_profile_set_version", 3),
        ("execution_profile_set_sha256", SHA_A),
        ("shared_buffer_policy_sha256", SHA_A),
    ],
)
def test_run_binds_exact_protocol_profile_v4_and_policy(field: str, value: object) -> None:
    values = _run_payload("C").model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        ExperimentRunPayloadV3.model_validate(values)


def test_arm_a_is_unrepresentable_because_it_has_no_teacher() -> None:
    values = _run_payload("B").model_dump(mode="python")
    values.update(
        {
            "arm_id": "A",
            "arm_kind": "base_llm",
            "request_graph": "frozen_backbone_prompt_only_v1",
            "teacher_graph": "none_frozen_backbone",
        }
    )

    with pytest.raises(ValidationError):
        ExperimentRunPayloadV3.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
@pytest.mark.parametrize("field", ["arm_kind", "request_graph", "teacher_graph"])
def test_run_rejects_every_cross_arm_profile_identity(
    arm_id: ExperimentArmIdV3,
    field: str,
) -> None:
    alternatives: dict[ExperimentArmIdV3, ExperimentArmIdV3] = {
        "B": "C",
        "C": "D",
        "D": "B",
    }
    values = _run_payload(arm_id).model_dump(mode="python")
    alternate = _run_payload(alternatives[arm_id]).model_dump(mode="python")
    values[field] = alternate[field]

    with pytest.raises(ValidationError):
        ExperimentRunPayloadV3.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "C"])
def test_cognitive_only_models_make_workspace_unrepresentable(
    arm_id: Literal["B", "C"],
) -> None:
    values = _configuration(arm_id).model_dump(mode="python")
    values["model"]["workspace"] = _workspace().model_dump(mode="python")

    with pytest.raises(ValidationError, match="workspace"):
        ExperimentTrainingConfigurationV3.model_validate(values)


def test_arm_d_requires_workspace_and_arm_c_requires_policy() -> None:
    arm_d = _configuration("D").model_dump(mode="python")
    del arm_d["model"]["workspace"]
    with pytest.raises(ValidationError, match="workspace"):
        ExperimentTrainingConfigurationV3.model_validate(arm_d)

    arm_c = _configuration("C").model_dump(mode="python")
    del arm_c["model"]["shared_buffer_policy_sha256"]
    with pytest.raises(ValidationError, match="shared_buffer_policy_sha256"):
        ExperimentTrainingConfigurationV3.model_validate(arm_c)

    arm_c = _configuration("C").model_dump(mode="python")
    arm_c["model"]["shared_buffer_policy_sha256"] = SHA_A
    with pytest.raises(ValidationError, match="shared_buffer_policy_sha256"):
        ExperimentTrainingConfigurationV3.model_validate(arm_c)


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_model_configuration_requires_exact_component_ownership(
    arm_id: ExperimentArmIdV3,
) -> None:
    values = _configuration(arm_id).model_dump(mode="python")
    values["model"]["component_names"] = ("layer_sampler",)

    with pytest.raises(ValidationError, match="component"):
        ExperimentTrainingConfigurationV3.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cognitive_cycles", 2),
        ("pre_update_read_count", 2),
        ("final_output_read_count", 2),
        ("entries_per_cycle", 2),
        ("final_entry_count", 8),
        ("latent_token_count", 2),
    ],
)
def test_arm_c_requires_exact_shared_buffer_lifecycle(field: str, value: object) -> None:
    values = _configuration("C").model_dump(mode="python")
    values["model"][field] = value

    with pytest.raises(ValidationError):
        ExperimentTrainingConfigurationV3.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_model_configuration_reuses_checkpoint_context_invariants(
    arm_id: ExperimentArmIdV3,
) -> None:
    values = _configuration(arm_id).model_dump(mode="python")
    values["model"]["runtime"]["max_target_tokens"] = 2

    with pytest.raises(ValidationError, match="target limit"):
        ExperimentTrainingConfigurationV3.model_validate(values)


@pytest.mark.parametrize(
    ("arm_id", "expected_type", "expected_version"),
    [
        ("B", ExperimentArmBCheckpointReferenceV3, 2),
        ("C", ExperimentArmCCheckpointReferenceV3, 3),
        ("D", ExperimentArmDCheckpointReferenceV3, 1),
    ],
)
def test_checkpoint_references_bind_exact_arm_schema_versions(
    arm_id: ExperimentArmIdV3,
    expected_type: type[ExperimentCheckpointReferenceV3],
    expected_version: int,
) -> None:
    checkpoint = _checkpoint(arm_id)

    assert type(checkpoint) is expected_type
    assert checkpoint.checkpoint_schema_version == expected_version


@pytest.mark.parametrize(
    ("arm_id", "checkpoint_version"),
    [
        ("B", 1),
        ("B", 3),
        ("B", True),
        ("C", 1),
        ("C", 2),
        ("C", "3"),
        ("D", 2),
        ("D", 3),
        ("D", False),
    ],
)
def test_checkpoint_reference_rejects_cross_arm_or_nonexact_version(
    arm_id: ExperimentArmIdV3,
    checkpoint_version: object,
) -> None:
    values = _checkpoint(arm_id).model_dump(mode="python")
    values["checkpoint_schema_version"] = checkpoint_version

    with pytest.raises(ValidationError):
        ExperimentCheckpointEventV3.model_validate(
            {"event_type": "checkpoint", "checkpoint": values}
        )


@pytest.mark.parametrize("run_arm", ["B", "C", "D"])
def test_run_rejects_cross_arm_model_and_checkpoint(run_arm: ExperimentArmIdV3) -> None:
    other_arm: dict[ExperimentArmIdV3, ExperimentArmIdV3] = {
        "B": "C",
        "C": "D",
        "D": "B",
    }
    values = _run_payload(run_arm).model_dump(mode="python")
    values["configuration"]["model"] = _model(other_arm[run_arm]).model_dump(mode="python")
    with pytest.raises(ValidationError, match="model configuration"):
        ExperimentRunPayloadV3.model_validate(values)

    values = _run_payload(run_arm).model_dump(mode="python")
    values["initial_checkpoint"] = _checkpoint(other_arm[run_arm]).model_dump(mode="python")
    with pytest.raises(ValidationError, match="initial checkpoint"):
        ExperimentRunPayloadV3.model_validate(values)


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_checkpoint_event_uses_exact_common_state(arm_id: ExperimentArmIdV3) -> None:
    record = ExperimentCheckpointEventV3(
        event_type="checkpoint",
        checkpoint=_checkpoint(arm_id, completed_steps=2),
    )
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
        ExperimentEventPayloadV3.model_validate(wrong_steps)

    wrong_state = event.payload.model_dump(mode="python")
    wrong_state["sidecar_state_sha256"] = SHA_B
    with pytest.raises(ValidationError, match="training state"):
        ExperimentEventPayloadV3.model_validate(wrong_state)


@pytest.mark.parametrize("event_arm", ["B", "C", "D"])
def test_checkpoint_event_rejects_other_arm(event_arm: ExperimentArmIdV3) -> None:
    other_arm: dict[ExperimentArmIdV3, ExperimentArmIdV3] = {
        "B": "C",
        "C": "D",
        "D": "B",
    }
    record = ExperimentCheckpointEventV3(
        event_type="checkpoint",
        checkpoint=_checkpoint(other_arm[event_arm], completed_steps=1),
    )

    with pytest.raises(ValidationError, match="arm identity"):
        _event(event_arm, record=record, sidecar_state_sha256=SHA_C)


def test_v1_metric_shapes_remain_fsm_compatible_inside_v3_events() -> None:
    evaluation = ExperimentEvaluationMetricV1(
        event_type="evaluation",
        split="validation",
        example_count=2,
        supervised_token_count=5,
        mean_loss=0.0,
        perplexity=1.0,
    )

    assert _event("C", record=evaluation).payload.record == evaluation
    assert _event("D", record=_training_record()).payload.record.event_type == "training_step"


@pytest.mark.parametrize("arm_id", ["B", "C", "D"])
def test_epoch_execution_bounds_remain_fsm_compatible(arm_id: ExperimentArmIdV3) -> None:
    values = _run_payload(arm_id).model_dump(mode="python")
    values["configuration"]["execution_policy"] = ExperimentEpochExecutionPolicyV1(
        mode="deterministic_epochs",
        epoch_count=1,
        training_order="manifest_order_each_epoch",
        checkpoint_policy="after_each_complete_epoch",
        validation_policy="after_each_complete_epoch",
        evaluate_test_after_training=False,
        resume_policy="latest_checkpoint_plus_verified_replay",
    ).model_dump(mode="python")
    values["initial_checkpoint"]["completed_steps"] = 2**24 - 2
    assert ExperimentRunPayloadV3.model_validate(values).initial_checkpoint.completed_steps == (
        2**24 - 2
    )

    values["initial_checkpoint"]["completed_steps"] = 2**24 - 1
    with pytest.raises(ValidationError, match="AdamW step"):
        ExperimentRunPayloadV3.model_validate(values)


class _DerivedArmBModel(ExperimentArmBModelConfigurationV3):
    pass


class _DerivedArmCCheckpoint(ExperimentArmCCheckpointReferenceV3):
    pass


class _DerivedRunPayload(ExperimentRunPayloadV3):
    pass


def test_persisted_boundaries_reject_schema_subclasses() -> None:
    derived_model = _DerivedArmBModel.model_validate(_model("B").model_dump(mode="python"))
    configuration_values = _configuration("B").model_dump(mode="python")
    configuration_values["model"] = derived_model
    with pytest.raises(ValidationError, match="exact version-3 schema type"):
        ExperimentTrainingConfigurationV3.model_validate(configuration_values)

    derived_checkpoint = _DerivedArmCCheckpoint.model_validate(
        _checkpoint("C").model_dump(mode="python")
    )
    with pytest.raises(ValidationError, match="exact version-3 schema type"):
        ExperimentCheckpointEventV3(
            event_type="checkpoint",
            checkpoint=derived_checkpoint,
        )

    derived_payload = _DerivedRunPayload.model_validate(
        _run_payload("D").model_dump(mode="python")
    )
    with pytest.raises(ValidationError, match="exact version-3 schema type"):
        IrisExperimentRunV3(
            schema_identity=_run("D").schema_identity,
            payload_sha256=SHA_A,
            payload=derived_payload,
        )


def test_tampered_nested_instances_are_revalidated_and_rejected() -> None:
    configuration = _configuration("C")
    object.__setattr__(configuration, "sha256", "invalid")
    values = _run_payload("C").model_dump(mode="python")
    values["configuration"] = configuration
    with pytest.raises(ValidationError, match="sha256"):
        ExperimentRunPayloadV3.model_validate(values)

    payload = _run_payload("C")
    object.__setattr__(payload, "execution_profile_set_sha256", SHA_A)
    with pytest.raises(ValidationError, match="execution_profile_set_sha256"):
        IrisExperimentRunV3(
            schema_identity=_run("C").schema_identity,
            payload_sha256=SHA_A,
            payload=payload,
        )


@pytest.mark.parametrize(
    "level",
    ["envelope", "schema", "payload", "configuration", "model", "checkpoint"],
)
def test_run_rejects_unknown_fields_at_every_v3_level(level: str) -> None:
    values: dict[str, Any] = _run("C").model_dump(mode="python")
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
        IrisExperimentRunV3.model_validate(values)


def test_predecessor_schema_versions_remain_distinct_and_unchanged() -> None:
    predecessor = ExperimentRunSchemaV2(
        artifact_type="iris.experiment_run",
        version=2,
        canonicalization="iris-json-v1",
        hash_algorithm="sha256",
        journal_layout="immutable_run_and_chained_events",
    )
    current = _run("B").schema_identity

    assert predecessor.version == 2
    assert current.version == 3
    with pytest.raises(ValidationError):
        ExperimentRunSchemaV3.model_validate(predecessor.model_dump(mode="python"))


def test_type_aliases_accept_exact_concrete_types() -> None:
    model: ExperimentModelConfigurationV3 = _model("C")
    checkpoint: ExperimentCheckpointReferenceV3 = _checkpoint("C")
    record: ExperimentEventRecordV3 = ExperimentCheckpointEventV3(
        event_type="checkpoint",
        checkpoint=checkpoint,
    )

    assert type(model) is ExperimentArmCModelConfigurationV3
    assert type(checkpoint) is ExperimentArmCCheckpointReferenceV3
    assert type(cast(ExperimentCheckpointEventV3, record)) is ExperimentCheckpointEventV3
