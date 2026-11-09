"""Schema contracts for the recovered Iris-GW-0.1 four-arm protocol."""

from typing import Any

import pytest
from pydantic import ValidationError

from iris.application import build_four_arm_protocol_declaration_v1
from iris.artifacts import (
    FourArmArchitectureConstantsV1,
    FourArmDefinitionV1,
    FourArmProtocolPayloadV1,
    FourArmProtocolSchemaV1,
    IrisFourArmProtocolDeclarationV1,
)


def test_declaration_contains_exact_recovered_arms_and_tasks() -> None:
    declaration = build_four_arm_protocol_declaration_v1()

    assert tuple((arm.arm_id, arm.arm_kind) for arm in declaration.payload.arms) == (
        ("A", "base_llm"),
        ("B", "llm_modules_no_workspace"),
        ("C", "llm_modules_unlimited_shared_buffer"),
        ("D", "llm_modules_limited_competitive_workspace"),
    )
    assert declaration.payload.task_families == (
        "split_information",
        "delayed_planning",
        "code_generation_and_verification",
        "hidden_goal_retention",
        "module_conflict",
    )
    criterion = declaration.payload.mechanistic_criterion
    assert criterion.treatment_arm == "D"
    assert criterion.control_arms == ("B", "C")
    assert criterion.relation == "strictly_outperforms"
    assert criterion.quantitative_rule_status == "not_recovered"


def test_declaration_records_source_boundary_and_execution_blockers() -> None:
    payload = build_four_arm_protocol_declaration_v1().payload

    assert payload.source.arm_definition_turn_id == "0b2e3f67-511d-40f1-a652-167abd8c1d58"
    assert payload.source.final_specification_recovery == "prefix_only_20000_characters"
    assert payload.execution_status == ("blocked_until_explicit_protocol_and_arm_implementations")
    assert payload.unresolved_items == (
        "arm_a_execution_semantics",
        "arm_b_execution_semantics",
        "arm_c_execution_semantics",
        "unlimited_shared_buffer_semantics",
        "main_backbone_pinned_revision",
        "dataset_construction_and_split_policy",
        "task_specific_evaluation_protocol",
        "seeds_repeats_epochs_and_optimizer",
        "comparison_aggregation_uncertainty_and_threshold",
        "study_storage_budget",
    )


def test_declaration_records_main_and_debug_backbones_without_claiming_a_revision() -> None:
    architecture = build_four_arm_protocol_declaration_v1().payload.architecture

    assert architecture.main_backbone_repository == "Qwen/Qwen3-1.7B"
    assert architecture.debug_backbone_repository == "Qwen/Qwen3-0.6B"
    assert architecture.cognitive_module_count == 8
    assert architecture.cognitive_state_width == 256
    assert architecture.limited_workspace_slot_count == 8
    assert architecture.limited_workspace_winners_per_cycle == 2
    assert architecture.cognitive_cycle_count == 3
    assert not architecture.fast_weights_enabled
    assert not architecture.ciel_enabled
    assert not architecture.snn_enabled


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_rejects_coerced_versions(version: Any) -> None:
    with pytest.raises(ValidationError, match="exact integer"):
        FourArmProtocolSchemaV1.model_validate(
            {
                "artifact_type": "iris.four_arm_protocol_declaration",
                "version": version,
                "canonicalization": "iris-json-v1",
                "hash_algorithm": "sha256",
                "declaration_scope": "recovered_facts_and_explicit_gaps",
            }
        )


@pytest.mark.parametrize("value", [True, 8.0, "8"])
def test_architecture_rejects_coerced_integer_constants(value: Any) -> None:
    raw = build_four_arm_protocol_declaration_v1().payload.architecture.model_dump(mode="python")
    raw["cognitive_module_count"] = value

    with pytest.raises(ValidationError, match="exact integers"):
        FourArmArchitectureConstantsV1.model_validate(raw)


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_architecture_rejects_coerced_feature_gates(value: Any) -> None:
    raw = build_four_arm_protocol_declaration_v1().payload.architecture.model_dump(mode="python")
    raw["fast_weights_enabled"] = value

    with pytest.raises(ValidationError, match="exact booleans"):
        FourArmArchitectureConstantsV1.model_validate(raw)


def test_payload_rejects_reordered_arm_definitions() -> None:
    payload = build_four_arm_protocol_declaration_v1().payload
    reordered = (payload.arms[1], payload.arms[0], *payload.arms[2:])

    with pytest.raises(ValidationError, match="exact ordered A/B/C/D"):
        FourArmProtocolPayloadV1.model_validate(
            {**payload.model_dump(mode="python"), "arms": reordered}
        )


def test_payload_rejects_arm_kind_substitution() -> None:
    payload = build_four_arm_protocol_declaration_v1().payload
    altered = (
        FourArmDefinitionV1(arm_id="A", arm_kind="base_llm"),
        FourArmDefinitionV1(arm_id="B", arm_kind="llm_modules_no_workspace"),
        FourArmDefinitionV1(
            arm_id="C",
            arm_kind="llm_modules_limited_competitive_workspace",
        ),
        payload.arms[3],
    )

    with pytest.raises(ValidationError, match="exact ordered A/B/C/D"):
        FourArmProtocolPayloadV1.model_validate(
            {**payload.model_dump(mode="python"), "arms": altered}
        )


def test_payload_rejects_reordered_task_families() -> None:
    payload = build_four_arm_protocol_declaration_v1().payload
    reordered = (payload.task_families[1], payload.task_families[0], *payload.task_families[2:])

    with pytest.raises(ValidationError, match="task families"):
        FourArmProtocolPayloadV1.model_validate(
            {**payload.model_dump(mode="python"), "task_families": reordered}
        )


def test_payload_rejects_removed_unresolved_item() -> None:
    payload = build_four_arm_protocol_declaration_v1().payload
    duplicated = (*payload.unresolved_items[:-1], payload.unresolved_items[-2])

    with pytest.raises(ValidationError, match="unresolved items"):
        FourArmProtocolPayloadV1.model_validate(
            {**payload.model_dump(mode="python"), "unresolved_items": duplicated}
        )


def test_envelope_rejects_unknown_fields_and_non_hexadecimal_hash() -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    raw = declaration.model_dump(mode="python")

    with pytest.raises(ValidationError, match="Extra inputs"):
        IrisFourArmProtocolDeclarationV1.model_validate({**raw, "unknown": 1})
    with pytest.raises(ValidationError, match="lower-case hexadecimal"):
        IrisFourArmProtocolDeclarationV1.model_validate({**raw, "payload_sha256": "G" * 64})
