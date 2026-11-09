"""Strict unit contracts for the Arm-C teacher checkpoint v3 identity."""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

import iris.application as application
import iris.artifacts as artifacts
from iris.application.shared_buffer_training_checkpoint import (
    build_shared_buffer_teacher_checkpoint_v3,
)
from iris.artifacts.checkpoint_v1 import TeacherBackboneRuntimeV1
from iris.artifacts.checkpoint_v3 import (
    SharedBufferTeacherCheckpointModelIdentityV3,
    SharedBufferTeacherCheckpointSchemaV3,
)
from tests.unit.test_teacher_checkpoint_schema import _backbone, _cognitive

SHA = "a" * 64
PROFILE_SET_V4_SHA256: Literal[
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
] = "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
POLICY_SHA256: Literal["56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"] = (
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
)
PROTOCOL_SHA256: Literal["b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"] = (
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
)
COMPONENT_NAMES = (
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


def _identity() -> SharedBufferTeacherCheckpointModelIdentityV3:
    return SharedBufferTeacherCheckpointModelIdentityV3(
        arm_id="C",
        arm_kind="llm_modules_unlimited_shared_buffer",
        teacher_graph=("append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"),
        execution_profile_set_sha256=PROFILE_SET_V4_SHA256,
        shared_buffer_policy_sha256=POLICY_SHA256,
        protocol_declaration_payload_sha256=PROTOCOL_SHA256,
        backbone=_backbone(),
        cognitive=_cognitive(),
        cognitive_cycles=3,
        pre_update_read_count=3,
        final_output_read_count=1,
        entries_per_cycle=3,
        final_entry_count=9,
        latent_token_count=3,
        component_names=COMPONENT_NAMES,
        runtime=TeacherBackboneRuntimeV1(
            hidden_size=6,
            num_hidden_layers=4,
            vocabulary_size=16,
            max_position_embeddings=11,
            max_prompt_tokens=5,
            max_target_tokens=3,
            device="cpu",
            dtype="float32",
        ),
        state_names=("weight",),
        sha256=SHA,
    )


def test_v3_schema_and_model_identity_are_exact_strict_and_frozen() -> None:
    schema = SharedBufferTeacherCheckpointSchemaV3(
        artifact_type="iris.teacher_checkpoint",
        version=3,
        canonicalization="iris-json-v1",
        tensor_encoding="safetensors-v1",
        hash_algorithm="sha256",
    )
    identity = _identity()

    assert (
        application.build_shared_buffer_teacher_checkpoint_v3
        is build_shared_buffer_teacher_checkpoint_v3
    )
    assert (
        artifacts.SharedBufferTeacherCheckpointSchemaV3
        is SharedBufferTeacherCheckpointSchemaV3
    )
    assert (
        artifacts.SharedBufferTeacherCheckpointModelIdentityV3
        is SharedBufferTeacherCheckpointModelIdentityV3
    )
    assert schema.version == 3
    assert identity.arm_id == "C"
    assert identity.execution_profile_set_sha256 == PROFILE_SET_V4_SHA256
    assert identity.shared_buffer_policy_sha256 == POLICY_SHA256
    assert identity.protocol_declaration_payload_sha256 == PROTOCOL_SHA256
    assert identity.final_entry_count == 3 * identity.entries_per_cycle
    assert identity.latent_token_count == identity.entries_per_cycle
    assert identity.component_names == COMPONENT_NAMES
    assert "workspace" not in identity.model_dump(mode="json")
    with pytest.raises(ValidationError):
        schema.version = 2  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        SharedBufferTeacherCheckpointSchemaV3.model_validate(
            {**schema.model_dump(mode="python"), "unknown": True}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_id", "B"),
        ("execution_profile_set_sha256", "0" * 64),
        ("shared_buffer_policy_sha256", "0" * 64),
        ("protocol_declaration_payload_sha256", "0" * 64),
        ("cognitive_cycles", 2),
        ("pre_update_read_count", 2),
        ("final_output_read_count", 2),
        ("entries_per_cycle", 2),
        ("final_entry_count", 8),
        ("latent_token_count", 9),
        ("component_names", tuple(reversed(COMPONENT_NAMES))),
    ],
)
def test_v3_identity_rejects_cross_arm_drift_and_noncanonical_lifecycle(
    field: str,
    value: object,
) -> None:
    values = _identity().model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError):
        SharedBufferTeacherCheckpointModelIdentityV3.model_validate(values)


@pytest.mark.parametrize("version", [True, "3", 3.0, 2, 4])
def test_v3_schema_rejects_nonexact_or_unsupported_version(version: object) -> None:
    with pytest.raises(ValidationError):
        SharedBufferTeacherCheckpointSchemaV3(
            artifact_type="iris.teacher_checkpoint",
            version=cast(Any, version),
            canonicalization="iris-json-v1",
            tensor_encoding="safetensors-v1",
            hash_algorithm="sha256",
        )
