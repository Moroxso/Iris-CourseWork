"""Tests for the canonical operational policy that resolves arm C."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import iris.shared_buffer as shared_buffer
from iris.shared_buffer import (
    IrisSharedBufferPolicyV1,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.utils.hashing import canonical_json_file_bytes

_POLICY_PATH = (
    Path(__file__).parents[2] / "docs" / "protocols" / "iris-gw-0.1-arm-c-buffer-policy.v1.json"
)
_POLICY_SHA256 = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"


def test_policy_is_exact_local_and_protocol_bound() -> None:
    policy = build_shared_buffer_policy_v1()

    assert policy.schema_version == 1
    assert policy.decision_scope == "local_experimental_decision_not_recovered"
    assert policy.cognitive_cycles == 3
    assert policy.native_state_fields == ("entries", "source_module", "write_cycle")
    assert policy.pre_update_read_count == 3
    assert policy.pre_update_entry_multipliers == (0, 1, 2)
    assert policy.final_output_read_count == 1
    assert policy.latent_rule == "one_final_read_token_per_module"
    assert policy.recovered_protocol_payload_sha256 == (
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    )


def test_policy_hash_is_deterministic_domain_separated_and_canonical() -> None:
    first = build_shared_buffer_policy_v1()
    second = build_shared_buffer_policy_v1()

    assert first == second
    assert first is not second
    assert shared_buffer_policy_sha256_v1() == _POLICY_SHA256
    assert shared_buffer_policy_sha256_v1() != first.recovered_protocol_payload_sha256
    assert _POLICY_PATH.read_bytes() == canonical_json_file_bytes(first.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("cognitive_cycles", 4),
        ("pre_update_read_count", 4),
        ("pre_update_entry_multipliers", (0, 1, 3)),
        ("latent_rule", "one_token_per_entry"),
        ("native_state_fields", ("entries", "confidence", "ttl")),
    ],
)
def test_policy_rejects_semantic_substitution(field: str, value: object) -> None:
    payload = build_shared_buffer_policy_v1().model_dump(mode="python")
    payload[field] = value
    with pytest.raises(ValidationError):
        IrisSharedBufferPolicyV1.model_validate(payload)


def test_policy_rejects_coercion_and_extra_fields() -> None:
    payload = build_shared_buffer_policy_v1().model_dump(mode="python")
    payload["cognitive_cycles"] = "3"
    with pytest.raises(ValidationError):
        IrisSharedBufferPolicyV1.model_validate(payload)
    payload = build_shared_buffer_policy_v1().model_dump(mode="python")
    payload["capacity"] = 24
    with pytest.raises(ValidationError):
        IrisSharedBufferPolicyV1.model_validate(payload)


def test_policy_api_is_public_from_shared_buffer_package() -> None:
    assert shared_buffer.IrisSharedBufferPolicyV1 is IrisSharedBufferPolicyV1
    assert shared_buffer.build_shared_buffer_policy_v1 is build_shared_buffer_policy_v1
    assert shared_buffer.shared_buffer_policy_sha256_v1 is shared_buffer_policy_sha256_v1
