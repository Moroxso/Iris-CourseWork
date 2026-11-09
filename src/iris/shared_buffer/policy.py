"""Content-addressed operational policy for the Iris-GW-0.1 arm-C buffer."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from iris.utils.hashing import canonical_json_bytes, domain_sha256

_POLICY_DOMAIN = "iris/shared-buffer-policy/v1"
_PROTOCOL_PAYLOAD_SHA256: Literal[
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
] = "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"


class IrisSharedBufferPolicyV1(BaseModel):
    """Exact local semantics that resolve the recovered arm-C source gap."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1]
    policy_id: Literal["iris-gw-0.1-arm-c-unlimited-shared-buffer"]
    recovered_protocol_payload_sha256: Literal[
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    ]
    decision_scope: Literal["local_experimental_decision_not_recovered"]
    cognitive_cycles: Literal[3]
    native_state_fields: tuple[
        Literal["entries"],
        Literal["source_module"],
        Literal["write_cycle"],
    ]
    entries_per_cycle_rule: Literal["one_transformed_proposal_per_module"]
    finite_horizon_entry_formula: Literal["num_modules_times_cognitive_cycles"]
    capacity_rule: Literal["no_capacity_constraint_within_finite_horizon"]
    visibility_rule: Literal["prior_completed_cycles_only"]
    ordering_rule: Literal["cycle_major_then_module_major"]
    write_rule: Literal["d_empty_slot_gated_transform_then_append_all"]
    lifecycle_rule: Literal["no_overwrite_eviction_decay_or_expiry"]
    pre_update_read_count: Literal[3]
    pre_update_entry_multipliers: tuple[Literal[0], Literal[1], Literal[2]]
    final_output_read_count: Literal[1]
    final_read_rule: Literal["same_reader_all_entries_one_query_per_module"]
    latent_rule: Literal["one_final_read_token_per_module"]
    proposal_fields_without_buffer_consumer: tuple[
        Literal["key"],
        Literal["bid"],
        Literal["confidence"],
        Literal["ttl"],
    ]

    @model_validator(mode="after")
    def validate_exact_policy(self) -> Self:
        if self.pre_update_read_count != self.cognitive_cycles:
            message = "Arm-C pre-update reads must equal the cognitive-cycle count"
            raise ValueError(message)
        if self.pre_update_entry_multipliers != tuple(range(self.cognitive_cycles)):
            message = "Arm-C pre-update entry multipliers must be 0, 1, and 2"
            raise ValueError(message)
        return self


def build_shared_buffer_policy_v1() -> IrisSharedBufferPolicyV1:
    """Build one owned exact policy for the finite Iris-GW-0.1 horizon."""
    return IrisSharedBufferPolicyV1(
        schema_version=1,
        policy_id="iris-gw-0.1-arm-c-unlimited-shared-buffer",
        recovered_protocol_payload_sha256=_PROTOCOL_PAYLOAD_SHA256,
        decision_scope="local_experimental_decision_not_recovered",
        cognitive_cycles=3,
        native_state_fields=("entries", "source_module", "write_cycle"),
        entries_per_cycle_rule="one_transformed_proposal_per_module",
        finite_horizon_entry_formula="num_modules_times_cognitive_cycles",
        capacity_rule="no_capacity_constraint_within_finite_horizon",
        visibility_rule="prior_completed_cycles_only",
        ordering_rule="cycle_major_then_module_major",
        write_rule="d_empty_slot_gated_transform_then_append_all",
        lifecycle_rule="no_overwrite_eviction_decay_or_expiry",
        pre_update_read_count=3,
        pre_update_entry_multipliers=(0, 1, 2),
        final_output_read_count=1,
        final_read_rule="same_reader_all_entries_one_query_per_module",
        latent_rule="one_final_read_token_per_module",
        proposal_fields_without_buffer_consumer=("key", "bid", "confidence", "ttl"),
    )


def shared_buffer_policy_sha256_v1() -> str:
    """Return the domain-separated identity of the exact arm-C policy."""
    canonical = canonical_json_bytes(build_shared_buffer_policy_v1().model_dump(mode="json"))
    return domain_sha256(_POLICY_DOMAIN, canonical)


__all__ = [
    "IrisSharedBufferPolicyV1",
    "build_shared_buffer_policy_v1",
    "shared_buffer_policy_sha256_v1",
]
