"""Contract tests for the arm-labelled persisted request schema v2."""

from __future__ import annotations

from typing import Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    GenerationProvenanceV1,
    RequestProvenanceV1,
)
from iris.artifacts.request_v2 import (
    EmptyWorkspaceTraceProvenanceV2,
    IrisRequestArtifactEnvelopeV2,
    IrisRequestArtifactPayloadV2,
    RequestArmATopologyV2,
    RequestArmBTopologyV2,
    RequestArmDTopologyV2,
    RequestArtifactSchemaV2,
    RequestRuntimeProvenanceV2,
    WorkspaceTraceProvenanceV2,
)
from tests.unit.test_request_artifact_schema import (
    decoding,
    private_text,
    software,
    tensor,
    valid_bootstrap_config,
    workspace_cycle,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
PROTOCOL_SHA256: Literal["b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"] = (
    "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
)
PROFILE_SHA256: Literal["5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"] = (
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
)
ARM_B_COMPONENTS = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "prefix_assembler",
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
    "prefix_assembler",
)


def _topology(
    arm_id: str,
) -> RequestArmATopologyV2 | RequestArmBTopologyV2 | RequestArmDTopologyV2:
    if arm_id == "A":
        return RequestArmATopologyV2(
            arm_id="A",
            arm_kind="base_llm",
            request_graph="frozen_backbone_prompt_only_v1",
            memory_model="none",
            trace_contract="empty_workspace_trace",
            cognitive_cycles=0,
            latent_token_count=0,
            zero_memory_read_cycles=0,
            sidecar_component_names=(),
            sidecar_state_sha256=None,
        )
    if arm_id == "B":
        return RequestArmBTopologyV2(
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            request_graph="independent_recurrent_modules_zero_memory_read_v1",
            memory_model="none",
            trace_contract="empty_workspace_trace",
            cognitive_cycles=3,
            latent_token_count=8,
            zero_memory_read_cycles=3,
            sidecar_component_names=ARM_B_COMPONENTS,
            sidecar_state_sha256=SHA_B,
        )
    return RequestArmDTopologyV2(
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        request_graph="limited_competitive_workspace_v1",
        memory_model="limited_competitive_workspace",
        trace_contract="workspace_state_per_cycle",
        cognitive_cycles=3,
        latent_token_count=8,
        zero_memory_read_cycles=0,
        sidecar_component_names=ARM_D_COMPONENTS,
        sidecar_state_sha256=SHA_B,
    )


def valid_payload(arm_id: str = "D") -> IrisRequestArtifactPayloadV2:
    sidecar = None if arm_id == "A" else SHA_B
    trace: EmptyWorkspaceTraceProvenanceV2 | WorkspaceTraceProvenanceV2
    if arm_id == "D":
        trace = WorkspaceTraceProvenanceV2(
            trace_contract="workspace_state_per_cycle",
            cycle_count=3,
            cycles=(workspace_cycle(1), workspace_cycle(2), workspace_cycle(3)),
            sha256=SHA_A,
        )
    else:
        trace = EmptyWorkspaceTraceProvenanceV2(
            trace_contract="empty_workspace_trace",
            cycle_count=0,
            cycles=(),
            sha256=SHA_A,
        )
    return IrisRequestArtifactPayloadV2(
        protocol_declaration_payload_sha256=PROTOCOL_SHA256,
        execution_profile_set_version=3,
        execution_profile_set_sha256=PROFILE_SHA256,
        topology=_topology(arm_id),
        configuration=ConfigurationProvenanceV1(
            sha256=SHA_A,
            value=valid_bootstrap_config(),
        ),
        request=RequestProvenanceV1(
            prompt=private_text(),
            prompt_input_ids=tensor(shape=(1, 4), dtype="int64"),
            prompt_attention_mask=tensor(shape=(1, 4), dtype="int64"),
            physical_prompt_tokens=4,
            attended_prompt_tokens=4,
            decoding=decoding(),
        ),
        runtime=RequestRuntimeProvenanceV2(
            initialization_seed=42,
            deterministic=True,
            backend="cpu",
            device="cpu",
            device_name="CPU",
            dtype="float32",
            sidecar_state_sha256=sidecar,
            software=software(),
        ),
        generation=GenerationProvenanceV1(
            text=private_text(sha256=SHA_C, byte_length=9),
            token_ids=tensor(shape=(1, 3), dtype="int64"),
            generated_tokens=3,
            stop_reason="stop_token",
            matched_stop_token_id=2,
        ),
        workspace_trace=trace,
    )


def valid_envelope(arm_id: str = "D") -> IrisRequestArtifactEnvelopeV2:
    return IrisRequestArtifactEnvelopeV2(
        schema=RequestArtifactSchemaV2(
            artifact_type="iris.request",
            version=2,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            content_policy="private_text_sha256_only",
        ),
        payload_sha256=SHA_C,
        payload=valid_payload(arm_id),
    )


def dumped(model: BaseModel) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="python", by_alias=True))


@pytest.mark.parametrize("arm_id", ["A", "B", "D"])
def test_each_executable_arm_round_trips_with_distinct_topology(arm_id: str) -> None:
    artifact = valid_envelope(arm_id)

    restored = IrisRequestArtifactEnvelopeV2.model_validate_json(
        artifact.model_dump_json(by_alias=True)
    )

    assert restored == artifact
    assert restored.schema_identity.version == 2
    assert restored.payload.topology.arm_id == arm_id
    assert restored.payload.execution_profile_set_version == 3
    assert restored.payload.execution_profile_set_sha256 == PROFILE_SHA256
    if arm_id == "A":
        assert restored.payload.runtime.sidecar_state_sha256 is None
    elif arm_id == "B":
        assert restored.payload.topology.zero_memory_read_cycles == 3
        assert restored.payload.workspace_trace.cycle_count == 0
    else:
        assert restored.payload.workspace_trace.cycle_count == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_declaration_payload_sha256", "0" * 64),
        ("execution_profile_set_version", 2),
        ("execution_profile_set_version", "3"),
        ("execution_profile_set_sha256", "0" * 64),
    ],
)
def test_profile_and_protocol_binding_is_exact(field: str, value: object) -> None:
    raw = dumped(valid_payload("B"))
    raw[field] = value

    with pytest.raises(ValidationError):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_arm_a_rejects_a_synthetic_sidecar() -> None:
    raw = dumped(valid_payload("A"))
    runtime = cast(dict[str, object], raw["runtime"])
    runtime["sidecar_state_sha256"] = SHA_B

    with pytest.raises(ValidationError, match="no sidecar"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_arm_b_rejects_cross_arm_workspace_trace() -> None:
    raw = dumped(valid_payload("B"))
    raw["workspace_trace"] = dumped(valid_payload("D"))["workspace_trace"]

    with pytest.raises(ValidationError):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_arm_b_rejects_wrong_module_latent_cardinality() -> None:
    raw = dumped(valid_payload("B"))
    topology = cast(dict[str, object], raw["topology"])
    topology["latent_token_count"] = 7

    with pytest.raises(ValidationError, match="arm B topology"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_arm_d_rejects_empty_trace_even_though_a_and_b_allow_it() -> None:
    raw = dumped(valid_payload("D"))
    raw["workspace_trace"] = dumped(valid_payload("A"))["workspace_trace"]

    with pytest.raises(ValidationError):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_schema_rejects_stop_token_outside_the_declared_decoding_policy() -> None:
    raw = dumped(valid_payload("A"))
    generation = cast(dict[str, object], raw["generation"])
    generation["matched_stop_token_id"] = 999

    with pytest.raises(ValidationError, match="absent from the decoding policy"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_schema_rejects_runtime_device_string_that_disagrees_with_backend() -> None:
    raw = dumped(valid_payload("B"))
    runtime = cast(dict[str, object], raw["runtime"])
    runtime["device"] = "cuda:0"
    request = cast(dict[str, object], raw["request"])
    cast(dict[str, object], request["prompt_input_ids"])["device"] = "cuda:0"
    cast(dict[str, object], request["prompt_attention_mask"])["device"] = "cuda:0"
    generation = cast(dict[str, object], raw["generation"])
    cast(dict[str, object], generation["token_ids"])["device"] = "cuda:0"

    with pytest.raises(ValidationError, match="resolved backend"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_schema_rejects_noncanonical_cuda_device_index() -> None:
    raw = dumped(valid_payload("D"))
    configuration = cast(dict[str, object], raw["configuration"])
    config_value = cast(dict[str, object], configuration["value"])
    hardware = cast(dict[str, object], config_value["hardware"])
    hardware["device"] = "cuda"
    hardware["allow_gpu"] = True
    runtime = cast(dict[str, object], raw["runtime"])
    runtime["backend"] = "cuda"
    runtime["device"] = "cuda:00"
    request = cast(dict[str, object], raw["request"])
    cast(dict[str, object], request["prompt_input_ids"])["device"] = "cuda:00"
    cast(dict[str, object], request["prompt_attention_mask"])["device"] = "cuda:00"
    generation = cast(dict[str, object], raw["generation"])
    cast(dict[str, object], generation["token_ids"])["device"] = "cuda:00"
    trace = cast(dict[str, object], raw["workspace_trace"])
    for cycle in cast(list[dict[str, object]], trace["cycles"]):
        for name in (
            "slots",
            "age",
            "occupancy",
            "source_module",
            "confidence",
            "ttl",
            "write_count",
        ):
            cast(dict[str, object], cycle[name])["device"] = "cuda:00"

    with pytest.raises(ValidationError, match="resolved backend"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_unresolved_arm_c_has_no_schema_variant() -> None:
    raw = dumped(valid_payload("B"))
    topology = cast(dict[str, object], raw["topology"])
    topology["arm_id"] = "C"

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        IrisRequestArtifactPayloadV2.model_validate(raw)


def test_schema_rejects_extra_fields_and_boolean_literal_counts() -> None:
    raw = dumped(valid_envelope("A"))
    raw["unexpected"] = "value"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IrisRequestArtifactEnvelopeV2.model_validate(raw)

    payload_raw = dumped(valid_payload("A"))
    topology = cast(dict[str, object], payload_raw["topology"])
    topology["cognitive_cycles"] = False
    with pytest.raises(ValidationError, match="exact integer"):
        IrisRequestArtifactPayloadV2.model_validate(payload_raw)
