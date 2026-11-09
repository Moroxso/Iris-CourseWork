from typing import Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    DecodingProvenanceV1,
    GenerationProvenanceV1,
    IrisRequestArtifactEnvelopeV1,
    IrisRequestArtifactPayloadV1,
    PrivateTextFingerprintV1,
    RequestProvenanceV1,
    RuntimeProvenanceV1,
    SchemaIdentityV1,
    SoftwareProvenanceV1,
    TensorFingerprintV1,
    WorkspaceCycleProvenanceV1,
    WorkspaceTraceProvenanceV1,
)
from iris.schemas import IrisBootstrapConfig

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def valid_bootstrap_config() -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "iris", "version": "0.1.0"},
            "hardware": {
                "name": "cpu_smoke",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 24.0,
                "allow_gpu": False,
            },
            "backbone": {
                "name": "qwen3_0_6b",
                "repository": "Qwen/Qwen3-0.6B",
                "revision": "1" * 40,
                "tokenizer_revision": "2" * 40,
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
                "limits": {"max_prompt_tokens": 256, "max_new_tokens": 32},
            },
            "cognitive": {
                "name": "iris_gw_0_1",
                "state_dim": 256,
                "num_modules": 8,
                "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
            },
            "workspace": {
                "name": "iris_gw_0_1",
                "num_slots": 8,
                "cycles": 3,
                "winners_per_cycle": 2,
                "decay_rate": 0.85,
                "empty_threshold": 0.001,
                "arbiter": {
                    "confidence_weight": 0.25,
                    "novelty_weight": 0.10,
                    "relevance_weight": 0.25,
                },
                "allocator": {
                    "empty_bonus": 1.0,
                    "allocation_age_penalty": 0.10,
                    "retention_age_penalty": 0.10,
                },
            },
            "runtime": {
                "seed": 42,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": "artifacts/runs",
            },
        }
    )


def private_text(*, sha256: str = SHA_A, byte_length: int = 6) -> PrivateTextFingerprintV1:
    return PrivateTextFingerprintV1(
        sha256=sha256,
        representation="sha256_only",
        encoding="utf-8",
        normalization="none",
        byte_length=byte_length,
    )


def tensor(
    *,
    sha256: str = SHA_B,
    shape: tuple[int, ...] = (1, 8),
    dtype: Literal["float16", "bfloat16", "float32", "int64", "bool"] = "float32",
    byte_length: int | None = None,
) -> TensorFingerprintV1:
    if byte_length is None:
        element_size = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int64": 8,
            "bool": 1,
        }[dtype]
        byte_length = element_size
        for dimension in shape:
            byte_length *= dimension
    return TensorFingerprintV1(
        sha256=sha256,
        shape=shape,
        dtype=dtype,
        device="cpu",
        byte_length=byte_length,
    )


def decoding(
    *,
    strategy: Literal["greedy", "sampling"] = "greedy",
    stop_token_ids: tuple[int, ...] = (2, 7),
    temperature: float = 0.8,
    top_p: float = 0.9,
    sampling_seed: int | None = None,
) -> DecodingProvenanceV1:
    return DecodingProvenanceV1(
        strategy=strategy,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        top_p=top_p,
        sampling_seed=sampling_seed,
        allow_download=False,
    )


def workspace_cycle(cycle: int) -> WorkspaceCycleProvenanceV1:
    return WorkspaceCycleProvenanceV1(
        sha256=SHA_C,
        cycle=cycle,
        slots=tensor(shape=(1, 8, 256)),
        age=tensor(shape=(1, 8), dtype="int64"),
        occupancy=tensor(shape=(1, 8), dtype="bool", byte_length=8),
        source_module=tensor(shape=(1, 8), dtype="int64"),
        confidence=tensor(shape=(1, 8)),
        ttl=tensor(shape=(1, 8)),
        write_count=tensor(shape=(1, 8), dtype="int64"),
    )


def workspace_trace() -> WorkspaceTraceProvenanceV1:
    return WorkspaceTraceProvenanceV1(
        sha256=SHA_A,
        cycle_count=3,
        cycles=(workspace_cycle(1), workspace_cycle(2), workspace_cycle(3)),
    )


def software() -> SoftwareProvenanceV1:
    return SoftwareProvenanceV1(
        iris="0.1.0",
        python="3.12.11",
        torch="2.13.0",
        transformers="5.14.1",
        tokenizers="0.22.2",
        safetensors="0.8.0",
        hydra_core="1.3.5",
        omegaconf="2.3.1",
    )


def valid_payload() -> IrisRequestArtifactPayloadV1:
    return IrisRequestArtifactPayloadV1(
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
        runtime=RuntimeProvenanceV1(
            initialization_seed=42,
            deterministic=True,
            backend="cpu",
            device="cpu",
            device_name="CPU",
            dtype="float32",
            sidecar_state_sha256=SHA_B,
            software=software(),
        ),
        generation=GenerationProvenanceV1(
            text=private_text(sha256=SHA_C, byte_length=9),
            token_ids=tensor(shape=(1, 3), dtype="int64"),
            generated_tokens=3,
            stop_reason="stop_token",
            matched_stop_token_id=2,
        ),
        workspace_trace=workspace_trace(),
    )


def valid_envelope() -> IrisRequestArtifactEnvelopeV1:
    return IrisRequestArtifactEnvelopeV1(
        schema=SchemaIdentityV1(
            artifact_type="iris.request",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
        ),
        payload_sha256=SHA_C,
        payload=valid_payload(),
    )


def dumped(model: BaseModel) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="python", by_alias=True))


def test_valid_envelope_has_exact_version_identity_and_canonical_alias() -> None:
    artifact = valid_envelope()

    serialized = artifact.model_dump(mode="json", by_alias=True)

    assert artifact.schema_identity.version == 1
    assert serialized["schema"] == {
        "artifact_type": "iris.request",
        "version": 1,
        "canonicalization": "iris-json-v1",
        "hash_algorithm": "sha256",
    }
    assert "schema_identity" not in serialized
    assert (
        IrisRequestArtifactEnvelopeV1.model_validate_json(artifact.model_dump_json(by_alias=True))
        == artifact
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_type", "iris.response"),
        ("version", 2),
        ("version", "1"),
        ("version", True),
        ("canonicalization", "json"),
        ("hash_algorithm", "sha512"),
    ],
)
def test_schema_identity_rejects_wrong_or_future_contract(field: str, value: object) -> None:
    values: dict[str, object] = {
        "artifact_type": "iris.request",
        "version": 1,
        "canonicalization": "iris-json-v1",
        "hash_algorithm": "sha256",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        SchemaIdentityV1.model_validate(values)


@pytest.mark.parametrize(
    "field",
    ["artifact_type", "version", "canonicalization", "hash_algorithm"],
)
def test_schema_identity_requires_every_field(field: str) -> None:
    values: dict[str, object] = {
        "artifact_type": "iris.request",
        "version": 1,
        "canonicalization": "iris-json-v1",
        "hash_algorithm": "sha256",
    }
    del values[field]

    with pytest.raises(ValidationError, match="Field required"):
        SchemaIdentityV1.model_validate(values)


@pytest.mark.parametrize("field", ["schema", "payload_sha256", "payload"])
def test_envelope_requires_every_top_level_field(field: str) -> None:
    values = dumped(valid_envelope())
    del values[field]

    with pytest.raises(ValidationError, match="Field required"):
        IrisRequestArtifactEnvelopeV1.model_validate(values)


@pytest.mark.parametrize("level", ["envelope", "schema", "payload"])
def test_artifact_rejects_extra_fields_at_every_level(level: str) -> None:
    values = dumped(valid_envelope())
    if level == "envelope":
        values["future"] = True
    else:
        nested = cast(dict[str, object], values[level])
        nested["future"] = True

    with pytest.raises(ValidationError, match="extra_forbidden"):
        IrisRequestArtifactEnvelopeV1.model_validate(values)


@pytest.mark.parametrize(
    "invalid_hash",
    ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 1, None],
)
def test_sha256_is_exact_lower_case_full_digest(invalid_hash: object) -> None:
    values = dumped(private_text())
    values["sha256"] = invalid_hash

    with pytest.raises(ValidationError, match="64 lower-case hexadecimal"):
        PrivateTextFingerprintV1.model_validate(values)


@pytest.mark.parametrize(
    ("model_type", "values"),
    [
        (PrivateTextFingerprintV1, dumped(private_text())),
        (TensorFingerprintV1, dumped(tensor())),
        (
            ConfigurationProvenanceV1,
            dumped(ConfigurationProvenanceV1(sha256=SHA_A, value=valid_bootstrap_config())),
        ),
        (WorkspaceCycleProvenanceV1, dumped(workspace_cycle(1))),
        (WorkspaceTraceProvenanceV1, dumped(workspace_trace())),
    ],
)
def test_every_shared_sha256_record_enforces_digest_contract(
    model_type: type[BaseModel],
    values: dict[str, object],
) -> None:
    values["sha256"] = "A" * 64

    with pytest.raises(ValidationError, match="64 lower-case hexadecimal"):
        model_type.model_validate(values)


@pytest.mark.parametrize("field", ["sidecar_state_sha256", "payload_sha256"])
@pytest.mark.parametrize("invalid_hash", ["f" * 63, "F" * 64, "z" * 64, False])
def test_named_hash_fields_enforce_digest_contract(field: str, invalid_hash: object) -> None:
    if field == "sidecar_state_sha256":
        values = dumped(valid_payload().runtime)
        model_type: type[BaseModel] = RuntimeProvenanceV1
    else:
        values = dumped(valid_envelope())
        model_type = IrisRequestArtifactEnvelopeV1
    values[field] = invalid_hash

    with pytest.raises(ValidationError, match="SHA-256"):
        model_type.model_validate(values)


def test_artifact_models_are_frozen() -> None:
    identity = valid_envelope().schema_identity

    with pytest.raises(ValidationError, match="frozen_instance"):
        identity.version = 1


def test_nested_artifact_instances_are_revalidated() -> None:
    identity = valid_envelope().schema_identity
    object.__setattr__(identity, "version", 2)

    with pytest.raises(ValidationError):
        IrisRequestArtifactEnvelopeV1(
            schema=identity,
            payload_sha256=SHA_A,
            payload=valid_payload(),
        )


def test_nested_bootstrap_configuration_is_revalidated() -> None:
    config = valid_bootstrap_config()
    object.__setattr__(config.backbone, "revision", "main")

    with pytest.raises(ValidationError, match="immutable 40-hex"):
        ConfigurationProvenanceV1(sha256=SHA_A, value=config)


@pytest.mark.parametrize(
    ("strategy", "sampling_seed"),
    [("greedy", None), ("sampling", 0), ("sampling", 2**63 - 1)],
)
def test_decoding_accepts_only_strategy_appropriate_seed_policy(
    strategy: Literal["greedy", "sampling"],
    sampling_seed: int | None,
) -> None:
    provenance = decoding(strategy=strategy, sampling_seed=sampling_seed)

    assert provenance.strategy == strategy
    assert provenance.sampling_seed == sampling_seed


@pytest.mark.parametrize(
    ("strategy", "sampling_seed"),
    [
        ("greedy", 0),
        ("sampling", None),
        ("sampling", -1),
        ("sampling", 2**63),
        ("sampling", True),
        ("sampling", "7"),
    ],
)
def test_decoding_rejects_missing_forbidden_or_invalid_sampling_seed(
    strategy: str,
    sampling_seed: object,
) -> None:
    values = dumped(decoding())
    values["strategy"] = strategy
    values["sampling_seed"] = sampling_seed

    with pytest.raises(ValidationError):
        DecodingProvenanceV1.model_validate(values)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_decoding_rejects_nonpositive_or_nonfinite_temperature(temperature: float) -> None:
    values = dumped(decoding())
    values["temperature"] = temperature

    with pytest.raises(ValidationError):
        DecodingProvenanceV1.model_validate(values)


@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_decoding_rejects_out_of_range_or_nonfinite_top_p(top_p: float) -> None:
    values = dumped(decoding())
    values["top_p"] = top_p

    with pytest.raises(ValidationError):
        DecodingProvenanceV1.model_validate(values)


@pytest.mark.parametrize(
    "stop_token_ids",
    [(), (2, 2), (-1,), (True,), ("2",), [2]],
)
def test_decoding_rejects_invalid_stop_token_sequence(stop_token_ids: object) -> None:
    values = dumped(decoding())
    values["stop_token_ids"] = stop_token_ids

    with pytest.raises(ValidationError):
        DecodingProvenanceV1.model_validate(values)


def test_decoding_preserves_caller_stop_token_order() -> None:
    provenance = decoding(stop_token_ids=(99, 2, 17))

    assert provenance.stop_token_ids == (99, 2, 17)


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "int64", "bool"],
)
def test_tensor_fingerprint_accepts_each_canonical_dtype(
    dtype: Literal["float16", "bfloat16", "float32", "int64", "bool"],
) -> None:
    fingerprint = tensor(shape=(0, 2, 3), dtype=dtype, byte_length=0)

    assert fingerprint.shape == (0, 2, 3)
    assert fingerprint.dtype == dtype


@pytest.mark.parametrize("dtype", ["float64", "int32", "torch.float32", 1])
def test_tensor_fingerprint_rejects_noncanonical_dtype(dtype: object) -> None:
    values = dumped(tensor())
    values["dtype"] = dtype

    with pytest.raises(ValidationError):
        TensorFingerprintV1.model_validate(values)


@pytest.mark.parametrize("shape", [(), (-1,), (True,), ("1",), [1, 2]])
def test_tensor_fingerprint_rejects_empty_ambiguous_or_negative_shapes(shape: object) -> None:
    values = dumped(tensor())
    values["shape"] = shape

    with pytest.raises(ValidationError):
        TensorFingerprintV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [("byte_length", -1), ("byte_length", True), ("device", "")],
)
def test_tensor_fingerprint_rejects_invalid_storage_metadata(field: str, value: object) -> None:
    values = dumped(tensor())
    values[field] = value

    with pytest.raises(ValidationError):
        TensorFingerprintV1.model_validate(values)


def test_workspace_trace_accepts_contiguous_one_indexed_cycles() -> None:
    trace = workspace_trace()

    assert trace.cycle_count == 3
    assert tuple(record.cycle for record in trace.cycles) == (1, 2, 3)


@pytest.mark.parametrize(
    ("cycle_count", "cycle_numbers"),
    [
        (2, (1,)),
        (1, (1, 2)),
        (2, (0, 1)),
        (2, (1, 3)),
        (2, (2, 1)),
        (2, (1, 1)),
        (3, (1, 3, 2)),
    ],
)
def test_workspace_trace_rejects_count_mismatch_or_noncontiguous_cycles(
    cycle_count: int,
    cycle_numbers: tuple[int, ...],
) -> None:
    cycle_values: list[dict[str, object]] = []
    for cycle_number in cycle_numbers:
        cycle_value = dumped(workspace_cycle(1))
        cycle_value["cycle"] = cycle_number
        cycle_values.append(cycle_value)
    values: dict[str, object] = {
        "sha256": SHA_A,
        "cycle_count": cycle_count,
        "cycles": tuple(cycle_values),
    }

    with pytest.raises(ValidationError):
        WorkspaceTraceProvenanceV1.model_validate(values)


def test_workspace_trace_revalidates_mutated_cycle_records() -> None:
    first = workspace_cycle(1)
    object.__setattr__(first, "cycle", 2)

    with pytest.raises(ValidationError, match="contiguous and one-indexed"):
        WorkspaceTraceProvenanceV1(
            sha256=SHA_A,
            cycle_count=2,
            cycles=(first, workspace_cycle(2)),
        )
