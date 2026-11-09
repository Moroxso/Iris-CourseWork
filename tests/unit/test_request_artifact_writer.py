import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from iris.application import request_artifact
from iris.application.errors import IrisRequestArtifactError
from iris.artifacts import (
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
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)


def _configuration(output_dir: Path) -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "iris", "version": "0.1.0"},
            "hardware": {
                "name": "cpu_test",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 1.0,
                "allow_gpu": False,
            },
            "backbone": {
                "name": "qwen3_0_6b",
                "repository": "Qwen/Qwen3-0.6B",
                "revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
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
                "limits": {"max_prompt_tokens": 4, "max_new_tokens": 1},
            },
            "cognitive": {
                "name": "iris_gw_0_1",
                "state_dim": 1,
                "num_modules": 1,
                "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
            },
            "workspace": {
                "name": "iris_gw_0_1",
                "num_slots": 1,
                "cycles": 1,
                "winners_per_cycle": 1,
                "decay_rate": 0.5,
                "empty_threshold": 0.001,
                "arbiter": {
                    "confidence_weight": 0.25,
                    "novelty_weight": 0.1,
                    "relevance_weight": 0.25,
                },
                "allocator": {
                    "empty_bonus": 1.0,
                    "allocation_age_penalty": 0.1,
                    "retention_age_penalty": 0.1,
                },
            },
            "runtime": {
                "seed": 42,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": str(output_dir),
            },
        }
    )


def _tensor(
    *,
    shape: tuple[int, ...],
    dtype: str,
    byte_length: int,
    digest_digit: str,
) -> TensorFingerprintV1:
    return TensorFingerprintV1.model_validate(
        {
            "shape": shape,
            "dtype": dtype,
            "device": "cpu",
            "byte_length": byte_length,
            "sha256": digest_digit * 64,
        }
    )


def _minimal_artifact(output_dir: Path) -> IrisRequestArtifactEnvelopeV1:
    config = _configuration(output_dir)
    config_bytes = canonical_json_bytes(config.model_dump(mode="json"))
    prompt_ids = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="1")
    prompt_mask = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="2")
    generated_ids = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="3")
    slots = _tensor(shape=(1, 1, 1), dtype="float32", byte_length=4, digest_digit="4")
    age = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="5")
    occupancy = _tensor(shape=(1, 1), dtype="bool", byte_length=1, digest_digit="6")
    source_module = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="7")
    confidence = _tensor(shape=(1, 1), dtype="float32", byte_length=4, digest_digit="8")
    ttl = _tensor(shape=(1, 1), dtype="float32", byte_length=4, digest_digit="9")
    write_count = _tensor(shape=(1, 1), dtype="int64", byte_length=8, digest_digit="a")

    cycle = WorkspaceCycleProvenanceV1(
        cycle=1,
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source_module,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
        sha256="0" * 64,
    )
    cycle = WorkspaceCycleProvenanceV1.model_validate(
        {
            **cycle.model_dump(mode="python"),
            "sha256": domain_sha256(
                "iris/request-artifact/v1/workspace/cycle/1",
                canonical_json_bytes(request_artifact._cycle_content(cycle)),
            ),
        }
    )
    trace_content = [cycle.model_dump(mode="json")]
    trace = WorkspaceTraceProvenanceV1(
        cycle_count=1,
        cycles=(cycle,),
        sha256=domain_sha256(
            "iris/request-artifact/v1/workspace-trace",
            canonical_json_bytes(trace_content),
        ),
    )
    payload = IrisRequestArtifactPayloadV1(
        configuration=ConfigurationProvenanceV1(
            value=config,
            sha256=domain_sha256("iris/configuration/v1", config_bytes),
        ),
        request=RequestProvenanceV1(
            prompt=PrivateTextFingerprintV1(
                representation="sha256_only",
                encoding="utf-8",
                normalization="none",
                byte_length=6,
                sha256="b" * 64,
            ),
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            physical_prompt_tokens=1,
            attended_prompt_tokens=1,
            decoding=DecodingProvenanceV1(
                strategy="greedy",
                stop_token_ids=(2,),
                temperature=1.0,
                top_p=1.0,
                sampling_seed=None,
                allow_download=False,
            ),
        ),
        runtime=RuntimeProvenanceV1(
            initialization_seed=42,
            deterministic=True,
            backend="cpu",
            device="cpu",
            device_name="Test CPU",
            dtype="float32",
            sidecar_state_sha256="c" * 64,
            software=SoftwareProvenanceV1(
                iris="0.1.0",
                python="3.12.test",
                torch="test",
                transformers="test",
                tokenizers="test",
                safetensors="test",
                hydra_core="test",
                omegaconf="test",
            ),
        ),
        generation=GenerationProvenanceV1(
            text=PrivateTextFingerprintV1(
                representation="sha256_only",
                encoding="utf-8",
                normalization="none",
                byte_length=7,
                sha256="d" * 64,
            ),
            token_ids=generated_ids,
            generated_tokens=1,
            stop_reason="max_new_tokens",
            matched_stop_token_id=None,
        ),
        workspace_trace=trace,
    )
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    artifact = IrisRequestArtifactEnvelopeV1(
        schema=SchemaIdentityV1(
            artifact_type="iris.request",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
        ),
        payload_sha256=domain_sha256(
            "iris/request-artifact/v1/payload",
            payload_bytes,
        ),
        payload=payload,
    )
    request_artifact._validate_integrity(artifact)
    return artifact


@pytest.fixture
def artifact(tmp_path: Path) -> IrisRequestArtifactEnvelopeV1:
    return _minimal_artifact(tmp_path / "artifact-output")


def _canonical_content(artifact: IrisRequestArtifactEnvelopeV1) -> bytes:
    return request_artifact._validate_integrity(artifact)


def test_writer_uses_content_addressed_path_and_exact_canonical_file_bytes(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    result = request_artifact.write_request_artifact_v1(artifact)
    content = result.path.read_bytes()
    output_dir = Path(artifact.payload.configuration.value.runtime.output_dir)

    assert result.path == (output_dir / "requests" / "v1" / f"{artifact.payload_sha256}.json")
    assert content == _canonical_content(artifact)
    assert content.count(b"\n") == 1
    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert not content.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in content
    assert result.payload_sha256 == artifact.payload_sha256
    assert result.file_sha256 == bytes_sha256(content)
    assert result.size_bytes == len(content)
    assert result.reused_existing is False


def test_writer_reuses_an_exact_existing_content_addressed_artifact(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    first = request_artifact.write_request_artifact_v1(artifact)
    original = first.path.read_bytes()

    second = request_artifact.write_request_artifact_v1(artifact)

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert second.path == first.path
    assert second.payload_sha256 == first.payload_sha256
    assert second.file_sha256 == first.file_sha256
    assert second.size_bytes == first.size_bytes
    assert second.path.read_bytes() == original
    assert list(second.path.parent.glob(".iris-*.tmp")) == []


def test_writer_rejects_conflicting_existing_content_without_mutation(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    output_dir = Path(artifact.payload.configuration.value.runtime.output_dir)
    target = output_dir / "requests" / "v1" / f"{artifact.payload_sha256}.json"
    target.parent.mkdir(parents=True)
    conflicting = b'{"conflict":true}\n'
    target.write_bytes(conflicting)

    with pytest.raises(IrisRequestArtifactError, match="conflicts"):
        request_artifact.write_request_artifact_v1(artifact)

    assert target.read_bytes() == conflicting
    assert list(target.parent.glob(".iris-*.tmp")) == []


def test_writer_handles_concurrent_identical_install_as_reuse_and_cleans_temp(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    expected = _canonical_content(artifact)

    def install_competing_copy(source: Path, target: Path) -> None:
        assert source.read_bytes() == expected
        target.write_bytes(expected)
        raise FileExistsError

    with patch(
        "iris.application.request_artifact.os.link",
        side_effect=install_competing_copy,
    ):
        result = request_artifact.write_request_artifact_v1(artifact)

    assert result.reused_existing is True
    assert result.path.read_bytes() == expected
    assert list(result.path.parent.glob(".iris-*.tmp")) == []


def test_writer_cleans_temporary_file_after_atomic_install_failure(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    failure = OSError("forced hard-link failure")
    output_dir = Path(artifact.payload.configuration.value.runtime.output_dir)
    target = output_dir / "requests" / "v1" / f"{artifact.payload_sha256}.json"

    with (
        patch("iris.application.request_artifact.os.link", side_effect=failure),
        pytest.raises(
            IrisRequestArtifactError,
            match="could not be installed atomically",
        ) as captured,
    ):
        request_artifact.write_request_artifact_v1(artifact)

    assert captured.value.__cause__ is failure
    assert not target.exists()
    assert list(target.parent.glob(".iris-*.tmp")) == []


def test_reader_returns_the_exact_validated_model_from_canonical_bytes(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> None:
    result = request_artifact.write_request_artifact_v1(artifact)

    loaded = request_artifact.read_request_artifact_v1(result.path)

    assert type(loaded) is IrisRequestArtifactEnvelopeV1
    assert loaded is not artifact
    assert loaded == artifact
    assert _canonical_content(loaded) == result.path.read_bytes()


def test_reader_rejects_utf8_bom(artifact: IrisRequestArtifactEnvelopeV1, tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + _canonical_content(artifact))

    with pytest.raises(IrisRequestArtifactError) as captured:
        request_artifact.read_request_artifact_v1(path)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "exactly one final LF" in str(captured.value.__cause__)


def test_reader_rejects_duplicate_json_keys(
    artifact: IrisRequestArtifactEnvelopeV1,
    tmp_path: Path,
) -> None:
    canonical = _canonical_content(artifact)
    duplicated = canonical.replace(
        b'{"payload":',
        b'{"payload":null,"payload":',
        1,
    )
    assert duplicated != canonical
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicated)

    with pytest.raises(IrisRequestArtifactError) as captured:
        request_artifact.read_request_artifact_v1(path)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "duplicate JSON key is forbidden: payload" in str(captured.value.__cause__)


def test_reader_rejects_noncanonical_but_valid_json(
    artifact: IrisRequestArtifactEnvelopeV1,
    tmp_path: Path,
) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_bytes(b" " + _canonical_content(artifact))

    with pytest.raises(IrisRequestArtifactError) as captured:
        request_artifact.read_request_artifact_v1(path)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "not canonical iris-json-v1" in str(captured.value.__cause__)


def test_reader_rejects_canonical_payload_tampering(
    artifact: IrisRequestArtifactEnvelopeV1,
    tmp_path: Path,
) -> None:
    raw = cast(
        dict[str, Any],
        json.loads(_canonical_content(artifact).decode("utf-8")),
    )
    payload = cast(dict[str, Any], raw["payload"])
    runtime = cast(dict[str, Any], payload["runtime"])
    runtime["device_name"] = "Tampered CPU"
    path = tmp_path / "tampered.json"
    path.write_bytes(canonical_json_file_bytes(raw))

    with pytest.raises(IrisRequestArtifactError) as captured:
        request_artifact.read_request_artifact_v1(path)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "payload digest is inconsistent" in str(captured.value.__cause__)
