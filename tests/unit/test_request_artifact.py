"""Contract tests for building privacy-preserving request provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from iris.application.bootstrap import IrisRequestRuntime
from iris.application.errors import IrisRequestArtifactError
from iris.application.request import IrisRequestExecutionOutput, IrisRequestOrchestrator
from iris.application.request_artifact import build_request_artifact_v1
from iris.backbone import BackboneAdapter
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace import WorkspaceState

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (7, 2)
INITIALIZATION_SEED = 41
SAMPLING_SEED = 9_973
SIDECAR_STATE_SHA256 = "c" * 64
SECRET = "IRIS_SECRET_48f34c_Привет\r\n"


class ProvenanceOrchestrator(IrisRequestOrchestrator):
    _test_sidecar_sha256: str

    @property
    def sidecar_state_sha256(self) -> str:
        return self._test_sidecar_sha256


def config_for(output_dir: Path, *, local_files_only: bool = True) -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "iris", "version": "0.1.0"},
            "hardware": {
                "name": "artifact_cpu",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 1.0,
                "allow_gpu": False,
            },
            "backbone": {
                "name": "offline",
                "repository": "offline/test",
                "revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
                "trust_remote_code": False,
                "local_files_only": local_files_only,
                "output_hidden_states": True,
                "use_cache": False,
                "dtype": "fp32",
                "tokenizer": {
                    "use_fast": True,
                    "padding_side": "left",
                    "truncation_side": "left",
                },
                "limits": {"max_prompt_tokens": 4, "max_new_tokens": 2},
            },
            "cognitive": {
                "name": "artifact",
                "state_dim": 3,
                "num_modules": 2,
                "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
            },
            "workspace": {
                "name": "artifact",
                "num_slots": 2,
                "cycles": 2,
                "winners_per_cycle": 1,
                "decay_rate": 0.85,
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
                "seed": INITIALIZATION_SEED,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": str(output_dir),
            },
        }
    )


def workspace_state(cycle: int) -> WorkspaceState:
    slots = torch.zeros((1, 2, 3), dtype=torch.float32)
    age = torch.zeros((1, 2), dtype=torch.int64)
    occupancy = torch.zeros((1, 2), dtype=torch.bool)
    source_module = torch.full((1, 2), -1, dtype=torch.int64)
    confidence = torch.zeros((1, 2), dtype=torch.float32)
    ttl = torch.zeros((1, 2), dtype=torch.float32)
    write_count = torch.zeros((1, 2), dtype=torch.int64)
    occupied = slice(0, cycle)
    slots[0, occupied] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])[:cycle]
    occupancy[0, occupied] = True
    source_module[0, occupied] = torch.arange(cycle, dtype=torch.int64)
    confidence[0, occupied] = torch.tensor([0.75, 0.5])[:cycle]
    ttl[0, occupied] = torch.tensor([2.0, 1.0])[:cycle]
    write_count[0, occupied] = 1
    return WorkspaceState(
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source_module,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
    )


def artifact_inputs(
    tmp_path: Path,
    *,
    strategy: str = "greedy",
    sampling_seed: int | None = None,
    secret: str = SECRET,
) -> dict[str, object]:
    config = config_for(tmp_path / "artifacts")
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode()
    orchestrator = object.__new__(ProvenanceOrchestrator)
    orchestrator._test_sidecar_sha256 = SIDECAR_STATE_SHA256
    runtime = IrisRequestRuntime(
        backbone=cast(BackboneAdapter, object()),
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Artifact Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=INITIALIZATION_SEED,
        deterministic=True,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256("iris/configuration/v1", configuration_json.encode()),
        sidecar_state_sha256=SIDECAR_STATE_SHA256,
    )
    output = IrisGeneratorOutput(
        text=secret,
        token_ids=torch.tensor([[5, 7]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=7,
        workspace_trace=(workspace_state(1), workspace_state(2)),
    )
    execution = IrisRequestExecutionOutput(
        generation=output,
        prompt_input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
    )
    return {
        "prompt": secret,
        "strategy": strategy,
        "stop_token_ids": STOP_TOKEN_IDS,
        "sampling": TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875),
        "sampling_seed": sampling_seed,
        "allow_download": False,
        "runtime": runtime,
        "execution": execution,
    }


def test_builder_records_exact_policy_and_strong_hash_only_privacy(tmp_path: Path) -> None:
    inputs = artifact_inputs(tmp_path)
    artifact = build_request_artifact_v1(**inputs)  # type: ignore[arg-type]
    payload = artifact.payload

    assert payload.configuration.value.backbone.local_files_only is True
    assert payload.configuration.value.runtime.seed == INITIALIZATION_SEED
    assert payload.request.decoding.stop_token_ids == STOP_TOKEN_IDS
    assert payload.request.decoding.temperature == 0.75
    assert payload.request.decoding.top_p == 0.875
    assert payload.request.decoding.sampling_seed is None
    assert payload.runtime.sidecar_state_sha256 == SIDECAR_STATE_SHA256
    assert payload.request.physical_prompt_tokens == 2
    assert payload.request.attended_prompt_tokens == 2
    assert payload.generation.generated_tokens == 2
    assert payload.workspace_trace.cycle_count == 2
    assert tuple(cycle.cycle for cycle in payload.workspace_trace.cycles) == (1, 2)

    raw = canonical_json_bytes(artifact.model_dump(mode="json", by_alias=True))
    secret_bytes = SECRET.encode()
    assert secret_bytes not in raw
    assert repr(SECRET).encode() not in raw
    parsed = artifact.model_dump(mode="json", by_alias=True)
    generation = cast(dict[str, object], cast(dict[str, object], parsed["payload"])["generation"])
    assert "text" in generation
    assert cast(dict[str, object], generation["text"])["representation"] == "sha256_only"
    assert "values" not in cast(dict[str, object], generation["token_ids"])


def test_builder_is_deterministic_and_sensitive_to_each_provenance_domain(
    tmp_path: Path,
) -> None:
    inputs = artifact_inputs(tmp_path)
    first = build_request_artifact_v1(**inputs)  # type: ignore[arg-type]
    second = build_request_artifact_v1(**inputs)  # type: ignore[arg-type]
    assert first == second

    execution = cast(IrisRequestExecutionOutput, inputs["execution"])
    changed_state = workspace_state(2)
    changed_state.slots[0, 0, 0] += 0.25
    changed_output = replace(
        execution.generation,
        workspace_trace=(workspace_state(1), changed_state),
    )
    changed_execution = replace(execution, generation=changed_output)
    changed = build_request_artifact_v1(
        **{**inputs, "execution": changed_execution}  # type: ignore[arg-type]
    )

    assert changed.payload.workspace_trace.sha256 != first.payload.workspace_trace.sha256
    assert changed.payload_sha256 != first.payload_sha256
    assert (
        changed.payload.workspace_trace.cycles[1].slots.sha256
        != first.payload.workspace_trace.cycles[1].slots.sha256
    )
    assert (
        changed.payload.workspace_trace.cycles[0].slots.sha256
        == first.payload.workspace_trace.cycles[0].slots.sha256
    )


def test_sampling_seed_is_distinct_from_initialization_seed_and_affects_payload(
    tmp_path: Path,
) -> None:
    first = build_request_artifact_v1(
        **artifact_inputs(tmp_path, strategy="sampling", sampling_seed=SAMPLING_SEED)  # type: ignore[arg-type]
    )
    second = build_request_artifact_v1(
        **artifact_inputs(tmp_path, strategy="sampling", sampling_seed=SAMPLING_SEED + 1)  # type: ignore[arg-type]
    )

    assert first.payload.runtime.initialization_seed == INITIALIZATION_SEED
    assert first.payload.request.decoding.sampling_seed == SAMPLING_SEED
    assert first.payload_sha256 != second.payload_sha256


def test_builder_rejects_sidecar_mutation_after_generation(tmp_path: Path) -> None:
    inputs = artifact_inputs(tmp_path)
    runtime = cast(IrisRequestRuntime, inputs["runtime"])
    cast(ProvenanceOrchestrator, runtime.orchestrator)._test_sidecar_sha256 = "d" * 64

    with pytest.raises(IrisRequestArtifactError, match="could not be built") as captured:
        build_request_artifact_v1(**inputs)  # type: ignore[arg-type]

    assert captured.value.__cause__ is not None
    assert "sidecar provenance changed" in str(captured.value.__cause__)


@pytest.mark.parametrize(
    "mutation",
    ["bad_trace_length", "noncanonical_empty", "invalid_stop", "bad_prompt_mask"],
)
def test_builder_rejects_corrupted_durable_boundaries(
    tmp_path: Path,
    mutation: str,
) -> None:
    inputs = artifact_inputs(tmp_path)
    execution = cast(IrisRequestExecutionOutput, inputs["execution"])
    if mutation == "bad_trace_length":
        output = replace(execution.generation, workspace_trace=(workspace_state(1),))
        inputs["execution"] = replace(execution, generation=output)
    elif mutation == "noncanonical_empty":
        state = workspace_state(1)
        state.slots[0, 1, 0] = 1.0
        output = replace(execution.generation, workspace_trace=(state, workspace_state(2)))
        inputs["execution"] = replace(execution, generation=output)
    elif mutation == "invalid_stop":
        output = replace(execution.generation, matched_stop_token_id=2)
        inputs["execution"] = replace(execution, generation=output)
    else:
        inputs["execution"] = replace(
            execution,
            prompt_attention_mask=torch.tensor([[1, 2]], dtype=torch.int64),
        )

    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        build_request_artifact_v1(**inputs)  # type: ignore[arg-type]


def test_prompt_fingerprint_uses_exact_utf8_without_normalization(tmp_path: Path) -> None:
    variants = ("é", "e\u0301", "x\n", "x\r\n", " x ")
    digests = {
        build_request_artifact_v1(
            **artifact_inputs(tmp_path, secret=variant)  # type: ignore[arg-type]
        ).payload.request.prompt.sha256
        for variant in variants
    }
    assert len(digests) == len(variants)


def test_unpaired_surrogate_is_rejected_without_filesystem_mutation(tmp_path: Path) -> None:
    inputs = artifact_inputs(tmp_path, secret="bad\ud800prompt")

    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        build_request_artifact_v1(**inputs)  # type: ignore[arg-type]

    assert list(tmp_path.iterdir()) == []
