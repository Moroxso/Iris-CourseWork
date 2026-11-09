"""Offline application integration for one request and its durable artifact."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import iris.application.request_command as request_command
from iris.application.bootstrap import IrisRequestRuntime
from iris.application.request import IrisRequestExecutionOutput, IrisRequestOrchestrator
from iris.application.request_artifact import read_request_artifact_v1
from iris.backbone import BackboneAdapter
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace import WorkspaceState

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (7,)
SIDECAR_STATE_SHA256 = "c" * 64
SECRET = "IRIS_SECRET_ECHO_5dc2_Привет"


class OfflineCommandOrchestrator(IrisRequestOrchestrator):
    def __init__(self, execution: IrisRequestExecutionOutput) -> None:
        self.execution = execution
        self.calls: list[tuple[str, str, torch.Generator | None]] = []

    @property
    def sidecar_state_sha256(self) -> str:
        return SIDECAR_STATE_SHA256

    def execute(
        self,
        prompt: str,
        *,
        strategy: str,
        generator: torch.Generator | None = None,
    ) -> IrisRequestExecutionOutput:
        self.calls.append((prompt, strategy, generator))
        return self.execution


def validated_config() -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "iris", "version": "0.1.0"},
            "hardware": {
                "name": "offline_cpu",
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
                "local_files_only": True,
                "output_hidden_states": True,
                "use_cache": False,
                "dtype": "fp32",
                "tokenizer": {
                    "use_fast": True,
                    "padding_side": "left",
                    "truncation_side": "left",
                },
                "limits": {"max_prompt_tokens": 2, "max_new_tokens": 1},
            },
            "cognitive": {
                "name": "offline",
                "state_dim": 1,
                "num_modules": 1,
                "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
            },
            "workspace": {
                "name": "offline",
                "num_slots": 1,
                "cycles": 1,
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
                "seed": 42,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": "artifacts/runs",
            },
        }
    )


def request_runtime() -> tuple[IrisRequestRuntime, OfflineCommandOrchestrator]:
    state = WorkspaceState(
        slots=torch.tensor([[[0.5]]], dtype=torch.float32),
        age=torch.zeros((1, 1), dtype=torch.int64),
        occupancy=torch.ones((1, 1), dtype=torch.bool),
        source_module=torch.zeros((1, 1), dtype=torch.int64),
        confidence=torch.full((1, 1), 0.75, dtype=torch.float32),
        ttl=torch.ones((1, 1), dtype=torch.float32),
        write_count=torch.ones((1, 1), dtype=torch.int64),
    )
    generation = IrisGeneratorOutput(
        text=SECRET,
        token_ids=torch.tensor([[7]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=7,
        workspace_trace=(state,),
    )
    execution = IrisRequestExecutionOutput(
        generation=generation,
        prompt_input_ids=torch.tensor([[11]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 1), dtype=torch.int64),
    )
    orchestrator = OfflineCommandOrchestrator(execution)
    config = validated_config()
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode()
    runtime = IrisRequestRuntime(
        backbone=cast(BackboneAdapter, object()),
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Offline Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=42,
        deterministic=True,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256("iris/configuration/v1", configuration_json.encode()),
        sidecar_state_sha256=SIDECAR_STATE_SHA256,
    )
    return runtime, orchestrator


def test_command_generates_once_then_installs_one_private_content_addressed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime, orchestrator = request_runtime()
    config = OmegaConf.create({"offline": True})
    assert isinstance(config, DictConfig)
    assert not GlobalHydra.instance().is_initialized()
    policy = TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)

    with (
        patch.object(request_command, "_compose_request_config", return_value=config) as compose,
        patch.object(
            request_command,
            "bootstrap_request_runtime",
            return_value=runtime,
        ) as bootstrap,
    ):
        first = request_command.generate_request(
            prompt=SECRET,
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )
        second = request_command.generate_request(
            prompt=SECRET,
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )

    assert first is orchestrator.execution.generation
    assert second is first
    assert orchestrator.calls == [
        (SECRET, "greedy", None),
        (SECRET, "greedy", None),
    ]
    assert compose.call_count == 2
    assert bootstrap.call_count == 2
    artifacts = list((tmp_path / "artifacts" / "runs" / "requests" / "v1").glob("*.json"))
    assert len(artifacts) == 1
    artifact_bytes = artifacts[0].read_bytes()
    assert SECRET.encode() not in artifact_bytes
    assert artifact_bytes.endswith(b"\n")
    assert not artifact_bytes.startswith(b"\xef\xbb\xbf")
    artifact = read_request_artifact_v1(artifacts[0])
    assert artifacts[0].stem == artifact.payload_sha256
    assert artifact.payload.request.decoding.stop_token_ids == STOP_TOKEN_IDS
    assert artifact.payload.generation.text.representation == "sha256_only"
    assert artifact.payload.generation.token_ids.shape == (1, 1)
    assert not GlobalHydra.instance().is_initialized()
