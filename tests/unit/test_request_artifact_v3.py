"""Profile-v4 builder and integrity tests for request artifacts v3."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import PropertyMock, patch

import pytest
import torch

import iris.application as application
import iris.artifacts as artifacts
from iris.application.arm_profiles import get_arm_execution_profile_v4
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestArtifactError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import (
    IrisRequestExecutionOutput,
    IrisRequestExecutionReceiptV1,
    IrisRequestOrchestrator,
)
from iris.application.request_artifact_v3 import (
    build_request_artifact_v3,
    read_request_artifact_v3,
    write_request_artifact_v3,
)
from iris.application.shared_buffer_request import (
    CognitiveSharedBufferRequestOrchestrator,
    IrisSharedBufferRequestExecutionOutputV1,
)
from iris.artifacts.request_v3 import (
    IrisRequestArtifactEnvelopeV3,
    RequestArmCTopologyV3,
    SharedBufferTraceProvenanceV3,
)
from iris.backbone import BackboneAdapter
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.shared_buffer import SharedBufferState
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace import WorkspaceReaderOutput, WorkspaceState
from tests.unit.test_request_artifact_schema import valid_bootstrap_config

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (2, 7)
SIDECAR_SHA256 = "c" * 64
PROFILE_SHA256: Literal["ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"] = (
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
)
POLICY_SHA256 = "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
SECRET = "IRIS_V3_PRIVATE_4ca8_Привет\r\n"
SAMPLING = TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)


def _config(output_dir: Path) -> IrisBootstrapConfig:
    raw = valid_bootstrap_config().model_dump(mode="python")
    cast(dict[str, object], raw["runtime"])["output_dir"] = str(output_dir)
    return IrisBootstrapConfig.model_validate(raw)


def _limited_workspace_state(cycle: int) -> WorkspaceState:
    slots = torch.zeros((1, 8, 256), dtype=torch.float32)
    age = torch.zeros((1, 8), dtype=torch.int64)
    occupancy = torch.zeros((1, 8), dtype=torch.bool)
    source_module = torch.full((1, 8), -1, dtype=torch.int64)
    confidence = torch.zeros((1, 8), dtype=torch.float32)
    ttl = torch.zeros((1, 8), dtype=torch.float32)
    write_count = torch.zeros((1, 8), dtype=torch.int64)
    slots[0, 0, 0] = float(cycle)
    occupancy[0, 0] = True
    source_module[0, 0] = cycle - 1
    confidence[0, 0] = 0.75
    ttl[0, 0] = 3.0
    write_count[0, 0] = cycle
    return WorkspaceState(
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source_module,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
    )


def _shared_state(cycle: int) -> SharedBufferState:
    entry_count = cycle * 8
    entries = torch.arange(entry_count * 256, dtype=torch.float32).reshape(1, entry_count, 256)
    entries = entries / 1000.0
    positions = torch.arange(entry_count, dtype=torch.int64).reshape(1, entry_count)
    return SharedBufferState(
        entries=entries,
        source_module=positions.remainder(8),
        write_cycle=torch.div(positions, 8, rounding_mode="floor") + 1,
        state_dim=256,
        num_modules=8,
        cycles=3,
    )


def _shared_read(entry_count: int, *, offset: float = 0.0) -> WorkspaceReaderOutput:
    if entry_count == 0:
        reads = torch.zeros((1, 8, 256), dtype=torch.float32)
        attention = torch.zeros((1, 8, 0), dtype=torch.float32)
    else:
        reads = torch.full((1, 8, 256), offset + float(entry_count), dtype=torch.float32)
        attention = torch.full((1, 8, entry_count), 1.0 / entry_count, dtype=torch.float32)
    return WorkspaceReaderOutput(workspace_reads=reads, attention_weights=attention)


def _receipt() -> IrisRequestExecutionReceiptV1:
    return IrisRequestExecutionReceiptV1(
        prompt_sha256=domain_sha256("iris/request-execution/v1/prompt", SECRET.encode()),
        strategy="greedy",
        stop_token_ids=STOP_TOKEN_IDS,
        sampling=SAMPLING,
        sampling_seed=None,
    )


def _generation(trace: tuple[WorkspaceState, ...]) -> IrisGeneratorOutput:
    return IrisGeneratorOutput(
        text=SECRET,
        token_ids=torch.tensor([[5, 2]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=2,
        workspace_trace=trace,
    )


def _execution(
    arm_id: str,
) -> IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1:
    if arm_id == "C":
        states = (_shared_state(1), _shared_state(2), _shared_state(3))
        return IrisSharedBufferRequestExecutionOutputV1(
            generation=_generation(tuple(state.to_workspace_state() for state in states)),
            prompt_input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
            receipt=_receipt(),
            shared_buffer_trace=states,
            pre_update_reads=(_shared_read(0), _shared_read(8), _shared_read(16)),
            final_output_read=_shared_read(24),
        )
    trace = tuple(_limited_workspace_state(cycle) for cycle in range(1, 4)) if arm_id == "D" else ()
    return IrisRequestExecutionOutput(
        generation=_generation(trace),
        prompt_input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
        receipt=_receipt(),
    )


def _runtime(tmp_path: Path, arm_id: str) -> IrisArmRequestRuntimeV3:
    config = _config(tmp_path / "artifacts")
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode()
    backbone = cast(BackboneAdapter, object())
    orchestrator: (
        BaseLlmRequestOrchestrator
        | CognitiveNoWorkspaceRequestOrchestrator
        | CognitiveSharedBufferRequestOrchestrator
        | IrisRequestOrchestrator
    )
    if arm_id == "A":
        orchestrator = object.__new__(BaseLlmRequestOrchestrator)
    elif arm_id == "B":
        orchestrator = object.__new__(CognitiveNoWorkspaceRequestOrchestrator)
    elif arm_id == "C":
        orchestrator = object.__new__(CognitiveSharedBufferRequestOrchestrator)
    else:
        orchestrator = object.__new__(IrisRequestOrchestrator)
    orchestrator._backbone = backbone
    return IrisArmRequestRuntimeV3(
        profile=get_arm_execution_profile_v4(arm_id),
        execution_profile_set_sha256=PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Request Artifact V3 Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256(
            "iris/configuration/v1",
            configuration_json.encode(),
        ),
        sidecar_state_sha256=None if arm_id == "A" else SIDECAR_SHA256,
    )


def _runtime_property_patches(
    arm_id: str,
    *,
    sidecar_sha256: str = SIDECAR_SHA256,
    policy_sha256: str = POLICY_SHA256,
) -> AbstractContextManager[object]:
    if arm_id == "A":
        return nullcontext()
    stack = ExitStack()
    owner: type[object]
    if arm_id == "B":
        owner = CognitiveNoWorkspaceRequestOrchestrator
    elif arm_id == "C":
        owner = CognitiveSharedBufferRequestOrchestrator
    else:
        owner = IrisRequestOrchestrator
    stack.enter_context(
        patch.object(
            owner,
            "sidecar_state_sha256",
            new_callable=PropertyMock,
            return_value=sidecar_sha256,
        )
    )
    if arm_id == "C":
        stack.enter_context(
            patch.object(
                CognitiveSharedBufferRequestOrchestrator,
                "shared_buffer_policy_sha256",
                new_callable=PropertyMock,
                return_value=policy_sha256,
            )
        )
    return stack


def artifact_inputs(
    tmp_path: Path,
    arm_id: str,
) -> tuple[
    IrisArmRequestRuntimeV3,
    IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1,
    dict[str, object],
]:
    runtime = _runtime(tmp_path, arm_id)
    execution = _execution(arm_id)
    kwargs: dict[str, object] = {
        "prompt": SECRET,
        "strategy": "greedy",
        "stop_token_ids": STOP_TOKEN_IDS,
        "sampling": SAMPLING,
        "sampling_seed": None,
        "allow_download": False,
        "runtime": runtime,
        "execution": execution,
    }
    return runtime, execution, kwargs


def build_artifact(tmp_path: Path, arm_id: str) -> IrisRequestArtifactEnvelopeV3:
    _, _, kwargs = artifact_inputs(tmp_path, arm_id)
    with _runtime_property_patches(arm_id):
        return build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arm_id", "request_graph", "trace_contract", "cycle_count"),
    [
        ("A", "frozen_backbone_prompt_only_v1", "empty_workspace_trace", 0),
        (
            "B",
            "independent_recurrent_modules_zero_memory_read_v1",
            "empty_workspace_trace",
            0,
        ),
        (
            "C",
            "append_only_all_proposals_prior_cycle_shared_buffer_v1",
            "append_only_shared_buffer_per_cycle_v1",
            3,
        ),
        ("D", "limited_competitive_workspace_v1", "workspace_state_per_cycle", 3),
    ],
)
def test_builder_binds_every_profile_v4_arm_to_its_exact_trace(
    tmp_path: Path,
    arm_id: str,
    request_graph: str,
    trace_contract: str,
    cycle_count: int,
) -> None:
    artifact = build_artifact(tmp_path, arm_id)

    assert artifact.payload.topology.arm_id == arm_id
    assert artifact.payload.topology.request_graph == request_graph
    assert artifact.payload.workspace_trace.trace_contract == trace_contract
    assert artifact.payload.workspace_trace.cycle_count == cycle_count
    assert artifact.payload.execution_profile_set_version == 4
    assert artifact.payload.execution_profile_set_sha256 == PROFILE_SHA256
    assert artifact.payload.shared_buffer_policy_sha256 == POLICY_SHA256
    raw = canonical_json_bytes(artifact.model_dump(mode="json", by_alias=True))
    assert SECRET.encode() not in raw


def test_arm_c_trace_proves_complete_native_lifecycle_without_tensor_values(
    tmp_path: Path,
) -> None:
    artifact = build_artifact(tmp_path, "C")
    topology = artifact.payload.topology
    trace = artifact.payload.workspace_trace

    assert isinstance(topology, RequestArmCTopologyV3)
    assert isinstance(trace, SharedBufferTraceProvenanceV3)
    assert topology.sidecar_component_names == (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "shared_buffer_reader",
        "cognitive_modules",
        "proposal_head",
        "shared_buffer_writer",
        "shared_buffer_latent_bridge",
        "prefix_assembler",
    )
    assert topology.entries_per_cycle == topology.latent_token_count == 8
    assert topology.final_entry_count == 24
    assert tuple(cycle.pre_update_read.entry_count for cycle in trace.cycles) == (0, 8, 16)
    assert tuple(cycle.post_cycle_state.entry_count for cycle in trace.cycles) == (8, 16, 24)
    assert trace.final_output_read.entry_count == 24
    assert trace.final_output_read.attention_weights.shape == (1, 8, 24)
    state_json = trace.cycles[-1].post_cycle_state.model_dump(mode="json")
    assert set(cast(dict[str, object], state_json["entries"])) == {
        "shape",
        "dtype",
        "device",
        "byte_length",
        "sha256",
    }
    raw = canonical_json_bytes(artifact.model_dump(mode="json", by_alias=True))
    assert b"23.552" not in raw


def test_arm_c_final_read_change_updates_read_trace_and_payload_hashes(tmp_path: Path) -> None:
    original = build_artifact(tmp_path, "C")
    runtime, execution, kwargs = artifact_inputs(tmp_path, "C")
    assert isinstance(execution, IrisSharedBufferRequestExecutionOutputV1)
    kwargs["execution"] = replace(execution, final_output_read=_shared_read(24, offset=0.5))

    with _runtime_property_patches("C"):
        changed = build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]

    original_trace = cast(SharedBufferTraceProvenanceV3, original.payload.workspace_trace)
    changed_trace = cast(SharedBufferTraceProvenanceV3, changed.payload.workspace_trace)
    assert original_trace.final_output_read.workspace_reads.sha256 != (
        changed_trace.final_output_read.workspace_reads.sha256
    )
    assert original_trace.sha256 != changed_trace.sha256
    assert original.payload_sha256 != changed.payload_sha256
    assert runtime.sidecar_state_sha256 == SIDECAR_SHA256


def test_arm_c_rejects_non_append_only_native_state(tmp_path: Path) -> None:
    _, execution, kwargs = artifact_inputs(tmp_path, "C")
    assert isinstance(execution, IrisSharedBufferRequestExecutionOutputV1)
    states = list(execution.shared_buffer_trace)
    entries = states[1].entries.clone()
    entries[0, 0, 0] += 1.0
    changed_states = (
        states[0],
        replace(states[1], entries=entries),
        states[2],
    )
    kwargs["execution"] = replace(execution, shared_buffer_trace=changed_states)

    with (
        _runtime_property_patches("C"),
        pytest.raises(IrisRequestArtifactError, match="could not be built") as raised,
    ):
        build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]
    assert "append-only prefix" in str(raised.value.__cause__)


def test_arm_c_rejects_read_that_omits_a_buffer_entry(tmp_path: Path) -> None:
    _, execution, kwargs = artifact_inputs(tmp_path, "C")
    assert isinstance(execution, IrisSharedBufferRequestExecutionOutputV1)
    invalid = WorkspaceReaderOutput(
        workspace_reads=execution.final_output_read.workspace_reads,
        attention_weights=torch.full((1, 8, 23), 1.0 / 23, dtype=torch.float32),
    )
    kwargs["execution"] = replace(execution, final_output_read=invalid)

    with (
        _runtime_property_patches("C"),
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]


def test_arm_c_rejects_policy_or_sidecar_mutation(tmp_path: Path) -> None:
    _, _, kwargs = artifact_inputs(tmp_path, "C")
    with (
        _runtime_property_patches("C", policy_sha256="d" * 64),
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]
    with (
        _runtime_property_patches("C", sidecar_sha256="e" * 64),
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]


def test_writer_reader_round_trip_and_detect_nested_arm_c_tampering(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path, "C")
    first = write_request_artifact_v3(artifact)
    second = write_request_artifact_v3(artifact)

    assert not first.reused_existing
    assert second.reused_existing
    assert read_request_artifact_v3(first.path) == artifact
    content = first.path.read_text(encoding="utf-8")
    first.path.write_text(
        content.replace('"entry_count":24', '"entry_count":23', 1), encoding="utf-8"
    )
    with pytest.raises(IrisRequestArtifactError, match="could not be read or verified"):
        read_request_artifact_v3(first.path)


def test_builder_rejects_cross_arm_profile_and_orchestrator_before_trace_use(
    tmp_path: Path,
) -> None:
    runtime, execution, kwargs = artifact_inputs(tmp_path, "C")
    kwargs["runtime"] = replace(runtime, profile=get_arm_execution_profile_v4("D"))

    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]
    assert isinstance(execution, IrisSharedBufferRequestExecutionOutputV1)


def test_request_v3_schema_is_exported_from_artifacts_package() -> None:
    assert artifacts.IrisRequestArtifactEnvelopeV3 is IrisRequestArtifactEnvelopeV3
    assert artifacts.RequestArmCTopologyV3 is RequestArmCTopologyV3
    assert artifacts.SharedBufferTraceProvenanceV3 is SharedBufferTraceProvenanceV3


def test_request_v3_services_are_exported_from_application_package() -> None:
    assert application.build_request_artifact_v3 is build_request_artifact_v3
    assert application.read_request_artifact_v3 is read_request_artifact_v3
    assert application.write_request_artifact_v3 is write_request_artifact_v3
