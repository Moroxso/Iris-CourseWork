"""Build, verify, and atomically persist profile-v4 request artifacts v3."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from re import fullmatch
from typing import Any, Literal, cast
from uuid import uuid4

import torch

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV4,
    get_arm_execution_profile_v4,
)
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestArtifactError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import (
    IrisRequestExecutionOutput,
    IrisRequestExecutionReceiptV1,
    IrisRequestOrchestrator,
)
from iris.application.request_artifact import (
    _SUPPORTED_RUNTIME_DTYPES,
    RequestArtifactWriteResult,
    _configuration,
    _exact_utf8,
    _existing_matches,
    _path_entry_exists,
    _private_text,
    _reject_duplicate_keys,
    _require_real_directory,
    _software,
    _tensor_fingerprint,
    _validate_generation,
    _validate_prompt_tokens,
)
from iris.application.shared_buffer_request import (
    CognitiveSharedBufferRequestOrchestrator,
    IrisSharedBufferRequestExecutionOutputV1,
)
from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    DecodingProvenanceV1,
    GenerationProvenanceV1,
    PrivateTextFingerprintV1,
    RequestProvenanceV1,
    WorkspaceCycleProvenanceV1,
)
from iris.artifacts.request_v3 import (
    EmptyWorkspaceTraceProvenanceV3,
    IrisRequestArtifactEnvelopeV3,
    IrisRequestArtifactPayloadV3,
    RequestArmATopologyV3,
    RequestArmBTopologyV3,
    RequestArmCTopologyV3,
    RequestArmDTopologyV3,
    RequestArmTopologyV3,
    RequestArtifactSchemaV3,
    RequestRuntimeProvenanceV3,
    RequestWorkspaceTraceProvenanceV3,
    SharedBufferCycleProvenanceV3,
    SharedBufferReadProvenanceV3,
    SharedBufferStateProvenanceV3,
    SharedBufferTraceProvenanceV3,
    WorkspaceTraceProvenanceV3,
)
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.shared_buffer import SharedBufferState, shared_buffer_policy_sha256_v1
from iris.shared_buffer.state import validate_shared_buffer_state
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.workspace import WorkspaceReaderOutput, WorkspaceState

_CONFIGURATION_DOMAIN = "iris/configuration/v1"
_PROMPT_DOMAIN = "iris/request-artifact/v3/prompt"
_TEXT_DOMAIN = "iris/request-artifact/v3/generated-text"
_PAYLOAD_DOMAIN = "iris/request-artifact/v3/payload"
_TRACE_DOMAIN = "iris/request-artifact/v3/workspace-trace"
_PROFILE_SET_SHA256: Literal["ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"] = (
    "ee3ca30272dc1b6831232b600017fd11f367f0821cfbcb28df7d95ace6440ae1"
)
_POLICY_SHA256: Literal["56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"] = (
    "56da19f1b86b6ee9a540b42a6556f6da6e359345dff833d1255a9a429c84719f"
)
_EXECUTION_PROMPT_DOMAIN = "iris/request-execution/v1/prompt"
_ARM_B_COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "prefix_assembler",
)
_ARM_C_COMPONENT_NAMES = (
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
_ARM_D_COMPONENT_NAMES = (
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

type RequestExecutionOutputV3 = (
    IrisRequestExecutionOutput | IrisSharedBufferRequestExecutionOutputV1
)


def _validate_runtime(
    runtime: IrisArmRequestRuntimeV3,
) -> tuple[IrisBootstrapConfig, RequestArmTopologyV3]:
    profile = runtime.profile
    if type(profile) is not IrisArmExecutionProfileV4:
        message = "request artifact v3 profile must be exact IrisArmExecutionProfileV4"
        raise TypeError(message)
    if profile != get_arm_execution_profile_v4(profile.arm_id):
        message = "request artifact v3 profile differs from execution profile set v4"
        raise ValueError(message)
    if runtime.execution_profile_set_sha256 != _PROFILE_SET_SHA256:
        message = "request artifact v3 profile-set SHA-256 is not canonical"
        raise ValueError(message)
    if type(runtime.device_info) is not DeviceInfo:
        message = "request artifact v3 runtime requires exact DeviceInfo"
        raise TypeError(message)
    config = _configuration(runtime)
    orchestrator = runtime.orchestrator
    expected_type: type[
        BaseLlmRequestOrchestrator
        | CognitiveNoWorkspaceRequestOrchestrator
        | CognitiveSharedBufferRequestOrchestrator
        | IrisRequestOrchestrator
    ]
    if profile.arm_id == "A":
        expected_type = BaseLlmRequestOrchestrator
    elif profile.arm_id == "B":
        expected_type = CognitiveNoWorkspaceRequestOrchestrator
    elif profile.arm_id == "C":
        expected_type = CognitiveSharedBufferRequestOrchestrator
    else:
        expected_type = IrisRequestOrchestrator
    if type(orchestrator) is not expected_type:
        message = f"arm-{profile.arm_id} request artifact runtime uses the wrong graph"
        raise TypeError(message)
    if orchestrator.backbone is not runtime.backbone:
        message = "request artifact runtime orchestrator does not own the runtime backbone"
        raise ValueError(message)

    if profile.arm_id == "A":
        if runtime.sidecar_state_sha256 is not None:
            message = "arm-A request artifact runtime must not own a sidecar"
            raise ValueError(message)
        return config, RequestArmATopologyV3(
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

    sidecar_sha256 = runtime.sidecar_state_sha256
    if type(sidecar_sha256) is not str or fullmatch(r"[0-9a-f]{64}", sidecar_sha256) is None:
        message = f"arm-{profile.arm_id} request artifact runtime requires a sidecar SHA-256"
        raise ValueError(message)
    sidecar_orchestrator = cast(
        CognitiveNoWorkspaceRequestOrchestrator
        | CognitiveSharedBufferRequestOrchestrator
        | IrisRequestOrchestrator,
        orchestrator,
    )
    if sidecar_orchestrator.sidecar_state_sha256 != sidecar_sha256:
        message = f"arm-{profile.arm_id} sidecar provenance changed after execution"
        raise ValueError(message)
    if profile.arm_id == "B":
        return config, RequestArmBTopologyV3(
            arm_id="B",
            arm_kind="llm_modules_no_workspace",
            request_graph="independent_recurrent_modules_zero_memory_read_v1",
            memory_model="none",
            trace_contract="empty_workspace_trace",
            cognitive_cycles=3,
            latent_token_count=config.cognitive.num_modules,
            zero_memory_read_cycles=3,
            sidecar_component_names=_ARM_B_COMPONENT_NAMES,
            sidecar_state_sha256=sidecar_sha256,
        )
    if profile.arm_id == "C":
        shared = cast(CognitiveSharedBufferRequestOrchestrator, orchestrator)
        if shared.shared_buffer_policy_sha256 != _POLICY_SHA256:
            message = "arm-C runtime does not own the canonical shared-buffer policy"
            raise ValueError(message)
        return config, RequestArmCTopologyV3(
            arm_id="C",
            arm_kind="llm_modules_unlimited_shared_buffer",
            request_graph="append_only_all_proposals_prior_cycle_shared_buffer_v1",
            memory_model="unlimited_shared_buffer",
            trace_contract="append_only_shared_buffer_per_cycle_v1",
            shared_buffer_policy_sha256=_POLICY_SHA256,
            cognitive_cycles=3,
            pre_update_read_count=3,
            final_read_count=1,
            entries_per_cycle=config.cognitive.num_modules,
            final_entry_count=3 * config.cognitive.num_modules,
            latent_token_count=config.cognitive.num_modules,
            sidecar_component_names=_ARM_C_COMPONENT_NAMES,
            sidecar_state_sha256=sidecar_sha256,
        )
    return config, RequestArmDTopologyV3(
        arm_id="D",
        arm_kind="llm_modules_limited_competitive_workspace",
        request_graph="limited_competitive_workspace_v1",
        memory_model="limited_competitive_workspace",
        trace_contract="workspace_state_per_cycle",
        cognitive_cycles=config.workspace.cycles,
        latent_token_count=config.workspace.num_slots,
        zero_memory_read_cycles=0,
        sidecar_component_names=_ARM_D_COMPONENT_NAMES,
        sidecar_state_sha256=sidecar_sha256,
    )


def _validate_workspace_state(
    state: WorkspaceState,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV3,
) -> None:
    if type(state) is not WorkspaceState:
        message = "arm-D request artifact cycles require exact WorkspaceState values"
        raise TypeError(message)
    metadata_shape = (1, config.workspace.num_slots)
    tensors = {
        "slots": (state.slots, (*metadata_shape, config.cognitive.state_dim), runtime.dtype),
        "age": (state.age, metadata_shape, torch.int64),
        "occupancy": (state.occupancy, metadata_shape, torch.bool),
        "source_module": (state.source_module, metadata_shape, torch.int64),
        "confidence": (state.confidence, metadata_shape, runtime.dtype),
        "ttl": (state.ttl, metadata_shape, runtime.dtype),
        "write_count": (state.write_count, metadata_shape, torch.int64),
    }
    for name, (tensor, shape, dtype) in tensors.items():
        if (
            type(tensor) is not torch.Tensor
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != runtime.device_info.device
            or tensor.requires_grad
            or tensor.grad_fn is not None
        ):
            message = f"arm-D request artifact workspace {name} tensor is invalid"
            raise ValueError(message)
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            message = f"arm-D request artifact workspace {name} contains NaN or Inf"
            raise ValueError(message)
    empty = ~state.occupancy
    if not (
        bool((state.slots[empty] == 0).all())
        and bool((state.age[empty] == 0).all())
        and bool((state.source_module[empty] == -1).all())
        and bool((state.confidence[empty] == 0).all())
        and bool((state.ttl[empty] == 0).all())
        and bool((state.write_count[empty] == 0).all())
    ):
        message = "arm-D request artifact workspace contains noncanonical empty slots"
        raise ValueError(message)
    if bool((state.age < 0).any()) or bool((state.write_count < 0).any()):
        message = "arm-D request artifact workspace counters cannot be negative"
        raise ValueError(message)
    if bool((state.confidence < 0).any()) or bool((state.confidence > 1).any()):
        message = "arm-D request artifact workspace confidence must lie in [0, 1]"
        raise ValueError(message)
    if bool((state.ttl < 0).any()):
        message = "arm-D request artifact workspace TTL cannot be negative"
        raise ValueError(message)
    occupied_sources = state.source_module[state.occupancy]
    if bool((occupied_sources < 0).any()) or bool(
        (occupied_sources >= config.cognitive.num_modules).any()
    ):
        message = "arm-D request artifact workspace contains an invalid source module"
        raise ValueError(message)


def _workspace_trace(
    output: IrisGeneratorOutput,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV3,
) -> RequestWorkspaceTraceProvenanceV3:
    trace = output.workspace_trace
    if type(trace) is not tuple:
        message = "request artifact workspace trace must be an exact tuple"
        raise TypeError(message)
    if runtime.profile.arm_id in {"A", "B"}:
        if trace:
            message = f"arm-{runtime.profile.arm_id} request artifact requires an empty trace"
            raise ValueError(message)
        return EmptyWorkspaceTraceProvenanceV3(
            trace_contract="empty_workspace_trace",
            cycle_count=0,
            cycles=(),
            sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes([])),
        )
    if runtime.profile.arm_id != "D":
        message = "arm-C native trace must be built from its exact execution output"
        raise TypeError(message)
    if len(trace) != config.workspace.cycles:
        message = "arm-D request artifact must preserve every configured workspace cycle"
        raise ValueError(message)
    records: list[WorkspaceCycleProvenanceV1] = []
    for cycle, state in enumerate(trace, start=1):
        _validate_workspace_state(state, config=config, runtime=runtime)
        fingerprints = {
            name: _tensor_fingerprint(
                getattr(state, name),
                domain=f"iris/request-artifact/v3/workspace/cycle/{cycle}/{name}",
            )
            for name in (
                "slots",
                "age",
                "occupancy",
                "source_module",
                "confidence",
                "ttl",
                "write_count",
            )
        }
        cycle_content = {
            "cycle": cycle,
            **{name: value.model_dump(mode="json") for name, value in fingerprints.items()},
        }
        records.append(
            WorkspaceCycleProvenanceV1(
                cycle=cycle,
                sha256=domain_sha256(
                    f"iris/request-artifact/v3/workspace/cycle/{cycle}",
                    canonical_json_bytes(cycle_content),
                ),
                **fingerprints,
            )
        )
    trace_content = [record.model_dump(mode="json") for record in records]
    return WorkspaceTraceProvenanceV3(
        trace_contract="workspace_state_per_cycle",
        cycle_count=len(records),
        cycles=tuple(records),
        sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_content)),
    )


def _validate_shared_state(
    state: SharedBufferState,
    *,
    cycle: int,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV3,
) -> None:
    contract = validate_shared_buffer_state(
        state,
        state_dim=config.cognitive.state_dim,
        num_modules=config.cognitive.num_modules,
        cycles=3,
        error_type=ValueError,
        component=f"Arm-C artifact cycle {cycle}",
    )
    if (
        contract.batch_size != 1
        or contract.entry_count != cycle * config.cognitive.num_modules
        or contract.dtype != runtime.dtype
        or contract.device != runtime.device_info.device
    ):
        message = "arm-C native state cardinality or placement is invalid"
        raise ValueError(message)
    for name in ("entries", "source_module", "write_cycle"):
        tensor = getattr(state, name)
        if tensor.requires_grad or tensor.grad_fn is not None:
            message = f"arm-C native state {name} is not an inference snapshot"
            raise ValueError(message)


def _validate_shared_read(
    read: WorkspaceReaderOutput,
    *,
    entry_count: int,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV3,
) -> None:
    if type(read) is not WorkspaceReaderOutput:
        message = "arm-C request artifact requires exact WorkspaceReaderOutput values"
        raise TypeError(message)
    expected_reads = (1, config.cognitive.num_modules, config.cognitive.state_dim)
    expected_attention = (1, config.cognitive.num_modules, entry_count)
    attention_dtype = (
        torch.float32 if runtime.dtype in {torch.float16, torch.bfloat16} else runtime.dtype
    )
    if (
        type(read.workspace_reads) is not torch.Tensor
        or tuple(read.workspace_reads.shape) != expected_reads
        or read.workspace_reads.dtype != runtime.dtype
        or read.workspace_reads.device != runtime.device_info.device
        or read.workspace_reads.requires_grad
        or read.workspace_reads.grad_fn is not None
        or not bool(torch.isfinite(read.workspace_reads).all())
        or type(read.attention_weights) is not torch.Tensor
        or tuple(read.attention_weights.shape) != expected_attention
        or read.attention_weights.dtype != attention_dtype
        or read.attention_weights.device != runtime.device_info.device
        or read.attention_weights.requires_grad
        or read.attention_weights.grad_fn is not None
        or not bool(torch.isfinite(read.attention_weights).all())
        or not bool((read.attention_weights >= 0).all())
    ):
        message = "arm-C request artifact reader snapshot is invalid"
        raise ValueError(message)
    expected_sum = 0.0 if entry_count == 0 else 1.0
    if not torch.allclose(
        read.attention_weights.sum(dim=-1),
        torch.full(
            (1, config.cognitive.num_modules),
            expected_sum,
            dtype=attention_dtype,
            device=runtime.device_info.device,
        ),
        rtol=1e-4,
        atol=1e-6,
    ):
        message = "arm-C reader attention does not cover its complete input buffer"
        raise ValueError(message)
    if entry_count == 0 and not bool((read.workspace_reads == 0).all()):
        message = "arm-C zero-entry pre-update read must return exact zero values"
        raise ValueError(message)


def _shared_read_record(
    read: WorkspaceReaderOutput,
    *,
    read_kind: Literal["pre_update", "final_output"],
    cycle: int,
    entry_count: int,
) -> SharedBufferReadProvenanceV3:
    base_domain = f"iris/request-artifact/v3/shared-buffer/{read_kind}/{cycle}"
    workspace_reads = _tensor_fingerprint(
        read.workspace_reads,
        domain=f"{base_domain}/workspace-reads",
    )
    attention_weights = _tensor_fingerprint(
        read.attention_weights,
        domain=f"{base_domain}/attention-weights",
    )
    content = {
        "read_kind": read_kind,
        "cycle": cycle,
        "entry_count": entry_count,
        "workspace_reads": workspace_reads.model_dump(mode="json"),
        "attention_weights": attention_weights.model_dump(mode="json"),
    }
    return SharedBufferReadProvenanceV3(
        read_kind=read_kind,
        cycle=cycle,
        entry_count=entry_count,
        workspace_reads=workspace_reads,
        attention_weights=attention_weights,
        sha256=domain_sha256(
            base_domain,
            canonical_json_bytes(content),
        ),
    )


def _shared_state_record(
    state: SharedBufferState,
    *,
    cycle: int,
) -> SharedBufferStateProvenanceV3:
    base_domain = f"iris/request-artifact/v3/shared-buffer/state/{cycle}"
    entries = _tensor_fingerprint(state.entries, domain=f"{base_domain}/entries")
    source_module = _tensor_fingerprint(
        state.source_module,
        domain=f"{base_domain}/source-module",
    )
    write_cycle = _tensor_fingerprint(
        state.write_cycle,
        domain=f"{base_domain}/write-cycle",
    )
    content = {
        "cycle": cycle,
        "entry_count": state.entry_count,
        "entries": entries.model_dump(mode="json"),
        "source_module": source_module.model_dump(mode="json"),
        "write_cycle": write_cycle.model_dump(mode="json"),
    }
    return SharedBufferStateProvenanceV3(
        cycle=cycle,
        entry_count=state.entry_count,
        entries=entries,
        source_module=source_module,
        write_cycle=write_cycle,
        sha256=domain_sha256(
            base_domain,
            canonical_json_bytes(content),
        ),
    )


def _shared_buffer_trace(
    execution: IrisSharedBufferRequestExecutionOutputV1,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV3,
) -> SharedBufferTraceProvenanceV3:
    if type(execution.shared_buffer_trace) is not tuple or len(execution.shared_buffer_trace) != 3:
        message = "arm-C request artifact requires exactly three native post-cycle states"
        raise ValueError(message)
    if type(execution.pre_update_reads) is not tuple or len(execution.pre_update_reads) != 3:
        message = "arm-C request artifact requires exactly three pre-update reads"
        raise ValueError(message)
    if shared_buffer_policy_sha256_v1() != _POLICY_SHA256:
        message = "installed shared-buffer policy differs from request artifact v3"
        raise ValueError(message)
    module_count = config.cognitive.num_modules
    cycles: list[SharedBufferCycleProvenanceV3] = []
    previous: SharedBufferState | None = None
    for cycle, (state, read) in enumerate(
        zip(execution.shared_buffer_trace, execution.pre_update_reads, strict=True),
        start=1,
    ):
        _validate_shared_state(state, cycle=cycle, config=config, runtime=runtime)
        entry_count_before = (cycle - 1) * module_count
        _validate_shared_read(
            read,
            entry_count=entry_count_before,
            config=config,
            runtime=runtime,
        )
        if previous is not None:
            previous_count = previous.entry_count
            if not (
                torch.equal(state.entries[:, :previous_count], previous.entries)
                and torch.equal(state.source_module[:, :previous_count], previous.source_module)
                and torch.equal(state.write_cycle[:, :previous_count], previous.write_cycle)
            ):
                message = "arm-C post-cycle states do not preserve an append-only prefix"
                raise ValueError(message)
        read_record = _shared_read_record(
            read,
            read_kind="pre_update",
            cycle=cycle,
            entry_count=entry_count_before,
        )
        state_record = _shared_state_record(state, cycle=cycle)
        cycle_content = {
            "cycle": cycle,
            "pre_update_read": read_record.model_dump(mode="json"),
            "post_cycle_state": state_record.model_dump(mode="json"),
        }
        cycles.append(
            SharedBufferCycleProvenanceV3(
                cycle=cycle,
                pre_update_read=read_record,
                post_cycle_state=state_record,
                sha256=domain_sha256(
                    f"iris/request-artifact/v3/shared-buffer/cycle/{cycle}",
                    canonical_json_bytes(cycle_content),
                ),
            )
        )
        previous = state
    _validate_shared_read(
        execution.final_output_read,
        entry_count=3 * module_count,
        config=config,
        runtime=runtime,
    )
    final_record = _shared_read_record(
        execution.final_output_read,
        read_kind="final_output",
        cycle=3,
        entry_count=3 * module_count,
    )
    public_trace = execution.generation.workspace_trace
    if type(public_trace) is not tuple or len(public_trace) != 3:
        message = "arm-C generation must preserve three public trace projections"
        raise ValueError(message)
    for projected, native in zip(public_trace, execution.shared_buffer_trace, strict=True):
        if (
            type(projected) is not WorkspaceState
            or not torch.equal(projected.slots, native.entries)
            or not torch.equal(projected.source_module, native.source_module)
            or not bool(projected.occupancy.all())
        ):
            message = "arm-C public trace is not a faithful native-state projection"
            raise ValueError(message)
    trace_hash_content = {
        "trace_contract": "append_only_shared_buffer_per_cycle_v1",
        "shared_buffer_policy_sha256": _POLICY_SHA256,
        "cycle_count": 3,
        "pre_update_read_count": 3,
        "final_read_count": 1,
        "entries_per_cycle": module_count,
        "final_entry_count": 3 * module_count,
        "latent_token_count": module_count,
        "cycles": [cycle.model_dump(mode="json") for cycle in cycles],
        "final_output_read": final_record.model_dump(mode="json"),
    }
    return SharedBufferTraceProvenanceV3(
        trace_contract="append_only_shared_buffer_per_cycle_v1",
        shared_buffer_policy_sha256=_POLICY_SHA256,
        cycle_count=3,
        pre_update_read_count=3,
        final_read_count=1,
        entries_per_cycle=module_count,
        final_entry_count=3 * module_count,
        latent_token_count=module_count,
        cycles=(cycles[0], cycles[1], cycles[2]),
        final_output_read=final_record,
        sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_hash_content)),
    )


def _validate_execution_receipt(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    execution: RequestExecutionOutputV3,
) -> None:
    receipt = execution.receipt
    if type(receipt) is not IrisRequestExecutionReceiptV1:
        message = "request artifact v3 requires an exact execution receipt"
        raise TypeError(message)
    prompt_bytes = _exact_utf8(prompt, name="prompt")
    if receipt.prompt_sha256 != domain_sha256(_EXECUTION_PROMPT_DOMAIN, prompt_bytes):
        message = "request artifact v3 prompt differs from the executed prompt"
        raise ValueError(message)
    if receipt.strategy != strategy:
        message = "request artifact v3 strategy differs from execution"
        raise ValueError(message)
    if receipt.stop_token_ids != stop_token_ids:
        message = "request artifact v3 stop-token policy differs from execution"
        raise ValueError(message)
    if receipt.sampling != sampling:
        message = "request artifact v3 sampling policy differs from execution"
        raise ValueError(message)
    if receipt.sampling_seed != sampling_seed:
        message = "request artifact v3 sampling seed differs from execution"
        raise ValueError(message)


def build_request_artifact_v3(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
    runtime: IrisArmRequestRuntimeV3,
    execution: RequestExecutionOutputV3,
) -> IrisRequestArtifactEnvelopeV3:
    """Build one validated A/B/C/D artifact without filesystem mutation."""
    try:
        if type(runtime) is not IrisArmRequestRuntimeV3:
            message = "request artifact v3 runtime must be exact IrisArmRequestRuntimeV3"
            raise TypeError(message)
        if type(sampling) is not TemperatureTopPSamplingConfig:
            message = "request artifact v3 sampling must be exact"
            raise TypeError(message)
        config, topology = _validate_runtime(runtime)
        if runtime.profile.arm_id == "C":
            if type(execution) is not IrisSharedBufferRequestExecutionOutputV1:
                message = "arm-C request artifact requires exact shared-buffer execution output"
                raise TypeError(message)
        elif type(execution) is not IrisRequestExecutionOutput:
            message = f"arm-{runtime.profile.arm_id} request artifact output has the wrong type"
            raise TypeError(message)
        _validate_execution_receipt(
            prompt=prompt,
            strategy=strategy,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            sampling_seed=sampling_seed,
            execution=execution,
        )
        if type(allow_download) is not bool or config.backbone.local_files_only is allow_download:
            message = "request artifact v3 download policy disagrees with configuration"
            raise ValueError(message)
        decoding = DecodingProvenanceV1(
            strategy=strategy,
            stop_token_ids=stop_token_ids,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            sampling_seed=sampling_seed,
            allow_download=allow_download,
        )
        prompt_bytes = _exact_utf8(prompt, name="prompt")
        execution_view = cast(IrisRequestExecutionOutput, execution)
        physical_tokens, attended_tokens = _validate_prompt_tokens(
            execution_view,
            device=runtime.device_info.device,
        )
        output = execution.generation
        _validate_generation(
            output,
            config=config,
            device=runtime.device_info.device,
            stop_token_ids=stop_token_ids,
        )
        if runtime.profile.arm_id == "C":
            assert isinstance(execution, IrisSharedBufferRequestExecutionOutputV1)
            trace: RequestWorkspaceTraceProvenanceV3 = _shared_buffer_trace(
                execution,
                config=config,
                runtime=runtime,
            )
        else:
            trace = _workspace_trace(output, config=config, runtime=runtime)
        payload = IrisRequestArtifactPayloadV3(
            protocol_declaration_payload_sha256=(
                "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
            ),
            execution_profile_set_version=4,
            execution_profile_set_sha256=_PROFILE_SET_SHA256,
            shared_buffer_policy_sha256=_POLICY_SHA256,
            topology=topology,
            configuration=ConfigurationProvenanceV1(
                value=config,
                sha256=runtime.configuration_sha256,
            ),
            request=RequestProvenanceV1(
                prompt=PrivateTextFingerprintV1(
                    representation="sha256_only",
                    encoding="utf-8",
                    normalization="none",
                    byte_length=len(prompt_bytes),
                    sha256=domain_sha256(_PROMPT_DOMAIN, prompt_bytes),
                ),
                prompt_input_ids=_tensor_fingerprint(
                    execution.prompt_input_ids,
                    domain="iris/request-artifact/v3/prompt-input-ids",
                ),
                prompt_attention_mask=_tensor_fingerprint(
                    execution.prompt_attention_mask,
                    domain="iris/request-artifact/v3/prompt-attention-mask",
                ),
                physical_prompt_tokens=physical_tokens,
                attended_prompt_tokens=attended_tokens,
                decoding=decoding,
            ),
            runtime=RequestRuntimeProvenanceV3(
                initialization_seed=runtime.seed,
                deterministic=runtime.deterministic,
                backend=runtime.device_info.backend,
                device=str(runtime.device_info.device),
                device_name=runtime.device_info.name,
                dtype=_SUPPORTED_RUNTIME_DTYPES[runtime.dtype],
                sidecar_state_sha256=runtime.sidecar_state_sha256,
                software=_software(),
            ),
            generation=GenerationProvenanceV1(
                text=_private_text(output.text, domain=_TEXT_DOMAIN),
                token_ids=_tensor_fingerprint(
                    output.token_ids,
                    domain="iris/request-artifact/v3/generated-token-ids",
                ),
                generated_tokens=output.token_ids.shape[1],
                stop_reason=output.stop_reason,
                matched_stop_token_id=output.matched_stop_token_id,
            ),
            workspace_trace=trace,
        )
        payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
        return IrisRequestArtifactEnvelopeV3(
            schema=RequestArtifactSchemaV3(
                artifact_type="iris.request",
                version=3,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_values_sha256_only",
            ),
            payload_sha256=domain_sha256(_PAYLOAD_DOMAIN, payload_bytes),
            payload=payload,
        )
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v3 could not be built"
        raise IrisRequestArtifactError(message) from error


def _content_without_sha256(model: Any) -> dict[str, Any]:
    return cast(dict[str, Any], model.model_dump(mode="json", exclude={"sha256"}))


def _validate_integrity(
    artifact: IrisRequestArtifactEnvelopeV3,
) -> tuple[IrisRequestArtifactEnvelopeV3, bytes]:
    owned = IrisRequestArtifactEnvelopeV3.model_validate(
        artifact.model_dump(mode="python", by_alias=True)
    )
    payload = owned.payload
    config_bytes = canonical_json_bytes(payload.configuration.value.model_dump(mode="json"))
    if payload.configuration.sha256 != domain_sha256(_CONFIGURATION_DOMAIN, config_bytes):
        message = "request artifact v3 configuration digest is inconsistent"
        raise ValueError(message)
    trace = payload.workspace_trace
    if isinstance(trace, WorkspaceTraceProvenanceV3):
        for workspace_cycle in trace.cycles:
            expected = domain_sha256(
                f"iris/request-artifact/v3/workspace/cycle/{workspace_cycle.cycle}",
                canonical_json_bytes(_content_without_sha256(workspace_cycle)),
            )
            if workspace_cycle.sha256 != expected:
                message = (
                    "request artifact v3 workspace cycle "
                    f"{workspace_cycle.cycle} digest is inconsistent"
                )
                raise ValueError(message)
        trace_content: object = [
            workspace_cycle.model_dump(mode="json") for workspace_cycle in trace.cycles
        ]
    elif isinstance(trace, SharedBufferTraceProvenanceV3):
        for shared_cycle in trace.cycles:
            read = shared_cycle.pre_update_read
            state = shared_cycle.post_cycle_state
            read_domain = (
                "iris/request-artifact/v3/shared-buffer/pre_update/"
                f"{shared_cycle.cycle}"
            )
            state_domain = (
                "iris/request-artifact/v3/shared-buffer/state/"
                f"{shared_cycle.cycle}"
            )
            if read.sha256 != domain_sha256(
                read_domain,
                canonical_json_bytes(_content_without_sha256(read)),
            ):
                message = (
                    f"request artifact v3 arm-C read {shared_cycle.cycle} digest is inconsistent"
                )
                raise ValueError(message)
            if state.sha256 != domain_sha256(
                state_domain,
                canonical_json_bytes(_content_without_sha256(state)),
            ):
                message = (
                    f"request artifact v3 arm-C state {shared_cycle.cycle} digest is inconsistent"
                )
                raise ValueError(message)
            if shared_cycle.sha256 != domain_sha256(
                f"iris/request-artifact/v3/shared-buffer/cycle/{shared_cycle.cycle}",
                canonical_json_bytes(_content_without_sha256(shared_cycle)),
            ):
                message = (
                    "request artifact v3 arm-C cycle "
                    f"{shared_cycle.cycle} digest is inconsistent"
                )
                raise ValueError(message)
        final_read = trace.final_output_read
        if final_read.sha256 != domain_sha256(
            "iris/request-artifact/v3/shared-buffer/final_output/3",
            canonical_json_bytes(_content_without_sha256(final_read)),
        ):
            message = "request artifact v3 arm-C final read digest is inconsistent"
            raise ValueError(message)
        trace_content = _content_without_sha256(trace)
    else:
        assert isinstance(trace, EmptyWorkspaceTraceProvenanceV3)
        trace_content = []
    if trace.sha256 != domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_content)):
        message = "request artifact v3 trace digest is inconsistent"
        raise ValueError(message)
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    if owned.payload_sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "request artifact v3 payload digest is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json", by_alias=True))
    return owned, content


def _prepare_content_addressed_parent_v3(path: Path) -> None:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if category_directory.name != "requests" or version_directory.name != "v3" or not root.name:
        message = "Request artifact v3 path must use the canonical <root>/requests/v3 layout"
        raise IrisRequestArtifactError(message)
    root.mkdir(parents=True, exist_ok=True)
    _require_real_directory(root, label="Request artifact root")
    for directory, label in (
        (category_directory, "Request artifact category directory"),
        (version_directory, "Request artifact version directory"),
    ):
        with suppress(FileExistsError):
            os.mkdir(directory, mode=0o700)
        _require_real_directory(directory, label=label)


def _install_content_addressed_v3(path: Path, content: bytes) -> bool:
    _prepare_content_addressed_parent_v3(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Existing request artifact v3 conflicts with its content-addressed digest"
        raise IrisRequestArtifactError(message)
    _prepare_content_addressed_parent_v3(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Concurrent request artifact v3 conflicts with its content-addressed digest"
        raise IrisRequestArtifactError(message)
    temporary_path = path.parent / f".iris-{uuid4().hex}.tmp"
    try:
        file_descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(file_descriptor, "wb", buffering=0) as stream:
                remaining = memoryview(content)
                while remaining:
                    written = stream.write(remaining)
                    if written is None or written <= 0:
                        message = "Request artifact v3 write made no forward progress"
                        raise OSError(message)
                    remaining = remaining[written:]
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        if temporary_path.read_bytes() != content:
            message = "Request artifact v3 temporary bytes failed verification"
            raise OSError(message)
        _prepare_content_addressed_parent_v3(path)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError as error:
            if not _existing_matches(path, content):
                message = "Concurrent request artifact v3 conflicts with its content digest"
                raise IrisRequestArtifactError(message) from error
            reused = True
        temporary_path.unlink()
        return reused
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def write_request_artifact_v3(
    artifact: IrisRequestArtifactEnvelopeV3,
) -> RequestArtifactWriteResult:
    """Install one v3 artifact at its deterministic content-addressed path."""
    try:
        if type(artifact) is not IrisRequestArtifactEnvelopeV3:
            message = "artifact writer requires exact IrisRequestArtifactEnvelopeV3"
            raise TypeError(message)
        owned, content = _validate_integrity(artifact)
        output_dir = Path(owned.payload.configuration.value.runtime.output_dir)
        path = output_dir / "requests" / "v3" / f"{owned.payload_sha256}.json"
        reused = _install_content_addressed_v3(path, content)
        return RequestArtifactWriteResult(
            path=path,
            payload_sha256=owned.payload_sha256,
            file_sha256=bytes_sha256(content),
            size_bytes=len(content),
            reused_existing=reused,
        )
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v3 could not be persisted"
        raise IrisRequestArtifactError(message) from error


def read_request_artifact_v3(path: Path) -> IrisRequestArtifactEnvelopeV3:
    """Read and verify one exact canonical request artifact v3."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "request artifact v3 reader requires an explicit file Path"
            raise TypeError(message)
        content = path.read_bytes()
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "request artifact v3 must be canonical UTF-8 JSON with one final LF"
            raise ValueError(message)
        decoded = content[:-1].decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number is forbidden: {value}")
            ),
        )
        if canonical_json_file_bytes(raw) != content:
            message = "request artifact v3 is not canonical iris-json-v1"
            raise ValueError(message)
        artifact = IrisRequestArtifactEnvelopeV3.model_validate_json(decoded)
        _, validated_content = _validate_integrity(artifact)
        if validated_content != content:
            message = "request artifact v3 canonical round trip changed its bytes"
            raise ValueError(message)
        return artifact
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v3 could not be read or verified"
        raise IrisRequestArtifactError(message) from error


__all__ = [
    "RequestExecutionOutputV3",
    "build_request_artifact_v3",
    "read_request_artifact_v3",
    "write_request_artifact_v3",
]
