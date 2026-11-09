"""Build, verify, and atomically persist arm-labelled request artifacts v2."""

from __future__ import annotations

import json
from pathlib import Path
from re import fullmatch
from typing import Any, Literal, cast

import torch

from iris.application.arm_profiles import (
    IrisArmExecutionProfileV3,
    get_arm_execution_profile_v3,
)
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import IrisArmRequestRuntimeV2
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
    _install_content_addressed,
    _private_text,
    _reject_duplicate_keys,
    _software,
    _tensor_fingerprint,
    _validate_generation,
    _validate_prompt_tokens,
)
from iris.artifacts.request_v1 import (
    ConfigurationProvenanceV1,
    DecodingProvenanceV1,
    GenerationProvenanceV1,
    PrivateTextFingerprintV1,
    RequestProvenanceV1,
    WorkspaceCycleProvenanceV1,
)
from iris.artifacts.request_v2 import (
    EmptyWorkspaceTraceProvenanceV2,
    IrisRequestArtifactEnvelopeV2,
    IrisRequestArtifactPayloadV2,
    RequestArmATopologyV2,
    RequestArmBTopologyV2,
    RequestArmDTopologyV2,
    RequestArmTopologyV2,
    RequestArtifactSchemaV2,
    RequestRuntimeProvenanceV2,
    RequestWorkspaceTraceProvenanceV2,
    WorkspaceTraceProvenanceV2,
)
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)
from iris.workspace import WorkspaceState

_CONFIGURATION_DOMAIN = "iris/configuration/v1"
_PROMPT_DOMAIN = "iris/request-artifact/v2/prompt"
_TEXT_DOMAIN = "iris/request-artifact/v2/generated-text"
_PAYLOAD_DOMAIN = "iris/request-artifact/v2/payload"
_TRACE_DOMAIN = "iris/request-artifact/v2/workspace-trace"
_PROFILE_SET_SHA256: Literal["5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"] = (
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
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


def _validate_runtime(
    runtime: IrisArmRequestRuntimeV2,
) -> tuple[IrisBootstrapConfig, RequestArmTopologyV2]:
    profile = runtime.profile
    if type(profile) is not IrisArmExecutionProfileV3:
        message = "request runtime profile must be exact IrisArmExecutionProfileV3"
        raise TypeError(message)
    expected_profile = get_arm_execution_profile_v3(profile.arm_id)
    if profile != expected_profile:
        message = "request runtime execution profile differs from profile set v3"
        raise ValueError(message)
    if runtime.execution_profile_set_sha256 != _PROFILE_SET_SHA256:
        message = "request runtime execution profile set SHA-256 is not canonical"
        raise ValueError(message)
    config = _configuration(runtime)
    orchestrator = runtime.orchestrator
    expected_orchestrator_type: type[
        BaseLlmRequestOrchestrator
        | CognitiveNoWorkspaceRequestOrchestrator
        | IrisRequestOrchestrator
    ]
    if profile.arm_id == "A":
        expected_orchestrator_type = BaseLlmRequestOrchestrator
    elif profile.arm_id == "B":
        expected_orchestrator_type = CognitiveNoWorkspaceRequestOrchestrator
    else:
        expected_orchestrator_type = IrisRequestOrchestrator
    if type(orchestrator) is not expected_orchestrator_type:
        message = f"arm-{profile.arm_id} request runtime uses the wrong request graph"
        raise TypeError(message)
    if orchestrator.backbone is not runtime.backbone:
        message = "request runtime orchestrator does not own the exact runtime backbone"
        raise ValueError(message)

    if profile.arm_id == "A":
        if runtime.sidecar_state_sha256 is not None:
            message = "arm-A request runtime must use the prompt-only graph without a sidecar"
            raise ValueError(message)
        return config, RequestArmATopologyV2(
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
        message = f"arm-{profile.arm_id} request runtime requires a valid sidecar SHA-256"
        raise ValueError(message)
    sidecar_orchestrator = cast(
        CognitiveNoWorkspaceRequestOrchestrator | IrisRequestOrchestrator,
        orchestrator,
    )
    current_sidecar_sha256 = sidecar_orchestrator.sidecar_state_sha256
    if current_sidecar_sha256 != sidecar_sha256:
        message = f"arm-{profile.arm_id} sidecar provenance changed after execution"
        raise ValueError(message)
    if profile.arm_id == "B":
        return config, RequestArmBTopologyV2(
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
    return config, RequestArmDTopologyV2(
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


def _validate_workspace_state_v2(
    state: WorkspaceState,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV2,
) -> None:
    if type(state) is not WorkspaceState:
        message = "arm-D request artifact cycles require exact WorkspaceState values"
        raise TypeError(message)
    metadata_shape = (1, config.workspace.num_slots)
    tensors = {
        "slots": (
            state.slots,
            (*metadata_shape, config.cognitive.state_dim),
            runtime.dtype,
        ),
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
            or tensor.shape != shape
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
    canonical_empty = (
        bool((state.slots[empty] == 0).all())
        and bool((state.age[empty] == 0).all())
        and bool((state.source_module[empty] == -1).all())
        and bool((state.confidence[empty] == 0).all())
        and bool((state.ttl[empty] == 0).all())
        and bool((state.write_count[empty] == 0).all())
    )
    if not canonical_empty:
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


def _workspace_trace_v2(
    output: IrisGeneratorOutput,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisArmRequestRuntimeV2,
) -> RequestWorkspaceTraceProvenanceV2:
    trace = output.workspace_trace
    if type(trace) is not tuple:
        message = "request artifact workspace trace must be an exact tuple"
        raise TypeError(message)
    if runtime.profile.arm_id in {"A", "B"}:
        if trace:
            message = f"arm-{runtime.profile.arm_id} request artifact requires an empty trace"
            raise ValueError(message)
        empty_content: list[object] = []
        return EmptyWorkspaceTraceProvenanceV2(
            trace_contract="empty_workspace_trace",
            cycle_count=0,
            cycles=(),
            sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(empty_content)),
        )
    if len(trace) != config.workspace.cycles:
        message = "arm-D request artifact must preserve every configured workspace cycle"
        raise ValueError(message)
    records: list[WorkspaceCycleProvenanceV1] = []
    for cycle, state in enumerate(trace, start=1):
        _validate_workspace_state_v2(state, config=config, runtime=runtime)
        fingerprints = {
            name: _tensor_fingerprint(
                getattr(state, name),
                domain=f"iris/request-artifact/v2/workspace/cycle/{cycle}/{name}",
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
                    f"iris/request-artifact/v2/workspace/cycle/{cycle}",
                    canonical_json_bytes(cycle_content),
                ),
                **fingerprints,
            )
        )
    trace_content = [record.model_dump(mode="json") for record in records]
    return WorkspaceTraceProvenanceV2(
        trace_contract="workspace_state_per_cycle",
        cycle_count=len(records),
        cycles=tuple(records),
        sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_content)),
    )


def _validate_execution_receipt_v2(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    execution: IrisRequestExecutionOutput,
) -> None:
    receipt = execution.receipt
    if type(receipt) is not IrisRequestExecutionReceiptV1:
        message = "request artifact v2 requires an exact execution receipt"
        raise TypeError(message)
    prompt_bytes = _exact_utf8(prompt, name="prompt")
    expected_prompt_sha256 = domain_sha256(_EXECUTION_PROMPT_DOMAIN, prompt_bytes)
    if receipt.prompt_sha256 != expected_prompt_sha256:
        message = "request artifact v2 prompt differs from the executed prompt"
        raise ValueError(message)
    if receipt.strategy != strategy:
        message = "request artifact v2 strategy differs from the executed strategy"
        raise ValueError(message)
    if receipt.stop_token_ids != stop_token_ids:
        message = "request artifact v2 stop-token policy differs from execution"
        raise ValueError(message)
    if receipt.sampling != sampling:
        message = "request artifact v2 sampling policy differs from execution"
        raise ValueError(message)
    if receipt.sampling_seed != sampling_seed:
        message = "request artifact v2 sampling seed differs from execution"
        raise ValueError(message)


def build_request_artifact_v2(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
    runtime: IrisArmRequestRuntimeV2,
    execution: IrisRequestExecutionOutput,
) -> IrisRequestArtifactEnvelopeV2:
    """Build one validated A/B/D request artifact without filesystem mutation."""
    try:
        if type(runtime) is not IrisArmRequestRuntimeV2:
            message = "request artifact v2 runtime must be exact IrisArmRequestRuntimeV2"
            raise TypeError(message)
        if type(execution) is not IrisRequestExecutionOutput:
            message = "request artifact v2 execution must be exact IrisRequestExecutionOutput"
            raise TypeError(message)
        if type(sampling) is not TemperatureTopPSamplingConfig:
            message = "request artifact v2 sampling must be exact TemperatureTopPSamplingConfig"
            raise TypeError(message)
        _validate_execution_receipt_v2(
            prompt=prompt,
            strategy=strategy,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            sampling_seed=sampling_seed,
            execution=execution,
        )
        config, topology = _validate_runtime(runtime)
        if type(allow_download) is not bool or config.backbone.local_files_only is allow_download:
            message = "request artifact v2 download policy disagrees with configuration"
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
        physical_tokens, attended_tokens = _validate_prompt_tokens(
            execution,
            device=runtime.device_info.device,
        )
        output = execution.generation
        _validate_generation(
            output,
            config=config,
            device=runtime.device_info.device,
            stop_token_ids=stop_token_ids,
        )
        workspace_trace = _workspace_trace_v2(
            output,
            config=config,
            runtime=runtime,
        )
        payload = IrisRequestArtifactPayloadV2(
            protocol_declaration_payload_sha256=(
                "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
            ),
            execution_profile_set_version=3,
            execution_profile_set_sha256=_PROFILE_SET_SHA256,
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
                    domain="iris/request-artifact/v2/prompt-input-ids",
                ),
                prompt_attention_mask=_tensor_fingerprint(
                    execution.prompt_attention_mask,
                    domain="iris/request-artifact/v2/prompt-attention-mask",
                ),
                physical_prompt_tokens=physical_tokens,
                attended_prompt_tokens=attended_tokens,
                decoding=decoding,
            ),
            runtime=RequestRuntimeProvenanceV2(
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
                    domain="iris/request-artifact/v2/generated-token-ids",
                ),
                generated_tokens=output.token_ids.shape[1],
                stop_reason=output.stop_reason,
                matched_stop_token_id=output.matched_stop_token_id,
            ),
            workspace_trace=workspace_trace,
        )
        payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
        return IrisRequestArtifactEnvelopeV2(
            schema=RequestArtifactSchemaV2(
                artifact_type="iris.request",
                version=2,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_text_sha256_only",
            ),
            payload_sha256=domain_sha256(_PAYLOAD_DOMAIN, payload_bytes),
            payload=payload,
        )
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v2 could not be built"
        raise IrisRequestArtifactError(message) from error


def _cycle_content(cycle: WorkspaceCycleProvenanceV1) -> dict[str, Any]:
    return cycle.model_dump(mode="json", exclude={"sha256"})


def _validate_integrity(
    artifact: IrisRequestArtifactEnvelopeV2,
) -> tuple[IrisRequestArtifactEnvelopeV2, bytes]:
    owned = IrisRequestArtifactEnvelopeV2.model_validate(
        artifact.model_dump(mode="python", by_alias=True)
    )
    payload = owned.payload
    config_bytes = canonical_json_bytes(payload.configuration.value.model_dump(mode="json"))
    if payload.configuration.sha256 != domain_sha256(_CONFIGURATION_DOMAIN, config_bytes):
        message = "request artifact v2 configuration digest is inconsistent"
        raise ValueError(message)
    trace = payload.workspace_trace
    for cycle in trace.cycles:
        expected = domain_sha256(
            f"iris/request-artifact/v2/workspace/cycle/{cycle.cycle}",
            canonical_json_bytes(_cycle_content(cycle)),
        )
        if cycle.sha256 != expected:
            message = f"request artifact v2 workspace cycle {cycle.cycle} digest is inconsistent"
            raise ValueError(message)
    trace_content = [cycle.model_dump(mode="json") for cycle in trace.cycles]
    if trace.sha256 != domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_content)):
        message = "request artifact v2 workspace trace digest is inconsistent"
        raise ValueError(message)
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    if owned.payload_sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "request artifact v2 payload digest is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json", by_alias=True))
    return owned, content


def write_request_artifact_v2(
    artifact: IrisRequestArtifactEnvelopeV2,
) -> RequestArtifactWriteResult:
    """Install one v2 artifact at its deterministic arm-labelled request path."""
    try:
        if type(artifact) is not IrisRequestArtifactEnvelopeV2:
            message = "artifact writer requires exact IrisRequestArtifactEnvelopeV2"
            raise TypeError(message)
        owned, content = _validate_integrity(artifact)
        output_dir = Path(owned.payload.configuration.value.runtime.output_dir)
        path = output_dir / "requests" / "v2" / f"{owned.payload_sha256}.json"
        reused = _install_content_addressed(path, content)
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
        message = "Iris request artifact v2 could not be persisted"
        raise IrisRequestArtifactError(message) from error


def read_request_artifact_v2(path: Path) -> IrisRequestArtifactEnvelopeV2:
    """Read and verify one exact canonical request artifact v2."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "request artifact v2 reader requires an explicit file Path"
            raise TypeError(message)
        content = path.read_bytes()
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "request artifact v2 must be canonical UTF-8 JSON with one final LF"
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
            message = "request artifact v2 is not canonical iris-json-v1"
            raise ValueError(message)
        artifact = IrisRequestArtifactEnvelopeV2.model_validate_json(decoded)
        _, validated_content = _validate_integrity(artifact)
        if validated_content != content:
            message = "request artifact v2 canonical round trip changed its bytes"
            raise ValueError(message)
        return artifact
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v2 could not be read or verified"
        raise IrisRequestArtifactError(message) from error


__all__ = [
    "build_request_artifact_v2",
    "read_request_artifact_v2",
    "write_request_artifact_v2",
]
