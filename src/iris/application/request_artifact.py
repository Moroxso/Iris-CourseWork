"""Build, verify, and atomically persist privacy-preserving request artifacts."""

import json
import os
import platform
import stat
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from re import fullmatch
from typing import Any, Literal, Protocol
from uuid import uuid4

import torch

from iris import __version__
from iris.application.bootstrap import IrisRequestRuntime
from iris.application.errors import IrisRequestArtifactError
from iris.application.request import IrisRequestExecutionOutput
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
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)
from iris.workspace import WorkspaceState

_CONFIGURATION_DOMAIN = "iris/configuration/v1"
_PROMPT_DOMAIN = "iris/request-artifact/v1/prompt"
_TEXT_DOMAIN = "iris/request-artifact/v1/generated-text"
_PAYLOAD_DOMAIN = "iris/request-artifact/v1/payload"
_TRACE_DOMAIN = "iris/request-artifact/v1/workspace-trace"
type _TensorDtypeName = Literal["float16", "bfloat16", "float32", "int64", "bool"]
type _RuntimeDtypeName = Literal["float16", "bfloat16", "float32"]

_SUPPORTED_TENSOR_DTYPES: dict[torch.dtype, _TensorDtypeName] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_SUPPORTED_RUNTIME_DTYPES: dict[torch.dtype, _RuntimeDtypeName] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


class _RequestRuntimeConfigurationSource(Protocol):
    @property
    def configuration_json(self) -> str: ...

    @property
    def configuration_sha256(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def deterministic(self) -> bool: ...

    @property
    def device_info(self) -> Any: ...

    @property
    def dtype(self) -> torch.dtype: ...


@dataclass(frozen=True, slots=True)
class RequestArtifactWriteResult:
    """Immutable result of one content-addressed artifact installation."""

    path: Path
    payload_sha256: str
    file_sha256: str
    size_bytes: int
    reused_existing: bool


def _exact_utf8(value: object, *, name: str) -> bytes:
    if type(value) is not str:
        message = f"{name} must be an exact string"
        raise TypeError(message)
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        message = f"{name} must be valid scalar Unicode encodable as UTF-8"
        raise ValueError(message) from error


def _private_text(value: str, *, domain: str) -> PrivateTextFingerprintV1:
    encoded = _exact_utf8(value, name="private text")
    return PrivateTextFingerprintV1(
        representation="sha256_only",
        encoding="utf-8",
        normalization="none",
        byte_length=len(encoded),
        sha256=domain_sha256(domain, encoded),
    )


def _tensor_fingerprint(
    tensor: torch.Tensor,
    *,
    domain: str,
) -> TensorFingerprintV1:
    if type(tensor) is not torch.Tensor:
        message = "artifact tensor provenance requires exact torch.Tensor instances"
        raise TypeError(message)
    dtype_name = _SUPPORTED_TENSOR_DTYPES.get(tensor.dtype)
    if dtype_name is None:
        message = f"artifact tensor provenance does not support dtype {tensor.dtype}"
        raise ValueError(message)
    if tensor.device.type not in {"cpu", "cuda"}:
        message = "artifact tensor provenance supports only CPU and CUDA tensors"
        raise ValueError(message)
    if tensor.layout != torch.strided:
        message = "artifact tensor provenance requires strided tensors"
        raise ValueError(message)
    if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
        message = "artifact tensor provenance rejects NaN and Inf"
        raise ValueError(message)
    return TensorFingerprintV1(
        shape=tuple(tensor.shape),
        dtype=dtype_name,
        device=str(tensor.device),
        byte_length=tensor.numel() * tensor.element_size(),
        sha256=tensor_sha256(tensor, domain=domain),
    )


def _configuration(runtime: _RequestRuntimeConfigurationSource) -> IrisBootstrapConfig:
    if type(runtime.configuration_json) is not str:
        message = "runtime configuration provenance must be exact JSON text"
        raise TypeError(message)
    try:
        config = IrisBootstrapConfig.model_validate_json(runtime.configuration_json)
    except Exception as error:
        message = "runtime configuration provenance is not a valid IrisBootstrapConfig"
        raise ValueError(message) from error
    canonical = canonical_json_bytes(config.model_dump(mode="json"))
    if runtime.configuration_json.encode() != canonical:
        message = "runtime configuration provenance is not canonical iris-json-v1"
        raise ValueError(message)
    digest = domain_sha256(_CONFIGURATION_DOMAIN, canonical)
    if runtime.configuration_sha256 != digest:
        message = "runtime configuration SHA-256 does not match its canonical value"
        raise ValueError(message)
    if (
        runtime.seed != config.runtime.seed
        or runtime.deterministic is not config.runtime.deterministic
    ):
        message = "runtime policy disagrees with its validated configuration provenance"
        raise ValueError(message)
    if runtime.device_info.backend != config.hardware.device:
        message = "resolved runtime backend disagrees with configuration provenance"
        raise ValueError(message)
    expected_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[config.hardware.precision]
    if runtime.dtype != expected_dtype:
        message = "resolved runtime dtype disagrees with configuration provenance"
        raise ValueError(message)
    return config


def _validate_prompt_tokens(
    execution: IrisRequestExecutionOutput,
    *,
    device: torch.device,
) -> tuple[int, int]:
    input_ids = execution.prompt_input_ids
    attention_mask = execution.prompt_attention_mask
    for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask)):
        if (
            type(tensor) is not torch.Tensor
            or tensor.ndim != 2
            or tensor.shape[0] != 1
            or tensor.shape[1] <= 0
            or tensor.dtype != torch.int64
            or tensor.device != device
            or tensor.requires_grad
            or tensor.grad_fn is not None
        ):
            message = f"artifact prompt {name} provenance is invalid"
            raise ValueError(message)
    if input_ids.shape != attention_mask.shape:
        message = "artifact prompt token IDs and attention mask shapes disagree"
        raise ValueError(message)
    if bool((input_ids < 0).any()):
        message = "artifact prompt token IDs cannot be negative"
        raise ValueError(message)
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        message = "artifact prompt attention mask must contain only zero and one"
        raise ValueError(message)
    attended = int(attention_mask.sum().item())
    if attended <= 0:
        message = "artifact prompt provenance must contain an attended token"
        raise ValueError(message)
    return input_ids.shape[1], attended


def _validate_generation(
    output: IrisGeneratorOutput,
    *,
    config: IrisBootstrapConfig,
    device: torch.device,
    stop_token_ids: tuple[int, ...],
) -> tuple[int, ...]:
    if type(output) is not IrisGeneratorOutput or type(output.text) is not str:
        message = "artifact generation provenance requires exact IrisGeneratorOutput"
        raise TypeError(message)
    token_ids = output.token_ids
    if (
        type(token_ids) is not torch.Tensor
        or token_ids.ndim != 2
        or token_ids.shape[0] != 1
        or token_ids.shape[1] <= 0
        or token_ids.shape[1] > config.backbone.limits.max_new_tokens
        or token_ids.dtype != torch.int64
        or token_ids.device != device
    ):
        message = "artifact generated token tensor is invalid"
        raise ValueError(message)
    values = tuple(int(value) for value in token_ids[0].detach().to(device="cpu").tolist())
    if any(value < 0 for value in values):
        message = "artifact generated token IDs cannot be negative"
        raise ValueError(message)
    stop_set = set(stop_token_ids)
    if output.stop_reason == "stop_token":
        if (
            type(output.matched_stop_token_id) is not int
            or output.matched_stop_token_id not in stop_set
            or values[-1] != output.matched_stop_token_id
            or any(value in stop_set for value in values[:-1])
        ):
            message = "artifact stop-token generation provenance is inconsistent"
            raise ValueError(message)
    elif output.stop_reason == "max_new_tokens":
        if (
            output.matched_stop_token_id is not None
            or len(values) != config.backbone.limits.max_new_tokens
            or any(value in stop_set for value in values)
        ):
            message = "artifact maximum-length generation provenance is inconsistent"
            raise ValueError(message)
    else:
        message = "artifact generation stop reason is invalid"
        raise ValueError(message)
    return values


def _validate_workspace_state(
    state: WorkspaceState,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisRequestRuntime,
) -> None:
    if type(state) is not WorkspaceState:
        message = "artifact workspace cycles require exact WorkspaceState values"
        raise TypeError(message)
    slot_shape = (1, config.workspace.num_slots)
    tensors = {
        "slots": (state.slots, (*slot_shape, config.cognitive.state_dim), runtime.dtype),
        "age": (state.age, slot_shape, torch.int64),
        "occupancy": (state.occupancy, slot_shape, torch.bool),
        "source_module": (state.source_module, slot_shape, torch.int64),
        "confidence": (state.confidence, slot_shape, runtime.dtype),
        "ttl": (state.ttl, slot_shape, runtime.dtype),
        "write_count": (state.write_count, slot_shape, torch.int64),
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
            message = f"artifact workspace {name} tensor is invalid"
            raise ValueError(message)
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            message = f"artifact workspace {name} contains NaN or Inf"
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
    occupied_sources = state.source_module[state.occupancy]
    if not canonical_empty:
        message = "artifact workspace contains noncanonical empty slots"
        raise ValueError(message)
    if bool((state.age < 0).any()) or bool((state.write_count < 0).any()):
        message = "artifact workspace counters cannot be negative"
        raise ValueError(message)
    if bool((occupied_sources < 0).any()) or bool(
        (occupied_sources >= config.cognitive.num_modules).any()
    ):
        message = "artifact workspace contains an invalid occupied source module"
        raise ValueError(message)


def _workspace_trace(
    output: IrisGeneratorOutput,
    *,
    config: IrisBootstrapConfig,
    runtime: IrisRequestRuntime,
) -> WorkspaceTraceProvenanceV1:
    trace = output.workspace_trace
    if type(trace) is not tuple or len(trace) != config.workspace.cycles:
        message = "artifact workspace trace must preserve every configured cycle"
        raise ValueError(message)
    cycle_records: list[WorkspaceCycleProvenanceV1] = []
    for cycle, state in enumerate(trace, start=1):
        _validate_workspace_state(state, config=config, runtime=runtime)
        fingerprints = {
            name: _tensor_fingerprint(
                getattr(state, name),
                domain=f"iris/request-artifact/v1/workspace/cycle/{cycle}/{name}",
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
        cycle_records.append(
            WorkspaceCycleProvenanceV1(
                cycle=cycle,
                sha256=domain_sha256(
                    f"iris/request-artifact/v1/workspace/cycle/{cycle}",
                    canonical_json_bytes(cycle_content),
                ),
                **fingerprints,
            )
        )
    trace_content = [record.model_dump(mode="json") for record in cycle_records]
    return WorkspaceTraceProvenanceV1(
        cycle_count=len(cycle_records),
        cycles=tuple(cycle_records),
        sha256=domain_sha256(_TRACE_DOMAIN, canonical_json_bytes(trace_content)),
    )


def _software() -> SoftwareProvenanceV1:
    return SoftwareProvenanceV1(
        iris=__version__,
        python=platform.python_version(),
        torch=str(torch.__version__),
        transformers=version("transformers"),
        tokenizers=version("tokenizers"),
        safetensors=version("safetensors"),
        hydra_core=version("hydra-core"),
        omegaconf=version("omegaconf"),
    )


def build_request_artifact_v1(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
    runtime: IrisRequestRuntime,
    execution: IrisRequestExecutionOutput,
) -> IrisRequestArtifactEnvelopeV1:
    """Build a fully validated in-memory artifact without filesystem mutation."""
    try:
        if type(runtime) is not IrisRequestRuntime:
            message = "artifact runtime must be exact IrisRequestRuntime"
            raise TypeError(message)
        if type(execution) is not IrisRequestExecutionOutput:
            message = "artifact execution must be exact IrisRequestExecutionOutput"
            raise TypeError(message)
        config = _configuration(runtime)
        if type(allow_download) is not bool or config.backbone.local_files_only is allow_download:
            message = "artifact download policy disagrees with effective configuration"
            raise ValueError(message)
        if (
            type(sampling) is not TemperatureTopPSamplingConfig
            or type(sampling.temperature) is not float
            or not isfinite(sampling.temperature)
            or sampling.temperature <= 0.0
            or type(sampling.top_p) is not float
            or not isfinite(sampling.top_p)
            or not 0.0 < sampling.top_p <= 1.0
        ):
            message = "artifact sampling policy is invalid"
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
        current_sidecar_sha256 = runtime.orchestrator.sidecar_state_sha256
        if (
            type(runtime.sidecar_state_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", runtime.sidecar_state_sha256) is None
            or current_sidecar_sha256 != runtime.sidecar_state_sha256
        ):
            message = "artifact sidecar provenance changed after generation"
            raise ValueError(message)
        workspace_trace = _workspace_trace(output, config=config, runtime=runtime)
        payload = IrisRequestArtifactPayloadV1(
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
                    domain="iris/request-artifact/v1/prompt-input-ids",
                ),
                prompt_attention_mask=_tensor_fingerprint(
                    execution.prompt_attention_mask,
                    domain="iris/request-artifact/v1/prompt-attention-mask",
                ),
                physical_prompt_tokens=physical_tokens,
                attended_prompt_tokens=attended_tokens,
                decoding=decoding,
            ),
            runtime=RuntimeProvenanceV1(
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
                    domain="iris/request-artifact/v1/generated-token-ids",
                ),
                generated_tokens=output.token_ids.shape[1],
                stop_reason=output.stop_reason,
                matched_stop_token_id=output.matched_stop_token_id,
            ),
            workspace_trace=workspace_trace,
        )
        payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
        return IrisRequestArtifactEnvelopeV1(
            schema=SchemaIdentityV1(
                artifact_type="iris.request",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
            ),
            payload_sha256=domain_sha256(_PAYLOAD_DOMAIN, payload_bytes),
            payload=payload,
        )
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact could not be built"
        raise IrisRequestArtifactError(message) from error


def _cycle_content(cycle: WorkspaceCycleProvenanceV1) -> dict[str, Any]:
    return cycle.model_dump(mode="json", exclude={"sha256"})


def _validate_integrity(artifact: IrisRequestArtifactEnvelopeV1) -> bytes:
    owned = IrisRequestArtifactEnvelopeV1.model_validate(
        artifact.model_dump(mode="python", by_alias=True)
    )
    payload = owned.payload
    config_bytes = canonical_json_bytes(payload.configuration.value.model_dump(mode="json"))
    if payload.configuration.sha256 != domain_sha256(_CONFIGURATION_DOMAIN, config_bytes):
        message = "artifact configuration digest is inconsistent"
        raise ValueError(message)
    for cycle in payload.workspace_trace.cycles:
        expected = domain_sha256(
            f"iris/request-artifact/v1/workspace/cycle/{cycle.cycle}",
            canonical_json_bytes(_cycle_content(cycle)),
        )
        if cycle.sha256 != expected:
            message = f"artifact workspace cycle {cycle.cycle} digest is inconsistent"
            raise ValueError(message)
    trace_content = [cycle.model_dump(mode="json") for cycle in payload.workspace_trace.cycles]
    if payload.workspace_trace.sha256 != domain_sha256(
        _TRACE_DOMAIN,
        canonical_json_bytes(trace_content),
    ):
        message = "artifact workspace trace digest is inconsistent"
        raise ValueError(message)
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    if owned.payload_sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "artifact payload digest is inconsistent"
        raise ValueError(message)
    return canonical_json_file_bytes(owned.model_dump(mode="json", by_alias=True))


def _path_is_link_or_reparse(path: Path) -> bool:
    """Return whether one existing path entry redirects through a link/reparse point."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


def _path_entry_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        message = f"{label} was not created"
        raise IrisRequestArtifactError(message) from error
    if _path_is_link_or_reparse(path) or not stat.S_ISDIR(metadata.st_mode):
        message = f"{label} must be a real directory, not a link or reparse point"
        raise IrisRequestArtifactError(message)


def _prepare_content_addressed_parent(path: Path) -> None:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if (
        category_directory.name != "requests"
        or version_directory.name not in {"v1", "v2"}
        or not root.name
    ):
        message = "Request artifact path must use the canonical <root>/requests/vN layout"
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


def _existing_matches(path: Path, expected: bytes) -> bool:
    if _path_is_link_or_reparse(path):
        return False
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        return False
    return path.read_bytes() == expected


def _install_content_addressed(path: Path, content: bytes) -> bool:
    _prepare_content_addressed_parent(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Existing request artifact conflicts with its content-addressed digest"
        raise IrisRequestArtifactError(message)
    _prepare_content_addressed_parent(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Concurrent request artifact conflicts with its content-addressed digest"
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
                        message = "Request artifact write made no forward progress"
                        raise OSError(message)
                    remaining = remaining[written:]
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        if temporary_path.read_bytes() != content:
            message = "Request artifact temporary bytes failed verification"
            raise OSError(message)
        _prepare_content_addressed_parent(path)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError as error:
            if not _existing_matches(path, content):
                message = "Concurrent request artifact conflicts with its content digest"
                raise IrisRequestArtifactError(message) from error
            reused = True
        temporary_path.unlink()
        return reused
    except BaseException as primary_error:
        try:
            temporary_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "Request artifact failure and temporary cleanup failure",
                [primary_error, cleanup_error],
            )
            message = "Iris request artifact failed and temporary cleanup was incomplete"
            raise IrisRequestArtifactError(message) from failures
        if isinstance(primary_error, IrisRequestArtifactError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris request artifact could not be installed atomically"
            raise IrisRequestArtifactError(message) from primary_error
        raise


def write_request_artifact_v1(
    artifact: IrisRequestArtifactEnvelopeV1,
) -> RequestArtifactWriteResult:
    """Install one validated artifact at its deterministic content-addressed path."""
    try:
        if type(artifact) is not IrisRequestArtifactEnvelopeV1:
            message = "artifact writer requires exact IrisRequestArtifactEnvelopeV1"
            raise TypeError(message)
        content = _validate_integrity(artifact)
        output_dir = Path(artifact.payload.configuration.value.runtime.output_dir)
        path = output_dir / "requests" / "v1" / f"{artifact.payload_sha256}.json"
        reused = _install_content_addressed(path, content)
        return RequestArtifactWriteResult(
            path=path,
            payload_sha256=artifact.payload_sha256,
            file_sha256=bytes_sha256(content),
            size_bytes=len(content),
            reused_existing=reused,
        )
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact could not be persisted"
        raise IrisRequestArtifactError(message) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        value[key] = item
    return value


def read_request_artifact_v1(path: Path) -> IrisRequestArtifactEnvelopeV1:
    """Read and verify one exact canonical request artifact v1."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "artifact reader requires an explicit file Path"
            raise TypeError(message)
        content = path.read_bytes()
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "artifact file must be UTF-8 canonical JSON with exactly one final LF"
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
            message = "artifact file is not canonical iris-json-v1"
            raise ValueError(message)
        artifact = IrisRequestArtifactEnvelopeV1.model_validate_json(decoded)
        if _validate_integrity(artifact) != content:
            message = "artifact canonical round trip changed its bytes"
            raise ValueError(message)
        return artifact
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact could not be read or verified"
        raise IrisRequestArtifactError(message) from error
