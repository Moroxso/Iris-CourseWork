"""Build and persist checkpoint-bound profile-v4 request artifacts v4."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from iris.application.arm_request_checkpoint import (
    IrisArmRequestCheckpointTransferResultV1,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV3
from iris.application.errors import IrisRequestArtifactError
from iris.application.request_artifact import (
    RequestArtifactWriteResult,
    _existing_matches,
    _path_entry_exists,
    _reject_duplicate_keys,
    _require_real_directory,
)
from iris.application.request_artifact_v3 import (
    RequestExecutionOutputV3,
    build_request_artifact_v3,
)
from iris.application.request_artifact_v3 import (
    _validate_integrity as _validate_integrity_v3,
)
from iris.artifacts.request_v3 import (
    IrisRequestArtifactEnvelopeV3,
    IrisRequestArtifactPayloadV3,
    RequestArtifactSchemaV3,
)
from iris.artifacts.request_v4 import (
    IrisRequestArtifactEnvelopeV4,
    IrisRequestArtifactPayloadV4,
    RequestArmBCheckpointTransferV4,
    RequestArmCCheckpointTransferV4,
    RequestArmCheckpointTransferV4,
    RequestArmDCheckpointTransferV4,
    RequestArtifactSchemaV4,
)
from iris.generation import DecodingStrategy, TemperatureTopPSamplingConfig
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
)

_PAYLOAD_DOMAIN = "iris/request-artifact/v4/payload"

type RequestExecutionOutputV4 = RequestExecutionOutputV3


def _checkpoint_transfer_provenance(
    *,
    runtime: IrisArmRequestRuntimeV3,
    transfer: IrisArmRequestCheckpointTransferResultV1 | None,
) -> RequestArmCheckpointTransferV4 | None:
    arm_id = runtime.profile.arm_id
    if arm_id == "A":
        if transfer is not None:
            message = "arm A request artifact v4 cannot carry checkpoint transfer"
            raise ValueError(message)
        return None
    if type(transfer) is not IrisArmRequestCheckpointTransferResultV1:
        message = f"arm {arm_id} request artifact v4 requires an exact checkpoint transfer"
        raise TypeError(message)
    if transfer.runtime is not runtime:
        message = "request artifact v4 must use the exact runtime returned by checkpoint transfer"
        raise ValueError(message)
    identity = transfer.checkpoint_identity
    if identity.arm_id != arm_id:
        message = "request runtime and checkpoint transfer arms differ"
        raise ValueError(message)
    if transfer.source_teacher_sidecar_sha256 != identity.sidecar_state_sha256:
        message = "checkpoint source sidecar hash differs from checkpoint identity"
        raise ValueError(message)
    if transfer.completed_steps != identity.completed_steps:
        message = "checkpoint transfer step count differs from checkpoint identity"
        raise ValueError(message)
    if transfer.installed_request_sidecar_sha256 != runtime.sidecar_state_sha256:
        message = "checkpoint transfer and request runtime sidecar hashes differ"
        raise ValueError(message)
    fields = dict(
        transfer_contract=transfer.transfer_contract,
        checkpoint_payload_sha256=identity.payload_sha256,
        checkpoint_state_file_sha256=identity.state_file_sha256,
        source_teacher_sidecar_sha256=transfer.source_teacher_sidecar_sha256,
        request_sidecar_state_sha256=transfer.installed_request_sidecar_sha256,
        completed_steps=transfer.completed_steps,
        transferred_state_names=transfer.state_names,
        transferred_tensor_count=transfer.transferred_tensor_count,
    )
    if arm_id == "B":
        return RequestArmBCheckpointTransferV4.model_validate(
            {"arm_id": "B", "checkpoint_schema_version": 2, **fields}
        )
    if arm_id == "C":
        return RequestArmCCheckpointTransferV4.model_validate(
            {"arm_id": "C", "checkpoint_schema_version": 3, **fields}
        )
    return RequestArmDCheckpointTransferV4.model_validate(
        {"arm_id": "D", "checkpoint_schema_version": 1, **fields}
    )


def build_request_artifact_v4(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    sampling_seed: int | None,
    allow_download: bool,
    runtime: IrisArmRequestRuntimeV3,
    execution: RequestExecutionOutputV4,
    checkpoint_transfer: IrisArmRequestCheckpointTransferResultV1 | None,
) -> IrisRequestArtifactEnvelopeV4:
    """Build one validated request artifact with exact checkpoint provenance."""
    try:
        if type(runtime) is not IrisArmRequestRuntimeV3:
            message = "request artifact v4 runtime must be exact IrisArmRequestRuntimeV3"
            raise TypeError(message)
        transfer = _checkpoint_transfer_provenance(
            runtime=runtime,
            transfer=checkpoint_transfer,
        )
        base = build_request_artifact_v3(
            prompt=prompt,
            strategy=strategy,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            sampling_seed=sampling_seed,
            allow_download=allow_download,
            runtime=runtime,
            execution=execution,
        )
        payload_content = base.payload.model_dump(mode="python")
        payload_content["checkpoint_transfer"] = transfer
        payload = IrisRequestArtifactPayloadV4.model_validate(payload_content)
        payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
        return IrisRequestArtifactEnvelopeV4(
            schema=RequestArtifactSchemaV4(
                artifact_type="iris.request",
                version=4,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_values_sha256_only",
            ),
            payload_sha256=domain_sha256(_PAYLOAD_DOMAIN, payload_bytes),
            payload=payload,
        )
    except Exception as error:
        message = "Iris request artifact v4 could not be built"
        raise IrisRequestArtifactError(message) from error


def _validate_v3_projection(payload: IrisRequestArtifactPayloadV4) -> None:
    content = payload.model_dump(mode="python", exclude={"checkpoint_transfer"})
    projected = IrisRequestArtifactPayloadV3.model_validate(content)
    projected_bytes = canonical_json_bytes(projected.model_dump(mode="json"))
    envelope = IrisRequestArtifactEnvelopeV3(
        schema=RequestArtifactSchemaV3(
            artifact_type="iris.request",
            version=3,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            content_policy="private_values_sha256_only",
        ),
        payload_sha256=domain_sha256("iris/request-artifact/v3/payload", projected_bytes),
        payload=projected,
    )
    _validate_integrity_v3(envelope)


def _validate_integrity(
    artifact: IrisRequestArtifactEnvelopeV4,
) -> tuple[IrisRequestArtifactEnvelopeV4, bytes]:
    owned = IrisRequestArtifactEnvelopeV4.model_validate(
        artifact.model_dump(mode="python", by_alias=True)
    )
    _validate_v3_projection(owned.payload)
    payload_bytes = canonical_json_bytes(owned.payload.model_dump(mode="json"))
    if owned.payload_sha256 != domain_sha256(_PAYLOAD_DOMAIN, payload_bytes):
        message = "request artifact v4 payload digest is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json", by_alias=True))
    return owned, content


def _prepare_content_addressed_parent_v4(path: Path) -> None:
    version_directory = path.parent
    category_directory = version_directory.parent
    root = category_directory.parent
    if category_directory.name != "requests" or version_directory.name != "v4" or not root.name:
        message = "Request artifact v4 path must use the canonical <root>/requests/v4 layout"
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


def _install_content_addressed_v4(path: Path, content: bytes) -> bool:
    _prepare_content_addressed_parent_v4(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Existing request artifact v4 conflicts with its content-addressed digest"
        raise IrisRequestArtifactError(message)
    _prepare_content_addressed_parent_v4(path)
    if _existing_matches(path, content):
        return True
    if _path_entry_exists(path):
        message = "Concurrent request artifact v4 conflicts with its content-addressed digest"
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
                        message = "Request artifact v4 write made no forward progress"
                        raise OSError(message)
                    remaining = remaining[written:]
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        if temporary_path.read_bytes() != content:
            message = "Request artifact v4 temporary bytes failed verification"
            raise OSError(message)
        _prepare_content_addressed_parent_v4(path)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError as error:
            if not _existing_matches(path, content):
                message = "Concurrent request artifact v4 conflicts with its content digest"
                raise IrisRequestArtifactError(message) from error
            reused = True
        temporary_path.unlink()
        return reused
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def write_request_artifact_v4(
    artifact: IrisRequestArtifactEnvelopeV4,
) -> RequestArtifactWriteResult:
    """Install one v4 artifact at its deterministic content-addressed path."""
    try:
        if type(artifact) is not IrisRequestArtifactEnvelopeV4:
            message = "artifact writer requires exact IrisRequestArtifactEnvelopeV4"
            raise TypeError(message)
        owned, content = _validate_integrity(artifact)
        output_dir = Path(owned.payload.configuration.value.runtime.output_dir)
        path = output_dir / "requests" / "v4" / f"{owned.payload_sha256}.json"
        reused = _install_content_addressed_v4(path, content)
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
        message = "Iris request artifact v4 could not be persisted"
        raise IrisRequestArtifactError(message) from error


def read_request_artifact_v4(path: Path) -> IrisRequestArtifactEnvelopeV4:
    """Read and verify one exact canonical request artifact v4."""
    try:
        if not isinstance(path, Path) or not path.name:
            message = "request artifact v4 reader requires an explicit file Path"
            raise TypeError(message)
        content = path.read_bytes()
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "request artifact v4 must be canonical UTF-8 JSON with one final LF"
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
            message = "request artifact v4 is not canonical iris-json-v1"
            raise ValueError(message)
        artifact = IrisRequestArtifactEnvelopeV4.model_validate_json(decoded)
        _, validated_content = _validate_integrity(artifact)
        if validated_content != content:
            message = "request artifact v4 canonical round trip changed its bytes"
            raise ValueError(message)
        return artifact
    except IrisRequestArtifactError:
        raise
    except Exception as error:
        message = "Iris request artifact v4 could not be read or verified"
        raise IrisRequestArtifactError(message) from error


__all__ = [
    "RequestExecutionOutputV4",
    "build_request_artifact_v4",
    "read_request_artifact_v4",
    "write_request_artifact_v4",
]
