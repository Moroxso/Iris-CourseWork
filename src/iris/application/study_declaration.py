"""Canonical Iris-GW-0.1 four-arm protocol declaration I/O."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from iris.application.errors import IrisFourArmProtocolError
from iris.artifacts.study_v1 import (
    FourArmArchitectureConstantsV1,
    FourArmDefinitionV1,
    FourArmMechanisticCriterionV1,
    FourArmProtocolPayloadV1,
    FourArmProtocolSchemaV1,
    FourArmProtocolSourceV1,
    IrisFourArmProtocolDeclarationV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/four-arm-protocol-declaration/v1/payload"
_MAX_DECLARATION_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class IrisFourArmProtocolWriteResult:
    """Identity of one canonical four-arm protocol declaration file."""

    path: Path
    payload_sha256: str
    size_bytes: int
    reused_existing: bool


def _build_payload() -> FourArmProtocolPayloadV1:
    return FourArmProtocolPayloadV1(
        protocol_id="iris-gw-0.1-four-arm",
        research_question="limited_competitive_workspace_advantage",
        source=FourArmProtocolSourceV1(
            conversation_id="6a748d3d-5864-83ed-9d15-3cd16ea8580e",
            arm_definition_turn_id="0b2e3f67-511d-40f1-a652-167abd8c1d58",
            arm_definition_message_id="f6e454fa-171c-4240-83d9-79be11765cf5",
            final_specification_turn_id="fb2807c0-b18c-47fd-b950-88784c477c8f",
            final_specification_message_id="cfb7cc81-ece7-4241-8ba1-889cc51997b9",
            final_specification_recovery="prefix_only_20000_characters",
        ),
        architecture=FourArmArchitectureConstantsV1(
            main_backbone_repository="Qwen/Qwen3-1.7B",
            debug_backbone_repository="Qwen/Qwen3-0.6B",
            cognitive_module_count=8,
            cognitive_state_width=256,
            limited_workspace_slot_count=8,
            limited_workspace_winners_per_cycle=2,
            cognitive_cycle_count=3,
            fast_weights_enabled=False,
            ciel_enabled=False,
            snn_enabled=False,
        ),
        arms=(
            FourArmDefinitionV1(arm_id="A", arm_kind="base_llm"),
            FourArmDefinitionV1(arm_id="B", arm_kind="llm_modules_no_workspace"),
            FourArmDefinitionV1(
                arm_id="C",
                arm_kind="llm_modules_unlimited_shared_buffer",
            ),
            FourArmDefinitionV1(
                arm_id="D",
                arm_kind="llm_modules_limited_competitive_workspace",
            ),
        ),
        task_families=(
            "split_information",
            "delayed_planning",
            "code_generation_and_verification",
            "hidden_goal_retention",
            "module_conflict",
        ),
        mechanistic_criterion=FourArmMechanisticCriterionV1(
            treatment_arm="D",
            control_arms=("B", "C"),
            relation="strictly_outperforms",
            interpretation="supports_competition_and_global_broadcast_usefulness",
            quantitative_rule_status="not_recovered",
        ),
        unresolved_items=(
            "arm_a_execution_semantics",
            "arm_b_execution_semantics",
            "arm_c_execution_semantics",
            "unlimited_shared_buffer_semantics",
            "main_backbone_pinned_revision",
            "dataset_construction_and_split_policy",
            "task_specific_evaluation_protocol",
            "seeds_repeats_epochs_and_optimizer",
            "comparison_aggregation_uncertainty_and_threshold",
            "study_storage_budget",
        ),
        execution_status="blocked_until_explicit_protocol_and_arm_implementations",
    )


def _validate_integrity(
    declaration: IrisFourArmProtocolDeclarationV1,
) -> tuple[IrisFourArmProtocolDeclarationV1, bytes]:
    if type(declaration) is not IrisFourArmProtocolDeclarationV1:
        message = "declaration must be exactly IrisFourArmProtocolDeclarationV1"
        raise TypeError(message)
    owned = IrisFourArmProtocolDeclarationV1.model_validate(declaration.model_dump(mode="python"))
    payload_bytes = canonical_json_bytes(owned.payload.model_dump(mode="json"))
    expected_sha256 = domain_sha256(_PAYLOAD_DOMAIN, payload_bytes)
    if owned.payload_sha256 != expected_sha256:
        message = "four-arm protocol payload SHA-256 is inconsistent"
        raise ValueError(message)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > _MAX_DECLARATION_BYTES:
        message = "four-arm protocol declaration exceeds the 1 MiB safety limit"
        raise ValueError(message)
    return owned, content


def build_four_arm_protocol_declaration_v1() -> IrisFourArmProtocolDeclarationV1:
    """Build the exact recovered protocol facts without inventing launch values."""
    try:
        payload = _build_payload()
        declaration = IrisFourArmProtocolDeclarationV1(
            schema_identity=FourArmProtocolSchemaV1(
                artifact_type="iris.four_arm_protocol_declaration",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                declaration_scope="recovered_facts_and_explicit_gaps",
            ),
            payload_sha256=domain_sha256(
                _PAYLOAD_DOMAIN,
                canonical_json_bytes(payload.model_dump(mode="json")),
            ),
            payload=payload,
        )
        owned, _ = _validate_integrity(declaration)
        return owned
    except IrisFourArmProtocolError:
        raise
    except Exception as error:
        message = "Iris four-arm protocol declaration could not be built"
        raise IrisFourArmProtocolError(message) from error


def validate_four_arm_protocol_declaration_v1(
    declaration: IrisFourArmProtocolDeclarationV1,
) -> IrisFourArmProtocolDeclarationV1:
    """Return one deeply owned declaration after schema, hash, and size validation."""
    try:
        owned, _ = _validate_integrity(declaration)
        return owned
    except IrisFourArmProtocolError:
        raise
    except Exception as error:
        message = "Iris four-arm protocol declaration could not be verified"
        raise IrisFourArmProtocolError(message) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            message = f"duplicate JSON key is forbidden: {key}"
            raise ValueError(message)
        value[key] = item
    return value


def read_four_arm_protocol_declaration_v1(
    path: Path,
) -> IrisFourArmProtocolDeclarationV1:
    """Read and verify one canonical local four-arm protocol declaration."""
    try:
        if not isinstance(path, Path) or path.suffix != ".json" or not path.name:
            message = "four-arm protocol reader requires an explicit .json file Path"
            raise TypeError(message)
        if path.is_symlink() or not path.is_file():
            message = "four-arm protocol path must be a real regular file"
            raise ValueError(message)
        size = path.stat().st_size
        if size <= 0 or size > _MAX_DECLARATION_BYTES:
            message = "four-arm protocol file size is invalid"
            raise ValueError(message)
        content = path.read_bytes()
        if len(content) != size:
            message = "four-arm protocol size changed while reading"
            raise ValueError(message)
        if (
            not content.endswith(b"\n")
            or content.endswith(b"\n\n")
            or content.startswith(b"\xef\xbb\xbf")
        ):
            message = "four-arm protocol must be canonical UTF-8 JSON with one final LF"
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
            message = "four-arm protocol file is not canonical iris-json-v1"
            raise ValueError(message)
        declaration = IrisFourArmProtocolDeclarationV1.model_validate_json(decoded)
        owned, expected_content = _validate_integrity(declaration)
        if expected_content != content:
            message = "four-arm protocol canonical round trip changed its bytes"
            raise ValueError(message)
        return owned
    except IrisFourArmProtocolError:
        raise
    except Exception as error:
        message = "Iris four-arm protocol declaration could not be read or verified"
        raise IrisFourArmProtocolError(message) from error


def _existing_matches(path: Path, expected: bytes) -> bool:
    try:
        return path.read_bytes() == expected
    except (FileNotFoundError, OSError):
        return False


def _install_no_clobber(path: Path, content: bytes) -> bool:
    if path.exists():
        if path.is_symlink() or not path.is_file() or not _existing_matches(path, content):
            message = "Existing four-arm protocol conflicts with expected bytes"
            raise IrisFourArmProtocolError(message)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
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
                        message = "Four-arm protocol write made no forward progress"
                        raise OSError(message)
                    remaining = remaining[written:]
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        if temporary_path.read_bytes() != content:
            message = "Four-arm protocol temporary bytes failed verification"
            raise OSError(message)
        try:
            os.link(temporary_path, path)
            reused = False
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or not _existing_matches(path, content):
                message = "Concurrent four-arm protocol conflicts with expected bytes"
                raise IrisFourArmProtocolError(message) from error
            reused = True
        temporary_path.unlink()
        return reused
    except BaseException as primary_error:
        try:
            temporary_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            failures = BaseExceptionGroup(
                "Four-arm protocol persistence and cleanup both failed",
                [primary_error, cleanup_error],
            )
            message = "Iris four-arm protocol failed and cleanup was incomplete"
            raise IrisFourArmProtocolError(message) from failures
        if isinstance(primary_error, IrisFourArmProtocolError):
            raise
        if isinstance(primary_error, Exception):
            message = "Iris four-arm protocol could not be installed atomically"
            raise IrisFourArmProtocolError(message) from primary_error
        raise


def write_four_arm_protocol_declaration_v1(
    declaration: IrisFourArmProtocolDeclarationV1,
    *,
    path: Path,
) -> IrisFourArmProtocolWriteResult:
    """Write one explicit protocol declaration atomically without overwrite."""
    try:
        if not isinstance(path, Path) or path.suffix != ".json" or not path.name:
            message = "four-arm protocol writer requires an explicit .json file Path"
            raise TypeError(message)
        owned, content = _validate_integrity(declaration)
        reused = _install_no_clobber(path, content)
        return IrisFourArmProtocolWriteResult(
            path=path,
            payload_sha256=owned.payload_sha256,
            size_bytes=len(content),
            reused_existing=reused,
        )
    except IrisFourArmProtocolError:
        raise
    except Exception as error:
        message = "Iris four-arm protocol declaration could not be persisted"
        raise IrisFourArmProtocolError(message) from error


__all__ = [
    "IrisFourArmProtocolWriteResult",
    "build_four_arm_protocol_declaration_v1",
    "read_four_arm_protocol_declaration_v1",
    "validate_four_arm_protocol_declaration_v1",
    "write_four_arm_protocol_declaration_v1",
]
