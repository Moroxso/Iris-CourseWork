"""Strict, distinct persisted-export intents and human JSON, no native construction."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from iris.application import scientific_ablation_operator_export as export
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
    ScientificAblationBestPayloadV1,
    ScientificAblationExportRequestPayloadV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def _request() -> IrisScientificAblationExportRequestV1:
    payload = ScientificAblationExportRequestPayloadV1(
        operator_sha256="1" * 64,
        execution_sha256="2" * 64,
        checkpoint_event_sha256="3" * 64,
        lifecycle_sha256="4" * 64,
        source_checkpoint_sha256="5" * 64,
        selection="selected_best",
        purpose="smoke",
        maximum_storage_bytes=4 * 1024**2,
        minimum_free_bytes=0,
    )
    return IrisScientificAblationExportRequestV1(
        sha256=domain_sha256(
            export._REQUEST, canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_storage_bytes", True),
        ("maximum_storage_bytes", "1"),
        ("maximum_storage_bytes", 0),
        ("maximum_storage_bytes", 16 * 1024**3 + 1),
        ("minimum_free_bytes", False),
        ("minimum_free_bytes", -1),
        ("minimum_free_bytes", 16 * 1024**3 + 1),
        ("selection", "best"),
        ("selection", 1),
        ("purpose", "test"),
        ("operator_sha256", "A" * 64),
        ("checkpoint_event_sha256", "../checkpoint"),
        ("kind", "iris-scientific-ablation-phase-request-v1"),
        ("callback", "launch"),
    ],
)
def test_export_request_controls_are_exact_and_not_baseline_or_native_authority(
    field: str,
    value: Any,
) -> None:
    raw = _request().payload.model_dump(mode="python") | {field: value}
    with pytest.raises(ValidationError):
        ScientificAblationExportRequestPayloadV1.model_validate(raw)


def test_human_request_read_is_data_only_and_whitespace_tolerant(tmp_path: Path) -> None:
    request = _request()
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(request.model_dump(mode="json"), indent=2), encoding="utf-8")
    with patch.object(
        export,
        "create_scientific_ablation_command_run_v1",
        side_effect=AssertionError("No native factory"),
    ) as factory:
        assert export.read_scientific_ablation_export_request_v1(path) == request
    assert factory.call_count == 0


@pytest.mark.parametrize(
    "fault", ["duplicate", "nonfinite", "extra", "domain", "digest", "oversize", "boolean"]
)
def test_malformed_export_json_is_rejected_before_native_work(tmp_path: Path, fault: str) -> None:
    request = _request().model_dump(mode="json")
    raw = json.dumps(request)
    if fault == "duplicate":
        raw = '{"sha256":"' + "0" * 64 + '",' + raw[1:]
    elif fault == "nonfinite":
        raw = raw.replace('"minimum_free_bytes": 0', '"minimum_free_bytes": NaN')
    elif fault == "oversize":
        raw = " " * (64 * 1024 + 1) + raw
    else:
        if fault == "extra":
            request["extra"] = "not allowed"
        elif fault == "digest":
            request["sha256"] = "0" * 64
        elif fault == "domain":
            request["payload"]["kind"] = "iris-scientific-ablation-phase-request-v1"
        elif fault == "boolean":
            request["payload"]["minimum_free_bytes"] = False
        raw = json.dumps(request)
    path = tmp_path / "intent.json"
    path.write_text(raw, encoding="utf-8")
    with (
        patch.object(export, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.read_scientific_ablation_export_request_v1(path)
    assert factory.call_count == 0


@pytest.mark.parametrize("fault", ["nested_extra", "hidden_top", "mutated_digest", "subclass"])
def test_unchecked_request_objects_do_not_bypass_exact_tree_guard(fault: str) -> None:
    request = _request()
    if fault == "nested_extra":
        vars(request.payload)["callback"] = "not native authority"
    elif fault == "hidden_top":
        vars(request)["callback"] = "not native authority"
    elif fault == "mutated_digest":
        vars(request)["sha256"] = "0" * 64
    else:

        class Foreign(IrisScientificAblationExportRequestV1):
            pass

        request = Foreign.model_validate(request.model_dump(mode="python"))
    with pytest.raises(ValueError):
        export._checked_request(request)


def _best() -> dict[str, Any]:
    return dict(
        operator_sha256="1" * 64,
        execution_sha256="2" * 64,
        head_sha256="3" * 64,
        validated_best_event_sha256=None,
        validated_best_checkpoint_sha256=None,
        validated_best_step=None,
        persisted_checkpoint_event_sha256=None,
        persisted_lifecycle_sha256=None,
        persisted_best_checkpoint_sha256=None,
        persisted_best_step=None,
        validated_best_is_persisted=False,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"validated_best_step": 1},
        {"validated_best_is_persisted": True},
        {"persisted_lifecycle_sha256": "4" * 64},
        {"persisted_best_checkpoint_sha256": "5" * 64, "persisted_best_step": 1},
        {"validated_best_event_sha256": "6" * 64},
    ],
)
def test_best_metadata_cannot_claim_a_missing_physical_boundary(changes: dict[str, Any]) -> None:
    assert (
        ScientificAblationBestPayloadV1.model_validate(_best()).validated_best_is_persisted is False
    )
    with pytest.raises(ValidationError):
        ScientificAblationBestPayloadV1.model_validate(_best() | changes)
