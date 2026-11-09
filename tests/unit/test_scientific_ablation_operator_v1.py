"""Strict distinct operator metadata and bounded request IO, never native authority."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iris.application.scientific_ablation_operator import (
    IrisScientificAblationOperatorError,
    read_scientific_ablation_phase_request_v1,
)
from iris.artifacts.scientific_ablation_operator_v1 import (
    IrisScientificAblationPhaseRequestV1,
    ScientificAblationOperatorPayloadV1,
    ScientificAblationPhaseRequestPayloadV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def _request() -> IrisScientificAblationPhaseRequestV1:
    payload = ScientificAblationPhaseRequestPayloadV1(
        operator_sha256="a" * 64,
        execution_sha256="b" * 64,
        expected_previous_sha256="c" * 64,
        phase="train",
        purpose="smoke",
    )
    return IrisScientificAblationPhaseRequestV1(
        sha256=domain_sha256(
            "iris/scientific-ablation-phase-request/v1",
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "validate"),
        ("phase", True),
        ("phase", "stopped"),
        ("purpose", "approved"),
        ("purpose", 1),
        ("operator_sha256", "A" * 64),
        ("execution_sha256", "b" * 63),
        ("expected_previous_sha256", "current"),
        ("kind", "iris-scientific-phase-request-v1"),
        ("native_owner", "restored"),
    ],
)
def test_strict_phase_request_rejects_foreign_fields_and_coercion(field: str, value: Any) -> None:
    raw = _request().payload.model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ValidationError):
        ScientificAblationPhaseRequestPayloadV1.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("maximum", [0, -1, True, "1024", 512 * 1024**2 + 1])
def test_descriptor_parameter_ceiling_is_not_a_native_or_unbounded_capability(maximum: Any) -> None:
    with pytest.raises(ValidationError):
        ScientificAblationOperatorPayloadV1.model_validate_json(
            json.dumps(
                dict(
                    recipe_json="{}",
                    recipe_sha256="a" * 64,
                    maximum_trainable_parameter_bytes=maximum,
                    execution_sha256="b" * 64,
                    initial_checkpoint_sha256="c" * 64,
                )
            )
        )


def test_bounded_request_reader_preserves_pretty_json_and_exact_intent(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    request = _request()
    path.write_text(json.dumps(request.model_dump(mode="json"), indent=2), encoding="utf-8")
    assert read_scientific_ablation_phase_request_v1(path) == request


@pytest.mark.parametrize("fault", ["duplicate", "oversize", "digest", "domain", "extra", "nan"])
def test_request_reader_rejects_ambiguous_or_foreign_metadata(tmp_path: Path, fault: str) -> None:
    request = _request()
    value = request.model_dump(mode="json")
    if fault == "digest":
        value["sha256"] = "0" * 64
    elif fault == "domain":
        value["sha256"] = domain_sha256(
            "iris/scientific-phase-request/v1", canonical_json_bytes(value["payload"])
        )
    elif fault == "extra":
        value["native_owner"] = "restore"
    raw = canonical_json_bytes(value)
    if fault == "duplicate":
        raw = b'{"sha256":"' + value["sha256"].encode() + b'",' + raw[1:]
    elif fault == "oversize":
        raw += b" " * 65536
    elif fault == "nan":
        raw = raw[:-1] + b',"native_owner":NaN}'
    path = tmp_path / "request.json"
    path.write_bytes(raw)
    with pytest.raises(IrisScientificAblationOperatorError):
        read_scientific_ablation_phase_request_v1(path)
