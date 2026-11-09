"""Portable reservation COMPONENT checks from pinned native metadata DATA.

These do not measure a staged model or authenticate physical/native execution.
The main full regression has a different frozen source tree and remains separate.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.artifacts import scientific_ablation_writer_source_v1 as module
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    MAX_REPORT_BYTES,
    ScientificAblationCausalReportBytePreflightV1,
    build_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterSourceMetadataV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/writer-source"
PINS = {
    "workspace": "7a94fe54012b09839e4e3f43146a2ba9e2ca3a72da54f47f30a7e1c085ad2a35",
    "message": "a00833de5615c22335aa41d67d7292f4f71ea18f99758ef2e6552dc4359c20e1",
}


@pytest.fixture(scope="module", params=("workspace", "message"))
def original(request: pytest.FixtureRequest) -> ScientificAblationCausalReportBytePreflightV1:
    name = str(request.param)
    content = (FIXTURES / (name + ".json")).read_bytes()
    assert sha256(content).hexdigest() == PINS[name]
    saved: dict[str, Any] = json.loads(content)
    source = ScientificAblationWriterSourceMetadataV1.model_validate(saved["source_metadata"])
    value = build_scientific_ablation_causal_report_byte_preflight_v1(
        source.observed_budget,
        evaluation_plan=source.evaluation_plan,
        evaluation=source.evaluation,
        evaluation_complete=source.evaluation_complete,
        run_seed=source.run_seed,
        ordinary_teacher_graph_sha256=source.ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=source.controlled_teacher_graph_sha256,
        sidecar_sha256=source.sidecar_sha256,
        maximum_report_bytes=saved["maximum_report_bytes"],
    )
    assert value.layout_sha256 == saved["layout_sha256"]
    assert value.report_byte_preflight_bound == saved["report_byte_preflight_bound"]
    return value


@pytest.fixture(scope="module")
def source(
    original: ScientificAblationCausalReportBytePreflightV1,
) -> module.IrisScientificAblationWriterSourceV1:
    return module.build_scientific_ablation_writer_source_v1(original)


def encoded(source: module.IrisScientificAblationWriterSourceV1) -> bytes:
    return module.scientific_ablation_writer_source_bytes_v1(source, expected_sha256=source.sha256)


def rehash(data: dict[str, Any]) -> bytes:
    data["sha256"] = domain_sha256(
        "iris/scientific-ablation-" + data["payload"]["protocol"] + "-writer-source/v1",
        canonical_json_bytes(data["payload"]),
    )
    return canonical_json_file_bytes(data)


def test_whole_original_reservation_roundtrip_without_shape_transport(
    original: ScientificAblationCausalReportBytePreflightV1,
    source: module.IrisScientificAblationWriterSourceV1,
) -> None:
    content = encoded(source)
    restored = module.read_scientific_ablation_writer_source_v1(
        content, expected_sha256=source.sha256, maximum_source_bytes=len(content)
    )
    rebuilt = module.scientific_ablation_writer_source_preflight_v1(
        restored, expected_sha256=source.sha256
    )
    assert rebuilt == original
    assert encoded(restored) == content
    assert set(json.loads(content)["payload"]) == {
        "kind",
        "scope",
        "scientific_acceptance",
        "protocol",
        "source_metadata",
        "maximum_report_bytes",
        "layout_sha256",
        "report_byte_preflight_bound",
    }
    assert b'"_shape"' not in content and b'"complete_layout"' not in content
    assert "runtime" not in type(restored.payload).model_fields
    assert "torch_version" not in type(restored.payload).model_fields
    assert restored.payload.scientific_acceptance == "not_assessed"
    geometry = rebuilt.observed_budget.payload.design.payload.geometry
    assert (geometry.num_modules, geometry.num_slots, geometry.state_dim) == (64, 1, 1)


def test_coherent_foreign_reservation_cannot_supply_its_own_expectation(
    original: ScientificAblationCausalReportBytePreflightV1,
    source: module.IrisScientificAblationWriterSourceV1,
) -> None:
    metadata = source.payload.source_metadata
    foreign_pf = build_scientific_ablation_causal_report_byte_preflight_v1(
        metadata.observed_budget,
        evaluation_plan=metadata.evaluation_plan,
        evaluation=metadata.evaluation,
        evaluation_complete=metadata.evaluation_complete,
        run_seed=metadata.run_seed,
        ordinary_teacher_graph_sha256=metadata.ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=metadata.controlled_teacher_graph_sha256,
        sidecar_sha256=metadata.sidecar_sha256,
        maximum_report_bytes=original.maximum_report_bytes - 1,
    )
    foreign = module.build_scientific_ablation_writer_source_v1(foreign_pf)
    assert foreign.sha256 != source.sha256
    body = encoded(foreign)
    with pytest.raises(ValueError, match="independently trusted"):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256=source.sha256, maximum_source_bytes=len(body)
        )
    with pytest.raises(ValueError, match="independently trusted"):
        module.scientific_ablation_writer_source_preflight_v1(
            foreign, expected_sha256=source.sha256
        )


@pytest.mark.parametrize("field", ("layout_sha256", "report_byte_preflight_bound", "protocol"))
def test_root_rehash_does_not_waive_original_reservation(
    source: module.IrisScientificAblationWriterSourceV1, field: str
) -> None:
    data = json.loads(encoded(source))
    if field == "layout_sha256":
        data["payload"][field] = "f" * 64
    elif field == "protocol":
        data["payload"][field] = (
            "message" if source.payload.protocol == "workspace" else "workspace"
        )
    else:
        data["payload"][field] += 1
    body = rehash(data)
    with pytest.raises(ValueError, match="reservation"):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256=data["sha256"], maximum_source_bytes=len(body)
        )


@pytest.mark.parametrize("extra", ("_shape", "complete_layout", "runtime", "torch_version"))
def test_private_layout_and_claimed_runtime_fields_denied(
    source: module.IrisScientificAblationWriterSourceV1, extra: str
) -> None:
    data = json.loads(encoded(source))
    data["payload"][extra] = "candidate-owned"
    body = rehash(data)
    with pytest.raises(ValueError):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256=data["sha256"], maximum_source_bytes=len(body)
        )


def test_byte_budget_precedes_json_allocation(
    source: module.IrisScientificAblationWriterSourceV1,
) -> None:
    body = encoded(source)
    with (
        patch.object(json, "loads", side_effect=AssertionError("No over-budget parsing")),
        pytest.raises(ValueError, match="bounded portable"),
    ):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256=source.sha256, maximum_source_bytes=len(body) - 1
        )


@pytest.mark.parametrize("cap", (False, 0, -1, MAX_REPORT_BYTES + 1, 1.0, "1024"))
def test_exact_positive_caller_byte_cap_required(cap: Any) -> None:
    with pytest.raises(ValueError, match="caller writer source byte cap"):
        module.read_scientific_ablation_writer_source_v1(
            b"{}", expected_sha256="0" * 64, maximum_source_bytes=cap
        )


@pytest.mark.parametrize("expected", (None, False, "F" * 64, "x", b"0" * 64))
def test_independent_digest_is_exact_before_candidate_access(expected: Any) -> None:
    class Trap:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("Candidate attributes must not be touched")

    candidate: Any = Trap()
    with pytest.raises(ValueError, match="trusted writer source digest"):
        module.scientific_ablation_writer_source_bytes_v1(candidate, expected_sha256=expected)


def test_foreign_root_attributes_not_observed() -> None:
    class Trap:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("No foreign root callback")

    candidate: Any = Trap()
    with pytest.raises(ValueError, match="Exact declared"):
        module.scientific_ablation_writer_source_bytes_v1(candidate, expected_sha256="0" * 64)


@pytest.mark.parametrize("field", ("hidden", "private", "extra", "fields"))
def test_hidden_constructed_model_metadata_denied(
    source: module.IrisScientificAblationWriterSourceV1, field: str
) -> None:
    broken = source.model_copy(deep=True)
    if field == "hidden":
        broken.__dict__["hidden"] = 1
    elif field == "fields":
        object.__setattr__(broken, "__pydantic_fields_set__", {"sha256", "payload", "hidden"})
    else:
        object.__setattr__(broken, "__pydantic_" + field + "__", {"hidden": 1})
    with pytest.raises(ValueError):
        module.scientific_ablation_writer_source_bytes_v1(broken, expected_sha256=source.sha256)


@pytest.mark.parametrize("body", (b"\xff", b"{}", b"[]", b'{"a":1,"a":2}', b'{"x":NaN}'))
def test_invalid_nonfinite_duplicate_json_denied(body: bytes) -> None:
    with pytest.raises(ValueError):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256="0" * 64, maximum_source_bytes=1024
        )


def test_depth_bound_precedes_parser() -> None:
    body = b"[" * 41 + b"0" + b"]" * 41
    with (
        patch.object(json, "loads", side_effect=AssertionError("No over-deep parse")),
        pytest.raises(ValueError, match="depth"),
    ):
        module.read_scientific_ablation_writer_source_v1(
            body, expected_sha256="0" * 64, maximum_source_bytes=1024
        )


@pytest.mark.parametrize("change", ("space", "extra_lf", "missing_lf", "bom"))
def test_noncanonical_bytes_denied(
    source: module.IrisScientificAblationWriterSourceV1, change: str
) -> None:
    body = encoded(source)
    changed = {
        "space": b" " + body,
        "extra_lf": body + b"\n",
        "missing_lf": body[:-1],
        "bom": b"\xef\xbb\xbf" + body,
    }[change]
    with pytest.raises(ValueError):
        module.read_scientific_ablation_writer_source_v1(
            changed, expected_sha256=source.sha256, maximum_source_bytes=len(changed)
        )
