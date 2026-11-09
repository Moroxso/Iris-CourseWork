"""Portable original writer reservation; metadata consistency, not execution proof.

No private layout objects, tensors, model/factory callbacks or CPU self-attestation
are transported. A reader needs an independently trusted source digest; accepting
a candidate's own digest proves only internal consistency. The source-owned
measurement service must separately authenticate physical inputs and native work.
"""

import json
from re import fullmatch
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    MAX_REPORT_BYTES,
    ScientificAblationCausalReportBytePreflightV1,
    build_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_writer_observations_v1 import _raw_model
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterSourceMetadataV1,
    _nonfinite,
    _pairs,
    _preflight,
    _source_metadata,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

type WriterSourceProtocol = Literal["workspace", "message"]


def _domain(protocol: WriterSourceProtocol) -> str:
    return f"iris/scientific-ablation-{protocol}-writer-source/v1"


def _digest(value: object) -> str:
    if type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Exact independently trusted writer source digest required")
    return value


def _protocol(pf: ScientificAblationCausalReportBytePreflightV1) -> WriterSourceProtocol:
    policy = pf.observed_budget.payload.design.payload.policy
    if type(policy) is ScientificAblationWorkspaceCausalPolicyV1:
        return "workspace"
    if type(policy) is ScientificAblationMessageCausalPolicyV1:
        return "message"
    raise ValueError("Exact distinct original writer source policy required")


def _rebuild(
    metadata: ScientificAblationWriterSourceMetadataV1, maximum: int
) -> ScientificAblationCausalReportBytePreflightV1:
    return build_scientific_ablation_causal_report_byte_preflight_v1(
        metadata.observed_budget,
        evaluation_plan=metadata.evaluation_plan,
        evaluation=metadata.evaluation,
        evaluation_complete=metadata.evaluation_complete,
        run_seed=metadata.run_seed,
        ordinary_teacher_graph_sha256=metadata.ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=metadata.controlled_teacher_graph_sha256,
        sidecar_sha256=metadata.sidecar_sha256,
        maximum_report_bytes=maximum,
    )


class _SourceWire(_Strict):
    @model_validator(mode="before")
    @classmethod
    def exact_source(cls, value: object) -> object:
        if not any(cls is declared for declared in _SOURCE_TYPES):
            raise ValueError("Exact declared portable writer source class required")
        return _raw_model(value, cls)


class ScientificAblationWriterSourcePayloadV1(_SourceWire):
    kind: Literal["iris-scientific-ablation-writer-source-v1"]
    scope: Literal["original-reservation-not-execution-attestation"]
    scientific_acceptance: Literal["not_assessed"]
    protocol: WriterSourceProtocol
    source_metadata: ScientificAblationWriterSourceMetadataV1
    maximum_report_bytes: int = Field(ge=1, le=MAX_REPORT_BYTES)
    layout_sha256: Digest
    report_byte_preflight_bound: int = Field(ge=1, le=MAX_REPORT_BYTES)

    @model_validator(mode="after")
    def original_reservation(self) -> Self:
        pf = _rebuild(self.source_metadata, self.maximum_report_bytes)
        if (
            self.protocol != _protocol(pf)
            or self.layout_sha256 != pf.layout_sha256
            or self.report_byte_preflight_bound != pf.report_byte_preflight_bound
        ):
            raise ValueError("Portable source differs from its complete original reservation")
        return self


class IrisScientificAblationWriterSourceV1(_SourceWire):
    sha256: Digest
    payload: ScientificAblationWriterSourcePayloadV1

    @model_validator(mode="after")
    def source_fingerprint(self) -> Self:
        expected = domain_sha256(
            _domain(self.payload.protocol),
            canonical_json_bytes(self.payload.model_dump(mode="json")),
        )
        if self.sha256 != expected:
            raise ValueError("Portable writer source domain digest differs")
        return self


_SOURCE_TYPES = (ScientificAblationWriterSourcePayloadV1, IrisScientificAblationWriterSourceV1)


def build_scientific_ablation_writer_source_v1(
    preflight: ScientificAblationCausalReportBytePreflightV1,
) -> IrisScientificAblationWriterSourceV1:
    """Own original metadata; the caller still supplies its physical source trust."""
    pf = _preflight(preflight)
    payload = ScientificAblationWriterSourcePayloadV1.model_validate(
        dict(
            kind="iris-scientific-ablation-writer-source-v1",
            scope="original-reservation-not-execution-attestation",
            scientific_acceptance="not_assessed",
            protocol=_protocol(pf),
            source_metadata=ScientificAblationWriterSourceMetadataV1.model_validate(
                _source_metadata(pf)
            ),
            maximum_report_bytes=pf.maximum_report_bytes,
            layout_sha256=pf.layout_sha256,
            report_byte_preflight_bound=pf.report_byte_preflight_bound,
        )
    )
    return IrisScientificAblationWriterSourceV1.model_validate(
        dict(
            sha256=domain_sha256(
                _domain(payload.protocol), canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
    )


def _owned_source(
    source: IrisScientificAblationWriterSourceV1, expected_sha256: str
) -> IrisScientificAblationWriterSourceV1:
    expected = _digest(expected_sha256)
    owned = IrisScientificAblationWriterSourceV1.model_validate(
        _raw_model(source, IrisScientificAblationWriterSourceV1)
    )
    if owned.sha256 != expected:
        raise ValueError("Portable source differs from independently trusted writer source")
    return owned


def scientific_ablation_writer_source_bytes_v1(
    source: IrisScientificAblationWriterSourceV1, *, expected_sha256: str
) -> bytes:
    owned = _owned_source(source, expected_sha256)
    content = canonical_json_file_bytes(owned.model_dump(mode="json"))
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Portable writer source exceeds the existing complete storage bound")
    return content


def scientific_ablation_writer_source_preflight_v1(
    source: IrisScientificAblationWriterSourceV1, *, expected_sha256: str
) -> ScientificAblationCausalReportBytePreflightV1:
    owned = _owned_source(source, expected_sha256)
    return _rebuild(owned.payload.source_metadata, owned.payload.maximum_report_bytes)


def read_scientific_ablation_writer_source_v1(
    content: bytes, *, expected_sha256: str, maximum_source_bytes: int
) -> IrisScientificAblationWriterSourceV1:
    _digest(expected_sha256)
    if type(maximum_source_bytes) is not int or not 1 <= maximum_source_bytes <= MAX_REPORT_BYTES:
        raise ValueError("Exact bounded caller writer source byte cap required")
    if type(content) is not bytes or not 1 <= len(content) <= maximum_source_bytes:
        raise ValueError("Exact bounded portable writer source bytes required")
    depth = 0
    quoted = escaped = False
    for byte in content:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 40:
                raise ValueError("Portable writer source JSON depth exceeded")
        elif byte in (93, 125):
            depth -= 1
    try:
        raw = json.loads(
            content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Valid exact UTF8 portable writer source JSON required") from error
    value = IrisScientificAblationWriterSourceV1.model_validate(raw)
    if (
        scientific_ablation_writer_source_bytes_v1(value, expected_sha256=expected_sha256)
        != content
    ):
        raise ValueError("Canonical portable writer source bytes required")
    return value
