"""Full native writer reports returned ONLY after successful source/caller cleanup.

One authenticated complete transaction, one actual writer stage, original immediate
retention seals reaudited during construction. No scorer/model/factory callbacks,
second baseline, generation, publication or partial report/file escape on failure.
"""

from dataclasses import dataclass

import torch

from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
    ScientificAblationCausalInputV1,
    _Source,
    _source_transaction,
)
from iris.application.scientific_ablation_causal_receipts import _observed_json
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_writer_effects import (
    _audit_writer_measurements,
    _measure_writer_effects,
    _WriterMeasurements,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    CausalDesignPolicy,
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    IrisScientificAblationMessageWriterReportV1,
    IrisScientificAblationWorkspaceWriterReportV1,
    ScientificAblationMessageWriterReportPayloadV1,
    ScientificAblationWorkspaceWriterReportPayloadV1,
    ScientificAblationWriterCPURuntimeV1,
    ScientificAblationWriterOrdinaryRowV1,
    ScientificAblationWriterSourceMetadataV1,
    WriterPayload,
    WriterReport,
    _runtime,
    _source_metadata,
    build_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    IrisScientificAblationWriterSourceV1,
    build_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
)
from iris.utils.hashing import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkspaceWriterCompletedV1:
    """Detached data returned after cleanup, NOT a caller-proof execution token."""

    report: IrisScientificAblationWorkspaceWriterReportV1
    source: IrisScientificAblationWriterSourceV1
    runtime: ScientificAblationWriterCPURuntimeV1


@dataclass(frozen=True, slots=True)
class ScientificAblationMessageWriterCompletedV1:
    """Distinct message result; portable source does not attest native execution."""

    report: IrisScientificAblationMessageWriterReportV1
    source: IrisScientificAblationWriterSourceV1
    runtime: ScientificAblationWriterCPURuntimeV1


type WriterCompleted = (
    ScientificAblationWorkspaceWriterCompletedV1 | ScientificAblationMessageWriterCompletedV1
)


def _snapshot(source: _Source, measurements: _WriterMeasurements) -> dict[str, object]:
    """Detached ENTIRE reserved payload from original native measurements while BUSY."""
    _audit_writer_measurements(source, measurements)
    pf = source.complete_preflight
    assert pf is not None
    message = type(source.design.payload.policy) is ScientificAblationMessageCausalPolicyV1
    protocol = "message" if message else "workspace"
    ordinary = [
        {
            name: value
            for name, value in row.row.model_dump(mode="json").items()
            if name in ScientificAblationWriterOrdinaryRowV1.model_fields
        }
        for row in measurements.rows
    ]
    frames = [
        dict(
            row_index=entry.row_index,
            role=entry.role,
            write_cycle=entry.write_cycle,
            writer=entry.writer,
            slot=entry.slot,
            donor_row_index=entry.donor_row_index,
            donor_pair_sha256=entry.donor_pair_sha256,
            output=_observed_json(entry.output),
        )
        for entry in measurements.passes
    ]
    metadata = ScientificAblationWriterSourceMetadataV1.model_validate(_source_metadata(pf))
    result: dict[str, object] = dict(
        kind=f"iris-scientific-ablation-{protocol}-writer-report-v1",
        scope="complete-teacher-writer-effects-not-generated-answer-quality",
        scientific_acceptance="not_assessed",
        source_metadata=metadata.model_dump(mode="json"),
        byte_preflight=dict(
            layout_sha256=pf.layout_sha256,
            maximum_report_bytes=pf.maximum_report_bytes,
            report_byte_preflight_bound=pf.report_byte_preflight_bound,
        ),
        actual_teacher_executions=measurements.actual_teacher_executions,
        actual_trace_proposal_observation_scalar_bound=measurements.actual_trace_proposal_observation_scalar_bound,
        actual_post_intervention_scalar_bound=measurements.actual_post_intervention_scalar_bound,
        ordinary_rows=ordinary,
        controlled_passes=dict(
            empty=[entry for entry in frames if entry["role"] == "empty"],
            trials={
                role: [entry for entry in frames if entry["role"] == role]
                for role in (("block", "donor") if message else ("shuffle", "zero"))
            },
        ),
        effects=[effect.model_dump(mode="json") for effect in measurements.effects],
        statistics=measurements.statistics.model_dump(mode="json"),
    )
    _audit_writer_measurements(source, measurements)
    return result


def _measure_completed(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> WriterCompleted:
    # Capture the ACTUAL trusted runtime, not a caller's/report's claimed version.
    runtime = ScientificAblationWriterCPURuntimeV1(
        torch_version=str(torch.__version__), torch_git_version=torch.version.git_version
    )
    _runtime(runtime)  # A future stream change is denied BEFORE transaction/forwards.
    with _source_transaction(
        heldout,
        input=input,
        policy=policy,
        maximum_report_bytes=limits.maximum_report_bytes,
        complete_limits=limits,
    ) as source:
        measurements = _measure_writer_effects(source)
        pf = source.complete_preflight
        assert pf is not None
        detached = _snapshot(source, measurements)
        original_payload = canonical_json_bytes(detached)
        # Independently detach the ORIGINAL reservation before either builder.
        # A coherent replacement source/report cannot supply its own expectation.
        original_source_payload = canonical_json_bytes(
            dict(
                kind="iris-scientific-ablation-writer-source-v1",
                scope="original-reservation-not-execution-attestation",
                scientific_acceptance="not_assessed",
                protocol=(
                    "message"
                    if type(policy) is ScientificAblationMessageCausalPolicyV1
                    else "workspace"
                ),
                source_metadata=detached["source_metadata"],
                maximum_report_bytes=pf.maximum_report_bytes,
                layout_sha256=pf.layout_sha256,
                report_byte_preflight_bound=pf.report_byte_preflight_bound,
            )
        )
        portable = build_scientific_ablation_writer_source_v1(pf)
        if type(portable) is not IrisScientificAblationWriterSourceV1:
            raise ValueError("Exact detached original writer source required")
        source_sha256 = portable.sha256
        source_bytes = scientific_ablation_writer_source_bytes_v1(
            portable, expected_sha256=source_sha256
        )
        if (
            canonical_json_bytes(portable.payload.model_dump(mode="json"))
            != original_source_payload
        ):
            raise ValueError("Detached writer source differs from original native reservation")
        if type(policy) is ScientificAblationMessageCausalPolicyV1:
            payload: WriterPayload = ScientificAblationMessageWriterReportPayloadV1.model_validate(
                detached
            )
        else:
            payload = ScientificAblationWorkspaceWriterReportPayloadV1.model_validate(detached)
        result = build_scientific_ablation_writer_report_v1(payload, preflight=pf, runtime=runtime)
        encoded = scientific_ablation_writer_report_bytes_v1(result, preflight=pf, runtime=runtime)
        # ALL original seals are checked again while measuring=True/BUSY, not after READY.
        late = _snapshot(source, measurements)
        if (
            scientific_ablation_writer_report_bytes_v1(result, preflight=pf, runtime=runtime)
            != encoded
        ):
            raise ValueError("Owned report changed during final native audit")
        _audit_writer_measurements(source, measurements)
        if canonical_json_bytes(late) != original_payload or (
            canonical_json_bytes(result.payload.model_dump(mode="json")) != original_payload
        ):
            raise ValueError("Full writer report/native measurements changed during construction")
        if (
            scientific_ablation_writer_source_bytes_v1(portable, expected_sha256=source_sha256)
            != source_bytes
            or canonical_json_bytes(portable.payload.model_dump(mode="json"))
            != original_source_payload
        ):
            raise ValueError("Detached writer source changed during final native audit")
    # Late source/original-owner/caller RNG/modes/default-dtype restoration must succeed.
    # Never publish or return a partial result from inside the BUSY transaction.
    if type(result) is IrisScientificAblationWorkspaceWriterReportV1:
        return ScientificAblationWorkspaceWriterCompletedV1(result, portable, runtime)
    if type(result) is IrisScientificAblationMessageWriterReportV1:
        return ScientificAblationMessageWriterCompletedV1(result, portable, runtime)
    raise ValueError("Exact distinct completed native writer report required")


def _measure(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> WriterReport:
    return _measure_completed(heldout, input=input, policy=policy, limits=limits).report


def measure_scientific_ablation_workspace_writer_completed_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: ScientificAblationWorkspaceCausalPolicyV1,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> ScientificAblationWorkspaceWriterCompletedV1:
    if type(policy) is not ScientificAblationWorkspaceCausalPolicyV1:
        raise ValueError("Exact distinct workspace writer policy required")
    if type(limits) is not ScientificAblationCausalCompleteLimitsV1:
        raise ValueError("Exact complete writer limits required")
    result = _measure_completed(heldout, input=input, policy=policy, limits=limits)
    if type(result) is not ScientificAblationWorkspaceWriterCompletedV1:
        raise ValueError("Exact completed workspace writer result required")
    return result


def measure_scientific_ablation_message_writer_completed_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: ScientificAblationMessageCausalPolicyV1,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> ScientificAblationMessageWriterCompletedV1:
    if type(policy) is not ScientificAblationMessageCausalPolicyV1:
        raise ValueError("Exact distinct message writer policy required")
    if type(limits) is not ScientificAblationCausalCompleteLimitsV1:
        raise ValueError("Exact complete writer limits required")
    result = _measure_completed(heldout, input=input, policy=policy, limits=limits)
    if type(result) is not ScientificAblationMessageWriterCompletedV1:
        raise ValueError("Exact completed message writer result required")
    return result


def measure_scientific_ablation_workspace_writer_report_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: ScientificAblationWorkspaceCausalPolicyV1,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> IrisScientificAblationWorkspaceWriterReportV1:
    if type(policy) is not ScientificAblationWorkspaceCausalPolicyV1:
        raise ValueError("Exact distinct workspace writer policy required")
    if type(limits) is not ScientificAblationCausalCompleteLimitsV1:
        raise ValueError("Exact complete writer limits required")
    result = _measure(heldout, input=input, policy=policy, limits=limits)
    if type(result) is not IrisScientificAblationWorkspaceWriterReportV1:
        raise ValueError("Owned native workspace writer report required")
    return result


def measure_scientific_ablation_message_writer_report_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: ScientificAblationMessageCausalPolicyV1,
    limits: ScientificAblationCausalCompleteLimitsV1,
) -> IrisScientificAblationMessageWriterReportV1:
    if type(policy) is not ScientificAblationMessageCausalPolicyV1:
        raise ValueError("Exact distinct message writer policy required")
    if type(limits) is not ScientificAblationCausalCompleteLimitsV1:
        raise ValueError("Exact complete writer limits required")
    result = _measure(heldout, input=input, policy=policy, limits=limits)
    if type(result) is not IrisScientificAblationMessageWriterReportV1:
        raise ValueError("Owned native message writer report required")
    return result
