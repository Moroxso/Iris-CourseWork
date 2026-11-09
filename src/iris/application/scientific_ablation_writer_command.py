"""Cold file/CLI delivery of complete writer DATA using independent job anchors.

Never measure, construct/load/call a model, score or execute candidate Python.
Original source/CPU/report expectations are caller-trusted, not file self-attestation.
Native measurement remains a separate completed-result API after source cleanup.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from iris.application import scientific_ablation_writer_io as io
from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _path,
    _read_file,
    _regular,
    _signature,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    read_scientific_ablation_writer_report_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    WriterSourceProtocol,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_preflight_v1,
)


class IrisScientificAblationWriterCommandError(ApplicationError):
    """A bounded writer file command could not complete; no private input details."""


@dataclass(frozen=True, slots=True)
class ScientificAblationWriterCommandResultV1:
    """Displayable metadata, NOT a forge-proof native execution receipt."""

    path: Path
    protocol: WriterSourceProtocol
    report_sha256: str
    source_sha256: str
    effects: int
    pairs: int
    stored_teacher_executions: int
    scope: Literal["stored-writer-data-not-fresh-native-execution"] = (
        "stored-writer-data-not-fresh-native-execution"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"


class _Options(TypedDict):
    expected_protocol: WriterSourceProtocol
    expected_report_sha256: str
    expected_source_sha256: str
    runtime: ScientificAblationWriterCPURuntimeV1
    maximum_source_bytes: int
    maximum_report_bytes: int


def _options(
    protocol: WriterSourceProtocol,
    report: str,
    source: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int,
) -> _Options:
    # Check independent expectations BEFORE opening either candidate file.
    io._anchors(protocol, report, source, runtime, maximum_source_bytes, maximum_report_bytes)
    return dict(
        expected_protocol=protocol,
        expected_report_sha256=report,
        expected_source_sha256=source,
        runtime=runtime,
        maximum_source_bytes=maximum_source_bytes,
        maximum_report_bytes=maximum_report_bytes,
    )


def _summary(
    path: Path, bundle: io.ScientificAblationWriterBundleV1
) -> ScientificAblationWriterCommandResultV1:
    return ScientificAblationWriterCommandResultV1(
        path=path,
        protocol=bundle.source.payload.protocol,
        report_sha256=bundle.report.sha256,
        source_sha256=bundle.source.sha256,
        effects=len(bundle.report.payload.effects),
        pairs=len(bundle.report.payload.statistics.pairs),
        stored_teacher_executions=bundle.report.payload.actual_teacher_executions,
    )


def publish_scientific_ablation_writer_command_v1(
    *,
    source_path: Path,
    report_path: Path,
    output_root: Path,
    expected_protocol: WriterSourceProtocol,
    expected_report_sha256: str,
    expected_source_sha256: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int = MAX_REPORT_BYTES,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> ScientificAblationWriterCommandResultV1:
    """Publish complete canonical existing files; exact retry never remeasures.

    Validation failures before publication create no output store. A signal or
    publication/cleanup failure may leave a valid commit or owned stage: preserve
    it and use the SAME independent job anchors for explicit cold read/retry.
    Local path/signature checks are not a hostile-filesystem/power-loss guarantee.
    """
    try:
        options = _options(
            expected_protocol,
            expected_report_sha256,
            expected_source_sha256,
            runtime,
            maximum_source_bytes,
            maximum_report_bytes,
        )
        if any(
            type(value) is not int or not lower <= value <= 16 * 1024**3
            for value, lower in ((maximum_storage_bytes, 1), (minimum_free_bytes, 0))
        ):
            raise ValueError("Exact bounded writer storage quota/reserve required")
        source_path, report_path = _path(source_path), _path(report_path)
        for path in (source_path, report_path):
            _chain(path.parent)
        source_signature = _signature(_regular(source_path, directory=False))
        source_content = _read_file(source_path, maximum_source_bytes)
        source = read_scientific_ablation_writer_source_v1(
            source_content,
            expected_sha256=expected_source_sha256,
            maximum_source_bytes=maximum_source_bytes,
        )
        if source.payload.protocol != expected_protocol:
            raise ValueError("Original source differs from independently selected protocol")
        pf = scientific_ablation_writer_source_preflight_v1(
            source, expected_sha256=expected_source_sha256
        )
        report_signature = _signature(_regular(report_path, directory=False))
        cap = min(maximum_report_bytes, pf.maximum_report_bytes, pf.report_byte_preflight_bound)
        report_content = _read_file(report_path, cap)
        report = read_scientific_ablation_writer_report_v1(
            report_content, preflight=pf, runtime=runtime
        )
        if report.sha256 != expected_report_sha256:
            raise ValueError("Report differs from independently trusted original job")
        for path, signature, content, maximum in (
            (source_path, source_signature, source_content, maximum_source_bytes),
            (report_path, report_signature, report_content, cap),
        ):
            _chain(path.parent)
            if (
                _signature(_regular(path, directory=False)) != signature
                or _read_file(path, maximum) != content
                or _signature(_regular(path, directory=False)) != signature
            ):
                raise ValueError("Either input changed before publication")
        path = io.write_scientific_ablation_writer_bundle_v1(
            report,
            source=source,
            root=output_root,
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
            **options,
        )
        # All source/caller measurement cleanup precedes this entire cold operation.
        # Store-lock cleanup must also succeed before any command result escapes.
        return _summary(path, io.read_scientific_ablation_writer_bundle_v1(path, **options))
    except Exception:
        raise IrisScientificAblationWriterCommandError(
            "Scientific writer publication did not complete"
        ) from None


def inspect_scientific_ablation_writer_command_v1(
    path: Path,
    *,
    expected_protocol: WriterSourceProtocol,
    expected_report_sha256: str,
    expected_source_sha256: str,
    runtime: ScientificAblationWriterCPURuntimeV1,
    maximum_source_bytes: int,
    maximum_report_bytes: int = MAX_REPORT_BYTES,
) -> ScientificAblationWriterCommandResultV1:
    """Read/validate the ENTIRE stored bundle; never infer expectations from it."""
    try:
        options = _options(
            expected_protocol,
            expected_report_sha256,
            expected_source_sha256,
            runtime,
            maximum_source_bytes,
            maximum_report_bytes,
        )
        path = _path(path)
        return _summary(path, io.read_scientific_ablation_writer_bundle_v1(path, **options))
    except Exception:
        raise IrisScientificAblationWriterCommandError(
            "Scientific writer inspection did not complete"
        ) from None


__all__ = [
    "IrisScientificAblationWriterCommandError",
    "ScientificAblationWriterCommandResultV1",
    "inspect_scientific_ablation_writer_command_v1",
    "publish_scientific_ablation_writer_command_v1",
]
