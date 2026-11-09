"""Explicit one-measurement writer job with detached DATA publication retries.

This process-local owner is NOT a durable execution token, cold native resume,
complete generation/scoring workflow or scientific acceptance. New handles may
measure again explicitly; publication never implicitly creates/measures a model.
No callbacks, candidate code or legacy capture/control protocol are accepted here.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypedDict

from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
    ScientificAblationCausalInputV1,
    _complete_limits,
    _input,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_heldout import _entry as _heldout_entry
from iris.application.scientific_ablation_inference import _exact_tree
from iris.application.scientific_ablation_writer_io import (
    read_scientific_ablation_writer_bundle_v1,
    write_scientific_ablation_writer_bundle_v1,
)
from iris.application.scientific_ablation_writer_reports import (
    ScientificAblationMessageWriterCompletedV1,
    ScientificAblationWorkspaceWriterCompletedV1,
    WriterCompleted,
    measure_scientific_ablation_message_writer_completed_v1,
    measure_scientific_ablation_workspace_writer_completed_v1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    CausalDesignPolicy,
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    IrisScientificAblationMessageWriterReportV1,
    IrisScientificAblationWorkspaceWriterReportV1,
    ScientificAblationWriterCPURuntimeV1,
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    WriterSourceProtocol,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
    scientific_ablation_writer_source_preflight_v1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256

type WriterJobPhase = Literal[
    "pending", "measuring", "measured", "reading", "publishing", "published", "failed"
]


class IrisScientificAblationWriterJobError(RuntimeError):
    """Wrong/changed process-local job state; never implies a native repair."""


@dataclass(frozen=True, slots=True, eq=False)
class _Pending:
    job: "IrisScientificAblationWriterJobV1"
    heldout: IrisScientificAblationHeldoutV1
    input: ScientificAblationCausalInputV1
    policy: CausalDesignPolicy
    limits: ScientificAblationCausalCompleteLimitsV1
    protocol: WriterSourceProtocol
    metadata: bytes


@dataclass(frozen=True, slots=True, eq=False)
class _Measured:
    job: "IrisScientificAblationWriterJobV1"
    protocol: WriterSourceProtocol
    source: bytes
    report: bytes
    runtime: tuple[str, str]
    source_sha256: str
    report_sha256: str
    maximum_report_bytes: int
    snapshot_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class _Failed:
    job: "IrisScientificAblationWriterJobV1"


class _Anchors(TypedDict):
    expected_protocol: WriterSourceProtocol
    expected_report_sha256: str
    expected_source_sha256: str
    runtime: ScientificAblationWriterCPURuntimeV1
    maximum_source_bytes: int
    maximum_report_bytes: int


class IrisScientificAblationWriterJobV1:
    """Explicit measure, defensive result and model-free publish; no JSON restore."""

    __slots__ = ("_phase", "_seal")
    _phase: WriterJobPhase
    _seal: _Pending | _Measured | _Failed

    def __init__(self) -> None:
        raise IrisScientificAblationWriterJobError("Use create_scientific_ablation_writer_job_v1")

    @property
    def phase(self) -> WriterJobPhase:
        _shape(self)
        return self._phase

    def measure(self) -> WriterCompleted:
        return _measure(self)

    @property
    def data(self) -> WriterCompleted:
        return _ready_data(self)

    def publish(
        self,
        root: Path,
        *,
        maximum_storage_bytes: int = 512 * 1024**2,
        minimum_free_bytes: int = 1024**3,
    ) -> Path:
        return _publish(self, root, maximum_storage_bytes, minimum_free_bytes)


def _shape(job: IrisScientificAblationWriterJobV1) -> None:
    if type(job) is not IrisScientificAblationWriterJobV1:
        raise IrisScientificAblationWriterJobError("Exact original writer job required")
    seal = getattr(job, "_seal", None)
    phase = getattr(job, "_phase", None)
    if not isinstance(seal, (_Pending, _Measured, _Failed)):
        raise IrisScientificAblationWriterJobError("Original writer job anchor missing")
    if (
        type(seal) not in (_Pending, _Measured, _Failed)
        or seal.job is not job
        or (type(seal) is _Pending and phase not in ("pending", "measuring"))
        or (
            type(seal) is _Measured
            and phase not in ("measured", "reading", "publishing", "published")
        )
        or (type(seal) is _Failed and phase != "failed")
    ):
        raise IrisScientificAblationWriterJobError("Original writer job anchor/phase changed")


def _policy(value: CausalDesignPolicy, protocol: WriterSourceProtocol) -> CausalDesignPolicy:
    if (
        type(protocol) is not str
        or protocol not in ("workspace", "message")
        or (
            protocol == "workspace" and type(value) is not ScientificAblationWorkspaceCausalPolicyV1
        )
        or (protocol == "message" and type(value) is not ScientificAblationMessageCausalPolicyV1)
    ):
        raise ValueError("Independent exact writer protocol/policy pair required")
    _exact_tree(value, type(value))
    if value.__pydantic_private__ is not None:
        raise ValueError("Hidden writer policy metadata forbidden")
    if type(value) is ScientificAblationWorkspaceCausalPolicyV1:
        return ScientificAblationWorkspaceCausalPolicyV1.model_validate(
            value.model_dump(mode="python")
        )
    return ScientificAblationMessageCausalPolicyV1.model_validate(value.model_dump(mode="python"))


def _metadata(
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    limits: ScientificAblationCausalCompleteLimitsV1,
    protocol: WriterSourceProtocol,
) -> bytes:
    owned = _input(input)
    return canonical_json_bytes(
        dict(
            protocol=protocol,
            policy=_policy(policy, protocol).model_dump(mode="json"),
            limits=dict(
                maximum_observation_scalars=limits.maximum_observation_scalars,
                maximum_report_bytes=limits.maximum_report_bytes,
            ),
            input={
                **{
                    name: str(getattr(owned, name))
                    for name in (
                        "evaluation_path",
                        "capture_path",
                        "lifecycle_path",
                        "inference_path",
                        "inputs_root",
                    )
                },
                **{
                    name: getattr(owned, name)
                    for name in (
                        "expected_evaluation_plan_sha256",
                        "expected_evaluation_sha256",
                        "expected_plan_sha256",
                        "expected_capture_sha256",
                        "maximum_capture_bytes",
                        "maximum_evaluation_bytes",
                    )
                },
                "training": owned.training.model_dump(mode="json"),
                "validation": owned.validation.model_dump(mode="json"),
                "test": owned.test.model_dump(mode="json"),
            },
        )
    )


def create_scientific_ablation_writer_job_v1(
    heldout: IrisScientificAblationHeldoutV1,
    *,
    input: ScientificAblationCausalInputV1,
    policy: CausalDesignPolicy,
    limits: ScientificAblationCausalCompleteLimitsV1,
    expected_protocol: WriterSourceProtocol,
) -> IrisScientificAblationWriterJobV1:
    """Audit/snapshot original native inputs; no factory/model/teacher invocation."""
    owned_policy = _policy(policy, expected_protocol)
    owned_input, owned_limits = _input(input), _complete_limits(limits)
    metadata = _metadata(owned_input, owned_policy, owned_limits, expected_protocol)
    _heldout_entry(heldout)
    job = object.__new__(IrisScientificAblationWriterJobV1)
    job._phase = "pending"
    job._seal = _Pending(
        job, heldout, owned_input, owned_policy, owned_limits, expected_protocol, metadata
    )
    return job


def _snapshot(seal: _Measured) -> str:
    return domain_sha256(
        "iris/scientific-ablation-writer-job-private-data/v1",
        canonical_json_bytes(
            dict(
                protocol=seal.protocol,
                source_sha256=seal.source_sha256,
                report_sha256=seal.report_sha256,
                source_file_sha256=sha256(seal.source).hexdigest(),
                report_file_sha256=sha256(seal.report).hexdigest(),
                runtime_binding=seal.runtime,
                maximum_report_bytes=seal.maximum_report_bytes,
            )
        ),
    )


def _data(seal: _Measured) -> WriterCompleted:
    if _snapshot(seal) != seal.snapshot_sha256:
        raise IrisScientificAblationWriterJobError("Private detached writer bytes changed")
    if type(seal.runtime) is not tuple or len(seal.runtime) != 2:
        raise IrisScientificAblationWriterJobError("Original CPU stream binding changed")
    runtime = ScientificAblationWriterCPURuntimeV1(seal.runtime[0], seal.runtime[1])
    source = read_scientific_ablation_writer_source_v1(
        seal.source,
        expected_sha256=seal.source_sha256,
        maximum_source_bytes=len(seal.source),
    )
    if source.payload.protocol != seal.protocol:
        raise IrisScientificAblationWriterJobError("Private writer protocol changed")
    pf = scientific_ablation_writer_source_preflight_v1(source, expected_sha256=seal.source_sha256)
    report = read_scientific_ablation_writer_report_v1(seal.report, preflight=pf, runtime=runtime)
    if report.sha256 != seal.report_sha256 or len(seal.report) > seal.maximum_report_bytes:
        raise IrisScientificAblationWriterJobError(
            "Private report differs from original native result"
        )
    if _snapshot(seal) != seal.snapshot_sha256:
        raise IrisScientificAblationWriterJobError("Private writer bytes changed during data read")
    if (
        type(report) is IrisScientificAblationWorkspaceWriterReportV1
        and seal.protocol == "workspace"
    ):
        return ScientificAblationWorkspaceWriterCompletedV1(report, source, runtime)
    if type(report) is IrisScientificAblationMessageWriterReportV1 and seal.protocol == "message":
        return ScientificAblationMessageWriterCompletedV1(report, source, runtime)
    raise IrisScientificAblationWriterJobError("Exact independent writer report/protocol required")


def _measured(job: IrisScientificAblationWriterJobV1) -> _Measured:
    _shape(job)
    if job._phase not in ("measured", "published") or type(job._seal) is not _Measured:
        raise IrisScientificAblationWriterJobError("Writer job has no ready detached native data")
    return job._seal


def _recover_data(
    job: IrisScientificAblationWriterJobV1,
    seal: _Measured,
    phase: WriterJobPhase,
) -> None:
    # Preserve the original signal even when cache validation itself fails.
    try:
        stable = job._seal is seal and _snapshot(seal) == seal.snapshot_sha256
    except BaseException:
        stable = False
    if stable:
        job._phase = phase
    else:
        job._seal, job._phase = _Failed(job), "failed"


def _ready_data(job: IrisScientificAblationWriterJobV1) -> WriterCompleted:
    seal = _measured(job)
    previous = job._phase
    job._phase = "reading"
    try:
        result = _data(seal)
        if job._seal is not seal or job._phase != "reading":
            raise IrisScientificAblationWriterJobError("Writer job changed during data read")
        job._phase = previous
        return result
    except BaseException:
        _recover_data(job, seal, previous)
        raise


def _capture(
    job: IrisScientificAblationWriterJobV1, pending: _Pending, value: WriterCompleted
) -> _Measured:
    expected = (
        ScientificAblationWorkspaceWriterCompletedV1
        if pending.protocol == "workspace"
        else ScientificAblationMessageWriterCompletedV1
    )
    if type(value) is not expected:
        raise IrisScientificAblationWriterJobError(
            "Exact native completed result/protocol required"
        )
    _heldout_entry(pending.heldout)  # Native caller/owners must already be READY.
    source = scientific_ablation_writer_source_bytes_v1(
        value.source, expected_sha256=value.source.sha256
    )
    pf = scientific_ablation_writer_source_preflight_v1(
        value.source, expected_sha256=value.source.sha256
    )
    report = scientific_ablation_writer_report_bytes_v1(
        value.report, preflight=pf, runtime=value.runtime
    )
    if job._seal is not pending or job._phase != "measuring":
        raise IrisScientificAblationWriterJobError("Writer job changed during native measurement")
    if (
        _metadata(pending.input, pending.policy, pending.limits, pending.protocol)
        != pending.metadata
    ):
        raise IrisScientificAblationWriterJobError(
            "Original writer job inputs changed during measurement"
        )
    provisional = _Measured(
        job,
        pending.protocol,
        source,
        report,
        (value.runtime.torch_version, value.runtime.torch_git_version),
        value.source.sha256,
        value.report.sha256,
        pending.limits.maximum_report_bytes,
        "",
    )
    result = _Measured(
        job,
        provisional.protocol,
        provisional.source,
        provisional.report,
        provisional.runtime,
        provisional.source_sha256,
        provisional.report_sha256,
        provisional.maximum_report_bytes,
        _snapshot(provisional),
    )
    _data(result)  # Revalidate/detach before releasing the original native owners.
    return result


def _measure(job: IrisScientificAblationWriterJobV1) -> WriterCompleted:
    _shape(job)
    if job._phase in ("measured", "published"):
        return _ready_data(job)
    if job._phase != "pending" or type(job._seal) is not _Pending:
        raise IrisScientificAblationWriterJobError(
            "Writer measurement is busy or permanently failed"
        )
    pending = job._seal
    job._phase = "measuring"
    try:
        if (
            _metadata(pending.input, pending.policy, pending.limits, pending.protocol)
            != pending.metadata
        ):
            raise IrisScientificAblationWriterJobError("Original native writer inputs changed")
        if type(pending.policy) is ScientificAblationWorkspaceCausalPolicyV1:
            value: WriterCompleted = measure_scientific_ablation_workspace_writer_completed_v1(
                pending.heldout,
                input=pending.input,
                policy=pending.policy,
                limits=pending.limits,
            )
        elif type(pending.policy) is ScientificAblationMessageCausalPolicyV1:
            value = measure_scientific_ablation_message_writer_completed_v1(
                pending.heldout,
                input=pending.input,
                policy=pending.policy,
                limits=pending.limits,
            )
        else:
            raise IrisScientificAblationWriterJobError("Original native writer policy changed")
        measured = _capture(job, pending, value)
        result = _data(measured)
        job._seal, job._phase = measured, "measured"
        return result
    except BaseException:
        job._seal, job._phase = _Failed(job), "failed"
        raise


def _publish(job: IrisScientificAblationWriterJobV1, root: Path, quota: int, reserve: int) -> Path:
    seal = _measured(job)
    job._phase = "publishing"
    try:
        data = _data(seal)
        anchors: _Anchors = dict(
            expected_protocol=seal.protocol,
            expected_report_sha256=seal.report_sha256,
            expected_source_sha256=seal.source_sha256,
            runtime=data.runtime,
            maximum_source_bytes=len(seal.source),
            maximum_report_bytes=seal.maximum_report_bytes,
        )
        path = write_scientific_ablation_writer_bundle_v1(
            data.report,
            source=data.source,
            root=root,
            **anchors,
            maximum_storage_bytes=quota,
            minimum_free_bytes=reserve,
        )
        bundle = read_scientific_ablation_writer_bundle_v1(path, **anchors)
        pf = scientific_ablation_writer_source_preflight_v1(
            bundle.source, expected_sha256=seal.source_sha256
        )
        if (
            scientific_ablation_writer_source_bytes_v1(
                bundle.source, expected_sha256=seal.source_sha256
            )
            != seal.source
            or scientific_ablation_writer_report_bytes_v1(
                bundle.report, preflight=pf, runtime=data.runtime
            )
            != seal.report
            or job._seal is not seal
            or job._phase != "publishing"
            or _snapshot(seal) != seal.snapshot_sha256
        ):
            raise IrisScientificAblationWriterJobError(
                "Actual publication or original job data changed"
            )
        job._phase = "published"
        return path
    except BaseException:
        _recover_data(job, seal, "measured")
        raise


__all__ = [
    "IrisScientificAblationWriterJobError",
    "IrisScientificAblationWriterJobV1",
    "WriterJobPhase",
    "create_scientific_ablation_writer_job_v1",
]
