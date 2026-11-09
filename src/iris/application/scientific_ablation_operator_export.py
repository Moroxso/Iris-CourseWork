"""Exact acknowledged-boundary export; completed retries construct no native graph.

The inference is a projection of actual current/selected-best persisted tensors,
not resumable training state. Unknown incomplete stages are preserved. Local
private-owner consistency is not hostile-host authentication or power-loss proof.
"""

import json
import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import save as save_tensors

from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    _RunConfiguration,
    _verify,
)
from iris.application.scientific_ablation_command_composition import (
    create_scientific_ablation_command_run_v1,
)
from iris.application.scientific_ablation_execution import (
    _checkpoint_for_event,
    _claim,
    _local,
    _member_names,
    _selected_best,
    _stamp,
)
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    _bounded_metadata,
    _exact_tree,
    build_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_inference_io import (
    read_scientific_ablation_inference_v1,
    write_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    _bundle_parts,
)
from iris.application.scientific_ablation_operator import (
    IrisScientificAblationOperatorSnapshotV1,
    _nonfinite,
    _read,
    _recipe,
)
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular, _unique
from iris.application.scientific_command import _input_bytes
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_execution_v1 import (
    ScientificAblationExecutionCheckpointV1,
    ScientificAblationExecutionValidationV1,
)
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationBestV1,
    IrisScientificAblationExportRequestV1,
    IrisScientificAblationExportResultV1,
    ScientificAblationBestPayloadV1,
    ScientificAblationExportRequestPayloadV1,
    ScientificAblationExportResultPayloadV1,
    Selection,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_BEST = "iris/scientific-ablation-operator-best/v1"
_REQUEST = "iris/scientific-ablation-operator-export-request/v1"
_RESULT = "iris/scientific-ablation-operator-export-result/v1"
_STORE = "scientific-ablation-operator-exports-v1"
_INFERENCE = "scientific-ablation-inference-v1"
_LOCK = ".iris-export.lock"
_LIMIT = 64 * 1024


class IrisScientificAblationOperatorExportError(ApplicationError):
    """Redacted export failure; inspect preserved source/staging and retry exact intent."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationOperatorExportReceiptV1:
    path: Path
    inference_path: Path
    result: IrisScientificAblationExportResultV1


def inspect_scientific_ablation_best_v1(path: Path) -> IrisScientificAblationBestV1:
    """Distinguish latest validation selection from physically persisted best; no graph."""
    try:
        snapshot = _read(path)
        best = None
        persisted_event = None
        persisted = None
        for event in snapshot.execution.events:
            if type(event.payload.record) is ScientificAblationExecutionCheckpointV1:
                persisted_event = event
                persisted = _checkpoint_for_event(
                    snapshot.execution.path, snapshot.execution.header, event, best_event=best
                )
            best = _selected_best(best, event)
        selected = None if best is None else best.payload.after
        physical = None if persisted is None else persisted.manifest.payload
        best_sha = None
        # The lifecycle state carries step/metric, not a checkpoint digest.
        if best is not None:
            record = best.payload.record
            if type(record) is not ScientificAblationExecutionValidationV1:
                raise ValueError("Selected validation event required")
            best_sha = record.selected_checkpoint_sha256
        physical_sha = None if physical is None or physical.best is None else physical.best.sha256
        payload = ScientificAblationBestPayloadV1(
            operator_sha256=snapshot.descriptor.sha256,
            execution_sha256=snapshot.execution.header.sha256,
            head_sha256=(
                snapshot.execution.events[-1].sha256
                if snapshot.execution.events
                else snapshot.execution.header.sha256
            ),
            validated_best_event_sha256=None if best is None else best.sha256,
            validated_best_checkpoint_sha256=best_sha,
            validated_best_step=None if selected is None else selected.best_step,
            persisted_checkpoint_event_sha256=None
            if persisted_event is None
            else persisted_event.sha256,
            persisted_lifecycle_sha256=None if persisted is None else persisted.manifest.sha256,
            persisted_best_checkpoint_sha256=physical_sha,
            persisted_best_step=None if physical is None else physical.state.best_step,
            validated_best_is_persisted=(
                best_sha is not None
                and best_sha == physical_sha
                and selected is not None
                and physical is not None
                and selected.best_step == physical.state.best_step
            ),
        )
        return IrisScientificAblationBestV1(
            sha256=domain_sha256(_BEST, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorExportError("Ablation best inspection failed") from None


def _checkpoint(
    snapshot: IrisScientificAblationOperatorSnapshotV1,
    event_sha256: str,
    lifecycle_sha256: str,
    selection: Selection,
) -> tuple[IrisScientificAblationLifecycleBundleV1, IrisScientificAblationCheckpointBundleV1]:
    best = None
    for event in snapshot.execution.events:
        if event.sha256 == event_sha256:
            record = event.payload.record
            if (
                type(record) is not ScientificAblationExecutionCheckpointV1
                or record.lifecycle_checkpoint_sha256 != lifecycle_sha256
            ):
                raise ValueError("Explicit acknowledged checkpoint/lifecycle binding differs")
            bundle = _checkpoint_for_event(
                snapshot.execution.path, snapshot.execution.header, event, best_event=best
            )
            manifest, current, selected = _bundle_parts(bundle)
            chosen = current if selection == "current" else selected
            if (
                chosen is None
                or chosen.manifest.payload.completed_steps < 1
                or manifest.payload.state.phase not in ("train", "stopped")
                or manifest.payload.state.last_checkpoint_step != manifest.payload.state.step
            ):
                raise ValueError("An actual complete persisted export boundary is required")
            return bundle, chosen
        best = _selected_best(best, event)
    raise ValueError("Requested checkpoint event is not in this actual journal")


def _checked_request(
    request: IrisScientificAblationExportRequestV1,
) -> IrisScientificAblationExportRequestV1:
    _exact_tree(request, IrisScientificAblationExportRequestV1)
    _bounded_metadata(request, maximum=_LIMIT)
    value = IrisScientificAblationExportRequestV1.model_validate(request.model_dump(mode="python"))
    if value.sha256 != domain_sha256(
        _REQUEST, canonical_json_bytes(value.payload.model_dump(mode="json"))
    ):
        raise ValueError("Export request digest differs")
    return value


def _source(
    snapshot: IrisScientificAblationOperatorSnapshotV1,
    request: IrisScientificAblationExportRequestV1,
) -> tuple[IrisScientificAblationLifecycleBundleV1, IrisScientificAblationCheckpointBundleV1]:
    value = request.payload
    if (
        value.operator_sha256 != snapshot.descriptor.sha256
        or value.execution_sha256 != snapshot.execution.header.sha256
        or value.purpose != _recipe(snapshot.descriptor).purpose
    ):
        raise ValueError("Export request belongs to another operator/execution/purpose")
    bundle, chosen = _checkpoint(
        snapshot, value.checkpoint_event_sha256, value.lifecycle_sha256, value.selection
    )
    if chosen.manifest.sha256 != value.source_checkpoint_sha256:
        raise ValueError("Actual selected checkpoint differs from saved export intent")
    return bundle, chosen


def make_scientific_ablation_export_request_v1(
    path: Path,
    *,
    checkpoint_event_sha256: str,
    expected_lifecycle_sha256: str,
    selection: Selection = "selected_best",
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> IrisScientificAblationExportRequestV1:
    """Explicit persisted event and lifecycle, never a moving latest checkpoint."""
    try:
        snapshot = _read(path)
        # Validate user controls before selecting a persisted export boundary.
        payload = ScientificAblationExportRequestPayloadV1(
            operator_sha256=snapshot.descriptor.sha256,
            execution_sha256=snapshot.execution.header.sha256,
            checkpoint_event_sha256=checkpoint_event_sha256,
            lifecycle_sha256=expected_lifecycle_sha256,
            source_checkpoint_sha256="0" * 64,
            selection=selection,
            purpose=_recipe(snapshot.descriptor).purpose,
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        _, chosen = _checkpoint(
            snapshot, checkpoint_event_sha256, expected_lifecycle_sha256, selection
        )
        payload = ScientificAblationExportRequestPayloadV1.model_validate(
            payload.model_dump(mode="python") | {"source_checkpoint_sha256": chosen.manifest.sha256}
        )
        return IrisScientificAblationExportRequestV1(
            sha256=domain_sha256(_REQUEST, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorExportError(
            "Ablation export request is invalid"
        ) from None


def read_scientific_ablation_export_request_v1(path: Path) -> IrisScientificAblationExportRequestV1:
    try:
        value = json.loads(
            _input_bytes(path, _LIMIT).decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_nonfinite,
        )
        return _checked_request(
            IrisScientificAblationExportRequestV1.model_validate_json(
                canonical_json_bytes(value), strict=True
            )
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorExportError(
            "Ablation export request is invalid"
        ) from None


def _projection(
    inference: IrisScientificAblationInferenceBundleV1,
    lifecycle: IrisScientificAblationLifecycleBundleV1,
    chosen: IrisScientificAblationCheckpointBundleV1,
    request: IrisScientificAblationExportRequestV1,
) -> None:
    payload, source = inference.manifest.payload, chosen.manifest.payload
    native, current, best = _bundle_parts(lifecycle)
    config = _RunConfiguration.model_validate(_json(source.configuration_json.encode()))
    _, tensors = _verify(chosen)
    prefix = "model/base.teacher."
    projected = {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    selection = payload.selection
    if (
        payload.source_checkpoint_sha256 != chosen.manifest.sha256
        or payload.source_run_configuration_json != source.configuration_json
        or payload.source_run_configuration_sha256 != source.configuration_sha256
        or payload.source_model_configuration_sha256 != config.model_configuration_sha256
        or payload.completed_steps != source.completed_steps
        or payload.ablation != config.ablation
        or payload.purpose != request.payload.purpose
        or selection.kind != request.payload.selection
        or selection.lifecycle_sha256 != native.sha256
        or selection.lifecycle_configuration_json != native.payload.configuration_json
        or selection.lifecycle_configuration_sha256 != native.payload.configuration_sha256
        or selection.state != native.payload.state
        or selection.current_checkpoint_sha256 != current.manifest.sha256
        or selection.current_completed_steps != current.manifest.payload.completed_steps
        or selection.best_checkpoint_sha256 != (None if best is None else best.manifest.sha256)
        or set(projected) != set(payload.model.state_names)
        or save_tensors(
            dict(sorted(projected.items())),
            metadata={"format": "iris.scientific-ablation-inference.v1"},
        )
        != inference.state_bytes
    ):
        raise ValueError(
            "Inference bytes/provenance do not project the requested actual checkpoint"
        )


def _read_export(
    path: Path,
    snapshot: IrisScientificAblationOperatorSnapshotV1,
    request: IrisScientificAblationExportRequestV1,
    *,
    staged: bool = False,
) -> IrisScientificAblationOperatorExportReceiptV1:
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    expected_name = f".iris-export-{request.sha256}.tmp" if staged else request.sha256
    if path.name != expected_name or _member_names(path, 2) != {"receipt.json", "inference"}:
        raise ValueError("Export intent directory layout differs")
    directory, receipt_pin = _stamp(path, directory=True), _stamp(path / "receipt.json")
    raw = _bounded_read(path / "receipt.json", _LIMIT)
    result = IrisScientificAblationExportResultV1.model_validate_json(raw, strict=True)
    if (
        canonical_json_file_bytes(result.model_dump(mode="json")) != raw
        or result.sha256
        != domain_sha256(_RESULT, canonical_json_bytes(result.payload.model_dump(mode="json")))
        or result.payload.request != request
    ):
        raise ValueError("Export receipt does not bind this exact saved intent")
    lifecycle, chosen = _source(snapshot, request)
    inference_root = path / "inference"
    _stamp(inference_root, directory=True)
    if _member_names(inference_root, 1) != {_INFERENCE}:
        raise ValueError("Inference root layout differs")
    store = inference_root / _INFERENCE
    _stamp(store, directory=True)
    if _member_names(store, 2) != {".iris-owner.lock", result.payload.inference_sha256}:
        raise ValueError("Inference export namespace differs")
    if _stamp(store / ".iris-owner.lock")[2] != 1:
        raise ValueError("Inference ownership file size differs")
    namespace_pins = [(node, _stamp(node, directory=True)) for node in (inference_root, store)]
    lock_pin = _stamp(store / ".iris-owner.lock")
    artifact_path = store / result.payload.inference_sha256
    artifact = read_scientific_ablation_inference_v1(
        artifact_path,
        expected_sha256=result.payload.inference_sha256,
        expected_source_checkpoint_sha256=request.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=request.payload.lifecycle_sha256,
    )
    _projection(artifact, lifecycle, chosen, request)
    if (
        result.payload.selected_completed_steps != chosen.manifest.payload.completed_steps
        or _stamp(path, directory=True) != directory
        or _stamp(path / "receipt.json") != receipt_pin
        or _bounded_read(path / "receipt.json", _LIMIT) != raw
        or any(_stamp(node, directory=True) != pin for node, pin in namespace_pins)
        or _stamp(store / ".iris-owner.lock") != lock_pin
    ):
        raise ValueError("Export receipt/source changed during readback")
    return IrisScientificAblationOperatorExportReceiptV1(path, artifact_path, result)


def inspect_scientific_ablation_operator_export_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    root: Path,
) -> IrisScientificAblationOperatorExportReceiptV1:
    """Read an ALREADY published exact export; never mkdir, lock, build or publish.

    A completed but unpublished stage is not a published artifact. Inspection
    rejects it without adopting it; explicit export retry remains separate.
    """
    try:
        request = _checked_request(request)
        root = _local(root)
        snapshot = _read(path)
        _source(snapshot, request)
        parent = root / _STORE
        if parent.is_relative_to(snapshot.path) or snapshot.path.is_relative_to(parent):
            raise ValueError("Export namespace cannot overlap its source operator")
        for node in (parent, *parent.parents):
            _regular(node, directory=True)
        parent_pin = _stamp(parent, directory=True)
        receipt = _read_export(parent / request.sha256, snapshot, request)
        current = _read(path)
        _source(current, request)
        if (
            current.descriptor != snapshot.descriptor
            or _stamp(parent, directory=True) != parent_pin
        ):
            raise ValueError("Export source or containing namespace changed during inspection")
        return receipt
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorExportError(
            "Ablation export inspection failed; no publication was requested"
        ) from None


def _occupied(parent: Path) -> int:
    names = _member_names(parent, 4096)
    for name in names:
        entry = parent / name
        if entry.name == _LOCK:
            if _stamp(entry)[2] != 1:
                raise ValueError("Export lock size differs")
        elif re.fullmatch(r"(?:[0-9a-f]{64}|\.iris-export-[0-9a-f]{64}\.tmp)", entry.name):
            _stamp(entry, directory=True)
        else:
            raise ValueError("Unknown export namespace entry; preserve it")
    total = count = 0

    pending = [parent]
    while pending:
        folder = pending.pop()
        _stamp(folder, directory=True)
        for entry in folder.iterdir():
            count += 1
            if count > 32768:
                raise ValueError("Bounded export store inventory exceeded")
            if entry.is_dir():
                _stamp(entry, directory=True)
                pending.append(entry)
            else:
                total += _stamp(entry)[2]
            if total > 16 * 1024**3:
                raise ValueError("Bounded export store byte inventory exceeded")
    return total


def export_scientific_ablation_operator_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    base: Path,
    root: Path,
) -> IrisScientificAblationOperatorExportReceiptV1:
    """Export one exact persisted selection; repeated completed intents are DATA-only.

    No training/validation phase, implicit checkpoint creation or candidate job.
    Complete staged publication is retryable. An incomplete stage is preserved and
    rejected, never reaped or treated as permission for another native graph.
    """
    try:
        request = _checked_request(request)
        base, root = _local(base), _local(root)
        snapshot = _read(path)
        lifecycle, chosen = _source(snapshot, request)
        parent = root / _STORE
        if parent.is_relative_to(snapshot.path) or snapshot.path.is_relative_to(parent):
            raise ValueError("Export namespace cannot overlap its source operator")
        for node in (parent, *parent.parents):
            with suppress(FileNotFoundError):
                _regular(node, directory=True)
        target, stage = parent / request.sha256, parent / f".iris-export-{request.sha256}.tmp"
        if os.path.lexists(target):
            return _read_export(target, snapshot, request)
        parent.mkdir(parents=True, exist_ok=True)
        parent_pin = _stamp(parent, directory=True)
        with _claim(parent, name=_LOCK) as claim:

            def check() -> None:
                claim.check()
                if _stamp(parent, directory=True) != parent_pin:
                    raise ValueError("Export parent identity changed")

            check()
            occupied = _occupied(parent)
            if occupied > request.payload.maximum_storage_bytes:
                raise ValueError("Export store already exceeds requested quota")
            if os.path.lexists(target):
                return _read_export(target, _read(path), request)
            if os.path.lexists(stage):
                _read_export(stage, _read(path), request, staged=True)
            else:
                minimum_required = len(canonical_json_bytes(request.model_dump(mode="json"))) + 1
                if occupied + minimum_required > request.payload.maximum_storage_bytes:
                    raise ValueError("Export metadata alone exceeds the remaining quota")
                if (
                    shutil.disk_usage(parent).free
                    < request.payload.minimum_free_bytes + minimum_required
                ):
                    raise ValueError("Export free-space reserve is unavailable")
                current = _read(path)
                lifecycle, chosen = _source(current, request)
                artifact = build_scientific_ablation_inference_v1(
                    lifecycle,
                    expected_lifecycle_sha256=request.payload.lifecycle_sha256,
                    selection=request.payload.selection,
                    make_fresh_run=lambda: create_scientific_ablation_command_run_v1(
                        config=_recipe(current.descriptor),
                        base=base,
                        maximum_trainable_parameter_bytes=current.descriptor.payload.maximum_trainable_parameter_bytes,
                    ),
                )
                _projection(artifact, lifecycle, chosen, request)
                payload = ScientificAblationExportResultPayloadV1(
                    request=request,
                    inference_sha256=artifact.manifest.sha256,
                    selected_completed_steps=chosen.manifest.payload.completed_steps,
                )
                result = IrisScientificAblationExportResultV1(
                    sha256=domain_sha256(
                        _RESULT, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                    payload=payload,
                )
                receipt_raw = canonical_json_file_bytes(result.model_dump(mode="json"))
                required = (
                    len(receipt_raw)
                    + len(artifact.state_bytes)
                    + len(canonical_json_file_bytes(artifact.manifest.model_dump(mode="json")))
                    + 1
                )
                check()
                fresh_lifecycle, fresh_chosen = _source(_read(path), request)
                if (
                    fresh_lifecycle.manifest != lifecycle.manifest
                    or fresh_chosen.manifest != chosen.manifest
                ):
                    raise ValueError("Export source changed during native projection")
                if (
                    _occupied(parent) + required > request.payload.maximum_storage_bytes
                    or shutil.disk_usage(parent).free
                    < required + request.payload.minimum_free_bytes
                ):
                    raise ValueError("Export quota/free-space reserve exceeded")
                check()
                stage.mkdir(mode=0o700)
                stage_pin = _stamp(stage, directory=True)
                _write_exact_file(stage / "receipt.json", receipt_raw)
                check()
                if _stamp(stage, directory=True) != stage_pin:
                    raise ValueError("Export staging identity changed")
                write_scientific_ablation_inference_v1(
                    artifact,
                    root=stage / "inference",
                    expected_sha256=artifact.manifest.sha256,
                    expected_source_checkpoint_sha256=request.payload.source_checkpoint_sha256,
                    expected_lifecycle_sha256=request.payload.lifecycle_sha256,
                    maximum_storage_bytes=request.payload.maximum_storage_bytes,
                    minimum_free_bytes=request.payload.minimum_free_bytes,
                )
                if _stamp(stage, directory=True) != stage_pin:
                    raise ValueError("Export staging identity changed during publication")
                _read_export(stage, _read(path), request, staged=True)
            check()
            rename_directory_with_windows_retry(stage, target)
            check()
        return _read_export(target, _read(path), request)
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationOperatorExportError(
            "Ablation export failed; inspect preserved staging or retry the identical request"
        ) from None


__all__ = [
    "IrisScientificAblationOperatorExportError",
    "IrisScientificAblationOperatorExportReceiptV1",
    "export_scientific_ablation_operator_v1",
    "inspect_scientific_ablation_best_v1",
    "inspect_scientific_ablation_operator_export_v1",
    "make_scientific_ablation_export_request_v1",
    "read_scientific_ablation_export_request_v1",
]
