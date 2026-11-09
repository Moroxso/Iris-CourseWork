"""Retryable current/best nine-file projection of committed lifecycle checkpoints.

The journal remains authoritative. No training, model restore or cleanup of old
artifacts occurs here. The delivery store and receipts must be outside the journal.
"""

import os
import re
import shutil
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointBundleV1,
    _bounded_read,
    _regular,
)
from iris.application.scientific_delivery import (
    _checked,
    build_scientific_checkpoint_delivery_v1,
    read_scientific_delivery_v1,
    write_scientific_delivery_v1,
)
from iris.application.scientific_execution import (
    IrisScientificExecutionSnapshotV1,
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_validation import IrisScientificValidationV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_delivery_v1 import ScientificDeliveryMetricsV1
from iris.artifacts.scientific_execution_v1 import (
    IrisScientificExecutionEventV1,
    ScientificExecutionCheckpointV1,
    ScientificExecutionValidationV1,
)
from iris.artifacts.scientific_publication_v1 import (
    IrisScientificPublicationV1,
    ScientificPublicationPayloadV1,
    ScientificPublishedSelectionV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-publication/v1"
_RECEIPT_LIMIT = 16384


class IrisScientificPublicationError(RuntimeError):
    """Projection failed; committed training and already published files are retained."""


@dataclass(frozen=True, slots=True)
class IrisScientificPublicationReportV1:
    execution_sha256: str
    source_event_count: int
    checkpoint_count: int
    receipts: tuple[Path, ...]

    @property
    def complete(self) -> bool:
        return len(self.receipts) == self.checkpoint_count


@dataclass(frozen=True, slots=True)
class _Plan:
    event: IrisScientificExecutionEventV1
    current: IrisScientificCheckpointBundleV1
    best: IrisScientificCheckpointBundleV1 | None
    current_validation: IrisScientificExecutionEventV1 | None
    best_validation: IrisScientificExecutionEventV1 | None


def _plans(snapshot: IrisScientificExecutionSnapshotV1) -> Iterator[_Plan]:
    validation = best = None
    for event in snapshot.events:
        best = _selected_best(best, event)
        if type(event.payload.record) is ScientificExecutionValidationV1:
            validation = event
        if type(event.payload.record) is ScientificExecutionCheckpointV1:
            current, selected = _bundle_parts(
                _checkpoint_for_event(snapshot.path, snapshot.header, event, best_event=best)
            )
            same_step = (
                validation is not None
                and validation.payload.after.step == current.manifest.payload.completed_steps
            )
            yield _Plan(event, current, selected, validation if same_step else None, best)


def _metric(event: IrisScientificExecutionEventV1 | None) -> IrisScientificValidationV1 | None:
    if event is None:
        return None
    record = event.payload.record
    if type(record) is not ScientificExecutionValidationV1:
        raise ValueError("Validation event required")
    return IrisScientificValidationV1(**record.metric.model_dump(mode="python"))


def _destination(root: Path, snapshot: IrisScientificExecutionSnapshotV1) -> tuple[Path, Path]:
    if not isinstance(root, Path) or not root.name or str(root).startswith(("//", "\\\\")):
        raise ValueError("Explicit local delivery root required")
    root = root.absolute()
    if root.is_relative_to(snapshot.path):
        raise ValueError("Deliveries cannot change the authoritative execution directory")
    receipts = root / "scientific-publications-v1" / snapshot.header.sha256
    for parent in (receipts, *receipts.parents):
        with suppress(FileNotFoundError):
            _regular(parent, directory=True)
    return root, receipts


def _receipt_bytes(receipt: IrisScientificPublicationV1) -> bytes:
    receipt = IrisScientificPublicationV1.model_validate(receipt.model_dump(mode="python"))
    if receipt.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(receipt.payload.model_dump(mode="json"))
    ):
        raise ValueError("Publication receipt identity differs")
    content = canonical_json_file_bytes(receipt.model_dump(mode="json"))
    if len(content) > _RECEIPT_LIMIT:
        raise ValueError("Publication receipt exceeds its byte budget")
    return content


def _inventory(directory: Path) -> tuple[dict[int, Path], int]:
    if not directory.exists():
        return {}, 0
    _regular(directory, directory=True)
    files: dict[int, Path] = {}
    size = orphan_count = 0
    for entry in directory.iterdir():
        if re.fullmatch(r"[0-9]{8}\.json", entry.name):
            _regular(entry, directory=False)
            length = entry.stat().st_size
            if not 0 < length <= _RECEIPT_LIMIT:
                raise ValueError("Invalid receipt file size")
            files[int(entry.stem)] = entry
            size += length
        elif re.fullmatch(r"\.iris-publication-[0-9a-f]{32}\.tmp", entry.name):
            _regular(entry, directory=True)
            orphan_count += 1
            for child in entry.iterdir():
                _regular(child, directory=False)
                if child.name != "publication.json" or child.stat().st_size > _RECEIPT_LIMIT:
                    raise ValueError("Invalid publication stage")
                size += child.stat().st_size
        else:
            raise ValueError("Unexpected publication inventory entry")
        if len(files) > 200000 or orphan_count > 1024:
            raise ValueError("Publication inventory exceeds its count budget")
    return files, size


def _verify_selection(
    selection: ScientificPublishedSelectionV1,
    *,
    expected: IrisScientificCheckpointBundleV1,
    validation: IrisScientificExecutionEventV1 | None,
    root: Path,
) -> None:
    if (
        selection.checkpoint_sha256 != expected.manifest.sha256
        or selection.validation_event_sha256 != (None if validation is None else validation.sha256)
    ):
        raise ValueError("Delivery selection differs from the committed source")
    path = root / "scientific-delivery-v1" / selection.delivery_sha256
    for parent in (path, *path.parents):
        _regular(parent, directory=True)
    delivery = read_scientific_delivery_v1(path)
    actual, _ = _checked(delivery)
    metric = ScientificDeliveryMetricsV1.model_validate_json(dict(delivery.files)["metrics.json"])
    expected_metric = _metric(validation)
    actual_values = (
        metric.validation.model_dump(mode="python") if metric.validation is not None else None
    )
    expected_values = None if expected_metric is None else asdict(expected_metric)
    if actual != expected or actual_values != expected_values:
        raise ValueError("Delivery weights/control or exact same-step validation differs")


def _verify_receipt(path: Path, *, plan: _Plan, root: Path, execution: str) -> None:
    content = _bounded_read(path, _RECEIPT_LIMIT)
    receipt = IrisScientificPublicationV1.model_validate_json(content)
    payload, event = receipt.payload, plan.event
    record = event.payload.record
    if (
        _receipt_bytes(receipt) != content
        or type(record) is not ScientificExecutionCheckpointV1
        or path.name != f"{event.payload.sequence:08d}.json"
        or payload.execution_sha256 != execution
        or payload.event_sequence != event.payload.sequence
        or payload.event_sha256 != event.sha256
        or payload.lifecycle_checkpoint_sha256 != record.lifecycle_checkpoint_sha256
        or (payload.best is None) != (plan.best is None)
    ):
        raise ValueError("Publication receipt contradicts its source checkpoint event")
    _verify_selection(
        payload.current, expected=plan.current, validation=plan.current_validation, root=root
    )
    if payload.best is not None and plan.best is not None:
        _verify_selection(
            payload.best, expected=plan.best, validation=plan.best_validation, root=root
        )


def _append_receipt(path: Path, receipt: IrisScientificPublicationV1, *, budget: int) -> None:
    content = _receipt_bytes(receipt)
    if path.exists():
        if _bounded_read(path, _RECEIPT_LIMIT) != content:
            raise ValueError("Existing publication receipt conflicts")
        return
    _, occupied = _inventory(path.parent)
    if occupied + _RECEIPT_LIMIT > budget or shutil.disk_usage(path.parent).free < 1024**3:
        raise ValueError("Publication receipt quota/free-space reserve exceeded")
    stage: Path | None = None
    try:
        candidate = path.parent / f".iris-publication-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        stage = candidate  # Cleanup ownership begins only after successful exclusive mkdir.
        staged = stage / "publication.json"
        _write_exact_file(staged, content)
        if _bounded_read(staged, _RECEIPT_LIMIT) != content:
            raise ValueError("Staged receipt bytes differ")
        try:
            os.link(staged, path)
        except OSError:
            if not path.exists() or _bounded_read(path, _RECEIPT_LIMIT) != content:
                raise
        if _bounded_read(path, _RECEIPT_LIMIT) != content:
            raise ValueError("Committed receipt bytes differ")
    finally:
        if stage is not None and stage.exists():
            _regular(stage, directory=True)
            (stage / "publication.json").unlink(missing_ok=True)
            stage.rmdir()


def inspect_scientific_publications_v1(
    *,
    execution_path: Path,
    output_root: Path,
) -> IrisScientificPublicationReportV1:
    """Audit existing receipts AND delivered bytes against authoritative source events."""
    try:
        snapshot = read_scientific_execution_v1(execution_path)
        root, directory = _destination(output_root, snapshot)
        files, _ = _inventory(directory)
        valid: list[Path] = []
        count = 0
        for plan in _plans(snapshot):
            count += 1
            path = files.pop(plan.event.payload.sequence, None)
            if path is not None:
                _verify_receipt(path, plan=plan, root=root, execution=snapshot.header.sha256)
                valid.append(path)
        if files or read_scientific_execution_v1(snapshot.path) != snapshot:
            raise ValueError("Unexpected receipt sequence or changed source journal")
        return IrisScientificPublicationReportV1(
            snapshot.header.sha256, len(snapshot.events), count, tuple(valid)
        )
    except Exception:
        raise IrisScientificPublicationError("Scientific publication inspection failed") from None


def publish_scientific_execution_deliveries_v1(
    *,
    execution_path: Path,
    context: IrisScientificRunV1,
    output_root: Path,
    maximum_delivery_storage_bytes: int = 1024**3,
    maximum_receipt_storage_bytes: int = 16 * 1024**2,
) -> IrisScientificPublicationReportV1:
    """Catch up all committed checkpoint exports; no forward, restore or training."""
    try:
        if (
            type(context) is not IrisScientificRunV1
            or type(maximum_delivery_storage_bytes) is not int
            or not 1 <= maximum_delivery_storage_bytes <= 16 * 1024**3
            or type(maximum_receipt_storage_bytes) is not int
            or not 1 <= maximum_receipt_storage_bytes <= 1024**3
        ):
            raise ValueError("Exact context and bounded publication budgets required")
        context._validate_boundary()
        snapshot = read_scientific_execution_v1(execution_path)
        if (
            context._busy
            or context.configuration_sha256
            != snapshot.header.payload.initial.payload.configuration_sha256
        ):
            raise ValueError("Scientific publication context differs")
        root, directory = _destination(output_root, snapshot)
        files, occupied = _inventory(directory)
        source_sequences = {
            event.payload.sequence
            for event in snapshot.events
            if type(event.payload.record) is ScientificExecutionCheckpointV1
        }
        if files.keys() - source_sequences:
            raise ValueError("Receipts refer to absent checkpoint events")
        for plan in _plans(snapshot):
            sequence = plan.event.payload.sequence
            path = directory / f"{sequence:08d}.json"
            if sequence in files:
                _verify_receipt(path, plan=plan, root=root, execution=snapshot.header.sha256)
                continue
            if occupied + _RECEIPT_LIMIT > maximum_receipt_storage_bytes:
                raise ValueError("Reserve receipt quota before publishing any delivery")
            directory.mkdir(parents=True, exist_ok=True)
            for parent in (directory, *directory.parents):
                _regular(parent, directory=True)

            def deliver(
                checkpoint: IrisScientificCheckpointBundleV1,
                validation: IrisScientificExecutionEventV1 | None,
            ) -> ScientificPublishedSelectionV1:
                bundle = build_scientific_checkpoint_delivery_v1(
                    checkpoint,
                    context=context,
                    validation=_metric(validation),
                )
                write_scientific_delivery_v1(
                    bundle, root=root, maximum_storage_bytes=maximum_delivery_storage_bytes
                )
                return ScientificPublishedSelectionV1(
                    checkpoint_sha256=checkpoint.manifest.sha256,
                    delivery_sha256=bundle.sha256,
                    validation_event_sha256=None if validation is None else validation.sha256,
                )

            current = deliver(plan.current, plan.current_validation)
            best = None if plan.best is None else deliver(plan.best, plan.best_validation)
            record = plan.event.payload.record
            assert type(record) is ScientificExecutionCheckpointV1
            payload = ScientificPublicationPayloadV1(
                execution_sha256=snapshot.header.sha256,
                event_sequence=sequence,
                event_sha256=plan.event.sha256,
                lifecycle_checkpoint_sha256=record.lifecycle_checkpoint_sha256,
                current=current,
                best=best,
            )
            receipt = IrisScientificPublicationV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            _append_receipt(path, receipt, budget=maximum_receipt_storage_bytes)
            _verify_receipt(path, plan=plan, root=root, execution=snapshot.header.sha256)
            occupied += len(_receipt_bytes(receipt))
        context._validate_boundary()
        if read_scientific_execution_v1(snapshot.path) != snapshot:
            raise ValueError("Source journal changed during publication")
        return inspect_scientific_publications_v1(execution_path=execution_path, output_root=root)
    except Exception:
        raise IrisScientificPublicationError(
            "Scientific delivery projection failed; committed source progress is retained"
        ) from None


__all__ = [
    "IrisScientificPublicationError",
    "IrisScientificPublicationReportV1",
    "inspect_scientific_publications_v1",
    "publish_scientific_execution_deliveries_v1",
]
