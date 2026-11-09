"""Owned compound phase journal and detached DATA; NEVER native execution authority.

An original coordinator must mark a native attempt before its own call. Existing
native attempts are uncertain, not a request to remeasure. Committed DATA requires
an independent expected digest. No stage adoption, reaping or deletion. Local
cooperating-process checks are not hostile-FS or power-loss guarantees.
"""

import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from pydantic import BaseModel

from iris.application.scientific_ablation_inference import _bounded_metadata, _digest, _exact_tree
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _finish_descriptor,
    _identity,
    _path,
    _read_file,
    _regular,
    _signature,
    _write_bytes,
)
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _json
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    COMMAND_WORKFLOW_NAMESPACE,
    IrisScientificAblationCommandWorkflowDefinitionV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    WORKFLOW_ATTEMPT_DOMAIN,
    WORKFLOW_DATA_DOMAIN,
    WORKFLOW_PLAN_DOMAIN,
    IrisScientificAblationWorkflowAttemptV1,
    IrisScientificAblationWorkflowPhaseDataV1,
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowAttemptPayloadV1,
    ScientificAblationWorkflowFileV1,
    ScientificAblationWorkflowIntentV1,
    ScientificAblationWorkflowPhasePayloadV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock

_STORE = "scientific-ablation-native-workflow-v1"
_LOCK = ".iris-owner.lock"
_MAX_METADATA = 4 * 1024**2
_STAGE = re.compile(r"\.iris-workflow-[0-9a-f]{32}\.tmp")
_HANDOFF_STAGE = re.compile(r"\.iris-handoff-[0-9a-f]{32}\.tmp")
_RECOVERY_STAGE = re.compile(r"\.iris-recovery-[0-9a-f]{32}\.tmp")
_HANDOFF_FILES = {
    "intent.json",
    "received.json",
    "producer-ack.json",
    "returned.json",
    "operator-ack.json",
}
_RECOVERY_FILES = {"intent.json", "result.json", "operator-ack.json"}
_BLOB_NAMES = {
    "capture.json",
    "evaluation.json",
    "ordinary.json",
    "collapse.json",
    "source.json",
    "report.json",
    "runtime.json",
    "summary.json",
}


class IrisScientificAblationWorkflowStorageError(RuntimeError):
    """Workflow storage did not complete; no native repair is implied."""


class IrisScientificAblationWorkflowUncertainError(IrisScientificAblationWorkflowStorageError):
    """An existing native attempt must NEVER automatically measure again."""


def _fail(primary: BaseException, cleanup: list[BaseException]) -> NoReturn:
    if _has_signal(primary):
        raise primary from (
            BaseExceptionGroup("Workflow storage cleanup failed", cleanup) if cleanup else None
        )
    signal = next((error for error in cleanup if _has_signal(error)), None)
    if signal is not None:
        raise signal from primary
    if not cleanup and isinstance(primary, IrisScientificAblationWorkflowStorageError):
        raise primary
    raise IrisScientificAblationWorkflowStorageError(
        "Compound workflow storage did not complete"
    ) from (
        primary
        if not cleanup
        else BaseExceptionGroup("Workflow storage failures", [primary, *cleanup])
    )


def checked_workflow_model_v1[T: BaseModel](value: T, expected: type[T]) -> T:
    """Reject unchecked/subclass/hidden/mutable fields before taking a DATA snapshot."""
    _exact_tree(value, expected)
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, BaseModel):
            if item.__pydantic_private__ is not None:
                raise ValueError("Hidden workflow metadata forbidden")
            pending.extend(vars(item).values())
        elif type(item) is tuple:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.values())
    _bounded_metadata(value, maximum=_MAX_METADATA)
    return expected.model_validate(value.model_dump(mode="python", warnings=False))


def build_scientific_ablation_workflow_plan_v1(
    intent: ScientificAblationWorkflowIntentV1,
) -> IrisScientificAblationWorkflowPlanV1:
    """Closed DATA intent only: no filesystem, factory, model or native authority."""
    owned = checked_workflow_model_v1(intent, ScientificAblationWorkflowIntentV1)
    return IrisScientificAblationWorkflowPlanV1(
        sha256=domain_sha256(
            WORKFLOW_PLAN_DOMAIN, canonical_json_bytes(owned.model_dump(mode="json"))
        ),
        payload=owned,
    )


def _encoded(value: BaseModel) -> bytes:
    return canonical_json_file_bytes(value.model_dump(mode="json"))


def _parse[T: BaseModel](path: Path, model: type[T]) -> T:
    raw = _read_file(path, _MAX_METADATA)
    if not raw.endswith(b"\n"):
        raise ValueError("Workflow metadata requires canonical JSON and one LF")
    _json(raw[:-1])
    value = model.model_validate_json(raw[:-1])
    if _encoded(value) != raw:
        raise ValueError("Canonical complete workflow metadata required")
    return value


def _tree_size(root: Path) -> tuple[int, int, int]:
    """Streaming occupancy under the outer lock, without accepting nested DATA."""
    total = directories = files = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        before = _signature(_regular(directory, directory=True))
        names: set[str] = set()
        for path in directory.iterdir():
            if len(names) >= 16384:
                raise ValueError("Workflow directory member ceiling exceeded")
            names.add(path.name)
            value = path.lstat()
            if path.is_dir():
                _regular(path, directory=True)
                pending.append(path)
                directories += 1
            else:
                _regular(path, directory=False)
                total += value.st_size
                files += 1
            if directories > 700000 or files > 2_100_000 or total > 16 * 1024**3:
                raise ValueError("Workflow occupied resource ceiling exceeded")
        if (
            _signature(_regular(directory, directory=True)) != before
            or {p.name for p in directory.iterdir()} != names
        ):
            raise ValueError("Workflow directory changed during occupied inventory")
    return total, directories, files


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkflowDataV1:
    """Defensively read detached DATA, not an original native completion token."""

    record: IrisScientificAblationWorkflowPhaseDataV1
    files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True, slots=True, eq=False)
class _PreparedPhase:
    """Live store-minted stage capability; serialized DATA cannot restore it."""

    store: "ScientificAblationWorkflowStoreV1"
    data: ScientificAblationWorkflowDataV1
    path: Path
    identity: tuple[int, int]
    files: tuple[tuple[str, tuple[int, int]], ...]


@dataclass(frozen=True, slots=True)
class _Claim:
    store: "ScientificAblationWorkflowStoreV1"
    identity: tuple[int, int]
    lock: ProcessFileLock
    reserved_bytes: int = 0
    namespace_definition: bytes | None = None

    def check(self) -> None:
        if (
            type(self.reserved_bytes) is not int
            or not 0
            <= self.reserved_bytes
            <= self.store.plan.payload.resources.maximum_storage_bytes
        ):
            raise ValueError("Exact bounded additional workflow reservation required")
        self.lock.check()
        _chain(self.store.path)
        if _identity(_regular(self.store.path, directory=True)) != self.identity:
            raise ValueError("Original workflow root identity changed")
        self.store._layout()
        if self.namespace_definition is not None and (
            self.store.namespace_definition_json != self.namespace_definition
            or _read_file(self.store.path / "definition.json", _MAX_METADATA)
            != self.namespace_definition
        ):
            raise ValueError("Live command claim namespace definition changed")

    def reserve(self, additional: int) -> None:
        self.check()
        resources = self.store.plan.payload.resources
        if type(additional) is not int or additional < 0:
            raise ValueError("Exact nonnegative workflow reservation required")
        occupied = _tree_size(self.store.path)[0]
        if (
            occupied + self.reserved_bytes + additional > resources.maximum_storage_bytes
            or shutil.disk_usage(self.store.path).free
            < self.reserved_bytes + additional + resources.minimum_free_bytes
        ):
            raise ValueError("Whole workflow byte quota or free-space reserve exceeded")
        self.check()

    def with_reserved_bytes(self, additional: int) -> "_Claim":
        """Immutable reservation VIEW under this SAME live original OS claim.

        Pass the derived view through all producer/kernel writes to preserve
        room for an independent result copy. No files/lock/limits/ownership are
        created. The original view is unchanged: this is NOT a global disk lease
        or a persisted reservation. It grants no native/result/ACK authority.
        """
        self.reserve(additional)
        return replace(self, reserved_bytes=self.reserved_bytes + additional)

    def minimum_free_for(self, additional: int) -> int:
        """Propagate the held bytes into the ORIGINAL kernel's disk-free floor."""
        self.check()
        if type(additional) is not int or additional < 0:
            raise ValueError("Exact nonnegative kernel free-space addition required")
        result = (
            self.store.plan.payload.resources.minimum_free_bytes + self.reserved_bytes + additional
        )
        if result > 16 * 1024**3:
            raise ValueError("Derived kernel free-space floor exceeds its existing16GiB limit")
        return result


class ScientificAblationWorkflowStoreV1:
    """DATA-only journal. This class never constructs or calls a native owner."""

    def __init__(self, root: Path, plan: IrisScientificAblationWorkflowPlanV1) -> None:
        owned = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
        self._plan_bytes = _encoded(owned)
        self._root = _path(root)
        self._definition_bytes: bytes | None = None
        self._prepared: _PreparedPhase | None = None
        self.path = self._root / _STORE / owned.sha256
        self._identity = (
            _identity(_regular(self.path, directory=True)) if os.path.lexists(self.path) else None
        )

    @classmethod
    def _for_commands(
        cls,
        root: Path,
        plan: IrisScientificAblationWorkflowPlanV1,
        definition: IrisScientificAblationCommandWorkflowDefinitionV1,
    ) -> "ScientificAblationWorkflowStoreV1":
        """Internal opt-in namespace; no legacy layout widening or extra quota.

        Construction reads existing identities only. The original concrete
        operator factory must supply the binding; constructible DATA grants no
        native ownership. Commands/results/ACK members are NOT accepted yet.
        """
        if cls is not ScientificAblationWorkflowStoreV1:
            raise ValueError("Exact original workflow store required")
        owned = checked_workflow_model_v1(
            definition, IrisScientificAblationCommandWorkflowDefinitionV1
        )
        owned_plan = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
        store = object.__new__(cls)
        store._plan_bytes = _encoded(owned_plan)
        store._root = _path(root)
        if owned.payload.workflow_plan_sha256 != store.plan.sha256:
            raise ValueError("Command namespace must bind the exact original workflow plan")
        store._definition_bytes = _encoded(owned)
        store._prepared = None
        store.path = store._root / COMMAND_WORKFLOW_NAMESPACE / owned.sha256
        store._identity = (
            _identity(_regular(store.path, directory=True)) if os.path.lexists(store.path) else None
        )
        return store

    @property
    def namespace_definition_json(self) -> bytes | None:
        """Exact immutable namespace DATA snapshot, not a command or result ACK."""
        raw = self._definition_bytes
        if raw is None:
            return None
        if type(raw) is not bytes:
            raise ValueError("Exact immutable command namespace bytes required")
        definition = IrisScientificAblationCommandWorkflowDefinitionV1.model_validate_json(raw)
        if (
            _encoded(definition) != raw
            or definition.payload.workflow_plan_sha256 != self.plan.sha256
        ):
            raise ValueError("Original command namespace definition differs")
        return raw

    def _namespace_path(self) -> Path:
        raw = self.namespace_definition_json
        if raw is None:
            return self._root / _STORE / self.plan.sha256
        definition = IrisScientificAblationCommandWorkflowDefinitionV1.model_validate_json(raw)
        return self._root / COMMAND_WORKFLOW_NAMESPACE / definition.sha256

    @property
    def plan(self) -> IrisScientificAblationWorkflowPlanV1:
        return IrisScientificAblationWorkflowPlanV1.model_validate_json(self._plan_bytes)

    def phase(self, value: ScientificAblationWorkflowPhaseV1) -> ScientificAblationWorkflowPhaseV1:
        owned = checked_workflow_model_v1(value, ScientificAblationWorkflowPhaseV1)
        phases = scientific_ablation_workflow_phases_v1(self.plan)
        if owned.index >= len(phases) or owned != phases[owned.index]:
            raise ValueError("Phase is not the original workflow plan phase")
        return owned

    def _layout(self) -> None:
        if self.path != self._namespace_path():
            raise ValueError("Original workflow storage namespace changed")
        phases = scientific_ablation_workflow_phases_v1(self.plan)
        allowed = {_LOCK, "plan.json", "kernels", "published"}
        definition = self.namespace_definition_json
        if definition is not None:
            allowed.update(("definition.json", "handoffs", "recoveries"))
        allowed.update(f"phase-{phase.index:03d}" for phase in phases)
        allowed.update(f"phase-{phase.index:03d}.attempt.json" for phase in phases)
        for entry in self.path.iterdir():
            if (
                entry.name not in allowed
                and _STAGE.fullmatch(entry.name) is None
                and not (definition is not None and _HANDOFF_STAGE.fullmatch(entry.name))
                and not (definition is not None and _RECOVERY_STAGE.fullmatch(entry.name))
            ):
                raise ValueError("Foreign workflow root member")
            _regular(
                entry,
                directory=entry.name in {"kernels", "published"}
                or (definition is not None and entry.name in {"handoffs", "recoveries"})
                or entry.name.startswith(".iris-workflow-")
                or re.fullmatch(r"phase-[0-9]{3}", entry.name) is not None,
            )
        if (
            definition is not None
            and os.path.lexists(self.path / "definition.json")
            and _read_file(self.path / "definition.json", _MAX_METADATA) != definition
        ):
            raise ValueError("Stored command namespace definition changed")
        for namespace, members in (("handoffs", _HANDOFF_FILES), ("recoveries", _RECOVERY_FILES)):
            if definition is None or not os.path.lexists(self.path / namespace):
                continue
            for shard in (self.path / namespace).iterdir():
                if re.fullmatch(r"[0-9a-f]{2}", shard.name) is None:
                    raise ValueError("Closed handoff shard required")
                _regular(shard, directory=True)
                for request in shard.iterdir():
                    if re.fullmatch(
                        r"[0-9a-f]{64}", request.name
                    ) is None or not request.name.startswith(shard.name):
                        raise ValueError("Closed handoff request identity required")
                    _regular(request, directory=True)
                    for entry in request.iterdir():
                        if entry.name not in members:
                            raise ValueError("Foreign handoff member")
                        _regular(entry, directory=False)

    def _check_claim(self, claim: _Claim) -> None:
        if type(claim) is not _Claim or claim.store is not self:
            raise ValueError("This exact workflow store's original live claim required")
        claim.check()

    @contextmanager
    def claim(self) -> Iterator[_Claim]:
        """Exclusive cooperating-process advance; keeps the lock inode on disk."""
        primary: BaseException | None = None
        cleanup: list[BaseException] = []
        try:
            for parent in (self.path, *self.path.parents):
                with suppress(FileNotFoundError):
                    _regular(parent, directory=True)
            self.path.mkdir(parents=True, exist_ok=True)
            _chain(self.path)
            identity = _identity(_regular(self.path, directory=True))
            if self._identity is not None and self._identity != identity:
                raise ValueError("Existing workflow root was replaced")
            self._identity = identity
            with process_file_lock(self.path / _LOCK) as lock:
                claim = _Claim(self, identity, lock)
                try:
                    claim.check()
                    if not os.path.lexists(self.path / "plan.json"):
                        if {p.name for p in self.path.iterdir()} != {_LOCK}:
                            raise ValueError("Uninitialized workflow root is not empty/owned")
                        claim.reserve(
                            len(self._plan_bytes) + len(self.namespace_definition_json or b"")
                        )
                        self._write_new(self.path / "plan.json", self._plan_bytes, claim)
                    if _read_file(self.path / "plan.json", _MAX_METADATA) != self._plan_bytes:
                        raise ValueError("Original workflow intent changed")
                    definition = self.namespace_definition_json
                    if definition is not None:
                        if not os.path.lexists(self.path / "definition.json"):
                            if {p.name for p in self.path.iterdir()} != {_LOCK, "plan.json"}:
                                raise ValueError("Uninitialized command namespace is not empty")
                            claim.reserve(len(definition))
                            self._write_new(self.path / "definition.json", definition, claim)
                        if _read_file(self.path / "definition.json", _MAX_METADATA) != definition:
                            raise ValueError("Original command namespace DATA changed")
                        claim = replace(claim, namespace_definition=definition)
                        claim.check()
                    yield claim
                    claim.check()
                except BaseException as error:
                    primary = error
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                cleanup.append(error)
        if primary is not None:
            _fail(primary, cleanup)

    def _write_new(self, path: Path, content: bytes, claim: _Claim) -> tuple[int, int]:
        self._check_claim(claim)
        if path.parent != self.path:
            if path.parent.parent != self.path or _STAGE.fullmatch(path.parent.name) is None:
                raise ValueError("Only this owned workflow stage/root may be written")
            _chain(path.parent)
        descriptor = os.open(
            path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        primary = None
        try:
            opened = _identity(os.fstat(descriptor))
            if opened != _identity(_regular(path, directory=False)):
                raise ValueError("New workflow file identity differs")
            _write_bytes(descriptor, content)
            if _identity(_regular(path, directory=False)) != opened:
                raise ValueError("Workflow file replaced during write")
        except BaseException as error:
            primary = error
        _finish_descriptor(primary, lambda: os.close(descriptor))
        if _read_file(path, len(content)) != content:
            raise ValueError("Exact written workflow bytes differ")
        claim.check()
        if _identity(_regular(path, directory=False)) != opened:
            raise ValueError("Written workflow file changed before releasing its identity")
        return opened

    def begin(
        self,
        phase: ScientificAblationWorkflowPhaseV1,
        predecessor: str | None,
        claim: _Claim,
    ) -> IrisScientificAblationWorkflowAttemptV1:
        self._check_claim(claim)
        phase = self.phase(phase)
        if predecessor is not None:
            _digest(predecessor)
        if (phase.index == 0) != (predecessor is None):
            raise ValueError("Original predecessor phase required")
        recovery = (
            "acknowledged_kernel_prefix_only"
            if phase.kind in ("capture", "score")
            else "data_publication_only"
            if phase.kind == "publish"
            else "never_auto_remeasure"
        )
        payload = ScientificAblationWorkflowAttemptPayloadV1.model_validate(
            dict(
                plan_sha256=self.plan.sha256,
                phase=phase,
                predecessor_sha256=predecessor,
                recovery=recovery,
            )
        )
        attempt = IrisScientificAblationWorkflowAttemptV1(
            sha256=domain_sha256(
                WORKFLOW_ATTEMPT_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
        path = self.path / f"phase-{phase.index:03d}.attempt.json"
        claim.check()
        if os.path.lexists(path):
            if _parse(path, IrisScientificAblationWorkflowAttemptV1) != attempt:
                raise ValueError("Existing workflow attempt conflicts with original intent")
            if recovery == "never_auto_remeasure":
                raise IrisScientificAblationWorkflowUncertainError(
                    "Native phase already attempted; DATA recovery needs independent anchors"
                )
        else:
            raw = _encoded(attempt)
            claim.reserve(len(raw))
            self._write_new(path, raw, claim)
        claim.check()
        return attempt

    def make_data(
        self,
        phase: ScientificAblationWorkflowPhaseV1,
        predecessor: str | None,
        *,
        origin: str,
        files: dict[str, bytes],
    ) -> ScientificAblationWorkflowDataV1:
        phase = self.phase(phase)
        if type(files) is not dict or not 1 <= len(files) <= 3:
            raise ValueError("Exact bounded phase files required")
        pins = []
        for name, content in sorted(files.items()):
            if type(name) is not str or name not in _BLOB_NAMES or type(content) is not bytes:
                raise ValueError("Exact DATA bytes and closed phase filename required")
            maximum = (
                64 * 1024**2
                if phase.kind == "diagnostics"
                else MAX_REPORT_BYTES
                if name in ("source.json", "report.json")
                else _MAX_METADATA
            )
            if not 1 <= len(content) <= maximum:
                raise ValueError("Complete phase DATA byte cap exceeded")
            pins.append(
                ScientificAblationWorkflowFileV1.model_validate(
                    dict(name=name, sha256=sha256(content).hexdigest(), bytes=len(content))
                )
            )
        payload = ScientificAblationWorkflowPhasePayloadV1.model_validate(
            dict(
                plan_sha256=self.plan.sha256,
                phase=phase,
                predecessor_sha256=predecessor,
                origin=origin,
                files=tuple(pins),
            )
        )
        record = IrisScientificAblationWorkflowPhaseDataV1(
            sha256=domain_sha256(
                WORKFLOW_DATA_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
        return ScientificAblationWorkflowDataV1(record, tuple(sorted(files.items())))

    def _read_at(
        self,
        path: Path,
        phase: ScientificAblationWorkflowPhaseV1,
        expected_sha256: str,
    ) -> ScientificAblationWorkflowDataV1:
        _digest(expected_sha256)
        _chain(path)
        before = _signature(_regular(path, directory=True))
        record = _parse(path / "complete.json", IrisScientificAblationWorkflowPhaseDataV1)
        if (
            record.sha256 != expected_sha256
            or record.payload.plan_sha256 != self.plan.sha256
            or record.payload.phase != self.phase(phase)
        ):
            raise ValueError("Original independent phase DATA anchors differ")
        expected = {file.name for file in record.payload.files} | {"complete.json"}
        if {p.name for p in path.iterdir()} != expected:
            raise ValueError("Exact complete phase directory required")
        files: dict[str, bytes] = {}
        for pin in record.payload.files:
            raw = _read_file(path / pin.name, pin.bytes)
            if len(raw) != pin.bytes or sha256(raw).hexdigest() != pin.sha256:
                raise ValueError("Original detached phase DATA file differs")
            files[pin.name] = raw
        result = self.make_data(
            phase, record.payload.predecessor_sha256, origin=record.payload.origin, files=files
        )
        if (
            result.record != record
            or _signature(_regular(path, directory=True)) != before
            or {p.name for p in path.iterdir()} != expected
        ):
            raise ValueError("Complete phase DATA changed during readback")
        return result

    def read(
        self,
        phase: ScientificAblationWorkflowPhaseV1,
        *,
        expected_sha256: str,
    ) -> ScientificAblationWorkflowDataV1:
        """Read DATA using independent anchors; does not create dirs or a native job."""
        phase = self.phase(phase)
        _chain(self.path)
        if (
            self._identity is not None
            and _identity(_regular(self.path, directory=True)) != self._identity
        ):
            raise ValueError("Original workflow root changed before DATA read")
        if _read_file(self.path / "plan.json", _MAX_METADATA) != self._plan_bytes:
            raise ValueError("Original workflow DATA intent differs")
        definition = self.namespace_definition_json
        if definition is not None:
            self._layout()
            if _read_file(self.path / "definition.json", _MAX_METADATA) != definition:
                raise ValueError("Original command workflow DATA namespace differs")
        result = self._read_at(self.path / f"phase-{phase.index:03d}", phase, expected_sha256)
        attempt = _parse(
            self.path / f"phase-{phase.index:03d}.attempt.json",
            IrisScientificAblationWorkflowAttemptV1,
        )
        if (
            attempt.payload.plan_sha256 != self.plan.sha256
            or attempt.payload.phase != phase
            or attempt.payload.predecessor_sha256 != result.record.payload.predecessor_sha256
        ):
            raise ValueError("Complete workflow DATA has no matching original attempt")
        return result

    def _prepare(self, data: ScientificAblationWorkflowDataV1, claim: _Claim) -> _PreparedPhase:
        self._check_claim(claim)
        if type(data) is not ScientificAblationWorkflowDataV1 or type(data.files) is not tuple:
            raise ValueError("Exact detached DATA required")
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not bytes
            for pair in data.files
        ):
            raise ValueError("Exact immutable detached DATA file pairs required")
        record = checked_workflow_model_v1(data.record, IrisScientificAblationWorkflowPhaseDataV1)
        if (
            len(dict(data.files)) != len(data.files)
            or self.make_data(
                record.payload.phase,
                record.payload.predecessor_sha256,
                origin=record.payload.origin,
                files=dict(data.files),
            )
            != data
        ):
            raise ValueError("Original complete detached DATA snapshot differs")
        phase = record.payload.phase
        attempt = _parse(
            self.path / f"phase-{phase.index:03d}.attempt.json",
            IrisScientificAblationWorkflowAttemptV1,
        )
        if (
            attempt.payload.plan_sha256 != self.plan.sha256
            or attempt.payload.phase != phase
            or attempt.payload.predecessor_sha256 != record.payload.predecessor_sha256
        ):
            raise ValueError("Detached DATA does not bind the original phase attempt")
        target = self.path / f"phase-{phase.index:03d}"
        claim.check()
        if self._prepared is not None:
            prepared = self._prepared
            if prepared.data != data:
                raise ValueError("A different original phase stage is still pending")
            self._check_prepared(prepared, claim)
            return prepared
        if not os.path.lexists(target):
            raw = _encoded(record)
            claim.reserve(sum(len(content) for _, content in data.files) + len(raw))
            stage = self.path / f".iris-workflow-{uuid4().hex}.tmp"
            os.mkdir(stage, 0o700)
            identity = _identity(_regular(stage, directory=True))
            identities: dict[str, tuple[int, int]] = {}

            def owned_stage() -> None:
                claim.check()
                if (
                    _identity(_regular(stage, directory=True)) != identity
                    or {p.name for p in stage.iterdir()} != set(identities)
                    or any(
                        _identity(_regular(stage / name, directory=False)) != native
                        for name, native in identities.items()
                    )
                ):
                    raise ValueError("Owned workflow stage/file identity or membership changed")

            for name, content in (*data.files, ("complete.json", raw)):
                owned_stage()
                identities[name] = self._write_new(stage / name, content, claim)
                owned_stage()
            if self._read_at(stage, phase, record.sha256) != data:
                raise ValueError("Owned staged DATA differs")
            claim.reserve(0)
            owned_stage()
            path = stage
        else:
            path = target
            identity = _identity(_regular(path, directory=True))
            identities = {
                name: _identity(_regular(path / name, directory=False))
                for name in (*dict(data.files), "complete.json")
            }
        if self._read_at(path, phase, record.sha256) != data:
            raise ValueError("Existing immutable phase DATA conflicts")
        prepared = _PreparedPhase(self, data, path, identity, tuple(sorted(identities.items())))
        self._prepared = prepared
        self._check_prepared(prepared, claim)
        return prepared

    def _check_prepared(self, prepared: _PreparedPhase, claim: _Claim) -> Path:
        self._check_claim(claim)
        if (
            type(prepared) is not _PreparedPhase
            or prepared.store is not self
            or self._prepared is not prepared
        ):
            raise ValueError("Exact live original prepared-stage capability required")
        target = self.path / f"phase-{prepared.data.record.payload.phase.index:03d}"
        path = prepared.path if os.path.lexists(prepared.path) else target
        if (
            path.parent != self.path
            or (path != target and _STAGE.fullmatch(path.name) is None)
            or _identity(_regular(path, directory=True)) != prepared.identity
            or {p.name for p in path.iterdir()} != {name for name, _ in prepared.files}
            or any(
                _identity(_regular(path / name, directory=False)) != identity
                for name, identity in prepared.files
            )
            or self._read_at(path, prepared.data.record.payload.phase, prepared.data.record.sha256)
            != prepared.data
        ):
            raise ValueError("Original prepared stage identity/bytes changed")
        return path

    def _publish_prepared(
        self, prepared: _PreparedPhase, claim: _Claim
    ) -> ScientificAblationWorkflowDataV1:
        path = self._check_prepared(prepared, claim)
        target = self.path / f"phase-{prepared.data.record.payload.phase.index:03d}"
        if path != target:
            if os.path.lexists(target):
                raise ValueError("Another phase target appeared; never adopt/replace it")
            rename_directory_with_windows_retry(path, target)
        self._check_prepared(prepared, claim)
        result = self._read_at(
            target, prepared.data.record.payload.phase, prepared.data.record.sha256
        )
        claim.check()
        self._prepared = None
        return result

    def commit(
        self, data: ScientificAblationWorkflowDataV1, claim: _Claim
    ) -> ScientificAblationWorkflowDataV1:
        """Original DATA commit; a cold caller never adopts an unknown stage."""
        return self._publish_prepared(self._prepare(data, claim), claim)
