"""Live original-claim-bound kernel interval permission, NEVER JSON ownership.

Internal foundation for explicit recovery, not a public recovery dispatcher or an
original saved reply. All four kernels retain their strict v1 default. A context
can only continue the unchanged original interval while BOTH original locks are
live. End-to-end recovered DATA/ACK publication and OS-cold races remain separate.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _identity,
    _read_file,
    _regular,
)
from iris.application.scientific_ablation_kernel_chunk import checked_kernel_chunk_v1
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import IrisScientificAblationWorkflowAttemptV1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock

if TYPE_CHECKING:
    from iris.application.scientific_ablation_workflow import _Seal
    from iris.application.scientific_ablation_workflow_io import _Claim

_PROTOCOLS = {
    ("capture", False): (
        "scientific-ablation-capture-v1",
        "iris/scientific-ablation-capture-plan/v1",
        "iris/scientific-ablation-capture-row-ack/v1",
    ),
    ("capture", True): (
        "scientific-ablation-controlled-capture-v1",
        "iris/scientific-ablation-controlled-capture-plan/v1",
        "iris/scientific-ablation-controlled-capture-row-ack/v1",
    ),
    ("score", False): (
        "scientific-ablation-evaluation-ledger-v2",
        "iris/scientific-ablation-evaluation-plan/v2",
        "iris/scientific-ablation-evaluation-ack/v2",
    ),
    ("score", True): (
        "scientific-ablation-controlled-evaluation-ledger-v1",
        "iris/scientific-ablation-controlled-evaluation-plan/v1",
        "iris/scientific-ablation-controlled-evaluation-ack/v1",
    ),
}


class _KernelIntervalPermission:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Only an actual original recovery context may mint permission")


@dataclass(frozen=True, slots=True, eq=False)
class _Binding:
    seal: "_Seal"
    claim: "_Claim"
    intent_path: Path
    intent_json: bytes
    intent_directory_identity: tuple[int, int]
    intent_file_identity: tuple[int, int]
    command: IrisScientificAblationWorkflowCommandIntentV1
    kernel_store: Path
    kernel_identity: tuple[int, int]
    plan_identity: tuple[int, int]
    plan_file_identity: tuple[int, int]
    plan_pin: tuple[int, str]
    genesis: str

    def check(self) -> None:
        from iris.application.scientific_ablation_operator_workflow import (
            _revalidate_operator_binding,
        )
        from iris.application.scientific_ablation_workflow import _entry, _prefix

        if _entry(self.seal.job, busy=True) is not self.seal:
            raise ValueError("Original recovery workflow seal changed")
        self.seal.store._check_claim(self.claim)
        _revalidate_operator_binding(self.seal.job, self.seal)
        if len(_prefix(self.seal)) != self.command.payload.phase.index:
            raise ValueError("Original independent phase prefix changed during recovery")
        _chain(self.intent_path.parent)
        if (
            _identity(_regular(self.intent_path.parent, directory=True))
            != self.intent_directory_identity
            or _identity(_regular(self.intent_path, directory=False)) != self.intent_file_identity
        ):
            raise ValueError("Original durable intent identity changed")
        if {p.name for p in self.intent_path.parent.iterdir()} != {"intent.json"}:
            raise ValueError("Retained produced/result/ACK DATA requires storage-only recovery")
        if _read_file(self.intent_path, 512 * 1024) != self.intent_json:
            raise ValueError("Exact original durable command intent changed")
        chunk = self.command.payload.expected_chunk
        if chunk is None:
            raise ValueError("Original recovery kernel interval missing")
        target = self.kernel_store / chunk.kernel_plan_sha256
        _chain(target)
        if (
            _identity(_regular(self.kernel_store, directory=True)) != self.kernel_identity
            or _identity(_regular(target, directory=True)) != self.plan_identity
            or _identity(_regular(target / "plan.json", directory=False)) != self.plan_file_identity
        ):
            raise ValueError("Original materialized kernel/plan identity changed")
        raw = _read_file(target / "plan.json", 64 * 1024**2)
        if (len(raw), sha256(raw).hexdigest()) != self.plan_pin:
            raise ValueError("Original materialized kernel plan bytes changed")


_LIVE: WeakKeyDictionary[_KernelIntervalPermission, _Binding] = WeakKeyDictionary()
_STARTS: WeakKeyDictionary[_KernelIntervalPermission, int] = WeakKeyDictionary()


def _interval_start_rows(value: _KernelIntervalPermission) -> int:
    """Actual locked kernel start of THIS call, not a later moving observation."""
    _binding(value).check()
    if value not in _STARTS:
        raise ValueError("Original recovery kernel has not selected its locked interval")
    return _STARTS[value]


def _binding(value: _KernelIntervalPermission) -> _Binding:
    if type(value) is not _KernelIntervalPermission or value not in _LIVE:
        raise ValueError("Actual live recovery permission required, never a DTO/copy/cold object")
    return _LIVE[value]


@contextmanager
def _original_interval_permission(
    seal: "_Seal", command: IrisScientificAblationWorkflowCommandIntentV1, claim: "_Claim"
) -> Iterator[_KernelIntervalPermission]:
    """Mint only from the actual bound busy owner/claim and INTENT-ONLY request.

    No storage, native reconstruction, stale-stage adoption or receipt inference.
    Always revoke in finally; closing/reopening/reusing OS descriptors does not
    resurrect this permission. Callers still need genuine original kernel inputs.
    """
    from iris.application import _scientific_ablation_workflow_steps as steps
    from iris.application.scientific_ablation_operator_workflow import _revalidate_operator_binding
    from iris.application.scientific_ablation_workflow import _entry, _prefix
    from iris.application.scientific_ablation_workflow_io import checked_workflow_model_v1

    command = checked_workflow_model_v1(command, IrisScientificAblationWorkflowCommandIntentV1)
    if _entry(seal.job, busy=True) is not seal or seal.operator_binding is None:
        raise ValueError("Original concrete operator workflow required")
    seal.store._check_claim(claim)
    _revalidate_operator_binding(seal.job, seal)
    request, chunk = command.payload, command.payload.expected_chunk
    if (
        chunk is None
        or request.phase.kind not in ("capture", "score")
        or steps.encoded(request.namespace) != seal.storage_definition
        or request.completed_phase_sha256 != seal.anchors
        or request.phase.index != len(seal.anchors)
        or seal.store.phase(request.phase) != request.phase
        or len(_prefix(seal)) != request.phase.index
    ):
        raise ValueError("Original exact kernel command/namespace/BEFORE phase prefix required")
    intent_path = seal.path / "handoffs" / command.sha256[:2] / command.sha256 / "intent.json"
    _chain(intent_path.parent)
    intent_json = steps.encoded(command)
    if _read_file(intent_path, 512 * 1024) != intent_json:
        raise ValueError("Independent original immutable intent must already be durable")
    attempt_raw = _read_file(
        seal.path / f"phase-{request.phase.index:03d}.attempt.json", 4 * 1024**2
    )
    attempt = IrisScientificAblationWorkflowAttemptV1.model_validate_json(attempt_raw)
    if (
        steps.encoded(attempt) != attempt_raw
        or attempt.payload.plan_sha256 != seal.store.plan.sha256
        or attempt.payload.phase != request.phase
        or attempt.payload.predecessor_sha256 != (seal.anchors[-1] if seal.anchors else None)
        or attempt.payload.recovery != "acknowledged_kernel_prefix_only"
    ):
        raise ValueError("Actual original resumable workflow attempt differs")
    namespace, plan_domain, ack_domain = _PROTOCOLS[
        request.phase.kind, request.phase.stream_index is not None
    ]
    store = steps._kernel(seal.store, request.phase) / namespace
    target = store / chunk.kernel_plan_sha256
    _chain(target)
    raw = _read_file(target / "plan.json", 64 * 1024**2)
    envelope = json.loads(raw)
    if (
        type(envelope) is not dict
        or set(envelope) != {"sha256", "payload"}
        or type(envelope["payload"]) is not dict
        or canonical_json_file_bytes(envelope) != raw
        or envelope["sha256"] != chunk.kernel_plan_sha256
        or domain_sha256(plan_domain, canonical_json_bytes(envelope["payload"]))
        != chunk.kernel_plan_sha256
    ):
        raise ValueError("Exact original materialized kernel plan DATA required")
    binding = _Binding(
        seal,
        claim,
        intent_path,
        intent_json,
        _identity(_regular(intent_path.parent, directory=True)),
        _identity(_regular(intent_path, directory=False)),
        command,
        store,
        _identity(_regular(store, directory=True)),
        _identity(_regular(target, directory=True)),
        _identity(_regular(target / "plan.json", directory=False)),
        (len(raw), sha256(raw).hexdigest()),
        domain_sha256(ack_domain + "/genesis", chunk.kernel_plan_sha256.encode()),
    )
    binding.check()
    permit = object.__new__(_KernelIntervalPermission)
    _LIVE[permit] = binding
    try:
        yield permit
    finally:
        _LIVE.pop(permit, None)
        _STARTS.pop(permit, None)


def _checked_interval_permission(
    value: _KernelIntervalPermission | None,
    *,
    root: Path,
    namespace: str,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None,
    plan_sha256: str,
    rows_per_call: int | None,
    maximum_storage_bytes: int,
    minimum_free_bytes: int,
) -> None:
    if value is None:
        return
    binding = _binding(value)
    checked = checked_kernel_chunk_v1(
        expected_chunk, expected_plan_sha256=plan_sha256, rows_per_call=rows_per_call
    )
    if (
        checked is None
        or checked != binding.command.payload.expected_chunk
        or type(root) is not type(Path())
        or not root.is_absolute()
        or root / namespace != binding.kernel_store
    ):
        raise ValueError("Recovery cannot rebase/change original chunk, budget, plan or kernel")
    binding.check()
    from iris.application import _scientific_ablation_workflow_steps as steps

    if (
        type(maximum_storage_bytes) is not int
        or maximum_storage_bytes < 1
        or type(minimum_free_bytes) is not int
        or maximum_storage_bytes
        > steps.budget(binding.seal.store, binding.kernel_store, binding.claim, 0)
        or minimum_free_bytes < binding.claim.minimum_free_for(0)
    ):
        raise ValueError(
            "Kernel recovery cannot enlarge whole-workflow room or lower its free floor"
        )


def _locked_interval_stop(
    value: _KernelIntervalPermission,
    *,
    root: Path,
    namespace: str,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None,
    plan_sha256: str,
    acknowledgement_head_sha256: str,
    acknowledged_rows: int,
    total_rows: int,
    rows_per_call: int | None,
    acknowledgement_prefix: tuple[str, ...],
    kernel_lock: ProcessFileLock,
    maximum_storage_bytes: int,
    minimum_free_bytes: int,
) -> int:
    """Use only AFTER the original kernel's complete verified _read under its lock."""
    _checked_interval_permission(
        value,
        root=root,
        namespace=namespace,
        expected_chunk=expected_chunk,
        plan_sha256=plan_sha256,
        rows_per_call=rows_per_call,
        maximum_storage_bytes=maximum_storage_bytes,
        minimum_free_bytes=minimum_free_bytes,
    )
    binding = _binding(value)
    chunk = binding.command.payload.expected_chunk
    assert chunk is not None
    if (
        type(kernel_lock) is not ProcessFileLock
        or kernel_lock.path != binding.kernel_store / ".iris-owner.lock"
    ):
        raise ValueError("Actual original kernel mutation lock required")
    kernel_lock.check()
    if (
        type(acknowledged_rows) is not int
        or type(total_rows) is not int
        or type(acknowledgement_prefix) is not tuple
        or len(acknowledgement_prefix) != acknowledged_rows
        or total_rows != chunk.total_rows
        or not chunk.acknowledged_rows <= acknowledged_rows <= chunk.stop_row_index
    ):
        raise ValueError(
            "ACK prefix outside the immutable original interval; explicit DATA recovery required"
        )
    for digest in acknowledgement_prefix:
        _digest(digest)
    head = acknowledgement_prefix[-1] if acknowledgement_prefix else binding.genesis
    start = (
        acknowledgement_prefix[chunk.acknowledged_rows - 1]
        if chunk.acknowledged_rows
        else binding.genesis
    )
    if head != acknowledgement_head_sha256 or start != chunk.acknowledgement_head_sha256:
        raise ValueError("Original starting ACK anchor differs from the verified locked chain")
    if value in _STARTS:
        raise ValueError("One permission may enter only one original kernel interval")
    _STARTS[value] = acknowledged_rows
    # Critical: never current_rows + original_budget, never a new budget2 request.
    return chunk.stop_row_index
