"""Public closed command dispatch using the ORIGINAL concrete workflow/claim.

Saved JSON is request/result DATA, not an owner, factory or stage capability.
This API does not recover a lost multirow producer or invent a returned result.
Unacknowledged cold/native boundaries retain the original kernel/UNCERTAIN rules.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application._scientific_ablation_workflow_handoff import _Completion, _Receiver
from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_operator_export import _checked_request
from iris.application.scientific_ablation_operator_workflow import (
    _create_operator_workflow,
    _OperatorWorkflowBinding,
    _revalidate_operator_binding,
)
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_ablation_workflow import (
    IrisScientificAblationWorkflowV1,
    ScientificAblationWorkflowAdvanceResultV1,
    ScientificAblationWorkflowProgressV1,
)
from iris.application.scientific_ablation_workflow import (
    _entry as _original_entry,
)
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
    checked_workflow_model_v1,
)
from iris.application.scientific_ablation_workflow_recovery import (
    ScientificAblationWorkflowRecoveryReplyV1,
    _recover_kernel_command_v1,
)
from iris.application.scientific_ablation_workflow_replay import (
    read_scientific_ablation_operator_workflow_command_result_v1,
)
from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    IrisScientificAblationCommandWorkflowDefinitionV1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    IrisScientificAblationWorkflowOperatorAckV1,
    IrisScientificAblationWorkflowReturnedV1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    IrisScientificAblationWorkflowKernelRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes

Operation = Literal["generate", "score", "diagnose", "publish"]
_OPERATIONS = {
    "capture": "generate",
    "score": "score",
    "diagnostics": "diagnose",
    "workspace": "diagnose",
    "message": "diagnose",
    "publish": "publish",
}


class IrisScientificAblationWorkflowCommandError(ApplicationError):
    """Rejected/unfinished command; no permission to remeasure or guess a result."""


@dataclass(frozen=True, slots=True)
class ScientificAblationWorkflowCommandReplyV1:
    """Retainable DATA only. Construction never grants native execution rights.

    original_reference includes storage-only retries, not a claim that THIS call
    executed the producer. saved_original_data is exact old replay, never a new
    execution or a reconstruction of the current moving head.
    """

    request_sha256: str
    operator_ack_json: bytes
    original_result: ScientificAblationWorkflowAdvanceResultV1
    delivery: Literal["original_reference", "saved_original_data"]

    @property
    def operator_ack(self) -> IrisScientificAblationWorkflowOperatorAckV1:
        return IrisScientificAblationWorkflowOperatorAckV1.model_validate_json(
            self.operator_ack_json
        )


@dataclass(frozen=True, slots=True, eq=False)
class _CommandSeal:
    owner: "IrisScientificAblationOperatorCommandWorkflowV1"
    job: IrisScientificAblationWorkflowV1
    binding: _OperatorWorkflowBinding
    store: ScientificAblationWorkflowStoreV1
    definition_json: bytes
    plan_json: bytes


def _entry(
    owner: "IrisScientificAblationOperatorCommandWorkflowV1", *, busy: bool = False
) -> _CommandSeal:
    if type(owner) is not IrisScientificAblationOperatorCommandWorkflowV1:
        raise ValueError("Exact original command workflow required")
    seal = getattr(owner, "_seal", None)
    if (
        type(seal) is not _CommandSeal
        or seal.owner is not owner
        or (getattr(owner, "_state", None) != ("busy" if busy else "idle"))
    ):
        raise ValueError("Original command owner/state changed")
    original = _original_entry(seal.job)
    if (
        original.operator_binding is not seal.binding
        or type(seal.binding) is not _OperatorWorkflowBinding
        or seal.binding.job is not seal.job
        or original.factory is not seal.binding.factory
        or original.store is not seal.store
        or original.storage_definition != seal.definition_json
        or original.plan != seal.plan_json
    ):
        raise ValueError("Original concrete operator/namespace/factory/plan changed")
    return seal


def _wrap(
    job: IrisScientificAblationWorkflowV1,
) -> "IrisScientificAblationOperatorCommandWorkflowV1":
    original = _original_entry(job)
    binding = original.operator_binding
    if type(binding) is not _OperatorWorkflowBinding or original.storage_definition is None:
        raise ValueError("Concrete original command-namespace workflow required")
    owner = object.__new__(IrisScientificAblationOperatorCommandWorkflowV1)
    owner._state = "idle"
    owner._seal = _CommandSeal(
        owner, job, binding, original.store, original.storage_definition, original.plan
    )
    _entry(owner)
    return owner


def _reply(
    seal: _CommandSeal,
    command: IrisScientificAblationWorkflowCommandIntentV1,
    completed: _Completion,
) -> ScientificAblationWorkflowCommandReplyV1:
    """Deliver the live minted original reference, never reopen/learn an FS ACK."""
    if type(completed) is not _Completion or type(completed.receiver) is not _Receiver:
        raise ValueError("Actual original completed receiver required")
    receiver = completed.receiver
    if (
        receiver._completion is not completed
        or completed.token is not receiver._token
        or receiver._job is not seal.job
        or receiver._store is not seal.store
        or receiver._factory is not seal.binding.factory
        or receiver._binding is not seal.binding
        or receiver._namespace != seal.definition_json
        or receiver._returned is not completed.result
        or completed.intent_json != steps.encoded(command)
        or receiver._intent != completed.intent_json
        or receiver._written.get("operator-ack.json") != completed.operator_ack_json
        or receiver._written.get("returned.json") != completed.returned_json
    ):
        raise ValueError("Actual live completion identity/reference/request differs")
    ack = IrisScientificAblationWorkflowOperatorAckV1.model_validate_json(
        completed.operator_ack_json
    )
    returned = IrisScientificAblationWorkflowReturnedV1.model_validate_json(completed.returned_json)
    if (
        steps.encoded(ack) != completed.operator_ack_json
        or steps.encoded(returned) != completed.returned_json
        or ack.payload.request_sha256 != command.sha256
        or returned.payload.request_sha256 != command.sha256
        or ack.payload.returned_result_sha256 != returned.sha256
        or ack.payload.producer_ack_sha256 != returned.payload.producer_ack_sha256
        or canonical_json_file_bytes(returned.payload.progress.model_dump(mode="json"))
        != canonical_json_file_bytes(asdict(completed.result.progress))
    ):
        raise ValueError("Actual original returned result/operator ACK differs")
    return ScientificAblationWorkflowCommandReplyV1(
        command.sha256, completed.operator_ack_json, completed.result, "original_reference"
    )


class IrisScientificAblationOperatorCommandWorkflowV1:
    """Original concrete owner facade; neither a DTO nor a deserialization target."""

    __slots__ = ("_seal", "_state")
    _seal: _CommandSeal
    _state: str

    def __init__(self) -> None:
        raise IrisScientificAblationWorkflowCommandError("Use the concrete public constructor")

    @property
    def definition(self) -> IrisScientificAblationCommandWorkflowDefinitionV1:
        return IrisScientificAblationCommandWorkflowDefinitionV1.model_validate_json(
            _entry(self).definition_json
        )

    @property
    def progress(self) -> ScientificAblationWorkflowProgressV1:
        return _entry(self).job.progress

    def inspect_kernel(
        self,
        *,
        expected_phase: ScientificAblationWorkflowPhaseV1,
        expected_kernel_plan_sha256: str,
        expected_acknowledgement_head_sha256: str | None = None,
    ) -> IrisScientificAblationWorkflowKernelProgressV1:
        """Explicit DATA inspection, not planning/recovery or an automatic next chunk."""
        return _entry(self).job.inspect_kernel(
            expected_phase=expected_phase,
            expected_kernel_plan_sha256=expected_kernel_plan_sha256,
            expected_acknowledgement_head_sha256=expected_acknowledgement_head_sha256,
        )

    def plan_kernel(
        self, request: IrisScientificAblationWorkflowKernelRequestV1
    ) -> ScientificAblationKernelChunkExpectationV1:
        """Explicit saved planning request; result DATA never grants execution rights."""
        from iris.application.scientific_ablation_workflow_planning import (
            plan_scientific_ablation_workflow_kernel_chunk_v1,
        )

        seal: _CommandSeal | None = None
        try:
            request = checked_workflow_model_v1(
                request, IrisScientificAblationWorkflowKernelRequestV1
            )
            seal = _entry(self)
            if steps.encoded(request.payload.namespace) != seal.definition_json:
                raise ValueError("Planning request belongs to another original namespace")
            self._state = "busy"
            result = plan_scientific_ablation_workflow_kernel_chunk_v1(seal.job, request)
            if _entry(self, busy=True) is not seal:
                raise ValueError("Original command facade changed during planning")
            return result
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationWorkflowCommandError(
                "Explicit kernel planning did not deliver; no native result was inferred"
            ) from None
        finally:
            if (
                seal is not None
                and getattr(self, "_seal", None) is seal
                and getattr(self, "_state", None) == "busy"
            ):
                self._state = "idle"

    def dispatch(
        self,
        command: IrisScientificAblationWorkflowCommandIntentV1,
        *,
        operation: Operation,
        expected_operator_ack_sha256: str | None = None,
    ) -> ScientificAblationWorkflowCommandReplyV1:
        """ONE exact explicit command, or explicit independently anchored OLD replay.

        Operation is uniquely determined by the immutable phase kind, not another
        unbound serialized routing field. No phase/budget/head is silently changed.
        Same-owner final-delivery retry uses the original live completion. Once a
        different command completes, older replay requires its independent ACK.
        Missing cold original results are NOT inferred/recomputed by this method.
        """
        seal: _CommandSeal | None = None
        try:
            command = checked_workflow_model_v1(
                command, IrisScientificAblationWorkflowCommandIntentV1
            )
            if type(operation) is not str or _OPERATIONS[command.payload.phase.kind] != operation:
                raise ValueError("Explicit closed operation must match the exact requested phase")
            if expected_operator_ack_sha256 is not None:
                _digest(expected_operator_ack_sha256)
            seal = _entry(self)
            if steps.encoded(command.payload.namespace) != seal.definition_json:
                raise ValueError("Command belongs to another original operator namespace")
            phases = scientific_ablation_workflow_phases_v1(seal.store.plan)
            phase = command.payload.phase
            if phase.index >= len(phases) or phase != phases[phase.index]:
                raise ValueError("Exact source-bound command phase required")
            self._state = "busy"
            _revalidate_operator_binding(seal.job, _original_entry(seal.job))
            if expected_operator_ack_sha256 is not None:
                binding = seal.binding
                result = read_scientific_ablation_operator_workflow_command_result_v1(
                    binding.operator_path,
                    request=IrisScientificAblationExportRequestV1.model_validate_json(
                        binding.request_json
                    ),
                    export_root=binding.export_root,
                    plan=seal.store.plan,
                    base=binding.base,
                    root=seal.store._root,
                    command=command,
                    expected_operator_ack_sha256=expected_operator_ack_sha256,
                )
                # The reader already verified this independent digest. Return
                # the exact linked ACK DATA, not a newly minted native completion.
                from iris.application.scientific_ablation_workflow_replay import _metadata

                ack = _metadata(
                    seal.store.path
                    / "handoffs"
                    / command.sha256[:2]
                    / command.sha256
                    / "operator-ack.json",
                    IrisScientificAblationWorkflowOperatorAckV1,
                )
                if ack.sha256 != expected_operator_ack_sha256:
                    raise ValueError("Independently anchored replay ACK changed before delivery")
                reply = ScientificAblationWorkflowCommandReplyV1(
                    command.sha256, steps.encoded(ack), result, "saved_original_data"
                )
            else:
                completed = _original_entry(seal.job).completed_handoff
                if completed is not None and completed.intent_json == steps.encoded(command):
                    reply = _reply(seal, command, completed)
                else:
                    result = seal.job.advance_expected_with_result(
                        expected_phase=phase,
                        expected_completed_phase_sha256=command.payload.completed_phase_sha256,
                        expected_chunk=command.payload.expected_chunk,
                        rows_per_call=command.payload.rows_per_call,
                    )
                    completed = _original_entry(seal.job).completed_handoff
                    if completed is None or completed.result is not result:
                        raise ValueError("Original producer returned without its live in-claim ACK")
                    reply = _reply(seal, command, completed)
            if _entry(self, busy=True) is not seal:
                raise ValueError("Original command facade changed during dispatch")
            return reply
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationWorkflowCommandError(
                "Explicit ablation command did not deliver; no implicit replay or recovery"
            ) from None
        finally:
            if (
                seal is not None
                and getattr(self, "_seal", None) is seal
                and getattr(self, "_state", None) == "busy"
            ):
                self._state = "idle"

    def recover(
        self,
        command: IrisScientificAblationWorkflowCommandIntentV1,
        *,
        operation: Operation,
        expected_operator_ack_sha256: str | None = None,
    ) -> ScientificAblationWorkflowRecoveryReplyV1:
        """Explicit original-stop recovery; returns DISTINCT recovered DATA.

        This is not ordinary dispatch or an original-return replay. Existing
        recovery DATA is published/read without another native call. Missing
        original nonresumable results are not remeasured by this operation.
        """
        seal: _CommandSeal | None = None
        try:
            command = checked_workflow_model_v1(
                command, IrisScientificAblationWorkflowCommandIntentV1
            )
            if type(operation) is not str or _OPERATIONS[command.payload.phase.kind] != operation:
                raise ValueError("Explicit recovery operation differs from original phase")
            seal = _entry(self)
            self._state = "busy"
            reply = _recover_kernel_command_v1(
                seal.job, command, expected_operator_ack_sha256=expected_operator_ack_sha256
            )
            if _entry(self, busy=True) is not seal:
                raise ValueError("Original command facade changed during recovery")
            return reply
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationWorkflowCommandError(
                "Explicit recovery did not deliver; no original result was invented"
            ) from None
        finally:
            if (
                seal is not None
                and getattr(self, "_seal", None) is seal
                and getattr(self, "_state", None) == "busy"
            ):
                self._state = "idle"


def create_scientific_ablation_operator_command_workflow_v1(
    path: Path,
    *,
    request: IrisScientificAblationExportRequestV1,
    export_root: Path,
    plan: IrisScientificAblationWorkflowPlanV1,
    base: Path,
    root: Path,
    completed_phase_sha256: tuple[str, ...] = (),
) -> IrisScientificAblationOperatorCommandWorkflowV1:
    """Compose only the real recipe factory and original command namespace.

    Reads/binds source DATA; no native execution, output mutation, callback,
    arbitrary namespace or allocation override. Independent phase anchors remain
    mandatory for reopening completed phase DATA; cold results grant no owner.
    """
    try:
        plan = checked_workflow_model_v1(plan, IrisScientificAblationWorkflowPlanV1)
        request = _checked_request(request)
        if not all(
            type(p) is type(Path()) and p.is_absolute() for p in (path, export_root, base, root)
        ):
            raise ValueError("Explicit absolute original command roots required")
        if type(completed_phase_sha256) is not tuple or len(completed_phase_sha256) > 134:
            raise ValueError("Exact independent phase-anchor tuple required")
        for digest in completed_phase_sha256:
            _digest(digest)
        return _wrap(
            _create_operator_workflow(
                path,
                request=request,
                export_root=export_root,
                plan=plan,
                base=base,
                root=root,
                completed_phase_sha256=completed_phase_sha256,
                command_namespace=True,
            )
        )
    except BaseException as error:
        if _has_signal(error):
            raise
        raise IrisScientificAblationWorkflowCommandError(
            "Concrete command workflow source/plan composition failed"
        ) from None


__all__ = [
    "IrisScientificAblationOperatorCommandWorkflowV1",
    "IrisScientificAblationWorkflowCommandError",
    "ScientificAblationWorkflowCommandReplyV1",
    "create_scientific_ablation_operator_command_workflow_v1",
]
