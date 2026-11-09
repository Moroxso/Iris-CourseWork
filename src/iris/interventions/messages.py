"""Closed diagnostic message transfer after allocation and before the real writer.

Donor content is a bounded detached snapshot from the actual proposal phase.
Recipient bids, keys, confidence, TTL, competition and allocation remain its own.
This protocol has a new identity; the twelve legacy intervention kinds are unchanged.
"""

from dataclasses import fields
from typing import Literal, Self, TypedDict

import torch
from pydantic import Field, model_validator

from iris.cognitive.outputs import ModuleProposal
from iris.interventions.phases import _validate_proposal
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.interventions.state import WorkspaceInterventionError
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import Digest
from iris.workspace.outputs import (
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.state import WorkspaceState
from iris.workspace.trace import _Strict


class ModuleMessageSnapshotV1(_Strict):
    """Private proposal vector, never a label/answer/slot blended with older memory."""

    kind: Literal["iris-module-message-snapshot-v1"] = "iris-module-message-snapshot-v1"
    example_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")
    prompt_sha256: Digest
    teacher_graph_sha256: Digest
    sidecar_state_sha256: Digest
    cycle: int = Field(ge=1)
    module: int = Field(ge=0)
    values: tuple[float, ...] = Field(min_length=1, max_length=4096, repr=False)
    content_sha256: Digest

    @model_validator(mode="after")
    def exact_content(self) -> Self:
        if self.content_sha256 != message_content_sha256_v1(self.values):
            raise ValueError("Message content digest differs from actual values")
        return self


def message_content_sha256_v1(values: tuple[float, ...]) -> str:
    return domain_sha256("iris/module-message-content/v1", canonical_json_bytes({"values": values}))


def message_snapshot_sha256_v1(snapshot: ModuleMessageSnapshotV1) -> str:
    snapshot = ModuleMessageSnapshotV1.model_validate(snapshot)
    return domain_sha256(
        "iris/module-message-snapshot/v1", canonical_json_bytes(snapshot.model_dump(mode="json"))
    )


class WorkspaceMessagePlanV1(_Strict):
    """Empty collects messages; nonempty transplants one donor module/cycle vector."""

    kind: Literal["iris-workspace-message-plan-v1"] = "iris-workspace-message-plan-v1"
    donor: ModuleMessageSnapshotV1 | None = Field(default=None, repr=False)


def message_plan_sha256_v1(plan: WorkspaceMessagePlanV1) -> str:
    if type(plan) is not WorkspaceMessagePlanV1:
        raise TypeError("Exact message transfer plan required")
    plan = WorkspaceMessagePlanV1.model_validate(plan)
    return domain_sha256(
        "iris/workspace-message-plan/v1", canonical_json_bytes(plan.model_dump(mode="json"))
    )


class _MessageContext(TypedDict):
    example_id: str
    prompt_sha256: str
    teacher_graph_sha256: str
    sidecar_state_sha256: str


class WorkspaceMessageSessionV1:
    """Per-evaluation trusted phases; no external callback, module hook or stored tensor."""

    def __init__(
        self,
        plan: WorkspaceMessagePlanV1,
        *,
        example_id: str,
        prompt_sha256: str,
        teacher_graph_sha256: str,
        sidecar_state_sha256: str,
        num_slots: int,
        num_modules: int,
        state_dim: int,
        cycles: int,
        winners_per_cycle: int,
        maximum_message_scalars: int,
    ) -> None:
        if type(plan) is not WorkspaceMessagePlanV1:
            raise TypeError("Exact diagnostic message plan required")
        self.plan = WorkspaceMessagePlanV1.model_validate(plan)
        if (
            type(maximum_message_scalars) is not int
            or not 1 <= maximum_message_scalars <= 2_000_000
            or not 1 <= state_dim <= 4096
            or num_modules * state_dim * cycles > maximum_message_scalars
        ):
            raise ValueError("Proposal message collection exceeds scalar budget")
        self.plan_sha256 = message_plan_sha256_v1(self.plan)
        self._inner = WorkspaceInterventionSessionV1(
            WorkspaceInterventionPlanV1(interventions=()),
            num_slots=num_slots,
            num_modules=num_modules,
            state_dim=state_dim,
            cycles=cycles,
            winners_per_cycle=winners_per_cycle,
        )
        self._context: _MessageContext = dict(
            example_id=example_id,
            prompt_sha256=prompt_sha256,
            teacher_graph_sha256=teacher_graph_sha256,
            sidecar_state_sha256=sidecar_state_sha256,
        )
        # Validate even empty-plan recipient labels/digests before first forward.
        ModuleMessageSnapshotV1(
            **self._context,
            cycle=1,
            module=0,
            values=(0.0,),
            content_sha256=message_content_sha256_v1((0.0,)),
        )
        self._modules, self._width, self._cycles = num_modules, state_dim, cycles
        self._prepared = 0
        self._finished = self._failed = False
        self._messages: list[ModuleMessageSnapshotV1] = []
        donor = self.plan.donor
        if donor is not None and (
            donor.teacher_graph_sha256 != teacher_graph_sha256
            or donor.sidecar_state_sha256 != sidecar_state_sha256
            or donor.module >= num_modules
            or donor.cycle > cycles
            or len(donor.values) != state_dim
        ):
            raise ValueError("Donor message differs from actual model/geometry")

    def before_read(self, state: WorkspaceState, cycle: int) -> WorkspaceState:
        self._ensure_open()
        return self._inner.before_read(state, cycle)

    def after_read(self, read: WorkspaceReaderOutput, cycle: int) -> WorkspaceReaderOutput:
        self._ensure_open()
        return self._inner.after_read(read, cycle)

    def after_proposal(self, proposal: ModuleProposal, cycle: int) -> ModuleProposal:
        try:
            self._ensure_open()
            proposal = self._inner.after_proposal(proposal, cycle)
            values = proposal.content.detach().to(device="cpu", dtype=torch.float64)
            if tuple(values.shape) != (1, self._modules, self._width):
                raise WorkspaceInterventionError("Messages require microbatch-one actual proposals")
            for module in range(self._modules):
                vector = tuple(float(v) for v in values[0, module].tolist())
                self._messages.append(
                    ModuleMessageSnapshotV1(
                        **self._context,
                        cycle=cycle,
                        module=module,
                        values=vector,
                        content_sha256=message_content_sha256_v1(vector),
                    )
                )
            return proposal
        except BaseException:
            self._failed = True
            raise

    def after_arbitration(
        self, arbitration: WorkspaceArbiterOutput, cycle: int
    ) -> WorkspaceArbiterOutput:
        self._ensure_open()
        return self._inner.after_arbitration(arbitration, cycle)

    def after_allocation(
        self, state: WorkspaceState, allocation: SlotAllocatorOutput, cycle: int
    ) -> SlotAllocatorOutput:
        self._ensure_open()
        return self._inner.after_allocation(state, allocation, cycle)

    def prepare_write(self, proposal: ModuleProposal, cycle: int) -> ModuleProposal:
        """Clone [1,M,D] proposals; replace content only after recipient allocation."""
        try:
            if (
                self._failed
                or self._finished
                or type(cycle) is not int
                or cycle != self._prepared + 1
                or self._inner._cycle != cycle
                or self._inner._phase != 5
            ):
                raise WorkspaceInterventionError("Message transfer is outside its allocated phase")
            _validate_proposal(proposal)
            if tuple(proposal.content.shape) != (1, self._modules, self._width):
                raise WorkspaceInterventionError("Message proposal geometry changed")
            result = ModuleProposal(
                **{field.name: getattr(proposal, field.name).clone() for field in fields(proposal)}
            )
            donor = self.plan.donor
            if donor is not None and donor.cycle == cycle:
                replacement = torch.tensor(
                    donor.values, dtype=result.content.dtype, device=result.content.device
                )
                if not bool(torch.isfinite(replacement).all()):
                    raise WorkspaceInterventionError("Donor message cannot fit recipient precision")
                result.content[0, donor.module] = replacement
            _validate_proposal(result)
            self._prepared = cycle
            return result
        except BaseException:
            self._failed = True
            raise

    def after_write(
        self,
        before: WorkspaceState,
        written: WorkspaceState,
        allocation: SlotAllocatorOutput,
        cycle: int,
    ) -> WorkspaceState:
        try:
            self._ensure_open()
            if self._prepared != cycle:
                raise WorkspaceInterventionError("Message preparation must precede the real writer")
            return self._inner.after_write(before, written, allocation, cycle)
        except BaseException:
            self._failed = True
            raise

    def finish(self) -> None:
        try:
            self._ensure_open()
            if (
                self._prepared != self._cycles
                or len(self._messages) != self._cycles * self._modules
            ):
                raise WorkspaceInterventionError(
                    "Message evaluation did not collect all actual proposals"
                )
            self._inner.finish()
            self._finished = True
        except BaseException:
            self._failed = True
            raise

    def _ensure_open(self) -> None:
        if self._failed or self._finished or self._inner._failed:
            raise WorkspaceInterventionError("Message session is invalid or closed")

    @property
    def messages(self) -> tuple[ModuleMessageSnapshotV1, ...]:
        if not self._finished or self._failed:
            raise WorkspaceInterventionError("Message snapshots are unavailable before completion")
        return tuple(self._messages)
