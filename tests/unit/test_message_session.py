"""Owned message phases, content-only cloning and failure closure."""

from dataclasses import fields
from typing import Any

import pytest
import torch

from iris.cognitive.outputs import ModuleProposal
from iris.interventions.messages import (
    ModuleMessageSnapshotV1,
    WorkspaceMessagePlanV1,
    WorkspaceMessageSessionV1,
    message_content_sha256_v1,
    message_snapshot_sha256_v1,
)
from iris.interventions.state import WorkspaceInterventionError
from tests.unit.test_workspace_intervention_phases import components
from tests.unit.test_workspace_intervention_session import Cycle, cycle_fixture
from tests.unit.test_workspace_intervention_state import DTYPES


def data(dtype: torch.dtype = torch.float32) -> Cycle:
    original = cycle_fixture(dtype)

    def one(value: Any) -> Any:
        return type(value)(
            **{field.name: getattr(value, field.name)[:1].clone() for field in fields(value)}
        )

    return Cycle(**{field.name: one(getattr(original, field.name)) for field in fields(original)})


def snapshot(values: tuple[float, ...] = (-0.7, 0.8, 0.9)) -> ModuleMessageSnapshotV1:
    return ModuleMessageSnapshotV1(
        example_id="donor",
        prompt_sha256="a" * 64,
        teacher_graph_sha256="b" * 64,
        sidecar_state_sha256="c" * 64,
        cycle=1,
        module=0,
        values=values,
        content_sha256=message_content_sha256_v1(values),
    )


def session(donor: ModuleMessageSnapshotV1 | None = None) -> WorkspaceMessageSessionV1:
    return WorkspaceMessageSessionV1(
        WorkspaceMessagePlanV1(donor=donor),
        example_id="recipient",
        prompt_sha256="d" * 64,
        teacher_graph_sha256="b" * 64,
        sidecar_state_sha256="c" * 64,
        num_slots=4,
        num_modules=4,
        state_dim=3,
        cycles=1,
        winners_per_cycle=2,
        maximum_message_scalars=12,
    )


def allocated(owner: WorkspaceMessageSessionV1, cycle: Cycle) -> None:
    owner.before_read(cycle.state, 1)
    owner.after_read(cycle.read, 1)
    owner.after_proposal(cycle.proposal, 1)
    owner.after_arbitration(cycle.arbitration, 1)
    owner.after_allocation(cycle.state, cycle.allocation, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_transfer_clones_only_content_after_recipient_allocation(dtype: torch.dtype) -> None:
    cycle, donor = data(dtype), snapshot()
    owner = session(donor)
    allocated(owner, cycle)
    transferred = owner.prepare_write(cycle.proposal, 1)
    assert torch.equal(transferred.content[0, 0], torch.tensor(donor.values, dtype=dtype))
    assert torch.equal(transferred.content[:, 1:], cycle.proposal.content[:, 1:])
    for field in fields(ModuleProposal):
        original, changed = getattr(cycle.proposal, field.name), getattr(transferred, field.name)
        assert original.data_ptr() != changed.data_ptr()
        if field.name != "content":
            assert torch.equal(original, changed)
    _, _, _, writer = components(dtype)
    with torch.no_grad():
        written = writer(transferred, cycle.state, cycle.allocation, torch.ones(1, 3, dtype=dtype))
    owner.after_write(cycle.state, written, cycle.allocation, 1)
    owner.finish()
    assert len(owner.messages) == 4
    assert owner.messages[0].values == tuple(float(v) for v in cycle.proposal.content[0, 0])
    assert owner.messages[0].values != donor.values
    before = cycle.proposal.content.clone()
    transferred.content.zero_()
    assert torch.equal(before, cycle.proposal.content)
    with pytest.raises(WorkspaceInterventionError):
        owner.prepare_write(cycle.proposal, 1)


def test_unfinished_snapshots_are_not_exposed_and_premature_finish_poisoned() -> None:
    owner, cycle = session(), data()
    with pytest.raises(WorkspaceInterventionError):
        _ = owner.messages
    with pytest.raises(WorkspaceInterventionError):
        owner.finish()
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        owner.before_read(cycle.state, 1)


@pytest.mark.parametrize("number", [True, 1.0, 0, 2])
def test_new_writer_phase_rejects_non_exact_or_wrong_cycle_and_closes(number: Any) -> None:
    owner, cycle = session(), data()
    allocated(owner, cycle)
    with pytest.raises(WorkspaceInterventionError):
        owner.prepare_write(cycle.proposal, number)
    with pytest.raises(WorkspaceInterventionError, match="closed"):
        owner.after_write(cycle.state, cycle.written, cycle.allocation, 1)
    with pytest.raises(WorkspaceInterventionError):
        _ = owner.messages


@pytest.mark.parametrize("mode", ["before-allocation", "repeat", "omitted"])
def test_prepare_write_must_occur_once_between_allocation_and_write(mode: str) -> None:
    owner, cycle = session(), data()
    if mode != "before-allocation":
        allocated(owner, cycle)
    if mode == "repeat":
        owner.prepare_write(cycle.proposal, 1)
    with pytest.raises(WorkspaceInterventionError):
        if mode == "omitted":
            owner.after_write(cycle.state, cycle.written, cycle.allocation, 1)
        else:
            owner.prepare_write(cycle.proposal, 1)
    with pytest.raises(WorkspaceInterventionError):
        owner.finish()


def test_overflowing_donor_cast_closes_without_mutating_recipient_proposal() -> None:
    cycle = data(torch.float16)
    owner = session(snapshot((1e40, 2e40, 3e40)))
    allocated(owner, cycle)
    original = cycle.proposal.content.clone()
    with pytest.raises(WorkspaceInterventionError, match="precision"):
        owner.prepare_write(cycle.proposal, 1)
    assert torch.equal(original, cycle.proposal.content)
    with pytest.raises(WorkspaceInterventionError):
        owner.finish()


@pytest.mark.parametrize(
    "changes",
    [
        {"sidecar_state_sha256": "e" * 64},
        {"teacher_graph_sha256": "e" * 64},
        {"module": 4},
        {"cycle": 2},
        {"values": (0.0,), "content_sha256": message_content_sha256_v1((0.0,))},
        {"content_sha256": "e" * 64},
    ],
)
def test_donor_model_geometry_or_content_mismatch_rejects_freshly(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        session(snapshot().model_copy(update=changes))


def test_message_snapshot_hash_binds_origin_not_only_equal_content() -> None:
    donor = snapshot()
    other = donor.model_copy(update={"example_id": "different-origin"})
    assert donor.content_sha256 == other.content_sha256
    assert message_snapshot_sha256_v1(donor) != message_snapshot_sha256_v1(other)
